#!python
# cython: boundscheck=False
# cython: wraparound=True
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import odeint

from mlm_thermodynamic_functions import *
# cimport mlm_thermodynamic_functions as mfun
from mlm_thermodynamic_functions cimport *
cimport Radiation
# import Radiation
include 'parameters.pxi'
cimport timestepping
from NetCDFIO cimport NetCDFIO_Stats
from libc.math cimport fmin, abs

# cdef extern from "mlm_thermodynamic_functions.h":

cdef class MixedLayerModel:
    def __init__(self, namelist):

        try:
            self.gamma_thetal = namelist['initial']['gamma']  # free tropospheric lapse rate
        except:
            self.gamma_thetal = 5.0 / 1000.

        try:
            self.thetal_i = namelist['initial']['SST'] + namelist['initial']['dSST']
        except:
            self.thetal_i = 265.0  # default value

        try:
            self.thetal_ft = self.thetal_i + namelist['initial']['dTi']
        except:
            self.thetal_ft = self.thetal_i + 5.0

        try:
            self.rh_ft = namelist['initial']['rh']
        except:
            self.rh_ft = 0.6  # default value

        try:
            self.rh_i = namelist['initial']['rh0']  # near surface relative humidity
        except:
            self.rh_i = 0.8  # default

        try:
            self.div_frac = namelist['initial']['div_frac']
        except:
            self.div_frac = 1.0

        try:
            self.efficiency = namelist['entrainment']['efficiency']
        except:
            self.efficiency = 0.7

        try:
            self.w_star = namelist['entrainment']['w_star']
        except:
            self.w_star = 0.7


        self.dz = 5.0  # z grid size
        self.z = np.arange(self.dz, 1501., self.dz)
        self.nz = len(self.z)

        self.p_surface = 102000.0  # surface pressure
        self.zi_i = 820.0  # initial BL height
        self.t_surface = self.thetal_i  # t_surface = thetal_i * (p_surface/p_tilde)**(Rd/cp)
        self.rho0 = self.p_surface / Rd / self.t_surface
        self.qt_surface = qv_unsat(self.p_surface, saturation_vapor_pressure(self.t_surface) * self.rh_i)

        self.pressure = get_pressure(self.z, self.p_surface, self.rho0)

        self.z_interface = np.zeros(self.nz+1)
        for i in xrange(self.nz):
            self.z_interface[i+1] = self.z[i] + 2.5
        self.z_interface[0] = self.dz/2

        self.pressure_i = get_pressure(self.z_interface, self.p_surface, self.rho0)


        self.temperature = np.zeros_like(self.pressure)
        self.qv = np.zeros_like(self.pressure)
        self.ql = np.zeros_like(self.pressure)
        self.qi = np.zeros_like(self.pressure)
        self.thetal = np.zeros_like(self.pressure)
        self.qt = np.zeros_like(self.pressure)

        self.values = np.zeros((3,), dtype=np.double)
        self.tendencies = np.zeros((3,), dtype=np.double)

        self.values[0] = self.zi_i
        self.values[1] = self.thetal_i
        self.values[2] = self.qt_surface

        # self.radiation = Radiation.RadiationRRTM(namelist)
        # self.count = 0
        # self.radiation_frequency = 10.0
        # self.next_radiation_calculate = 0.0

    cpdef initialize(self, NetCDFIO_Stats NS):

        NS.add_ts('zi')
        NS.add_ts('thetal_ml')
        NS.add_ts('qt_ml')

        NS.add_profile('thetal')
        NS.add_profile('qt')
        NS.add_profile('ql')
        NS.add_profile('temperature')

        NS.add_ts('cloud_base')
        NS.add_ts('lwp')

        return


    cpdef update(self, timestepping.TimeStepping TS, Radiation.Radiation Ra):
        cdef:
            double zi = self.values[0]
            double thetal = self.values[1]
            double qt = self.values[2]
            double dthetal = 0.0
            double dfrad = 0.0
            double w_ls
            double w_e
            Py_ssize_t idx, k, idx_top
            double [:] tmp = np.zeros((self.nz,), dtype=np.double)
            double [:] tmp2 = np.zeros((self.nz,), dtype=np.double)
            double qt_ft
            double temp

        # get profiles
        for k in xrange(self.nz):
            if self.z[k] <= zi:
                self.temperature[k], self.ql[k] = sat_adjst(self.pressure[k], thetal, qt)
                self.qt[k] = qt
                self.thetal[k] = thetal
            else:
                self.thetal[k] = (self.thetal_ft + (zi - self.zi_i) * self.gamma_thetal) + \
                                             (self.z[k] - zi) * self.gamma_thetal
                self.temperature[k] = self.thetal[k] * (self.pressure[k] / p_tilde) ** (Rd / cpd)
                self.qt[k] = qv_unsat(self.pressure[k], saturation_vapor_pressure(self.temperature[k]) * self.rh_ft)

            self.qv[k] = self.qt[k] - self.ql[k]


        # get radiative flux jump at the cloud top
        for k in xrange(self.nz):
            tmp[k] = self.z[k] - zi
            tmp2[k] = self.z[k] - zi*1.05
        idx = (abs(tmp)).argmin()
        idx_top = (abs(tmp2)).argmin()

        dfrad = Ra.net_lw_flux[idx+3] - Ra.net_lw_flux[idx]
        dthetal = self.thetal[idx+3] - self.thetal[idx]

        temp = self.thetal_ft * (self.pressure[idx]/p_tilde) ** (Rd/cpd)
        qt_ft = qv_unsat(self.pressure[idx], saturation_vapor_pressure(temp) * self.rh_ft)

        # w_e = entrainment_rate(self.efficiency, dfrad, dthetal, thetal, self.rho0)
        w_e = entrainment_moeng(self.temperature[0], zi, dthetal, self.w_star, dfrad, self.rho0)

        w_ls = get_ls_subsidence(self.z, zi, self.div_frac)[idx]

        self.tendencies[0] = w_e + w_ls
        self.tendencies[1] = (w_e * dthetal - dfrad/cpd/self.rho0)/zi
        self.tendencies[2] = w_e * (qt_ft - qt)/zi

        # self.count += 1
        # print('Timestep ' + str(self.count) + ' of integration')

        return

    cpdef stats_io(self, NetCDFIO_Stats NS):

        NS.write_ts('zi', self.values[0])
        NS.write_ts('thetal_ml', self.values[1])
        NS.write_ts('qt_ml', self.values[2])

        NS.write_profile('thetal', self.thetal)
        NS.write_profile('qt', self.qt)
        NS.write_profile('ql', self.ql)
        NS.write_profile('temperature', self.temperature)

        cdef:
            Py_ssize_t kmin = 0
            Py_ssize_t kmax = self.nz
            Py_ssize_t k
            double cb
            double lwp

        # Compute cloud bottom height
        cb = 99999.9
        with nogil:
            for k in xrange(kmin, kmax):
                if self.ql[k] > 0.0:
                    cb = fmin(cb, self.z[k])

        NS.write_ts('cloud_base', cb)

        # Compute liquid water path
        with nogil:
            for k in xrange(kmin, kmax):
                lwp += self.rho0 * self.ql[k] * self.dz

        NS.write_ts('lwp', lwp)

        return


cdef double entrainment_rate(double efficiency, double dfrad, double dthetal, double thetal, double rho0):
    cdef:
        w_e

    w_e = efficiency * dfrad / cpd / rho0 / dthetal
    return w_e

cdef double entrainment_moeng(double T0, double zi, double dthetal, double w_star, double dfrad, double rho0):
    cdef double A = 0.56
    return A*g*zi*dthetal/T0/w_star+dfrad/rho0/cpd/dthetal