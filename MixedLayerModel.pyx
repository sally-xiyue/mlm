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
cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from libc.math cimport fmin, fabs
from scipy.integrate import odeint

# cdef extern from "mlm_thermodynamic_functions.h":

cdef class MixedLayerModel:
    def __init__(self, namelist):

        #Get initial conditions from namelist or using ISDAC_i default numbers
        #Free tropospheric lapse rate
        try:
            self.gamma_thetal = namelist['initial']['gamma']
        except:
            self.gamma_thetal = 5.0 / 1000.

        #BL potential temperature
        try:
            self.thetal_i = namelist['initial']['SST'] + namelist['initial']['dSST']
        except:
            self.thetal_i = 265.0  # default value

        #Inversion strength at cloud top
        try:
            self.thetal_inv = namelist['initial']['dTi']
            self.thetal_ft = self.thetal_i + self.thetal_inv
        except:
            self.thetal_ft = self.thetal_i + 5.0

        #Relative humidity in the free troposphere (constant)
        try:
            self.rh_ft = namelist['initial']['rh']
        except:
            self.rh_ft = 0.6  # default value

        #Relative humidity near the surface
        try:
            self.rh_i = namelist['initial']['rh0']
        except:
            self.rh_i = 0.8  # default

        #Fractional coefficient of large-scale divergence
        try:
            self.div_frac = namelist['initial']['div_frac']
        except:
            self.div_frac = 1.0

        #Entrainment parameters
        try:
            self.a = namelist['entrainment']['efficiency']
        except:
            self.a = 0.86

        try:
            self.w0 = namelist['entrainment']['w0']
        except:
            self.w0 = 0.0002

        #Vertical grid resolution (matters for radiation and cloud top inversion)
        try:
            self.dz = namelist['grid']['dz']
        except:
            self.dz = 1.0

        #Set up vertical grid
        self.z = np.arange(self.dz, 2501., self.dz)
        self.nz = len(self.z)
        self.z_interface = np.zeros(self.nz+1) #For radiation purpose
        for i in xrange(self.nz):
            self.z_interface[i+1] = self.z[i] + self.dz/2
        self.z_interface[0] = self.dz/2

        #Additional initial condition parameters that are not in the namelist
        self.p_surface = 102000.0  # surface pressure
        self.zi_i = 820.0  # initial BL height
        self.dz_inv = 30.0

        self.t_surface = self.thetal_i  # t_surface = thetal_i * (p_surface/p_tilde)**(Rd/cp)
        self.rho0 = self.p_surface / Rd / self.t_surface
        self.qt_surface = qv_unsat(self.p_surface, saturation_vapor_pressure(self.t_surface) * self.rh_i)

        #Initialize profiles
        self.pressure = np.zeros_like(self.z)
        self.pressure_i = np.zeros_like(self.z_interface)

        self.temperature = np.zeros_like(self.pressure)
        self.qv = np.zeros_like(self.pressure)
        self.ql = np.zeros_like(self.pressure)
        self.qi = np.zeros_like(self.pressure)
        self.thetal = np.zeros_like(self.pressure)
        self.qt = np.zeros_like(self.pressure)
        self.rho = np.zeros_like(self.pressure)

        self.values = np.zeros((3,), dtype=np.double)
        self.tendencies = np.zeros((3,), dtype=np.double)

        self.values[0] = self.zi_i
        self.values[1] = self.thetal_i
        self.values[2] = self.qt_surface


    def initialize(self, NetCDFIO_Stats NS):

        # Get reference density profile
        cdef:
            double zi = self.values[0]
            double thetal = self.values[1]
            double qt = self.values[2]
            Py_ssize_t idx, k
            double [:] tmp = np.zeros((self.nz,), dtype=np.double)
            double [:] alpha = np.zeros((self.nz,), dtype=np.double)

        # Get pressure profile
        def rhs(p, z):
            T, ql, qi = sat_adjst(np.exp(p), self.thetal_i, self.qt_surface)
            return -g/(Rd*T*(1.0 - self.qt_surface + eps_vi*(self.qt_surface-ql)))

        p0 = np.log(self.p_surface)
        self.pressure = odeint(rhs, p0, np.array(self.z), hmax=1.0)[:, 0]
        self.pressure = np.exp(self.pressure)

        self.pressure_i = odeint(rhs, p0, np.array(self.z_interface), hmax=1.0)[:, 0]
        self.pressure_i = np.exp(self.pressure_i)

        # # Determine the cloud top level and above
        # for k in xrange(self.nz):
        #     tmp[k] = self.z[k] - zi
        # idx = (np.abs(tmp)).argmin()
        #
        # # get profiles
        # for k in xrange(self.nz):
        #     if k <= idx:
        #         self.temperature[k], self.ql[k], self.qi[k] = sat_adjst(self.pressure[k], thetal, qt)
        #         self.qt[k] = qt
        #         self.thetal[k] = thetal
        #     else:
        #         self.thetal[k] = (self.thetal_ft + (zi - self.zi_i) * self.gamma_thetal) + \
        #                                      (self.z[k] - zi) * self.gamma_thetal
        #         self.temperature[k] = self.thetal[k] * (self.pressure[k] / p_tilde) ** (Rd / cpd)
        #         self.qt[k] = qv_unsat(self.pressure[k], saturation_vapor_pressure(self.temperature[k]) * self.rh_ft)
        #
        #     self.qv[k] = self.qt[k] - self.ql[k] - self.qi[k]
        #     alpha[k] = get_alpha(self.pressure[k], self.temperature[k], self.qt[k], self.qv[k])

        # get profiles
        for k in xrange(self.nz):
            if self.z[k] <= zi:
                self.temperature[k], self.ql[k], self.qi[k] = sat_adjst(self.pressure[k], thetal, qt)
                self.thetal[k] = thetal
            elif zi < self.z[k] <= (zi + self.dz_inv):
                self.thetal[k] = thetal + (self.z[k] - zi)*self.thetal_inv / self.dz_inv
            else:
                self.thetal[k] = (self.thetal_ft + (zi - self.zi_i) * self.gamma_thetal) + \
                                             (self.z[k] - zi) * self.gamma_thetal
                self.temperature[k] = self.thetal[k] * (self.pressure[k] / p_tilde) ** (Rd / cpd)
                self.qt[k] = qv_unsat(self.pressure[k], saturation_vapor_pressure(self.temperature[k]) * self.rh_ft)

        qt_above_inv = np.amax(self.qt)

        for k in xrange(self.nz):
            if zi < self.z[k] <= (zi + self.dz_inv):
                self.qt[k] = qt - (self.z[k] - zi) * (qt - qt_above_inv) / self.dz_inv
                self.temperature[k], self.ql[k], self.qi[k] = sat_adjst(self.pressure[k], self.thetal[k], self.qt[k])

        for k in xrange(self.nz):
            if self.z[k] <= zi:
                self.qt[k] = qt

            self.qv[k] = self.qt[k] - self.ql[k] - self.qi[k]
            alpha[k] = get_alpha(self.pressure[k], self.temperature[k], self.qt[k], self.qv[k])

        self.rho = 1.0/np.array(alpha)

        NS.add_ts('zi')
        NS.add_ts('thetal_ml')
        NS.add_ts('qt_ml')

        NS.add_profile('thetal')
        NS.add_profile('qt')
        NS.add_profile('ql')
        NS.add_profile('qi')
        NS.add_profile('temperature')
        NS.add_profile('rho')
        NS.add_profile('pressure')

        NS.add_ts('cloud_base')
        NS.add_ts('lwp')
        NS.add_ts('iwp')

        return


    cpdef update(self, TimeStepping.TimeStepping TS, Radiation.Radiation Ra):
        cdef:
            double zi = self.values[0]
            double thetal = self.values[1]
            double qt = self.values[2]
            double dthetal = 0.0
            double dfrad = 0.0
            double dqt = 0.0
            double w_ls
            double w_e
            Py_ssize_t idx, k, idx_top
            double [:] tmp = np.zeros((self.nz,), dtype=np.double)
            double [:] tmp2 = np.zeros((self.nz,), dtype=np.double)
            # double qt_ft
            double temp
            double B_mean
            double alpha

        # Determine the cloud top level and above
        for k in xrange(self.nz):
            tmp[k] = self.z[k] - zi
            tmp2[k] = self.z[k] - zi*1.1
        idx = (np.abs(tmp)).argmin()
        idx_top = (np.abs(tmp2)).argmin()

        # # get profiles
        # for k in xrange(self.nz):
        #     if k <= idx:
        #         self.temperature[k], self.ql[k], self.qi[k] = sat_adjst(self.pressure[k], thetal, qt)
        #         self.qt[k] = qt
        #         self.thetal[k] = thetal
        #     else:
        #         self.thetal[k] = (self.thetal_ft + (zi - self.zi_i) * self.gamma_thetal) + \
        #                                      (self.z[k] - zi) * self.gamma_thetal
        #         self.temperature[k] = self.thetal[k] * (self.pressure[k] / p_tilde) ** (Rd / cpd)
        #         self.qt[k] = qv_unsat(self.pressure[k], saturation_vapor_pressure(self.temperature[k]) * self.rh_ft)
        #         self.ql[k] = 0.0
        #         self.qi[k] = 0.0
        #
        #     self.qv[k] = self.qt[k] - self.ql[k] - self.qi[k]
        #     alpha = get_alpha(self.pressure[k], self.temperature[k], self.qt[k], self.qv[k])
        #     self.rho[k] = 1.0/alpha

        # get profiles
        for k in xrange(self.nz):
            if self.z[k] <= zi:
                self.temperature[k], self.ql[k], self.qi[k] = sat_adjst(self.pressure[k], thetal, qt)
                self.thetal[k] = thetal
            elif zi < self.z[k] <= (zi + self.dz_inv):
                self.thetal[k] = thetal + (self.z[k] - zi)*self.thetal_inv / self.dz_inv
            else:
                self.thetal[k] = (self.thetal_ft + (zi - self.zi_i) * self.gamma_thetal) + \
                                             (self.z[k] - zi) * self.gamma_thetal
                self.temperature[k] = self.thetal[k] * (self.pressure[k] / p_tilde) ** (Rd / cpd)
                self.qt[k] = qv_unsat(self.pressure[k], saturation_vapor_pressure(self.temperature[k]) * self.rh_ft)

        qt_above_inv = np.amax(self.qt)

        for k in xrange(self.nz):
            if zi < self.z[k] <= (zi + self.dz_inv):
                self.qt[k] = qt - (self.z[k] - zi) * (qt - qt_above_inv) / self.dz_inv
                self.temperature[k], self.ql[k], self.qi[k] = sat_adjst(self.pressure[k], self.thetal[k], self.qt[k])

        for k in xrange(self.nz):
            if self.z[k] <= zi:
                self.qt[k] = qt

            self.qv[k] = self.qt[k] - self.ql[k] - self.qi[k]
            alpha = get_alpha(self.pressure[k], self.temperature[k], self.qt[k], self.qv[k])
            self.rho[k] = 1.0/alpha

        while self.ql[idx-1] > self.ql[idx]:
            idx -= 1

        # get radiative flux jump at the cloud top
        # dfrad = (Ra.net_lw_flux[idx_top] - np.min(Ra.net_lw_flux))*0.5
        dfrad = Ra.net_lw_flux[idx_top] - Ra.net_lw_flux[idx]
        dthetal = self.thetal[idx_top] - self.thetal[idx]
        dqt = self.qt[idx_top] - self.qt[idx]

        temp = self.thetal_ft * (self.pressure[idx]/p_tilde) ** (Rd/cpd)
        # qt_ft = qv_unsat(self.pressure[idx], saturation_vapor_pressure(temp) * self.rh_ft)

        B_mean = 6.63e-3 #BL-mean buoyancy flux
        # w_e = entrainment_moeng(B_mean, dthetal, dfrad, self.rho[idx])
        w_e = entrainment_rate(self.a, self.w0, dfrad, dthetal, self.rho[idx]) #2.24

        w_ls = get_ls_subsidence(self.z, self.zi_i, self.div_frac)[idx]
        # print(w_ls, w_e)
        self.tendencies[0] = w_e + w_ls
        # self.tendencies[1] = (w_e * dthetal - dfrad/cpd/self.rho[idx])/zi
        self.tendencies[1] = (w_e * dthetal - (Ra.net_lw_flux[idx_top] - Ra.net_lw_flux[0])/cpd/self.rho[idx])/zi
        self.tendencies[2] = w_e * dqt/zi

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
        NS.write_profile('qi', self.qi)
        NS.write_profile('temperature', self.temperature)
        NS.write_profile('rho', self.rho)
        NS.write_profile('pressure', self.pressure)

        cdef:
            Py_ssize_t kmin = 0
            Py_ssize_t kmax = self.nz
            Py_ssize_t k
            double cb
            double lwp
            double iwp

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
                lwp += self.rho[k] * self.ql[k] * self.dz

        NS.write_ts('lwp', lwp)

        # Compute ice water path
        with nogil:
            for k in xrange(kmin, kmax):
                iwp += self.rho[k] * self.qi[k] * self.dz

        NS.write_ts('iwp', iwp)

        return


cdef double entrainment_rate(double efficiency, double w0, double dfrad, double dthetal, double rho):
    cdef:
        w_e

    w_e = efficiency * dfrad / cpd / rho / dthetal + w0
    return w_e

cdef double entrainment_moeng(double B, double dthetal, double dfrad, double rho):
    cdef double a2 = 2.05
    return a2*B/dthetal + dfrad/rho/cpd/dthetal