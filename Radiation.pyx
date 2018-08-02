#!python
#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: cdivision=True


import pylab as plt
import numpy as np
cimport numpy as np
import netCDF4 as nc
from scipy.interpolate import pchip_interpolate
from libc.math cimport pow, cbrt, exp, fmin, fmax
# from thermodynamic_functions cimport cpm_c
include 'parameters.pxi'
from profiles import profile_data

cimport MixedLayerModel
cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from mlm_thermodynamic_functions import *
from mlm_thermodynamic_functions cimport *
#
# cdef class Radiation_:
#     def __init__(self, namelist):
#         # if namelist specifies RRTM is to be used, this will override any case-specific radiation schemes
#         try:
#             casename = namelist['radiation']
#         except:
#             casename = 'rrtm'
#
#         if casename == 'Isdac':
#             self.scheme = RadiationIsdac(namelist)
#         else:
#             self.scheme = RadiationRRTM(namelist)
#             print('Using default RRTM radiation scheme!')
#
#         return
#
#     cpdef initialize(self):
#         self.scheme.initialize()
#         return
#
#     cpdef initialize_profiles(self, MixedLayerModel.MixedLayerModel mlm):
#         self.scheme.initialize_profiles(mlm)
#         return
#
#     cpdef update(self, MixedLayerModel.MixedLayerModel mlm, TimeStepping.TimeStepping TS):
#         self.scheme.update(mlm, TS)
#         return

# Note: the RRTM modules are compiled in the 'RRTMG' directory:
cdef extern:
    void c_rrtmg_lw_init(double *cpdair)
    void c_rrtmg_lw (
             int *ncol    ,int *nlay    ,int *icld    ,int *idrv    ,
             double *play    ,double *plev    ,double *tlay    ,double *tlev    ,double *tsfc    ,
             double *h2ovmr  ,double *o3vmr   ,double *co2vmr  ,double *ch4vmr  ,double *n2ovmr  ,double *o2vmr,
             double *cfc11vmr,double *cfc12vmr,double *cfc22vmr,double *ccl4vmr ,double *emis    ,
             int *inflglw ,int *iceflglw,int *liqflglw,double *cldfr   ,
             double *taucld  ,double *cicewp  ,double *cliqwp  ,double *reice   ,double *reliq   ,
             double *tauaer  ,
             double *uflx    ,double *dflx    ,double *hr      ,double *uflxc   ,double *dflxc,  double *hrc,
             double *duflx_dt,double *duflxc_dt )
    void c_rrtmg_sw_init(double *cpdair)
    void c_rrtmg_sw (int *ncol    ,int *nlay    ,int *icld    ,int *iaer    ,
             double *play    ,double *plev    ,double *tlay    ,double *tlev    ,double *tsfc    ,
             double *h2ovmr  ,double *o3vmr   ,double *co2vmr  ,double *ch4vmr  ,double *n2ovmr  ,double *o2vmr,
             double *asdir   ,double *asdif   ,double *aldir   ,double *aldif   ,
             double *coszen  ,double *adjes   ,int *dyofyr  ,double *scon    ,
             int *inflgsw ,int *iceflgsw,int *liqflgsw,double *cldfr   ,
             double *taucld  ,double *ssacld  ,double *asmcld  ,double *fsfcld  ,
             double *cicewp  ,double *cliqwp  ,double *reice   ,double *reliq   ,
             double *tauaer  ,double *ssaaer  ,double *asmaer  ,double *ecaer   ,
             double *swuflx  ,double *swdflx  ,double *swhr    ,double *swuflxc ,double *swdflxc ,double *swhrc)


cdef class RadiationIsdac:
    def __init__(self, namelist):
        return

    cpdef initialize(self):
        return

    cpdef initialize_profiles(self, MixedLayerModel.MixedLayerModel mlm):
        self.net_lw_flux = np.zeros((mlm.nz,), dtype=np.double)
        # print('net_lw_flux initialized')
        return

    cpdef update(self, MixedLayerModel.MixedLayerModel mlm, TimeStepping.TimeStepping TS):
        cdef:
            double kap = 170.0
            double f0 = 72.0
            double f1 = 15.0
            Py_ssize_t nz = mlm.nz
            Py_ssize_t k
            double [:] f_rad = np.zeros((nz+1,), dtype=np.double)
            double [:] f_heat = np.zeros((nz,), dtype=np.double)
            double q_1 = 0.0
            double q_0 = 0.0

        f_rad[0] = f1 * exp(-q_1)
        for k in xrange(1, nz + 1):
            q_1 += kap * mlm.rho[k-1] * mlm.ql[k - 1] * mlm.dz
            f_rad[k] += f1 * exp(-q_1)

        f_rad[nz] += f0 * exp(-q_0)
        for k in xrange(nz - 1, -1, -1):
            q_0 += kap * mlm.rho[k] * mlm.ql[k] * mlm.dz
            f_rad[k] += f0 * exp(-q_0)

        for k in xrange(nz):
            # f_heat[k] = -(f_rad[k + 1] - f_rad[k]) / mlm.dz / mlm.rho0
            self.net_lw_flux[k] = f_rad[k+1]

        return


cdef class Radiation:
    def __init__(self, namelist):

        # Required for surface energy budget calculations, can also be used for stats io
        self.srf_lw_down = 0.0
        self.srf_sw_down = 0.0
        self.srf_lw_up = 0.0
        self.srf_sw_down = 0.0

        self.profile_name = 'arctic'

        try:
            self.n_buffer = namelist['radiation']['n_buffer']
        except:
            self.n_buffer = 15

        try:
            self.stretch_factor = namelist['radiation']['stretch_factor']
        except:
            self.stretch_factor = 1.5

        self.patch_pressure = 650.00*100.0

        # Namelist options related to gas concentrations
        self.co2_factor = 1.0
        self.h2o_factor = 1.0

        # Namelist options related to insolation
        self.dyofyr = 0
        self.adjes = 0.0
        self.scon = 1360.0
        self.coszen = 2.0/np.pi
        self.adif = 0.06
        self.adir = 0.0
        self.uniform_reliq = False

        try:
            self.radiation_frequency = namelist['radiation']['frequency']
        except:
            self.radiation_frequency = 60.0

        self.next_radiation_calculate = 0.0

        try:
            self.IsdacCC_dT = namelist['initial']['dSST'] + namelist['initial']['dTi'] - 5.0
            print('IsdacCC case: RRTM profiles are shifted according to %2.2f temperature change.'%(self.IsdacCC_dT))
        except:
            self.IsdacCC_dT = 0.0

        # self.bl = MixedLayerModel.boundary_layer_profiles(namelist)

        self.toa_lw_up = 0.0
        self.toa_lw_down = 0.0
        self.toa_lw_up_clear = 0.0
        self.toa_lw_down_clear = 0.0


        return

    cpdef initialize(self, MixedLayerModel.MixedLayerModel mlm, NetCDFIO_Stats NS):

        self.heating_rate = np.zeros((mlm.nz,), dtype=np.double)
        self.net_lw_flux = np.zeros((mlm.nz,), dtype=np.double)

        self.uflux_lw = np.zeros((mlm.nz,), dtype=np.double)
        self.dflux_lw = np.zeros((mlm.nz,), dtype=np.double)
        self.uflux_lw_clear = np.zeros((mlm.nz,), dtype=np.double)
        self.dflux_lw_clear = np.zeros((mlm.nz,), dtype=np.double)

        NS.add_profile('net_lw_flux')
        NS.add_profile('radiative_heating_rate')

        NS.add_profile('lw_flux_up')
        NS.add_profile('lw_flux_down')
        NS.add_profile('lw_flux_up_clear')
        NS.add_profile('lw_flux_down_clear')

        NS.add_ts('toa_lw_flux_up_clear')
        NS.add_ts('toa_lw_flux_down_clear')
        NS.add_ts('toa_lw_flux_up')
        NS.add_ts('toa_lw_flux_down')

        return

    cpdef initialize_profiles(self, MixedLayerModel.MixedLayerModel mlm):

        cdef:
            # Py_ssize_t qv_shift = DV.get_varshift(Gr, 'qv')
            # Py_ssize_t t_shift = DV.get_varshift(Gr, 'temperature')
            # double [:,:] qv_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[qv_shift])
            # double [:,:] t_pencils =  self.z_pencil.forward_double(&Gr.dims, Pa, &DV.values[t_shift])

            Py_ssize_t nz = mlm.nz
            Py_ssize_t i,k


        # mlm.get_profiles(mlm_vars)

        # Construct the extension of the profiles, including a blending region between the given profile and LES domain (if desired)
        pressures = profile_data[self.profile_name]['pressure'][:]
        temperatures = profile_data[self.profile_name]['temperature'][:]
        # vapor_mixing_ratios = profile_data[self.profile_name]['vapor_mixing_ratio'][:]
        specific_humidity = profile_data[self.profile_name]['specific_humidity'][:]

        n_profile = len(pressures[pressures<=self.patch_pressure]) # nprofile = # of points in the fixed profile to use
        self.n_ext =  n_profile + self.n_buffer # n_ext = total # of points to add to LES domain (buffer portion + fixed profile portion)


        # Create the space for the extensions (to be tacked on to top of LES pencils)
        # we declare these as class members in case we want to modify the buffer zone during run time
        # i.e. if there is some drift to top of LES profiles

        self.p_ext = np.zeros((self.n_ext,),dtype=np.double)
        self.t_ext = np.zeros((self.n_ext,),dtype=np.double)
        self.rv_ext = np.zeros((self.n_ext,),dtype=np.double)

        cdef Py_ssize_t count = 0
        for k in xrange(len(pressures)-n_profile, len(pressures)):
            self.p_ext[self.n_buffer+count] = pressures[k]
            qt_new = get_humidity(temperatures[k], specific_humidity[k], pressures[k], temperatures[k]+self.IsdacCC_dT)
            self.t_ext[self.n_buffer+count] = temperatures[k] + self.IsdacCC_dT
            self.rv_ext[self.n_buffer+count] = qt_new / (1.0 - qt_new)
            count += 1


        # Now  create the buffer zone
        if self.n_buffer > 0:
            #dp = np.abs(Ref.p0_half_global[nz + gw -1] - Ref.p0_half_global[nz + gw -2])
            dp = np.abs(mlm.pressure[-1] - mlm.pressure[-2])
            #self.p_ext[0] = Ref.p0_half_global[nz + gw -1] - dp
            self.p_ext[0] = mlm.pressure[-1] - dp

            # print(self.p_ext[0])
            for i in range(1,self.n_buffer):
                self.p_ext[i] = self.p_ext[i-1] - (i+1.0)**self.stretch_factor * dp

            # for i in xrange(self.n_ext):
                # print i, self.p_ext[i]

            # Pressures of "data" points for interpolation, must be INCREASING pressure
            xi = np.array([self.p_ext[self.n_buffer+1],self.p_ext[self.n_buffer],mlm.pressure[-1],mlm.pressure[-2] ],dtype=np.double)
            # print(xi)


            # interpolation for temperature
            ti = np.array([self.t_ext[self.n_buffer+1],self.t_ext[self.n_buffer], mlm.temperature[-1], mlm.temperature[-2] ], dtype = np.double)
            # interpolation for vapor mixing ratio
            # rv_m2 = qv_pencils[0, nz-2]/ (1.0 - qv_pencils[0, nz-2])
            # rv_m1 = qv_pencils[0,nz-1]/(1.0-qv_pencils[0,nz-1])

            rv_m2 = mlm.qv[nz-2]/(1.0 - mlm.qv[nz-2])
            rv_m1 = mlm.qv[nz-1]/(1.0 - mlm.qv[nz-1])

            ri = np.array([self.rv_ext[self.n_buffer+1],self.rv_ext[self.n_buffer], rv_m1, rv_m2 ], dtype = np.double)

            for i in xrange(self.n_buffer):
                self.rv_ext[i] = pchip_interpolate(xi, ri, self.p_ext[i] )
                self.t_ext[i] = pchip_interpolate(xi, ti, self.p_ext[i])

        # # Plotting to evaluate implementation of buffer zone
        # plt.figure(1)
        # plt.scatter(self.rv_ext,self.p_ext)
        # plt.plot(vapor_mixing_ratios, pressures)
        # plt.plot(mlm.qv, mlm.pressure)
        # plt.savefig('rrtm_buffer_rv.png')
        # plt.figure(2)
        # plt.scatter(self.t_ext,self.p_ext)
        # plt.plot(temperatures,pressures)
        # plt.plot(mlm.temperature, mlm.pressure)
        # plt.savefig('rrtm_buffer_t.png')
        # plt.figure(10)
        # plt.plot(mlm.ql, mlm.pressure)
        # plt.show()

        self.p_full = np.zeros((self.n_ext+nz,), dtype=np.double)
        self.pi_full = np.zeros((self.n_ext+1+nz,),dtype=np.double)

        self.p_full[0:nz] = mlm.pressure #Ref.p0_half_global[gw:nz+gw] # at cell center
        self.p_full[nz:]=self.p_ext[:]

        # self.pi_full[0:nz] = Ref.p0_global[gw:nz+gw] # at cell interface
        self.pi_full[0:nz] = mlm.pressure_i[1:]

        for i in range(nz,self.n_ext+nz):
            self.pi_full[i] = (self.p_full[i] + self.p_full[i-1]) * 0.5
        self.pi_full[self.n_ext +  nz] = 2.0 * self.p_full[self.n_ext + nz -1 ] - self.pi_full[self.n_ext + nz -1]

        # try to get ozone
        try:
            o3_trace = profile_data[self.profile_name]['o3_vmr'][:]   # O3 VMR (from SRF to TOP)
            o3_pressure = profile_data[self.profile_name]['pressure'][:]/100.0       # Pressure (from SRF to TOP) in hPa
            # can't do simple interpolation... Need to conserve column path !!!
            use_o3in = True
        except:
            try:
                o3_trace = profile_data[self.profile_name]['o3_mr'][:]*28.97/47.9982   # O3 MR converted to VMR
                o3_pressure = profile_data[self.profile_name]['pressure'][:]/100.0       # Pressure (from SRF to TOP) in hPa
                # can't do simple interpolation... Need to conserve column path !!!
                use_o3in = True

            except:
                print('O3 profile not set so default RRTM profile will be used.')
                use_o3in = False

        #Initialize rrtmg_lw and rrtmg_sw
        cdef double cpdair = np.float64(cpd)
        c_rrtmg_lw_init(&cpdair)
        c_rrtmg_sw_init(&cpdair)

        # Read in trace gas data
        lw_input_file = './RRTMG/lw/data/rrtmg_lw.nc'
        lw_gas = nc.Dataset(lw_input_file,  "r")

        lw_pressure = np.asarray(lw_gas.variables['Pressure'])
        lw_absorber = np.asarray(lw_gas.variables['AbsorberAmountMLS'])
        lw_absorber = np.where(lw_absorber>2.0, np.zeros_like(lw_absorber), lw_absorber)
        lw_ngas = lw_absorber.shape[1]
        lw_np = lw_absorber.shape[0]

        # 9 Gases: O3, CO2, CH4, N2O, O2, CFC11, CFC12, CFC22, CCL4
        # From rad_driver.f90, lines 546 to 552
        trace = np.zeros((9,lw_np),dtype=np.double,order='F')
        for i in xrange(lw_ngas):
            gas_name = ''.join(lw_gas.variables['AbsorberNames'][i,:])
            if 'O3' in gas_name:
                trace[0,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CO2' in gas_name:
                trace[1,:] = lw_absorber[:,i].reshape(1,lw_np)*self.co2_factor
            elif 'CH4' in gas_name:
                trace[2,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'N2O' in gas_name:
                trace[3,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'O2' in gas_name:
                trace[4,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC11' in gas_name:
                trace[5,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC12' in gas_name:
                trace[6,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CFC22' in gas_name:
                trace[7,:] = lw_absorber[:,i].reshape(1,lw_np)
            elif 'CCL4' in gas_name:
                trace[8,:] = lw_absorber[:,i].reshape(1,lw_np)

        # From rad_driver.f90, lines 585 to 620
        trpath = np.zeros((nz + self.n_ext + 1, 9),dtype=np.double,order='F')
        # plev = self.pi_full[:]/100.0
        for i in xrange(1, nz + self.n_ext + 1):
            trpath[i,:] = trpath[i-1,:]
            if (self.pi_full[i-1]/100.0 > lw_pressure[0]):
                trpath[i,:] = trpath[i,:] + (self.pi_full[i-1]/100.0 - np.max((self.pi_full[i]/100.0,lw_pressure[0])))/g*trace[:,0]
            for m in xrange(1,lw_np):
                #print i, m
                plow = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, lw_pressure[m-1]))))
                pupp = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, lw_pressure[m]))))
                if (plow > pupp):
                    pmid = 0.5*(plow+pupp)
                    wgtlow = (pmid-lw_pressure[m])/(lw_pressure[m-1]-lw_pressure[m])
                    wgtupp = (lw_pressure[m-1]-pmid)/(lw_pressure[m-1]-lw_pressure[m])
                    trpath[i,:] = trpath[i,:] + (plow-pupp)/g*(wgtlow*trace[:,m-1]  + wgtupp*trace[:,m])
            if (self.pi_full[i]/100.0 < lw_pressure[lw_np-1]):
                trpath[i,:] = trpath[i,:] + (np.min((self.pi_full[i-1]/100.0,lw_pressure[lw_np-1]))-self.pi_full[i]/100.0)/g*trace[:,lw_np-1]

        tmpTrace = np.zeros((nz + self.n_ext,9),dtype=np.double,order='F')
        for i in xrange(9):
            for k in xrange(nz + self.n_ext):
                tmpTrace[k,i] = g*100.0/(self.pi_full[k]-self.pi_full[k+1])*(trpath[k+1,i]-trpath[k,i])

        if use_o3in == False:
            self.o3vmr  = np.array(tmpTrace[:,0],dtype=np.double, order='F')
        else:
            # o3_trace, o3_pressure
            trpath_o3 = np.zeros(nz + self.n_ext+1, dtype=np.double, order='F')
            # plev = self.pi_full/100.0
            o3_np = o3_trace.shape[0]
            for i in xrange(1, nz + self.n_ext+1):
                trpath_o3[i] = trpath_o3[i-1]
                if (self.pi_full[i-1]/100.0 > o3_pressure[0]):
                    trpath_o3[i] = trpath_o3[i] + (self.pi_full[i-1]/100.0 - np.max((self.pi_full[i]/100.0,o3_pressure[0])))/g*o3_trace[0]
                for m in xrange(1,o3_np):
                    #print i, m
                    plow = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, o3_pressure[m-1]))))
                    pupp = np.min((self.pi_full[i-1]/100.0,np.max((self.pi_full[i]/100.0, o3_pressure[m]))))
                    if (plow > pupp):
                        pmid = 0.5*(plow+pupp)
                        wgtlow = (pmid-o3_pressure[m])/(o3_pressure[m-1]-o3_pressure[m])
                        wgtupp = (o3_pressure[m-1]-pmid)/(o3_pressure[m-1]-o3_pressure[m])
                        trpath_o3[i] = trpath_o3[i] + (plow-pupp)/g*(wgtlow*o3_trace[m-1]  + wgtupp*o3_trace[m])
                if (self.pi_full[i]/100.0 < o3_pressure[o3_np-1]):
                    trpath_o3[i] = trpath_o3[i] + (np.min((self.pi_full[i-1]/100.0,o3_pressure[o3_np-1]))-self.pi_full[i]/100.0)/g*o3_trace[o3_np-1]
            tmpTrace_o3 = np.zeros( nz + self.n_ext, dtype=np.double, order='F')
            for k in xrange(nz + self.n_ext):
                tmpTrace_o3[k] = g *100.0/(self.pi_full[k]-self.pi_full[k+1])*(trpath_o3[k+1]-trpath_o3[k])
            self.o3vmr = np.array(tmpTrace_o3[:],dtype=np.double, order='F')

        self.co2vmr = np.array(tmpTrace[:,1],dtype=np.double, order='F')
        self.ch4vmr =  np.array(tmpTrace[:,2],dtype=np.double, order='F')
        self.n2ovmr =  np.array(tmpTrace[:,3],dtype=np.double, order='F')
        self.o2vmr  =  np.array(tmpTrace[:,4],dtype=np.double, order='F')
        self.cfc11vmr =  np.array(tmpTrace[:,5],dtype=np.double, order='F')
        self.cfc12vmr =  np.array(tmpTrace[:,6],dtype=np.double, order='F')
        self.cfc22vmr = np.array( tmpTrace[:,7],dtype=np.double, order='F')
        self.ccl4vmr  =  np.array(tmpTrace[:,8],dtype=np.double, order='F')

        return

    cpdef update(self, MixedLayerModel.MixedLayerModel mlm, TimeStepping.TimeStepping TS):
        if TS.rk_step == 0:
            if self.radiation_frequency <= 0.0:
                self.update_RRTM(mlm)
            elif TS.t >= self.next_radiation_calculate:
                self.update_RRTM(mlm)
                self.next_radiation_calculate = (TS.t//self.radiation_frequency + 1.0) * self.radiation_frequency
        return

    cdef update_RRTM(self, MixedLayerModel.MixedLayerModel mlm):
        cdef:
            Py_ssize_t nz = mlm.nz
            Py_ssize_t nz_full = self.n_ext + nz
            Py_ssize_t k
            Py_ssize_t n_pencils = 1

        cdef:
            double [:] rl_full = np.zeros((nz_full,), dtype=np.double, order='F')
            double [:] ri_full = np.zeros((nz_full,), dtype=np.double, order='F')
            double [:] play_in = np.zeros((nz_full,), dtype=np.double, order='F')
            double [:] plev_in = np.zeros((nz_full + 1), dtype=np.double, order='F')
            double [:] tlay_in = np.zeros((nz_full,), dtype=np.double, order='F')
            double [:] tlev_in = np.zeros((nz_full + 1), dtype=np.double, order='F')
            double [:] tsfc_in = np.ones((n_pencils),dtype=np.double,order='F') * mlm.t_surface
            double [:] h2ovmr_in = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] o3vmr_in  = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] co2vmr_in = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] ch4vmr_in = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] n2ovmr_in = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] o2vmr_in  = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] cfc11vmr_in = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] cfc12vmr_in = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] cfc22vmr_in = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] ccl4vmr_in = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:,:] emis_in = np.ones((n_pencils,16),dtype=np.double,order='F') * 0.95
            double [:] cldfr_in  = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] cicewp_in = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] cliqwp_in = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] reice_in  = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] reliq_in  = np.zeros((nz_full,),dtype=np.double,order='F')
            double [:] coszen_in = np.ones((n_pencils),dtype=np.double,order='F') *self.coszen
            double [:] asdir_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adir
            double [:] asdif_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adif
            double [:] aldir_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adir
            double [:] aldif_in = np.ones((n_pencils),dtype=np.double,order='F') * self.adif
            double [:,:] taucld_lw_in  = np.zeros((16,nz_full,),dtype=np.double,order='F')
            double [:,:] tauaer_lw_in  = np.zeros((nz_full,16),dtype=np.double,order='F')
            double [:,:] taucld_sw_in  = np.zeros((14,nz_full,),dtype=np.double,order='F')
            double [:,:] ssacld_sw_in  = np.zeros((14,nz_full,),dtype=np.double,order='F')
            double [:,:] asmcld_sw_in  = np.zeros((14,nz_full,),dtype=np.double,order='F')
            double [:,:] fsfcld_sw_in  = np.zeros((14,nz_full,),dtype=np.double,order='F')
            double [:,:] tauaer_sw_in  = np.zeros((nz_full,14),dtype=np.double,order='F')
            double [:,:] ssaaer_sw_in  = np.zeros((nz_full,14),dtype=np.double,order='F')
            double [:,:] asmaer_sw_in  = np.zeros((nz_full,14),dtype=np.double,order='F')
            double [:,:] ecaer_sw_in  = np.zeros((nz_full,6),dtype=np.double,order='F')

            # Output
            double[:] uflx_lw_out = np.zeros((nz_full +1),dtype=np.double,order='F')
            double[:] dflx_lw_out = np.zeros((nz_full +1),dtype=np.double,order='F')
            double[:] hr_lw_out = np.zeros((nz_full,),dtype=np.double,order='F')
            double[:] uflxc_lw_out = np.zeros((nz_full + 1),dtype=np.double,order='F')
            double[:] dflxc_lw_out = np.zeros((nz_full +1),dtype=np.double,order='F')
            double[:] hrc_lw_out = np.zeros((nz_full,),dtype=np.double,order='F')
            double[:] duflx_dt_out = np.zeros((nz_full +1),dtype=np.double,order='F')
            double[:] duflxc_dt_out = np.zeros((nz_full +1),dtype=np.double,order='F')
            double[:] uflx_sw_out = np.zeros((nz_full +1),dtype=np.double,order='F')
            double[:] dflx_sw_out = np.zeros((nz_full +1),dtype=np.double,order='F')
            double[:] hr_sw_out = np.zeros((nz_full,),dtype=np.double,order='F')
            double[:] uflxc_sw_out = np.zeros((nz_full +1),dtype=np.double,order='F')
            double[:] dflxc_sw_out = np.zeros((nz_full +1),dtype=np.double,order='F')
            double[:] hrc_sw_out = np.zeros((nz_full,),dtype=np.double,order='F')

            double rv_to_reff = np.exp(np.log(1.2)**2.0)*10.0*1000.0

        # with nogil:
        for k in xrange(nz, nz_full):
            tlay_in[k] = self.t_ext[k-nz]
            h2ovmr_in[k] = self.rv_ext[k-nz] * Rv/Rd * self.h2o_factor
                # Assuming for now that there is no condensate above LES domain!
        for k in xrange(nz):
            tlay_in[k] = mlm.temperature[k]
            h2ovmr_in[k] = mlm.qv[k]/ (1.0 - mlm.qv[k])* Rv/Rd * self.h2o_factor
            rl_full[k] = (mlm.ql[k])/ (1.0 - mlm.qv[k])
            ri_full[k] = (mlm.qi[k])/ (1.0 - mlm.qv[k])
            # ri_full[k] = 0.0
            cliqwp_in[k] = ((mlm.ql[k])/ (1.0 - mlm.qv[k])
                               *1.0e3*(self.pi_full[k] - self.pi_full[k+1])/g)
            cicewp_in[k] = ((mlm.qi[k])/ (1.0 - mlm.qv[k])
                               *1.0e3*(self.pi_full[k] - self.pi_full[k+1])/g)
            if mlm.ql[k] + mlm.qi[k] > ql_threshold:
                cldfr_in[k] = 1.0


        with nogil:
            for k in xrange(nz_full):
                play_in[k] = self.p_full[k]/100.0
                o3vmr_in[k] = self.o3vmr[k]
                co2vmr_in[k] = self.co2vmr[k]
                ch4vmr_in[k] = self.ch4vmr[k]
                n2ovmr_in[k] = self.n2ovmr[k]
                o2vmr_in [k] = self.o2vmr[k]
                cfc11vmr_in[k] = self.cfc11vmr[k]
                cfc12vmr_in[k] = self.cfc12vmr[k]
                cfc22vmr_in[k] = self.cfc22vmr[k]
                ccl4vmr_in[k] = self.ccl4vmr[k]


                if self.uniform_reliq:
                    reliq_in[k] = 14.0*cldfr_in[k]
                else:
                    reliq_in[k] = ((3.0*self.p_full[k]/Rd/tlay_in[k]*rl_full[k]/
                                        fmax(cldfr_in[k],1.0e-6))/(4.0*pi*1.0e3*100.0))**(1.0/3.0)
                    reliq_in[k] = fmin(fmax(reliq_in[ k]*rv_to_reff, 2.5), 60.0)

                # Boudala et al. (2002) Eqn 10a
                reice_in[k] = 53.005 * ((self.p_full[k]/Rd/tlay_in[k]*ri_full[k]*1.0e3)/
                                            fmax(cldfr_in[k],1.0e-6)) ** 0.06 \
                                      * exp(0.013*(tlay_in[k] - 273.16))
                reice_in[k] = fmin(fmax(reice_in[k]/1.54, 5.0), 140.0) # Threshold from rrtmg sw instruction


            with gil:
                tlev_in[0] = mlm.t_surface
            plev_in[0] = self.pi_full[0]/100.0
            for k in xrange(1,nz_full):
                tlev_in[k] = 0.5*(tlay_in[k-1]+tlay_in[k])
                plev_in[k] = self.pi_full[k]/100.0
            tlev_in[nz_full] = 2.0*tlay_in[nz_full-1] - tlev_in[nz_full-1]
            plev_in[nz_full] = self.pi_full[nz_full]/100.0

        # Plot the variables to check
        # plt.figure(3)
        # plt.subplot(121)
        # plt.plot(tlay_in, self.p_full)
        # plt.plot(tlev_in, self.pi_full)
        # plt.xlabel('tlay_in tlev_in')
        # plt.subplot(122)
        # plt.plot(h2ovmr_in, self.p_full)
        # plt.xlabel('h2ovmr_in')
        # plt.figure(4)
        # plt.subplot(121)
        # plt.plot(rl_full, self.p_full)
        # plt.xlabel('rl_full')
        # plt.subplot(122)
        # plt.plot(cliqwp_in, self.p_full)
        # plt.plot(cicewp_in, self.p_full)
        # plt.xlabel('cliqwp_in & cicewp_in')
        # plt.figure(5)
        # plt.subplot(121)
        # plt.plot(cldfr_in, self.p_full)
        # plt.xlabel('cldfr_in')
        # plt.subplot(122)
        # plt.plot(reliq_in, self.p_full)
        # plt.xlabel('reliq_in')
        # plt.show()

        cdef:
            int ncol = n_pencils
            int nlay = nz_full
            int icld = 1
            int idrv = 0
            int iaer = 0
            int inflglw = 2
            int iceflglw = 3
            int liqflglw = 1
            int inflgsw = 2
            int iceflgsw = 3
            int liqflgsw = 1

        c_rrtmg_lw (
             &ncol    ,&nlay    ,&icld    ,&idrv,
             &play_in[0]    ,&plev_in[0]    ,&tlay_in[0]    ,&tlev_in[0]    ,&tsfc_in[0]    ,
             &h2ovmr_in[0]  ,&o3vmr_in[0]   ,&co2vmr_in[0]  ,&ch4vmr_in[0]  ,&n2ovmr_in[0]  ,&o2vmr_in[0],
             &cfc11vmr_in[0],&cfc12vmr_in[0],&cfc22vmr_in[0],&ccl4vmr_in[0] ,&emis_in[0,0]    ,
             &inflglw ,&iceflglw,&liqflglw,&cldfr_in[0]   ,
             &taucld_lw_in[0,0]  ,&cicewp_in[0]  ,&cliqwp_in[0]  ,&reice_in[0]   ,&reliq_in[0]   ,
             &tauaer_lw_in[0,0]  ,
             &uflx_lw_out[0]    ,&dflx_lw_out[0]    ,&hr_lw_out[0]      ,&uflxc_lw_out[0]   ,&dflxc_lw_out[0],  &hrc_lw_out[0],
             &duflx_dt_out[0],&duflxc_dt_out[0] )

        c_rrtmg_sw (
            &ncol, &nlay, &icld, &iaer, &play_in[0], &plev_in[0], &tlay_in[0], &tlev_in[0],&tsfc_in[0],
            &h2ovmr_in[0], &o3vmr_in[0], &co2vmr_in[0], &ch4vmr_in[0], &n2ovmr_in[0],&o2vmr_in[0],
             &asdir_in[0]   ,&asdif_in[0]   ,&aldir_in[0]   ,&aldif_in[0]   ,
             &coszen_in[0]  ,&self.adjes   ,&self.dyofyr  ,&self.scon   ,
             &inflgsw ,&iceflgsw,&liqflgsw,&cldfr_in[0]   ,
             &taucld_sw_in[0,0]  ,&ssacld_sw_in[0,0]  ,&asmcld_sw_in[0,0]  ,&fsfcld_sw_in[0,0]  ,
             &cicewp_in[0]  ,&cliqwp_in[0]  ,&reice_in[0]   ,&reliq_in[0]   ,
             &tauaer_sw_in[0,0]  ,&ssaaer_sw_in[0,0]  ,&asmaer_sw_in[0,0]  ,&ecaer_sw_in[0,0]   ,
             &uflx_sw_out[0]    ,&dflx_sw_out[0]    ,&hr_sw_out[0]      ,&uflxc_sw_out[0]   ,&dflxc_sw_out[0], &hrc_sw_out[0])


        for k in xrange(nz):
            self.heating_rate[k] = (hr_lw_out[k] + hr_sw_out[k]) * mlm.rho[k] * cpd/86400.0
            self.net_lw_flux[k] = uflx_lw_out[k] - dflx_lw_out[k]
            self.uflux_lw[k] = uflx_lw_out[k]
            self.dflux_lw[k] = dflx_lw_out[k]
            self.uflux_lw_clear[k] = uflxc_lw_out[k]
            self.dflux_lw_clear[k] = dflxc_lw_out[k]

        self.toa_lw_up = uflx_lw_out[nz_full]
        self.toa_lw_up_clear = uflxc_lw_out[nz_full]
        self.toa_lw_down = dflx_lw_out[nz_full]
        self.toa_lw_down_clear = dflxc_lw_out[nz_full]

        # plt.figure(6)
        # plt.subplot(121)
        # plt.plot(uflx_lw_out, self.pi_full)
        # plt.plot(dflx_lw_out, self.pi_full)
        # plt.xlabel('lw_out')
        # plt.subplot(122)
        # plt.plot(uflx_sw_out, self.pi_full)
        # plt.plot(dflx_sw_out, self.pi_full)
        # plt.xlabel('sw_out')
        # plt.figure(7)
        # plt.subplot(121)
        # plt.plot(self.heating_rate, mlm.pressure)
        # plt.xlabel('heating rate')
        # plt.subplot(122)
        # plt.plot(self.net_lw_flux, mlm.pressure)
        # plt.xlabel('net lw flux')
        # plt.show()


        return

    cpdef stats_io(self, NetCDFIO_Stats NS):

        NS.write_profile('net_lw_flux', self.net_lw_flux)
        NS.write_profile('radiative_heating_rate', self.heating_rate)

        NS.write_profile('lw_flux_up', self.uflux_lw)
        NS.write_profile('lw_flux_down', self.dflux_lw)
        NS.write_profile('lw_flux_up_clear', self.uflux_lw_clear)
        NS.write_profile('lw_flux_down_clear', self.dflux_lw_clear)

        NS.write_ts('toa_lw_flux_up', self.toa_lw_up)
        NS.write_ts('toa_lw_flux_down', self.toa_lw_down)
        NS.write_ts('toa_lw_flux_up_clear', self.toa_lw_up_clear)
        NS.write_ts('toa_lw_flux_down_clear', self.toa_lw_down_clear)

        return

def get_humidity(temperature_old, qt_old, pressure, temperature_new):
    pv_star_1 = saturation_vapor_pressure(temperature_old)
    pv_1 = (pressure * qt_old) / (eps_v * (1.0 - qt_old) + qt_old)
    rh_ = pv_1 / pv_star_1
    pv_star_2 = saturation_vapor_pressure(temperature_new)
    pv_2 = rh_ * pv_star_2
    qt_new = 1.0/(eps_vi * (pressure - pv_2)/pv_2 + 1.0)
    return qt_new