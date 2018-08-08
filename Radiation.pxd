cimport MixedLayerModel
cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats

# cdef class Radiation_:
#     cdef:
#         public object scheme
#         # public double [:] net_lw_flux
#     cpdef initialize(self)
#     cpdef initialize_profiles(self, MixedLayerModel.MixedLayerModel mlm)
#     cpdef update(self, MixedLayerModel.MixedLayerModel mlm, TimeStepping.TimeStepping TS)
#

cdef class RadiationIsdac:
    cdef:
        public double [:] net_lw_flux
    cpdef initialize(self)
    cpdef initialize_profiles(self, MixedLayerModel.MixedLayerModel mlm)
    cpdef update(self, MixedLayerModel.MixedLayerModel mlm, TimeStepping.TimeStepping TS)


cdef class Radiation:
    cdef:
        double srf_lw_down
        double srf_lw_up
        double srf_sw_down
        double srf_sw_up
        str profile_name
        Py_ssize_t n_buffer
        double stretch_factor
        double patch_pressure

        double co2_factor
        double h2o_factor
        int dyofyr
        double scon
        double adjes
        double coszen
        double adif
        double adir
        bint uniform_reliq
        double radiation_frequency
        double next_radiation_calculate
        double [:] heating_rate
        public double [:] net_lw_flux
        Py_ssize_t n_ext
        double [:] p_ext
        double [:] t_ext
        double [:] rv_ext
        double [:] p_full
        double [:] pi_full
        double [:] o3vmr
        double [:] co2vmr
        double [:] ch4vmr
        double [:] n2ovmr
        double [:] o2vmr
        double [:] cfc11vmr
        double [:] cfc12vmr
        double [:] cfc22vmr
        double [:] ccl4vmr
        double IsdacCC_dT
        double IsdacCC_dHf

        double [:] uflux_lw
        double [:] dflux_lw
        double [:] uflux_lw_clear
        double [:] dflux_lw_clear

        double toa_lw_up
        double toa_lw_down
        double toa_lw_up_clear
        double toa_lw_down_clear


    cpdef initialize(self, MixedLayerModel.MixedLayerModel mlm, NetCDFIO_Stats NS)
    cpdef initialize_profiles(self, MixedLayerModel.MixedLayerModel mlm)
    cpdef update(self, MixedLayerModel.MixedLayerModel mlm, TimeStepping.TimeStepping TS)
    cdef update_RRTM(self, MixedLayerModel.MixedLayerModel mlm)
    cpdef stats_io(self, NetCDFIO_Stats NS)