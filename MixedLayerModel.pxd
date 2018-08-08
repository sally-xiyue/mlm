cimport Radiation
cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats

cdef class MixedLayerModel:
    cdef:
        double gamma_thetal
        double thetal_i
        double thetal_ft
        double rh_ft
        double div_frac
        double dz
        Py_ssize_t nz
        double [:] z
        double p_surface
        double zi_i
        double t_surface
        double qt_surface
        double rho0
        double rh_i
        double [:] pressure
        double [:] temperature
        double [:] qv
        double [:] z_interface
        double [:] pressure_i
        double [:] ql
        double [:] qi
        double [:] thetal
        double [:] qt
        double [:] rho
        double [:] values
        double [:] tendencies
        double a
        double w0
        double dz_inv
        double thetal_inv

    # cpdef initialize(self, NetCDFIO_Stats NS)
    cpdef update(self, TimeStepping.TimeStepping TS, Radiation.Radiation Ra)
    cpdef stats_io(self, NetCDFIO_Stats NS)