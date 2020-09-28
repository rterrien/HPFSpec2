import numpy as np
import scipy.constants
c_kms = scipy.constants.c / 1e3

def redshift(x, vo=0., ve=0.,def_wlog=False):
    """
    x: The measured wavelength.
    v: Speed of the observer [km/s].
    ve: Speed of the emitter [km/s].

    Returns:
      The emitted wavelength l'.

    Notes:
      f_m = f_e (Wright & Eastman 2014)

    """
    if np.isnan(vo):
        vo = 0     # propagate nan as zero (@calibration in fib B)
    a = (1.0+vo/c_kms) / (1.0+ve/c_kms)
    if def_wlog:
        return x + np.log(a)   # logarithmic
        #return x + a          # logarithmic + approximation v << c
    else:
        return x * a
        #return x / (1.0-v/c)
