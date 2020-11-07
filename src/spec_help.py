from __future__ import print_function
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np
import scipy.ndimage.filters
import seaborn as sns
import scipy.interpolate
import astropy.time
import astropy.io
import specutils
import astropy.units as u
from astropy.nddata import StdDevUncertainty
import spectres
from astropy.modeling.models import Chebyshev1D
from specutils.fitting import fit_generic_continuum
c = 299792.4580   # [km/s]
cp = sns.color_palette("colorblind")

def vacuum_to_air(wl):
    """
    Converts vacuum wavelengths to air wavelengths using the Ciddor 1996 formula.

    :param wl: input vacuum wavelengths
    :type wl: numpy.ndarray

    :returns: numpy.ndarray

    .. note::

        CA Prieto recommends this as more accurate than the IAU standard.

    """
    if not isinstance(wl, np.ndarray):
        wl = np.array(wl)

    sigma = (1e4 / wl) ** 2
    f = 1.0 + 0.05792105 / (238.0185 - sigma) + 0.00167917 / (57.362 - sigma)
    return wl / f

def get_flux_from_file(filename,o=None,ext=1):
    """
    Get flat flux for a given order

    NOTES:
        f_flat = get_flat_flux('MASTER_FLATS/20180804/alphabright_fcu_march02_july21_deblazed.fits',5)
    """
    hdu = astropy.io.fits.open(filename)
    if o is None:
        return hdu[ext].data
    else:
        return hdu[ext].data[o]

def ax_apply_settings(ax,ticksize=None):
    """
    Apply axis settings that I keep applying
    """
    ax.minorticks_on()
    if ticksize is None:
        ticksize = 12
    ax.tick_params(pad=3,labelsize=ticksize)
    ax.grid(lw=0.5,alpha=0.5)

def jd2datetime(times):
    return np.array([astropy.time.Time(time,format="jd",scale="utc").datetime for time in times])


def detrend_maxfilter_gaussian(flux,n_max=300,n_gauss=500,plot=False):
    """
    A function useful to estimate spectral continuum

    INPUT:
        flux: a vector of fluxes
        n_max: window for max filter
        n_gauss: window for gaussian filter smoothing

    OUTPUT:
        flux/trend - the trend corrected flux
        trend - the estimated trend

    EXAMPLE:
        f_norm, trend = detrend_maxfilter_gaussian(df_temp.flux,plot=True)
    """
    flux_filt = scipy.ndimage.filters.maximum_filter1d(flux,n_max)
    trend = scipy.ndimage.filters.gaussian_filter1d(flux_filt,sigma=n_gauss)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(flux)
        ax.plot(trend)
        fig, ax = plt.subplots()
        ax.plot(flux/trend)
    return flux/trend, trend

def average_ccf(ccfs):
    """
    A function to average ccfs
    
    INPUT:
        An array of CCFs
        
    OUTPUT:
    
    """
    ccfs = np.sum(ccfs,axis=0)
    ccfs /= np.nanmedian(ccfs)
    return ccfs

def barshift(x, v=0.,def_wlog=False):
    """
    Convenience function for redshift.

    x: The measured wavelength.
    v: Speed of the observer [km/s].

    Returns:
      The true wavelengths at the barycentre.

    """
    return redshift(x, vo=v,def_wlog=def_wlog)


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
    a = (1.0+vo/c) / (1.0+ve/c)
    if def_wlog:
        return x + np.log(a)   # logarithmic
        #return x + a          # logarithmic + approximation v << c
    else:
        return x * a
        #return x / (1.0-v/c)

def weighted_rv_average(x,e):
    """
    Calculate weigted average

    INPUT:
        x
        e

    OUTPUT:
        xx: weighted average of x
        ee: associated error
    """
    xx, ww = np.average(x,weights=e**(-2.),returned=True)
    return xx, np.sqrt(1./ww)

def weighted_average(error):
    """
    Weighted average. Useful to add rv errors together.
    """
    w = np.array(error)**(-2.)
    return np.sum(w)**(-0.5)

def resample_interpolate(x,y,x_new,kind='cubic',fill_value=np.nan):
    """ Simple interpolation-based resampling
    
    Just a wrapper around scipy.interpolate.interp1d.
    
    Parameters
    ----------
    x : {ndarray}
        X-values to interpolate
    y : {ndarray}
        y-values to interpolate
    x_new : {ndarray}
        x-values to interpolate at
    kind : {str}, optional
        Kind of interpolation, argument to scipy.interpolate.interp1d. defaults to 'cubic'
    fill_value : {float}, optional
        Fill value for places where interpolation fails - defaults to np.nan
    """
    # Mask NaN
    mask = np.ma.masked_invalid(y)
    x_use = x[~mask.mask]
    y_use = y[~mask.mask]
    # Interpolate
    interp_func = scipy.interpolate.interp1d(x_use,y_use,kind=kind,fill_value=fill_value,bounds_error=False)
    return(interp_func(x_new))

def resample_to_median_sampling(x,y,e=None,kind='FluxConservingSpectRes',fill_value=np.nan,upsample_factor=1.):
    """ General-purpose resampling.
    
    Resampling routine, can do flux-conserving "spectres" resampling or simple interpolation.
    
    Parameters
    ----------
    x : {ndarray}
        X-values to interpolate [if spectrum, in ang]
    y : {ndarray}
        y-values to interpolate
    e : {ndarray}, optional
        "error" array to resample - this is only relevant for kind=FluxConservingSpectRes, which assumes
        a spectrum in flux vs wavelength, and an error array.
    kind : {str}, optional
        Kind of resampling/interpolation. Defaults to 'FluxConservingSpectRes'.
        Also valid: any "kind" for scipy.interpolate.interp1d
    fill_value : {float}, optional
        Fill value for places where interpolation fails - defaults to np.nan
    upsample_factor : {float}, optional
        Up or down-sample a spectrum - defaults to 1. which is no change
    """
    # Find median size of x-bins, range, and number of points used
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    med_dx = np.nanmedian(np.diff(x))
    # How many points are required to span the same range, with median sampling * upsample factor
    n_pts = int((x_max - x_min) / med_dx) * upsample_factor
    x_new = np.linspace(x_min,x_max,n_pts)
    if kind in ['linear','nearest','zero','slinear','quadratic','cubic','previous','next']:
        out = resample_interpolate(x,y,x_new,kind=kind,fill_value=fill_value)
    elif kind in ['FluxConservingSpectRes']:
        out = spectres.spectres(x_new, x, y, spec_errs=e, verbose=False)
        if e is not None:
            out, out_err = out[0], out[1]
            return(x_new,out,out_err)
    return(x_new,out)

def resample_combine(wl_base,wlarr,flarr,combine_type='biweight',sigma_clip=5.):
    """ Resample and combine spectra.
    
    Convenience function to resample and combine a bunch of spectra.
    
    Parameters
    ----------
    wl_base : {ndarray}
        Base wavelength array, to resample on [ang]
    wlarr : {ndarray} (n_specs x 2048)
        Wavelength array for all spectra to combine [ang] 
    flarr : {ndarray} (n_specs x 2048)
        Flux array for all spectra
    combine_type : {str}, optional
        How to combine spectra - ['biweight','mean','median','sigmaclippedmean','sigmaclippedmedian'] 
        (the default is 'biweight')
    sigma_clip : {float}, optional
        For sigma-clipped combinations, the clip limit (the default is 5.)
    """
    n_specs = len(flarr)
    n_wls = len(wl_base)    
    fullarr = np.full((n_specs,n_wls),np.nan)
    # For each spectrum, resample to the array provided in wl_base
    for si in range(n_specs):
        resampled_fl = spectres.spectres(wl_base,wlarr[si],flarr[si],verbose=False)
        fullarr[si,:] = resampled_fl

    # Combine spectra
    if combine_type == 'biweight':
        out = astropy.stats.biweight.biweight_location(fullarr,axis=0,ignore_nan=True)
    elif combine_type == 'mean':
        out = np.nanmean(fullarr,axis=0)
    elif combine_type == 'median':
        out = np.nanmedian(fullarr,axis=0)
    elif combine_type == 'sigmaclippedmean':
        mask = np.ma.masked_invalid(fullarr)
        out = astropy.stats.sigma_clipped_stats(fullarr,mask=mask,axis=0,sigma=sigma_clip)[0]
    elif combine_type == 'sigmaclippedmedian':
        mask = np.ma.masked_invalid(fullarr)
        out = astropy.stats.sigma_clipped_stats(fullarr,mask=mask,axis=0,sigma=sigma_clip)[1]
    else:
        raise(UnhandledException('Invalid combine type'))
    return(out)


def convert_to_spec1d(wl_angstrom,fl_counts,er_counts,resample=True,resample_kind='FluxConservingSpectRes',
                      resample_fill_value=np.nan,resample_upsample_factor=1.):
    """ Convert HPFSpectrum to a specutils.spec1D object
    
    Pull values from an HPFspectrum into a specutils.spec1D object.
    Pass a single order
    
    Parameters
    ----------
    wl_angstrom : {ndarray} 
        1D array of wavelengths [ang]
    fl_counts : {ndarray}
        1D array of fluxes
    er_counts : {ndarray}
        1D array of errors
    resample : {bool}, optional
        Resample the spectrum? - defaults to True
    resample_kind : {str}, optional
        Method for resampling, argument to resample_to_median_sampling (the default is 'FluxConservingSpectRes')
    resample_fill_value : {float}, optional
        Fill value for places where resampling fails (the default is np.nan)
    resample_upsample_factor : {float}, optional
        Resampling upsample factor (the default is 1., which is no upsampling)
    """
    # Filter the NaNs
    mask = np.ma.masked_invalid(fl_counts)
    wl_angstrom_in = wl_angstrom[~mask.mask] 
    fl_counts_in = fl_counts[~mask.mask]
    er_counts_in = er_counts[~mask.mask] # should this not have units attached?

    # Resample if needed
    if resample and (resample_kind is not 'FluxConservingSpectRes'):
        wl_angstrom, fl_counts = resample_to_median_sampling(wl_angstrom_in,fl_counts_in,kind=resample_kind,
                                                             fill_value=resample_fill_value,
                                                             upsample_factor=resample_upsample_factor)
        _, er_counts = resample_to_median_sampling(wl_angstrom_in,er_counts_in,kind=resample_kind,
                                                   fill_value=resample_fill_value,
                                                   upsample_factor=resample_upsample_factor)
    elif resample and (resample_kind is 'FluxConservingSpectRes'):
        wl_angstrom, fl_counts, er_counts = resample_to_median_sampling(wl_angstrom_in,fl_counts_in,e=er_counts_in,kind=resample_kind,
                                                                        fill_value=resample_fill_value,
                                                                        upsample_factor=resample_upsample_factor)
    else:
        wl_angstrom = wl_angstrom_in
        fl_counts = fl_counts_in
        er_counts = er_counts_in

    # Create a spec1d object
    out = specutils.Spectrum1D(spectral_axis=wl_angstrom*u.AA,
                               flux=fl_counts*u.count,
                               uncertainty=StdDevUncertainty(er_counts))
    return(out)


def specutils_continuum_normalize(w_ang,fl_counts,e_counts,median_window=51,model=Chebyshev1D(1),percentile_scaling=98):
    """ Continuum normalizer for one order
    
    Use specutils machinery to estimate a continuum normalization.
    
    Parameters
    ----------
    w_ang : {ndarray}
        One spectral order wavelengths [ang]
    fl_counts : {ndarray}
        One spectral order fluxes [counts]
    e_counts : {ndarray}
        One spectral order errors [counts]
    median_window : {int}
        Pre-smoothing median filter window [pixels]
    model : {astropy.modeling.models object}
        Model used to fit continuum
    """
    nonnorm = specutils.Spectrum1D(spectral_axis = w_ang * u.AA,
                                   flux = fl_counts * u.count,
                                   uncertainty = StdDevUncertainty(e_counts))
    norm_fit = fit_generic_continuum(nonnorm,median_window=median_window,model=model)
    norm_vals = norm_fit(w_ang * u.AA)
    out_fl = fl_counts * u.count / norm_vals
    out_e = e_counts / norm_vals
    if percentile_scaling is not None:
        perc = np.nanpercentile(out_fl,percentile_scaling)
        out_fl = out_fl / perc
        out_e = out_e / perc
    out_e = out_e.value
    return(out_fl, out_e)

def calculate_ew(wl,fl,limit_left,limit_right):
    # this amounts to calculating INT(1 - F / F_continuum)*d_wl with the bounds as the feature limits
    # our F_continuum is assumed to be 0 and we have a discrete sampling so use a sum
    
    # for now just force it to be that we have the feature entirely within the bounds
    assert limit_left > np.nanmin(wl)
    assert limit_right < np.nanmax(wl)
    
    # need to calculate the wavelength bin sizes to match against limits
    # each wavelength bin has a center, left, and right. We assume that we are given the center
    # need to calculate left and right
    bin_size = np.diff(wl)
    # assuming that the bin size doesn't change meaningfully from one bin to the next one
    bin_size = np.concatenate(([bin_size[0]],bin_size))
    bin_left = wl - bin_size/2.
    bin_right = wl + bin_size/2.
    
    # check to make sure which pixels are finite (i.e. not NaN) values to work with
    condition_finite = np.isfinite(fl)
    
    # handle pixels entirely within the bounds:
    condition_all_in = (bin_left >= limit_left) & (bin_right <= limit_right)
    
    # select the pixels that are finite and those that are all in
    use = np.nonzero(condition_finite & condition_all_in)[0]
    wluse = wl[use]
    fluse = fl[use]
    
    # recalculate bin boundaries, just in case we lost any pixels due to NaN
    bins = np.diff(wluse)
    bins = np.concatenate(([bins[0]],bins))
    
    # do the calculation and sum
    sub = (1. - fluse) * bins
    
    # add the left extra bin
    leftmost_index = use[0]
    left_extra_bin = bin_right[leftmost_index-1] - limit_left
    left_extra_val = (1. - fl[leftmost_index-1]) * left_extra_bin
    print(use)
    
    # right extra bin
    rightmost_index = use[-1]
    right_extra_bin = limit_right - bin_left[rightmost_index+1]
    right_extra_val = (1. - fl[rightmost_index+1]) * right_extra_bin
    
    return(np.sum(sub) + left_extra_val + right_extra_val)
