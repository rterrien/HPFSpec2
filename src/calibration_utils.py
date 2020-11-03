import numpy as np
from scipy.signal import savgol_filter, find_peaks
from collections import OrderedDict
import copy
import scipy
import neidspec
import json
import astropy
from astropy.stats import biweight_location, sigma_clip, mad_std
from skimage import filters
from skimage import morphology
from scipy import interpolate

def find_cal_peaks(wl,fl,thres=500.,fsr_ghz=20.,distance_scaling=0.9):
    assert len(wl) == len(fl)
    f_est_hz = 3e8 / (np.nanmedian(wl) * 1e-10)
    fsr_wl = 3e8 / f_est_hz**2. * fsr_ghz * 1e9
    wl_per_pix = np.nanmedian(np.diff(wl)) * 1e-10
    fsr_pix = fsr_wl / wl_per_pix
    #print(f_est_hz/1e12,fsr_wl,wl_per_pix,fsr_pix)
    peaks = find_peaks(fl,height=thres,distance=fsr_pix*distance_scaling)[0]
    return(peaks)

def find_cal_peaks_order(wl,fl,thres=500.,fsr_ghz=20.,savgol_window=9,savgol_poly=3,
                         distance_scaling_left=0.9, distance_scaling_right=0.7):
    nel = len(wl)
    nel2 = int(nel/2)
    fl_smoothed = savgol_filter(fl,savgol_window,savgol_poly)
    wl_a, fl_a = wl[:nel2], fl_smoothed[:nel2]
    wl_b, fl_b = wl[nel2:], fl_smoothed[nel2:]
    peaks_a = find_cal_peaks(wl_a,fl_a,distance_scaling=distance_scaling_left)
    peaks_b = find_cal_peaks(wl_b,fl_b,distance_scaling=distance_scaling_right) + nel/2
    peaks_all = np.hstack((peaks_a,peaks_b))
    return(peaks_all)

def measure_peaks_order(wl,fl,peak_locs,xx=None,pix_to_wvl=None,pix_to_wvl_per_pix=None,fitfunc='fgauss_const',continuum_subtract=False,
                        continuum_subtract_kw={}):
    if xx is None:
        xx = np.arange(len(wl))
    if not isinstance(peak_locs,dict):
        peak_locs_dict = OrderedDict()
        mode_names = range(len(peak_locs))
        for mi in mode_names:
            peak_locs_dict[mi] = peak_locs[mi]
    else:
        peak_locs_dict = copy.deepcopy(peak_locs)

    out = OrderedDict()

    if pix_to_wvl is None:
        pix_to_wvl = scipy.interpolate.interp1d(xx,wl,kind='cubic',bounds_error=False)
    if pix_to_wvl_per_pix is None:
        dwl = np.diff(wl)
        dwl = np.append(dwl,dwl[-1])
        pix_to_wvl_per_pix = scipy.interpolate.interp1d(xx,dwl,kind='cubic',bounds_error=False)

    if continuum_subtract:
        fl_subtracted, _, _ = subtract_Continuum_fromlines(fl,*continuum_subtract_kw)
        fl = fl_subtracted

    for mi in peak_locs_dict.keys():
        loc_this = peak_locs_dict[mi]
        if fitfunc == 'fgauss_const':
            p0 = [loc_this,2.5,1.,0.]
        elif fitfunc == 'fgauss_line':
            p0 = [loc_this,2.5,1.,0.,0.]
        tmp = neidspec.fitting_utils.fitProfile(xx,fl,loc_this,fit_width=8,sigma=None,
                                                func=fitfunc,p0=p0)
        #tmp['centroid_wl'] = interp(tmp['centroid'],xx_pix,xx_test)
        dwl_per_pix = pix_to_wvl_per_pix(tmp['centroid'])
        centroid_pix = tmp['centroid']
        centroid_wl = pix_to_wvl(centroid_pix)[()]
        fwhm_pix = 2.36 * tmp['sigma']
        fwhm_wl = fwhm_pix * dwl_per_pix
        fwhm_vel = fwhm_wl / centroid_wl * 3e8
        peak_counts = tmp['scale_value']

        out1 = OrderedDict()
        out1['fit_output'] = tmp
        out1['centroid_pix'] = centroid_pix
        out1['centroid_wl'] = centroid_wl
        out1['fwhm_pix'] = fwhm_pix
        out1['fwhm_wl'] = fwhm_wl
        out1['snr_peak'] = np.sqrt(peak_counts)
        out1['prec_est'] = 0.4 * fwhm_vel / (np.sqrt(fwhm_pix) * np.sqrt(peak_counts))

        out[mi] = out1
    return(out)

def json_save(filename,data):
    with open(filename,'w') as outfile:
        json.dump(data,outfile)

def json_load(filename,swap_order_mode_strings=True):
    with open(filename,'r') as infile:
        out = json.load(infile)
    if swap_order_mode_strings:
        #print('Doing swap')
        out_copy = OrderedDict()
        out_results = OrderedDict()
        for oi in out['results'].keys():
            out_order = OrderedDict()
            oi_int = int(oi)
            for mi in out['results'][oi].keys():
                mi_int = int(mi)
                out_order[mi_int] = out['results'][oi][mi]
            out_results[oi_int] = out_order
            #print(out_order.keys())
        out_copy['results'] = out_results
        #print(out_results.keys())
        #print(out_copy['results'].keys())
        for ki in out.keys():
            if ki != 'results':
                #print('tripped {}'.format(ki))
                out_copy[ki] = out[ki]
        #print(out_copy['results'].keys())
        out = out_copy
    return(out)

def make_velocity_list(filelist_json):
    dtimes = []
    vals = ['centroid_pix','centroid_wl','fwhm_pix','fwhm_wl','prec_est','snr_peak']
    out = OrderedDict()
    for ki in vals:
        out[ki] = OrderedDict()

    dat0 = json_load(filelist_json[0])
    dtimes.append(astropy.time.Time(dat0['time'],format='isot').to_datetime())
    for ki in vals:
        for oi in dat0['results'].keys():
            out[ki][oi] = OrderedDict()
            for mi in dat0['results'][oi].keys():
                out[ki][oi][mi] = [dat0['results'][oi][mi][ki]]

    for fi in filelist_json[1:]:
        dat = json_load(fi)
        dtimes.append(astropy.time.Time(dat['time'],format='isot').to_datetime())
        for ki in vals:
            for oi in dat['results'].keys():
                for mi in dat0['results'][oi].keys():
                    out[ki][oi][mi].append(dat['results'][oi][mi][ki])

    out['velocities'] = copy.deepcopy(out['centroid_wl'])
    for oi in out['centroid_wl'].keys():
        for mi in out['centroid_wl'][oi].keys():
            centroids_wl = np.array(copy.deepcopy(out['centroid_wl'][oi][mi]))
            overall_center = astropy.stats.biweight_location(centroids_wl,ignore_nan=True)
            vel_shifts = (centroids_wl - overall_center) / overall_center * 3e8
            out['velocities'][oi][mi] = vel_shifts

    return(dtimes,out)

def bugfix_biweight_location(array,**kargs):
    """ Temperory bug fix for biweight_location which returns nan for zero varience array """
    array = array[~np.isnan(array)] # Remove any nans
    if np.any(mad_std(array,**kargs)==0):
        return np.median(array,**kargs)
    else:
        return biweight_location(array,**kargs)

def subtract_Continuum_fromlines(inputspec,refspec=None,thresh_mask=None,thresh_window=21,mask_dilation=2,spline_kind='cubic'):
    """ Returns a smooth continuum subtracted `inputspec` . If `refspec` is provided, it is used to create the mask fo the continuum region.
    """ 
    # Use inputspec for thersholding if refspec is not provided
    if refspec is None:
        refspec = inputspec

    Xaxis = np.arange(len(refspec))

    if thresh_mask is None:
        # Create a mask for the emission lines
        ThresholdMask = np.atleast_2d(refspec) > filters.threshold_local(np.atleast_2d(refspec), thresh_window,offset=0)
        # Dilate the mask
        ThresholdMask = morphology.binary_dilation(ThresholdMask,selem=np.array([[1]*mask_dilation+[1]+[1]*mask_dilation]))[0]
    else:
        ThresholdMask = thresh_mask

    pix_pos_list = []
    continuum_list = []
    for sli in np.ma.clump_unmasked(np.ma.array(refspec,mask=ThresholdMask)):
        pix_pos_list.append(np.mean(Xaxis[sli]))
        continuum_list.append(bugfix_biweight_location(inputspec[sli]))

    Continuum_Func = interpolate.interp1d(pix_pos_list,continuum_list,kind=spline_kind,fill_value='extrapolate')
    Continuum = Continuum_Func(Xaxis)
    outspec = inputspec - Continuum

    return outspec, Continuum, ThresholdMask

