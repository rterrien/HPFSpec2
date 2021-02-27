from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.io
import scipy.interpolate
import scipy.optimize
import os
import ccf
from hpfspec2 import spec_help
from hpfspec2 import rv_utils
from hpfspec2.target import Target
import astropy
import astropy.stats
from hpfspec2 import fitting_utils
from PyAstronomy import pyasl
from collections import OrderedDict
from astropy.modeling.models import Chebyshev1D
import specutils
from numpy.random import default_rng

class HPFSpectrum(object):
    """ An object containing HPF spectral data.
    
    General purpose object for containing and analyzing HPF data for a single data frame.
    
    Hard-coded Attributes (file paths and settings)
    ----------
    path_flat_deblazed : {str}
        Path to a deblazed flat.
    path_flat_blazed : {str}
        Path to a (non-deblazed) flat.
    path_tellmask : {str}
        Path to a telluric mask.
    path_ccf_mask : {str}
        Path to a ccf mask.
    SKY_SCALING_FACTOR : {float}
        Scaling factor between sci and sky fibers.
    """
    
    path_flat_deblazed = "/data1/common/hpfspec_data/hpf/flats/alphabright_fcu_sept18_deblazed.fits"
    path_flat_blazed = "/data1/common/hpfspec_data/hpf/flats/alphabright_fcu_sept18.fits"
    path_tellmask = "/data1/common/hpfspec_data/masks/telluric/telfit_telmask_conv17_thres0.995_with17area.dat"
    path_ccf_mask = "/data1/common/hpfspec_data/masks/ccf/gj699_combined_stellarframe.mas"
    #path_wavelength_solution = "/data1/common/hpfspec_data/hpf/wavelength_solution/LFC_wavecal_scifiber_v2.fits" 
    SKY_SCALING_FACTOR = 0.90 # seems to work well for order 17
    
    def __init__(self,filename,targetname='',deblaze=True,ccf_redshift=True,target_kwargs={},keepsciHDU=False,keepflatHDU=False,
                loadmask=True, model_data=None,verbose=False,hpfspec_data=None,cal=False,rvshift_orders=[5]):
        """ Create the HPF spectrum object.
        
        Instantiate an HPF spectrum object for a single data frame or model spectrum.
        
        Parameters
        ----------
        filename : {str}
            Path to a 1D extracted spectrum data file (using the standard fits format, with SCI/SKY/CAL extensions)
        targetname : {str}, optional
            Simbad-resolvable name of target.
        deblaze : {bool}, optional
            Divide the spectrum by the "blazed" flat to remove the blaze (the default is True)
        ccf_redshift : {bool}, optional
            Use the CCF mask to measure the redshift and define the rest frame of the star (the default is True)
        target_kwargs : {dict}, optional
            Target keywords; use to provide target data in the case where name is not Simbad-able.
        keepsciHDU : {bool}, optional
            Retain the open HDU for the science fits file. (the default is False)
        keepflatHDU : {bool}, optional
            Retain the open HDU for the flat fits file. (the default is False)
        loadmask : {bool}, optional
            Load the mask file. (the default is True)
        model_data : {dict}, optional
            A dictionary containing keys 'w','fl'[,'er'] of model data. Each should be a 28x2048 array.(the default is None)
        """
        self.filename = filename
        self.basename = filename.split(os.sep)[-1]
        self.verbose = verbose
        self.rvshift_orders = rvshift_orders

        if hpfspec_data is not None:
            self.header = hpfspec_data.header
            self.sci_slope = hpfspec_data.f_sci * hpfspec_data.flat_sci / hpfspec_data.exptime  #       self.f_sci = self.hdu[1].data*self.exptime/self.flat_sci
            self.sky_slope = hpfspec_data.f_sky * hpfspec_data.flat_sky / hpfspec_data.exptime
            self.cal_slope = None #hpfspec_data.f_cal * hpfspec_data.flat_cal / hpfspec_data.exptime
            self.sci_variance = (hpfspec_data.e_sci / hpfspec_data.exptime)**2. #np.sqrt(self.hdu[5].data)*self.exptime*self.SKY_SCALING_FACTOR
            self.sky_variance = (hpfspec_data.e_sky / hpfspec_data.exptime / hpfspec_data.SKY_SCALING_FACTOR)**2.
            self.cal_variance = None #(hpfspec_data.e_cal / hpfspec_data.exptime)**2.
            # HPFSpec did not ingest the other wavelength arrays
            self.sci_wave = hpfspec_data.w
            self.sky_wave = None
            self.cal_wave = None
            self.exptime = hpfspec_data.exptime
            self.object = hpfspec_data.object
            self.qprog = hpfspec_data.qprog
            self.jd_midpoint = hpfspec_data.jd_midpoint
            self.target = hpfspec_data.target
            self.bjd = hpfspec_data.bjd
            self.berv = hpfspec_data.berv
            self.w_barycentric_shifted = rv_utils.redshift(hpfspec_data.w,ve=0.,vo=hpfspec_data.berv) # this does not exist but we can create it
            self.flat_header = hpfspec_data.header_flat
            self.flat_sci_slope = hpfspec_data.flat_sci
            self.flat_sky_slope = hpfspec_data.flat_sky #inp_flat[2].data.copy()
            self.flat_cal_slope = None
            self.sci_err = hpfspec_data.e_sci
            self.sky_err = hpfspec_data.e_sky
            self.cal_err = hpfspec_data.e_cal #np.sqrt(self.cal_variance)*self.exptime
            self.sci_and_sky_err = hpfspec_data.e
            self.f_sci = hpfspec_data.f_sci #self.sci_slope / self.flat_sci_slope * self.exptime
            self.f_sky = hpfspec_data.f_sky #self.sky_slope / self.flat_sky_slope * self.exptime * self.SKY_SCALING_FACTOR
            self.f_cal = None #self.cal_slope / self.flat_cal_slope * self.exptime

            self.f_sci_sky = hpfspec_data.f #hpfspec_data.f_sci - hpfspec_data.f_sky #self.f_sci - self.f_sky

            self.sn18 = hpfspec_data.sn18

            if deblaze:
                self.f_sci_debl = hpfspec_data.f_sci_debl #self.f_sci / hdu[1].data
                self.f_sky_debl = hpfspec_data.f_sky_debl #self.f_sky / hdu[2].data
                self.f_cal_debl = None

                self.e_sci_debl = hpfspec_data.e_debl
                self.e_sky_debl = None #self.sky_err / hdu[2].data
                self.e_cal_debl = None #self.cal_err / hdu[3].data

                #self.sci_and_sky_err = np.sqrt(self.sci_variance + self.sky_variance)*self.exptime
                self.e_sci_sky_debl = hpfspec_data.e_debl #np.sqrt(self.e_sci_debl**2. + self.e_sky_debl**2.)

                self.f_sci_sky_debl = hpfspec_data.f_debl #self.f_sci_debl - self.f_sky_debl * self.SKY_SCALING_FACTOR
                # if norm_percentile_per_order is not None:
                #     for i in range(28):
                #         norm_val = np.nanpercentile(self.f_sci_sky_debl[i],norm_percentile_per_order)
                #         self.f_sci_sky_debl[i] = self.f_sci_sky_debl[i] / norm_val
                #         self.e_sci_sky_debl[i] = self.e_sci_sky_debl[i] / norm_val
                
                self.f_debl = hpfspec_data.f_debl
                self.e_debl = hpfspec_data.e_debl

            if loadmask:
                self.M = hpfspec_data.M

            if ccf_redshift:
                self.rv = hpfspec_data.rv
                self.w_shifted = hpfspec_data.w_shifted

        else:
            if model_data is None:
                # Read science frame; copy all to eliminate references to HDU and allow it to close/garbage collect.
                inp = astropy.io.fits.open(filename)
                self.header = inp[0].header.copy()
                self.sci_slope = inp[1].data.copy()
                self.sky_slope = inp[2].data.copy()
                self.cal_slope = inp[3].data.copy()
                self.sci_variance = inp[4].data.copy()
                self.sky_variance = inp[5].data.copy()
                self.cal_variance = inp[6].data.copy()
                self.sci_wave = inp[7].data.copy()
                self.sky_wave = inp[8].data.copy()
                self.cal_wave = inp[9].data.copy()
                if keepsciHDU:
                    self.hdu = inp
                else:
                    inp.close()

                # Pull out some useful parts of the header
                self.exptime = self.header["EXPLNDR"]
                self.object = self.header["OBJECT"]
                try: 
                    self.qprog = self.header["QPROG"]
                except Exception:# as e: 
                    self.qprog = np.nan
                midpoint_keywords = ['JD_FW{}'.format(i) for i in range(28)]
                self.jd_midpoint = np.median(np.array([self.header[i] for i in midpoint_keywords]))

                # Identify/process target information
                if not cal:
                    if targetname == '':
                        targetname = self.object
                    self.target = Target(targetname,**target_kwargs)
                    self.bjd, self.berv = self.target.calc_barycentric_velocity(self.jd_midpoint,'McDonald Observatory')

                    # use the BJD and known barycentric velocity to define a wavelength array at rest in wrt solar system barycenter
                    self.w_barycentric_shifted = rv_utils.redshift(self.sci_wave,ve=0.,vo=self.berv)
                
                # Read Flat
                inp_flat = astropy.io.fits.open(self.path_flat_deblazed)
                self.flat_header = inp_flat[0].header.copy()
                self.flat_sci_slope = inp_flat[1].data.copy()
                self.flat_sky_slope = inp_flat[2].data.copy()
                self.flat_cal_slope = inp_flat[3].data.copy()
                if keepflatHDU:
                    self.flathdu = inp_flat
                else:
                    inp_flat.close()

                # Turn slopes into fluxes
                self.sci_err = np.sqrt(self.sci_variance)*self.exptime
                self.sky_err = np.sqrt(self.sky_variance)*self.exptime
                self.cal_err = np.sqrt(self.cal_variance)*self.exptime

                self.sci_and_sky_err = np.sqrt(self.sci_variance + self.sky_variance)*self.exptime

                self.f_sci = self.sci_slope / self.flat_sci_slope * self.exptime
                self.f_sky = self.sky_slope / self.flat_sky_slope * self.exptime #* self.SKY_SCALING_FACTOR
                self.f_cal = self.cal_slope / self.flat_cal_slope * self.exptime

                self.f_sci_sky = self.f_sci - self.f_sky

                self.sn18 = self.snr_order_median(18)

                if deblaze:
                    self.deblaze(norm_percentile_per_order=80.)
                    if self.verbose:
                        print('Spectrum Deblazed')
            else:
                # Read in model data, making sure the formatting is correct
                assert type(model_data) == dict
                assert 'fl' in model_data.keys()
                assert 'w' in model_data.keys()
                assert np.shape(model_data['w']) == (28,2048)
                assert np.shape(model_data['fl']) == (28,2048)

                # ingest model data; use dummy values where needed.
                self.exptime = np.NaN
                self.object = 'ModelSpectrum'
                self.qprog = np.NaN
                self.jd_midpoint = np.NaN

                fl = model_data['fl']
                w = model_data['w']
                # If no error provided, assume fl in counts and Poisson noise
                if 'er' in model_data.keys():
                    er = model_data['er']
                else:
                    er = np.sqrt(fl)

                # normalize the data; this will be different from de-blazing
                fl_norm = fl.copy()
                er_norm = er.copy()
                for oi in range(28):
                    fl_norm[oi,:] = fl[oi,:] / np.nanmax(fl[oi,:])
                    er_norm[oi,:] = er[oi,:] / np.nanmax(fl[oi,:])

                # f_sci, f_sci_sky hold un-normalized spectra
                # f_sci_sky_debl holds normalized spectra
                self.sci_slope = fl
                self.sci_variance = er**2.
                self.f_sci = fl_norm
                self.sci_err = er_norm
                self.f_sci_sky = fl_norm
                self.f_sci_sky_debl = fl_norm
                self.w = w
                self.w_barycentric_shifted = w
                self.sci_wave = w
                self.target = None
                self.bjd = np.NaN
                self.berv = 0.
                self.rv = 0.

            if loadmask:
                self.load_mask()
                if self.verbose:
                    print('CCF Mask loaded')
            
            #self.rv = 0.
            if ccf_redshift:
                print('Barycentric shifting')
                rabs,rabss = self.rvabs_orders(orders=self.rvshift_orders)
                self.rv = rabs #np.median(rabs)
                if self.verbose:
                    print(self.rv,rabs,rabss)
                self.redshift(rv=self.rv,berv=self.berv)
                # this function creates the w_shifted array


    def __repr__(self):
        return 'HPFSpec({},sn18={:0.1f})'.format(self.object,self.sn18)

    def snr_order_median(self,oi,):
        """Estimate Signal-to-noise of order
        
        Calculate median ratio of signal and sqrt(variance) in SCI-SKY spectrum.
        
        Parameters
        ----------
        oi : {int}
            Order index 0-27
        """
        return(np.nanmedian(self.f_sci_sky[oi] / self.sci_and_sky_err[oi]))

    def get_telluric_mask(self,w=None,o=None):
        """
        Return telluric mask interpolated onto a given grid.
        
        INPUT:
            w - wavelength grid to interpolate on
            o - order
            
        OUTPUT: Mask 
        
        EXAMPLE:
        """
        if w is None: 
            w = self.w
        mask = np.genfromtxt(self.path_tellmask)
        m = scipy.interpolate.interp1d(mask[:,0],mask[:,1])(w) > 0.01
        if o is None:
            return m
        else:
            m[o]

    def load_mask(self,maskpath=None):
        """Load a mask
        
        Load into this object a ccf.mask object containing left/right points and weights.
        
        Parameters
        ----------
        maskpath : {str}, optional
            Path to a ccf.mask file (defaults to path given in the object parameter)
        """
        if maskpath is not None:
            self.path_ccf_mask = maskpath
        try:
            self.M = ccf.mask.Mask(self.path_ccf_mask)
        except FileNotFoundError as e:
            print('Mask file not found',e)
            raise(e)

    def calculate_ccf_order(self,velocities,oi,fl=None,w=None,M=None):
        """ Caluclate the mask-based CCF for a single order.
        
        The mask based-CCF for a single order, using the currently-loaded mask.
        Defaults to using the non-deblazed SCI-SKY spectrum and barycentric-rest frame
        wavelengths.

        Optionally, different flux, wavelength arrays, or mask can be provided.
        Optional flux/wavel should be provided as a 1D array, and oi is ignored.
        
        Parameters
        ----------
        velocities : {ndarray}
            Array of velocity shifts over which to calculate the CCF.
        oi : {int}
            HPF order index
        fl : {ndarray}, optional
            1D array of fluxes for a single order (if provided, oi is ignored)
        w : {ndarray}, optional
            1D array of wavelengths (ang) for a single order (if provided, oi is ignored)
        M : {ccf.mask}, optional
            CCF mask
        """
        if fl is None:
            fl = self.f_sci_sky[oi]
        if w is None:
            w = self.w_barycentric_shifted[oi]
        if M is None:
            M = self.M
        out = ccf.ccf.calculate_ccf(w,
                                    fl,
                                    velocities,
                                    M.wi,
                                    M.wf,
                                    M.weight,
                                    0.)
        return(out)

    def calculate_ccf_orders(self,velocities,fl=None,w=None,M=None,orders=[3,4,5,6,14,15,16,17,18]):
        """ Calculate CCF for several orders
        
        Convenience function for calculating many CCFs at once.
        Defaults to using the non-deblazed SCI-SKY spectrum and barycentric-rest frame
        wavelengths.

        Optionally, different flux, wavelength arrays, or mask can be provided.
        Optional flux/wavel should be provided as a 2D arrays.
        
        Parameters
        ----------
        velocities : {ndarray}
            Array of velocity shifts over which to calculate the CCF.
        oi : {int}
            HPF order index
        fl : {ndarray}, optional
            2D array of fluxes (28 x 2048)
        w : {ndarray}, optional
            2D array of wavelengths (ang) (28 x 2048)
        M : {ccf.mask}, optional
            CCF mask
        orders : {list}, optional
            Orders to calculate the CCF on (the default is [3,4,5,6,14,15,16,17,18])
        """
        if fl is None:
            fl = self.f_sci_sky
        if w is None:
            w = self.w_barycentric_shifted
        if M is None:
            M = self.M
        out = np.full((len(orders),len(velocities)),np.nan)
        for ni,oi in enumerate(orders):
            out[ni,:] = self.calculate_ccf_order(velocities,oi,fl=fl[oi],w=w[oi],M=M)
        return(out)

    def rvabs_order(self,oi,velocities_1=np.linspace(-125,125,1501),width2=8,npoints2=61,fl=None,
                    w=None,M=None,debug=False,fit_width=5):
        """ Calculate the absolute RV for a single order.
        
        Use the CCF mask to calculate the absolute RV shift using a two-step CCF fit.
        Defaults to using non-deblazed SCI-SKY spectrum and barycentric-rest wavelengths.
        
        Parameters
        ----------
        oi : {int}
            Order to measure.
        velocities_1 : {ndarray}, optional
            First list of velocities [km/s] to calculate CCF (the default is np.linspace(-125,125,1501))
        width2 : {float}, optional
            Width of second CCF calculation [km/s] (the default is 8)
        npoints2 : {int}, optional
            Number of points to use in second CCF (the default is 61)
        fl : {ndarray}, optional
            1D array of fluxes (if given, oi is ignored)
        w : {ndarray}, optional
            1D array of wavelengths [ang] (if given, oi is ignored)
        M : {ccf.mask}, optional
            ccf.mask object
        debug : {bool}, optional
            Return full diagnostic output
        fit_width : {float}, optional
            Width of CCF fit in second-round CCF [km/s]
        """
        if fl is None:
            fl = self.f_sci_sky[oi]
        if w is None:
            w = self.w_barycentric_shifted[oi]
        if M is None:
            M = self.M

        # First round CCF, normalized for fitting stability
        out1 = self.calculate_ccf_order(velocities_1,oi,fl=fl,w=w,M=M)
        out1 = out1/np.nanmax(out1)
        # Estimate fit center
        imin = np.argmin(out1)
        vmin = velocities_1[imin]
        # Generate velocities needed around center estimate from CCF 1 and caluclate CCF 2
        velocities_2 = np.linspace(vmin-width2,vmin+width2,npoints2)
        out2 = self.calculate_ccf_order(velocities_2,oi,fl=fl,w=w,M=M)
        out2 = out2/np.nanmax(out2)

        # Find center and fit
        icenter = int(len(out2)/2)
        iwidth = fit_width / np.nanmedian(np.diff(velocities_2))
        fit = fitting_utils.fitProfile(velocities_2,out2,icenter,iwidth)
        if debug:
            full_out = {'ccf1':out1,'vel1':velocities_1,'ccf2':out2,'vel2':velocities_2,'fit':fit}
            return(full_out)
        else:
            return(fit['centroid'])

    def rvabs_orders(self,velocities_1=np.linspace(-125,125,1501),width2=8,npoints2=61,fl=None,w=None,M=None,
                     orders=[4,5,14,15,16,17],debug=False,fit_width=5.):
        """ Calculate absolute RV for several orders
        
        Convenience function for calculating RVs for many orders at once. Wraps rvabs_order.
        
        Parameters
        ----------
        oi : {int}
            Order to measure.
        velocities_1 : {ndarray}, optional
            First list of velocities [km/s] to calculate CCF (the default is np.linspace(-125,125,1501))
        width2 : {float}, optional
            Width of second CCF calculation [km/s] (the default is 8)
        npoints2 : {int}, optional
            Number of points to use in second CCF (the default is 61)
        fl : {ndarray}, optional
            1D array of fluxes (if given, oi is ignored)
        w : {ndarray}, optional
            1D array of wavelengths [ang] (if given, oi is ignored)
        M : {ccf.mask}, optional
            ccf.mask object
        debug : {bool}, optional
            Return full diagnostic output
        fit_width : {float}, optional
            Width of CCF fit in second-round CCF [km/s]
        orders : {list}, optional
            List of orders to do calculation on (the default is [4,5,14,15,16,17])
        """
        if fl is None:
            fl = self.f_sci_sky
        if w is None:
            w = self.w_barycentric_shifted
        if M is None:
            M = self.M
        out_velocities = np.full((len(orders)),np.nan)
        out_full_fits = []
        for ni,oi in enumerate(orders):
            out1 = self.rvabs_order(oi,velocities_1=velocities_1,width2=width2,npoints2=npoints2,fl=fl[oi],w=w[oi],M=M,debug=True)
            out_velocities[ni] = out1['fit']['centroid']
            out_full_fits.append(out1)
        # Calculate center using robust location
        biweight_mean_val = astropy.stats.biweight.biweight_location(out_velocities)
        if debug:
            out_all = {}
            out_all['full_fits'] = out_full_fits
            out_all['biweight_mean_val'] = biweight_mean_val
            return(out_all)
        else:
            return(biweight_mean_val,out_velocities)   

    def ccfwidth_order(self,oi,velocities=np.linspace(-30,30,100),fl=None,w=None,M=None,debug=False,fitwidth=5.,
                       fitfunc='fgauss_from_1'):
        """ Calculate the CCF and CCF width for an order.
        
        Use a mask-based CCF and fitting function to measure the CCF width. Useful for vsini.
        Defaults to using non-deblazed SCI-SKY spectrum and stellar rest frame wavelengths.
        
        Parameters
        ----------
        oi : {int}
            Order index
        velocities : {ndarray}, optional
            Array of velocities on which to calculate CCF. (the default is np.linspace(-30,30,100))
        fl : {ndarray}, optional
            1D array of fluxes. If provided, oi is ignored
        w : {[type]}, optional
            1D array of wavelengths [ang]. If provided, oi is ignored
        M : {ccf.mask}, optional
            CCF mask
        debug : {bool}, optional
            Provide full output. (the default is False)
        fitwidth : {float}, optional
            Width to use for CCF fit [km/s] (the default is 5.)
        fitfunc : {str}, optional
            Argument to fitting_utils.fitProfile, function to use for fitting 
            (the default is 'fgauss_from_1', which performs decently for these CCFs)
        """
        if fl is None:
            fl = self.f_sci_sky[oi]
        if w is None:
            w = self.w_shifted[oi]
        if M is None:
            M = self.M
        # Calculate the CCF and normalize for fitting
        out1 = self.calculate_ccf_order(velocities,oi,fl=fl,w=w,M=M)
        out1 = out1/np.nanmax(out1)
        # Estimate CCF center and fit
        icenter = int(len(out1)/2)
        iwidth = fitwidth / np.nanmedian(np.diff(velocities))
        fit = fitting_utils.fitProfile(velocities,out1,icenter,iwidth,func=fitfunc)
        if debug:
            full_out = {'ccf1':out1,'vel1':velocities,'fit':fit}
            return(full_out)
        else:
            return(fit['sigma'])

    def resample_and_broaden_order(self,oi,vsini=0.,eps=.6,fl=None,w=None,diag=False,trim_nan=True,upsample_factor=1.):
        """ Resample and vsini broaden an order.
        
        Use the standard vsini kernel to broaden an HPF spectrum. Resampling onto an even wavelength grid is required.
        Defaults to using non-deblazed SCI-SKY spectrum and stellar rest frame wavelengths.
        
        Parameters
        ----------
        oi : {int}
            Order index
        vsini : {float}, optional
            vsini value for broadening kernel [km/s] (the default is 0. - no broadening)
            Projected rotational velocity along the line-of-sight.
        eps : {float}, optional
            numeric scalar giving the limb-darkening coefficient, 
            default = 0.6 which is typical for  photospheric lines.    The
            specific intensity I at any angle theta from the specific intensity
            Icen at the center of the disk is given by:
            I = Icen*(1-epsilon*(1-cos(theta))
        fl : {ndarray}, optional
            1D array of fluxes. If provided, oi is ignored
        w : {ndarray}, optional
            1D array of wavelengths [ang]. If provided, oi is ignored
        diag : {bool}, optional
            Provide full output (the default is False)
        trim_nan : {bool}, optional
            Mask NaN values before resampling (the default is True)
        upsample_factor : {float}, optional
            Scale up (or down) the spectral sampling (the default is 1., which corresponds to no change)
        """
        if fl is None:
            fl = self.f_sci_sky[oi]
        if w is None:
            w = self.w_shifted[oi]
        if trim_nan:
            mask = np.ma.masked_invalid(fl)
            w = w[~mask.mask]
            fl = fl[~mask.mask]
        # Resample the spectrum
        w_resampled, fl_resampled = spec_help.resample_to_median_sampling(w,fl,upsample_factor=upsample_factor)
        if vsini > 0.:
            # PyAstronomy.pyasl.rotBroad() cannot handle NaNs.
            # if the SpectRes resampling is used there is a NaN in the last pixel... this is a bug
            # trim this manually for now. RCT 9-12-20
            if trim_nan:
                w_resampled = w_resampled[:-1]
                fl_resampled = fl_resampled[:-1]
            out = pyasl.rotBroad(w_resampled, fl_resampled, eps, vsini)
        else:
            out = fl_resampled

        if diag:
            full_out = {'w_in':w, 'fl_in':fl, 'w_resampled':w_resampled, 'fl_resampled':fl_resampled, 'fl_broadened':out}
            return(full_out)
        else:
            return(w_resampled,out)
            
    
    def deblaze(self,keepHDU=False,norm_percentile_per_order=None):
        """ Deblaze the spectrum.
        
        Normalize/flatten the spectrum by dividing by the "blazed" flat.
        This should remove the overall throughput profile of HPF with an order.
        Results stored in f_{sci|sky|cal}_debl
        Normalization only applies to SCI-SKY, result is in f_sci_sky_debl and f_debl (backwards-compatibility)
        
        Parameters
        ----------
        keepHDU : {bool}, optional
            Keep the HDU open for the flat file used. (the default is False)
        norm_percentile_per_order : {float}, optional
            Normalize each order to this percentile (the default is 80)
        """
        hdu = astropy.io.fits.open(self.path_flat_blazed)
        self.f_sci_debl = self.f_sci / hdu[1].data
        self.f_sky_debl = self.f_sky / hdu[2].data
        self.f_cal_debl = self.f_cal / hdu[3].data

        self.e_sci_debl = self.sci_err / hdu[1].data
        self.e_sky_debl = self.sky_err / hdu[2].data
        self.e_cal_debl = self.cal_err / hdu[3].data

        #self.sci_and_sky_err = np.sqrt(self.sci_variance + self.sky_variance)*self.exptime
        self.e_sci_sky_debl = np.sqrt(self.e_sci_debl**2. + self.e_sky_debl**2.)

        self.f_sci_sky_debl = self.f_sci_debl - self.f_sky_debl * self.SKY_SCALING_FACTOR
        if norm_percentile_per_order is not None:
            for i in range(28):
                norm_val = np.nanpercentile(self.f_sci_sky_debl[i],norm_percentile_per_order)
                self.f_sci_sky_debl[i] = self.f_sci_sky_debl[i] / norm_val
                self.e_sci_sky_debl[i] = self.e_sci_sky_debl[i] / norm_val
        
        self.f_debl = self.f_sci_sky_debl
        self.e_debl = self.e_sci_sky_debl

        #self.e_debl = self.f_debl/self.sn  #### RCT SEP 2020 need to check why this is here
            
    def redshift(self,berv=None,rv=None):
        """ Redshift the science spectrum
        
        Apply a redshift to the science spectrum. Accounts for observer and target motion.
        If rv is not provided, use RV provided in target file.
        
        Parameters
        ----------
        berv : {float}, optional
            Barycentric velocity (i.e. velocity of observer) [km/s] - defaults to using known berv
        rv : {float}, optional
            Target velocity [km/s]. - defaults to using target file RV
        """
        if berv is None:
            berv = self.berv
        if rv is None:
            print('Warning: Using target file estimate for target RV')
            rv = self.target.rv
        print('berv={},rv={}'.format(berv,rv))
        self.w_shifted = rv_utils.redshift(self.sci_wave,vo=berv,ve=rv)
        self.rv = rv

    def find_peaks_order(self,oi,fl=None,w=None,prominence=0.1,width=(0,8),
                         pixel_to_wl_interpolation_kind='cubic',fill_value=np.nan,
                         fit_width_kms=None):
        """ Find peaks in a spectral order
        
        Use the scipy.signal.find_peaks routine to locate lines in a spectral order.
        Defaults to using f_sci_sky_debl and stellar rest frame wavelengths.
        Presently the precision is only pixel-level, so interpolation is overkill.
        
        Parameters
        ----------
        oi : {int}
            Order index
        fl : {ndarray}, optional
            1D array of fluxes. if provided, oi is ignored
        w : {ndarray}, optional
            1D array of wavelengths [ang]. if provided, oi is ignored
        prominence : {float}, optional
            Height above surroundings. Argument to scipy.signal.find_peaks (the default is 0.1)
        width : {tuple}, optional
            Bounds on peak width. Argument to scipy.signal.find_peaks (the default is (0,8))
        pixel_to_wl_interpolation_kind : {str}, optional
            Interpolation to use for converting pixels to wavelength (the default is 'cubic')
        fill_value : {number}, optional
            Fill value in interpolation (the default is np.nan)
        fit_width_kms : {float}, optional
            Fit the lines to find a more precise centroid. [km/s]
            Skip this by not providing a number.
        """
        if fl is None:
            fl = self.f_sci_sky_debl[oi]
        if w is None:
            w = self.w_shifted[oi]
        # Find pixel centers and interpolate to wavelength values
        pixel_peaks = scipy.signal.find_peaks(-fl,prominence=prominence,width=width)[0] # propertes in [1]
        xx = np.arange(2048)
        wl_peaks = scipy.interpolate.interp1d(xx,w,kind=pixel_to_wl_interpolation_kind,fill_value=fill_value,bounds_error=False)(pixel_peaks)

        # If fit is not requested (i.e. fit_width_kms is None), just return pixel centers
        if fit_width_kms is None:
            return(wl_peaks)
        
        # Fit the centers using simple assumptions
        fitted_centers = []
        dwl_pix = np.nanmedian(np.diff(w))
        dv_pix = dwl_pix / np.nanmedian(w) * 3e5
        fit_width_pix = fit_width_kms / dv_pix
        for pi,wi in zip(pixel_peaks,wl_peaks):
            fitout = fitting_utils.fitProfile(w,fl,pi,fit_width_pix,func='fgauss_const',p0=[wi,-0.1,1.,1.])
            fitted_centers.append(fitout['centroid'])
        fitted_centers = np.array(fitted_centers)
        return(fitted_centers)

    def fit_peaks_order(self,oi,wl_peaks,fl=None,w=None,
                         pixel_to_wl_interpolation_kind='cubic',fill_value=np.nan,
                         fit_width_kms=8.):
        """ Fit peaks in a spectral order
        
        If you already have peaks roughly located in wavelength, use this routine to
        fit their locations more precisely.
        
        Parameters
        ----------
        oi : {int}
            Order index
        wl_peaks : [list]
            List of peak wavelengths in angstroms
        fl : {ndarray}, optional
            1D array of fluxes. if provided, oi is ignored
        w : {ndarray}, optional
            1D array of wavelengths [ang]. if provided, oi is ignored
        pixel_to_wl_interpolation_kind : {str}, optional
            Interpolation to use for converting pixels to wavelength (the default is 'cubic')
        fill_value : {number}, optional
            Fill value in interpolation (the default is np.nan)
        fit_width_kms : {float}, optional
            Fit the lines to find a more precise centroid. [km/s]
            Skip this by not providing a number.
        """
        if fl is None:
            fl = self.f_sci_sky_debl[oi]
        if w is None:
            w = self.w_shifted[oi]
        # Find pixel centers and interpolate to wavelength values
        #pixel_peaks = scipy.signal.find_peaks(-fl,prominence=prominence,width=width)[0] # propertes in [1]
        xx = np.arange(2048)
        pixel_peaks = scipy.interpolate.interp1d(w,xx,kind=pixel_to_wl_interpolation_kind,fill_value=fill_value,bounds_error=False)(wl_peaks)
        #wl_peaks = scipy.interpolate.interp1d(xx,w,kind=pixel_to_wl_interpolation_kind,fill_value=fill_value,bounds_error=False)(pixel_peaks)

        # If fit is not requested (i.e. fit_width_kms is None), just return pixel centers
        if fit_width_kms is None:
            return(wl_peaks)
        
        # Fit the centers using simple assumptions
        fitted_centers = []
        dwl_pix = np.nanmedian(np.diff(w))
        dv_pix = dwl_pix / np.nanmedian(w) * 3e5
        fit_width_pix = fit_width_kms / dv_pix
        for pi,wi in zip(pixel_peaks,wl_peaks):
            fitout = fitting_utils.fitProfile(w,fl,pi,fit_width_pix,func='fgauss_const',p0=[wi,-0.1,1.,1.])
            fitted_centers.append(fitout['centroid'])
        fitted_centers = np.array(fitted_centers)
        return(fitted_centers)
        
    def plot_order(self,o,deblazed=False):
        """
        Plot spectrum
        """
        mask = np.genfromtxt(self.path_tellmask)
        m = self.get_telluric_mask()

        fig, ax = plt.subplots(dpi=200)
        if deblazed:
            self.deblaze()
            ax.plot(self.w[o],self.f_debl[o])
            ax.fill_between(mask[:,0],mask[:,1],color='red',alpha=0.1)
            ax.plot(self.w[o][m[o]],self.f_debl[o][m[o]],lw=0,marker='.',markersize=2,color='red')
        else:
            ax.plot(self.w[o],self.f[o])
            ax.plot(self.w[o],self.f_sci[o],alpha=0.3)
            ax.fill_between(mask[:,0],mask[:,1],color='red',alpha=0.1)
            ax.plot(self.w[o][m[o]],self.f_debl[o][m[o]],lw=0,marker='.',markersize=2,color='red')

        #utils.ax_apply_settings(ax)
        ax.minorticks_on()
        ax.tick_params(pad=3,labelsize=12)
        ax.grid(lw=0.5,alpha=0.5)
        ax.set_title('{}, {}, order={}, SN18={:0.2f}\nBJD={}, BERV={:0.5f}km/s'.format(self.object,
                                                                                       self.basename,o,self.sn18,self.bjd,self.berv))
        ax.set_xlabel('Wavelength [A]')
        ax.set_ylabel('Flux')
        ax.set_xlim(np.nanmin(self.w[o]),np.nanmax(self.w[o]))

    def make_spec1ds(self,fl=None,w=None,e=None,note='',renormalize=False,renormalize_median_window=51,
                     renormalize_model=Chebyshev1D(1),renormalize_percentile_scaling=95):
        #w_ang,fl_counts,e_counts,median_window=51,model=Chebyshev1D(1),percentile_offset=98):
        """ Convert spectrum into a set of specutils.spec1d objects
        
        Translate the HPF spectrum (by default the stellar spectrum) into a specutils.spec1d object
        to take advantage of the analysis tools available.
        
        Parameters
        ----------
        fl : {ndarray}, optional
            28x2048 array of fluxes
        w : {ndarray}, optional
            28x2048 array of wavelengths [ang]
        e : {ndarray}, optional
            28x2048 array of errors
        """
        if fl is None:
            fl = self.f_sci_sky_debl
            note = note+'f_sci_sky_debl,'
        if w is None:
            w = self.w_shifted
            note = note + 'w_shifted,'
        if e is None:
            e = self.e_sci_sky_debl
            note = note + 'e_sci_sky_debl'
        norders, npix = np.shape(fl)
        assert np.shape(fl) == np.shape(w)
        assert np.shape(w) == np.shape(e)
        out = OrderedDict()
        for oi in range(norders):
            w_use = w[oi]
            e_use = e[oi]
            fl_use = fl[oi]
            mask = np.ma.masked_invalid(fl_use)
            w_use = w_use[~mask.mask] 
            fl_use = fl_use[~mask.mask]
            e_use = e_use[~mask.mask]
            if renormalize:
                fl_use, e_use = spec_help.specutils_continuum_normalize(w_use,fl_use,e_use,
                                                                        median_window=renormalize_median_window,
                                                                        model=renormalize_model,
                                                                        percentile_scaling=renormalize_percentile_scaling)
                #print(fl_use.unit,e_use.unit)
            out[oi] = spec_help.convert_to_spec1d(w_use,
                                                  fl_use,
                                                  e_use,
                                                  resample=False,
                                                  resample_kind='FluxConservingSpectRes',
                                                  resample_fill_value=np.nan,
                                                  resample_upsample_factor=1.)
        self.spec1Ddict = out
        self.spec1Dnote = note

    def specutils_measure_ew(self,feature,diag=False):
        assert self.spec1Ddict is not None
        lower = feature.lower
        upper = feature.upper
        for oi in self.spec1Ddict.keys():
            wls = self.spec1Ddict[oi].spectral_axis
            omin, omax = np.nanmin(wls), np.nanmax(wls)
            if (lower > omin) and (upper < omax):
                o_use = oi
                #print('Using oi {}'.format(oi))
                break
        if o_use is None:
            raise ValueError('Feature not entirely in orders')
        spec1d = self.spec1Ddict[oi]
        out = specutils.analysis.equivalent_width(spec1d,regions=feature)
        if not diag:
            return(out)
        else:
            wls = self.spec1Ddict[o_use].spectral_axis
            inds = np.nonzero( (wls > lower) & (wls < upper) )
            outdict = {'ew':out,
                       'order':o_use,
                       'inds':inds}
            return(outdict)

    def measure_ew(self,lower=None,upper=None,feature=None,w=None,fl=None,diag=False,const_continuum_regions=[],
                   slope_continuum_regions=[]):
        if ((lower is None) or (upper is None)) and (feature is None):
            raise ValueError
        if feature is not None:
            lower = feature.lower.value
            upper = feature.upper.value

        if fl is None:
            fl = self.f_sci_sky_debl
        if w is None:
            w = self.w_shifted
        norders, npix = np.shape(fl)

        for oi in range(norders):
            wls = w[oi]
            omin, omax = np.nanmin(wls), np.nanmax(wls)
            if (lower > omin) and (upper < omax):
                o_use = oi
                #print('Using oi {}'.format(oi))
                break
        if o_use is None:
            raise ValueError('Feature not entirely in orders')
        fl_use = fl[o_use]
        wl_use = w[o_use]

        if len(slope_continuum_regions) > 0:
            print('Renormalizing to slope')
            ci_use = []
            for clim in slope_continuum_regions:
                lower_lim = clim[0]
                upper_lim = clim[1]
                tmp_i = np.nonzero((wl_use >= lower_lim) & (wl_use <= upper_lim))[0]
                if len(tmp_i) > 0:
                    for j in tmp_i:
                        ci_use.append(j)
                else:
                    print('No suitable pixels found for {:.2f} to {:.2f}'.format(lower_lim,upper_lim))
            ww_fit = wl_use[ci_use]
            ff_fit = fl_use[ci_use]
            pp = np.polyfit(ww_fit,ff_fit,1)
            norm = np.polyval(pp,wl_use)
            fl_use = fl_use / norm
        elif len(const_continuum_regions) > 0:
            print('Renormalizing to constant value')
            ci_use = []
            for clim in const_continuum_regions:
                lower_lim = clim[0]
                upper_lim = clim[1]
                tmp_i = np.nonzero((wl_use >= lower_lim) & (wl_use <= upper_lim))[0]
                if len(tmp_i) > 0:
                    for j in tmp_i:
                        ci_use.append(j)
                else:
                    print('No suitable pixels found for {:.2f} to {:.2f}'.format(lower_lim,upper_lim))
            if len(ci_use) > 0:
                new_norm = astropy.stats.biweight_location(fl_use[ci_use])
                print('New norm: {:.3}'.format(new_norm))
                fl_use = fl_use / new_norm
            else:
                print('No renorm pixels found, skipping')
                


        out = spec_help.calculate_ew(wl_use,fl_use,lower,upper)

        if not diag:
            return(out)
        else:
            inds = np.nonzero( (wl_use > lower) & (wl_use < upper) )
            outdict = {'ew':out,
                       'order':o_use,
                       'inds':inds}
            return(outdict)

    def which_order(self,wavelength):
        ''' Say which order a wavelength falls in
        
        Parameters
        ----------
        wavelength : float
            wavelength to query
        '''
        wmins = np.nanmin(self.w_shifted,axis=1)
        wmaxs = np.nanmax(self.w_shifted,axis=1)
        for oi in range(28):
            if (wavelength > wmins[oi]) and (wavelength < wmaxs[oi]):
                return(oi)
        return(None)

    def jitter_spectrum(self):
        ''' Jitter the spectrum by the given variance
        
        Jitter each pixel's slope value by the pipeline-reported variance.
        Make sure to do this on a copy of the original spectrum - the 
        flux values will be permanently changed for this object.

        The object then re-does the ingestion of the spectrum (flattening, deblazing)
        '''
        # self.header = inp[0].header.copy()
        # self.sci_slope = inp[1].data.copy()
        # self.sky_slope = inp[2].data.copy()
        # self.cal_slope = inp[3].data.copy()
        # self.sci_variance = inp[4].data.copy()
        # self.sky_variance = inp[5].data.copy()
        # self.cal_variance = inp[6].data.copy()
        # self.sci_wave = inp[7].data.copy()
        # self.sky_wave = inp[8].data.copy()
        # self.cal_wave = inp[9].data.copy()
        # if keepsciHDU:
        #     self.hdu = inp
        # else:
        #     inp.close()
        shape = np.shape(self.sci_slope)
        jitter_sci = default_rng().normal(0,np.sqrt(self.sci_variance),shape)
        jitter_cal = default_rng().normal(0,np.sqrt(self.cal_variance),shape)
        jitter_sky = default_rng().normal(0,np.sqrt(self.sky_variance),shape)
        self.sci_slope = self.sci_slope + jitter_sci
        self.cal_slope = self.cal_slope + jitter_cal
        self.sky_slope = self.sky_slope + jitter_sky

        # Turn slopes into fluxes
        self.sci_err = np.sqrt(self.sci_variance)*self.exptime
        self.sky_err = np.sqrt(self.sky_variance)*self.exptime
        self.cal_err = np.sqrt(self.cal_variance)*self.exptime

        self.sci_and_sky_err = np.sqrt(self.sci_variance + self.sky_variance)*self.exptime

        self.f_sci = self.sci_slope / self.flat_sci_slope * self.exptime
        self.f_sky = self.sky_slope / self.flat_sky_slope * self.exptime #* self.SKY_SCALING_FACTOR
        self.f_cal = self.cal_slope / self.flat_cal_slope * self.exptime

        self.f_sci_sky = self.f_sci - self.f_sky

        self.sn18 = self.snr_order_median(18)


        self.deblaze(norm_percentile_per_order=80.)


        print('Spectrum Jittered (automatically deblazed)')



class HPFSpecList(object):

    def __init__(self,splist=None,filelist=None):
        if splist is not None:
            self.splist = splist
        else:
            self.splist = [HPFSpectrum(i) for i in filelist]

    def combine_specs(self,f_which='f_sci_sky_debl',w_which=None,combine_type='biweight',sigma_clip=5.):
        """ Combine spectra in a list
        
        Resample and combine spectra. This routine does not do anything clever to maintain resolution,
        so check that outputs are not dependent on, e.g., how much barycentric sampling there is.

        Output is stored in combined_spec
        
        Parameters
        ----------
        f_which : {str}, optional
            Which flux array to combine (the default is 'f_sci_sky_debl')
        w_which : {str}, optional
            Which wavelength array to use (the default is w_shifted - i.e. the stellar rest frame)
        combine_type : {str}, optional
            How to combine the spectra after resampling (the default is 'biweight')
        sigma_clip : {number}, optional
            If a sigma-clipped statistic is used for combining, the clip value (the default is 5.)
        """
        # For now, use the first spectrum as the baseline.
        spec1 = self.splist[0]
        # If w_which is not given: use cal_wave if f_cal/debl or cal_sky if f_sky/debl is used
        # otherwise, just use w_shifted
        if w_which is None:
            if f_which in ['f_cal','f_cal_debl']:
                w_which = 'cal_wave'
            elif f_which in ['f_sky','f_sky_debl']:
                w_which = 'sky_wave'
            else:
                w_which = 'w_shifted'
        wl_base = getattr(spec1,w_which)
        n_specs = len(self.splist)
        out = np.full((28,2048),np.nan)

        # For each spectrum, resample to the baseline wavelength grid and combine.
        for oi in range(28):
            warr = np.full((n_specs,2048),np.nan)
            flarr = np.full((n_specs,2048),np.nan)
            for si in range(n_specs):
                spec_this = self.splist[si]
                warr[si,:] = getattr(spec_this,w_which)[oi]
                flarr[si,:] = getattr(spec_this,f_which)[oi]
            order_combined = spec_help.resample_combine(wl_base[oi],warr,flarr,combine_type=combine_type,sigma_clip=sigma_clip)
            out[oi,:] = order_combined
        self.combined_spec = out

    def find_peaks_order(self,oi,f_which='f_sci_sky_debl',w_which=None,prominence=0.1,width=(0,8),
                         pixel_to_wl_interpolation_kind='cubic',fill_value=np.nan,fit_width_kms=None):
        """ Find peaks for an order on multiple spectra
        
        Find all peaks within an order in a list of spectra. This is simply a convenience wrapper around 
        the find_peaks method on an individual spectrum.
        
        Parameters
        ----------
        oi : {int}
            Order index
        f_which : {str}, optional
            Which flux array to combine (the default is 'f_sci_sky_debl')
        w_which : {str}, optional
            Which wavelength array to use (the default is w_shifted - i.e. the stellar rest frame)
        prominence : {float}, optional
            Height above surroundings. Argument to scipy.signal.find_peaks (the default is 0.1)
        width : {tuple}, optional
            Bounds on peak width. Argument to scipy.signal.find_peaks (the default is (0,8))
        pixel_to_wl_interpolation_kind : {str}, optional
            Interpolation to use for converting pixels to wavelength (the default is 'cubic')
        fill_value : {number}, optional
            Fill value in interpolation (the default is np.nan)
        fit_width_kms : {float}, optional
            Fit the lines to find a more precise centroid. [km/s]
            Skip this by not providing a number.
        """
        # If w_which is not given: use cal_wave if f_cal/debl or cal_sky if f_sky/debl is used
        # otherwise, just use w_shifted
        if w_which is None:
            if f_which in ['f_cal']:
                w_which = 'cal_wave'
            elif f_which in ['f_sky']:
                w_which = 'sky_wave'
            else:
                w_which = 'w_shifted'
        all_peaks = []
        for sp in self.splist:
            w_this = getattr(sp,w_which)
            fl_this = getattr(sp,f_which)
            all_peaks.append(sp.find_peaks_order(oi,w=w_this[oi],fl=fl_this[oi],prominence=prominence,width=width,
                             pixel_to_wl_interpolation_kind=pixel_to_wl_interpolation_kind,fill_value=fill_value,
                             fit_width_kms=fit_width_kms))
        return(all_peaks)

    def cross_match_masks_order(self,peak_list,needed_fraction=0.8,proximity=0.1):
        use_lines = []
        n_specs = len(peak_list)
        # just base on the set found in the first list
        # this will be coarse (pixel-level precision)
        # this routine does not really belong here...
        first_peaks = peak_list[0]
        for peak in first_peaks:
            count = 1.
            for later_peaks in peak_list[1:]:
                dists = np.abs(later_peaks - peak)
                if np.nanmin(dists) < proximity:
                    count = count + 1.
            if (count / n_specs) > needed_fraction:
                use_lines.append(peak)
        return(use_lines)

    

    @property
    def sn18(self):
        return [sp.sn18 for sp in self.splist]

    @property
    def filenames(self):
        return [sp.filename for sp in self.splist]

    @property
    def objects(self):
        return [sp.object for sp in self.splist]

    @property
    def exptimes(self):
        return [sp.exptime for sp in self.splist]

    @property
    def qprog(self):
        return [sp.qprog for sp in self.splist]

    @property
    def rv(self):
        return [sp.rv for sp in self.splist]

    @property
    def df(self):
        d = pd.DataFrame(zip(self.objects,self.filenames,self.exptimes,self.sn18,self.qprog,self.rv),columns=['OBJECT_ID','filename','exptime','sn18','qprog','rv'])
        return d
