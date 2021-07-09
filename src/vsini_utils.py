import hpfspec2
import numpy as np
import scipy
import copy
#from hpfspec import utils
##from hpfspec import spec_help
#from hpfspec import stats

class vsini_calibration(object):
    """ An object that holds all information necessary for a CCF-width-based vsini measurement.
    
    This object holds:
    -a reference HPFspec spectrum,
    -CCFs resulting from broadening this spectrum and measuring with its mask
    -CCF widths from measuring the CCFs above
    -routines for interpolating an input spectrum to estimate vsini based on the CCF width, compared to the reference.
    """
    def __init__(self,hpfspectrum,M=None,fitwidth=15,upsample_factor=10.,vsinis=None,velocities=None,
        eps=0.6,orders=[4,5,6,14,15,16,17]):
        """ Create the vsini_calibration object.
        
        This object holds all the needed info to make a CCF-width-based vsini measurement.
        
        Parameters
        ----------
        hpfspectrum : {hpfspec.HPFSpectrum}
            hpfspectrum object
        M : {ccf.mask}, optional
            CCF mask, defaults to using the one in the hpfpsec object
        fitwidth : {float}, optional
            CCF fitting width - argument to hpfspec.ccfwidth_order [km/s]
        upsample_factor : {float}, optional
            Upsample or downsample the spectrum - default is 10 (useful for smoothing out the calibration curves)
        vsinis : {ndarray}, optional
            Array of vsini values to use for calibration curve (defaults to 0-20 in 40 steps) [km/s]
        velocities : {ndarray}, optional
            Array of velocity shifts at which to calculate the CCF [km/s] - defaults to -50 - 50 in 100 steps
        eps : {float}, optional
            Limb-darkening parameter for vsini kernel. 0-1
        orders : {list}, optional
            Orders on which to perform calibration (the default is [4,5,6,14,15,16,17])
        """
        self.hpfspec = copy.deepcopy(hpfspectrum)
        if M is None:
            self.M = hpfspectrum.M
        else:
            self.M = M
        self.fitwidth = fitwidth
        self.upsample_factor = upsample_factor
        if vsinis is None:
            self.vsinis = np.linspace(0,20,40)
        else:
            self.vsinis = vsinis
        if velocities is None:
            self.velocities = np.linspace(-50,50,100)
        else:
            self.velocities = velocities
        self.eps = eps
        self.n_vsini = len(self.vsinis)
        self.n_velocities = len(self.velocities)
        self.calibration_widths = {}
        self.calibration_ccfs = {}
        self.orders = orders

        self.make_calcurve_orders()

    def make_calcurve_order(self,oi):
        """ Generate calibration curve for an order.
        
        This routine takes a given order in a calibration spectrum and:
        1. Resamples to even sampling (upsampling by 10x has made output CCFs smoother as a function of increasing vsini)
        2. Does vsini broadening using PyAstronomy.pyasl.rotBroad()
        3. Calculates the CCF using the mask object
        4. Fits for the width of that CCF.
        5. Stores resulting CCFs and widths in .calibration_ccfs and .calibration_widths
        
        Parameters
        ----------
        oi : {int}
            Order index
        """
        ccfs = np.full((self.n_vsini, self.n_velocities), np.nan)
        widths = np.full((self.n_vsini),np.nan)
        for i,v in enumerate(self.vsinis):
            # Make resampled/broadened spectra
            rotated = self.hpfspec.resample_and_broaden_order(oi, vsini=v, diag=True, upsample_factor=self.upsample_factor)
            # Make CCF and fit the output
            fit_output = self.hpfspec.ccfwidth_order(oi,w=rotated['w_resampled'], fl=rotated['fl_broadened'], debug=True, fitwidth=self.fitwidth, M=self.M, velocities=self.velocities)
            # Store results
            ccfs[i,:] = fit_output['ccf1']
            widths[i] = fit_output['fit']['sigma']
        self.calibration_widths[oi] = widths
        self.calibration_ccfs[oi] = ccfs

    def make_calcurve_orders(self):
        """ Make calibration curve for all orders requested
        
        Convenience function to calculate calibration for all orders
        """
        for oi in self.orders:
            self.make_calcurve_order(oi)

    def measure_vsini_order(self,hpfspec_target,oi,trim_nan=True,debug=False,calcurve_interpolation_kind='cubic',
                            velocities=None):
        """ Interpolate CCF width to measure vsini
        
        Measure CCF using the present mask, and interpolate onto current CCF width vs vsini curve to derive vsini.
        
        Parameters
        ----------
        hpfspec_target : {hpfspec.HPFSpectrum}
            Target spectrum data.
        oi : {int}
            Order index
        trim_nan : {bool}, optional
            Some resampling and rotation routines have issues with NaNs; mask them
        debug : {bool}, optional
            Provide full output
        calcurve_interpolation_kind : {str}, optional
            How to interpolate onto the calibration curve (argument to scipy.interpolate.interp1d) - defaults to 'cubic'
        """
        w = hpfspec_target.w_shifted[oi]
        fl = hpfspec_target.f_sci_sky[oi]
        if trim_nan:
            mask = np.ma.masked_invalid(fl)
            w = w[~mask.mask]
            fl = fl[~mask.mask]
        if velocities is None:
            velocities = self.velocities
        # resample target spectrum to the same level that the calibration spectrum was resampled
        w_resampled, fl_resampled = hpfspec2.spec_help.resample_to_median_sampling(w,fl,upsample_factor=self.upsample_factor)
        
        # Calculate CCF and width using the same mask used to generate calibration curve
        ccf_target = hpfspec_target.ccfwidth_order(oi, w=w_resampled, fl=fl_resampled, debug=True, fitwidth=self.fitwidth, 
                                                   M=self.M,velocities=velocities)
        
        # interpolate measured width onto calibration curve
        cal_ccf_widths = self.calibration_widths[oi]
        cal_ccf_vsinis = self.vsinis
        out_vsini = scipy.interpolate.interp1d(cal_ccf_widths, cal_ccf_vsinis, kind=calcurve_interpolation_kind, 
                                               fill_value=(0.,np.nanmax(self.vsinis)), bounds_error=False)(ccf_target['fit']['sigma'])
        if debug:
            full_out = {'w_resampled':w_resampled, 'fl_resampled':fl_resampled, 'ccf_target':ccf_target, 'out_vsini':out_vsini}
            return(full_out)
        else:
            return(out_vsini)
