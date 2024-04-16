"""
sky module containing...TODO
"""
# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import copy
import os
from time import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.table import Table
from astropy import stats
from astropy import units as u
# =============================================================================
# KOALA packages
# =============================================================================
# Modular
from koala.ancillary import vprint
from koala.exceptions.exceptions import TelluricNoFileError
from koala.corrections.correction import CorrectionBase
from koala.data_container import DataContainer
from koala.rss import RSS
from koala.cubing import Cube
# Original
from koala.onedspec import smooth_spectrum # TODO: Remove

# =============================================================================
# Background estimators
# =============================================================================

class BackgroundEstimator:
    # TODO: this is too heterogeneous
    def percentile(data, percentiles=[16, 50, 84], axis=0):
        """Compute the background data and dispersion from the percentiles."""
        plow, background, pup = np.nanpercentile(data, percentiles, axis=axis)
        background_sigma = (pup - plow) / 2
        return background, background_sigma

    def mad(data, axis=0):
        """Median absolute deviation background estimator."""
        background = np.nanmedian(data, axis=axis)
        mad = np.nanmedian(np.abs(data  - np.expand_dims(background, axis=axis)),
                        axis=axis)
        background_sigma = 1.4826 * mad
        return background, background_sigma

    def linear(data, axis=0):
        """Background estimator from linear fit (Work in progress)."""
        print('>>> LINEAR FIT (WARNING: work in progress!)')
        if data.ndim < 2:
            raise ValueError("Dataset must have at least two dimensions")
        else:
            wl_axis = -1
            axis = np.asarray(axis)
            for ax in range(data.ndim):
                if ax not in axis:
                    wl_axis = ax
            n_wavelength = data.shape[wl_axis]
            n_spectra = data.size // n_wavelength
            if wl_axis == 0:
                spectra = data.reshape((n_wavelength, n_spectra))
            elif wl_axis == data.ndim -1:
                spectra = data.reshape((n_spectra, n_wavelength)).T
            else:
                raise ValueError('Wavelength axis not found!')
        plow, background, pup = np.nanpercentile(spectra, [16, 50, 84], axis=1)
        background_sigma = (pup - plow) / 2
        print('>>>', data.shape, spectra.shape, background.shape)
        total_flux = np.nansum(spectra)
        return background, background_sigma
        
    def mode(data, axis=0, n_bins=None, bin_range=None):
        #TODO
        raise NotImplementedError("Sorry not implemented :(")


# =============================================================================
# Continuum estimators
# =============================================================================
class ContinuumEstimator:
    # TODO: refactor and homogeneize
    
    def medfilt_continuum(data, window_size=5):
        continuum = scipy.signal.medfilt(data, window_size)
        return continuum

    
    def pol_continuum(data, wavelength, pol_order=3, **polfit_kwargs):
        fit = np.polyfit(data, wavelength, pol_order, **polfit_kwargs)
        polynomial = np.poly1d(fit)
        return polynomial(wavelength)

    
    default_min_separation = 10
    @classmethod
    def lower_envelope(self, x, y, min_separation=None):
        '''
        Fit lower envelope of a single spectrum:
        1) Find local minima, with a minimum separation `min_separation`.
        2) Interpolate linearly between them.
        3) Add "typical" (~median) offset.
        '''
        if min_separation is None:
            min_separation = self.default_min_separation
        valleys = []
        y[np.isnan(y)] = np.inf
        for i in range(min_separation, y.size-min_separation-1):
            if np.argmin(y[i-min_separation:i+min_separation+1]) == min_separation:
                valleys.append(i)
        y[~np.isfinite(y)] = np.nan

        continuum = np.fmin(y, np.interp(x, x[valleys], y[valleys]))

        offset = y - continuum
        offset = np.nanpercentile(offset[offset > 0], np.linspace(1, 50, 51))
        density = (np.arange(offset.size) + 1) / offset
        offset = np.median(offset[density > np.max(density)/2])

        return continuum+offset, offset


class ContinuumModel:

    def __init__(self, dc:DataContainer, min_separation=None):
        self.update(dc, min_separation)
        
        
    # TODO: Don't assume RSS format (intensity[spec_id, wavelength])
    #       def update(self, dc:DataContainer, min_separation=None):
    def update(self, dc:RSS, min_separation=None):
        n_spectra = dc.intensity.shape[0]
        self.intensity = np.zeros_like(dc.intensity)
        self.scale = np.zeros(n_spectra)
        
        print(f"> Find continuum for {n_spectra} spectra:")
        t0 = time()

        for i in range(n_spectra):
            self.intensity[i], self.scale[i] = ContinuumEstimator.lower_envelope(dc.wavelength, dc.intensity[i], min_separation)

        print(f"  Done ({time()-t0:.3g} s)")
        self.strong_sky_lines = self.detect_lines(dc)


    def detect_lines(self, dc:DataContainer, n_sigmas=3):
        SN = (dc.intensity - self.intensity) / self.scale[:, np.newaxis]
        SN_p16, SN_p50, SN_p84 = np.nanpercentile(SN, [16, 50, 84], axis=0)

        line_mask = SN_p16 - (n_sigmas-1)*(SN_p84-SN_p16)/2 > 0
        line_mask[0] = False
        line_mask[-1] = False

        line_left = np.where(~line_mask[:-1] & line_mask[1:])[0]
        line_right = np.where(line_mask[:-1] & ~line_mask[1:])[0]
        line_right += 1
        print(f'  {line_left.size} strong sky lines ({np.count_nonzero(line_mask)} out of {dc.wavelength.size} wavelengths)')
        return Table((line_left, line_right), names=('left', 'right'))
        
    
    

# =============================================================================
# Sky lines library
# =============================================================================

def uves_sky_lines():
    """Library of sky emission lines measured with UVES@VLT.
    
    Description
    -----------
    For more details see https://www.eso.org/observing/dfo/quality/UVES/pipeline/sky_spectrum.html

    Returns
    -------
    - line_wavelength:
        Array containing the wavelenght position of each emission line centroid in Angstrom.
    - line_fwhm:
        Array containing the FWHM values of each line expressed in Ansgtrom.
    - line_flux: 
        Array containing the flux of each line expressed in 1e-16 ergs/s/A/cm^2/arcsec^2
    """
    # Prefix of each table
    prefix = ["346", "437", "580L", "580U", "800U", "860L", "860U"]
    # Path to tables
    data_path = os.path.join(os.path.dirname(__file__), "..", "input_data", "sky_lines", "ESO-UVES")

    line_wavelength = np.empty(1)
    line_fwhm = np.empty(1)
    line_flux = np.empty(1)

    for p in prefix:
        file = os.path.join(data_path, f"gident_{p}.tfits")
        if not os.path.isfile(file):
            raise NameError(f"File '{file}' could not be found")
        with fits.open(file) as f:
            wave, fwhm, flux = f[1].data['LAMBDA_AIR'], f[1].data['FWHM'], f[1].data['FLUX']
            line_wavelength = np.hstack((line_wavelength, wave))
            line_fwhm = np.hstack((line_fwhm, fwhm))
            line_flux = np.hstack((line_flux, flux))
    # Sort lines in terms of wavelenth
    sort_pos = np.argsort(line_wavelength[1:])
    return line_wavelength[1:][sort_pos], line_fwhm[1:][sort_pos], line_flux[1:][sort_pos]

# =============================================================================
# Sky models
# =============================================================================
# TODO: convert this class to an ABC
class SkyModel(object):
    """
    Abstract class of a sky emission model.

    Attributes
    ----------
    wavelength:
        TODO
    intensity:
        TODO 1D or 2D array, default None
    variance:
        TODO
    verbose:
        Print messages during the execution.
    Methods
    -------
    substract
    substract_PCA
    """
    verbose = True

    def __init__(self, **kwargs):
        self.wavelength = kwargs.get('wavelength', None)
        self.intensity = kwargs.get('intensity', None)
        self.variance = kwargs.get('variance', None)
        self.verbose = kwargs.get('verbose', True)

    def substract(self, data, variance, axis=-1, verbose=False):
        """Substracts the sky_model to all fibres in the rss

        Parameters
        ----------
        data: (np.ndarray)
            Data array for which the sky will be substracted
        variance: (np.ndarray)
            Array of variance data to include errors on determining the sky.
        axis: (np.ndarray)
            Spectral direction of data
        Returns
        -------
        data_subs: (np.ndarray)
        var_subs: (np.ndarray)
        """
        # TODO
        if data.ndim == 3 and self.intensity.ndim == 1:
            skymodel_intensity = self.intensity[:, np.newaxis, np.newaxis]
            skymodel_var = self.intensity[:, np.newaxis, np.newaxis]
        elif data.ndim == 2 and self.intensity.ndim == 1:
            skymodel_intensity = self.intensity[np.newaxis, :]
            skymodel_var = self.intensity[np.newaxis, :]
        else:
            self.vprint(f"Data dimensions ({data.shape}) cannot be reconciled with sky mode ({self.intensity.shape})")
        data_subs = data - skymodel_intensity
        var_subs = variance + skymodel_var
        return data_subs, var_subs

    def substract_pca():
        # TODO: Implement PCA substraction method
        pass
    
    def vprint(self, *messages):
        """Print a message"""
        if self.verbose:
            print("[SkyModel] ", *messages)



class SkyOffset(SkyModel):
    """
    Sky model based on a single RSS offset sky exposure

    Description
    -----------
    This class builds a sky emission model from individual sky exposures.

    Attributes
    ----------
    - dc:
        Data container used to estimate the sky
    - exptime:
        Data container net exposure time
    """
    def __init__(self, dc):
        """

        Parameters
        ----------
        rss: RSS
            Raw Stacked Spectra corresponding to the offset-sky exposure.
        """
        self.dc = dc
        self.exptime = dc.info['exptime']
        super().__init__()

    def estimate_sky(self):
        self.intensity, self.variance = BackgroundEstimator.percentile(
            self.dc.intensity_corrected, percentiles=[16, 50, 84])
        self.intensity, self.variance = (
            self.intensity / self.exptime,
            self.variance / self.exptime)


class SkyFromObject(SkyModel):
    """
    Sky model based on a single Data Container.

    Description
    -----------
    This class builds a sky emission model using the data
    from a given Data Container that includes the contribution
    of an additional source (i.e. star/galaxy).

    Attributes
    ----------

    """
    bckgr = None
    bckgr_sigma = None
    continuum = None

    def __init__(self, dc,
                 bckgr_estimator='mad',
                 bckgr_params=None,
                 source_mask_nsigma=3,
                 remove_cont=False,
                 cont_estimator='median',
                 cont_estimator_args=None):
        """
        Params
        ------
        - dc:
            Input DataContainer object
        - bckgr_estimator: (str, default='mad')
            Background estimator method to be used.
        - bckgr_params: (dict, default=None)
        - remove_cont: (bool, default=False)
            If True, the continuum will be removed.
        """
        self.vprint("Creating SkyModel from input Data Container")
        # Data container
        self.dc = dc
        # Background estimator
        self.vprint("Estimating sky background contribution...")
        self.estimate_background(bckgr_estimator, bckgr_params, source_mask_nsigma)
        if remove_cont:
            self.vprint("Removing background continuum")
            self.remove_continuum(cont_estimator, cont_estimator_args)

        super().__init__(wavelength=self.dc.wavelength,
                         intensity=self.bckgr,
                         variance=self.bckgr_sigma**2)

    def estimate_background(self, bckgr_estimator, bckgr_params=None, source_mask_nsigma=3):
        """Estimate the background.
        
        Parameters
        ----------
        - bckgr_estimator: (str)
            Background estimator method. Currently available:
            - mad (median absolute deviation),
            - percentile (median percentile +/- (84th - 16th) * 0.5).
            For details see BackgroundEstimator class.
        Returns
        -------
        - background: (np.ndarray)
            Estimated background
        - background_sigma: (np.ndarray)
            Estimated background standard deviation
        """
        if bckgr_params is None:
            bckgr_params = {}

        if hasattr(BackgroundEstimator, bckgr_estimator):
            estimator = getattr(BackgroundEstimator, bckgr_estimator)
        else:
            raise NameError(f"Input background estimator {bckgr_estimator} does not exist")        

        data = self.dc.intensity_corrected.copy()

        if data.ndim == 3:
            if "axis" not in bckgr_params.keys():
                bckgr_params["axis"] = (1, 2)
            else:
                bckgr_params["axis"] = (1, 2)
            dims_to_expand = (1, 2)
        elif data.ndim == 2:
            if "axis" not in bckgr_params.keys():
                bckgr_params["axis"] = (0)
            else:
                bckgr_params["axis"] = (0)
            dims_to_expand = (0)

        if source_mask_nsigma is not None:
            if self.bckgr is None:
                # Call it again
                self.vprint("Pre-estimating background using all data")
                self.estimate_background(bckgr_estimator=bckgr_estimator, bckgr_params=bckgr_params,
                                         source_mask_nsigma=None)
            self.vprint(f"Applying sigma-clipping mask (n-sigma={source_mask_nsigma})")
            source_mask = (data > np.expand_dims(self.bckgr, dims_to_expand)
                           + source_mask_nsigma * np.expand_dims(self.bckgr_sigma, dims_to_expand))
            data[source_mask] = np.nan
            self.bckgr, self.bckgr_sigma = estimator(data, **bckgr_params)
        else:
            self.bckgr, self.bckgr_sigma = estimator(data, **bckgr_params)
        return self.bckgr, self.bckgr_sigma

    def remove_continuum(self, cont_estimator="median", cont_estimator_args=None):
        """Remove the continuum from the background model.
        
        Parameters
        ----------
        - method: (str)
            Method used to estimate the continuum signal.
        """
        if cont_estimator_args is None:
            cont_estimator_args = {}
        if self.bckgr is not None:
            if hasattr(ContinuumEstimator, cont_estimator):
                estimator = getattr(ContinuumEstimator, cont_estimator)
            else:
                raise NameError(
                    f"{cont_estimator} does not correspond to any available continuum method")
            self.continuum = estimator(self.bckgr, **cont_estimator_args)
            self.bckgr -= self.continuum
        else:
            raise AttributeError("background model has not been computed")

    def fit_emission_lines(self, cont_clean_spec, errors=None, window_size=100,
                           resampling_wave=0.1, **fit_kwargs):
        """

        Parameters
        ----------
        errors
        resampling_wave
        window_size
        cont_clean_spec
        fit_kwargs

        Returns
        -------

        """
        if errors is None:
            errors = np.ones_like(cont_clean_spec)
        # Mask non-finite values
        finite_mask = np.isfinite(cont_clean_spec)
        # Initial guess of line gaussian amplitudes
        p0_amplitude = np.interp(self.sky_lines,
                                 self.dc.wavelength[finite_mask],
                                 cont_clean_spec[finite_mask])
        p0_amplitude = np.clip(p0_amplitude, a_min=0, a_max=None)
        # Fitter function
        fit_g = fitting.LevMarLSQFitter()
        # Initialize the model with a dummy gaussian
        emission_model = models.Gaussian1D(amplitude=0, mean=0, stddev=0)
        emission_spectra = np.zeros_like(self.dc.wavelength)
        # Select window steps
        wavelength_windows = np.arange(self.dc.wavelength.min(), self.dc.wavelength.max(), window_size)
        # Ensure the last element corresponds to the last wavelength point of the RSS
        wavelength_windows[-1] = self.dc.wavelength.max()
        print("Fitting all emission lines ({}) to continuum-substracted sky spectra".format(self.sky_lines.size))
        # Loop over each spectral window
        for wl_min, wl_max in zip(wavelength_windows[:-1], wavelength_windows[1:]):
            print("Starting fit in the wavelength range [{:.1f}, {:.1f}]".format(wl_min, wl_max))
            mask_lines = (self.sky_lines >= wl_min) & (self.sky_lines < wl_max)
            mask = (self.dc.wavelength >= wl_min
                    ) & (self.dc.wavelength < wl_max) & finite_mask
            # Oversample wavelength array to prevent fitting crash for excess of lines
            wave = np.arange(self.dc.wavelength[mask][0], self.dc.wavelength[mask][-1], resampling_wave)
            obs = np.interp(wave, self.dc.wavelength[mask], cont_clean_spec[mask])
            err = np.interp(wave, self.dc.wavelength[mask], errors[mask])
            if mask_lines.any():
                print("> Line to Fit {:.1f}".format(self.sky_lines[mask_lines][0]))
                window_model = models.Gaussian1D(
                    amplitude=p0_amplitude[mask_lines][0],
                    mean=self.sky_lines[mask_lines][0],
                    stddev=1, bounds={'amplitude': (p0_amplitude[mask_lines][0]*0.5, p0_amplitude[mask_lines][0]*10),
                                      'mean': (self.sky_lines[mask_lines][0] - 5, self.sky_lines[mask_lines][0] + 5),
                                      'stddev': (self.sky_lines_fwhm[mask_lines][0]/2, 5)})
                for line, p0, sigma in zip(self.sky_lines[mask_lines][1:], p0_amplitude[mask_lines][1:],
                                           self.sky_lines_fwhm[mask_lines][1:]):
                    print("Line to Fit {:.1f}".format(line))
                    model = models.Gaussian1D(amplitude=p0, mean=line, stddev=sigma,
                                              bounds={'amplitude': (p0*0.5, p0*10), 'mean': (line - 5, line + 5),
                                                      'stddev': (sigma/2, 5)})
                    window_model += model
                g = fit_g(window_model, wave, obs, weights=1/err, **fit_kwargs)
                emission_spectra += g(self.dc.wavelength)
                emission_model += g
        return emission_model, emission_spectra

    def load_sky_lines(self, path_to_table=None, lines_pct=84., **kwargs):
        """TODO"""
        if path_to_table is not None:
            path_to_table = os.path.join(os.path.dirname(__file__),
                                         'input_data', 'sky_lines',
                                         path_to_table)
            self.sky_lines, self.sky_lines_fwhm, self.sky_lines_f = np.loadtxt(
                path_to_table, usecols=(0, 1, 2), unpack=True, **kwargs)
        else:
            self.sky_lines, self.sky_lines_fwhm, self.sky_lines_f = uves_sky_lines()
        # Select only lines within the RSS spectral range
        common_lines = (self.sky_lines >= self.dc.wavelength[0]) & (self.sky_lines <= self.dc.wavelength[-1])
        self.sky_lines, self.sky_lines_fwhm, self.sky_lines_f = (
            self.sky_lines[common_lines], self.sky_lines_fwhm[common_lines],
            self.sky_lines_f[common_lines])
        # Bin lines unresolved for RSS data
        delta_lambda = np.median(np.diff(self.dc.wavelength))
        unresolved_lines = np.where(np.diff(self.sky_lines) <= delta_lambda)[0]
        while len(unresolved_lines) > 0:
            self.sky_lines[unresolved_lines] = (self.sky_lines[unresolved_lines]
                                                + self.sky_lines[unresolved_lines + 1])/2
            self.sky_lines_fwhm[unresolved_lines] = np.sqrt(self.sky_lines_fwhm[unresolved_lines]**2
                                                            + self.sky_lines_fwhm[unresolved_lines + 1]**2)
            self.sky_lines_f[unresolved_lines] = (self.sky_lines_f[unresolved_lines]
                                                  + self.sky_lines_f[unresolved_lines + 1])
            self.sky_lines = np.delete(self.sky_lines, unresolved_lines)
            self.sky_lines_fwhm = np.delete(self.sky_lines_fwhm, unresolved_lines)
            self.sky_lines_f = np.delete(self.sky_lines_f, unresolved_lines)
            unresolved_lines = np.where(np.diff(self.sky_lines) <= delta_lambda)[0]
        # Remove faint lines
        faint = np.where(self.sky_lines_f < np.nanpercentile(self.sky_lines_f, lines_pct))[0]
        self.sky_lines = np.delete(self.sky_lines, faint)
        self.sky_lines_fwhm = np.delete(self.sky_lines_fwhm, faint)
        self.sky_lines_f = np.delete(self.sky_lines_f, faint)

# =============================================================================
# Sky Substraction Correction
# =============================================================================

class SkySubsCorrection(CorrectionBase):
    """Correction for removing sky emission from a datacube."""
    name = "SkyCorrection"
    verbose = True

    def __init__(self, skymodel):
        self.skymodel = skymodel

    def plot_correction(self, data_cont, data_cont_corrected, **kwargs):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10),
                                sharex=True, sharey=True)
        if "im_args" in kwargs.keys():
            im_args = kwargs['im_args']
        else:
            im_args = dict(aspect='auto', interpolation='none',
                           cmap='nipy_spectral',
                           vmin=np.nanpercentile(data_cont.intensity_corrected, 1),
                           vmax=np.nanpercentile(data_cont.intensity_corrected, 99))
        ax = axs[0]
        ax.set_title("Input")
        ax.imshow(data_cont.intensity_corrected, **im_args)
        ax = axs[1]
        ax.set_title("Sky emission substracted")
        mappable = ax.imshow(data_cont_corrected.intensity_corrected, **im_args)
        
        cax = ax.inset_axes((-1.2, -.1, 2.2, 0.02))
        plt.colorbar(mappable, cax=cax, orientation="horizontal")
        
        plt.close(fig)
        return fig
        
    def apply(self, dc, pca=False, verbose=True, plot=False, **plot_kwargs):
        # Set print verbose
        self.verbose = verbose
        # Copy input RSS for storage the changes implemented in the task
        dc_out = copy.deepcopy(dc)
        self.corr_print("Applying sky substraction")
        if pca:
            dc_out.intensity_corrected, dc_out.variance_corrected = self.skymodel.substract_pca(
            dc_out.intensity_corrected, dc_out.variance_corrected)
        else:
            dc_out.intensity_corrected, dc_out.variance_corrected = self.skymodel.substract(
                 dc_out.intensity_corrected, dc_out.variance_corrected)
        self.log_correction(dc_out, status='applied')
        
        if plot:
            if not dc_out.intensity_corrected.ndim == 2:
                # TODO: Include 3D plots
                self.corr_print("Plots can only be produed for 2D Data containers (RSS)")
            fig = self.plot_correction(dc, dc_out, **plot_kwargs)
        else:
            fig = None
        return dc_out, fig

# =============================================================================
# Telluric Correction
# =============================================================================
class TelluricCorrection(CorrectionBase):
    """
    Telluric correction produced by atmosphere absorption. # TODO
    """
    name = "TelluricCorretion"
    target = RSS
    telluric_correction = None
    verbose = True

    def __init__(self,
                 data_container=None,
                 telluric_correction_file=None,
                 telluric_correction=None,
                 wavelength=None,
                 n_fibres=10,
                 verbose=True,
                 frac=0.5):
        
        self.verbose = verbose
        self.corr_print("Obtaining telluric correction using spectrophotometric star...")
        
        self.data_container = data_container

        # Store basic data
        if self.data_container is not None:
            self.corr_print("Estimating telluric correction using input observation")
            self.data_container = data_container
            self.wlm = self.data_container.wavelength

            if self.data_container.__class__ is Cube:
                self.spectra = self.data_container.get_integrated_light_frac(frac=frac)
            elif self.data_container.__class__ is RSS:
                integrated_fibre = np.nansum(self.data_container.intensity_corrected, axis=1)
                # The n-brightest fibres
                self.brightest_fibres = integrated_fibre.argsort()[-n_fibres:]
                self.spectra = np.nansum(
                    self.data_container.intensity_corrected[self.brightest_fibres], axis=0)
                self.spectra_var = np.nansum(
                    self.data_container.variance_corrected[self.brightest_fibres], axis=0)

            self.bad_pixels_mask = np.isfinite(self.spectra) & np.isfinite(self.spectra_var
                                                                           ) & (self.spectra / self.spectra_var > 0)
        elif telluric_correction_file is not None:
            self.corr_print(f"Reading telluric correction from input file {telluric_correction_file}")
            self.wlm, self.telluric_correction = np.loadtxt(telluric_correction_file, unpack=True)
        elif telluric_correction is not None and wavelength is not None:
            self.corr_print("Using user-provided telluric correction")
            self.telluric_correction = telluric_correction
            self.wlm = wavelength
        else:
            raise TelluricNoFileError()

    def telluric_from_smoothed_spec(self, exclude_wlm=None, step=10, weight_fit_median=0.5,
                                    wave_min=None, wave_max=None, plot=True, verbose=False):
        """Estimate the telluric correction function using the smoothed spectra of the input star.
        Parameters
        ----------
        - exclude_wlm
        - step
        - weight_fit_median
        - wave_min
        - wave_max

        Returns
        -------
        - telluric_correction
        """
        self.telluric_correction = np.ones_like(self.wlm)
        if wave_min is None:
            wave_min = self.wlm[0]
        if wave_max is None:
            wave_max = self.wlm[-1]
        if exclude_wlm is None:
            # TODO: This is quite dangerous
            exclude_wlm = [[6450, 6700], [6850, 7050], [7130, 7380]]
        # Mask containing the spectral points to include in the telluric correction
        correct_mask = (self.wlm >= wave_min) & (self.wlm <= wave_max)
        # Mask user-provided spectral regions
        spec_windows_mask = np.ones_like(self.wlm, dtype=bool)
        for window in exclude_wlm:
            spec_windows_mask[(self.wlm >= window[0]) & (self.wlm <= window[1])] = False
        # Master mask used to compute the Telluric correction
        telluric_mask = correct_mask & spec_windows_mask
        smooth_med_star = smooth_spectrum(self.wlm, self.spectra, wave_min=wave_min, wave_max=wave_max, step=step,
                                          weight_fit_median=weight_fit_median,
                                          exclude_wlm=exclude_wlm, plot=False, verbose=verbose)
        self.telluric_correction[telluric_mask] = smooth_med_star[telluric_mask] / self.spectra[telluric_mask]

        waves_for_tc_ = []
        for rango in exclude_wlm:
            if rango[0] < 6563. and rango[1] > 6563.:  # H-alpha is here, skip
                print("  Skipping range with H-alpha...")
            else:
                index_region = np.where((self.wlm >= rango[0]) & (self.wlm <= rango[1]))
                waves_for_tc_.append(index_region)

        waves_for_tc = []
        for rango in waves_for_tc_:
            waves_for_tc = np.concatenate((waves_for_tc, rango[0].tolist()), axis=None)

        # Now, change the value in telluric_correction
        for index in waves_for_tc:
            i = np.int(index)
            if smooth_med_star[i] / self.spectra[i] > 1.:
                self.telluric_correction[i] = smooth_med_star[i] / self.spectra[i]
        if plot:
            fig = self.plot_correction(wave_min=wave_min, wave_max=wave_max, exclude_wlm=exclude_wlm)
            return self.telluric_correction, fig
        return self.telluric_correction

    def telluric_from_model(self, file='telluric_lines.txt', width=30, extra_mask=None, pol_deg=5, plot=False):
        """Estimate the telluric correction function using a model of telluric absorption lines.
        Parameters
        ----------
        - file: str, (default="telluric_lines.txt")
            File containing the list of telluric lines to mask.
        - width: int
            Half-window size (AA) to account for the instrumental dispersion.
        - extra_mask: 1D bool array
            Mask containing additional spectral regions to mask during Telluric estimation.
        - pol_deg: int
            Polynomial degree to fit the masked stellar spectra.
        Returns
        -------
        - telluric_correction: 1D array
            Telluric correction function.
        """
        w_l_1, w_l_2, res_intensity, w_lines = np.loadtxt(
            os.path.join(os.path.dirname(__file__), '..', 'input_data', 'sky_lines', file), unpack=True)
        # Mask telluric regions
        mask = np.ones_like(self.wlm, dtype=bool)
        self.telluric_correction = np.ones(self.wlm.size, dtype=float)
        for b, r in zip(w_l_1, w_l_2):
            mask[(self.wlm >= b - width) & (self.wlm <= r + width)] = False
        if extra_mask is not None:
            mask = mask & extra_mask
        # Polynomial fit to unmasked regions
        # p = np.polyfit(self.wlm[mask & self.bad_pixels_mask],
        #                self.spectra[mask & self.bad_pixels_mask],
        #                w=1/self.spectra_var[mask & self.bad_pixels_mask]**0.5, deg=pol_deg)
        # pol_fit = np.poly1d(p)
        # self.telluric_correction[~mask] = pol_fit(self.wlm[~mask]) / (self.spectra[~mask])
        # Linear interpolation
        # std = std(star_flux) * tellurics \propto star_flux * tellurics
        std = np.nanstd(self.data_container.intensity_corrected, axis=0)
        stellar = np.interp(self.wlm, self.wlm[mask & self.bad_pixels_mask],
                            std[mask & self.bad_pixels_mask])
        # self.telluric_correction[~mask] = stellar[~mask] / self.spectra[~mask]
        self.telluric_correction[~mask] = stellar[~mask] / std[~mask]

        self.telluric_correction = np.clip(self.telluric_correction, a_min=1, a_max=None)
        if plot:
            fig = self.plot_correction(exclude_wlm=np.vstack((w_l_1 - width, w_l_2 + width)).T,
                                       wave_min=self.wlm[0], wave_max=self.wlm[-1])
            return self.telluric_correction, fig
        return self.telluric_correction

    def plot_correction(self, fig_size=12, wave_min=None, wave_max=None, exclude_wlm=None):
        fig = plt.figure(figsize=(fig_size, fig_size / 2.5))
        ax = fig.add_subplot(111)
        if self.data_container.__class__ is Cube:
            print("  Telluric correction for this star (" + self.data_container.combined_cube.object + ") :")
            ax.plot(self.wlm, self.spectra, color="b", alpha=0.3, label='Original')
            ax.plot(self.wlm, self.spectra * self.telluric_correction, color="g", alpha=0.5, label='Telluric corrected')
            ax.set_ylim(np.nanmin(self.spectra), np.nanmax(self.spectra))
        else:
            ax.set_title("Telluric correction using fibres {} (blue) and {} (red)"
                         .format(self.brightest_fibres[0], self.brightest_fibres[1]))
            ax.plot(self.wlm, self.data_container.intensity_corrected[self.brightest_fibres[0]], color="b",
                    label='Original', lw=3, alpha=0.8)
            ax.plot(self.wlm, self.data_container.intensity_corrected[self.brightest_fibres[0]] * self.telluric_correction,
                    color="g", alpha=1, lw=0.8, label='Telluric corrected')
            ax.plot(self.wlm, self.data_container.intensity_corrected[self.brightest_fibres[1]], color="r",
                    label='Original', lw=3, alpha=0.8)
            ax.plot(self.wlm,
                    self.data_container.intensity_corrected[self.brightest_fibres[1]] * self.telluric_correction,
                    color="purple", alpha=1, lw=.8, label='Telluric corrected')
            ax.set_ylim(np.nanpercentile(self.data_container.intensity_corrected[self.brightest_fibres[[0, 1]]], 1),
                        np.nanpercentile(self.data_container.intensity_corrected[self.brightest_fibres[[0, 1]]], 99))
        ax.legend(ncol=2)
        ax.axvline(x=wave_min, color='k', linestyle='--')
        ax.axvline(x=wave_max, color='k', linestyle='--')
        ax.set_xlim(self.wlm[0] - 10, self.wlm[-1] + 10)
        ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
        if exclude_wlm is not None:
            for i in range(len(exclude_wlm)):
                ax.axvspan(exclude_wlm[i][0], exclude_wlm[i][1], color='c', alpha=0.1)
        ax.minorticks_on()
        # plt.show()
        plt.close(fig)
        return fig

    def apply(self, rss, verbose=True, is_combined_cube=False, update=True):
        # Set print verbose
        self.verbose = verbose

        # Check wavelength
        if not rss.wavelength.size == self.wlm.size or not np.allclose(rss.wavelength, self.wlm, equal_nan=True):
            self.corr_vprint("Interpolating correction to input wavelength")
            self.interpolate_model(rss.wavelength, update=update)

            
        # Copy input RSS for storage the changes implemented in the task
        rss_out = copy.deepcopy(rss)
        self.corr_print("Applying telluric correction to this star...")
        rss_out.intensity_corrected *= self.telluric_correction
        rss_out.variance_corrected *= self.telluric_correction**2
        self.log_correction(rss, status='applied')
        return rss_out

    def interpolate_model(self, wavelength, update=True):
        """Interpolate the telluric correction model to the input wavelength array."""
        telluric_correction = np.interp(wavelength, self.wlm, self.telluric_correction, left=1, right=1)
        if update:
            self.teluric_correction = telluric_correction
            self.wlm = wavelength
        return telluric_correction

    def save(self, filename='telluric_correction.txt', **kwargs):
        """Save telluric correction function to text file."""
        self.corr_print(f"Saving telluric correction into file {filename}")
        np.savetxt(filename, np.array([self.wlm, self.telluric_correction]).T, **kwargs)

def combine_telluric_corrections(list_of_telcorr, ref_wavelength):
    """Combine a list of input telluric corrections."""
    print("Combining input telluric corrections")    
    telluric_corrections = np.zeros((len(list_of_telcorr), ref_wavelength.size))
    for i, telcorr in enumerate(list_of_telcorr):
        telluric_corrections[i] = telcorr.interpolate_model(ref_wavelength)

    telluric_correction = np.nanmedian(telluric_corrections, axis=0)
    return TelluricCorrection(telluric_correction=telluric_correction, wavelength=ref_wavelength, verbose=False)



# =============================================================================
# Self-calibration based on strong sky lines
# =============================================================================
class SkySelfCalibration(CorrectionBase):
    """Wavelength calibration, throughput, and sky model based on strong sky lines."""
    name = "SkySelfCalibration"
    verbose = True

    # TODO: Don't assume RSS format (intensity[spec_id, wavelength])
    #       def __init__(self, dc:DataContainer, continuum:ContinuumModel):
    def __init__(self, dc:RSS):
        self.update(dc)

    def update(self, dc:RSS):
        self.dc = dc
        self.biweight_sky(dc)
        self.update_sky_lines(dc)


    # TODO: use SkyModel as an argument to update()?
    def biweight_sky(self, dc):
        self.wavelength = dc.wavelength.to_value(u.Angstrom)
        self.sky_intensity = stats.biweight.biweight_location(dc.intensity, axis=0)


    def update_sky_lines(self, dc):
        sky_cont, sky_err = ContinuumEstimator.lower_envelope(self.wavelength, self.sky_intensity)#, min_separation)
        sky_lines = self.sky_intensity - sky_cont

        self.continuum = ContinuumModel(dc)
        self.continuum.strong_sky_lines.add_column(0., name='sky_wavelength')
        self.continuum.strong_sky_lines.add_column(0., name='sky_intensity')
        for line in self.continuum.strong_sky_lines:
            wavelength = self.wavelength[line['left']:line['right']]
            spectrum = sky_lines[line['left']:line['right']]
            weight = spectrum
            line['sky_wavelength'] = np.nansum(weight*wavelength) / np.nansum(weight)
            line['sky_intensity'] = np.nanmean(spectrum)

    def measure_lines(self, spec_id):
        intensity = self.dc.intensity[spec_id]
        continuum = self.continuum.intensity[spec_id]
        line_wavelength = np.zeros(len(self.continuum.strong_sky_lines))
        line_intensity = np.zeros(len(self.continuum.strong_sky_lines))
        for i, line in enumerate(self.continuum.strong_sky_lines):
            section_wavelength = self.wavelength[line['left']:line['right']]
            section_intensity = (intensity - continuum)[line['left']:line['right']]
            weight = section_intensity
            line_wavelength[i] = np.nansum(weight*section_wavelength) / np.nansum(weight)
            line_intensity[i] = np.nanmean(section_intensity)
        return line_wavelength, line_intensity


    def apply(self, rss, verbose=True, is_combined_cube=False, update=True):
        # Set print verbose
        self.verbose = verbose


# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
