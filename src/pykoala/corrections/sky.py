"""
sky module containing...TODO
"""
# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits
from astropy.modeling import models, fitting
# =============================================================================
# KOALA packages
# =============================================================================
# Modular
from pykoala.ancillary import vprint
from pykoala.exceptions.exceptions import TelluricNoFileError
from pykoala.corrections.correction import CorrectionBase
from pykoala.rss import RSS
from pykoala.cubing import Cube
# Original
from pykoala.ancillary import smooth_spectrum

# =============================================================================
# Background estimators
# =============================================================================

class BackgroundEstimator:
    """
    Class for estimating background and its dispersion using different statistical methods.

    Methods
    -------
    percentile(data, percentiles=[16, 50, 84], axis=0)
        Compute the background and dispersion from specified percentiles.
    
    mad(data, axis=0)
        Estimate the background and dispersion using the Median Absolute Deviation (MAD) method.
    
    mode(data, axis=0, n_bins=None, bin_range=None)
        Estimate the background and dispersion using the mode of the data distribution.
    """

    @staticmethod
    def percentile(data, percentiles=[16, 50, 84], axis=0):
        """
        Compute the background and dispersion from specified percentiles.

        Parameters
        ----------
        data : np.ndarray
            The input data array from which to compute the background and dispersion.
        percentiles : list of float, optional
            The percentiles to use for computation. Default is [16, 50, 84].
        axis : int, optional
            The axis along which to compute the percentiles. Default is 0.

        Returns
        -------
        background : np.ndarray
            The computed background (median) of the data.
        background_sigma : np.ndarray
            The dispersion (half the interpercentile range) of the data.
        """
        plow, background, pup = np.nanpercentile(data, percentiles, axis=axis)
        background_sigma = (pup - plow) / 2
        return background, background_sigma

    @staticmethod
    def mad(data, axis=0):
        """
        Estimate the background and dispersion using the Median Absolute Deviation (MAD) method.

        Parameters
        ----------
        data : np.ndarray
            The input data array from which to compute the background and dispersion.
        axis : int, optional
            The axis along which to compute the median and MAD. Default is 0.

        Returns
        -------
        background : np.ndarray
            The computed background (median) of the data.
        background_sigma : np.ndarray
            The dispersion (scaled MAD) of the data.
        """
        background = np.nanmedian(data, axis=axis)
        mad = np.nanmedian(np.abs(data - np.expand_dims(background, axis=axis)), axis=axis)
        background_sigma = 1.4826 * mad
        return background, background_sigma

    @staticmethod
    def mode(data, axis=0, n_bins=None, bin_range=None):
        """
        Estimate the background and dispersion using the mode of the data distribution.

        Parameters
        ----------
        data : np.ndarray
            The input data array from which to compute the background and dispersion.
        axis : int, optional
            The axis along which to compute the mode. Default is 0.
        n_bins : int, optional
            The number of bins to use for the histogram. Default is None.
        bin_range : tuple of float, optional
            The range of values for the histogram bins. Default is None.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        # TODO: Implement mode estimation method
        raise NotImplementedError("Sorry, not implemented :(")


# =============================================================================
# Continuum estimators
# =============================================================================
class ContinuumEstimator:
    """
    Class for estimating the continuum of spectral data using different methods.

    Methods
    -------
    medfilt_continuum(data, window_size=5)
        Estimate the continuum using a median filter.
    
    pol_continuum(data, wavelength, pol_order=3, **polfit_kwargs)
        Estimate the continuum using polynomial fitting.
    """

    @staticmethod
    def medfilt_continuum(data, window_size=5):
        """
        Estimate the continuum using a median filter.

        Parameters
        ----------
        data : np.ndarray
            The input data array for which to compute the continuum.
        window_size : int, optional
            The size of the window over which to compute the median filter. Default is 5.

        Returns
        -------
        continuum : np.ndarray
            The estimated continuum of the input data.
        """
        continuum = scipy.signal.medfilt(data, window_size)
        return continuum

    @staticmethod
    def pol_continuum(data, wavelength, pol_order=3, **polfit_kwargs):
        """
        Estimate the continuum using polynomial fitting.

        Parameters
        ----------
        data : np.ndarray
            The input data array for which to compute the continuum.
        wavelength : np.ndarray
            The wavelength array corresponding to the data.
        pol_order : int, optional
            The order of the polynomial to fit. Default is 3.
        **polfit_kwargs : dict, optional
            Additional keyword arguments to pass to `np.polyfit`.

        Returns
        -------
        continuum : np.ndarray
            The estimated continuum of the input data.
        """
        fit = np.polyfit(wavelength, data, pol_order, **polfit_kwargs)
        polynomial = np.poly1d(fit)
        return polynomial(wavelength)

# =============================================================================
# Sky lines library
# =============================================================================

def uves_sky_lines():
    """
    Library of sky emission lines measured with UVES@VLT.

    For more details, see the `UVES Sky Spectrum <https://www.eso.org/observing/dfo/quality/UVES/pipeline/sky_spectrum.html>`_.

    Returns
    -------
    line_wavelength : np.ndarray
        Array containing the wavelength positions of each emission line centroid in Angstroms.
    line_fwhm : np.ndarray
        Array containing the FWHM values of each line expressed in Angstroms.
    line_flux : np.ndarray
        Array containing the flux of each line expressed in 1e-16 ergs/s/A/cm^2/arcsec^2.
    """
    # Prefix of each table
    prefix = ["346", "437", "580L", "580U", "800U", "860L", "860U"]
    # Path to tables
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "input_data", "sky_lines", "ESO-UVES")

    # Initialize arrays to store line properties
    line_wavelength = np.empty(0)
    line_fwhm = np.empty(0)
    line_flux = np.empty(0)

    # Read data from each file
    for p in prefix:
        file = os.path.join(data_path, f"gident_{p}.tfits")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File '{file}' could not be found")
        
        with fits.open(file) as f:
            wave = f[1].data['LAMBDA_AIR']
            fwhm = f[1].data['FWHM']
            flux = f[1].data['FLUX']
            
            line_wavelength = np.hstack((line_wavelength, wave))
            line_fwhm = np.hstack((line_fwhm, fwhm))
            line_flux = np.hstack((line_flux, flux))

    # Sort lines by wavelength
    sort_pos = np.argsort(line_wavelength)
    return line_wavelength[sort_pos], line_fwhm[sort_pos], line_flux[sort_pos]

# =============================================================================
# Sky models
# =============================================================================

class SkyModel(object):
    """
    Abstract class for a sky emission model.

    Attributes
    ----------
    wavelength : np.ndarray
        1-D array representing the wavelengths of the sky model.
    intensity : np.ndarray
        Array representing the intensity of the sky model.
    variance : np.ndarray
        Array representing the variance associated with the sky model. 
        It must have the same dimensions as `intensity`.
    verbose : bool, optional
        If True, print messages during execution. Default is True.

    Methods
    -------
    subtract(data, variance, axis=-1)
        Subtracts the sky model from the given data.
    
    subtract_pca()
        Placeholder for PCA subtraction method.
    
    vprint(*messages)
        Print messages if `verbose` is True.
    """

    verbose = True

    def __init__(self, **kwargs):
        """
        Initialize the SkyModel object.

        Parameters
        ----------
        **kwargs : dict, optional
            Dictionary of parameters to initialize the SkyModel.
            Accepted keys are:
            - wavelength : np.ndarray
                1-D array representing the wavelengths of the sky model.
            - intensity : np.ndarray
                Array representing the intensity of the sky model.
            - variance : np.ndarray
                Array representing the variance associated with the sky model.
            - verbose : bool
                If True, print messages during execution. Default is True.
        """
        self.wavelength = kwargs.get('wavelength', None)
        self.intensity = kwargs.get('intensity', None)
        self.variance = kwargs.get('variance', None)
        self.verbose = kwargs.get('verbose', True)

    def substract(self, data, variance, axis=-1, verbose=False):
        """
        Subtracts the sky model from the given data.

        Parameters
        ----------
        data : np.ndarray
            Data array from which the sky will be subtracted.
        variance : np.ndarray
            Array of variance data to include errors in determining the sky.
        axis : int, optional
            Spectral direction of the data. Default is -1.

        Returns
        -------
        data_subs : np.ndarray
            Data array after the sky model has been subtracted.
        var_subs : np.ndarray
            Variance array after including the sky model variance.
        """
        if data.ndim == 3 and self.intensity.ndim == 1:
            skymodel_intensity = self.intensity[:, np.newaxis, np.newaxis]
            skymodel_var = self.intensity[:, np.newaxis, np.newaxis]
        elif data.ndim == 2 and self.intensity.ndim == 1:
            skymodel_intensity = self.intensity[np.newaxis, :]
            skymodel_var = self.intensity[np.newaxis, :]
        elif data.ndim == 2 and self.intensity.ndim == 2:  
            skymodel_intensity = self.intensity       
            skymodel_var = self.variance
        else:
            self.vprint(
                f"Data dimensions ({data.shape}) cannot be reconciled with "
                + f"sky mode ({self.intensity.shape})")
        data_subs = data - skymodel_intensity
        var_subs = variance + skymodel_var
        return data_subs, var_subs

    def substract_pca():
        """
        Placeholder for PCA subtraction method.

        This method is not yet implemented.
        """
        pass
    
    def vprint(self, *messages):
        """
        Print messages if `verbose` is True.

        Parameters
        ----------
        *messages : str
            Messages to be printed.
        """
        if self.verbose:
            print("[SkyModel] ", *messages)



class SkyOffset(SkyModel):
    """
    Sky model based on a single RSS offset sky exposure.

    This class builds a sky emission model from individual sky exposures.

    Attributes
    ----------
    dc : DataContainer
        Data container used to estimate the sky.
    exptime : float
        Net exposure time from the data container.
    """

    def __init__(self, dc):
        """
        Initialize the SkyOffset model with a data container.

        Parameters
        ----------
        dc : DataContainer
            Raw Stacked Spectra (RSS) corresponding to the offset-sky exposure.
        """
        self.dc = dc
        self.exptime = dc.info['exptime']
        super().__init__()

    def estimate_sky(self):
        """
        Estimate the sky emission model.

        This method calculates the intensity and variance of the sky model using
        percentiles, then normalizes them by the exposure time.
        """
        self.intensity, self.variance = BackgroundEstimator.percentile(
            self.dc.intensity, percentiles=[16, 50, 84])
        self.intensity, self.variance = (
            self.intensity / self.exptime,
            self.variance / self.exptime)

class SkyFromObject(SkyModel):
    """
    Sky model based on a single Data Container.

    This class builds a sky emission model using the data from a given Data Container
    that includes the contribution of an additional source (i.e., star/galaxy).

    Attributes
    ----------
    dc : DataContainer
        Input DataContainer object.
    exptime : float
        Net exposure time from the data container.
    bckgr : np.ndarray or None
        Estimated background. Initialized as None.
    bckgr_sigma : np.ndarray or None
        Estimated background standard deviation. Initialized as None.
    continuum : np.ndarray or None
        Estimated continuum. Initialized as None.
    """

    bckgr = None
    bckgr_sigma = None
    continuum = None

    def __init__(self, dc, bckgr_estimator='mad', bckgr_params=None, 
                 source_mask_nsigma=3, remove_cont=False, 
                 cont_estimator='median', cont_estimator_args=None):
        """
        Initialize the SkyFromObject model.

        Parameters
        ----------
        dc : DataContainer
            Input DataContainer object.
        bckgr_estimator : str, optional
            Background estimator method to be used. Default is 'mad'.
        bckgr_params : dict, optional
            Parameters for the background estimator. Default is None.
        source_mask_nsigma : float, optional
            Sigma level for masking sources. Default is 3.
        remove_cont : bool, optional
            If True, the continuum will be removed. Default is False.
        cont_estimator : str, optional
            Method to estimate the continuum signal. Default is 'median'.
        cont_estimator_args : dict, optional
            Arguments for the continuum estimator. Default is None.
        """
        self.vprint("Creating SkyModel from input Data Container")
        self.dc = dc
        self.exptime = dc.info['exptime']
        self.vprint("Estimating sky background contribution...")
        self.estimate_background(bckgr_estimator, bckgr_params, source_mask_nsigma)
        if remove_cont:
            self.vprint("Removing background continuum")
            self.remove_continuum(cont_estimator, cont_estimator_args)
        super().__init__(wavelength=self.dc.wavelength,
                         intensity=self.bckgr,
                         variance=self.bckgr_sigma**2)

    def estimate_background(self, bckgr_estimator, bckgr_params=None, source_mask_nsigma=3):
        """
        Estimate the background.

        Parameters
        ----------
        bckgr_estimator : str
            Background estimator method. Available methods: 'mad', 'percentile'.
        bckgr_params : dict, optional
            Parameters for the background estimator. Default is None.
        source_mask_nsigma : float, optional
            Sigma level for masking sources. Default is 3.

        Returns
        -------
        np.ndarray
            Estimated background.
        np.ndarray
            Estimated background standard deviation.
        """
        if bckgr_params is None:
            bckgr_params = {}

        if hasattr(BackgroundEstimator, bckgr_estimator):
            estimator = getattr(BackgroundEstimator, bckgr_estimator)
        else:
            raise NameError(f"Input background estimator {bckgr_estimator} does not exist")

        data = self.dc.intensity.copy()

        if data.ndim == 3:
            bckgr_params["axis"] = bckgr_params.get("axis", (1, 2))
            dims_to_expand = (1, 2)
        elif data.ndim == 2:
            bckgr_params["axis"] = bckgr_params.get("axis", 0)
            dims_to_expand = (0)

        if source_mask_nsigma is not None:
            if self.bckgr is None:
                self.vprint("Pre-estimating background using all data")
                self.estimate_background(bckgr_estimator, bckgr_params, None)
            self.vprint(f"Applying sigma-clipping mask (n-sigma={source_mask_nsigma})")
            source_mask = (data > np.expand_dims(self.bckgr, dims_to_expand) +
                           source_mask_nsigma
                           * np.expand_dims(self.bckgr_sigma, dims_to_expand))
            data[source_mask] = np.nan

        self.bckgr, self.bckgr_sigma = estimator(data, **bckgr_params)
        return self.bckgr, self.bckgr_sigma

    def remove_continuum(self, cont_estimator="median", cont_estimator_args=None):
        """
        Remove the continuum from the background model.

        Parameters
        ----------
        cont_estimator : str, optional
            Method to estimate the continuum signal. Default is 'median'.
        cont_estimator_args : dict, optional
            Arguments for the continuum estimator. Default is None.
        """
        if cont_estimator_args is None:
            cont_estimator_args = {}
        if self.bckgr is not None:
            if hasattr(ContinuumEstimator, cont_estimator):
                estimator = getattr(ContinuumEstimator, cont_estimator)
            else:
                raise NameError(f"{cont_estimator} does not correspond to any available continuum method")
            self.continuum = estimator(self.bckgr, **cont_estimator_args)
            self.bckgr -= self.continuum
        else:
            raise AttributeError("Background model has not been computed")

    def fit_emission_lines(self, cont_clean_spec, errors=None, window_size=100,
                           resampling_wave=0.1, **fit_kwargs):
        """
        Fit emission lines to the continuum-subtracted spectrum.

        Parameters
        ----------
        cont_clean_spec : np.ndarray
            Continuum-subtracted spectrum.
        errors : np.ndarray, optional
            Errors associated with the spectrum. Default is None.
        window_size : int, optional
            Size of the wavelength window for fitting. Default is 100.
        resampling_wave : float, optional
            Wavelength resampling interval. Default is 0.1.

        Returns
        -------
        emission_model : models.Gaussian1D
            Fitted emission line model.
        emission_spectra : np.ndarray
            Emission spectra.
        """
        if errors is None:
            errors = np.ones_like(cont_clean_spec)
        finite_mask = np.isfinite(cont_clean_spec)
        p0_amplitude = np.interp(self.sky_lines, self.dc.wavelength[finite_mask],
                                 cont_clean_spec[finite_mask])
        p0_amplitude = np.clip(p0_amplitude, a_min=0, a_max=None)
        fit_g = fitting.LevMarLSQFitter()
        emission_model = models.Gaussian1D(amplitude=0, mean=0, stddev=0)
        emission_spectra = np.zeros_like(self.dc.wavelength)
        wavelength_windows = np.arange(self.dc.wavelength.min(), self.dc.wavelength.max(), window_size)
        wavelength_windows[-1] = self.dc.wavelength.max()
        self.vprint(f"Fitting all emission lines ({self.sky_lines.size})"
                    + " to continuum-subtracted sky spectra")
        for wl_min, wl_max in zip(wavelength_windows[:-1], wavelength_windows[1:]):
            self.vprint(f"Starting fit in the wavelength range [{wl_min:.1f}, {wl_max:.1f}]")
            mask_lines = (self.sky_lines >= wl_min) & (self.sky_lines < wl_max)
            mask = (self.dc.wavelength >= wl_min) & (
                self.dc.wavelength < wl_max) & finite_mask
            wave = np.arange(self.dc.wavelength[mask][0],
                             self.dc.wavelength[mask][-1], resampling_wave)
            obs = np.interp(wave, self.dc.wavelength[mask], cont_clean_spec[mask])
            err = np.interp(wave, self.dc.wavelength[mask], errors[mask])
            if mask_lines.any():
                self.vprint(f"> Line to Fit {self.sky_lines[mask_lines][0]:.1f}")
                window_model = models.Gaussian1D(
                    amplitude=p0_amplitude[mask_lines][0],
                    mean=self.sky_lines[mask_lines][0],
                    stddev=1,
                    bounds={'amplitude': (p0_amplitude[mask_lines][0] * 0.5, p0_amplitude[mask_lines][0] * 10),
                            'mean': (self.sky_lines[mask_lines][0] - 5, self.sky_lines[mask_lines][0] + 5),
                            'stddev': (self.sky_lines_fwhm[mask_lines][0] / 2, 5)}
                )
                for line, p0, sigma in zip(
                    self.sky_lines[mask_lines][1:], p0_amplitude[mask_lines][1:],
                    self.sky_lines_fwhm[mask_lines][1:]):
                    self.vprint(f"Line to Fit {line:.1f}")
                    model = models.Gaussian1D(
                        amplitude=p0, mean=line, stddev=sigma,
                        bounds={'amplitude': (p0 * 0.5, p0 * 10), 'mean': (line - 5, line + 5),
                                'stddev': (sigma / 2, 5)}
                    )
                    window_model += model
                g = fit_g(window_model, wave, obs, weights=1 / err, **fit_kwargs)
                emission_spectra += g(self.dc.wavelength)
                emission_model += g
        return emission_model, emission_spectra

    def load_sky_lines(self, path_to_table=None, lines_pct=84., **kwargs):
        """
        Load sky lines from a file.

        Parameters
        ----------
        path_to_table : str, optional
            Path to the table containing sky lines. Default is None.
        lines_pct : float, optional
            Percentile for selecting faint lines. Default is 84.
        kwargs : dict, optional
            Additional arguments for `np.loadtxt`.

        Returns
        -------
        None
        """
        if path_to_table is not None:
            path_to_table = os.path.join(os.path.dirname(__file__), 'input_data', 'sky_lines', path_to_table)
            self.sky_lines, self.sky_lines_fwhm, self.sky_lines_f = np.loadtxt(
                path_to_table, usecols=(0, 1, 2), unpack=True, **kwargs)
        else:
            self.sky_lines, self.sky_lines_fwhm, self.sky_lines_f = uves_sky_lines()
        common_lines = (self.sky_lines >= self.dc.wavelength[0]) & (self.sky_lines <= self.dc.wavelength[-1])
        self.sky_lines, self.sky_lines_fwhm, self.sky_lines_f = (
            self.sky_lines[common_lines], self.sky_lines_fwhm[common_lines], self.sky_lines_f[common_lines]
        )
        delta_lambda = np.median(np.diff(self.dc.wavelength))
        unresolved_lines = np.where(np.diff(self.sky_lines) <= delta_lambda)[0]
        while len(unresolved_lines) > 0:
            self.sky_lines[unresolved_lines] = (
                self.sky_lines[unresolved_lines] + self.sky_lines[unresolved_lines + 1]) / 2
            self.sky_lines_fwhm[unresolved_lines] = np.sqrt(
                self.sky_lines_fwhm[unresolved_lines]**2 + self.sky_lines_fwhm[unresolved_lines + 1]**2
            )
            self.sky_lines_f[unresolved_lines] = (
                self.sky_lines_f[unresolved_lines] + self.sky_lines_f[unresolved_lines + 1]
            )
            self.sky_lines = np.delete(self.sky_lines, unresolved_lines)
            self.sky_lines_fwhm = np.delete(self.sky_lines_fwhm, unresolved_lines)
            self.sky_lines_f = np.delete(self.sky_lines_f, unresolved_lines)
            unresolved_lines = np.where(np.diff(self.sky_lines) <= delta_lambda)[0]
        faint = np.where(self.sky_lines_f < np.nanpercentile(self.sky_lines_f, lines_pct))[0]
        self.sky_lines = np.delete(self.sky_lines, faint)
        self.sky_lines_fwhm = np.delete(self.sky_lines_fwhm, faint)
        self.sky_lines_f = np.delete(self.sky_lines_f, faint)


# =============================================================================
# Sky Substraction Correction
# =============================================================================

class SkySubsCorrection(CorrectionBase):
    """
    Correction for removing sky emission from a DataContainer.

    This class applies sky emission correction to a DataContainer using a provided sky model. 
    It supports both standard and PCA-based sky subtraction methods and can generate 
    visualizations of the correction process.

    Attributes
    ----------
    name : str
        Name of the correction.
    verbose : bool
        Flag to control verbosity of the correction process.
    skymodel : SkyModel
        The sky model used for sky emission subtraction.
    """

    name = "SkyCorrection"
    verbose = True

    def __init__(self, skymodel):
        """
        Initialize the SkySubsCorrection with a given sky model.

        Parameters
        ----------
        skymodel : SkyModel
            The sky model to be used for sky emission subtraction.
        """
        self.skymodel = skymodel

    def plot_correction(self, data_cont, data_cont_corrected, **kwargs):
        """
        Plot the original and sky-corrected intensity of a DataContainer for comparison.

        Parameters
        ----------
        data_cont : DataContainer
            The original DC before sky correction.
        data_cont_corrected : DataContainer
            The DC after sky correction.
        kwargs : dict
            Additional keyword arguments for `imshow`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        """
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10),
                                sharex=True, sharey=True)
        if data_cont.intensity.ndim == 2:
            original_image = data_cont.intensity
            corr_image = data_cont_corrected.intensity
        elif data_cont.intensity.ndim == 3:
            original_image = data_cont.get_white_image()
            corr_image = data_cont_corrected.get_white_image()

        im_args = kwargs.get(
            'im_args', dict(aspect='auto', interpolation='none',
                            cmap='nipy_spectral',
                            vmin=np.nanpercentile(original_image.intensity, 1),
                            vmax=np.nanpercentile(original_image.intensity, 99))
                            )
        
        ax = axs[0]
        ax.set_title("Input")
        ax.imshow(original_image, **im_args)
        
        ax = axs[1]
        ax.set_title("Sky emission subtracted")
        mappable = ax.imshow(corr_image, **im_args)
        cax = ax.inset_axes((-1.2, -.1, 2.2, 0.02))
        plt.colorbar(mappable, cax=cax, orientation="horizontal")
        if kwargs.get("plot", False):
            plt.show()
        else:
            plt.close(fig)
        return fig

    def apply(self, dc, pca=False, verbose=True, plot=False, **plot_kwargs):
        """
        Apply the sky emission correction to the datacube.

        Parameters
        ----------
        dc : DataContainer
            The DataContainer to be corrected.
        pca : bool, optional
            If True, use PCA-based sky subtraction. Default is False.
        verbose : bool, optional
            If True, print progress messages. Default is True.
        plot : bool, optional
            If True, generate and return plots of the correction. Default is False.
        plot_kwargs : dict
            Additional keyword arguments for the plot.

        Returns
        -------
        dc_out : DataContainer
            The corrected datacube.
        fig : matplotlib.figure.Figure or None
            The figure object containing the plots if `plot` is True, otherwise None.
        """
        # Set verbosity
        self.verbose = verbose
        
        # Copy input datacube to store the changes
        dc_out = copy.deepcopy(dc)
        
        self.corr_print("Applying sky subtraction")
        
        if pca:
            dc_out.intensity, dc_out.variance = self.skymodel.substract_pca(
                dc_out.intensity, dc_out.variance)
        else:
            dc_out.intensity, dc_out.variance = self.skymodel.substract(
                dc_out.intensity, dc_out.variance)
        
        self.log_correction(dc_out, status='applied')
        
        if plot:
            if dc_out.intensity.ndim != 2:
                # TODO: Include 3D plots
                self.corr_print("Plots can only be produced for 2D Data containers (RSS)")
                fig = None
            else:
                fig = self.plot_correction(dc, dc_out, **plot_kwargs)
        else:
            fig = None
        
        return dc_out, fig


# =============================================================================
# Telluric Correction
# =============================================================================
class TelluricCorrection(CorrectionBase):
    """
    Corrects for telluric absorption caused by atmospheric effects.

    This class implements methods to estimate and apply corrections for telluric absorption
    effects to data containers.

    Attributes
    ----------
    name : str
        The name of the correction method.
    telluric_correction : array
        The computed telluric correction.
    verbose : bool
        Controls verbosity of logging messages.

    Methods
    -------
    telluric_from_smoothed_spec(exclude_wlm=None, step=10,
                                weight_fit_median=0.5, wave_min=None,
                                wave_max=None, plot=True, verbose=False):
        Estimates the telluric correction from smoothed spectra.

    telluric_from_model(file='telluric_lines.txt', width=30,
                        extra_mask=None, pol_deg=5, plot=False):
        Estimates the telluric correction using a model of absorption lines.

    plot_correction(fig_size=12, wave_min=None, wave_max=None,
                    exclude_wlm=None, **kwargs):
        Plots the telluric correction.

    apply(rss, verbose=True, is_combined_cube=False, update=True):
        Applies the telluric correction to the input data.

    interpolate_model(wavelength, update=True):
        Interpolates the telluric correction model to match the input wavelength array.

    save(filename='telluric_correction.txt', **kwargs):
        Saves the telluric correction to a text file.
    """
    name = "TelluricCorretion"
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
        """
        Initializes the TelluricCorrection object.

        Parameters
        ----------
        data_container : object, optional
            The data container to use for correction (default is None).
        telluric_correction_file : str, optional
            Path to a file containing the telluric correction data (default is None).
        telluric_correction : array, optional
            The telluric correction array (default is None).
        wavelength : array, optional
            Wavelength array for the data (default is None).
        n_fibres : int, optional
            Number of fibers to consider (default is 10).
        verbose : bool, optional
            Controls verbosity of logging messages (default is True).
        frac : float, optional
            Fraction of the data to use for correction (default is 0.5).
        """
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
                integrated_fibre = np.nansum(self.data_container.intensity, axis=1)
                # The n-brightest fibres
                self.brightest_fibres = integrated_fibre.argsort()[-n_fibres:]
                self.spectra = np.nansum(
                    self.data_container.intensity[self.brightest_fibres], axis=0)
                self.spectra_var = np.nansum(
                    self.data_container.variance[self.brightest_fibres], axis=0)

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
        """
        Estimate the telluric correction function using the smoothed spectra of the input star.

        Parameters
        ----------
        exclude_wlm : list of lists, optional
            List of wavelength ranges to exclude from correction (default is None).
        step : int, optional
            Step size for smoothing (default is 10).
        weight_fit_median : float, optional
            Weight parameter for fitting median (default is 0.5).
        wave_min : float, optional
            Minimum wavelength to consider (default is None).
        wave_max : float, optional
            Maximum wavelength to consider (default is None).
        plot : bool, optional
            Whether to plot the correction (default is True).
        verbose : bool, optional
            Controls verbosity of logging messages (default is False).

        Returns
        -------
        telluric_correction : array
            The computed telluric correction.
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
        """
        Estimate the telluric correction function using a model of telluric absorption lines.

        Parameters
        ----------
        file : str, optional
            Path to the file containing telluric lines (default is 'telluric_lines.txt').
        width : int, optional
            Half-window size to account for instrumental dispersion (default is 30).
        extra_mask : array, optional
            Mask of additional spectral regions to exclude (default is None).
        pol_deg : int, optional
            Polynomial degree for fitting (default is 5).
        plot : bool, optional
            Whether to plot the correction (default is False).

        Returns
        -------
        telluric_correction : array
            The computed telluric correction.
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
        std = np.nanstd(self.data_container.intensity, axis=0)
        stellar = np.interp(self.wlm, self.wlm[mask & self.bad_pixels_mask],
                            std[mask & self.bad_pixels_mask])
        self.telluric_correction[~mask] = stellar[~mask] / std[~mask]

        self.telluric_correction = np.clip(self.telluric_correction, a_min=1, a_max=None)
        if plot:
            fig = self.plot_correction(exclude_wlm=np.vstack((w_l_1 - width, w_l_2 + width)).T,
                                       wave_min=self.wlm[0], wave_max=self.wlm[-1])
            return self.telluric_correction, fig
        return self.telluric_correction

    def plot_correction(self, fig_size=12, wave_min=None, wave_max=None,
                        exclude_wlm=None, **kwargs):
        """
        Plot the telluric correction.

        Parameters
        ----------
        fig_size : float, optional
            Size of the figure (default is 12).
        wave_min : float, optional
            Minimum wavelength to display (default is None).
        wave_max : float, optional
            Maximum wavelength to display (default is None).
        exclude_wlm : array, optional
            List of wavelength ranges to exclude from plot (default is None).
        **kwargs
            Additional keyword arguments for plot customization.

        Returns
        -------
        fig : Figure
            The matplotlib figure object.
        """
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
            ax.plot(self.wlm, self.data_container.intensity[self.brightest_fibres[0]], color="b",
                    label='Original', lw=3, alpha=0.8)
            ax.plot(self.wlm, self.data_container.intensity[self.brightest_fibres[0]] * self.telluric_correction,
                    color="g", alpha=1, lw=0.8, label='Telluric corrected')
            ax.plot(self.wlm, self.data_container.intensity[self.brightest_fibres[1]], color="r",
                    label='Original', lw=3, alpha=0.8)
            ax.plot(self.wlm,
                    self.data_container.intensity[self.brightest_fibres[1]] * self.telluric_correction,
                    color="purple", alpha=1, lw=.8, label='Telluric corrected')
            ax.set_ylim(np.nanpercentile(self.data_container.intensity[self.brightest_fibres[[0, 1]]], 1),
                        np.nanpercentile(self.data_container.intensity[self.brightest_fibres[[0, 1]]], 99))
        ax.legend(ncol=2)
        ax.axvline(x=wave_min, color='k', linestyle='--')
        ax.axvline(x=wave_max, color='k', linestyle='--')
        ax.set_xlim(self.wlm[0] - 10, self.wlm[-1] + 10)
        ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
        if exclude_wlm is not None:
            for i in range(len(exclude_wlm)):
                ax.axvspan(exclude_wlm[i][0], exclude_wlm[i][1], color='c', alpha=0.1)
        ax.minorticks_on()
        if kwargs.get('plot', False):
            plt.show()
        else:
            plt.close(fig)
        return fig

    def apply(self, rss, verbose=True, is_combined_cube=False, update=True):
        """
        Apply the telluric correction to the input data.

        Parameters
        ----------
        rss : array
            The input data to correct.
        verbose : bool, optional
            Controls verbosity of logging messages (default is True).
        is_combined_cube : bool, optional
            Whether the input is a combined cube (default is False).
        update : bool, optional
            Whether to update the correction (default is True).

        Returns
        -------
        rss_out : array
            The corrected data.
        """
        self.verbose = verbose

        # Check wavelength
        if not rss.wavelength.size == self.wlm.size or not np.allclose(rss.wavelength, self.wlm, equal_nan=True):
            self.corr_vprint("Interpolating correction to input wavelength")
            self.interpolate_model(rss.wavelength, update=update)

            
        # Copy input RSS for storage the changes implemented in the task
        rss_out = copy.deepcopy(rss)
        self.corr_print("Applying telluric correction to this star...")
        rss_out.intensity *= self.telluric_correction
        rss_out.variance *= self.telluric_correction**2
        self.log_correction(rss, status='applied')
        return rss_out

    def interpolate_model(self, wavelength, update=True):
        """
        Interpolate the telluric correction model to match the input wavelength array.

        Parameters
        ----------
        wavelength : array
            The wavelength array to interpolate.
        update : bool, optional
            Whether to update the correction (default is True).

        Returns
        -------
        telluric_correction : array
            The interpolated telluric correction.
        """
        telluric_correction = np.interp(wavelength, self.wlm, self.telluric_correction, left=1, right=1)
        if update:
            self.teluric_correction = telluric_correction
            self.wlm = wavelength
        return telluric_correction

    def save(self, filename='telluric_correction.txt', **kwargs):
        """
        Save the telluric correction to a text file.

        Parameters
        ----------
        filename : str, optional
            The name of the output file.
        """
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
    
        
# Mr Krtxo \(ﾟ▽ﾟ)/
