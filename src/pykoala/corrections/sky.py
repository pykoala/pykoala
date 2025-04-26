"""
Module containing the corrections for estimating and correcting, and subtracting
the contribution of the sky emission in DataContainers.
"""
# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import copy
import os
from time import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import scipy
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.table import QTable
from astropy import stats
from astropy import units as u
import scipy.ndimage
import scipy.signal

# =============================================================================
# PyKOALA
# =============================================================================
from pykoala import vprint
from pykoala.plotting.utils import new_figure, plot_image, plot_fibres
from pykoala.corrections.correction import CorrectionBase
from pykoala.corrections.throughput import Throughput
from pykoala.corrections.wavelength import WavelengthOffset
from pykoala.data_container import DataContainer, RSS
from pykoala.ancillary import check_unit, symmetric_background


#TODO: Move to a math module
class BackgroundEstimator(object):
    """
    Class for estimating background and its dispersion using different statistical methods.
    """

    @staticmethod
    def mean(rss_intensity):
        """
        Compute the mean and standard deviation of the spectra.

        Parameters
        ----------
        rss_intensity : np.ndarray
            Input array (n_fibres x n_wavelength).

        Returns
        -------
        background : np.ndarray
            The computed background (mean) of the data.
        background_sigma : np.ndarray
            The dispersion (standard deviation) of the data.
        """
        background = np.nanmean(rss_intensity, axis=0)
        background_sigma = np.nanstd(rss_intensity, axis=0)
        return background, background_sigma

    @staticmethod
    def percentile(rss_intensity, percentiles=[16, 50, 84]):
        """
        Compute the background and dispersion from specified percentiles.

        Parameters
        ----------
        rss_intensity : np.ndarray
            Input array (n_fibres x n_wavelength).
        percentiles : list of float, optional
            The percentiles to use for computation. Default is [16, 50, 84].

        Returns
        -------
        background : np.ndarray
            The computed background (median) of the data.
        background_sigma : np.ndarray
            The dispersion (half the interpercentile range) of the data.
        """
        plow, background, pup = np.nanpercentile(
            rss_intensity, percentiles, axis=0)
        background_sigma = (pup - plow) / 2
        return background, background_sigma

    @staticmethod
    def mad(rss_intensity):
        """
        Estimate the background from the median and the dispersion from
        the Median Absolute Deviation (MAD) along the given axis.

        Parameters
        ----------
        rss_intensity : np.ndarray
            Input array (n_fibres x n_wavelength).

        Returns
        -------
        background : np.ndarray
            The computed background (median) of the data.
        background_sigma : np.ndarray
            The dispersion (scaled MAD) of the data.
        """
        background = np.nanmedian(rss_intensity, axis=0)
        mad = np.nanmedian(np.abs(rss_intensity - background[np.newaxis:]),
                           axis=0)
        background_sigma = 1.4826 * mad
        return background, background_sigma

    @staticmethod
    def biweight(rss_intensity):
        """
        Estimate the background and dispersion using the biweight method.

        Parameters
        ----------
        rss_intensity : np.ndarray
            Input array (n_fibres x n_wavelength).

        Returns
        -------
        background : np.ndarray
            Computed background (location) of the data.
        background_sigma : np.ndarray
            Dispersion (scale) of the data.
        """
        background = stats.biweight_location(rss_intensity, axis=0)
        background_sigma = stats.biweight_scale(rss_intensity, axis=0)
        return background, background_sigma

    @staticmethod
    def mode(rss_intensity):
        """
        Estimate the background and dispersion using the mode.

        Parameters
        ----------
        rss_intensity : np.ndarray
            Input array (n_fibres x n_wavelength).

        Returns
        -------
        background : np.ndarray
            The computed background sky.
        background_sigma : np.ndarray
            Scaled MAD of the data behind the mode.
        """
        mode = np.empty_like(rss_intensity[0])
        sigma = np.empty_like(mode)
        for wavelength_index in range(mode.size):
            fibre_flux = rss_intensity[:, wavelength_index]
            mode[wavelength_index], dummy = symmetric_background(fibre_flux)
            below_mode = fibre_flux < mode[wavelength_index]
            sigma[wavelength_index] = 1.4826 * np.nanmedian(
                mode[wavelength_index] - fibre_flux[below_mode])

        return mode, sigma

#TODO : Move to a math module
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
    def percentile_continuum(data, percentile, window_size=5):
        """
        Estimate the continuum using a percentile filter.

        Parameters
        ----------
        data : np.ndarray
            The input data array for which to compute the continuum.
        percentile : list or tuple
            The percentiles (0-100) to use for the continuum estimation.
        window_size : int, optional
            The size of the window over which to compute the percentile filter. Default is 5.

        Returns
        -------
        continuum : np.ndarray
            The estimated continuum of the input data.
        """
        continuum = scipy.ndimage.percentile_filter(data, percentile,
                                                    window_size)
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

    default_min_separation = 10

    @classmethod
    def lower_envelope(self, x, y, min_separation=None):
        '''
        #TODO --> Refactor
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

#TODO: Documentation
class ContinuumModel:

    def __init__(self, dc: DataContainer, min_separation=None):
        self.update(dc, min_separation)

    # TODO: Don't assume RSS format (intensity[spec_id, wavelength])
    #       def update(self, dc:DataContainer, min_separation=None):
    def update(self, dc: RSS, min_separation=None):
        n_spectra = dc.intensity.shape[0]
        self.intensity = np.zeros_like(dc.intensity)
        self.scale = np.zeros(n_spectra)

        print(f"> Find continuum for {n_spectra} spectra:")
        t0 = time()

        for i in range(n_spectra):
            self.intensity[i], self.scale[i] = ContinuumEstimator.lower_envelope(
                dc.wavelength, dc.intensity[i], min_separation)

        print(f"  Done ({time()-t0:.3g} s)")
        self.strong_sky_lines = self.detect_lines(dc)

    def detect_lines(self, dc: DataContainer, n_sigmas=3):
        SN = (dc.intensity - self.intensity) / self.scale[:, np.newaxis]
        SN_p16, SN_p50, SN_p84 = np.nanpercentile(SN, [16, 50, 84], axis=0)

        line_mask = SN_p16 - (n_sigmas-1)*(SN_p84-SN_p16)/2 > 0
        line_mask[0] = False
        line_mask[-1] = False

        line_left = np.where(~line_mask[:-1] & line_mask[1:])[0]
        line_right = np.where(line_mask[:-1] & ~line_mask[1:])[0]
        line_right += 1
        print(f'  {line_left.size} strong sky lines ({np.count_nonzero(line_mask)} out of {dc.wavelength.size} wavelengths)')
        return QTable((line_left, line_right), names=('left', 'right'))


# =============================================================================
# Line Spread Function
# =============================================================================
#TODO: Documenation and move to spectra module
class LSF_estimator:

    def __init__(self, wavelength_range, resolution):
        self.delta_lambda = np.linspace(-wavelength_range, wavelength_range, int(
            np.ceil(2 * wavelength_range / resolution)) + 1)

    def find_LSF(self, line_wavelengths, wavelength, intensity):
        line_spectra = np.zeros(
            (len(line_wavelengths), self.delta_lambda.size))
        for i, line_wavelength in enumerate(line_wavelengths):
            sed = np.interp(line_wavelength + self.delta_lambda,
                            wavelength, intensity)
            line_spectra[i] = self.normalise(sed)
        return self.normalise(np.nanmedian(line_spectra, axis=0))

    def normalise(self, x):
        x -= stats.biweight.biweight_location(x)  # subtract continuum
        # x -= np.median(x)
        norm = x[x.size//2]  # normalise at centre
        if norm > 0:
            x /= norm
        else:
            x *= np.nan
        return x

    def find_FWHM(self, profile):
        threshold = np.max(profile)/2
        left = np.max(
            self.delta_lambda[(self.delta_lambda < 0) & (profile < threshold)])
        right = np.min(
            self.delta_lambda[(self.delta_lambda > 0) & (profile < threshold)])
        return right-left


# =============================================================================
# Sky lines library
# =============================================================================
# TODO : this should be an instance of EMission lines in the spectra module
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
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f'Directory {data_path} not in {os.listdir(os.path.join(os.path.dirname(__file__), "..", "input_data", "sky_lines"))}')
    
    # Initialize arrays to store line properties
    line_wavelength = np.empty(0)
    line_fwhm = np.empty(0)
    line_flux = np.empty(0)

    # Read data from each file
    for p in prefix:
        file = os.path.join(data_path, f"gident_{p}.tfits")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File '{file}' could not be found in {os.listdir(data_path)}")

        with fits.open(file) as f:
            wave = f[1].data['LAMBDA_AIR']
            fwhm = f[1].data['FWHM']
            flux = f[1].data['FLUX']

            line_wavelength = np.hstack((line_wavelength, wave))
            line_fwhm = np.hstack((line_fwhm, fwhm))
            line_flux = np.hstack((line_flux, flux))

    line_fwhm = line_fwhm << u.angstrom
    line_wavelength = line_wavelength << u.angstrom
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
    continuum : np.ndarray
        Array representing the continuuum emission of the sky model.
    sky_lines: np.ndarray
        1-D array representing the wavelength of a collection of sky emission
        lines expressed in angstrom. This is used for fitting the emission lines
        from the (continuum-substracted) intensity.
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
    sky_lines = None

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
        self.wavelength = check_unit(kwargs.get('wavelength', None), u.angstrom)
        self.intensity = check_unit(kwargs.get('intensity', None))
        self.variance = check_unit(kwargs.get('variance', None))
        self.continuum = check_unit(kwargs.get('continuum', None))

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
            skymodel_var = self.variance[:, np.newaxis, np.newaxis]
        elif data.ndim == 2 and self.intensity.ndim == 1:
            skymodel_intensity = self.intensity[np.newaxis, :]
            skymodel_var = self.variance[np.newaxis, :]
        elif data.ndim == 2 and self.intensity.ndim == 2:
            skymodel_intensity = self.intensity
            skymodel_var = self.variance
        else:
            vprint(
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
        if self.intensity is not None:
            if hasattr(ContinuumEstimator, cont_estimator):
                estimator = getattr(ContinuumEstimator, cont_estimator)
            else:
                raise NameError(f"{cont_estimator} does not correspond to any"
                                + "available continuum method")
            self.continuum = estimator(self.intensity, **cont_estimator_args)
            self.intensity -= self.continuum
        else:
            raise AttributeError("Sky model intensity has not been computed")

    # TODO: This should make use of the methods in the "spectra" module
    def fit_emission_lines(self, window_size=100,
                           resampling_wave=0.1, **fit_kwargs):
        """
        Fit emission lines to the continuum-subtracted spectrum.

        Parameters
        ----------
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
        assert self.intensity is not None, "Sky Model intensity is None"

        if self.continuum is None:
            vprint("Sky Model intensity might contain continuum emission"
                   " leading to unsuccessful emission line fit")
        if self.variance is None:
            errors = np.ones_like(self.intensity, dtype=float)

        finite_mask = np.isfinite(self.intensity)
        p0_amplitude = np.interp(self.sky_lines, self.wavelength[finite_mask],
                                 self.intensity[finite_mask])
        p0_amplitude = np.clip(p0_amplitude, a_min=0, a_max=None)
        fit_g = fitting.LevMarLSQFitter()
        emission_model = models.Gaussian1D(amplitude=0, mean=0, stddev=0)
        emission_spectra = np.zeros_like(self.wavelength)
        wavelength_windows = np.arange(self.wavelength.min(),
                                       self.wavelength.max(), window_size)
        wavelength_windows[-1] = self.wavelength.max()
        vprint(f"Fitting all emission lines ({self.sky_lines.size})"
                    + " to continuum-subtracted sky spectra")
        for wl_min, wl_max in zip(wavelength_windows[:-1], wavelength_windows[1:]):
            vprint(f"Starting fit in the wavelength range [{wl_min:.1f}, "
                   + f"{wl_max:.1f}]")
            mask_lines = (self.sky_lines >= wl_min) & (self.sky_lines < wl_max)
            mask = (self.wavelength >= wl_min) & (
                self.wavelength < wl_max) & finite_mask
            wave = np.arange(self.wavelength[mask][0],
                             self.wavelength[mask][-1], resampling_wave)
            obs = np.interp(wave, self.wavelength[mask], self.intensity[mask])
            err = np.interp(wave, self.wavelength[mask], errors[mask])
            if mask_lines.any():
                vprint(f"> Line to Fit {self.sky_lines[mask_lines][0]:.1f}")
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
                    vprint(f"Line to Fit {line:.1f}")
                    model = models.Gaussian1D(
                        amplitude=p0, mean=line, stddev=sigma,
                        bounds={'amplitude': (p0 * 0.5, p0 * 10), 'mean': (line - 5, line + 5),
                                'stddev': (sigma / 2, 5)}
                    )
                    window_model += model
                g = fit_g(window_model, wave, obs,
                          weights=1 / err, **fit_kwargs)
                emission_spectra += g(self.wavelength)
                emission_model += g
        return emission_model, emission_spectra

    # TODO: This should create an instance of EmissionLines
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

        """
        if path_to_table is not None:
            vprint(f"Loading input sky line table {path_to_table}")
            path_to_table = os.path.join(os.path.dirname(__file__),
                                         'input_data', 'sky_lines',
                                         path_to_table)
            self.sky_lines, self.sky_lines_fwhm, self.sky_lines_f = np.loadtxt(
                path_to_table, usecols=(0, 1, 2), unpack=True, **kwargs)
        else:
            vprint("Loading UVES sky line table")
            self.sky_lines, self.sky_lines_fwhm, self.sky_lines_f = uves_sky_lines()
        # Select only those lines within the wavelength range of the model
        common_lines = (self.sky_lines >= self.wavelength[0]) & (
            self.sky_lines <= self.wavelength[-1])
        self.sky_lines = self.sky_lines[common_lines]
        self.sky_lines_fwhm = self.sky_lines_fwhm[common_lines]
        self.sky_lines_f = self.sky_lines_f[common_lines]
        vprint(f"Total number of sky lines: {self.sky_lines.size}")
        # Blend sky emission lines
        delta_lambda = self.wavelength[1] - self.wavelength[0]
        vprint("Blending sky emission lines according to"
                + f"wavelength resolution ({delta_lambda} AA)")
        unresolved_lines = np.where(np.diff(self.sky_lines) <= delta_lambda)[0]
        while len(unresolved_lines) > 0:
            self.sky_lines[unresolved_lines] = (
                self.sky_lines[unresolved_lines] + self.sky_lines[unresolved_lines + 1]) / 2
            self.sky_lines_fwhm[unresolved_lines] = np.sqrt(
                self.sky_lines_fwhm[unresolved_lines]**2 +
                self.sky_lines_fwhm[unresolved_lines + 1]**2
            )
            self.sky_lines_f[unresolved_lines] = (
                self.sky_lines_f[unresolved_lines] +
                self.sky_lines_f[unresolved_lines + 1]
            )
            self.sky_lines = np.delete(self.sky_lines, unresolved_lines)
            self.sky_lines_fwhm = np.delete(
                self.sky_lines_fwhm, unresolved_lines)
            self.sky_lines_f = np.delete(self.sky_lines_f, unresolved_lines)
            unresolved_lines = np.where(
                np.diff(self.sky_lines) <= delta_lambda)[0]
        vprint(f"Total number of sky lines after blending: {self.sky_lines.size}")
        # Remove faint lines
        # self.vprint(f"Selecting the  sky lines after blending: {self.sky_lines.size}")
        faint = np.where(self.sky_lines_f < np.nanpercentile(
            self.sky_lines_f, lines_pct))[0]
        self.sky_lines = np.delete(self.sky_lines, faint)
        self.sky_lines_fwhm = np.delete(self.sky_lines_fwhm, faint)
        self.sky_lines_f = np.delete(self.sky_lines_f, faint)

    def plot_sky_model(self, show=False, fig_name='sky_model'):
        """Plot the sky model

        Parameters
        ----------
        show : bool
            Show the resulting plot. Default is False.

        Returns
        -------
        fig : :class:`matplotlib.pyplot.Figure`
            Figure containing the Sky Model plot.
        """
        if self.intensity is None:
            return None
        
        elif self.intensity.ndim == 1:
            fig, axes = new_figure(fig_name)
            if 'dc' in self.__dict__:
                title = f"1-D Sky Model for {self.dc.info['name']}"
            else:
                title = '1-D Sky Model'
            fig.suptitle(title)
            ax = axes[0, 0]
            ax.plot(self.wavelength, self.intensity,
                    color='b', alpha=0.5, label='sky intensity')
            if self.variance is not None:
                ax.fill_between(self.wavelength, self.intensity - self.variance**0.5,
                                self.intensity + self.variance**0.5, color='b',
                                alpha=0.2, label='uncertainty')
            ax.set_ylim(np.nanpercentile(self.intensity, [1, 99]).value)
            ax.set_ylabel(f"Intensity [{self.intensity.unit}]")
            ax.set_xlabel(f"Wavelength [{self.wavelength.unit}]")
            ax.legend()

        elif self.intensity.ndim == 2:
            fig = plt.figure(fig_name, constrained_layout=True)
            im_args = dict(
                interpolation='none', aspect='auto', origin="lower",
                extent=(1, self.intensity.shape[0],
                        self.wavelength[0], self.wavelength[-1]))
            ax = fig.add_subplot(121, title='2-D Sky Model Intensity')
            mappable = ax.imshow(self.intensity, **im_args)
            plt.colorbar(mappable, ax=ax)
            ax = fig.add_subplot(122, title='2-D Sky Model STD')
            mappable = ax.imshow(self.variance**0.5, **im_args)
            plt.colorbar(mappable, ax=ax)

        #if show:
        #    plt.show(fig_name)
        #else:
        if not show:
            plt.close(fig_name)
        return fig


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
            Data container used to estimate the sky.
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
        #TODO: This should create a 2D model that uses the median sky
        # but shifts it to the posiotoin given on each individual fibre
        self.intensity, self.variance = BackgroundEstimator.percentile(
            self.dc.rss_intensity, percentiles=[16, 50, 84])
        self.intensity, self.variance = (
            self.intensity / self.exptime,
            self.variance / self.exptime)


class SkyFromObject(SkyModel):
    """
    Sky model based on a single Data Container.

    This class builds a sky emission model using the data
    from a given Data Container that includes the contribution
    of an additional source (i.e. star/galaxy).

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

    def __init__(self, dc,
                 bckgr_estimator='mad', bckgr_params=None,
                 sky_fibres=None, source_mask_nsigma=None,
                 remove_cont=False,
                 cont_estimator='median', cont_estimator_args=None,
                 qc_plots={'show': True}):
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
        sky_fibres : None, 'auto', 'all', or list of 1D indices (optional)
            Default is None, which will read the sky fibre list from
            the `info` attribute; if absent or empty, it will trigger 'auto'.
            The 'auto' option will call `estimate_sky_fibres` to identify
            sky fibres based on a mean intensity threshold, automatically
            determined from the shape of the normalised spectra.
            Selecting `all` will use every spectrum in the `DataContainer`.
            Othrwise, this argument will be interpreted as a list of
            integer indices identifying sky spectra, assuming 1D RSS order.
        source_mask_nsigma : float, optional
            Sigma level for masking sources. Default is None.
        remove_cont : bool, optional
            If True, the continuum will be removed. Default is False.
        cont_estimator : str, optional
            Method to estimate the continuum signal. Default is 'median'.
        cont_estimator_args : dict, optional
            Arguments for the continuum estimator. Default is None.
        qc_plots: dict
            Dictionary to control QC plots.
        """
        vprint("Creating SkyModel from input Data Container")
        self.dc = dc
        # self.exptime = dc.info['exptime']
        #vprint(f"Estimating sky background contribution from the {bckgr_estimator}...")
        
        self.qc_plots = {}

        bckg, bckg_sigma = self._estimate_background(
            bckgr_estimator, bckgr_params, sky_fibres, source_mask_nsigma, qc_plots)
        super().__init__(wavelength=self.dc.wavelength,
                         intensity=bckg,
                         variance=bckg_sigma**2)
        if remove_cont:
            vprint("Removing background continuum")
            self.remove_continuum(cont_estimator, cont_estimator_args)
        
        if len(qc_plots) > 0:
            show_plot = qc_plots.get('show', True)
            plot_filename = qc_plots.get('filename_base', None)
            fig_name = 'sky_model'
            fig = self.plot_sky_model(show=show_plot, fig_name=fig_name)
            ax = fig.axes[0]
            p16, p50, p84 = np.nanpercentile(self.dc.rss_intensity,
                                             [16, 50, 84], axis=0)
            ax.plot(self.dc.wavelength, p50, 'k-', alpha=.1)
            ax.fill_between(self.dc.wavelength, p16, p84,
                            color='k', alpha=.1,
                            label=f"{self.dc.info['name']} (all fibres)")
            if 'sky_CASU' in self.dc.info:
                ax.plot(self.dc.wavelength, self.dc.info['sky_CASU'],
                        'r-', alpha=.5, label='WEAVE pipeline')
            ax.legend()
            self.qc_plots[fig_name] = fig
            if plot_filename is not None:
                fig.savefig(f'{plot_filename}_{fig_name}.png')
            if not show_plot:
                plt.close(fig_name)

    def _estimate_background(self, bckgr_estimator, bckgr_params=None,
                            sky_fibres=None, source_mask_nsigma=None,
                            qc_plots={}):
        """
        Estimate the background.

        Parameters
        ----------
        bckgr_estimator : str
            Background estimator method. Available methods: 'mad', 'percentile'.
        bckgr_params : dict, optional
            Parameters for the background estimator. Default is None.
        sky_fibres : None, 'auto', 'all', or list of 1D indices (optional)
            Default is None, which will read the sky fibre list from
            the `info` attribute; if absent or empty, it will trigger 'auto'.
            The 'auto' option will call `estimate_sky_fibres` to identify
            sky fibres based on a mean intensity threshold, automatically
            determined from the shape of the normalised spectra.
            Selecting `all` will use every spectrum in the `DataContainer`.
            Othrwise, this argument will be interpreted as a list of
            integer indices identifying sky spectra, assuming 1D RSS order.
        source_mask_nsigma : float, optional
            Sigma level for masking sources. Default is None.
        qc_plots: dict
            Dictionary to control QC plots.

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
            raise NameError(
                f"Input background estimator {bckgr_estimator} does not exist")
        
        # Determine sky fibres:
        
        self.sky_fibres = sky_fibres
        if sky_fibres is None:
            self.sky_fibres = self.dc.info.get('sky_fibres', None)
            if self.sky_fibres is None or len(self.sky_fibres) == 0:
                sky_fibres = 'auto'
        if sky_fibres == "auto":
            self.sky_fibres = self._estimate_sky_fibres()
        elif sky_fibres == "all":
            self.sky_fibres = np.arange(int(self.dc.n_spectra))
        #bckgr_params["axis"] = bckgr_params.get("axis", 0)
        #data = np.take(self.dc.rss_intensity, self.sky_fibres, bckgr_params["axis"])
        data = self.dc.rss_intensity[self.sky_fibres]
        
        if (len(qc_plots) > 0) and (self.dc.intensity.ndim == 2):
            show_plot = qc_plots.get('show', True)
            plot_filename = qc_plots.get('filename_base', None)
            fig_name = 'sky_fibres'
            fig, axes = new_figure(fig_name, figsize=(8, 6))
            fig.suptitle(f"{self.dc.info['name']} {fig_name}")
            total_flux = np.nanmean(self.dc.rss_intensity.value, axis=1)
            ax, patch_collection, cb = plot_fibres(
                fig, axes[0, 0], self.dc, data=total_flux,
                cblabel=f'mean intensity [{self.dc.rss_intensity.unit}]',
                cmap='Spectral_r',
                norm=Normalize(vmax=np.nanmean(total_flux)))
            handle, = ax.plot(self.dc.info['fib_ra'][self.sky_fibres],
                              self.dc.info['fib_dec'][self.sky_fibres],
                              'k+', label=f'{self.sky_fibres.size} sky fibres')
            ax.legend(handles=[handle])
            if plot_filename is not None:
                fig.savefig(f'{plot_filename}_{fig_name}.png')
            self.qc_plots[fig_name] = fig
            #if show_plot:
            #    plt.show(fig_name)
            #else:
            if not show_plot:
                plt.close(fig_name)

        # Estimate sky spectrum:
        
        if source_mask_nsigma is not None:
            vprint("Pre-estimating background using all data")
            bckgr, bckgr_sigma = estimator(data, **bckgr_params)
            vprint(
                f"Applying sigma-clipping mask (n-sigma={source_mask_nsigma})")
            dims_to_expand = (0)
            source_mask = (data > np.expand_dims(bckgr, dims_to_expand) +
                           source_mask_nsigma
                           * np.expand_dims(bckgr_sigma, dims_to_expand))
            data[source_mask] = np.nan

        vprint(f"Applying the {bckgr_estimator} estimator to {data.shape[0]} sky fibres")
        bckgr, bckgr_sigma = estimator(data, **bckgr_params)
        return bckgr, bckgr_sigma

    def _estimate_sky_fibres(self):
        """
        Identify sky fibres by imposing a mean intensity threshold, based on
        the shape of the normalised spectra.
        """
        
        total_flux = np.nanmean(self.dc.rss_intensity, axis=1).value
        sorted_by_flux = np.argsort(total_flux)
        n = 1 + np.arange(sorted_by_flux.size)
        half_sample = sorted_by_flux.size // 2

        flux_mean = np.nancumsum(total_flux[sorted_by_flux]) / n

        norm_fibre = self.dc.rss_intensity.value[sorted_by_flux, :] / total_flux[sorted_by_flux][:, np.newaxis]
        norm_sum1 = np.nancumsum(norm_fibre, axis=0)
        norm_sum2 = np.nancumsum(norm_fibre**2, axis=0)

        left_side_mean = norm_sum1[1:half_sample] / n[1:half_sample, np.newaxis]
        right_side_mean = norm_sum1[3::2] / n[1:half_sample, np.newaxis] - left_side_mean

        left_side_err = norm_sum2[1:half_sample] / n[1:half_sample, np.newaxis]
        right_side_err = norm_sum2[3::2] / n[1:half_sample, np.newaxis] - left_side_err

        left_side_err -= left_side_mean**2
        left_side_err[~(left_side_err > 0)] = np.nan
        left_side_err = np.sqrt(left_side_err / n[:half_sample-1, np.newaxis])  # n-1

        right_side_err -= right_side_mean**2
        right_side_err[~(right_side_err > 0)] = np.nan
        right_side_err = np.sqrt(right_side_err / n[:half_sample-1, np.newaxis])  # n-1

        weight = np.exp(np.nanmean(
            -(right_side_mean - left_side_mean)**2 * (1/right_side_err**2 + 1/left_side_err**2),
            axis=1))
        sky_flux = np.nansum(total_flux[sorted_by_flux][3::2] * weight) / np.nansum(weight)

        n_sky = min(
            np.searchsorted(flux_mean, sky_flux),
            2 * np.searchsorted(total_flux[sorted_by_flux], sky_flux))
        flux_threshold = total_flux[sorted_by_flux[n_sky-1]]
        sky_fibres = sorted_by_flux[:n_sky]
        vprint(f'{n_sky} sky fibres found below {flux_threshold:.5g} (sky flux = {sky_flux:.5g}) {self.dc.rss_intensity.unit}')
        
        return sky_fibres

    def plot_individual_wavelength(self, wavelength):
        """
        Identify sky fibres by imposing a mean intensity threshold, based on
        the shape of the normalised spectra.
        """
        idx = np.searchsorted(self.dc.wavelength, check_unit(wavelength, u.Angstrom)) - 1

        intensity = self.dc.rss_intensity[:, idx].value
        total_flux = np.nanmean(self.dc.rss_intensity, axis=1)
        intensity_norm = intensity / total_flux

        sorted_by_flux = np.argsort(total_flux)
        flux_mean = np.nancumsum(total_flux[sorted_by_flux]) / np.arange(1, total_flux.size+1)
        half_sample = total_flux.size // 2
        
        fig, axes = new_figure('single_wavelength_sky', nrows=2)

        ax = axes[0, 0]
        ax.set_ylabel(f'intensity ($\\lambda={self.dc.wavelength[idx]:.2f}\\ \\AA$)')
        vmin = np.nanmin(intensity[self.sky_fibres])
        vmax = np.nanmax(intensity[self.sky_fibres])
        h = .1 * (vmax - vmin)
        ax.set_ylim(vmin-h, vmax+2*h)

        ax.plot(total_flux, intensity, 'k.', alpha=.1)
        ax.plot(total_flux[self.sky_fibres], intensity[self.sky_fibres], 'b+', alpha=.5)
        ax.axhline(self.intensity[idx], c='k', ls='--')
        std = np.sqrt(self.variance[idx])
        ax.axhline(self.intensity[idx] + std, c='k', ls=':')
        ax.axhline(self.intensity[idx] - std, c='k', ls=':')


        ax = axes[1, 0]
        ax.set_ylabel(f'normalised intensity')
        vmin = np.nanmin(intensity_norm[self.sky_fibres])
        vmax = np.nanmax(intensity_norm[self.sky_fibres])
        h = .1 * (vmax - vmin)
        ax.set_ylim(vmin-h, vmax+2*h)

        ax.plot(total_flux, intensity_norm, 'k.', alpha=.1)
        ax.plot(total_flux[self.sky_fibres], intensity_norm[self.sky_fibres], 'b+', alpha=.5)
        sky_flux = np.nanmean(self.intensity.value)
        ax.axhline(self.intensity.value[idx]/sky_flux, c='k', ls='--')
        ax.axhline((self.intensity[idx] + std).value/sky_flux, c='k', ls=':')
        ax.axhline((self.intensity[idx] - std).value/sky_flux, c='k', ls=':')

        ax.set_xlabel('total flux (mean fibre intensity)')
        vmin, vmax = (flux_mean[0], flux_mean[-1])
        h = .1 * (vmax - vmin)
        ax.set_xlim(vmin-h, vmax+2*h)

        return fig

# =============================================================================
# Sky Substraction Correction
# =============================================================================


class SkySubsCorrection(CorrectionBase):
    """
    Correction for removing sky emission from a DataContainer.

    This class applies sky emission correction to a `DataContainer`
    using a provided sky model.
    It supports both standard and PCA-based sky subtraction methods
    and can generate visualizations of the correction process.

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

    def __init__(self, skymodel, **correction_kwargs):
        """
        Initialize the SkySubsCorrection with a given sky model.

        Parameters
        ----------
        skymodel : SkyModel
            The sky model to be used for sky emission subtraction.
        """
        super().__init__(**correction_kwargs)
        self.skymodel = skymodel

    def plot_correction(self, data_cont, data_cont_corrected, **kwargs):
        """
        Plot the original and sky-corrected intensity of a DataContainer for comparison.

        Parameters
        ----------
        data_cont : :class:`pykoala.data_container.DataContainer`
            The original DC before sky correction.
        data_cont_corrected : :class:`pykoala.data_container.DataContainer`
            The DC after sky correction.
        kwargs : dict
            Additional keyword arguments for `imshow`.

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            The figure object containing the plots.
        """
        
        if data_cont.intensity.ndim == 2:
            original_image = data_cont.intensity
            corr_image = data_cont_corrected.intensity
        elif data_cont.intensity.ndim == 3:
            original_image = data_cont.get_white_image()
            data_unit = str(original_image.unit)
            original_image = original_image
            corr_image = data_cont_corrected.get_white_image()

        norm = plt.Normalize(np.nanpercentile(original_image.value, 1),
                             np.nanpercentile(original_image.value, 99))
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),
                                constrained_layout=True,
                                sharex=True, sharey=True)

        plot_image(fig, axs[0], data=original_image, norm=norm)
        plot_image(fig, axs[1], data=corr_image, norm=norm)
        axs[0].set_title("Original")
        axs[1].set_title("Sky emission subtracted")

        if kwargs.get("show", False):
            plt.show()
        else:
            plt.close(fig)
        return fig

    def apply(self, dc, pca=False, plot=False, **plot_kwargs):
        """
        Apply the sky emission correction to the datacube.

        Parameters
        ----------
        dc : :class:`pykoala.data_container.DataContainer`
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
        dc_out : :class:`pykoala.data_container.DataContainer`
            The corrected datacube.
        fig : matplotlib.figure.Figure or None
            The figure object containing the plots if `plot` is True, otherwise None.
        """
        # Set verbosity
        # Copy input datacube to store the changes
        dc_out = copy.deepcopy(dc)

        self.vprint("Applying sky subtraction")

        if pca:
            dc_out.intensity, dc_out.variance = self.skymodel.substract_pca(
                dc_out.intensity, dc_out.variance)
        else:
            dc_out.intensity, dc_out.variance = self.skymodel.substract(
                dc_out.intensity, dc_out.variance)

        self.record_correction(dc_out, status='applied')
        if plot:
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
    _telluric_correction = None
    _wavelength = None
    default_model_file = os.path.join(os.path.dirname(__file__), '..',
                                      'input_data', 'sky_lines',
                                      'telluric_lines.txt')
    
    @property
    def telluric_correction(self):
        return self._telluric_correction

    @telluric_correction.setter
    def telluric_correction(self, value):
        self._telluric_correction = value

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = check_unit(value, u.angstrom)

    @property
    def airmass(self) -> float:
        """Airmass at which the model has been created."""
        return self._airmass
    
    @airmass.setter
    def airmass(self, value):
        self._airmass = value

    def __init__(self,
                 telluric_correction,
                 wavelength,
                 airmass,
                 telluric_correction_file="unknown",
                 **correction_kwargs):
        """
        Initializes the TelluricCorrection object.

        Parameters
        ----------
        telluric_correction : :class:`numpy.ndarray`
            Array containing the values of the telluric correction.
        wavelength : :class:`astropy.units.Quantity`
            Wavelength values associated to the ``telluric_correction``.
        airmass : float
            Airmass at which the model is estimated.
        telluric_correction_file : str, optional
            Path to the file that contains the model.
        **correction_kwargs:
            Additional arguments passed to :class:`CorrectionBase`.
        """
        super().__init__(**correction_kwargs)

        self.telluric_correction = telluric_correction
        self.wavelength = wavelength
        if self.telluric_correction.size != self.wavelength.size:
            raise ValueError("Size of telluric correction and input wavelength"
                             " vector are not compatible: "
                             f"{self.telluric_correction.size}, {self.wavelength.size}")
        self.airmass = airmass
        self.telluric_correction_file = telluric_correction_file

    @classmethod
    def from_text_file(cls, path):
        """Initialise a TelluricCorrection from a text file.
        
        Parameters
        ----------
        path : str
            Path to the text file containing the telluric correction. The first
            and second columns must contain the wavelength and telluric correction,
            respectively.

        Returns
        -------
        telluric_correction : :class:`TelluricCorrection`
            The telluric correction.
        """
        vprint("Initialising telluric correction from text file")
        wavelength, telluric_correction = np.loadtxt(path, unpack=True,
                                                     usecols=(0, 1))
        # Read the value of the airmass at which the model was computed
        with open(path, "r") as f:
            line = f.readline()
            if "airmass" in line:
                airmass = float(line.split("=")[1])
            else:
                raise ValueError("Telluric correction file must include"
                                 " airmass=value in the first line")
        return cls(telluric_correction=telluric_correction,
                   wavelength=wavelength,
                   telluric_correction_file=path, airmass=airmass)

    @classmethod
    def from_smoothed_spectra_container(cls, spectra_container,
                                        exclude_wlm=None, 
                                        light_percentile=0.95,
                                        median_window=10 << u.angstrom,
                                        wave_min=None, wave_max=None,
                                        plot=True):
        """
        Estimate the telluric correction function using the smoothed spectra of the input star.

        Parameters
        ----------
        spectra_container: `pykoala.data_container.SpectraContainer`
            SpectraContainer used to estimate the telluric absorption.
        exclude_wlm : list of lists, optional
            List of wavelength ranges to exclude from correction (default is None).
        light_percentile: float, optional
            Percentile of light used to estimate the absorption. Default is 0.95.
        median_window : int, optional
            Step size for smoothing (default is 10).
        wave_min : float, optional
            Minimum wavelength to consider (default is None).
        wave_max : float, optional
            Maximum wavelength to consider (default is None).
        plot : bool, optional
            Whether to plot the correction (default is True).

        Returns
        -------
        telluric_correction : `TelluricCorrection`
            The computed telluric correction.
        fig : :class:`matplotlib.pyplot.Figure` or ``None``
            If ``plot=True``, it corresponds to a quality control plot.
        """
        vprint("Initialising telluric correction from input STD star")
        spectra = np.nanpercentile(spectra_container.rss_intensity,
                                   light_percentile, axis=0)
        telluric_correction = np.ones(spectra_container.wavelength.size)
        if wave_min is None:
            wave_min = spectra_container.wavelength[0]
        if wave_max is None:
            wave_max = spectra_container.wavelength[-1]
        if exclude_wlm is None:
            exclude_wlm = np.array([[6450 , 6700], [6850, 7050], [7130, 7380]]
                                   ) << u.angstrom
        # Mask containing the spectral points to include in the telluric correction
        correct_mask = (spectra_container.wavelength >= wave_min) & (
                        spectra_container.wavelength <= wave_max)
        # Mask user-provided spectral regions
        spec_windows_mask = np.ones_like(correct_mask, dtype=bool)
        for window in exclude_wlm:
            spec_windows_mask[(spectra_container.wavelength >= window[0]) &
                              (spectra_container.wavelength <= window[1])
                              ] = False
        # Master mask used to compute the Telluric correction
        telluric_mask = correct_mask & spec_windows_mask
        if not median_window % 2:
            median_window += 1
        smooth_med_star = ContinuumEstimator.medfilt_continuum(spectra,
                                                               median_window)
        telluric_correction[telluric_mask] = (smooth_med_star[telluric_mask]
                                              / spectra[telluric_mask])

        if plot:
            fig = TelluricCorrection.plot_correction(
                spectra_container, telluric_correction,
                wave_min=wave_min, wave_max=wave_max, exclude_wlm=exclude_wlm)
        else:
            fig = None
        return cls(telluric_correction=telluric_correction,
                   wavelength=spectra_container.wavelength,
                   airmass=spectra_container.info["airmass"]), fig

    @classmethod
    def from_model(cls, spectra_container, model_file=None,
                   width=30, extra_mask=None, plot=False):
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
        telluric_correction : `TelluricCorrection`
            The computed telluric correction.
        fig : :class:`matplotlib.pyplot.Figure` or ``None``
            If ``plot=True``, it corresponds to a quality control plot.
        """
        if model_file is None:
            model_file = TelluricCorrection.default_model_file

        width = check_unit(width, u.angstrom)
        w_l_1, w_l_2 = np.loadtxt(model_file, unpack=True, usecols=(0, 1))
        w_l_1 = w_l_1 << u.angstrom
        w_l_2 = w_l_2 << u.angstrom
        # Mask telluric regions
        mask = np.ones(spectra_container.wavelength.size, dtype=bool)
        telluric_correction = np.ones(spectra_container.wavelength.size,
                                      dtype=float)
        for b, r in zip(w_l_1, w_l_2):
            mask[(spectra_container.wavelength >= b - width) & (
                spectra_container.wavelength <= r + width)] = False
        if extra_mask is not None:
            mask = mask & extra_mask
        std = np.nanstd(spectra_container.rss_intensity, axis=0)
        stellar = np.interp(spectra_container.wavelength,
                            spectra_container.wavelength[mask], std[mask])
        telluric_correction[~mask] = stellar[~mask] / std[~mask]
        telluric_correction = np.clip(telluric_correction, a_min=1, a_max=None)
        if plot:
            fig = TelluricCorrection.plot_correction(
                spectra_container, telluric_correction,
                exclude_wlm=np.vstack((w_l_1 - width, w_l_2 + width)).T,
                wave_min=spectra_container.wavelength[0],
                wave_max=spectra_container.wavelength[-1])
        else:
            fig = None
        return cls(telluric_correction=telluric_correction,
                   wavelength=spectra_container.wavelength,
                   airmass=spectra_container.info["airmass"]), fig

    @classmethod
    def flag_data_container(cls, data_container, telluric_correction=None,
                            wavelength=None, path_to_model=None,
                            min_line_width=5 << u.AA,
                            flag_name="telluric"):
        """Flag the pixels of a data container affected by telluric absorption.
        
        Parameters
        ----------
        data_container : :class:`DataContainer`
            Input DataContainer
        telluric_correction : np.array, optional
            Input telluric correction values. Regions affected by telluric
            absortion should present values larget than 1, and 1 otherwise.
        wavelength : np.ndarray, optional
            Wavelength associated to the telluric correction.
        path_to_model : str
            Path to a text file containing the left and right edges of each
            telluric line.
        """

        telluric_flag = np.zeros(data_container.rss_intensity.shape,
                                 dtype=bool)
        if telluric_correction is not None:
            if wavelength is None:
                raise ValueError(
                    "Must provide a wavelength associated to the correction")
            else:
                interp_tell_corr = np.interp(
                    data_container.wavelength, wavelength, telluric_correction)
                telluric_flag[:, interp_tell_corr >= 1.001] = True
        else:
            if path_to_model is None:
                path_to_model = cls.default_model_file
        
            telluric_wl1, telluric_wl2 = np.loadtxt(
                path_to_model, unpack=True, usecols=(0, 1))
            telluric_wl1 = telluric_wl1 << u.AA
            telluric_wl2 = telluric_wl2 << u.AA
            for b, r in zip(telluric_wl1, telluric_wl2):
                # If the line is narrower than the minimum required
                if r - b < min_line_width:
                    d = min_line_width - (r - b)
                    r += d / 2
                    b -= d / 2
                b_idx = data_container.wcs.spectral.world_to_array_index(
                    b)
                r_idx = data_container.wcs.spectral.world_to_array_index(
                    r)
                if r_idx == b_idx:
                    r_idx += 1
                telluric_flag[:, slice(b_idx, r_idx)] = True
        telluric_flag = data_container.rss_to_original(telluric_flag)
        data_container.mask.flag_pixels(telluric_flag, flag_name,
                                        desc="telluric absoption contaminated")

    @staticmethod
    def plot_correction(spectra_container, telluric_correction,
                        wave_min=None, wave_max=None, exclude_wlm=None,
                        **kwargs):
        """
        Plot a telluric correction.

        Parameters
        ----------
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
        fig : :class:`matplotlib.pyplot.Figure`
            The matplotlib figure object.
        """
        fig, ax = plt.subplots()
        sorted_idx = spectra_container.get_spectra_sorted()
        ax.set_title("Telluric correction using fibres {} (cyan) and {} (fuchsia)"
                        .format(sorted_idx[-2], sorted_idx[-1]))

        ax.plot(spectra_container.wavelength,
                spectra_container.rss_intensity[sorted_idx[-2]], color="c",
                label='Original', lw=3, alpha=0.8)
        ax.plot(spectra_container.wavelength,
                spectra_container.rss_intensity[sorted_idx[-2]]
                * telluric_correction, color="deepskyblue",
                label='Corrected', lw=1, alpha=0.8)

        ax.plot(spectra_container.wavelength,
                spectra_container.rss_intensity[sorted_idx[-1]], color="fuchsia",
                label='Original', lw=3, alpha=0.8)
        ax.plot(spectra_container.wavelength,
                spectra_container.rss_intensity[sorted_idx[-1]]
                * telluric_correction, color="purple",
                label='Corrected', lw=1, alpha=0.8)
        ax.set_ylim(np.nanpercentile(
            spectra_container.rss_intensity[sorted_idx[-1]], [1, 99]).value)
        ax.set_ylabel(f"Flux ({spectra_container.intensity.unit})")
        ax.legend(ncol=2)
        ax.axvline(x=wave_min.to_value(spectra_container.wavelength.unit),
                   color='lime', linestyle='--')
        ax.axvline(x=wave_max.to_value(spectra_container.wavelength.unit),
                   color='lime', linestyle='--')
        ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
        if exclude_wlm is not None:
            for i in range(len(exclude_wlm)):
                ax.axvspan(
                    exclude_wlm[i][0].to_value(spectra_container.wavelength.unit),
                    exclude_wlm[i][1].to_value(spectra_container.wavelength.unit),
                    color='lightgreen', alpha=0.1)
        ax.minorticks_on()
        ax.set_xlim(spectra_container.wavelength[[0, -1]].value)
        if kwargs.get('plot', False):
            plt.show()
        else:
            plt.close()
        return fig

    def apply(self, spectra_container, update=True):
        """
        Apply the telluric correction to the input :class:`SpectraContainer`.

        Parameters
        ----------
        spectra_container : :class:`SpectraContainer`
            The input SpectraContainer to correct.
        update : bool, optional
            Whether to update the correction (default is True).

        Returns
        -------
        rss_out : :class:`SpectraContainer`
            The corrected copy of the input SpectraContainer.
        """

        # Check wavelength
        if not spectra_container.wavelength.size == self.wavelength.size or not np.allclose(
            spectra_container.wavelength, self.wavelength, equal_nan=True):
            self.vprint("Interpolating correction to input wavelength")
            tell_correction = self.interpolate_model(
                spectra_container.wavelength, update=update)
        else:
            tell_correction = self.telluric_correction

        if spectra_container.is_corrected(self.name):
            self.vprint("Data already calibrated")
            return spectra_container

        # Copy input RSS for storage the changes implemented in the task
        tell_correction = tell_correction**(
            spectra_container.info["airmass"] / self.airmass)
        spectra_container_out = spectra_container.copy()
        self.vprint("Applying telluric correction")
        spectra_container_out.rss_intensity = (spectra_container_out.rss_intensity
                                               * tell_correction)
        spectra_container_out.rss_variance = (spectra_container_out.rss_variance
                                              * tell_correction**2)
        self.record_correction(spectra_container_out, status='applied')
        return spectra_container_out

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
        telluric_correction = np.interp(
            wavelength, self.wavelength, self.telluric_correction,
            left=1, right=1)

        if update:
            self.telluric_correction = telluric_correction
            self.wavelength = wavelength
        return telluric_correction

    def save(self, filename='telluric_correction.txt', **kwargs):
        """
        Save the telluric correction to a text file.

        Parameters
        ----------
        filename : str, optional
            The name of the output file.
        **kwargs : dict
            Extra arguments to be passed to :func:`numpy.savetxt`.
        """
        self.vprint(f"Saving telluric correction into file {filename}")
        if "header" in kwargs:
            kwargs["header"] = f"airmass={self.airmass}\n" + kwargs["header"]
        else:
            kwargs["header"] = f"airmass={self.airmass}\n"

        np.savetxt(filename, np.array(
            [self.wavelength, self.telluric_correction]).T, **kwargs)


def combine_telluric_corrections(list_of_telcorr, ref_wavelength):
    """Combine a list of input telluric corrections.
    
    Parameters
    ----------
    list_of_telcorr : list
        List of :class:`TelluricCorrection` instances to combine.
    ref_wavelength : 1D np.ndarray
        Reference array grid to interpolate each correction before the combination.

    Returns
    -------
    combine_telcorr : :class:`TelluricCorrection`
        TelluricCorretion resulting from combining the input list.
    """
    vprint("Combining input telluric corrections")
    telluric_corrections = np.zeros(
        (len(list_of_telcorr), ref_wavelength.size))
    airmass = np.zeros(len(list_of_telcorr), dtype=float)
    for i, telcorr in enumerate(list_of_telcorr):
        telluric_corrections[i] = telcorr.interpolate_model(ref_wavelength)
        airmass[i] = telcorr.airmass

    telluric_correction = np.nanmedian(telluric_corrections, axis=0)
    return TelluricCorrection(telluric_correction=telluric_correction,
                              wavelength=ref_wavelength,
                              airmass=np.nanmedian(airmass), verbose=False)


# =============================================================================
# Self-calibration based on strong sky lines
# =============================================================================

# TODO: rename SkyWaveletFilter?

class WaveletFilter(object):
    '''
    Estimate overall fibre throughput and wavelength calibration based on sky emission lines (from wavelet transform).

    Description
    -----------

    Given a Row-Stacked Spectra (RSS) object:
    1. Estimate the FWHM of emission lines from the autocorrelation of the median (~ sky) spectrum.
    2. Apply a (mexican top hat) wavelet filter to detect features on that scale (i.e. filter out the continuum).
    3. Find regions actually dominated by sky lines (i.e. exclude lines from the target).
    4. Estimate fibre throughput from norm (standard deviation).
    5. Estimate wavelength offset of each fibre (in pixels) from cross-correlation with the sky.

    Attributes
    ----------
    - scale: (int)
        Scale of the wavelet filter, in pixels.
    - filtered: numpy.ndarray(float)
        Throughput-corrected wavelet coefficient (filtered rss.intensity).
    - sky_lo: numpy.ndarray(float)
        Lower percentile (16) of the estimated filtered sky.
    - sky: numpy.ndarray(float)
        Estimated (median) filtered sky.
    - sky_hi: numpy.ndarray(float)
        Upper percentile (84) of the estimated filtered sky.
    - sky_weight: numpy.ndarray(float)
        Estimated dominance of the sky (between 0 and 1).
    - fibre_throughput: numpy.ndarray(float)
        Overall throughput of each fibre.
    - wavelength: numpy.ndarray(float)
        Wavelength of the wavelet coefficients (only used for plotting).
    - fibre_offset: numpy.ndarray(float)
        Relative wavelenght calibration. Offset, in pixels, with respect to the sky (from weighted cross-correlation).
    '''

    def __init__(self, rss: RSS):

        # 1. Estimate the FWHM of emission lines from the autocorrelation of the median (~ sky) spectrum.

        x = np.nanmedian(rss.intensity, axis=0)  # median ~ sky spectrum
        x -= np.nanmean(x)
        x = scipy.signal.correlate(x, x, mode='same')
        h = (np.count_nonzero(x > 0.5*np.nanmax(x)) + 1) // 2
        # h = 0
        self.scale = 2*h + 1
        vprint(f'> Wavelet filter scale: {self.scale} pixels')

        # 2. Apply a (mexican top hat) wavelet filter to detect features on that scale (i.e. filter out the continuum).

        x = np.nancumsum(rss.intensity, axis=1)
        self.filtered = (x[:, 2*self.scale:-self.scale] -
                         x[:, self.scale:-2*self.scale]) / self.scale
        self.filtered -= (x[:, 3*self.scale:] -
                          x[:, :-3*self.scale]) / (3*self.scale)

        # 3. Find regions actually dominated by sky lines (i.e. exclude lines from the target).

        self.sky_lo, self.sky, self.sky_hi = np.nanpercentile(
            self.filtered, [16, 50, 84], axis=0)
        self.sky_weight = 1 - \
            np.exp(-.5 * (self.sky / np.fmax((self.sky - self.sky_lo),
                   (self.sky_hi - self.sky)))**2)
        self.sky *= self.sky_weight
        self.sky_lo *= self.sky_weight
        self.sky_hi *= self.sky_weight

        # 4. Estimate fibre throughput from norm (standard deviation).

        # should be irrelevant
        self.filtered -= np.nanmean(self.filtered, axis=1)[:, np.newaxis]
        self.filtered *= self.sky_weight
        # self.fibre_throughput = np.nanstd(self.filtered, axis=1)
        # self.fibre_throughput = np.nanmedian(self.filtered, axis=1)
        # self.filtered /= self.fibre_throughput[:, np.newaxis]
        self.filtered[~ np.isfinite(self.filtered)] = 0

        # self.fibre_throughput = np.nanmedian(self.filtered / self.sky, axis=1)
        x = np.exp(-.5 * ((self.filtered - self.sky) /
                   (self.sky_hi - self.sky_lo))**2)
        x *= self.sky_weight[np.newaxis, :]
        # self.fibre_throughput = np.nanmean(x * self.filtered / self.sky[np.newaxis, :], axis=1) / np.nanmean(x)
        x = np.where(x > 0.5, self.filtered / self.sky[np.newaxis, :], np.nan)
        self.fibre_throughput = np.nanmedian(x, axis=1)
        renorm = np.nanmedian(self.fibre_throughput)
        self.fibre_throughput /= renorm
        self.filtered /= self.fibre_throughput[:, np.newaxis]
        self.sky *= renorm
        self.sky_lo *= renorm
        self.sky_hi *= renorm

        # 5. Estimate wavelength offset of each fibre (in pixels) from cross-correlation with the sky.

        # only for plotting:
        x = u.Quantity(rss.wavelength[self.scale + h + 1: -self.scale - h])
        if x.unit.is_equivalent(u.AA):
            self.wavelength = x.to_value(u.AA)
        # assume it's already in Angstrom
        elif x.unit.is_equivalent(u.dimensionless_unscaled):
            self.wavelength = x.value
        else:
            raise TypeError(f'  ERROR: wrong wavelength units ({x.unit})')

        # mid = rss.intensity.shape[1] // 2
        mid = self.wavelength.size // 2
        s = self.scale
        x = np.nanmedian(self.filtered, axis=0)
        x[~ np.isfinite(x)] = 0
        x = scipy.signal.fftconvolve(
            self.filtered, x[np.newaxis, ::-1], mode='same', axes=1)[:, mid-s:mid+s+1]
        idx = np.arange(x.shape[1])
        weight = np.where(x > 0, x, 0)
        self.fibre_offset = np.nansum(
            (idx - s)[np.newaxis, :] * weight, axis=1) / np.nansum(weight, axis=1)

    def qc_plots(self, show=False, save_as=None):
        '''
        Top: Filtered intensity for every fibre.
        Centre: Sky-subtracted.
        Bottom: Filtered sky.
        '''
        fig, axes = new_figure('wavelet_filter', nrows=3, ncols=2, sharey=False, gridspec_kw={
                               'width_ratios': [1, .02], 'hspace': 0., 'wspace': .1})
        im, cb = plot_image(fig, axes[0, 0], 'wavelet coefficient', self.filtered,
                            x=self.wavelength, ylabel='spec_id', cbax=axes[0, 1])
        im, cb = plot_image(fig, axes[1, 0], 'sky-subtracted', self.filtered -
                            self.sky, x=self.wavelength, ylabel='spec_id', cbax=axes[1, 1])
        axes[1, 0].sharey(axes[0, 0])
        ax = axes[-1, 0]
        axes[-1, 1].axis('off')
        ax.plot(self.wavelength, self.sky, alpha=.5, c='tomato',
                label='sky (median coefficient)')
        ax.fill_between(self.wavelength, self.sky_lo, self.sky_hi, color='r',
                        alpha=0.1, label='uncertainty (16-84%)')
        ax.plot(self.wavelength, self.sky/self.sky_weight,
                alpha=.5, label='unweighted sky')
        ax.set_xlabel('wavelength [pix]')
        ax.set_ylabel(f'filtered (scale = {self.scale} pix)')
        ax.legend()
        if show:  # TODO: deal with ipympl
            fig.show()
        if save_as is not None:
            fig.savefig(save_as)

    def get_throughput_object(self):
        return Throughput(throughput_data=np.repeat(self.fibre_throughput[:, np.newaxis], self.wavelength.size + 3*self.scale, axis=1),
                          throughput_error=np.full([self.fibre_throughput.size, self.wavelength.size + 3*self.scale], np.nan)
                         )

    def get_wavelength_offset(self):
        return WavelengthOffset(offset_data=np.repeat(self.fibre_offset[:, np.newaxis], self.wavelength.size + 3*self.scale, axis=1) << u.pix)


# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
