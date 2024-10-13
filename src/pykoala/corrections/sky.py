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
# from matplotlib.colors import LogNorm
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
# KOALA packages
# =============================================================================
# Modular
from pykoala import vprint
from pykoala.plotting.utils import new_figure, plot_image
from pykoala.corrections.correction import CorrectionBase
from pykoala.corrections.throughput import Throughput
from pykoala.corrections.wavelength import WavelengthOffset
from pykoala.data_container import DataContainer
from pykoala.data_container import RSS

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
        mad = np.nanmedian(
            np.abs(data - np.expand_dims(background, axis=axis)), axis=axis)
        background_sigma = 1.4826 * mad
        return background, background_sigma

    def fit(data, axis, wavelet):
        """
        Background estimator from linear fit (Work in progress).

        Parameters
        ----------
        data : np.ndarray
            The input data array from which to compute the background.
            TODO: Make it a `DataContainer`.
        axis : int, optional
            TODO: remove from the call in `SkyFromObject`.
        wavelet: WaveletFilter
            Used to estimate the mean sky flux.

        Returns
        -------
        background : np.ndarray
            The computed background sky.
        background_sigma : np.ndarray
            TODO: An error estimate.
        """
        flux = np.nanmean(data, axis=1)
        mean_flux = np.nanmean(flux)
        flux_cut_low = np.nanmedian(flux[flux < mean_flux])
        flux_cut_hi = 2*mean_flux - flux_cut_low
        flux_low = np.nanmean(flux[flux < flux_cut_low])
        flux_med = np.nanmean(flux[(flux > flux_cut_low) & (flux < mean_flux)])
        flux_hi = np.nanmean(flux[(flux > mean_flux) & (flux < flux_cut_hi)])

        I_low = np.nanmean(data[flux < flux_cut_low, :], axis=0)
        I_med = np.nanmean(data[(flux > flux_cut_low) &
                           (flux < mean_flux), :], axis=0)
        I_hi = np.nanmean(
            data[(flux > mean_flux) & (flux < flux_cut_hi), :], axis=0)
        m = (I_hi - I_low) / (flux_hi - flux_low)
        b = I_low - m * flux_low

        sky_flux_candidate = np.arange(0, flux_cut_hi, .01*np.min(flux))
        sky_filtered = m[np.newaxis, :] * \
            sky_flux_candidate[:, np.newaxis] + b[np.newaxis, :]
        x = np.nancumsum(sky_filtered, axis=1)
        s = wavelet.scale
        sky_filtered = (x[:, 2*s:-s] - x[:, s:-2*s]) / s
        sky_filtered -= (x[:, 3*s:] - x[:, :-3*s]) / (3*s)
        chi2_no_sky = np.nanstd(sky_filtered*(1 - wavelet.sky_weight), axis=1)

        ''' Unsuccessful attempt (keep for a while, just in case)
        # Wavelet scale:
        x = b - np.nanmean(b)
        x = scipy.signal.correlate(x, x, mode='same')
        h = (np.count_nonzero(x > 0.5*np.nanmax(x)) + 1) // 2
        s = 2*h + 1

        sky_flux_candidate = np.arange(0, flux_cut_hi, .01*np.min(flux))
        sky_filtered = m[np.newaxis, :] * sky_flux_candidate[:, np.newaxis] + b[np.newaxis, :]
        x = np.nancumsum(sky_filtered, axis=1)
        sky_filtered = (x[:, 2*s:-s] - x[:, s:-2*s]) / s
        sky_filtered -= (x[:, 3*s:] - x[:, :-3*s]) / (3*s)

        # Sky-based weight:
        print(s, data.shape, sky_filtered.shape)
        p16, p50, p84 = np.nanpercentile(data[:, s+h+1:-s-h], [16, 50, 84], axis=0)
        not_sky = np.exp(-.5 * (p50 / np.fmax((p84 - p50), (p50 - p16)))**2)

        # Select optimal flux:
        chi2_no_sky = np.nanstd(sky_filtered * not_sky[np.newaxis, :], axis=1)
        '''

        sky_flux = sky_flux_candidate[np.nanargmin(chi2_no_sky)]
        sky_intensity = b + m*sky_flux
        vprint(f"{s} {sky_flux}")

        return sky_intensity, np.nan + sky_intensity

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
        self.wavelength = kwargs.get('wavelength', None)
        self.intensity = kwargs.get('intensity', None)
        self.variance = kwargs.get('variance', None)
        self.continuum = kwargs.get('continuum', None)

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
        assert self.intensity, "Sky Model intensity is None"

        if self.continuum is None:
            vprint("Sky Model intensity might contain continuum emission"
                   " leading to unsuccessful emission line fit")
        if self.variance is None:
            errors = np.ones_like(self.intensity, dtype=float)

        finite_mask = np.isfinite(self.intensity)
        p0_amplitude = np.interp(self.sky_lines, self.dc.wavelength[finite_mask],
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

    def plot_sky_model(self, show=False):
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
        fig = plt.figure(constrained_layout=True)
        if self.intensity.ndim == 1:
            ax = fig.add_subplot(111, title='1-D Sky Model')
            ax.fill_between(self.wavelength, self.intensity - self.variance**0.5,
                            self.intensity + self.variance**0.5, color='k',
                            alpha=0.5, label='STD')
            ax.plot(self.wavelength, self.intensity,
                    color='r', label='intensity')
            ax.set_ylim(np.nanpercentile(self.intensity, [1, 99]))
            ax.set_ylabel("Intensity")
            ax.set_xlabel("Wavelength (AA)")
            ax.legend()

        elif self.intensity.ndim == 2:
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
        if show:
            plt.show()
        else:
            plt.close()
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
        self.intensity, self.variance = BackgroundEstimator.percentile(
            self.dc.intensity, percentiles=[16, 50, 84])
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
        vprint("Creating SkyModel from input Data Container")
        self.dc = dc
        # self.exptime = dc.info['exptime']
        vprint("Estimating sky background contribution...")

        bckg, bckg_sigma = self.estimate_background(
            bckgr_estimator, bckgr_params, source_mask_nsigma)
        super().__init__(wavelength=self.dc.wavelength,
                         intensity=bckg,
                         variance=bckg_sigma**2)
        if remove_cont:
            vprint("Removing background continuum")
            self.remove_continuum(cont_estimator, cont_estimator_args)

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
            raise NameError(
                f"Input background estimator {bckgr_estimator} does not exist")

        data = self.dc.intensity.copy()

        if data.ndim == 3:
            bckgr_params["axis"] = bckgr_params.get("axis", (1, 2))
            dims_to_expand = (1, 2)
        elif data.ndim == 2:
            bckgr_params["axis"] = bckgr_params.get("axis", 0)
            dims_to_expand = (0)

        if source_mask_nsigma is not None:
            vprint("Pre-estimating background using all data")
            bckgr, bckgr_sigma = estimator(data, **bckgr_params)
            vprint(
                f"Applying sigma-clipping mask (n-sigma={source_mask_nsigma})")
            source_mask = (data > np.expand_dims(bckgr, dims_to_expand) +
                           source_mask_nsigma
                           * np.expand_dims(bckgr_sigma, dims_to_expand))
            data[source_mask] = np.nan

        bckgr, bckgr_sigma = estimator(data, **bckgr_params)
        return bckgr, bckgr_sigma


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
                            vmin=np.nanpercentile(original_image, 1),
                            vmax=np.nanpercentile(original_image, 99))
        )

        ax = axs[0]
        ax.set_title("Original")
        ax.imshow(original_image, **im_args)

        ax = axs[1]
        ax.set_title("Sky emission subtracted")
        mappable = ax.imshow(corr_image, **im_args)
        cax = ax.inset_axes((-1.2, -.1, 2.2, 0.02))
        plt.colorbar(mappable, cax=cax, orientation="horizontal")
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
    telluric_correction = None
    default_model_file = os.path.join(os.path.dirname(__file__), '..',
                                      'input_data', 'sky_lines',
                                      'telluric_lines.txt')
    def __init__(self,
                #  data_container=None,
                 telluric_correction,
                 wavelength,
                 telluric_correction_file="unknown",
                #  n_fibres=10,
                #  frac=0.5,
                 **correction_kwargs):
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
        super().__init__(**correction_kwargs)

        self.telluric_correction = telluric_correction
        self.wavelength = wavelength
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
        return cls(telluric_correction=telluric_correction,
                   wavelength=wavelength,
                   telluric_correction_file=path)

    @classmethod
    def from_smoothed_spectra_container(cls, spectra_container,
                                        exclude_wlm=None, 
                                        light_percentile=0.95,
                                        median_window=10,
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
        telluric_correction = np.ones_like(spectra_container.wavelength)
        if wave_min is None:
            wave_min = spectra_container.wavelength[0]
        if wave_max is None:
            wave_max = spectra_container.wavelength[-1]
        if exclude_wlm is None:
            exclude_wlm = [[6450, 6700], [6850, 7050], [7130, 7380]]
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
                   wavelength=spectra_container.wavelength), fig

    @classmethod
    def from_model(cls, spectra_container, model_file=None, width=30,
                   extra_mask=None, plot=False):
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

        w_l_1, w_l_2, res_intensity, w_lines = np.loadtxt(model_file, unpack=True)

        # Mask telluric regions
        mask = np.ones_like(spectra_container.wavelength, dtype=bool)
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
                wavelength=spectra_container.wavelength), fig


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
            spectra_container.rss_intensity[sorted_idx[-1]], [1, 99]))
        ax.legend(ncol=2)
        ax.axvline(x=wave_min, color='lime', linestyle='--')
        ax.axvline(x=wave_max, color='lime', linestyle='--')
        ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
        if exclude_wlm is not None:
            for i in range(len(exclude_wlm)):
                ax.axvspan(exclude_wlm[i][0],
                           exclude_wlm[i][1], color='lightgreen', alpha=0.1)
        ax.minorticks_on()
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
            self.interpolate_model(spectra_container.wavelength, update=update)

        if spectra_container.is_corrected(self.name):
            self.vprint("Data already calibrated")
            return spectra_container

        # Copy input RSS for storage the changes implemented in the task
        spectra_container_out = spectra_container.copy()
        self.vprint("Applying telluric correction")
        spectra_container_out.rss_intensity *= self.telluric_correction
        spectra_container_out.rss_variance *= self.telluric_correction**2
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
            wavelength, self.wavelength, self.telluric_correction, left=1, right=1)

        if update:
            self.teluric_correction = telluric_correction
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
    for i, telcorr in enumerate(list_of_telcorr):
        telluric_corrections[i] = telcorr.interpolate_model(ref_wavelength)

    telluric_correction = np.nanmedian(telluric_corrections, axis=0)
    return TelluricCorrection(telluric_correction=telluric_correction,
                              wavelength=ref_wavelength, verbose=False)


# =============================================================================
# Self-calibration based on strong sky lines
# =============================================================================

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
        return Throughput(throughput_data=np.repeat(self.fibre_throughput[:, np.newaxis], self.wavelength.size + 3*self.scale, axis=1))

    def get_wavelength_offset(self):
        return WavelengthOffset(offset_data=np.repeat(self.fibre_offset[:, np.newaxis], self.wavelength.size + 3*self.scale, axis=1))


class SkySelfCalibration(CorrectionBase):
    """TODO> Finish documentaion
    Wavelength calibration, throughput, and sky model based on strong sky lines."""
    name = "SkySelfCalibration"
    verbose = True

    # TODO: Don't assume RSS format (intensity[spec_id, wavelength])
    #       def __init__(self, dc:DataContainer, continuum:ContinuumModel):
    def __init__(self, dc: RSS, **correction_kwargs):
        super().__init__(**correction_kwargs)
        self.update(dc)

    def update(self, dc: RSS):
        self.dc = dc
        self.biweight_sky()
        self.update_sky_lines()
        self.calibrate()

    # TODO: use SkyModel as an argument to update()?

    def biweight_sky(self):
        # self.wavelength = self.dc.wavelength.to_value(u.Angstrom)
        self.sky_intensity = stats.biweight.biweight_location(
            self.dc.intensity, axis=0)

    def update_sky_lines(self):
        sky_cont, sky_err = ContinuumEstimator.lower_envelope(
            self.dc.wavelength, self.sky_intensity)  # , min_separation)
        sky_lines = self.sky_intensity - sky_cont

        self.continuum = ContinuumModel(self.dc)
        self.continuum.strong_sky_lines.add_column(
            0.*u.Angstrom, name='sky_wavelength')
        self.continuum.strong_sky_lines.add_column(0., name='sky_intensity')
        for line in self.continuum.strong_sky_lines:
            wavelength = self.dc.wavelength[line['left']:line['right']]
            spectrum = sky_lines[line['left']:line['right']]
            weight = spectrum**2
            line['sky_wavelength'] = np.nansum(
                weight*wavelength) / np.nansum(weight)
            line['sky_intensity'] = np.nanmean(spectrum)

    # TODO: Don't assume RSS format (intensity[spec_id, wavelength])

    def calibrate(self):
        n_spectra = self.dc.intensity.shape[0]
        self.wavelength_offset = np.zeros(
            n_spectra) * self.dc.wavelength[0]  # dirty hack to specify units
        self.wavelength_offset_err = np.zeros_like(self.wavelength_offset)
        self.relative_throughput = np.zeros(n_spectra)
        self.relative_throughput_err = np.zeros(n_spectra)
        self.vprint(f"> Calibrating for {n_spectra} spectra:")
        t0 = time()
        for spec_id in range(n_spectra):
            line_wavelength, line_intensity = self.measure_lines(spec_id)
            y = line_wavelength - \
                self.continuum.strong_sky_lines['sky_wavelength']
            self.wavelength_offset[spec_id] = stats.biweight.biweight_location(
                y)
            self.wavelength_offset_err[spec_id] = stats.biweight.biweight_scale(
                y)
            y = line_intensity / \
                self.continuum.strong_sky_lines['sky_intensity']
            self.relative_throughput[spec_id] = stats.biweight.biweight_location(
                y)
            self.relative_throughput_err[spec_id] = stats.biweight.biweight_scale(
                y)
        self.vprint(f"  Done ({time()-t0:.3g} s)")

    def measure_lines(self, spec_id):
        intensity = self.dc.intensity[spec_id]
        continuum = self.continuum.intensity[spec_id]
        sky_wavelength = self.continuum.strong_sky_lines['sky_wavelength'].to_value(
            u.Angstrom)
        line_wavelength = np.zeros_like(sky_wavelength)
        sky_intensity = self.continuum.strong_sky_lines['sky_intensity']
        line_intensity = np.zeros_like(sky_intensity)
        wavelength = self.dc.wavelength.to_value(u.Angstrom)
        for i, line in enumerate(self.continuum.strong_sky_lines):
            section_wavelength = wavelength[line['left']:line['right']]
            section_intensity = (
                intensity - continuum)[line['left']:line['right']]
            weight = section_intensity**2
            line_wavelength[i] = np.nansum(
                weight*section_wavelength) / np.nansum(weight)
            line_intensity[i] = np.nanmean(section_intensity)
        return line_wavelength*u.Angstrom, line_intensity

    def apply(self, rss):
        pass


# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
