import os

import numpy as np
from matplotlib import pyplot as plt

from astropy import units as u
from astropy import constants
from astropy.table import Table
from astropy import stats
from astropy.modeling.models import Gaussian1D, Lorentz1D, Chebyshev1D, custom_model
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter, TRFLSQFitter

from specutils import Spectrum1D
from specutils import fitting as specutils_fitting
from specutils.spectra import SpectralRegion
from specutils import manipulation    
    
from scipy import signal
from scipy import ndimage

from pykoala.corrections.sky import ContinuumEstimator
from pykoala import vprint
from pykoala.ancillary import flux_conserving_interpolation


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
    def _preserve_units(func):
        def wrapper(data, *args, **kwargs):
            if isinstance(data, u.Quantity):
                unit = data.unit
            else:
                unit = 1
            return func(data, *args, **kwargs) * unit
        return wrapper

    @staticmethod
    @_preserve_units
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
        continuum = signal.medfilt(data, window_size)
        return continuum

    @staticmethod
    @_preserve_units
    def percentile_continuum(data, percentile, window_size):
        continuum = ndimage.percentile_filter(data, percentile,
                                                    window_size)
        return continuum

    @staticmethod
    @_preserve_units
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
# Emission line profile
# =============================================================================

class LineProfile(Fittable1DModel):
    """One dimensional emission line profile.
    
    Parameters
    ----------
    central_wavelength : float or `~astropy.units.Quantity`.
        Centroid of the line.
    flux : float or `~astropy.units.Quantity`.
        Integrated flux of the line.
    fwhm : float or `~astropy.units.Quantity`.
        Full width at half maximum (FWHM).

     Notes
    -----
    Either all or none of input ``wavelength``, ``central_wavelength`` and
    ``fwhm`` must be provided consistently with compatible units or as unitless
    numbers.
    """

    central_wavelength = Parameter(default=1, description="Line profile centroid")
    flux = Parameter(default=0, bounds=(0, None), description="Line integrated flux")
    fwhm = Parameter(default=1, bounds=(1e-1 * u.angstrom, 1e1 * u.angstrom),
                     description="Line profile full width at half maximum")

    @property
    def input_units(self):
        if self.central_wavelength.input_unit is None:
            return None
        return {self.inputs[0]: self.central_wavelength.input_unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {
            "central_wavelength": inputs_unit[self.inputs[0]],
            "fwhm": inputs_unit[self.inputs[0]],
            "flux": outputs_unit[self.outputs[0]],
        }


class GaussianLineProfile(LineProfile):
    """Gaussian 1D line profile model adapted from `~astropy.modelling.models.Gaussian1D`."""
    fwhm_to_sigma = 1 / 2.35482

    def bounding_box(self, factor=3.5):
        """
        Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``.

        Parameters
        ----------
        factor : float
            The multiple of `stddev` used to define the limits.
            The default is 3.5, corresponding to a relative error < 3e-4.
        """
        return (self.central_wavelength - factor * self.fwhm * self.fwhm_to_sigma,
                self.central_wavelength + factor * self.fwhm * self.fwhm_to_sigma)

    @staticmethod
    def evaluate(wavelength, central_wavelength, flux, fwhm):
        """
        GaussianLineProfile model function.
        """
        sigma = fwhm / 2.35482
        return flux / np.sqrt(2) / fwhm / sigma * np.exp(
            -0.5 * ((wavelength - central_wavelength) / sigma)**2)

    @staticmethod
    def fit_deriv(wavelength, central_wavelength, flux, fwhm):
        """
        GaussianLineProfile model function derivatives.
        """
        sigma = fwhm / 2.35482
        d_flux = np.exp(-0.5 / sigma**2 * (wavelength - central_wavelength) ** 2)
        d_wave = flux * d_flux * (wavelength - central_wavelength) / sigma**2
        d_sigma = flux * d_flux * (wavelength - central_wavelength) ** 2 / sigma**3
        return [d_wave, d_flux, d_sigma * 2.35482]


class LorentzianLineProfile(LineProfile):
    """Lorentzian 1D line profile model adapted from `~astropy.modelling.models.Lorentz1D`."""

    @staticmethod
    def evaluate(wavelength, central_wavelength, flux, fwhm):
        """One dimensional Lorentzian model function."""
        gamma = fwhm / 2
        return flux * gamma / np.pi / ((wavelength - central_wavelength)**2 + gamma**2)

    @staticmethod
    def fit_deriv(wavelength, central_wavelength, flux, fwhm):
        """One dimensional Lorentzian model derivative with respect to parameters."""
        gamma = fwhm / 2
        d_flux = gamma / np.pi / ((wavelength - central_wavelength)**2 + gamma**2)
        d_wave = flux / gamma / np.pi * d_flux * (
            2 * wavelength - 2 * central_wavelength) / (
                fwhm**2 + (wavelength - central_wavelength)**2)
        d_fwhm = 2 * flux / gamma / np.pi * d_flux / fwhm * (1 - d_flux)
        return [d_wave, d_flux, d_fwhm]

    def bounding_box(self, factor=25):
        """Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``.

        Parameters
        ----------
        factor : float
            The multiple of FWHM used to define the limits.
            Default is chosen to include most (99%) of the
            area under the curve, while still showing the
            central feature of interest.

        """
        return (self.central_wavelength - factor * self.fwhm,
                self.central_wavelength + factor * self.fwhm)


class EmissionLine(object):
    """A representation of an emission line.
    
    Attributes
    ----------
    central_wavelength : float or `~astropy.units.Quantity`.
        Centroid of the line.
    flux : float or `~astropy.units.Quantity`.
        Integrated flux of the line.
    fwhm : float or `~astropy.units.Quantity`.
        Full width at half maximum (FWHM).
    ion : str
        Name of the atomic specie that originates the emission line.
    profile : LineProfile
        The line profile associated to this emission line.
    
    Methods
    -------
    sample_to(wavelenths)
        Sample the emission line into a given array of wavelengths.

    """
    def __init__(self, central_wavelength, flux, fwhm,
                 ion='n/a', profile=GaussianLineProfile, **profile_kwargs):
        self.central_wavelength = central_wavelength
        self.flux = flux
        self.fwhm = fwhm
        self.ion = ion
        self.profile_kwargs = profile_kwargs
        self.profile = profile

    @property
    def central_wavelength(self):
        return self._central_wavelength
    
    @central_wavelength.setter
    def central_wavelength(self, value):
        self._central_wavelength = value

    @property
    def flux(self):
        return self._flux
    
    @flux.setter
    def flux(self, value):
        self._flux = value

    @property
    def fwhm(self):
        return self._fwhm
    
    @fwhm.setter
    def fwhm(self, value):
        self._fwhm = value

    @property
    def ion(self):
        return self._ion
    
    @ion.setter
    def ion(self, value):
        self._ion = value

    @property
    def profile(self):
        return self._profile(self.central_wavelength, self.flux, self.fwhm,
                             **self.profile_kwargs)
    
    @profile.setter
    def profile(self, value):
        self._profile = value


    def sample_to(self, wavelenths):
        """Sample the emission line into the input array of wavelengths.
        
        Parameters
        ----------
        wavelengths : np.array or `~astropy.units.Quantity`.
            Wavelength grid to sample the emission line profile.

        Returns
        -------
        spectrum : `~specutils.Spectrum1D`
            Spectrum containing the emission line profile.
        """
        return Spectrum1D(spectral_axis=wavelenths, flux=self.profile(wavelenths))


class EmissionLinesCollection(object):
    def __init__(self, wavelengths=None, fluxes=None, fwhms=None, ions=None,
                 profiles=None, profiles_args=None):
        self.wavelengths = wavelengths
        self.fluxes = fluxes
        self.fwhms = fwhms
        self.ions = ions
        self.profiles = profiles
        self.profiles_args = profiles_args

        if self.profiles is None:
            self.profiles = ["gaussian"] * len(self.wavelengths)

        self.emission_lines = [EmissionLine(cw, f, fwhm, ion, prof
                                            ) for cw, f, fwhm, ion, prof in zip(
            self.wavelengths, self.fluxes, self.fwhms, self.ions, self.profiles)]

    @classmethod
    def from_fits_table(cls, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "em_line_list.fits")

        table = Table.read(path)

        # Force the table to include a column of wavelengths
        wavelengths = table['wavelength']

        if "flux" in table.keys():
            fluxes = table['flux']
        else:
            fluxes = np.ones_like(wavelengths, dtype=float)

        if "fwhm" in table.keys():
            fwhm = table['fwhm']
        else:
            fwhm = np.ones_like(wavelengths, dtype=float)    
        
        if "ion" in table.keys():
            ions = table['ion']
        else:
            ions = ["n/a"] * len(wavelengths)
        return cls(wavelengths=wavelengths, fwhms=fwhm, fluxes=fluxes, ions=ions)

    
    def sample_to(self, wavelength):
        f = np.zeros_like(wavelength, dtype=float)
        for line in self.emission_lines:
            f += line.sample_to(wavelength)
        return f


def calculate_z(l_obs=None, l_ref=None, v_rad=None):
    """
    Calculate z using a redshifted line.

    Compute the corresponding redshift using the optical regime approximation
    ::math
        z = \lambda_{obs} / \lamdba_ref - 1

    Parameters
    ----------
    l_obs : float, optional
        Observational wavelength. The default is None.
    l_ref : float, optional
        reference wavelength . The default is None.
    v_rad : float, optional
        radial velocity. The default is None.

    Returns
    -------
    z : float
        redshift.
    """

    if v_rad is not None:
        z = v_rad/ constants.c
    else:
        z = l_obs/l_ref - 1.
        v_rad = z * constants.c
        
    vprint("Using line {}, l_rest = {:.2f}, peak at l_obs = {:.2f}. ".format(
           l_ref, l_obs))
    return z


def substract_spectrum_continuum(spectrum, parametric_continuum=True,
                                 continuum_fit_args={}):
    """
    Substract the continuum of a given spectrum.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO

    """
    if parametric_continuum:
        continuum_model = specutils_fitting.fit_generic_continuum(
        spectrum, model=Chebyshev1D(3),
        fitter=LinearLSQFitter(),
        **continuum_fit_args)
        continuum = continuum_model(spectrum.wavelength)
    else:
        median_continuum = ContinuumEstimator.medfilt_continuum(
            spectrum.flux, window_size=11)
        median_rms_offset = np.median((spectrum.flux - median_continuum)**2)**0.5
        cont_offset, offset = ContinuumEstimator.lower_envelope(
            spectrum.wavelength, spectrum.flux)
        lower_env_continuum = (cont_offset - offset)

        good_continuum = median_continuum < lower_env_continuum + 3 * median_rms_offset
        continuum = np.interp(spectrum.wavelength,
                                spectrum.wavelength[good_continuum],
                                median_continuum[good_continuum])
        cont_free_spectrum = spectrum - continuum
    return cont_free_spectrum

def fit_emission_lines(spectrum: Spectrum1D, lines,
                       wave_range=None, continuum_fit_args={},
                       substract_continuum=True, parametric_continuum=False,
                       fitter=TRFLSQFitter(calc_uncertainties=True),
                       plot=True):
    """
    #TODO 
    """
    
    
    if wave_range is not None:
        if isinstance(wave_range, list):
            spec_region =SpectralRegion(wave_range[0], wave_range[1])
        elif isinstance(wave_range, SpectralRegion):
            spec_region = wave_range
        else:
            raise NameError("Unrecognized type fo wave_range")        
        spectrum = manipulation.extract_region(spectrum, spec_region)

    # Continuum fit
    weights = np.ones(spectrum.wavelength.size)

    if substract_continuum:
        cont_free_spectrum = substract_spectrum_continuum(spectrum,
                                     parametric_continuum=parametric_continuum,
                                     continuum_fit_args=continuum_fit_args)
        continuum = spectrum - cont_free_spectrum
    else:
        continuum = np.zeros_like(spectrum.flux)
        cont_free_spectrum = spectrum

    continuum_mad = np.median(
        np.abs(cont_free_spectrum.flux - np.median(cont_free_spectrum.flux)))
    weights[np.abs(cont_free_spectrum.flux) <= 3 * continuum_mad] = 0.0


    if isinstance(lines, EmissionLine):
        lines_model = lines.profile
    elif isinstance(lines, list):
        lines_model = None
        for line in lines:
            if lines_model is None:
                lines_model = line.profile
            else:
                lines_model += line.profile

    fit_em_line = fitter(lines_model, cont_free_spectrum.wavelength,
                         cont_free_spectrum.flux,
                         weights=weights,
                         maxiter=1000,
                         #acc=1e-05, epsilon=1.4901161193847656e-06
                         )
    if plot:
        fig, axs = plt.subplots(nrows=2, constrained_layout=True,sharex=True)
        ax = axs[0]
        ax.plot(spectrum.wavelength, spectrum.flux, lw=2, label='Input spectrum')
        ax.plot(spectrum.wavelength, continuum, lw=1, label='Continuum')
        ax.plot(spectrum.wavelength, fit_em_line(spectrum.wavelength) + continuum, lw=0.8, label='Line fit + continuum',
                color='r')
        ax.set_ylabel(f"Flux ({spectrum.flux.unit.to_string()})")
        ax.legend()

        ax = axs[1]
        ax.plot(spectrum.wavelength, spectrum.flux - continuum - fit_em_line(spectrum.wavelength),
                label='Residuals')
        ax.legend()
        ax.set_xlabel(f"Wavelength ({spectrum.wavelength.unit.to_string()})")
        ax.set_ylabel(f"Flux ({spectrum.flux.unit.to_string()})")
        twax = ax.twinx()
        twax.plot(spectrum.wavelength, weights, color='fuchsia', zorder=-1,
                  alpha=0.5)
        plt.show()
    else:
        fig = None
    return fit_em_line, Spectrum1D(spectral_axis=spectrum.wavelength,
                                   flux=continuum), fitter.fit_info, fig


def find_emission_lines(spectrum,
                        fit_continuum=True, continuum_fit_args={},
                        parametric_continuum=False,
                        sigma_threshold=3, plot=True):
    
    if fit_continuum:
        if parametric_continuum:
            continuum_model = specutils_fitting.fit_generic_continuum(
            spectrum, model=Chebyshev1D(3),
            fitter=LinearLSQFitter(),
            **continuum_fit_args)
            continuum = continuum_model(spectrum.wavelength)
        else:
            median_continuum = ContinuumEstimator.medfilt_continuum(
                spectrum.flux, window_size=11)
            median_rms_offset = np.median((spectrum.flux - median_continuum)**2)**0.5
            cont_offset, offset = ContinuumEstimator.lower_envelope(
                spectrum.wavelength, spectrum.flux)
            lower_env_continuum = (cont_offset - offset)

            good_continuum = median_continuum < lower_env_continuum + 3 * median_rms_offset
            continuum = np.interp(spectrum.wavelength,
                                  spectrum.wavelength[good_continuum],
                                  median_continuum[good_continuum])
            cont_free_spectrum = spectrum - continuum
    else:
        continuum = np.zeros_like(spectrum.flux)
        cont_free_spectrum = spectrum

    background, background_sigma = BackgroundEstimator.mad(cont_free_spectrum.flux)
    threshold = background + sigma_threshold * background_sigma

    peaks, peaks_prop = signal.find_peaks(cont_free_spectrum.flux.value,
                        height=background.value + sigma_threshold * background_sigma.value
                        )
    peaks = signal.find_peaks_cwt(cont_free_spectrum.flux.value,
                                  np.arange(5,20))
    
    good_peaks = np.where(
        cont_free_spectrum.flux[peaks] > threshold)[0]
    peaks = peaks[good_peaks]
    peaks_prop = {"peaks_wavelength": cont_free_spectrum.wavelength[peaks],
                  "peak_heights": cont_free_spectrum.flux[peaks]}
    vprint(f"Peaks found at pixel positions: {peaks}")
    if len(peaks) == 0:
        return peaks, peaks_prop, None

    if plot:
        fig, ax = plt.subplots()
        ax.plot(cont_free_spectrum.wavelength, cont_free_spectrum.flux)
        ax.plot(cont_free_spectrum.wavelength[peaks],
                cont_free_spectrum.flux[peaks], "+", markersize=12)
        ax.axhline(background.value + sigma_threshold * background_sigma.value,
                   ls='--')
        plt.show()
    else:
        fig = None
    peaks_prop['fig'] = fig
    return peaks, peaks_prop

# TODO: Include Yago's wavelet method here and segmentation maps

def interpolate_spectrum(new_wavelength, spectrum):
    """Interpolate a Spectrum1D into a new wavelength array."""
    new_flux = flux_conserving_interpolation(new_wavelength,
                                  spectrum.wavelength, spectrum.flux)
    return Spectrum1D(spectral_axis=new_wavelength, flux=new_flux)

def interpolate_spectrum_nans(spectrum):
    """
    Given a spectrum, remove nans using a linear interpolation.
    
    TODO
    """
    
    good_values = np.isfinite(spectrum.flux)
    filtered_flux = np.interp(spectrum.wavelength,
                              spectrum.wavelength[good_values],
                              spectrum.flux[good_values])
    return Spectrum1D(spectral_axis=spectrum.wavelength, flux=filtered_flux)
