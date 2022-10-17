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
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits
from astropy.modeling import models, fitting
# =============================================================================
# KOALA packages
# =============================================================================
# Modular
from koala.ancillary import vprint
# Modular
from koala.ancillary import vprint
from koala.exceptions import TelluricNoFileError
from koala.corrections.correction import Correction
from koala.rss import RSS
from koala.cubing import Cube
# Original
from koala.onedspec import smooth_spectrum # TODO: Remove


def uves_sky_lines():
    """TODO"""
    files = ["346", "437", "580L", "580U", "800U", "860L", "860U"]
    data_path = os.path.join(os.path.dirname(__file__), "..", "input_data", "sky_lines", "ESO-UVES")
    all_wavelengths = np.empty(1)
    all_fwmh = np.empty(1)
    all_flux = np.empty(1)
    for file in files:
        with fits.open(os.path.join(data_path, "gident_{}.tfits".format(file))) as f:
            wave, fwhm, flux = f[1].data['LAMBDA_AIR'], f[1].data['FWHM'], f[1].data['FLUX']
            all_wavelengths = np.hstack((all_wavelengths, wave))
            all_fwmh = np.hstack((all_fwmh, fwhm))
            all_flux = np.hstack((all_flux, flux))
    sort_pos = np.argsort(all_wavelengths[1:])
    return all_wavelengths[1:][sort_pos], all_fwmh[1:][sort_pos], all_flux[1:][sort_pos]


class SkyModel(object):
    """
    Abstract class of Sky models

    Attributes
    ----------
    intensity: 1D or 2D array, default None

    Methods
    -------
    substract_sky

    Examples
    --------
    """

    def __init__(self, **kwargs):
        self.intensity = kwargs.get('intensity', None)
        self.variance = kwargs.get('variance', None)

    def substract_sky(self, rss, verbose=False):
        """Substracts the sky_model to all fibres in the rss

        Parameters
        ----------
        rss: RSS
            RSS object to which substract sky.
        verbose

        Returns
        -------
        rss_out: RSS
            Sky substracted RSS
        """
        # Set print verbose
        vprint.verbose = verbose
        # Copy input RSS for storage the changes implemented in the task
        rss_out = copy.deepcopy(rss)
        # Substract sky in all fibers
        rss_out.intensity_corrected -= self.intensity[np.newaxis, :] * rss.info['exptime']
        rss_out.variance_corrected += self.variance[np.newaxis, :] * rss.info['exptime'] ** 2
        rss_out.log['sky'] = "Sky emission substracted"
        vprint("  Intensities corrected for sky emission and stored in self.intensity_corrected !")
        # history.append("  Intensities corrected for the sky emission")
        return rss_out


class SkyOffset(SkyModel):
    """
    Sky model based on a single RSS offset sky exposure
    """
    def __init__(self, rss):
        """

        Parameters
        ----------
        rss: RSS
            Raw Stacked Spectra corresponding to the offset-sky exposure.
        """
        self.rss = rss
        self.exptime = rss.info['exptime']
        super().__init__()

    def estimate_sky(self):
        pct = np.nanpercentile(self.rss.intensity_corrected, [16, 50, 84], axis=0
                               ) / self.exptime
        self.intensity = pct[1]
        self.variance = (pct[2] * 0.5 - pct[1] * 0.5)**2


class SkyLibrary(SkyModel):
    """
    Sky model based on several RSS files from which a set of sky spectra will be estimates.
    """

    def __init__(self, rss):
        self.rss = rss
        super().__init__()

    def estimate_sky(self):
        # TODO: compute percentiles of sky to create intensity and variance
        # TODO: Include the possibility of creating 1D or 2D models.
        pass


class SkyFromObject(SkyModel):
    """
    Sky model based on a single RSS science frame
    """
    def __init__(self, rss):
        self.sky_cont = None
        self.sky_lines = None
        self.sky_lines_fwhm = None
        self.sky_lines_f = None
        self.rss = rss
        self.exptime = rss.info['exptime']
        self.lines_mask = np.ones_like(self.rss.wavelength, dtype=bool)
        super().__init__()

    def estimate_sky(self):
        """TODO"""
        percentiles = np.nanpercentile(self.rss.intensity_corrected, [16, 50, 84], axis=0
                                       ) / self.exptime
        # TODO: compute percentiles of sky to create intensity and variance
        # TODO: Include the possibility of creating 1D or 2D models.
        return percentiles

    def estimate_sky_hist(self):
        """TODO"""
        mode_sky = np.zeros_like(self.rss.wavelength)
        bin_edges = np.logspace(-3, 5, 201)
        bins = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist_sky = np.zeros((mode_sky.size, bins.size))
        for i in range(self.rss.wavelength.size):
            f = self.rss.intensity_corrected[:, i] / self.exptime
            h, _ = np.histogram(f, bins=bin_edges)
            mode_sky[i] = np.nansum(h**5 * bins * np.diff(bin_edges)
                                    ) / np.nansum(h**5 * np.diff(bin_edges))
            hist_sky[i] = h
        return mode_sky, hist_sky, bins

    def fit_continuum(self, spectra, err=None, deg=3):
        print("[SkyFromObject] Estimating sky continuum")
        if self.lines_mask is None:
            self.load_sky_lines()
        if err is not None:
            w = 1/err
        else:
            w = np.ones_like(spectra)
        finite = np.isfinite(spectra) & np.isfinite(w)
        n = spectra[self.lines_mask].size
        dof = n - (deg + 1)
        #while True:
        p = np.polyfit(self.rss.wavelength[self.lines_mask & finite], spectra[self.lines_mask & finite],
                       deg=deg, w=w[self.lines_mask & finite])
        pol = np.poly1d(p)
        chi = (spectra[self.lines_mask & finite] - pol(self.rss.wavelength)[self.lines_mask & finite]
                   ) * w[self.lines_mask & finite]
        reduced_chi2 = np.sum(chi**2) / dof
        sky_cont = pol(self.rss.wavelength)
        #mask_cont = (spectra - sky_cont) / sky_cont < 0
        #sky_cont[mask_cont] = np.sqrt(spectra[mask_cont] * sky_cont[mask_cont])
        self.sky_cont = sky_cont

    def fit_emission_lines(self, cont_clean_spec, errors=None, window_size=100, resampling_wave=0.1, **fit_kwargs):
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
        # Initial guess of line gaussian amplitudes
        p0_amplitude = np.interp(self.sky_lines, self.rss.wavelength, cont_clean_spec)
        p0_amplitude = np.clip(p0_amplitude, a_min=0, a_max=None)
        # Fitter function
        fit_g = fitting.LevMarLSQFitter()
        # Initialize the model with a dummy gaussian
        emission_model = models.Gaussian1D(amplitude=0, mean=0, stddev=0)
        emission_spectra = np.zeros_like(self.rss.wavelength)
        # Select window steps
        wavelength_windows = np.arange(self.rss.wavelength.min(), self.rss.wavelength.max(), window_size)
        # Ensure the last element corresponds to the last wavelength point of the RSS
        wavelength_windows[-1] = self.rss.wavelength.max()
        print("Fitting all emission lines ({}) to continuum-substracted sky spectra".format(self.sky_lines.size))
        # Loop over each spectral window
        for wl_min, wl_max in zip(wavelength_windows[:-1], wavelength_windows[1:]):
            print("Starting fit in the wavelength range [{:.1f}, {:.1f}]".format(wl_min, wl_max))
            mask_lines = (self.sky_lines >= wl_min) & (self.sky_lines < wl_max)
            mask = (self.rss.wavelength >= wl_min) & (self.rss.wavelength < wl_max)
            # Oversample wavelength array to prevent fitting crash for excess of lines
            wave = np.arange(self.rss.wavelength[mask][0], self.rss.wavelength[mask][-1], resampling_wave)
            obs = np.interp(wave, self.rss.wavelength[mask], cont_clean_spec[mask])
            err = np.interp(wave, self.rss.wavelength[mask], errors[mask])
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
                emission_spectra += g(self.rss.wavelength)
                emission_model += g
        return emission_model, emission_spectra

    def load_sky_lines(self, path_to_table=None, default=True, lines_pct=84., **kwargs):
        """TODO"""
        if path_to_table is not None:
            path_to_table = os.path.join(os.path.dirname(__file__), 'input_data', 'sky_lines', path_to_table)
            self.sky_lines, self.sky_lines_fwhm, self.sky_lines_f = np.loadtxt(path_to_table, usecols=(0, 1, 2),
                                                                               unpack=True, **kwargs)
        elif default:
            self.sky_lines, self.sky_lines_fwhm, self.sky_lines_f = uves_sky_lines()
        # Select only lines within the RSS spectral range
        common_lines = (self.sky_lines >= self.rss.wavelength[0]) & (self.sky_lines <= self.rss.wavelength[-1])
        self.sky_lines, self.sky_lines_fwhm, self.sky_lines_f = (self.sky_lines[common_lines],
                                                                 self.sky_lines_fwhm[common_lines],
                                                                 self.sky_lines_f[common_lines])
        # Bin lines unresolved for RSS data
        delta_lambda = np.median(np.diff(self.rss.wavelength))
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
        for line, fwhm in zip(self.sky_lines, self.sky_lines_fwhm):
            # Mask lines up to 2-sigma
            self.lines_mask[(self.rss.wavelength > line - 5)
                            & (self.rss.wavelength < line + 5)] = False


# =============================================================================
# Telluric Correction
# =============================================================================
class Tellurics(Correction):
    """
    Telluric correction produced by atmosphere absorption. # TODO
    """
    def __init__(self,
                 data_container=None,
                 telluric_correction_file=None,
                 n_fibres=10,
                 verbose=False,
                 frac=0.5):

        # Initialise variables
        self.telluric_correction = None
        # Data container (RSS, Cube)
        self.data_container = data_container
        # Set print verbose
        vprint.verbose = verbose
        vprint("\n> Obtaining telluric correction using spectrophotometric star...")

        # Store basic data
        self.wlm = self.data_container.wavelength
        if self.data_container is not None:
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
            self.wlm, self.telluric_correction = np.loadtxt(telluric_correction_file, unpack=True)
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
            Half-window size to account for the instrumental dispersion.
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

    def apply(self, rss, verbose=False, is_combined_cube=False):
        # Set print verbose
        vprint.verbose = verbose
        # Copy input RSS for storage the changes implemented in the task
        rss_out = copy.deepcopy(rss)
        vprint("  Applying telluric correction to this star...")
        if is_combined_cube:
            # TODO: Deprecated
            rss.combined_cube.integrated_star_flux = rss.combined_cube.integrated_star_flux * self.telluric_correction
            for i in range(rss.combined_cube.n_rows):
                for j in range(rss.combined_cube.n_cols):
                    rss.combined_cube.data[:, i, j] = rss.combined_cube.data[:, i, j] * self.telluric_correction
        else:
            rss_out.intensity_corrected *= self.telluric_correction
            rss_out.variance_corrected *= self.telluric_correction**2
        return rss_out

    def save_to_txt(self, filename='telluric_correction.txt', **kwargs):
        """Save telluric correction function to text file."""
        np.savetxt(filename, np.array([self.wlm, self.telluric_correction]).T, **kwargs)

# Mr Krtxo \(ﾟ▽ﾟ)/
