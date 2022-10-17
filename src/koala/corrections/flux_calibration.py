"""
Module containing the cube class... TODO: Add proper description.
"""

# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import copy
import matplotlib.pyplot as plt
import os

# =============================================================================
# Astropy and associated packages
# =============================================================================

# =============================================================================
# KOALA packages
# =============================================================================
from koala.exceptions import ClassError, FitError, CalibrationError
from koala.corrections.correction import Correction
from koala.rss import RSS
from koala.cubing import Cube
from koala.ancillary import centre_of_mass, cumulative_1d_moffat, growth_curve_1d, flux_conserving_interpolation

"""
main methods

extract_spectra: DONE
read_response_from_file: DONE
extract response curve: DONE
apply response

"""


class FluxCalibration(Correction):
    """
    This class contains the methods for extracting and applying an absolute flux calibration.
    """
    def __init__(self, data_container=None, extract_spectra=None, extract_args=None, ):
        print("[Flux Calib.] Initialising Flux Calibration (Spectral Throughput)")
        self.calib_spectra = None
        self.calib_wave = None
        self.calib_units = 1e-16

    def auto(self, data, calib_stars, extract_args=None, response_params=None, save=True, fnames=None):
        """Automatic calibration process for extracting the calibration response curve from a set of stars"""
        if extract_args is None:
            extract_args = dict(wave_range=None, wave_window=30, plot=False,
                                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        if response_params is None:
            response_params = dict(pol_deg=5, plot=False)
        if save and fnames is None:
            fnames = calib_stars

        calib_stars_wave = []
        calib_stars_spectra = []
        extraction_results = []
        flux_cal_results = {}
        for star_i in range(len(calib_stars)):
            print("-" * 40 + "\nAutomatic calibration process for {}\n".format(calib_stars[star_i]) + "-" * 40)
            print("-> Loading template spectra")
            w, s = self.read_calibration_star(name=calib_stars[star_i])
            calib_stars_wave.append(w)
            calib_stars_spectra.append(s)
            print("-> Extracting stellar flux from data")
            result = self.extract_stellar_flux(copy.deepcopy(data[star_i]), **extract_args)
            flux_cal_results[calib_stars[star_i]] = dict(extraction=result)
            print("-> Interpolating template to observed flux wavelength")
            interp_s = flux_conserving_interpolation(
                new_wavelength=result['mean_wave'], wavelength=w, spectra=s)
            flux_cal_results[calib_stars[star_i]]['interp'] = interp_s
            resp_curve = self.get_response_curve(result['mean_wave'], result['optimal'][:, 0], interp_s,
                                                 **response_params)
            flux_cal_results[calib_stars[star_i]]['response'] = resp_curve(data[star_i].wavelength.copy())
            print("-> Saving response as {}".format(calib_stars[star_i]))
            if save:
                self.save_response(
                    fname=fnames[star_i],
                    response=resp_curve(data[star_i].wavelength),
                    wavelength=data[star_i].wavelength)
        return flux_cal_results

    @staticmethod
    def extract_stellar_flux(data_container, wave_range=None, wave_window=None,
                             profile=cumulative_1d_moffat, plot=False, **fitter_args):
        """
        Extract the stellar flux from an RSS or Cube.

        Parameters
        ----------
        data_container: RSS or Cube
            Source for extracting the stellar flux
        wave_range: list
            wavelength range to use for
        wave_window: int
            wavelength window size for averaging the input flux.
        profile: function, default=commulative_1d_moffat
            Profile function to model the cumulative ligth profile. Any function that accepts as first argument
            the square distance (r^2) and returns the cumulative profile can be used.
        plot: bool, default=False
            If True, for each wavelength step the fit will be shown in a plot.
        fitter_args: kwargs
            extra arguments to be passed to scipy.optimize.curve_fit


        Returns
        -------
        results: dict
            Dictionary containing the results from the fits:
                - mean_wave: mean wavelength value for each fit.
                - optimal: optimal parameters for the profile function.
                - variance: variance of each parameter.
        """
        wavelength = data_container.wavelength
        wave_mask = np.ones_like(wavelength, dtype=bool)
        if wave_range is not None:
            wave_mask[(wavelength < wave_range[0]) | (wavelength > wave_range[1])] = False
        if wave_window is None:
            wave_window = 1
        # Obtaining the data
        if type(data_container) is RSS:
            data = data_container.intensity_corrected.copy()
            # Invert the matrix to get the wavelength dimension as 0.
            data = data.T
            x = data_container.info['fib_ra_offset']
            y = data_container.info['fib_dec_offset']
        elif type(data_container) is Cube:
            data = data_container.intensity.copy()
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
            x = data_container.info['spax_ra_offset'].flatten()
            y = data_container.info['spax_dec_offset'].flatten()
        else:
            raise ClassError([RSS, Cube], type(data_container))
        wavelength = wavelength[wave_mask]
        data = data[wave_mask, :]
        # Loop over all spectral slices
        profile_popt = []
        profile_var = []
        running_wavelength = []
        for lambda_ in range(0, wavelength.size, wave_window):
            wave_slice = slice(lambda_, lambda_ + wave_window, 1)
            mean_wave = np.nanmean(wavelength[wave_slice])
            slice_data = data[wave_slice]
            # Compute the mean value only considering good values (i.e. excluding NaN)
            slice_data = np.nanmean(slice_data, axis=0)
            # Positions without signal are set it to 0.
            slice_data = np.nan_to_num(slice_data)
            # Get the centre of mass
            x0, y0 = centre_of_mass(slice_data, x, y)
            # Make growth curve
            r2, growth_c = growth_curve_1d(slice_data, x - x0, y - y0)
            # Fit a light profile
            try:
                popt, pcov = curve_fit(profile, r2, growth_c, **fitter_args)
            except FitError:
                print("[Flux Calib.] Fit at wavelength {:.1f} [{:.1f}-{:.1f}] ***unsuccessful***"
                      .format(mean_wave, wavelength[wave_slice][0], wavelength[wave_slice][-1]))
                raise FitError()
            else:
                profile_popt.append(popt)
                profile_var.append(pcov.diagonal())
                running_wavelength.append(mean_wave)
                if plot:
                    plt.figure()
                    plt.plot(r2, growth_c, 'k', lw=2)
                    plt.plot(r2, profile(r2, *popt), 'r')
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.show()
        result = dict(mean_wave=np.array(running_wavelength), optimal=np.array(profile_popt),
                      variance=np.array(profile_var))
        return result

    def read_calibration_star(self, name=None, path=None, flux_units=None):
        """
        Read from a file the spectra of a calibration star. TODO...
        Parameters
        ----------
        name
        path
        flux_units

        Returns
        -------

        """
        if flux_units is None:
            flux_units = self.calib_units
        if name is not None:
            name = name.lower()
            if name[0] != 'f':
                name = 'f' + name
            all_names, all_files = self.list_available_stars(verbose=False)
            match = np.where(all_names == name)[0]
            if len(match) > 0:
                print("Input name {}, matches {}".format(name, all_names[match]))
                path = os.path.join(os.path.dirname(__file__), '..',
                                    'input_data', 'spectrophotometric_stars',
                                    all_files[match[0]])
                self.calib_wave, self.calib_spectra = np.loadtxt(path, unpack=True, usecols=(0, 1))
                if len(match) > 1:
                    print("WARNING: More than one file found")
            else:
                raise FileNotFoundError("Calibration star: {} not found".format(name))
        if path is not None:
            self.calib_wave, self.calib_spectra = np.loadtxt(path, unpack=True, usecols=(0, 1))
        return self.calib_wave, self.calib_spectra

    @staticmethod
    def get_response_curve(wave, obs_spectra, ref_spectra, pol_deg=None, gauss_smooth_sigma=None, plot=False):
        """
        Compute the response curve (spectral throughput) from a given observed spectrum and a reference function.
        Parameters
        ----------
        wave: np.ndarray
            Wavelength array of observed and reference spectra.
        obs_spectra: np.ndarray
            Observed spectra
        ref_spectra: np.ndarry
            Reference spectra
        pol_deg: int (default=None)
            If not None, the response curve will be interpolated to a polynomial of degree pol_deg.
        gauss_smooth_sigma: float (default=None)
            If not None, the cumulative response function will be gaussian smoothed before interpolation. The units must
            be the same as the wavelength array.
        plot: bool (default=False)
            If True, will return a figure containing a plot of the interpolation.

        Returns
        -------
            response: function
                Interpolator function of the response curve.
            fig: plt.figure()
                If "plot=True".

        """
        dwave = np.diff(wave)
        wave_edges = np.hstack((wave[0] - dwave[0] / 2, wave[:-1] + dwave / 2,
                                wave[-1] + dwave[-1] / 2))
        cum_response = np.cumsum(obs_spectra / ref_spectra * np.diff(wave_edges))
        # Gaussian smoothing
        if gauss_smooth_sigma is not None:
            sigma_pixels = gauss_smooth_sigma / np.mean(dwave)
            cum_response = gaussian_filter1d(cum_response, sigma=sigma_pixels)
        # Polynomial interpolation
        if pol_deg is not None:
            p_fit = np.polyfit(wave, cum_response, deg=pol_deg)
            cum_r_polfit = np.poly1d(p_fit)
            response = cum_r_polfit.deriv(m=1)
        # Linear interpolation
        else:
            cum_response = np.insert(cum_response, 0, cum_response[0] - (cum_response[1] - cum_response[0]) / 2)
            r = np.diff(cum_response) / np.diff(wave_edges)
            response = interp1d(wave, r, fill_value=0, bounds_error=False)
            # Spline fit
            # response = UnivariateSpline(wave, obs_spectra / ref_spectra)

        if plot:
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(211)
            ax.annotate('{}-deg polynomial fit'.format(pol_deg), xy=(0.05, 0.95), xycoords='axes fraction',
                        va='top', ha='left')
            ax.plot(wave, cum_response, 'k', lw=2)
            ax.plot(wave, response(wave), 'r', lw=0.7)
            ax.set_ylabel(r'$\int_{\lambda_{min}}^{\lambda_{max}} \frac{F_{obs}}{F_{ref}} d\lambda$', fontsize=16)
            ax = fig.add_subplot(212)
            ax.plot(wave, obs_spectra / ref_spectra, 'k', lw=2)
            ax.plot(wave, response(wave), 'r')
            ax.set_xlabel('Wavelength')
            ax.set_ylabel(r'$R(\lambda) = F_{obs} / F_{ref}$', fontsize=16)
            fig.subplots_adjust(hspace=0.4)
            fig.show()
            return response, fig
        else:
            return response

    @staticmethod
    def list_available_stars(verbose=True):
        """List all available spectrophotometric standard star files."""
        files = os.listdir(os.path.join(os.path.dirname(__file__), '..',
                                        'input_data', 'spectrophotometric_stars'))
        files = np.sort(files)
        names = []
        for file in files:
            names.append(file.split('.')[0].split('_')[0])
            if verbose:
                print(" - Standard star file: {}\n   · Name: {}"
                      .format(file, names[-1]))
        return np.array(names), files

    @staticmethod
    def apply(response, data_container):
        """Apply a spectral response function to a Data Container object.
        If the object has already been calibrated an exception will be raised.

        Parameters
        ----------
        response: np.ndarray
            Response function interpolated to the same wavelength points as the Data Container data
        data_container: DataContainer
            Data to be calibrated.
        """
        print("[Flux Calib.] Applying response function to {}".format(data_container.info['name']))
        if data_container.is_corrected('flux_calibration'):
            raise CalibrationError()
        else:
            ss = [None] * data_container.intensity_corrected.ndim
            ss[0] = slice(None)
            data_container.intensity_corrected /= response[ss]
            data_container.log['corrections']['flux_calibration'] = 'applied'

    def save_response(self, fname, response, wavelength, units=None):
        # TODO Include R units in header
        if units is None:
            units = self.calib_units
        np.savetxt(fname, np.array([wavelength, response]).T,
                   header='Spectral Response curve \n wavelength (AA), R ({} counts / [erg/s/cm2/AA])'
                   .format(1/units))

# Mr Krtxo \(ﾟ▽ﾟ)/
