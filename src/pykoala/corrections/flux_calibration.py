# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from photutils.centroids import centroid_2dg
import copy
import matplotlib.pyplot as plt
import os

# =============================================================================
# Astropy and associated packages
# =============================================================================

# =============================================================================
# KOALA packages
# =============================================================================
from pykoala.exceptions.exceptions import ClassError, FitError, CalibrationError
from pykoala.corrections.correction import CorrectionBase
from pykoala.rss import RSS
from pykoala.cubing import Cube
from pykoala.ancillary import (centre_of_mass, cumulative_1d_moffat,
                             growth_curve_1d, flux_conserving_interpolation,
                             mask_lines, gaussian_2d)


class FluxCalibration(CorrectionBase):
    """
    This module contains the methods for extracting and applying an absolute flux calibration.

    Description
    -----------


    Attributes
    ----------

    Methods
    -------

    Example
    -------
    """
    name = "FluxCalibration"
    verbose = True
    calib_spectra = None
    calib_wave = None

    response = None
    resp_wave = None
    response_units = 1e16  # erg/s/cm2/AA

    def __init__(self, path_to_response=None, verbose=True):
        self.corr_print("Initialising Flux Calibration (Spectral Throughput)")
        self.verbose = verbose
            

        if path_to_response is not None:
            self.corr_print(f"Loading response from file {path_to_response}")
            self.resp_wave, self.response = np.loadtxt(path_to_response, unpack=True)

    def interpolate_model(self, wavelength, update=True):
        """Interpolate the spectral response to the input wavelength array."""
        self.corr_print("Interpolating spectral response to input wavelength array")
        response = np.interp(wavelength, self.resp_wave, self.response, right=0., left=0.)
        if update:
            self.response = response
            self.resp_wave = wavelength
        return response

    def auto(self, data, calib_stars, extract_args=None,
             response_params=None, save=None, fnames=None, combine=False):
        """Automatic calibration process for extracting the calibration response curve from a set of stars.

        Parameters
        ----------
        - data: (list) List of DataContainers corresponding to standard stars.
        - calib_stars: (list) List of stellar names. This names will be used to read
        the default files in the spectrophotometric_stars library.
        - extract_args: (dict) Dictionary containing the parameters used for extracting
        the stellar flux.
        The default dictionary corresponds to:
            {wave_rage: None, # Wavelength range used to derive the response function
            wave_window: 30, # Wavelength window in AA used to average the flux
            plot: False,  # If True, it will provide a plot showing the extraction results
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf])  # Bounds used for fitting
            }
            See FluxCalibration.extract_stellar_flux
        - response_params: default=None
        - save: bool, default=True
        - fnames: default=None


        Returns
        -------
        - flux_cal_results: (dict)
            Dictionary containing the results from the flux calibration process
            for each of the stars provided. Results for each star are contained
            within a dictionary that includes:
                · extraction: 
                    mean_wave
                    optimal
                · interp: Polynomial interpolation of the response function
                · response: Instrumental throughput.
                · resp_fig: If plot=True in **respones_params it will contain
                a plot of the wavelength versus the instrument response function.
            
        """
        if extract_args is None:
            # Default extraction arguments
            extract_args = dict(wave_range=None, wave_window=30, plot=False)
        if response_params is None:
            # Default response curve parameters
            response_params = dict(pol_deg=5, plot=False)
        if fnames is None:
            fnames = calib_stars.copy()

        # Initialise variables
        flux_cal_results = {}
        # Loop over all standard stars
        for i, name in enumerate(fnames):
            self.corr_print("\n" + "-" * 40 + "\nAutomatic calibration process for {}\n"
                            .format(calib_stars[i]) + "-" * 40 + '\n')
            self.corr_print("Loading template spectra")

            # Load standard star
            w, s = self.read_calibration_star(name=calib_stars[i])

            # Extract flux from std star
            self.corr_print("Extracting stellar flux from data")
            result = self.extract_stellar_flux(copy.deepcopy(data[i]),
                                               **extract_args)
            flux_cal_results[name] = dict(extraction=result)
            # Interpolate to the observed wavelength
            self.corr_print("Interpolating template to observed wavelength")
            interp_s = flux_conserving_interpolation(
                new_wavelength=result['mean_wave'], wavelength=w, spectra=s)
            flux_cal_results[name]['interp'] = interp_s
            # Compute the response curve
            resp_curve, resp_fig = self.get_response_curve(
                result['mean_wave'], result['optimal'][:, 0], interp_s,
                **response_params)
            flux_cal_results[name]['wavelength'] = data[i].wavelength.copy()
            flux_cal_results[name]['response'] = resp_curve(
                data[i].wavelength.copy())
            flux_cal_results[name]['response_fig'] = resp_fig
            self.corr_print("-> Saving response as {}".format(name))
            # TODO: dump the flux_cal_results to a json / yaml file
            if save:
                self.save_response(
                    fname=os.path.join(save, 'response_' + name),
                    response=resp_curve(data[i].wavelength),
                    wavelength=data[i].wavelength)
        if combine:
            self.master_flux_auto(flux_cal_results)
            if save:
                self.save_response(
                    fname=os.path.join(save, 'master_response'),
                    response=self.response,
                    wavelength=self.resp_wave)

        return flux_cal_results
    
    def master_flux_auto(self, flux_cal_results):
        """Create a master response function.
        
        Compute a master reponse function from the results returned by
        FluxCalibration.auto
        """
        self.corr_print("Mastering response function")
        reference_wavelength = None
        spectral_response = []
        for star, star_res in flux_cal_results.items():
            if reference_wavelength is None:
                reference_wavelength = star_res['wavelength']
            # Update model with the star values
            self.response = star_res['response']
            self.resp_wave = star_res['wavelength']
            spectral_response.append(
                self.interpolate_model(reference_wavelength))

        self.corr_print("Obtaining median spectral response")
        master_resp = np.nanmedian(spectral_response, axis=0)
        self.response = master_resp
        self.resp_wave = reference_wavelength
        return master_resp

    def extract_stellar_flux(self, data_container,
                             wave_range=None, wave_window=None,
                             profile=cumulative_1d_moffat,
                             bounds='auto',
                             plot=False, **fitter_args):
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
        wavelength = wavelength[wave_mask]

        self.corr_print("Extracting star flux.\n"
                        + " -> Wavelength range={}\n".format(wave_range)
                        + " -> Wavelength window={}\n".format(wave_window))

        # Formatting the data
        if type(data_container) is RSS:
            self.corr_print("Extracting flux from RSS")
            data = data_container.intensity.copy()
            # Invert the matrix to get the wavelength dimension as 0.
            data = data.T
            x = data_container.info['fib_ra']
            y = data_container.info['fib_dec']
        elif type(data_container) is Cube:
            self.corr_print("Extracting flux from Cube")
            data = data_container.intensity.copy()
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
            x, y = np.meshgrid(np.arange(data_container.n_cols),
                               np.arange(data_container.n_rows))
            x, y = x.flatten(), y.flatten()

        data = data[wave_mask, :]
        
        # Declare variables
        residuals = []
        profile_popt = []
        profile_var = []
        running_wavelength = []
        raw_flux = []
        # Fitting tolerance and number of evaluations
        if 'ftol' not in fitter_args.keys():
            fitter_args['ftol'] = 0.001
            fitter_args['xtol'] = 0.001
            fitter_args['gtol'] = 0.001
        if 'max_nfev' not in fitter_args.keys():
            fitter_args['max_nfev'] = 10000
 
        # Loop over all spectral slices
        self.corr_print("...Fitting wavelength chuncks...")
        for lambda_ in range(0, wavelength.size, wave_window):
            wave_slice = slice(lambda_, lambda_ + wave_window, 1)
            mean_wave = np.nanmedian(wavelength[wave_slice])
            slice_data = data[wave_slice]
            # Compute the median value only considering good values (i.e. excluding NaN)
            slice_data = np.nanmedian(slice_data, axis=0)
            if not np.isfinite(slice_data).any():
                self.corr_print("Chunk at {}+-{} contains no useful data"
                                .format(mean_wave, wave_window))
                continue
            # Positions without signal are set to 0.
            slice_data = np.nan_to_num(slice_data, posinf=0., neginf=0.)
            ###################################################################
            # Computing the curve of growth
            ###################################################################
            x0, y0 = centre_of_mass(slice_data, x, y)
            y_min, y_max = y.min(), y.max()
            x_min, x_max = x.min(), x.max()
            # Make the growth curve
            r2, growth_c = growth_curve_1d(slice_data, x - x0, y - y0)
            raw_flux.append(growth_c[-1])
            # Fit a light profile
            try:
                if bounds == 'auto':
                    self.corr_print("Automatic fit bounds")
                    p_bounds = ([0, 0, 0], [growth_c[-1] * 2,
                                            np.max((x_max - x_min, y_max - y_min)),
                                            4])
                else:
                    p_bounds = bounds
                # Initial guess
                p0=[growth_c[-1], np.min((x_max - x_min, y_max - y_min)) / 3, 1]
                # print("Bounds: ", bounds, "Initial guess: ", p0)
                popt, pcov = curve_fit(
                    profile, r2, growth_c, bounds=p_bounds,
                    p0=p0,
                    **fitter_args)
                # print(popt[0], growth_c[-1])
            except FitError:
                self.corr_print(
                    "Fit at wavelength {:.1f} [{:.1f}-{:.1f}] ***unsuccessful***"
                      .format(mean_wave, wavelength[wave_slice][0], wavelength[wave_slice][-1]))
                raise FitError()
            else:
                profile_popt.append(popt)
                profile_var.append(pcov.diagonal())
                running_wavelength.append(mean_wave)
                residuals.append(np.nanmean(growth_c - profile(r2, *popt)))
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(211)
            mappable = ax.scatter(x, y, c=np.nanmean(data, axis=0),
                                  cmap='Greys_r')
            plt.colorbar(mappable, ax=ax, label='Flux')
            ax.plot(x0, y0, '+', ms=5, color='r')

            ax = fig.add_subplot(212)
            ax.plot(running_wavelength, residuals, '-o', color='k', lw=2)
            ax.set_ylabel('Mean residuals')
            ax.set_xlabel('Wavelength')
            plt.close(fig)
        else:
            fig = None
        result = dict(mean_wave=np.array(running_wavelength),
                      optimal=np.array(profile_popt),
                      variance=np.array(profile_var),
                      figure=fig)
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
            flux_units = 1 / self.response_units
        if name is not None:
            name = name.lower()
            if name[0] != 'f' or 'feige' in name:
                name = 'f' + name
            all_names, all_files = self.list_available_stars(verbose=False)
            match = np.where(all_names == name)[0]
            if len(match) > 0:
                self.corr_print("Input name {}, matches {}".format(name, all_names[match]))
                path = os.path.join(os.path.dirname(__file__), '..',
                                    'input_data', 'spectrophotometric_stars',
                                    all_files[match[-1]])
                self.calib_wave, self.calib_spectra = np.loadtxt(path, unpack=True, usecols=(0, 1))
                if len(match) > 1:
                    self.corr_print("WARNING: More than one file found")
            else:
                raise FileNotFoundError("Calibration star: {} not found".format(name))
        if path is not None:
            self.calib_wave, self.calib_spectra = np.loadtxt(path, unpack=True, usecols=(0, 1))
        return self.calib_wave, self.calib_spectra

    @staticmethod
    def get_response_curve(wave, obs_spectra, ref_spectra,
                           pol_deg=None, spline=False, spline_args={}, gauss_smooth_sigma=None, plot=False,
                           mask_absorption=True):
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
            be the same as the wavelength array (AA).
        plot: bool (default=False)
            If True, will return a figure containing a plot of the interpolation.

        Returns
        -------
            response: function
                Interpolator function of the response curve.
            fig: default=None, if plot=True it will return a plt.figure() 
            containing the response function in terms of wavelength

        """
        dwave = np.diff(wave)
        wave_edges = np.hstack((wave[0] - dwave[0] / 2, wave[:-1] + dwave / 2,
                                wave[-1] + dwave[-1] / 2))
        if mask_absorption:
            lines_mask = mask_lines(wave)
            obs_spectra = np.interp(wave, wave[lines_mask],
                                    obs_spectra[lines_mask])
            ref_spectra = np.interp(wave, wave[lines_mask],
                                    ref_spectra[lines_mask])
        raw_response = obs_spectra / ref_spectra
        
        # Gaussian smoothing
        if gauss_smooth_sigma is not None:
            sigma_pixels = gauss_smooth_sigma / np.mean(dwave)
            raw_response = gaussian_filter1d(
                raw_response, sigma=sigma_pixels)

        # Polynomial interpolation
        if pol_deg is not None:
            p_fit = np.polyfit(wave, raw_response, deg=pol_deg)
            response = np.poly1d(p_fit)
        # Linear interpolation
        elif spline:
            response = UnivariateSpline(wave, raw_response, **spline_args)
        else:
            response = interp1d(wave, raw_response, fill_value=0, bounds_error=False)
            # Spline fit
            # response = UnivariateSpline(wave, obs_spectra / ref_spectra)
        if plot:
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            ax.annotate('{}-deg polynomial fit'.format(pol_deg), xy=(0.05, 0.95), xycoords='axes fraction',
                        va='top', ha='left')
            ax.annotate(r'Gaussian smoothin: $\sigma=${} AA'
                        .format(gauss_smooth_sigma),
                        xy=(0.05, 0.80), xycoords='axes fraction',
                        va='top', ha='left')
            ax.plot(wave, obs_spectra / ref_spectra, 'k', lw=2, label='Obs/Ref')
            ax.plot(wave, response(wave), 'r', label='Final response')
            ax.set_xlabel('Wavelength')
            ax.set_ylabel(r'$R(\lambda) = F_{obs} / F_{ref}$', fontsize=16)
            ax.legend(loc='lower right')
            plt.close(fig)
            return response, fig
        else:
            return response, None


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

    def apply(self, data_container, response=None, response_units=None):
        """Apply a spectral response function to a Data Container object.
        If the object has already been calibrated an exception will be raised.

        Parameters
        ----------
        response: np.ndarray
            Response function interpolated to the same wavelength points as the Data Container data
        data_container: DataContainer
            Data to be calibrated.
        """

        if data_container.is_corrected(self.name):
            raise CalibrationError()
        
        if response is None:
            if self.response is None:
                raise NameError("Spectral response function not provided")
            # Check that the model is sampled in the same wavelength grid
            if not data_container.wavelength.size == self.resp_wave.size or not np.allclose(
                data_container.wavelength, self.resp_wave, equal_nan=True):
                response = self.interpolate_model(data_container.wavelength)
                response_units = self.response_units
            else:
                response = self.response
                response_units = self.response_units
        
        # Account for the DataContainer data dimensions
        print(np.max(response))
        if type(data_container) is Cube:
            self.corr_print("Applying Flux Calibration to input Cube")
            response = response[:, np.newaxis, np.newaxis]
        elif type(data_container) is RSS:
            self.corr_print("Applying Flux Calibration to input RSS")
            response = response[np.newaxis, :]
        else:
            raise NameError(f"Unrecognised DataContainer of type : {type(data_container)}")
        # Apply the correction
        data_container.intensity /= response
        data_container.variance /= response**2
        self.log_correction(data_container, status='applied',
                            units=response_units)
        return data_container

    def save_response(self, fname, response, wavelength, units=None):
        # TODO Include response units in header
        if units is None:
            units = self.response_units
        np.savetxt(fname, np.array([wavelength, response]).T,
                   header='Spectral Response curve \n wavelength (AA), R ({} counts / [erg/s/cm2/AA])'
                   .format(1 / units))

# Mr Krtxo \(ﾟ▽ﾟ)/
