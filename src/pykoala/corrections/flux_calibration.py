# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, UnivariateSpline
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
from pykoala.corrections.correction import CorrectionBase
from pykoala.rss import RSS
from pykoala.cubing import Cube
from pykoala.ancillary import (centre_of_mass, cumulative_1d_moffat, mask_lines)


class FluxCalibration(CorrectionBase):
    """
    A class to handle the extraction and application of absolute flux calibration.

    Attributes
    ----------
    name : str
        The name of the Correction.
    verbose : bool
        If True, prints additional information during execution.
    calib_spectra : None or array-like
        The calibration spectra data.
    calib_wave : None or array-like
        The calibration wavelength data.
    response : None or array-like
        The response data.
    resp_wave : None or array-like
        The response wavelength data.
    response_units : float
        Units of the response function, default is 1e16 (erg/s/cm^2/AA).

    Methods
    -------
    interpolate_model(wavelength, update=True)
        Interpolates the spectral response to the input wavelength array.
    auto(data, calib_stars, extract_args=None, response_params=None, save=None, fnames=None, combine=False)
        Performs automatic calibration using a set of standard stars.
    master_flux_auto(flux_cal_results)
        Computes a master response function from results returned by the auto method.
    extract_stellar_flux(data_container, wave_range=None, wave_window=None, profile=cumulative_1d_moffat, bounds='auto', growth_r=np.arange(0, 10, 0.5), plot=False, **fitter_args)
        Extracts the stellar flux from a DataContainer object.
    list_available_stars(verbose=True)
        Lists all available spectrophotometric standard star files.
    read_calibration_star(name=None, path=None, flux_units=None)
        Reads the spectra of a calibration star from a file.
    get_response_curve(wave, obs_spectra, ref_spectra, pol_deg=None, spline=False, spline_args={}, gauss_smooth_sigma=None, plot=False, mask_absorption=True)
        Computes the response curve from observed and reference spectra.
    apply(data_container, response=None, response_units=None)
        Applies the spectral response function to a DataContainer object.
    save_response(fname, response, wavelength, units=None)
        Saves the response function to a file.
    """

    name = "FluxCalibration"
    verbose = True
    calib_spectra = None
    calib_wave = None

    response = None
    resp_wave = None
    response_units = 1e16  # erg/s/cm2/AA

    def __init__(self, path_to_response=None, verbose=True):
        """
        Initializes the FluxCalibration object.

        Parameters
        ----------
        path_to_response : str, optional
            Path to the response file.
        verbose : bool, optional
            If True, prints additional information during execution.
        """
        self.corr_print("Initialising Flux Calibration (Spectral Throughput)")
        self.verbose = verbose
            

        if path_to_response is not None:
            self.corr_print(f"Loading response from file {path_to_response}")
            self.resp_wave, self.response = np.loadtxt(path_to_response, unpack=True)

    def interpolate_model(self, wavelength, update=True):
        """
        Interpolates the spectral response to the input wavelength array.

        Parameters
        ----------
        wavelength : array-like
            The wavelength array to which the response will be interpolated.
        update : bool, optional
            If True, updates the internal response and wavelength attributes.

        Returns
        -------
        response : array-like
            The interpolated response.
        """
        self.corr_print("Interpolating spectral response to input wavelength array")
        response = np.interp(wavelength, self.resp_wave, self.response,
                             right=0., left=0.)
        if update:
            self.corr_print("Updating response and wavelength arrays")
            self.response = response
            self.resp_wave = wavelength
        return response

    def auto(self, data, calib_stars, extract_args=None,
             response_params=None, save=None, fnames=None, combine=False):
        """
        Automatic calibration process for extracting the calibration response curve from a set of stars.

        Parameters
        ----------
        data : list
            List of DataContainers corresponding to standard stars.
        calib_stars : list
            List of stellar names. These names will be used to read the default files in the spectrophotometric_stars library.
        extract_args : dict, optional
            Dictionary containing the parameters used for extracting the stellar flux from the input data.
            See the `FluxCalibration.extract_stellar_flux` method.
        response_params : dict, optional
            Dictionary containing the parameters for computing the response curve from the stellar spectra.
            See the `FluxCalibration.get_response_curve` method.
        save : str or None, optional
            Path where the response curves will be saved.
        fnames : list or None, optional
            Filenames corresponding to the calibration stars.
        combine : bool, optional
            If True, combines individual response curves into a master response curve.

        Returns
        -------
        flux_cal_results : dict
            Dictionary containing the results from the flux calibration process for each of the stars provided.
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
            # Extract flux from std star
            self.corr_print("Extracting stellar flux from data")
            result = self.extract_stellar_flux(copy.deepcopy(data[i]),
                                               **extract_args)
            flux_cal_results[name] = dict(extraction=result)
            # Interpolate to the observed wavelength
            self.corr_print("Interpolating template to observed wavelength")
            mean_wave = np.nanmean(result['wave_edges'], axis=1)
            result['mean_wave'] = mean_wave

            # Load standard star
            self.corr_print("Loading template spectra")
            ref_wave, ref_spectra = self.read_calibration_star(name=calib_stars[i])

            interp_s = np.interp(mean_wave, ref_wave, ref_spectra)

            flux_cal_results[name]['interp'] = interp_s
            # Compute the response curve
            resp_curve, resp_fig = self.get_response_curve(
                mean_wave, result['optimal'][:, 0], interp_s,
                **response_params)
            flux_cal_results[name]['wavelength'] = data[i].wavelength.copy()
            flux_cal_results[name]['response'] = resp_curve(
                data[i].wavelength.copy())
            flux_cal_results[name]['response_fig'] = resp_fig
            self.corr_print("-> Saving response as {}".format(name))
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
        """
        Create a master response function from the results returned by FluxCalibration.auto.

        Parameters
        ----------
        flux_cal_results : dict
            Dictionary containing the results from the flux calibration process for each of the stars provided.

        Returns
        -------
        master_resp : array-like
            The master response function.
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
                             growth_r=np.arange(0, 10, 0.5),
                             plot=False, **fitter_args):
        """
        Extract the stellar flux from an RSS or Cube.

        Parameters
        ----------
        data_container : DataContainer
            Source for extracting the stellar flux.
        wave_range : list, optional
            Wavelength range to use for the flux extraction.
        wave_window : int, optional
            Wavelength window size for averaging the input flux.
        profile : function, optional
            Profile function to model the cumulative light profile. Any function that accepts as first argument
            the square distance (r^2) and returns the cumulative profile can be used. Default is cumulative_1d_moffat.
        bounds : str or tuple, optional
            Bounds for the curve fit. Default is 'auto'.
        growth_r : np.ndarray, optional
            Radial bins relative to the center of the star in arcseconds that will be used to compute the curve of growth. Default is np.arange(0, 10, 0.5).
        plot : bool, optional
            If True, shows a plot of the fit for each wavelength step.
        fitter_args : dict, optional
            Extra arguments to be passed to scipy.optimize.curve_fit

        Returns
        -------
        result : dict
            Dictionary containing the extracted flux and other related data.
        """

        wavelength = data_container.wavelength
        wave_mask = np.ones_like(wavelength, dtype=bool)
        if wave_range is not None:
            wave_mask[(wavelength < wave_range[0]) | (wavelength > wave_range[1])] = False
        if wave_window is None:
            wave_window = 1
        wavelength = wavelength[wave_mask]

        # Curve of growth radial bins
        r2_dummy = growth_r**2

        self.corr_print("Extracting star flux.\n"
                        + " -> Wavelength range={}\n".format(wave_range)
                        + " -> Wavelength window={}\n".format(wave_window))

        # Formatting the data
        if isinstance(data_container, RSS):
            self.corr_print("Extracting flux from RSS")
            data = data_container.intensity.copy()
            variance = data_container.variance.copy()
            # Invert the matrix to get the wavelength dimension as 0.
            data, variance = data.T, variance.T
            x = data_container.info['fib_ra']
            y = data_container.info['fib_dec']
        elif isinstance(data_container, Cube):
            self.corr_print("Extracting flux from input Cube")
            data = data_container.intensity.copy()
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
            variance = data_container.variance.copy()
            variance = variance.reshape((variance.shape[0], variance.shape[1] * variance.shape[2]))
            x, y = np.indices((data_container.n_rows, data_container.n_cols))
            x, y = x.flatten(), y.flatten()
            x, y = data_container.wcs.celestial.pixel_to_world_values(x, y)

        data = data[wave_mask, :]
        variance = variance[wave_mask, :]

        # Declare variables
        residuals = []
        profile_popt = []
        profile_var = []
        running_wavelength = []
        raw_flux = []
        cog = []
        cog_var = []
        # Fitting tolerance and number of evaluations
        if 'ftol' not in fitter_args.keys():
            fitter_args['ftol'] = 0.001
            fitter_args['xtol'] = 0.001
            fitter_args['gtol'] = 0.001
        if 'max_nfev' not in fitter_args.keys():
            fitter_args['max_nfev'] = 1000
 
        # Loop over all spectral slices
        self.corr_print("...Fitting wavelength chuncks...")
        for lambda_ in range(0, wavelength.size, wave_window):
            wave_slice = slice(lambda_, lambda_ + wave_window, 1)
            wave_edges = [wavelength[wave_slice][0], wavelength[wave_slice][-1]]
            slice_data = data[wave_slice]
            slice_var = variance[wave_slice]
            # Compute the median value only considering good values (i.e. excluding NaN)
            slice_data = np.nanmedian(slice_data, axis=0)
            slice_var = np.nanmedian(slice_var, axis=0) / np.sqrt(slice_var.shape[0])
            slice_var[slice_var <= 0] = np.inf

            if not np.isfinite(slice_data).any():
                self.corr_print("Chunk between {} to {} contains no useful data"
                                .format(wave_edges[0], wave_edges[1]))
                continue
            # Pixels without signal are set to 0.
            mask = np.isfinite(slice_data) & np.isfinite(slice_var)
            ###################################################################
            # Computing the curve of growth
            ###################################################################
            x0, y0 = centre_of_mass(slice_data * mask, x, y)
            # Make the growth curve
            r2 = ((x - x0)**2 + (y - y0)**2) * 3600**2  # expressed in arcsec^2
            growth_c = np.array([np.nanmean(slice_data[mask & (r2 <= rad)]) * np.count_nonzero(r2 <= rad) for rad in r2_dummy])
            growth_c_var = np.array([np.nanmean(slice_var[mask & (r2 <= rad)]) * np.count_nonzero(r2 <= rad) for rad in r2_dummy])
            
            cog_mask = np.isfinite(growth_c)
            if not cog_mask.any():
                self.corr_print("Chunk between {} to {} contains no useful data"
                                .format(wave_edges[0], wave_edges[1]))
                continue
            r2 = r2_dummy
            # r2, growth_c = growth_curve_1d(slice_data, x - x0, y - y0)
            raw_flux.append(growth_c[-1])
            # Fit a light profile
            try:
                if bounds == 'auto':
                    self.corr_print("Automatic fit bounds")
                    p_bounds = ([0, 0, 0], [growth_c[-1] * 2, r2_dummy.max(), 4])
                else:
                    p_bounds = bounds
                # Initial guess
                p0=[growth_c[-1], r2_dummy.mean(), 1.0]
                popt, pcov = curve_fit(
                    profile, r2[cog_mask], growth_c[cog_mask], bounds=p_bounds,
                    p0=p0, **fitter_args)
            except Exception as e:
                self.corr_print("There was a problem during the fit:\n", e)
            else:
                cog.append(growth_c)
                cog_var.append(growth_c_var)
                profile_popt.append(popt)
                profile_var.append(pcov.diagonal())
                running_wavelength.append(wave_edges)
                residuals.append(np.nanmean(growth_c - profile(r2, *popt)))

        if plot:
            fig = self.plot_extraction(x, y, x0, y0, r2_dummy**0.5, cog,
                                 np.mean(running_wavelength, axis=1),
                                 residuals)
        else:
            fig = None
        result = dict(wave_edges=np.array(running_wavelength),
                      optimal=np.array(profile_popt),
                      variance=np.array(profile_var),
                      figure=fig)
        return result

    def plot_extraction(self, x, y, x0, y0, data, rad, cog, wavelength, residuals):
        """"""
        self.corr_print("Making stellar flux extraction plot")
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8),
                                gridspec_kw=dict(hspace=0.4, wspace=0.4),
                                constrained_layout=True)
        # Plot in the sky
        ax = axs[0, 0]
        ax.set_title("Median intensity")
        mappable = ax.scatter(x, y, c=np.log10(np.nanmedian(data, axis=0)),
                                cmap='nipy_spectral')
        plt.colorbar(mappable, ax=ax, label=r'$\log_{10}$(intensity)')
        ax.plot(x0, y0, '+', ms=5, color='r')
        # Plot cog
        ax = axs[1, 0]
        cog = np.array(cog)
        cog /= cog[:, -1][:, np.newaxis]
        pct_cog = np.nanpercentile(cog, [16, 50, 84], axis=0)
        for p, pct in zip([16, 50, 84], pct_cog):
            ax.plot(rad, pct, label=f'Percentile {p}')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Normalised Curve of Growth')
        ax.set_xlabel('Distance to COM (arcsec)')
        ax.grid(visible=True, which='both')
        ax = axs[0, 1]
        ax.plot(wavelength, residuals, '-o', color='k', lw=2)
        ax.set_ylabel('Mean residuals')
        ax.set_xlabel('Wavelength')
        axs[1, 1].axis('off')
        plt.close(fig)
        return fig
    
    @staticmethod
    def list_available_stars(verbose=True):
        """
        Lists all available spectrophotometric standard star files.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints the list of available stars.

        Returns
        -------
        stars : list
            List of available spectrophotometric standard stars.
        """
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

    def read_calibration_star(self, name=None, path=None, flux_units=None):
        """
        Reads the spectra of a calibration star from a file.

        Parameters
        ----------
        name : str, optional
            Name of the calibration star.
        path : str, optional
            Path to the file containing the calibration star spectra.
        flux_units : str, optional
            Units of the flux.

        Returns
        -------
        wave : array-like
            Wavelength array of the calibration star.
        flux : array-like
            Flux array of the calibration star.
        """
        if flux_units is None:
            flux_units = 1 / self.response_units
        if name is not None:
            name = name.lower()
            if name[0] != 'f' or 'feige' in name:
                name = 'f' + name
            all_names, all_files = self.list_available_stars(verbose=False)
            matched = np.where(all_names == name)[0]
            matched_name = all_names[matched]
            if len(matched_name) > 0:
                if len(matched_name) > 1:
                    self.corr_print("WARNING: More than one file found")
                self.corr_print("Input name {}, matches {}".format(name, matched_name),
                                f"\n Selecting {matched_name[-1]}")
                path = os.path.join(os.path.dirname(__file__), '..',
                                    'input_data', 'spectrophotometric_stars',
                                    matched_name[-1] + ".dat")
                self.calib_wave, self.calib_spectra = np.loadtxt(path, unpack=True, usecols=(0, 1))
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
        Computes the response curve from observed and reference spectra.

        Parameters
        ----------
        wave : array-like
            Wavelength array.
        obs_spectra : array-like
            Observed spectra.
        ref_spectra : array-like
            Reference spectra.
        pol_deg : int, optional
            Degree of the polynomial fit. If None, no polynomial fit is applied.
        spline : bool, optional
            If True, uses a spline fit.
        spline_args : dict, optional
            Additional arguments for the spline fit.
        gauss_smooth_sigma : float, optional
            Sigma for Gaussian smoothing.
        plot : bool, optional
            If True, plots the response curve.
        mask_absorption : bool, optional
            If True, masks absorption features.

        Returns
        -------
        response_curve : callable
            Function representing the response curve.
        response_fig : matplotlib.figure.Figure
            Figure of the response curve plot, if plot is True.
        """
        print("Computing spectrophotometric response")
        dwave = np.diff(wave)
        wave_edges = np.hstack((wave[0] - dwave[0] / 2, wave[:-1] + dwave / 2,
                                wave[-1] + dwave[-1] / 2))
        if mask_absorption:
            print("Masking lines")
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
            ax = fig.add_subplot(211)
            ax.annotate('{}-deg polynomial fit'.format(pol_deg), xy=(0.05, 0.95), xycoords='axes fraction',
                        va='top', ha='left')
            ax.annotate(r'Gaussian smoothin: $\sigma=${} AA'
                        .format(gauss_smooth_sigma),
                        xy=(0.05, 0.80), xycoords='axes fraction',
                        va='top', ha='left')
            ax.plot(wave, obs_spectra / ref_spectra, '.-', color='k', lw=2, label='Obs/Ref')
            ax.plot(wave, raw_response, '.-', color='orange', lw=2, label='Filtered')
            ax.plot(wave, response(wave), 'r', label='Final response')
            ax.set_xlabel('Wavelength')
            ax.set_title(r'$R(\lambda) [\rm counts\,s^{-1} / (erg\, s^{-1}\, cm^{-2}\, AA^{-1})]$', fontsize=16)
            ax.set_ylabel(r'$R(\lambda)$', fontsize=16)
            ax.legend(loc='lower right')
            ax.set_ylim(response(wave).min(), response(wave).max())

            ax = fig.add_subplot(212)
            ax.plot(wave,  ref_spectra, '.-', color='k', lw=2, label='Ref')
            ax.plot(wave, obs_spectra / raw_response, '.-', color='orange', lw=2, label='Filtered')
            ax.plot(wave, obs_spectra / response(wave), 'r', label='Final response')
            ax.set_xlabel('Wavelength')
            ax.set_ylabel(r'$F$', fontsize=16)
            ax.set_ylim(ref_spectra.min(), ref_spectra.max())
            ax.legend(ncol=1)
            plt.close(fig)
            return response, fig
        else:
            return response, None

    def apply(self, data_container, response=None, response_units=None):
        """
        Computes the response curve from observed and reference spectra.

        Parameters
        ----------
        wave : array-like
            Wavelength array.
        obs_spectra : array-like
            Observed spectra.
        ref_spectra : array-like
            Reference spectra.
        pol_deg : int, optional
            Degree of the polynomial fit. If None, no polynomial fit is applied.
        spline : bool, optional
            If True, uses a spline fit.
        spline_args : dict, optional
            Additional arguments for the spline fit.
        gauss_smooth_sigma : float, optional
            Sigma for Gaussian smoothing.
        plot : bool, optional
            If True, plots the response curve.
        mask_absorption : bool, optional
            If True, masks absorption features.

        Returns
        -------
        response_curve : callable
            Function representing the response curve.
        response_fig : matplotlib.figure.Figure
            Figure of the response curve plot, if plot is True.
        """

        if data_container.is_corrected(self.name):
            print("Data already calibrated")
            return data_container

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
                            units=str(response_units) + ' counts / (erg/s/AA/cm2)')
        return data_container

    def save_response(self, fname, response, wavelength, units=None):
        """
        Saves the response function to a file.

        Parameters
        ----------
        fname : str
            File name for saving the response function.
        response : array-like
            The response function data.
        wavelength : array-like
            The wavelength array corresponding to the response function.
        units : str, optional
            Units of the response function.

        Returns
        -------
        None
        """
        # TODO Include response units in header
        if units is None:
            units = self.response_units
        self.corr_print(f"Saving response function at :{fname}")
        np.savetxt(fname, np.array([wavelength, response]).T,
                   header='Spectral Response curve \n wavelength (AA), R ({} counts / [erg/s/cm2/AA])'
                   .format(1 / units))

# Mr Krtxo \(ﾟ▽ﾟ)/
