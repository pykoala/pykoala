"""
This module contains the corrections for performing an absolute or relative
flux calibration by accounting for the spectral sensitivity curve as function
of wavelength.
"""
# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib.pyplot as plt
import os

# =============================================================================
# KOALA packages
# =============================================================================
from pykoala import vprint
from pykoala.corrections.correction import CorrectionBase
from pykoala.data_container import SpectraContainer
from pykoala.rss import RSS
from pykoala.cubing import Cube
from pykoala.ancillary import (centre_of_mass, cumulative_1d_moffat, mask_lines,
                               flux_conserving_interpolation)


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
    response_wavelength : None or array-like
        The response wavelength data.
    response_units : float
        Units of the response function, default is 1e16 (erg/s/cm^2/AA).
    """

    name = "FluxCalibration"
    verbosity = True

    def __init__(self, response=None, response_wavelength=None, response_units=1e16,
                 response_file=None, **correction_kwargs):
        """
        Initializes the FluxCalibration object.

        Parameters
        ----------
        response: #TODO
        response_wavelength: #TODO
        response_units: #TODO
        """
        super().__init__(**correction_kwargs)
        
        self.vprint("Initialising Flux Calibration (Spectral Throughput)")

        self.response_units = response_units  # erg/s/cm2/AA
        self.response_wavelength, self.response = response_wavelength, response
        self.response_file = response_file
    
    @classmethod
    def from_text_file(cls, path=None):
        if path is None:
            path = cls.default_extinction
        wavelength, response = np.loadtxt(path, unpack=True)
        return cls(response=response,
                   response_wavelength=wavelength,
                   response_file=path)

    @classmethod
    def auto(cls, data, calib_stars, extract_args=None,
             response_params=None, fnames=None, combine=False):
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
        flux_corrections = []
        flux_cal_results = {}
        # Loop over all standard stars
        for i, name in enumerate(fnames):
            vprint("\n" + "-" * 40 + "\nAutomatic calibration process for {}\n"
                            .format(calib_stars[i]) + "-" * 40 + '\n')
            # Extract flux from std star
            vprint("Extracting stellar flux from data")
            result = FluxCalibration.extract_stellar_flux(
                data[i].copy(), **extract_args)
            flux_cal_results[name] = dict(extraction=result)
            # Interpolate to the observed wavelength
            vprint("Interpolating template to observed wavelength")
            mean_wave = np.nanmean(result['wave_edges'], axis=1)
            result['mean_wave'] = mean_wave

            # Load standard star
            vprint("Loading template spectra")
            ref_wave, ref_spectra = FluxCalibration.read_calibration_star(
                name=calib_stars[i])
            flux_cal_results[name]['ref_wavelength'] = ref_wave
            flux_cal_results[name]['ref_spectra'] = ref_spectra
            # Compute the response curve
            resp_curve, resp_fig = FluxCalibration.get_response_curve(
                mean_wave, result['optimal'][:, 0], ref_wave, ref_spectra,
                **response_params)

            flux_cal_results[name]['wavelength'] = data[i].wavelength.copy()
            flux_cal_results[name]['response'] = resp_curve(
                data[i].wavelength.copy())
            flux_cal_results[name]['response_fig'] = resp_fig
            vprint("-> Saving response as {}".format(name))
            
            flux_corrections.append(cls(response=resp_curve(data[i].wavelength),
                                    response_wavelength=data[i].wavelength))
        if combine:
            master_flux_corr = FluxCalibration.master_flux_auto(flux_corrections)
        else:
            master_flux_corr = None
        return flux_cal_results, flux_corrections, master_flux_corr

    @staticmethod
    def master_flux_auto(flux_calibration_corrections: list):
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
        vprint("Mastering response function")
        spectral_response = []
        if len(flux_calibration_corrections) == 1:
            return flux_calibration_corrections[0]

        for fcal_corr in flux_calibration_corrections[1:]:
            # Update model with the star values
            spectral_response.append(
                fcal_corr.interpolate_response(
                    flux_calibration_corrections[0].response_wavelength))

        vprint("Obtaining median spectral response")
        master_resp = np.nanmedian(spectral_response, axis=0)
        master_flux_calibration = FluxCalibration(
            response=master_resp,
            response_wavelength=flux_calibration_corrections[0]
            .response_wavelength)
        return master_flux_calibration

    @staticmethod
    def extract_stellar_flux(data_container,
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

        vprint("Extracting star flux.\n"
                + " -> Wavelength range={}\n".format(wave_range)
                + " -> Wavelength window={}\n".format(wave_window))

        # Formatting the data
        if isinstance(data_container, RSS):
            vprint("Extracting flux from RSS")
            data = data_container.intensity.copy()
            variance = data_container.variance.copy()
            # Invert the matrix to get the wavelength dimension as 0.
            data, variance = data.T, variance.T
            x = data_container.info['fib_ra']
            y = data_container.info['fib_dec']
        elif isinstance(data_container, Cube):
            vprint("Extracting flux from input Cube")
            data = data_container.intensity.copy()
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
            variance = data_container.variance.copy()
            variance = variance.reshape((variance.shape[0], variance.shape[1]
                                         * variance.shape[2]))
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
        vprint("...Fitting wavelength chuncks...")
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
                vprint("Chunk between {} to {} contains no useful data"
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
            growth_c = np.array(
                [np.nanmean(slice_data[mask & (r2 <= rad)]
                            ) * np.count_nonzero(r2 <= rad) for rad in r2_dummy])
            growth_c_var = np.array(
                [np.nanmean(slice_var[mask & (r2 <= rad)]
                            ) * np.count_nonzero(r2 <= rad) for rad in r2_dummy])
            
            cog_mask = np.isfinite(growth_c)
            if not cog_mask.any():
                vprint("Chunk between {} to {} contains no useful data"
                        .format(wave_edges[0], wave_edges[1]))
                continue
            r2 = r2_dummy
            # r2, growth_c = growth_curve_1d(slice_data, x - x0, y - y0)
            raw_flux.append(growth_c[-1])
            # Fit a light profile
            try:
                if bounds == 'auto':
                    vprint("Automatic fit bounds")
                    p_bounds = ([0, 0, 0], [growth_c[-1] * 2, r2_dummy.max(), 4])
                else:
                    p_bounds = bounds
                # Initial guess
                p0=[growth_c[-1], r2_dummy.mean(), 1.0]
                popt, pcov = curve_fit(
                    profile, r2[cog_mask], growth_c[cog_mask], bounds=p_bounds,
                    p0=p0, **fitter_args)
            except Exception as e:
                vprint("There was a problem during the fit:\n", e)
            else:
                cog.append(growth_c)
                cog_var.append(growth_c_var)
                profile_popt.append(popt)
                profile_var.append(pcov.diagonal())
                running_wavelength.append(wave_edges)
                residuals.append(np.nanmean(growth_c - profile(r2, *popt)))

        if plot:
            fig = FluxCalibration.plot_extraction(
                x, y, x0, y0, data, r2_dummy**0.5, cog,
                np.mean(running_wavelength, axis=1), residuals)
        else:
            fig = None
        result = dict(wave_edges=np.array(running_wavelength),
                      optimal=np.array(profile_popt),
                      variance=np.array(profile_var),
                      figure=fig)
        return result

    @staticmethod
    def plot_extraction(x, y, x0, y0, data, rad, cog, wavelength, residuals):
        """"""
        vprint("Making stellar flux extraction plot")
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8),
                                constrained_layout=True)
        # Plot in the sky
        ax = axs[0, 0]
        ax.set_title("Median intensity")
        mappable = ax.scatter(x, y, c=np.log10(np.nanmedian(data, axis=0)),
                              cmap='nipy_spectral')
        plt.colorbar(mappable, ax=ax, label=r'$\log_{10}$(intensity)')
        ax.plot(x0, y0, '+', ms=10, color='fuchsia')
        # Plot cog
        ax = axs[1, 0]
        ax.set_title("COG shape chromatic differences")
        cog = np.array(cog)
        cog /= cog[:, -1][:, np.newaxis]
        pct_cog = np.nanpercentile(cog, [16, 50, 84], axis=0)
        for p, pct in zip([16, 50, 84], pct_cog):
            ax.plot(rad, pct, label=f'Percentile {p}')
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        ax.set_ylabel('Normalised Curve of Growth')
        ax.set_xlabel('Distance to COM (arcsec)')
        ax.grid(visible=True, which='both')
        ax.legend()
        ax = axs[0, 1]
        ax.plot(wavelength, residuals, '-o', lw=2)
        ax.set_ylabel('Mean model residuals (counts)')
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
                vprint(" - Standard star file: {}\n   · Name: {}"
                      .format(file, names[-1]))
        return np.array(names), files

    @staticmethod
    def read_calibration_star(name=None, path=None):
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
        if name is not None:
            name = name.lower()
            if name[0] != 'f' or 'feige' in name:
                name = 'f' + name
            all_names, _ = FluxCalibration.list_available_stars(verbose=False)
            matched = np.where(all_names == name)[0]
            matched_name = all_names[matched]
            if len(matched_name) > 0:
                if len(matched_name) > 1:
                    vprint("WARNING: More than one file found")
                vprint("Input name {}, matches {}".format(name, matched_name)
                        + f"\nSelecting {matched_name[-1]}")
                path = os.path.join(os.path.dirname(__file__), '..',
                                    'input_data', 'spectrophotometric_stars',
                                    matched_name[-1] + ".dat")
                calib_wave, calib_spectra = np.loadtxt(path, unpack=True,
                                                       usecols=(0, 1))
            else:
                raise FileNotFoundError("Calibration star: {} not found".format(name))
        if path is not None:
            calib_wave, calib_spectra = np.loadtxt(path, unpack=True, usecols=(0, 1))
        return calib_wave, calib_spectra

    @staticmethod
    def get_response_curve(obs_wave, obs_spectra, ref_wave, ref_spectra,
                           pol_deg=None, spline=False, spline_args={},
                           median_filter_n=None, plot=False,
                           mask_absorption=True, mask_zeros=True):
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
        vprint("Computing spectrophotometric response")
        if ref_wave[1] - ref_wave[0] < obs_wave[1] - obs_wave[0]:
            int_ref_spectra = np.interp(obs_wave, ref_wave, ref_spectra)
        else:
            int_ref_spectra = flux_conserving_interpolation(obs_wave, ref_wave,
                                                            ref_spectra)
        raw_response = obs_spectra / int_ref_spectra
        weights = np.ones_like(raw_response)


        if mask_absorption:
            vprint("Masking lines")
            lines_mask = mask_lines(obs_wave)
            weights *= lines_mask
            # obs_spectra = np.interp(obs_wave, obs_wave[lines_mask],
            #                         obs_spectra[lines_mask])
            # int_ref_spectra = np.interp(obs_wave, obs_wave[lines_mask],
            #                             int_ref_spectra[lines_mask])
        
        if mask_zeros:
            vprint("Masking zeros")
            mask = obs_spectra >= 0
            weights *= mask
            # obs_spectra = np.interp(obs_wave, obs_wave[mask],
            #                         obs_spectra[mask])

        # Median filtering
        if median_filter_n is not None:
            filtered_raw_response = median_filter(
                raw_response, size=median_filter_n)

            weights *= 1 / (1 + np.abs(raw_response - filtered_raw_response))**2
            raw_response = filtered_raw_response

        # Polynomial interpolation
        if pol_deg is not None:
            p_fit = np.polyfit(obs_wave, raw_response, deg=pol_deg,
                                w=weights)
            response = np.poly1d(p_fit)
        # Linear interpolation
        elif spline:
            response = UnivariateSpline(obs_wave, raw_response, w=weights,
                                        **spline_args)
        else:
            response = interp1d(obs_wave, raw_response, fill_value=0, bounds_error=False)
            # Spline fit
            # response = UnivariateSpline(wave, obs_spectra / ref_spectra)
        final_response = response(obs_wave)

        if plot:
            fig, axs = plt.subplots(nrows=2, constrained_layout=True,
                                    sharex=True)
            ax = axs[0]
            ax.annotate('{}-deg polynomial fit'.format(pol_deg), xy=(0.05, 0.95), xycoords='axes fraction',
                        va='top', ha='left')
            ax.annotate(r'Median filter size: {}'.format(median_filter_n),
                        xy=(0.05, 0.80), xycoords='axes fraction',
                        va='top', ha='left')
            ax.plot(obs_wave, obs_spectra / int_ref_spectra, '.-', lw=2, label='Obs/Ref')
            ax.plot(obs_wave, raw_response, '-', lw=2, label='Filtered')
            ax.plot(obs_wave, final_response, label='Final response')
            ax.set_xlabel('Wavelength')
            ax.set_title(r'$R(\lambda) [\rm counts\,s^{-1} / (erg\, s^{-1}\, cm^{-2}\, AA^{-1})]$', fontsize=16)
            ax.set_ylabel(r'$R(\lambda)$', fontsize=16)
            ax.legend()
            ax.set_ylim(final_response.min()*0.8, final_response.max()*1.2)

            ax = axs[1]
            ax.plot(ref_wave,  ref_spectra, '-', lw=2, label='Ref')
            ax.plot(obs_wave, obs_spectra / raw_response, '-', lw=2, label='Filtered')
            ax.plot(obs_wave, obs_spectra / final_response, label='Final response')
            ax.set_xlabel('Wavelength')
            ax.set_ylabel(r'$F$', fontsize=16)
            ax.set_ylim(int_ref_spectra.min()*0.8, int_ref_spectra.max()*1.2)
            ax.set_xlim(obs_wave.min(), obs_wave.max())
            # ax.legend(ncol=1)
            twax = ax.twinx()
            twax.plot(obs_wave, weights, label='Relative weights', color='fuchsia',
                      alpha=0.7, lw=0.7)
            twax.legend()
            plt.close(fig)
            return response, fig
        else:
            return response, None

    def interpolate_response(self, wavelength, update=True):
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
        self.vprint("Interpolating spectral response to input wavelength array")
        response = np.interp(wavelength, self.response_wavelength, self.response,
                             right=0., left=0.)
        if update:
            self.vprint("Updating response and wavelength arrays")
            self.response = response
            self.response_wavelength = wavelength
        return response

    def save_response(self, fname):
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
        self.vprint(f"Saving response function at: {fname}")
        np.savetxt(fname, np.array([self.response_wavelength, self.response]).T,
                   header='Spectral Response curve \n wavelength (AA), R ({} counts / [erg/s/cm2/AA])'
                   .format(self.response_units))
        
    def apply(self, spectra_container):
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
        assert isinstance(spectra_container, SpectraContainer)
        spectra_container_out = spectra_container.copy()
        if spectra_container_out.is_corrected(self.name):
            self.vprint("Data already calibrated")
            return spectra_container_out

        # Check that the model is sampled in the same wavelength grid
        if not spectra_container_out.wavelength.size == self.response_wavelength.size or not np.allclose(
            spectra_container_out.wavelength, self.response_wavelength, equal_nan=True):
            response = self.interpolate_response(spectra_container_out.wavelength)
            response_units = self.response_units
        else:
            response = self.response
            response_units = self.response_units

        # Apply the correction
        spectra_container_out.rss_intensity = (spectra_container_out.rss_intensity
                                           / response[np.newaxis, :])
        spectra_container_out.rss_variance = (spectra_container_out.rss_intensity
                                           / response[np.newaxis, :]**2)
        self.record_correction(spectra_container_out, status='applied',
                            units=str(response_units) + ' counts / (erg/s/AA/cm2)')
        return spectra_container_out


# Mr Krtxo \(ﾟ▽ﾟ)/
