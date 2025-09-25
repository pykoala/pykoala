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
from scipy.interpolate import interp1d, make_smoothing_spline

import matplotlib.pyplot as plt
import os
from astropy import units as u
from astropy.visualization import quantity_support
from astropy.coordinates import SkyCoord
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala import vprint
from pykoala.corrections.correction import CorrectionBase
from pykoala.data_container import SpectraContainer
from pykoala.data_container import RSS
from pykoala.data_container import Cube
from pykoala.ancillary import (centre_of_mass, cumulative_1d_moffat,
                               mask_lines, mask_telluric_lines,
                               flux_conserving_interpolation, check_unit)

quantity_support()

def curve_of_growth(radii : u.Quantity, data : u.Quantity,
                    ref_radii : u.Quantity, mask=None) -> u.Quantity:
    """
    Compute a curve of growth (cumulative flux versus radius) and evaluate it
    at the requested reference radii.

    The algorithm sorts samples by radius, applies the mask (if given), computes
    the cumulative sum of the data over increasing radius, and linearly
    interpolates that cumulative sum to the reference radii. The returned curve
    is non decreasing by construction when data are non negative.

    Parameters
    ----------
    radii
        Radial distance of each sample. Must be a 1D astropy Quantity. It will
        be converted to the unit of ref_radii.
    data
        Flux value of each sample. Must be a 1D astropy Quantity with the same
        length as radii. The returned curve has the same unit as data.
    ref_radii
        Radii at which to evaluate the curve of growth. Must be a 1D astropy
        Quantity. radii will be converted to this unit for sorting and
        interpolation.
    mask
        Optional boolean array with the same length as radii. Samples with
        mask False are ignored. If None, all samples are considered.

    Returns
    -------
    cog
        Curve of growth evaluated at ref_radii, as an astropy Quantity with the
        same unit as data. If no valid samples are available, zeros are
        returned.

    Notes
    -----
    1. Non finite values in radii or data are ignored.
    2. Duplicate radii are handled by taking the cumulative sum at the last
       occurrence of each unique radius.
    3. Linear interpolation is used between unique radii. Values below the
       smallest radius are set to zero. Values above the largest radius are
       set to the total cumulative flux.

    Raises
    ------
    ValueError
        If radii and data shapes differ, or if mask has an incompatible shape.
    """
    if mask is None:
        mask = np.ones(data.size, dtype=bool)

    radii = check_unit(radii, ref_radii.unit)
    r_val = radii.to_value(ref_radii.unit)
    d_val = data.value
    data_unit = data.unit

    valid = np.isfinite(r_val) & np.isfinite(d_val)
    if mask is not None:
        valid &= mask.astype(bool)

    if not np.any(valid):
        # No usable samples
        return np.zeros(ref_radii.size, dtype=float) * data_unit

    sort_idx = np.argsort(r_val)
    r_sorted = r_val[sort_idx]
    d_sorted = d_val[sort_idx]
    v_sorted = valid[sort_idx]

    # Zero out invalid samples and accumulate
    d_sorted = np.where(v_sorted, d_sorted, 0.0)
    cumsum = np.cumsum(d_sorted)
    # Map each unique radius to the last cumulative value at or below it
    u_r = np.unique(r_sorted)
    # index of last element <= each unique radius
    last_idx = np.searchsorted(r_sorted, u_r, side="right") - 1
    cog_at_u = cumsum[last_idx]
    # Enforce non decreasing curve (robust to small negative values if present)
    cog_at_u = np.maximum.accumulate(cog_at_u)
    # Interpolate to requested radii
    ref_vals = np.interp(
        ref_radii.to_value(ref_radii.unit),
        u_r,
        cog_at_u,
        left=0.0,
        right=cog_at_u[-1] if cog_at_u.size > 0 else 0.0,
    )
    return ref_vals * data_unit

#TODO: create a method for extracting stellar spectra outside FluxCalibration
# extract stellar spectra


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

    def __init__(self, response=None, response_err=None, response_wavelength=None,
                 response_file=None, **correction_kwargs):
        super().__init__(**correction_kwargs)
        
        self.vprint("Initialising Flux Calibration (Spectral Throughput)")

        self.response_wavelength = check_unit(response_wavelength, u.angstrom)
        self.response = check_unit(response)
        if response_err is not None:
            self.response_err = check_unit(response_err, self.response.unit)
        else:
            self.response_err = np.zeros_like(self.response)

        self.response_file = response_file

    @classmethod
    def from_text_file(cls, path):
        """Load the resonse function from a text file.
        
        Parameters
        ----------
        path : str
            Path to the text file containing the response as function of
            wavelength. The text file is assumed to contain only two columns:
            the first one corresponding the the wavelength array, and the second
            the array of values associate to the response function.
        
        Returns
        -------
        flux_calibration : :class:`FluxCalibration`
            An instance of ``FluxCalibration``.
        """
        data = np.loadtxt(path, dtype=float, comments="#")
        if data.ndim != 2 or data.shape[1] not in (2, 3):
            raise ArithmeticError(f"Unrecognized shape: {data.shape}")

        wavelength = data[:, 0]
        response = data[:, 1]
        response_err = data[:, 2] if data.shape[1] == 3 else np.zeros_like(resp_vals)

        # Default units
        wave_unit = u.AA
        resp_unit = u.dimensionless_unscaled
        with open(path, "r") as f:
            _ = f.readline()
            line = f.readline()
            if line[0] == "#":
                wave_header, response_header, response_err_header = line.split(",")
                wave_idx = wave_header.find("("), wave_header.rfind(")")
                response_idx = response_header.find("("), response_header.rfind(")")
                wave_unit = u.Unit(wave_header[wave_idx[0] + 1 : wave_idx[1]])
                resp_unit = u.Unit(
                    response_header[response_idx[0] + 1 : response_idx[1]])
        return cls(response=response << resp_unit,
                   response_err=response_err << resp_unit,
                   response_wavelength=wavelength << wave_unit,
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
            extract_args = dict(wave_range=None, wave_window=None, plot=False)
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
            vprint("-" * 40 + "\nAutomatic calibration process for {}\n"
                       .format(calib_stars[i]) + "-" * 40 + '\n')
            # Extract flux from std star
            vprint("Extracting stellar flux from data")
            result = cls.extract_stellar_flux(data[i].copy(), **extract_args)
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
                data[i].wavelength, result['stellar_flux'], ref_wave, ref_spectra,
                **response_params)

            flux_cal_results[name]['wavelength'] = data[i].wavelength.copy()
            flux_cal_results[name]['response'] = resp_curve(
                data[i].wavelength.copy())
            flux_cal_results[name]['response_fig'] = resp_fig
            flux_corrections.append(
                cls(response=resp_curve(data[i].wavelength),
                    response_wavelength=data[i].wavelength))

        if len(flux_corrections) == 0:
            vprint("No flux calibration was created", level="warning")
            return None, None, None

        if combine:
            master_flux_corr = cls.master_flux_auto(flux_corrections)
        else:
            master_flux_corr = None
        return flux_cal_results, flux_corrections, master_flux_corr

    @classmethod
    def master_flux_auto(cls, flux_calibration_corrections: list,
                         combine_method="median"):
        """
        Combine several :class:`FluxCalibration` into a master calibration.

        Parameters
        ----------
        flux_calibration_corrections : list
            List containing the :class:`FluxCalibration` to combine.

        Returns
        -------
        master_resp : :class:`FluxCalibration`
            The master response function.
        """
        vprint("Mastering response function")
        if len(flux_calibration_corrections) == 1:
            return flux_calibration_corrections[0]

        # Select a reference wavelength array
        ref_wave = flux_calibration_corrections[0].response_wavelength
        ref_resp_unit = flux_calibration_corrections[0].response.unit
        # Place holder
        spectral_response = np.full((len(flux_calibration_corrections),
                                         ref_wave.size), fill_value=np.nan,
                                         dtype=float)
        spectral_response_err = np.full_like(spectral_response,
                                        fill_value=np.nan)

        spectral_response[0] = flux_calibration_corrections[0].response.to_value(ref_resp_unit)
        spectral_response_err[1] = flux_calibration_corrections[0].response_err.to_value(ref_resp_unit)

        for ith, fcal_corr in enumerate(flux_calibration_corrections[1:]):
            resp, resp_err = fcal_corr.interpolate_response(ref_wave)
            spectral_response[ith + 1] = resp.to_value(ref_resp_unit)
            spectral_response_err[ith + 1] = resp_err.to_value(ref_resp_unit)

        if combine_method == "median":
            master_resp = np.nanmedian(spectral_response, axis=0)
            mad = 1.4826 * np.nanmedian(
                np.abs(spectral_response - master_resp), axis=0)
            n_eff = np.sum(np.isfinite(spectral_response), axis=0).clip(min=1)
            master_resp_err = 1.2533 * mad / np.sqrt(n_eff)
        elif combine_method == "mean":
            master_resp = np.nanmean(spectral_response, axis=0)
            n_eff = np.sum(np.isfinite(spectral_response), axis=0).clip(min=1)
            master_resp_err = np.sqrt(
                np.nansum(spectral_response_err**2, axis=0) / n_eff)
        elif combine_method == "wmean":
            w = 1.0 / np.where(
                np.isfinite(spectral_response_err) & (spectral_response_err > 0),
                spectral_response_err**2, np.nan)
            num = np.nansum(w * spectral_response, axis=0)
            den = np.nansum(w, axis=0)
            master_resp = num / den
            master_resp_err = 1.0 / np.sqrt(den)
        else:
            raise ValueError(f"Unknown combine_method: {combine_method}")

        return FluxCalibration(response=master_resp << ref_resp_unit,
                               response_err=master_resp_err << ref_resp_unit,
                               response_wavelength=ref_wave)

    @staticmethod
    def extract_stellar_flux(data_container,
                             wave_range=None, wave_window=None,
                             profile=cumulative_1d_moffat,
                             bounds: tuple=None,
                             growth_r : u.Quantity=None,
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
            Profile function to model the cumulative light profile. Any function
            that accepts as first argument
            the square distance (r^2) and returns the cumulative profile can be
            used. Default is cumulative_1d_moffat.
        bounds : tuple, optional
            Bounds for the curve fit. If ``None``, the bounds will be estimated
            automatically.
        growth_r : :class:`astropy.units.Quantity`, optional
            Reference radial bins relative to the center of the star that will
            be used to compute the curve of growth. Default ranges from 0.5 to 
            10 arcsec in steps of 0.5 arcsec.
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
        wave_mask = np.ones(wavelength.size, dtype=bool)
        if wave_range is not None:
            wave_mask[
                (wavelength < check_unit(wave_range[0], wavelength.unit)
                ) | (wavelength > check_unit(wave_range[1], wavelength.unit))
                ] = False
        if wave_window is None:
            wave_window = 1
        if growth_r is None:
            growth_r = np.arange(0.5, 10, 0.5) << u.arcsec
        if bounds is None:
            bounds = "auto"

        wavelength = wavelength[wave_mask]

        # Curve of growth radial bins
        r_dummy = check_unit(growth_r, u.arcsec)

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
            x = data_container.info['fib_ra'].copy()
            y = data_container.info['fib_dec'].copy()
        elif isinstance(data_container, Cube):
            vprint("Extracting flux from input Cube")
            data = data_container.intensity.copy()
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
            variance = data_container.variance.copy()
            variance = variance.reshape((variance.shape[0], variance.shape[1]
                                         * variance.shape[2]))
            x, y = np.indices((data_container.n_rows, data_container.n_cols))
            x, y = x.flatten(), y.flatten()
            skycoords = data_container.wcs.celestial.array_index_to_world(x, y)
            x, y = skycoords.ra, skycoords.dec
        else:
            raise NameError(
                f"Unrecongnized datacontainer of type {type(data_container)}")
        # Convert x and y to relative offset position wrt the mean centre
        xy_coords = SkyCoord(ra=x, dec=y, frame="icrs")

        data = data[wave_mask, :]
        variance = variance[wave_mask, :]

        # Declare variables
        mean_residuals = np.full(wavelength.size, fill_value=np.nan) << data.unit
        star_flux = np.full_like(mean_residuals, fill_value=np.nan)
        profile_popt = []
        profile_var = []
        running_wavelength = []
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
            wave_edges = [wavelength[wave_slice][0].to_value("AA"),
                          wavelength[wave_slice][-1].to_value("AA")]
            slice_data = data[wave_slice]
            slice_var = variance[wave_slice]
            # Compute the median value only considering good values (i.e. excluding NaN)
            slice_data = np.nanmedian(slice_data, axis=0)
            slice_var = np.nanmedian(slice_var, axis=0) / np.sqrt(slice_var.shape[0])
            slice_var[slice_var <= 0] = np.inf << slice_var.unit

            mask = np.isfinite(slice_data) & np.isfinite(slice_var) & (slice_data > 0)
            if not mask.any():
                vprint("Chunk between {} to {} AA contains no useful data"
                        .format(wave_edges[0], wave_edges[1]))
                continue
            vprint(f"Number of valid pixels: {np.count_nonzero(mask)}")
            ###################################################################
            # Computing the curve of growth
            ###################################################################
            x0, y0 = centre_of_mass(slice_data[mask], x[mask], y[mask])
            vprint(f"COM: {x0}, {y0}")
            # Make the growth curve
            distance = xy_coords.separation(
                SkyCoord(ra=x0, dec=y0, frame="icrs"))

            growth_c = curve_of_growth(distance, slice_data, r_dummy, mask)
            growth_c_var = curve_of_growth(distance, slice_var, r_dummy, mask)

            cog_mask = np.isfinite(growth_c) & (growth_c > 0)
            if not cog_mask.any():
                vprint("Chunk between {} to {} AA contains no useful data"
                        .format(wave_edges[0], wave_edges[1]))
                continue
            # Profile fit
            try:
                if bounds == 'auto':
                    # vprint("Automatic fit bounds")
                    p_bounds = ([0, 0, 0],
                                [growth_c[cog_mask][-1].value * 2,
                                 r_dummy.to_value("arcsec").max(), 4])
                else:
                    p_bounds = bounds
                # Initial guess
                p0=[growth_c[cog_mask][-1].value,
                    r_dummy.to_value("arcsec").mean(), 1.0]
                popt, pcov = curve_fit(
                    profile, r_dummy[cog_mask].to_value("arcsec"),
                    growth_c[cog_mask].value, bounds=p_bounds,
                    p0=p0, **fitter_args)
                model_growth_c = profile(
                    r_dummy[cog_mask].to_value("arcsec"), *popt) << growth_c.unit
            except Exception as e:
                vprint("There was a problem during the fit:\n", e)
            else:
                cog.append(growth_c)
                cog_var.append(growth_c_var)
                profile_popt.append(popt)
                profile_var.append(pcov.diagonal())
                running_wavelength.append(wave_edges)

                res = growth_c[cog_mask] - model_growth_c
                mean_residuals[wave_slice] = np.nanmean(res)
                star_flux[wave_slice] = popt[0] << growth_c.unit

        star_good = np.isfinite(star_flux)
        star_flux = np.interp(
            data_container.wavelength, wavelength[star_good],
            star_flux[star_good])
        final_residuals = np.full(data_container.wavelength.size,
                                  fill_value=np.nan) << data.unit
        final_residuals[wave_mask] = mean_residuals

        if plot:
            fig = FluxCalibration.plot_extraction(
                x.value, y.value, x0.value, y0.value,
                data.value, r_dummy.value**0.5, cog,
                data_container.wavelength, star_flux, final_residuals)
        else:
            fig = None

        result = dict(wave_edges=np.array(running_wavelength) << u.AA,
                      optimal=np.array(profile_popt),
                      variance=np.array(profile_var),
                      stellar_flux=star_flux,
                      residuals=final_residuals,
                      figure=fig)
        return result

    @staticmethod
    def plot_extraction(x, y, x0, y0, data, rad, cog, wavelength, star_flux,
                        residuals):
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
        ax.plot(wavelength, star_flux, '-', lw=0.7, label=f'Extracted stellar flux ')
        ax.plot(wavelength, residuals, '-', lw=0.7, label=f'Mean model residuals')
        ax.legend()
        ax.set_ylabel(f'Flux ({star_flux.unit})')
        ax.set_xlabel('Wavelength')

        axs[1, 1].axis("off")
        
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
        if path is None and name is None:
            raise ValueError("Provide either a star `name` or a file `path`.")

        if path is None:
            nm = name.strip().lower()
            if nm[0] != 'f' or 'feige' in nm:
                nm = 'f' + nm
            all_names, files = FluxCalibration.list_available_stars(verbose=False)
            m = np.where(all_names == nm)[0]
            if len(m) == 0:
                raise FileNotFoundError(f"Calibration star not found: {name}")
            pick = files[m[-1]]
            path = os.path.join(os.path.dirname(__file__), '..', 'input_data',
                            'spectrophotometric_stars', pick)

        calib_wave, calib_spectra = np.loadtxt(path, unpack=True, usecols=(0, 1))
        return calib_wave << u.AA, calib_spectra << 1e-16 * u.erg / u.s / u.AA / u.cm**2

    @staticmethod
    def get_response_curve(obs_wave, obs_spectra, ref_wave, ref_spectra,
                           pol_deg=None, spline=False, spline_args={},
                           median_filter_n=None, plot=False,
                           mask_absorption=True, mask_tellurics=True, mask_zeros=True):
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
        if spline_args is None:
            spline_args = {}
        spline_k = int(spline_args.get("k", 3))
        spline_s = spline_args.get("s", None)

        int_ref_spectra = flux_conserving_interpolation(obs_wave, ref_wave,
                                                        ref_spectra)
        raw_response = obs_spectra / int_ref_spectra

        # Build weights
        weights = np.ones(raw_response.size, dtype=float)
        if mask_absorption:
            vprint("Masking stellar absoprtion lines")
            weights *= np.asarray(mask_lines(obs_wave), dtype=float)
        if mask_tellurics:
            vprint("Masking telluric lines")
            weights *= np.asarray(mask_telluric_lines(obs_wave), dtype=float)
        if mask_zeros:
            vprint("Masking zeros")
            weights *= (obs_spectra.to_value(obs_spectra.unit) > 0).astype(float)

        # Optional median filtering and robust reweighting
        if median_filter_n is not None and median_filter_n >= 2:
            filtered_raw_response = median_filter(
                raw_response, size=median_filter_n) << raw_response.unit
            weights *= 1.0 / (
                1.0 + np.abs(raw_response - filtered_raw_response).value)**2
            raw_response = filtered_raw_response
        
        weights = np.clip(weights, 0.0, None)
        # Interpolation
        if pol_deg is not None:
            p_fit = np.polyfit(obs_wave.to_value("AA"),
                               raw_response.value, deg=pol_deg, w=weights)
            response = np.poly1d(p_fit)
            fit_label = f"{pol_deg}-deg polynomial"
        elif spline:
            response = make_smoothing_spline(
                obs_wave.to_value("AA")[weights > 0],
                raw_response.value[weights > 0],
                w=weights[weights > 0])
            fit_label = f"spline k={spline_k} s={spline_s}"
        else:
            # Linear interpolation
            response = interp1d(obs_wave.to_value("AA")[weights > 0],
                                raw_response.value[weights > 0],
                                fill_value="extrapolate", bounds_error=False)
            fit_label = "linear interp (fallback)"

        def response_wrapper(x):
            if isinstance(x, u.Quantity):
                return response(x.to_value("AA")) << raw_response.unit
            else:
                return response(x) << raw_response.unit

        final_response = check_unit(response(obs_wave.to_value("AA")),
                                    raw_response.unit)
        if plot:
            fig = FluxCalibration.plot_response(
            obs_wave=obs_wave,
            obs_spectra=obs_spectra,
            ref_interp=int_ref_spectra,
            raw_response=raw_response,
            filtered_response=(use_vals * resp_unit),
            final_response=final_response,
            weights=weights,
            fit_label=fit_label,
            median_filter_n=mf_n,
            )
            return response_wrapper, fig
        else:
            return response_wrapper, None

    def plot_response(
        *,
        obs_wave,
        obs_spectra,
        ref_interp,
        raw_response,
        filtered_response,
        final_response,
        weights,
        fit_label,
        median_filter_n: None,
    ):
        """
        Generate a three-panel QA plot for the response fit.

        Top panel: raw obs/ref, filtered obs/ref (if any), and fitted response.
        Middle panel: reference spectrum (interpolated) and calibrated observation using
                    filtered response and final fitted response. Also overlays weights.
        Bottom panel: ratio of calibrated observation to reference, should be near 1.

        Parameters
        ----------
        obs_wave : Quantity, 1D
            Observed wavelength grid.
        obs_spectra : Quantity, 1D
            Observed flux density on obs_wave.
        ref_interp : Quantity, 1D
            Reference flux density interpolated onto obs_wave.
        raw_response : Quantity, 1D
            Raw response Obs / Ref.
        filtered_response : Quantity, 1D
            Median-filtered response if applied, otherwise equal to raw response values.
        final_response : Quantity, 1D
            Final fitted response evaluated on obs_wave.
        weights : ndarray, 1D
            Relative weights used during fitting, nonnegative.
        fit_label : str
            Description of the fit method for annotation.
        median_filter_n : int or None
            Median filter window size if applied.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The assembled QA figure.
        """
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 8),
                                constrained_layout=True, sharex=True,
                                height_ratios=[3, 3, 1])

        # Panel 1: responses
        ax = axs[0]
        ax.set_title("Instrumental response fit")
        ax.plot(obs_wave, raw_response, lw=0.7, label="Obs/Ref (raw)")
        if median_filter_n is not None and median_filter_n >= 2:
            ax.plot(obs_wave, filtered_response, lw=0.8, label=f"Obs/Ref (median {median_filter_n})")
        ax.plot(obs_wave, final_response, lw=1.2, label=f"Fit: {fit_label}")
        ax.set_ylabel("R(lambda)")
        ymin = np.nanmin(final_response.to_value(final_response.unit))
        ymax = np.nanmax(final_response.to_value(final_response.unit))
        if np.isfinite(ymin) and np.isfinite(ymax):
            ax.set_ylim(ymin * 0.8, ymax * 1.2)
        ax.legend(loc="best")

        # Panel 2: calibrated spectra and weights
        ax = axs[1]
        ax.plot(obs_wave, ref_interp, lw=0.7, label="Reference (interp)")
        ax.plot(obs_wave, obs_spectra / filtered_response, lw=0.7, label="Calibrated with median")
        ax.plot(obs_wave, obs_spectra / final_response, lw=1.0, label="Calibrated with fit")
        ax.set_ylabel("Flux")
        ax.legend(loc="best")
        tw = ax.twinx()
        tw.plot(obs_wave, weights, lw=0.7, alpha=0.6, color="fuchsia", label="weights")
        tw.set_ylabel("Relative weights")

        # Panel 3: ratio vs 1
        ax = axs[2]
        ratio = (obs_spectra / final_response) / ref_interp
        ax.axhline(1.00, ls="--", color="r", alpha=0.6)
        ax.axhline(1.10, ls=":", color="r", alpha=0.4)
        ax.axhline(0.90, ls=":", color="r", alpha=0.4)
        ax.plot(obs_wave, ratio.to_value(ratio.unit), lw=0.8, color="k")
        ax.set_ylim(0.7, 1.3)
        ax.set_ylabel("Obs_cal / Ref")
        ax.set_xlabel("Wavelength")
        plt.close(fig)
        return fig

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
        response_err = np.interp(wavelength, self.response_wavelength,
                                 self.response_err,
                                 right=np.nan, left=np.nan)
        if update:
            self.vprint("Updating response and wavelength arrays")
            self.response = response
            self.response_err = response_err
            self.response_wavelength = wavelength
        return response, response_err

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
        # TODO: Use a QTable and dump to FITS instead.
        np.savetxt(fname, np.array([self.response_wavelength, self.response,
                                    self.response_err]).T,
                   header="Spectral Response curve\n"
                    + f" wavelength ({self.response_wavelength.unit}),"
                    + f" R ({self.response.unit}), Rerr ({self.response_err.unit})" 
                   )

    def apply(self, spectra_container : SpectraContainer) -> SpectraContainer:
        """
        Applies the response curve to a given SpectraContainer.

        Parameters
        ----------
        spectra_container : :class:`SpectraContainer`
            Target object to be corrected.

        Returns
        -------
        spectra_container_out : :class:`SpectraContainer`
            Corrected version of the input :class:`SpectraContainer`.
        """
        assert isinstance(spectra_container, SpectraContainer)
        sc_out = spectra_container.copy()
        if sc_out.is_corrected(self.name):
            self.vprint("Data already calibrated")
            return sc_out

        # Check that the model is sampled in the same wavelength grid
        if not sc_out.wavelength.size == self.response_wavelength.size or not np.allclose(
            sc_out.wavelength, self.response_wavelength, equal_nan=True):
            response, response_err = self.interpolate_response(
                sc_out.wavelength, update=False)
        else:
            response, response_err = self.response, self.response_err

        response = np.where(np.isfinite(response) & (response > 0),
                            response, np.nan)
        # Apply the correction
        good = np.isfinite(response) & (response > 0)
        sc_out.rss_intensity = (
            sc_out.rss_intensity / response[np.newaxis, :])
        # Propagate the uncertainty of the flux calibration
        sc_out.rss_variance = sc_out.rss_variance / response[np.newaxis, :]**2
        sc_out.rss_variance += (
            spectra_container.rss_intensity**2
            * response_err[np.newaxis, :]**2
            / response[np.newaxis, :]**4)

        self.record_correction(sc_out, status='applied',
                               units=str(self.response.unit))
        return sc_out

# Mr Krtxo \(ﾟ▽ﾟ)/
