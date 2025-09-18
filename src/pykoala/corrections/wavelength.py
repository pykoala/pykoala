"""
Module for estimating and applying wavelength offset corrections related to
inaccuracies in the original wavelength calibration.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from astropy.io import fits
from astropy import units as u
from scipy.ndimage import median_filter, gaussian_filter, percentile_filter
from scipy.interpolate import interp1d
from scipy.signal import correlate, correlation_lags

from pykoala import vprint
from pykoala.corrections.correction import CorrectionBase
from pykoala.data_container import RSS, SpectraContainer
from pykoala.ancillary import flux_conserving_interpolation, vac_to_air, check_unit
from pykoala import ancillary


class WavelengthOffset(object):
    """Wavelength offset class.

    This class stores a 2D wavelength offset.

    Attributes
    ----------
    offset_data : :class:`astropy.units.Quantity`
        Wavelength offset expressed in pixels or wavelengths.
    offset_error : :class:`astropy.units.Quantity`
        Standard deviation of ``offset_data``.
    path: str
        Filename path.

    """

    offset_data = None
    offset_error = None

    def __init__(self, path=None, offset_data=None, offset_error=None):
        self.path = path
        # The input units can be either pixel or wavelength
        self.offset_data = check_unit(offset_data)
        self.offset_error = check_unit(offset_error)

    def to_fits(self, output_path=None):
        """Save the offset in a FITS file.
        
        Parameters
        ----------
        output_path: str, optional, default=None
            FITS file name path. If None, and ``self.path`` exists,
            the original file is overwritten.

        Notes
        -----
        The output fits file contains an empty PrimaryHDU, and two ImageHDU
        ("OFFSET", "OFFSET_ERR") containing the offset data and associated error.
        """
        if output_path is None:
            if self.path is None:
                raise NameError("Provide output path")
            else:
                output_path = self.path

        primary = fits.PrimaryHDU()
        # OFFSET
        hdr_data = fits.Header()
        if self.offset_data is None:
            raise ValueError("offset_data is None")
        hdr_data["BUNIT"] = self.offset_data.unit.to_string()
        hdu_data = fits.ImageHDU(data=self.offset_data.value, name='OFFSET', header=hdr_data)

        if self.offset_error is None:
            # create an array of NaN with same shape and unit as data
            err_values = np.full_like(self.offset_data.value, np.nan, dtype=float)
            err_unit = self.offset_data.unit
        else:
            err_values = self.offset_error.value
            err_unit = self.offset_error.unit
        hdr_err = fits.Header()
        hdr_err["BUNIT"] = err_unit.to_string()
        hdu_err = fits.ImageHDU(data=err_values, name="OFFSET_ERR", header=hdr_err)

        hdul = fits.HDUList([primary, hdu_data, hdu_err])
        hdul.writeto(output_path, overwrite=True)
        hdul.close()
        vprint(f"Wavelength offset saved at {output_path}")

    @classmethod
    def from_fits(cls, path):
        """Load the offset data from a fits file.

        Loads offset values (extension 1) and
        associated errors (extension 2) from a fits file.

        Parameters
        ----------
        path : str
            Path to the FITS file containing the offset data.

        Returns
        -------
        wavelength_offset : :class:`WavelengthOffset`
            A :class:`WavelengthOffset` initialised with the input data.
        """
        if not os.path.isfile(path):
            raise NameError(f"offset file {path} does not exist.")
        vprint(f"Loading wavelength offset from {path}")
        with fits.open(path) as hdul:
            offset_data = hdul[1].data << u.Unit(hdul[1].header.get("BUNIT", 1))
            offset_error = hdul[2].data << u.Unit(hdul[2].header.get("BUNIT", 1))
        return cls(offset_data=offset_data, offset_error=offset_error,
                   path=path)


class WavelengthCorrection(CorrectionBase):
    """Wavelength correction class.

    This class accounts for the relative wavelength offset between fibres.

    Attributes
    ----------
    name : str
        Correction name, to be recorded in the log.
    offset : :class:`WavelengthOffset`
        2D wavelength offset (n_fibres x n_wavelengths)
    """

    name = "WavelengthCorrection"
    offset = None
    verbose = False

    def __init__(self, offset_path: str=None, offset: WavelengthOffset=None,
                 **correction_kwargs):
        super().__init__(**correction_kwargs)
        self.path = offset_path
        self.offset = offset

    @classmethod
    def from_fits(cls, path: str):
        """Initialise a WavelengthOffset correction from an input FITS file.
        
        Parameters
        ----------
        path : str
            Path to the FITS file containing the offset data.

        Returns
        -------
        wave_correction : :class:`WavelengthCorrection`
            A :class:`WavelengthCorrection` initialised with the input data.
        """
        return cls(offset=WavelengthOffset.from_fits(path=path),
                   offset_path=path)

    def apply(self, rss : RSS) -> RSS:
        """Apply a 2D wavelength offset model to a RSS.

        Parameters
        ----------
        rss : :class:`pykoala.rss.RSS`
            Original Row-Stacked-Spectra object to be corrected.

        Returns
        -------
        rss_corrected : :class:`pykoala.rss.RSS`
            Corrected copy of the input RSS.
        """

        assert isinstance(rss, RSS)

        if self.offset is None or self.offset.offset_data is None:
            raise ValueError("No offset loaded")
        
        rss_out = rss.copy()
        self.vprint("Applying correction to input RSS")

        if self.offset.offset_data.unit == u.pixel:
            x = np.arange(rss.wavelength.size) << u.pixel
        elif self.offset.offset_data.unit.is_equivalent(u.AA):
            x = rss.wavelength.to(self.offset.offset_data.unit)
        else:
            raise ValueError("Offset units must be pixel or wavelength")

        # per-fibre scalar or vector offsets
        off = self.offset.offset_data
        if off.ndim == 1:
            if off.size != rss.intensity.shape[0]:
                raise ValueError("offset_data shape is invalid for RSS")
            for i in range(rss.intensity.shape[0]):
                rss_out.intensity[i] = flux_conserving_interpolation(
                    x, x - off[i], rss.intensity[i]
                )
                if hasattr(rss, "variance") and rss.variance is not None:
                    rss_out.variance[i] = flux_conserving_interpolation(
                        x, x - off[i], rss.variance[i]
                    )
        elif off.ndim == 2:
            if off.shape != rss.intensity.shape:
                raise ValueError("2D offset_data must match RSS intensity shape")
            for i in range(rss.intensity.shape[0]):
                rss_out.intensity[i] = flux_conserving_interpolation(
                    x, x - off[i], rss.intensity[i]
                )
                if hasattr(rss, "variance") and rss.variance is not None:
                    rss_out.variance[i] = flux_conserving_interpolation(
                        x, x - off[i], rss.variance[i]
                    )
        else:
            raise ValueError("offset_data must be 1D or 2D")

        comment = f"wave-offset_unit={self.offset.offset_data.unit}; shape={self.offset.offset_data.shape}"
        self.record_correction(rss_out, status="applied", comment=comment)

        return rss_out


class TelluricWavelengthCorrection(WavelengthCorrection):
    """WavelengthCorrection based on the cross-correlation of telluric lines."""

    name = "TelluricWavelengthCorrection"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_rss(cls, rss : RSS,
                 median_smooth=None, pol_fit_deg=None, oversampling=5,
                 wave_range=None, plot=False):
        """Estimate the wavelength offset from an input RSS using telluric absorption.
        
        Parameters
        ----------
        rss : :class:`RSS`
        median_smooth : int, optional
            Median filter size.
        pol_fit_deg : int, optional
            Polynomial degree to fit the resulting offset.
        oversampling : float, optional
            Oversampling factor to increase the accuracy of the cross-correlation.
        wave_range : list or tupla
            Wavelength range to fit the offset.
        plot : bool, optional
            If True, returns quality control plots.

        Returns
        -------
        WavelengthCorrection : :class:`WavelengthCorrection`
        figs : :class:`plt.Figure`
        """
        assert isinstance(rss, RSS), "Input data must be an instance of RSS"

        # Normalize each spectrum
        intensity = rss.intensity.value.copy()
        med = np.nanmedian(intensity, axis=1)
        med[~np.isfinite(med)] = 1.0
        intensity /= med[:, np.newaxis]

        # Optional wavelength range mask
        mask = np.isfinite(rss.wavelength)
        if wave_range is not None:
            lo = check_unit(wave_range[0], rss.wavelength.unit)
            hi = check_unit(wave_range[1], rss.wavelength.unit)
            mask &= (rss.wavelength >= lo) & (rss.wavelength <= hi)

        # Oversample in wavelength index domain
        new_wavelength = np.interp(
            np.arange(0, rss.wavelength.size, 1.0 / oversampling),
            np.arange(rss.wavelength.size),
            rss.wavelength,
        )
        interpolator = interp1d(rss.wavelength, intensity, axis=1)
        intensity = interpolator(new_wavelength)

        # Restrict mask to resampled grid
        res_mask = np.interp(new_wavelength, rss.wavelength, mask.astype(float)) > 0.5
        # Reference median spectrum
        median_intensity = np.nanmedian(intensity, axis=0)

        fibre_offset = np.zeros(intensity.shape[0], dtype=float)
        for ith, fibre in enumerate(intensity):
            fibre_mask = np.isfinite(fibre) & res_mask
            if not fibre_mask.any():
                continue

            corr = correlate(fibre[fibre_mask], median_intensity[fibre_mask],
                             mode="full", method="fft")
            lags = correlation_lags(fibre[fibre_mask].size,
                                    median_intensity[fibre_mask].size)
            max_corr = np.argmax(corr)
            # guard edges for parabolic interpolation
            i0 = max(max_corr - 1, 0)
            i1 = max_corr
            i2 = min(max_corr + 1, corr.size - 1)
            x3 = lags[[i0, i1, i2]]
            y3 = corr[[i0, i1, i2]]
            # if duplicates happen at edges, skip parabolic and take argmax
            if (i0 == i1) or (i1 == i2) or (i0 == i2):
                peak = lags[max_corr]
            else:
                peak = ancillary.parabolic_maximum(x3, y3)
            fibre_offset[ith] = peak / oversampling

        if median_smooth is not None:
            fibre_offset = median_filter(fibre_offset, size=median_smooth)

        if pol_fit_deg is not None:
            x = np.arange(fibre_offset.size, dtype=float)
            pol = np.polyfit(x, fibre_offset, deg=pol_fit_deg)
            fibre_offset = np.poly1d(pol)(x)

        figs = None
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(fibre_offset)
            ax.set_ylim(np.nanmin(fibre_offset), np.nanmax(fibre_offset))
            ax.set_xlabel("Fibre number")
            ax.set_ylabel("Average wavelength offset (pixel)")
            fibre_map_fig = rss.plot_fibres(data=fibre_offset)
            figs = (fig, fibre_map_fig)
            plt.close(fig)

        fibre_offset = fibre_offset << u.pixel
        offset = WavelengthOffset(
            offset_data=fibre_offset, offset_error=np.full_like(fibre_offset, fill_value=np.nan)
        )
        return cls(offset=offset), figs


class SolarCrossCorrOffset(WavelengthCorrection):
    """WavelengthCorrection based on solar spectra cross-correlation.
    
    Constructs a WavelengthOffset using a cross-correlation between a solar
    reference spectrum and a twilight exposure (dominated by solar features).

    Also implements LSF(lambda) estimation from the solar spectrum.
    """
    name = "SolarCrossCorrelationOffset"

    def __init__(self, sun_wavelength, sun_intensity, **kwargs):
        super().__init__(offset=WavelengthOffset(), **kwargs)
        self.sun_wavelength = check_unit(sun_wavelength, u.AA)
        self.sun_intensity = check_unit(sun_intensity,
                                        u.erg / u.s / u.AA / u.cm**2)
        self._lsf_knots_wave = None
        self._lsf_knots_sigma_pix = None
        self._lsf_poly_coeff = None
        self._lsf_poly_deg = None

    @classmethod
    def from_fits(cls, path=None, extension=1):
        """Initialise a WavelengthOffset correction from an input FITS file.
        
        Parameters
        ----------
        path : str, optional
            Path to the FITS file containing the reference Sun's spectra. The
            file must contain an extension with a table including a ``WAVELENGTH``
            and ``FLUX`` columns.The wavelength array must be angstrom in the
            vacuum frame.
        extension : int or str, optional
            HDU extension containing the table. Default is 1.

        Returns
        -------
        solar_offset_correction : :class:`SolarCrossCorrOffset`
            An instance of SolarCrossCorrOffset.
        """
        if path is None:
            path = os.path.join(os.path.dirname(__file__), '..',
                     'input_data', 'spectrophotometric_stars',
                     'sun_mod_001.fits')
        with fits.open(path) as hdul:
            sun_wavelength = hdul[extension].data['WAVELENGTH'] << u.AA
            sun_wavelength = vac_to_air(sun_wavelength)
            sun_intensity = hdul[extension].data['FLUX'] << u.erg / u.s / u.AA / u.cm**2
        return cls(sun_wavelength=sun_wavelength,
                   sun_intensity=sun_intensity)

    @classmethod
    def from_text_file(cls, path, loadtxt_args={}):
        """Initialise a :class:`SolarCrossCorrOffset` correction from an input text file.
        
        Parameters
        ----------
        path: str
            Path to the text file containing the reference Sun's spectra. The
            text file must contain two columns consisting of the
            vacuum wavelength array in angstrom and the solar flux or luminosity.
        loadtxt_args: dict, optional
            Additional arguments to be passed to ``numpy.loadtxt``.

        Returns
        -------
        solar_offset_correction: :class:`SolarCrossCorrOffset`
            An instance of SolarCrossCorrOffset.
        """
        sun_wavelength, sun_intensity = np.loadtxt(path, unpack=True,
                                                   usecols=(0, 1),
                                                   **loadtxt_args)
        #TODO: Handle units
        sun_wavelength = vac_to_air(sun_wavelength)
        return cls(sun_wavelength=sun_wavelength,
                   sun_intensity=sun_intensity)


    def get_solar_features(self, solar_wavelength, solar_spectra,
                            window_size_aa=20):
        """
        Estimate the regions of the solar spectrum dominated by absorption features.

        Description
        -----------
        First, a median filter is applied to estimate the upper envelope of the
        continuum. Then, the median ratio between the solar spectra and the median-filtered
        estimate is used to compute the relative weights:

        .. math::
            \\begin{equation}
                w = \\left\\|\\frac{F_\\odot}{F_{cont}} - Median(\\frac{F_\\odot}{F_{cont}})\\right\\|
            \\end{equation}

        Parameters
        ----------
        solar_wavelength: numpy.ndarray
            Solar spectra wavelengths array.
        solar_spectra: numpy.ndarray
            Array containing the flux of the solar spectra associated to a given
            wavelength.
        window_size_aa: int, optional
            Size of a spectral window in angstrom to perform a median filtering
            and estimate the underlying continuum. Default is 20 AA.

        Returns
        -------
        weights: numpy.ndarray
            Array of weights representing the prominance of an absorption feature.

        """
        self.vprint("Estimating regions of solar spectra dominated by absorption lines.")
        # convert window size to pixels over the current wavelength span
        delta_pixel = int(
            (
                check_unit(window_size_aa, u.AA)
                / (solar_wavelength[-1] - solar_wavelength[0])
                * solar_wavelength.size
            ).decompose()
        )
        if delta_pixel % 2 == 0:
            delta_pixel += 1

        solar_continuum = median_filter(solar_spectra, size=delta_pixel) << solar_spectra.unit
        ratio = solar_spectra / (solar_continuum + ancillary.EPSILON_FLOAT64 * solar_continuum.unit)
        # Detect absorption features
        median_continuum_ratio = np.nanmedian(ratio)
        weights = np.abs(ratio -  median_continuum_ratio)
        s = np.nansum(weights)
        if s > 0:
            weights = weights / s
        return weights

    def compute_grid_of_models(self, pix_shift_array, pix_std_array, pix_array,
                              sun_intensity, weights):
        """Compute a grid of Solar spectra models convolved with a gaussian LSF.
        
        Parameters
        ----------
        pix_shift_array: 1D-np.array
            Array containing the wavelength offsets expressed in pixels.
        pix_std_array: 1D-np.array
            Array containing the values of the gaussian LSF standard deviation
            in pixels.
        pix_array: 1D-np.array
            Array of pixels to sample the grid of models.
        sun_intensity: 1D-np.array
            Array of solar fluxes associated to ``pix_array``.
        weights: 1D-np.array
            Array of absorption-features weights associated to ``sun_intensity``.
        
        Returns
        -------
        models_grid: numpy.ndarray
            Grid of models with dimensions `(n, m, s)`, where `n`, `m` and `s`
            correspond to the size of `pix_shift_array`, `pix_std_array`, and
            `pix_array`, respectively.
        weights_grid: numpy.ndarray
            Grid of absorption-feature weights associated to `models_grid`.

        See also
        --------
        :For more details on the computation of the weights array see :func:`get_solar_features`.

        """
        models_grid = np.zeros(
            (pix_shift_array.size, pix_std_array.size, sun_intensity.size)
            ) << sun_intensity.unit
        weights_grid = np.zeros(
            (pix_shift_array.size, pix_std_array.size, sun_intensity.size)
            )
        shift_idx, std_idx = np.indices(models_grid.shape[:-1])

        for z, (velshift, gauss_std) in enumerate(
            zip(pix_shift_array[shift_idx.flatten()], pix_std_array[std_idx.flatten()])):

                i, j = np.unravel_index(z, models_grid.shape[:-1])

                new_pixel_array = pix_array + velshift
                interp_sun_intensity = flux_conserving_interpolation(
                    new_pixel_array, pix_array, sun_intensity)
                # gaussian_filter expects sigma in pixels
                interp_sun_intensity = gaussian_filter(
                    interp_sun_intensity, gauss_std.value)
                # Restore the intensity units removed by gaussian_filter
                models_grid[i, j] = interp_sun_intensity << sun_intensity.unit
                # propagate weights consistently, truncate to reduce long wings
                interp_sun_weight = flux_conserving_interpolation(
                    new_pixel_array, pix_array, weights)
                interp_sun_weight = gaussian_filter(
                    interp_sun_weight, gauss_std.value, truncate=2.0)
                norm = np.nansum(interp_sun_weight)
                if norm > 0:
                    interp_sun_weight /= norm
                weights_grid[i, j] = interp_sun_weight

        return models_grid, weights_grid

    def _build_overlapping_windows(self, wave, n_windows, window_overlap):
        """
        Create overlapping wavelength windows over the provided 1D wave array.

        Returns
        -------
        centers : list of Quantities (wavelength units)
        slices  : list of slice objects selecting indices per window
        """
        w0 = wave[0]
        w1 = wave[-1]
        # equal-size windows in wavelength space
        centers = np.linspace(w0, w1, n_windows + 2)[1:-1]  # exclude ends
        half_span = (w1 - w0) / (n_windows * 2.0)
        half_span = half_span * (1.0 + window_overlap)

        centers_out = []
        slices = []
        for c in centers:
            lo = c - half_span
            hi = c + half_span
            # indices in wave range
            i0 = int(np.searchsorted(wave, lo))
            i1 = int(np.searchsorted(wave, hi))
            i0 = max(i0, 0)
            i1 = min(i1, wave.size)
            if i1 - i0 < 5:
                continue
            centers_out.append(c)
            slices.append(slice(i0, i1))
        return centers_out, slices

    def estimate_lsf_and_shift_batched(
        self,
        spectra_container: SpectraContainer,
        n_windows=12,
        window_overlap=0.25,
        pix_shift_array=None,
        pix_std_array=None,
        sigma_batch=16,
        shift_batch=32,
        poly_deg_sigma=3,
        poly_deg_shift=2,
        sun_window_size_aa=20,
        keep_features_frac=0.15,
        response_window_size_aa=200,
        mask_tellurics=True,
        use_mean_for_offset=True,
    ):
        """
        Jointly estimate wavelength shift and LSF sigma per fibre, per wavelength window,
        using batched loops to limit memory usage.

        The math and outputs match the unbatched version:
        - Best, mean, and std for shift and sigma in each window and fibre
        - Per-fibre polynomials for sigma(lambda) and shift(lambda)
        - offset.offset_data filled with a per-fibre scalar shift (median of means)

        Batching
        --------
        The grid over (shift, sigma) is scanned in chunks:
        for sigma in batches:
            build broadened template once
            for shift in batches:
            reinterpolate to apply pixel shift
            compute weighted chi2 collapsed by windows

        Weighting
        ---------
        chi2 is computed with weights = line_weights * fibre_weights * 1/variance.

        Parameters are the same as in the unbatched solver, plus:
        - sigma_batch: number of sigma values per batch
        - shift_batch: number of shift values per batch
        """
        # Defaults
        if pix_shift_array is None:
            pix_shift_array = np.arange(-5.0, 5.0, 0.1)
        if pix_std_array is None:
            pix_std_array = np.arange(0.1, 5.0, 0.1)

        pix_shift_array = check_unit(pix_shift_array, u.pixel)
        pix_std_array = check_unit(pix_std_array, u.pixel)
        n_shift_total = pix_shift_array.size
        n_sigma_total = pix_std_array.size

        wave = spectra_container.wavelength
        n_fibre, n_wave = spectra_container.rss_intensity.shape

        # ----- Response normalization -----
        delta_pixel = int(
            (
                check_unit(response_window_size_aa, u.AA)
                / (wave[-1] - wave[0])
                * n_wave
            ).decompose()
        )
        if delta_pixel % 2 == 0:
            delta_pixel += 1

        sun_on_grid = flux_conserving_interpolation(
            wave, self.sun_wavelength, self.sun_intensity
        )

        response = spectra_container.rss_intensity / sun_on_grid[np.newaxis]
        smooth_resp = median_filter(response, size=(1, delta_pixel)) << response.unit
        upper_env = percentile_filter(
            smooth_resp, percentile=95, size=(1, delta_pixel)
        ) << response.unit

        fibre_weights = 1.0 / (
            1.0 + (upper_env / smooth_resp - np.nanmedian(upper_env / smooth_resp)) ** 2
        )

        normalized = spectra_container.rss_intensity / smooth_resp
        if getattr(spectra_container, "rss_variance", None) is not None:
            normalized_var = spectra_container.rss_variance / (smooth_resp ** 2)
        else:
            robust = np.nanmedian(np.abs(normalized), axis=1)
            robust[~np.isfinite(robust)] = (
                np.nanmedian(robust[np.isfinite(robust)]) if np.isfinite(robust).any() else 1.0
            )
            normalized_var = np.repeat((robust[:, np.newaxis] ** 2).value, n_wave, axis=1)
            normalized_var = normalized_var << (normalized.unit ** 2)

        # ----- Solar feature weights over full band -----
        weights_full = self.get_solar_features(
            wave, sun_on_grid, window_size_aa=sun_window_size_aa
        )
        if keep_features_frac is not None:
            thr = np.nanpercentile(weights_full, 100.0 * (1.0 - keep_features_frac))
            weights_full = np.where(weights_full >= thr, weights_full, 0.0)
        # Trim edges for large sigma stability
        max_pix_std = np.max(pix_std_array)
        sigma_wl = (wave[1] - wave[0]) * max_pix_std.value
        weights_full[wave < (wave[0] + 3 * sigma_wl)] = 0.0
        weights_full[wave > (wave[-1] - 3 * sigma_wl)] = 0.0
        s = np.nansum(weights_full)
        if s > 0:
            weights_full /= s
        else:
            raise RuntimeError("No valid solar-feature weights; adjust parameters.")

        if mask_tellurics:
            fibre_weights = fibre_weights * (weights_full[np.newaxis, :] + np.nanmedian(weights_full))

        # ----- Windows -----
        centers, slices = self._build_overlapping_windows(wave, n_windows, window_overlap)
        n_win = len(slices)
        centers_val = np.array([c.to_value(wave.unit) for c in centers], dtype=float)

        # Prepare per-window per-fibre accumulators

        # Pass 1: record chi2 minima for each fibre, window
        chi2_min = np.full((n_fibre, n_win), np.inf, dtype=float)

        # Pass 2 accumulators (likelihood moments)
        # We use chi2_min from pass 1 for stable likelihood
        sumL = np.zeros((n_fibre, n_win), dtype=float)
        sumL_shift = np.zeros((n_fibre, n_win), dtype=float)
        sumL_sigma = np.zeros((n_fibre, n_win), dtype=float)
        sumL_shift2 = np.zeros((n_fibre, n_win), dtype=float)
        sumL_sigma2 = np.zeros((n_fibre, n_win), dtype=float)

        # Also track argmin for "best"
        best_shift = np.zeros((n_fibre, n_win), dtype=float)
        best_sigma = np.zeros((n_fibre, n_win), dtype=float)
        best_chi2 = np.full((n_fibre, n_win), np.inf, dtype=float)

        # Precompute common arrays
        inv_var = 1.0 / (normalized_var.value + ancillary.EPSILON_FLOAT64)  # (n_fibre, n_wave)
        w_fibre = fibre_weights.value  # (n_fibre, n_wave)
        pix_array = (np.arange(n_wave)) << u.pixel

        # Helper to collapse chi2 over a window slice for all fibres at once
        def collapse_window_chi2(diff_sq, w_line, w_fibre, inv_var, sl):
            # diff_sq: (n_fibre, n_wave) float
            # w_line:  (n_wave,) float
            # w_fibre: (n_fibre, n_wave) float
            # inv_var: (n_fibre, n_wave) float
            W = w_line[np.newaxis, :] * w_fibre[:, :] * inv_var[:, :]
            num = np.nansum(diff_sq[:, sl] * W[:, sl], axis=1)  # (n_fibre,)
            den = np.nansum(W[:, sl], axis=1)                   # (n_fibre,)
            den[den <= 0] = 1.0
            return num / den

        # -------------------------
        # PASS 1: find per-window chi2 minima
        # -------------------------
        for sig_start in range(0, n_sigma_total, sigma_batch):
            sig_end = min(sig_start + sigma_batch, n_sigma_total)
            sigma_slice = slice(sig_start, sig_end)
            sigma_vals = pix_std_array[sigma_slice]  # Quantity subarray

            # Build broadened template and line weights for this sigma batch
            # models_sigma_batch: (n_sigma_b, n_wave)
            models_sigma_batch = []
            weights_sigma_batch = []
            for sigma_pix in sigma_vals:
                broadened = gaussian_filter(sun_on_grid, sigma_pix.value) << sun_on_grid.unit
                w_line = gaussian_filter(weights_full, sigma_pix.value, truncate=2.0)
                ss = np.nansum(w_line)
                w_line = w_line / ss if ss > 0 else w_line
                models_sigma_batch.append(broadened)
                weights_sigma_batch.append(w_line)
            models_sigma_batch = np.stack([m.value for m in models_sigma_batch], axis=0)  # float
            weights_sigma_batch = np.stack(weights_sigma_batch, axis=0)                   # float

            for sh_start in range(0, n_shift_total, shift_batch):
                sh_end = min(sh_start + shift_batch, n_shift_total)
                shift_slice = slice(sh_start, sh_end)
                shift_vals = pix_shift_array[shift_slice]  # Quantity subarray

                # Apply pixel shifts to each sigma model in batch
                # For memory, handle shift loop inside sigma loop
                for j in range(models_sigma_batch.shape[0]):  # loop sigma within batch
                    model_full = models_sigma_batch[j]  # (n_wave,)
                    w_line_full = weights_sigma_batch[j]  # (n_wave,)

                    for vel in shift_vals:
                        # shift model and weights in pixel space
                        shifted_model = flux_conserving_interpolation(
                            pix_array, pix_array + vel, model_full << sun_on_grid.unit
                        ).value
                        shifted_wline = flux_conserving_interpolation(
                            pix_array, pix_array + vel, w_line_full << u.dimensionless_unscaled
                        )

                        # Compute diff squared for all fibres at once
                        diff_sq = (normalized.value - shifted_model[np.newaxis, :]) ** 2  # (n_fibre, n_wave)

                        # Collapse into each window and update minima
                        for k, sl in enumerate(slices):
                            chi2_fk = collapse_window_chi2(diff_sq, shifted_wline, w_fibre, inv_var, sl)  # (n_fibre,)
                            better = chi2_fk < chi2_min[:, k]
                            chi2_min[better, k] = chi2_fk[better]

        # -------------------------
        # PASS 2: accumulate likelihood moments and argmin
        # -------------------------
        for sig_start in range(0, n_sigma_total, sigma_batch):
            sig_end = min(sig_start + sigma_batch, n_sigma_total)
            sigma_slice = slice(sig_start, sig_end)
            sigma_vals = pix_std_array[sigma_slice]

            models_sigma_batch = []
            weights_sigma_batch = []
            for sigma_pix in sigma_vals:
                broadened = gaussian_filter(sun_on_grid, sigma_pix.value) << sun_on_grid.unit
                w_line = gaussian_filter(weights_full, sigma_pix.value, truncate=2.0)
                ss = np.nansum(w_line)
                w_line = w_line / ss if ss > 0 else w_line
                models_sigma_batch.append(broadened)
                weights_sigma_batch.append(w_line)
            models_sigma_batch = np.stack([m.value for m in models_sigma_batch], axis=0)
            weights_sigma_batch = np.stack(weights_sigma_batch, axis=0)

            for sh_start in range(0, n_shift_total, shift_batch):
                sh_end = min(sh_start + shift_batch, n_shift_total)
                shift_slice = slice(sh_start, sh_end)
                shift_vals = pix_shift_array[shift_slice]

                for j in range(models_sigma_batch.shape[0]):  # sigma within batch
                    model_full = models_sigma_batch[j]
                    w_line_full = weights_sigma_batch[j]
                    sigma_val = sigma_vals[j].value

                    for vel in shift_vals:
                        shifted_model = flux_conserving_interpolation(
                            pix_array, pix_array + vel, model_full << sun_on_grid.unit
                        ).value
                        shifted_wline = flux_conserving_interpolation(
                            pix_array, pix_array + vel, w_line_full << u.dimensionless_unscaled
                        )
                        diff_sq = (normalized.value - shifted_model[np.newaxis, :]) ** 2

                        # collapse to windows
                        for k, sl in enumerate(slices):
                            chi2_fk = collapse_window_chi2(diff_sq, shifted_wline, w_fibre, inv_var, sl)  # (n_fibre,)

                            # update argmin
                            better = chi2_fk < best_chi2[:, k]
                            if np.any(better):
                                best_chi2[better, k] = chi2_fk[better]
                                best_shift[better, k] = vel.value
                                best_sigma[better, k] = sigma_val

                            # accumulate likelihood moments using chi2_min for numerical stability
                            # L = exp(-0.5 * (chi2 - chi2_min))
                            L = np.exp(-0.5 * (chi2_fk - chi2_min[:, k]))
                            sumL[:, k] += L
                            sumL_shift[:, k] += L * vel.value
                            sumL_sigma[:, k] += L * sigma_val
                            sumL_shift2[:, k] += L * (vel.value ** 2)
                            sumL_sigma2[:, k] += L * (sigma_val ** 2)

        # Compute means and stds
        safe_sumL = np.where(sumL > 0, sumL, 1.0)
        mean_shift = sumL_shift / safe_sumL
        mean_sigma = sumL_sigma / safe_sumL
        var_shift = np.maximum(sumL_shift2 / safe_sumL - mean_shift ** 2, 0.0)
        var_sigma = np.maximum(sumL_sigma2 / safe_sumL - mean_sigma ** 2, 0.0)
        std_shift = np.sqrt(var_shift)
        std_sigma = np.sqrt(var_sigma)

        # Fit per-fibre polynomials over window centers
        self._lsf_per_fibre = {}
        self._shift_per_fibre = {}

        for i in range(n_fibre):
            # sigma poly
            ysig = mean_sigma[i]
            degs = min(poly_deg_sigma, max(0, len(centers_val) - 1))
            if len(centers_val) < 3:
                coeffs_s = np.array([np.nanmedian(ysig) if np.isfinite(ysig).any() else 2.0])
                degs = 0
            else:
                coeffs_s = np.polyfit(centers_val, ysig, deg=degs)
            self._lsf_per_fibre[i] = {
                "poly_deg": degs,
                "poly_coeff": coeffs_s,
                "wave_centers": centers_val.copy(),
                "sigma_pix_best": best_sigma[i].copy(),
                "sigma_pix_mean": mean_sigma[i].copy(),
                "sigma_pix_std": std_sigma[i].copy(),
            }

            # shift poly
            ysh = mean_shift[i]
            degh = min(poly_deg_shift, max(0, len(centers_val) - 1))
            if len(centers_val) < 3:
                coeffs_h = np.array([np.nanmedian(ysh) if np.isfinite(ysh).any() else 0.0])
                degh = 0
            else:
                coeffs_h = np.polyfit(centers_val, ysh, deg=degh)
            self._shift_per_fibre[i] = {
                "poly_deg": degh,
                "poly_coeff": coeffs_h,
                "wave_centers": centers_val.copy(),
                "shift_pix_best": best_shift[i].copy(),
                "shift_pix_mean": mean_shift[i].copy(),
                "shift_pix_std": std_shift[i].copy(),
            }

        # Field average sigma model for compatibility
        field_avg_sigma = np.nanmedian(mean_sigma, axis=0)
        deg_field = min(poly_deg_sigma, max(0, len(centers_val) - 1))
        coeff_field = (
            np.polyfit(centers_val, field_avg_sigma, deg=deg_field)
            if len(centers_val) >= 3
            else np.array([np.nanmedian(field_avg_sigma)])
        )
        self._lsf_knots_wave = centers_val.copy()
        self._lsf_knots_sigma_pix = field_avg_sigma.copy()
        self._lsf_poly_deg = deg_field
        self._lsf_poly_coeff = coeff_field

        # Fill per-fibre scalar offset for downstream apply()
        per_fibre_scalar_shift = np.nanmedian(mean_shift, axis=1)
        if use_mean_for_offset:
            self.offset.offset_data = -per_fibre_scalar_shift << u.pixel
        else:
            self.offset.offset_data = -np.nanmedian(best_shift, axis=1) << u.pixel
        self.offset.offset_error = np.nanmedian(std_shift, axis=1) << u.pixel

        # Cache minimal diagnostics used by existing plotting tools
        self._lsf_diag = {
            "wave": wave,
            "centers": centers,
            "slices": slices,
            "pix_shift_array": pix_shift_array,
            "pix_std_array": pix_std_array,
            "normalized": normalized,
            "normalized_var": normalized_var,
            "fibre_weights": fibre_weights,
            "best_shift": best_shift,
            "best_sigma": best_sigma,
            "mean_shift": mean_shift,
            "mean_sigma": mean_sigma,
            "std_shift": std_shift,
            "std_sigma": std_sigma,
        }

        # Build report
        report = {"wave_centers": centers_val, "per_fibre": {}}
        for i in range(n_fibre):
            report["per_fibre"][i] = {
                "best": {"shift_pix": best_shift[i], "sigma_pix": best_sigma[i]},
                "mean": {"shift_pix": mean_shift[i], "sigma_pix": mean_sigma[i]},
                "std": {"shift_pix": std_shift[i], "sigma_pix": std_sigma[i]},
                "poly_sigma": {
                    "deg": self._lsf_per_fibre[i]["poly_deg"],
                    "coeff": self._lsf_per_fibre[i]["poly_coeff"],
                },
                "poly_shift": {
                    "deg": self._shift_per_fibre[i]["poly_deg"],
                    "coeff": self._shift_per_fibre[i]["poly_coeff"],
                },
            }
        return report



    def estimate_lsf_vs_wavelength(
        self,
        spectra_container: SpectraContainer,
        n_windows=12,
        window_overlap=0.25,
        pix_std_array=None,
        poly_deg=3,
        sun_window_size_aa=20,
        keep_features_frac=0.15,
        response_window_size_aa=200,
        mask_tellurics=True,
    ):
        """
        Estimate LSF sigma (in pixels) as a function of wavelength, per fibre.

        Faster approach:
        1) Build response-normalized spectra for all fibres over full band.
        2) Build absorption-feature weights over full band from the solar template.
        3) For each sigma in pix_std_array, convolve the solar template ONCE (full band).
        4) Compute weighted chi2 for each sigma x fibre x lambda.
        5) Collapse chi2 into overlapping wavelength windows; pick best sigma per fibre per window.
        6) Fit a smooth polynomial sigma(lambda) PER FIBRE and store models in self._lsf_per_fibre.

        Returns
        -------
        report : dict
            Keys:
            - "wave_centers": array of window centers
            - "per_fibre": dict fibre_index -> {"sigma_pix": per-window best sigma, "poly_deg": int, "poly_coeff": np.ndarray}
        """
        if pix_std_array is None:
            pix_std_array = np.arange(0.1, 5.0, 0.1)
        
        pix_std_array = check_unit(pix_std_array, u.pixel)
        max_pix_std = np.max(pix_std_array)

        wave = spectra_container.wavelength
        n_fibre, n_wave = spectra_container.rss_intensity.shape

        # 1) Build working arrays on the native grid
        # Response estimate per fibre (median filter scale set by response_window_size_aa)
        delta_pixel = int(
            (
                check_unit(response_window_size_aa, u.AA)
                / (wave[-1] - wave[0])
                * n_wave
            ).decompose()
        )
        if delta_pixel % 2 == 0:
            delta_pixel += 1

        # Interpolate solar template to instrument grid
        sun_on_grid = flux_conserving_interpolation(wave, self.sun_wavelength, self.sun_intensity)

        # Optional masking of telluric-dominated regions by down-weighting later
        # Weights that emphasize solar absorption features across full band
        weights_full = self.get_solar_features(wave, sun_on_grid, window_size_aa=sun_window_size_aa)
        if keep_features_frac is not None:
            thr = np.nanpercentile(weights_full, 100.0 * (1.0 - keep_features_frac))
            weights_full = np.where(weights_full >= thr, weights_full, 0.0)
        # Remove edges where convolution would be unstable
        sigma_wl = (wave[1] - wave[0]) * max_pix_std.value
        weights_full[wave < (wave[0] + 3 * sigma_wl)] = 0.0
        weights_full[wave > (wave[-1] - 3 * sigma_wl)] = 0.0
        s = np.nansum(weights_full)
        if s > 0:
            weights_full /= s
        else:
            raise RuntimeError("No weights available after masking telluric regions; adjust parameters.")
        # Response per fibre: data / solar
        response = spectra_container.rss_intensity / sun_on_grid[np.newaxis]
        smooth_resp = median_filter(response, size=(1, delta_pixel)) << response.unit
        upper_env = percentile_filter(smooth_resp, percentile=95, size=(1, delta_pixel)) << response.unit
        fibre_weights = 1.0 / (1.0 + (upper_env / smooth_resp - np.nanmedian(upper_env / smooth_resp)) ** 2)

        if mask_tellurics:
            # Optionally compress fibre_weights in typical telluric regions by multiplying with weights_full
            # This is a mild down-weighting if telluric windows were not fully excluded
            fibre_weights = fibre_weights * (weights_full[np.newaxis, :] + np.nanmedian(weights_full))

        # Normalize data by smooth response
        normalized = spectra_container.rss_intensity / smooth_resp  # shape (n_fibre, n_wave)

        # 2) Build model grid across full band for all sigma
        models_sigma = np.zeros((pix_std_array.size, n_wave)) << sun_on_grid.unit
        weights_sigma = np.zeros((pix_std_array.size, n_wave))

        for j, sigma_pix in enumerate(pix_std_array):
            # No shift here; only broadening for LSF(lambda)
            broadened = gaussian_filter(sun_on_grid, sigma_pix.value) << sun_on_grid.unit
            models_sigma[j] = broadened

            # Weights convolved for consistency; truncate wings
            w = gaussian_filter(weights_full, sigma_pix.value, truncate=2.0)
            s = np.nansum(w)
            weights_sigma[j] = w / s if s > 0 else w

        # 3) Compute weighted chi2 cube across full wavelength for all fibres
        #    chi2[j, i, k] for sigma_j, fibre_i, lambda_k
        #    Use broadcast: (n_sigma, 1, n_wave) vs (1, n_fibre, n_wave)
        model_3d = models_sigma[:, np.newaxis, :]      # (n_sigma, 1, n_wave)
        data_3d = normalized[np.newaxis, :, :]   # (1, n_fibre, n_wave)
        w_line_3d = weights_sigma[:, np.newaxis, :]    # (n_sigma, 1, n_wave)
        w_fibre_3d = fibre_weights.value[np.newaxis, :, :]  # (1, n_fibre, n_wave)

        # combine weights: per-sigma line weights times per-fibre stability weights
        weight_total = w_line_3d * w_fibre_3d 
        diff = (model_3d - data_3d)  # (n_sigma, n_fibre, n_wave)
        num_lambda = (diff.value ** 2) * weight_total                     # (n_sigma, n_fibre, n_wave)
        den_lambda = weight_total                                       # (n_sigma, n_fibre, n_wave)

        # 4) Define overlapping windows over wavelength and collapse chi2 into windows
        centers, slices = self._build_overlapping_windows(wave, n_windows, window_overlap)

        n_win = len(slices)
        chi2_win = np.zeros((pix_std_array.size, n_fibre, n_win))
        for k, sl in enumerate(slices):
            nsum = np.nansum(num_lambda[:, :, sl], axis=2)
            dsum = np.nansum(den_lambda[:, :, sl], axis=2)
            dsum[dsum <= 0] = 1.0
            chi2_win[:, :, k] = nsum / dsum  # (n_sigma, n_fibre)

        # 5) Best sigma per fibre per window, then per-fibre polynomial fit
        sigma_best_per_fibre = np.zeros((n_fibre, n_win), dtype=float)
        for i in range(n_fibre):
            # argmin over sigma for each window
            best_idx = np.argmin(chi2_win[:, i, :], axis=0)   # (n_win,)
            sigma_best = pix_std_array[best_idx].value        # float array
            sigma_best_per_fibre[i] = sigma_best

        # Fit polynomial per fibre over window centers
        self._lsf_per_fibre = {"sigma_best_per_fibre": sigma_best_per_fibre}
        centers_val = np.array([c.to_value(wave.unit) for c in centers], dtype=float)
        for i in range(n_fibre):
            y = sigma_best_per_fibre[i]
            # Guard against too few points
            deg = min(poly_deg, max(0, len(centers_val) - 1))
            if len(centers_val) < 3:
                coeff = np.array([np.nanmedian(y) if np.isfinite(y).any() else 2.0])
                deg = 0
            else:
                coeff = np.polyfit(centers_val, y, deg=deg)
            self._lsf_per_fibre[i] = {
                "poly_deg": deg,
                "poly_coeff": coeff,
                "wave_centers": centers_val.copy(),
                "sigma_pix": y.copy(),
            }

        # Also keep a simple field-average for convenience
        field_avg = np.nanmedian(sigma_best_per_fibre, axis=0)
        deg_field = min(poly_deg, max(0, len(centers_val) - 1))
        coeff_field = np.polyfit(centers_val, field_avg, deg=deg_field) if len(centers_val) >= 3 else np.array([np.nanmedian(field_avg)])
        self._lsf_knots_wave = centers_val.copy()
        self._lsf_knots_sigma_pix = field_avg.copy()
        self._lsf_poly_deg = deg_field
        self._lsf_poly_coeff = coeff_field

        self._lsf_diag = {
            "wave": wave,
            "centers": centers,
            "slices": slices,
            "pix_std_array": pix_std_array,
            "models_sigma": models_sigma,
            "weights_sigma": weights_sigma,
            "normalized": normalized,
            "fibre_weights": fibre_weights,
            "chi2_win": chi2_win,
            "sigma_best_per_fibre": sigma_best_per_fibre,
        }
        return {
            "wave_centers": centers_val,
            "per_fibre": {i: {"sigma_pix": self._lsf_per_fibre[i]["sigma_pix"],
                            "poly_deg": self._lsf_per_fibre[i]["poly_deg"],
                            "poly_coeff": self._lsf_per_fibre[i]["poly_coeff"]}
                        for i in range(n_fibre)},
        }


    def evaluate_lsf_sigma_pix(self, wavelength, fibre=None):
        """
        Evaluate sigma(lambda) in pixels.

        Parameters
        ----------
        wavelength : Quantity array
        fibre : int or None
            If int, return per-fibre model. If None, return field-average model.

        Returns
        -------
        sigma_pix : ndarray of floats
        """
        lam = check_unit(wavelength, self.sun_wavelength.unit).to_value(self.sun_wavelength.unit)

        if fibre is None:
            if self._lsf_poly_coeff is None:
                raise RuntimeError("Field-average LSF model not available. Run estimate_lsf_vs_wavelength.")
            return np.polyval(self._lsf_poly_coeff, lam)

        if (self._lsf_per_fibre is None) or (fibre not in self._lsf_per_fibre):
            raise RuntimeError("Per-fibre LSF model not available for the requested fibre.")
        pf = self._lsf_per_fibre[fibre]
        return np.polyval(pf["poly_coeff"], lam)

    def plot_lsf_fit_for_fibre(
        self,
        fibre_idx,
        max_windows=None,
        show_models=True,
        show_residuals=True,
        figsize=(10, 6),
    ):
        """
        Make QC plots for one fibre across all wavelength windows used in the LSF fit.

        Requirements
        ------------
        Call estimate_lsf_vs_wavelength first; this method consumes its cached diagnostics.

        Parameters
        ----------
        fibre_idx : int
            RSS fibre index to inspect.
        max_windows : int or None
            If set, limit to the first N windows (useful for quick checks).
        show_models : bool
            Plot model overlay in each window panel.
        show_residuals : bool
            Plot residuals panel beneath each window panel.
        figsize : tuple
            Base figure size per window.

        Returns
        -------
        figs : list of matplotlib.figure.Figure
            One figure per window, plus a summary figure at the end.
        """
        if not hasattr(self, "_lsf_diag"):
            raise RuntimeError("No diagnostics available. Run estimate_lsf_vs_wavelength first.")

        d = self._lsf_diag
        wave = d["wave"]
        centers = d["centers"]
        slices_ = d["slices"]
        pix_std_array = d["pix_std_array"]
        models_sigma = d["models_sigma"]
        weights_sigma = d["weights_sigma"]
        normalized = d["normalized"]
        fibre_weights = d["fibre_weights"]
        chi2_win = d["chi2_win"]
        sigma_best_per_fibre = d["sigma_best_per_fibre"]

        if fibre_idx < 0 or fibre_idx >= normalized.shape[0]:
            raise IndexError("fibre_idx out of range")

        n_win = len(slices_)
        use_windows = n_win if max_windows is None else min(max_windows, n_win)

        figs = []

        # window-by-window panels
        for w in range(use_windows):
            sl = slices_[w]
            lam = wave[sl]
            data = normalized[fibre_idx, sl]
            fw = fibre_weights[fibre_idx, sl]

            # best sigma index for this window and fibre
            j_best = int(np.argmin(chi2_win[:, fibre_idx, w]))
            sigma_best = pix_std_array[j_best]
            model = models_sigma[j_best, sl]
            weights_line = weights_sigma[j_best, sl]

            fig, (ax1, ax2) = plt.subplots(
                2 if show_residuals else 1, 1, figsize=figsize, sharex=True,
                gridspec_kw=dict(height_ratios=[2, 1] if show_residuals else [1])
            )
            if not isinstance(ax1, plt.Axes):
                # matplotlib returns ndarray of axes if 2 rows; normalize
                ax1, ax2 = ax1[0], ax1[1] if show_residuals else (ax1[0], None)

            ax1.set_title(f"Fibre {fibre_idx}  Window {w+1}/{n_win}  center={centers[w]:.3g}  sigma_best={sigma_best:.3g} pix")
            ax1.plot(lam, data, lw=1.0, color="k", label="data")
            if show_models:
                ax1.plot(lam, model, lw=1.0, color="r", label="model")
            # twin axis to show weights used
            twin = ax1.twinx()
            twin.plot(lam, weights_line, alpha=0.5, color="fuchsia", label="line weight")
            twin.plot(lam, fw.value if hasattr(fw, "value") else fw, alpha=0.5, color="orange", label="fibre weight")
            ax1.set_ylabel("Norm. intensity")
            ax1.legend(loc="upper left", fontsize=8)
            twin.legend(loc="upper right", fontsize=8)

            if show_residuals:
                resid = (data - model).value if hasattr(data, "value") else (data - model)
                ax2.plot(lam, resid, lw=0.8)
                ax2.axhline(0.0, color="k", lw=0.7, alpha=0.6)
                ax2.set_ylabel("Residual")
                ax2.set_xlabel(f"Wavelength [{wave.unit}]")
            else:
                ax1.set_xlabel(f"Wavelength [{wave.unit}]")

            plt.tight_layout()
            plt.close(fig)
            figs.append(fig)

        # summary panel for sigma(lambda)
        centers_val = np.array([c.to_value(wave.unit) for c in centers], dtype=float)
        y = sigma_best_per_fibre[fibre_idx]
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.set_title(f"Fibre {fibre_idx}  sigma(lambda) best per window")
        ax.plot(centers_val, y, "o", label="per-window best")
        # overlay per-fibre polynomial fit if present
        if hasattr(self, "_lsf_per_fibre") and (fibre_idx in self._lsf_per_fibre):
            pf = self._lsf_per_fibre[fibre_idx]
            xx = np.linspace(centers_val.min(), centers_val.max(), 512)
            ax.plot(xx, np.polyval(pf["poly_coeff"], xx), "-", label=f"poly deg={pf['poly_deg']}")
        ax.set_xlabel(f"Wavelength center [{wave.unit}]")
        ax.set_ylabel("sigma (pix)")
        ax.legend()
        plt.tight_layout()
        plt.close(fig)
        figs.append(fig)

        return figs


    def compute_shift_from_twilight(self, spectra_container,
                                    sun_window_size_aa=20, keep_features_frac=0.1,
                                    response_window_size_aa=200,
                                    wave_range=None,
                                    pix_shift_array=None,
                                    pix_std_array=None,
                                    logspace=True, use_mean=True,
                                    inspect_fibres=None):
        """Compute the wavelenght offset of between a given SpectraContainer and a reference Solar spectra.
        
        Parameters
        ----------
        spectra_container: `pykoala.data_container.SpectraContainer`
            Spectra container (RSS or Cube) to cross-correlate with the reference
            spectra.
        sun_window_size_aa: int, optional
            Size of a spectral window in angstrom to perform a median filtering
            and estimate the underlying continuum. Default is 20 AA.
            See `get_solar_features` for details.
        keep_features_frac: float, optional
            Fraction of absorption-features weights to keep. All values below
            that threshold will be set to 0. Default is 0.1.
        wave_range: list or tuple, optional
            If provided, the cross-correlation will only be done in the provided
            wavelength range. Default is None.
        pix_shift_array: 1D-np.array, optional, default=np.arange(-5, 5, 0.1)
            Array containing the wavelength offsets expressed in pixels.
        pix_std_array: 1D-np.array, optional, default=np.arange(0.1, 3, 0.1)
            Array containing the values of the gaussian LSF standard deviation
            in pixels. See `compute_grid_of_models` for details.
        logspace: bool, optional
            If True, the cross-correlation will be perform using a logarithmic
            sampling in terms of wavelength. Default is True.
        use_mean: bool, optional
            If True, the mean likelihood-weighted value of the wavelength offset
            is used to create the `WavelengthOffsetCorrection`. Otherwise, the
            best fit parameters of the input grid are used. Default is True.
        inspect_fibres: list or tuple, optional
            Iterable containing RSS-wise spectra indices. If provided, a
            quality-control plot of each fibre is produced.
        
        Returns
        -------
        results: dict
            The dictionary contains the ``best-fit`` and ``mean`` likelihood-weighted
            values of ``pix_shift_array`` and ``pix_std_array`` in a tuple, respectively.
            If ``inspect_fibres`` is not ``None``, it containes a list of figures
            for each fibre included in ``inspect_fibres``.

        """
        if pix_shift_array is None:
            pix_shift_array = np.arange(-5, 5, 0.1)
        if pix_std_array is None:
            pix_std_array = np.arange(0.1, 3, 0.1)

        pix_shift_array = check_unit(pix_shift_array, u.pixel)
        pix_std_array = check_unit(pix_std_array, u.pixel)

        if logspace:
            new_wavelength = np.geomspace(spectra_container.wavelength[0],
                                          spectra_container.wavelength[-1],
                                          spectra_container.wavelength.size)
            rss_intensity = np.array([flux_conserving_interpolation(
                new_wavelength, spectra_container.wavelength, fibre
                ) for fibre in spectra_container.rss_intensity]
                ) << spectra_container.intensity.unit
        else:
            new_wavelength = spectra_container.wavelength
            rss_intensity = spectra_container.rss_intensity
        
        # Interpolate the solar spectrum to the new grid of wavelengths
        sun_intensity = flux_conserving_interpolation(
        new_wavelength, self.sun_wavelength, self.sun_intensity)

        # Make an array of weights to focus on the absorption lines
        if wave_range is None:
            weights = self.get_solar_features(new_wavelength, sun_intensity,
                                            window_size_aa=sun_window_size_aa)
            weights[weights < np.nanpercentile(weights, 100*(1 - keep_features_frac))] = 0
            weights[:100] = 0
            weights[-100:] = 0
        else:
            weights = np.zeros(new_wavelength.size)
            weights[slice(*np.searchsorted(new_wavelength, wave_range))] = 1.0

        valid_pixels = weights > 0
        self.vprint("Number of pixels with non-zero weights: "
                    + f"{np.count_nonzero(valid_pixels)} out of {valid_pixels.size}")

        # Estimate the response curve for each individual fibre
        delta_pixel = int(
            (check_unit(response_window_size_aa, u.AA)
                        / (new_wavelength[-1] - new_wavelength[0])
                        * new_wavelength.size).decompose()
                        )
        if delta_pixel % 2 == 0:
            delta_pixel += 1

        response_spectrograph = rss_intensity / sun_intensity[np.newaxis]
        smoothed_r_spectrograph = median_filter(
            response_spectrograph, delta_pixel, axes=1) << response_spectrograph.unit
        spectrograph_upper_env = percentile_filter(
            smoothed_r_spectrograph, 95, delta_pixel, axes=1) << response_spectrograph.unit
        # Avoid regions dominated by telluric absorption
        self.vprint("Including the masking of pixels dominated by telluric absorption")
        fibre_weights =  1 / (1  + (
                spectrograph_upper_env / smoothed_r_spectrograph
                - np.nanmedian(spectrograph_upper_env / smoothed_r_spectrograph)
                )**2)

        normalized_rss_intensity = rss_intensity / smoothed_r_spectrograph
        # Generate and fit the model
        pix_array = np.arange(new_wavelength.size) << u.pixel

        models_grid, weights_grid = self.compute_grid_of_models(
            pix_shift_array, pix_std_array, pix_array, sun_intensity, weights)

        # loop over one variable to avoir memory errors
        all_chi2 = np.zeros((pix_shift_array.size,
                             pix_std_array.size,
                             rss_intensity.shape[0]))
        
        self.vprint("Performing the cross-correlation with the grid of models")
        for i in range(pix_shift_array.size):
            all_chi2[i] = np.nansum(
                (models_grid[i, :, np.newaxis]
                 - normalized_rss_intensity[np.newaxis, :, :]).value**2
                * weights_grid[i, :, np.newaxis]
                * fibre_weights[np.newaxis, :, :],
                axis=-1) / np.nansum(
                    weights_grid[i, :, np.newaxis]
                    * fibre_weights[np.newaxis, :, :],
                    axis=-1)
            
        likelihood = np.exp(- (all_chi2 - all_chi2.min())/ 2)
        likelihood /= np.nansum(likelihood, axis=(0, 1))[np.newaxis, np.newaxis, :]
        mean_pix_shift = np.sum(likelihood.sum(axis=1)
                                * pix_shift_array[:, np.newaxis], axis=0)
        mean_std = np.sum(likelihood.sum(axis=0)
                          * pix_std_array[:, np.newaxis], axis=0)

        best_fit_idx = np.argmax(likelihood.reshape((-1, likelihood.shape[-1])),
                                 axis=0)
        best_vel_idx, best_std_idx = np.unravel_index(
                best_fit_idx, all_chi2.shape[:-1])
        best_sigma, best_shift = (pix_std_array[best_std_idx],
                                    pix_shift_array[best_vel_idx])

        if inspect_fibres is not None:
            fibre_figures = self.inspect_fibres(
                inspect_fibres, pix_shift_array, pix_std_array,
                best_vel_idx, best_std_idx, mean_pix_shift, mean_std,
                likelihood, models_grid, weights_grid, normalized_rss_intensity,
                new_wavelength)
        else:
            fibre_figures= None
        if use_mean:
            self.vprint("Using mean likelihood-weighted values to compute the wavelength offset correction")
            self.offset.offset_data = - mean_pix_shift
        else:
            self.vprint("Using best fit values to compute the wavelength offset correction")
            self.offset.offset_data = - best_shift
        
        self.offset.offset_error = np.full_like(best_shift, fill_value=np.nan)

        return {"best-fit": (best_shift, best_sigma),
                "mean": (mean_pix_shift, mean_std),
                "fibre_figures": fibre_figures}
    
    def inspect_fibres(self, fibres, pix_shift_array, pix_std_array,
                       best_vel_idx, best_std_idx, mean_vel, mean_std,
                       likelihood,
                       models_grid, weights_grid,
                       normalized_rss_intensity, wavelength):
        """
        Create a quality control plot of the solar cross-correlation process of each input fibre.

        Parameters
        ----------
        fibres: iterable
            List of input fibres to check.
        pix_shift_array: 1D-np.array
            Array containing the wavelength offsets expressed in pixels.
        pix_std_array: 1D-np.array
            Array containing the values of the gaussian LSF standard deviation
            in pixels. See :func:`compute_grid_of_models` for details.
        best_vel_idx: int
            Index of ``pix_shift_array`` that correspond to the best fit.
        best_std_idx: int
            Index of ``pix_std_array`` that correspond to the best fit.
        mean_vel: float
            Mean likelihood-weighted values of ``pix_shift_array``.
        mean_std: float
            Mean likelihood-weighted values of ``pix_std_array``.
        likelihood: numpy.ndarray:
            Likelihood of the cross-correlation.
        models_grid: numpy.ndarray
            Grid of solar spectra models. See :func:`compute_grid_of_models` for details.
        weights_grid: numpy.ndarray
            Grid of solar spectra weights. See :func:`compute_grid_of_models` for details.
        normalized_rss_intensity: numpy.ndarray
            Array containing the RSS intensity values of a SpectraContainer including
            the correction of the spectrograph response curve.
        wavelength: np.array
            Wavelength array associated to ``normalized_rss_intensity`` and ``models_grid``.

        Returns
        -------
        fibres_figures: list
            List of figures containing a QC plot of each fibre.
        """
        fibre_figures = []
        best_sigma, best_shift = (pix_std_array[best_std_idx],
                                  pix_shift_array[best_vel_idx])
        for fibre in fibres:
            self.vprint(f"Inspecting input fibre: {fibre}")
            fig = plt.figure(constrained_layout=True, figsize=(10, 8))
            gs = GridSpec(2, 4, figure=fig, wspace=0.25, hspace=0.25)

            ax = fig.add_subplot(gs[0, 0])
            mappable = ax.pcolormesh(
                pix_std_array, pix_shift_array, likelihood[:, :, fibre],
                cmap='gnuplot',
                norm=LogNorm(vmin=likelihood.max() / 1e2, vmax=likelihood.max()))
            plt.colorbar(mappable, ax=ax,
                         label=r"$e^(-\sum_\lambda w(I - \hat{I}(s, \sigma))^2 / 2)$")
            ax.plot(best_sigma[fibre], best_shift[fibre], '+', color='cyan',
                    label=r'Best fit: $\Delta\lambda$='
                    + f'{best_shift[fibre]:.2}, ' + r'$\sigma$=' + f'{best_sigma[fibre]:.2f}')
            ax.plot(mean_std[fibre], mean_vel[fibre], 'o', mec='lime', mfc='none',
                    label=r'Mean value: $\Delta\lambda$='
                    + f'{mean_vel[fibre]:.2}, ' + r'$\sigma$=' + f'{mean_std[fibre]:.2f}')
            ax.set_xlabel(r"$\sigma$ (pix)")
            ax.set_ylabel(r"$\Delta \lambda$ (pix)")
            ax.legend(bbox_to_anchor=(0., 1.05), loc='lower left', fontsize=7)

            sun_intensity = models_grid[best_vel_idx[fibre],
                                        best_std_idx[fibre]]
            weight = weights_grid[best_vel_idx[fibre],
                                        best_std_idx[fibre]]

            ax = fig.add_subplot(gs[0, 1:])
            ax.set_title(f"Fibre: {fibre}")
            ax.plot(wavelength, sun_intensity, label='Sun Model')
            ax.plot(wavelength, normalized_rss_intensity[fibre],
                    label='Fibre', lw=2)
            twax = ax.twinx()
            twax.plot(wavelength, weight, c='fuchsia',
                    zorder=-1, alpha=0.5, label='Weight')
            ax.legend(fontsize=7)
            ax.set_ylabel("Intensity")
            ax.set_xlabel("Wavelength")
            twax.set_ylabel("Relative weight")

            ax = fig.add_subplot(gs[1, :])
            max_idx = np.argmax(weight)
            max_weight_range = range(np.max((max_idx - 80, 0)),
                  np.min((max_idx + 80, wavelength.size - 1)))
            ax.plot(wavelength[max_weight_range],
                    sun_intensity[max_weight_range], label='Model')
            ax.plot(wavelength[max_weight_range],
                    normalized_rss_intensity[fibre][max_weight_range],
                    label='Fibre', lw=2)
            
            ax.set_xlim(wavelength[max_weight_range][0],
                        wavelength[max_weight_range][-1])
            twax = ax.twinx()
            twax.plot(wavelength[max_weight_range], weight[max_weight_range],
                      c='fuchsia',
                    zorder=-1, alpha=0.5, label='Absorption-feature Weight')
            twax.axhline(0)
            twax.legend(fontsize=7)
            ax.set_ylabel("Intensity")
            ax.set_xlabel("Wavelength")
            twax.set_ylabel("Relative weight")
            fibre_figures.append(fig)
            plt.close(fig)
        return fibre_figures


# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
