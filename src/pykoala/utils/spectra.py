"""This module contains basic tools for manipulating spectra"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import astropy.units as u

from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import (percentile_filter, median_filter, maximum_filter,
                           gaussian_filter1d, generic_filter, label)
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d, make_smoothing_spline

from pykoala import vprint
from pykoala.ancillary import check_unit
from pykoala.utils.math import odd_int, integrated_autocorr_time, std_from_mad, poly_extrapolate_wrapper


def flux_conserving_interpolation(new_wave : u.Quantity, wave : u.Quantity,
                                  spectra : u.Quantity) -> u.Quantity:
    """Interpolate a spectra to a new grid of wavelengths preserving the flux density.
    
    Parameters
    ----------
    new_wave : :class:`np.ndarray` or :class:`astropy.units.Quantiy`
        New grid of wavelengths
    wave : np.ndarray
        Original grid of wavelengths
    spectra : :class:`astropy.units.Quantity`
        Spectra associated to `wave`.
    
    Returns
    -------
    interp_spectra : np.ndarray
        Interpolated spectra to `new_wave`
    """
    wave = check_unit(wave)
    new_wave = check_unit(new_wave, wave.unit)
    # Strict check
    # Spectra can have different, non-compatible units, such as ADU or flam
    spectra = check_unit(spectra)
    mask = np.isfinite(spectra)
    masked_wave = wave[mask]

    wave_limits = 1.5 * masked_wave[[0, -1]] - 0.5 * masked_wave[[1, -2]]
    wave_edges = np.hstack(
        [wave_limits[0],
         (masked_wave[1:] + masked_wave[:-1])/2,
         wave_limits[1]])

    new_wave_limits = 1.5 * new_wave[[0, -1]] - 0.5 * new_wave[[1, -2]]
    new_wave_edges = np.hstack(
        [new_wave_limits[0],
         (new_wave[1:] + new_wave[:-1])/2,
         new_wave_limits[1]])
    cumulative_spectra = np.cumsum(spectra[mask] * np.diff(wave_edges))
    cumulative_spectra = np.insert(cumulative_spectra, 0,
                                   0 << cumulative_spectra.unit)
    new_cumulative_spectra = np.interp(new_wave_edges, wave_edges,
                                       cumulative_spectra)
    interp_spectra = np.diff(new_cumulative_spectra) / np.diff(new_wave_edges)
    return interp_spectra

def local_sigma_from_residuals(residuals: np.ndarray, win_pix: int) -> np.ndarray:
    """Robust local noise estimator using the positive side of residuals.
    Approximates sigma as the 84th percentile of max(residual, 0) in a sliding window.
    """
    pos = np.clip(residuals, 0.0, None)
    # use percentile of positives; if all zeros locally, this returns 0
    sig = percentile_filter(pos, percentile=84, size=win_pix, mode="mirror")
    return sig

def estimate_continuum_and_mask_absorption(
    wave,
    flux,
    *,
    variance: u.Quantity = None,
    cont_window: u.Quantity = 50 << u.AA,
    cont_percentile: float = 90.0,
    cont_smooth_sigma: u.Quantity = 25.0 << u.AA,
    smooth_spline : bool = False,
    abs_kappa_sigma : int = 2,
    ):
    """
    Estimate a smooth continuum, identify absorption dominated regions, and
    summarize each absorption feature with pixel ids and equivalent width.

    The algorithm computes an upper envelope continuum via a running percentile
    filter and optional Gaussian smoothing. The spectrum is normalized by this
    continuum, absorption peaks are found on 1 - normalized with a prominence
    and depth threshold, and contiguous regions for each feature are defined
    from peak bases. The mask is True on pixels dominated by absorption lines.

    Parameters
    ----------
    wave
        Wavelength array, 1D. If not a Quantity, Angstrom is assumed.
    flux
        Flux density array on the same grid as wave. May be a Quantity in any
        flux unit or a plain array. The continuum will carry the same unit.
    cont_window
        Window size in pixels for the percentile filter used to trace the
        continuum. Must be positive; even values are rounded to the next odd.
    cont_percentile
        Upper percentile for the envelope, typically 85 to 95 for absorption
        dominated spectra.
    cont_smooth_sigma
        Standard deviation in pixels of a Gaussian applied to the percentile
        continuum. Set to None or 0 to skip.
    min_depth
        Minimum depth in normalized units required to consider a line. A value
        of 0.02 means a 2 percent dip below the continuum.
    min_prominence
        Minimum peak prominence on the inverted profile. If None, defaults to
        min_depth.
    min_distance_pix
        Minimum separation in pixels between absorption peaks.
    rel_height
        Relative height used by peak_widths to define left and right bases.
        A value of 0.5 approximates FWHM on the inverted profile.

    Returns
    -------
    continuum
        Continuum array with the same unit as flux and same shape as input.
    absorption_mask
        Boolean mask, True for pixels considered dominated by absorption
        features and recommended to be ignored during flux calibration.
    features
        Dictionary keyed by feature id (0..n_features-1). For each feature:
            center_index : int
            center_wavelength : Quantity
            left_index : int
            right_index : int
            left_wavelength : Quantity
            right_wavelength : Quantity
            pixel_indices : ndarray of int (the masked pixels for this line)
            equivalent_width : Quantity (same unit as wavelength)
            depth : float (1 - normalized at peak)
            prominence : float (from find_peaks on inverted profile)

    Notes
    -----
    1. The mask is built by union of intervals between left and right bases
       of each detected absorption feature as given by peak_widths.
    2. Equivalent widths are integrated as EW = integral (1 - normalized) d lambda
       between the bases on the native wavelength grid.
    """
    vprint("Estimating spectra continuum and absorption features")
    wave = check_unit(wave, u.AA)
    if isinstance(flux, u.Quantity):
        flux_unit = flux.unit
        fvals = flux.to_value(flux_unit)
    else:
        flux_unit = 1.0 * u.one  # unitless placeholder
        fvals = np.asarray(flux, dtype=float)

    if variance is not None:
        variance = check_unit(variance, flux_unit**2)
        var = variance.to_value(flux_unit**2).copy()
    else:
        var = np.ones_like(fvals)

    if wave.ndim != 1 or fvals.ndim != 1 or wave.size != fvals.size:
        raise ValueError("wave and flux must be 1D arrays of the same length")

    n = wave.size

    # Finite mask
    finite = np.isfinite(wave.value) & np.isfinite(fvals)
    if not np.any(finite):
        # Nothing usable
        continuum = np.full(n, np.nan) * flux_unit
        absorption_mask = np.zeros(n, dtype=bool)
        return continuum, absorption_mask, {}

    fvals_f = fvals[finite]
    # Percentile continuum (upper envelope)
    cont_window = check_unit(cont_window, u.AA)
    win = cont_window if cont_window is not None else 20 << u.AA
    dwl = np.diff(wave)
    win_pixel = np.mean(win / dwl).decompose()
    win_pixel = odd_int(int(win_pixel))

    # Mask emission lines
    def _nmad_filter(x, size, mode="nearest"):
        return generic_filter(x, std_from_mad, size=size, mode=mode)

    median_filter_extrap = poly_extrapolate_wrapper(
        median_filter, axis=-1, polyorder=1, pad_strategy="size"
    )

    perc_extrap = poly_extrapolate_wrapper(percentile_filter, axis=-1,
        polyorder=1, pad_strategy="size")

    nmad_extrap_filter = poly_extrapolate_wrapper(
        _nmad_filter, axis=-1, polyorder=1, pad_strategy="size"
    )

    f = fvals_f.copy()
    std_vals   = nmad_extrap_filter(f, size=win_pixel)
    median_vals = median_filter_extrap(f, size=win_pixel)
    iteration = 5
    while iteration:
        peaks, _ = find_peaks(f, height=median_vals + 2 * std_vals, distance=5)
        if peaks.size > 0:
            vprint("Emission lines found", peaks.size)
            # Mask all peaks with the median value
            widths, wheights, lpts, rpts = peak_widths(f, peaks)
            for l, r in zip(lpts, rpts):
                mask_line = slice(max(0, int(l) - 1), min(f.size, int(r) + 1))
                f[mask_line] = median_vals[mask_line]
                var[mask_line] = std_vals[mask_line]**2
            median_vals = median_filter_extrap(f, size=win_pixel)
            iteration -= 1
        else:
            vprint("No emission lines found")
            break
    # Upper-percentile envelope and clipping to local maxima to avoid overshoot
    vprint("Computing continuum via asymmetric least squares")
    # cont_vals = asls_baseline(f, w=1/var)

    cont_vals = perc_extrap(f, percentile=cont_percentile,
                            size=win_pixel, mode="nearest")

    # Optional Gaussian smoothing
    if cont_smooth_sigma is not None:
        cont_smooth_sigma = check_unit(cont_smooth_sigma, u.AA)
        sigma_pixel = np.mean(cont_smooth_sigma / dwl).decompose()
        cont_vals = gaussian_filter1d(cont_vals, sigma=sigma_pixel, mode="mirror")

    # Robust local noise from positive residuals
    resid = fvals_f - cont_vals
    sigma_loc = nmad_extrap_filter(resid, size=win_pixel)

    # Prevent division by zero
    small = max(1e-12, np.nanpercentile(cont_vals, 5) * 1e-6)
    sigma_loc = np.clip(sigma_loc, small, None)

    if smooth_spline:
        weights = 1 / np.clip(sigma_loc, small * 1e-2, None)
        try:
            spline = make_smoothing_spline(wave.to_value("AA"), cont_vals,
                                           w=weights)
        except Exception as e:
            vprint(f"Smoothing spline failed with err: {e}", level="warning")    
        #cont_vals = spline(wave)
        cont_vals = np.maximum(cont_vals, spline(wave))

    # More realistic continuum uncertainty:
    # statistical error ~ sigma_loc / sqrt(n_eff), with n_eff from autocorrelation
    tau = integrated_autocorr_time(resid)
    n_eff = max(1.0, win_pixel / tau)
    cont_err = sigma_loc / np.sqrt(n_eff)

    abs_regions = fvals_f < cont_vals - abs_kappa_sigma * std_vals
    regions, n_features = label(abs_regions)
    vprint(f"No. of absorption features found: {n_features}",
           f"({np.count_nonzero(regions > 0)} pixels)")
    return cont_vals << flux_unit, cont_err << flux_unit, regions
