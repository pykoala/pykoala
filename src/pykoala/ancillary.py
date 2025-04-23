"""
Utility functions.

Many of them are partially implemented. They were in the
old, non-modular version of PyKOALA and have not been included
in the current modular scheme.
"""

# =============================================================================
# Basics packages
# =============================================================================
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy import optimize
from shapely import geometry
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits
from astropy import units as u
from astropy.modeling.models import custom_model
# =============================================================================

# =============================================================================
# PyKOALA modules
# =============================================================================
from pykoala import vprint


def check_unit(quantity, default_unit=None, equivalencies=[]):
    """Check the units of an input quantity.
    
    Parameters
    ----------
    quantity : np.ndarray or astropy.units.Quantity
        Input quantity.
    default_unit : astropy.units.Quantity, default=None
        If `quantity` has not units, it corresponds to the unit assigned to it.
        Otherwise, it is used to check the equivalency with `quantity`.
    
    Returns
    -------
    quantity: :class:`astropy.units.Quantity`
        Converted quantity.
    """
    if quantity is None:
        return quantity
    isq = isinstance(quantity, u.Quantity)
    if isq and default_unit is not None:
        if not quantity.unit.is_equivalent(default_unit,
                                           equivalencies=equivalencies):
            raise u.UnitTypeError(
                "Input quantity does not have the appropriate units")
        else:
            return quantity.to(default_unit, equivalencies=equivalencies)
    elif not isq and default_unit is not None:
        return quantity * default_unit
    elif not isq and default_unit is None:
        raise ValueError("Input value must be a astropy.units.Quantity")
    else:
        return quantity

def remove_unit(quantity, default_unit=None):
    """Convert an :class:`astropy.units.Quantity` into a :class:`numpy.array`.
    
    This method converts an input quantity into a an array eighter by taking
    the value associated to the current units, or after converting the quantity
    into the input units.

    Parameters
    ----------
    quantity : np.ndarray or astropy.units.Quantity
        Input quantity.
    default_unit : astropy.units.Quantity, default=None
        If `quantity` has not units, it corresponds to the unit assigned to it.
        Otherwise, it is used to check the equivalency with `quantity`.
    
    Returns
    -------
    array: :class:`numpy.array`
        Array associated to the input quantity.
    """
    isq = isinstance(quantity, u.Quantity)
    if isq and default_unit is not None:
        if not quantity.unit.is_equivalent(default_unit):
            raise u.UnitTypeError(
                "Input quantity does not have the appropriate units")
        else:
            return quantity.to_value(default_unit)
    elif not isq:
        return quantity
    else:
        return quantity.value

def preserve_units_dec(func):
    """Decorator method to preserve `astropy.Units` on input arguments."""
    def wrapper(data, *args, **kwargs):
        if isinstance(data, u.Quantity):
            unit = data.unit
        else:
            unit = 1
        val = func(data, *args, **kwargs)
        
        if not isinstance(val, u.Quantity):
            return val << unit
        else:
            return val
    return wrapper

def remove_units_dec(func):
    """Decorator method to remove `astropy.Units` from input arguments."""
    def wrapper(*args, **kwargs):
        unitless_args = [remove_unit(a) for a in args]
        unitless_kwargs = {k : remove_unit(v) for k, v in kwargs.items()}
        return func(*unitless_args, **unitless_kwargs)
    return wrapper


def detect_edge(rss):
    """
    Detect the edges of a RSS. Returns the minimum and maximum wavelength that 
    determine the maximum interval with valid (i.e. no masked) data in all the 
    spaxels.

    Parameters
    ----------
    rss : RSS object.

    Returns
    -------
    min_w : float
        The lowest value (in units of the RSS wavelength) with 
        valid data in all spaxels.
    min_index : int
        Index of min_w in the RSS wavelength variable.
    max_w : float
        The higher value (in units of the RSS wavelength) with 
        valid data in all spaxels.
    max_index : int
        Index of max_w in the RSS wavelength variable.

    """
    collapsed = np.sum(rss.intensity, 0)
    nans = np.isfinite(collapsed)
    wavelength = rss.wavelength
    min_w = wavelength[nans].min()
    min_index = wavelength.tolist().index(min_w)
    max_w = wavelength[nans].max()
    max_index = wavelength.tolist().index(max_w)
    return min_w, min_index, max_w, max_index

# ----------------------------------------------------------------------------------------------------------------------
# WCS operations
# ----------------------------------------------------------------------------------------------------------------------


def update_wcs_coords(wcs, ra_dec_val=None, ra_dec_offset=None):
    """Update the celestial reference values of a WCS.

    Update the celestial coordinates of a WCS using new central values of RA
    and DEC or relative offsets expressed in degree.

    Parameters
    ----------
    - wcs: asteropy.wcs.WCS
        Target WCS to update.
    - ra_dec_val: list or tupla, default=None
        New CRVAL of RA and DEC. Both elements must be instances of
        :class:`astropy.units.Quantity`.
    - ra_dec_offset: list or tupla, default=None
        Relative offset that will be applyied to CRVAL of RA and DEC axis. If
        `ra_dec_val` is privided, this will be ignored. Both elements must be
        instances of :class:`astropy.units.Quantity`.

    Return
    ------
    - correct_wcs: astropy.wcs.WCS
        A copy of the original WCS with the reference values updated.
    """
    correc_wcs = wcs.deepcopy()
    if ra_dec_val is not None:
        if "RA" in correc_wcs.wcs.ctype[0]:
            correc_wcs.wcs.crval[0] = ra_dec_val[0].to_value(
                correc_wcs.wcs.cunit[0])
            correc_wcs.wcs.crval[1] = ra_dec_val[1].to_value(
                correc_wcs.wcs.cunit[1])
        elif "RA" in correc_wcs.wcs.ctype[1]:
            correc_wcs.wcs.crval[0] = ra_dec_val[1].to_value(
                correc_wcs.wcs.cunit[0])
            correc_wcs.wcs.crval[1] = ra_dec_val[0].to_value(
                correc_wcs.wcs.cunit[1])
        else:
            raise NameError(
                "RA coordinate could not be found in the WCS coordinate types:"
                + f"{correc_wcs.wcs.ctype[0]}, {correc_wcs.wcs.ctype[1]}")
    elif ra_dec_offset is not None:
        if "RA" in correc_wcs.wcs.ctype[0]:
            correc_wcs.wcs.crval[0] = correc_wcs.wcs.crval[0] + \
                ra_dec_offset[0].to_value(correc_wcs.wcs.cunit[0])
            correc_wcs.wcs.crval[1] = correc_wcs.wcs.crval[1] + \
                ra_dec_offset[1].to_value(correc_wcs.wcs.cunit[1])
        elif "RA" in correc_wcs.wcs.ctype[1]:
            correc_wcs.wcs.crval[0] = correc_wcs.wcs.crval[0] + \
                ra_dec_offset[1].to_value(correc_wcs.wcs.cunit[0])
            correc_wcs.wcs.crval[1] = correc_wcs.wcs.crval[1] + \
                ra_dec_offset[0].to_value(correc_wcs.wcs.cunit[1])
        else:
            raise NameError(
                "RA coordinate could not be found in the WCS coordinate types:"
                + f"{correc_wcs.wcs.ctype[0]}, {correc_wcs.wcs.ctype[1]}")

    return correc_wcs


# ----------------------------------------------------------------------------------------------------------------------
# Arithmetic operations
# ----------------------------------------------------------------------------------------------------------------------

def med_abs_dev(x, axis=0):
    """Compute the Median Absolute Deviation (MAD) from an input array.
    
    Parameters
    ----------
    x : :class:`np.ndarray`
        Input data.
    axis : int of tupla, optional
        Array axis along with the MAD will be computed
    
    Returns
    -------
    mad : np.ndarray
        Associated MAD to x along the chosen axes.
    """
    mad = np.nanmedian(
        np.abs(x - np.expand_dims(np.nanmedian(x, axis=axis), axis=axis)),
        axis=axis)
    return mad


def std_from_mad(x, axis=0):
    """Estimate the estandard deviation from the MAD.

    Parameters
    ----------
    x : :class:`np.ndarray`
        Input data.
    axis : int of tupla, optional
        Array axis along with the MAD will be computed

    Returns
    -------
    mad : np.ndarray
        Associated MAD to x along the chosen axes.
    
    See also
    --------
    :func:`med_abs_dev`
    """
    return 1.4826 * med_abs_dev(x, axis=axis)

def running_mean(x, n_window):
    """
    This function calculates the running mean of an array.

    Parameters
    ----------
    x : (n,) np.ndarray
        This is the given array.
    n_window : Int
        Number of neighbours for computing the running mean.

    Returns
    -------
    running_mean_x: (n - n_window,) np.ndarray
        Running mean array.
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n_window:] - cumsum[:-n_window]) / n_window

def symmetric_background(x, weights=None, fig_name=None):
    """
    This function calculates the mode of an array,
    assuming that the data can be decomposed as:
    signal plus uniform background plus noise with a symmetric
    probability distribution.

    Parameters
    ----------
    x : np.ndarray
        Data.
    weigths : np.ndarray (None)
        Statistical weight of each point (defaults to None; equal weights).
    fig_name : str (None)
        Name of the figure to be plotted, if desired.

    Returns
    -------
    mode : x.dtype
        Mode of the input array.
    threshold : x.dtype
        Optimal separation between signal and noise background.
    """
    if weights is None:
        weights = np.ones_like(x).astype(float)

    valid = np.where(np.isfinite(x) & np.isfinite(weights))
    n_valid = valid[0].size
    if not n_valid > 2:
        return np.nan, np.nan

    sorted_by_x = np.argsort(x[valid].flatten())
    sorted_x = x[valid].flatten()[sorted_by_x]
    sorted_weight = weights[valid].flatten()[sorted_by_x]

    cumulative_mass = np.cumsum(sorted_weight)
    total_mass = cumulative_mass[-1]
    sorted_weight /= total_mass
    cumulative_mass /= total_mass
    
    
    nbins = int(2 + np.sqrt(n_valid))
    m_left = np.linspace(0, 0.5, nbins)[1:-1]
    m_mid = 2 * m_left
    m_right = 0.5 + m_left

    x_left = np.interp(m_left, cumulative_mass, sorted_x)
    x_mid = np.interp(m_mid, cumulative_mass, sorted_x)
    x_right = np.interp(m_right, cumulative_mass, sorted_x)

    h = np.fmin(x_right-x_mid, x_mid-x_left)
    valid = np.where(h > 0)
    if not valid[0].size > 0:
        return np.nan, np.nan
    x_mid = x_mid[valid]
    h = h[valid]
    rho = (np.interp(x_mid+h, sorted_x, cumulative_mass) - np.interp(x_mid-h, sorted_x, cumulative_mass)) /2/h

    rho_threshold = np.nanpercentile(rho, 100*(1 - 1/np.sqrt(nbins)))
    peak_region = x_mid[rho >= rho_threshold]
    index_min = np.searchsorted(sorted_x, np.min(peak_region))
    index_max = np.searchsorted(sorted_x, np.max(peak_region))
    index_mode = (index_min+index_max) // 2
    mode = sorted_x[index_mode]
    m_mode = cumulative_mass[index_mode]

    rho_bg = np.fmin(rho, np.interp(x_mid, (2*mode - x_mid)[::-1], rho[::-1], left=0, right=0))
    if m_mode <= 0.5:
        total_bg = 2 * m_mode
        threshold = sorted_x[np.clip(int(n_valid * total_bg), 0, n_valid-1)]
        contamination = np.interp(2*mode-threshold, sorted_x, cumulative_mass)
    else:
        total_bg = 2 * (1 - m_mode)
        threshold = sorted_x[np.clip(int(n_valid * (1-total_bg)), 0, n_valid-1)]
        contamination = 1 - np.interp(2*mode-threshold, sorted_x, cumulative_mass)

    if fig_name is not None:
        plt.close(fig_name)
        fig = plt.figure(fig_name, figsize=(8, 5))
        axes = fig.subplots(nrows=1, ncols=1, squeeze=False,
                            sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0},
                           )

        ax = axes[0, 0]
        ax.set_ylabel('probability density')
        #ax.set_yscale('log')
        ax.plot(x_mid, rho-rho_bg, 'b-', alpha=.5, label='p(signal)')
        ax.plot(x_mid, rho_bg, 'r-', alpha=.5, label='p(background)')
        ax.plot(x_mid, rho, 'k-', alpha=.5, label='total')


        ax.set_xlabel('value')
        L = 5 * np.abs(threshold - mode)
        vmin = np.max([mode - L, x_mid[0]])
        vmax = np.min([mode + L, x_mid[-1]])
        ax.set_xlim(vmin, vmax)

        for ax in axes.flatten():
            ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
            ax.tick_params(which='major', direction='inout', length=8, grid_alpha=.3)
            ax.tick_params(which='minor', direction='in', length=2, grid_alpha=.1)
            ax.grid(True, which='both')
            ax.axvspan(sorted_x[index_min], sorted_x[index_max], color='k', alpha=.1)
            ax.axvline(mode, c='k', ls=':', label=f'background = {mode:.4g}')
            ax.axvline(threshold, c='b', ls='-.', label=f'signal threshold = {threshold:.4g}')
            ax.axvline(2*mode - threshold, c='r', ls='-.', alpha=.5, label=f'contamination = {100*contamination/(1-total_bg):.2f}%')
            ax.legend()

        fig.suptitle(fig_name)
        fig.set_tight_layout(True)
        return mode, threshold, fig
        
    return mode, threshold

def parabolic_maximum(x, f):
    """Find the maximum by fitting a parabolic function to three points.
    
    Parameters
    ----------
    x : np.array
        Values where the parabola is evaluated
    y : np.array
        Input function values
    
    Returns
    -------
    x_max
    """
    if x.size != 3:
        raise ValueError("Sice of x and f must be 3")
    f23 = f[1] - f[2]
    f21 = f[1] - f[0]
    x21 = x[1] - x[0]
    x23 = x[1] - x[2]
    x_max = x[1] - 0.5 * (x21**2 * f23 - x23**2 * f21) / (x21 * f23 - x23 * f21)
    return x_max

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


def centre_of_mass(w, x, y):
    """Compute the centre of mass from a distribution of points and weights.

    Parameters
    ----------
    x: np.ndarray
        Coordinates corresponding to the x-axis.
    y: np.ndarray
        Coordinates corresponding to the y-axis.
    w: np.ndarray
        Weights for computing the centre of mass.

    Returns
    -------
    center_of_mass : tupla
        Center of mass expressed as ``(x_com, y_com)``
    """
    norm = np.nansum(w)
    x_com, y_com = np.nansum(w * x) / norm, np.nansum(w * y) / norm
    if np.isfinite(x_com) and np.isfinite(y_com):
        return x_com, y_com
    else:
        raise RuntimeError(
            "Failed computing centre of mass computed for\n w={}\n x={}\n y={}"
            .format(w, x, y))

# TODO: Stale method
def growth_curve_1d(f, x, y):
    """TODO"""
    r2 = x**2 + y**2
    idx_sorted = np.argsort(r2)
    growth_c = np.nancumsum(f[idx_sorted])
    return r2[idx_sorted], growth_c

# TODO: Stale method
def growth_curve_2d(image, x0=None, y0=None):
    """Compute the curve of growth of an array f with respect to a given point (x0, y0).

    Parameters
    ----------
    image: np.ndarray
        2D array
    x0: float
        Origin of coordinates in pixels along the x-axis (columns).
    y0: float
        Origin of coordinates in pixels along the y-axis (rows).

    Returns
    -------
    r2: np.ndarray
        Vector containing the square radius with respect to (x0, y0).
    growth_curve: np.ndarray
        Curve of growth centered at (x0, y0).
    """
    xx, yy = np.meshgrid(np.arange(0, image.shape[1]),
                         np.arange(0, image.shape[1]))
    r2 = (xx - x0) ** 2 + (yy - y0) ** 2
    idx_sorted = np.argsort(r2)
    growth_c = np.cumsum(image.flatten()[idx_sorted])
    return r2[idx_sorted], growth_c

@preserve_units_dec
def interpolate_image_nonfinite(image):
    """Use :class:`scipy.interpolate.NearestNDInterpolator` to replace NaN values.

    Parameters
    ----------
    - image: :class:`np.ndarray`
        2D array to be interpolated
    Returnrs
    --------
    - interpolated_image: :class:`np.ndarray`
        Image with nan values replaced by their nearest-neightbour values.
    """
    if image.ndim != 2:
        raise ArithmeticError(f"Input image must have 2D not {image.ndim}")

    x, y = np.meshgrid(
        np.arange(0, image.shape[1], 1),
        np.arange(0, image.shape[0], 1))
    mask = np.isfinite(image)
    if not mask.any():
        raise ArithmeticError("All values of input image are non-finite")
    interp = interpolate.NearestNDInterpolator(
        list(zip(x[mask], y[mask])), image[mask].value)
    interp_image = interp(x, y)
    return interp_image

def vac_to_air(vac_wl: u.Quantity):
    """Convert wavelength in vacuum to air using Morton (1991, ApJS, 77, 119).
    
    Parameters
    ----------
    - vac_wl: :class:`astropy.units.Quantity`
        Vector of vacuum wavelengths in Angstrom.
    
    Returns
    -------
    - air_wl: :class:`astropy.units.Quantity`
        Vector of air wavelengths in Angstrom
    """
    sigma = 1 / vac_wl.to_value("micron")
    vac_over_air = (1 + 8.0605e-5 + 2.48099e-2 / (132.274 - sigma**2)
                    + 1.74557e-4 / (39.32957 - sigma**2)
                    ) << u.dimensionless_unscaled
    return vac_wl / vac_over_air

# TODO: stale
def smooth_spectrum(wlm, s, wave_min=0, wave_max=0, step=50, exclude_wlm=[[0, 0]], order=7,
                    weight_fit_median=0.5, plot=False, verbose=False, fig_size=12):
    """
    THIS IS NOT EXACTLY THE SAME THING THAT applying signal.medfilter()

    This needs to be checked, updated, and combine (if needed) with task fit_smooth_spectrum.
    The task gets the median value in steps of "step", gets an interpolated spectrum, 
    and fits a 7-order polynomy.

    It returns fit_median + fit_median_interpolated (each multiplied by their weights).

    Tasks that use this:  get_telluric_correction
    """

    if verbose:
        vprint("\n> Computing smooth spectrum...")

    if wave_min == 0:
        wave_min = wlm[0]
    if wave_max == 0:
        wave_max = wlm[-1]

    running_wave = []
    running_step_median = []
    cuts = np.int((wave_max - wave_min) / step)

    exclude = 0
    corte_index = -1
    for corte in range(cuts+1):
        next_wave = wave_min+step*corte
        if next_wave < wave_max:
            if next_wave > exclude_wlm[exclude][0] and next_wave < exclude_wlm[exclude][1]:
                if verbose:
                    vprint(f"  Skipping {next_wave}"
                           + f" as it is in the exclusion range [{exclude_wlm[exclude][0]}, {exclude_wlm[exclude][1]}]")

            else:
                corte_index = corte_index+1
                running_wave.append(next_wave)
                region = np.where(
                    (wlm > running_wave[corte_index]-step/2) & (wlm < running_wave[corte_index]+step/2))
                running_step_median.append(np.nanmedian(s[region]))
                if next_wave > exclude_wlm[exclude][1]:
                    exclude = exclude + 1
                    # if verbose and exclude_wlm[0] != [0,0] : print "--- End exclusion range ",exclude
                    if exclude == len(exclude_wlm):
                        exclude = len(exclude_wlm)-1

    running_wave.append(wave_max)
    region = np.where((wlm > wave_max-step) & (wlm < wave_max+0.1))
    running_step_median.append(np.nanmedian(s[region]))

    # Check not nan
    _running_wave_ = []
    _running_step_median_ = []
    for i in range(len(running_wave)):
        if np.isnan(running_step_median[i]):
            if verbose:
                vprint(f"There is a nan in {running_wave[i]}")
        else:
            _running_wave_.append(running_wave[i])
            _running_step_median_.append(running_step_median[i])

    fit = np.polyfit(_running_wave_, _running_step_median_, order)
    pfit = np.poly1d(fit)
    fit_median = pfit(wlm)

    interpolated_continuum_smooth = interpolate.splrep(
        _running_wave_, _running_step_median_, s=0.02)
    fit_median_interpolated = interpolate.splev(
        wlm, interpolated_continuum_smooth, der=0)

    if plot:
        plt.figure(figsize=(fig_size, fig_size/2.5))
        plt.plot(wlm, s, alpha=0.5)
        plt.plot(running_wave, running_step_median, "+", ms=15, mew=3)
        plt.plot(wlm, fit_median, label="fit median")
        plt.plot(wlm, fit_median_interpolated, label="fit median_interp")
        plt.plot(wlm, weight_fit_median*fit_median + (1-weight_fit_median)
                 * fit_median_interpolated, label="weighted")
        # extra_display = (np.nanmax(fit_median)-np.nanmin(fit_median)) / 10
        # plt.ylim(np.nanmin(fit_median)-extra_display, np.nanmax(fit_median)+extra_display)
        ymin = np.nanpercentile(s, 1)
        ymax = np.nanpercentile(s, 99)
        rango = (ymax-ymin)
        ymin = ymin - rango/10.
        ymax = ymax + rango/10.
        plt.ylim(ymin, ymax)
        plt.xlim(wlm[0]-10, wlm[-1]+10)
        plt.minorticks_on()
        plt.legend(frameon=False, loc=1, ncol=1)

        plt.axvline(x=wave_min, color='k', linestyle='--')
        plt.axvline(x=wave_max, color='k', linestyle='--')

        plt.xlabel(r"Wavelength [$\mathrm{\AA}$]")

        if exclude_wlm[0][0] != 0:
            for i in range(len(exclude_wlm)):
                plt.axvspan(exclude_wlm[i][0],
                            exclude_wlm[i][1], color='r', alpha=0.1)
        plt.close()
        vprint(f"Weights for getting smooth spectrum:\n fit_median ={weight_fit_median}"
               + f"\n Fit_median_interpolated = {(1-weight_fit_median)}")

    # (fit_median+fit_median_interpolated)/2      # Decide if fit_median or fit_median_interpolated
    return weight_fit_median*fit_median + (1-weight_fit_median)*fit_median_interpolated

def pixel_in_square(pixel_pos, pixel_size, pos, radius):
    """Compute the area of a pixel within a square.

    Parameters
    ----------
    - pixel_pos: tuple
        Position of the lower left corner of the pixel
    - pixel_size: float
        Size of the pixel.
    - pos: tuple
        Position of the square centre
    - radius: float
        Half size of the square.

    Returns
    -------
    - area_pixel:
        Area of the pixel contained within the square
    - area_fraction:
        Fration of the square area that overlaps with the pixel.
    """
    square = geometry.box(pos[0] - radius,
                          pos[1] - radius,
                          pos[0] + radius,
                          pos[1] + radius)
    rectangle = geometry.box(pixel_pos[0], pixel_pos[1],
                             pixel_pos[0] + pixel_size,
                             pixel_pos[1] + pixel_size)    
    intersection = square.intersection(rectangle)
    return intersection.area, intersection.area / square.area

def pixel_in_circle(pixel_pos, pixel_size, pos, radius):
    """Compute the area of a pixel within a circle.

    Parameters
    ----------
    - pixel_pos: tuple
        Position of the lower left corner of the pixel
    - pixel_size: float
        Size of the pixel.
    - pos: tuple
        Position of the circle centre
    - radius: float
        Radius of the circle.

    Returns
    -------
    - area_pixel:
        Area of the pixel contained within the circle
    - area_fraction:
        Fration of the circle area that overlaps with the pixel.
    """
    circle = geometry.Point(*pos).buffer(radius)
    rectangle = geometry.box(pixel_pos[0], pixel_pos[1],
                             pixel_pos[0] + pixel_size,
                             pixel_pos[1] + pixel_size)    
    intersection = circle.intersection(rectangle)
    return intersection.area, intersection.area / circle.area

def pixel_in_hexagon(pixel_pos, pixel_size, pos, radius):
    """Copmute the area of a pixel overlapping with a regular hexagon.
    
    Description
    -----------
    The hexagon is assumed to be regular.

    Parameters
    ----------
    - pixel_pos: tuple
        Position of the lower left corner of the pixel
    - pixel_size: float
        Size of the pixel.
    - pos: tuple
        Position of the hexagon centre.
    - radius: float
        Radius of the circle that inscribes the hexagon. In other words, the
        distance of the hexagon vertices to the centre.

    Returns
    -------
    - area_pixel:
        Area of the pixel contained within the circle
    - area_fraction:
        Fration of the circle area that overlaps with the pixel.
    """
    # cos(30) * rad / sin(30) * rad
    rad_cos_30 = 0.866 * radius
    rad_sin_30 = 0.5
    # Coordinates are provided in anticlock-wise order starting from the top
    hexagon = geometry.Polygon([
        (pos[0], pos[1] + radius),
        (pos[0] - rad_cos_30, pos[1] + rad_sin_30),
        (pos[0] - rad_cos_30, pos[1] - rad_sin_30),
        (pos[0], pos[1] - radius),
        (pos[0] + rad_cos_30, pos[1] - rad_sin_30),
        (pos[0] + rad_cos_30, pos[1] + rad_sin_30),
        (pos[0], pos[1] + radius)])
    rectangle = geometry.box(pixel_pos[0], pixel_pos[1],
                             pixel_pos[0] + pixel_size,
                             pixel_pos[1] + pixel_size)    
    intersection = hexagon.intersection(rectangle)
    return intersection.area, intersection.area / hexagon.area

# ----------------------------------------------------------------------------------------------------------------------
# Models and fitting
# ----------------------------------------------------------------------------------------------------------------------

def cumulative_1d_moffat(r2, l_star=1.0, alpha2=1.0, beta=1.0):
    """
    Cumulative Moffat ligth profile.
    Parameters
    ----------
    r2 : np.array(float)
        Square radius with respect to the profile centre.
    l_star : float
        Total luminosity integrating from 0 to inf.
    alpha2 : float
        Characteristic square radius.
    beta : float
        Power-low slope
    Returns
    -------
    cum_moffat_prof: np.array(float)
        Cumulative Moffat profile
    """
    return l_star * (1 - np.power(1 + (r2 / alpha2), -beta))

# =============================================================================
# Lines
# =============================================================================
# TODO : merge/remove with future "spectra" module

lines = {
    # Balmer
    'hepsilon': 3970.1,
    'hdelta': 4101.7,
    'hgamma': 4340.4,
    'hbeta': 4861.3,
    'halpha': 6562.79,
    # K
    'K': 3934.777,
    'H': 3969.588,
    'Mg': 5176.7,
    'Na': 5895.6,
    'CaII1': 8500.36,
    'CaII2': 8544.44,
    'CaII3': 8664.52,
}


def mask_lines(wave_array, width=30, lines=lines.values()):
    wave_array = check_unit(wave_array, u.AA)
    width = check_unit(width, wave_array.unit)
    if width.size == 1:
        width = width * np.ones(len(lines))
    mask = np.ones(wave_array.size, dtype=bool)
    for line, w in zip(lines, width):
        line = check_unit(line, u.AA)
        mask[(wave_array < line + w) & (wave_array > line - w)] = False
    return mask

def mask_telluric_lines(wave_array, width_clip=3 << u.AA):
    model_file = os.path.join(os.path.dirname(__file__),
                                      'input_data', 'sky_lines',
                                      'telluric_lines.txt')
    w_l_1, w_l_2 = np.loadtxt(model_file, unpack=True, usecols=(0, 1))
    w_l_1 = w_l_1 << u.angstrom
    w_l_2 = w_l_2 << u.angstrom
    return mask_lines(wave_array, width= np.clip(w_l_2 - w_l_1, a_min=width_clip, a_max=None),
                      lines=(w_l_1 + w_l_2) / 2)

# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
