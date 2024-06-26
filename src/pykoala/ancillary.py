"""
Utility functions.

Many of them are partially implemented. They were in the
old, non-modular version of PyKOALA and have not been included
in the current modular scheme.
"""

# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy import optimize
import logging

# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits
# =============================================================================

# =============================================================================
# PyKOALA modules
# =============================================================================

logger = logging.getLogger('pykoala.logger')

if not (logger.hasHandlers()):
    stdout = logging.StreamHandler()
    fmt = logging.Formatter(
    "[PyKOALA] %(asctime)s | %(levelname)s > %(message)s")   
    stdout.setFormatter(fmt)
    logger.addHandler(stdout)
    logger.setLevel(logging.INFO)

def log_into_file(filename, level='INFO'):
    logger = logging.getLogger('pykoala.logger')
    hdlr = logging.FileHandler(filename)
    fmt = logging.Formatter(
    "[PyKOALA] %(asctime)s | %(levelname)s > %(message)s")   
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr) 
    logger.setLevel(level)



def vprint(*arg, **kwargs):
    """
    Prints the arguments only if verbose=True.
    """
    if 'verbose' in kwargs:
        print(*arg)

# =============================================================================
# Ancillary Functions - RSS Related
# =============================================================================

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
        New CRVAL of RA and DEC.
    - ra_dec_offset: list or tupla, default=None
        Relative offset that will be applyied to CRVAL of RA and DEC axis. If
        `ra_dec_val` is privided, this will be ignored.

    Return
    ------
    - correct_wcs: astropy.wcs.WCS
        A copy of the original WCS with the reference values updated.
    """
    correc_wcs = wcs.deepcopy()
    if ra_dec_val is not None:
        if "RA" in correc_wcs.wcs.ctype[0]:
            correc_wcs.wcs.crval[0] = ra_dec_val[0]
            correc_wcs.wcs.crval[1] = ra_dec_val[1]
        elif "RA" in correc_wcs.wcs.ctype[1]:
            correc_wcs.wcs.crval[0] = ra_dec_val[1]
            correc_wcs.wcs.crval[1] = ra_dec_val[0]
        else:
            raise NameError(
                "RA coordinate could not be found in the WCS coordinate types:"
                + f"{correc_wcs.wcs.ctype[0]}, {correc_wcs.wcs.ctype[1]}")
    elif ra_dec_offset is not None:
        if "RA" in correc_wcs.wcs.ctype[0]:
            correc_wcs.wcs.crval[0] = correc_wcs.wcs.crval[0] + \
                ra_dec_offset[0]
            correc_wcs.wcs.crval[1] = correc_wcs.wcs.crval[1] + \
                ra_dec_offset[1]
        elif "RA" in correc_wcs.wcs.ctype[1]:
            correc_wcs.wcs.crval[0] = correc_wcs.wcs.crval[0] + \
                ra_dec_offset[1]
            correc_wcs.wcs.crval[1] = correc_wcs.wcs.crval[1] + \
                ra_dec_offset[0]
        else:
            raise NameError(
                "RA coordinate could not be found in the WCS coordinate types:"
                + f"{correc_wcs.wcs.ctype[0]}, {correc_wcs.wcs.ctype[1]}")

    return correc_wcs


# ----------------------------------------------------------------------------------------------------------------------
# Arithmetic operations
# ----------------------------------------------------------------------------------------------------------------------
def med_abs_dev(x, axis=0):
    mad = np.nanmedian(
        np.abs(x - np.expand_dims(np.nanmedian(x, axis=axis), axis=axis)),
        axis=axis)
    return mad


def std_from_mad(x, axis=0, k=1.4826):
    mad = med_abs_dev(x, axis=axis)
    return k * mad


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


def flux_conserving_interpolation(new_wavelength, wavelength, spectra, **interp_args):
    """Flux-conserving linear interpolation.

    Linear interpolation of a spectrum :math:`I_\lambda(\labmda)`
    as a function of wavelength :math:`\lambda`,
    ensuring that the integrated flux
    math::
    F(\lambda_a, \lambda_b)
    = \int_{\lambda_a}^{\lambda_b} I_\lambda(\labmda) d\lambda$
        
    is conserved for any :math:`(\lambda_a, \lambda_b)`.
    
    `np.nan` values become zero.

    Parameters
    ----------
    new_wavelength : ndarray
        New values of the x coordinate (wavelength) :math:`\lambda_{new}`.
    wavelength : ndarray
        Old values of the x coordinate (wavelength) :math:`\lambda`.
    spectra : ndarray
        Old values of the y coordinate (spectrum) :math:`I_\lambda(\labmda)`.
    **interp_args : dict, optional
        Additional parameters to be passed to `np.interp`

    Returns
    -------
    ndarray
        Interpolated spectra :math:`I_\lambda(\labmda_{new})`.
    
    Notes
    -----
    The function computes the cumulative flux with `np.nancumsum`,
    calls `np.interp`, and differentiates back.
    """
    dwave = wavelength[1:] - wavelength[:-1]
    wavelength_edges = np.hstack((wavelength[0] - dwave[0] / 2, wavelength[:-1] + dwave / 2,
                                  wavelength[-1] + dwave[-1] / 2))
    new_dwave = new_wavelength[1:] - new_wavelength[:-1]
    new_wavelength_edges = np.hstack((new_wavelength[0] - new_dwave[0] / 2, new_wavelength[:-1] + new_dwave / 2,
                                      new_wavelength[-1] + new_dwave[-1] / 2))
    cum_spectra = np.nancumsum(np.diff(wavelength_edges) * spectra)
    cum_spectra = np.hstack((0, cum_spectra))
    new_cum_spectra = np.interp(
        new_wavelength_edges, wavelength_edges, cum_spectra, **interp_args)
    new_spectra = np.diff(new_cum_spectra) / np.diff(new_wavelength_edges)
    return new_spectra


def centre_of_mass(w, x, y):
    """Compute the centre of mass of a given image.
    Parameters
    ----------
    w: np.ndarray(float)
        (n,) weights computing the centre of mass.
    x: np.ndarray(float)
        (n,) Coordinates corresponding to the x-axis (columns).
    y: np.ndarray(float)
        (n,) Coordinates corresponding to the y-axis (rows).
    Returns
    -------
    x_com: float
    y_com: float
    """
    norm = np.nansum(w)
    x_com, y_com = np.nansum(w * x) / norm, np.nansum(w * y) / norm
    if np.isfinite(x_com) and np.isfinite(y_com):
        return x_com, y_com
    else:
        raise RuntimeError(
            "Failed computing centre of mass computed for\n w={}\n x={}\n y={}"
            .format(w, x, y))


def growth_curve_1d(f, x, y):
    """TODO"""
    r2 = x**2 + y**2
    idx_sorted = np.argsort(r2)
    growth_c = np.nancumsum(f[idx_sorted])
    return r2[idx_sorted], growth_c


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


def interpolate_image_nonfinite(image):
    """Use scipy.interpolate.NearestNDInterpolator to replace NaN values.

    Parameters
    ----------
    - image: (np.ndarray)
        2D array to be interpolated
    Returnrs
    --------
    - interpolated_image: (np.ndarray)
    """
    if image.ndim != 2:
        raise ArithmeticError(f"Input image must have 2D not {image.ndim}")

    x, y = np.meshgrid(
        np.arange(0, image.shape[1], 1),
        np.arange(0, image.shape[0], 1))
    mask = np.isfinite(image)
    interp = interpolate.NearestNDInterpolator(
        list(zip(x[mask], y[mask])), image[mask])
    interp_image = interp(x, y)
    return interp_image


# TODO: refactor
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
        print("\n> Computing smooth spectrum...")

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
                    print("  Skipping ", next_wave,
                          " as it is in the exclusion range [", exclude_wlm[exclude][0], ",", exclude_wlm[exclude][1], "]")

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
                print("  There is a nan in ", running_wave[i])
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

        plt.xlabel("Wavelength [$\mathrm{\AA}$]")

        if exclude_wlm[0][0] != 0:
            for i in range(len(exclude_wlm)):
                plt.axvspan(exclude_wlm[i][0],
                            exclude_wlm[i][1], color='r', alpha=0.1)
        plt.show()
        plt.close()
        print('  Weights for getting smooth spectrum:  fit_median =',
              weight_fit_median, '    fit_median_interpolated =', (1-weight_fit_median))

    # (fit_median+fit_median_interpolated)/2      # Decide if fit_median or fit_median_interpolated
    return weight_fit_median*fit_median + (1-weight_fit_median)*fit_median_interpolated


def vect_norm(a, b):
    """Compute the norm of two vectors."""
    return np.sqrt(np.sum((a - b)**2, axis=-1))


def in_rectangle(pos, rectangle_pos):
    """Check if a point lies in the perimeter of a rectangle."""
    in_rectangle = (
        (pos[0] <= rectangle_pos[:, 0]).any()
        & (pos[0] >= rectangle_pos[:, 0]).any()
        & (pos[1] <= rectangle_pos[:, 1]).any()
        & (pos[1] >= rectangle_pos[:, 1]).any()
    )
    return in_rectangle


def pixel_in_circle(pixel_pos, pixel_size, circle_pos, circle_radius):
    """Compute the area of a pixel within a circle.

    Parameters
    ----------
    - pixel_pos: tuple
        Position of the lower left corner of the pixel
    - pixel_size: float
        Size of the pixel.
    - circle_pos: tuple
        Position of the circle centre
    - circle_raidus: float
        Radius of the circle.

    Returns
    -------
    - area_pixel:
        Area of the pixel contained within the circle
    - area_fraction:
        Fration of the circle area that overlaps with the pixel.
    """
    pixel_vertices = (np.atleast_1d(pixel_pos)[np.newaxis, :]
                      + pixel_size * np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                      )
    # Find those pixels inside the circle
    inside = vect_norm(pixel_vertices, np.array(circle_pos)[np.newaxis]
                       ) < circle_radius
    n_inside = np.count_nonzero(inside)
    if n_inside == 0:
        return 0, 0

    pixel_center = np.array([pixel_pos[0] + pixel_size / 2,
                             pixel_pos[1] + pixel_size / 2])
    pixel_area = pixel_size**2
    vertex_in = pixel_vertices[inside].squeeze()
    vertex_out = pixel_vertices[~inside].squeeze()
    cr_s = circle_radius**2
    circle_area = np.pi * cr_s

    if n_inside == 1:
        x_cross = (cr_s - (vertex_in[1] - circle_pos[1])**2)**0.5
        x_cross = np.array([[circle_pos[0] + x_cross, vertex_in[1]],
                            [circle_pos[0] - x_cross, vertex_in[1]]])
        xcross_pt = np.argmin(vect_norm(pixel_center[np.newaxis, :], x_cross))
        x_cross = x_cross[xcross_pt]

        y_cross = (cr_s - (vertex_in[0] - circle_pos[0])**2)**0.5
        y_cross = np.array([[vertex_in[0], circle_pos[1] + y_cross],
                            [vertex_in[0], circle_pos[1] - y_cross]])
        ycross_pt = np.argmin(vect_norm(pixel_center[np.newaxis, :], y_cross))
        y_cross = y_cross[ycross_pt]
        arc_length = vect_norm(x_cross, y_cross)
        phi = np.arccos(
            (2*circle_radius**2 - arc_length**2) / (2 * circle_radius**2))

        # Area of the triangle formed by the circle center and the intersection
        area_triangle = 0.5 * circle_radius**2 * np.sin(phi)
        area_sector = phi / 2 * circle_radius**2
        # Area of the triangle within the pixel surface inside the circle
        area_pixel_triangle = 0.5 * np.sqrt(
            (vertex_in[0] - x_cross[0])**2 * (vertex_in[1] - y_cross[1])**2)
        area_pixel = area_sector - area_triangle + area_pixel_triangle

    elif n_inside == 2:
        parallel_axis = np.where(vertex_in[0] == vertex_in[1])[0][0]
        perpendicular_axis = [0, 1]
        perpendicular_axis.remove(parallel_axis)
        dist = np.abs(vertex_in[0][perpendicular_axis
                                   ] - vertex_in[1][perpendicular_axis]).squeeze()
        intersection_points = np.zeros((2, 2))
        for i, v in enumerate(vertex_in):
            x_cross = (cr_s - (v[1] - circle_pos[1])**2)**0.5
            x_cross = np.array([[circle_pos[0] + x_cross, v[1]],
                                [circle_pos[0] - x_cross, v[1]]])
            xcross_pt = np.argmin(
                vect_norm(pixel_center[np.newaxis, :], x_cross))
            x_cross = x_cross[xcross_pt]
            if in_rectangle(x_cross, pixel_vertices):
                intersection_points[i] = x_cross
                continue
            y_cross = (cr_s - (v[0] - circle_pos[0])**2)**0.5
            y_cross = np.array([[v[0], circle_pos[1] + y_cross],
                                [v[0], circle_pos[1] - y_cross]])
            ycross_pt = np.argmin(
                vect_norm(pixel_center[np.newaxis, :], y_cross))
            y_cross = y_cross[ycross_pt]
            if in_rectangle(y_cross, pixel_vertices):
                intersection_points[i] = y_cross

        # Parallel sides of the trapezium
        a = np.abs(
            vertex_in[0][parallel_axis]
            - intersection_points[0][parallel_axis])
        b = np.abs(
            vertex_in[1][parallel_axis]
            - intersection_points[1][parallel_axis])

        area_trapezoid = dist * (a + b) / 2
        arc_length = vect_norm(intersection_points[0], intersection_points[1])
        phi = np.arccos(
            (2*circle_radius**2 - arc_length**2) / (2 * circle_radius**2))
        area_triangle = 0.5 * cr_s * np.sin(phi)
        area_sector = phi / 2 * cr_s
        area_pixel = area_sector - area_triangle + area_trapezoid

    elif n_inside == 3:
        dist_to_center = vect_norm(vertex_in, np.array(circle_pos)[np.newaxis])
        corner_pos = np.argmin(dist_to_center)
        intersection_points = np.zeros((2, 2))
        for i, v in enumerate(np.delete(vertex_in, corner_pos, axis=0)):
            x_cross = (cr_s - (v[1] - circle_pos[1])**2)**0.5
            x_cross = np.array([[circle_pos[0] + x_cross, v[1]],
                                [circle_pos[0] - x_cross, v[1]]])
            xcross_pt = np.argmin(
                vect_norm(pixel_center[np.newaxis, :], x_cross))
            x_cross = x_cross[xcross_pt]
            if in_rectangle(x_cross, pixel_vertices):
                intersection_points[i] = x_cross
                continue

            y_cross = (cr_s - (v[0] - circle_pos[0])**2)**0.5
            y_cross = np.array([[v[0], circle_pos[1] + y_cross],
                                [v[0], circle_pos[1] - y_cross]])
            ycross_pt = np.argmin(
                vect_norm(pixel_center[np.newaxis, :], y_cross))
            y_cross = y_cross[ycross_pt]
            if in_rectangle(y_cross, pixel_vertices):
                intersection_points[i] = y_cross

        arc_length = vect_norm(intersection_points[0], intersection_points[1])
        phi = np.arccos(
            (2*circle_radius**2 - arc_length**2) / (2 * circle_radius**2))

        area_inner_triangle = 0.5 * cr_s * np.sin(phi)
        area_sector = phi / 2 * cr_s

        area_outer_triangle = 0.5 * (
            vect_norm(intersection_points[0], vertex_out)
            * vect_norm(intersection_points[1], vertex_out))
        area_pixel = (pixel_area - area_outer_triangle
                      + (area_sector - area_inner_triangle))
    else:
        area_pixel = pixel_size**2
    area_fraction = area_pixel / (circle_area + 1e-100)
    return area_pixel, area_fraction
# ----------------------------------------------------------------------------------------------------------------------
# Models and fitting
# ----------------------------------------------------------------------------------------------------------------------


def cumulative_1d_sky(r2, sky_brightness):
    """
    1D cumulative sky brightness. F_sky = 4*pi*r2 * B_sky
    
    Parameters
    ----------
    r2 : np.array(float)
        Square radius from origin.
    sky_brightness : float
        Sky surface brightness.

    Returns
    -------
    cumulative_sky_brightness : np.array(float)
        Cumulative sky brightness.
    """
    return np.pi * r2 * sky_brightness


def cumulative_1d_moffat(r2, l_star, alpha2, beta):
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


def cumulative_1d_moffat_sky(r2, l_star, alpha2, beta, sky_brightness):
    """Combined model of cumulative_1d_moffat and cumulative_1d_sky."""
    return cumulative_1d_sky(r2, sky_brightness) + cumulative_1d_moffat(r2, l_star, alpha2, beta)


def fit_moffat(r2_growth_curve, f_growth_curve,
               f_guess, r2_half_light, r_max, plot=False):
    """
    Fits a Moffat profile to a flux growth curve
    as a function of radius squared,
    cutting at to r_max (in units of the half-light radius),
    provided an initial guess of the total flux and half-light radius squared.

    # TODO
    Parameters
    ----------
    r2_growth_curve : TYPE
        DESCRIPTION.
    F_growth_curve : TYPE
        DESCRIPTION.
    F_guess : TYPE
        DESCRIPTION.
    r2_half_light : TYPE
        DESCRIPTION.
    r_max : TYPE
        DESCRIPTION.
    plot : Boolean, optional
        If True generates and shows the plots. The default is False.

    Returns
    -------
    fit : TYPE
        DESCRIPTION.
    """
    index_cut = np.searchsorted(r2_growth_curve, r2_half_light * r_max ** 2)
    fit, cov = optimize.curve_fit(cumulative_1d_moffat,
                                  r2_growth_curve[:index_cut], f_growth_curve[:index_cut],
                                  p0=(f_guess, r2_half_light, 1)
                                  )
    if plot:
        print("Best-fit: L_star =", fit[0])
        print("          alpha =", np.sqrt(fit[1]))
        print("          beta =", fit[2])
        r_norm = np.sqrt(np.array(r2_growth_curve) / r2_half_light)
        plt.plot(r_norm, cumulative_1d_moffat(np.array(r2_growth_curve),
                                              fit[0], fit[1], fit[2]) / fit[0], ':')
    return fit


def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    exponent = -0.5 * (((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2)
    return amplitude * np.exp(exponent) + offset

# =============================================================================
# Lines
# =============================================================================


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
    mask = np.ones_like(wave_array, dtype=bool)
    for line in lines:
        mask[(wave_array < line + width) & (wave_array > line - width)] = False
    return mask

# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
