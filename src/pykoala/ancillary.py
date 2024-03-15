"""
This module contains a collection of ancilliary functions which are either partially implemented or not fully implemented. 
Currently this acts as a placeholder for functions that were in the original non-modular version of PyKOALA which have
not been included in the modular version
"""

# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy import optimize
# =============================================================================
# Astropy and associated packages
# =============================================================================
#from astropy.io import fits
# =============================================================================
# =============================================================================
# PyKOALA modules
# =============================================================================
from pykoala.cubing import Cube, build_wcs, build_cube
# =============================================================================
# =============================================================================
# =============================================================================
# ANGEL modules                                                            #!!!
# =============================================================================
# =============================================================================
# =============================================================================
def vprint(*arg, **kwargs):
    """
    Prints the arguments only if verbose=True.
    """
    try:
        if kwargs['verbose']:
            print(*arg)
    except Exception:
        do_nothing = 0                    #!!! Ángel: Una ayudita aquí... si no quiero que haga nada si falla, ¿qué pongo?
# =============================================================================
# =============================================================================
def vplot(*arg, **kwargs):
    """
    Check kwarguments and return True if plot=True.
    """
    try:
        if kwargs['plot']:
            return True
        else:
            return False
    except Exception:
        return False
# =============================================================================
# =============================================================================  
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
def find_index_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
# =============================================================================
# =============================================================================



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


# RSS info dictionary template
rss_info_template = dict(name=None,  # Name of the object
                         exptime=None,  # Total rss exposure time (seconds)
                         fib_ra=None, fib_dec=None,  # Fibres' celestial offset
                         airmass=None  # Airmass
                         )

def make_dummy_cube_from_rss(rss, spa_pix_arcsec=0.5, kernel_pix_arcsec=1.0):
    """Create an empty datacube array from an input RSS."""
    min_ra, max_ra = np.nanmin(rss.info['fib_ra']), np.nanmax(rss.info['fib_ra'])
    min_dec, max_dec = np.nanmin(rss.info['fib_dec']), np.nanmax(rss.info['fib_dec'])
    datacube_shape = (rss.wavelength.size,
                   int((max_ra - min_ra) * 3600 / spa_pix_arcsec),
                   int((max_dec - min_dec) * 3600 / spa_pix_arcsec))
    ref_position = (rss.wavelength[0], (min_ra + max_ra) / 2, (min_dec + max_dec) / 2)
    spatial_pixel_size = spa_pix_arcsec / 3600
    spectral_pixel_size = rss.wavelength[1] - rss.wavelength[0]

    wcs = build_wcs(datacube_shape=datacube_shape,
                    reference_position=ref_position,
                    spatial_pix_size=spatial_pixel_size,
                    spectra_pix_size=spectral_pixel_size,
                )
    cube = build_cube([rss], pixel_size_arcsec=spa_pix_arcsec, wcs=wcs,
                      kernel_size_arcsec=kernel_pix_arcsec)
    return cube

# ----------------------------------------------------------------------------------------------------------------------
# Arithmetic operations
# ----------------------------------------------------------------------------------------------------------------------
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
    """
    #TODO...
    Parameters
    ----------
    new_wavelength
    wavelength
    spectra
    interp_args

    Returns
    -------

    """
    dwave = wavelength[1:] - wavelength[:-1]
    wavelength_edges = np.hstack((wavelength[0] - dwave[0] / 2, wavelength[:-1] + dwave / 2,
                                  wavelength[-1] + dwave[-1] / 2))
    new_dwave = new_wavelength[1:] - new_wavelength[:-1]
    new_wavelength_edges = np.hstack((new_wavelength[0] - new_dwave[0] / 2, new_wavelength[:-1] + new_dwave / 2,
                                      new_wavelength[-1] + new_dwave[-1] / 2))
    cum_spectra = np.nancumsum(np.diff(wavelength_edges) * spectra)
    cum_spectra = np.hstack((0, cum_spectra))
    new_cum_spectra = np.interp(new_wavelength_edges, wavelength_edges, cum_spectra, **interp_args)
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


def make_white_image_from_array(data_array, wavelength=None, **args):
    """Create a white image from a 3D data array."""
    print(f"Creating a Cube of dimensions: {data_array.shape}")
    cube = Cube(intensity=data_array, wavelength=wavelength)
    return cube.get_white_image()

# TODO: refactor
def smooth_spectrum(wlm, s, wave_min=0, wave_max=0, step=50, exclude_wlm=[[0,0]], order=7,    
                    weight_fit_median=0.5, plot=False, verbose=False, fig_size=12): 
    """
    THIS IS NOT EXACTLY THE SAME THING THAT applying signal.medfilter()
    
    This needs to be checked, updated, and combine (if needed) with task fit_smooth_spectrum.
    The task gets the median value in steps of "step", gets an interpolated spectrum, 
    and fits a 7-order polynomy.
    
    It returns fit_median + fit_median_interpolated (each multiplied by their weights).
    
    Tasks that use this:  get_telluric_correction
    """

    if verbose: print("\n> Computing smooth spectrum...")

    if wave_min == 0 : wave_min = wlm[0]
    if wave_max == 0 : wave_max = wlm[-1]
        
    running_wave = []    
    running_step_median = []
    cuts=np.int( (wave_max - wave_min) /step)
   
    exclude = 0 
    corte_index=-1
    for corte in range(cuts+1):
        next_wave= wave_min+step*corte
        if next_wave < wave_max:
            if next_wave > exclude_wlm[exclude][0] and next_wave < exclude_wlm[exclude][1]:
               if verbose: print("  Skipping ",next_wave, " as it is in the exclusion range [",exclude_wlm[exclude][0],",",exclude_wlm[exclude][1],"]")    

            else:
                corte_index=corte_index+1
                running_wave.append (next_wave)
                region = np.where((wlm > running_wave[corte_index]-step/2) & (wlm < running_wave[corte_index]+step/2))              
                running_step_median.append (np.nanmedian(s[region]) )
                if next_wave > exclude_wlm[exclude][1]:
                    exclude = exclude + 1
                    #if verbose and exclude_wlm[0] != [0,0] : print "--- End exclusion range ",exclude 
                    if exclude == len(exclude_wlm) :  exclude = len(exclude_wlm)-1  
                        
    running_wave.append (wave_max)
    region = np.where((wlm > wave_max-step) & (wlm < wave_max+0.1))
    running_step_median.append (np.nanmedian(s[region]) )
    
    # Check not nan
    _running_wave_=[]
    _running_step_median_=[]
    for i in range(len(running_wave)):
        if np.isnan(running_step_median[i]):
            if verbose: print("  There is a nan in ",running_wave[i])
        else:
            _running_wave_.append (running_wave[i])
            _running_step_median_.append (running_step_median[i])
    
    fit = np.polyfit(_running_wave_, _running_step_median_, order)
    pfit = np.poly1d(fit)
    fit_median = pfit(wlm)
    
    interpolated_continuum_smooth = interpolate.splrep(_running_wave_, _running_step_median_, s=0.02)
    fit_median_interpolated = interpolate.splev(wlm, interpolated_continuum_smooth, der=0)
     
    if plot:       
        plt.figure(figsize=(fig_size, fig_size/2.5)) 
        plt.plot(wlm,s, alpha=0.5)
        plt.plot(running_wave,running_step_median, "+", ms=15, mew=3)
        plt.plot(wlm, fit_median, label="fit median")
        plt.plot(wlm, fit_median_interpolated, label="fit median_interp")
        plt.plot(wlm, weight_fit_median*fit_median + (1-weight_fit_median)*fit_median_interpolated, label="weighted")
        #extra_display = (np.nanmax(fit_median)-np.nanmin(fit_median)) / 10
        #plt.ylim(np.nanmin(fit_median)-extra_display, np.nanmax(fit_median)+extra_display)
        ymin = np.nanpercentile(s,1)
        ymax=  np.nanpercentile(s,99)
        rango = (ymax-ymin)
        ymin = ymin - rango/10.
        ymax = ymax + rango/10. 
        plt.ylim(ymin,ymax)
        plt.xlim(wlm[0]-10, wlm[-1]+10)
        plt.minorticks_on()
        plt.legend(frameon=False, loc=1, ncol=1)

        plt.axvline(x=wave_min, color='k', linestyle='--')
        plt.axvline(x=wave_max, color='k', linestyle='--')

        plt.xlabel("Wavelength [$\mathrm{\AA}$]")
        
        if exclude_wlm[0][0] != 0:
            for i in range(len(exclude_wlm)):
                plt.axvspan(exclude_wlm[i][0], exclude_wlm[i][1], color='r', alpha=0.1)                      
        plt.show()
        plt.close()
        print('  Weights for getting smooth spectrum:  fit_median =',weight_fit_median,'    fit_median_interpolated =',(1-weight_fit_median))

    return weight_fit_median*fit_median + (1-weight_fit_median)*fit_median_interpolated #   (fit_median+fit_median_interpolated)/2      # Decide if fit_median or fit_median_interpolated
# ----------------------------------------------------------------------------------------------------------------------
# Models and fitting
# ----------------------------------------------------------------------------------------------------------------------


def cumulative_1d_sky(r2, sky_brightness):
    """1D cumulative sky brightness.
        F_sky = 4*pi*r2 * B_sky
    Parameters
    ----------
    r2: np.array(float)
        Square radius from origin.
    sky_brightness: float
        Sky surface brightness.

    Returns
    -------
    cumulative_sky_brightness: np.array(float)
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

    Parameters #TODO
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

# Mr Krtxo \(ﾟ▽ﾟ)/
