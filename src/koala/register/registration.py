"""
This module contains the tools for registering different observational data.
"""

# =============================================================================
# Basics packages
# =============================================================================

from scipy.interpolate import NearestNDInterpolator
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# =============================================================================
# Astropy and associated packages
# =============================================================================
from photutils.centroids import centroid_2dg, centroid_1dg, centroid_com

# =============================================================================
# KOALA packages
# =============================================================================
from koala.ancillary import interpolate_image_nonfinite, make_white_image_from_array
from koala.plotting import qc_plot
from koala.plotting.qc_plot import qc_registration, qc_moffat, qc_registration_crosscorr
from koala.cubing import Cube, build_cube, interpolate_rss
from koala.rss import RSS
from koala.exceptions.exceptions import FitError



def fit_2d_gauss_profile(data, wave_range=None, p0=None, fitter_args=None,
                         plot=False, quick_cube_pix_size=0.3):
    """Fit a 2D Moffat profile to a data set (RSS or Cube) over a given wavelength range.

    Based on astropy.modelling.functional_models.Moffat2D, performs a fit to a data set using
    Levenberg-Marquardt Least Squares Fitting.

    When the provided data corresponds to a RSS, a datacube will be created.
    This helps to better sample the profile distribution of the star.
    """
    print("[Registration] Fitting 2D Moffat profile")
    if fitter_args is None:
        fitter_args = dict(maxiter=10000)
    try:
        from astropy.modeling.functional_models import Gaussian2D
        from astropy.modeling.fitting import LevMarLSQFitter
        fitter = LevMarLSQFitter()
    except ModuleNotFoundError as err:
        raise err

    if wave_range is not None:
        wave_mask = (data.wavelength >= wave_range[0]) & (
            data.wavelength <= wave_range[1])
    else:
        wave_mask = np.ones_like(data.wavelength, dtype=bool)

    # Collect the data to fit
    if type(data) is RSS:
        print(
            "[Registration]  Data provided in RSS format --> creating a dummy datacube")
        cube_size_arcsec = (
            data.info['fib_ra_offset'].max(
            ) - data.info['fib_ra_offset'].min(),
            data.info['fib_dec_offset'].max() - data.info['fib_dec_offset'].min())
        x, y = np.meshgrid(
            np.arange(- cube_size_arcsec[1] / 2 + quick_cube_pix_size / 2,
                      cube_size_arcsec[1] / 2 + quick_cube_pix_size / 2,
                      quick_cube_pix_size),
            np.arange(- cube_size_arcsec[0] / 2 + quick_cube_pix_size / 2,
                      cube_size_arcsec[0] / 2 + quick_cube_pix_size / 2,
                      quick_cube_pix_size))
        datacube = np.zeros((data.wavelength.size, *x.shape))

        # Interpolate RSS to a datacube
        datacube, datacube_var, _ = interpolate_rss(
            data, pixel_size_arcsec=quick_cube_pix_size,
            kernel_size_arcsec=1,
            datacube=datacube)
        intensity = np.nansum(datacube[wave_mask], axis=0)
        variance = np.nansum(datacube_var[wave_mask], axis=0)

    elif type(data) is Cube:
        intensity = np.nansum(
            data.intensity[wave_mask, :, :], axis=0).flatten()
        variance = np.nansum(data.variance[wave_mask, :, :], axis=0).flatten()
        x, y = data.info['spax_dec_offset'], data.info['spax_ra_offset']

    # Filter values
    finite_mask = np.isfinite(intensity)
    # -> Fitting
    if p0 is None:
        # Get centre of mass as initial guess
        amplitude = np.nansum(intensity[finite_mask])

        x_com, y_com = (
            np.nansum(x[finite_mask] * intensity[finite_mask]) / amplitude,
            np.nansum(y[finite_mask] * intensity[finite_mask]) / amplitude)
        # Assuming there is only one source, set the Amplitude as the total flux

        # standard deviation
        x_s = np.nansum((x[finite_mask] - x_com)**2 * intensity[finite_mask]
                        ) / amplitude
        y_s = np.nansum((y[finite_mask] - y_com) ** 2 * intensity[finite_mask]
                        ) / amplitude

        # Pack everything
        p0 = dict(amplitude=amplitude, x_mean=x_com, y_mean=y_com,
                  x_stddev=x_s, y_stddev=y_s, theta=0)
        print("[Registration] 2D Gaussian Initial guess: ", p0)
    # Initialise model
    profile_model = Gaussian2D(**p0)
    # Set bounds to improve performance and prevent unphysical results
    profile_model.amplitude.bounds = (0, None)
    profile_model.x_mean.bounds = (x.min(), x.max())
    profile_model.y_mean.bounds = (y.min(), y.max())
    profile_model.x_stddev.bounds = (
        .1, np.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2))
    profile_model.y_stddev.bounds = (
        .1, np.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2))
    # Fit model to data
    try:
        fit_model = fitter(profile_model, x[finite_mask], y[finite_mask],
                           intensity[finite_mask],
                           filter_non_finite=True,
                           **fitter_args)
    except FitError as err:
        raise err
    else:
        if plot:
            # fig = qc_moffat(intensity, x, y, fit_model)
            fig = None
        else:
            fig = None
    # For RSS data you need to convert the pixel values to arcsec
    # if type(data) is RSS:
    #     fit_model.x_0 *= quick_cube_pix_size
    #     fit_model.y_0 *= quick_cube_pix_size
    #     fit_model.gamma *= quick_cube_pix_size

    return fit_model, fig


def fit_moffat_profile(data, wave_range=None, p0=None, fitter_args=None,
                       plot=False, quick_cube_pix_size=0.3):
    """Fit a 2D Moffat profile to a data set (RSS or Cube) over a given wavelength range.

    Based on astropy.modelling.functional_models.Moffat2D, performs a fit to a data set using
    Levenberg-Marquardt Least Squares Fitting.

    When the provided data corresponds to a RSS, a datacube will be created.
    This helps to better sample the profile distribution of the star.
    """
    print("[Registration] Fitting 2D Moffat profile")
    if fitter_args is None:
        fitter_args = dict(maxiter=10000)
    try:
        from astropy.modeling.functional_models import Moffat2D
        from astropy.modeling.fitting import LevMarLSQFitter
        fitter = LevMarLSQFitter()
    except ModuleNotFoundError as err:
        raise err

    if wave_range is not None:
        wave_mask = (data.wavelength >= wave_range[0]) & (
            data.wavelength <= wave_range[1])
    else:
        wave_mask = np.ones_like(data.wavelength, dtype=bool)

    # Collect the data to fit
    if type(data) is RSS:
        print(
            "[Registration]  Data provided in RSS format --> creating a dummy datacube")
        cube_size_arcsec = (
            data.info['fib_ra_offset'].max(
            ) - data.info['fib_ra_offset'].min(),
            data.info['fib_dec_offset'].max() - data.info['fib_dec_offset'].min())
        x, y = np.meshgrid(
            np.arange(- cube_size_arcsec[1] / 2 + quick_cube_pix_size / 2,
                      cube_size_arcsec[1] / 2 + quick_cube_pix_size / 2,
                      quick_cube_pix_size),
            np.arange(- cube_size_arcsec[0] / 2 + quick_cube_pix_size / 2,
                      cube_size_arcsec[0] / 2 + quick_cube_pix_size / 2,
                      quick_cube_pix_size))
        datacube = np.zeros((data.wavelength.size, *x.shape))

        # Interpolate RSS to a datacube
        datacube, datacube_var, _ = interpolate_rss(
            data, pixel_size_arcsec=quick_cube_pix_size,
            kernel_size_arcsec=1,
            datacube=datacube)
        intensity = np.nansum(datacube[wave_mask], axis=0)
        # variance = np.nansum(datacube_var[wave_mask], axis=0)
        del datacube, datacube_var

    elif type(data) is Cube:
        intensity = np.nansum(
            data.intensity[wave_mask, :, :], axis=0).flatten()
        # variance = np.nansum(data.variance[wave_mask, :, :], axis=0).flatten()
        x, y = data.info['spax_dec_offset'], data.info['spax_ra_offset']

    # Filter values
    finite_mask = np.isfinite(intensity)
    # -> Fitting
    if p0 is None:
        # Get centre of mass as initial guess
        amplitude = np.nansum(intensity[finite_mask])

        x_com, y_com = (
            np.nansum(x[finite_mask] * intensity[finite_mask]) / amplitude,
            np.nansum(y[finite_mask] * intensity[finite_mask]) / amplitude)
        # Assuming there is only one source, set the Amplitude as the total flux

        # Characteristic square radius
        x_s = np.nansum((x[finite_mask] - x_com)**2 * intensity[finite_mask]
                        ) / amplitude
        y_s = np.nansum((y[finite_mask] - y_com) ** 2 * intensity[finite_mask]
                        ) / amplitude
        gamma = np.sqrt(x_s + y_s)
        # Power-law slope
        alpha = 1
        # Pack everything
        p0 = dict(amplitude=amplitude, x_0=x_com, y_0=y_com,
                  gamma=gamma, alpha=alpha)
        print("[Registration] 2D Moffat Initial guess: ", p0)
    # Initialise model
    profile_model = Moffat2D(**p0)
    # Set bounds to improve performance and prevent unphysical results
    profile_model.amplitude.bounds = (0, None)
    profile_model.x_0.bounds = (x.min(), x.max())
    profile_model.y_0.bounds = (y.min(), y.max())
    profile_model.gamma.bounds = (
        .1, np.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2))
    profile_model.alpha.bounds = (0, 10)
    # Fit model to data
    try:
        fit_model = fitter(profile_model, x[finite_mask], y[finite_mask],
                           intensity[finite_mask],
                           filter_non_finite=True,
                           **fitter_args)
    except FitError as err:
        raise err
    else:
        if plot:
            fig = qc_moffat(intensity, x, y, fit_model)
        else:
            fig = None
    # For RSS data you need to convert the pixel values to arcsec
    # if type(data) is RSS:
    #     fit_model.x_0 *= quick_cube_pix_size
    #     fit_model.y_0 *= quick_cube_pix_size
    #     fit_model.gamma *= quick_cube_pix_size

    return fit_model, fig


def register_stars(data_set, moffat=True, plot=False, com_power=5., **fit_args):
    """ Register a collection of data (either RSS or Cube) corresponding to stars.

    The registration is based on a 2D-Moffat profile fit, where the coordinates of each data will be updated to the
    star reference frame.

    Parameters
    ----------
    data_set: list
        Collection of RSS or Cube data.
    moffat: bool, default=True
        If True, a 2D-Moffat profile (koala.registration.fit_moffat_profile)
        will be fitted to derive the position of the star. If False, the centre
        of light will be used instead.
    plot: bool, default=False
        Set to True to produce a QC plot.
    fit_args:
        Extra arguments passed to the `fit_moffat_profile` function.
    Returns
    -------

    """
    plots = []

    if moffat:
        print("[Registering] Registering stars through 2D-Moffat modelling.")
    else:
        print("[Registering] Registering stars using the Center of Mass (light).")
    for i, data in enumerate(data_set):
        print("Object: ", data.info['name'])
        if moffat:
            fit_model, moffat_fig = fit_moffat_profile(
                data, plot=plot, **fit_args)
            print("[Registration] 2D Moffat fit results:\n", '-' * 50, '\n',
                  fit_model, '\n', '-' * 50, '\n')
            new_centre = fit_model.x_0.value / 3600, fit_model.y_0.value / 3600
            new_fib_offset_coord = (data.info['fib_ra_offset'] - new_centre[0] * 3600,
                                    data.info['fib_dec_offset'] - new_centre[1] * 3600)
            data.update_coordinates(new_centre=new_centre,
                                    new_fib_offset_coord=new_fib_offset_coord)
            plots.append(moffat_fig)
        else:
            x_com, y_com = data.get_centre_of_mass(
                power=com_power,
                wavelength_step=data.wavelength.size)  # arcsec
            new_centre = (np.nanmedian(x_com) / 3600,
                          np.nanmedian(y_com) / 3600)  # deg
            new_fib_offset_coord = (data.info['fib_ra_offset'] - new_centre[0] * 3600,
                                    data.info['fib_dec_offset'] - new_centre[1] * 3600)
            data.update_coordinates(new_centre=new_centre,
                                    new_fib_offset_coord=new_fib_offset_coord)
    if plot:
        fig = qc_registration(data_set)
        plots.append(fig)
    else:
        fig = None
    return plots


def register_centroid(data_set, wave_range=None,
                      plot=False, quick_cube_pix_size=0.3,
                      centroider='com',
                      subbox=None):
    """ TODO
    
    subbox (row, col)
    Returns
    -------

    """
    if centroider == 'com':
        centroider = centroid_com
    elif centroider == 'gauss':
        centroider = centroid_2dg

    if subbox is None:
        subbox = [[None, None], [None, None]]
    plots = []
    images = []
    centroids = []
    for data in data_set:
        if type(data) is RSS:
            print(
                "[Registration]  Data provided in RSS format --> creating a dummy datacube")
            cube_size_arcsec = (
                data.info['fib_ra_offset'].max(
                ) - data.info['fib_ra_offset'].min(),
                data.info['fib_dec_offset'].max()
                - data.info['fib_dec_offset'].min())
            x, y = np.meshgrid(
                np.arange(- cube_size_arcsec[1] / 2 + quick_cube_pix_size / 2,
                          cube_size_arcsec[1] / 2 + quick_cube_pix_size / 2,
                          quick_cube_pix_size),
                np.arange(- cube_size_arcsec[0] / 2 + quick_cube_pix_size / 2,
                          cube_size_arcsec[0] / 2 + quick_cube_pix_size / 2,
                          quick_cube_pix_size))
            datacube = np.zeros((data.wavelength.size, *x.shape))
    
            # Interpolate RSS to a datacube
            datacube, _, _ = interpolate_rss(
                data, pixel_size_arcsec=quick_cube_pix_size,
                kernel_size_arcsec=1.,
                datacube=datacube)

            image = make_white_image_from_array(datacube, wavelength=data.wavelength,
                                                wave_range=wave_range, s_clip=3.0)
        elif type(data) is Cube:
            image = data.get_white_image(wave_range=wave_range, s_clip=3.0)
            image /= np.nansum(image)

        # Select a subbox
        image = image[subbox[0][0]:subbox[0][1],
                      subbox[1][0]: subbox[1][1]]

        # Mask bad values
        image = interpolate_image_nonfinite(image)
        images.append(image)
        # Find centroid
        centroids.append(np.array(centroider(image)))


    ref_centre = (data_set[0].info['cen_ra'], data_set[0].info['cen_dec'])
    # Update the coordinats of the rest of frames
    for i, data in enumerate(data_set[1:]):
        print("[Registration] Object: ", data.info['name'])
        shift = centroids[0] - centroids[i+1]
        # Convert the shift in pixels to arcseconds
        shift *= quick_cube_pix_size
        print(f"[Registration] Shift found: {shift} (arcsec)")
        new_fib_offset_coord = (data.info['fib_ra_offset'] + shift[0],
                                data.info['fib_dec_offset'] + shift[1])
        data.update_coordinates(new_centre=ref_centre,
                                new_fib_offset_coord=new_fib_offset_coord)
        
        
    if plot:
        fig = qc_plot.qc_registration_centroids(images, centroids)
        plots.append(fig)
    else:
        fig = None
    return plots

# =============================================================================
# Cross-correlation
# =============================================================================

def cross_correlate_images(list_of_images, oversample=100):
    """Compute image cross-correlation shift.
    
    Description
    -----------
    Apply the skimage.registration.phase_cross_correlation method to find 
    the shift of a list of images with respect to the first one.

    """
    try:
        from skimage.registration import phase_cross_correlation
    except Exception:
        raise ImportError()

    results = []
    for i in range(len(list_of_images) - 1):
        # The shift ordering is consistent with the input image shape
        shift, error, diffphase = phase_cross_correlation(
            list_of_images[0], list_of_images[i+1],
            upsample_factor=100)
        results.append([shift, error, diffphase])
    return results

def register_crosscorr(data_set, ref_image=0,
                       oversample=100, quick_cube_pix_size=0.3,
                       wave_range=None, plot=False):
    """ Register a collection of data (either RSS or Cube) corresponding to stars.

    The registration is based on a 2D-Moffat profile fit, where the coordinates of each data will be updated to the
    star reference frame.

    Parameters
    ----------
    data_set: list
        Collection of RSS or Cube data.
    moffat: bool, default=True
        If True, a 2D-Moffat profile (koala.registration.fit_moffat_profile)
        will be fitted to derive the position of the star. If False, the centre
        of light will be used instead.
    plot: bool, default=False
        Set to True to produce a QC plot.
    fit_args:
        Extra arguments passed to the `fit_moffat_profile` function.
    Returns
    -------

    """

    plots = []
    images = []
    for data in data_set:
        if type(data) is RSS:
            print(
                "[Registration]  Data provided in RSS format --> creating a dummy datacube")
            cube_size_arcsec = (
                data.info['fib_ra_offset'].max(
                ) - data.info['fib_ra_offset'].min(),
                data.info['fib_dec_offset'].max()
                - data.info['fib_dec_offset'].min())
            x, y = np.meshgrid(
                np.arange(- cube_size_arcsec[1] / 2 + quick_cube_pix_size / 2,
                          cube_size_arcsec[1] / 2 + quick_cube_pix_size / 2,
                          quick_cube_pix_size),
                np.arange(- cube_size_arcsec[0] / 2 + quick_cube_pix_size / 2,
                          cube_size_arcsec[0] / 2 + quick_cube_pix_size / 2,
                          quick_cube_pix_size))
            datacube = np.zeros((data.wavelength.size, *x.shape))
    
            # Interpolate RSS to a datacube
            datacube, _, _ = interpolate_rss(
                data, pixel_size_arcsec=quick_cube_pix_size,
                kernel_size_arcsec=1.,
                datacube=datacube)
            image = make_white_image_from_array(datacube, data.wavelength,
                                                wave_range=wave_range, s_clip=3.0)
    
        elif type(data) is Cube:
            image = data.get_white_image(wave_range=wave_range, s_clip=3.0)

        # Mask NaN values
        image = interpolate_image_nonfinite(image)
        images.append(image)

    results = cross_correlate_images(images, oversample=oversample)
    ref_centre = (data_set[0].info['cen_ra'], data_set[0].info['cen_dec'])
    # Update the coordinats of the rest of frames
    for i, data in enumerate(data_set[1:]):
        print("[Registration] Object: ", data.info['name'])
        shift = results[i][0].copy()
        # Convert the shift in pixels to arcseconds
        shift *= quick_cube_pix_size
        print(f"[Registration] Shift found: {shift} (arcsec)")
        new_fib_offset_coord = (data.info['fib_ra_offset'] + shift[1],
                                data.info['fib_dec_offset'] + shift[0])
        data.update_coordinates(new_centre=ref_centre,
                                new_fib_offset_coord=new_fib_offset_coord)
        
        
    if plot:
        fig = qc_registration_crosscorr(images, results)
        plots.append(fig)
    else:
        fig = None
    return plots

# =============================================================================
# Manual
# =============================================================================

def register_manual(data_set, offset_set, absolute=False):
    """ Register a collection of data (either RSS or Cube) corresponding to stars.

    The registration is based on a 2D-Moffat profile fit, where the coordinates of each data will be updated to the
    star reference frame.

    Parameters
    ----------
    data_set: list
        Collection of RSS or Cube data.
    moffat: bool, default=True
        If True, a 2D-Moffat profile (koala.registration.fit_moffat_profile)
        will be fitted to derive the position of the star. If False, the centre
        of light will be used instead.
    plot: bool, default=False
        Set to True to produce a QC plot.
    fit_args:
        Extra arguments passed to the `fit_moffat_profile` function.
    Returns
    -------

    """
    for i, data in enumerate(data_set):
        print("[Registration] Object: ", data.info['name'])
        offset = offset_set[i]
        print(f"[Registration] Offset: {offset} (arcsec)")
        if absolute:
            ref_centre = (offset[0], offset[1])
        else:
            ref_centre = (data_set[0].info['cen_ra'] - offset[0] / 3600,
                          data_set[0].info['cen_dec'] - offset[1] / 3600)
        data.update_coordinates(new_centre=ref_centre)

# =============================================================================
# 
# =============================================================================
def register_interactive(data_set, quick_cube_pix_size=0.2, wave_range=None):
    """
    Fully manual registration via interactive plot
    """
    import matplotlib
    from matplotlib import pyplot as plt

    images = []

    for data in data_set:
        if type(data) is RSS:
            print(
                "[Registration]  Data provided in RSS format --> creating a dummy datacube")
            cube_size_arcsec = (
                data.info['fib_ra_offset'].max(
                ) - data.info['fib_ra_offset'].min(),
                data.info['fib_dec_offset'].max()
                - data.info['fib_dec_offset'].min())
            x, y = np.meshgrid(
                np.arange(- cube_size_arcsec[1] / 2 + quick_cube_pix_size / 2,
                          cube_size_arcsec[1] / 2 + quick_cube_pix_size / 2,
                          quick_cube_pix_size),
                np.arange(- cube_size_arcsec[0] / 2 + quick_cube_pix_size / 2,
                          cube_size_arcsec[0] / 2 + quick_cube_pix_size / 2,
                          quick_cube_pix_size))
            datacube = np.zeros((data.wavelength.size, *x.shape))
    
            # Interpolate RSS to a datacube
            datacube, _, _ = interpolate_rss(
                data, pixel_size_arcsec=quick_cube_pix_size,
                kernel_size_arcsec=1.,
                datacube=datacube)
            image = make_white_image_from_array(datacube, data.wavelength,
                                                wave_range=wave_range, s_clip=3.0)
            image_pix_size = quick_cube_pix_size

        elif type(data) is Cube:
            image = data.get_white_image(wave_range=wave_range, s_clip=3.0)
            image_pix_size = data.info['pixel_size_arcsec']

        # Mask NaN values
        image = interpolate_image_nonfinite(image)
        images.append(image)

    ### Start the interactive GUI ###
    # Close previus figures
    plt.close()
    plt.ion()
    centres = []

    def mouse_event(event):
        """..."""
        centres.append([event.xdata / 3600, event.ydata / 3600])
        print('x: {} and y: {}'.format(event.xdata, event.ydata))

    n_rss = len(data_set)

    fig = plt.figure()
    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
    for i in range(n_rss):
        ax = fig.add_subplot(1, n_rss, i + 1)
        ax.scatter(data_set[0].info["fib_ra_offset"], data_set[0].info["fib_dec_offset"],
                   c=np.nanmedian(data_set[0].intensity_corrected, axis=1))
    plt.show()
    plt.ioff()
    return centres

# Mr Krtxo \(ﾟ▽ﾟ)/



