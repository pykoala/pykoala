"""
This module contains the tools for registering extended sources.
"""

# =============================================================================
# Basics packages
# =============================================================================
from pykoala.plotting.qc_plot import qc_registracion, qc_moffat
from pykoala.cubing import Cube, build_cube, interpolate_rss
from pykoala.rss import RSS
from pykoala.exceptions.exceptions import FitError
import numpy as np
from photutils.centroids import centroid_2dg, centroid_1dg, centroid_com
import warnings

warnings.filterwarnings("ignore")
# =============================================================================
# Astropy and associated packages
# =============================================================================

# =============================================================================
# KOALA packages
# =============================================================================

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
        shift, error, diffphase = phase_cross_correlation(
            list_of_images[0], list_of_images[i+1],
            upsample_factor=100)
        results.append([shift, error, diffphase])
    return results

def register_crosscorr(data_set, ref_image=0,
                       oversample=100, quick_cube_pix_size=0.2,
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

        if wave_range is not None:
            wave_mask = (data.wavelength >= wave_range[0]) & (
                data.wavelength <= wave_range[1])
        else:
            wave_mask = np.ones_like(data.wavelength, dtype=bool)

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
                kernel_size_arcsec=1,
                datacube=datacube)
            image = np.nanmean(datacube[wave_mask], axis=0)
            image /= np.nansum(image)
            images.append(np.nanmean(datacube[wave_mask], axis=0))
    
        elif type(data) is Cube:
            image = np.nanmean(data.intensity[wave_mask, :, :], axis=0)
            image /= np.nansum(image)
            images.append(image)

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
            new_fib_offset_coord = (data.info['fib_ra_offset'] + new_centre[0] * 3600,
                                    data.info['fib_dec_offset'] - new_centre[1] * 3600)
            data.update_coordinates(new_centre=new_centre,
                                    new_fib_offset_coord=new_fib_offset_coord)
    if plot:
        fig = qc_registracion(data_set)
        plots.append(fig)
    else:
        fig = None
    return plots


def register_galaxy(data_set, plot=False, com_power=5., **fit_args):
    """ Register a collection of data (either RSS or Cube) corresponding to stars.

    The registration is based on a 2D-Moffat profile fit, where the coordinates of each data will be updated to the
    star reference frame.

    Parameters
    ----------
    data_set: list
        Collection of RSS or Cube data.
    moffat: bool, default=True
        If True, a 2D-Moffat profile will be fitted to derive the position of the star. If False, the centre of light
        will be used instead.
    fit_args:
        Extra arguments passed to the `fit_moffat_profile` function.
    Returns
    -------

    """
    for i, data in enumerate(data_set):
        temp_cube = build_cube(rss_set=[data],
                               reference_coords=(data.info['cen_ra'],
                                                 data.info['cen_dec']),
                               reference_pa=data.info['pos_angle'],
                               cube_size_arcsec=(60, 60),
                               pixel_size_arcsec=1.,
                               kernel_size_arcsec=1.5, )

        cube_collapsed = np.nanmean(temp_cube.intensity_corrected, axis=0)
        #cube_collapsed[~np.isfinite(cube_collapsed)] = 0
        centroid = centroid_com(cube_collapsed, mask=~
                                np.isfinite(cube_collapsed))
        ra_cen = np.interp(centroid[1], np.arange(0, temp_cube.n_cols, 1),
                           temp_cube.info['spax_ra_offset'][0, :])
        dec_cen = np.interp(centroid[0], np.arange(0, temp_cube.n_rows, 1),
                            temp_cube.info['spax_dec_offset'][:, 0])

        print(ra_cen, dec_cen)
        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(cube_collapsed, origin='lower')
        plt.plot(*centroid, 'r+', ms=10)
        del temp_cube

    if plot:
        fig = qc_registracion(data_set)
    else:
        fig = None
    return fig


def register_interactive(data_set):
    """
    Fully manual registration via interactive plot
    """
    import matplotlib
    from matplotlib import pyplot as plt
    # matplotlib.use("QtAgg")
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
    return centres


def register_new_centres(data_set, centroids):
    """TODO"""
    for i, data in enumerate(data_set):
        new_centre = centroids[i]
        new_fib_offset_coord = (data.info['fib_ra_offset'] - centroids[i][0] * 3600,
                                data.info['fib_dec_offset'] - centroids[i][1] * 3600)
        data.update_coordinates(new_centre=new_centre,
                                new_fib_offset_coord=new_fib_offset_coord)


def register_extended_source(data_set, pixel_size=0.1):
    """TODO"""
    from matplotlib import pyplot as plt
    from scipy.interpolate import griddata
    try:
        from image_registration import chi2_shift
    except ImportError as err:
        raise err()

    all_images = []
    for i, data in enumerate(data_set):
        x, y = np.meshgrid(np.arange(data.info['fib_ra_offset'].min(),
                                     data.info['fib_ra_offset'].max(),
                                     pixel_size),
                           np.arange(data.info['fib_dec_offset'].min(),
                                     data.info['fib_dec_offset'].max(),
                                     pixel_size))

        image = griddata(
            list(zip(data.info['fib_ra_offset'], data.info['fib_dec_offset'])),
            np.nanmedian(data.intensity_corrected, axis=1), (x, y))
        all_images.append(image)
    offsets = []
    fig, axs = plt.subplots(nrows=1, ncols=len(
        all_images), figsize=(len(all_images) * 4, 4))
    axs[0].imshow(all_images[0], origin='lower')
    x, y = np.meshgrid(np.arange(0.5, all_images[0].shape[1], all_images[0].shape[1] // 10),
                       np.arange(0.5, all_images[0].shape[0], all_images[0].shape[0] // 10))
    axs[0].scatter(x.flatten(), y.flatten(), marker='+', c='k')
    for i in range(len(all_images) - 1):
        xoff, yoff, exoff, eyoff = chi2_shift(all_images[0], all_images[i + 1],
                                              return_error=True, upsample_factor='auto')
        offsets.append([xoff * pixel_size, yoff * pixel_size,
                       exoff * pixel_size, eyoff * pixel_size])

        new_centre = (data_set[i + 1].info['cen_ra'] - xoff * pixel_size / 3600,
                      data_set[i + 1].info['cen_dec'] - yoff * pixel_size / 3600)
        new_fib_offset_coord = (data_set[i + 1].info['fib_ra_offset'] - xoff * pixel_size,
                                data.info['fib_dec_offset'] - yoff * pixel_size)
        data_set[i + 1].update_coordinates(new_centre=new_centre,
                                           new_fib_offset_coord=new_fib_offset_coord)
        # Plot
        axs[i + 1].imshow(all_images[i+1], origin='lower')
        axs[i + 1].scatter(x.flatten() - yoff,
                           y.flatten() - xoff, marker='+', c='k')
    plt.show()
    return offsets

# Mr Krtxo \(ﾟ▽ﾟ)/
