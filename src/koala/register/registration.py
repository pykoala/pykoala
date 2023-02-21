"""
This module contains the tools for registering different observational data.
"""

# =============================================================================
# Basics packages
# =============================================================================
import numpy as np

# =============================================================================
# Astropy and associated packages
# =============================================================================

# =============================================================================
# KOALA packages
# =============================================================================
from koala.exceptions.exceptions import ClassError, FitError
from koala.rss import RSS
from koala.cubing import Cube

def fit_moffat_profile(data, wave_range=None, p0=None, fitter_args=None, plot=False):
    """Fit a 2D Moffat profile to a data set (RSS or Cube) over a given wavelength range.

    Based on astropy.modelling.functional_models.Moffat2D, performs a fit to a data set using
    Levenberg-Marquardt Least Squares Fitting.
    """
    if fitter_args is None:
        fitter_args = dict(maxiter=10000)
    try:
        from astropy.modeling.functional_models import Moffat2D
        from astropy.modeling.fitting import LevMarLSQFitter
        fitter = LevMarLSQFitter()
    except ModuleNotFoundError as err:
        raise err

    if wave_range is not None:
        wave_mask = (data.wavelength >= wave_range[0]) & (data.wavelength <= wave_range[1])
    else:
        wave_mask = np.ones_like(data.wavelength, dtype=bool)

    # Collect the data to fit
    if type(data) is RSS:
        intensity = np.nansum(data.intensity_corrected[:, wave_mask], axis=1)
        variance = np.nansum(data.variance_corrected[:, wave_mask], axis=1)
        x = data.info['fib_ra_offset'].copy()
        y = data.info['fib_dec_offset'].copy()
    elif type(data) is Cube:
        intensity = np.nansum(data.intensity[wave_mask, :, :], axis=0).flatten()
        variance = np.nansum(data.variance[wave_mask, :, :], axis=0).flatten()
        x, y = np.meshgrid(np.arange(0, data.intensity.shape[2]), np.arange(0, data.intensity.shape[1]))
        x, y = x.flatten(), y.flatten()
    else:
        raise ClassError([RSS.__class__, Cube.__class__], data.__class__)

    # Remove residual sky
    intensity -= np.nanmedian(intensity)
    # -> Fitting
    if p0 is None:
        # Get centre of mass as initial guess
        x_com, y_com = data.get_centre_of_mass(wavelength_step=wave_mask.size)
        x_com, y_com = np.nanmedian(x_com[wave_mask]), np.nanmedian(y_com[0])
        # Assuming there is only one source, set the Amplitude as the total flux
        amplitude = np.nansum(intensity)
        # Characteristic square radius
        x_s = np.nansum((x - x_com)**2 * intensity) / amplitude
        y_s = np.nansum((y - y_com)** 2 * intensity) / amplitude
        gamma = x_s + y_s
        # Power-law slope
        alpha = 1
        # Pack everything
        p0 = dict(amplitude=amplitude, x_0=x_com, y_0=y_com, gamma=gamma, alpha=alpha)
    # Initialise model
    profile_model = Moffat2D(**p0)
    # Set bounds to improve performance and prevent unphysical results
    #profile_model.amplitude.bounds = (0, None)
    #profile_model.x_0.bounds = (x.min(), x.max())
    #profile_model.y_0.bounds = (y.min(), y.max())
    #profile_model.gamma.bounds = (p0['gamma'] * 0.1, p0['gamma'] * 10)
    #profile_model.alpha.bounds = (0, 100)
    # Fit model to data
    try:
        fit_model = fitter(profile_model, x, y, intensity,
                           # weights=1/variance**0.5, # TODO: THIS IS NOT WORKING PROPERLY
                           **fitter_args)
    except FitError as err:
        raise err
    else:
        if plot:
            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.set_title("2D Moffat fit")
            inax = ax.inset_axes((0, 0.65, 1, 0.35))
            inax.plot(np.sqrt((x-fit_model.x_0.value)**2 + (y-fit_model.y_0.value)**2),
                      (intensity - fit_model(x, y)) / intensity, 'k+')
            inax.axhspan(-0.3, 0.3, alpha=0.3, color='k')
            inax.grid(visible=True)
            inax.set_ylabel(r'$\frac{I-\hat{I}}{I}$', fontsize=17)
            inax.set_xlabel(r'$|r-\hat{r}_0|$', fontsize=15)
            inax.set_yscale('symlog', linthresh=0.1)
            #inax.set_ylim(-1, 100)
            inax = ax.inset_axes((0, 0.0, 0.5, 0.5))
            inax.set_title("Data")
            c = inax.hexbin(x, y, C=np.log10(intensity), gridsize=30, cmap='nipy_spectral')
            inax.plot(fit_model.x_0.value, fit_model.y_0.value, 'k+', ms=14, mew=2)
            plt.colorbar(mappable=c, ax=inax, orientation='horizontal', anchor=(0, -1))
            inax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            inax = ax.inset_axes((0.55, 0.0, 0.5, 0.5))
            inax.set_title(r"$\log_{10}$(Data/Model)")
            c = inax.hexbin(x, y, C=np.log10(intensity/fit_model(x, y)), gridsize=30,
                            vmin=-.5, vmax=.5, cmap='seismic')
            inax.plot(fit_model.x_0.value, fit_model.y_0.value, 'k+', ms=14, mew=2)
            plt.colorbar(mappable=c, ax=inax, orientation='horizontal', anchor=(0, -1))
            inax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            ax.axis("off")
            plt.show(fig)
    return fit_model


def register_stars(data_set, moffat=True, plot=False, com_power=5., **fit_args):
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
    if moffat:
        print("[Registering] Registering stars through 2D-Moffat modelling.")
    else:
        print("[Registering] Registering stars using the Center of Mass (light).")

    if plot:
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    for data in data_set:
        print("Object: ", data.info['name'])
        if moffat:
            fit_model = fit_moffat_profile(data, **fit_args)
            new_centre = fit_model.x_0.value / 3600, fit_model.y_0.value / 3600
            new_fib_offset_coord = (data.info['fib_ra_offset'] - new_centre[0],
                                    data.info['fib_dec_offset'] - new_centre[1])
            data.update_coordinates(new_centre=new_centre, new_fib_offset_coord=new_fib_offset_coord)
        else:
            x_com, y_com = data.get_centre_of_mass(power=com_power)  # arcsec
            new_centre = (np.nanmedian(x_com) / 3600, np.nanmedian(y_com) / 3600)  # deg
            new_fib_offset_coord = (data.info['fib_ra_offset'] - new_centre[0],
                                    data.info['fib_dec_offset'] - new_centre[1])
            data.update_coordinates(new_centre=new_centre, new_fib_offset_coord=new_fib_offset_coord)
        if plot:
            ax.scatter(data.info['fib_ra_offset'], data.info['fib_dec_offset'], ec='k', alpha=0.4)


def register_interactive(data_set):
    """
    Fully manual registration via interactive plot
    """
    import matplotlib
    from matplotlib import pyplot as plt
    #matplotlib.use("QtAgg")
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
        data.update_coordinates(new_centre=new_centre, new_fib_offset_coord=new_fib_offset_coord)


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
    fig, axs = plt.subplots(nrows=1, ncols=len(all_images), figsize=(len(all_images) * 4, 4))
    axs[0].imshow(all_images[0], origin='lower')
    x, y = np.meshgrid(np.arange(0.5, all_images[0].shape[1], all_images[0].shape[1] // 10),
                       np.arange(0.5, all_images[0].shape[0], all_images[0].shape[0] // 10))
    axs[0].scatter(x.flatten(), y.flatten(), marker='+', c='k')
    for i in range(len(all_images) - 1):
        xoff, yoff, exoff, eyoff = chi2_shift(all_images[0], all_images[i + 1],
                                              return_error=True, upsample_factor='auto')
        offsets.append([xoff * pixel_size, yoff * pixel_size, exoff * pixel_size, eyoff * pixel_size])

        new_centre = (data_set[i + 1].info['cen_ra'] - xoff * pixel_size / 3600,
                      data_set[i + 1].info['cen_dec'] - yoff * pixel_size / 3600)
        new_fib_offset_coord = (data_set[i + 1].info['fib_ra_offset'] - xoff * pixel_size,
                                data.info['fib_dec_offset'] - yoff * pixel_size)
        data_set[i + 1].update_coordinates(new_centre=new_centre, new_fib_offset_coord=new_fib_offset_coord)
        # Plot
        axs[i + 1].imshow(all_images[i+1], origin='lower')
        axs[i + 1].scatter(x.flatten() - yoff, y.flatten() - xoff, marker='+', c='k')
    plt.show()
    return offsets

# Mr Krtxo \(ﾟ▽ﾟ)/
