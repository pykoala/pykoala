from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable

import numpy as np

from astropy import units as u
from astropy import visualization
from astropy.visualization import quantity_support

from pykoala import vprint
from pykoala import ancillary

quantity_support()
# plt.style.use('dark_background')
SYMMETRIC_CMAP = plt.get_cmap('seismic').copy()
SYMMETRIC_CMAP.set_extremes(bad='gray', under='cyan', over='fuchsia')

DEFAULT_CMAP = plt.get_cmap("gist_earth").copy()
DEFAULT_CMAP.set_bad('gray')

def local_quantity_support(func):
    """Allow astropy Quantities support locally."""
    def wrapper(*args, **kwargs):
        with quantity_support():
            return func(*args, **kwargs)
    return wrapper
        

def default_ax_setting(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True,
                   top=True, left=True, right=True)
    ax.tick_params(which='major', direction='inout',
                   length=8, grid_alpha=.3)
    ax.tick_params(which='minor', direction='in', length=2, grid_alpha=.1)
    ax.grid(True, which='both')

def new_figure(fig_name,
               tweak_axes=True,
               figsize=None,
               **kwargs):
    """
    Close old version of the figure and create new one.
    

    Parameters
    ----------
    figsize : tuple
        Figure size.
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    sharex : str or bool, optional
        Whether panels share the x axis. True, False, or 'col'
        (default; x-axis shared accross columns)
    sharey : str or bool, optional
        Whether panels share the y axis. True, False, or 'row'
        (default; y-axis shared accross rows)
    **gridspec_kw : dict, optional
        Default sets height and width space to `{'hspace': 0, 'wspace': 0})`

    Returns
    -------
    fig : plt.Figure
    axes : ndarray(mpl.Axes)
    """

    plt.close(fig_name)

    if "sharex" not in kwargs:
        kwargs["sharex"] = "col"
    if "sharex" not in kwargs:
        kwargs["sharex"] = "row"
    if "gridspec_kw" not in kwargs:
        kwargs["gridspec_kw"] = {'hspace': 0, 'wspace': 0}
    if "squeeze" not in kwargs:
        kwargs["squeeze"] = False

    if figsize is None:
        figsize = (9 + kwargs.get("ncols", 1),
                   4 + kwargs.get("nrows", 1))

    fig, axes = plt.subplots(num=fig_name, figsize=figsize,
                             layout="constrained", **kwargs)
    if tweak_axes:
        for ax in axes.flat:
            default_ax_setting(ax)
    fig.suptitle(fig_name)

    return fig, axes

def plot_image(fig, ax, data,
               cmap=DEFAULT_CMAP,
               xlabel=None, x=None,
               ylabel=None, y=None,
               cbax=None, norm=None,
               cblabel=None,
               norm_interval=visualization.AsymmetricPercentileInterval,
               interval_args={"lower_percentile": 1.0,
                              "upper_percentile": 99.0},
               stretch=visualization.PowerStretch, stretch_args={"a": 0.7}):
    """
    Plot a colour map (imshow) with axes and colour scales.

    Parameters
    ----------
    fig : plt.Figure
        Figure where the colour map will be drawn.
    ax : mpl.Axes
        Axes where the colour map will be drawn.
    data : ndarray
        2D array to be represented.
    cmap : str or mpl.colors.Norm
    xlabel: str, optional
        Label of x axis.
    x : ndarray, optional
        Values along x axes (defaults to pixel number)
    ylabel : str, optional
        Label of y axis.
    y : ndarray, optional
        Values along y axes (defaults to pixel number)
    cbax: mpl.Axes
        Axes where the colour bar will be drawn.
    norm : mpl.colors.Norm
    cblabel : str
        Label of the colorbar
    Returns
    -------
    im : mpl.AxesImage
    cb : mpl.Colorbar
    """

    if y is None:
        y = np.arange(data.shape[0])
    if x is None:
        x = np.arange(data.shape[1])

    if isinstance(data, u.Quantity):
        unit = data.unit
        value = data.value
    else:
        value = data
        unit = None

    if norm is None:
        interval = norm_interval(**interval_args)
        norm = visualization.ImageNormalize(value, interval=interval,
                              stretch=stretch(**stretch_args),
                              clip=False)
    elif isinstance(norm, str):
        norm = getattr(colors, norm)()

    im = ax.imshow(value,
                   extent=(x[0]-(x[1]-x[0])/2, x[-1]+(x[-1]-x[-2])/2,
                           y[0]-(y[1]-y[0])/2, y[-1]+(y[-1]-y[-2])/2),
                   interpolation='none', origin='lower',
                   cmap=cmap,
                   norm=norm,
                   )
    ax.set_aspect('auto')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if cbax is None:
        cb = fig.colorbar(im, ax=ax, orientation='vertical', shrink=.9,
                          extend='both')
        cbax = cb.ax
    elif cbax:
        cb = fig.colorbar(im, cax=cbax, orientation='vertical', extend='both')
    else:
        cb = None
    if cbax:
        cb.ax.yaxis.set_label_position("left")
        if unit is not None:
            if cblabel is None:
                cblabel = f"{unit}"
        cb.set_label(cblabel)
        cb.ax.tick_params(labelsize='small')
    return im, cb

def plot_fibres(fig, ax, rss=None, x=None, y=None,
                fibre_diam=None, data=None, 
                patch_args={}, use_wcs=False, fix_limits=True,
                cmap=DEFAULT_CMAP, norm=None, cbax=None, cblabel=None, 
                norm_interval=visualization.MinMaxInterval, interval_args={},
                stretch=visualization.LinearStretch, stretch_args={}):
    """
    Plot a colour map of a physical magnitude defined on each fibre.

    Parameters
    ----------
    fig : plt.Figure
        Figure where the colour map will be drawn.
    ax : mpl.Axes
        Axes where the colour map will be drawn.
    rss : RSS, optional, default=None
        Row-Stacked Spectra containing the fibre positions.
    x : np.ndarray or astropy.units.Quantity, optional, default=None
        Fibre position values along the x axis.
    y : np.ndarray or astropy.units.Quantity, optional, default=None
        Fibre position values along the y axis.
    fibre_diam : float or astropy.units.Quantity, optional, default=1.25 arcsec
        Fibre diameter.
    data : ndarray, optional, default=None
        1D array to be represented. If None, fibres will appear as empty circles.
    patch_args : dict, optional
        Additional arguments passed to each fibre :class:`plt.Circle` patch.
    use_wcs : bool, optional, default=False
        If True, use axes WCS world transformation for plotting the patches.
    fix_limits : bool, optional, default=True
        If True, set the limits of the axes using the edge values of ``x`` and ``y``.
    cmap : str or mpl.colors.Norm, optional
        Colormap used to plot the values of ``data``.
    norm : str or mpl.colors.Norm, optional
        Normalization map for plotting ``data``.
    cbax: mpl.Axes
        Axes where the plt.Colorbar will be drawn.
    clabel: str
        Colorbar label.
    norm_interval : astropy.visualization.BaseInterval, optional, default=MinMaxInterval
        Interval to create a normalization map.
    interval_args : dict
        Additional arguments to be passed to the interval.
    stretch : astropy.visualization.BaseStretch, optional, default=LinearStretch
        Stretching used on the normalization map.
    stretch_args : dict, optional
        Additional arguments to be passed to stretch.
    Returns
    -------
    ax : mpl.Axes
    patch_collection : mpl.PatchCollection
    cb : mpl.Colorbar
    """
    if rss is not None:
        x = rss.info['fib_ra']
        y = rss.info['fib_dec']
        fibre_diam = rss.fibre_diameter
    else:
        if fibre_diam is None:
            raise ValueError("Must provide a fibre diameter value")

    if not isinstance(x, u.Quantity):
        # Assume that the values of x and y are sky positions
        x = x << u.degree
        y = y << u.degree

    if fix_limits:
        ax.set_xlim((x.min() - fibre_diam).value,
                    (x.max() + fibre_diam).value)
        ax.set_ylim((y.min() - fibre_diam).value,
                    (y.max() + fibre_diam).value)

    if norm is None:
        interval = norm_interval(**interval_args)
        norm = visualization.ImageNormalize(data, interval=interval,
                              stretch=stretch(**stretch_args),
                              clip=False)
    elif isinstance(norm, str):
        norm = getattr(colors, norm)()

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap).copy()

    if data is not None:
        fib_colors = cmap(norm(data))
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        if cbax is None:
            cb = fig.colorbar(mappable, ax=ax, orientation='vertical', shrink=.9)
            cbax = cb.ax
        elif cbax is False:
            cb = None
        else:
            cb = fig.colorbar(mappable, cax=cbax, orientation='vertical')
        if cbax:
            cb.ax.yaxis.set_label_position("left")
            cb.set_label(cblabel)
            cb.ax.tick_params(labelsize='small')
    else:
        fib_colors = ["none"] * len(x)
        cb = None

    if "edgecolor" not in patch_args:
        patch_args["edgecolor"] = "k"

    if use_wcs:
        patch_args["transform"] = ax.get_transform('world')

    patches = [plt.Circle(
        xy=(x_c, y_c), radius=fibre_diam.to_value(x.unit) / 2, facecolor=color,
        **patch_args
        ) for x_c, y_c, color in zip(x.value, y.value, fib_colors)]
    patch_collection = PatchCollection(patches, match_original=True,
                                       label='Fibre')
    ax.add_collection(patch_collection)
    return ax, patch_collection, cb

def qc_cube(cube, spax_pct=[75, 90, 99]):
    """Create a quality control (QC) plot for a Cube.

    Parameters
    ----------
    cube: pykoala.data_container.Cube
        The cube containing the data to plot.

    spax_pct: array_like (default: [75, 90, 99])
        Spaxel ranks. Number of spanxels per dimension. 

    Returns
    -------
    figure: matplotlib.Figure
    """

    fig = plt.figure(figsize=(12, 12))
    vprint(f"[QCPLOT] Cube QC plot for: {cube.info['name']}")
    plt.suptitle(cube.info['name'])
    gs = fig.add_gridspec(5, 4, wspace=0.35, hspace=0.25)

    # Maps -----
    wl_spaxel_idx = np.sort(np.random.randint(
        low=0, high=cube.intensity.shape[0], size=3))
    wl_col = ['b', 'g', 'r']
    for wl_idx, i in zip(wl_spaxel_idx, range(3)):
        ax = fig.add_subplot(gs[0, i:i+1])
        default_ax_setting(ax)
        ax.set_title(r"$\lambda@${:.1f}".format(cube.wavelength[wl_idx]),
                     fontdict=dict(color=wl_col[i]))
        ax, cb = plot_image(fig, ax, data=cube.intensity[wl_idx],
                            cblabel="Intensity", cmap="cividis")
        cb.ax.yaxis.set_label_position("right")
        ax = fig.add_subplot(gs[1, i:i+1])
        default_ax_setting(ax)
        ax, cb = plot_image(fig, ax, 
                   data=cube.intensity[wl_idx] / cube.variance[wl_idx]**0.5,
                   cblabel="SNR",
                   cmap="jet")
        cb.ax.yaxis.set_label_position("right")
    
    # Plot the mean intensity
    mean_intensity = np.nanmedian(cube.intensity, axis=0)
    mean_variance = np.nanmedian(cube.variance, axis=0)
    mean_intensity[~np.isfinite(mean_intensity)] = 0
    mean_instensity_pos = np.argsort(mean_intensity.flatten())
    mapax = fig.add_subplot(gs[1, -1])
    default_ax_setting(mapax)
    _, cb = plot_image(fig, mapax, 
                   data=mean_intensity / mean_variance**0.5,
                   cblabel="SNR",
                   cmap="jet")
    cb.ax.yaxis.set_label_position("right")

    mapax = fig.add_subplot(gs[0, -1])
    mapax.set_title("Median")
    _, cb = plot_image(fig, mapax,  data=mean_intensity, cblabel="Intensity",
                       cmap="cividis")
    cb.ax.yaxis.set_label_position("right")
    # ------ Spectra -------
    pos_col = ['purple', 'orange', 'cyan']
    x_spaxel_idx = np.random.randint(low=0,
                                     high=cube.intensity.shape[1],
                                     size=3)
    y_spaxel_idx = np.random.randint(low=0,
                                     high=cube.intensity.shape[2],
                                     size=3)
    spaxel_entries = mean_instensity_pos[
        np.array(mean_instensity_pos.size / 100 * np.array(spax_pct),
                 dtype=int)]
    x_spaxel_idx, y_spaxel_idx = np.unravel_index(spaxel_entries,
                                                  shape=mean_intensity.shape)
    ax = fig.add_subplot(gs[2:3, :])
    default_ax_setting(ax)
    for x_idx, y_idx, i in zip(x_spaxel_idx, y_spaxel_idx, range(3)):
        ax.plot(cube.wavelength, cube.intensity[:, x_idx, y_idx], lw=0.8,
                color=pos_col[i])
        mapax.plot(y_idx, x_idx, marker='+', ms=8, mew=2, lw=2, color=pos_col[i])
    for i, wl in enumerate(cube.wavelength[wl_spaxel_idx]):
        ax.axvline(wl, color=wl_col[i], zorder=-1, alpha=0.8)
    ax.axhline(0, alpha=0.2, color='r')
    ylim = np.nanpercentile(
        cube.intensity[np.isfinite(cube.intensity)].value, [40, 95])
    ylim[1] *= 20
    ylim[0] *= 0.1
    np.clip(ylim, a_min=0, a_max=None, out=ylim)
    ax.set_ylim(ylim)
    ax.set_yscale('symlog', linthresh=0.1)
    ax.set_ylabel(f"Flux ({cube.intensity.unit})")

    # SNR ------------
    ax = fig.add_subplot(gs[3:5, :], sharex=ax)
    default_ax_setting(ax)
    for x_idx, y_idx, i in zip(x_spaxel_idx, y_spaxel_idx, range(3)):
        ax.plot(cube.wavelength,
                cube.intensity[:, x_idx, y_idx] / cube.variance[:, x_idx, y_idx]**0.5, lw=0.8,
                color=pos_col[i],
                label=f"Spaxel rank={spax_pct[i]}")
    ax.legend()
    ax.set_ylabel("SNR/pix")
    ax.set_xlabel(f"Wavelength ({cube.wavelength.unit})")
    plt.close(fig)
    return fig

@ancillary.remove_units_dec
def qc_cubing(rss_weight_maps, exposure_times):
    """..."""
    if exposure_times.ndim == 1:
        exposure_times = (rss_weight_maps
        * exposure_times[:, np.newaxis, np.newaxis, np.newaxis])
    n_rss = rss_weight_maps.shape[0]

    p5, p95 = np.nanpercentile(rss_weight_maps, [.1 , 99.9])
    w_im_args = dict(vmin=p5 * 0.9, vmax=p95 * 1.1, cmap='nipy_spectral',
                     interpolation='none', origin='lower', aspect="auto")

    tp5, tp95 = np.nanpercentile(exposure_times, [5 , 95])
    print("Mean exposure time (s): ", np.nanmean(exposure_times))
    t_im_args = dict(vmin=tp5 * 1.1, vmax=tp95 * 1.1, cmap='gnuplot',
                     interpolation='none', origin='lower', aspect="auto")
    fig, axs = plt.subplots(ncols=n_rss + 1, nrows=2, sharex=True,
     sharey=True, constrained_layout=True, gridspec_kw={"hspace": 0.05})
    for i in range(n_rss):
        ax = axs[0, i]
        ax.set_title(f"RSS - {i + 1}")
        ax.imshow(np.nanmedian(rss_weight_maps[i], axis=0), **w_im_args)
        ax = axs[1, i]
        ax.imshow(np.nanmedian(exposure_times[i], axis=0), **t_im_args)

    axs[0, -1].set_title("Mean")
    mappable = axs[0, -1].imshow(np.nanmedian(np.nanmean(rss_weight_maps, axis=0), axis=0),
                      **w_im_args)
    plt.colorbar(mappable, ax=axs[0, -1], label='Median weight')
    mappable = axs[1, -1].imshow(np.nanmedian(np.nanmean(exposure_times, axis=0), axis=0),
                      **t_im_args)
    plt.colorbar(mappable, ax=axs[1, -1], label='Median exp. time (s) / pixel')
    plt.close()
    return fig

@ancillary.remove_units_dec
def qc_fibres_on_fov(fov_size, pixel_colum_pos, pixel_row_pos,
                     fibre_diam=1.25):
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.set_xlim(np.min([-10, pixel_colum_pos.min() - 5]),
                np.max([fov_size[1] + 10, pixel_colum_pos.max() + 5]))
    ax.set_ylim(np.min([-10, pixel_row_pos.min() - 5]),
                np.max([fov_size[0] + 10, pixel_row_pos.max() + 5]))
    ax.set_xlabel("Column pixel (RA)")
    ax.set_ylabel("Row pixel (DEC)")
    ax.grid(visible=True, alpha=0.3)
    
    patches = [plt.Circle(xy=(c, r), radius=fibre_diam/2) for c, r in zip(
        pixel_colum_pos, pixel_row_pos)]
    p = PatchCollection(patches, facecolors='tomato', edgecolors='k', label='Fibre')
    ax.add_collection(p)
    patch = plt.Rectangle(xy=(-.5, -.5),
                          width=fov_size[1] + 0.5,
                          height=fov_size[0] + 0.5,
                          fc='none', ec='k', lw=2, label='Cube FoV')
    ax.add_patch(patch)
    ax.legend(handles=[patch])
    return fig
# =============================================================================
# Star profile
# =============================================================================


def qc_moffat(intensity, ra_offset, dec_offset, fit_model):
    """#TODO"""

    r = np.sqrt(
        (ra_offset - fit_model.x_0.value)**2 + (dec_offset - fit_model.y_0.value)**2)
    r = r.flatten()
    I = intensity.flatten()
    I_hat = fit_model(ra_offset, dec_offset).flatten()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title("2D Moffat fit")

    inax = ax.inset_axes((0, 0.65, 1, 0.35))
    inax.plot(r, np.abs(I - I_hat) / I, 'k+')
    inax.grid(visible=True)
    inax.set_ylabel(r'$\frac{I-\hat{I}}{I}$', fontsize=17)
    inax.set_xlabel(r'$|r-\hat{r}_0|$ (arcsec)', fontsize=15)
    inax.set_ylim(-0.099, 2)
    # Input data
    inax = ax.inset_axes((0, 0.0, 0.5, 0.5))
    c = inax.pcolormesh(ra_offset, dec_offset,
                        np.log10(intensity), cmap='nipy_spectral')
    inax.plot(fit_model.x_0.value, fit_model.y_0.value, 'k+', ms=14, mew=2,
              label=f'centre ({fit_model.x_0.value:.1f}, {fit_model.y_0.value:.1f}) arcsec')
    plt.colorbar(mappable=c, ax=inax, orientation='horizontal', anchor=(0, -1),
                 label=r"$\log_{10}(I)$")
    inax.legend()
    inax.set_ylabel("DEC offset (arcsec)")
    inax.set_xlabel("RA offset (arcsec)")
    
    inax = ax.inset_axes((0.55, 0.0, 0.5, 0.5))
    c = inax.pcolormesh(ra_offset, dec_offset,
                        np.log10(intensity/I_hat.reshape(intensity.shape)),
                    vmin=-.5, vmax=.5, cmap='seismic')
    inax.plot(fit_model.x_0.value, fit_model.y_0.value, 'k+', ms=14, mew=2)
    plt.colorbar(mappable=c, ax=inax, orientation='horizontal', anchor=(0, -1),
                 label=r"$\log_{10}(I/\hat{I})$")
    inax.set_ylabel("DEC offset (arcsec)")
    inax.set_xlabel("RA offset (arcsec)")

    ax.axis("off")

    return fig


# =============================================================================
# Registration
# =============================================================================

def qc_registration(rss_list, **kwargs):
    n_rss = len(rss_list)
    fig, axs = plt.subplots(nrows=1, ncols=n_rss+1,
                            figsize=(4*(n_rss + 1), 4),
                            gridspec_kw=dict(hspace=0.50))
    axs[0].set_title("Input RSS overlap")
    axs[0].set_xlabel("RA (deg)")
    axs[0].set_ylabel("DEC (deg)")

    cmap = plt.get_cmap('jet', n_rss)
    for i, rss in enumerate(rss_list):
        axs[0].scatter(rss.info['ori_fib_ra'] / 3600 + rss.info['ori_cen_ra'],
                       rss.info['ori_fib_dec'] / 3600 + rss.info['ori_cen_dec'],
                       marker='o', ec=cmap(i / (n_rss - 1)), c='none',
                   #alpha=1/n_rss
                   )
        axs[i+1].set_title(f"Re-centered RSS-{i+1}" + "\n"
                           f"(ra, dec) shift: {rss.info['cen_ra'] * 3600:.2f}, {rss.info['cen_dec'] * 3600:.2f} ('')")
        axs[i+1].scatter(rss.info['fib_ra'],
                   rss.info['fib_dec'],
                   c=np.nansum(rss.intensity, axis=1),
                   norm=colors.LogNorm(),
                   marker='o', cmap='Greys_r',
                   )
        axs[i+1].axvline(0, c=cmap(i / (n_rss - 1)), lw=1.5)
        axs[i + 1].axhline(0, c=cmap(i / (n_rss - 1)), lw=1.5)
        axs[i + 1].set_xlabel("RA Offset (arcsec)")
        axs[i + 1].set_ylabel("DEC Offset (arcsec)")

    return fig

def qc_registration_crosscorr(images_list, cross_corr_results):
    """TODO..."""
    # Set reference points to illustrate the shifts

    vmin, vmax = np.nanpercentile(images_list, [5, 95])
    imargs = dict(vmin=vmin, vmax=vmax, cmap='viridis', interpolation='none')

    fig, axs = plt.subplots(nrows=1, ncols=len(images_list),
                          figsize=(4 * len(images_list), 4))
    plt.suptitle("QC cross-correlation-based image registration")
    
    axs[0].set_title("Reference image")
    mappable = axs[0].imshow(images_list[0], **imargs)

    # Reference image centre will be used as visual reference
    centre = (images_list[0].shape[1] / 2, images_list[0].shape[0] / 2)
    axs[0].plot(*centre, 'r+')
    # Plot the rest of images
    for i, im in enumerate(images_list[1:]):
        shift = cross_corr_results[i][0]
        ax = axs[i + 1]
        mappable = ax.imshow(images_list[i + 1], **imargs)
        ax.arrow(*centre, - shift[1], - shift[0], color='r', width=1)
        ax.plot(centre[0] - shift[1], centre[1] - shift[0])
        ax.annotate(f"Shif (row, col): {shift[0]:.2f}, {shift[1]:.2f}", xy=(0.05, 0.95),
                    color='red',
                    xycoords='axes fraction', va='top', ha='left')
    cax = ax.inset_axes((1.05, 0, 0.05, 1))
    plt.colorbar(mappable, cax=cax)
    plt.close(fig)
    return fig

def qc_registration_centroids(images_list, wcs_list, offsets, ref_pos):
    """TODO..."""
    # Account for images with different sizes
    vmin, vmax = np.nanpercentile(
        np.hstack([im.flatten().value for im in images_list]), [5, 95])
    imargs = dict(vmin=vmin, vmax=vmax, cmap='viridis', interpolation='none')

    ncols=len(images_list)
    fig = plt.Figure(figsize=(4 * ncols, 4))
    plt.suptitle("QC centroid-based image registration")
    for i in range(ncols):
        ax = fig.add_subplot(1, ncols, i + 1 , projection=wcs_list[i])
    
        mappable = ax.imshow(images_list[i].value, **imargs)
        ax.scatter(ref_pos.ra, ref_pos.dec, marker='*',
                   ec='r', label='Reference', transform=ax.get_transform('world'))
        ax.scatter(ref_pos.ra - offsets[i][0], ref_pos.dec - offsets[i][1], marker='o',
                   ec='k', label='Centroid', transform=ax.get_transform('world'))

        ax.annotate(f"Offset (ra, dec):\n {offsets[i][0].to('arcsec').value:.2f}, {offsets[i][1].to('arcsec').value:.2f} arcsec",
                    xy=(0.05, 0.95),
                    color='red',
                    xycoords='axes fraction', va='top', ha='left')
    ax.legend(bbox_to_anchor=(0.5, 1.1), loc='lower center')
    cax = ax.inset_axes((1.05, 0, 0.05, 1))
    plt.colorbar(mappable, cax=cax)
    plt.close(fig)
    return fig

# =============================================================================
# Astrometry
# =============================================================================


def qc_external_image(ref_image, ref_wcs, external_image, external_image_wcs):
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111, projection=external_image_wcs)

    contourf_params = dict(cmap='Spectral', levels=[18, 19, 20, 21, 22, 23],
                               vmin=19, vmax=23, extend='both')
    contour_params = dict(levels=[18, 19, 20, 21, 22, 23],
                          colors='k')

    ax.coords.grid(True, color='orange', ls='solid')
    ax.coords[0].set_format_unit('deg')
    mappable = ax.contourf(external_image, **contourf_params)
    plt.colorbar(mappable, ax=ax,
                    label=r"$\rm \log_{10}(F_\nu / 3631 Jy / arcsec^2)$")
    ax.contour(ref_image,
                transform=ax.get_transform(ref_wcs), **contour_params)
    plt.close(fig)
    return fig

# =============================================================================
# Flux calibration
# =============================================================================

def qc_stellar_extraction():
    pass
