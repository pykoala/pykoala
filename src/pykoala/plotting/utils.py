from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator

import numpy as np

from astropy.visualization import (MinMaxInterval, PercentileInterval,
                                   SqrtStretch, PowerStretch,
                                   ImageNormalize, AsymmetricPercentileInterval)


def new_figure(fig_name,
               figsize=None,
               nrows=1, ncols=1,
               sharex='col', sharey='row',
               gridspec_kw={'hspace': 0, 'wspace': 0}):
    """
    Close old version of the figure and create new one
    with default sizes and format.

    Parameters
    ----------
    figsize : tuple
        Figure size.
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    sharex : str/bool, optional
        Whether panels share the x axis. True, False, or 'col'
        (default; x-axis shared accross columns)
    sharey : str/bool, optional
        Whether panels share the y axis. True, False, or 'row'
        (default; y-axis shared accross rows)
    **gridspec_kw : dict, optinal
        Default sets height and width space to `{'hspace': 0, 'wspace': 0})`

    Returns
    -------
    fig : plt.Figure
    axes : ndarray(mpl.Axes)
    """

    plt.close(fig_name)

    if figsize is None:
        figsize = (9 + ncols, 4 + nrows)
    fig = plt.figure(fig_name, figsize=figsize)
    axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                        sharex=sharex, sharey=sharey,
                        gridspec_kw=gridspec_kw,
                        )
    # fig.set_tight_layout(True)
    for ax in axes.flat:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True,
                       top=True, left=True, right=True)
        ax.tick_params(which='major', direction='inout',
                       length=8, grid_alpha=.3)
        ax.tick_params(which='minor', direction='in', length=2, grid_alpha=.1)
        ax.grid(True, which='both')

    fig.suptitle(fig_name)

    return fig, axes


default_cmap = plt.get_cmap("gist_earth").copy()
default_cmap.set_bad('gray')


def colour_map(fig, ax, cblabel, data,
               cmap=default_cmap,
               xlabel=None, x=None,
               ylabel=None, y=None,
               cbax=None, norm=None,
               norm_interval=AsymmetricPercentileInterval,
               interval_args={"lower_percentile": 1.0,
                              "upper_percentile": 99.0},
               stretch=PowerStretch, stretch_args={"a": 0.7}):
    """
    Plot a colour map (imshow) with axes and colour scales.

    Parameters
    ----------
    fig : plt.Figure
        Figure where the colour map will be drawn.
    ax : mpl.Axes
        Axes where the colour map will be drawn.
    cblabel : str
        Label of the colorbar
    data : ndarray
        2D array to be represented.
    cmap : str or mpl.colors.Norm
    norm : mpl.colors.Norm
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

    Returns
    -------
    im : mpl.AxesImage
    cb : mpl.Colorbar
    """
    if norm is None:
        interval = norm_interval(**interval_args)
        norm = ImageNormalize(data, interval=interval,
                              stretch=stretch(**stretch_args),
                              clip=False)
    elif isinstance(norm, str):
        norm = getattr(colors, norm)

    if y is None:
        y = np.arange(data.shape[0])
    if x is None:
        x = np.arange(data.shape[1])

    im = ax.imshow(data,
                   extent=(x[0]-(x[1]-x[0])/2, x[-1]+(x[-1]-x[-2])/2,
                           y[0]-(y[1]-y[0])/2, y[-1]+(y[-1]-y[-2])/2),
                   interpolation='nearest', origin='lower',
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
        cb.set_label(cblabel)
        cb.ax.tick_params(labelsize='small')

    return im, cb

def fibre_map(fig, ax, cblabel,
              rss, data,
              s=100, cmap=default_cmap, norm=None, cbax=None):
    """
    Plot a colour map of a physical magnitude defined on each fibre.

    Parameters
    ----------
    fig : plt.Figure
        Figure where the colour map will be drawn.
    ax : mpl.Axes
        Axes where the colour map will be drawn.
    cblabel : str
        Label of the colorbar
    rss : RSS
        Row-Stacked Spectra containing the fibre positions.
    data : ndarray
        1D array to be represented.
    cmap : str or mpl.colors.Norm
    norm : mpl.colors.Norm
    cbax: mpl.Axes
        Axes where the colour bar will be drawn.

    Returns
    -------
    im : mpl.AxesImage
    cb : mpl.Colorbar
    """

    if norm is None:
        percentiles = np.array([1, 16, 50, 84, 99])
        ticks = np.nanpercentile(data, percentiles)
        linthresh = np.median(data[data > 0])
        norm = colors.SymLogNorm(vmin=2*ticks[0]-ticks[1],
                                 vmax=2*ticks[-1]-ticks[-2],
                                 linthresh=linthresh)
    else:
        ticks = None

    s = np.prod(ax.bbox.size) / data.size / 2
    im = ax.scatter(rss.info['fib_ra'], rss.info['fib_dec'], c=data,
                    s=s, cmap=cmap, norm=norm)

    if cbax is None:
        cb = fig.colorbar(im, ax=ax, orientation='vertical', shrink=.9)
        cbax = cb.ax
    elif cbax is False:
        cb = None
    else:
        cb = fig.colorbar(im, cax=cbax, orientation='vertical')
    if cbax:
        cb.ax.yaxis.set_label_position("left")
        cb.set_label(cblabel)
        if ticks is not None:
            cb.ax.set_yticks(ticks=ticks, labels=[f'{value:.3g} ({percent}\\%)'
                                                  for value, percent in
                                                  zip(ticks, percentiles)])
        cb.ax.tick_params(labelsize='small')

    return im, cb

# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
