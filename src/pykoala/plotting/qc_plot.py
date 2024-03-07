import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import  GridSpec
import os

from pykoala.corrections.throughput import Throughput

#TODO: This is not working
plt.style.use(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'pykoala.mplstyle')
)

throughput_cmap = plt.cm.get_cmap('jet').copy()
throughput_cmap.set_extremes(bad='gray', under='black', over='fuchsia')


def qc_throughput(throughput):
    """Create a QC plot of a 2D throughput.

    Parameters
    ----------
    - throughput: (np.ndarray or throughput.Throughput)

    Returns
    -------
    - fig
    """
    if type(throughput) is Throughput:
        throughput = throughput.throughput_data

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 4, wspace=0.15, hspace=0.35)

    ax = fig.add_subplot(gs[0, 0:-1])
    mappable = ax.imshow(throughput, origin='lower', cmap=throughput_cmap,
                         vmin=0.8, vmax=1.2, aspect='auto')
    plt.colorbar(mappable, ax=ax)
    ax.set_xlabel("wavelength axis")
    ax.set_ylabel("Fibre")

    ax = fig.add_subplot(gs[0, -1])
    ax.hist(throughput.flatten(), bins=throughput.size // 1000, range=[0.5, 1.5],
            log=True)
    ax.set_ylabel("N pixels")
    ax.set_xlabel("Throughput value")
    ax.set_ylim(10, throughput.size // 100)

    ax = fig.add_subplot(gs[1, :])
    fibre_idx = np.random.randint(low=0, high=throughput.shape[0], size=3)
    for idx in fibre_idx:
        ax.plot(throughput[idx], label='Fibre {}'.format(idx), lw=0.7, alpha=0.8)
    ax.set_ylim(0.75, 1.25)
    ax.set_xlabel("Spectral pixel")
    ax.legend(ncol=3)

    ax = fig.add_subplot(gs[-1, :])
    wl_idx = np.random.randint(low=0, high=throughput.shape[1], size=3)
    for idx in wl_idx:
        ax.plot(throughput[:, idx].squeeze(),
                label='Wave col {}'.format(idx), lw=0.7,
                alpha=0.8)
    ax.set_ylim(0.75, 1.25)
    ax.set_xlabel("Fibre number")
    ax.legend(ncol=3)
    return fig


def qc_cube(cube, spax_pct=[75, 90, 99]):
    """Create a QC plot for a Cube.

    Parameters
    ----------
    - cube: (Cube)
    - spax_pct: #TODO
    Returns
    -------
    - fig: (plt.Figure)
    """
    # collapsed_cube = np.nanmean(cube.intensity_corrected, axis=0)
    # collapsed_spectra = np.nansum(cube.intensity_corrected, axis=(1, 2))
    # collapsed_variance = np.nansum(cube.variance_corrected, axis=(1, 2))
    # p_spectra = np.nanpercentile(cube.intensity_corrected, pct,
    #                              axis=(1, 2))
    #
    # sn_pixel = cube.intensity_corrected / cube.variance_corrected ** 0.5
    # p_snr = np.nanpercentile(sn_pixel, pct, axis=(1, 2))

    fig = plt.figure(figsize=(12, 12))
    print("[QCPLOT] Cube QC plot for: ", cube.info['name'])
    plt.suptitle(cube.info['name'])
    gs = fig.add_gridspec(5, 4, wspace=0.15, hspace=0.25)
    # Maps -----
    wl_spaxel_idx = np.sort(np.random.randint(low=0,
                                      high=cube.intensity_corrected.shape[0],
                                      size=3))
    wl_col = ['b', 'g', 'r']
    for wl_idx, i in zip(wl_spaxel_idx, range(3)):
        ax = fig.add_subplot(gs[0, i:i+1])
        ax.set_title(r"$\lambda@{:.1f}$".format(cube.wavelength[wl_idx]),
                     fontdict=dict(color=wl_col[i]))
        ax.imshow(cube.intensity_corrected[wl_idx], aspect='auto', origin='lower',
                  interpolation='none', cmap='cividis')
        ax = fig.add_subplot(gs[1, i:i+1])
        mappable = ax.imshow(cube.intensity_corrected[wl_idx] / cube.variance_corrected[wl_idx]**0.5,
        interpolation='none', aspect='auto', origin='lower', cmap='jet')
        plt.colorbar(mappable, ax=ax)
    
    mean_intensity = np.nanmedian(cube.intensity_corrected, axis=0)
    mean_variance = np.nanmedian(cube.variance_corrected, axis=0)
    mean_intensity[~np.isfinite(mean_intensity)] = 0
    mean_instensity_pos = np.argsort(mean_intensity.flatten())
    mapax = fig.add_subplot(gs[1, -1])
    mappable = mapax.imshow(mean_intensity / mean_variance**0.5, aspect='auto',
                 interpolation='none', origin='lower', cmap='jet')
    plt.colorbar(mappable, ax=mapax, label='SNR')
    mapax = fig.add_subplot(gs[0, -1])
    mapax.set_title("Median")
    mapax.imshow(mean_intensity, aspect='auto', interpolation='none',
                 origin='lower', cmap='cividis')
    # ------ Spectra -------
    if cube.log is not None and 'FluxCalibration' in cube.log.keys():
        units = 1 / float(cube.log['FluxCalibration']['units'])
        units_label = r"(erg/s/cm2/AA)"
    else:
        units = 1.
        units_label = '(ADU)'

    pos_col = ['purple', 'orange', 'cyan']
    x_spaxel_idx = np.random.randint(low=0,
                                     high=cube.intensity_corrected.shape[1],
                                     size=3)
    y_spaxel_idx = np.random.randint(low=0,
                                     high=cube.intensity_corrected.shape[2],
                                     size=3)
    spaxel_entries = mean_instensity_pos[
        np.array(mean_instensity_pos.size / 100 * np.array(spax_pct),
                 dtype=int)]
    x_spaxel_idx, y_spaxel_idx = np.unravel_index(spaxel_entries,
                                                  shape=mean_intensity.shape)
    ax = fig.add_subplot(gs[2:3, :])
    for x_idx, y_idx, i in zip(x_spaxel_idx, y_spaxel_idx, range(3)):
        ax.plot(cube.wavelength, cube.intensity_corrected[:, x_idx, y_idx] * units, lw=0.8,
                color=pos_col[i])
        mapax.plot(y_idx, x_idx, marker='+', ms=8, mew=2, lw=2, color=pos_col[i])
    for i, wl in enumerate(cube.wavelength[wl_spaxel_idx]):
        ax.axvline(wl, color=wl_col[i], zorder=-1, alpha=0.8)
    ax.axhline(0, alpha=0.2, color='r')
    ylim = np.nanpercentile(
        cube.intensity_corrected[np.isfinite(cube.intensity_corrected)] * units, [40, 60])
    ylim[1] *= 10
    ylim[0] *= 0.1
    ax.set_ylim()
    ax.set_yscale('symlog', linthresh=1)
    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel("Flux " + units_label)

    # SNR ------------
    ax = fig.add_subplot(gs[3:5, :], sharex=ax)
    for x_idx, y_idx, i in zip(x_spaxel_idx, y_spaxel_idx, range(3)):
        ax.plot(cube.wavelength,
                cube.intensity_corrected[:, x_idx, y_idx] / cube.variance_corrected[:, x_idx, y_idx]**0.5, lw=0.8,
                color=pos_col[i],
                label=f"Spaxel rank={spax_pct[i]}")
    ax.legend()
    ax.set_ylabel("SNR/pix")
    ax.set_xlabel(r"Wavelength ($\AA$)")
    plt.close(fig)
    return fig

def qc_cubing(cube, ):
    """..."""

    pass

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
        axs[0].scatter(rss.info['ori_fib_ra_offset'] / 3600 + rss.info['ori_cen_ra'],
                       rss.info['ori_fib_dec_offset'] / 3600 + rss.info['ori_cen_dec'],
                       marker='o', ec=cmap(i / (n_rss - 1)), c='none',
                   #alpha=1/n_rss
                   )
        axs[i+1].set_title(f"Re-centered RSS-{i+1}" + "\n"
                           f"(ra, dec) shift: {rss.info['cen_ra'] * 3600:.2f}, {rss.info['cen_dec'] * 3600:.2f} ('')")
        axs[i+1].scatter(rss.info['fib_ra_offset'],
                   rss.info['fib_dec_offset'],
                   c=np.nansum(rss.intensity_corrected, axis=1),
                   norm=LogNorm(),
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

def qc_registration_centroids(images_list, centroids):
    """TODO..."""
    # Account for images with different sizes
    vmin, vmax = np.nanpercentile(np.hstack([im.flatten() for im in images_list]), [5, 95])
    imargs = dict(vmin=vmin, vmax=vmax, cmap='viridis', interpolation='none')

    fig, axs = plt.subplots(nrows=1, ncols=len(images_list),
                          figsize=(4 * len(images_list), 4))

    plt.suptitle("QC centroid-based image registration")
    axs[0].set_title("Reference image")
    mappable = axs[0].imshow(images_list[0], **imargs)
    # Reference points

    axs[0].plot(*centroids[0], '+', color='fuchsia', label='Reference point')
    axs[0].legend()
    # Plot the rest of images
    for i, im in enumerate(images_list[1:]):
        shift = centroids[i + 1] - centroids[0]
        ax = axs[i + 1]
        mappable = ax.imshow(im, **imargs)
        ax.plot(centroids[0][0] + shift[0],
                centroids[0][1] + shift[1], '+', color='fuchsia')
        ax.arrow(*centroids[0], *shift, color='r', width=1)
        ax.annotate(f"Shif: {shift[0]:.2f}, {shift[1]:.2f}", xy=(0.05, 0.95),
                    color='red',
                    xycoords='axes fraction', va='top', ha='left')
    cax = ax.inset_axes((1.05, 0, 0.05, 1))
    plt.colorbar(mappable, cax=cax)
    plt.close(fig)
    return fig

# =============================================================================
# Flux calibration
# =============================================================================

def qc_stellar_extraction():
    pass
