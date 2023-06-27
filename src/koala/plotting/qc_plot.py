import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import  GridSpec
import os

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
    - throughput: (np.ndarray)

    Returns
    -------
    - fig
    """
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
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("Spectral pixel")
    ax.legend(ncol=3)

    ax = fig.add_subplot(gs[-1, :])
    wl_idx = np.random.randint(low=0, high=throughput.shape[1], size=3)
    for idx in wl_idx:
        ax.plot(throughput[:, idx].squeeze(),
                label='Wave col {}'.format(idx), lw=0.7,
                alpha=0.8)
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("Fibre number")
    ax.legend(ncol=3)
    return fig


def qc_cube(cube, spax_pct=[50, 90, 99]):
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

    fig = plt.figure(figsize=(10, 10))
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
                  cmap='cividis')
    mapax = fig.add_subplot(gs[0, -1])
    mapax.set_title("Mean")
    mean_intensity = np.nanmean(cube.intensity_corrected, axis=0)
    mean_intensity[~np.isfinite(mean_intensity)] = 0
    mean_instensity_pos = np.argsort(mean_intensity.flatten())
    mapax.imshow(mean_intensity, aspect='auto',
                 origin='lower', cmap='cividis')
    # Spectra -------
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
    ax = fig.add_subplot(gs[1:3, :])
    for x_idx, y_idx, i in zip(x_spaxel_idx, y_spaxel_idx, range(3)):
        ax.plot(cube.wavelength, cube.intensity_corrected[:, x_idx, y_idx], lw=0.8,
                color=pos_col[i])
        mapax.plot(y_idx, x_idx, marker='+', ms=8, mew=2, lw=2, color=pos_col[i])
    for i, wl in enumerate(cube.wavelength[wl_spaxel_idx]):
        ax.axvline(wl, color=wl_col[i], zorder=-1, alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel("Flux")
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

def qc_moffat(intensity, x, y, fit_model):
    """#TODO"""

    r = np.sqrt((x-fit_model.x_0.value)**2 + (y-fit_model.y_0.value)**2)
    r = r.flatten()
    I = intensity.flatten()
    I_hat = fit_model(x, y).flatten()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title("2D Moffat fit")

    inax = ax.inset_axes((0, 0.65, 1, 0.35))
    inax.plot(r, np.abs(I - I_hat) / I, 'k+')
    inax.grid(visible=True)
    inax.set_ylabel(r'$\frac{I-\hat{I}}{I}$', fontsize=17)
    inax.set_xlabel(r'$|r-\hat{r}_0|$ (arcsec)', fontsize=15)
    inax.set_ylim(-0.099, 2)
    #inax.set_ylim(-1, 100)
    inax = ax.inset_axes((0, 0.0, 0.5, 0.5))
    c = inax.pcolormesh(x, y, np.log10(intensity), cmap='nipy_spectral')
    inax.plot(fit_model.x_0.value, fit_model.y_0.value, 'k+', ms=14, mew=2)
    plt.colorbar(mappable=c, ax=inax, orientation='horizontal', anchor=(0, -1),
                 label=r"$\log_{10}(I)$")
    inax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    inax = ax.inset_axes((0.55, 0.0, 0.5, 0.5))
    c = inax.pcolormesh(x, y, np.log10(intensity/fit_model(x, y)),
                    vmin=-.5, vmax=.5, cmap='seismic')
    inax.plot(fit_model.x_0.value, fit_model.y_0.value, 'k+', ms=14, mew=2)
    plt.colorbar(mappable=c, ax=inax, orientation='horizontal', anchor=(0, -1),
                 label=r"$\log_{10}(I/\hat{I})$")
    inax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.axis("off")

    return fig
# =============================================================================
# Registration
# =============================================================================

def qc_registration(rss_list, **kwargs):
    n_rss = len(rss_list)
    fig, axs = plt.subplots(nrows=1, ncols=n_rss+1,
                            figsize=(4*n_rss + 1, 4))
    axs[0].set_title("Input overlap")
    cmap = plt.get_cmap('jet', n_rss)
    for i, rss in enumerate(rss_list):
        axs[0].scatter(rss.info['ori_fib_ra_offset'] + rss.info['ori_cen_ra'] * 3600,
                   rss.info['ori_fib_dec_offset'] + rss.info['ori_cen_dec'] * 3600,
                   marker='o', ec=cmap(i / (n_rss - 1)), c='none',
                   #alpha=1/n_rss
                   )
        axs[i+1].scatter(rss.info['fib_ra_offset'],
                   rss.info['fib_dec_offset'],
                   c=np.nansum(rss.intensity_corrected, axis=1),
                   marker='o', cmap='Greys_r',
                   )
        axs[i+1].axvline(0, c='r', lw=0.5)
        axs[i + 1].axhline(0, c='r', lw=0.5)
        axs[i + 1].set_xlabel("RA Offset (arcsec)")
        axs[i + 1].set_xlabel("DEC Offset (arcsec)")
    return fig

def qc_registration_crosscorr(images_list, cross_corr_results):
    """TODO..."""
    # Set reference points to illustrate the shifts
    ref_points = np.array([
        [images_list[0].shape[0] / 2, images_list[0].shape[1] / 2],
        [0, 0],
        [0, images_list[0].shape[1]],
        [images_list[0].shape[0], 0],
        [images_list[0].shape[0], images_list[0].shape[1]]
        ])
    
    radial_points = (images_list[0].shape[0]**2
                     + images_list[0].shape[1]**2)**0.5 / 2 / np.array([8, 4, 2])
    vmin, vmax = np.nanpercentile(images_list, [5, 95])
    imargs = dict(vmin=vmin, vmax=vmax, cmap='viridis', interpolation='none')

    fig, axs = plt.subplots(nrows=1, ncols=len(images_list),
                          figsize=(4 * len(images_list), 4))

    
    axs[0].set_title("Reference image")
    mappable = axs[0].imshow(images_list[0], **imargs)
    # Reference points
    for point in ref_points:
        axs[0].plot(point[1], point[0], c='r', marker='+')
    for radius in radial_points:
        circle = plt.Circle((ref_points[0][1], ref_points[0][0]),
                             radius, color='r', fill=False)
        axs[0].add_patch(circle)
    # Plot the rest of images
    for i, im in enumerate(images_list[1:]):
        shift = cross_corr_results[i][0]
        ax = axs[i + 1]
        mappable = ax.imshow(images_list[i + 1], **imargs)
        for point in ref_points:
            ax.plot(point[1] - shift[1], point[0] - shift[0],
                        c='r', marker='+')
        for radius in radial_points:
            circle = plt.Circle((ref_points[0][1] - shift[1],
                                 ref_points[0][0] - shift[0]),
                                 radius, color='r',
                                fill=False)
            ax.add_patch(circle)
    cax = ax.inset_axes((1.05, 0, 0.05, 1))
    plt.colorbar(mappable, cax=cax)
    plt.close(fig)
    return fig

def qc_registration_centroids(images_list, centroids):
    """TODO..."""
    # Set reference points to illustrate the shifts
    ref_points = np.array([
        [images_list[0].shape[0] / 2, images_list[0].shape[1] / 2],
        [0, 0],
        [0, images_list[0].shape[1]],
        [images_list[0].shape[0], 0],
        [images_list[0].shape[0], images_list[0].shape[1]]
        ])
    
    radial_points = (images_list[0].shape[0]**2
                     + images_list[0].shape[1]**2)**0.5 / 2 / np.array([8, 4, 2])
    vmin, vmax = np.nanpercentile(images_list, [5, 95])
    imargs = dict(vmin=vmin, vmax=vmax, cmap='viridis', interpolation='none')

    fig, axs = plt.subplots(nrows=1, ncols=len(images_list),
                          figsize=(4 * len(images_list), 4))

    
    axs[0].set_title("Reference image")
    mappable = axs[0].imshow(images_list[0], **imargs)
    # Reference points
    for point in ref_points:
        axs[0].plot(point[1], point[0], c='r', marker='+')
    for radius in radial_points:
        circle = plt.Circle((ref_points[0][1], ref_points[0][0]),
                             radius, color='r', fill=False)
        axs[0].add_patch(circle)
    axs[0].plot(*centroids[0], '^', color='fuchsia')
    # Plot the rest of images
    for i, im in enumerate(images_list[1:]):
        shift = centroids[0] - centroids[i]
        ax = axs[i + 1]
        mappable = ax.imshow(images_list[i + 1], **imargs)
        for point in ref_points:
            ax.plot(point[1] - shift[1], point[0] - shift[0],
                        c='r', marker='+')
        for radius in radial_points:
            circle = plt.Circle((ref_points[0][1] - shift[1],
                                 ref_points[0][0] - shift[0]),
                                 radius, color='r',
                                fill=False)
            ax.add_patch(circle)
        ax.plot(*centroids[i], '^', color='fuchsia')
    cax = ax.inset_axes((1.05, 0, 0.05, 1))
    plt.colorbar(mappable, cax=cax)
    plt.close(fig)
    return fig

# =============================================================================
# Flux calibration
# =============================================================================

def qc_stellar_extraction():
    pass