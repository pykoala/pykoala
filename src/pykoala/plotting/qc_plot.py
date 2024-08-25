import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import  GridSpec
from matplotlib.collections import PatchCollection
import os

from pykoala.corrections.throughput import Throughput

plt.style.use('dark_background')

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

    throughput_data = throughput.throughput_data

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 4, wspace=0.15, hspace=0.35)

    ax = fig.add_subplot(gs[0, 0:-1])
    mappable = ax.imshow(throughput_data, origin='lower', cmap=throughput_cmap,
                         vmin=0.8, vmax=1.2, aspect='auto')
    plt.colorbar(mappable, ax=ax)
    ax.set_xlabel("wavelength axis")
    ax.set_ylabel("Fibre")

    ax = fig.add_subplot(gs[0, -1])
    ax.hist(throughput_data.flatten(), bins=throughput_data.size // 1000,
            range=[0.5, 1.5],
            log=True)
    ax.set_ylabel("N pixels")
    ax.set_xlabel("Throughput value")
    ax.set_ylim(10, throughput_data.size // 100)

    ax = fig.add_subplot(gs[1, :])

    median_wavelength_throughput = np.nanmedian(throughput_data, axis=0)
    std_wavelength_throughput = np.nanmedian(
        np.abs(throughput_data - median_wavelength_throughput[np.newaxis, :]),
        axis=0) * 1.4826
    ax.fill_between(np.arange(0, throughput_data.shape[1]),
                    median_wavelength_throughput - std_wavelength_throughput,
                    median_wavelength_throughput + std_wavelength_throughput,
                    alpha=0.3, color='r', label='Median +/- (MAD * 1.4826)')
    ax.plot(median_wavelength_throughput, label='Median',
            lw=0.7, color='r')

    fibre_idx = np.random.randint(low=0, high=throughput_data.shape[0], size=3)
    for idx in fibre_idx:
        ax.plot(throughput_data[idx], label='Fibre {}'.format(idx),
                lw=1., alpha=0.8)
    ax.set_ylim(0.75, 1.25)
    ax.set_xlabel("Spectral pixel")
    ax.legend(ncol=3)

    ax = fig.add_subplot(gs[-1, :])

    median_fibre_throughput = np.nanmedian(throughput_data, axis=1)
    std_fibre_throughput = np.nanmedian(
        np.abs(throughput_data - median_fibre_throughput[:, np.newaxis]),
        axis=1) * 1.4826
    ax.fill_between(np.arange(0, throughput_data.shape[0]),
                    median_fibre_throughput - std_fibre_throughput,
                    median_fibre_throughput + std_fibre_throughput,
                    alpha=0.3, color='r', label='Median +/- (MAD * 1.4826)')
    ax.plot(median_fibre_throughput, label='Median',
            lw=0.7, color='r')

    wl_idx = np.random.randint(low=0, high=throughput_data.shape[1], size=3)
    for idx in wl_idx:
        ax.plot(throughput_data[:, idx].squeeze(),
                label='Wave col {}'.format(idx), lw=0.7,
                alpha=1.0)
    ax.set_ylim(0.75, 1.25)
    ax.set_xlabel("Fibre number")
    ax.legend(ncol=4)
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

    fig = plt.figure(figsize=(12, 12))
    print("[QCPLOT] Cube QC plot for: ", cube.info['name'])
    plt.suptitle(cube.info['name'])
    gs = fig.add_gridspec(5, 4, wspace=0.15, hspace=0.25)
    # Maps -----
    wl_spaxel_idx = np.sort(np.random.randint(low=0,
                                      high=cube.intensity.shape[0],
                                      size=3))
    wl_col = ['b', 'g', 'r']
    for wl_idx, i in zip(wl_spaxel_idx, range(3)):
        ax = fig.add_subplot(gs[0, i:i+1])
        ax.set_title(r"$\lambda@{:.1f}$".format(cube.wavelength[wl_idx]),
                     fontdict=dict(color=wl_col[i]))
        ax.imshow(cube.intensity[wl_idx], aspect='auto', origin='lower',
                  interpolation='none', cmap='cividis')
        ax = fig.add_subplot(gs[1, i:i+1])
        mappable = ax.imshow(cube.intensity[wl_idx] / cube.variance[wl_idx]**0.5,
        interpolation='none', aspect='auto', origin='lower', cmap='jet')
        plt.colorbar(mappable, ax=ax)
    
    mean_intensity = np.nanmedian(cube.intensity, axis=0)
    mean_variance = np.nanmedian(cube.variance, axis=0)
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
    units = 1.
    units_label = '(counts)'
    if cube.history is not None:
        entries = cube.history.find_record(title='FluxCalibration', comment='units',
                                      tag='correction')
        print("Enrties found:", entries)
        for e in entries:
            unit_str = e.to_str(title=False).strip("units")
            units = 1 / float(''.join(filter(str.isdigit, unit_str)))
            units_label = ''.join(filter(str.isalpha, unit_str))

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
    for x_idx, y_idx, i in zip(x_spaxel_idx, y_spaxel_idx, range(3)):
        ax.plot(cube.wavelength, cube.intensity[:, x_idx, y_idx] * units, lw=0.8,
                color=pos_col[i])
        mapax.plot(y_idx, x_idx, marker='+', ms=8, mew=2, lw=2, color=pos_col[i])
    for i, wl in enumerate(cube.wavelength[wl_spaxel_idx]):
        ax.axvline(wl, color=wl_col[i], zorder=-1, alpha=0.8)
    ax.axhline(0, alpha=0.2, color='r')
    ylim = np.nanpercentile(
        cube.intensity[np.isfinite(cube.intensity)] * units, [40, 95])
    ylim[1] *= 20
    ylim[0] *= 0.1
    np.clip(ylim, a_min=0, a_max=None, out=ylim)
    ax.set_ylim(ylim)
    ax.set_yscale('symlog', linthresh=units * 0.1)
    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel("Flux " + units_label)

    # SNR ------------
    ax = fig.add_subplot(gs[3:5, :], sharex=ax)
    for x_idx, y_idx, i in zip(x_spaxel_idx, y_spaxel_idx, range(3)):
        ax.plot(cube.wavelength,
                cube.intensity[:, x_idx, y_idx] / cube.variance[:, x_idx, y_idx]**0.5, lw=0.8,
                color=pos_col[i],
                label=f"Spaxel rank={spax_pct[i]}")
    ax.legend()
    ax.set_ylabel("SNR/pix")
    ax.set_xlabel(r"Wavelength ($\AA$)")
    plt.close(fig)
    return fig

def qc_cubing(rss_weight_maps, exposure_times):
    """..."""
    if exposure_times.ndim == 1:
        exposure_times = (rss_weight_maps
        * exposure_times[:, np.newaxis, np.newaxis, np.newaxis])
    n_rss = rss_weight_maps.shape[0]

    p5, p95 = np.nanpercentile(rss_weight_maps, [.1 , 99.9])
    w_im_args = dict(vmin=p5 * 0.9, vmax=p95 * 1.1, cmap='nipy_spectral',
                     interpolation='none', origin='lower')

    tp5, tp95 = np.nanpercentile(exposure_times, [5 , 95])
    print("Mean exposure time: ", np.nanmean(exposure_times))
    t_im_args = dict(vmin=tp5 * 1.1, vmax=tp95 * 1.1, cmap='gnuplot',
                     interpolation='none', origin='lower')
    fig, axs = plt.subplots(ncols=n_rss + 1, nrows=2, sharex=True,
     sharey=True, constrained_layout=True)
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
    plt.colorbar(mappable, ax=axs[1, -1], label='Median exp. time / pixel')
    plt.close()
    return fig

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
    #plt.close()

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

def qc_registration_centroids(images_list, wcs_list, offsets, ref_pos):
    """TODO..."""
    # Account for images with different sizes
    vmin, vmax = np.nanpercentile(np.hstack([im.flatten() for im in images_list]), [5, 95])
    imargs = dict(vmin=vmin, vmax=vmax, cmap='viridis', interpolation='none')

    ncols=len(images_list)
    fig = plt.Figure(figsize=(4 * ncols, 4))
    plt.suptitle("QC centroid-based image registration")
    for i in range(ncols):
        ax = fig.add_subplot(1, ncols, i + 1 , projection=wcs_list[i])
    
        mappable = ax.imshow(images_list[i], **imargs)
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
