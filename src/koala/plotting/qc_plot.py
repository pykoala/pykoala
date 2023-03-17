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


def qc_cube(cube):
    """Create a QC plot for a Cube.

    Parameters
    ----------
    - cube: (Cube)

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
    mapax.imshow(np.nanmean(cube.intensity_corrected, axis=0), aspect='auto',
                 origin='lower', cmap='cividis')
    # Spectra -------
    pos_col = ['purple', 'orange', 'cyan']
    x_spaxel_idx = np.random.randint(low=0,
                                     high=cube.intensity_corrected.shape[1],
                                     size=3)
    y_spaxel_idx = np.random.randint(low=0,
                                     high=cube.intensity_corrected.shape[2],
                                     size=3)
    ax = fig.add_subplot(gs[1:3, :])
    for x_idx, y_idx, i in zip(x_spaxel_idx, y_spaxel_idx, range(3)):
        ax.plot(cube.wavelength, cube.intensity_corrected[:, x_idx, y_idx], lw=0.8,
                color=pos_col[i])
        mapax.plot(y_idx, x_idx, marker='+', ms=8, mew=2, lw=2, color=pos_col[i])
    for i, wl in enumerate(cube.wavelength[wl_spaxel_idx]):
        ax.axvline(wl, color=wl_col[i], zorder=-1, alpha=0.8)
    # ax.set_yscale('log')
    ax.set_xlabel(r"Wavelength ($\AA$)")
    # SNR ------------
    ax = fig.add_subplot(gs[3:5, :])
    for x_idx, y_idx, i in zip(x_spaxel_idx, y_spaxel_idx, range(3)):
        ax.plot(cube.wavelength,
                cube.intensity_corrected[:, x_idx, y_idx] / cube.variance_corrected[:, x_idx, y_idx]**0.5, lw=0.8,
                color=pos_col[i])

    # ax = fig.add_subplot(gs[0, :2])
    # ax.set_title('Collapsed cube')
    # ax.imshow(collapsed_cube, origin='lower', cmap='nipy_spectral',
    #           interpolation='none')
    #
    # ax = fig.add_subplot(gs[1, :2])
    # mappable = ax.imshow(np.nanmedian(sn_pixel, axis=0),
    #                      origin='lower', cmap='nipy_spectral',
    #                      interpolation='none', vmin=0)
    # plt.colorbar(mappable, ax=ax, label='median(SN/pixel)',
    #              orientation='horizontal')
    #
    # ax = fig.add_subplot(gs[1, 2:])
    # ax.set_title("Median SNR")
    # for i, p_ in enumerate(p_snr):
    #     ax.plot(cube.wavelength, p_, lw=0.5, label='P_{}'.format(pct[i]))
    # ax.set_yscale('log')
    # ax.set_ylabel('SRN / Pix')
    # ax.legend()
    # ax = fig.add_subplot(gs[0, 2:])
    # # ax.plot(cube.wavelength, collapsed_spectra, lw=0.5)
    # for p_ in p_spectra:
    #     ax.plot(cube.wavelength, p_, lw=0.5)
    # ax.set_yscale('log')
    # ax.set_ylabel('Flux')
    # fig.subplots_adjust(wspace=0.3, hspace=0.2)
    return fig



