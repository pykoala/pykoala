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
    gs = fig.add_gridspec(3, 4, wspace=0.15, hspace=0.25)

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

    ax = fig.add_subplot(gs[1, :])
    fibre_idx = np.random.randint(low=0, high=throughput.shape[0], size=3)
    for idx in fibre_idx:
        ax.plot(throughput[idx], label='Fibre {}'.format(idx), lw=0.7, alpha=0.8)
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("Spectral pixel")
    ax.legend()

    ax = fig.add_subplot(gs[-1, :])
    wl_idx = np.random.randint(low=0, high=throughput.shape[1], size=3)
    for idx in wl_idx:
        ax.plot(throughput[:, idx].squeeze(),
                label='Wl column {}'.format(idx), lw=0.7,
                alpha=0.8)
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("Fibre number")
    ax.legend()
    return fig




