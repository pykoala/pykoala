# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import copy
from astropy.io import fits
from scipy.interpolate import NearestNDInterpolator
# =============================================================================
# Astropy and associated packages
# =============================================================================

# =============================================================================
# KOALA packages
# =============================================================================
# Modular
from koala.corrections.correction import CorrectionBase
from koala.rss import RSS


class Throughput(CorrectionBase):
    """
    Throughput correction class.

    This class accounts for the relative flux loss due to differences on the fibre efficiencies.

    Attributes
    ----------
    - name
    -
    """
    name = "ThroughputCorrection"
    throughput = None
    verbose = False

    def __init__(self, **kwargs):
        super().__init__()
        self.throughput = kwargs.get('throughput', None)
        self.throughput_path = kwargs.get('throughput_path', None)
        if self.throughput_path is not None:
            self.load_throughput(self.throughput_path)

    @staticmethod
    def create_throughput_from_flat(rss_set, clear_nan=True,
                                    statistic='median',
                                    smooth=False):
        """Compute the throughput function from a set of flat exposures.

        Given a set of flat exposures, this method will estimate the average
        efficiency of each fibre.

        Parameters
        ----------
        - rss_set: (list) List of RSS data.
        - clean_nan: (bool, optional, default=True) If True, nan values will be replaced
        by a nearest neighbour interpolation.
        - statistic: (str, optional, default='median') Set to 'median' or 'mean'
        to compute the throughput function.
        - smooth: (bool, optional, default=False) Apply a smoothing gaussian function
        to the throughput solution.
        """
        if statistic == 'median':
            stat_func = np.nanmedian
        elif statistic == 'mean':
            stat_func = np.nanmean

        fluxes = []
        for rss in rss_set:
            f = rss.intensity_corrected / rss.info['exptime']
            fluxes.append(f)
        # Combine
        combined_throughput = stat_func(fluxes, axis=0)

        # Normalize flat
        throughput = combined_throughput / stat_func(
            combined_throughput, axis=0)[np.newaxis, :]
        if clear_nan:
            x, y = np.meshgrid(np.arange(0, throughput.shape[1]),
                               np.arange(0, throughput.shape[0]))
            nan_mask = np.isfinite(throughput)
            interpolator = NearestNDInterpolator(list(zip(x[nan_mask], y[nan_mask])),
                                                 throughput[nan_mask])
            throughput = interpolator(x, y)
        if smooth:
            raise NotImplementedError("Smoothing not implemented!")
        return throughput, None

    def load_throughput(self, path, extension=1):
        """Load a throughput map from a FITS file."""
        self.throughput_path = path
        self.corr_print("Loading throughput from: ", self.throughput_path)
        with fits.open(self.throughput_path) as f:
            self.throughput = f[extension].data.copy()

    def apply(self, rss, throughput=None, plot=True):
        """Apply a 2D throughput model to a RSS.

        Parameters
        ----------
        - throughput
        - rss: (RSS)
        - plot: (bool, optional, default=True)
        """
        
        if throughput is None and self.throughput is not None:
            throughput = self.throughput
        else:
            raise RuntimeError("Throughput not provided!")
            
        if type(rss) is not RSS:
            raise ValueError("Throughput can only be applied to RSS data:\n input {}"
                             .format(type(rss)))
        # =============================================================================
        # Copy input RSS for storage the changes implemented in the task   
        # =============================================================================
        rss_out = copy.deepcopy(rss)

        rss_out.intensity_corrected = rss_out.intensity_corrected / throughput
        rss_out.variance_corrected = rss_out.variance_corrected / throughput ** 2
        self.log_correction(rss, status='applied')
        return rss_out
