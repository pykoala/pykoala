# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import copy
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
        err_func = np.nanstd

        normalized_fluxes = []
        for rss in rss_set:
            f = rss.intensity_corrected / rss.info['exptime']
            normalized_fluxes.append(f / np.nanmedian(f, axis=0)[np.newaxis, :])
        mean_throughput = stat_func(normalized_fluxes, axis=0)
        std_throughput = err_func(normalized_fluxes, axis=0)
        if clear_nan:
            x, y = np.meshgrid(np.arange(0, mean_throughput.shape[1]),
                               np.arange(0, mean_throughput.shape[0]))
            nan_mask = np.isfinite(mean_throughput)
            interpolator = NearestNDInterpolator(list(zip(x[nan_mask], y[nan_mask])),
                                                 mean_throughput[nan_mask])
            mean_throughput = interpolator(x, y)
            # Fill the nan values with the average error value
            std_throughput[~nan_mask] = np.nanmean(std_throughput)
        if smooth:
            raise NotImplementedError("Smoothing not implemented!")
        return mean_throughput, std_throughput

    def apply(self, throughput, rss, plot=True):
        """Apply a 2D throughput model to a RSS.

        Parameters
        ----------
        - throughput
        - rss: (RSS)
        - plot: (bool, optional, default=True)
        """
        if type(rss) is not RSS:
            raise ValueError("Throughput can only be applied to RSS data:\n input {}"
                             .format(type(rss)))
        # =============================================================================
        # Copy input RSS for storage the changes implemented in the task   
        # =============================================================================
        rss_out = copy.deepcopy(rss)

        rss_out.intensity_corrected = rss_out.intensity_corrected / throughput
        rss_out.variance_corrected = rss_out.variance_corrected / throughput ** 2
        rss_out.log[self.name] = "2D throughput applied"
        return rss_out
