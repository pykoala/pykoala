# =============================================================================
# Basics packages
# =============================================================================
from os import path
import numpy as np
import copy
from astropy.io import fits
from scipy.ndimage import median_filter
# from scipy.ndimage import gaussian_filter
# =============================================================================
# Astropy and associated packages
# =============================================================================

# =============================================================================
# KOALA packages
# =============================================================================
# Modular
from pykoala.corrections.correction import CorrectionBase
from pykoala.rss import RSS
from pykoala import ancillary


class Throughput(object):
    def __init__(self, path=None, throughput_data=None, throughput_error=None):
        self.path = path
        self.throughput_data = throughput_data
        self.throughput_error = throughput_error

        if self.path is not None and self.throughput_data is None:
            self.load_fits()

    def tofits(self, output_path):
        primary = fits.PrimaryHDU()
        thr = fits.ImageHDU(data=self.throughput_data,
                            name='THROU')
        thr_err = fits.ImageHDU(data=self.throughput_error,
                                name='THROUERR')
        hdul = fits.HDUList([primary, thr, thr_err])
        hdul.writeto(output_path, overwrite=True)
        hdul.close(verbose=True)
        print(f"[Throughput] Throughput saved at {output_path}")

    def load_fits(self):
        """Load the throughput data from a fits file.

        Loads throughput values (extension 1) and
        associated errors (extension 2) from a fits file.
        """
        if not path.isfile(self.path):
            raise NameError(f"Throughput file {self.path} does not exists.")
        print(f"[Throughput] Loading throughput from {self.path}")
        with fits.open(self.path) as hdul:
            self.throughput_data = hdul[1].data
            self.throughput_error = hdul[2].data


class ThroughputCorrection(CorrectionBase):
    """
    Throughput correction class.

    This class accounts for the relative flux loss due to differences
    on the fibre efficiencies.

    Attributes
    ----------
    - name
    -
    name : str
        Correction name, to be recorded in the log.
    throughput : Throughput
        2D fibre throughput (n_fibres x n_wavelengths).
    verbose: bool
        False by default.
    """
    name = "ThroughputCorrection"
    throughput = None

    def __init__(self, throughput=None, throughput_path=None,
                 **correction_kwargs):
        super().__init__(**correction_kwargs)

        if throughput is None:
            self.throughput = Throughput()
        else:
            assert isinstance(throughput, Throughput)
            self.throughput = throughput

        self.throughput.path = throughput_path
        if self.throughput.throughput_data is None and self.throughput.path is not None:
            self.throughput.load_fits(self.throughput_path)

    @staticmethod
    def create_throughput_from_rss(rss_set, clear_nan=True,
                                   statistic='median',
                                   medfilt=None):
        """Compute the throughput map from a set of flat exposures.

        Given a set of flat exposures, this method will estimate the average
        efficiency of each fibre.

        Parameters
        ----------
        - rss_set: (list)
            List of RSS data.
        - clean_nan: (bool, optional, default=True)
            If True, nan values will be replaced by a
            nearest neighbour interpolation.
        - statistic: (str, optional, default='median')
            Set to 'median' or 'mean' to compute the throughput function.
        - medfilt: (float, optional, default=None)
            If provided, apply a median filter to the throughput estimate.
        """
        if statistic == 'median':
            stat_func = np.nanmedian
        elif statistic == 'mean':
            stat_func = np.nanmean

        fluxes = []
        for rss in rss_set:
            f = rss.intensity / rss.info['exptime']
            fluxes.append(f)
        # Combine
        combined_flux = stat_func(fluxes, axis=0)
        combined_flux_err = np.nanstd(fluxes, axis=0) / np.sqrt(len(fluxes))

        # Normalize
        reference_fibre = stat_func(combined_flux, axis=0)
        throughput_data = combined_flux / reference_fibre[np.newaxis, :]

        # Estimate the error
        # throughput_error = np.nanstd(np.array(fluxes) / stat_func(fluxes, axis=1)[:, np.newaxis, :], axis=0)
        throughput_error = combined_flux_err / reference_fibre[np.newaxis, :]

        if clear_nan:
            print("Nearest neighbour interpolation to remove NaN values")
            throughput_data = ancillary.interpolate_image_nonfinite(
                throughput_data)
            throughput_error = ancillary.interpolate_image_nonfinite(
                throughput_error**2)**0.5
        if medfilt is not None:
            print(f"Applying median filter (size={medfilt})")
            throughput_data = median_filter(throughput_data, size=medfilt)
            throughput_error = median_filter(
                throughput_error**2, size=medfilt)**0.5

        throughput = Throughput(throughput_data=throughput_data,
                                throughput_error=throughput_error)
        return throughput

    def apply(self, rss, throughput=None, plot=True):
        """Apply a 2D throughput model to a RSS.

        Parameters
        ----------
        rss : RSS
            Original Row-Stacked-Spectra object to be corrected.
        throughput: Throughput
            Throughput object to be applied.
        plot : bool, optional, default=True

        Returns
        -------
        RSS
            Corrected RSS object.
        """

        if throughput is None and self.throughput is not None:
            throughput = self.throughput
        else:
            raise RuntimeError("Throughput not provided!")

        if type(throughput) is not Throughput:
            raise AttributeError(
                "Input throughput must be an instance of Throughput class")

        if type(rss) is not RSS:
            raise ValueError(
                "Throughput can only be applied to RSS data:\n input {}"
                .format(type(rss)))
        # =============================================================================
        # Copy input RSS for storage the changes implemented in the task
        # =============================================================================
        rss_out = copy.deepcopy(rss)

        rss_out.intensity = rss_out.intensity / throughput.throughput_data
        rss_out.variance = rss_out.variance / throughput.throughput_data**2
        self.record_history(rss_out, status='applied')
        return rss_out

# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
