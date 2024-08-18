# =============================================================================
# Basics packages
# =============================================================================
from os import path
import numpy as np
import copy
from astropy.io import fits
from scipy.ndimage import median_filter, percentile_filter
# from scipy.ndimage import gaussian_filter
# =============================================================================
# Astropy and associated packages
# =============================================================================

# =============================================================================
# KOALA packages
# =============================================================================
# Modular
from pykoala import vprint
from pykoala.corrections.correction import CorrectionBase
from pykoala.rss import RSS
from pykoala import ancillary


class Throughput(object):
    def __init__(self, throughput_data, throughput_error,
                 throughput_file=None):
        self.throughput_data = throughput_data
        self.throughput_error = throughput_error
        self.throughput_file = throughput_file

    def to_fits(self, output_path=None):
        if output_path is None:
            if self.throughput_file is None:
                raise NameError("Provide output file name for saving throughput")
            else:
                output_path = self.throughput_file
        primary = fits.PrimaryHDU()
        thr = fits.ImageHDU(data=self.throughput_data,
                            name='THROU')
        thr_err = fits.ImageHDU(data=self.throughput_error,
                                name='THROUERR')
        hdul = fits.HDUList([primary, thr, thr_err])
        hdul.writeto(output_path, overwrite=True)
        hdul.close(verbose=True)
        vprint(f"Throughput saved at {output_path}")

    @classmethod
    def from_fits(cls, path):
        """Creates a `Throughput` from an input FITS file.
        
        Throughput data must be stored in extension 1, and
        associated errors in extension 2 of the HDUL.

        Parameters
        ----------
        - path: str
            Path to the FITS file containing the throughput data.
        """
        if not path.isfile(path):
            raise NameError(f"Throughput file {path} does not exists.")
        vprint(f"Loading throughput from {path}")
        with fits.open(path) as hdul:
            throughput_data = hdul[1].data
            throughput_error = hdul[2].data
        return cls(throughput_data, throughput_error, path)


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

    @classmethod
    def from_file(cls, path):
        """Creates a `ThroughputCorrection` using an input FITS file.
        
        Parameters
        ----------
        - path: str
            Path to the FITS file containing the throughput data.
        """
        throughput = Throughput.from_fits(path)
        return cls(throughput, path)

    @classmethod
    def from_rss(cls, rss_set, clear_nan=True, statistic='median', medfilt=5,
                 pct_outliers=[5, 95]):
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
            f[f <= 0] = np.nan
            fluxes.append(f)
        # Combine all RSS
        combined_flux = stat_func(fluxes, axis=0)
        combined_flux_err = np.nanstd(fluxes, axis=0) / np.sqrt(len(fluxes))

        # Normalize fibres
        reference_fibre = stat_func(combined_flux, axis=0)
        throughput_data = combined_flux / reference_fibre[np.newaxis, :]

        # Estimate the error
        throughput_error = combined_flux_err / reference_fibre[np.newaxis, :]

        p5, p95 = np.nanpercentile(throughput_data, pct_outliers, axis=1)
        outliers_mask = (throughput_data - p5[:, np.newaxis] <= 0
                         ) | (throughput_data - p95[:, np.newaxis] >= 0)
        throughput_data[outliers_mask] = np.nan

        if clear_nan:
            pix = np.arange(throughput_data.shape[1])
            for ith, (f, f_err) in enumerate(zip(throughput_data, throughput_error)):
                mask = np.isfinite(f) & np.isfinite(f_err)
                throughput_data[ith] = np.interp(pix, pix[mask], f[mask])
                throughput_error[ith] = np.interp(pix, pix[mask], f_err[mask])

        if medfilt is not None:
            print(f"Applying median filter (size={medfilt} px)")
            throughput_data = median_filter(throughput_data, size=medfilt,
                                            axes=1)
            throughput_error = median_filter(
                throughput_error**2, size=medfilt)**0.5

        throughput = Throughput(throughput_data=throughput_data,
                                throughput_error=throughput_error)
        return cls(throughput)


    def apply(self, rss):
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

        if not isinstance(rss, RSS):
            raise ValueError(
                "Throughput can only be applied to RSS data:\n input {}"
                .format(type(rss)))
        # =============================================================================
        # Copy input RSS for storage the changes implemented in the task
        # =============================================================================
        rss_out = copy.deepcopy(rss)

        rss_out.intensity = rss_out.intensity / self.throughput.throughput_data
        rss_out.variance = rss_out.variance / self.throughput.throughput_data**2
        self.record_correction(rss_out, status='applied')
        return rss_out

# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
