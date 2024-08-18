import os
import numpy as np
import copy
from astropy.io import fits

from pykoala import vprint
from pykoala.corrections.correction import CorrectionBase
from pykoala.rss import RSS
from pykoala.ancillary import flux_conserving_interpolation


class WavelengthOffset(object):
    """Wavelength offset class.

    This class stores a 2D wavelength offset.

    Attributes
    ----------
    offset_data : wavelength offset, in pixels
    offset_error : standard deviation of `offset_data`
    """
    offset_data = None
    offset_error = None

    def __init__(self, path=None, offset_data=None, offset_error=None):
        self.path = path
        self.offset_data = offset_data
        self.offset_error = offset_error

    def tofits(self, output_path=None):
        if output_path is None:
            if self.path is None:
                raise NameError("Provide output path")
            else:
                output_path = self.path
        primary = fits.PrimaryHDU()
        data = fits.ImageHDU(data=self.offset_data, name='OFFSET')
        error = fits.ImageHDU(data=self.offset_error, name='OFFSET_ERR')
        hdul = fits.HDUList([primary, data, error])
        hdul.writeto(output_path, overwrite=True)
        hdul.close(verbose=True)
        vprint(f"Wavelength offset saved at {output_path}")

    @classmethod
    def from_fits(cls, path):
        """Load the offset data from a fits file.

        Loads offset values (extension 1) and
        associated errors (extension 2) from a fits file.
        """
        if not os.path.isfile(path):
            raise NameError(f"offset file {path} does not exist.")
        vprint(f"Loading wavelength offset from {path}")
        with fits.open(path) as hdul:
            offset_data = hdul[1].data
            offset_error = hdul[2].data
        return cls(offset_data=offset_data, offset_error=offset_error,
                   path=path)


class WavelengthCorrection(CorrectionBase):
    """Wavelength correction class.

    This class accounts for the relative wavelength offset between fibres.

    Attributes
    ----------
    name : str
        Correction name, to be recorded in the log.
    offset : WavelengthOffset
        2D wavelength offset (n_fibres x n_wavelengths)
    verbose: bool
        False by default.
    """
    name = "WavelengthCorrection"
    offset = None
    verbose = False

    def __init__(self, offset_path=None, offset=None, **correction_kwargs):
        super().__init__(**correction_kwargs)

        self.path = offset_path
        self.offset = offset

    @classmethod
    def from_fits(cls, path):
        return cls(offset=WavelengthOffset.from_fits(path=path),
                   offset_path=path)

    def apply(self, rss):
        """Apply a 2D wavelength offset model to a RSS.

        Parameters
        ----------
        rss : RSS
            Original Row-Stacked-Spectra object to be corrected.

        Returns
        -------
        RSS
            Corrected RSS object.
        """

        assert isinstance(rss, RSS)

        rss_out = copy.deepcopy(rss)
        x = np.arange(rss.wavelength.size)
        for i in range(rss.intensity.shape[0]):
            rss_out.intensity[i] = flux_conserving_interpolation(
                x, x - self.offset.offset_data[i], rss.intensity[i])

        self.record_correction(rss_out, status='applied')
        return rss_out

# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
