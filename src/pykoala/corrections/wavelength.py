import os
import numpy as np
import copy
from astropy.io import fits

from pykoala.corrections.correction import CorrectionBase
from pykoala.rss import RSS
from pykoala.ancillary import flux_conserving_interpolation


class WavelengthOffset(object):
    """Wavelength offset class.

    Description
    -----------
    This class stores a 2D wavelength offset.

    Attributes
    ----------
    - `offset_data`: wavelength offset, in pixels
    - `offset_error`: standard deviation of `offset_data`
    """
    offset_data = None
    offset_error = None

    def __init__(self, path=None, offset_data=None, offset_error=None):
        self.path = path
        self.offset_data = offset_data
        self.offset_error = offset_error

        if self.path is not None and self.offset_data is None:
            self.load_fits()

    def tofits(self, output_path):
        primary = fits.PrimaryHDU()
        data = fits.ImageHDU(data=self.offset_data, name='OFFSET')
        error = fits.ImageHDU(data=self.offset_error, name='OFFSET_ERR')
        hdul = fits.HDUList([primary, data, error])
        hdul.writeto(output_path, overwrite=True)
        hdul.close(verbose=True)
        print(f"[wavelength offset] offset saved at {output_path}")

    def load_fits(self):
        """Load the offset data from a fits file.

        Description
        -----------
        Loads offset values (extension 1) and
        associated errors (extension 2) from a fits file.
        """
        if not os.path.isfile(self.path):
            raise NameError(f"offset file {self.path} does not exist.")
        print(f"[wavelength offset] Loading offset from {self.path}")
        with fits.open(self.path) as hdul:
            self.offset_data = hdul[1].data
            self.offset_error = hdul[2].data


class WavelengthCorrection(CorrectionBase):
    """Wavelength correction class.

    Description
    -----------
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

    def __init__(self, **kwargs):
        super().__init__()

        path = kwargs.get('offset_path', None)
        self.offset = kwargs.get('offset', WavelengthOffset(path=path))
        assert isinstance(self.offset, WavelengthOffset)

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

        self.log_correction(rss_out, status='applied')
        return rss_out

# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
