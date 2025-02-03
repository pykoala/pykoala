import unittest
import os
import numpy as np

from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits

from pykoala.data_container import RSS
from pykoala.instruments.mock import mock_rss


class TestRSS(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting up RSS for testing")
        self.rss = mock_rss(nan_frac=0.0)

    def test_attributes(self):
        """Test the existence of the main attributes of RSS objects."""
        assert self.rss.intensity is not None
        assert self.rss.variance is not None
        assert (
            self.rss.inverse_variance == 1 / self.rss.variance
            ).all()
        assert (
            self.rss.snr == self.rss.intensity / self.rss.variance**0.5
            ).all()

        self.assertTrue(
            (self.rss.rss_intensity == self.rss.intensity).all())
        self.assertTrue(
            (self.rss.rss_variance == self.rss.variance).all())

        assert self.rss.mask is not None
        assert self.rss.info is not None
        assert self.rss.history is not None
        assert self.rss.wcs is not None

        self.assertTrue(isinstance(self.rss.fibre_diameter,
                                   u.Quantity))
        assert self.rss.sky_fibres is not None
        assert self.rss.science_fibres is not None
        self.assertTrue(
            isinstance(self.rss.wcs, WCS))

    def test_io(self):
        """Test I/O methods of RSS."""
        print("Testing RSS I/O...")
        hdul = self.rss._to_hdul()
        self.assertTrue(isinstance(hdul, fits.HDUList))
        try:
            hdul["INTENSITY"]
        except KeyError as err:
            raise err
        
        try:
            hdul["VARIANCE"]
        except KeyError as err:
            raise err
        
        # Check units from header
        try:
            u.Unit(hdul["INTENSITY"].header["bunit"])
        except:
            raise ValueError("RSS BUNIT is not compatible with astropy units")

        print("Saving RSS to file test.fits")
        self.rss.to_fits(filename="test.fits", overwrite=True)
        # Check that an error is raised if tried to overwrite
        try:
            self.rss.to_fits(filename="test.fits")
        except OSError as e:
            self.assertTrue("overwrite=True" in str(e))

        # Now load the RSS from the FITS file
        print("Instanciating RSS from test.fits")
        rss = RSS.from_fits("test.fits")
        self.assertTrue((rss.intensity == self.rss.intensity).all())
        self.assertTrue((rss.variance == self.rss.variance).all())
        # Remove the file
        os.unlink("test.fits")

    def test_rss_methods(self):
        print("Testing RSS manipulation methods")
        integrated_int, integrated_var = self.rss.get_integrated_fibres()
        self.assertTrue(np.isfinite(integrated_int).all())
        self.assertTrue(np.isfinite(integrated_var).all())

        rss_footprint = self.rss.get_footprint()
        self.assertTrue(rss_footprint.shape == (4, 2))
        
        self.rss.update_coordinates(
            offset=(1 << u.arcsec, 1 << u.deg))
        print("All good!")

    def test_plotting_methods(self):
        print("Testint plotting methods")
        fig = self.rss.plot_rss_image()
        fig = self.rss.plot_mask()
        fig = self.rss.plot_fibres()
        print("All good!")

if __name__ == "__main__":
    unittest.main()
