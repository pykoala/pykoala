import unittest
import os

from astropy import units as u
from astropy.wcs import WCS

from pykoala.instruments.mock import mock_rss
from pykoala import cubing

class TestKernel(unittest.TestCase):
    pass

class TestInterpolation(unittest.TestCase):
    pass

class TestCubing(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting up RSS objects for testing")
        self.rss_1 = mock_rss(ra_cen=180 << u.deg,
                              dec_cen=45 << u.deg)
        self.rss_2 = mock_rss(ra_cen=180 << u.deg,
                              dec_cen=45 << u.deg)
        self.rss_3 = mock_rss(ra_cen=180 << u.deg,
                              dec_cen=45 << u.deg)
        self.rss_list = [self.rss_1, self.rss_2, self.rss_3]

    def test_cubing(self):
        # Build WCS from the list of RSS
        wcs = cubing.build_wcs_from_rss(self.rss_list,
                                        spatial_pix_size=0.5 << u.arcsec,
                                        spectra_pix_size=1 << u.AA)
        assert wcs.has_celestial, "WCS has not celestial axis"
        assert wcs.has_spectral, "WCS has not spectral axis"

        cube = cubing.build_cube(self.rss_list, wcs=wcs,
                                 kernel_size_arcsec=1.0)
        # Save the cube to a FITS file
        cube.to_fits(filename="test_cube.fits",
                     overwrite=True)
        # Load the cube from the FITS
        cube = cubing.Cube.from_fits("test_cube.fits")
        os.unlink("test_cube.fits")

if __name__ == "__main__":
    unittest.main()
