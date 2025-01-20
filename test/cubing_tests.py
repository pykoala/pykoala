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
        pass


if __name__ == "__main__":
    unittest.main()
