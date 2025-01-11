import unittest
import os
import numpy as np

from astropy import units as u
from astropy.wcs import WCS

from pykoala.data_container import RSS
from pykoala.instruments.mock import mock_rss
from pykoala.corrections.astrometry import AstrometryCorrection
from pykoala.corrections.throughput import Throughput, ThroughputCorrection

class TestAstrometry(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting up RSS objects for testing")
        self.rss_1 = mock_rss(ra_cen=180 << u.deg,
                              dec_cen=45 << u.deg)
        self.rss_2 = mock_rss(ra_cen=180 * u.deg + 3 * u.arcsec,
                              dec_cen=45 << u.deg)
        self.rss_3 = mock_rss(ra_cen=180 * u.deg + 3 * u.arcsec,
                              dec_cen=45 * u.deg + 3 * u.arcsec)
        self.rss_list = [self.rss_1, self.rss_2, self.rss_3]
        self.correction = AstrometryCorrection()

    def test_correction(self):
        offsets, fig = self.correction.register_centroids(
            self.rss_list, qc_plot=True, centroider='gauss')

        for offset in offsets:
            print("Offset (ra, dec) in arcsec: ",
            offset[0].to('arcsec'), offset[1].to('arcsec'))

        for rss, offset in zip(self.rss_list, offsets):
            self.correction.apply(rss, offset=offset)

class TestThroughput(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting up RSS objects for testing")
        self.rss_1 = mock_rss()
        self.rss_2 = mock_rss()
        self.rss_3 = mock_rss()
        self.rss_list = [self.rss_1, self.rss_2, self.rss_3]
    
    def test_correction(self):
        throughput_corr = ThroughputCorrection.from_rss(
            self.rss_list, clear_nan=True, medfilt=10)

    def test_io(self):
        pass


if __name__ == "__main__":
    unittest.main()
