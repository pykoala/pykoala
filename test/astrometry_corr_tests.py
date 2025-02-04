import unittest
import os
import numpy as np

from astropy import units as u

from pykoala.instruments.mock import mock_rss
from pykoala.corrections.astrometry import AstrometryCorrection

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

    def test_centroids(self):
        offsets, fig = self.correction.register_centroids(
            self.rss_list, qc_plot=True, centroider='gauss')

        for offset in offsets:
            print("Offset (ra, dec) in arcsec: ",
            offset[0].to('arcsec'), offset[1].to('arcsec'))

        assert np.isclose(
            offsets[0][0].to_value("arcsec"), 0.0, atol=0.1) & np.isclose(
            offsets[0][1].to_value("arcsec"), 0.0, atol=0.1), "First RSS not registered properly"
        assert np.isclose(
            offsets[1][0].to_value("arcsec"), -3.0, atol=0.1) & np.isclose(
            offsets[1][1].to_value("arcsec"), 0.0, atol=0.1), "Second RSS not registered properly"
        assert np.isclose(
            offsets[2][0].to_value("arcsec"), -3.0, atol=0.1) & np.isclose(
            offsets[2][1].to_value("arcsec"), -3.0, atol=0.1), "Third RSS not registered properly"

        for rss, offset in zip(self.rss_list, offsets):
            self.correction.apply(rss, offset=offset)

    def test_crosscorr(self):
        #TODO: test cross-correlation method
        pass

    def test_external_crosscorr(self):
        #TODO: test crosscorrelation using external imaging data
        pass

if __name__ == "__main__":
    unittest.main()
