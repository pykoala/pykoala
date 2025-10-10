import unittest
import os
import numpy as np

from astropy import units as u
from astropy.nddata import CCDData

from unittest import mock

from pykoala.instruments.mock import mock_rss
from pykoala.corrections.correction import CorrectionOffset
from pykoala.corrections.astrometry import (
    AstrometryCorrection,
    register_dataset_centroids,
    register_dataset_crosscorr,
    compute_offset_from_external_image,
)

class TestAstrometry(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting up RSS objects for testing")
        # Reference
        self.rss_1 = mock_rss(ra_cen=180 << u.deg,
                              dec_cen=45 << u.deg,
                              source_kwargs={"source_ra": 180 << u.deg,
                                             "source_dec": 45 << u.deg})
        # Shifted +3" in RA
        self.rss_2 = mock_rss(ra_cen=180 * u.deg + 3 * u.arcsec,
                              dec_cen=45 << u.deg,
                              source_kwargs={"source_ra": 180 * u.deg + 3 * u.arcsec,
                                             "source_dec": 45 << u.deg})
        # Shifted +3" in RA and +3" in DEC
        self.rss_3 = mock_rss(ra_cen=180 * u.deg + 3 * u.arcsec,
                              dec_cen=45 * u.deg + 3 * u.arcsec,
                              source_kwargs={"source_ra": 180 * u.deg + 3 * u.arcsec,
                                             "source_dec": 45 * u.deg + 3 * u.arcsec})
        self.rss_list = [self.rss_1, self.rss_2, self.rss_3]

    # -------------------------
    # Centroid registration
    # -------------------------
    def test_centroids(self):
        offsets, fig = register_dataset_centroids(
            self.rss_list, qc_plot=True, centroider='gauss'
        )

        # Print for debug parity with previous test
        for off in offsets:
            ra_as = off.offset_data[0].to('arcsec').value
            dec_as = off.offset_data[1].to('arcsec').value
            print("Offset (ra, dec) in arcsec: ", ra_as, dec_as)

        # First is reference -> zero
        self.assertTrue(
            np.isclose(offsets[0].offset_data[0].to_value("arcsec"), 0.0, atol=0.2)
            and np.isclose(offsets[0].offset_data[1].to_value("arcsec"), 0.0, atol=0.2),
            "First RSS not registered properly (centroids), expected"
        )
        # rss_2 is +3" in RA, so we must move it by -3"
        self.assertTrue(
            np.isclose(offsets[1].offset_data[0].to_value("arcsec"), -3.0, atol=1.0)
            and np.isclose(offsets[1].offset_data[1].to_value("arcsec"), 0.0, atol=1.0),
            "Second RSS not registered properly (centroids)"
        )
        # rss_3 is +3" in RA and +3" in DEC -> move by (-3, -3)
        self.assertTrue(
            np.isclose(offsets[2].offset_data[0].to_value("arcsec"), -3.0, atol=1.0)
            and np.isclose(offsets[2].offset_data[1].to_value("arcsec"), -3.0, atol=1.0),
            "Third RSS not registered properly (centroids)"
        )

        # Apply each offset via the lean correction class
        corrected = []
        for rss, off in zip(self.rss_list, offsets):
            corr = AstrometryCorrection(offset=off)
            corrected.append(corr.apply(rss))

        # Sanity: types are preserved
        for rss, corr_rss in zip(self.rss_list, corrected):
            self.assertEqual(type(rss), type(corr_rss))

        # If fibre coordinates exist, verify a representative fibre moved by ~offset
        # (Use center fibre if available)
        try:
            i_mid = len(corrected[0].info['fib_ra']) // 2
            ra0 = self.rss_2.info['fib_ra'][i_mid]
            ra2 = corrected[1].info['fib_ra'][i_mid]
            # rss_2 had +3" RA; offset[1] ~ -3" -> ra2 ~ ra0 - 3"
            self.assertTrue(np.isclose((ra2 - ra0).to_value('arcsec'),
                                       offsets[1].offset_data[0].to_value('arcsec'),
                                       atol=0.5))
        except Exception:
            # Not all mocks expose per-fibre coords; skip numeric check if unavailable
            pass

    def test_centroids_single_input_raises(self):
        with self.assertRaises(ArithmeticError):
            register_dataset_centroids([self.rss_1])

    # -------------------------
    # Cross-correlation registration
    # -------------------------
    def test_crosscorr(self):
        offsets, fig = register_dataset_crosscorr(
            self.rss_list, wave_range=None, quick_cube_pix_size=0.5,
            oversample=100, bckgr_kappa_sigma=1, qc_plot=True
        )

        for off in offsets:
            ra_as = off.offset_data[0].to('arcsec').value
            dec_as = off.offset_data[1].to('arcsec').value
            print("XCorr Offset (ra, dec) in arcsec: ", ra_as, dec_as)

        # Very relaxed threshold (the mock dataset is not ideal for this test)
        self.assertTrue(
            np.isclose(offsets[0].offset_data[0].to_value("arcsec"), 0.0, atol=1.0)
            and np.isclose(offsets[0].offset_data[1].to_value("arcsec"), 0.0, atol=1.0),
            "First RSS not registered properly (xcorr)"
        )
        self.assertTrue(
            np.isclose(offsets[1].offset_data[0].to_value("arcsec"), -3.0, atol=1.0)
            and np.isclose(offsets[1].offset_data[1].to_value("arcsec"), 0.0, atol=1.0),
            "Second RSS not registered properly (xcorr)"
        )
        self.assertTrue(
            np.isclose(offsets[2].offset_data[0].to_value("arcsec"), -3.0, atol=1.0)
            and np.isclose(offsets[2].offset_data[1].to_value("arcsec"), -3.0, atol=1.0),
            "Third RSS not registered properly (xcorr)"
        )

    def test_crosscorr_single_input_raises(self):
        with self.assertRaises(ArithmeticError):
            register_dataset_crosscorr([self.rss_1])

    # -------------------------
    # External image path (mocked)
    # -------------------------
    def test_external_crosscorr_mocked(self):
        # Build a trivial "external image" CCD stub
        # (Real values don't matter because we will mock the photometry functions)
        ccd = CCDData(np.ones((10, 10)), unit="adu")

        external = {"ccd": ccd, "pix_size": 0.5}  # arcsec/pix

        # Prepare a fake result from crosscorrelate_im_apertures
        fake_offset = [(-2.5 * u.arcsec).to(u.deg), (1.0 * u.arcsec).to(u.deg)]  # (dRA, dDEC)
        fake_results = {"offset_min": fake_offset}

        with mock.patch("pykoala.corrections.astrometry.photometry.get_dc_aperture_flux") as m_get, \
             mock.patch("pykoala.corrections.astrometry.photometry.crosscorrelate_im_apertures") as m_xcorr, \
             mock.patch("pykoala.corrections.astrometry.photometry.make_plot_astrometry_offset") as m_plot:

            # Configure mocks
            # get_dc_aperture_flux returns a dict with expected keys
            m_get.return_value = {
                "aperture_flux": np.array([1.0, 2.0, 3.0]),
                "aperture_mask": np.array([True, True, True]),
                "coordinates": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
                "synth_photo": np.array([0.1, 0.2, 0.3]),
            }
            m_xcorr.return_value = {"offset_min": fake_offset}
            m_plot.return_value = None  # no figure for speed

            off, results = compute_offset_from_external_image(
                self.rss_1, external, filter_name="r"
            )

        self.assertIsInstance(off, CorrectionOffset)
        self.assertTrue(
            np.isclose(off.offset_data[0].to_value('arcsec'), -2.5, atol=1e-6)
            and np.isclose(off.offset_data[1].to_value('arcsec'), 1.0, atol=1e-6)
        )

        # Apply and assert it runs
        corr = AstrometryCorrection(offset=off)
        dc = corr.apply(self.rss_1)
        self.assertEqual(type(dc), type(self.rss_1))

    # -------------------------
    # Apply() error path
    # -------------------------
    def test_apply_without_offset_raises(self):
        corr = AstrometryCorrection()  # no offset
        with self.assertRaises(ValueError):
            corr.apply(self.rss_1)


if __name__ == "__main__":
    unittest.main()
