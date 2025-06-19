import unittest
import os
import numpy as np

from astropy import units as u

from pykoala.instruments.mock import mock_rss
from pykoala.corrections.flux_calibration import FluxCalibration

class TestFluxCalibration(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting up RSS objects for testing")
        self.rss = mock_rss(ra_cen=180 << u.deg,
                              dec_cen=45 << u.deg,
                              source_kwargs={"source_ra": 180 << u.deg,
                                             "source_dec" :45 << u.deg})

    def test_factory(self):
        response_wavelength = np.arange(4000, 9000, 1) << u.AA
        response = np.ones(response_wavelength.size) << u.erg / u.s / u.adu
        response_error = np.ones(response_wavelength.size) << u.erg / u.s / u.adu

        # Save the response file
        np.savetxt("response_file.dat",
                   np.array([response_wavelength, response, response_error]).T,
                   header="Spectral Response curve\n"
                    + f" wavelength ({response_wavelength.unit}),"
                    + f" R ({response.unit}), Rerr ({response_error.unit})"
                   )

        fluxcal_corr = FluxCalibration.from_text_file("response_file.dat")
        os.unlink("response_file.dat")

    def test_master_calibration(self):

        response_wavelength = np.arange(4000, 9000, 1) << u.AA
        response = np.ones(response_wavelength.size) << u.erg / u.s / u.adu
        response_error = np.ones(response_wavelength.size) << u.erg / u.s / u.adu

        flux_cal_1 = FluxCalibration(response=response, response_err=response_error,
                                     response_wavelength=response_wavelength)
        flux_cal_2 = FluxCalibration(response=response, response_err=response_error,
                                     response_wavelength=response_wavelength)
        
        master_flux_cal = FluxCalibration.master_flux_auto(
            [flux_cal_1, flux_cal_2], combine_method="mean")
        self.assertTrue(
            np.allclose(master_flux_cal.response_err.value, ( 1 / np.sqrt(2))),
            f"Combined error does not have expected value ({1 / np.sqrt(2)}): {master_flux_cal.response_err.value}")
    
    def test_apply(self):
        response_wavelength = np.arange(4000, 9000, 1) << u.AA
        response = np.ones(response_wavelength.size) << u.adu / (u.erg / u.s)
        response_error = np.ones(response_wavelength.size) << u.adu / (u.erg / u.s)

        flux_cal = FluxCalibration(response=response, response_err=response_error,
                                   response_wavelength=response_wavelength)
        corrected_rss = flux_cal.apply(self.rss)
        print(corrected_rss.intensity.unit)

if __name__ == "__main__":
    unittest.main()
