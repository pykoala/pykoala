import unittest
import os
import numpy as np

from astropy import units as u

from pykoala.instruments.mock import mock_rss
from pykoala.corrections.atmospheric_corrections import AtmosphericExtCorrection

class TestAtmExtCorr(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting up RSS objects for testing")
        self.rss = mock_rss(ra_cen=180 << u.deg,
                              dec_cen=45 << u.deg,
                              source_kwargs={"source_ra": 180 << u.deg,
                                             "source_dec" :45 << u.deg})
  
    def test_from_file(self):
        # Using a dummy null extinction curve of 2.5 magnitudes per airmass
        np.savetxt("./extinction_curve_model.dat",
                   np.array([self.rss.wavelength.to_value("AA"),
                             2.5 * np.ones(self.rss.wavelength.size, dtype=float)]).T)
        correction = AtmosphericExtCorrection.from_text_file(
            path="./extinction_curve_model.dat")        
        extinction_array = correction.extinction(self.rss.wavelength, airmass=1.0)
        self.assertTrue(np.alltrue(extinction_array == 10.0))

        os.unlink("./extinction_curve_model.dat")

    def test_apply(self):
        # Using the default model
        correction = AtmosphericExtCorrection.from_text_file()
        correction.apply(self.rss)

if __name__ == "__main__":
    unittest.main()
