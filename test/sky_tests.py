import unittest
import os
import numpy as np
from astropy import units as u

from pykoala.instruments.mock import mock_rss
from pykoala.corrections.sky import BackgroundEstimator, SkyModel, SkyFromObject

class TestSkyCorrections(unittest.TestCase):
    
    def test_background_estimator(self):
        data = np.random.normal(size=(5, 10))
        for method in BackgroundEstimator.__dict__:
            if method[0] != '_':
                intensity, variance = BackgroundEstimator.__dict__[method](data)
                self.assertTrue(np.isfinite(intensity).all())
                self.assertTrue(np.isfinite(variance).all())
    
    def test_sky_model(self):
        wavelength=np.arange(6000., 9000., 1.) << u.Angstrom
        model = SkyModel(wavelength=wavelength, intensity=np.random.normal(size=wavelength.size) << u.adu)
        model.load_sky_lines()
        fig = model.plot_sky_model()
        # TODO: Are we going to use substract, substract_pca, remove_continuum, fit_emission_lines?

    def test_sky_from_object(self):
        sky_model = SkyFromObject(mock_rss())
        self.assertTrue(len(sky_model.qc_plots) > 0)
        for plot in sky_model.qc_plots:
            self.assertTrue(len(sky_model.qc_plots[plot].get_suptitle()) > 0)
        self.assertTrue(len(sky_model.plot_individual_wavelength(sky_model.wavelength[-1]).get_suptitle()) > 0)

if __name__ == "__main__":
    unittest.main()
