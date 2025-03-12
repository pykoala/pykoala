import unittest
import os
import numpy as np
from astropy import units as u
from astropy.wcs import WCS

from pykoala.instruments.mock import mock_rss
from pykoala import cubing
from pykoala.plotting.utils import qc_cube
from pykoala.corrections.atmospheric_corrections import get_adr

np.random.seed(50)

class TestKernel(unittest.TestCase):
    """Unit tests for cheking the available interpolation kernels."""
    @classmethod
    def setUpClass(self):
        # Define a regular grid
        self.xx, self.yy = np.meshgrid(np.arange(0, 10), np.arange(0, 20))
        self.pix_size = 1

    def test_parabolic_kernel(self):
        kernel = cubing.ParabolicKernel(pixel_scale= 1 * u.arcsec / u.pixel,
                                        scale=1 * u.arcsec)

        # Check that the cumulative mass function is correct
        self.assertTrue(kernel.cmf(-1.0) == 0)
        self.assertTrue(kernel.cmf(0) == 0.5)
        self.assertTrue(kernel.cmf(1.0) == 1.0)


class TestStacking(unittest.TestCase):

    def test_cube_stacking(self):
        _cube_data = np.random.rand(3, 100, 20, 20)
        _var = (0.05 * _cube_data)**2

        t_sigma = cubing.CubeStacking.sigma_clipping(
            cubes = _cube_data, variances = _var)
        t_mad = cubing.CubeStacking.mad_clipping(
            cubes = _cube_data, variances = _var)

        #TODO: add asserts


class TestInterpolation(unittest.TestCase):
    pass


class TestCubing(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Setting up RSS objects for testing")
        # Assume that there are no astrometric issues
        source_kwargs=dict(source_ra=180 * u.deg,
                           source_dec=45 * u.deg)
        self.rss_1 = mock_rss(ra_cen=180 * u.deg,
                              dec_cen=45 * u.deg,
                              source_kwargs=source_kwargs)
        self.rss_2 = mock_rss(ra_cen=180 * u.deg - 10 * u.arcsec,
                              dec_cen=45 * u.deg - 2 * u.arcsec,
                              source_kwargs=source_kwargs)
        self.rss_3 = mock_rss(ra_cen=180 * u.deg - 25 * u.arcsec,
                              dec_cen=45 * u.deg + 10 * u.arcsec,
                              source_kwargs=source_kwargs)
        self.rss_list = [self.rss_1, self.rss_2, self.rss_3]

    def test_cubing(self, save=False):
        # Estimate the Differential Atmospheric Refraction
        adr_corr_set = []
        for rss in self.rss_list:
            adr_corr_set.append(
                [np.random.normal(0, 0.3, size=rss.wavelength.size) << u.arcsec,
                 np.random.normal(0, 0.3, size=rss.wavelength.size) << u.arcsec])
        # Build WCS from the list of RSS
        wcs = cubing.build_wcs_from_rss(self.rss_list,
                                        spatial_pix_size=1.0 << u.arcsec,
                                        spectra_pix_size=1.5 << u.AA)
        assert wcs.has_celestial, "WCS has not celestial axis"
        assert wcs.has_spectral, "WCS has not spectral axis"

        interpolator = cubing.CubeInterpolator(self.rss_list, wcs=wcs,
                                               kernel_scale=2.0,
                                               kernel=cubing.DrizzlingKernel,
                                               adr_set=adr_corr_set,
                                               qc_plots=True,
                                               keep_individual_cubes=True)
        cube = interpolator.build_cube()
        # Save the cube to a FITS file
        cube.to_fits(filename="test_cube.fits",
                     overwrite=True)
        # Load the cube from the FITS
        cube = cubing.Cube.from_fits("test_cube.fits")
        if save:
            # Save QC plots
            interpolator.cube_plots["stack_cube"].savefig(
                f"./cube_qcplot_cubing_test.png",
                bbox_inches="tight", dpi=200)
            interpolator.cube_plots["weights"].savefig(
                f"./weights_cubing_test.png",
                bbox_inches="tight", dpi=200)

            for rss_n, prod in interpolator.rss_inter_products.items():
                # Fibre-coverage QC plot
                prod["qc_fibres_on_fov"].savefig(
                    f"./{rss_n}_cubing_test.png",
                    bbox_inches="tight", dpi=200)
                # RSS individual cube
                rss_cube, rss_cube_fig = prod["cube"]
                rss_cube_fig.savefig(f"./single_{rss_n}_cube_qc.png",
                                bbox_inches="tight", dpi=200)
                rss_cube.to_fits(filename=f"{rss_n}.fits", overwrite=True)    
        else:
            os.unlink("test_cube.fits")

if __name__ == "__main__":
    #unittest.main()
    test = TestCubing()
    test.setUpClass()
    test.test_cubing(save=True)