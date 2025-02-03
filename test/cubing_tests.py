import unittest
import os
import numpy as np
from astropy import units as u
from astropy.wcs import WCS

from pykoala.instruments.mock import mock_rss
from pykoala import cubing
from pykoala.plotting.utils import qc_cube

class TestKernel(unittest.TestCase):
    """Unit tests for cheking the available interpolation kernels."""
    @classmethod
    def setUpClass(self):
        # Define a regular grid
        self.xx, self.yy = np.meshgrid(np.arange(0, 10), np.arange(0, 20))
        self.pix_size = 1

    def test_parabolic_kernel(self):
        kernel = cubing.ParabolicKernel(scale=1)

        # Check that the cumulative mass function is correct
        self.assertTrue(kernel.cmf(-1.0) == 0)
        self.assertTrue(kernel.cmf(0) == 0.5)
        self.assertTrue(kernel.cmf(1.0) == 1.0)


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
        # Build WCS from the list of RSS
        wcs = cubing.build_wcs_from_rss(self.rss_list,
                                        spatial_pix_size=1.0 << u.arcsec,
                                        spectra_pix_size=1.5 << u.AA)
        assert wcs.has_celestial, "WCS has not celestial axis"
        assert wcs.has_spectral, "WCS has not spectral axis"

        cube, interm_prod = cubing.build_cube(self.rss_list, wcs=wcs,
                                              kernel_size_arcsec=1.0,
                                              qc_plots=True,
                                              keep_individual_cubes=True)
        # Save the cube to a FITS file
        cube.to_fits(filename="test_cube.fits",
                     overwrite=True)
        # Load the cube from the FITS
        cube = cubing.Cube.from_fits("test_cube.fits")
        if save:
            interm_prod["stack_cube"].savefig(
                f"./cube_qcplot_cubing_test.png",
                bbox_inches="tight", dpi=200)
            for name, prod in interm_prod.items():
                if "rss" in name:
                    prod.savefig(f"./{name}_cubing_test.png",
                                bbox_inches="tight", dpi=200)
                elif "cube_" in name:
                    print(prod)
                    single_exp_cube, fig = prod
                    fig.savefig(f"./single_{name}_cube_qc.png",
                                bbox_inches="tight", dpi=200)
                    single_exp_cube.to_fits(filename=f"{name}.fits",
                     overwrite=True)
        else:
            os.unlink("test_cube.fits")

if __name__ == "__main__":
    #unittest.main()
    test = TestCubing()
    test.setUpClass()
    test.test_cubing(save=True)