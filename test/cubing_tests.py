import unittest
import os
import numpy as np
from astropy import units as u
from astropy.wcs import WCS

from pykoala.instruments.mock import mock_rss
from pykoala import cubing
from pykoala.data_container import FibreLSFModel, GaussianFibreLSFModel
from pykoala.plotting.utils import qc_cube

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


# ---------------------------------------------------------------------------
# LSF propagation tests
# ---------------------------------------------------------------------------

def _make_rss_with_lsf(ra_cen, dec_cen, n_fibres_1d=5, n_wave=50):
    """Helper: mock RSS with a GaussianFibreLSFModel attached."""
    source_kwargs = dict(source_ra=180 * u.deg, source_dec=45 * u.deg)
    rss = mock_rss(
        ra_n_fibres=n_fibres_1d,
        dec_n_fibres=n_fibres_1d,
        n_wave=n_wave,
        ra_cen=ra_cen,
        dec_cen=dec_cen,
        source_kwargs=source_kwargs,
    )
    n_fibres = rss.intensity.shape[0]
    lsf_wave_edges = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]) << u.AA
    sigma = np.full((n_fibres, n_wave), 1.5) << u.AA
    rss.lsf_model = GaussianFibreLSFModel(
        wavelength=rss.wavelength,
        sigma=sigma,
        lsf_wave_edges=lsf_wave_edges,
    )
    return rss


class TestLSFCubing(unittest.TestCase):
    """Tests for LSF propagation through CubeInterpolator."""

    @classmethod
    def setUpClass(cls):
        cls.rss_1 = _make_rss_with_lsf(180 * u.deg, 45 * u.deg)
        cls.rss_2 = _make_rss_with_lsf(180 * u.deg - 5 * u.arcsec, 45 * u.deg)
        cls.rss_list = [cls.rss_1, cls.rss_2]

    def _build_lsf_cube(self):
        wcs = cubing.build_wcs_from_rss(
            self.rss_list,
            spatial_pix_size=1.0 << u.arcsec,
            spectra_pix_size=50.0 << u.AA,
        )
        interpolator = cubing.CubeInterpolator(
            self.rss_list,
            wcs=wcs,
            kernel_scale=2.0 << u.arcsec,
            cube_lsf=True,
        )
        cube = interpolator.build_cube()
        return cube, interpolator

    def test_cube_lsf_attached(self):
        """Output cube carries a FibreLSFModel when cube_lsf=True."""
        cube, _ = self._build_lsf_cube()
        self.assertIsNotNone(cube.lsf_model)
        self.assertIsInstance(cube.lsf_model, FibreLSFModel)

    def test_cube_lsf_shape(self):
        """Stacked LSF kernel has the expected (n_spaxels, n_lsf_wave, n_kernel) shape."""
        cube, interp = self._build_lsf_cube()
        n_rows = interp.target_wcs.array_shape[1]
        n_cols = interp.target_wcs.array_shape[2]
        n_spaxels = n_rows * n_cols
        n_lsf_wave = interp.lsf_wavelength.size
        n_kernel = interp.all_lsf_num.shape[-1]
        self.assertEqual(
            cube.lsf_model.kernel.shape, (n_spaxels, n_lsf_wave, n_kernel)
        )

    def test_cube_lsf_kernel_normalized(self):
        """Every spaxel kernel row sums to 1 along the kernel axis."""
        cube, _ = self._build_lsf_cube()
        kernel = cube.lsf_model.kernel  # (n_spaxels, n_lsf_wave, n_kernel)
        norm = kernel.sum(axis=-1)      # (n_spaxels, n_lsf_wave)
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_cube_lsf_wavelength_matches_reference(self):
        """Cube LSF wavelength grid equals the reference RSS LSF wavelength grid."""
        cube, interp = self._build_lsf_cube()
        np.testing.assert_allclose(
            cube.lsf_model.wavelength.to_value(u.AA),
            interp.lsf_wavelength.to_value(u.AA),
        )

    def test_cube_lsf_kernel_nonnegative(self):
        """All kernel values must be >= 0."""
        cube, _ = self._build_lsf_cube()
        self.assertTrue(np.all(cube.lsf_model.kernel >= 0))

    def test_cube_lsf_missing_rss_raises(self):
        """cube_lsf=True must raise ValueError when an RSS lacks lsf_model."""
        rss_no_lsf = mock_rss(
            ra_n_fibres=5, dec_n_fibres=5, n_wave=50,
            source_kwargs=dict(source_ra=180 * u.deg, source_dec=45 * u.deg),
        )
        rss_set = [self.rss_1, rss_no_lsf]
        wcs = cubing.build_wcs_from_rss(
            rss_set,
            spatial_pix_size=1.0 << u.arcsec,
            spectra_pix_size=50.0 << u.AA,
        )
        with self.assertRaises(ValueError):
            cubing.CubeInterpolator(rss_set, wcs=wcs, cube_lsf=True)

    def test_cube_lsf_disabled_by_default(self):
        """Output cube has no lsf_model when cube_lsf=False (default)."""
        wcs = cubing.build_wcs_from_rss(
            self.rss_list,
            spatial_pix_size=1.0 << u.arcsec,
            spectra_pix_size=50.0 << u.AA,
        )
        interpolator = cubing.CubeInterpolator(
            self.rss_list, wcs=wcs, kernel_scale=2.0 << u.arcsec,
        )
        cube = interpolator.build_cube()
        self.assertIsNone(cube.lsf_model)