import numpy as np
import pytest
from astropy import units as u
from scipy.ndimage import gaussian_filter1d

# Import the classes from your module
from pykoala.instruments.mock import mock_rss
from pykoala.corrections.wavelength import SolarCrossCorrOffset
from pykoala.data_container import FibreLSFModel, GaussianFibreLSFModel

# --------------------------
# Helpers
# --------------------------

def _apply_pixel_shift_in_index(values, shift_pix):
    """Shift by linear interpolation in pixel index space (approximation)."""
    n = values.size
    idx = np.arange(n, dtype=float)
    y = np.interp(idx, idx - float(shift_pix), values, left=np.nan, right=np.nan)
    # pad edges with nearest finite values to avoid NaNs in tests
    if np.isnan(y[0]):
        k = np.flatnonzero(~np.isnan(y))
        if k.size:
            y[0] = y[k[0]]
    if np.isnan(y[-1]):
        k = np.flatnonzero(~np.isnan(y))
        if k.size:
            y[-1] = y[k[-1]]
    return y


class MockSpectraContainer:
    def __init__(self, wavelength, rss_intensity, rss_variance):
        self.wavelength = wavelength
        self.rss_intensity = rss_intensity
        self.rss_variance = rss_variance
        self.intensity = type("U", (), {"unit": rss_intensity.unit})
        self.variance  = type("U", (), {"unit": rss_variance.unit})


# --------------------------
# Fixture: synthetic data built from the reference solar spectrum
# --------------------------

@pytest.fixture
def solar_based_data():
    """
    Build synthetic fibres directly from the same solar template used by
    SolarCrossCorrOffset, so the fitter sees the exact reference (modulo
    shift/sigma, response ripple, and noise).
    """
    rng = np.random.default_rng(123)

    # Get reference solar spectrum from the packaged FITS
    solar_correction = SolarCrossCorrOffset.from_fits()
    wave = solar_correction.sun_wavelength  # Quantity (n_wave,)
    sun  = solar_correction.sun_intensity   # Quantity (n_wave,)

    mask = (wave.value > 3800) & (wave.value < 6000)
    wave = wave[mask]
    sun = sun[mask]
    # Keep a compact test for speed
    # (Downsample if your template is huge; here we keep as-is for realism.)
    n_wave = wave.size
    n_fibre = 10

    # Inject per-fibre truth (modest to keep linearized behaviors)
    true_shift = np.linspace(-1.0, 1.0, n_fibre)      # pixels
    true_sigma = np.linspace(1.5, 2, n_fibre)       # pixels

    # Build fibres: convolve (sigma), shift (pixels), add ripple + noise
    data = np.zeros((n_fibre, n_wave), dtype=float)
    var  = np.zeros_like(data)

    # Use solar flux values (strip units for numerics)
    y0 = sun.to_value(sun.unit)

    for i in range(n_fibre):
        y = y0.copy()
        # broaden
        y = gaussian_filter1d(y, sigma=float(true_sigma[i]), mode="nearest")
        # shift (in pixel index)
        y = _apply_pixel_shift_in_index(y, shift_pix=float(true_shift[i]))
        # mild multiplicative response ripple (solver estimates/removes)
        # ripple = 1.0 + 0.02 * np.sin(2*np.pi*np.linspace(0, 3, n_wave))
        # y = y * ripple
        # noise
        noise = 0.01 * rng.normal(size=n_wave)
        data[i] = y + noise
        var[i]  = (0.01**2)

    # attach units consistent with SolarCrossCorrOffset internals
    iunit = sun.unit
    vunit = (sun.unit)**2
    rss_intensity = (data) * iunit
    rss_variance  = (var)  * vunit

    sc = MockSpectraContainer(wavelength=wave, rss_intensity=rss_intensity,
                               rss_variance=rss_variance)
    return solar_correction, sc, true_shift, true_sigma


# --------------------------
# Tests: SolarCrossCorrOffset
# --------------------------

def test_fit_solar_spectra_single_window(solar_based_data):
    solar_corr, sc, true_shift, true_sigma = solar_based_data

    # Single window => full bandpass
    res = solar_corr.fit_solar_spectra(
        sc,
        n_windows=1,
        response_window_size_aa=150,
        mask_tellurics=False,
    )

    assert "shift_pix_matrix" in res and "sigma_pix_matrix" in res
    assert res["shift_pix_matrix"].shape[1] == 1

    shifts = - res["shift_pix_matrix"][:, 0]
    sigmas = res["sigma_pix_matrix"][:, 0]

    # Sanity: finite & reasonably close to injected values
    assert np.isfinite(shifts).all()
    assert np.isfinite(sigmas).all()

    # These tolerances are generous—this is a quick regression guard, not a
    # performance benchmark. Tighten if your fitter is very accurate.
    print(sigmas, true_sigma, sigmas - true_sigma)
    assert np.median(np.abs(shifts - true_shift)) < 0.1
    assert np.median(np.abs(sigmas - true_sigma)) < 0.1


def test_fit_solar_spectra_multi_window(solar_based_data):
    solar_corr, sc, *_ = solar_based_data

    res = solar_corr.fit_solar_spectra(
        sc,
        n_windows=3,
        window_overlap=0.3,
        response_window_size_aa=150,
        mask_tellurics=False,
    )

    assert "windows" in res and "centers" in res["windows"] and "slices" in res["windows"]
    assert len(res["windows"]["slices"]) == res["shift_pix_matrix"].shape[1]
    # Some windows may fail (e.g. edge cases), but most should succeed
    ok = np.isfinite(res["shift_pix_matrix"]).sum() + np.isfinite(res["sigma_pix_matrix"]).sum()
    assert ok > 0


# --------------------------
# Tests: FibreLSFModel
# --------------------------

def test_fibrelsf_get_and_interpolate_along_lsf(solar_based_data):
    _, sc, *_ = solar_based_data
    wave = sc.wavelength
    n_fib = sc.rss_intensity.shape[0]
    n_kernel = 7
    lsf_wave_edges = np.linspace(-2, 2, n_kernel + 1) << u.AA

    rng = np.random.default_rng(1234)
    kernel = rng.random((n_fib, wave.size, n_kernel))
    kernel /= np.sum(kernel, axis=2, keepdims=True)

    lsf = FibreLSFModel(
        wavelength=wave,
        lsf_wave_edges=lsf_wave_edges,
        kernel=kernel,
    )

    fibre_kernel, fibre_edges = lsf.get_fibre_lsf(0)
    assert fibre_kernel.shape == (wave.size, n_kernel)
    assert np.allclose(
        fibre_edges.to_value(u.AA),
        lsf_wave_edges.to_value(u.AA),
    )

    new_lsf_wave_edges = np.linspace(-2.5, 2.5, n_kernel + 3) << u.AA
    lsf.interpolate_along_lsf(new_lsf_wave_edges)

    assert lsf.kernel.shape == (n_fib, wave.size, new_lsf_wave_edges.size - 1)
    np.testing.assert_allclose(np.sum(lsf.kernel, axis=2), 1.0, rtol=0, atol=1e-8)


def test_gaussian_fibrelsfmodel_builds_normalized_kernel(solar_based_data):
    _, sc, *_ = solar_based_data
    wave = sc.wavelength
    n_fib = sc.rss_intensity.shape[0]
    sigma = np.full((n_fib, wave.size), 1.0) << u.AA
    lsf_wave_edges = np.linspace(-3, 3, 13) << u.AA

    lsf = GaussianFibreLSFModel(
        wavelength=wave,
        sigma=sigma,
        lsf_wave_edges=lsf_wave_edges,
    )

    assert lsf.kernel.shape == (n_fib, wave.size, lsf_wave_edges.size - 1)
    np.testing.assert_allclose(np.sum(lsf.kernel, axis=2), 1.0, rtol=0, atol=1e-8)


def test_fibrelsf_interpolate_wavelength_grid_preserves_normalization(solar_based_data):
    _, sc, *_ = solar_based_data
    wave = sc.wavelength
    n_fib = sc.rss_intensity.shape[0]

    n_kernel = 9
    lsf_wave_edges = np.linspace(-3, 3, n_kernel + 1) << u.AA
    rng = np.random.default_rng(77)
    kernel = rng.random((n_fib, wave.size, n_kernel))
    kernel /= np.sum(kernel, axis=2, keepdims=True)

    lsf = FibreLSFModel(
        wavelength=wave,
        lsf_wave_edges=lsf_wave_edges,
        kernel=kernel,
    )

    new_wave = np.linspace(wave[0].value, wave[-1].value, wave.size // 2) << wave.unit
    lsf.interpolate_wavelength_grid(new_wave)

    assert lsf.kernel.shape == (n_fib, new_wave.size, n_kernel)
    np.testing.assert_allclose(np.sum(lsf.kernel, axis=2), 1.0, rtol=0, atol=1e-8)
    assert np.all(lsf.kernel >= 0)
