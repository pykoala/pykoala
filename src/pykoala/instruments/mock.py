"""Module to create mock datacontainers"""

# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import os
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
# =============================================================================
# pyKOALA packages
# =============================================================================
from pykoala import vprint
from pykoala.data_container import RSS
from pykoala.ancillary import check_unit


class NoiseModel:
    """Noise model for mock data."""
    def __init__(self, bias_bckgr, gaussian_sigma, poisson_flux_thresh, poisson_sigma):
        self.bias_bckgr = np.asarray(bias_bckgr)
        self.gaussian_sigma = gaussian_sigma
        self.poisson_flux_thresh = poisson_flux_thresh
        self.poisson_sigma = poisson_sigma

    def gaussian_noise(self, intensity):
        return np.random.normal(0, self.gaussian_sigma, size=intensity.shape)

    def poisson_noise(self, intensity):
        return self.poisson_sigma * np.sqrt(intensity / self.poisson_flux_thresh)

    def __call__(self, intensity):
        if (self.bias_bckgr.ndim > 0) and (self.bias_bckgr.shape != intensity.shape):
            raise ArithmeticError(f"Bias background dimensions ({self.bias_bckgr}) do not"
                                  f"match intensity shape {intensity.shape}")
        return (self.bias_bckgr + self.gaussian_noise(intensity)
                + self.poisson_noise(intensity))

def gaussian_source(fibre_ra, fibre_dec, source_ra, source_dec,
                    source_ra_sigma, source_dec_sigma, source_intensity=1):
    x = (fibre_ra - source_ra) / source_ra_sigma
    y = (fibre_dec - source_dec) / source_dec_sigma
    if x.ndim != source_intensity.ndim:
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
    intensity =  np.exp(- 0.5 * x**2 - 0.5 * y**2) * source_intensity
    return intensity

def mock_rss(ra_n_fibres=20, dec_n_fibres=20, fibre_diameter=1.5 << u.arcsec,
             fibre_separation=0 << u.arcsec,
             n_wave=1000, wave_range=(3000 << u.AA, 8000 << u.AA),
             ra_cen=180 << u.deg, dec_cen=45 << u.deg, exptime=600 << u.second,
             airmass=1.0,
             source_kwargs={},
             noise_kwargs={},
             nan_frac=0.0):
    """Create a mock RSS object."""
    vprint("Creating mock RSS data")
    n_fibres = ra_n_fibres * dec_n_fibres
    fibre_diameter = check_unit(fibre_diameter)
    fibre_separation = check_unit(fibre_separation)
    fov = (ra_n_fibres * (fibre_diameter + fibre_separation) - fibre_separation,
           dec_n_fibres * (fibre_diameter + fibre_separation) - fibre_separation)

    ra = np.linspace(- fov[0] / 2, fov[0] / 2, ra_n_fibres) + ra_cen
    dec = np.linspace(- fov[1] / 2, fov[1] / 2, dec_n_fibres) + dec_cen
    ra, dec = np.meshgrid(ra, dec)
    vprint(f"Total number of fibres: {n_fibres}")

    # Ensure units consistency
    wave_range = [check_unit(wr, u.AA) for wr in wave_range]
    wavelength = np.linspace(*wave_range, n_wave)

    # Create intensity and variance attributes
    intensity = np.zeros((n_fibres, wavelength.size))
    variance = np.zeros_like(intensity)
   
    # Add sources and noise
    if "source_ra" not in source_kwargs:
        source_kwargs["source_ra"] = ra_cen
    if "source_dec" not in source_kwargs:
        source_kwargs["source_dec"] = dec_cen
    if "source_ra_sigma" not in source_kwargs:
        source_kwargs["source_ra_sigma"] = fov[0] / 2
    if "source_dec_sigma" not in source_kwargs:
        source_kwargs["source_dec_sigma"] = fov[1] / 2
    if "source_intensity" not in source_kwargs:
        source_kwargs["source_intensity"] = 10 * np.ones_like(intensity)

    intensity += gaussian_source(ra.flatten(), dec.flatten(), **source_kwargs)
    
    # Add noise
    if "bias_bckgr" not in noise_kwargs:
        noise_kwargs["bias_bckgr"] = 0.05
    if "gaussian_sigma" not in noise_kwargs:
        noise_kwargs["gaussian_sigma"] = 0.1
    if "poisson_flux_thresh" not in noise_kwargs:
        noise_kwargs["poisson_flux_thresh"] = 0.75
    if "poisson_sigma" not in noise_kwargs:
        noise_kwargs["poisson_sigma"] = 0.1

    noise_model = NoiseModel(**noise_kwargs)
    intensity += noise_model(intensity)
    variance += noise_kwargs["gaussian_sigma"]**2

    # Add nans (e.g. masked cosmic rays)
    if nan_frac > 0:
        nan_idx = np.random.randint(0, intensity.size, size=int(intensity.size * nan_frac))
        intensity[np.unravel_index(nan_idx, intensity.shape)] = np.nan
        variance[np.unravel_index(nan_idx, intensity.shape)] = np.nan

    # Create the RSS object
    info = {}
    info['name'] = "MockRSS"
    info['exptime'] = exptime
    info['fib_ra'] = ra.flatten()
    info['fib_dec'] = dec.flatten()
    info['airmass'] = airmass

    rss = RSS(intensity=intensity << u.adu,
              variance=variance << u.adu**2,
              wavelength=wavelength,
              info=info,
              fibre_diameter=fibre_diameter,
              wcs=WCS())

    rss.history('Mock', "Mock RSS generated by pykoala")
    return rss

if __name__ == "__main__":
    rss = mock_rss()
    
    rss.plot_fibres()

