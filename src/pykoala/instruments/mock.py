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

def gaussian_source(fibre_ra, fibre_dec, source_ra, source_dec,
                    source_ra_sigma, source_dec_sigma, source_intensity=1):
    x = (fibre_ra - source_ra) / source_ra_sigma
    y = (fibre_dec - source_dec) / source_dec_sigma
    if x.ndim != source_intensity.ndim:
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
    intensity =  np.exp(- 0.5 * x**2 - 0.5 * y**2) * source_intensity
    return intensity

def gaussian_noise(intensity_shape, noise_intensity=1, sigma=1):
    return noise_intensity * np.random.normal(0, sigma, size=intensity_shape)

def mock_rss(ra_n_fibres=20, dec_n_fibres=20, fibre_diameter=1.5 << u.arcsec,
             fibre_separation=0 << u.arcsec,
             n_wave=5000, wave_range=(3000 << u.AA, 8000 << u.AA),
             ra_cen=180 << u.deg, dec_cen=45 << u.deg, exptime=600 << u.second,
             airmass=1.0):
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
    intensity += gaussian_source(
        ra.flatten(), dec.flatten(),
        source_ra=ra_cen, source_dec=dec_cen,
        source_ra_sigma=fov[0] / 2, source_dec_sigma=fov[1] / 2,
        source_intensity=np.ones_like(intensity))
    intensity += gaussian_noise(intensity.shape, sigma=1, noise_intensity=0.1)
    variance += 0.1**2
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

    # return rss

if __name__ == "__main__":
    rss = mock_rss()
    
    rss.plot_fibres()
