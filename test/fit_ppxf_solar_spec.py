#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:49:53 2024

@author: pcorchoc
"""


from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

from pykoala.instruments.koala_ifu import koala_rss

from scipy.ndimage import median_filter, gaussian_filter
from scipy.ndimage import percentile_filter
from scipy import signal

from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
from astropy import units as u
from astropy import constants


rss = koala_rss("/home/pcorchoc/Develop/pykoala-tutorials/tutorials/data/koala/385R/27feb20012red.fits")
fibre = 100
# rss = koala_rss("/home/pcorchoc/Develop/pykoala-tutorials/tutorials/data/koala/580V/27feb10012red.fits")

# sky_spectra /= (percentile_filter(sky_spectra, 90, 100) + 1e-100)
velscale = np.min(constants.c.to_value("km/s")*np.diff(np.log(rss.wavelength)))  # Preserve smallest velocity step

sun = fits.open("/home/pcorchoc/Research/obs_data/std_stars/calspec/sun_reference_stis_002.fits")
sun = fits.open("/home/pcorchoc/Research/obs_data/std_stars/calspec/sun_mod_001.fits")

sun_wavelength = sun[1].data['WAVELENGTH']
sun_mask = (sun_wavelength > 3500) & (sun_wavelength < 10000)
sun_wavelength = sun_wavelength[sun_mask]
sun_flux = sun[1].data['FLUX'][sun_mask]

sun_spectra = Spectrum1D(flux=sun_flux * u.erg / u.s / u.cm**2 / u.angstrom,
                         spectral_axis=sun_wavelength * u.angstrom)

rss_spectra = Spectrum1D(
    flux=rss.intensity[fibre] * u.erg / u.s / u.cm**2 / u.angstrom,
    spectral_axis=rss.wavelength * u.angstrom)

resampler = FluxConservingResampler()

new_wl = np.geomspace(8350, 8850, 200)
velscale = np.min(constants.c.to_value("km/s")*np.diff(np.log(new_wl)))  # Preserve smallest velocity step

interp_sun = resampler(sun_spectra, new_wl * u.angstrom)
interp_rss = resampler(rss_spectra, new_wl * u.angstrom)

resp_spectrograph = interp_rss.flux.value / interp_sun.flux.value

smoothed_r_spectrograph = median_filter(resp_spectrograph, 100)

plt.figure()
plt.plot(new_wl, resp_spectrograph)
plt.plot(new_wl, smoothed_r_spectrograph)


plt.figure()
plt.plot(interp_rss.wavelength, interp_rss.flux / smoothed_r_spectrograph)
plt.plot(interp_sun.wavelength, interp_sun.flux)
# plt.xlim(4800, 4900)

all_offset = []
all_corr = []
all_chi2 = []

velshift_array = np.arange(-5, 5, 0.1) 
gauss_std_array = np.arange(0.1, 4, 0.1)

chi2 = np.zeros((velshift_array.size, gauss_std_array.size))

for i, velshift in enumerate(velshift_array):
    for j, gauss_std in enumerate(gauss_std_array):
        print(i, j)
        z = velshift * velscale / constants.c.to_value("km/s")

        interp_sun = resampler(sun_spectra,
                               interp_rss.wavelength * (1 + z))

        chi2[i, j] = np.sum(
            (interp_rss.flux.value / smoothed_r_spectrograph
             - gaussian_filter(interp_sun.flux.value, gauss_std))**2)
        
best_vel_idx, best_sigma_idx = np.unravel_index(
    np.argmin(chi2), chi2.shape)
z = velshift_array[best_vel_idx] * velscale / constants.c.to_value("km/s")
interp_sun = resampler(sun_spectra, interp_rss.wavelength / (1 + z))
best_fit_spectra = gaussian_filter(interp_sun.flux.value,
                                   gauss_std_array[best_sigma_idx])

plt.figure()
plt.plot(sun_wavelength, sun_flux)
plt.plot(interp_rss.wavelength, interp_rss.flux.value / smoothed_r_spectrograph)
# plt.plot(rss.wavelength[2:], corr_fibre_int[:-2])
plt.plot(interp_sun.wavelength / (1 + z), best_fit_spectra)

# plt.xlim(4800, 4900)
plt.xlim(8495, 8710)
# plt.ylim(0.8, 1.2)
# %%
goodpixels = np.where(np.isfinite(sky_spectra))[0]
start = [0., 100.]

pp = ppxf(np.atleast_2d(sun_spectra).T, sky_spectra, noise,
          velscale, start,
          goodpixels=goodpixels, plot=False, moments=4,
          degree=0, lam=np.exp(ln_lam_sky), lam_temp=np.exp(ln_lam_sun))

pp.plot()
