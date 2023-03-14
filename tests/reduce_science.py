#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:33:09 2022

@author: pablo
"""
# =============================================================================
# Basics packages
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.wcs import WCS
from astropy.io import fits
# =============================================================================
# 1. Reading the data
# =============================================================================
from modular_koala.koala_ifu import koala_rss
# =============================================================================
# 5. Applying throughput
# =============================================================================
from modular_koala.throughput import get_from_sky_flat
from modular_koala.throughput import apply_throughput
# =============================================================================
# 6. Correcting for extinction
# =============================================================================
from modular_koala.atmospheric_corrections import AtmosphericExtinction
# =============================================================================
# 7. Telluric correction (only for red data) U
# =============================================================================
from modular_koala.sky import Tellurics
# =============================================================================
# Sky substraction
# =============================================================================
from modular_koala.sky import SkyFromObject, SkyOffset, uves_sky_lines
# =============================================================================
# Cubing
# =============================================================================
from modular_koala.cubing import interpolate_rss, build_cube
from modular_koala.rss import RSS


# List and description of the sample RSS in the red arm
# -------------------------------------------------------------------------------

# modular_koala/input_data/sample_RSS/385R/27feb20032red.fits He2-10 B 1.5S
# modular_koala/input_data/sample_RSS/385R/27feb20033red.fits He2-10 C 1.5S 3E
# modular_koala/input_data/sample_RSS/385R/27feb20034red.fits He2-10 D 18S 3E
# modular_koala/input_data/sample_RSS/385R/27feb20035red.fits He2-10 E 1.5E
# modular_koala/input_data/sample_RSS/385R/27feb20036red.fits He2-10 F 1.5E 1.5S

# modular_koala/input_data/sample_RSS/385R/combined_skyflat_red.fits SKYFLAT
## ============================================================================

grating = '385R'
grating = '580V'

prefixes = {'580V': '1', '385R': '2'}
# Read data and store data as RSS objects
science_rss = []
# for i in range(35, 37):
for i in [32, 33, 35, 36]:
    file_path = (
        'modular_koala/input_data/sample_RSS/{}/27feb{}00{}red.fits'
        .format(grating, prefixes[grating], i))
    science_rss.append(koala_rss(file_path))

unknown = koala_rss("modular_koala/input_data/sample_RSS/385R/27feb20034red.fits")

if grating == '385R':
    sky_flat_rss = koala_rss(
        'modular_koala/input_data/sample_RSS/{}/combined_skyflat_red.fits'
        .format(grating))
elif grating == '580V':
    sky_flat_rss = koala_rss(
        'modular_koala/input_data/sample_RSS/{}/combined_skyflat_blue.fits'
        .format(grating))

raw_intensity = science_rss[0].intensity_corrected.copy()
# %%===========================================================================
# Fibre Throuput
# =============================================================================
throughput_2D = get_from_sky_flat(sky_flat_rss)

for i in range(len(science_rss)):
    science_rss[i] = apply_throughput(science_rss[i], throughput_2D)

throughput_intensity = science_rss[0].intensity_corrected.copy()
# %%

# %%===========================================================================
# Atmospheric Extinction
# =============================================================================
atm_ext_corr = AtmosphericExtinction()

for i in range(len(science_rss)):
    atm_ext_corr.get_atmospheric_extinction(
        airmass=science_rss[i].info['airmass'])
    science_rss[i] = atm_ext_corr.apply(science_rss[i])

atmext_intensity = science_rss[0].intensity_corrected.copy()

# %%===========================================================================
# Telluric CorrectionBase
# =============================================================================

for i in range(len(science_rss)):
    telluric_correction = Tellurics(science_rss[i])
    # telluric_correction.telluric_from_smoothed_spec(plot=True)
    telluric_correction.telluric_from_model(plot=True)
    science_rss[i] = telluric_correction.apply(science_rss[i])

telluric_intensity = science_rss[0].intensity_corrected.copy()

# %%===========================================================================
# Sky substraction
# =============================================================================

for i in range(len(science_rss)):
    skymodel = SkyFromObject(science_rss[i])
    skymodel.load_sky_lines()
    # Quick sky subtraction based on the 16, 50 and 84 percentiles
    pct_sky = skymodel.estimate_sky()
    skymodel.intensity = pct_sky[1]
    skymodel.variance = (pct_sky[1] - pct_sky[0])**2
    science_rss[i] = skymodel.substract_sky(science_rss[i])

    skymodel = SkyFromObject(science_rss[i])
    skymodel.load_sky_lines()
    # Quick sky subtraction based on the 16, 50 and 84 percentiles
    pct_sky = skymodel.estimate_sky()
    skymodel.intensity = pct_sky[1]
    skymodel.variance = (pct_sky[1] - pct_sky[0])**2
    science_rss[i] = skymodel.substract_sky(science_rss[i])

skysub_intensity = science_rss[0].intensity_corrected.copy()

plt.figure(figsize=(10, 10))
plt.plot(science_rss[0].wavelength, np.nansum(raw_intensity, axis=(0)),
         label='RAW')
plt.plot(science_rss[0].wavelength, np.nansum(throughput_intensity, axis=(0)),
         label='Fib. Throughput')
plt.plot(science_rss[0].wavelength, np.nansum(atmext_intensity, axis=(0)),
         label='Atm Ext.')
plt.plot(science_rss[0].wavelength, np.nansum(telluric_intensity, axis=(0)),
         label='Tellurics')
plt.plot(science_rss[0].wavelength, np.nansum(skysub_intensity, axis=(0)),
         label='Sky')
plt.legend()

# %%
idx = 618

plt.figure()
h, xedges, _ = plt.hist(science_rss[0].intensity[:, idx], bins=np.logspace(1, 5, 100), log=True)
pos = np.argmax(h)
xbins = (xedges[:-1] + xedges[1:]) / 2

x_max = np.sum(xbins * h**10) / np.sum(h**10)
plt.axvline(xbins[pos], c='k')
plt.axvline(x_max, c='b')
plt.axvline(np.nanmedian(science_rss[0].intensity[:, idx]), c='r')
plt.axvline(350, c='g')
plt.xscale('log')
# %%===========================================================================
# Cubing
# =============================================================================

fig, axs = plt.subplots(ncols=len(science_rss),
                        figsize=(len(science_rss) * 4, 4))
for i, ax in enumerate(axs):
    ax.scatter(science_rss[i].info['fib_ra_offset'],
               science_rss[i].info['fib_dec_offset'],
               c=np.nansum(science_rss[i].intensity_corrected, axis=1),
               cmap='nipy_spectral')



fig, axs = plt.subplots(ncols=len(science_rss),
                        figsize=(len(science_rss) * 4, 4))
for i, ax in enumerate(axs):
    mappable = ax.scatter(science_rss[i].info['fib_ra_offset'],
               science_rss[i].info['fib_dec_offset'],
               c=science_rss[i].intensity[:, 618],
               cmap='nipy_spectral', vmin=300, vmax=1000)
plt.colorbar(mappable, ax=ax)

# Register each RSS file before merging into a cube
from modular_koala.registration import fit_moffat_profile, register_stars

# To register a set of rss it is possible to use a 2D moffat fit
# or simply the center of light

register_stars(science_rss, moffat=False, plot=True)

science_cube = build_cube(
    rss_set=science_rss,
    reference_coords=(0, 0),
    reference_pa=0, cube_size_arcsec=(50, 50),
    pixel_size_arcsec=.5,
    kernel_size_arcsec=1.5)

science_cube.to_fits(fname='test_cube_' + grating + '.fits.gz')

# Load response curve
wl1, r1 = np.loadtxt("HD60753_" + grating, unpack=True)
r1[r1 <= 0] = np.nan

flux = science_cube.intensity / r1[:, np.newaxis, np.newaxis]

i, j = 50, 65

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(np.log10(np.nanmedian(flux, axis=0)),
           cmap='nipy_spectral',
           # vmin=0,
           origin='lower', interpolation='none')
plt.colorbar(orientation='horizontal', label='log10(F / [1e-16 erg/s/cm2/AA/spx])')
plt.plot(j, i, 'k+', ms=10)
plt.subplot(122)
plt.plot(science_cube.wavelength,
         np.nansum(flux, axis=(1, 2)), label='Integrated spectrum', c='k')
# plt.plot(science_cube.wavelength,
#          np.nansum(science_cube.intensity, axis=(1, 2)), label='Counts/s')
plt.plot(science_cube.wavelength,
         np.nansum(science_cube.variance, axis=(1, 2))**0.5 / r1,
         label=r'Integrated std')
plt.plot(science_cube.wavelength, flux[:, i, j],
         label=r'Spx {}-{} flux'.format(i, j))
plt.plot(science_cube.wavelength, science_cube.variance[:, i, j]**0.5 / r1,
         label=r'Spx {}-{} std'.format(i, j))
plt.yscale('log')
plt.ylabel('F / [1e-16 erg/s/cm2/AA/spx]')
plt.xlabel(r'$\lambda$')
plt.legend(bbox_to_anchor=(1, 0.5))

plt.figure()
plt.subplot(111, title='Integrated signal to noise')
plt.plot(science_cube.wavelength,
         np.nansum(science_cube.intensity, axis=(1, 2))
         / np.nansum(science_cube.variance, axis=(1, 2))**0.5)
plt.yscale('log')
plt.ylim(1, 1e3)

plt.figure()

plt.subplot(111, title='Integrated signal to noise spaxel {} {}'.format(i, j))

plt.ylim(0, 30)

# plt.ylim(0, 80)
# plt.xlim(4000, 4500)