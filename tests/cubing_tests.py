#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 14:58:11 2022

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
from modular_koala.sky import SkyFromObject, uves_sky_lines
# =============================================================================
# Cubing
# =============================================================================
from modular_koala.cubing import interpolate_rss, build_cube
from modular_koala.rss import RSS
file_path = 'modular_koala/input_data/sample_RSS/580V/27feb10025red.fits'
file_path = 'modular_koala/input_data/sample_RSS/385R/27feb20025red.fits'
std_rss = koala_rss(file_path)

# combined_skyflat_red --------------------------------------------------------
file_path = 'modular_koala/input_data/sample_RSS/385R/combined_skyflat_red.fits'
skyflat_rss = koala_rss(file_path)

# Object: Tol30 ---------------------------------------------------------------
file_path = 'modular_koala/input_data/sample_RSS/385R/27feb20031red.fits'
tol30_rss = koala_rss(file_path)

# Object: He2_100 -------------------------------------------------------------
file_path = 'modular_koala/input_data/sample_RSS/385R/27feb20032red.fits'
he2_100_rss_1 = koala_rss(file_path)
file_path = 'modular_koala/input_data/sample_RSS/385R/27feb20033red.fits'
he2_100_rss_2 = koala_rss(file_path)
file_path = 'modular_koala/input_data/sample_RSS/385R/27feb20035red.fits'
he2_100_rss_3 = koala_rss(file_path)
file_path = 'modular_koala/input_data/sample_RSS/385R/27feb20036red.fits'
he2_100_rss_4 = koala_rss(file_path)
import copy

std_rss_2 = copy.deepcopy(std_rss)
# std_rss_2.info['pos_angle'] = 0.
# std_rss_2.info['cen_ra'] = std_rss_2.info['cen_ra'] - 10 / 3600

cube = build_cube(
    rss_set=[std_rss, std_rss_2
             ],
    reference_coords=(
        std_rss.info['cen_ra'], std_rss.info['cen_dec']),
    reference_pa=0.0, cube_size_arcsec=(60, 60))


rss_collapsed_spectra = np.nanmean(
    [np.nansum(std_rss.intensity_corrected, axis=0),
     np.nansum(std_rss_2.intensity_corrected, axis=0)], axis=0)

print(np.nansum(cube.intensity) / np.nansum(rss_collapsed_spectra))

data_collapsed = np.nansum(cube.intensity, axis=0)

plt.figure()
plt.imshow(cube.rss_mask.sum(axis=(0, 1)))
plt.imshow(cube.rss_mask[1, 1000, :, :] * 2)
plt.colorbar()

plt.figure()
plt.plot(np.nansum(cube.intensity, axis=(1, 2)))
plt.plot(rss_collapsed_spectra)

# %%
cube = build_cube(
    rss_set=[he2_100_rss_1, he2_100_rss_2, he2_100_rss_3, he2_100_rss_4],
    reference_coords=(
        he2_100_rss_1.info['cen_ra'], he2_100_rss_1.info['cen_dec']),
    reference_pa=0.0, cube_size_arcsec=(80, 80))
data_collapsed = np.nanmedian(cube.intensity[100:-100], axis=0)
# %%
# Getting the cube centre of mass
x_com, y_com = cube.get_centre_of_mass(wavelength_step=50)

plt.figure()
plt.plot(std_rss.wavelength, x_com - np.median(x_com))
plt.plot(std_rss.wavelength, y_com - np.median(y_com))
plt.ylim(-1, 1)

plt.figure()
plt.imshow(np.log10(data_collapsed), cmap='nipy_spectral')
plt.scatter(x_com, y_com, s=1, c='k')