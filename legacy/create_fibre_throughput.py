#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basic steps to create a fibre throughput"""
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
from koala.koala_ifu import koala_rss
# =============================================================================
# 5. Applying throughput
# =============================================================================
from koala.corrections.throughput import create_throughput_from_flat
from koala.corrections.throughput import get_from_sky_flat
from koala.corrections.throughput import apply_throughput
# =============================================================================
# 6. Correcting for extinction
# =============================================================================
from koala.corrections.atmospheric_corrections import AtmosphericExtinction
# =============================================================================
# 7. Telluric correction (only for red data) U
# =============================================================================
from koala.corrections.sky import Tellurics
# =============================================================================
# Sky substraction
# =============================================================================
from koala.corrections.sky import SkyFromObject, SkyOffset, uves_sky_lines
# =============================================================================
# Cubing
# =============================================================================
from koala.cubing import interpolate_rss, build_cube
from koala.rss import RSS


# ============================================================================
# Star 1
# koala/input_data/sample_RSS/385R/27feb20025red.fits HD60753 C 3W WIDE
# koala/input_data/sample_RSS/385R/27feb20026red.fits HD60753 B 3S WIDE
# koala/input_data/sample_RSS/385R/27feb20027red.fits HD60753 A WIDE

# Star 2
# koala/input_data/sample_RSS/385R/27feb20028red.fits HILT600 A
# koala/input_data/sample_RSS/385R/27feb20029red.fits HILT600 B 3S
# koala/input_data/sample_RSS/385R/27feb20030red.fits HILT600 C 3W

# Sky flat
# koala/input_data/sample_RSS/580V/combined_skyflat_blue.fits
# ============================================================================
# grating = '385R'
grating = '580V'

prefixes = {'580V': '1', '385R': '2'}
# Read data and store data as RSS objects
skyflat_rss = []
for i in range(10, 14):
    file_path = (
        '/home/pablo/Research/obs_data/HI-KIDS/RSS/sci/07sep{}00{}red.fits'
        .format(prefixes[grating], i))
    skyflat_rss.append(koala_rss(file_path))

# %% 
fig, axs = plt.subplots(ncols=len(skyflat_rss), figsize=(10, 5))

for i in range(axs.size):
    axs[i].imshow(skyflat_rss[i].intensity, aspect='auto', origin='lower',
                  cmap='nipy_spectral')

mean, std = create_throughput_from_flat(skyflat_rss, clear_nan=True)

plt.figure()
plt.imshow(mean, cmap='nipy_spectral', vmin=0.8, vmax=1.2)
plt.colorbar()