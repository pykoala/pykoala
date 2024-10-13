#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:21:36 2024

@author: pcorchoc
"""

from pykoala.instruments.koala_ifu import koala_rss

from pykoala.corrections.wavelength import SolarCrossCorrOffset
from time import time
from pykoala.plotting.utils import plot_fibres
from matplotlib import pyplot as plt
import numpy as np

solar_correction = SolarCrossCorrOffset()
# rss = koala_rss("/home/pcorchoc/Develop/pykoala-tutorials/tutorials/data/koala/385R/27feb20012red.fits")
fibre = 300
rss = koala_rss("/home/pcorchoc/Develop/pykoala-tutorials/tutorials/data/koala/580V/27feb10012red.fits")

solar_correction.load_solar_spectra()

t0 = time()
shift, sigma = solar_correction.compute_shift_from_twilight(
    rss, keep_features_frac=0.05,
    logspace=False)
fig, axs = plt.subplots(ncols=2, constrained_layout=True, sharex=True, sharey=True,
                        figsize=(8, 4))
plot_fibres(fig, axs[0], r'$\Delta\lambda$ (pix)', rss, shift, norm=plt.Normalize(),
          cmap='gnuplot')
plot_fibres(fig, axs[1], r'$\sigma$ (pix)', rss, sigma, norm=plt.Normalize(),
          cmap='gnuplot')
rss_corrected = solar_correction.apply(rss)

science_rss = koala_rss("/home/pcorchoc/Develop/pykoala-tutorials/tutorials/data/koala/580V/27feb10035red.fits")
corrected_science = solar_correction.apply(science_rss)


tend = time()
print("Elapsed time: ", tend - t0)

# %%
wave_range = [3923, 3943]
vmax = 15000
plt.figure()
plt.subplot(121)
plt.pcolormesh(rss.wavelength, np.arange(1, rss.intensity.shape[0] + 1),
               rss.intensity, vmin=0, vmax=vmax)
plt.axvline(5577.0, color='r')
plt.colorbar()
plt.xlim(wave_range)
plt.subplot(122)
plt.pcolormesh(rss_corrected.wavelength, np.arange(1, rss_corrected.intensity.shape[0] + 1),
               rss_corrected.intensity, vmin=0, vmax=vmax)
plt.colorbar()
plt.axvline(5577.0, color='r')
plt.xlim(wave_range)

# %%%
wave_range = [4858, 4864]
fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)

for f in range(rss.intensity.shape[0]):
    axs[0].plot(rss.wavelength, rss.intensity[f], c='k', alpha=0.02)
    axs[1].plot(rss_corrected.wavelength, rss_corrected.intensity[f], c='k', alpha=0.02)

axs[0].axvline(3933.663, color='r')
axs[1].axvline(3933.663, color='r')
axs[0].axvline(4861, color='r')
axs[1].axvline(4861, color='r')
axs[0].set_xlim(wave_range)
axs[1].set_ylim(0, 57000)

# %%
plt.figure()
plt.subplot(121)
plt.pcolormesh(science_rss.wavelength, np.arange(1, science_rss.intensity.shape[0] + 1),
               science_rss.intensity, vmax=1000)
plt.axvline(5577.0, color='r')
plt.colorbar()
plt.xlim(5570, 5585)
plt.subplot(122)
plt.pcolormesh(corrected_science.wavelength, np.arange(1, corrected_science.intensity.shape[0] + 1),
               corrected_science.intensity, vmax=1000)
plt.colorbar()
plt.axvline(5577.0, color='r')
plt.xlim(5570, 5585)
