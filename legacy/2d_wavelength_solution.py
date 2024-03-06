#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 11:43:44 2023

@author: pcorchoc
"""

from matplotlib import pyplot as plt
import numpy as np

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from time import time

from koala.corrections.sky import SkyFromObject
from koala.koala_ifu import koala_rss
from koala.ancillary import weighted_median_filter
from koala.corrections.throughput import Throughput
from koala.corrections.throughput import ThroughputCorrection
from koala.corrections.sky import BackgroundEstimator

rss = koala_rss("../tutorials/data/27feb20032red.fits")



th = ThroughputCorrection.create_throughput_from_rss(
    [koala_rss("reduced_data/385R/combined_skyflat_red.fits")])

throughput_corr = ThroughputCorrection(throughput=th)
rss = throughput_corr.apply(rss)

# %%
lines = rss.intensity_corrected.copy()

lines_map = np.zeros_like(lines, dtype=bool)

to = time()
for i, fibre in enumerate(rss.intensity_corrected):
    print(i)
    lines[i] = rss.intensity_corrected[i] - weighted_median_filter(
        fibre, 1 / fibre,
        window=50)
    
    background, background_sigma = BackgroundEstimator.mad(lines[i])

    peaks_found, peaks_prop = find_peaks(lines[i],
                             height=background + 3 * background_sigma)
    lines_map[i][peaks_found] = True

te = time()
line_detection_freq = np.sum(lines_map, axis=0)

master_lines, master_line_prop = find_peaks(
    line_detection_freq, height=lines_map.shape[0] * 0.7)

plt.figure()
plt.plot(rss.wavelength, line_detection_freq)
plt.plot(rss.wavelength[master_lines],
         master_line_prop['peak_heights'], '+')
print("Continuum filtering execution time: ", te - to)

# %%
def gaussian(x, a, mu, sigma):
    """1D-Gaussian function."""
    g = a * np.exp(- (x - mu)**2 / sigma**2 / 2)
    return g

lines_results = np.full((lines.shape[0], master_lines.size, 3), fill_value=np.nan)
to = time()
for i, fibre in enumerate(lines):
    print(i)
    for n, m in enumerate(master_lines):
        mask = np.isfinite(fibre)
        weights = np.ones_like(rss.wavelength)
        weights[:m - 5] = 0
        weights[m + 5:] = 0
        
        p0=(fibre[m], rss.wavelength[m],
            (rss.wavelength[m] - rss.wavelength[m - 1]))
        
        bounds=(
            (np.min((fibre[m] - 100, fibre[m] * 0.75)),
             rss.wavelength[m] - 5, 0.5),
            (np.abs(fibre[m]) * 1.25, rss.wavelength[m] + 5, 3.0)
                )
        
        # plt.plot(gaussian(rss.wavelength, *p0), lw=0.5, color='orange')
        
        popt, pcov = curve_fit(
            gaussian,
            rss.wavelength[mask],
            fibre[mask],
            # sigma=weights[mask],
            p0=p0,
            bounds=bounds,
            method='trf',
            ftol=1e-6, maxfev=1000000
            )
        
        lines_results[i, n] = popt

        
ref_fibre = np.argmax(np.nanmedian(lines_results[:, :, 2], axis=1))

median_res = np.nanmedian(lines_results, axis=0)

norm_lines = lines_results - lines_results[ref_fibre][np.newaxis]
plt.figure()
plt.hist(median_res[:, 2], bins='auto')
plt.xlabel(r"Median $\sigma$ ($\AA$)")


line_pos_pct = np.nanpercentile(norm_lines[:, :, 1], (16, 50, 84), axis=1)
line_sig_pct = np.nanpercentile(norm_lines[:, :, 2], (16, 50, 84), axis=1)

plt.figure()
plt.plot(line_pos_pct.T)
plt.ylabel("Pixel offset")
plt.xlabel("Fibre")


p = np.polyfit(np.arange(0, line_sig_pct[1].size),
               line_sig_pct[1], deg=2, w=line_sig_pct[2] - line_sig_pct[0])
plt.figure()
plt.plot(line_sig_pct.T)
plt.plot(np.polyval(p, np.arange(0, line_sig_pct[1].size)))
plt.ylabel("Pixel offset")
plt.xlabel("Fibre")
plt.ylim(-0.4, .4)

te = time()
print("Line modelling execution time: ", te - to)

# %%
fibre = 900

plt.figure(figsize=(10, 5))

plt.plot(rss.wavelength, rss.intensity_corrected[fibre], c='k')
plt.plot(rss.wavelength, lines[fibre], c='r', lw=0.7)
# plt.yscale('log')

# plt.figure()
# plt.imshow(residuals / rss.intensity_corrected))
# %%

continuum = rss.intensity.copy()
to = time()
for i, fibre in enumerate(rss.intensity):
    print(i)
    continuum[i] = weighted_median_filter(fibre, 1 / fibre,
                                          window=50)
te = time()

plt.figure(figsize=(10, 5))
plt.plot(rss.wavelength, np.nanmedian(rss.intensity, axis=0))
plt.plot(rss.wavelength, np.nanmedian(continuum, axis=0))

