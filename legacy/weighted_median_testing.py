#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:54:54 2023

@author: pcorchoc
"""

from matplotlib import pyplot as plt
import numpy as np


from time import time

from koala.corrections.sky import SkyFromObject
from koala.koala_ifu import koala_rss
from koala.ancillary import weighted_median_filter
from koala.corrections.throughput import Throughput
from koala.corrections.throughput import ThroughputCorrection
from koala.ancillary import interpolate_image_nonfinite
rss = koala_rss("../tutorials/data/27feb20032red.fits")



th = ThroughputCorrection.create_throughput_from_rss(
    [koala_rss("reduced_data/385R/combined_skyflat_red.fits")])

throughput_corr = ThroughputCorrection(throughput=th)
rss = throughput_corr.apply(rss)

# %%
skymodel = SkyFromObject(rss)

residuals = rss.intensity_corrected - skymodel.bckgr

residuals_lines = residuals.copy()

to = time()
for i, fibre in enumerate(residuals):
    print(i)
    residuals_lines[i] = residuals[i] - weighted_median_filter(
        fibre, 1 / fibre,
        window=100)
te = time()
print("Continuum filtering execution time: ", te - to)
# %%
fibre = 400


fibre_mask = np.sum(skymodel.source_mask, axis=1) < rss.wavelength.size / 2
std = np.nanstd(rss.intensity_corrected, axis=0)
mean = np.nanmean(rss.intensity_corrected, axis=0)
mean_std = np.nanmedian(std)
sky_mask = std < np.nanpercentile(std, 95)
sky_mask = std < 0.8 * mean

plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(rss.wavelength, rss.intensity_corrected[fibre], c='k', label='Input fibre')
plt.plot(rss.wavelength, skymodel.bckgr, c='r', lw=0.7, label='Zeroth-order sky')
plt.plot(rss.wavelength, skymodel.bckgr * sky_mask, lw=0.7, c='c', label='Masked sky')
plt.legend()
plt.yscale('log')

plt.subplot(212)
plt.plot(rss.wavelength, residuals[fibre], c='k', label='Residuals')
plt.plot(rss.wavelength, residuals_lines[fibre], lw=0.7, c='r', label='Cont-subs res.')
plt.plot(rss.wavelength, residuals_lines[fibre] * sky_mask, lw=0.7, c='c', label='cont-subs masked res.')
plt.legend()
# plt.ylim(0, 400)
plt.grid(visible=True)
plt.ylabel("Residuals (ADU)")
plt.semilogy(linthresh=100)

residuals_lines_masked = residuals_lines * sky_mask[np.newaxis, :]

residuals_lines_masked = interpolate_image_nonfinite(residuals_lines_masked)

# Renormalize
norm = np.nanstd(residuals_lines_masked, axis=0)
norm += 1
residuals_lines_masked_norm = residuals_lines_masked / norm[np.newaxis, :]

p5, p95 = np.nanpercentile(residuals_lines_masked_norm, (5, 95))

plt.figure()
plt.imshow(residuals_lines_masked_norm, vmin=p5, vmax=p95, interpolation='none',
           aspect='auto')
plt.colorbar()

_, s, vt = np.linalg.svd(residuals_lines_masked_norm, full_matrices=False)

from sklearn.decomposition import PCA

pca = PCA(n_components=0.9)
pca.fit(residuals_lines_masked_norm[fibre_mask])
transformed = pca.transform(residuals_lines_masked_norm)
res_pca = pca.inverse_transform(transformed)
# %%
plt.figure()
plt.subplot(311)
plt.plot(pca.explained_variance_)

plt.xlim(0, pca.n_components)

plt.subplot(312)
plt.plot(np.diff(pca.explained_variance_))
plt.ylim(-10, 1)
plt.xlim(0, pca.n_components)
plt.subplot(313)
plt.plot(np.diff(np.diff(pca.explained_variance_)))
plt.ylim(-1, 10)
plt.xlim(0, pca.n_components)
# %%
row = 610
plt.figure()
plt.plot(residuals_lines_masked_norm[row])
plt.plot(res_pca[row])
# %%
residuals_model = res_pca * (norm - 1)

sky = skymodel.bckgr[np.newaxis, :] + residuals_model

plt.figure()
plt.imshow(sky, interpolation='none')
plt.colorbar()


fibre = 410
plt.figure()
# plt.plot(rss.wavelength, rss.intensity_corrected[fibre])
plt.plot(rss.wavelength,rss.intensity_corrected[fibre] - (skymodel.bckgr))
plt.plot(rss.wavelength,rss.intensity_corrected[fibre] - (skymodel.bckgr + residuals_model[fibre]))
plt.xlim(8200, 9200)
# plt.ylim(3000, 7000)

# %%
p5, p95 = np.nanpercentile(vt, (5, 95))

plt.figure()
plt.subplot(121)
plt.imshow(vt, aspect='auto', interpolation='none', cmap='coolwarm',
           vmin=p5, vmax=p95)
plt.colorbar()
plt.ylim(30, 0)
plt.xlim(800, 1200)

plt.subplot(122)
plt.imshow(pca.components_, aspect='auto', interpolation='none', cmap='coolwarm',
           vmin=p5, vmax=p95)
plt.colorbar()
plt.ylim(30, 0)
plt.xlim(800, 1200)

# %%

plt.figure()
plt.plot(s, 'o-')
plt.xlim(0, 20)

plt.figure()
for i in range(10):
    plt.plot(vt[i])
    
