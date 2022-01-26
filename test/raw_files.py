#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:52:03 2022

@author: pablo
"""

from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from scipy.signal import find_peaks
from astropy.modeling import models, fitting

fitter = fitting.LevMarLSQFitter()


def master(path):
    files = glob(path + '/*.fits')
    exposures = []
    for file in files:
        with fits.open(file) as f:
            exposures.append(f[0].data)
    exposures = np.array(exposures)
    master = np.array(np.median(exposures, axis=0))
    return master


def double_gauss(a1, a2, mu1, mu2, sigma1, sigma2, bounds1, bounds2):
    g1 = models.Gaussian1D(amplitude=a1, mean=mu1, stddev=sigma1)
    g2 = models.Gaussian1D(amplitude=a2, mean=mu2, stddev=sigma2)
    for key in bounds1.keys():
        g1.bounds[key] = bounds1[key]
        g2.bounds[key] = bounds2[key]
    return g1 + g2


def triple_gauss(a1, a2, a3, mu1, mu2, mu3,
                 sigma1, sigma2, sigma3,
                 bounds1, bounds2, bounds3):
    g1 = models.Gaussian1D(amplitude=a1, mean=mu1, stddev=sigma1)
    g2 = models.Gaussian1D(amplitude=a2, mean=mu2, stddev=sigma2)
    g3 = models.Gaussian1D(amplitude=a3, mean=mu3, stddev=sigma3)
    for key in bounds1.keys():
        g1.bounds[key] = bounds1[key]
        g2.bounds[key] = bounds2[key]
        g3.bounds[key] = bounds3[key]
    return g1 + g2 + g3


class RawFits(object):
    """
    This class represents raw fits files as produced by some IFU instrument 
    """
    spectral_axis = 0
    spatial_axis = 1
    data = None
    n_spatial_pixels = None
    n_spectral_pixels = None
    n_fibres = None
    pixel_between_fibres = None

    def orientate_data(self):
        if self.spectral_axis == 0:
            self.data = self.data.T
            self.spectral_axis = 1
            self.spatial_axis = 0
        else:
            return

    def apply_dark(self, dark):
        self.data = self.data - dark

    def apply_flat(self, flat):
        self.data = self.data / flat

    def spectral_runningmean(self, data=None, pixel_window=50):
        if data is None:
            self.orientate_data()
            data = self.data
        _data = np.c_[np.zeros(data.shape[0]), data]
        cumdata = np.cumsum(_data, axis=self.spectral_axis)
        runmean = (cumdata[:, pixel_window:] - cumdata[:, :-pixel_window]
                   ) / pixel_window
        return runmean

    def spatial_runningmean(self, data=None, pixel_window=50):
        if data is None:
            self.orientate_data()
            data = self.data
        _data = np.vstack((np.zeros(data.shape[1]), data))
        cumdata = np.cumsum(_data, axis=self.spatial_axis)
        runmean = (cumdata[pixel_window:, :] - cumdata[:-pixel_window, :]
                   ) / pixel_window
        return runmean

    def get_tramlines(self, n_chunks=50):
        # Apply a running mean along spectral axis
        runmean = self.spectral_runningmean()
        # Create N chunks and find peaks
        pixels_per_chunk = runmean.shape[1] // n_chunks
        chunked_data = np.zeros((self.n_spatial_pixels,
                                 n_chunks * pixels_per_chunk))
        chunked_data[:, -1] = np.sum(runmean[:, n_chunks * pixels_per_chunk:],
                                     axis=-1)
        chunked_data += runmean[:, :n_chunks * pixels_per_chunk]
        chunked_data = chunked_data.reshape(
                        (self.n_spatial_pixels, n_chunks, pixels_per_chunk))
        chunked_data = chunked_data.sum(axis=-1)
        # chunked_data = self.spatial_runningmean(data=chunked_data,
        #                                         pixel_window=5)
        centroids = []
        n_centroids = []
        for chunk_i in range(n_chunks):
            p, _ = find_peaks(chunked_data[:, chunk_i],
                              distance=self.pixel_between_fibres)
            np.where(np.diff(p) > self.pixel_between_fibres * 2)[0]
            centroids.append(p)
            n_centroids.append(len(p))
            for i, centroid in enumerate(p):
                if (i == 0) | (i == p.size-1):
                    centroid_bounds = {
                        'mean':(centroid - self.pixel_between_fibres/2,
                                centroid + self.pixel_between_fibres/2),
                        'stddev':(1, self.pixel_between_fibres)}
                    next_centroid_bounds = {
                        'mean':(p[i+1] - self.pixel_between_fibres/2,
                                p[i+1] + self.pixel_between_fibres/2),
                        'stddev':(1, self.pixel_between_fibres)}
    
                    model = double_gauss(chunked_data[centroid, chunk_i],
                                 chunked_data[p[i+1], chunk_i],
                                 mu1=centroid, mu2=p[i+1],
                                 sigma1=self.pixel_between_fibres/2,
                                 sigma2=self.pixel_between_fibres/2,
                                 bounds1=centroid_bounds,
                                 bounds2=next_centroid_bounds)
                else:
                    centroid_bounds = {
                        'mean':(centroid - self.pixel_between_fibres/2,
                                centroid + self.pixel_between_fibres/2),
                        'stddev':(1, self.pixel_between_fibres)}
                    next_centroid_bounds = {
                        'mean':(p[i+1] - self.pixel_between_fibres/2,
                                p[i+1] + self.pixel_between_fibres/2),
                        'stddev':(1, self.pixel_between_fibres)}
                    prev_centroid_bounds = {
                        'mean':(p[i-1] - self.pixel_between_fibres/2,
                                p[i-1] + self.pixel_between_fibres/2),
                        'stddev':(1, self.pixel_between_fibres)}
    
                    model = triple_gauss(chunked_data[centroid, chunk_i],
                                 chunked_data[p[i+1], chunk_i],
                                 chunked_data[p[i-1], chunk_i],
                                 mu1=centroid, mu2=p[i+1], mu3=p[i-1],
                                 sigma1=self.pixel_between_fibres/2,
                                 sigma2=self.pixel_between_fibres/2,
                                 sigma3=self.pixel_between_fibres/2,
                                 bounds1=centroid_bounds,
                                 bounds2=next_centroid_bounds,
                                 bounds3=prev_centroid_bounds)
                # g_init = models.Gaussian1D(
                #     amplitude=chunked_data[centroid, chunk_i],
                #     mean=centroid,
                #     stddev=self.pixel_between_fibres/2)
                # g_init.bounds['mean'] = [
                #     centroid - self.pixel_between_fibres/2,
                #     centroid + self.pixel_between_fibres/2]
                # g_init.bounds['stddev'] = [1, self.pixel_between_fibres]
                g_init = model
                pts_width = 2 * self.pixel_between_fibres
                pts = (self.spatial_pixels >= np.max((0, centroid - pts_width))
                       ) & (self.spatial_pixels <=np.min(
                    (self.n_spatial_pixels - 1, centroid + pts_width)))
                
                g = fitter(g_init,
                           self.spatial_pixels[pts],
                           chunked_data[pts, chunk_i])
                plt.figure()
                plt.plot(self.spatial_pixels[pts],
                           chunked_data[pts, chunk_i])
                plt.plot(self.spatial_pixels[pts],
                           g(self.spatial_pixels[pts]))
                print(g)
                if i == 3:
                    break
            break
        centroids = np.array(centroids)
        n_centroids = np.array(n_centroids)
        return chunked_data, centroids, n_centroids


class KOALA_Raw(RawFits):
    def __init__(self, path=None, data=None):
        if data is not None:
            self.data = data
        else:
            with fits.open(path) as file:
                self.data = np.array(file[0].data, dtype=float)
        # AAOmega details
        self.spectral_axis = 1
        self.spatial_axis = 0
        self.n_spatial_pixels, self.n_spectral_pixels = self.data.shape
        self.n_fibres = 1000
        self.pixel_between_fibres = 3.
        self.spatial_pixels = np.arange(0, self.n_spatial_pixels, 1)
        self.spectral_pixels = np.arange(0, self.n_spectral_pixels, 1)

master_dark = master(
    '/home/pablo/obs_data/HI-KIDS/raw/20180310/ccd_1/darks')
master_detector_flat = master(
    '/home/pablo/obs_data/HI-KIDS/raw/20180310/ccd_1/detector_flat')
master_skyflats = master(
    '/home/pablo/obs_data/HI-KIDS/raw/20180310/ccd_1/skyflats')

# koala_raw = KOALA_Raw(
#     '/home/pablo/obs_data/HI-KIDS/raw/20180310/ccd_1/10mar10063.fits')
koala_raw = KOALA_Raw(
    data=master_skyflats)

# koala_raw.apply_dark(master_dark)
koala_raw.apply_flat(master_detector_flat)
r, peaks, n_peaks = koala_raw.get_tramlines(n_chunks=1000)

plt.figure()
plt.plot(n_peaks)

plt.figure(figsize=(8, 8))
plt.imshow(r, cmap='nipy_spectral', aspect='auto', interpolation='none')
for i in range(peaks.size):
    plt.scatter(i * np.ones_like(peaks[i]), peaks[i], s=1, c='white')
plt.ylim(2000, 2300)

chunk = 39

plt.figure()
plt.plot(r[:, chunk])
plt.plot(peaks[chunk], r[peaks[chunk], chunk], '*')

sep = np.mean(np.diff(peaks[chunk]))
theoretical_peaks = np.arange(peaks[chunk][0], 1000*sep + peaks[chunk][0], sep,
                              dtype=int)
plt.plot(theoretical_peaks, r[theoretical_peaks, chunk], '.', c='pink')
plt.annotate('Nº peaks={}'.format(peaks[chunk].size), xy=(.5, .9),
             xycoords='axes fraction')
# plt.yscale('log')
plt.xlim(100, 150)
