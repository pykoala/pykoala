#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 07:33:43 2022

@author: pablo
"""

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np

hdul = fits.open('reduced_data/385R/27feb20035red.fits')

central_wl = hdul[0].header['CRVAL1']
deltawl = hdul[0].header['CDELT1']
central_pix = hdul[0].header['CRPIX1']
n_pix = hdul[0].header['NAXIS1']
wl = central_wl + deltawl * np.arange(central_pix - n_pix, n_pix - central_pix,
                                      1)
fibre_data = hdul[0].data

max_flux = np.nanmax(fibre_data, axis=0)
min_flux = np.nanmin(fibre_data, axis=0)

norm_data = (fibre_data - min_flux[np.newaxis, :]) / (
    max_flux[np.newaxis, :] - min_flux[np.newaxis, :])

wl_pix = 305

plt.figure()
plt.title(r'$\lambda={:.2f}$'.format(wl[wl_pix]))
plt.hist(np.log10(norm_data[:, wl_pix]), range=[-4, 0], bins=100, log=True,
         density=True, cumulative=True)

background = np.nanpercentile(norm_data, 50, axis=0
                              ) * (max_flux - min_flux) + min_flux

fiber_numb = 600
plt.figure()
plt.subplot(211)
plt.plot(wl, fibre_data[fiber_numb, :], 'k', lw=0.5)
plt.plot(wl, background, 'r', lw=0.5)

# plt.xlim(6540, 6590)
plt.subplot(212)
plt.plot(wl, fibre_data[fiber_numb, :] - background)

plt.figure()
plt.imshow(np.log10(fibre_data - background[np.newaxis, :]))
