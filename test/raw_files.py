#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:52:03 2022

@author: pablo
"""

from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt

hdul = fits.open('/home/pablo/obs_data/HI-KIDS/raw/20180310/ccd_1/10mar10035.fits')
data = hdul[0].data
#%%
plt.figure()
plt.imshow(np.log10(data), cmap='nipy_spectral', aspect='auto',
           vmax=2.7)
plt.xlim(1000, 1500)
plt.ylim(2330, 2350)
plt.colorbar()

plt.plot(np.sum(data[:, 1100:1150], axis=1))
plt.xlim(2350, 2380)