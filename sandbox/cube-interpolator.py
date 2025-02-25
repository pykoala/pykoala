#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:02:20 2025

@author: pcorchoc
"""
from matplotlib import pyplot as plt
import numpy as np
from pykoala.cubing import CubeInterpolator, DrizzlingKernel
from pykoala.instruments.mock import mock_rss
from astropy import units as u

rss = mock_rss()

interpolator = CubeInterpolator(rss_set=[rss],
                                kernel=DrizzlingKernel(
                                scale=4 << u.dimensionless_unscaled,
                                pixel_scale_arcsec=1))


cube = interpolator.build_cube()

plt.figure(figsize=(5, 5))
plt.scatter(interpolator.rss_inter_products[0]["fib_pix_col"],
            interpolator.rss_inter_products[0]["fib_pix_row"])

plt.figure()
plt.imshow(cube.intensity[0].value)
plt.colorbar()

plt.figure()
plt.imshow(np.nansum(cube.intensity, axis=0).value)
plt.colorbar()

test_cube = np.zeros((1000, *cube.intensity.shape[1:]))

for w, fibre_int in zip(interpolator.rss_inter_products[0]["fibre_weights"], rss.intensity):
    weights = np.zeros(cube.intensity.shape[1:])
    weights[w[0][1], w[0][2]] = w[0][3]
    
    test_cube[:, w[0][1], w[0][2]] += fibre_int[:, np.newaxis, np.newaxis].value * w[0][3]
    plt.figure()
    plt.imshow(np.log10(weights))
    plt.show()
    