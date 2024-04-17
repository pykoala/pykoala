#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:12:45 2022

@author: pablo
"""

from astroquery.skyview import SkyView
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from astropy import units as u
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import Quadrangle, SphericalCircle

paths = SkyView.get_images(position='NGC1311', survey=['DSS'])

# %%
wcs = WCS(paths[0][0])

fig = plt.figure()
ax = fig.add_subplot(111, projection=wcs)
mappable = ax.imshow(paths[0][0].data, cmap='nipy_spectral',
                     norm=LogNorm())

r = SphericalCircle((paths[0][0].header['CRVAL1'] * u.deg,
                     paths[0][0].header['CRVAL2'] * u.deg),
                     2.6 * u.arcsec,
                     edgecolor='k', facecolor='none',
                     transform=ax.get_transform('fk5'))

ax.add_patch(r)

ax.set_xlim(110, 190)
ax.set_ylim(110, 190)
plt.colorbar(mappable, ax=ax)