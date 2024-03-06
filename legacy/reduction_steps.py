#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 14:40:13 2022

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
from modular_koala.throughput import get_from_sky_flat  #These are in modular_koala.corrections folder now 
from modular_koala.throughput import apply_throughput   #These are in modular_koala.corrections folder now 
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
from modular_koala.sky import SkyFromObject, SkyOffset, uves_sky_lines
# =============================================================================
# Cubing
# =============================================================================
from modular_koala.cubing import interpolate_rss, build_cube
from modular_koala.rss import RSS


# List and description of the sample RSS in the red arm
# -------------------------------------------------------------------------------

# modular_koala/input_data/sample_RSS/385R/27feb20025red.fits HD60753 C 3W WIDE
# modular_koala/input_data/sample_RSS/385R/27feb20026red.fits HD60753 B 3S WIDE
# modular_koala/input_data/sample_RSS/385R/27feb20027red.fits HD60753 A WIDE

# modular_koala/input_data/sample_RSS/385R/27feb20028red.fits HILT600 A
# modular_koala/input_data/sample_RSS/385R/27feb20029red.fits HILT600 B 3S
# modular_koala/input_data/sample_RSS/385R/27feb20030red.fits HILT600 C 3W

# modular_koala/input_data/sample_RSS/385R/27feb20031red.fits Tol30 A PA0

# modular_koala/input_data/sample_RSS/385R/27feb20032red.fits He2-10 B 1.5S
# modular_koala/input_data/sample_RSS/385R/27feb20033red.fits He2-10 C 1.5S 3E
# modular_koala/input_data/sample_RSS/385R/27feb20034red.fits He2-10 D 18S 3E
# modular_koala/input_data/sample_RSS/385R/27feb20035red.fits He2-10 E 1.5E
# modular_koala/input_data/sample_RSS/385R/27feb20036red.fits He2-10 F 1.5E 1.5S

# modular_koala/input_data/sample_RSS/385R/combined_skyflat_red.fits SKYFLAT
## ============================================================================
# BLUE CCD

std_star_1_rss = []
for i in range(25, 28):
    file_path = (
        'modular_koala/input_data/sample_RSS/580V/27feb100{}red.fits'
        .format(i))
    std_star_1_rss.append(koala_rss(file_path))

std_star_2_rss = []
for i in range(28, 31):
    file_path = (
        'modular_koala/input_data/sample_RSS/580V/27feb100{}red.fits'
        .format(i))
    std_star_2_rss.append(koala_rss(file_path))

sky_flat_rss = koala_rss(
    'modular_koala/input_data/sample_RSS/580V/combined_skyflat_blue.fits')

science_rss = []
for i in range(32, 37):
    file_path = (
        'modular_koala/input_data/sample_RSS/580V/27feb100{}red.fits'
        .format(i))
    science_rss.append(koala_rss(file_path))

# %%===========================================================================
# Fibre Throuput
# =============================================================================
throughput_2D = get_from_sky_flat(sky_flat_rss)

for rss in std_star_1_rss:
    rss = apply_throughput(rss, throughput_2D)

for rss in std_star_2_rss:
    rss = apply_throughput(rss, throughput_2D)

for rss in science_rss:
    rss = apply_throughput(rss, throughput_2D)


def plot_change(rss, title=''):
    """..."""
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(rss.wavelength,
             np.nanmean(rss.intensity_corrected, axis=0), '-',
             label='Corrected', color='k')
    plt.plot(rss.wavelength,
             np.nanmean(rss.intensity, axis=0),
             label='Original', color='r', alpha=0.7, lw=2)
    plt.legend()
    plt.show()


plot_change(rss, title='Throughput')

# %%===========================================================================
# Atmospheric Extinction
# =============================================================================
atm_ext_corr = AtmosphericExtinction()

for rss in std_star_1_rss:
    atm_ext_corr.get_atmospheric_extinction(airmass=rss.info['airmass'])
    atm_ext_corr.apply_correction(rss)

for rss in std_star_2_rss:
    atm_ext_corr.get_atmospheric_extinction(airmass=rss.info['airmass'])
    atm_ext_corr.apply_correction(rss)

for rss in science_rss:
    atm_ext_corr.get_atmospheric_extinction(airmass=rss.info['airmass'])
    atm_ext_corr.apply_correction(rss)

plot_change(rss, title='Atm Corr')

# %%===========================================================================
# Telluric CorrectionBase
# =============================================================================
telluric_correction = Tellurics(std_star_1_rss[0])
telluric_correction.telluric_from_model(plot=True)
for rss in std_star_1_rss + std_star_2_rss + science_rss:
    telluric_correction.apply(rss)

plot_change(rss, title='Telluric Corr')
# %%===========================================================================
# Sky substraction
# =============================================================================

# Since we have a sky-offset for science data, will only correct the stars

for rss in std_star_1_rss + std_star_2_rss + science_rss:
    skymodel = SkyFromObject(rss)
    skymodel.load_sky_lines()
    # Quick sky subtraction based on the 16, 50 and 84 percentiles
    pct_sky = skymodel.estimate_sky()
    skymodel.intensity = pct_sky[1]
    skymodel.variance = (pct_sky[1] - pct_sky[0])**2
    skymodel.fit_continuum(skymodel.intensity, skymodel.variance**0.5)

    rss = skymodel.substract_sky(rss)

    plot_change(rss, title='Sky substraction: ' + rss.info['name'])

# %%===========================================================================
# Cubing
# =============================================================================
rss = std_star_1_rss[1]
std_collapsed = np.nansum(rss.intensity_corrected, axis=1) #this needs to be double checked. Are we collapsing the spectra here?
std_collapsed_var = np.nansum(rss.variance_corrected, axis=1) 

x0, y0 = rss.get_centre_of_mass(wavelength_step=rss.wavelength.size,
                                stat=np.mean) #test this method 

xx = rss.info['fib_ra_offset']
yy = rss.info['fib_dec_offset']

r = np.sqrt(xx**2 + yy**2)
r_sort = np.argsort(r)
cum_f = np.cumsum(std_collapsed[r_sort])

def f_sky(sky, r):
    return np.pi * sky * r**2

from modular_koala.ancillary import cumulative_1d_moffat
from modular_koala.registration import fit_moffat_profile, register_stars

def moffa_sky(r, sky, l, beta, alpha):
    cum_f = f_sky(sky, r) + cumulative_moffat(r**2, l, beta, alpha)
    return cum_f

from scipy.optimize import curve_fit

popt, _ = curve_fit(f=moffa_sky, xdata=r[r_sort], ydata=cum_f,
                 p0=[2e4, 6e8, 20, 4])

mof_params = [6e8, 20, 4]
plt.figure()
plt.plot(r, moffa_sky(r, *popt), 'k-')
plt.plot(r[r_sort], cum_f, c='r')
plt.plot(r, f_sky(popt[0], r), '-', c='b')
plt.plot(r, cumulative_moffat(r**2, *popt[1:]), '-', c='orange')

fit = fit_moffat_profile(rss, plot=True)


plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.hexbin(xx,
           yy,
           C=np.log10(std_collapsed), gridsize=30, cmap='nipy_spectral')
plt.plot(0, 0, 'k+', markersize=12)
plt.colorbar()
plt.subplot(132)
plt.hexbin(xx,
           yy,
           C=np.log10(fit(xx, yy)), gridsize=30, cmap='nipy_spectral')
plt.plot(0, 0, 'k+', markersize=12)
plt.subplot(133)
plt.hexbin(xx,
           yy,
           C=(std_collapsed - fit(xx, yy))**2 / std_collapsed_var,
           gridsize=30, cmap='seismic', vmin=0, vmax=100)
plt.plot(0, 0, 'k+', markersize=12)
plt.colorbar()

register_stars(std_star_1_rss, moffat=False, plot=True)
rss = std_star_1_rss[1]
xx = rss.info['fib_ra_offset']
yy = rss.info['fib_dec_offset']


plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.hexbin(xx,
           yy,
           C=np.log10(std_collapsed), gridsize=30, cmap='nipy_spectral')
plt.plot(0, 0, 'k+', markersize=12)
plt.colorbar()
plt.subplot(132)
plt.hexbin(xx,
           yy,
           C=np.log10(fit(xx, yy)), gridsize=30, cmap='nipy_spectral')
plt.plot(0, 0, 'k+', markersize=12)
plt.subplot(133)
plt.hexbin(xx,
           yy,
           C=(std_collapsed - fit(xx, yy))**2 / std_collapsed_var,
           gridsize=30, cmap='seismic', vmin=0, vmax=100)
plt.plot(0, 0, 'k+', markersize=12)
plt.colorbar()
# %%


std_star_1_cube = build_cube(
    rss_set=std_star_1_rss,
    reference_coords=(
        std_star_1_rss[0].info['cen_ra'],
        std_star_1_rss[0].info['cen_dec']),
    reference_pa=0, cube_size_arcsec=(30, 30), pixel_size_arcsec=.2)

wave = std_star_1_cube.wavelength

plt.figure()
plt.imshow(np.log10(np.nansum(std_star_1_cube.intensity, axis=0)))
plt.show()


x_com, y_com = np.nanmedian(
    std_star_1_cube.get_centre_of_mass(), axis=1)

plt.imshow(np.nanmedian(std_star_1_cube.intensity, axis=0),
           interpolation='none')
plt.plot(x_com, y_com, 'r+')


from modular_koala.ancillary import (cumulative_1d_moffat_sky,
                                     cumulative_1d_moffat,
                                     cumulative_1d_sky)

xx, yy = np.meshgrid(np.arange(0, std_star_1_cube.n_cols),
                     np.arange(0, std_star_1_cube.n_rows))

wave_window = 50
popt = []
mean_wave = []
for lambda_ in range(0, wave.size, wave_window):
    mean_wave.append(np.mean(wave[lambda_: lambda_ + wave_window]))
    cube_data = np.nanmean(
        std_star_1_cube.intensity[lambda_: lambda_ + wave_window], axis=0)

    x_com, y_com = (np.nansum(cube_data * xx) / np.nansum(cube_data),
                    np.nansum(cube_data * yy) / np.nansum(cube_data))
    r2 = (xx - x_com)**2 + (yy - y_com)**2
    r = r2.flatten()**0.5
    r_sort = np.argsort(r)
    cum_f = np.nancumsum(cube_data.flatten()[r_sort])

    all_popt, _ = curve_fit(f=cumulative_1d_moffat_sky,
                            xdata=r[r_sort]**2, ydata=cum_f,
                            p0=[6e8, 20, 4, 2e4])
    popt.append(all_popt)
    moffat_popt, _ = curve_fit(f=cumulative_1d_moffat,
                               xdata=r[r_sort], ydata=cum_f,
                               p0=[6e8, 20, 4])
    print(all_popt)
popt = np.array(popt)
mean_wave = np.array(mean_wave)

plt.figure()
plt.plot(r[r_sort], cum_f, lw=3, c='k')
plt.plot(r[r_sort], cumulative_1d_moffat(r[r_sort], *moffat_popt), c='b')
plt.plot(r[r_sort], cumulative_1d_moffat_sky(r[r_sort]**2, *all_popt), c='r')
plt.plot(r[r_sort], cumulative_1d_moffat(r[r_sort]**2, *all_popt[:-1]), c='orange')
plt.plot(r[r_sort], cumulative_1d_sky(r[r_sort]**2, all_popt[-1]), c='cyan')
plt.xscale('log')

fit = fit_moffat_profile(std_star_1_cube, wave_range=(4000, 5000),
                         plot=True)