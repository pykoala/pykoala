"""Basic steps to reduce calibration stars."""
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


# Data used
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
grating = '385R'
# grating = '580V'

prefixes = {'580V': '1', '385R': '2'}
# Read data and store data as RSS objects
std_star_1_rss = []
for i in range(25, 28):
    file_path = (
        '../src/koala/input_data/sample_RSS/{}/27feb{}00{}red.fits'
        .format(grating, prefixes[grating], i))
    std_star_1_rss.append(koala_rss(file_path))

std_star_2_rss = []
for i in range(28, 31):
    file_path = (
        '../src/koala/input_data/sample_RSS/{}/27feb{}00{}red.fits'
        .format(grating, prefixes[grating], i))
    std_star_2_rss.append(koala_rss(file_path))

if grating == '385R':
    sky_flat_rss = koala_rss(
        '../src/koala/input_data/sample_RSS/{}/combined_skyflat_red.fits'
        .format(grating))
elif grating == '580V':
    sky_flat_rss = koala_rss(
        '../src/koala/input_data/sample_RSS/{}/combined_skyflat_blue.fits'
        .format(grating))


# %%===========================================================================
# Fibre Throuput
# =============================================================================
throughput_2D = get_from_sky_flat(sky_flat_rss)

for i in range(3):
    std_star_1_rss[i] = apply_throughput(std_star_1_rss[i], throughput_2D)
    std_star_2_rss[i] = apply_throughput(std_star_2_rss[i], throughput_2D)

# %%===========================================================================
# Atmospheric Extinction
# =============================================================================
atm_ext_corr = AtmosphericExtinction()

for i in range(3):
    atm_ext_corr.get_atmospheric_extinction(
        airmass=std_star_1_rss[i].info['airmass'])
    std_star_1_rss[i] = atm_ext_corr.apply(std_star_1_rss[i])

    atm_ext_corr.get_atmospheric_extinction(
        airmass=std_star_2_rss[i].info['airmass'])
    std_star_2_rss[i] = atm_ext_corr.apply(std_star_2_rss[i])


# %%===========================================================================
# Telluric Correction
# =============================================================================
# telluric_correction = Tellurics(std_star_1_rss[0])
# telluric_correction.telluric_from_smoothed_spec(plot=True)
# telluric_correction.telluric_from_model(plot=True)

all_tel_corr = []
for i in range(3):
    telluric_correction = Tellurics(std_star_1_rss[i])
    # telluric_correction.telluric_from_smoothed_spec(plot=True)
    telluric_correction.telluric_from_model(plot=True, width=30)
    std_star_1_rss[i] = telluric_correction.apply(std_star_1_rss[i])
    all_tel_corr.append(telluric_correction.telluric_correction)
    telluric_correction = Tellurics(std_star_2_rss[i])
    # telluric_correction.telluric_from_smoothed_spec(plot=True)
    telluric_correction.telluric_from_model(plot=True, width=30)
    std_star_2_rss[i] = telluric_correction.apply(std_star_2_rss[i])
    all_tel_corr.append(telluric_correction.telluric_correction)

master_telluric_corr = np.nanmean(all_tel_corr, axis=0)
std_master_telluric_corr = np.nanstd(all_tel_corr, axis=0)
plt.figure()
plt.plot(master_telluric_corr)
plt.plot(std_master_telluric_corr)
# plt.yscale('log')
# %%===========================================================================
# Sky substraction
# =============================================================================

for i in range(3):
    skymodel = SkyFromObject(std_star_1_rss[i])
    skymodel.load_sky_lines()
    # Quick sky subtraction based on the 16, 50 and 84 percentiles
    pct_sky = skymodel.estimate_sky()
    mode_sky, hist_sky, bins = skymodel.estimate_sky_hist()
    # skymodel.intensity = pct_sky[1]
    # skymodel.variance = (pct_sky[1] - pct_sky[0])**2
    # std_star_1_rss[i] = skymodel.substract_sky(std_star_1_rss[i])
    plt.figure()
    # plt.plot(np.nanmean(std_star_1_rss[i].intensity_corrected, axis=0))
    plt.plot(mode_sky / pct_sky[1])
    plt.yscale('log')

    skymodel = SkyFromObject(std_star_2_rss[i])
    skymodel.load_sky_lines()
    # Quick sky subtraction based on the 16, 50 and 84 percentiles
    pct_sky = skymodel.estimate_sky()
    mode_sky, hist_sky, bins = skymodel.estimate_sky_hist()
    # skymodel.intensity = pct_sky[1]
    # skymodel.variance = (pct_sky[1] - pct_sky[0])**2
    # std_star_2_rss[i] = skymodel.substract_sky(std_star_2_rss[i])
    plt.figure()
    # plt.plot(np.nanmean(std_star_2_rss[i].intensity_corrected, axis=0))
    plt.plot(mode_sky / pct_sky[1])
    plt.yscale('log')

    break
# %%===========================================================================
# Cubing
# =============================================================================

# Register each RSS file before merging into a cube
from koala.registration import fit_moffat_profile, register_stars

# To register a set of rss it is possible to use a 2D moffat fit
# or simply the center of light

for rss_set in [std_star_1_rss, std_star_2_rss]:
    register_stars(rss_set, moffat=False, plot=True)

std_star_1_cube = build_cube(
    rss_set=std_star_1_rss,
    reference_coords=(0, 0),
    reference_pa=0, cube_size_arcsec=(30, 30), pixel_size_arcsec=.2)

std_star_2_cube = build_cube(
    rss_set=std_star_2_rss,
    reference_coords=(0, 0),
    reference_pa=0, cube_size_arcsec=(30, 30), pixel_size_arcsec=.2)

median_star = np.nanmedian(std_star_1_cube.intensity, axis=0)
wave = std_star_1_cube.wavelength

plt.figure(figsize=(8, 4))
plt.subplot(121, title='Collapsed image')
plt.imshow(np.log10(median_star),
           cmap='nipy_spectral',
           vmin=np.log10(np.nanpercentile(median_star, 50)),
           aspect='auto')
plt.plot(median_star.shape[0] / 2, median_star.shape[1] / 2, 'r+')
plt.colorbar()
plt.subplot(122, title='Integrated spectrum')
plt.plot(std_star_1_cube.wavelength,
         np.nansum(std_star_1_cube.intensity, axis=(1, 2)))
plt.show()

median_star = np.nanmedian(std_star_2_cube.intensity, axis=0)
plt.figure(figsize=(8, 4))
plt.subplot(121, title='Collapsed image')
plt.imshow(np.log10(median_star),
           cmap='nipy_spectral',
           vmin=np.log10(np.nanpercentile(median_star, 50))
           )
plt.plot(median_star.shape[0] / 2, median_star.shape[1] / 2, 'r+')
plt.colorbar()
plt.subplot(122, title='Integrated spectrum')
plt.plot(std_star_2_cube.wavelength,
         np.nansum(std_star_2_cube.intensity, axis=(1, 2)))
plt.show()

# %%===========================================================================
# Flux calibration
# =============================================================================
from koala.flux_calibration import FluxCalibration
from koala.ancillary import (cumulative_1d_moffat_sky,
                                     cumulative_1d_moffat,
                                     cumulative_1d_sky,
                                     flux_conserving_interpolation)
fcal = FluxCalibration()
fnames = ["HD60753_" + grating, "HILT600_" + grating]
response_params = dict(pol_deg=None, gauss_smooth_sigma=30, plot=False)

results = fcal.auto(data=[std_star_1_cube, std_star_2_cube],
                    calib_stars=["HD60753", "HILT600"],
                    fnames=fnames,
                    save=True,
                    response_params=response_params)



plt.figure(figsize=(8, 5))
responses = []
for res in results.values():
    resp = res['extraction']['optimal'][:, 0] / res['interp']
    responses.append(resp)
    err = np.sqrt(res['extraction']['variance'][:, 0])
    plt.plot(res['extraction']['mean_wave'], resp, lw=0.7)

    plt.fill_between(
        res['extraction']['mean_wave'],
        (res['extraction']['optimal'][:, 0] - err) / res['interp'],
        (res['extraction']['optimal'][:, 0] + err) / res['interp'],
        color='k')

plt.plot(res['extraction']['mean_wave'],
         responses[0] * 0.5 + responses[1] * 0.5)
plt.ylim(resp.min()*0.9, resp.max()*1.1)

wl1, r1 = np.loadtxt(fnames[0], unpack=True)
wl2, r2 = np.loadtxt(fnames[1], unpack=True)

plt.plot(wl1, r1)
plt.plot(wl2, r2)
plt.minorticks_on()
plt.grid(which='both', visible=True)
