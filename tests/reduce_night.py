#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:15:24 2022

@author: pablo
"""

# General
import numpy as np
import yaml
import os

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# KOALA

from koala.koala_ifu import koala_rss

from koala.corrections.throughput import Throughput
# =============================================================================
# 6. Correcting for extinction
# =============================================================================
from koala.corrections.atmospheric_corrections import (AtmosphericExtinction,
                                                       get_adr)
# =============================================================================
# 7. Telluric correction (only for red data) U
# =============================================================================
from koala.corrections.sky import Tellurics
# =============================================================================
# Sky substraction
# =============================================================================
from koala.corrections.sky import SkyFromObject, SkyOffset
# =============================================================================
# Cubing
# =============================================================================
from koala.register.registration import register_stars
from koala.cubing import build_cube
# Extra
from koala.corrections.flux_calibration import FluxCalibration
from reduce_calib_stars import reduce_calibration_stars

from report import report_cube
# =============================================================================
# Night info
# =============================================================================
path_to_night = "/home/pablo/Research/obs_data/HI-KIDS/RSS/sci"


with open(os.path.join(path_to_night, "REDUCTION_FLAGS.yml"), "r") as stream:
    try:
        reduction_flag_log = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

output = 'night_red_test'
pdf = PdfPages(output + '/report.pdf')

if not os.path.isdir(output):
    print("New directory created: {}".format(output))
    os.mkdir(output)

# =============================================================================
# Througput
# =============================================================================
# Load skyflats to use as fibre throughput
throughput_rss = []
for file in reduction_flag_log.keys():
    if (reduction_flag_log[file]['NAME'] == 'SKYFLAT') & (
            reduction_flag_log[file]['FLAG'] == 'OK'):
        file_ori = os.path.join(path_to_night,
                                reduction_flag_log[file]['PATH'])
        file_red = file_ori.replace(".fits", "red.fits")
        throughput_rss.append(koala_rss(file_red))

# Create a master throughput

throughput, throughput_std = Throughput.create_throughput_from_flat(throughput_rss,
                                                         clear_nan=True)

plt.figure(figsize=(8, 5))
plt.subplot(121, title='Fibre Throughput')
plt.imshow(throughput, cmap='nipy_spectral', vmin=0.8, vmax=1.2, aspect='auto')
plt.colorbar(orientation='horizontal')
plt.subplot(122, title='Fibre Throughput hist')
plt.hist(throughput.flatten(), bins='auto', range=[0.5, 1.5])
plt.ylabel("# pix")
pdf.savefig()
# plt.savefig(os.path.join(output, "fibre_throughput.png"), bbox_inches='tight')
plt.close()
# =============================================================================
# Calibration stars
# =============================================================================
stars_rss = []
star_names = ['HR7596', 'EG274', 'HR9087']
file_n = [[23, 24, 25], [26, 27, 28], [47, 50, 51]]

for files in file_n:
    star_rss = []
    for file in files:
        path = os.path.join(path_to_night,
                            '07sep100{}red.fits'.format(file))
        print(path)
        star_rss.append(koala_rss(path))
    stars_rss.append(star_rss)


calibration = reduce_calibration_stars(stars_rss, star_names, throughput,
                                       output_path=output)

# %%===========================================================================
# Science data
# =============================================================================

with open("list_to_reduce.yml", "r") as stream:
    try:
        reduction_list = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


atm_ext_corr = AtmosphericExtinction()
science_cubes = []
for galaxy, files in reduction_list.items():
    science_rss = []
    adr_x_set = []
    adr_y_set = []

    for i, file in enumerate(files['data']):
        path = os.path.join(path_to_night,
                            '07sep100{}red.fits'.format(file))
        print(path)
        rss = koala_rss(path)
        rss = Throughput.apply(rss, throughput)
        atm_ext_corr.get_atmospheric_extinction(
            airmass=rss.info['airmass'])
        rss = atm_ext_corr.apply(rss)
        # Telluric correction
        telluric_correction = Tellurics(rss)
        telluric_correction.telluric_correction = calibration['mean_telluric']
        rss = telluric_correction.apply(rss)

        # Sky emission
        if len(files['offset']) == 0:
            # skymodel = SkyFromObject(rss)
            # pct_sky = skymodel.estimate_sky()
            # mode_sky, h_sky, c_bins = skymodel.estimate_sky_hist()
            # skymodel.intensity = pct_sky[1]
            # skymodel.variance = (pct_sky[1] - pct_sky[0])**2
            # rss = skymodel.substract_sky(rss)
            pass
        else:
            offset_path = os.path.join(path_to_night,
                                       '07sep100{}red.fits'
                                       .format(files['offset'][0]))
            offset_rss = koala_rss(offset_path)
            offset_rss = Throughput.apply(offset_rss, throughput)
            atm_ext_corr.get_atmospheric_extinction(
                airmass=offset_rss.info['airmass'])
            offset_rss = atm_ext_corr.apply(offset_rss)
            skymodel = SkyOffset(rss)
            skymodel.estimate_sky()
            rss = skymodel.substract_sky(rss)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Sky Model for {}".format(rss.info['name']))
        ax.plot(skymodel.rss.wavelength, skymodel.intensity, c='k', label='Sky')
        ax.plot(skymodel.rss.wavelength, skymodel.variance**0.5, c='r',
                label='Sky err')
        ax.legend()
        ax.set_yscale('log')
        # fig.savefig(os.path.join(
        #         output, 'sky_{}.png'.format(rss.info['name'])),
        #         bbox_inches='tight')
        pdf.savefig()
        plt.close()

        adr_x, adr_y, fig = get_adr(rss, plot=True)

        adr_x_set.append(adr_x)
        adr_y_set.append(adr_y)
        science_rss.append(rss)

    # -------------------------------------------------------------------------
    # Registration and Cubing

    # INTERACTIVE
    # new_centres = register_interactive(science_rss)
    # register_new_centres(science_rss, centroids=new_centres)

    # Registering with astropy
    # from koala.register.registration import register_extended_source
    # offsets = register_extended_source(science_rss)

    register_stars(science_rss, moffat=False, plot=False, com_power=4)

    cube = build_cube(rss_set=science_rss,
                      # reference_coords=(
                      #     science_rss[0].info['cen_ra'],
                      #     science_rss[0].info['cen_dec']),
                      # reference_pa=science_rss[0].info['pos_angle'],
                      reference_coords=(0., 0.),
                      reference_pa=0.,
                      cube_size_arcsec=(50, 30),
                      pixel_size_arcsec=.5,
                      adr_x_set=adr_x_set, adr_y_set=adr_y_set)
    cube.info['name'] = 'gal'
    FluxCalibration.apply(calibration['response'](cube.wavelength),
                          cube)
    cube.to_fits(fname=os.path.join(output, '{}_gal.fits.gz'.format(galaxy)))

    # science_cubes.append(cube)     
    fig = report_cube(cube)
    pdf.savefig()
    plt.close()

pdf.close()
