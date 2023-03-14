#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:33:47 2022

@author: pablo
"""

# =============================================================================
# Basics packages
# =============================================================================
import os
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# =============================================================================
# 5. Applying throughput
# =============================================================================
from koala.corrections.throughput import Throughput
# =============================================================================
# 6. Correcting for extinction
# =============================================================================
from koala.corrections.atmospheric_corrections import AtmosphericExtinction, get_adr
# =============================================================================
# 7. Telluric correction (only for red data) U
# =============================================================================
from koala.corrections.sky import Tellurics
# =============================================================================
# Sky substraction
# =============================================================================
from koala.corrections.sky import SkyFromObject
# =============================================================================
# Cubing
# =============================================================================
from koala.register.registration import register_stars
from koala.cubing import build_cube
# =============================================================================
# Flux calibration
# =============================================================================
from koala.corrections.flux_calibration import FluxCalibration


def reduce_calibration_stars(rss_set, star_names, throughput,
                             output_path=None):
    """TODO..."""
    output = {}
    n_stars = len(rss_set)
    atm_ext_corr = AtmosphericExtinction()
    all_telluric_corrections = []
    # Corrections -------------------------------------------------------------
    for i in range(n_stars):
        for j in range(len(rss_set[i])):
            # Apply throughput
            rss_set[i][j] = Throughput.apply(rss_set[i][j], throughput)
            # Atmospheric Extinction
            atm_ext_corr.get_atmospheric_extinction(
                airmass=rss_set[i][j].info['airmass'])
            rss_set[i][j] = atm_ext_corr.apply(rss_set[i][j])
            # Telluric correction
            telluric_correction = Tellurics(rss_set[i][j])
            _, fig = telluric_correction.telluric_from_model(
                plot=True, width=30)
            if output_path is not None:
                fig.savefig(os.path.join(
                    output_path, 'telluric_{}.png'.format(star_names[i])),
                    bbox_inches='tight')
            rss_set[i][j] = telluric_correction.apply(rss_set[i][j])
            all_telluric_corrections.append(
                telluric_correction.telluric_correction)
            # Sky emission
            skymodel = SkyFromObject(rss_set[i][j])
            pct_sky = skymodel.estimate_sky()
            skymodel.intensity = pct_sky[1]
            skymodel.variance = (pct_sky[1] - pct_sky[0])**2
            rss_set[i][j] = skymodel.substract_sky(rss_set[i][j])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("Sky Model for {}".format(star_names[i]))
            ax.plot(skymodel.rss.wavelength, skymodel.intensity, c='k')
            ax.plot(skymodel.rss.wavelength, skymodel.variance**0.5, c='r')
            if output_path is not None:
                fig.savefig(os.path.join(
                    output_path, 'sky_{}.png'.format(star_names[i])),
                    bbox_inches='tight')
            plt.close(fig)

    output['mean_telluric'] = np.nanmean(all_telluric_corrections, axis=0)
    output['std_telluric'] = np.nanstd(all_telluric_corrections, axis=0)

    # -------------------------------------------------------------------------
    # Registration and Cubing
    star_cubes = []

    for i in range(n_stars):
        register_stars(rss_set[i], moffat=False, plot=False)

        adr_corr_x = []
        adr_corr_y = []
        for rss in rss_set[i]:
            adr_pol_x, adr_pol_y, fig = get_adr(rss, max_adr=0.5, pol_deg=2,
                                                plot=True)
            adr_corr_x.append(adr_pol_x)
            adr_corr_y.append(adr_pol_y)
            if output_path is not None:
                fig.savefig(os.path.join(
                    output_path, 'adr_{}.png'.format(rss.info['name'])),
                    bbox_inches='tight')

        star_cubes.append(
            build_cube(rss_set=rss_set[i],
                       reference_coords=(0, 0),
                       reference_pa=0, cube_size_arcsec=(30, 30),
                       pixel_size_arcsec=.2,
                       name=rss_set[i][0].info['name'].split(' ')[0],
                       adr_x_set=adr_corr_x,
                       adr_y_set=adr_corr_y)
            )

    # Flux calibration
    response_params = dict(pol_deg=None, gauss_smooth_sigma=30,
                           plot=False)
    fcal = FluxCalibration()
    results = fcal.auto(data=star_cubes,
                        calib_stars=star_names,
                        fnames=None,
                        save=True,
                        response_params=response_params)

    responses = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Spectral response function")

    for i, res in enumerate(results.values()):
        resp = res['extraction']['optimal'][:, 0] / res['interp']
        responses.append(resp)
        # Plot individual responses
        ax.plot(res['extraction']['mean_wave'], resp, lw=2,
                label=star_names[i])

    mean_response = np.nanmean(responses, axis=0)
    smooth_response = savgol_filter(mean_response, 11, 2)
    resp_function = interp1d(res['extraction']['mean_wave'],
                             smooth_response, kind='cubic',
                             fill_value='extrapolate')
    dummy_wl = np.linspace(star_cubes[0].wavelength[0],
                           star_cubes[0].wavelength[-1])
    output['response'] = resp_function

    ax.plot(res['extraction']['mean_wave'],
            mean_response, c='purple', lw=2, alpha=0.8,
            label='Mean')
    ax.plot(res['extraction']['mean_wave'],
            smooth_response, '-.',
            c='fuchsia', lw=1, alpha=0.8, label='Smooth')
    ax.plot(dummy_wl,
            resp_function(dummy_wl),
            c='k', lw=1, alpha=1, label='Extrapolated')
    plt.ylim(0, resp.max()*1.1)
    plt.minorticks_on()
    plt.legend()
    plt.grid(which='both', visible=True)
    if output_path is not None:
        fig.savefig(os.path.join(output_path, 'response_function.png'),
                    bbox_inches='tight')
    plt.close(fig)

    # Plot collapsed cube and integrated spectra
    fig, axs = plt.subplots(nrows=n_stars, ncols=2,
                            figsize=(9, 4 * n_stars))
    for i, cube in enumerate(star_cubes):
        fcal.apply(response=resp_function(cube.wavelength),
                   data_container=cube)
        if output is not None:
            cube.to_fits(fname=os.path.join(
                output_path, '{}_calstar.fits.gz'.format(star_names[i])))
        wave = cube.wavelength
        median_star = np.nanmedian(cube.intensity_corrected, axis=0)
        collapsed_spectra = np.nansum(cube.intensity_corrected, axis=(1, 2))
        # Plot
        axs[i, 0].annotate('{}'.format(star_names[i]), xy=(.05, .95),
                           xycoords='axes fraction', va='top', ha='left')
        mappable = axs[i, 0].imshow(
            np.log10(median_star),
            cmap='gnuplot',
            vmin=np.log10(np.nanpercentile(median_star, 70)),
            vmax=np.log10(np.nanpercentile(median_star, 99.99)),
            aspect='auto')
        axs[i, 1].plot(wave, collapsed_spectra, c='k')
        axs[i, 1].set_ylim(np.nanpercentile(collapsed_spectra, [1., 99.]))
        plt.colorbar(mappable, ax=axs[i, 0])
    if output_path is not None:
        fig.savefig(os.path.join(output_path, 'stars.pdf'),
                    bbox_inches='tight')
    plt.close(fig)

    return output
