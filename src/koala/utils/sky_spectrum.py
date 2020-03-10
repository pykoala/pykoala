# -*- coding: utf-8 -*-
"""
File contains functions related to the sky spectrum.
"""
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import copy

from .plots import plot_plot
from .flux import search_peaks, fluxes

def median_filter(intensity_corrected, n_spectra, n_wave, win_sky=151):
    """
    Apply a median filter to a two dimensional (n_spectra x n_wave) array of sky spectra. The median filtering occurs at each wavelength slice between the n_spectra sky spectra. The actual filtering is handled by scipy.signal
    TODO: Check this is correct- I'm unsure why medfilt_sky should be a 2D array too? SPV

    Args:
        intensity_corrected (array of shape n_spectra x n_wave): Spectra we want to median filter
    """

    medfilt_sky = np.zeros((n_spectra, n_wave))
    for wave in range(n_wave):
        medfilt_sky[:, wave] = sig.medfilt(
            intensity_corrected[:, wave], kernel_size=win_sky
        )

    # replace crappy edge fibres with 0.5*win'th medsky
    for fibre_sky in range(n_spectra):
        if fibre_sky < np.rint(0.5 * win_sky):
            j = int(np.rint(0.5 * win_sky))
            medfilt_sky[fibre_sky, ] = copy.deepcopy(medfilt_sky[j, ])
        if fibre_sky > n_spectra - np.rint(0.5 * win_sky):
            j = int(np.rint(n_spectra - np.rint(0.5 * win_sky)))
            medfilt_sky[fibre_sky, ] = copy.deepcopy(medfilt_sky[j, ])
    return medfilt_sky


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def scale_sky_spectrum(
    wlm,
    sky_spectrum,
    spectra,
    cut_sky=4.0,
    fmax=10,
    fmin=1,
    valid_wave_min=0,
    valid_wave_max=0,
    fibre_list=[100, 200, 300, 400, 500, 600, 700, 800, 900],
    plot=True,
    verbose=True,
    warnings=True,
):

    # # Read sky lines provided by 2dFdr
    #    sky_line_,flux_sky_line_ = read_table("sky_lines_2dfdr.dat", ["f", "f"] )
    # # Choose those lines in the range
    #    sky_line=[]
    #    flux_sky_line=[]
    #    valid_wave_min = 6240
    #    valid_wave_max = 7355
    #    for i in range(len(sky_line_)):
    #        if valid_wave_min < sky_line_[i] < valid_wave_max:
    #            sky_line.append(sky_line_[i])
    #            flux_sky_line.append(flux_sky_line_[i])

    if valid_wave_min == 0:
        valid_wave_min = wlm[0]
    if valid_wave_max == 0:
        valid_wave_max = wlm[-1]

    if verbose:
        print("\n> Identifying sky lines using cut_sky = {} , allowed SKY/OBJ values = [ {} , {} ]".format(cut_sky, fmin, fmax))
    if verbose:
        print("  Using fibres = {}".format(fibre_list))

    peaks, peaks_name, peaks_rest, continuum_limits = search_peaks(
        wlm,
        sky_spectrum,
        plot=plot,
        cut=cut_sky,
        fmax=fmax,
        only_id_lines=False,
        verbose=False,
    )

    ratio_list = []
    valid_peaks = []

    if verbose:
        print("\n        Sky line    Gaussian ratio     Flux ratio")
    n_sky_lines_found = 0
    for i in range(len(peaks)):
        sky_spectrum_data = fluxes(
            wlm,
            sky_spectrum,
            peaks[i],
            fcal=False,
            lowlow=50,
            highhigh=50,
            plot=False,
            verbose=False,
            warnings=warnings,
        )

        object_spectrum_data_gauss = []
        object_spectrum_data_integrated = []
        for fibre in fibre_list:
            object_spectrum_flux = fluxes(
                wlm,
                spectra[fibre],
                peaks[i],
                fcal=False,
                lowlow=50,
                highhigh=50,
                plot=False,
                verbose=False,
                warnings=warnings,
            )
            object_spectrum_data_gauss.append(
                object_spectrum_flux[3]
            )  # Gaussian flux is 3
            object_spectrum_data_integrated.append(
                object_spectrum_flux[7]
            )  # integrated flux is 7
        object_spectrum_data = np.nanmedian(object_spectrum_data_gauss)
        object_spectrum_data_i = np.nanmedian(object_spectrum_data_integrated)

        if fmin < (object_spectrum_data/sky_spectrum_data[3]) < fmax:
            n_sky_lines_found = n_sky_lines_found + 1
            valid_peaks.append(peaks[i])
            ratio_list.append(object_spectrum_data/sky_spectrum_data[3])
            if verbose:
                print("{:3.0f}   {:5.3f}         {:2.3f}             {:2.3f}".format(
                    n_sky_lines_found,
                    peaks[i],
                    (object_spectrum_data/sky_spectrum_data[3]),
                    (object_spectrum_data_i/sky_spectrum_data[7]),
                ))

    # print "ratio_list =", ratio_list
    # fit = np.polyfit(valid_peaks, ratio_list, 0) # This is the same that doing an average/mean
    # fit_line = fit[0]+0*wlm
    fit_line = np.nanmedian(ratio_list)  # We just do a median
    # fit_line = fit[1]+fit[0]*wlm
    # fit_line = fit[2]+fit[1]*wlm+fit[0]*wlm**2
    # fit_line = fit[3]+fit[2]*wlm+fit[1]*wlm**2+fit[0]*wlm**3

    if plot:
        plt.plot(valid_peaks, ratio_list, "+")
        # plt.plot(wlm,fit_line)
        plt.axhline(y=fit_line, color="k", linestyle="--")
        plt.xlim(valid_wave_min - 10, valid_wave_max + 10)
        plt.ylim(np.nanmin(ratio_list) - 0.2, np.nanmax(ratio_list) + 0.2)
        plt.title("Scaling sky spectrum to object spectra")
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel("OBJECT / SKY")
        plt.minorticks_on()
        # plt.show()
        # plt.close()

        if verbose:
            print("  Using this fit to scale sky spectrum to object, the median value is {}  ...".format(fit_line))

    sky_corrected = sky_spectrum * fit_line

    #        plt.plot(wlm,sky_spectrum, "r", alpha=0.3)
    #        plt.plot(wlm,sky_corrected, "g", alpha=0.3)
    #        plt.show()
    #        plt.close()

    return sky_corrected


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def sky_spectrum_from_fibres(
    rss,
    list_spectra,
    win_sky=151,
    xmin=0,
    xmax=0,
    ymin=0,
    ymax=0,
    verbose=True,
    plot=True,
):

    if verbose:
        print("\n> Obtaining 1D sky spectrum using rss file and fibre list = {} ...".format(list_spectra))

    rss.intensity_corrected = median_filter(
        rss.intensity_corrected, rss.n_spectra, rss.n_wave, win_sky=win_sky
    )

    sky = rss.plot_combined_spectrum(list_spectra=list_spectra, median=True, plot=plot)

    return sky


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def sky_spectrum_from_fibres_using_file(
    rss_file,
    fibre_list=[],
    win_sky=151,
    n_sky=0,
    skyflat="",
    apply_throughput=True,
    correct_ccd_defects=False,
    fix_wavelengths=False,
    sol=[0, 0, 0],
    xmin=0,
    xmax=0,
    ymin=0,
    ymax=0,
    verbose=True,
    plot=True,
):
    from koala import KOALA_RSS  # TODO: currently importing like this for workaround of circular imports
    # Similar to in cube_alignement
    # TODO: this function is never called it seems

    if skyflat == "":
        apply_throughput = False
        plot_rss = False
    else:
        apply_throughput = True
        plot_rss = True

    if n_sky != 0:
        sky_method = "self"
        is_sky = False
        if verbose:
            print("\n> Obtaining 1D sky spectrum using {}  lowest fibres in this rss ...".format(n_sky))
    else:
        sky_method = "none"
        is_sky = True
        if verbose:
            print("\n> Obtaining 1D sky spectrum using fibre list = {} ...".format(fibre_list))

    _test_rss_ = KOALA_RSS(
        rss_file,
        apply_throughput=apply_throughput,
        skyflat=skyflat,
        correct_ccd_defects=correct_ccd_defects,
        fix_wavelengths=fix_wavelengths,
        sol=sol,
        sky_method=sky_method,
        n_sky=n_sky,
        is_sky=is_sky,
        win_sky=win_sky,
        do_extinction=False,
        plot=plot_rss,
        verbose=False,
    )

    if n_sky != 0:
        print("\n> Sky fibres used: {}".format(_test_rss_.sky_fibres))
        sky = _test_rss_.sky_emission
    else:
        sky = _test_rss_.plot_combined_spectrum(list_spectra=fibre_list, median=True)

    if plot:
        plt.figure(figsize=(14, 4))
        if n_sky != 0:
            plt.plot(_test_rss_.wavelength, sky, "b", linewidth=2, alpha=0.5)
            ptitle = "Sky spectrum combining using {} lowest fibres".format(n_sky)

        else:
            for i in range(len(fibre_list)):
                plt.plot(
                    _test_rss_.wavelength, _test_rss_.intensity_corrected[i], alpha=0.5
                )
                plt.plot(_test_rss_.wavelength, sky, "b", linewidth=2, alpha=0.5)
            ptitle = "Sky spectrum combining " + np.str(len(fibre_list)) + " fibres"

        plot_plot(_test_rss_.wavelength, sky, ptitle=ptitle)

    print("\n> Sky spectrum obtained!")
    return sky


def obtain_sky_spectrum(
    sky, low_fibres=200, plot=True, fig_size=12, fcal=False, verbose=True
):
    """
    Obtain a sky-spectrum using N fibres with the lowest intensity values.
    We sort skyfibres by their integrated flux. The lowest `low_fibres` fibres are then used to create a sky-spectrum by median-combining them. We then return this 1D spectrum
    TODO: Fix argument types

    Args:
        sky ():
        low_fibres (int): After sorting, the lowest `low_fibres` are combined together
        plot (bool): If True, plot the spectrum
        fcal (bool): Passed to plot_plot
        verbose (bool): If True, print out the regions we've included

    Returns:
        array: a 1D sky spectrum.
    """
    # It uses the lowest low_fibres fibres to get an integrated spectrum
    integrated_intensity_sorted = np.argsort(sky.integrated_fibre)
    region = []
    for fibre in range(low_fibres):
        region.append(integrated_intensity_sorted[fibre])
    sky_spectrum = np.nanmedian(sky.intensity_corrected[region], axis=0)

    print("  We use the {} fibres with the lowest integrated intensity to derive the sky spectrum".format(low_fibres))
    if verbose:
        print("  The list is = {}".format(region))

    if plot:
        plt.figure(figsize=(fig_size, fig_size / 2.5))
        plt.plot(sky.wavelength, sky_spectrum)
        ptitle = "Sky spectrum"
        plot_plot(sky.wavelength, sky_spectrum, ptitle=ptitle, fcal=fcal)

    return sky_spectrum
