# -*- coding: utf-8 -*-
"""
Functions related to moving spectrums.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from synphot import observation
from synphot import spectrum

from .plots import plot_plot, plot_telluric_correction, plot_weights_for_getting_smooth_spectrum

def rebin_spec(wave, specin, wavnew):
    """
    Rebin a spectrum with a new wavelength array

    Args:
        wave (array): wavelength arrau
        specin (array): Input spectrum to be shifted
        shift (float): Shift. Same units as wave?

    Returns:
        New spectrum at shifted wavelength values
    """
    spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    f = np.ones(len(wave))
    filt = spectrum.ArraySpectralElement(wave, f, waveunits="angstrom")
    obs = observation.Observation(spec, filt, binset=wavnew, force="taper")
    return obs.binflux


def rebin_spec_shift(wave, specin, shift):
    """
    Rebin a spectrum and shift in wavelength. Makes a new wavelength array and then passes this to rebin_spec

    Args:
        wave (array): wavelength arrau
        specin (array): Input spectrum to be shifted
        shift (float): Shift. Same units as wave?

    Returns:
        New spectrum at shifted wavelength values

    """
    wavnew = wave + shift
    obs = rebin_spec(wave, specin, wavnew)
    # Updating from pull request #16. rebin_spec returns a .binflux object. This function tried to create a
    # binflux.binflux object.
    return obs

def compare_fix_2dfdr_wavelengths(rss1, rss2):
    """
    Compare small fixes we've made to the 2dFdr wavelengths between two RSS files.

    Args:
        rss1 (RSS instance): An instance of the RSS class
        rss2 (RSS instance): An instance of the RSS class

    Returns:
        None
    """

    print("\n> Comparing small fixing of the 2dFdr wavelengths between two rss...")

    xfibre = list(range(0, rss1.n_spectra))
    rss1.wavelength_parameters[0]

    a0x, a1x, a2x = (
        rss1.wavelength_parameters[0],
        rss1.wavelength_parameters[1],
        rss1.wavelength_parameters[2],
    )
    aa0x, aa1x, aa2x = (
        rss2.wavelength_parameters[0],
        rss2.wavelength_parameters[1],
        rss2.wavelength_parameters[2],
    )

    fx = a0x + a1x * np.array(xfibre) + a2x * np.array(xfibre) ** 2
    fx2 = aa0x + aa1x * np.array(xfibre) + aa2x * np.array(xfibre) ** 2
    dif = fx - fx2

    plt.figure(figsize=(10, 4))
    plt.plot(xfibre, dif)
    plot_plot(
        xfibre,
        dif,
        ptitle="Fit 1 - Fit 2",
        xmin=-20,
        xmax=1000,
        xlabel="Fibre",
        ylabel="Dif",
    )

    resolution = rss1.wavelength[1] - rss1.wavelength[0]
    error = (np.nanmedian(dif)/resolution) * 100.0
    print("\n> The median rms is {:8.6f} A,  resolution = {:5.2f} A,  error = {:5.3} %".format(
        np.nanmedian(dif), resolution, error
    ))


def obtain_telluric_correction(wlm, telluric_correction_list, plot=True):
    """
    Take a list of telluric correction spectra and make a single median telluric spectrum

    Args:
        wlm (array): A wavelength array. Only used for plotting- should refactor this!
        telluric_correction_list (list): A list of telluric correction spectra
        plot (bool, default=True): Whether or not to plot the resulting spectrum.
    """
    telluric_correction = np.nanmedian(telluric_correction_list, axis=0)
    if plot:
        fig = plot_telluric_correction(wlm, telluric_correction_list, telluric_correction, fig_size=12)

    print("\n\t>Telluric correction = {}".format(telluric_correction))
    print("\n\tTelluric correction obtained!")
    return telluric_correction


def smooth_spectrum(
    wlm,
    s,
    wave_min=0,
    wave_max=0,
    step=50,
    exclude_wlm=[[0, 0]],
    weight_fit_median=0.5,
    plot=False,
    verbose=False,
):
    """
    Smooth a spectrum

    Parameters
    ----------
    wlm
    s
    wave_min
    wave_max
    step
    exclude_wlm
    weight_fit_median
    plot
    verbose

    Returns
    -------

    """

    if verbose:
        print("\n> Computing smooth spectrum...")

    if wave_min == 0:
        wave_min = wlm[0]
    if wave_max == 0:
        wave_max = wlm[-1]

    running_wave = []
    running_step_median = []
    cuts = np.int(((wave_max - wave_min)/step))

    exclude = 0
    corte_index = -1
    for corte in range(cuts + 1):
        next_wave = wave_min + step * corte
        if next_wave < wave_max:

            if (
                next_wave > exclude_wlm[exclude][0]
                and next_wave < exclude_wlm[exclude][1]
            ):
                if verbose:
                    print("  Skipping {} as it is in the exclusion range [ {} , {} ]".format(
                        next_wave, exclude_wlm[exclude][0], exclude_wlm[exclude][1]))

            else:
                corte_index = corte_index + 1
                running_wave.append(next_wave)
                # print running_wave
                region = np.where(
                    (wlm > running_wave[corte_index] - np.int(step/2))
                    & (wlm < running_wave[corte_index] + np.int(step/2))
                )
                running_step_median.append(np.nanmedian(s[region]))
                if next_wave > exclude_wlm[exclude][1]:
                    exclude = exclude + 1
                    if verbose:
                        print("--- End exclusion range {}".format(exclude))
                    if exclude == len(exclude_wlm):
                        exclude = len(exclude_wlm) - 1

    running_wave.append(wave_max)
    region = np.where((wlm > wave_max - step) & (wlm < wave_max + 0.1))
    running_step_median.append(np.nanmedian(s[region]))

    # print running_wave
    # print running_step_median
    # Check not nan
    _running_wave_ = []
    _running_step_median_ = []
    for i in range(len(running_wave)):
        if np.isnan(running_step_median[i]):
            if verbose:
                print("  There is a nan in {}".format(running_wave[i]))
        else:
            _running_wave_.append(running_wave[i])
            _running_step_median_.append(running_step_median[i])

    a7x, a6x, a5x, a4x, a3x, a2x, a1x, a0x = np.polyfit(
        _running_wave_, _running_step_median_, 7
    )
    fit_median = (
        a0x
        + a1x * wlm
        + a2x * wlm ** 2
        + a3x * wlm ** 3
        + a4x * wlm ** 4
        + a5x * wlm ** 5
        + a6x * wlm ** 6
        + a7x * wlm ** 7
    )

    interpolated_continuum_smooth = interpolate.splrep(
        _running_wave_, _running_step_median_, s=0.02
    )
    fit_median_interpolated = interpolate.splev(
        wlm, interpolated_continuum_smooth, der=0
    )

    if plot:

        fig = plot_weights_for_getting_smooth_spectrum(
            wlm,
            s,
            running_wave,
            running_step_median,
            fit_median,
            fit_median_interpolated,
            weight_fit_median,
            wave_min,
            wave_max,
            exclude_wlm)

        print("  Weights for getting smooth spectrum:  fit_median = {}    fit_median_interpolated = {}".format(
            weight_fit_median, 1 - weight_fit_median))

    return (
        weight_fit_median * fit_median
        + (1 - weight_fit_median) * fit_median_interpolated
    )  # (fit_median+fit_median_interpolated)/2      # Decide if fit_median or fit_median_interpolated
