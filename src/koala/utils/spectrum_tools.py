# -*- coding: utf-8 -*-
"""
Functions related to moving spectrums.
"""
import numpy as np
import matplotlib.pyplot as plt
from synphot import observation
from synphot import spectrum

from .plots import plot_plot

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
