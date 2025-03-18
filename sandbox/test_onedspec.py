#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 08:00:43 2024

@author: pcorchoc
"""
from pykoala.spectra import onedspec
from astropy import units as u
import numpy as np
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import find_lines_threshold
from specutils.spectra import SpectralRegion
from matplotlib import pyplot as plt

emissionLine = onedspec.EmissionLine(central_wavelength=6563. * u.angstrom,
                                     flux=2.5 * u.erg / u.s / u.cm**2,
                                     fwhm=5.5 * u.angstrom)
emissionLine2 = onedspec.EmissionLine(central_wavelength=6580. * u.angstrom,
                                      flux=6.6 * u.erg / u.s / u.cm**2,
                                      fwhm=6.5 * u.angstrom)

wavelength = np.arange(6520, 6600, 0.2) * u.angstrom

spectrum = emissionLine.sample_to(wavelength) + emissionLine2.sample_to(wavelength)
spectrum += 1 * np.ones_like(spectrum.flux)
spectrum += np.random.normal(.5, .01, size=spectrum.flux.size) * spectrum.flux.unit

plt.figure()
plt.plot(spectrum.wavelength, spectrum.flux)

line = onedspec.EmissionLine(central_wavelength=6564. * u.angstrom,
                             flux=1.5 * u.erg / u.s / u.cm**2 ,
                             fwhm=.5 * u.angstrom,
                             central_wavelength_bounds=(6559, 6568),
                             fwhm_bounds=(.1 * u.angstrom, 2.5 * u.angstrom)
                             )
line2 = onedspec.EmissionLine(central_wavelength=6578. * u.angstrom,
                             flux=0.5 * u.erg / u.s / u.cm**2 ,
                             fwhm=3.5 * u.angstrom,
                             central_wavelength_bounds=(6574, 6583),
                             profile=onedspec.LorentzianLineProfile,
                             fwhm_bounds=(.1 * u.angstrom, 2.5 * u.angstrom)
                             # flux_bounds=(1 * spectrum.flux.unit, 10 * spectrum.flux.unit)
                             )
  

fit_em_line, cont, fit_info, fig = onedspec.fit_emission_lines(
    spectrum, [line, line2], fit_continuum=True,
    wave_range=[6550 * u.angstrom, 6590 * u.angstrom])


# %%=============================================================================
# Finding emission lines
# =============================================================================

peaks_inf = onedspec.find_emission_lines(spectrum, fit_continuum=False)