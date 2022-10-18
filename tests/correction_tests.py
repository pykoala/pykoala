#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:57:03 2022

@author: pablo
"""

import numpy as np
from matplotlib import pyplot as plt

from modular_koala.koala_ifu import koala_rss
from modular_koala.atmospheric_corrections import AtmosphericExtinction

file_path = 'modular_koala/input_data/sample_RSS/580V/27feb10025red.fits'
file_path = 'modular_koala/input_data/sample_RSS/385R/27feb20025red.fits'
std_rss = koala_rss(file_path)

atm_ext_corr = AtmosphericExtinction(airmass=std_rss.info['airmass'])
atm_ext_corr.apply_correction(std_rss)


plt.figure()
plt.plot(std_rss.wavelength,
         atm_ext_corr.extinction_correction_model(std_rss.wavelength))
plt.xlabel(r'Wavelength $(\mathrm{\AA})$', fontsize=15)
plt.ylabel(r'$\frac{F_{corr}}{F_{obs}}$', fontsize=15)