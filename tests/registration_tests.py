#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:34:41 2022

@author: pablo
"""

from modular_koala.registration import fit_moffat_profile
import numpy as np


class Dummy(object):
    wavelength = np.linspace(0, 1)

dummy = Dummy()
fit_moffat_profile(dummy)