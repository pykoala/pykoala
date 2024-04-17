#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 09:13:57 2022

@author: pablo
"""

import numpy as np

from glob import glob
from astropy.io import ascii

stars = glob('modular_koala/input_data/spectrophotometric_stars/*')

for star in stars:
    table = ascii.read(star)
    break