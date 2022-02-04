#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PyKOALA: KOALA data processing and analysis 
# by Angel Lopez-Sanchez, Yago Ascasibar, Pablo Corcho-Caballero
# Extra work by Ben Lawson (MQ PACE student)
# Plus Taylah Beard and Matt Owers (sky substraction)
# Documenting: Nathan Pidcock, Giacomo Biviano, Jamila Scammill, Diana Dalae, Barr Perez
# version = "It will read it from the PyKOALA code..."
# This is Python 3.7
# To convert to Python 3 run this in a command line :
# cp PyKOALA_2021_02_02.py PyKOALA_2021_02_02_P3.py
# 2to3 -w PyKOALA_2021_02_02_P3.py
# Edit the file replacing:
#                 exec('print("    {:2}       {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(i+1, '+cube_aligned_object[i]+'.RA_centre_deg,'+cube_aligned_object[i]+'.DEC_centre_deg,'+cube_aligned_object[i]+'.pixel_size_arcsec,'+cube_aligned_object[i]+'.kernel_size_arcsec))')

# This is the first full version of the code DIVIDED


# -----------------------------------------------------------------------------
# Load all PyKOALA tasks
# -----------------------------------------------------------------------------


# pykoala_path = "/DATA/KOALA/Python/GitHub/koala/src/koala/"  # Provided by the user when calling this file

# 0. Read __init__ with the version ( or file version.txt )
import os.path

with open(os.path.join(pykoala_path, 'version.txt')) as f:
    version = f.read()

# 1. Add file with constant data
exec(compile(open(os.path.join(pykoala_path, "constants.py"), "rb").read(),
             os.path.join(pykoala_path, "constants.py"), 'exec'))   # This just reads the file.
#from pykoala import constants 

# 2. Add file with I/O tasks
exec(compile(open(os.path.join(pykoala_path, "io.py"), "rb").read(),
             os.path.join(pykoala_path, "io.py"), 'exec'))
# #from pykoala import io 

# 3. Add file with plot_plot and basic_statistics (task included in plot_plot.py)
exec(compile(open(os.path.join(pykoala_path, "plot_plot.py"), "rb").read(),
             os.path.join(pykoala_path, "plot_plot.py"), 'exec'))
# #from pykoala import plot_plot as plot_plot

# 4. Add file with 1D spectrum tasks
exec(compile(open(os.path.join(pykoala_path, "onedspec.py"), "rb").read(),
             os.path.join(pykoala_path, "onedspec.py"), 'exec'))
#from pykoala import onedspec 

# 5. Add file with RSS class & RSS tasks
exec(compile(open(os.path.join(pykoala_path, "RSS.py"), "rb").read(),
             os.path.join(pykoala_path, "RSS.py"), 'exec'))

# 5. Add file with KOALA_RSS class & KOALA_RSS specific tasks
exec(compile(open(os.path.join(pykoala_path, "KOALA_RSS.py"), "rb").read(),
             os.path.join(pykoala_path, "KOALA_RSS.py"), 'exec'))

# 7. Add file with Interpolated_cube class & cube specific tasks
exec(compile(open(os.path.join(pykoala_path, "cube.py"), "rb").read(),
             os.path.join(pykoala_path, "cube.py"), 'exec'))

# 8. Add the 4 AUTOMATIC SCRIPTS 
exec(compile(open(os.path.join(pykoala_path, "automatic_scripts", "automatic_calibration_night.py"), "rb").read(),
             os.path.join(pykoala_path, "automatic_scripts", "automatic_calibration_night.py"), 'exec'))

exec(compile(open(os.path.join(pykoala_path, "automatic_scripts", "run_automatic_star.py"), "rb").read(),
             os.path.join(pykoala_path, "automatic_scripts", "run_automatic_star.py"), 'exec'))

exec(compile(open(os.path.join(pykoala_path, "automatic_scripts", "automatic_koala_reduce.py"), "rb").read(),
             os.path.join(pykoala_path, "automatic_scripts", "automatic_koala_reduce.py"), 'exec'))

exec(compile(open(os.path.join(pykoala_path, "automatic_scripts", "koala_reduce.py"), "rb").read(),
             os.path.join(pykoala_path, "automatic_scripts", "koala_reduce.py"), 'exec'))


    