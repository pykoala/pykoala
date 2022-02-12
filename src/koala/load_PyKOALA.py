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


#import os.path
import sys

# pykoala_path = "/DATA/KOALA/Python/GitHub/koala/src/koala/"  # Provided by the user when calling this file


# # We are running in  []/koala/src/koala/, modules are in koala
# # With this we can run koala packages without importing them
# # Also avoiding including many times the directories in the path

original_system_path =[]
for item in sys.path:
    #print("Original",item)
    original_system_path.append(item)

# # This is from where Python will look for "koala"
sys.path.append(pykoala_path[:-6])


# # 1. Load file with constant data
# exec(compile(open(os.path.join(pykoala_path, "constants.py"), "rb").read(),
#              os.path.join(pykoala_path, "constants.py"), 'exec'))   # This just reads the file.
from koala.constants import * 

# 2. Load file with I/O tasks and version and developers
# exec(compile(open(os.path.join(pykoala_path, "io.py"), "rb").read(),
#              os.path.join(pykoala_path, "io.py"), 'exec'))
from koala.io import * 

# 3. Load file with plot_plot and basic_statistics (task included in plot_plot.py)
# exec(compile(open(os.path.join(pykoala_path, "plot_plot.py"), "rb").read(),
#              os.path.join(pykoala_path, "plot_plot.py"), 'exec'))
from koala.plot_plot import *

# 4. Load file with 1D spectrum tasks
# exec(compile(open(os.path.join(pykoala_path, "onedspec.py"), "rb").read(),
#              os.path.join(pykoala_path, "onedspec.py"), 'exec'))
from koala.onedspec import * 

# 5. Load file with RSS class & RSS tasks
# exec(compile(open(os.path.join(pykoala_path, "RSS.py"), "rb").read(),
#              os.path.join(pykoala_path, "RSS.py"), 'exec'))
from koala.RSS import *

# 6. Load file with KOALA_RSS class & KOALA_RSS specific tasks
# exec(compile(open(os.path.join(pykoala_path, "KOALA_RSS.py"), "rb").read(),
#              os.path.join(pykoala_path, "KOALA_RSS.py"), 'exec'))
from koala.KOALA_RSS import *

# 7. Load file with Interpolated_cube class & cube specific tasks
# exec(compile(open(os.path.join(pykoala_path, "cube.py"), "rb").read(),
#              os.path.join(pykoala_path, "cube.py"), 'exec'))
from koala.cube import *

# 8. Load file with map tasks
# exec(compile(open(os.path.join(pykoala_path, "maps.py"), "rb").read(),
#              os.path.join(pykoala_path, "maps.py"), 'exec'))
from koala.maps import *

# 9. Load the 4 AUTOMATIC SCRIPTS 
# exec(compile(open(os.path.join(pykoala_path, "automatic_scripts", "run_automatic_star.py"), "rb").read(),
#              os.path.join(pykoala_path, "automatic_scripts", "run_automatic_star.py"), 'exec'))
from koala.automatic_scripts.run_automatic_star import *

# exec(compile(open(os.path.join(pykoala_path, "automatic_scripts", "koala_reduce.py"), "rb").read(),
#              os.path.join(pykoala_path, "automatic_scripts", "koala_reduce.py"), 'exec'))
from koala.automatic_scripts.koala_reduce import *

# exec(compile(open(os.path.join(pykoala_path, "automatic_scripts", "automatic_koala_reduce.py"), "rb").read(),
#              os.path.join(pykoala_path, "automatic_scripts", "automatic_koala_reduce.py"), 'exec'))
from koala.automatic_scripts.automatic_koala_reduce import *
# exec(compile(open(os.path.join(pykoala_path, "automatic_scripts", "automatic_calibration_night.py"), "rb").read(),
#              os.path.join(pykoala_path, "automatic_scripts", "automatic_calibration_night.py"), 'exec'))
from koala.automatic_scripts.automatic_calibration_night import *



# Clean the path and leave only what matters

sys.path = []
for item in original_system_path:
    sys.path.append(item)
sys.path.append(pykoala_path[:-6])
#sys.path.append(os.path.join(pykoala_path,"RSS"))

#for item in sys.path:
#    print(item)


    