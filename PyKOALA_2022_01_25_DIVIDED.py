#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PyKOALA: KOALA data processing and analysis 
# by Angel Lopez-Sanchez, Yago Ascasibar, Pablo Corcho-Caballero
# Extra work by Ben Lawson (MQ PACE student)
# Plus Taylah Beard and Matt Owers (sky substraction)
# Documenting: Nathan Pidcock, Giacomo Biviano, Jamila Scammill, Diana Dalae, Barr Perez
version = "It will read it from the PyKOALA code..."
# This is Python 3.7
# To convert to Python 3 run this in a command line :
# cp PyKOALA_2021_02_02.py PyKOALA_2021_02_02_P3.py
# 2to3 -w PyKOALA_2021_02_02_P3.py
# Edit the file replacing:
#                 exec('print("    {:2}       {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(i+1, '+cube_aligned_object[i]+'.RA_centre_deg,'+cube_aligned_object[i]+'.DEC_centre_deg,'+cube_aligned_object[i]+'.pixel_size_arcsec,'+cube_aligned_object[i]+'.kernel_size_arcsec))')

# This is the first full version of the code DIVIDED


# -----------------------------------------------------------------------------
# Start timer
# -----------------------------------------------------------------------------
from timeit import default_timer as timer    
start = timer()
# -----------------------------------------------------------------------------
# Import PyKOALA
# -----------------------------------------------------------------------------
#import PyKOALA_2021_02_02 as PK    # This will NOT import variables and all tasks 
                                    # would need to be called using PK.task() ....
#exec(compile(open('PyKOALA_2021_02_13_P3.py', "rb").read(), 'PyKOALA_2021_02_13_P3.py', 'exec'))   # This just reads the file. 
                                    # If the PyKOALA code is not changed, there is no need of reading it again


pykoala_path = "/DATA/KOALA/Python/GitHub/koala/"


# 1. Add file with constant data
exec(compile(open(pykoala_path+"constants.py", "rb").read(), pykoala_path+"constants.py", 'exec'))   # This just reads the file. 
#from pykoala import constants 

# 2. Add file with I/O tasks
exec(compile(open(pykoala_path+"io.py", "rb").read(), pykoala_path+"io.py", 'exec'))   
# #from pykoala import io 

# 3. Add file with plot_plot and basic_statistics (task included in plot_plot.py)
exec(compile(open(pykoala_path+"plot_plot.py", "rb").read(), pykoala_path+"plot_plot.py", 'exec'))   
# #from pykoala import plot_plot as plot_plot

# 4. Add file with 1D spectrum tasks
exec(compile(open(pykoala_path+"onedspec.py", "rb").read(), pykoala_path+"onedspec.py", 'exec'))  
#from pykoala import onedspec 

# 5. Add file with RSS class & RSS tasks
exec(compile(open(pykoala_path+"rss.py", "rb").read(), pykoala_path+"rss.py", 'exec'))   

# 5. Add file with KOALA_RSS class & KOALA_RSS specific tasks
exec(compile(open(pykoala_path+"koala_rss.py", "rb").read(), pykoala_path+"koala_rss.py", 'exec'))   

# 7. Add file with Interpolated_cube class & cube specific tasks
exec(compile(open(pykoala_path+"cube.py", "rb").read(), pykoala_path+"cube.py", 'exec'))   

# 8. Add the 4 AUTOMATIC SCRIPTS 
exec(compile(open(pykoala_path+"automatic_scripts/automatic_calibration_night.py", "rb").read(), pykoala_path+"automatic_scripts/automatic_calibration_night.py", 'exec'))   

exec(compile(open(pykoala_path+"automatic_scripts/run_automatic_star.py", "rb").read(), pykoala_path+"automatic_scripts/run_automatic_star.py", 'exec'))   

exec(compile(open(pykoala_path+"automatic_scripts/automatic_koala_reduce.py", "rb").read(), pykoala_path+"automatic_scripts/automatic_koala_reduce.py", 'exec'))   

exec(compile(open(pykoala_path+"automatic_scripts/koala_reduce.py", "rb").read(), pykoala_path+"automatic_scripts/koala_reduce.py", 'exec'))   


version="Version 1.1 - 25 Januay 2022 - First one AFTER breaking the code"

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n> Testing PyKOALA. Running", version)


    path = "/DATA/KOALA/DATA/RSS/20180227/385R/"
    throughput_2D_file = path+"throughput_2D_20180227_385R_PABLO.fits"
    telluric_correction_file = path+"telluric_correction_20180227_385R.dat"

    file_in = path+"27feb20028red.fits"
    file_out= path+"27feb20028red_TEST_PABLO_TCWX.fits"
    file_out_final = path+"27feb20028red_TEST_PABLO_TCWX_S_NR.fits"

    # file_in = path+"27feb20029red.fits"
    # file_out= path+"27feb20029red_TEST_PABLO_TCWX.fits"
    # file_out_final = path+"27feb20029red_TEST_PABLO_TCWX_S_NR.fits"

    # file_in = path+"27feb20030red.fits"
    # file_out= path+"27feb20030red_TEST_PABLO_TCWX.fits"
    # file_out_final = path+"27feb20030red_TEST_PABLO_TCWX_S_NR.fits"

    
    # file_skyflat="/DATA/KOALA/DATA/RSS/20180227/385R/combined_skyflat_red.fits"
    # throughput_2D_385R, skyflat_385R = get_throughput_2D(file_skyflat, plot=False, also_return_skyflat=True, 
    #                                                       correct_ccd_defects=True, fix_wavelengths=True, 
    #                                                       sol=[0.0853325247121367,-0.0009925545410042428,1.582994591921196e-07],
    #                                                       throughput_2D_file=throughput_2D_file)    

    
    test = KOALA_RSS(file_in, #file_in, #save_rss_to_fits_file="auto",  
                      #save_rss_to_fits_file=file_out_final, #file_out_final, #save_rss_to_fits_file=file_out, 
                        apply_throughput=True, 
                        throughput_2D_file=throughput_2D_file,       # if throughput_2D_file given, use SOL in fits file for fixing wave
                        #throughput_2D=throughput_2D_20180227_385R,
                        correct_ccd_defects = True, 
                        fix_wavelengths = True, #sol=[-1],
                        #sol=[0.0853325247121367,-0.0009925545410042428,1.582994591921196e-07],
                        do_extinction=True,
                        telluric_correction_file=telluric_correction_file,
                        sky_method="self", sky_fibres= fibres_best_sky_100, # range(0,100), #n_sky=50,
                        correct_negative_sky = True, 
                        order_fit_negative_sky = 7, kernel_negative_sky=51, individual_check=True, 
                        use_fit_for_negative_sky=True, force_sky_fibres_to_zero=True,
                        remove_negative_median_values = True,
                        #clean_sky_residuals = True, features_to_fix = "big_telluric",
                        fix_edges = True,
                        clean_extreme_negatives=True, percentile_min=0.9,
                        clean_cosmics=True,
                        width_bl=0., kernel_median_cosmics=5, cosmic_higher_than = 100., extra_factor =	2.,
                        #plot_final_rss = True,
                      plot=True, warnings=False, verbose=True)



    # rss_list = [path+"27feb20028red_TCWX_S_NR",path+"27feb20029red_TCWX_S_NR",path+"27feb20030red_TCWX_S_NR"]

    # combined_cube_test=KOALA_reduce(rss_list, path=path, 
    #                                 fits_file="combined_cube_test_PABLO.fits",
    #                                 rss_clean=True,
    #                                 pixel_size_arcsec=.7, kernel_size_arcsec=1.4,
    #                                 plot= True, plot_rss=True, plot_weight=False, norm=colors.LogNorm(), fig_size=12,
    #                                 warnings=False, verbose = True)
  
    # combined_cube_test.combined_cube.plot_map(norm=colors.LogNorm(),vmin=3E4,vmax=1E8, cmap=fuego_color_map)
    

    
end= timer()
print("\n> Elapsing time = ",end-start, "s")        