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
# Start timer
# -----------------------------------------------------------------------------
import os
from timeit import default_timer as timer
start = timer()
# -----------------------------------------------------------------------------
# Load all PyKOALA tasks   / Import PyKOALA 
# -----------------------------------------------------------------------------

pykoala_path = os.getcwd()

exec(compile(open(os.path.join(pykoala_path, "load_PyKOALA.py"), "rb").read(),
             os.path.join(pykoala_path, "load_PyKOALA.py"), 'exec'))   # This just reads the file.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print("\n\n> Running PyKOALA -", version)
      
    # # VERY IMPORTANT :
    # # First, copy the input data to a local folder (not within PyKOALA)
    # # The RSS data are in pykoala_path/input_data/sample_RSS/
    # # That will be where your data are.
    # # 
    # # Type where your data will be:

        #check folder is correct for Barr -> /Users/barrp/KOALA/DATA/original_data_not_in_GIT/data_worked_on/
    path = "/Users/barrp/KOALA/DATA/original_data_not_in_GIT/data_worked_on/"

    #path = "/DATA/KOALA/Python/GitHub/test_reduce/"

    path = os.path.join('..', '..', 'tests', 'reduced_data')
    if not os.path.isdir(path):
        os.mkdir('../../tests/reduced_data')


    # # If needed, you can copy the example data using this:        
    #os.system("mkdir "+path)
    #os.system("cp -R ./input_data/sample_RSS/* "+path)

    # # For AAOmega, we have TWO folders per night: blue (580V) and red (385R)
    path_red = os.path.join(path, "385R")
    path_blue = os.path.join(path, "580V")

    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    # # Now, it is recommended to start processing the RED data
    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------

    # # List the files in the folder
    #list_fits_files_in_folder(path_red)
    
    # PyKOALA finds 4 objects: HD60753, HILT600 (calibration stars),
    #                          He2-10 (the galaxy),
    #                          SKYFLAT
    
    # # -----------------------------------------------------------------------

    # # Next, run this for AUTOMATICALLY processing calibration of the night
    # automatic_calibration_night(path=path_red, auto=True)
                                # , kernel_throughput = 21)
 
    
    # # This will create 2 (3 for red) files needed for the calibration:
    
    # # 1. The throughput_2D_file:
    throughput_2D_file = os.path.join(path_red, "throughput_2D_20180227_385R.fits")
    # # 2. The flux calibration file:
    flux_calibration_file = os.path.join(path_red, "flux_calibration_20180227_385R_0p7_1k10.dat")
    # # 3. The telluric correction file (only in red):
    telluric_correction_file = os.path.join(path_red, "telluric_correction_20180227_385R.dat")
    
    # # It will also create 2 Python objects:
    # # HD60753_385R_20180227 : Python object with calibration star HD60753
    # # Hilt600_385R_20180227 : Python object with calibration star Hiltner 600
    
    # # Run automatic calibration when throughput 2D has been obtained:
    # automatic_calibration_night(path=path_red, auto=True, 
    #                             throughput_2D_file=throughput_2D_file)
    
    # # Run automatic calibration when throughput 2D has been obtained
    # # AND 2 Python objects (only for checking FLUX calibration)
    # # In this case, adding abs_flux_scale =[1.1, 1.0],
    # # as HD60753 does not work as well as Hilt600
    
    # automatic_calibration_night(path=path_red, auto=True, 
    #                             #list_of_objects =["Hilt600_385R_20180227"],
    #                             #star_list =["Hilt600"],
    #                             grating="385R",
    #                             date="20180227",
    #                             throughput_2D_file=throughput_2D_file,
    #                             abs_flux_scale =[1.1, 1.0], # add this for SCALING stars, if needed
    #                             cal_from_calibrated_starcubes = True)
    
    # # -----------------------------------------------------------------------
    # # Now it is time to process an RSS file, using KOALA_RSS
    # # It is recommended to test it here BEFORE running automatic scripts
    
    file_in   = path_red+"/27feb20031red.fits"
    # file_med  = path_red+"27feb20031red_TCWXU____.fits"
    # file_med2 = path_red+"27feb20031red_TCWXUS___.fits"
    # file_out  = path_red+"27feb20031red_TCWXUS_NR.fits"

    # # As the critical part is the SKY SUBTRACION, first do everything till that
    
    test = KOALA_RSS(file_in, 
                      save_rss_to_fits_file="auto",  
                      # apply_throughput=True, 
                      # throughput_2D_file=throughput_2D_file,       # if throughput_2D_file given, use SOL in fits file for fixing wave
                      #throughput_2D=throughput_2D_20180227_385R,
                      correct_ccd_defects = True, 
                      # fix_wavelengths = True, 
                      #sol=[0.0853325247121367,-0.0009925545410042428,1.582994591921196e-07],
                      do_extinction=True,
                       # telluric_correction_file=telluric_correction_file
                       )

    # # This will automatically create the file "27feb20031red_TCWXU____.fits"
    
    # # Now it is time to check the sky
    # # Running just a "self" n_sky = 25, that is
    # # using the 25 fibres with LOWEST integrated values to get the sky:
    
    # test = KOALA_RSS(file_med, 
    #                  save_rss_to_fits_file="auto",  
    #                  sky_method="self", n_sky=25)
    
    # # This will create file_med2, that has keys "_TCWXUS___"
    # # Change manually the name of that file to compare in DS9 later with 
    # # the other tests we are performing (e.g. "_TCWXUSself___")
    
    # # There are some problems here:
    # # 1. H-alpha is everywhere, at ~6583. We can't use self
    # # 2. Still, many residuals after subtracting IR sky lines.

    # # Solving 1 is tricky, we should have used offset skies (2D)
    
    # # An option is getting the sky of a calibration star or a faint object 
    # # and SCALING 
    # #
    # sky1=KOALA_RSS(path_red+"27feb20030red.fits",
    #                #save_rss_to_fits_file="auto",
    #                apply_throughput=True, 
    #                throughput_2D_file=throughput_2D_file,
    #                correct_ccd_defects = True, 
    #                fix_wavelengths = True, 
    #                do_extinction=True,
    #                is_sky=True,
    #                sky_fibres= fibres_best_sky_100,
    #                plot=True, warnings=False)

    # # This will be our sky spectrum for replacing emission lines in rss
    # sky1_spec=sky1.sky_emission   
    
    # # If we provide sky_spectrum = sky1_spec, 
    # # and choose sky_method="self" or "selffit,
    # # it will run RSS task "replace_el_in_sky_spectrum"
    # # to get the sky spectrum to use.
    
    # # However this task currently DOES NOT PROPERLY WORK... it needs checking
    
    # # Let's obtain the self sky and fit a Gaussian in H-alpha:
        
    # sky1=KOALA_RSS(file_med, sky_method="self", n_sky=25)
    # sky1_spec=sky1.sky_emission 
    # w=test.wavelength
    # plot_plot(w,sky1_spec, vlines=[6583], xmin=6500, xmax=6650)
    
    # fluxes(w,sky1_spec ,6584, lowlow=50,lowhigh=30, highlow=30,highhigh=50, broad=1.5)
    # # Fails with a single Gaussian fit, trying a double Gaussian fit:
    # dfluxes(w,sky1_spec ,6577,6584, lowlow=50,lowhigh=30, highlow=30,highhigh=50)
    # # With this, x0, y0, sigma  = 6583.768393198269, 91.05682233417284, 2.1031790823208825
    # gaussHa = gauss(w, 6583.768393198269, 91.05682233417284, 2.1031790823208825)
    # sky1_spec_good = sky1_spec-gaussHa
    # plot_plot(w,[sky1_spec,gaussHa, sky1_spec_good], vlines=[6584], xmin=6500, xmax=6650,
    #           ptitle="Subtracting H-alpha to the sky emission", ymin=-20, ymax=600)
    
    # # For solving 2, it is recommended to individually fit the sky lines
    # # using Gaussians, applying sky_method="selffit"
    # # and using file "sky_lines_IR_short"  (we call it with "IRshort")
    # # We also need to add parameter brightest_line_wavelength
    # # with the OBSERVED H-alpha wavelength, around 6583

    # test = KOALA_RSS(file_med, 
    #                   save_rss_to_fits_file="auto",  
    #                   sky_spectrum = sky1_spec_good,
    #                   sky_method="1Dfit",
    #                   scale_sky_1D = 1.0,
    #                   brightest_line = "Ha",
    #                   brightest_line_wavelength = 6583,
    #                   sky_lines_file="IRshort")

    # # Don't forget to save the sky as a 1D fits file or text file:
    # spectrum_to_text_file(w, sky1_spec_good, filename=path_red+"27feb20031red_sky.txt", verbose=True )

    # # Finally, we can clean a bit more the sky residuals
    
    # test = KOALA_RSS(file_med2, 
    #                   save_rss_to_fits_file="auto",  
    #                   correct_negative_sky = True, 
    #                   order_fit_negative_sky = 7, kernel_negative_sky=51, individual_check=True, 
    #                   use_fit_for_negative_sky=True, force_sky_fibres_to_zero=True,
    #                   remove_negative_median_values = True,
    #                   clean_sky_residuals = True, features_to_fix = "big_telluric",
    #                   fix_edges = True,
    #                   clean_extreme_negatives=True, percentile_min=0.9,
    #                   clean_cosmics=True,
    #                   width_bl=0., kernel_median_cosmics=5, cosmic_higher_than = 100., extra_factor =	2.)
    
    # # We can compare what we have done opening the files in DS9 or running this:
    
    # print("\n\n - Plotting original RSS:")
    # test = KOALA_RSS(file_in, verbose=False)
    # print(" - Plotting RSS corrected by TCWXU:")
    # test = KOALA_RSS(file_med, verbose=False)
    # print(" - Plotting RSS corrected by TCWXU and 1Dfit sky:")
    # test = KOALA_RSS(file_med2, verbose=False)
    # print(" - Plotting RSS corrected by TCWXU, 1Dfit sky, and sky residuals:")
    # test = KOALA_RSS(file_out, verbose=False)

    # # If we have had the sky spectrum, we could have done it everything together:
        
    # test = KOALA_RSS(file_in, 
    #                  save_rss_to_fits_file="auto",  
    #                  apply_throughput=True, 
    #                  throughput_2D_file=throughput_2D_file,       # if throughput_2D_file given, use SOL in fits file for fixing wave
    #                  #throughput_2D=throughput_2D_20180227_385R,
    #                  correct_ccd_defects = True, 
    #                  fix_wavelengths = True, 
    #                  #sol=[0.0853325247121367,-0.0009925545410042428,1.582994591921196e-07],
    #                  do_extinction=True,
    #                  telluric_correction_file=telluric_correction_file,
    #                  sky_spectrum = sky1_spec_good,
    #                  #sky_spectrum_file = path_red+"27feb20031red_sky.txt",
    #                  sky_method="1Dfit",
    #                  scale_sky_1D = 1.0,
    #                  brightest_line = "Ha",
    #                  brightest_line_wavelength = 6583,
    #                  sky_lines_file="IRshort",
    #                  correct_negative_sky = True, 
    #                  order_fit_negative_sky = 7, kernel_negative_sky=51, individual_check=True, 
    #                  use_fit_for_negative_sky=True, force_sky_fibres_to_zero=True,
    #                  remove_negative_median_values = True,
    #                  clean_sky_residuals = True, features_to_fix = "big_telluric",
    #                  fix_edges = True,
    #                  clean_extreme_negatives=True, percentile_min=0.9,
    #                  clean_cosmics=True,
    #                  width_bl=0., kernel_median_cosmics=5, cosmic_higher_than = 100., extra_factor =	2.,
    #                  plot_final_rss = True,
    #                  plot=True, warnings=False, verbose=True)
    

    # # Now we need to repeat for the remaining 5 files of He 2-10.
    # # Again, the critical part is the SKY SUBTRACTION!
    # # As we are doing the Gaussian fit, we first need to process TCWXU,
    # # then extract self sky, fit Gaussian (double),
    # # and then running again KOALA_RSS with 1Dfit and residuals
    
    # # It's the same for the 5 files, just comment and uncomment these as needed
    
    # file_in   = path_red+"27feb20032red.fits"
    # file_med  = path_red+"27feb20032red_TCWXU____.fits"
    # file_sky  = path_red+"27feb20032red_sky.txt"

    # file_in   = path_red+"27feb20033red.fits"
    # file_med  = path_red+"27feb20033red_TCWXU____.fits"
    # file_sky  = path_red+"27feb20033red_sky.txt"
    
    # file_in   = path_red+"27feb20034red.fits"
    # file_med  = path_red+"27feb20034red_TCWXU____.fits"
    # file_sky  = path_red+"27feb20034red_sky.txt"

    # file_in   = path_red+"27feb20035red.fits"
    # file_med  = path_red+"27feb20035red_TCWXU____.fits"
    # file_sky  = path_red+"27feb20035red_sky.txt"
    
    # file_in   = path_red+"27feb20036red.fits"
    # file_med  = path_red+"27feb20036red_TCWXU____.fits"
    # file_sky  = path_red+"27feb20036red_sky.txt"
    
    # test = KOALA_RSS(file_in, 
    #                   save_rss_to_fits_file="auto",  
    #                   apply_throughput=True, 
    #                   throughput_2D_file=throughput_2D_file,       # if throughput_2D_file given, use SOL in fits file for fixing wave
    #                   correct_ccd_defects = True, 
    #                   fix_wavelengths = True, 
    #                   do_extinction=True,
    #                   telluric_correction_file=telluric_correction_file)
  
    # sky = KOALA_RSS(file_med, sky_method="self", n_sky=25)
    # sky_spec=sky.sky_emission     
    # dfluxes(w,sky_spec ,6577,6584, lowlow=50,lowhigh=30, highlow=30,highhigh=50)

    # For file 32:  x0, y0, sigma  = 6583.816181485515 97.79633090010093 2.1633059703557453
    # gaussHa = gauss(w, 6583.816181485515, 97.79633090010093, 2.1633059703557453)
    # For file 33:  x0, y0, sigma  = 6583.318685704395 118.9126748043965 2.3678850517593446
    # gaussHa = gauss(w, 6583.318685704395, 118.9126748043965, 2.3678850517593446)
    # For file 34:  x0, y0, sigma  = 6583.38576024161 113.68364263932791 2.0602389430115675
    # gaussHa = gauss(w, 6583.38576024161, 113.68364263932791, 2.0602389430115675)
    # For file 35:  x0, y0, sigma  = 6583.320956824095 117.5123159875465 2.19115121698132
    # gaussHa = gauss(w, 6583.320956824095, 117.5123159875465, 2.19115121698132)
    # For file 36:  x0, y0, sigma  = 6583.092264772729 120.6800674492026 2.2647860224835306
    # gaussHa = gauss(w, 6583.092264772729, 120.6800674492026, 2.2647860224835306)

    # sky_spec_good = sky_spec-gaussHa
    # plot_plot(w,[sky_spec,gaussHa, sky_spec_good], vlines=[6584], xmin=6500, xmax=6650,
    #           ptitle="Subtracting H-alpha to the sky emission", ymin=-20, ymax=600)    
    
    # spectrum_to_text_file(w, sky_spec_good, filename=file_sky, verbose=True )
    
    # test = KOALA_RSS(file_med, 
    #                   save_rss_to_fits_file="auto",  
    #                   #sky_spectrum = sky_spec_good,
    #                   sky_spectrum_file = file_sky,
    #                   sky_method="1Dfit",
    #                   scale_sky_1D = 1.0,
    #                   brightest_line = "Ha",
    #                   brightest_line_wavelength = 6583,
    #                   sky_lines_file="IRshort",
    #                   correct_negative_sky = True, 
    #                   order_fit_negative_sky = 7, kernel_negative_sky=51, individual_check=True, 
    #                   use_fit_for_negative_sky=True, force_sky_fibres_to_zero=True,
    #                   remove_negative_median_values = True,
    #                   clean_sky_residuals = True, features_to_fix = "big_telluric",
    #                   fix_edges = True,
    #                   clean_extreme_negatives=True, percentile_min=0.9,
    #                   clean_cosmics=True,
    #                   width_bl=0., kernel_median_cosmics=5, cosmic_higher_than = 100., extra_factor =	2.,
    #                   plot_final_rss = True,
    #                   plot=True, warnings=False, verbose=True)
  

    # # -----------------------------------------------------------------------
    # # Once we have the CLEAN RSS files, we can do the cubing & combine
    # # -----------------------------------------------------------------------

    # # For making a cube, we call Interpolated_cube defining the
    # # pixel and kernel size and adding the flux_calibration
    
    # file_out  = path_red+"27feb20031red_TCWXUS_NR.fits"
    
    # cube_test = Interpolated_cube(file_out, 
    #                               pixel_size_arcsec=0.7,
    #                               kernel_size_arcsec=1.1,
    #                               flux_calibration_file=flux_calibration_file,
    #                               plot=True)
    
    # # For this galaxy, as it is off-center, running ADR automatically will FAIL

    # cube_test = Interpolated_cube(file_out, 
    #                               pixel_size_arcsec=0.7,
    #                               kernel_size_arcsec=1.1,
    #                               flux_calibration_file=flux_calibration_file,
    #                               ADR=True,
    #                               plot_tracing_maps=[6250,7500,9000],
    #                               plot=True)    

    # # A way of solving this is providing the SIZE of the cube:
    # cube_test = Interpolated_cube(file_out, 
    #                               pixel_size_arcsec=0.7,
    #                               kernel_size_arcsec=1.1,
    #                               flux_calibration_file=flux_calibration_file,
    #                               ADR=True,
    #                               #plot_tracing_maps=[6250,7500,9000],
    #                               size_arcsec=[60,40],
    #                               step_tracing = 10,             # Increase the number of points for tracing
    #                               kernel_tracing = 19,           # Smooth tracing for removing outliers
    #                               g2d=False, 
    #                               adr_index_fit = 3,
    #                               trim_cube = True,             # Trimming the cube
    #                               plot=True)    


    # # For combining several cubes, the best way is running KOALA_reduce:
        
    # # We first check the 3 first files    

    # # list with RSS files (no need to include path)    
    # rss_list = ["27feb20031red_TCWXUS_NR.fits","27feb20032red_TCWXUS_NR.fits","27feb20033red_TCWXUS_NR.fits"]
    
    # # Once you have run KOALA_reduced and obtained ADRs, you can copy and paste values for speeding tests

    # ADR_x_fit_list =  [[-2.3540791265017944e-12, 6.762388793515361e-08, -0.0006627297804396334, 2.1666411712553812], [-3.0634647703827036e-12, 8.407943128848677e-08, -0.0007965956049746986, 2.545773557437055], [-8.043867327151777e-13, 2.8827477735272883e-08, -0.0003496319743184722, 1.34787667608832]]
    # ADR_y_fit_list =  [[1.0585167511142714e-12, -3.972561095302862e-08, 0.00047854197119031385, -1.806746479722604], [5.712676673528853e-13, -2.7118172653512425e-08, 0.00034731129018336117, -1.322842656375085], [-1.2201306546407504e-12, 2.3629083572706315e-08, -0.00013682268516072916, 0.21158456279065121]]

    # # Same thing with offsets
    
    # offsets =  [ -0.2773903893335756 , 0.7159980346947741 , 2.6672716919949306 , 0.8808234960793637 ]
    
    # combined_cube_test=KOALA_reduce(rss_list, path=path_red, 
    #                                 fits_file="combined_cube_test_.fits",
    #                                 rss_clean=True,                 # RSS files are clean
    #                                 pixel_size_arcsec=0.7, kernel_size_arcsec=1.1,
    #                                 flux_calibration_file=flux_calibration_file,
    #                                 #size_arcsec=[60,40],
    
    #                                 #ADR=True,
    #                                 #plot_tracing_maps=[6250,7500,9000],
    #                                 #box_x=[53,64], box_y=[22,32],
    #                                 #box_x = [10,70],
    #                                 #box_y = [10,60],
    #                                 half_size_for_centroid = 0,   # Using all data for registering
    #                                 step_tracing = 20,            # Increase the number of points for tracing
    #                                 kernel_tracing = 9,           # Smooth tracing for removing outliers
    #                                 g2d=False, 
    #                                 adr_index_fit = 3,

    #                                 ADR_x_fit_list = ADR_x_fit_list,
    #                                 ADR_y_fit_list = ADR_y_fit_list,
    #                                 offsets = offsets,
                                    
    #                                 trim_cube = True,             # Trimming the cube
    #                                 scale_cubes_using_integflux = False, # Not scaling cubes using integrated flux of common region
    #                                 plot= True, 
    #                                 plot_rss=False, 
    #                                 plot_weight=False,
    #                                 plot_spectra = False,
    #                                 fig_size=12,
    #                                 warnings=False, verbose = True)
  
    # # Plotting a larger map without plot_spaxel_grid, contours, or grid  
  
    # combined_cube_test.combined_cube.plot_map(log=True, cmap=fuego_color_map, fig_size=20, 
                                              # plot_centre=False, plot_spaxel_grid = False, 
                                              # contours= False, alpha_grid=0, fraction =0.027)
    

    # # Now we check the 3 last files    

    # # list with RSS files (no need to include path)    
    # rss_list = ["27feb20034red_TCWXUS_NR.fits","27feb20035red_TCWXUS_NR.fits","27feb20036red_TCWXUS_NR.fits"]
    
    # # For these, there is a star in the bottom left corner we can use for aligning / ADR
    # # g2d=True # it is a star
    # # box_x=[3,15], box_y=[3,15] # define the box
    
    # # Once you have run KOALA_reduced and obtained ADRs, you can copy and paste values for speeding tests

    # ADR_x_fit_list =  [[9.472735325782044e-13, -1.8414272142169123e-08, 0.00010639453474275714, -0.16011731471966129], [1.8706793461446896e-12, -4.151166766143467e-08, 0.00029190493266945084, -0.6408077338639665], [-1.3780071545760161e-12, 3.170991526784723e-08, -0.0002355233904377564, 0.5627653239513558]]
    # ADR_y_fit_list =  [[-2.0911079335920793e-12, 4.703018551076634e-08, -0.00033904804410673275, 0.7771617214421154], [-2.838537640353266e-12, 6.54908082091063e-08, -0.00048530769637374164, 1.148318135983978], [-3.1271687976010236e-12, 7.140942741912648e-08, -0.0005301049734410775, 1.2753542609892792]]    

    # # Same thing with offsets
    
    # offsets =  [ 1.4239020516303178 , 0.003232821107207684 , 1.4757701790022182 , 1.495021090623025 ]
    # offsets =  [1.5, 0, 1.5, 1.5]  # Given at the telescope
    
    # combined_cube_test=KOALA_reduce(rss_list, path=path_red, 
    #                                 fits_file="combined_cube_test_2.fits",
    #                                 rss_clean=True,                 # RSS files are clean
    #                                 pixel_size_arcsec=0.7, kernel_size_arcsec=1.1,
    #                                 flux_calibration_file=flux_calibration_file,
    #                                 ADR=True,
    #                                 plot_tracing_maps=[6400], #[6250,7500,9000],
    #                                 #size_arcsec=[60,40],
    #                                 box_x=[3,15], box_y=[3,15],
    #                                 half_size_for_centroid = 0,   # Using all data for registering
    #                                 step_tracing = 20,            # Increase the number of points for tracing
    #                                 kernel_tracing = 19,           # Smooth tracing for removing outliers
    #                                 g2d=True, 
    #                                 adr_index_fit = 3,
                                    
    #                                 #ADR_x_fit_list = ADR_x_fit_list,
    #                                 #ADR_y_fit_list = ADR_y_fit_list,
    #                                 #offsets = offsets,
                                    
    #                                 trim_cube = True,             # Trimming the cube
    #                                 scale_cubes_using_integflux = False, # Not scaling cubes using integrated flux of common region
    #                                 plot= True, 
    #                                 plot_rss=False, 
    #                                 plot_weight=False,
    #                                 plot_spectra = False,
    #                                 fig_size=12,
    #                                 warnings=False, verbose = True)
  


    # Now it is time to merge the 2 sets
    # Here I'm trusting that the offset between file 33 and 34 is that given at the telescope: 18S 3E (this should be checked)
    # and put everything together:
    
    # rss_list = ["27feb20031red_TCWXUS_NR.fits","27feb20032red_TCWXUS_NR.fits","27feb20033red_TCWXUS_NR.fits",
    #             "27feb20034red_TCWXUS_NR.fits","27feb20035red_TCWXUS_NR.fits","27feb20036red_TCWXUS_NR.fits"]

    # ADR_x_fit_list =  [[-2.3540791265017944e-12, 6.762388793515361e-08, -0.0006627297804396334, 2.1666411712553812], [-3.0634647703827036e-12, 8.407943128848677e-08, -0.0007965956049746986, 2.545773557437055], [-8.043867327151777e-13, 2.8827477735272883e-08, -0.0003496319743184722, 1.34787667608832],
    #                    [9.472735325782044e-13, -1.8414272142169123e-08, 0.00010639453474275714, -0.16011731471966129], [1.8706793461446896e-12, -4.151166766143467e-08, 0.00029190493266945084, -0.6408077338639665], [-1.3780071545760161e-12, 3.170991526784723e-08, -0.0002355233904377564, 0.5627653239513558]]
    # ADR_y_fit_list =  [[1.0585167511142714e-12, -3.972561095302862e-08, 0.00047854197119031385, -1.806746479722604], [5.712676673528853e-13, -2.7118172653512425e-08, 0.00034731129018336117, -1.322842656375085], [-1.2201306546407504e-12, 2.3629083572706315e-08, -0.00013682268516072916, 0.21158456279065121],
    #                    [-2.0911079335920793e-12, 4.703018551076634e-08, -0.00033904804410673275, 0.7771617214421154], [-2.838537640353266e-12, 6.54908082091063e-08, -0.00048530769637374164, 1.148318135983978], [-3.1271687976010236e-12, 7.140942741912648e-08, -0.0005301049734410775, 1.2753542609892792]] 

    # offsets =  [ -0.2773903893335756 , 0.7159980346947741 , 2.6672716919949306 , 0.8808234960793637,
    #               3, 18,   
    #             #0.5, 18.9, # I got these values with Jamila in Feb 21, but they don´t work 
    #             1.4239020516303178 , 0.003232821107207684 , 1.4757701790022182 , 1.495021090623025 ]
                
    # combined_cube=KOALA_reduce(rss_list, path=path_red, 
    #                            fits_file="combined_cube_.fits",
    #                            rss_clean=True,                 # RSS files are clean
    #                            pixel_size_arcsec=0.7, kernel_size_arcsec=1.1,
    #                            flux_calibration_file=flux_calibration_file,
    #                            #ADR=True,
    #                            plot_tracing_maps=[6400], #[6250,7500,9000],
    #                            size_arcsec=[80,75],
    #                            #box_x=[3,15], box_y=[3,15],
    #                            half_size_for_centroid = 0,   # Using all data for registering
    #                            step_tracing = 20,            # Increase the number of points for tracing
    #                            kernel_tracing = 19,           # Smooth tracing for removing outliers
    #                            g2d=False, 
    #                            adr_index_fit = 3,
                                    
    #                            ADR_x_fit_list = ADR_x_fit_list,
    #                            ADR_y_fit_list = ADR_y_fit_list,
    #                            offsets = offsets,
    #                            reference_rss = 0,
                                    
    #                            trim_cube = True,             # Trimming the cube
    #                            scale_cubes_using_integflux = False, # Not scaling cubes using integrated flux of common region
    #                            plot= True, 
    #                            plot_rss=False, 
    #                            plot_weight=False,
    #                            plot_spectra = False,
    #                            fig_size=12,
    #                            warnings=False, verbose = True)
  

    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    # # BLUE DATA 
    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------

    # # List the files in the folder
    # list_fits_files_in_folder(path_blue)
    
    # PyKOALA finds 4 objects: HD60753, HILT600 (calibration stars),
    #                          He2-10 (the galaxy),
    #                          SKYFLAT
    
    # # -----------------------------------------------------------------------

    # # Next, run this for AUTOMATICALLY processing calibration of the night
    # automatic_calibration_night(path=path_blue, auto=True) 
                                #, kernel_throughput = 21)
 
    
    # # This will create the 2 files needed for the calibration:
    
    # # 1. The throughput_2D_file:
    throughput_2D_file = path_blue+"throughput_2D_20180227_580V.fits"
    # # 2. The flux calibration file:
    flux_calibration_file = path_blue+"flux_calibration_20180227_580V_0p7_1k10.dat"

    # # It will also create 2 Python objects:
    # # HD60753_580V_20180227 : Python object with calibration star HD60753
    # # Hilt600_580V_20180227 : Python object with calibration star Hiltner 600 

    # As for the red, the calibration for HD 60753 is not as good as for Hilt600,
    # we can scale again the first cube using abs_flux_scale =[1.1, 1.0]:

    # automatic_calibration_night(path=path_blue, auto=True, 
    #                             pixel_size=0.7, kernel_size=1.1,
    #                             do_skyflat = False,
    #                             #list_of_objects =["HD60753_580V_20180227", "Hilt600_580V_20180227"], # if needed
    #                             abs_flux_scale =[1.1, 1.0], # add this for SCALING stars, if needed
    #                             cal_from_calibrated_starcubes = True)   # This assumes the cubes ARE created as Python objects
    
    
    # # Now, quickly check if [O III] 5007 and/or Hbeta are everywhere in the data
    
    # file_in   = path_blue+"27feb10031red.fits"

    # test = KOALA_RSS(file_in, 
    #                  save_rss_to_fits_file="auto",  
    #                  apply_throughput=True, 
    #                  throughput_2D_file=throughput_2D_file,       # if throughput_2D_file given, use SOL in fits file for fixing wave
    #                  correct_ccd_defects = True, 
    #                  fix_wavelengths = True, 
    #                  do_extinction=True)

    # # It doesn't look like there is emission everywhere, but a 1D plot can help
    
    # w = test.wavelength
    # f = test.intensity_corrected[0]   # Just a fibre
    # f = test.intensity_corrected[test.integrated_fibre_sorted[0]] # faintest fibre

    # test.find_sky_emission(n_sky=30, include_history = False)    # using auto self sky
    # f=test.sky_emission    
    
    # z = (6583 - 6563 )/6563.   # redshit of the galaxy using H-alpha
    
    # plot_plot(w,f, xmin=4700, xmax=5200, ymin=100, ymax=500, 
    #           vlines=[4861*(1+z), 5007*(1+z)], 
    #           ptitle="Checking emission lines H$\mathrm{\beta}$ and [O III] 5007$\mathrm{\AA}$")

    # # All good, we can process everything together using self for sky
    # # We already have the OFFSETS, these are the same we obtained for the RED
    # # We also use centre_deg and size_arcsec given by the TRIMMED red cube
    # # and select trim_cube = False
    
    # # But NOT the ADR correction for the blue files, as this is somehow a challenging object
    # # and we are doing mosaic we can't select a region that is good for all files
    # # let's try with half_size_for_centroid = 0
    
    # # Running the following should process the 6 RSS files, clean them, cube them, apply alignments and combine them
    # # producing a FINAL combined cube in fits format.
            
    # rss_list = ["27feb10031red.fits","27feb10032red.fits","27feb10033red.fits",
    #             "27feb10034red.fits","27feb10035red.fits","27feb10036red.fits"]
    
    # combined_cube_blue=KOALA_reduce(rss_list, path=path_blue, 
    #                                 fits_file="combined_cube_blue_test.fits",
                                
    #                                 # This is the rss part
    #                                 save_rss_to_fits_file_list="auto",  
    #                                 apply_throughput=True, 
    #                                 throughput_2D_file=throughput_2D_file,       # if throughput_2D_file given, use SOL in fits file for fixing wave
    #                                 correct_ccd_defects = True, 
    #                                 fix_wavelengths = True, 
    #                                 do_extinction=True,
    #                                 sky_method="self", n_sky = 30,
    #                                 remove_5577 = True,
    #                                 correct_negative_sky = True, 
    #                                 order_fit_negative_sky = 7, kernel_negative_sky=51, individual_check=False, 
    #                                 use_fit_for_negative_sky=False, force_sky_fibres_to_zero=True,
    #                                 remove_negative_median_values = False,
    #                                 fix_edges = False,  # This does not work well for the blue...
    #                                 clean_extreme_negatives=True, percentile_min=0.9,
    #                                 clean_cosmics=True,
    #                                 width_bl=0., kernel_median_cosmics=5, cosmic_higher_than = 100., extra_factor =	2.,
    #                                 max_number_of_cosmics_per_fibre = 12,
    #                                 brightest_line = "O3b",
    #                                 brightest_line_wavelength = 5022,
                                
    #                                 rss_clean=False,                 # RSS files are clean
                                    
    #                                 # This is the cubing part                                    
    #                                 pixel_size_arcsec=0.7, kernel_size_arcsec=1.1,
    #                                 flux_calibration_file=flux_calibration_file,
    #                                 ADR=True,
    #                                 #ADR_x_fit_list = ADR_x_fit_list,
    #                                 #ADR_y_fit_list = ADR_y_fit_list,
    #                                 plot_tracing_maps=[4500], 
    #                                 #box_x=[3,15], box_y=[3,15],
    #                                 half_size_for_centroid = 0,   # Using all data for registering
    #                                 step_tracing = 20,            # Increase the number of points for tracing
    #                                 kernel_tracing = 19,           # Smooth tracing for removing outliers
    #                                 g2d=False, 
    #                                 adr_index_fit = 3,
                                    
    #                                 # This is the part for alignment
    #                                 offsets = offsets,
    #                                 reference_rss = 0,
    #                                 centre_deg = [combined_cube.combined_cube.RA_centre_deg, combined_cube.combined_cube.DEC_centre_deg],
    #                                 size_arcsec= [combined_cube.combined_cube.RA_segment,combined_cube.combined_cube.DEC_segment],
                                        
    #                                 trim_cube = False,             # Trimming the cube
    #                                 scale_cubes_using_integflux = False, # Not scaling cubes using integrated flux of common region
    #                                 plot= True, 
    #                                 plot_rss=True, 
    #                                 plot_weight=False,
    #                                 plot_spectra = False,
    #                                 fig_size=12,
    #                                 warnings=False, verbose = True)

    # # This created combined cube /DATA/KOALA/Python/GitHub/test_reduce/580V/combined_cube_blue_test.fits
    # # Plus all the clean RSS files         

    # rss_list = ["27feb10031red_TCWX_S_NR.fits","27feb10032red_TCWX_S_NR.fits","27feb10033red_TCWX_S_NR.fits",
    #             "27feb10034red_TCWX_S_NR.fits","27feb10035red_TCWX_S_NR.fits","27feb10036red_TCWX_S_NR.fits"]

    # # We also obtained the ADR correction
    
    # ADR_x_fit_list =  [[9.791374902300966e-11, -1.361731424563216e-06, 0.006168569717114751, -9.062099268442289], [1.5178720729296336e-10, -2.149538359423654e-06, 0.010002596378191356, -15.27549983827299], [-1.5727880041293275e-10, 2.270172789723589e-06, -0.010746153798036104, 16.6669228618847], [-3.587184899956129e-11, 8.061339103441051e-07, -0.005438900897366207, 11.469478296922391], [-1.0888626198857115e-10, 1.833820655023091e-06, -0.010112882237282317, 18.262553011417417], [-2.0507071024794933e-10, 3.19853352438298e-06, -0.016448278016901505, 27.84319868752676]]
    # ADR_y_fit_list =  [[-5.958768890277437e-11, 8.335385578402792e-07, -0.003814158515275198, 5.685826415246691], [9.440481199724225e-11, -1.371028712714804e-06, 0.006603018298665971, -10.547255178451204], [1.5921988914264668e-10, -2.3121137328567566e-06, 0.011046520521171891, -17.337559244322566], [-9.532034466031645e-11, 1.3410760221244944e-06, -0.0062161978911407035, 9.489044815580575], [-3.289714426078901e-11, 4.280544076996474e-07, -0.0018416700437986987, 2.6479564442375496], [1.812993107807248e-11, -2.2424083469823508e-07, 0.000833064491704837, -0.8411422420774155]]


    # # If we want to iterate a bit more, we can use the RSS list with the CLEAN RSS files
    # # and select rss_clean=True
    
    # combined_cube_blue=KOALA_reduce(rss_list, path=path_blue, 
    #                                 fits_file="combined_cube_blue_test.fits",
                                
    #                                 # This is the rss part
    #                                 rss_clean=True,                 # RSS files are clean
                                    
    #                                 # This is the cubing part                                    
    #                                 pixel_size_arcsec=0.7, kernel_size_arcsec=1.1,
    #                                 flux_calibration_file=flux_calibration_file,
    #                                 #ADR=True,
    #                                 ADR_x_fit_list = ADR_x_fit_list,
    #                                 ADR_y_fit_list = ADR_y_fit_list,
    #                                 plot_tracing_maps=[4500], 
    #                                 #box_x=[3,15], box_y=[3,15],
    #                                 half_size_for_centroid = 0,   # Using all data for registering
    #                                 step_tracing = 20,            # Increase the number of points for tracing
    #                                 kernel_tracing = 19,          # Smooth tracing for removing outliers
    #                                 g2d=False, 
    #                                 adr_index_fit = 3,
                                        
    #                                 # This is the part for alignment
    #                                 offsets = offsets,
    #                                 reference_rss = 0,
    #                                 centre_deg = [combined_cube.combined_cube.RA_centre_deg, combined_cube.combined_cube.DEC_centre_deg],
    #                                 size_arcsec= [combined_cube.combined_cube.RA_segment,combined_cube.combined_cube.DEC_segment],
                                        
    #                                 trim_cube = False,             # Trimming the cube
    #                                 scale_cubes_using_integflux = False, # Not scaling cubes using integrated flux of common region
    #                                 plot= True, 
    #                                 plot_rss=True, 
    #                                 plot_weight=False,
    #                                 plot_spectra = False,
    #                                 fig_size=12,
    #                                 warnings=False, verbose = True)
         
    
    # # Time to check alignment between red and blue cubes!
    
    # red_cube = path_red + "combined_cube_He2-10.fits"
    # blue_cube = path_blue + "combined_cube_blue_test.fits"
    
    # align_blue_and_red_cubes(combined_cube_blue.combined_cube, combined_cube.combined_cube)
    # align_blue_and_red_cubes(blue_cube, red_cube, 
    #                          half_size_for_centroid = 12,                                  # These parameters are needed for getting
    #                          step_tracing = 20, kernel_tracing = 19, adr_index_fit = 3)    # the centroids and check alignment of cubes

    # # The red and the blue cubes are basically aligned, a very small offset between them:
    # #   > The offsets between the two cubes following tracing the peak are:
    # #   -> delta_RA  (blue -> red) = 0.076  spaxels         = 0.053 arcsec
    # #   -> delta_DEC (blue -> red) = 0.094  spaxels         = 0.066 arcsec
    # #   TOTAL     (blue -> red) = 0.121  spaxels         = 0.084 arcsec      (12.1% of the pix size)     



    # # With this, we have processed and aligned both the blue and red cubes of the galaxy He 2-10 !!


    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------
    # # -----------------------------------------------------------------------

    # Truco de Pablo para cambiar algo en header:
    # with fits.open(path_blue+"27feb10031red.fits", "update") as f:
    #     f[0].header["OBJECT"]="He2-10"
    
    # with fits.open(blue_cube, "update") as f:
    #     f[0].header["COMCUBE"]="T"
    
end= timer()
print("\n> Elapsing time = ",end-start, "s")        
