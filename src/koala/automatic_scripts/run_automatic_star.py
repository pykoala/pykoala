#!/usr/bin/python
# -*- coding: utf-8 -*-

from koala.constants import red_gratings, blue_gratings, fibres_best_sky_100
from koala.io import full_path, read_table, spectrum_to_text_file
#from koala.KOALA_RSS import KOALA_RSS
from koala.RSS import RSS
from koala.automatic_scripts.koala_reduce import KOALA_reduce
from koala.cube import read_cube, telluric_correction_from_star

import numpy as np
import copy

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_calibration_star_data(star, path_star, grating, pk):
 
    
    description = star
    fits_file = path_star+star+"_"+grating+pk+".fits"
    response_file = path_star+star+"_"+grating+pk+"_response.dat" 
    telluric_file = path_star+star+"_"+grating+pk+"_telluric_correction.dat" 
    
    if grating in blue_gratings : CONFIG_FILE="./configuration_files/STARS/calibration_star_blue.config"
    if grating in red_gratings : 
        CONFIG_FILE="./configuration_files/STARS/calibration_star_red.config"
        list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140],     # DEFAULT VALUES
                                     [7080,7140,7500,7580], [7400,7580,7705,7850],
                                     [7850,8090,8450,8700] ]
    else:
        list_of_telluric_ranges = [[]]
    
    calibration_stars_folder = "input_data/spectrophotometric_stars_data/" # "FLUX_CAL/"
    
    if star in ["cd32d9927", "CD32d9927", "CD32D9927", "cd32d9927auto", "CD32d9927auto", "CD32D9927auto"] : 
        absolute_flux_file = calibration_stars_folder+ 'fcd32d9927_edited.dat'  
        #list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140], 
        #                             [7080,7140,7500,7580], [7400,7580,7705,7850],
        #                             [7850,8090,8450,8700] ]
        # If needed, include here particular CONFIG FILES:
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_CD32d9927_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_CD32d9927_blue.config"
    if star in ["HD49798" , "hd49798" , "HD49798auto" , "hd49798auto"] : 
        absolute_flux_file = calibration_stars_folder + 'fhd49798.dat'  
        #list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140], 
        #                             [7080,7140,7500,7580], [7400,7580,7705,7850],
        #                             [7850,8090,8450,8700] ]           
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HD49798_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HD49798_blue.config"
    if star in ["HD60753", "hd60753" , "HD60753auto" ,"hd60753auto", "HD60753FLUX", "hd60753FLUX" , "HD60753FLUXauto" ,"hd60753FLUXauto" ] : 
        absolute_flux_file = calibration_stars_folder + 'fhd60753.dat'  
        # list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140], 
        #                             [7080,7140,7500,7580], [7400,7580,7705,7850],
        #                             [7850,8090,8450,8700] ]    
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HD60753_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HD60753_blue.config"    
    if star in [ "H600", "Hiltner600" , "Hilt600" ,"H600auto"] : 
        absolute_flux_file = calibration_stars_folder+'fhilt600_edited.dat'  
        # list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140], 
        #                            [7080,7140,7500,7580], [7400,7580,7705,7850],
        #                            [7850,8090,8450,8700] ] 
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_Hilt600_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/ccalibration_Hilt600_blue.config"
                          
    if star in [ "EG274" , "E274" , "eg274", "e274", "EG274auto", "E274auto" , "eg274auto", "e274auto" ] :
        absolute_flux_file = calibration_stars_folder +'feg274_edited.dat'
        list_of_telluric_ranges =  [ [6150,6245,6380,6430], [6720,6855,7080,7150], 
                                     [7080,7140,7500,7580], [7400,7580,7705,7850], 
                                     [7850,8090,8450,8700] ] 
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_EG274_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_EG274_blue.config"               
    if star in ["EG21", "eg21" , "Eg21", "EG21auto", "eg21auto" , "Eg21auto"]  : 
        absolute_flux_file = calibration_stars_folder+ 'feg21_edited.dat'
        list_of_telluric_ranges =  [ [6150,6245,6380,6430], [6720,6855,7080,7150], 
                                     [7080,7140,7500,7580], [7400,7580,7705,7850], 
                                     [7850,8090,8450,8700] ]
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_EG21_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_EG21_blue.config"
    if star in [ "HR3454" ,"Hr3454" , "hr3454", "HR3454auto" ,"Hr3454auto" , "hr3454auto" ]  : 
        absolute_flux_file = calibration_stars_folder+ 'fhr3454_edited.dat'
        list_of_telluric_ranges =  [ [6150,6245,6380,6430], [6720,6855,7080,7150], 
                                     [7080,7140,7500,7580], [7400,7580,7705,7850], 
                                     [7850,8090,8450,8700] ]
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HR3454_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HR3454_blue.config"
    if star in [ "HR718" ,"Hr718" , "hr718", "HR718FLUX","HR718auto" ,"Hr718auto" , "hr718auto", "HR718FLUXauto"  ]  : 
        absolute_flux_file = calibration_stars_folder + 'fhr718_edited.dat'
        #list_of_telluric_ranges =  [ [6150,6245,6380,6430], [6720,6855,7080,7150], 
        #                             [7080,7140,7500,7580], [7400,7580,7705,7850], 
        #                             [7850,8090,8450,8700] ]
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HR718_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HR718_blue.config"


    return CONFIG_FILE, description, fits_file, response_file, absolute_flux_file, telluric_file, list_of_telluric_ranges

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def run_automatic_star(CONFIG_FILE="",
                       instrument="KOALA",
                       star="",
                       description ="",
                       obj_name ="",
                       object_auto = "",
                       date="", grating="", 
                       pixel_size  = 0.7, 
                       kernel_size = 1.1, 
                       path_star ="",
                       rss_list=[],
                       path="",
                       reduced = False,
                       read_fits_cube = False,
                       fits_file="",
                       rss_clean = False,
                       save_rss = False,
                       save_rss_to_fits_file_list=[],
                       
                       apply_throughput = False,
                       throughput_2D_variable = "",
                       throughput_2D=[], throughput_2D_file = "",
                       throughput_2D_wavecor = False,
                       valid_wave_min = 0, valid_wave_max = 0,
                       correct_ccd_defects = False,
                       fix_wavelengths = False, sol =[0,0,0],
                       do_extinction = False,
                       sky_method ="none",
                       n_sky = 100,
                       sky_fibres =[], 
                       win_sky = 0,
                       remove_5577=False,
                       correct_negative_sky = False,     
                       order_fit_negative_sky =3, 
                       kernel_negative_sky = 51,
                       individual_check = True,
                       use_fit_for_negative_sky = False,
                       force_sky_fibres_to_zero = True,
                       low_fibres=10,
                       high_fibres=20,
                       
                       remove_negative_median_values = False ,   
                       fix_edges=False,
                       clean_extreme_negatives = False,
                       percentile_min=0.9  ,
                       clean_cosmics = False,
                       width_bl = 20.,
                       kernel_median_cosmics = 5 ,
                       cosmic_higher_than 	=	100. ,
                       extra_factor =	1.,
                                                                  
                       do_cubing = True, do_alignment=True, make_combined_cube=True,
                       edgelow = -1, edgehigh =-1,
                       ADR=False, jump = -1,
                       adr_index_fit=2, 
                       g2d = True,
                       step_tracing=100, 
                       kernel_tracing=0,
                                   
                       box_x=[0,-1], box_y=[0,-1],
                       trim_cube = True, trim_values =[],
                       scale_cubes_using_integflux = False,
                       flux_ratios =[],

                       telluric_file = "",
                       order_telluric = 2,
                       list_of_telluric_ranges = [[]],                                                
                       apply_tc=True, 
                       
                       do_calibration = True,
                       absolute_flux_file ="",
                       response_file="",
                       size_arcsec = [],
                       r_max=5.,
                       step_flux=10.,
                       ha_width = 0, exp_time = 0.,
                       min_wave_flux = 0, max_wave_flux = 0,
                       sky_annulus_low_arcsec = 5.,
                       sky_annulus_high_arcsec = 10.,
                       exclude_wlm=[[0,0]],
                       odd_number=0,
                       smooth =0.,
                       fit_weight=0., 
                       smooth_weight=0.,
                       
                       log = True, gamma = 0,   
                       fig_size = 12,                                     
                       plot = True, plot_rss = True, 
                       plot_weight=False, plot_spectra=False,
                       warnings = True, verbose = True  ): 
    """
    Use: 
        CONFIG_FILE_H600 = "./configuration_files/calibration_star1.config"
        H600auto=run_automatic_star(CONFIG_FILE_H600)
    """
    if plot == False:
        plot_rss = False 
        plot_weight=False
        plot_spectra=False
        

    global star_object   
    sky_fibres_print ="" 
    if object_auto == "" : print("\n> Running automatic script for processing a calibration star")
    
    rss_clean_given = rss_clean
    
# # Setting default values (now the majority as part of definition)
    
    pk = "_"+str(int(pixel_size))+"p"+str(int((abs(pixel_size)-abs(int(pixel_size)))*10))+"_"+str(int(kernel_size))+"k"+str(int(abs(kernel_size*100))-int(kernel_size)*100)   
    first_telluric_range = True 

# # if path is given, add it to the rss_list if needed
    if path_star == "" and path != "" : path_star=path
    if path == "" and path_star != "" : path=path_star
    if path != "" and len(rss_list) > 0:
        for i in range(len(rss_list)):
            rss_list[i]=full_path(rss_list[i],path)

# # If grating is not given, we can check reading a RSS file  
    if grating == "" :
        print("\n> No grating provided! Checking... ")
        _test_ = RSS(rss_list[0], instrument=instrument, verbose = False)
        #_test_ = KOALA_RSS(rss_list[0], plot_final_rss=False, verbose = False)
        grating = _test_.grating
        print("\n> Reading file",rss_list[0], "the grating is",grating)

    if CONFIG_FILE == "":    
        # If no configuration file is given, check if name of the star provided
        if star == "":
            print("  - No name for calibration star given, asuming name = star")
            star="star"
                    
        CONFIG_FILE, description_, fits_file_, response_file_, absolute_flux_file_, telluric_file_, list_of_telluric_ranges_ =  get_calibration_star_data (star, path_star, grating, pk)

        if description == "" : description = description_
        if fits_file == "" : fits_file = fits_file_
        if response_file == "" : response_file = response_file_
        if absolute_flux_file == "" : absolute_flux_file = absolute_flux_file_
        if telluric_file == "" : telluric_file = telluric_file_
        if len(list_of_telluric_ranges[0]) == 0 : list_of_telluric_ranges = list_of_telluric_ranges_
                
        # Check if folder has throughput if not given
        if throughput_2D_file == "":
            print("\n> No throughout file provided, using default file:")
            throughput_2D_file = path_star+"throughput_2D_"+date+"_"+grating+".fits" 
            print("  ",throughput_2D_file)
        

# # Read configuration file       
    config_property, config_value = read_table(CONFIG_FILE, ["s", "s"] )
        
    if object_auto == "" : 
        print("\n> Reading configuration file", CONFIG_FILE,"...\n")
        if obj_name =="":
            object_auto = star+"_"+grating+"_"+date
        else:
            object_auto = obj_name
       
    for i in range(len(config_property)):
        
        if  config_property[i] == "pixel_size" : 	 pixel_size = float(config_value[i]) 
        if  config_property[i] == "kernel_size" : 	 
            kernel_size = float(config_value[i])
            pk = "_"+str(int(pixel_size))+"p"+str(int((abs(pixel_size)-abs(int(pixel_size)))*10))+"_"+str(int(kernel_size))+"k"+str(int(abs(kernel_size*100))-int(kernel_size)*100)
        if  config_property[i] == "date" : 	 date = config_value[i]
        if  config_property[i] == "grating" : 	 grating = config_value[i]
        
        if  config_property[i] == "path_star" :  
            path_star = config_value[i] 
            if path_star[-1] != "/" : path_star =path_star +"/"
        if  config_property[i] == "obj_name" :  object_auto = config_value[i]        
        if  config_property[i] == "star" : 
            star = config_value[i]
            _CONFIG_FILE_, description, fits_file, response_file, absolute_flux_file, list_of_telluric_ranges =  get_calibration_star_data (star, path_star, grating, pk)

        if  config_property[i] == "description" :  description = config_value[i]
        if  config_property[i] == "fits_file"   :  fits_file = full_path(config_value[i],path_star)
        if  config_property[i] == "response_file" :  response_file = full_path(config_value[i],path_star)
        if  config_property[i] == "telluric_file" :  telluric_file = full_path(config_value[i],path_star)
        
        if  config_property[i] == "rss" : rss_list.append(full_path(config_value[i],path_star))  #list_of_files_of_stars
        if  config_property[i] == "reduced" :
            if config_value[i] == "True" :  reduced = True 
        
        if  config_property[i] == "read_cube" :
            if config_value[i] == "True" : read_fits_cube = True 
                    
        # RSS Section -----------------------------

        if  config_property[i] == "rss_clean" :
            if config_value[i] == "True" : 
                rss_clean = True 
            else: rss_clean = False 
            
            if rss_clean_given  == True: rss_clean = True

        if  config_property[i] == "save_rss" :
            if config_value[i] == "True" :   save_rss = True
        
        if  config_property[i] == "apply_throughput" :
            if config_value[i] == "True" : 
                apply_throughput = True 
            else: apply_throughput = False                                                 
        if  config_property[i] == "throughput_2D_file" : throughput_2D_file = full_path(config_value[i],path_star)
        if  config_property[i] == "throughput_2D" : throughput_2D_variable =config_value[i] # full_path(config_value[i],path_star)
        
        if  config_property[i] == "correct_ccd_defects" :
            if config_value[i] == "True" : 
                correct_ccd_defects = True 
            else: correct_ccd_defects = False 
        if  config_property[i] == "fix_wavelengths":
            if config_value[i] == "True" : fix_wavelengths = True 
        if  config_property[i] == "sol" :
            sol_ = config_value[i].strip('][').split(',')
            if float(sol_[0]) == -1:
                sol = [float(sol_[0])]
            else:
                if float(sol_[0]) != -0: sol = [float(sol_[0]),float(sol_[1]),float(sol_[2])]

        if  config_property[i] == "throughput_2D_wavecor" :
            if config_value[i] == "True" : 
                throughput_2D_wavecor = True 
            else: throughput_2D_wavecor = False      

        if  config_property[i] == "do_extinction":
            if config_value[i] == "True" : 
                do_extinction = True 
            else: do_extinction = False 
            
        if  config_property[i] == "sky_method" : sky_method = config_value[i]
        if  config_property[i] == "n_sky" : n_sky=int(config_value[i])
        if  config_property[i] == "win_sky" : win_sky =  int(config_value[i])
        if  config_property[i] == "remove_5577" : 
            if config_value[i] == "True" : remove_5577 = True 
        if  config_property[i] == "correct_negative_sky" :
            if config_value[i] == "True" : correct_negative_sky = True 

        if  config_property[i] == "order_fit_negative_sky" : order_fit_negative_sky =  int(config_value[i])
        if  config_property[i] == "kernel_negative_sky" : kernel_negative_sky =  int(config_value[i])            
        if  config_property[i] == "individual_check" :
            if config_value[i] == "True" : 
                individual_check = True 
            else: individual_check = False 
        if  config_property[i] == "use_fit_for_negative_sky" :
            if config_value[i] == "True" : 
                use_fit_for_negative_sky = True 
            else: use_fit_for_negative_sky = False  

        if  config_property[i] == "force_sky_fibres_to_zero" :
            if config_value[i] == "True" : 
                force_sky_fibres_to_zero = True 
            else: force_sky_fibres_to_zero = False 
        if  config_property[i] == "high_fibres" : high_fibres =  int(config_value[i])
        if  config_property[i] == "low_fibres" : low_fibres =  int(config_value[i])

        if config_property[i] == "sky_fibres" :  
            sky_fibres_ =  config_value[i]
            if sky_fibres_ == "fibres_best_sky_100":
                sky_fibres = fibres_best_sky_100
                sky_fibres_print =  "fibres_best_sky_100"
            else:
                if sky_fibres_[0:5] == "range":
                    sky_fibres_ = sky_fibres_[6:-1].split(',')
                    sky_fibres = list(range(np.int(sky_fibres_[0]),np.int(sky_fibres_[1])))
                    sky_fibres_print = "range("+sky_fibres_[0]+","+sky_fibres_[1]+")"
                else:
                    sky_fibres_ = config_value[i].strip('][').split(',')
                    for i in range(len(sky_fibres_)):
                        sky_fibres.append(float(sky_fibres_[i]))                    
                    sky_fibres_print =  sky_fibres  

        if  config_property[i] == "remove_negative_median_values":
            if config_value[i] == "True" : 
                remove_negative_median_values = True 
            else: remove_negative_median_values = False 

        if  config_property[i] == "fix_edges" and  config_value[i] == "True" : fix_edges = True 
        if  config_property[i] == "clean_extreme_negatives" :
            if config_value[i] == "True" : clean_extreme_negatives = True 
        if  config_property[i] == "percentile_min" : percentile_min = float(config_value[i])  

        if  config_property[i] == "clean_cosmics" and config_value[i] == "True" : clean_cosmics = True 
        if  config_property[i] == "width_bl" : width_bl = float(config_value[i])  
        if  config_property[i] == "kernel_median_cosmics" : kernel_median_cosmics = int(config_value[i])  
        if  config_property[i] == "cosmic_higher_than" : cosmic_higher_than = float(config_value[i])  
        if  config_property[i] == "extra_factor" : extra_factor = float(config_value[i])  
          
        # Cubing Section ------------------------------

        if  config_property[i] == "do_cubing" : 
            if config_value[i] == "False" :  
                do_cubing = False 
                do_alignment = False
                make_combined_cube = False  # LOki
                
        if  config_property[i] == "size_arcsec" :     
            size_arcsec_ = config_value[i].strip('][').split(',')
            for i in range(len(size_arcsec_)):
                size_arcsec.append(float(size_arcsec_[i]))         
        
        if  config_property[i] == "edgelow" : edgelow =  int(config_value[i])    
        if  config_property[i] == "edgehigh" : edgehigh =  int(config_value[i]) 

        if  config_property[i] == "ADR" and config_value[i] == "True" : ADR = True 
        if  config_property[i] == "adr_index_fit": adr_index_fit = int(config_value[i])
        if  config_property[i] == "g2d": 
            if config_value[i] == "True" : 
                g2d = True
            else: g2d = False
            
        if  config_property[i] == "kernel_tracing": kernel_tracing = int(config_value[i])

        if  config_property[i] == "jump": jump = int(config_value[i])
        
        if  config_property[i] == "trim_cube" : 
            if config_value[i] == "True" : 
                trim_cube = True 
            else: trim_cube = False 
            
        if  config_property[i] == "trim_values" :     
            trim_values_ = config_value[i].strip('][').split(',')
            trim_values = [int(trim_values_[0]),int(trim_values_[1]),int(trim_values_[2]),int(trim_values_[3])]           
            
        if  config_property[i] == "scale_cubes_using_integflux" : 
            if config_value[i] == "True" : 
                scale_cubes_using_integflux = True 
            else: scale_cubes_using_integflux = False 
 
        if  config_property[i] == "flux_ratios" :
            flux_ratios_ = config_value[i].strip('][').split(',')
            flux_ratios = []
            for i in range(len(flux_ratios_)):
                flux_ratios.append(float(flux_ratios_[i]))

        # Calibration  ---------------------------------

        if  config_property[i] == "do_calibration" : 
            if config_value[i] == "False" :  do_calibration = False 
            if config_value[i] == "True" :  do_calibration = True 

        if  config_property[i] == "r_max" : r_max = float(config_value[i])

        # CHECK HOW TO OBTAIN TELLURIC CORRECTION !!! 
        if  config_property[i] == "order_telluric" : order_telluric =  int(config_value[i])
        if  config_property[i] == "telluric_range" :           
            if first_telluric_range == True : 
                list_of_telluric_ranges =[]   
                first_telluric_range   = False  
            telluric_range_string = config_value[i].strip('][').split(',')
            telluric_range_float = [float(telluric_range_string[0]),float(telluric_range_string[1]),float(telluric_range_string[2]),float(telluric_range_string[3])]
            list_of_telluric_ranges.append(telluric_range_float)     
                                  
        if  config_property[i] == "apply_tc"  :	
            if config_value[i] == "True" : 
                apply_tc = True 
            else: apply_tc = False

        if  config_property[i] == "absolute_flux_file" : absolute_flux_file = config_value[i]
        if  config_property[i] == "min_wave_flux" : min_wave_flux = float(config_value[i])   
        if  config_property[i] == "max_wave_flux" : max_wave_flux = float(config_value[i])   
        if  config_property[i] == "step_flux" : step_flux = float(config_value[i])   
        if  config_property[i] == "exp_time" : exp_time = float(config_value[i])   
        if  config_property[i] == "fit_degree_flux" : fit_degree_flux = int(config_value[i])
        if  config_property[i] == "ha_width" : ha_width = float(config_value[i])  
                
        if  config_property[i] == "sky_annulus_low_arcsec" : sky_annulus_low_arcsec = float(config_value[i]) 
        if  config_property[i] == "sky_annulus_high_arcsec" : sky_annulus_high_arcsec = float(config_value[i]) 

        if  config_property[i] == "valid_wave_min" : valid_wave_min = float(config_value[i])
        if  config_property[i] == "valid_wave_max" : valid_wave_max = float(config_value[i])
        
        if  config_property[i] == "odd_number" : odd_number = int(config_value[i])
        if  config_property[i] == "smooth" : smooth = float(config_value[i])
        if  config_property[i] == "fit_weight" : fit_weight = float(config_value[i])
        if  config_property[i] == "smooth_weight" : smooth_weight = float(config_value[i])
        
        if  config_property[i] == "exclude_wlm" :       
            exclude_wlm=[]
            exclude_wlm_string_= config_value[i].replace("]","")
            exclude_wlm_string= exclude_wlm_string_.replace("[","").split(',')
            for i in np.arange(0, len(exclude_wlm_string),2) :
                exclude_wlm.append([float(exclude_wlm_string[i]),float(exclude_wlm_string[i+1])])    

        # Plotting, printing ------------------------------
        
        if  config_property[i] == "log" :  
            if config_value[i] == "True" : 
                log = True 
            else: log = False 
        if  config_property[i] == "gamma" : gamma = float(config_value[i])            
        if  config_property[i] == "fig_size" : fig_size = float(config_value[i])
        if  config_property[i] == "plot" : 
            if config_value[i] == "True" : 
                plot = True 
            else: 
                plot = False 
                plot_rss = False 
                plot_weight=False
                plot_spectra=False
                
        if  config_property[i] == "plot_rss" : 
            if config_value[i] == "True" : 
                plot_rss = True 
            else: plot_rss = False 
        if  config_property[i] == "plot_weight" : 
            if config_value[i] == "True" : 
                plot_weight = True 
            else: plot_weight = False             
        if  config_property[i] == "plot_spectra" : 
            if config_value[i] == "True" : 
                plot_spectra = True 
            else: plot_spectra = False                            
        if  config_property[i] == "warnings" : 
            if config_value[i] == "True" : 
                warnings = True 
            else: warnings = False     
        if  config_property[i] == "verbose" : 
            if config_value[i] == "True" : 
                verbose = True 
            else: verbose = False   

    if throughput_2D_variable != "":  throughput_2D = eval(throughput_2D_variable)

    if do_cubing == False:      
        fits_file = "" 
        make_combined_cube = False  
        do_alignment = False

# # Print the summary of parameters

    print("> Parameters for processing this calibration star :\n")
    print("  star                     = ",star) 
    if object_auto != "" : 
        if reduced == True and read_fits_cube == False :
            print("  Python object            = ",object_auto,"  already created !!")   
        else:
            print("  Python object            = ",object_auto,"  to be created")
    print("  path                     = ",path_star)
    print("  description              = ",description)
    print("  date                     = ",date)
    print("  grating                  = ",grating)
    
    if reduced == False and read_fits_cube == False :  
        for rss in range(len(rss_list)):
            if rss == 0 : 
                if len(rss_list) > 1:
                    print("  rss_list                 = [",rss_list[rss],",")
                else:
                    print("  rss_list                 = [",rss_list[rss],"]")
            else:
                if rss == len(rss_list)-1:
                    print("                              ",rss_list[rss]," ]")
                else:        
                    print("                              ",rss_list[rss],",")     

        if rss_clean:
            print("  rss_clean                =  True, skipping to cubing\n")
        else:  
            if save_rss : print("  'CLEANED' RSS files will be saved automatically")

            if apply_throughput:
                if throughput_2D_variable != "" :    
                    print("  throughput_2D variable   = ",throughput_2D_variable)
                else:
                    if throughput_2D_file != "" : print("  throughput_2D_file       = ",throughput_2D_file)

            if apply_throughput and throughput_2D_wavecor:
                 print("  throughput_2D will be applied AFTER correcting CCD defects and fixing small wavelenghts")

            print("  correct_ccd_defects      = ",correct_ccd_defects)
            print("  fix_wavelengths          = ",fix_wavelengths)
            if fix_wavelengths: 
                if sol[0] == -1:
                    print("    Only using few skylines in the edges")
                else:
                    if sol[0] != -1: print("    sol                    = ",sol)
    
            print("  do_extinction            = ",do_extinction)           
            print("  sky_method               = ",sky_method)     
            if sky_method != "none" :              
                if len(sky_fibres) > 1: 
                    print("    sky_fibres             = ",sky_fibres_print)
                else:
                    print("    n_sky                  = ",n_sky)    
            if win_sky > 0 : print("    win_sky                = ",win_sky)  
            if remove_5577: print("    remove 5577 skyline    = ",remove_5577)
            print("  correct_negative_sky     = ",correct_negative_sky)
            if correct_negative_sky:
                print("    order_fit_negative_sky   = ",order_fit_negative_sky)
                print("    kernel_negative_sky      = ",kernel_negative_sky)
                print("    use_fit_for_negative_sky = ",use_fit_for_negative_sky) 
                print("    low_fibres               = ",low_fibres)
                print("    individual_check         = ",individual_check)  
                if sky_method in ["self" , "selffit"]:  print("    force_sky_fibres_to_zero = ",force_sky_fibres_to_zero)

            if fix_edges: print("  fix_edges                = ",fix_edges)          
 
            print("  clean_cosmics            = ",clean_cosmics)
            if clean_cosmics:
                print("    width_bl               = ",width_bl)
                print("    kernel_median_cosmics  = ",kernel_median_cosmics)
                print("    cosmic_higher_than     = ",cosmic_higher_than)
                print("    extra_factor           = ",extra_factor)
 
            print("  clean_extreme_negatives  = ",clean_extreme_negatives)
            if clean_extreme_negatives:
                print("    percentile_min         = ",percentile_min)    
            if remove_negative_median_values:
                print("  Negative pixels will be set to 0 when median value of spectrum is negative")
                
        if valid_wave_min > 0 : print("  valid_wave_min           = ",valid_wave_min,"A")
        if valid_wave_max > 0 : print("  valid_wave_max           = ",valid_wave_max,"A")

        if do_cubing:
            if len(size_arcsec) > 0: print("  cube_size_arcsec         = ",size_arcsec)    
    
            if edgelow != -1:  print("  edgelow for tracing      = ",edgelow,"pixels")
            if edgehigh != -1: print("  edgehigh for tracing     = ",edgehigh,"pixels")
            print("  2D Gauss for tracing     = ",g2d)
            if kernel_tracing > 0 : print("  kernel_tracing           = ",kernel_tracing)
            
            print("  ADR                      = ",ADR)       
            if ADR: print("    adr_index_fit          = ",adr_index_fit)
    
            if jump != -1 :    print("    jump for ADR           = ",jump)
    
            if scale_cubes_using_integflux:
                if len(flux_ratios) == 0 :
                    print("  Scaling individual cubes using integrated flux of common region")
                else:
                    print("  Scaling individual cubes using flux_ratios = ",flux_ratios)
    
            if trim_cube: print("  Trim cube                = ",trim_cube)
            if len(trim_values) != 0: print("    Trim values            = ",trim_values)
        else:      
            print("\n> No cubing will be performed\n")
    
    if do_calibration:    
    
        if read_fits_cube:   
            print("\n> Input fits file with cube:\n ",fits_file,"\n")
        else:      
            print("  pixel_size               = ",pixel_size)
            print("  kernel_size              = ",kernel_size)    
        print("  plot                     = ",plot)
        print("  verbose                  = ",verbose)
    
        print("  warnings                 = ",warnings)
        print("  r_max                    = ",r_max, '" for extracting the star')
        if grating in red_gratings:
            #print "  telluric_file        = ",telluric_file
            print("  Parameters for obtaining the telluric correction:")
            print("    apply_tc               = ", apply_tc)
            print("    order continuum fit    = ",order_telluric)
            print("    telluric ranges        = ",list_of_telluric_ranges[0])
            for i in range(1,len(list_of_telluric_ranges)):
                print("                             ",list_of_telluric_ranges[i])    
        print("  Parameters for obtaining the absolute flux calibration:")    
        print("     absolute_flux_file    = ",absolute_flux_file)
        
        if min_wave_flux == 0 :  min_wave_flux = valid_wave_min
        if max_wave_flux == 0 :  max_wave_flux = valid_wave_max
        
        if min_wave_flux  > 0 : print("     min_wave_flux         = ",min_wave_flux)  
        if max_wave_flux  > 0 : print("     max_wave_flux         = ",max_wave_flux)   
        print("     step_flux             = ",step_flux) 
        if exp_time > 0 : 
            print("     exp_time              = ",exp_time) 
        else:
            print("     exp_time              =  reads it from .fits files")
        print("     fit_degree_flux       = ",fit_degree_flux) 
        print("     sky_annulus_low       = ",sky_annulus_low_arcsec,"arcsec")
        print("     sky_annulus_high      = ",sky_annulus_high_arcsec,"arcsec")
        if ha_width > 0 : print("     ha_width              = ",ha_width,"A")  
        if odd_number > 0 : print("     odd_number            = ",odd_number)  
        if smooth > 0 : print("     smooth                = ",smooth)  
        if fit_weight > 0 : print("     fit_weight            = ",fit_weight)  
        if smooth_weight > 0 :     print("     smooth_weight         = ",smooth_weight)  
        if exclude_wlm[0][0] != 0: print("     exclude_wlm           = ",exclude_wlm)  
        
         
        print("\n> Output files:\n")
        if read_fits_cube == "False" : print("  fits_file            =",fits_file)
        print("  integrated spectrum  =",fits_file[:-5]+"_integrated_star_flux.dat")
        if grating in red_gratings :
            print("  telluric_file        =",telluric_file) 
        print("  response_file        =",response_file)
        print(" ")

    else:
        print("\n> No calibration will be performed\n")

# # Read cube from fits file if given

    if read_fits_cube:  
        star_object = read_cube(fits_file, valid_wave_min = valid_wave_min, valid_wave_max = valid_wave_max)
        reduced = True
        exp_time = np.nanmedian(star_object.exptimes)
        
        print(" ")
        exec(object_auto+"=copy.deepcopy(star_object)", globals())
        print("> Cube saved in object", object_auto," !")


# # Running KOALA_REDUCE using rss_list
    
    if reduced == False:
        
        for rss in rss_list:
            if save_rss :
                save_rss_to_fits_file_list.append("auto")
            else:
                save_rss_to_fits_file_list.append("")
             
        if do_cubing:
            print("> Running KOALA_reduce to create combined datacube...")   
        else:
            print("> Running KOALA_reduce ONLY for processing the RSS files provided...") 
    
        star_object=KOALA_reduce(rss_list,
                           #instrument=instrument,
                           path=path,
                           fits_file=fits_file, 
                           obj_name=star,  
                           description=description,
                           save_rss_to_fits_file_list = save_rss_to_fits_file_list,
                           rss_clean=rss_clean,
                           grating = grating,
                           apply_throughput=apply_throughput, 
                           throughput_2D_file = throughput_2D_file,
                           throughput_2D = throughput_2D,
                           correct_ccd_defects = correct_ccd_defects, 
                           fix_wavelengths = fix_wavelengths, 
                           sol = sol,
                           throughput_2D_wavecor = throughput_2D_wavecor,
                           do_extinction= do_extinction,
                           sky_method=sky_method, 
                           n_sky=n_sky,
                           win_sky=win_sky,
                           remove_5577=remove_5577,
                           sky_fibres=sky_fibres,
                           correct_negative_sky = correct_negative_sky,
                           order_fit_negative_sky =order_fit_negative_sky, 
                           kernel_negative_sky = kernel_negative_sky,
                           individual_check = individual_check, 
                           use_fit_for_negative_sky = use_fit_for_negative_sky,
                           force_sky_fibres_to_zero=force_sky_fibres_to_zero,
                           low_fibres= low_fibres,
                           high_fibres=high_fibres,
                           
                           fix_edges=fix_edges,           
                           clean_extreme_negatives=clean_extreme_negatives, percentile_min=percentile_min,
                           remove_negative_median_values=remove_negative_median_values,
                           clean_cosmics = clean_cosmics,
                           width_bl = width_bl, kernel_median_cosmics = kernel_median_cosmics, 
                           cosmic_higher_than = cosmic_higher_than, extra_factor = extra_factor,
                                                    
                           do_cubing = do_cubing, do_alignment=do_alignment, make_combined_cube=make_combined_cube,
                           pixel_size_arcsec=pixel_size, 
                           kernel_size_arcsec=kernel_size, 
                           size_arcsec=size_arcsec,
                           edgelow = edgelow, edgehigh = edgehigh,
                           ADR= ADR,
                           adr_index_fit=adr_index_fit, g2d=g2d,
                           kernel_tracing = kernel_tracing,
                           
                           jump=jump,
                           box_x=box_x, box_y=box_y,
                           trim_values=trim_values,
                           scale_cubes_using_integflux = scale_cubes_using_integflux,
                           flux_ratios = flux_ratios,
                           valid_wave_min = valid_wave_min, 
                           valid_wave_max = valid_wave_max,
                           log=log,
                           gamma = gamma,
                           plot= plot, 
                           plot_rss=plot_rss,
                           plot_spectra = plot_spectra,
                           plot_weight=plot_weight,
                           fig_size=fig_size,
                           verbose = verbose,
                           warnings=warnings ) 
    
        # Save object is given
        if object_auto != 0: # and make_combined_cube == True:
            exec(object_auto+"=copy.deepcopy(star_object)", globals())
            print("> Cube saved in object", object_auto," !")

    else:
        if read_fits_cube == False:
            print("> Python object",object_auto,"already created.")
            exec("star_object=copy.deepcopy("+object_auto+")", globals())


    #  Perform the calibration

    if do_calibration :

        # Check exposition times
        if exp_time == 0:
            different_times = False
            try:
                exptimes = star_object.combined_cube.exptimes
            except Exception:    
                exptimes = star_object.exptimes
            
            exp_time1 = exptimes[0]
            print("\n> Exposition time reading from rss1: ",exp_time1," s")
            
            exp_time_list = [exp_time1]
            for i in range(1,len(exptimes)):
                exp_time_n = exptimes [i]
                exp_time_list.append(exp_time_n)
                if exp_time_n != exp_time1:
                    print("  Exposition time reading from rss"+np.str(i)," = ", exp_time_n," s")
                    different_times = True
            
            if  different_times:
                print("\n> WARNING! not all rss files have the same exposition time!")
                exp_time = np.nanmedian(exp_time_list)
                print("  The median exposition time is {} s, using for calibrating...".format(exp_time))
            else:
                print("  All rss files have the same exposition times!")
                exp_time = exp_time1
                        
    
         # Reading the cube from fits file, the parameters are in self.
         # But creating the cube from rss files keeps self.rss and self.combined_cube !! 
    
        if read_fits_cube == True:
            star_cube = star_object
        else:
            star_cube = star_object.combined_cube

    
        after_telluric_correction = False
        if grating in red_gratings :
    
            # Extract the integrated spectrum of the star & save it
    
            print("\n> Extracting the integrated spectrum of the star...")
           
            star_cube.half_light_spectrum(r_max=r_max, plot=plot)
            spectrum_to_text_file(star_cube.wavelength,
                              star_cube.integrated_star_flux, 
                              filename=fits_file[:-5]+"_integrated_star_flux_before_TC.dat")
        
           
    
     # Find telluric correction CAREFUL WITH apply_tc=True
    
            print("\n> Finding telluric correction...")
            try:                         
                telluric_correction_star = telluric_correction_from_star(star_object,
                                                                         list_of_telluric_ranges = list_of_telluric_ranges,
                                                                         order = order_telluric,
                                                                         apply_tc=apply_tc, 
                                                                         wave_min=valid_wave_min, 
                                                                         wave_max=valid_wave_max,
                                                                         plot=plot, verbose=True)
        
                if apply_tc:  
                    # Saving calibration as a text file 
                    spectrum_to_text_file(star_cube.wavelength,
                                          telluric_correction_star, 
                                          filename=telluric_file)
                
                    after_telluric_correction = True
                    if object_auto != 0: exec(object_auto+"=copy.deepcopy(star_object)", globals())
        
            except Exception:  
                print("\n> Finding telluric correction FAILED!")
    
    
     #Flux calibration
        
        print("\n> Finding absolute flux calibration...")
        
     # Now we read the absolute flux calibration data of the calibration star and get the response curve
     # (Response curve: correspondence between counts and physical values)
     # Include exp_time of the calibration star, as the results are given per second
     # For this BE CAREFUL WITH ABSORPTIONS (Halpha) and check behaviour in the edges of the CCD
     # Change fit_degree (3,5,7), step, min_wave, max_wave to get better fits !!!  
       
        #try: 
        pepe = 1
        if pepe == 1:  
            print("  - Absolute flux file = ", absolute_flux_file)
            star_cube.do_response_curve(absolute_flux_file, 
                                        plot=plot, 
                                        min_wave_flux=min_wave_flux, 
                                        max_wave_flux=max_wave_flux,
                                        step_flux=step_flux, 
                                        exp_time=exp_time, 
                                        fit_degree_flux=fit_degree_flux,
                                        ha_width=ha_width,
                                        sky_annulus_low_arcsec=sky_annulus_low_arcsec,
                                        sky_annulus_high_arcsec=sky_annulus_high_arcsec,
                                        after_telluric_correction=after_telluric_correction,
                                        exclude_wlm = exclude_wlm,
                                        odd_number=odd_number,
                                        smooth=smooth,
                                        fit_weight=fit_weight,
                                        smooth_weight=smooth_weight)
         
        
            spectrum_to_text_file(star_cube.wavelength,
                              star_cube.integrated_star_flux, 
                              filename=fits_file[:-5]+"_integrated_star_flux.dat")
        
         # Now we can save this calibration as a text file 
        
            spectrum_to_text_file(star_cube.wavelength,
                                  star_cube.response_curve, 
                                  filename=response_file, verbose = False)
            
            print('\n> Absolute flux calibration (response) saved in text file :\n  "'+response_file+'"')
            
            if object_auto != 0: exec(object_auto+"=copy.deepcopy(star_object)", globals())
                
        # except Exception:  
        #     print("\n> Finding absolute flux calibration FAILED!")
    
        if object_auto == 0: 
            print("\n> Calibration star processed and stored in object 'star_object' !")
            print("  Now run 'name = copy.deepcopy(star_object)' to save the object with other name")
    
        return star_object
    else:
        print("\n> No calibration has been performed")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------