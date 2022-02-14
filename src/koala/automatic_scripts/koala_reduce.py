#!/usr/bin/python
# -*- coding: utf-8 -*-

from koala.constants import red_gratings
from koala.io import full_path, read_table, save_cube_to_fits_file
from koala.cube import Interpolated_cube,build_combined_cube, align_n_cubes
from koala.RSS import RSS
from koala.KOALA_RSS import KOALA_RSS

import numpy as np


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#    MACRO FOR EVERYTHING 19 Sep 2019, including alignment n - cubes
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class KOALA_reduce(RSS,Interpolated_cube):                                      # TASK_KOALA_reduce
 
    def __init__(self, rss_list, fits_file="", obj_name="",  description = "", path="",
                 do_rss=True, do_cubing=True, do_alignment=True, make_combined_cube=True, rss_clean=False, 
                 save_aligned_cubes= False, save_rss_to_fits_file_list = [], #["","","","","","","","","",""],  
                 # RSS
                 flat="",
                 grating = "",
                 # skyflat_file is a RSS, skyflat and skyflat_list are the names of objects keeping the relative throughput of skyflats
                 apply_throughput=True,  
                 throughput_2D=[], throughput_2D_file="",
                 throughput_2D_wavecor = False,
                 #nskyflat=True, skyflat = "", skyflat_file ="",throughput_file ="", nskyflat_file="",
                 #skyflat_list=["","","","","","","","","",""], 
                 #This line is needed if doing FLAT when reducing (NOT recommended)
                 #plot_skyflat=False, wave_min_scale=0, wave_max_scale=0, ymin=0, ymax=0,
                 # Correct CCD defects & high cosmics
                 correct_ccd_defects = False, remove_5577 = False, kernel_correct_ccd_defects = 51, plot_suspicious_fibres=False,
                 # Correct for small shofts in wavelength
                 fix_wavelengths = False, sol = [0,0,0],
                 # Correct for extinction
                 do_extinction=True,
                 # Telluric correction                      
                 telluric_correction = [0], telluric_correction_file="",
                 telluric_correction_list=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]],                  	
                 # Sky substraction
                 sky_method="self", n_sky=50, sky_fibres=[], win_sky = 0, 
                 sky_spectrum=[], sky_rss=[0], scale_sky_rss=0, scale_sky_1D=0, 
                 sky_spectrum_file = "", sky_spectrum_file_list = ["","","","","","","","","",""],               
                 sky_list=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]],
                 ranges_with_emission_lines =  [0],
                 cut_red_end = 0,
                 
                 correct_negative_sky = False, 
                 order_fit_negative_sky = 3, kernel_negative_sky = 51, individual_check = True, use_fit_for_negative_sky = False,
                 force_sky_fibres_to_zero = True, high_fibres=20, low_fibres = 10,
                 auto_scale_sky = False,
                 brightest_line="Ha", brightest_line_wavelength = 0, sky_lines_file="", 
                 is_sky=False, sky_wave_min = 0, sky_wave_max =0, cut_sky=5., fmin=1, fmax=10,                  
                 individual_sky_substraction=False, fibre_list=[100,200,300,400,500,600,700,800,900], 
                 # Identify emission lines
                 id_el=False, cut=1.5, plot_id_el=True, broad=2.0, id_list=[0], 
                 # Clean sky residuals                    
                 fibres_to_fix=[],                                     
                 clean_sky_residuals = False, features_to_fix =[], sky_fibres_for_residuals=[],
                 remove_negative_median_values = False,
                 fix_edges = False,
                 clean_extreme_negatives = False, percentile_min = 0.5,
                 clean_cosmics = False, #show_cosmics_identification = True,                                                            
                 width_bl = 20., kernel_median_cosmics = 5, cosmic_higher_than = 100., extra_factor = 1.,  max_number_of_cosmics_per_fibre = 12,                                                    

                 # CUBING
                 pixel_size_arcsec=.4, kernel_size_arcsec=1.2,
                 offsets=[],
                 reference_rss = "",
                 ADR=False, ADR_cc = False, force_ADR= False,
                 box_x =[0,-1], box_y=[0,-1], jump = -1, 
                 half_size_for_centroid = 10,
                 ADR_x_fit_list=[],ADR_y_fit_list=[], 
                 g2d = False,
                 adr_index_fit = 2, 
                 step_tracing=25,
                 kernel_tracing = 5,
                 adr_clip_fit=0.3,
                 plot_tracing_maps =[],
                 
                 edgelow=-1, edgehigh=-1,
                 flux_calibration_file="",  # this can be a single file (string) or a list of files (list of strings)
                 flux_calibration=[],       # an array
                 flux_calibration_list=[],  # a list of arrays
                 trim_cube = True, trim_values =[],
                 remove_spaxels_not_fully_covered = True,
                 size_arcsec = [],
                 centre_deg=[],
                 scale_cubes_using_integflux = True,
                 apply_scale=True,
                 flux_ratios = [],
                 cube_list_names=[""],

                 # COMMON TO RSS AND CUBING & PLOTS
                 valid_wave_min = 0, valid_wave_max = 0,
                 log = True	,		# If True and gamma = 0, use colors.LogNorm() [LOG], if False colors.Normalize() [LINEAL] 
                 gamma	= 0,
                 plot= True, plot_rss=True, plot_weight=False, plot_spectra = True, fig_size=12,
                 warnings=False, verbose = False):
        """
        Example
        -------
        >>>  combined_KOALA_cube(['Ben/16jan10049red.fits','Ben/16jan10050red.fits','Ben/16jan10051red.fits'], 
                        fits_file="test_BLUE_reduced.fits", skyflat_file='Ben/16jan10086red.fits', 
                        pixel_size_arcsec=.3, kernel_size_arcsec=1.5, flux_calibration=flux_calibration, 
                        plot= True)    
        """
                
        print("\n\n======================= REDUCING KOALA data =======================\n")

        n_files = len(rss_list)
        sky_rss_list=[]
        pk = "_"+str(int(pixel_size_arcsec))+"p"+str(int((abs(pixel_size_arcsec)-abs(int(pixel_size_arcsec)))*10))+"_"+str(int(kernel_size_arcsec))+"k"+str(int(abs(kernel_size_arcsec*100))-int(kernel_size_arcsec)*100)

        if plot == False:
            plot_rss=False
            if verbose: print("No plotting anything.....\n" )
            
        #if plot_rss == False and plot == True and verbose: print(" plot_rss is false.....")
        
        print("1. Checking input values:")
        
        print("\n  - Using the following RSS files : ")
        rss_object=[]
        cube_object=[]
        cube_aligned_object=[]
        number=1
        
        for rss in range(n_files):
            rss_list[rss] =full_path(rss_list[rss],path)       
            print("    ",rss+1,". : ",rss_list[rss])
            _rss_ = "self.rss"+np.str(number)
            _cube_= "self.cube"+np.str(number)
            _cube_aligned_= "self.cube"+np.str(number)+"_aligned"
            rss_object.append(_rss_)
            cube_object.append(_cube_)
            cube_aligned_object.append(_cube_aligned_)
            number = number +1
            sky_rss_list.append([0])


        if len(save_rss_to_fits_file_list) > 0:
            try:
                if save_rss_to_fits_file_list == "auto":
                    save_rss_to_fits_file_list =[]
                    for rss in range(n_files):
                        save_rss_to_fits_file_list.append("auto")
            except Exception:
                if len(save_rss_to_fits_file_list) != len(n_files):
                    if verbose or warnings : 
                        print("\n  WARNING! List of RSS files to save provided does not have the same number of RSS files!!!\n")
                        print("\n           Using the automatic naming for saving RSS files... \n")
                    save_rss_to_fits_file_list=[]
                    for rss in range(n_files):
                        save_rss_to_fits_file_list.append("auto")
        else:
            for rss in range(n_files):
                save_rss_to_fits_file_list.append("")
            
            
        self.rss_list=rss_list
        
        if number == 1: 
            do_alignment  =False 
            make_combined_cube= False
         
        if rss_clean:
            print("\n  - These RSS files are ready to be cubed & combined, no further process required ...")           
        else:   
            # Check throughput
            if apply_throughput:
                if len(throughput_2D) == 0 and throughput_2D_file == "" :
                    print("\n\n\n  WARNING !!!! \n\n  No 2D throughput data provided, no throughput correction will be applied.\n\n\n")
                    apply_throughput = False
                else:
                    if len(throughput_2D) > 0 :
                        print("\n  - Using the variable provided for applying the 2D throughput correction ...")
                    else:
                        print("\n  - The 2D throughput correction will be applied using the file:")
                        print("  ",throughput_2D_file)
            else:
                print("\n  - No 2D throughput correction will be applied")
            

            # sky_method = "self" "1D" "2D" "none" #1Dfit" "selffit"
              
            if sky_method == "1D" or sky_method == "1Dfit" or sky_method == "selffit":    
                if np.nanmedian(sky_spectrum) != -1 and np.nanmedian(sky_spectrum) != 0 :
                    for i in range(n_files):
                        sky_list[i] = sky_spectrum
                    print("\n  - Using same 1D sky spectrum provided for all object files") 
                else:
                    if np.nanmedian(sky_list[0]) == 0:
                        print("\n  - 1D sky spectrum requested but not found, assuming n_sky =",n_sky,"from the same files")
                        if sky_method in ["1Dfit","1D"]: sky_method = "self"
                    else:
                        print("\n  - List of 1D sky spectrum provided for each object file")

            if sky_method == "2D": 
                try:
                    if np.nanmedian(sky_list[0].intensity_corrected) != 0 :
                        print("\n  - List of 2D sky spectra provided for each object file")
                        for i in range(n_files):
                            sky_rss_list[i]=sky_list[i]
                            sky_list[i] = [0]
                except Exception: 
                    try:
                        if sky_rss == 0 :
                            print("\n  - 2D sky spectra requested but not found, assuming n_sky = 50 from the same files")
                            sky_method = "self"
                    except Exception:       
                        for i in range(n_files):
                            sky_rss_list[i]=sky_rss
                        print("\n  - Using same 2D sky spectra provided for all object files")  
                        
            if sky_method == "self": # or  sky_method == "selffit":
                for i in range(n_files):
                    sky_list[i] = []
                if n_sky == 0 : n_sky = 50
                if len(sky_fibres) == 0:
                    print("\n  - Using n_sky =",n_sky,"to create a sky spectrum")
                else:
                    print("\n  - Using n_sky =",n_sky,"and sky_fibres =",sky_fibres,"to create a sky spectrum")
                                                                                                    
            if grating in red_gratings:                                                                                                                       
                if np.nanmedian(telluric_correction) == 0 and np.nanmedian (telluric_correction_list[0]) == 0 :
                    print("\n  - No telluric correction considered")
                else: 
                    if np.nanmedian (telluric_correction_list[0]) == 0: 
                        for i in range(n_files):
                            telluric_correction_list[i] = telluric_correction
                        print("\n  - Using same telluric correction for all object files")
                    else: print("\n  - List of telluric corrections provided!")
 
        if do_rss: 
            print("\n-------------------------------------------")
            print("2. Reading the data stored in RSS files ...")
                                    
            for i in range(n_files):
                     #skyflat=skyflat_list[i], plot_skyflat=plot_skyflat, throughput_file =throughput_file, nskyflat_file=nskyflat_file,\
                     # This considers the same throughput for ALL FILES !!
                exec(rss_object[i]+'= KOALA_RSS(rss_list[i], rss_clean = rss_clean, save_rss_to_fits_file = save_rss_to_fits_file_list[i], \
                     apply_throughput=apply_throughput, \
                     throughput_2D=throughput_2D, throughput_2D_file=throughput_2D_file, throughput_2D_wavecor=throughput_2D_wavecor, \
                     correct_ccd_defects = correct_ccd_defects, kernel_correct_ccd_defects = kernel_correct_ccd_defects, remove_5577 = remove_5577, plot_suspicious_fibres = plot_suspicious_fibres,\
                     fix_wavelengths = fix_wavelengths, sol = sol,\
                     do_extinction=do_extinction, \
                     telluric_correction = telluric_correction_list[i], telluric_correction_file = telluric_correction_file,\
                     sky_method=sky_method, n_sky=n_sky, sky_fibres=sky_fibres, \
                     sky_spectrum=sky_list[i], sky_rss=sky_rss_list[i], \
                     sky_spectrum_file=sky_spectrum_file, sky_lines_file= sky_lines_file, \
                     scale_sky_1D=scale_sky_1D, correct_negative_sky = correct_negative_sky, \
                     ranges_with_emission_lines=ranges_with_emission_lines,cut_red_end =cut_red_end,\
                     order_fit_negative_sky = order_fit_negative_sky, kernel_negative_sky = kernel_negative_sky, individual_check = individual_check, use_fit_for_negative_sky = use_fit_for_negative_sky,\
                     force_sky_fibres_to_zero = force_sky_fibres_to_zero, high_fibres=high_fibres, low_fibres=low_fibres,\
                     brightest_line_wavelength=brightest_line_wavelength, win_sky = win_sky, \
                     cut_sky=cut_sky, fmin=fmin, fmax=fmax, individual_sky_substraction = individual_sky_substraction, \
                     id_el=id_el, brightest_line=brightest_line, cut=cut, broad=broad, plot_id_el= plot_id_el,id_list=id_list,\
                     clean_sky_residuals = clean_sky_residuals, features_to_fix =features_to_fix, sky_fibres_for_residuals=sky_fibres_for_residuals,\
                     fibres_to_fix=fibres_to_fix, remove_negative_median_values = remove_negative_median_values, fix_edges = fix_edges,\
                     clean_extreme_negatives = clean_extreme_negatives, percentile_min = percentile_min, clean_cosmics = clean_cosmics,\
                     width_bl = width_bl, kernel_median_cosmics = kernel_median_cosmics, cosmic_higher_than = cosmic_higher_than, extra_factor = extra_factor, max_number_of_cosmics_per_fibre=max_number_of_cosmics_per_fibre,\
                     valid_wave_min=valid_wave_min, valid_wave_max=valid_wave_max,\
                     warnings=warnings, verbose = verbose, plot=plot_rss, plot_final_rss=plot_rss, fig_size=fig_size)')
          
        if len(offsets) > 0 and len (ADR_x_fit_list) >  0 and ADR == True :
            #print("\n  Offsets values for alignment AND fitting for ADR correction have been provided, skipping cubing no-aligned rss...")
            do_cubing = False
        elif len(offsets) > 0 and ADR == False :
            #print("\n  Offsets values for alignment given AND the ADR correction is NOT requested, skipping cubing no-aligned rss...")
            do_cubing = False


        if len (ADR_x_fit_list) ==  0 :   # Check if lists with ADR values have been provided, if not create lists with 0
            ADR_x_fit_list = []
            ADR_y_fit_list = []
            for i in range (n_files):
                ADR_x_fit_list.append([0,0,0])
                ADR_y_fit_list.append([0,0,0])
                
        fcal= False
        if flux_calibration_file  != "":   # If files have been provided for the flux calibration, we read them
            fcal = True
            if type(flux_calibration_file) == str :
                if path != "": flux_calibration_file = full_path(flux_calibration_file,path)
                w_star,flux_calibration = read_table(flux_calibration_file, ["f", "f"] ) 
                for i in range(n_files):
                    flux_calibration_list.append(flux_calibration)
                
                if verbose: print("\n  - Using for all the cubes the same flux calibration provided in file:\n   ",flux_calibration_file)
            else:
                if verbose: print("\n  - Using list of files for flux calibration:")
                for i in range(n_files):
                    if path != "": flux_calibration_file[i] = full_path(flux_calibration_file[i],path)
                    print("   ",flux_calibration_file[i])
                    w_star,flux_calibration = read_table(flux_calibration_file[i], ["f", "f"] ) 
                    flux_calibration_list.append(flux_calibration)    
        else:
            if len(flux_calibration) > 0:
                fcal = True
                for i in range(n_files):
                    flux_calibration_list.append(flux_calibration)
                if verbose: print("\n  - Using same flux calibration for all object files")
            else:
                if verbose or warnings: print("\n  - No flux calibration provided!")
                for i in range(n_files):
                    flux_calibration_list.append("")
                
        if do_cubing: 
            if fcal: 
                print("\n------------------------------------------------")
                print("3. Cubing applying flux calibration provided ...")                
            else:    
                print("\n------------------------------------------------------")
                print("3. Cubing without considering any flux calibration ...")
             
            for i in range(n_files):              
                exec(cube_object[i]+'=Interpolated_cube('+rss_object[i]+', pixel_size_arcsec=pixel_size_arcsec, kernel_size_arcsec=kernel_size_arcsec, plot=plot, \
                     half_size_for_centroid=half_size_for_centroid, box_x=box_x, box_y=box_y,\
                     ADR_x_fit = ADR_x_fit_list[i], ADR_y_fit = ADR_y_fit_list[i], ADR=ADR, apply_ADR = False,  \
                     g2d=g2d, kernel_tracing=kernel_tracing, step_tracing=step_tracing, adr_index_fit=adr_index_fit, adr_clip_fit=adr_clip_fit, \
                     plot_tracing_maps = plot_tracing_maps,  plot_spectra=plot_spectra,   \
                     flux_calibration=flux_calibration_list[i], edgelow=edgelow, edgehigh=edgehigh, size_arcsec=size_arcsec, centre_deg=centre_deg,warnings=warnings)')    
        else:
            if do_alignment:
                print("\n------------------------------------------------")
                if ADR == False:
                    print("3. Offsets provided, ADR correction NOT requested, cubing will be done using aligned cubes ...")   
                else:
                    print("3. Offsets AND correction for ADR provided, cubing will be done using aligned cubes ...")   

        rss_list_to_align=[]    
        cube_list = []
        for i in range(n_files):
            exec('rss_list_to_align.append('+rss_object[i]+')')
            if do_cubing:
                exec('cube_list.append('+cube_object[i]+')')
            else:
                cube_list.append([0])

        if do_alignment:   
            if len(offsets) == 0: 
                print("\n--------------------------------")
                print("4. Aligning individual cubes ...")
            else:
                print("\n-----------------------------------------------------")
                print("4. Checking offsets data provided and performing cubing ...")
                

            cube_aligned_list=align_n_cubes(rss_list_to_align, cube_list=cube_list, flux_calibration_list=flux_calibration_list, pixel_size_arcsec=pixel_size_arcsec, 
                                            kernel_size_arcsec=kernel_size_arcsec, 
                                            size_arcsec=size_arcsec,centre_deg=centre_deg,
                                            offsets=offsets, 
                                            reference_rss= reference_rss,
                                            ADR=ADR, jump=jump, force_ADR=force_ADR,
                                            edgelow=edgelow, edgehigh=edgehigh, 
                                            ADR_x_fit_list =ADR_x_fit_list, ADR_y_fit_list = ADR_y_fit_list, 
                                            half_size_for_centroid=half_size_for_centroid, box_x=box_x, box_y=box_y,
                                            adr_index_fit=adr_index_fit, g2d=g2d, step_tracing = step_tracing, kernel_tracing = kernel_tracing,
                                            adr_clip_fit = adr_clip_fit,
                                            plot=plot, plot_weight=plot_weight, 
                                            plot_tracing_maps=plot_tracing_maps, plot_spectra=plot_spectra,
                                            warnings=warnings, verbose=verbose)      

            for i in range(n_files):             
                exec(cube_aligned_object[i]+'=cube_aligned_list[i]')
                
        else:
            
            if ADR == True and np.nanmedian(ADR_x_fit_list) ==  0:
            # If not alignment but ADR is requested
            
                print("\n--------------------------------")
                print("4. Applying ADR ...")
            
                for i in range(n_files): 
                    exec(cube_object[i]+'=Interpolated_cube('+rss_object[i]+', pixel_size_arcsec, kernel_size_arcsec, plot=plot, half_size_for_centroid=half_size_for_centroid, adr_index_fit=adr_index_fit, g2d=g2d, step_tracing = step_tracing, kernel_tracing = kernel_tracing, adr_clip_fit=adr_clip_fit, plot_tracing_maps=plot_tracing_maps, \
                         ADR_x_fit = cube_list[i].ADR_x_fit, ADR_y_fit = cube_list[i].ADR_y_fit, box_x=box_x, box_y=box_y, check_ADR = True, \
                         flux_calibration=flux_calibration_list[i], edgelow=edgelow, edgehigh=edgehigh, size_arcsec=size_arcsec, centre_deg=centre_deg,warnings=warnings)')   
 

       # Save aligned cubes to fits files
        if save_aligned_cubes:             
            print("\n> Saving aligned cubes to fits files ...")
            if cube_list_names[0] == "" :  
                for i in range(n_files):
                    if i < 9: 
                        replace_text = "_"+obj_name+"_aligned_cube_0"+np.str(i+1)+pk+".fits" 
                    else: replace_text = "_aligned_cube_"+np.str(i+1)+pk+".fits"                   
                    
                    aligned_cube_name=  rss_list[i].replace(".fits", replace_text)
                    save_cube_to_fits_file(cube_aligned_list[i], aligned_cube_name, path=path) 
            else:
                for i in range(n_files):
                    save_cube_to_fits_file(cube_aligned_list[i], cube_list_names[i], path=path) 
       
        ADR_x_fit_list = []
        ADR_y_fit_list = []  
                  
        if ADR and do_cubing:
            print("\n> Values of the centroid fitting for the ADR correction:\n")   
            
            for i in range(n_files): 
                try:
                    if adr_index_fit == 1:
                        print("ADR_x_fit  = ["+np.str(cube_list[i].ADR_x_fit[0])+","+np.str(cube_list[i].ADR_x_fit[1])+"]")
                        print("ADR_y_fit  = ["+np.str(cube_list[i].ADR_y_fit[0])+","+np.str(cube_list[i].ADR_y_fit[1])+"]")           
                    elif adr_index_fit == 2:
                        print("ADR_x_fit  = ["+np.str(cube_list[i].ADR_x_fit[0])+","+np.str(cube_list[i].ADR_x_fit[1])+","+np.str(cube_list[i].ADR_x_fit[2])+"]")
                        print("ADR_y_fit  = ["+np.str(cube_list[i].ADR_y_fit[0])+","+np.str(cube_list[i].ADR_y_fit[1])+","+np.str(cube_list[i].ADR_y_fit[2])+"]")
                    elif adr_index_fit == 3:
                        print("ADR_x_fit  = ["+np.str(cube_list[i].ADR_x_fit[0])+","+np.str(cube_list[i].ADR_x_fit[1])+","+np.str(cube_list[i].ADR_x_fit[2])+","+np.str(cube_list[i].ADR_x_fit[3])+"]")
                        print("ADR_y_fit  = ["+np.str(cube_list[i].ADR_y_fit[0])+","+np.str(cube_list[i].ADR_y_fit[1])+","+np.str(cube_list[i].ADR_y_fit[2])+","+np.str(cube_list[i].ADR_y_fit[3])+"]")

                except Exception:
                    print("  WARNING: Something wrong happened printing the ADR fit values! Results are:")
                    print("  ADR_x_fit  = ",cube_list[i].ADR_x_fit)
                    print("  ADR_y_fit  = ",cube_list[i].ADR_y_fit)

                _x_ = []
                _y_ = []
                for j in range(len(cube_list[i].ADR_x_fit)):
                    _x_.append(cube_list[i].ADR_x_fit[j])
                    _y_.append(cube_list[i].ADR_y_fit[j])   
                ADR_x_fit_list.append(_x_)
                ADR_y_fit_list.append(_y_)
   
        if obj_name == "":
            exec('obj_name = '+rss_object[0]+'.object')
            obj_name=obj_name.replace(" ", "_")
 
        if make_combined_cube and n_files > 1 :   
            print("\n---------------------------")
            print("5. Making combined cube ...")
                       
            self.combined_cube=build_combined_cube(cube_aligned_list,   obj_name=obj_name, description=description,
                                                  fits_file = fits_file, path=path,
                                                  scale_cubes_using_integflux= scale_cubes_using_integflux, 
                                                  flux_ratios = flux_ratios, apply_scale = apply_scale,
                                                  edgelow=edgelow, edgehigh=edgehigh,
                                                  ADR=ADR, ADR_cc = ADR_cc, jump = jump, pk = pk, 
                                                  ADR_x_fit_list=ADR_x_fit_list, ADR_y_fit_list=ADR_y_fit_list,
                                                  force_ADR=force_ADR,
                                                  half_size_for_centroid = half_size_for_centroid,
                                                  box_x=box_x, box_y=box_y,  
                                                  adr_index_fit=adr_index_fit, g2d=g2d,
                                                  step_tracing = step_tracing, 
                                                  kernel_tracing = kernel_tracing, adr_clip_fit=adr_clip_fit,
                                                  plot_tracing_maps = plot_tracing_maps,
                                                  trim_cube = trim_cube,  trim_values =trim_values, 
                                                  remove_spaxels_not_fully_covered = remove_spaxels_not_fully_covered,
                                                  plot=plot, plot_weight= plot_weight, plot_spectra=plot_spectra,
                                                  verbose=verbose, say_making_combined_cube= False)
        else:
            if n_files > 1:
                if do_alignment == False and  do_cubing == False:
                    print("\n> As requested, skipping cubing...")
                else:
                    print("\n  No combined cube obtained!")
                    
            else:
                print("\n> Only one file provided, no combined cube obtained")
                # Trimming cube if requested or needed
                cube_aligned_list[0].trim_cube(trim_cube=trim_cube, trim_values=trim_values, ADR=ADR,
                               half_size_for_centroid =half_size_for_centroid, 
                               adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing, 
                               kernel_tracing = kernel_tracing, adr_clip_fit=adr_clip_fit,
                               plot_tracing_maps=plot_tracing_maps,
                               remove_spaxels_not_fully_covered = remove_spaxels_not_fully_covered,
                               box_x=box_x, box_y=box_y, edgelow=edgelow, edgehigh=edgehigh, 
                               plot_weight = plot_weight, fcal=fcal, plot=plot)    
                
                # Make combined cube = aligned cube

                self.combined_cube = cube_aligned_list[0]          


        
            
        self.parameters = locals()
            


        print("\n================== REDUCING KOALA DATA COMPLETED ====================\n\n")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------