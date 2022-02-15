#!/usr/bin/python
# -*- coding: utf-8 -*-

from koala.io import full_path, read_table
from koala.automatic_scripts.koala_reduce import KOALA_reduce
from koala.cube import build_combined_cube

import numpy as np


def automatic_KOALA_reduce(KOALA_REDUCE_FILE, path=""):   

    print("\n\n=============== running automatic_KOALA_reduce =======================")
 
    global hikids
    
    throughput_2D_variable = ""
    
    flux_calibration_file = ""
    flux_calibration_file_list = []
    flux_calibration = []
    flux_calibration = ""
    flux_calibration_list =[]  # Check
    
    telluric_correction_name = ""
    telluric_correction_file = ""
    
    skyflat_list = []
    
    telluric_correction_list = []
    telluric_correction_list_ = []
    
    do_cubing = True
    make_combined_cube = True  
    do_alignment = True
    read_cube = False
    
    # These are the default values in KOALA_REDUCE
    
    rss_list = [] 
    save_rss = False  # No in KOALA_REDUCE, as it sees if save_rss_list is empty or not
    save_rss_list = []
    cube_list=[]
    cube_list_names = []
    save_aligned_cubes = False

    apply_throughput=False
    throughput_2D = []
    throughput_2D_file = ""
    throughput_2D_wavecor = False

    correct_ccd_defects = False
    kernel_correct_ccd_defects=51

    fix_wavelengths = False
    sol =[0,0,0]
    do_extinction = False

    do_telluric_correction = False
        
    sky_method = "none"
    sky_spectrum = []
    sky_spectrum_name = ""
 #   sky_spectrum_file = ""   #### NEEDS TO BE IMPLEMENTED
    sky_list = []    
    sky_fibres  = [1000]         
    sky_lines_file = ""

    scale_sky_1D = 1.
    auto_scale_sky 	=	False
    n_sky = 50 
    print_n_sky = False
    win_sky = 0
    remove_5577 = False    

    correct_negative_sky = False
    order_fit_negative_sky =3 
    kernel_negative_sky = 51
    individual_check = True
    use_fit_for_negative_sky = False
    force_sky_fibres_to_zero = True
    high_fibres=20
    low_fibres=10
    
    brightest_line="Ha"
    brightest_line_wavelength=0.      
    ranges_with_emission_lines = [0]
    cut_red_end = 0			
    
    id_el=False
    id_list=[0]
    cut=1.5
    broad=1.8
    plot_id_el=False
    
    clean_sky_residuals = False
    features_to_fix =[]
    sky_fibres_for_residuals = []
    sky_fibres_for_residuals_print = "Using the same n_sky fibres"

    remove_negative_median_values = False    
    fix_edges=False
    clean_extreme_negatives = False
    percentile_min=0.9  

    clean_cosmics = False
    width_bl = 20.
    kernel_median_cosmics = 5 
    cosmic_higher_than 	=	100. 
    extra_factor =	1.
    max_number_of_cosmics_per_fibre = 12


    offsets=[]
    reference_rss = ""
    ADR = False
    ADR_cc = False
    force_ADR=False
    box_x=[]
    box_y=[]
    jump=-1
    half_size_for_centroid = 10
    ADR_x_fit_list = []
    ADR_y_fit_list = []
    
    g2d = False
    adr_index_fit = 2
    step_tracing=25
    kernel_tracing = 5
    adr_clip_fit=0.3
    plot_tracing_maps =[]
    
    edgelow  = -1
    edgehigh = -1


    delta_RA  = 0
    delta_DEC = 0

    trim_cube = False
    trim_values=[]
    size_arcsec=[]
    centre_deg =[]
    scale_cubes_using_integflux = True
    remove_spaxels_not_fully_covered = True
    flux_ratios =[]

    valid_wave_min=0 
    valid_wave_max=0 
            
    plot = True
    plot_rss = True
    plot_weight = False
    plot_spectra = True
    fig_size	=12.

    log = True
    gamma = 0.

    warnings	=False
    verbose=True    
              
    if path != "" : KOALA_REDUCE_FILE = full_path(KOALA_REDUCE_FILE, path)   # VR
    config_property, config_value = read_table(KOALA_REDUCE_FILE, ["s", "s"] )
    
    print("\n> Reading configuration file", KOALA_REDUCE_FILE,"...\n")
           
    for i in range(len(config_property)):
        
        if  config_property[i] == "pixel_size" : 	 pixel_size = float(config_value[i])         
        if  config_property[i] == "kernel_size" : 	 
            kernel_size = float(config_value[i])
            pk = "_"+str(int(pixel_size))+"p"+str(int((abs(pixel_size)-abs(int(pixel_size)))*10))+"_"+str(int(kernel_size))+"k"+str(int(abs(kernel_size*100))-int(kernel_size)*100)
        if  config_property[i] == "date" : 	 date = config_value[i]
        if  config_property[i] == "grating" : 	 grating = config_value[i]

        if  config_property[i] == "path" : 	 
            path = config_value[i]
       
        if  config_property[i] == "obj_name" : 
            obj_name = config_value[i]
            description = obj_name
            fits_file = path+obj_name+"_"+grating+pk+".fits"            
            Python_obj_name = obj_name + "_" + grating+pk
        if  config_property[i] == "description" :  description = config_value[i]       
        if  config_property[i] == "Python_obj_name" : Python_obj_name = config_value[i]
    
        if  config_property[i] == "flux_calibration_file" :
            flux_calibration_name = ""
            flux_calibration_file_  = config_value[i]
            if len(flux_calibration_file_.split("/")) == 1:
                flux_calibration_file = path+flux_calibration_file_
            else:
                flux_calibration_file = flux_calibration_file_                     
            flux_calibration_file_list.append(flux_calibration_file)
        if  config_property[i] == "telluric_correction_file" : 
            telluric_correction_name = ""
            telluric_correction_file_  = config_value[i]   
            if len(telluric_correction_file_.split("/")) == 1:
                telluric_correction_file = path+telluric_correction_file_
            else:
                telluric_correction_file = telluric_correction_file_               
            telluric_correction_list_.append(config_value[i])

        if  config_property[i] == "flux_calibration_name" :  
            flux_calibration_name = config_value[i]
            #flux_calibration_name_list.append(flux_calibration_name)
            
        if  config_property[i] == "telluric_correction_name" :  
            telluric_correction_name = config_value[i]

        if  config_property[i] == "fits_file"   :  
            fits_file_ = config_value[i]
            if len(fits_file_.split("/")) == 1:
                fits_file = path+fits_file_
            else:
                fits_file = fits_file_
            
        if  config_property[i] == "save_aligned_cubes" :
            if config_value[i] == "True" : save_aligned_cubes = True 
        
        if  config_property[i] == "rss_file" : 
            rss_file_ = config_value[i]
            if len(rss_file_.split("/")) == 1:
                _rss_file_ = path+rss_file_
            else:
                _rss_file_ = rss_file_    
            rss_list.append(_rss_file_)
        if  config_property[i] == "cube_file" : 
            cube_file_ = config_value[i]
            if len(cube_file_.split("/")) == 1:
                _cube_file_ = path+cube_file_
            else:
                _cube_file_ = cube_file_    
            cube_list_names.append(_cube_file_)
            cube_list.append(_cube_file_)   # I am not sure about this... 
            
        if  config_property[i] == "rss_clean" :
            if config_value[i] == "True" : 
                rss_clean = True 
            else: rss_clean = False          
        if  config_property[i] == "save_rss" :
            if config_value[i] == "True" : 
                save_rss = True 
            else: save_rss = False  
        if  config_property[i] == "do_cubing" and  config_value[i] == "False" :  do_cubing = False 

        if  config_property[i] == "apply_throughput" :
            if config_value[i] == "True" : 
                apply_throughput = True 
            else: apply_throughput = False              

        if  config_property[i] == "throughput_2D_file" : 
            throughput_2D_file_ = config_value[i]
            if len(throughput_2D_file_.split("/")) == 1:
                throughput_2D_file = path+throughput_2D_file_
            else:
                throughput_2D_file = throughput_2D_file_  
        
        if  config_property[i] == "throughput_2D" : throughput_2D_variable = config_value[i]

        if  config_property[i] == "throughput_2D_wavecor" :
            if config_value[i] == "True" : 
                throughput_2D_wavecor = True 
            else: throughput_2D_wavecor = False  

        if  config_property[i] == "correct_ccd_defects" :
            if config_value[i] == "True" : 
                correct_ccd_defects = True 
            else: correct_ccd_defects = False  
        if  config_property[i] == "kernel_correct_ccd_defects" : 	 kernel_correct_ccd_defects = float(config_value[i])     
                
        if  config_property[i] == "fix_wavelengths":
            if config_value[i] == "True" : 
                fix_wavelengths = True 
            else: fix_wavelengths = False
        if  config_property[i] == "sol" :
            sol_ = config_value[i].strip('][').split(',')
            if float(sol_[0]) == -1:
                sol = [float(sol_[0])]
            else:
                sol = [float(sol_[0]),float(sol_[1]),float(sol_[2])]
            
        if  config_property[i] == "do_extinction":
            if config_value[i] == "True" : 
                do_extinction = True 
            else: do_extinction = False            
            
        if  config_property[i] == "sky_method" : sky_method = config_value[i]
                
        if  config_property[i] == "sky_file" : sky_list.append(path+config_value[i])            
        if  config_property[i] == "n_sky" : n_sky=int(config_value[i])

        if config_property[i] == "sky_fibres" :  
            sky_fibres_ =  config_value[i]
            if sky_fibres_[0:5] == "range":
                sky_fibres_ = sky_fibres_[6:-1].split(',')
                sky_fibres = list(range(np.int(sky_fibres_[0]),np.int(sky_fibres_[1])))
                sky_fibres_print = "range("+sky_fibres_[0]+","+sky_fibres_[1]+")"
            else:
                sky_fibres_ = config_value[i].strip('][').split(',')
                for i in range(len(sky_fibres_)):
                    sky_fibres.append(float(sky_fibres_[i]))                    
                sky_fibres_print =  sky_fibres  

        if  config_property[i] == "win_sky" : win_sky =  int(config_value[i])

        if  config_property[i] == "sky_spectrum" :
            if config_value[i] != "[0]" :
                sky_spectrum_name = config_value[i]
                exec("sky_spectrum ="+sky_spectrum_name)
            else:
                sky_spectrum = []
        if  config_property[i] == "scale_sky_1D" : 	 scale_sky_1D = float(config_value[i]) 

        if  config_property[i] == "auto_scale_sky" :
            if config_value[i] == "True" : 
                auto_scale_sky = True 
            else: auto_scale_sky = False  

        if  config_property[i] == "sky_lines_file" : sky_lines_file = config_value[i]

        if  config_property[i] == "correct_negative_sky" :
            if config_value[i] == "True" : 
                correct_negative_sky = True 
            else: correct_negative_sky = False 
 
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
                
        if  config_property[i] == "remove_5577" :
            if config_value[i] == "True" : 
                remove_5577 = True 
            else: remove_5577 = False             

        if  config_property[i] == "do_telluric_correction" :
            if config_value[i] == "True" : 
                do_telluric_correction = True 
            else: 
                do_telluric_correction = False  
                telluric_correction_name = ""
                telluric_correction_file = ""
            
        if  config_property[i] == "brightest_line" :  brightest_line = config_value[i]
        if  config_property[i] == "brightest_line_wavelength" : brightest_line_wavelength = float(config_value[i])
             
        if config_property[i] == "ranges_with_emission_lines":
            ranges_with_emission_lines_ = config_value[i].strip('[]').replace('],[', ',').split(',')
            ranges_with_emission_lines=[]
            for i in range(len(ranges_with_emission_lines_)):
                if i % 2 == 0 : ranges_with_emission_lines.append([float(ranges_with_emission_lines_[i]),float(ranges_with_emission_lines_[i+1])])                       
        if  config_property[i] == "cut_red_end" :  cut_red_end = config_value[i]

        # CHECK id_el
        if  config_property[i] == "id_el" : 
            if config_value[i] == "True" : 
                id_el = True 
            else: id_el = False 
        if  config_property[i] == "cut" : cut = float(config_value[i])
        if  config_property[i] == "broad" : broad = float(config_value[i])
        if  config_property[i] == "id_list":
            id_list_ = config_value[i].strip('][').split(',')
            for i in range(len(id_list_)):
                id_list.append(float(id_list_[i]))               
        if  config_property[i] == "plot_id_el" : 
            if config_value[i] == "True" : 
                plot_id_el = True 
            else: plot_id_el = False 

        if  config_property[i] == "clean_sky_residuals" and  config_value[i] == "True" : clean_sky_residuals = True 
        if  config_property[i] == "fix_edges" and  config_value[i] == "True" : fix_edges = True 

        if  config_property[i] == "feature_to_fix" :     
            feature_to_fix_ = config_value[i]  #.strip('][').split(',')
            feature_to_fix = []
            feature_to_fix.append(feature_to_fix_[2:3])
            feature_to_fix.append(float(feature_to_fix_[5:9]))
            feature_to_fix.append(float(feature_to_fix_[10:14]))
            feature_to_fix.append(float(feature_to_fix_[15:19]))
            feature_to_fix.append(float(feature_to_fix_[20:24]))
            feature_to_fix.append(float(feature_to_fix_[25:26]))
            feature_to_fix.append(float(feature_to_fix_[27:29]))
            feature_to_fix.append(float(feature_to_fix_[30:31]))
            if feature_to_fix_[32:37] == "False":  
                feature_to_fix.append(False)
            else:
                feature_to_fix.append(True)
            if feature_to_fix_[-6:-1] == "False":  
                feature_to_fix.append(False)
            else:
                feature_to_fix.append(True)                
        
            features_to_fix.append(feature_to_fix)
        
        if config_property[i] == "sky_fibres_for_residuals" :  
            sky_fibres_for_residuals_ =  config_value[i]
            if sky_fibres_for_residuals_[0:5] == "range":
                sky_fibres_for_residuals_ = sky_fibres_for_residuals_[6:-1].split(',')
                sky_fibres_for_residuals = list(range(np.int(sky_fibres_for_residuals_[0]),np.int(sky_fibres_for_residuals_[1])))
                sky_fibres_for_residuals_print = "range("+sky_fibres_for_residuals_[0]+","+sky_fibres_for_residuals_[1]+")"

            else:
                sky_fibres_for_residuals_ = config_value[i].strip('][').split(',')
                for i in range(len(sky_fibres_for_residuals_)):
                    sky_fibres_for_residuals.append(float(sky_fibres_for_residuals_[i]))                    
                sky_fibres_for_residuals_print =   sky_fibres_for_residuals  

        if  config_property[i] == "clean_cosmics" and config_value[i] == "True" : clean_cosmics = True 
        if  config_property[i] == "width_bl" : width_bl = float(config_value[i])  
        if  config_property[i] == "kernel_median_cosmics" : kernel_median_cosmics = int(config_value[i])  
        if  config_property[i] == "max_number_of_cosmics_per_fibre" : max_number_of_cosmics_per_fibre = int(config_value[i])  
        
        if  config_property[i] == "cosmic_higher_than" : cosmic_higher_than = float(config_value[i])  
        if  config_property[i] == "extra_factor" : extra_factor = float(config_value[i])  

        if  config_property[i] == "clean_extreme_negatives" :
            if config_value[i] == "True" : clean_extreme_negatives = True 
        if  config_property[i] == "percentile_min" : percentile_min = float(config_value[i])  

        if  config_property[i] == "remove_negative_median_values":
            if config_value[i] == "True" : 
                remove_negative_median_values = True 
            else: remove_negative_median_values = False 
     
        if  config_property[i] == "read_cube" : 
            if config_value[i] == "True" : 
                read_cube = True 
            else: read_cube = False   
   
        if  config_property[i] == "offsets" :     
            offsets_ = config_value[i].strip('][').split(',')
            for i in range(len(offsets_)):
                offsets.append(float(offsets_[i]))
 
        if  config_property[i] == "reference_rss" : reference_rss =  int(config_value[i])
                  
        if  config_property[i] == "valid_wave_min" : valid_wave_min = float(config_value[i])
        if  config_property[i] == "valid_wave_max" : valid_wave_max = float(config_value[i])


        if  config_property[i] == "half_size_for_centroid": half_size_for_centroid = int(config_value[i])

        if  config_property[i] == "box_x" :     
            box_x_ = config_value[i].strip('][').split(',')
            for i in range(len(box_x_)):
                box_x.append(int(box_x_[i]))            
        if  config_property[i] == "box_y" :     
            box_y_ = config_value[i].strip('][').split(',')
            for i in range(len(box_y_)):
                box_y.append(int(box_y_[i]))         

        if  config_property[i] == "adr_index_fit": adr_index_fit = int(config_value[i])
        if  config_property[i] == "g2d": 
            if config_value[i] == "True" : 
                g2d = True
            else: g2d = False
        if  config_property[i] == "step_tracing" : step_tracing =  int(config_value[i])
        if  config_property[i] == "adr_clip_fit" : adr_clip_fit =  float(config_value[i])
        
        if  config_property[i] == "plot_tracing_maps" :
            plot_tracing_maps_ = config_value[i].strip('][').split(',')
            for i in range(len(plot_tracing_maps_)):
                plot_tracing_maps.append(float(plot_tracing_maps_[i]))

        if  config_property[i] == "edgelow" : edgelow = int(config_value[i])
        if  config_property[i] == "edgehigh" : edgehigh = int(config_value[i])
        
        if  config_property[i] == "ADR" : 
            if config_value[i] == "True" : 
                ADR = True 
            else: ADR = False   
        if  config_property[i] == "ADR_cc" : 
            if config_value[i] == "True" : 
                ADR_cc = True 
            else: ADR_cc = False              
        if  config_property[i] == "force_ADR" : 
            if config_value[i] == "True" : 
                force_ADR = True 
            else: force_ADR = False  

        if  config_property[i] == "ADR_x_fit" :     
            ADR_x_fit_ = config_value[i].strip('][').split(',')
            if len(ADR_x_fit_) == 4:
                ADR_x_fit_list.append([float(ADR_x_fit_[0]),float(ADR_x_fit_[1]), float(ADR_x_fit_[2]),  float(ADR_x_fit_[3])])
            else:
                ADR_x_fit_list.append([float(ADR_x_fit_[0]),float(ADR_x_fit_[1]), float(ADR_x_fit_[2])])
            
        if  config_property[i] == "ADR_y_fit" :     
            ADR_y_fit_ = config_value[i].strip('][').split(',')
            if len(ADR_y_fit_) == 4:
                ADR_y_fit_list.append([float(ADR_y_fit_[0]),float(ADR_y_fit_[1]), float(ADR_y_fit_[2]),  float(ADR_y_fit_[3])])
            else:
                ADR_y_fit_list.append([float(ADR_y_fit_[0]),float(ADR_y_fit_[1]), float(ADR_y_fit_[2])])

        if  config_property[i] == "kernel_tracing": kernel_tracing = int(config_value[i])
    
        if  config_property[i] == "jump": jump = int(config_value[i])
        
        if  config_property[i] == "size_arcsec" :     
            size_arcsec_ = config_value[i].strip('][').split(',')
            for i in range(len(size_arcsec_)):
                size_arcsec.append(float(size_arcsec_[i]))    

        if  config_property[i] == "centre_deg" :
            centre_deg_ = config_value[i].strip('][').split(',')
            centre_deg = [float(centre_deg_[0]),float(centre_deg_[1])]

        if  config_property[i] == "delta_RA"  : delta_RA = float(config_value[i])
        if  config_property[i] == "delta_DEC" : delta_DEC = float(config_value[i])
         
        if  config_property[i] == "scale_cubes_using_integflux" : 
            if config_value[i] == "True" : 
                scale_cubes_using_integflux = True 
            else: scale_cubes_using_integflux = False 

        if  config_property[i] == "flux_ratios" :
            flux_ratios_ = config_value[i].strip('][').split(',')
            flux_ratios = []
            for i in range(len(flux_ratios_)):
                flux_ratios.append(float(flux_ratios_[i]))
                
        if  config_property[i] == "apply_scale" : 
            if config_value[i] == "True" : 
                apply_scale = True 
            else: apply_scale = False         


        if  config_property[i] == "trim_cube" : 
            if config_value[i] == "True" : 
                trim_cube = True 
            else: trim_cube = False 
            
        if  config_property[i] == "trim_values" :     
            trim_values_ = config_value[i].strip('][').split(',')
            trim_values = [int(trim_values_[0]),int(trim_values_[1]),int(trim_values_[2]),int(trim_values_[3])]
            
        if  config_property[i] == "remove_spaxels_not_fully_covered" : 
            if config_value[i] == "True" : 
                remove_spaxels_not_fully_covered = True 
            else: remove_spaxels_not_fully_covered = False 
             
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

    # Save rss list if requested:
    if save_rss : 
        for i in range(len(rss_list)):
            save_rss_list.append("auto")
 
    if len(cube_list_names) < 1 : cube_list_names=[""]   
 
    # Asign names to variables
    # If files are given, they have preference over variables!
    
    
    if telluric_correction_file != "":
        w_star,telluric_correction = read_table(telluric_correction_file, ["f", "f"] )        
    if telluric_correction_name != "" :  
        exec("telluric_correction="+telluric_correction_name)
    else:
        telluric_correction=[0]

 # Check that skyflat, flux and telluric lists are more than 1 element

    if len(skyflat_list) < 2 : skyflat_list=["","","","","","","","","",""]   # CHECK THIS
    

        
    if len(telluric_correction_list_) < 2 : 
        telluric_correction_list=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
    else:
        for i in range(len(telluric_correction_list_)):
            w_star,telluric_correction_ = read_table(telluric_correction_list_[i], ["f", "f"] ) 
            telluric_correction_list.append(telluric_correction_)
  
 # More checks

    if len(box_x) < 1 : box_x = [0,-1]
    if len(box_y) < 1 : box_y = [0,-1]
    
    if do_cubing == False:      
        fits_file = "" 
        save_aligned_cubes = False 
        make_combined_cube = False  
        do_alignment = False

    if throughput_2D_variable != "":
        exec("throughput_2D = "+throughput_2D_variable)
          
 # Print the summary of parameters read with this script
  
    print("> Parameters for processing this object :\n")
    print("  obj_name                 = ",obj_name) 
    print("  description              = ",description)
    print("  path                     = ",path)
    print("  Python_obj_name          = ",Python_obj_name)
    print("  date                     = ",date)
    print("  grating                  = ",grating)
    
    if read_cube == False:   
        for rss in range(len(rss_list)):       
            if len(rss_list) > 1:       
                if rss == 0 : 
                    print("  rss_list                 = [",rss_list[rss],",")
                else:
                    if rss == len(rss_list)-1:
                        print("                              ",rss_list[rss]," ]")
                    else:        
                        print("                              ",rss_list[rss],",")     
            else:
                print("  rss_list                 = [",rss_list[rss],"]")
                
        if rss_clean: 
            print("  rss_clean                = ",rss_clean)
            print("  plot_rss                 = ",plot_rss)
        else:    
            print("  apply_throughput         = ",apply_throughput)          
            if apply_throughput:
                if throughput_2D_variable != "" :    
                    print("    throughput_2D variable = ",throughput_2D_variable)
                else:
                    if throughput_2D_file != "" : 
                        print("    throughput_2D_file     = ",throughput_2D_file)
                    else:
                        print("    Requested but no throughput 2D information provided !!!") 
                if throughput_2D_wavecor:
                    print("    throughput_2D will be applied AFTER correcting CCD defects and fixing small wavelenghts")
    
            
            print("  correct_ccd_defects      = ",correct_ccd_defects) 
            if correct_ccd_defects: print("    kernel_correct_ccd_defects = ",kernel_correct_ccd_defects) 
            
            if fix_wavelengths:
                print("  fix_wavelengths          = ",fix_wavelengths)
                print("    sol                    = ",sol)
        
            print("  do_extinction            = ",do_extinction)
    
            if do_telluric_correction: 
                print("  do_telluric_correction   = ",do_telluric_correction)
            else: 
                if grating == "385R" or grating == "1000R" or grating == "2000R" : 
                    print("  do_telluric_correction   = ",do_telluric_correction)
      
            print("  sky_method               = ",sky_method)
            if sky_method == "1D" or sky_method == "1Dfit" or sky_method == "2D":    
                for sky in range(len(sky_list)):
                    if sky == 0 : 
                        print("    sky_list               = [",sky_list[sky],",")
                    else:
                        if sky == len(sky_list)-1:
                            print("                              ",sky_list[sky]," ]")
                        else:        
                            print("                              ",sky_list[sky],",")     
            if sky_spectrum[0] != -1  and sky_spectrum_name !="" : 
                print("    sky_spectrum_name      = ",sky_spectrum_name)
                if sky_method == "1Dfit" or sky_method == "selffit":
                    print("    ranges_with_emis_lines = ",ranges_with_emission_lines)
                    print("    cut_red_end            = ",cut_red_end)
                
            if sky_method == "1D" or sky_method == "1Dfit" : print("    scale_sky_1D           = ",scale_sky_1D)
            if sky_spectrum[0] == -1 and len(sky_list) == 0 : print_n_sky = True
            if sky_method == "self" or sky_method == "selffit": print_n_sky = True  
            if print_n_sky: 
                if len(sky_fibres) > 1: 
                    print("    sky_fibres             = ",sky_fibres_print)
                else:
                    print("    n_sky                  = ",n_sky)   
    
            if win_sky > 0 : print("    win_sky                = ",win_sky)
            if auto_scale_sky: print("    auto_scale_sky         = ",auto_scale_sky)
            if remove_5577: print("    remove 5577 skyline    = ",remove_5577)
            print("  correct_negative_sky     = ",correct_negative_sky)
            if correct_negative_sky:
                print("    order_fit_negative_sky   = ",order_fit_negative_sky)
                print("    kernel_negative_sky      = ",kernel_negative_sky)
                print("    use_fit_for_negative_sky = ",use_fit_for_negative_sky) 
                print("    low_fibres               = ",low_fibres)
                print("    individual_check         = ",individual_check)  
                if sky_method in ["self" , "selffit"]:  print("    force_sky_fibres_to_zero = ",force_sky_fibres_to_zero)
                                 
            if sky_method == "1Dfit" or sky_method == "selffit" or id_el == True:
                if sky_lines_file != "": print("    sky_lines_file         = ",sky_lines_file)
                print("    brightest_line         = ",brightest_line)
                print("    brightest_line_wav     = ",brightest_line_wavelength)
          
            
            if  id_el == True:    # NEED TO BE CHECKED
                print("  id_el                = ",id_el)
                print("    high_fibres            = ",high_fibres)
                print("    cut                    = ",cut)
                print("    broad                  = ",broad)
                print("    id_list                = ",id_list)
                print("    plot_id_el             = ",plot_id_el)
                
            if fix_edges: print("  fix_edges                = ",fix_edges)          
            print("  clean_sky_residuals      = ",clean_sky_residuals)
            if clean_sky_residuals:
                if len(features_to_fix) > 0:
                    for feature in features_to_fix:
                        print("    feature_to_fix         = ",feature)
                else:
                    print("    No list with features_to_fix provided, using default list")
                print("    sky_fibres_residuals   = ",sky_fibres_for_residuals_print)    
                       
            print("  clean_cosmics            = ",clean_cosmics)
            if clean_cosmics:
                print("    width_bl                        = ",width_bl)
                print("    kernel_median_cosmics           = ",kernel_median_cosmics)
                print("    cosmic_higher_than              = ",cosmic_higher_than)
                print("    extra_factor                    = ",extra_factor)
                print("    max_number_of_cosmics_per_fibre = ",max_number_of_cosmics_per_fibre)
 
            print("  clean_extreme_negatives  = ",clean_extreme_negatives)
            if clean_extreme_negatives:
                print("    percentile_min         = ",percentile_min)    
            if remove_negative_median_values:
                print("  Negative pixels will be set to 0 when median value of spectrum is negative")
        
        if do_cubing:
            print(" ")
            print("  pixel_size               = ",pixel_size)
            print("  kernel_size              = ",kernel_size) 

            if len(size_arcsec) > 0:  print("  cube_size_arcsec         = ",size_arcsec)
            if len(centre_deg) > 0 :  print("  centre_deg               = ",centre_deg)

            if len(offsets) > 0 :
                print("  offsets                  = ",offsets)
            else:
                print("  offsets will be calculated automatically")
            print("  reference_rss            = ",reference_rss) 

            if half_size_for_centroid > 0 : print("  half_size_for_centroid   = ",half_size_for_centroid)
            if np.nanmedian(box_x+box_y) != -0.5: print("  box_x, box_y             = ", box_x, box_y)
            print("  adr_index_fit            = ",adr_index_fit)
            print("  2D Gauss for tracing     = ",g2d)
            print("  step_tracing             = ",step_tracing)
            print("  adr_clip_fit             = ",adr_clip_fit)
            if kernel_tracing > 0 : print("  kernel_tracing           = ",kernel_tracing)
            
            if len(plot_tracing_maps) > 0 : 
                print("  plot_tracing_maps        = ",plot_tracing_maps)

            if edgelow != -1: print("  edgelow for tracing      = ",edgelow)
            if edgehigh != -1:print("  edgehigh for tracing     = ",edgehigh)
            
            
            print("  ADR                      = ",ADR)
            print("  ADR in combined cube     = ",ADR_cc)
            print("  force_ADR                = ",force_ADR)
            if jump != -1 : print("  jump for ADR             = ",jump)        


            
            if len(ADR_x_fit_list)  >  0:
                print("  Fitting solution for correcting ADR provided!")
                for i in range(len(rss_list)):
                    print("                           = ",ADR_x_fit_list[i])
                    print("                           = ",ADR_y_fit_list[i])
            else:
                if ADR: print("    adr_index_fit          = ",adr_index_fit)
            
            if delta_RA+delta_DEC != 0:
                print("  delta_RA                 = ",delta_RA)
                print("  delta_DEC                = ",delta_DEC)
            
            if valid_wave_min > 0 : print("  valid_wave_min           = ",valid_wave_min)
            if valid_wave_max > 0 : print("  valid_wave_max           = ",valid_wave_max)
            if trim_cube: 
                print("  Trim cube                = ",trim_cube)
                print("     remove_spaxels_not_fully_covered = ",remove_spaxels_not_fully_covered)

        else:
            print("\n  No cubing will be done!\n")
            
        if do_cubing:
            print(" ")
            if flux_calibration_name == "": 
                if len(flux_calibration_file_list) != 0:
                    if len(flux_calibration_file_list) == 1:
                        print("  Using the same flux calibration for all files:")
                    else:
                        print("  Each rss file has a file with the flux calibration:")
                    for i in range(len(flux_calibration_file_list)):                   
                        print("    flux_calibration_file  = ",flux_calibration_file_list[i])
                else:    
                    if flux_calibration_file != "" : 
                        print("    flux_calibration_file  = ",flux_calibration_file)
                    else:
                        print("  No flux calibration will be applied")
            else:
                print("  Variable with the flux calibration :",flux_calibration_name)
        
        if do_telluric_correction:
            if telluric_correction_name == "":
                if np.nanmedian(telluric_correction_list) != 0:
                    print("  Each rss file has a telluric correction file:")
                    for i in range(len(telluric_correction_list)):                   
                        print("  telluric_correction_file = ",telluric_correction_list_[i])
                else:    
                    print("  telluric_correction_file = ",telluric_correction_file)        
            else:
                print("  Variable with the telluric calibration :",telluric_correction_name)
    
    else:
        print ("\n  List of ALIGNED cubes provided!")
        for cube in range(len(cube_list_names)):       
            if cube == 0 : 
                print("  cube_list                = [",cube_list_names[cube],",")
            else:
                if cube == len(cube_list_names)-1:
                    print("                              ",cube_list_names[cube]," ]")
                else:        
                    print("                              ",cube_list_names[cube],",")     
        
        print("  pixel_size               = ",pixel_size)
        print("  kernel_size              = ",kernel_size)  
        if half_size_for_centroid > 0 : print("  half_size_for_centroid   = ",half_size_for_centroid)
        if np.nanmedian(box_x+box_y) != -0.5: print("  box_x, box_y             = ", box_x, box_y)
        if jump != -1 : print("  jump for ADR             = ",jump)        
        if edgelow != -1: print("  edgelow for tracing      = ",edgelow)
        if edgehigh != -1:print("  edgehigh for tracing     = ",edgehigh)
        print("  ADR in combined cube     = ",ADR_cc)          
        if valid_wave_min > 0 : print("  valid_wave_min           = ",valid_wave_min)
        if valid_wave_max > 0 : print("  valid_wave_max           = ",valid_wave_max)
        if trim_cube: print("  Trim cube                = ",trim_cube)
        make_combined_cube = True

    if make_combined_cube:
        if scale_cubes_using_integflux:
            if len(flux_ratios) == 0 :
                print("  Scaling individual cubes using integrated flux of common region")
            else:
                print("  Scaling individual cubes using flux_ratios = ",flux_ratios)


    print("  plot                     = ",plot)
    if do_cubing or make_combined_cube:    
        if plot_weight: print("  plot weights             = ",plot_weight)
    if fig_size != 12. : print("  fig_size                 = ",fig_size)
    print("  warnings                 = ",warnings)
    if verbose == False: print("  verbose                  = ",verbose)    


    print("\n> Output files:\n")
    if fits_file != "" : print("  fits file with combined cube  =  ",fits_file)
    
    if read_cube == False:
        if len(save_rss_list) > 0 and rss_clean == False:
            for rss in range(len(save_rss_list)):
                if save_rss_list[0] != "auto":
                    if rss == 0 : 
                        print("  list of saved rss files       = [",save_rss_list[rss],",")
                    else:
                        if rss == len(save_rss_list)-1:
                            print("                                   ",save_rss_list[rss]," ]")
                        else:        
                            print("                                   ",save_rss_list[rss],",")      
                else: print("  Processed rss files will be saved using automatic naming")
                    
        else:
            save_rss_list = ["","","","","","","","","",""]
        if save_aligned_cubes:
            print("  Individual cubes will be saved as fits files")

     # Last little checks...
        if len(sky_list) == 0: sky_list=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
        if fix_wavelengths == False : sol =[0,0,0]
        if len (ADR_x_fit_list) == 0 :
            for i in range(len(rss_list)): 
                ADR_x_fit_list.append([0])
                ADR_y_fit_list.append([0])
            
      # Values for improving alignment:   # TODO: CHECK THIS!!! 
        if delta_RA+delta_DEC != 0:          
            for i in range(len(ADR_x_fit_list)):
                ADR_x_fit_list[i][2] = ADR_x_fit_list[i][2] + delta_RA
                ADR_y_fit_list[i][2] = ADR_y_fit_list[i][2] + delta_DEC
                
     # Now run KOALA_reduce      
        hikids = KOALA_reduce(rss_list,  
                              obj_name=obj_name,  description=description, 
                              fits_file=fits_file,  
                              rss_clean=rss_clean,
                              save_rss_to_fits_file_list=save_rss_list,
                              save_aligned_cubes = save_aligned_cubes,
                              cube_list_names = cube_list_names, 
                              apply_throughput=apply_throughput, 
                              throughput_2D = throughput_2D, 
                              throughput_2D_file = throughput_2D_file,
                              throughput_2D_wavecor = throughput_2D_wavecor,
                              #skyflat_list = skyflat_list,
                              correct_ccd_defects = correct_ccd_defects, 
                              kernel_correct_ccd_defects=kernel_correct_ccd_defects,
                              fix_wavelengths = fix_wavelengths, 
                              sol = sol,
                              do_extinction=do_extinction,                                          
                              
                              telluric_correction = telluric_correction,
                              telluric_correction_list = telluric_correction_list,
                              telluric_correction_file = telluric_correction_file,
                              
                              sky_method=sky_method,
                              sky_list=sky_list, 
                              n_sky = n_sky,
                              sky_fibres=sky_fibres,
                              win_sky = win_sky,
                              scale_sky_1D = scale_sky_1D, 
                              sky_lines_file=sky_lines_file,
                              ranges_with_emission_lines=ranges_with_emission_lines,
                              cut_red_end =cut_red_end,
                              remove_5577=remove_5577,
                              auto_scale_sky = auto_scale_sky,  
                              correct_negative_sky = correct_negative_sky, 
                              order_fit_negative_sky =order_fit_negative_sky, 
                              kernel_negative_sky = kernel_negative_sky,
                              individual_check = individual_check, 
                              use_fit_for_negative_sky = use_fit_for_negative_sky,
                              force_sky_fibres_to_zero = force_sky_fibres_to_zero,
                              high_fibres = high_fibres, 
                              low_fibres=low_fibres,
                              
                              brightest_line=brightest_line, 
                              brightest_line_wavelength = brightest_line_wavelength,
                              id_el = id_el,
                              id_list=id_list,
                              cut = cut, broad = broad, plot_id_el = plot_id_el,                  
                              
                              clean_sky_residuals = clean_sky_residuals,
                              features_to_fix = features_to_fix,
                              sky_fibres_for_residuals = sky_fibres_for_residuals, 
                              
                              fix_edges=fix_edges,           
                              clean_extreme_negatives=clean_extreme_negatives, percentile_min=percentile_min,
                              remove_negative_median_values=remove_negative_median_values,
                              clean_cosmics = clean_cosmics,
                              width_bl = width_bl, kernel_median_cosmics = kernel_median_cosmics, 
                              cosmic_higher_than = cosmic_higher_than, extra_factor = extra_factor, 
                              max_number_of_cosmics_per_fibre = max_number_of_cosmics_per_fibre,
                                                            
                              do_cubing= do_cubing,
                              do_alignment = do_alignment,
                              make_combined_cube = make_combined_cube,  
                              
                              pixel_size_arcsec=pixel_size, 
                              kernel_size_arcsec=kernel_size,
                              offsets = offsets,    # EAST+/WEST-  NORTH-/SOUTH+
                              reference_rss = reference_rss,
                                       
                              ADR=ADR, ADR_cc=ADR_cc, force_ADR=force_ADR,
                              jump = jump,
                              ADR_x_fit_list = ADR_x_fit_list, ADR_y_fit_list = ADR_y_fit_list,
                              
                              half_size_for_centroid = half_size_for_centroid,
                              box_x=box_x, box_y=box_y, 
                              adr_index_fit = adr_index_fit,
                              g2d = g2d,
                              kernel_tracing = kernel_tracing,
                              adr_clip_fit = adr_clip_fit,
                              plot_tracing_maps = plot_tracing_maps,
                              step_tracing=step_tracing,
                              edgelow=edgelow, edgehigh = edgehigh, 
                                                  
                              flux_calibration_file = flux_calibration_file_list,     # this can be a single file (string) or a list of files (list of strings)
                              flux_calibration = flux_calibration,                      # an array
                              flux_calibration_list  = flux_calibration_list,         # a list of arrays
                              
                              trim_cube = trim_cube,
                              trim_values = trim_values,
                              size_arcsec=size_arcsec,
                              centre_deg = centre_deg,
                              scale_cubes_using_integflux = scale_cubes_using_integflux,
                              apply_scale = apply_scale,
                              flux_ratios = flux_ratios,
                              remove_spaxels_not_fully_covered = remove_spaxels_not_fully_covered,
                              #cube_list_names =cube_list_names,
                              
                              valid_wave_min = valid_wave_min, valid_wave_max = valid_wave_max,
                              fig_size = fig_size,
                              log = log, gamma = gamma,
                              plot=plot, plot_rss = plot_rss, plot_weight=plot_weight, plot_spectra =plot_spectra,
                              warnings=warnings, verbose=verbose) 

    else:
        #print("else")
        hikids = build_combined_cube(cube_list, obj_name=obj_name, description=description,
                                     fits_file = fits_file, path=path,
                                     pk = pk,
                                     offsets = offsets,    # EAST+/WEST-  NORTH-/SOUTH+
                                     reference_rss = reference_rss,
                                     
                                     ADR=ADR, ADR_cc = ADR_cc, force_ADR=force_ADR,
                                     jump = jump, 
                                     ADR_x_fit_list=ADR_x_fit_list, ADR_y_fit_list=ADR_y_fit_list,
                                     
                                     half_size_for_centroid = half_size_for_centroid,
                                     box_x=box_x, box_y=box_y,  
                                     adr_index_fit=adr_index_fit, 
                                     adr_clip_fit = adr_clip_fit,
                                     g2d=g2d,
                                     kernel_tracing = kernel_tracing,
                                     plot_tracing_maps = plot_tracing_maps,
                                     step_tracing = step_tracing, 
                                     edgelow=edgelow, edgehigh=edgehigh,
                                     
                                     trim_cube = trim_cube,  
                                     trim_values =trim_values, 
                                     size_arcsec=size_arcsec,
                                     centre_deg = centre_deg,
                                     scale_cubes_using_integflux= scale_cubes_using_integflux, 
                                     apply_scale = apply_scale,
                                     flux_ratios = flux_ratios, 
                                     remove_spaxels_not_fully_covered = remove_spaxels_not_fully_covered,
                                     
                                     fig_size = fig_size,
                                     log = log, gamma = gamma,
                                     plot=plot, plot_weight= plot_weight, plot_spectra=plot_spectra,
                                     
                                     warnings=warnings, verbose=verbose, 
                                     say_making_combined_cube= False)                             
 
    if Python_obj_name != 0: exec(Python_obj_name+"=copy.deepcopy(hikids)", globals())

    
    print("> automatic_KOALA_reduce script completed !!!")
    print("\n  Python object created :",Python_obj_name)
    if fits_file != "" : print("  Fits file created     :",fits_file)
    
    return hikids