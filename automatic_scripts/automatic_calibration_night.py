def automatic_calibration_night(CALIBRATION_NIGHT_FILE="",
                                date="",
                                grating="",
                                pixel_size=0,
                                kernel_size=0,
                                path="",
                                file_skyflat="",
                                throughput_2D_file="",
                                throughput_2D = 0,
                                skyflat=0,
                                do_skyflat = True, 
                                kernel_throughput =0,
                                correct_ccd_defects=True,
                                fix_wavelengths = False,
                                sol=[0],
                                rss_star_file_for_sol = "",
                                plot=True,
                                CONFIG_FILE_path="",
                                CONFIG_FILE_list=[],
                                star_list=[],
                                abs_flux_scale=[],
                                flux_calibration_file="",
                                telluric_correction_file ="",
                                objects_auto=[],
                                auto = False,
                                rss_clean = False,
                                flux_calibration_name="flux_calibration_auto",
                                cal_from_calibrated_starcubes = False,
                                disable_stars=[],                      # stars in this list will not be used
                                skyflat_names=[]
                                ):
    """
    Use: 
        CALIBRATION_NIGHT_FILE = "./configuration_files/calibration_night.config"
        automatic_calibration_night(CALIBRATION_NIGHT_FILE)
    """
    
    if len(skyflat_names) == 0: skyflat_names=["SKYFLAT", "skyflat", "Skyflat", "SkyFlat", "SKYFlat", "SkyFLAT"]
    
    w=[]
    telluric_correction_list=[]  
    global skyflat_variable
    skyflat_variable = ""
    global skyflat_
    global throughput_2D_variable
    global flux_calibration_night
    global telluric_correction_night
    throughput_2D_variable = "" 
    global throughput_2D_
    throughput_2D_ = [0]
    
    if flux_calibration_file == "":flux_calibration_file=path+"flux_calibration_file_auto.dat"
    if telluric_correction_file == "":telluric_correction_file=path+"telluric_correction_file_auto.dat"
     
    
    check_nothing_done = 0

    print("\n===================================================================================")
    
    if auto:
        print("\n    COMPLETELY AUTOMATIC CALIBRATION OF THE NIGHT ")
        print("\n===================================================================================")
    
    if len(CALIBRATION_NIGHT_FILE) > 0:
        config_property, config_value = read_table(CALIBRATION_NIGHT_FILE, ["s", "s"] )    
        print("\n> Reading configuration file ", CALIBRATION_NIGHT_FILE)
        print("  for performing the automatic calibration of the night...\n")
    else:    
        print("\n> Using the values given in automatic_calibration_night()")
        print("  for performing the automatic calibration of the night...\n")
        config_property = []
        config_value    = []
        lista_propiedades = ["path", "file_skyflat", "rss_star_file_for_sol", "flux_calibration_file", "telluric_correction_file"]
        lista_valores     = [path, file_skyflat, rss_star_file_for_sol, flux_calibration_file, telluric_correction_file]
        for i in range(len(lista_propiedades)):
            if len(lista_valores[i]) > 0:
                config_property.append(lista_propiedades[i])
                config_value.append(lista_valores[i])       
        if pixel_size == 0:
            print ("  - No pixel size provided, considering pixel_size = 0.7")
            pixel_size = 0.7
        if kernel_size == 0:
            print ("  - No kernel size provided, considering kernel_size = 1.1")
            kernel_size = 1.1
        pk = "_"+str(int(pixel_size))+"p"+str(int((abs(pixel_size)-abs(int(pixel_size)))*10))+"_"+str(int(kernel_size))+"k"+str(int(abs(kernel_size*100))-int(kernel_size)*100)
        if sol[0] != 0 : fix_wavelengths = True
        if len(CONFIG_FILE_path) > 0:
            for i in range(len(CONFIG_FILE_list)):
                CONFIG_FILE_list[i] = full_path (CONFIG_FILE_list[i],CONFIG_FILE_path)                

        

#   Completely automatic reading folder:
    
    if auto:
        fix_wavelengths = True
        list_of_objetos,list_of_files, list_of_exptimes, date,grating=list_fits_files_in_folder(path, return_list=True)
        print(" ")
        
        list_of_files_of_stars=[]
        for i in range(len(list_of_objetos)):
            if list_of_objetos[i] in skyflat_names:
                file_skyflat=list_of_files[i][0]
                print ("  - SKYFLAT automatically identified")
                
            if list_of_objetos[i] in ["H600", "HILT600", "Hilt600", "Hiltner600", "HILTNER600"]  and list_of_objetos[i] not in disable_stars:
                print ("  - Calibration star Hilt600 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("Hilt600_"+grating)                 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("Hilt600", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("Hilt600")
                CONFIG_FILE_list.append("")
 
            if list_of_objetos[i] in ["EG274", "Eg274", "eg274", "eG274", "E274", "e274"]  and list_of_objetos[i] not in disable_stars:
                print ("  - Calibration star EG274 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("EG274_"+grating) 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("EG274", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("EG274")
                CONFIG_FILE_list.append("")
 
            if list_of_objetos[i] in ["HD60753", "hd60753", "Hd60753", "HD60753FLUX"]  and list_of_objetos[i] not in disable_stars:
                print ("  - Calibration star HD60753 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("HD60753_"+grating) 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("HD60753", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("HD60753")
                CONFIG_FILE_list.append("")

            if list_of_objetos[i] in ["HD49798", "hd49798", "Hd49798"]  and list_of_objetos[i] not in disable_stars:
                print ("  - Calibration star HD49798 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("HD49798_"+grating) 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("HD49798", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("HD49798")
                CONFIG_FILE_list.append("")

            if list_of_objetos[i] in ["cd32d9927", "CD32d9927", "CD32D9927", "CD-32d9927", "cd-32d9927", "Cd-32d9927", "CD-32D9927", "cd-32D9927", "Cd-32D9927"  ]  and list_of_objetos[i] not in disable_stars:
                print ("  - Calibration star CD32d9927 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("CD32d9927_"+grating) 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("CD32d9927", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("CD32d9927")
                CONFIG_FILE_list.append("")

            if list_of_objetos[i] in ["HR3454", "Hr3454", "hr3454"] and list_of_objetos[i] not in disable_stars:
                print ("  - Calibration star HR3454 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("HR3454_"+grating) 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("HR3454", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("HR3454")
                CONFIG_FILE_list.append("")
                
            if list_of_objetos[i] in [ "HR718" ,"Hr718" , "hr718", "HR718FLUX","HR718auto" ,"Hr718auto" , "hr718auto", "HR718FLUXauto"  ]  and list_of_objetos[i] not in disable_stars: 
                print ("  - Calibration star HR718 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("HR718_"+grating) 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("HR718", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("HR718")
                CONFIG_FILE_list.append("")


        if throughput_2D_file != "":
            throughput_2D_file = full_path(throughput_2D_file,path) 
            do_skyflat = False
            print ("  - throughput_2D_file provided, no need of processing skyflat")          
            sol=[0,0,0]
            ftf = fits.open(throughput_2D_file)
            if ftf[0].data[0][0] == 1. :           
                sol[0] = ftf[0].header["SOL0"]
                sol[1] = ftf[0].header["SOL1"]
                sol[2] = ftf[0].header["SOL2"]
                print ("  - solution for fixing small wavelength shifts included in this file :\n    sol = ",sol)  
        print(" ")
        
    else:
        list_of_files_of_stars=[[],[],[],[],[],[]]


    for i in range(len(config_property)):
        if  config_property[i] == "date" : 	 date = config_value[i]
        if  config_property[i] == "grating" : 	 grating = config_value[i]        
        if  config_property[i] == "pixel_size" : 	 pixel_size = float(config_value[i])         
        if  config_property[i] == "kernel_size" : 	 
            kernel_size = float(config_value[i])
            pk = "_"+str(int(pixel_size))+"p"+str(int((abs(pixel_size)-abs(int(pixel_size)))*10))+"_"+str(int(kernel_size))+"k"+str(int(abs(kernel_size*100))-int(kernel_size)*100)
        if  config_property[i] == "path" : 	
            path = config_value[i]
            if path[-1] != "/" : path = path+"/"
            throughput_2D_file = path+"throughput_2D_"+date+"_"+grating+".fits"
            flux_calibration_file = path+"flux_calibration_"+date+"_"+grating+pk+".dat" 
            if flux_calibration_name =="flux_calibration_auto" : flux_calibration_name = "flux_calibration_"+date+"_"+grating+pk 
            if grating == "385R" or grating == "1000R" :
                telluric_correction_file = path+"telluric_correction_"+date+"_"+grating+".dat" 
                telluric_correction_name = "telluric_correction_"+date+"_"+grating  
                
        if  config_property[i] == "file_skyflat" : file_skyflat = full_path(config_value[i],path)
                
        if  config_property[i] == "skyflat" : 
            exec("global "+config_value[i])
            skyflat_variable = config_value[i]

        if  config_property[i] == "do_skyflat" : 
            if config_value[i] == "True" : 
                do_skyflat = True 
            else: do_skyflat = False 

        if  config_property[i] == "correct_ccd_defects" : 
            if config_value[i] == "True" : 
                correct_ccd_defects = True 
            else: correct_ccd_defects = False 

        if  config_property[i] == "fix_wavelengths":
            if config_value[i] == "True" : fix_wavelengths = True 
        if  config_property[i] == "sol" :
            fix_wavelengths = True
            sol_ = config_value[i].strip('][').split(',')
            if float(sol_[0]) == -1:
                sol = [float(sol_[0])]
            else:
                if float(sol_[0]) != -0: sol = [float(sol_[0]),float(sol_[1]),float(sol_[2])]

        if  config_property[i] == "kernel_throughput" : 	 kernel_throughput = int(config_value[i])  

        if  config_property[i] == "rss_star_file_for_sol": rss_star_file_for_sol = full_path(config_value[i],path)

        if  config_property[i] == "throughput_2D_file" : throughput_2D_file = full_path(config_value[i],path) 
        if  config_property[i] == "throughput_2D" : throughput_2D_variable = config_value[i]  
        if  config_property[i] == "flux_calibration_file" : 	 flux_calibration_file  = full_path(config_value[i],path)      
        if  config_property[i] == "telluric_correction_file" : 	 telluric_correction_file  = full_path(config_value[i],path)      

        if  config_property[i] == "CONFIG_FILE_path" : CONFIG_FILE_path = config_value[i]       
        if  config_property[i] == "CONFIG_FILE" : CONFIG_FILE_list.append(full_path(config_value[i],CONFIG_FILE_path))
              
        if  config_property[i] == "abs_flux_scale":
            abs_flux_scale_ = config_value[i].strip('][').split(',')
            for j in range(len(abs_flux_scale_)):
                abs_flux_scale.append(float(abs_flux_scale_[j]))
            
        if  config_property[i] == "plot" : 
            if config_value[i] == "True" : 
                plot = True 
            else: plot = False 

        if  config_property[i] == "cal_from_calibrated_starcubes" and  config_value[i] == "True" : cal_from_calibrated_starcubes=True
              
        if  config_property[i] == "object" : 
            objects_auto.append(config_value[i])
  
    if len(abs_flux_scale) == 0:
        for i in range(len(CONFIG_FILE_list)): abs_flux_scale.append(1.)


# Print the summary of parameters

    print("> Parameters for automatically processing the calibrations of the night:\n")
    print("  date                       = ",date)
    print("  grating                    = ",grating)
    print("  path                       = ",path)
    if cal_from_calibrated_starcubes == False:
        if do_skyflat:     
            print("  file_skyflat               = ",file_skyflat)
            if skyflat_variable != "" : print("  Python object with skyflat = ",skyflat_variable)
            print("  correct_ccd_defects        = ",correct_ccd_defects)            
            if fix_wavelengths:
                print("  fix_wavelengths            = ",fix_wavelengths)     
                if sol[0] != 0 and sol[0] != -1:
                    print("    sol                      = ",sol)
                else:
                    if rss_star_file_for_sol =="" :
                        print("    ---> However, no solution given! Setting fix_wavelength = False !")
                        fix_wavelengths = False
                    else:
                        print("    Star RSS file for getting small wavelength solution:",rss_star_file_for_sol)
        else:
            print("  throughput_2D_file         = ",throughput_2D_file)
            if throughput_2D_variable != "" : print("  throughput_2D variable     = ",throughput_2D_variable)
    
        print("  pixel_size                 = ",pixel_size)
        print("  kernel_size                = ",kernel_size)
    
        if CONFIG_FILE_list[0] != "":
    
            for config_file in range(len(CONFIG_FILE_list)):
                if config_file == 0 : 
                    if len(CONFIG_FILE_list) > 1:       
                        print("  CONFIG_FILE_LIST           =  [",CONFIG_FILE_list[config_file],",")
                    else:
                        print("  CONFIG_FILE_LIST           =  [",CONFIG_FILE_list[config_file],"]")
                else:
                    if config_file == len(CONFIG_FILE_list)-1:
                        print("                                 ",CONFIG_FILE_list[config_file]," ]")
                    else:        
                        print("                                 ",CONFIG_FILE_list[config_file],",")           

    else:
        print("\n> The calibration of the night will be obtained using these fully calibrated starcubes:\n")

    if len(objects_auto) != 0 :
        pprint = ""
        for i in range(len(objects_auto)):
            pprint=pprint+objects_auto[i]+ "  " 
        print("  Using stars in objects     = ",pprint)

    if len(abs_flux_scale) > 0 : print("  abs_flux_scale             = ",abs_flux_scale)
    print("  plot                       = ",plot)
    
    print("\n> Output files:\n")
    if do_skyflat:
        if throughput_2D_variable != "" : print("  throughput_2D variable     = ",throughput_2D_variable)   
        print("  throughput_2D_file         = ",throughput_2D_file)
    print("  flux_calibration_file      = ",flux_calibration_file)
    if grating in red_gratings:
        print("  telluric_correction_file   = ",telluric_correction_file)

    print("\n===================================================================================")
               
    if do_skyflat:      
        if rss_star_file_for_sol != "" and sol[0] == 0 :
            print("\n> Getting the small wavelength solution, sol, using star RSS file")
            print(" ",rss_star_file_for_sol,"...")                                  
            if grating in red_gratings :
                _rss_star_ = KOALA_RSS(rss_star_file_for_sol, 
                                       correct_ccd_defects = False, 
                                       fix_wavelengths=True, sol = [0],
                                       plot= plot)
            if grating in ["580V"] :
                _rss_star_ = KOALA_RSS(rss_star_file_for_sol, 
                                       correct_ccd_defects = True, remove_5577 = True,
                                       plot= plot)               
            sol = _rss_star_.sol
            print("\n> Solution for the small wavelength variations:")
            print(" ",sol)
        
        throughput_2D_, skyflat_ =  get_throughput_2D(file_skyflat, plot = plot, also_return_skyflat = True,
                                            correct_ccd_defects = correct_ccd_defects,
                                            fix_wavelengths = fix_wavelengths, sol = sol,
                                            throughput_2D_file =throughput_2D_file, kernel_throughput = kernel_throughput)      
        
        if throughput_2D_variable != "":
            print("  Saving throughput 2D into Python variable:", throughput_2D_variable)
            exec(throughput_2D_variable+"=throughput_2D_", globals())

        if skyflat_variable != "":
            print("  Saving skyflat into Python variable:", skyflat_variable)
            exec(skyflat_variable+"=skyflat_", globals())

    else:
        if cal_from_calibrated_starcubes == False: print("\n> Skyflat will not be processed! Throughput 2D calibration already provided.\n")
        check_nothing_done = check_nothing_done + 1

    good_CONFIG_FILE_list =[]
    good_star_names =[]
    stars=[]
    if cal_from_calibrated_starcubes == False: 
        for i in range(len(CONFIG_FILE_list)):
      
            run_star = True
            
            if CONFIG_FILE_list[i] != "":            
                try:
                    config_property, config_value = read_table(CONFIG_FILE_list[i], ["s", "s"] )
                    if len(CONFIG_FILE_list) != len(objects_auto)  :               
                        for j in range (len(config_property)):
                            if config_property[j] == "obj_name" : running_star = config_value[j] 
                        if i < len(objects_auto) :
                            objects_auto[i] = running_star
                        else:    
                            objects_auto.append(running_star)                   
                    else:
                        running_star = objects_auto[i]
                except Exception:
                    print("===================================================================================")
                    print("\n> ERROR! config file {} not found!".format(CONFIG_FILE_list[i]))
                    run_star = False
            else:
                running_star = star_list[i]
                if i < len(objects_auto) :
                    objects_auto[i] = running_star
                else:    
                    objects_auto.append(running_star)   
                

            if run_star:
                pepe=0
                if pepe == 0:
                #try:
                    print("===================================================================================")        
                    print("\n> Running automatically calibration star",running_star, "in CONFIG_FILE:")
                    print(" ",CONFIG_FILE_list[i],"\n")
                    psol="["+np.str(sol[0])+","+np.str(sol[1])+","+np.str(sol[2])+"]"
                    exec('run_automatic_star(CONFIG_FILE_list[i], object_auto="'+running_star+'", star=star_list[i], sol ='+psol+', throughput_2D_file = "'+throughput_2D_file+'", rss_list = list_of_files_of_stars[i], path_star=path, date=date,grating=grating,pixel_size=pixel_size,kernel_size=kernel_size, rss_clean=rss_clean)')
                    print("\n> Running automatically calibration star in CONFIG_FILE")
                    print("  ",CONFIG_FILE_list[i]," SUCCESSFUL !!\n")
                    good_CONFIG_FILE_list.append(CONFIG_FILE_list[i])
                    good_star_names.append(running_star)
                    try: # This is for a combined cube
                        exec("stars.append("+running_star+".combined_cube)")      
                        if grating in red_gratings:
                            exec("telluric_correction_list.append("+running_star+".combined_cube.telluric_correction)")
                    except Exception: # This is when we read a cube from fits file
                        exec("stars.append("+running_star+")")      
                        if grating in red_gratings:
                            exec("telluric_correction_list.append("+running_star+".telluric_correction)")                                
                # except Exception:   
                #     print("===================================================================================")
                #     print("\n> ERROR! something wrong happened running config file {} !\n".format(CONFIG_FILE_list[i]))

    else:       # This is for the case that we have individual star cubes ALREADY calibrated in flux
        pprint = ""
        stars=[]
        good_star_names=[]
        for i in range(len(objects_auto)):
            pprint=pprint+objects_auto[i]+ "  " 
            try: # This is for a combined cube
                exec("stars.append("+objects_auto[i]+".combined_cube)")
                if grating in red_gratings:
                    exec("telluric_correction_list.append("+objects_auto[i]+".combined_cube.telluric_correction)")
            except Exception: # This is when we read a cube from fits file
                exec("stars.append("+objects_auto[i]+")") 
                if grating in red_gratings:
                    exec("telluric_correction_list.append("+objects_auto[i]+".telluric_correction)")  
            good_star_names.append(stars[i].object)
                
        print("\n> Fully calibrated star cubes provided :",pprint) 
        good_CONFIG_FILE_list = pprint

            
 # CHECK AND GET THE FLUX CALIBRATION FOR THE NIGHT

    if len(good_CONFIG_FILE_list) > 0:        
        # Define in "stars" the cubes we are using, and plotting their responses to check  
        plot_response(stars, scale=abs_flux_scale)

        # We obtain the flux calibration applying:    
        flux_calibration_night = obtain_flux_calibration(stars)
        exec(flux_calibration_name + '= flux_calibration_night', globals())
        print("  Flux calibration saved in variable:", flux_calibration_name)
    
        # And we save this absolute flux calibration as a text file
        w= stars[0].wavelength
        spectrum_to_text_file(w, flux_calibration_night, filename=flux_calibration_file)

        # Similarly, provide a list with the telluric corrections and apply:            
        if grating in red_gratings:
            telluric_correction_night = obtain_telluric_correction(w,telluric_correction_list, label_stars=good_star_names, scale=abs_flux_scale)            
            exec(telluric_correction_name + '= telluric_correction_night', globals())
            print("  Telluric calibration saved in variable:", telluric_correction_name)
    
 # Save this telluric correction to a file
            spectrum_to_text_file(w, telluric_correction_night, filename=telluric_correction_file)

    else:
        print("\n> No configuration files for stars available !")
        check_nothing_done = check_nothing_done + 1

 # Print Summary
    print("\n===================================================================================")   
    if len(CALIBRATION_NIGHT_FILE) > 0:
        print("\n> SUMMARY of running configuration file", CALIBRATION_NIGHT_FILE,":\n") 
    else:
        print("\n> SUMMARY of running automatic_calibration_night() :\n") 


    if len(objects_auto) > 0 and cal_from_calibrated_starcubes == False:    
        pprint = ""
        for i in range(len(objects_auto)):
            pprint=pprint+objects_auto[i]+ "  " 
        print("  Created objects for calibration stars           :",pprint) 
    
        if len(CONFIG_FILE_list) > 0:    
            print("  Variable with the flux calibration              :",flux_calibration_name)
            if grating in red_gratings:
                print("  Variable with the telluric calibration          :",telluric_correction_name)
                print(" ")
        print("  throughput_2D_file        = ",throughput_2D_file)
        if throughput_2D_variable != "" : print("  throughput_2D variable    = ",throughput_2D_variable)
    
        if sol[0] != -1 and sol[0] != 0:
            print("  The throughput_2D information HAS BEEN corrected for small wavelength variations:")
            print("  sol                       =  ["+np.str(sol[0])+","+np.str(sol[1])+","+np.str(sol[2])+"]")
    
        if skyflat_variable != "" : print("  Python object created with skyflat = ",skyflat_variable)
        
        if len(CONFIG_FILE_list) > 0:
            print('  flux_calibration_file     = "'+flux_calibration_file+'"')
            if grating in red_gratings:
                print('  telluric_correction_file  = "'+telluric_correction_file+'"')
 
    if cal_from_calibrated_starcubes:
        print("  Variable with the flux calibration              :",flux_calibration_name)
        if grating in red_gratings:
                print("  Variable with the telluric calibration          :",telluric_correction_name)
                print(" ")       
        print('  flux_calibration_file     = "'+flux_calibration_file+'"')
        if grating in red_gratings:
            print('  telluric_correction_file  = "'+telluric_correction_file+'"')
        
 
    if check_nothing_done == 2:
        print("\n> NOTHING DONE!")
              
    print("\n===================================================================================")   