from __future__ import print_function
from builtins import str
from koala import *
import os.path as pth


path_main = pth.join(pth.dirname(__file__), "data")


if __name__ == "__main__":

    print("\n> Testing KOALA RSS class. Running", version)
    print("\n\n\n> ANGEL is having a lot of FUN with GitHub!")


    

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # TESTING PyKOALA in GitHub  - Taylah, Sarah, James, Sam, Blake, Ángel
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    #  Data are 10 Mar 2018  RED
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    date = "20180310"
    grating = "385R"
    pixel_size = 0.6  # Just 0.1 precision
    kernel_size = 1.25
    pk = (
        "_"
        + str(int(pixel_size))
        + "p"
        + str(int((abs(pixel_size) - abs(int(pixel_size))) * 10))
        + "_"
        + str(int(kernel_size))
        + "k"
        + str(int((abs(kernel_size) - abs(int(kernel_size))) * 100))
    )

    # # ---------------------------------------------------------------------------
    # # THROUGHPUT CORRECTION USING SKYFLAT
    # # ---------------------------------------------------------------------------
    # #
    # # The very first thing that we need is to get the throughput correction.
    # # IMPORTANT: We use a skyflat that has not been divided by a flatfield in 2dFdr !!!!!!
 
    path_skyflat = path_main+"/" \
                             ""+grating+"/"
    file_skyflatr=path_skyflat+"10mar2_combined.fits"                                  # FILE NOT DIVIDED BY THE FLAT
    throughput_file_red=path_skyflat+date+"_"+grating+"_throughput_correction.dat"
    #
    ## #
    ## # If this has been done before, we can read the file containing the throughput correction
    ##    throughput_red = read_table(throughput_file_red, ["f"] )
    ## #
    ## # Now we read the RSS file, we ONLY correct for ccd defects and high cosmics
    skyflat_red = KOALA_RSS(file_skyflatr, flat="", apply_throughput=False, sky_method="none",                 #skyflat = skyflat_red,
                            do_extinction=False, correct_ccd_defects = False,
                            correct_high_cosmics = False, clip_high = 100, step_ccd = 50, plot=False)
    ## #
    ## # Next we find the relative throughput.
    ## # If the data have been normalized by the FLATFIELD, we only need a SCALE between fibres,
    ## # We consider the median value in the range [wave_min_scale, wave_max_scale] for all fibres and scale
    ## #
    skyflat_red.find_relative_throughput(ymin=0, ymax=800000,  wave_min_scale=6300, wave_max_scale=6500, plot=False)  #
    # #
    # # The relative throughput is an array stored in skyflat_red.relative_throughput
    # # We save that array in a text file that we can read in the future without the need of repeating this
    array_to_text_file(skyflat_red.relative_throughput, filename= throughput_file_red )
    # #
    # #
    # # ---------------------------------------------------------------------------
    # # OBTAIN ABSOLUTE FLUX CALIBRATION AND TELLURIC CORRECTION USING CALIBRATION STARS
    # # ---------------------------------------------------------------------------
    # #
    # #
    # # If these have been obtained already, we can read files containing arrays with the calibrations
    # # Uncomment the next two sections and skip the rest till "OBTAIN SKY SPECTRA"
    #
    # # READ FLUX CALIBRATION RED
    flux_cal_file=path_main+"/flux_calibration_20180310_385R_0p6_1k8.dat"
    w_star,flux_calibration = read_table(flux_cal_file, ["f", "f"] )
    print(flux_calibration)
    #
    # # READ TELLURIC CORRECTION FROM FILE
    telluric_correction_file=path_main+"/telluric_correction_20180310_385R_0p6_1k25.dat"
    w_star,telluric_correction = read_table(telluric_correction_file, ["f", "f"] )
    print(telluric_correction)

    # # READ STAR 1
    # # First we provide names, paths, files...
    #    star1="H600"
    #    path_star1 = path_main+date+"/"+grating+"/"
    #    starpos1r = path_star1+"10mar20082red.fits"
    #    starpos2r = path_star1+"10mar20083red.fits"
    #    starpos3r = path_star1+"10mar20084red.fits"
    #    fits_file_red = path_star1+star1+"_"+grating+pk
    #    response_file_red = path_star1+star1+"_"+grating+pk+"_response.dat"
    #    telluric_file = path_star1+star1+"_"+grating+pk+"_telluric_correction.dat"
    # #
    # # -------------------   IF USING ONLY 1 FILE FOR CALIBRATION STAR -----------
    # #
    # # Read RSS file and apply throughput correction, substract sky using n_sky=400 lowest intensity fibres,
    # # correct for CCD defects and high cosmics
    #
    #    star1r = KOALA_RSS(starpos3r,
    #                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
    #                       sky_method="self", n_sky=400,
    #                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
    #                       valid_wave_min = 6085, valid_wave_max = 9305,
    #                       plot=True, warnings=True)
    #
    # # Now we search for the telluric correction
    # # For this we stack n_fibres=15 highest intensity fibres, derive a continuum in steps of step=15 A
    # # excluding problematic wavelenght ranges (exclude_wlm), normalize flux by continuum & derive flux/flux_normalized
    # # This is done in the range [wave_min, wave_max], but only correct telluric in [correct_from, correct_to]
    # # Including apply_tc=True will apply correction to the data (do it when happy with results, as need this for flux calibration)
    # #
    #    telluric_correction_star1 = star1r.get_telluric_correction(n_fibres=15, correct_from=6830., correct_to=8380.,
    #                                                               exclude_wlm=[[6000,6350],[6460,6720],[6830,7450], [7550,7750],[8050,8400]],
    #                                                               apply_tc=True,
    #                                                               combined_cube = False,  weight_fit_median = 1.,
    #                                                               step = 15, wave_min=6085, wave_max=9305)
    # #
    # # Next we CREATE THE CUBE for this star, using THE SAME PARAMETERS we will later using for our objects
    # # 0.6 is the pixel size, 1.25 is the kernel size.
    # #
    #    cubes1r=Interpolated_cube(star1r, pixel_size, kernel_size, plot=True, ADR=True) #, force_ADR = True)     # CASCA con lo de Matt
    # #
    # #
    # # -------------------   IF USING AT LEAST 2 FILES FOR CALIBRATION STAR -----------
    # #
    # # Run KOALA_reduce and get a combined datacube with given pixel_size and kernel_size
    # #
    #    rss_list = [starpos1r,starpos2r,starpos3r]
    #    H600r=KOALA_reduce(rss_list, fits_file=fits_file_red+".fits", obj_name=star1,  description=star1,
    #                           apply_throughput=True, skyflat = skyflat_red,
    #                           correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
    #                           sky_method="self", n_sky=400,
    #                           pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
    #                           ADR= False,
    #                           valid_wave_min = 6085, valid_wave_max = 9305,
    #                           plot= True, warnings=False )
    # #
    # # Extract the integrated spectrum of the star & save it
    # #
    #    H600r.combined_cube.half_light_spectrum(r_max=5, plot=True)
    #    spectrum_to_text_file(H600r.combined_cube.wavelength,H600r.combined_cube.integrated_star_flux, filename=fits_file_red+"_integrated_star_flux.dat")
    # #
    # # Find telluric correction CAREFUL WITH apply_tc=True
    # #
    #    telluric_correction_star1 = H600r.get_telluric_correction(n_fibres=15, correct_from=6830., correct_to=8400.,
    #                                                               exclude_wlm=[[6000,6350],[6460,6720],[6830,7450], [7550,7750],[8050,8400]],
    #                                                               apply_tc=True,
    #                                                               combined_cube = True,  weight_fit_median = 1.,
    #                                                               step = 15, wave_min=6085, wave_max=9305)
    # # We can save this calibration as a text file
    #    spectrum_to_text_file(H600r.combined_cube.wavelength,telluric_correction_star1, filename=telluric_file)
    # #
    # # -------------------   FLUX CALIBRATION (COMMON) -----------
    # #
    # # Now we read the absolute flux calibration data of the calibration star and get the response curve
    # # (Response curve: correspondence between counts and physical values)
    # # Include exp_time of the calibration star, as the results are given per second
    # # For this BE CAREFUL WITH ABSORPTIONS (Halpha) and check behaviour in the edges of the CCD
    # # Change fit_degree (3,5,7), step, min_wave, max_wave to get better fits !!!
    #
    #    H600r.combined_cube.do_response_curve('FLUX_CAL/fhilt600_edited.dat', plot=True, min_wave=6110., max_wave=9305.,
    #                              step=20, exp_time=120., fit_degree=7)

    # # Now we can save this calibration as a text file
    #    spectrum_to_text_file(H600r.combined_cube.response_wavelength,H600r.combined_cube.response_curve, filename=response_file_red)

    # # STAR 2
    #    star="HD60753"
    #    path_star = path_main+date+"/"+grating+"/"
    #    starpos1r = path_star+"10mar20079red.fits"
    #    starpos2r = path_star+"10mar20080red.fits"
    #    starpos3r = path_star+"10mar20081red.fits"
    #    fits_file_red = path_star+star+"_"+grating+pk
    #    response_file_red = path_star+star+"_"+grating+pk+"_response.dat"
    #    telluric_file = path_star+star+"_"+grating+pk+"_telluric_correction.dat"
    # #
    #    rss_list = [starpos1r,starpos2r,starpos3r]
    #    HD60753r=KOALA_reduce(rss_list, fits_file=fits_file_red+".fits", obj_name=star,  description=star,
    #                           apply_throughput=True, skyflat = skyflat_red,
    #                           correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
    #                           sky_method="self", n_sky=400,
    #                           pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
    #                           ADR= False,
    #                           valid_wave_min = 6085, valid_wave_max = 9305,
    #                           plot= True, warnings=False )
    # #
    #    HD60753r.combined_cube.half_light_spectrum(r_max=5, plot=True)
    #    spectrum_to_text_file(HD60753r.combined_cube.wavelength,HD60753r.combined_cube.integrated_star_flux, filename=fits_file_red+"_integrated_star_flux.dat")
    # #
    #    telluric_correction_star2 = HD60753r.get_telluric_correction(apply_tc=True,  combined_cube = True,
    #                                                                weight_fit_median = 1., step = 15, wave_min=6085, wave_max=9305,
    #                                                                correct_from=6830., correct_to=8400.,
    #                                                                exclude_wlm=[[6000,6330],[6460,6720],[6830,7450], [7550,7750],[8050,8400]])
    # #
    #    spectrum_to_text_file(HD60753r.combined_cube.wavelength,telluric_correction_star2, filename=telluric_file)
    # #
    #    HD60753r.combined_cube.do_response_curve('FLUX_CAL/fhd60753.dat', plot=True, min_wave=6110., max_wave=9305.,
    #                              step=20, exp_time=15., fit_degree=5)
    #
    #    spectrum_to_text_file(HD60753r.combined_cube.response_wavelength,HD60753r.combined_cube.response_curve, filename=response_file_red)

    # # STAR 3
    #    star="HR3454"
    #    path_star = path_main+date+"/"+grating+"/"
    #    starpos1r = path_star+"10mar20094red.fits"
    #    starpos2r = path_star+"10mar20095red.fits"
    #    starpos3r = path_star+"10mar20096red.fits"
    #    fits_file_red = path_star+star+"_"+grating+pk
    #    response_file_red = path_star+star+"_"+grating+pk+"_response.dat"
    #    telluric_file = path_star+star+"_"+grating+pk+"_telluric_correction.dat"
    # #
    #    rss_list = [starpos1r,starpos2r,starpos3r]
    #    HR3454r=KOALA_reduce(rss_list, fits_file=fits_file_red+".fits", obj_name=star,  description=star,
    #                           apply_throughput=True, skyflat = skyflat_red,
    #                           correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
    #                           sky_method="self", n_sky=400,
    #                           pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
    #                           ADR= False,
    #                           valid_wave_min = 6085, valid_wave_max = 9305,
    #                           plot= True, warnings=False )
    # #
    #    HR3454r.combined_cube.half_light_spectrum(r_max=5, plot=True)
    #    spectrum_to_text_file(HR3454r.combined_cube.wavelength,HR3454r.combined_cube.integrated_star_flux, filename=fits_file_red+"_integrated_star_flux.dat")
    # #
    #    telluric_correction_star3 = HR3454r.get_telluric_correction(apply_tc=True,  combined_cube = True,
    #                                                                weight_fit_median = 1., step = 15, wave_min=6085, wave_max=9305,
    #                                                                correct_from=6830., correct_to=8420.,
    #                                                                exclude_wlm=[[6000,6330],[6460,6720],[6830,7450], [7550,7750],[8050,8420]])
    # #
    #    spectrum_to_text_file(HR3454r.combined_cube.wavelength,telluric_correction_star3, filename=telluric_file)
    # #
    #    HR3454r.combined_cube.do_response_curve('FLUX_CAL/fhr3454_edited.dat', plot=True, min_wave=6110., max_wave=9305.,
    #                              step=20, exp_time=2., fit_degree=7)
    #
    #    spectrum_to_text_file(HR3454r.combined_cube.response_wavelength,HR3454r.combined_cube.response_curve, filename=response_file_red)



    # STAR 4
    #    star="EG274"
    #    path_star = path_main+date+"/"+grating+"/"
    #    starpos1r = path_star+"10mar20104red.fits"
    #    starpos2r = path_star+"10mar20105red.fits"
    #    starpos3r = path_star+"10mar20106red.fits"
    #    fits_file_red = path_star+star+"_"+grating+pk
    #    response_file_red = path_star+star+"_"+grating+pk+"_response.dat"
    #    telluric_file = path_star+star+"_"+grating+pk+"_telluric_correction.dat"
    #
    #
    #    star3r = KOALA_RSS(starpos3r,
    #                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
    #                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 50, step_ccd=50,
    #                       fix_wavelengths = True, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07],
    #                       sky_method="self", n_sky=50, correct_negative_sky = True,
    #                       telluric_correction = telluric_correction_20180310,
    #                       valid_wave_min = 6085, valid_wave_max = 9305,
    #                       plot=True, warnings=True)
    #
    #    cubes3r=Interpolated_cube(star3r, pixel_size, kernel_size, plot=True) #, force_ADR = True)
    #
    #    cubes3r.do_response_curve('FLUX_CAL/feg274_edited.dat', plot=True, min_wave=6100., max_wave=9305.,
    #                              step=25, exp_time=180., fit_degree=7, ha_width=150)

    #    rss_list = [starpos1r,starpos2r,starpos3r]
    #    EG274r=KOALA_reduce(rss_list, fits_file=fits_file_red+".fits", obj_name=star,  description=star,
    #                           apply_throughput=True, skyflat = skyflat_red,
    #                           correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 50, step_ccd=50,
    #                           fix_wavelengths = True, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07],
    #                           sky_method="self", n_sky=50, correct_negative_sky = True,
    #                           pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
    #                           ADR= False,
    #                           valid_wave_min = 6085, valid_wave_max = 9305,
    #                           plot= True, warnings=False )
    #
    #    EG274r.combined_cube.half_light_spectrum(r_max=5, plot=True)
    ###    spectrum_to_text_file(EG274r.combined_cube.wavelength,EG274r.combined_cube.integrated_star_flux, filename=fits_file_red+"_integrated_star_flux.dat")
    #
    #    telluric_correction_star4 = EG274r.get_telluric_correction(apply_tc=True,  combined_cube = True,
    #                                                                weight_fit_median = 1., step = 15, wave_min=6085, wave_max=9305,
    #                                                                correct_from=6830., correct_to=8420.,
    #                                                                exclude_wlm=[[6000,6330],[6460,6720],[6830,7450], [7550,7750],[8050,8420]])
    # #
    #    spectrum_to_text_file(EG274r.combined_cube.wavelength,telluric_correction_star4, filename=telluric_file)
    # #
    #    EG274r.combined_cube.do_response_curve('FLUX_CAL/feg274_edited.dat', plot=True, min_wave=6080., max_wave=9305.,    # FIX BLUE END !!!
    #                              step=10, exp_time=180., fit_degree=7, ha_width=150)
    # #
    #    spectrum_to_text_file(EG274r.combined_cube.response_wavelength,EG274r.combined_cube.response_curve, filename=response_file_red)

    # #  CHECK AND GET THE FLUX CALIBRATION FOR THE NIGHT  RED

    # #  First we take another look to the RSS data ploting the integrated fibre values in a map
    #    star1r.RSS_map(star1r.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))  # Dead fibre!!!
    #    star2r.RSS_map(star2r.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))
    #    star3r.RSS_map(star3r.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))

    # # We check again that star1 is on a dead fibre, we don't use this star for absolute flux calibration

    # # Define in "stars" the 2 cubes we are using, and plotting their responses to check
    #    stars=[H600r.combined_cube,HD60753r.combined_cube,HR3454r.combined_cube,EG274r.combined_cube]
    #    plot_response(stars, scale=[1,1.14,1.48,1])

    #    stars=[EG274r.combined_cube] #H600r.combined_cube,EG274r.combined_cube]
    #    plot_response(stars, scale=[1,1])

    # # The shape of the curves are ~OK but they have a variation of ~5% in flux...
    # # Probably this would have been corrected obtaining at least TWO exposures per star..
    # # We obtain the flux calibration applying:
    #    flux_calibration_20180310_385R_0p6_1k25 = obtain_flux_calibration(stars)

    # # And we save this absolute flux calibration as a text file
    #    flux_calibration_file = path_main+date+"/flux_calibration_"+date+"_"+grating+pk+".dat"
    #    spectrum_to_text_file(H600r.combined_cube.wavelength,flux_calibration_20180310_385R_0p6_1k8, filename=flux_calibration_file)

    # #   CHECK AND GET THE TELLURIC CORRECTION

    # # Similarly, provide a list with the telluric corrections and apply:
    #    telluric_correction_list=[telluric_correction_star1,telluric_correction_star2,telluric_correction_star3,telluric_correction_star4]
    #    telluric_correction_list=[telluric_correction_star4]  # [telluric_correction_star1,]
    #    telluric_correction_20180310 = obtain_telluric_correction(EG274r.combined_cube.wavelength, telluric_correction_list)

    # # Save this telluric correction to a file
    #    telluric_correction_file = path_main+date+"/telluric_correction_"+date+"_"+grating+pk+".dat"
    #    spectrum_to_text_file(EG274r.combined_cube.wavelength,telluric_correction_20180310, filename=telluric_correction_file )

    # #
    # #
    # #
    # # ---------------------------------------------------------------------------
    # #  OBTAIN SKY SPECTRA IF NEEDED
    # # ---------------------------------------------------------------------------
    # #
    # #
    # #
    # #
    # # Using the same files than objects but chosing fibres without object emission
    #
    file1r=path_main+"/"+grating+"/10mar20091red.fits"
    file2r=path_main+"/"+grating+"/10mar20092red.fits"
    file3r=path_main+"/"+grating+"/10mar20093red.fits"

#    sky_r1 = KOALA_RSS(file1r, apply_throughput=True, skyflat = skyflat_red, do_extinction=False,
#                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd = 50,
#                       sky_method="none", is_sky=True, win_sky=151,
#                       plot=False)
#
#    sky1=sky_r1.plot_combined_spectrum(list_spectra=[870,871,872,873,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,900], median=True)
#
#    sky_r2 = KOALA_RSS(file2r, apply_throughput=True, skyflat = skyflat_red, do_extinction=False,
#                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd = 50,
#                       sky_method="none", is_sky=True, win_sky=151,
#                       plot=False)
#
#    sky2=sky_r2.plot_combined_spectrum(list_spectra=[870,871,872,873,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,900], median=True)
#
    sky_r3 = KOALA_RSS(file3r, apply_throughput=True, skyflat = skyflat_red, do_extinction=False,
                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd = 50,
                       sky_method="none", is_sky=True, win_sky=151,
                       plot=False)

    sky3=sky_r3.plot_combined_spectrum(list_spectra=[870,871,872,873,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,900], median=True)

# #
# #
# # ---------------------------------------------------------------------------
# # TIME FOR THE OBJECT !!
# # ---------------------------------------------------------------------------
# #

   
   
    OBJECT = "POX4"
    DESCRIPTION = "POX4 CUBE"   
    file1r=path_main+"/"+grating+"/10mar20091red.fits"
    file2r=path_main+"/"+grating+"/10mar20092red.fits"
    file3r=path_main+"/"+grating+"/10mar20093red.fits"
   
     
#    rss3_all = KOALA_RSS(file3r, #save_rss_to_fits_file=path_main+"Tol30Ar3_rss_tcwsreu.fits", 
#                         apply_throughput=True, skyflat = skyflat_red,
#                         correct_ccd_defects = True,
#                         fix_wavelengths = True, sol = [0.119694453613, -0.000707644207572, 2.03806478671e-07],
#                         #sky_method="none",
#                         sky_method="1D", sky_spectrum=sky3, auto_scale_sky = True,
#                         id_el=False, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, brightest_line_wavelength =6641., #fibre=422, #422
#                         id_list=[6300.30, 6312.1, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 8750.47, 8862.79, 9014.91, 9069.0],
#                         clean_sky_residuals = False, dclip=3.0, extra_w = 1.3, step_csr = 25, fibre = 0,
#                         telluric_correction = telluric_correction,
#                         do_extinction=False, correct_negative_sky = True,
#                         plot=False, warnings=True)
#   
#      Fitting a second-order polynomy a0x +  a1x * fibre + a2x * fibre**2:  
#      a0x = 0.119694453613    a1x = -0.000707644207572      a2x = 2.03806478671e-07
   
   
   
#    cube_test=Interpolated_cube(rss3_all, pixel_size, kernel_size, flux_calibration=flux_calibration, plot=False)   
#    save_fits_file(cube_test, path_main+"/"+grating+"/POX4_d_cube_test.fits", ADR=False)


    rss_list=[file2r,file3r] #,file3r]  #,file4r,file5r,file6r,file7r]
#    sky_list=[sky1,sky2,sky3]

    sky_list=[sky3,sky3]

    fits_file_red=path_main+"/"+grating+"/POX4_A_red_combined_cube_2_TEST_GitHub.fits"



    hikids_red = KOALA_reduce(rss_list,  obj_name=OBJECT,  description=DESCRIPTION,
                          #rss_clean=True,
                          fits_file=fits_file_red,  #save_rss_to_fits_file_list=save_rss_list,
                          apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
                          correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd=50,
                          #fix_wavelengths = True, sol = [0.119694453613, -0.000707644207572, 2.03806478671e-07],
                          #sky_method="1Dfit", sky_list=sky_list, scale_sky_1D = 1., auto_scale_sky = True,
                          sky_method="1D", sky_list=sky_list, scale_sky_1D = 1., auto_scale_sky = True,
                          brightest_line="Ha", brightest_line_wavelength = 6641.,
                          id_el=False, high_fibres=10, cut=1.5, plot_id_el=True, broad=1.8, 
                          id_list=[6300.30, 6312.1, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 8750.47, 8862.79, 9014.91, 9069.0],
                          #clean_sky_residuals = False, dclip=3.0, extra_w = 1.3, step_csr = 25,
                          telluric_correction = telluric_correction,
                          do_extinction=True, correct_negative_sky = False,
                                 
                          pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
                          #offsets=[-0.54, -0.87, 1.58, -1.26] # EAST-/WEST+  NORTH-/SOUTH+
                                  
                          ADR=False,
                          flux_calibration=flux_calibration,
                          #size_arcsec=[60,60],
                         
                          valid_wave_min = 6085, valid_wave_max = 9305,
                          plot=False, warnings=False)  
end = timer()
print("\n> Elapsing time = ", end - start, "s")
# -----------------------------------------------------------------------------
#                                     ... Paranoy@ Rulz! ;^D  & Angel R. :-)
# -----------------------------------------------------------------------------
