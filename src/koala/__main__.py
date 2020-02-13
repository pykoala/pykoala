from koala import *
import os.path as pth


path_main = pth.join(pth.dirname(__file__), "data")


if __name__ == "__main__":

    print "\n> Testing KOALA RSS class. Running", version

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    # THINGS STILL TO DO:

    # - Align n-cubes (it does 10 so far, easy to increase if needed)
    # - Check flux calibration (OK)

    # - CHECK wavelengths cuts are applied to integrated fibre
    # - INCLUDE remove bad wavelength ranges when computing response in calibration stars (Not needed anymore?)

    # Stage 2:
    #
    # - Use data from DIFFERENT nights (check self.wavelength)
    # - Combine 1000R + 385R data
    # - Mosaiquing (it basically works)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # Reducing Hi-KIDS data in AAOMC104IT - 10 Sep 2019
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    #    offset_positions(10, 10, 10.1, 20, 20, 10.1, 10, 10, 10.1, 20, 20, 15.1)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    #  Data 10 Mar 2018  RED
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    #date = "20180310"
    #grating = "385R"
    #pixel_size = 0.6  # Just 0.1 precision
    #kernel_size = 1.25
    #pk = (
    #    "_"
    #    + str(int(pixel_size))
    #    + "p"
    #    + str(int((abs(pixel_size) - abs(int(pixel_size))) * 10))
    #    + "_"
    #    + str(int(kernel_size))
    #    + "k"
    #    + str(int((abs(kernel_size) - abs(int(kernel_size))) * 100))
    #)

    # # ---------------------------------------------------------------------------
    # # THROUGHPUT CORRECTION USING SKYFLAT
    # # ---------------------------------------------------------------------------
    # #
    # # The very first thing that we need is to get the throughput correction.
    # # IMPORTANT: We use a skyflat that has not been divided by a flatfield in 2dFdr !!!!!!
    # #
    # # We may also need a normalized flatfield:
    #    path_flat = path_main
    #    file_flatr=path_flat+"FLAT/24oct20067red.fits"
    #    flat_red = KOALA_RSS(file_flatr, sky_method="none", do_extinction=False, apply_throughput=False, plot=True)
    #                        # correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 100, step_ccd = 50)
    # #
    # # Provide path, skyflat file and name of the file that will keep the throughput correction

    #    path_skyflat = path_main+date+"/"+grating+"/skyflats_normalized/"
    #    file_skyflatr=path_skyflat+"10mar20065red.fits"                                  # FILE  DIVIDED BY THE FLAT, DON'T USE THIS
    #    throughput_file_red=path_skyflat+date+"_"+grating+"_throughput_correction.dat"

    #    path_skyflat = path_main+date+"/"+grating+"/skyflats_non_normalized/"
    #    file_skyflatr=path_skyflat+"10mar2_combined.fits"                                  # FILE NOT DIVIDED BY THE FLAT
    #    throughput_file_red=path_skyflat+date+"_"+grating+"_throughput_correction.dat"
    #
    ## #
    ## # If this has been done before, we can read the file containing the throughput correction
    ##    throughput_red = read_table(throughput_file_red, ["f"] )
    ## #
    ## # Now we read the RSS file, we ONLY correct for ccd defects and high cosmics
    #    skyflat_red = KOALA_RSS(file_skyflatr, flat="", apply_throughput=False, sky_method="none",                 #skyflat = skyflat_red,
    #                             do_extinction=False, correct_ccd_defects = False,
    #                             correct_high_cosmics = False, clip_high = 100, step_ccd = 50, plot=True)
    ## #
    ## # Next we find the relative throughput.
    ## # If the data have been normalized by the FLATFIELD, we only need a SCALE between fibres,
    ## # We consider the median value in the range [wave_min_scale, wave_max_scale] for all fibres and scale
    ## #
    #    skyflat_red.find_relative_throughput(ymin=0, ymax=800000,  wave_min_scale=6300, wave_max_scale=6500)  #
    # #
    # # The relative throughput is an array stored in skyflat_red.relative_throughput
    # # We save that array in a text file that we can read in the future without the need of repeating this
    #    array_to_text_file(skyflat_red.relative_throughput, filename= throughput_file_red )
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
    #    flux_cal_file=path_main+"flux_calibration_20161024_1000R_0p6_1k25.dat"
    #    w_star,flux_calibration_20161024_1000R_0p6_1k25 = read_table(flux_cal_file, ["f", "f"] )
    #    print flux_calibration_20161024_1000R_0p6_1k25
    #
    # # READ TELLURIC CORRECTION FROM FILE
    #    telluric_correction_file=path_main+date+"/telluric_correction_20180310_385R.dat"
    #    w_star,telluric_correction_20180310 = read_table(telluric_correction_file, ["f", "f"] )
    #    print telluric_correction_20180310

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

    ##### Star 4 bis

    #    starpos3r = path_star+"test_flat/10mar20106red.fits"
    #    star3r = KOALA_RSS(starpos3r,
    #                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
    #                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 50, step_ccd=50,
    #                       sky_method="self", n_sky=50,
    #                       telluric_correction = telluric_correction_20180310,
    #                       valid_wave_min = 6085, valid_wave_max = 9305,
    #                       plot=True, warnings=True)
    #
    #    cubes3r=Interpolated_cube(star3r, pixel_size, kernel_size, plot=True) #, force_ADR = True)
    #
    #    cubes3r.do_response_curve('FLUX_CAL/feg274_edited.dat', plot=True, min_wave=6100., max_wave=9300.,
    #                              step=50, exp_time=180., fit_degree=9, ha_width=150)

    #### POX 4 quick skyflat as flat

    #    pox4arf = path_star+"10mar20091red.fits"
    #    pox4arf = "10mar20091red.fits"

    #    pox4ar_test_ = KOALA_RSS(pox4arf,
    #                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
    #                       correct_ccd_defects = True, #correct_high_cosmics = False, clip_high = 50, step_ccd=50,
    #                       sky_method="1Dfit", sky_spectrum=sky1, brightest_line_wavelength = 6640.,
    #                       do_extinction=True,
    #                       #sky_method="self", n_sky=50,
    #                       #sky_method="1D", sky_spectrum=sky1, scale_sky_1D = 1.0,  #0.97,
    #                       #telluric_correction = telluric_correction_20180310,
    #                       valid_wave_min = 6085, valid_wave_max = 9305,
    #                       plot=True, warnings=True)

    #    w=pox4ar_test.wavelength
    #    s=pox4ar_test.intensity_corrected[500]
    #    plt.plot(w,s)
    #    plt.plot(w,pox4ar_test.intensity[500])
    #    #plt.xlim(6000,6500)
    #    plt.ylim(-10,1000)
    #    #ptitle = "Fibre "+np.str(fibre)#+" with rms = "+np.str(rms[i])
    #    #plt.title(ptitle)
    #    plt.xlabel("Wavelength [$\AA$]")
    #    plt.ylabel("Flux [counts]")
    #    #plt.legend(frameon=True, loc=2, ncol=4)
    #    plt.minorticks_on()
    #    plt.show()
    #    plt.close()
    #
    #
    #
    #    pox4ar_no = KOALA_RSS(pox4arf,
    #                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
    #                       correct_ccd_defects = True, #correct_high_cosmics = False, clip_high = 50, step_ccd=50,
    #                       sky_method="none", do_extinction=False,
    #                       #sky_method="self", n_sky=50,
    #                       #sky_method="1D", sky_spectrum=sky1, scale_sky_1D = 1.0,  #0.97,
    #                       #individual_sky_substraction = True,
    #                       #telluric_correction = telluric_correction_20180310,
    ##                       id_el=True, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, # fibre=422, #422
    ##                       id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39,7329.66, 9069.0],
    ##                       clean_sky_residuals = True, dclip=3.0, extra_w = 1.3, step_csr = 25,
    #
    #                       valid_wave_min = 6085, valid_wave_max = 9305,
    #                       plot=True, warnings=True)
    #
    #    cubes_pox4ar=Interpolated_cube(pox4ar, pixel_size, kernel_size, plot=True) #, force_ADR = True)
    #    save_fits_file(cubes_pox4ar, path_star+"/POX4_a_test_nskyflat.fits", ADR=False)

    #    stars=[cubes3r]
    #    plot_response(stars)
    #    flux_calibration_=obtain_flux_calibration(stars)

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
    #    file1r=path_main+date+"/"+grating+"/10mar20091red.fits"
    #    file2r=path_main+date+"/"+grating+"/10mar20092red.fits"
    #    file3r=path_main+date+"/"+grating+"/10mar20093red.fits"
    #
    #    file1r=pox4arf
    #    sky_r1 = KOALA_RSS(file1r, apply_throughput=True, skyflat = skyflat_red, do_extinction=False,
    #                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd = 50,
    #                       sky_method="none", is_sky=True, win_sky=151,
    #                       plot=True)
    #
    #    sky1=sky_r1.plot_combined_spectrum(list_spectra=[870,871,872,873,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,900], median=True)
    #
    #    sky_r2 = KOALA_RSS(file2r, apply_throughput=True, skyflat = skyflat_red, do_extinction=False,
    #                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd = 50,
    #                       sky_method="none", is_sky=True, win_sky=151,
    #                       plot=True)
    #
    #    sky2=sky_r2.plot_combined_spectrum(list_spectra=[870,871,872,873,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,900], median=True)

    #    sky_r3 = KOALA_RSS(file3r, apply_throughput=True, skyflat = skyflat_red, do_extinction=False,
    #                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd = 50,
    #                       sky_method="none", is_sky=True, win_sky=151,
    #                       plot=True)
    #
    #    sky3=sky_r3.plot_combined_spectrum(list_spectra=[870,871,872,873,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,900], median=True)

    # #
    # #
    # #
    # # ---------------------------------------------------------------------------
    # # TIME FOR THE OBJECT !!
    # # ---------------------------------------------------------------------------
    # #
    # #
    # #

    # # READ FLUX CALIBRATION RED
    #    flux_calibration_file = path_main+date+"/flux_calibration_"+date+"_"+grating+pk+".dat"
    #    w_star,flux_calibration_20180310_385R_0p6_1k8 = read_table(flux_calibration_file, ["f", "f"] )
    #
    # # READ TELLURIC CORRECTION FROM FILE
    #    telluric_correction_file = path_main+date+"/telluric_correction_"+date+"_"+grating+pk+".dat"
    #    w_star,telluric_correction_20180310 = read_table(telluric_correction_file, ["f", "f"] )
    #
    #
    #    OBJECT = "POX4"
    #    DESCRIPTION = "POX4 CUBE"
    #    file1r=path_main+date+"/"+grating+"/10mar20091red.fits"
    #    file2r=path_main+date+"/"+grating+"/10mar20092red.fits"
    #    file3r=path_main+date+"/"+grating+"/10mar20093red.fits"
    #
    #    test3r = KOALA_RSS(file3r,
    #                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
    #                       correct_ccd_defects = False, correct_high_cosmics = False, clip_high = 50, step_ccd=50,
    #                       sky_method="1D", sky_spectrum=sky3, scale_sky_1D = 0.97, #n_sky=50,
    #                       telluric_correction = telluric_correction_20180310,
    #                       id_el=False, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, # fibre=422, #422
    #                       id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39,7329.66, 9069.0],
    #                       clean_sky_residuals = False, dclip=3.0, extra_w = 1.3, step_csr = 25,
    #                       valid_wave_min = 6085, valid_wave_max = 9305,
    #                       plot=True, warnings=True)
    #
    #    cube_test=Interpolated_cube(test3r, .6, 1.8, flux_calibration=flux_calibration_20180310_385R_0p6_1k8, plot=True)
    #    save_fits_file(cube_test, path_main+date+"/"+grating+"/POX4_d_cube_test.fits", ADR=False)

    #    ds9_offsets(42.357288,58.920888,43.256992,57.46697,pixel_size_arc=pixel_size) a->b
    #    ds9_offsets(43.256992,57.46697,40.624623,55.374314,pixel_size_arc=pixel_size) b->d
    #    ds9_offsets(42.357288,58.920888,40.624623,55.374314,pixel_size_arc=pixel_size) a->d

    #

    #    test1r = KOALA_RSS(file3r,
    #                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
    #                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 50, step_ccd=50,
    #                       sky_method="1D", sky_spectrum=sky3, scale_sky_1D = 0.98, #0.95, #n_sky=50, individual_sky_substraction=True,
    #                       telluric_correction = telluric_correction_20180310,
    #                       id_el=True, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, #fibre = 723, # fibre=422, #422
    #                       id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39,7329.66, 9069.0],
    #                       clean_sky_residuals = True, dclip=3.0, extra_w = 1.3, step_csr = 25,
    #                       valid_wave_min = 6085, valid_wave_max = 9305, #valid_wave_min = 6100, valid_wave_max = 9300,
    #                       plot=True, warnings=True)
    #
    #

    #    rss_list=[file1r,file2r,file3r]  #,file4r,file5r,file6r,file7r]
    #    sky_list=[sky1,sky2,sky3]
    #    fits_file_red=path_main+date+"/"+grating+"/POX4_A_red_combined_cube_3.fits"
    #
    #    hikids_red = KOALA_reduce(rss_list, fits_file=fits_file_red, obj_name=OBJECT,  description=DESCRIPTION,
    #                          apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
    #                          correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd=50,
    #                          sky_method="1D", sky_list=sky_list,  scale_sky_1D =0.95,
    #                          #sky_method="self", n_sky = 10, #sky_list=sky_list,
    #                          telluric_correction = telluric_correction_20180310,
    #                          do_extinction=True,
    #                          id_el=True, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, # fibre=422, #422
    #                          id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39,7329.66, 9069.0],
    #                          clean_sky_residuals = True, dclip=3.0, extra_w = 1.3, step_csr = 25,
    #
    #                          pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
    #                          offsets=[-0.54, -0.87, 1.58, -1.26],  # EAST-/WEST+  NORTH-/SOUTH+  #ad[-1.04, -2,128], #bd[-1.5794, -1.256], #[0.53, -0.872, -1.5794, -1.256],#a->b[1.55, -0.17], # 0.62, -1.08, -1.88, -1.08],
    #                          ADR=False,
    #                          flux_calibration=flux_calibration_20180310_385R_0p6_1k8,
    #
    #                          valid_wave_min = 6085, valid_wave_max = 9305,
    #                          plot=True, warnings=False)

    ############

    #    OBJECT      = "UGCA153"
    #    DESCRIPTION = "UGCA153"
    #    file1r=path_main+date+"/"+grating+"/10mar20088red.fits"
    #    file2r=path_main+date+"/"+grating+"/10mar20089red.fits"
    #    file3r=path_main+date+"/"+grating+"/10mar20090red.fits"

    #    test3r = KOALA_RSS(file3r,
    #                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
    #                       correct_ccd_defects = False, correct_high_cosmics = False, clip_high = 50, step_ccd=50,
    #                       sky_method="1D", sky_spectrum=sky3, scale_sky_1D = 0.97, #n_sky=50,
    #                       telluric_correction = telluric_correction_20180310,
    #                       id_el=False, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, # fibre=422, #422
    #                       id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39,7329.66, 9069.0],
    #                       clean_sky_residuals = False, dclip=3.0, extra_w = 1.3, step_csr = 25,
    #                       valid_wave_min = 6085, valid_wave_max = 9305,
    #                       plot=True, warnings=True)
    #
    #    cube_test=Interpolated_cube(test3r, .6, 1.8, flux_calibration=flux_calibration_20180310_385R_0p6_1k8, plot=True)
    #    save_fits_file(cube_test, path_main+date+"/"+grating+"/UGCA153__cube_test.fits", ADR=False)

    #    ds9_offsets(42.357288,58.920888,43.256992,57.46697,pixel_size_arc=pixel_size) a->b
    #    ds9_offsets(43.256992,57.46697,40.624623,55.374314,pixel_size_arc=pixel_size) b->d
    #    ds9_offsets(42.357288,58.920888,40.624623,55.374314,pixel_size_arc=pixel_size) a->d

    #

    #    sky_r1 = KOALA_RSS(file1r, apply_throughput=True, skyflat = skyflat_red, do_extinction=False,
    #                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd = 50,
    #                       sky_method="none", is_sky=True, win_sky=151,
    #                       plot=True)

    #    list_spectra=[]
    #    for i in range(100):
    #        list_spectra.append(100+i)
    #    sky1=sky_r1.plot_combined_spectrum(list_spectra=list_spectra, median=True)

    #    sky_r3 = KOALA_RSS(file3r, apply_throughput=True, skyflat = skyflat_red, do_extinction=False,
    #                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd = 50,
    #                       sky_method="none", is_sky=True, win_sky=151,
    #                       plot=True)
    #
    #    list_spectra=[]
    #    for i in range(100):
    #        list_spectra.append(100+i)
    #    sky3=sky_r3.plot_combined_spectrum(list_spectra=list_spectra, median=True)

    #    test1r = KOALA_RSS(file1r,
    #                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
    #                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 50, step_ccd=50,
    #                       sky_method="1D", sky_spectrum=sky1, scale_sky_1D = 0.98, #0.95, #n_sky=50, individual_sky_substraction=True,
    #                       #sky_method="self", n_sky = 100,
    #                       telluric_correction = telluric_correction_20180310,
    #                       id_el=True, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, brightest_line_wavelength =6580., #fibre = 723, # fibre=422, #422
    #                       id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39,7329.66, 9069.0],
    #                       clean_sky_residuals = True, dclip=3.0, extra_w = 1.3, step_csr = 25,
    #                       valid_wave_min = 6085, valid_wave_max = 9305, #valid_wave_min = 6100, valid_wave_max = 9300,
    #                       plot=True, warnings=True)
    #
    #    cube_test=Interpolated_cube(test1r, .6, 1.8, flux_calibration=flux_calibration_, plot=True)
    #    save_fits_file(cube_test, path_main+date+"/"+grating+"/UGCA153_a_cube.fits", ADR=False)

    ### Halpha in 6580

    #    rss_list=[file1r,file2r,file3r]  #,file4r,file5r,file6r,file7r]
    #    sky_list=[sky1,sky2,sky3]
    #    fits_file_red=path_main+date+"/"+grating+"/UGCA153_red_combined_cube_3.fits"
    #####
    #    hikids_red = KOALA_reduce(rss_list, fits_file=fits_file_red, obj_name=OBJECT,  description=DESCRIPTION,
    #                          apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
    #                          correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd=50,
    #                          sky_method="1D", sky_list=sky_list,  scale_sky_1D =0.98,
    #                          #sky_method="self", n_sky = 100, #sky_list=sky_list,
    #                          telluric_correction = telluric_correction_20180310,
    #                          do_extinction=True,
    #                          id_el=True, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, brightest_line_wavelength =6580., # fibre=422, #422
    #                          id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39,7329.66, 9069.0],
    #                          clean_sky_residuals = True, dclip=3.0, extra_w = 1.3, step_csr = 25,
    #
    #                          pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
    #                          offsets=[1.23, -0.22, 0.38, 2.13],  # EAST-/WEST+  NORTH-/SOUTH+  #ad[-1.04, -2,128], #bd[-1.5794, -1.256], #[0.53, -0.872, -1.5794, -1.256],#a->b[1.55, -0.17], # 0.62, -1.08, -1.88, -1.08],
    #                          ADR=False,
    #                          flux_calibration=flux_calibration_,#20180310_385R_0p6_1k8,
    #
    #                          valid_wave_min = 6085, valid_wave_max = 9305,
    #                          plot=True, warnings=False)
    #

    #    w,f = read_table('/Users/alopez/Documents/DATA/GAUSS/2018_03_Run_04_GAUSS/20180310/385R//ds9.dat', ["f","f"])
    #    fig_size=19
    #    plt.figure(figsize=(fig_size, fig_size/3))
    #    plt.plot(w,f)
    #    plt.ylim(0,1.E-18)
    #    plt.xlim(6100,9000)
    #    plt.show()
    #    plt.close()

    ############

    #    OBJECT      = "Tol30A"
    #    DESCRIPTION = "Tol30A - 3 cubes"
    #file1r = path_main + date + "/" + grating + "/10mar20097red"
    #file2r = path_main + date + "/" + grating + "/10mar20098red"
    #file3r = path_main + date + "/" + grating + "/10mar20099red"

    #    OFFSETS = [-1.1559397974054588, -0.03824158143106171, -0.13488941515513764, 1.8904461213338577]

    #    OBJECT      = "Tol30B"
    #    DESCRIPTION = "Tol30B - 3 cubes"
    #file4r = path_main + date + "/" + grating + "/10mar20101red"
    #file5r = path_main + date + "/" + grating + "/10mar20102red"
    #file6r = path_main + date + "/" + grating + "/10mar20103red"

    # OFFSETS = [0.05949437860580531, 2.1424443691864097, 1.357097981084074, -0.22020665715518595]






###### Tring all together...

# # ---------------------------------------------------------------------------
# #  OBTAIN SKY SPECTRA
# # ---------------------------------------------------------------------------

#    sky1_2=sky_spectrum_from_fibres_using_file(file1r+".fits", n_sky = 25, apply_throughput=True, skyflat=skyflat_red, correct_ccd_defects= True,
#                                             fix_wavelengths = False, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07], plot=False)
#
#    sky2_2=sky_spectrum_from_fibres_using_file(file2r+".fits", n_sky = 25, apply_throughput=True, skyflat=skyflat_red, correct_ccd_defects= True,
#                                             fix_wavelengths = False, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07], plot=False)
#
#    sky3_2=sky_spectrum_from_fibres_using_file(file3r+".fits", n_sky = 25, apply_throughput=True, skyflat=skyflat_red, correct_ccd_defects= True,
#                                             fix_wavelengths = False, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07], plot=False)

#    rss_list=[file1r+".fits",file2r+".fits",file3r+".fits"]  #,file4r,file5r,file6r,file7r]
#    save_rss_list=[file1r+"_TCWSULA.fits",file2r+"_TCWSULA.fits",file3r+"_TCWSULA.fits"]  #,file4r,file5r,file6r,file7r]
#    sky_list=[sky1_2,sky2_2,sky3_2]
##    fits_file_red=path_main+date+"/"+grating+"/Tol30_A_red_combined_cube_3.fits"
#    fits_file_red=path_main+date+"/"+grating+"/Tol30_B_red_combined_cube_3.fits"

#  COMBINED

#    OBJECT      = "Tol30"
#    DESCRIPTION = "Tol30 - test"
##
##    rss_list=[file1r+"_TCWSULA.fits",file2r+"_TCWSULA.fits",file3r+"_TCWSULA.fits",  file4r+"_TCWSULA.fits",file5r+"_TCWSULA.fits",file6r+"_TCWSULA.fits"]
##    sky_list=[sky1,sky2,sky3,  sky1_2,sky2_2,sky3_2]
###    fits_file_red=path_main+date+"/"+grating+"/Tol30_A_red_combined_cube_3.fits"
#    fits_file_red=path_main+date+"/"+grating+"/Tol30_red_combined_cube_TEST.fits"
#
#
#    rss_list=[file3r+"_TCWSULA.fits",  file4r+"_TCWSULA.fits"]
#
#
#    hikids_red = KOALA_reduce(rss_list,  obj_name=OBJECT,  description=DESCRIPTION, rss_clean=True,
#                          fits_file=fits_file_red,  #save_rss_to_fits_file_list=save_rss_list,
##                          apply_throughput=True, skyflat = skyflat_red,
##                          correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd=50,
##                          #fix_wavelengths = True, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07],
##                          sky_method="1Dfit", sky_list=sky_list, scale_sky_1D = 1., auto_scale_sky = True,
##                          brightest_line="Ha", brightest_line_wavelength = 6613.,
##                          #id_el=True, high_fibres=10, cut=1.5, plot_id_el=True, broad=1.8,
##                          #id_list=[6300.30, 6312.1, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 8750.47, 8862.79, 9014.91, 9069.0],
##                          #clean_sky_residuals = False, dclip=3.0, extra_w = 1.3, step_csr = 25,
##                          telluric_correction = telluric_correction_20180310,
##                          do_extinction=True, correct_negative_sky = True,
#
#                          pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
#                          offsets=[-1.25, 12.5],
#                          #offsets=[1.25, 0., 0, 2.17],  # EAST-/WEST+  NORTH-/SOUTH+
#                          #offsets=[1.25, 0., 0, 2.17, 1.25, 12.5, 0, 2.17, -0.63, -1.08],  # EAST-/WEST+  NORTH-/SOUTH+
##                          offsets=[-1.1559397974054588, -0.03824158143106171, -0.13488941515513764, 1.8904461213338577,
##                                   -1.25, 12.5,
##                                   0.05949437860580531, 2.1424443691864097, 1.357097981084074, -0.22020665715518595],
#
#                          ADR=False,
##                          flux_calibration=flux_calibration_20180310_385R_0p6_1k25,
#                          #size_arcsec=[60,60],
#
#                          valid_wave_min = 6085, valid_wave_max = 9305,
#                          plot=True, warnings=False)
#
#    rss_test=KOALA_RSS(file5r+"_TCWSULA.fits", rss_clean=True)
#    cube_test=Interpolated_cube(rss_test, pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size, plot=True, centre_deg=[196.4431100079797,-28.42711916881268], size_arcsec=[60,60]) #flux_calibration=flux_calibration_20180310_385R_0p6_1k25,
#   cube_test.plot_weight()

#    rss3_all_bis = KOALA_RSS(path_main+"Tol30Ar3_rss_all.fits", #save_rss_to_fits_file=path_main+"Tol30Ar3_rss_tcwsreu.fits",
#                         apply_throughput=False, skyflat = skyflat_red, correct_ccd_defects = False,
#                         #fix_wavelengths = True, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07],
#                         sky_method="none",
#                         #sky_method="1Dfit", sky_spectrum=sky3, scale_sky_1D = 1., auto_scale_sky = True,
#                         #id_el=True, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, brightest_line_wavelength =6613., #fibre=422, #422
#                         #id_list=[6300.30, 6312.1, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 8750.47, 8862.79, 9014.91, 9069.0],
#                         #clean_sky_residuals = True, dclip=3.0, extra_w = 1.3, step_csr = 25, fibre = 0,
#                         do_extinction=True, correct_negative_sky = True, plot=True, warnings=True)   # rss_clean


# # ---------------------------------------------------------------------------
# #  OBTAIN SKY SPECTRA IF NEEDED
# # ---------------------------------------------------------------------------

#    sky_r1 = KOALA_RSS(file1r, apply_throughput=True, skyflat = skyflat_red,                          # Include this as a sub process
#                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd = 50,
#                       sky_method="none", is_sky=True, win_sky=151,
#                       do_extinction=False, plot=True)
##
#    sky1=sky_r1.plot_combined_spectrum(list_spectra=[985, 983, 982, 966, 967, 956, 964, 965, 984, 981, 937, 979, 973,
#       969, 975, 958, 980, 957, 974, 976, 968, 955, 977, 963, 978, 946,
#       763, 972, 959, 962, 954, 945, 947, 970, 971, 753, 953, 944, 743,
#       936, 745, 960, 950, 951, 755, 952, 948, 754, 747, 748], median=True)
##
#    sky_r2 = KOALA_RSS(file2r, apply_throughput=True, skyflat = skyflat_red, do_extinction=False,
#                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd = 50,
#                       sky_method="none", is_sky=True, win_sky=151,
#                       plot=True)
#
#    sky2=sky_r2.plot_combined_spectrum(list_spectra=[870,871,872,873,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,900], median=True)

#    sky_r3 = KOALA_RSS(file3r, apply_throughput=True, skyflat = skyflat_red, do_extinction=False,
#                       correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd = 50,
#                       sky_method="none", is_sky=True, win_sky=151,
#                       plot=True)
#
#    sky3=sky_r3.plot_combined_spectrum(list_spectra=[870,871,872,873,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,900], median=True)


# Ha in 6613.

#    Tol30Ar = KOALA_RSS(file1r,
#                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
#                       correct_ccd_defects = True, #correct_high_cosmics = False, clip_high = 50, step_ccd=50,
#                       sky_method="1Dfit", sky_spectrum=sky1, brightest_line_wavelength = 6613.,
#                       do_extinction=True,
#                       #sky_method="self", n_sky=50,
#                       #sky_method="1D", sky_spectrum=sky1, scale_sky_1D = 1.0,  #0.97,
#                       telluric_correction = telluric_correction_20180310,
#                       valid_wave_min = 6085, valid_wave_max = 9305,
#                       plot=True, warnings=True)

#    file_rss_clean = path_star+"/Tol30Ar_RSS.fits"
#    save_rss_fits(Tol30Ar, fits_file=file_rss_clean)

#    test_rss_read = KOALA_RSS(file_rss_clean, apply_throughput=False, correct_ccd_defects = False, sky_method="none", do_extinction=False, plot=False, warnings=False)
#    cubes_Tol30Ar_no=Interpolated_cube(test_rss_read, pixel_size, kernel_size, plot=True) #, force_ADR = True)
#    save_fits_file(cubes_Tol30Ar_no, path_star+"/Tol30Ar_no.fits", ADR=False)

#    test_rss = KOALA_RSS(file1r, apply_throughput=True, skyflat = skyflat_red,
#                         correct_ccd_defects = False, sky_method="none", do_extinction=True, plot=True, warnings=True)
###    test_rss_file = path_star+"RSS_rss_test.fits"
#    save_rss_fits(test_rss, fits_file = test_rss_file)
#
#    test_rss_read = KOALA_RSS(test_rss_file, apply_throughput=False, correct_ccd_defects = False, sky_method="none", do_extinction=False, plot=False, warnings=False)
#    cubes_Tol30Ar_no=Interpolated_cube(test_rss_read, pixel_size, kernel_size, plot=True) #, force_ADR = True)


#    test_rss_read = KOALA_RSS("/Users/alopez/Documents/DATA/python/20sep20002red.fits", apply_throughput=False, correct_ccd_defects = False, sky_method="none", do_extinction=False, plot=False, warnings=False)
#    cube_test=Interpolated_cube(test_rss_read, pixel_size, kernel_size, plot=True)


#########   Further testing sky substraction


#    test_rss_file = path_main+"Tol30Ar1_rss_test.fits"
#    test_rss2 = KOALA_RSS(file1r, #save_rss_to_fits_file=test_rss_file,
#                         apply_throughput=True, skyflat = skyflat_red, correct_ccd_defects = True,
#                         sky_method="none", #sky_spectrum=sky1, brightest_line_wavelength = 6613., fibre = 400,
#                         do_extinction=False, plot=True, warnings=True)


#    test_rss = KOALA_RSS(test_rss_file, #save_rss_to_fits_file=test_rss_file,
#                         apply_throughput=False, skyflat = skyflat_red, correct_ccd_defects = False,
#                         sky_method="1Dfit", sky_spectrum=sky1, scale_sky_1D = 0.99, brightest_line_wavelength = 6613., fibre = 0,
#                         #telluric_correction = telluric_correction_20180310,
#                         do_extinction=False, plot=True, warnings=True, verbose = True)

#    test_rss.plot_spectra(list_spectra=[4],xmin=7800,xmax=8000,ymin=-50,ymax=300)
#    save_rss_fits(test_rss, fits_file = path_main+"Tol30Ar1_rss_test_clean_wave.fits")


###### WAVELENGTH TWEAK


#    w = test_rss.wavelength
#    f = test_rss.intensity_corrected[2]
#    w_shift = 0.5
#    f_new = rebin_spec_shift(w,f,w_shift)

#    a = test_rss.wavelength_offset_per_fibre
#    b=[]
#    for i in range(len(a)):
#        b.append(i+1)
#
#    a2x,a1x,a0x = np.polyfit(b, a, 2)
#    fx = a0x + a1x*np.array(b)+ a2x*np.array(b)**2
#    print a0x,a1x,a2x   # 0.46633025484751833 -0.0006269969715551119 1.2829287842692372e-07
#
#    plt.figure(figsize=(14, 4))
#    plt.plot(b,a,"g", alpha=0.7)
#    plt.plot(b,fx,"r", alpha=0.7)
#    plt.minorticks_on()
#    plt.show()
#    plt.close()
#
#    w = test_rss2.wavelength
#    for i in range(len(a)):
#        shift = a0x + a1x*(i+1)+ a2x*(i+1)**2
#        test_rss2.intensity_corrected[i]=rebin_spec_shift(w, test_rss2.intensity_corrected[i], shift)

#    save_rss_fits(test_rss2, fits_file = path_main+"Tol30Ar1_rss_test2.fits")

#    test_rss = KOALA_RSS(path_main+"Tol30Ar1_rss_test2.fits", #save_rss_to_fits_file=test_rss_file,
#                         apply_throughput=False, skyflat = skyflat_red, correct_ccd_defects = False,
#                         sky_method="1D", sky_spectrum=sky1, scale_sky_1D = 0.99, brightest_line_wavelength = 6613., fibre = 0,
#                         #telluric_correction = telluric_correction_20180310,
#                         do_extinction=False, plot=True, warnings=True, verbose = True)


#    w= test_rss.wavelength
#    f1 = test_rss.intensity_corrected[1]
#    #f1no=test_rss.intensity[1]
#    f900 = test_rss.intensity_corrected[900]
#    wmin,wmax=7250,7800
#    ymin,ymax=-30,300
#
#    plt.figure(figsize=(14, 4))
#    plt.plot(w,f1,"g", alpha=0.7, label="Fibre 1")
##    plt.plot(w,f1no,"r", alpha=0.7, label="Fibre 2")
##
#    plt.plot(w,f900,"b", alpha=0.7, label="Fibre 900")
##    #plt.plot(w,f_new,"b", alpha=0.7, label="Fibre 2 moved")
##
#    plt.xlim(wmin,wmax)
#    plt.ylim(ymin,ymax)
###    ptitle = "Fibre "+np.str(fibre)#+" with rms = "+np.str(rms[i])
###    plt.title(ptitle)
##    plt.xlabel("Wavelength [$\AA$]")
##    plt.ylabel("Flux [counts]")
##    plt.legend(frameon=True, loc=2, ncol=4)
#    plt.minorticks_on()
###    for i in range(len(el_list)):
###        plt.axvline(x=el_list[i], color="k", linestyle='--',alpha=0.5)
###    for i in range(number_sl):
###        plt.axvline(x=sl_center[i], color="y", linestyle='--',alpha=0.6)
#    plt.show()
#    plt.close()
#


# ------------

# Let's do it all together:

# First we get the sky AFTER

#    list_sky=[985, 983, 982, 966, 967, 956, 964, 965, 984, 981, 937, 979, 973,
#              969, 975, 958, 980, 957, 974, 976, 968, 955, 977, 963, 978, 946] #,
#              #763, 972, 959, 962, 954, 945, 947, 970, 971, 753, 953, 944, 743,
#              #936, 745, 960, 950, 951, 755, 952, 948, 754, 747, 748]

#    sky3=sky_spectrum_from_fibres_using_file(file3r, list_sky, apply_throughput=True, skyflat=skyflat_red, correct_ccd_defects= True,
#                                             fix_wavelengths = False, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07], plot=False)

#    sky3=sky_spectrum_from_fibres(rss3_tcw, list_sky, plot=True)

# Using EG274 to get the sky

#    filesky = path_main+date+"/"+grating+"/10mar20104red.fits"
#
#
#    sky3_new=sky_spectrum_from_fibres_using_file(filesky, fibre_list=[0], n_sky=200,
#                                             apply_throughput=True, skyflat=skyflat_red, correct_ccd_defects= True,
#                                             fix_wavelengths = True, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07], plot=False)


# Now we read the file correcting for everything

#    rss3_test= KOALA_RSS(file3r, save_rss_to_fits_file=path_main+"Tol30Ar3_rss_PUTA.fits",
#                         apply_throughput=True, skyflat = skyflat_red, correct_ccd_defects = True,
#                         #fix_wavelengths = True, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07],
#                         sky_method="none", do_extinction=False, plot=True, warnings=True)
#
#    rss3_test2= KOALA_RSS(path_main+"Tol30Ar3_rss_PUTA.fits", #save_rss_to_fits_file=path_main+"Tol30Ar3_rss_PUTA.fits",
#                         apply_throughput=False, skyflat = skyflat_red, correct_ccd_defects = False,
#                         #fix_wavelengths = True, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07],
#                         sky_method="none", do_extinction=True, plot=True, warnings=True)


#    rss3_tcws = KOALA_RSS(path_main+"Tol30Ar3_rss_tcw.fits", #save_rss_to_fits_file=path_main+"Tol30Ar3_rss_tcws.fits",
#                         apply_throughput=False, skyflat = skyflat_red, correct_ccd_defects = False,
#                         #fix_wavelengths = True, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07],
#                         #sky_method="none",
#                         sky_method="1Dfit", sky_spectrum=sky3, scale_sky_1D = 1, auto_scale_sky = True, brightest_line_wavelength = 6613., fibre = 900,
#                         do_extinction=False, plot=True, warnings=True)

#    rss3_tcws.plot_spectra(list_spectra=[230],ymin=0,ymax=300)

#    rss3_tcws.plot_spectra(list_spectra=[10],ymin=0,ymax=350,xmin=8500,xmax=9000)

#    fibre = 502
#    w=rss3_tcwsr.wavelength
#    f=rss3_tcwsr.intensity_corrected[fibre]
#    plot_spec(w,sky3_new,13)
#    plot_spec(w,f)
#    plot_plot(w,f, ymin=0,ymax=350,xmin=8600,xmax=8800)#, fig_size=13)


#    save_rss_fits(rss3_tcws, fits_file = path_main+"Tol30Ar3_rss_tcws.fits")


#    rss3_tcwsr = KOALA_RSS(path_main+"Tol30Ar3_rss_tcw.fits", save_rss_to_fits_file=path_main+"Tol30Ar3_rss_tcwsreu.fits",
#                         apply_throughput=False, skyflat = skyflat_red, correct_ccd_defects = False,
#                         #fix_wavelengths = True, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07],
#                         #sky_method="none",
#                         sky_method="1Dfit", sky_spectrum=sky3, scale_sky_1D = 1., auto_scale_sky = True,
#                         id_el=True, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, brightest_line_wavelength =6613., #fibre=422, #422
#                         id_list=[6300.30, 6312.1, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 8750.47, 8862.79, 9014.91, 9069.0],
#                         clean_sky_residuals = True, dclip=3.0, extra_w = 1.3, step_csr = 25, fibre = 0,
#                         do_extinction=True, plot=True, warnings=True)


#    rss2 = KOALA_RSS(file2r, #save_rss_to_fits_file=test_rss_file,
#                         apply_throughput=True, skyflat = skyflat_red, correct_ccd_defects = True,
#                         sky_method="none", #sky_spectrum=sky1, brightest_line_wavelength = 6613., fibre = 400,
#                         do_extinction=False, plot=True, warnings=True)
#
#
##    lista = [981,982,983,984,985]
##    sky = sky_spectrum_from_fibres(file1r, list_spectra=lista, plot=False) # correct_ccd_defects= True, skyflat = skyflat_red, plot=False)
#

###sky=sky_spectrum_from_fibres(rss, lista, win_sky=151, xmin=0, xmax=0, ymin=0, ymax=0, verbose = True, plot= True)


#########   Fixing more things...     BACKGROUND


#    rss3_tcwsru = KOALA_RSS(path_main+"Tol30Ar3_rss_tcwsreu.fits", #save_rss_to_fits_file=path_main+"Tol30Ar3_rss_tcwsreu.fits",
#                         apply_throughput=False, skyflat = skyflat_red, correct_ccd_defects = False,
#                         #fix_wavelengths = True, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07],
#                         sky_method="none",
#                         #sky_method="1Dfit", sky_spectrum=sky3, scale_sky_1D = 1., auto_scale_sky = True,
#                         #id_el=True, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, brightest_line_wavelength =6613., #fibre=422, #422
#                         #id_list=[6300.30, 6312.1, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 8750.47, 8862.79, 9014.91, 9069.0],
#                         #clean_sky_residuals = True, dclip=3.0, extra_w = 1.3, step_csr = 25, fibre = 0,
#                         do_extinction=False, plot=True, warnings=True)


######## telluric correction   telluric_correction = telluric_correction_20180310

#    save_rss_fits(rss3_all, fits_file = path_main+"Tol30Ar3_rss_all.fits")
#
#    rss3_all_bis = KOALA_RSS(path_main+"Tol30Ar3_rss_all.fits", #save_rss_to_fits_file=path_main+"Tol30Ar3_rss_tcwsreu.fits",
#                         apply_throughput=False, skyflat = skyflat_red, correct_ccd_defects = False,
#                         #fix_wavelengths = True, sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07],
#                         sky_method="none",
#                         #sky_method="1Dfit", sky_spectrum=sky3, scale_sky_1D = 1., auto_scale_sky = True,
#                         #id_el=True, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, brightest_line_wavelength =6613., #fibre=422, #422
#                         #id_list=[6300.30, 6312.1, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 8750.47, 8862.79, 9014.91, 9069.0],
#                         #clean_sky_residuals = True, dclip=3.0, extra_w = 1.3, step_csr = 25, fibre = 0,
#                         do_extinction=True, correct_negative_sky = True, plot=True, warnings=True)   # rss_clean
#
#
##    rss3_all_bis.plot_spectra([222], xmin=8200,xmax=9200, ymin=100, ymax=800)
#    rss3_all_bis.plot_spectra([977, 982, 983, 985, 984, 981, 975, 971, 974, 964], xmin=6200,xmax=7200, ymin=-50, ymax=50)

#    rss3_all = KOALA_RSS(file3r, #save_rss_to_fits_file=path_main+"Tol30Ar3_rss_tcwsreu.fits",
#                         apply_throughput=True, skyflat = skyflat_red,
#                         correct_ccd_defects = True,
#                         fix_wavelengths = True, #sol = [0.10198480885572622, -0.0006885696621193424, 1.8422163305742697e-07],
#                         #sky_method="none",
#                         sky_method="1Dfit", sky_spectrum=sky3, auto_scale_sky = True,
#                         id_el=False, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, brightest_line_wavelength =6613., #fibre=422, #422
#                         id_list=[6300.30, 6312.1, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 8750.47, 8862.79, 9014.91, 9069.0],
#                         clean_sky_residuals = False, dclip=3.0, extra_w = 1.3, step_csr = 25, fibre = 0,
#                         telluric_correction = telluric_correction_20180310,
#                         do_extinction=False, correct_negative_sky = False,
#                         plot=True, warnings=True)


#    w = rss.wavelength
#    plt.figure(figsize=(10, 4))
#    for i in [0,300,600,950]:
#        plt.plot(w,rss.intensity[i])
#    plot_plot(w,rss.intensity[0], ptitle= "Before corrections, fibres 0, 300, 600, 950", xmin=7740,xmax=7770, ymin=0, ymax=1000)


#    fix_2dfdr_wavelengths(rss2, #sol=[0.11615280893895372, -0.0006933150714461267, 1.8390944138355269e-07],
#                          xmin=7740,xmax=7770, ymin=0, ymax=1000, plot=True, verbose=True, warnings=True)


#    compare_fix_2dfdr_wavelengths(rss,rss2)


#
#    a2x,a1x,a0x = np.polyfit(xfibre, median_offset, 2)
#    fx = a0x + a1x*np.array(xfibre)+ a2x*np.array(xfibre)**2
#    plt.plot(xfibre,fx,"r")
#    print a0x,a1x,a2x   # 0.11615280893895372 -0.0006933150714461267 1.8390944138355269e-07
#
#    plot_plot(xfibre,median_offset, ptitle= "Second-order fit to individual offsets", xmin=-20,xmax=1000, ymin=-0.5, ymax=0.2, xlabel="Fibre", ylabel="offset")


#    save_rss_fits(test_rss, fits_file = test_rss_file)

#    test_rss = KOALA_RSS(test_rss_file, save_rss_to_fits_file=test_rss_file_2, apply_throughput=True, skyflat = skyflat_red,
#                         correct_ccd_defects = True, sky_method="none", do_extinction=False, plot=True, warnings=True)

#    test_rss = KOALA_RSS(test_rss_file_2, #save_rss_to_fits_file=test_rss_file_2,
#                         apply_throughput=False, skyflat = skyflat_red, correct_ccd_defects = False,
#                         sky_method="1Dfit", sky_spectrum=sky1, brightest_line_wavelength = 6613., fibre = 400,
#                         do_extinction=True, telluric_correction = telluric_correction_20180310, plot=True, warnings=True)


#    path_cynthia = "/Users/alopez/Documents/DATA/GAUSS/2019_09_Run_Cynthia_GAUSS/20190907/385R/"
#    file_skyflatr=path_cynthia+"07sep20010red.fits"                                  # FILE NOT DIVIDED BY THE FLAT

#    skyflat_red_cynthia = KOALA_RSS(file_skyflatr, flat="", apply_throughput=False, sky_method="none",
#                             do_extinction=False, correct_ccd_defects = False,
#                             correct_high_cosmics = False, clip_high = 100, step_ccd = 50, plot=True)
#    skyflat_red_cynthia.find_relative_throughput(ymin=0, ymax=20000,  wave_min_scale=6300, wave_max_scale=6500)  #


# Testing problem with offsets...


#    file_cynthia_1r = path_cynthia+"07sep20034red.fits"
#    file_cynthia_2r = path_cynthia+"07sep20035red.fits"
#    rss_list = [file_cynthia_1r,file_cynthia_2r]
#    cynthia_t=KOALA_reduce(rss_list, fits_file=path_cynthia+"V850_Aql.fits", obj_name="V850_Aql",  description="V850_Aql",
#                           apply_throughput=True, skyflat = skyflat_red_cynthia,
#                           correct_ccd_defects = False, correct_high_cosmics = False, clip_high = 50, step_ccd=50,
#                           sky_method="self", n_sky=400,
#                           telluric_correction = telluric_correction_20190907,
#                           pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
#                           flux_calibration=flux_calibration_20190907_385R_0p6_1k8,
#                           ADR= False,
#                           valid_wave_min = 6085, valid_wave_max = 9305,
#                           plot= True, warnings=False )
#


#    file_cynthia_1r = path_cynthia+"07sep20034red.fits"
#    file_cynthia_2r = path_cynthia+"07sep20035red.fits"
#    file_cynthia_3r = path_cynthia+"07sep20036red.fits"
#    rss_list = [file_cynthia_1r,file_cynthia_2r,file_cynthia_3r]
#    cynthia=KOALA_reduce(rss_list, fits_file=path_cynthia+"V850_Aql.fits", obj_name="V850_Aql",  description="V850_Aql",
#                           apply_throughput=True, skyflat = skyflat_red_cynthia,
#                           correct_ccd_defects = False, correct_high_cosmics = False, clip_high = 50, step_ccd=50,
#                           sky_method="self", n_sky=400,
#                           telluric_correction = telluric_correction_20190907,
#                           pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
#                           flux_calibration=flux_calibration_20190907_385R_0p6_1k8,
#                           #offsets=[0, 0, -2.2, 0.05],  # EAST-/WEST+  NORTH-/SOUTH+
#                           ADR= False,
#                           valid_wave_min = 6085, valid_wave_max = 9305,
#                           plot= True, warnings=False )
###
## # Extract the integrated spectrum of the star & save it
#    cynthia.combined_cube.half_light_spectrum(r_max=5, plot=True)
#    spectrum_to_text_file(H600r.combined_cube.wavelength,H600r.combined_cube.integrated_star_flux, filename=fits_file_red+"_integrated_star_flux.dat")


# # STAR 4
#    star="EG274"
#    path_star = path_cynthia #path_main+date+"/"+grating+"/"
#    starpos1r = path_star+"07sep20026red.fits"
#    starpos2r = path_star+"07sep20027red.fits"
#    starpos3r = path_star+"07sep20028red.fits"
#    fits_file_red = path_star+star+"_"+grating+pk
##    response_file_red = path_star+star+"_"+grating+pk+"_response.dat"
##    telluric_file = path_star+star+"_"+grating+pk+"_telluric_correction.dat"
#
#    rss_list = [starpos1r,starpos2r,starpos3r]
#    EG274r=KOALA_reduce(rss_list, fits_file="path:"+path_star, obj_name=star,  description=star, rss_clean=True, #save_aligned_cubes= True,    ############
#                           apply_throughput=True, skyflat = skyflat_red_cynthia,
#                           correct_ccd_defects = False, correct_high_cosmics = False, clip_high = 50, step_ccd=50,
#                           sky_method="self", n_sky=400,
#                           pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
#                           ADR= False,
#                           offsets=[-2.415, 6.156, -4.179, 0],  # EAST-/WEST+  NORTH-/SOUTH+
#                           valid_wave_min = 6085, valid_wave_max = 9305,
#                           plot= True, warnings=False )


##
#    EG274r.combined_cube.half_light_spectrum(r_max=5, plot=True)
##    spectrum_to_text_file(EG274r.combined_cube.wavelength,EG274r.combined_cube.integrated_star_flux, filename=fits_file_red+"_integrated_star_flux.dat")

#    telluric_correction_star4 = EG274r.get_telluric_correction(apply_tc=True,  combined_cube = True,
#                                                                weight_fit_median = 1., step = 15, wave_min=6085, wave_max=9305,
#                                                                correct_from=6830., correct_to=8420.,
#                                                                exclude_wlm=[[6000,6330],[6460,6720],[6830,7450], [7550,7750],[8050,8420]])
## #
#    spectrum_to_text_file(EG274r.combined_cube.wavelength,telluric_correction_star4, filename=telluric_file)
# #
#    EG274r.combined_cube.do_response_curve('FLUX_CAL/feg274_edited.dat', plot=True, min_wave=6080., max_wave=9305.,    # FIX BLUE END !!!
#                              step=10, exp_time=180., fit_degree=5)
# #
#    spectrum_to_text_file(EG274r.combined_cube.response_wavelength,EG274r.combined_cube.response_curve, filename=response_file_red)


# # Define in "stars" the 2 cubes we are using, and plotting their responses to check
#    stars=[EG274r.combined_cube]
#    plot_response(stars, scale=[1,1])
# # We obtain the flux calibration applying:
#    flux_calibration_20190907_385R_0p6_1k8 = obtain_flux_calibration(stars)

# # And we save this absolute flux calibration as a text file
#    flux_calibration_file = path_main+date+"/flux_calibration_"+date+"_"+grating+pk+".dat"
#    spectrum_to_text_file(H600r.combined_cube.wavelength,flux_calibration_20180310_385R_0p6_1k8, filename=flux_calibration_file)


# #   CHECK AND GET THE TELLURIC CORRECTION

# # Similarly, provide a list with the telluric corrections and apply:
#    telluric_correction_list=[telluric_correction_star1,telluric_correction_star2,telluric_correction_star3,telluric_correction_star4]
#    telluric_correction_list=[telluric_correction_star4]  # [telluric_correction_star1,]
#    telluric_correction_20190907 = obtain_telluric_correction(EG274r.combined_cube.wavelength, telluric_correction_list)

# # Save this telluric correction to a file
#    telluric_correction_file = path_main+date+"/telluric_correction_"+date+"_"+grating+pk+".dat"
#    spectrum_to_text_file(H600r.combined_cube.wavelength,telluric_correction_20180310, filename=telluric_correction_file )

#    EG274r=KOALA_reduce(rss_list, fits_file=fits_file_red+".fits", obj_name=star,  description=star,
#                           apply_throughput=True, skyflat = skyflat_red_cynthia,
#                           correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 50, step_ccd=50,
#                           sky_method="self", n_sky=400,
#                           pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
#                           ADR= False,
#                           offsets=[2.5, 6, 4.2, 0],  # EAST-/WEST+  NORTH-/SOUTH+
#                           valid_wave_min = 6085, valid_wave_max = 9305,
#                           plot= True, warnings=False )

# Sh 2-61
#    starpos1r = path_star+"07sep20030red.fits"
#    starpos2r = path_star+"07sep20031red.fits"
#    starpos3r = path_star+"07sep20032red.fits"
#    OBJECT      = "Sh2-61"
#    fits_file_red = path_star+OBJECT+"_"+grating+pk+".fits"
#
#    rss_list = [starpos1r,starpos2r,starpos3r]
#    Sh2r = KOALA_reduce(rss_list, fits_file=fits_file_red, obj_name=OBJECT,  description=OBJECT,
#                          apply_throughput=True, skyflat = skyflat_red_cynthia, plot_skyflat=False,
#                          correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd=50,
#                          #sky_method="1D", sky_list=sky_list,  scale_sky_1D =0.98,
#                          sky_method="self", n_sky = 100, #sky_list=sky_list,
#                          telluric_correction = telluric_correction_20190907,
#                          do_extinction=True,
#                          #id_el=True, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, #brightest_line_wavelength =6580., # fibre=422, #422
#                          #id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39,7329.66, 9069.0],
#                          #clean_sky_residuals = True, dclip=3.0, extra_w = 1.3, step_csr = 25,
#
#                          pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
#                          #offsets=[1.25, 0., 0, 2.17],  # EAST-/WEST+  NORTH-/SOUTH+  #ad[-1.04, -2,128], #bd[-1.5794, -1.256], #[0.53, -0.872, -1.5794, -1.256],#a->b[1.55, -0.17], # 0.62, -1.08, -1.88, -1.08],
#                          ADR=False,
#                          flux_calibration=flux_calibration_20190907_385R_0p6_1k8,
#                          valid_wave_min = 6085, valid_wave_max = 9305,
#                          plot=True, warnings=False)
#

########


#    test_rss = KOALA_RSS(file_rss_clean, apply_throughput=False, correct_ccd_defects = False, sky_method="none", do_extinction=False, plot=True, warnings=True)

#    Tol30Ar_no = KOALA_RSS(file1r,
#                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
#                       correct_ccd_defects = True, #correct_high_cosmics = False, clip_high = 50, step_ccd=50,
#                       #sky_method="none", do_extinction=False,
#                       #sky_method="self", n_sky=50,
#                       sky_method="1D", sky_spectrum=sky1, scale_sky_1D = 1.0,  #0.97,
#                       telluric_correction = telluric_correction_20180310,
#                       valid_wave_min = 6085, valid_wave_max = 9305,
#                       plot=True, warnings=True)
#
#    cubes_Tol30Ar_no=Interpolated_cube(Tol30Ar_no, pixel_size, kernel_size, plot=True) #, force_ADR = True)
#    save_fits_file(cubes_Tol30Ar_no, path_star+"/Tol30Ar_no.fits", ADR=False)


#
#    rss_list=[file1r,file2r,file3r]  #,file4r,file5r,file6r,file7r]
##    sky_list=[sky1,sky2,sky3]
#    fits_file_red=path_main+date+"/"+grating+"/Tol30_A_red_combined_cube_3.fits"
######
#    hikids_red = KOALA_reduce(rss_list, fits_file=fits_file_red, obj_name=OBJECT,  description=DESCRIPTION,
#                          apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
#                          correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd=50,
#                          #sky_method="1D", sky_list=sky_list,  scale_sky_1D =0.98,
#                          sky_method="self", n_sky = 100, #sky_list=sky_list,
#                          telluric_correction = telluric_correction_20180310,
#                          do_extinction=True,
#                          id_el=True, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, #brightest_line_wavelength =6580., # fibre=422, #422
#                          id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39,7329.66, 9069.0],
#                          clean_sky_residuals = True, dclip=3.0, extra_w = 1.3, step_csr = 25,
#
#                          pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
#                          offsets=[1.25, 0., 0, 2.17],  # EAST-/WEST+  NORTH-/SOUTH+  #ad[-1.04, -2,128], #bd[-1.5794, -1.256], #[0.53, -0.872, -1.5794, -1.256],#a->b[1.55, -0.17], # 0.62, -1.08, -1.88, -1.08],
#                          ADR=False,
#                          flux_calibration=flux_calibration_,#20180310_385R_0p6_1k8,
#
#                          valid_wave_min = 6085, valid_wave_max = 9305,
#                          plot=True, warnings=False)


#############

#    OBJECT      = "Tol30"
#    DESCRIPTION = "Tol30"
#    file1r=path_main+date+"/"+grating+"/10mar20097red.fits"
#    file2r=path_main+date+"/"+grating+"/10mar20098red.fits"
#    file3r=path_main+date+"/"+grating+"/10mar20099red.fits"
#    file4r=path_main+date+"/"+grating+"/10mar20101red.fits"
#    file5r=path_main+date+"/"+grating+"/10mar20102red.fits"
#    file6r=path_main+date+"/"+grating+"/10mar20103red.fits"
#
#    rss_list=[file1r,file2r,file3r,file4r,file5r,file6r]#,file7r]
##    sky_list=[sky1,sky2,sky3]
#    fits_file_red=path_main+date+"/"+grating+"/Tol30_mosaic_test_6_new.fits"
######
#    hikids_red = KOALA_reduce(rss_list, fits_file=fits_file_red, obj_name=OBJECT,  description=DESCRIPTION,
#                          apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
#                          correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd=50,
#                          #sky_method="1D", sky_list=sky_list,  scale_sky_1D =0.98,
#                          sky_method="self", n_sky = 50, #sky_list=sky_list,
#                          telluric_correction = telluric_correction_20180310,
#                          do_extinction=True,
#                          id_el=True, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=1.8, brightest_line_wavelength =6615.5, # fibre=422, #422
#                          id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39,7329.66, 9069.0],
#                          clean_sky_residuals = True, dclip=3.0, extra_w = 1.3, step_csr = 25,
#
#                          pixel_size_arcsec=pixel_size, kernel_size_arcsec=kernel_size,
#                          offsets=[1.25, 0., 0, 2.17, 1.25, 12.5, 0, 2.17, -0.63, -1.08],  # EAST-/WEST+  NORTH-/SOUTH+  #ad[-1.04, -2,128], #bd[-1.5794, -1.256], #[0.53, -0.872, -1.5794, -1.256],#a->b[1.55, -0.17], # 0.62, -1.08, -1.88, -1.08],
#                          ADR=False,
#                          flux_calibration=flux_calibration_,#20180310_385R_0p6_1k8,
#
#                          valid_wave_min = 6085, valid_wave_max = 9305,
#                          plot=True, warnings=False)


# # ---------------------------------------------------------------------------
# # ---------------------------------------------------------------------------
# # ---------------------------------------------------------------------------
# # ---------------------------------------------------------------------------
# # ---------------------------------------------------------------------------
# # ---------------------------------------------------------------------------
# # ---------------------------------------------------------------------------


# #
# #
# #
# # ---------------------------------------------------------------------------
# # TIME FOR THE OBJECT !!
# # ---------------------------------------------------------------------------
# #
# #
# #


#    OBJECT = "He2-10"
#    DESCRIPTION = "He2-10 CUBE"
#
#    file1r=path_main+date+"/385R/27feb20031red.fits"
#    file2r=path_main+date+"/385R/27feb20032red.fits"
#    file3r=path_main+date+"/385R/27feb20033red.fits"
#
#    rss_list=[file1r,file2r,file3r]  #,file4r,file5r,file6r,file7r]
#    #sky_list=[sky_r1,sky_r1,sky_r2,sky_r2,sky_r3,sky_r3,sky_r4]
#    fits_file_red=path_main+date+"/385R/"+OBJECT+"/He2-10_A_red_combined_cube_3.fits"
#
#    hikids_red = KOALA_reduce(rss_list, fits_file=fits_file_red, obj_name=OBJECT,  description=DESCRIPTION,
#                          apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
#                          correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd=50,
#                          do_extinction=True,
#                          sky_method="self", #sky_list=sky_list,
#                          telluric_correction = telluric_correction_20180227red,
#                          id_el=False, high_fibres=10, brightest_line="Ha", cut=1.1, broad=4.0, plot_id_el= False,
#                          id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6716.47, 6730.85],
#                          clean_sky_residuals = False, dclip=2.0, extra_w = 1.3, step_csr = 100, # fibre=427,
#
#                          pixel_size_arcsec=.6, kernel_size_arcsec=1.25,
#                          offsets=[-0.197,1.5555,   2.8847547360544095, 1.4900605034042318],
#                          ADR=False,
#                          flux_calibration=flux_calibration_20180227_385R_0p6_1k25,
#
#                          valid_wave_min = 6085, valid_wave_max = 9305,
#                          plot=True, warnings=False)
#


#                     # RSS
#                     # skyflat_file is a RSS, skyflat and skyflat_list are the names of objects keeping the relative throughput of skyflats
#                     apply_throughput=True, skyflat = "", skyflat_file="", flat="",
#                     skyflat_list=["","","","","","",""],
#                     # This line is needed if doing FLAT when reducing (NOT recommended)
#                     plot_skyflat=False, wave_min_scale=0, wave_max_scale=0, ymin=0, ymax=0,
#                     # Correct CCD defects & high cosmics
#                     correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 100, step_ccd = 50, remove_5578 = False, plot_suspicious_fibres=False,
#                     # Correct for extinction
#                     do_extinction=True,
#                     # Sky substraction
#                     sky_method="self", n_sky=50, sky_fibres=[1000], # do_sky=True
#                     sky_spectrum=[0], sky_rss=[0], scale_sky_rss=0,
#                     sky_wave_min = 0, sky_wave_max =0, cut_sky=5., fmin=1, fmax=10,
#                     individual_sky_substraction=False, fibre_list=[100,200,300,400,500,600,700,800,900],
#                     sky_list=[[0],[0],[0],[0],[0],[0],[0]],
#                     # Telluric correction
#                     telluric_correction = [0], telluric_correction_list=[[0],[0],[0],[0],[0],[0],[0]],
#                     # Identify emission lines
#                     id_el=False, high_fibres=10, brightest_line="Ha", cut=1.5, plot_id_el=True, broad=2.0, id_list=[0],
#                     # Clean sky residuals
#                     clean_sky_residuals = False, dclip=3.0, extra_w = 1.3, step_csr = 25,
#
#                     # CUBING
#                     pixel_size_arcsec=.4, kernel_size_arcsec=1.2,
#                     offsets=[1000],
#                     ADR=False,
#                     flux_calibration=[0], flux_calibration_list=[[0],[0],[0],[0],[0],[0],[0]],
#
#                     # COMMON TO RSS AND CUBING
#                     valid_wave_min = 0, valid_wave_max = 0,
#                     plot= True, norm=colors.LogNorm(), fig_size=12,
#                     warnings=False, verbose = False):

#
#    OBJECT = "He2-10"
#    DESCRIPTION = "He2-10 B CUBE"
#
#    file1r=path_main+date+"/385R/27feb20034red.fits"
#    file2r=path_main+date+"/385R/27feb20035red.fits"
#    file3r=path_main+date+"/385R/27feb20036red.fits"
#
#    rss_list=[file1r,file2r,file3r]  #,file4r,file5r,file6r,file7r]
#    #sky_list=[sky_r1,sky_r1,sky_r2,sky_r2,sky_r3,sky_r3,sky_r4]
#    fits_file_red=path_main+date+"/385R/"+OBJECT+"/He2-10_B_red_combined_cube_3.fits"
#
#    hikids_red = KOALA_reduce(rss_list, fits_file=fits_file_red, obj_name=OBJECT,  description=DESCRIPTION,
#                          apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
#                          correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd=50,
#                          do_extinction=True,
#                          sky_method="self", #sky_list=sky_list,
#                          telluric_correction = telluric_correction_20180227red,
#                          id_el=False, high_fibres=10, brightest_line="Ha", cut=1.1, broad=4.0, plot_id_el= False,
#                          id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6716.47, 6730.85],
#                          clean_sky_residuals = False, dclip=2.0, extra_w = 1.3, step_csr = 100, # fibre=427,
#
#                          pixel_size_arcsec=.6, kernel_size_arcsec=1.25,
#                          offsets=[-1.5,0,    -1.5, -1.5],
#                          ADR=False,
#                          flux_calibration=flux_calibration_20180227_385R_0p6_1k25,
#
#                          valid_wave_min = 6085, valid_wave_max = 9305,
#                          plot=True, warnings=False)
#
#
#


#    OBJECT = "Caleb1"
#    DESCRIPTION = "Caleb1 CUBE"
#
#    file1r=path_main+date+"/385R/27feb20040red.fits"
#    file2r=path_main+date+"/385R/27feb20041red.fits"
#    file3r=path_main+date+"/385R/27feb20042red.fits"
#
#    rss_list=[file1r,file2r,file3r]  #,file4r,file5r,file6r,file7r]
#    #sky_list=[sky_r1,sky_r1,sky_r2,sky_r2,sky_r3,sky_r3,sky_r4]
#    fits_file_red=path_main+date+"/385R/"+OBJECT+"/Caleb1_red_combined_cube_3.fits"
#
#    hikids_red = KOALA_reduce(rss_list, fits_file=fits_file_red, obj_name=OBJECT,  description=DESCRIPTION,
#                          apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
#                          correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd=50,
#                          do_extinction=True,
#                          sky_method="self", #sky_list=sky_list,
#                          telluric_correction = telluric_correction_20180227red,
#                          id_el=False, high_fibres=10, brightest_line="Ha", cut=1.1, broad=4.0, plot_id_el= False,
#                          id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6716.47, 6730.85],
#                          clean_sky_residuals = False, dclip=2.0, extra_w = 1.3, step_csr = 100, # fibre=427,
#
#                          pixel_size_arcsec=.6, kernel_size_arcsec=1.25,
#                          #offsets=[-1.5,0,    -1.5, -1.5],
#                          ADR=False,
#                          flux_calibration=flux_calibration_20180227_385R_0p6_1k25,
#
#                          valid_wave_min = 6085, valid_wave_max = 9305,
#                          plot=True, warnings=False)


# # ---------------------------------------------------------------------------
# # ---------------------------------------------------------------------------
# # ---------------------------------------------------------------------------
# # ---------------------------------------------------------------------------
# # ---------------------------------------------------------------------------
# # ---------------------------------------------------------------------------
# # ---------------------------------------------------------------------------


#######  SKY SUBSTRACTION - NEW 13 Sep 2019 - Included as task on 15 Sep 2019
#
#

#    fibre =500 #422 # 422 max
#    warnings = True
#    verbose = True
#    plot = True
#
#    #brightest_line="Ha"
#    brightest_line_wavelength = 6640.
#    redshift = brightest_line_wavelength/6562.82 - 1.
#    maxima_sigma=12.
#    w = pox4ar_no.wavelength
#    spec= pox4ar_no.intensity_corrected[fibre]
#    sky = sky1
#
#
#    # Read file with sky emission lines
#    sky_lines_file="sky_lines.dat"
#    sl_center,sl_name,sl_fnl,sl_lowlow,sl_lowhigh,sl_highlow,sl_highhigh,sl_lmin,sl_lmax = read_table(sky_lines_file, ["f", "s", "f", "f", "f", "f", "f", "f", "f"] )
#    number_sl = len(sl_center)
#
#    # MOST IMPORTANT EMISSION LINES IN RED
#    # 6300.30       [OI]  -0.263   30.0 15.0   20.0   40.0
#    # 6312.10     [SIII]  -0.264   30.0 18.0    5.0   20.0
#    # 6363.78       [OI]  -0.271   20.0  4.0    5.0   30.0
#    # 6548.03      [NII]  -0.296   45.0 15.0   55.0   75.0
#    # 6562.82         Ha  -0.298   50.0 25.0   35.0   60.0
#    # 6583.41      [NII]  -0.300   62.0 42.0    7.0   35.0
#    # 6678.15        HeI  -0.313   20.0  6.0    6.0   20.0
#    # 6716.47      [SII]  -0.318   40.0 15.0   22.0   45.0
#    # 6730.85      [SII]  -0.320   50.0 30.0    7.0   35.0
#    # 7065.28        HeI  -0.364   30.0  7.0    7.0   30.0
#    # 7135.78    [ArIII]  -0.374   25.0  6.0    6.0   25.0
#    # 7318.39      [OII]  -0.398   30.0  6.0   20.0   45.0
#    # 7329.66      [OII]  -0.400   40.0 16.0   10.0   35.0
#    # 7751.10    [ArIII]  -0.455   30.0 15.0   15.0   30.0
#    # 9068.90    [S-III]  -0.594   30.0 15.0   15.0   30.0
#
#    el_list_no_z = [6300.3, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 7751.1, 9068.9]
#    el_list = (redshift +1) * np.array(el_list_no_z)
#                      #  [OI]   [SIII]  [OI]   Ha+[NII]  HeI    [SII]     HeI   [ArIII]  [OII]  [ArIII]  [SIII]
#    el_low_list_no_z  =[6296.3, 6308.1, 6359.8, 6544.0, 6674.2, 6712.5, 7061.3, 7131.8, 7314.4, 7747.1, 9063.9]
#    el_high_list_no_z =[6304.3, 6316.1, 6367.8, 6590.0, 6682.2, 6736.9, 7069.3, 7139.8, 7333.7, 7755.1, 9073.9]
#    el_low_list= (redshift +1) * np.array(el_low_list_no_z)
#    el_high_list= (redshift +1) *np.array(el_high_list_no_z)
#
#
#    say_status = 0
#    if fibre != 0:
#        f_i=fibre
#        f_f=fibre+1
#        print "  Checking fibre ", fibre," (only this fibre is corrected, use fibre = 0 for all)..."
#        #plot=True
#        #verbose = True
#        #warnings = True
#    else:
#        f_i=0
#        f_f=pox4ar_no.n_spectra #  self.n_spectra
#    for fibre in range(f_i,f_f): #    (self.n_spectra):
#        if fibre == say_status :
#            print "  Checking fibre ", fibre," ..."
#            say_status=say_status+100
#
#        # Gaussian fits to the sky spectrum
#        sl_gaussian_flux=[]
#        sl_gaussian_sigma=[]
#        sl_gauss_center=[]
#        skip_sl_fit=[]   # True emission line, False no emission line
#
#        j_lines = 0
#        el_low=el_low_list[j_lines]
#        el_high=el_high_list[j_lines]
#        sky_sl_gaussian_fitted = copy.deepcopy(sky)
#        if verbose: print "\n> Performing Gaussian fitting to sky lines in sky spectrum..."
#        for i in range(number_sl):
#            if sl_center[i] > el_high:
#                while sl_center[i] > el_high:
#                    j_lines = j_lines+1
#                    if j_lines < len(el_low_list) -1 :
#                        el_low=el_low_list[j_lines]
#                        el_high=el_high_list[j_lines]
#                        #print "Change to range ",el_low,el_high
#                    else:
#                        el_low = w[-1]+1
#                        el_high= w[-1]+2
#
#            if sl_fnl[i] == 0 :
#                plot_fit = False
#            else: plot_fit = True
#            resultado=fluxes(w, sky_sl_gaussian_fitted, sl_center[i], lowlow= sl_lowlow[i], lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh = sl_highhigh[i], lmin=sl_lmin[i], lmax=sl_lmax[i], fmin=0, fmax=0,
#                         broad=2.1, plot=plot_fit, verbose=False, plot_sus = False, fcal = False, warnings = warnings )   # Broad is FWHM for Gaussian sigma= 1,
#            sl_gaussian_flux.append(resultado[3])
#            sky_sl_gaussian_fitted=resultado[11]
#            sl_gauss_center.append(resultado[1])
#            sl_gaussian_sigma.append(resultado[5]/2.355)
#            if el_low < sl_center[i] < el_high:
#                if verbose: print '  SKY line',sl_center[i],'in EMISSION LINE !'
#                skip_sl_fit.append(True)
#            else: skip_sl_fit.append(False)
#
#            #print "  Fitted wavelength for sky line ",sl_center[i]," : ",resultado[1],"   ",resultado[5]
#            if plot_fit:
#                if verbose: print "  Fitted wavelength for sky line ",sl_center[i]," : ",sl_gauss_center[i],"  sigma = ",sl_gaussian_sigma[i]
#                wmin=sl_lmin[i]
#                wmax=sl_lmax[i]
#
#        # Gaussian fit to object spectrum
#        object_sl_gaussian_flux=[]
#        object_sl_gaussian_sigma=[]
#        ratio_object_sky_sl_gaussian = []
#        object_sl_gaussian_fitted = copy.deepcopy(spec)
#        object_sl_gaussian_center = []
#        if verbose: print "\n> Performing Gaussian fitting to sky lines in fibre",fibre," of object data..."
#
#        for i in range(number_sl):
#            if sl_fnl[i] == 0 :
#                plot_fit = False
#            else: plot_fit = True
#            if skip_sl_fit[i]:
#                if verbose: print" SKIPPING SKY LINE",sl_center[i]," as located within the range of an emission line!"
#                object_sl_gaussian_flux.append(float('nan'))   # The value of the SKY SPECTRUM
#                object_sl_gaussian_center.append(float('nan'))
#                object_sl_gaussian_sigma.append(float('nan'))
#            else:
#                resultado=fluxes(w, object_sl_gaussian_fitted, sl_center[i], lowlow= sl_lowlow[i], lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh = sl_highhigh[i], lmin=sl_lmin[i], lmax=sl_lmax[i], fmin=0, fmax=0,
#                             broad=2.1, plot=plot_fit, verbose=False, plot_sus = False, fcal = False, warnings = warnings )   # Broad is FWHM for Gaussian sigma= 1,
#                if resultado[3] > 0: # and resultado[5] < maxima_sigma: # -100000.: #0:
#                    if resultado[5] < maxima_sigma:
#                        use_sigma = sl_gaussian_sigma[i]
#                    else:
#                        use_sigma =resultado[5]
#                    object_sl_gaussian_flux.append(resultado[3])
#                    object_sl_gaussian_fitted=resultado[11]
#                    object_sl_gaussian_center.append(resultado[1])
#                    object_sl_gaussian_sigma.append(use_sigma)
#                else:
#                    if verbose: print "  Bad fit for ",sl_center[i],"! ignoring it..."
#                    object_sl_gaussian_flux.append(float('nan'))
#                    object_sl_gaussian_center.append(float('nan'))
#                    object_sl_gaussian_sigma.append(float('nan'))
#                    skip_sl_fit[i] = False   # We don't substract this fit
#            ratio_object_sky_sl_gaussian.append(object_sl_gaussian_flux[i]/sl_gaussian_flux[i])
#
#        # Scale sky lines that are located in emission lines or provided negative values in fit
#        reference_sl = 1 # Position in the file! Position 1 is sky line 6363.4
#        sl_ref_ratio = sl_gaussian_flux/sl_gaussian_flux[reference_sl]
#        for i in range(number_sl):
#            if skip_sl_fit[i] == True:
#                #print 'Tenemos que arreglar', sl_center[i]
#                flujo_mas_cercano_ = np.argsort(np.abs(sl_ref_ratio-sl_ref_ratio[i]))
#                mas_cercano = 1  # 0 is itself
#                while skip_sl_fit[flujo_mas_cercano_[mas_cercano]] == True:
#                    mas_cercano = mas_cercano + 1
#                gauss_fix = sl_gaussian_sigma[flujo_mas_cercano_[mas_cercano]]
#                flujo = sl_ref_ratio[i] / sl_ref_ratio[flujo_mas_cercano_[mas_cercano]] * sl_gaussian_flux[flujo_mas_cercano_[mas_cercano]]
#                if verbose: print "  Fixing ",sl_center[i],"with r=",sl_ref_ratio[i]," and gauss =",object_sl_gaussian_sigma[i]," using",sl_center[flujo_mas_cercano_[mas_cercano]], "\n with r=",  sl_ref_ratio[flujo_mas_cercano_[mas_cercano]], ",  f=",sl_gaussian_flux[mas_cercano]," assuming f=", flujo," gauss =",sl_gaussian_sigma[flujo_mas_cercano_[mas_cercano]]
#                object_sl_gaussian_fitted=substract_given_gaussian(w, object_sl_gaussian_fitted, sl_center[i], peak=0, sigma=gauss_fix,  flux=flujo,
#                                 lowlow= sl_lowlow[i], lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh = sl_highhigh[i], lmin=sl_lmin[i], lmax=sl_lmax[i], plot=False, verbose=verbose)
#
#
#    #    wmin,wmax = 6100,6500
#    #    ymin,ymax= -100,400
#    #
#    #    wmin,wmax = 6350,6700
#    #    wmin,wmax = 7100,7700
#    #    wmin,wmax = 7600,8200
#    #    wmin,wmax = 8110,8760
#    #    wmin,wmax = 7350,7500
#        ymin,ymax= -50,500
#
#        wmin,wmax=6100, 6500 #6700,7000 #6300,6450#7500
#
#
#        if plot:
#            plt.figure(figsize=(10, 4))
#            plt.plot()
#            plt.plot(w,spec,"y", alpha=0.7, label="Object")
#            plt.plot(w,object_sl_gaussian_fitted, "k", alpha=0.5, label="Obj - sky fitted")
#            plt.plot(w,sky_sl_gaussian_fitted, "r", alpha=0.5)
#            plt.plot(w,spec-sky,"g", alpha=0.5, label="Obj - sky")
#            plt.plot(w,object_sl_gaussian_fitted-sky_sl_gaussian_fitted,"b", alpha=0.9, label="Obj - sky fitted - rest sky")
#            plt.xlim(wmin,wmax)
#            plt.ylim(ymin,ymax)
#            ptitle = "Fibre "+np.str(fibre)#+" with rms = "+np.str(rms[i])
#            plt.title(ptitle)
#            plt.xlabel("Wavelength [$\AA$]")
#            plt.ylabel("Flux [counts]")
#            plt.legend(frameon=True, loc=2, ncol=4)
#            plt.minorticks_on()
#            for i in range(len(el_list)):
#                plt.axvline(x=el_list[i], color="k", linestyle='--',alpha=0.5)
#            for i in range(number_sl):
#                plt.axvline(x=sl_center[i], color="y", linestyle='--',alpha=0.6)
#            plt.show()
#            plt.close()
#
#
#        if verbose:
#            reference_sl = 1 # Position in the file!
#            sl_ref_ratio = sl_gaussian_flux/sl_gaussian_flux[reference_sl]
#            print "\n n  line     fsky    fspec   fspec/fsky   l_obj-l_sky  fsky/6363.4   sigma_sky  sigma_fspec"
#            for i in range(number_sl):
#                print "{:2} {:6.1f} {:8.2f} {:8.2f}    {:7.4f}      {:5.2f}      {:6.3f}    {:6.3f}  {:6.3f}" .format(i+1,sl_center[i],sl_gaussian_flux[i],object_sl_gaussian_flux[i],ratio_object_sky_sl_gaussian[i],object_sl_gaussian_center[i]-sl_gauss_center[i],sl_ref_ratio[i],sl_gaussian_sigma[i],object_sl_gaussian_sigma[i])
#            print "\n>  Median center offset between OBJ and SKY :", np.nanmedian(np.array(object_sl_gaussian_center)-np.array(sl_gauss_center)), "   Median gauss for the OBJECT ", np.nanmedian(object_sl_gaussian_sigma)
#
#
#


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#  Data 27 Feb 2018  BLUE
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#    path_main ="/DATA/KOALA/" #20180227/580V/"
#    date="20180227"
#    grating="580V"


# # ---------------------------------------------------------------------------
# # THROUGHPUT CORRECTION USING SKYFLAT
# # ---------------------------------------------------------------------------
# #
# # The very first thing that we need is to get the throughput correction.
# # We use a skyflat that has not been divided by a flatfield
# #
# # IF NEEDED: read normalized flatfield:
#    path_flat = path_main+date+"/skyflats/"+grating
#    file_flatb=path_flat+"_bad/27feb1_combined.fits"
#    flat_blue = KOALA_RSS(file_flatb, sky_method="none", do_extinction=False, apply_throughput=False, plot=True)
#                        # correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 100, step_ccd = 50)
# #
# # Provide skyflat file and name of the file that will keep the throughput correction
#    path_skyflat = path_main+date+"/skyflats/"+grating
#    file_skyflatb=path_skyflat+"/27feb1_combined.fits"
#    throughput_file_blue=path_skyflat+"/"+date+"_"+grating+"_throughput_correction.dat"
# #
# # If this has been done before, we can read the file containing the throughput correction
#    throughput_blue = read_table(throughput_file_blue, ["f"] )
# #
# # Now we read the RSS file, we ONLY correct for ccd defects and high cosmics
#    skyflat_blue = KOALA_RSS(file_skyflatb, flat="", apply_throughput=False, sky_method="none",
#                             do_extinction=False, correct_ccd_defects = False,
#                             correct_high_cosmics = False, clip_high = 100, step_ccd = 50, plot=True)
# #
# # Next we find the relative throughput.
# # If the data have been normalized by the FLATFIELD, we only need a SCALE between fibres,
# # We consider the median value in the range [wave_min_scale, wave_max_scale] for all fibres and scale
# #
#    skyflat_blue.find_relative_throughput(ymin=0, ymax=1000000) #,  wave_min_scale=6630, wave_max_scale=6820)
# #
# # The relative throughput is an array stored in skyflat_red.relative_throughput
# # We save that array in a text file that we can read in the future without the need of repeating this
#    array_to_text_file(skyflat_blue.relative_throughput, filename= throughput_file_blue )
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
# # READ FLUX CALIBRATION BLUE
#    flux_cal_file=path_main+"flux_calibration_20161024_580V_0p6_1k25.dat"
#    w_star,flux_calibration_20161024_580V_0p6_1k25 = read_table(flux_cal_file, ["f", "f"] )
#    print flux_calibration_20161024_580V_0p6_1k25
#
# # TELLURIC CORRECTION FROM FILE NOT NEEDED FOR BLUE


# # READ STAR 1
#    star1="H600"
#    path_star1 = path_main+date+"/"+grating+"/"
#    starpos1b = path_star1+"27feb10030red.fits"
#    fits_file_blue = path_star1+"/"+star1+".fits"
#    text_file_blue = path_star1+"/"+star1+"_response.dat"

# Now we read RSS file
# We apply throughput correction, substract sky using n_sky=600 lowest intensity fibres,
# correct for CCD defects and high cosmics

#    star1b = KOALA_RSS(starpos1b,
#                       apply_throughput=True, skyflat = skyflat_blue, plot_skyflat=False,
#                       sky_method="self", n_sky=100,
#                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
#                       valid_wave_min = 3770, valid_wave_max = 5799,
#                       plot=True, warnings=True)

# # PAY ATTENTION to plots. In this case part of the star is in a DEAD FIBRE !!!
# # But we proceed anyway

# # Next we CREATE THE CUBE for this star, using THE SAME PARAMETERS we will later using for our objects
# # 0.6 is the pixel size, 1.25 is the kernel size.
#    cubes1b=Interpolated_cube(star1b, .6, 1.25, plot=True)

# # Now we read the absolute flux calibration data of the calibration star and get the response curve
# # (Response curve: correspondence between counts and physical values)
# # Include exp_time of the calibration star, as the results are given per second
# # For this BE CAREFUL WITH ABSORPTIONS (Halpha) and check behaviour in the edges of the CCD
#    cubes1b.do_response_curve('FLUX_CAL/fhilt600_edited.dat', plot=True, min_wave=3650., max_wave=5750.,
#                              step=10, exp_time=120., fit_degree=5)

# # Now we can save this calibration as a text file
#    spectrum_to_text_file(cubes1b.response_wavelength,cubes1b.response_curve, filename=text_file_blue)


# # REPEAT FOR STAR 2

#    star2="H600"
#    path_star2 = path_main+"H600/"
#    starpos2b = path_star2+"24oct10061red.fits"
#    fits_file_blue = path_star2+"/"+star2+".fits"
#    text_file_blue = path_star2+"/"+star2+"_response.dat"
#
#    star2b = KOALA_RSS(starpos2b,
#                       apply_throughput=True, skyflat = skyflat_blue, plot_skyflat=False,
#                       sky_method="self", n_sky=400,
#                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
#                       valid_wave_min = 3770, valid_wave_max = 5799,
#                       plot=True, warnings=True)
#
#    cubes2b=Interpolated_cube(star2b, .6, 1.25, plot=True)
#    cubes2b.do_response_curve('FLUX_CAL/fhilt600_edited.dat', plot=True, min_wave=3800., max_wave=5750.,
#                              step=10, exp_time=120., fit_degree=3)
#    spectrum_to_text_file(cubes2b.response_wavelength,cubes2b.response_curve, filename=text_file_blue)

# # REPEAT FOR STAR 3

#    star3="LTT2415"
#    path_star3 = path_main+"LTT2415/"   #+star1+"/"+date+"/"+grating+"/"
#    starpos3b = path_star3+"24oct10059red.fits"
#    fits_file_blue = path_star3+"/"+star3+".fits"#+star1+"_"+grating+"_"+date+".fits"
#    text_file_blue = path_star3+"/"+star3+"_response.dat" #+star1+"_"+grating+"_"+date+"_response.dat"
#
#    star3b = KOALA_RSS(starpos3b,
#                       apply_throughput=True, skyflat = skyflat_blue, plot_skyflat=False,
#                       sky_method="self", n_sky=600,
#                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
#                       valid_wave_min = 3770, valid_wave_max = 5799,
#                       plot=True, warnings=True)
#
#    cubes3b=Interpolated_cube(star3b, .6, 1.25, plot=True)
#    cubes3b.do_response_curve('FLUX_CAL/fltt2415_edited.dat', plot=True, min_wave=3800., max_wave=5750., step=10, exp_time=180., fit_degree=3)
#    spectrum_to_text_file(cubes3b.response_wavelength,cubes3b.response_curve, filename=text_file_blue)


# #  CHECK AND GET THE FLUX CALIBRATION FOR THE NIGHT  RED

# #  First we take another look to the RSS data ploting the integrated fibre values in a map
#    star1b.RSS_map(star1b.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))  # Dead fibre!!!
#    star2b.RSS_map(star2b.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))
#    star3b.RSS_map(star3b.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))

# # We check again that star1 is on a dead fibre, we don't use this star for absolute flux calibration

# # Define in "stars" the 2 cubes we are using, and plotting their responses to check
#    stars=[cubes1b] #,cubes2b,cubes3b]
#    plot_response(stars)

# # The shape of the curves are OK but they have a variation of ~3% in flux...
# # Probably this would have been corrected obtaining at least TWO exposures per star..
# # Anyway, that is what it is, we obtain the flux calibration applying:
#    flux_calibration_20180227_580V_0p6_1k25=obtain_flux_calibration(stars)

# # And we save this absolute flux calibration as a text file
#    flux_correction_file = path_main+"flux_calibration_20161024_580V_0p6_1k25.dat"
#    spectrum_to_text_file(star1b.wavelength,flux_calibration_20180227_580V_0p6_1k25, filename=flux_correction_file)

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
#    file_sky_b1 = path_main+"SKY/24oct10045red.fits"
#    file_sky_b2 = path_main+"SKY/24oct10048red.fits"
#    file_sky_b3 = path_main+"SKY/24oct10051red.fits"
#    file_sky_b4 = path_main+"SKY/24oct10054red.fits"
# #
#    sky_b1 = KOALA_RSS(file_sky_b1, apply_throughput=True, skyflat = skyflat_blue, do_extinction=True,
#                       sky_method="none", is_sky=True, win_sky=151,
#                       correct_ccd_defects = False, correct_high_cosmics = True, clip_high = 100, step_ccd = 50,
#                       plot=True)
#    sky_b2 = KOALA_RSS(file_sky_b2, apply_throughput=True, skyflat = skyflat_blue, do_extinction=True,
#                       sky_method="none", is_sky=True, win_sky=151,
#                       correct_ccd_defects = False, correct_high_cosmics = True, clip_high = 100, step_ccd = 50,
#                       plot=True)
#    sky_b3 = KOALA_RSS(file_sky_b3, apply_throughput=True, skyflat = skyflat_blue, do_extinction=True,
#                       sky_method="none", is_sky=True, win_sky=151,
#                       correct_ccd_defects = False, correct_high_cosmics = True, clip_high = 100, step_ccd = 50,
#                       plot=True)
#    sky_b4 = KOALA_RSS(file_sky_b4, apply_throughput=True, skyflat = skyflat_blue, do_extinction=True,
#                       sky_method="none", is_sky=True, win_sky=151,
#                       correct_ccd_defects = False, correct_high_cosmics = True, clip_high = 100, step_ccd = 50,
#                       plot=True)
# #
# #
# #
# # ---------------------------------------------------------------------------
# # TIME FOR THE OBJECT !!
# # ---------------------------------------------------------------------------
# #
# #
# #
#

#    OBJECT = "He2-10"
#    DESCRIPTION = "He2-10 CUBE"
#
#    file1b=path_main+date+"/580V/27feb10031red.fits"
#    file2b=path_main+date+"/580V/27feb10032red.fits"
#    file3b=path_main+date+"/580V/27feb10033red.fits"
#
#    rss_list=[file1b,file2b,file3b] #,file4b,file5b,file6b,file7b]
#    #sky_list=[sky_b1,sky_b1,sky_b2,sky_b2,sky_b3,sky_b3,sky_b4]
#    fits_file_blue=path_main+date+"/580V/"+OBJECT+"/He2-10_A_blue_combined_cube_3.fits"
##
#    hikids_blue = KOALA_reduce(rss_list, fits_file=fits_file_blue, obj_name=OBJECT,  description = DESCRIPTION,
#                          apply_throughput=True, skyflat = skyflat_blue, plot_skyflat=False,
#                          correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 50, step_ccd=50,
#                          remove_5578 = True,
#                          do_extinction=True,
#                          sky_method="self", #sky_list=sky_list,
#                          #id_el=True, high_fibres=10, brightest_line="Hb", cut=1.1, broad=3.0, plot_id_el= False,
#                          #id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6716.47, 6730.85],
#                          #clean_sky_residuals = False, dclip=2.0, extra_w = 1.3, step_csr = 100, # fibre=427,
#                          pixel_size_arcsec=.6, kernel_size_arcsec=1.25,
##                          offsets=[-0.197,1.5555,   2.8847547360544095, 1.4900605034042318],
##                          offsets=[2.1661863268521699, -1.0154564185927739,
##                                     -1.2906911413228896, -0.45669081905592757,
##                                     -0.07410120886509336, 1.7307043173337093,
##                                     0.80060388230072721, -0.79741816283239464,
##                                     2.4503657941496169, -0.74336415384485555,
##                                     -1.277432180113331, -0.38498639829597181],
#                          ADR=False,
#                          flux_calibration=flux_calibration_20180227_580V_0p6_1k25,
#
#                          valid_wave_min = 3770, valid_wave_max = 5799,
#                          plot=True, warnings=False)


# # First we provide all files:

#    OBJECT = "He2-10"
#    DESCRIPTION = "He2-10 B CUBE"
#
#    file1b=path_main+date+"/580V/27feb10034red.fits"
#    file2b=path_main+date+"/580V/27feb10035red.fits"
#    file3b=path_main+date+"/580V/27feb10036red.fits"
#
##    file4b=path_main+date+"/580V_bad/27feb10031red.fits"
##    file5b=path_main+date+"/580V_bad/27feb10032red.fits"
##    file6b=path_main+date+"/580V_bad/27feb10033red.fits"
##    file4b=path_main+"OBJECT/24oct10049red.fits"
##    file5b=path_main+"OBJECT/24oct10050red.fits"
##    file6b=path_main+"OBJECT/24oct10052red.fits"
##    file7b=path_main+"OBJECT/24oct10053red.fits"
##
##
#    rss_list=[file1b,file2b,file3b]#,file4b,file5b,file6b]# ,file7b]
#    #sky_list=[sky_b1,sky_b1,sky_b2,sky_b2,sky_b3,sky_b3,sky_b4]
#    fits_file_blue=path_main+date+"/580V/"+OBJECT+"/He2-10_B_blue_combined_cube_3.fits"
##
#    hikids_blue = KOALA_reduce(rss_list, fits_file=fits_file_blue, obj_name=OBJECT,  description=DESCRIPTION,
#                          apply_throughput=True, skyflat = skyflat_blue, plot_skyflat=False,
#                          correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 50, step_ccd=50,
#                          remove_5578 = True,
#                          do_extinction=True,
#                          sky_method="self", #sky_list=sky_list,
#                          #id_el=True, high_fibres=10, brightest_line="Hb", cut=1.1, broad=3.0, plot_id_el= False,
#                          #id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6716.47, 6730.85],
#                          #clean_sky_residuals = False, dclip=2.0, extra_w = 1.3, step_csr = 100, # fibre=427,
#                          pixel_size_arcsec=.6, kernel_size_arcsec=1.25,
#                          offsets=[-1.5,0,    -1.5, -1.5],
##                          offsets=[2.1661863268521699, -1.0154564185927739,
##                                     -1.2906911413228896, -0.45669081905592757,
##                                     -0.07410120886509336, 1.7307043173337093,
##                                     0.80060388230072721, -0.79741816283239464,
##                                     2.4503657941496169, -0.74336415384485555,
##                                     -1.277432180113331, -0.38498639829597181],
#                          ADR=False,
#                          flux_calibration=flux_calibration_20180227_580V_0p6_1k25,
#
#                          valid_wave_min = 3635, valid_wave_max = 5758,
#                          plot=True, warnings=False)
#


# # OBJECT: ASAS15feb

#    OBJECT = "ASAS15feb"
#    DESCRIPTION = "ASAS15feb CUBE"
#
#    file1b=path_main+date+"/580V/27feb10037red.fits"
#    file2b=path_main+date+"/580V/27feb10038red.fits"
#    file3b=path_main+date+"/580V/27feb10039red.fits"
#
#    rss_list=[file1b,file2b,file3b]#,file4b,file5b,file6b]# ,file7b]
#    #sky_list=[sky_b1,sky_b1,sky_b2,sky_b2,sky_b3,sky_b3,sky_b4]
#    fits_file_blue=path_main+date+"/580V/"+OBJECT+"/ASAS15feb_blue_combined_cube.fits"
##
#    hikids_blue = KOALA_reduce(rss_list, fits_file=fits_file_blue, obj_name=OBJECT,  description=DESCRIPTION,
#                          apply_throughput=True, skyflat = skyflat_blue, plot_skyflat=False,
#                          correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
#                          remove_5578 = True,
#                          do_extinction=True,
#                          sky_method="self", #sky_list=sky_list,
#                          #id_el=True, high_fibres=10, brightest_line="Hb", cut=1.1, broad=3.0, plot_id_el= False,
#                          #id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6716.47, 6730.85],
#                          #clean_sky_residuals = False, dclip=2.0, extra_w = 1.3, step_csr = 100, # fibre=427,
#                          pixel_size_arcsec=.6, kernel_size_arcsec=1.25,
#                          #offsets=[-1.5,0,    -1.5, -1.5],
##                          offsets=[2.1661863268521699, -1.0154564185927739,        # E-W, S-N
##                                     -1.2906911413228896, -0.45669081905592757,
##                                     -0.07410120886509336, 1.7307043173337093,
##                                     0.80060388230072721, -0.79741816283239464,
##                                     2.4503657941496169, -0.74336415384485555,
##                                     -1.277432180113331, -0.38498639829597181],
#                          ADR=False,
#                          flux_calibration=flux_calibration_20180227_580V_0p6_1k25,
#
#                          valid_wave_min = 3635, valid_wave_max = 5758,
#                          plot=True, warnings=False)

#    OBJECT = "SN2012bo"
#    DESCRIPTION = "SN2012bo CUBE"
#
#    file1b=path_main+date+"/580V/27feb10043red.fits"
#    file2b=path_main+date+"/580V/27feb10044red.fits"
#    file3b=path_main+date+"/580V/27feb10045red.fits"
#
#    rss_list=[file1b,file2b,file3b]#,file4b,file5b,file6b]# ,file7b]
#    #sky_list=[sky_b1,sky_b1,sky_b2,sky_b2,sky_b3,sky_b3,sky_b4]
#    fits_file_blue=path_main+date+"/580V/"+OBJECT+"/SN2012bo_blue_combined_cube.fits"
##
#    hikids_blue = KOALA_reduce(rss_list, fits_file=fits_file_blue, obj_name=OBJECT,  description=DESCRIPTION,
#                          apply_throughput=True, skyflat = skyflat_blue, plot_skyflat=False,
#                          correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
#                          remove_5578 = True,
#                          do_extinction=True,
#                          sky_method="self", #sky_list=sky_list,
#                          #id_el=True, high_fibres=10, brightest_line="Hb", cut=1.1, broad=3.0, plot_id_el= False,
#                          #id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6716.47, 6730.85],
#                          #clean_sky_residuals = False, dclip=2.0, extra_w = 1.3, step_csr = 100, # fibre=427,
#                          pixel_size_arcsec=.6, kernel_size_arcsec=1.25,
#                          #offsets=[-1.5,0,    -1.5, -1.5],
##                          offsets=[2.1661863268521699, -1.0154564185927739,        # -E+W, -S+N
##                                     -1.2906911413228896, -0.45669081905592757,
##                                     -0.07410120886509336, 1.7307043173337093,
##                                     0.80060388230072721, -0.79741816283239464,
##                                     2.4503657941496169, -0.74336415384485555,
##                                     -1.277432180113331, -0.38498639829597181],
#                          ADR=False,
#                          flux_calibration=flux_calibration_20180227_580V_0p6_1k25,
#
#                          valid_wave_min = 3635, valid_wave_max = 5758,
#                          plot=True, warnings=False)

end = timer()
print "\n> Elapsing time = ", end - start, "s"
# -----------------------------------------------------------------------------
#                                     ... Paranoy@ Rulz! ;^D  & Angel R. :-)
# -----------------------------------------------------------------------------
