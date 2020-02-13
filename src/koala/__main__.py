from koala import *

if __name__ == "__main__":

    print "\n> Testing KOALA RSS class. Running", version

    # -----------------------------------------------------------------------------
    #    MACRO FOR EVERYTHING 19 Sep 2019, including alignment 2-10 cubes
    # -----------------------------------------------------------------------------

    class KOALA_reduce(RSS, Interpolated_cube):  # TASK_KOALA_reduce
        def __init__(
            self,
            rss_list,
            fits_file="",
            obj_name="",
            description="",
            do_rss=True,
            do_cubing=True,
            do_alignment=True,
            make_combined_cube=True,
            rss_clean=False,
            save_rss_to_fits_file_list=["", "", "", "", "", "", "", "", "", ""],
            save_aligned_cubes=False,
            # RSS
            # skyflat_file is a RSS, skyflat and skyflat_list are the names of objects keeping the relative throughput of skyflats
            apply_throughput=True,
            skyflat="",
            skyflat_file="",
            flat="",
            skyflat_list=["", "", "", "", "", "", "", "", "", ""],
            # This line is needed if doing FLAT when reducing (NOT recommended)
            plot_skyflat=False,
            wave_min_scale=0,
            wave_max_scale=0,
            ymin=0,
            ymax=0,
            # Correct CCD defects & high cosmics
            correct_ccd_defects=False,
            correct_high_cosmics=False,
            clip_high=100,
            step_ccd=50,
            remove_5578=False,
            plot_suspicious_fibres=False,
            # Correct for small shofts in wavelength
            fix_wavelengths=False,
            sol=[0, 0, 0],
            # Correct for extinction
            do_extinction=True,
            # Sky substraction
            sky_method="self",
            n_sky=50,
            sky_fibres=[1000],  # do_sky=True
            sky_spectrum=[0],
            sky_rss=[0],
            scale_sky_rss=0,
            scale_sky_1D=0,
            correct_negative_sky=False,
            auto_scale_sky=False,
            sky_wave_min=0,
            sky_wave_max=0,
            cut_sky=5.0,
            fmin=1,
            fmax=10,
            individual_sky_substraction=False,
            fibre_list=[100, 200, 300, 400, 500, 600, 700, 800, 900],
            sky_list=[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
            # Telluric correction
            telluric_correction=[0],
            telluric_correction_list=[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
            # Identify emission lines
            id_el=False,
            high_fibres=10,
            brightest_line="Ha",
            cut=1.5,
            plot_id_el=True,
            broad=2.0,
            id_list=[0],
            brightest_line_wavelength=0,
            # Clean sky residuals
            clean_sky_residuals=False,
            dclip=3.0,
            extra_w=1.3,
            step_csr=25,
            # CUBING
            pixel_size_arcsec=0.4,
            kernel_size_arcsec=1.2,
            offsets=[1000],
            ADR=False,
            flux_calibration=[0],
            flux_calibration_list=[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
            # COMMON TO RSS AND CUBING
            valid_wave_min=0,
            valid_wave_max=0,
            plot=True,
            norm=colors.LogNorm(),
            fig_size=12,
            warnings=False,
            verbose=False,
        ):
            """
            Example
            -------
            >>>  combined_KOALA_cube(['Ben/16jan10049red.fits','Ben/16jan10050red.fits','Ben/16jan10051red.fits'], 
                    fits_file="test_BLUE_reduced.fits", skyflat_file='Ben/16jan10086red.fits', 
                    pixel_size_arcsec=.3, kernel_size_arcsec=1.5, flux_calibration=flux_calibration, 
                    plot= True)    
            """

            print "\n\n\n======================= REDUCING KOALA data =======================\n\n"

            n_files = len(rss_list)
            sky_rss_list = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
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

            print "  1. Checking input values: "

            print "\n  - Using the following RSS files : "
            for rss in range(n_files):
                print "    ", rss + 1, ". : ", rss_list[rss]
            self.rss_list = rss_list

            if rss_clean:
                print "\n  - These RSS files are ready to be cubed & combined, no further process required ..."

            else:

                if skyflat == "" and skyflat_list[0] == "" and skyflat_file == "":
                    print "\n  - No skyflat file considered, no throughput correction will be applied."
                else:
                    if skyflat_file == "":
                        print "\n  - Using skyflat to consider throughput correction ..."
                        if skyflat != "":
                            for i in range(n_files):
                                skyflat_list[i] = skyflat
                            print "    Using same skyflat for all object files"
                        else:
                            print "    List of skyflats provided!"
                    else:
                        print "\n  - Using skyflat file to derive the throughput correction ..."  # This assumes skyflat_file is the same for all the objects
                        skyflat = KOALA_RSS(
                            skyflat_file,
                            do_sky=False,
                            do_extinction=False,
                            apply_throughput=False,
                            plot=True,
                        )

                        skyflat.find_relative_throughput(
                            ymin=ymin,
                            ymax=ymax,
                            wave_min_scale=wave_min_scale,
                            wave_max_scale=wave_max_scale,
                            plot=plot_skyflat,
                        )

                        for i in range(n_files):
                            skyflat_list[i] = skyflat
                        print "  - Using same skyflat for all object files"

                # sky_method = "self" "1D" "2D" "none" #1Dfit"

                if sky_method == "1D" or sky_method == "1Dfit":
                    if np.nanmedian(sky_spectrum) != 0:
                        for i in range(n_files):
                            sky_list[i] = sky_spectrum
                        print "\n  - Using same 1D sky spectrum provided for all object files"
                    else:
                        if np.nanmedian(sky_list[0]) == 0:
                            print "\n  - 1D sky spectrum requested but not found, assuming n_sky = 50 from the same files"
                            sky_method = "self"
                        else:
                            print "\n  - List of 1D sky spectrum provided for each object file"

                if sky_method == "2D":
                    try:
                        if np.nanmedian(sky_list[0].intensity_corrected) != 0:
                            print "\n  - List of 2D sky spectra provided for each object file"
                            for i in range(n_files):
                                sky_rss_list[i] = sky_list[i]
                                sky_list[i] = [0]
                    except Exception:
                        try:
                            if sky_rss == 0:
                                print "\n  - 2D sky spectra requested but not found, assuming n_sky = 50 from the same files"
                                sky_method = "self"
                        except Exception:
                            for i in range(n_files):
                                sky_rss_list[i] = sky_rss
                            print "\n  - Using same 2D sky spectra provided for all object files"

                if sky_method == "self":
                    for i in range(n_files):
                        sky_list[i] = 0
                    if n_sky == 0:
                        n_sky = 50
                    if sky_fibres[0] == 1000:
                        print "\n  - Using n_sky =", n_sky, "to create a sky spectrum"
                    else:
                        print "\n  - Using n_sky =", n_sky, "and sky_fibres =", sky_fibres, "to create a sky spectrum"

                if (
                    np.nanmedian(telluric_correction) == 0
                    and np.nanmedian(telluric_correction_list[0]) == 0
                ):
                    print "\n  - No telluric correction considered"
                else:
                    if np.nanmedian(telluric_correction_list[0]) == 0:
                        for i in range(n_files):
                            telluric_correction_list[i] = telluric_correction
                        print "\n  - Using same telluric correction for all object files"
                    else:
                        print "\n  - List of telluric corrections provided!"

            if do_rss:
                print "\n  2. Reading the data stored in rss files ..."
                self.rss1 = KOALA_RSS(
                    rss_list[0],
                    rss_clean=rss_clean,
                    save_rss_to_fits_file=save_rss_to_fits_file_list[0],
                    apply_throughput=apply_throughput,
                    skyflat=skyflat_list[0],
                    plot_skyflat=plot_skyflat,
                    correct_ccd_defects=correct_ccd_defects,
                    correct_high_cosmics=correct_high_cosmics,
                    clip_high=clip_high,
                    step_ccd=step_ccd,
                    remove_5578=remove_5578,
                    plot_suspicious_fibres=plot_suspicious_fibres,
                    fix_wavelengths=fix_wavelengths,
                    sol=sol,
                    do_extinction=do_extinction,
                    scale_sky_1D=scale_sky_1D,
                    correct_negative_sky=correct_negative_sky,
                    sky_method=sky_method,
                    n_sky=n_sky,
                    sky_fibres=sky_fibres,
                    sky_spectrum=sky_list[0],
                    sky_rss=sky_rss_list[0],
                    brightest_line_wavelength=brightest_line_wavelength,
                    cut_sky=cut_sky,
                    fmin=fmin,
                    fmax=fmax,
                    individual_sky_substraction=individual_sky_substraction,
                    telluric_correction=telluric_correction_list[0],
                    id_el=id_el,
                    high_fibres=high_fibres,
                    brightest_line=brightest_line,
                    cut=cut,
                    broad=broad,
                    plot_id_el=plot_id_el,
                    id_list=id_list,
                    clean_sky_residuals=clean_sky_residuals,
                    dclip=dclip,
                    extra_w=extra_w,
                    step_csr=step_csr,
                    valid_wave_min=valid_wave_min,
                    valid_wave_max=valid_wave_max,
                    warnings=warnings,
                    verbose=verbose,
                    plot=plot,
                    norm=norm,
                    fig_size=fig_size,
                )

                if len(rss_list) > 1:
                    self.rss2 = KOALA_RSS(
                        rss_list[1],
                        rss_clean=rss_clean,
                        save_rss_to_fits_file=save_rss_to_fits_file_list[1],
                        apply_throughput=apply_throughput,
                        skyflat=skyflat_list[1],
                        plot_skyflat=plot_skyflat,
                        correct_ccd_defects=correct_ccd_defects,
                        correct_high_cosmics=correct_high_cosmics,
                        clip_high=clip_high,
                        step_ccd=step_ccd,
                        remove_5578=remove_5578,
                        plot_suspicious_fibres=plot_suspicious_fibres,
                        fix_wavelengths=fix_wavelengths,
                        sol=sol,
                        do_extinction=do_extinction,
                        scale_sky_1D=scale_sky_1D,
                        correct_negative_sky=correct_negative_sky,
                        sky_method=sky_method,
                        n_sky=n_sky,
                        sky_fibres=sky_fibres,
                        sky_spectrum=sky_list[1],
                        sky_rss=sky_rss_list[1],
                        brightest_line_wavelength=brightest_line_wavelength,
                        cut_sky=cut_sky,
                        fmin=fmin,
                        fmax=fmax,
                        individual_sky_substraction=individual_sky_substraction,
                        telluric_correction=telluric_correction_list[1],
                        id_el=id_el,
                        high_fibres=high_fibres,
                        brightest_line=brightest_line,
                        cut=cut,
                        broad=broad,
                        plot_id_el=plot_id_el,
                        id_list=id_list,
                        clean_sky_residuals=clean_sky_residuals,
                        dclip=dclip,
                        extra_w=extra_w,
                        step_csr=step_csr,
                        valid_wave_min=valid_wave_min,
                        valid_wave_max=valid_wave_max,
                        warnings=warnings,
                        verbose=verbose,
                        plot=plot,
                        norm=norm,
                        fig_size=fig_size,
                    )

                if len(rss_list) > 2:
                    self.rss3 = KOALA_RSS(
                        rss_list[2],
                        rss_clean=rss_clean,
                        save_rss_to_fits_file=save_rss_to_fits_file_list[2],
                        apply_throughput=apply_throughput,
                        skyflat=skyflat_list[2],
                        plot_skyflat=plot_skyflat,
                        correct_ccd_defects=correct_ccd_defects,
                        correct_high_cosmics=correct_high_cosmics,
                        clip_high=clip_high,
                        step_ccd=step_ccd,
                        remove_5578=remove_5578,
                        plot_suspicious_fibres=plot_suspicious_fibres,
                        fix_wavelengths=fix_wavelengths,
                        sol=sol,
                        do_extinction=do_extinction,
                        scale_sky_1D=scale_sky_1D,
                        correct_negative_sky=correct_negative_sky,
                        sky_method=sky_method,
                        n_sky=n_sky,
                        sky_fibres=sky_fibres,
                        sky_spectrum=sky_list[2],
                        sky_rss=sky_rss_list[2],
                        brightest_line_wavelength=brightest_line_wavelength,
                        cut_sky=cut_sky,
                        fmin=fmin,
                        fmax=fmax,
                        individual_sky_substraction=individual_sky_substraction,
                        telluric_correction=telluric_correction_list[2],
                        id_el=id_el,
                        high_fibres=high_fibres,
                        brightest_line=brightest_line,
                        cut=cut,
                        broad=broad,
                        plot_id_el=plot_id_el,
                        id_list=id_list,
                        clean_sky_residuals=clean_sky_residuals,
                        dclip=dclip,
                        extra_w=extra_w,
                        step_csr=step_csr,
                        valid_wave_min=valid_wave_min,
                        valid_wave_max=valid_wave_max,
                        warnings=warnings,
                        verbose=verbose,
                        plot=plot,
                        norm=norm,
                        fig_size=fig_size,
                    )

                if len(rss_list) > 3:
                    self.rss4 = KOALA_RSS(
                        rss_list[3],
                        rss_clean=rss_clean,
                        save_rss_to_fits_file=save_rss_to_fits_file_list[3],
                        apply_throughput=apply_throughput,
                        skyflat=skyflat_list[3],
                        plot_skyflat=plot_skyflat,
                        correct_ccd_defects=correct_ccd_defects,
                        correct_high_cosmics=correct_high_cosmics,
                        clip_high=clip_high,
                        step_ccd=step_ccd,
                        remove_5578=remove_5578,
                        plot_suspicious_fibres=plot_suspicious_fibres,
                        fix_wavelengths=fix_wavelengths,
                        sol=sol,
                        do_extinction=do_extinction,
                        scale_sky_1D=scale_sky_1D,
                        correct_negative_sky=correct_negative_sky,
                        sky_method=sky_method,
                        n_sky=n_sky,
                        sky_fibres=sky_fibres,
                        sky_spectrum=sky_list[3],
                        sky_rss=sky_rss_list[3],
                        brightest_line_wavelength=brightest_line_wavelength,
                        cut_sky=cut_sky,
                        fmin=fmin,
                        fmax=fmax,
                        individual_sky_substraction=individual_sky_substraction,
                        telluric_correction=telluric_correction_list[3],
                        id_el=id_el,
                        high_fibres=high_fibres,
                        brightest_line=brightest_line,
                        cut=cut,
                        broad=broad,
                        plot_id_el=plot_id_el,
                        id_list=id_list,
                        clean_sky_residuals=clean_sky_residuals,
                        dclip=dclip,
                        extra_w=extra_w,
                        step_csr=step_csr,
                        valid_wave_min=valid_wave_min,
                        valid_wave_max=valid_wave_max,
                        warnings=warnings,
                        verbose=verbose,
                        plot=plot,
                        norm=norm,
                        fig_size=fig_size,
                    )

                if len(rss_list) > 4:
                    self.rss5 = KOALA_RSS(
                        rss_list[4],
                        rss_clean=rss_clean,
                        save_rss_to_fits_file=save_rss_to_fits_file_list[4],
                        apply_throughput=apply_throughput,
                        skyflat=skyflat_list[4],
                        plot_skyflat=plot_skyflat,
                        correct_ccd_defects=correct_ccd_defects,
                        correct_high_cosmics=correct_high_cosmics,
                        clip_high=clip_high,
                        step_ccd=step_ccd,
                        remove_5578=remove_5578,
                        plot_suspicious_fibres=plot_suspicious_fibres,
                        fix_wavelengths=fix_wavelengths,
                        sol=sol,
                        do_extinction=do_extinction,
                        scale_sky_1D=scale_sky_1D,
                        correct_negative_sky=correct_negative_sky,
                        sky_method=sky_method,
                        n_sky=n_sky,
                        sky_fibres=sky_fibres,
                        sky_spectrum=sky_list[4],
                        sky_rss=sky_rss_list[4],
                        brightest_line_wavelength=brightest_line_wavelength,
                        cut_sky=cut_sky,
                        fmin=fmin,
                        fmax=fmax,
                        individual_sky_substraction=individual_sky_substraction,
                        telluric_correction=telluric_correction_list[4],
                        id_el=id_el,
                        high_fibres=high_fibres,
                        brightest_line=brightest_line,
                        cut=cut,
                        broad=broad,
                        plot_id_el=plot_id_el,
                        id_list=id_list,
                        clean_sky_residuals=clean_sky_residuals,
                        dclip=dclip,
                        extra_w=extra_w,
                        step_csr=step_csr,
                        valid_wave_min=valid_wave_min,
                        valid_wave_max=valid_wave_max,
                        warnings=warnings,
                        verbose=verbose,
                        plot=plot,
                        norm=norm,
                        fig_size=fig_size,
                    )

                if len(rss_list) > 5:
                    self.rss6 = KOALA_RSS(
                        rss_list[5],
                        rss_clean=rss_clean,
                        save_rss_to_fits_file=save_rss_to_fits_file_list[5],
                        apply_throughput=apply_throughput,
                        skyflat=skyflat_list[5],
                        plot_skyflat=plot_skyflat,
                        correct_ccd_defects=correct_ccd_defects,
                        correct_high_cosmics=correct_high_cosmics,
                        clip_high=clip_high,
                        step_ccd=step_ccd,
                        remove_5578=remove_5578,
                        plot_suspicious_fibres=plot_suspicious_fibres,
                        fix_wavelengths=fix_wavelengths,
                        sol=sol,
                        do_extinction=do_extinction,
                        scale_sky_1D=scale_sky_1D,
                        correct_negative_sky=correct_negative_sky,
                        sky_method=sky_method,
                        n_sky=n_sky,
                        sky_fibres=sky_fibres,
                        sky_spectrum=sky_list[5],
                        sky_rss=sky_rss_list[5],
                        brightest_line_wavelength=brightest_line_wavelength,
                        cut_sky=cut_sky,
                        fmin=fmin,
                        fmax=fmax,
                        individual_sky_substraction=individual_sky_substraction,
                        telluric_correction=telluric_correction_list[5],
                        id_el=id_el,
                        high_fibres=high_fibres,
                        brightest_line=brightest_line,
                        cut=cut,
                        broad=broad,
                        plot_id_el=plot_id_el,
                        id_list=id_list,
                        clean_sky_residuals=clean_sky_residuals,
                        dclip=dclip,
                        extra_w=extra_w,
                        step_csr=step_csr,
                        valid_wave_min=valid_wave_min,
                        valid_wave_max=valid_wave_max,
                        warnings=warnings,
                        verbose=verbose,
                        plot=plot,
                        norm=norm,
                        fig_size=fig_size,
                    )

                if len(rss_list) > 6:
                    self.rss7 = KOALA_RSS(
                        rss_list[6],
                        rss_clean=rss_clean,
                        save_rss_to_fits_file=save_rss_to_fits_file_list[6],
                        apply_throughput=apply_throughput,
                        skyflat=skyflat_list[6],
                        plot_skyflat=plot_skyflat,
                        correct_ccd_defects=correct_ccd_defects,
                        correct_high_cosmics=correct_high_cosmics,
                        clip_high=clip_high,
                        step_ccd=step_ccd,
                        remove_5578=remove_5578,
                        plot_suspicious_fibres=plot_suspicious_fibres,
                        fix_wavelengths=fix_wavelengths,
                        sol=sol,
                        do_extinction=do_extinction,
                        scale_sky_1D=scale_sky_1D,
                        correct_negative_sky=correct_negative_sky,
                        sky_method=sky_method,
                        n_sky=n_sky,
                        sky_fibres=sky_fibres,
                        sky_spectrum=sky_list[6],
                        sky_rss=sky_rss_list[6],
                        brightest_line_wavelength=brightest_line_wavelength,
                        cut_sky=cut_sky,
                        fmin=fmin,
                        fmax=fmax,
                        individual_sky_substraction=individual_sky_substraction,
                        telluric_correction=telluric_correction_list[6],
                        id_el=id_el,
                        high_fibres=high_fibres,
                        brightest_line=brightest_line,
                        cut=cut,
                        broad=broad,
                        plot_id_el=plot_id_el,
                        id_list=id_list,
                        clean_sky_residuals=clean_sky_residuals,
                        dclip=dclip,
                        extra_w=extra_w,
                        step_csr=step_csr,
                        valid_wave_min=valid_wave_min,
                        valid_wave_max=valid_wave_max,
                        warnings=warnings,
                        verbose=verbose,
                        plot=plot,
                        norm=norm,
                        fig_size=fig_size,
                    )

                if len(rss_list) > 7:
                    self.rss8 = KOALA_RSS(
                        rss_list[7],
                        rss_clean=rss_clean,
                        save_rss_to_fits_file=save_rss_to_fits_file_list[7],
                        apply_throughput=apply_throughput,
                        skyflat=skyflat_list[7],
                        plot_skyflat=plot_skyflat,
                        correct_ccd_defects=correct_ccd_defects,
                        correct_high_cosmics=correct_high_cosmics,
                        clip_high=clip_high,
                        step_ccd=step_ccd,
                        remove_5578=remove_5578,
                        plot_suspicious_fibres=plot_suspicious_fibres,
                        fix_wavelengths=fix_wavelengths,
                        sol=sol,
                        do_extinction=do_extinction,
                        scale_sky_1D=scale_sky_1D,
                        correct_negative_sky=correct_negative_sky,
                        sky_method=sky_method,
                        n_sky=n_sky,
                        sky_fibres=sky_fibres,
                        sky_spectrum=sky_list[7],
                        sky_rss=sky_rss_list[7],
                        brightest_line_wavelength=brightest_line_wavelength,
                        cut_sky=cut_sky,
                        fmin=fmin,
                        fmax=fmax,
                        individual_sky_substraction=individual_sky_substraction,
                        telluric_correction=telluric_correction_list[7],
                        id_el=id_el,
                        high_fibres=high_fibres,
                        brightest_line=brightest_line,
                        cut=cut,
                        broad=broad,
                        plot_id_el=plot_id_el,
                        id_list=id_list,
                        clean_sky_residuals=clean_sky_residuals,
                        dclip=dclip,
                        extra_w=extra_w,
                        step_csr=step_csr,
                        valid_wave_min=valid_wave_min,
                        valid_wave_max=valid_wave_max,
                        warnings=warnings,
                        verbose=verbose,
                        plot=plot,
                        norm=norm,
                        fig_size=fig_size,
                    )

                if len(rss_list) > 8:
                    self.rss9 = KOALA_RSS(
                        rss_list[8],
                        rss_clean=rss_clean,
                        save_rss_to_fits_file=save_rss_to_fits_file_list[8],
                        apply_throughput=apply_throughput,
                        skyflat=skyflat_list[8],
                        plot_skyflat=plot_skyflat,
                        correct_ccd_defects=correct_ccd_defects,
                        correct_high_cosmics=correct_high_cosmics,
                        clip_high=clip_high,
                        step_ccd=step_ccd,
                        remove_5578=remove_5578,
                        plot_suspicious_fibres=plot_suspicious_fibres,
                        fix_wavelengths=fix_wavelengths,
                        sol=sol,
                        do_extinction=do_extinction,
                        scale_sky_1D=scale_sky_1D,
                        correct_negative_sky=correct_negative_sky,
                        sky_method=sky_method,
                        n_sky=n_sky,
                        sky_fibres=sky_fibres,
                        sky_spectrum=sky_list[8],
                        sky_rss=sky_rss_list[8],
                        brightest_line_wavelength=brightest_line_wavelength,
                        cut_sky=cut_sky,
                        fmin=fmin,
                        fmax=fmax,
                        individual_sky_substraction=individual_sky_substraction,
                        telluric_correction=telluric_correction_list[8],
                        id_el=id_el,
                        high_fibres=high_fibres,
                        brightest_line=brightest_line,
                        cut=cut,
                        broad=broad,
                        plot_id_el=plot_id_el,
                        id_list=id_list,
                        clean_sky_residuals=clean_sky_residuals,
                        dclip=dclip,
                        extra_w=extra_w,
                        step_csr=step_csr,
                        valid_wave_min=valid_wave_min,
                        valid_wave_max=valid_wave_max,
                        warnings=warnings,
                        verbose=verbose,
                        plot=plot,
                        norm=norm,
                        fig_size=fig_size,
                    )

                if len(rss_list) > 9:
                    self.rss10 = KOALA_RSS(
                        rss_list[9],
                        rss_clean=rss_clean,
                        save_rss_to_fits_file=save_rss_to_fits_file_list[9],
                        apply_throughput=apply_throughput,
                        skyflat=skyflat_list[9],
                        plot_skyflat=plot_skyflat,
                        correct_ccd_defects=correct_ccd_defects,
                        correct_high_cosmics=correct_high_cosmics,
                        clip_high=clip_high,
                        step_ccd=step_ccd,
                        remove_5578=remove_5578,
                        plot_suspicious_fibres=plot_suspicious_fibres,
                        fix_wavelengths=fix_wavelengths,
                        sol=sol,
                        do_extinction=do_extinction,
                        scale_sky_1D=scale_sky_1D,
                        correct_negative_sky=correct_negative_sky,
                        sky_method=sky_method,
                        n_sky=n_sky,
                        sky_fibres=sky_fibres,
                        sky_spectrum=sky_list[9],
                        sky_rss=sky_rss_list[9],
                        brightest_line_wavelength=brightest_line_wavelength,
                        cut_sky=cut_sky,
                        fmin=fmin,
                        fmax=fmax,
                        individual_sky_substraction=individual_sky_substraction,
                        telluric_correction=telluric_correction_list[9],
                        id_el=id_el,
                        high_fibres=high_fibres,
                        brightest_line=brightest_line,
                        cut=cut,
                        broad=broad,
                        plot_id_el=plot_id_el,
                        id_list=id_list,
                        clean_sky_residuals=clean_sky_residuals,
                        dclip=dclip,
                        extra_w=extra_w,
                        step_csr=step_csr,
                        valid_wave_min=valid_wave_min,
                        valid_wave_max=valid_wave_max,
                        warnings=warnings,
                        verbose=verbose,
                        plot=plot,
                        norm=norm,
                        fig_size=fig_size,
                    )

            if (
                np.nanmedian(flux_calibration) == 0
                and np.nanmedian(flux_calibration_list[0]) == 0
            ):
                print "\n  3. Cubing without considering any flux calibration ..."
                fcal = False
            else:
                print "\n  3. Cubing applying flux calibration provided ..."
                fcal = True
                if np.nanmedian(flux_calibration) != 0:
                    for i in range(n_files):
                        flux_calibration_list[i] = flux_calibration
                    print "     Using same flux calibration for all object files"
                else:
                    print "     List of flux calibrations provided !"

            if offsets[0] != 1000:
                print "\n  Offsets values for alignment have been given, skipping cubing no-aligned rss..."
                do_cubing = False

            if do_cubing:
                self.cube1 = Interpolated_cube(
                    self.rss1,
                    pixel_size_arcsec,
                    kernel_size_arcsec,
                    plot=plot,
                    flux_calibration=flux_calibration_list[0],
                    warnings=warnings,
                )
                if len(rss_list) > 1:
                    self.cube2 = Interpolated_cube(
                        self.rss2,
                        pixel_size_arcsec,
                        kernel_size_arcsec,
                        plot=plot,
                        flux_calibration=flux_calibration_list[1],
                        warnings=warnings,
                    )
                if len(rss_list) > 2:
                    self.cube3 = Interpolated_cube(
                        self.rss3,
                        pixel_size_arcsec,
                        kernel_size_arcsec,
                        plot=plot,
                        flux_calibration=flux_calibration_list[2],
                        warnings=warnings,
                    )
                if len(rss_list) > 3:
                    self.cube4 = Interpolated_cube(
                        self.rss4,
                        pixel_size_arcsec,
                        kernel_size_arcsec,
                        plot=plot,
                        flux_calibration=flux_calibration_list[3],
                        warnings=warnings,
                    )
                if len(rss_list) > 4:
                    self.cube5 = Interpolated_cube(
                        self.rss5,
                        pixel_size_arcsec,
                        kernel_size_arcsec,
                        plot=plot,
                        flux_calibration=flux_calibration_list[4],
                        warnings=warnings,
                    )
                if len(rss_list) > 5:
                    self.cube6 = Interpolated_cube(
                        self.rss6,
                        pixel_size_arcsec,
                        kernel_size_arcsec,
                        plot=plot,
                        flux_calibration=flux_calibration_list[5],
                        warnings=warnings,
                    )
                if len(rss_list) > 6:
                    self.cube7 = Interpolated_cube(
                        self.rss7,
                        pixel_size_arcsec,
                        kernel_size_arcsec,
                        plot=plot,
                        flux_calibration=flux_calibration_list[6],
                        warnings=warnings,
                    )
                if len(rss_list) > 7:
                    self.cube8 = Interpolated_cube(
                        self.rss8,
                        pixel_size_arcsec,
                        kernel_size_arcsec,
                        plot=plot,
                        flux_calibration=flux_calibration_list[7],
                        warnings=warnings,
                    )
                if len(rss_list) > 8:
                    self.cube9 = Interpolated_cube(
                        self.rss9,
                        pixel_size_arcsec,
                        kernel_size_arcsec,
                        plot=plot,
                        flux_calibration=flux_calibration_list[8],
                        warnings=warnings,
                    )
                if len(rss_list) > 9:
                    self.cube10 = Interpolated_cube(
                        self.rss10,
                        pixel_size_arcsec,
                        kernel_size_arcsec,
                        plot=plot,
                        flux_calibration=flux_calibration_list[9],
                        warnings=warnings,
                    )

            if do_alignment:
                if offsets[0] == 1000:
                    print "\n  4. Aligning individual cubes ..."
                else:
                    print "\n  4. Checking given offsets data and perform cubing ..."

                rss_list_to_align = [self.rss1, self.rss2]
                if len(rss_list) > 2:
                    rss_list_to_align.append(self.rss3)
                if len(rss_list) > 3:
                    rss_list_to_align.append(self.rss4)
                if len(rss_list) > 4:
                    rss_list_to_align.append(self.rss5)
                if len(rss_list) > 5:
                    rss_list_to_align.append(self.rss6)
                if len(rss_list) > 6:
                    rss_list_to_align.append(self.rss7)
                if len(rss_list) > 7:
                    rss_list_to_align.append(self.rss8)
                if len(rss_list) > 8:
                    rss_list_to_align.append(self.rss9)
                if len(rss_list) > 9:
                    rss_list_to_align.append(self.rss10)

                if offsets[0] != 1000:
                    cube_list = []
                else:
                    cube_list = [self.cube1, self.cube2]
                    if len(rss_list) > 2:
                        cube_list.append(self.cube3)
                    if len(rss_list) > 3:
                        cube_list.append(self.cube4)
                    if len(rss_list) > 4:
                        cube_list.append(self.cube5)
                    if len(rss_list) > 5:
                        cube_list.append(self.cube6)
                    if len(rss_list) > 6:
                        cube_list.append(self.cube7)
                    if len(rss_list) > 7:
                        cube_list.append(self.cube8)
                    if len(rss_list) > 8:
                        cube_list.append(self.cube9)
                    if len(rss_list) > 9:
                        cube_list.append(self.cube10)

                cube_aligned_list = align_n_cubes(
                    rss_list_to_align,
                    cube_list=cube_list,
                    flux_calibration_list=flux_calibration_list,
                    pixel_size_arcsec=pixel_size_arcsec,
                    kernel_size_arcsec=kernel_size_arcsec,
                    plot=plot,
                    offsets=offsets,
                    ADR=ADR,
                    warnings=warnings,
                )
                self.cube1_aligned = cube_aligned_list[0]
                self.cube2_aligned = cube_aligned_list[1]
                if len(rss_list) > 2:
                    self.cube3_aligned = cube_aligned_list[2]
                if len(rss_list) > 3:
                    self.cube4_aligned = cube_aligned_list[3]
                if len(rss_list) > 4:
                    self.cube5_aligned = cube_aligned_list[4]
                if len(rss_list) > 5:
                    self.cube6_aligned = cube_aligned_list[5]
                if len(rss_list) > 6:
                    self.cube7_aligned = cube_aligned_list[6]
                if len(rss_list) > 7:
                    self.cube8_aligned = cube_aligned_list[7]
                if len(rss_list) > 8:
                    self.cube9_aligned = cube_aligned_list[8]
                if len(rss_list) > 9:
                    self.cube10_aligned = cube_aligned_list[9]

            if make_combined_cube:
                print "\n  5. Making combined cube ..."
                print "\n> Checking individual cubes: "
                print "   Cube         RA_centre             DEC_centre         Pix Size     Kernel Size"
                print "    1        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                    self.cube1_aligned.RA_centre_deg,
                    self.cube1_aligned.DEC_centre_deg,
                    self.cube1_aligned.pixel_size_arcsec,
                    self.cube1_aligned.kernel_size_arcsec,
                )
                print "    2        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                    self.cube2_aligned.RA_centre_deg,
                    self.cube2_aligned.DEC_centre_deg,
                    self.cube2_aligned.pixel_size_arcsec,
                    self.cube2_aligned.kernel_size_arcsec,
                )
                if len(rss_list) > 2:
                    print "    3        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                        self.cube3_aligned.RA_centre_deg,
                        self.cube3_aligned.DEC_centre_deg,
                        self.cube3_aligned.pixel_size_arcsec,
                        self.cube3_aligned.kernel_size_arcsec,
                    )
                if len(rss_list) > 3:
                    print "    4        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                        self.cube4_aligned.RA_centre_deg,
                        self.cube4_aligned.DEC_centre_deg,
                        self.cube4_aligned.pixel_size_arcsec,
                        self.cube4_aligned.kernel_size_arcsec,
                    )
                if len(rss_list) > 4:
                    print "    5        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                        self.cube5_aligned.RA_centre_deg,
                        self.cube5_aligned.DEC_centre_deg,
                        self.cube5_aligned.pixel_size_arcsec,
                        self.cube5_aligned.kernel_size_arcsec,
                    )
                if len(rss_list) > 5:
                    print "    6        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                        self.cube6_aligned.RA_centre_deg,
                        self.cube6_aligned.DEC_centre_deg,
                        self.cube6_aligned.pixel_size_arcsec,
                        self.cube6_aligned.kernel_size_arcsec,
                    )
                if len(rss_list) > 6:
                    print "    7        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                        self.cube7_aligned.RA_centre_deg,
                        self.cube7_aligned.DEC_centre_deg,
                        self.cube7_aligned.pixel_size_arcsec,
                        self.cube7_aligned.kernel_size_arcsec,
                    )
                if len(rss_list) > 7:
                    print "    8        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                        self.cube8_aligned.RA_centre_deg,
                        self.cube8_aligned.DEC_centre_deg,
                        self.cube8_aligned.pixel_size_arcsec,
                        self.cube8_aligned.kernel_size_arcsec,
                    )
                if len(rss_list) > 8:
                    print "    9        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                        self.cube9_aligned.RA_centre_deg,
                        self.cube9_aligned.DEC_centre_deg,
                        self.cube9_aligned.pixel_size_arcsec,
                        self.cube9_aligned.kernel_size_arcsec,
                    )
                if len(rss_list) > 9:
                    print "   10        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                        self.cube10_aligned.RA_centre_deg,
                        self.cube10_aligned.DEC_centre_deg,
                        self.cube10_aligned.pixel_size_arcsec,
                        self.cube10_aligned.kernel_size_arcsec,
                    )

                ####   THIS SHOULD BE A DEF within Interpolated_cube...
                # Create a cube with zero
                shape = [
                    self.cube1_aligned.data.shape[1],
                    self.cube1_aligned.data.shape[2],
                ]
                self.combined_cube = Interpolated_cube(
                    self.rss1,
                    self.cube1_aligned.pixel_size_arcsec,
                    self.cube1_aligned.kernel_size_arcsec,
                    zeros=True,
                    shape=shape,
                    offsets_files=self.cube1_aligned.offsets_files,
                )

                if obj_name != "":
                    self.combined_cube.object = obj_name
                if description == "":
                    self.combined_cube.description = (
                        self.combined_cube.object + " - COMBINED CUBE"
                    )
                else:
                    self.combined_cube.description = description

                print "\n> Combining cubes..."
                if len(rss_list) == 2:
                    if ADR:
                        print "  Using data corrected for ADR to get combined cube..."
                        self.combined_cube.data = np.nanmedian(
                            [self.cube1_aligned.data_ADR, self.cube2_aligned.data_ADR],
                            axis=0,
                        )
                    else:
                        print "  No ADR correction considered..."
                        self.combined_cube.data = np.nanmedian(
                            [self.cube1_aligned.data, self.cube2_aligned.data], axis=0
                        )
                    self.combined_cube.PA = np.mean(
                        [self.cube1_aligned.PA, self.cube2_aligned.PA]
                    )

                if len(rss_list) == 3:
                    if ADR:
                        print "  Using data corrected for ADR to get combined cube..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data_ADR,
                                self.cube2_aligned.data_ADR,
                                self.cube3_aligned.data_ADR,
                            ],
                            axis=0,
                        )
                    else:
                        print "  No ADR correction considered..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data,
                                self.cube2_aligned.data,
                                self.cube3_aligned.data,
                            ],
                            axis=0,
                        )
                    self.combined_cube.PA = np.mean(
                        [
                            self.cube1_aligned.PA,
                            self.cube2_aligned.PA,
                            self.cube3_aligned.PA,
                        ]
                    )
                if len(rss_list) == 4:
                    if ADR:
                        print "  Using data corrected for ADR to get combined cube..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data_ADR,
                                self.cube2_aligned.data_ADR,
                                self.cube3_aligned.data_ADR,
                                self.cube4_aligned.data_ADR,
                            ],
                            axis=0,
                        )
                    else:
                        print "  No ADR correction considered..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data,
                                self.cube2_aligned.data,
                                self.cube3_aligned.data,
                                self.cube4_aligned.data,
                            ],
                            axis=0,
                        )
                    self.combined_cube.PA = np.mean(
                        [
                            self.cube1_aligned.PA,
                            self.cube2_aligned.PA,
                            self.cube3_aligned.PA,
                            self.cube4_aligned.PA,
                        ]
                    )

                if len(rss_list) == 5:
                    if ADR:
                        print "  Using data corrected for ADR to get combined cube..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data_ADR,
                                self.cube2_aligned.data_ADR,
                                self.cube3_aligned.data_ADR,
                                self.cube4_aligned.data_ADR,
                                self.cube5_aligned.data_ADR,
                            ],
                            axis=0,
                        )
                    else:
                        print "  No ADR correction considered..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data,
                                self.cube2_aligned.data,
                                self.cube3_aligned.data,
                                self.cube4_aligned.data,
                                self.cube5_aligned.data,
                            ],
                            axis=0,
                        )
                    self.combined_cube.PA = np.mean(
                        [
                            self.cube1_aligned.PA,
                            self.cube2_aligned.PA,
                            self.cube3_aligned.PA,
                            self.cube4_aligned.PA,
                            self.cube5_aligned.PA,
                        ]
                    )

                if len(rss_list) == 6:
                    if ADR:
                        print "  Using data corrected for ADR to get combined cube..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data_ADR,
                                self.cube2_aligned.data_ADR,
                                self.cube3_aligned.data_ADR,
                                self.cube4_aligned.data_ADR,
                                self.cube5_aligned.data_ADR,
                                self.cube6_aligned.data_ADR,
                            ],
                            axis=0,
                        )
                    else:
                        print "  No ADR correction considered..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data,
                                self.cube2_aligned.data,
                                self.cube3_aligned.data,
                                self.cube4_aligned.data,
                                self.cube5_aligned.data,
                                self.cube6_aligned.data,
                            ],
                            axis=0,
                        )
                    self.combined_cube.PA = np.mean(
                        [
                            self.cube1_aligned.PA,
                            self.cube2_aligned.PA,
                            self.cube3_aligned.PA,
                            self.cube4_aligned.PA,
                            self.cube5_aligned.PA,
                            self.cube6_aligned.PA,
                        ]
                    )

                if len(rss_list) == 7:
                    if ADR:
                        print "  Using data corrected for ADR to get combined cube..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data_ADR,
                                self.cube2_aligned.data_ADR,
                                self.cube3_aligned.data_ADR,
                                self.cube4_aligned.data_ADR,
                                self.cube5_aligned.data_ADR,
                                self.cube6_aligned.data_ADR,
                                self.cube7_aligned.data_ADR,
                            ],
                            axis=0,
                        )
                    else:
                        print "  No ADR correction considered..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data,
                                self.cube2_aligned.data,
                                self.cube3_aligned.data,
                                self.cube4_aligned.data,
                                self.cube5_aligned.data,
                                self.cube6_aligned.data,
                                self.cube7_aligned.data,
                            ],
                            axis=0,
                        )
                    self.combined_cube.PA = np.mean(
                        [
                            self.cube1_aligned.PA,
                            self.cube2_aligned.PA,
                            self.cube3_aligned.PA,
                            self.cube4_aligned.PA,
                            self.cube5_aligned.PA,
                            self.cube6_aligned.PA,
                            self.cube7_aligned.PA,
                        ]
                    )

                if len(rss_list) == 8:
                    if ADR:
                        print "  Using data corrected for ADR to get combined cube..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data_ADR,
                                self.cube2_aligned.data_ADR,
                                self.cube3_aligned.data_ADR,
                                self.cube4_aligned.data_ADR,
                                self.cube5_aligned.data_ADR,
                                self.cube6_aligned.data_ADR,
                                self.cube7_aligned.data_ADR,
                                self.cube8_aligned.data_ADR,
                            ],
                            axis=0,
                        )
                    else:
                        print "  No ADR correction considered..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data,
                                self.cube2_aligned.data,
                                self.cube3_aligned.data,
                                self.cube4_aligned.data,
                                self.cube5_aligned.data,
                                self.cube6_aligned.data,
                                self.cube7_aligned.data,
                                self.cube8_aligned.data,
                            ],
                            axis=0,
                        )
                    self.combined_cube.PA = np.mean(
                        [
                            self.cube1_aligned.PA,
                            self.cube2_aligned.PA,
                            self.cube3_aligned.PA,
                            self.cube4_aligned.PA,
                            self.cube5_aligned.PA,
                            self.cube6_aligned.PA,
                            self.cube7_aligned.PA,
                            self.cube8_aligned.PA,
                        ]
                    )

                if len(rss_list) == 9:
                    if ADR:
                        print "  Using data corrected for ADR to get combined cube..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data_ADR,
                                self.cube2_aligned.data_ADR,
                                self.cube3_aligned.data_ADR,
                                self.cube4_aligned.data_ADR,
                                self.cube5_aligned.data_ADR,
                                self.cube6_aligned.data_ADR,
                                self.cube7_aligned.data_ADR,
                                self.cube8_aligned.data_ADR,
                                self.cube9_aligned.data_ADR,
                            ],
                            axis=0,
                        )
                    else:
                        print "  No ADR correction considered..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data,
                                self.cube2_aligned.data,
                                self.cube3_aligned.data,
                                self.cube4_aligned.data,
                                self.cube5_aligned.data,
                                self.cube6_aligned.data,
                                self.cube7_aligned.data,
                                self.cube8_aligned.data,
                                self.cube9_aligned.data,
                            ],
                            axis=0,
                        )
                    self.combined_cube.PA = np.mean(
                        [
                            self.cube1_aligned.PA,
                            self.cube2_aligned.PA,
                            self.cube3_aligned.PA,
                            self.cube4_aligned.PA,
                            self.cube5_aligned.PA,
                            self.cube6_aligned.PA,
                            self.cube7_aligned.PA,
                            self.cube8_aligned.PA,
                            self.cube9_aligned.PA,
                        ]
                    )

                if len(rss_list) == 10:
                    print (ADR)
                    if ADR:
                        print "  Using data corrected for ADR to get combined cube..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data_ADR,
                                self.cube2_aligned.data_ADR,
                                self.cube3_aligned.data_ADR,
                                self.cube4_aligned.data_ADR,
                                self.cube5_aligned.data_ADR,
                                self.cube6_aligned.data_ADR,
                                self.cube7_aligned.data_ADR,
                                self.cube8_aligned.data_ADR,
                                self.cube9_aligned.data_ADR,
                                self.cube10_aligned.data_ADR,
                            ],
                            axis=0,
                        )
                    else:
                        print "  No ADR correction considered..."
                        self.combined_cube.data = np.nanmedian(
                            [
                                self.cube1_aligned.data,
                                self.cube2_aligned.data,
                                self.cube3_aligned.data,
                                self.cube4_aligned.data,
                                self.cube5_aligned.data,
                                self.cube6_aligned.data,
                                self.cube7_aligned.data,
                                self.cube8_aligned.data,
                                self.cube9_aligned.data,
                                self.cube10_aligned.data,
                            ],
                            axis=0,
                        )
                    self.combined_cube.PA = np.mean(
                        [
                            self.cube1_aligned.PA,
                            self.cube2_aligned.PA,
                            self.cube3_aligned.PA,
                            self.cube4_aligned.PA,
                            self.cube5_aligned.PA,
                            self.cube6_aligned.PA,
                            self.cube7_aligned.PA,
                            self.cube8_aligned.PA,
                            self.cube9_aligned.PA,
                            self.cube10_aligned.PA,
                        ]
                    )

                # Include flux calibration, assuming it is the same to all cubes (need to be updated to combine data taken in different nights)
                #            if fcal:
                if np.nanmedian(self.cube1_aligned.flux_calibration) == 0:
                    print "  Flux calibration not considered"
                    fcal = False
                else:
                    self.combined_cube.flux_calibration = flux_calibration
                    print "  Flux calibration included!"
                    fcal = True

                #            # Check this when using files taken on different nights  --> Data in self.combined_cube
                #            self.wavelength=self.rss1.wavelength
                #            self.valid_wave_min=self.rss1.valid_wave_min
                #            self.valid_wave_max=self.rss1.valid_wave_max

                self.combined_cube.trace_peak(plot=plot)
                self.combined_cube.get_integrated_map_and_plot(fcal=fcal, plot=plot)

                self.combined_cube.total_exptime = self.rss1.exptime + self.rss2.exptime
                if len(rss_list) > 2:
                    self.combined_cube.total_exptime = (
                        self.combined_cube.total_exptime + self.rss3.exptime
                    )
                if len(rss_list) > 3:
                    self.combined_cube.total_exptime = (
                        self.combined_cube.total_exptime + self.rss4.exptime
                    )
                if len(rss_list) > 4:
                    self.combined_cube.total_exptime = (
                        self.combined_cube.total_exptime + self.rss5.exptime
                    )
                if len(rss_list) > 5:
                    self.combined_cube.total_exptime = (
                        self.combined_cube.total_exptime + self.rss6.exptime
                    )
                if len(rss_list) > 6:
                    self.combined_cube.total_exptime = (
                        self.combined_cube.total_exptime + self.rss7.exptime
                    )
                if len(rss_list) > 7:
                    self.combined_cube.total_exptime = (
                        self.combined_cube.total_exptime + self.rss8.exptime
                    )
                if len(rss_list) > 8:
                    self.combined_cube.total_exptime = (
                        self.combined_cube.total_exptime + self.rss9.exptime
                    )
                if len(rss_list) > 9:
                    self.combined_cube.total_exptime = (
                        self.combined_cube.total_exptime + self.rss10.exptime
                    )

                print "\n  Total exposition time = ", self.combined_cube.total_exptime, "seconds adding the ", len(
                    rss_list
                ), " files"

            # Save it to a fits file

            if save_aligned_cubes:
                print "\n  Saving aligned cubes to fits files ..."
                for i in range(n_files):
                    if i < 9:
                        replace_text = (
                            "_"
                            + obj_name
                            + "_aligned_cube_0"
                            + np.str(i + 1)
                            + pk
                            + ".fits"
                        )
                    else:
                        replace_text = "_aligned_cube_" + np.str(i + 1) + pk + ".fits"

                    aligned_cube_name = rss_list[i].replace(".fits", replace_text)

                    if i == 0:
                        save_fits_file(self.cube1_aligned, aligned_cube_name, ADR=ADR)
                    if i == 1:
                        save_fits_file(self.cube2_aligned, aligned_cube_name, ADR=ADR)
                    if i == 2:
                        save_fits_file(self.cube3_aligned, aligned_cube_name, ADR=ADR)
                    if i == 3:
                        save_fits_file(self.cube4_aligned, aligned_cube_name, ADR=ADR)
                    if i == 4:
                        save_fits_file(self.cube5_aligned, aligned_cube_name, ADR=ADR)
                    if i == 5:
                        save_fits_file(self.cube6_aligned, aligned_cube_name, ADR=ADR)
                    if i == 6:
                        save_fits_file(self.cube7_aligned, aligned_cube_name, ADR=ADR)
                    if i == 7:
                        save_fits_file(self.cube8_aligned, aligned_cube_name, ADR=ADR)
                    if i == 8:
                        save_fits_file(self.cube9_aligned, aligned_cube_name, ADR=ADR)
                    if i == 9:
                        save_fits_file(self.cube10_aligned, aligned_cube_name, ADR=ADR)

            if fits_file == "":
                print "\n  As requested, the combined cube will not be saved to a fits file"
            else:
                print "\n  6. Saving combined cube to a fits file ..."

                check_if_path = fits_file.replace("path:", "")

                if len(fits_file) != len(check_if_path):
                    fits_file = (
                        check_if_path
                        + obj_name
                        + "_"
                        + self.combined_cube.grating
                        + pk
                        + "_combining_"
                        + np.str(n_files)
                        + "_cubes.fits"
                    )

                save_fits_file(self.combined_cube, fits_file, ADR=ADR)

            print "\n================== REDUCING KOALA DATA COMPLETED ====================\n\n"

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

    path_main = "/Users/alopez/Documents/DATA/GAUSS/2018_03_Run_04_GAUSS/"
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
    file1r = path_main + date + "/" + grating + "/10mar20097red"
    file2r = path_main + date + "/" + grating + "/10mar20098red"
    file3r = path_main + date + "/" + grating + "/10mar20099red"

    #    OFFSETS = [-1.1559397974054588, -0.03824158143106171, -0.13488941515513764, 1.8904461213338577]

    #    OBJECT      = "Tol30B"
    #    DESCRIPTION = "Tol30B - 3 cubes"
    file4r = path_main + date + "/" + grating + "/10mar20101red"
    file5r = path_main + date + "/" + grating + "/10mar20102red"
    file6r = path_main + date + "/" + grating + "/10mar20103red"

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



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#  Taylah again, 29 MAY 2019  RED
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#    path_main ="/DATA/KOALA/Taylah/Taylah_20161024_data/ccd_2/"
#    date="20161024"
#    grating="1000R"


# # ---------------------------------------------------------------------------
# # THROUGHPUT CORRECTION USING SKYFLAT
# # ---------------------------------------------------------------------------
# #
# # The very first thing that we need is to get the throughput correction.
# # We use a skyflat that has not been divided by a flatfield
# #
# # We are also using a normalized flatfield:
#    path_flat = path_main
#    file_flatr=path_flat+"FLAT/24oct20067red.fits"
#    flat_red = KOALA_RSS(file_flatr, sky_method="none", do_extinction=False, apply_throughput=False, plot=True)
#                        # correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 100, step_ccd = 50)
# #
# # Provide path, skyflat file and name of the file that will keep the throughput correction
#    path_skyflat = path_main
#    file_skyflatr=path_skyflat+"SKYFLAT/24oct20001red.fits"
#    throughput_file_red=path_main+"SKYFLAT/20161024_1000R_throughput_correction.dat"
# #
# # If this has been done before, we can read the file containing the throughput correction
#    throughput_blue = read_table(throughput_file_blue, ["f"] )
# #
# # Now we read the RSS file, we ONLY correct for ccd defects and high cosmics
#    skyflat_red = KOALA_RSS(file_skyflatr, flat=flat_red, apply_throughput=False, sky_method="none",
#                             do_extinction=False, correct_ccd_defects = False,
#                             correct_high_cosmics = False, clip_high = 100, step_ccd = 50, plot=True)
# #
# # Next we find the relative throughput.
# # If the data have been normalized by the FLATFIELD, we only need a SCALE between fibres,
# # We consider the median value in the range [wave_min_scale, wave_max_scale] for all fibres and scale
# #
#    skyflat_red.find_relative_throughput(ymin=0, ymax=6000) #,  wave_min_scale=6630, wave_max_scale=6820)
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
#    telluric_correction_file=path_main+"telluric_correction_20161024_1000R_0p6_1k25.dat"
#    w_star,telluric_correction_20161024_1000R_0p6_1k25 = read_table(telluric_correction_file, ["f", "f"] )
#    print telluric_correction_20161024_1000R_0p6_1k25


# # READ STAR 1
# # First we provide names, paths, files...
#    star1="EG21"
#    path_star1 = path_main+"EG21/"
#    starpos1r = path_star1+"24oct20060red.fits"
#    fits_file_red = path_star1+"/"+star1+".fits"
#    text_file_red = path_star1+"/"+star1+"_response.dat"

# # Now we read RSS file
# # We apply throughput correction (using nskyflat=False), substract sky using n_sky=600 lowest intensity fibres,
# # correct for CCD defects and high cosmics

#    star1r = KOALA_RSS(starpos1r,
#                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
#                       sky_method="self", n_sky=600,
#                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
#                       valid_wave_min = 6240, valid_wave_max = 7355,
#                       plot=True, warnings=True)

# # PAY ATTENTION to plots. In this case part of the star is in a DEAD FIBRE !!!
# # But we proceed anyway

# # Now we search for the telluric correction
# # For this we stack n_fibres=15 highest intensity fibres, derive a continuum in steps of step=15 A
# # excluding problematic wavelenght ranges (exclude_wlm), normalize flux by continuum & derive flux/flux_normalized
# # This is done in the range [wave_min, wave_max], but only correct telluric in [correct_from, correct_to]
# # Including apply_tc=True will apply correction to the data (do it when happy with results, as need this for flux calibration)

#    telluric_correction_star1 = star1r.get_telluric_correction(n_fibres=15, correct_from=6850., correct_to=7321,
#                                                               exclude_wlm=[[6240,6360],[6430,6730],[6850,7025], [7150,7321], [7350,7480]],
#                                                               apply_tc=True,
#                                                               step = 15, wave_min=6240, wave_max=7355)

# # Next we CREATE THE CUBE for this star, using THE SAME PARAMETERS we will later using for our objects
# # 0.6 is the pixel size, 1.25 is the kernel size.
#    cubes1r=Interpolated_cube(star1r, .6, 1.25, plot=True)

# # Now we read the absolute flux calibration data of the calibration star and get the response curve
# # (Response curve: correspondence between counts and physical values)
# # Include exp_time of the calibration star, as the results are given per second
# # For this BE CAREFUL WITH ABSORPTIONS (Halpha) and check behaviour in the edges of the CCD
#    cubes1r.do_response_curve('FLUX_CAL/feg21_edited.dat', plot=True, min_wave=6240., max_wave=7355.,
#                              step=20, exp_time=180., fit_degree=3)

# # Now we can save this calibration as a text file
#    spectrum_to_text_file(cubes1r.response_wavelength,cubes1r.response_curve, filename=text_file_red)


# # REPEAT FOR STAR 2

#    star2="H600"
#    path_star2 = path_main+"H600/"
#    starpos2r = path_star2+"24oct20061red.fits"
#    fits_file_red = path_star2+"/"+star2+".fits"
#    text_file_red = path_star2+"/"+star2+"_response.dat"

#    star2r = KOALA_RSS(starpos2r,
#                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
#                       sky_method="self", n_sky=400,
#                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
#                       valid_wave_min = 6240, valid_wave_max = 7355,
#                       plot=True, warnings=True)

#    telluric_correction_star2 = star2r.get_telluric_correction(n_fibres=15, correct_from=6850., correct_to=7320.,
#                                                               exclude_wlm=[[6240,6360],[6480,6690],[6850,7070], [7150,7320],[7370,7480]],
#                                                               apply_tc=True,
#                                                               step = 15, wave_min=6240, wave_max=7355)
#
#    cubes2r=Interpolated_cube(star2r, .6, 1.25, plot=True)
#    cubes2r.do_response_curve('FLUX_CAL/fhilt600_edited.dat', plot=True, min_wave=6240., max_wave=7355., step=40, exp_time=120., fit_degree=3)
#    spectrum_to_text_file(cubes2r.response_wavelength,cubes2r.response_curve, filename=text_file_red)

# # REPEAT FOR STAR 3

#    star3="LTT2415"
#    path_star3 = path_main+"LTT2415/"   #+star1+"/"+date+"/"+grating+"/"
#    starpos3r = path_star3+"24oct20059red.fits"
#    fits_file_red = path_star3+"/"+star3+".fits"#+star1+"_"+grating+"_"+date+".fits"
#    text_file_red = path_star3+"/"+star3+"_response.dat" #+star1+"_"+grating+"_"+date+"_response.dat"
#
#    star3r = KOALA_RSS(starpos3r,
#                       apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
#                       sky_method="self", n_sky=600,
#                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
#                       valid_wave_min = 6240, valid_wave_max = 7355,
#                       plot=True, warnings=True)


#    telluric_correction_star3 = star3r.get_telluric_correction(n_fibres=15, correct_from=6850., correct_to=7320.,
#                                                               exclude_wlm=[[6240,6340],[6500,6650],[6850,7000], [7150,7320],[7370,7480]],
#                                                               apply_tc=True,
#                                                               step = 15, wave_min=6240, wave_max=7355)
#
#    cubes3r=Interpolated_cube(star3r, .6, 1.25, plot=True)
#    cubes3r.do_response_curve('FLUX_CAL/fltt2415_edited.dat', plot=True, min_wave=6240., max_wave=7355., step=40, exp_time=180., fit_degree=3)
#    spectrum_to_text_file(cubes3r.response_wavelength,cubes3r.response_curve, filename=text_file_red)


# #  CHECK AND GET THE FLUX CALIBRATION FOR THE NIGHT  RED

# #  First we take another look to the RSS data ploting the integrated fibre values in a map
#    star1r.RSS_map(star1r.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))  # Dead fibre!!!
#    star2r.RSS_map(star2r.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))
#    star3r.RSS_map(star3r.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))

# # We check again that star1 is on a dead fibre, we don't use this star for absolute flux calibration

# # Define in "stars" the 2 cubes we are using, and plotting their responses to check
#    stars=[cubes2r,cubes3r]
#    plot_response(stars)

# # The shape of the curves are ~OK but they have a variation of ~5% in flux...
# # Probably this would have been corrected obtaining at least TWO exposures per star..
# # Anyway, that is what it is, we obtain the flux calibration applying:
#    flux_calibration_20161024_1000R_0p6_1k25=obtain_flux_calibration(stars)

# # And we save this absolute flux calibration as a text file
#    flux_correction_file = path_main+"flux_calibration_20161024_1000R_0p6_1k25.dat"
#    spectrum_to_text_file(star1r.wavelength,flux_calibration_20161024_1000R_0p6_1k25, filename=flux_correction_file)


# #   CHECK AND GET THE TELLURIC CORRECTION

# # Similarly, provide a list with the telluric corrections and apply:
#    telluric_correction_list=[telluric_correction_star1,telluric_correction_star2,telluric_correction_star3]
#    telluric_correction_20161024red = obtain_telluric_correction(star1r.wavelength, telluric_correction_list)

# # Save this telluric correction to a file
#    telluric_correction_file=path_main+"telluric_correction_20161024_1000R_0p6_1k25.dat"
#    spectrum_to_text_file(star1r.wavelength,telluric_correction_20161024red, filename=telluric_correction_file )


# #
# #
# #
# # ---------------------------------------------------------------------------
# #  OBTAIN SKY SPECTRA
# # ---------------------------------------------------------------------------
# #
# #
# #
# #
#    file_sky_r1 = path_main+"SKY/24oct20045red.fits"
#    file_sky_r2 = path_main+"SKY/24oct20048red.fits"
#    file_sky_r3 = path_main+"SKY/24oct20051red.fits"
#    file_sky_r4 = path_main+"SKY/24oct20054red.fits"
#
#    sky_r1 = KOALA_RSS(file_sky_r1, apply_throughput=True, skyflat = skyflat_red, do_extinction=True,
#                       sky_method="none", is_sky=True, win_sky=151,
#                       correct_ccd_defects = False, correct_high_cosmics = True, clip_high = 100, step_ccd = 50,
#                       plot=True)
#    sky_r2 = KOALA_RSS(file_sky_r2, apply_throughput=True, skyflat = skyflat_red, do_extinction=True,
#                       sky_method="none", is_sky=True, win_sky=151,
#                       correct_ccd_defects = False, correct_high_cosmics = True, clip_high = 100, step_ccd = 50,
#                       plot=True)
#    sky_r3 = KOALA_RSS(file_sky_r3, apply_throughput=True, skyflat = skyflat_red, do_extinction=True,
#                       sky_method="none", is_sky=True, win_sky=151,
#                       correct_ccd_defects = False, correct_high_cosmics = True, clip_high = 100, step_ccd = 50,
#                       plot=True)
#    sky_r4 = KOALA_RSS(file_sky_r4, apply_throughput=True, skyflat = skyflat_red, do_extinction=True,
#                       sky_method="none", is_sky=True, win_sky=151,
#                       correct_ccd_defects = False, correct_high_cosmics = True, clip_high = 100, step_ccd = 50,
#                       plot=True)
#
# #
# #
# #
# # ---------------------------------------------------------------------------
# # TIME FOR THE OBJECT !!
# # ---------------------------------------------------------------------------
# #
# #
# #

# # First we provide all files:

#    file1r=path_main+"OBJECT/24oct20044red.fits"
#    file2r=path_main+"OBJECT/24oct20046red.fits"
#    file3r=path_main+"OBJECT/24oct20047red.fits"
#    file4r=path_main+"OBJECT/24oct20049red.fits"
#    file5r=path_main+"OBJECT/24oct20050red.fits"
#    file6r=path_main+"OBJECT/24oct20052red.fits"
#    file7r=path_main+"OBJECT/24oct20053red.fits"
#
#
#    rss_list=[file1r,file2r,file3r,file4r,file5r,file6r,file7r]
#    sky_list=[sky_r1,sky_r1,sky_r2,sky_r2,sky_r3,sky_r3,sky_r4]
#    fits_file_red=path_main+"OBJECT/taylah_red_combined_cube_7.fits"
#
#    taylah_test = KOALA_reduce(rss_list, fits_file=fits_file_red, obj_name="Taylah",  description = "TAYLAH 7 CUBE",
#                          apply_throughput=True, skyflat = skyflat_red, plot_skyflat=False,
#                          correct_ccd_defects = False, correct_high_cosmics = True, clip_high = 100, step_ccd=50,
#                          do_extinction=True,
#                          sky_method="2D", sky_list=sky_list,
#                          telluric_correction = telluric_correction_20161024red,
#                          id_el=True, high_fibres=10, brightest_line="Ha", cut=1.1, broad=4.0, plot_id_el= False,
#                          id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6716.47, 6730.85],
#                          clean_sky_residuals = True, dclip=2.0, extra_w = 1.3, step_csr = 100, # fibre=427,
#
#                          pixel_size_arcsec=.6, kernel_size_arcsec=1.25,
#                          #offsets=[1000],
#                          ADR=False,
#                          flux_calibration=flux_calibration_20161024_1000R_0p6_1k25,
#
#                          valid_wave_min = 6240, valid_wave_max = 7355,
#                          plot=True, warnings=False)


#        offsets=[2.1661863268521699, -1.0154564185927739,
#                 -1.2906911413228896, -0.45669081905592757,
#                 -0.07410120886509336, 1.7307043173337093,
#                 0.80060388230072721, -0.79741816283239464,
#                 2.4503657941496169, -0.74336415384485555,
#                 -1.277432180113331, -0.38498639829597181]


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


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#  Taylah 02 June 2019  BLUE
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#    path_main ="/DATA/KOALA/Taylah/Taylah_20161024_data/ccd_1/"
#    date="20161024"
#    grating="580V"


# # ---------------------------------------------------------------------------
# # THROUGHPUT CORRECTION USING SKYFLAT
# # ---------------------------------------------------------------------------
# #
# # The very first thing that we need is to get the throughput correction.
# # We use a skyflat that has not been divided by a flatfield
# #
# # We are also using a normalized flatfield:
#    path_flat = path_main
#    file_flatb=path_flat+"FLAT/24oct10067red.fits"
#    flat_blue = KOALA_RSS(file_flatb, sky_method="none", do_extinction=False, apply_throughput=False, plot=True)
#                        # correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 100, step_ccd = 50)
# #
# # Provide path, skyflat file and name of the file that will keep the throughput correction
#    path_skyflat = path_main
#    file_skyflatb=path_skyflat+"SKYFLAT/24oct10001red.fits"
#    throughput_file_blue=path_main+"SKYFLAT/20161024_580V_throughput_correction.dat"
# #
# # If this has been done before, we can read the file containing the throughput correction
#    throughput_blue = read_table(throughput_file_blue, ["f"] )
# #
# # Now we read the RSS file, we ONLY correct for ccd defects and high cosmics
#    skyflat_blue = KOALA_RSS(file_skyflatb, flat=flat_blue, apply_throughput=False, sky_method="none",
#                             do_extinction=False, correct_ccd_defects = False,
#                             correct_high_cosmics = False, clip_high = 100, step_ccd = 50, plot=True)
# #
# # Next we find the relative throughput.
# # If the data have been normalized by the FLATFIELD, we only need a SCALE between fibres,
# # We consider the median value in the range [wave_min_scale, wave_max_scale] for all fibres and scale
# #
#    skyflat_blue.find_relative_throughput(ymin=0, ymax=6000) #,  wave_min_scale=6630, wave_max_scale=6820)
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
#    star1="EG21"
#    path_star1 = path_main+"EG21/"
#    starpos1b = path_star1+"24oct10060red.fits"
#    fits_file_blue = path_star1+"/"+star1+".fits"
#    text_file_blue = path_star1+"/"+star1+"_response.dat"

# # Now we read RSS file
# # We apply throughput correction, substract sky using n_sky=600 lowest intensity fibres,
# # correct for CCD defects and high cosmics

#    star1b = KOALA_RSS(starpos1b,
#                       apply_throughput=True, skyflat = skyflat_blue, plot_skyflat=False,
#                       sky_method="self", n_sky=600,
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
#    cubes1b.do_response_curve('FLUX_CAL/feg21_edited.dat', plot=True, min_wave=3800., max_wave=5750.,
#                              step=10, exp_time=180., fit_degree=3)

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
#    stars=[cubes2b,cubes3b]
#    plot_response(stars)

# # The shape of the curves are OK but they have a variation of ~3% in flux...
# # Probably this would have been corrected obtaining at least TWO exposures per star..
# # Anyway, that is what it is, we obtain the flux calibration applying:
#    flux_calibration_20161024_580V_0p6_1k25=obtain_flux_calibration(stars)

# # And we save this absolute flux calibration as a text file
#    flux_correction_file = path_main+"flux_calibration_20161024_580V_0p6_1k25.dat"
#    spectrum_to_text_file(star1b.wavelength,flux_calibration_20161024_580V_0p6_1k25, filename=flux_correction_file)

# #
# #
# #
# # ---------------------------------------------------------------------------
# #  OBTAIN SKY SPECTRA
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

# # First we provide all files:

#    file1b=path_main+"OBJECT/24oct10044red.fits"
#    file2b=path_main+"OBJECT/24oct10046red.fits"
#    file3b=path_main+"OBJECT/24oct10047red.fits"
#    file4b=path_main+"OBJECT/24oct10049red.fits"
#    file5b=path_main+"OBJECT/24oct10050red.fits"
#    file6b=path_main+"OBJECT/24oct10052red.fits"
#    file7b=path_main+"OBJECT/24oct10053red.fits"
#
#
#    rss_list=[file1b,file2b,file3b,file4b,file5b,file6b,file7b]
#    sky_list=[sky_b1,sky_b1,sky_b2,sky_b2,sky_b3,sky_b3,sky_b4]
#    fits_file_blue=path_main+"OBJECT/taylah_blue_combined_cube_7_ADR.fits"
#
#    taylah_blue = KOALA_reduce(rss_list, fits_file=fits_file_blue, obj_name="Taylah",  description = "TAYLAH 7 CUBE",
#                          apply_throughput=True, skyflat = skyflat_blue, plot_skyflat=False,
#                          correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
#                          do_extinction=True,
#                          sky_method="2D", sky_list=sky_list,
#                          #id_el=True, high_fibres=10, brightest_line="Hb", cut=1.1, broad=3.0, plot_id_el= False,
#                          #id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6716.47, 6730.85],
#                          #clean_sky_residuals = False, dclip=2.0, extra_w = 1.3, step_csr = 100, # fibre=427,
#                          pixel_size_arcsec=.6, kernel_size_arcsec=1.25,
##                          offsets=[2.1661863268521699, -1.0154564185927739,
##                                     -1.2906911413228896, -0.45669081905592757,
##                                     -0.07410120886509336, 1.7307043173337093,
##                                     0.80060388230072721, -0.79741816283239464,
##                                     2.4503657941496169, -0.74336415384485555,
##                                     -1.277432180113331, -0.38498639829597181],
#                          ADR=True,
#                          flux_calibration=flux_calibration_20161024_580V_0p6_1k25,
#
#                          valid_wave_min = 3770, valid_wave_max = 5799,
#                          plot=True, warnings=False)


######## TESTING ALIGNMENT


#    taylah_r3 = KOALA_RSS(file3r, #flat=flat_red,
#                          apply_throughput=True, skyflat = skyflat_red,
#                          correct_ccd_defects = False, correct_high_cosmics = True, clip_high = 100, step_ccd=50,
#                          sky_method="self", n_sky=100,
##                          sky_method="1D", sky_spectrum=sky1D,
##                          sky_method="2D", sky_rss=sky_r1, #scale_sky_rss = 3.2, # 3.35662235438,
#                          id_el=False, high_fibres=10, brightest_line="Ha", cut=1.1, broad=3.5, plot_id_el= False,
#                          id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6716.47, 6730.85],
#                          clean_sky_residuals = False, dclip=2.0, extra_w = 1.3, step_csr = 100, fibre=0,
#                          valid_wave_min = 6240, valid_wave_max = 7355, plot=True, warnings=False)

#    cube1 =Interpolated_cube(taylah_r1, 0.6, 1.25, flux_calibration=flux_calibration_20161024_1000R_0p6_1k25)


#    cube_list=[cube1,cube2,cube3]
#    rss_list=[taylah_r1,taylah_r2,taylah_r3]
#    cube_align=[cube1_al,cube2_al,cube3_al]

#    cube1_al,cube2_al,cube3_al=align_n_cubes(cube_list, rss_list, pixel_size_arcsec=0.6, kernel_size_arcsec=1.25, offsets=[1000], plot= True, ADR=False, warnings=False)

#    shape = [cube1_al.data.shape[1], cube1_al.data.shape[2]]
#    combined_cube = Interpolated_cube(taylah_r1, 0.6, 1.25, zeros=True, shape=shape, offsets_files =cube1_al.offsets_files)
#    combined_cube.data = np.nanmedian([cube1_al.data, cube2_al.data, cube3_al.data], axis = 0)
#    combined_cube.trace_peak(plot=True)
#    combined_cube.get_integrated_map_and_plot(fcal=True, plot=True)


# # OLD -----------

# # That is for creating a 1D sky for tests
#    sky1D=np.nanmedian(sky_b1.intensity_corrected, axis=0)
#    plt.plot(sky_r1.wavelength,sky_r1.intensity_corrected[200])
#    plt.xlim(6946,6952)
#    plt.axvline(x=6949)


# # Now we read a science frame

#    file1r=path_main+"OBJECT/24oct20044red.fits"


#    taylah_r1 = KOALA_RSS(file1r, #flat=flat_red,
#                          apply_throughput=True, skyflat = skyflat_red,
#                          correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 100, step_ccd=50,
##                          sky_method="self", n_sky=100,
##                          sky_method="1D", sky_spectrum=sky1D,
#                          sky_method="2D", sky_rss=sky_r1, #scale_sky_rss = 3.2, # 3.35662235438,
#                          id_el=True, high_fibres=10, brightest_line="Ha", cut=1.1, broad=3.5, plot_id_el= False,
#                          id_list=[6300.30, 6548.03, 6562.82, 6583.41, 6716.47, 6730.85],
#                          clean_sky_residuals = True, dclip=2.0, extra_w = 1.3, step_csr = 100, fibre=0,
#                          valid_wave_min = 6240, valid_wave_max = 7355, plot=True, warnings=False)
#
#    p05=np.nanpercentile(taylah_b1.intensity_corrected,5)
#    p95=np.nanpercentile(taylah_b1.intensity_corrected,95)
#    print p05,p95
#    max_abs=np.nanmax([np.abs(p05),np.abs(p95)])
#    print max_abs

#    taylah_r1.RSS_image(cmap="fuego", clow=-10, chigh=100)
#    taylah_r1.RSS_image(image=taylah_r1.intensity, cmap="binary_r")
#    taylah_r1.RSS_image()
#    taylah_r1.pl

#    sky_r1.RSS_image(cmap="binary_r", clow=0,chigh=15)

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

# 2018 04 29


#    path_main ="/DATA/KOALA/"
#    date="20180429"
#    grating="385R"

# #
# # THROUGHPUT CORRECTION USING SKYFLAT
# #
# # The very first thing that we need is to get the throughput correction.
# # First, provide path, skyflat file and name of the file that will keep the throughput correction
#    path_flat = path_main+"SKYFLATS/"+date+"/"+grating+"/"
#    file_skyflatr=path_flat+"29apr21012red.fits"
#    throughput_file_red=path_flat+date+"_"+grating+"_throughput_correction.dat"
#
# # If this has been done before, we can read the file containing the throughput correction
#    throughput_red = read_table(throughput_file_red, ["f"] )
#
# # Now we read the RSS file, we ONLY correct for ccd defects and high cosmics
#    skyflat_red = KOALA_RSS(file_skyflatr, do_sky=False, do_extinction=False, apply_throughput=False, plot=True,
#                            correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd = 50)
#
# # Next we find the relative throughput.
# # We consider the median value in the range [wave_min_scale, wave_max_scale] for all fibres and scale
#
#    skyflat_red.find_relative_throughput(ymin=0, ymax=100000)#, wave_min_scale=6200, wave_max_scale=9000)
#                                         #step=100, fit_skyflat_degree = 0, wave_min_flat=6100, wave_max_flat=9250)
#
# # The relative throughput is an array stored in skyflat_red.relative_throughput
# # We save that array in a text file that we can read in the future without the need of repeating this
#
#    array_to_text_file(skyflat_red.relative_throughput, filename= throughput_file_red )
#
#


# #
# # OBTAIN ABSOLUTE FLUX CALIBRATION AND TELLURIC CORRECTION USING CALIBRATION STARS
# #
#
# # If these have been obtained already, we can read files containing arrays with the calibrations
# # Uncomment the next two sections and skip the rest till "OBTAIN SKY SPECTRA"
#
# # READ FLUX CALIBRATION RED
#    flux_cal_file = path_main+"/FLUX_CAL/"+date+"_"+grating+"_0p7_2k1_FLUX_CAL.dat"
#    w_star,flux_calibration_20161024red_0p7_2k1 = read_table(flux_cal_file, ["f", "f"] )
#    print flux_calibration_20161024red_0p7_2k1
#
# # READ TELLURIC CORRECTION FROM FILE
#    telluric_correction_file=path_main+"/FLUX_CAL/"+date+"_"+grating+"_telluric_correction.dat"
#    w_star,telluric_correction_20161024red = read_table(telluric_correction_file, ["f", "f"] )
#    print telluric_correction_20161024red


# # READ STAR 1   # ABRIL
# # First we provide names, paths, files...
#    star1="HD60753"
#    path_star1 = path_main+"STARS/"+star1+"/"+date+"/"+grating+"/"
#    starpos1r = path_star1+"29apr20025red.fits"
#    starpos2r = path_star1+"29apr20026red.fits"
#    starpos3r = path_star1+"29apr20027red.fits"
#    fits_file_red = path_star1+star1+"_"+grating+"_"+date+".fits"
#    text_file_red = path_star1+star1+"_"+grating+"_"+date+"_response.dat"

# # Now we read one of the RSS files
# # We apply throughput correction, substract sky using n_sky=300 lowest intensity fibres,
# # correct for CCD defects and high cosmics


#    star1r = KOALA_RSS(starpos1r,
#                       apply_throughput=True, skyflat = skyflat_red, #nskyflat=False, plot_skyflat=False,
#                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 5, step_ccd=100,
#                       plot_suspicious_fibres=False,
#                       do_sky=True, n_sky=300,
#                       valid_wave_min = 6150, valid_wave_max = 9220,
#                       plot=True, warnings=False)


# #
# # PAY ATTENTION to plots.
#
# # Now we search for the telluric correction
# # For this we stack n_fibres=15 highest intensity fibres, derive a continuum in steps of step=15 A
# # excluding problematic wavelenght ranges (exclude_wlm), normalize flux by continuum & derive flux/flux_normalized
# # This is done in the range [wave_min, wave_max], but only correct telluric in [correct_from, correct_to]
# # Including apply_tc=True will apply correction to the data (do it when happy with results, as need this for flux calibration)
#
#    telluric_correction_star1 = star1r.get_telluric_correction(n_fibres=15, correct_from=6830., correct_to=8380.,
#                                                               exclude_wlm=[[6500,6700],[6830,7150], [7150,7400], [7550,7700],[8080,8380]],
#                                                               apply_tc=False, #combined_cube = False,
#                                                               step = 15, wave_min=6150, wave_max=9300)
#
#
# # Next we CREATE THE CUBE for this star, using THE SAME PARAMETERS we will later using for our objects
# # 0.6 is the pixel size, 1.25 is the kernel size.
#
# # If we only have 1 rss file, assuming we used apply_tc=True before for applying telluric correction
#
#    cubes1r=Interpolated_cube(star1r, .6, 1.25, plot=True)
#
# # If we have several rss files, we use KOALA_reduce and include the telluric correction:

#    HD60753red=KOALA_reduce([starpos1r,starpos2r,starpos3r], fits_file=fits_file_red,
#                            apply_throughput=True, skyflat = skyflat_red,
#                            correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 5, step_ccd=100,
#                            do_sky=True, n_sky=300,
#                            telluric_correction = telluric_correction_star1,
#                            pixel_size_arcsec=.6, kernel_size_arcsec=1.25,
#                            valid_wave_min = 6150, valid_wave_max = 9220)

# # Now we read the absolute flux calibration data of the calibration star and get the response curve
# # (Response curve: correspondence between counts and physical values)
# # Include exp_time of the calibration star, as the results are given per second
# # For this BE CAREFUL WITH ABSORPTIONS (Halpha) and check behaviour in the edges of the CCD
#
#    HD60753red.combined_cube.do_response_curve('FLUX_CAL/fhd60753.dat', plot=True, min_wave=6150., max_wave=9220.,
# 			       step=10, exp_time=15., fit_degree=5)
#
# # Now we can save this calibration as a text file:
#
#    spectrum_to_text_file(cubes1r.response_wavelength,cubes1r.response_curve, filename=text_file_red)
#
#    spectrum_to_text_file(HD60753red.combined_cube.response_wavelength,HD60753red.combined_cube.response_curve, filename=text_file_red)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ----------- ANGEL POX 4: Night 2016 01 16 -----------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


#    path_main ="/DATA/KOALA/"
#    date="20160116"
#    grating="1000R"
#
#
# #
# # THROUGHPUT CORRECTION USING SKYFLAT
# #
# # The very first thing that we need is to get the throughput correction.
# # First, provide path, skyflat file and name of the file that will keep the throughput correction
#    path_flat = path_main+"SKYFLATS/"+date+"/"+grating+"/"
#    file_skyflatr=path_flat+"16jan2_combined.fits"
#    throughput_file_red=path_flat+date+"_"+grating+"_throughput_correction.dat"
#
# # If this has been done before, we can read the file containing the throughput correction
#    throughput_red = read_table(throughput_file_red, ["f"] )
#
# # Now we read the RSS file, we ONLY correct for ccd defects and high cosmics
#    skyflat_red = KOALA_RSS(file_skyflatr, do_sky=False, do_extinction=False, apply_throughput=False, plot=True,
#                            correct_ccd_defects = False, correct_high_cosmics = True, clip_high = 100, step_ccd = 50)
#
# # Next we find the relative throughput.
# # If the data have been normalized by the FLATFIELD, we only need a SCALE between fibres,
# # choose nskyflat=False and ignore step and fit_skyflat_degree (this is for creating a normalized flatfield here)
# # We consider the median value in the range [wave_min_scale, wave_max_scale] for all fibres and scale
#
#    skyflat_red.find_relative_throughput(ymin=5000, ymax=300000, step=100, fit_skyflat_degree = 3,   # Try fit_skyflat_degree = 0, wave_min_scale=6630, wave_max_scale=6820,
#                                         wave_max_flat=7420, nskyflat=True)
#
# # The relative throughput is an array stored in skyflat_red.relative_throughput
# # We save that array in a text file that we can read in the future without the need of repeating this
#
#    array_to_text_file(skyflat_red.relative_throughput, filename= throughput_file_red )
#
#

# #
# # OBTAIN ABSOLUTE FLUX CALIBRATION AND TELLURIC CORRECTION USING CALIBRATION STARS
# #
#
# # If these have been obtained already, we can read files containing arrays with the calibrations
# # Uncomment the next two sections and skip the rest till "OBTAIN SKY SPECTRA"
#
# # READ FLUX CALIBRATION RED
#    flux_cal_file = path_main+"/FLUX_CAL/"+date+"_"+grating+"_0p7_2k1_FLUX_CAL.dat"
#    w_star,flux_calibration_20161024red_0p7_2k1 = read_table(flux_cal_file, ["f", "f"] )
#    print flux_calibration_20161024red_0p7_2k1
#
# # READ TELLURIC CORRECTION FROM FILE
#    telluric_correction_file=path_main+"/FLUX_CAL/"+date+"_"+grating+"_telluric_correction.dat"
#    w_star,telluric_correction_20161024red = read_table(telluric_correction_file, ["f", "f"] )
#    print telluric_correction_20161024red


# # READ STAR 1
# # First we provide names, paths, files...
#    star1="EG21"
#    path_star1 = path_main+"STARS/"+star1+"/"+date+"/"+grating+"/"
#    starpos1r = path_star1+"16jan20046red.fits"
#    fits_file_red = path_star1+star1+"_"+grating+"_"+date+".fits"
#    text_file_red = path_star1+star1+"_"+grating+"_"+date+"_response.dat"
#
# # Now we read RSS file
# # We apply throughput correction (using nskyflat=True), substract sky using n_sky=600 lowest intensity fibres,
# # correct for CCD defects and high cosmics
#
#    star1r = KOALA_RSS(starpos1r,
#                       apply_throughput=True, skyflat = skyflat_red, nskyflat=False, plot_skyflat=False,
#                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 100, step_ccd=50,
#                       do_sky=True, n_sky=600,
#                       valid_wave_min = 6271, valid_wave_max = 7400,
#                       plot=True, warnings=True)
#
# # PAY ATTENTION to plots.
#
# # Now we search for the telluric correction
# # For this we stacj n_fibres=15 highest intensity fibres, derive a continuum in steps of step=15 A
# # excluding problematic wavelenght ranges (exclude_wlm), normalize flux by continuum & derive flux/flux_normalized
# # This is done in the range [wave_min, wave_max], but only correct telluric in [correct_from, correct_to]
# # Including apply_tc=True will apply correction to the data (do it when happy with results, as need this for flux calibration)


# # CHECK exclude_wlm=[[6240,6360],[6430,6730],[6850,7020], [7140,7320], [7360,7480]],    apply_tc=True

#    telluric_correction_star1 = star1r.get_telluric_correction(n_fibres=15, correct_from=6840., correct_to=7320.,
#                                                               exclude_wlm=[[6470,6680],[6840,7050], [7150,7350]],
#                                                               apply_tc=True,
#                                                               step = 15, wave_min=6271, wave_max=7400)
#
#
# # Next we CREATE THE CUBE for this star, using THE SAME PARAMETERS we will later using for our objects
# # 0.6 is the pixel size, 1.25 is the kernel size.
#
#    cubes1r=Interpolated_cube(star1r, .6, 1.25, plot=True)
#
# # Now we read the absolute flux calibration data of the calibration star and get the response curve
# # (Response curve: correspondence between counts and physical values)
# # Include exp_time of the calibration star, as the results are given per second
# # For this BE CAREFUL WITH ABSORPTIONS (Halpha) and check behaviour in the edges of the CCD
#
#    cubes1r.do_response_curve('FLUX_CAL/feg21_edited.dat', plot=True, min_wave=6280., max_wave=7400.,
# 			       step=10, exp_time=120., fit_degree=3)
#
# # Now we can save this calibration as a text file
#    spectrum_to_text_file(cubes1r.response_wavelength,cubes1r.response_curve, filename=text_file_red)
#


# # REPEAT FOR STAR 2
#
#
#    star2="H600"
#    path_star2 = path_main+"STARS/"+star2+"/"+date+"/"+grating+"/"
#    starpos2r = path_star2+"16jan20052red.fits"
#    fits_file_red = path_star2+star2+"_"+grating+"_"+date+".fits"
#    text_file_red = path_star2+star2+"_"+grating+"_"+date+"_response.dat"

#    star2r = KOALA_RSS(starpos2r,
#                       apply_throughput=True, skyflat = skyflat_red, nskyflat=False, plot_skyflat=False,
#                       do_sky=True, n_sky=600,
#                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
#                       valid_wave_min = 6271, valid_wave_max = 7400,
#                       plot=True, warnings=True)
#
#
#    telluric_correction_star2 = star2r.get_telluric_correction(n_fibres=15, correct_from=6850., correct_to=7320.,
#                                                               exclude_wlm=[[6400,6700],[6840,7050], [7150,7350]],
#                                                               apply_tc=True,
#                                                               step = 15, wave_min=6271, wave_max=7400)
#
#    cubes2r=Interpolated_cube(star2r, .6, 1.25, plot=True)
#    cubes2r.do_response_curve('FLUX_CAL/fhilt600_edited.dat', plot=True, min_wave=6280., max_wave=7400., step=10, exp_time=120., fit_degree=3)
#    spectrum_to_text_file(cubes2r.response_wavelength,cubes2r.response_curve, filename=text_file_red)

# # REPEAT FOR STAR 3

#    star3="Feige56"
#    path_star3 = path_main+"STARS/"+star3+"/"+date+"/"+grating+"/"
#    starpos3r = path_star3+"16jan20064red.fits"
#    fits_file_red = path_star3+star3+"_"+grating+"_"+date+".fits"
#    text_file_red = path_star3+star3+"_"+grating+"_"+date+"_response.dat"
#
#    star3r = KOALA_RSS(starpos3r,
#                       apply_throughput=True, skyflat = skyflat_red, nskyflat=False, plot_skyflat=False,
#                       do_sky=True, n_sky=600,
#                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
#                       valid_wave_min = 6271, valid_wave_max = 7400,
#                       plot=True, warnings=True)
#
#    telluric_correction_star3 = star3r.get_telluric_correction(n_fibres=15, correct_from=6850., correct_to=7320.,
#                                                               exclude_wlm=[[6450,6650],[6840,7050], [7150,7350]],
#                                                               apply_tc=True,
#                                                               step = 15, wave_min=6271, wave_max=7400)
#
#    cubes3r=Interpolated_cube(star3r, .6, 1.25, plot=True)
#    cubes3r.do_response_curve('FLUX_CAL/ffeige56_edited.dat', plot=True, min_wave=6280., max_wave=7420., step=10, exp_time=120., fit_degree=3)
#    spectrum_to_text_file(cubes3r.response_wavelength,cubes3r.response_curve, filename=text_file_red)


# #  CHECK AND GET THE FLUX CALIBRATION FOR THE NIGHT  RED

# #  First we take another look to the RSS data plotting the integrated fibre values in a map
#    star1r.RSS_map(star1r.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))
#    star2r.RSS_map(star2r.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))
#    star3r.RSS_map(star3r.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))


# # Define in "stars" the 2 cubes we are using, and plotting their responses to check
#    stars=[cubes1r,cubes2r,cubes3r]
#    plot_response(stars)

# # Anyway, that is what it is, we obtain the flux calibration applying:
#    flux_calibration_20160116red_0p7_2k1=obtain_flux_calibration(stars)

# # And we save this absolute flux calibration as a text file
#    flux_cal_file = path_main+"/FLUX_CAL/"+date+"_"+grating+"_0p7_2k1_FLUX_CAL.dat"
#    spectrum_to_text_file(star1r.wavelength,flux_calibration_20160116red_0p7_2k1, filename=flux_cal_file)


# #   CHECK AND GET THE TELLURIC CORRECTION

# # Similarly, provide a list with the telluric corrections and apply:
#    telluric_correction_list=[telluric_correction_star1,telluric_correction_star2,telluric_correction_star3]
#    telluric_correction_file=path_main+"/FLUX_CAL/"+date+"_"+grating+"_telluric_correction.dat"
#    telluric_correction_20160116red = obtain_telluric_correction(star1r.wavelength, telluric_correction_list)
# # Save this telluric correction to a file
#    spectrum_to_text_file(star1r.wavelength,telluric_correction_20160116red, filename=telluric_correction_file )


# #  GALAXY

#    galaxy="POX4"
#    folder="03_POX4"
#    path_galaxy = path_main+"Hi-KIDS/"+folder+"/"+grating+"/"
#    file1r = path_galaxy+"16jan20058red.fits"
#    file2r = path_galaxy+"16jan20059red.fits"
#    file3r = path_galaxy+"16jan20060red.fits"
#    fits_file_red = path_galaxy+galaxy+"_"+grating+"_"+date+".fits"


#    POX4_r1 = KOALA_RSS(file1r,
#                          apply_throughput=True, skyflat = skyflat_red, nskyflat=False, plot_skyflat=False,
#                          correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 100, step_ccd=50,
#                          telluric_correction = telluric_correction_20160116red,
#                          do_sky=True, n_sky=30, # sky_spectrum=sky_spectrum2, sky_exptime=600.,
#                          cut_sky=3., fmin=2.5, fmax=4., individual_sky_substraction=False, fibre_list=[100,200,300,400,500,600,700,800,900],
#                          id_el=True, high_fibres=10, brightest_line="Ha", cut=1.1, broad=1.5, plot_id_el= True,
#                          id_list=[6300.30, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7135.78, 7318.39, 7329.66 ],
#                          clean_sky_residuals = True, dclip=2.0, extra_w = 1.3, step_csr = 100, fibre=427,
#                          valid_wave_min = 6271, valid_wave_max = 7400,
#                          plot=True, warnings=False)


#    POX4_r1_cube=Interpolated_cube(POX4_r1, .6, 1.25, plot=True)


#    rss_list=[file1r,file2r,file3r]
#    POX4red=KOALA_reduce(rss_list, fits_file=fits_file_red, obj_name="POX4",  description = "POX 4 - 3 cubes",
#                     # RSS
#                     apply_throughput=True, skyflat = skyflat_red, nskyflat=False, skyflat_file="", # nskyflat is that we will use the normalized skyflat in sky_file.skyflat for throughput correction, step is to get smooth median flatfield
#                     #skyflat_list=["","","","","","",""],
#                     correct_ccd_defects = True, correct_high_cosmics = False, clip_high = 50, step_ccd = 50,
#                     telluric_correction = telluric_correction_20160116red,
#                     #telluric_correction_list=telluric_correction_list,
#                     do_sky=True, n_sky=30, #sky_fibres=[1000], sky_spectrum=sky_spectrum,  #sky_exptime=[600.],
#                     #sky_list=sky_list,
#                     #cut_sky=5., fmin=1, fmax=10, individual_sky_substraction = False,
#
#                     id_el=True, high_fibres=10, brightest_line="Ha", cut=1.1, broad=1.5, plot_id_el= True,
#                     id_list=[6300.30, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7135.78, 7318.39, 7329.66 ],
#                     clean_sky_residuals = True, dclip=2.0, extra_w = 1.3, step_csr = 25,
# 		          do_extinction=True,
#                     # CUBING
#                     pixel_size_arcsec=.6, kernel_size_arcsec=1.25,
#                     #offsets=[1000],
#                     ADR=False,
#                     fcal=True, flux_calibration=flux_calibration_20160116red_0p7_2k1,
#                     #flux_calibration_list=[[0],[0],[0],[0],[0],[0],[0]],
#                     # COMMON
#                     valid_wave_min = 6271, valid_wave_max = 7400,
#                     plot= True, norm=colors.LogNorm(), fig_size=12,
#                     warnings=False, verbose = False)


# offsets= [0.53484419813461281, 1.1422154711303474, 2.0030958654779734, -0.29545721118086909]


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# POX 4 BLUE
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


#    path_main ="/DATA/KOALA/"
#    date="20160116"
#    grating="580V"
#
#
# #
# # THROUGHPUT CORRECTION USING SKYFLAT
# #
# # The very first thing that we need is to get the throughput correction.
# # First, provide path, skyflat file and name of the file that will keep the throughput correction
#    path_flat = path_main+"SKYFLATS/"+date+"/"+grating+"/"
#    file_skyflatb=path_flat+"16jan1_combined.fits"
#    throughput_file_blue=path_flat+date+"_"+grating+"_throughput_correction.dat"
#
# # If this has been done before, we can read the file containing the throughput correction
#    throughput_blue = read_table(throughput_file_red, ["f"] )
#
# # Now we read the RSS file, we ONLY correct for ccd defects and high cosmics
#    skyflat_blue = KOALA_RSS(file_skyflatb, do_sky=False, do_extinction=False, apply_throughput=False, plot=True,
#                            correct_ccd_defects = False, correct_high_cosmics = True, clip_high = 100, step_ccd = 50)
#
# # Next we find the relative throughput.
# # If the data have been normalized by the FLATFIELD, we only need a SCALE between fibres,
# # choose nskyflat=False and ignore step and fit_skyflat_degree (this is for creating a normalized flatfield here)
# # We consider the median value in the range [wave_min_scale, wave_max_scale] for all fibres and scale
#
#    skyflat_blue.find_relative_throughput(ymin=0, ymax=300000, step=80, fit_skyflat_degree = 3,   # Try fit_skyflat_degree = 0, wave_min_scale=6630, wave_max_scale=6820,
#                                         wave_min_scale=4000, wave_max_scale=5500, wave_max_flat=5700, nskyflat=True)
#
# # The relative throughput is an array stored in skyflat_red.relative_throughput
# # We save that array in a text file that we can read in the future without the need of repeating this
#
#    array_to_text_file(skyflat_blue.relative_throughput, filename= throughput_file_blue )
#
#

# #
# # OBTAIN ABSOLUTE FLUX CALIBRATION AND TELLURIC CORRECTION USING CALIBRATION STARS
# #
#
# # If these have been obtained already, we can read files containing arrays with the calibrations
# # Uncomment the next two sections and skip the rest till "OBTAIN SKY SPECTRA"
#
# # READ FLUX CALIBRATION BLUE
#    flux_cal_file = path_main+"/FLUX_CAL/"+date+"_"+grating+"_0p7_2k1_FLUX_CAL.dat"
#    w_star,flux_calibration_20161024blue_0p7_2k1 = read_table(flux_cal_file, ["f", "f"] )
#    print flux_calibration_20161024blue_0p7_2k1
#


# # READ STAR 1  ... NO FOR POX4 (different wavelength)
# # First we provide names, paths, files...
#    star1="EG21"
#    path_star1 = path_main+"STARS/"+star1+"/"+date+"/"+grating+"/"
#    starpos1b = path_star1+"16jan10046red.fits"
#    fits_file_red = path_star1+star1+"_"+grating+"_"+date+".fits"
#    text_file_red = path_star1+star1+"_"+grating+"_"+date+"_response.dat"
#
# # Now we read RSS file
# # We apply throughput correction (using nskyflat=True), substract sky using n_sky=600 lowest intensity fibres,
# # correct for CCD defects and high cosmics
#
#    star1b = KOALA_RSS(starpos1b,
#                       apply_throughput=True, skyflat = skyflat_red, nskyflat=False, plot_skyflat=False,
#                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 100, step_ccd=50,
#                       do_sky=True, n_sky=600,
#                       valid_wave_min = 6271, valid_wave_max = 7400,
#                       plot=True, warnings=True)
#
# # PAY ATTENTION to plots.
#                     #
# # Next we CREATE THE CUBE for this star, using THE SAME PARAMETERS we will later using for our objects
# # 0.6 is the pixel size, 1.25 is the kernel size.
#
#    cubes1r=Interpolated_cube(star1r, .6, 1.25, plot=True)
#
# # Now we read the absolute flux calibration data of the calibration star and get the response curve
# # (Response curve: correspondence between counts and physical values)
# # Include exp_time of the calibration star, as the results are given per second
# # For this BE CAREFUL WITH ABSORPTIONS (Halpha) and check behaviour in the edges of the CCD
#
#    cubes1r.do_response_curve('FLUX_CAL/feg21_edited.dat', plot=True, min_wave=6280., max_wave=7400.,
# 			       step=10, exp_time=120., fit_degree=3)
#
# # Now we can save this calibration as a text file
#    spectrum_to_text_file(cubes1r.response_wavelength,cubes1r.response_curve, filename=text_file_red)
#


# # REPEAT FOR STAR 2
#
#
#    star2="H600"
#    path_star2 = path_main+"STARS/"+star2+"/"+date+"/"+grating+"/"
#    starpos2b = path_star2+"16jan10052red.fits"
#    fits_file_blue = path_star2+star2+"_"+grating+"_"+date+".fits"
#    text_file_blue = path_star2+star2+"_"+grating+"_"+date+"_response.dat"

#    star2b = KOALA_RSS(starpos2b,
#                       apply_throughput=True, skyflat = skyflat_blue, nskyflat=False, plot_skyflat=False,
#                       do_sky=True, n_sky=500, remove_5578 = False, sky_wave_min = 4500, sky_wave_max=5000,
#                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
#                       valid_wave_min = 3700, valid_wave_max = 5600,
#                       plot=True, warnings=False)
#
#    cubes2b=Interpolated_cube(star2b, .6, 1.25, plot=True)
#    cubes2b.do_response_curve('FLUX_CAL/fhilt600_edited.dat', plot=True, min_wave=3650., max_wave=5755., step=15, smooth=0.03, exp_time=120., fit_degree=0)
#    spectrum_to_text_file(cubes2b.response_wavelength,cubes2b.response_curve, filename=text_file_blue)


# # REPEAT FOR STAR 3

#    star3="Feige56"
#    path_star3 = path_main+"STARS/"+star3+"/"+date+"/"+grating+"/"
#    starpos3b = path_star3+"16jan10064red.fits"
#    fits_file_blue = path_star3+star3+"_"+grating+"_"+date+".fits"
#    text_file_blue = path_star3+star3+"_"+grating+"_"+date+"_response.dat"

#    star3b = KOALA_RSS(starpos3b,
#                       apply_throughput=True, skyflat = skyflat_blue, nskyflat=False, plot_skyflat=False,
#                       do_sky=True, n_sky=600,  remove_5578 = False,
#                       correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 50, step_ccd=50,
#                       valid_wave_min = 3700, valid_wave_max = 5650,
#                       plot=True, warnings=True)
#
#
#    cubes3b=Interpolated_cube(star3b, .6, 1.25, plot=True)
#    cubes3b.do_response_curve('FLUX_CAL/ffeige56_edited.dat', plot=True, min_wave=3670., max_wave=5755., step=15, exp_time=120., fit_degree=0, smooth=0.04)
#    spectrum_to_text_file(cubes3b.response_wavelength,cubes3b.response_curve, filename=text_file_blue)


# #  CHECK AND GET THE FLUX CALIBRATION FOR THE NIGHT  RED

# #  First we take another look to the RSS data plotting the integrated fibre values in a map
#    star1b.RSS_map(star1b.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))
#    star2b.RSS_map(star2b.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))
#    star3b.RSS_map(star3b.integrated_fibre, norm=colors.PowerNorm(gamma=1./4.))


# # Define in "stars" the 2 cubes we are using, and plotting their responses to check
#    stars=[cubes2b,cubes3b]
#    plot_response(stars)

# # Anyway, that is what it is, we obtain the flux calibration applying:
#    flux_calibration_20160116blue_0p6_1k25=obtain_flux_calibration(stars)

# # And we save this absolute flux calibration as a text file
#    flux_cal_file = path_main+"/FLUX_CAL/"+date+"_"+grating+"_0p7_2k1_FLUX_CAL.dat"
#    spectrum_to_text_file(star2b.wavelength,flux_calibration_20160116blue_0p6_1k25, filename=flux_cal_file)


# #  GALAXY

#    galaxy="POX4"
#    folder="03_POX4"
#    path_galaxy = path_main+"Hi-KIDS/"+folder+"/"+grating+"/"
#    file1b = path_galaxy+"16jan10058red.fits"
#    file2b = path_galaxy+"16jan10059red.fits"
#    file3b = path_galaxy+"16jan10060red.fits"
#    fits_file_red = path_galaxy+galaxy+"_"+grating+"_"+date+".fits"


#    POX4_b2 = KOALA_RSS(file2b,
#                          apply_throughput=True, skyflat = skyflat_blue, nskyflat=False, plot_skyflat=False,
#                          correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 325, step_ccd=50,
#                          plot_suspicious_fibres=False, remove_5578 = True,
#                          #telluric_correction = telluric_correction_20160116red,
#                          do_sky=True, n_sky=30, # sky_spectrum=sky_spectrum2, sky_exptime=600.,
#                          cut_sky=3., fmin=2.5, fmax=4., individual_sky_substraction=False, fibre_list=[100,200,300,400,500,600,700,800,900],
#                          id_el=False, high_fibres=10, brightest_line="O3b", cut=1.1, broad=1.5, plot_id_el= True,
#                          id_list=[3727.30, 4340.47, 4861.33, 4958.91,5006.84 ],
#                          clean_sky_residuals = False, dclip=2.0, extra_w = 1.3, step_csr = 100, fibre=427,
#                          valid_wave_min = 3600, valid_wave_max = 5700,
#                          plot=True, warnings=False, verbose=True)

#    POX4_b2_cube=Interpolated_cube(POX4_b2, .6, 1.25, plot=True, ADR=False, flux_calibration=flux_calibration_20160116blue_0p6_1k25)
#    save_fits_file(POX4_b2_cube, path_galaxy+galaxy+"_b2_cube_test.fits", ADR=False)


#    rss_list=[file1b,file2b,file3b]
#    POX4red=KOALA_reduce(rss_list, fits_file=fits_file_red, obj_name="POX4",  description = "POX 4 BLUE - 3 cubes",
#                     # RSS
#                     apply_throughput=True, skyflat = skyflat_blue, nskyflat=False, skyflat_file="",
#                     #skyflat_list=["","","","","","",""],
#                     correct_ccd_defects = True, correct_high_cosmics = True, clip_high = 330, step_ccd = 50,
#                     plot_suspicious_fibres=False, remove_5578 = True,
#                     #telluric_correction = telluric_correction_20160116red,
#                     #telluric_correction_list=telluric_correction_list,
#                     do_sky=True, n_sky=30, #sky_fibres=[1000], sky_spectrum=sky_spectrum,
#                     #sky_list=sky_list,
#                     #cut_sky=5., fmin=1, fmax=10, individual_sky_substraction = False,
#                     id_el=False, high_fibres=10, brightest_line="O3b", cut=1.1, broad=1.5, plot_id_el= True,
#                     id_list=[3727.30, 4340.47, 4861.33, 4958.91,5006.84 ],
#                     clean_sky_residuals = False, dclip=2.0, extra_w = 1.3, step_csr = 25,
# 		          do_extinction=True,
#                     # CUBING
#                     pixel_size_arcsec=.6, kernel_size_arcsec=1.25,
#                     offsets= [0.53484419813461281, 1.1422154711303474, 2.0030958654779734, -0.29545721118086909],
#                     ADR=False,
#                     flux_calibration=flux_calibration_20160116blue_0p6_1k25,
#                     #flux_calibration_list=[[0],[0],[0],[0],[0],[0],[0]],
#                     # COMMON
#                     valid_wave_min = 3600, valid_wave_max = 5700,
#                     plot= True, norm=colors.LogNorm(), fig_size=12,
#                     warnings=False, verbose = True)
#
#
#                     #offsets= [0.53484419813461281, 1.1422154711303474, 2.0030958654779734, -0.29545721118086909]


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
# --------------------------  OLD THINGS --------------------------------------
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
# Align n- cubes, testing...
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#    cube1=LG02rpos1_cube
#    cube2=LG02rpos1_cube
#    cube3=LG02rpos1_cube
#    cube4=LG02rpos1_cube
#    cube5=LG02rpos1_cube
#    rss1=LG02rpos1
#    rss2=LG02rpos1
#    rss3=LG02rpos1
#    rss4=LG02rpos1
#    rss5=LG02rpos1
#    cube_list=[cube1,cube2,cube3,cube4,cube5]
#
#    rss_list=[rss1,rss2,rss3,rss4,rss5]
#    offsets=[1,1,2,2,-3,-3,4,4]
#    cube2.RA_centre_deg=192.431
#    cube_aligned_list=align_n_cubes(cube_list, rss_list, pixel_size_arcsec=0.3, kernel_size_arcsec=1.5, offsets=offsets, plot= False, ADR=False, warnings=False)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# OBJETO DE Lluis
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


#    directorio_main ="/DATA/KOALA"


###### RED

#    date="20180226"
#    grating="385R"

#    SKYFLAT   RED

#    directorio_flat = directorio_main+"/SKYFLATS/"+date+"/"+grating+"/"
#    skyflatr=directorio_flat+"26feb2_combined.fits"
#    skyflat_red = KOALA_RSS(skyflatr, do_sky=False, do_extinction=False, apply_throughput=False, plot=False)
#    skyflat_red.find_relative_throughput(ymax=330000, step=40, fit_skyflat_degree = 3)
#    array_to_text_file(skyflat_red.relative_throughput, filename=directorio_flat+"skyflat_"+grating+"_"+date+".dat" )


#   FLUX CALIBRATION STAR 1   RED

#    star1="CD-32d9927"
#    directorio_star1 = directorio_main+"/STARS/"+star1+"/"+date+"/"+grating+"/"
#    starpos1r = directorio_star1+"26feb20027red.fits"
#    starpos2r = directorio_star1+"26feb20028red.fits"
#    starpos3r = directorio_star1+"26feb20029red.fits"
#    fits_file_red = directorio_star1+star1+"_"+grating+"_"+date+".fits"
#    text_file_red = directorio_star1+star1+"_"+grating+"_"+date+"_response.dat"

#    CD32d9927red=KOALA_reduce([starpos1r,starpos2r,starpos3r],
#                            fits_file=fits_file_red,
#                            skyflat = skyflat_red, nskyflat=True,
#                            correct_c_d = True, brightest_line="Ha",
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                            plot= True #, offsets=[-3.3, -0.09,  0, 3.2]
#                            )

#    CD32d9927red.combined_cube.do_response_curve('FLUX_CAL/fcd32d9927.dat', plot=True, step=20, smooth=31, exp_time=120., fit_degree=3, max_wave=9200,min_wave=6080) # Best values without telluric correction
#    CD32d9927red.combined_cube.do_response_curve('FLUX_CAL/fcd32d9927.dat', plot=True, step=50, smooth=11, exp_time=120., fit_degree=3, max_wave=9300,min_wave=6120)  #min_wave=3660., max_wave=5755.,
#    spectrum_to_text_file(CD32d9927red.combined_cube.response_wavelength,CD32d9927red.combined_cube.response_curve, filename=text_file_red)


#   FLUX CALIBRATION STAR 2 RES

#    star="EG274"
#    directorio_star = directorio_main+"/STARS/"+star+"/"+date+"/"+grating+"/"
#    starpos1r = directorio_star+"26feb20037red.fits"
#    starpos2r = directorio_star+"26feb20038red.fits"
#    starpos3r = directorio_star+"26feb20039red.fits"
#    fits_file_red = directorio_star+star+"_"+grating+"_"+date+".fits"
#    text_file_red = directorio_star+star+"_"+grating+"_"+date+"_response.dat"
#
#    EG274red=KOALA_reduce([starpos1r,starpos2r,starpos3r],
#                            fits_file=fits_file_red,
#                            skyflat = skyflat_red, nskyflat=True,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                            plot= True
#                            )
#
#    EG274red.combined_cube.do_response_curve('FLUX_CAL/feg274.dat', plot=True, step=100, smooth=20, exp_time=180.,  max_wave=9250, min_wave=6420,fit_degree=3) # Bueno para no telluric
#    EG274red.combined_cube.do_response_curve('FLUX_CAL/feg274.dat', plot=True, step=70, smooth=11, exp_time=180., fit_degree=3, max_wave=9300 ,min_wave=6200)
#    spectrum_to_text_file(EG274red.combined_cube.response_wavelength,EG274red.combined_cube.response_curve, filename=text_file_red)


#   FLUX CALIBRATION STAR 3 RED

#    star="HD49798"
#    directorio_star = directorio_main+"/STARS/"+star+"/"+date+"/"+grating+"/"
#    starpos1r = directorio_star+"26feb20024red.fits"
#    starpos2r = directorio_star+"26feb20025red.fits"
#    starpos3r = directorio_star+"26feb20026red.fits"
#    fits_file_red = directorio_star+star+"_"+grating+"_"+date+".fits"
#    text_file_red = directorio_star+star+"_"+grating+"_"+date+"_response.dat"
#
#    HD49798red=KOALA_reduce([starpos1r,starpos2r,starpos3r],
#                            fits_file=fits_file_red, n_sky=400,
#                            skyflat = skyflat_red, nskyflat=True,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                            plot= True
#                            )

#    HD49798red.combined_cube.do_response_curve('FLUX_CAL/fhd49798.dat', plot=True, step=20, smooth=21, exp_time=60.,fit_degree=3,  max_wave=9150, min_wave=6080)   # Bueno sin telluric
#    HD49798red.combined_cube.do_response_curve('FLUX_CAL/fhd49798.dat', plot=True, step=100, smooth=21, exp_time=60.,fit_degree=3,  max_wave=9300, min_wave=6140)
#    spectrum_to_text_file(HD49798red.combined_cube.response_wavelength,HD49798red.combined_cube.response_curve, filename=text_file_red)


#    CHECK AND GET FLUX CALIBRATION FOR THE NIGHT  RED

#    stars=[EG274red.combined_cube,HD49798red.combined_cube,CD32d9927red.combined_cube]
#    plot_response(stars)
##    # NOTE: Problem with HD49798blue, check ESO spectr. stars documentation
##    # Multiply HD49798blue x 1.1744 :    checking some numbers, better using 1.16
##    # Now similar to CD-32d9927, observed at the same time.
#    print EG274red.combined_cube.response_curve[1000]/HD49798red.combined_cube.response_curve[1000]
#    HD49798red.combined_cube.response_curve=HD49798red.combined_cube.response_curve/1.03*1.02
#    plot_response(stars)
#
#    flux_calibration_20180226_red_0p7_2k1=obtain_flux_calibration(stars)
#    spectrum_to_text_file(EG274red.combined_cube.wavelength,flux_calibration_20180226_red_0p7_2k1, filename=directorio_main+"/FLUX_CAL/20180226_385R_0p7_2k1_FLUX_CAL.dat" )


#    w_star,flux_calibration_20180226_red_0p7_2k1 = read_table(directorio_main+"/FLUX_CAL/20180226_385R_0p7_2k1_FLUX_CAL.dat", ["f", "f"] )
#    print flux_calibration_20180226_red_0p7_2k1


#   OBJECT RED

#    galaxia="LG02"
#    directorio_galaxia = directorio_main+"/Lluis_SN/"+galaxia+"/"+grating+"/"
#    file1r = directorio_galaxia+"26feb20034red.fits"
#    file2r = directorio_galaxia+"26feb20035red.fits"
#    file3r = directorio_galaxia+"26feb20036red.fits"
#    fits_file_red = directorio_galaxia+galaxia+"_"+grating+"_"+date+"_TEST.fits"

#    LG02rpos1 = KOALA_RSS(file1r, n_sky=25, skyflat=skyflat_red, nskyflat = True, plot_skyflat=False, plot=True, correct_c_d = True, brightest_line="Ha", do_sky=True, clean_sky_residuals = False, clean_telluric = False) #, do_sky=False ) #fcd32d9927
#    LG02rpos1_cube=Interpolated_cube(LG02rpos1, .7, 2.1, plot=True, flux_calibration=flux_calibration_20180226_red_0p7_2k1)
#    save_fits_file(LG02rpos1_cube, directorio_galaxia+"test.fits", ADR=False)

#    LG02rpos1.correct_high_cosmics_and_defects(clip_high=10)


#
##
#    fibre=408
#    wlm=LG02rpos1.wavelength
#
##    fig_size=12
##    plt.figure(figsize=(fig_size, fig_size/2.5))
##
###    sky100=LG02rpos1.intensity_sky_corrected[fibre]
###    sky25=LG02rpos1.intensity_sky_corrected[fibre]
##
###    plt.plot(wlm,LG02rpos1.sky_emission,"r-", alpha=0.1)
##
##    plt.plot(wlm,LG02rpos1.intensity_corrected[fibre], "b-", alpha=0.5)
##    plt.plot(wlm,LG02rpos1.intensity[fibre], "r-", alpha=0.2)
###    plt.plot(wlm,LG02rpos1.intensity_sky_corrected[fibre],"g-", alpha=0.5)
###    plt.plot(wlm,sky100,"k-", alpha=0.5)
###    plt.plot(wlm,sky25,"k-", alpha=0.5)
##
###    plt.plot(wlm,LG02rpos1.intensity_corrected[fibre]-0.9*LG02rpos1.sky_emission,"y-", alpha=0.5)
###    plt.plot(wlm,LG02rpos1.intensity_corrected[fibre]-1.1*LG02rpos1.sky_emission,"b-", alpha=0.5)
##
###    plt.xlim(8400,8800)
##    plt.xlim(7000,7500)
##    plt.ylim(200,600)
##    plt.show()
##    plt.close()
#
#    plot=False
#    espectro=LG02rpos1.intensity_corrected[fibre]
#
#    resultado = fluxes(wlm, espectro, 6870, lowlow= 200, lowhigh=80, highlow=40, highhigh = 100, lmin=6660, lmax=7000, broad=5, plot=plot, verbose=False)#, plot_sus=True)
#    espectro=resultado[11]
#    resultado = fluxes(wlm, espectro, 6890, lowlow= 200, lowhigh=90, highlow=40, highhigh = 100, lmin=6680, lmax=7000, broad=5, plot=plot, verbose=False)#, plot_sus=True)
#    espectro=resultado[11]
#    resultado = fluxes(wlm, espectro, 7210, lowlow= 40, lowhigh=15, highlow=30, highhigh = 50, lmin=6900, lmax=7600, broad=10, plot=plot, verbose=False)#, plot_sus=True)
#    espectro=resultado[11]
#    resultado = fluxes(wlm, espectro, 7230, lowlow= 200, lowhigh=120, highlow=120, highhigh = 200, lmin=6900, lmax=7600, broad=150,plot=plot, verbose=False)#, plot_sus=True)
#    espectro=resultado[11]
#    resultado = fluxes(wlm, espectro, 7616, lowlow= 200, lowhigh=80, highlow=80, highhigh = 200, broad=-25, lmin=7300, lmax=7900, plot=plot, verbose=False) #, plot_sus=True)
#    espectro=resultado[11]
#    resultado = fluxes(wlm, espectro, 8150, lowlow= 250, lowhigh=100, highlow=150, highhigh = 200, broad=-40, lmin=7900, lmax=8500, plot=plot, verbose=False)#, plot_sus=True)
#    espectro=resultado[11]
#    resultado = fluxes(wlm, espectro, 8670, lowlow= 200, lowhigh=100, highlow=100, highhigh = 200, broad=-25, lmin=8400, lmax=8900, plot=plot, verbose=False)#, plot_sus=True)
#    espectro=resultado[11]
#    resultado = fluxes(wlm, espectro, 8670, lowlow= 200, lowhigh=100, highlow=20, highhigh = 40, broad=-6, lmin=8400, lmax=8900, plot=plot, verbose=False)#, plot_sus=True)
#    espectro=resultado[11]
#    resultado = fluxes(wlm, espectro, 8620, lowlow= 100, lowhigh=20, highlow=20, highhigh = 80, broad=4, lmin=8400, lmax=8900, plot=plot, verbose=False)#, plot_sus=True)
#    espectro=resultado[11]
#    resultado = fluxes(wlm, espectro, 8785, lowlow= 100, lowhigh=20, highlow=20, highhigh = 100, broad=5, lmin=8600, lmax=9100, plot=plot, verbose=False)#, plot_sus=True)
#    espectro=resultado[11]
#    resultado = fluxes(wlm, espectro, 8987, lowlow= 150, lowhigh=60, highlow=80, highhigh = 200, broad=-20, lmin=8750, lmax=9300, plot=plot, verbose=False)#, plot_sus=True)
#    espectro=resultado[11]
#
#
#
#    fig_size=12
#    plt.figure(figsize=(fig_size, fig_size/2.5))
#    plt.plot(wlm,LG02rpos1.intensity_corrected[fibre], "r-", alpha=0.2)
#
#    plt.plot(wlm,espectro, "b-", alpha=0.5)
#
#
###    plt.xlim(7200,8000)
##    plt.xlim(6100,6600)
##    plt.xlim(6600,7200)
##    plt.xlim(7200,8200)
##    plt.xlim(8200,9400)
##    plt.xlim(8000,8500)
#    plt.ylim(200,1000)
#    plt.show()
#    plt.close()
##

#
#
#   # plt.plot(LG02rpos1.wavelength,LG02rpos1.intensity_corrected[202], "b-", alpha=0.5)
#
##    extra_w=1.3
#    exclude_ranges_low=[]
#    exclude_ranges_high=[]
#    print "  Emission lines identified:"
#    for el in range(len(LG02rpos1.el[0])):
#        print el+1, LG02rpos1.el[0][el],LG02rpos1.el[1][el],LG02rpos1.el[2][el],LG02rpos1.el[3][el]
#        exclude_ranges_low.append(LG02rpos1.el[2][el]- LG02rpos1.el[3][el]*1.3)    # center-1.3*FWHM
#        exclude_ranges_high.append(LG02rpos1.el[2][el]+ LG02rpos1.el[3][el]*1.3)    # center+1.3*FWHM
#
#
##    plt.plot(LG02rpos1.wavelength,LG02rpos1.intensity_corrected[203], "b-", alpha=1)
#
#    step=25
#    verbose=False
#    wave_min=wlm[0]
#    wave_max=wlm[-1]
#    say_status = 0
#    for fibre in range(LG02rpos1.n_spectra): # range(203,204): # range(LG02rpos1.n_spectra): #range(LG02rpos1.n_spectra):
#        if fibre == say_status :
#            print " Checking fibre ", fibre," ..."
#            say_status=say_status+100
#
#        s = LG02rpos1.intensity_corrected[fibre]
#
#        smooth_points=np.int(LG02rpos1.n_wave/step)
#        running_wave = np.zeros(smooth_points+1)
#        running_step_median = np.zeros(smooth_points+1)
#
#        for j in range(smooth_points):
#            running_wave[j] = np.nanmedian([wlm[i] for i in range(len(wlm)) if (i > step*j and i<step*(j+1))])
#            running_step_median[j] = np.nanmedian([s[i] for i in range(len(wlm)) if (i > step*j and i<step*(j+1))])   # / np.nanmedian(spectrum)
#            #print j,running_wave[j], running_step_median[j]
#        running_wave[-1]=wave_max
#        running_step_median[-1]=np.nanmedian([s[i] for i in range(len(wlm)) if (i > step*(j+1) and i < LG02rpos1.n_wave)])
#
#        interpolated_continuum_smooth = interpolate.splrep(running_wave, running_step_median, s=0)
#        fit_median = interpolate.splev(wlm, interpolated_continuum_smooth, der=0)
#
#        pclip=0.25
##        plt.plot(wlm,fit_median,"r-", alpha=0.5)
##        plt.plot(wlm,fit_median*(1+pclip),"r", alpha=0.5)
##        plt.plot(wlm,fit_median*(1-pclip),"r", alpha=0.5)
##        plt.plot(running_wave,running_step_median, "ro")
##        plt.plot(LG02rpos1.wavelength,s, "b-", alpha=0.5)
#
#        rango=0
#        imprimir = 1
#        espectro=np.zeros(LG02rpos1.n_wave)
##        for i in range(100,500):
#        for i in range(len(wlm)):
#            espectro[i] = s[i]
#            if wlm[i] >=  exclude_ranges_low[rango] and wlm[i] <=  exclude_ranges_high[rango]:
#                if verbose == True : print "  Excluding range [",exclude_ranges_low[rango],",",exclude_ranges_high[rango],"] as it has an emission line"
#                if imprimir == 1 : imprimir = 0
#                #print "    Checking ", wlm[i]," NOT CORRECTED ",s[i], s[i]-fit_median[i]
#            else:
#                if s[i] < 0: espectro[i] = fit_median[i]                        # Negative values for median values
#                if np.isnan(s[i]) == True : espectro[i] = fit_median[i]         # nan for median value
#                if s[i] >(1+pclip)*fit_median[i]:
#                    #if verbose: print "  CLIPPING HIGH=",(1+pclip),"in fibre",fibre," wave=",wlm[wave]," value=",s[wave]," median=",fit_median[wave]
#                    espectro[i] = fit_median[i]
#                if s[i] < (1-pclip)*fit_median[i]:
#                   # if verbose: print "  CLIPPING LOW=",(1-pclip),"in fibre",fibre," wave=",wlm[wave]," value=",s[wave]," median=",fit_median[wave]
#                    espectro[i] = fit_median[i]
#
#                if wlm[i] >  exclude_ranges_high[rango] and imprimir == 0:
#                    if verbose: print "  Checked", wlm[i],"  End rango ",rango,exclude_ranges_low[rango],exclude_ranges_high[rango]
#                    rango = rango+1
#                    imprimir = 1
#                if rango == len(exclude_ranges_low): rango = len(exclude_ranges_low)-1
#                #print "    Checking ", wlm[i]," CORRECTED IF NEEDED",s[i], s[i]-fit_median[i]
#
##
#        LG02rpos1.intensity_corrected[fibre,:] = espectro
##        plt.plot(LG02rpos1.wavelength,espectro, "g-", alpha=0.8)
#
##        plt.plot(wlm,intensity_corrected[100])
#
#
#
##    plt.xlim(7200,8000)
##    plt.xlim(6300,7000)
#    plt.ylim(0,600)
#    plt.show()
#    plt.close()
#
#    LG02rpos1intensity_corrected=clean_cosmics_keeping_el(LG02rpos1.wavelength, LG02rpos1.intensity_corrected, clip=10, brightest_line="Ha")

#    print LG02rpos1.CRVAL1_CDELT1_CRPIX1
#    LG02red.combined_cube.CRVAL1_CDELT1_CRPIX1=LG02rpos1.CRVAL1_CDELT1_CRPIX1

#    rss_list=[file1r,file2r,file3r]
#    LG02red=KOALA_reduce(rss_list, fits_file=fits_file_red, obj_name="LG02 - SN2005bo",
#                          skyflat = skyflat_red, nskyflat=True, n_sky=25,
#                          correct_c_d = True, brightest_line="Ha",
#                          clean_sky_residuals = False, clean_telluric = False,
#                          pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                          flux_calibration=flux_calibration_20180226_red_0p7_2k1,
#                          plot= True, warnings=False  #,
##                          offsets=[0.8, 0.96,  1.39, 1.66]
#                          )
#
##    fits_file_red = directorio_galaxia+galaxia+"_"+grating+"_"+date+"_TEST.fits"
#    save_fits_file(LG02red.combined_cube, fits_file_red, ADR=False)


#    fits_file_red = directorio_galaxia+galaxia+"_"+grating+"_"+date+"_DOS.fits"

#    rss_list=[file1r,file2r] #,file3r]
#    LG02dos=KOALA_NEW_reduce(rss_list, fits_file=fits_file_red,
#                          obj_name="LG02 - DOS",
#                          skyflat = skyflat_red, nskyflat=True, n_sky=25,
#                          correct_c_d = True, brightest_line="Ha",
#                          clean_sky_residuals = False, clean_telluric = False,
#                          pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                          flux_calibration=flux_calibration_20180226_red_0p7_2k1,
#                          plot= False, warnings=False  #,
##                          offsets=[0.8, 0.96,  1.39, 1.66]
#                          )


# CRVAL1  =   7.692370611909E+03 / Co-ordinate value of axis 1
# CDELT1  =   1.575182431607E+00 / Co-ordinate increment along axis 1

#    for i in range(LG02red.rss1.n_wave):
#        if LG02red.rss1.wavelength[i] == 7.692370611909E+03:
#            centro = i+1  # centro is a natural number
#            print centro, LG02red.rss1.wavelength[centro-1]
#    print LG02red.rss1.wavelength[0], LG02red.rss1.wavelength[0]-(LG02red.rss1.wavelength[centro-1]-1.575182431607E+00*(centro-1))
#    print LG02red.rss1.wavelength[-1], LG02red.rss1.wavelength[-1]-(LG02red.rss1.wavelength[centro-1]+1.575182431607E+00*(centro))


# LG02red.combined_cube.offsets_files
# offsets_OLD=[ -0.88019215901180736, -1.1088363999158928,  1.8011970718313495, -1.4720721710629561]
# offsets_OLD=[-0.87357246943603117, -1.1077408654643861, 1.797154315015268, -1.46743147005553]
#    offsets=[-0.86362073903839276, -1.0834011783531168, 1.7816598448354282, -1.4650780719338137]

# PROXIMO: USAR  estos offsets para el cubo azul, combinar los dos, check fcal, y conseguir wcs = lambda


######### BLUE part

#    date="20180226"
#    grating="580V"


#    SKYFLAT   BLUE

#    directorio_flat = directorio_main+"/SKYFLATS/"+date+"/"+grating+"/"
#    skyflatb=directorio_flat+"26feb1_combined.fits"
#    skyflat_blue = KOALA_RSS(skyflatb, do_sky=False, do_extinction=False, apply_throughput=False, plot=False)
#    skyflat_blue.find_relative_throughput(ymax=400000, step=40, fit_skyflat_degree = 3)
#    array_to_text_file(skyflat_blue.relative_throughput, filename=directorio_flat+"skyflat_"+grating+"_"+date+".dat" )

#    star1="CD-32d9927"
#    date="20180226"
#    grating="580V"
#
#    starpos1 = directorio+"/STARS/"+star1+"/"+date+"/"+grating+"/26feb10027red.fits"
#
#    star1b = KOALA_RSS(starpos1, n_sky=600, skyflat=skyflat_blue, nskyflat=True)


#   FLUX CALIBRATION STAR 1   BLUE

#    star1="CD-32d9927"
#    directorio_star1 = directorio_main+"/STARS/"+star1+"/"+date+"/"+grating+"/"
#    starpos1b = directorio_star1+"26feb10027red.fits"
#    starpos2b = directorio_star1+"26feb10028red.fits"
#    starpos3b = directorio_star1+"26feb10029red.fits"
#    fits_file_blue = directorio_star1+star1+"_"+grating+"_"+date+".fits"
#    text_file_blue = directorio_star1+star1+"_"+grating+"_"+date+"_response.dat"
#
#    CD32d9927blue=KOALA_reduce([starpos1b,starpos2b,starpos3b],
#                            fits_file=fits_file_blue,
#                            skyflat = skyflat_blue, nskyflat=True,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                            plot= True #, offsets=[-3.3, -0.09,  0, 3.2]
#                            )

#    CD32d9927blue.combined_cube.do_response_curve('FLUX_CAL/fcd32d9927.dat', plot=True, step=30, smooth=21, exp_time=120., fit_degree=7)
#    spectrum_to_text_file(CD32d9927blue.combined_cube.response_wavelength,CD32d9927blue.combined_cube.response_curve, filename=text_file_blue )


#   FLUX CALIBRATION STAR 2

#    star2="EG274"
#    directorio_star2 = directorio_main+"/STARS/"+star2+"/"+date+"/"+grating+"/"
#    starpos1b = directorio_star2+"26feb10037red.fits"
#    starpos2b = directorio_star2+"26feb10038red.fits"
#    starpos3b = directorio_star2+"26feb10039red.fits"
#    fits_file_blue = directorio_star2+star2+"_"+grating+"_"+date+".fits"
#    text_file_blue = directorio_star2+star2+"_"+grating+"_"+date+"_response.dat"
#
#    EG274blue=KOALA_reduce([starpos1b,starpos2b,starpos3b],
#                            fits_file=fits_file_blue,
#                            skyflat = skyflat_blue, nskyflat=True,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                            plot= False
#                            )

#    EG274blue.combined_cube.do_response_curve('FLUX_CAL/feg274.dat', plot=True, step=40, smooth=21, exp_time=180., fit_degree=7)
#    spectrum_to_text_file(EG274blue.combined_cube.response_wavelength,EG274blue.combined_cube.response_curve, filename=text_file_blue )


#   FLUX CALIBRATION STAR 3

#    star3="HD49798"
#    directorio_star3 = directorio_main+"/STARS/"+star3+"/"+date+"/"+grating+"/"
#    starpos1b = directorio_star3+"26feb10024red.fits"
#    starpos2b = directorio_star3+"26feb10025red.fits"
#    starpos3b = directorio_star3+"26feb10026red.fits"
#    fits_file_blue = directorio_star3+star3+"_"+grating+"_"+date+".fits"
#    text_file_blue = directorio_star3+star3+"_"+grating+"_"+date+"_response.dat"
#
#    HD49798blue=KOALA_reduce([starpos1b,starpos2b,starpos3b],
#                            fits_file=fits_file_blue,
#                            skyflat = skyflat_blue, nskyflat=True,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                            plot= False
#                            )
#
#    HD49798blue.combined_cube.do_response_curve('FLUX_CAL/fhd49798.dat', plot=True, step=40, smooth=21, exp_time=60., fit_degree=7)
#    spectrum_to_text_file(HD49798blue.combined_cube.response_wavelength,HD49798blue.combined_cube.response_curve, filename=text_file_blue)


#    CHECK AND GET FLUX CALIBRATION FOR THE NIGHT

#    stars=[EG274blue.combined_cube,HD49798blue.combined_cube,CD32d9927blue.combined_cube]
#    plot_response(stars)
#    # NOTE: Problem with HD49798blue, check ESO spectr. stars documentation
#    # Multiply HD49798blue x 1.1744 :    checking some numbers, better using 1.16
#    # Now similar to CD-32d9927, observed at the same time.
#    print EG274blue.combined_cube.response_curve[1000]/HD49798blue.combined_cube.response_curve[1000]
#    HD49798blue.combined_cube.response_curve=HD49798blue.combined_cube.response_curve/1.16
#    plot_response(stars)
#
#    flux_calibration_20180226_blue_0p7_2k1=obtain_flux_calibration(stars)
#    spectrum_to_text_file(EG274blue.combined_cube.wavelength,flux_calibration_20180226_blue_0p7_2k1, filename=directorio_main+"/FLUX_CAL/20180226_580V_0p7_2k1_FLUX_CAL.dat" )

#    IF WE NEED TO READ FLUX CALIBRATION

#    w_star,flux_calibration_20180226_blue_0p7_2k1 = read_table(directorio+"/FLUX_CAL/20180226_580V_0p7_2k1_FLUX_CAL.dat", ["f", "f"] )
#    print flux_calibration_20180226_blue_0p7_2k1


#   OBJECT


#    galaxia="LG02"
#    directorio_galaxia = directorio_main+"/Lluis_SN/"+galaxia+"/"+grating+"/"
#    file1r = directorio_galaxia+"26feb20034red.fits"
#    file2r = directorio_galaxia+"26feb20035red.fits"
#    file3r = directorio_galaxia+"26feb20036red.fits"
#    fits_file_red = directorio_galaxia+galaxia+"_"+grating+"_"+date+".fits"

#    LG02rpos1 = KOALA_RSS(file1r, n_sky=50, skyflat=skyflat_red, nskyflat = True, plot_skyflat=False, plot=True, correct_c_d = False, brightest_line="Ha") #, do_sky=False )  #fcd32d9927

#    print LG02rpos1.CRVAL1_CDELT1_CRPIX1
#    LG02red.combined_cube.CRVAL1_CDELT1_CRPIX1=LG02rpos1.CRVAL1_CDELT1_CRPIX1

#    rss_list=[file1r,file2r,file3r]
#    LG02red=KOALA_reduce(rss_list, fits_file=fits_file_red, obj_name="LG02 - SN2005bo",
#                          skyflat = skyflat_red, nskyflat=True, n_sky=50,
#                          correct_c_d = True, brightest_line="Ha",
#                          pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                          flux_calibration=flux_calibration_20180226_red_0p7_2k1,
#                          plot= True, warnings=False  #,
##                          offsets=[0.8, 0.96,  1.39, 1.66]
#                          )
#

#    galaxia="LG02"
#    directorio_galaxia = directorio_main+"/Lluis_SN/"+galaxia+"/"+grating+"/"
#    file1b = directorio_galaxia+"26feb10034red.fits"
#    file2b = directorio_galaxia+"26feb10035red.fits"
#    file3b = directorio_galaxia+"26feb10036red.fits"
#    fits_file_blue = directorio_galaxia+galaxia+"_"+grating+"_"+date+".fits"

#    rss_list=[file1b,file2b,file3b]
#    LG02blue=KOALA_reduce(rss_list, obj_name="LG02 - SN2005bo",fits_file=fits_file_blue,
#                          skyflat = skyflat_blue, nskyflat=True, n_sky=50,
#                          correct_c_d = True, brightest_line="Hb",
#                          clean_sky_residuals = False, clean_telluric = False,
#                          pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                          flux_calibration=flux_calibration_20180226_blue_0p7_2k1,
#                          plot= True, warnings=False,
#                          offsets=offsets #[-0.87357246943603117, -1.1077408654643861, 1.797154315015268, -1.46743147005553]
#                          )


#    LG02bpos1 = KOALA_RSS(file1b, n_sky=60, skyflat=skyflat_blue, nskyflat = True, plot_skyflat=False, plot=True) #, do_sky=False )  #fcd32d9927
#    LG02bpos2 = KOALA_RSS(file2b, n_sky=60, skyflat=skyflat_blue, nskyflat = True, plot_skyflat=False, plot=True) #, do_sky=False )  #fcd32d9927


#    LG02bpos1_cube=Interpolated_cube(LG02bpos1, .7, 2.1, plot=True, flux_calibration=flux_calibration_20180226_blue_0p7_2k1)


# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# 2018 BELOW...
# --------------------------------------------
# --------------------------------------------
# --------------------------------------------


# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# REDUCING FOR LLUIS, 25 May 2018, 2:33am...
# --------------------------------------------
# --------------------------------------------
# --------------------------------------------


# SN2013bc - RED

#    directorio ="/DATA/KOALA/lluis_SN/SN2013bc"
#    skyflatr=directorio+"/ccd_2/22may20012red.fits"
#    hr4963r = directorio+"/ccd_2/22may20036red.fits"
#    eg274r = directorio+"/ccd_2/22may20045red.fits"
#
#    file1r=directorio+"/ccd_2/22may20038red.fits"
#    file2r=directorio+"/ccd_2/22may20039red.fits"
#    file3r=directorio+"/ccd_2/22may20040red.fits"
#
#    fits_file_red = directorio+"/ccd_2/SN2013bc_RED_20180525_TEST.fits"

#    skyflat_red = KOALA_RSS(skyflatr, do_sky=False, do_extinction=False, apply_throughput=False, plot=False)
#    skyflat_red.find_relative_throughput(ymax=200000)
#
#    star1r = KOALA_RSS(hr4963r, n_sky=600, skyflat=skyflat_red)
#    cubes1r=Interpolated_cube(star1r, .7, 2.1, plot=True)
#    cubes1r.do_response_curve('FLUX_CAL/fhr4963.dat', plot=True, min_wave=6100., max_wave=9250., step=15)

#    star2r = KOALA_RSS(eg274r, n_sky=600)
#    cubes2r=Interpolated_cube(star2r, .7, 2.1, plot=True)
#    cubes2r.do_response_curve('FLUX_CAL/feg274.dat', plot=True, min_wave=6100., max_wave=9250., step=15)

#    stars_cubes =[cubes1r] #[cubes1r, cubes2r]
#    plot_response(stars_cubes)
#    flux_calibration_0p7_2k1_red = obtain_flux_calibration(stars_cubes)
#
#    cubo=KOALA_reduce([file1r,file2r,file3r],
#                            fits_file=fits_file_red,
#                            skyflat=skyflat_red, #skyflat_list=["","","",""], skyflat_name = "",
#                            n_sky=200,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                            flux_calibration=flux_calibration_0p7_2k1_red,
#                            plot= True)

#    SN2013bc_red = CUBE(fits_file_red)

#  Size [pix]     =  50  x  88
#  Size [arcsec]  =  35.0  x  61.6
#  Pix size       =  0.7  arcsec
#  Files combined =  4
#  Offsets used   =   1.296 0.078  ,  1.912 0.155  ,  0.0 0.0


#    SN2013bc_red.plot_spectrum_cube(30, 57, fcal=True, lmin=6100, lmax=9300, fmin=5E-16, fmax=4E-15, fig_size=11, fig_size_y=2.2, title="Spaxel (30,57) in SN2008cd - COMBINED CUBE")#, save_file="POX4/2018/POX4_Red_Sp56-45.png")


# SN2013bc - BLUE

#    directorio ="/DATA/KOALA/lluis_SN/SN2013bc"
#    skyflatb=directorio+"/ccd_1/22may10012red.fits"
#    hr4963b = directorio+"/ccd_1/22may10036red.fits"
#    eg274b = directorio+"/ccd_1/22may10045red.fits"
#
#    file1b=directorio+"/ccd_1/22may10038red.fits"
#    file2b=directorio+"/ccd_1/22may10039red.fits"
#    file3b=directorio+"/ccd_1/22may10040red.fits"
#
#    fits_file_blue = directorio+"/ccd_1/SN2013bc_BLUE_20180525_TEST.fits"

#    skyflat_blue = KOALA_RSS(skyflatb, do_sky=False, do_extinction=False, apply_throughput=False, plot=False)
#    skyflat_blue.find_relative_throughput(ymax=200000)

#    star1b = KOALA_RSS(hr4963b, n_sky=600, skyflat=skyflat_blue)
#    cubes1b=Interpolated_cube(star1b, .7, 2.1, plot=True)
#    cubes1b.do_response_curve('FLUX_CAL/fhr4963.dat', plot=True, min_wave=3660., max_wave=5755., step=15)

#    star2b = KOALA_RSS(hr4963b, n_sky=600)
#    cubes2b=Interpolated_cube(star2b, .7, 2.1, plot=True)
#    cubes2b.do_response_curve('FLUX_CAL/feg274.dat', plot=True, min_wave=3660., max_wave=5755., step=15)

#    stars_cubes =[cubes1b] #[cubes1b, cubes2b]
#    plot_response(stars_cubes)
#    flux_calibration_0p7_2k1_blue = obtain_flux_calibration(stars_cubes)

#    cubo=KOALA_reduce([file1b,file2b,file3b,file3b],
#                            fits_file=fits_file_blue,
#                            skyflat_file=skyflatb, n_sky=200,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                            flux_calibration=flux_calibration_0p7_2k1_blue,
#                            plot= True,
#                            offsets=[1.296, 0.078  ,  1.912, 0.155 ,  0.0, 0.0])

#    SN2013bc_blue = CUBE(fits_file_blue)


# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# REDUCING FOR Stargazing ABC
# --------------------------------------------
# --------------------------------------------
# --------------------------------------------


# SMT18as - StargazingABC

#    directorio ="/DATA/KOALA/lluis_SN/SMT18as"
#    skyflat_r=directorio+"/ccd_2/23may20011red.fits"
#    cd32d9927 = directorio+"/ccd_2/23may20025red.fits"
#
#    skyflat = KOALA_RSS(skyflat_r, do_sky=False, do_extinction=False, apply_throughput=False, plot=False)
#    skyflat.intensity_corrected=skyflat.intensity
#    skyflat.find_relative_throughput(ymax=200000)

#   Read data for standard stars and obtain response curve:
#    star2r = KOALA_RSS(cd32d9927, n_sky=600) #
#    cubes2r=Interpolated_cube(star2r, .7, 2.1, plot=True)
#    cubes2r.do_response_curve('FLUX_CAL/fcd32d9927.dat', plot=True, min_wave=6100., max_wave=9250., step=10)
#    stars_cubes =[cubes2r]  # [cubes2, cubes3]
#    plot_response(stars_cubes)
#    flux_calibration_0p7_2k1_red = obtain_flux_calibration(stars_cubes)

#    file1r=directorio+"/ccd_2/23may20030red.fits"
#    fits_file_red = directorio+"/ccd_2/SMT18as_RED_20180523.fits"

#    cubo=KOALA_reduce([file1,file1,file1,file1],
#                            fits_file=fits_file, skyflat_file=skyflatr, n_sky=200,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                            flux_calibration=flux_calibration_0p7_2k1_red,
#                            plot= True)

#    rss1r = KOALA_RSS(file1r, n_sky=300) #, plot=False)
#    cube1r=Interpolated_cube(rss1r, .7, 2.1, plot=True, flux_calibration=flux_calibration_0p7_2k1_red, ADR=False)
#    save_fits_file(cube1r,fits_file_red, fcal=flux_calibration_0p7_2k1_red, ADR=False )


#    SMT18as_RED = CUBE(fits_file_red)

# blue

#
#    directorio ="/DATA/KOALA/lluis_SN/SMT18as"
#    skyflat_b=directorio+"/ccd_1/23may10011red.fits"
#    cd32d9927 = directorio+"/ccd_1/23may10025red.fits"

#    skyflat = KOALA_RSS(skyflat_b, do_sky=False, do_extinction=False, apply_throughput=False, plot=False)
#    skyflat.intensity_corrected=skyflatb.intensity
#    skyflat.find_relative_throughput(ymax=200000)

##   Read data for standard stars and obtain response curve:
#    star2b = KOALA_RSS(cd32d9927, n_sky=600) #fcd32d9927
#    cubes2b=Interpolated_cube(star2b, .7, 2.1, plot=True)
#    cubes2b.do_response_curve('FLUX_CAL/fcd32d9927.dat', plot=True, min_wave=3700., max_wave=5670., step=10)
#    stars_cubes =[cubes2b]  # [cubes2, cubes3]
#    plot_response(stars_cubes)
#    flux_calibration_0p7_2k1_blue = obtain_flux_calibration(stars_cubes)
#
#    file1b=directorio+"/ccd_1/23may10030red.fits"
#    fits_file_blue = directorio+"/ccd_1/SMT18as_BLUE_20180523.fits"
##
#    rss1b = KOALA_RSS(file1b, n_sky=300) #, plot=False)
#    cube1b=Interpolated_cube(rss1b, .7, 2.1, plot=True, flux_calibration=flux_calibration_0p7_2k1_blue, ADR=False)
#    save_fits_file(cube1b,fits_file_blue, fcal=flux_calibration_0p7_2k1_blue, ADR=False )
#
#    SMT18as_BLUE = CUBE(fits_file_blue)


#    SMT18iy_BLUE.plot_spectrum_cube(20, 40, fcal=True, lmin=3500, lmax=5550, fmin=-1E-17, fmax=5E-17, fig_size=11, fig_size_y=2.2) #, title="Spaxel (30,57) in SN2008cd - COMBINED CUBE")#, save_file="POX4/2018/POX4_Red_Sp56-45.png")


#### Combine red + blue

#    fits_file = directorio+"/SMT18iy_BLUE+RED_20180522.fits"
#
#    cube1b.number_of_combined
#
#    save_bluered_fits_file(cube1b, cube1r, fits_file=fits_file,
#                           fcalb=flux_calibration_0p7_2k1_blue, fcalr=flux_calibration_0p7_2k1_red,
#                          # trimb=[3728,5550], trimr=[6303,7426],
#                           ADR=False, objeto="SMT18iy", description="BLUE + RED")


##################### SMT18iy - StargazingABC

#    directorio ="/DATA/KOALA/lluis_SN/SMT18iy"
#    skyflatr=directorio+"/ccd_2/22may20011red.fits"
#    eg274r = directorio+"/ccd_2/22may20045red.fits"
#
#    skyflat = KOALA_RSS(skyflatr, do_sky=False, do_extinction=False, apply_throughput=False, plot=False)
#    skyflat.intensity_corrected=skyflat.intensity
#    skyflat.find_relative_throughput(ymax=200000)

#   Read data for standard stars and obtain response curve:
#    star2r = KOALA_RSS(eg274r, n_sky=600) #fcd32d9927
#    cubes2r=Interpolated_cube(star2r, .7, 2.1, plot=True)
#    cubes2r.do_response_curve('FLUX_CAL/feg274.dat', plot=True, min_wave=6100., max_wave=9250., step=10)
#    stars_cubes =[cubes2r]  # [cubes2, cubes3]
#    plot_response(stars_cubes)
#    flux_calibration_0p7_2k1_red = obtain_flux_calibration(stars_cubes)

#    file1r=directorio+"/ccd_2/22may20047red.fits"
#    fits_file_red = directorio+"/ccd_2/SMT18iy_RED_20180522.fits"

#    cubo=KOALA_reduce([file1,file1,file1,file1],
#                            fits_file=fits_file, skyflat_file=skyflatr, n_sky=200,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1,
#                            flux_calibration=flux_calibration_0p7_2k1_red,
#                            plot= True)

#    rss1r = KOALA_RSS(file1r, n_sky=300) #, plot=False)
#    cube1r=Interpolated_cube(rss1r, .7, 2.1, plot=True, flux_calibration=flux_calibration_0p7_2k1_red, ADR=False)
#    save_fits_file(cube1r,fits_file_red, fcal=flux_calibration_0p7_2k1_red, ADR=False )


#    SMT18iy_RED = CUBE(fits_file_red)

# blue

#
#    directorio ="/DATA/KOALA/lluis_SN/SMT18iy"
#    skyflat_b=directorio+"/ccd_1/22may10011red.fits"
#    eg274b = directorio+"/ccd_1/22may10045red.fits"
##
#    skyflat = KOALA_RSS(skyflat_b, do_sky=False, do_extinction=False, apply_throughput=False, plot=False)
#    skyflat.intensity_corrected=skyflatb.intensity
#    skyflat.find_relative_throughput(ymax=200000)

##   Read data for standard stars and obtain response curve:
#    star2b = KOALA_RSS(eg274b, n_sky=600) #fcd32d9927
#    cubes2b=Interpolated_cube(star2b, .7, 2.1, plot=True)
#    cubes2b.do_response_curve('FLUX_CAL/feg274.dat', plot=True, min_wave=3700., max_wave=5670., step=10)
#    stars_cubes =[cubes2]  # [cubes2, cubes3]
#    plot_response(stars_cubes)
#    flux_calibration_0p7_2k1_blue = obtain_flux_calibration(stars_cubes)
#
#    file1b=directorio+"/ccd_1/22may10047red.fits"
#    fits_file_blue = directorio+"/ccd_1/SMT18iy_BLUE_20180522.fits"
#
#    rss1b = KOALA_RSS(file1b, n_sky=300) #, plot=False)
#    cube1b=Interpolated_cube(rss1b, .7, 2.1, plot=True, flux_calibration=flux_calibration_0p7_2k1_blue, ADR=False)
#    save_fits_file(cube1,fits_file, fcal=flux_calibration_0p7_2k1_blue, ADR=False )

#    SMT18iy_BLUE = CUBE(fits_file_red)


#    SMT18iy_BLUE.plot_spectrum_cube(20, 40, fcal=True, lmin=3500, lmax=5550, fmin=-1E-17, fmax=5E-17, fig_size=11, fig_size_y=2.2) #, title="Spaxel (30,57) in SN2008cd - COMBINED CUBE")#, save_file="POX4/2018/POX4_Red_Sp56-45.png")


#### Combine red + blue

#    fits_file = directorio+"/SMT18iy_BLUE+RED_20180522.fits"
#
#    cube1b.number_of_combined
#
#    save_bluered_fits_file(cube1b, cube1r, fits_file=fits_file,
#                           fcalb=flux_calibration_0p7_2k1_blue, fcalr=flux_calibration_0p7_2k1_red,
#                          # trimb=[3728,5550], trimr=[6303,7426],
#                           ADR=False, objeto="SMT18iy", description="BLUE + RED")


# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# REDUCING FOR LLUIS, 22 May 2018, 4:21am...
# --------------------------------------------
# --------------------------------------------
# --------------------------------------------


# SN2008cd - RED

#    directorio ="/DATA/KOALA/lluis_SN/SN2008cd"
#    skyflatr=directorio+"/ccd_2/21may20013red.fits"
#    CD23d9927r = directorio+"/ccd_2/21may20026red.fits"

#    skyflat = KOALA_RSS(skyflatr, do_sky=False, do_extinction=False, apply_throughput=False, plot=False)
#    skyflat.intensity_corrected=skyflat.intensity
#    skyflat.find_relative_throughput(ymax=200000)

#   Read data for standard stars and obtain response curve: ****** NOTE: THIS FILE HAS TO BE REDUCED AGAIN ****
#    star2 = KOALA_RSS(CD23d9927r, n_sky=600) #fcd32d9927
#    cubes2=Interpolated_cube(star2, .7, 2.1, plot=True)
#    cubes2.do_response_curve('FLUX_CAL/fcd32d9927.dat', plot=True, min_wave=6100., max_wave=9250., step=15)
#    stars_cubes =[cubes2]  # [cubes2, cubes3]
#    plot_response(stars_cubes)
#    flux_calibration_0p7_2k1_red = obtain_flux_calibration(stars_cubes)

#    file1=directorio+"/ccd_2/21may20023red.fits"
#    file2=directorio+"/ccd_2/21may20023red.fits"
#    file3=directorio+"/ccd_2/21may20023red.fits"

#    cubo=KOALA_reduce([file1,file2,file3,file3],
#                            fits_file=directorio+"/ccd_2/SN2008cd_RED_20180521.fits", skyflat_file=skyflatr, n_sky=200,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1, flux_calibration=flux_calibration_0p7_2k1_red,
#                            plot= True)

#    SN2008cd_red = CUBE(directorio+"/ccd_2/SN2008cd_RED_20180521.fits")
#  Size [pix]     =  88  x  80
#  Offsets used   =   -0.739 -0.924  ,  1.708 -0.764  ,  0.0 0.0

#    SN2008cd_red.plot_spectrum_cube(30, 57, fcal=True, lmin=6100, lmax=9300, fmin=5E-16, fmax=4E-15, fig_size=11, fig_size_y=2.2, title="Spaxel (30,57) in SN2008cd - COMBINED CUBE")#, save_file="POX4/2018/POX4_Red_Sp56-45.png")


#    SN2008cd_red.plot_spectrum_cube(30, 57, fcal=True, lmin=6500, lmax=6610, fmin=5E-16, fmax=4E-15, fig_size=11, fig_size_y=2.2, title="Spaxel (30,57) in SN2008cd - COMBINED CUBE")#, save_file="POX4/2018/POX4_Red_Sp56-45.png")
#    ha_map = SN2008cd_red.create_map(6538,6558, "ha_map")
#    mask=np.zeros((88,80))
#    for x in range(88):
#        for y in range(80):
#            if ha_map[x][y] > 200. : mask[x][y] = 1.    #180
#    ha_map_mask = ha_map * mask
#    #
#    titleha="SN2008cd host - H-alpha - Integrating [6638 - 6558] $\AA$"
#    SN2008cd_red.plot_map(ha_map, contours=True, title = titleha, log=False, vmin=100., vmax=40000, cmap="fuego") #, save_file="POX4/2018/POX4_ha_map.png" )#"gnuplot")
#
##    w=SN2008cd_red.wavelength
##    velha=np.zeros((56,86))
##    fluxha=np.zeros((56,86))
##    ewha = np.zeros((56,86))
##    for x in range(56):
##        for y in range(86):
##            if ha_map[x][y] > 9000. :
##                    spectrum=SN2008cd_red.data[:,x,y]/(SN2008cd_red.flux_calibration/1E-16)
##                    resultado = fluxes(w, spectrum, 6548, lmin=6450, lmax=6650, lowlow= 60, lowhigh=40, highlow=40, highhigh = 60, plot=False, verbose=False)
##                    velha[x][y] = resultado[1]
##                    fluxha[x][y] = resultado[3]   #7 para flujo integrado
##                    ewha[x][y] = resultado[9]
##            else:
##                velha[x][y] = "nan"
##                fluxha[x][y] = "nan"
##                ewha[x][y] = "nan"
##
##    meanvalue= np.nanmedian(velha)
##    veloha = C*(velha - meanvalue) / meanvalue
###
###
#
#    titleha="SN2008cd host - Continuum-substracted H-alpha emission"
#    SN2008cd_red.plot_map(fluxha, contours=True, title = titleha, log=True, vmin=1E-17, vmax=2E-14, cmap="fuego", fcal=True, barlabel="Flux [erg s$^{-1}$ cm$^{-2}$]" ) #, save_file="POX4/2018/POX4_haflux_map.eps" ) #, save_file="POX4/2018/POX4_ha_map.png" )#"gnuplot")
#
#
#    titlevelha="SN2008cd host - H-alpha velocity field"
#    SN2008cd_red.plot_map(veloha, contours=False, title = titlevelha, log=False, cmap="seismic", vmin=-100., vmax=100., norm=colors.Normalize(), barlabel="Relative velocity [km/s]") #, save_file="POX4/2018/POX4_ha_vel_map_TRES.eps")#, fig_size=30 )#"gnuplot")
#


# --------------------------------------------
# --------------------------------------------
# --------------------------------------------

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------


# POX4 - RED - 13 Mar 2018


#    skyflat = KOALA_RSS('POX4/2018/16jan20086red.fits', do_sky=False, do_extinction=False, apply_throughput=False, plot=False)
#    skyflat.intensity_corrected=skyflat.intensity
#    skyflat.find_relative_throughput()

#   Read data for standard stars and obtain response curve: ****** NOTE: THIS FILE HAS TO BE REDUCED AGAIN ****
#    star2 = KOALA_RSS('POX4/2018/16jan20064red.fits', n_sky=600)
#    cubes2=Interpolated_cube(star2, .7, 2.1, plot=True)
#    cubes2.do_response_curve('FLUX_CAL/ffeige56.dat', plot=True, min_wave=6300., max_wave=7415., step=15)
#    stars_cubes =[cubes2]  # [cubes2, cubes3]
#    plot_response(stars_cubes)
#    flux_calibration_0p7_2k1_red = obtain_flux_calibration(stars_cubes)

#    cubo=KOALA_reduce(['POX4/2018/16jan20058red.fits','POX4/2018/16jan20059red.fits','POX4/2018/16jan20060red.fits','POX4/2018/16jan20060red_bis.fits'],
#                            fits_file="POX4/2018/POX4_RED_20180313.fits", skyflat_file='POX4/2018/16jan20086red.fits', n_sky=200,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1, flux_calibration=flux_calibration_0p7_2k1_red,
#                            plot= True)

#    POX4_red = CUBE("POX4/2018/POX4_RED_20180313.fits")
#  Size [pix]     =  88  x  80
#  Offsets used   =   -0.739 -0.924  ,  1.708 -0.764  ,  0.0 0.0

#    POX4_red.plot_spectrum_cube(56, 45, fcal=True, lmin=6300, lmax=7430, fmin=5E-17, fmax=7E-16, fig_size=18, fig_size_y=2.2, title="Spaxel (56,45) in POX 4 - COMBINED CUBE", save_file="POX4/2018/POX4_Red_Sp56-45.png")


# POX4 - BLUE - 12 Mar 2018

#    skyflat = KOALA_RSS('POX4/2018/16jan10086red.fits', do_sky=False, do_extinction=False, apply_throughput=False, plot=False)
#    skyflat.intensity_corrected=skyflat.intensity
#    skyflat.find_relative_throughput()
#  Read data for standard stars and obtain response curve:
#    star2 = KOALA_RSS('POX4/2018/16jan10064red.fits', n_sky=600)
#    cubes2=Interpolated_cube(star2, .7, 2.1, plot=True)
#    cubes2.do_response_curve('FLUX_CAL/ffeige56.dat', plot=True, min_wave=3660., max_wave=5755., step=20)
#    stars_cubes =[cubes2]  # [cubes2, cubes3]
#    plot_response(stars_cubes)
#    flux_calibration_0p7_2k1_blue = obtain_flux_calibration(stars_cubes)

#    cubo=KOALA_reduce(['POX4/2018/16jan10058red.fits','POX4/2018/16jan10059red.fits','POX4/2018/16jan10060red.fits','POX4/2018/16jan10060red_bis.fits'],
#                            fits_file="POX4/2018/POX4_BLUE_20180313.fits", skyflat_file='POX4/2018/16jan10086red.fits', n_sky=200,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1, flux_calibration=flux_calibration_0p7_2k1_blue,
#                            plot= True, offsets=[-0.739, -0.924  ,  1.708, -0.764  ,  0.0, 0.0])

#    POX4_blue = CUBE("POX4/2018/POX4_BLUE_20180313.fits")
# Now it is good!
#  Size [pix]     =  88  x  80
#  Offsets used   =   -0.739 -0.924  ,  1.708 -0.764  ,  0.0 0.0


#    POX4_blue.plot_spectrum_cube(56, 45, fcal=True, lmin=3740, lmax=5150, fmin=2E-16, fmax=7E-16, fig_size=18, fig_size_y=2.2, title="Spaxel (56,45) in POX 4 - COMBINED CUBE", save_file="POX4/2018/POX4_Blue_Sp56-45.png")

#    POX4_blue.plot_spectrum_cube(56, 45, fcal=True, lmin=3740, lmax=5150, fmin=2E-16, fmax=7E-16, fig_size=18, fig_size_y=2.2, title="Spaxel (56,45) in POX 4 - COMBINED CUBE") #, save_file="POX4/2018/POX4_Blue_Sp56-45.png")

#    POX4_blue.plot_spectrum_cube([56,57,58,59,60, 56,57,58,59,60], [45,45,45,45,45, 46,46,46,46,46], fcal=True, lmin=3740, lmax=5150, fmin=1E-15, fmax=5E-15, fig_size=18, fig_size_y=2.2, title="Center of POX 4 - COMBINED CUBE", save_file="POX4/2018/POX4_Blue_range.png")

#    POX4_blue.plot_spectrum_cube(56, 45, fcal=True, lmin=3740, lmax=5150, fmin=2E-16, fmax=7E-16, fig_size=18, fig_size_y=2.2, title="Spaxel (56,45) in POX 4 - COMBINED CUBE") #, save_file="POX4/2018/POX4_Blue_Sp56-45.png")

#    hb_map_POX4 = POX4.create_map(4900,4950, "hb_map")
##    title="POX 4 - H-beta - Integrating [4900 - 4950] $\AA$"
##    POX4_blue.plot_map(hb_map, contours=True, title = title, log=True,vmin=200., vmax=15000, cmap="fuego" )#"gnuplot") #, save_file="Erik/NGC695/NGC695_ha_map.png" ) #, norm=colors.LogNorm(), vmin=10)
#
##    Create a mask
#
#    mask_POX4=np.zeros((88,80))
#    for x in range(88):
#        for y in range(80):
#            if hb_map_POX4[x][y] > 200. : mask_POX4[x][y] = 1.    #180
##
##    hb_map_mask = hb_map * mask
##    POX4_blue.plot_map(hb_map_mask, contours=True, title = title, log=True,vmin=200., vmax=15000, cmap="fuego", save_file="POX4/2018/hb_map.png") #"gnuplot") #, save_file="Erik/NGC695/NGC695_ha_map.png" ) #, norm=colors.LogNorm(), vmin=10)
##
##
#    o3_map_POX4 = POX4.create_map(5040,5100, "o3_map")
###    titleo3="POX 4 -[O III] - Integrating [5040 - 5100] $\AA$"
###    POX4_blue.plot_map(o3_map, contours=True, title = titleo3, log=False,vmin=800., vmax=35000, cmap="fuego" )#"gnuplot") #, save_file="Erik/NGC695/NGC695_ha_map.png" ) #, norm=colors.LogNorm(), vmin=10)
##
#    o3hb_map_POX4 = o3_map_POX4/hb_map_POX4 * mask_POX4
#    titleo3hb="POX 4 - [O III] / H-beta map"
#    POX4.plot_map(o3hb_map_POX4, contours=True, title = titleo3hb, log=False,vmin=1.1, vmax=5, cmap="gnuplot", norm=colors.LogNorm(), save_file="POX4/2018/o3hb_map.eps", barlabel="Flux ratio" )#"gnuplot") #, save_file="Erik/NGC695/NGC695_ha_map.png" ) #, norm=colors.LogNorm(), vmin=10)


######## NOTE: COMBINE RED AND BLUE:


#    save_bluered_fits_file(POX4_blue, POX4_red, fits_file="POX4/2018/POX4_BLUE+RED_20180313.fits",
#                           fcalb=flux_calibration_0p7_2k1_blue, fcalr=flux_calibration_0p7_2k1_red,
#                           trimb=[3728,5550], trimr=[6303,7426],
#                           ADR=False, objeto="POX 4", description="BLUE + RED")
#
#    POX4= CUBE("POX4/2018/POX4_BLUE+RED_20180313.fits")
#
#
#    POX4.plot_spectrum_cube(56, 45, fcal=True, lmin=3700, lmax=7450, fmin=-1E-16, fmax=7E-15, fig_size=18, fig_size_y=2.2, title="Spaxel (56,45) in POX 4 - COMBINED CUBE", save_file="POX4/2018/POX4_Blue+Red_Sp56-45_scale.png")
#    POX4.plot_spectrum_cube(56, 45, fcal=True, lmin=3720, lmax=5280, fmin=1E-16, fmax=1E-15, fig_size=20, fig_size_y=2.2, z=0.0118, title="Spaxel (56,45) in POX 4 - COMBINED CUBE", save_file="POX4/2018/POX4_Blue+Red_Sp56-45_blue_lines.eps")
#    POX4.plot_spectrum_cube(56, 45, fcal=True, lmin=6300, lmax=7430, fmin=5E-17, fmax=7E-16, fig_size=20, fig_size_y=2.2, z=0.0116, title="Spaxel (56,45) in POX 4 - COMBINED CUBE", save_file="POX4/2018/POX4_Blue+Red_Sp56-45_red_lines.eps")

#    POX4.plot_spectrum_cube(56, 45, fcal=True, lmin=3700, lmax=4050, fmin=7E-17, fmax=7E-16, fig_size=12, fig_size_y=2.2, title="Spaxel (56,45) in POX 4 - COMBINED CUBE")#, save_file="POX4/2018/POX4_Blue_Sp56-45.png")


#    POX4.plot_spectrum_cube(56, 45, fcal=True, lmin=6655, lmax=6665, fmin=7E-17, fmax=7E-16, fig_size=12, fig_size_y=2.2, title="Spaxel (56,45) in POX 4 - COMBINED CUBE")#, save_file="POX4/2018/POX4_Blue_Sp56-45.png")
#    ha_map = POX4.create_map(6655,6665, "ha_map")
#    mask=np.zeros((88,80))
#    for x in range(88):
#        for y in range(80):
#            if ha_map[x][y] > 200. : mask[x][y] = 1.    #180
#    ha_map_mask = ha_map * mask
#    #
#    titleha="POX 4 - H-alpha - Integrating [6655 - 6653] $\AA$"
#    POX4.plot_map(ha_map_mask, contours=True, title = titleha, log=False, vmin=100., vmax=5000, cmap="fuego") #, save_file="POX4/2018/POX4_ha_map.png" )#"gnuplot")

#    POX4.plot_spectrum_cube(56, 45, fcal=True, lmin=4910, lmax=4930, fmin=0, fmax=7E-15, fig_size=12, fig_size_y=2.2, title="Spaxel (56,45) in POX 4 - COMBINED CUBE")#, save_file="POX4/2018/POX4_Blue_Sp56-45.png")
#    hb_map = POX4.create_map(4913,4924, "hb_map")
#    hb_map_mask = hb_map * mask
#    titlehb="POX 4 - H-beta - Integrating [4913 - 4924] $\AA$"
#
##    POX4.plot_map(hb_map_mask, contours=True, title = titlehb, log=False, vmin=0., vmax=4500, cmap="fuego") #, save_file="POX4/2018/POX4_ha_map.png" )#"gnuplot")
#
#    hahb = ha_map_mask/ hb_map_mask *3.5   # Substract the continuum or it will never work!
#    titlehahb="POX 4 - H-alpha / H-beta map"
#    POX4.plot_map(hahb, contours=False, title = titlehahb, log=False, vmin=2.6, vmax=5., cmap="fuego") #, save_file="POX4/2018/POX4_ha_map.png" )#"gnuplot")


# --------------------------------------------
# Velocity map

#    POX4.plot_spectrum_cube(56, 45, fcal=True, lmin=6500, lmax=6750, fmin=7E-17, fmax=7E-16, fig_size=12, fig_size_y=2.2, title="Spaxel (56,45) in POX 4 - COMBINED CUBE")#, save_file="POX4/2018/POX4_Blue_Sp56-45.png")
#
#    w=POX4.wavelength
#    spectrum=POX4.data[:,56,45]/(POX4.flux_calibration/1E-16)
#
#    resultado = fluxes(w, spectrum, 6640, lmin=6550, lmax=6750, lowlow= 60, lowhigh=40, highlow=40, highhigh = 60, plot=True, verbose=True)


####### NOTA: CASCA EL FLUJO INTEGRADO EN EL CENTRO DE POX4, uso Gaussiana

#    POX4.plot_spectrum_cube(56, 45, fcal=True, lmin=6655, lmax=6665, fmin=7E-17, fmax=7E-16, fig_size=12, fig_size_y=2.2, title="Spaxel (56,45) in POX 4 - COMBINED CUBE")#, save_file="POX4/2018/POX4_Blue_Sp56-45.png")
#    ha_map = POX4.create_map(6657,6663, "ha_map")
#    titleha="POX 4 - H-alpha - Integrating [6657 - 6663] $\AA$"
#    POX4.plot_map(ha_map, contours=True, title = titleha, log=False, vmin=120., vmax=5000, cmap="fuego") #, save_file="POX4/2018/POX4_ha_map.png" )#"gnuplot")

#    velha_POX4=np.zeros((88,80))
#    fluxha_POX4 =np.zeros((88,80))
#    ewha_POX4 = np.zeros((88,80))
#    for x in range(88):
#        for y in range(80):
#            if ha_map[x][y] > 150. :
#                    spectrum=POX4.data[:,x,y]/(POX4.flux_calibration/1E-16)
#                    resultado = fluxes(w, spectrum, 6640, lmin=6550, lmax=6750, lowlow= 60, lowhigh=40, highlow=40, highhigh = 60, plot=False, verbose=False)
#                    velha_POX4[x][y] = resultado[1]
#                    fluxha_POX4[x][y] = resultado[3]   #7 para flujo integrado
#                    ewha_POX4[x][y] = resultado[9]
#            else:
#                velha_POX4[x][y] = "nan"
#                fluxha_POX4[x][y] = "nan"
#                ewha_POX4[x][y] = "nan"
#
#    meanvalue= np.nanmedian(velha_POX4)
#    veloha_POX4 = C*(velha_POX4 - meanvalue) / meanvalue
#
#
#    titlevelha="POX 4 - H-alpha velocity field"
#    POX4.plot_map(veloha_POX4, contours=False, title = titlevelha, log=False, cmap="seismic", vmin=-80., vmax=80., norm=colors.Normalize(), barlabel="Relative velocity [km/s]", save_file="POX4/2018/POX4_ha_vel_map_TRES.eps")#, fig_size=30 )#"gnuplot")


#    titleha_POX4="POX 4 - Continuum-substracted H-alpha emission"
#    POX4.plot_map(fluxha_POX4, contours=True, title = titleha_POX4, log=True, vmin=1E-17, vmax=2E-14, cmap="fuego", fcal=True, barlabel="Flux [erg s$^{-1}$ cm$^{-2}$]", save_file="POX4/2018/POX4_haflux_map.eps" ) #, save_file="POX4/2018/POX4_ha_map.png" )#"gnuplot")


#### Save file (for calibration stars)

#    spectrum_to_text_file(POX4_blue.wavelength,flux_calibration_0p7_2k1_blue, filename="FLUX_CAL/20160116_flux_calibration_0p7_2k1_blue.dat")
#    spectrum_to_text_file(POX4_red.wavelength,flux_calibration_0p7_2k1_red, filename="FLUX_CAL/20160116_flux_calibration_0p7_2k1_red.dat")

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------


# --------------------------------------------
# --------------------------------------------
# --------------------------------------------

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------


# NGC 1522 - RED - Reprise 14 Mar 2018


# Careful: always use the same order in red and blue for the offsets!!!

#    cubo=KOALA_reduce(['NGC1522/16jan20049red.fits','NGC1522/16jan20050red.fits','NGC1522/16jan20051red.fits','NGC1522/16jan20047red.fits'],
#                            fits_file="NGC1522/NGC1522_RED_20180313.fits", skyflat_file='POX4/2018/16jan20086red.fits', n_sky=200,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1, flux_calibration=flux_calibration_0p7_2k1_red,
#                            plot= True)

#    NGC1522_red = CUBE("NGC1522/NGC1522_RED_20180313.fits")

#  Size [pix]     =  88  x  88
#  Offsets used   =   1.457 1.595  ,  -1.751 -1.134  ,  -0.005 0.183

#    NGC1522_red.plot_spectrum_cube(42, 44, fcal=True, lmin=6300, lmax=7430, fmin=5E-17, fmax=7E-16, fig_size=12, fig_size_y=2.2, title="Spaxel (42,44) in NGC 1522 - COMBINED CUBE")# , save_file="POX4/2018/POX4_Red_Sp56-45.png")


#    cubo=KOALA_reduce(['NGC1522/16jan10049red.fits','NGC1522/16jan10050red.fits','NGC1522/16jan10051red.fits','NGC1522/16jan10051red_bis.fits'],
#                            fits_file="NGC1522/NGC1522_BLUE_20180313.fits", skyflat_file='POX4/2018/16jan10086red.fits', n_sky=200,
#                            pixel_size_arcsec=.7, kernel_size_arcsec=2.1, flux_calibration=flux_calibration_0p7_2k1_red,
#                            plot= True, offsets=[1.457, 1.595  ,  -1.751, -1.134  ,  0.0, 0.0])

#    NGC1522_blue = CUBE("NGC1522/NGC1522_BLUE_20180313.fits")


#    save_bluered_fits_file(NGC1522_blue, NGC1522_red, fits_file="NGC1522/NGC1522_BLUE+RED_20180313.fits",
#                           fcalb=flux_calibration_0p7_2k1_blue, fcalr=flux_calibration_0p7_2k1_red,
#                           trimb=[3710,5550], trimr=[6303,7426],
#                           ADR=False, objeto="NGC 1512", description="BLUE + RED")

#    NGC1522= CUBE("NGC1522/NGC1522_BLUE+RED_20180313.fits")

#    NGC1522.plot_spectrum_cube(42, 44, fcal=True, lmin=3650, lmax=7450, fmin=-1E-16, fmax=1E-15, fig_size=11, fig_size_y=2.2, title="Spaxel (42,44) in NGC 1522 - COMBINED CUBE" ) #, save_file="POX4/2018/POX4_Blue+Red_Sp56-45_scale.png")

#    NGC1522.plot_spectrum_cube(42, 44, fcal=True, lmin=3720, lmax=5280, fmin=2.2E-16, fmax=5.8E-16, fig_size=20, fig_size_y=2.2, z=0.002997, title="Spaxel (42,44) in NGC 1522 - COMBINED CUBE", save_file="NGC1522/NGC1522_Blue+Red_Sp42-44_blue_lines.eps")
#    NGC1522.plot_spectrum_cube(42, 44, fcal=True, lmin=6300, lmax=7430, fmin=1.3E-16, fmax=5E-16, fig_size=20, fig_size_y=5, z=0.002997, title="Spaxel (42,44) in NGC 1522 - COMBINED CUBE", save_file="NGC1522/NGC1522_Blue+Red_Sp42-44_red_lines_.eps")


#    NGC1522_red.plot_spectrum_cube(42, 44, fcal=True, lmin=6560, lmax=6620, fmin=0, fmax=7E-16, fig_size=12, fig_size_y=2.2, title="Spaxel (42,44) in NGC 1522 - COMBINED CUBE")#, save_file="POX4/2018/POX4_Blue_Sp56-45.png")
#    NGC1522_ha_map = NGC1522_red.create_map(6578,6588, "ha_map")
#    mask=np.zeros((88,88))
#    for x in range(88):
#        for y in range(88):
#            if ha_map[x][y] > 200. : mask[x][y] = 1.    #180
#    ha_map_mask = ha_map * mask
#    #
#    titleha_ngc1522="NGC1522 - H-alpha - Integrating [6578 - 6588] $\AA$"
#    NGC1522_red.plot_map(NGC1522_ha_map, contours=True, title = titleha_ngc1522, log=False, vmin=900., vmax=40000, cmap="fuego") #, save_file="POX4/2018/POX4_ha_map.png" )#"gnuplot")


# --------------------------------------------
# Velocity map and continuum substracted Ha flux

#    NGC1522_red.plot_spectrum_cube(42, 44, fcal=True, lmin=6560, lmax=6610, fmin=2E-16, fmax=5E-16, fig_size=12, fig_size_y=2.2, title="Spaxel (42,44) in NGC 1522 - COMBINED CUBE")#, save_file="POX4/2018/POX4_Blue_Sp56-45.png")
#    w=NGC1522_red.wavelength
#    spectrum=NGC1522_red.data[:,42,44]/(NGC1522_red.flux_calibration/1E-16)
#    resultado = fluxes(w, spectrum, 6583, lmin=6560, lmax=6610, lowlow= 11, lowhigh=7, highlow=8, highhigh = 14, fmax=5E-16, plot=True, verbose=True)

# resultado: 7 is flux, 9 is ew
#    velha_ngc1522=np.zeros((88,88))
#    fluxha_ngc1522 =np.zeros((88,88))
#    ewha_ngc1522 = np.zeros((88,88))
#    for x in range(88):
#        for y in range(88):
#            if NGC1522_ha_map[x][y] > 400. :
#                    spectrum=NGC1522_red.data[:,x,y]/(NGC1522_red.flux_calibration/1E-16)
#                    resultado = fluxes(w, spectrum, 6583, lmin=6560, lmax=6610, lowlow= 11, lowhigh=7, highlow=8, highhigh = 14, plot=False, verbose=False)
#                    velha_ngc1522[x][y] = resultado[1]
#                    fluxha_ngc1522[x][y] = resultado[7]
#                    ewha_ngc1522[x][y] = resultado[9]
#            else:
#                velha_ngc1522[x][y] = "nan"
#                fluxha_ngc1522[x][y] = "nan"
#                ewha_ngc1522[x][y] = "nan"
#
#    meanvalue= np.nanmedian(velha_ngc1522)
##    print meanvalue
#    veloha_ngc1522 = C*(velha_ngc1522 - meanvalue) / meanvalue


#    titlevelha="NGC 1522 - H-alpha velocity field"
#    NGC1522_red.plot_map(veloha_ngc1522, contours=False, title = titlevelha, log=False, cmap="seismic", vmin=-60., vmax=60., norm=colors.Normalize(), barlabel="Relative velocity [km/s]", save_file="NGC1522/NGC1522_ha_vel_map.eps" )# )#"gnuplot")

#    NGC1522_red.plot_map(fluxha_ngc1522, contours=True, title = titleha_ngc1522, log=False, vmin=1E-16, vmax=2E-14, cmap="fuego", fcal=True, barlabel="Flux [erg s$^{-1}$ cm$^{-2}$]", save_file="NGC1522/NGC1522_haflux_map.eps" ) #, save_file="POX4/2018/POX4_ha_map.png" )#"gnuplot")


# Hb y [O III] maps

#    NGC1522.plot_spectrum_cube(42, 44, fcal=True, lmin=4870, lmax=4885, fmin=0, fmax=1E-15, fig_size=11, fig_size_y=2.2, title="Spaxel (42,44) in NGC 1522 - COMBINED CUBE" ) #, save_file="POX4/2018/POX4_Blue+Red_Sp56-45_scale.png")


#    hb_map_NGC1522 = NGC1522.create_map(4873,4879, "hb_map")
#    title="NGC 1522 - H-beta - Integrating [4873 - 4879] $\AA$"
#    NGC1522.plot_map(hb_map_NGC1522, contours=True, title = title, log=True,vmin=10., vmax=2000, cmap="fuego" )#"gnuplot") #, save_file="Erik/NGC695/NGC695_ha_map.png" ) #, norm=colors.LogNorm(), vmin=10)

#    Create a mask

#    mask_NGC1522=np.zeros((88,88))
#    for x in range(88):
#        for y in range(82):
#            if hb_map_NGC1522[x][y] >80. :
#                mask_NGC1522[x][y] = 1.    #180
#            else:
#                mask_NGC1522[x][y] = "nan"
#
#    hb_map_mask_NGC1522 = hb_map_NGC1522 * mask_NGC1522
#    NGC1522.plot_map(hb_map_mask_NGC1522, contours=True, title = title, log=True,vmin=50., vmax=8000, cmap="fuego" )#"gnuplot") #, save_file="Erik/NGC695/NGC695_ha_map.png" ) #, norm=colors.LogNorm(), vmin=10)

#    NGC1522.plot_spectrum_cube(42, 44, fcal=True, lmin=5015, lmax=5030, fmin=0, fmax=1E-15, fig_size=11, fig_size_y=2.2, title="Spaxel (42,44) in NGC 1522 - COMBINED CUBE" ) #, save_file="POX4/2018/POX4_Blue+Red_Sp56-45_scale.png")

#    o3_map_NGC1522 = NGC1522.create_map(5019,5026, "o3_map")
#    titleo3_ngc1522="NGC 1522 -[O III] - Integrating [5019 - 5026] $\AA$"
#    o3_map_mask_NGC1522 = o3_map_NGC1522 * mask_NGC1522
#    NGC1522.plot_map(o3_map_mask_NGC1522, contours=True, title = titleo3_ngc1522, log=False,vmin=100., vmax=35000, cmap="fuego" )#"gnuplot") #, save_file="Erik/NGC695/NGC695_ha_map.png" ) #, norm=colors.LogNorm(), vmin=10)

#    o3hb_map_NGC1522 = o3_map_NGC1522 / hb_map_mask_NGC1522
#    titleo3hb_NGC1522="NGC 1522 - [O III] / H-beta map"
#    NGC1522.plot_map(o3hb_map_NGC1522, contours=True, title = titleo3hb_NGC1522, log=False,vmin=1, vmax=3.5, cmap="gnuplot", norm=colors.LogNorm(), save_file="NGC1522/o3hb_map_NGC1522.eps", barlabel="Flux ratio" )#"gnuplot") #, save_file="Erik/NGC695/NGC695_ha_map.png" ) #, norm=colors.LogNorm(), vmin=10)


# NGC 1522 - BLUE - Reprise 7 Mar 2018


### First we read a skyflat

#    skyflat = KOALA_RSS('Ben/16jan10086red.fits', do_sky=False, do_extinction=False, apply_throughput=False, plot=False)
#    skyflat.intensity_corrected=skyflat.intensity
####    skyflat.plot_spectra(ymin=-20000,ymax=50000.)
#    skyflat.find_relative_throughput()
#    print "Checking skyflat..."
#    skyflat_corrected = KOALA_RSS('Ben/16jan10086red.fits', do_sky=False, do_extinction=False, norm=colors.Normalize())

#
## Read data for standard stars and obtain response curve
#
#    star2 = KOALA_RSS('Ben/16jan10064red.fits', n_sky=600)
#    cubes2=Interpolated_cube(star2, .8, 2.4, plot=True)
#    cubes2.do_response_curve('FLUX_CAL/ffeige56.dat', plot=True, min_wave=3650., max_wave=5600., step=25)
#
#    star3 = KOALA_RSS('Erik/06sep20026red.fits', n_sky=600, norm=colors.Normalize())  ### NARROW
#    cubes3=Interpolated_cube(star3, .5, 1.5, plot=True)
#    cubes3.do_response_curve('FLUX_CAL/fltt7987.dat', plot=True,  min_wave=6250., max_wave=7425., step=25)
#
### Flux correction using response_curve obtained for standard stars
#
##   If galaxy field is WIDE:
#    cubes3.response_curve = 1.25**2 /0.7**2 * cubes3.response_curve    #   cubes3 was using the NARROW field
##   If galaxy field is NARROW:
##    cubes2.response_curve =  0.7**2 / 1.25**2 * cubes2.response_curve  #   cubes3 was using the NARROW field
##
#    stars_cubes =[cubes2]  # [cubes2, cubes3]
#    plot_response(stars_cubes)
#    flux_calibration_0p8_2k4_blue = obtain_flux_calibration(stars_cubes)


# HACERLO CON LOS OFFSETS DEL ROJO

#    cubo=KOALA_reduce(['Ben/16jan10049red_.fits','Ben/16jan10049red.fits','Ben/16jan10050red.fits','Ben/16jan10051red.fits'],
#                            fits_file="Ben/NGC1522_BLUE_20180307.fits", skyflat_file='Ben/16jan10085red.fits', n_sky=200,
#                            pixel_size_arcsec=.8, kernel_size_arcsec=2.4, flux_calibration=flux_calibration_0p8_2k4_blue,
#                            plot= True)
#

#    NGC1522_blue = CUBE("Ben/NGC1522_BLUE_20180307.fits")
#    NGC1522_blue.plot_spectrum_cube(42, 42, fcal=True, lmin=3670, lmax=4200, fmin=0.3E-15, fmax=1E-15 )
#    NGC1522_blue.plot_spectrum_cube(42, 42, fcal=True, lmin=4200, lmax=5000, fmin=0.3E-15, fmax=1E-15 )

#    Erik_RED_combined.plot_spectrum_cube(16, 21, fcal=True, lmin=6790, lmax=6805 )
#    ha_map = Erik_RED_combined.create_map(6765,6780, "ha_map")
#    n2_map = Erik_RED_combined.create_map(6791,6799, "n2_map")
#    Erik_red.map_wavelength(6680, contours=True, fcal=True)
#  title="{} - H-alpha - Integrating [6765 - 6780] $\AA$".format(Erik_RED_combined.description)
#    title="{} - [N II] - Integrating [6791 - 6799] $\AA$".format(Erik_RED_combined.description)


# -----------------------------------------------------------------------------
#    READ FILES, POX 4
# -----------------------------------------------------------------------------


#    rss1p = KOALA_RSS('POX4/16jan20058red.fits')
#    cube1p=Interpolated_cube(rss1p, .3, 1.5)
#    cube1p.plot_spectrum_cube(100,100)
##    cube1.plot_wavelength(6582, save_file="Ben/cube1_ha.png")
#
#    rss2p = KOALA_RSS('POX4/16jan20059red.fits')
#    cube2p=Interpolated_cube(rss2p, .3, 1.5)
#    cube2p.plot_spectrum_cube(100,100)
#
#    rss3p = KOALA_RSS('POX4/16jan20060red.fits')
#    cube3p=Interpolated_cube(rss3p, .3, 1.5)
#    cube3p.plot_spectrum_cube(100,100)
#
#
#    cube1p_aligned,cube2p_aligned,cube3p_aligned=align_3_cubes(cube1p, cube2p, cube3p, rss1p, rss2p, rss3p, pixel_size_arcsec=0.3, kernel_size_arcsec=1.5)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#  MAKE COMBINED CUBE

#    combined_cube=np.zeros_like(cube1_aligned)

#    combined_cube=cube1p_aligned
#    for i in range(0,combined_cube.n_cols):
#        print " Starting column ", i,"..."
#        for j in range (0, combined_cube.n_rows):
#            for k in range (0, combined_cube.n_wave):
#                datita=[cube1p_aligned.data[k,j,i], cube2p_aligned.data[k,j,i], cube3p_aligned.data[k,j,i]]
#                combined_cube.data[k,j,i] =np.nanmedian(datita)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Save FITS file

#    header = fits.open('Ben/16jan20047red.fits')[0].header
#    save_fits_file(combined_cube, header, "Ben/Ben_RED_3_.fits", fcal=flux_correction)

#    rss1 = rss1p
#    rss2 = rss2p
#    rss3 = rss3p
#    header = fits.open('POX4/16jan20058red.fits')[0].header
#    save_fits_file(combined_cube, header, "POX4/POX4_RED_3_.fits", fcal=flux_correction)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#  READ COMBINED CUBE and CREATE MAP and SPECTRUM PLOTS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# BEN: This is what you really need:


#    ben = CUBE("Ben/Ben_RED_3_.fits")
#    ben.map_wavelength(6582, contours=True)
#
#    ben.plot_spectrum_cube(100, 102, 6700, 6800, fcal=True)
##    print "Ratio for S2 at 102, 100 is " + str(ben.calculateRatio(100, 102, 6730, 6745, 6745, 6760))
#
#    ha_map = ben.create_map(6578,6586, "ha_map")
#    N2_map = ben.create_map(6600, 6606, "[NII] map")
#    HN_map = ben.createRatioMap(6575, 6590, 6595, 6610)
#    SS_map = ben.createRatioMap(6730, 6745, 6745, 6760)
#
#
#
#    title="{} - Integrating [6578 - 6586] $\AA$".format(ben.description)
#    ben.plot_map(ha_map, contours=True, title = title, norm=colors.LogNorm(), vmin=100)
#    title = "{} - Integrating [6600 - 6606] $\AA$".format(ben.description)
#    ben.plot_map(N2_map, contours=True, title = title, norm=colors.LogNorm(), vmin=100)
#    title = "{} - H-a/NII Ratio map".format(ben.description)
#    ben.plot_map(HN_map, contours = True, title = title, norm=colors.LogNorm(), vmin=100)
#    title = "{} - SII Ratio map".format(ben.description)
#    ben.plot_map(SS_map, contours = True, title = title, norm=colors.LogNorm(), vmin=100)
#    ben.plot_map(a, contours=True, title = title, vmin=10, vmax=600, cmap="seismic"  )

#    ben.description = "NGC 1522 PA 45$^{\circ}$ - COMBINED CUBE"
#    title="{} - H$\\alpha$ map".format(ben.description)
#    ben.plot_map(ha_map, contours=True, title = title, norm=colors.Normalize(), vmin=0, vmax =500, fig_size=20, save_file="ngc1522_ha.eps" )


#    POX4 = CUBE("POX4/POX4_RED_3_.fits")
#    POX4.plot_spectrum_cube(100, 102, 6635, 6644, fcal=True)
#    ha_map = POX4.create_map(6635, 6644, "ha_map")
#    ha_map = POX4.create_map(6638, 6641, "ha_map")


#    POX4.description = "POX4 PA 120$^{\circ}$ - COMBINED CUBE"
#    title="{} - H$\\alpha$ map".format(POX4.description)
#    POX4.plot_map(ha_map, contours=True, title = title, vmin=70, vmax =5000, fig_size=20, save_file="POX4_ha.png" )


end = timer()
print "\n> Elapsing time = ", end - start, "s"
# -----------------------------------------------------------------------------
#                                     ... Paranoy@ Rulz! ;^D  & Angel R. :-)
# -----------------------------------------------------------------------------
