from .RSS import RSS

class KOALA_RSS(RSS):
    """
    This class reads the FITS files returned by
    `2dfdr
    <https://www.aao.gov.au/science/software/2dfdr>`_
    and performs basic analysis tasks (see description under each method).

    Parameters
    ----------
    filename : string
      FITS file returned by 2dfdr, containing the Raw Stacked Spectra.
      The code makes sure that it contains 1000 spectra
      with 2048 wavelengths each.

    Example
    -------
    >>> pointing1 = KOALA_RSS('data/16jan20058red.fits')
    > Reading file "data/16jan20058red.fits" ...
      2048 wavelength points between 6271.33984375 and 7435.43408203
      1000 spaxels
      These numbers are the right ones for KOALA!
      DONE!
    """

    # -----------------------------------------------------------------------------
    def __init__(self, filename, save_rss_to_fits_file="", rss_clean=False, path="",
                 flat="",  # normalized flat, if needed
                 no_nans=False, mask="", mask_file="", plot_mask=False,  # Mask if given
                 valid_wave_min=0, valid_wave_max=0,  # These two are not needed if Mask is given
                 apply_throughput=False,
                 throughput_2D=[], throughput_2D_file="", throughput_2D_wavecor=False,
                 # nskyflat=True, skyflat="", throughput_file ="", nskyflat_file="", plot_skyflat=False,
                 correct_ccd_defects=False, remove_5577=False, kernel_correct_ccd_defects=51, fibre_p=-1,
                 plot_suspicious_fibres=False,
                 fix_wavelengths=False, sol=[0, 0, 0],
                 do_extinction=False,
                 telluric_correction=[0], telluric_correction_file="",
                 sky_method="none", n_sky=50, sky_fibres=[],  # do_sky=True
                 sky_spectrum=[], sky_rss=[0], scale_sky_rss=0, scale_sky_1D=0.,
                 maxima_sigma=3.,
                 sky_spectrum_file="",
                 brightest_line="Ha", brightest_line_wavelength=0, sky_lines_file="", exclude_wlm=[[0, 0]],
                 is_sky=False, win_sky=0, auto_scale_sky=False, ranges_with_emission_lines=[0], cut_red_end=0,
                 correct_negative_sky=False,
                 order_fit_negative_sky=3, kernel_negative_sky=51, individual_check=True,
                 use_fit_for_negative_sky=False,
                 force_sky_fibres_to_zero=True,
                 high_fibres=20, low_fibres=10,
                 sky_wave_min=0, sky_wave_max=0, cut_sky=5., fmin=1, fmax=10,
                 individual_sky_substraction=False,  # fibre_list=[100,200,300,400,500,600,700,800,900],
                 id_el=False, cut=1.5, broad=1.0, plot_id_el=False, id_list=[0],
                 fibres_to_fix=[],
                 clean_sky_residuals=False, features_to_fix=[], sky_fibres_for_residuals=[],
                 remove_negative_median_values=False,
                 fix_edges=False,
                 clean_extreme_negatives=False, percentile_min=0.5,
                 clean_cosmics=False,
                 # show_cosmics_identification = True,
                 width_bl=20., kernel_median_cosmics=5, cosmic_higher_than=100., extra_factor=1.,
                 max_number_of_cosmics_per_fibre=15,
                 warnings=True, verbose=True,
                 plot=True, plot_final_rss=True, norm=colors.LogNorm(), fig_size=12):

        # ---------------------------------------------- Checking some details

        if rss_clean:  # Just read file if rss_clean = True
            apply_throughput = False
            correct_ccd_defects = False
            fix_wavelengths = False
            sol = [0, 0, 0]
            sky_method = "none"
            do_extinction = False
            telluric_correction = [0]
            telluric_correction_file = ""
            id_el = False
            clean_sky_residuals = False
            fix_edges = False
            # plot_final_rss = plot
            plot = False
            correct_negative_sky = False
            clean_cosmics = False
            clean_extreme_negatives = False
            remove_negative_median_values = False
            verbose = False

        if len(telluric_correction_file) > 0 or telluric_correction[0] != 0:
            do_telluric_correction = True
        else:
            do_telluric_correction = False

        if apply_throughput == False and correct_ccd_defects == False and fix_wavelengths == False and sky_method == "none" and do_extinction == False and telluric_correction == [
            0] and clean_sky_residuals == False and correct_negative_sky == False and clean_cosmics == False and fix_edges == False and clean_extreme_negatives == False and remove_negative_median_values == False and do_telluric_correction == False and is_sky == False:
            # If nothing is selected to do, we assume that the RSS file is CLEAN
            rss_clean = True
            # plot_final_rss = plot
            plot = False
            verbose = False

        if sky_method not in ["self",
                              "selffit"]: force_sky_fibres_to_zero = False  # We don't have sky fibres, sky spectrum is given
        self.sky_fibres = []

        # --------------------------------------------------------------------
        # ------------------------------------------------ 0. Reading the data
        # --------------------------------------------------------------------

        # Create RSS object
        super(KOALA_RSS, self).__init__()

        if path != "": filename = full_path(filename, path)

        print("\n> Reading file", '"' + filename + '"', "...")
        RSS_fits_file = fits.open(filename)  # Open file
        # self.rss_list = []

        #  General info:
        self.object = RSS_fits_file[0].header['OBJECT']
        self.filename = filename
        self.description = self.object + ' \n ' + filename
        self.RA_centre_deg = RSS_fits_file[2].header['CENRA'] * 180 / np.pi
        self.DEC_centre_deg = RSS_fits_file[2].header['CENDEC'] * 180 / np.pi
        self.exptime = RSS_fits_file[0].header['EXPOSED']
        self.history_RSS = RSS_fits_file[0].header['HISTORY']
        self.history = []
        if sol[0] in [0, -1]:
            self.sol = [0, 0, 0]
        else:
            self.sol = sol

        # Read good/bad spaxels
        all_spaxels = list(range(len(RSS_fits_file[2].data)))
        quality_flag = [RSS_fits_file[2].data[i][1] for i in all_spaxels]
        good_spaxels = [i for i in all_spaxels if quality_flag[i] == 1]
        bad_spaxels = [i for i in all_spaxels if quality_flag[i] == 0]

        # Create wavelength, intensity, and variance arrays only for good spaxels
        wcsKOALA = WCS(RSS_fits_file[0].header)
        # variance = RSS_fits_file[1].data[good_spaxels]
        index_wave = np.arange(RSS_fits_file[0].header['NAXIS1'])
        wavelength = wcsKOALA.dropaxis(1).wcs_pix2world(index_wave, 0)[0]
        intensity = RSS_fits_file[0].data[good_spaxels]

        if rss_clean == False:
            print("\n  Number of spectra in this RSS =", len(RSS_fits_file[0].data), ",  number of good spectra =",
                  len(good_spaxels), " ,  number of bad spectra =", len(bad_spaxels))
            if len(bad_spaxels) > 0: print("  Bad fibres =", bad_spaxels)

        # Read errors using RSS_fits_file[1]
        # self.header1 = RSS_fits_file[1].data      # CHECK WHEN DOING ERRORS !!!

        # Read spaxel positions on sky using RSS_fits_file[2]
        self.header2_data = RSS_fits_file[2].data
        # print RSS_fits_file[2].data

        # CAREFUL !! header 2 has the info of BAD fibres, if we are reading from our created RSS files we have to do it in a different way...

        if len(bad_spaxels) == 0:
            offset_RA_arcsec_ = []
            offset_DEC_arcsec_ = []
            for i in range(len(good_spaxels)):
                offset_RA_arcsec_.append(self.header2_data[i][5])
                offset_DEC_arcsec_.append(self.header2_data[i][6])
            offset_RA_arcsec = np.array(offset_RA_arcsec_)
            offset_DEC_arcsec = np.array(offset_DEC_arcsec_)
            variance = np.zeros_like(intensity)  # CHECK FOR ERRORS

        else:
            offset_RA_arcsec = np.array([RSS_fits_file[2].data[i][5]
                                         for i in good_spaxels])
            offset_DEC_arcsec = np.array([RSS_fits_file[2].data[i][6]
                                          for i in good_spaxels])

            self.ID = np.array([RSS_fits_file[2].data[i][0] for i in good_spaxels])  # These are the good fibres
            variance = RSS_fits_file[1].data[good_spaxels]  # CHECK FOR ERRORS

        self.ZDSTART = RSS_fits_file[0].header['ZDSTART']
        self.ZDEND = RSS_fits_file[0].header['ZDEND']

        # KOALA-specific stuff
        self.PA = RSS_fits_file[0].header['TEL_PA']
        self.grating = RSS_fits_file[0].header['GRATID']
        # Check RED / BLUE arm for AAOmega
        if (RSS_fits_file[0].header['SPECTID'] == "RD"):
            AAOmega_Arm = "RED"
        if (RSS_fits_file[0].header['SPECTID'] == "BL"):  # VIRUS
            AAOmega_Arm = "BLUE"

        # For WCS
        self.CRVAL1_CDELT1_CRPIX1 = []
        self.CRVAL1_CDELT1_CRPIX1.append(RSS_fits_file[0].header['CRVAL1'])
        self.CRVAL1_CDELT1_CRPIX1.append(RSS_fits_file[0].header['CDELT1'])
        self.CRVAL1_CDELT1_CRPIX1.append(RSS_fits_file[0].header['CRPIX1'])

        # SET RSS
        # FROM HERE IT WAS self.set_data before   ------------------------------------------

        self.wavelength = wavelength
        self.n_wave = len(wavelength)

        # Check that dimensions match KOALA numbers
        if self.n_wave != 2048 and len(all_spaxels) != 1000:
            print("\n *** WARNING *** : These numbers are NOT the standard ones for KOALA")

        print("\n> Setting the data for this file:")

        if variance.shape != intensity.shape:
            print("\n* ERROR: * the intensity and variance matrices are", \
                  intensity.shape, "and", variance.shape, "respectively\n")
            raise ValueError
        n_dim = len(intensity.shape)
        if n_dim == 2:
            self.intensity = intensity
            self.variance = variance
        elif n_dim == 1:
            self.intensity = intensity.reshape((1, self.n_wave))
            self.variance = variance.reshape((1, self.n_wave))
        else:
            print("\n* ERROR: * the intensity matrix supplied has", \
                  n_dim, "dimensions\n")
            raise ValueError

        self.n_spectra = self.intensity.shape[0]
        self.n_wave = len(self.wavelength)
        print("  Found {} spectra with {} wavelengths" \
              .format(self.n_spectra, self.n_wave), \
              "between {:.2f} and {:.2f} Angstrom" \
              .format(self.wavelength[0], self.wavelength[-1]))
        if self.intensity.shape[1] != self.n_wave:
            print("\n* ERROR: * spectra have", self.intensity.shape[1], \
                  "wavelengths rather than", self.n_wave)
            raise ValueError
        if len(offset_RA_arcsec) != self.n_spectra or \
                len(offset_DEC_arcsec) != self.n_spectra:
            print("\n* ERROR: * offsets (RA, DEC) = ({},{})" \
                  .format(len(self.offset_RA_arcsec),
                          len(self.offset_DEC_arcsec)), \
                  "rather than", self.n_spectra)
            raise ValueError
        else:
            self.offset_RA_arcsec = offset_RA_arcsec
            self.offset_DEC_arcsec = offset_DEC_arcsec

        # Check if NARROW (spaxel_size = 0.7 arcsec)
        # or WIDE (spaxel_size=1.25) field of view
        # (if offset_max - offset_min > 31 arcsec in both directions)
        if np.max(offset_RA_arcsec) - np.min(offset_RA_arcsec) > 31 or \
                np.max(offset_DEC_arcsec) - np.min(offset_DEC_arcsec) > 31:
            self.spaxel_size = 1.25
            field = "WIDE"
        else:
            self.spaxel_size = 0.7
            field = "NARROW"

        # Get min and max for rss
        self.RA_min, self.RA_max, self.DEC_min, self.DEC_max = coord_range([self])
        self.DEC_segment = (self.DEC_max - self.DEC_min) * 3600.  # +1.25 for converting to total field of view
        self.RA_segment = (self.RA_max - self.RA_min) * 3600.  # +1.25

        # --------------------------------------------------------------------
        # ------------------------------------- 1. Reading or getting the mask
        # --------------------------------------------------------------------

        # Reading the mask if needed
        if mask == "" and mask_file == "":
            # print "\n> No mask is given, obtaining it from the RSS file ..." #
            # Only write it on history the first time, when apply_throughput = True
            self.get_mask(include_history=apply_throughput, plot=plot_mask, verbose=verbose)
        else:
            # Include it in the history ONLY if it is the first time (i.e. applying throughput)
            self.read_mask_from_fits_file(mask=mask, mask_file=mask_file, no_nans=no_nans, plot=plot_mask,
                                          verbose=verbose, include_history=apply_throughput)

        if valid_wave_min == 0 and valid_wave_max == 0:  ##############  DIANA FIX !!!
            self.valid_wave_min = self.mask_good_wavelength_range[0]
            self.valid_wave_max = self.mask_good_wavelength_range[1]
            print(
                "\n> Using the values provided by the mask for establishing the good wavelenth range:  [ {:.2f} , {:.2f} ]".format(
                    self.valid_wave_min, self.valid_wave_max))
        else:
            self.valid_wave_min = valid_wave_min
            self.valid_wave_max = valid_wave_max
            print("  As specified, we use the [", self.valid_wave_min, " , ", self.valid_wave_max, "] range.")

            # Plot RSS_image
        if plot: self.RSS_image(image=self.intensity, cmap="binary_r")

        # Deep copy of intensity into intensity_corrected
        self.intensity_corrected = copy.deepcopy(self.intensity)

        # ---------------------------------------------------
        # ------------- PROCESSING THE RSS FILE -------------
        # ---------------------------------------------------

        # ---------------------------------------------------
        # 0. Divide by flatfield if needed
        # Object "flat" has to have a normalized flat response in .intensity_corrected
        # Usually this is found .nresponse , see task "nresponse_flappyflat"
        # However, this correction is not needed is LFLATs have been used in 2dFdr
        # and using a skyflat to get .nresponse (small wavelength variations to throughput)
        if flat != "":
            print("\n> Dividing the data by the flatfield provided...")
            self.intensity_corrected = self.intensity_corrected / flat.intensity_corrected
            self.history.append("- Data divided by flatfield:")
            self.history.append(flat.filename)

        # ---------------------------------------------------
        # 1. Check if apply throughput & apply it if requested    (T)
        text_for_integrated_fibre = "..."
        title_for_integrated_fibre = ""
        plot_this = False
        if apply_throughput:
            # Check if throughput_2D[0][0] = 1., that means the throughput has been computed AFTER  fixing small wavelength variations
            if len(throughput_2D) > 0:
                if throughput_2D[0][0] == 1.:
                    throughput_2D_wavecor = True
                else:
                    throughput_2D_wavecor = False
            else:
                ftf = fits.open(throughput_2D_file)
                self.throughput_2D = ftf[0].data
                if self.throughput_2D[0][0] == 1.:
                    throughput_2D_wavecor = True
                    # throughput_2D_file has in the header the values for sol
                    sol = [0, 0, 0]
                    sol[0] = ftf[0].header["SOL0"]
                    sol[1] = ftf[0].header["SOL1"]
                    sol[2] = ftf[0].header["SOL2"]
                else:
                    throughput_2D_wavecor = False

            if throughput_2D_wavecor:
                print(
                    "\n> The provided throughput 2D information has been computed AFTER fixing small wavelength variations.")
                print(
                    "  Therefore, the throughput 2D will be applied AFTER correcting for ccd defects and small wavelength variations")
                if len(throughput_2D) == 0:
                    print("  The fits file with the throughput 2D has the solution for fixing small wavelength shifts.")
                if self.grating == "580V": remove_5577 = True
            else:
                self.apply_throughput_2D(throughput_2D=throughput_2D, throughput_2D_file=throughput_2D_file, plot=plot)
                text_for_integrated_fibre = "after throughput correction..."
                title_for_integrated_fibre = " - Throughput corrected"
        else:
            if rss_clean == False and verbose == True: print("\n> Intensities NOT corrected for 2D throughput")

        plot_integrated_fibre_again = 0  # Check if we need to plot it again

        # ---------------------------------------------------
        # 2. Correcting for CCD defects                          (C)
        if correct_ccd_defects:
            self.history.append("- Data corrected for CCD defects, kernel_correct_ccd_defects = " + np.str(
                kernel_correct_ccd_defects) + " for running median")
            if plot: plot_integrated_fibre_again = 1

            remove_5577_here = remove_5577
            if sky_method == "1D" and scale_sky_1D == 0: remove_5577_here = False

            if remove_5577_here: self.history.append("  Skyline 5577 removed while cleaning CCD using Gaussian fits")

            self.correct_ccd_defects(kernel_correct_ccd_defects=kernel_correct_ccd_defects,
                                     remove_5577=remove_5577_here,
                                     fibre_p=fibre_p, apply_throughput=apply_throughput, verbose=verbose, plot=plot)

            # Compare corrected vs uncorrected spectrum
            if plot: self.plot_corrected_vs_uncorrected_spectrum(high_fibres=high_fibres, fig_size=fig_size)

            # If removing_5577_here, use the linear fit to the 5577 Gaussian fits in "fix_2dFdr_wavelengths"
            if fix_wavelengths and sol[0] == 0: sol = self.sol

        # ---------------------------------------------------
        # 3. Fixing small wavelength shifts                  (W)
        if fix_wavelengths:
            if sol[0] == -1.0:
                self.fix_2dfdr_wavelengths_edges(verbose=verbose, plot=plot)
            else:
                self.fix_2dfdr_wavelengths(verbose=verbose, plot=plot, sol=sol)

        # Apply throughput 2D corrected for small wavelength shifts if needed
        if apply_throughput and throughput_2D_wavecor:
            self.apply_throughput_2D(throughput_2D=throughput_2D, throughput_2D_file=throughput_2D_file, plot=plot)
            text_for_integrated_fibre = "after throughput correction..."
            title_for_integrated_fibre = " - Throughput corrected"

            # Compute integrated map after throughput correction & plot if requested/needed
        if rss_clean == False:
            if plot == True and plot_integrated_fibre_again != 1:  # correct_ccd_defects == False:
                plot_this = True

            self.compute_integrated_fibre(plot=plot_this, title=title_for_integrated_fibre,
                                          text=text_for_integrated_fibre, warnings=warnings, verbose=verbose,
                                          correct_negative_sky=False)

            # ---------------------------------------------------
        # 4. Get airmass and correct for extinction         (X)
        # DO THIS BEFORE TELLURIC CORRECTION (that is extinction-corrected) OR SKY SUBTRACTION
        ZD = (self.ZDSTART + self.ZDEND) / 2
        self.airmass = 1 / np.cos(np.radians(ZD))
        self.extinction_correction = np.ones(self.n_wave)
        if do_extinction: self.do_extinction_curve(plot=plot, verbose=verbose, fig_size=fig_size)

        # ---------------------------------------------------
        # 5. Check if telluric correction is needed & apply    (U)
        telluric_correction_applied = False
        if do_telluric_correction:
            plot_integrated_fibre_again = plot_integrated_fibre_again + 1
            self.apply_telluric_correction(telluric_correction=telluric_correction,
                                           telluric_correction_file=telluric_correction_file, verbose=verbose)
            if np.nanmax(self.telluric_correction) != 1: telluric_correction_applied = True
        elif self.grating in red_gratings and rss_clean == False and verbose:
            print("\n> Telluric correction will NOT be applied in this RED rss file...")

        # 6. ---------------------------------------------------
        # SKY SUBSTRACTION      sky_method                      (S)
        #
        # Several options here: (1) "1D"      : Consider a single sky spectrum, scale it and substract it
        #                       (2) "2D"      : Consider a 2D sky. i.e., a sky image, scale it and substract it fibre by fibre
        #                       (3) "self"    : Obtain the sky spectrum using the n_sky lowest fibres in the RSS file (DEFAULT)
        #                       (4) "none"    : No sky substraction is performed
        #                       (5) "1Dfit"   : Using an external 1D sky spectrum, fits sky lines in both sky spectrum AND all the fibres
        #                       (6) "selffit" : Using the n_sky lowest fibres, obtain an sky spectrum, then fits sky lines in both sky spectrum AND all the fibres.

        if sky_spectrum_file != "":
            if verbose:
                print("\n> Reading file with a 1D sky spectrum :")
                print(" ", sky_spectrum_file)

            w_sky, sky_spectrum = read_table(sky_spectrum_file, ["f", "f"])

            if np.nanmedian(self.wavelength - w_sky) != 0:
                if verbose or warnings: print(
                    "\n\n  WARNING !!!! The wavelengths provided on this file do not match the wavelengths on this RSS !!\n\n")

            self.history.append('- 1D sky spectrum provided in file :')
            self.history.append('  ' + sky_spectrum_file)

        if sky_method != "none" and is_sky == False:
            plot_integrated_fibre_again = plot_integrated_fibre_again + 1
            self.history.append('- Sky sustraction using the ' + sky_method + ' method')

            if sky_method in ["1Dfit", "selffit"]: self.apply_mask(verbose=verbose)

            # (5) 1Dfit
            if sky_method == "1Dfit":
                self.apply_1Dfit_sky(sky_spectrum=sky_spectrum, n_sky=n_sky, sky_fibres=sky_fibres,
                                     sky_spectrum_file=sky_spectrum_file,
                                     sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, win_sky=win_sky,
                                     scale_sky_1D=scale_sky_1D,
                                     sky_lines_file=sky_lines_file, brightest_line_wavelength=brightest_line_wavelength,
                                     brightest_line=brightest_line, maxima_sigma=maxima_sigma,
                                     auto_scale_sky=auto_scale_sky,
                                     plot=plot, verbose=verbose, fig_size=fig_size, fibre_p=fibre_p,
                                     kernel_correct_ccd_defects=kernel_correct_ccd_defects)

            # (1) If a single sky_spectrum is provided:
            if sky_method == "1D":

                if len(sky_spectrum) > 0:
                    self.apply_1D_sky(sky_fibres=sky_fibres, sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max,
                                      win_sky=win_sky, include_history=True, sky_spectrum=sky_spectrum,
                                      scale_sky_1D=scale_sky_1D, remove_5577=remove_5577,
                                      # sky_spectrum_file = sky_spectrum_file,
                                      plot=plot, verbose=verbose)

                else:
                    print("\n> Sustracting the sky using a sky spectrum requested but any sky spectrum provided !")
                    sky_method = "self"
                    n_sky = 50

                    # (2) If a 2D sky, sky_rss, is provided
            if sky_method == "2D":  # if np.nanmedian(sky_rss.intensity_corrected) != 0:
                #
                # TODO : Needs to be checked and move to an INDEPENDENT task

                if scale_sky_rss != 0:
                    if verbose: print("\n> Using sky image provided to substract sky, considering a scale of",
                                      scale_sky_rss, "...")
                    self.sky_emission = scale_sky_rss * sky_rss.intensity_corrected
                    self.intensity_corrected = self.intensity_corrected - self.sky_emission
                else:
                    if verbose: print(
                        "\n> Using sky image provided to substract sky, computing the scale using sky lines")
                    # check scale fibre by fibre
                    self.sky_emission = copy.deepcopy(sky_rss.intensity_corrected)
                    scale_per_fibre = np.ones((self.n_spectra))
                    scale_per_fibre_2 = np.ones((self.n_spectra))
                    lowlow = 15
                    lowhigh = 5
                    highlow = 5
                    highhigh = 15
                    if self.grating == "580V":
                        if verbose: print("  For 580V we use bright skyline at 5577 AA ...")
                        sky_line = 5577
                        sky_line_2 = 0
                    if self.grating == "1000R":
                        # print "  For 1000R we use skylines at 6300.5 and 6949.0 AA ..."   ### TWO LINES GIVE WORSE RESULTS THAN USING ONLY 1...
                        if verbose: print("  For 1000R we use skyline at 6949.0 AA ...")
                        sky_line = 6949.0  # 6300.5
                        lowlow = 22  # for getting a good continuuem in 6949.0
                        lowhigh = 12
                        highlow = 36
                        highhigh = 52
                        sky_line_2 = 0  # 6949.0  #7276.5 fails
                        lowlow_2 = 22  # for getting a good continuuem in 6949.0
                        lowhigh_2 = 12
                        highlow_2 = 36
                        highhigh_2 = 52
                    if sky_line_2 != 0 and verbose: print("  ... first checking", sky_line, "...")
                    for fibre_sky in range(self.n_spectra):
                        skyline_spec = fluxes(self.wavelength, self.intensity_corrected[fibre_sky], sky_line,
                                              plot=False, verbose=False, lowlow=lowlow, lowhigh=lowhigh,
                                              highlow=highlow, highhigh=highhigh)  # fmin=-5.0E-17, fmax=2.0E-16,
                        # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                        self.intensity_corrected[fibre_sky] = skyline_spec[11]

                        skyline_sky = fluxes(self.wavelength, self.sky_emission[fibre_sky], sky_line, plot=False,
                                             verbose=False, lowlow=lowlow, lowhigh=lowhigh, highlow=highlow,
                                             highhigh=highhigh)  # fmin=-5.0E-17, fmax=2.0E-16,

                        scale_per_fibre[fibre_sky] = skyline_spec[3] / skyline_sky[3]
                        self.sky_emission[fibre_sky] = skyline_sky[11]

                    if sky_line_2 != 0:
                        if verbose: print("  ... now checking", sky_line_2, "...")
                        for fibre_sky in range(self.n_spectra):
                            skyline_spec = fluxes(self.wavelength, self.intensity_corrected[fibre_sky], sky_line_2,
                                                  plot=False, verbose=False, lowlow=lowlow_2, lowhigh=lowhigh_2,
                                                  highlow=highlow_2,
                                                  highhigh=highhigh_2)  # fmin=-5.0E-17, fmax=2.0E-16,
                            # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                            self.intensity_corrected[fibre_sky] = skyline_spec[11]

                            skyline_sky = fluxes(self.wavelength, self.sky_emission[fibre_sky], sky_line_2, plot=False,
                                                 verbose=False, lowlow=lowlow_2, lowhigh=lowhigh_2, highlow=highlow_2,
                                                 highhigh=highhigh_2)  # fmin=-5.0E-17, fmax=2.0E-16,

                            scale_per_fibre_2[fibre_sky] = skyline_spec[3] / skyline_sky[3]
                            self.sky_emission[fibre_sky] = skyline_sky[11]

                            # Median value of scale_per_fibre, and apply that value to all fibres
                    if sky_line_2 == 0:
                        scale_sky_rss = np.nanmedian(scale_per_fibre)
                        self.sky_emission = self.sky_emission * scale_sky_rss
                    else:
                        scale_sky_rss = np.nanmedian((scale_per_fibre + scale_per_fibre_2) / 2)
                        # Make linear fit
                        scale_sky_rss_1 = np.nanmedian(scale_per_fibre)
                        scale_sky_rss_2 = np.nanmedian(scale_per_fibre_2)
                        if verbose:
                            print("  Median scale for line 1 :", scale_sky_rss_1, "range [", np.nanmin(scale_per_fibre),
                                  ",", np.nanmax(scale_per_fibre), "]")
                            print("  Median scale for line 2 :", scale_sky_rss_2, "range [",
                                  np.nanmin(scale_per_fibre_2), ",", np.nanmax(scale_per_fibre_2), "]")

                        b = (scale_sky_rss_1 - scale_sky_rss_2) / (sky_line - sky_line_2)
                        a = scale_sky_rss_1 - b * sky_line
                        if verbose: print("  Appling linear fit with a =", a, "b =", b,
                                          "to all fibres in sky image...")  # ,a+b*sky_line,a+b*sky_line_2

                        for i in range(self.n_wave):
                            self.sky_emission[:, i] = self.sky_emission[:, i] * (a + b * self.wavelength[i])

                    if plot:
                        plt.figure(figsize=(fig_size, fig_size / 2.5))
                        label1 = "$\lambda$" + np.str(sky_line)
                        plt.plot(scale_per_fibre, alpha=0.5, label=label1)
                        plt.minorticks_on()
                        plt.ylim(np.nanmin(scale_per_fibre), np.nanmax(scale_per_fibre))
                        plt.axhline(y=scale_sky_rss, color='k', linestyle='--')
                        if sky_line_2 == 0:
                            text = "Scale OBJECT / SKY using sky line $\lambda$" + np.str(sky_line)
                            if verbose:
                                print("  Scale per fibre in the range [", np.nanmin(scale_per_fibre), ",",
                                      np.nanmax(scale_per_fibre), "], median value is", scale_sky_rss)
                                print("  Using median value to scale sky emission provided...")
                        if sky_line_2 != 0:
                            text = "Scale OBJECT / SKY using sky lines $\lambda$" + np.str(
                                sky_line) + " and $\lambda$" + np.str(sky_line_2)
                            label2 = "$\lambda$" + np.str(sky_line_2)
                            plt.plot(scale_per_fibre_2, alpha=0.5, label=label2)
                            plt.axhline(y=scale_sky_rss_1, color='k', linestyle=':')
                            plt.axhline(y=scale_sky_rss_2, color='k', linestyle=':')
                            plt.legend(frameon=False, loc=1, ncol=2)
                        plt.title(text)
                        plt.xlabel("Fibre")
                        plt.show()
                        plt.close()
                    self.intensity_corrected = self.intensity_corrected - self.sky_emission
                self.apply_mask(verbose=verbose)
                self.history(" - 2D sky subtraction performed")

            # (6) "selffit"
            if sky_method == "selffit":
                # TODO : Needs to be an independent task : apply_selffit_sky !!!

                if verbose: print("\n> 'sky_method = selffit', hence using", n_sky,
                                  "lowest intensity fibres to create a sky spectrum ...")

                self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                                       sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max,
                                       win_sky=win_sky, include_history=True)

                if sky_spectrum[0] != -1 and np.nanmedian(sky_spectrum) != 0:
                    if verbose: print(
                        "\n> Additional sky spectrum provided. Using this for replacing regions with bright emission lines...")

                    sky_r_self = self.sky_emission

                    self.sky_emission = replace_el_in_sky_spectrum(self, sky_r_self, sky_spectrum,
                                                                   scale_sky_1D=scale_sky_1D,
                                                                   brightest_line=brightest_line,
                                                                   brightest_line_wavelength=brightest_line_wavelength,
                                                                   ranges_with_emission_lines=ranges_with_emission_lines,
                                                                   cut_red_end=cut_red_end,
                                                                   plot=plot)
                    self.history.append('  Using sky spectrum provided for replacing regions with emission lines')

                self.fit_and_substract_sky_spectrum(self.sky_emission, sky_lines_file=sky_lines_file,
                                                    brightest_line_wavelength=brightest_line_wavelength,
                                                    brightest_line=brightest_line,
                                                    maxima_sigma=maxima_sigma, ymin=-50, ymax=600, wmin=0, wmax=0,
                                                    auto_scale_sky=auto_scale_sky,
                                                    warnings=False, verbose=False, plot=False, fig_size=fig_size,
                                                    fibre=fibre_p)

                if fibre_p == -1:
                    if verbose: print(
                        "\n> 'selffit' sky_method usually generates some nans, correcting ccd defects again...")
                    self.correct_ccd_defects(kernel_correct_ccd_defects=kernel_correct_ccd_defects, verbose=verbose,
                                             plot=plot, only_nans=True)  # not replacing values < 0

            # (3) "self": Obtain the sky using the n_sky lowest fibres
            #             If a 1D spectrum is provided, use it for replacing regions with bright emission lines   #DIANA
            if sky_method == "self":

                self.sky_fibres = sky_fibres
                if n_sky == 0: n_sky = len(sky_fibres)
                self.apply_self_sky(sky_fibres=self.sky_fibres, sky_spectrum=sky_spectrum, n_sky=n_sky,
                                    sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, win_sky=win_sky,
                                    scale_sky_1D=scale_sky_1D,
                                    brightest_line="Ha", brightest_line_wavelength=0, ranges_with_emission_lines=[0],
                                    cut_red_end=cut_red_end, low_fibres=low_fibres,
                                    use_fit_for_negative_sky=use_fit_for_negative_sky,
                                    kernel_negative_sky=kernel_negative_sky,
                                    order_fit_negative_sky=order_fit_negative_sky,
                                    plot=True, verbose=verbose)

        # Correct negative sky if requested
        if is_sky == False and correct_negative_sky == True:
            text_for_integrated_fibre = "after correcting negative sky"
            self.correcting_negative_sky(plot=plot, low_fibres=low_fibres, kernel_negative_sky=kernel_negative_sky,
                                         order_fit_negative_sky=order_fit_negative_sky,
                                         individual_check=individual_check,
                                         use_fit_for_negative_sky=use_fit_for_negative_sky,
                                         force_sky_fibres_to_zero=force_sky_fibres_to_zero)  # exclude_wlm=exclude_wlm

        # Check Median spectrum of the sky fibres AFTER subtracting the sky emission
        if plot == True and len(self.sky_fibres) > 0:
            sky_emission = sky_spectrum_from_fibres(self, self.sky_fibres, win_sky=0, plot=False, include_history=False,
                                                    verbose=False)
            plot_plot(self.wavelength, sky_emission, hlines=[0],
                      ptitle="Median spectrum of the sky fibres AFTER subtracting the sky emission")
            # plot_plot(self.wavelength,self.sky_emission,hlines=[0],ptitle = "Median spectrum using self.sky_emission")

        # If this RSS is an offset sky, perform a median filter to increase S/N
        if is_sky:
            self.is_sky(n_sky=n_sky, win_sky=win_sky, sky_fibres=sky_fibres, sky_wave_min=sky_wave_min,
                        sky_wave_max=sky_wave_max, plot=plot, verbose=verbose)
            if win_sky == 0: win_sky = 151  # Default value in is_sky

        # ---------------------------------------------------
        # 7. Check if identify emission lines is requested & do      (E)
        # TODO: NEEDS TO BE CHECKED !!!!
        if id_el:
            if brightest_line_wavelength == 0:
                self.el = self.identify_el(high_fibres=high_fibres, brightest_line=brightest_line,
                                           cut=cut, verbose=True, plot=plot_id_el, fibre=0, broad=broad)
                print("\n  Emission lines identified saved in self.el !!")
            else:
                brightest_line_rest_wave = 6562.82
                print("\n  As given, line ", brightest_line, " at rest wavelength = ", brightest_line_rest_wave,
                      " is at ", brightest_line_wavelength)
                self.el = [[brightest_line], [brightest_line_rest_wave], [brightest_line_wavelength], [7.2]]
                #  sel.el=[peaks_name,peaks_rest, p_peaks_l, p_peaks_fwhm]
        else:
            self.el = [[0], [0], [0], [0]]

        # Check if id_list provided
        if id_list[0] != 0:
            if id_el:
                print("\n> Checking if identified emission lines agree with list provided")
                # Read list with all emission lines to get the name of emission lines
                emission_line_file = "lineas_c89_python.dat"
                el_center, el_name = read_table(emission_line_file, ["f", "s"])

                # Find brightest line to get redshift
                for i in range(len(self.el[0])):
                    if self.el[0][i] == brightest_line:
                        obs_wave = self.el[2][i]
                        redshift = (self.el[2][i] - self.el[1][i]) / self.el[1][i]
                print("  Brightest emission line", brightest_line, "found at ", obs_wave, ", redshift = ", redshift)

                el_identified = [[], [], [], []]
                n_identified = 0
                for line in id_list:
                    id_check = 0
                    for i in range(len(self.el[1])):
                        if line == self.el[1][i]:
                            if verbose: print("  Emission line ", self.el[0][i], self.el[1][i], "has been identified")
                            n_identified = n_identified + 1
                            id_check = 1
                            el_identified[0].append(self.el[0][i])  # Name
                            el_identified[1].append(self.el[1][i])  # Central wavelength
                            el_identified[2].append(self.el[2][i])  # Observed wavelength
                            el_identified[3].append(self.el[3][i])  # "FWHM"
                    if id_check == 0:
                        for i in range(len(el_center)):
                            if line == el_center[i]:
                                el_identified[0].append(el_name[i])
                                print("  Emission line", el_name[i], line, "has NOT been identified, adding...")
                        el_identified[1].append(line)
                        el_identified[2].append(line * (redshift + 1))
                        el_identified[3].append(4 * broad)

                self.el = el_identified
                print("  Number of emission lines identified = ", n_identified, "of a total of", len(id_list),
                      "provided. self.el updated accordingly")
            else:
                if rss_clean == False: print("\n> List of emission lines provided but no identification was requested")

        # ---------------------------------------------------
        # 8.1. Clean sky residuals if requested           (R)
        if clean_sky_residuals:
            # plot_integrated_fibre_again = plot_integrated_fibre_again + 1
            # self.clean_sky_residuals(extra_w=extra_w, step=step_csr, dclip=dclip, verbose=verbose, fibre=fibre, wave_min=valid_wave_min,  wave_max=valid_wave_max)

            if len(features_to_fix) == 0:  # Add features that are known to be problematic

                if self.wavelength[0] < 6250 and self.wavelength[-1] > 6350:
                    features_to_fix.append(["r", 6250, 6292, 6308, 6350, 2, 98, 2, False, False])  # 6301
                if self.wavelength[0] < 7550 and self.wavelength[-1] > 7660:
                    features_to_fix.append(
                        ["r", 7558, 7595, 7615, 7652, 2, 98, 2, False, False])  # Big telluric absorption
                # if self.wavelength[0] < 8550 and  self.wavelength[-1] >  8770:
                #    features_to_fix.append(["s", 8560, 8610, 8685, 8767, 2, 98, 2, False,False])

            elif features_to_fix == "big_telluric" or features_to_fix == "big_telluric_absorption":
                features_to_fix = [["r", 7558, 7595, 7615, 7652, 2, 98, 2, False, False]]

            if len(features_to_fix) > 0:

                if verbose:
                    print("\n> Features to fix: ")
                    for feature in features_to_fix:
                        print("  -", feature)

                if len(sky_fibres_for_residuals) == 0:
                    self.find_sky_fibres(sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, n_sky=np.int(n_sky / 2),
                                         plot=plot, warnings=False)
                    sky_fibres_for_residuals = self.sky_fibres
                fix_these_features_in_all_spectra(self, features=features_to_fix,
                                                  fibre_list=fibres_to_fix,  # range(83,test.n_spectra),
                                                  sky_fibres=sky_fibres_for_residuals,
                                                  replace=True, plot=plot)
                # check functions for documentation
        # ---------------------------------------------------
        # 8.2. Clean edges if requested           (R)
        if fix_edges:
            self.fix_edges(verbose=verbose)
        # ---------------------------------------------------
        # 8.3. Remove negative median values      (R)
        if remove_negative_median_values:  # it was remove_negative_pixels_in_sky:
            self.intensity_corrected = remove_negative_pixels(self.intensity_corrected, verbose=verbose)
            self.history.append("- Spectra with negative median values corrected to median = 0")
        # ---------------------------------------------------
        # 8.4. Clean extreme negatives      (R)
        if clean_extreme_negatives:
            self.clean_extreme_negatives(fibre_list=fibres_to_fix, percentile_min=percentile_min, plot=plot,
                                         verbose=verbose)
        # ---------------------------------------------------
        # 8.5. Clean cosmics    (R)
        if clean_cosmics:
            self.kill_cosmics(brightest_line_wavelength, width_bl=width_bl, kernel_median_cosmics=kernel_median_cosmics,
                              cosmic_higher_than=cosmic_higher_than, extra_factor=extra_factor,
                              max_number_of_cosmics_per_fibre=max_number_of_cosmics_per_fibre,
                              fibre_list=fibres_to_fix, plot_cosmic_image=plot, plot_RSS_images=plot, verbose=verbose)
        # ---------------------------------------------------
        # 8.3 Clean ccd residuals, including extreme negative, if requested    (R)
        # if clean_ccd_residuals:
        #     self.intensity_corrected=clean_cosmics_and_extreme_negatives(self, #fibre_list="",
        #                                     clean_cosmics = clean_cosmics, only_positive_cosmics= only_positive_cosmics,
        #                                     disp_scale=disp_scale, disp_to_sqrt_scale = disp_to_sqrt_scale, ps_min = ps_min, width = width,
        #                                     clean_extreme_negatives = clean_extreme_negatives, percentile_min = percentile_min,
        #                                     remove_negative_median_values=remove_negative_median_values,
        #                                     show_correction_map = show_correction_map,
        #                                     show_fibres=show_fibres,
        #                                     show_cosmics_identification = show_cosmics_identification,
        #                                     plot=plot, verbose=False)

        # Finally, apply mask making nans
        if rss_clean == False: self.apply_mask(make_nans=True, verbose=verbose)

        # ---------------------------------------------------
        # LAST CHECKS and PLOTS

        # if fibre_p != 0: plot_integrated_fibre_again = 0

        if plot_integrated_fibre_again > 0:
            # Plot corrected values
            if rss_clean:
                text = "..."
            else:
                text = "after all corrections have been applied..."
            self.compute_integrated_fibre(plot=plot, title=" - Intensities Corrected", warnings=warnings, text=text,
                                          verbose=verbose,
                                          valid_wave_min=valid_wave_min, valid_wave_max=valid_wave_max, last_check=True,
                                          low_fibres=low_fibres, correct_negative_sky=False,
                                          individual_check=False, order_fit_negative_sky=order_fit_negative_sky,
                                          kernel_negative_sky=kernel_negative_sky,
                                          use_fit_for_negative_sky=use_fit_for_negative_sky)

        # Plot correct vs uncorrected spectra
        if plot == True:
            self.plot_corrected_vs_uncorrected_spectrum(high_fibres=high_fibres, fig_size=fig_size)
            self.plot_corrected_vs_uncorrected_spectrum(low_fibres=low_fibres, fig_size=fig_size)

        # Plot RSS_image
        if plot or plot_final_rss: self.RSS_image()

        # If this is a CLEAN RSS, be sure self.integrated_fibre is obtained
        if rss_clean: self.compute_integrated_fibre(plot=False, warnings=False, verbose=False)

        # Print summary and information from header
        print("\n> Summary of reading rss file", '"' + filename + '"', ":\n")
        print("  This is a KOALA {} file,".format(AAOmega_Arm), \
              "using the {} grating in AAOmega, ".format(self.grating), \
              "exposition time = {} s.".format(self.exptime))
        print("  Object:", self.object)
        print("  Field of view:", field, \
              "(spaxel size =", self.spaxel_size, "arcsec)")
        print("  Center position: (RA, DEC) = ({:.3f}, {:.3f}) degrees" \
              .format(self.RA_centre_deg, self.DEC_centre_deg))
        print("  Field covered [arcsec] = {:.1f} x {:.1f}".format(self.RA_segment + self.spaxel_size,
                                                                  self.DEC_segment + self.spaxel_size))
        print("  Position angle (PA) = {:.1f} degrees".format(self.PA))
        print(" ")

        if rss_clean == True and is_sky == False:
            print("  This was a CLEAN RSS file, no correction was applied!")
            print("  Values stored in self.intensity_corrected are the same that those in self.intensity")
        else:
            if flat != "": print("  Intensities divided by the given flatfield")
            if apply_throughput:
                if len(throughput_2D) > 0:
                    print("  Intensities corrected for throughput 2D using provided variable !")
                else:
                    print("  Intensities corrected for throughput 2D using provided file !")
                    # print " ",throughput_2D_file
            else:
                print("  Intensities NOT corrected for throughput 2D")
            if correct_ccd_defects:
                print("  Intensities corrected for CCD defects !")
            else:
                print("  Intensities NOT corrected for CCD defects")

            if sol[0] != 0 and fix_wavelengths:
                print("  All fibres corrected for small wavelength shifts using wavelength solution provided!")
            else:
                if fix_wavelengths:
                    print(
                        "  Wavelengths corrected for small shifts using Gaussian fit to selected bright skylines in all fibres!")
                else:
                    print("  Wavelengths NOT corrected for small shifts")

            if do_extinction:
                print("  Intensities corrected for extinction !")
            else:
                print("  Intensities NOT corrected for extinction")

            if telluric_correction_applied:
                print("  Intensities corrected for telluric absorptions !")
            else:
                if self.grating in red_gratings: print("  Intensities NOT corrected for telluric absorptions")

            if is_sky:
                print("  This is a SKY IMAGE, median filter with window", win_sky, "applied !")
                print("  The median 1D sky spectrum combining", n_sky, "lowest fibres is stored in self.sky_emission")
            else:
                if sky_method == "none": print("  Intensities NOT corrected for sky emission")
                if sky_method == "self": print("  Intensities corrected for sky emission using", n_sky,
                                               "spaxels with lowest values !")
                if sky_method == "selffit": print("  Intensities corrected for sky emission using", n_sky,
                                                  "spaxels with lowest values !")
                if sky_method == "1D": print(
                    "  Intensities corrected for sky emission using (scaled) spectrum provided ! ")
                if sky_method == "1Dfit": print(
                    "  Intensities corrected for sky emission fitting Gaussians to both 1D sky spectrum and each fibre ! ")
                if sky_method == "2D": print(
                    "  Intensities corrected for sky emission using sky image provided scaled by", scale_sky_rss, "!")

            if correct_negative_sky: print(
                "  Intensities corrected to make the integrated value of the lowest fibres = 0 !")

            if id_el:
                print(" ", len(self.el[0]), "emission lines identified and stored in self.el !")
                print(" ", self.el[0])

            if clean_sky_residuals:
                print("  Sky residuals CLEANED !")
            else:
                print("  Sky residuals have NOT been cleaned")

            if fix_edges: print("  The edges of the RSS have been fixed")
            if remove_negative_median_values: print("  Negative median values have been corrected")
            if clean_extreme_negatives: print("  Extreme negative values have been removed!")
            if clean_cosmics: print("  Cosmics have been removed!")

            print("\n  All applied corrections are stored in self.intensity_corrected !")

            if save_rss_to_fits_file != "":
                if save_rss_to_fits_file == "auto":
                    clean_residuals = False
                    if clean_cosmics == True or clean_extreme_negatives == True or remove_negative_median_values == True or fix_edges == True or clean_sky_residuals == True: clean_residuals = True
                    save_rss_to_fits_file = name_keys(filename, apply_throughput=apply_throughput,
                                                      correct_ccd_defects=correct_ccd_defects,
                                                      fix_wavelengths=fix_wavelengths, do_extinction=do_extinction,
                                                      sky_method=sky_method,
                                                      do_telluric_correction=telluric_correction_applied, id_el=id_el,
                                                      correct_negative_sky=correct_negative_sky,
                                                      clean_residuals=clean_residuals)

                save_rss_fits(self, fits_file=save_rss_to_fits_file)

        if rss_clean == False: print("\n> KOALA RSS file read !")