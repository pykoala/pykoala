# ask Angel about *** what they are, and description, Once you have type and decription look for all the same ones

# search for *** when looking for something that isn't done
# final check search for TYPE and DESCRIPTION


class Interpolated_cube(object):  # TASK_Interpolated_cube

    """
    Constructs a cube by accumulating RSS with given offsets.

    Default values:

    RSS : This is the file that has the raw stacked spectra, it can be a file or an object created with KOALA_RSS.

    Defining the cube:
        pixel_size_arcsec: This is the size of the pixels in arcseconds. (default 0.7)

        kernel_size_arcsec: This is the size of the kernels in arcseconds. (default 1.4)

        centre_deg=[]: This is the centre of the cube relative to the position in the sky, using RA and DEC, should be either empty or a List of 2 floats. (default empty List)

        size_arcsec=[]:This is the size of the cube in arcseconds, should be either empty or a List of 2 floats. (default empty List)

        zeros=False: This decides if the cube created is empty of not, if True creates an empty cube, if False uses RSS to create the cube. (default False)

        trim_cube = False: If True the cube witll be trimmed based on (box_x and box_y), else no trimming is done. (default False)

        shape=[]: This is the shape of the cube, should either be empty or an List of 2 integers. (default empty List)

        n_cols=2: This is the number of columns the cube has. (default 2)

        n_rows=2: This is the number of rows the cube has. (default 2)


    Directories:
        path="": This is the directory path to the folder where the rss_file is located. (default "")


    Alignment:
        box_x=[0,-1]: When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. (default box not used)

        box_y=[0,-1]: When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. (default box not used)

        half_size_for_centroid = 10: This determines half the width/height of the for the box centred at the maximum value, should be integer. (default 10)

        g2d=False: If True uses a 2D gaussian, else doesn't. (default False)

        aligned_coor=False: If True uses a 2D gaussian, else doesn't. (default False)

        delta_RA =0: This is a small offset of the RA (right ascension). (default 0)

        delta_DEC=0: This is a small offset of the DEC (declination). (default 0)

        offsets_files="": ***

        offsets_files_position ="": ***


    Flux calibration:
        flux_calibration=[0]: This is the flux calibration. (default empty List)

        flux_calibration_file="": This is the directory of the flux calibration file. (default "")


    ADR:
        ADR=False: If True will correct for ADR (Atmospheric Differential Refraction). (default False)

        force_ADR = False: If True will correct for ADR even considoring a small correction. (default False)

        jump = -1: If a positive number partitions the wavelengths with step size jump, if -1 will not partition. (default -1)

        adr_index_fit = 2: This is the fitted polynomial with highest degree n. (default n = 2)

        ADR_x_fit=[0]: This is the ADR coefficients values for the x axis. (default constant 0)

        ADR_y_fit=[0]: This is the ADR coefficients values for the x axis. (default constant 0)

        check_ADR = False: ***    # remove check_ADR?

        edgelow = -1: This is the lowest value in the wavelength range in terms of pixels. (default -1)

        edgehigh = -1, This is the highest value in the wavelength range in terms of pixels, (maximum wavelength - edgehigh). (default -1) (pixels)


    Reading cubes:
        read_fits_cube = False: If True uses the inputs for the cube, else uses RSS inputs. (default False)

        n_wave=2048: This is the number of wavelengths, which is the length of the array. (default 2048)

        wavelength=[]: This is an a List with the wavelength solution. (default empty List)

        description="": This is the description of the cube. (default "")

        objeto="": This is an object with an analysis. (default "")

        PA=0: This is the position angle. (default 0)

        valid_wave_min = 0: This is the minimum wavelength shown on the spectrum. (default 0)

        valid_wave_max = 0: This is the maximum wavelegth shown on the sprectrum. (defulat 0)

        grating="": This is the grating used in AAOmega. (default "")

        CRVAL1_CDELT1_CRPIX1=[0,0,0]: This are the values for the wavelength calibrations, provided by 2dFdr. (default [0,0,0])

        total_exptime=0: This is the exposition time. (default 0)

        number_of_combined_files = 1: This is the total number of files that have been cominded. (default 1)


    Plotting / Printing:
        plot_tracing_maps=[]: This is the maps to be plotted. (default empty List)

        plot=False: If True shows all the plots, else doesn't show plots. (default False)

        verbose=True: Prints the results. (default True)

        warnings=False: If True will show any problems that arose, else skipped. (default False)


    """

    # -----------------------------------------------------------------------------
    def __init__(self, RSS, pixel_size_arcsec=0.7, kernel_size_arcsec=1.4,  # RSS is an OBJECT
                 rss_file="", path="",
                 centre_deg=[], size_arcsec=[], aligned_coor=False,
                 delta_RA=0, delta_DEC=0,
                 flux_calibration=[0], flux_calibration_file="",
                 zeros=False,
                 ADR=False, force_ADR=False, jump=-1, adr_index_fit=2,
                 ADR_x_fit=[0], ADR_y_fit=[0], g2d=False, check_ADR=False,  # remove check_ADR?

                 step_tracing=100,

                 offsets_files="", offsets_files_position="", shape=[],
                 edgelow=-1, edgehigh=-1,
                 box_x=[0, -1], box_y=[0, -1], half_size_for_centroid=10,
                 trim_cube=False, remove_spaxels_not_fully_covered=True,

                 warnings=False,
                 read_fits_cube=False, n_wave=2048, wavelength=[], description="", objeto="", PA=0,
                 valid_wave_min=0, valid_wave_max=0,
                 grating="", CRVAL1_CDELT1_CRPIX1=[0, 0, 0], total_exptime=0, n_cols=2, n_rows=2,
                 number_of_combined_files=1,

                 plot_tracing_maps=[], plot_rss=True, plot=False, plot_spectra=True,
                 log=True, gamma=0.,
                 verbose=True, fig_size=12):

        if plot == False:
            plot_tracing_maps = []
            plot_rss = False
            plot_spectra = False

        self.pixel_size_arcsec = pixel_size_arcsec
        self.kernel_size_arcsec = kernel_size_arcsec
        self.kernel_size_pixels = kernel_size_arcsec / pixel_size_arcsec  # must be a float number!
        self.integrated_map = []

        self.history = []
        fcal = False

        if rss_file != "" or type(RSS) == str:
            if type(RSS) == str: rss_file = RSS
            rss_file = full_path(rss_file, path)  # RSS
            RSS = KOALA_RSS(rss_file, rss_clean=True, plot=plot, plot_final_rss=plot_rss, verbose=verbose)

        if read_fits_cube:  # RSS is a cube given in fits file
            self.n_wave = n_wave
            self.wavelength = wavelength
            self.description = description  # + " - CUBE"
            self.object = objeto
            self.PA = PA
            self.grating = grating
            self.CRVAL1_CDELT1_CRPIX1 = CRVAL1_CDELT1_CRPIX1
            self.total_exptime = total_exptime
            self.number_of_combined_files = number_of_combined_files
            self.valid_wave_min = valid_wave_min
            self.valid_wave_max = valid_wave_max

        else:
            # self.RSS = RSS
            self.n_spectra = RSS.n_spectra
            self.n_wave = RSS.n_wave
            self.wavelength = RSS.wavelength
            self.description = RSS.description + "\n CUBE"
            self.object = RSS.object
            self.PA = RSS.PA
            self.grating = RSS.grating
            self.CRVAL1_CDELT1_CRPIX1 = RSS.CRVAL1_CDELT1_CRPIX1
            self.total_exptime = RSS.exptime
            self.exptimes = [self.total_exptime]
            self.offset_RA_arcsec = RSS.offset_RA_arcsec
            self.offset_DEC_arcsec = RSS.offset_DEC_arcsec

            self.rss_list = RSS.filename
            self.valid_wave_min = RSS.valid_wave_min
            self.valid_wave_max = RSS.valid_wave_max
            self.valid_wave_min_index = RSS.valid_wave_min_index
            self.valid_wave_max_index = RSS.valid_wave_max_index

        self.offsets_files = offsets_files  # Offsets between files when align cubes
        self.offsets_files_position = offsets_files_position  # Position of this cube when aligning

        self.seeing = 0.0
        self.flux_cal_step = 0.0
        self.flux_cal_min_wave = 0.0
        self.flux_cal_max_wave = 0.0
        self.adrcor = False

        if zeros:
            if read_fits_cube == False and verbose:
                print("\n> Creating empty cube using information provided in rss file: ")
                print(" ", self.description.replace("\n", ""))
        else:
            if verbose: print("\n> Creating cube from file rss file: ")
            if verbose: print(" ", self.description.replace("\n", ""))
        if read_fits_cube == False and verbose:
            print("  Pixel size  = ", self.pixel_size_arcsec, " arcsec")
            print("  kernel size = ", self.kernel_size_arcsec, " arcsec")

        # centre_deg = [RA,DEC] if we need to give new RA, DEC centre
        if len(centre_deg) == 2:
            self.RA_centre_deg = centre_deg[0]
            self.DEC_centre_deg = centre_deg[1]
        else:
            self.RA_centre_deg = RSS.RA_centre_deg + delta_RA / 3600.
            self.DEC_centre_deg = RSS.DEC_centre_deg + delta_DEC / 3600.

        if read_fits_cube == False:
            if aligned_coor == True:
                self.xoffset_centre_arcsec = (self.RA_centre_deg - RSS.ALIGNED_RA_centre_deg) * 3600.
                self.yoffset_centre_arcsec = (self.DEC_centre_deg - RSS.ALIGNED_DEC_centre_deg) * 3600.
                if zeros == False and verbose:
                    print("  Using ALIGNED coordenates for centering cube...")
            else:
                self.xoffset_centre_arcsec = (self.RA_centre_deg - RSS.RA_centre_deg) * 3600.
                self.yoffset_centre_arcsec = (self.DEC_centre_deg - RSS.DEC_centre_deg) * 3600.

            if len(size_arcsec) == 2:
                if aligned_coor == False:
                    if verbose: print(
                        '  The size of the cube has been given: {}" x {}"'.format(size_arcsec[0], size_arcsec[1]))
                    self.n_cols = np.int(size_arcsec[0] / self.pixel_size_arcsec)
                    self.n_rows = np.int(size_arcsec[1] / self.pixel_size_arcsec)
                else:
                    self.n_cols = np.int(size_arcsec[0] / self.pixel_size_arcsec) + 2 * np.int(
                        self.kernel_size_arcsec / self.pixel_size_arcsec)
                    self.n_rows = np.int(size_arcsec[1] / self.pixel_size_arcsec) + 2 * np.int(
                        self.kernel_size_arcsec / self.pixel_size_arcsec)
            else:
                self.n_cols = 2 * \
                              (np.int(np.nanmax(
                                  np.abs(RSS.offset_RA_arcsec - self.xoffset_centre_arcsec)) / self.pixel_size_arcsec)
                               + np.int(self.kernel_size_pixels)) + 3  # -3    ### +1 added by Angel 25 Feb 2018 to put center in center
                self.n_rows = 2 * \
                              (np.int(np.nanmax(
                                  np.abs(RSS.offset_DEC_arcsec - self.yoffset_centre_arcsec)) / self.pixel_size_arcsec)
                               + np.int(self.kernel_size_pixels)) + 3  # -3   ### +1 added by Angel 25 Feb 2018 to put center in center

            # if self.n_cols % 2 != 0:
            #     self.n_cols += 1   # Even numbers to have [0,0] in the centre
            #     if len(size_arcsec) == 2 and aligned_coor == False and verbose: print("  We need an even number of spaxels, adding an extra column...")
            # if self.n_rows % 2 != 0:
            #     self.n_rows += 1
            #     if len(size_arcsec) == 2 and aligned_coor == False and verbose: print("  We need an even number of spaxels, adding an extra row...")
            # # If we define a specific shape
            if len(shape) == 2:
                self.n_rows = shape[0]
                self.n_cols = shape[1]
        else:
            self.n_cols = n_cols
            self.n_rows = n_rows

        self.spaxel_RA0 = self.n_cols / 2 - 1
        self.spaxel_DEC0 = self.n_rows / 2 - 1

        # Define zeros
        self.weighted_I = np.zeros((self.n_wave, self.n_rows, self.n_cols))
        self.weight = np.zeros_like(self.weighted_I)
        self.flux_calibration = np.zeros(self.n_wave)

        # Check ADR values
        self.ADR_x_fit = ADR_x_fit
        self.ADR_y_fit = ADR_y_fit
        pp = np.poly1d(ADR_x_fit)
        self.ADR_x = pp(self.wavelength)
        pp = np.poly1d(ADR_y_fit)
        self.ADR_y = pp(self.wavelength)

        ADR_repeat = True
        if np.nansum(self.ADR_y + self.ADR_x) != 0:  # When values for ADR correction are given
            self.history.append("- ADR fit values provided, cube built considering them")
            ADR_repeat = False  # Do not repeat the ADR correction
            self.adrcor = True  # The values for ADR are given
            # Computing jump automatically (only when not reading the cube)
            if read_fits_cube == False:
                if jump < -1: jump == np.abs(jump)
                if jump == -1:
                    cubo_ADR_total = np.sqrt(self.ADR_x ** 2 + self.ADR_y ** 2)
                    stop = 0
                    i = 1
                    while stop < 1:
                        if np.abs(cubo_ADR_total[i] - cubo_ADR_total[0]) > 0.022:  # 0.012:
                            jump = i
                            if verbose: print('  Automatically found that with jump = ', jump,
                                              ' in lambda, the ADR offset is lower than 0.01"')
                            self.history.append("  Automatically found jump = " + np.str(jump) + " for ADR")
                            stop = 2
                        else:
                            i = i + 1
                        if i == len(self.wavelength) / 2.:
                            jump = -1
                            if verbose: print(
                                '  No value found for jump smaller than half the size of the wavelength! \n  Using jump = -1 (no jump) for ADR.')
                            self.history.append("  Using jump = -1 (no jump) for ADR")

                            stop = 2
                else:
                    if verbose: print("  As requested, creating cube considering the median value each ", jump,
                                      " lambdas for correcting ADR...")
                    self.history.append("  Using given value of jump = " + np.str(jump) + " for ADR")
        else:
            self.history.append("- Cube built without considering ADR correction")

        self.RA_segment = self.n_cols * self.pixel_size_arcsec
        self.DEC_segment = self.n_rows * self.pixel_size_arcsec

        if zeros:
            self.data = np.zeros_like(self.weighted_I)
        else:
            # Build the cube
            self.data = self.build_cube(jump=jump, RSS=RSS)

            # Define box for tracing peaks if requested
            if half_size_for_centroid > 0 and np.nanmedian(box_x + box_y) == -0.5:
                box_x, box_y = self.box_for_centroid(half_size_for_centroid=half_size_for_centroid, verbose=verbose,
                                                     plot_map=plot, log=log)
                if verbose: print("  Using this box for tracing peaks and checking ADR ...")

            # Trace peaks (check ADR only if requested)
            if ADR_repeat:
                _check_ADR_ = False
            else:
                _check_ADR_ = True

            if ADR:
                self.trace_peak(box_x=box_x, box_y=box_y, edgelow=edgelow, edgehigh=edgehigh, plot=plot,
                                plot_tracing_maps=plot_tracing_maps,
                                verbose=verbose, adr_index_fit=adr_index_fit, g2d=g2d, check_ADR=_check_ADR_,
                                step_tracing=step_tracing)
            elif verbose:
                print("\n> ADR will NOT be checked!")
                if np.nansum(self.ADR_y + self.ADR_x) != 0:
                    print("  However ADR fits provided and applied:")
                    print("  ADR_x_fit = ", self.ADR_x_fit)
                    print("  ADR_y_fit = ", self.ADR_y_fit)

            # Correct for Atmospheric Differential Refraction (ADR) if requested and not done before
            if ADR and ADR_repeat:
                self.weighted_I = np.zeros((self.n_wave, self.n_rows, self.n_cols))
                self.weight = np.zeros_like(self.weighted_I)
                self.ADR_correction(RSS, plot=plot, force_ADR=force_ADR, jump=jump,
                                    remove_spaxels_not_fully_covered=remove_spaxels_not_fully_covered)
                self.trace_peak(check_ADR=True, box_x=box_x, box_y=box_y, edgelow=edgelow, edgehigh=edgehigh,
                                step_tracing=step_tracing, adr_index_fit=adr_index_fit, g2d=g2d,
                                plot_tracing_maps=plot_tracing_maps, plot=plot, verbose=verbose)

            # Apply flux calibration
            self.apply_flux_calibration(flux_calibration=flux_calibration, flux_calibration_file=flux_calibration_file,
                                        verbose=verbose, path=path)

            if np.nanmedian(self.flux_calibration) != 0: fcal = True

            if fcal == False and verbose: print(
                "\n> This interpolated cube does not include an absolute flux calibration")

            # Get integrated maps (all waves and valid range), plots
            self.get_integrated_map(plot=plot, plot_spectra=plot_spectra, fcal=fcal,  # box_x=box_x, box_y=box_y,
                                    verbose=verbose, plot_centroid=True, g2d=g2d, log=log, gamma=gamma,
                                    nansum=False)  # Barr

            # Trim the cube if requested
            if trim_cube:
                self.trim_cube(half_size_for_centroid=half_size_for_centroid, box_x=box_x, box_y=box_y, ADR=ADR,
                               verbose=verbose, plot=plot,
                               remove_spaxels_not_fully_covered=remove_spaxels_not_fully_covered,
                               g2d=g2d, adr_index_fit=adr_index_fit, step_tracing=step_tracing,
                               plot_tracing_maps=plot_tracing_maps)  #### UPDATE THIS, now it is run automatically

        if read_fits_cube == False and verbose: print("\n> Interpolated cube done!")

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def apply_flux_calibration(self, flux_calibration=[], flux_calibration_file="", path="", verbose=True):
        """
        Function for applying the flux calibration to a cube

        Parameters
        ----------
        flux_calibration : Float List, optional
            It is a list of floats. The default is empty

        flux_calibration_file : String, optional
            The file name of the flux_calibration. The default is ""

        path : String, optional
            The directory of the folder the flux_calibration_file is in. The default is ""

        verbose : Boolean, optional
            Print results. The default is True.

        Returns
        -------
        None.

        """
        if flux_calibration_file != "":
            flux_calibration_file = full_path(flux_calibration_file, path)
            if verbose: print("\n> Flux calibration provided in file:\n ", flux_calibration_file)
            w_star, flux_calibration = read_table(flux_calibration_file, ["f", "f"])

        if len(flux_calibration) > 0:
            if verbose: print("\n> Applying the absolute flux calibration...")
            self.flux_calibration = flux_calibration
            # This should be in 1 line of step of loop, I couldn't get it # Yago HELP !!
            for y in range(self.n_rows):
                for x in range(self.n_cols):
                    self.data[:, y, x] = self.data[:, y, x] / self.flux_calibration / 1E16 / self.total_exptime

            self.history.append("- Applied flux calibration")
            if flux_calibration_file != "": self.history.append("  Using file " + flux_calibration_file)

        else:
            if verbose: print("\n\n> Absolute flux calibration not provided! Nothing done!")
        # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    def build_cube(self, RSS, jump=-1, warnings=False, verbose=True):
        """
        This function builds the cube.

        Parameters
        ----------
        RSS : File/Object created with KOALA_RSS
            This is the file that has the raw stacked spectra.
        jump : Integer, optional
            If a positive number partitions the wavelengths with step size jump, if -1 will not partition. The default is -1.
        warnings : Boolean, optional
            If True will show any problems that arose, else skipped. The default is False.
        verbose : Boolean, optional
            Print results. The default is True.

        Returns
        -------
        Float
            The weighted average intensity.

        """

        if verbose:
            print("\n  Smooth cube, (RA, DEC)_centre = ({}, {}) degree" \
                  .format(self.RA_centre_deg, self.DEC_centre_deg))
            print("  Size = {} columns (RA) x {} rows (DEC); {:.2f} x {:.2f} arcsec" \
                  .format(self.n_cols, self.n_rows, self.RA_segment, self.DEC_segment))

        if np.nansum(self.ADR_y + self.ADR_x) != 0:
            if verbose: print("  Building cube considering the ADR correction")
            self.adrcor = True
        sys.stdout.write("  Adding {} spectra...       ".format(self.n_spectra))
        sys.stdout.flush()
        output_every_few = np.sqrt(self.n_spectra) + 1
        next_output = -1
        for i in range(self.n_spectra):
            if verbose:
                if i > next_output:
                    sys.stdout.write("\b" * 6)
                    sys.stdout.write("{:5.2f}%".format(i * 100. / self.n_spectra))
                    sys.stdout.flush()
                    next_output = i + output_every_few
            offset_rows = (self.offset_DEC_arcsec[i] - self.yoffset_centre_arcsec) / self.pixel_size_arcsec
            offset_cols = (-self.offset_RA_arcsec[i] + self.xoffset_centre_arcsec) / self.pixel_size_arcsec
            corrected_intensity = RSS.intensity_corrected[i]
            # self.add_spectrum(corrected_intensity, offset_rows, offset_cols, warnings=warnings)
            self.add_spectrum_ADR(corrected_intensity, offset_rows, offset_cols, ADR_x=self.ADR_x, ADR_y=self.ADR_y,
                                  jump=jump, warnings=warnings)

        if verbose:
            sys.stdout.write("\b" * 6)
            sys.stdout.write("{:5.2f}%".format(100.0))
            sys.stdout.flush()
            print(" ")

        return self.weighted_I / self.weight

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def ADR_correction(self, RSS, plot=True, force_ADR=False, method="new", remove_spaxels_not_fully_covered=True,
                       jump=-1, warnings=False, verbose=True):
        """
        Corrects for Atmospheric Differential Refraction (ADR)

        Parameters
        ----------
        RSS : File/Object created with KOALA_RSS
            This is the file that has the raw stacked spectra.
        plot : Boolean, optional
            If True generates and shows the plots. The default is True.
        force_ADR : Boolean, optional
            If True will correct for ADR even considoring a small correction. The default is False.
        method : String, optional
            DESCRIPTION. The default is "new". ***
        remove_spaxels_not_fully_covered : Boolean, optional
            DESCRIPTION. The default is True. ***
        jump : Integer, optional
            If a positive number partitions the wavelengths with step size jump, if -1 will not partition. The default is -1.
        warnings : Boolean, optional
            If True will show any problems that arose, else skipped. The default is False.
        verbose : Boolean, optional
            Print results. The default is True.

        Returns
        -------
        None.

        """

        # Check if this is a self.combined cube or a self
        try:
            _x_ = np.nanmedian(self.combined_cube.data)
            if _x_ > 0:
                cubo = self.combined_cube
                # data_ = np.zeros_like(cubo.weighted_I)
                method = "old"
                # is_combined_cube=True
        except Exception:
            cubo = self

        # Check if ADR is needed (unless forced)...
        total_ADR = np.sqrt(cubo.ADR_x_max ** 2 + cubo.ADR_y_max ** 2)

        self.adrcor = True
        if total_ADR < cubo.pixel_size_arcsec * 0.1:  # Not needed if correction < 10 % pixel size
            if verbose:
                print("\n> Atmospheric Differential Refraction (ADR) correction is NOT needed.")
                print(
                    '  The computed max ADR value, {:.3f}",  is smaller than 10% the pixel size of {:.2f} arcsec'.format(
                        total_ADR, cubo.pixel_size_arcsec))
            self.adrcor = False
            if force_ADR:
                self.adrcor = True
                if verbose: print('  However we proceed to do the ADR correction as indicated: "force_ADR = True" ...')

        if self.adrcor:
            if verbose:
                print("\n> Correcting for Atmospheric Differential Refraction (ADR) using: \n")
                print("  ADR_x_fit = ", self.ADR_x_fit)
                print("  ADR_y_fit = ", self.ADR_y_fit)

                # Computing jump automatically
            if jump == -1:
                cubo_ADR_total = np.sqrt(cubo.ADR_x ** 2 + cubo.ADR_y ** 2)
                stop = 0
                i = 1
                while stop < 1:
                    if np.abs(cubo_ADR_total[i] - cubo_ADR_total[0]) > 0.012:
                        jump = i
                        if verbose: print('  Automatically found that with jump = ', jump,
                                          ' in lambda, the ADR offset is lower than 0.01"')
                        stop = 2
                    else:
                        i = i + 1
                    if i == len(cubo.wavelength) / 2.:
                        jump = -1
                        if verbose: print(
                            '  No value found for jump smaller than half the size of the wavelength! \n  Using jump = -1 (no jump) for ADR.')
                        stop = 2

            if method == "old":
                # data_ = np.zeros_like(cubo.weighted_I)
                if verbose: print("\n  Using OLD method (moving planes) ...")

                sys.stdout.flush()
                output_every_few = np.sqrt(cubo.n_wave) + 1
                next_output = -1

                # First create a CUBE without NaNs and a mask
                cube_shifted = copy.deepcopy(cubo.data) * 0.
                tmp = copy.deepcopy(cubo.data)
                mask = copy.deepcopy(tmp) * 0.
                mask[np.where(np.isnan(tmp) == False)] = 1  # Nans stay the same, when a good value = 1.
                tmp_nonan = np.nan_to_num(tmp, nan=np.nanmedian(tmp))  # cube without nans, replaced for median value

                # for l in range(cubo.n_wave):
                for l in range(0, self.n_wave, jump):

                    median_ADR_x = np.nanmedian(cubo.ADR_x[l:l + jump])
                    median_ADR_y = np.nanmedian(cubo.ADR_y[l:l + jump])

                    if l > next_output:
                        sys.stdout.write("\b" * 37)
                        sys.stdout.write(
                            "  Moving plane {:5} /{:5}... {:5.2f}%".format(l, cubo.n_wave, l * 100. / cubo.n_wave))
                        sys.stdout.flush()
                        next_output = l + output_every_few

                    # For applying shift the array MUST NOT HAVE ANY nans

                    # tmp=copy.deepcopy(cubo.data[l:l+jump,:,:])
                    # mask=copy.deepcopy(tmp)*0.
                    # mask[np.where(np.isnan(tmp))]=1 #make mask where Nans are
                    # kernel = Gaussian2DKernel(5)
                    # tmp_nonan = interpolate_replace_nans(tmp, kernel)
                    # need to see if there are still nans. This can happen in the padded parts of the grid
                    # where the kernel is not large enough to cover the regions with NaNs.
                    # if np.isnan(np.sum(tmp_nonan)):
                    #    tmp_nonan=np.nan_to_num(tmp_nonan)
                    # tmp_shift=shift(tmp_nonan,[-median_ADR_y/cubo.pixel_size_arcsec,-median_ADR_x/cubo.pixel_size_arcsec],cval=np.nan)
                    # mask_shift=shift(mask,[-median_ADR_y/cubo.pixel_size_arcsec,-median_ADR_x/cubo.pixel_size_arcsec],cval=np.nan)
                    # tmp_shift[mask_shift > 0.5]=np.nan
                    # cubo.data[l,:,:]=copy.deepcopy(tmp_shift)

                    # tmp_shift=shift(tmp,[0,-median_ADR_y/cubo.pixel_size_arcsec,-median_ADR_x/cubo.pixel_size_arcsec],cval=np.nan)
                    # cubo.data[l:l+jump,:,:]=copy.deepcopy(tmp_shift)

                    cube_shifted[l:l + jump, :, :] = shift(tmp_nonan[l:l + jump, :, :],
                                                           [0, -median_ADR_y / cubo.pixel_size_arcsec,
                                                            -median_ADR_x / cubo.pixel_size_arcsec], cval=np.nan)

                    # cubo.data[l:l+jump,:,:]=shift(cubo.data[l:l+jump,:,:],[0,-median_ADR_y/cubo.pixel_size_arcsec, -median_ADR_x/cubo.pixel_size_arcsec],cval=np.nan)

                    # print(l,tmp.shape,2*self.ADR_y[l],2*self.ADR_x[l],np.sum(tmp_nonan),np.sum(tmp),np.sum(tmp_shift))
                    # for y in range(cubo.n_rows):
                    #      for x in range(cubo.n_cols):
                    #          mal = 0
                    #          if np.int(np.round(x+median_ADR_x/cubo.pixel_size_arcsec)) < cubo.n_cols :
                    #              if np.int(np.round(y+median_ADR_y/cubo.pixel_size_arcsec)) < cubo.n_rows :
                    #                 # print self.data.shape,x,"->",np.int(np.round(x+self.ADR_x[i]/self.pixel_size_arcsec)),"     ",y,"->",np.int(np.round(y+self.ADR_y[i]/self.pixel_size_arcsec))
                    #                  data_[l:l+jump,y,x]=cubo.data[l:l+jump, np.int(np.round(y+median_ADR_y/cubo.pixel_size_arcsec )), np.int(np.round(x+median_ADR_x/cubo.pixel_size_arcsec)) ]
                    #              else: mal = 1
                    #          else: mal = 1
                    #          if mal == 1:
                    #              if l == 0 and warnings == True : print("Warning: ", self.data.shape,x,"->",np.int(np.round(x+median_ADR_x/self.pixel_size_arcsec)),"     ",y,"->",np.int(np.round(y+median_ADR_y/self.pixel_size_arcsec))," bad data !")

                    # tmp=copy.deepcopy(cubo.data[l,:,:])
                    # mask=copy.deepcopy(tmp)*0.
                    # mask[np.where(np.isnan(tmp))]=1 #make mask where Nans are
                    # kernel = Gaussian2DKernel(5)
                    # tmp_nonan = interpolate_replace_nans(tmp, kernel)
                    # #need to see if there are still nans. This can happen in the padded parts of the grid
                    # #where the kernel is not large enough to cover the regions with NaNs.
                    # if np.isnan(np.sum(tmp_nonan)):
                    #     tmp_nonan=np.nan_to_num(tmp_nonan)
                    # tmp_shift=shift(tmp_nonan,[-cubo.ADR_y[l]/cubo.pixel_size_arcsec,-cubo.ADR_x[l]/cubo.pixel_size_arcsec],cval=np.nan)
                    # mask_shift=shift(mask,[-cubo.ADR_y[l]/cubo.pixel_size_arcsec,-cubo.ADR_x[l]/cubo.pixel_size_arcsec],cval=np.nan)
                    # tmp_shift[mask_shift > 0.5]=np.nan
                    # cubo.data[l,:,:]=copy.deepcopy(tmp_shift)

                    # #print(l,tmp.shape,2*self.ADR_y[l],2*self.ADR_x[l],np.sum(tmp_nonan),np.sum(tmp),np.sum(tmp_shift))
                    # for y in range(cubo.n_rows):
                    #      for x in range(cubo.n_cols):
                    #          mal = 0
                    #          if np.int(np.round(x+cubo.ADR_x[l]/cubo.pixel_size_arcsec)) < cubo.n_cols :
                    #              if np.int(np.round(y+cubo.ADR_y[l]/cubo.pixel_size_arcsec)) < cubo.n_rows :
                    #                 # print self.data.shape,x,"->",np.int(np.round(x+self.ADR_x[i]/self.pixel_size_arcsec)),"     ",y,"->",np.int(np.round(y+self.ADR_y[i]/self.pixel_size_arcsec))
                    #                  data_[l,y,x]=cubo.data[l, np.int(np.round(y+cubo.ADR_y[l]/cubo.pixel_size_arcsec )), np.int(np.round(x+cubo.ADR_x[l]/cubo.pixel_size_arcsec)) ]
                    #              else: mal = 1
                    #          else: mal = 1
                    #          if mal == 1:
                    #              if l == 0 and warnings == True : print("Warning: ", self.data.shape,x,"->",np.int(np.round(x+self.ADR_x[l]/self.pixel_size_arcsec)),"     ",y,"->",np.int(np.round(y+self.ADR_y[l]/self.pixel_size_arcsec))," bad data !")

                if verbose: print(" ")
                comparison_cube = copy.deepcopy(cubo)
                comparison_cube.data = (cube_shifted - cubo.data) * mask
                comparison_cube.description = "Comparing original and shifted cubes"
                vmin = -np.nanmax([np.abs(np.nanmin(comparison_cube.data)), np.abs(np.nanmax(comparison_cube.data))])
                vmax = np.nanmax([np.abs(np.nanmin(comparison_cube.data)), np.abs(np.nanmax(comparison_cube.data))])

                comparison_cube.get_integrated_map(plot=plot, plot_spectra=False, verbose=False, plot_centroid=False,
                                                   cmap="seismic", log=False, vmin=vmin, vmax=vmax)

                cubo.data = copy.deepcopy(cube_shifted) * mask

            # New procedure 2nd April 2020
            else:
                if verbose:
                    print("\n  Using the NEW method (building the cube including the ADR offsets)...")
                    print("  Creating new cube considering the median value each ", jump, " lambdas...")
                self.adrcor = True
                self.data = self.build_cube(jump=jump, RSS=RSS)

                # # Check flux calibration and apply to the new cube
                # if np.nanmedian(self.flux_calibration) == 0:
                #     if verbose: print("\n\n> No absolute flux calibration included.")
                # else:
                #     if verbose: print("\n\n> Applying the absolute flux calibration...")
                #     self.apply_flux_calibration(self.flux_calibration, verbose=verbose)

            # Now remove spaxels with not full wavelength if requested
            if remove_spaxels_not_fully_covered == True:

                if verbose: print(
                    "\n> Removing spaxels that are not fully covered in wavelength in the valid range...")  # Barr
                _mask_ = cubo.integrated_map / cubo.integrated_map
                for w in range(cubo.n_wave):
                    cubo.data[w] = cubo.data[w] * _mask_

        else:
            if verbose: print(" NOTHING APPLIED !!!")

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def add_spectrum_ADR(self, intensity, offset_rows, offset_cols, ADR_x=[0], ADR_y=[0],
                         jump=-1, warnings=False):
        """
        Add one single spectrum to the datacube

        Parameters
        ----------
        intensity: np.array(float)
          Spectrum.
        offset_rows, offset_cols: float
          Offset with respect to the image centre, in pixels.
        kernel_FWHM_pixels: float
          FWHM of the interpolating kernel, in pixels

        """
        if jump == -1: jump = self.n_wave

        if np.nanmedian(ADR_x) == 0 and np.nanmedian(ADR_y) == 0: jump = self.n_wave

        for l in range(0, self.n_wave, jump):

            median_ADR_x = np.nanmedian(self.ADR_x[l:l + jump])
            median_ADR_y = np.nanmedian(self.ADR_y[l:l + jump])

            kernel_centre_x = .5 * self.n_cols + offset_cols - median_ADR_x / self.pixel_size_arcsec  # *2.

            x_min = int(kernel_centre_x - self.kernel_size_pixels)
            x_max = int(kernel_centre_x + self.kernel_size_pixels) + 1
            n_points_x = x_max - x_min
            x = np.linspace(x_min - kernel_centre_x, x_max - kernel_centre_x, n_points_x) / self.kernel_size_pixels
            x[0] = -1.
            x[-1] = 1.
            weight_x = np.diff((3. * x - x ** 3 + 2.) / 4)

            kernel_centre_y = .5 * self.n_rows + offset_rows - median_ADR_y / self.pixel_size_arcsec  # *2.

            y_min = int(kernel_centre_y - self.kernel_size_pixels)
            y_max = int(kernel_centre_y + self.kernel_size_pixels) + 1
            n_points_y = y_max - y_min
            y = np.linspace(y_min - kernel_centre_y, y_max - kernel_centre_y, n_points_y) / self.kernel_size_pixels
            y[0] = -1.
            y[-1] = 1.
            weight_y = np.diff((3. * y - y ** 3 + 2.) / 4)

            if x_min < 0 or x_max > self.n_cols + 1 or y_min < 0 or y_max > self.n_rows + 1:
                if warnings:
                    print("**** WARNING **** : Spectra outside field of view:", x_min, kernel_centre_x, x_max)
                    print("                                                 :", y_min, kernel_centre_y, y_max)
            else:
                bad_wavelengths = np.argwhere(np.isnan(intensity))
                intensity[bad_wavelengths] = 0.
                ones = np.ones_like(intensity)
                ones[bad_wavelengths] = 0.
                self.weighted_I[l:l + jump, y_min:y_max - 1, x_min:x_max - 1] += intensity[l:l + jump, np.newaxis,
                                                                                 np.newaxis] * weight_y[np.newaxis, :,
                                                                                               np.newaxis] * weight_x[
                                                                                                             np.newaxis,
                                                                                                             np.newaxis,
                                                                                                             :]
                self.weight[l:l + jump, y_min:y_max - 1, x_min:x_max - 1] += ones[l:l + jump, np.newaxis,
                                                                             np.newaxis] * weight_y[np.newaxis, :,
                                                                                           np.newaxis] * weight_x[
                                                                                                         np.newaxis,
                                                                                                         np.newaxis, :]

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def get_peaks(self, box_x=[0, -1], box_y=[0, -1], verbose=True, plot=False):
        """
        This tasks quickly finds the values for the peak in x and y


            This is the part of the old get_peaks that we still need,
            but perhaps it can be deprecated if defining self.x_peaks etc using new task "centroid of cube"

        Parameters
        ----------
        box_x : Integer List, optional
            When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [0,-1].
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [0,-1].
        verbose : Boolean, optional
            Print results. The default is True.
        plot : Boolean, optional
            If True generates and shows the plots. The default is False.

        """

        if np.nanmedian(box_x + box_y) != -0.5:
            x0 = box_x[0]
            x1 = box_x[1]
            y0 = box_y[0]
            y1 = box_y[1]
            x = np.arange(x1 - x0)
            y = np.arange(y1 - y0)
            if verbose: print("\n> Getting the peaks in box at [{}:{} , {}:{}] ...".format(x0, x1, y0, y1))
            tmp = copy.deepcopy(self.data[:, y0:y1, x0:x1])
        else:
            if verbose: print("\n> Getting the peaks considering all the spaxels...")

            tmp = copy.deepcopy(self.data)
            x = np.arange(self.n_cols)
            y = np.arange(self.n_rows)

        tmp_img = np.nanmedian(tmp, axis=0)
        sort = np.sort(tmp_img.ravel())
        low_ind = np.where(tmp_img < sort[int(.9 * len(sort))])
        for i in np.arange(len(low_ind[0])):
            tmp[:, low_ind[0][i], low_ind[1][i]] = np.nan

        weight = np.nan_to_num(tmp)

        mean_image = np.nanmean(weight, axis=0)
        mean_image /= np.nanmean(mean_image)
        weight *= mean_image[np.newaxis, :, :]
        xw = x[np.newaxis, np.newaxis, :] * weight
        yw = y[np.newaxis, :, np.newaxis] * weight
        w = np.nansum(weight, axis=(1, 2))
        self.x_peaks = np.nansum(xw, axis=(1, 2)) / w  # Vector with the x-peak at each wavelength
        self.y_peaks = np.nansum(yw, axis=(1, 2)) / w  # Vector with the y-peak at each wavelength
        self.x_peak_median = np.nanmedian(self.x_peaks)  # Median value of the x-peak vector
        self.y_peak_median = np.nanmedian(self.y_peaks)  # Median value of the y-peak vector
        # self.x_peak_median_index = np.nanargmin(np.abs(self.x_peaks-self.x_peak_median)) # Index closest to the median value of the x-peak vector
        # self.y_peak_median_index = np.nanargmin(np.abs(self.y_peaks-self.y_peak_median)) # Index closest to the median value of the y-peak vector

        if np.nanmedian(box_x + box_y) != -0.5:  # Move peaks from position in box to position in cube
            self.x_peaks = self.x_peaks + box_x[0]
            self.y_peaks = self.y_peaks + box_y[0]
            self.x_peak_median = self.x_peak_median + box_x[0]
            self.y_peak_median = self.y_peak_median + box_y[0]

        self.offset_from_center_x_arcsec_tracing = (
                                                               self.x_peak_median - self.spaxel_RA0) * self.pixel_size_arcsec  # Offset from center using INTEGRATED map
        self.offset_from_center_y_arcsec_tracing = (
                                                               self.y_peak_median - self.spaxel_DEC0) * self.pixel_size_arcsec  # Offset from center using INTEGRATED map

        if plot:
            ptitle = "Peaks in x axis, median value found in x = " + np.str(np.round(self.x_peak_median, 2))
            plot_plot(self.wavelength, self.x_peaks, psym="+", markersize=2, ylabel="spaxel", ptitle=ptitle,
                      percentile_min=0.5, percentile_max=99.5, hlines=[self.x_peak_median])
            ptitle = "Peaks in y axis, median value found in y = " + np.str(np.round(self.y_peak_median, 2))
            plot_plot(self.wavelength, self.y_peaks, psym="+", markersize=2, ylabel="spaxel", ptitle=ptitle,
                      percentile_min=0.5, percentile_max=99.5, hlines=[self.y_peak_median])
            if verbose: print(" ")

        if verbose:  print(
            '  The peak of the emission is found in  [{:.2f}, {:.2f}] , Offset from center :   {:.2f}" , {:.2f}"'.format(
                self.x_peak_median, self.y_peak_median, self.offset_from_center_x_arcsec_tracing,
                self.offset_from_center_y_arcsec_tracing))

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def trace_peak(self, box_x=[0, -1], box_y=[0, -1], edgelow=-1, edgehigh=-1,
                   adr_index_fit=2, step_tracing=100, g2d=True, plot_tracing_maps=[],
                   plot=False, log=True, gamma=0., check_ADR=False, verbose=True):
        """
        ***

        Parameters
        ----------
        box_x : Integer List, optional
            When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [0,-1].
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [0,-1].
        edgelow : Integer, optional
            This is the lowest value in the wavelength range in terms of pixels. The default is -1.
        edgehigh : Integer, optional
            This is the highest value in the wavelength range in terms of pixels, (maximum wavelength - edgehigh). The default is -1.
        adr_index_fit : Integer, optional
            This is the fitted polynomial with highest degree n. The default is 2.
        step_tracing : Integer, optional
            DESCRIPTION. The default is 100. ***
        g2d : Boolean, optional
            If True uses a 2D gaussian, else doesn't. The default is True.
        plot_tracing_maps : List, optional
            DESCRIPTION. The default is []. ***
        plot : Boolean, optional
            If True generates and shows the plots. The default is False.
        log : Boolean, optional
            If true the map is plotted on a log scale. The default is True.
        gamma : Float, optional
            The value for power log. The default is 0..
        check_ADR : Boolean, optional
            DESCRIPTION. The default is False. ***
        verbose : Boolean, optional
            Print results. The default is True.

        Returns
        -------
        None.

        """

        if verbose: print("\n> Tracing peaks and checking ADR...")

        if self.grating == "580V":
            if edgelow == -1: edgelow = 150
            if edgehigh == -1: edgehigh = 10
        else:
            if edgelow == -1: edgelow = 10
            if edgehigh == -1: edgehigh = 10

        x0 = box_x[0]
        x1 = box_x[1]
        y0 = box_y[0]
        y1 = box_y[1]

        if check_ADR:
            plot_residua = False
        else:
            plot_residua = True

        ADR_x_fit, ADR_y_fit, ADR_x_max, ADR_y_max, ADR_total, x_peaks, y_peaks = centroid_of_cube(self, x0, x1, y0, y1,
                                                                                                   edgelow=edgelow,
                                                                                                   edgehigh=edgehigh,
                                                                                                   step_tracing=step_tracing,
                                                                                                   g2d=g2d,
                                                                                                   plot_tracing_maps=plot_tracing_maps,
                                                                                                   adr_index_fit=adr_index_fit,
                                                                                                   plot=plot, log=log,
                                                                                                   gamma=gamma,
                                                                                                   plot_residua=plot_residua,
                                                                                                   verbose=verbose)
        pp = np.poly1d(ADR_x_fit)
        ADR_x = pp(self.wavelength)
        pp = np.poly1d(ADR_y_fit)
        ADR_y = pp(self.wavelength)

        # self.get_peaks(box_x=box_x, box_y=box_y, verbose=verbose)  ---> Using old routine, but now we have the values from centroid!
        self.x_peaks = x_peaks  # Vector with the x-peak at each wavelength
        self.y_peaks = y_peaks  # Vector with the y-peak at each wavelength
        self.x_peak_median = np.nanmedian(self.x_peaks)  # Median value of the x-peak vector
        self.y_peak_median = np.nanmedian(self.y_peaks)  # Median value of the y-peak vector
        # self.x_peak_median_index = np.nanargmin(np.abs(self.x_peaks-self.x_peak_median)) # Index closest to the median value of the x-peak vector
        # self.y_peak_median_index = np.nanargmin(np.abs(self.y_peaks-self.y_peak_median)) # Index closest to the median value of the y-peak vector
        self.offset_from_center_x_arcsec_tracing = (
                                                               self.x_peak_median - self.spaxel_RA0) * self.pixel_size_arcsec  # Offset from center using CENTROID
        self.offset_from_center_y_arcsec_tracing = (
                                                               self.y_peak_median - self.spaxel_DEC0) * self.pixel_size_arcsec  # Offset from center using CENTROID

        self.ADR_x = ADR_x
        self.ADR_y = ADR_y
        self.ADR_x_max = ADR_x_max
        self.ADR_y_max = ADR_y_max
        self.ADR_total = ADR_total

        if ADR_total > self.pixel_size_arcsec * 0.1:
            if verbose: print(
                '\n  The combined ADR, {:.2f}", is larger than 10% of the pixel size! Applying this ADR correction is needed !!'.format(
                    ADR_total))
        elif verbose:
            print(
                '\n  The combined ADR, {:.2f}", is smaller than 10% of the pixel size! Applying this ADR correction is NOT needed'.format(
                    ADR_total))

        if check_ADR == False:
            self.ADR_x_fit = ADR_x_fit
            self.ADR_y_fit = ADR_y_fit

            if verbose:
                print("\n> Results of the ADR fit (to be applied in a next step if requested):\n")
                print("  ADR_x_fit = ", ADR_x_fit)
                print("  ADR_y_fit = ", ADR_y_fit)
        elif verbose:
            print("\n> We are only checking the ADR correction, data will NOT be corrected !")

        # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    def box_for_centroid(self, half_size_for_centroid=6, verbose=True, plot=False,
                         plot_map=True, log=True, gamma=0.):
        """
        Creates a box around centroid.

        Parameters
        ----------
        half_size_for_centroid : Integer, optional
            This is half the length/width of the box. The default is 6.
        verbose : Boolean, optional
            Print results. The default is True.
        plot : Boolean, optional
            If True generates and shows the plots. The default is False.
        plot_map : Boolean, optional
            If True will plot the maps. The default is True.
        log : Boolean, optional
            If true the map is plotted on a log scale. The default is True.
        gamma : Float, optional
            The value for power log. The default is 0..

        Returns
        -------
        box_x_centroid : Integer List
            This holds the maximum and minimum values for the box in the x direction.
        box_y_centroid : Integer List
            This holds the maximum and minimum values for the box in the y direction.

        """

        self.get_peaks(plot=plot, verbose=verbose)

        max_x = np.int(np.round(self.x_peak_median, 0))
        max_y = np.int(np.round(self.y_peak_median, 0))

        if verbose: print("\n> Defining a box centered in [ {} , {} ] and width +-{} spaxels:".format(max_x, max_y,
                                                                                                      half_size_for_centroid))
        box_x_centroid = [max_x - half_size_for_centroid, max_x + half_size_for_centroid]
        box_y_centroid = [max_y - half_size_for_centroid, max_y + half_size_for_centroid]
        if verbose: print(
            "  box_x =[ {}, {} ],  box_y =[ {}, {} ]".format(box_x_centroid[0], box_x_centroid[1], box_y_centroid[0],
                                                             box_y_centroid[1]))

        if plot_map: self.plot_map(plot_box=True, box_x=box_x_centroid, box_y=box_y_centroid, log=log, gamma=gamma,
                                   spaxel=[max_x, max_y], plot_centroid=True)

        return box_x_centroid, box_y_centroid

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def get_integrated_map(self, min_wave=0, max_wave=0, nansum=True,
                           vmin=1E-30, vmax=1E30, fcal=False, log=True, gamma=0., cmap="fuego",
                           box_x=[0, -1], box_y=[0, -1], g2d=False, plot_centroid=True,
                           trace_peaks=False, adr_index_fit=2, edgelow=-1, edgehigh=-1, step_tracing=100,
                           plot=False, plot_spectra=False, plot_tracing_maps=[], verbose=True):  ### CHECK
        """
        Compute Integrated map and plot if requested

        Parameters
        ----------
        min_wave : Integer, optional
            The minimum wavelength passed through the mask. The default is 0.
        max_wave : Integer, optional
            The maximum wavelength passed through the mask. The default is 0.
        nansum : Boolean, optional
            If True will sum the number of NaNs in the columns and rows in the intergrated map. The default is True.
        vmin : FLoat, optional
            DESCRIPTION. The default is 1E-30. ***
        vmax : Float, optional
            DESCRIPTION. The default is 1E30. ***
        fcal : Boolean, optional
            If fcal=True, cube.flux_calibration is used. The default is False.
        log : Boolean, optional
            If true the map is plotted on a log scale. The default is True.
        gamma : Float, optional
            The value for power log. The default is 0..
        cmap : String, optional
            This is the colour of the map. The default is "fuego".
        box_x : Integer List, optional
            When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [0,-1].
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [0,-1].
        g2d : Boolean, optional
            If True uses a 2D gaussian, else doesn't. The default is False.
        plot_centroid : Boolean, optional
            If True will plot the centroid. The default is True.
        trace_peaks : Boolean, optional
            DESCRIPTION. The default is False. ***
        adr_index_fit : Integer, optional
            This is the fitted polynomial with highest degree n. The default is 2.
        edgelow : Integer, optional
            This is the lowest value in the wavelength range in terms of pixels. The default is -1.
        edgehigh : Integer, optional
            This is the highest value in the wavelength range in terms of pixels, (maximum wavelength - edgehigh). The default is -1.
        step_tracing : Integer, optional
            DESCRIPTION. The default is 100. ***
        plot : Boolean, optional
            If True generates and shows the plots. The default is False.
        plot_spectra : Boolean, optional
            If True will plot the spectra. The default is False.
        plot_tracing_maps : List, optional
            If True will plot the tracing maps. The default is [].
        verbose : Boolean, optional
            Print results. The default is True.

        Returns
        -------
        None.

        """

        # Integrated map between good wavelengths

        if min_wave == 0: min_wave = self.valid_wave_min
        if max_wave == 0: max_wave = self.valid_wave_max

        if nansum:
            self.integrated_map_all = np.nansum(self.data, axis=0)
            self.integrated_map = np.nansum(
                self.data[np.searchsorted(self.wavelength, min_wave):np.searchsorted(self.wavelength, max_wave)],
                axis=0)
        else:
            self.integrated_map_all = np.sum(self.data, axis=0)
            self.integrated_map = np.sum(
                self.data[np.searchsorted(self.wavelength, min_wave):np.searchsorted(self.wavelength, max_wave)],
                axis=0)

            # Search for peak of emission in integrated map and compute offsets from centre
        self.max_y, self.max_x = np.unravel_index(np.nanargmax(self.integrated_map), self.integrated_map.shape)
        self.offset_from_center_x_arcsec_integrated = (
                                                                  self.max_x - self.spaxel_RA0) * self.pixel_size_arcsec  # Offset from center using INTEGRATED map
        self.offset_from_center_y_arcsec_integrated = (
                                                                  self.max_y - self.spaxel_DEC0) * self.pixel_size_arcsec  # Offset from center using INTEGRATED map

        for row in range(len(self.integrated_map[:])):  # Put nans instead of 0
            v_ = self.integrated_map[row]
            self.integrated_map[row] = [np.nan if x == 0 else x for x in v_]

        if plot_spectra:
            self.plot_spectrum_cube(-1, -1, fcal=fcal)
            print(" ")
            self.plot_spectrum_cube(self.max_x, self.max_y, fcal=fcal)

        if trace_peaks:
            if verbose: print("\n> Tracing peaks using all data...")
            self.trace_peak(edgelow=edgelow, edgehigh=edgehigh,  # box_x=box_x,box_y=box_y,
                            adr_index_fit=adr_index_fit, step_tracing=step_tracing, g2d=g2d,
                            plot_tracing_maps=plot_tracing_maps, plot=False, check_ADR=False, verbose=False)

        if verbose:
            print("\n> Created integrated map between {:5.2f} and {:5.2f} considering nansum = {:}".format(min_wave,
                                                                                                           max_wave,
                                                                                                           nansum))
            print("  The cube has a size of {} x {} spaxels = [ 0 ... {} ] x [ 0 ... {} ]".format(self.n_cols,
                                                                                                  self.n_rows,
                                                                                                  self.n_cols - 1,
                                                                                                  self.n_rows - 1))
            print("  The peak of the emission in integrated image is in spaxel [", self.max_x, ",", self.max_y, "]")
            print("  The peak of the emission tracing all wavelengths is in position [",
                  np.round(self.x_peak_median, 2), ",", np.round(self.y_peak_median, 2), "]")

        if plot:
            self.plot_map(log=log, gamma=gamma, spaxel=[self.max_x, self.max_y],
                          spaxel2=[self.x_peak_median, self.y_peak_median], fcal=fcal,
                          box_x=box_x, box_y=box_y, plot_centroid=plot_centroid, g2d=g2d, cmap=cmap, vmin=vmin,
                          vmax=vmax)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_spectrum_cube(self, x=-1, y=-1, lmin=0, lmax=0, fmin=1E-30, fmax=1E30,
                           fcal=False, fig_size=10., fig_size_y=0., save_file="", title="", z=0.,
                           median=False, plot=True, verbose=True):  # Must add: elines, alines...
        """
        Plot spectrum of a particular spaxel or list of spaxels.

        Parameters
        ----------
        x, y:
            coordenates of spaxel to show spectrum.
            if x is -1, gets all the cube
        fcal:
            Use flux calibration, default fcal=False.\n
            If fcal=True, cube.flux_calibration is used.
        save_file:
            (Optional) Save plot in file "file.extension"
        fig_size:
            Size of the figure (in x-axis), default: fig_size=10

        Example
        -------
        >>> cube.plot_spectrum_cube(20, 20, fcal=True)
        """
        if np.isscalar(x):
            if x == -1:
                if median:
                    if verbose: print("\n> Computing the median spectrum of the cube...")
                    spectrum = np.nanmedian(np.nanmedian(self.data, axis=1), axis=1)
                    if title == "": title = "Median spectrum in {}".format(self.description)
                else:
                    if verbose: print("\n> Computing the integrated spectrum of the cube...")
                    spectrum = np.nansum(np.nansum(self.data, axis=1), axis=1)
                    if title == "": title = "Integrated spectrum in {}".format(self.description)
            else:
                if verbose: print("> Spectrum of spaxel [", x, ",", y, "] :")
                spectrum = self.data[:, y, x]
                if title == "": title = "Spaxel [{},{}] in {}".format(x, y, self.description)
        else:
            list_of_spectra = []
            if verbose:
                if median:
                    print("\n> Median spectrum of selected spaxels...")
                else:
                    print("\n> Integrating spectrum of selected spaxels...")
                print("  Adding spaxel  1  = [", x[0], ",", y[0], "]")
            list_of_spectra.append(self.data[:, y[0], x[0]])
            for i in range(len(y) - 1):
                list_of_spectra.append(self.data[:, y[i + 1], x[i + 1]])
                if verbose: print("  Adding spaxel ", i + 2, " = [", x[i + 1], ",", y[i + 1], "]")
            n_spaxels = len(x)

            if median:
                spectrum = np.nanmedian(list_of_spectra, axis=0)
                if title == "": title = "Median spectrum adding {} spaxels in {}".format(n_spaxels, self.description)
            else:
                spectrum = np.nansum(list_of_spectra, axis=0)
                if title == "": title = "Integrated spectrum adding {} spaxels in {}".format(n_spaxels,
                                                                                             self.description)

        if fcal == False:
            ylabel = "Flux [relative units]"
        else:
            spectrum = spectrum * 1E16
            ylabel = "Flux [ 10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]"

        if plot:  # TODO: This should be done with PLOT PLOT
            # Set limits
            if fmin == 1E-30:
                fmin = np.nanmin(spectrum)
            if fmax == 1E30:
                fmax = np.nanmax(spectrum)
            if lmin == 0:
                lmin = self.wavelength[0]
            if lmax == 0:
                lmax = self.wavelength[-1]

            if fig_size_y == 0.: fig_size_y = fig_size / 3.
            plt.figure(figsize=(fig_size, fig_size_y))
            plt.plot(self.wavelength, spectrum)
            plt.title(title)
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            plt.ylabel(ylabel)
            plt.minorticks_on()
            plt.legend(frameon=False)

            plt.axvline(x=self.valid_wave_min, color="k", linestyle="--", alpha=0.3)
            plt.axvline(x=self.valid_wave_max, color="k", linestyle="--", alpha=0.3)

            try:
                plt.ylim(fmin, fmax)
                plt.xlim(lmin, lmax)
            except Exception:
                print("  WARNING! Something failed getting the limits of the plot...")
                print("  Values given: lmin = ", lmin, " , lmax = ", lmax)
                print("                fmin = ", fmin, " , fmax = ", fmax)

            # Identify lines
            if z != 0:
                # Emission lines
                elines = [3727.00, 3868.75, 3967.46, 3889.05, 4026., 4068.10, 4101.2, 4340.47, 4363.21, 4471.48,
                          4658.10, 4686., 4711.37, 4740.16, 4861.33, 4958.91, 5006.84, 5197.82, 6300.30, 6312.10,
                          6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7281.35,
                          7320, 7330]
                # elines=[3727.00, 3868.75, 3967.46, 3889.05, 4026., 4068.10, 4101.2, 4340.47, 4363.21, 4471.48, 4658.10, 4861.33, 4958.91, 5006.84, 5197.82, 6300.30, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7320, 7330 ]
                for i in elines:
                    plt.plot([i * (1 + z), i * (1 + z)], [fmin, fmax], "g:", alpha=0.95)
                # Absorption lines
                alines = [3934.777, 3969.588, 4308, 5175]  # ,4305.61, 5176.7]   # POX 4
                # alines=[3934.777,3969.588,4308,5170]    #,4305.61, 5176.7]
                for i in alines:
                    plt.plot([i * (1 + z), i * (1 + z)], [fmin, fmax], "r:", alpha=0.95)

            # Show or save file
            if save_file == "":
                plt.show()
            else:
                plt.savefig(save_file)
            plt.close()

        return spectrum

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def spectrum_of_box(self, box_x=[], box_y=[], center=[10, 10], width=3,
                        log=True, gamma=0.,
                        plot=True, verbose=True, median=False):
        """
        Given a box (or a center with a width size, all in spaxels),
        this task provides the integrated or median spectrum and the list of spaxels.

        Parameters
        ----------
        box_x : Integer List, optional
            When creating a box to show/trim/alignment these are the x coordinates in spaxels of the box. The default is [].
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y coordinates in spaxels of the box. The default is [].
        center : Integer List, optional
            The centre of the box. The default is [10,10].
        width : Integer, optional
            The width of the box. The default is 3.
        log : Boolean, optional
            If true the map is plotted on a log scale. The default is True.
        gamma : Float, optional
            The value for power log. The default is 0..
        plot : Boolean, optional
            If True generates and shows the plots. The default is True.
        verbose : Boolean, optional
            Print results. The default is True.
        median : Boolean, optional
            If True prints out the median values. The default is False.

        Returns
        -------
        spectrum_box : Plot
            This is the plot of the spectrum box.
        spaxel_list : List of Interger Lists
            The cooridinates of the box corners.

        """

        if len(box_x) == 0:
            # Center values must be integer and not float
            center = [np.int(np.round(center[0], 0)), np.int(np.round(center[1], 0))]
            if verbose:
                if median:
                    print(
                        "\n> Median spectrum of box with center [ {} , {} ] and width = {} spaxels (0 is valid)".format(
                            center[0], center[1], width))
                else:
                    print(
                        "\n> Integrating spectrum of box with center [ {} , {} ] and width = {} spaxels (0 is valid)".format(
                            center[0], center[1], width))
            if width % 2 == 0:
                if verbose: print("  Width is an EVEN number, the given center will not be in the center")
            hw = np.int((width - 1) / 2)
            box_x = [center[0] - hw, center[0] + hw + 1]
            box_y = [center[1] - hw, center[1] + hw + 1]
            description = "Spectrum of box, center " + np.str(center[0]) + " x " + np.str(
                center[1]) + ", width " + np.str(width)
        else:
            if verbose:
                if median:
                    print("\n> Median spectrum of box with [ {} : {} ] x  [ {} : {} ] (0 is valid)".format(box_x[0],
                                                                                                           box_x[1],
                                                                                                           box_y[0],
                                                                                                           box_y[1]))
                else:
                    print(
                        "\n> Integrating spectrum of box with [ {} : {} ] x  [ {} : {} ] (0 is valid)".format(box_x[0],
                                                                                                              box_x[1],
                                                                                                              box_y[0],
                                                                                                              box_y[1]))
            description = "Spectrum of box " + np.str(box_x[0]) + ":" + np.str(box_x[1]) + " x " + np.str(
                box_y[0]) + ":" + np.str(box_y[1])
            center = 0
            width = 0

        if plot:
            self.plot_map(spaxel=center, box_x=box_x, box_y=box_y, gamma=gamma, log=log, description=description,
                          verbose=verbose)

        list_x, list_y = [], []

        for i in range(box_x[0], box_x[1]):
            for j in range(box_y[0], box_y[1]):
                list_x.append(i)
                list_y.append(j)

        spectrum_box = self.plot_spectrum_cube(list_x, list_y, verbose=verbose, plot=plot, median=median)

        spaxel_list = []
        for i in range(len(list_x)):
            spaxel_list.append([list_x[i], list_y[i]])

        return spectrum_box, spaxel_list

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def spectrum_offcenter(self, offset_x=5, offset_y=5, distance=5, width=3, pa="", peak=[],
                           median=False, plot=True, verbose=True):  # self
        """
        This tasks calculates spaxels coordenates offcenter of the peak,
        and calls task spectrum_of_box to get the integrated or median spectrum of the region

        Example: spec,spaxel_list = cube.spectrum_offcenter(distance=10, width=5, pa=cube.pa)

        Parameters
        ----------
        offset_x : Integer, optional
            DESCRIPTION. The default is 5. ***
        offset_y : Integer, optional
            DESCRIPTION. The default is 5. ***
        distance : Integer, optional
            DESCRIPTION. The default is 5. ***
        width : Integer, optional
            DESCRIPTION. The default is 3. ***
        pa : String, optional
            This is the position angle. The default is "".
        peak : List, optional
            DESCRIPTION. The default is []. ***
        median : Boolean, optional
            If True will print out the median specrum of the box. The default is False.
        plot : Boolean, optional
            If True generates and shows the plots. The default is True.
        verbose : Boolean, optional
            Print results. The default is True.

        Returns
        -------
        spectrum_box : Plot
            This is the plot of the spectrum box.
        spaxel_list : List of Interger Lists
            The cooridinates of the box corners.

        """
        if peak != []:
            x0 = peak[0]
            y0 = peak[1]
        else:
            x0 = self.x_peak_median
            y0 = self.y_peak_median

        if pa != "":
            offset_x = distance * COS(pa)  # self.PA
            offset_y = distance * SIN(pa)

        center = [x0 + offset_x, y0 + offset_y]

        if verbose:
            if median:
                print("\n> Calculating median spectrum for a box of width {} and center [ {} , {} ]".format(width,
                                                                                                            np.round(
                                                                                                                center[
                                                                                                                    0],
                                                                                                                2),
                                                                                                            np.round(
                                                                                                                center[
                                                                                                                    1],
                                                                                                                2)))
            else:
                print("\n> Calculating integrated spectrum for a box of width {} and center [ {} , {} ]".format(width,
                                                                                                                np.round(
                                                                                                                    center[
                                                                                                                        0],
                                                                                                                    2),
                                                                                                                np.round(
                                                                                                                    center[
                                                                                                                        1],
                                                                                                                    2)))
            if peak != []:
                print("  This is an offset of [ {} , {} ] from given position at [ {} , {} ]".format(
                    np.round(offset_x, 2), np.round(offset_y, 2), np.round(x0, 2), np.round(y0, 2)))
            else:
                print(
                    "  This is an offset of [ {} , {} ] from peak emission at [ {} , {} ]".format(np.round(offset_x, 2),
                                                                                                  np.round(offset_y, 2),
                                                                                                  np.round(x0, 2),
                                                                                                  np.round(y0, 2)))

            if pa != "":
                print("  This was obtained using a distance of {} spaxels and a position angle of {}".format(distance,
                                                                                                             np.round(
                                                                                                                 pa,
                                                                                                                 2)))

        spectrum_box, spaxel_list = self.spectrum_of_box(box_x=[], center=center, width=width, median=median, plot=plot,
                                                         verbose=verbose)

        return spectrum_box, spaxel_list

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_weight(self, log=False, gamma=0., vmin=1E-30, vmax=1E30,
                    cmap="gist_gray", fig_size=10,
                    save_file="",  # ADR = True,
                    description="", contours=True, clabel=False, verbose=True,
                    spaxel=0, spaxel2=0, spaxel3=0,
                    box_x=[0, -1], box_y=[0, -1],
                    circle=[0, 0, 0], circle2=[0, 0, 0], circle3=[0, 0, 0],
                    plot_centre=True, plot_spaxel=False, plot_spaxel_grid=True,
                    label_axes_fontsize=15, axes_fontsize=14, c_fontsize=12, title_fontsize=16,
                    fraction=0.0457, pad=0.02, colorbar_ticksize=14, colorbar_fontsize=15, barlabel=""):
        """
        Plot weight map.

        Example
        ----------
        >>> cube1s.plot_weight()

        Parameters
        ----------
        log : Boolean, optional
            If true the map is plotted on a log scale. The default is False.
        gamma : Float, optional
            The value for power log. The default is 0..
        vmin : Float, optional
            DESCRIPTION. The default is 1E-30. ***
        vmax : Float, optional
            DESCRIPTION. The default is 1E30. ***
        cmap : String, optional
            This is the colour of the map. The default is "gist_gray".
        fig : Integer, optional
            DESCRIPTION. The default is 10. ***
        save_file : String, optional
            Save plot in file "file.extension". The default is "".
        #ADR : Boolean, optional
            If True will correct for ADR (Atmospheric Differential Refraction). The default is True.
        description : String, optional
            This is the description of the cube. The default is "".
        contours : Boolean, optional
            DESCRIPTION. The default is True. ***
        clabel : Boolean, optional
            DESCRIPTION. The default is False. ***
        verbose : Boolean, optional
            Print results. The default is True.
        spaxel : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel2 : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel3 : Integer, optional
            DESCRIPTION. The default is 0. ***
        box_x : Integer List, optional
            When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [0,-1].
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [0,-1].
        circle : Integer List, optional
            DESCRIPTION. The default is [0,0,0].
        circle2 : Integer List, optional
            DESCRIPTION. The default is [0,0,0].
        circle3 : Integer List, optional
            DESCRIPTION. The default is [0,0,0].
        plot_centre : Boolean, optional
            If True plots the centre. The default is True.
        plot_spaxel : Boolean, optional
            If True will plot the spaxel. The default is False.
        plot_spaxel_grid : Boolean, optional
            If True plots the spaxel grid. The default is True.
        label_axes_fontsize : Integer, optional
            This is the size of the axes labels. The default is 15.
        axes_fontsize : Integer, optional
            This is the size of the font on the axes. The default is 14.
        c_fontsize : Integer, optional
            DESCRIPTION. The default is 12. ***
        title_fontsize : Integer, optional
            This is the size of the font for the title. The default is 16.
        fraction : Float, optional
            DESCRIPTION. The default is 0.0457. ***
        pad : Float, optional
            DESCRIPTION. The default is 0.02. ***
        colorbar_ticksize : Integer, optional
            This is the size of the colourbars ticks. The default is 14.
        colorbar_fontsize : Integer, optional
            This is the fontsize of the text for the colourbar. The default is 15.
        barlabel : String, optional
            This is text for the colourbar. The default is "".

        Returns
        -------
        None.

        """

        interpolated_map = np.mean(self.weight, axis=0)

        if np.nansum(interpolated_map) == 0:
            print("\n> This cube does not have weights. There is nothing to plot!")
        else:
            if description == "": description = self.description + " - Weight map"
            self.plot_map(interpolated_map, fig_size=fig_size, cmap=cmap, save_file=save_file,
                          description=description, weight=True,
                          contours=contours, clabel=clabel, verbose=verbose,
                          spaxel=spaxel, spaxel2=spaxel2, spaxel3=spaxel3,
                          box_x=box_x, box_y=box_y,
                          circle=circle, circle2=circle2, circle3=circle3,
                          plot_centre=plot_centre, plot_spaxel=plot_spaxel, plot_spaxel_grid=plot_spaxel_grid,
                          log=log, gamma=gamma, vmin=vmin, vmax=vmax,
                          label_axes_fontsize=label_axes_fontsize, axes_fontsize=axes_fontsize, c_fontsize=c_fontsize,
                          title_fontsize=title_fontsize,
                          fraction=fraction, pad=pad, colorbar_ticksize=colorbar_ticksize,
                          colorbar_fontsize=colorbar_fontsize, barlabel=barlabel)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def create_map(self, line, w2=0., gaussian_fit=False, gf=False,
                   lowlow=50, lowhigh=10, highlow=10, highhigh=50,
                   show_spaxels=[], verbose=True, description=""):
        """
        Runs task "create_map"

        Parameters
        ----------
        line : TYPE
            DESCRIPTION.
        w2 : Float, optional
            DESCRIPTION. The default is 0..
        gaussian_fit : Boolean, optional
            DESCRIPTION. The default is False. ***
        gf : Boolean, optional
            DESCRIPTION. The default is False. ***
        lowlow : Integer, optional
            DESCRIPTION. The default is 50. ***
        lowhigh : Integer, optional
            DESCRIPTION. The default is 10. ***
        highlow : Integer, optional
            DESCRIPTION. The default is 10. ***
        highhigh : Integer, optional
            DESCRIPTION. The default is 50. ***
        show_spaxels : List, optional
            DESCRIPTION. The default is []. ***
        verbose : Boolean, optional
			Print results. The default is True.
        description : String, optional
            This is the description of the cube. The default is "".

        Returns
        -------
        mapa : Map
            DESCRIPTION. ***

        """

        mapa = create_map(cube=self, line=line, w2=w2, gaussian_fit=gaussian_fit, gf=gf,
                          lowlow=lowlow, lowhigh=lowhigh, highlow=highlow, highhigh=highhigh,
                          show_spaxels=show_spaxels, verbose=verbose, description=description)
        return mapa

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_wavelength(self, wavelength, w2=0.,
                        # norm=colors.PowerNorm(gamma=1./4.),
                        log=False, gamma=0., vmin=1E-30, vmax=1E30,
                        cmap=fuego_color_map, fig_size=10, fcal=False,
                        save_file="", description="", contours=True, clabel=False, verbose=True,
                        spaxel=0, spaxel2=0, spaxel3=0,
                        box_x=[0, -1], box_y=[0, -1],
                        circle=[0, 0, 0], circle2=[0, 0, 0], circle3=[0, 0, 0],
                        plot_centre=True, plot_spaxel=False, plot_spaxel_grid=True,
                        label_axes_fontsize=15, axes_fontsize=14, c_fontsize=12, title_fontsize=16,
                        fraction=0.0457, pad=0.02, colorbar_ticksize=14, colorbar_fontsize=15, barlabel=""):
        """
        Plot map at a particular wavelength or in a wavelength range

        Parameters
        ----------
        wavelength: Float
          wavelength to be mapped.
        w2 : Float, optional
            DESCRIPTION. The default is 0.. ***
        #norm : TYPE, optional ***
            DESCRIPTION. The default is colors.PowerNorm(gamma=1./4.). ***
        log : Boolean, optional
            If true the map is plotted on a log scale. The default is False.
        gamma : Float, optional
            DESCRIPTION. The default is 0.. ***
        vmin : Float, optional
            DESCRIPTION. The default is 1E-30. ***
        vmax : Float, optional
            DESCRIPTION. The default is 1E30. ***
        cmap:
            Color map used.
            Velocities: cmap="seismic". The default is fuego_color_map.
        fig_size : Integer, optional
            This is the size of the figure. The default is 10.
        fcal : Boolean, optional
            If fcal=True, cube.flux_calibration is used. The default is False. ***
        save_file : String, optional
            Save plot in file "file.extension". The default is "".
        description : String, optional
            This is the description of the cube. The default is "".
        contours : Boolean, optional
            DESCRIPTION. The default is True. ***
        clabel : Boolean, optional
            DESCRIPTION. The default is False. ***
        verbose : Boolean, optional
			Print results. The default is True.
        spaxel : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel2 : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel3 : Integer, optional
            DESCRIPTION. The default is 0. ***
        box_x : Integer List, optional
            When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [0,-1]. ***
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [0,-1]. ***
        circle : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. ***
        circle2 : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. ***
        circle3 : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. ***
        plot_centre : Boolean, optional
            DESCRIPTION. The default is True. ***
        plot_spaxel : Boolean, optional
            If True will plot the spaxel. The default is False. ***
        plot_spaxel_grid : Boolean, optional
            If True plots the spaxel grid. The default is True. ***
        label_axes_fontsize : Integer, optional
            This is the size of the axes labels. The default is 15.
        axes_fontsize : Integer, optional
            This is the size of the font on the axes. The default is 14.
        c_fontsize : Integer, optional
            DESCRIPTION. The default is 12. ***
        title_fontsize : Integer, optional
            This is the size of the titles text. The default is 16.
        fraction : Float, optional
            DESCRIPTION. The default is 0.0457. ***
        pad : Float, optional
            DESCRIPTION. The default is 0.02. ***
        colorbar_ticksize : Integer, optional
            This is the size of the colourbars ticks. The default is 14.
        colorbar_fontsize : Integer, optional
            This is the fontsize of the text for the colourbar. The default is 15.
        barlabel : String, optional
            This is text for the colourbar. The default is "".

        Returns
        -------
        None.

        """

        # mapa, description_ = self.make_map(wavelength, w2=w2)
        description_, mapa, w1_, w2_ = self.create_map(line=wavelength, w2=w2)

        if description == "": description = description_

        self.plot_map(mapa=mapa,
                      cmap=cmap, fig_size=fig_size, fcal=fcal,
                      save_file=save_file, description=description, contours=contours, clabel=clabel, verbose=verbose,
                      spaxel=spaxel, spaxel2=spaxel2, spaxel3=spaxel3,
                      box_x=box_x, box_y=box_y,
                      circle=circle, circle2=circle2, circle3=circle3,
                      plot_centre=plot_centre, plot_spaxel=plot_spaxel, plot_spaxel_grid=plot_spaxel_grid,
                      log=log, gamma=gamma, vmin=vmin, vmax=vmax,
                      label_axes_fontsize=label_axes_fontsize, axes_fontsize=axes_fontsize, c_fontsize=c_fontsize,
                      title_fontsize=title_fontsize,
                      fraction=fraction, pad=pad, colorbar_ticksize=colorbar_ticksize,
                      colorbar_fontsize=colorbar_fontsize, barlabel=barlabel)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_map(self, mapa="", log=False, gamma=0., vmin=1E-30, vmax=1E30, fcal=False,
                 # norm=colors.Normalize(),
                 trimmed=False,
                 cmap="fuego", weight=False, velocity=False, fwhm=False, ew=False, ratio=False,
                 contours=True, clabel=False,
                 line=0,
                 spaxel=0, spaxel2=0, spaxel3=0,
                 box_x=[0, -1], box_y=[0, -1], plot_centroid=False, g2d=True, half_size_for_centroid=0,
                 circle=[0, 0, 0], circle2=[0, 0, 0], circle3=[0, 0, 0],
                 plot_box=False, plot_centre=True, plot_spaxel=False, plot_spaxel_grid=True, alpha_grid=0.1,
                 plot_spaxel_list=[], color_spaxel_list="blue", alpha_spaxel_list=0.4,
                 label_axes_fontsize=15, axes_fontsize=14, c_fontsize=12, title_fontsize=16,
                 fraction=0.0457, pad=0.02, colorbar_ticksize=14, colorbar_fontsize=15, barlabel="",
                 description="", fig_size=10, save_file="", verbose=True):
        """
        Show a given map.

        map: np.array(float)
          Map to be plotted. If not given, it plots the integrated map.
        norm:
          Normalization scale, default is lineal scale.    NOW USE log and gamma
          Lineal scale: norm=colors.Normalize().
          Log scale:    norm=colors.LogNorm()
          Power law:    norm=colors.PowerNorm(gamma=1./4.)
        cmap: (default cmap="fuego").
            Color map used.
            Weight: cmap = "gist_gray"
            Velocities: cmap="seismic".
            Try also "inferno", "CMRmap", "gnuplot2", "gist_rainbow", "Spectral"
        spaxel,spaxel2,spaxel3:
            [x,y] positions of spaxels to show with a green circle, blue square and red triangle

        Parameters
        ----------
        mapa : String, optional
            DESCRIPTION. The default is "". ***
        log : Boolean, optional
            If true the map is plotted on a log scale. The default is False.
        gamma : Float, optional
            DESCRIPTION. The default is 0.. ***
        vmin : Float, optional
            DESCRIPTION. The default is 1E-30. ***
        vmax : Float, optional
            DESCRIPTION. The default is 1E30. ***
        fcal : Boolean, optional
            If fcal=True, cube.flux_calibration is used. The default is False. ***
        #norm : TYPE, optional ***
            DESCRIPTION. The default is colors.Normalize(). ***
        trimmed : Boolean, optional
            DESCRIPTION. The default is False. ***
        cmap : String, optional
            This is the colour of the map. The default is "fuego".
        weight : Boolean, optional
            DESCRIPTION. The default is False. ***
        velocity : Boolean, optional
            DESCRIPTION. The default is False. ***
        fwhm : Boolean, optional
            DESCRIPTION. The default is False. ***
        ew : Boolean, optional
            DESCRIPTION. The default is False. ***
        ratio : Boolean, optional
            DESCRIPTION. The default is False. ***
        contours : Boolean, optional
            DESCRIPTION. The default is True. ***
        clabel : Boolean, optional
            DESCRIPTION. The default is False. ***
        line : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel2 : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel3 : Integer, optional
            DESCRIPTION. The default is 0. ***
        box_x : Integer List, optional
             When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [0,-1].
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [0,-1].
        plot_centroid : Boolean, optional
            If True will plot the centroid. The default is False.
        g2d : Boolean, optional
            If True uses a 2D gaussian, else doesn't. The default is True.
        half_size_for_centroid : Integer, optional
            This is half the length/width of the box. The default is 0.
        circle : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. ***
        circle2 : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. ***
        circle3 : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. ***
        plot_box : Boolean, optional
            DESCRIPTION. The default is False. ***
        plot_centre : Boolean, optional
            DESCRIPTION. The default is True. ***
        plot_spaxel : Boolean, optional
            If True will plot the spaxel. The default is False. ***
        plot_spaxel_grid : Boolean, optional
            If True plots the spaxel grid. The default is True.
        alpha_grid : Float, optional
            DESCRIPTION. The default is 0.1. ***
        plot_spaxel_list : List, optional
            DESCRIPTION. The default is []. ***
        color_spaxel_list : String, optional
            DESCRIPTION. The default is "blue". ***
        alpha_spaxel_list : Float, optional
            DESCRIPTION. The default is 0.4. ***
        label_axes_fontsize : Integer, optional
            This is the size of the axes labels. The default is 15.
        axes_fontsize : Integer, optional
            This is the size of the font on the axes. The default is 14.
        c_fontsize : Integer, optional
            DESCRIPTION. The default is 12. ***
        title_fontsize : Integer, optional
            This is the size of the font for the title. The default is 16.
        fraction : Float, optional
            DESCRIPTION. The default is 0.0457. ***
        pad : Float, optional
            DESCRIPTION. The default is 0.02. ***
        colorbar_ticksize : Integer, optional
            This is the size of the colourbars ticks. The default is 14.
        colorbar_fontsize : Integer, optional
            This is the fontsize of the text for the colourbar. The default is 15.
        barlabel : String, optional
            This is text for the colourbar. The default is "".
        description : String, optional
            This is the description of the cube. The default is "".
        fig_size : Integer, optional
            This is the size of the figure. The default is 10.
        save_file : String, optional
            Save plot in file "file.extension". The default is "".
        verbose : Boolean, optional
			Print results. The default is True.

        Returns
        -------
        None.

        """

        mapa_ = mapa
        try:
            if type(mapa[0]) == str:  # Maps created by PyKOALA have [description, map, l1, l2 ...]
                mapa_ = mapa[1]
                if description == "": mapa = mapa[0]
        except Exception:
            if mapa == "":
                if len(self.integrated_map) == 0:  self.get_integrated_map(verbose=verbose)

                mapa_ = self.integrated_map
                if description == "": description = self.description + " - Integrated Map"

        if description == "":
            description = self.description

        # Trim the map if requested
        if np.nanmedian(box_x + box_y) != -0.5 and plot_box == False:
            trimmed = True
            mapa = copy.deepcopy(mapa_[box_y[0]:box_y[1], box_x[0]:box_x[1]])
        else:
            mapa = mapa_

        if trimmed:
            extent1 = 0
            # extent2 = (box_x[1]-box_x[0])*self.pixel_size_arcsec
            extent2 = len(mapa[0]) * self.pixel_size_arcsec
            extent3 = 0
            # extent4 = (box_y[1]-box_y[0])*self.pixel_size_arcsec
            extent4 = len(mapa) * self.pixel_size_arcsec
            alpha_grid = 0
            plot_spaxel_grid = False
            plot_centre = False
            fig_size = fig_size * 0.5
        else:
            extent1 = (0.5 - self.n_cols / 2) * self.pixel_size_arcsec
            extent2 = (0.5 + self.n_cols / 2) * self.pixel_size_arcsec
            extent3 = (0.5 - self.n_rows / 2) * self.pixel_size_arcsec
            extent4 = (0.5 + self.n_rows / 2) * self.pixel_size_arcsec

        if verbose: print("\n> Plotting map '" + description.replace("\n ", "") + "' :")
        if verbose and trimmed: print(
            "  Trimmed in x = [ {:} , {:} ]  ,  y = [ {:} , {:} ] ".format(box_x[0], box_x[1], box_y[0], box_y[1]))

        # Check fcal
        if fcal == False and np.nanmedian(self.flux_calibration) != 0: fcal = True

        if velocity and cmap == "fuego": cmap = "seismic"
        if fwhm and cmap == "fuego": cmap = "Spectral"
        if ew and cmap == "fuego": cmap = "CMRmap_r"
        if ratio and cmap == "fuego": cmap = "gnuplot2"

        if velocity or fwhm or ew or ratio or weight:
            fcal = False
            if vmin == 1E-30: vmin = np.nanpercentile(mapa, 5)
            if vmax == 1E30: vmax = np.nanpercentile(mapa, 95)

        # We want squared pixels for plotting
        try:
            aspect_ratio = self.combined_cube.n_cols / self.combined_cube.n_rows * 1.
        except Exception:
            aspect_ratio = self.n_cols / self.n_rows * 1.

        fig, ax = plt.subplots(figsize=(fig_size / aspect_ratio, fig_size))

        if log:
            norm = colors.LogNorm()
        else:
            norm = colors.Normalize()

        if gamma != 0: norm = colors.PowerNorm(gamma=gamma)  # Default = 0.25 = 1/4

        if vmin == 1E-30: vmin = np.nanmin(mapa)
        if vmin <= 0 and log == True:
            if verbose: print("  vmin is negative but log = True, using vmin = np.nanmin(np.abs())")
            vmin = np.nanmin(np.abs(mapa)) + 1E-30

        if vmax == 1E30: vmax = np.nanmax(mapa)

        cax = ax.imshow(mapa, origin='lower', interpolation='none', norm=norm, cmap=cmap,
                        extent=(extent1, extent2, extent3, extent4))
        cax.set_clim(vmin=vmin)
        cax.set_clim(vmax=vmax)

        if contours:
            CS = plt.contour(mapa, extent=(extent1, extent2, extent3, extent4))
            if clabel: plt.clabel(CS, inline=1, fontsize=c_fontsize)

        ax.set_title(description, fontsize=title_fontsize)
        plt.tick_params(labelsize=axes_fontsize)
        plt.xlabel('$\Delta$ RA [arcsec]', fontsize=label_axes_fontsize)
        plt.ylabel('$\Delta$ DEC [arcsec]', fontsize=label_axes_fontsize)
        plt.legend(loc='upper right', frameon=False)
        plt.minorticks_on()
        plt.grid(which='both', color="white", alpha=alpha_grid)
        # plt.axis('square')

        # IMPORTANT:
        #             If drawing INTEGER SPAXELS, use -ox, -oy
        #             If not, just use -self.spaxel_RA0, -self.spaxel_DEC0 or -ox+0.5, -oy+0.5
        # if np.nanmedian(box_x+box_y) != -0.5 and plot_box == False:
        if trimmed:
            ox = 0
            oy = 0
            spaxel = 0
            spaxel2 = 0
        else:
            ox = self.spaxel_RA0 + 0.5
            oy = self.spaxel_DEC0 + 0.5

        if verbose:
            if fcal:
                print("  Color scale range : [ {:.2e} , {:.2e}]".format(vmin, vmax))
            else:
                print("  Color scale range : [ {} , {}]".format(vmin, vmax))

        if plot_centre:
            if verbose: print("  - The center of the cube is in position [", self.spaxel_RA0, ",", self.spaxel_DEC0,
                              "]")
            if self.n_cols % 2 == 0:
                extra_x = 0
            else:
                extra_x = 0.5 * self.pixel_size_arcsec
            if self.n_rows % 2 == 0:
                extra_y = 0
            else:
                extra_y = 0.5 * self.pixel_size_arcsec

            plt.plot([0. + extra_x - self.pixel_size_arcsec * 0.00], [0. + extra_y - self.pixel_size_arcsec * 0.02],
                     "+", ms=14, color="black", mew=4)
            plt.plot([0 + extra_x], [0 + extra_y], "+", ms=10, color="white", mew=2)

        if plot_spaxel_grid:
            for i in range(self.n_cols):
                spaxel_position = [i - ox, 0 - oy]
                if i % 2 == 0:
                    color = "white"
                else:
                    color = "black"
                cuadrado = plt.Rectangle(
                    (spaxel_position[0] * self.pixel_size_arcsec, (spaxel_position[1] + 0) * self.pixel_size_arcsec),
                    self.pixel_size_arcsec, self.pixel_size_arcsec, color=color, linewidth=0, fill=True)
                ax.add_patch(cuadrado)

            for i in range(self.n_rows):
                spaxel_position = [0 - ox, i - oy]
                if i % 2 == 0:
                    color = "white"
                else:
                    color = "black"
                cuadrado = plt.Rectangle(
                    (spaxel_position[0] * self.pixel_size_arcsec, (spaxel_position[1] + 0) * self.pixel_size_arcsec),
                    self.pixel_size_arcsec, self.pixel_size_arcsec, color=color, linewidth=0, fill=True)
                ax.add_patch(cuadrado)

        if plot_spaxel:
            spaxel_position = [5 - ox, 5 - oy]
            cuadrado = plt.Rectangle(
                (spaxel_position[0] * self.pixel_size_arcsec, (spaxel_position[1] + 0) * self.pixel_size_arcsec),
                self.pixel_size_arcsec, self.pixel_size_arcsec, color=color, linewidth=0, fill=True, alpha=0.8)
            ax.add_patch(cuadrado)
            # vertices_x =[spaxel_position[0]*self.pixel_size_arcsec,spaxel_position[0]*self.pixel_size_arcsec,(spaxel_position[0]+1)*self.pixel_size_arcsec,(spaxel_position[0]+1)*self.pixel_size_arcsec,spaxel_position[0]*self.pixel_size_arcsec]
            # vertices_y =[spaxel_position[1]*self.pixel_size_arcsec,(1+spaxel_position[1])*self.pixel_size_arcsec,(1+spaxel_position[1])*self.pixel_size_arcsec,spaxel_position[1]*self.pixel_size_arcsec,spaxel_position[1]*self.pixel_size_arcsec]
            # plt.plot(vertices_x,vertices_y, color=color, linewidth=1., alpha=1)
            plt.plot([(spaxel_position[0] + 0.5) * self.pixel_size_arcsec],
                     [(spaxel_position[1] + .5) * self.pixel_size_arcsec], 'o', color="black", ms=1, alpha=1)

        if len(plot_spaxel_list) > 0:
            for _spaxel_ in plot_spaxel_list:  # If they are INTEGERS we identify the SPAXEL putting the cross IN THE CENTRE of the SPAXEL
                if isinstance(_spaxel_[0], int) and isinstance(_spaxel_[1], int):
                    extra = 0.5
                else:
                    extra = 0.
                plt.plot((_spaxel_[0] - ox + extra) * self.pixel_size_arcsec,
                         (_spaxel_[1] - oy + extra) * self.pixel_size_arcsec, color="black", marker="+", ms=15, mew=2)

        if np.nanmedian(circle) != 0:  # The center of the circle is given with decimals in spaxels
            # The radius is in arcsec, but for plt.Circle all in arcsec
            offset_from_center_x_arcsec = (circle[0] - self.spaxel_RA0) * self.pixel_size_arcsec
            offset_from_center_y_arcsec = (circle[1] - self.spaxel_DEC0) * self.pixel_size_arcsec
            circle_p = plt.Circle((offset_from_center_x_arcsec, offset_from_center_y_arcsec),
                                  circle[2], color='b', linewidth=3, fill=False)
            ax.add_patch(circle_p)
            if verbose: print(
                '  - Blue  circle:   [{:.2f}, {:.2f}] , Offset from center :   {:.2f}" , {:.2f}", radius = {:.2f}"'.format(
                    circle[0], circle[1], offset_from_center_x_arcsec, offset_from_center_y_arcsec, circle[2]))

        if np.nanmedian(circle2) != 0:
            offset_from_center_x_arcsec = (circle2[0] - self.spaxel_RA0) * self.pixel_size_arcsec
            offset_from_center_y_arcsec = (circle2[1] - self.spaxel_DEC0) * self.pixel_size_arcsec
            circle_p = plt.Circle((offset_from_center_x_arcsec, offset_from_center_y_arcsec),
                                  circle2[2], color='w', linewidth=4, fill=False, alpha=0.3)
            ax.add_patch(circle_p)

        if np.nanmedian(circle3) != 0:
            offset_from_center_x_arcsec = (circle3[0] - self.spaxel_RA0) * self.pixel_size_arcsec
            offset_from_center_y_arcsec = (circle3[1] - self.spaxel_DEC0) * self.pixel_size_arcsec
            circle_p = plt.Circle((offset_from_center_x_arcsec, offset_from_center_y_arcsec),
                                  circle3[2], color='w', linewidth=4, fill=False, alpha=0.3)
            ax.add_patch(circle_p)

        if spaxel != 0:  # SPAXEL
            if isinstance(spaxel[0], int) and isinstance(spaxel[1], int):
                extra = 0.5  # If they are INTEGERS we identify the SPAXEL putting the cross IN THE CENTRE of the SPAXEL
            else:
                extra = 0.

            offset_from_center_x_arcsec = (spaxel[0] - self.spaxel_RA0 + extra) * self.pixel_size_arcsec
            offset_from_center_y_arcsec = (spaxel[1] - self.spaxel_DEC0 + extra) * self.pixel_size_arcsec
            if verbose: print(
                '  - Blue  square:   {}          , Offset from center :   {:.2f}" , {:.2f}"'.format(np.round(spaxel, 2),
                                                                                                    offset_from_center_x_arcsec,
                                                                                                    offset_from_center_y_arcsec))
            cuadrado = plt.Rectangle(((spaxel[0] - ox) * self.pixel_size_arcsec,
                                      (spaxel[1] - oy) * self.pixel_size_arcsec),
                                     self.pixel_size_arcsec, self.pixel_size_arcsec, color="blue", linewidth=0,
                                     fill=True, alpha=1)
            ax.add_patch(cuadrado)
            # plt.plot((spaxel[0]-ox)*self.pixel_size_arcsec,(spaxel[0]-ox)*self.pixel_size_arcsec, color="blue", marker="s", ms=15, mew=2)

        if spaxel2 != 0:  #
            offset_from_center_x_arcsec = (spaxel2[0] - self.spaxel_RA0) * self.pixel_size_arcsec
            offset_from_center_y_arcsec = (spaxel2[1] - self.spaxel_DEC0) * self.pixel_size_arcsec
            if verbose: print(
                '  - Green circle:   {}    , Offset from center :   {:.2f}" , {:.2f}"'.format(np.round(spaxel2, 2),
                                                                                              offset_from_center_x_arcsec,
                                                                                              offset_from_center_y_arcsec))
            plt.plot([offset_from_center_x_arcsec], [offset_from_center_y_arcsec], 'o', color="green", ms=7)

        if spaxel3 != 0:  # SPAXEL
            offset_from_center_x_arcsec = (spaxel3[0] - self.spaxel_RA0) * self.pixel_size_arcsec
            offset_from_center_y_arcsec = (spaxel3[1] - self.spaxel_DEC0) * self.pixel_size_arcsec
            if verbose: print(
                '  - Red   square:   {}     , Offset from center :   {:.2f}" , {:.2f}"'.format(np.round(spaxel3, 2),
                                                                                               offset_from_center_x_arcsec,
                                                                                               offset_from_center_y_arcsec))
            cuadrado = plt.Rectangle(((spaxel3[0] - ox) * self.pixel_size_arcsec,
                                      (spaxel3[1] - oy) * self.pixel_size_arcsec),
                                     self.pixel_size_arcsec, self.pixel_size_arcsec, color="red", linewidth=0,
                                     fill=True, alpha=1)
            ax.add_patch(cuadrado)

        if plot_centroid:
            # If box defined, compute centroid there
            if np.nanmedian(box_x + box_y) != -0.5 and plot_box:
                # if trimmed:
                _mapa_ = mapa[box_y[0]:box_y[1], box_x[0]:box_x[1]]
            else:
                _mapa_ = mapa

            if g2d:
                xc, yc = centroid_2dg(_mapa_)
            else:
                xc, yc = centroid_com(_mapa_)

            # if box values not given, both box_x[0] and box_y[0] are 0
            offset_from_center_x_arcsec = (xc + box_x[0] - self.spaxel_RA0) * self.pixel_size_arcsec
            offset_from_center_y_arcsec = (yc + box_y[0] - self.spaxel_DEC0) * self.pixel_size_arcsec
            # print(xc,yc)
            # print(offset_from_center_x_arcsec,offset_from_center_y_arcsec)

            # if np.nanmedian(box_x+box_y) != -0.5 and plot_box == False:
            plt.plot((xc + box_x[0] - ox) * self.pixel_size_arcsec, (yc + box_y[0] - oy) * self.pixel_size_arcsec,
                     color="black", marker="+", ms=15, mew=2)

            if verbose:
                if trimmed:
                    if line != 0:
                        print('  Centroid at {} A found in spaxel [ {:.2f} , {:.2f} ]  =  [ {:.2f}" , {:.2f}" ]'.format(
                            line, xc, yc, xc * self.pixel_size_arcsec, yc * self.pixel_size_arcsec))
                    else:
                        print(
                            '  - Centroid:       [{:.2f} {:.2f}]    , Offset from center :   {:.2f}" , {:.2f}"'.format(
                                xc + box_x[0], yc + box_y[0], offset_from_center_x_arcsec, offset_from_center_y_arcsec))
                else:
                    if line != 0:
                        print('  Centroid at {} A found in spaxel [ {:.2f} , {:.2f} ]  =  [ {:.2f}" , {:.2f}" ]'.format(
                            line, xc, yc, xc * self.pixel_size_arcsec, yc * self.pixel_size_arcsec))
                    else:
                        print(
                            '  - Centroid (box): [{:.2f} {:.2f}]    , Offset from center :   {:.2f}" , {:.2f}"'.format(
                                xc + box_x[0], yc + box_y[0], offset_from_center_x_arcsec, offset_from_center_y_arcsec))

        if np.nanmedian(box_x + box_y) != -0.5 and plot_box:  # Plot box
            box_x = [box_x[0] - ox, box_x[1] - ox]
            box_y = [box_y[0] - oy, box_y[1] - oy]

            vertices_x = [box_x[0] * self.pixel_size_arcsec, box_x[0] * self.pixel_size_arcsec,
                          box_x[1] * self.pixel_size_arcsec, box_x[1] * self.pixel_size_arcsec,
                          box_x[0] * self.pixel_size_arcsec]
            vertices_y = [box_y[0] * self.pixel_size_arcsec, box_y[1] * self.pixel_size_arcsec,
                          box_y[1] * self.pixel_size_arcsec, box_y[0] * self.pixel_size_arcsec,
                          box_y[0] * self.pixel_size_arcsec]
            plt.plot(vertices_x, vertices_y, "-b", linewidth=2., alpha=0.6)

        cbar = fig.colorbar(cax, fraction=fraction, pad=pad)
        cbar.ax.tick_params(labelsize=colorbar_ticksize)

        if barlabel == "":
            if velocity:
                barlabel = str("Velocity [ km s$^{-1}$ ]")
            elif fwhm:
                barlabel = str("FWHM [ km s$^{-1}$ ]")
            elif ew:
                barlabel = str("EW [ $\mathrm{\AA}$ ]")
            else:
                if fcal:
                    barlabel = str("Integrated Flux [ erg s$^{-1}$ cm$^{-2}$ ]")
                else:
                    barlabel = str("Integrated Flux [ Arbitrary units ]")
        cbar.set_label(barlabel, rotation=270, labelpad=20, fontsize=colorbar_fontsize)
        #        cbar.ax.set_yticklabels(['< -1', '0', '> 1'])# vertically oriented colorbar

        if save_file == "":
            plt.show()
        else:
            print("  Plot saved to file ", save_file)
            plt.savefig(save_file)
        plt.close()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def mask_cube(self, min_wave=0, max_wave=0, include_partial_spectra=False,
                  cmap="binary_r", plot=False, verbose=False):
        """
        ***

        Parameters
        ----------
        min_wave : Integer, optional
            The minimum wavelength passed through the mask. The default is 0.
        max_wave : Integer, optional
            The maximum wavelength passed through the mask. The default is 0.
        include_partial_spectra : Boolean, optional
            DESCRIPTION. The default is False. ***
        cmap : String, optional
            This is the colour of the map. The default is "binary_r".
        plot : Boolean, optional
            If True generates and shows the plots. The default is False.
        verbose : Boolean, optional
			Print results. The default is False.

        Returns
        -------
        None.

        """

        if min_wave == 0: min_wave = self.valid_wave_min
        if max_wave == 0: max_wave = self.valid_wave_max

        if include_partial_spectra:  # IT DOES NOT WORK
            if verbose: print("\n> Creating cube mask considering ALL spaxels with some spectrum...")
            self.integrated_map = np.nansum(
                self.data[np.searchsorted(self.wavelength, min_wave):np.searchsorted(self.wavelength, max_wave)],
                axis=0)
        else:
            if verbose: print(
                "\n> Creating cube mask considering ONLY those spaxels with valid full spectrum in range {:.2f} - {:.2f} ...".format(
                    min_wave, max_wave))
            # Get integrated map but ONLY consering spaxels for which all wavelengths are good (i.e. NOT using nanmen)
            self.integrated_map = np.sum(
                self.data[np.searchsorted(self.wavelength, min_wave):np.searchsorted(self.wavelength, max_wave)],
                axis=0)

            # Create a mask with the same structura full of 1
        self.mask = np.ones_like(self.integrated_map)
        for y in range(self.n_rows):
            for x in range(self.n_cols):
                if np.isnan(self.integrated_map[y][x]):
                    self.mask[y][x] = np.nan  # If integrated map is nan, the mask is nan
        # Plot mask
        if plot:
            description = self.description + " - mask of good spaxels"
            self.plot_map(mapa=self.mask, cmap=cmap, description=description)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def growth_curve_between(self, min_wave=0, max_wave=0,
                             sky_annulus_low_arcsec=7., sky_annulus_high_arcsec=12.,
                             plot=False, verbose=False):  # LUKE
        """
        Compute growth curve in a wavelength range.
        Returns r2_growth_curve, F_growth_curve, flux, r2_half_light

        Example
        -------
        >>>r2_growth_curve, F_growth_curve, flux, r2_half_light = self.growth_curve_between(min_wave, max_wave, plot=True)    # 0,1E30 ??


        Parameters
        ----------
        min_wave : Integer, optional
            The minimum wavelength passed through the mask. The default is 0.
        max_wave : Integer, optional
            The maximum wavelength passed through the mask. The default is 0.
        sky_annulus_low_arcsec : Float, optional
            DESCRIPTION. The default is 7..
        sky_annulus_high_arcsec : Float, optional
            DESCRIPTION. The default is 12..
        plot : Boolean, optional
            If True generates and shows the plots. The default is False.
        verbose : Boolean, optional
			Print results. The default is False.

        Returns
        -------
        None.

        """

        if min_wave == 0: min_wave = self.valid_wave_min
        if max_wave == 0: max_wave = self.valid_wave_max

        index_min = np.searchsorted(self.wavelength, min_wave)
        index_max = np.searchsorted(self.wavelength, max_wave)

        intensity = np.nanmean(self.data[index_min:index_max, :, :], axis=0)
        x_peak = np.nanmedian(self.x_peaks[index_min:index_max])
        y_peak = np.nanmedian(self.y_peaks[index_min:index_max])

        if verbose:
            print("  - Peak found at spaxel position {:.3f} , {:.3f} ".format(x_peak, y_peak))
            print("  - Calculating growth curve between ", np.round(min_wave, 2), " and ", np.round(max_wave, 2), "...")

        x = np.arange(self.n_cols) - x_peak
        y = np.arange(self.n_rows) - y_peak
        r2 = np.sum(np.meshgrid(x ** 2, y ** 2), axis=0)
        r = np.sqrt(r2)
        sorted_by_distance = np.argsort(r, axis=None)

        F_curve = []
        F_growth_curve = []
        r2_growth_curve = []
        r_growth_curve = []
        sky_flux_r = []
        total_flux = 0.
        sky_flux = 0.
        F_total_star = 0
        spaxels_star = 0
        spaxels_sky = 0

        for spaxel in sorted_by_distance:
            index = np.unravel_index(spaxel, (self.n_rows, self.n_cols))
            I = intensity[index]
            #        if np.isnan(L) == False and L > 0:
            if np.isnan(I) == False:
                total_flux += I  # TODO: Properly account for solid angle...
                F_curve.append(I)
                F_growth_curve.append(total_flux)
                r2_growth_curve.append(r2[index])
                r_growth_curve.append(r[index])

                if r[index] > sky_annulus_low_arcsec / self.pixel_size_arcsec:
                    if r[index] > sky_annulus_high_arcsec / self.pixel_size_arcsec:
                        sky_flux_r.append(sky_flux)
                    else:
                        if sky_flux == 0:
                            F_total_star = total_flux
                        sky_flux = sky_flux + I
                        sky_flux_r.append(sky_flux)
                        spaxels_sky = spaxels_sky + 1

                else:
                    sky_flux_r.append(0)
                    spaxels_star = spaxels_star + 1

                    # IMPORTANT !!!! WE MUST SUBSTRACT THE RESIDUAL SKY !!!
        # F_guess = F_total_star # np.max(F_growth_curve)

        sky_per_spaxel = sky_flux / spaxels_sky
        sky_in_star = spaxels_star * sky_per_spaxel

        if verbose:
            print("  Valid spaxels in star = {}, valid spaxels in sky = {}".format(spaxels_star, spaxels_sky))
            print("  Sky value per spaxel = ", np.round(sky_per_spaxel, 3))
            print(
                "  We have to sustract {:.2f} to the total flux of the star, which is its {:.3f} % ".format(sky_in_star,
                                                                                                            sky_in_star / F_total_star * 100.))

        # r2_half_light = np.interp(.5*F_guess, F_growth_curve, r2_growth_curve)

        F_growth_star = np.ones_like(F_growth_curve) * (F_total_star - sky_in_star)

        for i in range(0, spaxels_star):
            F_growth_star[i] = F_growth_curve[i] - sky_per_spaxel * (i + 1)
            # if verbose: print  i+1, F_growth_curve[i], sky_per_spaxel * (i+1), F_growth_star[i]

        r_half_light = np.interp(.5 * (F_total_star - sky_in_star), F_growth_star, r_growth_curve)
        F_guess = F_total_star - sky_in_star
        self.seeing = 2 * r_half_light * self.pixel_size_arcsec
        # print "  Between {} and {} the seeing is {} and F_total_star = {} ".format(min_wave,max_wave,self.seeing,F_total_star)

        if plot:
            self.plot_map(circle=[x_peak, y_peak, self.seeing / 2.],
                          circle2=[x_peak, y_peak, sky_annulus_low_arcsec],
                          circle3=[x_peak, y_peak, sky_annulus_high_arcsec],
                          contours=False,
                          # spaxel2=[x_peak,y_peak],
                          verbose=True, plot_centre=False,
                          norm=colors.LogNorm())

        r_norm = r_growth_curve / r_half_light
        r_arcsec = np.array(r_growth_curve) * self.pixel_size_arcsec

        F_norm = np.array(F_growth_curve) / F_guess
        sky_norm = np.array(sky_flux_r) / F_guess
        F_star_norm = F_growth_star / F_guess

        if verbose:
            print("      Flux guess =", F_guess, " ~ ", np.nanmax(F_growth_star), " ratio = ",
                  np.nanmax(F_growth_star) / F_guess)
            print("      Half-light radius:", np.round(r_half_light * self.pixel_size_arcsec, 3),
                  " arcsec  -> seeing = ", np.round(self.seeing, 3), " arcsec, if object is a star ")
            print("      Light within 2, 3, 4, 5 half-light radii:", np.interp([2, 3, 4, 5], r_norm, F_norm))
        if plot:
            plt.figure(figsize=(10, 8))
            plt.plot(r_arcsec, F_norm, 'b-')
            plt.plot(r_arcsec, sky_norm, 'r-')
            # plt.plot(r_arcsec, F_norm-sky_norm, 'g-', linewidth=10, alpha = 0.6)
            plt.plot(r_arcsec, F_star_norm, 'g-', linewidth=10, alpha=0.5)
            plt.title("Growth curve between " + str(np.round(min_wave, 2)) + " and " + str(
                np.round(max_wave, 2)) + " in " + self.object)
            plt.xlabel("Radius [arcsec]")
            plt.ylabel("Amount of integrated flux")
            plt.xlim(0, sky_annulus_high_arcsec + 1)
            plt.axhline(y=0.5, color='k', linestyle=':', alpha=0.5)
            plt.axvline(x=self.seeing / 2, color='g', alpha=0.7)
            plt.axvline(x=self.seeing, color='k', linestyle=':', alpha=0.2)
            plt.axvline(x=3 * self.seeing / 2, color='k', linestyle=':', alpha=0.2)
            plt.axvline(x=4 * self.seeing / 2, color='k', linestyle=':', alpha=0.2)
            plt.axvline(x=5 * self.seeing / 2, color='r', linestyle='-', alpha=0.5)
            plt.axvspan(sky_annulus_low_arcsec, sky_annulus_high_arcsec, facecolor='r', alpha=0.15, zorder=3)
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.2)
            plt.minorticks_on()
            plt.show()
            plt.close()

        # return r2_growth_curve, F_growth_curve, F_guess, r2_half_light
        return r2_growth_curve, F_growth_star, F_guess, r_half_light

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def half_light_spectrum(self, r_max=1, smooth=21, min_wave=0, max_wave=0,
                            sky_annulus_low_arcsec=5., sky_annulus_high_arcsec=10.,
                            fig_size=12, plot=True, verbose=True):
        """
        Compute half light spectrum (for r_max=1) or integrated star spectrum (for r_max=5) in a wavelength range.

        Example
        -------
        >>> self.half_light_spectrum(5, plot=plot, min_wave=min_wave, max_wave=max_wave)


        Parameters
        ----------
        r_max : Integer, optional
            r_max to integrate, in units of r2_half_light (= seeing if object is a star, for flux calibration make r_max=5). The default is 1.
        smooth : Integer, optional
            smooths the data. The default is 21.
        min_wave : Integer, optional
            The minimum wavelength passed through the mask. The default is 0.
        max_wave : Integer, optional
            The maximum wavelength passed through the mask. The default is 0.
        sky_annulus_low_arcsec : Float, optional
            DESCRIPTION. The default is 5.. ***
        sky_annulus_high_arcsec : Float, optional
            DESCRIPTION. The default is 10.. ***
        fig_size : Integer, optional
            This is the size of the figure. The default is 12.
        plot : Boolean, optional
            If True generates and shows the plots. The default is True.
        verbose : Boolean, optional
            Print results. The default is True.

        Returns
        -------
        Numpy Array
            DESCRIPTION. ***
        """

        if min_wave == 0: min_wave = self.valid_wave_min
        if max_wave == 0: max_wave = self.valid_wave_max

        if verbose:
            print(
                '\n> Obtaining the integrated spectrum of star between {:.2f} and {:.2f} and radius {} r_half_light'.format(
                    min_wave, max_wave, r_max))
            print('  Considering the sky in an annulus between {}" and {}"'.format(sky_annulus_low_arcsec,
                                                                                   sky_annulus_high_arcsec))

        r2_growth_curve, F_growth_curve, flux, r2_half_light = self.growth_curve_between(min_wave, max_wave,
                                                                                         sky_annulus_low_arcsec=sky_annulus_low_arcsec,
                                                                                         sky_annulus_high_arcsec=sky_annulus_high_arcsec,
                                                                                         plot=plot,
                                                                                         verbose=verbose)  # 0,1E30 ??

        intensity = []
        smooth_x = signal.medfilt(self.x_peaks, smooth)  # originally, smooth = 11
        smooth_y = signal.medfilt(self.y_peaks, smooth)
        edgelow = (np.abs(self.wavelength - min_wave)).argmin()
        edgehigh = (np.abs(self.wavelength - max_wave)).argmin()
        valid_wl = self.wavelength[edgelow:edgehigh]

        for l in range(self.n_wave):
            x = np.arange(self.n_cols) - smooth_x[l]
            y = np.arange(self.n_rows) - smooth_y[l]
            r2 = np.sum(np.meshgrid(x ** 2, y ** 2), axis=0)
            spaxels = np.where(r2 < r2_half_light * r_max ** 2)
            intensity.append(np.nansum(self.data[l][spaxels]))

        valid_intensity = intensity[edgelow:edgehigh]
        valid_wl_smooth = signal.medfilt(valid_wl, smooth)
        valid_intensity_smooth = signal.medfilt(valid_intensity, smooth)

        if plot:
            plt.figure(figsize=(fig_size, fig_size / 2.5))
            plt.plot(self.wavelength, intensity, 'b', alpha=1, label='Intensity')
            plt.plot(valid_wl_smooth, valid_intensity_smooth, 'r-', alpha=0.5, label='Smooth = ' + str(smooth))
            margen = 0.1 * (np.nanmax(intensity) - np.nanmin(intensity))
            plt.ylim(np.nanmin(intensity) - margen, np.nanmax(intensity) + margen)
            plt.xlim(np.min(self.wavelength), np.max(self.wavelength))

            plt.ylabel("Flux")
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            plt.title("Integrated spectrum of " + self.object + " for r_max = " + str(r_max) + "r_half_light")
            plt.axvline(x=min_wave, color='k', linestyle='--', alpha=0.5)
            plt.axvline(x=max_wave, color='k', linestyle='--', alpha=0.5)
            plt.minorticks_on()
            plt.legend(frameon=False, loc=1)
            plt.show()
            plt.close()
        if r_max == 5:
            print("  Saving this integrated star flux in self.integrated_star_flux")
            self.integrated_star_flux = np.array(intensity)
        return np.array(intensity)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def do_response_curve(self, absolute_flux_file, min_wave_flux=0, max_wave_flux=0,
                          fit_degree_flux=3, step_flux=25., r_max=5, exp_time=0,
                          ha_width=0, after_telluric_correction=False,
                          sky_annulus_low_arcsec=5., sky_annulus_high_arcsec=10.,
                          odd_number=0, fit_weight=0., smooth_weight=0., smooth=0.,
                          exclude_wlm=[[0, 0]],
                          plot=True, verbose=False):

        """
        Compute the response curve of a spectrophotometric star.

        Parameters
        ----------
        absolute_flux_file: string
            filename where the spectrophotometric data are included (e.g. ffeige56.dat)
        min_wave, max_wave: floats
          wavelength range = [min_wave, max_wave] where the fit is performed
        step = 25: float
          Step (in A) for smoothing the data
        fit_degree = 3: integer
            degree of the polynomium used for the fit (3, 5, or 7).
            If fit_degree = 0 it interpolates the data
        exp_time = 60: float
          Exposition time of the calibration star
        smooth = 0.03: float
          Smooth value for interpolating the data for fit_degree = 0.
        plot: boolean
          Plot yes/no

        Example
        -------
        >>> babbsdsad   !

        Parameters
        ----------
        absolute_flux_file : String
            filename where the spectrophotometric data are included (e.g. ffeige56.dat).
        min_wave_flux : Integer, optional
            DESCRIPTION. The default is 0. ***
        max_wave_flux : Integer, optional
            DESCRIPTION. The default is 0. ***
        fit_degree_flux : Integer, optional
            DESCRIPTION. The default is 3. ***
        step_flux : Float, optional
            DESCRIPTION. The default is 25.. ***
        r_max : Integer, optional
            r_max to integrate, in units of r2_half_light (= seeing if object is a star, for flux calibration make r_max=5). The default is 5.
        exp_time : Integer, optional
            DESCRIPTION. The default is 0. ***
        ha_width : Integer, optional
            DESCRIPTION. The default is 0. ***
        after_telluric_correction : Boolean, optional
            DESCRIPTION. The default is False. ***
        sky_annulus_low_arcsec : Float, optional
            DESCRIPTION. The default is 5.. ***
        sky_annulus_high_arcsec : Float, optional
            DESCRIPTION. The default is 10.. ***
        odd_number : Integer, optional
            DESCRIPTION. The default is 0. ***
        fit_weight : Float, optional
            DESCRIPTION. The default is 0.. ***
        smooth_weight : Float, optional
            DESCRIPTION. The default is 0.. ***
        smooth : Float, optional
            smooths the data. The default is 0..
        exclude_wlm : List of Lists of Integers, optional
            DESCRIPTION. The default is [[0,0]]. ***
        plot : Boolean, optional
            If True generates and shows the plots. The default is True.
        verbose : Boolean, optional
            Print results. The default is False.

        Returns
        -------
        None.

        """
        if smooth == 0.0: smooth = 0.05

        if min_wave_flux == 0: min_wave_flux = self.valid_wave_min + step_flux
        if max_wave_flux == 0: max_wave_flux = self.valid_wave_max - step_flux
        # valid_wave_min=min_wave
        # valid_wave_max=max_wave

        print("\n> Computing response curve for", self.object, "using step=", step_flux, "A in range [",
              np.round(min_wave_flux, 2), ",", np.round(max_wave_flux, 2), "] ...")

        if exp_time == 0:
            try:
                exp_time = np.nanmedian(self.exptimes)
                print("  Exposition time from the median value of self.exptimes =", exp_time)
            except Exception:
                print(
                    "  Exposition time is not given, and failed to read it in object, assuming exp_time = 60 seconds...")
                exp_time = 60.
        else:
            print("  Exposition time provided =", exp_time, "s")

        #        flux_cal_read in units of ergs/cm/cm/s/A * 10**16
        #        lambda_cal_read, flux_cal_read, delta_lambda_read = np.loadtxt(filename, usecols=(0,1,3), unpack=True)
        lambda_cal_read, flux_cal_read = np.loadtxt(absolute_flux_file, usecols=(0, 1), unpack=True)

        valid_wl_smooth = np.arange(lambda_cal_read[0], lambda_cal_read[-1], step_flux)
        tck_star = interpolate.splrep(lambda_cal_read, flux_cal_read, s=0)
        valid_flux_smooth = interpolate.splev(valid_wl_smooth, tck_star, der=0)

        edgelow = (np.abs(valid_wl_smooth - min_wave_flux)).argmin()
        edgehigh = (np.abs(valid_wl_smooth - max_wave_flux)).argmin()

        lambda_cal = valid_wl_smooth[edgelow:edgehigh]
        flux_cal = valid_flux_smooth[edgelow:edgehigh]
        lambda_min = lambda_cal - step_flux
        lambda_max = lambda_cal + step_flux

        if self.flux_cal_step == step_flux and self.flux_cal_min_wave == min_wave_flux and self.flux_cal_max_wave == max_wave_flux and after_telluric_correction == False:
            print(
                "  This has been already computed for step = {} A in range [ {:.2f} , {:.2f} ] , using existing values ...".format(
                    step_flux, min_wave_flux, max_wave_flux))
            measured_counts = self.flux_cal_measured_counts
        else:
            self.integrated_star_flux = self.half_light_spectrum(r_max, plot=plot, min_wave=min_wave_flux,
                                                                 max_wave=max_wave_flux,
                                                                 sky_annulus_low_arcsec=sky_annulus_low_arcsec,
                                                                 sky_annulus_high_arcsec=sky_annulus_high_arcsec)

            print(
                "  Obtaining fluxes using step = {} A in range [ {:.2f} , {:.2f} ] ...".format(step_flux, min_wave_flux,
                                                                                               max_wave_flux))

            if after_telluric_correction:
                print("  Computing again after performing the telluric correction...")

            measured_counts = np.array([self.fit_Moffat_between(lambda_min[i],
                                                                lambda_max[i])[0]
                                        if lambda_cal[i] > min_wave_flux and
                                           lambda_cal[i] < max_wave_flux
                                        else np.NaN
                                        for i in range(len(lambda_cal))])

            self.flux_cal_step = step_flux
            self.flux_cal_min_wave = min_wave_flux
            self.flux_cal_max_wave = max_wave_flux
            self.flux_cal_measured_counts = measured_counts
            self.flux_cal_wavelength = lambda_cal

        _response_curve_ = measured_counts / flux_cal / exp_time  # Added exp_time Jan 2019       counts / (ergs/cm/cm/s/A * 10**16) / s  =>    10-16 erg/s/cm2/A =   counts /s

        if np.isnan(_response_curve_[0]) == True:
            _response_curve_[0] = _response_curve_[1]  # - (response_curve[2] - response_curve[1])

        scale = np.nanmedian(_response_curve_)

        response_wavelength = []
        response_curve = []

        ha_range = [6563. - ha_width / 2., 6563. + ha_width / 2.]
        # Skip bad ranges and Ha line
        if ha_width > 0:
            print(
                "  Skipping H-alpha absorption with width = {} A, adding [ {:.2f} , {:.2f} ] to list of ranges to skip...".format(
                    ha_width, ha_range[0], ha_range[1]))
            if exclude_wlm[0][0] == 0:
                _exclude_wlm_ = [ha_range]
            else:
                # _exclude_wlm_ = exclude_wlm.append(ha_range)
                _exclude_wlm_ = []
                ha_added = False
                for rango in exclude_wlm:
                    if rango[0] < ha_range[0] and rango[1] < ha_range[0]:  # Rango is BEFORE Ha
                        _exclude_wlm_.append(rango)
                    if rango[0] < ha_range[0] and rango[1] > ha_range[
                        0]:  # Rango starts BEFORE Ha but finishes WITHIN Ha
                        _exclude_wlm_.append([rango[0], ha_range[1]])
                    if rango[0] > ha_range[0] and rango[1] < ha_range[1]:  # Rango within Ha
                        _exclude_wlm_.append(ha_range)
                    if rango[0] > ha_range[0] and rango[0] < ha_range[
                        1]:  # Rango starts within Ha but finishes after Ha
                        _exclude_wlm_.append([ha_range[0], rango[1]])
                    if rango[0] > ha_range[1] and rango[1] > ha_range[1]:  # Rango is AFTER Ha, add both if needed
                        if ha_added == False:
                            _exclude_wlm_.append(ha_range)
                            ha_added = True
                        _exclude_wlm_.append(rango)
        else:
            _exclude_wlm_ = exclude_wlm

        if _exclude_wlm_[0][0] == 0:  # There is not any bad range to skip
            response_wavelength = lambda_cal
            response_curve = _response_curve_
            if verbose: print("  No ranges will be skipped")
        else:
            if verbose: print("  List of ranges to skip : ", _exclude_wlm_)
            skipping = 0
            rango_index = 0
            for i in range(len(lambda_cal) - 1):
                if rango_index < len(_exclude_wlm_):
                    if lambda_cal[i] >= _exclude_wlm_[rango_index][0] and lambda_cal[i] <= _exclude_wlm_[rango_index][
                        1]:
                        # print(" checking ", lambda_cal[i], rango_index, _exclude_wlm_[rango_index][0], _exclude_wlm_[rango_index][1])
                        skipping = skipping + 1
                        if lambda_cal[i + 1] > _exclude_wlm_[rango_index][
                            1]:  # If next value is out of range, change range_index
                            rango_index = rango_index + 1
                            # print(" changing to  range ",rango_index)
                    else:
                        response_wavelength.append(lambda_cal[i])
                        response_curve.append(_response_curve_[i])
                else:
                    response_wavelength.append(lambda_cal[i])
                    response_curve.append(_response_curve_[i])

            response_wavelength.append(lambda_cal[-1])
            response_curve.append(_response_curve_[-1])
            print("  Skipping a total of ", skipping, "wavelength points for skipping bad ranges")

        if plot:
            plt.figure(figsize=(12, 8))
            plt.plot(lambda_cal, measured_counts / exp_time, 'g+', ms=10, mew=3, label="measured counts")
            plt.plot(lambda_cal, flux_cal * scale, 'k*-', label="flux_cal * scale")
            plt.plot(lambda_cal, flux_cal * _response_curve_, 'r-', label="flux_cal * response")
            plt.xlim(self.wavelength[0] - 50, self.wavelength[-1] + 50)
            plt.axvline(x=self.wavelength[0], color='k', linestyle='-', alpha=0.7)
            plt.axvline(x=self.wavelength[-1], color='k', linestyle='-', alpha=0.7)
            plt.ylabel("Flux [counts]")
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            plt.title("Response curve for absolute flux calibration using " + self.object)
            plt.legend(frameon=True, loc=1)
            plt.grid(which='both')
            if ha_width > 0: plt.axvspan(ha_range[0], ha_range[1], facecolor='orange', alpha=0.15, zorder=3)
            if exclude_wlm[0][0] != 0:
                for i in range(len(exclude_wlm)):
                    plt.axvspan(exclude_wlm[i][0], exclude_wlm[i][1], color='r', alpha=0.1)
            plt.axvline(x=min_wave_flux, color='k', linestyle='--', alpha=0.5)
            plt.axvline(x=max_wave_flux, color='k', linestyle='--', alpha=0.5)
            plt.minorticks_on()
            plt.show()
            plt.close()

        # Obtainign smoothed response curve
        smoothfactor = 2
        if odd_number == 0: odd_number = smoothfactor * int(np.sqrt(len(response_wavelength)) / 2) - 1
        response_curve_medfilt_ = sig.medfilt(response_curve, np.int(odd_number))
        interpolated_curve = interpolate.splrep(response_wavelength, response_curve_medfilt_, s=smooth)
        response_curve_smoothed = interpolate.splev(self.wavelength, interpolated_curve, der=0)

        # Obtaining the fit
        fit = np.polyfit(response_wavelength, response_curve, fit_degree_flux)
        pp = np.poly1d(fit)
        response_curve_fitted = pp(self.wavelength)

        # Obtaining the fit using a  smoothed response curve
        # Adapting Matt code for trace peak ----------------------------------
        smoothfactor = 2
        wl = response_wavelength
        x = response_curve
        if odd_number == 0: odd_number = smoothfactor * int(np.sqrt(len(wl)) / 2) - 1  # Originarily, smoothfactor = 2
        print(
            "  Obtaining smoothed response curve using medfilt window = {} for fitting a {}-order polynomium...".format(
                odd_number, fit_degree_flux))

        wlm = signal.medfilt(wl, odd_number)
        wx = signal.medfilt(x, odd_number)

        # iteratively clip and refit for WX
        maxit = 10
        niter = 0
        stop = 0
        fit_len = 100  # -100
        resid = 0
        while stop < 1:
            fit_len_init = copy.deepcopy(fit_len)
            if niter == 0:
                fit_index = np.where(wx == wx)
                fit_len = len(fit_index)
                sigma_resid = 0.0
            if niter > 0:
                sigma_resid = MAD(resid)
                fit_index = np.where(np.abs(resid) < 4 * sigma_resid)[0]
                fit_len = len(fit_index)
            try:
                p = np.polyfit(wlm[fit_index], wx[fit_index], fit_degree_flux)
                pp = np.poly1d(p)
                fx = pp(wl)
                fxm = pp(wlm)
                resid = wx - fxm
                # print "  Iteration {:2} results in RA: sigma_residual = {:.6f}, fit_len = {:5}  fit_len ={:5}".format(niter,sigma_resid,fit_len_init,fit_len)
            except Exception:
                print('  Skipping iteration ', niter)
            if (niter >= maxit) or (fit_len_init == fit_len):
                if niter >= maxit: print("  Max iterations, {:2}, reached!")
                if fit_len_init == fit_len: print("  All interval fitted in iteration {:2} ! ".format(niter))
                stop = 2
            niter = niter + 1
            # --------------------------------------------------------------------
        interpolated_curve = interpolate.splrep(response_wavelength, fx)
        response_curve_interpolated = interpolate.splev(self.wavelength, interpolated_curve, der=0)

        # Choose solution:
        if fit_degree_flux == 0:
            print("\n> Using smoothed response curve with medfilt window =", np.str(odd_number), "and s =",
                  np.str(smooth), "as solution for the response curve")
            self.response_curve = response_curve_smoothed
        else:
            if fit_weight == 0 and smooth_weight == 0:
                print(
                    "\n> Using fit of a {}-order polynomium as solution for the response curve".format(fit_degree_flux))
                self.response_curve = response_curve_fitted
            else:
                if smooth_weight == 0:
                    print(
                        "\n> Using a combination of the fitted (weight = {:.2f}) and smoothed fitted (weight = {:.2f}) response curves".format(
                            fit_weight, 1 - fit_weight))
                    self.response_curve = response_curve_fitted * fit_weight + response_curve_interpolated * (
                                1 - fit_weight)
                else:
                    fit_smooth_weight = 1 - smooth_weight - fit_weight
                    if fit_smooth_weight <= 0:
                        print(
                            "\n> Using a combination of the fitted (weight = {:.2f}) and smoothed (weight = {:.2f}) response curves".format(
                                fit_weight, smooth_weight))
                        self.response_curve = response_curve_fitted * fit_weight + response_curve_smoothed * smooth_weight
                    else:
                        print(
                            "\n> Using a combination of the fitted (weight = {:.2f}), smoothed fitted (weight = {:.2f}) and smoothed (weight = {:.2f}) response curves".format(
                                fit_weight, fit_smooth_weight, smooth_weight))
                        self.response_curve = response_curve_fitted * fit_weight + response_curve_interpolated * fit_smooth_weight + response_curve_smoothed * smooth_weight

        if plot:
            plt.figure(figsize=(12, 8))
            plt.plot(lambda_cal, _response_curve_, 'k--', alpha=0.7, label="Raw response curve")
            plt.plot(response_wavelength, response_curve, 'k-', alpha=1., label='  "  excluding bad ranges')
            plt.plot(self.wavelength, self.response_curve, "g-", alpha=0.4, linewidth=12,
                     label="Obtained response curve")
            text = "Smoothed with medfilt window = " + np.str(odd_number) + " and s = " + np.str(smooth)
            plt.plot(self.wavelength, response_curve_smoothed, "-", color="orange", alpha=0.8, linewidth=2, label=text)
            if fit_degree_flux > 0:
                text = "Fit using polynomium of degree " + np.str(fit_degree_flux)
                plt.plot(self.wavelength, response_curve_fitted, "b-", alpha=0.6, linewidth=2, label=text)
                text = np.str(fit_degree_flux) + "-order fit smoothed with medfilt window = " + np.str(odd_number)
                plt.plot(self.wavelength, response_curve_interpolated, "-", color="purple", alpha=0.6, linewidth=2,
                         label=text)

            plt.xlim(self.wavelength[0] - 50, self.wavelength[-1] + 50)
            plt.axvline(x=self.wavelength[0], color='k', linestyle='-', alpha=0.7)
            plt.axvline(x=self.wavelength[-1], color='k', linestyle='-', alpha=0.7)
            if ha_range[0] != 0: plt.axvspan(ha_range[0], ha_range[1], facecolor='orange', alpha=0.15, zorder=3)
            if exclude_wlm[0][0] != 0:
                for i in range(len(exclude_wlm)):
                    plt.axvspan(exclude_wlm[i][0], exclude_wlm[i][1], color='r', alpha=0.1)
            plt.ylabel(
                "Flux calibration [ counts /s equivalent to 10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            plt.title("Response curve for absolute flux calibration using " + self.object)
            plt.minorticks_on()
            plt.grid(which='both')
            plt.axvline(x=min_wave_flux, color='k', linestyle='--', alpha=0.5)
            plt.axvline(x=max_wave_flux, color='k', linestyle='--', alpha=0.5)
            plt.legend(frameon=True, loc=4, ncol=2)
            plt.show()
            plt.close()

        print("  Min wavelength at {:.2f} with {:.3f} counts/s = 1E-16 erg/cm**2/s/A".format(self.wavelength[0],
                                                                                             self.response_curve[0]))
        print("  Max wavelength at {:.2f} with {:.3f} counts/s = 1E-16 erg/cm**2/s/A".format(self.wavelength[-1],
                                                                                             self.response_curve[-1]))
        print("  Response curve to all wavelengths stored in self.response_curve. Length of the vector = ",
              len(self.response_curve))

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def fit_Moffat_between(self, min_wave=0, max_wave=0, r_max=5, plot=False, verbose=False):
        """
        ***

        Parameters
        ----------
        min_wave : Integer, optional
            The minimum wavelength passed through the mask. The default is 0.
        max_wave : Integer, optional
            The maximum wavelength passed through the mask. The default is 0.
        r_max : Integer, optional
            r_max to integrate, in units of r2_half_light (= seeing if object is a star, for flux calibration make r_max=5). The default is 5.
        plot : Boolean, optional
            If True generates and shows the plots. The default is False.
        verbose : Boolean, optional
            Print results. The default is False.

        Returns
        -------
        flux : TYPE ***
            DESCRIPTION.
        TYPE ***
            DESCRIPTION.
        beta : TYPE ***
            DESCRIPTION.

        """

        if min_wave == 0: min_wave = self.valid_wave_min
        if max_wave == 0: max_wave = self.valid_wave_max

        r2_growth_curve, F_growth_curve, flux, r2_half_light = self.growth_curve_between(min_wave, max_wave, plot=plot,
                                                                                         verbose=verbose)
        flux, alpha, beta = fit_Moffat(r2_growth_curve, F_growth_curve,
                                       flux, r2_half_light, r_max, plot)
        r2_half_light = alpha * (np.power(2., 1. / beta) - 1)

        if plot == True: verbose == True
        if verbose:
            print("Moffat fit: Flux = {:.3e},".format(flux), \
                  "HWHM = {:.3f},".format(np.sqrt(r2_half_light) * self.pixel_size_arcsec), \
                  "beta = {:.3f}".format(beta))

        return flux, np.sqrt(r2_half_light) * self.pixel_size_arcsec, beta

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def trim_cube(self, trim_cube=True, trim_values=[], half_size_for_centroid=10,
                  remove_spaxels_not_fully_covered=True,  # nansum=False,
                  ADR=True, box_x=[0, -1], box_y=[0, -1], edgelow=-1, edgehigh=-1,
                  adr_index_fit=2, g2d=False, step_tracing=100, plot_tracing_maps=[],
                  plot_weight=False, fcal=False, plot=True, plot_spectra=False, verbose=True, warnings=True):
        """
        Task for trimming cubes in RA and DEC (not in wavelength)

        if nansum = True, it keeps spaxels in edges that only have a partial spectrum (default = False)
        if remove_spaxels_not_fully_covered = False, it keeps spaxels in edges that only have a partial spectrum (default = True)


        Parameters
        ----------
        trim_cube : Boolean, optional
            DESCRIPTION. The default is True. ***
        trim_values : List, optional
            DESCRIPTION. The default is []. ***
        half_size_for_centroid : Integer, optional
            This is half the length/width of the box. The default is 10.
        remove_spaxels_not_fully_covered : Boolean, optional
            DESCRIPTION. The default is True.
        nansum : Boolean, optional
            If True will sum the number of NaNs in the columns and rows in the intergrated map. The default is False.
        ADR : Boolean, optional
            If True will correct for ADR (Atmospheric Differential Refraction). The default is True.
        box_x : Integer List, optional
             When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [0,-1].
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [0,-1].
        edgelow : Integer, optional
            This is the lowest value in the wavelength range in terms of pixels. The default is -1.
        edgehigh : Integer, optional
            This is the highest value in the wavelength range in terms of pixels, (maximum wavelength - edgehigh). The default is -1.
        adr_index_fit : Integer, optional
            This is the fitted polynomial with highest degree n. The default is 2.
        g2d : Boolean, optional
            If True uses a 2D gaussian, else doesn't. The default is False.
        step_tracing : Integer, optional
            DESCRIPTION. The default is 100. ***
        plot_tracing_maps : List, optional
            If True will plot the tracing maps. The default is [].
        plot_weight : Boolean, optional
            DESCRIPTION. The default is False. ***
        fcal : Boolean, optional
            If fcal=True, cube.flux_calibration is used. The default is False.
        plot : Boolean, optional
            If True generates and shows the plots. The default is True.
        plot_spectra : Boolean, optional
            If True will plot the spectra. The default is False.
        verbose : Boolean, optional
            Print results. The default is True.
        warnings : Boolean, optional
            If True will show any problems that arose, else skipped. The default is True.

        Returns
        -------
        None.

        """

        # Check if this is a self.combined cube or a self
        try:
            _x_ = np.nanmedian(self.combined_cube.data)
            if _x_ > 0: cube = self.combined_cube
        except Exception:
            cube = self

        if remove_spaxels_not_fully_covered:
            nansum = False
        else:
            nansum = True

        if verbose:
            if nansum:
                print(
                    "\n> Preparing for trimming the cube INCLUDING those spaxels in edges with partial wavelength coverage...")
            else:
                print(
                    "\n> Preparing for trimming the cube DISCARDING those spaxels in edges with partial wavelength coverage...")

            print('  Size of old cube : {} (RA) x {} (DEC) = [0 ... {} , 0 ... {}]  =  {:.2f}" x {:.2f}"'.format(
                cube.n_cols, cube.n_rows, cube.n_cols - 1, cube.n_rows - 1, cube.RA_segment, cube.DEC_segment))

        if len(trim_values) == 0:  # Trim values not given. Checking size for trimming
            if verbose: print("  Automatically computing the trimming avoiding empty columns and rows...")
            trim_values = [0, cube.n_cols - 1, 0, cube.n_rows - 1]
            cube.get_integrated_map(nansum=nansum)
            n_row_values = np.nansum(cube.integrated_map, axis=1)
            n_col_values = np.nansum(cube.integrated_map, axis=0)

            stop = 0
            i = 0
            while stop < 1:
                if n_col_values[i] == 0:
                    trim_values[0] = i + 1
                    i = i + 1
                    if i == np.int(cube.n_cols / 2):
                        if verbose or warnings: print("  Something failed checking left trimming...")
                        trim_values[0] = -1
                        stop = 2
                else:
                    stop = 2
            stop = 0
            i = cube.n_cols - 1
            while stop < 1:
                if n_col_values[i] == 0:
                    trim_values[1] = i - 1
                    i = i - 1
                    if i == np.int(cube.n_cols / 2):
                        if verbose or warnings: print("  Something failed checking right trimming...")
                        trim_values[1] = cube.n_cols
                        stop = 2
                else:
                    stop = 2
            stop = 0
            i = 0
            while stop < 1:
                if n_row_values[i] == 0:
                    trim_values[2] = i + 1
                    i = i + 1
                    if i == np.int(cube.n_rows / 2):
                        if verbose or warnings: print("  Something failed checking bottom trimming...")
                        trim_values[2] = -1
                        stop = 2
                else:
                    stop = 2
            stop = 0
            i = cube.n_rows - 1
            while stop < 1:
                if n_row_values[i] == 0:
                    trim_values[3] = i - 1
                    i = i - 1
                    if i == np.int(cube.n_rows / 2):
                        if verbose or warnings: print("  Something failed checking top trimming...")
                        trim_values[3] = cube.n_rows
                        stop = 2
                else:
                    stop = 2
        else:
            if trim_values[0] == -1: trim_values[0] = 0
            if trim_values[1] == -1: trim_values[1] = cube.n_cols - 1
            if trim_values[2] == -1: trim_values[2] = 0
            if trim_values[3] == -1: trim_values[3] = cube.n_rows - 1

            if verbose: print(
                "  Trimming values provided: [ {}:{} , {}:{} ]".format(trim_values[0], trim_values[1], trim_values[2],
                                                                       trim_values[3]))
            if trim_values[0] < 0:
                trim_values[0] = 0
                if verbose: print("  trim_value[0] cannot be negative!")
            if trim_values[1] > cube.n_cols:
                trim_values[1] = cube.n_cols
                if verbose: print("  The requested value for trim_values[1] is larger than the RA size of the cube!")
            if trim_values[1] < 0:
                trim_values[1] = cube.n_cols
                if verbose: print("  trim_value[1] cannot be negative!")
            if trim_values[2] < 0:
                trim_values[2] = 0
                if verbose: print("  trim_value[2] cannot be negative!")
            if trim_values[3] > cube.n_rows:
                trim_values[3] = cube.n_rows
                if verbose: print("  The requested value for trim_values[3] is larger than the DEC size of the cube!")
            if trim_values[3] < 0:
                trim_values[3] = cube.n_rows
                if verbose: print("  trim_value[3] cannot be negative!")

        recorte_izquierda = (trim_values[0]) * cube.pixel_size_arcsec
        recorte_derecha = (cube.n_cols - trim_values[1] - 1) * cube.pixel_size_arcsec
        recorte_abajo = (trim_values[2]) * cube.pixel_size_arcsec
        recorte_arriba = (cube.n_rows - trim_values[3] - 1) * cube.pixel_size_arcsec

        corte_horizontal = (trim_values[0]) + (cube.n_cols - trim_values[1] - 1)
        corte_vertical = (trim_values[2]) + (cube.n_rows - trim_values[3] - 1)

        if verbose:
            print('  Left trimming   : from spaxel   0  to spaxel {:3}  -> {:3} spaxels = {:8.2f}" '.format(
                trim_values[0], (trim_values[0]), recorte_izquierda))
            print('  Right trimming  : from spaxel {:3}  to spaxel {:3}  -> {:3} spaxels = {:8.2f}" '.format(
                trim_values[1], cube.n_cols - 1, (cube.n_cols - trim_values[1] - 1), recorte_derecha))
            print('  Bottom trimming : from spaxel   0  to spaxel {:3}  -> {:3} spaxels = {:8.2f}" '.format(
                trim_values[2], (trim_values[2]), recorte_abajo))
            print('  Top trimming    : from spaxel {:3}  to spaxel {:3}  -> {:3} spaxels = {:8.2f}" '.format(
                trim_values[3], cube.n_rows - 1, (cube.n_rows - trim_values[3] - 1), recorte_arriba))

            print("  This will need a trimming of {} (RA) x {} (DEC) spaxels".format(corte_horizontal, corte_vertical))

        # if corte_horizontal % 2 != 0:
        #     corte_horizontal = corte_horizontal + 1
        #     trim_values[1] = trim_values[1] -1
        #     recorte_derecha =  (cube.n_cols-trim_values[1])*cube.pixel_size_arcsec
        #     print("  We need to trim another spaxel in RA to get an EVEN number of columns")

        # if corte_vertical % 2 != 0:
        #     corte_vertical = corte_vertical + 1
        #     trim_values[3] = trim_values[3] -1
        #     recorte_arriba = (cube.n_rows-trim_values[3])*cube.pixel_size_arcsec
        #     print("  We need to trim another spaxel in DEC to get an EVEN number of rows")

        x0 = trim_values[0]
        x1 = trim_values[0] + (cube.n_cols - corte_horizontal)  # trim_values[1]
        y0 = trim_values[2]
        y1 = trim_values[2] + (cube.n_rows - corte_vertical)  # trim_values[3]

        # print "  RA trim {}  DEC trim {}".format(corte_horizontal,corte_vertical)
        values = np.str(cube.n_cols - corte_horizontal) + ' (RA) x ' + np.str(
            cube.n_rows - corte_vertical) + ' (DEC) = ' + np.str(
            np.round((x1 - x0) * cube.pixel_size_arcsec, 2)) + '" x ' + np.str(
            np.round((y1 - y0) * cube.pixel_size_arcsec, 2)) + '"'

        if verbose:  print("\n> Recommended size values of combined cube = ", values)

        if corte_horizontal == 0 and corte_vertical == 0:
            if verbose: print("\n> No need of trimming the cube, all spaxels are valid.")
            trim_cube = False
            cube.get_integrated_map()
        if trim_cube:
            # if plot:
            #    print "> Plotting map with trimming box"
            #    cube.plot_map(mapa=cube.integrated_map, box_x=[x0,x1], box_y=[y0,y1])
            cube.RA_centre_deg = cube.RA_centre_deg + (recorte_derecha - recorte_izquierda) / 2 / 3600.
            cube.DEC_centre_deg = cube.DEC_centre_deg + (recorte_abajo - recorte_arriba) / 2. / 3600.
            if verbose:
                print("\n> Starting trimming procedure:")
                print(
                    '  Size of old cube : {} (RA) x {} (DEC) = [0 ... {} , 0 ... {}]  =  {}" x {}"'.format(cube.n_cols,
                                                                                                           cube.n_rows,
                                                                                                           cube.n_cols - 1,
                                                                                                           cube.n_rows - 1,
                                                                                                           np.round(
                                                                                                               cube.RA_segment,
                                                                                                               2),
                                                                                                           np.round(
                                                                                                               cube.DEC_segment,
                                                                                                               2)))
                print("  Reducing size of the old cube in {} (RA) and {} (DEC) spaxels...".format(corte_horizontal,
                                                                                                  corte_vertical))
                print("  Centre coordenates of the old cube: ", cube.RA_centre_deg, cube.DEC_centre_deg)
                print('  Offset for moving the center from the old to the new cube:  {:.2f}" x {:.2f}"'.format(
                    (recorte_derecha - recorte_izquierda) / 2., (recorte_abajo - recorte_arriba) / 2.))
                print("  Centre coordenates of the new cube: ", cube.RA_centre_deg, cube.DEC_centre_deg)
                print("  Trimming the cube [{}:{} , {}:{}] ...".format(x0, x1 - 1, y0, y1 - 1))
            cube.data = copy.deepcopy(cube.data[:, y0:y1, x0:x1])
            # cube.data_no_ADR = cube.data_no_ADR[:,y0:y1,x0:x1]
            cube.weight = cube.weight[:, y0:y1, x0:x1]
            if plot_weight: cube.plot_weight()
            cube.n_cols = cube.data.shape[2]
            cube.n_rows = cube.data.shape[1]
            cube.RA_segment = cube.n_cols * cube.pixel_size_arcsec
            cube.DEC_segment = cube.n_rows * cube.pixel_size_arcsec

            if verbose: print(
                '  Size of new cube : {} (RA) x {} (DEC) = [0 ... {} , 0 ... {}]  =  {}" x {}"'.format(cube.n_cols,
                                                                                                       cube.n_rows,
                                                                                                       cube.n_cols - 1,
                                                                                                       cube.n_rows - 1,
                                                                                                       np.round(
                                                                                                           cube.RA_segment,
                                                                                                           2), np.round(
                        cube.DEC_segment, 2)))

            cube.spaxel_RA0 = cube.n_cols / 2 - 1
            cube.spaxel_DEC0 = cube.n_rows / 2 - 1

            if verbose: print(
                '  The center of the cube is in position  [ {} , {} ]'.format(cube.spaxel_RA0, cube.spaxel_DEC0))

            if ADR:
                if np.nanmedian(box_x + box_y) != -0.5:
                    box_x_ = [box_x[0] - x0, box_x[1] - x0]
                    box_y_ = [box_y[0] - y0, box_y[1] - y0]
                else:
                    box_x_ = box_x
                    box_y_ = box_y

                if half_size_for_centroid > 0 and np.nanmedian(box_x + box_y) == -0.5:
                    cube.get_integrated_map()
                    if verbose: print(
                        "\n> As requested, using a box centered at the peak of emission, [ {} , {} ], and width +-{} spaxels for tracing...".format(
                            cube.max_x, cube.max_y, half_size_for_centroid))
                    box_x_ = [cube.max_x - half_size_for_centroid, cube.max_x + half_size_for_centroid]
                    box_y_ = [cube.max_y - half_size_for_centroid, cube.max_y + half_size_for_centroid]

                cube.trace_peak(check_ADR=True, box_x=box_x_, box_y=box_y_, edgelow=edgelow, edgehigh=edgehigh,
                                adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing,
                                plot=plot, plot_tracing_maps=plot_tracing_maps, verbose=verbose)
            cube.get_integrated_map(fcal=fcal, plot=plot, plot_spectra=plot_spectra, plot_centroid=False, nansum=nansum)


        else:
            if corte_horizontal != 0 and corte_vertical != 0 and verbose: print(
                "\n> Trimming the cube was not requested")

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def get_bad_spaxels(self, verbose=False, show_results=True, valid_wave_min=0, valid_wave_max=0):
        """
        Get a list with the bad spaxels (spaxels that don't have a valid spectrum in the range [valid_wave_min, valid_wave_max])

        Parameters
        ----------
        verbose : Boolean, optional
            Print results. The default is False.
        show_results : Boolean, optional
            DESCRIPTION. The default is True. ***
        valid_wave_min : Integer, optional
            DESCRIPTION. The default is 0. ***
        valid_wave_max : Integer, optional
            DESCRIPTION. The default is 0. ***

        Returns
        -------
        self.bad_spaxels with the list of bad spaxels

        """

        if valid_wave_min == 0: valid_wave_min = self.valid_wave_min
        if valid_wave_max == 0: valid_wave_max = self.valid_wave_max

        if verbose: print(
            "\n> Checking bad spaxels (spaxels that don't have a valid spectrum in the range [ {} , {} ] ) :\n".format(
                np.round(valid_wave_min, 2), np.round(valid_wave_max, 2)))

        list_of_bad_spaxels = []
        for x in range(self.n_cols):
            for y in range(self.n_rows):
                integrated_spectrum_of_spaxel = self.plot_spectrum_cube(x=x, y=y, plot=False, verbose=False)
                median = np.median(integrated_spectrum_of_spaxel[
                                   np.searchsorted(self.wavelength, valid_wave_min):np.searchsorted(self.wavelength,
                                                                                                    valid_wave_max)])
                if np.isnan(median):
                    if verbose: print(
                        "  - spaxel {:3},{:3} does not have a spectrum in all valid wavelenghts".format(x, y))
                    list_of_bad_spaxels.append([x, y])
                    # integrated_spectrum_of_spaxel= cubo.plot_spectrum_cube(x=x,y=y, plot=True, verbose=False)
        self.bad_spaxels = list_of_bad_spaxels
        if show_results: print("\n> List of bad spaxels :", self.bad_spaxels)