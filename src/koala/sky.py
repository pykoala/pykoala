"""

INCLUDE BRIEF DESCRIPTION OF THE MODULE

"""
import numpy as np


def compute_1D_sky_from_RSS(RSS, sky_wave_min=None, sky_wave_max=None, n_sky=200, sigma_clip=2.0):
    """
        Identify n_sky spaxels with the LOWEST INTEGRATED VALUES and store them in RSS.sky_fibres

        Parameters
        ----------
        sky_wave_min, sky_wave_max : float, float (default 0, 0)
            Consider the integrated flux in the range [sky_wave_min, sky_wave_max]
            If 0, they are set to RSS.valid_wave_min or RSS.valid_wave_max
        n_sky : integer (default = 200)
            number of spaxels used for identifying sky.
            200 is a good number for calibration stars
            for real objects, particularly extense objects, set n_sky = 30 - 50
        sigma_clip: float (default 2.0)
            Number of standard deviations for sigma-clip the selected fibre spectra.
        """
    print(' [SKY]  Computing 1D model sky spectra from RSS')
    # Set wavelength edges for estimating sky emission
    if sky_wave_min is None:
        sky_wave_min = RSS.valid_wave_min
    if sky_wave_max is None:
        sky_wave_max = RSS.valid_wave_max
    print(' [SKY] Estimating sky spectra in the range {:.1f} -- {:.1f} AA'.format(sky_wave_min, sky_wave_max))
    # Collapse fibre spectra
    wl_range = np.where((RSS.wavelength > valid_wave_min) & (RSS.wavelength < valid_wave_max))[0]
    collapsed_spectra = np.nansum(RSS.intensity_corrected[:, wl_range], axis=1)
    # Sort the fibres as function of total integrated flux
    sorted_positions = np.argsort(collapsed_spectra)
    print(' [SKY] The {} fibres with the lowest integrated flux will be selected as sky-fibres'.format(n_sky))
    # Select sky-fibres candidates and get flux
    RSS.sky_fibres = sorted_positions[:n_sky]
    skyfibres_flux = RSS.intensity_corrected[RSS.sky_fibres, :].copy()
    # Apply 2-sigma clipping for removing outliers
    if sigma_clip is not None:
        print(' [SKY] Applying sigma-clipping to remove outliers before sky flux estimation')
        mean = np.nanmean(skyfibres_flux, axis=0)
        std = np.nanstd(skyfibres_flux, axis=0)
        outliers = np.abs(skyfibres_flux - mean[np.newaxis, :]) > sigma_clip * std[:, np.newaxis]
        skyfibres_flux[outliers] = np.nan
    # Compute the median and percentiles of sky emission
    p16 = np.nanpercentile(skyfibres_flux, 16, axis=0)
    p50 = np.nanpercentile(skyfibres_flux, 50, axis=0)
    p84 = np.nanpercentile(skyfibres_flux, 84, axis=0)
    RSS.sky_emission =

def find_sky_fibres(RSS, sky_wave_min=None, sky_wave_max=None, n_sky=200, plot=False, verbose=True, warnings=True):
    """
    Identify n_sky spaxels with the LOWEST INTEGRATED VALUES and store them in RSS.sky_fibres

    Parameters
    ----------
    sky_wave_min, sky_wave_max : float, float (default 0, 0)
        Consider the integrated flux in the range [sky_wave_min, sky_wave_max]
        If 0, they are set to RSS.valid_wave_min or RSS.valid_wave_max
    n_sky : integer (default = 200)
        number of spaxels used for identifying sky.
        200 is a good number for calibration stars
        for real objects, particularly extense objects, set n_sky = 30 - 50
    plot : boolean (default = False)
        plots a RSS map with sky positions
    verbose : bool (default True)
    warnings : bool (default True)
    """
    if sky_wave_min is None:
        sky_wave_min = RSS.valid_wave_min
    if sky_wave_max is None:
        sky_wave_max = RSS.valid_wave_max
    RSS.compute_integrated_fibre(valid_wave_min=sky_wave_min, valid_wave_max=sky_wave_max, plot=False, verbose=verbose,
                                 warnings=warnings)
    sorted_positions = np.argsort(RSS.integrated_fibre)
    print("\n> Identifying sky spaxels using the lowest integrated values in the [", np.round(sky_wave_min, 2), ",",
          np.round(sky_wave_max, 2), "] range ...")
    print("  We use the lowest", n_sky, "fibres for getting sky. Their positions are:")
    # Compute sky spectrum and plot RSS map with sky positions if requested
    RSS.sky_fibres = sorted_positions[:n_sky]
    if plot:
        RSS.RSS_map(RSS.integrated_fibre, None, RSS.sky_fibres, title=" - Sky Spaxels")


def find_sky_emission(RSS, sky_fibres=None, sky_wave_min=None, sky_wave_max=None, n_sky=200, win_sky=None,
                      include_history=True, log=True, gamma=0, plot=True):
    """
    Find the sky emission given fibre list or taking n_sky fibres with lowest integrated value.

    Parameters
    ----------
    intensidad :
        Matrix with intensities (RSS.intensity or RSS.intensity_corrected)
    plot : boolean (default = True)
        Plots results
    n_sky : integer (default = 200)
        number of lowest intensity fibres that will be used to create a SKY specrum
        200 is a good number for calibration stars
        for real objects, particularly extense objects, set n_sky = 30 - 50
    sky_fibres : list of floats (default = [1000])
       fibre or fibre range associated with sky
       If [1000], then the fibre list of sky will be computed automatically using n_sky
    sky_wave_min, sky_wave_max : float, float (default 0, 0)
        Only used when sky_fibres is [1000]
        Consider the integrated flux in the range [sky_wave_min, sky_wave_max]
        If 0, they are set to RSS.valid_wave_min or RSS.valid_wave_max
    log, gamma:
        Normalization scale, default is lineal scale.
        Lineal scale: norm=colors.Normalize().   log = False, gamma = 0
        Log scale:    norm=colors.LogNorm()      log = True, gamma = 0
        Power law:    norm=colors.PowerNorm(gamma=1./4.) when gamma != 0
        //
    win_sky : odd integer (default = 0)
        Width in fibres of a median filter applied to obtain sky spectrum
        If 0, it will not apply any median filter.
    include_history : boolean (default = True)
        If True, it includes RSS.history the basic information
    """
    if sky_fibres is None:
        RSS.find_sky_fibres(sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, n_sky=n_sky)
    else:  # We provide a list with sky positions
        print("  We use the list provided to get the sky spectrum")
        print("  sky_fibres = ", sky_fibres)
        RSS.sky_fibres = np.array(sky_fibres)

    if plot:
        RSS.RSS_map(RSS.integrated_fibre, list_spectra=RSS.sky_fibres, log=log, gamma=gamma, title=" - Sky Spaxels")
    print("  List of fibres used for sky saved in RSS.sky_fibres")

    if include_history:
        RSS.history.append("- Obtaining the sky emission using " + np.str(n_sky) + " fibres")
    RSS.sky_emission = sky_spectrum_from_fibres(RSS, RSS.sky_fibres, win_sky=win_sky, plot=False,
                                                include_history=include_history)

    if plot: plot_plot(RSS.wavelength, RSS.sky_emission, color="c",
                       ylabel="Relative flux [counts]", xlabel="Wavelength [$\mathrm{\AA}$]",
                       xmin=RSS.wavelength[0] - 10, xmax=RSS.wavelength[-1] + 10,
                       ymin=np.nanpercentile(RSS.sky_emission, 1), ymax=np.nanpercentile(RSS.sky_emission, 99),
                       vlines=[RSS.valid_wave_min, RSS.valid_wave_max],
                       ptitle="Combined sky spectrum using the requested fibres")
    print("  Sky spectrum obtained and stored in RSS.sky_emission !! ")


def sky_spectrum_from_fibres(rss, list_spectra, win_sky=None, wave_to_fit=300, fit_order=2, include_history=True,
                             verbose=True, plot=True):
    if verbose:
        print("\n> Obtaining 1D sky spectrum using the rss file and fibre list = ")
        print("  ", list_spectra)

    _rss_ = copy.deepcopy(rss)
    w = _rss_.wavelength

    if win_sky is not None:
        if verbose:
            print("  after applying a median filter with kernel ", win_sky, "...")
        _rss_.intensity_corrected = median_2D_filter(_rss_.intensity_corrected, _rss_.n_spectra, _rss_.n_wave,
                                                     win_sky=win_sky)
    # TODO: THIS IS VERY OBSCURE AND INEFFICIENT
    sky = _rss_.plot_combined_spectrum(list_spectra=list_spectra, median=True, plot=plot)

    # Find the last good pixel in sky
    last_good_pixel_sky = _rss_.n_wave - 1
    found = 0
    while found < 1:
        if sky[last_good_pixel_sky] > 0:
            found = 2
        else:
            last_good_pixel_sky = last_good_pixel_sky - 1
            if last_good_pixel_sky == _rss_.mask_good_index_range[1]:
                if verbose: print(" WARNING ! last_good_pixel_sky is the same than in file")
                found = 2

    if verbose: print(
        "\n - Using a 2-order fit to the valid red end of sky spectrum to extend continuum to all wavelengths")

    if rss.grating == "385R":
        wave_to_fit = 200
        fit_order = 1

    lmin = _rss_.mask_good_wavelength_range[1] - wave_to_fit  # GAFAS
    w_spec = []
    f_spec = []
    w_spec.extend((w[i]) for i in range(len(w)) if (w[i] > lmin) and (w[i] < _rss_.mask_good_wavelength_range[1]))
    f_spec.extend((sky[i]) for i in range(len(w)) if (w[i] > lmin) and (w[i] < _rss_.mask_good_wavelength_range[1]))

    fit = np.polyfit(w_spec, f_spec, fit_order)
    # if fit_order == 2:
    #     ptitle="Fitting to skyline "+np.str(sky_line)+" : {:.3e} x$^2$  +  {:.3e} x  +  {:.3e} ".format(fit[0],fit[1],fit[2])+text
    # if fit_order == 1:
    #     ptitle="Fitting to skyline "+np.str(sky_line)+" : {:.3e} x  +  {:.3e} ".format(fit[0],fit[1])+text
    # if fit_order > 2:
    #     ptitle="Fitting an order "+np.str(fit_order)+" polinomium to skyline "+np.str(sky_line)+text

    y = np.poly1d(fit)
    y_fitted_all = y(w)

    if plot:
        plot_plot(w, [sky, y_fitted_all], xmin=lmin, percentile_min=0, percentile_max=100,
                  ptitle="Extrapolating the sky continuum for the red edge",
                  vlines=[_rss_.mask_good_wavelength_range[1], w[last_good_pixel_sky - 3]])

    sky[last_good_pixel_sky - 3:-1] = y_fitted_all[last_good_pixel_sky - 3:-1]

    if include_history: rss.history.append("  Mask used to get a rough sky in the pixels of the red edge")
    return sky
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------


def substract_sky(RSS, correct_negative_sky=False, plot=True, verbose=True, warnings=True,
                  order_fit_negative_sky=3, kernel_negative_sky=51, exclude_wlm=[[0, 0]],
                  individual_check=True, use_fit_for_negative_sky=False, low_fibres=10):
    """
    Substracts the sky stored in RSS.sky_emission to all fibres in RSS.intensity_corrected

    Parameters
    ----------
    correct_negative_sky : boolean (default = True)
        If True, and if the integrated value of the median sky is negative, this is corrected
    plot : boolean (default = True)
        Plots results
    see task 'correcting_negative_sky()' for definition of the rest of the parameters
    """
    # Substract sky in all intensities
    # for i in range(RSS.n_spectra):
    #    RSS.intensity_corrected[i,:]=RSS.intensity_corrected[i,:] - RSS.sky_emission
    RSS.intensity_corrected -= RSS.sky_emission[np.newaxis, :]
    # TODO: Warning!!! THIS IS CORRECT? SHOULDN WE NEED TO COMPUTE VAR_{SKY}?
    # RSS.variance_corrected -= RSS.sky_emission[np.newaxis, :]

    if len(RSS.sky_fibres) > 0: last_sky_fibre = RSS.sky_fibres[-1]
    median_sky_corrected = np.zeros(RSS.n_spectra)

    for i in range(RSS.n_spectra):
        median_sky_corrected[i] = np.nanmedian(
            RSS.intensity_corrected[i, RSS.valid_wave_min_index:RSS.valid_wave_max_index], axis=0)
    if len(RSS.sky_fibres) > 0: median_sky_per_fibre = np.nanmedian(median_sky_corrected[RSS.sky_fibres])

    if verbose:
        print("  Median flux all fibres          = ", np.round(np.nanmedian(median_sky_corrected), 3))
        if len(RSS.sky_fibres) > 0:
            print("  Median flux sky fibres          = ", np.round(median_sky_per_fibre, 3))
            print("  Median flux brightest sky fibre = ", np.round(median_sky_corrected[last_sky_fibre], 3))
            print("  Median flux faintest  sky fibre = ", np.round(median_sky_corrected[RSS.sky_fibres[0]], 3))

    # Plot median value of fibre vs. fibre
    if plot:

        if len(RSS.sky_fibres) > 0:
            ymin = median_sky_corrected[RSS.sky_fibres[0]] - 1
            # ymax = np.nanpercentile(median_sky_corrected,90),
            hlines = [np.nanmedian(median_sky_corrected), median_sky_corrected[RSS.sky_fibres[0]],
                      median_sky_corrected[last_sky_fibre], median_sky_per_fibre]
            chlines = ["r", "k", "k", "g"]
            ptitle = "Median flux per fibre after sky substraction\n (red = median flux all fibres, green = median flux sky fibres, grey = median flux faintest/brightest sky fibre)"
        else:
            ymin = np.nanpercentile(median_sky_corrected, 1)
            # ymax=np.nanpercentile(RSS.sky_emission, 1)
            hlines = [np.nanmedian(median_sky_corrected), 0]
            chlines = ["r", "k"]
            ptitle = "Median flux per fibre after sky substraction (red = median flux all fibres)"

        plot_plot(list(range(RSS.n_spectra)), median_sky_corrected,
                  ylabel="Median Flux [counts]", xlabel="Fibre",
                  ymin=ymin, ymax=np.nanpercentile(median_sky_corrected, 90),
                  hlines=hlines, chlines=chlines,
                  ptitle=ptitle)

    if len(RSS.sky_fibres) > 0:
        if median_sky_corrected[RSS.sky_fibres[0]] < 0:
            if verbose or warnings: print(
                "  WARNING !  The integrated value of the sky fibre with the smallest value is negative!")
            if correct_negative_sky:
                if verbose: print("  Fixing this, as 'correct_negative_sky' = True  ... ")
                RSS.correcting_negative_sky(plot=plot, low_fibres=low_fibres, exclude_wlm=exclude_wlm,
                                             kernel_negative_sky=kernel_negative_sky,
                                             use_fit_for_negative_sky=use_fit_for_negative_sky,
                                             order_fit_negative_sky=order_fit_negative_sky,
                                             individual_check=individual_check)

    if verbose: print("  Intensities corrected for sky emission and stored in RSS.intensity_corrected !")
    RSS.history.append("  Intensities corrected for the sky emission")
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------


def read_sky_spectrum(RSS, sky_spectrum_file, path="", verbose=True):
    """
    Reads a TXT file with a 1D spectrum

    Parameters
    ----------
    sky_spectrum_file : string (default = None)
        Specify the name of sky spectrum file (including or not the path)
    path: string (default = "")
        path to the sky spectrum file
    verbose : Boolean (optional)
        Print what is doing. The default is True.

    Returns
    -------
    sky_spectrum : array
        1D sky spectrum

    It also adds the 1D sky spectrum to RSS.sky_spectrum
    """

    if path != "": sky_spectrum_file = full_path(sky_spectrum_file, path)

    if verbose:
        print("\n> Reading file with a 1D sky spectrum :")
        print(" ", sky_spectrum_file)

    w_sky, sky_spectrum = read_table(sky_spectrum_file, ["f", "f"])

    RSS.sky_spectrum = sky_spectrum

    RSS.history.append('- 1D sky spectrum provided in file :')
    RSS.history.append('  ' + sky_spectrum_file)

    if np.nanmedian(RSS.wavelength - w_sky) != 0:
        if verbose or warnings: print(
            "\n\n  WARNING !!!! The wavelengths provided on the sky file do not match the wavelengths on this RSS !!\n\n")
        RSS.history.append(
            '  WARNING: The wavelengths provided on the sky file do not match the wavelengths on this RSS')
    return sky_spectrum

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------


def apply_RSS_sky(RSS, sky_fibres=[], sky_spectrum=[], sky_spectrum_file="", path="",
                   sky_wave_min=0, sky_wave_max=0, win_sky=0, scale_sky_1D=0,
                   brightest_line="Ha", brightest_line_wavelength=0, ranges_with_emission_lines=[0],
                   cut_red_end=0, low_fibres=10, use_fit_for_negative_sky=False, kernel_negative_sky=51,
                   order_fit_negative_sky=3, n_sky=50, correct_negative_sky=False,
                   individual_check=False, verbose=True, plot=True):
    """

    Apply sky correction using the specified number of lowest fibres in the RSS file to obtain the sky spectrum

    Parameters
    ----------
    sky_fibres : list of integers (default = none)
        Specify the fibres to use to obtain sky spectrum. Will automatically determine the best fibres if not specified
    sky_spectrum : list of floats (default = none)
        Specify the sky spectrum to be used for correction. If not specified, will derive it automatically
    sky_spectrum_file : string (default = None)
        Specify the name of sky spectrum file (including or not the path)
    path: string (default = "")
        path to the sky spectrum file
    plot : boolean (default = True)
        Show the plots in the console
    sky_wave_min : float (default = 0)
        Specify the lower bound on wavelength range. If 0, it is set to RSS.valid_wave_min
    sky_wave_max : float (default = 0)
        Specify the upper bound on wavelength range. If 0, it is set to RSS.valid_wave_max
    win_sky : odd integer (default = 0)
        Width in fibres of a median filter applied to obtain sky spectrum, if 0, it will not apply any median filter
    scale_sky_1D : float (default = 0)
        Specify the scale between the sky emission and the object, if 0, will find it automatically
    brightest_line : string (default = "Ha")
        Specify the brightest emission line in the object spectrum, by default it is H-alpha
        Options: “O3”: [OIII] 5007, “O3b”: [OIII] 4959, “Ha”: H-alpha 6563, “Hb”: H-beta 4861.
    brightest_line_wavelength : float (default = 0)
        Wavelength of the brightest emission line, if 0, will take a stored value for emission line specified
    ranges_with_emission_lines = list of floats (default = [0])
        Specify ranges containing emission lines than needs to be corrected
    cut_red_end : float (default = 0)
        Apply mask to the red end of the spectrum. If 0, will proceed, if -1, will do nothing
    low_fibres : integer (default = 10)
        amount of fibres allocated to act as fibres with the lowest intensity
    use_fit_for_negative_sky: boolean (default = False)
        Substract the order-order fit instead of the smoothed median spectrum
    kernel_negative_sky : odd integer (default = 51)
        kernel parameter for smooth median spectrum
    order_fit_negative_sky : integer (default = 3)
        order of polynomial used for smoothening and fitting the spectrum
    verbose : boolean (default = True)
        Print detailed description of steps taken in console
    n_sky : integer (default = 50)
        Number of fibres to use for finding sky spectrum
    correct_negative_sky : boolean (default = True)
        If True, and if the integrated value of the median sky is negative, this is corrected
    individual_check: boolean (default = True)
        Check individual fibres and correct if integrated value is negative
    """

    RSS.history.append('- Sky sustraction using the RSS method')

    if len(sky_fibres) != 0:
        n_sky = len(sky_fibres)
        print("\n> 'sky_method = RSS', using list of", n_sky, "fibres to create a sky spectrum ...")
        RSS.history.append('  A list of ' + np.str(n_sky) + ' fibres was provided to create the sky spectrum')
        RSS.history.append(np.str(sky_fibres))
    else:
        print("\n> 'sky_method = RSS', hence using", n_sky, "lowest intensity fibres to create a sky spectrum ...")
        RSS.history.append(
            '  The ' + np.str(n_sky) + ' lowest intensity fibres were used to create the sky spectrum')

    if sky_spectrum_file != "":
        sky_spectrum = RSS.read_sky_spectrum(sky_spectrum_file, path=path, verbose=verbose)

    if len(sky_spectrum) == 0:
        RSS.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                               sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max,
                               win_sky=win_sky, include_history=True)

    else:
        print("  Sky spectrum provided. Using this for replacing regions with bright emission lines...")

        RSS.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                               sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max,
                               win_sky=win_sky, include_history=False)

        sky_r_RSS = RSS.sky_emission

        RSS.sky_emission = replace_el_in_sky_spectrum(RSS, sky_r_RSS, sky_spectrum, scale_sky_1D=scale_sky_1D,
                                                       brightest_line=brightest_line,
                                                       brightest_line_wavelength=brightest_line_wavelength,
                                                       ranges_with_emission_lines=ranges_with_emission_lines,
                                                       cut_red_end=cut_red_end,
                                                       plot=plot)
        RSS.history.append('  Using sky spectrum provided for replacing regions with emission lines')

    RSS.substract_sky(plot=plot, low_fibres=low_fibres,
                       correct_negative_sky=correct_negative_sky, use_fit_for_negative_sky=use_fit_for_negative_sky,
                       kernel_negative_sky=kernel_negative_sky, order_fit_negative_sky=order_fit_negative_sky,
                       individual_check=individual_check)

    RSS.apply_mask(verbose=verbose, make_nans=True)
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------


def apply_1D_sky(RSS, sky_fibres=[], sky_spectrum=[], sky_wave_min=0, sky_wave_max=0,
                 win_sky=0, include_history=True,
                 scale_sky_1D=0, remove_5577=True, sky_spectrum_file="", path="",
                 plot=True, verbose=True, n_sky=50):
    """

    Apply sky correction using 1D spectrum provided

    Parameters
    ----------
    sky_fibres : list of integers (default = none)
        Specify the fibres to use to obtain sky spectrum. Will automatically determine the best fibres if not specified
    sky_spectrum : list of floats (default = none)
        Specify the sky spectrum to be used for correction. If not specified, will derive it automatically
    sky_wave_min : float (default = 0)
        Specify the lower bound on wavelength range. If 0, it is set to RSS.valid_wave_min
    sky_wave_max : float (default = 0)
        Specify the upper bound on wavelength range. If 0, it is set to RSS.valid_wave_max
    win_sky : odd integer (default = 0)
        Width in fibres of a median filter applied to obtain sky spectrum, if 0, it will not apply any median filter
    scale_sky_1D : float (default = 0)
        Specify the scale between the sky emission and the object, if 0, will find it automatically
    include_history : boolean (default = True)
        Include the task completion into the RSS object history
    remove_5577 : boolean (default = True)
        Remove the line 5577 from the data
    sky_spectrum_file : string (default = None)
        Specify the name of sky spectrum file (including or not the path)
    path: string (default = "")
        path to the sky spectrum file
    plot : boolean (default = True)
        Show the plots in the console
    verbose : boolean (default = True)
        Print detailed description of steps taken in console
    n_sky : integer (default = 50)
        Number of fibres to use for finding sky spectrum
    """

    RSS.history.append('- Sky sustraction using the 1D method')

    if sky_spectrum_file != "":
        sky_spectrum = RSS.read_sky_spectrum(sky_spectrum_file, path=path, verbose=verbose)

    if verbose:
        print("\n> Sustracting the sky using the sky spectrum provided, checking the scale OBJ/SKY...")
    if scale_sky_1D == 0:
        if verbose:
            print("  No scale between 1D sky spectrum and object given, calculating...")

        # TODO !
        # Task "scale_sky_spectrum" uses sky lines, needs to be checked...
        # RSS.sky_emission,scale_sky_1D_auto=scale_sky_spectrum(RSS.wavelength, sky_spectrum, RSS.intensity_corrected,
        #                                     cut_sky=cut_sky, fmax=fmax, fmin=fmin, fibre_list=fibre_list)

        # Find RSS sky emission using only the lowest n_sky fibres (this should be small, 20-25)
        if n_sky == 50: n_sky = 20
        RSS.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                               sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max,
                               win_sky=win_sky, include_history=include_history)

        sky_r_RSS = RSS.sky_emission

        scale_sky_1D = auto_scale_two_spectra(RSS, sky_r_RSS, sky_spectrum, scale=[0.1, 1.01, 0.025],
                                              w_scale_min=RSS.valid_wave_min, w_scale_max=RSS.valid_wave_max,
                                              plot=plot, verbose=True)

    elif verbose:
        print("  As requested, we scale the given 1D sky spectrum by", scale_sky_1D)

    RSS.sky_emission = sky_spectrum * scale_sky_1D
    RSS.history.append('  1D sky spectrum scaled by =' + np.str(scale_sky_1D))

    if verbose: print("\n> Scaled sky spectrum stored in RSS.sky_emission, substracting to all fibres...")

    # For blue spectra, remove 5577 in the sky spectrum...
    if RSS.valid_wave_min < 5577 and remove_5577 == True:
        if verbose: print("  Removing sky line 5577.34 from the sky spectrum...")
        resultado = fluxes(RSS.wavelength, RSS.sky_emission, 5577.34, lowlow=30, lowhigh=10, highlow=10,
                           highhigh=30,
                           plot=False, verbose=False)  # fmin=-5.0E-17, fmax=2.0E-16,
        # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
        RSS.sky_emission = resultado[11]
    else:
        if RSS.valid_wave_min < 5577 and verbose: print(
            "  Sky line 5577.34 is not removed from the sky spectrum...")

    # Remove 5577 in the object
    if RSS.valid_wave_min < 5577 and remove_5577 == True and scale_sky_1D == 0:  # and individual_sky_substraction == False:
        if verbose:
            print("  Removing sky line 5577.34 from the object...")
        RSS.history.append("  Sky line 5577.34 removed performing Gaussian fit")

        wlm = RSS.wavelength
        for i in range(RSS.n_spectra):
            s = RSS.intensity_corrected[i]
            # Removing Skyline 5577 using Gaussian fit if requested
            resultado = fluxes(wlm, s, 5577.34, lowlow=30, lowhigh=10, highlow=10, highhigh=30,
                               plot=False, verbose=False)  # fmin=-5.0E-17, fmax=2.0E-16,
            # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
            RSS.intensity_corrected[i] = resultado[11]
    else:
        if RSS.valid_wave_min < 5577 and verbose:
            if scale_sky_1D == 0:
                print("  Sky line 5577.34 is not removed from the object...")
            else:
                print("  Sky line 5577.34 already removed in object during CCD cleaning...")

    RSS.substract_sky(plot=plot, verbose=verbose)

    if plot:
        text = "Sky spectrum (scaled using a factor " + np.str(scale_sky_1D) + " )"
        plot_plot(RSS.wavelength, RSS.sky_emission, hlines=[0], ptitle=text,
                  xmin=RSS.wavelength[0] - 10, xmax=RSS.wavelength[-1] + 10, color="c",
                  vlines=[RSS.valid_wave_min, RSS.valid_wave_max])
    if verbose:
        print("  Intensities corrected for sky emission and stored in RSS.intensity_corrected !")
    RSS.sky_emission = sky_spectrum  # Restore sky_emission to original sky_spectrum
    # RSS.apply_mask(verbose=verbose)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------


def apply_1Dfit_sky(RSS, sky_spectrum=[], n_sky=50, sky_fibres=[], sky_spectrum_file="", path="",
                    sky_wave_min=0, sky_wave_max=0, win_sky=0, scale_sky_1D=0,
                    sky_lines_file="", brightest_line_wavelength=0,
                    brightest_line="Ha", maxima_sigma=3, auto_scale_sky=False,
                    plot=True, verbose=True, fig_size=12, fibre_p=-1, kernel_correct_ccd_defects=51):
    """
    Apply 1Dfit sky correction.

    Parameters
    ----------
    sky_spectrum : array or list of floats (default = none)
        Specify the sky spectrum to be used for correction. If not specified, will derive it automatically
    n_sky : integer (default = 50)
        Number of fibres to use for finding sky spectrum
    sky_fibres : list of integers (default = none)
        Specify the fibres to use to obtain sky spectrum. Will automatically determine the best fibres if not specified
    sky_spectrum_file : string (default = None)
        Specify the name of sky spectrum file (including or not the path)
    path: string (default = "")
        path to the sky spectrum file
    sky_wave_min : float (default = 0)
        Specify the lower bound on wavelength range. If 0, it is set to RSS.valid_wave_min
    sky_wave_max : float (default = 0)
        Specify the upper bound on wavelength range. If 0, it is set to RSS.valid_wave_max
    win_sky : odd integer (default = 0)
        Width in fibres of a median filter applied to obtain sky spectrum, if 0, it will not apply any median filter
    scale_sky_1D : float (default = 0)
        Specify the scale between the sky emission and the object, if 0, will find it automatically
    sky_lines_file : string (default = None)
        Specify the path and name of sky lines file
    brightest_line_wavelength : float (default = 0)
        Wavelength of the brightest emission line, if 0, will take a stored value for emission line specified
    brightest_line : string (default = "Ha")
        Specify the brightest emission line in the object spectrum, by default it is H-alpha
        Options: “O3”: [OIII] 5007, “O3b”: [OIII] 4959, “Ha”: H-alpha 6563, “Hb”: H-beta 4861.
    maxima_sigma : float (default = 3)
        Maximum allowed standard deviation for Gaussian fit
    auto_scale_sky : boolean (default = False)
        Scales sky spectrum for subtraction if True
    plot : boolean (default = True)
        Show the plots in the console
    verbose : boolean (default = True)
        Print detailed description of steps taken in console
    fig_size : integer (default = 12)
        Size of the image plotted
    fibre_p: integer (default = -1)
        if fibre_p=fibre only corrects that fibre and plots the corrections, if -1, applies correction to all fibres
    kernel_correct_ccd_defects : odd integer (default = 51)
        width used for the median filter
    """
    RSS.history.append('- Sky sustraction using the 1Dfit method')

    if sky_spectrum_file != "":
        sky_spectrum = RSS.read_sky_spectrum(sky_spectrum_file, path=path, verbose=verbose)

    if verbose:
        print("\n> Fitting sky lines in both a provided sky spectrum AND all the fibres")
        print("  This process takes ~20 minutes for 385R if all skylines are considered!\n")
    if len(sky_spectrum) == 0:
        if verbose:
            print("  No sky spectrum provided, using", n_sky, "lowest intensity fibres to create a sky...")
        RSS.history.append('  ERROR! No sky spectrum provided, using RSS method with n_sky =' + np.str(n_sky))
        RSS.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                               sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, win_sky=win_sky)
    else:
        if scale_sky_1D != 0:
            RSS.history.append('  1D sky spectrum scaled by =' + np.str(scale_sky_1D))
            if verbose:
                print("  1D sky spectrum scaled by ", scale_sky_1D)
        else:
            if verbose:
                print("  No scale between 1D sky spectrum and object given, calculating...")
            if n_sky == 50: n_sky = 20
            RSS.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                                   sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max,
                                   win_sky=win_sky, include_history=False)

            sky_r_RSS = RSS.sky_emission

            scale_sky_1D = auto_scale_two_spectra(RSS, sky_r_RSS, sky_spectrum, scale=[0.1, 1.01, 0.025],
                                                  w_scale_min=RSS.valid_wave_min, w_scale_max=RSS.valid_wave_max,
                                                  plot=plot, verbose=True)

        RSS.history.append('  1D sky spectrum scaled by =' + np.str(scale_sky_1D))

        RSS.sky_emission = np.array(sky_spectrum) * scale_sky_1D

    RSS.fit_and_substract_sky_spectrum(RSS.sky_emission, sky_lines_file=sky_lines_file,
                                        brightest_line_wavelength=brightest_line_wavelength,
                                        brightest_line=brightest_line,
                                        maxima_sigma=maxima_sigma, ymin=-50, ymax=600, wmin=0, wmax=0,
                                        auto_scale_sky=auto_scale_sky,
                                        warnings=False, verbose=False, plot=False, fig_size=fig_size, fibre=fibre_p)

    if fibre_p == -1:
        if verbose:
            print("\n> 1Dfit sky_method usually generates some nans, correcting ccd defects again...")
        RSS.correct_ccd_defects(kernel_correct_ccd_defects=kernel_correct_ccd_defects, verbose=verbose, plot=plot,
                                 only_nans=True)  # Not replacing values <0
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------


def apply_RSSfit_sky(RSS, sky_spectrum=[], n_sky=50, sky_fibres=[], sky_spectrum_file="", path="",
                      sky_wave_min=0, sky_wave_max=0, win_sky=0, scale_sky_1D=0,
                      sky_lines_file="", brightest_line_wavelength=0,
                      ranges_with_emission_lines=[0],
                      cut_red_end=0,
                      brightest_line="Ha", maxima_sigma=3, auto_scale_sky=False,
                      fibre_p=-1, kernel_correct_ccd_defects=51,
                      plot=True, verbose=True, fig_size=12):
    """
    Subtract sky using the RSSfit method.

    Parameters
    ----------
    sky_spectrum : TYPE, optional
        DESCRIPTION. The default is [].
    n_sky : TYPE, optional
        DESCRIPTION. The default is 50.
    sky_fibres : TYPE, optional
        DESCRIPTION. The default is [].
    sky_spectrum_file : TYPE, optional
        DESCRIPTION. The default is "".
    path : TYPE, optional
        DESCRIPTION. The default is "".
    sky_wave_min : TYPE, optional
        DESCRIPTION. The default is 0.
    sky_wave_max : TYPE, optional
        DESCRIPTION. The default is 0.
    win_sky : TYPE, optional
        DESCRIPTION. The default is 0.
    scale_sky_1D : TYPE, optional
        DESCRIPTION. The default is 0.
    sky_lines_file : TYPE, optional
        DESCRIPTION. The default is "".
    brightest_line_wavelength : TYPE, optional
        DESCRIPTION. The default is 0.
    ranges_with_emission_lines : TYPE, optional
        DESCRIPTION. The default is [0].
    cut_red_end : TYPE, optional
        DESCRIPTION. The default is 0.
    brightest_line : TYPE, optional
        DESCRIPTION. The default is "Ha".
    maxima_sigma : TYPE, optional
        DESCRIPTION. The default is 3.
    auto_scale_sky : TYPE, optional
        DESCRIPTION. The default is False.
    fibre_p : TYPE, optional
        DESCRIPTION. The default is -1.
    kernel_correct_ccd_defects : TYPE, optional
        DESCRIPTION. The default is 51.
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.
    fig_size : TYPE, optional
        DESCRIPTION. The default is 12.


    """

    RSS.history.append('- Sky sustraction using the RSSfit method')

    if verbose: print("\n> 'sky_method = RSSfit', hence using", n_sky,
                      "lowest intensity fibres to create a sky spectrum ...")

    RSS.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                           sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max,
                           win_sky=win_sky, include_history=True)

    if sky_spectrum_file != "":
        sky_spectrum = RSS.read_sky_spectrum(sky_spectrum_file, path=path, verbose=verbose)

    if sky_spectrum[0] != -1 and np.nanmedian(sky_spectrum) != 0:
        if verbose: print(
            "\n> Additional sky spectrum provided. Using this for replacing regions with bright emission lines...")

        sky_r_RSS = RSS.sky_emission

        RSS.sky_emission = replace_el_in_sky_spectrum(RSS, sky_r_RSS, sky_spectrum,
                                                       scale_sky_1D=scale_sky_1D,
                                                       brightest_line=brightest_line,
                                                       brightest_line_wavelength=brightest_line_wavelength,
                                                       ranges_with_emission_lines=ranges_with_emission_lines,
                                                       cut_red_end=cut_red_end,
                                                       plot=plot)
        RSS.history.append('  Using sky spectrum provided for replacing regions with emission lines')

    RSS.fit_and_substract_sky_spectrum(RSS.sky_emission, sky_lines_file=sky_lines_file,
                                        brightest_line_wavelength=brightest_line_wavelength,
                                        brightest_line=brightest_line,
                                        maxima_sigma=maxima_sigma, ymin=-50, ymax=600, wmin=0, wmax=0,
                                        auto_scale_sky=auto_scale_sky,
                                        warnings=False, verbose=False, plot=False, fig_size=fig_size,
                                        fibre=fibre_p)

    if fibre_p == -1:
        if verbose: print("\n> 'RSSfit' sky_method usually generates some nans, correcting ccd defects again...")
        RSS.correct_ccd_defects(kernel_correct_ccd_defects=kernel_correct_ccd_defects, verbose=verbose,
                                 plot=plot, only_nans=True)  # not replacing values < 0
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------


def apply_2D_sky(RSS, sky_rss, scale_sky_rss=0,
                 plot=True, verbose=True, fig_size=12):
    """
    Task that uses a RSS file with a sky, scale and sustract it.
    #TODO: this method needs to be checked and use plot_plot in plots

    Parameters
    ----------
    sky_rss : OBJECT #TODO This needs to be also a file
        A RSS file with offset sky.
    scale_sky_rss : float, optional
        scale applied to the SKY RSS before sustracting
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.
    fig_size : TYPE, optional
        DESCRIPTION. The default is 12.
    """
    RSS.history.append('- Sky sustraction using the 2D method')

    if scale_sky_rss != 0:
        if verbose: print("\n> Using sky image provided to substract sky, considering a scale of",
                          scale_sky_rss, "...")
        RSS.sky_emission = scale_sky_rss * sky_rss.intensity_corrected
        RSS.intensity_corrected = RSS.intensity_corrected - RSS.sky_emission
    else:
        if verbose: print(
            "\n> Using sky image provided to substract sky, computing the scale using sky lines")
        # check scale fibre by fibre
        RSS.sky_emission = copy.deepcopy(sky_rss.intensity_corrected)
        scale_per_fibre = np.ones((RSS.n_spectra))
        scale_per_fibre_2 = np.ones((RSS.n_spectra))
        lowlow = 15
        lowhigh = 5
        highlow = 5
        highhigh = 15
        if RSS.grating == "580V":
            if verbose: print("  For 580V we use bright skyline at 5577 AA ...")
            sky_line = 5577
            sky_line_2 = 0
        if RSS.grating == "1000R":
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
        for fibre_sky in range(RSS.n_spectra):
            skyline_spec = fluxes(RSS.wavelength, RSS.intensity_corrected[fibre_sky], sky_line,
                                  plot=False, verbose=False, lowlow=lowlow, lowhigh=lowhigh,
                                  highlow=highlow, highhigh=highhigh)  # fmin=-5.0E-17, fmax=2.0E-16,
            # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
            RSS.intensity_corrected[fibre_sky] = skyline_spec[11]

            skyline_sky = fluxes(RSS.wavelength, RSS.sky_emission[fibre_sky], sky_line, plot=False,
                                 verbose=False, lowlow=lowlow, lowhigh=lowhigh, highlow=highlow,
                                 highhigh=highhigh)  # fmin=-5.0E-17, fmax=2.0E-16,

            scale_per_fibre[fibre_sky] = skyline_spec[3] / skyline_sky[3]
            RSS.sky_emission[fibre_sky] = skyline_sky[11]

        if sky_line_2 != 0:
            if verbose: print("  ... now checking", sky_line_2, "...")
            for fibre_sky in range(RSS.n_spectra):
                skyline_spec = fluxes(RSS.wavelength, RSS.intensity_corrected[fibre_sky], sky_line_2,
                                      plot=False, verbose=False, lowlow=lowlow_2, lowhigh=lowhigh_2,
                                      highlow=highlow_2,
                                      highhigh=highhigh_2)  # fmin=-5.0E-17, fmax=2.0E-16,
                # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                RSS.intensity_corrected[fibre_sky] = skyline_spec[11]

                skyline_sky = fluxes(RSS.wavelength, RSS.sky_emission[fibre_sky], sky_line_2, plot=False,
                                     verbose=False, lowlow=lowlow_2, lowhigh=lowhigh_2, highlow=highlow_2,
                                     highhigh=highhigh_2)  # fmin=-5.0E-17, fmax=2.0E-16,

                scale_per_fibre_2[fibre_sky] = skyline_spec[3] / skyline_sky[3]
                RSS.sky_emission[fibre_sky] = skyline_sky[11]

                # Median value of scale_per_fibre, and apply that value to all fibres
        if sky_line_2 == 0:
            scale_sky_rss = np.nanmedian(scale_per_fibre)
            RSS.sky_emission = RSS.sky_emission * scale_sky_rss
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

            for i in range(RSS.n_wave):
                RSS.sky_emission[:, i] = RSS.sky_emission[:, i] * (a + b * RSS.wavelength[i])

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
        RSS.intensity_corrected = RSS.intensity_corrected - RSS.sky_emission
    RSS.apply_mask(verbose=verbose)
    RSS.history(" - 2D sky subtraction performed")
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------


def is_sky(RSS, n_sky=50, win_sky=0, sky_fibres=[], sky_wave_min=0,
           sky_wave_max=0, plot=True, verbose=True):
    """
    If frame is an empty sky, apply median filter to it for further use in 2D sky emission fitting

    Parameters
    ----------
    n_sky : integer (default = 50)
        Number of fibres to use for finding sky spectrum
    sky_fibres : list of integers (default = none)
        Specify the fibres to use to obtain sky spectrum. Will automatically determine the best fibres if not specified
    sky_wave_min : float (default = 0)
        Specify the lower bound on wavelength range. If 0, it is set to RSS.valid_wave_min
    sky_wave_max : float (default = 0)
        Specify the upper bound on wavelength range. If 0, it is set to RSS.valid_wave_max
    win_sky : odd integer (default = 0)
        Width in fibres of a median filter applied to obtain sky spectrum, if 0, it will not apply any median filter
     plot : boolean (default = True)
        Show the plots in the console
    verbose : boolean (default = True)
        Print detailed description of steps taken in console
    """

    if verbose: print("\n> This RSS file is defined as SKY... identifying", n_sky,
                      " lowest fibres for getting 1D sky spectrum...")
    RSS.history.append('- This RSS file is defined as SKY:')
    RSS.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                           sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, win_sky=0)
    # print "\n> This RSS file is defined as SKY... applying median filter with window",win_sky,"..."
    if win_sky == 0:  # Default when it is not a win_sky
        win_sky = 151
    print("\n  ... applying median filter with window", win_sky, "...\n")

    medfilt_sky = median_2D_filter(RSS.intensity_corrected, RSS.n_spectra, RSS.n_wave, win_sky=win_sky)
    RSS.intensity_corrected = copy.deepcopy(medfilt_sky)
    print("  Median filter applied, results stored in RSS.intensity_corrected !")
    RSS.history.append('  Median filter ' + np.str(win_sky) + ' applied to all fibres')


def obtain_sky_spectrum(sky, low_fibres=200, plot=True, fig_size=12, fcal=False, verbose=True):
    """
    This uses the lowest low_fibres fibres to get an integrated spectrum
    """
    integrated_intensity_sorted = np.argsort(sky.integrated_fibre)
    region = []
    for fibre in range(low_fibres):
        region.append(integrated_intensity_sorted[fibre])
    sky_spectrum = np.nanmedian(sky.intensity_corrected[region], axis=0)

    if verbose:
        print("  We use the ", low_fibres, " fibres with the lowest integrated intensity to derive the sky spectrum")
        print("  The list is = ", region)

    if plot:
        plt.figure(figsize=(fig_size, fig_size / 2.5))
        plt.plot(sky.wavelength, sky_spectrum)
        ptitle = "Sky spectrum"
        plot_plot(sky.wavelength, sky_spectrum, ptitle=ptitle, fcal=fcal)

    return sky_spectrum


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def scale_sky_spectrum(wlm, sky_spectrum, spectra, cut_sky=4., fmax=10, fmin=1, valid_wave_min=0, valid_wave_max=0,
                       fibre_list=[100, 200, 300, 400, 500, 600, 700, 800, 900], plot=True, verbose=True,
                       warnings=True):
    """
    This task needs to be checked.
    Using the continuum, the scale between 2 spectra can be determined runnning
    auto_scale_two_spectra()
    """

    # # Read sky lines provided by 2dFdr
    #    sky_line_,flux_sky_line_ = read_table("sky_lines_2dfdr.dat", ["f", "f"] )
    # # Choose those lines in the range
    #    sky_line=[]
    #    flux_sky_line=[]
    #    valid_wave_min = 6240
    #    valid_wave_max = 7355
    #    for i in range(len(sky_line_)):
    #        if valid_wave_min < sky_line_[i] < valid_wave_max:
    #            sky_line.append(sky_line_[i])
    #            flux_sky_line.append(flux_sky_line_[i])

    if valid_wave_min == 0: valid_wave_min = wlm[0]
    if valid_wave_max == 0: valid_wave_max = wlm[-1]

    if verbose: print("\n> Identifying sky lines using cut_sky =", cut_sky, ", allowed SKY/OBJ values = [", fmin, ",",
                      fmax, "]")
    if verbose: print("  Using fibres = ", fibre_list)

    peaks, peaks_name, peaks_rest, continuum_limits = search_peaks(wlm, sky_spectrum, plot=plot, cut=cut_sky, fmax=fmax,
                                                                   only_id_lines=False, verbose=False)

    ratio_list = []
    valid_peaks = []

    if verbose: print("\n      Sky line     Gaussian ratio      Flux ratio")
    n_sky_lines_found = 0
    for i in range(len(peaks)):
        sky_spectrum_data = fluxes(wlm, sky_spectrum, peaks[i], fcal=False, lowlow=50, highhigh=50, plot=False,
                                   verbose=False, warnings=False)

        sky_median_continuum = np.nanmedian(sky_spectrum_data[11])

        object_spectrum_data_gauss = []
        object_spectrum_data_integrated = []
        median_list = []
        for fibre in fibre_list:
            object_spectrum_flux = fluxes(wlm, spectra[fibre], peaks[i], fcal=False, lowlow=50, highhigh=50, plot=False,
                                          verbose=False, warnings=False)
            object_spectrum_data_gauss.append(object_spectrum_flux[3])  # Gaussian flux is 3
            object_spectrum_data_integrated.append(object_spectrum_flux[7])  # integrated flux is 7
            median_list.append(np.nanmedian(object_spectrum_flux[11]))
        object_spectrum_data = np.nanmedian(object_spectrum_data_gauss)
        object_spectrum_data_i = np.nanmedian(object_spectrum_data_integrated)

        object_median_continuum = np.nanmin(median_list)

        if fmin < object_spectrum_data / sky_spectrum_data[3] * sky_median_continuum / object_median_continuum < fmax:
            n_sky_lines_found = n_sky_lines_found + 1
            valid_peaks.append(peaks[i])
            ratio_list.append(object_spectrum_data / sky_spectrum_data[3])
            if verbose: print(
                "{:3.0f}   {:5.3f}         {:2.3f}             {:2.3f}".format(n_sky_lines_found, peaks[i],
                                                                               object_spectrum_data / sky_spectrum_data[
                                                                                   3], object_spectrum_data_i /
                                                                               sky_spectrum_data[7]))

            # print "ratio_list =", ratio_list
    # fit = np.polyfit(valid_peaks, ratio_list, 0) # This is the same that doing an average/mean
    # fit_line = fit[0]+0*wlm
    fit_line = np.nanmedian(ratio_list)  # We just do a median
    # fit_line = fit[1]+fit[0]*wlm
    # fit_line = fit[2]+fit[1]*wlm+fit[0]*wlm**2
    # fit_line = fit[3]+fit[2]*wlm+fit[1]*wlm**2+fit[0]*wlm**3

    if plot:
        plt.plot(valid_peaks, ratio_list, "+")
        # plt.plot(wlm,fit_line)
        plt.axhline(y=fit_line, color='k', linestyle='--')
        plt.xlim(valid_wave_min - 10, valid_wave_max + 10)
        # if len(ratio_list) > 0:
        plt.ylim(np.nanmin(ratio_list) - 0.2, np.nanmax(ratio_list) + 0.2)
        plt.title("Scaling sky spectrum to object spectra")
        plt.xlabel("Wavelength [$\mathrm{\AA}$]")
        plt.ylabel("OBJECT / SKY")
        plt.minorticks_on()
        plt.show()
        plt.close()

        if verbose: print("  Using this fit to scale sky spectrum to object, the median value is ",
                          np.round(fit_line, 3), "...")

    sky_corrected = sky_spectrum * fit_line

    #        plt.plot(wlm,sky_spectrum, "r", alpha=0.3)
    #        plt.plot(wlm,sky_corrected, "g", alpha=0.3)
    #        plt.show()
    #        plt.close()

    return sky_corrected, np.round(fit_line, 3)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def sky_spectrum_from_fibres_using_file(rss_file, path="", instrument="",
                                        fibre_list=[], win_sky=151, n_sky=0,
                                        skyflat="", apply_throughput=True, correct_ccd_defects=False,
                                        fix_wavelengths=False, sol=[0, 0, 0], xmin=0, xmax=0, ymin=0, ymax=0,
                                        verbose=True, plot=True):
    if skyflat == "":
        apply_throughput = False
        plot_rss = False
    else:
        apply_throughput = True
        plot_rss = True

    if n_sky != 0:
        sky_method = "self"
        is_sky = False
        if verbose: print("\n> Obtaining 1D sky spectrum using ", n_sky, " lowest fibres in this rss ...")
    else:
        sky_method = "none"
        is_sky = True
        if verbose: print("\n> Obtaining 1D sky spectrum using fibre list = ", fibre_list, " ...")

    _test_rss_ = RSS(rss_file, path=path, instrument=instrument)
    _test_rss_.process_rss(apply_throughput=apply_throughput, skyflat=skyflat,
                           correct_ccd_defects=correct_ccd_defects,
                           fix_wavelengths=fix_wavelengths, sol=sol,
                           sky_method=sky_method, n_sky=n_sky, is_sky=is_sky, win_sky=win_sky,
                           do_extinction=False, plot=plot_rss, verbose=False)

    if n_sky != 0:
        print("\n> Sky fibres used: ", _test_rss_.sky_fibres)
        sky = _test_rss_.sky_emission
    else:
        sky = _test_rss_.plot_combined_spectrum(list_spectra=fibre_list, median=True)

    if plot:
        plt.figure(figsize=(14, 4))
        if n_sky != 0:
            plt.plot(_test_rss_.wavelength, sky, "b", linewidth=2, alpha=0.5)
            ptitle = "Sky spectrum combining using " + np.str(n_sky) + " lowest fibres"

        else:
            for i in range(len(fibre_list)):
                plt.plot(_test_rss_.wavelength, _test_rss_.intensity_corrected[i], alpha=0.5)
                plt.plot(_test_rss_.wavelength, sky, "b", linewidth=2, alpha=0.5)
            ptitle = "Sky spectrum combining " + np.str(len(fibre_list)) + " fibres"

        plot_plot(_test_rss_.wavelength, sky, ptitle=ptitle)

    print("\n> Sky spectrum obtained!")
    return sky