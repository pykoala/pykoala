import numpy as np
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------------
# GENERAL TASKS
# -----------------------------------------------------------------------------

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def cumulaive_Moffat(r2, L_star, alpha2, beta):
    return L_star * (1 - np.power(1 + (r2 / alpha2), -beta))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fit_Moffat(r2_growth_curve, F_growth_curve,
               F_guess, r2_half_light, r_max, plot=False):
    """
    Fits a Moffat profile to a flux growth curve
    as a function of radius squared,
    cutting at to r_max (in units of the half-light radius),
    provided an initial guess of the total flux and half-light radius squared.
    """
    index_cut = np.searchsorted(r2_growth_curve, r2_half_light * r_max ** 2)
    fit, cov = optimize.curve_fit(cumulaive_Moffat,
                                  r2_growth_curve[:index_cut], F_growth_curve[:index_cut],
                                  p0=(F_guess, r2_half_light, 1)
                                  )
    if plot:
        print("Best-fit: L_star =", fit[0])
        print("          alpha =", np.sqrt(fit[1]))
        print("          beta =", fit[2])
        r_norm = np.sqrt(np.array(r2_growth_curve) / r2_half_light)
        plt.plot(r_norm, cumulaive_Moffat(np.array(r2_growth_curve),
                                          fit[0], fit[1], fit[2]) / fit[0], ':')
    return fit

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def KOALA_offsets(s, pa):
    print("\n> Offsets towards North and East between pointings," \
          "according to KOALA manual, for pa =", pa, "degrees")
    pa *= np.pi / 180
    print("  a -> b :", s * np.sin(pa), -s * np.cos(pa))
    print("  a -> c :", -s * np.sin(60 - pa), -s * np.cos(60 - pa))
    print("  b -> d :", -np.sqrt(3) * s * np.cos(pa), -np.sqrt(3) * s * np.sin(pa))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def offset_between_cubes(cube1, cube2, plot=True):
    x = (cube2.x_peak - cube2.n_cols / 2. + cube2.RA_centre_deg * 3600. / cube2.pixel_size_arcsec) \
        - (cube1.x_peak - cube1.n_cols / 2. + cube1.RA_centre_deg * 3600. / cube1.pixel_size_arcsec)
    y = (cube2.y_peak - cube2.n_rows / 2. + cube2.DEC_centre_deg * 3600. / cube2.pixel_size_arcsec) \
        - (cube1.y_peak - cube1.n_rows / 2. + cube1.DEC_centre_deg * 3600. / cube1.pixel_size_arcsec)
    delta_RA_pix = np.nanmedian(x)
    delta_DEC_pix = np.nanmedian(y)
    delta_RA_arcsec = delta_RA_pix * cube1.pixel_size_arcsec
    delta_DEC_arcsec = delta_DEC_pix * cube1.pixel_size_arcsec
    print('(delta_RA, delta_DEC) = ({:.3f}, {:.3f}) arcsec' \
          .format(delta_RA_arcsec, delta_DEC_arcsec))
    if plot:
        x -= delta_RA_pix
        y -= delta_DEC_pix
        smooth_x = signal.medfilt(x, 151)
        smooth_y = signal.medfilt(y, 151)

        print(np.nanmean(smooth_x))
        print(np.nanmean(smooth_y))

        plt.figure(figsize=(10, 5))
        wl = cube1.RSS.wavelength
        plt.plot(wl, x, 'k.', alpha=0.1)
        plt.plot(wl, y, 'r.', alpha=0.1)
        plt.plot(wl, smooth_x, 'k-')
        plt.plot(wl, smooth_y, 'r-')
        #    plt.plot(wl, x_max-np.nanmedian(x_max), 'g-')
        #    plt.plot(wl, y_max-np.nanmedian(y_max), 'y-')
        plt.ylim(-1.6, 1.6)
        plt.show()
        plt.close()
    return delta_RA_arcsec, delta_DEC_arcsec


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def compare_cubes(cube1, cube2, line=0):
    if line == 0:
        map1 = cube1.integrated_map
        map2 = cube2.integrated_map
    else:
        l = np.searchsorted(cube1.RSS.wavelength, line)
        map1 = cube1.data[l]
        map2 = cube2.data[l]

    scale = np.nanmedian(map1 + map2) * 3
    scatter = np.nanmedian(np.nonzero(map1 - map2))

    plt.figure(figsize=(12, 8))
    plt.imshow(map1 - map2, vmin=-scale, vmax=scale, cmap=plt.cm.get_cmap('RdBu'))  # vmin = -scale
    plt.colorbar()
    plt.contour(map1, colors='w', linewidths=2, norm=colors.LogNorm())
    plt.contour(map2, colors='k', linewidths=1, norm=colors.LogNorm())
    if line != 0:
        plt.title("{:.2f} AA".format(line))
    else:
        plt.title("Integrated Map")
    plt.show()
    plt.close()
    print("  Medium scatter : ", scatter)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_response(calibration_star_cubes, scale=[], use_median=False):
    n_cubes = len(calibration_star_cubes)
    if len(scale) == 0:
        for i in range(n_cubes):
            scale.append(1.)

    wavelength = calibration_star_cubes[0].wavelength

    print("\n> Comparing response curve of standard stars...\n")
    for i in range(n_cubes):
        ci = calibration_star_cubes[i].response_curve * scale[i]
        ci_name = calibration_star_cubes[i].object
        for j in range(i + 1, n_cubes):
            cj = calibration_star_cubes[j].response_curve * scale[j]
            cj_name = calibration_star_cubes[j].object
            ptitle = "Comparison of flux calibration for " + ci_name + " and " + cj_name
            ylabel = ci_name + " / " + cj_name
            plot_plot(wavelength, ci / cj, hlines=[0.85, 0.9, 0.95, 1, 1, 1, 1, 1.05, 1.1, 1.15], ymin=0.8, ymax=1.2,
                      ylabel=ylabel, ptitle=ptitle)
    print("\n> Plotting response curve (absolute flux calibration) of standard stars...\n")

    plt.figure(figsize=(11, 8))
    mean_curve = np.zeros_like(wavelength)
    mean_values = []
    list_of_scaled_curves = []
    i = 0
    for star in calibration_star_cubes:
        list_of_scaled_curves.append(star.response_curve * scale[i])
        mean_curve = mean_curve + star.response_curve * scale[i]
        plt.plot(star.wavelength, star.response_curve * scale[i],
                 label=star.description, alpha=0.2, linewidth=2)
        if use_median:
            print("  Median value for ", star.object, " = ", np.nanmedian(star.response_curve * scale[i]),
                  "      scale = ", scale[i])
        else:
            print("  Mean value for ", star.object, " = ", np.nanmean(star.response_curve * scale[i]), "      scale = ",
                  scale[i])
        mean_values.append(np.nanmean(star.response_curve) * scale[i])
        i = i + 1

    mean_curve /= len(calibration_star_cubes)
    median_curve = np.nanmedian(list_of_scaled_curves, axis=0)

    response_rms = np.zeros_like(wavelength)
    for i in range(len(calibration_star_cubes)):
        if use_median:
            response_rms += np.abs(calibration_star_cubes[i].response_curve * scale[i] - median_curve)
        else:
            response_rms += np.abs(calibration_star_cubes[i].response_curve * scale[i] - mean_curve)

    response_rms /= len(calibration_star_cubes)
    if use_median:
        dispersion = np.nansum(response_rms) / np.nansum(median_curve)
    else:
        dispersion = np.nansum(response_rms) / np.nansum(mean_curve)

    if len(calibration_star_cubes) > 1: print("  Variation in flux calibrations =  {:.2f} %".format(dispersion * 100.))

    # dispersion=np.nanmax(mean_values)-np.nanmin(mean_values)
    # print "  Variation in flux calibrations =  {:.2f} %".format(dispersion/np.nanmedian(mean_values)*100.)

    if use_median:
        plt.plot(wavelength, median_curve, "k", label='Median response curve', alpha=0.2, linewidth=10)
    else:
        plt.plot(wavelength, mean_curve, "k", label='mean response curve', alpha=0.2, linewidth=10)
    plt.legend(frameon=False, loc=2)
    plt.ylabel("Flux calibration [ counts /s equivalent to 10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
    plt.xlabel("Wavelength [$\mathrm{\AA}$]")
    plt.title("Response curve for calibration stars")
    plt.minorticks_on()
    plt.show()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def obtain_flux_calibration(calibration_star_cubes):
    #    print "\n> Obtaining flux calibration...\n"
    vector_wave = []
    vector_response = []
    cube_star = calibration_star_cubes[0]
    for i in range(len(cube_star.response_curve)):
        if np.isnan(cube_star.response_curve[i]) == False:
            # vector_wave.append(cube_star.response_wavelength[i])
            vector_wave.append(cube_star.wavelength[i])
            vector_response.append(cube_star.response_curve[i])
            # print "  For wavelength = ",cube_star.response_wavelength[i], " the flux correction is = ", cube_star.response_curve[i]

    interpolated_response = interpolate.splrep(vector_wave, vector_response, s=0)
    flux_calibration = interpolate.splev(cube_star.wavelength, interpolated_response, der=0)
    #    flux_correction = flux_calibration

    print("\n> Flux calibration for all wavelengths = ", flux_calibration)
    print("\n  Flux calibration obtained!")
    return flux_calibration


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def obtain_telluric_correction(w, telluric_correction_list, plot=True, label_stars=[], scale=[]):
    if len(scale) == 0:
        for star in telluric_correction_list: scale.append(1.)

    for i in range(len(telluric_correction_list)):
        telluric_correction_list[i] = [1. if x * scale[i] < 1 else x * scale[i] for x in telluric_correction_list[i]]

    telluric_correction = np.nanmedian(telluric_correction_list, axis=0)
    if plot:
        fig_size = 12
        plt.figure(figsize=(fig_size, fig_size / 2.5))
        plt.title("Telluric correction")
        for i in range(len(telluric_correction_list)):
            if len(label_stars) > 0:
                label = label_stars[i]
            else:
                label = "star" + str(i + 1)
            plt.plot(w, telluric_correction_list[i], alpha=0.3, label=label)
        plt.plot(w, telluric_correction, alpha=0.5, color="k", label="Median")
        plt.minorticks_on()
        plt.legend(frameon=False, loc=2, ncol=1)
        step_up = 1.15 * np.nanmax(telluric_correction)
        plt.ylim(0.9, step_up)
        plt.xlim(w[0] - 10, w[-1] + 10)
        plt.show()
        plt.close()

    print("\n> Telluric correction = ", telluric_correction)
    if np.nanmean(scale) != 1.: print("  Telluric correction scale provided : ", scale)
    print("\n  Telluric correction obtained!")
    return telluric_correction


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def coord_range(rss_list):
    RA = [rss.RA_centre_deg + rss.offset_RA_arcsec / 3600. for rss in rss_list]
    RA_min = np.nanmin(RA)
    RA_max = np.nanmax(RA)
    DEC = [rss.DEC_centre_deg + rss.offset_DEC_arcsec / 3600. for rss in rss_list]
    DEC_min = np.nanmin(DEC)
    DEC_max = np.nanmax(DEC)
    return RA_min, RA_max, DEC_min, DEC_max


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def align_n_cubes(rss_list, cube_list=[0], flux_calibration_list=[[]], pixel_size_arcsec=0.3, kernel_size_arcsec=1.5,
                  offsets=[1000],
                  plot=False, plot_weight=False, plot_tracing_maps=[], plot_spectra=True,
                  ADR=False, jump=-1, ADR_x_fit_list=[0], ADR_y_fit_list=[0], force_ADR=False,
                  half_size_for_centroid=10, box_x=[0, -1], box_y=[0, -1], adr_index_fit=2, g2d=False, step_tracing=100,
                  edgelow=-1, edgehigh=-1, size_arcsec=[], centre_deg=[], warnings=False, verbose=True):
    """
    Routine to align n cubes

    Parameters
    ----------
    Cubes:
        Cubes
    pointings_RSS :
        list with RSS files
    pixel_size_arcsec:
        float, default = 0.3
    kernel_size_arcsec:
        float, default = 1.5

    """
    n_rss = len(rss_list)

    if verbose:
        if n_rss > 1:
            print("\n> Starting alignment procedure...")
        else:
            print("\n> Only one file provided, no need of performing alignment ...")
            if np.nanmedian(ADR_x_fit_list) == 0 and ADR: print(
                "  But ADR data provided and ADR correction requested, rebuiding the cube...")

    xx = [0]  # This will have 0, x12, x23, x34, ... xn1
    yy = [0]  # This will have 0, y12, y23, y34, ... yn1

    if len(flux_calibration_list[0]) == 0:
        for i in range(1, n_rss): flux_calibration_list.append([])

    if len(offsets) == 0:
        if verbose and n_rss > 1: print("\n  Using peak of the emission tracing all wavelengths to align cubes:")
        n_cubes = len(cube_list)
        if n_cubes != n_rss:
            if verbose:
                print("\n\n\n ERROR: number of cubes and number of rss files don't match!")
                print("\n\n THIS IS GOING TO FAIL ! \n\n\n")

        for i in range(n_rss - 1):
            xx.append(
                cube_list[i + 1].offset_from_center_x_arcsec_tracing - cube_list[i].offset_from_center_x_arcsec_tracing)
            yy.append(
                cube_list[i + 1].offset_from_center_y_arcsec_tracing - cube_list[i].offset_from_center_y_arcsec_tracing)
        xx.append(cube_list[0].offset_from_center_x_arcsec_tracing - cube_list[-1].offset_from_center_x_arcsec_tracing)
        yy.append(cube_list[0].offset_from_center_y_arcsec_tracing - cube_list[-1].offset_from_center_y_arcsec_tracing)

    else:
        if verbose and n_rss > 1: print("\n  Using offsets provided!")
        for i in range(0, 2 * n_rss - 2, 2):
            xx.append(offsets[i])
            yy.append(offsets[i + 1])
        xx.append(-np.nansum(xx))  #
        yy.append(-np.nansum(yy))

    # Estimate median value of the centre of files
    list_RA_centre_deg = []
    list_DEC_centre_deg = []

    for i in range(n_rss):
        list_RA_centre_deg.append(rss_list[i].RA_centre_deg)
        list_DEC_centre_deg.append(rss_list[i].DEC_centre_deg)

    median_RA_centre_deg = np.nanmedian(list_RA_centre_deg)
    median_DEC_centre_deg = np.nanmedian(list_DEC_centre_deg)

    distance_from_median = []

    for i in range(n_rss):
        rss_list[i].ALIGNED_RA_centre_deg = median_RA_centre_deg + np.nansum(
            xx[1:i + 1]) / 3600.  # CHANGE SIGN 26 Apr 2019    # ERA cube_list[0]
        rss_list[i].ALIGNED_DEC_centre_deg = median_DEC_centre_deg - np.nansum(
            yy[1:i + 1]) / 3600.  # rss_list[0].DEC_centre_deg

        distance_from_median.append(np.sqrt(
            (rss_list[i].RA_centre_deg - median_RA_centre_deg) ** 2 +
            (rss_list[i].DEC_centre_deg - median_DEC_centre_deg) ** 2))

    reference_rss = distance_from_median.index(np.nanmin(distance_from_median))

    if len(centre_deg) == 0:
        if verbose and n_rss > 1: print(
            "  No central coordenates given, using RSS {} for getting the central coordenates:".format(
                reference_rss + 1))
        RA_centre_deg = rss_list[reference_rss].ALIGNED_RA_centre_deg
        DEC_centre_deg = rss_list[reference_rss].ALIGNED_DEC_centre_deg
    else:
        if verbose and n_rss > 1: print("  Central coordenates provided: ")
        RA_centre_deg = centre_deg[0]
        DEC_centre_deg = centre_deg[1]

    if verbose and n_rss > 1:
        print("\n> Median central coordenates of RSS files: RA =", RA_centre_deg, " DEC =", DEC_centre_deg)

        print(
            "\n  Offsets (in arcsec):        x             y                          ( EAST- / WEST+   NORTH- / SOUTH+) ")
        for i in range(1, len(xx) - 1):
            print("         {:2.0f} -> {:2.0f}           {:+5.3f}         {:+5.3f}".format(i, i + 1, xx[i], yy[i]))
        print("         {:2.0f} ->  1           {:+5.3f}         {:+5.3f}".format(len(xx) - 1, xx[-1], yy[-1]))
        print("           TOTAL:            {:5.3f}          {:5.3f}".format(np.nansum(xx), np.nansum(yy)))

        print("\n         New_RA_centre_deg       New_DEC_centre_deg      Diff with respect Cube 1 [arcsec]")

        for i in range(0, n_rss):
            print("  Cube {:2.0f}:     {:5.8f}          {:5.8f}           {:+5.3f}   ,  {:+5.3f}   ".format(i + 1,
                                                                                                            rss_list[
                                                                                                                i].ALIGNED_RA_centre_deg,
                                                                                                            rss_list[
                                                                                                                i].ALIGNED_DEC_centre_deg,
                                                                                                            (rss_list[
                                                                                                                 i].ALIGNED_RA_centre_deg -
                                                                                                             rss_list[
                                                                                                                 0].ALIGNED_RA_centre_deg) * 3600.,
                                                                                                            (rss_list[
                                                                                                                 i].ALIGNED_DEC_centre_deg -
                                                                                                             rss_list[
                                                                                                                 0].ALIGNED_DEC_centre_deg) * 3600.))

    offsets_files = []
    for i in range(1, n_rss):  # For keeping in the files with self.offsets_files
        vector = [xx[i], yy[i]]
        offsets_files.append(vector)

    xx_dif = np.nansum(xx[0:-1])
    yy_dif = np.nansum(yy[0:-1])

    if verbose and n_rss > 1: print('\n  Accumulative difference of offsets: {:.2f}" x {:.2f}" '.format(xx_dif, yy_dif))

    if len(size_arcsec) == 0:
        RA_size_arcsec = rss_list[0].RA_segment + np.abs(xx_dif) + 3 * kernel_size_arcsec
        DEC_size_arcsec = rss_list[0].DEC_segment + np.abs(yy_dif) + 3 * kernel_size_arcsec
        size_arcsec = [RA_size_arcsec, DEC_size_arcsec]

    if verbose and n_rss > 1: print(
        '\n  RA_size x DEC_size  = {:.2f}" x {:.2f}" '.format(size_arcsec[0], size_arcsec[1]))

    cube_aligned_list = []

    for i in range(1, n_rss + 1):
        # escribe="cube"+np.str(i)+"_aligned"
        cube_aligned_list.append("cube" + np.str(i) + "_aligned")

    if np.nanmedian(ADR_x_fit_list) == 0 and ADR:  # Check if ADR info is provided and ADR is requested
        ADR_x_fit_list = []
        ADR_y_fit_list = []
        for i in range(n_rss):
            _x_ = []
            _y_ = []
            for j in range(len(cube_list[i].ADR_x_fit)):
                _x_.append(cube_list[i].ADR_x_fit[j])
                _y_.append(cube_list[i].ADR_y_fit[j])
            ADR_x_fit_list.append(_x_)
            ADR_y_fit_list.append(_y_)

    for i in range(n_rss):

        if n_rss > 1 or np.nanmedian(ADR_x_fit_list) != 0:

            if verbose: print("\n> Creating aligned cube", i + 1, "of a total of", n_rss, "...")

            cube_aligned_list[i] = Interpolated_cube(rss_list[i], pixel_size_arcsec=pixel_size_arcsec,
                                                     kernel_size_arcsec=kernel_size_arcsec,
                                                     centre_deg=[RA_centre_deg, DEC_centre_deg],
                                                     size_arcsec=size_arcsec,
                                                     aligned_coor=True, flux_calibration=flux_calibration_list[i],
                                                     offsets_files=offsets_files, offsets_files_position=i + 1,
                                                     ADR=ADR, jump=jump, ADR_x_fit=ADR_x_fit_list[i],
                                                     ADR_y_fit=ADR_y_fit_list[i], check_ADR=True,
                                                     half_size_for_centroid=half_size_for_centroid, box_x=box_x,
                                                     box_y=box_y,
                                                     adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing,
                                                     plot_tracing_maps=plot_tracing_maps,
                                                     plot=plot, plot_spectra=plot_spectra, edgelow=edgelow,
                                                     edgehigh=edgehigh,

                                                     warnings=warnings, verbose=verbose)
            if plot_weight: cube_aligned_list[i].plot_weight()
        else:
            cube_aligned_list[i] = cube_list[i]
            if verbose: print(
                "\n> Only one file provided and no ADR correction given, the aligned cube is the same than the original cube...")

    if verbose and n_rss > 1:
        print("\n> Checking offsets of ALIGNED cubes (in arcsec, everything should be close to 0):")
        print(
            "  Offsets (in arcsec):        x             y                          ( EAST- / WEST+   NORTH- / SOUTH+) ")

    xxx = []
    yyy = []

    for i in range(1, n_rss):
        xxx.append(cube_aligned_list[i - 1].offset_from_center_x_arcsec_tracing - cube_aligned_list[
            i].offset_from_center_x_arcsec_tracing)
        yyy.append(cube_aligned_list[i - 1].offset_from_center_y_arcsec_tracing - cube_aligned_list[
            i].offset_from_center_y_arcsec_tracing)
    xxx.append(cube_aligned_list[-1].offset_from_center_x_arcsec_tracing - cube_aligned_list[
        0].offset_from_center_x_arcsec_tracing)
    yyy.append(cube_aligned_list[-1].offset_from_center_y_arcsec_tracing - cube_aligned_list[
        0].offset_from_center_y_arcsec_tracing)

    if verbose and n_rss > 1:

        for i in range(1, len(xx) - 1):
            print("         {:2.0f} -> {:2.0f}           {:+5.3f}         {:+5.3f}".format(i, i + 1, xxx[i - 1],
                                                                                           yyy[i - 1]))
        print("         {:2.0f} ->  1           {:+5.3f}         {:+5.3f}".format(len(xxx), xxx[-1], yyy[-1]))
        print("           TOTAL:            {:5.3f}          {:5.3f}".format(np.nansum(xxx), np.nansum(yyy)))

        print("\n> Alignment of n = {} cubes COMPLETED !".format(n_rss))
    return cube_aligned_list


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def save_cube_to_fits_file(cube, fits_file, description="", obj_name="", path=""):
    """
    Routine to save a cube as a fits file

    Parameters
    ----------
    Combined cube:
        cube
    Header:
        Header
    """

    if path != "": fits_file = full_path(fits_file, path)

    fits_image_hdu = fits.PrimaryHDU(cube.data)
    #    errors = cube.data*0  ### TO BE DONE
    #    error_hdu = fits.ImageHDU(errors)

    # wavelength =  cube.wavelength

    if cube.offsets_files_position == "":
        fits_image_hdu.header['HISTORY'] = 'Combined datacube using PyKOALA'
    else:
        fits_image_hdu.header['HISTORY'] = 'Interpolated datacube using PyKOALA'

    fits_image_hdu.header['HISTORY'] = 'Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany,'
    fits_image_hdu.header['HISTORY'] = 'Blake Staples, Taylah Beard, Matt Owers, James Tocknell et al.'

    fits_image_hdu.header['HISTORY'] = version  # 'Version 0.10 - 12th February 2019'
    now = datetime.datetime.now()
    fits_image_hdu.header['HISTORY'] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    fits_image_hdu.header['DATE'] = now.strftime(
        "%Y-%m-%dT%H:%M:%S")  # '2002-09-16T18:52:44'   # /Date of FITS file creation

    fits_image_hdu.header['BITPIX'] = 16

    fits_image_hdu.header["ORIGIN"] = 'AAO'  # / Originating Institution
    fits_image_hdu.header["TELESCOP"] = 'Anglo-Australian Telescope'  # / Telescope Name
    fits_image_hdu.header["ALT_OBS"] = 1164  # / Altitude of observatory in metres
    fits_image_hdu.header["LAT_OBS"] = -31.27704  # / Observatory latitude in degrees
    fits_image_hdu.header["LONG_OBS"] = 149.0661  # / Observatory longitude in degrees

    fits_image_hdu.header["INSTRUME"] = "AAOMEGA-KOALA"  # / Instrument in use
    fits_image_hdu.header["GRATID"] = cube.grating  # / Disperser ID
    if cube.grating in red_gratings: SPECTID = "RD"
    if cube.grating in blue_gratings: SPECTID = "BL"
    fits_image_hdu.header["SPECTID"] = SPECTID  # / Spectrograph ID
    fits_image_hdu.header["DICHROIC"] = 'X5700'  # / Dichroic name   ---> CHANGE if using X6700!!

    if obj_name == "":
        fits_image_hdu.header['OBJECT'] = cube.object
    else:
        fits_image_hdu.header['OBJECT'] = obj_name
    fits_image_hdu.header['TOTALEXP'] = cube.total_exptime
    fits_image_hdu.header['EXPTIMES'] = np.str(cube.exptimes)

    fits_image_hdu.header['NAXIS'] = 3  # / number of array dimensions
    fits_image_hdu.header['NAXIS1'] = cube.data.shape[1]  ##### CHECK !!!!!!!
    fits_image_hdu.header['NAXIS2'] = cube.data.shape[2]
    fits_image_hdu.header['NAXIS3'] = cube.data.shape[0]

    # WCS
    fits_image_hdu.header["RADECSYS"] = 'FK5'  # / FK5 reference system
    fits_image_hdu.header["EQUINOX"] = 2000  # / [yr] Equinox of equatorial coordinates
    fits_image_hdu.header["WCSAXES"] = 3  # / Number of coordinate axes

    fits_image_hdu.header['CRPIX1'] = cube.data.shape[1] / 2.  # / Pixel coordinate of reference point
    fits_image_hdu.header['CDELT1'] = -cube.pixel_size_arcsec / 3600.  # / Coordinate increment at reference point
    fits_image_hdu.header[
        'CTYPE1'] = "RA--TAN"  # 'DEGREE'                               # / Coordinate type code
    fits_image_hdu.header['CRVAL1'] = cube.RA_centre_deg  # / Coordinate value at reference point

    fits_image_hdu.header['CRPIX2'] = cube.data.shape[2] / 2.  # / Pixel coordinate of reference point
    fits_image_hdu.header['CDELT2'] = cube.pixel_size_arcsec / 3600.  # Coordinate increment at reference point
    fits_image_hdu.header[
        'CTYPE2'] = "DEC--TAN"  # 'DEGREE'                               # / Coordinate type code
    fits_image_hdu.header['CRVAL2'] = cube.DEC_centre_deg  # / Coordinate value at reference point

    fits_image_hdu.header['RAcen'] = cube.RA_centre_deg
    fits_image_hdu.header['DECcen'] = cube.DEC_centre_deg
    fits_image_hdu.header['PIXsize'] = cube.pixel_size_arcsec
    fits_image_hdu.header['KERsize'] = cube.kernel_size_arcsec
    fits_image_hdu.header['Ncols'] = cube.data.shape[2]
    fits_image_hdu.header['Nrows'] = cube.data.shape[1]
    fits_image_hdu.header['PA'] = cube.PA

    # Wavelength calibration
    fits_image_hdu.header["CTYPE3"] = 'Wavelength'  # / Label for axis 3
    fits_image_hdu.header["CUNIT3"] = 'Angstroms'  # / Units for axis 3
    fits_image_hdu.header["CRVAL3"] = cube.CRVAL1_CDELT1_CRPIX1[0]  # 7.692370611909E+03  / Co-ordinate value of axis 3
    fits_image_hdu.header["CDELT3"] = cube.CRVAL1_CDELT1_CRPIX1[1]  # 1.575182431607E+00
    fits_image_hdu.header["CRPIX3"] = cube.CRVAL1_CDELT1_CRPIX1[2]  # 1024. / Reference pixel along axis 3
    fits_image_hdu.header["N_WAVE"] = cube.n_wave  # 1024. / Reference pixel along axis 3

    fits_image_hdu.header["V_W_MIN"] = cube.valid_wave_min
    fits_image_hdu.header["V_W_MAX"] = cube.valid_wave_max

    scale_flux = 1.0
    try:
        scale_flux = cube.scale_flux
    except Exception:
        scale_flux = 1.0
    fits_image_hdu.header[
        "SCAFLUX"] = scale_flux  # If the cube has been scaled in flux using scale_cubes_using_common_region

    if cube.offsets_files_position == "":  # If cube.offsets_files_position is not given, it is a combined_cube
        # print("   THIS IS A COMBINED CUBE")
        fits_image_hdu.header["COMCUBE"] = True
        is_combined_cube = True
        cofiles = len(cube.offsets_files) + 1
        fits_image_hdu.header['COFILES'] = cofiles  # Number of combined files
        if cofiles > 1:
            for i in (list(range(cofiles))):
                if i < 9:
                    text = "RSS_0" + np.str(i + 1)
                else:
                    text = "RSS_" + np.str(i + 1)
                fits_image_hdu.header[text] = cube.rss_list[i]
    else:
        # print(" THIS IS NOT A COMBINED CUBE")
        fits_image_hdu.header["COMCUBE"] = False
        is_combined_cube = False
        fits_image_hdu.header['COFILES'] = 1

    offsets_text = " "
    if len(cube.offsets_files) != 0:  # If offsets provided/obtained, this will not be 0
        for i in range(len(cube.offsets_files)):
            if i != 0: offsets_text = offsets_text + "  ,  "
            offsets_text = offsets_text + np.str(np.around(cube.offsets_files[i][0], 3)) + " " + np.str(
                np.around(cube.offsets_files[i][1], 3))
        fits_image_hdu.header['OFFSETS'] = offsets_text  # Offsets
        if is_combined_cube:
            fits_image_hdu.header['OFF_POS'] = 0
        else:
            fits_image_hdu.header['OFF_POS'] = cube.offsets_files_position

    fits_image_hdu.header['ADRCOR'] = np.str(cube.adrcor)  # ADR before, ADR was given as an input

    if len(cube.ADR_x_fit) != 0:
        text = ""
        for i in range(len(cube.ADR_x_fit)):
            if i != 0: text = text + "  ,  "
            text = text + np.str(cube.ADR_x_fit[i])
        fits_image_hdu.header['ADRxFIT'] = text

        text = ""
        for i in range(len(cube.ADR_y_fit)):
            if i != 0: text = text + "  ,  "
            text = text + np.str(cube.ADR_y_fit[i])
        fits_image_hdu.header['ADRyFIT'] = text

    if np.nanmedian(cube.data) > 1:
        fits_image_hdu.header['FCAL'] = "False"
        fits_image_hdu.header['F_UNITS'] = "Counts"
        # flux_correction_hdu = fits.ImageHDU(0*wavelength)
    else:
        # flux_correction = fcal
        # flux_correction_hdu = fits.ImageHDU(flux_correction)
        fits_image_hdu.header['FCAL'] = "True"
        fits_image_hdu.header['F_UNITS'] = "erg s-1 cm-2 A-1"

    if description == "":
        description = cube.description
    fits_image_hdu.header['DESCRIP'] = description.replace("\n", "")
    fits_image_hdu.header['FILE_OUT'] = fits_file

    #    hdu_list = fits.HDUList([fits_image_hdu, error_hdu])
    #    hdu_list = fits.HDUList([fits_image_hdu, wavelengths_hdu, flux_correction_hdu])
    hdu_list = fits.HDUList([fits_image_hdu])  # , flux_correction_hdu])

    hdu_list.writeto(fits_file, overwrite=True)
    if is_combined_cube:
        print("\n> Combined cube saved to file:")
        print(" ", fits_file)
    else:
        print("\n> Cube saved to file:")
        print(" ", fits_file)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def read_cube(filename, description="", half_size_for_centroid=10, plot_spectra=False,
              valid_wave_min=0, valid_wave_max=0, edgelow=50, edgehigh=50, g2d=False,
              plot=False, verbose=True, print_summary=True,
              text_intro="\n> Reading datacube from fits file:"):
    if verbose: print(text_intro)
    if verbose: print('  "' + filename + '"', "...")
    cube_fits_file = fits.open(filename)  # Open file

    objeto = cube_fits_file[0].header['OBJECT']
    if description == "": description = objeto + " - CUBE"
    grating = cube_fits_file[0].header['GRATID']

    total_exptime = cube_fits_file[0].header['TOTALEXP']
    exptimes_ = cube_fits_file[0].header['EXPTIMES'].strip('][').split(',')
    exptimes = []
    for j in range(len(exptimes_)):
        exptimes.append(float(exptimes_[j]))
    number_of_combined_files = cube_fits_file[0].header['COFILES']
    # fcal = cube_fits_file[0].header['FCAL']

    filename = cube_fits_file[0].header['FILE_OUT']
    RACEN = cube_fits_file[0].header['RACEN']
    DECCEN = cube_fits_file[0].header['DECCEN']
    centre_deg = [RACEN, DECCEN]
    pixel_size = cube_fits_file[0].header['PIXsize']
    kernel_size = cube_fits_file[0].header['KERsize']
    n_cols = cube_fits_file[0].header['NCOLS']
    n_rows = cube_fits_file[0].header['NROWS']
    PA = cube_fits_file[0].header['PA']

    CRVAL3 = cube_fits_file[0].header['CRVAL3']  # 4695.841684048
    CDELT3 = cube_fits_file[0].header['CDELT3']  # 1.038189521346
    CRPIX3 = cube_fits_file[0].header['CRPIX3']  # 1024.0
    CRVAL1_CDELT1_CRPIX1 = [CRVAL3, CDELT3, CRPIX3]
    n_wave = cube_fits_file[0].header['NAXIS3']

    adrcor = cube_fits_file[0].header['ADRCOR']

    number_of_combined_files = cube_fits_file[0].header['COFILES']

    ADR_x_fit_ = cube_fits_file[0].header['ADRXFIT'].split(',')
    ADR_x_fit = []
    for j in range(len(ADR_x_fit_)):
        ADR_x_fit.append(float(ADR_x_fit_[j]))
    ADR_y_fit_ = cube_fits_file[0].header['ADRYFIT'].split(',')
    ADR_y_fit = []
    for j in range(len(ADR_y_fit_)):
        ADR_y_fit.append(float(ADR_y_fit_[j]))

    adr_index_fit = len(ADR_y_fit) - 1

    rss_files = []
    offsets_files_position = cube_fits_file[0].header['OFF_POS']
    offsets_files = []
    offsets_files_ = cube_fits_file[0].header['OFFSETS'].split(',')
    for j in range(len(offsets_files_)):
        valor = offsets_files_[j].split(' ')
        offset_ = []
        p = 0
        for k in range(len(valor)):
            try:
                offset_.append(np.float(valor[k]))
            except Exception:
                p = p + 1
                # print j,k,valor[k], "no es float"
        offsets_files.append(offset_)

    if number_of_combined_files > 1:
        for i in range(number_of_combined_files):
            if i < 10:
                head = "RSS_0" + np.str(i + 1)
            else:
                head = "RSS_" + np.str(i + 1)
            rss_files.append(cube_fits_file[0].header[head])

    if valid_wave_min == 0: valid_wave_min = cube_fits_file[0].header["V_W_MIN"]
    if valid_wave_max == 0: valid_wave_max = cube_fits_file[0].header["V_W_MAX"]

    wavelength = np.array([0.] * n_wave)
    wavelength[np.int(CRPIX3) - 1] = CRVAL3
    for i in range(np.int(CRPIX3) - 2, -1, -1):
        wavelength[i] = wavelength[i + 1] - CDELT3
    for i in range(np.int(CRPIX3), n_wave):
        wavelength[i] = wavelength[i - 1] + CDELT3

    cube = Interpolated_cube(filename, pixel_size, kernel_size, plot=False, verbose=verbose,
                             read_fits_cube=True, zeros=True,
                             ADR_x_fit=np.array(ADR_x_fit), ADR_y_fit=np.array(ADR_y_fit),
                             objeto=objeto, description=description,
                             n_cols=n_cols, n_rows=n_rows, PA=PA,
                             wavelength=wavelength, n_wave=n_wave,
                             total_exptime=total_exptime, valid_wave_min=valid_wave_min, valid_wave_max=valid_wave_max,
                             CRVAL1_CDELT1_CRPIX1=CRVAL1_CDELT1_CRPIX1,
                             grating=grating, centre_deg=centre_deg, number_of_combined_files=number_of_combined_files)

    cube.exptimes = exptimes
    cube.data = cube_fits_file[0].data
    if half_size_for_centroid > 0:
        box_x, box_y = cube.box_for_centroid(half_size_for_centroid=half_size_for_centroid, verbose=verbose)
    else:
        box_x = [0, -1]
        box_y = [0, -1]
    cube.trace_peak(box_x=box_x, box_y=box_y, plot=plot, edgelow=edgelow, edgehigh=edgehigh,
                    adr_index_fit=adr_index_fit, g2d=g2d,
                    verbose=False)
    cube.get_integrated_map(plot=plot, plot_spectra=plot_spectra, verbose=verbose, plot_centroid=True,
                            g2d=g2d)  # ,fcal=fcal, box_x=box_x, box_y=box_y)
    # For calibration stars, we get an integrated star flux and a seeing
    cube.integrated_star_flux = np.zeros_like(cube.wavelength)
    cube.offsets_files = offsets_files
    cube.offsets_files_position = offsets_files_position
    cube.rss_files = rss_files  # Add this in Interpolated_cube
    cube.adrcor = adrcor
    cube.rss_list = filename

    if number_of_combined_files > 1 and verbose:
        print("\n> This cube was created using the following rss files:")
        for i in range(number_of_combined_files):
            print(" ", rss_files[i])

        print_offsets = "  Offsets used : "
        for i in range(number_of_combined_files - 1):
            print_offsets = print_offsets + (np.str(offsets_files[i]))
            if i < number_of_combined_files - 2: print_offsets = print_offsets + " , "
        print(print_offsets)

    if verbose and print_summary:
        print("\n> Summary of reading cube :")
        print("  Object          = ", cube.object)
        print("  Description     = ", cube.description)
        print("  Centre:  RA     = ", cube.RA_centre_deg, "Deg")
        print("          DEC     = ", cube.DEC_centre_deg, "Deg")
        print("  PA              = ", np.round(cube.PA, 2), "Deg")
        print("  Size [pix]      = ", cube.n_cols, " x ", cube.n_rows)
        print("  Size [arcsec]   = ", np.round(cube.n_cols * cube.pixel_size_arcsec, 1), " x ",
              np.round(cube.n_rows * cube.pixel_size_arcsec, 1))
        print("  Pix size        = ", cube.pixel_size_arcsec, " arcsec")
        print("  Total exp time  = ", total_exptime, "s")
        if number_of_combined_files > 1: print("  Files combined  = ", cube.number_of_combined_files)
        print("  Med exp time    = ", np.nanmedian(cube.exptimes), "s")
        print("  ADR corrected   = ", cube.adrcor)
        #    print "  Offsets used   = ",self.offsets_files
        print("  Wave Range      =  [", np.round(cube.wavelength[0], 2), ",", np.round(cube.wavelength[-1], 2), "]")
        print("  Wave Resolution =  {:.3} A/pix".format(CDELT3))
        print("  Valid wav range =  [", np.round(valid_wave_min, 2), ",", np.round(valid_wave_max, 2), "]")

        if np.nanmedian(cube.data) < 0.001:
            print("  Cube is flux calibrated ")
        else:
            print("  Cube is NOT flux calibrated ")

        print("\n> Use these parameters for acceding the data :\n")
        print("  cube.wavelength       : Array with wavelengths")
        print("  cube.data[w, y, x]    : Flux of the w wavelength in spaxel (x,y) - NOTE x and y positions!")
    #    print "\n> Cube read! "

    return cube


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def save_rss_fits(rss, data=[[0], [0]], fits_file="RSS_rss.fits", text="RSS data", sol="",
                  description="", verbose=True):  # fcal=[0],     # TASK_save_rss_fits
    """
    Routine to save RSS data as fits

    Parameters
    ----------
    rss is the rss
    description = if you want to add a description
    """
    if np.nanmedian(data[0]) == 0:
        data = rss.intensity_corrected
        if verbose: print("\n> Using rss.intensity_corrected of given RSS file to create fits file...")
    else:
        if len(np.array(data).shape) != 2:
            print("\n> The data provided are NOT valid, as they have a shape", data.shape)
            print("  Using rss.intensity_corrected instead to create a RSS fits file !")
            data = rss.intensity_corrected
        else:
            print("\n> Using the data provided + structure of given RSS file to create fits file...")
    fits_image_hdu = fits.PrimaryHDU(data)

    fits_image_hdu.header['BITPIX'] = 16

    fits_image_hdu.header["ORIGIN"] = 'AAO'  # / Originating Institution
    fits_image_hdu.header["TELESCOP"] = 'Anglo-Australian Telescope'  # / Telescope Name
    fits_image_hdu.header["ALT_OBS"] = 1164  # / Altitude of observatory in metres
    fits_image_hdu.header["LAT_OBS"] = -31.27704  # / Observatory latitude in degrees
    fits_image_hdu.header["LONG_OBS"] = 149.0661  # / Observatory longitude in degrees

    fits_image_hdu.header["INSTRUME"] = "AAOMEGA-KOALA"  # / Instrument in use
    fits_image_hdu.header["GRATID"] = rss.grating  # / Disperser ID
    SPECTID = "UNKNOWN"
    if rss.grating in red_gratings: SPECTID = "RD"
    if rss.grating in blue_gratings: SPECTID = "BL"
    fits_image_hdu.header["SPECTID"] = SPECTID  # / Spectrograph ID

    fits_image_hdu.header["DICHROIC"] = 'X5700'  # / Dichroic name   ---> CHANGE if using X6700!!

    fits_image_hdu.header['OBJECT'] = rss.object
    fits_image_hdu.header["EXPOSED"] = rss.exptime
    fits_image_hdu.header["ZDSTART"] = rss.ZDSTART
    fits_image_hdu.header["ZDEND"] = rss.ZDEND

    fits_image_hdu.header['NAXIS'] = 2  # / number of array dimensions
    fits_image_hdu.header['NAXIS1'] = rss.intensity_corrected.shape[0]
    fits_image_hdu.header['NAXIS2'] = rss.intensity_corrected.shape[1]

    fits_image_hdu.header['RAcen'] = rss.RA_centre_deg
    fits_image_hdu.header['DECcen'] = rss.DEC_centre_deg
    fits_image_hdu.header['TEL_PA'] = rss.PA

    fits_image_hdu.header["CTYPE2"] = 'Fibre number'  # / Label for axis 2
    fits_image_hdu.header["CUNIT2"] = ' '  # / Units for axis 2
    fits_image_hdu.header["CTYPE1"] = 'Wavelength'  # / Label for axis 2
    fits_image_hdu.header["CUNIT1"] = 'Angstroms'  # / Units for axis 2

    fits_image_hdu.header["CRVAL1"] = rss.CRVAL1_CDELT1_CRPIX1[0]  # / Co-ordinate value of axis 2
    fits_image_hdu.header["CDELT1"] = rss.CRVAL1_CDELT1_CRPIX1[1]  #
    fits_image_hdu.header["CRPIX1"] = rss.CRVAL1_CDELT1_CRPIX1[2]  # 1024. / Reference pixel along axis 2
    fits_image_hdu.header["CRVAL2"] = 5.000000000000E-01  # / Co-ordinate value of axis 2
    fits_image_hdu.header["CDELT2"] = 1.000000000000E+00  # / Co-ordinate increment along axis 2
    fits_image_hdu.header["CRPIX2"] = 1.000000000000E+00  # / Reference pixel along axis 2

    if len(sol) > 0:  # sol has been provided
        fits_image_hdu.header["SOL0"] = sol[0]
        fits_image_hdu.header["SOL1"] = sol[1]
        fits_image_hdu.header["SOL2"] = sol[2]

    if description == "":
        description = rss.object
    fits_image_hdu.header['DESCRIP'] = description

    for item in rss.history_RSS:
        if item == "- Created fits file (this file) :":
            fits_image_hdu.header['HISTORY'] = "- Created fits file :"
        else:
            fits_image_hdu.header['HISTORY'] = item
    fits_image_hdu.header['FILE_IN'] = rss.filename

    fits_image_hdu.header['HISTORY'] = '-- RSS processing using PyKOALA ' + version
    # fits_image_hdu.header['HISTORY'] = 'Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany et al.'
    # fits_image_hdu.header['HISTORY'] =  version #'Version 0.10 - 12th February 2019'
    now = datetime.datetime.now()

    fits_image_hdu.header['HISTORY'] = now.strftime("File created on %d %b %Y, %H:%M:%S using input file:")
    fits_image_hdu.header['DATE'] = now.strftime(
        "%Y-%m-%dT%H:%M:%S")  # '2002-09-16T18:52:44'   # /Date of FITS file creation
    # fits_image_hdu.header['HISTORY'] = 'using input file:'
    fits_image_hdu.header['HISTORY'] = rss.filename

    for item in rss.history:
        fits_image_hdu.header['HISTORY'] = item

    fits_image_hdu.header['HISTORY'] = "- Created fits file (this file) :"
    fits_image_hdu.header['HISTORY'] = " " + fits_file
    fits_image_hdu.header['FILE_OUT'] = fits_file

    # TO BE DONE
    errors = [0]  ### TO BE DONE
    error_hdu = fits.ImageHDU(errors)

    # Header 2 with the RA and DEC info!
    header2_all_fibres = rss.header2_data
    header2_good_fibre = []
    header2_original_fibre = []
    header2_new_fibre = []
    header2_delta_RA = []
    header2_delta_DEC = []
    header2_2048 = []
    header2_0 = []

    fibre = 1
    for i in range(len(header2_all_fibres)):
        if header2_all_fibres[i][1] == 1:
            header2_original_fibre.append(i + 1)
            header2_new_fibre.append(fibre)
            header2_good_fibre.append(1)
            header2_delta_RA.append(header2_all_fibres[i][5])
            header2_delta_DEC.append(header2_all_fibres[i][6])
            header2_2048.append(2048)
            header2_0.append(0)
            fibre = fibre + 1

    #    header2_=[header2_new_fibre, header2_good_fibre, header2_good_fibre, header2_2048, header2_0,  header2_delta_RA,  header2_delta_DEC,  header2_original_fibre]
    #    header2 = np.array(header2_).T.tolist()
    #    header2_hdu = fits.ImageHDU(header2)

    col1 = fits.Column(name='Fibre', format='I', array=np.array(header2_new_fibre))
    col2 = fits.Column(name='Status', format='I', array=np.array(header2_good_fibre))
    col3 = fits.Column(name='Ones', format='I', array=np.array(header2_good_fibre))
    col4 = fits.Column(name='Wavelengths', format='I', array=np.array(header2_2048))
    col5 = fits.Column(name='Zeros', format='I', array=np.array(header2_0))
    col6 = fits.Column(name='Delta_RA', format='D', array=np.array(header2_delta_RA))
    col7 = fits.Column(name='Delta_Dec', format='D', array=np.array(header2_delta_DEC))
    col8 = fits.Column(name='Fibre_OLD', format='I', array=np.array(header2_original_fibre))

    cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8])
    header2_hdu = fits.BinTableHDU.from_columns(cols)

    header2_hdu.header['CENRA'] = rss.RA_centre_deg / (180 / np.pi)  # Must be in radians
    header2_hdu.header['CENDEC'] = rss.DEC_centre_deg / (180 / np.pi)

    hdu_list = fits.HDUList([fits_image_hdu, error_hdu,
                             header2_hdu])  # hdu_list = fits.HDUList([fits_image_hdu, wavelengths_hdu, flux_correction_hdu])

    hdu_list.writeto(fits_file, overwrite=True)

    print('\n> ' + text + 'saved to file "' + fits_file + '"')


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def align_blue_and_red_cubes(blue, red, half_size_for_centroid=8, box_x=[], box_y=[],
                             verbose=True, plot=True, plot_centroid=True, g2d=False):
    """
    For aligning the BLUE and RED cubes, follow these steps:\n
    1. First process the RED cube. It's much better for providing the best values of the offsets between RSS files.\n
    2. Save the following parameters:
        - offsets (for alignment)
        - size_arcsec
        - centre_deg
        - flux_ratios
    2. Process the BLUE cube including these parameters as given in the red cube.\n
        - Ideally, process the BLUE data at least one to get the ADR_x_fit and ADR_y_fit solutions and include them in the .config file
        - trim_cube = False to be sure that the resulting blue cube has the same size than the red cube
    3. Run this taks and save the (very probably small, < 50% pixel_size) offsets between the blue and red cube, delta_RA and delta_DEC
    4. Reprocess the BLUE cube including delta_RA and delta_DEC
    5. Run this task and check that the offsets between the red and blue cubes is < 0.1 arcsec (perhaps a couple of iterations are needed to get this)
    6. Trim the red and blue cubes if needed.
    """

    print("\n> Checking the alignment between a blue cube and a red cube...")

    try:
        try_read = blue + "  "
        if verbose: text_intro = "\n> Reading the blue cube from the fits file..." + try_read[-2:-1]
        blue_cube = read_cube(blue, text_intro=text_intro,
                              plot=plot, half_size_for_centroid=half_size_for_centroid, plot_spectra=False,
                              verbose=verbose)
    except Exception:
        print("  - The blue cube is an object")
        blue_cube = blue

    try:
        try_read = red + "  "
        if verbose: text_intro = "\n> Reading the red cube from the fits file..." + try_read[-2:-1]
        red_cube = read_cube(red, text_intro=text_intro,
                             plot=plot, half_size_for_centroid=half_size_for_centroid, plot_spectra=False,
                             verbose=verbose)
    except Exception:
        print("  - The red  cube is an object")
        red_cube = red
        if box_x == [] or box_y == []:
            box_x, box_y = red_cube.box_for_centroid(half_size_for_centroid=half_size_for_centroid, verbose=verbose)
        blue_cube.get_integrated_map(box_x=box_x, box_y=box_y, plot_spectra=False, plot=plot, verbose=verbose,
                                     plot_centroid=plot_centroid, g2d=g2d)
        red_cube.get_integrated_map(box_x=box_x, box_y=box_y, plot_spectra=False, plot=plot, verbose=verbose,
                                    plot_centroid=plot_centroid, g2d=g2d)

    print("\n> Checking the properties of these cubes:\n")
    print(
        "  CUBE      RA_centre             DEC_centre     pixel size   kernel size   n_cols      n_rows      x_max      y_max")
    print("  blue   {}   {}      {}           {}         {}          {}          {}         {}".format(
        blue_cube.RA_centre_deg, blue_cube.DEC_centre_deg, blue_cube.pixel_size_arcsec, blue_cube.kernel_size_arcsec,
        blue_cube.n_cols, blue_cube.n_rows, blue_cube.max_x, blue_cube.max_y))
    print("  red    {}   {}      {}           {}         {}          {}          {}         {}".format(
        red_cube.RA_centre_deg, red_cube.DEC_centre_deg, red_cube.pixel_size_arcsec, red_cube.kernel_size_arcsec,
        red_cube.n_cols, red_cube.n_rows, red_cube.max_x, red_cube.max_y))

    all_ok = True
    to_do_list = []
    for _property_ in ["RA_centre_deg", "DEC_centre_deg", "pixel_size_arcsec", "kernel_size_arcsec", "n_cols",
                       "n_rows"]:
        property_values = [_property_]
        exec("property_values.append(blue_cube." + _property_ + ")")
        exec("property_values.append(red_cube." + _property_ + ")")
        property_values.append(property_values[-2] - property_values[-1])

        if property_values[-1] != 0:
            print("  - Property {} has DIFFERENT values !!!".format(_property_))
            all_ok = False
            if _property_ == "RA_centre_deg": to_do_list.append(
                "  - Check the RA_centre_deg to get the same value in both cubes")
            if _property_ == "DEC_centre_deg": to_do_list.append(
                "  - Check the DEC_centre_deg to get the same value in both cubes")
            if _property_ == "pixel_size_arcsec": to_do_list.append("  - The pixel size of the cubes is not the same!")
            if _property_ == "kernel_size_arcsec": to_do_list.append(
                "  - The kernel size of the cubes is not the same!")
            if _property_ == "n_cols": to_do_list.append(
                "  - The number of columns is not the same! Trim the largest cube")
            if _property_ == "n_rows": to_do_list.append(
                "  - The number of rows is not the same! Trim the largest cube")
            if _property_ == "x_max": to_do_list.append("  - x_max is not the same!")
            if _property_ == "y_max": to_do_list.append("  - y_max is not the same!")

    pixel_size_arcsec = red_cube.pixel_size_arcsec

    x_peak_red = red_cube.x_peak_median
    y_peak_red = red_cube.y_peak_median
    x_peak_blue = blue_cube.x_peak_median
    y_peak_blue = blue_cube.y_peak_median
    delta_x = x_peak_blue - x_peak_red
    delta_y = y_peak_blue - y_peak_red

    print("\n> The offsets between the two cubes following tracing the peak are:\n")
    print("  -> delta_RA  (blue -> red) = {}  spaxels         = {} arcsec".format(round(delta_x, 3),
                                                                                  round(delta_x * pixel_size_arcsec,
                                                                                        3)))
    print("  -> delta_DEC (blue -> red) = {}  spaxels         = {} arcsec".format(round(delta_y, 3),
                                                                                  round(delta_y * pixel_size_arcsec,
                                                                                        3)))
    delta = np.sqrt(delta_x ** 2 + delta_y ** 2)
    print("\n     TOTAL     (blue -> red) = {}  spaxels         = {} arcsec      ({}% of the pix size)".format(
        round(delta, 3), round(delta * pixel_size_arcsec, 3), round(delta * 100, 1)))

    if delta > 0.5:
        to_do_list.append("  - delta_RA and delta_DEC differ for more than half a pixel!")
        all_ok = False

    if all_ok:
        print("\n  GREAT! Both cubes are aligned!")
    else:
        print("\n  This is what is needed for aligning the cubes:")
        for item in to_do_list:
            print(item)

    if delta > 0.1 and delta < 0.5:
        print("  However, considering using the values of delta_RA and delta_DEC for improving alignment!")

    if plot:
        mapa_blue = blue_cube.integrated_map / np.nanmedian(blue_cube.integrated_map)
        mapa_red = red_cube.integrated_map / np.nanmedian(red_cube.integrated_map)
        red_cube.plot_map(mapa=(mapa_blue + mapa_red) / 2, log=True, barlabel="Blue and red maps combined",
                          description="Normalized BLUE+RED", cmap="binary_r")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# This needs to be updated!
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# def save_bluered_fits_file(blue_cube,red_cube, fits_file, fcalb=[0], fcalr=[0], ADR=False, objeto="", description="", trimb=[0], trimr=[0]):
#     """
#     Routine combine blue + red files and save result in a fits file fits file

#     Parameters
#     ----------
#     Combined cube:
#         Combined cube
#     Header:
#         Header
#     """

#     # Prepare the red+blue datacube
#     print("\n> Combining blue + red datacubes...")

#     if trimb[0] == 0:
#         lb=blue_cube.wavelength
#         b=blue_cube.data
#     else:
#         print("  Trimming blue cube in range [{},{}]".format(trimb[0],trimb[1]))
#         index_min = np.searchsorted(blue_cube.wavelength, trimb[0])
#         index_max = np.searchsorted(blue_cube.wavelength, trimb[1])+1
#         lb=blue_cube.wavelength[index_min:index_max]
#         b=blue_cube.data[index_min:index_max]
#         fcalb=fcalb[index_min:index_max]

#     if trimr[0] == 0:
#         lr=red_cube.wavelength
#         r=red_cube.data
#     else:
#         print("  Trimming red cube in range [{},{}]".format(trimr[0],trimr[1]))
#         index_min = np.searchsorted(red_cube.wavelength, trimr[0])
#         index_max = np.searchsorted(red_cube.wavelength, trimr[1])+1
#         lr=red_cube.wavelength[index_min:index_max]
#         r=red_cube.data[index_min:index_max]
#         fcalr=fcalr[index_min:index_max]

#     l=np.concatenate((lb,lr), axis=0)
#     blue_red_datacube=np.concatenate((b,r), axis=0)

#     if fcalb[0] == 0:
#         print("  No absolute flux calibration included")
#     else:
#         flux_calibration=np.concatenate((fcalb,fcalr), axis=0)


#     if objeto == "" : description = "UNKNOWN OBJECT"

#     fits_image_hdu = fits.PrimaryHDU(blue_red_datacube)
#     #    errors = combined_cube.data*0  ### TO BE DONE
#     #    error_hdu = fits.ImageHDU(errors)

#     wavelengths_hdu = fits.ImageHDU(l)

#     fits_image_hdu.header['ORIGIN'] = 'Combined datacube from KOALA Python scripts'

#     fits_image_hdu.header['BITPIX']  =  16
#     fits_image_hdu.header['NAXIS']   =   3
#     fits_image_hdu.header['NAXIS1']  =   len(l)
#     fits_image_hdu.header['NAXIS2']  =   blue_red_datacube.shape[1]        ##### CHECK !!!!!!!
#     fits_image_hdu.header['NAXIS2']  =   blue_red_datacube.shape[2]

#     fits_image_hdu.header['OBJECT'] = objeto
#     fits_image_hdu.header['RAcen'] = blue_cube.RA_centre_deg
#     fits_image_hdu.header['DECcen'] = blue_cube.DEC_centre_deg
#     fits_image_hdu.header['PIXsize'] = blue_cube.pixel_size_arcsec
#     fits_image_hdu.header['Ncols'] = blue_cube.data.shape[2]
#     fits_image_hdu.header['Nrows'] = blue_cube.data.shape[1]
#     fits_image_hdu.header['PA'] = blue_cube.PA
# #    fits_image_hdu.header["CTYPE1"] = 'LINEAR  '
# #    fits_image_hdu.header["CRVAL1"] = wavelength[0]
# #    fits_image_hdu.header["CRPIX1"] = 1.
# #    fits_image_hdu.header["CDELT1"] = (wavelength[-1]-wavelength[0])/len(wavelength)
# #    fits_image_hdu.header["CD1_1"]  = (wavelength[-1]-wavelength[0])/len(wavelength)
# #    fits_image_hdu.header["LTM1_1"] = 1.

#     fits_image_hdu.header['COFILES'] = blue_cube.number_of_combined_files   # Number of combined files
#     fits_image_hdu.header['OFFSETS'] = blue_cube.offsets_files            # Offsets

#     fits_image_hdu.header['ADRCOR'] = np.str(ADR)

#     if fcalb[0] == 0:
#         fits_image_hdu.header['FCAL'] = "False"
#         flux_correction_hdu = fits.ImageHDU(0*l)
#     else:
#         flux_correction = flux_calibration
#         flux_correction_hdu = fits.ImageHDU(flux_correction)
#         fits_image_hdu.header['FCAL'] = "True"

#     if description == "":
#         description = flux_calibration.description
#     fits_image_hdu.header['DESCRIP'] = description


# #    hdu_list = fits.HDUList([fits_image_hdu, error_hdu])
#     hdu_list = fits.HDUList([fits_image_hdu, wavelengths_hdu, flux_correction_hdu])
#     hdu_list.writeto(fits_file, overwrite=True)
#     print("\n> Combined datacube saved to file ",fits_file)
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
#
# Extra tools for analysis, Angel 21st October 2017
#
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def read_table(fichero, formato):
    """
    Read data from and txt file (sorted by columns), the type of data
    (string, integer or float) MUST be given in "formato".
    This routine will ONLY read the columns for which "formato" is defined.
    E.g. for a txt file with 7 data columns, using formato=["f", "f", "s"] will only read the 3 first columns.

    Parameters
    ----------
    fichero:
        txt file to be read.
    formato:
        List with the format of each column of the data, using:\n
        "i" for a integer\n
        "f" for a float\n
        "s" for a string (text)

    Example
    -------
    >>> el_center,el_fnl,el_name = read_table("lineas_c89_python.dat", ["f", "f", "s"] )
    """
    datos_len = len(formato)
    datos = [[] for x in range(datos_len)]
    for i in range(0, datos_len):
        if formato[i] == "i": datos[i] = np.loadtxt(fichero, skiprows=0, unpack=True, usecols=[i], dtype=int)
        if formato[i] == "s": datos[i] = np.loadtxt(fichero, skiprows=0, unpack=True, usecols=[i], dtype=str)
        if formato[i] == "f": datos[i] = np.loadtxt(fichero, skiprows=0, unpack=True, usecols=[i], dtype=float)
    return datos


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def array_to_text_file(data, filename="array.dat", verbose=True):
    """
    Write array into a text file.

    Parameters
    ----------
    data: float
        flux per wavelenght
    filename: string (default = "array.dat")
        name of the text file where the data will be written.

    Example
    -------
    >>> array_to_text_file(data, filename="data.dat" )
    """
    f = open(filename, "w")
    for i in range(len(data)):
        escribe = np.str(data[i]) + " \n"
        f.write(escribe)
    f.close()
    if verbose: print("\n> Array saved in text file", filename, " !!")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def spectrum_to_text_file(wavelength, flux, filename="spectrum.txt", verbose=True):
    """
    Write given 1D spectrum into a text file.

    Parameters
    ----------
    wavelenght: float
        wavelength.
    flux: float
        flux per wavelenght
    filename: string (default = "spectrum.txt")
        name of the text file where the data will be written.

    Example
    -------
    >>> spectrum_to_text_file(wavelength, spectrum, filename="fantastic_spectrum.txt" )
    """
    f = open(filename, "w")
    for i in range(len(wavelength)):
        escribe = np.str(wavelength[i]) + "  " + np.str(flux[i]) + " \n"
        f.write(escribe)
    f.close()
    if verbose: print('\n> Spectrum saved in text file :\n  "' + filename + '"')


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def spectrum_to_fits_file(wavelength, flux, filename="spectrum.fits", name="spectrum", exptime=1,
                          CRVAL1_CDELT1_CRPIX1=[0, 0, 0]):
    """
    Routine to save a given 1D spectrum into a fits file.

    If CRVAL1_CDELT1_CRPIX1 it not given, it assumes a LINEAR dispersion,
    with Delta_pix = (wavelength[-1]-wavelength[0])/(len(wavelength)-1).

    Parameters
    ----------
    wavelenght: float
        wavelength.
    flux: float
        flux per wavelenght
    filename: string (default = "spectrum.fits")
        name of the fits file where the data will be written.
    Example
    -------
    >>> spectrum_to_fits_file(wavelength, spectrum, filename="fantastic_spectrum.fits",
                              exptime=600,name="POX 4")
    """
    hdu = fits.PrimaryHDU()
    hdu.data = (flux)
    hdu.header['ORIGIN'] = 'Data from KOALA Python scripts'
    # Wavelength calibration
    hdu.header['NAXIS'] = 1
    hdu.header['NAXIS1'] = len(wavelength)
    hdu.header["CTYPE1"] = 'Wavelength'
    hdu.header["CUNIT1"] = 'Angstroms'
    if CRVAL1_CDELT1_CRPIX1[0] == 0:
        hdu.header["CRVAL1"] = wavelength[0]
        hdu.header["CRPIX1"] = 1.
        hdu.header["CDELT1"] = (wavelength[-1] - wavelength[0]) / (len(wavelength) - 1)
    else:
        hdu.header["CRVAL1"] = CRVAL1_CDELT1_CRPIX1[0]  # 7.692370611909E+03  / Co-ordinate value of axis 1
        hdu.header["CDELT1"] = CRVAL1_CDELT1_CRPIX1[1]  # 1.575182431607E+00
        hdu.header["CRPIX1"] = CRVAL1_CDELT1_CRPIX1[2]  # 1024. / Reference pixel along axis 1
    # Extra info
    hdu.header['OBJECT'] = name
    hdu.header["TOTALEXP"] = exptime
    hdu.header['HISTORY'] = 'Spectrum derived using the KOALA Python pipeline'
    hdu.header['HISTORY'] = 'Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany et al.'
    hdu.header['HISTORY'] = version
    now = datetime.datetime.now()
    hdu.header['HISTORY'] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    hdu.header['DATE'] = now.strftime("%Y-%m-%dT%H:%M:%S")  # '2002-09-16T18:52:44'   # /Date of FITS file creation

    hdu.writeto(filename, overwrite=True)
    print("\n> Spectrum saved in fits file", filename, " !!")
    if name == "spectrum": print("  No name given to the spectrum, named 'spectrum'.")
    if exptime == 1: print("  No exposition time given, assumed exptime = 1")
    if CRVAL1_CDELT1_CRPIX1[0] == 0: print("  CRVAL1_CDELT1_CRPIX1 values not given, using ", wavelength[0], "1",
                                           (wavelength[-1] - wavelength[0]) / (len(wavelength) - 1))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def apply_z(lambdas, z=0, v_rad=0, ref_line="Ha", l_ref=6562.82, l_obs=6562.82, verbose=True):
    if ref_line == "Ha": l_ref = 6562.82
    if ref_line == "O3": l_ref = 5006.84
    if ref_line == "Hb": l_ref = 4861.33

    if v_rad != 0:
        z = v_rad / C
    if z == 0:
        if verbose:
            print("  Using line {}, l_rest = {:.2f}, observed at l_obs = {:.2f}. ".format(ref_line, l_ref, l_obs))
        z = l_obs / l_ref - 1.
        v_rad = z * C

    zlambdas = (z + 1) * np.array(lambdas)

    if verbose:
        print("  Computing observed wavelengths using v_rad = {:.2f} km/s, redshift z = {:.06} :".format(v_rad, z))
        print("  REST :", lambdas)
        print("  z    :", np.round(zlambdas, 2))

    return zlambdas


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def gauss(x, x0, y0, sigma):
    p = [x0, y0, sigma]
    return p[1] * np.exp(-0.5 * ((x - p[0]) / p[2]) ** 2)


def gauss_fix_x0(x, x0, y0, sigma):
    p = [y0, sigma]
    return p[0] * np.exp(-0.5 * ((x - x0) / p[1]) ** 2)


def gauss_flux(y0, sigma):  ### THIS DOES NOT WORK...
    return y0 * sigma * np.sqrt(2 * np.pi)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def substract_given_gaussian(wavelength, spectrum, centre, peak=0, sigma=0, flux=0, search_peak=False,
                             allow_absorptions=False,
                             lowlow=20, lowhigh=10, highlow=10, highhigh=20,
                             lmin=0, lmax=0, fmin=0, fmax=0, plot=True, fcal=False, verbose=True):
    """
    Substract a give Gaussian to a spectrum after fitting the continuum.
    """
    do_it = False
    # Check that we have the numbers!
    if peak != 0 and sigma != 0: do_it = True

    if peak == 0 and flux != 0 and sigma != 0:
        # flux = peak * sigma * np.sqrt(2*np.pi)
        peak = flux / (sigma * np.sqrt(2 * np.pi))
        do_it = True

    if sigma == 0 and flux != 0 and peak != 0:
        # flux = peak * sigma * np.sqrt(2*np.pi)
        sigma = flux / (peak * np.sqrt(2 * np.pi))
        do_it = True

    if flux == 0 and sigma != 0 and peak != 0:
        flux = peak * sigma * np.sqrt(2 * np.pi)
        do_it = True

    if sigma != 0 and search_peak == True:   do_it = True

    if do_it == False:
        print(
            "> Error! We need data to proceed! Give at least two of [peak, sigma, flux], or sigma and force peak to f[centre]")
        s_s = spectrum
    else:
        # Setup wavelength limits
        if lmin == 0:
            lmin = centre - 65.  # By default, +-65 A with respect to line
        if lmax == 0:
            lmax = centre + 65.

        # Extract subrange to fit
        w_spec = []
        f_spec = []
        w_spec.extend(
            (wavelength[i]) for i in range(len(wavelength)) if (wavelength[i] > lmin and wavelength[i] < lmax))
        f_spec.extend((spectrum[i]) for i in range(len(wavelength)) if (wavelength[i] > lmin and wavelength[i] < lmax))

        # Setup min and max flux values in subrange to fit
        if fmin == 0:
            fmin = np.nanmin(f_spec)
        if fmax == 0:
            fmax = np.nanmax(f_spec)

            # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to centre
        w_cont = []
        f_cont = []
        w_cont.extend((w_spec[i]) for i in range(len(w_spec)) if
                      (w_spec[i] > centre - lowlow and w_spec[i] < centre - lowhigh) or (
                                  w_spec[i] > centre + highlow and w_spec[i] < centre + highhigh))
        f_cont.extend((f_spec[i]) for i in range(len(w_spec)) if
                      (w_spec[i] > centre - lowlow and w_spec[i] < centre - lowhigh) or (
                                  w_spec[i] > centre + highlow and w_spec[i] < centre + highhigh))

        # Linear Fit to continuum
        try:
            mm, bb = np.polyfit(w_cont, f_cont, 1)
        except Exception:
            bb = np.nanmedian(spectrum)
            mm = 0.
            if verbose:
                print("      Impossible to get the continuum!")
                print("      Scaling the continuum to the median value")
        continuum = mm * np.array(w_spec) + bb
        # c_cont = mm*np.array(w_cont)+bb
        # rms continuum
        # rms_cont = np.nansum([ np.abs(f_cont[i] - c_cont[i])  for i in range(len(w_cont)) ]) / len(c_cont)

        if search_peak:
            # Search for index here w_spec(index) closest to line
            try:
                min_w = np.abs(np.array(w_spec) - centre)
                mini = np.nanmin(min_w)
                peak = f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]
                flux = peak * sigma * np.sqrt(2 * np.pi)
                if verbose: print("    Using peak as f[", np.round(centre, 2), "] = ", np.round(peak, 2),
                                  " and sigma = ", np.round(sigma, 2), "    flux = ", np.round(flux, 2))
            except Exception:
                if verbose: print("    Error trying to get the peak as requested wavelength is ", np.round(centre, 2),
                                  "! Ignoring this fit!")
                peak = 0.
                flux = -0.0001

        no_substract = False
        if flux < 0:
            if allow_absorptions == False:
                if verbose and np.isnan(centre) == False: print(
                    "    WARNING! This is an ABSORPTION Gaussian! As requested, this Gaussian is NOT substracted!")
                no_substract = True
        if no_substract == False:
            if verbose: print(
                "    Substracting Gaussian at {:7.1f}  with peak ={:10.4f}   sigma ={:6.2f}  and flux ={:9.4f}".format(
                    centre, peak, sigma, flux))

            gaussian_fit = gauss(w_spec, centre, peak, sigma)

            index = 0
            s_s = np.zeros_like(spectrum)
            for wave in range(len(wavelength)):
                s_s[wave] = spectrum[wave]
                if wavelength[wave] == w_spec[0]:
                    s_s[wave] = f_spec[0] - gaussian_fit[0]
                    index = 1
                if wavelength[wave] > w_spec[0] and wavelength[wave] <= w_spec[-1]:
                    s_s[wave] = f_spec[index] - gaussian_fit[index]
                    index = index + 1
            if plot:
                plt.figure(figsize=(10, 4))
                plt.plot(np.array(w_spec), np.array(f_spec), "b", lw=3, alpha=0.5)
                plt.minorticks_on()
                plt.xlabel("Wavelength [$\mathrm{\AA}$]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
                else:
                    plt.ylabel("Flux [ counts ]")
                plt.xlim(lmin, lmax)
                plt.ylim(fmin, fmax)

                # Vertical line at line
                plt.axvline(x=centre, color='k', linestyle='-', alpha=0.8)
                # Horizontal line at y = 0
                plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
                # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
                plt.axvspan(centre + highlow, centre + highhigh, facecolor='g', alpha=0.15, zorder=3)
                plt.axvspan(centre - lowlow, centre - lowhigh, facecolor='g', alpha=0.15, zorder=3)
                # Plot linear fit for continuum
                plt.plot(w_spec, continuum, "g--")
                # Plot Gaussian fit
                plt.plot(w_spec, gaussian_fit + continuum, 'r-', alpha=0.8)
                # Vertical lines to emission line
                # plt.axvline(x= low_limit, color='k', linestyle=':', alpha=0.5)
                # plt.axvline(x= high_limit, color='k', linestyle=':', alpha=0.5)
                # Plot residuals
                # plt.plot(w_spec, residuals, 'k')
                # plt.title('Fit: x0=%.2f y0=%.2e sigma=%.2f  flux=%.2e  rms=%.3e' % (fit[0], fit[1], fit[2], gaussian_flux, rms_fit))
                plt.show()
                plt.close()

                plt.figure(figsize=(10, 4))
                plt.plot(wavelength, spectrum, "r")
                plt.plot(wavelength, s_s, "c")
                plt.minorticks_on()
                plt.xlabel("Wavelength [$\mathrm{\AA}$]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
                else:
                    plt.ylabel("Flux [ counts ]")
                plt.xlim(lmin, lmax)
                plt.ylim(fmin, fmax)
                plt.show()
                plt.close()
        else:
            s_s = spectrum
    return s_s


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fluxes(wavelength, s, line, lowlow=14, lowhigh=6, highlow=6, highhigh=14, lmin=0, lmax=0, fmin=0, fmax=0,
           broad=2.355, plot=True, verbose=True, plot_sus=False, fcal=True, fit_continuum=True, median_kernel=35,
           warnings=True):  # Broad is FWHM for Gaussian sigma= 1,
    """
    Provides integrated flux and perform a Gaussian fit to a given emission line.
    It follows the task "splot" in IRAF, with "e -> e" for integrated flux and "k -> k" for a Gaussian.

    Info from IRAF:\n
        - Integrated flux:\n
            center = sum (w(i) * (I(i)-C(i))**3/2) / sum ((I(i)-C(i))**3/2) (NOT USED HERE)\n
            continuum = C(midpoint) (NOT USED HERE) \n
            flux = sum ((I(i)-C(i)) * (w(i2) - w(i1)) / (i2 - i1)\n
            eq. width = sum (1 - I(i)/C(i))\n
        - Gaussian Fit:\n
             I(w) = cont + core * exp (-0.5*((w-center)/sigma)**2)\n
             fwhm = 2.355 * sigma\n
             flux = core * sigma * sqrt (2*pi)\n
             eq. width = abs (flux) / cont\n

    Result
    ------

    This routine provides a list compiling the results. The list has the the following format:

        resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]

    "spectrum" in resultado[11] is the spectrum-fit (New 22 Jan 2019).

    Parameters
    ----------
    wavelenght: float
        wavelength.
    spectrum: float
        flux per wavelenght
    line: float
        approx. observed central wavelength of emission line to fit.
    lmin, lmax: float
        wavelength range to be analized
    fmin, fmax: float (defaut = 0, 0.)
        minimum and maximun values of flux to be plotted.
        If 0 is given (i.e. defaul) the routine uses the nanmin and nanmax values of the given spectrum.
    plot: boolean (default = True)
        Plot a figure with the emission lines identifications.
    verbose: boolean (default = True)
        Print results.
    fit_continuum: boolean (default = True)
        Perform a linear fit of the continuum using all data, otherwise it just does a linear fit considering only the two median values in each continuum range.
    median_kernel: odd integer (defaut = 35)
        size of the median filter to be applied to the continuum.
    Example
    -------
    >>> resultado = fluxes(wavelength, spectrum, 6603, fmin=-5.0E-17, fmax=2.0E-16, plot=True, verbose=False)
    """
    # s must be an array, no a list
    try:
        index_maximo_del_rango = s.tolist().index(np.nanmax(s))
        # print " is AN ARRAY"
    except Exception:
        # print " s is A LIST  -> must be converted into an ARRAY"
        s = np.array(s)

    # Setup wavelength limits
    if lmin == 0:
        lmin = line - 65.  # By default, +-65 A with respect to line
    if lmax == 0:
        lmax = line + 65.

    # Extract subrange to fit
    w_spec = []
    f_spec = []
    w_spec.extend((wavelength[i]) for i in range(len(wavelength)) if (wavelength[i] > lmin and wavelength[i] < lmax))
    f_spec.extend((s[i]) for i in range(len(wavelength)) if (wavelength[i] > lmin and wavelength[i] < lmax))

    if np.isnan(np.nanmedian(f_spec)):
        # The data are NAN!! Nothing to do
        if verbose: print("    There is no valid data in the wavelength range [{},{}] !!".format(lmin, lmax))

        resultado = [0, line, 0, 0, 0, 0, 0, 0, 0, 0, 0, s]

        return resultado

    else:

        ## 20 Sep 2020
        f_spec_m = signal.medfilt(f_spec, median_kernel)  # median_kernel = 35 default

        # Remove nans
        median_value = np.nanmedian(f_spec)
        f_spec = [median_value if np.isnan(x) else x for x in f_spec]

        # Setup min and max flux values in subrange to fit
        if fmin == 0:
            fmin = np.nanmin(f_spec)
        if fmax == 0:
            fmax = np.nanmax(f_spec)

            # We have to find some "guess numbers" for the Gaussian. Now guess_centre is line
        guess_centre = line

        # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to guess_centre

        w_cont = []
        f_cont = []
        w_cont.extend((w_spec[i]) for i in range(len(w_spec)) if
                      (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh) or (
                                  w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh))
        f_cont.extend((f_spec[i]) for i in range(len(w_spec)) if
                      (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh) or (
                                  w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh))

        if fit_continuum:
            # Linear Fit to continuum
            f_cont_filtered = sig.medfilt(f_cont, np.int(median_kernel))
            # print line #f_cont
            #        if line == 8465.0:
            #            print w_cont
            #            print f_cont_filtered
            #            plt.plot(w_cont,f_cont_filtered)
            #            plt.show()
            #            plt.close()
            #            warnings=True
            try:
                mm, bb = np.polyfit(w_cont, f_cont_filtered, 1)
            except Exception:
                bb = np.nanmedian(f_cont_filtered)
                mm = 0.
                if warnings:
                    print("    Impossible to get the continuum!")
                    print("    Scaling the continuum to the median value b = ", bb, ":  cont =  0 * w_spec  + ", bb)
            continuum = mm * np.array(w_spec) + bb
            c_cont = mm * np.array(w_cont) + bb

        else:
            # Median value in each continuum range  # NEW 15 Sep 2019
            w_cont_low = []
            f_cont_low = []
            w_cont_low.extend((w_spec[i]) for i in range(len(w_spec)) if
                              (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh))
            f_cont_low.extend((f_spec[i]) for i in range(len(w_spec)) if
                              (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh))
            median_w_cont_low = np.nanmedian(w_cont_low)
            median_f_cont_low = np.nanmedian(f_cont_low)
            w_cont_high = []
            f_cont_high = []
            w_cont_high.extend((w_spec[i]) for i in range(len(w_spec)) if
                               (w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh))
            f_cont_high.extend((f_spec[i]) for i in range(len(w_spec)) if
                               (w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh))
            median_w_cont_high = np.nanmedian(w_cont_high)
            median_f_cont_high = np.nanmedian(f_cont_high)

            b = (median_f_cont_low - median_f_cont_high) / (median_w_cont_low - median_w_cont_high)
            a = median_f_cont_low - b * median_w_cont_low

            continuum = a + b * np.array(w_spec)
            c_cont = a + b * np.array(w_cont)

            # rms continuum
        rms_cont = np.nansum([np.abs(f_cont[i] - c_cont[i]) for i in range(len(w_cont))]) / len(c_cont)

        # Search for index here w_spec(index) closest to line
        min_w = np.abs(np.array(w_spec) - line)
        mini = np.nanmin(min_w)
        #    guess_peak = f_spec[min_w.tolist().index(mini)]   # WE HAVE TO SUSTRACT CONTINUUM!!!
        guess_peak = f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]

        # LOW limit
        low_limit = 0
        w_fit = []
        f_fit = []
        w_fit.extend(
            (w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre - 15 and w_spec[i] < guess_centre))
        f_fit.extend(
            (f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre - 15 and w_spec[i] < guess_centre))
        if fit_continuum:
            c_fit = mm * np.array(w_fit) + bb
        else:
            c_fit = b * np.array(w_fit) + a

        fs = []
        ws = []
        for ii in range(len(w_fit) - 1, 1, -1):
            if f_fit[ii] / c_fit[ii] < 1.05 and f_fit[ii - 1] / c_fit[ii - 1] < 1.05 and low_limit == 0: low_limit = \
            w_fit[ii]
            #        if f_fit[ii]/c_fit[ii] < 1.05 and low_limit == 0: low_limit = w_fit[ii]
            fs.append(f_fit[ii] / c_fit[ii])
            ws.append(w_fit[ii])
        if low_limit == 0:
            sorted_by_flux = np.argsort(fs)
            try:
                low_limit = ws[sorted_by_flux[0]]
            except Exception:
                plot = True
                low_limit = 0

        # HIGH LIMIT
        high_limit = 0
        w_fit = []
        f_fit = []
        w_fit.extend(
            (w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre and w_spec[i] < guess_centre + 15))
        f_fit.extend(
            (f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre and w_spec[i] < guess_centre + 15))
        if fit_continuum:
            c_fit = mm * np.array(w_fit) + bb
        else:
            c_fit = b * np.array(w_fit) + a

        fs = []
        ws = []
        for ii in range(len(w_fit) - 1):
            if f_fit[ii] / c_fit[ii] < 1.05 and f_fit[ii + 1] / c_fit[ii + 1] < 1.05 and high_limit == 0: high_limit = \
            w_fit[ii]
            #        if f_fit[ii]/c_fit[ii] < 1.05 and high_limit == 0: high_limit = w_fit[ii]
            fs.append(f_fit[ii] / c_fit[ii])
            ws.append(w_fit[ii])
        if high_limit == 0:
            sorted_by_flux = np.argsort(fs)
            try:
                high_limit = ws[sorted_by_flux[0]]
            except Exception:
                plot = True
                high_limit = 0

                # Guess centre will be the highest value in the range defined by [low_limit,high_limit]

        try:
            rango = np.where((high_limit >= wavelength) & (low_limit <= wavelength))
            index_maximo_del_rango = s.tolist().index(np.nanmax(s[rango]))
            guess_centre = wavelength[index_maximo_del_rango]
        except Exception:
            guess_centre = line  ####  It was 0 before

        # Fit a Gaussian to data - continuum
        p0 = [guess_centre, guess_peak, broad / 2.355]  # broad is the Gaussian sigma, 1.0 for emission lines
        try:
            fit, pcov = curve_fit(gauss, w_spec, f_spec - continuum, p0=p0,
                                  maxfev=10000)  # If this fails, increase maxfev...
            fit_error = np.sqrt(np.diag(pcov))

            # New 28th Feb 2019: Check central value between low_limit and high_limit
            # Better: between guess_centre - broad, guess_centre + broad
            # If not, redo fit fixing central value to the peak (it does not work... just fix FWHM= (high_limit-low_limit)/2.5 )

            if verbose != False: print(
                " ----------------------------------------------------------------------------------------")
            #        if low_limit < fit[0] < high_limit:
            if fit[0] < guess_centre - broad or fit[0] > guess_centre + broad:
                #            if verbose: print "  Fitted center wavelength", fit[0],"is NOT in the range [",low_limit,",",high_limit,"]"
                if verbose: print("    Fitted center wavelength", fit[0], "is NOT in the expected range [",
                                  guess_centre - broad, ",", guess_centre + broad, "]")

                #            print "Re-do fitting fixing center wavelength"
                #            p01 = [guess_peak, broad]
                #            fit1, pcov1 = curve_fit(gauss_fix_x0, w_spec, f_spec-continuum, p0=p01, maxfev=100000)   # If this fails, increase maxfev...
                #            fit_error1 = np.sqrt(np.diag(pcov1))
                #            fit[0]=guess_centre
                #            fit_error[0] = 0.
                #            fit[1] = fit1[0]
                #            fit_error[1] = fit_error1[0]
                #            fit[2] = fit1[1]
                #            fit_error[2] = fit_error1[1]

                fit[0] = guess_centre
                fit_error[0] = 0.000001
                fit[1] = guess_peak
                fit_error[1] = 0.000001
                fit[2] = broad / 2.355
                fit_error[2] = 0.000001
            else:
                if verbose: print("    Fitted center wavelength", fit[0], "IS in the expected range [",
                                  guess_centre - broad, ",", guess_centre + broad, "]")

            if verbose: print("    Fit parameters =  ", fit[0], fit[1], fit[2])
            if fit[2] == broad and warnings == True:
                print("    WARNING: Fit in", fit[0],
                      "failed! Using given centre wavelength (cw), peak at (cv) & sigma = broad/2.355 given.")
            gaussian_fit = gauss(w_spec, fit[0], fit[1], fit[2])

            # Estimate rms of the Gaussian fit in range [low_limit, high_limit]
            residuals = f_spec - gaussian_fit - continuum
            rms_fit = np.nansum([((residuals[i] ** 2) / (len(residuals) - 2)) ** 0.5 for i in range(len(w_spec)) if
                                 (w_spec[i] >= low_limit and w_spec[i] <= high_limit)])

            # Fluxes, FWHM and Eq. Width calculations
            gaussian_flux = gauss_flux(fit[1], fit[2])
            error1 = np.abs(gauss_flux(fit[1] + fit_error[1], fit[2]) - gaussian_flux)
            error2 = np.abs(gauss_flux(fit[1], fit[2] + fit_error[2]) - gaussian_flux)
            gaussian_flux_error = 1 / (1 / error1 ** 2 + 1 / error2 ** 2) ** 0.5

            fwhm = fit[2] * 2.355
            fwhm_error = fit_error[2] * 2.355
            fwhm_vel = fwhm / fit[0] * C
            fwhm_vel_error = fwhm_error / fit[0] * C

            gaussian_ew = gaussian_flux / np.nanmedian(f_cont)
            gaussian_ew_error = gaussian_ew * gaussian_flux_error / gaussian_flux

            # Integrated flux
            # IRAF: flux = sum ((I(i)-C(i)) * (w(i2) - w(i1)) / (i2 - i2)
            flux = np.nansum([(f_spec[i] - continuum[i]) * (w_spec[i + 1] - w_spec[i]) for i in range(len(w_spec)) if
                              (w_spec[i] >= low_limit and w_spec[i] <= high_limit)])
            flux_error = rms_cont * (high_limit - low_limit)
            wave_resolution = (wavelength[-1] - wavelength[0]) / len(wavelength)
            ew = wave_resolution * np.nansum([(1 - f_spec[i] / continuum[i]) for i in range(len(w_spec)) if
                                              (w_spec[i] >= low_limit and w_spec[i] <= high_limit)])
            ew_error = np.abs(ew * flux_error / flux)
            gauss_to_integrated = gaussian_flux / flux * 100.

            index = 0
            s_s = np.zeros_like(s)
            for wave in range(len(wavelength)):
                s_s[wave] = s[wave]
                if wavelength[wave] == w_spec[0]:
                    s_s[wave] = f_spec[0] - gaussian_fit[0]
                    index = 1
                if wavelength[wave] > w_spec[0] and wavelength[wave] <= w_spec[-1]:
                    s_s[wave] = f_spec[index] - gaussian_fit[index]
                    index = index + 1

            # Plotting
            ptitle = 'Fit: x0=%.2f y0=%.2e sigma=%.2f  flux=%.2e  rms=%.3e' % (
            fit[0], fit[1], fit[2], gaussian_flux, rms_fit)
            if plot:
                plt.figure(figsize=(10, 4))
                # Plot input spectrum
                plt.plot(np.array(w_spec), np.array(f_spec), "b", lw=3, alpha=0.8)
                # Plot median input spectrum
                plt.plot(np.array(w_spec), np.array(f_spec_m), "orange", lw=3, alpha=0.5)  # 2021: era "g"
                # Plot spectrum - gauss subtracted
                plt.plot(wavelength, s_s, "g", lw=3, alpha=0.6)

                plt.minorticks_on()
                plt.xlabel("Wavelength [$\mathrm{\AA}$ ]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
                else:
                    plt.ylabel("Flux [ counts ]")
                plt.xlim(lmin, lmax)
                plt.ylim(fmin, fmax)

                # Vertical line at guess_centre
                plt.axvline(x=guess_centre, color='r', linestyle='-', alpha=0.3)
                # Horizontal line at y = 0
                plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
                # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
                plt.axvspan(guess_centre + highlow, guess_centre + highhigh, facecolor='g', alpha=0.15, zorder=3)
                plt.axvspan(guess_centre - lowlow, guess_centre - lowhigh, facecolor='g', alpha=0.15, zorder=3)
                # Plot linear fit for continuum
                plt.plot(w_spec, continuum, "g--")
                # Plot Gaussian fit
                plt.plot(w_spec, gaussian_fit + continuum, 'r-', alpha=0.8)
                # Vertical line at Gaussian center
                plt.axvline(x=fit[0], color='k', linestyle='-', alpha=0.5)
                # Vertical lines to emission line
                plt.axvline(x=low_limit, color='k', linestyle=':', alpha=0.5)
                plt.axvline(x=high_limit, color='k', linestyle=':', alpha=0.5)
                # Plot residuals
                plt.plot(w_spec, residuals, 'k')
                plt.title(ptitle)
                plt.show()

            # Printing results
            if verbose:
                print("\n  - Gauss and continuum fitting + integrated flux calculations:\n")
                print("    rms continuum = %.3e erg/cm/s/A " % (rms_cont))
                print("    Gaussian Fit parameters: x0 = ( %.2f +- %.2f )  A " % (fit[0], fit_error[0]))
                print("                             y0 = ( %.3f +- %.3f )  1E-16 erg/cm2/s/A" % (
                fit[1] / 1E-16, fit_error[1] / 1E-16))
                print("                          sigma = ( %.3f +- %.3f )  A" % (fit[2], fit_error[2]))
                print("                        rms fit = %.3e erg/cm2/s/A" % (rms_fit))
                print("    Gaussian Flux = ( %.2f +- %.2f ) 1E-16 erg/s/cm2         (error = %.1f per cent)" % (
                gaussian_flux / 1E-16, gaussian_flux_error / 1E-16, gaussian_flux_error / gaussian_flux * 100))
                print("    FWHM          = ( %.3f +- %.3f ) A    =   ( %.1f +- %.1f ) km/s " % (
                fwhm, fwhm_error, fwhm_vel, fwhm_vel_error))
                print("    Eq. Width     = ( %.1f +- %.1f ) A" % (-gaussian_ew, gaussian_ew_error))
                print("\n    Integrated flux  = ( %.2f +- %.2f ) 1E-16 erg/s/cm2      (error = %.1f per cent) " % (
                flux / 1E-16, flux_error / 1E-16, flux_error / flux * 100))
                print("    Eq. Width        = ( %.1f +- %.1f ) A" % (ew, ew_error))
                print("    Gauss/Integrated = %.2f per cent " % gauss_to_integrated)

            # Plot independent figure with substraction if requested
            if plot_sus: plot_plot(wavelength, [s, s_s], xmin=lmin, xmax=lmax, ymin=fmin, ymax=fmax, fcal=fcal,
                                   frameon=True, ptitle=ptitle)

            #                     0      1         2                3               4              5      6         7        8        9     10      11
            resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux,
                         flux_error, ew, ew_error, s_s]
            return resultado
        except Exception:
            if verbose:
                print("  - Gaussian fit failed!")
                print("    However, we can compute the integrated flux and the equivalent width:")

            flux = np.nansum([(f_spec[i] - continuum[i]) * (w_spec[i + 1] - w_spec[i]) for i in range(len(w_spec)) if
                              (w_spec[i] >= low_limit and w_spec[i] <= high_limit)])
            flux_error = rms_cont * (high_limit - low_limit)
            wave_resolution = (wavelength[-1] - wavelength[0]) / len(wavelength)
            ew = wave_resolution * np.nansum([(1 - f_spec[i] / continuum[i]) for i in range(len(w_spec)) if
                                              (w_spec[i] >= low_limit and w_spec[i] <= high_limit)])
            ew_error = np.abs(ew * flux_error / flux)

            if verbose:
                print("    Integrated flux  = ( %.2f +- %.2f ) 1E-16 erg/s/cm2      (error = %.1f per cent) " % (
                flux / 1E-16, flux_error / 1E-16, flux_error / flux * 100))
                print("    Eq. Width        = ( %.1f +- %.1f ) A" % (ew, ew_error))

            resultado = [0, guess_centre, 0, 0, 0, 0, 0, flux, flux_error, ew, ew_error,
                         s]  # guess_centre was identified at maximum value in the [low_limit,high_limit] range but Gaussian fit failed

            # Plotting
            if plot:
                plt.figure(figsize=(10, 4))
                plt.plot(np.array(w_spec), np.array(f_spec), "b", lw=3, alpha=0.5)
                plt.minorticks_on()
                plt.xlabel("Wavelength [$\mathrm{\AA}$]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
                else:
                    plt.ylabel("Flux [ counts ]")
                plt.xlim(lmin, lmax)
                plt.ylim(fmin, fmax)

                # Vertical line at guess_centre
                plt.axvline(x=guess_centre, color='r', linestyle='-', alpha=0.5)
                # Horizontal line at y = 0
                plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
                # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
                plt.axvspan(guess_centre + highlow, guess_centre + highhigh, facecolor='g', alpha=0.15, zorder=3)
                plt.axvspan(guess_centre - lowlow, guess_centre - lowhigh, facecolor='g', alpha=0.15, zorder=3)
                # Plot linear fit for continuum
                plt.plot(w_spec, continuum, "g--")
                # Plot Gaussian fit
                #            plt.plot(w_spec, gaussian_fit+continuum, 'r-', alpha=0.8)
                # Vertical line at Gaussian center
                #            plt.axvline(x=fit[0], color='k', linestyle='-', alpha=0.5)
                # Vertical lines to emission line
                plt.axvline(x=low_limit, color='k', linestyle=':', alpha=0.5)
                plt.axvline(x=high_limit, color='k', linestyle=':', alpha=0.5)
                # Plot residuals
                #            plt.plot(w_spec, residuals, 'k')
                plt.title("No Gaussian fit obtained...")
                plt.show()

            return resultado
        # -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def dgauss(x, x0, y0, sigma0, x1, y1, sigma1):
    p = [x0, y0, sigma0, x1, y1, sigma1]
    #         0   1    2      3    4  5
    return p[1] * np.exp(-0.5 * ((x - p[0]) / p[2]) ** 2) + p[4] * np.exp(-0.5 * ((x - p[3]) / p[5]) ** 2)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def dfluxes(wavelength, s, line1, line2, lowlow=25, lowhigh=15, highlow=15, highhigh=25,
            lmin=0, lmax=0, fmin=0, fmax=0,
            broad1=2.355, broad2=2.355, sus_line1=True, sus_line2=True,
            plot=True, verbose=True, plot_sus=False, fcal=True,
            fit_continuum=True, median_kernel=35, warnings=True):  # Broad is FWHM for Gaussian sigma= 1,
    """
    Provides integrated flux and perform a double Gaussian fit.

    Result
    ------

    This routine provides a list compiling the results. The list has the the following format:

        resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]

    "spectrum" in resultado[11] is the spectrum-fit (New 22 Jan 2019).

    Parameters
    ----------
    wavelenght: float
        wavelength.
    spectrum: float
        flux per wavelenght
    line: float
        approx. observed central wavelength of emission line to fit.
    lmin, lmax: float
        wavelength range to be analized
    fmin, fmax: float (defaut = 0, 0.)
        minimum and maximun values of flux to be plotted.
        If 0 is given (i.e. defaul) the routine uses the nanmin and nanmax values of the given spectrum.
    plot: boolean (default = True)
        Plot a figure with the emission lines identifications.
    verbose: boolean (default = True)
        Print results.
    fit_continuum: boolean (default = True)
        Perform a linear fit of the continuum using all data, otherwise it just does a linear fit considering only the two median values in each continuum range.
    median_kernel: odd integer (defaut = 35)
        size of the median filter to be applied to the continuum.
    Example
    -------
    >>> resultado = fluxes(wavelength, spectrum, 6603, fmin=-5.0E-17, fmax=2.0E-16, plot=True, verbose=False)
    """
    # Setup wavelength limits
    if lmin == 0:
        lmin = line1 - 65.  # By default, +-65 A with respect to line
    if lmax == 0:
        lmax = line2 + 65.

    # Extract subrange to fit
    w_spec = []
    f_spec = []
    w_spec.extend((wavelength[i]) for i in range(len(wavelength)) if (wavelength[i] > lmin and wavelength[i] < lmax))
    f_spec.extend((s[i]) for i in range(len(wavelength)) if (wavelength[i] > lmin and wavelength[i] < lmax))

    if np.nanmedian(f_spec) == np.nan: print("  NO HAY DATOS.... todo son NANs!")

    # Setup min and max flux values in subrange to fit
    if fmin == 0:
        fmin = np.nanmin(f_spec)
    if fmax == 0:
        fmax = np.nanmax(f_spec)

        # We have to find some "guess numbers" for the Gaussian
    # Now guess_centre is line
    guess_centre1 = line1
    guess_centre2 = line2
    guess_centre = (guess_centre1 + guess_centre2) / 2.
    # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to guess_centre

    w_cont = []
    f_cont = []
    w_cont.extend((w_spec[i]) for i in range(len(w_spec)) if
                  (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh) or (
                              w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh))
    f_cont.extend((f_spec[i]) for i in range(len(w_spec)) if
                  (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh) or (
                              w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh))

    if fit_continuum:
        # Linear Fit to continuum
        f_cont_filtered = sig.medfilt(f_cont, np.int(median_kernel))
        try:
            mm, bb = np.polyfit(w_cont, f_cont_filtered, 1)
        except Exception:
            bb = np.nanmedian(f_cont_filtered)
            mm = 0.
            if warnings:
                print("  Impossible to get the continuum!")
                print("  Scaling the continuum to the median value")
        continuum = mm * np.array(w_spec) + bb
        c_cont = mm * np.array(w_cont) + bb

    else:
        # Median value in each continuum range  # NEW 15 Sep 2019
        w_cont_low = []
        f_cont_low = []
        w_cont_low.extend((w_spec[i]) for i in range(len(w_spec)) if
                          (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh))
        f_cont_low.extend((f_spec[i]) for i in range(len(w_spec)) if
                          (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh))
        median_w_cont_low = np.nanmedian(w_cont_low)
        median_f_cont_low = np.nanmedian(f_cont_low)
        w_cont_high = []
        f_cont_high = []
        w_cont_high.extend((w_spec[i]) for i in range(len(w_spec)) if
                           (w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh))
        f_cont_high.extend((f_spec[i]) for i in range(len(w_spec)) if
                           (w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh))
        median_w_cont_high = np.nanmedian(w_cont_high)
        median_f_cont_high = np.nanmedian(f_cont_high)

        b = (median_f_cont_low - median_f_cont_high) / (median_w_cont_low - median_w_cont_high)
        a = median_f_cont_low - b * median_w_cont_low

        continuum = a + b * np.array(w_spec)
        c_cont = b * np.array(w_cont) + a

        # rms continuum
    rms_cont = np.nansum([np.abs(f_cont[i] - c_cont[i]) for i in range(len(w_cont))]) / len(c_cont)

    # Search for index here w_spec(index) closest to line
    min_w = np.abs(np.array(w_spec) - line1)
    mini = np.nanmin(min_w)
    guess_peak1 = f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]
    min_w = np.abs(np.array(w_spec) - line2)
    mini = np.nanmin(min_w)
    guess_peak2 = f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]

    # Search for beginning/end of emission line, choosing line +-10
    # 28th Feb 2019: Check central value between low_limit and high_limit

    # LOW limit
    low_limit = 0
    w_fit = []
    f_fit = []
    w_fit.extend(
        (w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre1 - 15 and w_spec[i] < guess_centre1))
    f_fit.extend(
        (f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre1 - 15 and w_spec[i] < guess_centre1))
    if fit_continuum:
        c_fit = mm * np.array(w_fit) + bb
    else:
        c_fit = b * np.array(w_fit) + a

    fs = []
    ws = []
    for ii in range(len(w_fit) - 1, 1, -1):
        if f_fit[ii] / c_fit[ii] < 1.05 and f_fit[ii - 1] / c_fit[ii - 1] < 1.05 and low_limit == 0: low_limit = w_fit[
            ii]
        #        if f_fit[ii]/c_fit[ii] < 1.05 and low_limit == 0: low_limit = w_fit[ii]
        fs.append(f_fit[ii] / c_fit[ii])
        ws.append(w_fit[ii])
    if low_limit == 0:
        sorted_by_flux = np.argsort(fs)
        low_limit = ws[sorted_by_flux[0]]

    # HIGH LIMIT
    high_limit = 0
    w_fit = []
    f_fit = []
    w_fit.extend(
        (w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre2 and w_spec[i] < guess_centre2 + 15))
    f_fit.extend(
        (f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre2 and w_spec[i] < guess_centre2 + 15))
    if fit_continuum:
        c_fit = mm * np.array(w_fit) + bb
    else:
        c_fit = b * np.array(w_fit) + a

    fs = []
    ws = []
    for ii in range(len(w_fit) - 1):
        if f_fit[ii] / c_fit[ii] < 1.05 and f_fit[ii + 1] / c_fit[ii + 1] < 1.05 and high_limit == 0: high_limit = \
        w_fit[ii]
        #        if f_fit[ii]/c_fit[ii] < 1.05 and high_limit == 0: high_limit = w_fit[ii]
        fs.append(f_fit[ii] / c_fit[ii])
        ws.append(w_fit[ii])
    if high_limit == 0:
        sorted_by_flux = np.argsort(fs)
        high_limit = ws[sorted_by_flux[0]]

        # Fit a Gaussian to data - continuum
    p0 = [guess_centre1, guess_peak1, broad1 / 2.355, guess_centre2, guess_peak2,
          broad2 / 2.355]  # broad is the Gaussian sigma, 1.0 for emission lines
    try:
        fit, pcov = curve_fit(dgauss, w_spec, f_spec - continuum, p0=p0,
                              maxfev=10000)  # If this fails, increase maxfev...
        fit_error = np.sqrt(np.diag(pcov))

        # New 28th Feb 2019: Check central value between low_limit and high_limit
        # Better: between guess_centre - broad, guess_centre + broad
        # If not, redo fit fixing central value to the peak (it does not work... just fix FWHM= (high_limit-low_limit)/2.5 )

        if verbose != False: print(
            " ----------------------------------------------------------------------------------------")
        if fit[0] < guess_centre1 - broad1 or fit[0] > guess_centre1 + broad1 or fit[3] < guess_centre2 - broad2 or fit[
            3] > guess_centre2 + broad2:
            if warnings:
                if fit[0] < guess_centre1 - broad1 or fit[0] > guess_centre1 + broad1:
                    print("    Fitted center wavelength", fit[0], "is NOT in the expected range [",
                          guess_centre1 - broad1, ",", guess_centre1 + broad1, "]")
                else:
                    print("    Fitted center wavelength", fit[0], "is in the expected range [", guess_centre1 - broad1,
                          ",", guess_centre1 + broad1, "]")
                if fit[3] < guess_centre2 - broad2 or fit[3] > guess_centre2 + broad2:
                    print("    Fitted center wavelength", fit[3], "is NOT in the expected range [",
                          guess_centre2 - broad2, ",", guess_centre2 + broad2, "]")
                else:
                    print("    Fitted center wavelength", fit[3], "is in the expected range [", guess_centre2 - broad2,
                          ",", guess_centre2 + broad2, "]")
                print("    Fit failed!")

            fit[0] = guess_centre1
            fit_error[0] = 0.000001
            fit[1] = guess_peak1
            fit_error[1] = 0.000001
            fit[2] = broad1 / 2.355
            fit_error[2] = 0.000001
            fit[3] = guess_centre2
            fit_error[3] = 0.000001
            fit[4] = guess_peak2
            fit_error[4] = 0.000001
            fit[5] = broad2 / 2.355
            fit_error[5] = 0.000001
        else:
            if warnings: print("    Fitted center wavelength", fit[0], "is in the expected range [",
                               guess_centre1 - broad1, ",", guess_centre1 + broad1, "]")
            if warnings: print("    Fitted center wavelength", fit[3], "is in the expected range [",
                               guess_centre2 - broad2, ",", guess_centre2 + broad2, "]")

        if warnings:
            print("    Fit parameters =  ", fit[0], fit[1], fit[2])
            print("                      ", fit[3], fit[4], fit[5])
        if fit[2] == broad1 / 2.355 and warnings == True:
            print("    WARNING: Fit in", fit[0],
                  "failed! Using given centre wavelengths (cw), peaks at (cv) & sigmas=broad/2.355 given.")  # CHECK THIS

        gaussian_fit = dgauss(w_spec, fit[0], fit[1], fit[2], fit[3], fit[4], fit[5])

        gaussian_1 = gauss(w_spec, fit[0], fit[1], fit[2])
        gaussian_2 = gauss(w_spec, fit[3], fit[4], fit[5])

        # Estimate rms of the Gaussian fit in range [low_limit, high_limit]
        residuals = f_spec - gaussian_fit - continuum
        rms_fit = np.nansum([((residuals[i] ** 2) / (len(residuals) - 2)) ** 0.5 for i in range(len(w_spec)) if
                             (w_spec[i] >= low_limit and w_spec[i] <= high_limit)])

        # Fluxes, FWHM and Eq. Width calculations  # CHECK THIS , not well done for dfluxes !!!

        gaussian_flux_1 = gauss_flux(fit[1], fit[2])
        gaussian_flux_2 = gauss_flux(fit[4], fit[5])
        gaussian_flux = gaussian_flux_1 + gaussian_flux_2
        if warnings:
            print("    Gaussian flux  =  ", gaussian_flux_1, " + ", gaussian_flux_2, " = ", gaussian_flux)
            print("    Gaussian ratio =  ", gaussian_flux_1 / gaussian_flux_2)

        error1 = np.abs(gauss_flux(fit[1] + fit_error[1], fit[2]) - gaussian_flux)
        error2 = np.abs(gauss_flux(fit[1], fit[2] + fit_error[2]) - gaussian_flux)
        gaussian_flux_error = 1 / (1 / error1 ** 2 + 1 / error2 ** 2) ** 0.5

        fwhm = fit[2] * 2.355
        fwhm_error = fit_error[2] * 2.355
        fwhm_vel = fwhm / fit[0] * C
        fwhm_vel_error = fwhm_error / fit[0] * C

        gaussian_ew = gaussian_flux / np.nanmedian(f_cont)
        gaussian_ew_error = gaussian_ew * gaussian_flux_error / gaussian_flux

        # Integrated flux
        # IRAF: flux = sum ((I(i)-C(i)) * (w(i2) - w(i1)) / (i2 - i2)
        flux = np.nansum([(f_spec[i] - continuum[i]) * (w_spec[i + 1] - w_spec[i]) for i in range(len(w_spec)) if
                          (w_spec[i] >= low_limit and w_spec[i] <= high_limit)])
        flux_error = rms_cont * (high_limit - low_limit)
        wave_resolution = (wavelength[-1] - wavelength[0]) / len(wavelength)
        ew = wave_resolution * np.nansum([(1 - f_spec[i] / continuum[i]) for i in range(len(w_spec)) if
                                          (w_spec[i] >= low_limit and w_spec[i] <= high_limit)])
        ew_error = np.abs(ew * flux_error / flux)
        gauss_to_integrated = gaussian_flux / flux * 100.

        # Plotting
        if plot:
            plt.figure(figsize=(10, 4))
            # Plot input spectrum
            plt.plot(np.array(w_spec), np.array(f_spec), "blue", lw=2, alpha=0.7)
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim((line1 + line2) / 2 - 40, (line1 + line2) / 2 + 40)
            plt.ylim(fmin, fmax)

            # Vertical line at guess_centre
            plt.axvline(x=guess_centre1, color='r', linestyle='-', alpha=0.5)
            plt.axvline(x=guess_centre2, color='r', linestyle='-', alpha=0.5)

            # Horizontal line at y = 0
            plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
            # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
            plt.axvspan(guess_centre + highlow, guess_centre + highhigh, facecolor='g', alpha=0.15, zorder=3)
            plt.axvspan(guess_centre - lowlow, guess_centre - lowhigh, facecolor='g', alpha=0.15, zorder=3)
            # Plot linear fit for continuum
            plt.plot(w_spec, continuum, "g--")
            # Plot Gaussian fit
            plt.plot(w_spec, gaussian_fit + continuum, 'r-', alpha=0.8)
            # Vertical line at Gaussian center
            plt.axvline(x=fit[0], color='k', linestyle='-', alpha=0.5)
            plt.axvline(x=fit[3], color='k', linestyle='-', alpha=0.5)
            # Plot Gaussians + cont
            plt.plot(w_spec, gaussian_fit + continuum, 'r-', alpha=0.5, lw=3)
            plt.plot(w_spec, gaussian_1 + continuum, color="navy", linestyle='--', alpha=0.8)
            plt.plot(w_spec, gaussian_2 + continuum, color="#1f77b4", linestyle='--', alpha=0.8)
            plt.plot(w_spec, np.array(f_spec) - (gaussian_fit), 'orange', alpha=0.4, linewidth=5)

            # Vertical lines to emission line
            plt.axvline(x=low_limit, color='k', linestyle=':', alpha=0.5)
            plt.axvline(x=high_limit, color='k', linestyle=':', alpha=0.5)
            plt.title(
                'Double Gaussian Fit')  # Fit: x0=%.2f y0=%.2e sigma=%.2f  flux=%.2e  rms=%.3e' % (fit[0], fit[1], fit[2], gaussian_flux, rms_fit))
            plt.show()
            plt.close()

            # Plot residuals
        #            plt.figure(figsize=(10, 1))
        #            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
        #            plt.ylabel("RMS")
        #            plt.xlim((line1+line2)/2-40,(line1+line2)/2+40)
        #            plt.axvline(x=fit[0], color='k', linestyle='-', alpha=0.5)
        #            plt.axvline(x=fit[3], color='k', linestyle='-', alpha=0.5)
        #            plt.plot(w_spec, residuals, 'k')
        #            plt.minorticks_on()
        #            plt.show()
        #            plt.close()

        # Printing results
        if verbose:
            # print "\n> WARNING !!! CAREFUL WITH THE VALUES PROVIDED BELOW, THIS TASK NEEDS TO BE UPDATED!\n"
            print("\n> Gauss and continuum fitting + integrated flux calculations:\n")
            print("  rms continuum = %.3e erg/cm/s/A " % (rms_cont))
            print("  Gaussian Fit parameters: x0 = ( %.2f +- %.2f )  A " % (fit[0], fit_error[0]))
            print("                           y0 = ( %.3f +- %.3f )  1E-16 erg/cm2/s/A" % (
            fit[1] / 1E-16, fit_error[1] / 1E-16))
            print("                        sigma = ( %.3f +- %.3f )  A" % (fit[2], fit_error[2]))
            print("                      rms fit = %.3e erg/cm2/s/A" % (rms_fit))
            print("  Gaussian Flux = ( %.2f +- %.2f ) 1E-16 erg/s/cm2         (error = %.1f per cent)" % (
            gaussian_flux / 1E-16, gaussian_flux_error / 1E-16, gaussian_flux_error / gaussian_flux * 100))
            print("  FWHM          = ( %.3f +- %.3f ) A    =   ( %.1f +- %.1f ) km/s " % (
            fwhm, fwhm_error, fwhm_vel, fwhm_vel_error))
            print("  Eq. Width     = ( %.1f +- %.1f ) A" % (-gaussian_ew, gaussian_ew_error))
            print("\n  Integrated flux  = ( %.2f +- %.2f ) 1E-16 erg/s/cm2      (error = %.1f per cent) " % (
            flux / 1E-16, flux_error / 1E-16, flux_error / flux * 100))
            print("  Eq. Width        = ( %.1f +- %.1f ) A" % (ew, ew_error))
            print("  Gauss/Integrated = %.2f per cent " % gauss_to_integrated)

        # New 22 Jan 2019: sustract Gaussian fit
        index = 0
        s_s = np.zeros_like(s)
        sustract_this = np.zeros_like(gaussian_fit)
        if sus_line1:
            sustract_this = sustract_this + gaussian_1
        if sus_line2:
            sustract_this = sustract_this + gaussian_2

        for wave in range(len(wavelength)):
            s_s[wave] = s[wave]
            if wavelength[wave] == w_spec[0]:
                s_s[wave] = f_spec[0] - sustract_this[0]
                index = 1
            if wavelength[wave] > w_spec[0] and wavelength[wave] <= w_spec[-1]:
                s_s[wave] = f_spec[index] - sustract_this[index]
                index = index + 1
        if plot_sus:
            plt.figure(figsize=(10, 4))
            plt.plot(wavelength, s, "r")
            plt.plot(wavelength, s_s, "c")
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(lmin, lmax)
            plt.ylim(fmin, fmax)
            plt.show()
            plt.close()

        # This gaussian_flux in 3  is gaussian 1 + gaussian 2, given in 15, 16, respectively
        #                0      1         2                3               4              5      6         7        8        9     10      11   12       13      14         15                16
        resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux,
                     flux_error, ew, ew_error, s_s, fit[3], fit[4], fit[5], gaussian_flux_1, gaussian_flux_2]
        return resultado
    except Exception:
        if verbose: print("  Double Gaussian fit failed!")
        resultado = [0, line1, 0, 0, 0, 0, 0, 0, 0, 0, 0, s, 0, 0, 0, 0,
                     0]  # line was identified at lambda=line but Gaussian fit failed

        # NOTA: PUEDE DEVOLVER EL FLUJO INTEGRADO AUNQUE FALLE EL AJUSTE GAUSSIANO...

        # Plotting
        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(np.array(w_spec), np.array(f_spec), "b", lw=3, alpha=0.5)
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(lmin, lmax)
            plt.ylim(fmin, fmax)

            # Vertical line at guess_centre
            plt.axvline(x=guess_centre, color='r', linestyle='-', alpha=0.5)
            # Horizontal line at y = 0
            plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
            # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
            plt.axvspan(guess_centre + highlow, guess_centre + highhigh, facecolor='g', alpha=0.15, zorder=3)
            plt.axvspan(guess_centre - lowlow, guess_centre - lowhigh, facecolor='g', alpha=0.15, zorder=3)
            # Plot linear fit for continuum
            plt.plot(w_spec, continuum, "g--")
            # Plot Gaussian fit
            #            plt.plot(w_spec, gaussian_fit+continuum, 'r-', alpha=0.8)
            # Vertical line at Gaussian center
            #            plt.axvline(x=fit[0], color='k', linestyle='-', alpha=0.5)
            # Vertical lines to emission line
            plt.axvline(x=low_limit, color='k', linestyle=':', alpha=0.5)
            plt.axvline(x=high_limit, color='k', linestyle=':', alpha=0.5)
            # Plot residuals
            #            plt.plot(w_spec, residuals, 'k')
            plt.title("No Gaussian fit obtained...")
            plt.show()

        return resultado
    # -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def search_peaks(wavelength, flux, smooth_points=20, lmin=0, lmax=0, fmin=0.5, fmax=3.,
                 emission_line_file="lineas_c89_python.dat", brightest_line="Ha", cut=1.2,
                 check_redshift=0.0003, only_id_lines=True, plot=True, verbose=True, fig_size=12):
    """
    Search and identify emission lines in a given spectrum.\n
    For this the routine first fits a rough estimation of the global continuum.\n
    Then it uses "flux"/"continuum" > "cut" to search for the peaks, assuming this
    is satisfied in at least 2 consecutive wavelengths.\n
    Once the peaks are found, the routine identifies the brightest peak with the given "brightest_line",
    that has to be included in the text file "emission_line_file".\n
    After that the routine checks if the rest of identified peaks agree with the emission
    lines listed in text file "emission_line_file".
    If abs(difference in wavelength) > 2.5, we don't consider the line identified.\n
    Finally, it checks if all redshifts are similar, assuming "check_redshift" = 0.0003 by default.

    Result
    ------
    The routine returns FOUR lists:

    peaks: (float) wavelength of the peak of the detected emission lines.
        It is NOT necessarily the central wavelength of the emission lines.
    peaks_name: (string).
        name of the detected emission lines.
    peaks_rest: (float)
        rest wavelength of the detected emission lines
    continuum_limits: (float):
        provides the values for fitting the local continuum
        for each of the detected emission lines, given in the format
        [lowlow, lowhigh, highlow, highhigh]

    Parameters
    ----------
    wavelenght: float
        wavelength.
    flux: float
        flux per wavelenght
    smooth_points: float (default = 20)
        Number of points for a smooth spectrum to get a rough estimation of the global continuum
    lmin, lmax: float
        wavelength range to be analized
    fmin, fmax: float (defaut = 0.5, 2.)
        minimum and maximun values of flux/continuum to be plotted
    emission_line_file: string (default = "lineas_c89_python.dat")
        tex file with a list of emission lines to be found.
        This text file has to have the following format per line:

        rest_wavelength  name   f(lambda) lowlow lowhigh  highlow  highhigh

        E.g.: 6300.30       [OI]  -0.263   15.0  4.0   20.0   40.0

    brightest_line: string (default="Ha")
        expected emission line in the spectrum
    cut: float (default = 1.2)
        minimum value of flux/continuum to check for emission lines
    check_redshift: float (default = 0.0003)
        check if the redshifts derived using the detected emission lines agree with that obtained for
        the brightest emission line (ref.). If abs(z - zred) > check_redshift a warning is shown.
    plot: boolean (default = True)
        Plot a figure with the emission lines identifications.
    verbose: boolean (default = True)
        Print results.
    only_id_lines: boolean (default = True)
        Provide only the list of the identified emission lines

    Example
    -------
    >>> peaks,peaks_name,peaks_rest,continuum_limits=search_peaks(wavelength, spectrum, plot=False)
    """
    # Setup wavelength limits
    if lmin == 0:
        lmin = np.nanmin(wavelength)
    if lmax == 0:
        lmax = np.nanmax(wavelength)

    # Fit a smooth continuum
    # smooth_points = 20      # Points in the interval
    step = np.int(len(wavelength) / smooth_points)  # step
    w_cont_smooth = np.zeros(smooth_points)
    f_cont_smooth = np.zeros(smooth_points)

    for j in range(smooth_points):
        w_cont_smooth[j] = np.nanmedian(
            [wavelength[i] for i in range(len(wavelength)) if (i > step * j and i < step * (j + 1))])
        f_cont_smooth[j] = np.nanmedian([flux[i] for i in range(len(wavelength)) if
                                         (i > step * j and i < step * (j + 1))])  # / np.nanmedian(spectrum)
        # print j,w_cont_smooth[j], f_cont_smooth[j]

    interpolated_continuum_smooth = interpolate.splrep(w_cont_smooth, f_cont_smooth, s=0)
    interpolated_continuum = interpolate.splev(wavelength, interpolated_continuum_smooth, der=0)

    funcion = flux / interpolated_continuum

    # Searching for peaks using cut = 1.2 by default
    peaks = []
    index_low = 0
    for i in range(len(wavelength)):
        if funcion[i] > cut and funcion[i - 1] < cut:
            index_low = i
        if funcion[i] < cut and funcion[i - 1] > cut:
            index_high = i
            if index_high != 0:
                pfun = np.nanmax([funcion[j] for j in range(len(wavelength)) if (j > index_low and j < index_high + 1)])
                peak = wavelength[funcion.tolist().index(pfun)]
                if (index_high - index_low) > 1:
                    peaks.append(peak)

    # Identify lines
    # Read file with data of emission lines:
    # 6300.30 [OI] -0.263    15   5    5    15
    # el_center el_name el_fnl lowlow lowhigh highlow highigh
    # Only el_center and el_name are needed
    el_center, el_name, el_fnl, el_lowlow, el_lowhigh, el_highlow, el_highhigh = read_table(emission_line_file,
                                                                                            ["f", "s", "f", "f", "f",
                                                                                             "f", "f"])
    # for i in range(len(el_name)):
    #    print " %8.2f  %9s  %6.3f   %4.1f %4.1f   %4.1f   %4.1f" % (el_center[i],el_name[i],el_fnl[i],el_lowlow[i], el_lowhigh[i], el_highlow[i], el_highhigh[i])
    # el_center,el_name = read_table("lineas_c89_python.dat", ["f", "s"] )

    # In case this is needed in the future...
    #    el_center = [6300.30, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66]
    #    el_fnl    = [-0.263, -0.264, -0.271, -0.296,    -0.298, -0.300, -0.313, -0.318, -0.320,    -0.364,   -0.374, -0.398, -0.400 ]
    #    el_name   = ["[OI]", "[SIII]", "[OI]", "[NII]", "Ha", "[NII]",  "HeI", "[SII]", "[SII]",   "HeI",  "[ArIII]", "[OII]", "[OII]" ]

    # Search for the brightest line in given spectrum ("Ha" by default)
    peaks_flux = np.zeros(len(peaks))
    for i in range(len(peaks)):
        peaks_flux[i] = flux[wavelength.tolist().index(peaks[i])]
    Ha_w_obs = peaks[peaks_flux.tolist().index(np.nanmax(peaks_flux))]

    # Estimate redshift of the brightest line ( Halpha line by default)
    Ha_index_list = el_name.tolist().index(brightest_line)
    Ha_w_rest = el_center[Ha_index_list]
    Ha_redshift = (Ha_w_obs - Ha_w_rest) / Ha_w_rest
    if verbose: print("\n> Detected %i emission lines using %8s at %8.2f A as brightest line!!\n" % (
        len(peaks), brightest_line, Ha_w_rest))
    #    if verbose: print "  Using %8s at %8.2f A as brightest line  --> Found in %8.2f with a redshift %.6f " % (brightest_line, Ha_w_rest, Ha_w_obs, Ha_redshift)

    # Identify lines using brightest line (Halpha by default) as reference.
    # If abs(wavelength) > 2.5 we don't consider it identified.
    peaks_name = [None] * len(peaks)
    peaks_rest = np.zeros(len(peaks))
    peaks_redshift = np.zeros(len(peaks))
    peaks_lowlow = np.zeros(len(peaks))
    peaks_lowhigh = np.zeros(len(peaks))
    peaks_highlow = np.zeros(len(peaks))
    peaks_highhigh = np.zeros(len(peaks))

    for i in range(len(peaks)):
        minimo_w = np.abs(peaks[i] / (1 + Ha_redshift) - el_center)
        if np.nanmin(minimo_w) < 2.5:
            indice = minimo_w.tolist().index(np.nanmin(minimo_w))
            peaks_name[i] = el_name[indice]
            peaks_rest[i] = el_center[indice]
            peaks_redshift[i] = (peaks[i] - el_center[indice]) / el_center[indice]
            peaks_lowlow[i] = el_lowlow[indice]
            peaks_lowhigh[i] = el_lowhigh[indice]
            peaks_highlow[i] = el_highlow[indice]
            peaks_highhigh[i] = el_highhigh[indice]
            if verbose: print("%9s %8.2f found in %8.2f at z=%.6f   |z-zref| = %.6f" % (
            peaks_name[i], peaks_rest[i], peaks[i], peaks_redshift[i], np.abs(peaks_redshift[i] - Ha_redshift)))
            # print peaks_lowlow[i],peaks_lowhigh[i],peaks_highlow[i],peaks_highhigh[i]
    # Check if all redshifts are similar, assuming check_redshift = 0.0003 by default
    # If OK, add id_peaks[i]=1, if not, id_peaks[i]=0
    id_peaks = []
    for i in range(len(peaks_redshift)):
        if np.abs(peaks_redshift[i] - Ha_redshift) > check_redshift:
            if verbose: print("  WARNING!!! Line %8s in w = %.2f has redshift z=%.6f, different than zref=%.6f" % (
            peaks_name[i], peaks[i], peaks_redshift[i], Ha_redshift))
            id_peaks.append(0)
        else:
            id_peaks.append(1)

    if plot:
        plt.figure(figsize=(fig_size, fig_size / 2.5))
        plt.plot(wavelength, funcion, "r", lw=1, alpha=0.5)
        plt.minorticks_on()
        plt.xlabel("Wavelength [$\mathrm{\AA}$]")
        plt.ylabel("Flux / continuum")

        plt.xlim(lmin, lmax)
        plt.ylim(fmin, fmax)
        plt.axhline(y=cut, color='k', linestyle=':', alpha=0.5)
        for i in range(len(peaks)):
            plt.axvline(x=peaks[i], color='k', linestyle=':', alpha=0.5)
            label = peaks_name[i]
            plt.text(peaks[i], 1.8, label)
        plt.show()

    continuum_limits = [peaks_lowlow, peaks_lowhigh, peaks_highlow, peaks_highhigh]

    if only_id_lines:
        peaks_r = []
        peaks_name_r = []
        peaks_rest_r = []
        peaks_lowlow_r = []
        peaks_lowhigh_r = []
        peaks_highlow_r = []
        peaks_highhigh_r = []

        for i in range(len(peaks)):
            if id_peaks[i] == 1:
                peaks_r.append(peaks[i])
                peaks_name_r.append(peaks_name[i])
                peaks_rest_r.append(peaks_rest[i])
                peaks_lowlow_r.append(peaks_lowlow[i])
                peaks_lowhigh_r.append(peaks_lowhigh[i])
                peaks_highlow_r.append(peaks_highlow[i])
                peaks_highhigh_r.append(peaks_highhigh[i])
        continuum_limits_r = [peaks_lowlow_r, peaks_lowhigh_r, peaks_highlow_r, peaks_highhigh_r]

        return peaks_r, peaks_name_r, peaks_rest_r, continuum_limits_r
    else:
        return peaks, peaks_name, peaks_rest, continuum_limits
    # -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fit_smooth_spectrum(wl, x, edgelow=20, edgehigh=20, order=9, kernel=11, verbose=True,
                        plot=True, hlines=[1.], ptitle="", fcal=False):
    """
    Apply f1,f2 = fit_smooth_spectrum(wl,spectrum) and returns:

    f1 is the smoothed spectrum, with edges 'fixed'
    f2 is the fit to the smooth spectrum
    """

    if verbose:
        print(
            '\n> Fitting an order {} polynomium to a spectrum smoothed with medfilt window of {}'.format(order, kernel))
        print("  trimming the edges [0:{}] and [{}:{}] ...".format(edgelow, len(wl) - edgehigh, len(wl)))
        # fit, trimming edges
    index = np.arange(len(x))
    valid_ind = np.where((index >= edgelow) & (index <= len(wl) - edgehigh) & (~np.isnan(x)))[0]
    valid_wl = wl[valid_ind]
    valid_x = x[valid_ind]
    wlm = signal.medfilt(valid_wl, kernel)
    wx = signal.medfilt(valid_x, kernel)

    # iteratively clip and refit
    maxit = 10
    niter = 0
    stop = 0
    fit_len = 100  # -100
    resid = 0
    while stop < 1:
        # print '  Trying iteration ', niter,"..."
        fit_len_init = copy.deepcopy(fit_len)
        if niter == 0:
            fit_index = np.where(wx == wx)
            fit_len = len(fit_index)
            sigma_resid = 0.0
            # print fit_index, fit_len
        if niter > 0:
            sigma_resid = MAD(resid)
            fit_index = np.where(np.abs(resid) < 4 * sigma_resid)[0]
            fit_len = len(fit_index)
        try:
            # print " Fitting between ", wlm[fit_index][0],wlm[fit_index][-1]
            p = np.polyfit(wlm[fit_index], wx[fit_index], order)  # It was 2
            pp = np.poly1d(p)
            fx = pp(wl)
            fxm = pp(wlm)
            resid = wx - fxm
            # print niter,wl,fx, fxm
            # print "  Iteration {:2} results in RA: sigma_residual = {:.6f}, fit_len_init = {:5}  fit_len ={:5}".format(niter,sigma_resid,fit_len_init,fit_len)
        except Exception:
            if verbose: print('  Skipping iteration ', niter)
        if (niter >= maxit) or (fit_len_init == fit_len):
            if verbose:
                if niter >= maxit: print("  Max iterations, {:2}, reached!".format(niter))
                if fit_len_init == fit_len: print("  All interval fitted in iteration {} ! ".format(niter))
            stop = 2
        niter = niter + 1

    # Smoothed spectrum, adding the edges
    f_ = signal.medfilt(valid_x, kernel)
    f = np.zeros_like(x)
    f[valid_ind] = f_
    half_kernel = np.int(kernel / 2)
    if half_kernel > edgelow:
        f[np.where(index < half_kernel)] = f_[half_kernel - edgelow]
    else:
        f[np.where(index < edgelow)] = f_[0]
    if half_kernel > edgehigh:
        f[np.where(index > len(wl) - half_kernel)] = f_[-1 - half_kernel + edgehigh]
    else:
        f[np.where(index < edgehigh)] = f_[-1]

    if plot:
        ymin = np.nanpercentile(x[edgelow:len(x) - edgehigh], 1.2)
        ymax = np.nanpercentile(x[edgelow:len(x) - edgehigh], 99)
        rango = (ymax - ymin)
        ymin = ymin - rango / 10.
        ymax = ymax + rango / 10.
        if ptitle == "": ptitle = "Order " + np.str(
            order) + " polynomium fitted to a spectrum smoothed with a " + np.str(kernel) + " kernel window"
        plot_plot(wl, [x, f, fx], ymin=ymin, ymax=ymax, color=["red", "green", "blue"], alpha=[0.2, 0.5, 0.5],
                  label=["spectrum", "smoothed", "fit"], ptitle=ptitle, fcal=fcal,
                  vlines=[wl[edgelow], wl[-1 - edgehigh]], hlines=hlines)

    return f, fx


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def smooth_spectrum(wlm, s, wave_min=0, wave_max=0, step=50, exclude_wlm=[[0, 0]], order=7,
                    weight_fit_median=0.5, plot=False, verbose=False, fig_size=12):
    """
    THIS IS NOT EXACTLY THE SAME THING THAT applying signal.medfilter()

    This needs to be checked, updated, and combine with task fit_smooth_spectrum.
    The task gets the median value in steps of "step", gets an interpolated spectrum,
    and fits a 7-order polynomy.

    It returns fit_median + fit_median_interpolated (each multiplied by their weights).

    Tasks that use this: correcting_negative_sky, get_telluric_correction
    """

    if verbose: print("\n> Computing smooth spectrum...")

    if wave_min == 0: wave_min = wlm[0]
    if wave_max == 0: wave_max = wlm[-1]

    running_wave = []
    running_step_median = []
    cuts = np.int((wave_max - wave_min) / step)

    exclude = 0
    corte_index = -1
    for corte in range(cuts + 1):
        next_wave = wave_min + step * corte
        if next_wave < wave_max:
            if next_wave > exclude_wlm[exclude][0] and next_wave < exclude_wlm[exclude][1]:
                if verbose: print("  Skipping ", next_wave, " as it is in the exclusion range [",
                                  exclude_wlm[exclude][0], ",", exclude_wlm[exclude][1], "]")

            else:
                corte_index = corte_index + 1
                running_wave.append(next_wave)
                region = np.where(
                    (wlm > running_wave[corte_index] - step / 2) & (wlm < running_wave[corte_index] + step / 2))
                running_step_median.append(np.nanmedian(s[region]))
                if next_wave > exclude_wlm[exclude][1]:
                    exclude = exclude + 1
                    # if verbose and exclude_wlm[0] != [0,0] : print "--- End exclusion range ",exclude
                    if exclude == len(exclude_wlm):  exclude = len(exclude_wlm) - 1

    running_wave.append(wave_max)
    region = np.where((wlm > wave_max - step) & (wlm < wave_max + 0.1))
    running_step_median.append(np.nanmedian(s[region]))

    # Check not nan
    _running_wave_ = []
    _running_step_median_ = []
    for i in range(len(running_wave)):
        if np.isnan(running_step_median[i]):
            if verbose: print("  There is a nan in ", running_wave[i])
        else:
            _running_wave_.append(running_wave[i])
            _running_step_median_.append(running_step_median[i])

    fit = np.polyfit(_running_wave_, _running_step_median_, order)
    pfit = np.poly1d(fit)
    fit_median = pfit(wlm)

    interpolated_continuum_smooth = interpolate.splrep(_running_wave_, _running_step_median_, s=0.02)
    fit_median_interpolated = interpolate.splev(wlm, interpolated_continuum_smooth, der=0)

    if plot:
        plt.figure(figsize=(fig_size, fig_size / 2.5))
        plt.plot(wlm, s, alpha=0.5)
        plt.plot(running_wave, running_step_median, "+", ms=15, mew=3)
        plt.plot(wlm, fit_median, label="fit median")
        plt.plot(wlm, fit_median_interpolated, label="fit median_interp")
        plt.plot(wlm, weight_fit_median * fit_median + (1 - weight_fit_median) * fit_median_interpolated,
                 label="weighted")
        # extra_display = (np.nanmax(fit_median)-np.nanmin(fit_median)) / 10
        # plt.ylim(np.nanmin(fit_median)-extra_display, np.nanmax(fit_median)+extra_display)
        ymin = np.nanpercentile(s, 1)
        ymax = np.nanpercentile(s, 99)
        rango = (ymax - ymin)
        ymin = ymin - rango / 10.
        ymax = ymax + rango / 10.
        plt.ylim(ymin, ymax)
        plt.xlim(wlm[0] - 10, wlm[-1] + 10)
        plt.minorticks_on()
        plt.legend(frameon=False, loc=1, ncol=1)

        plt.axvline(x=wave_min, color='k', linestyle='--')
        plt.axvline(x=wave_max, color='k', linestyle='--')

        plt.xlabel("Wavelength [$\mathrm{\AA}$]")

        if exclude_wlm[0][0] != 0:
            for i in range(len(exclude_wlm)):
                plt.axvspan(exclude_wlm[i][0], exclude_wlm[i][1], color='r', alpha=0.1)
        plt.show()
        plt.close()
        print('  Weights for getting smooth spectrum:  fit_median =', weight_fit_median,
              '    fit_median_interpolated =', (1 - weight_fit_median))

    return weight_fit_median * fit_median + (
                1 - weight_fit_median) * fit_median_interpolated  # (fit_median+fit_median_interpolated)/2      # Decide if fit_median or fit_median_interpolated


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
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
def median_filter(intensity_corrected, n_spectra, n_wave, win_sky=151):
    """
    Matt's code to get a median filter of all fibres in a RSS
    This is useful when having 2D sky
    """

    medfilt_sky = np.zeros((n_spectra, n_wave))
    for wave in range(n_wave):
        medfilt_sky[:, wave] = sig.medfilt(intensity_corrected[:, wave], kernel_size=win_sky)

    # replace crappy edge fibres with 0.5*win'th medsky
    for fibre_sky in range(n_spectra):
        if fibre_sky < np.rint(0.5 * win_sky):
            j = int(np.rint(0.5 * win_sky))
            medfilt_sky[fibre_sky,] = copy.deepcopy(medfilt_sky[j,])
        if fibre_sky > n_spectra - np.rint(0.5 * win_sky):
            j = int(np.rint(n_spectra - np.rint(0.5 * win_sky)))
            medfilt_sky[fibre_sky,] = copy.deepcopy(medfilt_sky[j,])
    return medfilt_sky


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
def sky_spectrum_from_fibres(rss, list_spectra, win_sky=0, wave_to_fit=300, fit_order=2, include_history=True,
                             xmin="", xmax="", ymin="", ymax="", verbose=True, plot=True):
    if verbose:
        print("\n> Obtaining 1D sky spectrum using the rss file and fibre list = ")
        print("  ", list_spectra)

    _rss_ = copy.deepcopy(rss)
    w = _rss_.wavelength

    if win_sky > 0:
        if verbose: print("  after applying a median filter with kernel ", win_sky, "...")
        _rss_.intensity_corrected = median_filter(_rss_.intensity_corrected, _rss_.n_spectra, _rss_.n_wave,
                                                  win_sky=win_sky)
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
# -----------------------------------------------------------------------------
def sky_spectrum_from_fibres_using_file(rss_file, fibre_list=[], win_sky=151, n_sky=0,
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

    _test_rss_ = KOALA_RSS(rss_file, apply_throughput=apply_throughput, skyflat=skyflat,
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


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def ds9_offsets(x1, y1, x2, y2, pixel_size_arc=0.6):
    delta_x = x2 - x1
    delta_y = y2 - y1

    print("\n> Offsets in pixels : ", delta_x, delta_y)
    print("  Offsets in arcsec : ", pixel_size_arc * delta_x, pixel_size_arc * delta_y)
    offset_RA = np.abs(pixel_size_arc * delta_x)
    if delta_x < 0:
        direction_RA = "W"
    else:
        direction_RA = "E"
    offset_DEC = np.abs(pixel_size_arc * delta_y)
    if delta_y < 0:
        direction_DEC = "N"
    else:
        direction_DEC = "S"
    print("  Assuming N up and E left, the telescope did an offset of ----> {:5.2f} {:1} {:5.2f} {:1}".format(offset_RA,
                                                                                                              direction_RA,
                                                                                                              offset_DEC,
                                                                                                              direction_DEC))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def offset_positions(ra1h, ra1m, ra1s, dec1d, dec1m, dec1s, ra2h, ra2m, ra2s, dec2d, dec2m, dec2s, decimals=2):
    """
    CHECK THE GOOD ONE in offset_positions.py !!!
    """

    ra1 = ra1h + ra1m / 60. + ra1s / 3600.
    ra2 = ra2h + ra2m / 60. + ra2s / 3600.

    if dec1d < 0:
        dec1 = dec1d - dec1m / 60. - dec1s / 3600.
    else:
        dec1 = dec1d + dec1m / 60. + dec1s / 3600
    if dec2d < 0:
        dec2 = dec2d - dec2m / 60. - dec2s / 3600.
    else:
        dec2 = dec2d + dec2m / 60. + dec2s / 3600.

    avdec = (dec1 + dec2) / 2

    deltadec = round(3600. * (dec2 - dec1), decimals)
    deltara = round(15 * 3600. * (ra2 - ra1) * (np.cos(np.radians(avdec))), decimals)

    tdeltadec = np.fabs(deltadec)
    tdeltara = np.fabs(deltara)

    if deltadec < 0:
        t_sign_deltadec = "South"
        t_sign_deltadec_invert = "North"

    else:
        t_sign_deltadec = "North"
        t_sign_deltadec_invert = "South"

    if deltara < 0:
        t_sign_deltara = "West"
        t_sign_deltara_invert = "East"

    else:
        t_sign_deltara = "East"
        t_sign_deltara_invert = "West"

    print("\n> POS1: RA = {:3}h {:2}min {:2.4f}sec, DEC = {:3}d {:2}m {:2.4f}s".format(ra1h, ra1m, ra1s, dec1d, dec1m,
                                                                                       dec1s))
    print("  POS2: RA = {:3}h {:2}min {:2.4f}sec, DEC = {:3}d {:2}m {:2.4f}s".format(ra2h, ra2m, ra2s, dec2d, dec2m,
                                                                                     dec2s))

    print("\n> Offset 1 -> 2 : ", tdeltara, t_sign_deltara, "     ", tdeltadec, t_sign_deltadec)
    print("  Offset 2 -> 1 : ", tdeltara, t_sign_deltara_invert, "     ", tdeltadec, t_sign_deltadec_invert)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def basic_statistics(y, x="", xmin="", xmax="", return_data=False, verbose=True):
    """
    Provides basic statistics: min, median, max, std, rms, and snr"
    """
    if len(x) == 0:
        y_ = y
    else:
        y_ = []
        if xmin == "": xmin = x[0]
        if xmax == "": xmax = x[-1]
        y_.extend((y[j]) for j in range(len(x)) if (x[j] > xmin and x[j] < xmax))

    median_value = np.nanmedian(y_)
    min_value = np.nanmin(y_)
    max_value = np.nanmax(y_)

    n_ = len(y_)
    # mean_ = np.sum(y_) / n_
    mean_ = np.nanmean(y_)
    # var_ = np.sum((item - mean_)**2 for item in y_) / (n_ - 1)
    var_ = np.nanvar(y_)

    std = np.sqrt(var_)
    ave_ = np.nanmean(y_)
    disp_ = max_value - min_value

    rms_v = ((y_ - mean_) / disp_) ** 2
    rms = disp_ * np.sqrt(np.nansum(rms_v) / (n_ - 1))
    snr = ave_ / rms

    if verbose:
        print("  min_value  = {}, median value = {}, max_value = {}".format(min_value, median_value, max_value))
        print("  standard deviation = {}, rms = {}, snr = {}".format(std, rms, snr))  # TIGRE

    if return_data: return min_value, median_value, max_value, std, rms, snr


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_plot(x, y, xmin="", xmax="", ymin="", ymax="", percentile_min=2, percentile_max=98,
              ptitle="Pretty plot", xlabel="Wavelength [$\mathrm{\AA}$]", ylabel="", fcal="",
              psym="", color="blue", alpha="", linewidth=1, linestyle="-", markersize=10,
              vlines=[], hlines=[], chlines=[], axvspan=[[0, 0]], hwidth=1, vwidth=1,
              frameon=False, loc=0, ncol=5, label="", text=[],
              title_fontsize=12, label_axes_fontsize=10, axes_fontsize=10, tick_size=[5, 1, 2, 1], axes_thickness=0,
              save_file="", path="", fig_size=12, warnings=True, show=True, statistics=""):
    """
    Plot this plot! An easy way of plotting plots in Python.

    Parameters
     ----------
     x,y : floats (default = none)
         Positional arguments for plotting the data
     xmin, xmax, ymin, ymax : floats (default = none)
         Plotting limits
     percentile_min : integer (default = 2)
         Lower bound percentile for filtering outliers in data
     percentile_max : integer (default = 98)
         Higher bound percentile for filtering outliers in data
     ptitle : string (default = "Pretty plot")
         Title of the plot
     xlabel : string (default = "Wavelength [$\mathrm{\AA}$]")
         Label for x axis of the plot
     ylabel : string (default = "Flux [counts]")
         Label for y axis of the plot
     fcal : boolean (default = none)
         If True that means that flux is calibrated to CGS units, changes the y axis label to match
     psym : string or list of strings (default = "")
         Symbol marker, is given. If "" then it is a line.
     color : string (default = "blue")
         Color pallete of the plot. Default order is "red","blue","green","k","orange", "purple", "cyan", "lime"
     alpha : float or list of floats (default = 1)
         Opacity of the graph, 1 is opaque, 0 is transparent
     linewidth: float or list of floats (default=1)
         Linewidth of each line or marker
     linestyle:  string or list of strings (default="-")
         Style of the line
     markersize: float or list of floats (default = 10)
         Size of the marker
     vlines : list of floats (default = [])
         Draws vertical lines at the specified x positions
     hlines : list of floats (default = [])
         Draws horizontal lines at the specified y positions
     chlines : list of strings (default = [])
         Color of the horizontal lines
     axvspan : list of floats (default = [[0,0]])
         Shades the region between the x positions specified
     hwidth: float (default =1)
         thickness of horizontal lines
     vwidth: float (default =1)
         thickness of vertical lines
     frameon : boolean (default = False)
         Display the frame of the legend section
     loc : string or pair of floats or integer (default = 0)
         Location of the legend in pixels. See matplotlib.pyplot.legend() documentation
     ncol : integer (default = 5)
         Number of columns of the legend
     label : string or list of strings (default = "")
         Specify labels for the graphs
     title_fontsize: float (default=12)
         Size of the font of the title
     label_axes_fontsize: float (default=10)
         Size of the font of the label of the axes (e.g. Wavelength or Flux)
     axes_fontsize: float (default=10)
         Size of the font of the axes (e.g. 5000, 6000, ....)
     tick_size: list of 4 floats (default=[5,1,2,1])
         [length_major_tick_axes, thickness_major_tick_axes, length_minor_tick_axes, thickness_minor_tick_axes]
         For defining the length and the thickness of both the major and minor ticks
     axes_thickness: float (default=0)
         Thickness of the axes
     save_file : string (default = none)
         Specify path and filename to save the plot
     path: string  (default = "")
         path of the file to be saved
     fig_size : float (default = 12)
         Size of the figure
     warnings : boolean (default = True)
         Print the warnings in the console if something works incorrectly or might require attention
     show : boolean (default = True)
         Show the plot
     statistics : boolean (default = False)
         Print statistics of the data in the console

    """

    if fig_size == "big":
        fig_size = 20
        label_axes_fontsize = 20
        axes_fontsize = 15
        title_fontsize = 22
        tick_size = [10, 1, 5, 1]
        axes_thickness = 3
        hwidth = 2
        vwidth = 2

    if fig_size in ["very_big", "verybig", "vbig"]:
        fig_size = 35
        label_axes_fontsize = 30
        axes_fontsize = 25
        title_fontsize = 28
        tick_size = [15, 2, 8, 2]
        axes_thickness = 3
        hwidth = 4
        vwidth = 4

    if fig_size != 0: plt.figure(figsize=(fig_size, fig_size / 2.5))

    if np.isscalar(x[0]):
        xx = []
        for i in range(len(y)):
            xx.append(x)
    else:
        xx = x

    if xmin == "": xmin = np.nanmin(xx[0])
    if xmax == "": xmax = np.nanmax(xx[0])

    alpha_ = alpha
    psym_ = psym
    label_ = label
    linewidth_ = linewidth
    markersize_ = markersize
    linestyle_ = linestyle

    n_plots = len(y)

    if np.isscalar(y[0]) == False:
        if np.isscalar(alpha):
            if alpha_ == "":
                alpha = [0.5] * n_plots
            else:
                alpha = [alpha_] * n_plots
        if np.isscalar(psym): psym = [psym_] * n_plots
        if np.isscalar(label): label = [label_] * n_plots
        if np.isscalar(linewidth): linewidth = [linewidth_] * n_plots
        if np.isscalar(markersize): markersize = [markersize_] * n_plots
        if np.isscalar(linestyle): linestyle = [linestyle_] * n_plots
        if color == "blue": color = ["red", "blue", "green", "k", "orange", "purple", "cyan", "lime"]
        if ymax == "": y_max_list = []
        if ymin == "": y_min_list = []

        if fcal == "":
            if np.nanmedian(np.abs(y[0])) < 1E-10:
                fcal = True
                if np.nanmedian(y[0]) < 1E-20 and np.var(y[0]) > 0: fcal = False
        for i in range(len(y)):
            if psym[i] == "":
                plt.plot(xx[i], y[i], color=color[i], alpha=alpha[i], label=label[i], linewidth=linewidth[i],
                         linestyle=linestyle[i])
            else:
                plt.plot(xx[i], y[i], psym[i], color=color[i], alpha=alpha[i], label=label[i], mew=linewidth[i],
                         markersize=markersize[i])
            if ymax == "":
                y_max_ = []
                y_max_.extend((y[i][j]) for j in range(len(xx[i])) if (xx[i][j] > xmin and xx[i][j] < xmax))
                y_max_list.append(np.nanpercentile(y_max_, percentile_max))
            if ymin == "":
                y_min_ = []
                y_min_.extend((y[i][j]) for j in range(len(xx[i])) if (xx[i][j] > xmin and xx[i][j] < xmax))
                y_min_list.append(np.nanpercentile(y_min_, percentile_min))
        if ymax == "":
            ymax = np.nanmax(y_max_list)
        if ymin == "":
            ymin = np.nanmin(y_min_list)
    else:
        if alpha == "": alpha = 1
        if statistics == "": statistics = True
        if fcal == "":
            if np.nanmedian(np.abs(y)) < 1E-10:
                fcal = True
                if np.nanmedian(np.abs(y)) < 1E-20 and np.nanvar(np.abs(y)) > 0: fcal = False
        if psym == "":
            plt.plot(x, y, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        else:
            plt.plot(x, y, psym, color=color, alpha=alpha, mew=linewidth, markersize=markersize)
        if ymin == "":
            y_min_ = []
            y_min_.extend((y[j]) for j in range(len(x)) if (x[j] > xmin and x[j] < xmax))
            ymin = np.nanpercentile(y_min_, percentile_min)
        if ymax == "":
            y_max_ = []
            y_max_.extend((y[j]) for j in range(len(x)) if (x[j] > xmin and x[j] < xmax))
            ymax = np.nanpercentile(y_max_, percentile_max)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    try:
        plt.title(ptitle, fontsize=title_fontsize)
    except Exception:
        if warnings: print("  WARNING: Something failed when including the title of the plot")

    plt.minorticks_on()
    plt.xlabel(xlabel, fontsize=label_axes_fontsize)
    # plt.xticks(rotation=90)
    plt.tick_params('both', length=tick_size[0], width=tick_size[1], which='major')
    plt.tick_params('both', length=tick_size[2], width=tick_size[3], which='minor')
    plt.tick_params(labelsize=axes_fontsize)
    plt.axhline(y=ymin, linewidth=axes_thickness,
                color="k")  # These 4 are for making the axes thicker, it works but it is not ideal...
    plt.axvline(x=xmin, linewidth=axes_thickness, color="k")
    plt.axhline(y=ymax, linewidth=axes_thickness, color="k")
    plt.axvline(x=xmax, linewidth=axes_thickness, color="k")

    if ylabel == "":
        if fcal:
            ylabel = "Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]"
        else:
            ylabel = "Flux [counts]"

    plt.ylabel(ylabel, fontsize=label_axes_fontsize)

    if len(chlines) != len(hlines):
        for i in range(len(hlines) - len(chlines)):
            chlines.append("k")

    for i in range(len(hlines)):
        if chlines[i] != "k":
            hlinestyle = "-"
            halpha = 0.8
        else:
            hlinestyle = "--"
            halpha = 0.3
        plt.axhline(y=hlines[i], color=chlines[i], linestyle=hlinestyle, alpha=halpha, linewidth=hwidth)
    for i in range(len(vlines)):
        plt.axvline(x=vlines[i], color="k", linestyle="--", alpha=0.3, linewidth=vwidth)

    if label_ != "":
        plt.legend(frameon=frameon, loc=loc, ncol=ncol)

    if axvspan[0][0] != 0:
        for i in range(len(axvspan)):
            plt.axvspan(axvspan[i][0], axvspan[i][1], facecolor='orange', alpha=0.15, zorder=3)

    if len(text) > 0:
        for i in range(len(text)):
            plt.text(text[i][0], text[i][1], text[i][2], size=axes_fontsize)

    if save_file == "":
        if show:
            plt.show()
            plt.close()
    else:
        if path != "": save_file = full_path(save_file, path)
        plt.savefig(save_file)
        plt.close()
        print("  Figure saved in file", save_file)

    if statistics == "": statistics = False
    if statistics:
        if np.isscalar(y[0]):
            basic_statistics(y, x=x, xmin=xmin, xmax=xmax)
        else:
            for i in range(len(y)):
                basic_statistics(y[i], x=xx[i], xmin=xmin, xmax=xmax)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Definition introduced by Matt - NOT USED ANYMORE
# def MAD(x):
#     MAD=np.nanmedian(np.abs(x-np.nanmedian(x)))
#     return MAD/0.6745
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def SIN(x):
    SIN = np.sin(x * np.pi / 180)
    return SIN


def COS(x):
    COS = np.cos(x * np.pi / 180)
    return COS


def TAN(x):
    TAN = np.tan(x * np.pi / 180)
    return TAN


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def rebin_spec(wave, specin, wavnew):
    # spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    spec = SourceSpectrum(Empirical1D, points=wave, lookup_table=specin, keep_neg=True)
    f = np.ones(len(wave))
    # filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
    filt = SpectralElement(Empirical1D, points=wave, lookup_table=f, keep_neg=True)  # LOKI
    obs = observation.Observation(spec, filt, binset=wavnew, force='taper')
    return obs.binflux.value


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def rebin_spec_shift(wave, specin, shift):
    wavnew = wave + shift
    rebined = rebin_spec(wave, specin, wavnew)
    return rebined

    ##spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    # spec = SourceSpectrum(Empirical1D, points=wave, lookup_table=specin, keep_neg=True)
    # f = np.ones(len(wave))
    # filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
    # filt=SpectralElement(Empirical1D, points=wave, lookup_table=f, keep_neg=True)
    # obs = observation.Observation(spec, filt, binset=wavnew, force='taper')
    # obs = observation.Observation(spec, filt, binset=wavnew, force='taper')
    # return obs.binflux.value


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def compare_fix_2dfdr_wavelengths(rss1, rss2):
    print("\n> Comparing small fixing of the 2dFdr wavelengths between two rss...")

    xfibre = list(range(0, rss1.n_spectra))
    rss1.wavelength_parameters[0]

    a0x, a1x, a2x = rss1.wavelength_parameters[0], rss1.wavelength_parameters[1], rss1.wavelength_parameters[2]
    aa0x, aa1x, aa2x = rss2.wavelength_parameters[0], rss2.wavelength_parameters[1], rss2.wavelength_parameters[2]

    fx = a0x + a1x * np.array(xfibre) + a2x * np.array(xfibre) ** 2
    fx2 = aa0x + aa1x * np.array(xfibre) + aa2x * np.array(xfibre) ** 2
    dif = fx - fx2

    plot_plot(xfibre, dif, ptitle="Fit 1 - Fit 2", xmin=-20, xmax=1000, xlabel="Fibre", ylabel="Dif")

    resolution = rss1.wavelength[1] - rss1.wavelength[0]
    error = np.nanmedian(dif) / resolution * 100.
    print("\n> The median rms is {:8.6f} A,  resolution = {:5.2f} A,  error = {:5.3} %".format(np.nanmedian(dif),
                                                                                               resolution, error))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def save_nresponse(nflat, filename, mask=[[-1]], no_nans=True, flappyflat=False):
    """
    For masks nflat has to be a rss used to create the mask.
    no_nans = True for mask having only 1s and 0s (otherwise 1s and nans)
    """

    if mask[0][0] != -1:
        fits_image_hdu = fits.PrimaryHDU(mask)
    else:
        fits_image_hdu = fits.PrimaryHDU(nflat.nresponse)

    fits_image_hdu.header["ORIGIN"] = 'AAO'  # / Originating Institution
    fits_image_hdu.header["TELESCOP"] = 'Anglo-Australian Telescope'  # / Telescope Name
    fits_image_hdu.header["ALT_OBS"] = 1164  # / Altitude of observatory in metres
    fits_image_hdu.header["LAT_OBS"] = -31.27704  # / Observatory latitude in degrees
    fits_image_hdu.header["LONG_OBS"] = 149.0661  # / Observatory longitude in degrees

    fits_image_hdu.header["INSTRUME"] = "AAOMEGA-KOALA"  # / Instrument in use
    fits_image_hdu.header["GRATID"] = nflat.grating  # / Disperser ID
    if nflat.grating in red_gratings: SPECTID = "RD"
    if nflat.grating in blue_gratings: SPECTID = "BL"
    fits_image_hdu.header["SPECTID"] = SPECTID  # / Spectrograph ID
    fits_image_hdu.header["DICHROIC"] = 'X5700'  # / Dichroic name   ---> CHANGE if using X6700!!

    fits_image_hdu.header['OBJECT'] = "Normalized skyflat response"
    # fits_image_hdu.header["TOTALEXP"] = combined_cube.total_exptime

    fits_image_hdu.header['NAXIS'] = 2  # / number of array dimensions
    fits_image_hdu.header['NAXIS1'] = nflat.intensity.shape[0]  ##### CHECK !!!!!!!
    fits_image_hdu.header['NAXIS2'] = nflat.intensity.shape[1]

    # WCS
    fits_image_hdu.header["RADECSYS"] = 'FK5'  # / FK5 reference system
    fits_image_hdu.header["EQUINOX"] = 2000  # / [yr] Equinox of equatorial coordinates
    fits_image_hdu.header["WCSAXES"] = 2  # / Number of coordinate axes
    fits_image_hdu.header["CRVAL2"] = 5.000000000000E-01  # / Co-ordinate value of axis 2
    fits_image_hdu.header["CDELT2"] = 1.000000000000E+00  # / Co-ordinate increment along axis 2
    fits_image_hdu.header["CRPIX2"] = 1.000000000000E+00  # / Reference pixel along axis 2

    # Wavelength calibration
    fits_image_hdu.header["CTYPE1"] = 'Wavelength'  # / Label for axis 3
    fits_image_hdu.header["CUNIT1"] = 'Angstroms'  # / Units for axis 3
    fits_image_hdu.header["CRVAL1"] = nflat.CRVAL1_CDELT1_CRPIX1[0]  # 7.692370611909E+03  / Co-ordinate value of axis 3
    fits_image_hdu.header["CDELT1"] = nflat.CRVAL1_CDELT1_CRPIX1[1]  # 1.575182431607E+00
    fits_image_hdu.header["CRPIX1"] = nflat.CRVAL1_CDELT1_CRPIX1[2]  # 1024. / Reference pixel along axis 3

    fits_image_hdu.header['FCAL'] = "False"
    fits_image_hdu.header['F_UNITS'] = "Counts"
    if mask[0][0] != -1:
        if no_nans:
            fits_image_hdu.header['DESCRIP'] = "Mask (1 = good, 0 = bad)"
            fits_image_hdu.header['HISTORY'] = "Mask (1 = good, 0 = bad)"
        else:
            fits_image_hdu.header['DESCRIP'] = "Mask (1 = good, nan = bad)"
            fits_image_hdu.header['HISTORY'] = "Mask (1 = good, nan = bad)"
    else:
        if flappyflat:
            fits_image_hdu.header['DESCRIP'] = "Normalized flappy flat (nresponse)"
            fits_image_hdu.header['HISTORY'] = "Normalized flappy flat (nresponse)"
        else:
            fits_image_hdu.header['DESCRIP'] = "Wavelength dependence of the throughput"
            fits_image_hdu.header['HISTORY'] = "Wavelength dependence of the throughput"

    fits_image_hdu.header['HISTORY'] = "using PyKOALA " + version  # 'Version 0.10 - 12th February 2019'
    now = datetime.datetime.now()
    fits_image_hdu.header['HISTORY'] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    if mask[0][0] == -1:
        if flappyflat:
            fits_image_hdu.header['HISTORY'] = "Created processing flappy flat filename:"
        else:
            fits_image_hdu.header['HISTORY'] = "Created processing skyflat filename:"
        fits_image_hdu.header['HISTORY'] = nflat.filename
    fits_image_hdu.header['DATE'] = now.strftime(
        "%Y-%m-%dT%H:%M:%S")  # '2002-09-16T18:52:44'   # /Date of FITS file creation

    fits_image_hdu.header['BITPIX'] = 16

    hdu_list = fits.HDUList([fits_image_hdu])

    hdu_list.writeto(filename, overwrite=True)

    if mask[0][0] != -1:
        print("\n> Mask saved in file:")
    else:
        if flappyflat:
            print("\n> Normalized flappy flat (nresponse) saved in file:")
        else:
            print("\n> Wavelength dependence of the throughput saved in file:")
    print(" ", filename)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def nresponse_flappyflat(file_f, flappyflat="", nresponse_file="",
                         correct_ccd_defects=True,
                         kernel=51, ymin=0.75, ymax=1.25, plot_fibres=[], plot=True):
    # order=13,  edgelow=20, edgehigh=20,
    """
    This task reads a flappy flat, only correcting for CCD defects, and performs
    a fit to the smoothed spectrum per fibre. It returns the normalized response.
    This also adds the attribute .nresponse to the flappyflat.
    """
    if flappyflat == "":
        if correct_ccd_defects:
            print("\n> Reading the flappyflat only correcting for CCD defects...")
        else:
            print("\n> Just reading the flappyflat correcting anything...")
        flappyflat = KOALA_RSS(file_f,
                               apply_throughput=False,
                               correct_ccd_defects=correct_ccd_defects,
                               remove_5577=False,
                               do_extinction=False,
                               sky_method="none",
                               correct_negative_sky=False,
                               plot=plot, warnings=False)
    else:
        print("\n> Flappy flat already read")

        # print "\n> Performing a {} polynomium fit to smoothed spectrum with window {} to all fibres...\n".format(order,kernel)
    print("\n> Applying median filter with window {} to all fibres to get nresponse...\n".format(kernel))

    if plot_fibres == []: plot_fibres = [0, 200, 500, 501, 700, flappyflat.n_spectra - 1]
    nresponse = np.zeros_like(flappyflat.intensity_corrected)
    for i in range(flappyflat.n_spectra):

        spectrum_ = flappyflat.intensity_corrected[
            i]  # [np.nan if x == 0 else x for x in flappyflat.intensity_corrected[i]]
        nresponse_ = signal.medfilt(spectrum_, kernel)
        nresponse[i] = [0 if x < ymin or x > ymax else x for x in nresponse_]
        nresponse[i] = [0 if np.isnan(x) else x for x in nresponse_]

        if i in plot_fibres:
            ptitle = "nresponse for fibre " + np.str(i)
            plot_plot(flappyflat.wavelength, [spectrum_, nresponse[i]], hlines=np.arange(ymin + 0.05, ymax, 0.05),
                      label=["spectrum", "medfilt"], ymin=ymin, ymax=ymax, ptitle=ptitle)

    if plot: flappyflat.RSS_image(image=nresponse, cmap=fuego_color_map)

    flappyflat.nresponse = nresponse

    print("\n> Normalized flatfield response stored in self.nresponse !!")

    if nresponse_file != "":
        print("  Also saving the obtained nresponse to file")
        print(" ", nresponse_file)
        save_nresponse(flappyflat, filename=nresponse_file, flappyflat=True)
    return nresponse


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_throughput_2D(file_skyflat, throughput_2D_file="", plot=True, also_return_skyflat=True,
                      correct_ccd_defects=True, fix_wavelengths=False, sol=[0], kernel_throughput=0):
    """
    Get a 2D array with the throughput 2D using a COMBINED skyflat / domeflat.
    It is important that this is a COMBINED file, as we should not have any cosmics/nans left.
    A COMBINED flappy flat could be also used, but that is not as good as the dome / sky flats.

    Parameters
    ----------
    file_skyflat: string
        The fits file containing the skyflat/domeflat
    plot: boolean
        If True, it plots the results
    throughput_2D_file: string
        the name of the fits file to be created with the output throughput 2D
    no_nas: booleans
        If False, it indicates the mask will be built using the nan in the edges
    correct_ccd_deffects: boolean
        If True, it corrects for ccd defects when reading the skyflat fits file
    kernel_throughput: odd number
        If not 0, the 2D throughput will be smoothed with a this kernel
    """

    print("\n> Reading a COMBINED skyflat / domeflat to get the 2D throughput...")

    if sol[0] == 0 or sol[0] == -1:
        if fix_wavelengths:
            print("  Fix wavelength requested but not solution given, ignoring it...")
            fix_wavelengths = False
    # else:
    #    if len(sol) == 3 : fix_wavelengths = True

    skyflat = KOALA_RSS(file_skyflat, correct_ccd_defects=correct_ccd_defects,
                        fix_wavelengths=fix_wavelengths, sol=sol, plot=plot)

    skyflat.apply_mask(make_nans=True)
    throughput_2D_ = np.zeros_like(skyflat.intensity_corrected)
    print("\n> Getting the throughput per wavelength...")
    for i in range(skyflat.n_wave):
        column = skyflat.intensity_corrected[:, i]
        mcolumn = column / np.nanmedian(column)
        throughput_2D_[:, i] = mcolumn

    if kernel_throughput > 0:
        print("\n  - Applying smooth with kernel =", kernel_throughput)
        throughput_2D = np.zeros_like(throughput_2D_)
        for i in range(skyflat.n_spectra):
            throughput_2D[i] = signal.medfilt(throughput_2D_[i], kernel_throughput)
        skyflat.RSS_image(image=throughput_2D, chigh=1.1, clow=0.9, cmap="binary_r")
        skyflat.history.append('- Throughput 2D smoothed with kernel ' + np.str(kernel_throughput))
    else:
        throughput_2D = throughput_2D_

    skyflat.sol = sol
    # Saving the information of fix_wavelengths in throughput_2D[0][0]
    if sol[0] != 0:
        print("\n  - The solution for fixing wavelengths has been provided")
        if sol[0] != -1:
            throughput_2D[0][
                0] = 1.0  # if throughput_2D[0][0] is 1.0, the throughput has been corrected for small wavelength variations
            skyflat.history.append('- Written data[0][0] = 1.0 for automatically identifing')
            skyflat.history.append('  that the throughput 2D data has been obtained')
            skyflat.history.append('  AFTER correcting for small wavelength variations')

    if plot:
        x = np.arange(skyflat.n_spectra)
        median_throughput = np.nanmedian(throughput_2D, axis=1)
        plot_plot(x, median_throughput, ymin=0.2, ymax=1.2, hlines=[1, 0.9, 1.1],
                  ptitle="Median value of the 2D throughput per fibre", xlabel="Fibre")
        skyflat.RSS_image(image=throughput_2D, cmap="binary_r",
                          title="\n ---- 2D throughput ----")

    skyflat_corrected = skyflat.intensity_corrected / throughput_2D
    if plot: skyflat.RSS_image(image=skyflat_corrected, title="\n Skyflat CORRECTED for 2D throughput")
    if throughput_2D_file != "":
        save_rss_fits(skyflat, data=throughput_2D, fits_file=throughput_2D_file, text="Throughput 2D ", sol=sol)

    print("\n> Throughput 2D obtained!")
    if also_return_skyflat:
        return throughput_2D, skyflat
    else:
        return throughput_2D


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_continuum_in_range(w, s, low_low, low_high, high_low, high_high,
                           pmin=12, pmax=88, only_correct_negative_values=False,
                           fit_degree=2, plot=True, verbose=True, warnings=True):
    """
    This task computes the continuum of a 1D spectrum using the intervals [low_low, low_high]
    and [high_low, high_high] and returns the spectrum but with the continuum in the range
    [low_high, high_low] (where a feature we want to remove is located).
    """
    s_low = s[np.where((w <= low_low))]
    s_high = s[np.where((w >= high_high))]

    w_fit = w[np.where((w > low_low) & (w < high_high))]
    w_fit_low = w[np.where((w > low_low) & (w < low_high))]
    w_fit_high = w[np.where((w > high_low) & (w < high_high))]

    y_fit = s[np.where((w > low_low) & (w < high_high))]
    y_fit_low = s[np.where((w > low_low) & (w < low_high))]
    y_fit_high = s[np.where((w > high_low) & (w < high_high))]

    # Remove outliers
    median_y_fit_low = np.nanmedian(y_fit_low)
    for i in range(len(y_fit_low)):
        if np.nanpercentile(y_fit_low, 2) > y_fit_low[i] or y_fit_low[i] > np.nanpercentile(y_fit_low, 98): y_fit_low[
            i] = median_y_fit_low

    median_y_fit_high = np.nanmedian(y_fit_high)
    for i in range(len(y_fit_high)):
        if np.nanpercentile(y_fit_high, 2) > y_fit_high[i] or y_fit_high[i] > np.nanpercentile(y_fit_high, 98):
        y_fit_high[i] = median_y_fit_high

    w_fit_cont = np.concatenate((w_fit_low, w_fit_high))
    y_fit_cont = np.concatenate((y_fit_low, y_fit_high))

    try:
        fit = np.polyfit(w_fit_cont, y_fit_cont, fit_degree)
        yfit = np.poly1d(fit)
        y_fitted = yfit(w_fit)

        y_fitted_low = yfit(w_fit_low)
        median_low = np.nanmedian(y_fit_low - y_fitted_low)
        rms = []
        for i in range(len(y_fit_low)):
            rms.append(y_fit_low[i] - y_fitted_low[i] - median_low)

        #    rms=y_fit-y_fitted
        lowlimit = np.nanpercentile(rms, pmin)
        highlimit = np.nanpercentile(rms, pmax)

        corrected_s_ = copy.deepcopy(y_fit)
        for i in range(len(w_fit)):
            if w_fit[i] >= low_high and w_fit[i] <= high_low:  # ONLY CORRECT in [low_high,high_low]
                if only_correct_negative_values:
                    if y_fit[i] <= 0:
                        corrected_s_[i] = y_fitted[i]
                else:
                    if y_fit[i] - y_fitted[i] <= lowlimit or y_fit[i] - y_fitted[i] >= highlimit: corrected_s_[i] = \
                    y_fitted[i]

        corrected_s = np.concatenate((s_low, corrected_s_))
        corrected_s = np.concatenate((corrected_s, s_high))

        if plot:
            ptitle = "Correction in range  " + np.str(np.round(low_low, 2)) + " - [ " + np.str(
                np.round(low_high, 2)) + " - " + np.str(np.round(high_low, 2)) + " ] - " + np.str(
                np.round(high_high, 2))
            plot_plot(w_fit, [y_fit, y_fitted, y_fitted - highlimit, y_fitted - lowlimit, corrected_s_],
                      color=["r", "b", "black", "black", "green"], alpha=[0.3, 0.7, 0.2, 0.2, 0.5], xmin=low_low - 40,
                      xmax=high_high + 40, vlines=[low_low, low_high, high_low, high_high], ptitle=ptitle,
                      ylabel="Normalized flux")
            # plot_plot(w,[s,corrected_s],xmin=low_low-40, xmax=high_high+40,vlines=[low_low,low_high,high_low,high_high])
    except Exception:
        if warnings: print("  Fitting the continuum failed! Nothing done.")
        corrected_s = s

    return corrected_s


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def telluric_correction_from_star(objeto, save_telluric_file="",
                                  object_rss=False,
                                  high_fibres=20,
                                  list_of_telluric_ranges=[[0]], order=2,
                                  apply_tc=False,
                                  wave_min=0, wave_max=0,
                                  plot=True, fig_size=12, verbose=True):
    """
    Get telluric correction using a spectrophotometric star

    IMPORTANT! Check task self.get_telluric_correction !!!

    Parameters
    ----------
    high_fibres: integer
        number of fibers to add for obtaining spectrum
    apply_tc : boolean (default = False)
        apply telluric correction to data

    Example
    ----------
    telluric_correction_star1 = star1r.get_telluric_correction(high_fibres=15)
    """

    print("\n> Obtaining telluric correction using spectrophotometric star...")

    try:
        wlm = objeto.combined_cube.wavelength
        rss = objeto.rss1
        is_combined_cube = True
    except Exception:
        wlm = objeto.wavelength
        rss = objeto
        is_combined_cube = False

    if wave_min == 0: wave_min = wlm[0]
    if wave_max == 0: wave_max = wlm[-1]

    if is_combined_cube:
        if verbose: print(
            "  The given object is a combined cube. Using this cube for extracting the spectrum of the star...")
        if objeto.combined_cube.seeing == 0:
            objeto.combined_cube.half_light_spectrum(5, plot=plot, min_wave=wave_min, max_wave=wave_max)
        estrella = objeto.combined_cube.integrated_star_flux
    else:
        if object_rss:
            if verbose: print("  The given object is a RSS. Using the", high_fibres,
                              " fibres with the highest intensity to get the spectrum of the star...")
            integrated_intensity_sorted = np.argsort(objeto.integrated_fibre)
            intensidad = objeto.intensity_corrected
            region = []
            for fibre in range(high_fibres):
                region.append(integrated_intensity_sorted[-1 - fibre])
            estrella = np.nansum(intensidad[region], axis=0)
            # bright_spectrum = objeto.plot_combined_spectrum(list_spectra=fibre_list, median=True, plot=False)
        else:
            if verbose: print(
                "  The given object is a cube. Using this cube for extracting the spectrum of the star...")
            if objeto.seeing == 0:
                objeto.half_light_spectrum(5, plot=plot, min_wave=wave_min, max_wave=wave_max)
            estrella = objeto.integrated_star_flux

    if list_of_telluric_ranges[0][0] == 0:
        list_of_telluric_ranges = [[6150, 6245, 6350, 6430], [6720, 6855, 7080, 7150], [7080, 7150, 7500, 7580],
                                   [7400, 7580, 7720, 7850], [7850, 8100, 8450, 8700]]

    telluric_correction = telluric_correction_using_bright_continuum_source(rss, bright_spectrum=estrella,
                                                                            list_of_telluric_ranges=list_of_telluric_ranges,
                                                                            order=order, plot=plot, verbose=verbose)

    if plot:
        plt.figure(figsize=(fig_size, fig_size / 2.5))
        if object_rss:
            print("  Example of telluric correction using fibres", region[0], " and ", region[1], ":")
            plt.plot(wlm, intensidad[region[0]], color="b", alpha=0.3)
            plt.plot(wlm, intensidad[region[0]] * telluric_correction, color="g", alpha=0.5)
            plt.plot(wlm, intensidad[region[1]], color="b", alpha=0.3)
            plt.plot(wlm, intensidad[region[1]] * telluric_correction, color="g", alpha=0.5)
            plt.ylim(np.nanmin(intensidad[region[1]]), np.nanmax(intensidad[region[0]]))  # CHECK THIS AUTOMATICALLY
        else:
            if is_combined_cube:
                print("  Telluric correction applied to this star (" + objeto.combined_cube.object + ") :")
            else:
                print("  Telluric correction applied to this star (" + objeto.object + ") :")
            plt.plot(wlm, estrella, color="b", alpha=0.3)
            plt.plot(wlm, estrella * telluric_correction, color="g", alpha=0.5)
            plt.ylim(np.nanmin(estrella), np.nanmax(estrella))

        plt.axvline(x=wave_min, color='k', linestyle='--')
        plt.axvline(x=wave_max, color='k', linestyle='--')
        plt.xlim(wlm[0] - 10, wlm[-1] + 10)
        plt.xlabel("Wavelength [$\mathrm{\AA}$]")
        if list_of_telluric_ranges[0][0] != 0:
            for i in range(len(list_of_telluric_ranges)):
                plt.axvspan(list_of_telluric_ranges[i][1], list_of_telluric_ranges[i][2], color='r', alpha=0.1)
        plt.minorticks_on()
        plt.show()
        plt.close()

    if apply_tc:  # Check this
        print("  Applying telluric correction to this star...")
        if object_rss:
            for i in range(objeto.n_spectra):
                objeto.intensity_corrected[i, :] = objeto.intensity_corrected[i, :] * telluric_correction
        else:
            if is_combined_cube:
                objeto.combined_cube.integrated_star_flux = objeto.combined_cube.integrated_star_flux * telluric_correction
                for i in range(objeto.combined_cube.n_rows):
                    for j in range(objeto.combined_cube.n_cols):
                        objeto.combined_cube.data[:, i, j] = objeto.combined_cube.data[:, i, j] * telluric_correction
            else:
                objeto.integrated_star_flux = objeto.integrated_star_flux * telluric_correction
                for i in range(objeto.n_rows):
                    for j in range(objeto.n_cols):
                        objeto.data[:, i, j] = objeto.data[:, i, j] * telluric_correction
    else:
        print("  As apply_tc = False , telluric correction is NOT applied...")

    if is_combined_cube:
        objeto.combined_cube.telluric_correction = telluric_correction
    else:
        objeto.telluric_correction = telluric_correction

        # save file if requested
    if save_telluric_file != "":
        spectrum_to_text_file(wlm, telluric_correction, filename=save_telluric_file, verbose=False)
        if verbose: print("\n> Telluric correction saved in text file", save_telluric_file, " !!")
    return telluric_correction


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def telluric_correction_using_bright_continuum_source(objeto, save_telluric_file="",
                                                      fibre_list=[-1], high_fibres=20,
                                                      bright_spectrum=[0], odd_number=51,
                                                      list_of_telluric_ranges=[[0]], order=2,
                                                      plot=True, verbose=True):
    """
    "objeto" can be a cube that has been read from a fits file or
    an rss, from which getting the integrated spectrum of a bright source.
    If bright_spectrum is given, for example, an 1D spectrum from a cube in "spec",
    "rss" have to be a valid rss for getting wavelength,
    use:> telluric_correction_with_bright_continuum_source(EG21_red.rss1, bright_spectrum=spec)
    """
    w = objeto.wavelength

    if list_of_telluric_ranges[0][0] == 0:
        list_of_telluric_ranges = [[6150, 6245, 6350, 6430], [6650, 6800, 7080, 7150],
                                   [7080, 7150, 7440, 7580], [7440, 7580, 7720, 7820], [7720, 8050, 8400, 8640]]

    if verbose: print("\n> Obtaining telluric correction using a bright continuum source...")

    if np.nanmedian(bright_spectrum) == 0:
        if fibre_list[0] == -1:
            if verbose: print("  Using the", high_fibres,
                              "fibres with highest intensity for obtaining normalized spectrum of bright source...")
            integrated_intensity_sorted = np.argsort(objeto.integrated_fibre)
            fibre_list = []
            for fibre_ in range(high_fibres):
                fibre_list.append(integrated_intensity_sorted[-1 - fibre_])
        else:
            if verbose: print(
                "  Using the list of fibres provided for obtaining normalized spectrum of bright source...")
        bright_spectrum = objeto.plot_combined_spectrum(list_spectra=fibre_list, median=True, plot=False)
    else:
        if verbose: print("  Using the normalized spectrum of a bright source provided...")

    if verbose: print("  Deriving median spectrum using a", odd_number, "window...")
    bs_m = signal.medfilt(bright_spectrum, odd_number)

    # Normalizing the spectrum
    # bright = bright_spectrum / np.nanmedian(bright_spectrum)

    # telluric_correction[l]= smooth_med_star[l]/estrella[l]   # LUIGI

    bright = bright_spectrum

    vlines = []
    axvspan = []
    for t_range in list_of_telluric_ranges:
        vlines.append(t_range[0])
        vlines.append(t_range[3])
        axvspan.append([t_range[1], t_range[2]])

    if plot: plot_plot(w, [bright_spectrum / np.nanmedian(bright_spectrum), bs_m / np.nanmedian(bright_spectrum)],
                       color=["b", "g"], alpha=[0.4, 0.8], vlines=vlines, axvspan=axvspan,
                       ymax=np.nanmax(bs_m) / np.nanmedian(bright_spectrum) + 0.1,
                       ptitle="Combined bright spectrum (blue) and median bright spectrum (green)")

    ntc = np.ones_like(objeto.wavelength)

    if verbose: print("  Getting the telluric correction in specified ranges using ", order,
                      " order fit to continuum:\n")

    for t_range in list_of_telluric_ranges:

        low_low = t_range[0]
        low_high = t_range[1]
        high_low = t_range[2]
        high_high = t_range[3]

        ptitle = "Telluric correction in range " + np.str(low_low) + " - [ " + np.str(low_high) + " , " + np.str(
            high_low) + " ] - " + np.str(high_high)

        if verbose: print("  - ", ptitle)

        # bright = bright_spectrum / np.nanmedian(bright_spectrum[np.where((w > low_low) & (w < high_high))])

        w_fit = w[np.where((w > low_low) & (w < high_high))]
        w_fit_low = w[np.where((w > low_low) & (w < low_high))]
        w_fit_range = w[np.where((w >= low_high) & (w <= high_low))]
        w_fit_high = w[np.where((w > high_low) & (w < high_high))]

        y_fit = bright[np.where((w > low_low) & (w < high_high))]
        y_fit_low = bright[np.where((w > low_low) & (w < low_high))]
        y_fit_range = bright[np.where((w >= low_high) & (w <= high_low))]
        y_fit_high = bright[np.where((w > high_low) & (w < high_high))]

        w_fit_cont = np.concatenate((w_fit_low, w_fit_high))
        y_fit_cont = np.concatenate((y_fit_low, y_fit_high))

        fit = np.polyfit(w_fit_cont, y_fit_cont, order)
        yfit = np.poly1d(fit)
        y_fitted = yfit(w_fit)
        y_fitted_range = yfit(w_fit_range)

        ntc_ = y_fitted_range / y_fit_range

        ntc_low_index = w.tolist().index(w[np.where((w >= low_high) & (w <= high_low))][0])
        ntc_high_index = w.tolist().index(w[np.where((w >= low_high) & (w <= high_low))][-1])

        # ntc = [ntc_(j) for j in range(ntc_low_index,ntc_high_index+1)  ]
        j = 0
        for i in range(ntc_low_index, ntc_high_index + 1):
            ntc[i] = ntc_[j]
            j = j + 1

        y_range_corr = y_fit_range * ntc_

        y_corr_ = np.concatenate((y_fit_low, y_range_corr))
        y_corr = np.concatenate((y_corr_, y_fit_high))

        if plot: plot_plot(w_fit, [y_fit, y_fitted, y_corr], color=["b", "r", "g"], xmin=low_low - 40,
                           xmax=high_high + 40,
                           axvspan=[[low_high, high_low]], vlines=[low_low, low_high, high_low, high_high],
                           ptitle=ptitle, ylabel="Normalized flux")

    telluric_correction = np.array(
        [1.0 if x < 1.0 else x for x in ntc])  # Telluric correction should not have corrections < 1.0

    if plot:
        plot_plot(w, telluric_correction, ptitle="Telluric correction", ylabel="Intensity",  # vlines=vlines,
                  axvspan=axvspan, ymax=3, ymin=0.9, hlines=[1])

    # save file if requested
    if save_telluric_file != "":
        spectrum_to_text_file(w, telluric_correction, filename=save_telluric_file, verbose=False)
        if verbose: print("\n> Telluric correction saved in text file", save_telluric_file, " !!")

    return telluric_correction


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def create_fits_with_mask(list_files_for_mask, filename="", plot=True, no_nans=True):
    """
    Creates a mask using list_files_for_mask
    """
    print("\n> Creating mask using files provided...")

    # First, read the rss files
    intensities_for_mask = []
    for i in range(len(list_files_for_mask)):
        rss = KOALA_RSS(list_files_for_mask[i], plot_final_rss=False, verbose=False)
        intensities_for_mask.append(rss.intensity)

    # Combine intensities to eliminate nans because of cosmic rays
    mask_ = np.nanmedian(intensities_for_mask, axis=0)

    # divide it by itself to get 1 and nans
    mask = mask_ / mask_

    # Change nans to 0 (if requested)
    if no_nans:
        for i in range(len(mask)):
            mask[i] = [0 if np.isnan(x) else 1 for x in mask[i]]

    # Plot for fun if requested
    if plot:
        if no_nans:
            rss.RSS_image(image=mask, cmap="binary_r", clow=-0.0001, chigh=1., title=" - Mask",
                          color_bar_text="Mask value (black = 0, white = 1)")
        else:
            rss.RSS_image(image=mask, cmap="binary", clow=-0.0001, chigh=1., title=" - Mask",
                          color_bar_text="Mask value (black = 1, white = nan)")

    # Save mask in file if requested
    if filename != "":
        save_nresponse(rss, filename, mask=mask, no_nans=no_nans)

    print("\n> Mask created!")
    return mask


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_minimum_spectra(rss_file_list, percentile=0,
                        apply_throughput=False,
                        throughput_2D=[], throughput_2D_file="",
                        correct_ccd_defects=False, plot=True):
    ic_list = []
    for name in rss_file_list:
        rss = KOALA_RSS(name, apply_throughput=apply_throughput,
                        throughput_2D=throughput_2D, throughput_2D_file=throughput_2D_file,
                        correct_ccd_defects=correct_ccd_defects, plot=False)

        ic_list.append(rss.intensity_corrected)

    n_rss = len(rss_file_list)
    ic_min = np.nanmedian(ic_list, axis=0)
    if percentile == 0:
        percentile = 100. / n_rss - 2
        # ic_min = np.percentile(ic_list, percentile, axis=0)

    if plot:
        rss.RSS_image(image=ic_min, cmap="binary_r")
    return ic_min


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def merge_extracted_flappyflats(flappyflat_list, write_file="", path="", verbose=True):
    print("\n> Merging flappy flats...")
    if verbose: print("  - path : ", path)
    data_list = []
    for flappyflat in flappyflat_list:
        file_to_fix = path + flappyflat
        ftf = fits.open(file_to_fix)
        data_ = ftf[0].data
        exptime = ftf[0].header['EXPOSED']
        data_list.append(data_ / exptime)
        if verbose: print("  - File", flappyflat, "   exptime =", exptime)

    merged = np.nanmedian(data_list, axis=0)

    # Save file
    if write_file != "":
        ftf[0].data = merged
        ftf[0].header['EXPOSED'] = 1.0
        ftf[0].header['HISTORY'] = "Median flappyflat using Python - A.L-S"
        print("\n  Saving merged flappyflat to file ", write_file, "...")
        ftf.writeto(path + write_file, overwrite=True)

    return merged


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fix_these_features(w, s, features=[], sky_fibres=[], sky_spectrum=[], objeto="", plot_all=False):  # objeto=test):

    ff = copy.deepcopy(s)

    kind_of_features = [features[i][0] for i in range(len(features))]
    #    if "g" in kind_of_features or "s" in kind_of_features:
    #        if len(sky_spectrum) == 0 :
    #            sky_spectrum=objeto.plot_combined_spectrum(list_spectra=sky_fibres, plot=False, median=True)
    if "s" in kind_of_features:
        if len(sky_spectrum) == 0:
            sky_spectrum = objeto.plot_combined_spectrum(list_spectra=sky_fibres, plot=plot_all, median=True)

    for feature in features:
        # plot_plot(w,ff,xmin=feature[1]-20,xmax=feature[4]+20)
        if feature[0] == "l":  # Line
            resultado = fluxes(w, ff, feature[1], lowlow=feature[2], lowhigh=feature[3], highlow=feature[4],
                               highhigh=feature[5], broad=feature[6], plot=feature[7], verbose=feature[8])
            ff = resultado[11]
        if feature[0] == "r":  # range
            ff = get_continuum_in_range(w, ff, feature[1], feature[2], feature[3], feature[4], pmin=feature[5],
                                        pmax=feature[6], fit_degree=feature[7], plot=feature[8], verbose=feature[9])
        if feature[0] == "g":  # gaussian
            #            resultado = fluxes(w,sky_spectrum, feature[1], lowlow=feature[2],lowhigh=feature[3],highlow=feature[4],highhigh=feature[5],broad=feature[6],plot=feature[7],verbose=feature[8])
            #            sky_feature=sky_spectrum-resultado[11]
            resultado = fluxes(w, s, feature[1], lowlow=feature[2], lowhigh=feature[3], highlow=feature[4],
                               highhigh=feature[5], broad=feature[6], plot=feature[7], verbose=feature[8])
            sky_feature = s - resultado[11]
            ff = ff - sky_feature
        if feature[0] == "n":  # negative values
            ff = get_continuum_in_range(w, ff, feature[1], feature[2], feature[3], feature[4], pmin=feature[5],
                                        pmax=feature[6], fit_degree=feature[7], plot=feature[8], verbose=feature[9],
                                        only_correct_negative_values=True)
        if feature[0] == "s":  # sustract
            ff_low = ff[np.where(w < feature[2])]
            ff_high = ff[np.where(w > feature[3])]
            subs = ff - sky_spectrum
            ff_replace = subs[np.where((w >= feature[2]) & (w <= feature[3]))]
            ff_ = np.concatenate((ff_low, ff_replace))
            ff_ = np.concatenate((ff_, ff_high))

            c = get_continuum_in_range(w, ff_, feature[1], feature[2], feature[3], feature[4], pmin=feature[5],
                                       pmax=feature[6], fit_degree=feature[7], plot=feature[8], verbose=feature[9],
                                       only_correct_negative_values=True)

            if feature[8] or plot_all:  # plot
                vlines = [feature[1], feature[2], feature[3], feature[4]]
                plot_plot(w, [ff, ff_, c], xmin=feature[1] - 20, xmax=feature[4] + 20, vlines=vlines,
                          alpha=[0.1, 0.2, 0.8], ptitle="Correcting 's'")

            ff = c

    return ff


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fix_these_features_in_all_spectra(objeto, features=[], sky_fibres=[], sky_spectrum=[],
                                      fibre_list=[-1],
                                      replace=False, plot=True):
    if len(sky_fibres) != 0 and len(sky_spectrum) == 0:
        sky_spectrum = objeto.plot_combined_spectrum(list_spectra=sky_fibres, plot=False, median=True)

    fix = copy.deepcopy(objeto.intensity_corrected)

    if len(fibre_list) == 0:  # [0] == -1:
        print("\n> Fixing the requested features in all the fibres...")
        fibre_list = list(range(len(fix)))
    else:
        print("\n> Fixing the requested features in the given fibres...")

    n_spectra = len(fibre_list)
    w = objeto.wavelength

    if plot: objeto.RSS_image(title=" - Before fixing features")

    sys.stdout.write("  Fixing {} spectra...       ".format(n_spectra))
    sys.stdout.flush()
    output_every_few = np.sqrt(n_spectra) + 1
    next_output = -1
    i = 0
    for fibre in fibre_list:  # range(n_spectra):
        i = i + 1
        if fibre > next_output:
            sys.stdout.write("\b" * 6)
            sys.stdout.write("{:5.2f}%".format(i * 100. / n_spectra))
            sys.stdout.flush()
            next_output = fibre + output_every_few
        fix[fibre] = fix_these_features(w, fix[fibre], features=features, sky_fibres=sky_fibres,
                                        sky_spectrum=sky_spectrum)

    if plot: objeto.RSS_image(image=fix, title=" - After fixing features")
    if replace:
        print("\n  Replacing everything in self.intensity_corrected...")
        objeto.intensity_corrected = copy.deepcopy(fix)

    objeto.history.append("- Sky residuals cleaned on these features:")
    for feature in features:
        objeto.history.append("  " + np.str(feature))

    return fix


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fix_red_edge(w, f, fix_from=9220, median_from=8800, kernel_size=101, disp=1.5, plot=False):
    """
    CAREFUL! This should consider the REAL emission lines in the red edge!

    Parameters
    ----------
    w : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    fix_from : TYPE, optional
        DESCRIPTION. The default is 9220.
    median_from : TYPE, optional
        DESCRIPTION. The default is 8800.
    kernel_size : TYPE, optional
        DESCRIPTION. The default is 101.
    disp : TYPE, optional
        DESCRIPTION. The default is 1.5.
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    ff : TYPE
        DESCRIPTION.

    """

    min_value, median_value, max_value, std, rms, snr = basic_statistics(f, x=w, xmin=median_from, xmax=fix_from,
                                                                         return_data=True, verbose=False)

    f_fix = []
    f_fix.extend((f[i]) for i in range(len(w)) if (w[i] >= fix_from))
    f_still = []
    f_still.extend((f[i]) for i in range(len(w)) if (w[i] < fix_from))

    f_fix = [median_value if (median_value + disp * std < x) or (median_value - disp * std > x) else x for x in f_fix]
    ff = np.concatenate((f_still, f_fix))

    if plot: plot_plot(w, [f, ff], vlines=[median_from, fix_from], xmin=median_from)
    return ff


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fix_blue_edge(w, f, fix_to=6100, median_to=6300, kernel_size=101, disp=1.5, plot=False):
    min_value, median_value, max_value, std, rms, snr = basic_statistics(f, x=w, xmin=fix_to, xmax=median_to,
                                                                         return_data=True, verbose=False)

    f_fix = []
    f_fix.extend((f[i]) for i in range(len(w)) if (w[i] <= fix_to))
    f_still = []
    f_still.extend((f[i]) for i in range(len(w)) if (w[i] > fix_to))

    f_fix = [median_value if (median_value + disp * std < x) or (median_value - disp * std > x) else x for x in f_fix]

    ff = np.concatenate((f_fix, f_still))

    if plot: plot_plot(w, [f, ff], vlines=[median_to, fix_to], xmax=median_to)
    return ff


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def find_cosmics_in_cut(x, cut_wave, cut_brightest_line, line_wavelength=0.,
                        kernel_median_cosmics=5, cosmic_higher_than=100, extra_factor=1., plot=False, verbose=False):
    """
    Task used in  rss.kill_cosmics()

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    cut_wave : TYPE
        DESCRIPTION.
    cut_brightest_line : TYPE
        DESCRIPTION.
    line_wavelength : TYPE, optional
        DESCRIPTION. The default is 0..
    kernel_median_cosmics : TYPE, optional
        DESCRIPTION. The default is 5.
    cosmic_higher_than : TYPE, optional
        DESCRIPTION. The default is 100.
    extra_factor : TYPE, optional
        DESCRIPTION. The default is 1..
    plot : TYPE, optional
        DESCRIPTION. The default is False.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    cosmics_list : TYPE
        DESCRIPTION.

    """

    gc_bl = signal.medfilt(cut_brightest_line, kernel_size=kernel_median_cosmics)
    max_val = np.abs(cut_brightest_line - gc_bl)

    gc = signal.medfilt(cut_wave, kernel_size=kernel_median_cosmics)
    verde = np.abs(cut_wave - gc) - extra_factor * max_val

    cosmics_list = [i for i, x in enumerate(verde) if x > cosmic_higher_than]

    if plot:
        ptitle = "Cosmic identification in cut"
        if line_wavelength != 0: ptitle = "Cosmic identification in cut at " + np.str(
            line_wavelength) + " $\mathrm{\AA}$"
        plot_plot(x, verde, ymin=0, ymax=200, hlines=[cosmic_higher_than], ptitle=ptitle,
                  ylabel="abs (cut - medfilt(cut)) - extra_factor * max_val")

    if verbose:
        if line_wavelength == 0:
            print("\n> Identified", len(cosmics_list), "cosmics in fibres", cosmics_list)
        else:
            print("\n> Identified", len(cosmics_list), "cosmics at", np.str(line_wavelength), "A in fibres",
                  cosmics_list)
    return cosmics_list


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# def kill_cosmics_OLD(w, spectrum, rms_factor=3., extra_indices = 3,
#                      brightest_line="Ha", brightest_line_wavelength =6563,
#                      ymin="",ymax="",plot=True, plot_final_plot=True, verbose=True, warnings=True):

#     if verbose: print("\n> Killing cosmics... ")
#     spectrum_m = signal.medfilt(spectrum,151)
#     disp_vector =spectrum-spectrum_m
#     dispersion = np.nanpercentile(disp_vector,90)
#     rms =np.sqrt(np.abs(np.nansum(spectrum)/len(spectrum)))

#     # CHECK WHAT TO PUT HERE.... maybe add rms del spectrum_m
#     #max_rms = rms * rms_factor
#     max_rms = dispersion * rms_factor
#     if plot or plot_final_plot:
#         ptitle = "Searching cosmics. Grey line: rms = "+np.str(np.round(rms,3))+" , green line : dispersion p90 = "+np.str(np.round(dispersion,3))+" , red line: "+np.str(np.round(rms_factor,1))+" * d_p99.95 = "+np.str(np.round(max_rms,3))
#         plot_plot(w,disp_vector,hlines=[rms, max_rms, dispersion], chlines=["k","r","g"],ptitle=ptitle, ylabel="Dispersion (spectrum - spectrum_median_filter_151",
#                   ymin=np.nanpercentile(disp_vector,2),ymax=np.nanpercentile(disp_vector,99.95))
#     bad_waves_= w[np.where(disp_vector > max_rms)]
#     #if verbose: print bad_waves_

#     bad_waves_indices=[]
#     rango=[]
#     next_contiguous=0
#     for j in range(len(bad_waves_)):
#         _bw_ = [i for i,x in enumerate(w) if x==bad_waves_[j]]
#         if j == 0:
#             rango.append(_bw_[0])
#         else:
#             if _bw_[0] != next_contiguous:
#                 bad_waves_indices.append(rango)
#                 rango=[_bw_[0]]
#             else:
#                 rango.append(_bw_[0])
#         next_contiguous =_bw_[0] + 1
#     bad_waves_indices.append(rango) # The last one
#     #if verbose: print  bad_waves_indices

#     if len(bad_waves_indices[0]) > 0:
#         bad_waves_ranges = []
#         for i in range(len(bad_waves_indices)):
#             #if verbose: print "  - ", w[bad_waves_indices[i]
#             if len(bad_waves_indices[i]) > 1:
#                 a = bad_waves_indices[i][0] - extra_indices
#                 b = bad_waves_indices[i][-1] + extra_indices
#             else:
#                 a = bad_waves_indices[i][0] - extra_indices
#                 b = bad_waves_indices[i][0] + extra_indices
#             bad_waves_ranges.append([a,b])

#         # HERE WE SHOULD REMOVE CLEANING IN RANGES WITH EMISSION LINES
#         # First check that the brightest_line is not observabled (e.g. Gaussian fit does not work)
#         # Then remove all ranges with emission lines

#         if verbose:
#             print("  Cosmics found in these ranges:")
#             for i in range(len(bad_waves_ranges)):
#                 if bad_waves_ranges[i][-1] < len(w):
#                     print("  - ",np.round(w[bad_waves_ranges[i]],2))

#         kc = copy.deepcopy(spectrum)
#         vlines=[]
#         for i in range(len(bad_waves_ranges)):
#             try:
# #            print bad_waves_ranges[i][0], 21, "  -  ", bad_waves_ranges[i][-1]-21, len(w)
# #            if bad_waves_ranges[i][0] > 21 and bad_waves_ranges[i][-1]-21 < len(w) :
#                 k1=get_continuum_in_range(w,kc,w[bad_waves_ranges[i][0]-20],w[bad_waves_ranges[i][0]],w[bad_waves_ranges[i][1]],w[bad_waves_ranges[i][1]+20], plot=plot)
#                 kc = copy.deepcopy(k1)
#                 vlines.append((w[bad_waves_ranges[i][0]]+w[bad_waves_ranges[i][-1]])/2)
#             except Exception:
#                 if warnings: print("  WARNING: This cosmic is in the edge or there is a 'nan' there, and therefore it can't be corrected...")

#         if plot or plot_final_plot:
#             plot_plot(w,[spectrum_m,spectrum,kc], vlines=vlines,ymin=ymin,ymax=ymax,
#                       label=["Medium spectrum","Spectrum","Spectrum corrected"],  ptitle = "Removing cosmics in given spectrum")
#     else:
#         if verbose: print("\n> No cosmics found! Nothing done!")
#         kc=spectrum

#     return kc
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def replace_el_in_sky_spectrum(rss, sky_r_self, sky_r_star, cut_red_end=0,
                               ranges_with_emission_lines=[0], scale_sky_1D=0,
                               brightest_line="Ha", brightest_line_wavelength=0,
                               plot=True, verbose=True):
    """
    Replace emission lines in sky spectrum using another sky spectrum.
    This task is useful when H-alpha or other bright lines are all over the place in a
    science rss. Using the sky obtained from the rss of a star nearby,
    the task scales both spectra and then replace the selected ranges that have
    emission lines with the values of the scaled sky spectrum.
    The task also replaces the red end (right edge) of object spectrum.

    """
    #### NOTE : the scaling should be done LINE BY LINE !!!!

    if verbose:
        print("\n> Replacing ranges with emission lines in object spectrum using a sky spectrum...")

    good_sky_red = copy.deepcopy(sky_r_self)

    w = rss.wavelength

    if scale_sky_1D == 0:
        print("  Automatically searching for the scale factor between two spectra (object and sky)...")
        scale_sky_1D = auto_scale_two_spectra(rss, sky_r_self, sky_r_star, plot=plot, verbose=False)
    else:
        print("  Using the scale factor provided, scale_sky_1D = ", scale_sky_1D, "...")

    if ranges_with_emission_lines[0] == 0:
        #                             He I 5875.64      [O I]           Ha+[N II]   He I 6678.15   [S II]    [Ar III] 7135.78   [S III] 9069
        ranges_with_emission_lines_ = [[5870., 5882.], [6290, 6308], [6546, 6591], [6674, 6684], [6710, 6742],
                                       [7128, 7148], [9058, 9081]]
        ranges_with_emission_lines = []
        for i in range(len(ranges_with_emission_lines_)):
            if ranges_with_emission_lines_[i][0] > w[0] and ranges_with_emission_lines_[i][1] < w[-1]:
                ranges_with_emission_lines.append(ranges_with_emission_lines_[i])

    brightest_line_wavelength_rest = 6562.82
    if brightest_line == "O3" or brightest_line == "O3b": brightest_line_wavelength_rest = 5006.84
    if brightest_line == "Hb" or brightest_line == "hb": brightest_line_wavelength_rest = 4861.33

    redshift = brightest_line_wavelength / brightest_line_wavelength_rest - 1.

    do_this = True
    if brightest_line_wavelength != 0:
        print("  Brightest emission line in object is ", brightest_line, ", centered at ", brightest_line_wavelength,
              "A, redshift = ", redshift)
    else:
        print(
            "\n\n\n****************************************************************************************************")
        print("\n> WARNING !!   No wavelength provided to 'brightest_line_wavelength', no replacement can be done !!!")
        print("               Run this again providing a value to 'brightest_line_wavelength' !!!\n")
        print(
            "****************************************************************************************************\n\n\n")

        do_this = False
        good_sky_red = sky_r_self

    if do_this:

        print("\n  Wavelength ranges to replace (redshift considered) : ")
        for rango in ranges_with_emission_lines:
            print("  - ", np.round((redshift + 1) * rango[0], 2), " - ", np.round((redshift + 1) * rango[1], 2))

        change_rango = False
        rango = 1
        i = 0
        while rango < len(ranges_with_emission_lines) + 1:
            if w[i] > (redshift + 1) * ranges_with_emission_lines[rango - 1][0] and w[i] < (redshift + 1) * \
                    ranges_with_emission_lines[rango - 1][1]:
                good_sky_red[i] = sky_r_star[i] * scale_sky_1D
                change_rango = True
            else:
                if change_rango:
                    change_rango = False
                    rango = rango + 1
            i = i + 1

            # Add the red end  if cut_red_end is NOT -1
        if cut_red_end != -1:
            if cut_red_end == 0:
                # Using the value of the mask of the rss
                cut_red_end = rss.valid_wave_max - 6  # a bit extra
            if verbose: print("  Also fixing the red end of the object spectrum from ", np.round(cut_red_end, 2), "...")
            w_ = np.abs(w - cut_red_end)
            i_corte = w_.tolist().index(np.nanmin(w_))
            good_sky_red[i_corte:-1] = sky_r_star[i_corte:-1] * scale_sky_1D
        else:
            if verbose: print("  The red end of the object spectrum has not been modified as cut_red_end = -1")

        if plot:
            if verbose: print("\n  Plotting the results ...")

            vlines = []
            rango_plot = []
            for rango in ranges_with_emission_lines:
                vlines.append(rango[0] * (redshift + 1))
                vlines.append(rango[1] * (redshift + 1))
                _rango_plot_ = [vlines[-2], vlines[-1]]
                rango_plot.append(_rango_plot_)

            ptitle = "Checking the result of replacing ranges with emission lines with sky"
            label = ["Sky * scale", "Self sky", "Replacement Sky"]

            # print(vlines)
            # print(rango_plot)

            if ranges_with_emission_lines[0][0] == 5870:
                plot_plot(w, [sky_r_star * scale_sky_1D, sky_r_self, good_sky_red],
                          ptitle=ptitle, label=label, axvspan=rango_plot,
                          xmin=5800 * (redshift + 1), xmax=6100 * (redshift + 1), vlines=vlines)
            plot_plot(w, [sky_r_star * scale_sky_1D, sky_r_self, good_sky_red],
                      ptitle=ptitle, label=label,
                      xmin=6200 * (redshift + 1), xmax=6400 * (redshift + 1), vlines=vlines, axvspan=rango_plot)
            plot_plot(w, [sky_r_star * scale_sky_1D, sky_r_self, good_sky_red], ptitle=ptitle,
                      xmin=6500 * (redshift + 1), xmax=6800 * (redshift + 1), vlines=vlines, label=label,
                      axvspan=rango_plot)
            plot_plot(w, [sky_r_star * scale_sky_1D, sky_r_self, good_sky_red], ptitle=ptitle,
                      xmin=6800 * (redshift + 1), xmax=7200 * (redshift + 1), vlines=vlines, label=label,
                      axvspan=rango_plot)
            if ranges_with_emission_lines[-1][0] == 9058:
                plot_plot(w, [sky_r_star * scale_sky_1D, sky_r_self, good_sky_red], ptitle=ptitle,
                          xmin=9000 * (redshift + 1), xmax=9130 * (redshift + 1), vlines=vlines, label=label,
                          axvspan=rango_plot)

    return good_sky_red


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def auto_scale_two_spectra(rss, sky_r_self, sky_r_star, scale=[0.1, 1.11, 0.025],
                           # w_scale_min = 6400,  w_scale_max = 7200,
                           w_scale_min="", w_scale_max="",
                           plot=True, verbose=True):
    """

    THIS NEEDS TO BE CHECKED TO BE SURE IT WORKS OK FOR CONTINUUM

    Parameters
    ----------
    rss : TYPE
        DESCRIPTION.
    sky_r_self : TYPE
        DESCRIPTION.
    sky_r_star : TYPE
        DESCRIPTION.
    scale : TYPE, optional
        DESCRIPTION. The default is [0.1,1.11,0.025].
    #w_scale_min : TYPE, optional
        DESCRIPTION. The default is 6400.
    w_scale_max : TYPE, optional
        DESCRIPTION. The default is 7200.
    w_scale_min : TYPE, optional
        DESCRIPTION. The default is "".
    w_scale_max : TYPE, optional
        DESCRIPTION. The default is "".
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if verbose: print("\n> Automatically searching for the scale factor between two spectra (object and sky)...")

    w = rss.wavelength
    if w_scale_min == "": w_scale_min = w[0]
    if w_scale_max == "": w_scale_max = w[-1]

    if len(scale) == 2: scale.append(0.025)

    steps = np.arange(scale[0], scale[1], scale[2])
    region = np.where((w > w_scale_min) & (w < w_scale_max))

    factor = []
    rsm = []

    for step in steps:
        sub = np.abs(sky_r_self - step * sky_r_star)
        factor.append(step)
        rsm.append(np.nansum(sub[region]))

    auto_scale = factor[rsm.index(np.nanmin(rsm))]
    factor_v = factor[rsm.index(np.nanmin(rsm)) - 5:rsm.index(np.nanmin(rsm)) + 5]
    rsm_v = rsm[rsm.index(np.nanmin(rsm)) - 5:rsm.index(np.nanmin(rsm)) + 5]

    if auto_scale == steps[0] or auto_scale == steps[-1]:
        if verbose:
            print("  No minimum found in the scaling interval {} - {} ...".format(scale[0], scale[1]))
            print("  NOTHING DONE ! ")
        return 1.
    else:
        fit = np.polyfit(factor_v, rsm_v, 2)
        yfit = np.poly1d(fit)
        vector = np.arange(scale[0], scale[1], 0.001)
        rsm_fitted = yfit(vector)
        auto_scale_fit = vector[rsm_fitted.tolist().index(np.nanmin(rsm_fitted))]

        if plot:
            ptitle = "Auto scale factor found (using fit) between OBJ and SKY = " + np.str(np.round(auto_scale_fit, 3))
            plot_plot([factor, vector], [rsm, rsm_fitted], vlines=[auto_scale, auto_scale_fit],
                      label=["Measured", "Fit"],
                      xlabel="Scale factor", ylabel="Absolute flux difference ( OBJ - SKY ) [counts]",
                      ymin=np.nanmin(rsm) - (np.nanmax(rsm) - np.nanmin(rsm)) / 10., ptitle=ptitle)

            sub = sky_r_self - auto_scale_fit * sky_r_star
            # plot_plot(w, sub ,ymax=np.percentile(sub,99.6), hlines=[0], ptitle="Sky sustracted")

            ptitle = "Sky substraction applying the automatic factor of " + np.str(
                np.round(auto_scale_fit, 3)) + " to the sky emission"
            plot_plot(w, [sky_r_self, auto_scale_fit * sky_r_star, sub], xmin=w_scale_min, xmax=w_scale_max,
                      color=["b", "r", "g"],
                      label=["obj", "sky", "obj-sky"], ymax=np.nanpercentile(sky_r_self[region], 98), hlines=[0],
                      # ymin=np.percentile(sky_r_self[region],0.01))
                      ymin=np.nanpercentile(sub[region] - 10, 0.01), ptitle=ptitle)

        if verbose:
            print("  Auto scale factor       = ", np.round(auto_scale, 3))
            print("  Auto scale factor (fit) = ", np.round(auto_scale_fit, 3))

        return np.round(auto_scale_fit, 3)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def remove_negative_pixels(spectra, verbose=True):
    """
    Makes sure the median value of all spectra is not negative. Typically these are the sky fibres.

    Parameters
    ----------
    spectra : TYPE
        DESCRIPTION.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    """
    cuenta = 0

    output = copy.deepcopy(spectra)
    for fibre in range(len(spectra)):
        vector_ = spectra[fibre]
        stats_ = basic_statistics(vector_, return_data=True, verbose=False)
        # rss.low_cut.append(stats_[1])
        if stats_[1] < 0.:
            cuenta = cuenta + 1
            vector_ = vector_ - stats_[1]
            output[fibre] = [0. if x < 0. else x for x in vector_]

    if verbose: print(
        "\n> Found {} spectra for which the median value is negative, they have been corrected".format(cuenta))
    return output


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def centroid_of_cube(cube, x0=0, x1=-1, y0=0, y1=-1, box_x=[], box_y=[],
                     step_tracing=100, g2d=True, adr_index_fit=2,
                     edgelow=-1, edgehigh=-1,
                     plot=True, log=True, gamma=0.,
                     plot_residua=True, plot_tracing_maps=[], verbose=True):
    """
    New Routine 20 Nov 2021 for using astropy photutils tasks for centroid



    """

    if plot == False: plot_residua = False

    if len(box_x) == 2:
        x0 = box_x[0]
        x1 = box_x[1]
    if len(box_y) == 2:
        y0 = box_y[0]
        y1 = box_y[1]

    if verbose:
        if np.nanmedian([x0, x1, y0, y1]) != -0.5:
            print(
                "\n> Computing the centroid of the cube in box [ {:.0f} , {:.0f} ] , [ {:.0f} , {:.0f} ] with the given parameters:".format(
                    x0, x1, y0, y1))
        else:
            print("\n> Computing the centroid of the cube using all spaxels with the given parameters:")
        if g2d:
            print("  step =", step_tracing, " , adr_index_fit =", adr_index_fit, " , using a 2D Gaussian fit")
        else:
            print("  step =", step_tracing, " , adr_index_fit =", adr_index_fit,
                  " , using the center of mass of the image")

    cube_trimmed = copy.deepcopy(cube)

    if np.nanmedian([x0, x1, y0, y1]) != -0.5:
        cube_trimmed.data = cube.data[:, y0:y1, x0:x1]
        trimmed = True
    else:
        trimmed = False

    w_vector = []
    wc_vector = []
    xc_vector = []
    yc_vector = []

    if edgelow == -1: edgelow = 0
    if edgehigh == -1: edgehigh = 0

    valid_wave_min_index = cube.valid_wave_min_index + edgelow
    valid_wave_max_index = cube.valid_wave_max_index - edgehigh

    for i in range(valid_wave_min_index, valid_wave_max_index + step_tracing, step_tracing):
        if i < len(cube.wavelength): w_vector.append(cube.wavelength[i])

    show_map = -1
    if len(plot_tracing_maps) > 0:  show_map = 0

    for i in range(len(w_vector) - 1):
        wc_vector.append((w_vector[i] + w_vector[i + 1]) / 2.)

        _map_ = cube_trimmed.create_map(line=w_vector[i], w2=w_vector[i + 1], verbose=False)

        # Searching for centroid
        if g2d:
            xc, yc = centroid_2dg(_map_[1])
            ptitle = "Fit of order " + np.str(
                adr_index_fit) + " to centroids computed using a 2D Gaussian fit in steps of " + np.str(
                step_tracing) + " $\mathrm{\AA}$"
        else:
            xc, yc = centroid_com(_map_[1])
            ptitle = "Fit of order " + np.str(
                adr_index_fit) + " to centroids computed using the center of mass in steps of " + np.str(
                step_tracing) + " $\mathrm{\AA}$"

        if show_map > -1 and plot:
            if w_vector[i] < plot_tracing_maps[show_map] and w_vector[i + 1] > plot_tracing_maps[show_map]:  # show map
                # print(xc,yc)
                description = "Centroid for " + np.str(plot_tracing_maps[show_map]) + " $\mathrm{\AA}$"
                cube_trimmed.plot_map(_map_, description=description,
                                      # plot_spaxel_list = [[0.,0.],[1.,1.],[2.,2.],[xc,yc]],
                                      plot_spaxel_list=[[xc, yc]], log=log, gamma=gamma,
                                      g2d=g2d,
                                      verbose=False, trimmed=trimmed)  # FORO
                if verbose: print(
                    '  Centroid at {} A found in spaxel [ {:.2f} , {:.2f} ]  =  [ {:.2f}" , {:.2f}" ]'.format(
                        plot_tracing_maps[show_map], xc, yc, xc * cube.pixel_size_arcsec, yc * cube.pixel_size_arcsec))
                show_map = show_map + 1
                if show_map == len(plot_tracing_maps): show_map = -1

        xc_vector.append(xc)
        yc_vector.append(yc)

    x_peaks_fit = np.polyfit(wc_vector, xc_vector, adr_index_fit)
    pp = np.poly1d(x_peaks_fit)
    x_peaks = pp(cube.wavelength) + x0

    y_peaks_fit = np.polyfit(wc_vector, yc_vector, adr_index_fit)
    pp = np.poly1d(y_peaks_fit)
    y_peaks = pp(cube.wavelength) + y0

    xc_vector = (xc_vector - np.nanmedian(xc_vector)) * cube.pixel_size_arcsec
    yc_vector = (yc_vector - np.nanmedian(yc_vector)) * cube.pixel_size_arcsec

    ADR_x_fit = np.polyfit(wc_vector, xc_vector, adr_index_fit)
    pp = np.poly1d(ADR_x_fit)
    fx = pp(wc_vector)

    ADR_y_fit = np.polyfit(wc_vector, yc_vector, adr_index_fit)
    pp = np.poly1d(ADR_y_fit)
    fy = pp(wc_vector)

    vlines = [cube.wavelength[valid_wave_min_index], cube.wavelength[valid_wave_max_index]]
    if plot:
        plot_plot(wc_vector, [xc_vector, yc_vector, fx, fy], psym=["+", "o", "", ""], color=["r", "k", "g", "b"],
                  alpha=[1, 1, 1, 1], label=["RA", "Dec", "RA fit", "Dec fit"],
                  xmin=cube.wavelength[0], xmax=cube.wavelength[-1], vlines=vlines, markersize=[10, 7],
                  ylabel="$\Delta$ offset [arcsec]", ptitle=ptitle, hlines=[0], frameon=True,
                  ymin=np.nanmin([np.nanmin(xc_vector), np.nanmin(yc_vector)]),
                  ymax=np.nanmax([np.nanmax(xc_vector), np.nanmax(yc_vector)]))

    # ADR_x_max=np.nanmax(xc_vector)-np.nanmin(xc_vector)    ##### USE FITS INSTEAD OF VECTOR FOR REMOVING OUTLIERS
    # ADR_y_max=np.nanmax(yc_vector)-np.nanmin(yc_vector)

    ADR_x_max = np.nanmax(fx) - np.nanmin(fx)  ##### USING FITS
    ADR_y_max = np.nanmax(fy) - np.nanmin(fy)

    ADR_total = np.sqrt(ADR_x_max ** 2 + ADR_y_max ** 2)

    stat_x = basic_statistics(xc_vector - fx, verbose=False, return_data=True)
    stat_y = basic_statistics(yc_vector - fy, verbose=False, return_data=True)
    stat_total = np.sqrt(stat_x[3] ** 2 + stat_y[3] ** 2)

    if verbose: print(
        '  ADR variation in valid interval using fit : RA = {:.3f}" , Dec = {:.3f}" , total = {:.3f}"  that is {:.0f}% of a spaxel'.format(
            ADR_x_max, ADR_y_max, ADR_total, ADR_total * 100. / cube.pixel_size_arcsec))

    if plot_residua:
        plot_plot(wc_vector, [xc_vector - fx, yc_vector - fy], color=["r", "k"], alpha=[1, 1], ymin=-0.1, ymax=0.1,
                  hlines=[-0.08, -0.06, -0.04, -0.02, 0, 0, 0, 0, 0.02, 0.04, 0.06, 0.08],
                  xmin=cube.wavelength[0], xmax=cube.wavelength[-1], frameon=True, label=["RA residua", "Dec residua"],
                  vlines=vlines, ylabel="$\Delta$ offset [arcsec]", ptitle="Residua of the fit to the centroid fit")

    if verbose: print(
        '  Standard deviation of residua :             RA = {:.3f}" , Dec = {:.3f}" , total = {:.3f}"  that is {:.0f}% of a spaxel'.format(
            stat_x[3], stat_y[3], stat_total, stat_total * 100. / cube.pixel_size_arcsec))

    return ADR_x_fit, ADR_y_fit, ADR_x_max, ADR_y_max, ADR_total, x_peaks, y_peaks


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def scale_cubes_using_common_region(cube_list, flux_ratios=[], min_wave=0, max_wave=0,
                                    apply_scale=True, verbose=True, plot=False):  # SCORE

    if verbose: print("\n> Scaling intensities of the cubes using the integrated value of their common region...")
    # Check if cube_list are fits or objects
    object_name_list = []
    try:
        try_read = cube_list[0] + "  "  # This will work ONLY if cube_list are strings (fits names)
        if verbose: print("  - Reading the cubes from the list of fits files provided:" + try_read[-2:-1])
        object_list = []
        for i in range(len(cube_list)):
            if i < 9:
                name = "cube_0" + np.str(i + 1)
            else:
                name = "cube_" + np.str(i + 1)
            object_name_list.append(name)
            exec(name + "=read_cube(cube_list[i])")
            exec("object_list.append(" + name + ")")
        print(" ")
    except Exception:
        object_list = cube_list
        for i in range(len(cube_list)):
            object_name_list.append(cube_list[i].object)

    if len(flux_ratios) == 0:  # flux_ratios are not given
        if verbose:
            if np.nanmedian(object_list[0].data) < 1E-6:
                print("  - Cubes are flux calibrated. Creating mask for each cube...")
            else:
                print(
                    "  - Cubes are NOT flux calibrated. Creating mask and scaling with the cube with the largest integrated value...")

        # Create a mask for each cube
        list_of_masks = []
        for i in range(len(cube_list)):
            object_list[i].mask_cube(min_wave=min_wave, max_wave=max_wave)
            list_of_masks.append(object_list[i].mask)

        # Get mask combining all mask
        mask = np.median(list_of_masks, axis=0)
        if plot:
            object_list[0].plot_map(mapa=mask, fcal=False, cmap="binary", description="mask",
                                    barlabel=" ", vmin=0., vmax=1., contours=False)

        # Compute integrated flux within the good values in all cubes
        integrated_flux_region = []
        for i in range(len(cube_list)):
            im_mask = object_list[i].integrated_map * mask
            ifr_ = np.nansum(im_mask)
            integrated_flux_region.append(ifr_)

        # If data are NOT flux calibrated, it does not matter the exposition time, just scale
        # Find the maximum value and scale!
        max_irf = np.nanmax(integrated_flux_region)
        flux_ratios = integrated_flux_region / max_irf

        if verbose:
            print("  - Cube  Name                               Total valid integrated flux      Flux ratio")
            for i in range(len(cube_list)):
                print("    {:2}   {:30}            {:.4}                  {:.4}".format(i + 1, object_name_list[i],
                                                                                        integrated_flux_region[i],
                                                                                        flux_ratios[i]))
    else:
        if verbose:
            print("  - Scale values provided !")
            print("  - Cube  Name                             Flux ratio provided")
            for i in range(len(cube_list)):
                print("    {:2}   {:30}                {:.4}".format(i + 1, object_name_list[i], flux_ratios[i]))

    if apply_scale:
        if verbose: print("  - Applying flux ratios to cubes...")
        for i in range(len(cube_list)):
            object_list[i].scale_flux = flux_ratios[i]
            _cube_ = object_list[i]
            _data_ = _cube_.data / flux_ratios[i]
            object_list[i].data = _data_
            _cube_.integrated_map = np.sum(
                _cube_.data[np.searchsorted(_cube_.wavelength, min_wave):np.searchsorted(_cube_.wavelength, max_wave)],
                axis=0)

    return object_list


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def build_combined_cube(cube_list, obj_name="", description="", fits_file="", path="",
                        scale_cubes_using_integflux=True, flux_ratios=[], apply_scale=True,
                        edgelow=30, edgehigh=30,
                        ADR=True, ADR_cc=False, jump=-1, pk="",
                        ADR_x_fit_list=[], ADR_y_fit_list=[], force_ADR=False,
                        half_size_for_centroid=10, box_x=[0, -1], box_y=[0, -1],
                        adr_index_fit=2, g2d=False, step_tracing=100, plot_tracing_maps=[],
                        trim_cube=True, trim_values=[], remove_spaxels_not_fully_covered=True,
                        plot=True, plot_weight=True, plot_spectra=True,
                        verbose=True, say_making_combined_cube=True):
    if say_making_combined_cube: print("\n> Making combined cube ...")
    n_files = len(cube_list)
    # Check if cube_list are fits or objects
    object_name_list = []
    try:
        try_read = cube_list[0] + "  "
        if verbose: print(" - Reading the cubes from the list of fits files provided:" + try_read[-2:-1])
        object_list = []
        for i in range(n_files):
            if i < 9:
                name = "cube_0" + np.str(i + 1)
            else:
                name = "cube_" + np.str(i + 1)
            object_name_list.append(name)
            exec(name + "=read_cube(cube_list[i])")
            exec("object_list.append(" + name + ")")
            print(" ")
    except Exception:
        object_list = cube_list
        for i in range(n_files):
            object_name_list.append(cube_list[i].object)

    cube_aligned_object = object_list
    print("\n> Checking individual cubes: ")
    print(
        "  Cube      name                          RA_centre           DEC_centre     Pix Size  Kernel Size   n_cols  n_rows")
    for i in range(n_files):
        print("    {:2}    {:25}   {:18.12f} {:18.12f}    {:4.1f}      {:5.2f}       {:4}    {:4}".format(i + 1,
                                                                                                          cube_aligned_object[
                                                                                                              i].object,
                                                                                                          cube_aligned_object[
                                                                                                              i].RA_centre_deg,
                                                                                                          cube_aligned_object[
                                                                                                              i].DEC_centre_deg,
                                                                                                          cube_aligned_object[
                                                                                                              i].pixel_size_arcsec,
                                                                                                          cube_aligned_object[
                                                                                                              i].kernel_size_arcsec,
                                                                                                          cube_aligned_object[
                                                                                                              i].n_cols,
                                                                                                          cube_aligned_object[
                                                                                                              i].n_rows))

    # Check that RA_centre, DEC_centre, pix_size and kernel_size are THE SAME in all input cubes
    do_not_combine = False
    for _property_ in ["RA_centre_deg", "DEC_centre_deg", "pixel_size_arcsec", "kernel_size_arcsec", "n_cols",
                       "n_rows"]:
        property_values = [_property_]
        for i in range(n_files):
            exec("property_values.append(cube_aligned_object[" + np.str(i) + "]." + _property_ + ")")
        # print(property_values)
        if np.nanvar(property_values[1:-1]) != 0.:
            print(" - Property {} has DIFFERENT values !!!".format(_property_))
            do_not_combine = True

    if do_not_combine:
        print("\n> Cubes CANNOT be combined as they don't have the same basic properties !!!")
    else:
        print("\n> Cubes CAN be combined as they DO have the same basic properties !!!")

        if pk == "":
            pixel_size_arcsec = cube_aligned_object[0].pixel_size_arcsec
            kernel_size_arcsec = cube_aligned_object[0].kernel_size_arcsec
            pk = "_" + str(int(pixel_size_arcsec)) + "p" + str(
                int((abs(pixel_size_arcsec) - abs(int(pixel_size_arcsec))) * 10)) + "_" + str(
                int(kernel_size_arcsec)) + "k" + str(int(abs(kernel_size_arcsec * 100)) - int(kernel_size_arcsec) * 100)

        # Create a cube with zero - In the past we run Interpolated_cube,
        # now we use the first cube as a template
        # shape = [self.cube1_aligned.data.shape[1], self.cube1_aligned.data.shape[2]]
        # self.combined_cube = Interpolated_cube(self.rss1, self.cube1_aligned.pixel_size_arcsec, self.cube1_aligned.kernel_size_arcsec, zeros=True, shape=shape, offsets_files =self.cube1_aligned.offsets_files)

        combined_cube = copy.deepcopy(cube_aligned_object[0])
        combined_cube.data = np.zeros_like(combined_cube.data)

        combined_cube.ADR_total = 0.
        combined_cube.ADR_x_max = 0.
        combined_cube.ADR_y_max = 0.
        combined_cube.ADR_x = np.zeros_like(combined_cube.wavelength)
        combined_cube.ADR_y = np.zeros_like(combined_cube.wavelength)
        combined_cube.ADR_x_fit = []
        combined_cube.ADR_y_fit = []
        combined_cube.history = []

        combined_cube.number_of_combined_files = n_files

        # delattr(combined_cube, "ADR_total")

        if obj_name != "":
            combined_cube.object = obj_name
        if description == "":
            combined_cube.description = combined_cube.object + " - COMBINED CUBE"
        else:
            combined_cube.description = description

        # delattr(object, property) ### If we need to delete a property in object

        print("\n> Combining cubes...")

        # Checking ADR
        if combined_cube.adrcor:
            print("  - Using data cubes corrected for ADR to get combined cube")
            combined_cube.history.append("- Using data cubes corrected for ADR to get combined cube")
        else:
            print("  - Using data cubes NOT corrected for ADR to get combined cube")
            combined_cube.history.append("- Using data cubes NOT corrected for ADR to get combined cube")
        combined_cube.adrcor = False

        # Include flux calibration, assuming it is the same to all cubes
        # (it needs to be updated to combine data taken in different nights)
        if np.nanmedian(cube_aligned_object[0].flux_calibration) == 0:
            print("  - Flux calibration not considered")
            combined_cube.history.append("- Flux calibration not considered")
            fcal = False
        else:
            combined_cube.flux_calibration = cube_aligned_object[0].flux_calibration
            print("  - Flux calibration included!")
            combined_cube.history.append("- Flux calibration included")
            fcal = True

        if scale_cubes_using_integflux:
            cube_list = scale_cubes_using_common_region(cube_list, flux_ratios=flux_ratios)
        else:
            print("  - No scaling of the cubes using integrated flux requested")
            combined_cube.history.append("- No scaling of the cubes using integrated flux requested")

        _data_ = []
        _PA_ = []
        _weight_ = []

        print("\n> Combining data cubes...")
        for i in range(n_files):
            _data_.append(cube_aligned_object[i].data)
            _PA_.append(cube_aligned_object[i].PA)
            _weight_.append(cube_aligned_object[i].weight)

        combined_cube.data = np.nanmedian(_data_, axis=0)
        combined_cube.PA = np.mean(_PA_)
        combined_cube.weight = np.nanmean(_weight_, axis=0)
        combined_cube.offsets_files_position = 0

        # # Plot combined weight if requested
        if plot: combined_cube.plot_weight()

        # # Check this when using files taken on different nights  --> Data in self.combined_cube
        # #self.wavelength=self.rss1.wavelength
        # #self.valid_wave_min=self.rss1.valid_wave_min
        # #self.valid_wave_max=self.rss1.valid_wave_max

        if ADR:
            # Get integrated map for finding max_x and max_y
            combined_cube.get_integrated_map()
            # Trace peaks of combined cube
            box_x_centroid = box_x
            box_y_centroid = box_y
            if np.nanmedian(box_x + box_y) == -0.5 and half_size_for_centroid > 0:
                combined_cube.get_integrated_map()
                # if verbose: print("\n> Peak of emission found in [ {} , {} ]".format(combined_cube.max_x,combined_cube.max_y))
                if verbose: print(
                    "\n> As requested, using a box centered at the peak of emission, [ {} , {} ], and width +-{} spaxels for tracing...".format(
                        combined_cube.max_x, combined_cube.max_y, half_size_for_centroid))

                # if verbose: print("  As requested, using a box centered there and width +-{} spaxels for tracing...".format(half_size_for_centroid))
                box_x_centroid = [combined_cube.max_x - half_size_for_centroid,
                                  combined_cube.max_x + half_size_for_centroid]
                box_y_centroid = [combined_cube.max_y - half_size_for_centroid,
                                  combined_cube.max_y + half_size_for_centroid]

            if ADR_cc:
                check_ADR = False
            else:
                check_ADR = True

                combined_cube.trace_peak(box_x=box_x_centroid, box_y=box_y_centroid, edgelow=edgelow, edgehigh=edgehigh,
                                         plot=plot, adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing,
                                         plot_tracing_maps=plot_tracing_maps, check_ADR=check_ADR)

        # ADR correction to the combined cube
        if ADR_cc:
            combined_cube.adrcor = True
            combined_cube.ADR_correction(RSS, plot=plot, jump=jump, method="old", force_ADR=force_ADR,
                                         remove_spaxels_not_fully_covered=remove_spaxels_not_fully_covered)
            if ADR:
                combined_cube.trace_peak(box_x=box_x_centroid, box_y=box_y_centroid,
                                         edgelow=edgelow, edgehigh=edgehigh,
                                         plot=plot, check_ADR=True, step_tracing=step_tracing,
                                         plot_tracing_maps=plot_tracing_maps,
                                         adr_index_fit=adr_index_fit, g2d=g2d)

        combined_cube.get_integrated_map(box_x=box_x, box_y=box_y, fcal=fcal, plot=plot, plot_spectra=plot_spectra,
                                         plot_centroid=True, g2d=g2d)

        # Trimming combined cube if requested or needed
        combined_cube.trim_cube(trim_cube=trim_cube, trim_values=trim_values,
                                half_size_for_centroid=half_size_for_centroid, ADR=ADR,
                                adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing,
                                plot_tracing_maps=plot_tracing_maps,
                                remove_spaxels_not_fully_covered=remove_spaxels_not_fully_covered,
                                box_x=box_x, box_y=box_y, edgelow=edgelow, edgehigh=edgehigh,
                                plot_weight=plot_weight, fcal=fcal, plot=plot, plot_spectra=plot_spectra)

        # Computing total exposition time of combined cube
        combined_cube.total_exptime = 0.
        combined_cube.exptimes = []
        combined_cube.rss_list = []
        for i in range(n_files):
            combined_cube.total_exptime = combined_cube.total_exptime + cube_aligned_object[i].total_exptime
            combined_cube.exptimes.append(cube_aligned_object[i].total_exptime)
            combined_cube.rss_list.append(cube_aligned_object[i].rss_list)

        print("\n> Total exposition time = ", combined_cube.total_exptime, "seconds adding the", n_files, "files")

        if combined_cube.total_exptime / n_files == combined_cube.exptimes[0]:
            print("  All {} cubes have the same exposition time, {} s".format(n_files, combined_cube.exptimes[0]))
        else:
            print("  The individual cubes have different exposition times.")

        if np.nanmedian(combined_cube.offsets_files) != 0:
            offsets_print = "[ "
            for i in range(len(combined_cube.offsets_files)):
                offsets_print = offsets_print + np.str(combined_cube.offsets_files[i][0]) + " , " + np.str(
                    combined_cube.offsets_files[i][1]) + " , "
            offsets_print = offsets_print[:-2] + "]"
            print("\n  offsets = ", offsets_print)

        if len(ADR_x_fit_list) > 0:
            print("\n  ADR_x_fit_list = ", ADR_x_fit_list)
            print("\n  ADR_y_fit_list = ", ADR_y_fit_list)

        if fits_file == "":
            print("\n> As requested, the combined cube will not be saved to a fits file")
        else:
            # print("\n> Saving combined cube to a fits file ...")
            if fits_file == "auto":
                fits_file = path + obj_name + "_" + combined_cube.grating + pk + "_combining_" + np.str(
                    n_files) + "_cubes.fits"
            save_cube_to_fits_file(combined_cube, fits_file, path=path, description=description, obj_name=obj_name)

        # if obj_name != "":
        #     print("\n> Saving combined cube into Python object:", obj_name)
        #     exec(obj_name+"=combined_cube", globals())
        # else:
        #     print("\n> No name fot the Python object with the combined cube provided, returning combined_cube")
        return combined_cube


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def name_keys(filename, apply_throughput=False, correct_ccd_defects=False,
              fix_wavelengths=False, do_extinction=False, sky_method="none",
              do_telluric_correction=False, id_el=False,
              correct_negative_sky=False, clean_residuals=False):
    """
    Task for automatically naming output rss files.
    """
    if apply_throughput:
        clave = "__________"
    else:
        clave = filename[-15:-5]

    if apply_throughput:
        T = "T"  # T = throughput
    else:
        T = clave[-9]
    if correct_ccd_defects:
        C = "C"  # C = corrected CCD defects
    else:
        C = clave[-8]
    if fix_wavelengths:
        W = "W"  # W = Wavelength tweak
    else:
        W = clave[-7]
    if do_extinction:
        X = "X"  # X = extinction corrected
    else:
        X = clave[-6]
    if do_telluric_correction:
        U = "U"  # U = Telluric corrected
    else:
        U = clave[-5]
    if sky_method != "none":
        S = "S"  # S = Sky substracted
    else:
        S = clave[-4]
    if id_el:
        E = "E"  # E = Emission lines identified
    else:
        E = clave[-3]
    if correct_negative_sky:
        N = "N"  # N = Negative values
    else:
        N = clave[-2]
    if clean_residuals:
        R = "R"  # R = Sky and CCD residuals
    else:
        R = clave[-1]

    clave = "_" + T + C + W + X + U + S + E + N + R

    if apply_throughput:
        return filename[0:-5] + clave + ".fits"
    else:
        return filename[0:-15] + clave + ".fits"


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def full_path(filename, path, verbose=False):
    """
    Check if string variable filename has the full path to that file. If it hasn't it adds the path.

    Parameters
    ----------
    filename : string
        Name of a file.
    path : string
        Full path to file filename
    verbose : Boolean

    Returns
    -------
    fullpath : string
        The file with the full path
    """
    if path[-1] != "/": path = path + "/"  # If path does not end in "/" it is added

    if len(filename.replace("/", "")) == len(filename):
        if verbose: print("\n> Variable {} does not include the full path {}".format(filename, path))
        fullpath = path + filename
    else:
        if verbose: print("\n> Variable {} includes the full path {}".format(filename, path))
        fullpath = filename
    return fullpath


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def list_fits_files_in_folder(path, verbose=True, use2=True, use3=False, ignore_offsets=True,
                              skyflat_names=[], ignore_list=[], return_list=False):
    nothing = 0  # Stupid thing for controling Exceptions
    list_of_objetos = []
    list_of_files = []
    list_of_exptimes = []
    if len(skyflat_names) == 0:
        skyflat_names = ["skyflat", "SKYFLAT", "SkyFlat"]

    if len(ignore_list) == 0:
        ignore_list = ["a", "b", "c", "d", "e", "f", "p", "pos", "Pos",
                       "A", "B", "C", "D", "E", "F", "P", "POS",
                       "p1", "p2", "p3", "p4", "p5", "p6",
                       "P1", "P2", "P3", "P4", "P5", "P6",
                       "pos1", "pos2", "pos3", "pos4", "pos5", "pos6",
                       "Pos1", "Pos2", "Pos3", "Pos4", "Pos5", "Pos6",
                       "POS1", "POS2", "POS3", "POS4", "POS5", "POS6"]

    if verbose: print("\n> Listing fits files in folder", path, ":\n")

    if path[-1] != "/": path = path + "/"

    for fitsName in sorted(glob.glob(path + '*.fits')):
        check_file = True
        if fitsName[-8:] != "red.fits":
            check_file = False
        if fitsName[0:8] == "combined" and check_file == False:
            check_file = True
        for skyflat_name in skyflat_names:
            if skyflat_name in fitsName: check_file = True

        hdulist = pyfits.open(fitsName)

        object_fits = hdulist[0].header['OBJECT'].split(" ")
        if object_fits[0] in ["HD", "NGC", "IC"] or use2:
            try:
                if ignore_offsets == False:
                    object_fits[0] = object_fits[0] + object_fits[1]
                elif object_fits[1] not in ignore_list:
                    object_fits[0] = object_fits[0] + object_fits[1]
            except Exception:
                nothing = 0
        if use3:
            try:
                if ignore_offsets == False:
                    object_fits[0] = object_fits[0] + object_fits[2]
                elif object_fits[2] not in ignore_list:
                    object_fits[0] = object_fits[0] + object_fits[2]
            except Exception:
                nothing = 0

        try:
            exptime = hdulist[0].header['EXPOSED']
        except Exception:
            check_file = False

        grating = hdulist[0].header['GRATID']
        try:
            date_ = hdulist[0].header['UTDATE']
        except Exception:
            check_file = False
        hdulist.close()

        if check_file:
            found = False
            for i in range(len(list_of_objetos)):
                if list_of_objetos[i] == object_fits[0]:
                    found = True
                    list_of_files[i].append(fitsName)
                    list_of_exptimes[i].append(exptime)
            if found == False:
                list_of_objetos.append(object_fits[0])
                list_of_files.append([fitsName])
                list_of_exptimes.append([exptime])

    date = date_[0:4] + date_[5:7] + date_[8:10]

    if verbose:
        for i in range(len(list_of_objetos)):
            for j in range(len(list_of_files[i])):
                if j == 0:
                    print("  {:15s}  {}          {:.1f} s".format(list_of_objetos[i], list_of_files[i][0],
                                                                  list_of_exptimes[i][0]))
                else:
                    print("                   {}          {:.1f} s".format(list_of_files[i][j], list_of_exptimes[i][j]))

        print("\n  They were obtained on {} using the grating {}".format(date, grating))

    if return_list: return list_of_objetos, list_of_files, list_of_exptimes, date, grating
    if nothing > 10: print(nothing)  # Stupid thing for controling Exceptions


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def save_map(cube, mapa, fits_file, mask=[], description="", path="", verbose=True):
    if path != "": fits_file = full_path(fits_file, path)

    if description == "": description = mapa[0]

    fits_image_hdu = fits.PrimaryHDU(mapa[1])

    fits_image_hdu.header['HISTORY'] = 'Map created by PyKOALA'
    fits_image_hdu.header['HISTORY'] = 'Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany,'
    fits_image_hdu.header['HISTORY'] = 'Blake Staples, Taylah Beard, Matt Owers, James Tocknell et al.'
    fits_image_hdu.header['HISTORY'] = version
    now = datetime.datetime.now()
    fits_image_hdu.header['HISTORY'] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    fits_image_hdu.header['DATE'] = now.strftime(
        "%Y-%m-%dT%H:%M:%S")  # '2002-09-16T18:52:44'   # /Date of FITS file creation

    fits_image_hdu.header['BITPIX'] = 16

    fits_image_hdu.header["ORIGIN"] = 'AAO'  # / Originating Institution
    fits_image_hdu.header["TELESCOP"] = 'Anglo-Australian Telescope'  # / Telescope Name
    fits_image_hdu.header["ALT_OBS"] = 1164  # / Altitude of observatory in metres
    fits_image_hdu.header["LAT_OBS"] = -31.27704  # / Observatory latitude in degrees
    fits_image_hdu.header["LONG_OBS"] = 149.0661  # / Observatory longitude in degrees

    fits_image_hdu.header["INSTRUME"] = "AAOMEGA-KOALA"  # / Instrument in use
    fits_image_hdu.header["GRATID"] = cube.grating  # / Disperser ID
    if cube.grating in red_gratings: SPECTID = "RD"
    if cube.grating in blue_gratings: SPECTID = "BL"
    fits_image_hdu.header["SPECTID"] = SPECTID  # / Spectrograph ID
    fits_image_hdu.header["DICHROIC"] = 'X5700'  # / Dichroic name   ---> CHANGE if using X6700!!

    fits_image_hdu.header['OBJECT'] = cube.object
    fits_image_hdu.header['TOTALEXP'] = cube.total_exptime
    fits_image_hdu.header['EXPTIMES'] = np.str(cube.exptimes)

    fits_image_hdu.header['NAXIS'] = 2  # / number of array dimensions
    fits_image_hdu.header['NAXIS1'] = cube.data.shape[1]  ##### CHECK !!!!!!!
    fits_image_hdu.header['NAXIS2'] = cube.data.shape[2]

    # WCS
    fits_image_hdu.header["RADECSYS"] = 'FK5'  # / FK5 reference system
    fits_image_hdu.header["EQUINOX"] = 2000  # / [yr] Equinox of equatorial coordinates
    fits_image_hdu.header["WCSAXES"] = 2  # / Number of coordinate axes

    fits_image_hdu.header['CRPIX1'] = cube.data.shape[1] / 2.  # / Pixel coordinate of reference point
    fits_image_hdu.header['CDELT1'] = -cube.pixel_size_arcsec / 3600.  # / Coordinate increment at reference point
    fits_image_hdu.header[
        'CTYPE1'] = "RA--TAN"  # 'DEGREE'                               # / Coordinate type code
    fits_image_hdu.header['CRVAL1'] = cube.RA_centre_deg  # / Coordinate value at reference point

    fits_image_hdu.header['CRPIX2'] = cube.data.shape[2] / 2.  # / Pixel coordinate of reference point
    fits_image_hdu.header['CDELT2'] = cube.pixel_size_arcsec / 3600.  # Coordinate increment at reference point
    fits_image_hdu.header[
        'CTYPE2'] = "DEC--TAN"  # 'DEGREE'                               # / Coordinate type code
    fits_image_hdu.header['CRVAL2'] = cube.DEC_centre_deg  # / Coordinate value at reference point

    fits_image_hdu.header['RAcen'] = cube.RA_centre_deg
    fits_image_hdu.header['DECcen'] = cube.DEC_centre_deg
    fits_image_hdu.header['PIXsize'] = cube.pixel_size_arcsec
    fits_image_hdu.header['KERsize'] = cube.kernel_size_arcsec
    fits_image_hdu.header['Ncols'] = cube.data.shape[2]
    fits_image_hdu.header['Nrows'] = cube.data.shape[1]
    fits_image_hdu.header['PA'] = cube.PA
    fits_image_hdu.header['DESCRIP'] = description

    if len(mask) > 0:
        try:
            for i in range(len(mask)):
                mask_name1 = "MASK" + np.str(i + 1) + "1"
                mask_name2 = "MASK" + np.str(i + 1) + "2"
                fits_image_hdu.header[mask_name1] = mask[i][1]
                fits_image_hdu.header[mask_name2] = mask[i][2]
        except Exception:
            fits_image_hdu.header['MASK11'] = mask[1]
            fits_image_hdu.header['MASK12'] = mask[2]

    fits_image_hdu.header['HISTORY'] = 'Extension[1] is the integrated map'

    try:
        fits_velocity = fits.ImageHDU(mapa[2])
        fits_fwhm = fits.ImageHDU(mapa[3])
        fits_ew = fits.ImageHDU(mapa[4])
        hdu_list = fits.HDUList([fits_image_hdu, fits_velocity, fits_fwhm, fits_ew])  # , fits_mask])
        fits_image_hdu.header['HISTORY'] = 'This was obtained doing a Gassian fit'
        fits_image_hdu.header['HISTORY'] = 'Extension[2] is the velocity map [km/s]'
        fits_image_hdu.header['HISTORY'] = 'Extension[3] is the FWHM map [km/s]'
        fits_image_hdu.header['HISTORY'] = 'Extension[4] is the EW map [A]'

    except Exception:
        hdu_list = fits.HDUList([fits_image_hdu])  # , fits_mask])

    hdu_list.writeto(fits_file, overwrite=True)
    if verbose:
        print("\n> Map saved to file:")
        print(" ", fits_file)
        print("  Description:", description)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def load_map(mapa_fits, description="", path="", verbose=True):
    if verbose: print("\n> Reading map(s) stored in file", mapa_fits, "...")

    if path != "": mapa_fits = full_path(mapa_fits, path)
    mapa_fits_data = fits.open(mapa_fits)  # Open file

    if description == "": description = mapa_fits_data[0].header['DESCRIP']  #
    if verbose: print("- Description stored in [0]")

    intensity_map = mapa_fits_data[0].data

    try:
        vel_map = mapa_fits_data[1].data
        fwhm_map = mapa_fits_data[2].data
        ew_map = mapa_fits_data[3].data
        mapa = [description, intensity_map, vel_map, fwhm_map, ew_map]
        if verbose:
            print("  This map comes from a Gaussian fit: ")
            print("- Intensity map stored in [1]")
            print("- Radial velocity map [km/s] stored in [2]")
            print("- FWHM map [km/s] stored in [3]")
            print("- EW map [A] stored in [4]")

    except Exception:
        if verbose: print("- Map stored in [1]")
        mapa = [description, intensity_map]

    fail = 0
    try:
        for i in range(4):
            try:
                mask_name1 = "MASK" + np.str(i + 1) + "1"
                mask_name2 = "MASK" + np.str(i + 1) + "2"
                mask_low_limit = mapa_fits_data[0].header[mask_name1]
                mask_high_limit = mapa_fits_data[0].header[mask_name2]
                _mask_ = create_mask(mapa_fits_data[i].data, low_limit=mask_low_limit, high_limit=mask_high_limit,
                                     verbose=False)
                mapa.append(_mask_)
                if verbose: print(
                    "- Mask with good values between {} and {} created and stored in [{}]".format(mask_low_limit,
                                                                                                  mask_high_limit,
                                                                                                  len(mapa) - 1))
            except Exception:
                fail = fail + 1

    except Exception:
        if verbose: print("- Map does not have any mask.")

    return mapa


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def automatic_calibration_night(CALIBRATION_NIGHT_FILE="",
                                date="",
                                grating="",
                                pixel_size=0,
                                kernel_size=0,
                                path="",
                                file_skyflat="",
                                throughput_2D_file="",
                                throughput_2D=0,
                                skyflat=0,
                                do_skyflat=True,
                                kernel_throughput=0,
                                correct_ccd_defects=True,
                                fix_wavelengths=False,
                                sol=[0],
                                rss_star_file_for_sol="",
                                plot=True,
                                CONFIG_FILE_path="",
                                CONFIG_FILE_list=[],
                                star_list=[],
                                abs_flux_scale=[],
                                flux_calibration_file="",
                                telluric_correction_file="",
                                objects_auto=[],
                                auto=False,
                                rss_clean=False,
                                flux_calibration_name="flux_calibration_auto",
                                cal_from_calibrated_starcubes=False,
                                disable_stars=[],  # stars in this list will not be used
                                skyflat_names=[]
                                ):
    """
    Use:
        CALIBRATION_NIGHT_FILE = "./CONFIG_FILES/calibration_night.config"
        automatic_calibration_night(CALIBRATION_NIGHT_FILE)
    """

    if len(skyflat_names) == 0: skyflat_names = ["SKYFLAT", "skyflat", "Skyflat", "SkyFlat", "SKYFlat", "SkyFLAT"]

    w = []
    telluric_correction_list = []
    global skyflat_variable
    skyflat_variable = ""
    global skyflat_
    global throughput_2D_variable
    global flux_calibration_night
    global telluric_correction_night
    throughput_2D_variable = ""
    global throughput_2D_
    throughput_2D_ = [0]

    if flux_calibration_file == "": flux_calibration_file = path + "flux_calibration_file_auto.dat"
    if telluric_correction_file == "": telluric_correction_file = path + "telluric_correction_file_auto.dat"

    check_nothing_done = 0

    print("\n===================================================================================")

    if auto:
        print("\n    COMPLETELY AUTOMATIC CALIBRATION OF THE NIGHT ")
        print("\n===================================================================================")

    if len(CALIBRATION_NIGHT_FILE) > 0:
        config_property, config_value = read_table(CALIBRATION_NIGHT_FILE, ["s", "s"])
        print("\n> Reading configuration file ", CALIBRATION_NIGHT_FILE)
        print("  for performing the automatic calibration of the night...\n")
    else:
        print("\n> Using the values given in automatic_calibration_night()")
        print("  for performing the automatic calibration of the night...\n")
        config_property = []
        config_value = []
        lista_propiedades = ["path", "file_skyflat", "rss_star_file_for_sol", "flux_calibration_file",
                             "telluric_correction_file"]
        lista_valores = [path, file_skyflat, rss_star_file_for_sol, flux_calibration_file, telluric_correction_file]
        for i in range(len(lista_propiedades)):
            if len(lista_valores[i]) > 0:
                config_property.append(lista_propiedades[i])
                config_value.append(lista_valores[i])
        if pixel_size == 0:
            print("  - No pixel size provided, considering pixel_size = 0.7")
            pixel_size = 0.7
        if kernel_size == 0:
            print("  - No kernel size provided, considering kernel_size = 1.1")
            kernel_size = 1.1
        pk = "_" + str(int(pixel_size)) + "p" + str(int((abs(pixel_size) - abs(int(pixel_size))) * 10)) + "_" + str(
            int(kernel_size)) + "k" + str(int(abs(kernel_size * 100)) - int(kernel_size) * 100)
        if sol[0] != 0: fix_wavelengths = True
        if len(CONFIG_FILE_path) > 0:
            for i in range(len(CONFIG_FILE_list)):
                CONFIG_FILE_list[i] = full_path(CONFIG_FILE_list[i], CONFIG_FILE_path)

            #   Completely automatic reading folder:

    if auto:
        fix_wavelengths = True
        list_of_objetos, list_of_files, list_of_exptimes, date, grating = list_fits_files_in_folder(path,
                                                                                                    return_list=True)
        print(" ")

        list_of_files_of_stars = []
        for i in range(len(list_of_objetos)):
            if list_of_objetos[i] in skyflat_names:
                file_skyflat = list_of_files[i][0]
                print("  - SKYFLAT automatically identified")

            if list_of_objetos[i] in ["H600", "HILT600", "Hilt600", "Hiltner600", "HILTNER600"] and list_of_objetos[
                i] not in disable_stars:
                print("  - Calibration star Hilt600 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol = list_of_files[i][0]
                objects_auto.append("Hilt600_" + grating)
                # _CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("Hilt600", path, grating, pk)
                # CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("Hilt600")
                CONFIG_FILE_list.append("")

            if list_of_objetos[i] in ["EG274", "Eg274", "eg274", "eG274", "E274", "e274"] and list_of_objetos[
                i] not in disable_stars:
                print("  - Calibration star EG274 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol = list_of_files[i][0]
                objects_auto.append("EG274_" + grating)
                # _CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("EG274", path, grating, pk)
                # CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("EG274")
                CONFIG_FILE_list.append("")

            if list_of_objetos[i] in ["HD60753", "hd60753", "Hd60753", "HD60753FLUX"] and list_of_objetos[
                i] not in disable_stars:
                print("  - Calibration star HD60753 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol = list_of_files[i][0]
                objects_auto.append("HD60753_" + grating)
                # _CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("HD60753", path, grating, pk)
                # CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("HD60753")
                CONFIG_FILE_list.append("")

            if list_of_objetos[i] in ["HD49798", "hd49798", "Hd49798"] and list_of_objetos[i] not in disable_stars:
                print("  - Calibration star HD49798 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol = list_of_files[i][0]
                objects_auto.append("HD49798_" + grating)
                # _CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("HD49798", path, grating, pk)
                # CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("HD49798")
                CONFIG_FILE_list.append("")

            if list_of_objetos[i] in ["cd32d9927", "CD32d9927", "CD32D9927", "CD-32d9927", "cd-32d9927", "Cd-32d9927",
                                      "CD-32D9927", "cd-32D9927", "Cd-32D9927"] and list_of_objetos[
                i] not in disable_stars:
                print("  - Calibration star CD32d9927 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol = list_of_files[i][0]
                objects_auto.append("CD32d9927_" + grating)
                # _CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("CD32d9927", path, grating, pk)
                # CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("CD32d9927")
                CONFIG_FILE_list.append("")

            if list_of_objetos[i] in ["HR3454", "Hr3454", "hr3454"] and list_of_objetos[i] not in disable_stars:
                print("  - Calibration star HR3454 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol = list_of_files[i][0]
                objects_auto.append("HR3454_" + grating)
                # _CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("HR3454", path, grating, pk)
                # CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("HR3454")
                CONFIG_FILE_list.append("")

            if list_of_objetos[i] in ["HR718", "Hr718", "hr718", "HR718FLUX", "HR718auto", "Hr718auto", "hr718auto",
                                      "HR718FLUXauto"] and list_of_objetos[i] not in disable_stars:
                print("  - Calibration star HR718 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol = list_of_files[i][0]
                objects_auto.append("HR718_" + grating)
                # _CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("HR718", path, grating, pk)
                # CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("HR718")
                CONFIG_FILE_list.append("")

        if throughput_2D_file != "":
            throughput_2D_file = full_path(throughput_2D_file, path)
            do_skyflat = False
            print("  - throughput_2D_file provided, no need of processing skyflat")
            sol = [0, 0, 0]
            ftf = fits.open(throughput_2D_file)
            if ftf[0].data[0][0] == 1.:
                sol[0] = ftf[0].header["SOL0"]
                sol[1] = ftf[0].header["SOL1"]
                sol[2] = ftf[0].header["SOL2"]
                print("  - solution for fixing small wavelength shifts included in this file :\n    sol = ", sol)
        print(" ")

    else:
        list_of_files_of_stars = [[], [], [], [], [], []]

    for i in range(len(config_property)):
        if config_property[i] == "date":     date = config_value[i]
        if config_property[i] == "grating":     grating = config_value[i]
        if config_property[i] == "pixel_size":     pixel_size = float(config_value[i])
        if config_property[i] == "kernel_size":
            kernel_size = float(config_value[i])
            pk = "_" + str(int(pixel_size)) + "p" + str(int((abs(pixel_size) - abs(int(pixel_size))) * 10)) + "_" + str(
                int(kernel_size)) + "k" + str(int(abs(kernel_size * 100)) - int(kernel_size) * 100)
        if config_property[i] == "path":
            path = config_value[i]
            if path[-1] != "/": path = path + "/"
            throughput_2D_file = path + "throughput_2D_" + date + "_" + grating + ".fits"
            flux_calibration_file = path + "flux_calibration_" + date + "_" + grating + pk + ".dat"
            if flux_calibration_name == "flux_calibration_auto": flux_calibration_name = "flux_calibration_" + date + "_" + grating + pk
            if grating == "385R" or grating == "1000R":
                telluric_correction_file = path + "telluric_correction_" + date + "_" + grating + ".dat"
                telluric_correction_name = "telluric_correction_" + date + "_" + grating

        if config_property[i] == "file_skyflat": file_skyflat = full_path(config_value[i], path)

        if config_property[i] == "skyflat":
            exec("global " + config_value[i])
            skyflat_variable = config_value[i]

        if config_property[i] == "do_skyflat":
            if config_value[i] == "True":
                do_skyflat = True
            else:
                do_skyflat = False

        if config_property[i] == "correct_ccd_defects":
            if config_value[i] == "True":
                correct_ccd_defects = True
            else:
                correct_ccd_defects = False

        if config_property[i] == "fix_wavelengths":
            if config_value[i] == "True": fix_wavelengths = True
        if config_property[i] == "sol":
            fix_wavelengths = True
            sol_ = config_value[i].strip('][').split(',')
            if float(sol_[0]) == -1:
                sol = [float(sol_[0])]
            else:
                if float(sol_[0]) != -0: sol = [float(sol_[0]), float(sol_[1]), float(sol_[2])]

        if config_property[i] == "kernel_throughput":     kernel_throughput = int(config_value[i])

        if config_property[i] == "rss_star_file_for_sol": rss_star_file_for_sol = full_path(config_value[i], path)

        if config_property[i] == "throughput_2D_file": throughput_2D_file = full_path(config_value[i], path)
        if config_property[i] == "throughput_2D": throughput_2D_variable = config_value[i]
        if config_property[i] == "flux_calibration_file":     flux_calibration_file = full_path(config_value[i], path)
        if config_property[i] == "telluric_correction_file":     telluric_correction_file = full_path(config_value[i],
                                                                                                      path)

        if config_property[i] == "CONFIG_FILE_path": CONFIG_FILE_path = config_value[i]
        if config_property[i] == "CONFIG_FILE": CONFIG_FILE_list.append(full_path(config_value[i], CONFIG_FILE_path))

        if config_property[i] == "abs_flux_scale":
            abs_flux_scale_ = config_value[i].strip('][').split(',')
            for j in range(len(abs_flux_scale_)):
                abs_flux_scale.append(float(abs_flux_scale_[j]))

        if config_property[i] == "plot":
            if config_value[i] == "True":
                plot = True
            else:
                plot = False

        if config_property[i] == "cal_from_calibrated_starcubes" and config_value[
            i] == "True": cal_from_calibrated_starcubes = True

        if config_property[i] == "object":
            objects_auto.append(config_value[i])

    if len(abs_flux_scale) == 0:
        for i in range(len(CONFIG_FILE_list)): abs_flux_scale.append(1.)

    # Print the summary of parameters

    print("> Parameters for automatically processing the calibrations of the night:\n")
    print("  date                       = ", date)
    print("  grating                    = ", grating)
    print("  path                       = ", path)
    if cal_from_calibrated_starcubes == False:
        if do_skyflat:
            print("  file_skyflat               = ", file_skyflat)
            if skyflat_variable != "": print("  Python object with skyflat = ", skyflat_variable)
            print("  correct_ccd_defects        = ", correct_ccd_defects)
            if fix_wavelengths:
                print("  fix_wavelengths            = ", fix_wavelengths)
                if sol[0] != 0 and sol[0] != -1:
                    print("    sol                      = ", sol)
                else:
                    if rss_star_file_for_sol == "":
                        print("    ---> However, no solution given! Setting fix_wavelength = False !")
                        fix_wavelengths = False
                    else:
                        print("    Star RSS file for getting small wavelength solution:", rss_star_file_for_sol)
        else:
            print("  throughput_2D_file         = ", throughput_2D_file)
            if throughput_2D_variable != "": print("  throughput_2D variable     = ", throughput_2D_variable)

        print("  pixel_size                 = ", pixel_size)
        print("  kernel_size                = ", kernel_size)

        if CONFIG_FILE_list[0] != "":

            for config_file in range(len(CONFIG_FILE_list)):
                if config_file == 0:
                    if len(CONFIG_FILE_list) > 1:
                        print("  CONFIG_FILE_LIST           =  [", CONFIG_FILE_list[config_file], ",")
                    else:
                        print("  CONFIG_FILE_LIST           =  [", CONFIG_FILE_list[config_file], "]")
                else:
                    if config_file == len(CONFIG_FILE_list) - 1:
                        print("                                 ", CONFIG_FILE_list[config_file], " ]")
                    else:
                        print("                                 ", CONFIG_FILE_list[config_file], ",")

    else:
        print("\n> The calibration of the night will be obtained using these fully calibrated starcubes:\n")

    if len(objects_auto) != 0:
        pprint = ""
        for i in range(len(objects_auto)):
            pprint = pprint + objects_auto[i] + "  "
        print("  Using stars in objects     = ", pprint)

    if len(abs_flux_scale) > 0: print("  abs_flux_scale             = ", abs_flux_scale)
    print("  plot                       = ", plot)

    print("\n> Output files:\n")
    if do_skyflat:
        if throughput_2D_variable != "": print("  throughput_2D variable     = ", throughput_2D_variable)
        print("  throughput_2D_file         = ", throughput_2D_file)
    print("  flux_calibration_file      = ", flux_calibration_file)
    if grating in red_gratings:
        print("  telluric_correction_file   = ", telluric_correction_file)

    print("\n===================================================================================")

    if do_skyflat:
        if rss_star_file_for_sol != "" and sol[0] == 0:
            print("\n> Getting the small wavelength solution, sol, using star RSS file")
            print(" ", rss_star_file_for_sol, "...")
            if grating in red_gratings:
                _rss_star_ = KOALA_RSS(rss_star_file_for_sol,
                                       correct_ccd_defects=False,
                                       fix_wavelengths=True, sol=[0],
                                       plot=plot)
            if grating in ["580V"]:
                _rss_star_ = KOALA_RSS(rss_star_file_for_sol,
                                       correct_ccd_defects=True, remove_5577=True,
                                       plot=plot)
            sol = _rss_star_.sol
            print("\n> Solution for the small wavelength variations:")
            print(" ", sol)

        throughput_2D_, skyflat_ = get_throughput_2D(file_skyflat, plot=plot, also_return_skyflat=True,
                                                     correct_ccd_defects=correct_ccd_defects,
                                                     fix_wavelengths=fix_wavelengths, sol=sol,
                                                     throughput_2D_file=throughput_2D_file,
                                                     kernel_throughput=kernel_throughput)

        if throughput_2D_variable != "":
            print("  Saving throughput 2D into Python variable:", throughput_2D_variable)
            exec(throughput_2D_variable + "=throughput_2D_", globals())

        if skyflat_variable != "":
            print("  Saving skyflat into Python variable:", skyflat_variable)
            exec(skyflat_variable + "=skyflat_", globals())

    else:
        if cal_from_calibrated_starcubes == False: print(
            "\n> Skyflat will not be processed! Throughput 2D calibration already provided.\n")
        check_nothing_done = check_nothing_done + 1

    good_CONFIG_FILE_list = []
    good_star_names = []
    stars = []
    if cal_from_calibrated_starcubes == False:
        for i in range(len(CONFIG_FILE_list)):

            run_star = True

            if CONFIG_FILE_list[i] != "":
                try:
                    config_property, config_value = read_table(CONFIG_FILE_list[i], ["s", "s"])
                    if len(CONFIG_FILE_list) != len(objects_auto):
                        for j in range(len(config_property)):
                            if config_property[j] == "obj_name": running_star = config_value[j]
                        if i < len(objects_auto):
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
                if i < len(objects_auto):
                    objects_auto[i] = running_star
                else:
                    objects_auto.append(running_star)

            if run_star:
                pepe = 0
                if pepe == 0:
                    # try:
                    print("===================================================================================")
                    print("\n> Running automatically calibration star", running_star, "in CONFIG_FILE:")
                    print(" ", CONFIG_FILE_list[i], "\n")
                    psol = "[" + np.str(sol[0]) + "," + np.str(sol[1]) + "," + np.str(sol[2]) + "]"
                    exec(
                        'run_automatic_star(CONFIG_FILE_list[i], object_auto="' + running_star + '", star=star_list[i], sol =' + psol + ', throughput_2D_file = "' + throughput_2D_file + '", rss_list = list_of_files_of_stars[i], path_star=path, date=date,grating=grating,pixel_size=pixel_size,kernel_size=kernel_size, rss_clean=rss_clean)')
                    print("\n> Running automatically calibration star in CONFIG_FILE")
                    print("  ", CONFIG_FILE_list[i], " SUCCESSFUL !!\n")
                    good_CONFIG_FILE_list.append(CONFIG_FILE_list[i])
                    good_star_names.append(running_star)
                    try:  # This is for a combined cube
                        exec("stars.append(" + running_star + ".combined_cube)")
                        if grating in red_gratings:
                            exec(
                                "telluric_correction_list.append(" + running_star + ".combined_cube.telluric_correction)")
                    except Exception:  # This is when we read a cube from fits file
                        exec("stars.append(" + running_star + ")")
                        if grating in red_gratings:
                            exec("telluric_correction_list.append(" + running_star + ".telluric_correction)")
                            # except Exception:
                #     print("===================================================================================")
                #     print("\n> ERROR! something wrong happened running config file {} !\n".format(CONFIG_FILE_list[i]))

    else:  # This is for the case that we have individual star cubes ALREADY calibrated in flux
        pprint = ""
        stars = []
        good_star_names = []
        for i in range(len(objects_auto)):
            pprint = pprint + objects_auto[i] + "  "
            try:  # This is for a combined cube
                exec("stars.append(" + objects_auto[i] + ".combined_cube)")
                if grating in red_gratings:
                    exec("telluric_correction_list.append(" + objects_auto[i] + ".combined_cube.telluric_correction)")
            except Exception:  # This is when we read a cube from fits file
                exec("stars.append(" + objects_auto[i] + ")")
                if grating in red_gratings:
                    exec("telluric_correction_list.append(" + objects_auto[i] + ".telluric_correction)")
            good_star_names.append(stars[i].object)

        print("\n> Fully calibrated star cubes provided :", pprint)
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
        w = stars[0].wavelength
        spectrum_to_text_file(w, flux_calibration_night, filename=flux_calibration_file)

        # Similarly, provide a list with the telluric corrections and apply:
        if grating in red_gratings:
            telluric_correction_night = obtain_telluric_correction(w, telluric_correction_list,
                                                                   label_stars=good_star_names, scale=abs_flux_scale)
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
        print("\n> SUMMARY of running configuration file", CALIBRATION_NIGHT_FILE, ":\n")
    else:
        print("\n> SUMMARY of running automatic_calibration_night() :\n")

    if len(objects_auto) > 0 and cal_from_calibrated_starcubes == False:
        pprint = ""
        for i in range(len(objects_auto)):
            pprint = pprint + objects_auto[i] + "  "
        print("  Created objects for calibration stars           :", pprint)

        if len(CONFIG_FILE_list) > 0:
            print("  Variable with the flux calibration              :", flux_calibration_name)
            if grating in red_gratings:
                print("  Variable with the telluric calibration          :", telluric_correction_name)
                print(" ")
        print("  throughput_2D_file        = ", throughput_2D_file)
        if throughput_2D_variable != "": print("  throughput_2D variable    = ", throughput_2D_variable)

        if sol[0] != -1 and sol[0] != 0:
            print("  The throughput_2D information HAS BEEN corrected for small wavelength variations:")
            print(
                "  sol                       =  [" + np.str(sol[0]) + "," + np.str(sol[1]) + "," + np.str(sol[2]) + "]")

        if skyflat_variable != "": print("  Python object created with skyflat = ", skyflat_variable)

        if len(CONFIG_FILE_list) > 0:
            print('  flux_calibration_file     = "' + flux_calibration_file + '"')
            if grating in red_gratings:
                print('  telluric_correction_file  = "' + telluric_correction_file + '"')

    if cal_from_calibrated_starcubes:
        print("  Variable with the flux calibration              :", flux_calibration_name)
        if grating in red_gratings:
            print("  Variable with the telluric calibration          :", telluric_correction_name)
            print(" ")
        print('  flux_calibration_file     = "' + flux_calibration_file + '"')
        if grating in red_gratings:
            print('  telluric_correction_file  = "' + telluric_correction_file + '"')

    if check_nothing_done == 2:
        print("\n> NOTHING DONE!")

    print("\n===================================================================================")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_calibration_star_data(star, path_star, grating, pk):
    description = star
    fits_file = path_star + star + "_" + grating + pk + ".fits"
    response_file = path_star + star + "_" + grating + pk + "_response.dat"
    telluric_file = path_star + star + "_" + grating + pk + "_telluric_correction.dat"

    if grating in blue_gratings: CONFIG_FILE = "CONFIG_FILES/STARS/calibration_star_blue.config"
    if grating in red_gratings:
        CONFIG_FILE = "CONFIG_FILES/STARS/calibration_star_red.config"
        list_of_telluric_ranges = [[6150, 6240, 6410, 6490], [6720, 6855, 7080, 7140],  # DEFAULT VALUES
                                   [7080, 7140, 7500, 7580], [7400, 7580, 7705, 7850],
                                   [7850, 8090, 8450, 8700]]
    else:
        list_of_telluric_ranges = [[0]]

    if star in ["cd32d9927", "CD32d9927", "CD32D9927", "cd32d9927auto", "CD32d9927auto", "CD32D9927auto"]:
        absolute_flux_file = 'FLUX_CAL/fcd32d9927_edited.dat'
        # list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140],
        #                             [7080,7140,7500,7580], [7400,7580,7705,7850],
        #                             [7850,8090,8450,8700] ]
        # If needed, include here particular CONFIG FILES:
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_CD32d9927_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_CD32d9927_blue.config"
    if star in ["HD49798", "hd49798", "HD49798auto", "hd49798auto"]:
        absolute_flux_file = 'FLUX_CAL/fhd49798.dat'
        # list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140],
        #                             [7080,7140,7500,7580], [7400,7580,7705,7850],
        #                             [7850,8090,8450,8700] ]
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HD49798_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HD49798_blue.config"
    if star in ["HD60753", "hd60753", "HD60753auto", "hd60753auto", "HD60753FLUX", "hd60753FLUX", "HD60753FLUXauto",
                "hd60753FLUXauto"]:
        absolute_flux_file = 'FLUX_CAL/fhd60753.dat'
        # list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140],
        #                             [7080,7140,7500,7580], [7400,7580,7705,7850],
        #                             [7850,8090,8450,8700] ]
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HD60753_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HD60753_blue.config"
    if star in ["H600", "Hiltner600", "Hilt600", "H600auto"]:
        absolute_flux_file = 'FLUX_CAL/fhilt600_edited.dat'
        # list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140],
        #                            [7080,7140,7500,7580], [7400,7580,7705,7850],
        #                            [7850,8090,8450,8700] ]
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_Hilt600_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/ccalibration_Hilt600_blue.config"

    if star in ["EG274", "E274", "eg274", "e274", "EG274auto", "E274auto", "eg274auto", "e274auto"]:
        absolute_flux_file = '=FLUX_CAL/feg274_edited.dat'
        list_of_telluric_ranges = [[6150, 6245, 6380, 6430], [6720, 6855, 7080, 7150],
                                   [7080, 7140, 7500, 7580], [7400, 7580, 7705, 7850],
                                   [7850, 8090, 8450, 8700]]
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_EG274_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_EG274_blue.config"
    if star in ["EG21", "eg21", "Eg21", "EG21auto", "eg21auto", "Eg21auto"]:
        absolute_flux_file = 'FLUX_CAL/feg21_edited.dat'
        list_of_telluric_ranges = [[6150, 6245, 6380, 6430], [6720, 6855, 7080, 7150],
                                   [7080, 7140, 7500, 7580], [7400, 7580, 7705, 7850],
                                   [7850, 8090, 8450, 8700]]
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_EG21_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_EG21_blue.config"
    if star in ["HR3454", "Hr3454", "hr3454", "HR3454auto", "Hr3454auto", "hr3454auto"]:
        absolute_flux_file = 'FLUX_CAL/fhr3454_edited.dat'
        list_of_telluric_ranges = [[6150, 6245, 6380, 6430], [6720, 6855, 7080, 7150],
                                   [7080, 7140, 7500, 7580], [7400, 7580, 7705, 7850],
                                   [7850, 8090, 8450, 8700]]
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HR3454_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HR3454_blue.config"
    if star in ["HR718", "Hr718", "hr718", "HR718FLUX", "HR718auto", "Hr718auto", "hr718auto", "HR718FLUXauto"]:
        absolute_flux_file = 'FLUX_CAL/fhr718_edited.dat'
        # list_of_telluric_ranges =  [ [6150,6245,6380,6430], [6720,6855,7080,7150],
        #                             [7080,7140,7500,7580], [7400,7580,7705,7850],
        #                             [7850,8090,8450,8700] ]
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HR718_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HR718_blue.config"

    return CONFIG_FILE, description, fits_file, response_file, absolute_flux_file, list_of_telluric_ranges


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def run_automatic_star(CONFIG_FILE="",
                       star="",
                       description="",
                       obj_name="",
                       object_auto="",
                       date="", grating="",
                       pixel_size=0.7,
                       kernel_size=1.1,
                       path_star="",
                       rss_list=[],
                       path="",
                       reduced=False,
                       read_fits_cube=False,
                       fits_file="",
                       rss_clean=False,
                       save_rss=False,
                       save_rss_to_fits_file_list=[],

                       apply_throughput=False,
                       throughput_2D_variable="",
                       throughput_2D=[], throughput_2D_file="",
                       throughput_2D_wavecor=False,
                       valid_wave_min=0, valid_wave_max=0,
                       correct_ccd_defects=False,
                       fix_wavelengths=False, sol=[0, 0, 0],
                       do_extinction=False,
                       sky_method="none",
                       n_sky=100,
                       sky_fibres=[],
                       win_sky=0,
                       remove_5577=False,
                       correct_negative_sky=False,
                       order_fit_negative_sky=3,
                       kernel_negative_sky=51,
                       individual_check=True,
                       use_fit_for_negative_sky=False,
                       force_sky_fibres_to_zero=True,
                       low_fibres=10,
                       high_fibres=20,

                       remove_negative_median_values=False,
                       fix_edges=False,
                       clean_extreme_negatives=False,
                       percentile_min=0.9,
                       clean_cosmics=False,
                       width_bl=20.,
                       kernel_median_cosmics=5,
                       cosmic_higher_than=100.,
                       extra_factor=1.,

                       do_cubing=True, do_alignment=True, make_combined_cube=True,
                       edgelow=-1, edgehigh=-1,
                       ADR=False, jump=-1,
                       adr_index_fit=2, g2d=True,
                       box_x=[0, -1], box_y=[0, -1],
                       trim_cube=True, trim_values=[],
                       scale_cubes_using_integflux=False,
                       flux_ratios=[],

                       do_calibration=True,
                       absolute_flux_file="",
                       response_file="",
                       size_arcsec=[],
                       r_max=5.,
                       step_flux=10.,
                       ha_width=0, exp_time=0.,
                       min_wave_flux=0, max_wave_flux=0,
                       sky_annulus_low_arcsec=5.,
                       sky_annulus_high_arcsec=10.,
                       exclude_wlm=[[0, 0]],
                       odd_number=0,
                       smooth=0.,
                       fit_weight=0.,
                       smooth_weight=0.,

                       order_telluric=2,
                       list_of_telluric_ranges=[[0]],
                       apply_tc=True,
                       log=True, gamma=0,
                       fig_size=12,
                       plot=True, warnings=True, verbose=True):
    """
    Use:
        CONFIG_FILE_H600 = "./CONFIG_FILES/calibration_star1.config"
        H600auto=run_automatic_star(CONFIG_FILE_H600)
    """

    global star_object
    sky_fibres_print = ""
    if object_auto == "": print("\n> Running automatic script for processing a calibration star")

    rss_clean_given = rss_clean

    # # Setting default values (now the majority as part of definition)

    pk = "_" + str(int(pixel_size)) + "p" + str(int((abs(pixel_size) - abs(int(pixel_size))) * 10)) + "_" + str(
        int(kernel_size)) + "k" + str(int(abs(kernel_size * 100)) - int(kernel_size) * 100)
    first_telluric_range = True

    # # if path is given, add it to the rss_list if needed
    if path_star == "" and path != "": path_star = path
    if path == "" and path_star != "": path = path_star
    if path != "" and len(rss_list) > 0:
        for i in range(len(rss_list)):
            rss_list[i] = full_path(rss_list[i], path)

    if CONFIG_FILE == "":

        # If no configuration file is given, check if name of the star provided
        if star == "":
            print("  - No name for calibration star given, asuming name = star")
            star = "star"

        # If grating is not given, we can check reading a RSS file
        if grating == "":
            print("\n> No grating provided! Checking... ")
            _test_ = KOALA_RSS(rss_list[0], plot_final_rss=False, verbose=False)
            grating = _test_.grating
            print("\n> Reading file", rss_list[0], "the grating is", grating)

        CONFIG_FILE, description_, fits_file_, response_file_, absolute_flux_file_, list_of_telluric_ranges_ = get_calibration_star_data(
            star, path_star, grating, pk)

        if description == "": description = description_
        if fits_file == "": fits_file = fits_file_
        if response_file == "": response_file = response_file_
        if absolute_flux_file == "": absolute_flux_file = absolute_flux_file_
        if list_of_telluric_ranges == "": list_of_telluric_ranges = list_of_telluric_ranges_

        # Check if folder has throughput if not given
        if throughput_2D_file == "":
            print("\n> No throughout file provided, using default file:")
            throughput_2D_file = path_star + "throughput_2D_" + date + "_" + grating + ".fits"
            print("  ", throughput_2D_file)

    # # Read configuration file

    # print("\n  CONFIG FILE: ",CONFIG_FILE)
    config_property, config_value = read_table(CONFIG_FILE, ["s", "s"])

    if object_auto == "":
        print("\n> Reading configuration file", CONFIG_FILE, "...\n")
        if obj_name == "":
            object_auto = star + "_" + grating + "_" + date
        else:
            object_auto = obj_name

    for i in range(len(config_property)):

        if config_property[i] == "pixel_size":     pixel_size = float(config_value[i])
        if config_property[i] == "kernel_size":
            kernel_size = float(config_value[i])
            pk = "_" + str(int(pixel_size)) + "p" + str(int((abs(pixel_size) - abs(int(pixel_size))) * 10)) + "_" + str(
                int(kernel_size)) + "k" + str(int(abs(kernel_size * 100)) - int(kernel_size) * 100)
        if config_property[i] == "date":     date = config_value[i]
        if config_property[i] == "grating":     grating = config_value[i]

        if config_property[i] == "path_star":
            path_star = config_value[i]
            if path_star[-1] != "/": path_star = path_star + "/"
        if config_property[i] == "obj_name":  object_auto = config_value[i]
        if config_property[i] == "star":
            star = config_value[i]
            _CONFIG_FILE_, description, fits_file, response_file, absolute_flux_file, list_of_telluric_ranges = get_calibration_star_data(
                star, path_star, grating, pk)

        if config_property[i] == "description":  description = config_value[i]
        if config_property[i] == "fits_file":  fits_file = full_path(config_value[i], path_star)
        if config_property[i] == "response_file":  response_file = full_path(config_value[i], path_star)
        if config_property[i] == "telluric_file":  telluric_file = full_path(config_value[i], path_star)

        if config_property[i] == "rss": rss_list.append(full_path(config_value[i], path_star))  # list_of_files_of_stars
        if config_property[i] == "reduced":
            if config_value[i] == "True":  reduced = True

        if config_property[i] == "read_cube":
            if config_value[i] == "True": read_fits_cube = True

            # RSS Section -----------------------------

        if config_property[i] == "rss_clean":
            if config_value[i] == "True":
                rss_clean = True
            else:
                rss_clean = False

            if rss_clean_given == True: rss_clean = True

        if config_property[i] == "save_rss":
            if config_value[i] == "True":   save_rss = True

        if config_property[i] == "apply_throughput":
            if config_value[i] == "True":
                apply_throughput = True
            else:
                apply_throughput = False
        if config_property[i] == "throughput_2D_file": throughput_2D_file = full_path(config_value[i], path_star)
        if config_property[i] == "throughput_2D": throughput_2D_variable = config_value[
            i]  # full_path(config_value[i],path_star)

        if config_property[i] == "correct_ccd_defects":
            if config_value[i] == "True":
                correct_ccd_defects = True
            else:
                correct_ccd_defects = False
        if config_property[i] == "fix_wavelengths":
            if config_value[i] == "True": fix_wavelengths = True
        if config_property[i] == "sol":
            sol_ = config_value[i].strip('][').split(',')
            if float(sol_[0]) == -1:
                sol = [float(sol_[0])]
            else:
                if float(sol_[0]) != -0: sol = [float(sol_[0]), float(sol_[1]), float(sol_[2])]

        if config_property[i] == "throughput_2D_wavecor":
            if config_value[i] == "True":
                throughput_2D_wavecor = True
            else:
                throughput_2D_wavecor = False

        if config_property[i] == "do_extinction":
            if config_value[i] == "True":
                do_extinction = True
            else:
                do_extinction = False

        if config_property[i] == "sky_method": sky_method = config_value[i]
        if config_property[i] == "n_sky": n_sky = int(config_value[i])
        if config_property[i] == "win_sky": win_sky = int(config_value[i])
        if config_property[i] == "remove_5577":
            if config_value[i] == "True": remove_5577 = True
        if config_property[i] == "correct_negative_sky":
            if config_value[i] == "True": correct_negative_sky = True

        if config_property[i] == "order_fit_negative_sky": order_fit_negative_sky = int(config_value[i])
        if config_property[i] == "kernel_negative_sky": kernel_negative_sky = int(config_value[i])
        if config_property[i] == "individual_check":
            if config_value[i] == "True":
                individual_check = True
            else:
                individual_check = False
        if config_property[i] == "use_fit_for_negative_sky":
            if config_value[i] == "True":
                use_fit_for_negative_sky = True
            else:
                use_fit_for_negative_sky = False

        if config_property[i] == "force_sky_fibres_to_zero":
            if config_value[i] == "True":
                force_sky_fibres_to_zero = True
            else:
                force_sky_fibres_to_zero = False
        if config_property[i] == "high_fibres": high_fibres = int(config_value[i])
        if config_property[i] == "low_fibres": low_fibres = int(config_value[i])

        if config_property[i] == "sky_fibres":
            sky_fibres_ = config_value[i]
            if sky_fibres_ == "fibres_best_sky_100":
                sky_fibres = fibres_best_sky_100
                sky_fibres_print = "fibres_best_sky_100"
            else:
                if sky_fibres_[0:5] == "range":
                    sky_fibres_ = sky_fibres_[6:-1].split(',')
                    sky_fibres = list(range(np.int(sky_fibres_[0]), np.int(sky_fibres_[1])))
                    sky_fibres_print = "range(" + sky_fibres_[0] + "," + sky_fibres_[1] + ")"
                else:
                    sky_fibres_ = config_value[i].strip('][').split(',')
                    for i in range(len(sky_fibres_)):
                        sky_fibres.append(float(sky_fibres_[i]))
                    sky_fibres_print = sky_fibres

        if config_property[i] == "remove_negative_median_values":
            if config_value[i] == "True":
                remove_negative_median_values = True
            else:
                remove_negative_median_values = False

        if config_property[i] == "fix_edges" and config_value[i] == "True": fix_edges = True
        if config_property[i] == "clean_extreme_negatives":
            if config_value[i] == "True": clean_extreme_negatives = True
        if config_property[i] == "percentile_min": percentile_min = float(config_value[i])

        if config_property[i] == "clean_cosmics" and config_value[i] == "True": clean_cosmics = True
        if config_property[i] == "width_bl": width_bl = float(config_value[i])
        if config_property[i] == "kernel_median_cosmics": kernel_median_cosmics = int(config_value[i])
        if config_property[i] == "cosmic_higher_than": cosmic_higher_than = float(config_value[i])
        if config_property[i] == "extra_factor": extra_factor = float(config_value[i])

        # Cubing Section ------------------------------

        if config_property[i] == "do_cubing":
            if config_value[i] == "False":
                do_cubing = False
                do_alignment = False
                make_combined_cube = False  # LOki

        if config_property[i] == "size_arcsec":
            size_arcsec_ = config_value[i].strip('][').split(',')
            for i in range(len(size_arcsec_)):
                size_arcsec.append(float(size_arcsec_[i]))

        if config_property[i] == "edgelow": edgelow = int(config_value[i])
        if config_property[i] == "edgehigh": edgehigh = int(config_value[i])

        if config_property[i] == "ADR" and config_value[i] == "True": ADR = True
        if config_property[i] == "adr_index_fit": adr_index_fit = int(config_value[i])
        if config_property[i] == "g2d":
            if config_value[i] == "True":
                g2d = True
            else:
                g2d = False

        if config_property[i] == "jump": jump = int(config_value[i])

        if config_property[i] == "trim_cube":
            if config_value[i] == "True":
                trim_cube = True
            else:
                trim_cube = False

        if config_property[i] == "trim_values":
            trim_values_ = config_value[i].strip('][').split(',')
            trim_values = [int(trim_values_[0]), int(trim_values_[1]), int(trim_values_[2]), int(trim_values_[3])]

        if config_property[i] == "scale_cubes_using_integflux":
            if config_value[i] == "True":
                scale_cubes_using_integflux = True
            else:
                scale_cubes_using_integflux = False

        if config_property[i] == "flux_ratios":
            flux_ratios_ = config_value[i].strip('][').split(',')
            flux_ratios = []
            for i in range(len(flux_ratios_)):
                flux_ratios.append(float(flux_ratios_[i]))

        # Calibration  ---------------------------------

        if config_property[i] == "do_calibration":
            if config_value[i] == "False":  do_calibration = False
            if config_value[i] == "True":  do_calibration = True

        if config_property[i] == "r_max": r_max = float(config_value[i])

        # CHECK HOW TO OBTAIN TELLURIC CORRECTION !!!
        if config_property[i] == "order_telluric": order_telluric = int(config_value[i])
        if config_property[i] == "telluric_range":
            if first_telluric_range == True:
                list_of_telluric_ranges = []
                first_telluric_range = False
            telluric_range_string = config_value[i].strip('][').split(',')
            telluric_range_float = [float(telluric_range_string[0]), float(telluric_range_string[1]),
                                    float(telluric_range_string[2]), float(telluric_range_string[3])]
            list_of_telluric_ranges.append(telluric_range_float)

        if config_property[i] == "apply_tc":
            if config_value[i] == "True":
                apply_tc = True
            else:
                apply_tc = False

        if config_property[i] == "absolute_flux_file": absolute_flux_file = config_value[i]
        if config_property[i] == "min_wave_flux": min_wave_flux = float(config_value[i])
        if config_property[i] == "max_wave_flux": max_wave_flux = float(config_value[i])
        if config_property[i] == "step_flux": step_flux = float(config_value[i])
        if config_property[i] == "exp_time": exp_time = float(config_value[i])
        if config_property[i] == "fit_degree_flux": fit_degree_flux = int(config_value[i])
        if config_property[i] == "ha_width": ha_width = float(config_value[i])

        if config_property[i] == "sky_annulus_low_arcsec": sky_annulus_low_arcsec = float(config_value[i])
        if config_property[i] == "sky_annulus_high_arcsec": sky_annulus_high_arcsec = float(config_value[i])

        if config_property[i] == "valid_wave_min": valid_wave_min = float(config_value[i])
        if config_property[i] == "valid_wave_max": valid_wave_max = float(config_value[i])

        if config_property[i] == "odd_number": odd_number = int(config_value[i])
        if config_property[i] == "smooth": smooth = float(config_value[i])
        if config_property[i] == "fit_weight": fit_weight = float(config_value[i])
        if config_property[i] == "smooth_weight": smooth_weight = float(config_value[i])

        if config_property[i] == "exclude_wlm":
            exclude_wlm = []
            exclude_wlm_string_ = config_value[i].replace("]", "")
            exclude_wlm_string = exclude_wlm_string_.replace("[", "").split(',')
            for i in np.arange(0, len(exclude_wlm_string), 2):
                exclude_wlm.append([float(exclude_wlm_string[i]), float(exclude_wlm_string[i + 1])])

                # Plotting, printing ------------------------------

        if config_property[i] == "log":
            if config_value[i] == "True":
                log = True
            else:
                log = False
        if config_property[i] == "gamma": gamma = float(config_value[i])
        if config_property[i] == "fig_size": fig_size = float(config_value[i])
        if config_property[i] == "plot":
            if config_value[i] == "True":
                plot = True
            else:
                plot = False
        if config_property[i] == "plot_rss":
            if config_value[i] == "True":
                plot_rss = True
            else:
                plot_rss = False
        if config_property[i] == "plot_weight":
            if config_value[i] == "True":
                plot_weight = True
            else:
                plot_weight = False

        if config_property[i] == "warnings":
            if config_value[i] == "True":
                warnings = True
            else:
                warnings = False
        if config_property[i] == "verbose":
            if config_value[i] == "True":
                verbose = True
            else:
                verbose = False

    if throughput_2D_variable != "":  throughput_2D = eval(throughput_2D_variable)

    if do_cubing == False:
        fits_file = ""
        make_combined_cube = False
        do_alignment = False

    # # Print the summary of parameters

    print("> Parameters for processing this calibration star :\n")
    print("  star                     = ", star)
    if object_auto != "":
        if reduced == True and read_fits_cube == False:
            print("  Python object            = ", object_auto, "  already created !!")
        else:
            print("  Python object            = ", object_auto, "  to be created")
    print("  path                     = ", path_star)
    print("  description              = ", description)
    print("  date                     = ", date)
    print("  grating                  = ", grating)

    if reduced == False and read_fits_cube == False:
        for rss in range(len(rss_list)):
            if rss == 0:
                if len(rss_list) > 1:
                    print("  rss_list                 = [", rss_list[rss], ",")
                else:
                    print("  rss_list                 = [", rss_list[rss], "]")
            else:
                if rss == len(rss_list) - 1:
                    print("                              ", rss_list[rss], " ]")
                else:
                    print("                              ", rss_list[rss], ",")

        if rss_clean:
            print("  rss_clean                =  True, skipping to cubing\n")
        else:
            if save_rss: print("  'CLEANED' RSS files will be saved automatically")

            if apply_throughput:
                if throughput_2D_variable != "":
                    print("  throughput_2D variable   = ", throughput_2D_variable)
                else:
                    if throughput_2D_file != "": print("  throughput_2D_file       = ", throughput_2D_file)

            if apply_throughput and throughput_2D_wavecor:
                print("  throughput_2D will be applied AFTER correcting CCD defects and fixing small wavelenghts")

            print("  correct_ccd_defects      = ", correct_ccd_defects)
            print("  fix_wavelengths          = ", fix_wavelengths)
            if fix_wavelengths:
                if sol[0] == -1:
                    print("    Only using few skylines in the edges")
                else:
                    if sol[0] != -1: print("    sol                    = ", sol)

            print("  do_extinction            = ", do_extinction)
            print("  sky_method               = ", sky_method)
            if sky_method != "none":
                if len(sky_fibres) > 1:
                    print("    sky_fibres             = ", sky_fibres_print)
                else:
                    print("    n_sky                  = ", n_sky)
            if win_sky > 0: print("    win_sky                = ", win_sky)
            if remove_5577: print("    remove 5577 skyline    = ", remove_5577)
            print("  correct_negative_sky     = ", correct_negative_sky)
            if correct_negative_sky:
                print("    order_fit_negative_sky   = ", order_fit_negative_sky)
                print("    kernel_negative_sky      = ", kernel_negative_sky)
                print("    use_fit_for_negative_sky = ", use_fit_for_negative_sky)
                print("    low_fibres               = ", low_fibres)
                print("    individual_check         = ", individual_check)
                if sky_method in ["self", "selffit"]:  print("    force_sky_fibres_to_zero = ",
                                                             force_sky_fibres_to_zero)

            if fix_edges: print("  fix_edges                = ", fix_edges)

            print("  clean_cosmics            = ", clean_cosmics)
            if clean_cosmics:
                print("    width_bl               = ", width_bl)
                print("    kernel_median_cosmics  = ", kernel_median_cosmics)
                print("    cosmic_higher_than     = ", cosmic_higher_than)
                print("    extra_factor           = ", extra_factor)

            print("  clean_extreme_negatives  = ", clean_extreme_negatives)
            if clean_extreme_negatives:
                print("    percentile_min         = ", percentile_min)
            if remove_negative_median_values:
                print("  Negative pixels will be set to 0 when median value of spectrum is negative")

        if valid_wave_min > 0: print("  valid_wave_min           = ", valid_wave_min, "A")
        if valid_wave_max > 0: print("  valid_wave_max           = ", valid_wave_max, "A")

        if do_cubing:
            if len(size_arcsec) > 0: print("  cube_size_arcsec         = ", size_arcsec)

            if edgelow != -1:  print("  edgelow for tracing      = ", edgelow, "pixels")
            if edgehigh != -1: print("  edgehigh for tracing     = ", edgehigh, "pixels")
            print("  2D Gauss for tracing     = ", g2d)

            print("  ADR                      = ", ADR)
            if ADR: print("    adr_index_fit          = ", adr_index_fit)

            if jump != -1:    print("    jump for ADR           = ", jump)

            if scale_cubes_using_integflux:
                if len(flux_ratios) == 0:
                    print("  Scaling individual cubes using integrated flux of common region")
                else:
                    print("  Scaling individual cubes using flux_ratios = ", flux_ratios)

            if trim_cube: print("  Trim cube                = ", trim_cube)
            if len(trim_values) != 0: print("    Trim values            = ", trim_values)
        else:
            print("\n> No cubing will be performed\n")

    if do_calibration:

        if read_fits_cube:
            print("\n> Input fits file with cube:\n ", fits_file, "\n")
        else:
            print("  pixel_size               = ", pixel_size)
            print("  kernel_size              = ", kernel_size)
        print("  plot                     = ", plot)
        print("  verbose                  = ", verbose)

        print("  warnings                 = ", warnings)
        print("  r_max                    = ", r_max, '" for extracting the star')
        if grating in red_gratings:
            # print "  telluric_file        = ",telluric_file
            print("  Parameters for obtaining the telluric correction:")
            print("    apply_tc               = ", apply_tc)
            print("    order continuum fit    = ", order_telluric)
            print("    telluric ranges        = ", list_of_telluric_ranges[0])
            for i in range(1, len(list_of_telluric_ranges)):
                print("                             ", list_of_telluric_ranges[i])
        print("  Parameters for obtaining the absolute flux calibration:")
        print("     absolute_flux_file    = ", absolute_flux_file)

        if min_wave_flux == 0:  min_wave_flux = valid_wave_min
        if max_wave_flux == 0:  max_wave_flux = valid_wave_max

        if min_wave_flux > 0: print("     min_wave_flux         = ", min_wave_flux)
        if max_wave_flux > 0: print("     max_wave_flux         = ", max_wave_flux)
        print("     step_flux             = ", step_flux)
        if exp_time > 0:
            print("     exp_time              = ", exp_time)
        else:
            print("     exp_time              =  reads it from .fits files")
        print("     fit_degree_flux       = ", fit_degree_flux)
        print("     sky_annulus_low       = ", sky_annulus_low_arcsec, "arcsec")
        print("     sky_annulus_high      = ", sky_annulus_high_arcsec, "arcsec")
        if ha_width > 0: print("     ha_width              = ", ha_width, "A")
        if odd_number > 0: print("     odd_number            = ", odd_number)
        if smooth > 0: print("     smooth                = ", smooth)
        if fit_weight > 0: print("     fit_weight            = ", fit_weight)
        if smooth_weight > 0:     print("     smooth_weight         = ", smooth_weight)
        if exclude_wlm[0][0] != 0: print("     exclude_wlm           = ", exclude_wlm)

        print("\n> Output files:\n")
        if read_fits_cube == "False": print("  fits_file            =", fits_file)
        print("  integrated spectrum  =", fits_file[:-5] + "_integrated_star_flux.dat")
        if grating in red_gratings:
            print("  telluric_file        =", telluric_file)
        print("  response_file        =", response_file)
        print(" ")

    else:
        print("\n> No calibration will be performed\n")

    # # Read cube from fits file if given

    if read_fits_cube:
        star_object = read_cube(fits_file, valid_wave_min=valid_wave_min, valid_wave_max=valid_wave_max)
        reduced = True
        exp_time = np.nanmedian(star_object.exptimes)

        print(" ")
        exec(object_auto + "=copy.deepcopy(star_object)", globals())
        print("> Cube saved in object", object_auto, " !")

    # # Running KOALA_REDUCE using rss_list

    if reduced == False:

        for rss in rss_list:
            if save_rss:
                save_rss_to_fits_file_list.append("auto")
            else:
                save_rss_to_fits_file_list.append("")

        if do_cubing:
            print("> Running KOALA_reduce to create combined datacube...")
        else:
            print("> Running KOALA_reduce ONLY for processing the RSS files provided...")

        star_object = KOALA_reduce(rss_list,
                                   path=path,
                                   fits_file=fits_file,
                                   obj_name=star,
                                   description=description,
                                   save_rss_to_fits_file_list=save_rss_to_fits_file_list,
                                   rss_clean=rss_clean,
                                   grating=grating,
                                   apply_throughput=apply_throughput,
                                   throughput_2D_file=throughput_2D_file,
                                   throughput_2D=throughput_2D,
                                   correct_ccd_defects=correct_ccd_defects,
                                   fix_wavelengths=fix_wavelengths,
                                   sol=sol,
                                   throughput_2D_wavecor=throughput_2D_wavecor,
                                   do_extinction=do_extinction,
                                   sky_method=sky_method,
                                   n_sky=n_sky,
                                   win_sky=win_sky,
                                   remove_5577=remove_5577,
                                   sky_fibres=sky_fibres,
                                   correct_negative_sky=correct_negative_sky,
                                   order_fit_negative_sky=order_fit_negative_sky,
                                   kernel_negative_sky=kernel_negative_sky,
                                   individual_check=individual_check,
                                   use_fit_for_negative_sky=use_fit_for_negative_sky,
                                   force_sky_fibres_to_zero=force_sky_fibres_to_zero,
                                   low_fibres=low_fibres,
                                   high_fibres=high_fibres,

                                   fix_edges=fix_edges,
                                   clean_extreme_negatives=clean_extreme_negatives, percentile_min=percentile_min,
                                   remove_negative_median_values=remove_negative_median_values,
                                   clean_cosmics=clean_cosmics,
                                   width_bl=width_bl, kernel_median_cosmics=kernel_median_cosmics,
                                   cosmic_higher_than=cosmic_higher_than, extra_factor=extra_factor,

                                   do_cubing=do_cubing, do_alignment=do_alignment,
                                   make_combined_cube=make_combined_cube,
                                   pixel_size_arcsec=pixel_size,
                                   kernel_size_arcsec=kernel_size,
                                   size_arcsec=size_arcsec,
                                   edgelow=edgelow, edgehigh=edgehigh,
                                   ADR=ADR,
                                   adr_index_fit=adr_index_fit, g2d=g2d,
                                   jump=jump,
                                   box_x=box_x, box_y=box_y,
                                   trim_values=trim_values,
                                   scale_cubes_using_integflux=scale_cubes_using_integflux,
                                   flux_ratios=flux_ratios,
                                   valid_wave_min=valid_wave_min,
                                   valid_wave_max=valid_wave_max,
                                   log=log,
                                   gamma=gamma,
                                   plot=plot,
                                   plot_rss=plot_rss,
                                   plot_weight=plot_weight,
                                   fig_size=fig_size,
                                   verbose=verbose,
                                   warnings=warnings)

        # Save object is given
        if object_auto != 0:  # and make_combined_cube == True:
            exec(object_auto + "=copy.deepcopy(star_object)", globals())
            print("> Cube saved in object", object_auto, " !")

    else:
        if read_fits_cube == False:
            print("> Python object", object_auto, "already created.")
            exec("star_object=copy.deepcopy(" + object_auto + ")", globals())

    #  Perform the calibration

    if do_calibration:

        # Check exposition times
        if exp_time == 0:
            different_times = False
            try:
                exptimes = star_object.combined_cube.exptimes
            except Exception:
                exptimes = star_object.exptimes

            exp_time1 = exptimes[0]
            print("\n> Exposition time reading from rss1: ", exp_time1, " s")

            exp_time_list = [exp_time1]
            for i in range(1, len(exptimes)):
                exp_time_n = exptimes[i]
                exp_time_list.append(exp_time_n)
                if exp_time_n != exp_time1:
                    print("  Exposition time reading from rss" + np.str(i), " = ", exp_time_n, " s")
                    different_times = True

            if different_times:
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
        if grating in red_gratings:

            # Extract the integrated spectrum of the star & save it

            print("\n> Extracting the integrated spectrum of the star...")

            star_cube.half_light_spectrum(r_max=r_max, plot=plot)
            spectrum_to_text_file(star_cube.wavelength,
                                  star_cube.integrated_star_flux,
                                  filename=fits_file[:-5] + "_integrated_star_flux_before_TC.dat")

            # Find telluric correction CAREFUL WITH apply_tc=True

            print("\n> Finding telluric correction...")
            try:
                telluric_correction_star = telluric_correction_from_star(star_object,
                                                                         list_of_telluric_ranges=list_of_telluric_ranges,
                                                                         order=order_telluric,
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
                    if object_auto != 0: exec(object_auto + "=copy.deepcopy(star_object)", globals())

            except Exception:
                print("\n> Finding telluric correction FAILED!")

        # Flux calibration

        print("\n> Finding absolute flux calibration...")

        # Now we read the absolute flux calibration data of the calibration star and get the response curve
        # (Response curve: correspondence between counts and physical values)
        # Include exp_time of the calibration star, as the results are given per second
        # For this BE CAREFUL WITH ABSORPTIONS (Halpha) and check behaviour in the edges of the CCD
        # Change fit_degree (3,5,7), step, min_wave, max_wave to get better fits !!!

        # try:
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
                                        exclude_wlm=exclude_wlm,
                                        odd_number=odd_number,
                                        smooth=smooth,
                                        fit_weight=fit_weight,
                                        smooth_weight=smooth_weight)

            spectrum_to_text_file(star_cube.wavelength,
                                  star_cube.integrated_star_flux,
                                  filename=fits_file[:-5] + "_integrated_star_flux.dat")

            # Now we can save this calibration as a text file

            spectrum_to_text_file(star_cube.wavelength,
                                  star_cube.response_curve,
                                  filename=response_file, verbose=False)

            print('\n> Absolute flux calibration (response) saved in text file :\n  "' + response_file + '"')

            if object_auto != 0: exec(object_auto + "=copy.deepcopy(star_object)", globals())

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
def automatic_KOALA_reduce(KOALA_REDUCE_FILE, path=""):
    print("\n\n=============== running automatic_KOALA_reduce =======================")

    global hikids

    throughput_2D_variable = ""

    flux_calibration_file = ""
    flux_calibration_file_list = []
    flux_calibration = []
    flux_calibration = ""

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
    cube_list = []
    cube_list_names = []
    save_aligned_cubes = False

    apply_throughput = False
    throughput_2D = []
    throughput_2D_file = ""
    throughput_2D_wavecor = False

    correct_ccd_defects = False
    kernel_correct_ccd_defects = 51

    fix_wavelengths = False
    sol = [0, 0, 0]
    do_extinction = False

    do_telluric_correction = False

    sky_method = "none"
    sky_spectrum = []
    sky_spectrum_name = ""
    #   sky_spectrum_file = ""   #### NEEDS TO BE IMPLEMENTED
    sky_list = []
    sky_fibres = [1000]
    sky_lines_file = ""

    scale_sky_1D = 1.
    auto_scale_sky = False
    n_sky = 50
    print_n_sky = False
    win_sky = 0
    remove_5577 = False

    correct_negative_sky = False
    order_fit_negative_sky = 3
    kernel_negative_sky = 51
    individual_check = True
    use_fit_for_negative_sky = False
    force_sky_fibres_to_zero = True
    high_fibres = 20
    low_fibres = 10

    brightest_line = "Ha"
    brightest_line_wavelength = 0.
    ranges_with_emission_lines = [0]
    cut_red_end = 0

    id_el = False
    id_list = [0]
    cut = 1.5
    broad = 1.8
    plot_id_el = False

    clean_sky_residuals = False
    features_to_fix = []
    sky_fibres_for_residuals = []
    sky_fibres_for_residuals_print = "Using the same n_sky fibres"

    remove_negative_median_values = False
    fix_edges = False
    clean_extreme_negatives = False
    percentile_min = 0.9

    clean_cosmics = False
    width_bl = 20.
    kernel_median_cosmics = 5
    cosmic_higher_than = 100.
    extra_factor = 1.

    offsets = []
    ADR = False
    ADR_cc = False
    force_ADR = False
    box_x = []
    box_y = []
    jump = -1
    half_size_for_centroid = 10
    ADR_x_fit_list = []
    ADR_y_fit_list = []
    adr_index_fit = 2
    g2d = False
    step_tracing = 100
    plot_tracing_maps = []
    edgelow = -1
    edgehigh = -1

    delta_RA = 0
    delta_DEC = 0

    trim_cube = False
    trim_values = []
    size_arcsec = []
    centre_deg = []
    scale_cubes_using_integflux = True
    remove_spaxels_not_fully_covered = True
    flux_ratios = []

    valid_wave_min = 0
    valid_wave_max = 0

    plot = True
    plot_rss = True
    plot_weight = False
    plot_spectra = True
    fig_size = 12.

    log = True
    gamma = 0.

    warnings = False
    verbose = True

    if path != "": KOALA_REDUCE_FILE = full_path(KOALA_REDUCE_FILE, path)  # VR
    config_property, config_value = read_table(KOALA_REDUCE_FILE, ["s", "s"])

    print("\n> Reading configuration file", KOALA_REDUCE_FILE, "...\n")

    for i in range(len(config_property)):

        if config_property[i] == "pixel_size":     pixel_size = float(config_value[i])
        if config_property[i] == "kernel_size":
            kernel_size = float(config_value[i])
            pk = "_" + str(int(pixel_size)) + "p" + str(int((abs(pixel_size) - abs(int(pixel_size))) * 10)) + "_" + str(
                int(kernel_size)) + "k" + str(int(abs(kernel_size * 100)) - int(kernel_size) * 100)
        if config_property[i] == "date":     date = config_value[i]
        if config_property[i] == "grating":     grating = config_value[i]

        if config_property[i] == "path":
            path = config_value[i]

        if config_property[i] == "obj_name":
            obj_name = config_value[i]
            description = obj_name
            fits_file = path + obj_name + "_" + grating + pk + ".fits"
            Python_obj_name = obj_name + "_" + grating + pk
        if config_property[i] == "description":  description = config_value[i]
        if config_property[i] == "Python_obj_name": Python_obj_name = config_value[i]

        if config_property[i] == "flux_calibration_file":
            flux_calibration_name = ""
            flux_calibration_file_ = config_value[i]
            if len(flux_calibration_file_.split("/")) == 1:
                flux_calibration_file = path + flux_calibration_file_
            else:
                flux_calibration_file = flux_calibration_file_
            flux_calibration_file_list.append(flux_calibration_file)
        if config_property[i] == "telluric_correction_file":
            telluric_correction_name = ""
            telluric_correction_file_ = config_value[i]
            if len(telluric_correction_file_.split("/")) == 1:
                telluric_correction_file = path + telluric_correction_file_
            else:
                telluric_correction_file = telluric_correction_file_
            telluric_correction_list_.append(config_value[i])

        if config_property[i] == "flux_calibration_name":
            flux_calibration_name = config_value[i]
            # flux_calibration_name_list.append(flux_calibration_name)

        if config_property[i] == "telluric_correction_name":
            telluric_correction_name = config_value[i]

        if config_property[i] == "fits_file":
            fits_file_ = config_value[i]
            if len(fits_file_.split("/")) == 1:
                fits_file = path + fits_file_
            else:
                fits_file = fits_file_

        if config_property[i] == "save_aligned_cubes":
            if config_value[i] == "True": save_aligned_cubes = True

        if config_property[i] == "rss_file":
            rss_file_ = config_value[i]
            if len(rss_file_.split("/")) == 1:
                _rss_file_ = path + rss_file_
            else:
                _rss_file_ = rss_file_
            rss_list.append(_rss_file_)
        if config_property[i] == "cube_file":
            cube_file_ = config_value[i]
            if len(cube_file_.split("/")) == 1:
                _cube_file_ = path + cube_file_
            else:
                _cube_file_ = cube_file_
            cube_list_names.append(_cube_file_)
            cube_list.append(_cube_file_)  # I am not sure about this...

        if config_property[i] == "rss_clean":
            if config_value[i] == "True":
                rss_clean = True
            else:
                rss_clean = False
        if config_property[i] == "save_rss":
            if config_value[i] == "True":
                save_rss = True
            else:
                save_rss = False
        if config_property[i] == "do_cubing" and config_value[i] == "False":  do_cubing = False

        if config_property[i] == "apply_throughput":
            if config_value[i] == "True":
                apply_throughput = True
            else:
                apply_throughput = False

        if config_property[i] == "throughput_2D_file":
            throughput_2D_file_ = config_value[i]
            if len(throughput_2D_file_.split("/")) == 1:
                throughput_2D_file = path + throughput_2D_file_
            else:
                throughput_2D_file = throughput_2D_file_

        if config_property[i] == "throughput_2D": throughput_2D_variable = config_value[i]

        if config_property[i] == "throughput_2D_wavecor":
            if config_value[i] == "True":
                throughput_2D_wavecor = True
            else:
                throughput_2D_wavecor = False

        if config_property[i] == "correct_ccd_defects":
            if config_value[i] == "True":
                correct_ccd_defects = True
            else:
                correct_ccd_defects = False
        if config_property[i] == "kernel_correct_ccd_defects":     kernel_correct_ccd_defects = float(config_value[i])

        if config_property[i] == "fix_wavelengths":
            if config_value[i] == "True":
                fix_wavelengths = True
            else:
                fix_wavelengths = False
        if config_property[i] == "sol":
            sol_ = config_value[i].strip('][').split(',')
            if float(sol_[0]) == -1:
                sol = [float(sol_[0])]
            else:
                sol = [float(sol_[0]), float(sol_[1]), float(sol_[2])]

        if config_property[i] == "do_extinction":
            if config_value[i] == "True":
                do_extinction = True
            else:
                do_extinction = False

        if config_property[i] == "sky_method": sky_method = config_value[i]

        if config_property[i] == "sky_file": sky_list.append(path + config_value[i])
        if config_property[i] == "n_sky": n_sky = int(config_value[i])

        if config_property[i] == "sky_fibres":
            sky_fibres_ = config_value[i]
            if sky_fibres_[0:5] == "range":
                sky_fibres_ = sky_fibres_[6:-1].split(',')
                sky_fibres = list(range(np.int(sky_fibres_[0]), np.int(sky_fibres_[1])))
                sky_fibres_print = "range(" + sky_fibres_[0] + "," + sky_fibres_[1] + ")"
            else:
                sky_fibres_ = config_value[i].strip('][').split(',')
                for i in range(len(sky_fibres_)):
                    sky_fibres.append(float(sky_fibres_[i]))
                sky_fibres_print = sky_fibres

        if config_property[i] == "win_sky": win_sky = int(config_value[i])

        if config_property[i] == "sky_spectrum":
            if config_value[i] != "[0]":
                sky_spectrum_name = config_value[i]
                exec("sky_spectrum =" + sky_spectrum_name)
            else:
                sky_spectrum = []
        if config_property[i] == "scale_sky_1D":     scale_sky_1D = float(config_value[i])

        if config_property[i] == "auto_scale_sky":
            if config_value[i] == "True":
                auto_scale_sky = True
            else:
                auto_scale_sky = False

        if config_property[i] == "sky_lines_file": sky_lines_file = config_value[i]

        if config_property[i] == "correct_negative_sky":
            if config_value[i] == "True":
                correct_negative_sky = True
            else:
                correct_negative_sky = False

        if config_property[i] == "order_fit_negative_sky": order_fit_negative_sky = int(config_value[i])
        if config_property[i] == "kernel_negative_sky": kernel_negative_sky = int(config_value[i])
        if config_property[i] == "individual_check":
            if config_value[i] == "True":
                individual_check = True
            else:
                individual_check = False
        if config_property[i] == "use_fit_for_negative_sky":
            if config_value[i] == "True":
                use_fit_for_negative_sky = True
            else:
                use_fit_for_negative_sky = False
        if config_property[i] == "force_sky_fibres_to_zero":
            if config_value[i] == "True":
                force_sky_fibres_to_zero = True
            else:
                force_sky_fibres_to_zero = False
        if config_property[i] == "high_fibres": high_fibres = int(config_value[i])
        if config_property[i] == "low_fibres": low_fibres = int(config_value[i])

        if config_property[i] == "remove_5577":
            if config_value[i] == "True":
                remove_5577 = True
            else:
                remove_5577 = False

        if config_property[i] == "do_telluric_correction":
            if config_value[i] == "True":
                do_telluric_correction = True
            else:
                do_telluric_correction = False
                telluric_correction_name = ""
                telluric_correction_file = ""

        if config_property[i] == "brightest_line":  brightest_line = config_value[i]
        if config_property[i] == "brightest_line_wavelength": brightest_line_wavelength = float(config_value[i])

        if config_property[i] == "ranges_with_emission_lines":
            ranges_with_emission_lines_ = config_value[i].strip('[]').replace('],[', ',').split(',')
            ranges_with_emission_lines = []
            for i in range(len(ranges_with_emission_lines_)):
                if i % 2 == 0: ranges_with_emission_lines.append(
                    [float(ranges_with_emission_lines_[i]), float(ranges_with_emission_lines_[i + 1])])
        if config_property[i] == "cut_red_end":  cut_red_end = config_value[i]

        # CHECK id_el
        if config_property[i] == "id_el":
            if config_value[i] == "True":
                id_el = True
            else:
                id_el = False
        if config_property[i] == "cut": cut = float(config_value[i])
        if config_property[i] == "broad": broad = float(config_value[i])
        if config_property[i] == "id_list":
            id_list_ = config_value[i].strip('][').split(',')
            for i in range(len(id_list_)):
                id_list.append(float(id_list_[i]))
        if config_property[i] == "plot_id_el":
            if config_value[i] == "True":
                plot_id_el = True
            else:
                plot_id_el = False

        if config_property[i] == "clean_sky_residuals" and config_value[i] == "True": clean_sky_residuals = True
        if config_property[i] == "fix_edges" and config_value[i] == "True": fix_edges = True

        if config_property[i] == "feature_to_fix":
            feature_to_fix_ = config_value[i]  # .strip('][').split(',')
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

        if config_property[i] == "sky_fibres_for_residuals":
            sky_fibres_for_residuals_ = config_value[i]
            if sky_fibres_for_residuals_[0:5] == "range":
                sky_fibres_for_residuals_ = sky_fibres_for_residuals_[6:-1].split(',')
                sky_fibres_for_residuals = list(
                    range(np.int(sky_fibres_for_residuals_[0]), np.int(sky_fibres_for_residuals_[1])))
                sky_fibres_for_residuals_print = "range(" + sky_fibres_for_residuals_[0] + "," + \
                                                 sky_fibres_for_residuals_[1] + ")"

            else:
                sky_fibres_for_residuals_ = config_value[i].strip('][').split(',')
                for i in range(len(sky_fibres_for_residuals_)):
                    sky_fibres_for_residuals.append(float(sky_fibres_for_residuals_[i]))
                sky_fibres_for_residuals_print = sky_fibres_for_residuals

        if config_property[i] == "clean_cosmics" and config_value[i] == "True": clean_cosmics = True
        if config_property[i] == "width_bl": width_bl = float(config_value[i])
        if config_property[i] == "kernel_median_cosmics": kernel_median_cosmics = int(config_value[i])
        if config_property[i] == "cosmic_higher_than": cosmic_higher_than = float(config_value[i])
        if config_property[i] == "extra_factor": extra_factor = float(config_value[i])

        if config_property[i] == "clean_extreme_negatives":
            if config_value[i] == "True": clean_extreme_negatives = True
        if config_property[i] == "percentile_min": percentile_min = float(config_value[i])

        if config_property[i] == "remove_negative_median_values":
            if config_value[i] == "True":
                remove_negative_median_values = True
            else:
                remove_negative_median_values = False

        if config_property[i] == "read_cube":
            if config_value[i] == "True":
                read_cube = True
            else:
                read_cube = False

        if config_property[i] == "offsets":
            offsets_ = config_value[i].strip('][').split(',')
            for i in range(len(offsets_)):
                offsets.append(float(offsets_[i]))

        if config_property[i] == "valid_wave_min": valid_wave_min = float(config_value[i])
        if config_property[i] == "valid_wave_max": valid_wave_max = float(config_value[i])

        if config_property[i] == "half_size_for_centroid": half_size_for_centroid = int(config_value[i])

        if config_property[i] == "box_x":
            box_x_ = config_value[i].strip('][').split(',')
            for i in range(len(box_x_)):
                box_x.append(int(box_x_[i]))
        if config_property[i] == "box_y":
            box_y_ = config_value[i].strip('][').split(',')
            for i in range(len(box_y_)):
                box_y.append(int(box_y_[i]))

        if config_property[i] == "adr_index_fit": adr_index_fit = int(config_value[i])
        if config_property[i] == "g2d":
            if config_value[i] == "True":
                g2d = True
            else:
                g2d = False
        if config_property[i] == "step_tracing": step_tracing = int(config_value[i])

        if config_property[i] == "plot_tracing_maps":
            plot_tracing_maps_ = config_value[i].strip('][').split(',')
            for i in range(len(plot_tracing_maps_)):
                plot_tracing_maps.append(float(plot_tracing_maps_[i]))

        if config_property[i] == "edgelow": edgelow = int(config_value[i])
        if config_property[i] == "edgehigh": edgehigh = int(config_value[i])

        if config_property[i] == "ADR":
            if config_value[i] == "True":
                ADR = True
            else:
                ADR = False
        if config_property[i] == "ADR_cc":
            if config_value[i] == "True":
                ADR_cc = True
            else:
                ADR_cc = False
        if config_property[i] == "force_ADR":
            if config_value[i] == "True":
                force_ADR = True
            else:
                force_ADR = False

        if config_property[i] == "ADR_x_fit":
            ADR_x_fit_ = config_value[i].strip('][').split(',')
            if len(ADR_x_fit_) == 4:
                ADR_x_fit_list.append(
                    [float(ADR_x_fit_[0]), float(ADR_x_fit_[1]), float(ADR_x_fit_[2]), float(ADR_x_fit_[3])])
            else:
                ADR_x_fit_list.append([float(ADR_x_fit_[0]), float(ADR_x_fit_[1]), float(ADR_x_fit_[2])])

        if config_property[i] == "ADR_y_fit":
            ADR_y_fit_ = config_value[i].strip('][').split(',')
            if len(ADR_y_fit_) == 4:
                ADR_y_fit_list.append(
                    [float(ADR_y_fit_[0]), float(ADR_y_fit_[1]), float(ADR_y_fit_[2]), float(ADR_y_fit_[3])])
            else:
                ADR_y_fit_list.append([float(ADR_y_fit_[0]), float(ADR_y_fit_[1]), float(ADR_y_fit_[2])])

        if config_property[i] == "jump": jump = int(config_value[i])

        if config_property[i] == "size_arcsec":
            size_arcsec_ = config_value[i].strip('][').split(',')
            for i in range(len(size_arcsec_)):
                size_arcsec.append(float(size_arcsec_[i]))

        if config_property[i] == "centre_deg":
            centre_deg_ = config_value[i].strip('][').split(',')
            centre_deg = [float(centre_deg_[0]), float(centre_deg_[1])]

        if config_property[i] == "delta_RA": delta_RA = float(config_value[i])
        if config_property[i] == "delta_DEC": delta_DEC = float(config_value[i])

        if config_property[i] == "scale_cubes_using_integflux":
            if config_value[i] == "True":
                scale_cubes_using_integflux = True
            else:
                scale_cubes_using_integflux = False

        if config_property[i] == "flux_ratios":
            flux_ratios_ = config_value[i].strip('][').split(',')
            flux_ratios = []
            for i in range(len(flux_ratios_)):
                flux_ratios.append(float(flux_ratios_[i]))

        if config_property[i] == "apply_scale":
            if config_value[i] == "True":
                apply_scale = True
            else:
                apply_scale = False

        if config_property[i] == "trim_cube":
            if config_value[i] == "True":
                trim_cube = True
            else:
                trim_cube = False

        if config_property[i] == "trim_values":
            trim_values_ = config_value[i].strip('][').split(',')
            trim_values = [int(trim_values_[0]), int(trim_values_[1]), int(trim_values_[2]), int(trim_values_[3])]

        if config_property[i] == "remove_spaxels_not_fully_covered":
            if config_value[i] == "True":
                remove_spaxels_not_fully_covered = True
            else:
                remove_spaxels_not_fully_covered = False

        if config_property[i] == "log":
            if config_value[i] == "True":
                log = True
            else:
                log = False

        if config_property[i] == "gamma": gamma = float(config_value[i])

        if config_property[i] == "fig_size": fig_size = float(config_value[i])

        if config_property[i] == "plot":
            if config_value[i] == "True":
                plot = True
            else:
                plot = False
        if config_property[i] == "plot_rss":
            if config_value[i] == "True":
                plot_rss = True
            else:
                plot_rss = False
        if config_property[i] == "plot_weight":
            if config_value[i] == "True":
                plot_weight = True
            else:
                plot_weight = False

        if config_property[i] == "warnings":
            if config_value[i] == "True":
                warnings = True
            else:
                warnings = False
        if config_property[i] == "verbose":
            if config_value[i] == "True":
                verbose = True
            else:
                verbose = False

            # Save rss list if requested:
    if save_rss:
        for i in range(len(rss_list)):
            save_rss_list.append("auto")

    if len(cube_list_names) < 1: cube_list_names = [""]

    # Asign names to variables
    # If files are given, they have preference over variables!

    if telluric_correction_file != "":
        w_star, telluric_correction = read_table(telluric_correction_file, ["f", "f"])
    if telluric_correction_name != "":
        exec("telluric_correction=" + telluric_correction_name)
    else:
        telluric_correction = [0]

    # Check that skyflat, flux and telluric lists are more than 1 element

    if len(skyflat_list) < 2: skyflat_list = ["", "", "", "", "", "", "", "", "", ""]  # CHECK THIS

    if len(telluric_correction_list_) < 2:
        telluric_correction_list = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    else:
        for i in range(len(telluric_correction_list_)):
            w_star, telluric_correction_ = read_table(telluric_correction_list_[i], ["f", "f"])
            telluric_correction_list.append(telluric_correction_)

    # More checks

    if len(box_x) < 1: box_x = [0, -1]
    if len(box_y) < 1: box_y = [0, -1]

    if do_cubing == False:
        fits_file = ""
        save_aligned_cubes = False
        make_combined_cube = False
        do_alignment = False

    if throughput_2D_variable != "":
        exec("throughput_2D = " + throughput_2D_variable)

    # Print the summary of parameters read with this script

    print("> Parameters for processing this object :\n")
    print("  obj_name                 = ", obj_name)
    print("  description              = ", description)
    print("  path                     = ", path)
    print("  Python_obj_name          = ", Python_obj_name)
    print("  date                     = ", date)
    print("  grating                  = ", grating)

    if read_cube == False:
        for rss in range(len(rss_list)):
            if len(rss_list) > 1:
                if rss == 0:
                    print("  rss_list                 = [", rss_list[rss], ",")
                else:
                    if rss == len(rss_list) - 1:
                        print("                              ", rss_list[rss], " ]")
                    else:
                        print("                              ", rss_list[rss], ",")
            else:
                print("  rss_list                 = [", rss_list[rss], "]")

        if rss_clean:
            print("  rss_clean                = ", rss_clean)
            print("  plot_rss                 = ", plot_rss)
        else:
            print("  apply_throughput         = ", apply_throughput)
            if apply_throughput:
                if throughput_2D_variable != "":
                    print("    throughput_2D variable = ", throughput_2D_variable)
                else:
                    if throughput_2D_file != "":
                        print("    throughput_2D_file     = ", throughput_2D_file)
                    else:
                        print("    Requested but no throughput 2D information provided !!!")
                if throughput_2D_wavecor:
                    print("    throughput_2D will be applied AFTER correcting CCD defects and fixing small wavelenghts")

            print("  correct_ccd_defects      = ", correct_ccd_defects)
            if correct_ccd_defects: print("    kernel_correct_ccd_defects = ", kernel_correct_ccd_defects)

            if fix_wavelengths:
                print("  fix_wavelengths          = ", fix_wavelengths)
                print("    sol                    = ", sol)

            print("  do_extinction            = ", do_extinction)

            if do_telluric_correction:
                print("  do_telluric_correction   = ", do_telluric_correction)
            else:
                if grating == "385R" or grating == "1000R" or grating == "2000R":
                    print("  do_telluric_correction   = ", do_telluric_correction)

            print("  sky_method               = ", sky_method)
            if sky_method == "1D" or sky_method == "1Dfit" or sky_method == "2D":
                for sky in range(len(sky_list)):
                    if sky == 0:
                        print("    sky_list               = [", sky_list[sky], ",")
                    else:
                        if sky == len(sky_list) - 1:
                            print("                              ", sky_list[sky], " ]")
                        else:
                            print("                              ", sky_list[sky], ",")
            if sky_spectrum[0] != -1 and sky_spectrum_name != "":
                print("    sky_spectrum_name      = ", sky_spectrum_name)
                if sky_method == "1Dfit" or sky_method == "selffit":
                    print("    ranges_with_emis_lines = ", ranges_with_emission_lines)
                    print("    cut_red_end            = ", cut_red_end)

            if sky_method == "1D" or sky_method == "1Dfit": print("    scale_sky_1D           = ", scale_sky_1D)
            if sky_spectrum[0] == -1 and len(sky_list) == 0: print_n_sky = True
            if sky_method == "self" or sky_method == "selffit": print_n_sky = True
            if print_n_sky:
                if len(sky_fibres) > 1:
                    print("    sky_fibres             = ", sky_fibres_print)
                else:
                    print("    n_sky                  = ", n_sky)

            if win_sky > 0: print("    win_sky                = ", win_sky)
            if auto_scale_sky: print("    auto_scale_sky         = ", auto_scale_sky)
            if remove_5577: print("    remove 5577 skyline    = ", remove_5577)
            print("  correct_negative_sky     = ", correct_negative_sky)
            if correct_negative_sky:
                print("    order_fit_negative_sky   = ", order_fit_negative_sky)
                print("    kernel_negative_sky      = ", kernel_negative_sky)
                print("    use_fit_for_negative_sky = ", use_fit_for_negative_sky)
                print("    low_fibres               = ", low_fibres)
                print("    individual_check         = ", individual_check)
                if sky_method in ["self", "selffit"]:  print("    force_sky_fibres_to_zero = ",
                                                             force_sky_fibres_to_zero)

            if sky_method == "1Dfit" or sky_method == "selffit" or id_el == True:
                if sky_lines_file != "": print("    sky_lines_file         = ", sky_lines_file)
                print("    brightest_line         = ", brightest_line)
                print("    brightest_line_wav     = ", brightest_line_wavelength)

            if id_el == True:  # NEED TO BE CHECKED
                print("  id_el                = ", id_el)
                print("    high_fibres            = ", high_fibres)
                print("    cut                    = ", cut)
                print("    broad                  = ", broad)
                print("    id_list                = ", id_list)
                print("    plot_id_el             = ", plot_id_el)

            if fix_edges: print("  fix_edges                = ", fix_edges)
            print("  clean_sky_residuals      = ", clean_sky_residuals)
            if clean_sky_residuals:
                if len(features_to_fix) > 0:
                    for feature in features_to_fix:
                        print("    feature_to_fix         = ", feature)
                else:
                    print("    No list with features_to_fix provided, using default list")
                print("    sky_fibres_residuals   = ", sky_fibres_for_residuals_print)

            print("  clean_cosmics            = ", clean_cosmics)
            if clean_cosmics:
                print("    width_bl               = ", width_bl)
                print("    kernel_median_cosmics  = ", kernel_median_cosmics)
                print("    cosmic_higher_than     = ", cosmic_higher_than)
                print("    extra_factor           = ", extra_factor)

            print("  clean_extreme_negatives  = ", clean_extreme_negatives)
            if clean_extreme_negatives:
                print("    percentile_min         = ", percentile_min)
            if remove_negative_median_values:
                print("  Negative pixels will be set to 0 when median value of spectrum is negative")

        if do_cubing:
            print(" ")
            print("  pixel_size               = ", pixel_size)
            print("  kernel_size              = ", kernel_size)

            if len(size_arcsec) > 0:  print("  cube_size_arcsec         = ", size_arcsec)
            if len(centre_deg) > 0:  print("  centre_deg               = ", centre_deg)

            if len(offsets) > 0:
                print("  offsets                  = ", offsets)
            else:
                print("  offsets will be calculated automatically")

            if half_size_for_centroid > 0: print("  half_size_for_centroid   = ", half_size_for_centroid)
            if np.nanmedian(box_x + box_y) != -0.5: print("  box_x, box_y             = ", box_x, box_y)
            print("  adr_index_fit            = ", adr_index_fit)
            print("  2D Gauss for tracing     = ", g2d)
            print("  step_tracing             = ", step_tracing)
            if len(plot_tracing_maps) > 0:
                print("  plot_tracing_maps        = ", plot_tracing_maps)

            if edgelow != -1: print("  edgelow for tracing      = ", edgelow)
            if edgehigh != -1: print("  edgehigh for tracing     = ", edgehigh)

            print("  ADR                      = ", ADR)
            print("  ADR in combined cube     = ", ADR_cc)
            print("  force_ADR                = ", force_ADR)
            if jump != -1: print("  jump for ADR             = ", jump)

            if len(ADR_x_fit_list) > 0:
                print("  Fitting solution for correcting ADR provided!")
                for i in range(len(rss_list)):
                    print("                           = ", ADR_x_fit_list[i])
                    print("                           = ", ADR_y_fit_list[i])
            else:
                if ADR: print("    adr_index_fit          = ", adr_index_fit)

            if delta_RA + delta_DEC != 0:
                print("  delta_RA                 = ", delta_RA)
                print("  delta_DEC                = ", delta_DEC)

            if valid_wave_min > 0: print("  valid_wave_min           = ", valid_wave_min)
            if valid_wave_max > 0: print("  valid_wave_max           = ", valid_wave_max)
            if trim_cube:
                print("  Trim cube                = ", trim_cube)
                print("     remove_spaxels_not_fully_covered = ", remove_spaxels_not_fully_covered)

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
                        print("    flux_calibration_file  = ", flux_calibration_file_list[i])
                else:
                    if flux_calibration_file != "":
                        print("    flux_calibration_file  = ", flux_calibration_file)
                    else:
                        print("  No flux calibration will be applied")
            else:
                print("  Variable with the flux calibration :", flux_calibration_name)

        if do_telluric_correction:
            if telluric_correction_name == "":
                if np.nanmedian(telluric_correction_list) != 0:
                    print("  Each rss file has a telluric correction file:")
                    for i in range(len(telluric_correction_list)):
                        print("  telluric_correction_file = ", telluric_correction_list_[i])
                else:
                    print("  telluric_correction_file = ", telluric_correction_file)
            else:
                print("  Variable with the telluric calibration :", telluric_correction_name)

    else:
        print("\n  List of ALIGNED cubes provided!")
        for cube in range(len(cube_list_names)):
            if cube == 0:
                print("  cube_list                = [", cube_list_names[cube], ",")
            else:
                if cube == len(cube_list_names) - 1:
                    print("                              ", cube_list_names[cube], " ]")
                else:
                    print("                              ", cube_list_names[cube], ",")

        print("  pixel_size               = ", pixel_size)
        print("  kernel_size              = ", kernel_size)
        if half_size_for_centroid > 0: print("  half_size_for_centroid   = ", half_size_for_centroid)
        if np.nanmedian(box_x + box_y) != -0.5: print("  box_x, box_y             = ", box_x, box_y)
        if jump != -1: print("  jump for ADR             = ", jump)
        if edgelow != -1: print("  edgelow for tracing      = ", edgelow)
        if edgehigh != -1: print("  edgehigh for tracing     = ", edgehigh)
        print("  ADR in combined cube     = ", ADR_cc)
        if valid_wave_min > 0: print("  valid_wave_min           = ", valid_wave_min)
        if valid_wave_max > 0: print("  valid_wave_max           = ", valid_wave_max)
        if trim_cube: print("  Trim cube                = ", trim_cube)
        make_combined_cube = True

    if make_combined_cube:
        if scale_cubes_using_integflux:
            if len(flux_ratios) == 0:
                print("  Scaling individual cubes using integrated flux of common region")
            else:
                print("  Scaling individual cubes using flux_ratios = ", flux_ratios)

    print("  plot                     = ", plot)
    if do_cubing or make_combined_cube:
        if plot_weight: print("  plot weights             = ", plot_weight)
    # if norm != "colors.LogNorm()" :  print("  norm                     = ",norm)
    if fig_size != 12.: print("  fig_size                 = ", fig_size)
    print("  warnings                 = ", warnings)
    if verbose == False: print("  verbose                  = ", verbose)

    print("\n> Output files:\n")
    if fits_file != "": print("  fits file with combined cube  =  ", fits_file)

    if read_cube == False:
        if len(save_rss_list) > 0 and rss_clean == False:
            for rss in range(len(save_rss_list)):
                if save_rss_list[0] != "auto":
                    if rss == 0:
                        print("  list of saved rss files       = [", save_rss_list[rss], ",")
                    else:
                        if rss == len(save_rss_list) - 1:
                            print("                                   ", save_rss_list[rss], " ]")
                        else:
                            print("                                   ", save_rss_list[rss], ",")
                else:
                    print("  Processed rss files will be saved using automatic naming")

        else:
            save_rss_list = ["", "", "", "", "", "", "", "", "", ""]
        if save_aligned_cubes:
            print("  Individual cubes will be saved as fits files")

        # Last little checks...
        if len(sky_list) == 0: sky_list = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        if fix_wavelengths == False: sol = [0, 0, 0]
        if len(ADR_x_fit_list) == 0:
            for i in range(len(rss_list)):
                ADR_x_fit_list.append([0])
                ADR_y_fit_list.append([0])

        # Values for improving alignment:   # TODO: CHECK THIS!!! Now I think it is wrong!!!
        if delta_RA + delta_DEC != 0:
            for i in range(len(ADR_x_fit_list)):
                ADR_x_fit_list[i][2] = ADR_x_fit_list[i][2] + (delta_RA / 2)
                ADR_y_fit_list[i][2] = ADR_y_fit_list[i][2] + (delta_DEC / 2)


    # Now run KOALA_reduce
    # hikids = KOALA_reduce(rss_list,
    #                       obj_name=obj_name,  description=description,
    #                       fits_file=fits_file,
    #                       rss_clean=rss_clean,
    #                       save_rss_to_fits_file_list=save_rss_list,
    #                       save_aligned_cubes = save_aligned_cubes,
    #                       cube_list_names = cube_list_names,
    #                       apply_throughput=apply_throughput,
    #                       throughput_2D = throughput_2D,
    #                       throughput_2D_file = throughput_2D_file,
    #                       throughput_2D_wavecor = throughput_2D_wavecor,
    #                       #skyflat_list = skyflat_list,
    #                       correct_ccd_defects = correct_ccd_defects,
    #                       kernel_correct_ccd_defects=kernel_correct_ccd_defects,
    #                       fix_wavelengths = fix_wavelengths,
    #                       sol = sol,
    #                       do_extinction=do_extinction,

    #                       telluric_correction = telluric_correction,
    #                       telluric_correction_list = telluric_correction_list,
    #                       telluric_correction_file = telluric_correction_file,

    #                       sky_method=sky_method,
    #                       sky_list=sky_list,
    #                       n_sky = n_sky,
    #                       sky_fibres=sky_fibres,
    #                       win_sky = win_sky,
    #                       scale_sky_1D = scale_sky_1D,
    #                       sky_lines_file=sky_lines_file,
    #                       ranges_with_emission_lines=ranges_with_emission_lines,
    #                       cut_red_end =cut_red_end,
    #                       remove_5577=remove_5577,
    #                       auto_scale_sky = auto_scale_sky,
    #                       correct_negative_sky = correct_negative_sky,
    #                       order_fit_negative_sky =order_fit_negative_sky,
    #                       kernel_negative_sky = kernel_negative_sky,
    #                       individual_check = individual_check,
    #                       use_fit_for_negative_sky = use_fit_for_negative_sky,
    #                       force_sky_fibres_to_zero = force_sky_fibres_to_zero,
    #                       high_fibres = high_fibres,
    #                       low_fibres=low_fibres,

    #                       brightest_line=brightest_line,
    #                       brightest_line_wavelength = brightest_line_wavelength,
    #                       id_el = id_el,
    #                       id_list=id_list,
    #                       cut = cut, broad = broad, plot_id_el = plot_id_el,

    #                       clean_sky_residuals = clean_sky_residuals,
    #                       features_to_fix = features_to_fix,
    #                       sky_fibres_for_residuals = sky_fibres_for_residuals,

    #                       fix_edges=fix_edges,
    #                       clean_extreme_negatives=clean_extreme_negatives, percentile_min=percentile_min,
    #                       remove_negative_median_values=remove_negative_median_values,
    #                       clean_cosmics = clean_cosmics,
    #                       width_bl = width_bl, kernel_median_cosmics = kernel_median_cosmics,
    #                       cosmic_higher_than = cosmic_higher_than, extra_factor = extra_factor,

    #                       do_cubing= do_cubing,
    #                       do_alignment = do_alignment,
    #                       make_combined_cube = make_combined_cube,

    #                       pixel_size_arcsec=pixel_size,
    #                       kernel_size_arcsec=kernel_size,
    #                       offsets = offsets,    # EAST-/WEST+  NORTH-/SOUTH+

    #                       ADR=ADR, ADR_cc=ADR_cc, force_ADR=force_ADR,
    #                       box_x=box_x, box_y=box_y,
    #                       jump = jump,
    #                       half_size_for_centroid = half_size_for_centroid,
    #                       ADR_x_fit_list = ADR_x_fit_list, ADR_y_fit_list = ADR_y_fit_list,
    #                       adr_index_fit = adr_index_fit,
    #                       g2d = g2d,
    #                       plot_tracing_maps = plot_tracing_maps,
    #                       step_tracing=step_tracing,
    #                       edgelow=edgelow, edgehigh = edgehigh,

    #                       flux_calibration_file = flux_calibration_file_list,     # this can be a single file (string) or a list of files (list of strings)
    #                       flux_calibration=flux_calibration,                      # an array
    #                       flux_calibration_list  = flux_calibration_list          # a list of arrays

    #                       trim_cube = trim_cube,
    #                       trim_values = trim_values,
    #                       size_arcsec=size_arcsec,
    #                       centre_deg = centre_deg,
    #                       scale_cubes_using_integflux = scale_cubes_using_integflux,
    #                       apply_scale = apply_scale,
    #                       flux_ratios = flux_ratios,
    #                       #cube_list_names =cube_list_names,

    #                       valid_wave_min = valid_wave_min, valid_wave_max = valid_wave_max,
    #                       fig_size = fig_size,
    #                       norm = norm,
    #                       plot=plot, plot_rss = plot_rss, plot_weight=plot_weight, plot_spectra =plot_spectra,
    #                       warnings=warnings, verbose=verbose)

    else:
        print("else")
        # hikids = build_combined_cube(cube_list, obj_name=obj_name, description=description,
        #                              fits_file = fits_file, path=path,
        #                              scale_cubes_using_integflux= scale_cubes_using_integflux,
        #                              flux_ratios = flux_ratios, apply_scale = apply_scale,
        #                              edgelow=edgelow, edgehigh=edgehigh,
        #                              ADR=ADR, ADR_cc = ADR_cc, jump = jump, pk = pk,
        #                              ADR_x_fit_list=ADR_x_fit_list, ADR_y_fit_list=ADR_y_fit_list,
        #                              force_ADR=force_ADR,
        #                              half_size_for_centroid = half_size_for_centroid,
        #                              box_x=box_x, box_y=box_y,
        #                              adr_index_fit=adr_index_fit, g2d=g2d,
        #                              step_tracing = step_tracing,
        #                              plot_tracing_maps = plot_tracing_maps,
        #                              trim_cube = trim_cube,  trim_values =trim_values,
        #                              remove_spaxels_not_fully_covered = remove_spaxels_not_fully_covered,
        #                              plot=plot, plot_weight= plot_weight, plot_spectra=plot_spectra,
        #                              verbose=verbose, say_making_combined_cube= False)

    #    if Python_obj_name != 0: exec(Python_obj_name+"=copy.deepcopy(hikids)", globals())

    print("> automatic_KOALA_reduce script completed !!!")
    print("\n  Python object created :", Python_obj_name)
    if fits_file != "": print("  Fits file created     :", fits_file)


#    return hikids
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#    MACRO FOR EVERYTHING 19 Sep 2019, including alignment n - cubes
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class KOALA_reduce(RSS, Interpolated_cube):  # TASK_KOALA_reduce

    def __init__(self, rss_list, fits_file="", obj_name="", description="", path="",
                 do_rss=True, do_cubing=True, do_alignment=True, make_combined_cube=True, rss_clean=False,
                 save_aligned_cubes=False, save_rss_to_fits_file_list=[],  # ["","","","","","","","","",""],
                 # RSS
                 flat="",
                 grating="",
                 # skyflat_file is a RSS, skyflat and skyflat_list are the names of objects keeping the relative throughput of skyflats
                 apply_throughput=True,
                 throughput_2D=[], throughput_2D_file="",
                 throughput_2D_wavecor=False,
                 # nskyflat=True, skyflat = "", skyflat_file ="",throughput_file ="", nskyflat_file="",
                 # skyflat_list=["","","","","","","","","",""],
                 # This line is needed if doing FLAT when reducing (NOT recommended)
                 # plot_skyflat=False, wave_min_scale=0, wave_max_scale=0, ymin=0, ymax=0,
                 # Correct CCD defects & high cosmics
                 correct_ccd_defects=False, remove_5577=False, kernel_correct_ccd_defects=51,
                 plot_suspicious_fibres=False,
                 # Correct for small shofts in wavelength
                 fix_wavelengths=False, sol=[0, 0, 0],
                 # Correct for extinction
                 do_extinction=True,
                 # Telluric correction
                 telluric_correction=[0], telluric_correction_file="",
                 telluric_correction_list=[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                 # Sky substraction
                 sky_method="self", n_sky=50, sky_fibres=[], win_sky=0,
                 sky_spectrum=[], sky_rss=[0], scale_sky_rss=0, scale_sky_1D=0,
                 sky_spectrum_file="", sky_spectrum_file_list=["", "", "", "", "", "", "", "", "", ""],
                 sky_list=[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                 ranges_with_emission_lines=[0],
                 cut_red_end=0,

                 correct_negative_sky=False,
                 order_fit_negative_sky=3, kernel_negative_sky=51, individual_check=True,
                 use_fit_for_negative_sky=False,
                 force_sky_fibres_to_zero=True, high_fibres=20, low_fibres=10,
                 auto_scale_sky=False,
                 brightest_line="Ha", brightest_line_wavelength=0, sky_lines_file="",
                 is_sky=False, sky_wave_min=0, sky_wave_max=0, cut_sky=5., fmin=1, fmax=10,
                 individual_sky_substraction=False, fibre_list=[100, 200, 300, 400, 500, 600, 700, 800, 900],
                 # Identify emission lines
                 id_el=False, cut=1.5, plot_id_el=True, broad=2.0, id_list=[0],
                 # Clean sky residuals
                 fibres_to_fix=[],
                 clean_sky_residuals=False, features_to_fix=[], sky_fibres_for_residuals=[],
                 remove_negative_median_values=False,
                 fix_edges=False,
                 clean_extreme_negatives=False, percentile_min=0.5,
                 clean_cosmics=False,
                 # show_cosmics_identification = True,
                 width_bl=20., kernel_median_cosmics=5, cosmic_higher_than=100., extra_factor=1.,

                 # CUBING
                 pixel_size_arcsec=.4, kernel_size_arcsec=1.2,
                 offsets=[],
                 ADR=False, ADR_cc=False, force_ADR=False,
                 box_x=[0, -1], box_y=[0, -1], jump=-1, half_size_for_centroid=10,
                 ADR_x_fit_list=[], ADR_y_fit_list=[], adr_index_fit=2,
                 g2d=False,
                 plot_tracing_maps=[],
                 step_tracing=100,
                 edgelow=-1, edgehigh=-1,
                 flux_calibration_file="",  # this can be a single file (string) or a list of files (list of strings)
                 flux_calibration=[],  # an array
                 flux_calibration_list=[],  # a list of arrays
                 trim_cube=True, trim_values=[],
                 remove_spaxels_not_fully_covered=True,
                 size_arcsec=[],
                 centre_deg=[],
                 scale_cubes_using_integflux=True,
                 apply_scale=True,
                 flux_ratios=[],
                 cube_list_names=[""],

                 # COMMON TO RSS AND CUBING & PLOTS
                 valid_wave_min=0, valid_wave_max=0,
                 log=True,  # If True and gamma = 0, use colors.LogNorm() [LOG], if False colors.Normalize() [LINEAL]
                 gamma=0,
                 plot=True, plot_rss=True, plot_weight=False, plot_spectra=True, fig_size=12,
                 warnings=False, verbose=False):  # norm
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
        sky_rss_list = []
        pk = "_" + str(int(pixel_size_arcsec)) + "p" + str(
            int((abs(pixel_size_arcsec) - abs(int(pixel_size_arcsec))) * 10)) + "_" + str(
            int(kernel_size_arcsec)) + "k" + str(int(abs(kernel_size_arcsec * 100)) - int(kernel_size_arcsec) * 100)

        if plot == False:
            plot_rss = False
            if verbose: print("No plotting anything.....\n")

        # if plot_rss == False and plot == True and verbose: print(" plot_rss is false.....")

        print("1. Checking input values:")

        print("\n  - Using the following RSS files : ")
        rss_object = []
        cube_object = []
        cube_aligned_object = []
        number = 1

        for rss in range(n_files):
            rss_list[rss] = full_path(rss_list[rss], path)
            print("    ", rss + 1, ". : ", rss_list[rss])
            _rss_ = "self.rss" + np.str(number)
            _cube_ = "self.cube" + np.str(number)
            _cube_aligned_ = "self.cube" + np.str(number) + "_aligned"
            rss_object.append(_rss_)
            cube_object.append(_cube_)
            cube_aligned_object.append(_cube_aligned_)
            number = number + 1
            sky_rss_list.append([0])

        if len(save_rss_to_fits_file_list) > 0:
            try:
                if save_rss_to_fits_file_list == "auto":
                    save_rss_to_fits_file_list = []
                    for rss in range(n_files):
                        save_rss_to_fits_file_list.append("auto")
            except Exception:
                if len(save_rss_to_fits_file_list) != len(n_files) and verbose: print(
                    "  WARNING! List of rss files to save provided does not have the same number of rss files!!!")

        else:
            for rss in range(n_files):
                save_rss_to_fits_file_list.append("")

        self.rss_list = rss_list

        if number == 1:
            do_alignment = False
            make_combined_cube = False

        if rss_clean:
            print("\n  - These RSS files are ready to be cubed & combined, no further process required ...")
        else:
            # Check throughput
            if apply_throughput:
                if len(throughput_2D) == 0 and throughput_2D_file == "":
                    print(
                        "\n\n\n  WARNING !!!! \n\n  No 2D throughput data provided, no throughput correction will be applied.\n\n\n")
                    apply_throughput = False
                else:
                    if len(throughput_2D) > 0:
                        print("\n  - Using the variable provided for applying the 2D throughput correction ...")
                    else:
                        print("\n  - The 2D throughput correction will be applied using the file:")
                        print("  ", throughput_2D_file)
            else:
                print("\n  - No 2D throughput correction will be applied")

            # sky_method = "self" "1D" "2D" "none" #1Dfit" "selffit"

            if sky_method == "1D" or sky_method == "1Dfit" or sky_method == "selffit":
                if np.nanmedian(sky_spectrum) != -1 and np.nanmedian(sky_spectrum) != 0:
                    for i in range(n_files):
                        sky_list[i] = sky_spectrum
                    print("\n  - Using same 1D sky spectrum provided for all object files")
                else:
                    if np.nanmedian(sky_list[0]) == 0:
                        print("\n  - 1D sky spectrum requested but not found, assuming n_sky =", n_sky,
                              "from the same files")
                        if sky_method in ["1Dfit", "1D"]: sky_method = "self"
                    else:
                        print("\n  - List of 1D sky spectrum provided for each object file")

            if sky_method == "2D":
                try:
                    if np.nanmedian(sky_list[0].intensity_corrected) != 0:
                        print("\n  - List of 2D sky spectra provided for each object file")
                        for i in range(n_files):
                            sky_rss_list[i] = sky_list[i]
                            sky_list[i] = [0]
                except Exception:
                    try:
                        if sky_rss == 0:
                            print(
                                "\n  - 2D sky spectra requested but not found, assuming n_sky = 50 from the same files")
                            sky_method = "self"
                    except Exception:
                        for i in range(n_files):
                            sky_rss_list[i] = sky_rss
                        print("\n  - Using same 2D sky spectra provided for all object files")

            if sky_method == "self":  # or  sky_method == "selffit":
                for i in range(n_files):
                    sky_list[i] = []
                if n_sky == 0: n_sky = 50
                if len(sky_fibres) == 0:
                    print("\n  - Using n_sky =", n_sky, "to create a sky spectrum")
                else:
                    print("\n  - Using n_sky =", n_sky, "and sky_fibres =", sky_fibres, "to create a sky spectrum")

            if grating in red_gratings:
                if np.nanmedian(telluric_correction) == 0 and np.nanmedian(telluric_correction_list[0]) == 0:
                    print("\n  - No telluric correction considered")
                else:
                    if np.nanmedian(telluric_correction_list[0]) == 0:
                        for i in range(n_files):
                            telluric_correction_list[i] = telluric_correction
                        print("\n  - Using same telluric correction for all object files")
                    else:
                        print("\n  - List of telluric corrections provided!")

        if do_rss:
            print("\n-------------------------------------------")
            print("2. Reading the data stored in rss files ...")

            for i in range(n_files):
                # skyflat=skyflat_list[i], plot_skyflat=plot_skyflat, throughput_file =throughput_file, nskyflat_file=nskyflat_file,\
                # This considers the same throughput for ALL FILES !!
                exec(rss_object[i] + '= KOALA_RSS(rss_list[i], rss_clean = rss_clean, save_rss_to_fits_file = save_rss_to_fits_file_list[i], \
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
                     width_bl = width_bl, kernel_median_cosmics = kernel_median_cosmics, cosmic_higher_than = cosmic_higher_than, extra_factor = extra_factor,\
                     valid_wave_min=valid_wave_min, valid_wave_max=valid_wave_max,\
                     warnings=warnings, verbose = verbose, plot=plot_rss, plot_final_rss=plot_rss, fig_size=fig_size)')

        if len(offsets) > 0 and len(ADR_x_fit_list) > 0 and ADR == True:
            # print("\n  Offsets values for alignment AND fitting for ADR correction have been provided, skipping cubing no-aligned rss...")
            do_cubing = False
        elif len(offsets) > 0 and ADR == False:
            # print("\n  Offsets values for alignment given AND the ADR correction is NOT requested, skipping cubing no-aligned rss...")
            do_cubing = False

        if len(ADR_x_fit_list) == 0:  # Check if lists with ADR values have been provided, if not create lists with 0
            ADR_x_fit_list = []
            ADR_y_fit_list = []
            for i in range(n_files):
                ADR_x_fit_list.append([0, 0, 0])
                ADR_y_fit_list.append([0, 0, 0])

        fcal = False
        if flux_calibration_file != "":  # If files have been provided for the flux calibration, we read them
            fcal = True
            if type(flux_calibration_file) == str:
                if path != "": flux_calibration_file = full_path(flux_calibration_file, path)
                w_star, flux_calibration = read_table(flux_calibration_file, ["f", "f"])
                for i in range(n_files):
                    flux_calibration_list.append(flux_calibration)

                if verbose: print("\n  - Using for all the cubes the same flux calibration provided in file:\n   ",
                                  flux_calibration_file)
            else:
                if verbose: print("\n  - Using list of files for flux calibration:")
                for i in range(n_files):
                    if path != "": flux_calibration_file[i] = full_path(flux_calibration_file[i], path)
                    print("   ", flux_calibration_file[i])
                    w_star, flux_calibration = read_table(flux_calibration_file[i], ["f", "f"])
                    flux_calibration_list.append(flux_calibration)
        else:
            if len(flux_calibration) > 0:
                fcal = True
                for i in range(n_files):
                    flux_calibration_list.append(flux_calibration)
                if verbose: print("\n  - Using same flux calibration for all object files")
            else:
                if verbose or warning: print("\n  - No flux calibration provided!")
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
                exec(cube_object[i] + '=Interpolated_cube(' + rss_object[i] + ', pixel_size_arcsec=pixel_size_arcsec, kernel_size_arcsec=kernel_size_arcsec, plot=plot, half_size_for_centroid=half_size_for_centroid,\
                     ADR_x_fit = ADR_x_fit_list[i], ADR_y_fit = ADR_y_fit_list[i], box_x=box_x, box_y=box_y, plot_spectra=plot_spectra, \
                     adr_index_fit=adr_index_fit, g2d=g2d, plot_tracing_maps = plot_tracing_maps, step_tracing=step_tracing,  ADR=ADR, apply_ADR = False, \
                     flux_calibration=flux_calibration_list[i], edgelow=edgelow, edgehigh=edgehigh, size_arcsec=size_arcsec, centre_deg=centre_deg,warnings=warnings)')
        else:
            if do_alignment:
                print("\n------------------------------------------------")
                if ADR == False:
                    print(
                        "3. Offsets provided, ADR correction NOT requested, cubing will be done using aligned cubes ...")
                else:
                    print("3. Offsets AND correction for ADR provided, cubing will be done using aligned cubes ...")

        rss_list_to_align = []
        cube_list = []
        for i in range(n_files):
            exec('rss_list_to_align.append(' + rss_object[i] + ')')
            if do_cubing:
                exec('cube_list.append(' + cube_object[i] + ')')
            else:
                cube_list.append([0])

        if do_alignment:
            if len(offsets) == 0:
                print("\n--------------------------------")
                print("4. Aligning individual cubes ...")
            else:
                print("\n-----------------------------------------------------")
                print("4. Checking offsets data provided and performing cubing ...")

            cube_aligned_list = align_n_cubes(rss_list_to_align, cube_list=cube_list,
                                              flux_calibration_list=flux_calibration_list,
                                              pixel_size_arcsec=pixel_size_arcsec,
                                              kernel_size_arcsec=kernel_size_arcsec, plot=plot, plot_weight=plot_weight,
                                              offsets=offsets, ADR=ADR, jump=jump, edgelow=edgelow, edgehigh=edgehigh,
                                              size_arcsec=size_arcsec, centre_deg=centre_deg,
                                              half_size_for_centroid=half_size_for_centroid, box_x=box_x, box_y=box_y,
                                              ADR_x_fit_list=ADR_x_fit_list, ADR_y_fit_list=ADR_y_fit_list,
                                              adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing,
                                              plot_tracing_maps=plot_tracing_maps, plot_spectra=plot_spectra,
                                              force_ADR=force_ADR, warnings=warnings)

            for i in range(n_files):
                exec(cube_aligned_object[i] + '=cube_aligned_list[i]')

        else:

            if ADR == True and np.nanmedian(ADR_x_fit_list) == 0:
                # If not alignment but ADR is requested

                print("\n--------------------------------")
                print("4. Applying ADR ...")

                for i in range(n_files):
                    exec(cube_object[i] + '=Interpolated_cube(' + rss_object[i] + ', pixel_size_arcsec, kernel_size_arcsec, plot=plot, half_size_for_centroid=half_size_for_centroid,\
                         ADR_x_fit = cube_list[i].ADR_x_fit, ADR_y_fit = cube_list[i].ADR_y_fit, box_x=box_x, box_y=box_y, check_ADR = True, \
                         flux_calibration=flux_calibration_list[i], edgelow=edgelow, edgehigh=edgehigh, size_arcsec=size_arcsec, centre_deg=centre_deg,warnings=warnings)')

                    # Save aligned cubes to fits files
        if save_aligned_cubes:
            print("\n> Saving aligned cubes to fits files ...")
            if cube_list_names[0] == "":
                for i in range(n_files):
                    if i < 9:
                        replace_text = "_" + obj_name + "_aligned_cube_0" + np.str(i + 1) + pk + ".fits"
                    else:
                        replace_text = "_aligned_cube_" + np.str(i + 1) + pk + ".fits"

                    aligned_cube_name = rss_list[i].replace(".fits", replace_text)
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
                        print("ADR_x_fit  = [" + np.str(cube_list[i].ADR_x_fit[0]) + "," + np.str(
                            cube_list[i].ADR_x_fit[1]) + "]")
                        print("ADR_y_fit  = [" + np.str(cube_list[i].ADR_y_fit[0]) + "," + np.str(
                            cube_list[i].ADR_y_fit[1]) + "]")
                    elif adr_index_fit == 2:
                        print("ADR_x_fit  = [" + np.str(cube_list[i].ADR_x_fit[0]) + "," + np.str(
                            cube_list[i].ADR_x_fit[1]) + "," + np.str(cube_list[i].ADR_x_fit[2]) + "]")
                        print("ADR_y_fit  = [" + np.str(cube_list[i].ADR_y_fit[0]) + "," + np.str(
                            cube_list[i].ADR_y_fit[1]) + "," + np.str(cube_list[i].ADR_y_fit[2]) + "]")
                    elif adr_index_fit == 3:
                        print("ADR_x_fit  = [" + np.str(cube_list[i].ADR_x_fit[0]) + "," + np.str(
                            cube_list[i].ADR_x_fit[1]) + "," + np.str(cube_list[i].ADR_x_fit[2]) + "," + np.str(
                            cube_list[i].ADR_x_fit[3]) + "]")
                        print("ADR_y_fit  = [" + np.str(cube_list[i].ADR_y_fit[0]) + "," + np.str(
                            cube_list[i].ADR_y_fit[1]) + "," + np.str(cube_list[i].ADR_y_fit[2]) + "," + np.str(
                            cube_list[i].ADR_y_fit[3]) + "]")

                except Exception:
                    print("  WARNING: Something wrong happened printing the ADR fit values! Results are:")
                    print("  ADR_x_fit  = ", cube_list[i].ADR_x_fit)
                    print("  ADR_y_fit  = ", cube_list[i].ADR_y_fit)

                _x_ = []
                _y_ = []
                for j in range(len(cube_list[i].ADR_x_fit)):
                    _x_.append(cube_list[i].ADR_x_fit[j])
                    _y_.append(cube_list[i].ADR_y_fit[j])
                ADR_x_fit_list.append(_x_)
                ADR_y_fit_list.append(_y_)

        if obj_name == "":
            exec('obj_name = ' + rss_object[0] + '.object')
            obj_name = obj_name.replace(" ", "_")

        if make_combined_cube and n_files > 1:
            print("\n---------------------------")
            print("5. Making combined cube ...")

            self.combined_cube = build_combined_cube(cube_aligned_list, obj_name=obj_name, description=description,
                                                     fits_file=fits_file, path=path,
                                                     scale_cubes_using_integflux=scale_cubes_using_integflux,
                                                     flux_ratios=flux_ratios, apply_scale=apply_scale,
                                                     edgelow=edgelow, edgehigh=edgehigh,
                                                     ADR=ADR, ADR_cc=ADR_cc, jump=jump, pk=pk,
                                                     ADR_x_fit_list=ADR_x_fit_list, ADR_y_fit_list=ADR_y_fit_list,
                                                     force_ADR=force_ADR,
                                                     half_size_for_centroid=half_size_for_centroid,
                                                     box_x=box_x, box_y=box_y,
                                                     adr_index_fit=adr_index_fit, g2d=g2d,
                                                     step_tracing=step_tracing,
                                                     plot_tracing_maps=plot_tracing_maps,
                                                     trim_cube=trim_cube, trim_values=trim_values,
                                                     remove_spaxels_not_fully_covered=remove_spaxels_not_fully_covered,
                                                     plot=plot, plot_weight=plot_weight, plot_spectra=plot_spectra,
                                                     verbose=verbose, say_making_combined_cube=False)
        else:
            if n_files > 1:
                if do_alignment == False and do_cubing == False:
                    print("\n> As requested, skipping cubing...")
                else:
                    print("\n  No combined cube obtained!")

            else:
                print("\n> Only one file provided, no combined cube obtained")
                # Trimming cube if requested or needed
                cube_aligned_list[0].trim_cube(trim_cube=trim_cube, trim_values=trim_values, ADR=ADR,
                                               half_size_for_centroid=half_size_for_centroid,
                                               adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing,
                                               plot_tracing_maps=plot_tracing_maps,
                                               remove_spaxels_not_fully_covered=remove_spaxels_not_fully_covered,
                                               box_x=box_x, box_y=box_y, edgelow=edgelow, edgehigh=edgehigh,
                                               plot_weight=plot_weight, fcal=fcal, plot=plot)

                # Make combined cube = aligned cube

                self.combined_cube = cube_aligned_list[0]

        self.parameters = locals()

        print("\n================== REDUCING KOALA DATA COMPLETED ====================\n\n")


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

# TESTING TASKS

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def create_map(cube, line, w2=0., gaussian_fit=False, gf=False,
               lowlow=50, lowhigh=10, highlow=10, highhigh=50, no_nans=False,
               show_spaxels=[], verbose=True, description=""):
    if gaussian_fit or gf:

        if description == "":
            description = "{} - Gaussian fit to {}".format(cube.description, line)
            description = description + " $\mathrm{\AA}$"

        map_flux = np.zeros((cube.n_rows, cube.n_cols))
        map_vel = np.zeros((cube.n_rows, cube.n_cols))
        map_fwhm = np.zeros((cube.n_rows, cube.n_cols))
        map_ew = np.zeros((cube.n_rows, cube.n_cols))

        n_fits = cube.n_rows * cube.n_cols
        w = cube.wavelength
        if verbose: print("\n> Fitting emission line", line, "A in cube ( total = ", n_fits, "fits) ...")

        # For showing fits
        show_x = []
        show_y = []
        name_list = []
        for i in range(len(show_spaxels)):
            show_x.append(show_spaxels[i][0])
            show_y.append(show_spaxels[i][1])
            name_list_ = "[" + np.str(show_x[i]) + "," + np.str(show_y[i]) + "]"
            name_list.append(name_list_)

        empty_spaxels = []
        fit_failed_list = []
        for x in range(cube.n_rows):
            for y in range(cube.n_cols):
                plot_fit = False
                verbose_fit = False
                warnings_fit = False

                for i in range(len(show_spaxels)):
                    if x == show_x[i] and y == show_y[i]:
                        if verbose: print("\n  - Showing fit and results for spaxel", name_list[i], ":")
                        plot_fit = True
                        if verbose: verbose_fit = True
                        if verbose: warnings_fit = True

                spectrum = cube.data[:, x, y]

                if np.isnan(np.nanmedian(spectrum)):
                    if verbose_fit: print("  SPAXEL ", x, y, " is empty! Skipping Gaussian fit...")
                    resultado = [np.nan] * 10
                    empty_spaxel = [x, y]
                    empty_spaxels.append(empty_spaxel)

                else:
                    resultado = fluxes(w, spectrum, line, lowlow=lowlow, lowhigh=lowhigh, highlow=highlow,
                                       highhigh=highhigh,
                                       plot=plot_fit, verbose=verbose_fit, warnings=warnings_fit)
                map_flux[x][y] = resultado[3]  # Gaussian Flux, use 7 for integrated flux
                map_vel[x][y] = resultado[1]
                map_fwhm[x][y] = resultado[5] * C / resultado[1]  # In km/s
                map_ew[x][y] = resultado[9]  # In \AA
                # checking than resultado[3] is NOT 0 (that means the Gaussian fit has failed!)
                if resultado[3] == 0:
                    map_flux[x][y] = np.nan
                    map_vel[x][y] = np.nan
                    map_fwhm[x][y] = np.nan
                    map_ew[x][y] = np.nan
                    # if verbose_fit: print "  Gaussian fit has FAILED in SPAXEL ",x,y,"..."
                    fit_failed = [x, y]
                    fit_failed_list.append(fit_failed)

        median_velocity = np.nanmedian(map_vel)
        map_vel = C * (map_vel - median_velocity) / median_velocity

        if verbose:
            # print "\n> Summary of Gaussian fitting : "
            print("\n> Found ", len(empty_spaxels), " the list with empty spaxels is ", empty_spaxels)
            print("  Gaussian fit FAILED in", len(fit_failed_list), " spaxels = ", fit_failed_list)
            print("\n> Returning [map_flux, map_vel, map_fwhm, map_ew, description] ")
        return description, map_flux, map_vel, map_fwhm, map_ew

    else:
        w1 = line
        if w2 == 0.:
            if verbose: print("\n> Creating map using channel closest to ", w1)
            interpolated_map = cube.data[np.searchsorted(cube.wavelength, w1)]
            descr = "{} - only {} ".format(cube.description, w1)

        else:
            if verbose: print("\n> Creating map integrating [{}-{}]".format(w1, w2))
            interpolated_map = np.nansum(
                cube.data[np.searchsorted(cube.wavelength, w1):np.searchsorted(cube.wavelength, w2)], axis=0)
            descr = "{} - Integrating [{}-{}] ".format(cube.description, w1, w2)

        if description == "": description = descr + "$\mathrm{\AA}$"
        if verbose: print("  Description =", description)
        # All 0 values should be nan
        if no_nans == False: interpolated_map[interpolated_map == 0] = np.nan
        return description, interpolated_map, w1, w2


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def create_mask(mapa, low_limit, high_limit=1E20, plot=False, verbose=True):
    n_rows = mapa.shape[0]
    n_cols = mapa.shape[1]

    mask = np.ones((n_rows, n_cols))

    for x in range(n_rows):
        for y in range(n_cols):
            value = mapa[x, y]
            if value < low_limit or value > high_limit:
                mask[x][y] = np.nan

    if verbose: print("\n> Mask with good values between", low_limit, "and", high_limit, "created!")

    return mask, low_limit, high_limit


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


# THINGS STILL TO DO:

# - save the mask with the information of all good spaxels when combining cubes
# - Include automatic naming in KOALA_RSS
# - CHECK and include exposition time when scaling cubes with integrated flux
# - Make "telluric_correction_file" be "telluric_file" in automatic_KOALA
# - When doing selffit in sky, do not consider skylines that are within the mask
# - Check size_arc and centre_degree when making mosaics...
# - INCLUDE THE VARIANCE!!!


# Stage 2:
#
# - Use data from DIFFERENT nights (check self.wavelength)
# - Combine 1000R + 385R data
# - Mosaiquing (it basically works)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

print("\n> PyKOALA", version, "read !!")
