from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import matplotlib.pyplot as plt


def plot_redshift_peaks(fig_size,
                        funcion,
                        wavelength,
                        lmin,
                        lmax,
                        fmin,
                        fmax,
                        cut,
                        peaks,
                        peaks_name,
                        label):
    """
    Check if all redshifts are similar, assuming check_redshift = 0.0003 by default
    # If OK, add id_peaks[i]=1, if not, id_peaks[i]=0

    This function plots after the above ^ is performed :)
    """
    fig, ax = plt.subplots(figsize=(fig_size, fig_size / 2.5))
    ax.plot(wavelength, funcion, "r", lw=1, alpha=0.5)
    
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel("Flux / continuum")

    ax.set_xlim(lmin, lmax)
    ax.set_ylim(fmin, fmax)
    ax.axhline(y=cut, color="k", linestyle=":", alpha=0.5)
    for i in range(len(peaks)):
        ax.axvline(x=peaks[i], color="k", linestyle=":", alpha=0.5)
        label = peaks_name[i]
        ax.text(peaks[i], 1.8, label)
    
    plt.show()
    return fig


def plot_weights_for_getting_smooth_spectrum(wlm,
                                             s,
                                             running_wave,
                                             running_step_median,
                                             fit_median,
                                             fit_median_interpolated,
                                             weight_fit_median,
                                             wave_min,
                                             wave_max,
                                             exclude_wlm):
    """
    Weights for getting smooth spectrum
    """
    fig_size = 12
    fig, ax = plt.subplots(figsize=(fig_size, fig_size / 2.5))
    ax.plot(wlm, s, alpha=0.5)
    ax.plot(running_wave, running_step_median, "+", ms=15, mew=3)
    ax.plot(wlm, fit_median, label="fit median")
    ax.plot(wlm, fit_median_interpolated, label="fit median_interp")
    ax.plot(wlm, weight_fit_median * fit_median + (1 - weight_fit_median) * fit_median_interpolated, label="weighted")

    extra_display = old_div((np.nanmax(fit_median) - np.nanmin(fit_median)), 10)
    ax.set_ylim(
        np.nanmin(fit_median) - extra_display, np.nanmax(fit_median) + extra_display
    )
    ax.set_xlim(wlm[0] - 10, wlm[-1] + 10)
    ax.tick_params(axis='both', which='minor')
    ax.legend(frameon=False, loc=1, ncol=1)

    ax.axvline(x=wave_min, color="k", linestyle="--")
    ax.axvline(x=wave_max, color="k", linestyle="--")

    ax.set_xlabel(r"Wavelength [$\AA$]")

    if exclude_wlm[0][0] != 0:
        for i in range(len(exclude_wlm)):
            ax.axvspan(exclude_wlm[i][0], exclude_wlm[i][1], color="r", alpha=0.1)

    plt.show()
    return fig


def plot_correction_in_fibre_p_fibre(fig_size,
                                     wlm,
                                     espectro_old,
                                     espectro_fit_median,
                                     espectro_new,
                                     fibre_p,
                                     clip_high):
    """
    Plot correction in fibre p_fibre
    """
    fig, ax = plt.subplots(figsize=(fig_size, fig_size / 2.5))
    ax.plot(
        wlm,
        old_div(espectro_old, espectro_fit_median),
        "r",
        label="Uncorrected",
        alpha=0.5,
    )
    ax.plot(
        wlm,
        old_div(espectro_new, espectro_fit_median),
        "b",
        label="Corrected",
        alpha=0.5,
    )
    const = old_div((np.nanmax(espectro_new) - np.nanmin(espectro_new)), 2)
    ax.plot(
        wlm,
        old_div((const + espectro_new - espectro_old), espectro_fit_median),
        "k",
        label="Dif + const",
        alpha=0.5,
    )
    ax.axhline(y=clip_high, color="k", linestyle=":", alpha=0.5)
    ax.set_ylabel("Flux / Continuum")

    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_title("Checking correction in fibre {}".format(str(fibre_p)))
    ax.legend(frameon=False, loc=1, ncol=4)

    plt.show()
    return fig


def plot_suspicious_fibres(suspicious_fibres,
                           fig_size,
                           wave_min,
                           wave_max,
                           intensity_corrected_fiber):
    """
    Plotting suspicious fibres
    """
    figures = []
    for fibre in suspicious_fibres:
        fig, ax = plt.subplots(figsize=(fig_size, fig_size / 2.5))
        ax.tick_params(axis='both', which='minor')
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel("Relative Flux")
        ax.set_xlim(wave_min, wave_max)
        scale = np.nanmax(intensity_corrected_fiber[fibre]) - np.nanmin(
            intensity_corrected_fiber[fibre]
        )
        ax.set_ylim(
            np.nanmin(intensity_corrected_fiber[fibre]) - old_div(scale, 15),
            np.nanmax(intensity_corrected_fiber[fibre]) + old_div(scale, 15),
        )
        ax.set_title("Checking spectrum of suspicious fibre {}. Do you see a cosmic?".format(np.str(fibre)))
        self.plot_spectrum(fibre)
        plt.show()
        figures.append(fig)
    return figures


def plot_skyline_5578(fig_size,
                      flux_5578,
                      flux_5578_medfilt):
    """
    Checking throughput correction using skyline 5578
    """
    fig, ax = plt.subplots(figsize=(fig_size, fig_size / 2.5))
    ax.plot(flux_5578, alpha=0.5)
    ax.plot(flux_5578_medfilt, alpha=1.0)
    ax.set_ylabel(r"Integrated flux of skyline 5578 $\AA$")
    ax.set_xlabel("Fibre")
    p01 = np.nanpercentile(flux_5578, 1)
    p99 = np.nanpercentile(flux_5578, 99)
    ax.set_ylim(p01, p99)
    ax.set_title(r"Checking throughput correction using skyline 5578 $\AA$")
    ax.tick_params(axis='both', which='minor')
    plt.show()

    return fig


def plot_offset_between_cubes(cube, x, y, wl, medfilt_window=151):

    """
    Plot the offset between two cubes

    Args:
        cube (Cube object): A cube instance
        x (array): ?
        y (array): ?
        wl (array): A wavelenhgth array
        medfilt_window (int): Window size for median filtering
    """

    smooth_x = signal.medfilt(x, medfilt_window)
    smooth_y = signal.medfilt(y, medfilt_window)

    print(np.nanmean(smooth_x))
    print(np.nanmean(smooth_y))

    fig, ax = plt.subplots(figsize=(10, 5))
    wl = cube.RSS.wavelength
    ax.plot(wl, x, "k.", alpha=0.1)
    ax.plot(wl, y, "r.", alpha=0.1)
    ax.plot(wl, smooth_x, "k-")
    ax.plot(wl, smooth_y, "r-")
    #    plt.plot(wl, x_max-np.nanmedian(x_max), 'g-')
    #    plt.plot(wl, y_max-np.nanmedian(y_max), 'y-')
    ax.set_ylim(-1.6, 1.6)
    plt.show()
    return fig

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_response(calibration_star_cubes, scale=[1, 1, 1, 1]):

    """
    Plot the response of standard stars. TODO rename this function to make it more explicit

    Args:
        calibration_star_cubes (list): A list of standard star objects
        scale (list): ?
    """

    print("\n> Plotting response of standard stars...\n")
    fig, ax = plt.subplots(figsize=(11, 8))
    wavelength = calibration_star_cubes[0].wavelength
    mean_curve = np.zeros_like(wavelength)
    mean_values = []
    i = 0
    for star in calibration_star_cubes:
        good = np.where(~np.isnan(star.response_curve))
        wl = star.response_wavelength[good]
        R = star.response_curve[good] * scale[i]
        mean_curve += np.interp(wavelength, wl, R)
        star.response_full = np.interp(wavelength, wl, R)
        ax.plot(
            star.response_wavelength,
            star.response_curve * scale[i],
            label=star.description,
            alpha=0.5,
            linewidth=2,
        )
        mean_value = star.response_curve * scale[i]
        print("\tMean value for {} = {}, scale = {}".format(star.object, np.nanmean(mean_value), scale[i]))
        mean_values.append(mean_value)
        i = i + 1
    mean_curve /= len(calibration_star_cubes)

    response_rms = np.zeros_like(wavelength)
    for star in calibration_star_cubes:
        response_rms += np.abs(star.response_full - mean_curve)
    response_rms /= len(calibration_star_cubes)
    dispersion = old_div(np.nansum(response_rms), np.nansum(mean_curve))
    print("\tVariation in flux calibrations =  {:.2f} %".format(dispersion * 100.0))

    # dispersion=np.nanmax(mean_values)-np.nanmin(mean_values)
    # print "  Variation in flux calibrations =  {:.2f} %".format(dispersion/np.nanmedian(mean_values)*100.)

    ax.plot(
        wavelength,
        mean_curve,
        "k",
        label="mean response curve",
        alpha=0.2,
        linewidth=10,
    )
    ax.legend(frameon=False, loc=2)
    ax.set_ylabel("Flux")
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_title("Response curve for calibration stars")
    ax.tick_params(axis='both', which='minor')
    plt.show()

    return fig


def plot_telluric_correction(wlm, telluric_correction_list, telluric_correction, figsize=12):
    """
    Make a plot of the telluric corretion as well as all of the individual stars which have gone into this telluric correction.

    Args:
        wlm (array): Wavelength array
        telluric_correction_list (list): A list of standard star spectra
        telluric_correction (array): The final telluric correction we'll use
        figsize (int, default=12): Size of the figure in inches

    Returns:
        A matplotlib figure object
    """

    fig, ax = plt.subplots(figsize=(fig_size, fig_size / 2.5))
    ax.set_title("Telluric correction")

    for i in range(len(telluric_correction_list)):
        label = r"star {}".format(i + 1)
        plt.plot(wlm, telluric_correction_list[i], alpha=0.3, label=label)

    ax.plot(wlm, telluric_correction, alpha=0.5, color="k", label="Median")
    plt.minorticks_on()
    plt.show()

