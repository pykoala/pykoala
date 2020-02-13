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
    plt.figure(figsize=(fig_size, fig_size / 2.5))
    plt.plot(wavelength, funcion, "r", lw=1, alpha=0.5)
    plt.minorticks_on()
    plt.xlabel("Wavelength [$\AA$]")
    plt.ylabel("Flux / continuum")

    plt.xlim(lmin, lmax)
    plt.ylim(fmin, fmax)
    plt.axhline(y=cut, color="k", linestyle=":", alpha=0.5)
    for i in range(len(peaks)):
        plt.axvline(x=peaks[i], color="k", linestyle=":", alpha=0.5)
        label = peaks_name[i]
        plt.text(peaks[i], 1.8, label)
    plt.show()


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
    plt.figure(figsize=(fig_size, fig_size / 2.5))
    plt.plot(wlm, s, alpha=0.5)
    plt.plot(running_wave, running_step_median, "+", ms=15, mew=3)
    plt.plot(wlm, fit_median, label="fit median")
    plt.plot(wlm, fit_median_interpolated, label="fit median_interp")
    plt.plot(
        wlm,
        weight_fit_median * fit_median
        + (1 - weight_fit_median) * fit_median_interpolated,
        label="weighted",
    )
    extra_display = old_div((np.nanmax(fit_median) - np.nanmin(fit_median)), 10)
    plt.ylim(
        np.nanmin(fit_median) - extra_display, np.nanmax(fit_median) + extra_display
    )
    plt.xlim(wlm[0] - 10, wlm[-1] + 10)
    plt.minorticks_on()
    plt.legend(frameon=False, loc=1, ncol=1)

    plt.axvline(x=wave_min, color="k", linestyle="--")
    plt.axvline(x=wave_max, color="k", linestyle="--")

    plt.xlabel("Wavelength [$\AA$]")

    if exclude_wlm[0][0] != 0:
        for i in range(len(exclude_wlm)):
            plt.axvspan(exclude_wlm[i][0], exclude_wlm[i][1], color="r", alpha=0.1)

    plt.show()
    plt.close()


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
    plt.figure(figsize=(fig_size, fig_size / 2.5))
    plt.plot(
        wlm,
        old_div(espectro_old, espectro_fit_median),
        "r",
        label="Uncorrected",
        alpha=0.5,
    )
    plt.plot(
        wlm,
        old_div(espectro_new, espectro_fit_median),
        "b",
        label="Corrected",
        alpha=0.5,
    )
    const = old_div((np.nanmax(espectro_new) - np.nanmin(espectro_new)), 2)
    plt.plot(
        wlm,
        old_div((const + espectro_new - espectro_old), espectro_fit_median),
        "k",
        label="Dif + const",
        alpha=0.5,
    )
    plt.axhline(y=clip_high, color="k", linestyle=":", alpha=0.5)
    plt.ylabel("Flux / Continuum")

    plt.xlabel("Wavelength [$\AA$]")
    plt.title("Checking correction in fibre " + str(fibre_p))
    plt.legend(frameon=False, loc=1, ncol=4)
    plt.minorticks_on()
    plt.show()
    plt.close()


def plot_suspicious_fibres(suspicious_fibres,
                           fig_size,
                           wave_min,
                           wave_max,
                           intensity_corrected_fiber):
    """
    Plotting suspicious fibres
    """
    for fibre in suspicious_fibres:
        plt.figure(figsize=(fig_size, fig_size / 2.5))
        plt.minorticks_on()
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel("Relative Flux")
        plt.xlim(wave_min, wave_max)
        scale = np.nanmax(intensity_corrected_fiber[fibre]) - np.nanmin(
            intensity_corrected_fiber[fibre]
        )
        plt.ylim(
            np.nanmin(intensity_corrected_fiber[fibre]) - old_div(scale, 15),
            np.nanmax(intensity_corrected_fiber[fibre]) + old_div(scale, 15),
        )
        text = (
            "Checking spectrum of suspicious fibre "
            + np.str(fibre)
            + ". Do you see a cosmic?"
        )
        plt.title(text)
        self.plot_spectrum(fibre)
        plt.show()
        plt.close()


def plot_skyline_5578(fig_size,
                      flux_5578,
                      flux_5578_medfilt):
    """
    Checking throughput correction using skyline 5578
    """
    plt.figure(figsize=(fig_size, fig_size / 2.5))
    plt.plot(flux_5578, alpha=0.5)
    plt.plot(flux_5578_medfilt, alpha=1.0)
    plt.ylabel("Integrated flux of skyline 5578 $\AA$")
    plt.xlabel("Fibre")
    p01 = np.nanpercentile(flux_5578, 1)
    p99 = np.nanpercentile(flux_5578, 99)
    plt.ylim(p01, p99)
    plt.title("Checking throughput correction using skyline 5578 $\AA$")
    plt.minorticks_on()
    plt.show()
    plt.close()
