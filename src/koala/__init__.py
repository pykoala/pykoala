#!/usr/bin/python
# -*- coding: utf-8 -*-
# # PyKOALA: KOALA data processing and analysis
# by Angel Lopez-Sanchez and Yago Ascasibar
# Extra work by Ben Lawson (MQ PACE student)
# Plus Taylah and Matt (sky substraction)
from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
version = "Version 0.72 - 13th February 2020"

# -----------------------------------------------------------------------------
# Start timer
# -----------------------------------------------------------------------------


from timeit import default_timer as timer

start = timer()

# -----------------------------------------------------------------------------
# Import Python routines
# -----------------------------------------------------------------------------

from koala.utils.plots import (plot_redshift_peaks,
                               plot_weights_for_getting_smooth_spectrum,
                               plot_correction_in_fibre_p_fibre,
                               plot_suspicious_fibres,
                               plot_skyline_5578)

from astropy.io import fits
from astropy.wcs import WCS

from pysynphot import observation
from pysynphot import spectrum

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys

from scipy import interpolate, signal, optimize
from scipy.optimize import curve_fit
import scipy.signal as sig

# from scipy.optimize import leastsq

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.ndimage.interpolation import shift

import datetime
import copy

# -----------------------------------------------------------------------------
# Define constants
# -----------------------------------------------------------------------------

pc = 3.086e18  # pc in cm
C = 299792.458  # c in km/s

# -----------------------------------------------------------------------------
# Define COLOR scales
# -----------------------------------------------------------------------------

fuego_color_map = colors.LinearSegmentedColormap.from_list(
    "fuego",
    (
        (0.25, 0, 0),
        (0.5, 0, 0),
        (1, 0, 0),
        (1, 0.5, 0),
        (1, 0.75, 0),
        (1, 1, 0),
        (1, 1, 1),
    ),
    N=256,
    gamma=1.0,
)
fuego_color_map.set_bad("lightgray")  # ('black')
plt.register_cmap(cmap=fuego_color_map)

projo = [0.25, 0.5, 1, 1.0, 1.00, 1, 1]
pverde = [0.00, 0.0, 0, 0.5, 0.75, 1, 1]
pazul = [0.00, 0.0, 0, 0.0, 0.00, 0, 1]

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
# RSS CLASS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class RSS(object):
    """
    Collection of row-stacked spectra (RSS).

    Attributes
    ----------
    wavelength: np.array(float)
      Wavelength, in Angstrom.
    intensity: np.array(float)
      Intensity :math:`I_\lambda` per unit wavelength.
    variance: np.array(float)
      Variance :math:`\sigma^2_\lambda` per unit wavelength
      (note the square in the definition of the variance).
    """

    # -----------------------------------------------------------------------------
    def __init__(self):
        self.description = "Undefined row-stacked spectra (RSS)"

        self.n_spectra = 0
        self.n_wave = 0

        self.wavelength = np.zeros((0))
        self.intensity = np.zeros((0, 0))
        self.intensity_corrected = self.intensity
        self.variance = np.zeros_like(self.intensity)

        self.RA_centre_deg = 0.0
        self.DEC_centre_deg = 0.0
        self.offset_RA_arcsec = np.zeros((0))
        self.offset_DEC_arcsec = np.zeros_like(self.offset_RA_arcsec)

        self.ALIGNED_RA_centre_deg = 0.0  # Added by ANGEL, 6 Sep
        self.ALIGNED_DEC_centre_deg = 0.0  # Added by ANGEL, 6 Sep
        self.relative_throughput = np.ones((0))  # Added by ANGEL, 16 Sep

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def compute_integrated_fibre(
        self,
        list_spectra="all",
        valid_wave_min=0,
        valid_wave_max=0,
        min_value=0.1,
        plot=False,
        title=" - Integrated values",
        warnings=True,
        text="...",
        correct_negative_sky=False,
    ):
        """
        Compute the integrated flux of a fibre in a particular range

        Parameters
        ----------
        list_spectra: float (default "all")
            list with the number of fibres for computing integrated value
            if using "all" it does all fibres
        valid_wave_min, valid_wave_max :  float
            the integrated flux value will be computed in the range [valid_wave_min,valid_wave_max]
            (default = , if they all 0 we use [self.valid_wave_min,self.valid_wave_max]
        min_value: float (default 0)
            For values lower than min_value, we set them as min_value
        plot : boolean (default = False)
            Plot
        title : string
            Tittle for the plot
        text: string
            A bit of extra text
        warnings : boolean (default = False)
            Write warnings, e.g. when the integrated flux is negative
        correct_negative_sky : boolean (default = False)
            Corrects negative values making 0 the integrated flux of the lowest fibre

        Example
        ----------
        integrated_fibre_6500_6600 = star1r.compute_integrated_fibre(valid_wave_min=6500, valid_wave_max=6600,
        title = " - [6500,6600]", plot = True)
        """

        print("\n  Computing integrated fibre values", text)

        if list_spectra == "all":
            list_spectra = list(range(self.n_spectra))
        if valid_wave_min == 0:
            valid_wave_min = self.valid_wave_min
        if valid_wave_max == 0:
            valid_wave_max = self.valid_wave_max

        self.integrated_fibre = np.zeros(self.n_spectra)
        region = np.where(
            (self.wavelength > valid_wave_min) & (self.wavelength < valid_wave_max)
        )
        waves_in_region = len(region[0])
        n_negative_fibres = 0
        negative_fibres = []
        for i in range(self.n_spectra):
            self.integrated_fibre[i] = np.nansum(self.intensity_corrected[i, region])
            if self.integrated_fibre[i] < 0:
                if warnings:
                    print("  WARNING: The integrated flux in fibre {:4} is negative, flux/wave = {:10.2f}, (probably sky), CHECK !".format(
                        i, old_div(self.integrated_fibre[i], waves_in_region)
                    ))
                n_negative_fibres = n_negative_fibres + 1
                # self.integrated_fibre[i] = min_value
                negative_fibres.append(i)

        if len(negative_fibres) != 0:
            print("\n> Number of fibres with integrated flux < 0 : {:4}, that is the {:5.2f} % of the total !".format(
                n_negative_fibres, n_negative_fibres * 100.0 / self.n_spectra
            ))

            negative_fibres_sorted = []
            integrated_intensity_sorted = np.argsort(
                old_div(self.integrated_fibre, waves_in_region)
            )
            for fibre_ in range(n_negative_fibres):
                negative_fibres_sorted.append(integrated_intensity_sorted[fibre_])
            # print "\n> Checking results using",n_negative_fibres,"fibres with the lowest integrated intensity"
            # print "  which are :",negative_fibres_sorted

            if correct_negative_sky:
                min_sky_value = self.integrated_fibre[negative_fibres_sorted[0]]
                min_sky_value_per_wave = old_div(min_sky_value, waves_in_region)
                print("\n> Correcting negative values making 0 the integrated flux of the lowest fibre, which is {:4} with {:10.2f} counts/wave".format(
                    negative_fibres_sorted[0], min_sky_value_per_wave
                ))
                # print self.integrated_fibre[negative_fibres_sorted[0]]
                self.integrated_fibre = self.integrated_fibre - min_sky_value
                for i in range(self.n_spectra):
                    self.intensity_corrected[i] = (
                        self.intensity_corrected[i] - min_sky_value_per_wave
                    )

            else:
                print("\n> Adopting integrated flux = {:5.2f} for all fibres with negative integrated flux (for presentation purposes)".format(
                    min_value
                ))
                for i in negative_fibres_sorted:
                    self.integrated_fibre[i] = min_value

            # for i in range(self.n_spectra):
            #    if self.integrated_fibre[i] < 0:
            #        if warnings: print "  WARNING: The integrated flux in fibre {:4} STILL is negative, flux/wave = {:10.2f}, (probably sky), CHECK !".format(i,self.integrated_fibre[i]/waves_in_region)

        if plot:
            # print"\n  Plotting map with integrated values:"
            self.RSS_map(
                self.integrated_fibre,
                norm=colors.PowerNorm(gamma=1.0 / 4.0),
                title=title,
            )

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def identify_el(
        self,
        high_fibres=10,
        brightest_line="Ha",
        cut=1.5,
        fibre=0,
        broad=1.0,
        verbose=True,
        plot=True,
    ):
        """
        Identify fibers with highest intensity (high_fibres=10).
        Add all in a single spectrum.
        Identify emission features.
        These emission features should be those expected in all the cube!
        Also, chosing fibre=number, it identifies el in a particular fibre.

        Parameters
        ----------
        high_fibres: float (default 10)
            use the high_fibres highest intensity fibres for identifying
        brightest_line :  string (default "Ha")
            string name with the emission line that is expected to be the brightest in integrated spectrum
        cut: float (default 1.5)
            The peak has to have a cut higher than cut to be considered as emission line
        fibre: integer (default 0)
            If fibre is given, it identifies emission lines in the given fibre
        broad: float (default 1.0)
            Broad (FWHM) of the expected emission lines
        verbose : boolean (default = True)
            Write results
        plot : boolean (default = False)
            Plot results

        Example
        ----------
        self.el=self.identify_el(high_fibres=10, brightest_line = "Ha",
                                 cut=2., verbose=True, plot=True, fibre=0, broad=1.5)
        """

        if fibre == 0:
            integrated_intensity_sorted = np.argsort(self.integrated_fibre)
            region = []
            for fibre in range(high_fibres):
                region.append(integrated_intensity_sorted[-1 - fibre])
            if verbose:
                print("\n> Identifying emission lines using the", high_fibres, "fibres with the highest integrated intensity")
                print("  which are :", region)
            combined_high_spectrum = np.nansum(self.intensity_corrected[region], axis=0)
        else:
            combined_high_spectrum = self.intensity_corrected[fibre]
            if verbose:
                print("\n> Identifying emission lines in fibre", fibre)

        # Search peaks
        peaks, peaks_name, peaks_rest, continuum_limits = search_peaks(
            self.wavelength,
            combined_high_spectrum,
            plot=plot,
            cut=cut,
            brightest_line=brightest_line,
            verbose=False,
        )
        p_peaks_l = []
        p_peaks_fwhm = []

        # Do Gaussian fit and provide center & FWHM (flux could be also included, not at the moment as not abs. flux-cal done)
        if verbose:
            print("\n  Emission lines identified:")
        for eline in range(len(peaks)):
            lowlow = continuum_limits[0][eline]
            lowhigh = continuum_limits[1][eline]
            highlow = continuum_limits[2][eline]
            highhigh = continuum_limits[3][eline]
            resultado = fluxes(
                self.wavelength,
                combined_high_spectrum,
                peaks[eline],
                verbose=False,
                broad=broad,
                lowlow=lowlow,
                lowhigh=lowhigh,
                highlow=highlow,
                highhigh=highhigh,
                plot=plot,
                fcal=False,
            )
            p_peaks_l.append(resultado[1])
            p_peaks_fwhm.append(resultado[5])
            if verbose:
                print("  {:3}. {:7s} {:8.2f} centered at {:8.2f} and FWHM = {:6.2f}".format(
                    eline + 1,
                    peaks_name[eline],
                    peaks_rest[eline],
                    p_peaks_l[eline],
                    p_peaks_fwhm[eline],
                ))

        return [peaks_name, peaks_rest, p_peaks_l, p_peaks_fwhm]

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def correct_high_cosmics_and_defects(
        self,
        step=50,
        correct_high_cosmics=False,
        fibre_p=0,
        remove_5578=False,  # if fibre_p=fibre plots the corrections in that fibre
        clip_high=100,
        warnings=False,
        plot=True,
        plot_suspicious_fibres=True,
        verbose=False,
        fig_size=12,
    ):
        """
        Correct for Cosmic Rays and defects on the CCD
        """

        print("\n> Correcting for high cosmics and CCD defects...")

        wave_min = self.valid_wave_min  # CHECK ALL OF THIS...
        wave_max = self.valid_wave_max
        wlm = self.wavelength

        if correct_high_cosmics == False:
            print("  Only CCD defects (nan and negative values) are considered.")
        else:
            print("  Using clip_high = ", clip_high, " for high cosmics")
            print("  IMPORTANT: Be sure that any emission or sky line is fainter than clip_high/continuum !! ")

        flux_5578 = []  # For correcting sky line 5578 if requested
        if wave_min < 5578 and remove_5578:
            print("  Sky line 5578 will be removed using a Gaussian fit...")

        integrated_fibre_uncorrected = self.integrated_fibre
        print(" ")
        output_every_few = np.sqrt(self.n_spectra) + 1
        next_output = -1
        max_ratio_list = []
        for fibre in range(self.n_spectra):
            if fibre > next_output:
                sys.stdout.write("\b" * 30)
                sys.stdout.write(
                    "  Cleaning... {:5.2f}% completed".format(
                        fibre * 100.0 / self.n_spectra
                    )
                )
                sys.stdout.flush()
                next_output = fibre + output_every_few

            s = self.intensity_corrected[fibre]
            running_wave = []
            running_step_median = []
            cuts = np.int(old_div(self.n_wave, step))
            for corte in range(cuts):
                if corte == 0:
                    next_wave = wave_min
                else:
                    next_wave = np.nanmedian(
                        old_div((wlm[np.int(corte * step)] + wlm[np.int((corte + 1) * step)]), 2)
                    )
                if next_wave < wave_max:
                    running_wave.append(next_wave)
                    region = np.where(
                        (wlm > running_wave[corte] - old_div(step, 2))
                        & (wlm < running_wave[corte] + old_div(step, 2))
                    )
                    running_step_median.append(
                        np.nanmedian(self.intensity_corrected[fibre, region])
                    )

            running_wave.append(wave_max)
            region = np.where((wlm > wave_max - step) & (wlm < wave_max))
            running_step_median.append(
                np.nanmedian(self.intensity_corrected[fibre, region])
            )

            for i in range(len(running_step_median)):
                if np.isnan(running_step_median[i]) == True:
                    if i < 10:
                        running_step_median[i] = np.nanmedian(running_step_median[0:9])
                    if i > 10:
                        running_step_median[i] = np.nanmedian(
                            running_step_median[-9:-1]
                        )

            a7x, a6x, a5x, a4x, a3x, a2x, a1x, a0x = np.polyfit(
                running_wave, running_step_median, 7
            )
            fit_median = (
                a0x
                + a1x * wlm
                + a2x * wlm ** 2
                + a3x * wlm ** 3
                + a4x * wlm ** 4
                + a5x * wlm ** 5
                + a6x * wlm ** 6
                + a7x * wlm ** 7
            )

            if fibre == fibre_p:
                espectro_old = copy.copy(self.intensity_corrected[fibre, :])
                espectro_fit_median = fit_median

            for wave in range(self.n_wave):  # (1,self.n_wave-3):
                if s[wave] < 0:
                    s[wave] = fit_median[wave]  # Negative values for median values
                if np.isnan(s[wave]) == True:
                    s[wave] = fit_median[wave]  # nan for median value
                if (
                    correct_high_cosmics and fit_median[wave] > 0
                ):  # NEW 15 Feb 2019, v7.1 2dFdr takes well cosmic rays
                    if s[wave] > clip_high * fit_median[wave]:
                        if verbose:
                            print("  CLIPPING HIGH =", clip_high, "in fibre", fibre, "w =", wlm[
                                wave
                            ], "value=", s[
                                wave
                            ], "v/median=", old_div(s[
                                wave
                            ], fit_median[
                                wave
                            ]))  # " median=",fit_median[wave]
                        s[wave] = fit_median[wave]

            if fibre == fibre_p:
                espectro_new = copy.copy(s)

            max_ratio_list.append(np.nanmax(old_div(s, fit_median)))
            self.intensity_corrected[fibre, :] = s

            # Removing Skyline 5578 using Gaussian fit if requested
            if wave_min < 5578 and remove_5578:
                resultado = fluxes(
                    wlm, s, 5578, plot=False, verbose=False
                )  # fmin=-5.0E-17, fmax=2.0E-16,
                # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                self.intensity_corrected[fibre] = resultado[11]
                flux_5578.append(resultado[3])

        sys.stdout.write("\b" * 30)
        sys.stdout.write("  Cleaning... 100.00 completed")
        sys.stdout.flush()

        max_ratio = np.nanmax(max_ratio_list)
        print("\n  Maximum value found of flux/continuum = ", max_ratio)
        if correct_high_cosmics:
            print("  Recommended value for clip_high = ", int(
                max_ratio + 1
            ), ", here we used ", clip_high)

        # Plot correction in fibre p_fibre
        if fibre_p > 0:
            plot_correction_in_fibre_p_fibre(fig_size,
                                             wlm,
                                             espectro_old,
                                             espectro_fit_median,
                                             espectro_new,
                                             fibre_p,
                                             clip_high)
        # print" "
        if correct_high_cosmics == False:
            text = "for spectra corrected for defects..."
            title = " - Throughput + CCD defects corrected"
        else:
            text = "for spectra corrected for high cosmics and defects..."
            title = " - Throughput + high-C & D corrected"
        self.compute_integrated_fibre(
            valid_wave_min=wave_min,
            valid_wave_max=wave_max,
            text=text,
            plot=plot,
            title=title,
        )

        if plot:
            print("  Plotting integrated fibre values before and after correcting for high cosmics and CCD defects:\n")
            plt.figure(figsize=(fig_size, fig_size / 2.5))
            plt.plot(integrated_fibre_uncorrected, "r", label="Uncorrected", alpha=0.5)
            plt.ylabel("Integrated Flux")
            plt.xlabel("Fibre")
            plt.ylim(
                [np.nanmin(self.integrated_fibre), np.nanmax(self.integrated_fibre)]
            )
            plt.title(self.description)

            # Check if integrated value is high
            median_running = []
            step_f = 10
            max_value = 2.0  # For stars this is not accurate, as i/m might be between 5 and 100 in the fibres with the star
            skip = 0
            suspicious_fibres = []
            for fibre in range(self.n_spectra):
                if fibre < step_f:
                    median_value = np.nanmedian(
                        self.integrated_fibre[0: np.int(step_f)]
                    )
                    skip = 1
                if fibre > self.n_spectra - step_f:
                    median_value = np.nanmedian(
                        self.integrated_fibre[-1 - np.int(step_f): -1]
                    )
                    skip = 1
                if skip == 0:
                    median_value = np.nanmedian(
                        self.integrated_fibre[
                            fibre - np.int(old_div(step_f, 2)): fibre + np.int(old_div(step_f, 2))
                        ]
                    )
                median_running.append(median_value)

                if old_div(self.integrated_fibre[fibre], median_running[fibre]) > max_value:
                    print("  Fibre ", fibre, " has a integrated/median ratio of ", old_div(self.integrated_fibre[
                        fibre
                    ], median_running[
                        fibre
                    ]), "  -> Might be a cosmic left!")
                    label = np.str(fibre)
                    plt.axvline(x=fibre, color="k", linestyle="--")
                    plt.text(fibre, self.integrated_fibre[fibre] / 2.0, label)
                    suspicious_fibres.append(fibre)
                skip = 0

            plt.plot(self.integrated_fibre, label="Corrected", alpha=0.6)
            plt.plot(median_running, "k", label="Median", alpha=0.6)
            plt.legend(frameon=False, loc=1, ncol=3)
            plt.minorticks_on()
            plt.show()
            plt.close()

        if plot_suspicious_fibres == True and len(suspicious_fibres) > 0:
            # Plotting suspicious fibres..
            plot_suspicious_fibres(suspicious_fibres,
                                   fig_size,
                                   wave_min,
                                   wave_max,
                                   intensity_corrected_fiber=self.intensity_corrected)

        if remove_5578 and wave_min < 5578:
            print("  Skyline 5578 has been removed. Checking throughput correction...")
            flux_5578_medfilt = sig.medfilt(flux_5578, np.int(5))
            median_flux_5578_medfilt = np.nanmedian(flux_5578_medfilt)
            extra_throughput_correction = old_div(flux_5578_medfilt, median_flux_5578_medfilt)
            # plt.plot(extra_throughput_correction)
            # plt.show()
            # plt.close()
            if plot:
                plot_skyline_5578(fig_size, flux_5578, flux_5578_medfilt)

            print("  Variations in throughput between", np.nanmin(
                extra_throughput_correction
            ), "and", np.nanmax(extra_throughput_correction))
            print("  Applying this extra throughtput correction to all fibres...")

            for i in range(self.n_spectra):
                self.intensity_corrected[i, :] = (
                    old_div(self.intensity_corrected[i, :], extra_throughput_correction[i])
                )
            self.relative_throughput = (
                self.relative_throughput * extra_throughput_correction
            )

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def clean_sky_residuals(
        self,
        extra_w=1.3,
        step=25,
        dclip=3.0,
        wave_min=0,
        wave_max=0,
        verbose=False,
        plot=False,
        fig_size=12,
        fibre=0,
    ):
        # verbose=True
        wlm = self.wavelength
        if wave_min == 0:
            wave_min = self.valid_wave_min
        if wave_max == 0:
            wave_max = self.valid_wave_max

        # Exclude ranges with emission lines if needed
        exclude_ranges_low = []
        exclude_ranges_high = []
        exclude_ranges_low_ = []
        exclude_ranges_high_ = []

        if self.el[1][0] != 0:
            #        print "  Emission lines identified in the combined spectrum:"
            for el in range(len(self.el[0])):
                #            print "  {:3}. - {:7s} {:8.2f} centered at {:8.2f} and FWHM = {:6.2f}".format(el+1,self.el[0][el],self.el[1][el],self.el[2][el],self.el[3][el])
                if (
                    self.el[0][el] == "Ha" or self.el[1][el] == 6583.41
                ):  # Extra extend for Ha and [N II] 6583
                    extra = extra_w * 1.6
                else:
                    extra = extra_w
                exclude_ranges_low_.append(
                    self.el[2][el] - self.el[3][el] * extra
                )  # center-1.3*FWHM/2
                exclude_ranges_high_.append(
                    self.el[2][el] + self.el[3][el] * extra
                )  # center+1.3*FWHM/2
                # print self.el[0][el],self.el[1][el],self.el[2][el],self.el[3][el],exclude_ranges_low[el],exclude_ranges_high[el],extra

            # Check overlapping ranges
            skip_next = 0
            for i in range(len(exclude_ranges_low_) - 1):
                if skip_next == 0:
                    if exclude_ranges_high_[i] > exclude_ranges_low_[i + 1]:
                        # Ranges overlap, now check if next range also overlaps
                        if i + 2 < len(exclude_ranges_low_):
                            if exclude_ranges_high_[i + 1] > exclude_ranges_low_[i + 2]:
                                exclude_ranges_low.append(exclude_ranges_low_[i])
                                exclude_ranges_high.append(exclude_ranges_high_[i + 2])
                                skip_next = 2
                                if verbose:
                                    print("Double overlap", exclude_ranges_low[
                                        -1
                                    ], exclude_ranges_high[-1])
                            else:
                                exclude_ranges_low.append(exclude_ranges_low_[i])
                                exclude_ranges_high.append(exclude_ranges_high_[i + 1])
                                skip_next = 1
                                if verbose:
                                    print("Overlap", exclude_ranges_low[
                                        -1
                                    ], exclude_ranges_high[-1])
                    else:
                        exclude_ranges_low.append(exclude_ranges_low_[i])
                        exclude_ranges_high.append(exclude_ranges_high_[i])
                        if verbose:
                            print("Overlap", exclude_ranges_low[
                                -1
                            ], exclude_ranges_high[-1])
                else:
                    if skip_next == 1:
                        skip_next = 0
                    if skip_next == 2:
                        skip_next = 1
                if verbose:
                    print(exclude_ranges_low_[i], exclude_ranges_high_[i], skip_next)
            if skip_next == 0:
                exclude_ranges_low.append(exclude_ranges_low_[-1])
                exclude_ranges_high.append(exclude_ranges_high_[-1])
                if verbose:
                    print(exclude_ranges_low_[-1], exclude_ranges_high_[-1], skip_next)

            # print "\n> Cleaning sky residuals in range [",wave_min,",",wave_max,"] avoiding emission lines... "
            print("\n> Cleaning sky residuals avoiding emission lines... ")

            if verbose:
                print("  Excluded ranges using emission line parameters:")
                for i in range(len(exclude_ranges_low_)):
                    print(exclude_ranges_low_[i], exclude_ranges_high_[i])
                print("  Excluded ranges considering overlaps: ")
                for i in range(len(exclude_ranges_low)):
                    print(exclude_ranges_low[i], exclude_ranges_high[i])
                print(" ")
        else:
            exclude_ranges_low.append(20000.0)
            exclude_ranges_high.append(30000.0)
            print("\n> Cleaning sky residuals...")

        say_status = 0
        if fibre != 0:
            f_i = fibre
            f_f = fibre + 1
            print("  Checking fibre ", fibre, " (only this fibre is corrected, use fibre = 0 for all)...")
            plot = True
        else:
            f_i = 0
            f_f = self.n_spectra
        for fibre in range(f_i, f_f):  # (self.n_spectra):
            if fibre == say_status:
                print("  Checking fibre ", fibre, " ...")
                say_status = say_status + 100

            s = self.intensity_corrected[fibre]

            fit_median = smooth_spectrum(
                wlm,
                s,
                step=step,
                wave_min=wave_min,
                wave_max=wave_max,
                weight_fit_median=1.0,
                plot=False,
            )
            old = []
            if plot:
                for i in range(len(s)):
                    old.append(s[i])

            disp = s - fit_median
            dispersion = np.nanmedian(np.abs(disp))

            rango = 0
            imprimir = 1
            for i in range(len(wlm) - 1):
                # if wlm[i] > wave_min and wlm[i] < wave_max :  # CLEAN ONLY IN VALID WAVEVELENGTHS
                if (
                    wlm[i] >= exclude_ranges_low[rango]
                    and wlm[i] <= exclude_ranges_high[rango]
                ):
                    if verbose == True and imprimir == 1:
                        print("  Excluding range [", exclude_ranges_low[
                            rango
                        ], ",", exclude_ranges_high[
                            rango
                        ], "] as it has an emission line")
                    if imprimir == 1:
                        imprimir = 0
                    # print "    Checking ", wlm[i]," NOT CORRECTED ",s[i], s[i]-fit_median[i]
                else:
                    if np.isnan(s[i]) == True:
                        s[i] = fit_median[i]  # nan for median value

                    if (
                        disp[i] > dispersion * dclip
                        and disp[i + 1] < -dispersion * dclip
                    ):
                        s[i] = fit_median[i]
                        s[i + 1] = fit_median[i + 1]  # "P-Cygni-like structures
                        if verbose:
                            print("  Found P-Cygni-like feature in ", wlm[i])
                    if disp[i] > dispersion * dclip or disp[i] < -dispersion * dclip:
                        s[i] = fit_median[i]
                        if verbose:
                            print("  Clipping feature in ", wlm[i])

                    if wlm[i] > exclude_ranges_high[rango] and imprimir == 0:
                        if verbose:
                            print("  Checked", wlm[
                                i
                            ], "  End range ", rango, exclude_ranges_low[
                                rango
                            ], exclude_ranges_high[
                                rango
                            ])
                        rango = rango + 1
                        imprimir = 1
                    if rango == len(exclude_ranges_low):
                        rango = len(exclude_ranges_low) - 1
                    # print "    Checking ", wlm[i]," CORRECTED IF NEEDED",s[i], s[i]-fit_median[i]

            #            if plot:
            #                for i in range(6):
            #                    plt.figure(figsize=(fig_size, fig_size/2.5))
            #                    plt.plot(wlm,old-fit_median, "r-", alpha=0.4)
            #                    plt.plot(wlm,fit_median-fit_median,"g-", alpha=0.5)
            #                    plt.axhline(y=dispersion*dclip, color="g", alpha=0.5)
            #                    plt.axhline(y=-dispersion*dclip, color="g", alpha=0.5)
            #                    plt.plot(wlm,s-fit_median, "b-", alpha=0.7)
            #
            #                    for exclude in range(len(exclude_ranges_low)):
            #                        plt.axvspan(exclude_ranges_low[exclude], exclude_ranges_high[exclude], facecolor='g', alpha=0.15,zorder=3)
            #
            #                    plt.ylim(-100,200)
            #                    if i == 0: plt.xlim(wlm[0]-10,wlm[-1]+10)
            #                    if i == 1: plt.xlim(wlm[0],6500)               # THIS IS FOR 1000R
            #                    if i == 2: plt.xlim(6500,6700)
            #                    if i == 3: plt.xlim(6700,7000)
            #                    if i == 4: plt.xlim(7000,7300)
            #                    if i == 5: plt.xlim(7300,wlm[-1])
            #                    plt.minorticks_on()
            #                    plt.xlabel("Wavelength [$\AA$]")
            #                    plt.ylabel("Flux / continuum")
            #                    plt.show()
            #                    plt.close()

            if plot:
                for i in range(6):
                    plt.figure(figsize=(fig_size, fig_size / 2.5))
                    plt.plot(wlm, old, "r-", alpha=0.4)
                    plt.plot(wlm, fit_median, "g-", alpha=0.5)
                    # plt.axhline(y=dispersion*dclip, color="g", alpha=0.5)
                    # plt.axhline(y=-dispersion*dclip, color="g", alpha=0.5)
                    plt.plot(wlm, s, "b-", alpha=0.7)

                    for exclude in range(len(exclude_ranges_low)):
                        plt.axvspan(
                            exclude_ranges_low[exclude],
                            exclude_ranges_high[exclude],
                            facecolor="g",
                            alpha=0.15,
                            zorder=3,
                        )

                    plt.ylim(-300, 300)
                    if i == 0:
                        plt.xlim(wlm[0] - 10, wlm[-1] + 10)
                    if i == 1:
                        plt.xlim(wlm[0], 6500)  # THIS IS FOR 1000R
                    if i == 2:
                        plt.xlim(6500, 6700)
                    if i == 3:
                        plt.xlim(6700, 7000)
                    if i == 4:
                        plt.xlim(7000, 7300)
                    if i == 5:
                        plt.xlim(7300, wlm[-1])
                    plt.minorticks_on()
                    plt.xlabel("Wavelength [$\AA$]")
                    plt.ylabel("Flux / continuum")
                    plt.show()
                    plt.close()

            self.intensity_corrected[fibre, :] = s

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    def fit_and_substract_sky_spectrum(
        self,
        sky,
        w=1000,
        spectra=1000,
        # If rebin == True, it fits all wavelengths to be at the same wavelengths that SKY spectrum...
        rebin=False,
        brightest_line="Ha",
        brightest_line_wavelength=6563.0,
        maxima_sigma=3.0,
        ymin=-50,
        ymax=1000,
        wmin=0,
        wmax=0,
        auto_scale_sky=False,
        warnings=False,
        verbose=False,
        plot=False,
        fig_size=12,
        fibre=0,
    ):
        """
        Given a 1D sky spectrum, this task fits
        sky lines of each spectrum individually and substracts sky
        Needs the observed wavelength (brightest_line_wavelength) of the brightest emission line (brightest_line) .
        w is the wavelength
        spec the 2D spectra
        """

        if brightest_line_wavelength == 6563:
            print("\n\n> WARNING: This is going to FAIL as the wavelength of the brightest emission line has not been included !!!")
            print("           USING brightest_line_wavelength = 6563 as default ...\n\n")

        brightest_line_wavelength_rest = 6562.82
        if brightest_line == "O3" or brightest_line == "O3b":
            brightest_line_wavelength_rest = 5006.84
        if brightest_line == "Hb" or brightest_line == "hb":
            brightest_line_wavelength_rest = 4861.33

        print("  Using {:3} at rest wavelength {:6.2f} identified by the user at {:6.2f} to avoid fitting emission lines...".format(
            brightest_line, brightest_line_wavelength_rest, brightest_line_wavelength
        ))

        redshift = old_div(brightest_line_wavelength, brightest_line_wavelength_rest) - 1.0

        if w == 1000:
            w = self.wavelength
        if spectra == 1000:
            spectra = copy.deepcopy(self.intensity_corrected)

        if wmin == 0:
            wmin = w[0]
        if wmax == 0:
            wmax = w[-1]

        # Read file with sky emission lines
        sky_lines_file = "sky_lines.dat"
        (
            sl_center,
            sl_name,
            sl_fnl,
            sl_lowlow,
            sl_lowhigh,
            sl_highlow,
            sl_highhigh,
            sl_lmin,
            sl_lmax,
        ) = read_table(sky_lines_file, ["f", "s", "f", "f", "f", "f", "f", "f", "f"])
        number_sl = len(sl_center)

        # MOST IMPORTANT EMISSION LINES IN RED
        # 6300.30       [OI]  -0.263   30.0 15.0   20.0   40.0
        # 6312.10     [SIII]  -0.264   30.0 18.0    5.0   20.0
        # 6363.78       [OI]  -0.271   20.0  4.0    5.0   30.0
        # 6548.03      [NII]  -0.296   45.0 15.0   55.0   75.0
        # 6562.82         Ha  -0.298   50.0 25.0   35.0   60.0
        # 6583.41      [NII]  -0.300   62.0 42.0    7.0   35.0
        # 6678.15        HeI  -0.313   20.0  6.0    6.0   20.0
        # 6716.47      [SII]  -0.318   40.0 15.0   22.0   45.0
        # 6730.85      [SII]  -0.320   50.0 30.0    7.0   35.0
        # 7065.28        HeI  -0.364   30.0  7.0    7.0   30.0
        # 7135.78    [ArIII]  -0.374   25.0  6.0    6.0   25.0
        # 7318.39      [OII]  -0.398   30.0  6.0   20.0   45.0
        # 7329.66      [OII]  -0.400   40.0 16.0   10.0   35.0
        # 7751.10    [ArIII]  -0.455   30.0 15.0   15.0   30.0
        # 9068.90    [S-III]  -0.594   30.0 15.0   15.0   30.0

        el_list_no_z = [
            6300.3,
            6312.10,
            6363.78,
            6548.03,
            6562.82,
            6583.41,
            6678.15,
            6716.47,
            6730.85,
            7065.28,
            7135.78,
            7318.39,
            7329.66,
            7751.1,
            9068.9,
        ]
        el_list = (redshift + 1) * np.array(el_list_no_z)
        #  [OI]   [SIII]  [OI]   Ha+[NII]  HeI    [SII]     HeI   [ArIII]  [OII]  [ArIII]  [SIII]
        el_low_list_no_z = [
            6296.3,
            6308.1,
            6359.8,
            6544.0,
            6674.2,
            6712.5,
            7061.3,
            7131.8,
            7314.4,
            7747.1,
            9063.9,
        ]
        el_high_list_no_z = [
            6304.3,
            6316.1,
            6367.8,
            6590.0,
            6682.2,
            6736.9,
            7069.3,
            7139.8,
            7333.7,
            7755.1,
            9073.9,
        ]
        el_low_list = (redshift + 1) * np.array(el_low_list_no_z)
        el_high_list = (redshift + 1) * np.array(el_high_list_no_z)

        # Double Skylines
        dsky1 = [
            6257.82,
            6465.34,
            6828.22,
            6969.70,
            7239.41,
            7295.81,
            7711.50,
            7750.56,
            7853.391,
            7913.57,
            7773.00,
            7870.05,
            8280.94,
            8344.613,
            9152.2,
            9092.7,
            9216.5,
            8827.112,
            8761.2,
            0,
        ]  # 8760.6, 0]#
        dsky2 = [
            6265.50,
            6470.91,
            6832.70,
            6978.45,
            7244.43,
            7303.92,
            7715.50,
            7759.89,
            7860.662,
            7921.02,
            7780.43,
            7879.96,
            8288.34,
            8352.78,
            9160.9,
            9102.8,
            9224.8,
            8836.27,
            8767.7,
            0,
        ]  # 8767.2, 0] #

        say_status = 0
        # plot=True
        #        verbose = True
        # warnings = True
        self.wavelength_offset_per_fibre = []
        self.sky_auto_scale = []
        if fibre != 0:
            f_i = fibre
            f_f = fibre + 1
            print("  Checking fibre ", fibre, " (only this fibre is corrected, use fibre = 0 for all)...")
            plot = True
            verbose = True
            warnings = True
        else:
            f_i = 0
            f_f = self.n_spectra
        for fibre in range(f_i, f_f):  # (self.n_spectra):
            if fibre == say_status:
                print("  Checking fibre {:4} ...  ({:6.2f} % completed) ...".format(
                    fibre, fibre * 100.0 / self.n_spectra
                ))
                say_status = say_status + 20

            # Gaussian fits to the sky spectrum
            sl_gaussian_flux = []
            sl_gaussian_sigma = []
            sl_gauss_center = []
            skip_sl_fit = []  # True emission line, False no emission line

            j_lines = 0
            el_low = el_low_list[j_lines]
            el_high = el_high_list[j_lines]
            sky_sl_gaussian_fitted = copy.deepcopy(sky)
            di = 0
            if verbose:
                print("\n> Performing Gaussian fitting to sky lines in sky spectrum...")
            for i in range(number_sl):
                if sl_center[i] > el_high:
                    while sl_center[i] > el_high:
                        j_lines = j_lines + 1
                        if j_lines < len(el_low_list) - 1:
                            el_low = el_low_list[j_lines]
                            el_high = el_high_list[j_lines]
                            # print "Change to range ",el_low,el_high
                        else:
                            el_low = w[-1] + 1
                            el_high = w[-1] + 2

                if sl_fnl[i] == 0:
                    plot_fit = False
                else:
                    plot_fit = True

                if sl_center[i] == dsky1[di]:
                    warnings_ = False
                    if sl_fnl[i] == 1:
                        warnings_ = True
                        if verbose:
                            print("  Line ", sl_center[i], " blended with ", dsky2[di])
                    resultado = dfluxes(
                        w,
                        sky_sl_gaussian_fitted,
                        sl_center[i],
                        dsky2[di],
                        lowlow=sl_lowlow[i],
                        lowhigh=sl_lowhigh[i],
                        highlow=sl_highlow[i],
                        highhigh=sl_highhigh[i],
                        lmin=sl_lmin[i],
                        lmax=sl_lmax[i],
                        fmin=0,
                        fmax=0,
                        broad1=2.1 * 2.355,
                        broad2=2.1 * 2.355,
                        plot=plot_fit,
                        verbose=False,
                        plot_sus=False,
                        fcal=False,
                        warnings=warnings_,
                    )  # Broad is FWHM for Gaussian sigm a= 1,
                    di = di + 1
                else:

                    resultado = fluxes(
                        w,
                        sky_sl_gaussian_fitted,
                        sl_center[i],
                        lowlow=sl_lowlow[i],
                        lowhigh=sl_lowhigh[i],
                        highlow=sl_highlow[i],
                        highhigh=sl_highhigh[i],
                        lmin=sl_lmin[i],
                        lmax=sl_lmax[i],
                        fmin=0,
                        fmax=0,
                        broad=2.1 * 2.355,
                        plot=plot_fit,
                        verbose=False,
                        plot_sus=False,
                        fcal=False,
                        warnings=warnings,
                    )  # Broad is FWHM for Gaussian sigm a= 1,
                sl_gaussian_flux.append(resultado[3])
                sky_sl_gaussian_fitted = resultado[11]
                sl_gauss_center.append(resultado[1])
                sl_gaussian_sigma.append(resultado[5] / 2.355)
                if el_low < sl_center[i] < el_high:
                    if verbose:
                        print("  SKY line", sl_center[i], "in EMISSION LINE !")
                    skip_sl_fit.append(True)
                else:
                    skip_sl_fit.append(False)

                # print "  Fitted wavelength for sky line ",sl_center[i]," : ",resultado[1],"   ",resultado[5]
                if plot_fit:
                    if verbose:
                        print("  Fitted wavelength for sky line ", sl_center[
                            i
                        ], " : ", sl_gauss_center[i], "  sigma = ", sl_gaussian_sigma[i])
                    wmin = sl_lmin[i]
                    wmax = sl_lmax[i]

            # Gaussian fit to object spectrum
            object_sl_gaussian_flux = []
            object_sl_gaussian_sigma = []
            ratio_object_sky_sl_gaussian = []
            dif_center_obj_sky = []
            spec = spectra[fibre]
            object_sl_gaussian_fitted = copy.deepcopy(spec)
            object_sl_gaussian_center = []
            di = 0
            if verbose:
                print("\n> Performing Gaussian fitting to sky lines in fibre", fibre, " of object data...")

            for i in range(number_sl):
                if sl_fnl[i] == 0:
                    plot_fit = False
                else:
                    plot_fit = True
                if skip_sl_fit[i]:
                    if verbose:
                        print(" SKIPPING SKY LINE", sl_center[
                            i
                        ], " as located within the range of an emission line!")
                    object_sl_gaussian_flux.append(
                        float("nan")
                    )  # The value of the SKY SPECTRUM
                    object_sl_gaussian_center.append(float("nan"))
                    object_sl_gaussian_sigma.append(float("nan"))
                    dif_center_obj_sky.append(float("nan"))
                else:

                    if sl_center[i] == dsky1[di]:
                        warnings_ = False
                        if sl_fnl[i] == 1:
                            warnings_ = True
                            if verbose:
                                print("  Line ", sl_center[i], " blended with ", dsky2[
                                    di
                                ])
                        resultado = dfluxes(
                            w,
                            object_sl_gaussian_fitted,
                            sl_center[i],
                            dsky2[di],
                            lowlow=sl_lowlow[i],
                            lowhigh=sl_lowhigh[i],
                            highlow=sl_highlow[i],
                            highhigh=sl_highhigh[i],
                            lmin=sl_lmin[i],
                            lmax=sl_lmax[i],
                            fmin=0,
                            fmax=0,
                            broad1=sl_gaussian_sigma[i] * 2.355,
                            broad2=sl_gaussian_sigma[i] * 2.355,
                            plot=plot_fit,
                            verbose=False,
                            plot_sus=False,
                            fcal=False,
                            warnings=warnings_,
                        )
                        di = di + 1
                        if (
                            resultado[3] > 0
                            and resultado[5] / 2.355 < maxima_sigma
                            and resultado[13] > 0
                            and resultado[14] / 2.355 < maxima_sigma
                        ):  # and resultado[5] < maxima_sigma: # -100000.: #0:
                            use_sigma = resultado[5] / 2.355
                            object_sl_gaussian_flux.append(resultado[3])
                            object_sl_gaussian_fitted = resultado[11]
                            object_sl_gaussian_center.append(resultado[1])
                            object_sl_gaussian_sigma.append(use_sigma)
                            dif_center_obj_sky.append(
                                object_sl_gaussian_center[i] - sl_gauss_center[i]
                            )
                        else:
                            if verbose:
                                print("  Bad fit for ", sl_center[i], "! ignoring it...")
                            object_sl_gaussian_flux.append(float("nan"))
                            object_sl_gaussian_center.append(float("nan"))
                            object_sl_gaussian_sigma.append(float("nan"))
                            dif_center_obj_sky.append(float("nan"))
                            skip_sl_fit[i] = True  # We don't substract this fit

                    else:
                        resultado = fluxes(
                            w,
                            object_sl_gaussian_fitted,
                            sl_center[i],
                            lowlow=sl_lowlow[i],
                            lowhigh=sl_lowhigh[i],
                            highlow=sl_highlow[i],
                            highhigh=sl_highhigh[i],
                            lmin=sl_lmin[i],
                            lmax=sl_lmax[i],
                            fmin=0,
                            fmax=0,
                            broad=sl_gaussian_sigma[i] * 2.355,
                            plot=plot_fit,
                            verbose=False,
                            plot_sus=False,
                            fcal=False,
                            warnings=warnings,
                        )  # Broad is FWHM for Gaussian sigma= 1,
                        # print sl_center[i],sl_gaussian_sigma[i], resultado[5]/2.355, maxima_sigma
                        if (
                            resultado[3] > 0 and resultado[5] / 2.355 < maxima_sigma
                        ):  # and resultado[5] < maxima_sigma: # -100000.: #0:
                            object_sl_gaussian_flux.append(resultado[3])
                            object_sl_gaussian_fitted = resultado[11]
                            object_sl_gaussian_center.append(resultado[1])
                            object_sl_gaussian_sigma.append(resultado[5] / 2.355)
                            dif_center_obj_sky.append(
                                object_sl_gaussian_center[i] - sl_gauss_center[i]
                            )
                        else:
                            if verbose:
                                print("  Bad fit for ", sl_center[i], "! ignoring it...")
                            object_sl_gaussian_flux.append(float("nan"))
                            object_sl_gaussian_center.append(float("nan"))
                            object_sl_gaussian_sigma.append(float("nan"))
                            dif_center_obj_sky.append(float("nan"))
                            skip_sl_fit[i] = True  # We don't substract this fit

                ratio_object_sky_sl_gaussian.append(
                    old_div(object_sl_gaussian_flux[i], sl_gaussian_flux[i])
                )

            # Scale sky lines that are located in emission lines or provided negative values in fit
            # reference_sl = 1 # Position in the file! Position 1 is sky line 6363.4
            # sl_ref_ratio = sl_gaussian_flux/sl_gaussian_flux[reference_sl]
            if verbose:
                print("\n> Correcting skylines for which we couldn't get a Gaussian fit...\n")
            for i in range(number_sl):
                if skip_sl_fit[i] == True:
                    # Use known center, sigma of the sky and peak
                    gauss_fix = sl_gaussian_sigma[i]
                    small_center_correction = 0.0
                    # Check if center of previous sky line has a small difference in wavelength
                    small_center_correction = np.nanmedian(dif_center_obj_sky[0:i])
                    if verbose:
                        print("- Small correction of center wavelength of sky line ", sl_center[
                            i
                        ], "  :", small_center_correction)

                    object_sl_gaussian_fitted = substract_given_gaussian(
                        w,
                        object_sl_gaussian_fitted,
                        sl_center[i] + small_center_correction,
                        peak=0,
                        sigma=gauss_fix,
                        flux=0,
                        search_peak=True,
                        lowlow=sl_lowlow[i],
                        lowhigh=sl_lowhigh[i],
                        highlow=sl_highlow[i],
                        highhigh=sl_highhigh[i],
                        lmin=sl_lmin[i],
                        lmax=sl_lmax[i],
                        plot=False,
                        verbose=verbose,
                    )

                    # Substract second Gaussian if needed !!!!!
                    for di in range(len(dsky1) - 1):
                        if sl_center[i] == dsky1[di]:
                            if verbose:
                                print("  This was a double sky line, also substracting ", dsky2[
                                    di
                                ], "  at ", np.array(
                                    dsky2[di]
                                ) + small_center_correction)
                            object_sl_gaussian_fitted = substract_given_gaussian(
                                w,
                                object_sl_gaussian_fitted,
                                np.array(dsky2[di]) + small_center_correction,
                                peak=0,
                                sigma=gauss_fix,
                                flux=0,
                                search_peak=True,
                                lowlow=sl_lowlow[i],
                                lowhigh=sl_lowhigh[i],
                                highlow=sl_highlow[i],
                                highhigh=sl_highhigh[i],
                                lmin=sl_lmin[i],
                                lmax=sl_lmax[i],
                                plot=False,
                                verbose=verbose,
                            )

            #    wmin,wmax = 6100,6500
            #    ymin,ymax= -100,400
            #
            #    wmin,wmax = 6350,6700
            #    wmin,wmax = 7100,7700
            #    wmin,wmax = 7600,8200
            #    wmin,wmax = 8200,8500
            #    wmin,wmax = 7350,7500
            #    wmin,wmax=6100, 8500 #7800, 8000#6820, 6850 #6700,7000 #6300,6450#7500
            #    wmin,wmax = 8700,9300
            #    ymax=800

            if plot:
                plt.figure(figsize=(11, 4))
                plt.plot(w, spec, "y", alpha=0.7, label="Object")
                plt.plot(
                    w,
                    object_sl_gaussian_fitted,
                    "k",
                    alpha=0.5,
                    label="Obj - sky fitted",
                )
                plt.plot(w, sky_sl_gaussian_fitted, "r", alpha=0.5, label="Sky fitted")
                plt.plot(w, spec - sky, "g", alpha=0.5, label="Obj - sky")
                plt.plot(
                    w,
                    object_sl_gaussian_fitted - sky_sl_gaussian_fitted,
                    "b",
                    alpha=0.9,
                    label="Obj - sky fitted - rest sky",
                )
                plt.xlim(wmin, wmax)
                plt.ylim(ymin, ymax)
                ptitle = "Fibre " + np.str(fibre)  # +" with rms = "+np.str(rms[i])
                plt.title(ptitle)
                plt.xlabel("Wavelength [$\AA$]")
                plt.ylabel("Flux [counts]")
                plt.legend(frameon=True, loc=2, ncol=5)
                plt.minorticks_on()
                for i in range(len(el_list)):
                    plt.axvline(x=el_list[i], color="k", linestyle="--", alpha=0.5)
                for i in range(number_sl):
                    if sl_fnl[i] == 1:
                        plt.axvline(
                            x=sl_center[i], color="brown", linestyle="-", alpha=1
                        )
                    else:
                        plt.axvline(
                            x=sl_center[i], color="y", linestyle="--", alpha=0.6
                        )
                for i in range(len(dsky2) - 1):
                    plt.axvline(x=dsky2[i], color="orange", linestyle="--", alpha=0.6)
                plt.show()
                plt.close()

            offset = np.nanmedian(
                np.array(object_sl_gaussian_center) - np.array(sl_gauss_center)
            )
            if verbose:
                #                reference_sl = 1 # Position in the file!
                #                sl_ref_ratio = sl_gaussian_flux/sl_gaussian_flux[reference_sl]
                #                print "\n n  line     fsky    fspec   fspec/fsky   l_obj-l_sky  fsky/6363.4   sigma_sky  sigma_fspec"
                #                #print "\n n    c_object c_sky   c_obj-c_sky"
                #                for i in range(number_sl):
                #                    if skip_sl_fit[i] == False: print "{:2} {:6.1f} {:8.2f} {:8.2f}    {:7.4f}      {:5.2f}      {:6.3f}    {:6.3f}  {:6.3f}" .format(i+1,sl_center[i],sl_gaussian_flux[i],object_sl_gaussian_flux[i],ratio_object_sky_sl_gaussian[i],object_sl_gaussian_center[i]-sl_gauss_center[i],sl_ref_ratio[i],sl_gaussian_sigma[i],object_sl_gaussian_sigma[i])
                #                    #if skip_sl_fit[i] == False: print  "{:2}   {:9.3f} {:9.3f}   {:9.3f}".format(i+1, object_sl_gaussian_center[i], sl_gauss_center[i], dif_center_obj_sky[i])
                #
                print("\n> Median center offset between OBJ and SKY :", offset, " A\n> Median gauss for the OBJECT ", np.nanmedian(
                    object_sl_gaussian_sigma
                ), " A")
                print("> Median flux OBJECT / SKY = ", np.nanmedian(
                    ratio_object_sky_sl_gaussian
                ))

            self.wavelength_offset_per_fibre.append(offset)

            # plt.plot(object_sl_gaussian_center, ratio_object_sky_sl_gaussian, "r+")

            if auto_scale_sky:
                if verbose:
                    print("\n> As requested, using this value to scale sky spectrum before substraction... ")
                auto_scale = np.nanmedian(ratio_object_sky_sl_gaussian)
                self.sky_auto_scale.append(np.nanmedian(ratio_object_sky_sl_gaussian))
                # self.sky_emission = auto_scale * self.sky_emission
            else:
                auto_scale = 1.0
                self.sky_auto_scale.append(1.0)

            if rebin:
                if verbose:
                    print("\n> Rebinning the spectrum of fibre", fibre, "to match sky spectrum...")
                f = object_sl_gaussian_fitted
                f_new = rebin_spec_shift(w, f, offset)
            else:
                f_new = object_sl_gaussian_fitted

            self.intensity_corrected[fibre] = (
                f_new - auto_scale * sky_sl_gaussian_fitted
            )

            # check offset center wavelength

    #            good_sl_center=[]
    #            good_sl_center_dif=[]
    #            plt.figure(figsize=(14, 4))
    #            for i in range(number_sl):
    #                if skip_sl_fit[i] == False:
    #                    plt.plot(sl_center[i],dif_center_obj_sky[i],"g+", alpha=0.7, label="Object")
    #                    good_sl_center.append(sl_center[i])
    #                    good_sl_center_dif.append(dif_center_obj_sky[i])
    #
    #            a1x,a0x = np.polyfit(good_sl_center, good_sl_center_dif, 1)
    #            fx = a0x + a1x*w
    #            #print a0x, a1x
    #            offset = np.nanmedian(good_sl_center_dif)
    #            print "median =",offset
    #            plt.plot(w,fx,"b", alpha=0.7, label="Fit")
    #            plt.axhline(y=offset, color='r', linestyle='--')
    #            plt.xlim(6100,9300)
    #            #plt.ylim(ymin,ymax)
    #            ptitle = "Fibre "+np.str(fibre)#+" with rms = "+np.str(rms[i])
    #            plt.title(ptitle)
    #            plt.xlabel("Wavelength [$\AA$]")
    #            plt.ylabel("c_obj - c_sky")
    #            #plt.legend(frameon=True, loc=2, ncol=4)
    #            plt.minorticks_on()
    #            plt.show()
    #            plt.close()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def do_extinction_curve(self, observatory_file="ssoextinct.dat", plot=True):

        print("\n> Computing extinction at given airmass...")

        # Read data
        data_observatory = np.loadtxt(observatory_file, unpack=True)
        extinction_curve_wavelengths = data_observatory[0]
        extinction_curve = data_observatory[1]
        extinction_corrected_airmass = 10 ** (0.4 * self.airmass * extinction_curve)

        # Make fit
        tck = interpolate.splrep(
            extinction_curve_wavelengths, extinction_corrected_airmass, s=0
        )
        self.extinction_correction = interpolate.splev(self.wavelength, tck, der=0)

        # Plot
        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(extinction_curve_wavelengths, extinction_corrected_airmass, "+")
            plt.xlim(np.min(self.wavelength), np.max(self.wavelength))
            cinco_por_ciento = 0.05 * (
                np.max(self.extinction_correction) - np.min(self.extinction_correction)
            )
            plt.ylim(
                np.min(self.extinction_correction) - cinco_por_ciento,
                np.max(self.extinction_correction) + cinco_por_ciento,
            )
            plt.plot(self.wavelength, self.extinction_correction, "g")
            plt.minorticks_on()
            plt.title("Correction for extinction using airmass = " + str(self.airmass))
            plt.ylabel("Flux correction")
            plt.xlabel("Wavelength [$\AA$]")
            plt.show()
            plt.close()

        # Correct for extinction at given airmass
        print("  Airmass = ", self.airmass)
        print("  Observatory file with extinction curve :", observatory_file)
        for i in range(self.n_spectra):
            self.intensity_corrected[i, :] = (
                self.intensity_corrected[i, :] * self.extinction_correction
            )
        print("  Intensities corrected for extinction stored in self.intensity_corrected !")

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def find_sky_emission(
        self,
        intensidad=[0, 0],
        plot=True,
        n_sky=200,
        sky_fibres=[1000],
        sky_wave_min=0,
        sky_wave_max=0,
        norm=colors.LogNorm(),
    ):

        if sky_wave_min == 0:
            sky_wave_min = self.valid_wave_min
        if sky_wave_max == 0:
            sky_wave_max = self.valid_wave_max

        if np.nanmedian(intensidad) == 0:
            intensidad = self.intensity_corrected
            ic = 1
        else:
            ic = 0

        if sky_fibres[0] == 1000:  # As it was original

            # sorted_by_flux = np.argsort(flux_ratio)   ORIGINAL till 21 Jan 2019
            # NEW 21 Jan 2019: Assuming cleaning of cosmics and CCD defects, we just use the spaxels with the LOWEST INTEGRATED VALUES
            self.compute_integrated_fibre(
                valid_wave_min=sky_wave_min, valid_wave_max=sky_wave_max, plot=False
            )
            sorted_by_flux = np.argsort(
                self.integrated_fibre
            )  # (self.integrated_fibre)
            print("\n> Identifying sky spaxels using the lowest integrated values in the [", sky_wave_min, ",", sky_wave_max, "] range ...")

            #            if plot:
            # #               print "\n  Plotting fluxes and flux ratio: "
            #                plt.figure(figsize=(10, 4))
            #                plt.plot(flux_ratio[sorted_by_flux], 'r-', label='flux ratio')
            #                plt.plot(flux_sky[sorted_by_flux], 'c-', label='flux sky')
            #                plt.plot(flux_object[sorted_by_flux], 'k-', label='flux object')
            #                plt.axvline(x=n_sky)
            #                plt.xlabel("Spaxel")
            #                plt.ylabel("Flux")
            #                plt.yscale('log')
            #                plt.legend(frameon=False, loc=4)
            #                plt.show()

            # Angel routine: just take n lowest spaxels!
            optimal_n = n_sky
            print("  We use the lowest", optimal_n, "fibres for getting sky. Their positions are:")
            # Compute sky spectrum and plot it
            self.sky_fibres = sorted_by_flux[:optimal_n]
            self.sky_emission = np.nanmedian(
                intensidad[sorted_by_flux[:optimal_n]], axis=0
            )
            print("  List of fibres used for sky saved in self.sky_fibres")

        else:  # We provide a list with sky positions
            print("  We use the list provided to get the sky spectrum")
            print("  sky_fibres = ", sky_fibres)
            self.sky_fibres = np.array(sky_fibres)
            self.sky_emission = np.nanmedian(intensidad[self.sky_fibres], axis=0)

        if plot:
            self.RSS_map(
                self.integrated_fibre, None, self.sky_fibres, title=" - Sky Spaxels"
            )  # flux_ratio
            # print "  Combined sky spectrum:"
            plt.figure(figsize=(10, 4))
            plt.plot(self.wavelength, self.sky_emission, "c-", label="sky")
            plt.yscale("log")
            plt.ylabel("FLux")
            plt.xlabel("Wavelength [$\AA$]")
            plt.xlim(self.wavelength[0] - 10, self.wavelength[-1] + 10)
            plt.axvline(x=self.valid_wave_min, color="k", linestyle="--")
            plt.axvline(x=self.valid_wave_max, color="k", linestyle="--")
            plt.ylim([np.nanmin(intensidad), np.nanmax(intensidad)])
            plt.minorticks_on()
            plt.title(self.description + " - Combined Sky Spectrum")
            plt.legend(frameon=False)
            plt.show()
            plt.close()

        # Substract sky in all intensities
        self.intensity_sky_corrected = np.zeros_like(self.intensity)
        for i in range(self.n_spectra):
            if ic == 1:
                self.intensity_corrected[i, :] = (
                    self.intensity_corrected[i, :] - self.sky_emission
                )
            if ic == 0:
                self.intensity_sky_corrected[i, :] = (
                    self.intensity_corrected[i, :] - self.sky_emission
                )

        last_sky_fibre = self.sky_fibres[-1]
        # Plot median value of fibre vs. fibre
        if plot:
            median_sky_corrected = np.zeros(self.n_spectra)
            for i in range(self.n_spectra):
                if ic == 1:
                    median_sky_corrected[i] = np.nanmedian(
                        self.intensity_corrected[i, :], axis=0
                    )
                if ic == 0:
                    median_sky_corrected[i] = np.nanmedian(
                        self.intensity_sky_corrected[i, :], axis=0
                    )
            plt.figure(figsize=(10, 4))
            plt.plot(median_sky_corrected)
            plt.plot(
                [0, 1000],
                [
                    median_sky_corrected[last_sky_fibre],
                    median_sky_corrected[last_sky_fibre],
                ],
                "r",
            )
            plt.minorticks_on()
            plt.ylabel("Median Flux")
            plt.xlabel("Fibre")
            plt.yscale("log")
            plt.ylim([np.nanmin(median_sky_corrected), np.nanmax(median_sky_corrected)])
            plt.title(self.description)
            plt.legend(frameon=False)
            plt.show()
            plt.close()

        print("  Sky spectrum obtained and stored in self.sky_emission !! ")
        print("  Intensities corrected for sky emission and stored in self.intensity_corrected !")

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def find_relative_throughput(
        self,
        ymin=10000,
        ymax=200000,  # nskyflat=False,
        kernel_sky_spectrum=5,
        wave_min_scale=0,
        wave_max_scale=0,
        plot=True,
    ):
        # These are for the normalized flat:
        # fit_skyflat_degree=0, step=50, wave_min_flat=0, wave_max_flat=0):
        """
        Determine the relative transmission of each spectrum
        using a skyflat.
        """

        print("\n> Using this skyflat to find relative throughput (a scale per fibre)...")

        # Check grating to chose wavelength range to get median values
        if wave_min_scale == 0 and wave_max_scale == 0:
            if self.grating == "1000R":
                wave_min_scale = 6600.0
                wave_max_scale = 6800.0
                print("  For 1000R, we use the median value in the [6600, 6800] range.")
            if self.grating == "1500V":
                wave_min_scale = 5100.0
                wave_max_scale = 5300.0
                print("  For 1500V, we use the median value in the [5100, 5300] range.")
            if self.grating == "580V":
                wave_min_scale = 4700.0
                wave_max_scale = 4800.0
                print("  For 580V, we use the median value in the [4700, 4800] range.")
            if self.grating == "385R":
                wave_min_scale = 6600.0
                wave_max_scale = 6800.0
                print("  For 385R, we use the median value in the [6600, 6800] range.")
        else:
            if wave_min_scale == 0:
                wave_min_scale = self.wavelength[0]
            if wave_max_scale == 0:
                wave_max_scale = self.wavelength[-1]
            print("  As given by the user, we use the median value in the [", wave_min_scale, ",", wave_max_scale, "] range.")

        median_region = np.zeros(self.n_spectra)
        for i in range(self.n_spectra):
            region = np.where(
                (self.wavelength > wave_min_scale) & (self.wavelength < wave_max_scale)
            )
            median_region[i] = np.nanmedian(self.intensity[i, region])

        median_value_skyflat = np.nanmedian(median_region)
        self.relative_throughput = old_div(median_region, median_value_skyflat)
        print("  Median value of skyflat in the [", wave_min_scale, ",", wave_max_scale, "] range = ", median_value_skyflat)
        print("  Individual fibre corrections:  min =", np.nanmin(
            self.relative_throughput
        ), " max =", np.nanmax(self.relative_throughput))

        if plot:
            plt.figure(figsize=(10, 4))
            x = list(range(self.n_spectra))
            plt.plot(x, self.relative_throughput)
            # plt.ylim(0.5,4)
            plt.minorticks_on()
            plt.xlabel("Fibre")
            plt.ylabel("Throughput using scale")
            plt.title("Throughput correction using scale")
            plt.show()
            plt.close()

            # print "\n  Plotting spectra WITHOUT considering throughput correction..."
            plt.figure(figsize=(10, 4))
            for i in range(self.n_spectra):
                plt.plot(self.wavelength, self.intensity[i, ])
            plt.xlabel("Wavelength [$\AA$]")
            plt.title("Spectra WITHOUT considering any throughput correction")
            plt.xlim(self.wavelength[0] - 10, self.wavelength[-1] + 10)
            plt.ylim(ymin, ymax)
            plt.minorticks_on()
            plt.show()
            plt.close()

            # print "  Plotting spectra CONSIDERING throughput correction..."
            plt.figure(figsize=(10, 4))
            for i in range(self.n_spectra):
                # self.intensity_corrected[i,] = self.intensity[i,] * self.relative_throughput[i]
                plot_this = old_div(self.intensity[i, ], self.relative_throughput[i])
                plt.plot(self.wavelength, plot_this)
            plt.ylim(ymin, ymax)
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            plt.title("Spectra CONSIDERING throughput correction (scale)")
            plt.xlim(self.wavelength[0] - 10, self.wavelength[-1] + 10)
            plt.axvline(x=wave_min_scale, color="k", linestyle="--")
            plt.axvline(x=wave_max_scale, color="k", linestyle="--")
            plt.show()
            plt.close()

        print("\n>  Using median value of skyflat considering a median filter of", kernel_sky_spectrum, "...")  # LUKE
        median_sky_spectrum = np.nanmedian(self.intensity, axis=0)
        self.response_sky_spectrum = np.zeros_like(self.intensity)
        rms = np.zeros(self.n_spectra)
        plot_fibres = [100, 500, 501, 900]
        pf = 0
        for i in range(self.n_spectra):
            self.response_sky_spectrum[i] = (
                old_div(old_div(self.intensity[i], self.relative_throughput[i]), median_sky_spectrum)
            )
            filter_response_sky_spectrum = sig.medfilt(
                self.response_sky_spectrum[i], kernel_size=kernel_sky_spectrum
            )
            rms[i] = old_div(np.nansum(
                np.abs(self.response_sky_spectrum[i] - filter_response_sky_spectrum)
            ), np.nansum(self.response_sky_spectrum[i]))

            if plot:
                if i == plot_fibres[pf]:
                    plt.figure(figsize=(10, 4))
                    plt.plot(
                        self.wavelength,
                        self.response_sky_spectrum[i],
                        alpha=0.3,
                        label="Response Sky",
                    )
                    plt.plot(
                        self.wavelength,
                        filter_response_sky_spectrum,
                        alpha=0.7,
                        linestyle="--",
                        label="Filtered Response Sky",
                    )
                    plt.plot(
                        self.wavelength,
                        old_div(self.response_sky_spectrum[i], filter_response_sky_spectrum),
                        alpha=1,
                        label="Normalized Skyflat",
                    )
                    plt.xlim(self.wavelength[0] - 50, self.wavelength[-1] + 50)
                    plt.ylim(0.95, 1.05)
                    ptitle = "Fibre " + np.str(i) + " with rms = " + np.str(rms[i])
                    plt.title(ptitle)
                    plt.xlabel("Wavelength [$\AA$]")
                    plt.legend(frameon=False, loc=3, ncol=1)
                    plt.show()
                    plt.close()
                    if pf < len(plot_fibres) - 1:
                        pf = pf + 1

        print("  median rms = ", np.nanmedian(rms), "  min rms = ", np.nanmin(
            rms
        ), "  max rms = ", np.nanmax(rms))
        #        if plot:
        #            plt.figure(figsize=(10, 4))
        #            for i in range(self.n_spectra):
        #                #plt.plot(self.wavelength,self.intensity[i,]/median_sky_spectrum)
        #                plot_this =  self.intensity[i,] / self.relative_throughput[i] /median_sky_spectrum
        #                plt.plot(self.wavelength, plot_this)
        #            plt.xlabel("Wavelength [$\AA$]")
        #            plt.title("Spectra CONSIDERING throughput correction (scale) / median sky spectrum")
        #            plt.xlim(self.wavelength[0]-10,self.wavelength[-1]+10)
        #            plt.ylim(0.7,1.3)
        #            plt.minorticks_on()
        #            plt.show()
        #            plt.close()
        #
        #            plt.plot(self.wavelength, median_sky_spectrum, color='r',alpha=0.7)
        #            plt.plot(self.wavelength, filter_median_sky_spectrum, color='blue',alpha=0.7)
        #            plt.show()
        #            plt.close()
        #
        #            plt.plot(self.wavelength, median_sky_spectrum/filter_median_sky_spectrum, color='r',alpha=0.7)
        #            plt.show()
        #            plt.close()

        #            for i in range(2):
        #                response_sky_spectrum_ = self.intensity[500+i,] / self.relative_throughput[500+i] /median_sky_spectrum
        #                filter_response_sky_spectrum = sig.medfilt(response_sky_spectrum_,kernel_size=kernel_sky_spectrum)
        #                rms=np.nansum(np.abs(response_sky_spectrum_ - filter_response_sky_spectrum))/np.nansum(response_sky_spectrum_)

        # for i in range(5):
        #                filter_response_sky_spectrum_ = (self.intensity[500+i,] / self.relative_throughput[500+i] ) / median_sky_spectrum
        #                filter_response_sky_spectrum = sig.medfilt(filter_response_sky_spectrum_,kernel_size=kernel_sky_spectrum)
        #
        #                plt.plot(self.wavelength, filter_response_sky_spectrum,alpha=0.7)
        #            plt.ylim(0.95,1.05)
        #            plt.show()
        #            plt.close()

        print("\n> Relative throughput using skyflat scaled stored in self.relative_throughput !!")

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def get_telluric_correction(
        self,
        n_fibres=10,
        correct_from=6850.0,
        correct_to=10000.0,
        apply_tc=False,
        step=10,
        combined_cube=False,
        weight_fit_median=0.5,
        exclude_wlm=[
            [6450, 6700],
            [6850, 7050],
            [7130, 7380],
        ],  # This is range for 1000R
        wave_min=0,
        wave_max=0,
        plot=True,
        fig_size=12,
        verbose=False,
    ):
        """
        Get telluric correction using a spectrophotometric star

        Parameters
        ----------
        n_fibres: integer
            number of fibers to add for obtaining spectrum
        correct_from :  float
            wavelength from which telluric correction is applied (default = 6850)
        apply_tc : boolean (default = False)
            apply telluric correction to data
        exclude_wlm=[[6450,6700],[6850,7050], [7130,7380]]:
            Wavelength ranges not considering for normalising stellar continuum

        Example
        ----------
        telluric_correction_star1 = star1r.get_telluric_correction(n_fibres=15)
        """

        print("\n> Obtaining telluric correction using spectrophotometric star...")

        if combined_cube:
            wlm = self.combined_cube.wavelength
        else:
            wlm = self.wavelength

        if wave_min == 0:
            wave_min = wlm[0]
        if wave_max == 0:
            wave_max = wlm[-1]

        if combined_cube:
            if self.combined_cube.seeing == 0:
                self.combined_cube.half_light_spectrum(
                    5, plot=plot, min_wave=wave_min, max_wave=wave_max
                )
            estrella = self.combined_cube.integrated_star_flux

        else:
            integrated_intensity_sorted = np.argsort(self.integrated_fibre)
            intensidad = self.intensity_corrected
            region = []
            for fibre in range(n_fibres):
                region.append(integrated_intensity_sorted[-1 - fibre])
            estrella = np.nansum(intensidad[region], axis=0)

        smooth_med_star = smooth_spectrum(
            wlm,
            estrella,
            wave_min=wave_min,
            wave_max=wave_max,
            step=step,
            weight_fit_median=weight_fit_median,
            exclude_wlm=exclude_wlm,
            plot=plot,
            verbose=verbose,
        )

        telluric_correction = np.ones(len(wlm))
        for l in range(len(wlm)):
            if wlm[l] > correct_from and wlm[l] < correct_to:
                telluric_correction[l] = old_div(smooth_med_star[l], estrella[l])

        if plot:
            plt.figure(figsize=(fig_size, fig_size / 2.5))
            if combined_cube:
                print("  Telluric correction for this star (" + self.combined_cube.object + ") :")
                plt.plot(wlm, estrella, color="b", alpha=0.3)
                plt.plot(wlm, estrella * telluric_correction, color="g", alpha=0.5)
                plt.ylim(np.nanmin(estrella), np.nanmax(estrella))

            else:
                print("  Example of telluric correction using fibres", region[
                    0
                ], " and ", region[1], ":")

                plt.plot(wlm, intensidad[region[0]], color="b", alpha=0.3)
                plt.plot(
                    wlm,
                    intensidad[region[0]] * telluric_correction,
                    color="g",
                    alpha=0.5,
                )
                plt.plot(wlm, intensidad[region[1]], color="b", alpha=0.3)
                plt.plot(
                    wlm,
                    intensidad[region[1]] * telluric_correction,
                    color="g",
                    alpha=0.5,
                )
                plt.ylim(
                    np.nanmin(intensidad[region[1]]), np.nanmax(intensidad[region[0]])
                )  # CHECK THIS AUTOMATICALLY

            plt.axvline(x=wave_min, color="k", linestyle="--")
            plt.axvline(x=wave_max, color="k", linestyle="--")
            plt.xlim(wlm[0] - 10, wlm[-1] + 10)

            plt.xlabel("Wavelength [$\AA$]")
            if exclude_wlm[0][0] != 0:
                for i in range(len(exclude_wlm)):
                    plt.axvspan(
                        exclude_wlm[i][0], exclude_wlm[i][1], color="r", alpha=0.1
                    )
            plt.minorticks_on()
            plt.show()
            plt.close()

        if apply_tc:  # Check this
            print("  Applying telluric correction to this star...")
            if combined_cube:
                self.combined_cube.integrated_star_flux = (
                    self.combined_cube.integrated_star_flux * telluric_correction
                )
                for i in range(self.combined_cube.n_rows):
                    for j in range(self.combined_cube.n_cols):
                        self.combined_cube.data[:, i, j] = (
                            self.combined_cube.data[:, i, j] * telluric_correction
                        )
            else:
                for i in range(self.n_spectra):
                    self.intensity_corrected[i, :] = (
                        self.intensity_corrected[i, :] * telluric_correction
                    )

        return telluric_correction

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_spectrum(self, spectrum_number, sky=True, xmin=0, xmax=0, ymax=0, ymin=0):
        """
        Plot spectrum of a particular spaxel.

        Parameters
        ----------
        spectrum_number:
            spaxel to show spectrum.
        sky:
            if True substracts the sky

        Example
        -------
        >>> rss1.plot_spectrum(550, sky=True)
        """

        if sky:
            spectrum = self.intensity_corrected[spectrum_number]
        else:
            spectrum = self.intensity_corrected[spectrum_number] + self.sky_emission

        plt.plot(self.wavelength, spectrum)
        # error = 3*np.sqrt(self.variance[spectrum_number])
        # plt.fill_between(self.wavelength, spectrum-error, spectrum+error, alpha=.1)

        if xmin != 0 or xmax != 0 or ymax != 0 or ymin != 0:

            if xmin == 0:
                xmin = self.wavelength[0]
            if xmax == 0:
                xmax = self.wavelength[-1]
            if ymin == 0:
                ymin = np.nanmin(spectrum)
            if ymax == 0:
                ymax = np.nanmax(spectrum)

            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            plt.ylabel("Relative Flux")
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.show()
            plt.close()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_spectra(
        self,
        list_spectra="all",
        wavelength_range=[0],
        xmin="",
        xmax="",
        ymax=1000,
        ymin=-100,
        fig_size=10,
        save_file="",
        sky=True,
    ):
        """
        Plot spectrum of a list pf spaxels.

        Parameters
        ----------
        list_spectra:
            spaxels to show spectrum. Default is all.
        save_file:
            (Optional) Save plot in file "file.extension"
        fig_size:
            Size of the figure (in x-axis), default: fig_size=10
        Example
        -------
        >>> rss1.plot_spectra([1200,1300])
        """
        plt.figure(figsize=(fig_size, fig_size / 2.5))

        if list_spectra == "all":
            list_spectra = list(range(self.n_spectra))
        if len(wavelength_range) == 2:
            plt.xlim(wavelength_range[0], wavelength_range[1])
        if xmin == "":
            xmin = np.nanmin(self.wavelength)
        if xmax == "":
            xmax = np.nanmax(self.wavelength)

        #        title = "Spectrum of spaxel {} in {}".format(spectrum_number, self.description)
        #        plt.title(title)
        plt.minorticks_on()
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel("Relative Flux")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        for i in list_spectra:
            self.plot_spectrum(i, sky)

        if save_file == "":
            plt.show()
        else:
            plt.savefig(save_file)
        plt.close()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_combined_spectrum(
        self,
        list_spectra="all",
        sky=True,
        median=False,
        xmin="",
        xmax="",
        ymax="",
        ymin="",
        fig_size=10,
        save_file="",
        plot=True,
    ):
        """
        Plot combined spectrum of a list and return the combined spectrum.

        Parameters
        ----------
        list_spectra:
            spaxels to show combined spectrum. Default is all.
        sky:
            if True substracts the sky
        Example
        -------
        >>> rss1.plot_spectrum(550, sky=True)
        """

        if list_spectra == "all":
            list_spectra = list(range(self.n_spectra))

        spectrum = np.zeros_like(self.intensity_corrected[list_spectra[0]])
        value_list = []
        # Note: spectrum of fibre is located in position fibre-1, e.g., spectrum of fibre 1 -> intensity_corrected[0]
        if sky:
            for fibre in list_spectra:
                value_list.append(self.intensity_corrected[fibre - 1])
        else:
            for fibre in list_spectra:
                value_list.append(
                    self.intensity_corrected[fibre - 1] + self.sky_emission
                )

        if median:
            spectrum = np.nanmedian(value_list, axis=0)
        else:
            spectrum = np.nansum(value_list, axis=0)

        if plot:
            plt.figure(figsize=(fig_size, fig_size / 2.5))

            if xmin == "":
                xmin = np.nanmin(self.wavelength)
            if xmax == "":
                xmax = np.nanmax(self.wavelength)
            if ymin == "":
                ymin = np.nanmin(spectrum)
            if ymax == "":
                ymax = np.nanmax(spectrum)

            plt.plot(self.wavelength, spectrum)

            if len(list_spectra) == list_spectra[-1] - list_spectra[0] + 1:
                title = "{} - Combined spectrum in range [{},{}]".format(
                    self.description, list_spectra[0], list_spectra[-1]
                )
            else:
                title = "Combined spectrum using requested fibres"

            plt.title(title)
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            plt.ylabel("Relative Flux")
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            if save_file == "":
                plt.show()
            else:
                plt.savefig(save_file)
            plt.close()
        return spectrum

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def flux_between(self, lambda_min, lambda_max, list_spectra=[]):
        index_min = np.searchsorted(self.wavelength, lambda_min)
        index_max = np.searchsorted(self.wavelength, lambda_max) + 1
        if len(list_spectra) == 0:
            list_spectra = list(range(self.n_spectra))

        n_spectra = len(list_spectra)
        fluxes = np.empty(n_spectra)
        variance = np.empty(n_spectra)
        for i in range(n_spectra):
            fluxes[i] = np.nanmean(self.intensity[list_spectra[i], index_min:index_max])
            variance[i] = np.nanmean(
                self.variance[list_spectra[i], index_min:index_max]
            )

        return fluxes * (lambda_max - lambda_min), variance * (lambda_max - lambda_min)

    #        WARNING: Are we overestimating errors?
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def median_between(self, lambda_min, lambda_max, list_spectra=[]):
        index_min = np.searchsorted(self.wavelength, lambda_min)
        index_max = np.searchsorted(self.wavelength, lambda_max) + 1
        if len(list_spectra) == 0:
            list_spectra = list(range(self.n_spectra))

        n_spectra = len(list_spectra)
        medians = np.empty(n_spectra)
        for i in range(n_spectra):
            medians[i] = np.nanmedian(
                self.intensity[list_spectra[i], index_min:index_max]
            )
        return medians

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def line_flux(
        self,
        left_min,
        left_max,
        line_min,
        line_max,
        right_min,
        right_max,
        list_spectra=[],
    ):
        if len(list_spectra) == 0:
            list_spectra = list(range(self.n_spectra))

        line, var_line = self.flux_between(line_min, line_max, list_spectra)
        left, var_left = old_div(self.flux_between(left_min, left_max, list_spectra), (
            left_max - left_min
        ))
        right, var_right = old_div(self.flux_between(right_min, right_max, list_spectra), (
            left_max - left_min
        ))
        wavelength_left = old_div((left_min + left_max), 2)
        wavelength_line = old_div((line_min + line_max), 2)
        wavelength_right = old_div((right_min + right_max), 2)
        continuum = left + old_div((right - left) * (wavelength_line - wavelength_left), (
            wavelength_right - wavelength_left
        ))
        var_continuum = old_div((var_left + var_right), 2)

        return (
            line - continuum * (line_max - line_min),
            var_line + var_continuum * (line_max - line_min),
        )

    #           WARNING: Are we overestimating errors?
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def RSS_map(
        self,
        variable,
        norm=colors.LogNorm(),
        list_spectra=[],
        title=" - RSS map",
        color_bar_text="Integrated Flux [Arbitrary units]",
    ):
        """
        Plot map showing the offsets, coloured by variable.
        """
        if len(list_spectra) == 0:
            list_spectra = list(range(self.n_spectra))

        plt.figure(figsize=(10, 10))
        plt.scatter(
            self.offset_RA_arcsec[list_spectra],
            self.offset_DEC_arcsec[list_spectra],
            c=variable[list_spectra],
            cmap=fuego_color_map,
            norm=norm,
            s=260,
            marker="h",
        )
        plt.title(self.description + title)
        plt.xlim(
            np.nanmin(self.offset_RA_arcsec) - 0.7,
            np.nanmax(self.offset_RA_arcsec) + 0.7,
        )
        plt.ylim(
            np.nanmin(self.offset_DEC_arcsec) - 0.7,
            np.nanmax(self.offset_DEC_arcsec) + 0.7,
        )
        plt.xlabel("$\Delta$ RA [arcsec]")
        plt.ylabel("$\Delta$ DEC [arcsec]")
        plt.minorticks_on()
        plt.grid(which="both")
        plt.gca().invert_xaxis()

        cbar = plt.colorbar()
        plt.clim(np.nanmin(variable[list_spectra]), np.nanmax(variable[list_spectra]))
        cbar.set_label(str(color_bar_text), rotation=90, labelpad=40)
        cbar.ax.tick_params()

        plt.show()
        plt.close()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def RSS_image(
        self,
        image=[0],
        norm=colors.Normalize(),
        cmap="seismic_r",
        clow=0,
        chigh=0,
        labelpad=10,
        title=" - RSS image",
        color_bar_text="Integrated Flux [Arbitrary units]",
        fig_size=13.5,
    ):
        """
        Plot RSS image coloured by variable.
        cmap = "binary_r" nice greyscale
        """

        if np.nanmedian(image) == 0:
            image = self.intensity_corrected

        if clow == 0:
            clow = np.nanpercentile(image, 5)
        if chigh == 0:
            chigh = np.nanpercentile(image, 95)
        if cmap == "seismic_r":
            max_abs = np.nanmax([np.abs(clow), np.abs(chigh)])
            clow = -max_abs
            chigh = max_abs

        plt.figure(figsize=(fig_size, fig_size / 2.5))
        plt.imshow(image, norm=norm, cmap=cmap, clim=(clow, chigh))
        plt.title(self.description + title)
        plt.minorticks_on()
        plt.gca().invert_yaxis()

        # plt.colorbar()
        cbar = plt.colorbar()
        cbar.set_label(str(color_bar_text), rotation=90, labelpad=labelpad)
        cbar.ax.tick_params()

        plt.show()
        plt.close()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_corrected_vs_uncorrected_spectrum(self, high_fibres=10, fig_size=12):

        integrated_intensity_sorted = np.argsort(self.integrated_fibre)
        region = []
        for fibre_ in range(high_fibres):
            region.append(integrated_intensity_sorted[-1 - fibre_])

        plt.figure(figsize=(fig_size, fig_size / 2.5))
        I = np.nansum(self.intensity[region], axis=0)
        plt.plot(self.wavelength, I, "r-", label="Uncorrected", alpha=0.3)
        Ic = np.nansum(self.intensity_corrected[region], axis=0)
        I_ymin = np.nanmin([np.nanmin(I), np.nanmin(Ic)])
        I_ymax = np.nanmax([np.nanmax(I), np.nanmax(Ic)])
        I_rango = I_ymax - I_ymin
        plt.plot(self.wavelength, Ic, "g-", label="Corrected", alpha=0.4)
        plt.ylabel("Flux")
        plt.xlabel("Wavelength [$\AA$]")
        plt.minorticks_on()
        plt.xlim(self.wavelength[0] - 10, self.wavelength[-1] + 10)
        plt.axvline(x=self.valid_wave_min, color="k", linestyle="--")
        plt.axvline(x=self.valid_wave_max, color="k", linestyle="--")
        plt.ylim([I_ymin - old_div(I_rango, 10), I_ymax + old_div(I_rango, 10)])
        plt.title(
            self.object
            + " - Combined spectrum - "
            + str(high_fibres)
            + " fibres with highest intensity"
        )
        plt.legend(frameon=False, loc=4, ncol=2)
        plt.show()
        plt.close()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # Idea: take a RSS dominated by skylines. Read it (only throughput correction). For each fibre, fit Gaussians to ~10 skylines.
    # Compare with REST wavelengths. Get a median value per fibre. Perform a second-order fit to all median values.
    # Correct for that using a reference fibre (1). Save results to be applied to the rest of files of the night (assuming same configuration).

    def fix_2dfdr_wavelengths(
        self,
        sol=[0, 0, 0],
        fibre=0,
        maxima_sigma=2.5,
        maxima_offset=1.5,
        xmin=7740,
        xmax=7770,
        ymin=0,
        ymax=1000,
        plot=True,
        verbose=True,
        warnings=True,
    ):

        print("\n> Fixing 2dfdr wavelengths using skylines.")

        w = self.wavelength

        if sol[0] == 0:  # Solutions are not given

            # Read file with sky emission line
            sky_lines_file = "sky_lines_rest.dat"
            (
                sl_center,
                sl_name,
                sl_fnl,
                sl_lowlow,
                sl_lowhigh,
                sl_highlow,
                sl_highhigh,
                sl_lmin,
                sl_lmax,
            ) = read_table(
                sky_lines_file, ["f", "s", "f", "f", "f", "f", "f", "f", "f"]
            )
            number_sl = len(sl_center)

            # Fitting Gaussians to skylines...
            say_status = 0
            self.wavelength_offset_per_fibre = []
            wave_median_offset = []
            print("\n> Performing a Gaussian fit to selected, bright skylines... (this will FAIL if RSS is not corrected for CCD defects...)")

            if fibre != 0:
                f_i = fibre
                f_f = fibre + 1
                print("  Checking fibre ", fibre, " (only this fibre is corrected, use fibre = 0 for all)...")
                verbose = True
                warnings = True
            else:
                f_i = 0
                f_f = self.n_spectra
                verbose = False
            for fibre in range(f_i, f_f):  # (self.n_spectra):

                spectrum = self.intensity_corrected[fibre]
                if fibre == say_status:
                    print("  Checking fibre {:4} ...  ({:6.2f} % completed) ...".format(
                        fibre, fibre * 100.0 / self.n_spectra
                    ))
                    say_status = say_status + 20

                # Gaussian fits to the sky spectrum
                sl_gaussian_flux = []
                sl_gaussian_sigma = []
                sl_gauss_center = []
                sl_offset = []
                sl_offset_good = []

                if verbose:
                    print("\n> Performing Gaussian fitting to bright sky lines in all fibres of rss file...")
                for i in range(number_sl):
                    if sl_fnl[i] == 0:
                        plot_fit = False
                    else:
                        plot_fit = True

                    resultado = fluxes(
                        w,
                        spectrum,
                        sl_center[i],
                        lowlow=sl_lowlow[i],
                        lowhigh=sl_lowhigh[i],
                        highlow=sl_highlow[i],
                        highhigh=sl_highhigh[i],
                        lmin=sl_lmin[i],
                        lmax=sl_lmax[i],
                        fmin=0,
                        fmax=0,
                        broad=2.1 * 2.355,
                        plot=plot_fit,
                        verbose=False,
                        plot_sus=False,
                        fcal=False,
                        warnings=warnings,
                    )  # Broad is FWHM for Gaussian sigm a= 1,

                    sl_gaussian_flux.append(resultado[3])
                    sl_gauss_center.append(resultado[1])
                    sl_gaussian_sigma.append(resultado[5] / 2.355)
                    sl_offset.append(sl_gauss_center[i] - sl_center[i])

                    if (
                        sl_gaussian_flux[i] < 0
                        or np.abs(sl_center[i] - sl_gauss_center[i]) > maxima_offset
                        or sl_gaussian_sigma[i] > maxima_sigma
                    ):
                        if verbose:
                            print("  Bad fitting for ", sl_center[
                                i
                            ], "... ignoring this fit...")
                    else:
                        sl_offset_good.append(sl_offset[i])
                        if verbose:
                            print("  Fitted wavelength for sky line {:8.3f}:    center = {:8.3f}     sigma = {:6.3f}    offset = {:7.3f} ".format(
                                sl_center[i],
                                sl_gauss_center[i],
                                sl_gaussian_sigma[i],
                                sl_offset[i],
                            ))

                median_offset_fibre = np.nanmedian(sl_offset_good)
                wave_median_offset.append(median_offset_fibre)
                if verbose:
                    print("\n> Median offset for fibre {:3} = {:7.3f}".format(
                        fibre, median_offset_fibre
                    ))

            # Second-order fit ...
            xfibre = list(range(0, self.n_spectra))
            a2x, a1x, a0x = np.polyfit(xfibre, wave_median_offset, 2)
            print("\n> Fitting a second-order polynomy a0x +  a1x * fibre + a2x * fibre**2:")
        else:
            print("\n> Solution to the second-order polynomy a0x +  a1x * fibre + a2x * fibre**2 have been provided:")
            a0x = sol[0]
            a1x = sol[1]
            a2x = sol[2]
            xfibre = list(range(0, self.n_spectra))

        print("  a0x =", a0x, "   a1x =", a1x, "     a2x =", a2x)
        self.wavelength_parameters = [a0x, a1x, a2x]  # Save solutions

        fx = a0x + a1x * np.array(xfibre) + a2x * np.array(xfibre) ** 2

        if plot:
            plt.figure(figsize=(10, 4))
            if sol[0] == 0:
                plt.plot(xfibre, wave_median_offset)
                pf = wave_median_offset
            else:
                pf = fx
            plt.plot(xfibre, fx, "r")
            plot_plot(
                xfibre,
                pf,
                ptitle="Second-order fit to individual offsets",
                xmin=-20,
                xmax=1000,
                xlabel="Fibre",
                ylabel="offset",
            )

        # Applying results
        print("\n> Applying results to all fibres...")
        for fibre in xfibre:
            f = self.intensity_corrected[fibre]
            w_shift = fx[fibre]
            self.intensity_corrected[fibre] = rebin_spec_shift(w, f, w_shift)

        # Check results
        if plot:
            plt.figure(figsize=(10, 4))
            for i in [0, 300, 600, 950]:
                plt.plot(w, self.intensity[i])
            plot_plot(
                w,
                self.intensity[0],
                ptitle="Before corrections, fibres 0, 300, 600, 950",
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
            )

            plt.figure(figsize=(10, 4))
            for i in [0, 300, 600, 950]:
                plt.plot(w, self.intensity_corrected[i])
            plot_plot(
                w,
                self.intensity_corrected[0],
                ptitle="Checking wavelength corrections in fibres 0, 300, 600, 950",
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
            )

        print("\n> Small fixing of the 2dFdr wavelengths done!")
        return


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
# KOALA_RSS CLASS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
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
    def __init__(
        self,
        filename,
        save_rss_to_fits_file="",
        rss_clean=False,  # TASK_KOALA_RSS
        apply_throughput=True,
        skyflat="",
        plot_skyflat=False,
        flat="",
        nskyflat=True,
        correct_ccd_defects=False,
        correct_high_cosmics=False,
        clip_high=100,
        step_ccd=50,
        remove_5578=False,
        plot_suspicious_fibres=False,
        fix_wavelengths=False,
        sol=[0, 0, 0],
        sky_method="self",
        n_sky=50,
        sky_fibres=[1000],  # do_sky=True
        sky_spectrum=[0],
        sky_rss=[0],
        scale_sky_rss=0,
        scale_sky_1D=1.0,
        is_sky=False,
        win_sky=151,
        auto_scale_sky=False,
        correct_negative_sky=False,
        sky_wave_min=0,
        sky_wave_max=0,
        cut_sky=5.0,
        fmin=1,
        fmax=10,
        individual_sky_substraction=False,
        fibre_list=[100, 200, 300, 400, 500, 600, 700, 800, 900],
        do_extinction=True,
        telluric_correction=[0],
        id_el=False,
        high_fibres=10,
        brightest_line="Ha",
        cut=1.5,
        broad=1.0,
        plot_id_el=False,
        id_list=[0],
        brightest_line_wavelength=0,
        clean_sky_residuals=False,
        dclip=3.0,
        extra_w=1.3,
        step_csr=25,
        fibre=0,
        valid_wave_min=0,
        valid_wave_max=0,
        warnings=True,
        verbose=False,
        plot=True,
        norm=colors.LogNorm(),
        fig_size=12,
    ):

        # Just read file if rss_clean = True
        if rss_clean:
            apply_throughput = False
            correct_ccd_defects = False
            fix_wavelengths = False
            sol = [0, 0, 0]
            sky_method = "none"
            do_extinction = False
            telluric_correction = [0]
            id_el = False
            clean_sky_residuals = False
            plot = False
            correct_negative_sky = False

        # Create RSS object
        super(KOALA_RSS, self).__init__()

        print("\n> Reading file", '"' + filename + '"', "...")
        RSS_fits_file = fits.open(filename)  # Open file
        self.rss_list = []

        #  General info:
        self.object = RSS_fits_file[0].header["OBJECT"]
        self.description = self.object + " - " + filename
        self.RA_centre_deg = old_div(RSS_fits_file[2].header["CENRA"] * 180, np.pi)
        self.DEC_centre_deg = old_div(RSS_fits_file[2].header["CENDEC"] * 180, np.pi)
        self.exptime = RSS_fits_file[0].header["EXPOSED"]
        #  WARNING: Something is probably wrong/inaccurate here!
        #  Nominal offsets between pointings are totally wrong!

        # Read good/bad spaxels
        all_spaxels = list(range(len(RSS_fits_file[2].data)))
        quality_flag = [RSS_fits_file[2].data[i][1] for i in all_spaxels]
        good_spaxels = [i for i in all_spaxels if quality_flag[i] == 1]
        bad_spaxels = [i for i in all_spaxels if quality_flag[i] == 0]

        #        for i in range(1):
        #            print i, RSS_fits_file[2]
        #

        # Create wavelength, intensity, and variance arrays only for good spaxels
        wcsKOALA = WCS(RSS_fits_file[0].header)
        # variance = RSS_fits_file[1].data[good_spaxels]
        index_wave = np.arange(RSS_fits_file[0].header["NAXIS1"])
        wavelength = wcsKOALA.dropaxis(1).wcs_pix2world(index_wave, 0)[0]
        intensity = RSS_fits_file[0].data[good_spaxels]

        print("\n  Number of spectra in this RSS =", len(
            RSS_fits_file[0].data
        ), ",  number of good spectra =", len(
            good_spaxels
        ), " ,  number of bad spectra =", len(
            bad_spaxels
        ))
        print("  Bad fibres =", bad_spaxels)

        # Read errors using RSS_fits_file[1]
        # self.header1 = RSS_fits_file[1].data      # CHECK WHEN DOING ERRORS !!!

        # Read spaxel positions on sky using RSS_fits_file[2]
        self.header2_data = RSS_fits_file[2].data

        # CAREFUL !! header 2 has the info of BAD fibres, if we are reading from our created RSS files we have to do it in a different way...

        # print RSS_fits_file[2].data

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
            offset_RA_arcsec = np.array(
                [RSS_fits_file[2].data[i][5] for i in good_spaxels]
            )
            offset_DEC_arcsec = np.array(
                [RSS_fits_file[2].data[i][6] for i in good_spaxels]
            )

            self.ID = np.array(
                [RSS_fits_file[2].data[i][0] for i in good_spaxels]
            )  # These are the good fibres
            variance = RSS_fits_file[1].data[good_spaxels]  # CHECK FOR ERRORS

        self.ZDSTART = RSS_fits_file[0].header["ZDSTART"]
        self.ZDEND = RSS_fits_file[0].header["ZDEND"]

        # KOALA-specific stuff
        self.PA = RSS_fits_file[0].header["TEL_PA"]
        self.grating = RSS_fits_file[0].header["GRATID"]
        # Check RED / BLUE arm for AAOmega
        if RSS_fits_file[0].header["SPECTID"] == "RD":
            AAOmega_Arm = "RED"
        if RSS_fits_file[0].header["SPECTID"] == "BL":
            AAOmega_Arm = "BLUE"

        # For WCS
        self.CRVAL1_CDELT1_CRPIX1 = []
        self.CRVAL1_CDELT1_CRPIX1.append(RSS_fits_file[0].header["CRVAL1"])
        self.CRVAL1_CDELT1_CRPIX1.append(RSS_fits_file[0].header["CDELT1"])
        self.CRVAL1_CDELT1_CRPIX1.append(RSS_fits_file[0].header["CRPIX1"])

        # SET RSS
        # FROM HERE IT WAS self.set_data before   ------------------------------------------

        self.wavelength = wavelength
        self.n_wave = len(wavelength)

        # Check that dimensions match KOALA numbers
        if self.n_wave != 2048 and len(all_spaxels) != 1000:
            print("\n *** WARNING *** : These numbers are NOT the standard ones for KOALA")

        print("\n> Setting the data for this file:")

        if variance.shape != intensity.shape:
            print("\n* ERROR: * the intensity and variance matrices are", intensity.shape, "and", variance.shape, "respectively\n")
            raise ValueError
        n_dim = len(intensity.shape)
        if n_dim == 2:
            self.intensity = intensity
            self.variance = variance
        elif n_dim == 1:
            self.intensity = intensity.reshape((1, self.n_wave))
            self.variance = variance.reshape((1, self.n_wave))
        else:
            print("\n* ERROR: * the intensity matrix supplied has", n_dim, "dimensions\n")
            raise ValueError

        self.n_spectra = self.intensity.shape[0]
        self.n_wave = len(self.wavelength)
        print("  Found {} spectra with {} wavelengths".format(
            self.n_spectra, self.n_wave
        ), "between {:.2f} and {:.2f} Angstrom".format(
            self.wavelength[0], self.wavelength[-1]
        ))
        if self.intensity.shape[1] != self.n_wave:
            print("\n* ERROR: * spectra have", self.intensity.shape[
                1
            ], "wavelengths rather than", self.n_wave)
            raise ValueError
        if (
            len(offset_RA_arcsec) != self.n_spectra
            or len(offset_DEC_arcsec) != self.n_spectra
        ):
            print("\n* ERROR: * offsets (RA, DEC) = ({},{})".format(
                len(self.offset_RA_arcsec), len(self.offset_DEC_arcsec)
            ), "rather than", self.n_spectra)
            raise ValueError
        else:
            self.offset_RA_arcsec = offset_RA_arcsec
            self.offset_DEC_arcsec = offset_DEC_arcsec

        # Check if NARROW (spaxel_size = 0.7 arcsec)
        # or WIDE (spaxel_size=1.25) field of view
        # (if offset_max - offset_min > 31 arcsec in both directions)
        if (
            np.max(offset_RA_arcsec) - np.min(offset_RA_arcsec) > 31
            or np.max(offset_DEC_arcsec) - np.min(offset_DEC_arcsec) > 31
        ):
            self.spaxel_size = 1.25
            field = "WIDE"
        else:
            self.spaxel_size = 0.7
            field = "NARROW"

        # Get min and max for rss
        self.RA_min, self.RA_max, self.DEC_min, self.DEC_max = coord_range([self])
        self.DEC_segment = (
            self.DEC_max - self.DEC_min
        ) * 3600.0  # +1.25 for converting to total field of view
        self.RA_segment = (self.RA_max - self.RA_min) * 3600.0  # +1.25

        # UPDATE THIS TO BE VALID TO ALL GRATINGS!
        # ALSO CONSIDER WAVELENGTH RANGE FOR SKYFLATS AND OBJECTS

        if valid_wave_min == 0 and valid_wave_max == 0:
            self.valid_wave_min = np.min(self.wavelength)
            self.valid_wave_max = np.max(self.wavelength)
        #            if self.grating == "1000R":
        #                self.valid_wave_min = 6600.    # CHECK ALL OF THIS...
        #                self.valid_wave_max = 6800.
        #                print "  For 1000R, we use the [6200, 7400] range."
        #            if self.grating == "1500V":
        #                self.valid_wave_min = np.min(self.wavelength)
        #                self.valid_wave_max = np.max(self.wavelength)
        #                print "  For 1500V, we use all the range."
        #            if self.grating == "580V":
        #                self.valid_wave_min = 3650.
        #                self.valid_wave_max = 5700.
        #                print "  For 580V, we use the [3650, 5700] range."
        #            if self.grating == "1500V":
        #                self.valid_wave_min = 4620.     #4550
        #                self.valid_wave_max = 5350.     #5350
        #                print "  For 1500V, we use the [4550, 5350] range."
        else:
            self.valid_wave_min = valid_wave_min
            self.valid_wave_max = valid_wave_max
            print("  As specified, we use the [", self.valid_wave_min, " , ", self.valid_wave_max, "] range.")

        # Plot RSS_image
        if plot:
            self.RSS_image(image=self.intensity, cmap="binary_r")

        # Deep copy of intensity into intensity_corrected
        self.intensity_corrected = copy.deepcopy(self.intensity)

        # Divide by flatfield if needed
        if flat != "":
            print("\n> Dividing the data by the flatfield provided...")
            self.intensity_corrected = (
                old_div(self.intensity_corrected, flat.intensity_corrected)
            )

        # Check if apply relative throughput & apply it if requested
        if apply_throughput:
            if plot_skyflat:
                plt.figure(figsize=(10, 4))
                for i in range(self.n_spectra):
                    plt.plot(self.wavelength, self.intensity[i, ])
                plt.ylim(0, 200 * np.nanmedian(self.intensity))
                plt.minorticks_on()
                plt.xlabel("Wavelength [$\AA$]")
                plt.title("Spectra WITHOUT CONSIDERING throughput correction")
                plt.show()
                plt.close()

            print("\n> Applying relative throughput correction using median skyflat values per fibre...")
            self.relative_throughput = skyflat.relative_throughput
            self.response_sky_spectrum = skyflat.response_sky_spectrum
            for i in range(self.n_spectra):
                self.intensity_corrected[i, :] = (
                    old_div(self.intensity_corrected[i, :], self.relative_throughput[i])
                )

            if nskyflat:
                print("\n  IMPORTANT: We are dividing intensity data by the sky.response_sky_spectrum !!! ")
                print("  This is kind of a flat, the changes are between ", np.nanmin(
                    skyflat.response_sky_spectrum
                ), "and ", np.nanmax(skyflat.response_sky_spectrum))
                print(" ")
                self.intensity_corrected = (
                    old_div(self.intensity_corrected, self.response_sky_spectrum)
                )

            if plot_skyflat:
                plt.figure(figsize=(10, 4))
                for i in range(self.n_spectra):
                    plt.plot(self.wavelength, self.intensity_corrected[i, ])
                plt.ylim(0, 200 * np.nanmedian(self.intensity_corrected))
                plt.minorticks_on()
                plt.xlabel("Wavelength [$\AA$]")
                plt.title(
                    "Spectra CONSIDERING throughput correction (median value per fibre)"
                )
                plt.show()
                plt.close()

            print("  Intensities corrected for relative throughput stored in self.intensity_corrected !")
            text_for_integrated_fibre = "after throughput correction..."
            title_for_integrated_fibre = " - Throughput corrected"
        else:
            if rss_clean == False:
                print("\n> Intensities NOT corrected for relative throughput")
            self.relative_throughput = np.ones(self.n_spectra)
            text_for_integrated_fibre = "..."
            title_for_integrated_fibre = ""

        # Compute integrated map after throughput correction & plot if requested
        self.compute_integrated_fibre(
            plot=plot,
            title=title_for_integrated_fibre,
            text=text_for_integrated_fibre,
            warnings=warnings,
            correct_negative_sky=correct_negative_sky,
            valid_wave_min=valid_wave_min,
            valid_wave_max=valid_wave_max,
        )
        plot_integrated_fibre_again = 0  # Check if we need to plot it again

        # Compare corrected vs uncorrected spectrum
        # self.plot_corrected_vs_uncorrected_spectrum(high_fibres=high_fibres, fig_size=fig_size)

        # Cleaning high cosmics and defects

        if sky_method == "1D" or sky_method == "2D":
            # If not it will not work when applying scale for sky substraction...
            remove_5578 = False

        if correct_ccd_defects:
            if plot:
                plot_integrated_fibre_again = plot_integrated_fibre_again + 1
            self.correct_high_cosmics_and_defects(
                correct_high_cosmics=correct_high_cosmics,
                step=step_ccd,
                remove_5578=remove_5578,
                clip_high=clip_high,
                plot_suspicious_fibres=plot_suspicious_fibres,
                warnings=warnings,
                verbose=verbose,
                plot=plot,
            )

            # Compare corrected vs uncorrected spectrum
            if plot:
                self.plot_corrected_vs_uncorrected_spectrum(
                    high_fibres=high_fibres, fig_size=fig_size
                )

        # Fixing small wavelengths
        if sol[0] != 0:
            self.fix_2dfdr_wavelengths(sol=sol)
        else:
            if fix_wavelengths:
                self.fix_2dfdr_wavelengths()
        #            else:
        #                print "\n> We don't fix 2dfdr wavelengths on this rss."

        # SKY SUBSTRACTION      sky_method
        #
        # Several options here: (1) "1D"   : Consider a single sky spectrum, scale it and substract it
        #                       (2) "2D"   : Consider a 2D sky. i.e., a sky image, scale it and substract it fibre by fibre
        #                       (3) "self" : Obtain the sky spectrum using the n_sky lowest fibres in the RSS file (DEFAULT)
        #                       (4) "none" : None sky substraction is performed
        #                       (5) "1Dfit": Using an external 1D sky spectrum, fits sky lines in both sky spectrum AND all the fibres

        if sky_method != "none" and is_sky == False:
            plot_integrated_fibre_again = plot_integrated_fibre_again + 1

            # (5)
            if sky_method == "1Dfit":
                print("\n> Fitting sky lines in both a provided sky spectrum AND all the fibres")
                print("  This process takes ~20 minutes for 385R!\n")

                if scale_sky_1D != 0:
                    print("  Sky spectrum scaled by ", scale_sky_1D)
                sky = np.array(sky_spectrum) * scale_sky_1D
                print("  Sky spectrum provided =", sky)
                self.sky_emission = sky
                self.fit_and_substract_sky_spectrum(
                    sky,
                    brightest_line_wavelength=brightest_line_wavelength,
                    brightest_line=brightest_line,
                    maxima_sigma=3.0,
                    ymin=-50,
                    ymax=1000,
                    wmin=0,
                    wmax=0,
                    auto_scale_sky=auto_scale_sky,
                    warnings=False,
                    verbose=False,
                    plot=False,
                    fig_size=12,
                    fibre=fibre,
                )

            # (1) If a single sky_spectrum is provided:
            if sky_method == "1D":
                if sky_spectrum[0] != 0:
                    print("\n> Sustracting the sky using the sky spectrum provided, checking the scale OBJ/SKY...")
                    if scale_sky_1D == 0:
                        self.sky_emission = scale_sky_spectrum(
                            self.wavelength,
                            sky_spectrum,
                            self.intensity_corrected,
                            cut_sky=cut_sky,
                            fmax=fmax,
                            fmin=fmin,
                            fibre_list=fibre_list,
                        )
                    else:
                        self.sky_emission = sky_spectrum * scale_sky_1D
                        print("  As requested, we scale the given 1D spectrum by", scale_sky_1D)

                    if individual_sky_substraction:
                        print("\n  As requested, performing individual sky substraction in each fibre...")
                    else:
                        print("\n  Substracting sky to all fibres using scaled sky spectrum provided...")

                    # For blue spectra, remove 5578 in the sky spectrum...
                    if self.valid_wave_min < 5578:
                        resultado = fluxes(
                            self.wavelength,
                            self.sky_emission,
                            5578,
                            plot=False,
                            verbose=False,
                        )  # fmin=-5.0E-17, fmax=2.0E-16,
                        # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                        self.sky_emission = resultado[11]

                    for i in range(self.n_spectra):
                        # Clean 5578 if needed in RSS data
                        if self.valid_wave_min < 5578:
                            resultado = fluxes(
                                self.wavelength,
                                self.intensity_corrected[i],
                                5578,
                                plot=False,
                                verbose=False,
                            )  # fmin=-5.0E-17, fmax=2.0E-16,
                            # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                            self.intensity_corrected[i] = resultado[11]

                        if individual_sky_substraction:
                            # Do this INDIVIDUALLY for each fibre
                            if i == 100:
                                print("  Substracting sky in fibre 100...")
                            if i == 200:
                                print("  Substracting sky in fibre 200...")
                            if i == 300:
                                print("  Substracting sky in fibre 300...")
                            if i == 400:
                                print("  Substracting sky in fibre 400...")
                            if i == 500:
                                print("  Substracting sky in fibre 500...")
                            if i == 600:
                                print("  Substracting sky in fibre 600...")
                            if i == 700:
                                print("  Substracting sky in fibre 700...")
                            if i == 800:
                                print("  Substracting sky in fibre 800...")
                            if i == 900:
                                print("  Substracting sky in fibre 900...")

                            sky_emission = scale_sky_spectrum(
                                self.wavelength,
                                sky_spectrum,
                                self.intensity_corrected,
                                cut_sky=cut_sky,
                                fmax=fmax,
                                fmin=fmin,
                                fibre_list=[i],
                                verbose=False,
                                plot=False,
                                warnings=False,
                            )

                            self.intensity_corrected[i, :] = (
                                self.intensity_corrected[i, :] - sky_emission
                            )  # sky_spectrum  * self.exptime/sky_exptime
                        else:
                            self.intensity_corrected[i, :] = (
                                self.intensity_corrected[i, :] - self.sky_emission
                            )  # sky_spectrum  * self.exptime/sky_exptime

                    if plot:
                        plt.figure(figsize=(fig_size, fig_size / 2.5))
                        plt.plot(self.wavelength, sky_spectrum)
                        plt.minorticks_on()
                        plt.axvline(x=self.valid_wave_min, color="k", linestyle="--")
                        plt.axvline(x=self.valid_wave_max, color="k", linestyle="--")
                        plt.xlim(self.wavelength[0] - 10, self.wavelength[-1] + 10)
                        plt.title("Sky spectrum provided (Scaled)")
                        plt.xlabel("Wavelength [$\AA$]")
                        plt.show()
                        plt.close()
                    print("  Intensities corrected for sky emission and stored in self.intensity_corrected !")
                    self.sky_emission = sky_spectrum
                else:
                    print("\n> Sustracting the sky using the sky spectrum requested but any sky spectrum provided !")
                    sky_method = "self"
                    n_sky = 50

            # (2) If a 2D sky, sky_rss, is provided
            if sky_method == "2D":  # if np.nanmedian(sky_rss.intensity_corrected) != 0:
                if scale_sky_rss != 0:
                    print("\n> Using sky image provided to substract sky, considering a scale of", scale_sky_rss, "...")
                    self.sky_emission = scale_sky_rss * sky_rss.intensity_corrected
                    self.intensity_corrected = (
                        self.intensity_corrected - self.sky_emission
                    )
                else:
                    print("\n> Using sky image provided to substract sky, computing the scale using sky lines")
                    # check scale fibre by fibre
                    self.sky_emission = copy.deepcopy(sky_rss.intensity_corrected)
                    scale_per_fibre = np.ones((self.n_spectra))
                    scale_per_fibre_2 = np.ones((self.n_spectra))
                    lowlow = 15
                    lowhigh = 5
                    highlow = 5
                    highhigh = 15
                    if self.grating == "580V":
                        print("  For 580V we use bright skyline at 5578 AA ...")
                        sky_line = 5578
                        sky_line_2 = 0
                    if self.grating == "1000R":
                        # print "  For 1000R we use skylines at 6300.5 and 6949.0 AA ..."   ### TWO LINES GIVE WORSE RESULTS THAN USING ONLY 1...
                        print("  For 1000R we use skyline at 6949.0 AA ...")
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
                    if sky_line_2 != 0:
                        print("  ... first checking", sky_line, "...")
                    for fibre_sky in range(self.n_spectra):
                        skyline_spec = fluxes(
                            self.wavelength,
                            self.intensity_corrected[fibre_sky],
                            sky_line,
                            plot=False,
                            verbose=False,
                            lowlow=lowlow,
                            lowhigh=lowhigh,
                            highlow=highlow,
                            highhigh=highhigh,
                        )  # fmin=-5.0E-17, fmax=2.0E-16,
                        # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                        self.intensity_corrected[fibre_sky] = skyline_spec[11]

                        skyline_sky = fluxes(
                            self.wavelength,
                            self.sky_emission[fibre_sky],
                            sky_line,
                            plot=False,
                            verbose=False,
                            lowlow=lowlow,
                            lowhigh=lowhigh,
                            highlow=highlow,
                            highhigh=highhigh,
                        )  # fmin=-5.0E-17, fmax=2.0E-16,

                        scale_per_fibre[fibre_sky] = old_div(skyline_spec[3], skyline_sky[3])
                        self.sky_emission[fibre_sky] = skyline_sky[11]

                    if sky_line_2 != 0:
                        print("  ... now checking", sky_line_2, "...")
                        for fibre_sky in range(self.n_spectra):
                            skyline_spec = fluxes(
                                self.wavelength,
                                self.intensity_corrected[fibre_sky],
                                sky_line_2,
                                plot=False,
                                verbose=False,
                                lowlow=lowlow_2,
                                lowhigh=lowhigh_2,
                                highlow=highlow_2,
                                highhigh=highhigh_2,
                            )  # fmin=-5.0E-17, fmax=2.0E-16,
                            # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                            self.intensity_corrected[fibre_sky] = skyline_spec[11]

                            skyline_sky = fluxes(
                                self.wavelength,
                                self.sky_emission[fibre_sky],
                                sky_line_2,
                                plot=False,
                                verbose=False,
                                lowlow=lowlow_2,
                                lowhigh=lowhigh_2,
                                highlow=highlow_2,
                                highhigh=highhigh_2,
                            )  # fmin=-5.0E-17, fmax=2.0E-16,

                            scale_per_fibre_2[fibre_sky] = (
                                old_div(skyline_spec[3], skyline_sky[3])
                            )
                            self.sky_emission[fibre_sky] = skyline_sky[11]

                    # Median value of scale_per_fibre, and apply that value to all fibres
                    if sky_line_2 == 0:
                        scale_sky_rss = np.nanmedian(scale_per_fibre)
                        self.sky_emission = self.sky_emission * scale_sky_rss
                    else:
                        scale_sky_rss = np.nanmedian(
                            old_div((scale_per_fibre + scale_per_fibre_2), 2)
                        )
                        # Make linear fit
                        scale_sky_rss_1 = np.nanmedian(scale_per_fibre)
                        scale_sky_rss_2 = np.nanmedian(scale_per_fibre_2)
                        print("  Median scale for line 1 :", scale_sky_rss_1, "range [", np.nanmin(
                            scale_per_fibre
                        ), ",", np.nanmax(
                            scale_per_fibre
                        ), "]")
                        print("  Median scale for line 2 :", scale_sky_rss_2, "range [", np.nanmin(
                            scale_per_fibre_2
                        ), ",", np.nanmax(
                            scale_per_fibre_2
                        ), "]")

                        b = old_div((scale_sky_rss_1 - scale_sky_rss_2), (
                            sky_line - sky_line_2
                        ))
                        a = scale_sky_rss_1 - b * sky_line
                        # ,a+b*sky_line,a+b*sky_line_2
                        print("  Appling linear fit with a =", a, "b =", b, "to all fibres in sky image...")

                        for i in range(self.n_wave):
                            self.sky_emission[:, i] = self.sky_emission[:, i] * (
                                a + b * self.wavelength[i]
                            )

                    if plot:
                        plt.figure(figsize=(fig_size, fig_size / 2.5))
                        label1 = "$\lambda$" + np.str(sky_line)
                        plt.plot(scale_per_fibre, alpha=0.5, label=label1)
                        plt.minorticks_on()
                        plt.ylim(np.nanmin(scale_per_fibre), np.nanmax(scale_per_fibre))
                        plt.axhline(y=scale_sky_rss, color="k", linestyle="--")
                        if sky_line_2 == 0:
                            text = (
                                "Scale OBJECT / SKY using sky line $\lambda$"
                                + np.str(sky_line)
                            )
                            print("  Scale per fibre in the range [", np.nanmin(
                                scale_per_fibre
                            ), ",", np.nanmax(
                                scale_per_fibre
                            ), "], median value is", scale_sky_rss)
                            print("  Using median value to scale sky emission provided...")
                        if sky_line_2 != 0:
                            text = (
                                "Scale OBJECT / SKY using sky lines $\lambda$"
                                + np.str(sky_line)
                                + " and $\lambda$"
                                + np.str(sky_line_2)
                            )
                            label2 = "$\lambda$" + np.str(sky_line_2)
                            plt.plot(scale_per_fibre_2, alpha=0.5, label=label2)
                            plt.axhline(y=scale_sky_rss_1, color="k", linestyle=":")
                            plt.axhline(y=scale_sky_rss_2, color="k", linestyle=":")
                            plt.legend(frameon=False, loc=1, ncol=2)
                        plt.title(text)
                        plt.xlabel("Fibre")
                        plt.show()
                        plt.close()

                    self.intensity_corrected = (
                        self.intensity_corrected - self.sky_emission
                    )

            # (3) No sky spectrum or image is provided, obtain the sky using the n_sky lowest fibres
            if sky_method == "self":
                print("\n  Using", n_sky, "lowest intensity fibres to create a sky...")
                self.find_sky_emission(
                    n_sky=n_sky,
                    plot=plot,
                    sky_fibres=sky_fibres,
                    sky_wave_min=sky_wave_min,
                    sky_wave_max=sky_wave_max,
                )

        #        print "\n  AFTER SKY SUBSTRACTION:"
        #        self.compute_integrated_fibre(plot=False, warnings=warnings)  #title =" - Throughput corrected", text="after throughput correction..."
        #        count_negative = 0
        #        for i in range(self.n_spectra):
        #            if self.integrated_fibre[i] < 0.11 :
        #                #print "  Fibre ",i," has an integrated flux of ", self.integrated_fibre[i]
        #                count_negative=count_negative+1
        # print self.integrated_fibre
        #        print "  Number of fibres with NEGATIVE integrated value AFTER SKY SUBSTRACTION = ", count_negative

        # If this RSS is an offset sky, perform a median filter to increase S/N
        if is_sky:
            print("\n> This RSS file is defined as SKY... applying median filter with window", win_sky, "...")
            medfilt_sky = median_filter(
                self.intensity_corrected, self.n_spectra, self.n_wave, win_sky=win_sky
            )
            self.intensity_corrected = copy.deepcopy(medfilt_sky)
            print("  Median filter applied, results stored in self.intensity_corrected !")

        # Get airmass and correct for extinction AFTER SKY SUBTRACTION
        ZD = old_div((self.ZDSTART + self.ZDEND), 2)
        self.airmass = old_div(1, np.cos(np.radians(ZD)))
        self.extinction_correction = np.ones(self.n_wave)
        if do_extinction:
            self.do_extinction_curve("ssoextinct.dat", plot=plot)

        # Check if telluric correction is needed & apply
        if telluric_correction[0] != 0:
            plot_integrated_fibre_again = plot_integrated_fibre_again + 1
            print("\n> Applying telluric correction...")
            if plot:
                plt.figure(figsize=(fig_size, fig_size / 2.5))
                plt.plot(self.wavelength, telluric_correction)
                plt.minorticks_on()
                plt.axvline(x=self.valid_wave_min, color="k", linestyle="--")
                plt.axvline(x=self.valid_wave_max, color="k", linestyle="--")
                plt.xlim(self.wavelength[0] - 10, self.wavelength[-1] + 10)
                plt.ylim(0.9, 2)
                plt.title("Telluric correction")
                plt.xlabel("Wavelength [$\AA$]")
                plt.show()
                plt.close()

            if plot:
                integrated_intensity_sorted = np.argsort(self.integrated_fibre)
                region = [
                    integrated_intensity_sorted[-1],
                    integrated_intensity_sorted[0],
                ]
                print("  Example of telluric correction using fibres", region[
                    0
                ], " and ", region[1], ":")
                plt.figure(figsize=(fig_size, fig_size / 2.5))
                plt.plot(
                    self.wavelength,
                    self.intensity_corrected[region[0]],
                    color="r",
                    alpha=0.3,
                )
                plt.plot(
                    self.wavelength,
                    self.intensity_corrected[region[1]],
                    color="r",
                    alpha=0.3,
                )

            for i in range(self.n_spectra):
                self.intensity_corrected[i, :] = (
                    self.intensity_corrected[i, :] * telluric_correction
                )

            if plot:
                plt.plot(
                    self.wavelength,
                    self.intensity_corrected[region[0]],
                    color="b",
                    alpha=0.5,
                )
                plt.plot(
                    self.wavelength,
                    self.intensity_corrected[region[1]],
                    color="g",
                    alpha=0.5,
                )
                plt.minorticks_on()
                plt.axvline(x=self.valid_wave_min, color="k", linestyle="--")
                plt.axvline(x=self.valid_wave_max, color="k", linestyle="--")
                plt.xlim(self.wavelength[0] - 10, self.wavelength[-1] + 10)
                plt.ylim(
                    np.nanmin(self.intensity_corrected[region[1]]),
                    np.nanmax(self.intensity_corrected[region[0]]),
                )  # CHECK THIS AUTOMATICALLY
                plt.xlabel("Wavelength [$\AA$]")
                plt.show()
                plt.close()

        # Check if identify emission lines is requested & do
        if id_el:
            if brightest_line_wavelength == 0:
                self.el = self.identify_el(
                    high_fibres=high_fibres,
                    brightest_line=brightest_line,
                    cut=cut,
                    verbose=True,
                    plot=plot_id_el,
                    fibre=0,
                    broad=broad,
                )
                print("\n  Emission lines identified saved in self.el !!")
            else:
                brightest_line_rest_wave = 6562.82
                print("\n  As given, line ", brightest_line, " at rest wavelength = ", brightest_line_rest_wave, " is at ", brightest_line_wavelength)
                self.el = [
                    [brightest_line],
                    [brightest_line_rest_wave],
                    [brightest_line_wavelength],
                    [7.2],
                ]
                # PUTAAA  sel.el=[peaks_name,peaks_rest, p_peaks_l, p_peaks_fwhm]
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
                        redshift = old_div((self.el[2][i] - self.el[1][i]), self.el[1][i])
                print("  Brightest emission line", brightest_line, "found at ", obs_wave, ", redshift = ", redshift)

                el_identified = [[], [], [], []]
                n_identified = 0
                for line in id_list:
                    id_check = 0
                    for i in range(len(self.el[1])):
                        if line == self.el[1][i]:
                            if verbose:
                                print("  Emission line ", self.el[0][i], self.el[1][
                                    i
                                ], "has been identified")
                            n_identified = n_identified + 1
                            id_check = 1
                            el_identified[0].append(self.el[0][i])  # Name
                            el_identified[1].append(self.el[1][i])  # Central wavelength
                            el_identified[2].append(
                                self.el[2][i]
                            )  # Observed wavelength
                            el_identified[3].append(self.el[3][i])  # "FWHM"
                    if id_check == 0:
                        for i in range(len(el_center)):
                            if line == el_center[i]:
                                el_identified[0].append(el_name[i])
                                print("  Emission line", el_name[
                                    i
                                ], line, "has NOT been identified, adding...")
                        el_identified[1].append(line)
                        el_identified[2].append(line * (redshift + 1))
                        el_identified[3].append(4 * broad)

                self.el = el_identified
                print("  Number of emission lines identified = ", n_identified, "of a total of", len(
                    id_list
                ), "provided. self.el updated accordingly")
            else:
                print("\n> List of emission lines provided but no identification was requested")

        # Clean sky residuals if requested
        if clean_sky_residuals:
            plot_integrated_fibre_again = plot_integrated_fibre_again + 1
            self.clean_sky_residuals(
                extra_w=extra_w,
                step=step_csr,
                dclip=dclip,
                verbose=verbose,
                fibre=fibre,
                wave_min=valid_wave_min,
                wave_max=valid_wave_max,
            )

        #  set_data was till here...  -------------------------------------------------------------------

        if fibre != 0:
            plot_integrated_fibre_again = 0

        # Plot corrected values
        if plot == True and rss_clean == False:  # plot_integrated_fibre_again > 0 :
            self.compute_integrated_fibre(
                plot=plot,
                title=" - Intensities Corrected",
                warnings=warnings,
                text="after all corrections have been applied...",
                valid_wave_min=valid_wave_min,
                valid_wave_max=valid_wave_max,
                correct_negative_sky=correct_negative_sky,
            )

            integrated_intensity_sorted = np.argsort(self.integrated_fibre)
            region = []
            for fibre_ in range(high_fibres):
                region.append(integrated_intensity_sorted[-1 - fibre_])
            print("\n> Checking results using", high_fibres, "fibres with the highest integrated intensity")
            print("  which are :", region)

            plt.figure(figsize=(fig_size, fig_size / 2.5))
            I = np.nansum(self.intensity[region], axis=0)
            plt.plot(self.wavelength, I, "r-", label="Uncorrected", alpha=0.3)
            Ic = np.nansum(self.intensity_corrected[region], axis=0)
            plt.axhline(y=0, color="k", linestyle=":")
            plt.plot(self.wavelength, Ic, "g-", label="Corrected", alpha=0.4)
            plt.ylabel("Flux")
            plt.xlabel("Wavelength [$\AA$]")
            plt.minorticks_on()
            plt.xlim(self.wavelength[0] - 10, self.wavelength[-1] + 10)
            plt.axvline(x=self.valid_wave_min, color="k", linestyle="--")
            plt.axvline(x=self.valid_wave_max, color="k", linestyle="--")
            yy1 = np.nanpercentile(Ic, 0)
            yy2 = np.nanpercentile(Ic, 99)
            rango = yy2 - yy1
            plt.ylim(yy1 - rango * 0.05, yy2)
            plt.title(
                self.object
                + " - Combined spectrum - "
                + str(high_fibres)
                + " fibres with highest intensity"
            )
            plt.legend(frameon=False, loc=4, ncol=2)
            plt.show()
            plt.close()

            region = []
            for fibre_ in range(high_fibres):
                region.append(integrated_intensity_sorted[fibre_])
            print("\n> Checking results using", high_fibres, "fibres with the lowest integrated intensity")
            print("  which are :", region)

            plt.figure(figsize=(fig_size, fig_size / 2.5))
            I = np.nansum(self.intensity[region], axis=0)
            plt.plot(self.wavelength, I, "r-", label="Uncorrected", alpha=0.3)
            Ic = np.nansum(self.intensity_corrected[region], axis=0)
            I_ymin = np.nanmin([np.nanmin(I), np.nanmin(Ic)])
            I_ymax = np.nanmax([np.nanmax(I), np.nanmax(Ic)])
            I_med = np.nanmedian(Ic)
            I_rango = I_ymax - I_ymin
            plt.axhline(y=0, color="k", linestyle=":")
            plt.plot(self.wavelength, Ic, "g-", label="Corrected", alpha=0.4)
            plt.ylabel("Flux")
            plt.xlabel("Wavelength [$\AA$]")
            plt.minorticks_on()
            plt.xlim(self.wavelength[0] - 10, self.wavelength[-1] + 10)
            plt.axvline(x=self.valid_wave_min, color="k", linestyle="--")
            plt.axvline(x=self.valid_wave_max, color="k", linestyle="--")
            #            plt.ylim([I_ymin-I_rango/18,I_ymax-I_rango*0.65])
            plt.ylim([I_med - I_rango * 0.65, I_med + I_rango * 0.65])
            plt.title(
                self.object
                + " - Combined spectrum - "
                + str(high_fibres)
                + " fibres with lowest intensity"
            )
            plt.legend(frameon=False, loc=4, ncol=2)
            plt.show()
            plt.close()

        # Plot RSS_image
        if plot:
            self.RSS_image()

        if rss_clean:
            self.RSS_image()

        # Print summary and information from header
        print("\n> Summary of reading rss file", '"' + filename + '"', ":\n")
        print("  This is a KOALA '{}' file,".format(
            AAOmega_Arm
        ), "using grating '{}' in AAOmega".format(self.grating))
        print("  Object:", self.object)
        print("  Field of view:", field, "(spaxel size =", self.spaxel_size, "arcsec)")
        print("  Center position: (RA, DEC) = ({:.3f}, {:.3f}) degrees".format(
            self.RA_centre_deg, self.DEC_centre_deg
        ))
        print("  Field covered [arcsec] = {:.1f} x {:.1f}".format(
            self.RA_segment + self.spaxel_size, self.DEC_segment + self.spaxel_size
        ))
        print("  Position angle (PA) = {:.1f} degrees".format(self.PA))
        print(" ")

        if rss_clean:
            print("  This was a CLEAN RSS file, no correction was applied!")
            print("  Values stored in self.intensity_corrected are the same that those in self.intensity")
        else:
            if flat != "":
                print("  Intensities divided by the given flatfield")
            if apply_throughput:
                print("  Intensities corrected for throughput !")
            else:
                print("  Intensities NOT corrected for throughput")
            if correct_ccd_defects == True and correct_high_cosmics == True:
                print("  Intensities corrected for high cosmics and CCD defects !")
            if correct_ccd_defects == True and correct_high_cosmics == False:
                print("  Intensities corrected for CCD defects (but NOT for high cosmics) !")
            if correct_ccd_defects == False and correct_high_cosmics == False:
                print("  Intensities NOT corrected for high cosmics and CCD defects")

            if sol[0] != 0:
                print("  All fibres corrected for small wavelength shifts using wavelength solution provided!")
            else:
                if fix_wavelengths:
                    print("  Wavelengths corrected for small shifts using Gaussian fit to selected bright skylines in all fibres!")
                else:
                    print("  Wavelengths NOT corrected for small shifts")

            if is_sky:
                print("  This is a SKY IMAGE, median filter with window", win_sky, "applied !")
            else:
                if sky_method == "none":
                    print("  Intensities NOT corrected for sky emission")
                if sky_method == "self":
                    print("  Intensities corrected for sky emission using", n_sky, "spaxels with lowest values !")
                if sky_method == "1D":
                    print("  Intensities corrected for sky emission using (scaled) spectrum provided ! ")
                if sky_method == "1Dfit":
                    print("  Intensities corrected for sky emission fitting Gaussians to both 1D sky spectrum and each fibre ! ")
                if sky_method == "2D":
                    print("  Intensities corrected for sky emission using sky image provided scaled by", scale_sky_rss, "!")
            if telluric_correction[0] != 0:
                print("  Intensities corrected for telluric absorptions !")
            else:
                print("  Intensities NOT corrected for telluric absorptions")

            if do_extinction:
                print("  Intensities corrected for extinction !")
            else:
                print("  Intensities NOT corrected for extinction")

            if correct_negative_sky:
                print("  Intensities CORRECTED (if needed) for negative integrate flux values!")

            if id_el:
                print(" ", len(
                    self.el[0]
                ), "emission lines identified and stored in self.el !")
                print(" ", self.el[0])
            if clean_sky_residuals == True and fibre == 0:
                print("  Intensities cleaned for sky residuals !")
            if clean_sky_residuals == True and fibre != 0:
                print("  Only fibre ", fibre, " has been corrected for sky residuals")
            if clean_sky_residuals == False:
                print("  Intensities NOT corrected for sky residuals")

            print("  All applied corrections are stored in self.intensity_corrected !")

            if save_rss_to_fits_file != "":
                save_rss_fits(self, fits_file=save_rss_to_fits_file)

        print("\n> KOALA RSS file read !")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# INTERPOLATED CUBE CLASS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class Interpolated_cube(object):  # TASK_Interpolated_cube
    """
    Constructs a cube by accumulating RSS with given offsets.
    """

    # -----------------------------------------------------------------------------
    def __init__(
        self,
        RSS,
        pixel_size_arcsec,
        kernel_size_arcsec,
        centre_deg=[],
        size_arcsec=[],
        aligned_coor=False,
        plot=False,
        flux_calibration=[0],
        zeros=False,
        ADR=False,
        force_ADR=False,
        offsets_files="",
        offsets_files_position="",
        shape=[],
        rss_file="",
        warnings=False,
    ):  # Angel added aligned_coor 6 Sep, flux_calibration, zeros 27 Oct;
        # added ADR 28 Feb offsets_files, shape for defining shape of cube
        # warnings (when cubing) added 13 Jan 2019
        self.RSS = RSS
        self.n_wave = RSS.n_wave
        self.pixel_size_arcsec = pixel_size_arcsec
        self.kernel_size_arcsec = kernel_size_arcsec
        self.kernel_size_pixels = (
            old_div(kernel_size_arcsec, pixel_size_arcsec)
        )  # must be a float number!

        self.wavelength = RSS.wavelength
        self.description = RSS.description + " - CUBE"
        self.object = RSS.object
        self.PA = RSS.PA
        self.grating = RSS.grating
        self.CRVAL1_CDELT1_CRPIX1 = RSS.CRVAL1_CDELT1_CRPIX1
        self.total_exptime = RSS.exptime
        self.rss_list = RSS.rss_list
        self.RA_segment = RSS.RA_segment
        self.offsets_files = offsets_files  # Offsets between files when align cubes
        self.offsets_files_position = (
            offsets_files_position  # Position of this cube when aligning
        )

        self.valid_wave_min = RSS.valid_wave_min
        self.valid_wave_max = RSS.valid_wave_max

        self.seeing = 0.0
        self.flux_cal_step = 0.0
        self.flux_cal_min_wave = 0.0
        self.flux_cal_max_wave = 0.0

        if zeros:
            print("\n> Creating empty cube using information provided in rss file: ")
            print(" ", self.description)
        else:
            print("\n> Creating cube from file rss file: ", self.description)
        print("  Pixel size  = ", self.pixel_size_arcsec, " arcsec")
        print("  kernel size = ", self.kernel_size_arcsec, " arcsec")

        # centre_deg = [RA,DEC] if we need to give new RA, DEC centre
        if len(centre_deg) == 2:
            self.RA_centre_deg = centre_deg[0]
            self.DEC_centre_deg = centre_deg[1]
        else:
            self.RA_centre_deg = RSS.RA_centre_deg
            self.DEC_centre_deg = RSS.DEC_centre_deg

        if aligned_coor == True:
            self.xoffset_centre_arcsec = (
                self.RA_centre_deg - RSS.ALIGNED_RA_centre_deg
            ) * 3600.0
            self.yoffset_centre_arcsec = (
                self.DEC_centre_deg - RSS.ALIGNED_DEC_centre_deg
            ) * 3600.0

            print(self.RA_centre_deg)
            print(RSS.ALIGNED_RA_centre_deg)
            print((self.RA_centre_deg - RSS.ALIGNED_RA_centre_deg) * 3600.0)
            print("\n\n\n\n")

            if zeros == False:
                print("  Using ALIGNED coordenates for centering cube...")
        else:
            self.xoffset_centre_arcsec = (
                self.RA_centre_deg - RSS.RA_centre_deg
            ) * 3600.0
            self.yoffset_centre_arcsec = (
                self.DEC_centre_deg - RSS.DEC_centre_deg
            ) * 3600.0

        if len(size_arcsec) == 2:
            self.n_cols = np.int(old_div(size_arcsec[0], self.pixel_size_arcsec)) + 2 * np.int(
                old_div(self.kernel_size_arcsec, self.pixel_size_arcsec)
            )
            self.n_rows = np.int(old_div(size_arcsec[1], self.pixel_size_arcsec)) + 2 * np.int(
                old_div(self.kernel_size_arcsec, self.pixel_size_arcsec)
            )
        else:
            self.n_cols = (
                2
                * (
                    np.int(
                        old_div(np.nanmax(
                            np.abs(RSS.offset_RA_arcsec - self.xoffset_centre_arcsec)
                        ), self.pixel_size_arcsec)
                    )
                    + np.int(self.kernel_size_pixels)
                )
                + 3
            )  # -3    ### +1 added by Angel 25 Feb 2018 to put center in center
            self.n_rows = (
                2
                * (
                    np.int(
                        old_div(np.nanmax(
                            np.abs(RSS.offset_DEC_arcsec - self.yoffset_centre_arcsec)
                        ), self.pixel_size_arcsec)
                    )
                    + np.int(self.kernel_size_pixels)
                )
                + 3
            )  # -3   ### +1 added by Angel 25 Feb 2018 to put center in center

        if self.n_cols % 2 != 0:
            self.n_cols += 1  # Even numbers to have [0,0] in the centre
        if self.n_rows % 2 != 0:
            self.n_rows += 1

        # If we define a specific shape
        if len(shape) == 2:
            self.n_rows = shape[0]
            self.n_cols = shape[1]

        # Define zeros
        self._weighted_I = np.zeros((self.n_wave, self.n_rows, self.n_cols))
        self._weight = np.zeros_like(self._weighted_I)
        self.flux_calibration = np.zeros(self.n_wave)
        #        self.offset_from_center_x_arcsec = 0.
        #        self.offset_from_center_y_arcsec = 0.

        if zeros:
            self.data = np.zeros_like(self._weighted_I)
        else:
            print("\n  Smooth cube, (RA, DEC)_centre = ({}, {}) degree".format(
                self.RA_centre_deg, self.DEC_centre_deg
            ))
            print("  Size = {} columns (RA) x {} rows (DEC); {:.2f} x {:.2f} arcsec".format(
                self.n_cols,
                self.n_rows,
                (self.n_cols + 1) * pixel_size_arcsec,
                (self.n_rows + 1) * pixel_size_arcsec,
            ))
            sys.stdout.write("  Adding {} spectra...       ".format(RSS.n_spectra))
            sys.stdout.flush()
            output_every_few = np.sqrt(RSS.n_spectra) + 1
            next_output = -1
            for i in range(RSS.n_spectra):
                if i > next_output:
                    sys.stdout.write("\b" * 6)
                    sys.stdout.write("{:5.2f}%".format(i * 100.0 / RSS.n_spectra))
                    sys.stdout.flush()
                    next_output = i + output_every_few
                offset_rows = old_div((
                    RSS.offset_DEC_arcsec[i] - self.yoffset_centre_arcsec
                ), pixel_size_arcsec)
                offset_cols = old_div((
                    -RSS.offset_RA_arcsec[i] + self.xoffset_centre_arcsec
                ), pixel_size_arcsec)
                corrected_intensity = RSS.intensity_corrected[i]
                self.add_spectrum(
                    corrected_intensity, offset_rows, offset_cols, warnings=warnings
                )

            self.data = old_div(self._weighted_I, self._weight)
            self.trace_peak(plot=plot)

            # Check flux calibration
            if np.nanmedian(flux_calibration) == 0:
                fcal = False
            else:
                self.flux_calibration = flux_calibration
                fcal = True
                # This should be in 1 line of step of loop, I couldn't get it # Yago HELP !!
                for x in range(self.n_rows):
                    for y in range(self.n_cols):
                        self.data[:, x, y] = (
                            old_div(old_div(old_div(self.data[:, x, y], self.flux_calibration), 1e16), self.RSS.exptime)
                        )
            #                        plt.plot(self.wavelength,self.data[:,x,y]) #
            # ylabel="Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]"

            # Correct for Atmospheric Differential Refraction (ADR) if requested
            if ADR:
                self.ADR_correction(plot=plot, force_ADR=force_ADR)
            else:
                print("\n> Data NO corrected for Atmospheric Differential Refraction (ADR).")

            # Get integrated maps (all waves and valid range), locate peaks, plots
            self.get_integrated_map_and_plot(plot=plot, fcal=fcal)

            # For calibration stars, we get an integrated star flux and a seeing
            self.integrated_star_flux = np.zeros_like(self.wavelength)

            if fcal:
                print("\n> Absolute flux calibration included in this interpolated cube.")
            else:
                print("\n> This interpolated cube does not include an absolute flux calibration.")

        print("> Interpolated cube done!\n")

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def ADR_correction(self, plot=True, force_ADR=False):
        """
        Correct for Atmospheric Diferential Refraction (ADR)
        """
        self.data_ADR = copy.deepcopy(self.data)
        do_ADR = True
        # First we check if it is needed (unless forced)...

        if (
            self.ADR_x_max < old_div(self.pixel_size_arcsec, 2)
            and self.ADR_y_max < old_div(self.pixel_size_arcsec, 2)
        ):
            print("\n> Atmospheric Differential Refraction (ADR) correction is NOT needed.")
            print("  The computed max ADR values ({:.2f},{:.2f}) are smaller than half the pixel size of {:.2f} arcsec".format(
                self.ADR_x_max, self.ADR_y_max, self.pixel_size_arcsec
            ))
            do_ADR = False
            if force_ADR:
                print('  However we proceed to do the ADR correction as indicated: "force_ADR = True" ...')
                do_ADR = True
        if do_ADR:
            print("\n> Correcting for Atmospheric Differential Refraction (ADR)...")
            sys.stdout.flush()
            output_every_few = np.sqrt(self.n_wave) + 1
            next_output = -1
            for l in range(self.n_wave):
                if l > next_output:
                    sys.stdout.write("\b" * 36)
                    sys.stdout.write(
                        "  Moving plane {:5}/{:5}... {:5.2f}%".format(
                            l, self.n_wave, l * 100.0 / self.n_wave
                        )
                    )
                    sys.stdout.flush()
                    next_output = l + output_every_few
                tmp = copy.deepcopy(self.data_ADR[l, :, :])
                mask = copy.deepcopy(tmp) * 0.0
                mask[np.where(np.isnan(tmp))] = 1  # make mask where Nans are
                kernel = Gaussian2DKernel(5)
                tmp_nonan = interpolate_replace_nans(tmp, kernel)
                # need to see if there are still nans. This can happen in the padded parts of the grid
                # where the kernel is not large enough to cover the regions with NaNs.
                if np.isnan(np.sum(tmp_nonan)):
                    tmp_nonan = np.nan_to_num(tmp_nonan)
                tmp_shift = shift(
                    tmp_nonan,
                    [
                        old_div(-2 * self.ADR_y[l], self.pixel_size_arcsec),
                        old_div(-2 * self.ADR_x[l], self.pixel_size_arcsec),
                    ],
                    cval=np.nan,
                )
                mask_shift = shift(
                    mask,
                    [
                        old_div(-2 * self.ADR_y[l], self.pixel_size_arcsec),
                        old_div(-2 * self.ADR_x[l], self.pixel_size_arcsec),
                    ],
                    cval=np.nan,
                )
                tmp_shift[mask_shift > 0.5] = np.nan
                self.data_ADR[l, :, :] = copy.deepcopy(tmp_shift)
                # print(l,tmp.shape,2*self.ADR_y[l],2*self.ADR_x[l],np.sum(tmp_nonan),np.sum(tmp),np.sum(tmp_shift))
            #             for y in range(self.n_rows):
            #                 for x in range(self.n_cols):
            # #                            mal = 0
            #                     if np.int(np.round(x+2*self.ADR_x[l]/self.pixel_size_arcsec)) < self.n_cols :
            #                         if np.int(np.round(y+2*self.ADR_y[l]/self.pixel_size_arcsec)) < self.n_rows :
            #                            # print self.data.shape,x,"->",np.int(np.round(x+self.ADR_x[i]/self.pixel_size_arcsec)),"     ",y,"->",np.int(np.round(y+self.ADR_y[i]/self.pixel_size_arcsec))
            #                             self.data_ADR[l,y,x]=self.data[l, np.int(np.round(y+2*self.ADR_y[l]/self.pixel_size_arcsec )), np.int(np.round(x+2*self.ADR_x[l]/self.pixel_size_arcsec)) ]
            #                                else: mal = 1
            #                            else: mal = 1
            #                            if mal == 1:
            #                                if l == 0 : print self.data.shape,x,"->",np.int(np.round(x+self.ADR_x[i]/self.pixel_size_arcsec)),"     ",y,"->",np.int(np.round(y+self.ADR_y[i]/self.pixel_size_arcsec))," bad data !"
            # Check values tracing ADR data ...
            self.trace_peak(ADR=True, plot=plot)
            # SAVE DATA !!!!
            # In prep...

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def get_integrated_map_and_plot(
        self, min_wave=[0], max_wave=[0], plot=True, fcal=False
    ):  # CHECK
        """
        Integrated map and plot
        """
        # Integrated map between good wavelengths

        if min_wave == [0]:
            min_wave = self.valid_wave_min
        if max_wave == [0]:
            max_wave = self.valid_wave_max

        self.integrated_map_all = np.nansum(self.data, axis=0)
        self.integrated_map = np.nansum(
            self.data[
                np.searchsorted(self.wavelength, min_wave): np.searchsorted(
                    self.wavelength, max_wave
                )
            ],
            axis=0,
        )

        # Search for peak of emission in integrated map and compute offsets from centre
        self.max_y, self.max_x = np.unravel_index(
            self.integrated_map.argmax(), self.integrated_map.shape
        )
        self.spaxel_RA0 = old_div(self.n_cols, 2) + 1
        self.spaxel_DEC0 = old_div(self.n_rows, 2) + 1
        self.offset_from_center_x_arcsec_integrated = (
            self.max_x - self.spaxel_RA0 + 1
        ) * self.pixel_size_arcsec  # Offset from center using INTEGRATED map
        self.offset_from_center_y_arcsec_integrated = (
            self.max_y - self.spaxel_DEC0 + 1
        ) * self.pixel_size_arcsec  # Offset from center using INTEGRATED map

        if plot:
            self.plot_spectrum_integrated_cube(fcal=fcal)
            self.plot_spectrum_cube(self.max_y, self.max_x, fcal=fcal)

        print("\n> Created integrated map between {:5.2f} and {:5.2f}.".format(
            min_wave, max_wave
        ))
        print("  The peak of the emission in integrated image is in spaxel [", self.max_x, ",", self.max_y, "]")
        print("  The peak of the emission tracing all wavelengths is in spaxel [", np.round(
            self.x_peak_median, 2
        ), ",", np.round(
            self.y_peak_median, 2
        ), "]")

        self.offset_from_center_x_arcsec_tracing = (
            self.x_peak_median - self.spaxel_RA0 + 1
        ) * self.pixel_size_arcsec  # Offset from center using INTEGRATED map
        self.offset_from_center_y_arcsec_tracing = (
            self.y_peak_median - self.spaxel_DEC0 + 1
        ) * self.pixel_size_arcsec  # Offset from center using INTEGRATED map
        if plot:
            self.plot_map(
                norm=colors.Normalize(),
                spaxel=[self.max_x, self.max_y],
                spaxel2=[self.x_peak_median, self.y_peak_median],
                fcal=fcal,
            )

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def add_spectrum(self, intensity, offset_rows, offset_cols, warnings=False):
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
        kernel_centre_x = 0.5 * self.n_cols + offset_cols
        x_min = int(kernel_centre_x - self.kernel_size_pixels)
        x_max = int(kernel_centre_x + self.kernel_size_pixels) + 1
        n_points_x = x_max - x_min
        x = (
            old_div(np.linspace(x_min - kernel_centre_x, x_max - kernel_centre_x, n_points_x), self.kernel_size_pixels)
        )
        x[0] = -1.0
        x[-1] = 1.0
        weight_x = np.diff(old_div((3.0 * x - x ** 3 + 2.0), 4))

        kernel_centre_y = 0.5 * self.n_rows + offset_rows
        y_min = int(kernel_centre_y - self.kernel_size_pixels)
        y_max = int(kernel_centre_y + self.kernel_size_pixels) + 1
        n_points_y = y_max - y_min
        y = (
            old_div(np.linspace(y_min - kernel_centre_y, y_max - kernel_centre_y, n_points_y), self.kernel_size_pixels)
        )
        y[0] = -1.0
        y[-1] = 1.0
        weight_y = np.diff(old_div((3.0 * y - y ** 3 + 2.0), 4))

        if x_min < 0 or x_max >= self.n_cols or y_min < 0 or y_max >= self.n_rows:
            if warnings:
                print("**** WARNING **** : Spectra outside field of view:", x_min, kernel_centre_x, x_max)
                print("                                                 :", y_min, kernel_centre_y, y_max)
        else:
            bad_wavelengths = np.argwhere(np.isnan(intensity))
            intensity[bad_wavelengths] = 0.0
            ones = np.ones_like(intensity)
            ones[bad_wavelengths] = 0.0
            self._weighted_I[:, y_min: y_max - 1, x_min: x_max - 1] += (
                intensity[:, np.newaxis, np.newaxis]
                * weight_y[np.newaxis, :, np.newaxis]
                * weight_x[np.newaxis, np.newaxis, :]
            )
            self._weight[:, y_min: y_max - 1, x_min: x_max - 1] += (
                ones[:, np.newaxis, np.newaxis]
                * weight_y[np.newaxis, :, np.newaxis]
                * weight_x[np.newaxis, np.newaxis, :]
            )

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_spectrum_cube(
        self,
        x,
        y,
        lmin=0,
        lmax=0,
        fmin=1e-30,
        fmax=1e30,
        fcal=False,
        fig_size=10.0,
        fig_size_y=0.0,
        save_file="",
        title="",
        z=0.0,
    ):  # Angel added 8 Sep
        """
        Plot spectrum of a particular spaxel.

        Parameters
        ----------
        x, y:
            coordenates of spaxel to show spectrum.
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
            if fcal == False:
                spectrum = self.data[:, x, y]
                ylabel = "Flux [relative units]"
            else:
                spectrum = self.data[:, x, y] * 1e16  # /self.flux_calibration  / 1E16
                # ylabel="Flux [ 10$^{-16}$ * erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]"
                ylabel = "Flux [ 10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]"
        else:
            print("  Adding spaxel  1  = [", x[0], ",", y[0], "]")
            spectrum = self.data[:, x[0], y[0]]
            for i in range(len(x) - 1):
                spectrum = spectrum + self.data[:, x[i + 1], y[i + 1]]
                print("  Adding spaxel ", i + 2, " = [", x[i + 1], ",", y[i + 1], "]")
                ylabel = "Flux [relative units]"
            if fcal:
                spectrum = old_div(old_div(spectrum, self.flux_calibration), 1e16)
                ylabel = "Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]"

        # Set limits
        if fmin == 1e-30:
            fmin = np.nanmin(spectrum)
        if fmax == 1e30:
            fmax = np.nanmax(spectrum)
        if lmin == 0:
            lmin = self.wavelength[0]
        if lmax == 0:
            lmax = self.wavelength[-1]

        if fig_size_y == 0.0:
            fig_size_y = fig_size / 3.0
        plt.figure(figsize=(fig_size, fig_size_y))
        plt.plot(self.wavelength, spectrum)
        plt.minorticks_on()
        plt.ylim(fmin, fmax)
        plt.xlim(lmin, lmax)

        if title == "":
            title = "Spaxel ({} , {}) in {}".format(x, y, self.description)
        plt.title(title)
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel(ylabel)

        # Identify lines

        if z != 0:
            elines = [
                3727.00,
                3868.75,
                3967.46,
                3889.05,
                4026.0,
                4068.10,
                4101.2,
                4340.47,
                4363.21,
                4471.48,
                4658.10,
                4686.0,
                4711.37,
                4740.16,
                4861.33,
                4958.91,
                5006.84,
                5197.82,
                6300.30,
                6312.10,
                6363.78,
                6548.03,
                6562.82,
                6583.41,
                6678.15,
                6716.47,
                6730.85,
                7065.28,
                7135.78,
                7281.35,
                7320,
                7330,
            ]
            #            elines=[3727.00, 3868.75, 3967.46, 3889.05, 4026., 4068.10, 4101.2, 4340.47, 4363.21, 4471.48, 4658.10, 4861.33, 4958.91, 5006.84, 5197.82, 6300.30, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7320, 7330 ]

            for i in elines:
                plt.plot([i * (1 + z), i * (1 + z)], [fmin, fmax], "g:", alpha=0.95)
            alines = [3934.777, 3969.588, 4308, 5175]  # ,4305.61, 5176.7]   # POX 4
            #            alines=[3934.777,3969.588,4308,5170]    #,4305.61, 5176.7]
            for i in alines:
                plt.plot([i * (1 + z), i * (1 + z)], [fmin, fmax], "r:", alpha=0.95)

        if save_file == "":
            plt.show()
        else:
            plt.savefig(save_file)
        plt.close()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_spectrum_integrated_cube(
        self,
        lmin=0,
        lmax=0,
        fmin=1e-30,
        fmax=1e30,
        fcal=False,
        fig_size=10,
        save_file="",
    ):  # Angel added 8 Sep
        """
        Plot integrated spectrum

        Parameters
        ----------
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
        spectrum = np.nansum(np.nansum(self.data, axis=1), axis=1)
        if fcal == False:
            ylabel = "Flux [relative units]"
        else:
            spectrum = spectrum * 1e16
            # ylabel="Flux [ 10$^{-16}$ * erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]"
            ylabel = "Flux [  10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]"

        # Set limits
        if fmin == 1e-30:
            fmin = np.nanmin(spectrum)
        if fmax == 1e30:
            fmax = np.nanmax(spectrum)
        if lmin == 0:
            lmin = self.wavelength[0]
        if lmax == 0:
            lmax = self.wavelength[-1]

        plt.figure(figsize=(fig_size, fig_size / 2.5))
        plt.plot(self.wavelength, spectrum)
        plt.minorticks_on()
        plt.ylim(fmin, fmax)
        plt.xlim(lmin, lmax)

        title = "Integrated spectrum in {}".format(self.description)
        plt.title(title)
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel(ylabel)

        if save_file == "":
            plt.show()
        else:
            plt.savefig(save_file)
        plt.close()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_weight(
        self, norm=colors.Normalize(), cmap="gist_gray", fig_size=10, save_file=""
    ):
        """
        Plot weitgh map."

        Example
        ----------
        >>> cube1s.plot_weight()
        """
        interpolated_map = np.mean(self._weight, axis=0)
        self.plot_map(
            interpolated_map,
            norm=norm,
            fig_size=fig_size,
            cmap=cmap,
            save_file=save_file,
            description=self.description + " - Weight map",
        )

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_wavelength(
        self,
        wavelength,
        w2=0.0,
        cmap=fuego_color_map,
        fig_size=10,
        norm=colors.PowerNorm(gamma=1.0 / 4.0),
        save_file="",
        fcal=False,
    ):
        """
        Plot map at a particular wavelength or in a wavelength range

        Parameters
        ----------
        wavelength: float
          wavelength to be mapped.
        norm:
          Colour scale, default = colors.PowerNorm(gamma=1./4.)
            Normalization scale
            Lineal scale: norm=colors.Normalize().
            Log scale:norm=colors.LogNorm()
        cmap:
            Color map used, default cmap=fuego_color_map
            Velocities: cmap="seismic"
        save_file:
            (Optional) Save plot in file "file.extension"
        """

        if w2 == 0.0:
            interpolated_map = self.data[np.searchsorted(self.wavelength, wavelength)]
            description = "{} - {} $\AA$".format(self.description, wavelength)
        else:
            interpolated_map = np.nansum(
                self.data[
                    np.searchsorted(self.wavelength, wavelength): np.searchsorted(
                        self.wavelength, w2
                    )
                ],
                axis=0,
            )
            description = "{} - Integrating [{}-{}] $\AA$".format(
                self.description, wavelength, w2
            )

        self.plot_map(
            mapa=interpolated_map,
            norm=norm,
            fig_size=fig_size,
            cmap=cmap,
            save_file=save_file,
            description=description,
            fcal=fcal,
        )

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_map(
        self,
        mapa="",
        norm=colors.Normalize(),
        cmap="fuego",
        fig_size=10,
        fcal=False,
        save_file="",
        description="",
        contours=True,
        clabel=False,
        spaxel=0,
        spaxel2=0,
        spaxel3=0,
    ):
        """
        Show a given map.

        Parameters
        ----------
        map: np.array(float)
          Map to be plotted. If not given, it plots the integrated map.
        norm:
          Normalization scale, default is lineal scale.
          Lineal scale: norm=colors.Normalize().
          Log scale:    norm=colors.LogNorm()
          Power law:    norm=colors.PowerNorm(gamma=1./4.)
        cmap: (default cmap="fuego").
            Color map used.
            Weight: cmap = "gist_gray"
            Velocities: cmap="seismic".
            Try also "inferno",
        spaxel,spaxel2,spaxel3:
            [x,y] positions of spaxels to show with a green circle, blue square and red triangle
        """
        if description == "":
            description = self.description
        if mapa == "":
            mapa = self.integrated_map
            description = description + " - Integrated Map"

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        cax = ax.imshow(
            mapa,
            origin="lower",
            interpolation="none",
            norm=norm,
            cmap=cmap,
            extent=(
                -0.5 * self.n_cols * self.pixel_size_arcsec,
                0.5 * self.n_cols * self.pixel_size_arcsec,
                -0.5 * self.n_rows * self.pixel_size_arcsec,
                +0.5 * self.n_rows * self.pixel_size_arcsec,
            ),
        )
        if contours:
            CS = plt.contour(
                mapa,
                extent=(
                    -0.5 * self.n_cols * self.pixel_size_arcsec,
                    0.5 * self.n_cols * self.pixel_size_arcsec,
                    -0.5 * self.n_rows * self.pixel_size_arcsec,
                    +0.5 * self.n_rows * self.pixel_size_arcsec,
                ),
            )
            if clabel:
                plt.clabel(CS, inline=1, fontsize=10)

        ax.set_title(description, fontsize=14)
        plt.tick_params(labelsize=12)
        plt.xlabel("$\Delta$ RA [arcsec]", fontsize=12)
        plt.ylabel("$\Delta$ DEC [arcsec]", fontsize=12)
        plt.legend(loc="upper right", frameon=False)
        plt.minorticks_on()
        plt.grid(which="both", color="white")
        # plt.gca().invert_xaxis()   #MAMA

        if spaxel != 0:
            print("  The center of the cube is in spaxel [", self.spaxel_RA0, ",", self.spaxel_DEC0, "]")
            plt.plot([0], [0], "+", ms=13, color="black", mew=4)
            plt.plot([0], [0], "+", ms=10, color="white", mew=2)

            offset_from_center_x_arcsec = (
                spaxel[0] - self.spaxel_RA0 + 1.5
            ) * self.pixel_size_arcsec
            offset_from_center_y_arcsec = (
                spaxel[1] - self.spaxel_DEC0 + 1.5
            ) * self.pixel_size_arcsec
            print("  - Green circle:  ", spaxel, ",        Offset from center [arcsec] :   ", offset_from_center_x_arcsec, ",", offset_from_center_y_arcsec)
            plt.plot(
                [offset_from_center_x_arcsec],
                [offset_from_center_y_arcsec],
                "o",
                color="green",
                ms=7,
            )

        if spaxel2 != 0:
            offset_from_center_x_arcsec = (
                spaxel2[0] - self.spaxel_RA0 + 1.5
            ) * self.pixel_size_arcsec
            offset_from_center_y_arcsec = (
                spaxel2[1] - self.spaxel_DEC0 + 1.5
            ) * self.pixel_size_arcsec
            print("  - Blue  square:  ", np.round(
                spaxel2, 2
            ), ", Offset from center [arcsec] : ", np.round(
                offset_from_center_x_arcsec, 3
            ), ",", np.round(
                offset_from_center_y_arcsec, 3
            ))
            plt.plot(
                [offset_from_center_x_arcsec],
                [offset_from_center_y_arcsec],
                "s",
                color="blue",
                ms=7,
            )

        if spaxel3 != 0:
            offset_from_center_x_arcsec = (
                spaxel3[0] - self.spaxel_RA0 + 1.5
            ) * self.pixel_size_arcsec
            offset_from_center_y_arcsec = (
                spaxel3[1] - self.spaxel_DEC0 + 1.5
            ) * self.pixel_size_arcsec
            print("  - Red triangle:  ", np.round(
                spaxel3, 2
            ), ", Offset from center [arcsec] : ", np.round(
                offset_from_center_x_arcsec, 3
            ), ",", np.round(
                offset_from_center_y_arcsec, 3
            ))
            plt.plot(
                [offset_from_center_x_arcsec],
                [offset_from_center_y_arcsec],
                "v",
                color="red",
                ms=7,
            )

        cbar = fig.colorbar(cax, fraction=0.0457, pad=0.04)

        if fcal:
            barlabel = str("Integrated Flux [erg s$^{-1}$ cm$^{-2}$]")
        else:
            barlabel = str("Integrated Flux [Arbitrary units]")
        cbar.set_label(barlabel, rotation=270, labelpad=20, fontsize=14)
        #        cbar.ax.set_yticklabels(['< -1', '0', '> 1'])# vertically oriented colorbar

        if save_file == "":
            plt.show()
        else:
            plt.savefig(save_file)
        plt.close()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def create_map(self, wavelength1, wavelength2, name="NEW_MAP"):
        """
        Create map adding maps in a wavelength range."

        Parameters
        ----------
        wavelength1, wavelength2: floats
          The map will integrate all flux in the range [wavelength1, wavelength2].
        map_name: string
          String with the name of the map, must be the same than file created here.

        Example
        -------
        >>> a = cube.create_map(6810,6830, "a")
        > Created map with name  a  integrating range [ 6810 , 6830 ]
        """

        mapa = np.nansum(
            self.data[
                np.searchsorted(self.wavelength, wavelength1): np.searchsorted(
                    self.wavelength, wavelength2
                )
            ],
            axis=0,
        )

        print("\n> Created map with name ", name, " integrating range [", wavelength1, ",", wavelength2, "]")
        print("    Data shape" + str(np.shape(self.data)))
        print("    Int map shape" + str(np.shape(mapa)))
        return mapa

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def trace_peak_old(
        self, edgelow=10, edgehigh=10, plot=False, ADR=False, smoothfactor=2
    ):  # data=[-1000], wl=0, pixel_size_arcsec=0):
        print("\n\n> Tracing intensity peak over all wavelengths...")
        x = np.arange(self.n_cols)
        y = np.arange(self.n_rows)
        if ADR:
            print("  Checking ADR correction (small jumps are due to pixel size) ...")
            weight = np.nan_to_num(self.data_ADR)
            smoothfactor = 10
        else:
            weight = np.nan_to_num(self.data)

        mean_image = np.nanmean(weight, axis=0)
        mean_image /= np.nanmean(mean_image)
        weight *= mean_image[np.newaxis, :, :]
        xw = x[np.newaxis, np.newaxis, :] * weight
        yw = y[np.newaxis, :, np.newaxis] * weight
        w = np.nansum(weight, axis=(1, 2))
        self.x_peak = old_div(np.nansum(xw, axis=(1, 2)), w)
        self.y_peak = old_div(np.nansum(yw, axis=(1, 2)), w)
        self.x_peak_median = np.nanmedian(self.x_peak)
        self.y_peak_median = np.nanmedian(self.y_peak)
        self.x_peak_median_index = np.nanargmin(
            np.abs(self.x_peak - self.x_peak_median)
        )
        self.y_peak_median_index = np.nanargmin(
            np.abs(self.y_peak - self.y_peak_median)
        )

        wl = self.wavelength
        x = (
            self.x_peak - self.x_peak[self.x_peak_median_index]
        ) * self.pixel_size_arcsec
        y = (
            self.y_peak - self.y_peak[self.y_peak_median_index]
        ) * self.pixel_size_arcsec
        odd_number = (
            smoothfactor * int(old_div(np.sqrt(self.n_wave), 2)) + 1
        )  # Originarily, smoothfactor = 2

        # fit, trimming edges
        valid_wl = wl[edgelow: len(wl) - edgehigh]
        valid_x = x[edgelow: len(wl) - edgehigh]
        wlm = signal.medfilt(valid_wl, odd_number)
        wx = signal.medfilt(valid_x, odd_number)
        a3x, a2x, a1x, a0x = np.polyfit(wlm, wx, 3)
        fx = a0x + a1x * wl + a2x * wl ** 2 + a3x * wl ** 3
        fxm = a0x + a1x * wlm + a2x * wlm ** 2 + a3x * wlm ** 3

        valid_y = y[edgelow: len(wl) - edgehigh]
        wy = signal.medfilt(valid_y, odd_number)
        a3y, a2y, a1y, a0y = np.polyfit(wlm, wy, 3)
        fy = a0y + a1y * wl + a2y * wl ** 2 + a3y * wl ** 3
        fym = a0y + a1y * wlm + a2y * wlm ** 2 + a3y * wlm ** 3

        self.ADR_x = fx
        self.ADR_y = fy
        self.ADR_x_max = np.nanmax(self.ADR_x) - np.nanmin(self.ADR_x)
        self.ADR_y_max = np.nanmax(self.ADR_y) - np.nanmin(self.ADR_y)
        ADR_xy = np.sqrt(self.ADR_x ** 2 + self.ADR_y ** 2)
        self.ADR_total = np.nanmax(ADR_xy) - np.nanmin(ADR_xy)

        if plot:
            plt.figure(figsize=(10, 5))

            plt.plot(wl, fx, "-g", linewidth=3.5)
            plt.plot(wl, fy, "-g", linewidth=3.5)

            plt.plot(wl, x, "k.", alpha=0.2)
            plt.plot(wl, y, "r.", alpha=0.2)

            plt.plot(wl, signal.medfilt(x, odd_number), "k-")
            plt.plot(wl, signal.medfilt(y, odd_number), "r-")

            hi = np.max([np.nanpercentile(x, 95), np.nanpercentile(y, 95)])
            lo = np.min([np.nanpercentile(x, 5), np.nanpercentile(y, 5)])
            plt.ylim(lo, hi)
            plt.ylabel("$\Delta$ offset [arcsec]")
            plt.xlabel("Wavelength [$\AA$]")
            plt.title(self.description)
            plt.show()
            plt.close()
        print("> Peak coordinates tracing all wavelengths found in spaxel: ({:.2f}, {:.2f})".format(
            self.x_peak_median, self.y_peak_median
        ))
        print("  Effect of the ADR : {:.2f} in RA (black), {:.2f} in DEC (red),  TOTAL = +/- {:.2f} arcsec".format(
            self.ADR_x_max, self.ADR_y_max, self.ADR_total
        ))

        # Check numbers using SMOOTH data
        ADR_x_max = np.nanmax(fxm) - np.nanmin(fxm)
        ADR_y_max = np.nanmax(fym) - np.nanmin(fym)
        ADR_xy = np.sqrt(fxm ** 2 + fym ** 2)
        ADR_total = np.nanmax(ADR_xy) - np.nanmin(ADR_xy)
        print("  Using SMOOTH values: ")
        print("  Effect of the ADR : {:.2f} in RA (black), {:.2f} in DEC (red),  TOTAL = +/- {:.2f} arcsec".format(
            ADR_x_max, ADR_y_max, ADR_total
        ))

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def trace_peak(
        self, edgelow=10, edgehigh=10, plot=False, ADR=False, smoothfactor=2
    ):  # TASK_trace_peak
        print("\n\n> Tracing intensity peak over all wavelengths...")
        x = np.arange(self.n_cols)
        y = np.arange(self.n_rows)
        if ADR:
            print("  Checking ADR correction (small jumps are due to pixel size) ...")
            tmp = copy.deepcopy(self.data_ADR)
            tmp_img = np.nanmedian(tmp, axis=0)
            sort = np.sort(tmp_img.ravel())
            low_ind = np.where(tmp_img < sort[int(0.8 * len(sort))])
            for i in np.arange(len(low_ind[0])):
                tmp[:, low_ind[0][i], low_ind[1][i]] = np.nan
            weight = np.nan_to_num(tmp)  # self.data_ADR)
            smoothfactor = 10
        else:
            tmp = copy.deepcopy(self.data)
            tmp_img = np.nanmedian(tmp, axis=0)
            sort = np.sort(tmp_img.ravel())
            low_ind = np.where(tmp_img < sort[int(0.9 * len(sort))])
            # print(low_ind.shape)
            for i in np.arange(len(low_ind[0])):
                tmp[:, low_ind[0][i], low_ind[1][i]] = np.nan

            weight = np.nan_to_num(tmp)  # self.data)
        # try to median smooth image for better results?
        # weight=sig.medfilt(weight,kernel_size=[51,1,1])
        # also threshold the image so only the top 80% are used

        mean_image = np.nanmean(weight, axis=0)
        mean_image /= np.nanmean(mean_image)
        weight *= mean_image[np.newaxis, :, :]
        xw = x[np.newaxis, np.newaxis, :] * weight
        yw = y[np.newaxis, :, np.newaxis] * weight
        w = np.nansum(weight, axis=(1, 2))
        self.x_peak = old_div(np.nansum(xw, axis=(1, 2)), w)
        self.y_peak = old_div(np.nansum(yw, axis=(1, 2)), w)
        self.x_peak_median = np.nanmedian(self.x_peak)
        self.y_peak_median = np.nanmedian(self.y_peak)
        self.x_peak_median_index = np.nanargmin(
            np.abs(self.x_peak - self.x_peak_median)
        )
        self.y_peak_median_index = np.nanargmin(
            np.abs(self.y_peak - self.y_peak_median)
        )

        wl = self.wavelength
        x = (
            self.x_peak - self.x_peak[self.x_peak_median_index]
        ) * self.pixel_size_arcsec
        y = (
            self.y_peak - self.y_peak[self.y_peak_median_index]
        ) * self.pixel_size_arcsec
        odd_number = (
            smoothfactor * int(old_div(np.sqrt(self.n_wave), 2)) + 1
        )  # Originarily, smoothfactor = 2
        print("  Using medfilt window = ", odd_number)
        # fit, trimming edges
        index = np.arange(len(x))
        valid_ind = np.where(
            (index >= edgelow)
            & (index <= len(wl) - edgehigh)
            & (~np.isnan(x))
            & (~np.isnan(y))
        )[0]
        valid_wl = wl[valid_ind]
        valid_x = x[valid_ind]
        wlm = signal.medfilt(valid_wl, odd_number)
        wx = signal.medfilt(valid_x, odd_number)

        # iteratively clip and refit for WX
        maxit = 10
        niter = 0
        stop = 0
        fit_len = 100  # -100
        while stop < 1:
            # print '  Trying iteration ', niter,"..."
            # a2x,a1x,a0x = np.polyfit(wlm, wx, 2)
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
                p = np.polyfit(wlm[fit_index], wx[fit_index], 2)
                pp = np.poly1d(p)
                fx = pp(wl)
                fxm = pp(wlm)
                resid = wx - fxm
                # print "  Iteration {:2} results in RA: sigma_residual = {:.6f}, fit_len = {:5}  fit_len ={:5}".format(niter,sigma_resid,fit_len_init,fit_len)
            except Exception:
                print("  Skipping iteration ", niter)
            if (niter >= maxit) or (fit_len_init == fit_len):
                if niter >= maxit:
                    print("  x: Max iterations, {:2}, reached!")
                if fit_len_init == fit_len:
                    print("  x: All interval fitted in iteration {:2} ! ".format(niter))
                stop = 2
            niter = niter + 1

        # valid_y = y[edgelow:len(wl)-edgehigh]
        valid_ind = np.where(
            (index >= edgelow)
            & (index <= len(wl) - edgehigh)
            & (~np.isnan(x))
            & (~np.isnan(y))
        )[0]
        valid_y = y[valid_ind]
        wy = signal.medfilt(valid_y, odd_number)

        # iteratively clip and refit for WY
        maxit = 10
        niter = 0
        stop = 0
        fit_len = -100
        while stop < 1:
            fit_len_init = copy.deepcopy(fit_len)
            if niter == 0:
                fit_index = np.where(wy == wy)
                fit_len = len(fit_index)
                sigma_resid = 0.0
            if niter > 0:
                sigma_resid = MAD(resid)
                fit_index = np.where(np.abs(resid) < 4 * sigma_resid)[0]
                fit_len = len(fit_index)
            try:
                p = np.polyfit(wlm[fit_index], wy[fit_index], 2)
                pp = np.poly1d(p)
                fy = pp(wl)
                fym = pp(wlm)
                resid = wy - fym
                # print "  Iteration {:2} results in DEC: sigma_residual = {:.6f}, fit_len = {:5}  fit_len ={:5}".format(niter,sigma_resid,fit_len_init,fit_len)
            except Exception:
                print("  Skipping iteration ", niter)
            if (niter >= maxit) or (fit_len_init == fit_len):
                if niter >= maxit:
                    print("  y: Max iterations, {:2}, reached!")
                if fit_len_init == fit_len:
                    print("  y: All interval fitted in iteration {:2} ! ".format(niter))
                stop = 2
            niter = niter + 1

        self.ADR_x = fx
        self.ADR_y = fy
        self.ADR_x_max = np.nanmax(self.ADR_x) - np.nanmin(self.ADR_x)
        self.ADR_y_max = np.nanmax(self.ADR_y) - np.nanmin(self.ADR_y)
        ADR_xy = np.sqrt(self.ADR_x ** 2 + self.ADR_y ** 2)
        self.ADR_total = np.nanmax(ADR_xy) - np.nanmin(ADR_xy)
        if plot:
            plt.figure(figsize=(10, 5))

            plt.plot(wl, fx, "-g", linewidth=3.5)
            plt.plot(wl, fy, "-g", linewidth=3.5)

            plt.plot(wl, x, "k.", alpha=0.2)
            plt.plot(wl, y, "r.", alpha=0.2)

            plt.plot(wl, signal.medfilt(x, odd_number), "k-")
            plt.plot(wl, signal.medfilt(y, odd_number), "r-")

            hi = np.max([np.nanpercentile(x, 95), np.nanpercentile(y, 95)])
            lo = np.min([np.nanpercentile(x, 5), np.nanpercentile(y, 5)])
            plt.ylim(lo, hi)
            plt.ylabel("$\Delta$ offset [arcsec]")
            plt.xlabel("Wavelength [$\AA$]")
            plt.title(self.description)
            plt.show()
            plt.close()
        print("> Peak coordinates tracing all wavelengths found in spaxel: ({:.2f}, {:.2f})".format(
            self.x_peak_median, self.y_peak_median
        ))
        print("  Effect of the ADR : {:.2f} in RA (black), {:.2f} in DEC (red),  TOTAL = +/- {:.2f} arcsec".format(
            self.ADR_x_max, self.ADR_y_max, self.ADR_total
        ))

        # Check numbers using SMOOTH data
        ADR_x_max = np.nanmax(fxm) - np.nanmin(fxm)
        ADR_y_max = np.nanmax(fym) - np.nanmin(fym)
        ADR_xy = np.sqrt(fxm ** 2 + fym ** 2)
        ADR_total = np.nanmax(ADR_xy) - np.nanmin(ADR_xy)
        print("  Using SMOOTH values: ")
        print("  Effect of the ADR : {:.2f} in RA (black), {:.2f} in DEC (red),  TOTAL = +/- {:.2f} arcsec".format(
            ADR_x_max, ADR_y_max, ADR_total
        ))

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def growth_curve_between(self, min_wave=0, max_wave=0, plot=False, verbose=False):
        """
        Compute growth curve in a wavelength range.
        Returns r2_growth_curve, F_growth_curve, flux, r2_half_light

        Parameters
        ----------
        min_wave, max_wave: floats
          wavelength range = [min_wave, max_wave].
        plot: boolean
          Plot yes/no

        Example
        -------
        >>>r2_growth_curve, F_growth_curve, flux, r2_half_light = self.growth_curve_between(min_wave, max_wave, plot=True)    # 0,1E30 ??

        """

        if min_wave == 0:
            min_wave = self.valid_wave_min
        if max_wave == 0:
            max_wave = self.valid_wave_max

        if verbose:
            print("  - Calculating growth curve between ", min_wave, max_wave, " :")

        index_min = np.searchsorted(self.wavelength, min_wave)
        index_max = np.searchsorted(self.wavelength, max_wave)
        intensity = np.nanmean(self.data[index_min:index_max, :, :], axis=0)
        x_peak = np.median(self.x_peak[index_min:index_max])
        y_peak = np.median(self.y_peak[index_min:index_max])
        x = np.arange(self.n_cols) - x_peak
        y = np.arange(self.n_rows) - y_peak
        r2 = np.sum(np.meshgrid(x ** 2, y ** 2), axis=0)
        sorted_by_distance = np.argsort(r2, axis=None)

        F_growth_curve = []
        r2_growth_curve = []
        total_flux = 0.0
        for spaxel in sorted_by_distance:
            index = np.unravel_index(spaxel, (self.n_rows, self.n_cols))
            I = intensity[index]
            #        print spaxel, r2[index], L, total_flux, np.isnan(L)
            #        if np.isnan(L) == False and L > 0:
            if np.isnan(I) == False:
                total_flux += I  # TODO: Properly account for solid angle...
                F_growth_curve.append(total_flux)
                r2_growth_curve.append(r2[index])

        F_guess = np.max(F_growth_curve)
        r2_half_light = np.interp(0.5 * F_guess, F_growth_curve, r2_growth_curve)

        self.seeing = np.sqrt(r2_half_light) * self.pixel_size_arcsec

        if plot:
            r_norm = np.sqrt(old_div(np.array(r2_growth_curve), r2_half_light))
            F_norm = old_div(np.array(F_growth_curve), F_guess)
            print("      Flux guess =", F_guess, np.nansum(
                intensity
            ), " ratio = ", old_div(np.nansum(intensity), F_guess))
            print("      Half-light radius:", self.seeing, " arcsec  = seeing if object is a star ")
            print("      Light within 2, 3, 4, 5 half-light radii:", np.interp(
                [2, 3, 4, 5], r_norm, F_norm
            ))
            plt.figure(figsize=(10, 8))
            plt.plot(r_norm, F_norm, "-")
            plt.title(
                "Growth curve between "
                + str(min_wave)
                + " and "
                + str(max_wave)
                + " in "
                + self.object
            )
            plt.xlabel("Radius [arcsec]")
            plt.ylabel("Flux")
            plt.axvline(x=self.seeing, color="g", alpha=0.7)
            plt.axhline(y=0.5, color="k", linestyle=":", alpha=0.5)
            plt.axvline(x=2 * self.seeing, color="k", linestyle=":", alpha=0.2)
            plt.axvline(x=3 * self.seeing, color="k", linestyle=":", alpha=0.2)
            plt.axvline(x=4 * self.seeing, color="k", linestyle=":", alpha=0.2)
            plt.axvline(x=5 * self.seeing, color="r", linestyle="--", alpha=0.2)
            #            plt.axhline(y=np.interp([2, 3, 4], r_norm, F_norm), color='k', linestyle=':', alpha=0.2)
            plt.axhline(
                y=np.interp([6], r_norm, F_norm), color="r", linestyle="--", alpha=0.2
            )
            plt.minorticks_on()
            plt.show()
            plt.close()

        return r2_growth_curve, F_growth_curve, F_guess, r2_half_light

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def half_light_spectrum(
        self, r_max=1, plot=False, smooth=21, min_wave=0, max_wave=0
    ):
        """
        Compute half light spectrum (for r_max=1) or integrated star spectrum (for r_max=5) in a wavelength range.

        Parameters
        ----------
        r_max = 1: float
          r_max to integrate, in units of r2_half_light (= seeing if object is a star, for flux calibration make r_max=5)
        min_wave, max_wave: floats
          wavelength range = [min_wave, max_wave]
        smooth = 21: float
          smooth the data
        plot: boolean
          Plot yes/no

        Example
        -------
        >>> self.half_light_spectrum(5, plot=plot, min_wave=min_wave, max_wave=max_wave)
        """

        if min_wave == 0:
            min_wave = self.valid_wave_min
        if max_wave == 0:
            max_wave = self.valid_wave_max

        (
            r2_growth_curve,
            F_growth_curve,
            flux,
            r2_half_light,
        ) = self.growth_curve_between(
            min_wave, max_wave, plot=True, verbose=True
        )  # 0,1E30 ??
        # print "\n> Computing growth-curve spectrum..."
        intensity = []
        smooth_x = signal.medfilt(self.x_peak, smooth)  # originally, smooth = 11
        smooth_y = signal.medfilt(self.y_peak, smooth)
        edgelow = (np.abs(self.wavelength - min_wave)).argmin()
        edgehigh = (np.abs(self.wavelength - max_wave)).argmin()
        valid_wl = self.wavelength[edgelow:edgehigh]

        for l in range(self.n_wave):  # self.n_wave
            # wavelength = self.wavelength[l]
            # if l % (self.n_wave/10+1) == 0:
            #    print "  {:.2f} Angstroms (wavelength {}/{})..." \
            #          .format(wavelength, l+1, self.n_wave)
            x = np.arange(self.n_cols) - smooth_x[l]
            y = np.arange(self.n_rows) - smooth_y[l]
            r2 = np.sum(np.meshgrid(x ** 2, y ** 2), axis=0)
            spaxels = np.where(r2 < r2_half_light * r_max ** 2)
            intensity.append(np.nansum(self.data[l][spaxels]))

        valid_intensity = intensity[edgelow:edgehigh]
        valid_wl_smooth = signal.medfilt(valid_wl, smooth)
        valid_intensity_smooth = signal.medfilt(valid_intensity, smooth)

        if plot:
            fig_size = 12
            plt.figure(figsize=(fig_size, fig_size / 2.5))
            plt.plot(self.wavelength, intensity, "b", alpha=1, label="Intensity")
            plt.plot(
                valid_wl_smooth,
                valid_intensity_smooth,
                "r-",
                alpha=0.5,
                label="Smooth = " + str(smooth),
            )
            margen = 0.1 * (np.nanmax(intensity) - np.nanmin(intensity))
            plt.ylim(np.nanmin(intensity) - margen, np.nanmax(intensity) + margen)
            plt.xlim(np.min(self.wavelength), np.max(self.wavelength))

            plt.ylabel("Flux")
            plt.xlabel("Wavelength [$\AA$]")
            plt.title(
                "Integrated spectrum of "
                + self.object
                + " for r_half_light = "
                + str(r_max)
            )
            plt.axvline(x=min_wave, color="k", linestyle="--", alpha=0.5)
            plt.axvline(x=max_wave, color="k", linestyle="--", alpha=0.5)
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
    def do_response_curve(
        self,
        filename,
        min_wave=0,
        max_wave=0,
        step=25.0,
        fit_degree=3,
        exp_time=60,
        smooth=0.03,
        ha_width=0,
        plot=True,
        verbose=False,
    ):  # smooth new 5 Mar, smooth=21, now we don't use it
        """
        Compute the response curve of a spectrophotometric star.

        Parameters
        ----------
        filename: string
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
        >>> babbsdsad
        """
        if min_wave == 0:
            min_wave = self.valid_wave_min
        if max_wave == 0:
            max_wave = self.valid_wave_max

        print("\n> Computing response curve for", self.object, "using step=", step, " in range [", min_wave, ",", max_wave, "] ...")

        #        flux_cal_read in units of ergs/cm/cm/s/A * 10**16
        #        lambda_cal_read, flux_cal_read, delta_lambda_read = np.loadtxt(filename, usecols=(0,1,3), unpack=True)
        lambda_cal_read, flux_cal_read = np.loadtxt(
            filename, usecols=(0, 1), unpack=True
        )

        valid_wl_smooth = np.arange(lambda_cal_read[0], lambda_cal_read[-1], step)
        tck_star = interpolate.splrep(lambda_cal_read, flux_cal_read, s=0)
        valid_flux_smooth = interpolate.splev(valid_wl_smooth, tck_star, der=0)
        valid_wave_min = min_wave
        valid_wave_max = max_wave
        edgelow = (np.abs(valid_wl_smooth - valid_wave_min)).argmin()
        edgehigh = (np.abs(valid_wl_smooth - valid_wave_max)).argmin()

        lambda_cal = valid_wl_smooth[edgelow:edgehigh]
        flux_cal = valid_flux_smooth[edgelow:edgehigh]
        lambda_min = lambda_cal - step
        lambda_max = lambda_cal + step

        if (
            self.flux_cal_step == step
            and self.flux_cal_min_wave == min_wave
            and self.flux_cal_max_wave == max_wave
        ):
            print("  This has been computed before for step=", step, " in range [", min_wave, ",", max_wave, "], using values computed before...")
            measured_counts = self.flux_cal_measured_counts
        else:
            measured_counts = np.array(
                [
                    self.fit_Moffat_between(lambda_min[i], lambda_max[i])[0]
                    if lambda_cal[i] > min_wave
                    and lambda_cal[i] < max_wave  # 6200  #3650  # 7400  #5700
                    else np.NaN
                    for i in range(len(lambda_cal))
                ]
            )

            self.flux_cal_step = step
            self.flux_cal_min_wave = min_wave
            self.flux_cal_max_wave = max_wave
            self.flux_cal_measured_counts = measured_counts

        _response_curve_ = (
            old_div(old_div(measured_counts, flux_cal), exp_time)
        )  # Added exp_time Jan 2019       counts / (ergs/cm/cm/s/A * 10**16) / s  = counts * ergs*cm*cm*A / 10**16

        if np.isnan(_response_curve_[0]) == True:
            _response_curve_[0] = _response_curve_[
                1
            ]  # - (response_curve[2] - response_curve[1])

        scale = np.nanmedian(_response_curve_)
        # self.integrated_star_flux = self.half_light_spectrum(5, plot=plot, min_wave=min_wave, max_wave=max_wave)

        edgelow_ = (np.abs(self.wavelength - lambda_cal[0])).argmin()
        edgehigh_ = (np.abs(self.wavelength - lambda_cal[-1])).argmin()
        self.response_wavelength = self.wavelength[edgelow_:edgehigh_]

        response_wavelength = []
        response_curve = []

        if ha_width > 0:
            skipping = 0
            print("  Skipping H-alpha absorption with width =", ha_width, "A ...")
            for i in range(len(lambda_cal)):
                if (
                    lambda_cal[i] > 6563 - ha_width / 2.0
                    and lambda_cal[i] < 6563 + ha_width / 2.0
                ):
                    # print "  Skipping ",lambda_cal[i]
                    skipping = skipping + 1
                else:
                    response_wavelength.append(lambda_cal[i])
                    response_curve.append(_response_curve_[i])
            print("  ... Skipping a total of ", skipping, "wavelength points")
        else:
            response_wavelength = lambda_cal
            response_curve = _response_curve_

        if fit_degree == 0:
            print("  Using interpolated data with smooth = ", smooth, " for computing the response curve... ")

            median_kernel = 151
            response_curve_medfilt = sig.medfilt(response_curve, np.int(median_kernel))
            interpolated_flat = interpolate.splrep(
                response_wavelength, response_curve_medfilt, s=smooth
            )
            self.response_curve = interpolate.splev(
                self.response_wavelength, interpolated_flat, der=0
            )

        else:
            if fit_degree != 9:
                if fit_degree != 7:
                    if fit_degree != 5:
                        if fit_degree != 3:
                            print("  We can't use a polynomium of grade ", fit_degree, " here, using fit_degree = 3 instead")
                            fit_degree = 3
            if fit_degree == 3:
                a3x, a2x, a1x, a0x = np.polyfit(response_wavelength, response_curve, 3)
                a4x = 0
                a5x = 0
                a6x = 0
                a7x = 0
                a8x = 0
                a9x = 0
            if fit_degree == 5:
                a5x, a4x, a3x, a2x, a1x, a0x = np.polyfit(
                    response_wavelength, response_curve, 5
                )
                a6x = 0
                a7x = 0
                a8x = 0
                a9x = 0
            if fit_degree == 7:
                a7x, a6x, a5x, a4x, a3x, a2x, a1x, a0x = np.polyfit(
                    response_wavelength, response_curve, 7
                )
                a8x = 0
                a9x = 0
            if fit_degree == 9:
                a9x, a8x, a7x, a6x, a5x, a4x, a3x, a2x, a1x, a0x = np.polyfit(
                    response_wavelength, response_curve, 9
                )
            wlm = self.response_wavelength
            self.response_curve = (
                a0x
                + a1x * wlm
                + a2x * wlm ** 2
                + a3x * wlm ** 3
                + a4x * wlm ** 4
                + a5x * wlm ** 5
                + a6x * wlm ** 6
                + a7x * wlm ** 7
                + a8x * wlm ** 8
                + a9x * wlm ** 9
            )  # Better use next

        # Adapting Matt code for trace peak ----------------------------------

        smoothfactor = 2
        wl = response_wavelength  # response_wavelength
        x = response_curve
        odd_number = (
            smoothfactor * int(old_div(np.sqrt(len(wl)), 2)) - 1
        )  # Originarily, smoothfactor = 2
        print("  Using medfilt window = ", odd_number, " for fitting...")
        # fit, trimming edges
        # index=np.arange(len(x))
        # edgelow=0
        # edgehigh=1
        # valid_ind=np.where((index >= edgelow) & (index <= len(wl)-edgehigh) & (~np.isnan(x)) )[0]
        # print valid_ind
        # valid_wl = wl[edgelow:-edgehigh] # wl[valid_ind]
        # valid_x = x[edgelow:-edgehigh] #x[valid_ind]
        # wlm = signal.medfilt(valid_wl, odd_number)
        # wx = signal.medfilt(valid_x, odd_number)
        wlm = signal.medfilt(wl, odd_number)
        wx = signal.medfilt(x, odd_number)

        # iteratively clip and refit for WX
        maxit = 10
        niter = 0
        stop = 0
        fit_len = 100  # -100
        while stop < 1:
            # print '  Trying iteration ', niter,"..."
            # a2x,a1x,a0x = np.polyfit(wlm, wx, 2)
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
                p = np.polyfit(wlm[fit_index], wx[fit_index], fit_degree)
                pp = np.poly1d(p)
                fx = pp(wl)
                fxm = pp(wlm)
                resid = wx - fxm
                # print "  Iteration {:2} results in RA: sigma_residual = {:.6f}, fit_len = {:5}  fit_len ={:5}".format(niter,sigma_resid,fit_len_init,fit_len)
            except Exception:
                print("  Skipping iteration ", niter)
            if (niter >= maxit) or (fit_len_init == fit_len):
                if niter >= maxit:
                    print("  Max iterations, {:2}, reached!")
                if fit_len_init == fit_len:
                    print("  All interval fitted in iteration {:2} ! ".format(niter))
                stop = 2
            niter = niter + 1

        # --------------------------------------------------------------------

        if plot:
            plt.figure(figsize=(10, 8))
            plt.plot(
                lambda_cal,
                old_div(measured_counts, exp_time),
                "g+",
                ms=10,
                mew=3,
                label="measured counts",
            )
            plt.plot(lambda_cal, flux_cal * scale, "k*-", label="flux_cal * scale")
            plt.plot(
                lambda_cal,
                flux_cal * _response_curve_,
                "c:",
                label="flux_cal * response",
            )
            plt.xlim(np.min(self.wavelength), np.max(self.wavelength))
            plt.ylabel("Flux")
            plt.xlabel("Wavelength [$\AA$]")
            plt.title(
                "Response curve for absolute flux calibration using " + self.object
            )
            plt.legend(frameon=False, loc=1)
            plt.grid(which="both")
            plt.axvline(x=min_wave, color="k", linestyle="--", alpha=0.5)
            plt.axvline(x=max_wave, color="k", linestyle="--", alpha=0.5)
            plt.minorticks_on()
            plt.show()
            plt.close()

            plt.figure(figsize=(10, 8))
            if fit_degree > 0:
                text = "Fit using polynomium of degree " + np.str(fit_degree)
            else:
                text = "Using interpolated data with smooth = " + np.str(smooth)
            plt.plot(
                self.response_wavelength,
                self.response_curve,
                "r-",
                alpha=0.4,
                linewidth=4,
                label=text,
            )
            plt.plot(lambda_cal, _response_curve_, "k--", alpha=0.8)
            plt.plot(
                response_wavelength,
                response_curve,
                "g-",
                alpha=0.8,
                label="Response curve",
            )

            plt.plot(
                wl, fx, "b-", linewidth=6, alpha=0.5, label="Response curve (filtered)"
            )

            plt.xlim(np.min(self.wavelength), np.max(self.wavelength))
            plt.ylabel("Flux")
            plt.xlabel("Wavelength [$\AA$]")
            plt.title(
                "Response curve for absolute flux calibration using " + self.object
            )
            plt.minorticks_on()
            plt.grid(which="both")
            plt.axvline(x=min_wave, color="k", linestyle="--", alpha=0.5)
            plt.axvline(x=max_wave, color="k", linestyle="--", alpha=0.5)
            plt.legend(frameon=True, loc=4, ncol=4)
            plt.show()
            plt.close()

        interpolated_flat = interpolate.splrep(response_wavelength, fx)  # , s=smooth)
        self.response_curve = interpolate.splev(
            self.response_wavelength, interpolated_flat, der=0
        )

        #    plt.plot(self.response_wavelength, self.response_curve, "b-", alpha=0.5, linewidth=6, label = "Response curve (filtered)")

        print("  Min wavelength at {:.2f} with value = {:.3f} /s".format(
            self.response_wavelength[0], self.response_curve[0]
        ))
        print("  Max wavelength at {:.2f} with value = {:.3f} /s".format(
            self.response_wavelength[-1], self.response_curve[-1]
        ))

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def fit_Moffat_between(self, min_wave=0, max_wave=0, r_max=5, plot=False):

        if min_wave == 0:
            min_wave = self.valid_wave_min
        if max_wave == 0:
            max_wave = self.valid_wave_max

        (
            r2_growth_curve,
            F_growth_curve,
            flux,
            r2_half_light,
        ) = self.growth_curve_between(min_wave, max_wave, plot)
        flux, alpha, beta = fit_Moffat(
            r2_growth_curve, F_growth_curve, flux, r2_half_light, r_max, plot
        )
        r2_half_light = alpha * (np.power(2.0, 1.0 / beta) - 1)
        if plot:
            print("Moffat fit: Flux = {:.3e},".format(flux), "HWHM = {:.3f},".format(
                np.sqrt(r2_half_light) * self.pixel_size_arcsec
            ), "beta = {:.3f}".format(beta))

        return flux, np.sqrt(r2_half_light) * self.pixel_size_arcsec, beta


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# GENERAL TASKS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return old_div((cumsum[N:] - cumsum[:-N]), N)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def cumulaive_Moffat(r2, L_star, alpha2, beta):
    return L_star * (1 - np.power(1 + (old_div(r2, alpha2)), -beta))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fit_Moffat(
    r2_growth_curve, F_growth_curve, F_guess, r2_half_light, r_max, plot=False
):
    """
    Fits a Moffat profile to a flux growth curve
    as a function of radius squared,
    cutting at to r_max (in units of the half-light radius),
    provided an initial guess of the total flux and half-light radius squared.
    """
    index_cut = np.searchsorted(r2_growth_curve, r2_half_light * r_max ** 2)
    fit, cov = optimize.curve_fit(
        cumulaive_Moffat,
        r2_growth_curve[:index_cut],
        F_growth_curve[:index_cut],
        p0=(F_guess, r2_half_light, 1),
    )
    if plot:
        print("Best-fit: L_star =", fit[0])
        print("          alpha =", np.sqrt(fit[1]))
        print("          beta =", fit[2])
        r_norm = np.sqrt(old_div(np.array(r2_growth_curve), r2_half_light))
        plt.plot(
            r_norm,
            old_div(cumulaive_Moffat(np.array(r2_growth_curve), fit[0], fit[1], fit[2]), fit[0]),
            ":",
        )

    return fit


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def KOALA_offsets(s, pa):
    print("\n> Offsets towards North and East between pointings," "according to KOALA manual, for pa =", pa, "degrees")
    pa *= old_div(np.pi, 180)
    print("  a -> b :", s * np.sin(pa), -s * np.cos(pa))
    print("  a -> c :", -s * np.sin(60 - pa), -s * np.cos(60 - pa))
    print("  b -> d :", -np.sqrt(3) * s * np.cos(pa), -np.sqrt(3) * s * np.sin(pa))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def offset_between_cubes(cube1, cube2, plot=True):
    x = (
        cube2.x_peak
        - cube2.n_cols / 2.0
        + cube2.RA_centre_deg * 3600.0 / cube2.pixel_size_arcsec
    ) - (
        cube1.x_peak
        - cube1.n_cols / 2.0
        + cube1.RA_centre_deg * 3600.0 / cube1.pixel_size_arcsec
    )
    y = (
        cube2.y_peak
        - cube2.n_rows / 2.0
        + cube2.DEC_centre_deg * 3600.0 / cube2.pixel_size_arcsec
    ) - (
        cube1.y_peak
        - cube1.n_rows / 2.0
        + cube1.DEC_centre_deg * 3600.0 / cube1.pixel_size_arcsec
    )
    delta_RA_pix = np.nanmedian(x)
    delta_DEC_pix = np.nanmedian(y)
    #    weight = np.nansum(cube1.data+cube2.data, axis=(1, 2))
    #    total_weight = np.nansum(weight)
    #    print "--- lambda=", np.nansum(cube1.RSS.wavelength*weight) / total_weight
    #    delta_RA_pix = np.nansum(x*weight) / total_weight
    #    delta_DEC_pix = np.nansum(y*weight) / total_weight
    delta_RA_arcsec = delta_RA_pix * cube1.pixel_size_arcsec
    delta_DEC_arcsec = delta_DEC_pix * cube1.pixel_size_arcsec
    print("(delta_RA, delta_DEC) = ({:.3f}, {:.3f}) arcsec".format(
        delta_RA_arcsec, delta_DEC_arcsec
    ))
    #    delta_RA_headers = (cube2.RSS.RA_centre_deg - cube1.RSS.RA_centre_deg) * 3600
    #    delta_DEC_headers = (cube2.RSS.DEC_centre_deg - cube1.RSS.DEC_centre_deg) * 3600
    #    print '                        ({:.3f}, {:.3f}) arcsec according to headers!!!???' \
    #        .format(delta_RA_headers, delta_DEC_headers)
    #    print 'difference:             ({:.3f}, {:.3f}) arcsec' \
    #        .format(delta_RA-delta_RA_headers, delta_DEC-delta_DEC_headers)

    if plot:
        x -= delta_RA_pix
        y -= delta_DEC_pix
        smooth_x = signal.medfilt(x, 151)
        smooth_y = signal.medfilt(y, 151)

        print(np.nanmean(smooth_x))
        print(np.nanmean(smooth_y))

        plt.figure(figsize=(10, 5))
        wl = cube1.RSS.wavelength
        plt.plot(wl, x, "k.", alpha=0.1)
        plt.plot(wl, y, "r.", alpha=0.1)
        plt.plot(wl, smooth_x, "k-")
        plt.plot(wl, smooth_y, "r-")
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
    plt.imshow(
        map1 - map2, vmin=-scale, vmax=scale, cmap=plt.cm.get_cmap("RdBu")
    )  # vmin = -scale
    plt.colorbar()
    plt.contour(map1, colors="w", linewidths=2, norm=colors.LogNorm())
    plt.contour(map2, colors="k", linewidths=1, norm=colors.LogNorm())
    if line != 0:
        plt.title("{:.2f} AA".format(line))
    else:
        plt.title("Integrated Map")
    plt.show()
    plt.close()
    print("  Medium scatter : ", scatter)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_response(calibration_star_cubes, scale=[1, 1, 1, 1]):
    print("\n> Plotting response of standard stars...\n")
    plt.figure(figsize=(11, 8))
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
        plt.plot(
            star.response_wavelength,
            star.response_curve * scale[i],
            label=star.description,
            alpha=0.5,
            linewidth=2,
        )
        print("  Mean value for ", star.object, " = ", np.nanmean(
            star.response_curve * scale[i]
        ), "      scale = ", scale[i])
        mean_values.append(np.nanmean(star.response_curve) * scale[i])
        i = i + 1
    mean_curve /= len(calibration_star_cubes)

    response_rms = np.zeros_like(wavelength)
    for star in calibration_star_cubes:
        response_rms += np.abs(star.response_full - mean_curve)
    response_rms /= len(calibration_star_cubes)
    dispersion = old_div(np.nansum(response_rms), np.nansum(mean_curve))
    print("  Variation in flux calibrations =  {:.2f} %".format(dispersion * 100.0))

    # dispersion=np.nanmax(mean_values)-np.nanmin(mean_values)
    # print "  Variation in flux calibrations =  {:.2f} %".format(dispersion/np.nanmedian(mean_values)*100.)

    plt.plot(
        wavelength,
        mean_curve,
        "k",
        label="mean response curve",
        alpha=0.2,
        linewidth=10,
    )
    plt.legend(frameon=False, loc=2)
    plt.ylabel("FLux")
    plt.xlabel("Wavelength [$\AA$]")
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
            vector_wave.append(cube_star.response_wavelength[i])
            vector_response.append(cube_star.response_curve[i])
            # print "  For wavelength = ",cube_star.response_wavelength[i], " the flux correction is = ", cube_star.response_curve[i]

    interpolated_response = interpolate.splrep(vector_wave, vector_response, s=0)
    flux_calibration = interpolate.splev(
        cube_star.wavelength, interpolated_response, der=0
    )
    #    flux_correction = flux_calibration

    print("\n> Flux calibration for all wavelengths = ", flux_calibration)
    print("\n  Flux calibration obtained!")
    return flux_calibration


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def obtain_telluric_correction(wlm, telluric_correction_list, plot=True):
    telluric_correction = np.nanmedian(telluric_correction_list, axis=0)
    if plot:
        fig_size = 12
        plt.figure(figsize=(fig_size, fig_size / 2.5))
        plt.title("Telluric correction")
        for i in range(len(telluric_correction_list)):
            label = "star" + str(i + 1)
            plt.plot(wlm, telluric_correction_list[i], alpha=0.3, label=label)
        plt.plot(wlm, telluric_correction, alpha=0.5, color="k", label="Median")
        plt.minorticks_on()
        plt.legend(frameon=False, loc=2, ncol=1)
        step_up = 1.15 * np.nanmax(telluric_correction)
        plt.ylim(0.9, step_up)
        plt.xlim(wlm[0] - 10, wlm[-1] + 10)
        plt.show()
        plt.close()

    print("\n> Telluric correction = ", telluric_correction)
    print("\n  Telluric correction obtained!")
    return telluric_correction


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def coord_range(rss_list):
    RA = [rss.RA_centre_deg + rss.offset_RA_arcsec / 3600.0 for rss in rss_list]
    RA_min = np.nanmin(RA)
    RA_max = np.nanmax(RA)
    DEC = [rss.DEC_centre_deg + rss.offset_DEC_arcsec / 3600.0 for rss in rss_list]
    DEC_min = np.nanmin(DEC)
    DEC_max = np.nanmax(DEC)
    return RA_min, RA_max, DEC_min, DEC_max


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def align_3_cubes(
    cube1,
    cube2,
    cube3,
    rss1,
    rss2,
    rss3,
    pixel_size_arcsec=0.3,
    kernel_size_arcsec=1.5,
    offsets=[1000],
    plot=False,
    ADR=False,
    warnings=False,
):
    """
    Routine to align 3 cubes

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
    print("\n> Starting alignment procedure...")

    # pointings_RSS=[rss1, rss2, rss3, rss4]
    # RA_min, RA_max, DEC_min, DEC_max = coord_range(pointings_RSS)

    if offsets[0] == 1000:
        #        print "  Using peak in integrated image to align cubes:"
        #        x12 = cube1.offset_from_center_x_arcsec_integrated - cube2.offset_from_center_x_arcsec_integrated
        #        y12 = cube1.offset_from_center_y_arcsec_integrated - cube2.offset_from_center_y_arcsec_integrated
        #        x23 = cube2.offset_from_center_x_arcsec_integrated - cube3.offset_from_center_x_arcsec_integrated
        #        y23 = cube2.offset_from_center_y_arcsec_integrated - cube3.offset_from_center_y_arcsec_integrated
        print("  Using peak of the emission tracing all wavelengths to align cubes:")
        x12 = (
            cube2.offset_from_center_x_arcsec_tracing
            - cube1.offset_from_center_x_arcsec_tracing
        )
        y12 = (
            cube2.offset_from_center_y_arcsec_tracing
            - cube1.offset_from_center_y_arcsec_tracing
        )
        x23 = (
            cube3.offset_from_center_x_arcsec_tracing
            - cube2.offset_from_center_x_arcsec_tracing
        )
        y23 = (
            cube3.offset_from_center_y_arcsec_tracing
            - cube2.offset_from_center_y_arcsec_tracing
        )
        x31 = (
            cube1.offset_from_center_x_arcsec_tracing
            - cube3.offset_from_center_x_arcsec_tracing
        )
        y31 = (
            cube1.offset_from_center_y_arcsec_tracing
            - cube3.offset_from_center_y_arcsec_tracing
        )

    else:
        print("  Using offsets given by the user:")
        x12 = offsets[0]
        y12 = offsets[1]
        x23 = offsets[2]
        y23 = offsets[3]
        x31 = -(offsets[0] + offsets[2])
        y31 = -(offsets[1] + offsets[3])

    rss1.ALIGNED_RA_centre_deg = cube1.RA_centre_deg
    rss1.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg
    rss2.ALIGNED_RA_centre_deg = cube1.RA_centre_deg - x12 / 3600.0
    rss2.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg - y12 / 3600.0
    rss3.ALIGNED_RA_centre_deg = cube1.RA_centre_deg + x31 / 3600.0
    rss3.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg + y31 / 3600.0

    RA_centre_deg = rss1.ALIGNED_RA_centre_deg
    DEC_centre_deg = rss1.ALIGNED_DEC_centre_deg

    print("\n  Offsets (in arcsec):")
    print("  Offsets in x : ", x12, x23, "      Total offset in x = ", x12 + x23 + x31)
    print("  Offsets in y : ", y12, y23, "      Total offset in y = ", y12 + y23 + y31)

    print("\n>        New_RA_centre_deg       New_DEC_centre_deg       Diff respect Cube 1 (arcsec)")
    print("  Cube 1 : ", rss1.ALIGNED_RA_centre_deg, "     ", rss1.ALIGNED_DEC_centre_deg, "           0 0")
    print("  Cube 2 : ", rss2.ALIGNED_RA_centre_deg, "     ", rss2.ALIGNED_DEC_centre_deg, "          ", (
        rss2.ALIGNED_RA_centre_deg - rss1.ALIGNED_RA_centre_deg
    ) * 3600.0, (
        rss2.ALIGNED_DEC_centre_deg - rss1.ALIGNED_DEC_centre_deg
    ) * 3600.0)
    print("  Cube 3 : ", rss3.ALIGNED_RA_centre_deg, "     ", rss3.ALIGNED_DEC_centre_deg, "          ", (
        rss3.ALIGNED_RA_centre_deg - rss1.ALIGNED_RA_centre_deg
    ) * 3600.0, (
        rss3.ALIGNED_DEC_centre_deg - rss1.ALIGNED_DEC_centre_deg
    ) * 3600.0)

    offsets_files = [
        [x12, y12],
        [x23, y23],
    ]  # For keeping in the files with self.offsets_files
    #    RA_size_arcsec = rss1.RA_segment + np.abs(x12)+np.abs(x23) + 2*kernel_size_arcsec
    #    DEC_size_arcsec =rss1.DEC_segment +np.abs(y12)+np.abs(y23)  +2*kernel_size_arcsec

    RA_size_arcsec = rss1.RA_segment + x12 + x23 + 2 * kernel_size_arcsec
    DEC_size_arcsec = rss1.DEC_segment + y12 + y23 + 2 * kernel_size_arcsec

    #    print "  RA_centre_deg , DEC_centre_deg   = ", RA_centre_deg, DEC_centre_deg
    print("  RA_size x DEC_size  = {:.2f} arcsec x {:.2f} arcsec".format(
        RA_size_arcsec, DEC_size_arcsec
    ))

    #    probando=raw_input("Continue?")

    cube1_aligned = Interpolated_cube(
        rss1,
        pixel_size_arcsec,
        kernel_size_arcsec,
        centre_deg=[RA_centre_deg, DEC_centre_deg],
        size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
        aligned_coor=True,
        flux_calibration=cube1.flux_calibration,
        offsets_files=offsets_files,
        offsets_files_position=1,
        plot=plot,
        ADR=ADR,
        warnings=warnings,
    )
    cube2_aligned = Interpolated_cube(
        rss2,
        pixel_size_arcsec,
        kernel_size_arcsec,
        centre_deg=[RA_centre_deg, DEC_centre_deg],
        size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
        aligned_coor=True,
        flux_calibration=cube2.flux_calibration,
        offsets_files=offsets_files,
        offsets_files_position=2,
        plot=plot,
        ADR=ADR,
        warnings=warnings,
    )
    cube3_aligned = Interpolated_cube(
        rss3,
        pixel_size_arcsec,
        kernel_size_arcsec,
        centre_deg=[RA_centre_deg, DEC_centre_deg],
        size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
        aligned_coor=True,
        flux_calibration=cube3.flux_calibration,
        offsets_files=offsets_files,
        offsets_files_position=3,
        plot=plot,
        ADR=ADR,
        warnings=warnings,
    )

    print("\n> Checking offsets of ALIGNED cubes (in arcsec):")

    x12 = (
        cube1_aligned.offset_from_center_x_arcsec_tracing
        - cube2_aligned.offset_from_center_x_arcsec_tracing
    )
    y12 = (
        cube1_aligned.offset_from_center_y_arcsec_tracing
        - cube2_aligned.offset_from_center_y_arcsec_tracing
    )
    x23 = (
        cube2_aligned.offset_from_center_x_arcsec_tracing
        - cube3_aligned.offset_from_center_x_arcsec_tracing
    )
    y23 = (
        cube2_aligned.offset_from_center_y_arcsec_tracing
        - cube3_aligned.offset_from_center_y_arcsec_tracing
    )
    x31 = (
        cube3_aligned.offset_from_center_x_arcsec_tracing
        - cube1_aligned.offset_from_center_x_arcsec_tracing
    )
    y31 = (
        cube3_aligned.offset_from_center_y_arcsec_tracing
        - cube1_aligned.offset_from_center_y_arcsec_tracing
    )

    print("  Offsets in x : {:.3f}   {:.3f}   {:.3f}       Total offset in x = {:.3f}".format(
        x12, x23, x31, x12 + x23 + x31
    ))
    print("  Offsets in y : {:.3f}   {:.3f}   {:.3f}       Total offset in y = {:.3f}".format(
        y12, y23, y31, y12 + y23 + y31
    ))

    print("\n> Updated values for Alignment DONE")
    return cube1_aligned, cube2_aligned, cube3_aligned


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def align_4_cubes(
    cube1,
    cube2,
    cube3,
    cube4,
    rss1,
    rss2,
    rss3,
    rss4,
    pixel_size_arcsec=0.3,
    kernel_size_arcsec=1.5,
    offsets=[1000],
    plot=False,
    ADR=False,
    warnings=False,
):
    """
    Routine to align 4 cubes

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
    print("\n> Starting alignment procedure...")

    # pointings_RSS=[rss1, rss2, rss3, rss4]
    # RA_min, RA_max, DEC_min, DEC_max = coord_range(pointings_RSS)

    if offsets[0] == 1000:
        #        print "  Using peak in integrated image to align cubes:"
        #        x12 = cube1.offset_from_center_x_arcsec_integrated - cube2.offset_from_center_x_arcsec_integrated
        #        y12 = cube1.offset_from_center_y_arcsec_integrated - cube2.offset_from_center_y_arcsec_integrated
        #        x23 = cube2.offset_from_center_x_arcsec_integrated - cube3.offset_from_center_x_arcsec_integrated
        #        y23 = cube2.offset_from_center_y_arcsec_integrated - cube3.offset_from_center_y_arcsec_integrated
        #        x34 = cube3.offset_from_center_x_arcsec_integrated - cube4.offset_from_center_x_arcsec_integrated
        #        y34 = cube3.offset_from_center_y_arcsec_integrated - cube4.offset_from_center_y_arcsec_integrated
        #        x41 = cube4.offset_from_center_x_arcsec_integrated - cube1.offset_from_center_x_arcsec_integrated
        #        y41 = cube4.offset_from_center_y_arcsec_integrated - cube1.offset_from_center_y_arcsec_integrated
        print("  Using peak of the emission tracing all wavelengths to align cubes:")
        x12 = (
            cube2.offset_from_center_x_arcsec_tracing
            - cube1.offset_from_center_x_arcsec_tracing
        )
        y12 = (
            cube2.offset_from_center_y_arcsec_tracing
            - cube1.offset_from_center_y_arcsec_tracing
        )
        x23 = (
            cube3.offset_from_center_x_arcsec_tracing
            - cube2.offset_from_center_x_arcsec_tracing
        )
        y23 = (
            cube3.offset_from_center_y_arcsec_tracing
            - cube2.offset_from_center_y_arcsec_tracing
        )
        x34 = (
            cube4.offset_from_center_x_arcsec_tracing
            - cube3.offset_from_center_x_arcsec_tracing
        )
        y34 = (
            cube4.offset_from_center_y_arcsec_tracing
            - cube3.offset_from_center_y_arcsec_tracing
        )
        x41 = (
            cube1.offset_from_center_x_arcsec_tracing
            - cube4.offset_from_center_x_arcsec_tracing
        )
        y41 = (
            cube1.offset_from_center_y_arcsec_tracing
            - cube4.offset_from_center_y_arcsec_tracing
        )

    else:
        print("  Using offsets given by the user:")
        x12 = offsets[0]
        y12 = offsets[1]
        x23 = offsets[2]
        y23 = offsets[3]
        x34 = offsets[4]
        y34 = offsets[5]
        x41 = -(offsets[0] + offsets[2] + offsets[4])
        y41 = -(offsets[1] + offsets[3] + offsets[5])

    rss1.ALIGNED_RA_centre_deg = cube1.RA_centre_deg
    rss1.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg
    rss2.ALIGNED_RA_centre_deg = cube1.RA_centre_deg - x12 / 3600.0
    rss2.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg - y12 / 3600.0
    rss3.ALIGNED_RA_centre_deg = cube1.RA_centre_deg - (x12 + x23) / 3600.0
    rss3.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg - (y12 + y23) / 3600.0
    rss4.ALIGNED_RA_centre_deg = cube1.RA_centre_deg + x41 / 3600.0
    rss4.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg + y41 / 3600.0

    RA_centre_deg = rss1.ALIGNED_RA_centre_deg
    DEC_centre_deg = rss1.ALIGNED_DEC_centre_deg

    print("\n  Offsets (in arcsec):")
    print("  Offsets in x : ", x12, x23, x34, "      Total offset in x = ", x12 + x23 + x34 + x41)
    print("  Offsets in y : ", y12, y23, y34, "      Total offset in y = ", y12 + y23 + y34 + y41)

    print("\n>        New_RA_centre_deg       New_DEC_centre_deg       Diff respect Cube 1 (arcsec)")
    print("  Cube 1 : ", rss1.ALIGNED_RA_centre_deg, "     ", rss1.ALIGNED_DEC_centre_deg, "      0,0")
    print("  Cube 2 : ", rss2.ALIGNED_RA_centre_deg, "     ", rss2.ALIGNED_DEC_centre_deg, "    ", (
        rss2.ALIGNED_RA_centre_deg - rss1.ALIGNED_RA_centre_deg
    ) * 3600.0, (
        rss2.ALIGNED_DEC_centre_deg - rss1.ALIGNED_DEC_centre_deg
    ) * 3600.0)
    print("  Cube 3 : ", rss3.ALIGNED_RA_centre_deg, "     ", rss3.ALIGNED_DEC_centre_deg, "    ", (
        rss3.ALIGNED_RA_centre_deg - rss1.ALIGNED_RA_centre_deg
    ) * 3600.0, (
        rss3.ALIGNED_DEC_centre_deg - rss1.ALIGNED_DEC_centre_deg
    ) * 3600.0)
    print("  Cube 4 : ", rss4.ALIGNED_RA_centre_deg, "     ", rss4.ALIGNED_DEC_centre_deg, "    ", (
        rss4.ALIGNED_RA_centre_deg - rss1.ALIGNED_RA_centre_deg
    ) * 3600.0, (
        rss4.ALIGNED_DEC_centre_deg - rss1.ALIGNED_DEC_centre_deg
    ) * 3600.0)

    offsets_files = [
        [x12, y12],
        [x23, y23],
        [x34, y34],
    ]  # For keeping in the files with self.offsets_files
    #    RA_size_arcsec = 1.1*(RA_max - RA_min)*3600.
    #    DEC_size_arcsec = 1.1*(DEC_max - DEC_min)*3600.
    RA_size_arcsec = rss1.RA_segment + x12 + x23 + x34 + 2 * kernel_size_arcsec
    DEC_size_arcsec = rss1.DEC_segment + y12 + y23 + y34 + 2 * kernel_size_arcsec

    #    print "  RA_centre_deg , DEC_centre_deg   = ", RA_centre_deg, DEC_centre_deg
    print("  RA_size x DEC_size  = {:.2f} arcsec x {:.2f} arcsec".format(
        RA_size_arcsec, DEC_size_arcsec
    ))

    #    probando=raw_input("Continue?")

    cube1_aligned = Interpolated_cube(
        rss1,
        pixel_size_arcsec,
        kernel_size_arcsec,
        centre_deg=[RA_centre_deg, DEC_centre_deg],
        size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
        aligned_coor=True,
        flux_calibration=cube1.flux_calibration,
        offsets_files=offsets_files,
        offsets_files_position=1,
        plot=plot,
        ADR=ADR,
        warnings=warnings,
    )
    cube2_aligned = Interpolated_cube(
        rss2,
        pixel_size_arcsec,
        kernel_size_arcsec,
        centre_deg=[RA_centre_deg, DEC_centre_deg],
        size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
        aligned_coor=True,
        flux_calibration=cube2.flux_calibration,
        offsets_files=offsets_files,
        offsets_files_position=2,
        plot=plot,
        ADR=ADR,
        warnings=warnings,
    )
    cube3_aligned = Interpolated_cube(
        rss3,
        pixel_size_arcsec,
        kernel_size_arcsec,
        centre_deg=[RA_centre_deg, DEC_centre_deg],
        size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
        aligned_coor=True,
        flux_calibration=cube3.flux_calibration,
        offsets_files=offsets_files,
        offsets_files_position=3,
        plot=plot,
        ADR=ADR,
        warnings=warnings,
    )
    cube4_aligned = Interpolated_cube(
        rss4,
        pixel_size_arcsec,
        kernel_size_arcsec,
        centre_deg=[RA_centre_deg, DEC_centre_deg],
        size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
        aligned_coor=True,
        flux_calibration=cube3.flux_calibration,
        offsets_files=offsets_files,
        offsets_files_position=4,
        plot=plot,
        ADR=ADR,
        warnings=warnings,
    )

    print("\n> Checking offsets of ALIGNED cubes (in arcsec):")

    x12 = (
        cube1_aligned.offset_from_center_x_arcsec_tracing
        - cube2_aligned.offset_from_center_x_arcsec_tracing
    )
    y12 = (
        cube1_aligned.offset_from_center_y_arcsec_tracing
        - cube2_aligned.offset_from_center_y_arcsec_tracing
    )
    x23 = (
        cube2_aligned.offset_from_center_x_arcsec_tracing
        - cube3_aligned.offset_from_center_x_arcsec_tracing
    )
    y23 = (
        cube2_aligned.offset_from_center_y_arcsec_tracing
        - cube3_aligned.offset_from_center_y_arcsec_tracing
    )
    x34 = (
        cube3_aligned.offset_from_center_x_arcsec_tracing
        - cube4_aligned.offset_from_center_x_arcsec_tracing
    )
    y34 = (
        cube3_aligned.offset_from_center_y_arcsec_tracing
        - cube4_aligned.offset_from_center_y_arcsec_tracing
    )
    x41 = (
        cube4_aligned.offset_from_center_x_arcsec_tracing
        - cube1_aligned.offset_from_center_x_arcsec_tracing
    )
    y41 = (
        cube4_aligned.offset_from_center_y_arcsec_tracing
        - cube1_aligned.offset_from_center_y_arcsec_tracing
    )

    print("  Offsets in x : {:.3f}   {:.3f}   {:.3f}   {:.3f}    Total offset in x = {:.3f}".format(
        x12, x23, x34, x41, x12 + x23 + x34 + x41
    ))
    print("  Offsets in y : {:.3f}   {:.3f}   {:.3f}   {:.3f}    Total offset in y = {:.3f}".format(
        y12, y23, y34, y41, y12 + y23 + y34 + y41
    ))

    print("\n> Updated values for Alignment DONE")
    return cube1_aligned, cube2_aligned, cube3_aligned, cube4_aligned


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def align_n_cubes(
    rss_list,
    cube_list=[0],
    flux_calibration_list=[[0]],
    pixel_size_arcsec=0.3,
    kernel_size_arcsec=1.5,
    offsets=[1000],
    plot=False,
    ADR=False,
    warnings=False,
):  # TASK_align_n_cubes
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
    print("\n> Starting alignment procedure...")

    n_rss = len(rss_list)

    xx = [0]  # This will have 0, x12, x23, x34, ... xn1
    yy = [0]  # This will have 0, y12, y23, y34, ... yn1

    if np.nanmedian(flux_calibration_list[0]) == 0:
        flux_calibration_list[0] = [0]
        for i in range(1, n_rss):
            flux_calibration_list.append([0])

    if offsets[0] == 1000:
        print("\n  Using peak of the emission tracing all wavelengths to align cubes:")
        n_cubes = len(cube_list)
        if n_cubes != n_rss:
            print("\n\n\n ERROR: number of cubes and number of rss files don't match!")
            print("\n\n THIS IS GOING TO FAIL ! \n\n\n")

        for i in range(n_rss - 1):
            xx.append(
                cube_list[i + 1].offset_from_center_x_arcsec_tracing
                - cube_list[i].offset_from_center_x_arcsec_tracing
            )
            yy.append(
                cube_list[i + 1].offset_from_center_y_arcsec_tracing
                - cube_list[i].offset_from_center_y_arcsec_tracing
            )
        xx.append(
            cube_list[0].offset_from_center_x_arcsec_tracing
            - cube_list[-1].offset_from_center_x_arcsec_tracing
        )
        yy.append(
            cube_list[0].offset_from_center_y_arcsec_tracing
            - cube_list[-1].offset_from_center_y_arcsec_tracing
        )

    else:
        print("\n  Using offsets given by the user:")
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

    print("\n\n\n\n\n\n")
    print(list_RA_centre_deg, median_RA_centre_deg)
    print(list_DEC_centre_deg, median_DEC_centre_deg)
    print("\n\n\n\n\n\n")

    for i in range(n_rss):
        # print i, np.nansum(xx[1:i+1]) ,  np.nansum(yy[1:i+1])
        rss_list[i].ALIGNED_RA_centre_deg = (
            median_RA_centre_deg + np.nansum(xx[1: i + 1]) / 3600.0
        )  # CHANGE SIGN 26 Apr 2019    # ERA cube_list[0]
        rss_list[i].ALIGNED_DEC_centre_deg = (
            median_DEC_centre_deg - np.nansum(yy[1: i + 1]) / 3600.0
        )  # rss_list[0].DEC_centre_deg

        print(rss_list[i].RA_centre_deg, rss_list[i].ALIGNED_RA_centre_deg)
        print(rss_list[i].ALIGNED_DEC_centre_deg)

    RA_centre_deg = rss_list[0].ALIGNED_RA_centre_deg
    DEC_centre_deg = rss_list[0].ALIGNED_DEC_centre_deg

    print("  Offsets (in arcsec):        x             y                          ( EAST- / WEST+   NORTH- / SOUTH+) ")
    for i in range(1, len(xx) - 1):
        print("         {:2.0f} -> {:2.0f}           {:+5.3f}         {:+5.3f}".format(
            i, i + 1, xx[i], yy[i]
        ))
    print("         {:2.0f} ->  1           {:+5.3f}         {:+5.3f}".format(
        len(xx) - 1, xx[-1], yy[-1]
    ))
    print("           TOTAL:            {:5.3f}          {:5.3f}".format(
        np.nansum(xx), np.nansum(yy)
    ))

    print("\n         New_RA_centre_deg       New_DEC_centre_deg      Diff with respect Cube 1 [arcsec]")

    for i in range(0, n_rss):
        print("  Cube {:2.0f}:     {:5.8f}          {:5.8f}           {:5.3f}   ,  {:5.3f}   ".format(
            i + 1,
            rss_list[i].ALIGNED_RA_centre_deg,
            rss_list[i].ALIGNED_DEC_centre_deg,
            (rss_list[i].ALIGNED_RA_centre_deg - rss_list[0].ALIGNED_RA_centre_deg)
            * 3600.0,
            (rss_list[i].ALIGNED_DEC_centre_deg - rss_list[0].ALIGNED_DEC_centre_deg)
            * 3600.0,
        ))

    offsets_files = []
    for i in range(1, n_rss):  # For keeping in the files with self.offsets_files
        vector = [xx[i], yy[i]]
        offsets_files.append(vector)

    RA_size_arcsec = (
        rss_list[0].RA_segment + np.nansum(np.abs(xx[0:-1])) + 2 * kernel_size_arcsec
    )
    DEC_size_arcsec = (
        rss_list[0].DEC_segment + np.nansum(np.abs(yy[0:-1])) + 2 * kernel_size_arcsec
    )
    print("\n  RA_size x DEC_size  = {:.2f} arcsec x {:.2f} arcsec".format(
        RA_size_arcsec, DEC_size_arcsec
    ))

    cube_aligned_list = []
    for i in range(1, n_rss + 1):
        escribe = "cube" + np.str(i) + "_aligned"
        cube_aligned_list.append(escribe)

    for i in range(n_rss):
        print("\n> Creating aligned cube ", i + 1, " of a total of ", n_rss, "...")
        cube_aligned_list[i] = Interpolated_cube(
            rss_list[i],
            pixel_size_arcsec,
            kernel_size_arcsec,
            centre_deg=[RA_centre_deg, DEC_centre_deg],
            size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
            aligned_coor=True,
            flux_calibration=flux_calibration_list[i],
            offsets_files=offsets_files,
            offsets_files_position=i + 1,
            plot=plot,
            ADR=ADR,
            warnings=warnings,
        )

    print("\n> Checking offsets of ALIGNED cubes (in arcsec, everything should be close to 0):")
    print("  Offsets (in arcsec):        x             y                          ( EAST- / WEST+   NORTH- / SOUTH+) ")

    xxx = []
    yyy = []

    for i in range(1, n_rss):
        xxx.append(
            cube_aligned_list[i - 1].offset_from_center_x_arcsec_tracing
            - cube_aligned_list[i].offset_from_center_x_arcsec_tracing
        )
        yyy.append(
            cube_aligned_list[i - 1].offset_from_center_y_arcsec_tracing
            - cube_aligned_list[i].offset_from_center_y_arcsec_tracing
        )
    xxx.append(
        cube_aligned_list[-1].offset_from_center_x_arcsec_tracing
        - cube_aligned_list[0].offset_from_center_x_arcsec_tracing
    )
    yyy.append(
        cube_aligned_list[-1].offset_from_center_y_arcsec_tracing
        - cube_aligned_list[0].offset_from_center_y_arcsec_tracing
    )

    for i in range(1, len(xx) - 1):
        print("         {:2.0f} -> {:2.0f}           {:+5.3f}         {:+5.3f}".format(
            i, i + 1, xxx[i - 1], yyy[i - 1]
        ))
    print("         {:2.0f} ->  1           {:+5.3f}         {:+5.3f}".format(
        len(xxx), xxx[-1], yyy[-1]
    ))
    print("           TOTAL:            {:5.3f}          {:5.3f}".format(
        np.nansum(xxx), np.nansum(yyy)
    ))

    print("\n> Updated values for Alignment cubes DONE")
    return cube_aligned_list


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def save_fits_file(combined_cube, fits_file, description="", ADR=False):  # fcal=[0],
    """
    Routine to save a fits file

    Parameters
    ----------
    Combined cube:
        Combined cube
    Header:
        Header
    """
    fits_image_hdu = fits.PrimaryHDU(combined_cube.data)
    #    errors = combined_cube.data*0  ### TO BE DONE
    #    error_hdu = fits.ImageHDU(errors)

    # wavelength =  combined_cube.wavelength

    fits_image_hdu.header["HISTORY"] = "Combined datacube from KOALA Python pipeline"
    fits_image_hdu.header[
        "HISTORY"
    ] = "Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany et al."
    fits_image_hdu.header["HISTORY"] = version  # 'Version 0.10 - 12th February 2019'
    now = datetime.datetime.now()
    fits_image_hdu.header["HISTORY"] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    fits_image_hdu.header["DATE"] = now.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )  # '2002-09-16T18:52:44'   # /Date of FITS file creation

    fits_image_hdu.header["BITPIX"] = 16

    fits_image_hdu.header["ORIGIN"] = "AAO"  # / Originating Institution
    fits_image_hdu.header["TELESCOP"] = "Anglo-Australian Telescope"  # / Telescope Name
    fits_image_hdu.header["ALT_OBS"] = 1164  # / Altitude of observatory in metres
    fits_image_hdu.header["LAT_OBS"] = -31.27704  # / Observatory latitude in degrees
    fits_image_hdu.header["LONG_OBS"] = 149.0661  # / Observatory longitude in degrees

    fits_image_hdu.header["INSTRUME"] = "AAOMEGA-KOALA"  # / Instrument in use
    fits_image_hdu.header["GRATID"] = combined_cube.RSS.grating  # / Disperser ID
    if combined_cube.RSS.grating == "385R":
        SPECTID = "RD"
    if combined_cube.RSS.grating == "580V":
        SPECTID = "BD"
    if combined_cube.RSS.grating == "1000R":
        SPECTID = "RD"
    if combined_cube.RSS.grating == "1000I":
        SPECTID = "RD"
    fits_image_hdu.header["SPECTID"] = SPECTID  # / Spectrograph ID

    fits_image_hdu.header[
        "DICHROIC"
    ] = "X5700"  # / Dichroic name   ---> CHANGE if using X6700!!

    fits_image_hdu.header["OBJECT"] = combined_cube.object
    fits_image_hdu.header["TOTALEXP"] = combined_cube.total_exptime

    fits_image_hdu.header["NAXIS"] = 3  # / number of array dimensions
    fits_image_hdu.header["NAXIS1"] = combined_cube.data.shape[1]  # CHECK !!!!!!!
    fits_image_hdu.header["NAXIS2"] = combined_cube.data.shape[2]
    fits_image_hdu.header["NAXIS3"] = combined_cube.data.shape[0]

    # WCS
    fits_image_hdu.header["RADECSYS"] = "FK5"  # / FK5 reference system
    fits_image_hdu.header["EQUINOX"] = 2000  # / [yr] Equinox of equatorial coordinates
    fits_image_hdu.header["WCSAXES"] = 3  # / Number of coordinate axes

    fits_image_hdu.header["CRPIX1"] = (
        combined_cube.data.shape[1] / 2.0
    )  # / Pixel coordinate of reference point
    fits_image_hdu.header["CDELT1"] = (
        -combined_cube.pixel_size_arcsec / 3600.0
    )  # / Coordinate increment at reference point
    fits_image_hdu.header[
        "CTYPE1"
    ] = "RA--TAN"  # 'DEGREE'                               # / Coordinate type code
    fits_image_hdu.header[
        "CRVAL1"
    ] = combined_cube.RA_centre_deg  # / Coordinate value at reference point

    fits_image_hdu.header["CRPIX2"] = (
        combined_cube.data.shape[2] / 2.0
    )  # / Pixel coordinate of reference point
    fits_image_hdu.header["CDELT2"] = (
        combined_cube.pixel_size_arcsec / 3600.0
    )  # Coordinate increment at reference point
    fits_image_hdu.header[
        "CTYPE2"
    ] = "DEC--TAN"  # 'DEGREE'                               # / Coordinate type code
    fits_image_hdu.header[
        "CRVAL2"
    ] = combined_cube.DEC_centre_deg  # / Coordinate value at reference point

    fits_image_hdu.header["RAcen"] = combined_cube.RA_centre_deg
    fits_image_hdu.header["DECcen"] = combined_cube.DEC_centre_deg
    fits_image_hdu.header["PIXsize"] = combined_cube.pixel_size_arcsec
    fits_image_hdu.header["Ncols"] = combined_cube.data.shape[2]
    fits_image_hdu.header["Nrows"] = combined_cube.data.shape[1]
    fits_image_hdu.header["PA"] = combined_cube.PA

    # Wavelength calibration
    fits_image_hdu.header["CTYPE3"] = "Wavelength"  # / Label for axis 3
    fits_image_hdu.header["CUNIT3"] = "Angstroms"  # / Units for axis 3
    fits_image_hdu.header["CRVAL3"] = combined_cube.CRVAL1_CDELT1_CRPIX1[
        0
    ]  # 7.692370611909E+03  / Co-ordinate value of axis 3
    fits_image_hdu.header["CDELT3"] = combined_cube.CRVAL1_CDELT1_CRPIX1[
        1
    ]  # 1.575182431607E+00
    fits_image_hdu.header["CRPIX3"] = combined_cube.CRVAL1_CDELT1_CRPIX1[
        2
    ]  # 1024. / Reference pixel along axis 3

    fits_image_hdu.header["COFILES"] = (
        len(combined_cube.offsets_files) + 1
    )  # Number of combined files
    offsets_text = " "
    for i in range(len(combined_cube.offsets_files)):
        if i != 0:
            offsets_text = offsets_text + "  ,  "
        offsets_text = (
            offsets_text
            + np.str(np.around(combined_cube.offsets_files[i][0], 3))
            + " "
            + np.str(np.around(combined_cube.offsets_files[i][1], 3))
        )
    fits_image_hdu.header["OFFSETS"] = offsets_text  # Offsets

    fits_image_hdu.header["ADRCOR"] = np.str(ADR)

    if np.nanmedian(combined_cube.data) > 1:
        fits_image_hdu.header["FCAL"] = "False"
        fits_image_hdu.header["F_UNITS"] = "Counts"
        # flux_correction_hdu = fits.ImageHDU(0*wavelength)
    else:
        # flux_correction = fcal
        # flux_correction_hdu = fits.ImageHDU(flux_correction)
        fits_image_hdu.header["FCAL"] = "True"
        fits_image_hdu.header["F_UNITS"] = "erg s-1 cm-2 A-1"

    if description == "":
        description = combined_cube.description
    fits_image_hdu.header["DESCRIP"] = description

    for file in range(len(combined_cube.rss_list)):
        fits_image_hdu.header["HISTORY"] = (
            "RSS file " + np.str(file + 1) + ":" + combined_cube.rss_list[file]
        )

    #    hdu_list = fits.HDUList([fits_image_hdu, error_hdu])
    #    hdu_list = fits.HDUList([fits_image_hdu, wavelengths_hdu, flux_correction_hdu])
    hdu_list = fits.HDUList([fits_image_hdu])  # , flux_correction_hdu])

    hdu_list.writeto(fits_file, overwrite=True)
    print("\n> Combined datacube saved to file:", fits_file)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def save_rss_fits(
    rss, data=[[0], [0]], fits_file="RSS_rss.fits", description=""
):  # fcal=[0],     # TASK_save_rss_fits
    """
    Routine to save RSS data as fits

    Parameters
    ----------
    rss is the rss
    description = if you want to add a description
    """
    if np.nanmedian(data[0]) == 0:
        data = rss.intensity_corrected
        print("\n> Using rss.intensity_corrected of given RSS file to create fits file...")
    else:
        if len(np.array(data).shape) != 2:
            print("\n> The data provided are NOT valid, as they have a shape", data.shape)
            print("  Using rss.intensity_corrected instead to create a RSS fits file !")
            data = rss.intensity_corrected
        else:
            print("\n> Using the data provided + structure of given RSS file to create fits file...")
    fits_image_hdu = fits.PrimaryHDU(data)

    fits_image_hdu.header["HISTORY"] = "RSS from KOALA Python pipeline"
    fits_image_hdu.header[
        "HISTORY"
    ] = "Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany et al."
    fits_image_hdu.header["HISTORY"] = version  # 'Version 0.10 - 12th February 2019'
    now = datetime.datetime.now()
    fits_image_hdu.header["HISTORY"] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    fits_image_hdu.header["DATE"] = now.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )  # '2002-09-16T18:52:44'   # /Date of FITS file creation

    fits_image_hdu.header["BITPIX"] = 16

    fits_image_hdu.header["ORIGIN"] = "AAO"  # / Originating Institution
    fits_image_hdu.header["TELESCOP"] = "Anglo-Australian Telescope"  # / Telescope Name
    fits_image_hdu.header["ALT_OBS"] = 1164  # / Altitude of observatory in metres
    fits_image_hdu.header["LAT_OBS"] = -31.27704  # / Observatory latitude in degrees
    fits_image_hdu.header["LONG_OBS"] = 149.0661  # / Observatory longitude in degrees

    fits_image_hdu.header["INSTRUME"] = "AAOMEGA-KOALA"  # / Instrument in use
    fits_image_hdu.header["GRATID"] = rss.grating  # / Disperser ID
    if rss.grating == "385R":
        SPECTID = "RD"
    if rss.grating == "580V":
        SPECTID = "BD"
    if rss.grating == "1000R":
        SPECTID = "RD"
    if rss.grating == "1000I":
        SPECTID = "RD"
    fits_image_hdu.header["SPECTID"] = SPECTID  # / Spectrograph ID

    fits_image_hdu.header[
        "DICHROIC"
    ] = "X5700"  # / Dichroic name   ---> CHANGE if using X6700!!

    fits_image_hdu.header["OBJECT"] = rss.object
    fits_image_hdu.header["EXPOSED"] = rss.exptime
    fits_image_hdu.header["ZDSTART"] = rss.ZDSTART
    fits_image_hdu.header["ZDEND"] = rss.ZDEND

    fits_image_hdu.header["NAXIS"] = 2  # / number of array dimensions
    fits_image_hdu.header["NAXIS1"] = rss.intensity_corrected.shape[0]
    fits_image_hdu.header["NAXIS2"] = rss.intensity_corrected.shape[1]

    fits_image_hdu.header["RAcen"] = rss.RA_centre_deg
    fits_image_hdu.header["DECcen"] = rss.DEC_centre_deg
    fits_image_hdu.header["TEL_PA"] = rss.PA

    fits_image_hdu.header["CTYPE2"] = "Fibre number"  # / Label for axis 2
    fits_image_hdu.header["CUNIT2"] = " "  # / Units for axis 2
    fits_image_hdu.header["CTYPE1"] = "Wavelength"  # / Label for axis 2
    fits_image_hdu.header["CUNIT1"] = "Angstroms"  # / Units for axis 2

    fits_image_hdu.header["CRVAL1"] = rss.CRVAL1_CDELT1_CRPIX1[
        0
    ]  # / Co-ordinate value of axis 2
    fits_image_hdu.header["CDELT1"] = rss.CRVAL1_CDELT1_CRPIX1[1]  #
    fits_image_hdu.header["CRPIX1"] = rss.CRVAL1_CDELT1_CRPIX1[
        2
    ]  # 1024. / Reference pixel along axis 2
    fits_image_hdu.header[
        "CRVAL2"
    ] = 5.000000000000e-01  # / Co-ordinate value of axis 2
    fits_image_hdu.header[
        "CDELT2"
    ] = 1.000000000000e00  # / Co-ordinate increment along axis 2
    fits_image_hdu.header[
        "CRPIX2"
    ] = 1.000000000000e00  # / Reference pixel along axis 2

    if description == "":
        description = rss.description
    fits_image_hdu.header["DESCRIP"] = description

    # TO BE DONE
    errors = [0]  # TO BE DONE
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

    col1 = fits.Column(name="Fibre", format="I", array=np.array(header2_new_fibre))
    col2 = fits.Column(name="Status", format="I", array=np.array(header2_good_fibre))
    col3 = fits.Column(name="Ones", format="I", array=np.array(header2_good_fibre))
    col4 = fits.Column(name="Wavelengths", format="I", array=np.array(header2_2048))
    col5 = fits.Column(name="Zeros", format="I", array=np.array(header2_0))
    col6 = fits.Column(name="Delta_RA", format="D", array=np.array(header2_delta_RA))
    col7 = fits.Column(name="Delta_Dec", format="D", array=np.array(header2_delta_DEC))
    col8 = fits.Column(
        name="Fibre_OLD", format="I", array=np.array(header2_original_fibre)
    )

    cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8])
    header2_hdu = fits.BinTableHDU.from_columns(cols)

    header2_hdu.header["CENRA"] = old_div(rss.RA_centre_deg, (
        old_div(180, np.pi)
    ))  # Must be in radians
    header2_hdu.header["CENDEC"] = old_div(rss.DEC_centre_deg, (old_div(180, np.pi)))

    hdu_list = fits.HDUList(
        [fits_image_hdu, error_hdu, header2_hdu]
    )  # hdu_list = fits.HDUList([fits_image_hdu, wavelengths_hdu, flux_correction_hdu])

    hdu_list.writeto(fits_file, overwrite=True)
    print("  RSS data saved to file ", fits_file)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def save_bluered_fits_file(
    blue_cube,
    red_cube,
    fits_file,
    fcalb=[0],
    fcalr=[0],
    ADR=False,
    objeto="",
    description="",
    trimb=[0],
    trimr=[0],
):
    """
    Routine combine blue + red files and save result in a fits file fits file

    Parameters
    ----------
    Combined cube:
        Combined cube
    Header:
        Header
    """

    # Prepare the red+blue datacube
    print("\n> Combining blue + red datacubes...")

    if trimb[0] == 0:
        lb = blue_cube.wavelength
        b = blue_cube.data
    else:
        print("  Trimming blue cube in range [{},{}]".format(trimb[0], trimb[1]))
        index_min = np.searchsorted(blue_cube.wavelength, trimb[0])
        index_max = np.searchsorted(blue_cube.wavelength, trimb[1]) + 1
        lb = blue_cube.wavelength[index_min:index_max]
        b = blue_cube.data[index_min:index_max]
        fcalb = fcalb[index_min:index_max]

    if trimr[0] == 0:
        lr = red_cube.wavelength
        r = red_cube.data
    else:
        print("  Trimming red cube in range [{},{}]".format(trimr[0], trimr[1]))
        index_min = np.searchsorted(red_cube.wavelength, trimr[0])
        index_max = np.searchsorted(red_cube.wavelength, trimr[1]) + 1
        lr = red_cube.wavelength[index_min:index_max]
        r = red_cube.data[index_min:index_max]
        fcalr = fcalr[index_min:index_max]

    l = np.concatenate((lb, lr), axis=0)
    blue_red_datacube = np.concatenate((b, r), axis=0)

    if fcalb[0] == 0:
        print("  No absolute flux calibration included")
    else:
        flux_calibration = np.concatenate((fcalb, fcalr), axis=0)

    if objeto == "":
        description = "UNKNOWN OBJECT"

    fits_image_hdu = fits.PrimaryHDU(blue_red_datacube)
    #    errors = combined_cube.data*0  ### TO BE DONE
    #    error_hdu = fits.ImageHDU(errors)

    wavelengths_hdu = fits.ImageHDU(l)

    fits_image_hdu.header["ORIGIN"] = "Combined datacube from KOALA Python scripts"

    fits_image_hdu.header["BITPIX"] = 16
    fits_image_hdu.header["NAXIS"] = 3
    fits_image_hdu.header["NAXIS1"] = len(l)
    fits_image_hdu.header["NAXIS2"] = blue_red_datacube.shape[1]  # CHECK !!!!!!!
    fits_image_hdu.header["NAXIS2"] = blue_red_datacube.shape[2]

    fits_image_hdu.header["OBJECT"] = objeto
    fits_image_hdu.header["RAcen"] = blue_cube.RA_centre_deg
    fits_image_hdu.header["DECcen"] = blue_cube.DEC_centre_deg
    fits_image_hdu.header["PIXsize"] = blue_cube.pixel_size_arcsec
    fits_image_hdu.header["Ncols"] = blue_cube.data.shape[2]
    fits_image_hdu.header["Nrows"] = blue_cube.data.shape[1]
    fits_image_hdu.header["PA"] = blue_cube.PA
    #    fits_image_hdu.header["CTYPE1"] = 'LINEAR  '
    #    fits_image_hdu.header["CRVAL1"] = wavelength[0]
    #    fits_image_hdu.header["CRPIX1"] = 1.
    #    fits_image_hdu.header["CDELT1"] = (wavelength[-1]-wavelength[0])/len(wavelength)
    #    fits_image_hdu.header["CD1_1"]  = (wavelength[-1]-wavelength[0])/len(wavelength)
    #    fits_image_hdu.header["LTM1_1"] = 1.

    fits_image_hdu.header[
        "COFILES"
    ] = blue_cube.number_of_combined_files  # Number of combined files
    fits_image_hdu.header["OFFSETS"] = blue_cube.offsets_files  # Offsets

    fits_image_hdu.header["ADRCOR"] = np.str(ADR)

    if fcalb[0] == 0:
        fits_image_hdu.header["FCAL"] = "False"
        flux_correction_hdu = fits.ImageHDU(0 * l)
    else:
        flux_correction = flux_calibration
        flux_correction_hdu = fits.ImageHDU(flux_correction)
        fits_image_hdu.header["FCAL"] = "True"

    if description == "":
        description = flux_calibration.description
    fits_image_hdu.header["DESCRIP"] = description

    #    hdu_list = fits.HDUList([fits_image_hdu, error_hdu])
    hdu_list = fits.HDUList([fits_image_hdu, wavelengths_hdu, flux_correction_hdu])
    hdu_list.writeto(fits_file, overwrite=True)
    print("\n> Combined datacube saved to file ", fits_file)


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
# CUBE CLASS   (ANGEL + BEN)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class CUBE(RSS, Interpolated_cube):
    """
    This class reads the FITS files with COMBINED datacubes.

    Routines included:

    - cube.map_wavelength(wavelength, contours=True)\n
    - cube.plot_spectrum_cube(x,y, fcal=True)

    """

    # -----------------------------------------------------------------------------
    def __init__(self, filename):

        # Create RSS object
        super(CUBE, self).__init__()

        print("\n> Reading combined datacube", '"' + filename + '"', "...")
        RSS_fits_file = fits.open(filename)  # Open file

        # General info:
        self.object = RSS_fits_file[0].header["OBJECT"]
        #        self.description = self.object + ' - ' + filename
        self.description = RSS_fits_file[0].header[
            "DESCRIP"
        ]  # NOTE: it was originally "DEF"
        self.RA_centre_deg = RSS_fits_file[0].header["RAcen"]
        self.DEC_centre_deg = RSS_fits_file[0].header["DECcen"]
        self.PA = RSS_fits_file[0].header["PA"]
        self.wavelength = RSS_fits_file[1].data
        self.flux_calibration = RSS_fits_file[2].data
        self.n_wave = len(self.wavelength)
        self.data = RSS_fits_file[0].data
        self.wave_resolution = old_div((self.wavelength[-1] - self.wavelength[0]), self.n_wave)

        self.n_cols = RSS_fits_file[0].header["Ncols"]
        self.n_rows = RSS_fits_file[0].header["Nrows"]
        self.pixel_size_arcsec = RSS_fits_file[0].header["PIXsize"]
        self.flux_calibrated = RSS_fits_file[0].header["FCAL"]

        self.number_of_combined_files = RSS_fits_file[0].header["COFILES"]
        self.offsets_files = RSS_fits_file[0].header["OFFSETS"]

        print("\n  Object         = ", self.object)
        print("  Description    = ", self.description)
        print("  Centre:  RA    = ", self.RA_centre_deg, "Deg")
        print("          DEC    =", self.DEC_centre_deg, "Deg")
        print("  PA             = ", self.PA, "Deg")
        print("  Size [pix]     = ", self.n_rows, " x ", self.n_cols)
        print("  Size [arcsec]  = ", self.n_rows * self.pixel_size_arcsec, " x ", self.n_cols * self.pixel_size_arcsec)
        print("  Pix size       = ", self.pixel_size_arcsec, " arcsec")
        print("  Files combined = ", self.number_of_combined_files)
        print("  Offsets used   = ", self.offsets_files)

        print("  Wave Range     = [", self.wavelength[0], ",", self.wavelength[-1], "]")
        print("  Wave Resol.    = ", self.wave_resolution, " A/pix")
        print("  Flux Cal.      = ", self.flux_calibrated)

        print("\n> Use these parameters for acceding the data :\n")
        print("  cube.wavelength       : Array with wavelengths")
        print("  cube.data[w,x,y]      : Flux of the w wavelength in spaxel (x,y)")

        if self.flux_calibrated:
            print("  cube.flux_calibration : Flux calibration per wavelength [ 1 / (1E-16 * erg/cm**2/s/A) ] ")
        print("\n> Cube readed! ")

    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    def map_wavelength(
        self,
        wavelength,
        cmap="fuego",
        fig_size=10,
        norm=colors.PowerNorm(gamma=1.0 / 4.0),
        save_file="",
        contours=True,
        fcal=False,
    ):
        """
        Plot map at a particular wavelength.

        Parameters
        ----------
        wavelength: float
          wavelength to be mapped.
        norm:
          Colour scale, default = colors.PowerNorm(gamma=1./4.)\n
          Log scale: norm=colors.LogNorm() \n
          Lineal scale: norm=colors.Normalize().
        cmap:
            Color map used, default cmap="fuego"\n
            Weight: cmap = "gist_gray" \n
            Velocities: cmap="seismic".\n
            Try also "inferno",
        save_file:
            (Optional) Save plot in file "file.extension"

        Example
        -------
        >>> cube.map_wavelength(6820, contours=True, cmap="seismic")
        """
        if fcal:
            interpolated_map = self.data[np.searchsorted(self.wavelength, wavelength)]
        else:
            interpolated_map = self.data[np.searchsorted(self.wavelength, wavelength)]

        title = "{} - {} $\AA$".format(self.description, wavelength)

        self.plot_map(
            interpolated_map,
            cmap=cmap,
            fig_size=fig_size,
            norm=norm,
            contours=contours,
            save_file=save_file,
            title=title,
            fcal=fcal,
        )  # CHECK

    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    def plot_map(
        self,
        mapa,
        cmap="fuego",
        fig_size=10,
        norm=colors.PowerNorm(gamma=1.0 / 4.0),
        save_file="",
        contours=True,
        title="",
        vmin=0,
        vmax=1000,
        fcal=False,
        log=False,
        clabel=False,
        barlabel="",
    ):
        """
        Plot a given map.

        Parameters
        ----------
        wavelength: float
          wavelength to be mapped.
        norm:
          Colour scale, default = colors.PowerNorm(gamma=1./4.)\n
          Log scale: norm=colors.LogNorm() \n
          Lineal scale: norm=colors.Normalize().
        cmap:
            Color map used, default cmap="fuego"\n
            Weight: cmap = "gist_gray" \n
            Velocities: cmap="seismic".\n
            Try also "inferno",
        save_file:
            (Optional) Save plot in file "file.extension"

        Example
        -------
        >>> cube.plot_map(mapa, contours=True, cmap="seismic")
        """

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        if log:
            cax = ax.imshow(
                mapa,
                origin="lower",
                interpolation="none",
                norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                cmap=cmap,
                extent=(
                    -0.5 * self.n_cols * self.pixel_size_arcsec,
                    0.5 * self.n_cols * self.pixel_size_arcsec,
                    -0.5 * self.n_rows * self.pixel_size_arcsec,
                    0.5 * self.n_rows * self.pixel_size_arcsec,
                ),
            )
            print("Map in log scale")
        else:
            cax = ax.imshow(
                mapa,
                origin="lower",
                interpolation="none",
                norm=norm,
                cmap=cmap,
                extent=(
                    -0.5 * self.n_cols * self.pixel_size_arcsec,
                    0.5 * self.n_cols * self.pixel_size_arcsec,
                    -0.5 * self.n_rows * self.pixel_size_arcsec,
                    0.5 * self.n_rows * self.pixel_size_arcsec,
                ),
                vmin=vmin,
                vmax=vmax,
            )

        if contours:
            CS = plt.contour(
                mapa,
                extent=(
                    -0.5 * self.n_cols * self.pixel_size_arcsec,
                    0.5 * self.n_cols * self.pixel_size_arcsec,
                    -0.5 * self.n_rows * self.pixel_size_arcsec,
                    0.5 * self.n_rows * self.pixel_size_arcsec,
                ),
            )
            if clabel:
                plt.clabel(CS, inline=1, fontsize=10)

        ax.set_title(title, fontsize=fig_size * 1.3)
        plt.tick_params(labelsize=fig_size)
        plt.xlabel("$\Delta$ RA [arcsec]", fontsize=fig_size * 1.2)
        plt.ylabel("$\Delta$ DEC [arcsec]", fontsize=fig_size * 1.2)
        #        plt.legend(loc='upper right', frameon=False)
        plt.minorticks_on()
        plt.grid(which="both", color="green")
        plt.gca().invert_xaxis()

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar = fig.colorbar(sm, fraction=0.0499, pad=0.02)
        #        cbar = fig.colorbar(cax, fraction=0.0490, pad=0.04, norm=colors.Normalize(clip=False))

        if barlabel == "":
            if fcal:
                barlabel = str("Integrated Flux [10$^{-16}$ erg s$^{-1}$ cm$^{-2}$]")
            else:
                barlabel = str("Integrated Flux [Arbitrary units]")
        #        if fcal:
        #            cbar.set_label(str("Integrated Flux [10$^{-16}$ erg s$^{-1}$ cm$^{-2}$]"), rotation=270, labelpad=40, fontsize=fig_size*1.2)
        #        else:
        #            cbar.set_label(str("Integrated Flux [Arbitrary units]"), rotation=270, labelpad=40, fontsize=fig_size*1.2)
        cbar.set_label(barlabel, rotation=270, labelpad=20, fontsize=fig_size * 1.2)

        #        cbar.ax.set_yticklabels(['< -1', '0', '> 1'])# vertically oriented colorbar
        #        cbar.set_ticks([1.5,2,3,4,5,6], update_ticks=True)
        #        cbar.set_ticklabels([1.5,2,3,4,5,6])

        if save_file == "":
            plt.show()
        else:
            plt.savefig(save_file)
        plt.close()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    #
    #   BEN ROUTINES
    #
    #
    def subtractContinuum(self, spectrum):
        """
        Subtract the median value from each intensity in a provided spectrum.

        Parameters
        ----------
        spectrum:
            The list of intensities.
        """

        med = np.nanmedian(spectrum)
        for i in range(len(spectrum)):
            spectrum[i] = spectrum[i] - med
            if spectrum[i] < 0:
                spectrum[i] = 0
        return spectrum

    def plot_spectrum_cube_ben(
        self,
        x,
        y,
        lmin=0,
        lmax=0,
        fmin=1e-30,
        fmax=1e30,
        fig_size=10,
        save_file="",
        fcal=False,
    ):
        """
        Plot spectrum of a particular spaxel.

        Parameters
        ----------
        x, y:
            coordenates of spaxel to show spectrum.
        lmin, lmax:
            The range of wavelengths to plot. Default is whole spectrum.
        fmin, fmax:
            Plot spectrum in flux range [fmin, fmax]
        fcal:
            Use flux calibration, default fcal=False.\n
            If fcal=True, cube.flux_calibration is used.
        save_file:
            (Optional) Save plot in file "file.extension"
        fig_size:
            Size of the figure (in x-axis), default: fig_size=10

        Example
        -------
        >>> cube.plot_spectrum_cube_ben(20, 20, fcal=True)
        """

        # Define x and y axis to plot
        newWave = []
        newSpectrum = []

        if fcal == False:
            spectrum = self.data[:, x, y]
            ylabel = "Flux [relative units]"
        else:
            spectrum = old_div(old_div(self.data[:, x, y], self.flux_calibration), 1e16)
            # ylabel="Flux [ 10$^{-16}$ * erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]"
            ylabel = "Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]"
        # Remove NaN values from spectrum and replace them with zero.
        spectrum = np.nan_to_num(spectrum)
        # Subtract continuum from spectrum
        subSpectrum = self.subtractContinuum(spectrum)

        if fmin == 1e-30:
            fmin = np.nanmin(spectrum)
        if fmax == 1e30:
            fmax = np.nanmax(spectrum)

        # Since I can't define the correct default startpoint/endpoint within the
        # function arguments section, I set them here.
        if lmin == 0:
            lmin = self.wavelength[0]
        if lmax == 0:
            lmax = self.wavelength[-1]

        # Create a new list of wavelengths to plot based on the provided
        # wavelength startpoint and endpoint.
        for i in range(len(self.wavelength)):
            if self.wavelength[i] >= lmin and self.wavelength[i] <= lmax:
                newWave.append(self.wavelength[i])
                newSpectrum.append(subSpectrum[i])

        plt.figure(figsize=(fig_size, fig_size / 2.5))
        plt.plot(newWave, newSpectrum)
        plt.ylim([fmin, fmax])
        plt.xlim([lmin, lmax])
        plt.minorticks_on()

        title = "Spectrum of spaxel ({} , {}) in {}".format(x, y, self.description)
        plt.title(title, fontsize=fig_size * 1.2)
        plt.tick_params(labelsize=fig_size * 0.8)
        plt.xlabel("Wavelength [$\AA$]", fontsize=fig_size * 1)
        plt.ylabel(ylabel, fontsize=fig_size * 1)

        if save_file == "":
            plt.show()
        else:
            plt.savefig(save_file)
        plt.close()

    def calculateRatio(self, x, y, aStart, aEnd, bStart, bEnd, fcal=False):
        """
        Given two wavelengths ranges, find the peak intensities and calculate the ratio
        between them.

        Parameters
        ----------
        x, y:
            The spaxel we are interested in.
        aStart, aEnd:
            The startpoint and endpoint of the range that the first emission line
            will fall in.
        bStart, bEnd:
            The startpoint and endpoint of the range that the second emission line
            will fall in.
        """

        aFirstIndex = np.searchsorted(self.wavelength, aStart)
        aLastIndex = np.searchsorted(self.wavelength, aEnd)
        bFirstIndex = np.searchsorted(self.wavelength, bStart)
        bLastIndex = np.searchsorted(self.wavelength, bEnd)
        if fcal == False:
            spectrum = self.data[:, x, y]
        else:
            spectrum = old_div(old_div(self.data[:, x, y], self.flux_calibration), 1e16)
            spectrum = np.nan_to_num(spectrum)
        subSpectrum = self.subtractContinuum(spectrum)
        aValues = []
        tempIndex = aFirstIndex
        while tempIndex <= aLastIndex:
            aValues.append(subSpectrum[tempIndex])
            tempIndex = tempIndex + 1
        aMax = np.nanmax(aValues)

        bValues = []
        tempIndex = bFirstIndex
        while tempIndex <= bLastIndex:
            bValues.append(subSpectrum[tempIndex])
            tempIndex = tempIndex + 1
        bMax = np.nanmax(bValues)

        return old_div(aMax, bMax)

    def createRatioMap(self, aStart, aEnd, bStart, bEnd, fcal=False):
        xLength = len(self.data[0, :, 0])
        yLength = len(self.data[0, 0, :])
        aFirstIndex = np.searchsorted(self.wavelength, aStart)
        aLastIndex = np.searchsorted(self.wavelength, aEnd)
        bFirstIndex = np.searchsorted(self.wavelength, bStart)
        bLastIndex = np.searchsorted(self.wavelength, bEnd)
        ratioMap = [[i for i in range(yLength)] for j in range(xLength)]
        for y in range(yLength):
            print("Column " + str(y))
            for x in range(xLength):
                if fcal == False:
                    spectrum = self.data[:, x, y]
                else:
                    spectrum = old_div(old_div(self.data[:, x, y], self.flux_calibration), 1e16)
                spectrum = np.nan_to_num(spectrum)
                subSpectrum = self.subtractContinuum(spectrum)
                subAvg = np.average(subSpectrum)

                aValues = []
                tempIndex = aFirstIndex
                while tempIndex <= aLastIndex:
                    aValues.append(subSpectrum[tempIndex])
                    tempIndex = tempIndex + 1
                aMax = np.nanmax(aValues)

                bValues = []
                tempIndex = bFirstIndex
                while tempIndex <= bLastIndex:
                    bValues.append(subSpectrum[tempIndex])
                    tempIndex = tempIndex + 1
                bMax = np.nanmax(bValues)

                if aMax > subAvg and bMax > subAvg:
                    ratio = old_div(aMax, bMax)
                else:
                    ratio = 0
                ratioMap[x][y] = ratio

        return ratioMap


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#
# Extra tools for analysis, Angel 21st October 2017

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
        if formato[i] == "i":
            datos[i] = np.loadtxt(
                fichero, skiprows=0, unpack=True, usecols=[i], dtype=int
            )
        if formato[i] == "s":
            datos[i] = np.loadtxt(
                fichero, skiprows=0, unpack=True, usecols=[i], dtype=str
            )
        if formato[i] == "f":
            datos[i] = np.loadtxt(
                fichero, skiprows=0, unpack=True, usecols=[i], dtype=float
            )
    return datos


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def array_to_text_file(data, filename="array.dat"):
    """
    Write array into a text file.

    Parameters
    ----------
    data: float
        flux per wavelength
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
    print("\n> Array saved in text file", filename, " !!")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def spectrum_to_text_file(wavelength, flux, filename="spectrum.txt"):
    """
    Write given 1D spectrum into a text file.

    Parameters
    ----------
    wavelength: float
        wavelength.
    flux: float
        flux per wavelength
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
    print("\n> Spectrum saved in text file", filename, " !!")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def spectrum_to_fits_file(
    wavelength,
    flux,
    filename="spectrum.fits",
    name="spectrum",
    exptime=1,
    CRVAL1_CDELT1_CRPIX1=[0, 0, 0],
):
    """
    Routine to save a given 1D spectrum into a fits file.

    If CRVAL1_CDELT1_CRPIX1 it not given, it assumes a LINEAR dispersion,
    with Delta_pix = (wavelength[-1]-wavelength[0])/(len(wavelength)-1).

    Parameters
    ----------
    wavelength: float
        wavelength.
    flux: float
        flux per wavelength
    filename: string (default = "spectrum.fits")
        name of the fits file where the data will be written.
    Example
    -------
    >>> spectrum_to_fits_file(wavelength, spectrum, filename="fantastic_spectrum.fits",
                              exptime=600,name="POX 4")
    """
    hdu = fits.PrimaryHDU()
    hdu.data = flux
    hdu.header["ORIGIN"] = "Data from KOALA Python scripts"
    # Wavelength calibration
    hdu.header["NAXIS"] = 1
    hdu.header["NAXIS1"] = len(wavelength)
    hdu.header["CTYPE1"] = "Wavelength"
    hdu.header["CUNIT1"] = "Angstroms"
    if CRVAL1_CDELT1_CRPIX1[0] == 0:
        hdu.header["CRVAL1"] = wavelength[0]
        hdu.header["CRPIX1"] = 1.0
        hdu.header["CDELT1"] = old_div((wavelength[-1] - wavelength[0]), (len(wavelength) - 1))
    else:
        hdu.header["CRVAL1"] = CRVAL1_CDELT1_CRPIX1[
            0
        ]  # 7.692370611909E+03  / Co-ordinate value of axis 1
        hdu.header["CDELT1"] = CRVAL1_CDELT1_CRPIX1[1]  # 1.575182431607E+00
        hdu.header["CRPIX1"] = CRVAL1_CDELT1_CRPIX1[
            2
        ]  # 1024. / Reference pixel along axis 1
    # Extra info
    hdu.header["OBJECT"] = name
    hdu.header["TOTALEXP"] = exptime
    hdu.header["HISTORY"] = "Spectrum derived using the KOALA Python pipeline"
    hdu.header[
        "HISTORY"
    ] = "Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany et al."
    hdu.header["HISTORY"] = version
    now = datetime.datetime.now()
    hdu.header["HISTORY"] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    hdu.header["DATE"] = now.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )  # '2002-09-16T18:52:44'   # /Date of FITS file creation

    hdu.writeto(filename, overwrite=True)
    print("\n> Spectrum saved in fits file", filename, " !!")
    if name == "spectrum":
        print("  No name given to the spectrum, named 'spectrum'.")
    if exptime == 1:
        print("  No exposition time given, assumed exptime = 1")
    if CRVAL1_CDELT1_CRPIX1[0] == 0:
        print("  CRVAL1_CDELT1_CRPIX1 values not given, using ", wavelength[0], "1", old_div((
            wavelength[-1] - wavelength[0]
        ), (len(wavelength) - 1)))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def gauss(x, x0, y0, sigma):
    p = [x0, y0, sigma]
    return p[1] * np.exp(-0.5 * (old_div((x - p[0]), p[2])) ** 2)


def gauss_fix_x0(x, x0, y0, sigma):
    """
    A Gaussian of fixed location (x0)

    Args:
        x (array): A list of x locations to make the Gaussian at
        x0 (float): Location of the Gaussian
        y0 (float): Amplitude
        sigma (float): Gaussian width
    """
    p = [y0, sigma]
    return p[0] * np.exp(-0.5 * (old_div((x - x0), p[1])) ** 2)


def gauss_flux(y0, sigma):  # THIS DOES NOT WORK...
    return y0 * sigma * np.sqrt(2 * np.pi)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def substract_given_gaussian(
    wavelength,
    spectrum,
    centre,
    peak=0,
    sigma=0,
    flux=0,
    search_peak=False,
    allow_absorptions=False,
    lowlow=20,
    lowhigh=10,
    highlow=10,
    highhigh=20,
    lmin=0,
    lmax=0,
    fmin=0,
    fmax=0,
    plot=True,
    fcal=False,
    verbose=True,
):
    """
    Substract a give Gaussian to a spectrum after fitting the continuum.
    """
    # plot = True
    # verbose = True

    # Check that we have the numbers!
    if peak != 0 and sigma != 0:
        do_it = True

    if peak == 0 and flux != 0 and sigma != 0:
        # flux = peak * sigma * np.sqrt(2*np.pi)
        peak = old_div(flux, (sigma * np.sqrt(2 * np.pi)))
        do_it = True

    if sigma == 0 and flux != 0 and peak != 0:
        # flux = peak * sigma * np.sqrt(2*np.pi)
        sigma = old_div(flux, (peak * np.sqrt(2 * np.pi)))
        do_it = True

    if flux == 0 and sigma != 0 and peak != 0:
        flux = peak * sigma * np.sqrt(2 * np.pi)
        do_it = True

    if sigma != 0 and search_peak == True:
        do_it = True

    if do_it == False:
        print("> Error! We need data to proceed! Give at least two of [peak, sigma, flux], or sigma and force peak to f[centre]")
    else:
        # Setup wavelength limits
        if lmin == 0:
            lmin = centre - 65.0  # By default, +-65 A with respect to line
        if lmax == 0:
            lmax = centre + 65.0

        # Extract subrange to fit
        w_spec = []
        f_spec = []
        w_spec.extend(
            (wavelength[i])
            for i in range(len(wavelength))
            if (wavelength[i] > lmin and wavelength[i] < lmax)
        )
        f_spec.extend(
            (spectrum[i])
            for i in range(len(wavelength))
            if (wavelength[i] > lmin and wavelength[i] < lmax)
        )

        # Setup min and max flux values in subrange to fit
        if fmin == 0:
            fmin = np.nanmin(f_spec)
        if fmax == 0:
            fmax = np.nanmax(f_spec)

        # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to centre
        w_cont = []
        f_cont = []
        w_cont.extend(
            (w_spec[i])
            for i in range(len(w_spec))
            if (w_spec[i] > centre - lowlow and w_spec[i] < centre - lowhigh)
            or (w_spec[i] > centre + highlow and w_spec[i] < centre + highhigh)
        )
        f_cont.extend(
            (f_spec[i])
            for i in range(len(w_spec))
            if (w_spec[i] > centre - lowlow and w_spec[i] < centre - lowhigh)
            or (w_spec[i] > centre + highlow and w_spec[i] < centre + highhigh)
        )

        # Linear Fit to continuum
        try:
            mm, bb = np.polyfit(w_cont, f_cont, 1)
        except Exception:
            bb = np.nanmedian(spectrum)
            mm = 0.0
            if verbose:
                print("  Impossible to get the continuum!")
                print("  Scaling the continuum to the median value")
        continuum = mm * np.array(w_spec) + bb
        # c_cont = mm*np.array(w_cont)+bb
        # rms continuum
        # rms_cont = np.nansum([ np.abs(f_cont[i] - c_cont[i])  for i in range(len(w_cont)) ]) / len(c_cont)

        if search_peak:
            # Search for index here w_spec(index) closest to line
            try:
                min_w = np.abs(np.array(w_spec) - centre)
                mini = np.nanmin(min_w)
                peak = (
                    f_spec[min_w.tolist().index(mini)]
                    - continuum[min_w.tolist().index(mini)]
                )
                flux = peak * sigma * np.sqrt(2 * np.pi)
                if verbose:
                    print("  Using peak as f[", centre, "] = ", peak, " and sigma = ", sigma, "    flux = ", flux)
            except Exception:
                print("  Error trying to get the peak as requested wavelength is ", centre, "! Ignoring this fit!")
                peak = 0.0
                flux = -0.0001

        no_substract = False
        if flux < 0:
            if allow_absorptions == False:
                if verbose:
                    print("  WARNING! This is an ABSORPTION Gaussian! As requested, this Gaussian is NOT substracted!")
                    no_substract = True
        # print no_substract
        if no_substract == False:
            if verbose:
                print("  Substracting Gaussian at {:7.1f}  with peak ={:10.4f}   sigma ={:6.2f}  and flux ={:9.4f}".format(
                    centre, peak, sigma, flux
                ))

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
                plt.xlabel("Wavelength [$\AA$]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
                else:
                    plt.ylabel("Flux [ counts ]")
                plt.xlim(lmin, lmax)
                plt.ylim(fmin, fmax)

                # Vertical line at line
                plt.axvline(x=centre, color="k", linestyle="-", alpha=0.8)
                # Horizontal line at y = 0
                plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
                # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
                plt.axvspan(
                    centre + highlow,
                    centre + highhigh,
                    facecolor="g",
                    alpha=0.15,
                    zorder=3,
                )
                plt.axvspan(
                    centre - lowlow,
                    centre - lowhigh,
                    facecolor="g",
                    alpha=0.15,
                    zorder=3,
                )
                # Plot linear fit for continuum
                plt.plot(w_spec, continuum, "g--")
                # Plot Gaussian fit
                plt.plot(w_spec, gaussian_fit + continuum, "r-", alpha=0.8)
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
                plt.xlabel("Wavelength [$\AA$]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
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


def fluxes(
    wavelength,
    s,
    line,
    lowlow=14,
    lowhigh=6,
    highlow=6,
    highhigh=14,
    lmin=0,
    lmax=0,
    fmin=0,
    fmax=0,
    broad=2.355,
    plot=True,
    verbose=True,
    plot_sus=False,
    fcal=True,
    fit_continuum=True,
    median_kernel=35,
    warnings=True,
):  # Broad is FWHM for Gaussian sigma= 1,
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

    Returns
    -------

    This routine provides a list compiling the results. The list has the the following format:

        resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]

    "spectrum" in resultado[11] is the spectrum-fit (New 22 Jan 2019).

    Parameters
    ----------
    wavelength: float
        wavelength.
    spectrum: float
        flux per wavelength
    line: float
        approx. observed central wavelength of emission line to fit.
    lmin, lmax: float
        wavelength range to be analysed
    fmin, fmax: float (default = 0, 0.)
        minimum and maximum values of flux to be plotted.
        If 0 is given (i.e. defaul) the routine uses the nanmin and nanmax values of the given spectrum.
    plot: boolean (default = True)
        Plot a figure with the emission lines identifications.
    verbose: boolean (default = True)
        Print results.
    fit_continuum: boolean (default = True)
        Perform a linear fit of the continuum using all data, otherwise it just does a linear fit considering only the two median values in each continuum range.
    median_kernel: odd integer (default = 35)
        size of the median filter to be applied to the continuum.
    Example
    -------
    >>> resultado = fluxes(wavelength, spectrum, 6603, fmin=-5.0E-17, fmax=2.0E-16, plot=True, verbose=False)
    """

    # Setup wavelength limits
    if lmin == 0:
        lmin = line - 65.0  # By default, +-65 A with respect to line
    if lmax == 0:
        lmax = line + 65.0

    # Extract subrange to fit
    w_spec = []
    f_spec = []
    w_spec.extend(
        (wavelength[i])
        for i in range(len(wavelength))
        if (wavelength[i] > lmin and wavelength[i] < lmax)
    )
    f_spec.extend(
        (s[i])
        for i in range(len(wavelength))
        if (wavelength[i] > lmin and wavelength[i] < lmax)
    )

    # Setup min and max flux values in subrange to fit
    if fmin == 0:
        fmin = np.nanmin(f_spec)
    if fmax == 0:
        fmax = np.nanmax(f_spec)

    # We have to find some "guess numbers" for the Gaussian
    # Now guess_centre is line
    guess_centre = line

    # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to guess_centre
    #        lowlow   = 16.
    #        lowhigh  = 6.
    #        highlow  = 20.
    #        highhigh = 30.

    w_cont = []
    f_cont = []
    w_cont.extend(
        (w_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh)
        or (w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh)
    )
    f_cont.extend(
        (f_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh)
        or (w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh)
    )

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
            mm = 0.0
            if warnings:
                print("  Impossible to get the continuum!")
                print("  Scaling the continuum to the median value")
        continuum = mm * np.array(w_spec) + bb
        c_cont = mm * np.array(w_cont) + bb
    else:
        # Median value in each continuum range  # NEW 15 Sep 2019
        w_cont_low = []
        f_cont_low = []
        w_cont_low.extend(
            (w_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh
            )
        )
        f_cont_low.extend(
            (f_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh
            )
        )
        median_w_cont_low = np.nanmedian(w_cont_low)
        median_f_cont_low = np.nanmedian(f_cont_low)
        w_cont_high = []
        f_cont_high = []
        w_cont_high.extend(
            (w_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre + highlow
                and w_spec[i] < guess_centre + highhigh
            )
        )
        f_cont_high.extend(
            (f_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre + highlow
                and w_spec[i] < guess_centre + highhigh
            )
        )
        median_w_cont_high = np.nanmedian(w_cont_high)
        median_f_cont_high = np.nanmedian(f_cont_high)

        b = old_div((median_f_cont_low - median_f_cont_high), (
            median_w_cont_low - median_w_cont_high
        ))
        a = median_f_cont_low - b * median_w_cont_low

        continuum = a + b * np.array(w_spec)
        c_cont = b * np.array(w_cont) + a

    # rms continuum
    rms_cont = old_div(np.nansum(
        [np.abs(f_cont[i] - c_cont[i]) for i in range(len(w_cont))]
    ), len(c_cont))

    # Search for index here w_spec(index) closest to line
    min_w = np.abs(np.array(w_spec) - line)
    mini = np.nanmin(min_w)
    #    guess_peak = f_spec[min_w.tolist().index(mini)]   # WE HAVE TO SUSTRACT CONTINUUM!!!
    guess_peak = (
        f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]
    )

    # Search for beginning/end of emission line, choosing line +-10    -------- MODIFICO DESDE AQUI
    # 28th Feb 2019: Check central value between low_limit and high_limit
    #    w_fit = []
    #    f_fit = []
    #    w_fit.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-15 and w_spec[i] < guess_centre+15))
    #    f_fit.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-15 and w_spec[i] < guess_centre+15))
    #    c_fit=mm*np.array(w_fit)+bb
    #
    #    maximo_index = f_fit.index(np.nanmax(f_fit))
    #    last_index=len(w_fit)
    #    low_limit = w_fit[0]
    #    high_limit = 0
    #    ii=0
    #    while high_limit == 0 :    #### CHECK HOW low_limit is obtained !!!
    #        ii=ii+1
    # print f_fit[ii]/c_fit[ii],f_fit[ii-1]/c_fit[ii-1]
    #        if ii < last_index :
    #            if f_fit[ii]/c_fit[ii] > 1.05 and f_fit[ii-1]/c_fit[ii-1] < 1.05 and ii < maximo_index :
    #                low_limit = w_fit[ii-1]
    #            if f_fit[ii]/c_fit[ii] < 1.05 and ii > maximo_index :  #1.05
    #                high_limit = w_fit[ii-1]
    #        else:
    #            high_limit = last_index

    # ------------ HASTA AQUI

    # LOW limit
    low_limit = 0
    w_fit = []
    f_fit = []
    w_fit.extend(
        (w_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre - 15 and w_spec[i] < guess_centre)
    )
    f_fit.extend(
        (f_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre - 15 and w_spec[i] < guess_centre)
    )
    if fit_continuum:
        c_fit = mm * np.array(w_fit) + bb
    else:
        c_fit = b * np.array(w_fit) + a

    fs = []
    ws = []
    for ii in range(len(w_fit) - 1, 1, -1):
        if (
            old_div(f_fit[ii], c_fit[ii]) < 1.05
            and old_div(f_fit[ii - 1], c_fit[ii - 1]) < 1.05
            and low_limit == 0
        ):
            low_limit = w_fit[ii]
        #        if f_fit[ii]/c_fit[ii] < 1.05 and low_limit == 0: low_limit = w_fit[ii]
        fs.append(old_div(f_fit[ii], c_fit[ii]))
        ws.append(w_fit[ii])
    if low_limit == 0:
        sorted_by_flux = np.argsort(fs)
        low_limit = ws[sorted_by_flux[0]]

    # HIGH LIMIT
    high_limit = 0
    w_fit = []
    f_fit = []
    w_fit.extend(
        (w_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre and w_spec[i] < guess_centre + 15)
    )
    f_fit.extend(
        (f_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre and w_spec[i] < guess_centre + 15)
    )
    if fit_continuum:
        c_fit = mm * np.array(w_fit) + bb
    else:
        c_fit = b * np.array(w_fit) + a

    fs = []
    ws = []
    for ii in range(len(w_fit) - 1):
        if (
            old_div(f_fit[ii], c_fit[ii]) < 1.05
            and old_div(f_fit[ii + 1], c_fit[ii + 1]) < 1.05
            and high_limit == 0
        ):
            high_limit = w_fit[ii]
        #        if f_fit[ii]/c_fit[ii] < 1.05 and high_limit == 0: high_limit = w_fit[ii]
        fs.append(old_div(f_fit[ii], c_fit[ii]))
        ws.append(w_fit[ii])
    if high_limit == 0:
        sorted_by_flux = np.argsort(fs)
        high_limit = ws[sorted_by_flux[0]]

    # Fit a Gaussian to data - continuum
    p0 = [
        guess_centre,
        guess_peak,
        broad / 2.355,
    ]  # broad is the Gaussian sigma, 1.0 for emission lines
    try:
        fit, pcov = curve_fit(
            gauss, w_spec, f_spec - continuum, p0=p0, maxfev=10000
        )  # If this fails, increase maxfev...
        fit_error = np.sqrt(np.diag(pcov))

        # New 28th Feb 2019: Check central value between low_limit and high_limit
        # Better: between guess_centre - broad, guess_centre + broad
        # If not, redo fit fixing central value to the peak (it does not work... just fix FWHM= (high_limit-low_limit)/2.5 )

        if verbose != False:
            print(" ----------------------------------------------------------------------------------------")
        #        if low_limit < fit[0] < high_limit:
        if fit[0] < guess_centre - broad or fit[0] > guess_centre + broad:
            #            if verbose: print "  Fitted center wavelength", fit[0],"is NOT in the range [",low_limit,",",high_limit,"]"
            if verbose:
                print("  Fitted center wavelength", fit[
                    0
                ], "is NOT in the expected range [", guess_centre - broad, ",", guess_centre + broad, "]")

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
            if verbose:
                print("  Fitted center wavelength", fit[
                    0
                ], "is NOT in the expected range [", guess_centre - broad, ",", guess_centre + broad, "]")

        # TILL HERE

        if verbose:
            print("  Fit parameters =  ", fit[0], fit[1], fit[2])
        if fit[2] == broad and warnings == True:
            print("  WARNING: Fit in", fit[
                0
            ], "failed! Using given centre wavelength (cw), peak at (cv) & sigma = broad/2.355 given.")
        gaussian_fit = gauss(w_spec, fit[0], fit[1], fit[2])

        # Estimate rms of the Gaussian fit in range [low_limit, high_limit]
        residuals = f_spec - gaussian_fit - continuum
        rms_fit = np.nansum(
            [
                (old_div((residuals[i] ** 2), (len(residuals) - 2))) ** 0.5
                for i in range(len(w_spec))
                if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)
            ]
        )

        # Fluxes, FWHM and Eq. Width calculations
        gaussian_flux = gauss_flux(fit[1], fit[2])
        error1 = np.abs(gauss_flux(fit[1] + fit_error[1], fit[2]) - gaussian_flux)
        error2 = np.abs(gauss_flux(fit[1], fit[2] + fit_error[2]) - gaussian_flux)
        gaussian_flux_error = old_div(1, (old_div(1, error1 ** 2) + old_div(1, error2 ** 2)) ** 0.5)

        fwhm = fit[2] * 2.355
        fwhm_error = fit_error[2] * 2.355
        fwhm_vel = old_div(fwhm, fit[0]) * C
        fwhm_vel_error = old_div(fwhm_error, fit[0]) * C

        gaussian_ew = old_div(gaussian_flux, np.nanmedian(f_cont))
        gaussian_ew_error = old_div(gaussian_ew * gaussian_flux_error, gaussian_flux)

        # Integrated flux
        # IRAF: flux = sum ((I(i)-C(i)) * (w(i2) - w(i1)) / (i2 - i2)
        flux = np.nansum(
            [
                (f_spec[i] - continuum[i]) * (w_spec[i + 1] - w_spec[i])
                for i in range(len(w_spec))
                if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)
            ]
        )
        flux_error = rms_cont * (high_limit - low_limit)
        wave_resolution = old_div((wavelength[-1] - wavelength[0]), len(wavelength))
        ew = wave_resolution * np.nansum(
            [
                (1 - old_div(f_spec[i], continuum[i]))
                for i in range(len(w_spec))
                if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)
            ]
        )
        ew_error = np.abs(old_div(ew * flux_error, flux))
        gauss_to_integrated = old_div(gaussian_flux, flux) * 100.0

        # Plotting
        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(np.array(w_spec), np.array(f_spec), "b", lw=3, alpha=0.5)
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(lmin, lmax)
            plt.ylim(fmin, fmax)

            # Vertical line at guess_centre
            plt.axvline(x=guess_centre, color="r", linestyle="-", alpha=0.5)
            # Horizontal line at y = 0
            plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
            # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
            plt.axvspan(
                guess_centre + highlow,
                guess_centre + highhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            plt.axvspan(
                guess_centre - lowlow,
                guess_centre - lowhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            # Plot linear fit for continuum
            plt.plot(w_spec, continuum, "g--")
            # Plot Gaussian fit
            plt.plot(w_spec, gaussian_fit + continuum, "r-", alpha=0.8)
            # Vertical line at Gaussian center
            plt.axvline(x=fit[0], color="k", linestyle="-", alpha=0.5)
            # Vertical lines to emission line
            plt.axvline(x=low_limit, color="k", linestyle=":", alpha=0.5)
            plt.axvline(x=high_limit, color="k", linestyle=":", alpha=0.5)
            # Plot residuals
            plt.plot(w_spec, residuals, "k")
            plt.title(
                "Fit: x0=%.2f y0=%.2e sigma=%.2f  flux=%.2e  rms=%.3e"
                % (fit[0], fit[1], fit[2], gaussian_flux, rms_fit)
            )
            plt.show()

        # Printing results
        if verbose:
            print("\n> Gauss and continuum fitting + integrated flux calculations:\n")
            print("rms continuum = %.3e erg/cm/s/A " % (rms_cont))
            print("Gaussian Fit parameters: x0 = ( %.2f +- %.2f )  A " % (
                fit[0],
                fit_error[0],
            ))
            print("                         y0 = ( %.3f +- %.3f )  1E-16 erg/cm2/s/A" % (
                old_div(fit[1], 1e-16),
                old_div(fit_error[1], 1e-16),
            ))
            print("                      sigma = ( %.3f +- %.3f )  A" % (
                fit[2],
                fit_error[2],
            ))
            print("                    rms fit = %.3e erg/cm2/s/A" % (rms_fit))
            print("Gaussian Flux = ( %.2f +- %.2f ) 1E-16 erg/s/cm2         (error = %.1f per cent)" % (
                old_div(gaussian_flux, 1e-16),
                old_div(gaussian_flux_error, 1e-16),
                old_div(gaussian_flux_error, gaussian_flux) * 100,
            ))
            print("FWHM          = ( %.3f +- %.3f ) A    =   ( %.1f +- %.1f ) km/s " % (
                fwhm,
                fwhm_error,
                fwhm_vel,
                fwhm_vel_error,
            ))
            print("Eq. Width     = ( %.1f +- %.1f ) A" % (
                -gaussian_ew,
                gaussian_ew_error,
            ))
            print("\nIntegrated flux  = ( %.2f +- %.2f ) 1E-16 erg/s/cm2      (error = %.1f per cent) " % (
                old_div(flux, 1e-16),
                old_div(flux_error, 1e-16),
                old_div(flux_error, flux) * 100,
            ))
            print("Eq. Width        = ( %.1f +- %.1f ) A" % (ew, ew_error))
            print("Gauss/Integrated = %.2f per cent " % gauss_to_integrated)

        # New 22 Jan 2019: sustract Gaussian fit

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
        if plot_sus:
            plt.figure(figsize=(10, 4))
            plt.plot(wavelength, s, "r")
            plt.plot(wavelength, s_s, "c")
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(lmin, lmax)
            plt.ylim(fmin, fmax)
            plt.show()
            plt.close()

        #                     0      1         2                3               4              5      6         7        8        9     10      11
        resultado = [
            rms_cont,
            fit[0],
            fit_error[0],
            gaussian_flux,
            gaussian_flux_error,
            fwhm,
            fwhm_error,
            flux,
            flux_error,
            ew,
            ew_error,
            s_s,
        ]
        return resultado
    except Exception:
        if verbose:
            print("  Gaussian fit failed!")
        resultado = [
            0,
            line,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            s,
        ]  # line was identified at lambda=line but Gaussian fit failed

        # NOTA: PUEDE DEVOLVER EL FLUJO INTEGRADO AUNQUE FALLE EL AJUSTE GAUSSIANO...

        # Plotting
        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(np.array(w_spec), np.array(f_spec), "b", lw=3, alpha=0.5)
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(lmin, lmax)
            plt.ylim(fmin, fmax)

            # Vertical line at guess_centre
            plt.axvline(x=guess_centre, color="r", linestyle="-", alpha=0.5)
            # Horizontal line at y = 0
            plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
            # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
            plt.axvspan(
                guess_centre + highlow,
                guess_centre + highhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            plt.axvspan(
                guess_centre - lowlow,
                guess_centre - lowhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            # Plot linear fit for continuum
            plt.plot(w_spec, continuum, "g--")
            # Plot Gaussian fit
            #            plt.plot(w_spec, gaussian_fit+continuum, 'r-', alpha=0.8)
            # Vertical line at Gaussian center
            #            plt.axvline(x=fit[0], color='k', linestyle='-', alpha=0.5)
            # Vertical lines to emission line
            plt.axvline(x=low_limit, color="k", linestyle=":", alpha=0.5)
            plt.axvline(x=high_limit, color="k", linestyle=":", alpha=0.5)
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
    return p[1] * np.exp(-0.5 * (old_div((x - p[0]), p[2])) ** 2) + p[4] * np.exp(
        -0.5 * (old_div((x - p[3]), p[5])) ** 2
    )


def dfluxes(
    wavelength,
    s,
    line1,
    line2,
    lowlow=14,
    lowhigh=6,
    highlow=6,
    highhigh=14,
    lmin=0,
    lmax=0,
    fmin=0,
    fmax=0,
    broad1=2.355,
    broad2=2.355,
    plot=True,
    verbose=True,
    plot_sus=False,
    fcal=True,
    fit_continuum=True,
    median_kernel=35,
    warnings=True,
):  # Broad is FWHM for Gaussian sigma= 1,
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

    Returns
    -------

    This routine provides a list compiling the results. The list has the following format:

        resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]

    "spectrum" in resultado[11] is the spectrum-fit (New 22 Jan 2019).

    Parameters
    ----------
    wavelength: float
        wavelength.
    spectrum: float
        flux per wavelength
    line: float
        approx. observed central wavelength of emission line to fit.
    lmin, lmax: float
        wavelength range to be analysed
    fmin, fmax: float (default = 0, 0.)
        minimum and maximum values of flux to be plotted.
        If 0 is given (i.e. defaul) the routine uses the nanmin and nanmax values of the given spectrum.
    plot: boolean (default = True)
        Plot a figure with the emission lines identifications.
    verbose: boolean (default = True)
        Print results.
    fit_continuum: boolean (default = True)
        Perform a linear fit of the continuum using all data, otherwise it just does a linear fit considering only the two median values in each continuum range.
    median_kernel: odd integer (default = 35)
        size of the median filter to be applied to the continuum.

    Example
    -------
    >>> resultado = fluxes(wavelength, spectrum, 6603, fmin=-5.0E-17, fmax=2.0E-16, plot=True, verbose=False)
    """

    # Setup wavelength limits
    if lmin == 0:
        lmin = line1 - 65.0  # By default, +-65 A with respect to line
    if lmax == 0:
        lmax = line2 + 65.0

    # Extract subrange to fit
    w_spec = []
    f_spec = []
    w_spec.extend(
        (wavelength[i])
        for i in range(len(wavelength))
        if (wavelength[i] > lmin and wavelength[i] < lmax)
    )
    f_spec.extend(
        (s[i])
        for i in range(len(wavelength))
        if (wavelength[i] > lmin and wavelength[i] < lmax)
    )

    # Setup min and max flux values in subrange to fit
    if fmin == 0:
        fmin = np.nanmin(f_spec)
    if fmax == 0:
        fmax = np.nanmax(f_spec)

    # We have to find some "guess numbers" for the Gaussian
    # Now guess_centre is line
    guess_centre1 = line1
    guess_centre2 = line2
    guess_centre = (guess_centre1 + guess_centre2) / 2.0
    # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to guess_centre

    w_cont = []
    f_cont = []
    w_cont.extend(
        (w_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh)
        or (w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh)
    )
    f_cont.extend(
        (f_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh)
        or (w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh)
    )

    if fit_continuum:
        # Linear Fit to continuum
        f_cont_filtered = sig.medfilt(f_cont, np.int(median_kernel))
        try:
            mm, bb = np.polyfit(w_cont, f_cont_filtered, 1)
        except Exception:
            bb = np.nanmedian(f_cont_filtered)
            mm = 0.0
            if warnings:
                print("  Impossible to get the continuum!")
                print("  Scaling the continuum to the median value")
        continuum = mm * np.array(w_spec) + bb
        c_cont = mm * np.array(w_cont) + bb
    else:
        # Median value in each continuum range  # NEW 15 Sep 2019
        w_cont_low = []
        f_cont_low = []
        w_cont_low.extend(
            (w_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh
            )
        )
        f_cont_low.extend(
            (f_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh
            )
        )
        median_w_cont_low = np.nanmedian(w_cont_low)
        median_f_cont_low = np.nanmedian(f_cont_low)
        w_cont_high = []
        f_cont_high = []
        w_cont_high.extend(
            (w_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre + highlow
                and w_spec[i] < guess_centre + highhigh
            )
        )
        f_cont_high.extend(
            (f_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre + highlow
                and w_spec[i] < guess_centre + highhigh
            )
        )
        median_w_cont_high = np.nanmedian(w_cont_high)
        median_f_cont_high = np.nanmedian(f_cont_high)

        b = old_div((median_f_cont_low - median_f_cont_high), (
            median_w_cont_low - median_w_cont_high
        ))
        a = median_f_cont_low - b * median_w_cont_low

        continuum = a + b * np.array(w_spec)
        c_cont = b * np.array(w_cont) + a

    # rms continuum
    rms_cont = old_div(np.nansum(
        [np.abs(f_cont[i] - c_cont[i]) for i in range(len(w_cont))]
    ), len(c_cont))

    # Search for index here w_spec(index) closest to line
    min_w = np.abs(np.array(w_spec) - line1)
    mini = np.nanmin(min_w)
    guess_peak1 = (
        f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]
    )
    min_w = np.abs(np.array(w_spec) - line2)
    mini = np.nanmin(min_w)
    guess_peak2 = (
        f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]
    )

    # Search for beginning/end of emission line, choosing line +-10
    # 28th Feb 2019: Check central value between low_limit and high_limit

    # LOW limit
    low_limit = 0
    w_fit = []
    f_fit = []
    w_fit.extend(
        (w_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre1 - 15 and w_spec[i] < guess_centre1)
    )
    f_fit.extend(
        (f_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre1 - 15 and w_spec[i] < guess_centre1)
    )
    if fit_continuum:
        c_fit = mm * np.array(w_fit) + bb
    else:
        c_fit = b * np.array(w_fit) + a

    fs = []
    ws = []
    for ii in range(len(w_fit) - 1, 1, -1):
        if (
            old_div(f_fit[ii], c_fit[ii]) < 1.05
            and old_div(f_fit[ii - 1], c_fit[ii - 1]) < 1.05
            and low_limit == 0
        ):
            low_limit = w_fit[ii]
        #        if f_fit[ii]/c_fit[ii] < 1.05 and low_limit == 0: low_limit = w_fit[ii]
        fs.append(old_div(f_fit[ii], c_fit[ii]))
        ws.append(w_fit[ii])
    if low_limit == 0:
        sorted_by_flux = np.argsort(fs)
        low_limit = ws[sorted_by_flux[0]]

    # HIGH LIMIT
    high_limit = 0
    w_fit = []
    f_fit = []
    w_fit.extend(
        (w_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre2 and w_spec[i] < guess_centre2 + 15)
    )
    f_fit.extend(
        (f_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre2 and w_spec[i] < guess_centre2 + 15)
    )
    if fit_continuum:
        c_fit = mm * np.array(w_fit) + bb
    else:
        c_fit = b * np.array(w_fit) + a

    fs = []
    ws = []
    for ii in range(len(w_fit) - 1):
        if (
            old_div(f_fit[ii], c_fit[ii]) < 1.05
            and old_div(f_fit[ii + 1], c_fit[ii + 1]) < 1.05
            and high_limit == 0
        ):
            high_limit = w_fit[ii]
        #        if f_fit[ii]/c_fit[ii] < 1.05 and high_limit == 0: high_limit = w_fit[ii]
        fs.append(old_div(f_fit[ii], c_fit[ii]))
        ws.append(w_fit[ii])
    if high_limit == 0:
        sorted_by_flux = np.argsort(fs)
        high_limit = ws[sorted_by_flux[0]]

    # Fit a Gaussian to data - continuum
    p0 = [
        guess_centre1,
        guess_peak1,
        broad1 / 2.355,
        guess_centre2,
        guess_peak2,
        broad2 / 2.355,
    ]  # broad is the Gaussian sigma, 1.0 for emission lines
    try:
        fit, pcov = curve_fit(
            dgauss, w_spec, f_spec - continuum, p0=p0, maxfev=10000
        )  # If this fails, increase maxfev...
        fit_error = np.sqrt(np.diag(pcov))

        # New 28th Feb 2019: Check central value between low_limit and high_limit
        # Better: between guess_centre - broad, guess_centre + broad
        # If not, redo fit fixing central value to the peak (it does not work... just fix FWHM= (high_limit-low_limit)/2.5 )

        if verbose != False:
            print(" ----------------------------------------------------------------------------------------")
        if (
            fit[0] < guess_centre1 - broad1
            or fit[0] > guess_centre1 + broad1
            or fit[3] < guess_centre2 - broad2
            or fit[3] > guess_centre2 + broad2
        ):
            if warnings:
                if fit[0] < guess_centre1 - broad1 or fit[0] > guess_centre1 + broad1:
                    print("  Fitted center wavelength", fit[
                        0
                    ], "is NOT in the expected range [", guess_centre1 - broad1, ",", guess_centre1 + broad1, "]")
                else:
                    print("  Fitted center wavelength", fit[
                        0
                    ], "is in the expected range [", guess_centre1 - broad1, ",", guess_centre1 + broad1, "]")
                if fit[3] < guess_centre2 - broad2 or fit[3] > guess_centre2 + broad2:
                    print("  Fitted center wavelength", fit[
                        3
                    ], "is NOT in the expected range [", guess_centre2 - broad2, ",", guess_centre2 + broad2, "]")
                else:
                    print("  Fitted center wavelength", fit[
                        3
                    ], "is in the expected range [", guess_centre2 - broad2, ",", guess_centre2 + broad2, "]")
                print("  Fit failed!")

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
            if warnings:
                print("  Fitted center wavelength", fit[
                    0
                ], "is in the expected range [", guess_centre1 - broad1, ",", guess_centre1 + broad1, "]")
            if warnings:
                print("  Fitted center wavelength", fit[
                    3
                ], "is in the expected range [", guess_centre2 - broad2, ",", guess_centre2 + broad2, "]")

        gaussian_fit = dgauss(w_spec, fit[0], fit[1], fit[2], fit[3], fit[4], fit[5])

        if warnings:
            print("  Fit parameters =  ", fit[0], fit[1], fit[2], fit[3], fit[4], fit[5])
        if fit[2] == broad1 and warnings == True:
            print("  WARNING: Fit in", fit[
                0
            ], "failed! Using given centre wavelength (cw), peak at (cv) & sigma = broad/2.355 given.")  # CHECK THIS
        # gaussian_fit =  gauss(w_spec, fit[0], fit[1], fit[2])

        # Estimate rms of the Gaussian fit in range [low_limit, high_limit]
        residuals = f_spec - gaussian_fit - continuum
        rms_fit = np.nansum(
            [
                (old_div((residuals[i] ** 2), (len(residuals) - 2))) ** 0.5
                for i in range(len(w_spec))
                if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)
            ]
        )

        # Fluxes, FWHM and Eq. Width calculations  # CHECK THIS
        gaussian_flux = gauss_flux(fit[1], fit[2])
        error1 = np.abs(gauss_flux(fit[1] + fit_error[1], fit[2]) - gaussian_flux)
        error2 = np.abs(gauss_flux(fit[1], fit[2] + fit_error[2]) - gaussian_flux)
        gaussian_flux_error = old_div(1, (old_div(1, error1 ** 2) + old_div(1, error2 ** 2)) ** 0.5)

        fwhm = fit[2] * 2.355
        fwhm_error = fit_error[2] * 2.355
        fwhm_vel = old_div(fwhm, fit[0]) * C
        fwhm_vel_error = old_div(fwhm_error, fit[0]) * C

        gaussian_ew = old_div(gaussian_flux, np.nanmedian(f_cont))
        gaussian_ew_error = old_div(gaussian_ew * gaussian_flux_error, gaussian_flux)

        # Integrated flux
        # IRAF: flux = sum ((I(i)-C(i)) * (w(i2) - w(i1)) / (i2 - i2)
        flux = np.nansum(
            [
                (f_spec[i] - continuum[i]) * (w_spec[i + 1] - w_spec[i])
                for i in range(len(w_spec))
                if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)
            ]
        )
        flux_error = rms_cont * (high_limit - low_limit)
        wave_resolution = old_div((wavelength[-1] - wavelength[0]), len(wavelength))
        ew = wave_resolution * np.nansum(
            [
                (1 - old_div(f_spec[i], continuum[i]))
                for i in range(len(w_spec))
                if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)
            ]
        )
        ew_error = np.abs(old_div(ew * flux_error, flux))
        gauss_to_integrated = old_div(gaussian_flux, flux) * 100.0

        # Plotting
        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(np.array(w_spec), np.array(f_spec), "b", lw=3, alpha=0.5)
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(old_div((line1 + line2), 2) - 40, old_div((line1 + line2), 2) + 40)
            plt.ylim(fmin, fmax)

            # Vertical line at guess_centre
            plt.axvline(x=guess_centre1, color="r", linestyle="-", alpha=0.5)
            plt.axvline(x=guess_centre2, color="r", linestyle="-", alpha=0.5)

            # Horizontal line at y = 0
            plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
            # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
            plt.axvspan(
                guess_centre + highlow,
                guess_centre + highhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            plt.axvspan(
                guess_centre - lowlow,
                guess_centre - lowhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            # Plot linear fit for continuum
            plt.plot(w_spec, continuum, "g--")
            # Plot Gaussian fit
            plt.plot(w_spec, gaussian_fit + continuum, "r-", alpha=0.8)
            # Vertical line at Gaussian center
            plt.axvline(x=fit[0], color="k", linestyle="-", alpha=0.5)
            plt.axvline(x=fit[3], color="k", linestyle="-", alpha=0.5)

            # Vertical lines to emission line
            plt.axvline(x=low_limit, color="k", linestyle=":", alpha=0.5)
            plt.axvline(x=high_limit, color="k", linestyle=":", alpha=0.5)
            # Plot residuals
            plt.plot(w_spec, residuals, "k")
            plt.title(
                "Double Gaussian Fit"
            )  # Fit: x0=%.2f y0=%.2e sigma=%.2f  flux=%.2e  rms=%.3e' % (fit[0], fit[1], fit[2], gaussian_flux, rms_fit))
            plt.show()

        # Printing results
        if verbose:
            print("\n> Gauss and continuum fitting + integrated flux calculations:\n")
            print("rms continuum = %.3e erg/cm/s/A " % (rms_cont))
            print("Gaussian Fit parameters: x0 = ( %.2f +- %.2f )  A " % (
                fit[0],
                fit_error[0],
            ))
            print("                         y0 = ( %.3f +- %.3f )  1E-16 erg/cm2/s/A" % (
                old_div(fit[1], 1e-16),
                old_div(fit_error[1], 1e-16),
            ))
            print("                      sigma = ( %.3f +- %.3f )  A" % (
                fit[2],
                fit_error[2],
            ))
            print("                    rms fit = %.3e erg/cm2/s/A" % (rms_fit))
            print("Gaussian Flux = ( %.2f +- %.2f ) 1E-16 erg/s/cm2         (error = %.1f per cent)" % (
                old_div(gaussian_flux, 1e-16),
                old_div(gaussian_flux_error, 1e-16),
                old_div(gaussian_flux_error, gaussian_flux) * 100,
            ))
            print("FWHM          = ( %.3f +- %.3f ) A    =   ( %.1f +- %.1f ) km/s " % (
                fwhm,
                fwhm_error,
                fwhm_vel,
                fwhm_vel_error,
            ))
            print("Eq. Width     = ( %.1f +- %.1f ) A" % (
                -gaussian_ew,
                gaussian_ew_error,
            ))
            print("\nIntegrated flux  = ( %.2f +- %.2f ) 1E-16 erg/s/cm2      (error = %.1f per cent) " % (
                old_div(flux, 1e-16),
                old_div(flux_error, 1e-16),
                old_div(flux_error, flux) * 100,
            ))
            print("Eq. Width        = ( %.1f +- %.1f ) A" % (ew, ew_error))
            print("Gauss/Integrated = %.2f per cent " % gauss_to_integrated)

        # New 22 Jan 2019: sustract Gaussian fit

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
        if plot_sus:
            plt.figure(figsize=(10, 4))
            plt.plot(wavelength, s, "r")
            plt.plot(wavelength, s_s, "c")
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(lmin, lmax)
            plt.ylim(fmin, fmax)
            plt.show()
            plt.close()

        #                     0      1         2                3               4              5      6         7        8        9     10      11   12       13      14
        resultado = [
            rms_cont,
            fit[0],
            fit_error[0],
            gaussian_flux,
            gaussian_flux_error,
            fwhm,
            fwhm_error,
            flux,
            flux_error,
            ew,
            ew_error,
            s_s,
            fit[3],
            fit[4],
            fit[5],
        ]
        return resultado
    except Exception:
        if verbose:
            print("  Gaussian fit failed!")
        resultado = [
            0,
            line1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            s,
            0,
            0,
            0,
        ]  # line was identified at lambda=line but Gaussian fit failed

        # NOTA: PUEDE DEVOLVER EL FLUJO INTEGRADO AUNQUE FALLE EL AJUSTE GAUSSIANO...

        # Plotting
        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(np.array(w_spec), np.array(f_spec), "b", lw=3, alpha=0.5)
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(lmin, lmax)
            plt.ylim(fmin, fmax)

            # Vertical line at guess_centre
            plt.axvline(x=guess_centre, color="r", linestyle="-", alpha=0.5)
            # Horizontal line at y = 0
            plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
            # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
            plt.axvspan(
                guess_centre + highlow,
                guess_centre + highhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            plt.axvspan(
                guess_centre - lowlow,
                guess_centre - lowhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            # Plot linear fit for continuum
            plt.plot(w_spec, continuum, "g--")
            # Plot Gaussian fit
            #            plt.plot(w_spec, gaussian_fit+continuum, 'r-', alpha=0.8)
            # Vertical line at Gaussian center
            #            plt.axvline(x=fit[0], color='k', linestyle='-', alpha=0.5)
            # Vertical lines to emission line
            plt.axvline(x=low_limit, color="k", linestyle=":", alpha=0.5)
            plt.axvline(x=high_limit, color="k", linestyle=":", alpha=0.5)
            # Plot residuals
            #            plt.plot(w_spec, residuals, 'k')
            plt.title("No Gaussian fit obtained...")
            plt.show()

        return resultado


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def search_peaks(
    wavelength,
    flux,
    smooth_points=20,
    lmin=0,
    lmax=0,
    fmin=0.5,
    fmax=3.0,
    emission_line_file="lineas_c89_python.dat",
    brightest_line="Ha",
    cut=1.2,
    check_redshift=0.0003,
    only_id_lines=True,
    plot=True,
    verbose=True,
    fig_size=12,
):
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

    Returns
    -------

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
    wavelength: float
        wavelength.
    flux: float
        flux per wavelength
    smooth_points: float (default = 20)
        Number of points for a smooth spectrum to get a rough estimation of the global continuum
    lmin, lmax: float
        wavelength range to be analysed
    fmin, fmax: float (default = 0.5, 2.)
        minimum and maximum values of flux/continuum to be plotted
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
    step = np.int(old_div(len(wavelength), smooth_points))  # step
    w_cont_smooth = np.zeros(smooth_points)
    f_cont_smooth = np.zeros(smooth_points)

    for j in range(smooth_points):
        w_cont_smooth[j] = np.nanmedian(
            [
                wavelength[i]
                for i in range(len(wavelength))
                if (i > step * j and i < step * (j + 1))
            ]
        )
        f_cont_smooth[j] = np.nanmedian(
            [
                flux[i]
                for i in range(len(wavelength))
                if (i > step * j and i < step * (j + 1))
            ]
        )  # / np.nanmedian(spectrum)
        # print j,w_cont_smooth[j], f_cont_smooth[j]

    interpolated_continuum_smooth = interpolate.splrep(
        w_cont_smooth, f_cont_smooth, s=0
    )
    interpolated_continuum = interpolate.splev(
        wavelength, interpolated_continuum_smooth, der=0
    )

    funcion = old_div(flux, interpolated_continuum)

    # Searching for peaks using cut = 1.2 by default
    peaks = []
    index_low = 0
    for i in range(len(wavelength)):
        if funcion[i] > cut and funcion[i - 1] < cut:
            index_low = i
        if funcion[i] < cut and funcion[i - 1] > cut:
            index_high = i
            if index_high != 0:
                pfun = np.nanmax(
                    [
                        funcion[j]
                        for j in range(len(wavelength))
                        if (j > index_low and j < index_high + 1)
                    ]
                )
                peak = wavelength[funcion.tolist().index(pfun)]
                if (index_high - index_low) > 1:
                    peaks.append(peak)

    # Identify lines
    # Read file with data of emission lines:
    # 6300.30 [OI] -0.263    15   5    5    15
    # el_center el_name el_fnl lowlow lowhigh highlow highigh
    # Only el_center and el_name are needed
    (
        el_center,
        el_name,
        el_fnl,
        el_lowlow,
        el_lowhigh,
        el_highlow,
        el_highhigh,
    ) = read_table(emission_line_file, ["f", "s", "f", "f", "f", "f", "f"])
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
    Ha_redshift = old_div((Ha_w_obs - Ha_w_rest), Ha_w_rest)
    if verbose:
        print("\n> Detected %i emission lines using %8s at %8.2f A as brightest line!!\n" % (
            len(peaks),
            brightest_line,
            Ha_w_rest,
        ))
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
        minimo_w = np.abs(old_div(peaks[i], (1 + Ha_redshift)) - el_center)
        if np.nanmin(minimo_w) < 2.5:
            indice = minimo_w.tolist().index(np.nanmin(minimo_w))
            peaks_name[i] = el_name[indice]
            peaks_rest[i] = el_center[indice]
            peaks_redshift[i] = old_div((peaks[i] - el_center[indice]), el_center[indice])
            peaks_lowlow[i] = el_lowlow[indice]
            peaks_lowhigh[i] = el_lowhigh[indice]
            peaks_highlow[i] = el_highlow[indice]
            peaks_highhigh[i] = el_highhigh[indice]
            if verbose:
                print("%9s %8.2f found in %8.2f at z=%.6f   |z-zref| = %.6f" % (
                    peaks_name[i],
                    peaks_rest[i],
                    peaks[i],
                    peaks_redshift[i],
                    np.abs(peaks_redshift[i] - Ha_redshift),
                ))
            # print peaks_lowlow[i],peaks_lowhigh[i],peaks_highlow[i],peaks_highhigh[i]
    # Check if all redshifts are similar, assuming check_redshift = 0.0003 by default
    # If OK, add id_peaks[i]=1, if not, id_peaks[i]=0
    id_peaks = []
    for i in range(len(peaks_redshift)):
        if np.abs(peaks_redshift[i] - Ha_redshift) > check_redshift:
            if verbose:
                print("  WARNING!!! Line %8s in w = %.2f has redshift z=%.6f, different than zref=%.6f" % (
                    peaks_name[i],
                    peaks[i],
                    peaks_redshift[i],
                    Ha_redshift,
                ))
            id_peaks.append(0)
        else:
            id_peaks.append(1)

    if plot:
        plot_redshift_peaks(fig_size,
                            funcion,
                            wavelength,
                            lmin,
                            lmax,
                            fmin,
                            fmax,
                            cut,
                            peaks,
                            peaks_name,
                            label)

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
        continuum_limits_r = [
            peaks_lowlow_r,
            peaks_lowhigh_r,
            peaks_highlow_r,
            peaks_highhigh_r,
        ]

        return peaks_r, peaks_name_r, peaks_rest_r, continuum_limits_r
    else:
        return peaks, peaks_name, peaks_rest, continuum_limits


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def smooth_spectrum(
    wlm,
    s,
    wave_min=0,
    wave_max=0,
    step=50,
    exclude_wlm=[[0, 0]],
    weight_fit_median=0.5,
    plot=False,
    verbose=False,
):
    """
    Smooth a spectrum
    TODO: More here
    """

    if verbose:
        print("\n> Computing smooth spectrum...")

    if wave_min == 0:
        wave_min = wlm[0]
    if wave_max == 0:
        wave_max = wlm[-1]

    running_wave = []
    running_step_median = []
    cuts = np.int(old_div((wave_max - wave_min), step))

    exclude = 0
    corte_index = -1
    for corte in range(cuts + 1):
        next_wave = wave_min + step * corte
        if next_wave < wave_max:

            if (
                next_wave > exclude_wlm[exclude][0]
                and next_wave < exclude_wlm[exclude][1]
            ):
                if verbose:
                    print("  Skipping ", next_wave, " as it is in the exclusion range [", exclude_wlm[
                        exclude
                    ][
                        0
                    ], ",", exclude_wlm[
                        exclude
                    ][
                        1
                    ], "]")

            else:
                corte_index = corte_index + 1
                running_wave.append(next_wave)
                # print running_wave
                region = np.where(
                    (wlm > running_wave[corte_index] - old_div(step, 2))
                    & (wlm < running_wave[corte_index] + old_div(step, 2))
                )
                running_step_median.append(np.nanmedian(s[region]))
                if next_wave > exclude_wlm[exclude][1]:
                    exclude = exclude + 1
                    if verbose:
                        print("--- End exclusion range ", exclude)
                    if exclude == len(exclude_wlm):
                        exclude = len(exclude_wlm) - 1

    running_wave.append(wave_max)
    region = np.where((wlm > wave_max - step) & (wlm < wave_max + 0.1))
    running_step_median.append(np.nanmedian(s[region]))

    # print running_wave
    # print running_step_median
    # Check not nan
    _running_wave_ = []
    _running_step_median_ = []
    for i in range(len(running_wave)):
        if np.isnan(running_step_median[i]):
            if verbose:
                print("  There is a nan in ", running_wave[i])
        else:
            _running_wave_.append(running_wave[i])
            _running_step_median_.append(running_step_median[i])

    a7x, a6x, a5x, a4x, a3x, a2x, a1x, a0x = np.polyfit(
        _running_wave_, _running_step_median_, 7
    )
    fit_median = (
        a0x
        + a1x * wlm
        + a2x * wlm ** 2
        + a3x * wlm ** 3
        + a4x * wlm ** 4
        + a5x * wlm ** 5
        + a6x * wlm ** 6
        + a7x * wlm ** 7
    )

    interpolated_continuum_smooth = interpolate.splrep(
        _running_wave_, _running_step_median_, s=0.02
    )
    fit_median_interpolated = interpolate.splev(
        wlm, interpolated_continuum_smooth, der=0
    )

    if plot:

        plot_weights_for_getting_smooth_spectrum(
            wlm,
            s,
            running_wave,
            running_step_median,
            fit_median,
            fit_median_interpolated,
            weight_fit_median,
            fit_median_interpolated,
            wave_min,
            wave_max,
            exclude_wlm)

        print("  Weights for getting smooth spectrum:  fit_median =", weight_fit_median, "    fit_median_interpolated =", (
            1 - weight_fit_median
        ))

    return (
        weight_fit_median * fit_median
        + (1 - weight_fit_median) * fit_median_interpolated
    )  # (fit_median+fit_median_interpolated)/2      # Decide if fit_median or fit_median_interpolated


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def obtain_sky_spectrum(
    sky, low_fibres=200, plot=True, fig_size=12, fcal=False, verbose=True
):
    """
    Obtain a sky-spectrum using N fibres with the lowest intensity values.
    We sort skyfibres by their integrated flux. The lowest `low_fibres` fibres are then used to create a sky-spectrum by median-combining them. We then return this 1D spectrum
    TODO: Fix argument types

    Args:
        sky ():
        low_fibres (int): After sorting, the lowest `low_fibres` are combined together
        plot (bool): If True, plot the spectrum
        fcal (bool): Passed to plot_plot
        verbose (bool): If True, print out the regions we've included

    Returns:
        array: a 1D sky spectrum.
    """
    # It uses the lowest low_fibres fibres to get an integrated spectrum
    integrated_intensity_sorted = np.argsort(sky.integrated_fibre)
    region = []
    for fibre in range(low_fibres):
        region.append(integrated_intensity_sorted[fibre])
    sky_spectrum = np.nanmedian(sky.intensity_corrected[region], axis=0)

    print("  We use the ", low_fibres, " fibres with the lowest integrated intensity to derive the sky spectrum")
    if verbose:
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
    Apply a median filter to a two dimensional (n_spectra x n_wave) array of sky spectra. The median filtering occurs at each wavelength slice between the n_spectra sky spectra. The actual filtering is handled by scipy.signal
    TODO: Check this is correct- I'm unsure why medfilt_sky should be a 2D array too? SPV

    Args:
        intensity_corrected (array of shape n_spectra x n_wave): Spectra we want to median filter
    """

    medfilt_sky = np.zeros((n_spectra, n_wave))
    for wave in range(n_wave):
        medfilt_sky[:, wave] = sig.medfilt(
            intensity_corrected[:, wave], kernel_size=win_sky
        )

    # replace crappy edge fibres with 0.5*win'th medsky
    for fibre_sky in range(n_spectra):
        if fibre_sky < np.rint(0.5 * win_sky):
            j = int(np.rint(0.5 * win_sky))
            medfilt_sky[fibre_sky, ] = copy.deepcopy(medfilt_sky[j, ])
        if fibre_sky > n_spectra - np.rint(0.5 * win_sky):
            j = int(np.rint(n_spectra - np.rint(0.5 * win_sky)))
            medfilt_sky[fibre_sky, ] = copy.deepcopy(medfilt_sky[j, ])
    return medfilt_sky


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def scale_sky_spectrum(
    wlm,
    sky_spectrum,
    spectra,
    cut_sky=4.0,
    fmax=10,
    fmin=1,
    valid_wave_min=0,
    valid_wave_max=0,
    fibre_list=[100, 200, 300, 400, 500, 600, 700, 800, 900],
    plot=True,
    verbose=True,
    warnings=True,
):

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

    if valid_wave_min == 0:
        valid_wave_min = wlm[0]
    if valid_wave_max == 0:
        valid_wave_max = wlm[-1]

    if verbose:
        print("\n> Identifying sky lines using cut_sky =", cut_sky, ", allowed SKY/OBJ values = [", fmin, ",", fmax, "]")
    if verbose:
        print("  Using fibres = ", fibre_list)

    peaks, peaks_name, peaks_rest, continuum_limits = search_peaks(
        wlm,
        sky_spectrum,
        plot=plot,
        cut=cut_sky,
        fmax=fmax,
        only_id_lines=False,
        verbose=False,
    )

    ratio_list = []
    valid_peaks = []

    if verbose:
        print("\n        Sky line    Gaussian ratio     Flux ratio")
    n_sky_lines_found = 0
    for i in range(len(peaks)):
        sky_spectrum_data = fluxes(
            wlm,
            sky_spectrum,
            peaks[i],
            fcal=False,
            lowlow=50,
            highhigh=50,
            plot=False,
            verbose=False,
            warnings=warnings,
        )

        object_spectrum_data_gauss = []
        object_spectrum_data_integrated = []
        for fibre in fibre_list:
            object_spectrum_flux = fluxes(
                wlm,
                spectra[fibre],
                peaks[i],
                fcal=False,
                lowlow=50,
                highhigh=50,
                plot=False,
                verbose=False,
                warnings=warnings,
            )
            object_spectrum_data_gauss.append(
                object_spectrum_flux[3]
            )  # Gaussian flux is 3
            object_spectrum_data_integrated.append(
                object_spectrum_flux[7]
            )  # integrated flux is 7
        object_spectrum_data = np.nanmedian(object_spectrum_data_gauss)
        object_spectrum_data_i = np.nanmedian(object_spectrum_data_integrated)

        if fmin < old_div(object_spectrum_data, sky_spectrum_data[3]) < fmax:
            n_sky_lines_found = n_sky_lines_found + 1
            valid_peaks.append(peaks[i])
            ratio_list.append(old_div(object_spectrum_data, sky_spectrum_data[3]))
            if verbose:
                print("{:3.0f}   {:5.3f}         {:2.3f}             {:2.3f}".format(
                    n_sky_lines_found,
                    peaks[i],
                    old_div(object_spectrum_data, sky_spectrum_data[3]),
                    old_div(object_spectrum_data_i, sky_spectrum_data[7]),
                ))

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
        plt.axhline(y=fit_line, color="k", linestyle="--")
        plt.xlim(valid_wave_min - 10, valid_wave_max + 10)
        plt.ylim(np.nanmin(ratio_list) - 0.2, np.nanmax(ratio_list) + 0.2)
        plt.title("Scaling sky spectrum to object spectra")
        plt.xlabel("Wavelength [$\AA$]")
        plt.ylabel("OBJECT / SKY")
        plt.minorticks_on()
        plt.show()
        plt.close()

        if verbose:
            print("  Using this fit to scale sky spectrum to object, the median value is ", fit_line, "...")

    sky_corrected = sky_spectrum * fit_line

    #        plt.plot(wlm,sky_spectrum, "r", alpha=0.3)
    #        plt.plot(wlm,sky_corrected, "g", alpha=0.3)
    #        plt.show()
    #        plt.close()

    return sky_corrected


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def sky_spectrum_from_fibres(
    rss,
    list_spectra,
    win_sky=151,
    xmin=0,
    xmax=0,
    ymin=0,
    ymax=0,
    verbose=True,
    plot=True,
):

    if verbose:
        print("\n> Obtaining 1D sky spectrum using rss file and fibre list = ", list_spectra, " ...")

    rss.intensity_corrected = median_filter(
        rss.intensity_corrected, rss.n_spectra, rss.n_wave, win_sky=win_sky
    )

    sky = rss.plot_combined_spectrum(list_spectra=list_spectra, median=True, plot=plot)

    return sky


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def sky_spectrum_from_fibres_using_file(
    rss_file,
    fibre_list=[],
    win_sky=151,
    n_sky=0,
    skyflat="",
    apply_throughput=True,
    correct_ccd_defects=False,
    fix_wavelengths=False,
    sol=[0, 0, 0],
    xmin=0,
    xmax=0,
    ymin=0,
    ymax=0,
    verbose=True,
    plot=True,
):

    if skyflat == "":
        apply_throughput = False
        plot_rss = False
    else:
        apply_throughput = True
        plot_rss = True

    if n_sky != 0:
        sky_method = "self"
        is_sky = False
        if verbose:
            print("\n> Obtaining 1D sky spectrum using ", n_sky, " lowest fibres in this rss ...")
    else:
        sky_method = "none"
        is_sky = True
        if verbose:
            print("\n> Obtaining 1D sky spectrum using fibre list = ", fibre_list, " ...")

    _test_rss_ = KOALA_RSS(
        rss_file,
        apply_throughput=apply_throughput,
        skyflat=skyflat,
        correct_ccd_defects=correct_ccd_defects,
        fix_wavelengths=fix_wavelengths,
        sol=sol,
        sky_method=sky_method,
        n_sky=n_sky,
        is_sky=is_sky,
        win_sky=win_sky,
        do_extinction=False,
        plot=plot_rss,
        verbose=False,
    )

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
                plt.plot(
                    _test_rss_.wavelength, _test_rss_.intensity_corrected[i], alpha=0.5
                )
                plt.plot(_test_rss_.wavelength, sky, "b", linewidth=2, alpha=0.5)
            ptitle = "Sky spectrum combining " + np.str(len(fibre_list)) + " fibres"

        plot_plot(_test_rss_.wavelength, sky, ptitle=ptitle)

    print("\n> Sky spectrum obtained!")
    return sky


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def ds9_offsets(x1, y1, x2, y2, pixel_size_arc=0.6):
    """
    Print information about offsets in pixels between (x1, y1) and (x2, y2). This assumes that (x1, y1) and (x2, y2) are close on the sky and small amngle approximations are valid!

    Args:
        x1 (float): x position 1 (in pixels)
        y1 (float): y position 1 (in pixels
        x2 (float): x position 2 (in pixels)
        y2 (float): y position 2 (in pixels)
        pixel_size_arc (float, default=0.6): The pixel size in arcseconds

    Returns:
        None
    """

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
    print("  Assuming N up and E left, the telescope did an offset of ----> {:5.2f} {:1} {:5.2f} {:1}".format(
        offset_RA, direction_RA, offset_DEC, direction_DEC
    ))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def offset_positions(
    ra1h,
    ra1m,
    ra1s,
    dec1d,
    dec1m,
    dec1s,
    ra2h,
    ra2m,
    ra2s,
    dec2d,
    dec2m,
    dec2s,
    decimals=2,
):
    """
    Work out offsets between two sky positions and print them to the screen. This could probably be replaced with some astropy functions.
    TODO: Include arguments

    Returns:
        None
    """

    ra1 = ra1h + ra1m / 60.0 + ra1s / 3600.0
    ra2 = ra2h + ra2m / 60.0 + ra2s / 3600.0

    if dec1d < 0:
        dec1 = dec1d - dec1m / 60.0 - dec1s / 3600.0
    else:
        dec1 = dec1d + dec1m / 60.0 + old_div(dec1s, 3600)
    if dec2d < 0:
        dec2 = dec2d - dec2m / 60.0 - dec2s / 3600.0
    else:
        dec2 = dec2d + dec2m / 60.0 + dec2s / 3600.0

    avdec = old_div((dec1 + dec2), 2)

    deltadec = round(3600.0 * (dec2 - dec1), decimals)
    deltara = round(15 * 3600.0 * (ra2 - ra1) * (np.cos(np.radians(avdec))), decimals)

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

    print("\n> POS1: RA = {:3}h {:2}min {:2.4f}sec, DEC = {:3}d {:2}m {:2.4f}s".format(
        ra1h, ra1m, ra1s, dec1d, dec1m, dec1s
    ))
    print("  POS2: RA = {:3}h {:2}min {:2.4f}sec, DEC = {:3}d {:2}m {:2.4f}s".format(
        ra2h, ra2m, ra2s, dec2d, dec2m, dec2s
    ))

    print("\n> Offset 1 -> 2 : ", tdeltara, t_sign_deltara, "     ", tdeltadec, t_sign_deltadec)
    print("  Offset 2 -> 1 : ", tdeltara, t_sign_deltara_invert, "     ", tdeltadec, t_sign_deltadec_invert)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_spec(w, f, size=0):
    if size != 0:
        plt.figure(figsize=(size, size / 2.5))
    plt.plot(w, f)
    return


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_plot(
    x,
    y,
    xmin=0,
    xmax=0,
    ymin=0,
    ymax=0,
    ptitle="Pretty plot",
    xlabel="Wavelength [$\AA$]",
    ylabel="Flux [counts]",
    fcal=False,
    save_file="",
    frameon=False,
    loc=0,
    ncol=4,
    fig_size=0,
):

    if fig_size != 0:
        plt.figure(figsize=(fig_size, fig_size / 2.5))
        plt.plot(x, y)

    if xmin == 0:
        xmin = np.nanmin(x)
    if xmax == 0:
        xmax = np.nanmax(x)
    if ymin == 0:
        ymin = np.nanmin(y)
    if ymax == 0:
        ymax = np.nanmax(y)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(ptitle)
    plt.minorticks_on()
    if loc != 0:
        plt.legend(frameon=frameon, loc=loc, ncol=ncol)
    plt.xlabel(xlabel)
    if fcal:
        ylabel = "Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]"
    plt.ylabel(ylabel)

    if save_file == "":
        plt.show()
    else:
        plt.savefig(save_file)
    plt.close()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Definition introduced by Matt
def MAD(x):
    """
    Derive the Median Absolute Deviation of an array
    Args:
        x (array): Array of numbers to find the median of

    Returns:
        float:
    """
    MAD = np.nanmedian(np.abs(x - np.nanmedian(x)))
    return MAD / 0.6745


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def rebin_spec(wave, specin, wavnew):
    """
    Rebin a spectrum with a new wavelength array

    Args:
        wave (array): wavelength arrau
        specin (array): Input spectrum to be shifted
        shift (float): Shift. Same units as wave?

    Returns:
        New spectrum at shifted wavelength values
    """
    spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    f = np.ones(len(wave))
    filt = spectrum.ArraySpectralElement(wave, f, waveunits="angstrom")
    obs = observation.Observation(spec, filt, binset=wavnew, force="taper")
    return obs.binflux


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def rebin_spec_shift(wave, specin, shift):
    """
    Rebin a spectrum and shift in wavelength. Makes a new wavelength array and then passes this to rebin_spec

    Args:
        wave (array): wavelength arrau
        specin (array): Input spectrum to be shifted
        shift (float): Shift. Same units as wave?

    Returns:
        New spectrum at shifted wavelength values

    """
    wavnew = wave + shift
    obs = rebin_spec(wave, specin, wavnew)
    # Updating from pull request #16. rebin_spec returns a .binflux object. This function tried to create a
    # binflux.binflux object.
    return obs


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def compare_fix_2dfdr_wavelengths(rss1, rss2):
    """
    Compare small fixes we've made to the 2dFdr wavelengths between two RSS files.

    Args:
        rss1 (RSS instance): An instance of the RSS class
        rss2 (RSS instance): An instance of the RSS class

    Returns:
        None
    """

    print("\n> Comparing small fixing of the 2dFdr wavelengths between two rss...")

    xfibre = list(range(0, rss1.n_spectra))
    rss1.wavelength_parameters[0]

    a0x, a1x, a2x = (
        rss1.wavelength_parameters[0],
        rss1.wavelength_parameters[1],
        rss1.wavelength_parameters[2],
    )
    aa0x, aa1x, aa2x = (
        rss2.wavelength_parameters[0],
        rss2.wavelength_parameters[1],
        rss2.wavelength_parameters[2],
    )

    fx = a0x + a1x * np.array(xfibre) + a2x * np.array(xfibre) ** 2
    fx2 = aa0x + aa1x * np.array(xfibre) + aa2x * np.array(xfibre) ** 2
    dif = fx - fx2

    plt.figure(figsize=(10, 4))
    plt.plot(xfibre, dif)
    plot_plot(
        xfibre,
        dif,
        ptitle="Fit 1 - Fit 2",
        xmin=-20,
        xmax=1000,
        xlabel="Fibre",
        ylabel="Dif",
    )

    resolution = rss1.wavelength[1] - rss1.wavelength[0]
    error = old_div(np.nanmedian(dif), resolution) * 100.0
    print("\n> The median rms is {:8.6f} A,  resolution = {:5.2f} A,  error = {:5.3} %".format(
        np.nanmedian(dif), resolution, error
    ))








# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------



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

        print("\n\n\n======================= REDUCING KOALA data =======================\n\n")

        n_files = len(rss_list)
        sky_rss_list = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        pk = (
            "_"
            + str(int(pixel_size_arcsec))
            + "p"
            + str(int((abs(pixel_size_arcsec) - abs(int(pixel_size_arcsec))) * 10))
            + "_"
            + str(int(kernel_size_arcsec))
            + "k"
            + str(int((abs(kernel_size_arcsec) - abs(int(kernel_size_arcsec))) * 100))
        )

        print("  1. Checking input values: ")

        print("\n  - Using the following RSS files : ")
        for rss in range(n_files):
            print("    ", rss + 1, ". : ", rss_list[rss])
        self.rss_list = rss_list

        if rss_clean:
            print("\n  - These RSS files are ready to be cubed & combined, no further process required ...")

        else:

            if skyflat == "" and skyflat_list[0] == "" and skyflat_file == "":
                print("\n  - No skyflat file considered, no throughput correction will be applied.")
            else:
                if skyflat_file == "":
                    print("\n  - Using skyflat to consider throughput correction ...")
                    if skyflat != "":
                        for i in range(n_files):
                            skyflat_list[i] = skyflat
                        print("    Using same skyflat for all object files")
                    else:
                        print("    List of skyflats provided!")
                else:
                    print("\n  - Using skyflat file to derive the throughput correction ...")  # This assumes skyflat_file is the same for all the objects
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
                    print("  - Using same skyflat for all object files")

            # sky_method = "self" "1D" "2D" "none" #1Dfit"

            if sky_method == "1D" or sky_method == "1Dfit":
                if np.nanmedian(sky_spectrum) != 0:
                    for i in range(n_files):
                        sky_list[i] = sky_spectrum
                    print("\n  - Using same 1D sky spectrum provided for all object files")
                else:
                    if np.nanmedian(sky_list[0]) == 0:
                        print("\n  - 1D sky spectrum requested but not found, assuming n_sky = 50 from the same files")
                        sky_method = "self"
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
                            print("\n  - 2D sky spectra requested but not found, assuming n_sky = 50 from the same files")
                            sky_method = "self"
                    except Exception:
                        for i in range(n_files):
                            sky_rss_list[i] = sky_rss
                        print("\n  - Using same 2D sky spectra provided for all object files")

            if sky_method == "self":
                for i in range(n_files):
                    sky_list[i] = 0
                if n_sky == 0:
                    n_sky = 50
                if sky_fibres[0] == 1000:
                    print("\n  - Using n_sky =", n_sky, "to create a sky spectrum")
                else:
                    print("\n  - Using n_sky =", n_sky, "and sky_fibres =", sky_fibres, "to create a sky spectrum")

            if (
                np.nanmedian(telluric_correction) == 0
                and np.nanmedian(telluric_correction_list[0]) == 0
            ):
                print("\n  - No telluric correction considered")
            else:
                if np.nanmedian(telluric_correction_list[0]) == 0:
                    for i in range(n_files):
                        telluric_correction_list[i] = telluric_correction
                    print("\n  - Using same telluric correction for all object files")
                else:
                    print("\n  - List of telluric corrections provided!")

        if do_rss:
            print("\n  2. Reading the data stored in rss files ...")
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
            print("\n  3. Cubing without considering any flux calibration ...")
            fcal = False
        else:
            print("\n  3. Cubing applying flux calibration provided ...")
            fcal = True
            if np.nanmedian(flux_calibration) != 0:
                for i in range(n_files):
                    flux_calibration_list[i] = flux_calibration
                print("     Using same flux calibration for all object files")
            else:
                print("     List of flux calibrations provided !")

        if offsets[0] != 1000:
            print("\n  Offsets values for alignment have been given, skipping cubing no-aligned rss...")
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
                print("\n  4. Aligning individual cubes ...")
            else:
                print("\n  4. Checking given offsets data and perform cubing ...")

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
            print("\n  5. Making combined cube ...")
            print("\n> Checking individual cubes: ")
            print("   Cube         RA_centre             DEC_centre         Pix Size     Kernel Size")
            print("    1        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                self.cube1_aligned.RA_centre_deg,
                self.cube1_aligned.DEC_centre_deg,
                self.cube1_aligned.pixel_size_arcsec,
                self.cube1_aligned.kernel_size_arcsec,
            ))
            print("    2        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                self.cube2_aligned.RA_centre_deg,
                self.cube2_aligned.DEC_centre_deg,
                self.cube2_aligned.pixel_size_arcsec,
                self.cube2_aligned.kernel_size_arcsec,
            ))
            if len(rss_list) > 2:
                print("    3        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                    self.cube3_aligned.RA_centre_deg,
                    self.cube3_aligned.DEC_centre_deg,
                    self.cube3_aligned.pixel_size_arcsec,
                    self.cube3_aligned.kernel_size_arcsec,
                ))
            if len(rss_list) > 3:
                print("    4        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                    self.cube4_aligned.RA_centre_deg,
                    self.cube4_aligned.DEC_centre_deg,
                    self.cube4_aligned.pixel_size_arcsec,
                    self.cube4_aligned.kernel_size_arcsec,
                ))
            if len(rss_list) > 4:
                print("    5        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                    self.cube5_aligned.RA_centre_deg,
                    self.cube5_aligned.DEC_centre_deg,
                    self.cube5_aligned.pixel_size_arcsec,
                    self.cube5_aligned.kernel_size_arcsec,
                ))
            if len(rss_list) > 5:
                print("    6        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                    self.cube6_aligned.RA_centre_deg,
                    self.cube6_aligned.DEC_centre_deg,
                    self.cube6_aligned.pixel_size_arcsec,
                    self.cube6_aligned.kernel_size_arcsec,
                ))
            if len(rss_list) > 6:
                print("    7        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                    self.cube7_aligned.RA_centre_deg,
                    self.cube7_aligned.DEC_centre_deg,
                    self.cube7_aligned.pixel_size_arcsec,
                    self.cube7_aligned.kernel_size_arcsec,
                ))
            if len(rss_list) > 7:
                print("    8        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                    self.cube8_aligned.RA_centre_deg,
                    self.cube8_aligned.DEC_centre_deg,
                    self.cube8_aligned.pixel_size_arcsec,
                    self.cube8_aligned.kernel_size_arcsec,
                ))
            if len(rss_list) > 8:
                print("    9        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                    self.cube9_aligned.RA_centre_deg,
                    self.cube9_aligned.DEC_centre_deg,
                    self.cube9_aligned.pixel_size_arcsec,
                    self.cube9_aligned.kernel_size_arcsec,
                ))
            if len(rss_list) > 9:
                print("   10        {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(
                    self.cube10_aligned.RA_centre_deg,
                    self.cube10_aligned.DEC_centre_deg,
                    self.cube10_aligned.pixel_size_arcsec,
                    self.cube10_aligned.kernel_size_arcsec,
                ))

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

            print("\n> Combining cubes...")
            if len(rss_list) == 2:
                if ADR:
                    print("  Using data corrected for ADR to get combined cube...")
                    self.combined_cube.data = np.nanmedian(
                        [self.cube1_aligned.data_ADR, self.cube2_aligned.data_ADR],
                        axis=0,
                    )
                else:
                    print("  No ADR correction considered...")
                    self.combined_cube.data = np.nanmedian(
                        [self.cube1_aligned.data, self.cube2_aligned.data], axis=0
                    )
                self.combined_cube.PA = np.mean(
                    [self.cube1_aligned.PA, self.cube2_aligned.PA]
                )

            if len(rss_list) == 3:
                if ADR:
                    print("  Using data corrected for ADR to get combined cube...")
                    self.combined_cube.data = np.nanmedian(
                        [
                            self.cube1_aligned.data_ADR,
                            self.cube2_aligned.data_ADR,
                            self.cube3_aligned.data_ADR,
                        ],
                        axis=0,
                    )
                else:
                    print("  No ADR correction considered...")
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
                    print("  Using data corrected for ADR to get combined cube...")
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
                    print("  No ADR correction considered...")
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
                    print("  Using data corrected for ADR to get combined cube...")
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
                    print("  No ADR correction considered...")
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
                    print("  Using data corrected for ADR to get combined cube...")
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
                    print("  No ADR correction considered...")
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
                    print("  Using data corrected for ADR to get combined cube...")
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
                    print("  No ADR correction considered...")
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
                    print("  Using data corrected for ADR to get combined cube...")
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
                    print("  No ADR correction considered...")
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
                    print("  Using data corrected for ADR to get combined cube...")
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
                    print("  No ADR correction considered...")
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
                    print("  Using data corrected for ADR to get combined cube...")
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
                    print("  No ADR correction considered...")
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
                print("  Flux calibration not considered")
                fcal = False
            else:
                self.combined_cube.flux_calibration = flux_calibration
                print("  Flux calibration included!")
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

            print("\n  Total exposition time = ", self.combined_cube.total_exptime, "seconds adding the ", len(
                rss_list
            ), " files")

        # Save it to a fits file

        if save_aligned_cubes:
            print("\n  Saving aligned cubes to fits files ...")
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
            print("\n  As requested, the combined cube will not be saved to a fits file")
        else:
            print("\n  6. Saving combined cube to a fits file ...")

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
# MAIN & TESTS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
