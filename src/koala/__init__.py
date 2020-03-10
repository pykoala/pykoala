#!/usr/bin/python
# -*- coding: utf-8 -*-
# # PyKOALA: KOALA data processing and analysis
# by Angel Lopez-Sanchez and Yago Ascasibar
# Extra work by Ben Lawson (MQ PACE student)
# Plus Taylah and Matt (sky subtraction)
from __future__ import absolute_import, division, print_function
from past.utils import old_div
version = "Version 0.72 - 13th February 2020"

from .utils.io import read_table, save_rss_fits, save_fits_file
from .utils.utils import FitsExt, FitsFibresIFUIndex
from .utils.cube_alignment import offset_between_cubes, compare_cubes, align_n_cubes
from .utils.flux import search_peaks, fluxes, dfluxes, substract_given_gaussian
from .utils.sky_spectrum import scale_sky_spectrum, median_filter

import copy
import os.path as pth
import sys

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.io import fits
from astropy.wcs import WCS

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

from synphot import observation
from synphot import spectrum

from scipy import interpolate
from scipy.ndimage.interpolation import shift
from scipy.optimize import curve_fit
import scipy.signal as sig

# -----------------------------------------------------------------------------
# Import Python routines
# -----------------------------------------------------------------------------

from .utils.plots import (
    plot_redshift_peaks, plot_weights_for_getting_smooth_spectrum,
    plot_correction_in_fibre_p_fibre, plot_suspicious_fibres_graph, plot_skyline_5578,
    plot_offset_between_cubes, plot_response, plot_telluric_correction, plot_plot
)
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

# -----------------------------------------------------------------------------
# Define constants
# -----------------------------------------------------------------------------

DATA_PATH = pth.join(pth.dirname(__file__), "data")

from .constants import C, PARSEC as pc


# -----------------------------------------------------------------------------
# Define COLOUR scales
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
fuego_color_map.set_bad("lightgray")
plt.register_cmap(cmap=fuego_color_map)

projo = [0.25, 0.5, 1, 1.0, 1.00, 1, 1]
pverde = [0.00, 0.0, 0, 0.5, 0.75, 1, 1]
pazul = [0.00, 0.0, 0, 0.0, 0.00, 0, 1]

# -----------------------------------------------------------------------------
# RSS CLASS
# -----------------------------------------------------------------------------


class RSS(object):
    """
    Collection of row-stacked spectra (RSS).

    Attributes
    ----------
    wavelength: np.array(float)
      Wavelength, in Angstroms.
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
        Compute the integrated flux of a fibre in a particular range, valid_wave_min to valid_wave_max.

        Parameters
        ----------
        list_spectra: float (default "all")
            list with the number of fibres for computing integrated value
            if using "all" it does all fibres
        valid_wave_min, valid_wave_max :  float
            the integrated flux value will be computed in the range [valid_wave_min, valid_wave_max]
            (default = , if they all 0 we use [self.valid_wave_min, self.valid_wave_max]
        min_value: float (default 0)
            For values lower than min_value, we set them as min_value
        plot : Boolean (default = False)
            Plot
        title : string
            Title for the plot
        text: string
            A bit of extra text
        warnings : Boolean (default = False)
            Write warnings, e.g. when the integrated flux is negative
        correct_negative_sky : Boolean (default = False)
            Corrects negative values making 0 the integrated flux of the lowest fibre

        Example
        ----------
        integrated_fibre_6500_6600 = star1r.compute_integrated_fibre(valid_wave_min=6500, valid_wave_max=6600,
        title = " - [6500,6600]", plot = True)
        """

        print("\n  Computing integrated fibre values {}".format(text))

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
                    print(
                        "  WARNING: The integrated flux in fibre {:4} is negative, flux/wave = {:10.2f}, (probably sky), CHECK !".format(
                            i, self.integrated_fibre[i]/waves_in_region
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
                self.integrated_fibre/waves_in_region
            )
            for fibre_ in range(n_negative_fibres):
                negative_fibres_sorted.append(integrated_intensity_sorted[fibre_])
            # print "\n> Checking results using",n_negative_fibres,"fibres with the lowest integrated intensity"
            # print "  which are :",negative_fibres_sorted

            if correct_negative_sky:
                min_sky_value = self.integrated_fibre[negative_fibres_sorted[0]]
                min_sky_value_per_wave = min_sky_value/waves_in_region
                print(
                    "\n> Correcting negative values making 0 the integrated flux of the lowest fibre, which is {:4} with {:10.2f} counts/wave".format(
                        negative_fibres_sorted[0], min_sky_value_per_wave
                    ))
                # print self.integrated_fibre[negative_fibres_sorted[0]]
                self.integrated_fibre = self.integrated_fibre - min_sky_value
                for i in range(self.n_spectra):
                    self.intensity_corrected[i] = (
                            self.intensity_corrected[i] - min_sky_value_per_wave
                    )

            else:
                print(
                    "\n> Adopting integrated flux = {:5.2f} for all fibres with negative integrated flux (for presentation purposes)".format(
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
        Identify fibres with highest intensity (high_fibres=10).
        Add all in a single spectrum.
        Identify emission features.
        These emission features should be those expected in all the cube!
        Also, choosing fibre=number, it identifies el in a particular fibre.

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
                print("\n> Identifying emission lines using the {} fibres with the highest integrated intensity".format(high_fibres))
                print("  which are : {}".format(region))
            combined_high_spectrum = np.nansum(self.intensity_corrected[region], axis=0)
        else:
            combined_high_spectrum = self.intensity_corrected[fibre]
            if verbose:
                print("\n> Identifying emission lines in fibre {}".format(fibre))

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
    	Task for correcting high cosmics and CCD defects using median values of nearby pixels.
        2dFdr corrects for (the majority) of the cosmic rays, usually correct_high_cosmics = False.
        ANGEL COMMENT: Check, probably can be improved using MATT median running + plotting outside

        Parameters
        ----------
        rect_high_cosmics: boolean (default = False)
    		Correct ONLY CCD defects
        re_p: integer (default = 0)
    		Plots the corrections in fibre fibre_p
        ove_5578: boolean (default = False)
    		Removes skyline 5578 (blue spectrum) using Gaussian fit
            ND CHECK: This also MODIFIES the throughput correction correcting for flux_5578_medfilt /median_flux_5578_medfilt
        step: integer (default = 50)
    	    Number of points for calculating median value
        clip_high : float (default = 100)
    		Minimum value of flux/median in a pixel to be consider as a cosmic
		    if s[wave] > clip_high*fit_median[wave] -> IT IS A COSMIC
        verbose: boolean (default = False)
            Write results 
        warnings: boolean (default = False)
            Write warnings
        plot: boolean (default = False)
            Plot results
        plot_suspicious_fibres: boolean (default = False)
    	    Plots fibre(s) that could have a cosmic left (but it could be OK)
            IF self.integrated_fibre[fibre]/median_running[fibre] > max_value  -> SUSPICIOUS FIBRE

        Example
        ----------
    	self.correct_high_cosmics_and_defects(correct_high_cosmics=False, step=40, remove_5578 = True,
                                              clip_high=120, plot_suspicious_fibres=True, warnings=True, 									      verbose=False, plot=True)
	    """
        print("\n> Correcting for high cosmics and CCD defects...")

        wave_min = self.valid_wave_min  # CHECK ALL OF THIS...
        wave_max = self.valid_wave_max
        wlm = self.wavelength

        if correct_high_cosmics == False:
            print("  Only CCD defects (nan and negative values) are considered.")
        else:
            print("  Using clip_high = {} for high cosmics".format(clip_high))
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
            cuts = np.int(self.n_wave/step)  # using np.int instead of // for improved readability
            for cut in range(cuts):
                if cut == 0:
                    next_wave = wave_min
                else:
                    next_wave = np.nanmedian(
                        (wlm[np.int(cut * step)] + wlm[np.int((cut + 1) * step)])/2
                    )

                if next_wave < wave_max:
                    running_wave.append(next_wave)
                    # print("SEARCHFORME1", step, running_wave[cut])
                    region = np.where(
                        (wlm > running_wave[cut] - np.int(step/2))   # step/2 doesn't need to be an int, but probably
                        & (wlm < running_wave[cut] + np.int(step/2))   # want it to be so the cuts are uniform.
                    )
                    # print('SEARCHFORME3', region)
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
                            print("  "
                                  "CLIPPING HIGH = {} in fibre {} w = {} value= {} v/median= {}".format(clip_high, fibre, wlm[wave], s[wave], s[wave]/fit_median[wave]))  # " median=",fit_median[wave]
                        s[wave] = fit_median[wave]

            if fibre == fibre_p:
                espectro_new = copy.copy(s)
            max_ratio_list.append(np.nanmax(s/fit_median))
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
        print("\n  Maximum value found of flux/continuum = {}".format(max_ratio))
        if correct_high_cosmics:
            print("  Recommended value for clip_high = {} , here we used {}".format(int(max_ratio + 1), clip_high))

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
                            fibre - np.int(step_f/2): fibre + np.int(step_f/2)  # np.int is used instead of // of readability
                        ]
                    )
                median_running.append(median_value)
                if self.integrated_fibre[fibre]/median_running[fibre] > max_value:
                    print("  Fibre {} has a integrated/median ratio of {}    -> Might be a cosmic left!".format(fibre, self.integrated_fibre[fibre]/median_running[fibre]))
                    label = np.str(fibre)
                    plt.axvline(x=fibre, color="k", linestyle="--")
                    plt.text(fibre, self.integrated_fibre[fibre] / 2.0, label)
                    suspicious_fibres.append(fibre)
                skip = 0

            plt.plot(self.integrated_fibre, label="Corrected", alpha=0.6)
            plt.plot(median_running, "k", label="Median", alpha=0.6)
            plt.legend(frameon=False, loc=1, ncol=3)
            plt.minorticks_on()
            #plt.show()
            #plt.close()

        if plot_suspicious_fibres == True and len(suspicious_fibres) > 0:
            # Plotting suspicious fibres..
            figures = plot_suspicious_fibres_graph(
                self,
                suspicious_fibres,
                fig_size,
                wave_min,
                wave_max,
                intensity_corrected_fiber=self.intensity_corrected)

        if remove_5578 and wave_min < 5578:
            print("  Skyline 5578 has been removed. Checking throughput correction...")
            flux_5578_medfilt = sig.medfilt(flux_5578, np.int(5))
            median_flux_5578_medfilt = np.nanmedian(flux_5578_medfilt)
            extra_throughput_correction = flux_5578_medfilt/median_flux_5578_medfilt
            # plt.plot(extra_throughput_correction)
            # plt.show()
            # plt.close()
            if plot:
                fig = plot_skyline_5578(fig_size, flux_5578, flux_5578_medfilt)

            print("  Variations in throughput between {} and {} ".format(
                np.nanmin(extra_throughput_correction), np.nanmax(extra_throughput_correction)
            ))
            print("  Applying this extra throughtput correction to all fibres...")

            for i in range(self.n_spectra):
                self.intensity_corrected[i, :] = (
                    self.intensity_corrected[i, :]/extra_throughput_correction[i]
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
        """
        This task HAVE TO BE USED WITH EXTREME CARE 
        as it has not been properly tested!!!
        It CAN DELETE REAL (faint) ABSORPTION/EMISSION features in spectra!!!
        Use the "1dfit" option for getting a better sky substraction
        ANGEL is keeping this here just in case it is eventually useful...
        """
        
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
                                    print("Double overlap  {}  {}".format(exclude_ranges_low[-1], exclude_ranges_high[-1]))
                            else:
                                exclude_ranges_low.append(exclude_ranges_low_[i])
                                exclude_ranges_high.append(exclude_ranges_high_[i + 1])
                                skip_next = 1
                                if verbose:
                                    print("Overlap  {}  {}".format(exclude_ranges_low[-1], exclude_ranges_high[-1]))
                    else:
                        exclude_ranges_low.append(exclude_ranges_low_[i])
                        exclude_ranges_high.append(exclude_ranges_high_[i])
                        if verbose:
                            print("Overlap  {}  {}".format(exclude_ranges_low[-1], exclude_ranges_high[-1]))
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
            print("  Checking fibre {} (only this fibre is corrected, use fibre = 0 for all)...".format(fibre))
            plot = True
        else:
            f_i = 0
            f_f = self.n_spectra
        for fibre in range(f_i, f_f):  # (self.n_spectra):
            if fibre == say_status:
                print("  Checking fibre {}  ...".format(fibre))
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
                        print("  Excluding range [ {} , {} ] as it has an emission line".format(
                            exclude_ranges_low[rango], exclude_ranges_high[rango]))
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
                            print("  Found P-Cygni-like feature in {}".format(wlm[i]))
                    if disp[i] > dispersion * dclip or disp[i] < -dispersion * dclip:
                        s[i] = fit_median[i]
                        if verbose:
                            print("  Clipping feature in {}".format(wlm[i]))

                    if wlm[i] > exclude_ranges_high[rango] and imprimir == 0:
                        if verbose:
                            print("  Checked {}  End range {} {} {}".format(
                                wlm[i], rango,
                                exclude_ranges_low[rango],
                                exclude_ranges_high[rango]
                                )
                            )
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
                    # plt.show()
                    # plt.close()

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

        redshift = brightest_line_wavelength/brightest_line_wavelength_rest - 1.0

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
            print("  Checking fibre {} (only this fibre is corrected, use fibre = 0 for all)...".format(fibre))
            plot = True
            verbose = True
            warnings = True
        else:
            f_i = 0
            f_f = self.n_spectra
        for fibre in range(f_i, f_f):  # (self.n_spectra):
            if fibre == say_status:
                print("  Checking fibre {:4} ...  ({:6.2f} % completed) ...".format(
                    fibre,
                    fibre * 100.0 / self.n_spectra
                    )
                )
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
                            print("  Line {} blended with {}".format(sl_center[i], dsky2[di]))
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
                        print("  SKY line {} in EMISSION LINE !".format(sl_center[i]))
                    skip_sl_fit.append(True)
                else:
                    skip_sl_fit.append(False)

                # print "  Fitted wavelength for sky line ",sl_center[i]," : ",resultado[1],"   ",resultado[5]
                if plot_fit:
                    if verbose:
                        print("  Fitted wavelength for sky line {} : {}   sigma = {}".format(
                            sl_center[i], sl_gauss_center[i], sl_gaussian_sigma[i])
                        )
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
                print("\n> Performing Gaussian fitting to sky lines in fibre {} of object data...".format(fibre))

            for i in range(number_sl):
                if sl_fnl[i] == 0:
                    plot_fit = False
                else:
                    plot_fit = True
                if skip_sl_fit[i]:
                    if verbose:
                        print(" SKIPPING SKY LINE {} as located within the range of an emission line!".format(
                            sl_center[i]))
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
                                print("  Line  {} blended with {}".format(sl_center[i], dsky2[di]))
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
                                print("  Bad fit for {}! ignoring it...".format(sl_center[i]))
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
                                print("  Bad fit for {}! ignoring it...".format(sl_center[i]))
                            object_sl_gaussian_flux.append(float("nan"))
                            object_sl_gaussian_center.append(float("nan"))
                            object_sl_gaussian_sigma.append(float("nan"))
                            dif_center_obj_sky.append(float("nan"))
                            skip_sl_fit[i] = True  # We don't substract this fit

                ratio_object_sky_sl_gaussian.append(
                    old_div(object_sl_gaussian_flux[i], sl_gaussian_flux[i])
                )  # TODO: to remove once sky_line_fitting is active and we can do 1Dfit

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
                        print("- Small correction of center wavelength of sky line {}  :  {}".format(
                            sl_center[i], small_center_correction))

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
                                print("  This was a double sky line, also substracting {} at {}".format(
                                    dsky2[di], np.array(dsky2[di]) + small_center_correction))
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
                # plt.show()
                # plt.close()

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
                print("\n> Median center offset between OBJ and SKY : {} A\n> Median gauss for the OBJECT {}  A".format(offset, np.nanmedian(object_sl_gaussian_sigma)))
                print("> Median flux OBJECT / SKY = {}".format(np.nanmedian(ratio_object_sky_sl_gaussian)))

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
                    print("\n> Rebinning the spectrum of fibre {} to match sky spectrum...".format(fibre))
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
    def do_extinction_curve(
        self, observatory_file=pth.join(DATA_PATH, "ssoextinct.dat"), plot=True
    ):

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
            plt.title("Correction for extinction using airmass = {}".format(self.airmass))
            plt.ylabel("Flux correction")
            plt.xlabel("Wavelength [$\AA$]")
            # plt.show()
            # plt.close()

        # Correct for extinction at given airmass
        print("  Airmass = {}".format(self.airmass))
        print("  Observatory file with extinction curve : {}".format(observatory_file))
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
            print("\n> Identifying sky spaxels using the lowest integrated values in the [ {} , {}] range ...".format(sky_wave_min, sky_wave_max))

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
            print("  We use the lowest {} fibres for getting sky. Their positions are:".format(optimal_n))
            # Compute sky spectrum and plot it
            self.sky_fibres = sorted_by_flux[:optimal_n]
            self.sky_emission = np.nanmedian(
                intensidad[sorted_by_flux[:optimal_n]], axis=0
            )
            print("  List of fibres used for sky saved in self.sky_fibres")

        else:  # We provide a list with sky positions
            print("  We use the list provided to get the sky spectrum")
            print("  sky_fibres = {}".format(sky_fibres))
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
            plt.title("{} - Combined Sky Spectrum".format(self.description))
            plt.legend(frameon=False)
            # plt.show()
            # plt.close()

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
            # plt.show()
            # plt.close()

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
            # print("  For {}, we use the median value in the [{}, {}] range.".format(
            #     self.grating, wave_min_scale, wave_max_scale))
        else:
            if wave_min_scale == 0:
                wave_min_scale = self.wavelength[0]
            if wave_max_scale == 0:
                wave_max_scale = self.wavelength[-1]
            print("  As given by the user, we use the median value in the [{} , {}] range.".format(wave_min_scale, wave_max_scale))

        median_region = np.zeros(self.n_spectra)
        for i in range(self.n_spectra):
            region = np.where(
                (self.wavelength > wave_min_scale) & (self.wavelength < wave_max_scale)
            )
            median_region[i] = np.nanmedian(self.intensity[i, region])

        median_value_skyflat = np.nanmedian(median_region)
        self.relative_throughput = median_region/median_value_skyflat
        print("  Median value of skyflat in the [ {} , {}] range = {}".format(wave_min_scale, wave_max_scale, median_value_skyflat))
        print("  Individual fibre corrections:  min = {}  max = {}".format(np.nanmin(self.relative_throughput), np.nanmax(self.relative_throughput)))

        if plot:
            plt.figure(figsize=(10, 4))
            x = list(range(self.n_spectra))
            plt.plot(x, self.relative_throughput)
            # plt.ylim(0.5,4)
            plt.minorticks_on()
            plt.xlabel("Fibre")
            plt.ylabel("Throughput using scale")
            plt.title("Throughput correction using scale")
            # plt.show()
            # plt.close()

            # print "\n  Plotting spectra WITHOUT considering throughput correction..."
            plt.figure(figsize=(10, 4))
            for i in range(self.n_spectra):
                plt.plot(self.wavelength, self.intensity[i, ])
            plt.xlabel("Wavelength [$\AA$]")
            plt.title("Spectra WITHOUT considering any throughput correction")
            plt.xlim(self.wavelength[0] - 10, self.wavelength[-1] + 10)
            plt.ylim(ymin, ymax)
            plt.minorticks_on()
            # plt.show()
            # plt.close()

            # print "  Plotting spectra CONSIDERING throughput correction..."
            plt.figure(figsize=(10, 4))
            for i in range(self.n_spectra):
                # self.intensity_corrected[i,] = self.intensity[i,] * self.relative_throughput[i]
                plot_this = self.intensity[i, ]/self.relative_throughput[i]
                plt.plot(self.wavelength, plot_this)
            plt.ylim(ymin, ymax)
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            plt.title("Spectra CONSIDERING throughput correction (scale)")
            plt.xlim(self.wavelength[0] - 10, self.wavelength[-1] + 10)
            plt.axvline(x=wave_min_scale, color="k", linestyle="--")
            plt.axvline(x=wave_max_scale, color="k", linestyle="--")
            # plt.show()
            # plt.close()

        print("\n>  Using median value of skyflat considering a median filter of {} ...".format(kernel_sky_spectrum))  # LUKE
        median_sky_spectrum = np.nanmedian(self.intensity, axis=0)
        self.response_sky_spectrum = np.zeros_like(self.intensity)
        rms = np.zeros(self.n_spectra)
        plot_fibres = [100, 500, 501, 900]
        pf = 0
        for i in range(self.n_spectra):
            self.response_sky_spectrum[i] = (
                (self.intensity[i]/self.relative_throughput[i])/median_sky_spectrum
            )
            filter_response_sky_spectrum = sig.medfilt(
                self.response_sky_spectrum[i], kernel_size=kernel_sky_spectrum
            )
            rms[i] = np.nansum(
                np.abs(self.response_sky_spectrum[i] - filter_response_sky_spectrum)
            )/np.nansum(self.response_sky_spectrum[i])

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
                        self.response_sky_spectrum[i]/filter_response_sky_spectrum,
                        alpha=1,
                        label="Normalized Skyflat",
                    )
                    plt.xlim(self.wavelength[0] - 50, self.wavelength[-1] + 50)
                    plt.ylim(0.95, 1.05)
                    ptitle = "Fibre {} with rms = {}".format(i, rms[i])
                    plt.title(ptitle)
                    plt.xlabel("Wavelength [$\AA$]")
                    plt.legend(frameon=False, loc=3, ncol=1)
                    # plt.show()
                    # plt.close()
                    if pf < len(plot_fibres) - 1:
                        pf = pf + 1

        print("  median rms = {} min rms = {}    max rms = {}".format(np.nanmedian(rms), np.nanmin(rms),np.nanmax(rms)))
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
        """  # TODO BLAKE: always use false, use plots to make sure it's good. prob just save as a different file.
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
                telluric_correction[l] = smooth_med_star[l]/estrella[l]  # TODO: should be float, check when have star data

        if plot:
            plt.figure(figsize=(fig_size, fig_size / 2.5))
            if combined_cube:
                print("  Telluric correction for this star ({}) :".format(self.combined_cube.object))
                plt.plot(wlm, estrella, color="b", alpha=0.3)
                plt.plot(wlm, estrella * telluric_correction, color="g", alpha=0.5)
                plt.ylim(np.nanmin(estrella), np.nanmax(estrella))

            else:
                print("  Example of telluric correction using fibres {} and {}  :".format(region[0], region[1]))

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
            # plt.show()
            # plt.close()

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
            # plt.show()
            # plt.close()

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
            #plt.show()
            pass
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
                #plt.show()
                pass
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
        # TODO: can remove old_div once this function is understood, currently not called in whole module.
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
        cbar.set_label(color_bar_text, rotation=90, labelpad=40)
        cbar.ax.tick_params()

        # plt.show()
        # plt.close()

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
        cbar.set_label(color_bar_text, rotation=90, labelpad=labelpad)
        cbar.ax.tick_params()

        # plt.show()
        # plt.close()

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
        plt.ylim([I_ymin - (I_rango/10), I_ymax + (I_rango/10)])
        plt.title(
            self.object
            + " - Combined spectrum - "
            + "{}".format(high_fibres)
            + " fibres with highest intensity"
        )
        plt.legend(frameon=False, loc=4, ncol=2)
        # plt.show()
        # plt.close()

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
                print("  Checking fibre {} (only this fibre is corrected, use fibre = 0 for all)...".format(fibre))
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
                            print("  Bad fitting for {} ... ignoring this fit...".format(sl_center[i]))
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

        print("  a0x = {}    a1x = {}     a2x = {}".format(a0x, a1x, a2x))
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
    <https://aat.anu.edu.au/science/instruments/current/AAOmega/reduction>`_
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
        self.object = RSS_fits_file[FitsExt.main].header["OBJECT"]
        self.description = self.object + " - " + filename
        self.RA_centre_deg = RSS_fits_file[FitsExt.fibres_ifu].header["CENRA"] * 180/np.pi
        self.DEC_centre_deg = RSS_fits_file[FitsExt.fibres_ifu].header["CENDEC"] * 180/np.pi
        self.exptime = RSS_fits_file[FitsExt.main].header["EXPOSED"]
        #  WARNING: Something is probably wrong/inaccurate here!
        #  Nominal offsets between pointings are totally wrong!

        # Read good/bad spaxels
        all_spaxels = list(range(len(RSS_fits_file[FitsExt.fibres_ifu].data)))
        quality_flag = [RSS_fits_file[FitsExt.fibres_ifu].data[i][FitsFibresIFUIndex.quality_flag] for i in all_spaxels]
        good_spaxels = [i for i in all_spaxels if quality_flag[i] == 1]
        bad_spaxels = [i for i in all_spaxels if quality_flag[i] == 0]

        #        for i in range(1):
        #            print i, RSS_fits_file[2]
        #

        # Create wavelength, intensity, and variance arrays only for good spaxels
        wcsKOALA = WCS(RSS_fits_file[FitsExt.main].header)
        # variance = RSS_fits_file[1].data[good_spaxels]
        index_wave = np.arange(RSS_fits_file[FitsExt.main].header["NAXIS1"])
        wavelength = wcsKOALA.dropaxis(1).wcs_pix2world(index_wave, 0)[0]
        intensity = RSS_fits_file[FitsExt.main].data[good_spaxels]

        print("\n  Number of spectra in this RSS = {},  number of good spectra = {} ,  number of bad spectra ={}".format(
            len(RSS_fits_file[FitsExt.main].data), len(good_spaxels), len(bad_spaxels)))
        print("  Bad fibres = {}".format(bad_spaxels))

        # Read errors using RSS_fits_file[1]
        # self.header1 = RSS_fits_file[1].data      # CHECK WHEN DOING ERRORS !!!

        # Read spaxel positions on sky using RSS_fits_file[2]
        self.header2_data = RSS_fits_file[FitsExt.fibres_ifu].data

        # CAREFUL !! header 2 has the info of BAD fibres, if we are reading from our created RSS files we have to do it in a different way...

        # print RSS_fits_file[2].data

        if len(bad_spaxels) == 0:
            offset_RA_arcsec_ = []
            offset_DEC_arcsec_ = []
            for i in range(len(good_spaxels)):
                offset_RA_arcsec_.append(self.header2_data[i][FitsFibresIFUIndex.ra_offset])
                offset_DEC_arcsec_.append(self.header2_data[i][FitsFibresIFUIndex.dec_offset])
            offset_RA_arcsec = np.array(offset_RA_arcsec_)
            offset_DEC_arcsec = np.array(offset_DEC_arcsec_)
            variance = np.zeros_like(intensity)  # CHECK FOR ERRORS

        else:
            offset_RA_arcsec = np.array(
                [RSS_fits_file[FitsExt.fibres_ifu].data[i][FitsFibresIFUIndex.ra_offset] for i in good_spaxels]
            )
            offset_DEC_arcsec = np.array(
                [RSS_fits_file[FitsExt.fibres_ifu].data[i][FitsFibresIFUIndex.dec_offset] for i in good_spaxels]
            )

            self.ID = np.array(
                [RSS_fits_file[FitsExt.fibres_ifu].data[i][FitsFibresIFUIndex.spec_id] for i in good_spaxels]
            )  # These are the good fibres
            variance = RSS_fits_file[FitsExt.var].data[good_spaxels]  # CHECK FOR ERRORS

        self.ZDSTART = RSS_fits_file[FitsExt.main].header["ZDSTART"]  # Zenith distance (degrees?)
        self.ZDEND = RSS_fits_file[FitsExt.main].header["ZDEND"]
        # KOALA-specific stuff
        self.PA = RSS_fits_file[FitsExt.main].header["TEL_PA"]   # Position angle?
        self.grating = RSS_fits_file[FitsExt.main].header["GRATID"]
        # Check RED / BLUE arm for AAOmega
        if RSS_fits_file[FitsExt.main].header["SPECTID"] == "RD":
            AAOmega_Arm = "RED"
        if RSS_fits_file[FitsExt.main].header["SPECTID"] == "BL":
            AAOmega_Arm = "BLUE"

        # For WCS
        self.CRVAL1_CDELT1_CRPIX1 = []
        self.CRVAL1_CDELT1_CRPIX1.append(RSS_fits_file[FitsExt.main].header["CRVAL1"])  # see https://idlastro.gsfc.nasa.gov/ftp/pro/astrom/aaareadme.txt maybe?
        self.CRVAL1_CDELT1_CRPIX1.append(RSS_fits_file[FitsExt.main].header["CDELT1"])
        self.CRVAL1_CDELT1_CRPIX1.append(RSS_fits_file[FitsExt.main].header["CRPIX1"])

        # SET RSS
        # FROM HERE IT WAS self.set_data before   ------------------------------------------

        self.wavelength = wavelength
        self.n_wave = len(wavelength)

        # Check that dimensions match KOALA numbers
        if self.n_wave != 2048 and len(all_spaxels) != 1000:
            print("\n *** WARNING *** : These numbers are NOT the standard ones for KOALA")

        print("\n> Setting the data for this file:")

        if variance.shape != intensity.shape:
            print("\n* ERROR: * the intensity and variance matrices are {} and {} respectively\n".format(intensity.shape, variance.shape))
            raise ValueError
        n_dim = len(intensity.shape)
        if n_dim == 2:
            self.intensity = intensity
            self.variance = variance
        elif n_dim == 1:
            self.intensity = intensity.reshape((1, self.n_wave))
            self.variance = variance.reshape((1, self.n_wave))
        else:
            print("\n* ERROR: * the intensity matrix supplied has {} dimensions\n".format(n_dim))
            raise ValueError

        self.n_spectra = self.intensity.shape[0]
        self.n_wave = len(self.wavelength)
        print("  Found {} spectra with {} wavelengths".format(
            self.n_spectra, self.n_wave
        ), "between {:.2f} and {:.2f} Angstrom".format(
            self.wavelength[0], self.wavelength[-1]
        ))
        if self.intensity.shape[1] != self.n_wave:
            print("\n* ERROR: * spectra have {} wavelengths rather than {}".format(self.intensity.shape[1], self.n_wave))
            raise ValueError
        if (
            len(offset_RA_arcsec) != self.n_spectra
            or len(offset_DEC_arcsec) != self.n_spectra
        ):
            print("\n* ERROR: * offsets (RA, DEC) = ({},{}) rather than {}".format(
                len(self.offset_RA_arcsec), len(self.offset_DEC_arcsec), self.n_spectra
                )
            )
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
            print("  As specified, we use the [ {} , {} ] range.".format(self.valid_wave_min, self.valid_wave_max))

        # Plot RSS_image
        if plot:
            self.RSS_image(image=self.intensity, cmap="binary_r")

        # Deep copy of intensity into intensity_corrected
        self.intensity_corrected = copy.deepcopy(self.intensity)

        # Divide by flatfield if needed
        if flat != "":
            print("\n> Dividing the data by the flatfield provided...")
            self.intensity_corrected = (self.intensity_corrected/flat.intensity_corrected)  # todo: check division per pixel works.

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
                # plt.show()
                # plt.close()

            print("\n> Applying relative throughput correction using median skyflat values per fibre...")
            self.relative_throughput = skyflat.relative_throughput
            self.response_sky_spectrum = skyflat.response_sky_spectrum
            for i in range(self.n_spectra):
                self.intensity_corrected[i, :] = (
                    self.intensity_corrected[i, :]/self.relative_throughput[i]
                )

            if nskyflat:
                print("\n  IMPORTANT: We are dividing intensity data by the sky.response_sky_spectrum !!! ")
                print("  This is kind of a flat, the changes are between {} and {}".format(
                    np.nanmin(skyflat.response_sky_spectrum), np.nanmax(skyflat.response_sky_spectrum)))
                print(" ")
                self.intensity_corrected = (
                    self.intensity_corrected/self.response_sky_spectrum
                )

            if plot_skyflat:
                plt.figure(figsize=(10, 4))
                for i in range(self.n_spectra):
                    plt.plot(self.wavelength, self.intensity_corrected[i, ])
                plt.ylim(0, 200 * np.nanmedian(self.intensity_corrected))
                plt.minorticks_on()
                plt.xlabel("Wavelength [$\AA$]")
                plt.title("Spectra CONSIDERING throughput correction (median value per fibre)")
                # plt.show()
                # plt.close()

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
                    print("  Sky spectrum scaled by {}".format(scale_sky_1D))
                sky = np.array(sky_spectrum) * scale_sky_1D
                print("  Sky spectrum provided = {}".format(sky))
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
                        print("  As requested, we scale the given 1D spectrum by {}".format(scale_sky_1D))

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
                        # plt.show()
                        # plt.close()
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
                        print("  ... first checking {} ...".format(sky_line))
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

                        scale_per_fibre[fibre_sky] = old_div(skyline_spec[3], skyline_sky[3])    # TODO: get data for 2D and test if can remove
                        self.sky_emission[fibre_sky] = skyline_sky[11]

                    if sky_line_2 != 0:
                        print("  ... now checking {} ...".format(sky_line_2))
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
                                old_div(skyline_spec[3], skyline_sky[3])  # TODO: get data for 2D and test if can remove
                            )
                            self.sky_emission[fibre_sky] = skyline_sky[11]

                    # Median value of scale_per_fibre, and apply that value to all fibres
                    if sky_line_2 == 0:
                        scale_sky_rss = np.nanmedian(scale_per_fibre)
                        self.sky_emission = self.sky_emission * scale_sky_rss
                    else:
                        scale_sky_rss = np.nanmedian(
                            old_div((scale_per_fibre + scale_per_fibre_2), 2)  # TODO: get data for 2D and test if can remove
                        )
                        # Make linear fit
                        scale_sky_rss_1 = np.nanmedian(scale_per_fibre)
                        scale_sky_rss_2 = np.nanmedian(scale_per_fibre_2)
                        print(
                            "  Median scale for line 1 : {} range [ {}, {} ]]".format(
                            scale_sky_rss_1, np.nanmin(scale_per_fibre), np.nanmax(scale_per_fibre)
                            )
                        )
                        print(
                            "  Median scale for line 2 : {} range [ {}, {} ]]".format(
                            scale_sky_rss_2, np.nanmin(scale_per_fibre_2), np.nanmax(scale_per_fibre_2)
                            )
                        )

                        b = old_div((scale_sky_rss_1 - scale_sky_rss_2), (
                            sky_line - sky_line_2   # TODO: get data for 2D and test if can remove
                        ))
                        a = scale_sky_rss_1 - b * sky_line
                        # ,a+b*sky_line,a+b*sky_line_2
                        print("  Appling linear fit with a = {} b = {} to all fibres in sky image...".format(a, b))

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
                                "Scale OBJECT / SKY using sky line $\lambda$ {}".format(sky_line))
                            print("  Scale per fibre in the range [{} , {} ], median value is {}".format(np.nanmin(scale_per_fibre), np.nanmax(scale_per_fibre), scale_sky_rss))
                            print("  Using median value to scale sky emission provided...")
                        if sky_line_2 != 0:
                            text = (
                                "Scale OBJECT / SKY using sky lines $\lambda$ {}  and $\lambda$".format(sky_line, sky_line_2))
                            label2 = "$\lambda$ {}".format(sky_line_2)
                            plt.plot(scale_per_fibre_2, alpha=0.5, label=label2)
                            plt.axhline(y=scale_sky_rss_1, color="k", linestyle=":")
                            plt.axhline(y=scale_sky_rss_2, color="k", linestyle=":")
                            plt.legend(frameon=False, loc=1, ncol=2)
                        plt.title(text)
                        plt.xlabel("Fibre")
                        # plt.show()
                        # plt.close()

                    self.intensity_corrected = (
                        self.intensity_corrected - self.sky_emission
                    )

            # (3) No sky spectrum or image is provided, obtain the sky using the n_sky lowest fibres
            if sky_method == "self":
                print("\n  Using {} lowest intensity fibres to create a sky...".format(n_sky))
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
            print("\n> This RSS file is defined as SKY... applying median filter with window {} ...".format(win_sky))
            medfilt_sky = median_filter(
                self.intensity_corrected, self.n_spectra, self.n_wave, win_sky=win_sky
            )
            self.intensity_corrected = copy.deepcopy(medfilt_sky)
            print("  Median filter applied, results stored in self.intensity_corrected !")

        # Get airmass and correct for extinction AFTER SKY SUBTRACTION
        ZD = (self.ZDSTART + self.ZDEND)/2
        self.airmass = 1/np.cos(np.radians(ZD))
        self.extinction_correction = np.ones(self.n_wave)
        if do_extinction:
            self.do_extinction_curve(pth.join(DATA_PATH, "ssoextinct.dat"), plot=plot)

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
                # plt.show()
                # plt.close()

            if plot:
                integrated_intensity_sorted = np.argsort(self.integrated_fibre)
                region = [
                    integrated_intensity_sorted[-1],
                    integrated_intensity_sorted[0],
                ]
                print("  Example of telluric correction using fibres {} and {} :".format(region[0], region[1]))
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
                # plt.show()
                # plt.close()

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
                print("\n  As given, line {} at rest wavelength = {} is at {}".format(brightest_line, brightest_line_rest_wave, brightest_line_wavelength))
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
                emission_line_file = "data/lineas_c89_python.dat"
                el_center, el_name = read_table(emission_line_file, ["f", "s"])

                # Find brightest line to get redshift
                for i in range(len(self.el[0])):
                    if self.el[0][i] == brightest_line:
                        obs_wave = self.el[2][i]
                        redshift = (self.el[2][i] - self.el[1][i])/self.el[1][i]
                print("  Brightest emission line {} foud at {} , redshift = {}".format(brightest_line, obs_wave, redshift))

                el_identified = [[], [], [], []]
                n_identified = 0
                for line in id_list:
                    id_check = 0
                    for i in range(len(self.el[1])):
                        if line == self.el[1][i]:
                            if verbose:
                                print("  Emission line {} {} has been identified".format(self.el[0][i], self.el[1][i]))
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
                                print("  Emission line {} {} has NOT been identified, adding...".format(el_name[i], line))
                        el_identified[1].append(line)
                        el_identified[2].append(line * (redshift + 1))
                        el_identified[3].append(4 * broad)

                self.el = el_identified
                print("  Number of emission lines identified = {} of a total of {} provided. self.el updated accordingly".format(n_identified, len(id_list)))
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
            print("\n> Checking results using {} fibres with the highest integrated intensity".format(high_fibres))
            print("  which are : {}".format(region))

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
            plt.title("{} - Combined spectrum - {} fibres with highest intensity".format(self.object, high_fibres))

            plt.legend(frameon=False, loc=4, ncol=2)
            # plt.show()
            # plt.close()

            region = []
            for fibre_ in range(high_fibres):
                region.append(integrated_intensity_sorted[fibre_])
            print("\n> Checking results using {} fibres with the lowest integrated intensity".format(high_fibres))
            print("  which are : {}".format(region))

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
            plt.title("{} - Combined spectrum - {} fibres with lowest intensity".format(self.object, high_fibres))

            plt.legend(frameon=False, loc=4, ncol=2)
            # plt.show()
            # plt.close()

        # Plot RSS_image
        if plot:
            self.RSS_image()

        if rss_clean:
            self.RSS_image()

        # Print summary and information from header
        print("\n> Summary of reading rss file ''{}'' :".format(filename))
        print("\n  This is a KOALA '{}' file, using grating '{}' in AAOmega".format(AAOmega_Arm, self.grating))
        print("  Object: {}".format(self.object))
        print("  Field of view: {} (spaxel size = {} arcsec)".format(field, self.spaxel_size))
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
                print("  This is a SKY IMAGE, median filter with window {} applied !".format(win_sky))
            else:
                if sky_method == "none":
                    print("  Intensities NOT corrected for sky emission")
                if sky_method == "self":
                    print("  Intensities corrected for sky emission using {} spaxels with lowest values !".format(n_sky))
                if sky_method == "1D":
                    print("  Intensities corrected for sky emission using (scaled) spectrum provided ! ")
                if sky_method == "1Dfit":
                    print("  Intensities corrected for sky emission fitting Gaussians to both 1D sky spectrum and each fibre ! ")
                if sky_method == "2D":
                    print("  Intensities corrected for sky emission using sky image provided scaled by {} !".format(scale_sky_rss))
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
                print("  Only fibre {} has been corrected for sky residuals".format(fibre))
            if clean_sky_residuals == False:
                print("  Intensities NOT corrected for sky residuals")

            print("  All applied corrections are stored in self.intensity_corrected !")

            if save_rss_to_fits_file != "":
                save_rss_fits(self, fits_file=save_rss_to_fits_file)

        print("\n> KOALA RSS file read !")


# -----------------------------------------------------------------------------
# INTERPOLATED CUBE CLASS
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
            float(kernel_size_arcsec/pixel_size_arcsec)
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
            print(" {}".format(self.description))
        else:
            print("\n> Creating cube from file rss file: {}".format(self.description))
        print("  Pixel size  = {} arcsec".format(self.pixel_size_arcsec))
        print("  kernel size = {} arcsec".format(self.kernel_size_arcsec))

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
            self.n_cols = np.int((size_arcsec[0]/self.pixel_size_arcsec)) + 2 * np.int(
                (self.kernel_size_arcsec/self.pixel_size_arcsec)
            )
            self.n_rows = np.int((size_arcsec[1]/self.pixel_size_arcsec)) + 2 * np.int(
                (self.kernel_size_arcsec/self.pixel_size_arcsec)
            )
        else:
            self.n_cols = (
                2
                * (
                    np.int(
                        (np.nanmax(
                            np.abs(RSS.offset_RA_arcsec - self.xoffset_centre_arcsec)
                        )/self.pixel_size_arcsec)
                    )
                    + np.int(self.kernel_size_pixels)
                )
                + 3
            )  # -3    ### +1 added by Angel 25 Feb 2018 to put center in center
            self.n_rows = (
                2
                * (
                    np.int(
                        (np.nanmax(
                            np.abs(RSS.offset_DEC_arcsec - self.yoffset_centre_arcsec)
                        )/self.pixel_size_arcsec)
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
                offset_rows = ((
                    RSS.offset_DEC_arcsec[i] - self.yoffset_centre_arcsec
                )/pixel_size_arcsec)
                offset_cols = ((
                    -RSS.offset_RA_arcsec[i] + self.xoffset_centre_arcsec
                )/pixel_size_arcsec)
                corrected_intensity = RSS.intensity_corrected[i]
                self.add_spectrum(
                    corrected_intensity, offset_rows, offset_cols, warnings=warnings
                )
            self.data = self._weighted_I/self._weight
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
                            (((self.data[:, x, y]/self.flux_calibration)/1e16)/self.RSS.exptime)
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
            self.ADR_x_max < self.pixel_size_arcsec/2
            and self.ADR_y_max < self.pixel_size_arcsec/2
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
                        (-2 * self.ADR_y[l]/self.pixel_size_arcsec),
                        (-2 * self.ADR_x[l]/self.pixel_size_arcsec),
                    ],
                    cval=np.nan,
                )
                mask_shift = shift(
                    mask,
                    [
                        (-2 * self.ADR_y[l]/self.pixel_size_arcsec),
                        (-2 * self.ADR_x[l]/self.pixel_size_arcsec),
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
        self.spaxel_RA0 = np.int(self.n_cols/2) + 1   # Using np.int for readability
        self.spaxel_DEC0 = np.int(self.n_rows/2) + 1  # Using np.int for readability
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
        print("  The peak of the emission in integrated image is in spaxel [ {} , {} ]".format(self.max_x, self.max_y))
        print("  The peak of the emission tracing all wavelengths is in spaxel [ {} , {} ]".format(
            np.round(self.x_peak_median, 2), np.round(self.y_peak_median, 2)))

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
            (np.linspace(x_min - kernel_centre_x, x_max - kernel_centre_x, n_points_x)/self.kernel_size_pixels)
        )
        x[0] = -1.0
        x[-1] = 1.0
        weight_x = np.diff(((3.0 * x - x ** 3 + 2.0)/4))

        kernel_centre_y = 0.5 * self.n_rows + offset_rows
        y_min = int(kernel_centre_y - self.kernel_size_pixels)
        y_max = int(kernel_centre_y + self.kernel_size_pixels) + 1
        n_points_y = y_max - y_min
        y = (
            (np.linspace(y_min - kernel_centre_y, y_max - kernel_centre_y, n_points_y)/self.kernel_size_pixels)
        )
        y[0] = -1.0
        y[-1] = 1.0
        weight_y = np.diff(((3.0 * y - y ** 3 + 2.0)/4))
        if x_min < 0 or x_max >= self.n_cols or y_min < 0 or y_max >= self.n_rows:
            if warnings:
                print("**** WARNING **** : Spectra outside field of view: {} {} {}".format(x_min, kernel_centre_x, x_max))
                print("                                                 : {} {} {}".format(y_min, kernel_centre_y, y_max))
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
            print("  Adding spaxel  1  = [ {} , {} ]".format(x[0], y[0]))
            spectrum = self.data[:, x[0], y[0]]
            for i in range(len(x) - 1):
                spectrum = spectrum + self.data[:, x[i + 1], y[i + 1]]
                print("  Adding spaxel {} = [ {} , {}]".format(i + 2, x[i + 1],[i + 1]))
                ylabel = "Flux [relative units]"
            if fcal:
                spectrum = (spectrum/self.flux_calibration)/1e16
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
            #plt.show()
            pass
        else:
            plt.savefig(save_file)
        #plt.close()

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
            #plt.show()
            pass
        else:
            plt.savefig(save_file)
        #plt.close()

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
            print("  The center of the cube is in spaxel [ {} , {} ]".format(self.spaxel_RA0, self.spaxel_DEC0))
            plt.plot([0], [0], "+", ms=13, color="black", mew=4)
            plt.plot([0], [0], "+", ms=10, color="white", mew=2)

            offset_from_center_x_arcsec = (
                spaxel[0] - self.spaxel_RA0 + 1.5
            ) * self.pixel_size_arcsec
            offset_from_center_y_arcsec = (
                spaxel[1] - self.spaxel_DEC0 + 1.5
            ) * self.pixel_size_arcsec
            print("  - Green circle:  {},        Offset from center [arcsec] :   {} {}".format(spaxel, offset_from_center_x_arcsec, offset_from_center_y_arcsec))
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
            print("  - Blue  square:  {} , Offset from center [arcsec] : {} , {}".format(np.round(spaxel2, 2), np.round(offset_from_center_x_arcsec, 3), np.round(offset_from_center_y_arcsec, 3)))
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
            print("  - Red triangle:  {} , Offset from center [arcsec] : {} , {}".format(np.round(spaxel3, 2), np.round(offset_from_center_x_arcsec, 3), np.round(offset_from_center_y_arcsec, 3)))
            plt.plot(
                [offset_from_center_x_arcsec],
                [offset_from_center_y_arcsec],
                "v",
                color="red",
                ms=7,
            )

        cbar = fig.colorbar(cax, fraction=0.0457, pad=0.04)

        if fcal:
            barlabel = "{}".format("Integrated Flux [erg s$^{-1}$ cm$^{-2}$]")
        else:
            barlabel = "{}".format("Integrated Flux [Arbitrary units]")
        cbar.set_label(barlabel, rotation=270, labelpad=20, fontsize=14)
        #        cbar.ax.set_yticklabels(['< -1', '0', '> 1'])# vertically oriented colorbar

        if save_file == "":
            #plt.show()
            pass
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
        print("\n> Created map with name {} integrating range [ {} , {} ]".format(name, wavelength1, wavelength2))
        print("    Data shape {}".format(np.shape(self.data)))
        print("    Int map shape {}".format(np.shape(mapa)))

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
        self.x_peak = old_div(np.nansum(xw, axis=(1, 2)), w)  # TODO: function is never called, check once func. understood
        self.y_peak = old_div(np.nansum(yw, axis=(1, 2)), w)  # TODO: function is never called, check once func. understood
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
            smoothfactor * int(old_div(np.sqrt(self.n_wave), 2)) + 1  # TODO: function is never called, check once func. understood
        )  # Originarily, smoothfactor = 2

        # fit, trimming edges
        valid_wl = wl[edgelow: len(wl) - edgehigh]
        valid_x = x[edgelow: len(wl) - edgehigh]
        wlm = sig.medfilt(valid_wl, odd_number)
        wx = sig.medfilt(valid_x, odd_number)
        a3x, a2x, a1x, a0x = np.polyfit(wlm, wx, 3)
        fx = a0x + a1x * wl + a2x * wl ** 2 + a3x * wl ** 3
        fxm = a0x + a1x * wlm + a2x * wlm ** 2 + a3x * wlm ** 3

        valid_y = y[edgelow: len(wl) - edgehigh]
        wy = sig.medfilt(valid_y, odd_number)
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

            plt.plot(wl, sig.medfilt(x, odd_number), "k-")
            plt.plot(wl, sig.medfilt(y, odd_number), "r-")

            hi = np.max([np.nanpercentile(x, 95), np.nanpercentile(y, 95)])
            lo = np.min([np.nanpercentile(x, 5), np.nanpercentile(y, 5)])
            plt.ylim(lo, hi)
            plt.ylabel("$\Delta$ offset [arcsec]")
            plt.xlabel("Wavelength [$\AA$]")
            plt.title(self.description)
            # plt.show()
            # plt.close()
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
        self.x_peak = np.nansum(xw, axis=(1, 2))/w
        self.y_peak = np.nansum(yw, axis=(1, 2))/w
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
            smoothfactor * int((np.sqrt(self.n_wave)/2)) + 1
        )  # Originarily, smoothfactor = 2
        print("  Using medfilt window = {}".format(odd_number))
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
        wlm = sig.medfilt(valid_wl, odd_number)
        wx = sig.medfilt(valid_x, odd_number)

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
                print("  Skipping iteration {}".format(niter))
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
        wy = sig.medfilt(valid_y, odd_number)

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
                print("  Skipping iteration {}".format(niter))
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

            plt.plot(wl, sig.medfilt(x, odd_number), "k-")
            plt.plot(wl, sig.medfilt(y, odd_number), "r-")

            hi = np.max([np.nanpercentile(x, 95), np.nanpercentile(y, 95)])
            lo = np.min([np.nanpercentile(x, 5), np.nanpercentile(y, 5)])
            plt.ylim(lo, hi)
            plt.ylabel("$\Delta$ offset [arcsec]")
            plt.xlabel("Wavelength [$\AA$]")
            plt.title(self.description)
            # plt.show()
            # plt.close()
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
            print("  - Calculating growth curve between {} {} :".format(min_wave, max_wave))

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
            r_norm = np.sqrt(np.array(r2_growth_curve)/r2_half_light)
            F_norm = np.array(F_growth_curve)/F_guess
            print("      Flux guess = {} {}  ratio = {}".format(F_guess, np.nansum(intensity), np.nansum(intensity)/F_guess))
            print("      Half-light radius: {} arcsec  = seeing if object is a star ".format(self.seeing))
            print("      Light within 2, 3, 4, 5 half-light radii: {}".format(np.interp([2, 3, 4, 5], r_norm, F_norm)))
            plt.figure(figsize=(10, 8))
            plt.plot(r_norm, F_norm, "-")
            plt.title(
                "Growth curve between {} and {} in {}".format(min_wave, max_wave, self.object))
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
            # plt.show()
            # plt.close()

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
        smooth_x = sig.medfilt(self.x_peak, smooth)  # originally, smooth = 11
        smooth_y = sig.medfilt(self.y_peak, smooth)
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
        valid_wl_smooth = sig.medfilt(valid_wl, smooth)
        valid_intensity_smooth = sig.medfilt(valid_intensity, smooth)

        if plot:
            fig_size = 12
            plt.figure(figsize=(fig_size, fig_size / 2.5))
            plt.plot(self.wavelength, intensity, "b", alpha=1, label="Intensity")
            plt.plot(
                valid_wl_smooth,
                valid_intensity_smooth,
                "r-",
                alpha=0.5,
                label="Smooth = " + "{}".format(smooth),
            )
            margen = 0.1 * (np.nanmax(intensity) - np.nanmin(intensity))
            plt.ylim(np.nanmin(intensity) - margen, np.nanmax(intensity) + margen)
            plt.xlim(np.min(self.wavelength), np.max(self.wavelength))

            plt.ylabel("Flux")
            plt.xlabel("Wavelength [$\AA$]")
            plt.title("Integrated spectrum of {} for r_half_light = {}".format(self.object, r_max))

            plt.axvline(x=min_wave, color="k", linestyle="--", alpha=0.5)
            plt.axvline(x=max_wave, color="k", linestyle="--", alpha=0.5)
            plt.minorticks_on()
            plt.legend(frameon=False, loc=1)
            # plt.show()
            # plt.close()
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

        print("\n> Computing response curve for {} using step= {},  in range [ {} , {} ]".format(self.object, step, min_wave, max_wave))

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
            print("  This has been computed before for step= {} in range [ {} , {} ], using values computed before...".format(step, min_wave, max_wave))
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
            old_div(old_div(measured_counts, flux_cal), exp_time)  # TODO, function is not called. fix once called
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
            print("  Skipping H-alpha absorption with width ={} A ...".format(ha_width))
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
            print("  ... Skipping a total of {} wavelength points".format(skipping))
        else:
            response_wavelength = lambda_cal
            response_curve = _response_curve_

        if fit_degree == 0:
            print("  Using interpolated data with smooth = {} for computing the response curve... ".format(smooth))

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
                            print("  We can't use a polynomium of grade  here, using fit_degree = 3 instead".format(fit_degree))
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
            smoothfactor * int((np.sqrt(len(wl))/2)) - 1
        )  # Originarily, smoothfactor = 2
        print("  Using medfilt window = {} for fitting...".format(odd_number))
        # fit, trimming edges
        # index=np.arange(len(x))
        # edgelow=0
        # edgehigh=1
        # valid_ind=np.where((index >= edgelow) & (index <= len(wl)-edgehigh) & (~np.isnan(x)) )[0]
        # print valid_ind
        # valid_wl = wl[edgelow:-edgehigh] # wl[valid_ind]
        # valid_x = x[edgelow:-edgehigh] #x[valid_ind]
        # wlm = sig.medfilt(valid_wl, odd_number)
        # wx = sig.medfilt(valid_x, odd_number)
        wlm = sig.medfilt(wl, odd_number)
        wx = sig.medfilt(x, odd_number)

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
                print("  Skipping iteration {}".format(niter))
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
                old_div(measured_counts, exp_time),  # TODO, function is not called. fix once called
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
            plt.title("Response curve for absolute flux calibration using {}".format(self.object))
            plt.legend(frameon=False, loc=1)
            plt.grid(which="both")
            plt.axvline(x=min_wave, color="k", linestyle="--", alpha=0.5)
            plt.axvline(x=max_wave, color="k", linestyle="--", alpha=0.5)
            plt.minorticks_on()
            # plt.show()
            # plt.close()

            plt.figure(figsize=(10, 8))
            if fit_degree > 0:
                text = "Fit using polynomium of degree {}".format(fit_degree)
            else:
                text = "Using interpolated data with smooth = {}".format(smooth)
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
            plt.title("Response curve for absolute flux calibration using {}".format(self.object))
            plt.minorticks_on()
            plt.grid(which="both")
            plt.axvline(x=min_wave, color="k", linestyle="--", alpha=0.5)
            plt.axvline(x=max_wave, color="k", linestyle="--", alpha=0.5)
            plt.legend(frameon=True, loc=4, ncol=4)
            # plt.show()
            # plt.close()

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
# CUBE CLASS (ANGEL + BEN)  ALL OF THIS NEEDS TO BE CAREFULLY TESTED & UPDATED!
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

        print("\n> Reading combined datacube  ''{}''".format(filename))
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
        self.wavelength = RSS_fits_file[1].data   # TODO: why is this 1? shouldn't it be [0], maybe cause we are doing the biggest variance?
        self.flux_calibration = RSS_fits_file[2].data
        self.n_wave = len(self.wavelength)
        self.data = RSS_fits_file[0].data
        self.wave_resolution = (self.wavelength[-1] - self.wavelength[0])/self.n_wave

        self.n_cols = RSS_fits_file[0].header["Ncols"]
        self.n_rows = RSS_fits_file[0].header["Nrows"]
        self.pixel_size_arcsec = RSS_fits_file[0].header["PIXsize"]
        self.flux_calibrated = RSS_fits_file[0].header["FCAL"]

        self.number_of_combined_files = RSS_fits_file[0].header["COFILES"]
        self.offsets_files = RSS_fits_file[0].header["OFFSETS"]

        print("\n  Object         = {}".format(self.object))
        print("  Description    = {}".format(self.description))
        print("  Centre:  RA    = {} Deg".format(self.RA_centre_deg))
        print("          DEC    = {} Deg".format(self.DEC_centre_deg))
        print("  PA             = {} Deg".format(self.PA))
        print("  Size [pix]     = {}   x  {}".format(self.n_rows, self.n_cols))
        print("  Size [arcsec]  = {}   x  {}".format(self.n_rows * self.pixel_size_arcsec, self.n_cols * self.pixel_size_arcsec))
        print("  Pix size       = {}  arcsec".format(self.pixel_size_arcsec))
        print("  Files combined = {}".format(self.number_of_combined_files))
        print("  Offsets used   = {}".format(self.offsets_files))

        print("  Wave Range     = [ {} , {} ]".format(self.wavelength[0], self.wavelength[-1]))
        print("  Wave Resol.    = {}  A/pix".format(self.wave_resolution))
        print("  Flux Cal.      = {}".format(self.flux_calibrated))

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
                barlabel = "{}".format("Integrated Flux [10$^{-16}$ erg s$^{-1}$ cm$^{-2}$]")
            else:
                barlabel = "{}".format("Integrated Flux [Arbitrary units]")
        #        if fcal:
        #            cbar.set_label("{}".format("Integrated Flux [10$^{-16}$ erg s$^{-1}$ cm$^{-2}$]"), rotation=270, labelpad=40, fontsize=fig_size*1.2)
        #        else:
        #            cbar.set_label("{}".format("Integrated Flux [Arbitrary units]"), rotation=270, labelpad=40, fontsize=fig_size*1.2)
        cbar.set_label(barlabel, rotation=270, labelpad=20, fontsize=fig_size * 1.2)

        #        cbar.ax.set_yticklabels(['< -1', '0', '> 1'])# vertically oriented colorbar
        #        cbar.set_ticks([1.5,2,3,4,5,6], update_ticks=True)
        #        cbar.set_ticklabels([1.5,2,3,4,5,6])

        if save_file == "":
            #plt.show()
            pass
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
            spectrum = (self.data[:, x, y]/self.flux_calibration)/1e16
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
            #plt.show()
            pass
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
            spectrum = (self.data[:, x, y]/self.flux_calibration)/1e16
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

        return aMax/bMax

    def createRatioMap(self, aStart, aEnd, bStart, bEnd, fcal=False):
        xLength = len(self.data[0, :, 0])
        yLength = len(self.data[0, 0, :])
        aFirstIndex = np.searchsorted(self.wavelength, aStart)
        aLastIndex = np.searchsorted(self.wavelength, aEnd)
        bFirstIndex = np.searchsorted(self.wavelength, bStart)
        bLastIndex = np.searchsorted(self.wavelength, bEnd)
        ratioMap = [[i for i in range(yLength)] for j in range(xLength)]
        for y in range(yLength):
            print("Column {}".format(y))
            for x in range(xLength):
                if fcal == False:
                    spectrum = self.data[:, x, y]
                else:
                    spectrum = (self.data[:, x, y]/self.flux_calibration)/1e16
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
                    ratio = aMax/bMax
                else:
                    ratio = 0
                ratioMap[x][y] = ratio

        return ratioMap



# -----------------------------------------------------------------------------
# GENERAL TASKS
# -----------------------------------------------------------------------------
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N])/N


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def cumulaive_Moffat(r2, L_star, alpha2, beta):
    return L_star * (1 - np.power(1 + (r2/alpha2), -beta))


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
    fit, cov = curve_fit(
        cumulaive_Moffat,
        r2_growth_curve[:index_cut],
        F_growth_curve[:index_cut],
        p0=(F_guess, r2_half_light, 1),
    )
    if plot:
        print("Best-fit: L_star = {}".format(fit[0]))
        print("          alpha = {}".format(np.sqrt(fit[1])))
        print("          beta = {}".format(fit[2]))
        r_norm = np.sqrt(np.array(r2_growth_curve)/r2_half_light)
        plt.plot(
            r_norm,
            cumulaive_Moffat(np.array(r2_growth_curve), fit[0], fit[1], fit[2])/fit[0],
            ":",
        )

    return fit


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def KOALA_offsets(s, pa):
    print("\n> Offsets towards North and East between pointings," "according to KOALA manual, for pa = {} degrees".format(pa))
    pa *= np.pi/180
    print("  a -> b : {} {}".format(s * np.sin(pa), -s * np.cos(pa)))
    print("  a -> c : {} {}".format(-s * np.sin(60 - pa), -s * np.cos(60 - pa)))
    print("  b -> d : {} {}".format(-np.sqrt(3) * s * np.cos(pa), -np.sqrt(3) * s * np.sin(pa)))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def obtain_telluric_correction(wlm, telluric_correction_list, plot=True):
    """
    Take a list of telluric correction spectra and make a single median telluric spectrum

    Args:
        wlm (array): A wavelength array. Only used for plotting- should refactor this!
        telluric_correction_list (list): A list of telluric correction spectra
        plot (bool, default=True): Whether or not to plot the resulting spectrum.
    """
    telluric_correction = np.nanmedian(telluric_correction_list, axis=0)
    if plot:
        fig = plot_telluric_correction(wlm, telluric_correction_list, telluric_correction, fig_size=12)

    print("\n\t>Telluric correction = {}".format(telluric_correction))
    print("\n\tTelluric correction obtained!")
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
    cuts = np.int(((wave_max - wave_min)/step))

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
                    print("  Skipping {} as it is in the exclusion range [ {} , {} ]".format(
                        next_wave, exclude_wlm[exclude][0], exclude_wlm[exclude][1]))

            else:
                corte_index = corte_index + 1
                running_wave.append(next_wave)
                # print running_wave
                region = np.where(
                    (wlm > running_wave[corte_index] - np.int(step/2))
                    & (wlm < running_wave[corte_index] + np.int(step/2))
                )
                running_step_median.append(np.nanmedian(s[region]))
                if next_wave > exclude_wlm[exclude][1]:
                    exclude = exclude + 1
                    if verbose:
                        print("--- End exclusion range {}".format(exclude))
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
                print("  There is a nan in {}".format(running_wave[i]))
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

        fig = plot_weights_for_getting_smooth_spectrum(
            wlm,
            s,
            running_wave,
            running_step_median,
            fit_median,
            fit_median_interpolated,
            weight_fit_median,
            wave_min,
            wave_max,
            exclude_wlm)

        print("  Weights for getting smooth spectrum:  fit_median = {}    fit_median_interpolated = {}".format(
            weight_fit_median, 1 - weight_fit_median))

    return (
        weight_fit_median * fit_median
        + (1 - weight_fit_median) * fit_median_interpolated
    )  # (fit_median+fit_median_interpolated)/2      # Decide if fit_median or fit_median_interpolated


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

    print("\n> Offsets in pixels : {} {}".format(delta_x, delta_y))
    print("  Offsets in arcsec : {} {}".format(pixel_size_arc * delta_x, pixel_size_arc * delta_y))
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
        dec1 = dec1d + dec1m / 60.0 + dec1s / 3600.0
    if dec2d < 0:
        dec2 = dec2d - dec2m / 60.0 - dec2s / 3600.0
    else:
        dec2 = dec2d + dec2m / 60.0 + dec2s / 3600.0

    avdec = (dec1 + dec2)/2

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

    print("\n> Offset 1 -> 2 : {} {}       {} {}".format(tdeltara, t_sign_deltara, tdeltadec, t_sign_deltadec))
    print("  Offset 2 -> 1 : {} {}       {} {}".format(tdeltara, t_sign_deltara_invert, tdeltadec, t_sign_deltadec_invert))




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
    error = (np.nanmedian(dif)/resolution) * 100.0
    print("\n> The median rms is {:8.6f} A,  resolution = {:5.2f} A,  error = {:5.3} %".format(
        np.nanmedian(dif), resolution, error
    ))


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
        pixel_size_arcsec=0.4,   # NOTE: changed pixel_size_arcsec to kernel_size to fix name errors
        kernel_size_arcsec=1.2, # NOTE: changed kernel_size_arcsec to kernel_size to fix name errors
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
        pk = ("_{}p{}_{}k{}".format(
            int(pixel_size_arcsec), int((abs(pixel_size_arcsec) - abs(int(pixel_size_arcsec))) * 10),
            int(kernel_size_arcsec), int((abs(kernel_size_arcsec) - abs(int(kernel_size_arcsec))) * 100)
            )
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

            print("\n  Total exposition time = {} seconds adding the {} files".format(
                self.combined_cube.total_exptime, len(rss_list)))

        # Save it to a fits file

        if save_aligned_cubes:
            print("\n  Saving aligned cubes to fits files ...")
            for i in range(n_files):
                if i < 9:
                    replace_text = "_{}_aligned_cube_0{}{}.fits".format(obj_name, i + 1, pk)
                else:
                    replace_text = "_aligned_cube_{}{}.fits".format(i + 1, pk)

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
                fits_file = ("{}{}_{}{}_combining_{}_cubes.fits".format(
                    check_if_path, obj_name, self.combined_cube.grating, pk, n_files)
                )


            save_fits_file(self.combined_cube, fits_file, ADR=ADR)

        print("\n================== REDUCING KOALA DATA COMPLETED ====================\n\n")
