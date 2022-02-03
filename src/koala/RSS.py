# !/usr/bin/python
# -*- coding: utf-8 -*-
import os.path

from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy import interpolate
from scipy.signal import medfilt
import sys
import copy

# Disable some annoying warnings
import warnings

from koala.constants import red_gratings, fuego_color_map
from koala.io import read_table, spectrum_to_text_file, full_path, save_nresponse, save_rss_fits
from koala.onedspec import fluxes, search_peaks, fit_smooth_spectrum, dfluxes, substract_given_gaussian, \
    rebin_spec_shift, smooth_spectrum, fix_red_edge, fix_blue_edge, find_cosmics_in_cut, fix_these_features
from koala.plot_plot import plot_plot, basic_statistics

warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.CRITICAL)


# =============================================================================
# RSS CLASS
# =============================================================================


class RSS(object):
    """
    Collection of row-stacked spectra (RSS).

    Attributes
    ----------
    wavelength: np.array(float)
        Wavelength, in Angstrom
    intensity: np.array(float)
        Intensity :math:`I_\lambda` per unit wavelength.
        Axis 0 corresponds to fiber ID
        Axis 1 Corresponds to spectral dimension
    intensity_corrected: np.array(float)
        Intensity with all the corresponding corrections applied.
    variance: np.array(float)
        Variance :math:`\sigma^2_\lambda` per unit wavelength
        (note the square in the definition of the variance).
    variance_corrected: np.array(float)
        Variance with all the corresponding corrections applied.
    """

    # -----------------------------------------------------------------------------

    def __init__(self):
        self.description = "Undefined row-stacked spectra (RSS)"
        self.n_spectra = 0
        self.n_wave = 0
        self.wavelength = np.zeros(0)
        self.intensity = np.zeros((0, 0))
        self.intensity_corrected = self.intensity
        self.variance = np.zeros_like(self.intensity)
        self.variance_corrected = np.zeros_like(self.intensity)
        self.corrections = []
        self.RA_centre_deg = 0.
        self.DEC_centre_deg = 0.
        self.offset_RA_arcsec = np.zeros(0)
        self.offset_DEC_arcsec = np.zeros_like(self.offset_RA_arcsec)
        self.ALIGNED_RA_centre_deg = 0.
        self.ALIGNED_DEC_centre_deg = 0.
        self.integrated_fibre = 0
        # self.throughput = np.ones((0))

    # %% =============================================================================
    # Basic methods
    # =============================================================================
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def read_mask_from_fits_file(self, mask=[[]], mask_file="", no_nans=True, plot=True,
                                 verbose=True, include_history=False):
        """
        This task reads a fits file containing a full mask and save it as self.mask.
        Note that this mask is an IMAGE,
        the default values for self.mask,following the tast 'get_mask' below, are two vectors
        with the left -self.mask[0]- and right -self.mask[1]- valid pixels of the RSS.
        This takes more memory & time to process.

        Parameters
        ----------
        mask :  array[float]
            a mask is read from this Python object instead of reading the fits file
        mask_file : string
            fits file with the mask
        no_nans : boolean
            If False, NaN values are used, otherwise, if True, NaN's will be replaced with zero values
        verbose : boolean (default = True)
            Print results
        plot: boolean (default = True)
            Plot the mask
        include_history  boolean (default = False)
            Include the basic information in the rss.history
        """
        # Read mask
        if mask_file != "":
            print("\n> Reading the mask from fits file : ")
            print(" ", mask_file)
            ftf = fits.open(mask_file)
            self.mask = ftf[0].data
            if include_history:
                self.history.append("- Mask read from fits file")
                self.history.append("  " + mask_file)
        else:
            print("\n> Reading the mask stored in Python variable...")
            self.mask = mask
            if include_history: self.history.append("- Mask read using a Python variable")
        if no_nans:
            print("  We are considering that the mask does not have 'nans' but 0s in the bad pixels")
        else:
            print("  We are considering that the mask DOES have 'nans' in the bad pixels")

        # Check edges
        suma_good_pixels = np.nansum(self.mask, axis=0)
        nspec = self.n_spectra
        w = self.wavelength
        # Left edge
        found = 0
        j = 0
        if verbose: print("\n- Checking the left edge of the ccd...")
        while found < 1:
            if suma_good_pixels[j] == nspec:
                first_good_pixel = j
                found = 2
            else:
                j = j + 1
        if verbose: print("  First good pixels is ", first_good_pixel, ", that corresponds to ", w[first_good_pixel],
                          "A")

        if plot:
            ptitle = "Left edge of the mask, valid minimun wavelength = " + np.str(
                np.round(w[first_good_pixel], 2)) + " , that is  w [ " + np.str(first_good_pixel) + " ]"
            plot_plot(w, np.nansum(self.mask, axis=0), ymax=1000, ymin=suma_good_pixels[0] - 10,
                      xmax=w[first_good_pixel * 3], vlines=[w[first_good_pixel]],
                      hlines=[nspec], ptitle=ptitle, ylabel="Sum of good fibres")

        mask_first_good_value_per_fibre = []
        for fibre in range(len(self.mask)):
            found = 0
            j = 0
            while found < 1:
                if no_nans:
                    if self.mask[fibre][j] == 0:
                        j = j + 1
                    else:
                        mask_first_good_value_per_fibre.append(j)
                        found = 2
                else:
                    if np.isnan(self.mask[fibre][j]):
                        j = j + 1
                    else:
                        mask_first_good_value_per_fibre.append(j)
                        found = 2

        mask_max = np.nanmax(mask_first_good_value_per_fibre)
        if plot: plot_plot(np.arange(nspec), mask_first_good_value_per_fibre, ymax=mask_max + 1,
                           hlines=[mask_max], xlabel="Fibre", ylabel="First good pixel in mask",
                           ptitle="Left edge of the mask")

        # Right edge, important for RED
        if verbose: print("\n- Checking the right edge of the ccd...")
        mask_last_good_value_per_fibre = []
        mask_list_fibres_all_good_values = []

        for fibre in range(len(self.mask)):
            found = 0
            j = len(self.mask[0]) - 1
            while found < 1:
                if no_nans:
                    if self.mask[fibre][j] == 0:
                        j = j - 1
                    else:
                        mask_last_good_value_per_fibre.append(j)
                        if j == len(self.mask[0]) - 1:
                            mask_list_fibres_all_good_values.append(fibre)
                        found = 2
                else:
                    if np.isnan(self.mask[fibre][j]):
                        j = j - 1
                    else:
                        mask_last_good_value_per_fibre.append(j)
                        if j == len(self.mask[0]) - 1:
                            mask_list_fibres_all_good_values.append(fibre)
                        found = 2

        mask_min = np.nanmin(mask_last_good_value_per_fibre)
        if plot:
            ptitle = "Fibres with all good values in the right edge of the mask : " + np.str(
                len(mask_list_fibres_all_good_values))
            plot_plot(np.arange(nspec), mask_last_good_value_per_fibre, ymin=np.nanmin(mask_min),
                      ymax=2050, hlines=[2047], xlabel="Fibre", ylabel="Last good pixel in mask", ptitle=ptitle)
        if verbose: print("  Minimun value of good pixel =", mask_min, " that corresponds to ", w[mask_min])
        if verbose: print("\n  --> The valid range for these data is", np.round(w[mask_max], 2), " to ",
                          np.round(w[mask_min], 2), ",  in pixels = [", mask_max, " , ", mask_min, "]")

        self.mask_good_index_range = [mask_max, mask_min]
        self.mask_good_wavelength_range = [w[mask_max], w[mask_min]]
        self.mask_list_fibres_all_good_values = mask_list_fibres_all_good_values

        if verbose:
            print("\n> Mask stored in self.mask !")
            print("  Valid range of the data stored in self.mask_good_index_range (index)")
            print("                             and in self.mask_good_wavelength  (wavelenghts)")
            print("  List of fibres with all good values in self.mask_list_fibres_all_good_values")

        if include_history:
            self.history.append("  Valid range of data using the mask:")
            self.history.append(
                "  " + np.str(w[mask_max]) + " to " + np.str(w[mask_min]) + ",  in pixels = [ " + np.str(
                    mask_max) + " , " + np.str(mask_min) + " ]")
        # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    def get_mask(self, verbose=True, plot=True, include_history=False):
        """
        Task for getting the mask using the very same RSS file.
        This assumes that the RSS does not have nans or 0 as consequence of cosmics
        in the edges.
        The task is run once at the very beginning, before applying flat or throughput.
        It provides the self.mask data

        Parameters
        ----------
        no_nans : boolean
            If False, NaN values are used, otherwise, if True, NaN's will be replaced with zero values
        verbose : boolean (default = True)
            Print results
        plot: boolean (default = True)
            Plot the mask
        include_history  boolean (default = False)
            Include the basic information in the rss.history
        """
        if verbose: print("\n> Getting the mask using the good pixels of this RSS file ...")

        #  Check if file has 0 or nans in edges
        if np.isnan(self.intensity[0][-1]):
            no_nans = False
        else:
            no_nans = True
            if self.intensity[0][-1] != 0:
                print(
                    "  Careful!!! pixel [0][-1], fibre = 0, wave = -1, that should be in the mask has a value that is not nan or 0 !!!!!")

        w = self.wavelength
        x = list(range(self.n_spectra))

        if verbose and plot: print("\n- Checking the left edge of the ccd...")
        mask_first_good_value_per_fibre = []
        for fibre in range(self.n_spectra):
            found = 0
            j = 0
            while found < 1:
                if no_nans:
                    if self.intensity[fibre][j] == 0:
                        j = j + 1
                    else:
                        mask_first_good_value_per_fibre.append(j)
                        found = 2
                else:
                    if np.isnan(self.intensity[fibre][j]):
                        j = j + 1
                    else:
                        mask_first_good_value_per_fibre.append(j)
                        found = 2
                if j > 101:
                    print(" No nan or 0 found in the fist 100 pixels, ", w[j], " for fibre", fibre)
                    mask_first_good_value_per_fibre.append(j)
                    found = 2

        mask_max = np.nanmax(mask_first_good_value_per_fibre)
        if plot:
            plot_plot(x, mask_first_good_value_per_fibre, ymax=mask_max + 1, xlabel="Fibre",
                      ptitle="Left edge of the RSS", hlines=[mask_max], ylabel="First good pixel in RSS")

        # Right edge, important for RED
        if verbose and plot: print("\n- Checking the right edge of the ccd...")
        mask_last_good_value_per_fibre = []
        mask_list_fibres_all_good_values = []

        for fibre in range(self.n_spectra):
            found = 0
            j = self.n_wave - 1
            while found < 1:
                if no_nans:
                    if self.intensity[fibre][j] == 0:
                        j = j - 1
                    else:
                        mask_last_good_value_per_fibre.append(j)
                        if j == self.n_wave - 1:
                            mask_list_fibres_all_good_values.append(fibre)
                        found = 2
                else:
                    if np.isnan(self.intensity[fibre][j]):
                        j = j - 1
                    else:
                        mask_last_good_value_per_fibre.append(j)
                        if j == len(self.intensity[0]) - 1:
                            mask_list_fibres_all_good_values.append(fibre)
                        found = 2

                if j < self.n_wave - 1 - 300:
                    print(" No nan or 0 found in the last 300 pixels, ", w[j], " for fibre", fibre)
                    mask_last_good_value_per_fibre.append(j)
                    found = 2

        mask_min = np.nanmin(mask_last_good_value_per_fibre)
        if plot:
            ptitle = "Fibres with all good values in the right edge of the RSS file : " + np.str(
                len(mask_list_fibres_all_good_values))
            plot_plot(x, mask_last_good_value_per_fibre, ymin=np.nanmin(mask_min),
                      ymax=2050, hlines=[2047], xlabel="Fibre", ylabel="Last good pixel in RSS", ptitle=ptitle)

        if verbose: print(
            "\n  --> The valid range for this RSS is {:.2f} to {:.2f} ,  in pixels = [ {} ,{} ]".format(w[mask_max],
                                                                                                        w[mask_min],
                                                                                                        mask_max,
                                                                                                        mask_min))

        self.mask = [mask_first_good_value_per_fibre, mask_last_good_value_per_fibre]
        self.mask_good_index_range = [mask_max, mask_min]
        self.mask_good_wavelength_range = [w[mask_max], w[mask_min]]
        self.mask_list_fibres_all_good_values = mask_list_fibres_all_good_values

        if verbose:
            print("\n> Mask stored in self.mask !")
            print("  self.mask[0] contains the left edge, self.mask[1] the right edge")
            print("  Valid range of the data stored in self.mask_good_index_range (index)")
            print("                             and in self.mask_good_wavelength  (wavelenghts)")
            print("  Fibres with all good values (in right edge) in self.mask_list_fibres_all_good_values")

        if include_history:
            self.history.append("- Mask obtainted using the RSS file, valid range of data:")
            self.history.append(
                "  " + np.str(w[mask_max]) + " to " + np.str(w[mask_min]) + ",  in pixels = [ " + np.str(
                    mask_max) + " , " + np.str(mask_min) + " ]")
        # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def apply_mask(self, mask_from_file=False, make_nans=False,
                   replace_nans=False, verbose=True):
        """
        Apply a mask to a RSS file.

        Parameters
        ----------
        mask_from_file : boolean (default = False)
            If a full mask (=image) has previously been created and stored in self.mask, apply this mask.
            Otherwise, self.mask should has two lists = [ list_of_first_good_fibres, list_of_last_good_fibres ].
        make_nans : boolean
            If True, apply the mask making nan all bad pixels
        replace_nans : boolean
            If False, NaN values are used, otherwise, if True, NaN's will be replaced with zero values
        verbose : boolean (default = True)
            Print results
        """
        # FIXME: IF THE MASK IS A BOOLEAN FILE WHY MULTIPLY THE DATA?
        # GENERATING FALSE 0 IS NOT CONVENIENT. IT IS BETTER TO FILL BAD VALUES
        # WITH NANs
        # Angel: If nans the tasks for substracting the sky fitting
        #        Gaussiang will fail...
        #        After all processes are done, PyKOALA put nans to all
        #        values in the mask.
        if mask_from_file:
            self.intensity_corrected = self.intensity_corrected * self.mask
            self.variance_corrected = self.variance_corrected * self.mask
        else:
            for fibre in range(self.n_spectra):
                # FIXME: VERY INNEFICIENT, TALK WITH ANGEL ABOUT THE architecture
                # Apply left part
                for i in range(self.mask[0][fibre]):
                    if make_nans:
                        self.intensity_corrected[fibre][i] = np.nan
                        # self.variance_corrected[fibre][i] = np.nan
                    else:
                        self.intensity_corrected[fibre][i] = 0
                        # self.variance_corrected[fibre][i] = 0
                # now right part
                for i in range(self.mask[1][fibre] + 1, self.n_wave):
                    if make_nans:
                        self.intensity_corrected[fibre][i] = np.nan
                        # self.variance_corrected[fibre][i] = np.nan
                    else:
                        self.intensity_corrected[fibre][i] = 0
                        # self.variance_corrected[fibre][i] = 0
        if replace_nans:
            # Change nans to 0: # TODO: IS THIS A GOOD IDEA?
            # for i in range(self.n_spectra):
            #     self.intensity_corrected[i] = [0 if np.isnan(x) else x for x in self.intensity_corrected[i]]
            self.intensity_corrected[np.isnan(self.intensity_corrected)] = 0

            if verbose: print("\n> Mask applied to eliminate nans and make 0 all bad pixels")
        else:
            if verbose:
                if make_nans:
                    print("\n> Mask applied to make nan all bad pixels")
                else:
                    print("\n> Mask applied to make 0 all bad pixels")

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def compute_integrated_fibre(self, list_spectra="all", valid_wave_min=0,
                                 valid_wave_max=0, min_value=0.01,
                                 norm=colors.PowerNorm(gamma=1. / 4.),
                                 title=" - Integrated values",
                                 text="...",
                                 correct_negative_sky=False,
                                 order_fit_negative_sky=3,
                                 kernel_negative_sky=51, low_fibres=10,
                                 individual_check=True,
                                 use_fit_for_negative_sky=False,
                                 last_check=False,
                                 plot=False, warnings=True, verbose=True):
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
        norm :  string
            normalization for the scale
        text: string
            A bit of extra text
        warnings : boolean (default = False)
            Write warnings, e.g. when the integrated flux is negative
        correct_negative_sky : boolean (default = False)
            Corrects negative values making 0 the integrated flux of the lowest fibre
        last_check: boolean (default = False)
            If that is the last correction to perform, say if there is not any fibre
            with has an integrated value < 0.

        Example
        ----------
        integrated_fibre_6500_6600 = star1r.compute_integrated_fibre(valid_wave_min=6500, valid_wave_max=6600,
        title = " - [6500,6600]", plot = True)
        """
        if list_spectra == 'all':
            list_spectra = list(range(self.n_spectra))
        if valid_wave_min == 0: valid_wave_min = self.valid_wave_min
        if valid_wave_max == 0: valid_wave_max = self.valid_wave_max

        if verbose: print("\n> Computing integrated fibre values in range [ {:.2f} , {:.2f} ] {}".format(
            valid_wave_min, valid_wave_max, text))

        # TODO: Angel: we need these or other things will fail!
        v = np.abs(self.wavelength - valid_wave_min)
        self.valid_wave_min_index = v.tolist().index(np.nanmin(v))
        v = np.abs(self.wavelength - valid_wave_max)
        self.valid_wave_max_index = v.tolist().index(np.nanmin(v))

        # TODO: Updated by Pablo
        self.integrated_fibre = np.zeros(self.n_spectra)
        self.integrated_fibre_variance = np.zeros(self.n_spectra)

        region = np.where((self.wavelength > valid_wave_min
                           ) & (self.wavelength < valid_wave_max))[0]
        waves_in_region = len(region)

        self.integrated_fibre = np.nansum(self.intensity_corrected[:, region],
                                          axis=1)
        self.integrated_fibre_variance = np.nansum(self.variance_corrected[:, region],
                                                   axis=1)
        n_negative_fibres = len(self.integrated_fibre[self.integrated_fibre < 0])
        negative_fibres = np.where(self.integrated_fibre < 0)[0]

        if verbose:
            print("  - Median value of the integrated flux =", np.round(np.nanmedian(self.integrated_fibre), 2))
            print("                                    min =", np.round(np.nanmin(self.integrated_fibre), 2), ", max =",
                  np.round(np.nanmax(self.integrated_fibre), 2))
            print("  - Median value per wavelength         =",
                  np.round(np.nanmedian(self.integrated_fibre) / waves_in_region, 2))
            print("                                    min = {:9.3f} , max = {:9.3f}".format(
                np.nanmin(self.integrated_fibre) / waves_in_region, np.nanmax(self.integrated_fibre) / waves_in_region))

        if len(negative_fibres) != 0:
            if warnings or verbose: print(
                "\n> WARNING! : Number of fibres with integrated flux < 0 : {}, that is the {:5.2f} % of the total !".format(
                    n_negative_fibres, n_negative_fibres * 100. / self.n_spectra))
            if correct_negative_sky:
                # TODO: SHOULD WE INCLUDE THE VARIANCE?
                self.correcting_negative_sky(plot=plot, order_fit_negative_sky=order_fit_negative_sky,
                                             individual_check=individual_check,
                                             kernel_negative_sky=kernel_negative_sky,
                                             use_fit_for_negative_sky=use_fit_for_negative_sky, low_fibres=low_fibres)
            else:
                if plot and verbose:
                    print(
                        "\n> Adopting integrated flux = {:5.2f} for all fibres with negative integrated flux (for presentation purposes)".format(
                            min_value))
                    print("  This value is {:5.2f} % of the median integrated flux per wavelength".format(
                        min_value * 100. / np.nanmedian(self.integrated_fibre) * waves_in_region))
                self.integrated_fibre = self.integrated_fibre.clip(
                    min=min_value, max=self.integrated_fibre.max())
        else:
            if last_check:
                if warnings or verbose: print("\n> There are no fibres with integrated flux < 0 !")

        self.integrated_fibre_sorted = np.argsort(self.integrated_fibre)
        if plot: self.RSS_map(self.integrated_fibre, norm=norm, title=title)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def correct_ccd_defects(self, kernel_correct_ccd_defects=51, fibre_p=-1, remove_5577=False,
                            # if fibre_p=fibre plots the corrections in that fibre
                            only_nans=False,
                            plot=True, verbose=False, warnings=True, fig_size=12, apply_throughput=False):
        """
        Replaces "nan" for median value, remove remove_5577 if requested.  # NEEDS TO BE CHECKED, OPTIMIZE AND CLEAN

        Parameters
        ----------
        kernel_correct_ccd_defects : odd integer (default = 51)
            width used for the median filter
        fibre_p: integer (default = "")
            if fibre_p=fibre only corrects that fibre and plots the corrections
        remove_5577 : boolean (default = False)
            Remove the 5577 sky line if detected
        verbose : boolean (default = False)
            Print results
        plot : boolean (default = True)
            Plot results
        fig_size: float (default = 12)
            Size of the figure
        apply_throughput: boolean (default = False)
            If true, the title of the plot says:  " - Throughput + CCD defects corrected"
            If false, the title of the plot says: " - CCD defects corrected"
        Example
        ----------
        self.correct_ccd_defects()
        """

        if only_nans:
            self.history.append("- Data corrected for CCD defects (nan and inf values)")
            print("\n> Correcting CCD defects (nan and inf values) using medfilt with kernel",
                  kernel_correct_ccd_defects, " ...")
        else:
            self.history.append("- Data corrected for CCD defects (nan, inf, and negative values)")
            print("\n> Correcting CCD defects (nan, inf, and negative values) using medfilt with kernel",
                  kernel_correct_ccd_defects, " ...")

        self.history.append(
            "  kernel_correct_ccd_defects = " + np.str(kernel_correct_ccd_defects) + " for running median")

        wave_min = self.valid_wave_min
        wave_max = self.valid_wave_max
        wlm = self.wavelength
        if wave_min < 5577 and remove_5577:
            flux_5577 = []  # For correcting sky line 5577 if requested
            offset_5577 = []
            if verbose: print("  Sky line 5577.34 will be removed using a Gaussian fit...")
            self.history.append("  Sky line 5577.34 is removed using a Gaussian fit")

        print(" ")
        output_every_few = np.sqrt(self.n_spectra) + 1
        next_output = -1
        if fibre_p < 0: fibre_p = ""
        if fibre_p == "":
            ri = 0
            rf = self.n_spectra
        else:
            if verbose: print("  Only fibre {} is corrected ...".format(fibre_p))
            ri = fibre_p
            rf = fibre_p + 1

        for fibre in range(ri, rf):
            if fibre > next_output and fibre_p == "":
                sys.stdout.write("\b" * 30)
                sys.stdout.write("  Cleaning... {:5.2f}% completed".format(fibre * 100. / self.n_spectra))
                sys.stdout.flush()
                next_output = fibre + output_every_few

            s = self.intensity_corrected[fibre]
            if only_nans:
                s = [0 if np.isnan(x) or np.isinf(x) else x for x in s]  # Fix nans & inf
            else:
                s = [0 if np.isnan(x) or x < 0. or np.isinf(x) else x for x in
                     s]  # Fix nans, inf & negative values = 0
            s_m = medfilt(s, kernel_correct_ccd_defects)

            fit_median = medfilt(s, kernel_correct_ccd_defects)
            bad_indices = [i for i, x in enumerate(s) if x == 0]
            for index in bad_indices:
                s[index] = s_m[index]  # Replace 0s for median value

            if fibre == fibre_p:
                espectro_old = copy.copy(self.intensity_corrected[fibre, :])
                espectro_fit_median = fit_median
                espectro_new = copy.copy(s)

            self.intensity_corrected[fibre, :] = s

            # Removing Skyline 5577 using Gaussian fit if requested
            if wave_min < 5577 and remove_5577:
                resultado = fluxes(wlm, s, 5577.34, lowlow=30, lowhigh=10, highlow=10, highhigh=30,
                                   plot=False, verbose=False, fcal=False,
                                   plot_sus=False)  # fmin=-5.0E-17, fmax=2.0E-16,
                # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                self.intensity_corrected[fibre] = resultado[11]
                flux_5577.append(resultado[3])
                offset_5577.append(resultado[1] - 5577.34)

        if fibre_p == "":
            sys.stdout.write("\b" * 30)
            sys.stdout.write("  Cleaning... 100.00 completed!")
            sys.stdout.flush()
        if verbose: print(" ")

        if wave_min < 5577 and remove_5577 and fibre_p == "":
            if verbose: print("\n> Checking centroid of skyline 5577.34 obtained during removing sky...")
            fibre_vector = np.array(list(range(len(offset_5577))))
            offset_5577_m = medfilt(offset_5577, 101)
            a1x, a0x = np.polyfit(fibre_vector, offset_5577, 1)
            fx = a0x + a1x * fibre_vector
            self.wavelength_offset_per_fibre = offset_5577

            if plot: plot_plot(fibre_vector, [offset_5577, fx, offset_5577_m], psym=["+", "-", "-"],
                               color=["r", "b", "g"], alpha=[1, 0.7, 0.8],
                               xmin=-20, xmax=fibre_vector[-1] + 20,
                               percentile_min=0.5, percentile_max=99.5, hlines=[-0.5, -0.25, 0, 0.25, 0.5],
                               ylabel="fit(5577.34) - 5577.34", xlabel="Fibre",
                               ptitle="Checking wavelength centroid of fitted skyline 5577.34",
                               label=["data", "Fit", "median k=101"],
                               fig_size=fig_size)
            if verbose:
                print("  The median value of the fit(5577.34) - 5577.34 is ", np.nanmedian(offset_5577))
                print("  A linear fit y = a + b * fibre provides a =", a0x, " and b =", a1x)
            if np.abs(a1x) * self.n_spectra < 0.01:
                if verbose:
                    print("  Wavelengths variations are smaller than 0.01 A in all rss file (largest =",
                          np.abs(a1x) * self.n_spectra, "A).")
                    print("  No need of correcting for small wavelengths shifts!")
            else:
                if verbose:
                    print("  Wavelengths variations are larger than 0.01 A in all rss file (largest =",
                          np.abs(a1x) * self.n_spectra, "A).")
                    print("  Perhaps correcting for small wavelengths shifts is needed, use:")
                    print("  sol = [ {} , {},  0 ]".format(a0x, a1x))
                if self.sol[0] != 0:
                    if verbose: print(
                        "\n  But sol already provided as an input, sol = [ {} , {},  {} ]".format(self.sol[0],
                                                                                                  self.sol[1],
                                                                                                  self.sol[2]))
                    fx_given = self.sol[0] + self.sol[1] * fibre_vector
                    rms = fx - fx_given
                    if verbose: print("  The median diference of the two solutions is {:.4}".format(np.nanmedian(rms)))
                    if plot: plot_plot(fibre_vector, [fx, fx_given, rms], psym=["-", "-", "--"],
                                       color=["b", "g", "k"], alpha=[0.7, 0.7, 1],
                                       xmin=-20, xmax=fibre_vector[-1] + 20,
                                       percentile_min=0.5, percentile_max=99.5, hlines=[-0.25, -0.125, 0, 0.125, 0.5],
                                       ylabel="fit(5577.34) - 5577.34", xlabel="Fibre",
                                       ptitle="Small wavelength variations",
                                       label=["5577 fit", "Provided", "Difference"],
                                       fig_size=fig_size)

                    if verbose: print("  Updating this solution with the NEW sol values...")
                self.sol = [a0x, a1x, 0.]

        # Plot correction in fibre p_fibre
        if fibre_p != "":
            const = (np.nanmax(espectro_new) - np.nanmin(espectro_new)) / 2
            yy = [espectro_old / espectro_fit_median, espectro_new / espectro_fit_median,
                  (const + espectro_new - espectro_old) / espectro_fit_median]
            ptitle = "Checking correction in fibre " + str(fibre_p)
            plot_plot(wlm, yy,
                      color=["r", "b", "k"], alpha=[0.5, 0.5, 0.5],
                      percentile_min=0.5, percentile_max=98,
                      ylabel="Flux / Continuum",
                      ptitle=ptitle, loc=1, ncol=4,
                      label=["Uncorrected", "Corrected", "Dif + const"],
                      fig_size=fig_size)
        else:
            # Recompute the integrated fibre
            text = "for spectra corrected for CCD defects..."
            if apply_throughput:
                title = " - CCD defects corrected"
            else:
                title = " - Throughput + CCD defects corrected"
            self.compute_integrated_fibre(valid_wave_min=wave_min, valid_wave_max=wave_max, text=text, plot=plot,
                                          title=title, verbose=verbose, warnings=warnings)

            if remove_5577 and wave_min < 5577:
                if verbose: print("  Skyline 5577.34 has been removed. Checking throughput correction...")
                extra_throughput_correction = flux_5577 / np.nanmedian(flux_5577)
                extra_throughput_correction_median = np.round(np.nanmedian(extra_throughput_correction), 3)
                if plot:
                    ptitle = "Checking throughput correction using skyline 5577 $\mathrm{\AA}$"
                    plot_plot(fibre_vector, extra_throughput_correction, color="#1f77b4",
                              percentile_min=1, percentile_max=99, hlines=[extra_throughput_correction_median],
                              ylabel="Integrated flux per fibre / median value", xlabel="Fibre",
                              ptitle=ptitle, fig_size=fig_size)
                if verbose: print("  Variations in throughput between", np.nanmin(extra_throughput_correction), "and",
                                  np.nanmax(extra_throughput_correction), ", median = ",
                                  extra_throughput_correction_median)

            # Apply mask
            self.apply_mask(verbose=verbose)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def flux_between(self, lambda_min, lambda_max, list_spectra=None):
        """
        Computes and returns the flux in range  [lambda_min, lambda_max] of
        a list of spectra.

        Parameters
        ----------
        lambda_min : float
            sets the lower wavelength range (minimum)
        lambda_max : float
            sets the upper wavelength range (maximum)
        list_spectra : list of integers (default = [])
            list with the number of fibres for computing integrated value
            If not given it does all fibres
        """
        # TODO: VARIANCE INCLUDED. NOW IT RETURNS TWO VARIABLES!!!!
        points = np.where(
            (self.wavelength > lambda_min) & (self.wavelength < lambda_max))[0]
        if not list_spectra:
            list_spectra = list(range(self.n_spectra))
        fluxes = np.nanmean(self.intensity[list_spectra, points], axis=1)
        variance = np.nanmean(self.variance[list_spectra, points], axis=1)
        return fluxes * (lambda_max - lambda_min), variance * (lambda_max - lambda_min)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def identify_el(self, high_fibres=10, brightest_line="Ha", cut=1.5,
                    fibre=0, broad=1.0, verbose=True, plot=True):
        """
        Identify fibers with highest intensity (high_fibres=10).
        Add all in a single spectrum.
        Identify emission features.
        These emission features should be those expected in all the cube!
        Also, chosing fibre=number, it identifies el in a particular fibre.

        OLD TASK - IT NEEDS TO BE CHECKED!!!

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
                print("\n> Identifying emission lines using the", high_fibres,
                      "fibres with the highest integrated intensity")
                print("  which are :", region)
            combined_high_spectrum = np.nansum(self.intensity_corrected[region], axis=0)
        else:
            combined_high_spectrum = self.intensity_corrected[fibre]
            if verbose: print("\n> Identifying emission lines in fibre", fibre)

        # Search peaks
        peaks, peaks_name, peaks_rest, continuum_limits = search_peaks(self.wavelength, combined_high_spectrum,
                                                                       plot=plot,
                                                                       cut=cut, brightest_line=brightest_line,
                                                                       verbose=False)
        p_peaks_l = []
        p_peaks_fwhm = []

        # Do Gaussian fit and provide center & FWHM (flux could be also included, not at the moment as not abs. flux-cal done)
        if verbose: print("\n  Emission lines identified:")
        for eline in range(len(peaks)):
            lowlow = continuum_limits[0][eline]
            lowhigh = continuum_limits[1][eline]
            highlow = continuum_limits[2][eline]
            highhigh = continuum_limits[3][eline]
            resultado = fluxes(self.wavelength, combined_high_spectrum, peaks[eline], verbose=False, broad=broad,
                               lowlow=lowlow, lowhigh=lowhigh, highlow=highlow, highhigh=highhigh, plot=plot,
                               fcal=False)
            p_peaks_l.append(resultado[1])
            p_peaks_fwhm.append(resultado[5])
            if verbose:  print(
                "  {:3}. {:7s} {:8.2f} centered at {:8.2f} and FWHM = {:6.2f}".format(eline + 1, peaks_name[eline],
                                                                                      peaks_rest[eline],
                                                                                      p_peaks_l[eline],
                                                                                      p_peaks_fwhm[eline]))

        return [peaks_name, peaks_rest, p_peaks_l, p_peaks_fwhm]

    # %% ===========================================================================
    # Sky substraction
    # =============================================================================
    def correcting_negative_sky(self, low_fibres=10, kernel_negative_sky=51, order_fit_negative_sky=3, edgelow=0,
                                edgehigh=0,  # step=11, weight_fit_median = 1, scale = 1.0,
                                use_fit_for_negative_sky=False, individual_check=True, force_sky_fibres_to_zero=True,
                                exclude_wlm=[[0, 0]],
                                show_fibres=[0, 450, 985], fig_size=12, plot=True, verbose=True):
        """
        Corrects negative sky with a median spectrum of the lowest intensity fibres

        Parameters
        ----------
        low_fibres : integer (default = 10)
            amount of fibres allocated to act as fibres with the lowest intensity
        kernel_negative_sky : odd integer (default = 51)
            kernel parameter for smooth median spectrum
        order_fit_negative_sky : integer (default = 3)
            order of polynomial used for smoothening and fitting the spectrum
        edgelow, edgehigh : integers (default = 0, 0)
            Minimum and maximum pixel number such that any pixel in between this range is only to be considered
        use_fit_for_negative_sky: boolean (default = False)
            Substract the order-order fit instead of the smoothed median spectrum
        individual_check: boolean (default = True)
            Check individual fibres and correct if integrated value is negative
        exclude_wlm : list
            exclusion command to prevent large absorption lines from being affected by negative sky correction : (lower wavelength, upper wavelength)
        show_fibres : list of integers (default = [0,450,985])
            List of fibres to show
        force_sky_fibres_to_zero : boolean (default = True)
            If True, fibres defined at the sky will get an integrated value = 0
        fig_size: float (default = 12)
            Size of the figure
        plot : boolean (default = False)
           Plot figure
        """

        # CHECK fit_smooth_spectrum and compare with medfilt
        w = self.wavelength
        # Set limits
        if edgelow == 0: edgelow = self.valid_wave_min_index
        if edgehigh == 0: edgehigh = np.int((self.n_wave - self.valid_wave_max_index) / 2)

        plot_this = False
        if len(show_fibres) > 0:
            show_fibres.append(self.integrated_fibre_sorted[-1])  # Adding the brightest fibre
            show_fibres.append(self.integrated_fibre_sorted[0])  # Adding the faintest fibre

        if individual_check:
            if verbose: print("\n> Individual correction of fibres with negative sky ... ")
            if force_sky_fibres_to_zero and verbose: print("  Also forcing integrated spectrum of sky_fibres = 0 ... ")
            corrected_not_sky_fibres = 0
            total_corrected = 0
            sky_fibres_to_zero = 0
            for fibre in range(self.n_spectra):
                corregir = False
                if fibre in show_fibres:
                    print("\n - Checking fibre", fibre, "...")
                    plot_this = True
                else:
                    plot_this = False
                smooth, fit = fit_smooth_spectrum(w, self.intensity_corrected[fibre], kernel=kernel_negative_sky,
                                                  edgelow=edgelow, edgehigh=edgehigh, verbose=False,
                                                  order=order_fit_negative_sky, plot=plot_this, hlines=[0.], ptitle="",
                                                  fcal=False)
                if np.nanpercentile(fit, 5) < 0:
                    if fibre not in self.sky_fibres: corrected_not_sky_fibres = corrected_not_sky_fibres + 1
                    corregir = True
                else:
                    if fibre in self.sky_fibres and force_sky_fibres_to_zero:
                        corregir == True
                        sky_fibres_to_zero = sky_fibres_to_zero + 1

                if corregir == True:
                    total_corrected = total_corrected + 1
                    if use_fit_for_negative_sky:
                        if fibre in show_fibres and verbose: print(
                            "   Using fit to smooth spectrum for correcting the negative sky in fibre", fibre, " ...")
                        self.intensity_corrected[fibre] -= fit
                        # self.variance_corrected[fibre] -= fit
                    else:
                        if fibre in show_fibres and verbose: print(
                            "   Using smooth spectrum for correcting the negative sky in fibre", fibre, " ...")
                        self.intensity_corrected[fibre] -= smooth
                        # self.variance_corrected[fibre] -= smooth
                else:
                    if fibre in show_fibres and verbose: print("   Fibre", fibre,
                                                               "does not need to be corrected for negative sky ...")

            corrected_sky_fibres = total_corrected - corrected_not_sky_fibres
            if verbose:
                print("\n> Corrected {} fibres (not defined as sky) and {} out of {} sky fibres !".format(
                    corrected_not_sky_fibres, corrected_sky_fibres, len(self.sky_fibres)))
                if force_sky_fibres_to_zero:
                    print("  The integrated spectrum of", sky_fibres_to_zero, "sky fibres have been forced to 0.")
                    print("  The integrated spectrum of all sky_fibres have been set to 0.")
            self.history.append("- Individual correction of negative sky applied")
            self.history.append("  Corrected " + np.str(corrected_not_sky_fibres) + " not-sky fibres")
            if force_sky_fibres_to_zero:
                self.history.append("  All the " + np.str(len(self.sky_fibres)) + " sky fibres have been set to 0")
            else:
                self.history.append("  Corrected " + np.str(corrected_sky_fibres) + " out of " + np.str(
                    len(self.sky_fibres)) + " sky fibres")

        else:
            # Get integrated spectrum of n_low lowest fibres and use this for ALL FIBRES
            integrated_intensity_sorted = np.argsort(self.integrated_fibre)
            region = integrated_intensity_sorted[0:low_fibres]
            Ic = np.nanmedian(self.intensity_corrected[region], axis=0)

            if verbose:
                print("\n> Correcting negative sky using median spectrum combining the", low_fibres,
                      "fibres with the lowest integrated intensity")
                print("  which are :", region)
                print("  Obtaining smoothed spectrum using a {} kernel and fitting a {} order polynomium...".format(
                    kernel_negative_sky, order_fit_negative_sky))
            ptitle = self.object + " - " + str(low_fibres) + " fibres with lowest intensity - Fitting an order " + str(
                order_fit_negative_sky) + " polynomium to spectrum smoothed with a " + str(
                kernel_negative_sky) + " kernel window"
            smooth, fit = fit_smooth_spectrum(self.wavelength, Ic, kernel=kernel_negative_sky, edgelow=edgelow,
                                              edgehigh=edgehigh, verbose=False,
                                              order=order_fit_negative_sky, plot=plot, hlines=[0.], ptitle=ptitle,
                                              fcal=False)
            if use_fit_for_negative_sky:
                self.smooth_negative_sky = fit
                if verbose: print(
                    "  Sustracting fit to smoothed spectrum of {} low intensity fibres to all fibres ...".format(
                        low_fibres))
            else:
                self.smooth_negative_sky = smooth
                if verbose: print(
                    "  Sustracting smoothed spectrum of {} low intensity fibres to all fibres ...".format(low_fibres))

            for i in range(self.n_spectra):
                self.intensity_corrected[i, :] = self.intensity_corrected[i, :] - self.smooth_negative_sky
                # self.sky_emission = self.sky_emission - self.smooth_negative_sky

            # TODO: New implementation including variance
            # self.intensity_corrected -= self.smooth_negative_sky[np.newaxis, :]
            # self.variance_corrected -= self.smooth_negative_sky[np.newaxis, :]

            if verbose: print("  This smoothed spectrum is stored in self.smooth_negative_sky")
            self.history.append("- Correcting negative sky using smoothed spectrum of the")
            self.history.append("  " + np.str(low_fibres) + " fibres with the lowest integrated value")
        # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def fit_and_substract_sky_spectrum(self, sky, w=1000, spectra=1000, rebin=False,
                                       # If rebin == True, it fits all wavelengths to be at the same wavelengths that SKY spectrum...
                                       brightest_line="Ha", brightest_line_wavelength=0,
                                       maxima_sigma=3.0, ymin=-50, ymax=600, wmin=0, wmax=0,
                                       auto_scale_sky=False,
                                       sky_lines_file="",
                                       warnings=False, verbose=False, plot=False, plot_step_fibres=True, step=100,
                                       fig_size=12, fibre=-1, max_flux_variation=15.,
                                       min_flux_ratio=-1, max_flux_ratio=-1):
        """
        Given a 1D sky spectrum, this task fits
        sky lines of each spectrum individually and substracts sky
        Needs the observed wavelength (brightest_line_wavelength) of the brightest emission line (brightest_line) .
        w is the wavelength
        spec the 2D spectra
        max_flux_variation = 15. is the % of the maximum value variation for the flux OBJ/SKY
                             A value of 15 will restrict fits to 0.85 < OBJ/SKY < 1.15
                             Similarly, we can use:  min_flux_ratio <  OBJ/SKY < max_flux_ratio
        Parameters
        ----------
        sky : list of floats
            Given sky spectrum
        w : list of floats  (default = 1000 )
            If given, sets this vector as wavelength
            It has to have the same len than sky.
        spectra : integer
            2D spectra
        rebin :  boolean
            wavelengths are fitted to the same wavelengths of the sky spectrum if True
        brightest_line : string (default "Ha")
            string name with the emission line that is expected to be the brightest in integrated spectrum
        brightest_line_wavelength : integer
            wavelength that corresponds to the brightest line
        maxima_sigma : float
            ## Sets maximum removal of FWHM for data
        ymin : integer
           sets the bottom edge of the bounding box (plotting)
        ymax : integer
           sets the top edge of the bounding box (plotting)
        wmin : integer
            sets the lower wavelength range (minimum), if 0, no range is set, specified wavelength (w) is investigated
        wmax : integer
            sets the upper wavelength range (maximum), if 0, no range is set, specified wavelength (w) is investigated
        auto_scale_sky : boolean
            scales sky spectrum for subtraction if True
        sky_lines_file :
            file containing list of sky lines to fit
        warnings : boolean
            disables warnings if set to False
        verbose : boolean (default = True)
            Print results
        plot : boolean (default = False)
            Plot results
        plot_step_fibres : boolean
           if True, plots ever odd fibre spectrum...
        step : integer (default = 50)
           step using for estimating the local medium value
        fig_size:
           Size of the figure (in x-axis), default: fig_size=10
        fibre: integer (default 0)
           If fibre is given, it identifies emission lines in the given fibre
        """
        if min_flux_ratio == -1: min_flux_ratio = 1. - max_flux_variation / 100.
        if max_flux_ratio == -1: max_flux_ratio = 1. + max_flux_variation / 100.

        template_path_prefix = './input_data/sky_lines/'
        # TODO: (FUTURE WORK) Include the possibility of using other files provided by the user
        if not sky_lines_file:
            sky_lines_file = template_path_prefix + "sky_lines_bright.dat"
            print(' > Using "sky_lines_bright.dat" as sky line template')
        if sky_lines_file == "ALL":
            sky_lines_file = template_path_prefix + "sky_lines.dat"
        if sky_lines_file == "BRIGHT":
            sky_lines_file = template_path_prefix + "sky_lines_bright.dat"
        if sky_lines_file == "IR":
            sky_lines_file = template_path_prefix + "sky_lines_IR.dat"
        if sky_lines_file in ["IRshort", "IRs", "IR_short"]:
            sky_lines_file = template_path_prefix + "sky_lines_IR_short.dat"

        self.history.append('  Skylines fitted following file:')
        self.history.append('  ' + sky_lines_file)

        print("\n> Fitting selected sky lines to both sky spectrum and object spectra ...\n")

        # TODO: It is easier to use dictionaries (also it is possible to
        # include a longer dictionary at constants module)
        common_bright_lines = {'Ha': 6562.82, 'O3b': 5006.84, 'Hb': 4861.33}
        if brightest_line in list(common_bright_lines.keys()):
            brightest_line_wavelength_rest = common_bright_lines[brightest_line]

        if brightest_line_wavelength != 0:
            print(
                "  - Using {} at rest wavelength {:6.2f} identified by the user at {:6.2f} to avoid fitting emission lines...".format(
                    brightest_line, brightest_line_wavelength_rest, brightest_line_wavelength))
        else:
            print(
                "  - No wavelength provided to 'brightest_line_wavelength', the object is NOT expected to have emission lines\n")

        redshift = brightest_line_wavelength / brightest_line_wavelength_rest - 1.

        if w == 1000: w = self.wavelength
        if spectra == 1000: spectra = copy.deepcopy(self.intensity_corrected)

        if wmin == 0: wmin = w[0]
        if wmax == 0: wmax = w[-1]

        print("  - Reading file with the list of sky lines to fit :")
        print("   ", sky_lines_file)

        # Read file with sky emission lines
        sl_center_, sl_name_, sl_fnl_, sl_lowlow_, sl_lowhigh_, sl_highlow_, sl_highhigh_, sl_lmin_, sl_lmax_ = read_table(
            sky_lines_file, ["f", "s", "f", "f", "f", "f", "f", "f", "f"])
        # number_sl = len(sl_center)

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

        el_list_no_z = [6300.3, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28,
                        7135.78, 7318.39, 7329.66, 7751.1, 9068.9]
        el_list = (redshift + 1) * np.array(el_list_no_z)
        #  [OI]   [SIII]  [OI]   Ha+[NII]  HeI    [SII]     HeI   [ArIII]  [OII]  [ArIII]  [SIII]
        el_low_list_no_z = [6296.3, 6308.1, 6359.8, 6544.0, 6674.2, 6712.5, 7061.3, 7129., 7312., 7747.1, 9063.9]
        el_high_list_no_z = [6304.3, 6316.1, 6367.8, 6590.0, 6682.2, 6736.9, 7069.3, 7141., 7336., 7755.1, 9073.9]
        el_low_list = (redshift + 1) * np.array(el_low_list_no_z)
        el_high_list = (redshift + 1) * np.array(el_high_list_no_z)

        # Double Skylines
        dsky1_ = [6257.82, 6465.34, 6828.22, 6969.70, 7239.41, 7295.81, 7711.50, 7750.56, 7853.391, 7913.57, 7773.00,
                  7870.05, 8280.94, 8344.613, 9152.2, 9092.7, 9216.5, 8827.112, 8761.2, 0]  # 8760.6, 0]#
        dsky2_ = [6265.50, 6470.91, 6832.70, 6978.45, 7244.43, 7303.92, 7715.50, 7759.89, 7860.662, 7921.02, 7780.43,
                  7879.96, 8288.34, 8352.78, 9160.9, 9102.8, 9224.8, 8836.27, 8767.7, 0]  # 8767.2, 0] #

        # Be sure the lines we are using are in the requested wavelength range
        # print "  Checking the values of skylines in the file", sky_lines_file
        # for i in range(len(sl_center_)):
        #    print sl_center_[i],sl_fnl_[i],sl_lowlow_[i],sl_lowhigh_[i],sl_highlow_[i],sl_highhigh_[i],sl_lmin_[i],sl_lmax_[i]
        # print "  We only need skylines in the {} - {} range:".format(self.valid_wave_min, self.valid_wave_max)
        print("  - We only need sky lines in the {} - {} range ".format(np.round(self.wavelength[0], 2),
                                                                        np.round(self.wavelength[-1], 2)))

        # valid_skylines = np.where((sl_center_ < self.valid_wave_max) & (sl_center_ > self.valid_wave_min))

        valid_skylines = np.where((sl_center_ < self.wavelength[-1]) & (sl_center_ > self.wavelength[0]))

        sl_center = sl_center_[valid_skylines]
        sl_fnl = sl_fnl_[valid_skylines]
        sl_lowlow = sl_lowlow_[valid_skylines]
        sl_lowhigh = sl_lowhigh_[valid_skylines]
        sl_highlow = sl_highlow_[valid_skylines]
        sl_highhigh = sl_highhigh_[valid_skylines]
        sl_lmin = sl_lmin_[valid_skylines]
        sl_lmax = sl_lmax_[valid_skylines]
        number_sl = len(sl_center)

        dsky1 = []
        dsky2 = []
        for l in range(number_sl):
            if sl_center[l] in dsky1_:
                dsky1.append(dsky1_[dsky1_.index(sl_center[l])])
                dsky2.append(dsky2_[dsky1_.index(sl_center[l])])

        print("  - All sky lines: ", sl_center)
        print("  - Double sky lines: ", dsky1)
        print("  - Total number of skylines to fit =", len(sl_center))
        print("  - Valid values for OBJ / SKY Gauss ratio  = ( ", min_flux_ratio, ",", max_flux_ratio, ")")
        print("  - Maxima sigma to consider a valid fit  = ", maxima_sigma, " A\n")

        say_status = 0
        self.wavelength_offset_per_fibre = []
        self.sky_auto_scale = []
        f_new_ALL = []
        sky_sl_gaussian_fitted_ALL = []
        only_fibre = False
        if fibre != -1:
            f_i = fibre
            f_f = fibre + 1
            print("\n ----> Checking fibre ", fibre, " (only this fibre is corrected, use fibre = -1 for all)...")
            plot = True
            verbose = True
            warnings = True
            only_fibre = True
            say_status = fibre
        else:
            f_i = 0
            f_f = self.n_spectra

        # Check if skylines are located within the range of an emission line !
        skip_sl_fit = [False] * number_sl
        if verbose or fibre == -1: print("  - Checking skylines within emission line ranges...")
        for i in range(number_sl):
            for j in range(len(el_low_list)):
                if el_low_list[j] < sl_center[i] < el_high_list[j]:
                    skip_sl_fit[i] = True
                    if verbose or fibre == -1: print('  ------> SKY line', sl_center[i], 'in EMISSION LINE !  ',
                                                     el_low_list[j], sl_center[i], el_high_list[j])

                    # Gaussian fits to the sky spectrum
        sl_gaussian_flux = []
        sl_gaussian_sigma = []
        sl_gauss_center = []
        sky_sl_gaussian_fitted = copy.deepcopy(sky)
        if verbose or fibre == -1: print("  - Performing Gaussian fitting to sky lines in sky spectrum...")
        for i in range(number_sl):
            if sl_fnl[i] == 0:
                plot_fit = False
            else:
                plot_fit = True
            if sl_center[i] in dsky1:  # == dsky1[di] :
                if fibre == -1: print("    DOUBLE IN SKY: ", sl_center[i], dsky2[dsky1.index(sl_center[i])])
                warnings_ = False
                if sl_fnl[i] == 1:
                    warnings_ = True
                    if verbose: print("    Line ", sl_center[i], " blended with ", dsky2[dsky1.index(sl_center[i])])
                resultado = dfluxes(w, sky_sl_gaussian_fitted, sl_center[i], dsky2[dsky1.index(sl_center[i])],
                                    lowlow=sl_lowlow[i], lowhigh=sl_lowhigh[i], highlow=sl_highlow[i],
                                    highhigh=sl_highhigh[i], lmin=sl_lmin[i], lmax=sl_lmax[i], fmin=-20, fmax=0,
                                    broad1=2.1 * 2.355, broad2=2.1 * 2.355, plot=plot_fit, verbose=False,
                                    plot_sus=False,
                                    fcal=False, warnings=warnings_)  # Broad is FWHM for Gaussian sigm a= 1,

                sl_gaussian_flux.append(resultado[3])  # 15 is Gauss 1, 16 is Gauss 2, 3 is Total Gauss
                sl_gauss_center.append(resultado[1])
                sl_gaussian_sigma.append(resultado[5] / 2.355)
                # sl_gaussian_flux.append(resultado[16])
                # sl_gauss_center.append(resultado[12])
                # sl_gaussian_sigma.append(resultado[14]/2.355)
                # 12     13      14        15              16
                # fit[3], fit[4],fit[5], gaussian_flux_1, gaussian_flux_2 # KANAN

            else:
                resultado = fluxes(w, sky_sl_gaussian_fitted, sl_center[i], lowlow=sl_lowlow[i], lowhigh=sl_lowhigh[i],
                                   highlow=sl_highlow[i], highhigh=sl_highhigh[i], lmin=sl_lmin[i], lmax=sl_lmax[i],
                                   fmin=-20, fmax=0,
                                   broad=2.1 * 2.355, plot=plot_fit, verbose=False, plot_sus=False, fcal=False,
                                   warnings=warnings)  # Broad is FWHM for Gaussian sigm a= 1,

                sl_gaussian_flux.append(resultado[3])
                sl_gauss_center.append(resultado[1])
                sl_gaussian_sigma.append(resultado[5] / 2.355)

            if plot_fit:
                if verbose:  print("    Fitted wavelength for sky line ", sl_center[i], " : ", sl_gauss_center[-1],
                                   "  sigma = ", sl_gaussian_sigma[-1])
                wmin = sl_lmin[i]
                wmax = sl_lmax[i]

            if skip_sl_fit[i] == False:
                sky_sl_gaussian_fitted = resultado[11]
            else:
                if verbose: print('  ------> SKY line', sl_center[i], 'in EMISSION LINE !')

        # Now Gaussian fits to fibres
        for fibre in range(f_i, f_f):  # (self.n_spectra):
            if fibre == say_status:
                if fibre == 0: print(" ")
                print("  - Checking fibre {:4} ...  ({:6.2f} % completed) ...".format(fibre,
                                                                                      fibre * 100. / self.n_spectra))
                say_status = say_status + step
                if plot_step_fibres: plot = True
            else:
                plot = False

            skip_el_fit = copy.deepcopy(skip_sl_fit)

            # Gaussian fit to object spectrum
            object_sl_gaussian_flux = []
            object_sl_gaussian_sigma = []
            ratio_object_sky_sl_gaussian = []
            dif_center_obj_sky = []
            spec = spectra[fibre]
            object_sl_gaussian_fitted = copy.deepcopy(spec)
            object_sl_gaussian_center = []
            if verbose: print("\n  - Performing Gaussian fitting to sky lines in fibre", fibre, "of object data ...")

            for i in range(number_sl):
                if sl_fnl[i] == 0:
                    plot_fit = False
                else:
                    plot_fit = True
                if skip_el_fit[i]:
                    if verbose: print("    SKIPPING SKY LINE", sl_center[i],
                                      "as located within the range of an emission line!")
                    object_sl_gaussian_flux.append(float('nan'))  # The value of the SKY SPECTRUM
                    object_sl_gaussian_center.append(float('nan'))
                    object_sl_gaussian_sigma.append(float('nan'))
                    dif_center_obj_sky.append(float('nan'))
                else:
                    if sl_center[i] in dsky1:  # == dsky1[di] :
                        if fibre == -1: print("    DOUBLE IN SKY: ", sl_center[i], dsky2[dsky1.index(sl_center[i])])
                        warnings_ = False
                        if sl_fnl[i] == 1:
                            if fibre == -1:
                                warnings_ = True
                            if verbose: print("    Line ", sl_center[i], " blended with ",
                                              dsky2[dsky1.index(sl_center[i])])
                        resultado = dfluxes(w, object_sl_gaussian_fitted, sl_center[i],
                                            dsky2[dsky1.index(sl_center[i])], lowlow=sl_lowlow[i],
                                            lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh=sl_highhigh[i],
                                            lmin=sl_lmin[i], lmax=sl_lmax[i], fmin=-20, fmax=0,
                                            broad1=sl_gaussian_sigma[i] * 2.355, broad2=sl_gaussian_sigma[i] * 2.355,
                                            plot=plot_fit, verbose=False, plot_sus=False, fcal=False,
                                            warnings=warnings_)
                        if verbose:
                            print(
                                "    line = {:.3f} : center = {:.3f}, gauss = {:.2f},  sigma = {:.2f}, flux = {:.2f}".format(
                                    sl_center[i], resultado[1], sl_gaussian_sigma[i], resultado[5] / 2.355,
                                    resultado[15]))
                            print(
                                "    line = {:.3f} : center = {:.3f}, gauss = {:.2f},  sigma = {:.2f}, flux = {:.2f}".format(
                                    dsky2[dsky1.index(sl_center[i])], resultado[12], sl_gaussian_sigma[i],
                                    resultado[14] / 2.355, resultado[16]))
                            print("    For skylines ", sl_center[i], "+", dsky2[dsky1.index(sl_center[i])],
                                  " the total flux is ", np.round(sl_gaussian_flux[i], 3),
                                  ",                     OBJ/SKY = ", np.round(resultado[3] / sl_gaussian_flux[i], 3))

                        if resultado[3] > 0 and resultado[5] / 2.355 < maxima_sigma and resultado[15] > 0 and resultado[
                            14] / 2.355 < maxima_sigma and resultado[3] / sl_gaussian_flux[i] > min_flux_ratio and \
                                resultado[3] / sl_gaussian_flux[
                            i] < max_flux_ratio * 1.25:  # and resultado[5] < maxima_sigma: # -100000.: #0:
                            object_sl_gaussian_fitted = resultado[11]

                            object_sl_gaussian_flux.append(resultado[15])
                            object_sl_gaussian_center.append(resultado[1])
                            object_sl_gaussian_sigma.append(resultado[5] / 2.355)
                            # object_sl_gaussian_flux.append(resultado[16])
                            # object_sl_gaussian_center.append(resultado[12])
                            # object_sl_gaussian_sigma.append(resultado[14]/2.355)

                            dif_center_obj_sky.append(object_sl_gaussian_center[i] - sl_gauss_center[i])
                        else:
                            if verbose: print("    Bad double fit for ", sl_center[i], "! trying single fit...")
                            average_wave = (sl_center[i] + dsky2[dsky1.index(sl_center[i])]) / 2
                            resultado = fluxes(w, object_sl_gaussian_fitted, average_wave, lowlow=sl_lowlow[i],
                                               lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh=sl_highhigh[i],
                                               lmin=average_wave - 50, lmax=average_wave + 50, fmin=-20, fmax=0,
                                               broad=4.5, plot=plot_fit, verbose=False, plot_sus=False, fcal=False,
                                               warnings=warnings)  # Broad is FWHM for Gaussian sigma= 1,
                            if verbose: print(
                                "    line = {:.3f} : center = {:.3f}, gauss = {:.2f},  sigma = {:.2f}, flux = {:.2f},   OBJ/SKY = {:.3f}".format(
                                    sl_center[i], resultado[1], sl_gaussian_sigma[i], resultado[5] / 2.355,
                                    resultado[3], resultado[3] / sl_gaussian_flux[i]))
                            if resultado[3] > 0 and resultado[5] / 2.355 < maxima_sigma * 2. and resultado[3] / \
                                    sl_gaussian_flux[i] > min_flux_ratio and resultado[3] / sl_gaussian_flux[
                                i] < max_flux_ratio:  # and resultado[5] < maxima_sigma: # -100000.: #0:
                                object_sl_gaussian_flux.append(resultado[3])
                                object_sl_gaussian_fitted = resultado[11]
                                object_sl_gaussian_center.append(resultado[1])
                                object_sl_gaussian_sigma.append(resultado[5] / 2.355)
                                dif_center_obj_sky.append(object_sl_gaussian_center[i] - sl_gauss_center[i])
                            else:
                                if verbose: print("    -> Bad fit for ", sl_center[i], "! ignoring it...")
                                object_sl_gaussian_flux.append(float('nan'))
                                object_sl_gaussian_center.append(float('nan'))
                                object_sl_gaussian_sigma.append(float('nan'))
                                dif_center_obj_sky.append(float('nan'))
                                skip_el_fit[i] = True  # We don't substract this fit
                    else:
                        resultado = fluxes(w, object_sl_gaussian_fitted, sl_center[i], lowlow=sl_lowlow[i],
                                           lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh=sl_highhigh[i],
                                           lmin=sl_lmin[i], lmax=sl_lmax[i], fmin=0, fmax=0,
                                           broad=sl_gaussian_sigma[i] * 2.355, plot=plot_fit, verbose=False,
                                           plot_sus=False, fcal=False,
                                           warnings=warnings)  # Broad is FWHM for Gaussian sigma= 1,
                        if verbose: print(
                            "    line = {:.3f} : center = {:.3f}, gauss = {:.2f},  sigma = {:.2f}, flux = {:.2f},   OBJ/SKY = {:.3f}".format(
                                sl_center[i], resultado[1], sl_gaussian_sigma[i], resultado[5] / 2.355, resultado[3],
                                resultado[3] / sl_gaussian_flux[i]))
                        if resultado[3] > 0 and resultado[5] / 2.355 < maxima_sigma and resultado[3] / sl_gaussian_flux[
                            i] > min_flux_ratio and resultado[3] / sl_gaussian_flux[
                            i] < max_flux_ratio:  # and resultado[5] < maxima_sigma: # -100000.: #0:
                            object_sl_gaussian_flux.append(resultado[3])
                            object_sl_gaussian_fitted = resultado[11]
                            object_sl_gaussian_center.append(resultado[1])
                            object_sl_gaussian_sigma.append(resultado[5] / 2.355)
                            dif_center_obj_sky.append(object_sl_gaussian_center[i] - sl_gauss_center[i])
                        else:
                            if verbose: print("    -> Bad fit for ", sl_center[i], "! ignoring it...")
                            object_sl_gaussian_flux.append(float('nan'))
                            object_sl_gaussian_center.append(float('nan'))
                            object_sl_gaussian_sigma.append(float('nan'))
                            dif_center_obj_sky.append(float('nan'))
                            skip_el_fit[i] = True  # We don't substract this fit

                try:
                    ratio_object_sky_sl_gaussian.append(object_sl_gaussian_flux[i] / sl_gaussian_flux[i])
                except Exception:
                    print("\n\n\n\n\n DIVISION FAILED in ", sl_center[i], "!!!!!   sl_gaussian_flux[i] = ",
                          sl_gaussian_flux[i], "\n\n\n\n")
                    ratio_object_sky_sl_gaussian.append(1.)

            # Scale sky lines that are located in emission lines or provided negative values in fit
            # reference_sl = 1 # Position in the file! Position 1 is sky line 6363.4
            # sl_ref_ratio = sl_gaussian_flux/sl_gaussian_flux[reference_sl]
            if verbose:
                print("  - Correcting skylines for which we couldn't get a Gaussian fit and are not in an emission line range...")
            for i in range(number_sl):
                if skip_el_fit[i] == True and skip_sl_fit[i] == False:  # Only those that are NOT in emission lines
                    # Use known center, sigma of the sky and peak
                    gauss_fix = sl_gaussian_sigma[i]
                    small_center_correction = 0.
                    # Check if center of previous sky line has a small difference in wavelength
                    small_center_correction = np.nanmedian(dif_center_obj_sky[0:i])
                    if verbose:
                        print("  - Small correction of center wavelength of sky line ", sl_center[i], "  :",
                              small_center_correction)

                    object_sl_gaussian_fitted = substract_given_gaussian(w, object_sl_gaussian_fitted,
                                                                         sl_center[i] + small_center_correction, peak=0,
                                                                         sigma=gauss_fix, flux=0, search_peak=True,
                                                                         lowlow=sl_lowlow[i], lowhigh=sl_lowhigh[i],
                                                                         highlow=sl_highlow[i], highhigh=sl_highhigh[i],
                                                                         lmin=sl_lmin[i], lmax=sl_lmax[i], plot=False,
                                                                         verbose=verbose)

                    # Substract second Gaussian if needed !!!!!
                    for di in range(len(dsky1)):
                        if sl_center[i] == dsky1[di]:
                            if verbose: print("    This was a double sky line, also substracting ",
                                              dsky2[dsky1.index(sl_center[i])], "  at ", np.round(
                                    np.array(dsky2[dsky1.index(sl_center[i])]) + small_center_correction, 2))
                            object_sl_gaussian_fitted = substract_given_gaussian(w, object_sl_gaussian_fitted, np.array(
                                dsky2[dsky1.index(sl_center[i])]) + small_center_correction, peak=0, sigma=gauss_fix,
                                                                                 flux=0, search_peak=True,
                                                                                 lowlow=sl_lowlow[i],
                                                                                 lowhigh=sl_lowhigh[i],
                                                                                 highlow=sl_highlow[i],
                                                                                 highhigh=sl_highhigh[i],
                                                                                 lmin=sl_lmin[i], lmax=sl_lmax[i],
                                                                                 plot=False, verbose=verbose)
                else:
                    if skip_sl_fit[i] == True and skip_el_fit[i] == True:
                        if verbose: print("     - SKIPPING SKY LINE", sl_center[i],
                                          " as located within the range of an emission line!")

            offset = np.nanmedian(np.array(object_sl_gaussian_center) - np.array(sl_gauss_center))
            offset_std = np.nanstd(np.array(object_sl_gaussian_center) - np.array(sl_gauss_center))

            good_ratio_values = []
            for ratio in ratio_object_sky_sl_gaussian:
                if np.isnan(ratio) == False:
                    if ratio > min_flux_ratio and ratio < max_flux_ratio:
                        good_ratio_values.append(ratio)

            valid_median_flux = np.nanmedian(good_ratio_values)

            if verbose:
                print("  - Median center offset between OBJ and SKY :", np.round(offset, 3), " A ,    std = ",
                      np.round(offset_std, 3))
                print("    Median gauss for the OBJECT              :",
                      np.round(np.nanmedian(object_sl_gaussian_sigma), 3), " A ,    std = ",
                      np.round(np.nanstd(object_sl_gaussian_sigma), 3))
                print("    Median flux OBJECT / SKY                 :",
                      np.round(np.nanmedian(ratio_object_sky_sl_gaussian), 3), "   ,    std = ",
                      np.round(np.nanstd(ratio_object_sky_sl_gaussian), 3))
                print("    Median flux OBJECT / SKY VALID VALUES    :", np.round(valid_median_flux, 3),
                      "   ,    std = ", np.round(np.nanstd(good_ratio_values), 3))
                print("  - min and max flux OBJECT / SKY = ", np.round(np.nanmin(ratio_object_sky_sl_gaussian), 3), ",",
                      np.round(np.nanmax(ratio_object_sky_sl_gaussian), 3), "  -> That is a variation of ",
                      np.round(-100. * (np.nanmin(ratio_object_sky_sl_gaussian) - 1), 2), "% and ",
                      np.round(100. * (np.nanmax(ratio_object_sky_sl_gaussian) - 1), 2), "%")
                print("                                                        but only fits with < ",
                      max_flux_variation, "% have been considered")
            if plot == True and only_fibre == True:
                # for i in range(len(sl_gauss_center)):
                #    print i+1, sl_gauss_center[i],ratio_object_sky_sl_gaussian[i]
                plt.figure(figsize=(12, 5))
                plt.plot(sl_gauss_center, ratio_object_sky_sl_gaussian, "+", ms=12, mew=2)
                plt.axhline(y=np.nanmedian(ratio_object_sky_sl_gaussian), color="k", linestyle='--', alpha=0.3)
                plt.axhline(y=valid_median_flux, color="g", linestyle='-', alpha=0.3)
                plt.axhline(y=valid_median_flux + np.nanstd(good_ratio_values), color="c", linestyle=':', alpha=0.5)
                plt.axhline(y=valid_median_flux - np.nanstd(good_ratio_values), color="c", linestyle=':', alpha=0.5)
                plt.axhline(y=min_flux_ratio, color="r", linestyle='-', alpha=0.5)
                plt.axhline(y=max_flux_ratio, color="r", linestyle='-', alpha=0.5)
                # plt.ylim(0.7,1.3)
                ptitle = "Checking flux OBJECT / SKY for fitted skylines in fibre " + np.str(
                    fibre)  # +" with rms = "+np.str(rms[i])
                plt.title(ptitle)
                plt.xlabel("Wavelength [$\mathrm{\AA}$]")
                plt.ylabel("OBJECT / SKY ")
                # plt.legend(frameon=True, loc=2, ncol=6)
                plt.minorticks_on()
                plt.show()
                plt.close()

            self.wavelength_offset_per_fibre.append(offset)
            # self.sky_auto_scale.append(np.nanmedian(ratio_object_sky_sl_gaussian))
            self.sky_auto_scale.append(valid_median_flux)

            if auto_scale_sky:
                if verbose:  print("  - As requested, using this value to scale sky spectrum before substraction... ")
                auto_scale = np.nanmedian(ratio_object_sky_sl_gaussian)
            else:
                if verbose:  print(
                    "  - As requested, DO NOT using this value to scale sky spectrum before substraction... ")
                auto_scale = 1.0
            if rebin:
                if verbose:
                    print("\n> Rebinning the spectrum of fibre", fibre, "to match sky spectrum...")
                f = object_sl_gaussian_fitted
                f_new = rebin_spec_shift(w, f, offset)
            else:
                f_new = object_sl_gaussian_fitted

            # This must be corrected at then end to use the median auto_scale value
            # self.intensity_corrected[fibre] = f_new - auto_scale * sky_sl_gaussian_fitted
            f_new_ALL.append(f_new)
            sky_sl_gaussian_fitted_ALL.append(sky_sl_gaussian_fitted)

            if plot:
                plt.figure(figsize=(12, 5))
                plt.plot(w, spec, "purple", alpha=0.7, label="Obj")
                plt.plot(w, auto_scale * sky, "r", alpha=0.5, label="Scaled sky")
                plt.plot(w, auto_scale * sky_sl_gaussian_fitted, "lime", alpha=0.8, label="Scaled sky fit")
                plt.plot(w, object_sl_gaussian_fitted, "k", alpha=0.5, label="Obj - sky fit")
                plt.plot(w, spec - auto_scale * sky, "orange", alpha=0.4, label="Obj - scaled sky")
                plt.plot(w, object_sl_gaussian_fitted - sky_sl_gaussian_fitted, "b", alpha=0.9,
                         label="Obj - sky fit - scale * rest sky")

                plt.xlim(wmin, wmax)
                plt.ylim(ymin, ymax)
                ptitle = "Fibre " + np.str(fibre)  # +" with rms = "+np.str(rms[i])
                plt.title(ptitle)
                plt.xlabel("Wavelength [$\mathrm{\AA}$]")
                plt.ylabel("Flux [counts]")
                plt.legend(frameon=True, loc=2, ncol=6)
                plt.minorticks_on()
                for i in range(len(el_list)):
                    plt.axvline(x=el_list[i], color="k", linestyle='-', alpha=0.5)  # MARIO
                for i in range(number_sl):
                    if skip_sl_fit[i]:
                        alpha = 0.1
                    else:
                        alpha = 0.6
                    if sl_fnl[i] == 1:
                        plt.axvline(x=sl_center[i], color="brown", linestyle='-', alpha=alpha + 0.4)  # alpha=1)
                    else:
                        plt.axvline(x=sl_center[i], color="y", linestyle='--', alpha=alpha)
                for i in range(len(dsky2) - 1):
                    plt.axvline(x=dsky2[i], color="orange", linestyle='--', alpha=0.6)
                plt.show()
                plt.close()

            if only_fibre:
                ymax = np.nanpercentile(self.intensity_corrected[fibre], 99.5)
                ymin = np.nanpercentile(self.intensity_corrected[fibre], 0.1) - (
                            np.nanpercentile(self.intensity_corrected[fibre], 99.5) - np.nanpercentile(
                        self.intensity_corrected[fibre], 0.1)) / 15.
                ymax = np.nanpercentile(self.intensity_corrected[fibre], 99.5)
                self.intensity_corrected[fibre] = f_new - auto_scale * sky_sl_gaussian_fitted
                plot_plot(w, [self.intensity_corrected[fibre], self.intensity[fibre]], color=["b", "r"], ymin=ymin,
                          ymax=ymax,
                          ptitle="Comparison before (red) and after (blue) sky substraction using Gaussian fit to skylines")
                print("\n  Only fibre", fibre, " is corrected, use fibre = -1 for all...")

        if only_fibre == False:
            # To avoid bad auto scaling with bright fibres or weird fibres,
            # we fit a 2nd order polynomium to a filtered median value
            sas_m = medfilt(self.sky_auto_scale, 21)  ## Assuming odd_number = 21
            # fit=np.polyfit(range(self.n_spectra),sas_m,2)   # If everything is OK this should NOT be a fit, but a median
            fit = np.nanmedian(sas_m)
            # y=np.poly1d(fit)
            fity = [fit] * self.n_spectra
            # fity=y(range(self.n_spectra))
            if plot_step_fibres:
                # ptitle = "Fit to autoscale values:\n"+np.str(y)
                ptitle = "Checking autoscale values, median value = " + np.str(
                    np.round(fit, 2)) + " using median filter 21"
                ymin_ = np.nanmin(sas_m) - 0.1
                ymax_ = np.nanmax(sas_m) + 0.4
                plot_plot(list(range(self.n_spectra)), [sas_m, fity, self.sky_auto_scale, ],
                          color=["b", "g", "r"], alpha=[0.5, 0.5, 0.8], ptitle=ptitle, ymin=ymin_, ymax=ymax_,
                          xlabel="Fibre", ylabel="Flux ratio", label=["autoscale medfilt=21", "median", "autoscale"])
                # label=["autoscale med=21", "fit","autoscale"])

            self.sky_auto_scale_fit = fity
            if auto_scale_sky:
                # print "  Correcting all fluxes adding the autoscale value of the FIT above for each fibre..."
                print("  Correcting all fluxes adding the median autoscale value to each fibre (green line)...")
            else:
                # print "  Correcting all fluxes WITHOUT CONSIDERING the autoscale value of the FIT above for each fibre..."
                print("  Correcting all fluxes WITHOUT CONSIDERING the median autoscale value ...")

            for fibre in range(self.n_spectra):
                if auto_scale_sky:
                    self.intensity_corrected[fibre] = f_new_ALL[fibre] - self.sky_auto_scale_fit[fibre] * \
                                                      sky_sl_gaussian_fitted_ALL[fibre]
                else:
                    self.intensity_corrected[fibre] = f_new_ALL[fibre] - sky_sl_gaussian_fitted_ALL[fibre]
            print("\n  All fibres corrected for sky emission performing individual Gaussian fits to each fibre !")
            self.history.append(
                "  Intensities corrected for the sky emission performing individual Gaussian fits to each fibre")
            # self.variance_corrected += self.sky_variance # TODO: Check if telluric/ext corrections were applied before

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def find_sky_fibres(self, sky_wave_min=0, sky_wave_max=0, n_sky=200, plot=False, verbose=True, warnings=True):
        """
        Identify n_sky spaxels with the LOWEST INTEGRATED VALUES and store them in self.sky_fibres

        Parameters
        ----------
        sky_wave_min, sky_wave_max : float, float (default 0, 0)
            Consider the integrated flux in the range [sky_wave_min, sky_wave_max]
            If 0, they are set to self.valid_wave_min or self.valid_wave_max
        n_sky : integer (default = 200)
            number of spaxels used for identifying sky.
            200 is a good number for calibration stars
            for real objects, particularly extense objects, set n_sky = 30 - 50
        plot : boolean (default = False)
            plots a RSS map with sky positions
        """
        if sky_wave_min == 0: sky_wave_min = self.valid_wave_min
        if sky_wave_max == 0: sky_wave_max = self.valid_wave_max
        # Assuming cleaning of cosmics and CCD defects, we just use the spaxels with the LOWEST INTEGRATED VALUES
        self.compute_integrated_fibre(valid_wave_min=sky_wave_min, valid_wave_max=sky_wave_max, plot=False,
                                      verbose=verbose, warnings=warnings)
        sorted_by_flux = np.argsort(self.integrated_fibre)
        print("\n> Identifying sky spaxels using the lowest integrated values in the [", np.round(sky_wave_min, 2), ",",
              np.round(sky_wave_max, 2), "] range ...")
        print("  We use the lowest", n_sky, "fibres for getting sky. Their positions are:")
        # Compute sky spectrum and plot RSS map with sky positions if requested
        self.sky_fibres = sorted_by_flux[:n_sky]
        if plot: self.RSS_map(self.integrated_fibre, None, self.sky_fibres, title=" - Sky Spaxels")

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def find_sky_emission(self, intensidad=[0, 0], plot=True, n_sky=200,
                          sky_fibres=[], sky_wave_min=0, sky_wave_max=0,
                          # substract_sky=True, correct_negative_sky= False,
                          norm=colors.LogNorm(), win_sky=0, include_history=True):
        """
        Find the sky emission given fibre list or taking n_sky fibres with lowest integrated value.

        Parameters
        ----------
        intensidad :
            Matrix with intensities (self.intensity or self.intensity_corrected)
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
            If 0, they are set to self.valid_wave_min or self.valid_wave_max
        norm=colors.LogNorm :
            normalises values from 0 to 1 range on a log scale for colour plotting
            //
            norm:
            Normalization scale, default is lineal scale.
            Lineal scale: norm=colors.Normalize().
            Log scale:    norm=colors.LogNorm()
            Power law:    norm=colors.PowerNorm(gamma=1./4.)
            //
        win_sky : odd integer (default = 0)
            Width in fibres of a median filter applied to obtain sky spectrum
            If 0, it will not apply any median filter.
        include_history : boolean (default = True)
            If True, it includes RSS.history the basic information
        """
        if len(sky_fibres) == 0:
            if sky_wave_min == 0: sky_wave_min = self.valid_wave_min
            if sky_wave_max == 0: sky_wave_max = self.valid_wave_max
            self.find_sky_fibres(sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, n_sky=n_sky)
        else:  # We provide a list with sky positions
            print("  We use the list provided to get the sky spectrum")
            print("  sky_fibres = ", sky_fibres)
            self.sky_fibres = np.array(sky_fibres)

        if plot: self.RSS_map(self.integrated_fibre, None, self.sky_fibres, title=" - Sky Spaxels")
        print("  List of fibres used for sky saved in self.sky_fibres")

        if include_history: self.history.append("- Obtaining the sky emission using " + np.str(n_sky) + " fibres")
        self.sky_emission = sky_spectrum_from_fibres(self, self.sky_fibres, win_sky=win_sky, plot=False,
                                                     include_history=include_history)

        if plot: plot_plot(self.wavelength, self.sky_emission, color="c",
                           ylabel="Relative flux [counts]", xlabel="Wavelength [$\mathrm{\AA}$]",
                           xmin=self.wavelength[0] - 10, xmax=self.wavelength[-1] + 10,
                           ymin=np.nanpercentile(self.sky_emission, 1), ymax=np.nanpercentile(self.sky_emission, 99),
                           vlines=[self.valid_wave_min, self.valid_wave_max],
                           ptitle="Combined sky spectrum using the requested fibres")
        print("  Sky spectrum obtained and stored in self.sky_emission !! ")

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def substract_sky(self, correct_negative_sky=False, plot=True, verbose=True, warnings=True,
                      order_fit_negative_sky=3, kernel_negative_sky=51, exclude_wlm=[[0, 0]],
                      individual_check=True, use_fit_for_negative_sky=False, low_fibres=10):
        """
        Substracts the sky stored in self.sky_emission to all fibres in self.intensity_corrected

        Parameters
        ----------
        correct_negative_sky : boolean (default = True)
            If True, and if the integrated value of the median sky is negative, this is corrected
        plot : boolean (default = True)
            Plots results
        see task 'correcting_negative_sky()' for definition of the rest of the parameters
        """
        # Substract sky in all intensities
        # for i in range(self.n_spectra):
        #    self.intensity_corrected[i,:]=self.intensity_corrected[i,:] - self.sky_emission
        self.intensity_corrected -= self.sky_emission[np.newaxis, :]
        # TODO: Warning!!! THIS IS CORRECT? SHOULDN WE NEED TO COMPUTE VAR_{SKY}?
        # self.variance_corrected -= self.sky_emission[np.newaxis, :]

        if len(self.sky_fibres) > 0: last_sky_fibre = self.sky_fibres[-1]
        median_sky_corrected = np.zeros(self.n_spectra)

        for i in range(self.n_spectra):
            median_sky_corrected[i] = np.nanmedian(
                self.intensity_corrected[i, self.valid_wave_min_index:self.valid_wave_max_index], axis=0)
        if len(self.sky_fibres) > 0: median_sky_per_fibre = np.nanmedian(median_sky_corrected[self.sky_fibres])

        if verbose:
            print("  Median flux all fibres          = ", np.round(np.nanmedian(median_sky_corrected), 3))
            if len(self.sky_fibres) > 0:
                print("  Median flux sky fibres          = ", np.round(median_sky_per_fibre, 3))
                print("  Median flux brightest sky fibre = ", np.round(median_sky_corrected[last_sky_fibre], 3))
                print("  Median flux faintest  sky fibre = ", np.round(median_sky_corrected[self.sky_fibres[0]], 3))

        # Plot median value of fibre vs. fibre
        if plot:

            if len(self.sky_fibres) > 0:
                ymin = median_sky_corrected[self.sky_fibres[0]] - 1
                # ymax = np.nanpercentile(median_sky_corrected,90),
                hlines = [np.nanmedian(median_sky_corrected), median_sky_corrected[self.sky_fibres[0]],
                          median_sky_corrected[last_sky_fibre], median_sky_per_fibre]
                chlines = ["r", "k", "k", "g"]
                ptitle = "Median flux per fibre after sky substraction\n (red = median flux all fibres, green = median flux sky fibres, grey = median flux faintest/brightest sky fibre)"
            else:
                ymin = np.nanpercentile(median_sky_corrected, 1)
                # ymax=np.nanpercentile(self.sky_emission, 1)
                hlines = [np.nanmedian(median_sky_corrected), 0]
                chlines = ["r", "k"]
                ptitle = "Median flux per fibre after sky substraction (red = median flux all fibres)"

            plot_plot(list(range(self.n_spectra)), median_sky_corrected,
                      ylabel="Median Flux [counts]", xlabel="Fibre",
                      ymin=ymin, ymax=np.nanpercentile(median_sky_corrected, 90),
                      hlines=hlines, chlines=chlines,
                      ptitle=ptitle)

        if len(self.sky_fibres) > 0:
            if median_sky_corrected[self.sky_fibres[0]] < 0:
                if verbose or warnings: print(
                    "  WARNING !  The integrated value of the sky fibre with the smallest value is negative!")
                if correct_negative_sky:
                    if verbose: print("  Fixing this, as 'correct_negative_sky' = True  ... ")
                    self.correcting_negative_sky(plot=plot, low_fibres=low_fibres, exclude_wlm=exclude_wlm,
                                                 kernel_negative_sky=kernel_negative_sky,
                                                 use_fit_for_negative_sky=use_fit_for_negative_sky,
                                                 order_fit_negative_sky=order_fit_negative_sky,
                                                 individual_check=individual_check)

        if verbose: print("  Intensities corrected for sky emission and stored in self.intensity_corrected !")
        self.history.append("  Intensities corrected for the sky emission")

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def apply_self_sky(self, sky_fibres=[], sky_spectrum=[], plot=True,
                       sky_wave_min=0, sky_wave_max=0, win_sky=0, scale_sky_1D=0,
                       brightest_line="Ha", brightest_line_wavelength=0, ranges_with_emission_lines=[0],
                       cut_red_end=0, low_fibres=10, use_fit_for_negative_sky=False, kernel_negative_sky=51,
                       order_fit_negative_sky=3, verbose=True, n_sky=50, correct_negative_sky=False,
                       individual_check=False):
        """

        Apply sky correction using the specified number of lowest fibres in the RSS file to obtain the sky spectrum

        Parameters
        ----------
        sky_fibres : list of integers (default = none)
            Specify the fibres to use to obtain sky spectrum. Will automatically determine the best fibres if not specified
        sky_spectrum : list of floats (default = none)
            Specify the sky spectrum to be used for correction. If not specified, will derive it automatically
        plot : boolean (default = True)
            Show the plots in the console
        sky_wave_min : float (default = 0)
            Specify the lower bound on wavelength range. If 0, it is set to self.valid_wave_min
        sky_wave_max : float (default = 0)
            Specify the upper bound on wavelength range. If 0, it is set to self.valid_wave_max
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

        self.history.append('- Sky sustraction using the self method')

        if len(sky_fibres) != 0:
            n_sky = len(sky_fibres)
            print("\n> 'sky_method = self', using list of", n_sky, "fibres to create a sky spectrum ...")
            self.history.append('  A list of ' + np.str(n_sky) + ' fibres was provided to create the sky spectrum')
            self.history.append(np.str(sky_fibres))
        else:
            print("\n> 'sky_method = self', hence using", n_sky, "lowest intensity fibres to create a sky spectrum ...")
            self.history.append(
                '  The ' + np.str(n_sky) + ' lowest intensity fibres were used to create the sky spectrum')

        if len(sky_spectrum) == 0:
            self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                                   sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max,
                                   win_sky=win_sky, include_history=True)

        else:
            print("  Sky spectrum provided. Using this for replacing regions with bright emission lines...")

            self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                                   sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max,
                                   win_sky=win_sky, include_history=False)

            sky_r_self = self.sky_emission

            self.sky_emission = replace_el_in_sky_spectrum(self, sky_r_self, sky_spectrum, scale_sky_1D=scale_sky_1D,
                                                           brightest_line=brightest_line,
                                                           brightest_line_wavelength=brightest_line_wavelength,
                                                           ranges_with_emission_lines=ranges_with_emission_lines,
                                                           cut_red_end=cut_red_end,
                                                           plot=plot)
            self.history.append('  Using sky spectrum provided for replacing regions with emission lines')

        self.substract_sky(plot=plot, low_fibres=low_fibres,
                           correct_negative_sky=correct_negative_sky, use_fit_for_negative_sky=use_fit_for_negative_sky,
                           kernel_negative_sky=kernel_negative_sky, order_fit_negative_sky=order_fit_negative_sky,
                           individual_check=individual_check)

        self.apply_mask(verbose=verbose, make_nans=True)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def apply_1D_sky(self, sky_fibres=[], sky_spectrum=[], sky_wave_min=0, sky_wave_max=0,
                     win_sky=0, include_history=True,
                     scale_sky_1D=0, remove_5577=True, sky_spectrum_file="",
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
            Specify the lower bound on wavelength range. If 0, it is set to self.valid_wave_min
        sky_wave_max : float (default = 0)
            Specify the upper bound on wavelength range. If 0, it is set to self.valid_wave_max
        win_sky : odd integer (default = 0)
            Width in fibres of a median filter applied to obtain sky spectrum, if 0, it will not apply any median filter
        scale_sky_1D : float (default = 0)
            Specify the scale between the sky emission and the object, if 0, will find it automatically
        include_history : boolean (default = True)
            Include the task completion into the RSS object history
        remove_5577 : boolean (default = True)
            Remove the line 5577 from the data
        sky_spectrum_file : string (default = None)
            Specify the path and name of sky spectrum file
        plot : boolean (default = True)
            Show the plots in the console
        verbose : boolean (default = True)
            Print detailed description of steps taken in console
        n_sky : integer (default = 50)
            Number of fibres to use for finding sky spectrum
        """

        self.history.append('- Sky sustraction using the 1D method')

        if sky_spectrum_file != "":

            if verbose:
                print("\n> Reading file with a 1D sky spectrum :")
                print(" ", sky_spectrum_file)

            w_sky, sky_spectrum = read_table(sky_spectrum_file, ["f", "f"])

            if np.nanmedian(self.wavelength - w_sky) != 0:
                if verbose or warnings:
                    print(
                        "\n\n  WARNING !!!! The wavelengths provided on this file do not match the wavelengths on this RSS !!\n\n")

            self.history.append('- 1D sky spectrum provided in file :')
            self.history.append('  ' + sky_spectrum_file)

        if verbose:
            print("\n> Sustracting the sky using the sky spectrum provided, checking the scale OBJ/SKY...")
        if scale_sky_1D == 0:
            if verbose:
                print("  No scale between 1D sky spectrum and object given, calculating...")

            # TODO !
            # Task "scale_sky_spectrum" uses sky lines, needs to be checked...
            # self.sky_emission,scale_sky_1D_auto=scale_sky_spectrum(self.wavelength, sky_spectrum, self.intensity_corrected,
            #                                     cut_sky=cut_sky, fmax=fmax, fmin=fmin, fibre_list=fibre_list)

            # Find self sky emission using only the lowest n_sky fibres (this should be small, 20-25)
            if n_sky == 50: n_sky = 20
            self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                                   sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max,
                                   win_sky=win_sky, include_history=include_history)

            sky_r_self = self.sky_emission

            scale_sky_1D = auto_scale_two_spectra(self, sky_r_self, sky_spectrum, scale=[0.1, 1.01, 0.025],
                                                  w_scale_min=self.valid_wave_min, w_scale_max=self.valid_wave_max,
                                                  plot=plot, verbose=True)


        elif verbose:
            print("  As requested, we scale the given 1D sky spectrum by", scale_sky_1D)

        self.sky_emission = sky_spectrum * scale_sky_1D
        self.history.append('  1D sky spectrum scaled by =' + np.str(scale_sky_1D))

        if verbose: print("\n> Scaled sky spectrum stored in self.sky_emission, substracting to all fibres...")

        # For blue spectra, remove 5577 in the sky spectrum...
        if self.valid_wave_min < 5577 and remove_5577 == True:
            if verbose: print("  Removing sky line 5577.34 from the sky spectrum...")
            resultado = fluxes(self.wavelength, self.sky_emission, 5577.34, lowlow=30, lowhigh=10, highlow=10,
                               highhigh=30,
                               plot=False, verbose=False)  # fmin=-5.0E-17, fmax=2.0E-16,
            # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
            self.sky_emission = resultado[11]
        else:
            if self.valid_wave_min < 5577 and verbose: print(
                "  Sky line 5577.34 is not removed from the sky spectrum...")

        # Remove 5577 in the object
        if self.valid_wave_min < 5577 and remove_5577 == True and scale_sky_1D == 0:  # and individual_sky_substraction == False:
            if verbose:
                print("  Removing sky line 5577.34 from the object...")
            self.history.append("  Sky line 5577.34 removed performing Gaussian fit")

            wlm = self.wavelength
            for i in range(self.n_spectra):
                s = self.intensity_corrected[i]
                # Removing Skyline 5577 using Gaussian fit if requested
                resultado = fluxes(wlm, s, 5577.34, lowlow=30, lowhigh=10, highlow=10, highhigh=30,
                                   plot=False, verbose=False)  # fmin=-5.0E-17, fmax=2.0E-16,
                # resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                self.intensity_corrected[i] = resultado[11]
        else:
            if self.valid_wave_min < 5577 and verbose:
                if scale_sky_1D == 0:
                    print("  Sky line 5577.34 is not removed from the object...")
                else:
                    print("  Sky line 5577.34 already removed in object during CCD cleaning...")

        self.substract_sky(plot=plot, verbose=verbose)

        if plot:
            text = "Sky spectrum (scaled using a factor " + np.str(scale_sky_1D) + " )"
            plot_plot(self.wavelength, self.sky_emission, hlines=[0], ptitle=text,
                      xmin=self.wavelength[0] - 10, xmax=self.wavelength[-1] + 10, color="c",
                      vlines=[self.valid_wave_min, self.valid_wave_max])
        if verbose:
            print("  Intensities corrected for sky emission and stored in self.intensity_corrected !")
        self.sky_emission = sky_spectrum  # Restore sky_emission to original sky_spectrum
        # self.apply_mask(verbose=verbose)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def apply_1Dfit_sky(self, sky_spectrum=[], n_sky=50, sky_fibres=[], sky_spectrum_file="",
                        sky_wave_min=0, sky_wave_max=0, win_sky=0, scale_sky_1D=0,
                        sky_lines_file="", brightest_line_wavelength=0,
                        brightest_line="Ha", maxima_sigma=3, auto_scale_sky=False,
                        plot=True, verbose=True, fig_size=12, fibre_p=-1, kernel_correct_ccd_defects=51):
        """
        Apply 1Dfit sky correction.

        Parameters
        ----------
        sky_spectrum : list of floats (default = none)
            Specify the sky spectrum to be used for correction. If not specified, will derive it automatically
        n_sky : integer (default = 50)
            Number of fibres to use for finding sky spectrum
         sky_fibres : list of integers (default = none)
            Specify the fibres to use to obtain sky spectrum. Will automatically determine the best fibres if not specified
        sky_spectrum_file : string (default = None)
            Specify the path and name of sky spectrum file
        sky_wave_min : float (default = 0)
            Specify the lower bound on wavelength range. If 0, it is set to self.valid_wave_min
        sky_wave_max : float (default = 0)
            Specify the upper bound on wavelength range. If 0, it is set to self.valid_wave_max
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
        self.history.append('- Sky sustraction using the 1Dfit method')

        if sky_spectrum_file != "":
            if verbose:
                print("\n> Reading file with a 1D sky spectrum :")
                print(" ", sky_spectrum_file)

            w_sky, sky_spectrum = read_table(sky_spectrum_file, ["f", "f"])

            if np.nanmedian(self.wavelength - w_sky) != 0:
                if verbose:
                    print(
                        "\n\n  WARNING !!!! The wavelengths provided on this file do not match the wavelengths on this RSS !!\n\n")

            self.history.append('- 1D sky spectrum provided in file :')
            self.history.append('  ' + sky_spectrum_file)
        if verbose:
            print("\n> Fitting sky lines in both a provided sky spectrum AND all the fibres")
            print("  This process takes ~20 minutes for 385R if all skylines are considered!\n")
        if len(sky_spectrum) == 0:
            if verbose:
                print("  No sky spectrum provided, using", n_sky, "lowest intensity fibres to create a sky...")
            self.history.append('  ERROR! No sky spectrum provided, using self method with n_sky =' + np.str(n_sky))
            self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                                   sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, win_sky=win_sky)
        else:
            if scale_sky_1D != 0:
                self.history.append('  1D sky spectrum scaled by =' + np.str(scale_sky_1D))
                if verbose:
                    print("  1D sky spectrum scaled by ", scale_sky_1D)
            else:
                if verbose:
                    print("  No scale between 1D sky spectrum and object given, calculating...")
                if n_sky == 50: n_sky = 20
                self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                                       sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max,
                                       win_sky=win_sky, include_history=False)

                sky_r_self = self.sky_emission

                scale_sky_1D = auto_scale_two_spectra(self, sky_r_self, sky_spectrum, scale=[0.1, 1.01, 0.025],
                                                      w_scale_min=self.valid_wave_min, w_scale_max=self.valid_wave_max,
                                                      plot=plot, verbose=True)

            self.history.append('  1D sky spectrum scaled by =' + np.str(scale_sky_1D))

            self.sky_emission = np.array(sky_spectrum) * scale_sky_1D

        self.fit_and_substract_sky_spectrum(self.sky_emission, sky_lines_file=sky_lines_file,
                                            brightest_line_wavelength=brightest_line_wavelength,
                                            brightest_line=brightest_line,
                                            maxima_sigma=maxima_sigma, ymin=-50, ymax=600, wmin=0, wmax=0,
                                            auto_scale_sky=auto_scale_sky,
                                            warnings=False, verbose=False, plot=False, fig_size=fig_size, fibre=fibre_p)

        if fibre_p == -1:
            if verbose:
                print("\n> 1Dfit sky_method usually generates some nans, correcting ccd defects again...")
            self.correct_ccd_defects(kernel_correct_ccd_defects=kernel_correct_ccd_defects, verbose=verbose, plot=plot,
                                     only_nans=True)  # Not replacing values <0

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def is_sky(self, n_sky=50, win_sky=0, sky_fibres=[], sky_wave_min=0,
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
            Specify the lower bound on wavelength range. If 0, it is set to self.valid_wave_min
        sky_wave_max : float (default = 0)
            Specify the upper bound on wavelength range. If 0, it is set to self.valid_wave_max
        win_sky : odd integer (default = 0)
            Width in fibres of a median filter applied to obtain sky spectrum, if 0, it will not apply any median filter
         plot : boolean (default = True)
            Show the plots in the console
        verbose : boolean (default = True)
            Print detailed description of steps taken in console
        """

        if verbose: print("\n> This RSS file is defined as SKY... identifying", n_sky,
                          " lowest fibres for getting 1D sky spectrum...")
        self.history.append('- This RSS file is defined as SKY:')
        self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                               sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, win_sky=0)
        # print "\n> This RSS file is defined as SKY... applying median filter with window",win_sky,"..."
        if win_sky == 0:  # Default when it is not a win_sky
            win_sky = 151
        print("\n  ... applying median filter with window", win_sky, "...\n")

        medfilt_sky = median_filter(self.intensity_corrected, self.n_spectra, self.n_wave, win_sky=win_sky)
        self.intensity_corrected = copy.deepcopy(medfilt_sky)
        print("  Median filter applied, results stored in self.intensity_corrected !")
        self.history.append('  Median filter ' + np.str(win_sky) + ' applied to all fibres')

    # %% =============================================================================
    # Extinction
    # =============================================================================
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def do_extinction_curve(self, observatory_extinction_file=None, fig_size=12,
                            apply_extinction=True, plot=False, verbose=True):
        """
        This task accounts and corrects for extinction due to gas and dusty
        between target and observer. It creates a extinction curve based off
        airmass input and observatory file data.

                Parameters
        ----------
        apply_extinction : boolean (default = True)
            Apply extinction curve to the data
        observatory_extinction_file : string, input data file (default = 'ssoextinct.dat')
            file containing the standard extinction curve at the observatory
        plot : boolean (default = True)
            Plot that generates extinction curve [extinction_curve_wavelengths,extinction_corrected_airmass]
        verbose : boolean (default = True)
            Print results
        """
        if verbose:
            print("\n> Computing extinction at given airmass...")
            print("  Airmass = ", np.round(self.airmass, 3))
        # Read data
        if observatory_extinction_file is None:
            observatory_extinction_file = os.path.join('.', 'input_data', 'observatory_extinction', 'ssoextinct.dat')

        data_observatory = np.loadtxt(observatory_extinction_file, unpack=True)
        extinction_curve_wavelenghts = data_observatory[0]
        extinction_curve = data_observatory[1]
        extinction_corrected_airmass = 10 ** (0.4 * self.airmass * extinction_curve)
        # Make fit
        tck = interpolate.splrep(extinction_curve_wavelenghts,
                                 extinction_corrected_airmass, s=0)
        self.extinction_correction = interpolate.splev(self.wavelength, tck, der=0)

        if verbose: print("  Observatory file with extinction curve :\n ", observatory_extinction_file)

        if plot:
            cinco_por_ciento = 0.05 * (np.max(self.extinction_correction) - np.min(self.extinction_correction))
            plot_plot(extinction_curve_wavelenghts, extinction_corrected_airmass, xmin=np.min(self.wavelength),
                      xmax=np.max(self.wavelength), ymin=np.min(self.extinction_correction) - cinco_por_ciento,
                      ymax=np.max(self.extinction_correction) - cinco_por_ciento,
                      vlines=[self.valid_wave_min, self.valid_wave_max],
                      ptitle='Correction for extinction using airmass = ' + str(np.round(self.airmass, 3)),
                      xlabel="Wavelength [$\mathrm{\AA}$]", ylabel="Flux correction", fig_size=fig_size,
                      statistics=False)

        if apply_extinction:
            self.apply_extinction_correction(self.extinction_correction,
                                             observatory_extinction_file=observatory_extinction_file)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def apply_extinction_correction(self, extinction_correction, observatory_extinction_file="", verbose=True):
        """
        This corrects for extinction using a given extiction correction and an observatory file

        Parameters
        ----------
        extinction_correction: array
            array with the extiction correction derived using the airmass

        observatory_extinction_file : string, input data file (default = 'ssoextinct.dat')
            file containing the standard extinction curve at the observatory
        verbose : boolean (default = True)
            Print results
        """
        if verbose:
            print("  Intensities corrected for extinction stored in self.intensity_corrected")
            print("  Variance corrected for extinction stored in self.variance_corrected")

        self.intensity_corrected *= extinction_correction[np.newaxis, :]
        # self.variance_corrected *= extinction_correction[np.newaxis, :]**2
        # self.corrections.append('Extinction correction')
        self.history.append("- Data corrected for extinction using file :")
        self.history.append("  " + observatory_extinction_file)
        self.history.append("  Average airmass = " + np.str(self.airmass))

    # %% =============================================================================
    # Telluric correction
    # =============================================================================
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def get_telluric_correction(self, n_fibres=10, correct_from=6850., correct_to=10000.,
                                save_telluric_file="",
                                apply_tc=False, step=10, is_combined_cube=False, weight_fit_median=0.5,
                                exclude_wlm=[[6450, 6700], [6850, 7050], [7130, 7380]],  # This is range for 1000R
                                wave_min=0, wave_max=0, plot=True, fig_size=12, verbose=False):
        """
        Get telluric correction using a spectrophotometric star

        IMPORTANT: check tasks "telluric_correction_from_star" and "telluric_correction_using_bright_continuum_source"

        Parameters
        ----------
        n_fibres: integer (default = 10)
            number of fibers to add for obtaining spectrum
        correct_from :  float (default = 6850)
            wavelength from which telluric correction is applied
        correct_to :  float (default = 10000)
            last wavelength where telluric correction is applied
        save_telluric_file : string (default = "")
            If given, it saves the telluric correction in a file with that name
        apply_tc : boolean (default = False)
            apply telluric correction to data
            Only do this when absolutely sure the correction is good!
        step : integer (default = 10)
           step using for estimating the local medium value
        is_combined_cube : boolean (default = False)
            Use True if the cube is a combined cube and needs to read from self.combined_cube
        weight_fit_median : float between 0 and 1 (default = 0.5)
            weight of the median value when calling task smooth_spectrum
        exclude_wlm : list of [float, float] (default = [[6450,6700],[6850,7050], [7130,7380]] )
            Wavelength ranges not considering for normalising stellar continuum
            The default values are for 1000R grating
        wave_min, wave_max : float (default = 0,0)
            Wavelength range to consider, [wave_min, wave_max]
            if 0, it uses  wave_min=wavelength[0] and wave_max=wavelength[-1]
        plot : boolean (default = True)
            Plot
        fig_size: float (default = 12)
           Size of the figure
        verbose : boolean (default = True)
            Print results

        Example
        ----------
        telluric_correction_star1 = star1r.get_telluric_correction(n_fibres=15,
                                    exclude_wlm= [ [6245,6390],[6450,6750],[6840,7000],
                                    [7140,7400],[7550,7720],[8050,8450]])
        """
        print("\n> Obtaining telluric correction using spectrophotometric star...")

        if is_combined_cube:
            wlm = self.combined_cube.wavelength
        else:
            wlm = self.wavelength

        if wave_min == 0: wave_min = wlm[0]
        if wave_max == 0: wave_max = wlm[-1]

        if is_combined_cube:
            if self.combined_cube.seeing == 0:
                self.combined_cube.half_light_spectrum(5, plot=plot, min_wave=wave_min, max_wave=wave_max)
            estrella = self.combined_cube.integrated_star_flux
        else:
            integrated_intensity_sorted = np.argsort(self.integrated_fibre)
            intensidad = self.intensity_corrected
            region = []
            # TODO: SPEED THIS UP BY REMOVING THE LOOP
            for fibre in range(n_fibres):
                region.append(integrated_intensity_sorted[-1 - fibre])
            estrella = np.nansum(intensidad[region], axis=0)

        smooth_med_star = smooth_spectrum(wlm, estrella, wave_min=wave_min, wave_max=wave_max, step=step,
                                          weight_fit_median=weight_fit_median,
                                          exclude_wlm=exclude_wlm, plot=plot, verbose=verbose)

        telluric_correction = np.ones(len(wlm))

        estrella_m = medfilt(estrella, 151)
        plot_plot(wlm, [estrella, smooth_med_star, estrella_m])

        # Avoid H-alpha absorption
        rango_ha = [0, 0]
        for rango in exclude_wlm:
            if rango[0] < 6563. and rango[1] > 6563.:  # H-alpha is here, skip
                print("  Skipping range with H-alpha...")
                rango_ha = rango

        # TODO: REMOVE LOOPS
        correct_from = 6000.
        for l in range(len(wlm)):
            if wlm[l] > correct_from and wlm[l] < correct_to:

                if wlm[l] > rango_ha[0] and wlm[l] < rango_ha[1]:
                    step = step + 0
                    # skipping Ha
                else:
                    telluric_correction[l] = smooth_med_star[l] / estrella[l]

        waves_for_tc_ = []
        for rango in exclude_wlm:
            if rango[0] < 6563. and rango[1] > 6563.:  # H-alpha is here, skip
                print("  Skipping range with H-alpha...")
            else:
                index_region = np.where((wlm >= rango[0]) & (wlm <= rango[1]))
                waves_for_tc_.append(index_region)

        waves_for_tc = []
        for rango in waves_for_tc_:
            waves_for_tc = np.concatenate((waves_for_tc, rango[0].tolist()), axis=None)

        # Now, change the value in telluric_correction
        for index in waves_for_tc:
            i = np.int(index)
            if smooth_med_star[i] / estrella[i] > 1.:
                telluric_correction[i] = smooth_med_star[i] / estrella[i]

        if plot:
            plt.figure(figsize=(fig_size, fig_size / 2.5))
            if is_combined_cube:
                print("  Telluric correction for this star (" + self.combined_cube.object + ") :")
                plt.plot(wlm, estrella, color="b", alpha=0.3)
                plt.plot(wlm, estrella * telluric_correction, color="g", alpha=0.5)
                plt.ylim(np.nanmin(estrella), np.nanmax(estrella))
            else:
                print("  Example of telluric correction using fibres", region[0], " (blue) and ", region[1],
                      " (green):")
                plt.plot(wlm, intensidad[region[0]], color="b", alpha=0.3)
                plt.plot(wlm, intensidad[region[0]] * telluric_correction, color="g", alpha=0.5)
                plt.plot(wlm, intensidad[region[1]], color="b", alpha=0.3)
                plt.plot(wlm, intensidad[region[1]] * telluric_correction, color="g", alpha=0.5)
                plt.ylim(np.nanmin(intensidad[region[1]]),
                         np.nanmax(intensidad[region[0]]))  # CHECK THIS AUTOMATICALLY
            plt.axvline(x=wave_min, color='k', linestyle='--')
            plt.axvline(x=wave_max, color='k', linestyle='--')
            plt.xlim(wlm[0] - 10, wlm[-1] + 10)
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            if exclude_wlm[0][0] != 0:
                for i in range(len(exclude_wlm)):
                    plt.axvspan(exclude_wlm[i][0], exclude_wlm[i][1], color='r', alpha=0.1)
            plt.minorticks_on()
            plt.show()
            plt.close()

        if apply_tc:  # Check this
            print("  Applying telluric correction to this star...")
            if is_combined_cube:
                self.combined_cube.integrated_star_flux = self.combined_cube.integrated_star_flux * telluric_correction
                for i in range(self.combined_cube.n_rows):
                    for j in range(self.combined_cube.n_cols):
                        self.combined_cube.data[:, i, j] = self.combined_cube.data[:, i, j] * telluric_correction
            else:
                for i in range(self.n_spectra):
                    self.intensity_corrected[i, :] = self.intensity_corrected[i, :] * telluric_correction
        else:
            print("  As apply_tc = False , telluric correction is NOT applied...")

        if is_combined_cube:
            self.combined_cube.telluric_correction = telluric_correction
        else:
            self.telluric_correction = telluric_correction

            # save file if requested
        if save_telluric_file != "":
            spectrum_to_text_file(wlm, telluric_correction, filename=save_telluric_file)

        return telluric_correction

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def apply_telluric_correction(self, telluric_correction_file="", telluric_correction=[0],
                                  plot=True, fig_size=12, verbose=True):
        """

        Apply telluric correction to the data

        Parameters
        ----------
        telluric_correction_file : string (default = none)
            Name of the file containing data necessary to apply telluric correction, if not in the immediate directory, path needs to be specified
        telluric_correction : list of floats (default = [0])
            Table data from the telluric correction file in format of Python list
        plot : boolean (default = True)
            Show the plots in the console
        fig_size : float (default = 12)
            Size of the plots
        verbose : boolean (default = True)
            Print detailed description of steps taken in console
        """
        plot_integrated_fibre_again = 0
        if telluric_correction_file != "":
            print("\n> Reading file with the telluric correction: ")
            print(" ", telluric_correction_file)
            w_star, telluric_correction = read_table(telluric_correction_file, ["f", "f"])

        if telluric_correction[0] != 0:
            plot_integrated_fibre_again = plot_integrated_fibre_again + 1
            self.telluric_correction = telluric_correction

            # print(telluric_correction)

            print("\n> Applying telluric correction...")

            before_telluric_correction = copy.deepcopy(self.intensity_corrected)

            self.intensity_corrected *= telluric_correction[np.newaxis, :]
            # self.variance_corrected *= telluric_correction[np.newaxis, :]

            # for i in range(self.n_spectra):
            #    self.intensity_corrected[i,:]=self.intensity_corrected[i,:] * telluric_correction

            if plot:
                plot_plot(self.wavelength, telluric_correction, xmin=self.wavelength[0] - 10,
                          xmax=self.wavelength[-1] + 10, statistics=False,
                          ymin=0.9, ymax=2, ptitle="Telluric correction", xlabel="Wavelength [$\mathrm{\AA}$]",
                          vlines=[self.valid_wave_min, self.valid_wave_max])

                integrated_intensity_sorted = np.argsort(self.integrated_fibre)
                region = [integrated_intensity_sorted[-1], integrated_intensity_sorted[0]]
                print("  Example of telluric correction using faintest fibre", region[1], ":")
                ptitle = "Telluric correction in fibre " + np.str(region[1])
                plot_plot(self.wavelength, [before_telluric_correction[region[1]], self.intensity_corrected[region[1]]],
                          xmin=self.wavelength[0] - 10, xmax=self.wavelength[-1] + 10,
                          ymin=np.nanpercentile(self.intensity_corrected[region[1]], 1),
                          ymax=np.nanpercentile(self.intensity_corrected[region[1]], 99),
                          vlines=[self.valid_wave_min, self.valid_wave_max],
                          xlabel="Wavelength [$\mathrm{\AA}$]", ptitle=ptitle)
                print("  Example of telluric correction using brightest fibre", region[0], ":")
                ptitle = "Telluric correction in fibre " + np.str(region[0])
                plot_plot(self.wavelength, [before_telluric_correction[region[0]], self.intensity_corrected[region[0]]],
                          xmin=self.wavelength[0] - 10, xmax=self.wavelength[-1] + 10,
                          ymin=np.nanpercentile(self.intensity_corrected[region[0]], 1),
                          ymax=np.nanpercentile(self.intensity_corrected[region[0]], 99),
                          vlines=[self.valid_wave_min, self.valid_wave_max],
                          xlabel="Wavelength [$\mathrm{\AA}$]", ptitle=ptitle)

            if telluric_correction_file != "":
                self.history.append("- Telluric correction applied reading from file:")
                self.history.append("  " + telluric_correction_file)
            else:
                self.history.append("- Telluric correction applied using a Python variable")
        else:
            self.telluric_correction = np.ones_like(self.wavelength)
            if self.grating in red_gratings:  # and rss_clean == False
                if verbose: print("\n> Telluric correction will NOT be applied...")

    # -----------------------------------------------------------------------------
    # %% =============================================================================
    # Plots
    # =============================================================================
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_spectrum(self, spectrum_number, sky=True, xmin="", xmax="", ymax="", ymin=""):
        """
        Plot spectrum of a particular spaxel.

        Parameters
        ----------
        spectrum_number: float
            fibre to show spectrum.
        sky: boolean (default = True)
            if True the given spectrum has been sky sustracted
            if False it sustract the sky spectrum stored in self.sky_emission
        xmin, xmax : float (default 0, 0 )
            Plot spectrum in wavelength range [xmin, xmax]
            If not given, use xmin = self.wavelength[0] and xmax = self.wavelength[-1]
        ymin, ymax : float (default 0, 0 )
            Plot spectrum in flux range [ymin, ymax]
            If not given, use ymin = np.nanmin(spectrum) and ymax = np.nanmax(spectrum)

        Example
        -------
        >>> rss1.plot_spectrum(550)
        """
        if sky:
            spectrum = self.intensity_corrected[spectrum_number]
        else:
            spectrum = self.intensity_corrected[spectrum_number] + self.sky_emission

        ptitle = self.description + " - Fibre " + np.str(spectrum_number)
        plot_plot(self.wavelength, spectrum, xmin=xmin, xmax=xmax, ymax=ymax, ymin=ymin,
                  ptitle=ptitle, statistics=True)  # TIGRE

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_spectra(self, list_spectra='all', wavelength_range=[0],
                     xmin="", xmax="", ymax=1000, ymin=-100, sky=True,
                     save_file="", fig_size=10):
        """
        Plot spectrum of a list of fibres.

        Parameters
        ----------
        list_spectra:
            spaxels to show spectrum. Default is all.
        wavelength_range : [ float, float ] (default = [0] )
            if given, plots spectrum in wavelength range wavelength_range
        xmin, xmax : float (default "", "" )
            Plot spectra in wavelength range [xmin, xmax]
            If not given, use xmin = self.wavelength[0] and xmax = self.wavelength[-1]
        ymin, ymax : float (default -100, 1000 )
            Plot spectra in flux range [ymin, ymax]
        sky: boolean (default = True)
            if True the given spectrum has been sky sustracted
            if False it sustract the sky spectrum stored in self.sky_emission
        save_file: string (default = "")
            If given, save plot in file "file.extension"
        fig_size: float (default = 10)
            Size of the figure
        Example
        -------
        >>> rss1.plot_spectra([1200,1300])
        """
        plt.figure(figsize=(fig_size, fig_size / 2.5))

        if list_spectra == 'all': list_spectra = list(range(self.n_spectra))
        if len(wavelength_range) == 2: plt.xlim(wavelength_range[0], wavelength_range[1])
        if xmin == "": xmin = np.nanmin(self.wavelength)
        if xmax == "": xmax = np.nanmax(self.wavelength)
        plt.minorticks_on()
        plt.xlabel("Wavelength [$\mathrm{\AA}$]")
        plt.ylabel("Relative Flux")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        for i in list_spectra:
            if sky:
                spectrum = self.intensity_corrected[i]
            else:
                spectrum = self.intensity_corrected[i] + self.sky_emission
            plt.plot(self.wavelength, spectrum)

        if save_file == "":
            plt.show()
        else:
            plt.savefig(save_file)
        plt.close()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def plot_combined_spectrum(self, list_spectra='', sky=True, median=False, ptitle="",
                               xmin="", xmax="", ymax="", ymin="", percentile_min=2, percentile_max=98,
                               plot=True, fig_size=10, save_file=""):
        """
        Plot combined spectrum of a list and return the combined spectrum.

        Parameters
        ----------
        list_spectra:
            spaxels to show combined spectrum. Default is all.
        sky: boolean (default = True)
            if True the given spectrum has been sky sustracted
            if False it sustract the sky spectrum stored in self.sky_emission
        median : boolean (default = False)
            if True the combined spectrum is the median spectrum
            if False the combined spectrum is the sum of the list of spectra
        xmin, xmax : float (default "", "" )
            Plot spectra in wavelength range [xmin, xmax]
            If not given, use xmin = self.wavelength[0] and xmax = self.wavelength[-1]
        ymin, ymax : float (default -100, 1000 )
            Plot spectra in flux range [ymin, ymax]
        plot: boolean (default = True)
            if True it plots the combined spectrum
        fig_size: float (default = 10)
            Size of the figure
        save_file: string (default = "")
            If given, save plot in file "file.extension"

        Example
        -------
        >>> star_peak = rss1.plot_combined_spectrum([550:555])
        """
        if len(list_spectra) == 0:
            list_spectra = list(range(self.n_spectra))

        spectrum = np.zeros_like(self.intensity_corrected[list_spectra[0]])
        value_list = []

        if sky:
            for fibre in list_spectra:
                value_list.append(self.intensity_corrected[fibre])
        else:
            for fibre in list_spectra:
                value_list.append(self.intensity_corrected[fibre] + self.sky_emission)

        if median:
            spectrum = np.nanmedian(value_list, axis=0)
        else:
            spectrum = np.nansum(value_list, axis=0)

        if plot:
            vlines = [self.valid_wave_min, self.valid_wave_max]
            if len(list_spectra) == list_spectra[-1] - list_spectra[0] + 1:
                if ptitle == "": ptitle = "{} - Combined spectrum in range [{},{}]".format(self.description,
                                                                                           list_spectra[0],
                                                                                           list_spectra[-1])
            else:
                if ptitle == "": ptitle = "Combined spectrum using requested fibres"
            plot_plot(self.wavelength, spectrum, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, vlines=vlines,
                      ptitle=ptitle, save_file=save_file, percentile_min=percentile_min, percentile_max=percentile_max)

        return spectrum

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def median_between(self, lambda_min, lambda_max, list_spectra=[]):
        """
        Computes and returns the median flux in range [lambda_min, lambda_max] of a list of spectra.

        Parameters
        ----------
        lambda_min : float
            sets the lower wavelength range (minimum)
        lambda_max : float
            sets the upper wavelength range (maximum)
        list_spectra : list of integers (default = [])
            list with the number of fibres for computing integrated value
            If not given it does all fibres
        """
        index_min = np.searchsorted(self.wavelength, lambda_min)
        index_max = np.searchsorted(self.wavelength, lambda_max) + 1
        if len(list_spectra) == 0:
            list_spectra = list(range(self.n_spectra))

        n_spectra = len(list_spectra)
        medians = np.empty(n_spectra)
        for i in range(n_spectra):
            medians[i] = np.nanmedian(self.intensity[list_spectra[i],
                                      index_min:index_max])
        return medians

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def cut_wave(self, wave, wave_index=-1, plot=False, ymax=""):
        """

        Parameters
        ----------
        wave : wavelength to be cut
            DESCRIPTION.
        plot : TYPE, optional
            DESCRIPTION. The default is False.
        ymax : TYPE, optional
            DESCRIPTION. The default is "".

        Returns
        -------
        corte_wave : TYPE
            DESCRIPTION.

        """
        w = self.wavelength
        if wave_index == -1:
            _w_ = np.abs(w - wave)
            w_min = np.nanmin(_w_)
            wave_index = _w_.tolist().index(w_min)
        else:
            wave = w[wave_index]
        corte_wave = self.intensity_corrected[:, wave_index]

        if plot:
            x = range(self.n_spectra)
            ptitle = "Intensity cut at " + np.str(wave) + " $\mathrm{\AA}$ - index =" + np.str(wave_index)
            plot_plot(x, corte_wave, ymax=ymax, xlabel="Fibre", ptitle=ptitle)

        return corte_wave

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def RSS_map(self, variable=[0], norm=colors.LogNorm(), list_spectra=[], log="",
                title=" - RSS map", clow="", chigh="",
                color_bar_text="Integrated Flux [Arbitrary units]"):
        """
        Plot map showing the offsets, coloured by variable.

        Parameters
        ----------
        norm:
            Normalization scale, default is lineal scale.
            Lineal scale: norm=colors.Normalize()
            Log scale:    norm=colors.LogNorm()
            Power law:    norm=colors.PowerNorm(gamma=1./4.)
        list_spectra : list of floats (default = none)
            List of RSS spectra
        title : string (default = " - RSS image")
            Set plot title
        color_bar_text : string (default = "Integrated Flux [Arbitrary units]")
            Colorbar's label text
        """
        if variable[0] == 0: variable = self.integrated_fibre

        if log == False: norm = colors.Normalize()

        if len(list_spectra) == 0:
            list_spectra = list(range(self.n_spectra))

        plt.figure(figsize=(10, 10))
        plt.scatter(self.offset_RA_arcsec[list_spectra],
                    self.offset_DEC_arcsec[list_spectra],
                    c=variable[list_spectra], cmap=fuego_color_map, norm=norm,
                    s=260, marker="h")
        plt.title(self.description + title)
        plt.xlim(np.nanmin(self.offset_RA_arcsec) - 0.7, np.nanmax(self.offset_RA_arcsec) + 0.7)
        plt.ylim(np.nanmin(self.offset_DEC_arcsec) - 0.7, np.nanmax(self.offset_DEC_arcsec) + 0.7)
        plt.xlabel("$\Delta$ RA [arcsec]")
        plt.ylabel("$\Delta$ DEC [arcsec]")
        plt.minorticks_on()
        plt.grid(which='both')
        plt.gca().invert_xaxis()

        cbar = plt.colorbar()
        if clow == "": clow = np.nanmin(variable[list_spectra])
        if chigh == "": chigh = np.nanmax(variable[list_spectra])
        plt.clim(clow, chigh)
        cbar.set_label(str(color_bar_text), rotation=90, labelpad=40)
        cbar.ax.tick_params()

        plt.show()
        plt.close()

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def RSS_image(self, image="", norm=colors.Normalize(), cmap="seismic_r", clow="", chigh="", labelpad=10, log=False,
                  title=" - RSS image", color_bar_text="Integrated Flux [Arbitrary units]", fig_size=13.5):
        """
        Plot RSS image coloured by variable.
        cmap = "binary_r" nice greyscale

        Parameters
        ----------
        image : string (default = none)
            Specify the name of saved RSS image
        norm:
            Normalization scale, default is lineal scale.
            Lineal scale: norm=colors.Normalize().
            Log scale:    norm=colors.LogNorm()
            Power law:    norm=colors.PowerNorm(gamma=1./4.)
        log:
        cmap : string (default = "seismic_r")
            Colour map for the plot
        clow : float (default = none)
            Lower bound for filtering out outliers, if not specified, 5th percentile is chosen
        chigh: float (default = none)
            Higher bound for filtering out outliers, if not specified, 95th percentile is chosen
        labelpad : integer (default = 10)
            Distance from the colorbar to label in pixels
        title : string (default = " - RSS image")
            Set plot title
        color_bar_text : string (default = "Integrated Flux [Arbitrary units]")
            Colorbar's label text
        fig_size : float (default = 13.5)
            Size of the figure

        """

        if log: norm = colors.LogNorm()

        if image == "":
            image = self.intensity_corrected

        if clow == "":
            clow = np.nanpercentile(image, 5)
        if chigh == "":
            chigh = np.nanpercentile(image, 95)
        if cmap == "seismic_r":
            max_abs = np.nanmax([np.abs(clow), np.abs(chigh)])
            clow = -max_abs
            chigh = max_abs

        plt.figure(figsize=(fig_size, fig_size / 2.5))
        plt.imshow(image, norm=norm, cmap=cmap, clim=(clow, chigh))
        plt.title(self.description + title)
        plt.xlim(0, self.n_wave)
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
    def plot_corrected_vs_uncorrected_spectrum(self, high_fibres=20, low_fibres=0, kernel=51,
                                               fig_size=12, fcal=False, verbose=True):
        """

        Plots the uncorrercted and corrected spectrum of an RSS object together

        Parameters
        ----------
        high_fibres : integer (default = 20)
            Number of highest intensity fibres to use
        low_fibres : integer (default = 0)
            Number of lowest intensity fibres to use. If 0, will do higherst fibres instead
        kernel : odd integer (default = 51)
            Length of kernel for applying median filter
        fig_size : float (default = 12)
            Size of the plots
        fcal : boolean (default = False)
            Calibrate flux units
        verbose : boolean (default = True)
            Print results

        """
        plt.figure(figsize=(fig_size, fig_size / 2.5))

        integrated_intensity_sorted = np.argsort(self.integrated_fibre)
        region = []

        if low_fibres == 0:
            for fibre_ in range(high_fibres):
                region.append(integrated_intensity_sorted[-1 - fibre_])
            if verbose: print("\n> Checking combined spectrum using", high_fibres,
                              "fibres with the highest integrated intensity")
            plt.title(self.object + " - Combined spectrum - " + str(high_fibres) + " fibres with highest intensity")
            I = np.nansum(self.intensity[region], axis=0)
            Ic = np.nansum(self.intensity_corrected[region], axis=0)
        else:
            for fibre_ in range(low_fibres):
                region.append(integrated_intensity_sorted[fibre_])
            if verbose: print("\n> Checking median spectrum using", low_fibres,
                              "fibres with the lowest integrated intensity")
            plt.title(self.object + " - Median spectrum - " + str(low_fibres) + " fibres with lowest intensity")
            I = np.nanmedian(self.intensity[region], axis=0)
            Ic = np.nanmedian(self.intensity_corrected[region], axis=0)
        if verbose: print("  which are :", region)

        Ic_m, fit = fit_smooth_spectrum(self.wavelength, Ic, kernel=kernel, verbose=False,  # edgelow=0, edgehigh=0,
                                        order=3, plot=False, hlines=[0.], fcal=False)  # ptitle= ptitle,

        I_ymin = np.nanmin(Ic_m)
        I_ymax = np.nanmax(Ic_m)
        I_rango = I_ymax - I_ymin

        plt.plot(self.wavelength, I, 'r-', label='Uncorrected', alpha=0.3)
        plt.plot(self.wavelength, Ic, 'g-', label='Corrected', alpha=0.4)

        text = "Corrected with median filter " + np.str(kernel)

        if low_fibres > 0:
            plt.plot(self.wavelength, Ic_m, 'b-', label=text, alpha=0.4)
            plt.plot(self.wavelength, fit, color="purple", linestyle='-', label='Fit', alpha=0.4)
        if fcal:
            ylabel = "Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]"
        else:
            ylabel = "Flux [counts]"
        plt.ylabel(ylabel)
        plt.xlabel("Wavelength [$\mathrm{\AA}$]")
        plt.minorticks_on()
        plt.xlim(self.wavelength[0] - 10, self.wavelength[-1] + 10)
        plt.axvline(x=self.valid_wave_min, color='k', linestyle='--', alpha=0.8)
        plt.axvline(x=self.valid_wave_max, color='k', linestyle='--', alpha=0.8)
        if low_fibres == 0:
            plt.ylim([I_ymin - I_rango / 10, I_ymax + I_rango / 10])
        else:
            plt.axhline(y=0., color="k", linestyle="--", alpha=0.8)  # teta
            I_ymin = np.nanpercentile(Ic, 2)
            I_ymax = np.nanpercentile(Ic, 98)
            I_rango = I_ymax - I_ymin
            plt.ylim([I_ymin - I_rango / 10, I_ymax + I_rango / 10])
        plt.legend(frameon=False, loc=4, ncol=4)
        plt.show()
        plt.close()

    # %% =============================================================================
    # Throughput
    # =============================================================================
    # -----------------------------------------------------------------------------
    def find_relative_throughput(self):  # TODO: New dummy function
        mean_count = np.nanmean(self.intensity_corrected, axis=1)
        perc50 = np.nanpercentile(mean_count, 50)
        # self.low_throughput = mean_count < perc5
        # self.high_throughput = mean_count > perc95
        return mean_count / perc50

    def apply_throughput_2D(self, throughput_2D=[], throughput_2D_file="", path="", plot=True):
        """
        Apply throughput_2D using the information of a variable or a fits file.
        """
        if len(throughput_2D) > 0:
            print("\n> Applying 2D throughput correction using given variable ...")
            self.throughput_2D = throughput_2D
            self.history.append("- Applied 2D throughput correction using a variable")
        else:
            if path != "": throughput_2D_file = full_path(throughput_2D_file, path)
            print("\n> Applying 2D throughput correction reading file :")
            print(" ", throughput_2D_file)
            self.history.append("- Applied 2D throughput correction using file:")
            self.history.append("  " + throughput_2D_file)
            ftf = fits.open(throughput_2D_file)
            self.throughput_2D = ftf[0].data
        if plot:
            print("\n> Plotting map BEFORE correcting throughput:")
            self.RSS_image()

        self.intensity_corrected = self.intensity_corrected / self.throughput_2D
        if plot:
            print("  Plotting map AFTER correcting throughput:")
            self.RSS_image()
        # %% =============================================================================

    # Flatfield
    # =============================================================================
    # -----------------------------------------------------------------------------
    def apply_flatfield(self, flatfield):  # TODO: New function
        # TODO: Add description
        if flatfield.shape != self.intensity_corrected.shape:
            raise NameError('ERROR: Flatfield dim: {}, RSS dim: {}' \
                            .format(flatfield.shape,
                                    self.intensity_corrected.shape))
        else:
            self.intensity_corrected /= flatfield
            self.variance_corrected /= flatfield
        self.corrections.append('Flatfield correction')

    # %% =============================================================================
    # Cleaning residuals
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def fix_edges(self, kernel_size=101, disp=1.5, fix_from=0, median_from=0, fix_to=0, median_to=0,
                  only_red_edge=False, only_blue_edge=False, do_blue_edge=True, do_red_edge=True, verbose=True):
        """
        Fixing edges of a RSS file

        Parameters
        ----------
        kernel_size : integer odd number, optional
            kernel size for smoothing. The default is 101.
        disp : real, optional
            max dispersion. The default is 1.5.
        fix_from : real
            fix red edge from this wavelength. If not given uses self.valid_wave_max
        median_from : real, optional
            wavelength from here use median value for fixing red edge. If not given is self.valid_wave_max - 300
        fix_to : real, optional
            fix blue edge to this wavelength. If not given uses self.valid_wave_min
        median_to : TYPE, optional
            wavelength till here use median value for fixing blue edge. If not given is self.valid_wave_min + 200.
        only_red_edge:  boolean (default = False)
            Fix only the red edge
        only_blue_edge:  boolean (default = False)
            Fix only the blue edge
        do_blue_edge:  boolean (default = True)
            Fix the blue edge
        do_red_edge:  boolean (default = True)
            Fix the red edge
        verbose: boolean (default = True)
            Print what is doing
        """
        self.apply_mask(make_nans=True, verbose=False)
        if only_red_edge == True:  do_blue_edge = False
        if only_blue_edge == True: do_red_edge = False
        if verbose: print("\n> Fixing the BLUE and RED edges of the RSS file...")
        w = self.wavelength
        self.RSS_image(title=" - Before correcting edges")
        if fix_from == 0: fix_from = self.valid_wave_max
        if median_from == 0: median_from = self.valid_wave_max - 300.
        if fix_to == 0: fix_to = self.valid_wave_min
        if median_to == 0: median_to = self.valid_wave_min + 200.
        self.apply_mask(make_nans=True, verbose=False)
        for i in range(self.n_spectra):
            if do_red_edge: self.intensity_corrected[i] = fix_red_edge(w, self.intensity_corrected[i],
                                                                       fix_from=fix_from, median_from=median_from,
                                                                       kernel_size=kernel_size, disp=disp)
            if do_blue_edge: self.intensity_corrected[i] = fix_blue_edge(w, self.intensity_corrected[i],
                                                                         kernel_size=kernel_size, disp=disp,
                                                                         fix_to=fix_to, median_to=median_to)

        self.RSS_image(title=" - After correcting edges")
        if do_blue_edge: self.history.append("- Blue edge has been corrected to " + np.str(fix_to))
        if do_red_edge: self.history.append("- Red edge has been corrected from " + np.str(fix_from))

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def kill_cosmics(self, brightest_line_wavelength, width_bl=20., fibre_list=[], max_number_of_cosmics_per_fibre=10,
                     kernel_median_cosmics=5, cosmic_higher_than=100, extra_factor=1.,
                     plot_waves=[], plot_cosmic_image=True, plot_RSS_images=True, plot=True,
                     verbose=True, warnings=True):
        """
        Kill cosmics in a RSS.

        Parameters
        ----------
        brightest_line_wavelength : float
            wavelength in A of the brightest emission line found in the RSS
        width_bl : float, optional
            broad in A of the bright emission line. The default is 20..
        fibre_list : array of integers, optional
            fibres to be modified. The default is "", that will be all
        kernel_median_cosmics : odd integer (default = 5)
            Width of the median filter
        cosmic_higher_than : float (default = 100)
            Upper boundary for pixel flux to be considered a cosmic
        extra_factor : float (default = 1.)
            Extra factor to be considered as maximum value
        plot_waves : list, optional
            list of wavelengths to plot. The default is none.
        plot_cosmic_image : boolean (default = True)
            Plot the image with the cosmic identification.
        plot_RSS_images : boolean (default = True)
            Plot comparison between RSS images before and after correcting cosmics
        plot : boolean (default = False)
            Display all the plots
        verbose: boolean (default = True)
            Print what is doing
        warnings: boolean (default = True)
            Print warnings

        Returns
        -------
        Save the corrected RSS to self.intensity_corrected

        """
        g = copy.deepcopy(self.intensity_corrected)

        if plot == False:
            plot_RSS_images = False
            plot_cosmic_image = False

        x = range(self.n_spectra)
        w = self.wavelength
        if len(fibre_list) == 0:
            fibre_list_ALL = True
            fibre_list = list(range(self.n_spectra))
            if verbose: print("\n> Finding and killing cosmics in all fibres...")
        else:
            fibre_list_ALL = False
            if verbose: print("\n> Finding and killing cosmics in given fibres...")

        if brightest_line_wavelength == 0:
            if warnings or verbose: print("\n\n\n  WARNING !!!!! brightest_line_wavelength is NOT given!\n")

            median_spectrum = self.plot_combined_spectrum(plot=plot, median=True,
                                                          list_spectra=self.integrated_fibre_sorted[-11:-1],
                                                          ptitle="Combined spectrum using 10 brightest fibres",
                                                          percentile_max=99.5, percentile_min=0.5)
            # brightest_line_wavelength=w[np.int(self.n_wave/2)]
            brightest_line_wavelength = self.wavelength[median_spectrum.tolist().index(np.nanmax(median_spectrum))]

            if brightest_line_wavelength < self.valid_wave_min: brightest_line_wavelength = self.valid_wave_min
            if brightest_line_wavelength > self.valid_wave_max: brightest_line_wavelength = self.valid_wave_max

            if warnings or verbose: print(
                "  Assuming brightest_line_wavelength is the max of median spectrum of 10 brightest fibres =",
                brightest_line_wavelength)

        # Get the cut at the brightest_line_wavelength
        corte_wave_bl = self.cut_wave(brightest_line_wavelength)
        gc_bl = medfilt(corte_wave_bl, kernel_size=kernel_median_cosmics)
        max_val = np.abs(corte_wave_bl - gc_bl)

        if plot:
            ptitle = "Intensity cut at brightest line wavelength = " + np.str(
                np.round(brightest_line_wavelength, 2)) + " $\mathrm{\AA}$ and extra_factor = " + np.str(extra_factor)
            plot_plot(x, [max_val, extra_factor * max_val], percentile_max=99, xlabel="Fibre", ptitle=ptitle,
                      ylabel="abs (f - medfilt(f))",
                      label=["intensity_cut", "intensity_cut * extra_factor"])

        # List of waves to plot:
        plot_waves_index = []
        for wave in plot_waves:
            wave_min_vector = np.abs(w - wave)
            plot_waves_index.append(wave_min_vector.tolist().index(np.nanmin(wave_min_vector)))
        if len(plot_waves) > 0: print("  List of waves to plot:", plot_waves)

        # Start loop
        lista_cosmicos = []
        cosmic_image = np.zeros_like(self.intensity_corrected)
        for i in range(len(w)):
            wave = w[i]
            # Perhaps we should include here not cleaning in emission lines...
            correct_cosmics_in_fibre = True
            if width_bl != 0:
                if wave > brightest_line_wavelength - width_bl / 2 and wave < brightest_line_wavelength + width_bl / 2:
                    if verbose: print(
                        "  Skipping {:.4f} as it is adjacent to brightest line wavelenght {:.4f}".format(wave,
                                                                                                         brightest_line_wavelength))
                    correct_cosmics_in_fibre = False
            if correct_cosmics_in_fibre:
                if i in plot_waves_index:
                    plot_ = True
                    verbose_ = True
                else:
                    plot_ = False
                    verbose_ = False
                corte_wave = self.cut_wave(wave)
                cosmics_found = find_cosmics_in_cut(x, corte_wave, corte_wave_bl * extra_factor, line_wavelength=wave,
                                                    plot=plot_, verbose=verbose_, cosmic_higher_than=cosmic_higher_than)
                if len(cosmics_found) <= max_number_of_cosmics_per_fibre:
                    for cosmic in cosmics_found:
                        lista_cosmicos.append([wave, cosmic])
                        cosmic_image[cosmic, i] = 1.
                else:
                    if warnings: print("  WARNING! Wavelength", np.round(wave, 2), "has", len(cosmics_found),
                                       "cosmics found, this is larger than", max_number_of_cosmics_per_fibre,
                                       "and hence these are NOT corrected!")

        # Check number of cosmics found
        if plot_cosmic_image: self.RSS_image(image=cosmic_image, cmap="binary_r", title=" - Cosmics identification")
        # print(lista_cosmicos)
        if verbose: print("\n> Total number of cosmics found = ", len(lista_cosmicos), " , correcting now ...")

        if plot_RSS_images: self.RSS_image(cmap="binary_r", title=" - Before correcting cosmics")

        if fibre_list_ALL == False and verbose == True: print("  Correcting cosmics in selected fibres...")
        cosmics_cleaned = 0
        for fibre in fibre_list:
            if np.nansum(cosmic_image[fibre]) > 0:  # A cosmic is found
                # print("Fibre ",fibre," has cosmics!")
                f = g[fibre]
                gc = medfilt(f, kernel_size=21)
                bad_indices = [i for i, x in enumerate(cosmic_image[fibre]) if x == 1]
                if len(bad_indices) <= max_number_of_cosmics_per_fibre:
                    for index in bad_indices:
                        g[fibre, index] = gc[index]
                        cosmics_cleaned = cosmics_cleaned + 1
                else:
                    cosmic_image[fibre] = np.zeros_like(w)
                    if warnings: print("  WARNING! Fibre", fibre, "has", len(bad_indices),
                                       "cosmics found, this is larger than", max_number_of_cosmics_per_fibre,
                                       "and hence is NOT corrected!")

        self.intensity_corrected = copy.deepcopy(g)
        if plot_RSS_images: self.RSS_image(cmap="binary_r", title=" - After correcting cosmics")

        # Check number of cosmics eliminated
        if verbose: print("\n> Total number of cosmics cleaned = ", cosmics_cleaned)
        if cosmics_cleaned != len(lista_cosmicos):
            if plot_cosmic_image: self.RSS_image(image=cosmic_image, cmap="binary_r", title=" - Cosmics cleaned")

        self.history.append("- " + np.str(cosmics_cleaned) + " cosmics cleaned using:")
        self.history.append("  brightest_line_wavelength = " + np.str(brightest_line_wavelength))
        self.history.append(
            "  width_bl = " + np.str(width_bl) + ", kernel_median_cosmics = " + np.str(kernel_median_cosmics))
        self.history.append(
            "  cosmic_higher_than = " + np.str(cosmic_higher_than) + ", extra_factor = " + np.str(extra_factor))
        return g

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def clean_extreme_negatives(self, fibre_list=[], percentile_min=0.5, plot=True, verbose=True):
        """
        Remove pixels that have extreme negative values (that is below percentile_min) and replace for the median value

        Parameters
        ----------
        fibre_list : list of integers (default all)
            List of fibers to clean. The default is [], that means it will do everything.
        percentile_min : float, (default = 0.5)
            Minimum value accepted as good.
        plot : boolean (default = False)
            Display all the plots
        verbose: boolean (default = True)
            Print what is doing
        """
        if len(fibre_list) == 0:
            fibre_list = list(range(self.n_spectra))
            if verbose: print("\n> Correcting the extreme negatives in all fibres, making any pixel below")
        else:
            if verbose: print("\n> Correcting the extreme negatives in given fibres, making any pixel below")

        g = copy.deepcopy(self.intensity_corrected)
        minimo = np.nanpercentile(g, percentile_min)

        if verbose:
            print("  np.nanpercentile(intensity_corrected, ", percentile_min, ") = ", np.round(minimo, 2))
            print("  to have the median value of the fibre...")

        for fibre in fibre_list:
            g[fibre] = [np.nanmedian(g[fibre]) if x < minimo else x for x in g[fibre]]
        self.history.append(
            "- Extreme negatives (values below percentile " + np.str(np.round(percentile_min, 3)) + " = " + np.str(
                np.round(minimo, 3)) + " ) cleaned")

        if plot:
            correction_map = g / self.intensity_corrected  # / g
            self.RSS_image(image=correction_map, cmap="binary_r", title=" - Correction map")

        self.intensity_corrected = g

    # %% =============================================================================


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#
# Tasks involving RSS but not part of class RSS
#
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def median_filter(intensity_corrected, n_spectra, n_wave, win_sky=151):
    """
    Matt's code to get a median filter of all fibres in a RSS
    This is useful when having 2D sky
    """

    medfilt_sky = np.zeros((n_spectra, n_wave))
    for wave in range(n_wave):
        medfilt_sky[:, wave] = medfilt(intensity_corrected[:, wave], kernel_size=win_sky)

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
        nresponse_ = medfilt(spectrum_, kernel)
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
            throughput_2D[i] = medfilt(throughput_2D_[i], kernel_throughput)
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
# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
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

