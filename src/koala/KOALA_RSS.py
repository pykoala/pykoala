#!/usr/bin/python
# -*- coding: utf-8 -*-

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import copy
# from scipy import signal
# Disable some annoying warnings
import warnings

from koala.RSS import RSS, coord_range
from koala.io import full_path

warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.CRITICAL)

red_gratings = ["385R","1000R","2000R", "1000I", "1700D","1700I"]
blue_gratings = ["580V" , "1500V" ,"1700B" , "3200B" , "2500V"]   


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
                 flat=None,  # normalized flat, if needed
                 no_nans=False, mask="", mask_file="", plot_mask=False,  # Mask if given
                 valid_wave_min=0, valid_wave_max=0,  # These two are not needed if Mask is given
                 apply_throughput=False,
                 throughput_2D=[], throughput_2D_file="", throughput_2D_wavecor=False,
                 correct_ccd_defects=False, remove_5577=False, kernel_correct_ccd_defects=51, fibre_p=-1,
                 plot_suspicious_fibres=False,
                 fix_wavelengths=False, sol=[0, 0, 0],
                 do_extinction=False,
                 telluric_correction=[0], telluric_correction_file="",
                 sky_method="none", n_sky=50, sky_fibres=[],  # do_sky=True
                 sky_spectrum=[], sky_rss=[0], scale_sky_rss=0, scale_sky_1D=0.,
                 maxima_sigma=3.,
                 sky_spectrum_file="",
                 brightest_line="Ha", brightest_line_wavelength=0, 
                 sky_lines_file="", exclude_wlm=[[0, 0]], emission_line_file = "",
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
                 width_bl=20., kernel_median_cosmics=5, cosmic_higher_than=100., extra_factor=1.,
                 max_number_of_cosmics_per_fibre=12,
                 warnings=True, verbose=True, print_summary=False,
                 plot=True, plot_final_rss=True,
                 log= True, gamma = 0.,fig_size=12):

        # --------------------------------------------------------------------
        # Reading KOALA data using the products of 2dFdr...
        # --------------------------------------------------------------------

        # Create RSS object
        super(KOALA_RSS, self).__init__()
        #if path != "": filename = full_path(filename, path)

        self.instrument["instrument"] ="KOALA + AAOmega"
        
        self.read_rss_file(filename, path, rss_clean=rss_clean, instrument = self.instrument["instrument"],
                           warnings=warnings, verbose=verbose)
                
        # if verbose: print("\n> Reading file", '"' + filename + '"', "...")
        # rss_fits_file = fits.open(filename)  # Open file

        # #  General info:
        # self.object = rss_fits_file[0].header['OBJECT']
        # self.filename = filename
        # self.description = self.object + ' \n ' + filename
        # self.RA_centre_deg = rss_fits_file[2].header['CENRA'] * 180 / np.pi
        # self.DEC_centre_deg = rss_fits_file[2].header['CENDEC'] * 180 / np.pi
        # self.exptime = rss_fits_file[0].header['EXPOSED']
        # self.history_RSS = rss_fits_file[0].header['HISTORY']

        # # Read good/bad spaxels
        # all_spaxels = list(range(len(rss_fits_file[2].data)))
        # quality_flag = [rss_fits_file[2].data[i][1] for i in all_spaxels]
        # good_spaxels = [i for i in all_spaxels if quality_flag[i] == 1]
        # bad_spaxels = [i for i in all_spaxels if quality_flag[i] == 0]

        # # Create wavelength
        # wcsKOALA = WCS(rss_fits_file[0].header)
        # index_wave = np.arange(rss_fits_file[0].header['NAXIS1'])
        # wavelength = wcsKOALA.dropaxis(1).wcs_pix2world(index_wave, 0)[0]
        # self.wavelength = wavelength
        # self.n_wave = len(wavelength)
        
        # # For WCS
        # self.CRVAL1_CDELT1_CRPIX1 = []
        # self.CRVAL1_CDELT1_CRPIX1.append(rss_fits_file[0].header['CRVAL1'])
        # self.CRVAL1_CDELT1_CRPIX1.append(rss_fits_file[0].header['CDELT1'])
        # self.CRVAL1_CDELT1_CRPIX1.append(rss_fits_file[0].header['CRPIX1'])
        
        # # Read intensity using rss_fits_file[0]
        # intensity = rss_fits_file[0].data[good_spaxels]
        
        # # Read errors using rss_fits_file[1]
        # try:
        #     variance = rss_fits_file[1].data[good_spaxels]
        # except Exception:
        #     variance = copy.deepcopy(intensity)
        #     if warnings or verbose: print("\n  WARNING! Variance extension not found in fits file!")

        # if not rss_clean and verbose:
        #     print("\n  Number of spectra in this RSS =", len(rss_fits_file[0].data), ",  number of good spectra =",
        #           len(good_spaxels), " ,  number of bad spectra =", len(bad_spaxels))
        #     if len(bad_spaxels) > 0: print("  Bad fibres =", bad_spaxels)

        # # Read spaxel positions on sky using rss_fits_file[2]
        # self.header2_data = rss_fits_file[2].data

        # # But only keep the GOOD data!
        # # CAREFUL !! header 2 has the info of BAD fibres, if we are reading 
        # # from our created RSS files we have to do it in a different way...

        # if len(bad_spaxels) == 0:
        #     offset_RA_arcsec_ = []
        #     offset_DEC_arcsec_ = []
        #     for i in range(len(good_spaxels)):
        #         offset_RA_arcsec_.append(self.header2_data[i][5])
        #         offset_DEC_arcsec_.append(self.header2_data[i][6])
        #     offset_RA_arcsec = np.array(offset_RA_arcsec_)
        #     offset_DEC_arcsec = np.array(offset_DEC_arcsec_)

        # else:
        #     offset_RA_arcsec = np.array([rss_fits_file[2].data[i][5]
        #                                  for i in good_spaxels])
        #     offset_DEC_arcsec = np.array([rss_fits_file[2].data[i][6]
        #                                   for i in good_spaxels])

        #     #self.ID = np.array([rss_fits_file[2].data[i][0] for i in good_spaxels])  # These are the good fibres

        # # Get ZD, airmass
        # self.ZDSTART = rss_fits_file[0].header['ZDSTART']
        # self.ZDEND = rss_fits_file[0].header['ZDEND']
        # ZD = (self.ZDSTART + self.ZDEND) / 2
        # self.airmass = 1 / np.cos(np.radians(ZD))
        # self.extinction_correction = np.ones(self.n_wave)

        # # KOALA-specific stuff
        # self.PA = rss_fits_file[0].header['TEL_PA']
        # self.grating = rss_fits_file[0].header['GRATID']
        # # Check RED / BLUE arm for AAOmega
        # if (rss_fits_file[0].header['SPECTID'] == "RD"):
        #     AAOmega_Arm = "RED"
        # if (rss_fits_file[0].header['SPECTID'] == "BL"):  
        #     AAOmega_Arm = "BLUE"
        # self.instrument.append(AAOmega_Arm)

        # rss_fits_file.close()
        
        # # Check that dimensions match KOALA numbers
        # if self.n_wave != 2048 and len(all_spaxels) != 1000:
        #     if warnings or verbose:
        #         print("\n *** WARNING *** : These numbers are NOT the standard ones for KOALA")

        # if verbose: print("\n> Setting the data for this file:")

        # if variance.shape != intensity.shape:
        #     if warnings or verbose:
        #         print("\n* ERROR: * the intensity and variance arrays are",
        #               intensity.shape, "and", variance.shape, "respectively\n")
        #     raise ValueError
        # n_dim = len(intensity.shape)
        # if n_dim == 2:
        #     self.intensity = intensity
        #     self.variance = variance
        # elif n_dim == 1:
        #     self.intensity = intensity.reshape((1, self.n_wave))
        #     self.variance = variance.reshape((1, self.n_wave))
        # else:
        #     if warnings or verbose:
        #         print("\n* ERROR: * the intensity matrix supplied has", n_dim, "dimensions\n")
        #     raise ValueError

        # self.n_spectra = self.intensity.shape[0]

        # if verbose:
        #     print("  Found {} spectra with {} wavelengths".format(self.n_spectra, self.n_wave),
        #           "between {:.2f} and {:.2f} Angstrom".format(self.wavelength[0], self.wavelength[-1]))
        # if self.intensity.shape[1] != self.n_wave:
        #     if warnings or verbose:
        #         print("\n* ERROR: * spectra have", self.intensity.shape[1], "wavelengths rather than", self.n_wave)
        #     raise ValueError
        # if (len(offset_RA_arcsec) != self.n_spectra) |(len(offset_DEC_arcsec) != self.n_spectra):
        #     if warnings | verbose:
        #         print("\n* ERROR: * offsets (RA, DEC) = ({},{})".format(len(self.offset_RA_arcsec),
        #                                                                 len(self.offset_DEC_arcsec)),
        #               "rather than", self.n_spectra)
        #     raise ValueError
        # else:
        #     self.offset_RA_arcsec = offset_RA_arcsec
        #     self.offset_DEC_arcsec = offset_DEC_arcsec

        # # Check if NARROW (spaxel_size = 0.7 arcsec)
        # # or WIDE (spaxel_size=1.25) field of view
        # # (if offset_max - offset_min > 31 arcsec in both directions)
        # if np.max(offset_RA_arcsec) - np.min(offset_RA_arcsec) > 31 or \
        #         np.max(offset_DEC_arcsec) - np.min(offset_DEC_arcsec) > 31:
        #     self.spaxel_size = 1.25
        #     field = "WIDE"
        # else:
        #     self.spaxel_size = 0.7
        #     field = "NARROW"
            
        # self.instrument.append(field)
        # self.instrument.append(self.spaxel_size)

        # # Get min and max for rss
        # self.RA_min, self.RA_max, self.DEC_min, self.DEC_max = coord_range([self])
        # self.DEC_segment = (self.DEC_max - self.DEC_min) * 3600.  # +1.25 for converting to total field of view
        # self.RA_segment = (self.RA_max - self.RA_min) * 3600.  # +1.25

        # # Deep copy of intensity into intensity_corrected
        # self.intensity_corrected = copy.deepcopy(self.intensity)
        # self.variance_corrected = variance.copy()
        
        # ---------------------------------------------------
        # ------------- PROCESSING THE RSS FILE -------------
        # ---------------------------------------------------
        
        self.process_rss(save_rss_to_fits_file=save_rss_to_fits_file, rss_clean=rss_clean,
                 path =path, flat=flat,  
                 no_nans=no_nans, mask=mask, mask_file=mask_file, plot_mask=plot_mask,  # Mask if given
                 valid_wave_min=valid_wave_min, valid_wave_max=valid_wave_max,  # These two are not needed if Mask is given
                 apply_throughput=apply_throughput,
                 throughput_2D=throughput_2D, throughput_2D_file=throughput_2D_file, throughput_2D_wavecor=throughput_2D_wavecor,
                 correct_ccd_defects=correct_ccd_defects, remove_5577=remove_5577, kernel_correct_ccd_defects=kernel_correct_ccd_defects, 
                 fibre_p=fibre_p, plot_suspicious_fibres=plot_suspicious_fibres,
                 fix_wavelengths=fix_wavelengths, sol=sol,
                 do_extinction=do_extinction,
                 telluric_correction=telluric_correction, telluric_correction_file=telluric_correction_file,
                 sky_method=sky_method, n_sky=n_sky, sky_fibres=sky_fibres,  # do_sky=True
                 sky_spectrum=sky_spectrum, sky_rss=sky_rss, scale_sky_rss=scale_sky_rss, scale_sky_1D=scale_sky_1D,
                 maxima_sigma=maxima_sigma,
                 sky_spectrum_file=sky_spectrum_file,
                 brightest_line=brightest_line, brightest_line_wavelength=brightest_line_wavelength, 
                 sky_lines_file=sky_lines_file, exclude_wlm=exclude_wlm, emission_line_file = emission_line_file,
                 is_sky=is_sky, win_sky=win_sky, auto_scale_sky=auto_scale_sky, ranges_with_emission_lines=ranges_with_emission_lines, cut_red_end=cut_red_end,
                 correct_negative_sky=correct_negative_sky,
                 order_fit_negative_sky=order_fit_negative_sky, kernel_negative_sky=kernel_negative_sky, individual_check=individual_check,
                 use_fit_for_negative_sky=use_fit_for_negative_sky,
                 force_sky_fibres_to_zero=force_sky_fibres_to_zero,
                 high_fibres=high_fibres, low_fibres=low_fibres,
                 sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, cut_sky=cut_sky, fmin=fmin, fmax=fmax,
                 individual_sky_substraction=individual_sky_substraction,  # fibre_list=[100,200,300,400,500,600,700,800,900],
                 id_el=id_el, cut=cut, broad=broad, plot_id_el=plot_id_el, id_list=id_list,
                 fibres_to_fix=fibres_to_fix,
                 clean_sky_residuals=clean_sky_residuals, features_to_fix=features_to_fix, sky_fibres_for_residuals=sky_fibres_for_residuals,
                 remove_negative_median_values=remove_negative_median_values,
                 fix_edges=fix_edges,
                 clean_extreme_negatives=clean_extreme_negatives, percentile_min=percentile_min,
                 clean_cosmics=clean_cosmics,
                 width_bl=width_bl, kernel_median_cosmics=kernel_median_cosmics, cosmic_higher_than=cosmic_higher_than, extra_factor=extra_factor,
                 max_number_of_cosmics_per_fibre=max_number_of_cosmics_per_fibre,
                 warnings=warnings, verbose=verbose, print_summary=print_summary,
                 plot=plot, plot_final_rss=plot_final_rss,
                 log= log, gamma = gamma,fig_size=fig_size)


        if rss_clean == False and verbose: print("\n> KOALA RSS file read !")

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    # KOALA specific tasks

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
def KOALA_offsets(s, pa):
    print("\n> Offsets towards North and East between pointings," \
          "according to KOALA manual, for pa =", pa, "degrees")
    pa *= np.pi / 180
    print("  a -> b :", s * np.sin(pa), -s * np.cos(pa))
    print("  a -> c :", -s * np.sin(60 - pa), -s * np.cos(60 - pa))
    print("  b -> d :", -np.sqrt(3) * s * np.cos(pa), -np.sqrt(3) * s * np.sin(pa))
