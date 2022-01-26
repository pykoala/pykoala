#!/usr/bin/python
# -*- coding: utf-8 -*-

from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import copy
#from scipy import signal
# Disable some annoying warnings
import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.CRITICAL)

import sys
# from utils import generic_functions as gf
# TODO: Fix this
#sys.path.append('../utils')
#import constants
#import generic_functions as gf 
#from rss import RSS


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
    def __init__(self, filename, save_rss_to_fits_file ="",  rss_clean = False, path= "",
                 flat="", # normalized flat, if needed 
                 no_nans = False, mask="", mask_file="", plot_mask=False, # Mask if given
                 valid_wave_min=0, valid_wave_max=0, # These two are not needed if Mask is given
                 apply_throughput=False, 
                 throughput_2D=[], throughput_2D_file="", throughput_2D_wavecor = False,
                 #nskyflat=True, skyflat="", throughput_file ="", nskyflat_file="", plot_skyflat=False, 
                 correct_ccd_defects = False, remove_5577 = False, kernel_correct_ccd_defects=51, fibre_p=-1,
                 plot_suspicious_fibres=False,
                 fix_wavelengths=False, sol=[0,0,0],
                 do_extinction=False,
                 telluric_correction = [0],  telluric_correction_file="",
                 sky_method="none", n_sky=50, sky_fibres=[], # do_sky=True 
                 sky_spectrum=[], sky_rss=[0], scale_sky_rss=0, scale_sky_1D=0., 
                 maxima_sigma = 3.,
                 sky_spectrum_file = "",
                 brightest_line="Ha", brightest_line_wavelength = 0, sky_lines_file="", exclude_wlm=[[0,0]],
                 is_sky=False, win_sky=0, auto_scale_sky = False, ranges_with_emission_lines = [0], cut_red_end = 0,
                 correct_negative_sky = False,
                 order_fit_negative_sky = 3, kernel_negative_sky = 51, individual_check = True, use_fit_for_negative_sky = False,
                 force_sky_fibres_to_zero = True,
                 high_fibres=20, low_fibres=10,
                 sky_wave_min = 0, sky_wave_max =0, cut_sky=5., fmin=1, fmax=10,                  
                 individual_sky_substraction=False, #fibre_list=[100,200,300,400,500,600,700,800,900], 
                 id_el=False,  cut=1.5, broad=1.0, plot_id_el= False, id_list=[0],              
                 fibres_to_fix=[],                                     
                 clean_sky_residuals = False, features_to_fix =[], sky_fibres_for_residuals=[],
                 remove_negative_median_values = False,
                 fix_edges = False,
                 clean_extreme_negatives = False, percentile_min = 0.5,
                 clean_cosmics = False, #show_cosmics_identification = True,                                                            
                 width_bl = 20., kernel_median_cosmics = 5, cosmic_higher_than = 100., extra_factor = 1., max_number_of_cosmics_per_fibre = 15,
                 warnings=True, verbose = True,
                 plot=True, plot_final_rss=True, norm=colors.LogNorm(), fig_size=12):                 

        # ---------------------------------------------- Checking some details 
             
        if rss_clean:                     # Just read file if rss_clean = True
            apply_throughput = False
            correct_ccd_defects = False
            fix_wavelengths=False
            sol=[0,0,0]
            sky_method="none"
            do_extinction = False
            telluric_correction = [0]
            telluric_correction_file = ""
            id_el = False
            clean_sky_residuals = False
            fix_edges= False
            #plot_final_rss = plot
            plot = False
            correct_negative_sky = False
            clean_cosmics = False
            clean_extreme_negatives = False
            remove_negative_median_values = False
            verbose=False
  
        if len(telluric_correction_file) > 0 or telluric_correction[0] != 0 :
            do_telluric_correction = True
        else:
            do_telluric_correction = False


        if apply_throughput == False and correct_ccd_defects == False and fix_wavelengths==False and sky_method == "none" and do_extinction == False and telluric_correction == [0] and clean_sky_residuals == False and correct_negative_sky == False and clean_cosmics == False and fix_edges==False and  clean_extreme_negatives == False and remove_negative_median_values == False and do_telluric_correction == False and is_sky == False:    
            # If nothing is selected to do, we assume that the RSS file is CLEAN
            rss_clean = True
            #plot_final_rss = plot
            plot = False
            verbose=False
        
        if sky_method not in ["self" , "selffit"]: force_sky_fibres_to_zero = False # We don't have sky fibres, sky spectrum is given
        self.sky_fibres=[]
        
        # --------------------------------------------------------------------
        # ------------------------------------------------ 0. Reading the data
        # --------------------------------------------------------------------
        
        # Create RSS object
        super(KOALA_RSS, self).__init__()
        
        if path != "" : filename = full_path(filename,path)

        print("\n> Reading file", '"'+filename+'"', "...")
        RSS_fits_file = fits.open(filename)  # Open file
        #self.rss_list = []

        #  General info:
        self.object = RSS_fits_file[0].header['OBJECT']
        self.filename = filename
        self.description = self.object + ' \n ' + filename
        self.RA_centre_deg = RSS_fits_file[2].header['CENRA'] * 180/np.pi
        self.DEC_centre_deg = RSS_fits_file[2].header['CENDEC'] * 180/np.pi
        self.exptime = RSS_fits_file[0].header['EXPOSED']
        self.history_RSS = RSS_fits_file[0].header['HISTORY']
        self.history = []
        if sol[0] in [0,-1]:
            self.sol=[0,0,0]
        else:
            self.sol=sol

        # Read good/bad spaxels
        all_spaxels = list(range(len(RSS_fits_file[2].data)))
        quality_flag = [RSS_fits_file[2].data[i][1] for i in all_spaxels]
        good_spaxels = [i for i in all_spaxels if quality_flag[i] == 1]
        bad_spaxels = [i for i in all_spaxels if quality_flag[i] == 0]
                        
        # Create wavelength, intensity, and variance arrays only for good spaxels
        wcsKOALA = WCS(RSS_fits_file[0].header)
        #variance = RSS_fits_file[1].data[good_spaxels]
        index_wave = np.arange(RSS_fits_file[0].header['NAXIS1'])
        wavelength = wcsKOALA.dropaxis(1).wcs_pix2world(index_wave, 0)[0]
        intensity = RSS_fits_file[0].data[good_spaxels]
        
        if rss_clean == False:
            print("\n  Number of spectra in this RSS =",len(RSS_fits_file[0].data),",  number of good spectra =",len(good_spaxels)," ,  number of bad spectra =", len(bad_spaxels))
            if len(bad_spaxels) > 0 : print("  Bad fibres =",bad_spaxels)
        
        # Read errors using RSS_fits_file[1]
        # self.header1 = RSS_fits_file[1].data      # CHECK WHEN DOING ERRORS !!!
        
        # Read spaxel positions on sky using RSS_fits_file[2]
        self.header2_data = RSS_fits_file[2].data
        #print RSS_fits_file[2].data

                
        # CAREFUL !! header 2 has the info of BAD fibres, if we are reading from our created RSS files we have to do it in a different way...  
                
        if len(bad_spaxels) == 0:
            offset_RA_arcsec_ = []
            offset_DEC_arcsec_ = []
            for i in range(len(good_spaxels)):
                offset_RA_arcsec_.append(self.header2_data[i][5])
                offset_DEC_arcsec_.append(self.header2_data[i][6])
            offset_RA_arcsec =np.array(offset_RA_arcsec_)
            offset_DEC_arcsec =np.array(offset_DEC_arcsec_)
            variance = np.zeros_like(intensity)   # CHECK FOR ERRORS
            
        else:
            offset_RA_arcsec = np.array([RSS_fits_file[2].data[i][5]
                                     for i in good_spaxels])
            offset_DEC_arcsec = np.array([RSS_fits_file[2].data[i][6]
                                      for i in good_spaxels])
        
            self.ID = np.array([RSS_fits_file[2].data[i][0] for i in good_spaxels])   # These are the good fibres        
            variance = RSS_fits_file[1].data[good_spaxels]   # CHECK FOR ERRORS


        self.ZDSTART = RSS_fits_file[0].header['ZDSTART']
        self.ZDEND = RSS_fits_file[0].header['ZDEND']  

        # KOALA-specific stuff
        self.PA = RSS_fits_file[0].header['TEL_PA']
        self.grating = RSS_fits_file[0].header['GRATID']
        # Check RED / BLUE arm for AAOmega
        if (RSS_fits_file[0].header['SPECTID'] == "RD"):
            AAOmega_Arm = "RED"
        if (RSS_fits_file[0].header['SPECTID'] == "BL"):   #VIRUS
            AAOmega_Arm = "BLUE"

        # For WCS
        self.CRVAL1_CDELT1_CRPIX1=[]
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
        if np.max(offset_RA_arcsec)-np.min(offset_RA_arcsec) > 31 or \
           np.max(offset_DEC_arcsec)-np.min(offset_DEC_arcsec) > 31:
            self.spaxel_size = 1.25
            field = "WIDE"
        else:
            self.spaxel_size = 0.7
            field = "NARROW"

        # Get min and max for rss
        self.RA_min, self.RA_max, self.DEC_min, self.DEC_max =coord_range([self])
        self.DEC_segment = (self.DEC_max-self.DEC_min)*3600. # +1.25 for converting to total field of view
        self.RA_segment = (self.RA_max-self.RA_min)*3600.    # +1.25 

        # --------------------------------------------------------------------
        # ------------------------------------- 1. Reading or getting the mask
        # --------------------------------------------------------------------

        # Reading the mask if needed
        if mask == "" and mask_file == "":
            #print "\n> No mask is given, obtaining it from the RSS file ..." #        
            # Only write it on history the first time, when apply_throughput = True
            self.get_mask(include_history=apply_throughput, plot = plot_mask, verbose=verbose)       
        else:
            # Include it in the history ONLY if it is the first time (i.e. applying throughput)
            self.read_mask_from_fits_file(mask=mask, mask_file=mask_file, no_nans=no_nans, plot = plot_mask, verbose= verbose, include_history=apply_throughput) 

        if valid_wave_min == 0 and valid_wave_max == 0:   ##############  DIANA FIX !!!
            self.valid_wave_min = self.mask_good_wavelength_range[0]
            self.valid_wave_max = self.mask_good_wavelength_range[1]            
            print("\n> Using the values provided by the mask for establishing the good wavelenth range:  [ {:.2f} , {:.2f} ]".format(self.valid_wave_min,self.valid_wave_max)) 
        else:
            self.valid_wave_min = valid_wave_min
            self.valid_wave_max = valid_wave_max
            print("  As specified, we use the [",self.valid_wave_min," , ",self.valid_wave_max,"] range.")           
  
        # Plot RSS_image
        if plot: self.RSS_image(image=self.intensity, cmap="binary_r")
          
        # Deep copy of intensity into intensity_corrected
        self.intensity_corrected=copy.deepcopy(self.intensity)    

        # ---------------------------------------------------
        # ------------- PROCESSING THE RSS FILE -------------
        # ---------------------------------------------------
        
        # ---------------------------------------------------
        # 0. Divide by flatfield if needed
        # Object "flat" has to have a normalized flat response in .intensity_corrected
        # Usually this is found .nresponse , see task "nresponse_flappyflat"
        # However, this correction is not needed is LFLATs have been used in 2dFdr
        # and using a skyflat to get .nresponse (small wavelength variations to throughput)
        if flat != "" :
            print("\n> Dividing the data by the flatfield provided...")
            self.intensity_corrected=self.intensity_corrected/flat.intensity_corrected
            self.history.append("- Data divided by flatfield:")
            self.history.append(flat.filename)

        # ---------------------------------------------------
        # 1. Check if apply throughput & apply it if requested    (T)
        text_for_integrated_fibre="..." 
        title_for_integrated_fibre=""   
        plot_this = False
        if apply_throughput: 
            # Check if throughput_2D[0][0] = 1., that means the throughput has been computed AFTER  fixing small wavelength variations            
            if len(throughput_2D) > 0:     
                if throughput_2D[0][0] == 1. :
                    throughput_2D_wavecor = True
                else:
                    throughput_2D_wavecor = False
            else:             
                ftf = fits.open(throughput_2D_file)
                self.throughput_2D = ftf[0].data 
                if self.throughput_2D[0][0] == 1. :
                    throughput_2D_wavecor = True
                    # throughput_2D_file has in the header the values for sol
                    sol=[0,0,0]
                    sol[0] = ftf[0].header["SOL0"]
                    sol[1] = ftf[0].header["SOL1"]
                    sol[2] = ftf[0].header["SOL2"]
                else:
                    throughput_2D_wavecor = False

            if throughput_2D_wavecor:
                print("\n> The provided throughput 2D information has been computed AFTER fixing small wavelength variations.") 
                print("  Therefore, the throughput 2D will be applied AFTER correcting for ccd defects and small wavelength variations")            
                if len(throughput_2D) == 0:
                    print("  The fits file with the throughput 2D has the solution for fixing small wavelength shifts.")
                if self.grating == "580V" : remove_5577 = True    
            else:
                self.apply_throughput_2D(throughput_2D=throughput_2D, throughput_2D_file=throughput_2D_file, plot=plot)
                text_for_integrated_fibre="after throughput correction..."  
                title_for_integrated_fibre= " - Throughput corrected"              
        else:
            if rss_clean == False and verbose == True: print("\n> Intensities NOT corrected for 2D throughput")                        

        plot_integrated_fibre_again=0   # Check if we need to plot it again        
       
        # ---------------------------------------------------
        # 2. Correcting for CCD defects                          (C)    
        if correct_ccd_defects:
            self.history.append("- Data corrected for CCD defects, kernel_correct_ccd_defects = "+np.str(kernel_correct_ccd_defects)+" for running median")
            if plot: plot_integrated_fibre_again = 1  
            
            remove_5577_here = remove_5577
            if sky_method == "1D" and scale_sky_1D == 0: remove_5577_here = False
            
            if remove_5577_here: self.history.append("  Skyline 5577 removed while cleaning CCD using Gaussian fits")
                            
            self.correct_ccd_defects(kernel_correct_ccd_defects=kernel_correct_ccd_defects, remove_5577 = remove_5577_here, 
                                     fibre_p=fibre_p, apply_throughput=apply_throughput,verbose=verbose, plot=plot)

            # Compare corrected vs uncorrected spectrum
            if plot: self.plot_corrected_vs_uncorrected_spectrum(high_fibres=high_fibres, fig_size=fig_size) 
 
            # If removing_5577_here, use the linear fit to the 5577 Gaussian fits in "fix_2dFdr_wavelengths"
            if fix_wavelengths and sol[0] == 0 : sol = self.sol

        # --------------------------------------------------- 
        # 3. Fixing small wavelength shifts                  (W)        
        if fix_wavelengths:                     
            if sol[0] == -1.0: 
                self.fix_2dfdr_wavelengths_edges(verbose=verbose, plot=plot)
            else:
                self.fix_2dfdr_wavelengths(verbose=verbose, plot=plot, sol= sol)
    
        # Apply throughput 2D corrected for small wavelength shifts if needed
        if apply_throughput and throughput_2D_wavecor:
                self.apply_throughput_2D(throughput_2D=throughput_2D, throughput_2D_file=throughput_2D_file, plot=plot)
                text_for_integrated_fibre="after throughput correction..."  
                title_for_integrated_fibre= " - Throughput corrected"   

        # Compute integrated map after throughput correction & plot if requested/needed  
        if rss_clean == False:
            if plot == True and plot_integrated_fibre_again != 1: #correct_ccd_defects == False: 
                plot_this = True
            
            self.compute_integrated_fibre(plot=plot_this, title =title_for_integrated_fibre, 
                                          text=text_for_integrated_fibre,warnings=warnings, verbose=verbose,
                                          correct_negative_sky = False)                    

        # ---------------------------------------------------
        # 4. Get airmass and correct for extinction         (X)
        # DO THIS BEFORE TELLURIC CORRECTION (that is extinction-corrected) OR SKY SUBTRACTION
        ZD = (self.ZDSTART+self.ZDEND) / 2              
        self.airmass = 1 / np.cos(np.radians(ZD))
        self.extinction_correction = np.ones(self.n_wave)
        if do_extinction: self.do_extinction_curve(plot=plot, verbose=verbose, fig_size=fig_size)

        # ---------------------------------------------------                            
        # 5. Check if telluric correction is needed & apply    (U)   
        telluric_correction_applied = False
        if do_telluric_correction:
            plot_integrated_fibre_again = plot_integrated_fibre_again + 1
            self.apply_telluric_correction(telluric_correction = telluric_correction, 
                                           telluric_correction_file = telluric_correction_file, verbose = verbose)
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
                print(" ",sky_spectrum_file)
            
            w_sky,sky_spectrum = read_table(sky_spectrum_file, ["f", "f"] )
            
            if np.nanmedian(self.wavelength-w_sky) != 0:
                if verbose or warnings: print("\n\n  WARNING !!!! The wavelengths provided on this file do not match the wavelengths on this RSS !!\n\n") 
            
            self.history.append('- 1D sky spectrum provided in file :')
            self.history.append('  '+sky_spectrum_file)
                    
        if sky_method != "none" and is_sky == False:
            plot_integrated_fibre_again = plot_integrated_fibre_again + 1 
            self.history.append('- Sky sustraction using the '+sky_method+' method')

            if sky_method in ["1Dfit","selffit"] : self.apply_mask(verbose=verbose)     

            # (5) 1Dfit
            if sky_method == "1Dfit" :
                 self.apply_1Dfit_sky(sky_spectrum = sky_spectrum, n_sky = n_sky, sky_fibres = sky_fibres, sky_spectrum_file = sky_spectrum_file,
                    sky_wave_min = sky_wave_min, sky_wave_max = sky_wave_max, win_sky = win_sky, scale_sky_1D = scale_sky_1D,
                    sky_lines_file = sky_lines_file, brightest_line_wavelength = brightest_line_wavelength,
                    brightest_line = brightest_line, maxima_sigma = maxima_sigma, auto_scale_sky = auto_scale_sky,
                    plot = plot, verbose = verbose, fig_size = fig_size, fibre_p = fibre_p, kernel_correct_ccd_defects = kernel_correct_ccd_defects)

               
            # (1) If a single sky_spectrum is provided: 
            if sky_method == "1D" : 
               
                if len(sky_spectrum) > 0:
                    self.apply_1D_sky(sky_fibres = sky_fibres, sky_wave_min = sky_wave_min, sky_wave_max = sky_wave_max,
                                 win_sky = win_sky, include_history = True, sky_spectrum = sky_spectrum,
                                 scale_sky_1D = scale_sky_1D, remove_5577 = remove_5577, #sky_spectrum_file = sky_spectrum_file,
                                 plot = plot, verbose = verbose)
                   
                else:
                    print("\n> Sustracting the sky using a sky spectrum requested but any sky spectrum provided !")
                    sky_method = "self"
                    n_sky=50    
          
            # (2) If a 2D sky, sky_rss, is provided  
            if sky_method == "2D" :    # if np.nanmedian(sky_rss.intensity_corrected) != 0:
                #
                # TODO : Needs to be checked and move to an INDEPENDENT task
                
                if scale_sky_rss != 0:   
                    if verbose: print("\n> Using sky image provided to substract sky, considering a scale of",scale_sky_rss,"...") 
                    self.sky_emission=scale_sky_rss * sky_rss.intensity_corrected
                    self.intensity_corrected = self.intensity_corrected - self.sky_emission
                else:
                    if verbose: print("\n> Using sky image provided to substract sky, computing the scale using sky lines") 
                    # check scale fibre by fibre
                    self.sky_emission=copy.deepcopy(sky_rss.intensity_corrected)
                    scale_per_fibre=np.ones((self.n_spectra))
                    scale_per_fibre_2=np.ones((self.n_spectra))
                    lowlow=15  
                    lowhigh=5
                    highlow=5
                    highhigh=15 
                    if self.grating == "580V": 
                        if verbose: print("  For 580V we use bright skyline at 5577 AA ...") 
                        sky_line = 5577                     
                        sky_line_2 = 0
                    if self.grating == "1000R": 
                        #print "  For 1000R we use skylines at 6300.5 and 6949.0 AA ..."   ### TWO LINES GIVE WORSE RESULTS THAN USING ONLY 1...
                        if verbose: print("  For 1000R we use skyline at 6949.0 AA ...")
                        sky_line = 6949.0 #6300.5
                        lowlow=22  # for getting a good continuuem in 6949.0
                        lowhigh=12
                        highlow=36
                        highhigh=52                      
                        sky_line_2 = 0 #6949.0  #7276.5 fails
                        lowlow_2=22  # for getting a good continuuem in 6949.0
                        lowhigh_2=12
                        highlow_2=36
                        highhigh_2=52
                    if sky_line_2 != 0 and verbose: print("  ... first checking",sky_line,"...")   
                    for fibre_sky in range(self.n_spectra):
                        skyline_spec = fluxes(self.wavelength, self.intensity_corrected[fibre_sky], sky_line, plot=False, verbose=False,lowlow=lowlow,lowhigh=lowhigh,highlow=highlow,highhigh=highhigh)  #fmin=-5.0E-17, fmax=2.0E-16, 
                        #resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                        self.intensity_corrected[fibre_sky] = skyline_spec[11]
    
                        skyline_sky  = fluxes(self.wavelength, self.sky_emission[fibre_sky], sky_line, plot=False, verbose=False, lowlow=lowlow,lowhigh=lowhigh,highlow=highlow,highhigh=highhigh)  #fmin=-5.0E-17, fmax=2.0E-16, 
                        
                        scale_per_fibre[fibre_sky] =   skyline_spec[3] / skyline_sky[3]
                        self.sky_emission[fibre_sky] = skyline_sky[11] 

                    if sky_line_2 != 0:
                        if verbose: print("  ... now checking",sky_line_2,"...")
                        for fibre_sky in range(self.n_spectra):
                            skyline_spec = fluxes(self.wavelength, self.intensity_corrected[fibre_sky], sky_line_2, plot=False, verbose=False, lowlow=lowlow_2,lowhigh=lowhigh_2,highlow=highlow_2,highhigh=highhigh_2)  #fmin=-5.0E-17, fmax=2.0E-16, 
                            #resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                            self.intensity_corrected[fibre_sky] = skyline_spec[11]
        
                            skyline_sky  = fluxes(self.wavelength, self.sky_emission[fibre_sky], sky_line_2, plot=False, verbose=False, lowlow=lowlow_2,lowhigh=lowhigh_2,highlow=highlow_2,highhigh=highhigh_2)  #fmin=-5.0E-17, fmax=2.0E-16, 
                            
                            scale_per_fibre_2[fibre_sky] =   skyline_spec[3] / skyline_sky[3]
                            self.sky_emission[fibre_sky] = skyline_sky[11] 

    
                    # Median value of scale_per_fibre, and apply that value to all fibres
                    if sky_line_2 == 0:
                        scale_sky_rss=np.nanmedian(scale_per_fibre)
                        self.sky_emission= self.sky_emission * scale_sky_rss
                    else:
                        scale_sky_rss=np.nanmedian((scale_per_fibre+scale_per_fibre_2)/2)
                        # Make linear fit
                        scale_sky_rss_1 = np.nanmedian(scale_per_fibre)
                        scale_sky_rss_2 = np.nanmedian(scale_per_fibre_2)
                        if verbose:
                            print("  Median scale for line 1 :", scale_sky_rss_1,"range [",np.nanmin(scale_per_fibre),",",np.nanmax(scale_per_fibre),"]")
                            print("  Median scale for line 2 :",scale_sky_rss_2,"range [",np.nanmin(scale_per_fibre_2),",",np.nanmax(scale_per_fibre_2),"]")
                                              
                        b = (scale_sky_rss_1 - scale_sky_rss_2) / (sky_line - sky_line_2)
                        a = scale_sky_rss_1 - b * sky_line
                        if verbose: print("  Appling linear fit with a =",a,"b =",b,"to all fibres in sky image...") #,a+b*sky_line,a+b*sky_line_2
                        
                        for i in range(self.n_wave):
                            self.sky_emission[:,i]=self.sky_emission[:,i] * (a+b*self.wavelength[i])
                    
                    if plot:
                        plt.figure(figsize=(fig_size, fig_size/2.5))
                        label1="$\lambda$"+np.str(sky_line)
                        plt.plot(scale_per_fibre, alpha=0.5, label=label1)                        
                        plt.minorticks_on()
                        plt.ylim(np.nanmin(scale_per_fibre),np.nanmax(scale_per_fibre))
                        plt.axhline(y=scale_sky_rss, color='k', linestyle='--') 
                        if sky_line_2 == 0: 
                            text="Scale OBJECT / SKY using sky line $\lambda$"+np.str(sky_line)
                            if verbose:
                                print("  Scale per fibre in the range [",np.nanmin(scale_per_fibre),",",np.nanmax(scale_per_fibre),"], median value is",scale_sky_rss)
                                print("  Using median value to scale sky emission provided...")
                        if sky_line_2 != 0: 
                            text="Scale OBJECT / SKY using sky lines $\lambda$"+np.str(sky_line)+" and $\lambda$"+np.str(sky_line_2)
                            label2="$\lambda$"+np.str(sky_line_2)
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
                
                if verbose: print("\n> 'sky_method = selffit', hence using",n_sky,"lowest intensity fibres to create a sky spectrum ...")

                self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                               sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, 
                               win_sky=win_sky, include_history=True) 
                
                if sky_spectrum[0] != -1 and np.nanmedian(sky_spectrum) != 0:
                    if verbose: print("\n> Additional sky spectrum provided. Using this for replacing regions with bright emission lines...")

                    sky_r_self = self.sky_emission
                
                    self.sky_emission = replace_el_in_sky_spectrum(self, sky_r_self, sky_spectrum, scale_sky_1D=scale_sky_1D , 
                                                                   brightest_line=brightest_line, 
                                                                   brightest_line_wavelength = brightest_line_wavelength,
                                                                   ranges_with_emission_lines = ranges_with_emission_lines,
                                                                   cut_red_end=cut_red_end,
                                                                   plot=plot)
                    self.history.append('  Using sky spectrum provided for replacing regions with emission lines')

                self.fit_and_substract_sky_spectrum(self.sky_emission, sky_lines_file = sky_lines_file,
                                       brightest_line_wavelength = brightest_line_wavelength, brightest_line = brightest_line,
                                       maxima_sigma=maxima_sigma, ymin =-50, ymax=600, wmin = 0, wmax =0, auto_scale_sky = auto_scale_sky,                                       
                                       warnings = False, verbose=False, plot=False, fig_size=fig_size, fibre=fibre_p )           
            
                if fibre_p == -1:
                    if verbose: print("\n> 'selffit' sky_method usually generates some nans, correcting ccd defects again...")
                    self.correct_ccd_defects(kernel_correct_ccd_defects=kernel_correct_ccd_defects, verbose=verbose, plot=plot, only_nans=True)    # not replacing values < 0

               
            # (3) "self": Obtain the sky using the n_sky lowest fibres 
            #             If a 1D spectrum is provided, use it for replacing regions with bright emission lines   #DIANA
            if sky_method == "self":   
                
                self.sky_fibres =  sky_fibres               
                if n_sky == 0 : n_sky=len(sky_fibres) 
                self.apply_self_sky(sky_fibres = self.sky_fibres, sky_spectrum = sky_spectrum, n_sky=n_sky,
                          sky_wave_min = sky_wave_min, sky_wave_max = sky_wave_max, win_sky = win_sky, scale_sky_1D = scale_sky_1D,
                          brightest_line = "Ha", brightest_line_wavelength = 0, ranges_with_emission_lines = [0],
                          cut_red_end = cut_red_end, low_fibres = low_fibres, use_fit_for_negative_sky = use_fit_for_negative_sky, 
                          kernel_negative_sky = kernel_negative_sky, order_fit_negative_sky = order_fit_negative_sky, 
                          plot = True, verbose = verbose)
                


        # Correct negative sky if requested 
        if is_sky == False and correct_negative_sky == True:
            text_for_integrated_fibre="after correcting negative sky"
            self.correcting_negative_sky(plot=plot, low_fibres=low_fibres, kernel_negative_sky = kernel_negative_sky, order_fit_negative_sky=order_fit_negative_sky,
                                         individual_check = individual_check, use_fit_for_negative_sky = use_fit_for_negative_sky, force_sky_fibres_to_zero = force_sky_fibres_to_zero) # exclude_wlm=exclude_wlm
 
        # Check Median spectrum of the sky fibres AFTER subtracting the sky emission
        if plot == True and len(self.sky_fibres) > 0 :
            sky_emission = sky_spectrum_from_fibres(self, self.sky_fibres, win_sky=0, plot=False, include_history=False, verbose = False)
            plot_plot(self.wavelength,sky_emission,hlines=[0],ptitle = "Median spectrum of the sky fibres AFTER subtracting the sky emission")
            #plot_plot(self.wavelength,self.sky_emission,hlines=[0],ptitle = "Median spectrum using self.sky_emission")


        # If this RSS is an offset sky, perform a median filter to increase S/N 
        if is_sky:
            self.is_sky(n_sky = n_sky, win_sky = win_sky, sky_fibres = sky_fibres, sky_wave_min = sky_wave_min, 
                        sky_wave_max = sky_wave_max, plot = plot, verbose = verbose)
            if win_sky  == 0 : win_sky = 151 # Default value in is_sky


        # ---------------------------------------------------
        # 7. Check if identify emission lines is requested & do      (E)
        # TODO: NEEDS TO BE CHECKED !!!!
        if id_el:
            if brightest_line_wavelength == 0 :           
                self.el=self.identify_el(high_fibres=high_fibres, brightest_line = brightest_line,
                                         cut=cut, verbose=True, plot=plot_id_el, fibre=0, broad=broad)
                print("\n  Emission lines identified saved in self.el !!")
            else:
                brightest_line_rest_wave = 6562.82
                print("\n  As given, line ",brightest_line," at rest wavelength = ",brightest_line_rest_wave," is at ", brightest_line_wavelength)
                self.el=[[brightest_line],[brightest_line_rest_wave],[brightest_line_wavelength],[7.2]]
                #  sel.el=[peaks_name,peaks_rest, p_peaks_l, p_peaks_fwhm]      
        else:
            self.el=[[0],[0],[0],[0]]

        # Check if id_list provided
        if id_list[0] != 0:
            if id_el: 
                print("\n> Checking if identified emission lines agree with list provided")
                # Read list with all emission lines to get the name of emission lines
                emission_line_file="lineas_c89_python.dat"
                el_center,el_name = read_table(emission_line_file, ["f", "s"] )
                
                # Find brightest line to get redshift
                for i in range(len(self.el[0])):
                    if self.el[0][i] == brightest_line:
                        obs_wave= self.el[2][i]
                        redshift = (self.el[2][i]-self.el[1][i]) / self.el[1][i]                        
                print("  Brightest emission line",  brightest_line, "found at ",  obs_wave,", redshift = ",redshift)     
             
                el_identified = [[], [], [], []]                               
                n_identified=0
                for line in id_list:                     
                    id_check=0
                    for i in range(len(self.el[1])):
                        if line == self.el[1][i] :
                            if verbose: print("  Emission line ", self.el[0][i],self.el[1][i],"has been identified")
                            n_identified = n_identified + 1
                            id_check = 1
                            el_identified[0].append(self.el[0][i])  # Name
                            el_identified[1].append(self.el[1][i])  # Central wavelength
                            el_identified[2].append(self.el[2][i])  # Observed wavelength
                            el_identified[3].append(self.el[3][i])  # "FWHM"
                    if id_check == 0:                    
                        for i in range(len(el_center)):
                            if line == el_center[i] : 
                                el_identified[0].append(el_name[i])
                                print("  Emission line",el_name[i],line,"has NOT been identified, adding...")
                        el_identified[1].append(line)
                        el_identified[2].append(line *(redshift+1))
                        el_identified[3].append(4*broad)
                        
                self.el=el_identified                
                print("  Number of emission lines identified = ",n_identified,"of a total of",len(id_list),"provided. self.el updated accordingly")
            else:
                if rss_clean == False: print("\n> List of emission lines provided but no identification was requested")
                
        # ---------------------------------------------------
        # 8.1. Clean sky residuals if requested           (R)      
        if clean_sky_residuals:
            #plot_integrated_fibre_again = plot_integrated_fibre_again + 1  
            #self.clean_sky_residuals(extra_w=extra_w, step=step_csr, dclip=dclip, verbose=verbose, fibre=fibre, wave_min=valid_wave_min,  wave_max=valid_wave_max) 

            if len(features_to_fix) == 0 :    # Add features that are known to be problematic  

                if self.wavelength[0] < 6250 and  self.wavelength[-1] >  6350:
                    features_to_fix.append(["r", 6250, 6292, 6308, 6350, 2, 98, 2, False,False]) # 6301
                if self.wavelength[0] < 7550 and  self.wavelength[-1] >  7660:
                    features_to_fix.append(["r", 7558, 7595, 7615, 7652, 2, 98, 2, False,False]) # Big telluric absorption
                #if self.wavelength[0] < 8550 and  self.wavelength[-1] >  8770:
                #    features_to_fix.append(["s", 8560, 8610, 8685, 8767, 2, 98, 2, False,False])

            elif features_to_fix == "big_telluric" or features_to_fix =="big_telluric_absorption":
                features_to_fix=[["r", 7558, 7595, 7615, 7652, 2, 98, 2, False,False]] 
                
                
            if len(features_to_fix) > 0:
                
                if verbose:
                    print("\n> Features to fix: ")
                    for feature in features_to_fix:
                        print("  -",feature)
                
                if len(sky_fibres_for_residuals) == 0:                 
                    self.find_sky_fibres(sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, n_sky=np.int(n_sky/2), plot=plot, warnings=False)                    
                    sky_fibres_for_residuals = self.sky_fibres
                fix_these_features_in_all_spectra(self,features=features_to_fix,
                                                  fibre_list=fibres_to_fix, #range(83,test.n_spectra),
                                                  sky_fibres = sky_fibres_for_residuals,  
                                                  replace=True, plot=plot) 
                #check functions for documentation
        # ---------------------------------------------------
        # 8.2. Clean edges if requested           (R)  
        if fix_edges:
            self.fix_edges(verbose=verbose)
        # ---------------------------------------------------
        # 8.3. Remove negative median values      (R)
        if remove_negative_median_values:  #  it was remove_negative_pixels_in_sky: 
            self.intensity_corrected = remove_negative_pixels(self.intensity_corrected, verbose=verbose)
            self.history.append("- Spectra with negative median values corrected to median = 0")
        # ---------------------------------------------------
        # 8.4. Clean extreme negatives      (R)        
        if clean_extreme_negatives: 
            self.clean_extreme_negatives(fibre_list=fibres_to_fix, percentile_min = percentile_min, plot=plot, verbose=verbose)
        # ---------------------------------------------------
        # 8.5. Clean cosmics    (R)
        if clean_cosmics:
            self.kill_cosmics(brightest_line_wavelength, width_bl = width_bl, kernel_median_cosmics = kernel_median_cosmics, 
                              cosmic_higher_than = cosmic_higher_than, extra_factor = extra_factor, max_number_of_cosmics_per_fibre=max_number_of_cosmics_per_fibre,
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

        #if fibre_p != 0: plot_integrated_fibre_again = 0
        
        if plot_integrated_fibre_again >0 :
            # Plot corrected values
            if rss_clean:
                text="..."
            else:
                text="after all corrections have been applied..."
            self.compute_integrated_fibre(plot=plot,title =" - Intensities Corrected", warnings=warnings, text=text, verbose=verbose,
                                          valid_wave_min=valid_wave_min, valid_wave_max=valid_wave_max, last_check=True,
                                          low_fibres=low_fibres, correct_negative_sky = False, 
                                          individual_check=False,  order_fit_negative_sky=order_fit_negative_sky, kernel_negative_sky=kernel_negative_sky,use_fit_for_negative_sky=use_fit_for_negative_sky)

        # Plot correct vs uncorrected spectra
        if plot == True:    
            self.plot_corrected_vs_uncorrected_spectrum(high_fibres=high_fibres, fig_size=fig_size) 
            self.plot_corrected_vs_uncorrected_spectrum(low_fibres=low_fibres, fig_size=fig_size)

        # Plot RSS_image
        if plot or plot_final_rss: self.RSS_image()

        # If this is a CLEAN RSS, be sure self.integrated_fibre is obtained
        if rss_clean: self.compute_integrated_fibre(plot=False, warnings=False, verbose=False)

            
        # Print summary and information from header
        print("\n> Summary of reading rss file", '"'+filename+'"', ":\n")
        print("  This is a KOALA {} file,".format(AAOmega_Arm), \
              "using the {} grating in AAOmega, ".format(self.grating), \
              "exposition time = {} s.".format(self.exptime))
        print("  Object:", self.object)
        print("  Field of view:", field, \
              "(spaxel size =", self.spaxel_size, "arcsec)")
        print("  Center position: (RA, DEC) = ({:.3f}, {:.3f}) degrees" \
              .format(self.RA_centre_deg, self.DEC_centre_deg))
        print("  Field covered [arcsec] = {:.1f} x {:.1f}".format(self.RA_segment+self.spaxel_size, self.DEC_segment+self.spaxel_size )) 
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
                    #print " ",throughput_2D_file
            else:
                print("  Intensities NOT corrected for throughput 2D")
            if correct_ccd_defects:
                print("  Intensities corrected for CCD defects !") 
            else:    
                print("  Intensities NOT corrected for CCD defects")                 
  
            if sol[0] != 0 and fix_wavelengths :
                print("  All fibres corrected for small wavelength shifts using wavelength solution provided!")
            else:    
                if fix_wavelengths:    
                    print("  Wavelengths corrected for small shifts using Gaussian fit to selected bright skylines in all fibres!")        
                else:
                    print("  Wavelengths NOT corrected for small shifts")

            if do_extinction:
                print("  Intensities corrected for extinction !") 
            else:
                print("  Intensities NOT corrected for extinction")    

            if telluric_correction_applied : 
                print("  Intensities corrected for telluric absorptions !") 
            else:
                if self.grating in red_gratings : print("  Intensities NOT corrected for telluric absorptions")
                    
            if is_sky:
                print("  This is a SKY IMAGE, median filter with window",win_sky,"applied !")
                print("  The median 1D sky spectrum combining",n_sky,"lowest fibres is stored in self.sky_emission")
            else:
                if sky_method == "none" : print("  Intensities NOT corrected for sky emission")
                if sky_method == "self" : print("  Intensities corrected for sky emission using",n_sky,"spaxels with lowest values !")
                if sky_method == "selffit" : print("  Intensities corrected for sky emission using",n_sky,"spaxels with lowest values !")
                if sky_method == "1D" : print("  Intensities corrected for sky emission using (scaled) spectrum provided ! ")
                if sky_method == "1Dfit" : print("  Intensities corrected for sky emission fitting Gaussians to both 1D sky spectrum and each fibre ! ")
                if sky_method == "2D" : print("  Intensities corrected for sky emission using sky image provided scaled by",scale_sky_rss,"!")
                
            if correct_negative_sky: print("  Intensities corrected to make the integrated value of the lowest fibres = 0 !")  
                           
            if id_el:
                print(" ", len(self.el[0]), "emission lines identified and stored in self.el !") 
                print(" ", self.el[0])

            if clean_sky_residuals : 
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
                    if clean_cosmics == True or clean_extreme_negatives == True or remove_negative_median_values ==True or fix_edges == True or clean_sky_residuals == True: clean_residuals = True
                    save_rss_to_fits_file = name_keys(filename, apply_throughput=apply_throughput, correct_ccd_defects = correct_ccd_defects,
                                                      fix_wavelengths = fix_wavelengths, do_extinction = do_extinction, sky_method = sky_method,
                                                      do_telluric_correction = telluric_correction_applied, id_el = id_el,
                                                      correct_negative_sky = correct_negative_sky, clean_residuals = clean_residuals)
                
                save_rss_fits(self, fits_file=save_rss_to_fits_file)    

        if rss_clean == False: print("\n> KOALA RSS file read !")
        
        
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
    def fix_2dfdr_wavelengths_edges(self, #sky_lines =[6300.309, 7316.290, 8430.147, 8465.374],
                                    sky_lines =[6300.309, 8430.147, 8465.374],
                                    #valid_ranges=[[-0.25,0.25],[-0.5,0.5],[-0.5,0.5]],
                                    #valid_ranges=[[-0.4,0.3],[-0.4,0.45],[-0.5,0.5],[-0.5,0.5]], # ORIGINAL
                                    valid_ranges=[[-1.2,0.6],[-1.2,0.6],[-1.2,0.6]],
                                    fit_order = 2, apply_median_filter = True, kernel_median=51,
                                    fibres_to_plot=[0,100,300,500,700,850,985],
                                    show_fibres=[0,500,985] ,
                                    plot_fits = False,
                                    xmin=8450,xmax=8475, ymin=-10, ymax=250, 
                                    check_throughput = False,
                                    plot=True,verbose=True, warnings=True, fig_size=12):
        """
        Using bright skylines, performs small wavelength corrections to each fibre
        
        Parameters:
        ----------
        sky_lines : list of floats (default = [6300.309, 8430.147, 8465.374])
            Chooses the sky lines to run calibration for
        valid_ranges : list of lists of floats (default = [[-1.2,0.6],[-1.2,0.6],[-1.2,0.6]])
            Ranges of flux offsets for each sky line
        fit_order : integer (default = 2)
            Order of polynomial for fitting
        apply_median_filter : boolean (default = True)
            Choose if we want to apply median filter to the image
        kernel_median : odd integer (default = 51)
            Length of the median filter interval
        fibres_to_plot : list of integers (default = [0,100,300,500,700,850,985])
            Choose specific fibres to visualise fitted offsets per wavelength using plots
        show_fibres : list of integers (default = [0,500,985])
            Plot the comparission between uncorrected and corrected flux per wavelength for specific fibres
        plot_fits : boolean (default = False)
            Plot the Gaussian fits for each iteration of fitting
        xmin, xmax, ymin, ymax : integers (default = 8450, 8475, -10, 250)
            Plot ranges for Gaussian fit plots, x = wavelength in Angstroms, y is flux in counts
        plot : boolean (default = True)
            Plot the resulting KOALA RSS image
        verbose : boolean (default = True)
            Print detailed description of steps being done on the image in the console as code runs
        warnings : boolean (default = True)
            Print the warnings in the console if something works incorrectly or might require attention 
        fig_size : integer (default = 12)
            Size of the image plotted          
        """
    
        print("\n> Fixing 2dfdr wavelengths using skylines in edges")       
        print("\n  Using skylines: ",sky_lines,"\n")

        # Find offsets using 6300.309 in the blue end and average of 8430.147, 8465.374 in red end
        w = self.wavelength
        nspec = self.n_spectra 
        
        #fibres_to_plot = [544,545,546,547,555]
        #plot_fits = True
                     
        self.sol_edges =[]
        
        offset_sky_lines=[]
        fitted_offset_sky_lines =[]
        gauss_fluxes_sky_lines=[]
        for sky_line in sky_lines:    
            gauss_fluxes =[]
            x=[]
            offset_ = []
            for i in range(nspec):
                x.append(i*1.) 
                f = self.intensity_corrected[i]                
                if i in fibres_to_plot and plot_fits:
                    plot_fit = True
                else:
                    plot_fit = False
                if i == 0 : plot_fit = True
                if plot_fit: print(" - Plotting Gaussian fitting for skyline",sky_line,"in fibre",i,":")
                resultado = fluxes(w,f,sky_line, lowlow=80,lowhigh=20,highlow=20,highhigh=80,broad=2.0,
                                   fcal=False,plot=plot_fit,verbose=False)
                offset_.append(resultado[1])
                gauss_fluxes.append(resultado[3])
            offset = np.array(offset_) - sky_line   #offset_[500]
            offset_sky_lines.append(offset)
    
            offset_in_range =[]
            x_in_range = []
            valid_range = valid_ranges[sky_lines.index(sky_line)]
            offset_m = signal.medfilt(offset, kernel_median)
            text=""
            if apply_median_filter:       
                #xm = signal.medfilt(x, odd_number)
                text=" applying a "+np.str(kernel_median)+" median filter"
                for i in range(len(offset_m)):       
                    if offset_m[i] > valid_range[0] and offset_m[i] < valid_range[1]: 
                        offset_in_range.append(offset_m[i])
                        x_in_range.append(x[i])
            else:
                for i in range(len(offset)):       
                    if offset[i] > valid_range[0] and offset[i] < valid_range[1]: 
                        offset_in_range.append(offset[i])
                        x_in_range.append(i)
            
            fit=np.polyfit(x_in_range, offset_in_range,fit_order)
            if fit_order == 2: 
                ptitle="Fitting to skyline "+np.str(sky_line)+" : {:.3e} x$^2$  +  {:.3e} x  +  {:.3e} ".format(fit[0],fit[1],fit[2])+text
            if fit_order == 1:
                ptitle="Fitting to skyline "+np.str(sky_line)+" : {:.3e} x  +  {:.3e} ".format(fit[0],fit[1])+text
            if fit_order > 2:
                ptitle="Fitting an order "+np.str(fit_order)+" polinomium to skyline "+np.str(sky_line)+text
            
            y=np.poly1d(fit)
            fity = y(list(range(nspec)))
            fitted_offset_sky_lines.append(fity)
            self.sol_edges.append(fit)   # GAFAS
        
            if plot: 
                plot_plot( x, [offset,offset_m,fity], ymin=valid_range[0],ymax=valid_range[1], 
                          xlabel="Fibre", ylabel="$\Delta$ Offset",ptitle=ptitle )
    
            gauss_fluxes_sky_lines.append(gauss_fluxes)
        sky_lines_edges=[sky_lines[0],(sky_lines[-1]+sky_lines[-2])/2]
        
        nspec_vector= list(range(nspec))
        fitted_offset_sl_median = np.nanmedian(fitted_offset_sky_lines, axis = 0)
        
        fitted_solutions = np.nanmedian(self.sol_edges, axis = 0)
        y=np.poly1d(fitted_solutions)
        fitsol = y(list(range(nspec)))
        self.sol = [fitted_solutions[2],fitted_solutions[1],fitted_solutions[0]]
        print("\n> sol = ["+np.str(fitted_solutions[2])+","+np.str(fitted_solutions[1])+","+np.str(fitted_solutions[0])+"]")
        
        plot_plot(nspec_vector, [fitted_offset_sky_lines[0],fitted_offset_sky_lines[1],fitted_offset_sky_lines[2], 
                fitted_offset_sl_median, fitsol], color=["r","orange","b","k", "g"], alpha=[0.3,0.3,0.3,0.5,0.8],
                hlines=[-0.75,-0.5,-0.25,0,0.25,0.5],
                label =[np.str(sky_lines[0]), np.str(sky_lines[1]), np.str(sky_lines[2]), "median", "median sol"],
                ptitle="Checking fitting solutions",
                ymin=-1, ymax = 0.6, xlabel="Fibre",ylabel="Fitted offset")  
        
        # Plot corrections
        if plot:
            plt.figure(figsize=(fig_size, fig_size/2.5))  
            for show_fibre in fibres_to_plot:
                offsets_fibre=[fitted_offset_sky_lines[0][show_fibre],(fitted_offset_sky_lines[1][show_fibre]+fitted_offset_sky_lines[2][show_fibre])/2]
                plt.plot(sky_lines_edges,offsets_fibre,"+")
                plt.plot(sky_lines_edges,offsets_fibre,"--",label=np.str(show_fibre))
            plt.minorticks_on()
            plt.legend(frameon=False, ncol=9)
            plt.title("Small wavelength offsets per fibre")
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            plt.ylabel("Fitted offset")
            plt.show()
            plt.close()
        
    
       # Apply corrections to all fibres  
        #show_fibres=[0,500,985]  # plot only the spectrum of these fibres
        intensity=copy.deepcopy(self.intensity_corrected)
        #intensity_wave_fixed = np.zeros_like(intensity)
        
 
               
        for fibre in range(nspec): #show_fibres:
            offsets_fibre=[fitted_offset_sky_lines[0][fibre],(fitted_offset_sky_lines[-1][fibre]+fitted_offset_sky_lines[-2][fibre])/2]    
            fit_edges_offset=np.polyfit(sky_lines_edges, offsets_fibre, 1)
            y=np.poly1d(fit_edges_offset)
            w_offset = y(w)
            w_fixed = w - w_offset
    
            #Apply correction to fibre
            #intensity_wave_fixed[fibre] =rebin_spec(w_fixed, intensity[fibre], w)
            self.intensity_corrected[fibre]=rebin_spec(w_fixed, intensity[fibre], w) #=copy.deepcopy(intensity_wave_fixed) 
                    
            if fibre in show_fibres: 
                plt.figure(figsize=(fig_size, fig_size/4.5))
                plt.plot(w,intensity[fibre], "r-",alpha=0.2, label="No corrected")
                plt.plot(w_fixed,intensity[fibre], "b-",alpha=0.2, label="No corrected - Shifted")               
                plt.plot(w,self.intensity_corrected[fibre], "g-",label="Corrected after rebinning",alpha=0.6,linewidth=2.)
                for line in sky_lines:
                    plt.axvline(x=line, color="k", linestyle="--", alpha=0.3)
                #plt.xlim(6280,6320)                    
                plt.xlim(xmin,xmax)                    
                plt.ylim(ymin,ymax)    
                plt.minorticks_on() 
                ptitle="Fibre "+np.str(fibre)
                plt.title(ptitle)
                plt.legend(frameon=False, ncol=3)
                plt.xlabel("Wavelength [$\mathrm{\AA}$]")
                plt.ylabel("Flux")
                plt.show()
                plt.close()
  
        print("\n> Small fixing of the 2dFdr wavelengths considering only the edges done!")
        self.history.append("- Fixing wavelengths using skylines in the edges")
        self.history.append("  sol (found) = "+np.str(self.sol))

        if check_throughput:
            print("\n> As an extra, checking the Gaussian flux of the fitted skylines in all fibres:")
    
            vector_x = np.arange(nspec)
            vector_y =[]
            label_skylines=[]
            alpha=[]
            for i in range(len(sky_lines)):
                med_gaussian_flux = np.nanmedian(gauss_fluxes_sky_lines[i])
                vector_y.append(gauss_fluxes_sky_lines[i]/med_gaussian_flux)
                label_skylines.append(np.str(sky_lines[i]))
                alpha.append(0.3)
                #print "  - For line ",sky_lines[i],"the median flux is",med_gaussian_flux
                
            vector_y.append(np.nanmedian(vector_y, axis=0))
            label_skylines.append("Median")
            alpha.append(0.5)
             
           
            for i in range(len(sky_lines)):
                ptitle="Checking Gaussian flux of skyline "+label_skylines[i]
                plot_plot(vector_x,vector_y[i],
                          label=label_skylines[i],
                          hlines=[0.8,0.9,1.0,1.1,1.2],ylabel="Flux / Median flux", xlabel="Fibre",
                          ymin=0.7,ymax=1.3, ptitle=ptitle)
                
            ptitle="Checking Gaussian flux of the fitted skylines (this should be all 1.0 in skies)"
    #        plot_plot(vector_x,vector_y,label=label_skylines,hlines=[0.9,1.0,1.1],ylabel="Flux / Median flux", xlabel="Fibre",
    #                  ymin=0.7,ymax=1.3, alpha=alpha,ptitle=ptitle)
            plot_plot(vector_x,vector_y[:-1],label=label_skylines[:-1], alpha=alpha[:-1],
                      hlines=[0.8,0.9,1.0,1.1,1.2],ylabel="Flux / Median flux", xlabel="Fibre",
                      ymin=0.7,ymax=1.3,ptitle=ptitle)
    
            
            vlines=[]
            for j in vector_x:
                if vector_y[-1][j] > 1.1 or vector_y[-1][j] < 0.9:
                    #print "  Fibre ",j,"  ratio value = ", vector_y[-1][j]
                    vlines.append(j)
            print("\n  TOTAL = ",len(vlines)," fibres with flux differences > 10 % !!")
    
            plot_plot(vector_x,vector_y[-1],label=label_skylines[-1], alpha=1, vlines=vlines,
                      hlines=[0.8,0.9,1.0,1.1,1.2],ylabel="Flux / Median flux", xlabel="Fibre",
                      ymin=0.7,ymax=1.3,ptitle=ptitle)
            
            
            # CHECKING SOMETHING...
            self.throughput_extra_checking_skylines = vector_y[-1]
    #        for i in range(self.n_spectra):
    #            if i != 546 or i != 547:
    #                self.intensity_corrected[i] = self.intensity_corrected[i] / vector_y[-1][i]
        
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    # Idea: take a RSS dominated by skylines. Read it (only throughput correction). For each fibre, fit Gaussians to ~10 skylines. 
    # Compare with REST wavelengths. Get a median value per fibre. Perform a second-order fit to all median values.
    # Correct for that using a reference fibre (1). Save results to be applied to the rest of files of the night (assuming same configuration).
    
    def fix_2dfdr_wavelengths(self, sol=[0,0,0], fibre = -1,  edges=False,        
                              maxima_sigma=2.5, maxima_offset=1.5, 
                              sky_lines_file="",
                              #xmin=7740,xmax=7770, ymin="", ymax="", 
                              xmin = [6270, 8315], xmax = [6330, 8375], ymax ="",
                              fibres_to_plot=[0,100,400,600,950],
                              plot=True, plot_all=False,verbose=True, warnings=True):
        """
        Using bright skylines, performs small wavelength corrections to each fibre
                
        Parameters:
        ----------
        sol : list of floats (default = [0,0,0])
            Specify the parameters of the second degree polynomial fit
        fibre : integer (default = -1)
            Choose a specific fibre to run correction for. If not specified, all fibres will be corrected
        maxima_sigma : float (default = 2.5)
            Maximum allowed standard deviation for Gaussian fit
        maxima_offset : float (default 1.5)
            Maximum allowed wavelength offset in Angstroms      
        xmin : list of integer (default = [6270, 8315])
            Minimum wavelength values in Angstrom for plots before and after correction
        xmax : list of integer (default = [6330, 8375])
            Maximum wavelength values in Angstrom for plots before and after correction                
        ymax : float (default = none)
            Maximum y value to be plot, if not given, will estimate it automatically
        fibres_to_plot : list of integers (default = [0,100,400,600,950])
            Plot the comparission between uncorrected and corrected flux per wavelength for specific 
        plot : boolean (default = True)
            Plot the plots
        plot_all : boolean (default = False)
            Plot the Gaussian fits for each iteration of fitting
        verbose : boolean (default = True)
            Print detailed description of steps being done on the image in the console as code runs
        warnings : boolean (default = True)
            Print the warnings in the console if something works incorrectly or might require attention        
        """
        if verbose: print("\n> Fixing 2dfdr wavelengths using skylines...")
        if self.grating == "580V":
            xmin=[5555]
            xmax=[5600]
            if sol[0] != [0] and sol[2] == 0:
                print("  Only using the fit to the 5577 emission line...") 
                self.history.append("- Fixing wavelengths using fits to skyline 5577")
        else:
            self.history.append("- Fixing wavelengths using fits to bright skylines")
        
        w = self.wavelength
        xfibre = list(range(0,self.n_spectra))
        plot_this_again = True

        if sol[0] == 0:     # Solutions are not given   
            # Read file with sky emission line
            
            if len(sky_lines_file) == 0: sky_lines_file="./input_data/sky_lines/sky_lines_rest.dat"
            sl_center_,sl_name_,sl_fnl_,sl_lowlow_,sl_lowhigh_,sl_highlow_,sl_highhigh_,sl_lmin_,sl_lmax_ = read_table(sky_lines_file, ["f", "s", "f", "f", "f", "f", "f", "f", "f"] )
            
            # Be sure the lines we are using are in the requested wavelength range        
            #if fibre != -1:
            if verbose: print("  Checking the values of skylines in the file", sky_lines_file)
            for i in range(len(sl_center_)):
                if verbose: print("  - {:.3f}  {:.0f}  {:5.1f} {:5.1f} {:5.1f} {:5.1f}    {:6.1f} {:6.1f}".format(sl_center_[i],sl_fnl_[i],sl_lowlow_[i],sl_lowhigh_[i],sl_highlow_[i],sl_highhigh_[i],sl_lmin_[i],sl_lmax_[i]))         
            if verbose: print("\n  We only need skylines in the {:.2f} - {:.2f} range".format(np.round(self.valid_wave_min,2),np.round(self.valid_wave_max,2)))
                       
            valid_skylines = np.where((sl_center_ < self.valid_wave_max) & (sl_center_ > self.valid_wave_min))
            sl_center=sl_center_[valid_skylines]
            sl_fnl = sl_fnl_[valid_skylines]
            sl_lowlow = sl_lowlow_[valid_skylines]
            sl_lowhigh = sl_lowhigh_[valid_skylines]
            sl_highlow = sl_highlow_[valid_skylines]
            sl_highhigh = sl_highhigh_[valid_skylines]
            sl_lmin = sl_lmin_[valid_skylines]
            sl_lmax = sl_lmax_[valid_skylines]            
            number_sl = len(sl_center)
            if fibre != -1: print(" ",sl_center)
    
            # Fitting Gaussians to skylines...         
            self.wavelength_offset_per_fibre=[]
            wave_median_offset=[]
            if verbose: print("\n> Performing a Gaussian fit to selected, bright skylines...") 
            if verbose: print("  (this might FAIL if RSS is NOT corrected for CCD defects...)")
                      
            if fibre != -1:
                f_i=fibre
                f_f=fibre+1
                if verbose: print("  Checking fibre ", fibre," (only this fibre is corrected, use fibre = -1 for all)...") 
                verbose_ = True
                warnings = True
                plot_all = True
            else:
                f_i = 0 
                f_f = self.n_spectra
                verbose_ = False
                
            number_fibres_to_check = len(list(range(f_i,f_f)))    
            output_every_few = np.sqrt(len(list(range(f_i,f_f))))+1    
            next_output = -1   
            for fibre in range(f_i,f_f): #    (self.n_spectra): 
                spectrum = self.intensity_corrected[fibre]
                if verbose: 
                    if fibre > next_output:
                        sys.stdout.write("\b"*51)
                        sys.stdout.write("  Checking fibre {:4} ...  ({:6.2f} % completed) ...".format(fibre, fibre*100./number_fibres_to_check))
                        sys.stdout.flush()
                        next_output = fibre + output_every_few   
                           
                # Gaussian fits to the sky spectrum
                sl_gaussian_flux=[]
                sl_gaussian_sigma=[]    
                sl_gauss_center=[]
                sl_offset=[]
                sl_offset_good=[]
                
                for i in range(number_sl):
                    if sl_fnl[i] == 0 :
                        plot_fit = False
                    else: plot_fit = True                   
                    if plot_all: plot_fit =True
                    
                    resultado=fluxes(w, spectrum, sl_center[i], lowlow= sl_lowlow[i], lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh = sl_highhigh[i], lmin=sl_lmin[i], lmax=sl_lmax[i], fmin=0, fmax=0, 
                                 broad=2.1*2.355 , plot=plot_fit, verbose=False, plot_sus = False, fcal = False, warnings = warnings )   # Broad is FWHM for Gaussian sigm a= 1, 
    
                    sl_gaussian_flux.append(resultado[3])
                    sl_gauss_center.append(resultado[1])
                    sl_gaussian_sigma.append(resultado[5]/2.355)
                    sl_offset.append(sl_gauss_center[i]-sl_center[i])
                    
                    if sl_gaussian_flux[i] < 0 or np.abs(sl_center[i]-sl_gauss_center[i]) > maxima_offset or sl_gaussian_sigma[i] > maxima_sigma:
                        if verbose_: print("  Bad fitting for ", sl_center[i], "... ignoring this fit...")
                    else:
                        sl_offset_good.append(sl_offset[i])
                        if verbose_: print("    Fitted wavelength for sky line {:8.3f}:    center = {:8.3f}     sigma = {:6.3f}    offset = {:7.3f} ".format(sl_center[i],sl_gauss_center[i],sl_gaussian_sigma[i],sl_offset[i]))
                           
                median_offset_fibre = np.nanmedian(sl_offset_good)
                wave_median_offset.append(median_offset_fibre)
                if verbose_: print("\n> Median offset for fibre {:3} = {:7.3f}".format(fibre, median_offset_fibre))
           
            if verbose: 
                sys.stdout.write("\b"*51)
                sys.stdout.write("  Checking fibres completed!                  ")
                sys.stdout.flush()
                print(" ")
            
            # Second-order fit ...         
            bad_numbers = 0
            try:
                xfibre_=[]
                wave_median_offset_ = []
                for i in xfibre:
                    if np.isnan(wave_median_offset[i]) == True:
                        bad_numbers = bad_numbers+1
                    else: 
                        if wave_median_offset[i] == 0: 
                            bad_numbers = bad_numbers+1
                        else:
                            xfibre_.append(i)
                            wave_median_offset_.append(wave_median_offset[i])               
                if bad_numbers > 0 and verbose: print("\n> Skipping {} bad points for the fit...".format(bad_numbers))
                a2x,a1x,a0x = np.polyfit(xfibre_, wave_median_offset_, 2)
                sol=[a0x,a1x,a2x] 
                fx_ = a0x + a1x*np.array(xfibre_)+ a2x*np.array(xfibre_)**2
                ptitle = "Second-order fit to individual offsets"
                if plot:
                    plot_plot(xfibre_, [wave_median_offset_,fx_], xmin=-20,xmax=1000, ptitle=ptitle, xlabel="Fibre", ylabel="offset", hlines=[0])
                    plot_this_again=False
                if verbose: print("\n> Fitting a second-order polynomy a0x +  a1x * fibre + a2x * fibre**2:")
                self.history.append("  sol (found) = "+np.str(sol))   
            except Exception:
                if warnings:
                    print("\n> Something failed doing the fit...")
                    print("  These are the data:")
                    print(" - xfibre =", xfibre_)
                    print(" - wave_median_offset = ",wave_median_offset_)
                    plot_plot(xfibre_,wave_median_offset_)
                    ptitle = "This plot may don't have any sense..."
        else:
            if verbose: print("\n> Solution to the second-order polynomy a0x +  a1x * fibre + a2x * fibre**2 has been provided:")
            a0x = sol[0]
            a1x = sol[1]
            a2x = sol[2]
            ptitle = "Second-order polynomy provided"
            self.history.append("  sol (provided) = "+np.str(sol))  
        
        if verbose: 
            print("  a0x =",a0x,"   a1x =",a1x,"     a2x =",a2x)
            print("\n> sol = [{},{},{}]".format(a0x,a1x,a2x))
        self.sol=[a0x,a1x,a2x]    # Save solution
        fx = a0x + a1x*np.array(xfibre)+ a2x*np.array(xfibre)**2
        
        if plot:
            if sol[0] == 0: pf = wave_median_offset
            else: pf = fx   
            if plot_this_again:
                plot_plot(xfibre,[fx,pf], ptitle= ptitle, color=['red','blue'], xmin=-20,xmax=1000, xlabel="Fibre", ylabel="offset", hlines=[0])

        # Applying results
        if verbose: print("\n> Applying results to all fibres...")
        for fibre in xfibre:
            f = self.intensity_corrected[fibre]
            w_shift = fx[fibre]
            self.intensity_corrected[fibre] =  rebin_spec_shift(w,f,w_shift)                    
        
        # Check results
        if plot:
            if verbose: print("\n> Plotting some results: ")
            
            for line in range(len(xmin)): 
                
                xmin_ = xmin[line]
                xmax_ = xmax[line]
                
                plot_y =[]
                plot_y_corrected =[]
                ptitle = "Before corrections, fibres "
                ptitle_corrected = "After wavelength correction, fibres "           
                if ymax == "": y_max_list = []
                for fibre in fibres_to_plot:
                    plot_y.append(self.intensity[fibre])
                    plot_y_corrected.append(self.intensity_corrected[fibre])
                    ptitle = ptitle+np.str(fibre)+" "
                    ptitle_corrected = ptitle_corrected+np.str(fibre)+" "
                    if ymax == "":
                        y_max_ = []                
                        y_max_.extend((self.intensity[fibre,i]) for i in range(len(w)) if (w[i] > xmin_ and w[i] < xmax_) )  
                        y_max_list.append(np.nanmax(y_max_))
                if ymax == "": ymax = np.nanmax(y_max_list) + 20   # TIGRE
                plot_plot(w,plot_y, ptitle= ptitle, xmin=xmin_,xmax=xmax_, percentile_min=0.1, ymax=ymax) #ymin=ymin, ymax=ymax)   
                plot_plot(w,plot_y_corrected, ptitle= ptitle_corrected, xmin=xmin_,xmax=xmax_, percentile_min=0.1, ymax=ymax) # ymin=ymin, ymax=ymax)    
                y_max_list = []
                ymax = ""
                
        if verbose: print("\n> Small fixing of the 2dFdr wavelengths done!")
        #return 

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def compare_fix_2dfdr_wavelengths(rss1,rss2):
    
    print("\n> Comparing small fixing of the 2dFdr wavelengths between two rss...")
    
    xfibre = list(range(0,rss1.n_spectra))      
    rss1.wavelength_parameters[0]

    a0x,a1x,a2x = rss1.wavelength_parameters[0], rss1.wavelength_parameters[1], rss1.wavelength_parameters[2]
    aa0x,aa1x,aa2x = rss2.wavelength_parameters[0], rss2.wavelength_parameters[1], rss2.wavelength_parameters[2]
    
    fx = a0x + a1x*np.array(xfibre)+ a2x*np.array(xfibre)**2
    fx2 = aa0x + aa1x*np.array(xfibre)+ aa2x*np.array(xfibre)**2
    dif = fx-fx2
    
    plot_plot(xfibre,dif, ptitle= "Fit 1 - Fit 2", xmin=-20,xmax=1000, xlabel="Fibre", ylabel="Dif")
    
    resolution = rss1.wavelength[1]-rss1.wavelength[0]
    error = np.nanmedian(dif)/resolution * 100.
    print("\n> The median rms is {:8.6f} A,  resolution = {:5.2f} A,  error = {:5.3} %".format(np.nanmedian(dif),resolution,error)) 
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
    pa *= np.pi/180
    print("  a -> b :", s*np.sin(pa), -s*np.cos(pa))
    print("  a -> c :", -s*np.sin(60-pa), -s*np.cos(60-pa))
    print("  b -> d :", -np.sqrt(3)*s*np.cos(pa), -np.sqrt(3)*s*np.sin(pa))            
            