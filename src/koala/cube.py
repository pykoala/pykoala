#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 19:43:46 2021

@author: pablo
"""

from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys
from os.path import realpath, dirname#, join

from scipy import interpolate, signal, optimize
from photutils.centroids import centroid_com, centroid_2dg

from scipy.ndimage.interpolation import shift
from scipy.signal import medfilt
import copy

# Import other PyKOALA tasks
from koala.KOALA_RSS import KOALA_RSS
from koala.RSS import RSS
from koala.constants import fuego_color_map
#from koala.io import full_path, read_table, read_cube, spectrum_to_text_file, save_cube_to_fits_file
from koala.io import full_path, read_table, spectrum_to_text_file, save_cube_to_fits_file
from koala.onedspec import COS, SIN, MAD, fit_clip
from koala.plot_plot import plot_plot, basic_statistics
from koala.maps import shift_map, create_map

# Disable some annoying warnings
import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.CRITICAL)

current = dirname(realpath(__file__))
parent = dirname(current)
sys.path.append(current)

# to notify people write #TODO

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

class Interpolated_cube(object):                       # TASK_Interpolated_cube
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
        
        g2d=False: If True uses a 2D Gaussian, else doesn't. (default False)
            
        aligned_coor=False: If True assumes the Cube has been aligned and uses inputted values, else calculates them. (default False)
            
        delta_RA =0: This is a small offset of the RA (right ascension). (default 0)
        
        delta_DEC=0: This is a small offset of the DEC (declination). (default 0)
        
        offsets_files="": The number of files to be aligned. (default "")
    
        offsets_files_position = 0: The position of the current cube in the list of cubes to be aligned. (default 0)
            

    Flux calibration:
        flux_calibration=[0]: This is the flux calibration. (default empty List)
            
        flux_calibration_file="": This is the directory of the flux calibration file. (default "")
        
        
    ADR:
        ADR=False: If True will correct for ADR (Atmospheric Differential Refraction). (default False)
    
        force_ADR = False: If True will correct for ADR even considoring a small correction. (default False)
    
        jump = -1: If a positive number partitions the wavelengths with step size jump, if -1 will not partition. (default -1)
    
        adr_index_fit = 2: This is the fitted polynomial with highest degree n. (default n = 2)
        
        adr_clip_fit = 0.4
        
        ADR_x_fit=[0]: This is the ADR coefficients values for the x axis. (default constant 0)
        
        ADR_y_fit=[0]: This is the ADR coefficients values for the y axis. (default constant 0)               
        
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
    def __init__(self, RSS, rss_file="",  path="",      # RSS can be an OBJECT or a FILE
    
                 pixel_size_arcsec=0.7, kernel_size_arcsec=1.1,
                 shape=[],
                 zeros=False,
                 
                 centre_deg=[], size_arcsec=[], aligned_coor=False, 
                 delta_RA =0,  delta_DEC=0,
                  
                 ADR=False, force_ADR = False, jump = -1, apply_ADR = True,
                 ADR_x_fit=[0],ADR_y_fit=[0], check_ADR = False,    
                 
                 box_x=[0,-1],box_y=[0,-1], half_size_for_centroid = 10,
                 step_tracing = 25, g2d=False, adr_index_fit = 2,
                 kernel_tracing = 5, adr_clip_fit = 0.3,
                 plot_tracing_maps=[], 
                 edgelow = -1, edgehigh = -1,
                 
                 offsets_files="", offsets_files_position =0, 

                 flux_calibration=[], flux_calibration_file ="",
                 
                 trim_cube = False, remove_spaxels_not_fully_covered = True,

                 read_fits_cube = False, n_wave=2048, wavelength=[],description="",objeto="",PA=0,
                 valid_wave_min = 0, valid_wave_max = 0,
                 grating="",CRVAL1_CDELT1_CRPIX1=[0,0,0],total_exptime=0, n_cols=2,n_rows=2, 
                 number_of_combined_files = 1,
                 
                 plot=False, log=True, gamma=0., 
                 plot_rss=True, plot_spectra = True,
                 warnings=False, verbose=True, fig_size=12):   

        
        if plot == False: 
            plot_tracing_maps=[]
            plot_rss = False
            plot_spectra = False
            
            
        self.pixel_size_arcsec = pixel_size_arcsec
        self.kernel_size_arcsec = kernel_size_arcsec
        self.kernel_size_pixels = kernel_size_arcsec/pixel_size_arcsec  # must be a float number!
        self.integrated_map = []
        self.rss_file =[]
        
        self.history=[]
        fcal=False
        

        if read_fits_cube:    # RSS is a cube given in fits file
            self.n_wave = n_wave       
            self.wavelength = wavelength                   
            self.description = description      
            self.object = objeto
            self.PA=PA
            self.grating = grating
            self.CRVAL1_CDELT1_CRPIX1 = CRVAL1_CDELT1_CRPIX1
            self.total_exptime=total_exptime
            self.number_of_combined_files = number_of_combined_files
            self.valid_wave_min = valid_wave_min        
            self.valid_wave_max = valid_wave_max
            
            v = np.abs(self.wavelength-valid_wave_min)
            self.valid_wave_min_index = v.tolist().index(np.nanmin(v))
            v = np.abs(self.wavelength-valid_wave_max)
            self.valid_wave_max_index = v.tolist().index(np.nanmin(v))
 
        else:    
            #self.RSS = RSS
            if rss_file != "" or type(RSS) == str:
                if  type(RSS) == str: rss_file=RSS
                rss_file =full_path(rss_file,path)  #RSS
                RSS=KOALA_RSS(rss_file, rss_clean=True, plot=plot, plot_final_rss = plot_rss,  verbose=verbose)
                
            self.n_spectra = RSS.n_spectra
            self.n_wave = RSS.n_wave        
            self.wavelength = RSS.wavelength                   
            self.description = RSS.description + "\n CUBE"      
            self.object = RSS.object
            self.PA=RSS.PA
            self.grating = RSS.grating
            self.CRVAL1_CDELT1_CRPIX1 = RSS.CRVAL1_CDELT1_CRPIX1
            self.total_exptime=RSS.exptime            
            self.exptimes=[self.total_exptime]
            self.offset_RA_arcsec = RSS.offset_RA_arcsec
            self.offset_DEC_arcsec = RSS.offset_DEC_arcsec
        
            self.rss_file=RSS.filename
            self.rss_file_list = [RSS.filename]
            self.valid_wave_min = RSS.valid_wave_min        
            self.valid_wave_max = RSS.valid_wave_max
            self.valid_wave_min_index = RSS.valid_wave_min_index        
            self.valid_wave_max_index = RSS.valid_wave_max_index        
        
        self.offsets_files = offsets_files                     # Offsets between files when align cubes
        self.offsets_files_position = offsets_files_position   # Position of this cube when aligning
      
        self.seeing = 0.0
        self.flux_cal_step = 0.0
        self.flux_cal_min_wave = 0.0
        self.flux_cal_max_wave = 0.0
        self.adrcor = False
                
        if zeros:
            if read_fits_cube == False and verbose:
                print("\n> Creating empty cube using information provided in rss file: ") 
                print(" ",self.description.replace("\n","") )
        else: 
            if verbose: print("\n> Creating cube from file rss file: ") 
            if verbose: print(" ",self.description.replace("\n",""))
        if read_fits_cube == False and verbose:
            print("  Pixel size  = ",self.pixel_size_arcsec," arcsec")
            print("  kernel size = ",self.kernel_size_arcsec," arcsec")
            
        # centre_deg = [RA,DEC] if we need to give new RA, DEC centre
        if len(centre_deg) == 2:
            self.RA_centre_deg = centre_deg[0]
            self.DEC_centre_deg = centre_deg[1]
        else:
            self.RA_centre_deg = RSS.RA_centre_deg             + delta_RA/3600.
            self.DEC_centre_deg = RSS.DEC_centre_deg           + delta_DEC/3600.
            
        if read_fits_cube == False:                      
            if aligned_coor == True:        
                self.xoffset_centre_arcsec = (self.RA_centre_deg-RSS.ALIGNED_RA_centre_deg)*3600.
                self.yoffset_centre_arcsec = (self.DEC_centre_deg-RSS.ALIGNED_DEC_centre_deg)*3600.            
                if zeros == False and verbose: 
                    print("  Using ALIGNED coordenates for centering cube...")
            else:
                self.xoffset_centre_arcsec = (self.RA_centre_deg-RSS.RA_centre_deg)*3600.  
                self.yoffset_centre_arcsec = (self.DEC_centre_deg-RSS.DEC_centre_deg)*3600. 
                                 
            if len(size_arcsec) == 2:
                if aligned_coor == False:            
                    if verbose: print('  The size of the cube has been given: {}" x {}"'.format(size_arcsec[0],size_arcsec[1]))
                    self.n_cols = np.int(size_arcsec[0]/self.pixel_size_arcsec)
                    self.n_rows = np.int(size_arcsec[1]/self.pixel_size_arcsec)
                else:
                    self.n_cols = np.int(size_arcsec[0]/self.pixel_size_arcsec)  + 2*np.int(self.kernel_size_arcsec/self.pixel_size_arcsec)
                    self.n_rows = np.int(size_arcsec[1]/self.pixel_size_arcsec)  + 2*np.int(self.kernel_size_arcsec/self.pixel_size_arcsec)
            else:
                self.n_cols = 2 * \
                    (np.int(np.nanmax(np.abs(RSS.offset_RA_arcsec-self.xoffset_centre_arcsec))/self.pixel_size_arcsec) 
                     + np.int(self.kernel_size_pixels )) +3 #-3    ### +1 added by Angel 25 Feb 2018 to put center in center
                self.n_rows = 2 * \
                    (np.int(np.nanmax(np.abs(RSS.offset_DEC_arcsec-self.yoffset_centre_arcsec))/self.pixel_size_arcsec) 
                    +  np.int(self.kernel_size_pixels )) +3 #-3   ### +1 added by Angel 25 Feb 2018 to put center in center
    
            # # If we define a specific shape
            if len (shape) == 2:
                self.n_rows = shape[0]
                self.n_cols = shape[1]               
        else:           
            self.n_cols = n_cols
            self.n_rows = n_rows
    
        self.spaxel_RA0= self.n_cols/2  - 1    #TODO: check if these -1 are correct
        self.spaxel_DEC0= self.n_rows/2 - 1

        # Define zeros
        self.weighted_I = np.zeros((self.n_wave, self.n_rows, self.n_cols))
        self.weight = np.zeros_like(self.weighted_I)
        self.flux_calibration=np.zeros(self.n_wave)

        # Check ADR values 
        self.ADR_x_fit = ADR_x_fit
        self.ADR_y_fit = ADR_y_fit
        pp=np.poly1d(ADR_x_fit)
        self.ADR_x = pp(self.wavelength)      
        pp=np.poly1d(ADR_y_fit)
        self.ADR_y = pp(self.wavelength)
                
        ADR_repeat = True
        if np.nansum(self.ADR_y + self.ADR_x) != 0:      # When values for ADR correction are given
            self.history.append("- ADR fit values provided, cube built considering them")
            ADR_repeat = False                           # Do not repeat the ADR correction
            self.adrcor = True                           # The values for ADR are given
            # Computing jump automatically (only when not reading the cube)
            if read_fits_cube == False:
                if jump < -1 : jump == np.abs(jump)
                if jump == -1:
                    cubo_ADR_total = np.sqrt(self.ADR_x**2 +self.ADR_y**2)
                    stop = 0
                    i=1
                    while stop < 1:
                        if np.abs(cubo_ADR_total[i]-cubo_ADR_total[0]) > 0.022:  #0.012:
                            jump=i
                            if verbose: print('  Automatically found that with jump = ', jump,' in lambda, the ADR offset is lower than 0.01"')
                            self.history.append("  Automatically found jump = "+np.str(jump)+" for ADR")
                            stop = 2
                        else:
                            i = i+1
                        if i == len(self.wavelength)/2. :
                            jump = -1
                            if verbose: print('  No value found for jump smaller than half the size of the wavelength! \n  Using jump = -1 (no jump) for ADR.')                       
                            self.history.append("  Using jump = -1 (no jump) for ADR")

                            stop = 2
                else:
                    if verbose: print("  As requested, creating cube considering the median value each ",jump," lambdas for correcting ADR...")
                    self.history.append("  Using given value of jump = "+np.str(jump)+" for ADR")
        else:
            self.history.append("- Cube built without considering ADR correction")
                  
        self.RA_segment = (self.n_cols -1) * self.pixel_size_arcsec
        self.DEC_segment= (self.n_rows -1) * self.pixel_size_arcsec

        if zeros:
            self.data=np.zeros_like(self.weighted_I)
        else:  
            # Build the cube
            self.data=self.build_cube(jump=jump, RSS=RSS) 

            # Trace peaks (check ADR only if requested)          
            if ADR_repeat or check_ADR:
                _check_ADR_ = False
            else:
                _check_ADR_ = True
                        
            if ADR:
                # Define box for tracing peaks if requested / needed
                box_x,box_y = self.box_for_centroid(half_size_for_centroid=half_size_for_centroid, verbose=verbose, plot_map=plot, log=log, g2d=g2d)
                if verbose: print("  Using this box for tracing peaks and checking ADR ...")
                self.trace_peak(box_x=box_x, box_y=box_y, #half_size_for_centroid = half_size_for_centroid,
                                edgelow=edgelow, edgehigh =edgehigh, plot=plot, plot_tracing_maps=plot_tracing_maps,
                                verbose=verbose, adr_index_fit=adr_index_fit, g2d=g2d, kernel_tracing = kernel_tracing,
                                check_ADR = _check_ADR_, step_tracing= step_tracing, adr_clip_fit=adr_clip_fit)
            else:
                self.get_peaks(plot=False, verbose=True)
                if verbose:
                    print("\n> ADR will NOT be checked!")
                    if np.nansum(self.ADR_y + self.ADR_x) != 0:
                        print("  However ADR fits provided and applied:")
                        print("  ADR_x_fit = ",self.ADR_x_fit)
                        print("  ADR_y_fit = ",self.ADR_y_fit)
                    
                    
            # Correct for Atmospheric Differential Refraction (ADR) if requested and not done before
            if ADR and ADR_repeat and apply_ADR: 
                self.weighted_I = np.zeros((self.n_wave, self.n_rows, self.n_cols))
                self.weight = np.zeros_like(self.weighted_I)
                self.ADR_correction(RSS, plot=plot, force_ADR=force_ADR, jump=jump, remove_spaxels_not_fully_covered=remove_spaxels_not_fully_covered)                
                self.trace_peak(check_ADR=True, box_x=box_x, box_y=box_y, half_size_for_centroid=half_size_for_centroid,
                                edgelow=edgelow, edgehigh =edgehigh, 
                                step_tracing =step_tracing,  adr_index_fit=adr_index_fit, g2d=g2d, 
                                kernel_tracing = kernel_tracing, adr_clip_fit=adr_clip_fit,
                                plot_tracing_maps = plot_tracing_maps, plot=plot, verbose = verbose)
                  
            # Apply flux calibration
            self.apply_flux_calibration(flux_calibration=flux_calibration, flux_calibration_file=flux_calibration_file, verbose=verbose, path=path)
            
            if len(self.flux_calibration) != 0: fcal=True
            
            if fcal == False and verbose : print("\n> This interpolated cube does not include an absolute flux calibration")

            # Get integrated maps (all waves and valid range), plots
            self.get_integrated_map(plot=plot,plot_spectra=plot_spectra,fcal=fcal, #box_x=box_x, box_y=box_y, 
                                    verbose=verbose, plot_centroid=True, g2d=g2d, kernel_tracing = kernel_tracing,
                                    log=log, gamma=gamma, nansum = False)  # Barr
                
            # Trim the cube if requested            
            if trim_cube:
                self.trim_cube(half_size_for_centroid=half_size_for_centroid, box_x=box_x, box_y=box_y, ADR=ADR,
                               verbose=verbose, plot=plot, remove_spaxels_not_fully_covered=remove_spaxels_not_fully_covered, 
                               g2d=g2d, adr_index_fit=adr_index_fit, step_tracing=step_tracing, 
                               plot_tracing_maps=plot_tracing_maps)    #### UPDATE THIS, now it is run automatically
        
    
        if read_fits_cube == False and verbose: print("\n> Interpolated cube done!")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def apply_flux_calibration(self, flux_calibration=None, path="",
                               flux_calibration_file = "",  verbose=True):
        """
        Function for applying the flux calibration to a cube.

        Parameters
        ----------
        flux_calibration : Float List, optional
            It is a list of floats. The default is empty
            
        flux_calibration_file : String, optional
            The file name of the flux_calibration. The default is None
            
        path : String, optional
            The directory of the folder the flux_calibration_file is in. The default is None 
        
        verbose : Boolean, optional
            Print results. The default is True.
                         
        Returns
        -------
        self.flux_calibration

        """
        
        if flux_calibration_file != "": # is not None : 
            flux_calibration_file = full_path(flux_calibration_file,path) 
            w_star,flux_calibration = read_table(flux_calibration_file, ["f", "f"] ) 

        if len(flux_calibration) > 0: 
            if verbose: 
                if flux_calibration_file != "": # is not None:
                    print("\n> Applying the absolute flux calibration provided in file:\n ",flux_calibration_file," ...")
                else:
                    print("\n> Applying the absolute flux calibration...")
            
            self.flux_calibration=flux_calibration        
            # This should be in 1 line of step of loop, I couldn't get it # Yago HELP !!
            for y in range(self.n_rows):
                for x in range(self.n_cols):
                    self.data[:,y,x]=self.data[:,y,x] / self.flux_calibration  / 1E16 / self.total_exptime

            self.history.append("- Applied flux calibration")
            #if flux_calibration_file  is not None: self.history.append("  Using file "+flux_calibration_file)
            if flux_calibration_file != "": self.history.append("  Using file "+flux_calibration_file)
            
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
        output_every_few = np.sqrt(self.n_spectra)+1
        next_output = -1
        for i in range(self.n_spectra):
            if verbose:        
                if i > next_output:
                    sys.stdout.write("\b"*6)
                    sys.stdout.write("{:5.2f}%".format(i*100./self.n_spectra))
                    sys.stdout.flush()
                    next_output = i + output_every_few
            offset_rows = (self.offset_DEC_arcsec[i]-self.yoffset_centre_arcsec) / self.pixel_size_arcsec
            offset_cols = (-self.offset_RA_arcsec[i]+self.xoffset_centre_arcsec) / self.pixel_size_arcsec  
            corrected_intensity = RSS.intensity_corrected[i]
            #self.add_spectrum(corrected_intensity, offset_rows, offset_cols, warnings=warnings)
            self.add_spectrum_ADR(corrected_intensity, offset_rows, offset_cols, ADR_x=self.ADR_x, ADR_y=self.ADR_y, jump=jump, warnings=warnings)

        if verbose:
            sys.stdout.write("\b"*6)
            sys.stdout.write("{:5.2f}%".format(100.0))
            sys.stdout.flush()
            print(" ")

        return self.weighted_I / self.weight
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def ADR_correction(self, RSS, plot=True, force_ADR=False, method="new",  remove_spaxels_not_fully_covered = True,
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
            DESCRIPTION. The default is "new". #TODO 
        remove_spaxels_not_fully_covered : Boolean, optional
            DESCRIPTION. The default is True. #TODO 
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
            if _x_ > 0 : 
                cubo = self.combined_cube
                #data_ = np.zeros_like(cubo.weighted_I)
                method = "old"
                #is_combined_cube=True
        except Exception:
            cubo = self
            
            
        # Check if ADR is needed (unless forced)...       
        total_ADR = np.sqrt(cubo.ADR_x_max**2 + cubo.ADR_y_max**2)
        
        cubo.adrcor = True
        if total_ADR < cubo.pixel_size_arcsec * 0.1:   # Not needed if correction < 10 % pixel size
            if verbose:
                print("\n> Atmospheric Differential Refraction (ADR) correction is NOT needed.")
                print('  The computed max ADR value, {:.3f}",  is smaller than 10% the pixel size of {:.2f} arcsec'.format(total_ADR, cubo.pixel_size_arcsec))
            cubo.adrcor = False
            if force_ADR:
                cubo.adrcor = True
                if verbose: print('  However we proceed to do the ADR correction as indicated: "force_ADR = True" ...')
                            

        if cubo.adrcor:  
            cubo.history.append("- Correcting for Atmospheric Differential Refraction (ADR) using:")
            cubo.history.append("  ADR_x_fit = "+np.str(cubo.ADR_x_fit))
            cubo.history.append("  ADR_y_fit = "+np.str(cubo.ADR_y_fit))
            cubo.history.append("  Residua in RA  = "+np.str(np.round(cubo.ADR_x_residua,3))+'" ')
            cubo.history.append("  Residua in Dec = "+np.str(np.round(cubo.ADR_y_residua,3))+'" ')
            cubo.history.append("  Total residua  = "+np.str(np.round(cubo.ADR_total_residua,3))+'" ')
            
            if verbose:
                print("\n> Correcting for Atmospheric Differential Refraction (ADR) using: \n")   
                print("  ADR_x_fit = ",cubo.ADR_x_fit)                  
                print("  ADR_y_fit = ",cubo.ADR_y_fit)                  

            # Computing jump automatically        
            if jump == -1:
                cubo_ADR_total = np.sqrt(cubo.ADR_x**2 +cubo.ADR_y**2)
                stop = 0
                i=1
                while stop < 1:
                    if np.abs(cubo_ADR_total[i]-cubo_ADR_total[0]) > 0.012:
                        jump=i
                        if verbose: print('  Automatically found that with jump = ', jump,' in lambda, the ADR offset is lower than 0.01"')
                        stop = 2
                    else:
                        i = i+1
                    if i == len(cubo.wavelength)/2.:
                        jump = -1
                        if verbose: print('  No value found for jump smaller than half the size of the wavelength! \n  Using jump = -1 (no jump) for ADR.')                       
                        stop = 2
                  
            if method == "old":
                #data_ = np.zeros_like(cubo.weighted_I)
                if verbose: print("\n  Using OLD method (moving planes) ...")
                cubo.history.append("  Using OLD method (moving planes)")
                                
                sys.stdout.flush()
                output_every_few = np.sqrt(cubo.n_wave)+1
                next_output = -1
                
                # First create a CUBE without NaNs and a mask
                cube_shifted = copy.deepcopy(cubo.data) * 0.
                tmp=copy.deepcopy(cubo.data)
                mask=copy.deepcopy(tmp)*0.
                mask[np.where( np.isnan(tmp) == False  )]=1      # Nans stay the same, when a good value = 1.
                tmp_nonan=np.nan_to_num(tmp, nan=np.nanmedian(tmp))  # cube without nans, replaced for median value
                            
                #for l in range(cubo.n_wave):
                for l in range(0,cubo.n_wave,jump):
                    
                    median_ADR_x = np.nanmedian(cubo.ADR_x[l:l+jump])
                    median_ADR_y = np.nanmedian(cubo.ADR_y[l:l+jump])

                    if l > next_output:
                        sys.stdout.write("\b"*37)
                        sys.stdout.write("  Moving plane {:5} /{:5}... {:5.2f}%".format(l, cubo.n_wave, l*100./cubo.n_wave))
                        sys.stdout.flush()
                        next_output = l + output_every_few
                    
                    # For applying shift the array MUST NOT HAVE ANY nans
                    
                    
                    #tmp=copy.deepcopy(cubo.data[l:l+jump,:,:])
                    #mask=copy.deepcopy(tmp)*0.
                    #mask[np.where(np.isnan(tmp))]=1 #make mask where Nans are
                    #kernel = Gaussian2DKernel(5)
                    #tmp_nonan = interpolate_replace_nans(tmp, kernel)
                    #need to see if there are still nans. This can happen in the padded parts of the grid
                    #where the kernel is not large enough to cover the regions with NaNs.
                    #if np.isnan(np.sum(tmp_nonan)):
                    #    tmp_nonan=np.nan_to_num(tmp_nonan)
                    #tmp_shift=shift(tmp_nonan,[-median_ADR_y/cubo.pixel_size_arcsec,-median_ADR_x/cubo.pixel_size_arcsec],cval=np.nan)
                    #mask_shift=shift(mask,[-median_ADR_y/cubo.pixel_size_arcsec,-median_ADR_x/cubo.pixel_size_arcsec],cval=np.nan)
                    #tmp_shift[mask_shift > 0.5]=np.nan
                    #cubo.data[l,:,:]=copy.deepcopy(tmp_shift)

                    #tmp_shift=shift(tmp,[0,-median_ADR_y/cubo.pixel_size_arcsec,-median_ADR_x/cubo.pixel_size_arcsec],cval=np.nan)
                    #cubo.data[l:l+jump,:,:]=copy.deepcopy(tmp_shift)


                    cube_shifted[l:l+jump,:,:]=shift(tmp_nonan[l:l+jump,:,:],[0,-median_ADR_y/cubo.pixel_size_arcsec, -median_ADR_x/cubo.pixel_size_arcsec],cval=np.nan)
    



                    #cubo.data[l:l+jump,:,:]=shift(cubo.data[l:l+jump,:,:],[0,-median_ADR_y/cubo.pixel_size_arcsec, -median_ADR_x/cubo.pixel_size_arcsec],cval=np.nan)
    
                    #print(l,tmp.shape,2*self.ADR_y[l],2*self.ADR_x[l],np.sum(tmp_nonan),np.sum(tmp),np.sum(tmp_shift))
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
                comparison_cube =  copy.deepcopy(cubo)
                comparison_cube.data =  (cube_shifted  - cubo.data) * mask
                comparison_cube.description="Comparing original and shifted cubes"
                vmin=-np.nanmax([np.abs(np.nanmin(comparison_cube.data)), np.abs(np.nanmax(comparison_cube.data))])
                vmax= np.nanmax([np.abs(np.nanmin(comparison_cube.data)), np.abs(np.nanmax(comparison_cube.data))])
                
                comparison_cube.get_integrated_map(plot=plot, plot_spectra=False, verbose=False, plot_centroid=False, 
                                                   cmap="seismic", log=False,vmin=vmin, vmax=vmax)
                
    
                cubo.data = copy.deepcopy(cube_shifted) * mask

            # New procedure 2nd April 2020
            else: 
                if verbose:
                    print("\n  Using the NEW method (building the cube including the ADR offsets)...")                
                    print("  Creating new cube considering the median value each ",jump," lambdas...")
                cubo.adrcor = True
                cubo.data=cubo.build_cube(jump=jump, RSS=RSS)
                cubo.get_integrated_map()
                cubo.history.append("- New cube built considering the ADR correction using jump = "+np.str(jump))           


            # Now remove spaxels with not full wavelength if requested
            if remove_spaxels_not_fully_covered == True:            
                if verbose: print("\n> Removing spaxels that are not fully covered in wavelength in the valid range...")   # Barr

                _mask_ = cubo.integrated_map / cubo.integrated_map     
                for w in range(cubo.n_wave):
                    cubo.data[w] = cubo.data[w]*_mask_
                cubo.history.append("  Spaxels that are not fully covered in wavelength in the valid range have been removed")  

        else:    
            if verbose: print (" NOTHING APPLIED !!!")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def shift_cube(self, jump=-1, delta_RA = 0, delta_DEC = 0, 
                   plot = True, plot_comparison = True,
                   warnings=True, verbose =True, return_cube = False):
        """
        New task for shifting cubes, it works but it needs to be checked
        is the mask is working well and include jump
        Still is best to build the cube with delta_RA and delta_DEC than this

        Parameters
        ----------
        jump : TYPE, optional
            DESCRIPTION. The default is -1.
        delta_RA : TYPE, optional
            DESCRIPTION. The default is 0.
        delta_DEC : TYPE, optional
            DESCRIPTION. The default is 0.
        plot : TYPE, optional
            DESCRIPTION. The default is True.
        warnings : TYPE, optional
            DESCRIPTION. The default is True.
        verbose : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        
        # Check if this is a self.combined cube or a self
        try:
            _x_ = np.nanmedian(self.combined_cube.data)
            if _x_ > 0 : 
                cubo = self.combined_cube
        except Exception:
            cubo = self

        


        cubo.get_integrated_map(plot=plot, log=True)

        if verbose: print('\n> Shifting the cube in RA = '+np.str(delta_RA)+'" and DEC = '+np.str(delta_DEC)+'" ...')
                        
        #sys.stdout.flush()
        #output_every_few = np.sqrt(cubo.n_wave)+1
        #next_output = -1
        
        # First create a CUBE without NaNs and a mask
        cube_shifted = copy.deepcopy(cubo.data) * 0.
        tmp=copy.deepcopy(cubo.data)
        mask=copy.deepcopy(tmp)*0.
        mask[np.where( np.isnan(tmp) == False  )]=1      # Nans stay the same, when a good value = 1.
        #median_value_tmp = np.nanmedian(tmp)
        #cubo.plot_map(mapa=mask[1000])
        tmp_nonan=np.nan_to_num(tmp, nan=np.nanmedian(tmp))  # cube without nans, replaced for median value
                    
        # #for l in range(cubo.n_wave):
        # for l in range(0,cubo.n_wave,jump):
            
        #     median_ADR_x = np.nanmedian(cubo.ADR_x[l:l+jump])
        #     median_ADR_y = np.nanmedian(cubo.ADR_y[l:l+jump])

        #     if l > next_output:
        #         sys.stdout.write("\b"*37)
        #         sys.stdout.write("  Moving plane {:5} /{:5}... {:5.2f}%".format(l, cubo.n_wave, l*100./cubo.n_wave))
        #         sys.stdout.flush()
        #         next_output = l + output_every_few
            
        #     # For applying shift the array MUST NOT HAVE ANY nans
        #     cube_shifted[l:l+jump,:,:]=shift(tmp_nonan[l:l+jump,:,:],[0,-median_ADR_y/cubo.pixel_size_arcsec, -median_ADR_x/cubo.pixel_size_arcsec],cval=np.nan)
 
    
        cube_shifted[:,:,:]=shift(tmp_nonan[:,:,:],[0,delta_DEC/cubo.pixel_size_arcsec, delta_RA/cubo.pixel_size_arcsec],cval=np.nan)
        
        # The mask should be also moved!
        mask_shifted_ = copy.deepcopy(mask)
        mask_shifted  = np.nan_to_num(mask_shifted_, nan=0)  # Change nans to 0
        #cubo.plot_map(mapa=mask_shifted[1000])
        
        mask_shifted[:,:,:]=shift(mask_shifted[:,:,:],[0,delta_DEC/cubo.pixel_size_arcsec, delta_RA/cubo.pixel_size_arcsec],cval=np.nan)
        
        mask_shifted=np.where( mask_shifted < 0.5 , np.nan,  mask_shifted  )   # Solo valen 0 y 1 para máscara
                
        #cubo.plot_map(mapa=mask_shifted[1000])
        

        cubo_final = copy.deepcopy(cubo)
        cubo_final.data = cube_shifted * mask_shifted
        cubo_final.get_peaks(verbose=verbose)
        cubo_final.get_integrated_map(plot=plot, log=True)



        if plot_comparison:
            comparison_cube =  copy.deepcopy(cubo)
            comparison_cube.data =  (cube_shifted  - cubo.data) * mask
            comparison_cube.description="Comparing original and shifted cubes"
            vmin=-np.nanmax([np.abs(np.nanmin(comparison_cube.data)), np.abs(np.nanmax(comparison_cube.data))])
            vmax= np.nanmax([np.abs(np.nanmin(comparison_cube.data)), np.abs(np.nanmax(comparison_cube.data))])
            
            comparison_cube.get_integrated_map(plot=plot, plot_spectra=False, verbose=False, plot_centroid=False, 
                                               cmap="seismic", log=False,vmin=vmin, vmax=vmax)

        if return_cube:
            return cubo_final.data
        else:
            cubo.data = cubo_final.data
            cubo.history.append('- Cube shifted in RA = '+np.str(delta_RA)+'" and DEC = '+np.str(delta_DEC)+'"')

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
        if jump == -1 : jump = self.n_wave

        if np.nanmedian(ADR_x) == 0 and np.nanmedian(ADR_y) == 0 : jump = self.n_wave 

        for l in range(0,self.n_wave,jump):
        
            median_ADR_x = np.nanmedian(self.ADR_x[l:l+jump])
            median_ADR_y = np.nanmedian(self.ADR_y[l:l+jump])
            
            kernel_centre_x = .5*self.n_cols + offset_cols - median_ADR_x /self.pixel_size_arcsec   # *2.
        
            x_min = int(kernel_centre_x - self.kernel_size_pixels)
            x_max = int(kernel_centre_x + self.kernel_size_pixels) + 1
            n_points_x = x_max-x_min
            x = np.linspace(x_min-kernel_centre_x, x_max-kernel_centre_x, n_points_x) / self.kernel_size_pixels
            x[0] = -1.
            x[-1] = 1.
            weight_x = np.diff((3.*x - x**3 + 2.) / 4)
    
            kernel_centre_y = .5*self.n_rows + offset_rows - median_ADR_y /self.pixel_size_arcsec #  *2.

            y_min = int(kernel_centre_y - self.kernel_size_pixels)
            y_max = int(kernel_centre_y + self.kernel_size_pixels) + 1
            n_points_y = y_max-y_min
            y = np.linspace(y_min-kernel_centre_y, y_max-kernel_centre_y, n_points_y) / self.kernel_size_pixels
            y[0] = -1.
            y[-1] = 1.
            weight_y = np.diff((3.*y - y**3 + 2.) / 4)
    
            if x_min < 0 or x_max > self.n_cols+1 or y_min < 0 or y_max > self.n_rows+1:
                if warnings:
                    print("**** WARNING **** : Spectra outside field of view:",x_min,kernel_centre_x,x_max)
                    print("                                                 :",y_min,kernel_centre_y,y_max)
            else:
                bad_wavelengths = np.argwhere(np.isnan(intensity))
                intensity[bad_wavelengths] = 0.
                ones = np.ones_like(intensity)
                ones[bad_wavelengths] = 0.
                self.weighted_I[l:l+jump, y_min:y_max-1, x_min:x_max-1] += intensity[l:l+jump, np.newaxis, np.newaxis] * weight_y[np.newaxis, :, np.newaxis] * weight_x[np.newaxis, np.newaxis, :]
                self.weight[l:l+jump, y_min:y_max-1, x_min:x_max-1] += ones[l:l+jump, np.newaxis, np.newaxis] * weight_y[np.newaxis, :, np.newaxis] * weight_x[np.newaxis, np.newaxis, :]


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def get_peaks(self, box_x=[0,-1],box_y=[0,-1], verbose=True, plot = False):
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
    
        if np.nanmedian(box_x+box_y) != -0.5:
            x0 = box_x[0]
            x1 = box_x[1]           
            y0 = box_y[0]
            y1 = box_y[1]
            x = np.arange(x1-x0)
            y = np.arange(y1-y0)
            if verbose: print("\n> Getting the peaks in box at [{}:{} , {}:{}] ...".format(x0,x1,y0,y1))    
            tmp=copy.deepcopy(self.data[:,y0:y1,x0:x1])
        else:
            if verbose: print("\n> Getting the peaks considering all the spaxels...")  

            tmp=copy.deepcopy(self.data)
            x = np.arange(self.n_cols)
            y = np.arange(self.n_rows)
              
        tmp_img=np.nanmedian(tmp,axis=0)
        sort=np.sort(tmp_img.ravel())
        low_ind=np.where(tmp_img < sort[int(.9*len(sort))])
        for i in np.arange(len(low_ind[0])):
            tmp[:,low_ind[0][i],low_ind[1][i]]=np.nan
            
        weight = np.nan_to_num(tmp)
              
        mean_image = np.nanmean(weight, axis=0)
        mean_image /= np.nanmean(mean_image)
        weight *= mean_image[np.newaxis, :, :]
        xw = x[np.newaxis, np.newaxis, :] * weight
        yw = y[np.newaxis, :, np.newaxis] * weight
        w = np.nansum(weight, axis=(1, 2))
        self.x_peaks = np.nansum(xw, axis=(1, 2)) / w                   # Vector with the x-peak at each wavelength 
        self.y_peaks = np.nansum(yw, axis=(1, 2)) / w                   # Vector with the y-peak at each wavelength 
        self.x_peak_median = np.nanmedian(self.x_peaks)                 # Median value of the x-peak vector
        self.y_peak_median = np.nanmedian(self.y_peaks)                 # Median value of the y-peak vector
        #self.x_peak_median_index = np.nanargmin(np.abs(self.x_peaks-self.x_peak_median)) # Index closest to the median value of the x-peak vector
        #self.y_peak_median_index = np.nanargmin(np.abs(self.y_peaks-self.y_peak_median)) # Index closest to the median value of the y-peak vector       

        if np.nanmedian(box_x+box_y) != -0.5:           # Move peaks from position in box to position in cube
            self.x_peaks = self.x_peaks + box_x[0]
            self.y_peaks = self.y_peaks + box_y[0]
            self.x_peak_median = self.x_peak_median + box_x[0]
            self.y_peak_median = self.y_peak_median + box_y[0]
            
        self.offset_from_center_x_arcsec_tracing = (self.x_peak_median-self.spaxel_RA0)*self.pixel_size_arcsec    # Offset from center using INTEGRATED map
        self.offset_from_center_y_arcsec_tracing = (self.y_peak_median-self.spaxel_DEC0)*self.pixel_size_arcsec    # Offset from center using INTEGRATED map


        if plot:
            ptitle = "Peaks in x axis, median value found in x = "+np.str(np.round(self.x_peak_median,2))
            plot_plot(self.wavelength,self.x_peaks, psym="+", markersize=2, ylabel="spaxel", ptitle=ptitle, percentile_min=0.5, percentile_max=99.5, hlines=[self.x_peak_median])
            ptitle = "Peaks in y axis, median value found in y = "+np.str(np.round(self.y_peak_median,2))            
            plot_plot(self.wavelength,self.y_peaks, psym="+", markersize=2, ylabel="spaxel", ptitle=ptitle, percentile_min=0.5, percentile_max=99.5, hlines=[self.y_peak_median])
            if verbose: print(" ") 

        if verbose:  print('  The peak of the emission is found in  [{:.2f}, {:.2f}] , Offset from center :   {:.2f}" , {:.2f}"'.format(self.x_peak_median,self.y_peak_median, self.offset_from_center_x_arcsec_tracing,self.offset_from_center_y_arcsec_tracing))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def trace_peak(self, box_x=[0,-1],box_y=[0,-1], edgelow=-1, edgehigh=-1,  
                   adr_index_fit=2, step_tracing = 25, g2d = True, 
                   kernel_tracing = 5, adr_clip_fit = 0.3,
                   half_size_for_centroid=0, plot_tracing_maps = [],
                   plot=False, log=True, gamma=0., check_ADR=False, verbose = True): 
        """
        #TODO
        
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
            DESCRIPTION. The default is 100. #TODO 
        g2d : Boolean, optional
            If True uses a 2D Gaussian, else doesn't. The default is True. 
        plot_tracing_maps : List, optional
            DESCRIPTION. The default is []. #TODO 
        plot : Boolean, optional
            If True generates and shows the plots. The default is False.
        log : Boolean, optional
            If true the map is plotted on a log scale. The default is True.
        gamma : Float, optional
            The value for power log. The default is 0..
        check_ADR : Boolean, optional
            DESCRIPTION. The default is False. #TODO 
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

        if half_size_for_centroid != 0 and box_x[1] == -1 and box_y[1] == -1:
            box_x,box_y = self.box_for_centroid(half_size_for_centroid=half_size_for_centroid, plot=False, verbose=False, plot_map=False)
        
        x0=box_x[0]
        x1=box_x[1]
        y0=box_y[0]
        y1=box_y[1]  
        
        if check_ADR:
            plot_residua = False
        else:
            plot_residua = True
        
        ADR_x_fit, ADR_y_fit, ADR_x_max, ADR_y_max, ADR_total, x_peaks, y_peaks, self.ADR_x_residua, self.ADR_y_residua, self.ADR_total_residua, vectors = centroid_of_cube(self, x0,x1,y0,y1, edgelow=edgelow, edgehigh=edgehigh, adr_clip_fit=adr_clip_fit,
                                                                                                   step_tracing=step_tracing, g2d=g2d, plot_tracing_maps=plot_tracing_maps,
                                                                                                   adr_index_fit=adr_index_fit,
                                                                                                   kernel_tracing =kernel_tracing,
                                                                                                   plot=plot, log=log, gamma=gamma,
                                                                                                   plot_residua=plot_residua, verbose=verbose)      
        pp=np.poly1d(ADR_x_fit)
        ADR_x=pp(self.wavelength)    
        pp=np.poly1d(ADR_y_fit)
        ADR_y=pp(self.wavelength)
        
        #self.get_peaks(box_x=box_x, box_y=box_y, verbose=verbose)  ---> Using old routine, but now we have the values from centroid!
        self.x_peaks = x_peaks                                          # Vector with the x-peak at each wavelength 
        self.y_peaks = y_peaks                                          # Vector with the y-peak at each wavelength 
        self.x_peak_median = np.nanmedian(self.x_peaks)                 # Median value of the x-peak vector
        self.y_peak_median = np.nanmedian(self.y_peaks)                 # Median value of the y-peak vector
        #self.x_peak_median_index = np.nanargmin(np.abs(self.x_peaks-self.x_peak_median)) # Index closest to the median value of the x-peak vector
        #self.y_peak_median_index = np.nanargmin(np.abs(self.y_peaks-self.y_peak_median)) # Index closest to the median value of the y-peak vector           
        self.offset_from_center_x_arcsec_tracing = (self.x_peak_median-self.spaxel_RA0)*self.pixel_size_arcsec     # Offset from center using CENTROID
        self.offset_from_center_y_arcsec_tracing = (self.y_peak_median-self.spaxel_DEC0)*self.pixel_size_arcsec    # Offset from center using CENTROID

        self.ADR_x=ADR_x  
        self.ADR_y=ADR_y
        self.ADR_x_max = ADR_x_max   
        self.ADR_y_max = ADR_y_max
        self.ADR_total = ADR_total 

        if ADR_total > self.pixel_size_arcsec*0.1:
            if verbose: print('\n  The combined ADR, {:.2f}", is larger than 10% of the pixel size! Applying this ADR correction is needed !!'.format(ADR_total))
        elif verbose: print('\n  The combined ADR, {:.2f}", is smaller than 10% of the pixel size! Applying this ADR correction is NOT needed'.format(ADR_total))
                   

        if check_ADR == False:
            self.ADR_x_fit = ADR_x_fit
            self.ADR_y_fit = ADR_y_fit
            if verbose:
                print("\n> Results of the ADR fit (to be applied in a next step if requested):\n")
                print("  ADR_x_fit = ",ADR_x_fit)
                print("  ADR_y_fit = ",ADR_y_fit)
        elif verbose:
                print("\n> We are only checking the ADR correction, data will NOT be corrected !")             
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def box_for_centroid(self, half_size_for_centroid=6, verbose=True, plot=False, g2d=False,
                         plot_map = True, log= True, gamma=0.):
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
        
        max_x = np.int(np.round(self.x_peak_median,0))
        max_y = np.int(np.round(self.y_peak_median,0)) 
            
        if verbose: print("\n> Defining a box centered in [ {} , {} ] and width +-{} spaxels:".format(max_x, max_y, half_size_for_centroid))
        box_x_centroid = [max_x-half_size_for_centroid,max_x+half_size_for_centroid]
        box_y_centroid = [max_y-half_size_for_centroid,max_y+half_size_for_centroid]
        
        if box_x_centroid[0] < 0 : box_x_centroid[0] = 0
        if box_y_centroid[0] < 0 : box_y_centroid[0] = 0
        if box_x_centroid[1] > self.n_cols-1 :  box_x_centroid[1] = self.n_cols-1 
        if box_y_centroid[1] > self.n_rows-1 :  box_y_centroid[1] = self.n_rows-1 
        
        if verbose: print("  box_x =[ {}, {} ],  box_y =[ {}, {} ]".format(box_x_centroid[0],box_x_centroid[1],box_y_centroid[0],box_y_centroid[1]))

        if plot_map: self.plot_map(plot_box=True, box_x=box_x_centroid, box_y=box_y_centroid, log=log, gamma=gamma, g2d=g2d,
                                   spaxel=[max_x,max_y], plot_centroid=True)


        return box_x_centroid,box_y_centroid
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def get_integrated_map(self, min_wave = None, max_wave= None, nansum=True, 
                           vmin=1E-30, vmax=1E30, fcal=False,  log=True, gamma=0., cmap="fuego",
                           box_x=[0,-1], box_y=[0,-1], trimmed=False,
                           g2d=False, plot_centroid=True, 
                           trace_peaks=False, adr_index_fit=2, edgelow=-1, edgehigh=-1, step_tracing=25,
                           kernel_tracing = 5,  adr_clip_fit=0.3,
                           plot=False, plot_spectra=False, plot_tracing_maps=[], verbose=True) :  ### CHECK
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
        vmin : Float, optional
            Minimum value to consider in the colour map. The default is 1E-30. 
        vmax : Float, optional
            Maximum value to consider in the colour map. The default is 1E30. 
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
        trimmed : Boolean, optional
            If True only plots the map within box_x and box_y. The default is False
        g2d : Boolean, optional
            If True uses a 2D Gaussian, else doesn't. The default is False. 
        plot_centroid : Boolean, optional
            If True will plot the centroid. The default is True.
        trace_peaks : Boolean, optional
            DESCRIPTION. The default is False. #TODO 
        adr_index_fit : Integer, optional
            This is the fitted polynomial with highest degree n. The default is 2. 
        edgelow : Integer, optional
            This is the lowest value in the wavelength range in terms of pixels. The default is -1. 
        edgehigh : Integer, optional
            This is the highest value in the wavelength range in terms of pixels, (maximum wavelength - edgehigh). The default is -1. 
        step_tracing : Integer, optional
            DESCRIPTION. The default is 100. #TODO 
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
        
        if min_wave is None: min_wave = self.valid_wave_min
        if max_wave is None: max_wave = self.valid_wave_max
                
        if nansum:
            self.integrated_map_all = np.nansum(self.data, axis=0)                        
            self.integrated_map = np.nansum(self.data[np.searchsorted(self.wavelength, min_wave):np.searchsorted(self.wavelength, max_wave)],axis=0)            
        else:
            self.integrated_map_all = np.sum(self.data, axis=0)                        
            self.integrated_map = np.sum(self.data[np.searchsorted(self.wavelength, min_wave):np.searchsorted(self.wavelength, max_wave)],axis=0)            
            
        # Search for peak of emission in integrated map and compute offsets from centre
        self.max_y,self.max_x = np.unravel_index(np.nanargmax(self.integrated_map), self.integrated_map.shape)
        self.offset_from_center_x_arcsec_integrated = (self.max_x-self.spaxel_RA0)*self.pixel_size_arcsec    # Offset from center using INTEGRATED map
        self.offset_from_center_y_arcsec_integrated = (self.max_y-self.spaxel_DEC0)*self.pixel_size_arcsec    # Offset from center using INTEGRATED map

        for row in range(len(self.integrated_map[:])):  # Put nans instead of 0
            v_ = self.integrated_map[row] 
            self.integrated_map[row] = [np.nan if x == 0 else x for x in v_]

        if plot_spectra:            
            self.plot_spectrum_cube(-1,-1,fcal=fcal)
            print(" ")
            self.plot_spectrum_cube(self.max_x,self.max_y,fcal=fcal)

        if trace_peaks:
            if verbose: print("\n> Tracing peaks using all data...")
            self.trace_peak(edgelow=edgelow, edgehigh=edgehigh,  #box_x=box_x,box_y=box_y,
                   adr_index_fit=adr_index_fit, step_tracing = step_tracing, g2d = g2d, adr_clip_fit=adr_clip_fit,
                   plot_tracing_maps = plot_tracing_maps, plot=False, check_ADR=False, verbose = False)
        #else:
            #self.get_peaks(verbose=verbose)

        if verbose: 
            print("\n> Created integrated map between {:5.2f} and {:5.2f} considering nansum = {:}".format(min_wave, max_wave,nansum))
            print("  The cube has a size of {} x {} spaxels = [ 0 ... {} ] x [ 0 ... {} ]".format(self.n_cols, self.n_rows, self.n_cols-1, self.n_rows-1))
            print("  The peak of the emission in integrated image is in spaxel [",self.max_x,",",self.max_y ,"]")
            print("  The peak of the emission tracing all wavelengths is in position [",np.round(self.x_peak_median,2),",",np.round(self.y_peak_median,2),"]")

        if plot:
            self.plot_map(log=log, gamma=gamma, spaxel=[self.max_x,self.max_y], spaxel2=[self.x_peak_median,self.y_peak_median], fcal=fcal, 
                          box_x=box_x, box_y=box_y, plot_centroid=plot_centroid, g2d=g2d, cmap=cmap, vmin=vmin, vmax=vmax, trimmed=trimmed)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def plot_spectrum_cube(self, x=-1, y=-1, lmin=0, lmax=0, fmin=1E-30, fmax=1E30, 
                           fcal=False, fig_size=10., fig_size_y=0., save_file="", title="", z=0.,
                           median=False, plot = True, verbose=True):    # Must add: elines, alines...
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
                    spectrum=np.nanmedian(np.nanmedian(self.data, axis=1),axis=1)
                    if title == "" : title = "Median spectrum in {}".format(self.description)
                else:
                    if verbose: print("\n> Computing the integrated spectrum of the cube...")
                    spectrum=np.nansum(np.nansum(self.data, axis=1),axis=1)
                    if title == "" : title = "Integrated spectrum in {}".format(self.description)
            else:
                if verbose: print("> Spectrum of spaxel [",x,",",y,"] :")
                spectrum=self.data[:,y,x]
                if title == "" : title = "Spaxel [{},{}] in {}".format(x,y, self.description)
        else:
            list_of_spectra=[]
            if verbose: 
                if median:
                    print("\n> Median spectrum of selected spaxels...")
                else:    
                    print("\n> Integrating spectrum of selected spaxels...")
                print("  Adding spaxel  1  = [",x[0],",",y[0],"]")
            list_of_spectra.append(self.data[:,y[0],x[0]])
            for i in range(len(y)-1):
               list_of_spectra.append(self.data[:,y[i+1],x[i+1]])
               if verbose: print("  Adding spaxel ",i+2," = [",x[i+1],",",y[i+1],"]")             
            n_spaxels = len(x)   
            
            if median:                
                spectrum=np.nanmedian(list_of_spectra, axis=0)
                if title == "" : title = "Median spectrum adding {} spaxels in {}".format(n_spaxels, self.description)
            else:    
                spectrum=np.nansum(list_of_spectra, axis=0)
                if title == "" : title = "Integrated spectrum adding {} spaxels in {}".format(n_spaxels, self.description)

        if fcal == False:
            ylabel="Flux [relative units]"
        else:
            spectrum=spectrum *1E16 
            ylabel="Flux [ 10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]"


        if plot:    # TODO: This should be done with plot_plot()
            # Set limits        
            if fmin == 1E-30:
                fmin = np.nanmin(spectrum)
            if fmax == 1E30:
                fmax = np.nanmax(spectrum)                
            if lmin == 0:
                lmin = self.wavelength[0]
            if lmax == 0:
                lmax = self.wavelength[-1]
    
            if fig_size_y == 0. : fig_size_y = fig_size/3.  
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
                plt.ylim(fmin,fmax)
                plt.xlim(lmin,lmax)
            except Exception:  
                print("  WARNING! Something failed getting the limits of the plot...")
                print("  Values given: lmin = ",lmin, " , lmax = ", lmax)
                print("                fmin = ",fmin, " , fmax = ", fmax)

            # Identify lines  
            if z != 0:
                # Emission lines
                elines=[3727.00, 3868.75, 3967.46, 3889.05, 4026., 4068.10, 4101.2, 4340.47, 4363.21, 4471.48, 4658.10, 4686., 4711.37, 4740.16, 4861.33, 4958.91, 5006.84, 5197.82, 6300.30, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7281.35, 7320, 7330 ]           
                #elines=[3727.00, 3868.75, 3967.46, 3889.05, 4026., 4068.10, 4101.2, 4340.47, 4363.21, 4471.48, 4658.10, 4861.33, 4958.91, 5006.84, 5197.82, 6300.30, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7320, 7330 ]           
                for i in elines:
                    plt.plot([i*(1+z),i*(1+z)],[fmin,fmax],"g:",alpha=0.95)
                # Absorption lines
                alines=[3934.777,3969.588,4308,5175]    #,4305.61, 5176.7]   # POX 4         
                #alines=[3934.777,3969.588,4308,5170]    #,4305.61, 5176.7]            
                for i in alines:
                    plt.plot([i*(1+z),i*(1+z)],[fmin,fmax],"r:",alpha=0.95)
     
            # Show or save file
            if save_file == "":
               plt.show()
            else:
               plt.savefig(save_file)
            plt.close()
        
        return spectrum
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def spectrum_of_box(self, box_x=[], box_y=[], center=[10,10], width=3, 
                        log=True, gamma=0.,
                        plot =True, verbose = True, median=False):
        """
        Given a box (or a center with a width size, all in spaxels), 
        this task provides the integrated or median spectrum and the list of spaxels.

        Parameters
        ----------
        box_x : Integer List, optional
            When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [].
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [].
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
            center = [np.int(np.round(center[0],0)), np.int(np.round(center[1],0))]
            if verbose: 
                if median:
                    print("\n> Median spectrum of box with center [ {} , {} ] and width = {} spaxels (0 is valid)".format(center[0],center[1],width))
                else:
                    print("\n> Integrating spectrum of box with center [ {} , {} ] and width = {} spaxels (0 is valid)".format(center[0],center[1],width))                    
            if width % 2 == 0:
                if verbose: print("  Width is an EVEN number, the given center will not be in the center")
            hw = np.int((width-1)/2)
            box_x=[center[0]-hw,center[0]+hw+1]
            box_y=[center[1]-hw,center[1]+hw+1]
            description = "Spectrum of box, center "+np.str(center[0])+" x "+np.str(center[1])+", width "+np.str(width)
        else:
            if verbose: 
                if median:
                    print("\n> Median spectrum of box with [ {} : {} ] x  [ {} : {} ] (0 is valid)".format(box_x[0],box_x[1], box_y[0],box_y[1]))
                else:
                    print("\n> Integrating spectrum of box with [ {} : {} ] x  [ {} : {} ] (0 is valid)".format(box_x[0],box_x[1], box_y[0],box_y[1]))                 
            description = "Spectrum of box "+np.str(box_x[0])+":"+np.str(box_x[1])+" x "+np.str(box_y[0])+":"+np.str(box_y[1])
            center = 0
            width = 0
        
        if plot:
            self.plot_map(spaxel=center,  box_x= box_x,  box_y= box_y, gamma=gamma, log=log, 
                          description=description, verbose=verbose)

        list_x,list_y = [],[]
      
        for i in range(box_x[0],box_x[1]):
            for j in range(box_y[0],box_y[1]):
                list_x.append(i)
                list_y.append(j)

        spectrum_box=self.plot_spectrum_cube(list_x,list_y, verbose=verbose, plot=plot, median=median)
        
        spaxel_list =[]
        for i in range(len(list_x)):
            spaxel_list.append([list_x[i],list_y[i]])
            
        return spectrum_box,spaxel_list
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def spectrum_offcenter(self, offset_x=5, offset_y=5, distance=5, width=3, pa="", peak=[],
                           median=False, plot=True, verbose=True):   # self
        """
        This tasks calculates spaxels coordenates offcenter of the peak, 
        and calls task spectrum_of_box to get the integrated or median spectrum of the region   
        
        Example: spec,spaxel_list = cube.spectrum_offcenter(distance=10, width=5, pa=cube.pa)
        
        Parameters
        ----------
        offset_x : Integer, optional
            DESCRIPTION. The default is 5. #TODO 
        offset_y : Integer, optional
            DESCRIPTION. The default is 5. #TODO 
        distance : Integer, optional
            DESCRIPTION. The default is 5. #TODO 
        width : Integer, optional
            DESCRIPTION. The default is 3. #TODO 
        pa : String, optional
            This is the position angle. The default is "".
        peak : List, optional
            DESCRIPTION. The default is []. #TODO 
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
            x0=self.x_peak_median
            y0=self.y_peak_median
        
        if pa != "":
            offset_x = distance*COS(pa)   #self.PA
            offset_y = distance*SIN(pa)

        center=[x0 + offset_x, y0 + offset_y]
        
        if verbose: 
            if median:
                print("\n> Calculating median spectrum for a box of width {} and center [ {} , {} ]".format(width,np.round(center[0],2),np.round(center[1],2)))         
            else:
                print("\n> Calculating integrated spectrum for a box of width {} and center [ {} , {} ]".format(width,np.round(center[0],2),np.round(center[1],2)))                 
            if peak != []:
                print("  This is an offset of [ {} , {} ] from given position at [ {} , {} ]".format(np.round(offset_x,2),np.round(offset_y,2),np.round(x0,2),np.round(y0,2)))
            else:                
                print("  This is an offset of [ {} , {} ] from peak emission at [ {} , {} ]".format(np.round(offset_x,2),np.round(offset_y,2),np.round(x0,2),np.round(y0,2)))

            if pa != "":
                print("  This was obtained using a distance of {} spaxels and a position angle of {}".format(distance,np.round(pa,2)))
                       
        spectrum_box,spaxel_list =  self.spectrum_of_box(box_x=[], center=center, width=width, median=median, plot=plot, verbose=verbose )  

        return spectrum_box,spaxel_list
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def plot_weight(self, log = False, gamma=0., vmin=1E-30, vmax=1E30,
                    cmap="gist_gray", fig_size= 10, 
                    save_file="", #ADR = True, 
                    description="", contours=True, clabel=False, verbose = True,
                    spaxel=0, spaxel2=0, spaxel3=0,
                    box_x=[0,-1], box_y=[0,-1], 
                    circle=[0,0,0],circle2=[0,0,0],circle3=[0,0,0],
                    plot_centre=True, plot_spaxel=False, plot_spaxel_grid=True,
                    label_axes_fontsize=15, axes_fontsize = 14, c_fontsize = 12, title_fontsize= 16,
                    fraction=None, pad=None, colorbar_ticksize= 14, colorbar_fontsize = 15, barlabel=""):
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
            DESCRIPTION. The default is 1E-30. #TODO 
        vmax : Float, optional
            DESCRIPTION. The default is 1E30. #TODO 
        cmap : String, optional
            This is the colour of the map. The default is "gist_gray".
        fig : Integer, optional
            DESCRIPTION. The default is 10. #TODO 
        save_file : String, optional
            Save plot in file "file.extension". The default is "".
        #ADR : Boolean, optional
            If True will correct for ADR (Atmospheric Differential Refraction). The default is True.
        description : String, optional
            This is the description of the cube. The default is "".
        contours : Boolean, optional
            DESCRIPTION. The default is True. #TODO 
        clabel : Boolean, optional
            DESCRIPTION. The default is False. #TODO 
        verbose : Boolean, optional
            Print results. The default is True.
        spaxel : Integer, optional
            DESCRIPTION. The default is 0. #TODO 
        spaxel2 : Integer, optional
            DESCRIPTION. The default is 0. #TODO 
        spaxel3 : Integer, optional
            DESCRIPTION. The default is 0. #TODO 
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
            DESCRIPTION. The default is 12. #TODO 
        title_fontsize : Integer, optional
            This is the size of the font for the title. The default is 16.
        fraction : Float, optional
            DESCRIPTION. The default is 0.0457. #TODO 
        pad : Float, optional
            DESCRIPTION. The default is 0.02. #TODO 
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
            if description == "" : description = self.description+" - Weight map"               
            self.plot_map(interpolated_map, fig_size=fig_size, cmap=cmap, save_file=save_file, 
                          description=description, weight = True,
                          contours=contours, clabel=clabel, verbose=verbose,
                          spaxel=spaxel, spaxel2=spaxel2, spaxel3=spaxel3,
                          box_x = box_x, box_y = box_y,
                          circle = circle, circle2=circle2, circle3 = circle3,
                          plot_centre = plot_centre, plot_spaxel=plot_spaxel, plot_spaxel_grid=plot_spaxel_grid,
                          log = log, gamma=gamma, vmin=vmin, vmax=vmax,
                          label_axes_fontsize=label_axes_fontsize, axes_fontsize = axes_fontsize, c_fontsize = c_fontsize, title_fontsize= title_fontsize,
                          fraction=fraction, pad=pad, colorbar_ticksize= colorbar_ticksize, colorbar_fontsize = colorbar_fontsize, barlabel = barlabel)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def create_map(self, line, w2 = 0., gaussian_fit = False, gf=False,
                   lowlow= 50, lowhigh=10, highlow=10, highhigh = 50,
                   show_spaxels=[], verbose = True, description = "" ):
        """
        Runs task "create_map"

        Parameters
        ----------
        line : TYPE
            DESCRIPTION.
        w2 : Float, optional
            DESCRIPTION. The default is 0.. #TODO
        gaussian_fit : Boolean, optional
            DESCRIPTION. The default is False. #TODO 
        gf : Boolean, optional
            DESCRIPTION. The default is False. #TODO 
        lowlow : Integer, optional
            DESCRIPTION. The default is 50. #TODO 
        lowhigh : Integer, optional
            DESCRIPTION. The default is 10. #TODO 
        highlow : Integer, optional
            DESCRIPTION. The default is 10. #TODO 
        highhigh : Integer, optional
            DESCRIPTION. The default is 50. #TODO 
        show_spaxels : List, optional
            DESCRIPTION. The default is []. #TODO 
        verbose : Boolean, optional
			Print results. The default is True.
        description : String, optional
            This is the description of the cube. The default is "".

        Returns
        -------
        mapa : Map
            DESCRIPTION. #TODO 

        """
        mapa = create_map(cube=self, line=line, w2 = w2, gaussian_fit = gaussian_fit, gf=gf,
                   lowlow= lowlow, lowhigh=lowhigh, highlow=highlow, highhigh = highhigh,
                   show_spaxels=show_spaxels, verbose = verbose, description = description )
        return mapa
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def plot_wavelength(self, wavelength, w2=0.,  
                        log = False, gamma=0., vmin=1E-30, vmax=1E30,
                        cmap=fuego_color_map, fig_size= 10, fcal=False,
                        save_file="", description="", contours=True, clabel=False, verbose = True,
                        spaxel=0, spaxel2=0, spaxel3=0,
                        box_x=[0,-1], box_y=[0,-1],  plot_centroid=False, g2d=True, half_size_for_centroid  = 0,
                        circle=[0,0,0],circle2=[0,0,0],circle3=[0,0,0],
                        plot_centre=True, plot_spaxel=False, plot_spaxel_grid=True, 
                        label_axes_fontsize=15, axes_fontsize = 14, c_fontsize = 12, title_fontsize= 16,
                        fraction=None, pad=None, colorbar_ticksize= 14, colorbar_fontsize = 15, barlabel="") :
        """
        Plot map at a particular wavelength or in a wavelength range.

        Parameters
        ----------
        wavelength: Float
              wavelength to be mapped.
        w2 : Float, optional
            if this parameter is given, the map is the integrated map between "wavelength" and "w2". The default is 0 (not given) 
        log : Boolean, optional
            If true the map is plotted on a log scale. The default is False.
        gamma : Float, optional
            Exponent of a powerlaw for plotting the map. Default is 0.. If given, this has preference over log.
        vmin : Float, optional
            DESCRIPTION. The default is 1E-30. #TODO 
        vmax : Float, optional
            DESCRIPTION. The default is 1E30. #TODO 
        cmap: 
            Color map used.
            Velocities: cmap="seismic". The default is fuego_color_map.
        fig_size : Integer, optional
            This is the size of the figure. The default is 10.
        fcal : Boolean, optional
            If fcal=True, cube.flux_calibration is used. The default is False. #TODO 
        save_file : String, optional
            Save plot in file "file.extension". The default is "".
        description : String, optional
            This is the description of the cube. The default is "".
        contours : Boolean, optional
            DESCRIPTION. The default is True. #TODO 
        clabel : Boolean, optional
            DESCRIPTION. The default is False. #TODO 
        verbose : Boolean, optional
			Print results. The default is True.
        spaxel : Integer, optional
            DESCRIPTION. The default is 0. #TODO 
        spaxel2 : Integer, optional
            DESCRIPTION. The default is 0. #TODO 
        spaxel3 : Integer, optional
            DESCRIPTION. The default is 0. #TODO 
        box_x : Integer List, optional
            When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [0,-1].
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [0,-1].
        circle : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. #TODO 
        circle2 : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. #TODO 
        circle3 : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. #TODO 
        plot_centre : Boolean, optional
            DESCRIPTION. The default is True. #TODO 
        plot_spaxel : Boolean, optional
            If True will plot the spaxel. The default is False. 
        plot_spaxel_grid : Boolean, optional
            If True plots the spaxel grid. The default is True.
        label_axes_fontsize : Integer, optional
            This is the size of the axes labels. The default is 15.
        axes_fontsize : Integer, optional
            This is the size of the font on the axes. The default is 14.
        c_fontsize : Integer, optional
            DESCRIPTION. The default is 12. #TODO 
        title_fontsize : Integer, optional
            This is the size of the titles text. The default is 16.
        fraction : Float, optional
            DESCRIPTION. The default is 0.0457. #TODO 
        pad : Float, optional
            DESCRIPTION. The default is 0.02. #TODO 
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
        
        #mapa, description_ = self.make_map(wavelength, w2=w2)
        description_, mapa, w1_, w2_ = self.create_map(line=wavelength, w2=w2)
        
        if description == "" : description = description_
        
        self.plot_map(mapa=mapa, 
                      cmap=cmap, fig_size=fig_size, fcal=fcal, 
                      save_file=save_file, description=description,  contours=contours, clabel=clabel, verbose=verbose,
                      spaxel=spaxel, spaxel2=spaxel2, spaxel3=spaxel3,
                      box_x = box_x, box_y = box_y,
                      plot_centroid=plot_centroid, g2d=g2d, 
                      circle = circle, circle2=circle2, circle3 = circle3,half_size_for_centroid=half_size_for_centroid,
                      plot_centre = plot_centre, plot_spaxel=plot_spaxel, plot_spaxel_grid=plot_spaxel_grid,
                      log=log, gamma=gamma, vmin=vmin, vmax=vmax,
                      label_axes_fontsize=label_axes_fontsize, axes_fontsize = axes_fontsize, c_fontsize = c_fontsize, title_fontsize= title_fontsize,
                      fraction=fraction, pad=pad, colorbar_ticksize= colorbar_ticksize, colorbar_fontsize = colorbar_fontsize, barlabel = barlabel)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def plot_map(self, mapa="", log = False, gamma = 0., vmin=1E-30, vmax=1E30, fcal=False,
                 trimmed=False,
                 cmap="fuego", weight = False, velocity= False, fwhm=False, ew=False, ratio=False,
                 contours=True, clabel=False,
                 line =0,  
                 spaxel=0, spaxel2=0, spaxel3=0,
                 box_x=[0,-1], box_y=[0,-1], 
                 plot_centroid=False, g2d=True, half_size_for_centroid  = 0,
                 circle=[0,0,0],circle2=[0,0,0],circle3=[0,0,0],
                 plot_box=False, plot_centre=True, plot_spaxel=False, plot_spaxel_grid=True, alpha_grid=0.1, 
                 plot_spaxel_list=[], color_spaxel_list="blue", alpha_spaxel_list=0.4,
                 label_axes_fontsize=15, axes_fontsize = 14, c_fontsize = 12, title_fontsize= 16,
                 fraction=None, pad=None, colorbar_ticksize= 14, colorbar_fontsize = 15, barlabel="",
                 description="", fig_size=10, save_file="", verbose = True):
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
        mapa : np.array(float), optional
            Map to be plotted. If not given, it plots the integrated map.
        log : Boolean, optional
            If true the map is plotted on a log scale. The default is False.
        gamma : Float, optional
            The value for power log. The default is 0..
        vmin : Float, optional
            DESCRIPTION. The default is 1E-30. #TODO 
        vmax : Float, optional
            DESCRIPTION. The default is 1E30. #TODO 
        fcal : Boolean, optional
            If fcal=True, cube.flux_calibration is used. The default is False.
        trimmed : Boolean, optional
            DESCRIPTION. The default is False. #TODO
        cmap : String, optional
            This is the colour of the map. The default is "fuego". 
        weight : Boolean, optional
            DESCRIPTION. The default is False. #TODO
        velocity : Boolean, optional
            DESCRIPTION. The default is False. #TODO
        fwhm : Boolean, optional
            DESCRIPTION. The default is False. #TODO
        ew : Boolean, optional
            DESCRIPTION. The default is False. #TODO
        ratio : Boolean, optional
            DESCRIPTION. The default is False. #TODO
        contours : Boolean, optional
            DESCRIPTION. The default is True. #TODO
        clabel : Boolean, optional
            DESCRIPTION. The default is False. #TODO
        line : Integer, optional
            DESCRIPTION. The default is 0. #TODO
        spaxel : Integer, optional
            DESCRIPTION. The default is 0. #TODO
        spaxel2 : Integer, optional
            DESCRIPTION. The default is 0. #TODO
        spaxel3 : Integer, optional
            DESCRIPTION. The default is 0. #TODO
        box_x : Integer List, optional
             When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [0,-1].
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [0,-1].
        plot_centroid : Boolean, optional
            If True will plot the centroid. The default is False.
        g2d : Boolean, optional
            If True uses a 2D Gaussian, else doesn't. The default is True. 
        half_size_for_centroid : Integer, optional
            This is half the length/width of the box. The default is 0.
        circle : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. #TODO
        circle2 : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. #TODO
        circle3 : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. #TODO
        plot_box : Boolean, optional
            DESCRIPTION. The default is False. #TODO
        plot_centre : Boolean, optional
            DESCRIPTION. The default is True. #TODO
        plot_spaxel : Boolean, optional
            If True will plot the spaxel. The default is False.
        plot_spaxel_grid : Boolean, optional
            If True plots the spaxel grid. The default is True.
        alpha_grid : Float, optional
            DESCRIPTION. The default is 0.1. #TODO
        plot_spaxel_list : List, optional
            DESCRIPTION. The default is []. #TODO
        color_spaxel_list : String, optional
            DESCRIPTION. The default is "blue". #TODO
        alpha_spaxel_list : Float, optional
            DESCRIPTION. The default is 0.4. #TODO
        label_axes_fontsize : Integer, optional
            This is the size of the axes labels. The default is 15.
        axes_fontsize : Integer, optional
            This is the size of the font on the axes. The default is 14.
        c_fontsize : Integer, optional
            DESCRIPTION. The default is 12. #TODO
        title_fontsize : Integer, optional
            This is the size of the font for the title. The default is 16.
        fraction : Float, optional
            DESCRIPTION. The default is 0.0457. #TODO
        pad : Float, optional
            DESCRIPTION. The default is 0.02. #TODO
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
                  
        #plot_spaxel_list_ = copy.deepcopy(plot_spaxel_list)
        #plot_spaxel_list = copy.deepcopy(lista)
        #if len(plot_spaxel_list_) > 0: plot_spaxel_list.append(plot_spaxel_list_[0])
        #print (plot_spaxel_list)
                
        mapa_=mapa
        
        # Check if map needs to be trimmed:
        if np.nanmedian(box_x+box_y)!=-0.5 and plot_box is False: trimmed=True  

        # Get description
        ptrimmed = " - trimmed x = [ {:}, {:} ] , y = [ {:} , {:} ]".format(box_x[0],box_x[1], box_y[0],box_y[1]) 
        try:
            if type(mapa[0]) == str:       # Maps created by PyKOALA have [description, map, l1, l2 ...] 
                mapa_ = mapa[1]
                if description == "": 
                    if trimmed:
                        description = mapa[0]+ ptrimmed   
                    else:
                        description = mapa[0]
        except Exception:
            if mapa == "" :
                if len(self.integrated_map) == 0:  self.get_integrated_map(verbose=verbose)
                       
                mapa_=self.integrated_map 
                if description == "": 
                    if trimmed:
                        description = self.description+" - Integrated Map" + ptrimmed           
                    else:
                        description = self.description+" - Integrated Map" 
                        
        if description == "" :
            if trimmed:
                description = self.description + ptrimmed
            else:
                description = self.description

        # Trim the map if requested 
        if trimmed:
            mapa= copy.deepcopy(mapa_[box_y[0]:box_y[1],box_x[0]:box_x[1]])
            extent1 = 0 
            #extent2 = (box_x[1]-box_x[0])*self.pixel_size_arcsec 
            extent2 = len(mapa[0])*self.pixel_size_arcsec 
            extent3 = 0                       
            #extent4 = (box_y[1]-box_y[0])*self.pixel_size_arcsec 
            extent4 = len(mapa)*self.pixel_size_arcsec 
            alpha_grid = 0
            plot_spaxel_grid = False
            plot_centre = False
            fig_size= fig_size*0.75 # Make it a bit smaller
            label_axes_fontsize = label_axes_fontsize *0.85
            colorbar_fontsize = colorbar_fontsize * 0.75
            title_fontsize = title_fontsize * 0.7
            if len(mapa[0]) * len(mapa) < 100 : fig_size = fig_size * 0.6
            aspect_ratio = len(mapa[0]) / len(mapa)
            xlabel='$\Delta$ RA [ spaxels ]'
            ylabel='$\Delta$ DEC [ spaxels ]'
        else:
            mapa = mapa_
            extent1 = (0.5-self.n_cols/2) * self.pixel_size_arcsec
            extent2 = (0.5+self.n_cols/2) * self.pixel_size_arcsec
            extent3 = (0.5-self.n_rows/2) * self.pixel_size_arcsec
            extent4 = (0.5+self.n_rows/2) * self.pixel_size_arcsec
            xlabel='$\Delta$ RA [ arcsec ]'
            ylabel='$\Delta$ DEC [ arcsec ]'
            # We want squared pixels for plotting
            try:
                aspect_ratio = self.combined_cube.n_cols/self.combined_cube.n_rows * 1.
            except Exception:
                aspect_ratio = self.n_cols/self.n_rows *1.
        if aspect_ratio > 1.5 : fig_size = fig_size *1.7 # Make figure bigger
                                
        if verbose: print("\n> Plotting map '"+description.replace("\n "," - ")+"' :")
        if verbose and trimmed: print("  Trimmed in x = [ {:} , {:} ]  ,  y = [ {:} , {:} ] ".format(box_x[0],box_x[1],box_y[0],box_y[1]))
        
        # Check fcal
        if fcal == False and np.nanmedian(self.flux_calibration) != 0: fcal = True
        
        # Check specific maps
        if velocity and cmap=="fuego": cmap="seismic" 
        if fwhm and cmap=="fuego": cmap="Spectral" 
        if ew and cmap=="fuego": cmap="CMRmap_r"
        if ratio and cmap=="fuego": cmap="gnuplot2" 
        
        if velocity or fwhm or ew or ratio or weight : 
            fcal=False
            if vmin == 1E-30 : vmin=np.nanpercentile(mapa,5)
            if vmax == 1E30 : vmax=np.nanpercentile(mapa,95)
            
        fig, ax = plt.subplots(figsize=(fig_size/aspect_ratio, fig_size))
                
        if log: 
            norm = colors.LogNorm()
        else:
            norm=colors.Normalize()
        
        if gamma != 0 : norm=colors.PowerNorm(gamma=gamma)   # Default = 0.25 = 1/4

        if vmin == 1E-30 : vmin = np.nanmin(mapa)
        if vmin <= 0 and log == True : 
            if verbose : print("  vmin is negative but log = True, using vmin = np.nanmin(np.abs())")
            vmin = np.nanmin(np.abs(mapa))+1E-30
            
        if vmax == 1E30  : vmax = np.nanmax(mapa)
                           
        pmapa=ax.imshow(mapa, origin='lower', interpolation='none', norm = norm, cmap=cmap,    # cax=ax.imshow
                      extent=(extent1, extent2, extent3, extent4)) #,    aspect="auto")         
        
        pmapa.set_clim(vmin=vmin) 
        pmapa.set_clim(vmax=vmax)

        if contours:
            CS=plt.contour(mapa, extent=(extent1, extent2, extent3, extent4))
            if clabel: plt.clabel(CS, inline=1, fontsize=c_fontsize)
        
        ax.set_title(description, fontsize=title_fontsize)  
        plt.tick_params(labelsize=axes_fontsize)
        plt.xlabel(xlabel, fontsize=label_axes_fontsize)
        plt.ylabel(ylabel, fontsize=label_axes_fontsize)
        plt.legend(loc='upper right', frameon=False)
        plt.minorticks_on()
        plt.grid(which='both', color="white", alpha=alpha_grid)
        
        
        # IMPORTANT:
        #             If drawing INTEGER SPAXELS, use -ox, -oy
        #             If not, just use -self.spaxel_RA0, -self.spaxel_DEC0 or -ox+0.5, -oy+0.5
        #if np.nanmedian(box_x+box_y) != -0.5 and plot_box == False:
        if trimmed:       
            ox = 0
            oy = 0
            spaxel=0
            spaxel2=0
        else:
            ox = self.spaxel_RA0+0.5    
            oy = self.spaxel_DEC0+0.5   

        if verbose: 
            if fcal: 
                print("  Color scale range : [ {:.2e} , {:.2e}]".format(vmin,vmax))
            else:
                print("  Color scale range : [ {} , {}]".format(vmin,vmax))

        if plot_centre:
            if verbose: print("  - The center of the cube is in position [",self.spaxel_RA0,",", self.spaxel_DEC0,"]")  
            if self.n_cols % 2 == 0:
                extra_x = 0
            else:
                extra_x = 0.5*self.pixel_size_arcsec
            if self.n_rows % 2 == 0:
                extra_y = 0
            else:
                extra_y = 0.5*self.pixel_size_arcsec
                
            plt.plot([0.+extra_x-self.pixel_size_arcsec*0.00],[0.+extra_y-self.pixel_size_arcsec*0.02],"+",ms=14,color="black", mew=4)
            plt.plot([0+extra_x],[0+extra_y],"+",ms=10,color="white", mew=2)

        if plot_spaxel_grid:
            for i in range(self.n_cols):
                spaxel_position = [i-ox,0-oy]
                if i % 2 == 0 :
                    color = "white" 
                else: color = "black"
                cuadrado = plt.Rectangle((spaxel_position[0]*self.pixel_size_arcsec, (spaxel_position[1]+0)*self.pixel_size_arcsec), self.pixel_size_arcsec,self.pixel_size_arcsec,color=color, linewidth=0, fill=True)
                ax.add_patch(cuadrado)

            for i in range(self.n_rows):
                spaxel_position = [0-ox,i-oy]
                if i % 2 == 0 :
                    color = "white" 
                else: color = "black"
                cuadrado = plt.Rectangle((spaxel_position[0]*self.pixel_size_arcsec, (spaxel_position[1]+0)*self.pixel_size_arcsec), self.pixel_size_arcsec,self.pixel_size_arcsec,color=color, linewidth=0, fill=True)
                ax.add_patch(cuadrado)

        if plot_spaxel:  
            spaxel_position = [5-ox, 5-oy]
            cuadrado = plt.Rectangle((spaxel_position[0]*self.pixel_size_arcsec, (spaxel_position[1]+0)*self.pixel_size_arcsec), self.pixel_size_arcsec,self.pixel_size_arcsec,color=color, linewidth=0, fill=True, alpha=0.8)
            ax.add_patch(cuadrado)           
            #vertices_x =[spaxel_position[0]*self.pixel_size_arcsec,spaxel_position[0]*self.pixel_size_arcsec,(spaxel_position[0]+1)*self.pixel_size_arcsec,(spaxel_position[0]+1)*self.pixel_size_arcsec,spaxel_position[0]*self.pixel_size_arcsec]
            #vertices_y =[spaxel_position[1]*self.pixel_size_arcsec,(1+spaxel_position[1])*self.pixel_size_arcsec,(1+spaxel_position[1])*self.pixel_size_arcsec,spaxel_position[1]*self.pixel_size_arcsec,spaxel_position[1]*self.pixel_size_arcsec]          
            #plt.plot(vertices_x,vertices_y, color=color, linewidth=1., alpha=1)
            plt.plot([(spaxel_position[0]+0.5)*self.pixel_size_arcsec],[(spaxel_position[1]+.5)*self.pixel_size_arcsec], 'o', color="black", ms=1,  alpha=1)

        if len(plot_spaxel_list) > 0 :  
            for _spaxel_ in plot_spaxel_list:    # If they are INTEGERS we identify the SPAXEL putting the cross IN THE CENTRE of the SPAXEL
                if isinstance(_spaxel_[0], int) and isinstance(_spaxel_[1], int) :
                    extra = 0.5
                else:
                    extra = 0.       
                plt.plot((_spaxel_[0]-ox+extra)*self.pixel_size_arcsec,(_spaxel_[1]-oy+extra)*self.pixel_size_arcsec, color="black", marker="+", ms=15, mew=2)

        
        if np.nanmedian(circle) != 0:    # The center of the circle is given with decimals in spaxels
                                         # The radius is in arcsec, but for plt.Circle all in arcsec
            offset_from_center_x_arcsec =  (circle[0]-self.spaxel_RA0)*self.pixel_size_arcsec         
            offset_from_center_y_arcsec =  (circle[1]-self.spaxel_DEC0)*self.pixel_size_arcsec 
            circle_p = plt.Circle((offset_from_center_x_arcsec, offset_from_center_y_arcsec), 
                                  circle[2], color='b', linewidth=3, fill=False)
            ax.add_patch(circle_p)
            if verbose: print('  - Blue  circle:   [{:.2f}, {:.2f}] , Offset from center :   {:.2f}" , {:.2f}", radius = {:.2f}"'.format(circle[0], circle[1] ,offset_from_center_x_arcsec,offset_from_center_y_arcsec, circle[2]))
          
        if np.nanmedian(circle2) != 0:
            offset_from_center_x_arcsec =  (circle2[0]-self.spaxel_RA0)*self.pixel_size_arcsec         
            offset_from_center_y_arcsec =  (circle2[1]-self.spaxel_DEC0)*self.pixel_size_arcsec 
            circle_p = plt.Circle((offset_from_center_x_arcsec, offset_from_center_y_arcsec), 
                                  circle2[2], color='w', linewidth=4, fill=False, alpha=0.3)
            ax.add_patch(circle_p)
            
        if np.nanmedian(circle3) != 0:
            offset_from_center_x_arcsec =  (circle3[0]-self.spaxel_RA0)*self.pixel_size_arcsec         
            offset_from_center_y_arcsec =  (circle3[1]-self.spaxel_DEC0)*self.pixel_size_arcsec 
            circle_p = plt.Circle((offset_from_center_x_arcsec, offset_from_center_y_arcsec), 
                                  circle3[2], color='w', linewidth=4, fill=False, alpha=0.3)
            ax.add_patch(circle_p)
                   
        if spaxel != 0:   # SPAXEL 
            if isinstance(spaxel[0], int) and isinstance(spaxel[1], int) :
                extra = 0.5     # If they are INTEGERS we identify the SPAXEL putting the cross IN THE CENTRE of the SPAXEL
            else:
                extra = 0.  
        
            offset_from_center_x_arcsec =  (spaxel[0]-self.spaxel_RA0  +extra)*self.pixel_size_arcsec         
            offset_from_center_y_arcsec =  (spaxel[1]-self.spaxel_DEC0 +extra)*self.pixel_size_arcsec         
            if verbose: print('  - Blue  square:   {}          , Offset from center :   {:.2f}" , {:.2f}"'.format(np.round(spaxel,2),offset_from_center_x_arcsec, offset_from_center_y_arcsec))
            cuadrado = plt.Rectangle(( (spaxel[0]-ox)*self.pixel_size_arcsec,
                                        (spaxel[1]-oy)*self.pixel_size_arcsec),
                                        self.pixel_size_arcsec,self.pixel_size_arcsec,color="blue", linewidth=0, fill=True, alpha=1)
            ax.add_patch(cuadrado) 
            #plt.plot((spaxel[0]-ox)*self.pixel_size_arcsec,(spaxel[0]-ox)*self.pixel_size_arcsec, color="blue", marker="s", ms=15, mew=2)


        if spaxel2 != 0:  # 
            offset_from_center_x_arcsec =  (spaxel2[0]-self.spaxel_RA0)*self.pixel_size_arcsec         
            offset_from_center_y_arcsec =  (spaxel2[1]-self.spaxel_DEC0)*self.pixel_size_arcsec         
            if verbose: print('  - Green circle:   {}    , Offset from center :   {:.2f}" , {:.2f}"'.format(np.round(spaxel2,2),offset_from_center_x_arcsec,offset_from_center_y_arcsec))
            plt.plot([offset_from_center_x_arcsec],[offset_from_center_y_arcsec], 'o', color="green", ms=7)

        if spaxel3 != 0:  # SPAXEL
            offset_from_center_x_arcsec =  (spaxel3[0]-self.spaxel_RA0)*self.pixel_size_arcsec         
            offset_from_center_y_arcsec =  (spaxel3[1]-self.spaxel_DEC0)*self.pixel_size_arcsec         
            if verbose: print('  - Red   square:   {}     , Offset from center :   {:.2f}" , {:.2f}"'.format(np.round(spaxel3,2),offset_from_center_x_arcsec, offset_from_center_y_arcsec))
            cuadrado = plt.Rectangle(( (spaxel3[0]-ox)*self.pixel_size_arcsec,
                                       (spaxel3[1]-oy)*self.pixel_size_arcsec),
                                        self.pixel_size_arcsec,self.pixel_size_arcsec,color="red", linewidth=0, fill=True, alpha=1)
            ax.add_patch(cuadrado) 

 
        if plot_centroid:
            # If box defined, compute centroid there
            if np.nanmedian(box_x+box_y) != -0.5 and plot_box:
                _mapa_ = mapa[box_y[0]:box_y[1],box_x[0]:box_x[1]]
            else:
                _mapa_ = mapa
                  
            if g2d:
                xc, yc = centroid_2dg(_mapa_)
            else:
                xc, yc = centroid_com(_mapa_)
            
            # if box values not given, both box_x[0] and box_y[0] are 0
            offset_from_center_x_arcsec =  (xc+box_x[0]-self.spaxel_RA0)*self.pixel_size_arcsec         
            offset_from_center_y_arcsec =  (yc+box_y[0]-self.spaxel_DEC0)*self.pixel_size_arcsec 
            
            plt.plot((xc+box_x[0]-ox)*self.pixel_size_arcsec,(yc+box_y[0]-oy)*self.pixel_size_arcsec, color="black", marker="+", ms=15, mew=2)
    
            if verbose: 
                if trimmed:
                    if line != 0:
                        print('  Centroid at {} A found in spaxel [ {:.2f} , {:.2f} ]  =  [ {:.2f}" , {:.2f}" ]'.format(line, xc,yc,xc*self.pixel_size_arcsec,yc*self.pixel_size_arcsec))            
                    else:    
                        print('  - Centroid:       [{:.2f} {:.2f}]    , Offset from center :   {:.2f}" , {:.2f}"'.format(xc+box_x[0],yc+box_y[0], offset_from_center_x_arcsec, offset_from_center_y_arcsec))            
                else:
                    if line != 0:
                        print('  Centroid at {} A found in spaxel [ {:.2f} , {:.2f} ]  =  [ {:.2f}" , {:.2f}" ]'.format(line, xc,yc,xc*self.pixel_size_arcsec,yc*self.pixel_size_arcsec))            
                    else:
                        print('  - Centroid (box): [{:.2f} {:.2f}]    , Offset from center :   {:.2f}" , {:.2f}"'.format(xc+box_x[0],yc+box_y[0], offset_from_center_x_arcsec, offset_from_center_y_arcsec))            

                  
        # Plox box if requested
        if np.nanmedian(box_x+box_y) != -0.5 and plot_box:  
            box_x=[box_x[0]-ox,box_x[1]-ox]
            box_y=[box_y[0]-oy,box_y[1]-oy]    
            vertices_x =[box_x[0]*self.pixel_size_arcsec,box_x[0]*self.pixel_size_arcsec,box_x[1]*self.pixel_size_arcsec,box_x[1]*self.pixel_size_arcsec,box_x[0]*self.pixel_size_arcsec]
            vertices_y =[box_y[0]*self.pixel_size_arcsec,box_y[1]*self.pixel_size_arcsec,box_y[1]*self.pixel_size_arcsec,box_y[0]*self.pixel_size_arcsec,box_y[0]*self.pixel_size_arcsec]          
            plt.plot(vertices_x,vertices_y, "-b", linewidth=2., alpha=0.6)

        # Plot color bar
        if 0.8  <= aspect_ratio <= 1.2:
            bth = 0.05
            gap = 0.03
        else:
            bth = 0.03
            gap = 0.02

        if fraction is None:
            fraction = np.max((aspect_ratio * bth, bth))
        if pad is None:
            pad = np.max((aspect_ratio * gap, gap))

        cax  = ax.inset_axes((1+pad, 0, fraction , 1))
        cbar = fig.colorbar(pmapa,cax=cax) 
        cbar.ax.tick_params(labelsize=colorbar_ticksize)
        
        if barlabel == "" :
            if velocity:
                barlabel = str("Velocity [ km s$^{-1}$ ]")
            elif fwhm:
                barlabel = str("FWHM [ km s$^{-1}$ ]")
            elif ew:
                barlabel = str("EW [ $\mathrm{\AA}$ ]" )
            else:
                if fcal: 
                    barlabel = str("Integrated Flux [ erg s$^{-1}$ cm$^{-2}$ ]")
                else:    
                    barlabel = str("Integrated Flux [ Arbitrary units ]")
        cbar.set_label(barlabel, rotation=270, labelpad=20, fontsize=colorbar_fontsize)
#        cbar.ax.set_yticklabels(['< -1', '0', '> 1'])# vertically oriented colorbar
        
        # Save plot in figure if requested
        if save_file == "":
            plt.show()
        else:
            print("  Plot saved to file ",save_file)
            plt.savefig(save_file)
        plt.close()
# -----------------------------------------------------------------------------        
# -----------------------------------------------------------------------------
    def mask_cube(self, min_wave = 0, max_wave = 0, include_partial_spectra= False,
                  cmap="binary_r", plot=False, verbose = False):
        """
        #TODO

        Parameters
        ----------
        min_wave : Integer, optional
            The minimum wavelength passed through the mask. The default is 0.
        max_wave : Integer, optional
            The maximum wavelength passed through the mask. The default is 0.
        include_partial_spectra : Boolean, optional
            DESCRIPTION. The default is False. #TODO
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
        
        if min_wave == 0 : min_wave = self.valid_wave_min
        if max_wave == 0 : max_wave = self.valid_wave_max
        
        if include_partial_spectra:  # IT DOES NOT WORK
            if verbose: print("\n> Creating cube mask considering ALL spaxels with some spectrum...")
            self.integrated_map = np.nansum(self.data[np.searchsorted(self.wavelength, min_wave):np.searchsorted(self.wavelength, max_wave)],axis=0)            
        else:
            if verbose: print("\n> Creating cube mask considering ONLY those spaxels with valid full spectrum in range {:.2f} - {:.2f} ...".format(min_wave,max_wave))
            # Get integrated map but ONLY consering spaxels for which all wavelengths are good (i.e. NOT using nanmen)
            self.integrated_map = np.sum(self.data[np.searchsorted(self.wavelength, min_wave):np.searchsorted(self.wavelength, max_wave)],axis=0)            
            
    
        # Create a mask with the same structura full of 1
        self.mask = np.ones_like(self.integrated_map)
        for y in range(self.n_rows):
            for x in range(self.n_cols):
                if np.isnan(self.integrated_map[y][x]):
                    self.mask[y][x] = np.nan    # If integrated map is nan, the mask is nan
        # Plot mask
        if plot:
            description = self.description+" - mask of good spaxels"
            self.plot_map(mapa=self.mask, cmap=cmap, description =description)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def growth_curve_between(self, min_wave=0, max_wave=0, 
                             sky_annulus_low_arcsec = 7., sky_annulus_high_arcsec = 12.,
                             plot=False, verbose=False, log = True):  
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
        if min_wave == 0 : min_wave = self.valid_wave_min 
        if max_wave == 0 : max_wave = self.valid_wave_max
        
        index_min = np.searchsorted(self.wavelength, min_wave)
        index_max = np.searchsorted(self.wavelength, max_wave)
        
        intensity = np.nanmean(self.data[index_min:index_max, :, :], axis=0)
        x_peak = np.nanmedian(self.x_peaks[index_min:index_max])
        y_peak = np.nanmedian(self.y_peaks[index_min:index_max])
        
        if verbose: 
            print("  - Peak found at spaxel position {:.3f} , {:.3f} ".format (x_peak,y_peak)) 
            print("  - Calculating growth curve between ",np.round(min_wave,2)," and ", np.round(max_wave,2),"...")
                
        x = np.arange(self.n_cols) - x_peak
        y = np.arange(self.n_rows) - y_peak
        r2 = np.sum(np.meshgrid(x**2, y**2), axis=0)
        r  = np.sqrt(r2)
        sorted_by_distance = np.argsort(r, axis=None)

        F_curve=[]
        F_growth_curve = []
        r2_growth_curve = []
        r_growth_curve = []
        sky_flux_r = []
        total_flux = 0.
        sky_flux = 0.
        F_total_star = 0
        spaxels_star=0
        spaxels_sky =0
              
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
                        if sky_flux == 0 : 
                            F_total_star = total_flux
                        sky_flux = sky_flux + I
                        sky_flux_r.append(sky_flux)
                        spaxels_sky = spaxels_sky +1
                        
                else:
                    sky_flux_r.append(0)
                    spaxels_star = spaxels_star+1                
        
        # IMPORTANT !!!! WE MUST SUBSTRACT THE RESIDUAL SKY !!!       
        #F_guess = F_total_star # np.max(F_growth_curve)   
        
        sky_per_spaxel = sky_flux/spaxels_sky
        sky_in_star = spaxels_star * sky_per_spaxel
        
        if verbose: 
            print("  Valid spaxels in star = {}, valid spaxels in sky = {}".format(spaxels_star,spaxels_sky))
            print("  Sky value per spaxel = ",np.round(sky_per_spaxel,3))
            print("  We have to sustract {:.2f} to the total flux of the star, which is its {:.3f} % ".format(sky_in_star, sky_in_star/F_total_star*100.))
        
        #r2_half_light = np.interp(.5*F_guess, F_growth_curve, r2_growth_curve)
                
        F_growth_star=np.ones_like(F_growth_curve) * (F_total_star-sky_in_star)
        
        for i in range(0,spaxels_star):
            F_growth_star[i] = F_growth_curve[i] - sky_per_spaxel * (i+1)            
            #if verbose: print  i+1, F_growth_curve[i], sky_per_spaxel * (i+1), F_growth_star[i]        
        
        r_half_light = np.interp(.5*(F_total_star-sky_in_star), F_growth_star, r_growth_curve)        
        F_guess = F_total_star-sky_in_star        
        self.seeing = 2*r_half_light*self.pixel_size_arcsec
        #print "  Between {} and {} the seeing is {} and F_total_star = {} ".format(min_wave,max_wave,self.seeing,F_total_star) 
        
        if plot:
            self.plot_map(circle=[x_peak,y_peak, self.seeing/2.], 
                          circle2=[x_peak,y_peak,sky_annulus_low_arcsec],
                          circle3=[x_peak,y_peak,sky_annulus_high_arcsec],
                          contours=False,
                          #spaxel2=[x_peak,y_peak], 
                          verbose=True, plot_centre=False, log=log)

        r_norm = r_growth_curve / r_half_light
        r_arcsec = np.array(r_growth_curve) * self.pixel_size_arcsec 

        F_norm = np.array(F_growth_curve) / F_guess     
        sky_norm = np.array(sky_flux_r) / F_guess 
        F_star_norm = F_growth_star /F_guess
        
        if verbose:
            print("      Flux guess =", F_guess," ~ ", np.nanmax(F_growth_star), " ratio = ", np.nanmax(F_growth_star)/F_guess)
            print("      Half-light radius:", np.round(r_half_light*self.pixel_size_arcsec,3), " arcsec  -> seeing = ",np.round(self.seeing,3)," arcsec, if object is a star ")
            print("      Light within 2, 3, 4, 5 half-light radii:", np.interp([2, 3, 4, 5], r_norm, F_norm))
        if plot:
            plt.figure(figsize=(10, 8))
            plt.plot(r_arcsec,F_norm, 'b-')
            plt.plot(r_arcsec, sky_norm, 'r-')
            #plt.plot(r_arcsec, F_norm-sky_norm, 'g-', linewidth=10, alpha = 0.6)
            plt.plot(r_arcsec, F_star_norm, 'g-',linewidth=10, alpha = 0.5)
            plt.title("Growth curve between "+str(np.round(min_wave,2))+" and "+str(np.round(max_wave,2))+" in "+self.object)
            plt.xlabel("Radius [arcsec]")
            plt.ylabel("Amount of integrated flux")
            plt.xlim(0,sky_annulus_high_arcsec+1)
            plt.axhline(y=0.5, color='k', linestyle=':', alpha=0.5)
            plt.axvline(x=self.seeing/2, color='g', alpha=0.7)
            plt.axvline(x=self.seeing, color='k', linestyle=':', alpha=0.2)
            plt.axvline(x=3*self.seeing/2, color='k', linestyle=':', alpha=0.2)
            plt.axvline(x=4*self.seeing/2, color='k', linestyle=':', alpha=0.2)
            plt.axvline(x=5*self.seeing/2, color='r', linestyle='-', alpha=0.5)            
            plt.axvspan(sky_annulus_low_arcsec, sky_annulus_high_arcsec, facecolor='r', alpha=0.15,zorder=3)            
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.2)
            plt.minorticks_on()
            plt.show()
            plt.close()

        #return r2_growth_curve, F_growth_curve, F_guess, r2_half_light
        return r2_growth_curve, F_growth_star, F_guess, r_half_light
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def half_light_spectrum(self, r_max=1, smooth=21, min_wave=0, max_wave=0, 
                            sky_annulus_low_arcsec = 5., sky_annulus_high_arcsec = 10.,
                            fig_size=12, plot=True, verbose = True):
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
            DESCRIPTION. The default is 5.. #TODO
        sky_annulus_high_arcsec : Float, optional
            DESCRIPTION. The default is 10.. #TODO
        fig_size : Integer, optional
            This is the size of the figure. The default is 12.
        plot : Boolean, optional
            If True generates and shows the plots. The default is True.
        verbose : Boolean, optional
            Print results. The default is True.

        Returns
        -------
        Numpy Array
            This is a Numpy Array of the intensities.
        """        

        if min_wave == 0 : min_wave = self.valid_wave_min 
        if max_wave == 0 : max_wave = self.valid_wave_max 

        if verbose: 
            print('\n> Obtaining the integrated spectrum of star between {:.2f} and {:.2f} and radius {} r_half_light'.format(min_wave,max_wave,r_max))
            print('  Considering the sky in an annulus between {}" and {}"'.format(sky_annulus_low_arcsec,sky_annulus_high_arcsec))
        
        r2_growth_curve, F_growth_curve, flux, r2_half_light = self.growth_curve_between(min_wave, max_wave, 
                                                                                         sky_annulus_low_arcsec = sky_annulus_low_arcsec, 
                                                                                         sky_annulus_high_arcsec = sky_annulus_high_arcsec,
                                                                                         plot=plot, verbose=verbose)    # 0,1E30 ??
        
        intensity = []
        smooth_x = signal.medfilt(self.x_peaks, smooth)     # originally, smooth = 11
        smooth_y = signal.medfilt(self.y_peaks, smooth)
        edgelow= (np.abs(self.wavelength-min_wave)).argmin() 
        edgehigh= (np.abs(self.wavelength-max_wave)).argmin()
        valid_wl = self.wavelength[edgelow:edgehigh]        
            
        for l in range(self.n_wave):    
            x = np.arange(self.n_cols) - smooth_x[l]
            y = np.arange(self.n_rows) - smooth_y[l]
            r2 = np.sum(np.meshgrid(x**2, y**2), axis=0)
            spaxels = np.where(r2 < r2_half_light*r_max**2)
            intensity.append(np.nansum(self.data[l][spaxels]))
                       
        valid_intensity = intensity[edgelow:edgehigh]
        valid_wl_smooth =    signal.medfilt(valid_wl,smooth)
        valid_intensity_smooth = signal.medfilt(valid_intensity,smooth)
                
        if plot:
            plt.figure(figsize=(fig_size, fig_size/2.5))
            plt.plot(self.wavelength, intensity, 'b', alpha=1, label='Intensity')
            plt.plot(valid_wl_smooth,valid_intensity_smooth, 'r-', alpha=0.5,label='Smooth = '+str(smooth))
            margen = 0.1*(np.nanmax(intensity) - np.nanmin(intensity))
            plt.ylim(np.nanmin(intensity)-margen, np.nanmax(intensity) + margen  )
            plt.xlim(np.min(self.wavelength), np.max(self.wavelength))

            plt.ylabel("Flux")
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            plt.title("Integrated spectrum of "+self.object+" for r_max = "+str(r_max) + "r_half_light")
            plt.axvline(x=min_wave, color='k', linestyle='--', alpha=0.5)
            plt.axvline(x=max_wave, color='k', linestyle='--', alpha=0.5)
            plt.minorticks_on()
            plt.legend(frameon=False, loc=1)
            plt.show()
            plt.close()
        if r_max == 5 : 
            print("  Saving this integrated star flux in self.integrated_star_flux")
            self.integrated_star_flux =   np.array(intensity)  
        return np.array(intensity)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def do_response_curve(self, absolute_flux_file, min_wave_flux=0, max_wave_flux=0, 
                          fit_degree_flux=3, step_flux=25., r_max = 5, exp_time=0, 
                          ha_width=0, after_telluric_correction=False, 
                          sky_annulus_low_arcsec = 5., sky_annulus_high_arcsec = 10.,
                          odd_number=0 , fit_weight=0., smooth_weight=0., smooth=0., 
                          exclude_wlm=[[0,0]],
                          plot=True, verbose= False): 
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
            DESCRIPTION. The default is 0. #TODO
        max_wave_flux : Integer, optional
            DESCRIPTION. The default is 0. #TODO
        fit_degree_flux : Integer, optional
            DESCRIPTION. The default is 3. #TODO
        step_flux : Float, optional
            DESCRIPTION. The default is 25.. #TODO
        r_max : Integer, optional
            r_max to integrate, in units of r2_half_light (= seeing if object is a star, for flux calibration make r_max=5). The default is 5.
        exp_time : Integer, optional
            DESCRIPTION. The default is 0. #TODO
        ha_width : Integer, optional
            DESCRIPTION. The default is 0. #TODO
        after_telluric_correction : Boolean, optional
            DESCRIPTION. The default is False. #TODO
        sky_annulus_low_arcsec : Float, optional
            DESCRIPTION. The default is 5.. #TODO
        sky_annulus_high_arcsec : Float, optional
            DESCRIPTION. The default is 10.. #TODO
        odd_number : Integer, optional
            DESCRIPTION. The default is 0. #TODO
        fit_weight : Float, optional
            DESCRIPTION. The default is 0.. #TODO
        smooth_weight : Float, optional
            DESCRIPTION. The default is 0.. #TODO
        smooth : Float, optional
            smooths the data. The default is 0..
        exclude_wlm : List of Lists of Integers, optional
            DESCRIPTION. The default is [[0,0]]. #TODO
        plot : Boolean, optional
            If True generates and shows the plots. The default is True.
        verbose : Boolean, optional
            Print results. The default is False.

        Returns
        -------
        None.

        """   
        if smooth == 0.0 : smooth=0.05
        
        if min_wave_flux == 0 : min_wave_flux = self.valid_wave_min + step_flux 
        if max_wave_flux == 0 : max_wave_flux = self.valid_wave_max - step_flux
        #valid_wave_min=min_wave 
        #valid_wave_max=max_wave 
                        
        print("\n> Computing response curve for",self.object,"using step=",step_flux,"A in range [",np.round(min_wave_flux,2),",",np.round(max_wave_flux,2),"] ...")

        if exp_time == 0:
            try:
                exp_time = np.nanmedian(self.exptimes)
                print("  Exposition time from the median value of self.exptimes =",exp_time)
            except Exception:    
                print("  Exposition time is not given, and failed to read it in object, assuming exp_time = 60 seconds...")
                exp_time = 60.
        else:
            print("  Exposition time provided =",exp_time,"s")

#        flux_cal_read in units of ergs/cm/cm/s/A * 10**16
#        lambda_cal_read, flux_cal_read, delta_lambda_read = np.loadtxt(filename, usecols=(0,1,3), unpack=True)
        lambda_cal_read, flux_cal_read = np.loadtxt(absolute_flux_file, usecols=(0,1), unpack=True)

        valid_wl_smooth = np.arange(lambda_cal_read[0], lambda_cal_read[-1], step_flux)         
        tck_star = interpolate.splrep(lambda_cal_read, flux_cal_read, s=0)
        valid_flux_smooth = interpolate.splev(valid_wl_smooth, tck_star, der=0)
    
        edgelow= (np.abs(valid_wl_smooth-min_wave_flux)).argmin() 
        edgehigh= (np.abs(valid_wl_smooth-max_wave_flux)).argmin()
        
        lambda_cal = valid_wl_smooth[edgelow:edgehigh]            
        flux_cal = valid_flux_smooth[edgelow:edgehigh]
        lambda_min = lambda_cal - step_flux
        lambda_max = lambda_cal + step_flux
        
        if self.flux_cal_step == step_flux and self.flux_cal_min_wave == min_wave_flux and self.flux_cal_max_wave == max_wave_flux and after_telluric_correction == False:
            print("  This has been already computed for step = {} A in range [ {:.2f} , {:.2f} ] , using existing values ...".format(step_flux,min_wave_flux,max_wave_flux))
            measured_counts = self.flux_cal_measured_counts
        else:
            self.integrated_star_flux = self.half_light_spectrum(r_max, plot=plot, min_wave=min_wave_flux, max_wave=max_wave_flux, 
                                                                 sky_annulus_low_arcsec = sky_annulus_low_arcsec, 
                                                                 sky_annulus_high_arcsec = sky_annulus_high_arcsec)
            
            print("  Obtaining fluxes using step = {} A in range [ {:.2f} , {:.2f} ] ...".format(step_flux,min_wave_flux,max_wave_flux))

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

    
        _response_curve_ = measured_counts / flux_cal   / exp_time     # Added exp_time Jan 2019       counts / (ergs/cm/cm/s/A * 10**16) / s  =>    10-16 erg/s/cm2/A =   counts /s
        
        if np.isnan(_response_curve_[0]) == True  :
            _response_curve_[0] = _response_curve_[1] # - (response_curve[2] - response_curve[1])
        
        scale = np.nanmedian(_response_curve_)
                 
        response_wavelength =[]
        response_curve =[]

        ha_range=[6563.-ha_width/2.,6563.+ha_width/2.]
        # Skip bad ranges and Ha line
        if ha_width > 0: 
            print("  Skipping H-alpha absorption with width = {} A, adding [ {:.2f} , {:.2f} ] to list of ranges to skip...".format(ha_width,ha_range[0],ha_range[1]))
            if exclude_wlm[0][0] == 0:
                _exclude_wlm_ = [ha_range]
            else:           
                #_exclude_wlm_ = exclude_wlm.append(ha_range)
                _exclude_wlm_ = []
                ha_added=False
                for rango in exclude_wlm:
                    if rango[0] < ha_range[0] and rango[1] < ha_range[0] : # Rango is BEFORE Ha
                        _exclude_wlm_.append(rango)
                    if rango[0] < ha_range[0] and rango[1] > ha_range[0] : # Rango starts BEFORE Ha but finishes WITHIN Ha
                        _exclude_wlm_.append([rango[0],ha_range[1]])
                    if rango[0] > ha_range[0] and rango[1] < ha_range[1] : # Rango within Ha
                        _exclude_wlm_.append(ha_range)
                    if rango[0] > ha_range[0] and rango[0] < ha_range[1] : # Rango starts within Ha but finishes after Ha
                        _exclude_wlm_.append([ha_range[0],rango[1]])    
                    if rango[0] > ha_range[1] and rango[1] > ha_range[1] : # Rango is AFTER Ha, add both if needed
                        if ha_added == False: 
                            _exclude_wlm_.append(ha_range)
                            ha_added= True
                        _exclude_wlm_.append(rango)                                                             
        else:
            _exclude_wlm_ = exclude_wlm
        
        if _exclude_wlm_[0][0] == 0:  # There is not any bad range to skip
            response_wavelength =lambda_cal
            response_curve = _response_curve_  
            if verbose: print ("  No ranges will be skipped")
        else:
            if verbose: print ("  List of ranges to skip : ",_exclude_wlm_)
            skipping = 0
            rango_index = 0
            for i in range(len(lambda_cal)-1):
                if rango_index < len(_exclude_wlm_) :
                    if lambda_cal[i] >= _exclude_wlm_[rango_index][0]  and lambda_cal[i] <= _exclude_wlm_[rango_index][1]:
                        #print(" checking ", lambda_cal[i], rango_index, _exclude_wlm_[rango_index][0], _exclude_wlm_[rango_index][1]) 
                        skipping = skipping+1
                        if lambda_cal[i+1] > _exclude_wlm_[rango_index][1] : # If next value is out of range, change range_index
                            rango_index =rango_index +1
                            #print(" changing to  range ",rango_index)
                    else:    
                        response_wavelength.append(lambda_cal[i])
                        response_curve.append(_response_curve_[i])
                else:    
                    response_wavelength.append(lambda_cal[i])
                    response_curve.append(_response_curve_[i])
    
            response_wavelength.append(lambda_cal[-1])
            response_curve.append(_response_curve_[-1])
            print("  Skipping a total of ",skipping,"wavelength points for skipping bad ranges") 

        if plot:           
            plt.figure(figsize=(12, 8))
            plt.plot(lambda_cal, measured_counts/exp_time, 'g+', ms=10, mew=3, label="measured counts")
            plt.plot(lambda_cal, flux_cal*scale, 'k*-',label="flux_cal * scale")
            plt.plot(lambda_cal, flux_cal*_response_curve_, 'r-', label="flux_cal * response")
            plt.xlim(self.wavelength[0]-50, self.wavelength[-1]+50)    
            plt.axvline(x=self.wavelength[0], color='k', linestyle='-', alpha=0.7)
            plt.axvline(x=self.wavelength[-1], color='k', linestyle='-', alpha=0.7)
            plt.ylabel("Flux [counts]")
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            plt.title("Response curve for absolute flux calibration using "+self.object)
            plt.legend(frameon=True, loc=1)
            plt.grid(which='both')
            if ha_width > 0 : plt.axvspan(ha_range[0], ha_range[1], facecolor='orange', alpha=0.15, zorder=3) 
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
        if odd_number == 0 : odd_number = smoothfactor * int(np.sqrt(len(response_wavelength )) / 2) - 1
        response_curve_medfilt_ =medfilt(response_curve,np.int(odd_number))
        interpolated_curve = interpolate.splrep(response_wavelength, response_curve_medfilt_, s=smooth)
        response_curve_smoothed = interpolate.splev(self.wavelength, interpolated_curve, der=0)  

        # Obtaining the fit
        fit = np.polyfit(response_wavelength, response_curve, fit_degree_flux)
        pp=np.poly1d(fit)
        response_curve_fitted=pp(self.wavelength)

        # Obtaining the fit using a  smoothed response curve
        # Adapting Matt code for trace peak ----------------------------------        
        smoothfactor = 2
        wl = response_wavelength 
        x = response_curve
        if odd_number == 0 : odd_number = smoothfactor * int(np.sqrt(len(wl)) / 2) - 1  # Originarily, smoothfactor = 2
        print("  Obtaining smoothed response curve using medfilt window = {} for fitting a {}-order polynomium...".format(odd_number, fit_degree_flux))

        wlm = signal.medfilt(wl, odd_number)
        wx = signal.medfilt(x, odd_number)
        
        #iteratively clip and refit for WX
        maxit=10
        niter=0
        stop=0
        fit_len=100# -100
        resid = 0
        while stop < 1:
            fit_len_init=copy.deepcopy(fit_len)
            if niter == 0:
                fit_index=np.where(wx == wx)
                fit_len=len(fit_index)
                sigma_resid=0.0
            if niter > 0:
                sigma_resid=MAD(resid)
                fit_index=np.where(np.abs(resid) < 4*sigma_resid)[0]
                fit_len=len(fit_index)
            try:
                p=np.polyfit(wlm[fit_index], wx[fit_index], fit_degree_flux)
                pp=np.poly1d(p)
                fx=pp(wl)
                fxm=pp(wlm)
                resid=wx-fxm
                #print "  Iteration {:2} results in RA: sigma_residual = {:.6f}, fit_len = {:5}  fit_len ={:5}".format(niter,sigma_resid,fit_len_init,fit_len)             
            except Exception:  
                print('  Skipping iteration ',niter)
            if (niter >= maxit) or (fit_len_init == fit_len): 
                if niter >= maxit : print("  Max iterations, {:2}, reached!")
                if fit_len_init == fit_len : print("  All interval fitted in iteration {:2} ! ".format(niter))
                stop=2     
            niter=niter+1      
        # --------------------------------------------------------------------
        interpolated_curve = interpolate.splrep(response_wavelength, fx) 
        response_curve_interpolated = interpolate.splev(self.wavelength, interpolated_curve, der=0) 
        
        # Choose solution:           
        if fit_degree_flux == 0:
            print ("\n> Using smoothed response curve with medfilt window =",np.str(odd_number),"and s =",np.str(smooth),"as solution for the response curve")
            self.response_curve = response_curve_smoothed
        else:
            if fit_weight == 0 and smooth_weight == 0:
                print ("\n> Using fit of a {}-order polynomium as solution for the response curve".format(fit_degree_flux))                
                self.response_curve = response_curve_fitted
            else:
                if smooth_weight == 0:
                    print ("\n> Using a combination of the fitted (weight = {:.2f}) and smoothed fitted (weight = {:.2f}) response curves".format(fit_weight,1-fit_weight)) 
                    self.response_curve = response_curve_fitted * fit_weight + response_curve_interpolated   * (1-fit_weight)         
                else:
                    fit_smooth_weight = 1 - smooth_weight -fit_weight
                    if fit_smooth_weight <= 0:
                        print ("\n> Using a combination of the fitted (weight = {:.2f}) and smoothed (weight = {:.2f}) response curves".format(fit_weight,smooth_weight)) 
                        self.response_curve = response_curve_fitted * fit_weight + response_curve_smoothed   * smooth_weight        
                    else:                   
                        print ("\n> Using a combination of the fitted (weight = {:.2f}), smoothed fitted (weight = {:.2f}) and smoothed (weight = {:.2f}) response curves".format(fit_weight,fit_smooth_weight,smooth_weight)) 
                        self.response_curve = response_curve_fitted * fit_weight + response_curve_interpolated   * fit_smooth_weight  + response_curve_smoothed   * smooth_weight       
                    
    
        if plot: 
            plt.figure(figsize=(12, 8))       
            plt.plot(lambda_cal, _response_curve_, 'k--', alpha=0.7, label ="Raw response curve")
            plt.plot(response_wavelength, response_curve, 'k-', alpha=1., label ='  "  excluding bad ranges')
            plt.plot(self.wavelength, self.response_curve, "g-", alpha=0.4, linewidth=12, label="Obtained response curve")  
            text="Smoothed with medfilt window = "+np.str(odd_number)+" and s = "+np.str(smooth)
            plt.plot(self.wavelength, response_curve_smoothed, "-", color="orange", alpha=0.8,linewidth=2, label=text)
            if fit_degree_flux > 0 :
                text="Fit using polynomium of degree "+np.str(fit_degree_flux)
                plt.plot(self.wavelength, response_curve_fitted, "b-", alpha=0.6,linewidth=2, label=text)
                text=np.str(fit_degree_flux)+"-order fit smoothed with medfilt window = "+np.str(odd_number)
                plt.plot(self.wavelength, response_curve_interpolated, "-", color="purple", alpha=0.6,linewidth=2, label=text)
                               
            plt.xlim(self.wavelength[0]-50, self.wavelength[-1]+50)    
            plt.axvline(x=self.wavelength[0], color='k', linestyle='-', alpha=0.7)
            plt.axvline(x=self.wavelength[-1], color='k', linestyle='-', alpha=0.7)
            if ha_range[0] != 0: plt.axvspan(ha_range[0], ha_range[1], facecolor='orange', alpha=0.15, zorder=3) 
            if exclude_wlm[0][0] != 0:
                for i in range(len(exclude_wlm)):
                    plt.axvspan(exclude_wlm[i][0], exclude_wlm[i][1], color='r', alpha=0.1)   
            plt.ylabel("Flux calibration [ counts /s equivalent to 10$^{-16}$ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            plt.title("Response curve for absolute flux calibration using "+self.object)
            plt.minorticks_on()
            plt.grid(which='both')
            plt.axvline(x=min_wave_flux, color='k', linestyle='--', alpha=0.5)
            plt.axvline(x=max_wave_flux, color='k', linestyle='--', alpha=0.5)
            plt.legend(frameon=True, loc=4, ncol=2)
            plt.show()
            plt.close()
            
        print("  Min wavelength at {:.2f} with {:.3f} counts/s = 1E-16 erg/cm**2/s/A".format(self.wavelength[0], self.response_curve[0]))
        print("  Max wavelength at {:.2f} with {:.3f} counts/s = 1E-16 erg/cm**2/s/A".format(self.wavelength[-1], self.response_curve[-1]))   
        print("  Response curve to all wavelengths stored in self.response_curve. Length of the vector = ",len(self.response_curve)) 
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def fit_Moffat_between(self, min_wave=0, max_wave=0, r_max=5, plot=False, verbose = False):
        """
        Method of Interpolated cube instance.
        #TODO

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
        flux : TYPE #TODO
            DESCRIPTION.
        TYPE #TODO
            DESCRIPTION.
        beta : TYPE #TODO
            DESCRIPTION.

        """
        
        if min_wave == 0 : min_wave = self.valid_wave_min 
        if max_wave == 0 : max_wave = self.valid_wave_max 
              
        r2_growth_curve, F_growth_curve, flux, r2_half_light = self.growth_curve_between(min_wave, max_wave, plot=plot, verbose=verbose)
        flux, alpha, beta = fit_Moffat(r2_growth_curve, F_growth_curve,
                                       flux, r2_half_light, r_max, plot)
        r2_half_light = alpha * (np.power(2., 1./beta) - 1)
        
        if plot == True : verbose == True
        if verbose:
            print("Moffat fit: Flux = {:.3e},".format(flux), \
                "HWHM = {:.3f},".format(np.sqrt(r2_half_light)*self.pixel_size_arcsec), \
                "beta = {:.3f}".format(beta))

        return flux, np.sqrt(r2_half_light)*self.pixel_size_arcsec, beta
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def trim_cube(self, trim_cube=True, trim_values=[],   half_size_for_centroid = 10,     
                  remove_spaxels_not_fully_covered = True,      #nansum=False,          
                  ADR=True, box_x=[0,-1], box_y=[0,-1], edgelow=-1, edgehigh=-1, 
                  adr_index_fit = 2, g2d=False, step_tracing=25, adr_clip_fit=0.3,
                  kernel_tracing = 5, plot_tracing_maps =[],
                  plot_weight = False, fcal=False, plot=True, plot_spectra=False, verbose=True, warnings=True):
        """ 
        Task for trimming cubes in RA and DEC (not in wavelength)
        
        if nansum = True, it keeps spaxels in edges that only have a partial spectrum (default = False)
        if remove_spaxels_not_fully_covered = False, it keeps spaxels in edges that only have a partial spectrum (default = True)
        
        """    
        """
        Task for trimming cubes in RA and DEC (not in wavelength)

        Parameters
        ----------
        trim_cube : Boolean, optional
            DESCRIPTION. The default is True. #TODO
        trim_values : List, optional
            DESCRIPTION. The default is []. #TODO
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
            If True uses a 2D Gaussian, else doesn't. The default is False. 
        step_tracing : Integer, optional
            DESCRIPTION. The default is 100. #TODO
        plot_tracing_maps : List, optional
            If True will plot the tracing maps. The default is [].
        plot_weight : Boolean, optional
            DESCRIPTION. The default is False. #TODO
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
            if _x_ > 0 : cube = self.combined_cube
        except Exception:
            cube = self
        
        if remove_spaxels_not_fully_covered : 
            nansum = False
        else: nansum = True
        
        if verbose:
            if nansum:
                print("\n> Preparing for trimming the cube INCLUDING those spaxels in edges with partial wavelength coverage...")
            else:
                print("\n> Preparing for trimming the cube DISCARDING those spaxels in edges with partial wavelength coverage...")

            print('  Size of old cube : {} (RA) x {} (DEC) = [0 ... {} , 0 ... {}]  =  {:.2f}" x {:.2f}"'.format(cube.n_cols, cube.n_rows,cube.n_cols-1, cube.n_rows-1, cube.RA_segment,cube.DEC_segment))
        
        if len(trim_values) == 0:    # Trim values not given. Checking size for trimming             
            if verbose: print("  Automatically computing the trimming avoiding empty columns and rows...") 
            trim_values = [0, cube.n_cols-1, 0, cube.n_rows-1]  
            cube.get_integrated_map(nansum=nansum)
            n_row_values= np.nansum(cube.integrated_map, axis=1)           
            n_col_values= np.nansum(cube.integrated_map, axis=0)             
            
            stop = 0
            i = 0
            while stop < 1 :
                if n_col_values[i] == 0 :
                    trim_values[0] = i +1
                    i=i+1
                    if i == np.int(cube.n_cols/2):
                        if verbose or warnings: print("  Something failed checking left trimming...")
                        trim_values[0] = -1
                        stop = 2             
                else: stop = 2
            stop = 0
            i = cube.n_cols-1
            while stop < 1 :   
                if n_col_values[i] == 0 :
                    trim_values[1] = i-1 
                    i=i-1
                    if i == np.int(cube.n_cols/2):
                        if verbose or warnings: print("  Something failed checking right trimming...")
                        trim_values[1] = cube.n_cols
                        stop = 2 
                else: stop = 2                   
            stop = 0
            i=0
            while stop < 1 :
                if n_row_values[i] == 0 : 
                    trim_values[2] = i+1
                    i=i+1
                    if i == np.int(cube.n_rows/2):
                        if verbose or warnings: print("  Something failed checking bottom trimming...")
                        trim_values[2] = -1
                        stop = 2 
                else: stop = 2
            stop = 0
            i=cube.n_rows-1
            while stop < 1 :
                if n_row_values[i] == 0 : 
                    trim_values[3] = i-1 
                    i=i-1
                    if i == np.int(cube.n_rows/2):
                        if verbose or warnings: print("  Something failed checking top trimming...")
                        trim_values[3] = cube.n_rows
                        stop = 2 
                else: stop = 2  
        else:
            if trim_values[0] == -1 : trim_values[0] = 0 
            if trim_values[1] == -1 : trim_values[1] = cube.n_cols-1 
            if trim_values[2] == -1 : trim_values[2] = 0 
            if trim_values[3] == -1 : trim_values[3] = cube.n_rows-1 
                
            if verbose: print("  Trimming values provided: [ {}:{} , {}:{} ]".format(trim_values[0],trim_values[1],trim_values[2],trim_values[3])) 
            if trim_values[0] < 0 : 
                trim_values[0] = 0
                if verbose: print("  trim_value[0] cannot be negative!")
            if trim_values[1] > cube.n_cols: 
                trim_values[1] = cube.n_cols
                if verbose: print("  The requested value for trim_values[1] is larger than the RA size of the cube!")
            if trim_values[1] < 0 : 
                trim_values[1] = cube.n_cols
                if verbose: print("  trim_value[1] cannot be negative!")
            if trim_values[2] < 0 : 
                trim_values[2] = 0
                if verbose: print("  trim_value[2] cannot be negative!")
            if trim_values[3] > cube.n_rows: 
                trim_values[3] = cube.n_rows
                if verbose: print("  The requested value for trim_values[3] is larger than the DEC size of the cube!")
            if trim_values[3] < 0 : 
                trim_values[3] = cube.n_rows
                if verbose: print("  trim_value[3] cannot be negative!")
                               
        recorte_izquierda = (trim_values[0])*cube.pixel_size_arcsec  
        recorte_derecha =  (cube.n_cols-trim_values[1]-1)*cube.pixel_size_arcsec
        recorte_abajo = (trim_values[2])*cube.pixel_size_arcsec
        recorte_arriba = (cube.n_rows-trim_values[3]-1)*cube.pixel_size_arcsec

        corte_horizontal = (trim_values[0])+ (cube.n_cols-trim_values[1]-1)
        corte_vertical = (trim_values[2])+ (cube.n_rows-trim_values[3]-1)

        if verbose: 
            print('  Left trimming   : from spaxel   0  to spaxel {:3}  -> {:3} spaxels = {:8.2f}" '.format(trim_values[0],(trim_values[0]), recorte_izquierda))
            print('  Right trimming  : from spaxel {:3}  to spaxel {:3}  -> {:3} spaxels = {:8.2f}" '.format(trim_values[1],cube.n_cols-1,(cube.n_cols-trim_values[1]-1), recorte_derecha))
            print('  Bottom trimming : from spaxel   0  to spaxel {:3}  -> {:3} spaxels = {:8.2f}" '.format(trim_values[2],(trim_values[2]), recorte_abajo))                
            print('  Top trimming    : from spaxel {:3}  to spaxel {:3}  -> {:3} spaxels = {:8.2f}" '.format(trim_values[3],cube.n_rows-1,(cube.n_rows-trim_values[3]-1), recorte_arriba))
    
            print("  This will need a trimming of {} (RA) x {} (DEC) spaxels".format(corte_horizontal,corte_vertical))
               
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
        x1 = trim_values[0]  + (cube.n_cols-corte_horizontal) #  trim_values[1]      
        y0 = trim_values[2] 
        y1 = trim_values[2]  + (cube.n_rows-corte_vertical) #trim_values[3]      
         
        #print "  RA trim {}  DEC trim {}".format(corte_horizontal,corte_vertical)
        values = np.str(cube.n_cols-corte_horizontal)+' (RA) x '+np.str(cube.n_rows-corte_vertical)+' (DEC) = '+np.str(np.round((x1-x0)*cube.pixel_size_arcsec,2))+'" x '+np.str(np.round((y1-y0)*cube.pixel_size_arcsec,2))+'"'      

        if verbose:  print("\n> Recommended size values of combined cube = ",values)
                    
        if corte_horizontal == 0 and  corte_vertical == 0:
            if verbose: print("\n> No need of trimming the cube, all spaxels are valid.")
            trim_cube = False     
            cube.get_integrated_map()
        if trim_cube: 
            #if plot:
            #    print "> Plotting map with trimming box"
            #    cube.plot_map(mapa=cube.integrated_map, box_x=[x0,x1], box_y=[y0,y1])   
            cube.RA_centre_deg = cube.RA_centre_deg +(recorte_derecha-recorte_izquierda)/2 /3600.
            cube.DEC_centre_deg = cube.DEC_centre_deg + (recorte_abajo-recorte_arriba)/2./3600.
            if verbose: 
                print("\n> Starting trimming procedure:")
                print('  Size of old cube : {} (RA) x {} (DEC) = [0 ... {} , 0 ... {}]  =  {}" x {}"'.format(cube.n_cols, cube.n_rows,cube.n_cols-1, cube.n_rows-1, np.round(cube.RA_segment,2),np.round(cube.DEC_segment,2)))
                print("  Reducing size of the old cube in {} (RA) and {} (DEC) spaxels...".format(corte_horizontal, corte_vertical))                                
                print("  Centre coordenates of the old cube: ",cube.RA_centre_deg ,cube.DEC_centre_deg) 
                print('  Offset for moving the center from the old to the new cube:  {:.2f}" x {:.2f}"'.format(
                        (recorte_derecha-recorte_izquierda)/2., (recorte_abajo-recorte_arriba)/2.))                            
                print("  Centre coordenates of the new cube: ",cube.RA_centre_deg ,cube.DEC_centre_deg)                                                   
                print("  Trimming the cube [{}:{} , {}:{}] ...".format(x0,x1-1,y0,y1-1))
            cube.data=copy.deepcopy(cube.data[:,y0:y1,x0:x1])
            #cube.data_no_ADR = cube.data_no_ADR[:,y0:y1,x0:x1]
            cube.weight=cube.weight[:,y0:y1,x0:x1]                    
            if plot_weight: cube.plot_weight()                                        
            cube.n_cols = cube.data.shape[2]
            cube.n_rows = cube.data.shape[1]
            cube.RA_segment = cube.n_cols *cube.pixel_size_arcsec
            cube.DEC_segment = cube.n_rows *cube.pixel_size_arcsec
                         
            if verbose: print('  Size of new cube : {} (RA) x {} (DEC) = [0 ... {} , 0 ... {}]  =  {}" x {}"'.format(cube.n_cols, cube.n_rows,cube.n_cols-1, cube.n_rows-1, np.round(cube.RA_segment,2),np.round(cube.DEC_segment,2)))
                     
            cube.spaxel_RA0= cube.n_cols/2  -1 
            cube.spaxel_DEC0= cube.n_rows/2 -1

            if verbose: print('  The center of the cube is in position  [ {} , {} ]'.format(cube.spaxel_RA0,cube.spaxel_DEC0))
 
    
            if ADR:
                if np.nanmedian(box_x+box_y) != -0.5:
                    box_x_=[box_x[0]-x0, box_x[1]-x0]
                    box_y_=[box_y[0]-y0, box_y[1]-y0]
                else:
                    box_x_=box_x
                    box_y_=box_y
    
                if half_size_for_centroid > 0 and np.nanmedian(box_x+box_y) == -0.5:            
                    cube.get_integrated_map()            
                    if verbose: print("\n> As requested, using a box centered at the peak of emission, [ {} , {} ], and width +-{} spaxels for tracing...".format(cube.max_x,cube.max_y,half_size_for_centroid))      
                    box_x_ = [cube.max_x-half_size_for_centroid,cube.max_x+half_size_for_centroid]
                    box_y_ = [cube.max_y-half_size_for_centroid,cube.max_y+half_size_for_centroid]
           
                try:
                    cube.trace_peak(check_ADR=True, box_x=box_x_, box_y=box_y_, edgelow=edgelow, edgehigh =edgehigh, 
                                    adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing,
                                    kernel_tracing=kernel_tracing,adr_clip_fit=adr_clip_fit,
                                    plot=plot, plot_tracing_maps=plot_tracing_maps, verbose=verbose)
                except Exception:
                    if verbose or warnings: print("\n  WARNING !! Failing tracing the peak of the trimmed cube...\n\n")
            cube.get_integrated_map(fcal=fcal, plot=plot, plot_spectra=plot_spectra,plot_centroid=False, nansum=nansum)

            
        else:
            if corte_horizontal != 0 and  corte_vertical != 0 and verbose: print("\n> Trimming the cube was not requested")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def get_bad_spaxels(self, verbose = False, show_results = True, valid_wave_min = 0, valid_wave_max = 0):
        """
        Get a list with the bad spaxels (spaxels that don't have a valid spectrum in the range [valid_wave_min, valid_wave_max])

        Parameters
        ----------
        verbose : Boolean, optional
            Print results. The default is False.
        show_results : Boolean, optional
            DESCRIPTION. The default is True. #TODO
        valid_wave_min : Integer, optional
            DESCRIPTION. The default is 0. #TODO
        valid_wave_max : Integer, optional
            DESCRIPTION. The default is 0. #TODO

        Returns
        -------
        self.bad_spaxels with the list of bad spaxels

        """
        
        if valid_wave_min == 0 : valid_wave_min = self.valid_wave_min
        if valid_wave_max == 0 : valid_wave_max = self.valid_wave_max
        
        if verbose: print("\n> Checking bad spaxels (spaxels that don't have a valid spectrum in the range [ {} , {} ] ) :\n".format(np.round(valid_wave_min,2), np.round(valid_wave_max,2)))
        
        list_of_bad_spaxels=[]
        for x in range(self.n_cols):
            for y in range(self.n_rows):
                integrated_spectrum_of_spaxel= self.plot_spectrum_cube(x=x,y=y, plot=False, verbose=False)
                median = np.median(integrated_spectrum_of_spaxel[np.searchsorted(self.wavelength, valid_wave_min):np.searchsorted(self.wavelength,valid_wave_max)])
                if np.isnan(median) : 
                    if verbose: print("  - spaxel {:3},{:3} does not have a spectrum in all valid wavelenghts".format(x,y))
                    list_of_bad_spaxels.append([x,y])
                    #integrated_spectrum_of_spaxel= cubo.plot_spectrum_cube(x=x,y=y, plot=True, verbose=False)
        self.bad_spaxels = list_of_bad_spaxels
        if show_results: print("\n> List of bad spaxels :", self.bad_spaxels )

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
def get_offsets_between_cubes(cube_list, compare_cubes = False,
                              plot=True,verbose=True):
    
    n_cubes = len(cube_list)
    
    xx=[0]      # This will have 0, x12, x23, x34, ... xn1
    yy=[0]      # This will have 0, y12, y23, y34, ... yn1

    if compare_cubes:    # New method included 7 Feb 2022
        if verbose and n_cubes > 1: print("\n  Using comparison of cubes to align cubes...")
        for i in range(n_cubes-1):
            eocc=estimate_offsets_comparing_cubes(cube_list[i],cube_list[i+1], n_ite= 1, 
                                                  delta_RA_max = 4,
                                                  delta_DEC_max = 4,
                                                  #line=6400,line2=6500,
                                                  index_fit =0,
                                                  step=0.01,
                                                  plot=plot, plot_comparison=False,
                                                  verbose=verbose, return_values=True)      
            xx.append(-eocc[0])
            yy.append(-eocc[1])  
        eocc=estimate_offsets_comparing_cubes(cube_list[-1],cube_list[0], n_ite= 1, 
                                                  delta_RA_max = 4,
                                                  delta_DEC_max = 4,
                                                  #line=6400,line2=6500,
                                                  index_fit =0,
                                                  step=0.01,
                                                  plot=plot, plot_comparison=False,
                                                  verbose=verbose, return_values=True)      
        xx.append(-eocc[0])
        yy.append(-eocc[1])
    else:
        if verbose and n_cubes > 1: print("\n  Using peak of the emission tracing all wavelengths to align cubes...")
            
        for i in range(n_cubes-1):
            xx.append(cube_list[i+1].offset_from_center_x_arcsec_tracing - cube_list[i].offset_from_center_x_arcsec_tracing) 
            yy.append(cube_list[i+1].offset_from_center_y_arcsec_tracing - cube_list[i].offset_from_center_y_arcsec_tracing)  
        xx.append(cube_list[0].offset_from_center_x_arcsec_tracing - cube_list[-1].offset_from_center_x_arcsec_tracing)
        yy.append(cube_list[0].offset_from_center_y_arcsec_tracing - cube_list[-1].offset_from_center_y_arcsec_tracing)
    
    return xx,yy
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def align_n_cubes(rss_file_list, cube_list=[0], flux_calibration_list=[[]], 
                  reference_rss = "",  compare_cubes = False,
                  pixel_size_arcsec=0.7, kernel_size_arcsec=1.1, 
                  edgelow=-1, edgehigh=-1, size_arcsec=[], centre_deg=[], 
                  offsets=[], 
                  ADR=False, jump=-1, ADR_x_fit_list=[0], ADR_y_fit_list=[0], force_ADR = False,
                  half_size_for_centroid =10, box_x=[0,-1], box_y=[0,-1], 
                  adr_index_fit = 2, g2d=False, step_tracing = 25, kernel_tracing = 5, adr_clip_fit=0.3,
                  plot= False, plot_weight=False, plot_tracing_maps=[], plot_spectra=True,
                  warnings=False, verbose= True):
    """
    Routine to align n cubes. CAREFUL : rss_file_list HAS TO BE a list of RSS objects #TODO change to RSS files if needed

    Parameters #TODO
    ----------
    rss_file_list : List of RSS objects
        This is a list of RSS objects.
    cube_list : List of Cube Objects, optional
        DESCRIPTION. The default is [0].
    flux_calibration_list : TYPE, optional
        DESCRIPTION. The default is [[]].
    reference_rss : TYPE, optional
        DESCRIPTION. The default is "".
    pixel_size_arcsec : TYPE, optional
        DESCRIPTION. The default is 0.3.
    kernel_size_arcsec : TYPE, optional
        DESCRIPTION. The default is 1.5.
    edgelow : Integer, optional
        This is the lowest value in the wavelength range in terms of pixels. The default is -1.
    edgehigh : Integer, optional
        This is the highest value in the wavelength range in terms of pixels. The default is -1.
    size_arcsec : TYPE, optional
        DESCRIPTION. The default is [].
    centre_deg : TYPE, optional
        DESCRIPTION. The default is [].
    offsets : TYPE, optional
        DESCRIPTION. The default is [].
    ADR : Boolean, optional
        If True will correct for ADR (Atmospheric Differential Refraction). The default is False.
    jump : Integer, optional
        If a positive number partitions the wavelengths with step size jump, if -1 will not partition. The default is -1.
    ADR_x_fit_list : Integer List, optional
        This is a list of ADR x fits. The default is [0].
    ADR_y_fit_list : Integer List, optional
        This is a list of ADR y fits. The default is [0].
    force_ADR : Boolean, optional
        If True will correct for ADR even considoring a small correction. The default is False.
    half_size_for_centroid : Integer, optional
        This is half the length/width of the box. The default is 10.
    box_x : Integer List, optional
        When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [0,-1].
    box_y : Integer List, optional
        When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [0,-1].
    adr_index_fit : Integer, optional
        This is the fitted polynomial with highest degree n. The default is 2.
    g2d : Boolean, optional
        If True uses a 2D Gaussian, else doesn't. The default is False.
    step_tracing : Integer, optional
        DESCRIPTION. The default is 100.
    kernel_tracing : TYPE, optional
        DESCRIPTION. The default is 0.
    adr_clip_fit:
        
    plot : Boolean, optional
        If True generates and shows the plots. The default is False.
    plot_weight : Boolean, optional
        If True will plot the weight. The default is False.
    plot_tracing_maps : Boolean, optional
        If True will plot the tracing maps. The default is [].
    plot_spectra : Boolean, optional
        If True will plot the spectra. The default is True.
    warnings : Boolean, optional
        If True will show any problems that arose, else skipped. The default is False.
    verbose : Boolean, optional
        Print results. The default is True.

    Returns
    -------
    cube_aligned_list : List of aligned Cube Objects
        DESCRIPTION.

    """
       
    n_rss = len(rss_file_list)

    if verbose: 
        if n_rss > 1:
            print("\n> Starting alignment procedure...")
        else:
            print("\n> Only one file provided, no need of performing alignment ...")
            if np.nanmedian(ADR_x_fit_list) == 0 and ADR: print ("  But ADR data provided and ADR correction requested, rebuiding the cube...")
            
            
            
    xx=[0]      # This will have 0, x12, x23, x34, ... xn1
    yy=[0]      # This will have 0, y12, y23, y34, ... yn1
    
    if len(flux_calibration_list[0]) == 0:
        for i in range(1,n_rss): flux_calibration_list.append([])

    if len(offsets) == 0:  
        #if verbose and n_rss > 1: print("\n  Using peak of the emission tracing all wavelengths to align cubes:") 
        n_cubes = len(cube_list)
        if n_cubes != n_rss:
            if verbose:
                print("\n\n\n ERROR: number of cubes and number of rss files don't match!")
                print("\n\n THIS IS GOING TO FAIL ! \n\n\n")
                
        xx,yy = get_offsets_between_cubes (cube_list, compare_cubes=compare_cubes, plot=plot, verbose=verbose)
    
    else:
        if verbose and n_rss > 1: print("\n  Using offsets provided!")   
        for i in range(0,2*n_rss-2,2):
            xx.append(offsets[i])
            yy.append(offsets[i+1])
        xx.append(-np.nansum(xx))    #
        yy.append(-np.nansum(yy))
            
    # Estimate median value of the centre of files
    list_RA_centre_deg=[]
    list_DEC_centre_deg=[]
    
    for i in range(n_rss):
        list_RA_centre_deg.append(rss_file_list[i].RA_centre_deg)
        list_DEC_centre_deg.append(rss_file_list[i].DEC_centre_deg)        
    
    median_RA_centre_deg = np.nanmedian (list_RA_centre_deg)
    median_DEC_centre_deg = np.nanmedian (list_DEC_centre_deg)
    
    distance_from_median  = []
    
    for i in range(n_rss):
        rss_file_list[i].ALIGNED_RA_centre_deg = median_RA_centre_deg + np.nansum(xx[1:i+1])/3600.    # CHANGE SIGN 26 Apr 2019    # ERA cube_list[0]
        rss_file_list[i].ALIGNED_DEC_centre_deg = median_DEC_centre_deg  - np.nansum(yy[1:i+1])/3600.        # rss_file_list[0].DEC_centre_deg
    
        distance_from_median.append(np.sqrt( 
                (rss_file_list[i].RA_centre_deg - median_RA_centre_deg)**2 +
                (rss_file_list[i].DEC_centre_deg - median_DEC_centre_deg)**2) )
        
    if reference_rss == "":
        reference_rss = distance_from_median.index(np.nanmin(distance_from_median))
    
    if len(centre_deg) == 0:    
        if verbose and n_rss > 1: print("  No central coordenates given, using RSS {} for getting the central coordenates:".format(reference_rss+1))   
        RA_centre_deg = rss_file_list[reference_rss].ALIGNED_RA_centre_deg
        DEC_centre_deg = rss_file_list[reference_rss].ALIGNED_DEC_centre_deg  
    else:
        if verbose and n_rss > 1: print("  Central coordenates provided: ")   
        RA_centre_deg = centre_deg[0]
        DEC_centre_deg = centre_deg[1]  
        

    if verbose and n_rss > 1:
        print("\n> Median central coordenates of RSS files: RA =",RA_centre_deg," DEC =", DEC_centre_deg)
              
        print("\n  Offsets (in arcsec):        x             y                          ( EAST+ / WEST-   NORTH- / SOUTH+) ")
        for i in range(1,len(xx)-1):
            print("         {:2.0f} -> {:2.0f}           {:+5.3f}         {:+5.3f}".format(i,i+1,xx[i],yy[i]))      
        print("         {:2.0f} ->  1           {:+5.3f}         {:+5.3f}".format(len(xx)-1,xx[-1],yy[-1]))      
        print("           TOTAL:            {:5.3f}          {:5.3f}".format(np.nansum(xx),np.nansum(yy))) 
       
        print("\n         New_RA_centre_deg       New_DEC_centre_deg      Diff with respect Cube 1 [arcsec]")  
       
        for i in range (0,n_rss):
            print("  Cube {:2.0f}:     {:5.8f}          {:5.8f}           {:+5.3f}   ,  {:+5.3f}   ".format(i+1,rss_file_list[i].ALIGNED_RA_centre_deg, rss_file_list[i].ALIGNED_DEC_centre_deg, (rss_file_list[i].ALIGNED_RA_centre_deg-rss_file_list[0].ALIGNED_RA_centre_deg)*3600.,(rss_file_list[i].ALIGNED_DEC_centre_deg-rss_file_list[0].ALIGNED_DEC_centre_deg)*3600.))  
    
    offsets_files=[]
    for i in range(1,n_rss):           # For keeping in the files with self.offsets_files
        vector=[xx[i],yy[i]]        
        offsets_files.append(vector)

    xx_dif = np.nansum(xx[0:-1])   
    yy_dif = np.nansum(yy[0:-1]) 

    if verbose and n_rss > 1: print('\n  Accumulative difference of offsets: {:.2f}" x {:.2f}" '.format(xx_dif, yy_dif))
       
    if len(size_arcsec) == 0:
        RA_size_arcsec = rss_file_list[0].RA_segment + np.abs(xx_dif) + 3*kernel_size_arcsec
        DEC_size_arcsec =rss_file_list[0].DEC_segment + np.abs(yy_dif) + 3*kernel_size_arcsec 
        size_arcsec=[RA_size_arcsec,DEC_size_arcsec]

    if verbose and n_rss > 1: print('\n  RA_size x DEC_size  = {:.2f}" x {:.2f}" '.format(size_arcsec[0], size_arcsec[1]))

    cube_aligned_list=[]
    
    for i in range(1,n_rss+1):
        #escribe="cube"+np.str(i)+"_aligned"
        cube_aligned_list.append("cube"+np.str(i)+"_aligned")

    ADR_x_fit_list = []
    ADR_y_fit_list = []

    if np.nanmedian(ADR_x_fit_list) == 0 and ADR:   # Check if ADR info is provided and ADR is requested
        for i in range(n_rss): 
            _x_ = []
            _y_ = []
            for j in range(len(cube_list[i].ADR_x_fit)):
                _x_.append(cube_list[i].ADR_x_fit[j])
                _y_.append(cube_list[i].ADR_y_fit[j])   
            ADR_x_fit_list.append(_x_)
            ADR_y_fit_list.append(_y_)
    else:
        for rss in rss_file_list:
            ADR_x_fit_list.append([0])
            ADR_y_fit_list.append([0])
            

    for i in range(n_rss):
        
        if n_rss > 1 or np.nanmedian(ADR_x_fit_list) != 0:

            if verbose: print("\n> Creating aligned cube",i+1,"of a total of",n_rss,"...")
            
            cube_aligned_list[i]=Interpolated_cube(rss_file_list[i], pixel_size_arcsec=pixel_size_arcsec, kernel_size_arcsec=kernel_size_arcsec, 
                                                   centre_deg=[RA_centre_deg, DEC_centre_deg], size_arcsec=size_arcsec, 
                                                   aligned_coor=True, flux_calibration=flux_calibration_list[i],  offsets_files = offsets_files, offsets_files_position =i+1, 
                                                   ADR=ADR, jump=jump, ADR_x_fit = ADR_x_fit_list[i], ADR_y_fit = ADR_y_fit_list[i], check_ADR=True,
                                                   half_size_for_centroid=half_size_for_centroid, box_x=box_x, box_y=box_y,
                                                   adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing, plot_tracing_maps=plot_tracing_maps, kernel_tracing = kernel_tracing, adr_clip_fit=adr_clip_fit,
                                                   plot=plot, plot_spectra=plot_spectra, edgelow=edgelow, edgehigh=edgehigh, 
                                                   
                                                   warnings=warnings, verbose=verbose)
            if plot_weight: cube_aligned_list[i].plot_weight()
        else:
            cube_aligned_list[i] = cube_list[i]
            if verbose: print("\n> Only one file provided and no ADR correction given, the aligned cube is the same than the original cube...")

    xxx,yyy = get_offsets_between_cubes (cube_aligned_list, compare_cubes=compare_cubes, plot=plot, verbose=verbose)
    
    if verbose and n_rss > 1:
        print("\n> Checking offsets of ALIGNED cubes (in arcsec, everything should be close to 0):")
        print("  Offsets (in arcsec):        x             y                          ( EAST+ / WEST-   NORTH- / SOUTH+) ")

        for i in range(1,len(xxx)-1):
            print("         {:2.0f} -> {:2.0f}           {:+5.3f}         {:+5.3f}".format(i,i+1,xxx[i-1],yyy[i-1]))      
        print("         {:2.0f} ->  1           {:+5.3f}         {:+5.3f}".format(len(xxx)-1,xxx[-1],yyy[-1]))      
        print("           TOTAL:            {:5.3f}          {:5.3f}".format(np.nansum(xxx),np.nansum(yyy))) 
    
        print("\n> Alignment of n = {} cubes COMPLETED !".format(n_rss))
    return cube_aligned_list
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def align_blue_and_red_cubes(blue, red, half_size_for_centroid = 12, box_x= [], box_y =[], 
                             g2d=False, step_tracing = 25,  adr_index_fit = 2, kernel_tracing = 5, adr_clip_fit = 0.3,
                             verbose = True, plot = True, plot_centroid=True, ):
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
    """
    This Function aligns the blue and red cubes

    Parameters
    ----------
    blue : TYPE
        DESCRIPTION.
    red : TYPE
        DESCRIPTION.
    half_size_for_centroid : Integer, optional
        This is half the length/width of the box. The default is 8.
    box_x : Integer List, optional
        When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [].
    box_y : Integer List, optional
        When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [].
    kernel_tracing : TYPE, optional
        DESCRIPTION. The default is 0.
    verbose : Boolean, optional
        Print results. The default is True.
    plot : Boolean, optional
        If True generates and shows the plots. The default is True.
    plot_centroid : Boolean, optional
        If True will plot the centroid. The default is True.
    g2d : Boolean, optional
        If True uses a 2D Gaussian, else doesn't. The default is False.

    Returns
    -------
    None.

    """
    
    

    print("\n> Checking the alignment between a blue cube and a red cube...")

    try:
        try_read=blue+"  "
        if verbose: text_intro = "\n> Reading the blue cube from the fits file..."+try_read[-2:-1]
        blue_cube=read_cube(blue, text_intro = text_intro,
                            half_size_for_centroid=half_size_for_centroid, 
                            g2d=g2d, step_tracing = step_tracing,  adr_index_fit = adr_index_fit, kernel_tracing = kernel_tracing, adr_clip_fit=adr_clip_fit,
                            plot=plot, plot_spectra=False, verbose = verbose)
    except Exception:
        print("  - The blue cube is an object")
        blue_cube = blue
    
        
    try:
        try_read=red+"  "
        if verbose: text_intro = "\n> Reading the red cube from the fits file..."+try_read[-2:-1]
        red_cube=read_cube(red, text_intro = text_intro,
                           half_size_for_centroid=half_size_for_centroid, 
                           g2d=g2d, step_tracing = step_tracing,  adr_index_fit = adr_index_fit, kernel_tracing = kernel_tracing,adr_clip_fit=adr_clip_fit,
                           plot=plot, plot_spectra=False, verbose = verbose)
    except Exception:
        print("  - The red  cube is an object")
        red_cube = red
        if box_x == [] or box_y ==[] :
            box_x, box_y = red_cube.box_for_centroid(half_size_for_centroid = half_size_for_centroid, verbose=verbose)
        blue_cube.get_integrated_map(box_x = box_x, box_y = box_y, plot_spectra=False, plot=plot, verbose = verbose, plot_centroid=plot_centroid, g2d=g2d, kernel_tracing=kernel_tracing, trimmed = False)
        red_cube.get_integrated_map(box_x = box_x, box_y = box_y, plot_spectra=False, plot=plot, verbose = verbose, plot_centroid=plot_centroid, g2d=g2d, kernel_tracing=kernel_tracing, trimmed = False)
  
    print("\n> Checking the properties of these cubes:\n")
    print("  CUBE      RA_centre             DEC_centre     pixel size   kernel size   n_cols      n_rows      x_max      y_max")
    print("  blue   {}   {}      {}           {}         {}          {}          {}         {}".format(blue_cube.RA_centre_deg,blue_cube.DEC_centre_deg, blue_cube.pixel_size_arcsec, blue_cube.kernel_size_arcsec, blue_cube.n_cols,blue_cube.n_rows, blue_cube.max_x, blue_cube.max_y))
    print("  red    {}   {}      {}           {}         {}          {}          {}         {}".format(red_cube.RA_centre_deg,red_cube.DEC_centre_deg, red_cube.pixel_size_arcsec, red_cube.kernel_size_arcsec,red_cube.n_cols,red_cube.n_rows, red_cube.max_x, red_cube.max_y))

    all_ok = True  
    to_do_list=[]
    for _property_ in ["RA_centre_deg", "DEC_centre_deg", "pixel_size_arcsec", "kernel_size_arcsec", "n_cols", "n_rows"]:
        property_values = [_property_]
        exec("property_values.append(blue_cube."+_property_+")")
        exec("property_values.append(red_cube."+_property_+")")
        property_values.append(property_values[-2]-property_values[-1])
        
        if property_values[-1] != 0 :
            print("  - Property {} has DIFFERENT values !!!".format(_property_))
            all_ok = False
            if _property_ == "RA_centre_deg" : to_do_list.append("  - Check the RA_centre_deg to get the same value in both cubes")
            if _property_ == "DEC_centre_deg" : to_do_list.append("  - Check the DEC_centre_deg to get the same value in both cubes")
            if _property_ == "pixel_size_arcsec" : to_do_list.append("  - The pixel size of the cubes is not the same!")
            if _property_ == "kernel_size_arcsec" : to_do_list.append("  - The kernel size of the cubes is not the same!")
            if _property_ == "n_cols" : to_do_list.append("  - The number of columns is not the same! Trim the largest cube")
            if _property_ == "n_rows" : to_do_list.append("  - The number of rows is not the same! Trim the largest cube")
            if _property_ == "x_max" : to_do_list.append("  - x_max is not the same!")
            if _property_ == "y_max" : to_do_list.append("  - y_max is not the same!")
            
    pixel_size_arcsec = red_cube.pixel_size_arcsec     
   
    x_peak_red =  red_cube.x_peak_median
    y_peak_red =  red_cube.y_peak_median
    x_peak_blue =  blue_cube.x_peak_median
    y_peak_blue =  blue_cube.y_peak_median    
    delta_x = x_peak_blue-x_peak_red
    delta_y = y_peak_blue-y_peak_red
    
    print("\n> The offsets between the two cubes following tracing the peak are:\n")
    print("  -> delta_RA  (blue -> red) = {}  spaxels         = {} arcsec".format(round(delta_x,3), round(delta_x*pixel_size_arcsec, 3)))
    print("  -> delta_DEC (blue -> red) = {}  spaxels         = {} arcsec".format(round(delta_y,3), round(delta_y*pixel_size_arcsec, 3)))
    delta = np.sqrt(delta_x**2 + delta_y**2)
    print("\n     TOTAL     (blue -> red) = {}  spaxels         = {} arcsec      ({}% of the pix size)".format(round(delta,3), round(delta*pixel_size_arcsec, 3), round(delta*100,1)))
       
    if delta > 0.5 :
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
        mapa_red  = red_cube.integrated_map  / np.nanmedian(red_cube.integrated_map)      
        red_cube.plot_map(mapa=(mapa_blue+mapa_red)/2,  log=True, barlabel="Blue and red maps combined", description="Normalized BLUE+RED", cmap="binary_r")

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
    """
    This function calculates the running mean of an array

    Parameters
    ----------
    x : Numpy Array
        This is the given array.
    N : Integer
        This is the number of values to be considered in the array ***. NOTE: N =/= 0 and if N = 1 we just get the array back as floats.

    Returns
    -------
    Numpy Array of Floats
        This is the running mean array.

    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def cumulative_Moffat(r2, L_star, alpha2, beta):
    """
    #TODO

    Parameters
    ----------
    r2 : TYPE
        DESCRIPTION.
    L_star : TYPE
        DESCRIPTION.
    alpha2 : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return L_star*(1 - np.power(1+(r2/alpha2), -beta))
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fit_Moffat(r2_growth_curve, F_growth_curve,
               F_guess, r2_half_light, r_max, plot=False):
    """
    Fits a Moffat profile to a flux growth curve
    as a function of radius squared,
    cutting at to r_max (in units of the half-light radius),
    provided an initial guess of the total flux and half-light radius squared.

    Parameters #TODO
    ----------
    r2_growth_curve : TYPE
        DESCRIPTION.
    F_growth_curve : TYPE
        DESCRIPTION.
    F_guess : TYPE
        DESCRIPTION.
    r2_half_light : TYPE
        DESCRIPTION.
    r_max : TYPE
        DESCRIPTION.
    plot : Boolean, optional
        If True generates and shows the plots. The default is False.

    Returns
    -------
    fit : TYPE
        DESCRIPTION.
    
    """
    index_cut = np.searchsorted(r2_growth_curve, r2_half_light*r_max**2)
    fit, cov = optimize.curve_fit(cumulative_Moffat,
                                  r2_growth_curve[:index_cut], F_growth_curve[:index_cut],
                                  p0=(F_guess, r2_half_light, 1)
                                  )
    if plot:
        print("Best-fit: L_star =", fit[0])
        print("          alpha =", np.sqrt(fit[1]))
        print("          beta =", fit[2])
        r_norm = np.sqrt(np.array(r2_growth_curve) / r2_half_light)
        plt.plot(r_norm, cumulative_Moffat(np.array(r2_growth_curve),
                                          fit[0], fit[1], fit[2])/fit[0], ':')
    return fit
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def offset_between_cubes(cube1, cube2, plot=True, verbose = True, smooth = 51):   
    """
    This is the offset between the 2 cubes, in terms of Right Ascension and Declination.

    Parameters #TODO
    ----------
    cube1 : Object Cube
        DESCRIPTION.
    cube2 : Object Cube
        DESCRIPTION.
    plot : Boolean, optional
        If True generates and shows the plots. The default is True.

    Returns
    -------
    delta_RA_arcsec : TYPE
        DESCRIPTION.
    delta_DEC_arcsec : TYPE
        DESCRIPTION.

    """
    p = cube1.pixel_size_arcsec
    
    x = (cube2.x_peaks - cube2.n_cols/2. + cube2.RA_centre_deg*3600./cube2.pixel_size_arcsec) \
        - (cube1.x_peaks - cube1.n_cols/2. + cube1.RA_centre_deg*3600./cube1.pixel_size_arcsec)
    y = (cube2.y_peaks - cube2.n_rows/2. + cube2.DEC_centre_deg*3600./cube2.pixel_size_arcsec) \
        - (cube1.y_peaks - cube1.n_rows/2. + cube1.DEC_centre_deg*3600./cube1.pixel_size_arcsec)
        
    delta_RA_pix = np.nanmedian(x)
    delta_DEC_pix = np.nanmedian(y)
    
    delta_RA_arcsec = delta_RA_pix * p
    delta_DEC_arcsec = delta_DEC_pix * p
    
    bsx = basic_statistics(x, return_data=True, verbose = False)
    bsy = basic_statistics(y, return_data=True, verbose = False)
    
    #dx_arcsec = (x - delta_RA_pix) * p
    #dy_arcsec = (y - delta_DEC_pix) * p  
    
    smooth_x = signal.medfilt(x, smooth)
    smooth_y = signal.medfilt(y, smooth)

    delta_RA_pix_smooth = np.nanmedian(smooth_x)
    delta_DEC_pix_smooth = np.nanmedian(smooth_y)

    delta_RA_arcsec_smooth = delta_RA_pix_smooth * p
    delta_DEC_arcsec_smooth = delta_DEC_pix_smooth * p
    
    bsx_s = basic_statistics(smooth_x, return_data=True, verbose = False)
    bsy_s = basic_statistics(smooth_y, return_data=True, verbose = False)
    
    #dx_arcsec_smooth = (smooth_x - delta_RA_pix_smooth) * p
    #dy_arcsec_smooth = (smooth_y - delta_DEC_pix_smooth) * p 
    
    if verbose:
        print("\n> Computing offsets between 2 cubes:\n")
        print('  - No smoothing:            (delta_RA, delta_DEC) = ({:.2f} +- {:.2f} , {:.2f} +- {:.2f}) arcsec' \
              .format(delta_RA_arcsec, bsx[3], delta_DEC_arcsec, bsy[3]))
        print('  - With smoothing {:4.0f} :    (delta_RA, delta_DEC) = ({:.2f} +- {:.2f} , {:.2f} +- {:.2f}) arcsec' \
            .format(smooth, delta_RA_arcsec_smooth, bsx_s[3], delta_DEC_arcsec_smooth, bsy_s[3]))   
                        
    if plot:
        w = cube1.wavelength
        plot_plot(w, [x*p, y*p, smooth_x*p, smooth_y*p], 
                  alpha=[0.1,0.1,1,1],
                  ptitle="Offsets between cubes",
                  color=["k","r","k","r"],
                  psym=[".",".","-","-"],
                  hlines =[0], ylabel="Offset [arcsec]",
                  label=["RA","DEC","RA smooth", "DEC smooth"])
        
    return delta_RA_arcsec, delta_DEC_arcsec
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def compare_cubes(cube1, cube2, line=None, line2=None, 
                  map1 = None, map2 = None,
                  delta_RA=0, delta_DEC=0, plot =True, verbose = True):
    """
    This function compares the inputted cubes and plots them

    Parameters
    ----------
    cube1 : TYPE
        DESCRIPTION.
    cube2 : TYPE
        DESCRIPTION.
    line : Integer, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    
    if map1 is None:
        if line is not None:
            cube1.get_integrated_map(min_wave=line, max_wave=line2, 
                                        plot=False, verbose=False)
            if line2 is not None:
                if verbose: print("\n> Comparing 2 cubes using the integrated map between {:.2f} and {:.2f} A ...".format(line,line2))
            else:
                if verbose: print("\n> Comparing 2 cubes using the map closest to {:.2f} A ...".format(line))
        else:
            if verbose: print("\n> Comparing 2 cubes using their integrated map ...")
        map1=cube1.integrated_map
    else:
        if verbose: print("\n> Comparing 2 cubes using the maps provided ...")
        
    if map2 is None:
        if line is not None:
            cube2.get_integrated_map(min_wave=line, max_wave=line2, 
                                        plot=False, verbose=False)
        map2=cube2.integrated_map

        
    # Shift the second map if requested
    if delta_RA + delta_DEC != 0:
        if verbose: 
            print('  - Shifting map2 in RA = {:.2f}" and DEC = {:.2f}" ...'.format(delta_RA, delta_DEC))            
        map2_shifted=shift_map(map2, delta_RA=delta_RA, delta_DEC=delta_DEC, pixel_size_arcsec=cube2.pixel_size_arcsec, verbose=False)
    else:
        map2_shifted=map2
        
    # Normalize maps using common region
    map_median = map1-map2_shifted
    mask = map_median*0.
    mask[np.where( np.isnan(map_median) == False  )]=1.
    
    spaxels_comunes = np.nansum(mask)
    
    map1_median_mask = np.nanmedian(mask * map1)
    map2_shifted_median_mask = np.nanmedian(mask * map2_shifted)
    
    factor12 = map1_median_mask / map2_shifted_median_mask
    scale = np.nanmedian(map1+map2_shifted * factor12) * 3
    scatter = np.nansum(np.abs((map1-map2_shifted * factor12))) / spaxels_comunes
    
    if verbose: 
        print("  - Spaxels in common =  {:.0f}".format(spaxels_comunes))
        print("  - Scale for cube2   = ", factor12)
        print("  - Medium scatter    = ",scatter)

    if plot:
        plt.figure(figsize=(12, 8))
        plt.imshow(map1-map2_shifted*factor12, vmin=-scale, vmax=scale, cmap=plt.cm.get_cmap('RdBu'))
        plt.colorbar()
        plt.contour(map1, colors='g', linewidths=2, norm=colors.LogNorm())
        plt.contour(map2_shifted, colors='k', linewidths=1, norm=colors.LogNorm())
        plt.minorticks_on()                
        plt.xlabel('$\Delta$ RA [ pix ]', fontsize=10)
        plt.ylabel('$\Delta$ DEC [ pix ]', fontsize=10)
        plt.legend(loc='upper right', frameon=False)

        if line is None:
            ptitle="Comparing cubes using their integrated maps"
        else:
            AA = "$\mathrm{\AA}$"
            if line2 is None: 
                ptitle = "Comparing cubes using wavelength closest to {:.2f} ".format(line)
            else:
                ptitle= "Comparing cubes using integrated map between {:.2f} and {:.2f} ".format(line, line2)
            ptitle = ptitle+AA
        plt.title(ptitle)
        plt.show()
        plt.close()
    return scatter
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def estimate_offsets_comparing_cubes(cube1, cube2, line=None, line2=None,   # BOBA
                                     map1=None, map2=None,
                                     step=0.1, delta_RA_max=2, delta_DEC_max=2,
                                     delta_RA_values=None, 
                                     delta_DEC_values = None, index_fit = 2, n_ite = 1,  # Iterations do not properly work...
                                     plot=True, plot_comparison=True, 
                                     verbose = True, return_values = False): 
        
    # Get maps
    if map1 is None:
        if line is not None:
            # cube1.get_integrated_map(min_wave=line, max_wave=line2, 
            #                             plot=False, verbose=False)
            map1_ = create_map(cube1, line, w2=line2, verbose = False )
            map1 = map1_[1]
            
            if line2 is not None:
                if verbose: print("\n> Estimating offsets comparing two cubes using the integrated map between {:.2f} and {:.2f} A ...".format(line,line2))
            else:
                if verbose: print("\n> Estimating offsets comparing two cubes using the map closest to {:.2f} A ...".format(line))
        else:
            if verbose: print("\n> Estimating offsets comparing two cubes using their integrated map...")
            
    else:
        map1=cube1.integrated_map
        if verbose: print("\n> Estimating offsets comparing two cubes using the maps provided ...")
        
    if map2 is None:
        if line is not None:
            # cube2.get_integrated_map(min_wave=line, max_wave=line2, 
            #                             plot=False, verbose=False)
            map2_ = create_map(cube2, line, w2=line2, verbose = False )
            map2 = map2_[1]
        else:
            map2=cube2.integrated_map
    

    # Iterate at least 2 times : one broad step*10, another small around the minimum value
    if n_ite > 1 and verbose: print(" ")
    for i in range(n_ite):
        if i == 0:
            best_delta_RA = 0
            best_delta_DEC = 0
            if n_ite == 1:
                step_here = step*100
            else:
                step_here = step*1000
            # Get offsets values
            if delta_RA_values is None:
                delta_RA_values = np.arange(-delta_RA_max*100,delta_RA_max*100,step_here)/100
        
            if delta_DEC_values is None:
                delta_DEC_values = np.arange(-delta_DEC_max*100,delta_DEC_max*100,step_here)/100
        else:
            delta_RA_values = np.arange((best_delta_RA -1)*100,(best_delta_RA +1)*100,step*100)/100
            delta_DEC_values = np.arange((best_delta_DEC -1)*100,(best_delta_DEC +1)*100,step*100)/100
    
        # Iterate      
        scatter_x =[]    
        for delta_RA in delta_RA_values:
            scatter_x.append(compare_cubes(cube1, cube2, map1=map1, map2=map2, 
                                           line=line, delta_RA=delta_RA, delta_DEC=best_delta_DEC ,
                                           plot=False, verbose =False))
            
        scatter_x_min = np.nanmin(scatter_x)
        v = np.abs(scatter_x-scatter_x_min)
        x_index = v.tolist().index(np.nanmin(v))
    
        scatter_y =[]    
        for delta_DEC in delta_DEC_values:
            scatter_y.append(compare_cubes(cube1, cube2, map1=map1, map2=map2, 
                                           line=line, delta_DEC=delta_DEC, delta_RA=best_delta_RA,
                                           plot=False, verbose =False))
    
        scatter_y_min = np.nanmin(scatter_y)
        v = np.abs(scatter_y-scatter_y_min)
        y_index = v.tolist().index(np.nanmin(v))
        
        # Compute
        
        best_delta_RA  = delta_RA_values[x_index]
        best_delta_DEC = delta_DEC_values[y_index]
        
        if index_fit > 0:
            fit_RA = np.polyfit(delta_RA_values,scatter_x, index_fit)
            yfit = np.poly1d(fit_RA)
            scatter_x_fit = yfit(delta_RA_values)
            fit_DEC = np.polyfit(delta_DEC_values,scatter_y, index_fit)
            yfit = np.poly1d(fit_DEC)
            scatter_y_fit = yfit(delta_DEC_values)

        if plot:
            #Determine max and min for plotting
            ymin_ = np.nanmin([scatter_x_min,scatter_y_min])   
            ymax_ = np.nanmax([np.nanmax(scatter_x),np.nanmax(scatter_y)])                  
            rango = ymax_ - ymin_
            ymin = ymin_ -rango/15.
            ymax = ymax_ + rango/15.        
    
            if index_fit > 0:
                x=[delta_RA_values, delta_DEC_values, delta_RA_values, delta_DEC_values]
                y=[scatter_x, scatter_y, scatter_x_fit, scatter_y_fit]
                label=["RA", "DEC", "RA fit", "DEC fit"]
                psym=[".","+", "-","-"]
            else:
                x=[delta_RA_values, delta_DEC_values]
                y=[scatter_x, scatter_y]
                label=["RA", "DEC"]
                psym=[".","+"]            
                  
            plot_plot(x,y,label=label, psym=psym,
                      xlabel = "Offset [ arcsec ]", 
                      ylabel="Scatter [ counts / spaxels ]",
                      ymax=ymax, ymin = ymin, 
                      vlines=[0, best_delta_RA, best_delta_DEC],
                      cvlines=["k", "r", "b"],
                      ptitle="Estimating offsets comparing cubes")
        
        
        if verbose and n_ite > 1: print('  - Iteration {:}: best_delta_RA = {:6.2f}", best_delta_DEC = {:6.2f}"'.format(i+1, best_delta_RA,best_delta_DEC))
        

    
        
    if plot_comparison:
        compare_cubes(cube1, cube2, line=line, line2=line2, map1=map1,map2=map2,
                      delta_RA=best_delta_RA, delta_DEC=best_delta_DEC, 
                      plot=True, verbose =verbose)

    if verbose:
        print('  - The offsets with the smallest scatter in common region are:  delta_RA = {:.2f}" , delta_DEC = {:.2f}"'.format(np.round(best_delta_RA,2), np.round(best_delta_DEC,2)))

    if return_values:
        return best_delta_RA, best_delta_DEC
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_response(calibration_star_cubes, scale=[], use_median=False, verbose = True, plot=True):
    """
    #TODO

    Parameters #TODO
    ----------
    calibration_star_cubes : TYPE
        DESCRIPTION.
    scale : List, optional
        DESCRIPTION. The default is [].
    use_median : Boolean, optional
        If True will use the median instead of mean. The default is False.

    Returns
    -------
    None.

    """
    
 
    n_cubes = len(calibration_star_cubes)
    if len(scale) == 0:
        for i in range(n_cubes):
            scale.append(1.)    

    wavelength = calibration_star_cubes[0].wavelength

    if verbose: print("\n> Comparing response curve of standard stars...\n")
    for i in range(n_cubes):
        ci = calibration_star_cubes[i].response_curve * scale[i]
        ci_name=calibration_star_cubes[i].object
        for j in range(i+1,n_cubes):
            cj = calibration_star_cubes[j].response_curve * scale[j]
            cj_name=calibration_star_cubes[j].object
            ptitle = "Comparison of flux calibration for "+ci_name+" and "+cj_name
            ylabel = ci_name+" / "+cj_name
            plot_plot(wavelength,ci/cj, hlines=[0.85, 0.9,0.95,1,1,1,1,1.05,1.1,1.15], ymin=0.8, ymax=1.2, 
                      ylabel=ylabel, ptitle=ptitle)
    if verbose: print("\n> Plotting response curve (absolute flux calibration) of standard stars...\n")
    
    
    mean_curve = np.zeros_like(wavelength)
    mean_values=[]
    list_of_scaled_curves=[]
    i = 0
    for star in calibration_star_cubes:    
        list_of_scaled_curves.append(star.response_curve * scale[i])
        mean_curve = mean_curve + star.response_curve * scale[i]
        if plot:
            plt.plot(star.wavelength, star.response_curve * scale[i],
                     label=star.description, alpha=0.2, linewidth=2)
        if use_median:
            if verbose: print("  Median value for ",star.object," = ",np.nanmedian(star.response_curve * scale[i]),"      scale = ",scale[i])
        else:
            if verbose: print("  Mean value for ",star.object," = ",np.nanmean(star.response_curve * scale[i]),"      scale = ",scale[i])
        mean_values.append(np.nanmean(star.response_curve)* scale[i])
        i=i+1

    mean_curve /= len(calibration_star_cubes)
    median_curve = np.nanmedian(list_of_scaled_curves, axis=0)
        
    response_rms= np.zeros_like(wavelength)
    for i in range(len(calibration_star_cubes)):
        if use_median:
            response_rms +=  np.abs(calibration_star_cubes[i].response_curve * scale[i] - median_curve) 
        else:
            response_rms +=  np.abs(calibration_star_cubes[i].response_curve * scale[i] - mean_curve) 
            
        
    response_rms /= len(calibration_star_cubes)
    if use_median:
        dispersion = np.nansum(response_rms)/np.nansum(median_curve)
    else:
        dispersion = np.nansum(response_rms)/np.nansum(mean_curve)

    if len(calibration_star_cubes) > 1 and verbose: print("  Variation in flux calibrations =  {:.2f} %".format(dispersion*100.))

    #dispersion=np.nanmax(mean_values)-np.nanmin(mean_values)
    #print "  Variation in flux calibrations =  {:.2f} %".format(dispersion/np.nanmedian(mean_values)*100.)

    if plot: 
        plt.figure(figsize=(11, 8))
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
def obtain_flux_calibration(calibration_star_cubes, verbose = True):
    """
    This function obtains the flux calibration of the star cubes

    Parameters
    ----------
    calibration_star_cubes : TYPE
        DESCRIPTION.

    Returns
    -------
    flux_calibration : TYPE
        DESCRIPTION.

    """
    
#    print "\n> Obtaining flux calibration...\n" 
    vector_wave = []
    vector_response = []
    cube_star=calibration_star_cubes[0]
    for i in range(len(cube_star.response_curve)):
        if np.isnan(cube_star.response_curve[i]) == False:
            #vector_wave.append(cube_star.response_wavelength[i])
            vector_wave.append(cube_star.wavelength[i])
            vector_response.append(cube_star.response_curve[i])
            #print "  For wavelength = ",cube_star.response_wavelength[i], " the flux correction is = ", cube_star.response_curve[i]

    interpolated_response = interpolate.splrep(vector_wave, vector_response, s=0)
    flux_calibration = interpolate.splev(cube_star.wavelength, interpolated_response, der=0)
#    flux_correction = flux_calibration
    if verbose:
        print("\n> Flux calibration for all wavelengths = ",flux_calibration)
        print("\n  Flux calibration obtained!")
    return flux_calibration
# -----------------------------------------------------------------------------   
# -----------------------------------------------------------------------------
def obtain_telluric_correction(w, telluric_correction_list, plot=True, verbose = True,
                               label_stars=[], scale=[]):
    """
    This function obtains the telluric correction.

    Parameters
    ----------
    w : TYPE
        DESCRIPTION.
    telluric_correction_list : TYPE
        DESCRIPTION.
    plot : Boolean, optional
        If True generates and shows the plots. The default is True.
    label_stars : List, optional
        DESCRIPTION. The default is [].
    scale : List, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    telluric_correction : TYPE
        This is the telluric correction.

    """
    
    if len(scale) == 0:
        for star in telluric_correction_list: scale.append(1.)
    
    for i in range(len(telluric_correction_list)): 
        telluric_correction_list[i] = [1. if x*scale[i] < 1 else x*scale[i] for x in telluric_correction_list[i]]
                
    telluric_correction=np.nanmedian(telluric_correction_list, axis=0)
    if plot:
        fig_size=12
        plt.figure(figsize=(fig_size, fig_size/2.5))
        plt.title("Telluric correction")
        for i in range(len(telluric_correction_list)):
            if len(label_stars) > 0:
                label=label_stars[i]
            else:    
                label="star"+str(i+1)
            plt.plot(w, telluric_correction_list[i], alpha=0.3, label=label)              
        plt.plot(w, telluric_correction, alpha=0.5, color="k", label="Median")        
        plt.minorticks_on()
        plt.legend(frameon=False, loc=2, ncol=1)
        step_up = 1.15*np.nanmax(telluric_correction)
        plt.ylim(0.9,step_up)
        plt.xlim(w[0]-10,w[-1]+10)
        plt.show()
        plt.close()    

    if verbose: 
        print("\n> Telluric correction = ",telluric_correction)
        if np.nanmean(scale) != 1. : print("  Telluric correction scale provided : ",scale)
        print("\n  Telluric correction obtained!")
    return telluric_correction
# -----------------------------------------------------------------------------   
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------     
def telluric_correction_from_star(objeto, save_telluric_file="",   
                                  object_rss = False,
                                  high_fibres=20, 
                                  list_of_telluric_ranges = [[0]], order = 2,                                
                                  apply_tc=False, 
                                  wave_min=0, wave_max=0, 
                                  plot=True, fig_size=12, verbose=True):

    """
    Get telluric correction using a spectrophotometric star  
    
    IMPORTANT! Check task self.get_telluric_correction !!!
        
    Example
    ----------
    telluric_correction_star1 = star1r.get_telluric_correction(high_fibres=15)            
    """    """
    

    Parameters
    ----------
    objeto : object
        This is the object.
    save_telluric_file : String, optional
        This is what the file name will be when saving the telluric file. The default is "".
    object_rss : Boolean, optional
        If the object is an rss object. The default is False.
    high_fibres : Integer, optional
        Number of fibers to add for obtaining spectrum. The default is 20.
    list_of_telluric_ranges : List of Integer List, optional
        DESCRIPTION. The default is [[0]].
    order : Integer, optional
        DESCRIPTION. The default is 2.
    apply_tc : Boolean, optional
        Apply telluric correction to data. The default is False.
    wave_min : Integer, optional
        DESCRIPTION. The default is 0.
    wave_max : Integer, optional
        DESCRIPTION. The default is 0.
    plot : Boolean, optional
        If True generates and shows the plots. The default is True.
    fig_size : Integer, optional
        DESCRIPTION. The default is 12.
    verbose : Boolean, optional
        Print results. The default is True.

    Returns
    -------
    telluric_correction : TYPE
        This is the telluric correction.

    """
             

    print("\n> Obtaining telluric correction using spectrophotometric star...")
    
 
    try:
        wlm=objeto.combined_cube.wavelength
        rss=objeto.rss1
        is_combined_cube = True
    except Exception:
        wlm=objeto.wavelength
        rss=objeto
        is_combined_cube = False
        
    if wave_min == 0 : wave_min=wlm[0]
    if wave_max == 0 : wave_max=wlm[-1]
    
        
    if is_combined_cube:
        if verbose: print("  The given object is a combined cube. Using this cube for extracting the spectrum of the star...")          
        if objeto.combined_cube.seeing == 0: 
            objeto.combined_cube.half_light_spectrum(5, plot=plot, min_wave=wave_min, max_wave=wave_max)
        estrella = objeto.combined_cube.integrated_star_flux                                    
    else:
        if object_rss:     
            if verbose: print("  The given object is a RSS. Using the",high_fibres," fibres with the highest intensity to get the spectrum of the star...")             
            integrated_intensity_sorted=np.argsort(objeto.integrated_fibre)
            intensidad=objeto.intensity_corrected 
            region=[]
            for fibre in range(high_fibres):
                region.append(integrated_intensity_sorted[-1-fibre])
            estrella=np.nansum(intensidad[region], axis=0)   
            #bright_spectrum = objeto.plot_combined_spectrum(list_spectra=fibre_list, median=True, plot=False)
        else:
            if verbose: print("  The given object is a cube. Using this cube for extracting the spectrum of the star...")          
            if objeto.seeing == 0: 
                objeto.half_light_spectrum(5, plot=plot, min_wave=wave_min, max_wave=wave_max)
            estrella = objeto.integrated_star_flux                                    


    if list_of_telluric_ranges[0][0] == 0:
        list_of_telluric_ranges = [ [6150,6245, 6350, 6430], [6720, 6855, 7080,7150], [7080,7150,7500,7580], [7400,7580,7720,7850], [7850,8100,8450,8700]   ]

    telluric_correction = telluric_correction_using_bright_continuum_source(rss, bright_spectrum = estrella,
                                                                            list_of_telluric_ranges = list_of_telluric_ranges,
                                                                            order = order, plot=plot,verbose=verbose)

    if plot: 
        plt.figure(figsize=(fig_size, fig_size/2.5)) 
        if object_rss: 
            print("  Example of telluric correction using fibres",region[0]," and ",region[1],":")                
            plt.plot(wlm, intensidad[region[0]], color="b", alpha=0.3)
            plt.plot(wlm, intensidad[region[0]]*telluric_correction, color="g", alpha=0.5)
            plt.plot(wlm, intensidad[region[1]], color="b", alpha=0.3)
            plt.plot(wlm, intensidad[region[1]]*telluric_correction, color="g", alpha=0.5)
            plt.ylim(np.nanmin(intensidad[region[1]]),np.nanmax(intensidad[region[0]]))   # CHECK THIS AUTOMATICALLY
        else:
            if is_combined_cube :
                print("  Telluric correction applied to this star ("+objeto.combined_cube.object+") :")
            else:
                print("  Telluric correction applied to this star ("+objeto.object+") :")               
            plt.plot(wlm, estrella, color="b", alpha=0.3)
            plt.plot(wlm, estrella*telluric_correction, color="g", alpha=0.5)
            plt.ylim(np.nanmin(estrella),np.nanmax(estrella))          
                            
        plt.axvline(x=wave_min, color='k', linestyle='--')
        plt.axvline(x=wave_max, color='k', linestyle='--')
        plt.xlim(wlm[0]-10,wlm[-1]+10)             
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
                objeto.intensity_corrected[i,:]=objeto.intensity_corrected[i,:] * telluric_correction               
        else:
            if is_combined_cube:
                objeto.combined_cube.integrated_star_flux = objeto.combined_cube.integrated_star_flux * telluric_correction
                for i in range(objeto.combined_cube.n_rows):
                    for j in range(objeto.combined_cube.n_cols):
                        objeto.combined_cube.data[:,i,j] = objeto.combined_cube.data[:,i,j] * telluric_correction               
            else: 
                objeto.integrated_star_flux = objeto.integrated_star_flux * telluric_correction
                for i in range(objeto.n_rows):
                    for j in range(objeto.n_cols):
                        objeto.data[:,i,j] = objeto.data[:,i,j] * telluric_correction                               
    else:
        print("  As apply_tc = False , telluric correction is NOT applied...")

    if is_combined_cube:
        objeto.combined_cube.telluric_correction =   telluric_correction 
    else:
        objeto.telluric_correction =   telluric_correction 
    
    # save file if requested
    if save_telluric_file != "":
        spectrum_to_text_file(wlm, telluric_correction, filename=save_telluric_file, verbose=False) 
        if verbose: print("\n> Telluric correction saved in text file",save_telluric_file," !!")
    return telluric_correction 
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------     
def telluric_correction_using_bright_continuum_source(objeto, save_telluric_file="",
                                                     fibre_list=[-1], high_fibres=20,
                                                     bright_spectrum=[0], odd_number=51,
                                                     list_of_telluric_ranges = [[0]], order = 2,
                                                     plot=True, verbose=True):
    """
    If bright_spectrum is given, for example, an 1D spectrum from a cube in "spec",
    "rss" have to be a valid rss for getting wavelength, 
    use:> telluric_correction_with_bright_continuum_source(EG21_red.rss1, bright_spectrum=spec)
    """
    """
    #TODO

    Parameters
    ----------
    objeto : Object
        Can be a cube that has been read from a fits file or an rss, from which getting the integrated spectrum of a bright source..
    save_telluric_file : String, optional
        This is what the file name will be when saving the telluric file. The default is "".
    fibre_list : Integer List, optional
        DESCRIPTION. The default is [-1].
    high_fibres : Integer, optional
        DESCRIPTION. The default is 20.
    bright_spectrum : Integer List, optional
        DESCRIPTION. The default is [0].
    odd_number : Integer, optional
        DESCRIPTION. The default is 51.
    list_of_telluric_ranges : List of Integer List, optional
        DESCRIPTION. The default is [[0]].
    order : Integer, optional
        DESCRIPTION. The default is 2.
    plot : Boolean, optional
        If True generates and shows the plots. The default is True.
    verbose : Boolean, optional
        Print results. The default is True.

    Returns
    -------
    telluric_correction : TYPE
        This is the telluric correction.

    """

    w = objeto.wavelength
    
    if list_of_telluric_ranges[0][0] == 0: 
        list_of_telluric_ranges = [[6150,6245, 6350, 6430], [6650, 6800, 7080,7150], 
                                   [7080,7150,7440,7580], [7440,7580,7720,7820], [7720,8050,8400,8640]   ]
            
    if verbose: print("\n> Obtaining telluric correction using a bright continuum source...")
    
    if np.nanmedian(bright_spectrum) == 0:
        if fibre_list[0] == -1:
            if verbose:print("  Using the",high_fibres,"fibres with highest intensity for obtaining normalized spectrum of bright source...")
            integrated_intensity_sorted=np.argsort(objeto.integrated_fibre)
            fibre_list=[]    
            for fibre_ in range(high_fibres):
                fibre_list.append(integrated_intensity_sorted[-1-fibre_])
        else:
            if verbose:print("  Using the list of fibres provided for obtaining normalized spectrum of bright source...")
        bright_spectrum = objeto.plot_combined_spectrum(list_spectra=fibre_list, median=True, plot=False)
    else:
        if verbose:print("  Using the normalized spectrum of a bright source provided...")


    if verbose:print("  Deriving median spectrum using a",odd_number,"window...")
    bs_m = signal.medfilt(bright_spectrum, odd_number)

    # Normalizing the spectrum
    #bright = bright_spectrum / np.nanmedian(bright_spectrum)

    #telluric_correction[l]= smooth_med_star[l]/estrella[l]   # LUIGI

    bright = bright_spectrum
    
    vlines=[]
    axvspan=[]
    for t_range in list_of_telluric_ranges:
        vlines.append(t_range[0])
        vlines.append(t_range[3])
        axvspan.append([t_range[1],t_range[2]])
    
        
    if plot: plot_plot(w, [bright_spectrum/np.nanmedian(bright_spectrum),bs_m/np.nanmedian(bright_spectrum)], color=["b","g"], alpha=[0.4,0.8], vlines = vlines, axvspan=axvspan,
                       ymax = np.nanmax(bs_m)/np.nanmedian(bright_spectrum)+0.1,
                       ptitle = "Combined bright spectrum (blue) and median bright spectrum (green)")
     
    
    ntc = np.ones_like(objeto.wavelength)
    
    if verbose: print("  Getting the telluric correction in specified ranges using ",order," order fit to continuum:\n")      

    for t_range in list_of_telluric_ranges:
        
        low_low   = t_range[0]
        low_high  = t_range[1]
        high_low  = t_range[2]
        high_high = t_range[3]
        
        ptitle = "Telluric correction in range "+np.str(low_low)+" - [ "+np.str(low_high)+" , "+np.str(high_low)+" ] - "+np.str(high_high)

        if verbose: print("  - ",ptitle)
      
        #bright = bright_spectrum / np.nanmedian(bright_spectrum[np.where((w > low_low) & (w < high_high))])
        
        w_fit = w[np.where((w > low_low) & (w < high_high))]
        w_fit_low = w[np.where((w > low_low) & (w < low_high))]
        w_fit_range = w[np.where((w >= low_high) & (w <= high_low))]
        w_fit_high = w[np.where((w > high_low) & (w < high_high))]

        y_fit = bright[np.where((w > low_low) & (w < high_high))]
        y_fit_low = bright[np.where((w > low_low) & (w < low_high))]
        y_fit_range = bright[np.where((w >= low_high) & (w <= high_low))]
        y_fit_high = bright[np.where((w > high_low) & (w < high_high))]

        w_fit_cont = np.concatenate((w_fit_low,w_fit_high))
        y_fit_cont = np.concatenate((y_fit_low,y_fit_high))
        
        fit = np.polyfit(w_fit_cont,y_fit_cont, order)
        yfit = np.poly1d(fit)
        y_fitted = yfit(w_fit)
        y_fitted_range=yfit(w_fit_range)
        
        ntc_ =  y_fitted_range / y_fit_range
        
        ntc_low_index = w.tolist().index(w[np.where((w >= low_high) & (w <= high_low))][0])
        ntc_high_index = w.tolist().index(w[np.where((w >= low_high) & (w <= high_low))][-1])

        #ntc = [ntc_(j) for j in range(ntc_low_index,ntc_high_index+1)  ]
        j=0
        for i in range(ntc_low_index,ntc_high_index+1):
            ntc[i] = ntc_[j]
            j=j+1
       
        y_range_corr = y_fit_range * ntc_
        
        y_corr_ = np.concatenate((y_fit_low,y_range_corr))
        y_corr = np.concatenate((y_corr_,y_fit_high))
        
        if plot: plot_plot(w_fit,[y_fit,y_fitted,y_corr], color=["b","r","g"], xmin=low_low-40, xmax=high_high+40,
                           axvspan=[[low_high,high_low]],  vlines=[low_low,low_high,high_low,high_high],
                           ptitle=ptitle, ylabel="Normalized flux")
        
                    
    telluric_correction=np.array([1.0 if x < 1.0 else x for x in ntc])   # Telluric correction should not have corrections < 1.0

    if plot : 
        plot_plot(w,telluric_correction,ptitle="Telluric correction",ylabel="Intensity", #vlines=vlines,
                  axvspan=axvspan,ymax=3, ymin=0.9, hlines=[1])

    # save file if requested
    if save_telluric_file != "":
        spectrum_to_text_file(w, telluric_correction, filename=save_telluric_file, verbose=False) 
        if verbose: print("\n> Telluric correction saved in text file",save_telluric_file," !!")


    return telluric_correction 
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def centroid_of_cube(cube, x0=0,x1=-1,y0=0,y1=-1, box_x=[], box_y=[],
                     step_tracing=25, g2d=True, adr_index_fit=2,
                     kernel_tracing = 5 , adr_clip_fit = 0.3,
                     edgelow=-1, edgehigh=-1,
                     plot=True, log=True, gamma=0.,
                     plot_residua=True, plot_tracing_maps=[], verbose=True) :
    """
    New Routine 20 Nov 2021 for using astropy photutils tasks for centroid

    """
    """
    This finds the centroid of an inputted cube and returns the relevent attributes.

    Parameters #TODO
    ----------
    cube : Cube Object
        This is the Cube object being worked on.
    x0 : TYPE, optional
        DESCRIPTION. The default is 0.
    x1 : TYPE, optional
        DESCRIPTION. The default is -1.
    y0 : TYPE, optional
        DESCRIPTION. The default is 0.
    y1 : TYPE, optional
        DESCRIPTION. The default is -1.
    box_x : List, optional
        When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [].
    box_y : List, optional
        When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [].
    step_tracing : Integer, optional
        DESCRIPTION. The default is 100.
    g2d : Boolean, optional
        If True uses a 2D Gaussian, else doesn't. The default is True.
    adr_index_fit : Integer, optional
        This is the fitted polynomial with highest degree n. The default is 2.
    kernel_tracing : Odd integer, optional
        DESCRIPTION. The default is 0.
    adr_clip_fit: float, optional. The default is 0.4.
        Value over the std to clip
    edgelow : Integer, optional
        This is the lowest value in the wavelength range in terms of pixels. The default is -1.
    edgehigh : Integer, optional
        This is the highest value in the wavelength range in terms of pixels. The default is -1.
    plot : Boolean, optional
        If True generates and shows the plots. The default is True.
    log : Boolean, optional
        If true the map is plotted on a log scale. The default is True.
    gamma : Float, optional
        The value for power log. The default is 0..
    plot_residua : Boolean, optional
        DESCRIPTION. The default is True.
    plot_tracing_maps : TYPE, optional
        If True will plot the tracing maps. The default is [].
    verbose : Boolean, optional
        Print results. The default is True.

    Returns
    -------
    ADR_x_fit : TYPE
        DESCRIPTION.
    ADR_y_fit : TYPE
        DESCRIPTION.
    ADR_x_max : TYPE
        DESCRIPTION.
    ADR_y_max : TYPE
        DESCRIPTION.
    ADR_total : TYPE
        DESCRIPTION.
    x_peaks : TYPE
        DESCRIPTION.
    y_peaks : TYPE
        DESCRIPTION.
    stat_x[3] : TYPE
        DESCRIPTION.
    stat_y[3] : TYPE
        DESCRIPTION.
    stat_total : TYPE
        DESCRIPTION.

    """
    
    
    
    if plot == False: plot_residua = False
    
    if len(box_x) == 2:
        x0 = box_x[0]
        x1 = box_x[1]
    if len(box_y) == 2:
        y0 = box_y[0]
        y1 = box_y[1]
    
    if verbose: 
        if np.nanmedian([x0,x1,y0,y1]) != -0.5:
            print("\n> Computing the centroid of the cube in box [ {:.0f} , {:.0f} ] , [ {:.0f} , {:.0f} ] with the given parameters:".format(x0,x1,y0,y1))
        else:
            print("\n> Computing the centroid of the cube using all spaxels with the given parameters:")
        if g2d:
            print("  step =", step_tracing, ", adr_index_fit =", adr_index_fit, ", kernel_tracing =",kernel_tracing,", adr_clip_fit =",adr_clip_fit,", using a 2D Gaussian fit")
        else:
            print("  step =", step_tracing, ", adr_index_fit =", adr_index_fit, ", kernel_tracing =",kernel_tracing,", adr_clip_fit =",adr_clip_fit,", using the center of mass of the image")
            
    cube_trimmed = copy.deepcopy(cube)
    
    if np.nanmedian([x0,x1,y0,y1]) != -0.5:
        cube_trimmed.data = copy.deepcopy(cube.data[:,y0:y1,x0:x1])
        trimmed = True
    else:
        trimmed = False
    
    w_vector = []
    wc_vector =[]
    xc_vector =[]
    yc_vector =[]
    
    if edgelow == -1 : edgelow = 0
    if edgehigh == -1 : edgehigh = 0
    
    valid_wave_min_index = cube.valid_wave_min_index + edgelow
    valid_wave_max_index = cube.valid_wave_max_index - edgehigh
    
    for i in range(valid_wave_min_index, valid_wave_max_index+step_tracing, step_tracing):
        if i < len(cube.wavelength): w_vector.append(cube.wavelength[i])
        
    show_map = -1
    if len(plot_tracing_maps) > 0:  show_map = 0
        
    for i in range(len(w_vector)-1):    
        wc_vector.append( (w_vector[i] + w_vector[i+1])/2. )
        
        _map_ = cube_trimmed.create_map(line=w_vector[i], w2 = w_vector[i+1], verbose=False)
        
        #Searching for centroid
        if g2d:
            xc, yc = centroid_2dg(_map_[1])
            ptitle = "Fit of order "+np.str(adr_index_fit)+" to centroids computed using a 2D Gaussian fit in steps of "+np.str(step_tracing)+" $\mathrm{\AA}$"
        else:
            xc, yc = centroid_com(_map_[1])          
            ptitle = "Fit of order "+np.str(adr_index_fit)+" to centroids computed using the center of mass in steps of "+np.str(step_tracing)+" $\mathrm{\AA}$"
        
        if show_map > -1 and plot:
            if  w_vector[i] <  plot_tracing_maps[show_map] and  w_vector[i+1] >   plot_tracing_maps[show_map]:   # show map
                #print(xc,yc)
                description="Centroid for "+np.str(plot_tracing_maps[show_map])+" $\mathrm{\AA}$"
                cube_trimmed.plot_map(_map_, description=description, #plot_spaxel_list = [[0.,0.],[1.,1.],[2.,2.],[xc,yc]], 
                                      plot_spaxel_list = [[xc+0.5,yc+0.5]], log=log, gamma=gamma,
                                      g2d=g2d,
                                      verbose=False, trimmed=trimmed)    # FORO
                if verbose: print('  Centroid at {} A found in spaxel [ {:.2f} , {:.2f} ]  =  [ {:.2f}" , {:.2f}" ]'.format(plot_tracing_maps[show_map], xc,yc,xc*cube.pixel_size_arcsec,yc*cube.pixel_size_arcsec))            
                show_map = show_map +1
                if show_map == len(plot_tracing_maps) : show_map = -1
                       
        xc_vector.append(xc)
        yc_vector.append(yc)
    
        
    fit_x, pp, x_peaks_fit, fxx, wc_vector_c, xc_vector_c = fit_clip(wc_vector,xc_vector, 
                                        clip = adr_clip_fit,
                                        index_fit = adr_index_fit,
                                        kernel = kernel_tracing,
                                        plot=False, verbose = False)
    
    x_peaks=pp(cube.wavelength) +x0

    fit_y, pp, y_peaks_fit, fyy, wc_vector_c, yc_vector_c = fit_clip(wc_vector,yc_vector, 
                                        clip = adr_clip_fit,
                                        index_fit = adr_index_fit,
                                        kernel = kernel_tracing,
                                        plot=False, verbose = False)

    y_peaks=pp(cube.wavelength) +y0 
        
    xc_vector= (xc_vector - np.nanmedian(xc_vector)) *cube.pixel_size_arcsec
    yc_vector= (yc_vector - np.nanmedian(yc_vector)) *cube.pixel_size_arcsec
    
    ADR_x_fit, pp, fx, fxx, wcx, xxc = fit_clip(wc_vector,xc_vector, 
                                        clip = adr_clip_fit,
                                        index_fit = adr_index_fit,
                                        kernel = kernel_tracing,
                                        plot=False, verbose = False)        

    ADR_y_fit, pp, fy, fyy, wcy, yyc = fit_clip(wc_vector,yc_vector, 
                                        clip = adr_clip_fit,
                                        index_fit = adr_index_fit,
                                        kernel = kernel_tracing,
                                        plot=False, verbose = False) 
      
    vlines = [cube.wavelength[valid_wave_min_index], cube.wavelength[valid_wave_max_index]]
    if plot: 
        plot_plot([wc_vector,wc_vector,  wcx,   wcy,       wc_vector,wc_vector], 
                  [xc_vector,yc_vector,  xxc,   yyc,      fx,fy], 
                  psym=["+", "o", "+","o", "-","-"], color=["k", "k", "r", "b",  "g","brown"], 
                  alpha=[0.4,0.4, 1,  1, 1,1], label=["RA", "Dec", "RA clip",  "Dec clip",  "RA fit", "Dec fit"],
                  xmin=cube.wavelength[0],xmax=cube.wavelength[-1], vlines=vlines, markersize=[10,6,10,6,0,0],
                  ylabel="$\Delta$ offset [arcsec]",ptitle=ptitle, hlines=[0], frameon=True, 
                  ymin = np.nanmin([np.nanmin(xc_vector), np.nanmin(yc_vector)]),    
                  ymax = np.nanmax([np.nanmax(xc_vector), np.nanmax(yc_vector)]))


    ADR_x_max=np.nanmax(fx)-np.nanmin(fx)                    ##### USING FITS
    ADR_y_max=np.nanmax(fy)-np.nanmin(fy)
    ADR_total = np.sqrt(ADR_x_max**2 + ADR_y_max**2)   

    stat_x=basic_statistics(xxc-fxx, verbose=False, return_data=True)
    stat_y=basic_statistics(yyc-fyy, verbose=False, return_data=True)
    stat_total = np.sqrt(stat_x[3]**2 + stat_y[3]**2)  
    
    if verbose: print('  ADR variation in valid interval using fit : RA = {:.3f}" , Dec = {:.3f}" , total = {:.3f}"  that is {:.0f}% of a spaxel'.format(ADR_x_max, ADR_y_max, ADR_total, ADR_total*100./cube.pixel_size_arcsec))

    if plot_residua: 
        plot_plot(wc_vector, [xc_vector-fx,yc_vector-fy], color=["r", "k"], alpha=[1,1], ymin=-0.1, ymax=0.1,
                  hlines=[-0.08,-0.06,-0.04,-0.02,0,0,0,0,0.02,0.04,0.06,0.08], 
                  xmin=cube.wavelength[0],xmax=cube.wavelength[-1],frameon=True, label=["RA residua","Dec residua"],
                  vlines=vlines, ylabel="$\Delta$ offset [arcsec]",ptitle="Residua of the fit to the centroid fit")

    if verbose: print('  Standard deviation of residua :             RA = {:.3f}" , Dec = {:.3f}" , total = {:.3f}"  that is {:.0f}% of a spaxel'.format(stat_x[3], stat_y[3], stat_total, stat_total*100./cube.pixel_size_arcsec))
    

    return ADR_x_fit, ADR_y_fit, ADR_x_max, ADR_y_max, ADR_total, x_peaks, y_peaks, stat_x[3], stat_y[3], stat_total, [wc_vector, xc_vector, yc_vector]  
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def scale_cubes_using_common_region(cube_list, flux_ratios=[], min_wave = 0, max_wave = 0,
                                    apply_scale = True, verbose=True, plot=False):    #SCORE
    """
    This scales the cubes based on a common region the cubes have.

    Parameters
    ----------
    cube_list : Cube List
        This is a list of Cube Objects.
    flux_ratios : List, optional
        This is a list of flux ratios between the cubes. The default is [].
    min_wave : Integer, optional
        The minimum wavelength passed through the mask. The default is 0.
    max_wave : Integer, optional
        The maximum wavelength passed through the mask. The default is 0.
    apply_scale : Boolean, optional
        DESCRIPTION. The default is True. #TODO
    verbose : Boolean, optional
        Print results. The default is True.
    plot : Boolean, optional
        If True generates and shows the plots. The default is False.

    Returns
    -------
    object_list : Object List
        A list of Objects.

    """

    
    if verbose: print("\n> Scaling intensities of the cubes using the integrated value of their common region...")
    # Check if cube_list are fits or objects
    object_name_list =[]
    try: 
        try_read=cube_list[0]+"  "      # This will work ONLY if cube_list are strings (fits names)
        if verbose: print("  - Reading the cubes from the list of fits files provided:"+try_read[-2:-1])
        object_list = []
        for i in range(len(cube_list)):
            if i< 9:
                name = "cube_0"+np.str(i+1)
            else:
                name = "cube_"+np.str(i+1)
            object_name_list.append(name)
            exec(name+"=read_cube(cube_list[i])")
            exec("object_list.append("+name+")")
        print(" ")    
    except Exception:
        object_list = cube_list   
        for i in range(len(cube_list)):
            object_name_list.append(cube_list[i].object)

    if len(flux_ratios) == 0:  # flux_ratios are not given
        if verbose: 
            if np.nanmedian(object_list[0].data) < 1E-6 :
                print("  - Cubes are flux calibrated. Creating mask for each cube...")
            else:
                print("  - Cubes are NOT flux calibrated. Creating mask and scaling with the cube with the largest integrated value...")
                
        # Create a mask for each cube
        list_of_masks = []
        for i in range(len(cube_list)):                 
            object_list[i].mask_cube(min_wave = min_wave, max_wave = max_wave)
            list_of_masks.append(object_list[i].mask)
        
        # Get mask combining all mask    
        mask = np.median(list_of_masks, axis=0)   
        if plot:      
            object_list[0].plot_map(mapa=mask, fcal=False, cmap="binary", description="mask", 
                            barlabel=" ", vmin=0.,vmax=1., contours=False)
    
        # Compute integrated flux within the good values in all cubes    
        integrated_flux_region = []
        for i in range(len(cube_list)):   
            im_mask = object_list[i].integrated_map * mask
            ifr_ = np.nansum(im_mask)
            integrated_flux_region.append(ifr_)
            
        # If data are NOT flux calibrated, it does not matter the exposition time, just scale     
        # Find the maximum value and scale!
        max_irf = np.nanmax(integrated_flux_region)  
        flux_ratios =  integrated_flux_region /     max_irf
    
        if verbose:
            print("  - Cube  Name                               Total valid integrated flux      Flux ratio")
            for i in range(len(cube_list)): 
                    print("    {:2}   {:30}            {:.4}                  {:.4}".format(i+1,object_name_list[i],integrated_flux_region[i],flux_ratios[i]) )
    else:
        if verbose: 
            print("  - Scale values provided !")
            print("  - Cube  Name                             Flux ratio provided")
            for i in range(len(cube_list)): 
                    print("    {:2}   {:30}                {:.4}".format(i+1,object_name_list[i],flux_ratios[i]) )
    
    if apply_scale:
        if verbose: print("  - Applying flux ratios to cubes...")
        for i in range(len(cube_list)):
            object_list[i].scale_flux = flux_ratios[i]
            _cube_ = object_list[i]
            _data_= _cube_.data / flux_ratios[i]
            object_list[i].data = _data_
            _cube_.integrated_map = np.sum(_cube_.data[np.searchsorted(_cube_.wavelength, min_wave):np.searchsorted(_cube_.wavelength, max_wave)],axis=0)            

    return object_list
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def build_combined_cube(cube_list, obj_name="", description="", fits_file = "", path="",
                        scale_cubes_using_integflux= True, flux_ratios = [], apply_scale = True, 
                        edgelow=30, edgehigh=30,
                        ADR=True, ADR_cc = False, jump =-1, pk = "", 
                        ADR_x_fit_list=[], ADR_y_fit_list=[],  force_ADR= False,
                        half_size_for_centroid = 10, box_x=[0,-1],box_y=[0,-1],  
                        adr_index_fit = 2, g2d=False, step_tracing=25, 
                        kernel_tracing = 5, adr_clip_fit=0.3,
                        plot_tracing_maps=[],                       
                        trim_cube = True,  trim_values =[], remove_spaxels_not_fully_covered = True,                                              
                        plot=True, plot_weight= True, plot_spectra=True, 
                        verbose=True, say_making_combined_cube = True):
    """
    This function builds a cube that is a combination of individual cubes.

    Parameters #TODO
    ----------
    cube_list : Cube List
        This is a list of Cube Objects.
    obj_name : String, optional
        DESCRIPTION. The default is "".
    description : String, optional
        This is the description of the cube. The default is "".
    fits_file : String, optional
        The name of the fits file, if left "" will not save. The default is "".
    path : String, optional
        Where you want to save the combinded cube. The default is "".
    scale_cubes_using_integflux : Boolean, optional
        DESCRIPTION. The default is True.
    flux_ratios : List, optional
        This is a list of flux ratios between the cubes. The default is [].
    apply_scale : Boolean, optional
        DESCRIPTION. The default is True.
    edgelow : Integer, optional
        This is the lowest value in the wavelength range in terms of pixels. The default is 30.
    edgehigh : Integer, optional
        This is the highest value in the wavelength range in terms of pixels. The default is 30.
    ADR : Boolean, optional
        If True will correct for ADR even considoring a small correction. The default is True.
    ADR_cc : Boolean, optional
        DESCRIPTION. The default is False.
    jump : Integer, optional
        If a positive number partitions the wavelengths with step size jump, if -1 will not partition. The default is -1.
    pk : TYPE, optional
        DESCRIPTION. The default is "".
    ADR_x_fit_list : List, optional
        This is a list of ADR x fits. The default is [].
    ADR_y_fit_list : List, optional
        This is a list of ADR y fits. The default is [].
    force_ADR : Boolean, optional
        If True will correct for ADR even considoring a small correction. The default is False.
    half_size_for_centroid : Integer, optional
        This is half the length/width of the box. The default is 10.
    box_x : Integer List, optional
        When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [0,-1].
    box_y : Integer List, optional
        When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [0,-1].
    adr_index_fit : Integer, optional
        This is the fitted polynomial with highest degree n. The default is 2.
    g2d : Boolean, optional
        If True uses a 2D Gaussian, else doesn't. The default is False.
    step_tracing : Integer, optional
        DESCRIPTION. The default is 100.
    kernel_tracing : Integer, optional
        DESCRIPTION. The default is 0.
    plot_tracing_maps : List, optional
        If True will plot the tracing maps. The default is [].
    trim_cube : Boolean, optional
        DESCRIPTION. The default is True.
    trim_values : List, optional
        DESCRIPTION. The default is [].
    remove_spaxels_not_fully_covered : Boolean, optional
        DESCRIPTION. The default is True.
    plot : Boolean, optional
        If True generates and shows the plots. The default is True.
    plot_weight : Boolean, optional
        If True will plot the weight. The default is True.
    plot_spectra : Boolean, optional
        If True will plot the spectra. The default is True.
    verbose : Boolean, optional
        Print results. The default is True.
    say_making_combined_cube : Boolean, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    combined_cube : Cube Object
        This is a cube combined of multiple cubes.

    """
    
                            
    if say_making_combined_cube: print("\n> Making combined cube ...")
    n_files = len(cube_list)
    # Check if cube_list are fits or objects
    object_name_list =[]
    try:
        try_read=cube_list[0]+"  "
        if verbose: print(" - Reading the cubes from the list of fits files provided:"+try_read[-2:-1])
        object_list = []
        for i in range(n_files):
            if i< 9:
                name = "cube_0"+np.str(i+1)
            else:
                name = "cube_"+np.str(i+1)
            object_name_list.append(name)
            exec(name+"=read_cube(cube_list[i])")
            exec("object_list.append("+name+")")
            print(" ") 
    except Exception:
        object_list = cube_list   
        for i in range(n_files):
            object_name_list.append(cube_list[i].object)
    
    cube_aligned_object = object_list
    print("\n> Checking individual cubes: ")
    print("  Cube      name                          RA_centre           DEC_centre     Pix Size  Kernel Size   n_cols  n_rows")
    for i in range(n_files):
        print("    {:2}    {:25}   {:18.12f} {:18.12f}    {:4.1f}      {:5.2f}       {:4}    {:4}".format(i+1, cube_aligned_object[i].object, 
              cube_aligned_object[i].RA_centre_deg, cube_aligned_object[i].DEC_centre_deg, cube_aligned_object[i].pixel_size_arcsec, 
              cube_aligned_object[i].kernel_size_arcsec, cube_aligned_object[i].n_cols, cube_aligned_object[i].n_rows))
      
    # Check that RA_centre, DEC_centre, pix_size and kernel_size are THE SAME in all input cubes
    do_not_combine = False
    for _property_ in ["RA_centre_deg", "DEC_centre_deg", "pixel_size_arcsec", "kernel_size_arcsec", "n_cols", "n_rows"]:
        property_values = [_property_]
        for i in range(n_files):               
            exec("property_values.append(cube_aligned_object["+np.str(i)+"]."+_property_+")")
        #print(property_values)
        if np.nanvar(property_values[1:-1]) > 1E-20:
            print(" - Property {} has DIFFERENT values !!!".format(_property_))
            print("   Variance of the data = ",np.nanvar(property_values[1:-1])) 
            print("   Values =  ",property_values)
            do_not_combine = True
     
    if do_not_combine:
        print("\n> Cubes CANNOT be combined as they don't have the same basic properties !!!")
    else:
        print("\n> Cubes CAN be combined as they DO have the same basic properties !!!")

        if pk == "":
            pixel_size_arcsec = cube_aligned_object[0].pixel_size_arcsec
            kernel_size_arcsec = cube_aligned_object[0].kernel_size_arcsec
            pk = "_"+str(int(pixel_size_arcsec))+"p"+str(int((abs(pixel_size_arcsec)-abs(int(pixel_size_arcsec)))*10))+"_"+str(int(kernel_size_arcsec))+"k"+str(int(abs(kernel_size_arcsec*100))-int(kernel_size_arcsec)*100)
    
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
        combined_cube.history=[]
        
        combined_cube.number_of_combined_files  = n_files
        
        #delattr(combined_cube, "ADR_total")
               
        if obj_name != "":
            combined_cube.object= obj_name
        if description == "":
            combined_cube.description = combined_cube.object+" - COMBINED CUBE"
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
            fcal=False
        else:    
            combined_cube.flux_calibration=cube_aligned_object[0].flux_calibration
            print("  - Flux calibration included!")
            combined_cube.history.append("- Flux calibration included")
            fcal=True
                      
        if scale_cubes_using_integflux:
            cube_list =   scale_cubes_using_common_region(cube_list, flux_ratios=flux_ratios)             
        else:
            print("  - No scaling of the cubes using integrated flux requested")
            combined_cube.history.append("- No scaling of the cubes using integrated flux requested")
        
        _data_   = []
        _PA_     = []
        _weight_ = []

        print ("\n> Combining data cubes...")        
        for i in range(n_files): 
            _data_.append(cube_aligned_object[i].data)
            _PA_.append(cube_aligned_object[i].PA)
            _weight_.append(cube_aligned_object[i].weight)
            
        combined_cube.data = np.nanmedian(_data_, axis = 0)
        combined_cube.PA = np.mean(_PA_)
        combined_cube.weight = np.nanmean(_weight_ , axis = 0)
        combined_cube.offsets_files_position = ""
        
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
            box_x_centroid=box_x
            box_y_centroid=box_y
            if np.nanmedian(box_x+box_y) == -0.5 and half_size_for_centroid > 0:            
                combined_cube.get_integrated_map()        
                #if verbose: print("\n> Peak of emission found in [ {} , {} ]".format(combined_cube.max_x,combined_cube.max_y))
                if verbose: print("\n> As requested, using a box centered at the peak of emission, [ {} , {} ], and width +-{} spaxels for tracing...".format(combined_cube.max_x,combined_cube.max_y,half_size_for_centroid))
    
                #if verbose: print("  As requested, using a box centered there and width +-{} spaxels for tracing...".format(half_size_for_centroid))
                box_x_centroid = [combined_cube.max_x-half_size_for_centroid,combined_cube.max_x+half_size_for_centroid]
                box_y_centroid = [combined_cube.max_y-half_size_for_centroid,combined_cube.max_y+half_size_for_centroid]    
            
            if ADR_cc :
                check_ADR = False
            else:
                check_ADR = True
            
    
                combined_cube.trace_peak(box_x=box_x_centroid, box_y=box_y_centroid, edgelow=edgelow, edgehigh =edgehigh, 
                                         plot=plot,adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing, 
                                         kernel_tracing = kernel_tracing, adr_clip_fit=adr_clip_fit,
                                         plot_tracing_maps=plot_tracing_maps,  check_ADR=check_ADR)
                        
        # ADR correction to the combined cube    
        if ADR_cc :
            combined_cube.adrcor = True
            combined_cube.ADR_correction(RSS, plot=plot, jump=jump, method="old", force_ADR=force_ADR,
                                         remove_spaxels_not_fully_covered=remove_spaxels_not_fully_covered)
            if ADR:
                combined_cube.trace_peak(box_x=box_x_centroid, box_y=box_y_centroid, 
                                         edgelow=edgelow, edgehigh =edgehigh, 
                                         check_ADR=True, 
                                         adr_index_fit=adr_index_fit, g2d=g2d,
                                         step_tracing=step_tracing, adr_clip_fit=adr_clip_fit,
                                         plot_tracing_maps=plot_tracing_maps, plot=plot)
                                                   
        combined_cube.get_integrated_map(box_x=box_x, box_y=box_y, fcal=fcal, plot=plot, plot_spectra=plot_spectra, 
                                         plot_centroid=True, g2d=g2d, kernel_tracing = kernel_tracing)


        # Trimming combined cube if requested or needed
        combined_cube.trim_cube(trim_cube=trim_cube, trim_values=trim_values, 
                                half_size_for_centroid =half_size_for_centroid, ADR=ADR, 
                                adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing, 
                                adr_clip_fit=adr_clip_fit, plot_tracing_maps=plot_tracing_maps,
                                remove_spaxels_not_fully_covered = remove_spaxels_not_fully_covered,
                                box_x=box_x, box_y=box_y, edgelow=edgelow, edgehigh=edgehigh, 
                                plot_weight = plot_weight, fcal=fcal, plot=plot, plot_spectra= plot_spectra)
          
        # Computing total exposition time of combined cube  
        combined_cube.total_exptime = 0.
        combined_cube.exptimes=[]
        combined_cube.rss_file_list=[]
        for i in range(n_files):
            combined_cube.total_exptime = combined_cube.total_exptime + cube_aligned_object[i].total_exptime
            combined_cube.exptimes.append(cube_aligned_object[i].total_exptime)
            combined_cube.rss_file_list.append(cube_aligned_object[i].rss_file)
        
        print("\n> Total exposition time = ",combined_cube.total_exptime,"seconds adding the", n_files,"files")
        
        if combined_cube.total_exptime / n_files == combined_cube.exptimes[0] :
            print("  All {} cubes have the same exposition time, {} s".format(n_files, combined_cube.exptimes[0]))
        else:
            print("  The individual cubes have different exposition times.")
                
        
        if np.nanmedian(combined_cube.offsets_files) != 0:      
            offsets_print="[ "
            for i in range(len(combined_cube.offsets_files)):
                offsets_print=offsets_print+np.str(combined_cube.offsets_files[i][0])+" , "+np.str(combined_cube.offsets_files[i][1])+" , "
            offsets_print = offsets_print[:-2]+"]"
            print("\n  offsets = ",offsets_print) 
         

        if len(ADR_x_fit_list) > 0:      
            print("\n  ADR_x_fit_list = ",ADR_x_fit_list)
            print("\n  ADR_y_fit_list = ",ADR_y_fit_list)

            
        if fits_file == "": 
            print("\n> As requested, the combined cube will not be saved to a fits file")
        else:    
            #print("\n> Saving combined cube to a fits file ...")                    
            if fits_file == "auto":
                fits_file = path+obj_name + "_"+combined_cube.grating+pk+ "_combining_"+np.str(n_files)+"_cubes.fits"                          
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
def read_cube(filename, path="", description="", half_size_for_centroid = 10, 
              valid_wave_min = 0, valid_wave_max = 0, edgelow=50,edgehigh=50, 
              g2d=False, step_tracing=25, adr_index_fit=2, 
              kernel_tracing = 5, adr_clip_fit=0.3,
              plot = False, verbose = True, plot_spectra = False,
              print_summary = True,
              text_intro ="\n> Reading datacube from fits file:" ):
    """
    This function reads the cube.

    Parameters #TODO
    ----------
    filename : String
        The name of the file.
    description : String, optional
        This is the description of the cube. The default is "".
    half_size_for_centroid : Integer, optional
        This is half the length/width of the box. The default is 10.
    valid_wave_min : Integer, optional
        DESCRIPTION. The default is 0.
    valid_wave_max : Integer, optional
        DESCRIPTION. The default is 0.
    edgelow : Integer, optional
        This is the lowest value in the wavelength range in terms of pixels. The default is 50.
    edgehigh : Integer, optional
        This is the highest value in the wavelength range in terms of pixels. The default is 50.
    g2d : Boolean, optional
        If True uses a 2D Gaussian, else doesn't. The default is False.
    step_tracing : Integer, optional
        DESCRIPTION. The default is 100.
    adr_index_fit : Integer, optional
        This is the fitted polynomial with highest degree n. The default is 2.
    kernel_tracing : Integer, optional
        DESCRIPTION. The default is 0.
    plot : Boolean, optional
        If True generates and shows the plots. The default is False.
    verbose : Boolean, optional
        Print results. The default is True.
    plot_spectra : Boolean, optional
        If True will plot the spectra. The default is False.
    print_summary : Boolean, optional
        DESCRIPTION. The default is True.
    text_intro : TYPE, optional
        DESCRIPTION. The default is "\n> Reading datacube from fits file:".

    Returns
    -------
    cube : Cube Object
        This is the inputted Cube Object.

    """
    
    
    errors = 0
    
    if verbose: print(text_intro)
    
    if path != "": filename =full_path(filename,path)
    if verbose: print('  "'+filename+'"', "...")
    cube_fits_file = fits.open(filename)  # Open file
    
    objeto = cube_fits_file[0].header['OBJECT']
    if description == "": description = objeto + " - CUBE"
    grating = cube_fits_file[0].header['GRATID']
    
    total_exptime=  cube_fits_file[0].header['TOTALEXP']  
    exptimes_ =  cube_fits_file[0].header['EXPTIMES'].strip('][').split(',')
    exptimes = []
    for j in range(len(exptimes_)):
        exptimes.append(float(exptimes_[j]))
    number_of_combined_files  = cube_fits_file[0].header['COFILES']          
    #fcal = cube_fits_file[0].header['FCAL']

    filename = cube_fits_file[0].header['FILE_OUT'] 
    RACEN   =  cube_fits_file[0].header['RACEN']                                                   
    DECCEN  =  cube_fits_file[0].header['DECCEN'] 
    centre_deg =[RACEN,DECCEN]
    pixel_size = cube_fits_file[0].header['PIXsize']
    kernel_size= cube_fits_file[0].header['KERsize']
    n_cols   =  cube_fits_file[0].header['NCOLS']                                               
    n_rows   =  cube_fits_file[0].header['NROWS']                                                
    PA      =   cube_fits_file[0].header['PA']                                               

    CRVAL3  =  cube_fits_file[0].header['CRVAL3']    # 4695.841684048                                                  
    CDELT3  =  cube_fits_file[0].header['CDELT3']    # 1.038189521346                                                  
    CRPIX3  =  cube_fits_file[0].header['CRPIX3']    #         1024.0  
    CRVAL1_CDELT1_CRPIX1 = [CRVAL3,CDELT3,CRPIX3]
    n_wave  =  cube_fits_file[0].header['NAXIS3']
    
    adrcor = cube_fits_file[0].header['ADRCOR']
    
    number_of_combined_files = cube_fits_file[0].header['COFILES']
    
    ADR_x_fit =[]
    ADR_y_fit =[]

    try:
        ADR_x_fit_ = cube_fits_file[0].header['ADRXFIT'].split(',')
        for j in range(len(ADR_x_fit_)):
                    ADR_x_fit.append(float(ADR_x_fit_[j]))    
        ADR_y_fit_ = cube_fits_file[0].header['ADRYFIT'].split(',')
        for j in range(len(ADR_y_fit_)):
                    ADR_y_fit.append(float(ADR_y_fit_[j]))    
        adr_index_fit=len(ADR_y_fit) -1
    except Exception:
        errors=errors+1
 
    rss_file_list = []
    offsets_files_position = cube_fits_file[0].header['OFF_POS']
    offsets_files =[]
    offsets_files_ =  cube_fits_file[0].header['OFFSETS'].split(',')           
    for j in range(len(offsets_files_)):
        valor = offsets_files_[j].split(' ') 
        offset_=[]
        p = 0
        for k in range(len(valor)):
            try:            
                offset_.append(np.float(valor[k]))
            except Exception:
                p = p + 1
                #print j,k,valor[k], "no es float"
        offsets_files.append(offset_)        
        
    if  number_of_combined_files > 1:   
        for i in range(number_of_combined_files):
            if i < 10 :
                head = "RSS_0"+np.str(i+1)
            else:
                head = "RSS_"+np.str(i+1)
            rss_file_list.append(cube_fits_file[0].header[head])
    else: 
        rss_file_list.append(cube_fits_file[0].header["RSS_01"])
           
    wavelength = np.array([0.] * n_wave)    
    wavelength[np.int(CRPIX3)-1] = CRVAL3
    for i in range(np.int(CRPIX3)-2,-1,-1):
        wavelength[i] = wavelength[i+1] - CDELT3
    for i in range(np.int(CRPIX3),n_wave):
         wavelength[i] = wavelength[i-1] + CDELT3
     
    if valid_wave_min == 0 : valid_wave_min = cube_fits_file[0].header["V_W_MIN"]  
    if valid_wave_max == 0 : valid_wave_max = cube_fits_file[0].header["V_W_MAX"] 
     
    cube = Interpolated_cube(filename, pixel_size, kernel_size, plot=False, verbose=verbose,
                             read_fits_cube=True, zeros=True,
                             ADR_x_fit=np.array(ADR_x_fit),ADR_y_fit=np.array(ADR_y_fit),
                             objeto = objeto, description = description,
                             n_cols = n_cols, n_rows=n_rows, PA=PA,
                             wavelength = wavelength, n_wave = n_wave, 
                             total_exptime = total_exptime, valid_wave_min = valid_wave_min, valid_wave_max=valid_wave_max,
                             CRVAL1_CDELT1_CRPIX1 = CRVAL1_CDELT1_CRPIX1,
                             grating = grating, centre_deg=centre_deg, number_of_combined_files=number_of_combined_files)
    
    cube.exptimes=exptimes
    cube.data = cube_fits_file[0].data   
    if half_size_for_centroid > 0:
        box_x,box_y = cube.box_for_centroid(half_size_for_centroid=half_size_for_centroid, verbose=verbose)
    else:
        box_x = [0,-1]
        box_y = [0,-1]
    cube.trace_peak(box_x=box_x, box_y=box_y, plot=plot, edgelow=edgelow,edgehigh=edgehigh, 
                    adr_index_fit=adr_index_fit, g2d=g2d, 
                    step_tracing = step_tracing, kernel_tracing = kernel_tracing, adr_clip_fit=adr_clip_fit,
                    verbose =False)
    cube.get_integrated_map(plot=plot, plot_spectra=plot_spectra, verbose=verbose, plot_centroid=True, g2d=g2d) #,fcal=fcal, box_x=box_x, box_y=box_y)
    # For calibration stars, we get an integrated star flux and a seeing
    cube.integrated_star_flux = np.zeros_like(cube.wavelength) 
    cube.offsets_files = offsets_files
    cube.offsets_files_position = offsets_files_position
    cube.adrcor = adrcor
    cube.rss_file = filename
    cube.rss_file_list = [filename]    
    
    if number_of_combined_files > 1 and verbose:
        print("\n> This cube was created using the following rss files:")
        for i in range(number_of_combined_files): print(" ",rss_file_list[i])
        print_offsets = "  Offsets used : "
        for i in range(number_of_combined_files-1):
            print_offsets=print_offsets+(np.str(offsets_files[i]))
            if i <number_of_combined_files-2 : print_offsets=print_offsets+ " , "
        print(print_offsets)

    if verbose and print_summary:
        print("\n> Summary of reading cube :")        
        print("  Object          = ",cube.object)
        print("  Description     = ",cube.description)
        print("  Centre:  RA     = ",cube.RA_centre_deg, "Deg")
        print("          DEC     = ",cube.DEC_centre_deg, "Deg")
        print("  PA              = ",np.round(cube.PA,2), "Deg")
        print("  Size [pix]      = ",cube.n_cols," x ", cube.n_rows)
        print("  Size [arcsec]   = ",np.round(cube.n_cols*cube.pixel_size_arcsec,1)," x ", np.round(cube.n_rows*cube.pixel_size_arcsec,1))
        print("  Pix size        = ",cube.pixel_size_arcsec, " arcsec")
        print("  Total exp time  = ",total_exptime, "s")
        if number_of_combined_files > 1 : print("  Files combined  = ",cube.number_of_combined_files)
        print("  Med exp time    = ",np.nanmedian(cube.exptimes),"s")
        print("  ADR corrected   = ",cube.adrcor)
    #    print "  Offsets used   = ",self.offsets_files
        print("  Wave Range      =  [",np.round(cube.wavelength[0],2),",",np.round(cube.wavelength[-1],2),"]")
        print("  Wave Resolution =  {:.3} A/pix".format(CDELT3))
        print("  Valid wav range =  [",np.round(valid_wave_min,2),",",np.round(valid_wave_max,2),"]")
        
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