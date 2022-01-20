#!/usr/bin/python
# -*- coding: utf-8 -*-
# # PyKOALA: KOALA data processing and analysis 
# by Angel Lopez-Sanchez, Yago Ascasibar, Pablo Corcho-Caballero
# Extra work by Ben Lawson (MQ PACE student)
# Plus Taylah Beard and Matt Owers (sky substraction)
# Documenting: Nathan Pidcock, Giacomo Biviano, Jamila Scammill, Diana Dalae, Barr Perez
version="Version 1.07 - 20 January 2022 - Really last one before breaking code"
# This is Python 3.7
# To convert to Python 3 run this in a command line :
# cp PyKOALA_2021_02_02.py PyKOALA_2021_02_02_P3.py
# 2to3 -w PyKOALA_2021_02_02_P3.py
# Edit the file replacing:
#                 exec('print("    {:2}       {:18.12f}   {:18.12f}        {:4.1f}        {:5.2f}".format(i+1, '+cube_aligned_object[i]+'.RA_centre_deg,'+cube_aligned_object[i]+'.DEC_centre_deg,'+cube_aligned_object[i]+'.pixel_size_arcsec,'+cube_aligned_object[i]+'.kernel_size_arcsec))')


# -----------------------------------------------------------------------------
# Import Python routines
# -----------------------------------------------------------------------------
from astropy.io import fits
from astropy.wcs import WCS

from synphot import observation   ########from pysynphot import observation
#from synphot import spectrum      ########from pysynphot import spectrum
from synphot import SourceSpectrum, SpectralElement         # conda install synphot -c http://ssb.stsci.edu/astroconda  or pip install synphot 
from synphot.models import Empirical1D
from photutils.centroids import centroid_com, centroid_2dg  # pip install photutils     or    conda install -c conda-forge photutils

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys

from scipy import interpolate, signal, optimize
from scipy.optimize import curve_fit
import scipy.signal as sig
#from scipy.optimize import leastsq

#from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.ndimage.interpolation import shift
#import astropy.table

import datetime
import copy

import glob
from astropy.io import fits as pyfits 

# Disable some annoying warnings
import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.CRITICAL)
#warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
# -----------------------------------------------------------------------------
# Define constants
# -----------------------------------------------------------------------------
pc=3.086E18    # pc in cm
C =299792.458  # c in km/s
nebula_lines = [6300.3, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 7751.1, 9068.9] 
red_gratings = ["385R","1000R","2000R", "1000I", "1700D","1700I"]
blue_gratings = ["580V" , "1500V" ,"1700B" , "3200B" , "2500V"]   


edge_fibres_100 = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                   34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951,
                   952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964,
                   965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977,
                   978, 979, 980, 981, 982, 983, 984, 985]

fibres_best_sky_100 = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                       34, 35, 36, 37, 38, 39, 40, 41, 42,
                       795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809,
                       944, 945, 946, 947, 948, 949, 950, 951,
                       952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964,
                       965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977,
                       978, 979, 980, 981, 982, 983, 984, 985]
              
# -----------------------------------------------------------------------------
# Define COLOR scales
# -----------------------------------------------------------------------------
fuego_color_map = colors.LinearSegmentedColormap.from_list("fuego", ((0.25, 0, 0),  (0.5,0,0),    (1, 0, 0), (1, 0.5, 0), (1, 0.75, 0), (1, 1, 0), (1, 1, 1)), N=256, gamma=1.0)
fuego_color_map.set_bad('lightgray')  #('black')
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
        self.intensity_corrected=self.intensity
        self.variance = np.zeros_like(self.intensity)
        self.RA_centre_deg = 0.
        self.DEC_centre_deg = 0.
        self.offset_RA_arcsec = np.zeros((0))
        self.offset_DEC_arcsec = np.zeros_like(self.offset_RA_arcsec)                
        self.ALIGNED_RA_centre_deg = 0.    
        self.ALIGNED_DEC_centre_deg = 0.  
        self.sky_emission=[]
        #self.throughput = np.ones((0))   
# -----------------------------------------------------------------------------           
# -----------------------------------------------------------------------------
    def read_mask_from_fits_file(self, mask=[[]], mask_file="", no_nans=True, plot = True, 
                  verbose= True, include_history=False):
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
            print(" ",mask_file)
            ftf = fits.open(mask_file)
            self.mask = ftf[0].data  
            if include_history:
                self.history.append("- Mask read from fits file")
                self.history.append("  "+mask_file)
        else:
            print("\n> Reading the mask stored in Python variable...")
            self.mask=mask
            if include_history: self.history.append("- Mask read using a Python variable")
        if  no_nans:   
            print("  We are considering that the mask does not have 'nans' but 0s in the bad pixels")
        else:
            print("  We are considering that the mask DOES have 'nans' in the bad pixels")
            
        # Check edges
        suma_good_pixels=np.nansum(self.mask, axis=0) 
        nspec = self.n_spectra    
        w=self.wavelength
        # Left edge
        found = 0 
        j = 0
        if verbose : print("\n- Checking the left edge of the ccd...")
        while found < 1:
            if suma_good_pixels[j] == nspec:
                first_good_pixel = j
                found = 2
            else:
                j=j+1
        if verbose: print("  First good pixels is ",first_good_pixel,", that corresponds to ",w[first_good_pixel],"A") 
        
        if plot:
            ptitle = "Left edge of the mask, valid minimun wavelength = "+np.str(np.round(w[first_good_pixel],2))+" , that is  w [ "+np.str(first_good_pixel)+" ]"
            plot_plot(w,np.nansum(self.mask, axis=0),ymax=1000,ymin=suma_good_pixels[0]-10,
                  xmax=w[first_good_pixel*3],vlines=[w[first_good_pixel]],
                  hlines=[nspec], ptitle=ptitle, ylabel="Sum of good fibres")      
    
        mask_first_good_value_per_fibre=[]
        for fibre in range(len(self.mask)):
            found=0
            j=0
            while found < 1:
                if no_nans:
                    if self.mask[fibre][j] == 0:
                        j=j+1
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
        if plot: plot_plot(np.arange(nspec),mask_first_good_value_per_fibre, ymax=mask_max+1,
                  hlines=[mask_max], xlabel="Fibre", ylabel="First good pixel in mask",
                  ptitle="Left edge of the mask")
          
        # Right edge, important for RED 
        if verbose : print("\n- Checking the right edge of the ccd...")
        mask_last_good_value_per_fibre=[]
        mask_list_fibres_all_good_values=[]
        
        for fibre in range(len(self.mask)):
            found=0
            j=len(self.mask[0])-1
            while found < 1:
                if no_nans:
                    if self.mask[fibre][j] == 0:
                        j = j - 1
                    else:
                        mask_last_good_value_per_fibre.append(j)
                        if j == len(self.mask[0])-1:
                            mask_list_fibres_all_good_values.append(fibre)
                        found = 2                    
                else:
                    if np.isnan(self.mask[fibre][j]):
                        j = j - 1
                    else:
                        mask_last_good_value_per_fibre.append(j)
                        if j == len(self.mask[0])-1:
                            mask_list_fibres_all_good_values.append(fibre)
                        found = 2
    
        mask_min = np.nanmin(mask_last_good_value_per_fibre)
        if plot:
            ptitle="Fibres with all good values in the right edge of the mask : "+np.str(len(mask_list_fibres_all_good_values))
            plot_plot(np.arange(nspec),mask_last_good_value_per_fibre, ymin=np.nanmin(mask_min),
                  ymax=2050,hlines=[2047], xlabel="Fibre", ylabel="Last good pixel in mask", ptitle=ptitle)
        if verbose: print("  Minimun value of good pixel =",mask_min," that corresponds to ",w[mask_min])
        if verbose: print("\n  --> The valid range for these data is",np.round(w[mask_max],2)," to ",np.round(w[mask_min],2), ",  in pixels = [",mask_max," , ",mask_min,"]")
            
        self.mask_good_index_range=[mask_max, mask_min]
        self.mask_good_wavelength_range=[w[mask_max], w[mask_min]]
        self.mask_list_fibres_all_good_values = mask_list_fibres_all_good_values
        
        if verbose: 
            print("\n> Mask stored in self.mask !")
            print("  Valid range of the data stored in self.mask_good_index_range (index)")
            print("                             and in self.mask_good_wavelength  (wavelenghts)")
            print("  List of fibres with all good values in self.mask_list_fibres_all_good_values")
        
        if include_history:
            self.history.append("  Valid range of data using the mask:")     
            self.history.append("  "+np.str(w[mask_max])+" to "+np.str(w[mask_min])+",  in pixels = [ "+np.str(mask_max)+" , "+np.str(mask_min)+" ]")                
# -----------------------------------------------------------------------------           
# -----------------------------------------------------------------------------
    def get_mask(self, verbose= True, plot = True, include_history=False):
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
                print("  Careful!!! pixel [0][-1], fibre = 0, wave = -1, that should be in the mask has a value that is not nan or 0 !!!!!")
 
        w = self.wavelength
        x = list(range(self.n_spectra))
        
        if verbose and plot : print("\n- Checking the left edge of the ccd...")
        mask_first_good_value_per_fibre =[]
        for fibre in range(self.n_spectra):
            found=0
            j=0
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
                    print(" No nan or 0 found in the fist 100 pixels, ", w[j]," for fibre", fibre)
                    mask_first_good_value_per_fibre.append(j)
                    found = 2    
        
        mask_max = np.nanmax(mask_first_good_value_per_fibre)
        if plot: 
            plot_plot(x,mask_first_good_value_per_fibre, ymax=mask_max+1,  xlabel="Fibre",
                      ptitle="Left edge of the RSS",hlines=[mask_max], ylabel="First good pixel in RSS")
                               
        # Right edge, important for RED 
        if verbose and plot : print("\n- Checking the right edge of the ccd...")
        mask_last_good_value_per_fibre=[]   
        mask_list_fibres_all_good_values=[]
        
        for fibre in range(self.n_spectra):
            found=0
            j=self.n_wave-1
            while found < 1:
                if no_nans:
                    if self.intensity[fibre][j] == 0:
                        j = j - 1
                    else:
                        mask_last_good_value_per_fibre.append(j)
                        if j == self.n_wave-1:
                            mask_list_fibres_all_good_values.append(fibre)
                        found = 2                    
                else:
                    if np.isnan(self.intensity[fibre][j]):
                        j = j - 1
                    else:
                        mask_last_good_value_per_fibre.append(j)
                        if j == len(self.intensity[0])-1:
                            mask_list_fibres_all_good_values.append(fibre)
                        found = 2
      
                if j <    self.n_wave-1 - 300:
                    print(" No nan or 0 found in the last 300 pixels, ", w[j]," for fibre", fibre)
                    mask_last_good_value_per_fibre.append(j)
                    found = 2 
                           
        mask_min = np.nanmin(mask_last_good_value_per_fibre)
        if plot:
            ptitle="Fibres with all good values in the right edge of the RSS file : "+np.str(len(mask_list_fibres_all_good_values))
            plot_plot(x,mask_last_good_value_per_fibre, ymin=np.nanmin(mask_min),
                  ymax=2050,hlines=[2047], xlabel="Fibre", ylabel="Last good pixel in RSS", ptitle=ptitle)
    
        if verbose: print("\n  --> The valid range for this RSS is {:.2f} to {:.2f} ,  in pixels = [ {} ,{} ]".format(w[mask_max],w[mask_min],mask_max,mask_min))
        
        self.mask = [mask_first_good_value_per_fibre,mask_last_good_value_per_fibre]
        self.mask_good_index_range=[mask_max, mask_min]
        self.mask_good_wavelength_range=[w[mask_max], w[mask_min]]
        self.mask_list_fibres_all_good_values = mask_list_fibres_all_good_values
        
        if verbose: 
            print("\n> Mask stored in self.mask !")
            print("  self.mask[0] contains the left edge, self.mask[1] the right edge")
            print("  Valid range of the data stored in self.mask_good_index_range (index)")
            print("                             and in self.mask_good_wavelength  (wavelenghts)")
            print("  Fibres with all good values (in right edge) in self.mask_list_fibres_all_good_values")
        
        if include_history:
            self.history.append("- Mask obtainted using the RSS file, valid range of data:")                            
            self.history.append("  "+np.str(w[mask_max])+" to "+np.str(w[mask_min])+",  in pixels = [ "+np.str(mask_max)+" , "+np.str(mask_min)+" ]")                
# -----------------------------------------------------------------------------           
# -----------------------------------------------------------------------------
    def apply_mask(self, mask_from_file=False, make_nans = False,
                   replace_nans=False, verbose = True ):
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
        if mask_from_file:           
            self.intensity_corrected = self.intensity_corrected * self.mask
        else:           
            for fibre in range(self.n_spectra):
                # Apply left part
                for i in range(self.mask[0][fibre]):
                    if make_nans :
                        self.intensity_corrected[fibre][i] = np.nan
                    else:
                        self.intensity_corrected[fibre][i] = 0
                # now right part
                for i in range(self.mask[1][fibre]+1,self.n_wave):
                    if make_nans :
                        self.intensity_corrected[fibre][i] = np.nan
                    else:
                        self.intensity_corrected[fibre][i] = 0   
        if replace_nans:
            # Change nans to 0:
            for i in range(self.n_spectra):
                self.intensity_corrected[i] = [0 if np.isnan(x) else x for x in self.intensity_corrected[i]]
            if verbose: print("\n> Mask applied to eliminate nans and make 0 all bad pixels")        
        else:
            if verbose: 
                if make_nans:
                    print("\n> Mask applied to make nan all bad pixels")  
                else:
                    print("\n> Mask applied to make 0 all bad pixels")
# -----------------------------------------------------------------------------           
# -----------------------------------------------------------------------------
    def compute_integrated_fibre(self, list_spectra = "all", valid_wave_min=0, valid_wave_max=0, 
                                 min_value = 0.01, plot=False, title =" - Integrated values", text = "...",
                                 norm=colors.PowerNorm(gamma=1./4.),
                                 correct_negative_sky = False, order_fit_negative_sky =3, kernel_negative_sky = 51, low_fibres=10,
                                 individual_check = True, use_fit_for_negative_sky = False,                                  
                                 last_check = False, verbose=True, warnings=True): 
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
        if valid_wave_min == 0 : valid_wave_min = self.valid_wave_min
        if valid_wave_max == 0 : valid_wave_max = self.valid_wave_max
        
        if verbose: print("\n> Computing integrated fibre values in range [ {:.2f} , {:.2f} ] {}".format(valid_wave_min,valid_wave_max,text))
               
        v = np.abs(self.wavelength-valid_wave_min)
        self.valid_wave_min_index = v.tolist().index(np.nanmin(v))
        v = np.abs(self.wavelength-valid_wave_max)
        self.valid_wave_max_index = v.tolist().index(np.nanmin(v))
                
        self.integrated_fibre = np.zeros(self.n_spectra) 
        region = np.where((self.wavelength > valid_wave_min) & (self.wavelength < valid_wave_max))
        waves_in_region=len(region[0])
        n_negative_fibres=0
        negative_fibres=[]
        for i in range(self.n_spectra):
            self.integrated_fibre[i] = np.nansum(self.intensity_corrected[i,region]) 
            if self.integrated_fibre[i] < 0: 
                if warnings == True and last_check == False: print("  WARNING: The integrated flux in fibre {:4} is negative, flux/wave = {:10.2f}, (probably sky), CHECK !".format(i,self.integrated_fibre[i]/waves_in_region))
                n_negative_fibres = n_negative_fibres +1              
                negative_fibres.append(i)

        if verbose:
            print("  - Median value of the integrated flux =",np.round(np.nanmedian(self.integrated_fibre),2))
            print("                                    min =",np.round(np.nanmin(self.integrated_fibre),2),", max =",np.round(np.nanmax(self.integrated_fibre),2))
            print("  - Median value per wavelength         =",np.round(np.nanmedian(self.integrated_fibre)/waves_in_region,2))
            print("                                    min = {:9.3f} , max = {:9.3f}".format(np.nanmin(self.integrated_fibre)/waves_in_region, np.nanmax(self.integrated_fibre)/waves_in_region))
            
        if len(negative_fibres) != 0:
            if warnings or verbose: print("\n> WARNING! : Number of fibres with integrated flux < 0 : {}, that is the {:5.2f} % of the total !".format(n_negative_fibres, n_negative_fibres*100./self.n_spectra))           
            if correct_negative_sky: # and len(negative_fibres) > 9:        
                self.correcting_negative_sky(plot=plot, order_fit_negative_sky=order_fit_negative_sky, individual_check=individual_check,
                                             kernel_negative_sky=kernel_negative_sky, use_fit_for_negative_sky = use_fit_for_negative_sky, low_fibres=low_fibres)                                                                    
            else:
                if plot:
                    if verbose:
                        print("\n> Adopting integrated flux = {:5.2f} for all fibres with negative integrated flux (for presentation purposes)".format(min_value))
                        print("  This value is {:5.2f} % of the median integrated flux per wavelength".format(min_value*100./np.nanmedian(self.integrated_fibre)*waves_in_region))
                negative_fibres_sorted=[]
                integrated_intensity_sorted=np.argsort(self.integrated_fibre/waves_in_region)
                for fibre_ in range(n_negative_fibres):
                    negative_fibres_sorted.append(integrated_intensity_sorted[fibre_])
                for i in negative_fibres_sorted:
                    self.integrated_fibre[i] = min_value
        else:
            if last_check: 
                if warnings or verbose: print("\n> There is no fibres with integrated flux < 0 !")
        self.integrated_fibre_sorted=np.argsort(self.integrated_fibre)     
        if plot: self.RSS_map(self.integrated_fibre, norm=norm, title=title) 
# -----------------------------------------------------------------------------           
# -----------------------------------------------------------------------------    
    def correcting_negative_sky(self, low_fibres=10, kernel_negative_sky = 51, order_fit_negative_sky = 3, edgelow=0, edgehigh=0,       #step=11, weight_fit_median = 1, scale = 1.0,
                                use_fit_for_negative_sky=False, individual_check = True, force_sky_fibres_to_zero = True,
                                exclude_wlm=[[0,0]], 
                                show_fibres=[0,450,985],fig_size = 12, plot=True, verbose=True):
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
        
        # CHECK fit_smooth_spectrum and compare with signal.medfilt
        w=self.wavelength
        # Set limits
        if edgelow == 0 : edgelow=self.valid_wave_min_index
        if edgehigh == 0 : edgehigh=np.int((self.n_wave-self.valid_wave_max_index)/2)
        
        plot_this=False
        if len(show_fibres) > 0 : 
            show_fibres.append(self.integrated_fibre_sorted[-1]) # Adding the brightest fibre 
            show_fibres.append(self.integrated_fibre_sorted[0]) # Adding the faintest fibre 
            
        if individual_check:
            if verbose: print("\n> Individual correction of fibres with negative sky ... ") 
            if force_sky_fibres_to_zero and verbose: print("  Also forcing integrated spectrum of sky_fibres = 0 ... ")
            corrected_not_sky_fibres = 0 
            total_corrected = 0
            sky_fibres_to_zero = 0
            for fibre in range(self.n_spectra):
                corregir = False
                if fibre in show_fibres:
                    print("\n - Checking fibre", fibre,"...")
                    plot_this=True
                else:
                    plot_this = False
                smooth,fit=fit_smooth_spectrum(w,self.intensity_corrected[fibre], kernel=kernel_negative_sky, 
                                                   edgelow=edgelow, edgehigh=edgehigh, verbose=False,
                                                   order=order_fit_negative_sky, plot=plot_this, hlines=[0.], ptitle= "", fcal = False)
                if np.nanpercentile(fit,5) < 0:
                    if fibre not in self.sky_fibres : corrected_not_sky_fibres = corrected_not_sky_fibres+1
                    corregir = True    
                else:
                    if fibre in self.sky_fibres and force_sky_fibres_to_zero : 
                        corregir == True 
                        sky_fibres_to_zero = sky_fibres_to_zero +1
                                          
              
                if corregir == True :  
                    total_corrected=total_corrected+1
                    if use_fit_for_negative_sky:
                        if fibre in show_fibres and verbose: print("   Using fit to smooth spectrum for correcting the negative sky in fibre",fibre," ...")
                        self.intensity_corrected[fibre] -= fit
                        #self.variance_corrected[fibre] -= fit
                    else:
                        if fibre in show_fibres and verbose: print("   Using smooth spectrum for correcting the negative sky in fibre",fibre," ...")
                        self.intensity_corrected[fibre] -= smooth
                        #self.variance_corrected[fibre] -= smooth
                else:
                    if fibre in show_fibres and verbose: print("   Fibre",fibre,"does not need to be corrected for negative sky ...")
                    
  
            corrected_sky_fibres = total_corrected-corrected_not_sky_fibres           
            if verbose:
                print("\n> Corrected {} fibres (not defined as sky) and {} out of {} sky fibres !".format(corrected_not_sky_fibres, corrected_sky_fibres, len(self.sky_fibres)))
                if force_sky_fibres_to_zero :  
                    print("  The integrated spectrum of",sky_fibres_to_zero,"sky fibres have been forced to 0.")
                    print("  The integrated spectrum of all sky_fibres have been set to 0.")
            self.history.append("- Individual correction of negative sky applied")
            self.history.append("  Corrected "+np.str(corrected_not_sky_fibres)+" not-sky fibres")
            if force_sky_fibres_to_zero:
                self.history.append("  All the "+np.str(len(self.sky_fibres))+" sky fibres have been set to 0")
            else:    
                self.history.append("  Corrected "+np.str(corrected_sky_fibres)+" out of "+np.str(len(self.sky_fibres))+" sky fibres")
        
        else:
            # Get integrated spectrum of n_low lowest fibres and use this for ALL FIBRES
            integrated_intensity_sorted=np.argsort(self.integrated_fibre)
            region=integrated_intensity_sorted[0:low_fibres]   
            Ic=np.nanmedian(self.intensity_corrected[region], axis=0)

            if verbose:
                print("\n> Correcting negative sky using median spectrum combining the",low_fibres,"fibres with the lowest integrated intensity")
                print("  which are :",region)      
                print("  Obtaining smoothed spectrum using a {} kernel and fitting a {} order polynomium...".format(kernel_negative_sky,order_fit_negative_sky))
            ptitle = self.object+" - "+str(low_fibres)+" fibres with lowest intensity - Fitting an order "+str(order_fit_negative_sky)+" polynomium to spectrum smoothed with a "+str(kernel_negative_sky)+" kernel window"
            smooth,fit=fit_smooth_spectrum(self.wavelength,Ic, kernel=kernel_negative_sky, edgelow=edgelow, edgehigh=edgehigh, verbose=False,
                                                 order=order_fit_negative_sky, plot=plot, hlines=[0.], ptitle= ptitle, fcal = False)            
            if use_fit_for_negative_sky:
                self.smooth_negative_sky = fit         
                if verbose: print("  Sustracting fit to smoothed spectrum of {} low intensity fibres to all fibres ...".format(low_fibres))
            else:
                self.smooth_negative_sky = smooth         
                if verbose: print("  Sustracting smoothed spectrum of {} low intensity fibres to all fibres ...".format(low_fibres))    
                
            for i in range(self.n_spectra):
                self.intensity_corrected[i,:]=self.intensity_corrected[i,:] - self.smooth_negative_sky
                #self.sky_emission = self.sky_emission - self.smooth_negative_sky
    
            if verbose: print("  This smoothed spectrum is stored in self.smooth_negative_sky")
            self.history.append("- Correcting negative sky using smoothed spectrum of the")
            self.history.append("  "+np.str(low_fibres)+" fibres with the lowest integrated value")           
# -----------------------------------------------------------------------------           
# -----------------------------------------------------------------------------    
    def identify_el(self, high_fibres=10, brightest_line = "Ha", cut=1.5, 
                    fibre=0, broad = 1.0, verbose=True, plot=True):
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
            integrated_intensity_sorted=np.argsort(self.integrated_fibre)
            region=[]
            for fibre in range(high_fibres):
                region.append(integrated_intensity_sorted[-1-fibre])
            if verbose:
                print("\n> Identifying emission lines using the",high_fibres,"fibres with the highest integrated intensity")
                print("  which are :",region)
            combined_high_spectrum=np.nansum(self.intensity_corrected[region], axis=0)
        else:
            combined_high_spectrum=self.intensity_corrected[fibre]
            if verbose: print("\n> Identifying emission lines in fibre", fibre)
               
        # Search peaks
        peaks,peaks_name,peaks_rest,continuum_limits=search_peaks(self.wavelength, combined_high_spectrum, plot=plot, 
                                                                  cut=cut, brightest_line=brightest_line, verbose=False)                                                                                                                                 
        p_peaks_l=[]
        p_peaks_fwhm=[]
    
        # Do Gaussian fit and provide center & FWHM (flux could be also included, not at the moment as not abs. flux-cal done)
        if verbose: print("\n  Emission lines identified:")
        for eline in range(len(peaks)):
            lowlow=continuum_limits[0][eline]
            lowhigh=continuum_limits[1][eline]
            highlow=continuum_limits[2][eline]
            highhigh=continuum_limits[3][eline]       
            resultado = fluxes(self.wavelength,combined_high_spectrum, peaks[eline], verbose=False, broad=broad, 
                               lowlow=lowlow, lowhigh=lowhigh, highlow=highlow, highhigh=highhigh, plot=plot, fcal=False)
            p_peaks_l.append(resultado[1])
            p_peaks_fwhm.append(resultado[5])                   
            if verbose:  print("  {:3}. {:7s} {:8.2f} centered at {:8.2f} and FWHM = {:6.2f}".format(eline+1,peaks_name[eline],peaks_rest[eline],p_peaks_l[eline],p_peaks_fwhm[eline]))
        
        return [peaks_name,peaks_rest, p_peaks_l, p_peaks_fwhm]         
# -----------------------------------------------------------------------------           
# -----------------------------------------------------------------------------
    def correct_ccd_defects(self, kernel_correct_ccd_defects = 51, fibre_p=-1, remove_5577=False, # if fibre_p=fibre plots the corrections in that fibre 
                            only_nans=False,
                            plot=True,  verbose=False, warnings=True, fig_size=12, apply_throughput=False):
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
        if verbose: 
            if only_nans :
                print("\n> Correcting CCD defects (nan and inf values) using medfilt with kernel",kernel_correct_ccd_defects," ...")
            else:
                print("\n> Correcting CCD defects (nan, inf, and negative values) using medfilt with kernel",kernel_correct_ccd_defects," ...")

        wave_min=self.valid_wave_min
        wave_max=self.valid_wave_max 
        wlm=self.wavelength
        if wave_min < 5577 and remove_5577 : 
            flux_5577=[] # For correcting sky line 5577 if requested
            offset_5577=[] 
            if verbose: print("  Sky line 5577.34 will be removed using a Gaussian fit...")
                
        print(" ") 
        output_every_few = np.sqrt(self.n_spectra)+1
        next_output = -1
        if fibre_p < 0 : fibre_p = ""
        if fibre_p == "" :
            ri =  0
            rf = self.n_spectra
        else:
            if verbose: print("  Only fibre {} is corrected ...".format(fibre_p))
            ri = fibre_p
            rf = fibre_p +1
        
        for fibre in range(ri,rf):
            if fibre > next_output and fibre_p == "":
                sys.stdout.write("\b"*30)
                sys.stdout.write("  Cleaning... {:5.2f}% completed".format(fibre*100./self.n_spectra))
                sys.stdout.flush()
                next_output = fibre + output_every_few    

            s = self.intensity_corrected[fibre]
            if only_nans:  
                s = [0 if np.isnan(x) or np.isinf(x) else x for x in s]   # Fix nans & inf    
            else:
                s = [0 if np.isnan(x) or x < 0. or np.isinf(x) else x for x in s]   # Fix nans, inf & negative values = 0                    
            s_m = signal.medfilt(s,kernel_correct_ccd_defects)

            fit_median = signal.medfilt(s, kernel_correct_ccd_defects)        
            bad_indices = [i for i, x in enumerate(s) if x == 0]
            for index in bad_indices:
                s[index] = s_m[index]                                   # Replace 0s for median value
    
            if fibre == fibre_p :
                espectro_old = copy.copy(self.intensity_corrected[fibre,:])
                espectro_fit_median=fit_median
                espectro_new = copy.copy(s)
                
            self.intensity_corrected[fibre,:] = s
                                                 
            # Removing Skyline 5577 using Gaussian fit if requested
            if wave_min < 5577 and remove_5577 :            
                resultado = fluxes(wlm, s, 5577.34, lowlow=30, lowhigh=10, highlow=10, highhigh=30,
                                   plot=False, verbose=False, fcal=False, plot_sus=False)  #fmin=-5.0E-17, fmax=2.0E-16, 
                #resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                self.intensity_corrected[fibre] = resultado[11]
                flux_5577.append(resultado[3])
                offset_5577.append(resultado[1]-5577.34)
 
        if fibre_p == "" :
            sys.stdout.write("\b"*30)
            sys.stdout.write("  Cleaning... 100.00 completed!")
            sys.stdout.flush()
        if verbose: print(" ")
                
        if wave_min < 5577 and remove_5577 and fibre_p == "":
            if verbose: print("\n> Checking centroid of skyline 5577.34 obtained during removing sky...") 
            fibre_vector = np.array(list(range(len(offset_5577))))
            offset_5577_m = signal.medfilt(offset_5577, 101)           
            a1x,a0x = np.polyfit(fibre_vector, offset_5577, 1)
            fx = a0x + a1x*fibre_vector 
            self.wavelength_offset_per_fibre = offset_5577

            if plot: plot_plot (fibre_vector, [offset_5577, fx, offset_5577_m], psym=["+","-", "-"], 
                                color=["r","b", "g"], alpha=[1,0.7, 0.8],
                                xmin=-20,xmax=fibre_vector[-1]+20,
                                percentile_min=0.5, percentile_max= 99.5, hlines=[-0.5,-0.25,0,0.25,0.5],
                                ylabel= "fit(5577.34) - 5577.34", xlabel="Fibre",
                                ptitle ="Checking wavelength centroid of fitted skyline 5577.34",
                                label=["data","Fit", "median k=101"],
                                fig_size=fig_size)
            if verbose:
                print("  The median value of the fit(5577.34) - 5577.34 is ",np.nanmedian(offset_5577))
                print("  A linear fit y = a + b * fibre provides a =",a0x," and b =",a1x)
            if np.abs(a1x) * self.n_spectra < 0.01 :
                if verbose:
                    print("  Wavelengths variations are smaller than 0.01 A in all rss file (largest =",np.abs(a1x) * self.n_spectra,"A).")
                    print("  No need of correcting for small wavelengths shifts!")
            else:
                if verbose:
                    print("  Wavelengths variations are larger than 0.01 A in all rss file (largest =",np.abs(a1x) * self.n_spectra,"A).")
                    print("  Perhaps correcting for small wavelengths shifts is needed, use:")
                    print("  sol = [ {} , {},  0 ]".format(a0x,a1x))
                if self.sol[0] != 0:
                    if verbose: print("\n  But sol already provided as an input, sol = [ {} , {},  {} ]".format(self.sol[0],self.sol[1],self.sol[2]))
                    fx_given = self.sol[0] + self.sol[1]* fibre_vector
                    rms = fx-fx_given
                    if verbose: print("  The median diference of the two solutions is {:.4}".format(np.nanmedian(rms)))
                    if plot: plot_plot (fibre_vector, [fx, fx_given, rms], psym=["-","-", "--"], 
                                        color=["b", "g", "k"], alpha=[0.7, 0.7, 1],
                                        xmin=-20,xmax=fibre_vector[-1]+20,
                                        percentile_min=0.5, percentile_max= 99.5, hlines=[-0.25,-0.125,0,0.125,0.5],
                                        ylabel= "fit(5577.34) - 5577.34", xlabel="Fibre",
                                        ptitle ="Small wavelength variations",
                                        label=["5577 fit","Provided", "Difference"],
                                        fig_size=fig_size)
                            
                    if verbose: print("  Updating this solution with the NEW sol values...")
                self.sol=[a0x, a1x, 0.]
             
        # Plot correction in fibre p_fibre
        if fibre_p != "" :
            const=(np.nanmax(espectro_new)-np.nanmin(espectro_new))/2
            yy=[espectro_old/espectro_fit_median,espectro_new/espectro_fit_median,(const+espectro_new-espectro_old)/espectro_fit_median]
            ptitle ="Checking correction in fibre "+str(fibre_p)
            plot_plot (wlm, yy,  
                       color=["r","b","k"], alpha=[0.5,0.5,0.5],
                       percentile_min=0.5, percentile_max= 98,
                       ylabel= "Flux / Continuum",
                       ptitle =ptitle,loc=1,ncol=4,
                       label=["Uncorrected","Corrected","Dif + const"],
                       fig_size=fig_size)
        else:        
            # Recompute the integrated fibre
            text= "for spectra corrected for CCD defects..."    
            if apply_throughput:
                title =" - CCD defects corrected"    
            else:
                title =" - Throughput + CCD defects corrected"   
            self.compute_integrated_fibre(valid_wave_min=wave_min, valid_wave_max=wave_max, text = text, plot=plot, title=title, verbose=verbose, warnings=warnings)
        
            if remove_5577 and wave_min < 5577 :
                if verbose: print("  Skyline 5577.34 has been removed. Checking throughput correction...")        
                extra_throughput_correction = flux_5577 / np.nanmedian(flux_5577)
                extra_throughput_correction_median=np.round(np.nanmedian(extra_throughput_correction),3)
                if plot: 
                    ptitle="Checking throughput correction using skyline 5577 $\mathrm{\AA}$"
                    plot_plot (fibre_vector, extra_throughput_correction,  color="#1f77b4",
                               percentile_min=1, percentile_max= 99, hlines =[extra_throughput_correction_median],
                               ylabel= "Integrated flux per fibre / median value", xlabel="Fibre",
                               ptitle =ptitle,fig_size=fig_size)               
                if verbose: print("  Variations in throughput between",np.nanmin(extra_throughput_correction),"and",np.nanmax(extra_throughput_correction),", median = ",extra_throughput_correction_median)
    
            # Apply mask
            self.apply_mask(verbose=verbose)
# -----------------------------------------------------------------------------        
# -----------------------------------------------------------------------------
    def fit_and_substract_sky_spectrum(self, sky, w=1000, spectra=1000, rebin = False,   # If rebin == True, it fits all wavelengths to be at the same wavelengths that SKY spectrum...
                                       brightest_line="Ha", brightest_line_wavelength = 0,
                                       maxima_sigma=3.0, ymin =-50, ymax=600, wmin = 0, wmax =0, 
                                       auto_scale_sky = False,
                                       sky_lines_file="",
                                       warnings = False, verbose=False, plot=False, plot_step_fibres=True, step = 100,
                                       fig_size=12, fibre = -1, max_flux_variation = 15. ,
                                       min_flux_ratio = -1, max_flux_ratio =-1 ):       
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
        if min_flux_ratio == -1: min_flux_ratio = 1.- max_flux_variation/100.
        if max_flux_ratio == -1: max_flux_ratio = 1.+ max_flux_variation/100.
                
        if sky_lines_file == "":
            #sky_lines_file="sky_lines_interval5.dat"   
            sky_lines_file="sky_lines_bright.dat"
            #sky_lines_file="sky_lines_test.txt" #  EZRA
            
        if sky_lines_file == "ALL": sky_lines_file="sky_lines.dat"
        if sky_lines_file == "BRIGHT": sky_lines_file="sky_lines_bright.dat"
        if sky_lines_file == "IR": sky_lines_file="sky_lines_IR.dat"
        if sky_lines_file == "BRIGHT+IR": sky_lines_file="sky_lines_bright+IR.dat"
        if sky_lines_file in ["IRshort", "IRs", "IR_short"]: sky_lines_file="sky_lines_IR_short.dat"
        
        self.history.append('  Skylines fitted following file:')
        self.history.append('  '+sky_lines_file)
        
        print("\n> Fitting selected sky lines to both sky spectrum and object spectra ...\n")
        
        brightest_line_wavelength_rest = 6562.82
        if brightest_line == "O3" or brightest_line == "O3b" : brightest_line_wavelength_rest = 5006.84
        if brightest_line == "Hb" or brightest_line == "hb" : brightest_line_wavelength_rest = 4861.33
        
        if brightest_line_wavelength != 0:
            print("  - Using {} at rest wavelength {:6.2f} identified by the user at {:6.2f} to avoid fitting emission lines...".format(brightest_line,brightest_line_wavelength_rest,brightest_line_wavelength))
        else:
            print("  - No wavelength provided to 'brightest_line_wavelength', the object is NOT expected to have emission lines\n")
        
        redshift = brightest_line_wavelength/brightest_line_wavelength_rest - 1.
        
        if w == 1000: w = self.wavelength
        if spectra == 1000: spectra= copy.deepcopy(self.intensity_corrected)

        if wmin == 0 : wmin = w[0]
        if wmax == 0 : wmax = w[-1]
        
        print("  - Reading file with the list of sky lines to fit :")
        print("   ",sky_lines_file)
                   
        # Read file with sky emission lines
        sl_center_,sl_name_,sl_fnl_,sl_lowlow_,sl_lowhigh_,sl_highlow_,sl_highhigh_,sl_lmin_,sl_lmax_ = read_table(sky_lines_file, ["f", "s", "f", "f", "f", "f", "f", "f", "f"] )
        #number_sl = len(sl_center)
          
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
                
        el_list_no_z = [6300.3, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 7751.1, 9068.9] 
        el_list = (redshift +1) * np.array(el_list_no_z)
                          #  [OI]   [SIII]  [OI]   Ha+[NII]  HeI    [SII]     HeI   [ArIII]  [OII]  [ArIII]  [SIII]  
        el_low_list_no_z  =[6296.3, 6308.1, 6359.8, 6544.0, 6674.2, 6712.5, 7061.3, 7129., 7312., 7747.1, 9063.9]
        el_high_list_no_z =[6304.3, 6316.1, 6367.8, 6590.0, 6682.2, 6736.9, 7069.3, 7141., 7336., 7755.1, 9073.9]
        el_low_list= (redshift +1) * np.array(el_low_list_no_z)
        el_high_list= (redshift +1) *np.array(el_high_list_no_z)
        
        #Double Skylines
        dsky1_=[6257.82, 6465.34, 6828.22, 6969.70, 7239.41, 7295.81, 7711.50, 7750.56, 7853.391, 7913.57, 7773.00, 7870.05, 8280.94, 8344.613, 9152.2, 9092.7, 9216.5,  8827.112, 8761.2, 0] # 8760.6, 0]#
        dsky2_=[6265.50, 6470.91, 6832.70, 6978.45, 7244.43, 7303.92, 7715.50, 7759.89, 7860.662, 7921.02, 7780.43, 7879.96, 8288.34, 8352.78,  9160.9, 9102.8, 9224.8,  8836.27 , 8767.7, 0] # 8767.2, 0] #     
        
        
        # Be sure the lines we are using are in the requested wavelength range        
        #print "  Checking the values of skylines in the file", sky_lines_file
        #for i in range(len(sl_center_)):
        #    print sl_center_[i],sl_fnl_[i],sl_lowlow_[i],sl_lowhigh_[i],sl_highlow_[i],sl_highhigh_[i],sl_lmin_[i],sl_lmax_[i]          
        #print "  We only need skylines in the {} - {} range:".format(self.valid_wave_min, self.valid_wave_max)
        print("  - We only need sky lines in the {} - {} range ".format(np.round(self.wavelength[0],2), np.round(self.wavelength[-1],2)))

                   
        #valid_skylines = np.where((sl_center_ < self.valid_wave_max) & (sl_center_ > self.valid_wave_min))
        
        valid_skylines = np.where((sl_center_ < self.wavelength[-1]) & (sl_center_ > self.wavelength[0]))
        
        sl_center=sl_center_[valid_skylines]
        sl_fnl = sl_fnl_[valid_skylines]
        sl_lowlow = sl_lowlow_[valid_skylines]
        sl_lowhigh = sl_lowhigh_[valid_skylines]
        sl_highlow = sl_highlow_[valid_skylines]
        sl_highhigh = sl_highhigh_[valid_skylines]
        sl_lmin = sl_lmin_[valid_skylines]
        sl_lmax = sl_lmax_[valid_skylines]            
        number_sl = len(sl_center)
        
        dsky1=[]
        dsky2=[]
        for l in range(number_sl):
            if sl_center[l] in dsky1_:                
                dsky1.append(dsky1_[dsky1_.index(sl_center[l])])
                dsky2.append(dsky2_[dsky1_.index(sl_center[l])])                               
        
        print("  - All sky lines: ",sl_center)
        print("  - Double sky lines: ", dsky1)
        print("  - Total number of skylines to fit =",len(sl_center))
        print("  - Valid values for OBJ / SKY Gauss ratio  = ( ",min_flux_ratio,",",max_flux_ratio,")")
        print("  - Maxima sigma to consider a valid fit  = ",maxima_sigma," A\n")
   
        say_status = 0
        self.wavelength_offset_per_fibre=[]
        self.sky_auto_scale=[]
        f_new_ALL=[]
        sky_sl_gaussian_fitted_ALL =[]
        only_fibre = False
        if fibre != -1:
            f_i=fibre
            f_f=fibre+1
            print("\n ----> Checking fibre ", fibre," (only this fibre is corrected, use fibre = -1 for all)...") 
            plot=True
            verbose = True
            warnings = True
            only_fibre= True
            say_status = fibre
        else:
            f_i = 0 
            f_f = self.n_spectra
            
        # Check if skylines are located within the range of an emission line !  
        skip_sl_fit=[False] * number_sl
        if verbose or fibre == -1: print("  - Checking skylines within emission line ranges...")
        for i in range(number_sl):
            for j in range(len(el_low_list)):
                if el_low_list[j] < sl_center[i] < el_high_list[j]:
                    skip_sl_fit[i]=True
                    if verbose or fibre == -1: print('  ------> SKY line',sl_center[i],'in EMISSION LINE !  ',el_low_list[j], sl_center[i], el_high_list[j])                  

        # Gaussian fits to the sky spectrum
        sl_gaussian_flux=[]
        sl_gaussian_sigma=[]    
        sl_gauss_center=[]
        sky_sl_gaussian_fitted = copy.deepcopy(sky)
        if verbose or fibre == -1: print("  - Performing Gaussian fitting to sky lines in sky spectrum...")
        for i in range(number_sl):                                               
            if sl_fnl[i] == 0 :
                plot_fit = False
            else: plot_fit = True                        
            if sl_center[i] in dsky1 : #  == dsky1[di] :
                    if fibre == -1: print("    DOUBLE IN SKY: ",sl_center[i],dsky2[dsky1.index(sl_center[i])])
                    warnings_ = False
                    if sl_fnl[i] == 1 : 
                        warnings_=True
                        if verbose: print("    Line ", sl_center[i], " blended with ",dsky2[dsky1.index(sl_center[i])])
                    resultado=dfluxes(w, sky_sl_gaussian_fitted, sl_center[i],dsky2[dsky1.index(sl_center[i])], lowlow= sl_lowlow[i], lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh = sl_highhigh[i], lmin=sl_lmin[i], lmax=sl_lmax[i], fmin=-20, fmax=0, 
                         broad1=2.1*2.355, broad2=2.1*2.355, plot=plot_fit, verbose=False, plot_sus = False, fcal = False, warnings = warnings_ )   # Broad is FWHM for Gaussian sigm a= 1, 

                    sl_gaussian_flux.append(resultado[3])    # 15 is Gauss 1, 16 is Gauss 2, 3 is Total Gauss
                    sl_gauss_center.append(resultado[1])
                    sl_gaussian_sigma.append(resultado[5]/2.355)
                    #sl_gaussian_flux.append(resultado[16])
                    #sl_gauss_center.append(resultado[12])
                    #sl_gaussian_sigma.append(resultado[14]/2.355)                    
                    # 12     13      14        15              16
                    #fit[3], fit[4],fit[5], gaussian_flux_1, gaussian_flux_2 # KANAN
                    
            else:                           
                resultado=fluxes(w, sky_sl_gaussian_fitted, sl_center[i], lowlow= sl_lowlow[i], lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh = sl_highhigh[i], lmin=sl_lmin[i], lmax=sl_lmax[i], fmin=-20, fmax=0, 
                         broad=2.1*2.355, plot=plot_fit, verbose=False, plot_sus = False, fcal = False, warnings = warnings )   # Broad is FWHM for Gaussian sigm a= 1, 

                sl_gaussian_flux.append(resultado[3])
                sl_gauss_center.append(resultado[1])
                sl_gaussian_sigma.append(resultado[5]/2.355)

            if plot_fit:
                if verbose:  print("    Fitted wavelength for sky line ",sl_center[i]," : ",sl_gauss_center[-1],"  sigma = ",sl_gaussian_sigma[-1])
                wmin=sl_lmin[i]
                wmax=sl_lmax[i]
                
            if skip_sl_fit[i] == False:
                sky_sl_gaussian_fitted=resultado[11]
            else:
                if verbose: print('  ------> SKY line',sl_center[i],'in EMISSION LINE !')
        
        # Now Gaussian fits to fibres        
        for fibre in range(f_i,f_f): #    (self.n_spectra): 
            if fibre == say_status : 
                if fibre == 0 : print(" ")
                print("  - Checking fibre {:4} ...  ({:6.2f} % completed) ...".format(fibre, fibre*100./self.n_spectra))
                say_status=say_status+step
                if plot_step_fibres: plot=True
            else: plot = False    
        
            skip_el_fit = copy.deepcopy(skip_sl_fit)
            
            # Gaussian fit to object spectrum
            object_sl_gaussian_flux=[]
            object_sl_gaussian_sigma=[]
            ratio_object_sky_sl_gaussian = []
            dif_center_obj_sky = []
            spec=spectra[fibre]
            object_sl_gaussian_fitted = copy.deepcopy(spec) 
            object_sl_gaussian_center = []
            if verbose: print("\n  - Performing Gaussian fitting to sky lines in fibre",fibre,"of object data ...")
        
            for i in range(number_sl):
                if sl_fnl[i] == 0 :
                    plot_fit = False
                else: 
                    plot_fit = True
                if skip_el_fit[i]:
                    if verbose: print("    SKIPPING SKY LINE",sl_center[i],"as located within the range of an emission line!")
                    object_sl_gaussian_flux.append(float('nan'))   # The value of the SKY SPECTRUM
                    object_sl_gaussian_center.append(float('nan'))
                    object_sl_gaussian_sigma.append(float('nan'))
                    dif_center_obj_sky.append(float('nan'))
                else:
                    if sl_center[i] in dsky1: #== dsky1[di] :
                        if fibre == -1: print("    DOUBLE IN SKY: ",sl_center[i],dsky2[dsky1.index(sl_center[i])])
                        warnings_ = False                     
                        if sl_fnl[i] == 1 : 
                            if fibre == -1: 
                                warnings_=True
                            if verbose: print("    Line ", sl_center[i], " blended with ",dsky2[dsky1.index(sl_center[i])])
                        resultado=dfluxes(w, object_sl_gaussian_fitted, sl_center[i],dsky2[dsky1.index(sl_center[i])], lowlow= sl_lowlow[i], lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh = sl_highhigh[i], lmin=sl_lmin[i], lmax=sl_lmax[i], fmin=-20, fmax=0, 
                                 broad1=sl_gaussian_sigma[i]*2.355,broad2=sl_gaussian_sigma[i]*2.355, plot=plot_fit, verbose=False, plot_sus = False, fcal = False, warnings = warnings_ )
                        if verbose: 
                            print("    line = {:.3f} : center = {:.3f}, gauss = {:.2f},  sigma = {:.2f}, flux = {:.2f}".format(sl_center[i],resultado[1],sl_gaussian_sigma[i], resultado[5]/2.355, resultado[15]))
                            print("    line = {:.3f} : center = {:.3f}, gauss = {:.2f},  sigma = {:.2f}, flux = {:.2f}".format(dsky2[dsky1.index(sl_center[i])],resultado[12],sl_gaussian_sigma[i], resultado[14]/2.355, resultado[16]))
                            print("    For skylines ",sl_center[i],"+",dsky2[dsky1.index(sl_center[i])]," the total flux is ",np.round(sl_gaussian_flux[i],3), ",                     OBJ/SKY = ",np.round(resultado[3]/sl_gaussian_flux[i],3))

                        if resultado[3] > 0 and resultado[5]/2.355 < maxima_sigma  and resultado[15] > 0 and resultado[14]/2.355 < maxima_sigma  and resultado[3]/sl_gaussian_flux[i] > min_flux_ratio and resultado[3]/sl_gaussian_flux[i] < max_flux_ratio *1.25   : # and resultado[5] < maxima_sigma: # -100000.: #0:                            
                            object_sl_gaussian_fitted=resultado[11]

                            object_sl_gaussian_flux.append(resultado[15])            
                            object_sl_gaussian_center.append(resultado[1])
                            object_sl_gaussian_sigma.append(resultado[5]/2.355)                                                        
                            #object_sl_gaussian_flux.append(resultado[16])
                            #object_sl_gaussian_center.append(resultado[12])
                            #object_sl_gaussian_sigma.append(resultado[14]/2.355) 
                       
                            dif_center_obj_sky.append(object_sl_gaussian_center[i]-sl_gauss_center[i])
                        else:
                            if verbose: print("    Bad double fit for ",sl_center[i],"! trying single fit...")
                            average_wave= (sl_center[i]+dsky2[dsky1.index(sl_center[i])])/2
                            resultado=fluxes(w, object_sl_gaussian_fitted, average_wave, lowlow= sl_lowlow[i], lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh = sl_highhigh[i], lmin=average_wave-50, lmax=average_wave+50, fmin=-20, fmax=0, 
                                         broad=4.5, plot=plot_fit, verbose=False, plot_sus = False, fcal = False, warnings = warnings )   # Broad is FWHM for Gaussian sigma= 1, 
                            if verbose: print("    line = {:.3f} : center = {:.3f}, gauss = {:.2f},  sigma = {:.2f}, flux = {:.2f},   OBJ/SKY = {:.3f}".format(sl_center[i],resultado[1],sl_gaussian_sigma[i], resultado[5]/2.355, resultado[3], resultado[3]/sl_gaussian_flux[i]))
                            if resultado[3] > 0 and resultado[5]/2.355 < maxima_sigma*2.   and resultado[3]/sl_gaussian_flux[i] > min_flux_ratio and resultado[3]/sl_gaussian_flux[i] < max_flux_ratio  : # and resultado[5] < maxima_sigma: # -100000.: #0:
                                object_sl_gaussian_flux.append(resultado[3])            
                                object_sl_gaussian_fitted=resultado[11]
                                object_sl_gaussian_center.append(resultado[1])
                                object_sl_gaussian_sigma.append(resultado[5]/2.355)
                                dif_center_obj_sky.append(object_sl_gaussian_center[i]-sl_gauss_center[i])
                            else:
                                if verbose: print("    -> Bad fit for ",sl_center[i],"! ignoring it...")
                                object_sl_gaussian_flux.append(float('nan'))            
                                object_sl_gaussian_center.append(float('nan'))
                                object_sl_gaussian_sigma.append(float('nan'))
                                dif_center_obj_sky.append(float('nan'))
                                skip_el_fit[i] = True   # We don't substract this fit
                    else:
                        resultado=fluxes(w, object_sl_gaussian_fitted, sl_center[i], lowlow= sl_lowlow[i], lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh = sl_highhigh[i], lmin=sl_lmin[i], lmax=sl_lmax[i], fmin=0, fmax=0, 
                                         broad=sl_gaussian_sigma[i]*2.355, plot=plot_fit, verbose=False, plot_sus = False, fcal = False, warnings = warnings )   # Broad is FWHM for Gaussian sigma= 1, 
                        if verbose: print("    line = {:.3f} : center = {:.3f}, gauss = {:.2f},  sigma = {:.2f}, flux = {:.2f},   OBJ/SKY = {:.3f}".format(sl_center[i],resultado[1],sl_gaussian_sigma[i], resultado[5]/2.355, resultado[3], resultado[3]/sl_gaussian_flux[i]))
                        if resultado[3] > 0 and resultado[5]/2.355 < maxima_sigma   and resultado[3]/sl_gaussian_flux[i] > min_flux_ratio and resultado[3]/sl_gaussian_flux[i] < max_flux_ratio  : # and resultado[5] < maxima_sigma: # -100000.: #0:
                            object_sl_gaussian_flux.append(resultado[3])            
                            object_sl_gaussian_fitted=resultado[11]
                            object_sl_gaussian_center.append(resultado[1])
                            object_sl_gaussian_sigma.append(resultado[5]/2.355)
                            dif_center_obj_sky.append(object_sl_gaussian_center[i]-sl_gauss_center[i])
                        else:
                            if verbose: print("    -> Bad fit for ",sl_center[i],"! ignoring it...")
                            object_sl_gaussian_flux.append(float('nan'))            
                            object_sl_gaussian_center.append(float('nan'))
                            object_sl_gaussian_sigma.append(float('nan'))
                            dif_center_obj_sky.append(float('nan'))
                            skip_el_fit[i] = True   # We don't substract this fit
                
                try:
                    ratio_object_sky_sl_gaussian.append(object_sl_gaussian_flux[i]/sl_gaussian_flux[i]) 
                except Exception:
                    print("\n\n\n\n\n DIVISION FAILED in ",sl_center[i],"!!!!!   sl_gaussian_flux[i] = ", sl_gaussian_flux[i],"\n\n\n\n")
                    ratio_object_sky_sl_gaussian.append(1.)
        
            # Scale sky lines that are located in emission lines or provided negative values in fit
            #reference_sl = 1 # Position in the file! Position 1 is sky line 6363.4 
            #sl_ref_ratio = sl_gaussian_flux/sl_gaussian_flux[reference_sl]
            if verbose:
                print("  - Correcting skylines for which we couldn't get a Gaussian fit and are not in an emission line range...")
            for i in range(number_sl):
                if skip_el_fit[i] == True and skip_sl_fit[i] == False :  # Only those that are NOT in emission lines
                    # Use known center, sigma of the sky and peak 
                    gauss_fix = sl_gaussian_sigma[i]
                    small_center_correction = 0.
                    # Check if center of previous sky line has a small difference in wavelength
                    small_center_correction=np.nanmedian(dif_center_obj_sky[0:i])
                    if verbose: 
                        print("  - Small correction of center wavelength of sky line ",sl_center[i],"  :",small_center_correction)
                            
                    object_sl_gaussian_fitted=substract_given_gaussian(w, object_sl_gaussian_fitted, sl_center[i]+small_center_correction, peak=0, sigma=gauss_fix,  flux=0, search_peak=True,
                                     lowlow= sl_lowlow[i], lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh = sl_highhigh[i], lmin=sl_lmin[i], lmax=sl_lmax[i], plot=False, verbose=verbose)

                    # Substract second Gaussian if needed !!!!!  
                    for di in range(len(dsky1)):
                        if sl_center[i] == dsky1[di]:
                            if verbose: print("    This was a double sky line, also substracting ",dsky2[dsky1.index(sl_center[i])],"  at ",np.round(np.array(dsky2[dsky1.index(sl_center[i])])+small_center_correction,2))
                            object_sl_gaussian_fitted=substract_given_gaussian(w, object_sl_gaussian_fitted, np.array(dsky2[dsky1.index(sl_center[i])])+small_center_correction, peak=0, sigma=gauss_fix,  flux=0, search_peak=True,
                                     lowlow= sl_lowlow[i], lowhigh=sl_lowhigh[i], highlow=sl_highlow[i], highhigh = sl_highhigh[i], lmin=sl_lmin[i], lmax=sl_lmax[i], plot=False, verbose=verbose)
                else:
                    if skip_sl_fit[i] == True and skip_el_fit[i] == True :
                        if verbose: print("     - SKIPPING SKY LINE",sl_center[i]," as located within the range of an emission line!")
        
                     
            offset =  np.nanmedian(np.array(object_sl_gaussian_center)-np.array(sl_gauss_center))
            offset_std =np.nanstd(np.array(object_sl_gaussian_center)-np.array(sl_gauss_center))
            
            good_ratio_values = []
            for ratio in ratio_object_sky_sl_gaussian:
                if np.isnan(ratio) == False:
                    if ratio > min_flux_ratio and ratio < max_flux_ratio:
                        good_ratio_values.append(ratio)
                    
            valid_median_flux=np.nanmedian(good_ratio_values)
            
            if verbose:             
                print("  - Median center offset between OBJ and SKY :", np.round(offset,3), " A ,    std = ",np.round(offset_std,3))   
                print("    Median gauss for the OBJECT              :", np.round(np.nanmedian(object_sl_gaussian_sigma),3)," A ,    std = ", np.round(np.nanstd(object_sl_gaussian_sigma),3))
                print("    Median flux OBJECT / SKY                 :", np.round(np.nanmedian(ratio_object_sky_sl_gaussian),3), "   ,    std = ",np.round(np.nanstd(ratio_object_sky_sl_gaussian),3))            
                print("    Median flux OBJECT / SKY VALID VALUES    :", np.round(valid_median_flux,3), "   ,    std = ",np.round(np.nanstd(good_ratio_values),3))            
                print("  - min and max flux OBJECT / SKY = ",np.round(np.nanmin(ratio_object_sky_sl_gaussian),3),",",np.round(np.nanmax(ratio_object_sky_sl_gaussian),3),"  -> That is a variation of ",np.round(-100.*(np.nanmin(ratio_object_sky_sl_gaussian)-1),2),"% and ",np.round(100.*(np.nanmax(ratio_object_sky_sl_gaussian)-1),2),"%")
                print("                                                        but only fits with < ",max_flux_variation,"% have been considered")
            if plot== True and only_fibre == True:              
                #for i in range(len(sl_gauss_center)):
                #    print i+1, sl_gauss_center[i],ratio_object_sky_sl_gaussian[i]              
                plt.figure(figsize=(12, 5))
                plt.plot(sl_gauss_center,ratio_object_sky_sl_gaussian,"+",ms=12,mew=2)
                plt.axhline(y=np.nanmedian(ratio_object_sky_sl_gaussian), color="k", linestyle='--',alpha=0.3)
                plt.axhline(y=valid_median_flux, color="g", linestyle='-',alpha=0.3)
                plt.axhline(y=valid_median_flux+np.nanstd(good_ratio_values), color="c", linestyle=':',alpha=0.5)
                plt.axhline(y=valid_median_flux-np.nanstd(good_ratio_values), color="c", linestyle=':',alpha=0.5)              
                plt.axhline(y=min_flux_ratio, color="r", linestyle='-',alpha=0.5)
                plt.axhline(y=max_flux_ratio, color="r", linestyle='-',alpha=0.5)                
                #plt.ylim(0.7,1.3)                
                ptitle = "Checking flux OBJECT / SKY for fitted skylines in fibre "+np.str(fibre)#+" with rms = "+np.str(rms[i])
                plt.title(ptitle)
                plt.xlabel("Wavelength [$\mathrm{\AA}$]")
                plt.ylabel("OBJECT / SKY ")
                #plt.legend(frameon=True, loc=2, ncol=6)
                plt.minorticks_on()                
                plt.show()
                plt.close()
            
            self.wavelength_offset_per_fibre.append(offset)                   
            #self.sky_auto_scale.append(np.nanmedian(ratio_object_sky_sl_gaussian))
            self.sky_auto_scale.append(valid_median_flux)
                        
            if auto_scale_sky:
                if verbose:  print("  - As requested, using this value to scale sky spectrum before substraction... ")
                auto_scale = np.nanmedian(ratio_object_sky_sl_gaussian)
            else:
                if verbose:  print("  - As requested, DO NOT using this value to scale sky spectrum before substraction... ")
                auto_scale = 1.0
            if rebin:
                if verbose: 
                    print("\n> Rebinning the spectrum of fibre",fibre,"to match sky spectrum...")
                f = object_sl_gaussian_fitted
                f_new = rebin_spec_shift(w,f,offset)
            else:
                f_new = object_sl_gaussian_fitted
                
            # This must be corrected at then end to use the median auto_scale value
            #self.intensity_corrected[fibre] = f_new - auto_scale * sky_sl_gaussian_fitted
            f_new_ALL.append(f_new)
            sky_sl_gaussian_fitted_ALL.append(sky_sl_gaussian_fitted)

            if plot:
                plt.figure(figsize=(12, 5))
                plt.plot(w,spec,"purple", alpha=0.7, label="Obj")
                plt.plot(w,auto_scale*sky, "r", alpha=0.5, label ="Scaled sky")
                plt.plot(w,auto_scale*sky_sl_gaussian_fitted, "lime", alpha=0.8, label ="Scaled sky fit")
                plt.plot(w,object_sl_gaussian_fitted, "k", alpha=0.5, label="Obj - sky fit")
                plt.plot(w,spec-auto_scale*sky,"orange", alpha=0.4, label="Obj - scaled sky")
                plt.plot(w,object_sl_gaussian_fitted-sky_sl_gaussian_fitted,"b", alpha=0.9, label="Obj - sky fit - scale * rest sky")                        
                
                plt.xlim(wmin,wmax)                    
                plt.ylim(ymin,ymax)
                ptitle = "Fibre "+np.str(fibre)#+" with rms = "+np.str(rms[i])
                plt.title(ptitle)
                plt.xlabel("Wavelength [$\mathrm{\AA}$]")
                plt.ylabel("Flux [counts]")
                plt.legend(frameon=True, loc=2, ncol=6)
                plt.minorticks_on()
                for i in range(len(el_list)):    
                    plt.axvline(x=el_list[i], color="k", linestyle='-',alpha=0.5)   # MARIO
                for i in range(number_sl):                   
                    if skip_sl_fit[i]:
                        alpha = 0.1
                    else:
                        alpha = 0.6
                    if sl_fnl[i] == 1 :
                        plt.axvline(x=sl_center[i], color="brown", linestyle='-',alpha=alpha+0.4) # alpha=1)
                    else: plt.axvline(x=sl_center[i], color="y", linestyle='--',alpha=alpha)   
                for i in range(len(dsky2)-1):   
                    plt.axvline(x=dsky2[i], color="orange", linestyle='--',alpha=0.6)
                plt.show()
                plt.close()
           
            if only_fibre:
                ymax = np.nanpercentile(self.intensity_corrected[fibre], 99.5)
                ymin = np.nanpercentile(self.intensity_corrected[fibre], 0.1) - (np.nanpercentile(self.intensity_corrected[fibre], 99.5) - np.nanpercentile(self.intensity_corrected[fibre], 0.1))/15.  
                ymax = np.nanpercentile(self.intensity_corrected[fibre], 99.5)
                self.intensity_corrected[fibre] = f_new - auto_scale * sky_sl_gaussian_fitted
                plot_plot(w,[self.intensity_corrected[fibre],self.intensity[fibre]], color=["b","r"], ymin = ymin, ymax=ymax, ptitle="Comparison before (red) and after (blue) sky substraction using Gaussian fit to skylines")    
                print("\n  Only fibre" , fibre," is corrected, use fibre = -1 for all...")

        if only_fibre == False:
            # To avoid bad auto scaling with bright fibres or weird fibres, 
            # we fit a 2nd order polynomium to a filtered median value
            sas_m = signal.medfilt(self.sky_auto_scale, 21)   ## Assuming odd_number = 21
            #fit=np.polyfit(range(self.n_spectra),sas_m,2)   # If everything is OK this should NOT be a fit, but a median    
            fit = np.nanmedian(sas_m)
            #y=np.poly1d(fit)
            fity =[fit]*self.n_spectra
            #fity=y(range(self.n_spectra))
            if plot_step_fibres: 
                #ptitle = "Fit to autoscale values:\n"+np.str(y)
                ptitle = "Checking autoscale values, median value = "+np.str(np.round(fit,2))+" using median filter 21"
                ymin_ = np.nanmin(sas_m) - 0.1
                ymax_ = np.nanmax(sas_m) + 0.4
                plot_plot(list(range(self.n_spectra)),[sas_m,fity,self.sky_auto_scale,], 
                          color=["b","g","r"],alpha=[0.5,0.5,0.8], ptitle=ptitle, ymin=ymin_,ymax=ymax_,
                          xlabel="Fibre",ylabel="Flux ratio", label=["autoscale medfilt=21", "median","autoscale"])
                          #label=["autoscale med=21", "fit","autoscale"])
                        
            self.sky_auto_scale_fit = fity
            if auto_scale_sky:
                #print "  Correcting all fluxes adding the autoscale value of the FIT above for each fibre..."
                print("  Correcting all fluxes adding the median autoscale value to each fibre (green line)...")
            else:
                #print "  Correcting all fluxes WITHOUT CONSIDERING the autoscale value of the FIT above for each fibre..."
                print("  Correcting all fluxes WITHOUT CONSIDERING the median autoscale value ...")

            for fibre in range(self.n_spectra):
                if auto_scale_sky:
                    self.intensity_corrected[fibre] = f_new_ALL[fibre] -  self.sky_auto_scale_fit[fibre] * sky_sl_gaussian_fitted_ALL[fibre]
                else:
                    self.intensity_corrected[fibre] = f_new_ALL[fibre] -  sky_sl_gaussian_fitted_ALL[fibre]                                     
            print("\n  All fibres corrected for sky emission performing individual Gaussian fits to each fibre !") 
# -----------------------------------------------------------------------------        
# -----------------------------------------------------------------------------
    def do_extinction_curve(self, apply_extinction=True,  fig_size=12,
                            observatory_extinction_file='ssoextinct.dat', plot=True, verbose=True):
        """
        This task accounts and corrects for extinction due to gas and dusty between target and observer.
        creates a extinction curve based off airmass input and observatory file data
       
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
            print("\n> Computing extinction curve...")      
            print("  Airmass = ", np.round(self.airmass,3))
            print("  Observatory file with extinction curve :", observatory_extinction_file)

        # Read data
        data_observatory = np.loadtxt(observatory_extinction_file, unpack=True)
        extinction_curve_wavelenghts = data_observatory[0]
        extinction_curve = data_observatory[1]
        extinction_corrected_airmass = 10**(0.4*self.airmass*extinction_curve)
        # Make fit
        tck = interpolate.splrep(extinction_curve_wavelenghts, extinction_corrected_airmass, s=0)
        self.extinction_correction = interpolate.splev(self.wavelength, tck, der=0)
        
        if plot:
            cinco_por_ciento = 0.05 * (np.max(self.extinction_correction)- np.min(self.extinction_correction))
            plot_plot(extinction_curve_wavelenghts,extinction_corrected_airmass,xmin = np.min(self.wavelength),
                      xmax = np.max(self.wavelength),ymin = np.min(self.extinction_correction)-cinco_por_ciento,
                      ymax = np.max(self.extinction_correction)-cinco_por_ciento,
                      vlines = [self.valid_wave_min,self.valid_wave_max],
                      ptitle = 'Correction for extinction using airmass = '+str(np.round(self.airmass,3)),
                      xlabel = "Wavelength [$\mathrm{\AA}$]", ylabel = "Flux correction",fig_size = fig_size, statistics=False)
            
        # Correct for extinction at given airmass
        if apply_extinction:
            self.apply_extinction_correction(self.extinction_correction, observatory_extinction_file=observatory_extinction_file)
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
        #self.variance_corrected *= extinction_correction[np.newaxis, :]**2
        #self.corrections.append('Extinction correction')
        self.history.append("- Data corrected for extinction using file :")
        self.history.append("  "+observatory_extinction_file)
        self.history.append("  Average airmass = "+np.str(self.airmass))
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def find_sky_fibres(self, sky_wave_min=0, sky_wave_max=0, n_sky=200, plot = False, verbose=True, warnings=True):
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
        if sky_wave_min == 0 : sky_wave_min = self.valid_wave_min
        if sky_wave_max == 0 : sky_wave_max = self.valid_wave_max
        # Assuming cleaning of cosmics and CCD defects, we just use the spaxels with the LOWEST INTEGRATED VALUES                 
        self.compute_integrated_fibre(valid_wave_min=sky_wave_min, valid_wave_max=sky_wave_max, plot=False, verbose=verbose, warnings=warnings) 
        sorted_by_flux = np.argsort(self.integrated_fibre)  
        print("\n> Identifying sky spaxels using the lowest integrated values in the [",np.round(sky_wave_min,2),",",np.round(sky_wave_max,2),"] range ...")   
        print("  We use the lowest", n_sky, "fibres for getting sky. Their positions are:")   
        # Compute sky spectrum and plot RSS map with sky positions if requested
        self.sky_fibres = sorted_by_flux[:n_sky]         
        if plot: self.RSS_map(self.integrated_fibre, None, self.sky_fibres, title =" - Sky Spaxels")        
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def find_sky_emission(self, intensidad=[0,0], plot=True, n_sky=200,
                          sky_fibres=[], sky_wave_min=0, sky_wave_max=0,    #substract_sky=True, correct_negative_sky= False,
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
            if sky_wave_min == 0 : sky_wave_min = self.valid_wave_min
            if sky_wave_max == 0 : sky_wave_max = self.valid_wave_max
            self.find_sky_fibres(sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, n_sky=n_sky)
        else:     # We provide a list with sky positions
            print("  We use the list provided to get the sky spectrum")
            print("  sky_fibres = ",sky_fibres)
            self.sky_fibres = np.array(sky_fibres)

        if plot: self.RSS_map(self.integrated_fibre, None, self.sky_fibres, title =" - Sky Spaxels")
        print("  List of fibres used for sky saved in self.sky_fibres")

        if include_history : self.history.append("- Obtaining the sky emission using "+np.str(n_sky)+" fibres")
        self.sky_emission = sky_spectrum_from_fibres(self, self.sky_fibres, win_sky=win_sky, plot=False, include_history=include_history)

        if plot: plot_plot(self.wavelength, self.sky_emission, color="c", 
                      ylabel="Relative flux [counts]", xlabel="Wavelength [$\mathrm{\AA}$]",
                      xmin=self.wavelength[0]-10,xmax=self.wavelength[-1]+10,
                      ymin = np.nanpercentile(self.sky_emission,1), ymax = np.nanpercentile(self.sky_emission,99),
                      vlines=[self.valid_wave_min,self.valid_wave_max],
                      ptitle = "Combined sky spectrum using the requested fibres")                          
        print("  Sky spectrum obtained and stored in self.sky_emission !! ")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def substract_sky(self, correct_negative_sky=False, plot=True, verbose = True, warnings=True,
                      order_fit_negative_sky=3, kernel_negative_sky = 51,  exclude_wlm=[[0,0]], 
                      individual_check = True, use_fit_for_negative_sky = False, low_fibres=10):
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
        #for i in range(self.n_spectra):
        #    self.intensity_corrected[i,:]=self.intensity_corrected[i,:] - self.sky_emission
        self.intensity_corrected -= self.sky_emission[np.newaxis, :]               
        if len(self.sky_fibres) > 0: last_sky_fibre=self.sky_fibres[-1]            
        median_sky_corrected = np.zeros(self.n_spectra)
        
        for i in range(self.n_spectra):
                median_sky_corrected[i] = np.nanmedian(self.intensity_corrected[i,self.valid_wave_min_index:self.valid_wave_max_index],axis=0)
        if len(self.sky_fibres) > 0: median_sky_per_fibre=np.nanmedian(median_sky_corrected[self.sky_fibres])
        
        if verbose:
                print("  Median flux all fibres          = ",np.round(np.nanmedian(median_sky_corrected),3))
                if len(self.sky_fibres) > 0:
                    print("  Median flux sky fibres          = ",np.round(median_sky_per_fibre,3))
                    print("  Median flux brightest sky fibre = ",np.round(median_sky_corrected[last_sky_fibre],3))
                    print("  Median flux faintest  sky fibre = ",np.round(median_sky_corrected[self.sky_fibres[0]],3))
    
        # Plot median value of fibre vs. fibre           
        if plot: 
            
             if len(self.sky_fibres) > 0:
                 ymin = median_sky_corrected[self.sky_fibres[0]]-1
                 #ymax = np.nanpercentile(median_sky_corrected,90),
                 hlines=[np.nanmedian(median_sky_corrected),median_sky_corrected[self.sky_fibres[0]],median_sky_corrected[last_sky_fibre],median_sky_per_fibre]
                 chlines=["r","k","k","g"]
                 ptitle="Median flux per fibre after sky substraction\n (red = median flux all fibres, green = median flux sky fibres, grey = median flux faintest/brightest sky fibre)"
             else:
                 ymin=np.nanpercentile(median_sky_corrected, 1)
                 #ymax=np.nanpercentile(self.sky_emission, 1)
                 hlines =[np.nanmedian(median_sky_corrected),0]
                 chlines=["r","k"]
                 ptitle="Median flux per fibre after sky substraction (red = median flux all fibres)"
            
             plot_plot(list(range(self.n_spectra)), median_sky_corrected,
                      ylabel="Median Flux [counts]",xlabel="Fibre", 
                      ymin = ymin, ymax = np.nanpercentile(median_sky_corrected,90),
                      hlines=hlines, chlines=chlines,
                      ptitle=ptitle)
        
        if len(self.sky_fibres) > 0: 
            if median_sky_corrected[self.sky_fibres[0]] < 0:
                if verbose or warnings: print("  WARNING !  The integrated value of the sky fibre with the smallest value is negative!")
                if correct_negative_sky:        
                    if verbose: print("  Fixing this, as 'correct_negative_sky' = True  ... ")
                    self.correcting_negative_sky(plot=plot, low_fibres=low_fibres, exclude_wlm=exclude_wlm, kernel_negative_sky=kernel_negative_sky, 
                                                 use_fit_for_negative_sky = use_fit_for_negative_sky,
                                                 order_fit_negative_sky=order_fit_negative_sky, individual_check=individual_check) 
        
        if verbose: print("  Intensities corrected for sky emission and stored in self.intensity_corrected !")   
        self.history.append("  Intensities corrected for the sky emission")              
# -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------- 
    def get_telluric_correction(self, n_fibres=10, correct_from=6850., correct_to=10000., 
                                save_telluric_file = "",
                                apply_tc=False, step = 10, is_combined_cube=False, weight_fit_median = 0.5,
                                exclude_wlm=[[6450,6700],[6850,7050], [7130,7380]], # This is range for 1000R
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
            wlm=self.combined_cube.wavelength
        else:
            wlm=self.wavelength
            
        if wave_min == 0 : wave_min=wlm[0]
        if wave_max == 0 : wave_max=wlm[-1]
        
        if is_combined_cube:
            if self.combined_cube.seeing == 0: 
                self.combined_cube.half_light_spectrum(5, plot=plot, min_wave=wave_min, max_wave=wave_max)
            estrella = self.combined_cube.integrated_star_flux                                    
        else:
            integrated_intensity_sorted=np.argsort(self.integrated_fibre)
            intensidad=self.intensity_corrected 
            region=[]
            for fibre in range(n_fibres):
                region.append(integrated_intensity_sorted[-1-fibre])
            estrella=np.nansum(intensidad[region], axis=0)    
                  
        smooth_med_star=smooth_spectrum(wlm, estrella, wave_min=wave_min, wave_max=wave_max, step=step, weight_fit_median=weight_fit_median,
                                            exclude_wlm=exclude_wlm, plot=plot, verbose=verbose)
                
        telluric_correction = np.ones(len(wlm))
        
        estrella_m = signal.medfilt(estrella,151)
        plot_plot(wlm,[estrella,smooth_med_star, estrella_m ])

        # Avoid H-alpha absorption
        rango_ha=[0,0]
        for rango in exclude_wlm:
            if rango[0] < 6563. and rango[1] > 6563.:  # H-alpha is here, skip
                print("  Skipping range with H-alpha...")
                rango_ha = rango
            
        correct_from= 6000.
        for l in range(len(wlm)):
            if wlm[l] > correct_from and wlm[l]< correct_to:
                
                if wlm[l] > rango_ha[0] and wlm[l] < rango_ha[1]:
                    step=step+0
                    #skipping Ha
                else:
                    telluric_correction[l]= smooth_med_star[l]/estrella[l]   

        waves_for_tc_ =[]
        for rango in exclude_wlm:
            if rango[0] < 6563. and rango[1] > 6563.:  # H-alpha is here, skip
                print("  Skipping range with H-alpha...")
            else:
                index_region = np.where( (wlm >= rango[0]) & (wlm <= rango[1]))   
                waves_for_tc_.append(index_region)    

        waves_for_tc =[]
        for rango in waves_for_tc_:
            waves_for_tc=np.concatenate( (waves_for_tc, rango[0].tolist()), axis=None)
            
        # Now, change the value in telluric_correction
        for index in waves_for_tc:
            i= np.int(index)
            if smooth_med_star[i]/estrella[i] >1. :
                telluric_correction[i] = smooth_med_star[i]/estrella[i]
 
        if plot: 
            plt.figure(figsize=(fig_size, fig_size/2.5)) 
            if is_combined_cube:
                print("  Telluric correction for this star ("+self.combined_cube.object+") :")
                plt.plot(wlm, estrella, color="b", alpha=0.3)
                plt.plot(wlm, estrella*telluric_correction, color="g", alpha=0.5)
                plt.ylim(np.nanmin(estrella),np.nanmax(estrella))          
            else:
                print("  Example of telluric correction using fibres",region[0]," (blue) and ",region[1]," (green):")               
                plt.plot(wlm, intensidad[region[0]], color="b", alpha=0.3)
                plt.plot(wlm, intensidad[region[0]]*telluric_correction, color="g", alpha=0.5)
                plt.plot(wlm, intensidad[region[1]], color="b", alpha=0.3)
                plt.plot(wlm, intensidad[region[1]]*telluric_correction, color="g", alpha=0.5)
                plt.ylim(np.nanmin(intensidad[region[1]]),np.nanmax(intensidad[region[0]]))   # CHECK THIS AUTOMATICALLY                   
            plt.axvline(x=wave_min, color='k', linestyle='--')
            plt.axvline(x=wave_max, color='k', linestyle='--')
            plt.xlim(wlm[0]-10,wlm[-1]+10)   
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
                        self.combined_cube.data[:,i,j] = self.combined_cube.data[:,i,j] * telluric_correction               
            else:    
                for i in range(self.n_spectra):
                    self.intensity_corrected[i,:]=self.intensity_corrected[i,:] * telluric_correction               
        else:
            print("  As apply_tc = False , telluric correction is NOT applied...")
    
        if is_combined_cube:
            self.combined_cube.telluric_correction =   telluric_correction 
        else:
            self.telluric_correction =   telluric_correction 
               
        # save file if requested
        if save_telluric_file != "":
            spectrum_to_text_file(wlm, telluric_correction, filename=save_telluric_file) 
               
        return telluric_correction  
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def plot_spectrum(self, spectrum_number, sky=True, xmin="",xmax="",ymax="",ymin=""):
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
            spectrum = self.intensity_corrected[spectrum_number]+self.sky_emission
            
        ptitle = self.description+" - Fibre "+np.str(spectrum_number)    
        plot_plot(self.wavelength, spectrum, xmin=xmin,xmax=xmax,ymax=ymax,ymin=ymin,
                  ptitle=ptitle, statistics=True) # TIGRE                                 
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def plot_spectra(self, list_spectra='all', wavelength_range=[0], 
                     xmin="",xmax="",ymax=1000,ymin=-100, sky=True,
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
        plt.figure(figsize=(fig_size, fig_size/2.5))         

        if list_spectra == 'all': list_spectra = list(range(self.n_spectra))
        if len(wavelength_range) == 2: plt.xlim(wavelength_range[0], wavelength_range[1])
        if xmin == "": xmin = np.nanmin(self.wavelength)
        if xmax == "": xmax = np.nanmax(self.wavelength)                 
        plt.minorticks_on() 
        plt.xlabel("Wavelength [$\mathrm{\AA}$]")
        plt.ylabel("Relative Flux")
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        
        for i in list_spectra:
            if sky:
                spectrum = self.intensity_corrected[i]
            else:
                spectrum = self.intensity_corrected[i]+self.sky_emission            
            plt.plot(self.wavelength, spectrum)
            
        if save_file == "":
           plt.show()
        else:
           plt.savefig(save_file)
        plt.close()    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def plot_combined_spectrum(self, list_spectra='',sky=True, median=False, ptitle="",
                     xmin="",xmax="",ymax="",ymin="",  percentile_min=2, percentile_max=98,
                     plot= True, fig_size=10, save_file=""):
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
        value_list=[]
        
        if sky:
            for fibre in list_spectra:
                value_list.append(self.intensity_corrected[fibre])
        else:
            for fibre in list_spectra:
                value_list.append(self.intensity_corrected[fibre]+self.sky_emission)

        if median:
            spectrum = np.nanmedian(value_list, axis=0) 
        else:
            spectrum = np.nansum(value_list, axis=0)            

        if plot:       
            vlines=[self.valid_wave_min, self.valid_wave_max]                   
            if len(list_spectra) == list_spectra[-1] - list_spectra[0] + 1:
                if ptitle == "": ptitle = "{} - Combined spectrum in range [{},{}]".format(self.description, list_spectra[0], list_spectra[-1])
            else:
                if ptitle == "": ptitle = "Combined spectrum using requested fibres"         
            plot_plot(self.wavelength, spectrum, xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,vlines=vlines,
                      ptitle=ptitle,save_file=save_file, percentile_min=percentile_min, percentile_max=percentile_max)
            
        return spectrum
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def flux_between(self, lambda_min, lambda_max, list_spectra=[]):
        """
        Computes and returns the flux in range  [lambda_min, lambda_max] of a list of spectra.

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
        index_max = np.searchsorted(self.wavelength, lambda_max)+1
        if len(list_spectra) == 0:
            list_spectra = list(range(self.n_spectra))

        n_spectra = len(list_spectra)
        fluxes = np.empty(n_spectra)
        variance = np.empty(n_spectra)
        for i in range(n_spectra):
            fluxes[i] = np.nanmean(self.intensity[list_spectra[i],
                                                  index_min:index_max])
            variance[i] = np.nanmean(self.variance[list_spectra[i],
                                                   index_min:index_max])

        return fluxes*(lambda_max-lambda_min), variance*(lambda_max-lambda_min)
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
        index_max = np.searchsorted(self.wavelength, lambda_max)+1
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
        w=self.wavelength
        if wave_index == -1:
            _w_=np.abs(w-wave)
            w_min = np.nanmin(_w_) 
            wave_index=_w_.tolist().index(w_min)
        else:
            wave=w[wave_index]
        corte_wave=self.intensity_corrected[:,wave_index]        
    
        if plot:
            x=range(self.n_spectra)
            ptitle="Intensity cut at "+np.str(wave)+" $\mathrm{\AA}$ - index ="+np.str(wave_index)
            plot_plot(x,corte_wave, ymax=ymax, xlabel="Fibre", ptitle=ptitle)
        
        return corte_wave
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def RSS_map(self, variable=[0], norm=colors.LogNorm(), list_spectra=[], log="",
                title = " - RSS map", clow="", chigh="",
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
        
        if log == False : norm=colors.Normalize()
        
        if len(list_spectra) == 0:
            list_spectra = list(range(self.n_spectra))
                
        plt.figure(figsize=(10, 10))
        plt.scatter(self.offset_RA_arcsec[list_spectra],
                    self.offset_DEC_arcsec[list_spectra],
                    c=variable[list_spectra], cmap=fuego_color_map, norm=norm,
                    s=260, marker="h")
        plt.title(self.description+title)
        plt.xlim(np.nanmin(self.offset_RA_arcsec)-0.7, np.nanmax(self.offset_RA_arcsec)+0.7)
        plt.ylim(np.nanmin(self.offset_DEC_arcsec)-0.7, np.nanmax(self.offset_DEC_arcsec)+0.7)
        plt.xlabel("$\Delta$ RA [arcsec]")
        plt.ylabel("$\Delta$ DEC [arcsec]")
        plt.minorticks_on()
        plt.grid(which='both')
        plt.gca().invert_xaxis()

        cbar=plt.colorbar()
        if clow == "": clow=np.nanmin(variable[list_spectra])
        if chigh == "": chigh=np.nanmax(variable[list_spectra])    
        plt.clim(clow,chigh)
        cbar.set_label(str(color_bar_text), rotation=90, labelpad=40)        
        cbar.ax.tick_params()
        
        plt.show()
        plt.close()
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def RSS_image(self, image="", norm=colors.Normalize(), cmap="seismic_r", clow="", chigh="", labelpad=10, log=False,
                  title = " - RSS image", color_bar_text="Integrated Flux [Arbitrary units]", fig_size=13.5): 
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
        
        if log: norm=colors.LogNorm()
        
        if image == "":
            image = self.intensity_corrected
        
        if clow == "": 
            clow=np.nanpercentile(image,5) 
        if chigh == "": 
            chigh=np.nanpercentile(image,95) 
        if cmap == "seismic_r" :
            max_abs=np.nanmax([np.abs(clow),np.abs(chigh)])
            clow= -max_abs
            chigh= max_abs

        plt.figure(figsize=(fig_size, fig_size/2.5))
        plt.imshow(image, norm=norm, cmap=cmap, clim=(clow,chigh))
        plt.title(self.description+title)
        plt.xlim(0, self.n_wave)
        plt.minorticks_on()
        plt.gca().invert_yaxis()
        
        #plt.colorbar()
        cbar=plt.colorbar()
        cbar.set_label(str(color_bar_text), rotation=90, labelpad=labelpad)        
        cbar.ax.tick_params()        
        
        plt.show()
        plt.close()   
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def plot_corrected_vs_uncorrected_spectrum(self, high_fibres=20, low_fibres=0, kernel=51,
                                               fig_size=12, fcal=False, verbose = True):
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
        plt.figure(figsize=(fig_size, fig_size/2.5)) 
        
        integrated_intensity_sorted=np.argsort(self.integrated_fibre)
        region=[]
        
        if low_fibres == 0:
            for fibre_ in range(high_fibres):
                region.append(integrated_intensity_sorted[-1-fibre_])
            if verbose : print("\n> Checking combined spectrum using",high_fibres,"fibres with the highest integrated intensity")
            plt.title(self.object+" - Combined spectrum - "+str(high_fibres)+" fibres with highest intensity")  
            I=np.nansum(self.intensity[region], axis=0)            
            Ic=np.nansum(self.intensity_corrected[region], axis=0) 
        else:
            for fibre_ in range(low_fibres):
                region.append(integrated_intensity_sorted[fibre_])
            if verbose: print("\n> Checking median spectrum using",low_fibres,"fibres with the lowest integrated intensity")
            plt.title(self.object+" - Median spectrum - "+str(low_fibres)+" fibres with lowest intensity")  
            I=np.nanmedian(self.intensity[region], axis=0)            
            Ic=np.nanmedian(self.intensity_corrected[region], axis=0)
        if verbose: print("  which are :",region)  
                                        

        Ic_m,fit=fit_smooth_spectrum(self.wavelength, Ic, kernel=kernel,  verbose=False, #edgelow=0, edgehigh=0,
                                     order=3, plot=False, hlines=[0.], fcal = False)     # ptitle= ptitle,       
        
        I_ymin = np.nanmin(Ic_m)
        I_ymax = np.nanmax(Ic_m)        
        I_rango = I_ymax-I_ymin

        plt.plot(self.wavelength, I, 'r-', label='Uncorrected', alpha=0.3)        
        plt.plot(self.wavelength, Ic, 'g-', label='Corrected', alpha=0.4)
        
        text="Corrected with median filter "+np.str(kernel)
        
        if low_fibres > 0 : 
            plt.plot(self.wavelength, Ic_m, 'b-', label=text, alpha=0.4)
            plt.plot(self.wavelength, fit, color="purple", linestyle='-', label='Fit', alpha=0.4)
        if fcal:    
            ylabel = "Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]"
        else:
            ylabel="Flux [counts]"
        plt.ylabel(ylabel)    
        plt.xlabel("Wavelength [$\mathrm{\AA}$]")
        plt.minorticks_on()
        plt.xlim(self.wavelength[0]-10,self.wavelength[-1]+10)
        plt.axvline(x=self.valid_wave_min, color='k', linestyle='--', alpha=0.8)
        plt.axvline(x=self.valid_wave_max, color='k', linestyle='--', alpha=0.8)
        if low_fibres == 0: 
            plt.ylim([I_ymin-I_rango/10,I_ymax+I_rango/10])
        else:
            plt.axhline(y=0., color="k", linestyle="--", alpha=0.8) #teta
            I_ymin = np.nanpercentile(Ic,2)
            I_ymax = np.nanpercentile(Ic,98)
            I_rango = I_ymax-I_ymin
            plt.ylim([I_ymin-I_rango/10,I_ymax+I_rango/10])
        plt.legend(frameon=False, loc=4, ncol=4)
        plt.show()
        plt.close()          
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
            if len(sky_lines_file) == 0: sky_lines_file="sky_lines_rest.dat"
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
    def apply_throughput_2D(self, throughput_2D=[], throughput_2D_file="", path="", plot=True):
        """
        Apply throughput_2D using the information of a variable or a fits file.
        """    
        if len(throughput_2D) > 0:     
            print("\n> Applying 2D throughput correction using given variable ...")
            self.throughput_2D = throughput_2D
            self.history.append("- Applied 2D throughput correction using a variable")
        else:
            if path != "": throughput_2D_file = full_path(throughput_2D_file,path)
            print("\n> Applying 2D throughput correction reading file :")
            print(" ",throughput_2D_file)
            self.history.append("- Applied 2D throughput correction using file:")
            self.history.append("  "+throughput_2D_file)              
            ftf = fits.open(throughput_2D_file)
            self.throughput_2D = ftf[0].data 
        if plot: 
            print("\n> Plotting map BEFORE correcting throughput:")
            self.RSS_image()
        
        self.intensity_corrected = self.intensity_corrected / self.throughput_2D
        if plot: 
            print("  Plotting map AFTER correcting throughput:")
            self.RSS_image()        
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def fix_edges(self, kernel_size=101, disp=1.5 , fix_from=0, median_from=0, fix_to=0, median_to=0, 
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
        if only_blue_edge == True : do_red_edge = False
        if verbose: print("\n> Fixing the BLUE and RED edges of the RSS file...")
        w = self.wavelength
        self.RSS_image(title=" - Before correcting edges")
        if fix_from==0: fix_from = self.valid_wave_max
        if median_from == 0 : median_from = self.valid_wave_max - 300.
        if fix_to == 0 : fix_to = self.valid_wave_min
        if median_to == 0: median_to = self.valid_wave_min + 200.       
        self.apply_mask(make_nans=True, verbose=False)
        for i in range(self.n_spectra):        
            if do_red_edge: self.intensity_corrected[i]=fix_red_edge(w,self.intensity_corrected[i],fix_from=fix_from, median_from=median_from,kernel_size=kernel_size, disp=disp )    
            if do_blue_edge: self.intensity_corrected[i]=fix_blue_edge(w,self.intensity_corrected[i], kernel_size=kernel_size, disp=disp, fix_to=fix_to, median_to=median_to)    
        
        self.RSS_image(title=" - After correcting edges")
        if do_blue_edge: self.history.append("- Blue edge has been corrected to "+np.str(fix_to))
        if do_red_edge: self.history.append("- Red edge has been corrected from "+np.str(fix_from))
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def kill_cosmics(self, brightest_line_wavelength, width_bl = 20., fibre_list=[], max_number_of_cosmics_per_fibre = 10,
                     kernel_median_cosmics = 5, cosmic_higher_than = 100, extra_factor = 1., 
                     plot_waves=[], plot_cosmic_image=True, plot_RSS_images= True, plot=True,
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
            plot_RSS_images= False
            plot_cosmic_image=False
        
        x=range(self.n_spectra)
        w=self.wavelength
        if len(fibre_list) == 0 : 
            fibre_list_ALL = True
            fibre_list=list(range(self.n_spectra))
            if verbose: print("\n> Finding and killing cosmics in all fibres...")
        else:
            fibre_list_ALL = False
            if verbose: print("\n> Finding and killing cosmics in given fibres...")
            
        if brightest_line_wavelength == 0:
            if warnings or verbose: print("\n\n\n  WARNING !!!!! brightest_line_wavelength is NOT given!\n")

            median_spectrum=self.plot_combined_spectrum(plot=plot,median=True, list_spectra=self.integrated_fibre_sorted[-11:-1], 
                                                        ptitle = "Combined spectrum using 10 brightest fibres", percentile_max=99.5, percentile_min=0.5)
            #brightest_line_wavelength=w[np.int(self.n_wave/2)]
            brightest_line_wavelength=self.wavelength[median_spectrum.tolist().index(np.nanmax(median_spectrum))]
            
            if brightest_line_wavelength < self.valid_wave_min: brightest_line_wavelength = self.valid_wave_min
            if brightest_line_wavelength > self.valid_wave_max: brightest_line_wavelength = self.valid_wave_max
            
            if warnings or verbose: print("  Assuming brightest_line_wavelength is the max of median spectrum of 10 brightest fibres =", brightest_line_wavelength)
       
        # Get the cut at the brightest_line_wavelength
        corte_wave_bl=self.cut_wave(brightest_line_wavelength)
        gc_bl=signal.medfilt(corte_wave_bl,kernel_size=kernel_median_cosmics)
        max_val = np.abs(corte_wave_bl-gc_bl)
        
        if plot:
            ptitle="Intensity cut at brightest line wavelength = "+np.str(np.round(brightest_line_wavelength,2))+" $\mathrm{\AA}$ and extra_factor = "+np.str(extra_factor)
            plot_plot(x,[max_val, extra_factor*max_val],percentile_max=99,xlabel="Fibre", ptitle=ptitle, ylabel="abs (f - medfilt(f))",
                      label=["intensity_cut","intensity_cut * extra_factor"])
        
        # List of waves to plot:
        plot_waves_index=[]
        for wave in plot_waves:
            wave_min_vector=np.abs(w-wave)
            plot_waves_index.append(wave_min_vector.tolist().index(np.nanmin(wave_min_vector)))       
        if len(plot_waves) > 0 : print("  List of waves to plot:",plot_waves)
        
        # Start loop
        lista_cosmicos =[]
        cosmic_image=np.zeros_like(self.intensity_corrected)
        for i in range(len(w)):
            wave=w[i]
            # Perhaps we should include here not cleaning in emission lines...
            correct_cosmics_in_fibre = True
            if width_bl != 0:
                if wave > brightest_line_wavelength - width_bl/2 and wave < brightest_line_wavelength + width_bl/2:
                    if verbose: print("  Skipping {:.4f} as it is adjacent to brightest line wavelenght {:.4f}".format(wave,brightest_line_wavelength))
                    correct_cosmics_in_fibre = False
            if correct_cosmics_in_fibre:
                if i in plot_waves_index:
                    plot_=True
                    verbose_=True
                else:
                    plot_=False
                    verbose_=False
                corte_wave=self.cut_wave(wave)
                cosmics_found = find_cosmics_in_cut(x,corte_wave, corte_wave_bl*extra_factor, line_wavelength= wave, plot=plot_, verbose=verbose_, cosmic_higher_than=cosmic_higher_than)
                if len(cosmics_found) < max_number_of_cosmics_per_fibre :
                    for cosmic in cosmics_found:
                        lista_cosmicos.append([wave,cosmic])
                        cosmic_image[cosmic,i] = 1.   
                else:
                    if warnings: print("  WARNING! Wavelength",np.round(wave,2),"has",len(cosmics_found),"cosmics found, this is larger than",max_number_of_cosmics_per_fibre,"and hence these are NOT corrected!")

        # Check number of cosmics found
        if plot_cosmic_image: self.RSS_image(image=cosmic_image, cmap="binary_r", title=" - Cosmics identification")
        #print(lista_cosmicos)
        if verbose: print("\n> Total number of cosmics found = ",len(lista_cosmicos), " , correcting now ...")                
                
        if plot_RSS_images: self.RSS_image(cmap="binary_r", title=" - Before correcting cosmics")
    
        if fibre_list_ALL == False and verbose== True: print("  Correcting cosmics in selected fibres...")
        cosmics_cleaned=0
        for fibre in fibre_list:
            if np.nansum(cosmic_image[fibre]) > 0 :  # A cosmic is found
                #print("Fibre ",fibre," has cosmics!")
                f=g[fibre]
                gc=signal.medfilt(f,kernel_size=21)
                bad_indices = [i for i, x in enumerate(cosmic_image[fibre]) if x == 1]
                if len(bad_indices) < max_number_of_cosmics_per_fibre:
                    for index in bad_indices:
                        g[fibre,index] = gc[index]
                        cosmics_cleaned =cosmics_cleaned+1
                else:
                    cosmic_image[fibre]=np.zeros_like(w)
                    if warnings: print("  WARNING! Fibre",fibre,"has",len(bad_indices),"cosmics found, this is larger than",max_number_of_cosmics_per_fibre,"and hence is NOT corrected!")
        
        self.intensity_corrected = copy.deepcopy(g)        
        if plot_RSS_images: self.RSS_image(cmap="binary_r", title=" - After correcting cosmics")
        
        # Check number of cosmics eliminated
        if verbose: print("\n> Total number of cosmics cleaned = ",cosmics_cleaned)
        if cosmics_cleaned != len(lista_cosmicos):       
            if plot_cosmic_image: self.RSS_image(image=cosmic_image, cmap="binary_r", title=" - Cosmics cleaned")
        
        self.history.append("- "+np.str(cosmics_cleaned)+" cosmics cleaned using:")
        self.history.append("  brightest_line_wavelength = "+np.str(brightest_line_wavelength))
        self.history.append("  width_bl = "+np.str(width_bl)+", kernel_median_cosmics = "+np.str(kernel_median_cosmics))
        self.history.append("  cosmic_higher_than = "+np.str(cosmic_higher_than)+", extra_factor = "+np.str(extra_factor))
        return g
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------  
    def clean_extreme_negatives(self, fibre_list=[], percentile_min = 0.5, plot=True, verbose=True):
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
        if len(fibre_list) == 0 : 
            fibre_list=list(range(self.n_spectra))
            if verbose: print("\n> Correcting the extreme negatives in all fibres, making any pixel below") 
        else:
            if verbose: print("\n> Correcting the extreme negatives in given fibres, making any pixel below") 
                   
        g = copy.deepcopy(self.intensity_corrected)
        minimo=np.nanpercentile(g, percentile_min)
    
        if verbose:
            print("  np.nanpercentile(intensity_corrected, ", percentile_min, ") = ",np.round(minimo,2))
            print("  to have the median value of the fibre...")           
                
        for fibre in fibre_list:
            g[fibre] = [np.nanmedian(g[fibre]) if x < minimo else x for x in g[fibre]  ]
        self.history.append("- Extreme negatives (values below percentile "+np.str(np.round(percentile_min,3))+" = "+np.str(np.round(minimo,3))+" ) cleaned")                  
        
        if plot:
            correction_map = g / self.intensity_corrected #/ g
            self.RSS_image(image=correction_map, cmap="binary_r", title=" - Correction map")
                      
        self.intensity_corrected = g        
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------  
    def apply_telluric_correction(self,telluric_correction_file = "", telluric_correction = [0], 
                                  plot = True, fig_size = 12, verbose = True):     
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
        if telluric_correction_file != ""  :

            print("\n> Reading file with the telluric correction: ")
            print(" ",telluric_correction_file)          
            w_star,telluric_correction = read_table(telluric_correction_file, ["f", "f"] )
            
        
        if telluric_correction[0] != 0 :
            plot_integrated_fibre_again = plot_integrated_fibre_again + 1  
            self.telluric_correction = telluric_correction
            
            #print(telluric_correction)
            
            print("\n> Applying telluric correction...")
            
            before_telluric_correction = copy.deepcopy(self.intensity_corrected)
            for i in range(self.n_spectra):
                self.intensity_corrected[i,:]=self.intensity_corrected[i,:] * telluric_correction 
                
            if plot:
                plot_plot(self.wavelength, telluric_correction, xmin = self.wavelength[0]-10,  xmax = self.wavelength[-1]+10, statistics = False,
                ymin = 0.9, ymax = 2, ptitle = "Telluric correction", xlabel = "Wavelength [$\mathrm{\AA}$]" , vlines = [self.valid_wave_min,self.valid_wave_max])
                
                integrated_intensity_sorted=np.argsort(self.integrated_fibre)
                region=[integrated_intensity_sorted[-1],integrated_intensity_sorted[0]]
                print("  Example of telluric correction using faintest fibre",region[1],":")
                ptitle="Telluric correction in fibre " +np.str(region[1])
                plot_plot(self.wavelength,[before_telluric_correction[region[1]],self.intensity_corrected[region[1]]],
                          xmin = self.wavelength[0]-10, xmax = self.wavelength[-1]+10, ymin = np.nanpercentile(self.intensity_corrected[region[1]], 1),
                          ymax = np.nanpercentile(self.intensity_corrected[region[1]], 99), vlines = [self.valid_wave_min,self.valid_wave_max],
                          xlabel = "Wavelength [$\mathrm{\AA}$]", ptitle = ptitle)
                print("  Example of telluric correction using brightest fibre",region[0],":")
                ptitle="Telluric correction in fibre " +np.str(region[0])
                plot_plot(self.wavelength,[before_telluric_correction[region[0]],self.intensity_corrected[region[0]]],
                          xmin = self.wavelength[0]-10, xmax = self.wavelength[-1]+10, ymin = np.nanpercentile(self.intensity_corrected[region[0]], 1),
                          ymax = np.nanpercentile(self.intensity_corrected[region[0]], 99), vlines = [self.valid_wave_min,self.valid_wave_max],
                          xlabel = "Wavelength [$\mathrm{\AA}$]", ptitle = ptitle)                        

            if telluric_correction_file != "":
                self.history.append("- Telluric correction applied reading from file:")
                self.history.append("  "+telluric_correction_file)
            else:
                self.history.append("- Telluric correction applied using a Python variable")   
        else:
            self.telluric_correction  = np.ones_like(self.wavelength)
            if self.grating in red_gratings:# and rss_clean == False         
                if verbose: print("\n> Telluric correction will NOT be applied...")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------                  
    def apply_self_sky(self, sky_fibres = [], sky_spectrum = [], plot = True,
                       sky_wave_min = 0, sky_wave_max = 0, win_sky = 0, scale_sky_1D = 0,
                       brightest_line = "Ha", brightest_line_wavelength = 0, ranges_with_emission_lines = [0],
                       cut_red_end = 0, low_fibres = 10, use_fit_for_negative_sky = False, kernel_negative_sky = 51,
                       order_fit_negative_sky = 3, verbose = True, n_sky = 50, correct_negative_sky = False, individual_check = False):
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
        
        if len(sky_fibres) != 0: 
            n_sky=len(sky_fibres)               
            print("\n> 'sky_method = self', using list of",n_sky,"fibres to create a sky spectrum ...")
            self.history.append('  A list of '+np.str(n_sky)+' fibres was provided to create the sky spectrum')
            self.history.append(np.str(sky_fibres))
        else:
            print("\n> 'sky_method = self', hence using",n_sky,"lowest intensity fibres to create a sky spectrum ...")
            self.history.append('  The '+np.str(n_sky)+' lowest intensity fibres were used to create the sky spectrum')
    
        if len(sky_spectrum) == 0 :
            self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                                   sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max,                              
                                   win_sky=win_sky, include_history=True)
    
        else:
            print("  Sky spectrum provided. Using this for replacing regions with bright emission lines...")
    
            self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                           sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, 
                           win_sky=win_sky, include_history=False) 
    
            sky_r_self = self.sky_emission
            
            self.sky_emission = replace_el_in_sky_spectrum(self, sky_r_self, sky_spectrum, scale_sky_1D = scale_sky_1D,
                                                           brightest_line=brightest_line, 
                                                           brightest_line_wavelength = brightest_line_wavelength,
                                                           ranges_with_emission_lines = ranges_with_emission_lines,
                                                           cut_red_end=cut_red_end,
                                                           plot=plot)
            self.history.append('  Using sky spectrum provided for replacing regions with emission lines')
    
        self.substract_sky(plot=plot, low_fibres=low_fibres,
                           correct_negative_sky=correct_negative_sky,  use_fit_for_negative_sky = use_fit_for_negative_sky,
                           kernel_negative_sky = kernel_negative_sky, order_fit_negative_sky=order_fit_negative_sky,
                           individual_check=individual_check)
    
        self.apply_mask(verbose=verbose, make_nans=True)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------  
    def apply_1D_sky(self, sky_fibres = [], sky_spectrum = [], sky_wave_min = 0, sky_wave_max = 0,
                     win_sky = 0, include_history = True,
                     scale_sky_1D = 0, remove_5577 = True, sky_spectrum_file = "",
                     plot = True, verbose = True, n_sky = 50):
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
        if sky_spectrum_file != "":
            
            if verbose:        
                print("\n> Reading file with a 1D sky spectrum :")
                print(" ",sky_spectrum_file)
            
            w_sky,sky_spectrum = read_table(sky_spectrum_file, ["f", "f"] )
            
            if np.nanmedian(self.wavelength-w_sky) != 0:
                if verbose or warnings:
                    print("\n\n  WARNING !!!! The wavelengths provided on this file do not match the wavelengths on this RSS !!\n\n") 
            
            self.history.append('- 1D sky spectrum provided in file :')
            self.history.append('  '+sky_spectrum_file)
            
        if verbose:    
            print("\n> Sustracting the sky using the sky spectrum provided, checking the scale OBJ/SKY...")  
        if scale_sky_1D == 0:
            if verbose:                      
                print("  No scale between 1D sky spectrum and object given, calculating...")
        
        
            # TODO !
            # Task "scale_sky_spectrum" uses sky lines, needs to be checked...
            #self.sky_emission,scale_sky_1D_auto=scale_sky_spectrum(self.wavelength, sky_spectrum, self.intensity_corrected, 
            #                                     cut_sky=cut_sky, fmax=fmax, fmin=fmin, fibre_list=fibre_list)
        
            # Find self sky emission using only the lowest n_sky fibres (this should be small, 20-25)
            if n_sky == 50: n_sky =20
            self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                           sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, 
                           win_sky=win_sky,  include_history=include_history) 
        
            sky_r_self = self.sky_emission
        
            scale_sky_1D  = auto_scale_two_spectra(self, sky_r_self, sky_spectrum, scale=[0.1,1.01,0.025], 
                                                   w_scale_min = self.valid_wave_min,  w_scale_max = self.valid_wave_max, plot=plot, verbose = True )
        
           
        elif verbose:
            print("  As requested, we scale the given 1D sky spectrum by",scale_sky_1D)

        self.sky_emission=sky_spectrum *   scale_sky_1D       
        self.history.append('  1D sky spectrum scaled by a factor '+np.str(scale_sky_1D))
        
        if verbose: print("\n> Scaled sky spectrum stored in self.sky_emission, substracting to all fibres...")
                
        # For blue spectra, remove 5577 in the sky spectrum...                   
        if self.valid_wave_min < 5577 and remove_5577 == True:
            if verbose: print("  Removing sky line 5577.34 from the sky spectrum...") 
            resultado = fluxes(self.wavelength, self.sky_emission, 5577.34, lowlow=30, lowhigh=10, highlow=10, highhigh=30, 
                               plot=False, verbose=False)  #fmin=-5.0E-17, fmax=2.0E-16, 
            #resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
            self.sky_emission = resultado[11] 
        else:
            if self.valid_wave_min < 5577 and verbose: print("  Sky line 5577.34 is not removed from the sky spectrum...")
            
            
        # Remove 5577 in the object
        if self.valid_wave_min < 5577 and remove_5577 == True and scale_sky_1D == 0:# and individual_sky_substraction == False:      
            if verbose:                
                print("  Removing sky line 5577.34 from the object...")
            wlm=self.wavelength
            for i in range(self.n_spectra):
                s = self.intensity_corrected[i]
                # Removing Skyline 5577 using Gaussian fit if requested
                resultado = fluxes(wlm, s, 5577.34, lowlow=30, lowhigh=10, highlow=10, highhigh=30,
                                   plot=False, verbose=False)  #fmin=-5.0E-17, fmax=2.0E-16, 
                #resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]
                self.intensity_corrected[i] = resultado[11]
        else:
            if self.valid_wave_min < 5577 and verbose: 
                if scale_sky_1D == 0:
                    print("  Sky line 5577.34 is not removed from the object...")
                else:
                    print("  Sky line 5577.34 already removed in object during CCD cleaning...")
                                                                              
        self.substract_sky(plot = plot, verbose = verbose)

        if plot:
            text = "Sky spectrum (scaled using a factor "+np.str(scale_sky_1D)+" )"
            plot_plot(self.wavelength, self.sky_emission, hlines=[0], ptitle=text,
                      xmin=self.wavelength[0]-10, xmax=self.wavelength[-1]+10, color="c",
                      vlines=[self.valid_wave_min,self.valid_wave_max])
        if verbose:
            print("  Intensities corrected for sky emission and stored in self.intensity_corrected !")                 
        self.sky_emission=sky_spectrum   # Restore sky_emission to original sky_spectrum
        #self.apply_mask(verbose=verbose)        
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------   
    def apply_1Dfit_sky(self, sky_spectrum = [], n_sky = 50, sky_fibres = [], sky_spectrum_file = "",
                    sky_wave_min = 0, sky_wave_max = 0, win_sky = 0, scale_sky_1D = 0,
                    sky_lines_file = "", brightest_line_wavelength = 0,
                    brightest_line = "Ha", maxima_sigma = 3, auto_scale_sky = False,
                    plot = True, verbose = True, fig_size = 12, fibre_p = -1, kernel_correct_ccd_defects = 51):            
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
        if sky_spectrum_file != "":
            if verbose:
                print("\n> Reading file with a 1D sky spectrum :")
                print(" ",sky_spectrum_file)
            
            w_sky,sky_spectrum = read_table(sky_spectrum_file, ["f", "f"] )
            
            if np.nanmedian(self.wavelength-w_sky) != 0:
                if verbose:
                    print("\n\n  WARNING !!!! The wavelengths provided on this file do not match the wavelengths on this RSS !!\n\n") 
            
            self.history.append('- 1D sky spectrum provided in file :')
            self.history.append('  '+sky_spectrum_file)        
        if verbose:
            print("\n> Fitting sky lines in both a provided sky spectrum AND all the fibres")
            print("  This process takes ~20 minutes for 385R if all skylines are considered!\n")
        if len(sky_spectrum) == 0: 
            if verbose:
                print("  No sky spectrum provided, using",n_sky,"lowest intensity fibres to create a sky...")
            self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                               sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, win_sky=win_sky)
        else:
            if scale_sky_1D != 0 :
                if verbose:
                    print("  1D Sky spectrum scaled by ",scale_sky_1D)
            else:
                if verbose:
                    print("  No scale between 1D sky spectrum and object given, calculating...")
                if n_sky == 50: n_sky =20
                self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                               sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max,  
                               win_sky=win_sky,include_history=False) 

                sky_r_self = self.sky_emission

                scale_sky_1D  = auto_scale_two_spectra(self, sky_r_self, sky_spectrum, scale=[0.1,1.01,0.025], 
                                                       w_scale_min = self.valid_wave_min,  w_scale_max = self.valid_wave_max, plot=plot, verbose = True )
                                               
            self.sky_emission = np.array(sky_spectrum) * scale_sky_1D
            
        self.fit_and_substract_sky_spectrum(self.sky_emission, sky_lines_file = sky_lines_file,
                               brightest_line_wavelength = brightest_line_wavelength, brightest_line = brightest_line,
                               maxima_sigma=maxima_sigma, ymin =-50, ymax=600, wmin = 0, wmax =0, auto_scale_sky = auto_scale_sky,                                       
                               warnings = False, verbose=False, plot=False, fig_size=fig_size, fibre=fibre_p )           
    
        if fibre_p == -1:
            if verbose:
                print("\n> 1Dfit sky_method usually generates some nans, correcting ccd defects again...")
            self.correct_ccd_defects(kernel_correct_ccd_defects=kernel_correct_ccd_defects, verbose=verbose, plot=plot, only_nans=True)    # Not replacing values <0
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------              
    def is_sky(self, n_sky = 50, win_sky = 0 , sky_fibres = [], sky_wave_min = 0,
               sky_wave_max = 0, plot = True, verbose = True):
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
        
        if verbose: print("\n> This RSS file is defined as SKY... identifying",n_sky," lowest fibres for getting 1D sky spectrum...")  
        self.history.append('- This RSS file is defined as SKY:')
        self.find_sky_emission(n_sky=n_sky, plot=plot, sky_fibres=sky_fibres,
                               sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, win_sky=0)
        #print "\n> This RSS file is defined as SKY... applying median filter with window",win_sky,"..."            
        if win_sky == 0: # Default when it is not a win_sky
            win_sky = 151
        print("\n  ... applying median filter with window",win_sky,"...\n")            

        medfilt_sky=median_filter(self.intensity_corrected, self.n_spectra, self.n_wave, win_sky=win_sky)
        self.intensity_corrected=copy.deepcopy(medfilt_sky)
        print("  Median filter applied, results stored in self.intensity_corrected !")
        self.history.append('  Median filter '+np.str(win_sky)+' applied to all fibres')
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

#ask Angel about *** what they are, and description, Once you have type and decription look for all the same ones

#search for *** when looking for something that isn't done
#final check search for TYPE and DESCRIPTION


class Interpolated_cube(object):            # TASK_Interpolated_cube

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
        
        g2d=False: If True uses a 2D gaussian, else doesn't. (default False)
            
        aligned_coor=False: If True uses a 2D gaussian, else doesn't. (default False)
            
        delta_RA =0: This is a small offset of the RA (right ascension). (default 0)
        
        delta_DEC=0: This is a small offset of the DEC (declination). (default 0)
        
        offsets_files="": ***
    
        offsets_files_position ="": ***
            

    Flux calibration:
        flux_calibration=[0]: This is the flux calibration. (default empty List)
            
        flux_calibration_file="": This is the directory of the flux calibration file. (default "")
        
        
    ADR:
        ADR=False: If True will correct for ADR (Atmospheric Differential Refraction). (default False)
    
        force_ADR = False: If True will correct for ADR even considoring a small correction. (default False)
    
        jump = -1: If a positive number partitions the wavelengths with step size jump, if -1 will not partition. (default -1)
    
        adr_index_fit = 2: This is the fitted polynomial with highest degree n. (default n = 2)
        
        ADR_x_fit=[0]: This is the ADR coefficients values for the x axis. (default constant 0)
        
        ADR_y_fit=[0]: This is the ADR coefficients values for the x axis. (default constant 0)               
        
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
    def __init__(self, RSS, pixel_size_arcsec=0.7, kernel_size_arcsec=1.4,  # RSS is an OBJECT
                 rss_file="",  path="",
                 centre_deg=[], size_arcsec=[], aligned_coor=False, 
                 delta_RA =0,  delta_DEC=0,
                 flux_calibration=[0], flux_calibration_file ="",
                 zeros=False, 
                 ADR=False, force_ADR = False, jump = -1, adr_index_fit = 2,
                 ADR_x_fit=[0],ADR_y_fit=[0], g2d=False,                         check_ADR = False,    # remove check_ADR?
                 
                 step_tracing = 100,
                 
                 offsets_files="", offsets_files_position ="", shape=[],
                 edgelow = -1, edgehigh = -1,
                 box_x=[0,-1],box_y=[0,-1], half_size_for_centroid = 10,
                 trim_cube = False, remove_spaxels_not_fully_covered = True,

                 warnings=False,
                 read_fits_cube = False, n_wave=2048, wavelength=[],description="",objeto="",PA=0,
                 valid_wave_min = 0, valid_wave_max = 0,
                 grating="",CRVAL1_CDELT1_CRPIX1=[0,0,0],total_exptime=0, n_cols=2,n_rows=2, 
                 number_of_combined_files = 1,
                 
                 plot_tracing_maps=[], plot_rss=True, plot=False, plot_spectra = True,
                 log=True, gamma=0., 
                 verbose=True, fig_size=12):
        
        if plot == False: 
            plot_tracing_maps=[]
            plot_rss = False
            plot_spectra = False
            
            
        self.pixel_size_arcsec = pixel_size_arcsec
        self.kernel_size_arcsec = kernel_size_arcsec
        self.kernel_size_pixels = kernel_size_arcsec/pixel_size_arcsec  # must be a float number!
        self.integrated_map = []
        
        self.history=[]
        fcal=False
        
        
        if rss_file != "" or type(RSS) == str:
            if  type(RSS) == str: rss_file=RSS
            rss_file =full_path(rss_file,path)  #RSS
            RSS=KOALA_RSS(rss_file, rss_clean=True, plot=plot, plot_final_rss = plot_rss,  verbose=verbose)
            

        if read_fits_cube:    # RSS is a cube given in fits file
            self.n_wave = n_wave       
            self.wavelength = wavelength                   
            self.description = description #+ " - CUBE"      
            self.object = objeto
            self.PA=PA
            self.grating = grating
            self.CRVAL1_CDELT1_CRPIX1 = CRVAL1_CDELT1_CRPIX1
            self.total_exptime=total_exptime
            self.number_of_combined_files = number_of_combined_files
            self.valid_wave_min = valid_wave_min        
            self.valid_wave_max = valid_wave_max
 
        else:    
            #self.RSS = RSS
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
        
            self.rss_list = RSS.filename  
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
    
            # if self.n_cols % 2 != 0: 
            #     self.n_cols += 1   # Even numbers to have [0,0] in the centre
            #     if len(size_arcsec) == 2 and aligned_coor == False and verbose: print("  We need an even number of spaxels, adding an extra column...") 
            # if self.n_rows % 2 != 0: 
            #     self.n_rows += 1
            #     if len(size_arcsec) == 2 and aligned_coor == False and verbose: print("  We need an even number of spaxels, adding an extra row...") 
            # # If we define a specific shape
            if len (shape) == 2:
                self.n_rows = shape[0]
                self.n_cols = shape[1]               
        else:           
            self.n_cols = n_cols
            self.n_rows = n_rows
    
        self.spaxel_RA0= self.n_cols/2  - 1  
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
                  
        self.RA_segment = self.n_cols *self.pixel_size_arcsec
        self.DEC_segment= self.n_rows*self.pixel_size_arcsec

        if zeros:
            self.data=np.zeros_like(self.weighted_I)
        else:  
            # Build the cube
            self.data=self.build_cube(jump=jump, RSS=RSS) 

            # Define box for tracing peaks if requested
            if half_size_for_centroid > 0 and np.nanmedian(box_x+box_y) == -0.5:   
                box_x,box_y = self.box_for_centroid(half_size_for_centroid=half_size_for_centroid, verbose=verbose, plot_map=plot, log=log)
                if verbose: print("  Using this box for tracing peaks and checking ADR ...")
                            
            # Trace peaks (check ADR only if requested)          
            if ADR_repeat:
                _check_ADR_ = False
            else:
                _check_ADR_ = True
                        
            if ADR:
                self.trace_peak(box_x=box_x, box_y=box_y, edgelow=edgelow, edgehigh =edgehigh, plot=plot, plot_tracing_maps=plot_tracing_maps,
                                verbose=verbose, adr_index_fit=adr_index_fit, g2d=g2d, check_ADR = _check_ADR_, step_tracing= step_tracing)
            elif verbose:
                print("\n> ADR will NOT be checked!")
                if np.nansum(self.ADR_y + self.ADR_x) != 0:
                    print("  However ADR fits provided and applied:")
                    print("  ADR_x_fit = ",self.ADR_x_fit)
                    print("  ADR_y_fit = ",self.ADR_y_fit)
                    
                    
            # Correct for Atmospheric Differential Refraction (ADR) if requested and not done before
            if ADR and ADR_repeat: 
                self.weighted_I = np.zeros((self.n_wave, self.n_rows, self.n_cols))
                self.weight = np.zeros_like(self.weighted_I)
                self.ADR_correction(RSS, plot=plot, force_ADR=force_ADR, jump=jump, remove_spaxels_not_fully_covered=remove_spaxels_not_fully_covered)                
                self.trace_peak(check_ADR=True, box_x=box_x, box_y=box_y, edgelow=edgelow, edgehigh =edgehigh, 
                                step_tracing =step_tracing,  adr_index_fit=adr_index_fit, g2d=g2d, 
                                plot_tracing_maps = plot_tracing_maps, plot=plot, verbose = verbose)
                  
            # Apply flux calibration
            self.apply_flux_calibration(flux_calibration=flux_calibration, flux_calibration_file=flux_calibration_file, verbose=verbose, path=path)
            
            if np.nanmedian(self.flux_calibration) != 0: fcal=True
            
            if fcal == False and verbose : print("\n> This interpolated cube does not include an absolute flux calibration")

            # Get integrated maps (all waves and valid range), plots
            self.get_integrated_map(plot=plot,plot_spectra=plot_spectra,fcal=fcal, #box_x=box_x, box_y=box_y, 
                                    verbose=verbose, plot_centroid=True, g2d=g2d, log=log, gamma=gamma, nansum = False)  # Barr
                
            # Trim the cube if requested            
            if trim_cube:
                self.trim_cube(half_size_for_centroid=half_size_for_centroid, box_x=box_x, box_y=box_y, ADR=ADR,
                               verbose=verbose, plot=plot, remove_spaxels_not_fully_covered=remove_spaxels_not_fully_covered, 
                               g2d=g2d, adr_index_fit=adr_index_fit, step_tracing=step_tracing, 
                               plot_tracing_maps=plot_tracing_maps)    #### UPDATE THIS, now it is run automatically
        
    
        if read_fits_cube == False and verbose: print("\n> Interpolated cube done!")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def apply_flux_calibration(self, flux_calibration=[], flux_calibration_file = "", path="", verbose=True):
        """
        Function for applying the flux calibration to a cube

        Parameters
        ----------
        flux_calibration : Float List, optional
            It is a list of floats. The default is empty
            
        flux_calibration_file : String, optional
            The file name of the flux_calibration. The default is ""
            
        path : String, optional
            The directory of the folder the flux_calibration_file is in. The default is "" 
        
        verbose : Boolean, optional
            Print results. The default is True.
                         
        Returns
        -------
        None.

        """      
        if flux_calibration_file  != "": 
            flux_calibration_file = full_path(flux_calibration_file,path) 
            if verbose: print("\n> Flux calibration provided in file:\n ",flux_calibration_file)
            w_star,flux_calibration = read_table(flux_calibration_file, ["f", "f"] ) 

        if len(flux_calibration) > 0: 
            if verbose: print("\n> Applying the absolute flux calibration...")
            self.flux_calibration=flux_calibration        
            # This should be in 1 line of step of loop, I couldn't get it # Yago HELP !!
            for y in range(self.n_rows):
                for x in range(self.n_cols):
                    self.data[:,y,x]=self.data[:,y,x] / self.flux_calibration  / 1E16 / self.total_exptime

            self.history.append("- Applied flux calibration")
            if flux_calibration_file  != "": self.history.append("  Using file "+flux_calibration_file)
            
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
            DESCRIPTION. The default is "new". ***
        remove_spaxels_not_fully_covered : Boolean, optional
            DESCRIPTION. The default is True. ***
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
        
        self.adrcor = True
        if total_ADR < cubo.pixel_size_arcsec * 0.1:   # Not needed if correction < 10 % pixel size
            if verbose:
                print("\n> Atmospheric Differential Refraction (ADR) correction is NOT needed.")
                print('  The computed max ADR value, {:.3f}",  is smaller than 10% the pixel size of {:.2f} arcsec'.format(total_ADR, cubo.pixel_size_arcsec))
            self.adrcor = False
            if force_ADR:
                self.adrcor = True
                if verbose: print('  However we proceed to do the ADR correction as indicated: "force_ADR = True" ...')
                            

        if self.adrcor:   
            if verbose:
                print("\n> Correcting for Atmospheric Differential Refraction (ADR) using: \n")   
                print("  ADR_x_fit = ",self.ADR_x_fit)                  
                print("  ADR_y_fit = ",self.ADR_y_fit)                  

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
                for l in range(0,self.n_wave,jump):
                    
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
                self.adrcor = True
                self.data=self.build_cube(jump=jump, RSS=RSS)
                           
                # # Check flux calibration and apply to the new cube                        
                # if np.nanmedian(self.flux_calibration) == 0:
                #     if verbose: print("\n\n> No absolute flux calibration included.")
                # else:
                #     if verbose: print("\n\n> Applying the absolute flux calibration...")
                #     self.apply_flux_calibration(self.flux_calibration, verbose=verbose)
                


            # Now remove spaxels with not full wavelength if requested
            if remove_spaxels_not_fully_covered == True:
            
                if verbose: print("\n> Removing spaxels that are not fully covered in wavelength in the valid range...")   # Barr
                _mask_ = cubo.integrated_map / cubo.integrated_map     
                for w in range(cubo.n_wave):
                    cubo.data[w] = cubo.data[w]*_mask_

        else:    
            if verbose: print (" NOTHING APPLIED !!!")
                    
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
                   adr_index_fit=2, step_tracing = 100, g2d = True, plot_tracing_maps = [],
                   plot=False, log=True, gamma=0., check_ADR=False, verbose = True):
        """
        ***        
        
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
            DESCRIPTION. The default is 100. ***
        g2d : Boolean, optional
            If True uses a 2D gaussian, else doesn't. The default is True. 
        plot_tracing_maps : List, optional
            DESCRIPTION. The default is []. ***
        plot : Boolean, optional
            If True generates and shows the plots. The default is False.
        log : Boolean, optional
            If true the map is plotted on a log scale. The default is True.
        gamma : Float, optional
            The value for power log. The default is 0..
        check_ADR : Boolean, optional
            DESCRIPTION. The default is False. ***
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

        x0=box_x[0]
        x1=box_x[1]
        y0=box_y[0]
        y1=box_y[1]  
        
        if check_ADR:
            plot_residua = False
        else:
            plot_residua = True
        
        ADR_x_fit, ADR_y_fit, ADR_x_max, ADR_y_max, ADR_total, x_peaks, y_peaks = centroid_of_cube(self, x0,x1,y0,y1, edgelow=edgelow, edgehigh=edgehigh,
                                                                                                   step_tracing=step_tracing, g2d=g2d, plot_tracing_maps=plot_tracing_maps,
                                                                                                   adr_index_fit=adr_index_fit,
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
    def box_for_centroid(self, half_size_for_centroid=6, verbose=True, plot=False, 
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
        if verbose: print("  box_x =[ {}, {} ],  box_y =[ {}, {} ]".format(box_x_centroid[0],box_x_centroid[1],box_y_centroid[0],box_y_centroid[1]))

        if plot_map: self.plot_map(plot_box=True, box_x=box_x_centroid, box_y=box_y_centroid, log=log, gamma=gamma, spaxel=[max_x,max_y], plot_centroid=True)


        return box_x_centroid,box_y_centroid
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def get_integrated_map(self, min_wave = 0, max_wave=0, nansum=True, 
                           vmin=1E-30, vmax=1E30, fcal=False,  log=True, gamma=0., cmap="fuego",
                           box_x=[0,-1], box_y=[0,-1], g2d=False, plot_centroid=True, 
                           trace_peaks=False, adr_index_fit=2, edgelow=-1, edgehigh=-1, step_tracing=100,
                           plot=False, plot_spectra=False, plot_tracing_maps=[], verbose=True) : ### CHECK
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
        vmin : FLoat, optional
            DESCRIPTION. The default is 1E-30. ***
        vmax : Float, optional
            DESCRIPTION. The default is 1E30. ***
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
        g2d : Boolean, optional
            If True uses a 2D gaussian, else doesn't. The default is False. 
        plot_centroid : Boolean, optional
            If True will plot the centroid. The default is True.
        trace_peaks : Boolean, optional
            DESCRIPTION. The default is False. ***
        adr_index_fit : Integer, optional
            This is the fitted polynomial with highest degree n. The default is 2. 
        edgelow : Integer, optional
            This is the lowest value in the wavelength range in terms of pixels. The default is -1. 
        edgehigh : Integer, optional
            This is the highest value in the wavelength range in terms of pixels, (maximum wavelength - edgehigh). The default is -1. 
        step_tracing : Integer, optional
            DESCRIPTION. The default is 100. ***
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
        
        if min_wave == 0: min_wave = self.valid_wave_min
        if max_wave == 0: max_wave = self.valid_wave_max
        
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
                   adr_index_fit=adr_index_fit, step_tracing = step_tracing, g2d = g2d, 
                   plot_tracing_maps = plot_tracing_maps, plot=False, check_ADR=False, verbose = False)

        if verbose: 
            print("\n> Created integrated map between {:5.2f} and {:5.2f} considering nansum = {:}".format(min_wave, max_wave,nansum))
            print("  The cube has a size of {} x {} spaxels = [ 0 ... {} ] x [ 0 ... {} ]".format(self.n_cols, self.n_rows, self.n_cols-1, self.n_rows-1))
            print("  The peak of the emission in integrated image is in spaxel [",self.max_x,",",self.max_y ,"]")
            print("  The peak of the emission tracing all wavelengths is in position [",np.round(self.x_peak_median,2),",",np.round(self.y_peak_median,2),"]")

        if plot:
            self.plot_map(log=log, gamma=gamma, spaxel=[self.max_x,self.max_y], spaxel2=[self.x_peak_median,self.y_peak_median], fcal=fcal, 
                          box_x=box_x, box_y=box_y, plot_centroid=plot_centroid, g2d=g2d, cmap=cmap, vmin=vmin, vmax=vmax)
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


        if plot:    # TODO: This should be done with PLOT PLOT
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
            When creating a box to show/trim/alignment these are the x coordinates in spaxels of the box. The default is [].
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y coordinates in spaxels of the box. The default is [].
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
            self.plot_map(spaxel=center,  box_x= box_x,  box_y= box_y, gamma=gamma, log=log, description=description, verbose=verbose)

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
            DESCRIPTION. The default is 5. ***
        offset_y : Integer, optional
            DESCRIPTION. The default is 5. ***
        distance : Integer, optional
            DESCRIPTION. The default is 5. ***
        width : Integer, optional
            DESCRIPTION. The default is 3. ***
        pa : String, optional
            This is the position angle. The default is "".
        peak : List, optional
            DESCRIPTION. The default is []. ***
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
                    fraction=0.0457, pad=0.02, colorbar_ticksize= 14, colorbar_fontsize = 15, barlabel=""):
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
            DESCRIPTION. The default is 1E-30. ***
        vmax : Float, optional
            DESCRIPTION. The default is 1E30. ***
        cmap : String, optional
            This is the colour of the map. The default is "gist_gray".
        fig : Integer, optional
            DESCRIPTION. The default is 10. ***
        save_file : String, optional
            Save plot in file "file.extension". The default is "".
        #ADR : Boolean, optional
            If True will correct for ADR (Atmospheric Differential Refraction). The default is True.
        description : String, optional
            This is the description of the cube. The default is "".
        contours : Boolean, optional
            DESCRIPTION. The default is True. ***
        clabel : Boolean, optional
            DESCRIPTION. The default is False. ***
        verbose : Boolean, optional
            Print results. The default is True.
        spaxel : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel2 : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel3 : Integer, optional
            DESCRIPTION. The default is 0. ***
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
            DESCRIPTION. The default is 12. ***
        title_fontsize : Integer, optional
            This is the size of the font for the title. The default is 16.
        fraction : Float, optional
            DESCRIPTION. The default is 0.0457. ***
        pad : Float, optional
            DESCRIPTION. The default is 0.02. ***
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
            DESCRIPTION. The default is 0..
        gaussian_fit : Boolean, optional
            DESCRIPTION. The default is False. ***
        gf : Boolean, optional
            DESCRIPTION. The default is False. ***
        lowlow : Integer, optional
            DESCRIPTION. The default is 50. ***
        lowhigh : Integer, optional
            DESCRIPTION. The default is 10. ***
        highlow : Integer, optional
            DESCRIPTION. The default is 10. ***
        highhigh : Integer, optional
            DESCRIPTION. The default is 50. ***
        show_spaxels : List, optional
            DESCRIPTION. The default is []. ***
        verbose : Boolean, optional
			Print results. The default is True.
        description : String, optional
            This is the description of the cube. The default is "".

        Returns
        -------
        mapa : Map
            DESCRIPTION. ***

        """

        mapa = create_map(cube=self, line=line, w2 = w2, gaussian_fit = gaussian_fit, gf=gf,
                   lowlow= lowlow, lowhigh=lowhigh, highlow=highlow, highhigh = highhigh,
                   show_spaxels=show_spaxels, verbose = verbose, description = description )
        return mapa
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def plot_wavelength(self, wavelength, w2=0.,  
                        #norm=colors.PowerNorm(gamma=1./4.), 
                        log = False, gamma=0., vmin=1E-30, vmax=1E30,
                        cmap=fuego_color_map, fig_size= 10, fcal=False,
                        save_file="", description="", contours=True, clabel=False, verbose = True,
                        spaxel=0, spaxel2=0, spaxel3=0,
                        box_x=[0,-1], box_y=[0,-1], 
                        circle=[0,0,0],circle2=[0,0,0],circle3=[0,0,0],
                        plot_centre=True, plot_spaxel=False, plot_spaxel_grid=True, 
                        label_axes_fontsize=15, axes_fontsize = 14, c_fontsize = 12, title_fontsize= 16,
                        fraction=0.0457, pad=0.02, colorbar_ticksize= 14, colorbar_fontsize = 15, barlabel="") :
        """
        Plot map at a particular wavelength or in a wavelength range

        Parameters
        ----------
        wavelength: Float
          wavelength to be mapped.
        w2 : Float, optional
            DESCRIPTION. The default is 0.. ***
        #norm : TYPE, optional ***
            DESCRIPTION. The default is colors.PowerNorm(gamma=1./4.). ***
        log : Boolean, optional
            If true the map is plotted on a log scale. The default is False.
        gamma : Float, optional
            DESCRIPTION. The default is 0.. ***
        vmin : Float, optional
            DESCRIPTION. The default is 1E-30. ***
        vmax : Float, optional
            DESCRIPTION. The default is 1E30. ***
        cmap: 
            Color map used.
            Velocities: cmap="seismic". The default is fuego_color_map.
        fig_size : Integer, optional
            This is the size of the figure. The default is 10.
        fcal : Boolean, optional
            If fcal=True, cube.flux_calibration is used. The default is False. ***
        save_file : String, optional
            Save plot in file "file.extension". The default is "".
        description : String, optional
            This is the description of the cube. The default is "".
        contours : Boolean, optional
            DESCRIPTION. The default is True. ***
        clabel : Boolean, optional
            DESCRIPTION. The default is False. ***
        verbose : Boolean, optional
			Print results. The default is True.
        spaxel : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel2 : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel3 : Integer, optional
            DESCRIPTION. The default is 0. ***
        box_x : Integer List, optional
            When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [0,-1]. ***
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [0,-1]. ***
        circle : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. ***
        circle2 : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. ***
        circle3 : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. ***
        plot_centre : Boolean, optional
            DESCRIPTION. The default is True. ***
        plot_spaxel : Boolean, optional
            If True will plot the spaxel. The default is False. ***
        plot_spaxel_grid : Boolean, optional
            If True plots the spaxel grid. The default is True. ***
        label_axes_fontsize : Integer, optional
            This is the size of the axes labels. The default is 15.
        axes_fontsize : Integer, optional
            This is the size of the font on the axes. The default is 14.
        c_fontsize : Integer, optional
            DESCRIPTION. The default is 12. ***
        title_fontsize : Integer, optional
            This is the size of the titles text. The default is 16.
        fraction : Float, optional
            DESCRIPTION. The default is 0.0457. ***
        pad : Float, optional
            DESCRIPTION. The default is 0.02. ***
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
                      circle = circle, circle2=circle2, circle3 = circle3,
                      plot_centre = plot_centre, plot_spaxel=plot_spaxel, plot_spaxel_grid=plot_spaxel_grid,
                      log=log, gamma=gamma, vmin=vmin, vmax=vmax,
                      label_axes_fontsize=label_axes_fontsize, axes_fontsize = axes_fontsize, c_fontsize = c_fontsize, title_fontsize= title_fontsize,
                      fraction=fraction, pad=pad, colorbar_ticksize= colorbar_ticksize, colorbar_fontsize = colorbar_fontsize, barlabel = barlabel)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def plot_map(self, mapa="", log = False, gamma = 0., vmin=1E-30, vmax=1E30, fcal=False,
                 #norm=colors.Normalize(), 
                 trimmed= False,
                 cmap="fuego", weight = False, velocity= False, fwhm=False, ew=False, ratio=False,
                 contours=True, clabel=False,
                 line =0,  
                 spaxel=0, spaxel2=0, spaxel3=0,
                 box_x=[0,-1], box_y=[0,-1], plot_centroid=False, g2d=True, half_size_for_centroid  = 0,
                 circle=[0,0,0],circle2=[0,0,0],circle3=[0,0,0],
                 plot_box=False, plot_centre=True, plot_spaxel=False, plot_spaxel_grid=True, alpha_grid=0.1, 
                 plot_spaxel_list=[], color_spaxel_list="blue", alpha_spaxel_list=0.4,
                 label_axes_fontsize=15, axes_fontsize = 14, c_fontsize = 12, title_fontsize= 16,
                 fraction=0.0457, pad=0.02, colorbar_ticksize= 14, colorbar_fontsize = 15, barlabel="",
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
        mapa : String, optional
            DESCRIPTION. The default is "". ***
        log : Boolean, optional
            If true the map is plotted on a log scale. The default is False.
        gamma : Float, optional
            DESCRIPTION. The default is 0.. ***
        vmin : Float, optional
            DESCRIPTION. The default is 1E-30. ***
        vmax : Float, optional
            DESCRIPTION. The default is 1E30. ***
        fcal : Boolean, optional
            If fcal=True, cube.flux_calibration is used. The default is False. ***
        #norm : TYPE, optional ***
            DESCRIPTION. The default is colors.Normalize(). ***
        trimmed : Boolean, optional
            DESCRIPTION. The default is False. ***
        cmap : String, optional
            This is the colour of the map. The default is "fuego". 
        weight : Boolean, optional
            DESCRIPTION. The default is False. ***
        velocity : Boolean, optional
            DESCRIPTION. The default is False. ***
        fwhm : Boolean, optional
            DESCRIPTION. The default is False. ***
        ew : Boolean, optional
            DESCRIPTION. The default is False. ***
        ratio : Boolean, optional
            DESCRIPTION. The default is False. ***
        contours : Boolean, optional
            DESCRIPTION. The default is True. ***
        clabel : Boolean, optional
            DESCRIPTION. The default is False. ***
        line : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel2 : Integer, optional
            DESCRIPTION. The default is 0. ***
        spaxel3 : Integer, optional
            DESCRIPTION. The default is 0. ***
        box_x : Integer List, optional
             When creating a box to show/trim/alignment these are the x cooridnates in spaxels of the box. The default is [0,-1].
        box_y : Integer List, optional
            When creating a box to show/trim/alignment these are the y cooridnates in spaxels of the box. The default is [0,-1].
        plot_centroid : Boolean, optional
            If True will plot the centroid. The default is False.
        g2d : Boolean, optional
            If True uses a 2D gaussian, else doesn't. The default is True. 
        half_size_for_centroid : Integer, optional
            This is half the length/width of the box. The default is 0.
        circle : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. ***
        circle2 : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. ***
        circle3 : Integer List, optional
            DESCRIPTION. The default is [0,0,0]. ***
        plot_box : Boolean, optional
            DESCRIPTION. The default is False. ***
        plot_centre : Boolean, optional
            DESCRIPTION. The default is True. ***
        plot_spaxel : Boolean, optional
            If True will plot the spaxel. The default is False. ***
        plot_spaxel_grid : Boolean, optional
            If True plots the spaxel grid. The default is True.
        alpha_grid : Float, optional
            DESCRIPTION. The default is 0.1. ***
        plot_spaxel_list : List, optional
            DESCRIPTION. The default is []. ***
        color_spaxel_list : String, optional
            DESCRIPTION. The default is "blue". ***
        alpha_spaxel_list : Float, optional
            DESCRIPTION. The default is 0.4. ***
        label_axes_fontsize : Integer, optional
            This is the size of the axes labels. The default is 15.
        axes_fontsize : Integer, optional
            This is the size of the font on the axes. The default is 14.
        c_fontsize : Integer, optional
            DESCRIPTION. The default is 12. ***
        title_fontsize : Integer, optional
            This is the size of the font for the title. The default is 16.
        fraction : Float, optional
            DESCRIPTION. The default is 0.0457. ***
        pad : Float, optional
            DESCRIPTION. The default is 0.02. ***
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

                
        mapa_=mapa
        try:
            if type(mapa[0]) == str:       # Maps created by PyKOALA have [description, map, l1, l2 ...] 
                mapa_ = mapa[1]
                if description == "": mapa = mapa[0]
        except Exception:
            if mapa == "" :
                if len(self.integrated_map) == 0:  self.get_integrated_map(verbose=verbose)
                       
                mapa_=self.integrated_map 
                if description == "": description = self.description+" - Integrated Map"                 

        if description == "" :
            description = self.description
            
            
        # Trim the map if requested   
        if np.nanmedian(box_x+box_y) != -0.5 and plot_box == False:
            trimmed= True        
            mapa= copy.deepcopy(mapa_[box_y[0]:box_y[1],box_x[0]:box_x[1]])
        else:
            mapa = mapa_
            

        if trimmed:
            extent1 = 0 
            #extent2 = (box_x[1]-box_x[0])*self.pixel_size_arcsec 
            extent2 = len(mapa[0])*self.pixel_size_arcsec 
            extent3 = 0                       
            #extent4 = (box_y[1]-box_y[0])*self.pixel_size_arcsec 
            extent4 = len(mapa)*self.pixel_size_arcsec 
            alpha_grid = 0
            plot_spaxel_grid = False
            plot_centre = False
            fig_size= fig_size*0.5
        else:
            extent1 = (0.5-self.n_cols/2) * self.pixel_size_arcsec
            extent2 = (0.5+self.n_cols/2) * self.pixel_size_arcsec
            extent3 = (0.5-self.n_rows/2) * self.pixel_size_arcsec
            extent4 = (0.5+self.n_rows/2) * self.pixel_size_arcsec
            
            
        if verbose: print("\n> Plotting map '"+description.replace("\n ","")+"' :")
        if verbose and trimmed: print("  Trimmed in x = [ {:} , {:} ]  ,  y = [ {:} , {:} ] ".format(box_x[0],box_x[1],box_y[0],box_y[1]))
        
        
        # Check fcal
        if fcal == False and np.nanmedian(self.flux_calibration) != 0: fcal = True
        
        
        if velocity and cmap=="fuego": cmap="seismic" 
        if fwhm and cmap=="fuego": cmap="Spectral" 
        if ew and cmap=="fuego": cmap="CMRmap_r"
        if ratio and cmap=="fuego": cmap="gnuplot2" 
        
        if velocity or fwhm or ew or ratio or weight : 
            fcal=False
            if vmin == 1E-30 : vmin=np.nanpercentile(mapa,5)
            if vmax == 1E30 : vmax=np.nanpercentile(mapa,95)
            
        # We want squared pixels for plotting
        try:
            aspect_ratio = self.combined_cube.n_cols/self.combined_cube.n_rows * 1.
        except Exception:
            aspect_ratio = self.n_cols/self.n_rows *1.
            
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
                           
        cax=ax.imshow(mapa, origin='lower', interpolation='none', norm = norm, cmap=cmap,
                      extent=(extent1, extent2, extent3, extent4))
        cax.set_clim(vmin=vmin) 
        cax.set_clim(vmax=vmax)

        if contours:
            CS=plt.contour(mapa, extent=(extent1, extent2, extent3, extent4))
            if clabel: plt.clabel(CS, inline=1, fontsize=c_fontsize)
        
        ax.set_title(description, fontsize=title_fontsize)  
        plt.tick_params(labelsize=axes_fontsize)
        plt.xlabel('$\Delta$ RA [arcsec]', fontsize=label_axes_fontsize)
        plt.ylabel('$\Delta$ DEC [arcsec]', fontsize=label_axes_fontsize)
        plt.legend(loc='upper right', frameon=False)
        plt.minorticks_on()
        plt.grid(which='both', color="white", alpha=alpha_grid)
        #plt.axis('square')
        
        
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
            #if trimmed:    
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
            #print(xc,yc)
            #print(offset_from_center_x_arcsec,offset_from_center_y_arcsec)
            
            #if np.nanmedian(box_x+box_y) != -0.5 and plot_box == False:
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

                  

        if np.nanmedian(box_x+box_y) != -0.5 and plot_box:   # Plot box
            box_x=[box_x[0]-ox,box_x[1]-ox]
            box_y=[box_y[0]-oy,box_y[1]-oy]
            
            vertices_x =[box_x[0]*self.pixel_size_arcsec,box_x[0]*self.pixel_size_arcsec,box_x[1]*self.pixel_size_arcsec,box_x[1]*self.pixel_size_arcsec,box_x[0]*self.pixel_size_arcsec]
            vertices_y =[box_y[0]*self.pixel_size_arcsec,box_y[1]*self.pixel_size_arcsec,box_y[1]*self.pixel_size_arcsec,box_y[0]*self.pixel_size_arcsec,box_y[0]*self.pixel_size_arcsec]          
            plt.plot(vertices_x,vertices_y, "-b", linewidth=2., alpha=0.6)


        cbar = fig.colorbar(cax, fraction=fraction, pad=pad)  
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
        ***

        Parameters
        ----------
        min_wave : Integer, optional
            The minimum wavelength passed through the mask. The default is 0.
        max_wave : Integer, optional
            The maximum wavelength passed through the mask. The default is 0.
        include_partial_spectra : Boolean, optional
            DESCRIPTION. The default is False. ***
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
                             plot=False, verbose=False): #LUKE
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
                          verbose=True, plot_centre=False,
                          norm=colors.LogNorm())

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
            DESCRIPTION. The default is 5.. ***
        sky_annulus_high_arcsec : Float, optional
            DESCRIPTION. The default is 10.. ***
        fig_size : Integer, optional
            This is the size of the figure. The default is 12.
        plot : Boolean, optional
            If True generates and shows the plots. The default is True.
        verbose : Boolean, optional
            Print results. The default is True.

        Returns
        -------
        Numpy Array
            DESCRIPTION. ***
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
            DESCRIPTION. The default is 0. ***
        max_wave_flux : Integer, optional
            DESCRIPTION. The default is 0. ***
        fit_degree_flux : Integer, optional
            DESCRIPTION. The default is 3. ***
        step_flux : Float, optional
            DESCRIPTION. The default is 25.. ***
        r_max : Integer, optional
            r_max to integrate, in units of r2_half_light (= seeing if object is a star, for flux calibration make r_max=5). The default is 5.
        exp_time : Integer, optional
            DESCRIPTION. The default is 0. ***
        ha_width : Integer, optional
            DESCRIPTION. The default is 0. ***
        after_telluric_correction : Boolean, optional
            DESCRIPTION. The default is False. ***
        sky_annulus_low_arcsec : Float, optional
            DESCRIPTION. The default is 5.. ***
        sky_annulus_high_arcsec : Float, optional
            DESCRIPTION. The default is 10.. ***
        odd_number : Integer, optional
            DESCRIPTION. The default is 0. ***
        fit_weight : Float, optional
            DESCRIPTION. The default is 0.. ***
        smooth_weight : Float, optional
            DESCRIPTION. The default is 0.. ***
        smooth : Float, optional
            smooths the data. The default is 0..
        exclude_wlm : List of Lists of Integers, optional
            DESCRIPTION. The default is [[0,0]]. ***
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
        response_curve_medfilt_ =sig.medfilt(response_curve,np.int(odd_number))
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
        ***

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
        flux : TYPE ***
            DESCRIPTION.
        TYPE ***
            DESCRIPTION.
        beta : TYPE ***
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
                  adr_index_fit = 2, g2d=False, step_tracing=100, plot_tracing_maps =[],
                  plot_weight = False, fcal=False, plot=True, plot_spectra=False, verbose=True, warnings=True):
        """ 
        Task for trimming cubes in RA and DEC (not in wavelength)
        
        if nansum = True, it keeps spaxels in edges that only have a partial spectrum (default = False)
        if remove_spaxels_not_fully_covered = False, it keeps spaxels in edges that only have a partial spectrum (default = True)
    

        Parameters
        ----------
        trim_cube : Boolean, optional
            DESCRIPTION. The default is True. ***
        trim_values : List, optional
            DESCRIPTION. The default is []. ***
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
            If True uses a 2D gaussian, else doesn't. The default is False. 
        step_tracing : Integer, optional
            DESCRIPTION. The default is 100. ***
        plot_tracing_maps : List, optional
            If True will plot the tracing maps. The default is [].
        plot_weight : Boolean, optional
            DESCRIPTION. The default is False. ***
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
           
                cube.trace_peak(check_ADR=True, box_x=box_x_, box_y=box_y_, edgelow=edgelow, edgehigh =edgehigh, 
                                adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing,
                                plot=plot, plot_tracing_maps=plot_tracing_maps, verbose=verbose)
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
            DESCRIPTION. The default is True. ***
        valid_wave_min : Integer, optional
            DESCRIPTION. The default is 0. ***
        valid_wave_max : Integer, optional
            DESCRIPTION. The default is 0. ***

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
    return (cumsum[N:] - cumsum[:-N]) / N
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def cumulaive_Moffat(r2, L_star, alpha2, beta):
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
    """
    index_cut = np.searchsorted(r2_growth_curve, r2_half_light*r_max**2)
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
                                          fit[0], fit[1], fit[2])/fit[0], ':')
    return fit
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def KOALA_offsets(s, pa):
    print("\n> Offsets towards North and East between pointings," \
        "according to KOALA manual, for pa =", pa, "degrees")
    pa *= np.pi/180
    print("  a -> b :", s*np.sin(pa), -s*np.cos(pa))
    print("  a -> c :", -s*np.sin(60-pa), -s*np.cos(60-pa))
    print("  b -> d :", -np.sqrt(3)*s*np.cos(pa), -np.sqrt(3)*s*np.sin(pa))
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def offset_between_cubes(cube1, cube2, plot=True):
    x = (cube2.x_peak - cube2.n_cols/2. + cube2.RA_centre_deg*3600./cube2.pixel_size_arcsec) \
        - (cube1.x_peak - cube1.n_cols/2. + cube1.RA_centre_deg*3600./cube1.pixel_size_arcsec)
    y = (cube2.y_peak - cube2.n_rows/2. + cube2.DEC_centre_deg*3600./cube2.pixel_size_arcsec) \
        - (cube1.y_peak - cube1.n_rows/2. + cube1.DEC_centre_deg*3600./cube1.pixel_size_arcsec)
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
        map1=cube1.integrated_map
        map2=cube2.integrated_map
    else:     
        l = np.searchsorted(cube1.RSS.wavelength, line)
        map1 = cube1.data[l]
        map2 = cube2.data[l]
        
    scale = np.nanmedian(map1+map2)*3
    scatter = np.nanmedian(np.nonzero(map1-map2))

    plt.figure(figsize=(12, 8))
    plt.imshow(map1-map2, vmin=-scale, vmax=scale, cmap=plt.cm.get_cmap('RdBu'))   # vmin = -scale
    plt.colorbar()
    plt.contour(map1, colors='w', linewidths=2, norm=colors.LogNorm())
    plt.contour(map2, colors='k', linewidths=1, norm=colors.LogNorm())
    if line != 0:
        plt.title("{:.2f} AA".format(line))
    else:
        plt.title("Integrated Map")
    plt.show()
    plt.close()
    print("  Medium scatter : ",scatter)
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
        ci_name=calibration_star_cubes[i].object
        for j in range(i+1,n_cubes):
            cj = calibration_star_cubes[j].response_curve * scale[j]
            cj_name=calibration_star_cubes[j].object
            ptitle = "Comparison of flux calibration for "+ci_name+" and "+cj_name
            ylabel = ci_name+" / "+cj_name
            plot_plot(wavelength,ci/cj, hlines=[0.85, 0.9,0.95,1,1,1,1,1.05,1.1,1.15], ymin=0.8, ymax=1.2, 
                      ylabel=ylabel, ptitle=ptitle)
    print("\n> Plotting response curve (absolute flux calibration) of standard stars...\n")
    
    
    plt.figure(figsize=(11, 8))
    mean_curve = np.zeros_like(wavelength)
    mean_values=[]
    list_of_scaled_curves=[]
    i = 0
    for star in calibration_star_cubes:    
        list_of_scaled_curves.append(star.response_curve * scale[i])
        mean_curve = mean_curve + star.response_curve * scale[i]
        plt.plot(star.wavelength, star.response_curve * scale[i],
                 label=star.description, alpha=0.2, linewidth=2)
        if use_median:
            print("  Median value for ",star.object," = ",np.nanmedian(star.response_curve * scale[i]),"      scale = ",scale[i])
        else:
            print("  Mean value for ",star.object," = ",np.nanmean(star.response_curve * scale[i]),"      scale = ",scale[i])
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

    if len(calibration_star_cubes) > 1 : print("  Variation in flux calibrations =  {:.2f} %".format(dispersion*100.))

    #dispersion=np.nanmax(mean_values)-np.nanmin(mean_values)
    #print "  Variation in flux calibrations =  {:.2f} %".format(dispersion/np.nanmedian(mean_values)*100.)

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
    
    print("\n> Flux calibration for all wavelengths = ",flux_calibration)
    print("\n  Flux calibration obtained!")
    return flux_calibration
# -----------------------------------------------------------------------------   
# -----------------------------------------------------------------------------
def obtain_telluric_correction(w, telluric_correction_list, plot=True, label_stars=[], scale=[]):
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

    print("\n> Telluric correction = ",telluric_correction)
    if np.nanmean(scale) != 1. : print("  Telluric correction scale provided : ",scale)
    print("\n  Telluric correction obtained!")
    return telluric_correction
# -----------------------------------------------------------------------------   
# -----------------------------------------------------------------------------
def coord_range(rss_list):
    RA = [rss.RA_centre_deg+rss.offset_RA_arcsec/3600. for rss in rss_list]
    RA_min = np.nanmin(RA)
    RA_max = np.nanmax(RA)
    DEC = [rss.DEC_centre_deg+rss.offset_DEC_arcsec/3600. for rss in rss_list]
    DEC_min = np.nanmin(DEC)
    DEC_max = np.nanmax(DEC)
    return RA_min, RA_max, DEC_min, DEC_max
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def align_n_cubes(rss_list, cube_list=[0], flux_calibration_list=[[]], pixel_size_arcsec=0.3, kernel_size_arcsec=1.5, offsets=[1000], 
                  plot= False, plot_weight=False, plot_tracing_maps=[], plot_spectra=True,
                  ADR=False, jump=-1, ADR_x_fit_list=[0], ADR_y_fit_list=[0], force_ADR = False,
                  half_size_for_centroid =10, box_x=[0,-1], box_y=[0,-1], adr_index_fit = 2, g2d=False, step_tracing = 100,
                  edgelow=-1, edgehigh=-1, size_arcsec=[], centre_deg=[], warnings=False, verbose= True):  
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
            if np.nanmedian(ADR_x_fit_list) == 0 and ADR: print ("  But ADR data provided and ADR correction requested, rebuiding the cube...")
            
            
            
    xx=[0]      # This will have 0, x12, x23, x34, ... xn1
    yy=[0]      # This will have 0, y12, y23, y34, ... yn1
    
    if len(flux_calibration_list[0]) == 0:
        for i in range(1,n_rss): flux_calibration_list.append([])

    if len(offsets) == 0:  
        if verbose and n_rss > 1: print("\n  Using peak of the emission tracing all wavelengths to align cubes:") 
        n_cubes = len(cube_list)
        if n_cubes != n_rss:
            if verbose:
                print("\n\n\n ERROR: number of cubes and number of rss files don't match!")
                print("\n\n THIS IS GOING TO FAIL ! \n\n\n")

        for i in range(n_rss-1):
            xx.append(cube_list[i+1].offset_from_center_x_arcsec_tracing - cube_list[i].offset_from_center_x_arcsec_tracing) 
            yy.append(cube_list[i+1].offset_from_center_y_arcsec_tracing - cube_list[i].offset_from_center_y_arcsec_tracing)  
        xx.append(cube_list[0].offset_from_center_x_arcsec_tracing - cube_list[-1].offset_from_center_x_arcsec_tracing)
        yy.append(cube_list[0].offset_from_center_y_arcsec_tracing - cube_list[-1].offset_from_center_y_arcsec_tracing)
    
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
        list_RA_centre_deg.append(rss_list[i].RA_centre_deg)
        list_DEC_centre_deg.append(rss_list[i].DEC_centre_deg)        
    
    median_RA_centre_deg = np.nanmedian (list_RA_centre_deg)
    median_DEC_centre_deg = np.nanmedian (list_DEC_centre_deg)
    
    distance_from_median  = []
    
    for i in range(n_rss):
        rss_list[i].ALIGNED_RA_centre_deg = median_RA_centre_deg + np.nansum(xx[1:i+1])/3600.    # CHANGE SIGN 26 Apr 2019    # ERA cube_list[0]
        rss_list[i].ALIGNED_DEC_centre_deg = median_DEC_centre_deg  - np.nansum(yy[1:i+1])/3600.        # rss_list[0].DEC_centre_deg
    
        distance_from_median.append(np.sqrt( 
                (rss_list[i].RA_centre_deg - median_RA_centre_deg)**2 +
                (rss_list[i].DEC_centre_deg - median_DEC_centre_deg)**2) )
    
    reference_rss = distance_from_median.index(np.nanmin(distance_from_median))
    
    if len(centre_deg) == 0:    
        if verbose and n_rss > 1: print("  No central coordenates given, using RSS {} for getting the central coordenates:".format(reference_rss+1))   
        RA_centre_deg = rss_list[reference_rss].ALIGNED_RA_centre_deg
        DEC_centre_deg = rss_list[reference_rss].ALIGNED_DEC_centre_deg  
    else:
        if verbose and n_rss > 1: print("  Central coordenates provided: ")   
        RA_centre_deg = centre_deg[0]
        DEC_centre_deg = centre_deg[1]  
        

    if verbose and n_rss > 1:
        print("\n> Median central coordenates of RSS files: RA =",RA_centre_deg," DEC =", DEC_centre_deg)
              
        print("\n  Offsets (in arcsec):        x             y                          ( EAST- / WEST+   NORTH- / SOUTH+) ")
        for i in range(1,len(xx)-1):
            print("         {:2.0f} -> {:2.0f}           {:+5.3f}         {:+5.3f}".format(i,i+1,xx[i],yy[i]))      
        print("         {:2.0f} ->  1           {:+5.3f}         {:+5.3f}".format(len(xx)-1,xx[-1],yy[-1]))      
        print("           TOTAL:            {:5.3f}          {:5.3f}".format(np.nansum(xx),np.nansum(yy))) 
    
          
        print("\n         New_RA_centre_deg       New_DEC_centre_deg      Diff with respect Cube 1 [arcsec]")  
       
        for i in range (0,n_rss):
            print("  Cube {:2.0f}:     {:5.8f}          {:5.8f}           {:+5.3f}   ,  {:+5.3f}   ".format(i+1,rss_list[i].ALIGNED_RA_centre_deg, rss_list[i].ALIGNED_DEC_centre_deg, (rss_list[i].ALIGNED_RA_centre_deg-rss_list[0].ALIGNED_RA_centre_deg)*3600.,(rss_list[i].ALIGNED_DEC_centre_deg-rss_list[0].ALIGNED_DEC_centre_deg)*3600.))  
    
    offsets_files=[]
    for i in range(1,n_rss):           # For keeping in the files with self.offsets_files
        vector=[xx[i],yy[i]]        
        offsets_files.append(vector)

    xx_dif = np.nansum(xx[0:-1])   
    yy_dif = np.nansum(yy[0:-1]) 

    if verbose and n_rss > 1: print('\n  Accumulative difference of offsets: {:.2f}" x {:.2f}" '.format(xx_dif, yy_dif))
       
    if len(size_arcsec) == 0:
        RA_size_arcsec = rss_list[0].RA_segment + np.abs(xx_dif) + 3*kernel_size_arcsec
        DEC_size_arcsec =rss_list[0].DEC_segment + np.abs(yy_dif) + 3*kernel_size_arcsec 
        size_arcsec=[RA_size_arcsec,DEC_size_arcsec]

    if verbose and n_rss > 1: print('\n  RA_size x DEC_size  = {:.2f}" x {:.2f}" '.format(size_arcsec[0], size_arcsec[1]))

    cube_aligned_list=[]
    
    for i in range(1,n_rss+1):
        #escribe="cube"+np.str(i)+"_aligned"
        cube_aligned_list.append("cube"+np.str(i)+"_aligned")

    if np.nanmedian(ADR_x_fit_list) == 0 and ADR:   # Check if ADR info is provided and ADR is requested
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

            if verbose: print("\n> Creating aligned cube",i+1,"of a total of",n_rss,"...")

            cube_aligned_list[i]=Interpolated_cube(rss_list[i], pixel_size_arcsec=pixel_size_arcsec, kernel_size_arcsec=kernel_size_arcsec, 
                                                   centre_deg=[RA_centre_deg, DEC_centre_deg], size_arcsec=size_arcsec, 
                                                   aligned_coor=True, flux_calibration=flux_calibration_list[i],  offsets_files = offsets_files, offsets_files_position =i+1, 
                                                   ADR=ADR, jump=jump, ADR_x_fit = ADR_x_fit_list[i], ADR_y_fit = ADR_y_fit_list[i], check_ADR=True,
                                                   half_size_for_centroid=half_size_for_centroid, box_x=box_x, box_y=box_y,
                                                   adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing, plot_tracing_maps=plot_tracing_maps,
                                                   plot=plot, plot_spectra=plot_spectra, edgelow=edgelow, edgehigh=edgehigh, 
                                                   
                                                   warnings=warnings, verbose=verbose)
            if plot_weight: cube_aligned_list[i].plot_weight()
        else:
            cube_aligned_list[i] = cube_list[i]
            if verbose: print("\n> Only one file provided and no ADR correction given, the aligned cube is the same than the original cube...")


    if verbose and n_rss > 1:
        print("\n> Checking offsets of ALIGNED cubes (in arcsec, everything should be close to 0):")
        print("  Offsets (in arcsec):        x             y                          ( EAST- / WEST+   NORTH- / SOUTH+) ")

    xxx=[]
    yyy=[]
    
    for i in range(1,n_rss):
        xxx.append(cube_aligned_list[i-1].offset_from_center_x_arcsec_tracing - cube_aligned_list[i].offset_from_center_x_arcsec_tracing)
        yyy.append(cube_aligned_list[i-1].offset_from_center_y_arcsec_tracing - cube_aligned_list[i].offset_from_center_y_arcsec_tracing)
    xxx.append(cube_aligned_list[-1].offset_from_center_x_arcsec_tracing - cube_aligned_list[0].offset_from_center_x_arcsec_tracing)
    yyy.append(cube_aligned_list[-1].offset_from_center_y_arcsec_tracing - cube_aligned_list[0].offset_from_center_y_arcsec_tracing)

    if verbose and n_rss > 1:

        for i in range(1,len(xx)-1):
            print("         {:2.0f} -> {:2.0f}           {:+5.3f}         {:+5.3f}".format(i,i+1,xxx[i-1],yyy[i-1]))      
        print("         {:2.0f} ->  1           {:+5.3f}         {:+5.3f}".format(len(xxx),xxx[-1],yyy[-1]))      
        print("           TOTAL:            {:5.3f}          {:5.3f}".format(np.nansum(xxx),np.nansum(yyy))) 
    
        print("\n> Alignment of n = {} cubes COMPLETED !".format(n_rss))
    return cube_aligned_list
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def save_cube_to_fits_file(cube, fits_file, description="", obj_name = "", path=""):  
    """
    Routine to save a cube as a fits file

    Parameters
    ----------
    Combined cube:
        cube
    Header: 
        Header        
    """
    
    if path != "" : fits_file=full_path(fits_file,path)
    
    fits_image_hdu = fits.PrimaryHDU(cube.data)
    #    errors = cube.data*0  ### TO BE DONE                
    #    error_hdu = fits.ImageHDU(errors)

    #wavelength =  cube.wavelength

    if cube.offsets_files_position == "" :
        fits_image_hdu.header['HISTORY'] = 'Combined datacube using PyKOALA'
    else:
        fits_image_hdu.header['HISTORY'] = 'Interpolated datacube using PyKOALA'
        
    fits_image_hdu.header['HISTORY'] = 'Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany,'
    fits_image_hdu.header['HISTORY'] = 'Blake Staples, Taylah Beard, Matt Owers, James Tocknell et al.'

    fits_image_hdu.header['HISTORY'] =  version #'Version 0.10 - 12th February 2019'    
    now=datetime.datetime.now()
    fits_image_hdu.header['HISTORY'] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    fits_image_hdu.header['DATE'] = now.strftime("%Y-%m-%dT%H:%M:%S") #'2002-09-16T18:52:44'   # /Date of FITS file creation
    
    fits_image_hdu.header['BITPIX']  =  16  

    fits_image_hdu.header["ORIGIN"]  = 'AAO'    #    / Originating Institution                        
    fits_image_hdu.header["TELESCOP"]= 'Anglo-Australian Telescope'    # / Telescope Name  
    fits_image_hdu.header["ALT_OBS"] =                 1164 # / Altitude of observatory in metres              
    fits_image_hdu.header["LAT_OBS"] =            -31.27704 # / Observatory latitude in degrees                
    fits_image_hdu.header["LONG_OBS"]=             149.0661 # / Observatory longitude in degrees 

    fits_image_hdu.header["INSTRUME"] = "AAOMEGA-KOALA"             # / Instrument in use  
    fits_image_hdu.header["GRATID"]  = cube.grating      # / Disperser ID 
    if cube.grating in red_gratings : SPECTID="RD"
    if cube.grating in blue_gratings : SPECTID="BL"
    fits_image_hdu.header["SPECTID"] = SPECTID                        # / Spectrograph ID                                
    fits_image_hdu.header["DICHROIC"]= 'X5700'                        # / Dichroic name   ---> CHANGE if using X6700!!
    
    if obj_name == "" :
        fits_image_hdu.header['OBJECT'] = cube.object    
    else:
        fits_image_hdu.header['OBJECT'] = obj_name
    fits_image_hdu.header['TOTALEXP'] = cube.total_exptime    
    fits_image_hdu.header['EXPTIMES'] = np.str(cube.exptimes)
                                       
    fits_image_hdu.header['NAXIS']   =   3                              # / number of array dimensions                       
    fits_image_hdu.header['NAXIS1']  =   cube.data.shape[1]        ##### CHECK !!!!!!!           
    fits_image_hdu.header['NAXIS2']  =   cube.data.shape[2]                 
    fits_image_hdu.header['NAXIS3']  =   cube.data.shape[0]                                     

    # WCS
    fits_image_hdu.header["RADECSYS"]= 'FK5'          # / FK5 reference system   
    fits_image_hdu.header["EQUINOX"] = 2000           # / [yr] Equinox of equatorial coordinates                         
    fits_image_hdu.header["WCSAXES"] =  3             # / Number of coordinate axes                      

    fits_image_hdu.header['CRPIX1']  = cube.data.shape[1]/2.         # / Pixel coordinate of reference point            
    fits_image_hdu.header['CDELT1']  = -cube.pixel_size_arcsec/3600. # / Coordinate increment at reference point      
    fits_image_hdu.header['CTYPE1']  = "RA--TAN" #'DEGREE'                               # / Coordinate type code                           
    fits_image_hdu.header['CRVAL1']  = cube.RA_centre_deg            # / Coordinate value at reference point            

    fits_image_hdu.header['CRPIX2']  = cube.data.shape[2]/2.         # / Pixel coordinate of reference point            
    fits_image_hdu.header['CDELT2']  = cube.pixel_size_arcsec/3600.  #  Coordinate increment at reference point        
    fits_image_hdu.header['CTYPE2']  = "DEC--TAN" #'DEGREE'                               # / Coordinate type code                           
    fits_image_hdu.header['CRVAL2']  = cube.DEC_centre_deg           # / Coordinate value at reference point            
 
    fits_image_hdu.header['RAcen'] = cube.RA_centre_deg
    fits_image_hdu.header['DECcen'] = cube.DEC_centre_deg
    fits_image_hdu.header['PIXsize'] = cube.pixel_size_arcsec
    fits_image_hdu.header['KERsize'] = cube.kernel_size_arcsec
    fits_image_hdu.header['Ncols'] = cube.data.shape[2]
    fits_image_hdu.header['Nrows'] = cube.data.shape[1]
    fits_image_hdu.header['PA'] = cube.PA

    # Wavelength calibration
    fits_image_hdu.header["CTYPE3"] = 'Wavelength'          # / Label for axis 3  
    fits_image_hdu.header["CUNIT3"] = 'Angstroms'           # / Units for axis 3     
    fits_image_hdu.header["CRVAL3"] = cube.CRVAL1_CDELT1_CRPIX1[0] # 7.692370611909E+03  / Co-ordinate value of axis 3
    fits_image_hdu.header["CDELT3"] = cube.CRVAL1_CDELT1_CRPIX1[1] # 1.575182431607E+00 
    fits_image_hdu.header["CRPIX3"] = cube.CRVAL1_CDELT1_CRPIX1[2] # 1024. / Reference pixel along axis 3
    fits_image_hdu.header["N_WAVE"] = cube.n_wave # 1024. / Reference pixel along axis 3

    fits_image_hdu.header["V_W_MIN"] = cube.valid_wave_min
    fits_image_hdu.header["V_W_MAX"] = cube.valid_wave_max
        
    scale_flux = 1.0
    try:
        scale_flux = cube.scale_flux
    except Exception:
        scale_flux = 1.0
    fits_image_hdu.header["SCAFLUX"] = scale_flux       # If the cube has been scaled in flux using scale_cubes_using_common_region  
    
    if cube.offsets_files_position == "" : # If cube.offsets_files_position is not given, it is a combined_cube
        #print("   THIS IS A COMBINED CUBE")
        fits_image_hdu.header["COMCUBE"] = True 
        is_combined_cube = True
        cofiles = len(cube.offsets_files) + 1 
        fits_image_hdu.header['COFILES'] =  cofiles # Number of combined files        
        if cofiles > 1:
            for i in (list(range(cofiles))):
                if i < 9:
                    text = "RSS_0"+np.str(i+1)
                else:
                    text = "RSS_"+np.str(i+1)
                fits_image_hdu.header[text] = cube.rss_list[i]    
    else: 
        #print(" THIS IS NOT A COMBINED CUBE")
        fits_image_hdu.header["COMCUBE"] = False 
        is_combined_cube = False
        fits_image_hdu.header['COFILES'] = 1

    offsets_text=" "
    if len(cube.offsets_files) != 0 :  # If offsets provided/obtained, this will not be 0 
        for i in range(len(cube.offsets_files)):  
            if i != 0: offsets_text=offsets_text+"  ,  "
            offsets_text=offsets_text+np.str(np.around(cube.offsets_files[i][0],3)) +" "+np.str(np.around(cube.offsets_files[i][1],3))
        fits_image_hdu.header['OFFSETS'] = offsets_text            # Offsets
        if is_combined_cube:         
            fits_image_hdu.header['OFF_POS'] = 0
        else:
            fits_image_hdu.header['OFF_POS'] = cube.offsets_files_position
       
    fits_image_hdu.header['ADRCOR'] = np.str(cube.adrcor) # ADR before, ADR was given as an input
    
    if len(cube.ADR_x_fit) != 0 :
        text=""
        for i in range(len(cube.ADR_x_fit)):
            if i != 0: text=text+"  ,  "
            text = text+np.str(cube.ADR_x_fit[i])
        fits_image_hdu.header['ADRxFIT'] = text
    
        text=""  
        for i in range(len(cube.ADR_y_fit)):
            if i != 0: text=text+"  ,  "
            text = text+np.str(cube.ADR_y_fit[i])
        fits_image_hdu.header['ADRyFIT'] = text 
    
    if np.nanmedian(cube.data) > 1:
        fits_image_hdu.header['FCAL'] = "False"
        fits_image_hdu.header['F_UNITS'] = "Counts"
        #flux_correction_hdu = fits.ImageHDU(0*wavelength)
    else:
        #flux_correction = fcal
        #flux_correction_hdu = fits.ImageHDU(flux_correction)
        fits_image_hdu.header['FCAL'] = "True"
        fits_image_hdu.header['F_UNITS'] = "erg s-1 cm-2 A-1"
                  
    if description == "":
        description = cube.description
    fits_image_hdu.header['DESCRIP'] = description.replace("\n","")
    fits_image_hdu.header['FILE_OUT'] = fits_file


#    hdu_list = fits.HDUList([fits_image_hdu, error_hdu])
#    hdu_list = fits.HDUList([fits_image_hdu, wavelengths_hdu, flux_correction_hdu])
    hdu_list = fits.HDUList([fits_image_hdu]) #, flux_correction_hdu])

    hdu_list.writeto(fits_file, overwrite=True) 
    if is_combined_cube:
        print("\n> Combined cube saved to file:")
        print(" ",fits_file)
    else:
        print("\n> Cube saved to file:")
        print(" ",fits_file)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def read_cube(filename, description="", half_size_for_centroid = 10, plot_spectra = False,
              valid_wave_min = 0, valid_wave_max = 0, edgelow=50,edgehigh=50, g2d=False,
              plot = False, verbose = True, print_summary = True,
              text_intro ="\n> Reading datacube from fits file:" ):
    
    if verbose: print(text_intro)
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
    
    ADR_x_fit_ = cube_fits_file[0].header['ADRXFIT'].split(',')
    ADR_x_fit =[]
    for j in range(len(ADR_x_fit_)):
                ADR_x_fit.append(float(ADR_x_fit_[j]))    
    ADR_y_fit_ = cube_fits_file[0].header['ADRYFIT'].split(',')
    ADR_y_fit =[]
    for j in range(len(ADR_y_fit_)):
                ADR_y_fit.append(float(ADR_y_fit_[j]))    


    adr_index_fit=len(ADR_y_fit) -1

    rss_files = []
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
            rss_files.append(cube_fits_file[0].header[head])
        
    if valid_wave_min == 0 : valid_wave_min = cube_fits_file[0].header["V_W_MIN"]  
    if valid_wave_max == 0 : valid_wave_max = cube_fits_file[0].header["V_W_MAX"] 
    
    wavelength = np.array([0.] * n_wave)    
    wavelength[np.int(CRPIX3)-1] = CRVAL3
    for i in range(np.int(CRPIX3)-2,-1,-1):
        wavelength[i] = wavelength[i+1] - CDELT3
    for i in range(np.int(CRPIX3),n_wave):
         wavelength[i] = wavelength[i-1] + CDELT3
   
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
    cube.trace_peak(box_x=box_x, box_y=box_y, plot=plot, edgelow=edgelow,edgehigh=edgehigh, adr_index_fit=adr_index_fit, g2d=g2d,
                    verbose =False)
    cube.get_integrated_map(plot=plot, plot_spectra=plot_spectra, verbose=verbose, plot_centroid=True, g2d=g2d) #,fcal=fcal, box_x=box_x, box_y=box_y)
    # For calibration stars, we get an integrated star flux and a seeing
    cube.integrated_star_flux = np.zeros_like(cube.wavelength) 
    cube.offsets_files = offsets_files
    cube.offsets_files_position = offsets_files_position
    cube.rss_files = rss_files    # Add this in Interpolated_cube
    cube.adrcor = adrcor
    cube.rss_list = filename
    
    if number_of_combined_files > 1 and verbose:
        print("\n> This cube was created using the following rss files:")
        for i in range(number_of_combined_files):
            print(" ",rss_files[i])
        
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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def save_rss_fits(rss, data=[[0],[0]], fits_file="RSS_rss.fits", text="RSS data", sol="",
                  description="", verbose=True): # fcal=[0],     # TASK_save_rss_fits
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
            print("\n> The data provided are NOT valid, as they have a shape",data.shape)
            print("  Using rss.intensity_corrected instead to create a RSS fits file !")
            data = rss.intensity_corrected
        else:
            print("\n> Using the data provided + structure of given RSS file to create fits file...")
    fits_image_hdu = fits.PrimaryHDU(data)
    
    fits_image_hdu.header['BITPIX']  =  16  

    fits_image_hdu.header["ORIGIN"]  = 'AAO'    #    / Originating Institution                        
    fits_image_hdu.header["TELESCOP"]= 'Anglo-Australian Telescope'    # / Telescope Name  
    fits_image_hdu.header["ALT_OBS"] =                 1164 # / Altitude of observatory in metres              
    fits_image_hdu.header["LAT_OBS"] =            -31.27704 # / Observatory latitude in degrees                
    fits_image_hdu.header["LONG_OBS"]=             149.0661 # / Observatory longitude in degrees 

    fits_image_hdu.header["INSTRUME"] = "AAOMEGA-KOALA"             # / Instrument in use  
    fits_image_hdu.header["GRATID"]  = rss.grating      # / Disperser ID 
    SPECTID = "UNKNOWN"
    if rss.grating in red_gratings : SPECTID="RD"
    if rss.grating in blue_gratings : SPECTID="BL"
    fits_image_hdu.header["SPECTID"] = SPECTID                        # / Spectrograph ID                                

    fits_image_hdu.header["DICHROIC"]= 'X5700'                        # / Dichroic name   ---> CHANGE if using X6700!!
    
    fits_image_hdu.header['OBJECT'] = rss.object    
    fits_image_hdu.header["EXPOSED"] = rss.exptime
    fits_image_hdu.header["ZDSTART"]= rss.ZDSTART 
    fits_image_hdu.header["ZDEND"]= rss.ZDEND
                                       
    fits_image_hdu.header['NAXIS']   =   2                              # / number of array dimensions                       
    fits_image_hdu.header['NAXIS1']  =   rss.intensity_corrected.shape[0]                 
    fits_image_hdu.header['NAXIS2']  =   rss.intensity_corrected.shape[1]                 

 
    fits_image_hdu.header['RAcen'] = rss.RA_centre_deg 
    fits_image_hdu.header['DECcen'] = rss.DEC_centre_deg 
    fits_image_hdu.header['TEL_PA'] = rss.PA

    fits_image_hdu.header["CTYPE2"] = 'Fibre number'          # / Label for axis 2  
    fits_image_hdu.header["CUNIT2"] = ' '           # / Units for axis 2     
    fits_image_hdu.header["CTYPE1"] = 'Wavelength'          # / Label for axis 2  
    fits_image_hdu.header["CUNIT1"] = 'Angstroms'           # / Units for axis 2     

    fits_image_hdu.header["CRVAL1"] = rss.CRVAL1_CDELT1_CRPIX1[0] #  / Co-ordinate value of axis 2
    fits_image_hdu.header["CDELT1"] = rss.CRVAL1_CDELT1_CRPIX1[1] # 
    fits_image_hdu.header["CRPIX1"] = rss.CRVAL1_CDELT1_CRPIX1[2] # 1024. / Reference pixel along axis 2
    fits_image_hdu.header["CRVAL2"] = 5.000000000000E-01 # / Co-ordinate value of axis 2  
    fits_image_hdu.header["CDELT2"] = 1.000000000000E+00 # / Co-ordinate increment along axis 2
    fits_image_hdu.header["CRPIX2"] = 1.000000000000E+00 # / Reference pixel along axis 2 
                 
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

    fits_image_hdu.header['HISTORY'] = '-- RSS processing using PyKOALA '+version
    #fits_image_hdu.header['HISTORY'] = 'Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany et al.'
    #fits_image_hdu.header['HISTORY'] =  version #'Version 0.10 - 12th February 2019'    
    now=datetime.datetime.now()
    
    fits_image_hdu.header['HISTORY'] = now.strftime("File created on %d %b %Y, %H:%M:%S using input file:")
    fits_image_hdu.header['DATE'] = now.strftime("%Y-%m-%dT%H:%M:%S") #'2002-09-16T18:52:44'   # /Date of FITS file creation
    #fits_image_hdu.header['HISTORY'] = 'using input file:'
    fits_image_hdu.header['HISTORY'] = rss.filename

    for item in rss.history:
        fits_image_hdu.header['HISTORY'] = item

    fits_image_hdu.header['HISTORY'] = "- Created fits file (this file) :"
    fits_image_hdu.header['HISTORY'] = " "+fits_file   
    fits_image_hdu.header['FILE_OUT'] = fits_file
    
    # TO BE DONE    
    errors = [0]  ### TO BE DONE                
    error_hdu = fits.ImageHDU(errors)


    # Header 2 with the RA and DEC info!    
    header2_all_fibres = rss.header2_data  
    header2_good_fibre = []
    header2_original_fibre = []
    header2_new_fibre = []
    header2_delta_RA=[]
    header2_delta_DEC=[]
    header2_2048 =[]
    header2_0 =[]
    
    fibre = 1
    for i in range (len(header2_all_fibres)):
        if header2_all_fibres[i][1]  == 1:
            header2_original_fibre.append(i+1)
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
    
    cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8])
    header2_hdu = fits.BinTableHDU.from_columns(cols)
    
    header2_hdu.header['CENRA']  =  rss.RA_centre_deg  / ( 180/np.pi )   # Must be in radians
    header2_hdu.header['CENDEC']  =  rss.DEC_centre_deg / ( 180/np.pi )
    
    hdu_list = fits.HDUList([fits_image_hdu,error_hdu, header2_hdu]) #  hdu_list = fits.HDUList([fits_image_hdu, wavelengths_hdu, flux_correction_hdu])

    hdu_list.writeto(fits_file, overwrite=True) 

    print('\n> '+text+'saved to file "'+fits_file+'"')      
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def align_blue_and_red_cubes(blue, red, half_size_for_centroid = 8, box_x= [], box_y =[],
                             verbose = True, plot = True, plot_centroid=True, g2d=False):
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
        try_read=blue+"  "
        if verbose: text_intro = "\n> Reading the blue cube from the fits file..."+try_read[-2:-1]
        blue_cube=read_cube(blue, text_intro = text_intro,
                            plot=plot, half_size_for_centroid=half_size_for_centroid, plot_spectra=False, verbose = verbose)
    except Exception:
        print("  - The blue cube is an object")
        blue_cube = blue
        
    try:
        try_read=red+"  "
        if verbose: text_intro = "\n> Reading the red cube from the fits file..."+try_read[-2:-1]
        red_cube=read_cube(red, text_intro = text_intro,
                           plot=plot, half_size_for_centroid=half_size_for_centroid, plot_spectra=False, verbose = verbose)
    except Exception:
        print("  - The red  cube is an object")
        red_cube = red
        if box_x == [] or box_y ==[] :
            box_x, box_y = red_cube.box_for_centroid(half_size_for_centroid = half_size_for_centroid, verbose=verbose)
        blue_cube.get_integrated_map(box_x = box_x, box_y = box_y, plot_spectra=False, plot=plot, verbose = verbose, plot_centroid=plot_centroid, g2d=g2d)
        red_cube.get_integrated_map(box_x = box_x, box_y = box_y, plot_spectra=False, plot=plot, verbose = verbose, plot_centroid=plot_centroid, g2d=g2d)
  
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
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


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
    for i in range (0,datos_len) :
        if formato[i] == "i" : datos[i]=np.loadtxt(fichero, skiprows=0, unpack=True, usecols=[i], dtype=int)
        if formato[i] == "s" : datos[i]=np.loadtxt(fichero, skiprows=0, unpack=True, usecols=[i], dtype=str)
        if formato[i] == "f" : datos[i]=np.loadtxt(fichero, skiprows=0, unpack=True, usecols=[i], dtype=float)    
    return datos
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def array_to_text_file(data, filename="array.dat", verbose=True  ):
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
    f=open(filename,"w")
    for i in range(len(data)):
        escribe = np.str(data[i])+" \n"
        f.write(escribe)
    f.close()
    if verbose: print("\n> Array saved in text file",filename," !!")
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def spectrum_to_text_file(wavelength, flux, filename="spectrum.txt", verbose=True ):
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
    f=open(filename,"w")
    for i in range(len(wavelength)):
        escribe = np.str(wavelength[i])+"  "+np.str(flux[i])+" \n"
        f.write(escribe)
    f.close()
    if verbose: print('\n> Spectrum saved in text file :\n  "'+filename+'"')
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def spectrum_to_fits_file(wavelength, flux, filename="spectrum.fits", name="spectrum", exptime=1, CRVAL1_CDELT1_CRPIX1=[0,0,0]): 
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
    hdu.header['NAXIS']   =   1                                                 
    hdu.header['NAXIS1']  =   len(wavelength)          
    hdu.header["CTYPE1"] = 'Wavelength' 
    hdu.header["CUNIT1"] = 'Angstroms'           
    if CRVAL1_CDELT1_CRPIX1[0] == 0: 
        hdu.header["CRVAL1"] = wavelength[0]
        hdu.header["CRPIX1"] = 1. 
        hdu.header["CDELT1"] = (wavelength[-1]-wavelength[0])/(len(wavelength)-1)
    else:           
        hdu.header["CRVAL1"] = CRVAL1_CDELT1_CRPIX1[0] # 7.692370611909E+03  / Co-ordinate value of axis 1
        hdu.header["CDELT1"] = CRVAL1_CDELT1_CRPIX1[1] # 1.575182431607E+00 
        hdu.header["CRPIX1"] = CRVAL1_CDELT1_CRPIX1[2] # 1024. / Reference pixel along axis 1
    # Extra info
    hdu.header['OBJECT'] = name    
    hdu.header["TOTALEXP"] = exptime
    hdu.header['HISTORY'] = 'Spectrum derived using the KOALA Python pipeline'
    hdu.header['HISTORY'] = 'Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany et al.'
    hdu.header['HISTORY'] =  version     
    now=datetime.datetime.now()
    hdu.header['HISTORY'] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    hdu.header['DATE'] = now.strftime("%Y-%m-%dT%H:%M:%S") #'2002-09-16T18:52:44'   # /Date of FITS file creation
    
    hdu.writeto(filename, overwrite=True) 
    print("\n> Spectrum saved in fits file",filename," !!")
    if name == "spectrum" : print("  No name given to the spectrum, named 'spectrum'.")
    if exptime == 1 : print("  No exposition time given, assumed exptime = 1")
    if CRVAL1_CDELT1_CRPIX1[0] == 0: print("  CRVAL1_CDELT1_CRPIX1 values not given, using ",wavelength[0],"1", (wavelength[-1]-wavelength[0])/(len(wavelength)-1))
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def apply_z(lambdas, z=0, v_rad=0, ref_line="Ha", l_ref=6562.82, l_obs = 6562.82, verbose = True):
    
    if ref_line=="Ha": l_ref = 6562.82
    if ref_line=="O3": l_ref = 5006.84
    if ref_line=="Hb": l_ref = 4861.33 
    
    if v_rad != 0:
        z = v_rad/C  
    if z == 0 :
        if verbose: 
            print("  Using line {}, l_rest = {:.2f}, observed at l_obs = {:.2f}. ".format(ref_line,l_ref,l_obs))
        z = l_obs/l_ref - 1.
        v_rad = z *C
        
    zlambdas =(z+1) * np.array(lambdas)
    
    if verbose:
        print("  Computing observed wavelengths using v_rad = {:.2f} km/s, redshift z = {:.06} :".format(v_rad,z))
        print("  REST :",lambdas)
        print("  z    :",np.round(zlambdas,2))
        
    return zlambdas
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def gauss(x, x0, y0, sigma):
    p = [x0, y0, sigma]
    return p[1]* np.exp(-0.5*((x-p[0])/p[2])**2)
def gauss_fix_x0(x, x0, y0, sigma):
    p = [y0, sigma]
    return p[0]* np.exp(-0.5*((x-x0)/p[1])**2)      
def gauss_flux (y0,sigma):  ### THIS DOES NOT WORK...
    return y0 * sigma * np.sqrt(2*np.pi)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def substract_given_gaussian(wavelength, spectrum, centre, peak=0, sigma=0,  flux=0, search_peak=False, allow_absorptions = False,
                             lowlow= 20, lowhigh=10, highlow=10, highhigh = 20, 
                             lmin=0, lmax=0, fmin=0, fmax=0, plot=True, fcal=False, verbose = True):
    """
    Substract a give Gaussian to a spectrum after fitting the continuum.
    """    
    do_it = False
    # Check that we have the numbers!
    if peak != 0 and sigma != 0 : do_it = True

    if peak == 0 and flux != 0 and sigma != 0:
        #flux = peak * sigma * np.sqrt(2*np.pi)
        peak = flux / (sigma * np.sqrt(2*np.pi))
        do_it = True 

    if sigma == 0 and flux != 0 and peak != 0 :
        #flux = peak * sigma * np.sqrt(2*np.pi)
        sigma = flux / (peak * np.sqrt(2*np.pi)) 
        do_it = True 
        
    if flux == 0 and sigma != 0 and peak != 0 :
        flux = peak * sigma * np.sqrt(2*np.pi)
        do_it = True

    if sigma != 0 and  search_peak == True:   do_it = True     

    if do_it == False:
        print("> Error! We need data to proceed! Give at least two of [peak, sigma, flux], or sigma and force peak to f[centre]")
        s_s = spectrum
    else:
        # Setup wavelength limits
        if lmin == 0 :
            lmin = centre-65.    # By default, +-65 A with respect to line
        if lmax == 0 :
            lmax = centre+65.
            
        # Extract subrange to fit
        w_spec = []
        f_spec = []
        w_spec.extend((wavelength[i]) for i in range(len(wavelength)) if (wavelength[i] > lmin and wavelength[i] < lmax) )    
        f_spec.extend((spectrum[i]) for i in range(len(wavelength)) if (wavelength[i] > lmin and wavelength[i] < lmax) )  
     
        # Setup min and max flux values in subrange to fit
        if fmin == 0 :
            fmin = np.nanmin(f_spec)            
        if fmax == 0 :
            fmax = np.nanmax(f_spec)                                 
    
        # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to centre
        w_cont=[]
        f_cont=[]
        w_cont.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > centre-lowlow and w_spec[i] < centre-lowhigh) or (w_spec[i] > centre+highlow and w_spec[i] < centre+highhigh)   )    
        f_cont.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > centre-lowlow and w_spec[i] < centre-lowhigh) or (w_spec[i] > centre+highlow and w_spec[i] < centre+highhigh)   )    
    
        # Linear Fit to continuum 
        try:    
            mm,bb = np.polyfit(w_cont, f_cont, 1)
        except Exception:
            bb = np.nanmedian(spectrum)
            mm = 0.
            if verbose: 
                print("      Impossible to get the continuum!")
                print("      Scaling the continuum to the median value") 
        continuum =   mm*np.array(w_spec)+bb  
        # c_cont = mm*np.array(w_cont)+bb  
        # rms continuum
        # rms_cont = np.nansum([ np.abs(f_cont[i] - c_cont[i])  for i in range(len(w_cont)) ]) / len(c_cont)

        if search_peak:
        # Search for index here w_spec(index) closest to line
            try:
                min_w = np.abs(np.array(w_spec)-centre)
                mini = np.nanmin(min_w)
                peak = f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]
                flux = peak * sigma * np.sqrt(2*np.pi)   
                if verbose: print("    Using peak as f[",np.round(centre,2),"] = ",np.round(peak,2)," and sigma = ", np.round(sigma,2), "    flux = ",np.round(flux,2))
            except Exception:
                if verbose: print("    Error trying to get the peak as requested wavelength is ",np.round(centre,2),"! Ignoring this fit!")
                peak = 0.
                flux = -0.0001
    
        no_substract = False
        if flux < 0:
            if allow_absorptions == False:
                if verbose and np.isnan(centre) == False : print("    WARNING! This is an ABSORPTION Gaussian! As requested, this Gaussian is NOT substracted!")
                no_substract = True
        if no_substract == False:     
            if verbose: print("    Substracting Gaussian at {:7.1f}  with peak ={:10.4f}   sigma ={:6.2f}  and flux ={:9.4f}".format(centre, peak,sigma,flux))
                
            gaussian_fit =  gauss(w_spec, centre, peak, sigma)
        
        
            index=0
            s_s=np.zeros_like(spectrum)
            for wave in range(len(wavelength)):
                s_s[wave]=spectrum[wave]
                if wavelength[wave] == w_spec[0] : 
                    s_s[wave] = f_spec[0]-gaussian_fit[0]
                    index=1
                if wavelength[wave] > w_spec[0] and wavelength[wave] <= w_spec[-1]:
                    s_s[wave] = f_spec[index]-gaussian_fit[index]
                    index=index+1
            if plot:  
                plt.figure(figsize=(10, 4))
                plt.plot(np.array(w_spec),np.array(f_spec), "b", lw=3, alpha = 0.5)
                plt.minorticks_on() 
                plt.xlabel("Wavelength [$\mathrm{\AA}$]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
                else:
                    plt.ylabel("Flux [ counts ]")
                plt.xlim(lmin,lmax)
                plt.ylim(fmin,fmax)
            
                # Vertical line at line
                plt.axvline(x=centre, color='k', linestyle='-', alpha=0.8)
                # Horizontal line at y = 0
                plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)    
                # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
                plt.axvspan(centre+highlow, centre+highhigh, facecolor='g', alpha=0.15,zorder=3)
                plt.axvspan(centre-lowlow, centre-lowhigh, facecolor='g', alpha=0.15,zorder=3)
                # Plot linear fit for continuum
                plt.plot(w_spec, continuum,"g--")
                # Plot Gaussian fit     
                plt.plot(w_spec, gaussian_fit+continuum, 'r-', alpha=0.8)    
                # Vertical lines to emission line
                #plt.axvline(x= low_limit, color='k', linestyle=':', alpha=0.5)
                #plt.axvline(x= high_limit, color='k', linestyle=':', alpha=0.5)  
                # Plot residuals
                #plt.plot(w_spec, residuals, 'k')
                #plt.title('Fit: x0=%.2f y0=%.2e sigma=%.2f  flux=%.2e  rms=%.3e' % (fit[0], fit[1], fit[2], gaussian_flux, rms_fit))
                plt.show()        
                plt.close()
                
                plt.figure(figsize=(10, 4))
                plt.plot(wavelength,spectrum, "r")
                plt.plot(wavelength,s_s, "c")
                plt.minorticks_on() 
                plt.xlabel("Wavelength [$\mathrm{\AA}$]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
                else:
                    plt.ylabel("Flux [ counts ]")
                plt.xlim(lmin,lmax)
                plt.ylim(fmin,fmax)
                plt.show()
                plt.close()
        else:
            s_s = spectrum
    return s_s
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fluxes(wavelength, s, line, lowlow= 14, lowhigh=6, highlow=6, highhigh = 14, lmin=0, lmax=0, fmin=0, fmax=0, 
           broad=2.355, plot=True, verbose=True, plot_sus = False, fcal = True, fit_continuum = True, median_kernel=35, warnings = True ):   # Broad is FWHM for Gaussian sigma= 1, 
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
        #print " is AN ARRAY"
    except Exception:
        #print " s is A LIST  -> must be converted into an ARRAY" 
        s = np.array(s)
    
    # Setup wavelength limits
    if lmin == 0 :
        lmin = line-65.    # By default, +-65 A with respect to line
    if lmax == 0 :
        lmax = line+65.
        
    # Extract subrange to fit
    w_spec = []
    f_spec = []
    w_spec.extend((wavelength[i]) for i in range(len(wavelength)) if (wavelength[i] > lmin and wavelength[i] < lmax) )    
    f_spec.extend((s[i]) for i in range(len(wavelength)) if (wavelength[i] > lmin and wavelength[i] < lmax) )  
 
    if np.isnan(np.nanmedian(f_spec)): 
        # The data are NAN!! Nothing to do
        if verbose: print("    There is no valid data in the wavelength range [{},{}] !!".format(lmin,lmax))
        
        resultado = [0, line, 0, 0, 0, 0, 0, 0, 0, 0, 0, s  ]  

        return resultado
        
    else:    
    
        ## 20 Sep 2020
        f_spec_m=signal.medfilt(f_spec,median_kernel)    # median_kernel = 35 default
        
        
        # Remove nans
        median_value = np.nanmedian(f_spec)
        f_spec = [median_value if np.isnan(x) else x for x in f_spec]  
            
            
        # Setup min and max flux values in subrange to fit
        if fmin == 0 :
            fmin = np.nanmin(f_spec)            
        if fmax == 0 :
            fmax = np.nanmax(f_spec) 
         
        # We have to find some "guess numbers" for the Gaussian. Now guess_centre is line
        guess_centre = line
                   
        # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to guess_centre
    
        w_cont=[]
        f_cont=[]
        w_cont.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-lowlow and w_spec[i] < guess_centre-lowhigh) or (w_spec[i] > guess_centre+highlow and w_spec[i] < guess_centre+highhigh)   )    
        f_cont.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-lowlow and w_spec[i] < guess_centre-lowhigh) or (w_spec[i] > guess_centre+highlow and w_spec[i] < guess_centre+highhigh)   )    
    
        if fit_continuum:
            # Linear Fit to continuum    
            f_cont_filtered=sig.medfilt(f_cont,np.int(median_kernel))
            #print line #f_cont
    #        if line == 8465.0:
    #            print w_cont
    #            print f_cont_filtered
    #            plt.plot(w_cont,f_cont_filtered)
    #            plt.show()
    #            plt.close()
    #            warnings=True
            try:    
                mm,bb = np.polyfit(w_cont, f_cont_filtered, 1)
            except Exception:
                bb = np.nanmedian(f_cont_filtered)
                mm = 0.
                if warnings: 
                    print("    Impossible to get the continuum!")
                    print("    Scaling the continuum to the median value b = ",bb,":  cont =  0 * w_spec  + ", bb)
            continuum =   mm*np.array(w_spec)+bb  
            c_cont = mm*np.array(w_cont)+bb  
    
        else:    
            # Median value in each continuum range  # NEW 15 Sep 2019
            w_cont_low = []
            f_cont_low = []
            w_cont_low.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-lowlow and w_spec[i] < guess_centre-lowhigh)   )    
            f_cont_low.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-lowlow and w_spec[i] < guess_centre-lowhigh)   )    
            median_w_cont_low = np.nanmedian(w_cont_low)
            median_f_cont_low = np.nanmedian(f_cont_low)
            w_cont_high = []
            f_cont_high = []
            w_cont_high.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre+highlow and w_spec[i] < guess_centre+highhigh)   )    
            f_cont_high.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre+highlow and w_spec[i] < guess_centre+highhigh)   )    
            median_w_cont_high = np.nanmedian(w_cont_high)
            median_f_cont_high = np.nanmedian(f_cont_high)    
            
            b = (median_f_cont_low-median_f_cont_high)/(median_w_cont_low-median_w_cont_high)
            a = median_f_cont_low- b * median_w_cont_low
                
            continuum =  a + b*np.array(w_spec)
            c_cont    =  a + b*np.array(w_cont)  
        
        
        # rms continuum
        rms_cont = np.nansum([ np.abs(f_cont[i] - c_cont[i])  for i in range(len(w_cont)) ]) / len(c_cont)
    
        # Search for index here w_spec(index) closest to line
        min_w = np.abs(np.array(w_spec)-line)
        mini = np.nanmin(min_w)
    #    guess_peak = f_spec[min_w.tolist().index(mini)]   # WE HAVE TO SUSTRACT CONTINUUM!!!
        guess_peak = f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]
    
        # LOW limit
        low_limit=0
        w_fit = []
        f_fit = []
        w_fit.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-15 and w_spec[i] < guess_centre))    
        f_fit.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-15 and w_spec[i] < guess_centre))        
        if fit_continuum: 
            c_fit=mm*np.array(w_fit)+bb 
        else: 
            c_fit=b*np.array(w_fit)+a   
    
        fs=[]
        ws=[]
        for ii in range(len(w_fit)-1,1,-1):
            if f_fit[ii]/c_fit[ii] < 1.05 and f_fit[ii-1]/c_fit[ii-1] < 1.05 and low_limit == 0: low_limit = w_fit[ii]
    #        if f_fit[ii]/c_fit[ii] < 1.05 and low_limit == 0: low_limit = w_fit[ii]
            fs.append(f_fit[ii]/c_fit[ii])
            ws.append(w_fit[ii])
        if low_limit == 0: 
            sorted_by_flux=np.argsort(fs)
            try:
                low_limit =  ws[sorted_by_flux[0]]
            except Exception:
                plot=True
                low_limit = 0
            
        # HIGH LIMIT        
        high_limit=0
        w_fit = []
        f_fit = []
        w_fit.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre and w_spec[i] < guess_centre+15))    
        f_fit.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre and w_spec[i] < guess_centre+15))    
        if fit_continuum: 
            c_fit=mm*np.array(w_fit)+bb 
        else: 
            c_fit=b*np.array(w_fit)+a
            
        fs=[]
        ws=[]
        for ii in range(len(w_fit)-1):
            if f_fit[ii]/c_fit[ii] < 1.05 and f_fit[ii+1]/c_fit[ii+1] < 1.05 and high_limit == 0: high_limit = w_fit[ii]
    #        if f_fit[ii]/c_fit[ii] < 1.05 and high_limit == 0: high_limit = w_fit[ii]
            fs.append(f_fit[ii]/c_fit[ii])
            ws.append(w_fit[ii])
        if high_limit == 0: 
            sorted_by_flux=np.argsort(fs)
            try:
                high_limit =  ws[sorted_by_flux[0]]    
            except Exception:
                plot=True
                high_limit = 0        
    
        # Guess centre will be the highest value in the range defined by [low_limit,high_limit]
        
        try:    
            rango = np.where((high_limit >= wavelength ) & (low_limit <= wavelength)) 
            index_maximo_del_rango = s.tolist().index(np.nanmax(s[rango]))
            guess_centre = wavelength[index_maximo_del_rango]
        except Exception:
            guess_centre = line ####  It was 0 before
            
            
        # Fit a Gaussian to data - continuum   
        p0 = [guess_centre, guess_peak, broad/2.355]  # broad is the Gaussian sigma, 1.0 for emission lines
        try:
            fit, pcov = curve_fit(gauss, w_spec, f_spec-continuum, p0=p0, maxfev=10000)   # If this fails, increase maxfev...
            fit_error = np.sqrt(np.diag(pcov))
            
            # New 28th Feb 2019: Check central value between low_limit and high_limit
            # Better: between guess_centre - broad, guess_centre + broad
            # If not, redo fit fixing central value to the peak (it does not work... just fix FWHM= (high_limit-low_limit)/2.5 )
    
            if verbose != False: print(" ----------------------------------------------------------------------------------------")
    #        if low_limit < fit[0] < high_limit:
            if fit[0] <  guess_centre - broad  or fit[0] >  guess_centre + broad:
    #            if verbose: print "  Fitted center wavelength", fit[0],"is NOT in the range [",low_limit,",",high_limit,"]"
                if verbose: print("    Fitted center wavelength", fit[0],"is NOT in the expected range [",guess_centre - broad,",",guess_centre + broad,"]")
    
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
    
                fit[0]=guess_centre
                fit_error[0] = 0.000001
                fit[1]=guess_peak
                fit_error[1] = 0.000001
                fit[2] = broad/2.355
                fit_error[2] = 0.000001            
            else:
                if verbose: print("    Fitted center wavelength", fit[0],"IS in the expected range [",guess_centre - broad,",",guess_centre + broad,"]")
    
    
            if verbose: print("    Fit parameters =  ", fit[0], fit[1], fit[2])
            if fit[2] == broad and warnings == True : 
                print("    WARNING: Fit in",fit[0],"failed! Using given centre wavelength (cw), peak at (cv) & sigma = broad/2.355 given.")             
            gaussian_fit =  gauss(w_spec, fit[0], fit[1], fit[2])
     
    
            # Estimate rms of the Gaussian fit in range [low_limit, high_limit]
            residuals = f_spec-gaussian_fit-continuum
            rms_fit = np.nansum([ ((residuals[i]**2)/(len(residuals)-2))**0.5 for i in range(len(w_spec))  if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)  ])  
        
            # Fluxes, FWHM and Eq. Width calculations
            gaussian_flux = gauss_flux(fit[1],fit[2])
            error1 = np.abs(gauss_flux(fit[1]+fit_error[1],fit[2]) - gaussian_flux)
            error2 = np.abs(gauss_flux(fit[1],fit[2]+fit_error[2]) - gaussian_flux)
            gaussian_flux_error = 1 / ( 1/error1**2  + 1/error2**2 )**0.5
    
       
            fwhm=fit[2]*2.355
            fwhm_error = fit_error[2] *2.355
            fwhm_vel = fwhm / fit[0] * C  
            fwhm_vel_error = fwhm_error / fit[0] * C  
        
            gaussian_ew = gaussian_flux/np.nanmedian(f_cont)
            gaussian_ew_error =   gaussian_ew * gaussian_flux_error/gaussian_flux  
            
            # Integrated flux
            # IRAF: flux = sum ((I(i)-C(i)) * (w(i2) - w(i1)) / (i2 - i2)    
            flux = np.nansum([ (f_spec[i]-continuum[i])*(w_spec[i+1]-w_spec[i])    for  i in range(len(w_spec))  if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)  ]) 
            flux_error = rms_cont * (high_limit - low_limit)
            wave_resolution = (wavelength[-1]-wavelength[0])/len(wavelength)
            ew =  wave_resolution * np.nansum ([ (1 - f_spec[i]/continuum[i]) for i in range(len(w_spec))  if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)  ])   
            ew_error = np.abs(ew*flux_error/flux)   
            gauss_to_integrated = gaussian_flux/flux * 100.
      
            index=0
            s_s=np.zeros_like(s)
            for wave in range(len(wavelength)):
                s_s[wave]=s[wave]
                if wavelength[wave] == w_spec[0] : 
                    s_s[wave] = f_spec[0]-gaussian_fit[0]
                    index=1
                if wavelength[wave] > w_spec[0] and wavelength[wave] <= w_spec[-1]:
                    s_s[wave] = f_spec[index]-gaussian_fit[index]
                    index=index+1
        
            # Plotting 
            ptitle = 'Fit: x0=%.2f y0=%.2e sigma=%.2f  flux=%.2e  rms=%.3e' % (fit[0], fit[1], fit[2], gaussian_flux, rms_fit)
            if plot :
                plt.figure(figsize=(10, 4))
                # Plot input spectrum
                plt.plot(np.array(w_spec),np.array(f_spec), "b", lw=3, alpha = 0.8)
                # Plot median input spectrum
                plt.plot(np.array(w_spec),np.array(f_spec_m), "orange", lw=3, alpha = 0.5)   # 2021: era "g"
                # Plot spectrum - gauss subtracted
                plt.plot(wavelength,s_s,"g",lw=3, alpha = 0.6)
                                
                plt.minorticks_on() 
                plt.xlabel("Wavelength [$\mathrm{\AA}$ ]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
                else:
                    plt.ylabel("Flux [ counts ]")
                plt.xlim(lmin,lmax)
                plt.ylim(fmin,fmax)
            
                # Vertical line at guess_centre
                plt.axvline(x=guess_centre, color='r', linestyle='-', alpha=0.3)
                # Horizontal line at y = 0
                plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)    
                # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
                plt.axvspan(guess_centre+highlow, guess_centre+highhigh, facecolor='g', alpha=0.15,zorder=3)
                plt.axvspan(guess_centre-lowlow, guess_centre-lowhigh, facecolor='g', alpha=0.15,zorder=3)
                # Plot linear fit for continuum
                plt.plot(w_spec, continuum,"g--")
                # Plot Gaussian fit     
                plt.plot(w_spec, gaussian_fit+continuum, 'r-', alpha=0.8)    
                # Vertical line at Gaussian center
                plt.axvline(x=fit[0], color='k', linestyle='-', alpha=0.5)
                # Vertical lines to emission line
                plt.axvline(x= low_limit, color='k', linestyle=':', alpha=0.5)
                plt.axvline(x= high_limit, color='k', linestyle=':', alpha=0.5)  
                # Plot residuals
                plt.plot(w_spec, residuals, 'k')
                plt.title(ptitle)
                plt.show()
          
            # Printing results
            if verbose :
                print("\n  - Gauss and continuum fitting + integrated flux calculations:\n")
                print("    rms continuum = %.3e erg/cm/s/A " % (rms_cont))       
                print("    Gaussian Fit parameters: x0 = ( %.2f +- %.2f )  A " % (fit[0], fit_error[0]))
                print("                             y0 = ( %.3f +- %.3f )  1E-16 erg/cm2/s/A" % (fit[1]/1E-16, fit_error[1]/1E-16 ))
                print("                          sigma = ( %.3f +- %.3f )  A" % (fit[2], fit_error[2]))  
                print("                        rms fit = %.3e erg/cm2/s/A" % (rms_fit))
                print("    Gaussian Flux = ( %.2f +- %.2f ) 1E-16 erg/s/cm2         (error = %.1f per cent)" % (gaussian_flux/1E-16, gaussian_flux_error/1E-16, gaussian_flux_error/gaussian_flux*100))
                print("    FWHM          = ( %.3f +- %.3f ) A    =   ( %.1f +- %.1f ) km/s " % (fwhm, fwhm_error, fwhm_vel, fwhm_vel_error))
                print("    Eq. Width     = ( %.1f +- %.1f ) A" % (-gaussian_ew, gaussian_ew_error))           
                print("\n    Integrated flux  = ( %.2f +- %.2f ) 1E-16 erg/s/cm2      (error = %.1f per cent) " % ( flux/1E-16, flux_error/1E-16, flux_error/flux *100))    
                print("    Eq. Width        = ( %.1f +- %.1f ) A" % (ew, ew_error))
                print("    Gauss/Integrated = %.2f per cent " % gauss_to_integrated)
        
                
            # Plot independent figure with substraction if requested        
            if plot_sus: plot_plot(wavelength,[s,s_s], xmin=lmin, xmax=lmax, ymin=fmin, ymax=fmax, fcal=fcal, frameon=True, ptitle=ptitle)
          
        #                     0      1         2                3               4              5      6         7        8        9     10      11
            resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, s_s  ]
            return resultado 
        except Exception:
            if verbose: 
                print("  - Gaussian fit failed!")
                print("    However, we can compute the integrated flux and the equivalent width:")
           
            flux = np.nansum([ (f_spec[i]-continuum[i])*(w_spec[i+1]-w_spec[i])    for  i in range(len(w_spec))  if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)  ]) 
            flux_error = rms_cont * (high_limit - low_limit)
            wave_resolution = (wavelength[-1]-wavelength[0])/len(wavelength)
            ew =  wave_resolution * np.nansum ([ (1 - f_spec[i]/continuum[i]) for i in range(len(w_spec))  if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)  ])   
            ew_error = np.abs(ew*flux_error/flux)   
    
            if verbose:
                print("    Integrated flux  = ( %.2f +- %.2f ) 1E-16 erg/s/cm2      (error = %.1f per cent) " % ( flux/1E-16, flux_error/1E-16, flux_error/flux *100))    
                print("    Eq. Width        = ( %.1f +- %.1f ) A" % (ew, ew_error))
            
            resultado = [0, guess_centre, 0, 0, 0, 0, 0, flux, flux_error, ew, ew_error, s  ]  # guess_centre was identified at maximum value in the [low_limit,high_limit] range but Gaussian fit failed
    
    
           # Plotting 
            if plot :
                plt.figure(figsize=(10, 4))
                plt.plot(np.array(w_spec),np.array(f_spec), "b", lw=3, alpha = 0.5)
                plt.minorticks_on() 
                plt.xlabel("Wavelength [$\mathrm{\AA}$]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
                else:
                    plt.ylabel("Flux [ counts ]")            
                plt.xlim(lmin,lmax)
                plt.ylim(fmin,fmax)
            
                # Vertical line at guess_centre
                plt.axvline(x=guess_centre, color='r', linestyle='-', alpha=0.5)
                # Horizontal line at y = 0
                plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)    
                # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
                plt.axvspan(guess_centre+highlow, guess_centre+highhigh, facecolor='g', alpha=0.15,zorder=3)
                plt.axvspan(guess_centre-lowlow, guess_centre-lowhigh, facecolor='g', alpha=0.15,zorder=3)
                # Plot linear fit for continuum
                plt.plot(w_spec, continuum,"g--")
                # Plot Gaussian fit     
    #            plt.plot(w_spec, gaussian_fit+continuum, 'r-', alpha=0.8)    
                # Vertical line at Gaussian center
    #            plt.axvline(x=fit[0], color='k', linestyle='-', alpha=0.5)
                # Vertical lines to emission line
                plt.axvline(x= low_limit, color='k', linestyle=':', alpha=0.5)
                plt.axvline(x= high_limit, color='k', linestyle=':', alpha=0.5)  
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
    return p[1]* np.exp(-0.5*((x-p[0])/p[2])**2) + p[4]* np.exp(-0.5*((x-p[3])/p[5])**2)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def dfluxes(wavelength, s, line1, line2, lowlow= 25, lowhigh=15, highlow=15, highhigh = 25, 
            lmin=0, lmax=0, fmin=0, fmax=0,
            broad1=2.355, broad2=2.355, sus_line1=True, sus_line2=True,
            plot=True, verbose=True, plot_sus = False, fcal = True, 
            fit_continuum = True, median_kernel=35, warnings = True ):   # Broad is FWHM for Gaussian sigma= 1, 
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
    if lmin == 0 :
        lmin = line1-65.    # By default, +-65 A with respect to line
    if lmax == 0 :
        lmax = line2+65.
        
    # Extract subrange to fit
    w_spec = []
    f_spec = []
    w_spec.extend((wavelength[i]) for i in range(len(wavelength)) if (wavelength[i] > lmin and wavelength[i] < lmax) )    
    f_spec.extend((s[i]) for i in range(len(wavelength)) if (wavelength[i] > lmin and wavelength[i] < lmax) )  
 
    
    if np.nanmedian(f_spec) == np.nan: print("  NO HAY DATOS.... todo son NANs!")

    
    # Setup min and max flux values in subrange to fit
    if fmin == 0 :
        fmin = np.nanmin(f_spec)            
    if fmax == 0 :
        fmax = np.nanmax(f_spec) 
     

    # We have to find some "guess numbers" for the Gaussian
    # Now guess_centre is line
    guess_centre1 = line1
    guess_centre2 = line2  
    guess_centre = (guess_centre1+guess_centre2)/2.         
    # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to guess_centre
 

    w_cont=[]
    f_cont=[]
    w_cont.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-lowlow and w_spec[i] < guess_centre-lowhigh) or (w_spec[i] > guess_centre+highlow and w_spec[i] < guess_centre+highhigh)   )    
    f_cont.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-lowlow and w_spec[i] < guess_centre-lowhigh) or (w_spec[i] > guess_centre+highlow and w_spec[i] < guess_centre+highhigh)   )    

    if fit_continuum:
        # Linear Fit to continuum    
        f_cont_filtered=sig.medfilt(f_cont,np.int(median_kernel))
        try:    
            mm,bb = np.polyfit(w_cont, f_cont_filtered, 1)
        except Exception:
            bb = np.nanmedian(f_cont_filtered)
            mm = 0.
            if warnings: 
                print("  Impossible to get the continuum!")
                print("  Scaling the continuum to the median value")          
        continuum =   mm*np.array(w_spec)+bb  
        c_cont = mm*np.array(w_cont)+bb  

    else:    
        # Median value in each continuum range  # NEW 15 Sep 2019
        w_cont_low = []
        f_cont_low = []
        w_cont_low.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-lowlow and w_spec[i] < guess_centre-lowhigh)   )    
        f_cont_low.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-lowlow and w_spec[i] < guess_centre-lowhigh)   )    
        median_w_cont_low = np.nanmedian(w_cont_low)
        median_f_cont_low = np.nanmedian(f_cont_low)
        w_cont_high = []
        f_cont_high = []
        w_cont_high.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre+highlow and w_spec[i] < guess_centre+highhigh)   )    
        f_cont_high.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre+highlow and w_spec[i] < guess_centre+highhigh)   )    
        median_w_cont_high = np.nanmedian(w_cont_high)
        median_f_cont_high = np.nanmedian(f_cont_high)    
        
        b = (median_f_cont_low-median_f_cont_high)/(median_w_cont_low-median_w_cont_high)
        a = median_f_cont_low- b * median_w_cont_low
            
        continuum =  a + b*np.array(w_spec)
        c_cont = b*np.array(w_cont)+ a  
    
    # rms continuum
    rms_cont = np.nansum([ np.abs(f_cont[i] - c_cont[i])  for i in range(len(w_cont)) ]) / len(c_cont)

    # Search for index here w_spec(index) closest to line
    min_w = np.abs(np.array(w_spec)-line1)
    mini = np.nanmin(min_w)
    guess_peak1 = f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]
    min_w = np.abs(np.array(w_spec)-line2)
    mini = np.nanmin(min_w)
    guess_peak2 = f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]

    # Search for beginning/end of emission line, choosing line +-10    
    # 28th Feb 2019: Check central value between low_limit and high_limit

    # LOW limit
    low_limit=0
    w_fit = []
    f_fit = []
    w_fit.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre1-15 and w_spec[i] < guess_centre1))    
    f_fit.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre1-15 and w_spec[i] < guess_centre1))        
    if fit_continuum: 
        c_fit=mm*np.array(w_fit)+bb 
    else: 
        c_fit=b*np.array(w_fit)+a
    

    fs=[]
    ws=[]
    for ii in range(len(w_fit)-1,1,-1):
        if f_fit[ii]/c_fit[ii] < 1.05 and f_fit[ii-1]/c_fit[ii-1] < 1.05 and low_limit == 0: low_limit = w_fit[ii]
#        if f_fit[ii]/c_fit[ii] < 1.05 and low_limit == 0: low_limit = w_fit[ii]
        fs.append(f_fit[ii]/c_fit[ii])
        ws.append(w_fit[ii])
    if low_limit == 0: 
        sorted_by_flux=np.argsort(fs)
        low_limit =  ws[sorted_by_flux[0]]
        
    # HIGH LIMIT        
    high_limit=0
    w_fit = []
    f_fit = []
    w_fit.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre2 and w_spec[i] < guess_centre2+15))    
    f_fit.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre2 and w_spec[i] < guess_centre2+15))    
    if fit_continuum: 
        c_fit=mm*np.array(w_fit)+bb 
    else: 
        c_fit=b*np.array(w_fit)+a
        
    fs=[]
    ws=[]
    for ii in range(len(w_fit)-1):
        if f_fit[ii]/c_fit[ii] < 1.05 and f_fit[ii+1]/c_fit[ii+1] < 1.05 and high_limit == 0: high_limit = w_fit[ii]
#        if f_fit[ii]/c_fit[ii] < 1.05 and high_limit == 0: high_limit = w_fit[ii]
        fs.append(f_fit[ii]/c_fit[ii])
        ws.append(w_fit[ii])
    if high_limit == 0: 
        sorted_by_flux=np.argsort(fs)
        high_limit =  ws[sorted_by_flux[0]]     
                   
    # Fit a Gaussian to data - continuum   
    p0 = [guess_centre1, guess_peak1, broad1/2.355, guess_centre2, guess_peak2, broad2/2.355]  # broad is the Gaussian sigma, 1.0 for emission lines
    try:
        fit, pcov = curve_fit(dgauss, w_spec, f_spec-continuum, p0=p0, maxfev=10000)   # If this fails, increase maxfev...
        fit_error = np.sqrt(np.diag(pcov))


        # New 28th Feb 2019: Check central value between low_limit and high_limit
        # Better: between guess_centre - broad, guess_centre + broad
        # If not, redo fit fixing central value to the peak (it does not work... just fix FWHM= (high_limit-low_limit)/2.5 )

        if verbose != False: print(" ----------------------------------------------------------------------------------------")
        if fit[0] <  guess_centre1 - broad1  or fit[0] >  guess_centre1 + broad1 or fit[3] <  guess_centre2 - broad2  or fit[3] >  guess_centre2 + broad2:
            if warnings: 
                if fit[0] <  guess_centre1 - broad1  or fit[0] >  guess_centre1 + broad1: 
                    print("    Fitted center wavelength", fit[0],"is NOT in the expected range [",guess_centre1 - broad1,",",guess_centre1 + broad1,"]")
                else:
                    print("    Fitted center wavelength", fit[0],"is in the expected range [",guess_centre1 - broad1,",",guess_centre1 + broad1,"]")
                if fit[3] <  guess_centre2 - broad2  or fit[3] >  guess_centre2 + broad2: 
                    print("    Fitted center wavelength", fit[3],"is NOT in the expected range [",guess_centre2 - broad2,",",guess_centre2 + broad2,"]")
                else:
                    print("    Fitted center wavelength", fit[3],"is in the expected range [",guess_centre2 - broad2,",",guess_centre2 + broad2,"]")
                print("    Fit failed!")
                
            fit[0]=guess_centre1
            fit_error[0] = 0.000001
            fit[1]=guess_peak1
            fit_error[1] = 0.000001
            fit[2] = broad1/2.355
            fit_error[2] = 0.000001    
            fit[3]=guess_centre2
            fit_error[3] = 0.000001
            fit[4]=guess_peak2
            fit_error[4] = 0.000001
            fit[5] = broad2/2.355
            fit_error[5] = 0.000001
        else:
            if warnings: print("    Fitted center wavelength", fit[0],"is in the expected range [",guess_centre1 - broad1,",",guess_centre1 + broad1,"]")
            if warnings: print("    Fitted center wavelength", fit[3],"is in the expected range [",guess_centre2 - broad2,",",guess_centre2 + broad2,"]")
        

        if warnings: 
            print("    Fit parameters =  ", fit[0], fit[1], fit[2]) 
            print("                      ", fit[3], fit[4], fit[5])
        if fit[2] == broad1/2.355 and warnings == True : 
            print("    WARNING: Fit in",fit[0],"failed! Using given centre wavelengths (cw), peaks at (cv) & sigmas=broad/2.355 given.")       # CHECK THIS         

        gaussian_fit =  dgauss(w_spec, fit[0], fit[1], fit[2],fit[3], fit[4], fit[5])
        
        gaussian_1 = gauss(w_spec, fit[0], fit[1], fit[2])
        gaussian_2 = gauss(w_spec, fit[3], fit[4], fit[5])
 

        # Estimate rms of the Gaussian fit in range [low_limit, high_limit]
        residuals = f_spec-gaussian_fit-continuum
        rms_fit = np.nansum([ ((residuals[i]**2)/(len(residuals)-2))**0.5 for i in range(len(w_spec))  if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)  ])  
    
        # Fluxes, FWHM and Eq. Width calculations  # CHECK THIS , not well done for dfluxes !!!
        
        gaussian_flux_1 = gauss_flux(fit[1],fit[2])
        gaussian_flux_2 = gauss_flux(fit[4],fit[5]) 
        gaussian_flux = gaussian_flux_1+ gaussian_flux_2      
        if warnings: 
            print("    Gaussian flux  =  ", gaussian_flux_1, " + ",gaussian_flux_2," = ",gaussian_flux)
            print("    Gaussian ratio =  ", gaussian_flux_1/gaussian_flux_2)
        
        error1 = np.abs(gauss_flux(fit[1]+fit_error[1],fit[2]) - gaussian_flux)
        error2 = np.abs(gauss_flux(fit[1],fit[2]+fit_error[2]) - gaussian_flux)
        gaussian_flux_error = 1 / ( 1/error1**2  + 1/error2**2 )**0.5
    
        fwhm=fit[2]*2.355
        fwhm_error = fit_error[2] *2.355
        fwhm_vel = fwhm / fit[0] * C  
        fwhm_vel_error = fwhm_error / fit[0] * C  
    
        gaussian_ew = gaussian_flux/np.nanmedian(f_cont)
        gaussian_ew_error =   gaussian_ew * gaussian_flux_error/gaussian_flux  
        
        # Integrated flux
        # IRAF: flux = sum ((I(i)-C(i)) * (w(i2) - w(i1)) / (i2 - i2)    
        flux = np.nansum([ (f_spec[i]-continuum[i])*(w_spec[i+1]-w_spec[i])    for  i in range(len(w_spec))  if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)  ]) 
        flux_error = rms_cont * (high_limit - low_limit)
        wave_resolution = (wavelength[-1]-wavelength[0])/len(wavelength)
        ew =  wave_resolution * np.nansum ([ (1 - f_spec[i]/continuum[i]) for i in range(len(w_spec))  if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)  ])   
        ew_error = np.abs(ew*flux_error/flux)   
        gauss_to_integrated = gaussian_flux/flux * 100.
    
        # Plotting 
        if plot :
            plt.figure(figsize=(10, 4))
            #Plot input spectrum
            plt.plot(np.array(w_spec),np.array(f_spec), "blue", lw=2, alpha = 0.7)
            plt.minorticks_on() 
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim((line1+line2)/2-40,(line1+line2)/2+40)
            plt.ylim(fmin,fmax)
        
            # Vertical line at guess_centre
            plt.axvline(x=guess_centre1, color='r', linestyle='-', alpha=0.5)
            plt.axvline(x=guess_centre2, color='r', linestyle='-', alpha=0.5)

            # Horizontal line at y = 0
            plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)    
            # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
            plt.axvspan(guess_centre+highlow, guess_centre+highhigh, facecolor='g', alpha=0.15,zorder=3)
            plt.axvspan(guess_centre-lowlow, guess_centre-lowhigh, facecolor='g', alpha=0.15,zorder=3)
            # Plot linear fit for continuum
            plt.plot(w_spec, continuum,"g--")
            # Plot Gaussian fit     
            plt.plot(w_spec, gaussian_fit+continuum, 'r-', alpha=0.8)    
            # Vertical line at Gaussian center
            plt.axvline(x=fit[0], color='k', linestyle='-', alpha=0.5)
            plt.axvline(x=fit[3], color='k', linestyle='-', alpha=0.5)
            # Plot Gaussians + cont
            plt.plot(w_spec, gaussian_fit+continuum, 'r-', alpha=0.5, lw=3)            
            plt.plot(w_spec, gaussian_1+continuum, color="navy",linestyle='--', alpha=0.8)
            plt.plot(w_spec, gaussian_2+continuum, color="#1f77b4",linestyle='--', alpha=0.8)
            plt.plot(w_spec, np.array(f_spec)-(gaussian_fit), 'orange', alpha=0.4, linewidth=5)          

            # Vertical lines to emission line
            plt.axvline(x= low_limit, color='k', linestyle=':', alpha=0.5)
            plt.axvline(x= high_limit, color='k', linestyle=':', alpha=0.5)  
            plt.title('Double Gaussian Fit') # Fit: x0=%.2f y0=%.2e sigma=%.2f  flux=%.2e  rms=%.3e' % (fit[0], fit[1], fit[2], gaussian_flux, rms_fit))
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
        if verbose :
            #print "\n> WARNING !!! CAREFUL WITH THE VALUES PROVIDED BELOW, THIS TASK NEEDS TO BE UPDATED!\n"
            print("\n> Gauss and continuum fitting + integrated flux calculations:\n")
            print("  rms continuum = %.3e erg/cm/s/A " % (rms_cont))       
            print("  Gaussian Fit parameters: x0 = ( %.2f +- %.2f )  A " % (fit[0], fit_error[0]))
            print("                           y0 = ( %.3f +- %.3f )  1E-16 erg/cm2/s/A" % (fit[1]/1E-16, fit_error[1]/1E-16 ))
            print("                        sigma = ( %.3f +- %.3f )  A" % (fit[2], fit_error[2]))  
            print("                      rms fit = %.3e erg/cm2/s/A" % (rms_fit))
            print("  Gaussian Flux = ( %.2f +- %.2f ) 1E-16 erg/s/cm2         (error = %.1f per cent)" % (gaussian_flux/1E-16, gaussian_flux_error/1E-16, gaussian_flux_error/gaussian_flux*100))
            print("  FWHM          = ( %.3f +- %.3f ) A    =   ( %.1f +- %.1f ) km/s " % (fwhm, fwhm_error, fwhm_vel, fwhm_vel_error))
            print("  Eq. Width     = ( %.1f +- %.1f ) A" % (-gaussian_ew, gaussian_ew_error))           
            print("\n  Integrated flux  = ( %.2f +- %.2f ) 1E-16 erg/s/cm2      (error = %.1f per cent) " % ( flux/1E-16, flux_error/1E-16, flux_error/flux *100))    
            print("  Eq. Width        = ( %.1f +- %.1f ) A" % (ew, ew_error))
            print("  Gauss/Integrated = %.2f per cent " % gauss_to_integrated)
    
    
        # New 22 Jan 2019: sustract Gaussian fit
        index=0
        s_s=np.zeros_like(s)
        sustract_this = np.zeros_like(gaussian_fit)
        if sus_line1:
            sustract_this = sustract_this + gaussian_1
        if sus_line2:
            sustract_this = sustract_this + gaussian_2    
        
        
        for wave in range(len(wavelength)):
            s_s[wave]=s[wave]
            if wavelength[wave] == w_spec[0] : 
                s_s[wave] = f_spec[0]-sustract_this[0]
                index=1
            if wavelength[wave] > w_spec[0] and wavelength[wave] <= w_spec[-1]:
                s_s[wave] = f_spec[index]-sustract_this[index]
                index=index+1
        if plot_sus:  
            plt.figure(figsize=(10, 4))
            plt.plot(wavelength,s, "r")
            plt.plot(wavelength,s_s, "c")
            plt.minorticks_on() 
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(lmin,lmax)
            plt.ylim(fmin,fmax)
            plt.show()
            plt.close()
    
        # This gaussian_flux in 3  is gaussian 1 + gaussian 2, given in 15, 16, respectively
        #                0      1         2                3               4              5      6         7        8        9     10      11   12       13      14         15                16
        resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, s_s, fit[3], fit[4],fit[5], gaussian_flux_1, gaussian_flux_2 ]
        return resultado 
    except Exception:
        if verbose: print("  Double Gaussian fit failed!")
        resultado = [0, line1, 0, 0, 0, 0, 0, 0, 0, 0, 0, s, 0, 0, 0, 0, 0  ]  # line was identified at lambda=line but Gaussian fit failed

        # NOTA: PUEDE DEVOLVER EL FLUJO INTEGRADO AUNQUE FALLE EL AJUSTE GAUSSIANO...

       # Plotting 
        if plot :
            plt.figure(figsize=(10, 4))
            plt.plot(np.array(w_spec),np.array(f_spec), "b", lw=3, alpha = 0.5)
            plt.minorticks_on() 
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
            else:
                plt.ylabel("Flux [ counts ]")            
            plt.xlim(lmin,lmax)
            plt.ylim(fmin,fmax)
        
            # Vertical line at guess_centre
            plt.axvline(x=guess_centre, color='r', linestyle='-', alpha=0.5)
            # Horizontal line at y = 0
            plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)    
            # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
            plt.axvspan(guess_centre+highlow, guess_centre+highhigh, facecolor='g', alpha=0.15,zorder=3)
            plt.axvspan(guess_centre-lowlow, guess_centre-lowhigh, facecolor='g', alpha=0.15,zorder=3)
            # Plot linear fit for continuum
            plt.plot(w_spec, continuum,"g--")
            # Plot Gaussian fit     
#            plt.plot(w_spec, gaussian_fit+continuum, 'r-', alpha=0.8)    
            # Vertical line at Gaussian center
#            plt.axvline(x=fit[0], color='k', linestyle='-', alpha=0.5)
            # Vertical lines to emission line
            plt.axvline(x= low_limit, color='k', linestyle=':', alpha=0.5)
            plt.axvline(x= high_limit, color='k', linestyle=':', alpha=0.5)  
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
                 check_redshift = 0.0003, only_id_lines=True, plot=True, verbose=True, fig_size=12):
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
    if lmin == 0 :
        lmin = np.nanmin(wavelength)
    if lmax == 0 :
        lmax = np.nanmax(wavelength)
    
    # Fit a smooth continuum
    #smooth_points = 20      # Points in the interval
    step = np.int(len(wavelength)/smooth_points)  # step
    w_cont_smooth = np.zeros(smooth_points)   
    f_cont_smooth = np.zeros(smooth_points)     

    for j in range(smooth_points):
        w_cont_smooth[j] = np.nanmedian([wavelength[i] for i in range(len(wavelength)) if (i > step*j and i<step*(j+1))])
        f_cont_smooth[j] = np.nanmedian([flux[i] for i in range(len(wavelength)) if (i > step*j and i<step*(j+1))])   # / np.nanmedian(spectrum)
        #print j,w_cont_smooth[j], f_cont_smooth[j]

    interpolated_continuum_smooth = interpolate.splrep(w_cont_smooth, f_cont_smooth, s=0)
    interpolated_continuum = interpolate.splev(wavelength, interpolated_continuum_smooth, der=0)


    funcion = flux/interpolated_continuum
        
    # Searching for peaks using cut = 1.2 by default
    peaks = []
    index_low = 0
    for i in range(len(wavelength)):
        if funcion[i] > cut and funcion[i-1] < cut :
            index_low = i
        if funcion[i] < cut and funcion[i-1] > cut :
            index_high = i
            if index_high != 0 :
                pfun = np.nanmax([funcion[j] for j in range(len(wavelength)) if (j > index_low and j<index_high+1 )])
                peak = wavelength[funcion.tolist().index(pfun)]
                if (index_high - index_low) > 1 :
                    peaks.append(peak)
    
    # Identify lines
    # Read file with data of emission lines: 
    # 6300.30 [OI] -0.263    15   5    5    15
    # el_center el_name el_fnl lowlow lowhigh highlow highigh 
    # Only el_center and el_name are needed
    el_center,el_name,el_fnl,el_lowlow,el_lowhigh,el_highlow,el_highhigh = read_table(emission_line_file, ["f", "s", "f", "f", "f", "f", "f"] )
    #for i in range(len(el_name)):
    #    print " %8.2f  %9s  %6.3f   %4.1f %4.1f   %4.1f   %4.1f" % (el_center[i],el_name[i],el_fnl[i],el_lowlow[i], el_lowhigh[i], el_highlow[i], el_highhigh[i])
    #el_center,el_name = read_table("lineas_c89_python.dat", ["f", "s"] )

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
    Ha_redshift = (Ha_w_obs-Ha_w_rest)/Ha_w_rest
    if verbose: print("\n> Detected %i emission lines using %8s at %8.2f A as brightest line!!\n" % (len(peaks),brightest_line, Ha_w_rest))    
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
        minimo_w = np.abs(peaks[i]/(1+Ha_redshift)-el_center)
        if np.nanmin(minimo_w) < 2.5:
           indice = minimo_w.tolist().index(np.nanmin(minimo_w))
           peaks_name[i]=el_name[indice]
           peaks_rest[i]=el_center[indice]
           peaks_redshift[i] = (peaks[i]-el_center[indice])/el_center[indice]
           peaks_lowlow[i] = el_lowlow[indice]
           peaks_lowhigh[i] = el_lowhigh[indice]
           peaks_highlow[i] = el_highlow[indice]
           peaks_highhigh[i] = el_highhigh[indice]
           if verbose: print("%9s %8.2f found in %8.2f at z=%.6f   |z-zref| = %.6f" % (peaks_name[i], peaks_rest[i],peaks[i], peaks_redshift[i],np.abs(peaks_redshift[i]- Ha_redshift) ))
           #print peaks_lowlow[i],peaks_lowhigh[i],peaks_highlow[i],peaks_highhigh[i]
    # Check if all redshifts are similar, assuming check_redshift = 0.0003 by default
    # If OK, add id_peaks[i]=1, if not, id_peaks[i]=0       
    id_peaks=[]
    for i in range(len(peaks_redshift)):
        if np.abs(peaks_redshift[i]-Ha_redshift) > check_redshift:
            if verbose: print("  WARNING!!! Line %8s in w = %.2f has redshift z=%.6f, different than zref=%.6f" %(peaks_name[i],peaks[i],peaks_redshift[i], Ha_redshift))
            id_peaks.append(0)
        else:
            id_peaks.append(1)

    if plot:
        plt.figure(figsize=(fig_size, fig_size/2.5)) 
        plt.plot(wavelength, funcion, "r", lw=1, alpha = 0.5)
        plt.minorticks_on() 
        plt.xlabel("Wavelength [$\mathrm{\AA}$]")
        plt.ylabel("Flux / continuum")
    
        plt.xlim(lmin,lmax)
        plt.ylim(fmin,fmax)
        plt.axhline(y=cut, color='k', linestyle=':', alpha=0.5) 
        for i in range(len(peaks)):
            plt.axvline(x=peaks[i], color='k', linestyle=':', alpha=0.5)
            label=peaks_name[i]
            plt.text(peaks[i], 1.8, label)          
        plt.show()    
    
    continuum_limits = [peaks_lowlow, peaks_lowhigh, peaks_highlow, peaks_highhigh]
       
    if only_id_lines:
        peaks_r=[]
        peaks_name_r=[]
        peaks_rest_r=[]
        peaks_lowlow_r=[]
        peaks_lowhigh_r=[]
        peaks_highlow_r=[]
        peaks_highhigh_r=[]
        
        for i in range(len(peaks)):  
            if id_peaks[i] == 1:
                peaks_r.append(peaks[i])
                peaks_name_r.append(peaks_name[i])
                peaks_rest_r.append(peaks_rest[i])
                peaks_lowlow_r.append(peaks_lowlow[i])
                peaks_lowhigh_r.append(peaks_lowhigh[i])
                peaks_highlow_r.append(peaks_highlow[i])
                peaks_highhigh_r.append(peaks_highhigh[i])
        continuum_limits_r=[peaks_lowlow_r,peaks_lowhigh_r,peaks_highlow_r,peaks_highhigh_r] 

        return peaks_r, peaks_name_r , peaks_rest_r, continuum_limits_r         
    else:      
        return peaks, peaks_name , peaks_rest, continuum_limits  
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fit_smooth_spectrum(wl,x, edgelow=20,edgehigh=20, order= 9, kernel=11, verbose=True, 
                        plot=True, hlines=[1.], ptitle= "", fcal=False):
    """
    Apply f1,f2 = fit_smooth_spectrum(wl,spectrum) and returns:
    
    f1 is the smoothed spectrum, with edges 'fixed'
    f2 is the fit to the smooth spectrum
    """
    
    if verbose: 
        print('\n> Fitting an order {} polynomium to a spectrum smoothed with medfilt window of {}'.format(order,kernel))
        print("  trimming the edges [0:{}] and [{}:{}] ...".format(edgelow,len(wl)-edgehigh, len(wl)))  
    # fit, trimming edges
    index=np.arange(len(x))
    valid_ind=np.where((index >= edgelow) & (index <= len(wl)-edgehigh) & (~np.isnan(x)))[0]
    valid_wl = wl[valid_ind]
    valid_x = x[valid_ind] 
    wlm = signal.medfilt(valid_wl, kernel)
    wx = signal.medfilt(valid_x, kernel) 
    
    #iteratively clip and refit
    maxit=10
    niter=0
    stop=0
    fit_len=100# -100
    resid=0
    while stop < 1:
        #print '  Trying iteration ', niter,"..."
        fit_len_init=copy.deepcopy(fit_len)
        if niter == 0:
            fit_index=np.where(wx == wx)
            fit_len=len(fit_index)
            sigma_resid=0.0
            #print fit_index, fit_len
        if niter > 0:
            sigma_resid=MAD(resid)
            fit_index=np.where(np.abs(resid) < 4*sigma_resid)[0]
            fit_len=len(fit_index)
        try:
            #print " Fitting between ", wlm[fit_index][0],wlm[fit_index][-1]
            p=np.polyfit(wlm[fit_index], wx[fit_index], order)    # It was 2
            pp=np.poly1d(p)
            fx=pp(wl)
            fxm=pp(wlm)
            resid=wx-fxm
            #print niter,wl,fx, fxm
            #print "  Iteration {:2} results in RA: sigma_residual = {:.6f}, fit_len_init = {:5}  fit_len ={:5}".format(niter,sigma_resid,fit_len_init,fit_len)             
        except Exception:  
            if verbose: print('  Skipping iteration ',niter)
        if (niter >= maxit) or (fit_len_init == fit_len): 
            if verbose: 
                if niter >= maxit : print("  Max iterations, {:2}, reached!".format(niter))
                if fit_len_init == fit_len : print("  All interval fitted in iteration {} ! ".format(niter))
            stop=2     
        niter=niter+1

    # Smoothed spectrum, adding the edges
    f_ = signal.medfilt(valid_x, kernel)
    f = np.zeros_like(x)
    f[valid_ind] = f_
    half_kernel = np.int(kernel/2)
    if half_kernel > edgelow:
        f[np.where(index < half_kernel)] = f_[half_kernel-edgelow]
    else:
        f[np.where(index < edgelow)] = f_[0]
    if half_kernel > edgehigh:    
        f[np.where(index > len(wl)-half_kernel)] = f_[-1-half_kernel+edgehigh]
    else:
        f[np.where(index < edgehigh)] = f_[-1]
                    
    if plot:
        ymin = np.nanpercentile(x[edgelow:len(x)-edgehigh],1.2)
        ymax=  np.nanpercentile(x[edgelow:len(x)-edgehigh],99)
        rango = (ymax-ymin)
        ymin = ymin - rango/10.
        ymax = ymax + rango/10.             
        if ptitle == "" : ptitle= "Order "+np.str(order)+" polynomium fitted to a spectrum smoothed with a "+np.str(kernel)+" kernel window"
        plot_plot(wl, [x,f,fx], ymin=ymin, ymax=ymax, color=["red","green","blue"], alpha=[0.2,0.5,0.5], label=["spectrum","smoothed","fit"], ptitle=ptitle, fcal=fcal, vlines=[wl[edgelow],wl[-1-edgehigh]], hlines=hlines)
      
    return f,fx
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def smooth_spectrum(wlm, s, wave_min=0, wave_max=0, step=50, exclude_wlm=[[0,0]], order=7,    
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

    if wave_min == 0 : wave_min = wlm[0]
    if wave_max == 0 : wave_max = wlm[-1]
        
    running_wave = []    
    running_step_median = []
    cuts=np.int( (wave_max - wave_min) /step)
   
    exclude = 0 
    corte_index=-1
    for corte in range(cuts+1):
        next_wave= wave_min+step*corte
        if next_wave < wave_max:
            if next_wave > exclude_wlm[exclude][0] and next_wave < exclude_wlm[exclude][1]:
               if verbose: print("  Skipping ",next_wave, " as it is in the exclusion range [",exclude_wlm[exclude][0],",",exclude_wlm[exclude][1],"]")    

            else:
                corte_index=corte_index+1
                running_wave.append (next_wave)
                region = np.where((wlm > running_wave[corte_index]-step/2) & (wlm < running_wave[corte_index]+step/2))              
                running_step_median.append (np.nanmedian(s[region]) )
                if next_wave > exclude_wlm[exclude][1]:
                    exclude = exclude + 1
                    #if verbose and exclude_wlm[0] != [0,0] : print "--- End exclusion range ",exclude 
                    if exclude == len(exclude_wlm) :  exclude = len(exclude_wlm)-1  
                        
    running_wave.append (wave_max)
    region = np.where((wlm > wave_max-step) & (wlm < wave_max+0.1))
    running_step_median.append (np.nanmedian(s[region]) )
    
    # Check not nan
    _running_wave_=[]
    _running_step_median_=[]
    for i in range(len(running_wave)):
        if np.isnan(running_step_median[i]):
            if verbose: print("  There is a nan in ",running_wave[i])
        else:
            _running_wave_.append (running_wave[i])
            _running_step_median_.append (running_step_median[i])
    
    fit = np.polyfit(_running_wave_, _running_step_median_, order)
    pfit = np.poly1d(fit)
    fit_median = pfit(wlm)
    
    interpolated_continuum_smooth = interpolate.splrep(_running_wave_, _running_step_median_, s=0.02)
    fit_median_interpolated = interpolate.splev(wlm, interpolated_continuum_smooth, der=0)
     
    if plot:       
        plt.figure(figsize=(fig_size, fig_size/2.5)) 
        plt.plot(wlm,s, alpha=0.5)
        plt.plot(running_wave,running_step_median, "+", ms=15, mew=3)
        plt.plot(wlm, fit_median, label="fit median")
        plt.plot(wlm, fit_median_interpolated, label="fit median_interp")
        plt.plot(wlm, weight_fit_median*fit_median + (1-weight_fit_median)*fit_median_interpolated, label="weighted")
        #extra_display = (np.nanmax(fit_median)-np.nanmin(fit_median)) / 10
        #plt.ylim(np.nanmin(fit_median)-extra_display, np.nanmax(fit_median)+extra_display)
        ymin = np.nanpercentile(s,1)
        ymax=  np.nanpercentile(s,99)
        rango = (ymax-ymin)
        ymin = ymin - rango/10.
        ymax = ymax + rango/10. 
        plt.ylim(ymin,ymax)
        plt.xlim(wlm[0]-10, wlm[-1]+10)
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
        print('  Weights for getting smooth spectrum:  fit_median =',weight_fit_median,'    fit_median_interpolated =',(1-weight_fit_median))

    return weight_fit_median*fit_median + (1-weight_fit_median)*fit_median_interpolated #   (fit_median+fit_median_interpolated)/2      # Decide if fit_median or fit_median_interpolated
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def obtain_sky_spectrum(sky, low_fibres=200, plot=True, fig_size=12, fcal=False, verbose=True):     
    """
    This uses the lowest low_fibres fibres to get an integrated spectrum
    """
    integrated_intensity_sorted=np.argsort(sky.integrated_fibre)  
    region=[]
    for fibre in range(low_fibres):
        region.append(integrated_intensity_sorted[fibre])
    sky_spectrum=np.nanmedian(sky.intensity_corrected[region], axis=0)
    
    if verbose: 
        print("  We use the ",low_fibres," fibres with the lowest integrated intensity to derive the sky spectrum")
        print("  The list is = ",region)
    
    if plot:
        plt.figure(figsize=(fig_size, fig_size/2.5))
        plt.plot(sky.wavelength,sky_spectrum)
        ptitle="Sky spectrum"        
        plot_plot(sky.wavelength,sky_spectrum,  ptitle=ptitle, fcal=fcal)

    return sky_spectrum
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def median_filter(intensity_corrected, n_spectra, n_wave, win_sky=151):
    """
    Matt's code to get a median filter of all fibres in a RSS
    This is useful when having 2D sky
    """
    
    medfilt_sky=np.zeros((n_spectra,n_wave))
    for wave in range(n_wave):
        medfilt_sky[:,wave]=sig.medfilt(intensity_corrected[:,wave],kernel_size=win_sky)
        
    #replace crappy edge fibres with 0.5*win'th medsky
    for fibre_sky in range(n_spectra):
        if fibre_sky < np.rint(0.5*win_sky):
            j=int(np.rint(0.5*win_sky))
            medfilt_sky[fibre_sky,]=copy.deepcopy(medfilt_sky[j,])
        if fibre_sky > n_spectra - np.rint(0.5*win_sky):
            j=int(np.rint(n_spectra - np.rint(0.5*win_sky)))
            medfilt_sky[fibre_sky,]=copy.deepcopy(medfilt_sky[j,]) 
    return medfilt_sky
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def scale_sky_spectrum(wlm, sky_spectrum, spectra, cut_sky=4., fmax=10, fmin=1, valid_wave_min=0, valid_wave_max=0, 
                       fibre_list=[100,200,300,400,500,600,700,800,900], plot=True, verbose=True, warnings=True):
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
        
    if verbose: print("\n> Identifying sky lines using cut_sky =",cut_sky,", allowed SKY/OBJ values = [",fmin,",",fmax,"]")
    if verbose: print("  Using fibres = ",fibre_list)

    peaks,peaks_name,peaks_rest,continuum_limits=search_peaks(wlm,sky_spectrum, plot=plot, cut=cut_sky, fmax=fmax, only_id_lines=False, verbose=False)   

    ratio_list=[]
    valid_peaks=[]
        
    if verbose: print("\n      Sky line     Gaussian ratio      Flux ratio")
    n_sky_lines_found=0
    for i in range(len(peaks)):
        sky_spectrum_data=fluxes(wlm,sky_spectrum, peaks[i], fcal=False, lowlow=50,highhigh=50, plot=False, verbose=False, warnings=False)
 
        sky_median_continuum = np.nanmedian(sky_spectrum_data[11])
               
        object_spectrum_data_gauss=[]
        object_spectrum_data_integrated=[] 
        median_list=[]
        for fibre in fibre_list:   
            object_spectrum_flux=fluxes(wlm, spectra[fibre], peaks[i], fcal=False, lowlow=50,highhigh=50, plot=False, verbose=False, warnings=False)
            object_spectrum_data_gauss.append(object_spectrum_flux[3])       # Gaussian flux is 3
            object_spectrum_data_integrated.append(object_spectrum_flux[7])  # integrated flux is 7
            median_list.append(np.nanmedian(object_spectrum_flux[11]))
        object_spectrum_data=np.nanmedian(object_spectrum_data_gauss)
        object_spectrum_data_i=np.nanmedian(object_spectrum_data_integrated)
        
        object_median_continuum=np.nanmin(median_list)     
        
        if fmin < object_spectrum_data/sky_spectrum_data[3] *  sky_median_continuum/object_median_continuum    < fmax :
            n_sky_lines_found = n_sky_lines_found + 1
            valid_peaks.append(peaks[i])
            ratio_list.append(object_spectrum_data/sky_spectrum_data[3])
            if verbose: print("{:3.0f}   {:5.3f}         {:2.3f}             {:2.3f}".format(n_sky_lines_found,peaks[i],object_spectrum_data/sky_spectrum_data[3], object_spectrum_data_i/sky_spectrum_data[7]))  


    #print "ratio_list =", ratio_list
    #fit = np.polyfit(valid_peaks, ratio_list, 0) # This is the same that doing an average/mean
    #fit_line = fit[0]+0*wlm
    fit_line =np.nanmedian(ratio_list)  # We just do a median
    #fit_line = fit[1]+fit[0]*wlm
    #fit_line = fit[2]+fit[1]*wlm+fit[0]*wlm**2
    #fit_line = fit[3]+fit[2]*wlm+fit[1]*wlm**2+fit[0]*wlm**3
   
    
    if plot:
        plt.plot(valid_peaks,ratio_list,"+")
        #plt.plot(wlm,fit_line)
        plt.axhline(y=fit_line, color='k', linestyle='--')
        plt.xlim(valid_wave_min-10, valid_wave_max+10)      
        #if len(ratio_list) > 0:
        plt.ylim(np.nanmin(ratio_list)-0.2,np.nanmax(ratio_list)+0.2)
        plt.title("Scaling sky spectrum to object spectra")
        plt.xlabel("Wavelength [$\mathrm{\AA}$]")
        plt.ylabel("OBJECT / SKY")
        plt.minorticks_on()
        plt.show()
        plt.close()
        
        if verbose: print("  Using this fit to scale sky spectrum to object, the median value is ",np.round(fit_line,3),"...")      
    
    sky_corrected = sky_spectrum  * fit_line

#        plt.plot(wlm,sky_spectrum, "r", alpha=0.3)
#        plt.plot(wlm,sky_corrected, "g", alpha=0.3)
#        plt.show()
#        plt.close()
    
    return sky_corrected, np.round(fit_line,3)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def sky_spectrum_from_fibres(rss, list_spectra, win_sky=0, wave_to_fit=300, fit_order = 2, include_history= True,
                             xmin="", xmax="", ymin="", ymax="", verbose = True, plot= True): 
    
    if verbose: 
        print("\n> Obtaining 1D sky spectrum using the rss file and fibre list = ")
        print("  ",list_spectra)
        
    _rss_ = copy.deepcopy(rss)
    w = _rss_.wavelength

    if win_sky > 0:
        if verbose: print("  after applying a median filter with kernel ",win_sky,"...")        
        _rss_.intensity_corrected = median_filter(_rss_.intensity_corrected, _rss_.n_spectra, _rss_.n_wave, win_sky=win_sky)    
    sky = _rss_.plot_combined_spectrum(list_spectra=list_spectra, median=True, plot=plot)
         
    # Find the last good pixel in sky
    last_good_pixel_sky =_rss_.n_wave-1
    found=0
    while found < 1:
        if sky[last_good_pixel_sky] > 0:
            found = 2
        else:
            last_good_pixel_sky = last_good_pixel_sky -1
            if last_good_pixel_sky == _rss_.mask_good_index_range[1]:
                if verbose: print(" WARNING ! last_good_pixel_sky is the same than in file")
                found=2
    
    if verbose: print("\n - Using a 2-order fit to the valid red end of sky spectrum to extend continuum to all wavelengths")
    
    if rss.grating == "385R": 
        wave_to_fit = 200
        fit_order = 1
        
    lmin=_rss_.mask_good_wavelength_range[1]-wave_to_fit   # GAFAS
    w_spec = []
    f_spec = []
    w_spec.extend((w[i]) for i in range(len(w)) if (w[i] > lmin) and (w[i] < _rss_.mask_good_wavelength_range[1]) )  
    f_spec.extend((sky[i]) for i in range(len(w)) if (w[i] > lmin) and (w[i] < _rss_.mask_good_wavelength_range[1]))  
    

    fit=np.polyfit(w_spec, f_spec,fit_order)
            # if fit_order == 2: 
            #     ptitle="Fitting to skyline "+np.str(sky_line)+" : {:.3e} x$^2$  +  {:.3e} x  +  {:.3e} ".format(fit[0],fit[1],fit[2])+text
            # if fit_order == 1:
            #     ptitle="Fitting to skyline "+np.str(sky_line)+" : {:.3e} x  +  {:.3e} ".format(fit[0],fit[1])+text
            # if fit_order > 2:
            #     ptitle="Fitting an order "+np.str(fit_order)+" polinomium to skyline "+np.str(sky_line)+text
            
    y=np.poly1d(fit)
    y_fitted_all = y(w)
    
    if plot:            
        plot_plot(w,[sky,y_fitted_all], xmin=lmin,percentile_min=0,percentile_max=100, 
                  ptitle="Extrapolating the sky continuum for the red edge",
                  vlines=[_rss_.mask_good_wavelength_range[1], w[last_good_pixel_sky-3]])
    
    sky[last_good_pixel_sky-3:-1] = y_fitted_all[last_good_pixel_sky-3:-1]
     
    
    if include_history: rss.history.append("  Mask used to get a rough sky in the pixels of the red edge")           
    return sky    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def sky_spectrum_from_fibres_using_file(rss_file, fibre_list=[], win_sky=151, n_sky=0,
                                        skyflat="",  apply_throughput=True, correct_ccd_defects= False,
                                        fix_wavelengths = False, sol = [0,0,0], xmin=0, xmax=0, ymin=0, ymax=0, verbose = True, plot= True):
       
    if skyflat == "":
        apply_throughput = False
        plot_rss = False
    else:
        apply_throughput = True
        plot_rss = True
    
        
    if n_sky != 0:
        sky_method="self"
        is_sky=False
        if verbose: print("\n> Obtaining 1D sky spectrum using ",n_sky," lowest fibres in this rss ...")
    else:
        sky_method="none"
        is_sky=True
        if verbose: print("\n> Obtaining 1D sky spectrum using fibre list = ",fibre_list," ...")
        
    
    _test_rss_ = KOALA_RSS(rss_file, apply_throughput=apply_throughput, skyflat = skyflat, correct_ccd_defects = correct_ccd_defects,
                           fix_wavelengths = fix_wavelengths, sol = sol,
                           sky_method=sky_method, n_sky=n_sky, is_sky=is_sky, win_sky=win_sky, 
                           do_extinction=False, plot=plot_rss, verbose = False)
    
    if n_sky != 0:
        print("\n> Sky fibres used: ",  _test_rss_.sky_fibres)
        sky = _test_rss_.sky_emission
    else:    
        sky = _test_rss_.plot_combined_spectrum(list_spectra=fibre_list, median=True)
    
    if plot:        
        plt.figure(figsize=(14, 4))
        if n_sky != 0:
            plt.plot(_test_rss_.wavelength,sky, "b", linewidth=2, alpha=0.5)
            ptitle = "Sky spectrum combining using "+np.str(n_sky)+" lowest fibres"
            
        else:    
            for i in range(len(fibre_list)):
                plt.plot(_test_rss_.wavelength, _test_rss_.intensity_corrected[i], alpha=0.5)
                plt.plot(_test_rss_.wavelength,sky, "b", linewidth=2, alpha=0.5) 
            ptitle = "Sky spectrum combining "+np.str(len(fibre_list))+" fibres"
                
        plot_plot(_test_rss_.wavelength,sky, ptitle=ptitle)
    
    print("\n> Sky spectrum obtained!")    
    return sky   
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def ds9_offsets(x1,y1,x2,y2,pixel_size_arc=0.6):

    delta_x = x2-x1
    delta_y = y2-y1
    
    print("\n> Offsets in pixels : ",delta_x,delta_y) 
    print("  Offsets in arcsec : ",pixel_size_arc*delta_x , pixel_size_arc*delta_y) 
    offset_RA = np.abs(pixel_size_arc*delta_x)
    if delta_x < 0:
        direction_RA = "W"
    else:
        direction_RA = "E"
    offset_DEC = np.abs(pixel_size_arc*delta_y)
    if delta_y < 0:
        direction_DEC = "N"
    else:
        direction_DEC = "S"
    print("  Assuming N up and E left, the telescope did an offset of ----> {:5.2f} {:1} {:5.2f} {:1}".format(offset_RA,direction_RA,offset_DEC,direction_DEC))   
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def offset_positions(ra1h,ra1m,ra1s,dec1d,dec1m,dec1s, ra2h,ra2m,ra2s,dec2d,dec2m,dec2s, decimals=2):
  """
  CHECK THE GOOD ONE in offset_positions.py !!!
  """
  
  ra1=ra1h+ra1m/60.+ ra1s/3600.
  ra2=ra2h+ra2m/60.+ ra2s/3600.

  if dec1d < 0:        
      dec1=dec1d-dec1m/60.- dec1s/3600.
  else:
      dec1=dec1d+dec1m/60.+ dec1s/3600
  if dec2d < 0:                
      dec2=dec2d-dec2m/60.- dec2s/3600.
  else:
      dec2=dec2d+dec2m/60.+ dec2s/3600.
  
  avdec = (dec1+dec2)/2
  
  deltadec=round(3600.*(dec2-dec1), decimals)
  deltara =round(15*3600.*(ra2-ra1)*(np.cos(np.radians(avdec))) ,decimals)

  tdeltadec=np.fabs(deltadec)
  tdeltara=np.fabs(deltara)
    
  if deltadec < 0:
      t_sign_deltadec="South"
      t_sign_deltadec_invert="North"

  else:
      t_sign_deltadec="North"
      t_sign_deltadec_invert="South"

  if deltara < 0:
      t_sign_deltara="West"
      t_sign_deltara_invert="East"

  else:
      t_sign_deltara="East"
      t_sign_deltara_invert="West"         


  print("\n> POS1: RA = {:3}h {:2}min {:2.4f}sec, DEC = {:3}d {:2}m {:2.4f}s".format(ra1h,ra1m,ra1s,dec1d,dec1m,dec1s))
  print("  POS2: RA = {:3}h {:2}min {:2.4f}sec, DEC = {:3}d {:2}m {:2.4f}s".format(ra2h,ra2m,ra2s,dec2d,dec2m,dec2s))

  print("\n> Offset 1 -> 2 : ",tdeltara,t_sign_deltara,"     ", tdeltadec,t_sign_deltadec)
  print("  Offset 2 -> 1 : ",tdeltara,t_sign_deltara_invert,"     ", tdeltadec,t_sign_deltadec_invert)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def basic_statistics(y, x="", xmin="", xmax="", return_data=False, verbose = True):
    """
    Provides basic statistics: min, median, max, std, rms, and snr"
    """    
    if len(x) == 0:
        y_ = y
    else:          
        y_ = []      
        if xmin == "" : xmin = x[0]
        if xmax == "" : xmax = x[-1]          
        y_.extend((y[j]) for j in range(len(x)) if (x[j] > xmin and x[j] < xmax) )  
     
    median_value = np.nanmedian(y_)
    min_value = np.nanmin(y_)
    max_value = np.nanmax(y_)
    
    n_ = len(y_)
    #mean_ = np.sum(y_) / n_
    mean_ = np.nanmean(y_)
    #var_ = np.sum((item - mean_)**2 for item in y_) / (n_ - 1)  
    var_ = np.nanvar(y_)

    std = np.sqrt(var_)
    ave_ = np.nanmean(y_)
    disp_ =  max_value - min_value
    
    rms_v = ((y_ - mean_) / disp_ ) **2
    rms = disp_ * np.sqrt(np.nansum(rms_v)/ (n_-1))
    snr = ave_ / rms
    
    if verbose:
        print("  min_value  = {}, median value = {}, max_value = {}".format(min_value,median_value,max_value))
        print("  standard deviation = {}, rms = {}, snr = {}".format(std, rms, snr))   #TIGRE
    
    if return_data : return min_value,median_value,max_value,std, rms, snr
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_plot(x, y,  xmin="",xmax="",ymin="",ymax="",percentile_min=2, percentile_max=98,
              ptitle="Pretty plot", xlabel="Wavelength [$\mathrm{\AA}$]", ylabel="", fcal="", 
              psym="", color="blue", alpha="", linewidth=1,  linestyle="-", markersize = 10,
              vlines=[], hlines=[], chlines=[], axvspan=[[0,0]], hwidth =1, vwidth =1,
              frameon = False, loc = 0, ncol = 5, label="",  text=[],
              title_fontsize=12, label_axes_fontsize=10, axes_fontsize=10, tick_size=[5,1,2,1], axes_thickness =0,
              save_file="", path="", fig_size=12, warnings = True, show=True, statistics=""):
    
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
       fig_size=20
       label_axes_fontsize=20 
       axes_fontsize=15
       title_fontsize=22
       tick_size=[10,1,5,1]
       axes_thickness =3
       hwidth =2
       vwidth =2

   if fig_size in ["very_big", "verybig", "vbig"]:
       fig_size=35
       label_axes_fontsize=30 
       axes_fontsize=25
       title_fontsize=28 
       tick_size=[15,2,8,2]
       axes_thickness =3
       hwidth =4 
       vwidth =4

   if fig_size != 0 : plt.figure(figsize=(fig_size, fig_size/2.5))
   
   if np.isscalar(x[0]) :
       xx=[]
       for i in range(len(y)):
           xx.append(x)
   else:
       xx=x

   if xmin == "" : xmin = np.nanmin(xx[0])
   if xmax == "" : xmax = np.nanmax(xx[0])  
   
   alpha_=alpha
   psym_=psym
   label_=label
   linewidth_=linewidth
   markersize_=markersize
   linestyle_=linestyle

   n_plots=len(y)
       
   if np.isscalar(y[0]) ==  False:
       if np.isscalar(alpha):        
           if alpha_ == "":
               alpha =[0.5]*n_plots
           else:
               alpha=[alpha_]*n_plots
       if np.isscalar(psym): psym=[psym_]*n_plots
       if np.isscalar(label): label=[label_]*n_plots
       if np.isscalar(linewidth): linewidth=[linewidth_]*n_plots
       if np.isscalar(markersize):markersize=[markersize_]*n_plots
       if np.isscalar(linestyle): linestyle=[linestyle_]*n_plots
       if color == "blue" : color = ["red","blue","green","k","orange", "purple", "cyan", "lime"]
       if ymax == "": y_max_list = []
       if ymin == "": y_min_list = []
              
       if fcal == "":
           if np.nanmedian(np.abs(y[0])) < 1E-10:
               fcal = True
               if np.nanmedian(y[0]) < 1E-20 and np.var(y[0]) > 0 : fcal = False
       for i in range(len(y)):
           if psym[i] == "":
               plt.plot(xx[i],y[i], color=color[i], alpha=alpha[i], label=label[i], linewidth=linewidth[i], linestyle=linestyle[i])
           else:
               plt.plot(xx[i],y[i], psym[i], color=color[i], alpha=alpha[i], label=label[i], mew=linewidth[i], markersize=markersize[i])
           if ymax == "":
                    y_max_ = []                
                    y_max_.extend((y[i][j]) for j in range(len(xx[i])) if (xx[i][j] > xmin and xx[i][j] < xmax) )  
                    y_max_list.append(np.nanpercentile(y_max_, percentile_max))
           if ymin == "":
                    y_min_ = []                
                    y_min_.extend((y[i][j]) for j in range(len(xx[i])) if (xx[i][j] > xmin and xx[i][j] < xmax) )  
                    y_min_list.append(np.nanpercentile(y_min_, percentile_min))
       if ymax == "":
           ymax = np.nanmax(y_max_list)
       if ymin == "":
           ymin = np.nanmin(y_min_list)
   else:
       if alpha == "": alpha=1
       if statistics =="": statistics = True
       if fcal == "":
           if np.nanmedian(np.abs(y)) < 1E-10 : 
               fcal= True 
               if np.nanmedian(np.abs(y)) < 1E-20 and np.nanvar(np.abs(y)) > 0 : fcal = False
       if psym == "":
             plt.plot(x,y, color=color, alpha=alpha,linewidth=linewidth,  linestyle=linestyle)
       else:
           plt.plot(x,y, psym, color=color, alpha=alpha, mew=linewidth, markersize=markersize)
       if ymin == "" :
           y_min_ = []                
           y_min_.extend((y[j]) for j in range(len(x)) if (x[j] > xmin and x[j] < xmax) )  
           ymin = np.nanpercentile(y_min_, percentile_min)
       if ymax == "" :
           y_max_ = []                
           y_max_.extend((y[j]) for j in range(len(x)) if (x[j] > xmin and x[j] < xmax) )  
           ymax = np.nanpercentile(y_max_, percentile_max)    
       
   plt.xlim(xmin,xmax)                    
   plt.ylim(ymin,ymax)
   try:   
       plt.title(ptitle, fontsize=title_fontsize)
   except Exception:
       if warnings : print("  WARNING: Something failed when including the title of the plot")
   
   plt.minorticks_on()
   plt.xlabel(xlabel, fontsize=label_axes_fontsize)
   #plt.xticks(rotation=90)
   plt.tick_params('both', length=tick_size[0], width=tick_size[1], which='major')
   plt.tick_params('both', length=tick_size[2], width=tick_size[3], which='minor')
   plt.tick_params(labelsize=axes_fontsize)
   plt.axhline(y=ymin,linewidth=axes_thickness, color="k")     # These 4 are for making the axes thicker, it works but it is not ideal...
   plt.axvline(x=xmin,linewidth=axes_thickness, color="k")    
   plt.axhline(y=ymax,linewidth=axes_thickness, color="k")    
   plt.axvline(x=xmax,linewidth=axes_thickness, color="k")    
   
   if ylabel=="": 
       if fcal: 
           ylabel = "Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]"
       else:         
           ylabel = "Flux [counts]"  
 
   plt.ylabel(ylabel, fontsize=label_axes_fontsize)
   
   if len(chlines) != len(hlines):
       for i in range(len(hlines)-len(chlines)):
           chlines.append("k")  
           
   for i in range(len(hlines)):
       if chlines[i] != "k":
           hlinestyle="-"
           halpha=0.8
       else:
           hlinestyle="--" 
           halpha=0.3
       plt.axhline(y=hlines[i], color=chlines[i], linestyle=hlinestyle, alpha=halpha, linewidth=hwidth)
   for i in range(len(vlines)):
       plt.axvline(x=vlines[i], color="k", linestyle="--", alpha=0.3, linewidth=vwidth)
    
   if label_ != "" : 
       plt.legend(frameon=frameon, loc=loc, ncol=ncol)
       
   if axvspan[0][0] != 0:
       for i in range(len(axvspan)):
           plt.axvspan(axvspan[i][0], axvspan[i][1], facecolor='orange', alpha=0.15, zorder=3)   
           
   if len(text)  > 0:
        for i in range(len(text)):
            plt.text(text[i][0],text[i][1], text[i][2], size=axes_fontsize)
                    
   if save_file == "":
       if show: 
           plt.show()
           plt.close() 
   else:
       if path != "" : save_file=full_path(save_file,path)
       plt.savefig(save_file)
       plt.close() 
       print("  Figure saved in file",save_file)
   
   if statistics == "": statistics=False
   if statistics:
       if np.isscalar(y[0]) : 
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
    SIN=np.sin(x*np.pi/180)
    return SIN
def COS(x):
    COS=np.cos(x*np.pi/180)
    return COS
def TAN(x):
    TAN=np.tan(x*np.pi/180)
    return TAN
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------- 
def rebin_spec(wave, specin, wavnew):
    #spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    spec = SourceSpectrum(Empirical1D, points=wave, lookup_table=specin, keep_neg=True)
    f = np.ones(len(wave))
    #filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
    filt=SpectralElement(Empirical1D, points=wave, lookup_table=f, keep_neg=True)      # LOKI
    obs = observation.Observation(spec, filt, binset=wavnew, force='taper') 
    return obs.binflux.value
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def rebin_spec_shift(wave, specin, shift):
    wavnew=wave+shift
    rebined = rebin_spec(wave, specin, wavnew)
    return rebined
    
    ##spec = spectrum.ArraySourceSpectrum(wave=wave, flux=specin)
    #spec = SourceSpectrum(Empirical1D, points=wave, lookup_table=specin, keep_neg=True)
    #f = np.ones(len(wave))
    #filt = spectrum.ArraySpectralElement(wave, f, waveunits='angstrom')
    #filt=SpectralElement(Empirical1D, points=wave, lookup_table=f, keep_neg=True) 
    #obs = observation.Observation(spec, filt, binset=wavnew, force='taper') 
    #obs = observation.Observation(spec, filt, binset=wavnew, force='taper') 
    #return obs.binflux.value
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
def save_nresponse(nflat, filename, mask=[[-1]], no_nans=True, flappyflat = False):
    """
    For masks nflat has to be a rss used to create the mask. 
    no_nans = True for mask having only 1s and 0s (otherwise 1s and nans)
    """
    
    if mask[0][0] != -1:
        fits_image_hdu = fits.PrimaryHDU(mask)
    else:
        fits_image_hdu = fits.PrimaryHDU(nflat.nresponse)

    fits_image_hdu.header["ORIGIN"]  = 'AAO'    #    / Originating Institution                        
    fits_image_hdu.header["TELESCOP"]= 'Anglo-Australian Telescope'    # / Telescope Name  
    fits_image_hdu.header["ALT_OBS"] =                 1164 # / Altitude of observatory in metres              
    fits_image_hdu.header["LAT_OBS"] =            -31.27704 # / Observatory latitude in degrees                
    fits_image_hdu.header["LONG_OBS"]=             149.0661 # / Observatory longitude in degrees 

    fits_image_hdu.header["INSTRUME"] = "AAOMEGA-KOALA"             # / Instrument in use  
    fits_image_hdu.header["GRATID"]  = nflat.grating      # / Disperser ID 
    if nflat.grating in red_gratings : SPECTID="RD"
    if nflat.grating in blue_gratings : SPECTID="BL"
    fits_image_hdu.header["SPECTID"] = SPECTID                        # / Spectrograph ID                                
    fits_image_hdu.header["DICHROIC"]= 'X5700'                        # / Dichroic name   ---> CHANGE if using X6700!!
    
    fits_image_hdu.header['OBJECT'] = "Normalized skyflat response"    
    #fits_image_hdu.header["TOTALEXP"] = combined_cube.total_exptime
                                       
    fits_image_hdu.header['NAXIS']   =   2                              # / number of array dimensions                       
    fits_image_hdu.header['NAXIS1']  =   nflat.intensity.shape[0]        ##### CHECK !!!!!!!           
    fits_image_hdu.header['NAXIS2']  =   nflat.intensity.shape[1]                                     

    # WCS
    fits_image_hdu.header["RADECSYS"]= 'FK5'          # / FK5 reference system   
    fits_image_hdu.header["EQUINOX"] = 2000           # / [yr] Equinox of equatorial coordinates                         
    fits_image_hdu.header["WCSAXES"] =  2             # / Number of coordinate axes                      
    fits_image_hdu.header["CRVAL2"] = 5.000000000000E-01 # / Co-ordinate value of axis 2  
    fits_image_hdu.header["CDELT2"] = 1.000000000000E+00 # / Co-ordinate increment along axis 2
    fits_image_hdu.header["CRPIX2"] = 1.000000000000E+00 # / Reference pixel along axis 2 

    # Wavelength calibration
    fits_image_hdu.header["CTYPE1"] = 'Wavelength'          # / Label for axis 3  
    fits_image_hdu.header["CUNIT1"] = 'Angstroms'           # / Units for axis 3     
    fits_image_hdu.header["CRVAL1"] = nflat.CRVAL1_CDELT1_CRPIX1[0] # 7.692370611909E+03  / Co-ordinate value of axis 3
    fits_image_hdu.header["CDELT1"] = nflat.CRVAL1_CDELT1_CRPIX1[1] # 1.575182431607E+00 
    fits_image_hdu.header["CRPIX1"] = nflat.CRVAL1_CDELT1_CRPIX1[2] # 1024. / Reference pixel along axis 3

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
            
    fits_image_hdu.header['HISTORY'] = "using PyKOALA "+version #'Version 0.10 - 12th February 2019'    
    now=datetime.datetime.now()
    fits_image_hdu.header['HISTORY'] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    if mask[0][0] == -1:
        if flappyflat:
            fits_image_hdu.header['HISTORY'] = "Created processing flappy flat filename:"
        else:
            fits_image_hdu.header['HISTORY'] = "Created processing skyflat filename:"
        fits_image_hdu.header['HISTORY'] = nflat.filename
    fits_image_hdu.header['DATE'] = now.strftime("%Y-%m-%dT%H:%M:%S") #'2002-09-16T18:52:44'   # /Date of FITS file creation
    
    fits_image_hdu.header['BITPIX']  =  16  

    hdu_list = fits.HDUList([fits_image_hdu]) 

    hdu_list.writeto(filename, overwrite=True) 
    
    if mask[0][0] != -1:
        print("\n> Mask saved in file:")
    else:
        if flappyflat:
            print("\n> Normalized flappy flat (nresponse) saved in file:")
        else:
            print("\n> Wavelength dependence of the throughput saved in file:")
    print(" ",filename)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def nresponse_flappyflat(file_f, flappyflat="", nresponse_file="", 
                         correct_ccd_defects = True,
                         kernel=51, ymin = 0.75, ymax = 1.25, plot_fibres=[], plot=True):
                         #order=13,  edgelow=20, edgehigh=20, 
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
        flappyflat =  KOALA_RSS(file_f, 
                             apply_throughput=False, 
                             correct_ccd_defects = correct_ccd_defects, 
                             remove_5577 = False,
                             do_extinction=False,
                             sky_method="none",
                             correct_negative_sky = False,
                             plot=plot, warnings=False)    
    else:
        print("\n> Flappy flat already read")            
      
    #print "\n> Performing a {} polynomium fit to smoothed spectrum with window {} to all fibres...\n".format(order,kernel)      
    print("\n> Applying median filter with window {} to all fibres to get nresponse...\n".format(kernel))      
    
    if plot_fibres == []: plot_fibres=[0, 200, 500,501, 700, flappyflat.n_spectra-1]   
    nresponse=np.zeros_like(flappyflat.intensity_corrected)
    for i in range(flappyflat.n_spectra):

        spectrum_ = flappyflat.intensity_corrected[i] #[np.nan if x == 0 else x for x in flappyflat.intensity_corrected[i]]              
        nresponse_ = signal.medfilt(spectrum_, kernel)
        nresponse[i] = [0 if x < ymin or x > ymax else x for x in nresponse_]
        nresponse[i] = [0 if np.isnan(x) else x for x in nresponse_]
        
        if i in plot_fibres:            
            ptitle = "nresponse for fibre "+np.str(i)
            plot_plot(flappyflat.wavelength,[spectrum_,nresponse[i]],hlines=np.arange(ymin+0.05,ymax,0.05),
                      label=["spectrum","medfilt"],ymin=ymin,ymax=ymax, ptitle=ptitle)

    if plot: flappyflat.RSS_image(image=nresponse, cmap=fuego_color_map)
                
    flappyflat.nresponse = nresponse
    
    print("\n> Normalized flatfield response stored in self.nresponse !!")
    
    if nresponse_file != "":    
        print("  Also saving the obtained nresponse to file")
        print(" ",nresponse_file)
        save_nresponse(flappyflat, filename=nresponse_file, flappyflat=True)
    return nresponse
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_throughput_2D(file_skyflat, throughput_2D_file = "", plot = True, also_return_skyflat=True,
                      correct_ccd_defects = True, fix_wavelengths=False, sol=[0], kernel_throughput=0):
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

    if sol[0] == 0 or sol[0] == -1 :
        if fix_wavelengths:
            print("  Fix wavelength requested but not solution given, ignoring it...")
            fix_wavelengths = False
    #else:
    #    if len(sol) == 3 : fix_wavelengths = True

    skyflat=KOALA_RSS(file_skyflat, correct_ccd_defects = correct_ccd_defects, 
                      fix_wavelengths=fix_wavelengths, sol= sol, plot= plot)
    
    skyflat.apply_mask(make_nans = True)
    throughput_2D_ = np.zeros_like(skyflat.intensity_corrected)
    print("\n> Getting the throughput per wavelength...")
    for i in range(skyflat.n_wave):
        column = skyflat.intensity_corrected[:,i]
        mcolumn = column / np.nanmedian(column)
        throughput_2D_[:,i] =   mcolumn
        
    if kernel_throughput > 0 :
        print("\n  - Applying smooth with kernel =",kernel_throughput)
        throughput_2D = np.zeros_like(throughput_2D_)
        for i in range(skyflat.n_spectra):
            throughput_2D[i] = signal.medfilt(throughput_2D_[i],kernel_throughput)
        skyflat.RSS_image(image=throughput_2D,chigh=1.1, clow=0.9, cmap="binary_r")
        skyflat.history.append('- Throughput 2D smoothed with kernel '+np.str(kernel_throughput))
    else:
        throughput_2D = throughput_2D_
    
    
    skyflat.sol = sol    
    # Saving the information of fix_wavelengths in throughput_2D[0][0]
    if sol[0] != 0:
        print("\n  - The solution for fixing wavelengths has been provided") 
        if sol[0] != -1:
            throughput_2D[0][0] = 1.0  # if throughput_2D[0][0] is 1.0, the throughput has been corrected for small wavelength variations
            skyflat.history.append('- Written data[0][0] = 1.0 for automatically identifing')
            skyflat.history.append('  that the throughput 2D data has been obtained')
            skyflat.history.append('  AFTER correcting for small wavelength variations')
        
    if plot: 
        x = np.arange(skyflat.n_spectra)        
        median_throughput = np.nanmedian(throughput_2D, axis=1)
        plot_plot(x,median_throughput, ymin=0.2,ymax=1.2, hlines=[1,0.9,1.1], 
                  ptitle="Median value of the 2D throughput per fibre", xlabel="Fibre")       
        skyflat.RSS_image(image=throughput_2D, cmap ="binary_r",
                          title = "\n ---- 2D throughput ----")

    skyflat_corrected = skyflat.intensity_corrected / throughput_2D
    if plot: skyflat.RSS_image(image=skyflat_corrected, title = "\n Skyflat CORRECTED for 2D throughput")
    if throughput_2D_file != "" :
        save_rss_fits(skyflat, data=throughput_2D, fits_file=throughput_2D_file, text="Throughput 2D ", sol=sol)
        
    print("\n> Throughput 2D obtained!")  
    if also_return_skyflat:
        return throughput_2D,skyflat
    else:    
        return throughput_2D
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_continuum_in_range(w,s,low_low, low_high, high_low, high_high,
                           pmin=12,pmax=88, only_correct_negative_values = False,
                           fit_degree=2, plot = True, verbose = True, warnings=True)  :
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
        if np.nanpercentile(y_fit_low,2) >  y_fit_low[i] or y_fit_low[i] > np.nanpercentile(y_fit_low,98): y_fit_low[i] =median_y_fit_low

    median_y_fit_high = np.nanmedian(y_fit_high)
    for i in range(len(y_fit_high)):
        if np.nanpercentile(y_fit_high,2) >  y_fit_high[i] or y_fit_high[i] > np.nanpercentile(y_fit_high,98): y_fit_high[i] =median_y_fit_high
            
    w_fit_cont = np.concatenate((w_fit_low,w_fit_high))
    y_fit_cont = np.concatenate((y_fit_low,y_fit_high))
        
    try:
        fit = np.polyfit(w_fit_cont,y_fit_cont, fit_degree)
        yfit = np.poly1d(fit)
        y_fitted = yfit(w_fit)
        
        y_fitted_low = yfit(w_fit_low)
        median_low = np.nanmedian(y_fit_low-y_fitted_low)
        rms=[]
        for i in range(len(y_fit_low)):
            rms.append(y_fit_low[i]-y_fitted_low[i]-median_low)
        
    #    rms=y_fit-y_fitted
        lowlimit=np.nanpercentile(rms,pmin)
        highlimit=np.nanpercentile(rms,pmax)
            
        corrected_s_ =copy.deepcopy(y_fit)
        for i in range(len(w_fit)):
            if w_fit[i] >= low_high and w_fit[i] <= high_low:   # ONLY CORRECT in [low_high,high_low]           
                if only_correct_negative_values:
                    if y_fit[i] <= 0 : 
                        corrected_s_[i] = y_fitted[i]
                else:
                    if y_fit[i]-y_fitted[i] <= lowlimit or y_fit[i]-y_fitted[i] >= highlimit: corrected_s_[i] = y_fitted[i]
    
    
        corrected_s = np.concatenate((s_low,corrected_s_))
        corrected_s = np.concatenate((corrected_s,s_high))
        
    
        if plot:
            ptitle = "Correction in range  "+np.str(np.round(low_low,2))+" - [ "+np.str(np.round(low_high,2))+" - "+np.str(np.round(high_low,2))+" ] - "+np.str(np.round(high_high,2))
            plot_plot(w_fit,[y_fit,y_fitted,y_fitted-highlimit,y_fitted-lowlimit,corrected_s_], color=["r","b", "black","black","green"], alpha=[0.3,0.7,0.2,0.2,0.5],xmin=low_low-40, xmax=high_high+40,vlines=[low_low,low_high,high_low,high_high],ptitle=ptitle, ylabel="Normalized flux")  
            #plot_plot(w,[s,corrected_s],xmin=low_low-40, xmax=high_high+40,vlines=[low_low,low_high,high_low,high_high])
    except Exception:
        if warnings: print("  Fitting the continuum failed! Nothing done.")
        corrected_s = s

    return corrected_s
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
    "objeto" can be a cube that has been read from a fits file or  
    an rss, from which getting the integrated spectrum of a bright source.
    If bright_spectrum is given, for example, an 1D spectrum from a cube in "spec",
    "rss" have to be a valid rss for getting wavelength, 
    use:> telluric_correction_with_bright_continuum_source(EG21_red.rss1, bright_spectrum=spec)
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
def create_fits_with_mask(list_files_for_mask, filename="", plot=True, no_nans=True):  
    """
    Creates a mask using list_files_for_mask
    """   
    print("\n> Creating mask using files provided...")

    # First, read the rss files    
    intensities_for_mask=[]
    for i in range(len(list_files_for_mask)):
        rss=KOALA_RSS(list_files_for_mask[i], plot_final_rss=False, verbose=False)
        intensities_for_mask.append(rss.intensity)
        
    # Combine intensities to eliminate nans because of cosmic rays
    mask_ = np.nanmedian(intensities_for_mask, axis=0)
        
    # divide it by itself to get 1 and nans
    mask = mask_ / mask_

    # Change nans to 0 (if requested)
    if no_nans:
        for i in range(len(mask)):
            mask[i] = [0 if np.isnan(x) else 1 for x in mask[i] ]
        
    # Plot for fun if requested
    if plot:
        if no_nans:
            rss.RSS_image(image=mask, cmap="binary_r", clow=-0.0001, chigh=1., title=" - Mask", color_bar_text="Mask value (black = 0, white = 1)")
        else:
            rss.RSS_image(image=mask, cmap="binary", clow=-0.0001, chigh=1., title=" - Mask", color_bar_text="Mask value (black = 1, white = nan)")
           
    # Save mask in file if requested
    if filename != "":
        save_nresponse(rss,filename, mask=mask, no_nans=no_nans)
            
    print("\n> Mask created!")
    return mask
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_minimum_spectra(rss_file_list, percentile=0,
                        apply_throughput=False,
                        throughput_2D=[], throughput_2D_file="",
                        correct_ccd_defects=False, plot=True):
                
    ic_list=[]
    for name in rss_file_list:
        rss=KOALA_RSS(name, apply_throughput=apply_throughput, 
                      throughput_2D=throughput_2D, throughput_2D_file=throughput_2D_file,
                      correct_ccd_defects=correct_ccd_defects, plot=False)
        
        ic_list.append(rss.intensity_corrected)    
        
    n_rss = len(rss_file_list)   
    ic_min = np.nanmedian(ic_list, axis=0)
    if percentile == 0:
        percentile = 100./n_rss - 2    
    #ic_min = np.percentile(ic_list, percentile, axis=0)
    
    if plot:           
        rss.RSS_image(image=ic_min, cmap="binary_r")
    return ic_min 
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def merge_extracted_flappyflats(flappyflat_list,  write_file="", path="", verbose = True):
    
    print("\n> Merging flappy flats...")
    if verbose: print("  - path : ",path)
    data_list=[]
    for flappyflat in flappyflat_list:
        file_to_fix = path+flappyflat
        ftf = fits.open(file_to_fix)
        data_ = ftf[0].data 
        exptime = ftf[0].header['EXPOSED'] 
        data_list.append(data_/exptime)
        if verbose: print("  - File",flappyflat,"   exptime =",exptime)
        
    merged=np.nanmedian(data_list, axis=0)
    
    # Save file
    if write_file != "":
        ftf[0].data = merged
        ftf[0].header['EXPOSED'] = 1.0
        ftf[0].header['HISTORY'] = "Median flappyflat using Python - A.L-S"
        print("\n  Saving merged flappyflat to file ",write_file,"...")
        ftf.writeto(path+write_file, overwrite=True)
    
    return merged
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------       
def fix_these_features(w,s,features=[],sky_fibres=[], sky_spectrum=[], objeto="", plot_all = False): #objeto=test):
            
    ff=copy.deepcopy(s)
    
    kind_of_features = [features[i][0] for i in range(len(features))]
#    if "g" in kind_of_features or "s" in kind_of_features:
#        if len(sky_spectrum) == 0 :             
#            sky_spectrum=objeto.plot_combined_spectrum(list_spectra=sky_fibres, plot=False, median=True)
    if "s" in kind_of_features:
        if len(sky_spectrum) == 0 :             
            sky_spectrum=objeto.plot_combined_spectrum(list_spectra=sky_fibres, plot=plot_all, median=True)

                        
    for feature in features:
        #plot_plot(w,ff,xmin=feature[1]-20,xmax=feature[4]+20)
        if feature[0] == "l":   # Line
            resultado = fluxes(w,ff, feature[1], lowlow=feature[2],lowhigh=feature[3],highlow=feature[4],highhigh=feature[5],broad=feature[6],plot=feature[7],verbose=feature[8])
            ff=resultado[11] 
        if feature[0] == "r":   # range
            ff = get_continuum_in_range(w,ff,feature[1],feature[2],feature[3],feature[4],pmin=feature[5],pmax=feature[6],fit_degree=feature[7],plot=feature[8],verbose=feature[9])
        if feature[0] == "g":   # gaussian           
#            resultado = fluxes(w,sky_spectrum, feature[1], lowlow=feature[2],lowhigh=feature[3],highlow=feature[4],highhigh=feature[5],broad=feature[6],plot=feature[7],verbose=feature[8])
#            sky_feature=sky_spectrum-resultado[11]
            resultado = fluxes(w,s, feature[1], lowlow=feature[2],lowhigh=feature[3],highlow=feature[4],highhigh=feature[5],broad=feature[6],plot=feature[7],verbose=feature[8])
            sky_feature=s-resultado[11]
            ff = ff - sky_feature
        if feature[0] == "n":    # negative values
            ff = get_continuum_in_range(w,ff,feature[1],feature[2],feature[3],feature[4],pmin=feature[5],pmax=feature[6],fit_degree=feature[7],plot=feature[8],verbose=feature[9],only_correct_negative_values = True)
        if feature[0] == "s":    # sustract
            ff_low = ff[np.where(w < feature[2])]
            ff_high = ff[np.where(w > feature[3])]
            subs = ff - sky_spectrum
            ff_replace = subs[np.where((w >= feature[2]) & (w <= feature[3]))]
            ff_ = np.concatenate((ff_low,ff_replace))
            ff_ = np.concatenate((ff_,ff_high))
            
            c = get_continuum_in_range(w,ff_,feature[1],feature[2],feature[3],feature[4],pmin=feature[5],pmax=feature[6],fit_degree=feature[7],plot=feature[8],verbose=feature[9],only_correct_negative_values=True)

            
            if feature[8] or plot_all : #plot
                vlines=[feature[1],feature[2],feature[3],feature[4]]
                plot_plot(w,[ff, ff_,c],xmin=feature[1]-20,xmax=feature[4]+20,vlines=vlines,alpha=[0.1,0.2,0.8],ptitle="Correcting 's'")
            
            ff=c    

    return ff   
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------  
def fix_these_features_in_all_spectra(objeto,features=[], sky_fibres=[], sky_spectrum=[],
                                      fibre_list=[-1],
                                      replace=False, plot = True):

    if len(sky_fibres) != 0 and len(sky_spectrum) == 0:
        sky_spectrum=objeto.plot_combined_spectrum(list_spectra=sky_fibres, plot=False, median=True)

    fix = copy.deepcopy(objeto.intensity_corrected)

    if len(fibre_list) == 0 : #[0] == -1:
        print("\n> Fixing the requested features in all the fibres...")
        fibre_list = list(range(len(fix))) 
    else:
        print("\n> Fixing the requested features in the given fibres...")

    n_spectra = len(fibre_list)
    w = objeto.wavelength
    
    if plot: objeto.RSS_image(title=" - Before fixing features")
    
    sys.stdout.write("  Fixing {} spectra...       ".format(n_spectra))
    sys.stdout.flush()
    output_every_few = np.sqrt(n_spectra)+1
    next_output = -1
    i=0
    for fibre in fibre_list: #range(n_spectra):
        i=i+1
        if fibre > next_output:
                sys.stdout.write("\b"*6)
                sys.stdout.write("{:5.2f}%".format(i*100./n_spectra))
                sys.stdout.flush()
                next_output = fibre + output_every_few
        fix[fibre] = fix_these_features(w,fix[fibre],features=features,sky_fibres=sky_fibres, sky_spectrum=sky_spectrum)
        
    if plot: objeto.RSS_image(image=fix,title=" - After fixing features")    
    if replace:
        print("\n  Replacing everything in self.intensity_corrected...")
        objeto.intensity_corrected=copy.deepcopy(fix)


    objeto.history.append("- Sky residuals cleaned on these features:")
    for feature in features:                   
        objeto.history.append("  "+np.str(feature))

    return fix
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fix_red_edge(w,f, fix_from=9220, median_from=8800,kernel_size=101, disp=1.5,plot=False):
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

    min_value,median_value,max_value,std, rms, snr=basic_statistics(f,x=w,xmin=median_from,xmax=fix_from, return_data=True, verbose = False)
    
    f_fix=[]  
    f_fix.extend((f[i]) for i in range(len(w)) if (w[i] >= fix_from) )
    f_still=[] 
    f_still.extend((f[i]) for i in range(len(w)) if (w[i] < fix_from) )

    f_fix =[median_value if (median_value+disp*std < x) or (median_value-disp*std > x) else x for x in f_fix ]        
    ff = np.concatenate((f_still,f_fix))
          
    if plot: plot_plot(w,[f,ff],vlines=[median_from,fix_from], xmin=median_from)
    return ff
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fix_blue_edge(w,f, fix_to=6100, median_to=6300,kernel_size=101, disp=1.5,plot=False):

    min_value,median_value,max_value,std, rms, snr=basic_statistics(f,x=w,xmin=fix_to,xmax=median_to, return_data=True, verbose = False)
    
    f_fix=[] 
    f_fix.extend((f[i]) for i in range(len(w)) if (w[i] <= fix_to) )
    f_still=[]
    f_still.extend((f[i]) for i in range(len(w)) if (w[i] > fix_to) )
    
    f_fix =[median_value if (median_value+disp*std < x) or (median_value-disp*std > x) else x for x in f_fix ]        
    
    ff = np.concatenate((f_fix,f_still))
    
    if plot: plot_plot(w,[f,ff],vlines=[median_to,fix_to], xmax=median_to)
    return ff
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def find_cosmics_in_cut(x, cut_wave, cut_brightest_line, line_wavelength = 0.,
                       kernel_median_cosmics = 5, cosmic_higher_than = 100, extra_factor = 1., plot=False, verbose=False):
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
           
    gc_bl=signal.medfilt(cut_brightest_line,kernel_size=kernel_median_cosmics)
    max_val = np.abs(cut_brightest_line-gc_bl)

    gc=signal.medfilt(cut_wave,kernel_size=kernel_median_cosmics)
    verde=np.abs(cut_wave-gc)-extra_factor*max_val
   
    cosmics_list = [i for i, x in enumerate(verde) if x > cosmic_higher_than]
 
    if plot:
        ptitle="Cosmic identification in cut"
        if line_wavelength != 0 : ptitle="Cosmic identification in cut at "+np.str(line_wavelength)+" $\mathrm{\AA}$"        
        plot_plot(x,verde, ymin=0,ymax=200, hlines=[cosmic_higher_than], ptitle=ptitle,  ylabel="abs (cut - medfilt(cut)) - extra_factor * max_val")
 
    if verbose:
        if line_wavelength == 0:
            print("\n> Identified", len(cosmics_list),"cosmics in fibres",cosmics_list)
        else:
            print("\n> Identified", len(cosmics_list),"cosmics at",np.str(line_wavelength),"A in fibres",cosmics_list)
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
                               ranges_with_emission_lines = [0], scale_sky_1D = 0,
                               brightest_line="Ha", brightest_line_wavelength = 0,
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
        print("  Using the scale factor provided, scale_sky_1D = ",scale_sky_1D,"...")
           
    if ranges_with_emission_lines[0] == 0 :
        #                             He I 5875.64      [O I]           Ha+[N II]   He I 6678.15   [S II]    [Ar III] 7135.78   [S III] 9069
        ranges_with_emission_lines_ =[ [5870.,5882.], [6290,6308],    [6546,6591], [6674,6684], [6710,6742], [7128,7148], [9058, 9081] ]
        ranges_with_emission_lines = []
        for i in range(len(ranges_with_emission_lines_)):
            if ranges_with_emission_lines_[i][0] > w[0] and ranges_with_emission_lines_[i][1] < w[-1]:
                ranges_with_emission_lines.append(ranges_with_emission_lines_[i])

    brightest_line_wavelength_rest = 6562.82
    if brightest_line == "O3" or brightest_line == "O3b" : brightest_line_wavelength_rest = 5006.84
    if brightest_line == "Hb" or brightest_line == "hb" : brightest_line_wavelength_rest = 4861.33

    redshift = brightest_line_wavelength/brightest_line_wavelength_rest - 1.
        
    do_this = True
    if brightest_line_wavelength != 0:
        print("  Brightest emission line in object is ",brightest_line,", centered at ", brightest_line_wavelength,"A, redshift = ",redshift)
    else:
        print("\n\n\n****************************************************************************************************")
        print("\n> WARNING !!   No wavelength provided to 'brightest_line_wavelength', no replacement can be done !!!")
        print("               Run this again providing a value to 'brightest_line_wavelength' !!!\n")
        print("****************************************************************************************************\n\n\n")

        do_this= False
        good_sky_red = sky_r_self
        
    if do_this:   
        
        print("\n  Wavelength ranges to replace (redshift considered) : ")
        for rango in ranges_with_emission_lines:
            print("  - ", np.round((redshift +1)*rango[0],2), " - ",np.round((redshift +1)*rango[1],2))
            
        change_rango = False
        rango = 1
        i=0         
        while rango < len(ranges_with_emission_lines)+1:    
            if w[i] > (redshift +1)*ranges_with_emission_lines[rango-1][0] and w[i] < (redshift +1)*ranges_with_emission_lines[rango-1][1]:
                good_sky_red[i] = sky_r_star[i]*scale_sky_1D
                change_rango = True
            else:
                if change_rango :
                    change_rango = False
                    rango = rango + 1               
            i = i+1  
            
        # Add the red end  if cut_red_end is NOT -1
        if cut_red_end != -1:
            if cut_red_end == 0:
                # Using the value of the mask of the rss
                cut_red_end = rss.valid_wave_max - 6    # a bit extra
            if verbose: print("  Also fixing the red end of the object spectrum from ",np.round(cut_red_end,2),"...")       
            w_ = np.abs(w - cut_red_end)
            i_corte = w_.tolist().index(np.nanmin(w_))
            good_sky_red[i_corte:-1] = sky_r_star[i_corte:-1]*scale_sky_1D
        else:
            if verbose: print("  The red end of the object spectrum has not been modified as cut_red_end = -1")             
        
        if plot:   
            if verbose: print("\n  Plotting the results ...")
                
            vlines = [] 
            rango_plot=[]
            for rango in ranges_with_emission_lines:
                vlines.append(rango[0]*(redshift +1))
                vlines.append(rango[1]*(redshift +1))
                _rango_plot_=[vlines[-2],vlines[-1]]
                rango_plot.append(_rango_plot_)
                            
            ptitle = "Checking the result of replacing ranges with emission lines with sky" 
            label=["Sky * scale", "Self sky", "Replacement Sky"]
            
            #print(vlines)
            #print(rango_plot)

            if ranges_with_emission_lines[0][0] == 5870:           
                plot_plot(w, [sky_r_star*scale_sky_1D,sky_r_self, good_sky_red], 
                          ptitle=ptitle,label=label, axvspan=rango_plot,
                          xmin=5800*(redshift +1),xmax=6100*(redshift +1), vlines=vlines)
            plot_plot(w, [sky_r_star*scale_sky_1D,sky_r_self, good_sky_red], 
                      ptitle=ptitle,label=label,
                      xmin=6200*(redshift +1),xmax=6400*(redshift +1), vlines=vlines, axvspan=rango_plot)
            plot_plot(w, [sky_r_star*scale_sky_1D,sky_r_self, good_sky_red], ptitle=ptitle,xmin=6500*(redshift +1),xmax=6800*(redshift +1), vlines=vlines,label=label,axvspan=rango_plot)  
            plot_plot(w, [sky_r_star*scale_sky_1D,sky_r_self, good_sky_red], ptitle=ptitle,xmin=6800*(redshift +1),xmax=7200*(redshift +1), vlines=vlines,label=label,axvspan=rango_plot)
            if ranges_with_emission_lines[-1][0] == 9058:
                plot_plot(w, [sky_r_star*scale_sky_1D,sky_r_self, good_sky_red], ptitle=ptitle,xmin=9000*(redshift +1),xmax=9130*(redshift +1), vlines=vlines,label=label,axvspan=rango_plot)
                
                
    return good_sky_red
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def auto_scale_two_spectra(rss, sky_r_self, sky_r_star, scale=[0.1,1.11,0.025], 
                           #w_scale_min = 6400,  w_scale_max = 7200,
                           w_scale_min = "",  w_scale_max = "",
                           plot=True, verbose = True ):
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
    
    w=rss.wavelength
    if w_scale_min == "" : w_scale_min=w[0]
    if w_scale_max == "" : w_scale_max=w[-1]
   
    if len(scale) == 2: scale.append(0.025)
    
    steps = np.arange(scale[0],scale[1],scale[2])
    region = np.where((w > w_scale_min) & (w<w_scale_max))
    
    factor =[]
    rsm =[]
    
    for step in steps:
        sub = np.abs(sky_r_self - step * sky_r_star)
        factor.append(step)
        rsm.append(np.nansum(sub[region]))

    auto_scale = factor[rsm.index(np.nanmin(rsm))]
    factor_v = factor[rsm.index(np.nanmin(rsm))-5:rsm.index(np.nanmin(rsm))+5]
    rsm_v = rsm[rsm.index(np.nanmin(rsm))-5:rsm.index(np.nanmin(rsm))+5]
    
    if auto_scale == steps[0] or auto_scale == steps[-1]:
        if verbose: 
            print("  No minimum found in the scaling interval {} - {} ...".format(scale[0],scale[1]))
            print("  NOTHING DONE ! ")
        return 1.
    else:           
        fit= np.polyfit(factor_v,rsm_v,2)
        yfit = np.poly1d(fit)
        vector = np.arange(scale[0],scale[1],0.001)
        rsm_fitted = yfit(vector)
        auto_scale_fit = vector[rsm_fitted.tolist().index(np.nanmin(rsm_fitted))]
        
        if plot:        
            ptitle="Auto scale factor found (using fit) between OBJ and SKY = "+np.str(np.round(auto_scale_fit,3))
            plot_plot([factor,vector], [rsm,rsm_fitted] , vlines=[auto_scale, auto_scale_fit], label=["Measured","Fit"],
                  xlabel="Scale factor", ylabel="Absolute flux difference ( OBJ - SKY ) [counts]",
                  ymin = np.nanmin(rsm) - (np.nanmax(rsm)-np.nanmin(rsm))/10.   , ptitle=ptitle)
    
    
            sub = sky_r_self-auto_scale_fit*sky_r_star
            #plot_plot(w, sub ,ymax=np.percentile(sub,99.6), hlines=[0], ptitle="Sky sustracted")
        
            ptitle = "Sky substraction applying the automatic factor of "+np.str(np.round(auto_scale_fit,3))+" to the sky emission"
            plot_plot(w, [sky_r_self, auto_scale_fit*sky_r_star, sub], xmin=w_scale_min,xmax=w_scale_max, color=["b","r", "g"],
                  label=["obj","sky", "obj-sky"], ymax=np.nanpercentile(sky_r_self[region],98), hlines=[0],
                  #ymin=np.percentile(sky_r_self[region],0.01))
                  ymin=np.nanpercentile(sub[region]-10,0.01), ptitle=ptitle)
    
        if verbose:
            print("  Auto scale factor       = ", np.round(auto_scale,3))
            print("  Auto scale factor (fit) = ", np.round(auto_scale_fit,3))
      
        return np.round(auto_scale_fit,3)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def remove_negative_pixels(spectra, verbose = True):
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
    cuenta=0

    output=copy.deepcopy(spectra)        
    for fibre in range(len(spectra)):
        vector_ = spectra[fibre]   
        stats_=basic_statistics(vector_, return_data=True, verbose=False)
        #rss.low_cut.append(stats_[1])
        if stats_[1] < 0.:
            cuenta = cuenta + 1
            vector_ = vector_ - stats_[1]
            output[fibre] = [0. if x < 0. else x for x in vector_]  
            
    if verbose: print("\n> Found {} spectra for which the median value is negative, they have been corrected".format(cuenta))
    return output

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def centroid_of_cube(cube, x0=0,x1=-1,y0=0,y1=-1, box_x=[], box_y=[],
                     step_tracing=100, g2d=True, adr_index_fit=2,
                     edgelow=-1, edgehigh=-1,
                     plot=True, log=True, gamma=0.,
                     plot_residua=True, plot_tracing_maps=[], verbose=True) :
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
        if np.nanmedian([x0,x1,y0,y1]) != -0.5:
            print("\n> Computing the centroid of the cube in box [ {:.0f} , {:.0f} ] , [ {:.0f} , {:.0f} ] with the given parameters:".format(x0,x1,y0,y1))
        else:
            print("\n> Computing the centroid of the cube using all spaxels with the given parameters:")
        if g2d:
            print("  step =", step_tracing, " , adr_index_fit =", adr_index_fit, " , using a 2D Gaussian fit")
        else:
            print("  step =", step_tracing, " , adr_index_fit =", adr_index_fit, " , using the center of mass of the image")
            
    cube_trimmed = copy.deepcopy(cube)
    
    if np.nanmedian([x0,x1,y0,y1]) != -0.5:
        cube_trimmed.data = cube.data[:,y0:y1,x0:x1]
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
                                      plot_spaxel_list = [[xc,yc]], log=log, gamma=gamma,
                                      g2d=g2d,
                                      verbose=False, trimmed=trimmed)    # FORO
                if verbose: print('  Centroid at {} A found in spaxel [ {:.2f} , {:.2f} ]  =  [ {:.2f}" , {:.2f}" ]'.format(plot_tracing_maps[show_map], xc,yc,xc*cube.pixel_size_arcsec,yc*cube.pixel_size_arcsec))            
                show_map = show_map +1
                if show_map == len(plot_tracing_maps) : show_map = -1
                       
        xc_vector.append(xc)
        yc_vector.append(yc)
        
    x_peaks_fit  = np.polyfit(wc_vector, xc_vector, adr_index_fit) 
    pp=np.poly1d(x_peaks_fit)
    x_peaks=pp(cube.wavelength) +x0

    y_peaks_fit  = np.polyfit(wc_vector, yc_vector, adr_index_fit) 
    pp=np.poly1d(y_peaks_fit)
    y_peaks=pp(cube.wavelength) +y0 
        
    xc_vector= (xc_vector - np.nanmedian(xc_vector)) *cube.pixel_size_arcsec
    yc_vector= (yc_vector - np.nanmedian(yc_vector)) *cube.pixel_size_arcsec
        
    ADR_x_fit=np.polyfit(wc_vector, xc_vector, adr_index_fit)  
    pp=np.poly1d(ADR_x_fit)
    fx=pp(wc_vector)

    ADR_y_fit=np.polyfit(wc_vector, yc_vector, adr_index_fit)  
    pp=np.poly1d(ADR_y_fit)
    fy=pp(wc_vector)
    
      
    vlines = [cube.wavelength[valid_wave_min_index], cube.wavelength[valid_wave_max_index]]
    if plot: 
        plot_plot(wc_vector, [xc_vector,yc_vector, fx,fy], psym=["+", "o", "",""], color=["r", "k", "g","b"], 
                  alpha=[1,1,1,1], label=["RA","Dec", "RA fit", "Dec fit"],
                  xmin=cube.wavelength[0],xmax=cube.wavelength[-1], vlines=vlines, markersize=[10,7],
                  ylabel="$\Delta$ offset [arcsec]",ptitle=ptitle, hlines=[0], frameon=True, 
                  ymin = np.nanmin([np.nanmin(xc_vector), np.nanmin(yc_vector)]),    
                  ymax = np.nanmax([np.nanmax(xc_vector), np.nanmax(yc_vector)]))

    # ADR_x_max=np.nanmax(xc_vector)-np.nanmin(xc_vector)    ##### USE FITS INSTEAD OF VECTOR FOR REMOVING OUTLIERS
    # ADR_y_max=np.nanmax(yc_vector)-np.nanmin(yc_vector)

    ADR_x_max=np.nanmax(fx)-np.nanmin(fx)                    ##### USING FITS
    ADR_y_max=np.nanmax(fy)-np.nanmin(fy)

    ADR_total = np.sqrt(ADR_x_max**2 + ADR_y_max**2)   

    stat_x=basic_statistics(xc_vector-fx, verbose=False, return_data=True)
    stat_y=basic_statistics(yc_vector-fy, verbose=False, return_data=True)
    stat_total = np.sqrt(stat_x[3]**2 + stat_y[3]**2)  
    
    if verbose: print('  ADR variation in valid interval using fit : RA = {:.3f}" , Dec = {:.3f}" , total = {:.3f}"  that is {:.0f}% of a spaxel'.format(ADR_x_max, ADR_y_max, ADR_total, ADR_total*100./cube.pixel_size_arcsec))

    if plot_residua: 
        plot_plot(wc_vector, [xc_vector-fx,yc_vector-fy], color=["r", "k"], alpha=[1,1], ymin=-0.1, ymax=0.1,
                  hlines=[-0.08,-0.06,-0.04,-0.02,0,0,0,0,0.02,0.04,0.06,0.08], 
                  xmin=cube.wavelength[0],xmax=cube.wavelength[-1],frameon=True, label=["RA residua","Dec residua"],
                  vlines=vlines, ylabel="$\Delta$ offset [arcsec]",ptitle="Residua of the fit to the centroid fit")

    if verbose: print('  Standard deviation of residua :             RA = {:.3f}" , Dec = {:.3f}" , total = {:.3f}"  that is {:.0f}% of a spaxel'.format(stat_x[3], stat_y[3], stat_total, stat_total*100./cube.pixel_size_arcsec))
    

    return ADR_x_fit, ADR_y_fit, ADR_x_max, ADR_y_max, ADR_total, x_peaks, y_peaks    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def scale_cubes_using_common_region(cube_list, flux_ratios=[], min_wave = 0, max_wave = 0,
                                    apply_scale = True, verbose=True, plot=False):   #SCORE
    
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
                        adr_index_fit = 2, g2d=False, step_tracing=100, plot_tracing_maps=[],                       
                        trim_cube = True,  trim_values =[], remove_spaxels_not_fully_covered = True,                                              
                        plot=True, plot_weight= True, plot_spectra=True, 
                        verbose=True, say_making_combined_cube = True):
                            
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
                                         plot_tracing_maps=plot_tracing_maps,  check_ADR=check_ADR)
                        
        # ADR correction to the combined cube    
        if ADR_cc :
            combined_cube.adrcor = True
            combined_cube.ADR_correction(RSS, plot=plot, jump=jump, method="old", force_ADR=force_ADR, remove_spaxels_not_fully_covered=remove_spaxels_not_fully_covered)
            if ADR:
                combined_cube.trace_peak(box_x=box_x_centroid, box_y=box_y_centroid, 
                                         edgelow=edgelow, edgehigh =edgehigh, 
                                         plot=plot, check_ADR=True, step_tracing=step_tracing, plot_tracing_maps=plot_tracing_maps, 
                                         adr_index_fit=adr_index_fit, g2d=g2d)
                                                   
        combined_cube.get_integrated_map(box_x=box_x, box_y=box_y, fcal=fcal, plot=plot, plot_spectra=plot_spectra, plot_centroid=True, g2d=g2d)


        # Trimming combined cube if requested or needed
        combined_cube.trim_cube(trim_cube=trim_cube, trim_values=trim_values, 
                                half_size_for_centroid =half_size_for_centroid, ADR=ADR, 
                                adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing, plot_tracing_maps=plot_tracing_maps,
                                remove_spaxels_not_fully_covered = remove_spaxels_not_fully_covered,
                                box_x=box_x, box_y=box_y, edgelow=edgelow, edgehigh=edgehigh, 
                                plot_weight = plot_weight, fcal=fcal, plot=plot, plot_spectra= plot_spectra)
          
        # Computing total exposition time of combined cube  
        combined_cube.total_exptime = 0.
        combined_cube.exptimes=[]
        combined_cube.rss_list=[]
        for i in range(n_files):
            combined_cube.total_exptime = combined_cube.total_exptime + cube_aligned_object[i].total_exptime
            combined_cube.exptimes.append(cube_aligned_object[i].total_exptime)
            combined_cube.rss_list.append(cube_aligned_object[i].rss_list)
        
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
def name_keys(filename, apply_throughput=False, correct_ccd_defects = False,
                 fix_wavelengths = False, do_extinction = False, sky_method ="none",
                 do_telluric_correction = False, id_el = False,
                 correct_negative_sky = False, clean_residuals = False):
    """
    Task for automatically naming output rss files.
    """    
    if apply_throughput:
        clave = "__________" 
    else:
        clave = filename[-15:-5]
                             
    if apply_throughput: 
        T = "T"         # T = throughput
    else:
        T = clave[-9]
    if correct_ccd_defects: 
        C= "C"          # C = corrected CCD defects
    else:
        C = clave[-8]
    if fix_wavelengths: 
        W="W"          # W = Wavelength tweak
    else:
        W = clave[-7]
    if do_extinction : 
        X="X"           # X = extinction corrected
    else:
        X = clave[-6]
    if do_telluric_correction : 
        U="U"           # U = Telluric corrected
    else:
        U = clave[-5]    
    if sky_method != "none" : 
        S="S"           # S = Sky substracted
    else:
        S = clave[-4]
    if id_el :     
        E="E"           # E = Emission lines identified
    else:
        E = clave[-3]    
    if correct_negative_sky : 
        N="N"           # N = Negative values
    else:
        N = clave[-2]    
    if clean_residuals : 
        R="R"           # R = Sky and CCD residuals
    else:
        R = clave[-1]    

    clave="_"+T+C+W+X+U+S+E+N+R
            
    if apply_throughput:       
        return filename[0:-5]+clave+".fits"
    else:
        return filename[0:-15]+clave+".fits"
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def full_path(filename,path, verbose=False):
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
    if path[-1] != "/" : path = path+"/" # If path does not end in "/" it is added
    
    if len(filename.replace("/","")) == len(filename):
        if verbose: print("\n> Variable {} does not include the full path {}".format(filename,path))
        fullpath = path+filename
    else:
        if verbose: print("\n> Variable {} includes the full path {}".format(filename,path))
        fullpath = filename
    return fullpath
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def list_fits_files_in_folder(path, verbose = True, use2=True, use3=False, ignore_offsets=True, 
                              skyflat_names=[], ignore_list=[], return_list=False):  
    
    nothing=0           # Stupid thing for controling Exceptions
    list_of_objetos=[]
    list_of_files=[]
    list_of_exptimes=[]
    if len(skyflat_names) ==0:
        skyflat_names = ["skyflat", "SKYFLAT", "SkyFlat"]

    if len(ignore_list) == 0:   
        ignore_list = ["a", "b", "c", "d", "e", "f", "p", "pos", "Pos",
                       "A", "B", "C", "D", "E", "F", "P", "POS",
                       "p1", "p2","p3","p4","p5","p6",
                       "P1", "P2","P3","P4","P5","P6",
                       "pos1", "pos2","pos3","pos4","pos5","pos6",
                       "Pos1", "Pos2","Pos3","Pos4","Pos5","Pos6",
                       "POS1", "POS2","POS3","POS4","POS5","POS6"] 
        
    
    if verbose: print("\n> Listing fits files in folder",path,":\n")
    
    if path[-1] != "/" : path=path+"/"
   
    for fitsName in sorted(glob.glob(path+'*.fits')):
        check_file = True
        if fitsName[-8:] != "red.fits" : 
            check_file = False
        if fitsName[0:8] == "combined" and check_file == False: 
            check_file = True
        for skyflat_name in skyflat_names:
            if skyflat_name in fitsName : check_file = True
        
        hdulist = pyfits.open(fitsName)

        object_fits = hdulist[0].header['OBJECT'].split(" ")
        if object_fits[0] in ["HD", "NGC", "IC"] or use2:
            try:
                if ignore_offsets == False:
                    object_fits[0]=object_fits[0]+object_fits[1]
                elif object_fits[1] not in ignore_list:
                    object_fits[0]=object_fits[0]+object_fits[1]
            except Exception:
                nothing=0
        if use3:
            try:
                if ignore_offsets == False:
                    object_fits[0]=object_fits[0]+object_fits[2]
                elif object_fits[2] not in ignore_list:
                    object_fits[0]=object_fits[0]+object_fits[2]
            except Exception:
                nothing=0
            
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
            found=False
            for i in range(len(list_of_objetos)):          
                if list_of_objetos[i] == object_fits[0]:
                    found=True
                    list_of_files[i].append(fitsName)               
                    list_of_exptimes[i].append(exptime)
            if found == False:
                list_of_objetos.append(object_fits[0])
                list_of_files.append([fitsName])
                list_of_exptimes.append([exptime])
             
    date =date_[0:4]+date_[5:7]+date_[8:10]   
             
    if verbose:
        for i in range(len(list_of_objetos)):
            for j in range(len(list_of_files[i])):
                if j == 0: 
                    print("  {:15s}  {}          {:.1f} s".format(list_of_objetos[i], list_of_files[i][0], list_of_exptimes[i][0]))
                else:
                    print("                   {}          {:.1f} s".format(list_of_files[i][j], list_of_exptimes[i][j]))
                        
        print("\n  They were obtained on {} using the grating {}".format(date,grating))

    if return_list: return list_of_objetos,list_of_files, list_of_exptimes, date,grating
    if nothing > 10 : print(nothing)  # Stupid thing for controling Exceptions
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def save_map(cube, mapa, fits_file, mask=[], description="", path="", verbose = True):
    
    if path != "" : fits_file=full_path(fits_file,path)

    if description == "" : description =mapa[0]

    fits_image_hdu = fits.PrimaryHDU(mapa[1])
         
    fits_image_hdu.header['HISTORY'] = 'Map created by PyKOALA'        
    fits_image_hdu.header['HISTORY'] = 'Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany,'
    fits_image_hdu.header['HISTORY'] = 'Blake Staples, Taylah Beard, Matt Owers, James Tocknell et al.'
    fits_image_hdu.header['HISTORY'] =  version    
    now=datetime.datetime.now()
    fits_image_hdu.header['HISTORY'] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    fits_image_hdu.header['DATE'] = now.strftime("%Y-%m-%dT%H:%M:%S") #'2002-09-16T18:52:44'   # /Date of FITS file creation
    
    fits_image_hdu.header['BITPIX']  =  16  

    fits_image_hdu.header["ORIGIN"]  = 'AAO'    #    / Originating Institution                        
    fits_image_hdu.header["TELESCOP"]= 'Anglo-Australian Telescope'    # / Telescope Name  
    fits_image_hdu.header["ALT_OBS"] =                 1164 # / Altitude of observatory in metres              
    fits_image_hdu.header["LAT_OBS"] =            -31.27704 # / Observatory latitude in degrees                
    fits_image_hdu.header["LONG_OBS"]=             149.0661 # / Observatory longitude in degrees 

    fits_image_hdu.header["INSTRUME"] = "AAOMEGA-KOALA"             # / Instrument in use  
    fits_image_hdu.header["GRATID"]  = cube.grating      # / Disperser ID 
    if cube.grating in red_gratings : SPECTID="RD"
    if cube.grating in blue_gratings : SPECTID="BL"
    fits_image_hdu.header["SPECTID"] = SPECTID                        # / Spectrograph ID                                
    fits_image_hdu.header["DICHROIC"]= 'X5700'                        # / Dichroic name   ---> CHANGE if using X6700!!
    
    fits_image_hdu.header['OBJECT'] = cube.object    
    fits_image_hdu.header['TOTALEXP'] = cube.total_exptime    
    fits_image_hdu.header['EXPTIMES'] = np.str(cube.exptimes)
                                       
    fits_image_hdu.header['NAXIS']   =   2                              # / number of array dimensions                       
    fits_image_hdu.header['NAXIS1']  =   cube.data.shape[1]        ##### CHECK !!!!!!!           
    fits_image_hdu.header['NAXIS2']  =   cube.data.shape[2]                 

    # WCS
    fits_image_hdu.header["RADECSYS"]= 'FK5'          # / FK5 reference system   
    fits_image_hdu.header["EQUINOX"] = 2000           # / [yr] Equinox of equatorial coordinates                         
    fits_image_hdu.header["WCSAXES"] =  2             # / Number of coordinate axes                      

    fits_image_hdu.header['CRPIX1']  = cube.data.shape[1]/2.         # / Pixel coordinate of reference point            
    fits_image_hdu.header['CDELT1']  = -cube.pixel_size_arcsec/3600. # / Coordinate increment at reference point      
    fits_image_hdu.header['CTYPE1']  = "RA--TAN" #'DEGREE'                               # / Coordinate type code                           
    fits_image_hdu.header['CRVAL1']  = cube.RA_centre_deg            # / Coordinate value at reference point            

    fits_image_hdu.header['CRPIX2']  = cube.data.shape[2]/2.         # / Pixel coordinate of reference point            
    fits_image_hdu.header['CDELT2']  = cube.pixel_size_arcsec/3600.  #  Coordinate increment at reference point        
    fits_image_hdu.header['CTYPE2']  = "DEC--TAN" #'DEGREE'                               # / Coordinate type code                           
    fits_image_hdu.header['CRVAL2']  = cube.DEC_centre_deg           # / Coordinate value at reference point            
 
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
                mask_name1="MASK"+np.str(i+1)+"1" 
                mask_name2="MASK"+np.str(i+1)+"2" 
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
        hdu_list = fits.HDUList([fits_image_hdu, fits_velocity, fits_fwhm, fits_ew]) #, fits_mask])
        fits_image_hdu.header['HISTORY'] = 'This was obtained doing a Gassian fit'
        fits_image_hdu.header['HISTORY'] = 'Extension[2] is the velocity map [km/s]'
        fits_image_hdu.header['HISTORY'] = 'Extension[3] is the FWHM map [km/s]'
        fits_image_hdu.header['HISTORY'] = 'Extension[4] is the EW map [A]'
        
    except Exception:
        hdu_list = fits.HDUList([fits_image_hdu]) #, fits_mask])

    hdu_list.writeto(fits_file, overwrite=True) 
    if verbose:
        print("\n> Map saved to file:")
        print(" ",fits_file)
        print("  Description:",description)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def load_map(mapa_fits, description="", path="", verbose = True):
    
    if verbose: print("\n> Reading map(s) stored in file", mapa_fits,"...")
        
    if path != "" : mapa_fits=full_path(mapa_fits,path)
    mapa_fits_data = fits.open(mapa_fits)  # Open file

    if description == "" : description = mapa_fits_data[0].header['DESCRIP']    #
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
    
    fail=0
    try:
        for i in range(4):
            try:
                mask_name1="MASK"+np.str(i+1)+"1" 
                mask_name2="MASK"+np.str(i+1)+"2" 
                mask_low_limit = mapa_fits_data[0].header[mask_name1] 
                mask_high_limit = mapa_fits_data[0].header[mask_name2] 
                _mask_ = create_mask(mapa_fits_data[i].data, low_limit=mask_low_limit,  high_limit=mask_high_limit, verbose=False)
                mapa.append(_mask_)
                if verbose: print("- Mask with good values between {} and {} created and stored in [{}]".format(mask_low_limit,mask_high_limit,len(mapa)-1))
            except Exception:
                fail=fail+1                    
            
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
                                throughput_2D = 0,
                                skyflat=0,
                                do_skyflat = True, 
                                kernel_throughput =0,
                                correct_ccd_defects=True,
                                fix_wavelengths = False,
                                sol=[0],
                                rss_star_file_for_sol = "",
                                plot=True,
                                CONFIG_FILE_path="",
                                CONFIG_FILE_list=[],
                                star_list=[],
                                abs_flux_scale=[],
                                flux_calibration_file="",
                                telluric_correction_file ="",
                                objects_auto=[],
                                auto = False,
                                rss_clean = False,
                                flux_calibration_name="flux_calibration_auto",
                                cal_from_calibrated_starcubes = False,
                                disable_stars=[],                      # stars in this list will not be used
                                skyflat_names=[]
                                ):
    """
    Use: 
        CALIBRATION_NIGHT_FILE = "./CONFIG_FILES/calibration_night.config"
        automatic_calibration_night(CALIBRATION_NIGHT_FILE)
    """
    
    if len(skyflat_names) == 0: skyflat_names=["SKYFLAT", "skyflat", "Skyflat", "SkyFlat", "SKYFlat", "SkyFLAT"]
    
    w=[]
    telluric_correction_list=[]  
    global skyflat_variable
    skyflat_variable = ""
    global skyflat_
    global throughput_2D_variable
    global flux_calibration_night
    global telluric_correction_night
    throughput_2D_variable = "" 
    global throughput_2D_
    throughput_2D_ = [0]
    
    if flux_calibration_file == "":flux_calibration_file=path+"flux_calibration_file_auto.dat"
    if telluric_correction_file == "":telluric_correction_file=path+"telluric_correction_file_auto.dat"
     
    
    check_nothing_done = 0

    print("\n===================================================================================")
    
    if auto:
        print("\n    COMPLETELY AUTOMATIC CALIBRATION OF THE NIGHT ")
        print("\n===================================================================================")
    
    if len(CALIBRATION_NIGHT_FILE) > 0:
        config_property, config_value = read_table(CALIBRATION_NIGHT_FILE, ["s", "s"] )    
        print("\n> Reading configuration file ", CALIBRATION_NIGHT_FILE)
        print("  for performing the automatic calibration of the night...\n")
    else:    
        print("\n> Using the values given in automatic_calibration_night()")
        print("  for performing the automatic calibration of the night...\n")
        config_property = []
        config_value    = []
        lista_propiedades = ["path", "file_skyflat", "rss_star_file_for_sol", "flux_calibration_file", "telluric_correction_file"]
        lista_valores     = [path, file_skyflat, rss_star_file_for_sol, flux_calibration_file, telluric_correction_file]
        for i in range(len(lista_propiedades)):
            if len(lista_valores[i]) > 0:
                config_property.append(lista_propiedades[i])
                config_value.append(lista_valores[i])       
        if pixel_size == 0:
            print ("  - No pixel size provided, considering pixel_size = 0.7")
            pixel_size = 0.7
        if kernel_size == 0:
            print ("  - No kernel size provided, considering kernel_size = 1.1")
            kernel_size = 1.1
        pk = "_"+str(int(pixel_size))+"p"+str(int((abs(pixel_size)-abs(int(pixel_size)))*10))+"_"+str(int(kernel_size))+"k"+str(int(abs(kernel_size*100))-int(kernel_size)*100)
        if sol[0] != 0 : fix_wavelengths = True
        if len(CONFIG_FILE_path) > 0:
            for i in range(len(CONFIG_FILE_list)):
                CONFIG_FILE_list[i] = full_path (CONFIG_FILE_list[i],CONFIG_FILE_path)                

        

#   Completely automatic reading folder:
    
    if auto:
        fix_wavelengths = True
        list_of_objetos,list_of_files, list_of_exptimes, date,grating=list_fits_files_in_folder(path, return_list=True)
        print(" ")
        
        list_of_files_of_stars=[]
        for i in range(len(list_of_objetos)):
            if list_of_objetos[i] in skyflat_names:
                file_skyflat=list_of_files[i][0]
                print ("  - SKYFLAT automatically identified")
                
            if list_of_objetos[i] in ["H600", "HILT600", "Hilt600", "Hiltner600", "HILTNER600"]  and list_of_objetos[i] not in disable_stars:
                print ("  - Calibration star Hilt600 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("Hilt600_"+grating)                 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("Hilt600", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("Hilt600")
                CONFIG_FILE_list.append("")
 
            if list_of_objetos[i] in ["EG274", "Eg274", "eg274", "eG274", "E274", "e274"]  and list_of_objetos[i] not in disable_stars:
                print ("  - Calibration star EG274 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("EG274_"+grating) 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("EG274", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("EG274")
                CONFIG_FILE_list.append("")
 
            if list_of_objetos[i] in ["HD60753", "hd60753", "Hd60753", "HD60753FLUX"]  and list_of_objetos[i] not in disable_stars:
                print ("  - Calibration star HD60753 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("HD60753_"+grating) 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("HD60753", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("HD60753")
                CONFIG_FILE_list.append("")

            if list_of_objetos[i] in ["HD49798", "hd49798", "Hd49798"]  and list_of_objetos[i] not in disable_stars:
                print ("  - Calibration star HD49798 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("HD49798_"+grating) 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("HD49798", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("HD49798")
                CONFIG_FILE_list.append("")

            if list_of_objetos[i] in ["cd32d9927", "CD32d9927", "CD32D9927", "CD-32d9927", "cd-32d9927", "Cd-32d9927", "CD-32D9927", "cd-32D9927", "Cd-32D9927"  ]  and list_of_objetos[i] not in disable_stars:
                print ("  - Calibration star CD32d9927 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("CD32d9927_"+grating) 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("CD32d9927", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("CD32d9927")
                CONFIG_FILE_list.append("")

            if list_of_objetos[i] in ["HR3454", "Hr3454", "hr3454"] and list_of_objetos[i] not in disable_stars:
                print ("  - Calibration star HR3454 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("HR3454_"+grating) 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("HR3454", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("HR3454")
                CONFIG_FILE_list.append("")
                
            if list_of_objetos[i] in [ "HR718" ,"Hr718" , "hr718", "HR718FLUX","HR718auto" ,"Hr718auto" , "hr718auto", "HR718FLUXauto"  ]  and list_of_objetos[i] not in disable_stars: 
                print ("  - Calibration star HR718 automatically identified")
                list_of_files_of_stars.append(list_of_files[i])
                rss_star_file_for_sol=list_of_files[i][0]
                objects_auto.append("HR718_"+grating) 
                #_CONFIG_FILE_, _description_, _fits_file_, _response_file_, _absolute_flux_file_, _list_of_telluric_ranges_ =  get_calibration_star_data ("HR718", path, grating, pk)
                #CONFIG_FILE_list.append(_CONFIG_FILE_)
                star_list.append("HR718")
                CONFIG_FILE_list.append("")


        if throughput_2D_file != "":
            throughput_2D_file = full_path(throughput_2D_file,path) 
            do_skyflat = False
            print ("  - throughput_2D_file provided, no need of processing skyflat")          
            sol=[0,0,0]
            ftf = fits.open(throughput_2D_file)
            if ftf[0].data[0][0] == 1. :           
                sol[0] = ftf[0].header["SOL0"]
                sol[1] = ftf[0].header["SOL1"]
                sol[2] = ftf[0].header["SOL2"]
                print ("  - solution for fixing small wavelength shifts included in this file :\n    sol = ",sol)  
        print(" ")
        
    else:
        list_of_files_of_stars=[[],[],[],[],[],[]]


    for i in range(len(config_property)):
        if  config_property[i] == "date" : 	 date = config_value[i]
        if  config_property[i] == "grating" : 	 grating = config_value[i]        
        if  config_property[i] == "pixel_size" : 	 pixel_size = float(config_value[i])         
        if  config_property[i] == "kernel_size" : 	 
            kernel_size = float(config_value[i])
            pk = "_"+str(int(pixel_size))+"p"+str(int((abs(pixel_size)-abs(int(pixel_size)))*10))+"_"+str(int(kernel_size))+"k"+str(int(abs(kernel_size*100))-int(kernel_size)*100)
        if  config_property[i] == "path" : 	
            path = config_value[i]
            if path[-1] != "/" : path = path+"/"
            throughput_2D_file = path+"throughput_2D_"+date+"_"+grating+".fits"
            flux_calibration_file = path+"flux_calibration_"+date+"_"+grating+pk+".dat" 
            if flux_calibration_name =="flux_calibration_auto" : flux_calibration_name = "flux_calibration_"+date+"_"+grating+pk 
            if grating == "385R" or grating == "1000R" :
                telluric_correction_file = path+"telluric_correction_"+date+"_"+grating+".dat" 
                telluric_correction_name = "telluric_correction_"+date+"_"+grating  
                
        if  config_property[i] == "file_skyflat" : file_skyflat = full_path(config_value[i],path)
                
        if  config_property[i] == "skyflat" : 
            exec("global "+config_value[i])
            skyflat_variable = config_value[i]

        if  config_property[i] == "do_skyflat" : 
            if config_value[i] == "True" : 
                do_skyflat = True 
            else: do_skyflat = False 

        if  config_property[i] == "correct_ccd_defects" : 
            if config_value[i] == "True" : 
                correct_ccd_defects = True 
            else: correct_ccd_defects = False 

        if  config_property[i] == "fix_wavelengths":
            if config_value[i] == "True" : fix_wavelengths = True 
        if  config_property[i] == "sol" :
            fix_wavelengths = True
            sol_ = config_value[i].strip('][').split(',')
            if float(sol_[0]) == -1:
                sol = [float(sol_[0])]
            else:
                if float(sol_[0]) != -0: sol = [float(sol_[0]),float(sol_[1]),float(sol_[2])]

        if  config_property[i] == "kernel_throughput" : 	 kernel_throughput = int(config_value[i])  

        if  config_property[i] == "rss_star_file_for_sol": rss_star_file_for_sol = full_path(config_value[i],path)

        if  config_property[i] == "throughput_2D_file" : throughput_2D_file = full_path(config_value[i],path) 
        if  config_property[i] == "throughput_2D" : throughput_2D_variable = config_value[i]  
        if  config_property[i] == "flux_calibration_file" : 	 flux_calibration_file  = full_path(config_value[i],path)      
        if  config_property[i] == "telluric_correction_file" : 	 telluric_correction_file  = full_path(config_value[i],path)      

        if  config_property[i] == "CONFIG_FILE_path" : CONFIG_FILE_path = config_value[i]       
        if  config_property[i] == "CONFIG_FILE" : CONFIG_FILE_list.append(full_path(config_value[i],CONFIG_FILE_path))
              
        if  config_property[i] == "abs_flux_scale":
            abs_flux_scale_ = config_value[i].strip('][').split(',')
            for j in range(len(abs_flux_scale_)):
                abs_flux_scale.append(float(abs_flux_scale_[j]))
            
        if  config_property[i] == "plot" : 
            if config_value[i] == "True" : 
                plot = True 
            else: plot = False 

        if  config_property[i] == "cal_from_calibrated_starcubes" and  config_value[i] == "True" : cal_from_calibrated_starcubes=True
              
        if  config_property[i] == "object" : 
            objects_auto.append(config_value[i])
  
    if len(abs_flux_scale) == 0:
        for i in range(len(CONFIG_FILE_list)): abs_flux_scale.append(1.)


# Print the summary of parameters

    print("> Parameters for automatically processing the calibrations of the night:\n")
    print("  date                       = ",date)
    print("  grating                    = ",grating)
    print("  path                       = ",path)
    if cal_from_calibrated_starcubes == False:
        if do_skyflat:     
            print("  file_skyflat               = ",file_skyflat)
            if skyflat_variable != "" : print("  Python object with skyflat = ",skyflat_variable)
            print("  correct_ccd_defects        = ",correct_ccd_defects)            
            if fix_wavelengths:
                print("  fix_wavelengths            = ",fix_wavelengths)     
                if sol[0] != 0 and sol[0] != -1:
                    print("    sol                      = ",sol)
                else:
                    if rss_star_file_for_sol =="" :
                        print("    ---> However, no solution given! Setting fix_wavelength = False !")
                        fix_wavelengths = False
                    else:
                        print("    Star RSS file for getting small wavelength solution:",rss_star_file_for_sol)
        else:
            print("  throughput_2D_file         = ",throughput_2D_file)
            if throughput_2D_variable != "" : print("  throughput_2D variable     = ",throughput_2D_variable)
    
        print("  pixel_size                 = ",pixel_size)
        print("  kernel_size                = ",kernel_size)
    
        if CONFIG_FILE_list[0] != "":
    
            for config_file in range(len(CONFIG_FILE_list)):
                if config_file == 0 : 
                    if len(CONFIG_FILE_list) > 1:       
                        print("  CONFIG_FILE_LIST           =  [",CONFIG_FILE_list[config_file],",")
                    else:
                        print("  CONFIG_FILE_LIST           =  [",CONFIG_FILE_list[config_file],"]")
                else:
                    if config_file == len(CONFIG_FILE_list)-1:
                        print("                                 ",CONFIG_FILE_list[config_file]," ]")
                    else:        
                        print("                                 ",CONFIG_FILE_list[config_file],",")           

    else:
        print("\n> The calibration of the night will be obtained using these fully calibrated starcubes:\n")

    if len(objects_auto) != 0 :
        pprint = ""
        for i in range(len(objects_auto)):
            pprint=pprint+objects_auto[i]+ "  " 
        print("  Using stars in objects     = ",pprint)

    if len(abs_flux_scale) > 0 : print("  abs_flux_scale             = ",abs_flux_scale)
    print("  plot                       = ",plot)
    
    print("\n> Output files:\n")
    if do_skyflat:
        if throughput_2D_variable != "" : print("  throughput_2D variable     = ",throughput_2D_variable)   
        print("  throughput_2D_file         = ",throughput_2D_file)
    print("  flux_calibration_file      = ",flux_calibration_file)
    if grating in red_gratings:
        print("  telluric_correction_file   = ",telluric_correction_file)

    print("\n===================================================================================")
               
    if do_skyflat:      
        if rss_star_file_for_sol != "" and sol[0] == 0 :
            print("\n> Getting the small wavelength solution, sol, using star RSS file")
            print(" ",rss_star_file_for_sol,"...")                                  
            if grating in red_gratings :
                _rss_star_ = KOALA_RSS(rss_star_file_for_sol, 
                                       correct_ccd_defects = False, 
                                       fix_wavelengths=True, sol = [0],
                                       plot= plot)
            if grating in ["580V"] :
                _rss_star_ = KOALA_RSS(rss_star_file_for_sol, 
                                       correct_ccd_defects = True, remove_5577 = True,
                                       plot= plot)               
            sol = _rss_star_.sol
            print("\n> Solution for the small wavelength variations:")
            print(" ",sol)
        
        throughput_2D_, skyflat_ =  get_throughput_2D(file_skyflat, plot = plot, also_return_skyflat = True,
                                            correct_ccd_defects = correct_ccd_defects,
                                            fix_wavelengths = fix_wavelengths, sol = sol,
                                            throughput_2D_file =throughput_2D_file, kernel_throughput = kernel_throughput)      
        
        if throughput_2D_variable != "":
            print("  Saving throughput 2D into Python variable:", throughput_2D_variable)
            exec(throughput_2D_variable+"=throughput_2D_", globals())

        if skyflat_variable != "":
            print("  Saving skyflat into Python variable:", skyflat_variable)
            exec(skyflat_variable+"=skyflat_", globals())

    else:
        if cal_from_calibrated_starcubes == False: print("\n> Skyflat will not be processed! Throughput 2D calibration already provided.\n")
        check_nothing_done = check_nothing_done + 1

    good_CONFIG_FILE_list =[]
    good_star_names =[]
    stars=[]
    if cal_from_calibrated_starcubes == False: 
        for i in range(len(CONFIG_FILE_list)):
      
            run_star = True
            
            if CONFIG_FILE_list[i] != "":            
                try:
                    config_property, config_value = read_table(CONFIG_FILE_list[i], ["s", "s"] )
                    if len(CONFIG_FILE_list) != len(objects_auto)  :               
                        for j in range (len(config_property)):
                            if config_property[j] == "obj_name" : running_star = config_value[j] 
                        if i < len(objects_auto) :
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
                if i < len(objects_auto) :
                    objects_auto[i] = running_star
                else:    
                    objects_auto.append(running_star)   
                

            if run_star:
                pepe=0
                if pepe == 0:
                #try:
                    print("===================================================================================")        
                    print("\n> Running automatically calibration star",running_star, "in CONFIG_FILE:")
                    print(" ",CONFIG_FILE_list[i],"\n")
                    psol="["+np.str(sol[0])+","+np.str(sol[1])+","+np.str(sol[2])+"]"
                    exec('run_automatic_star(CONFIG_FILE_list[i], object_auto="'+running_star+'", star=star_list[i], sol ='+psol+', throughput_2D_file = "'+throughput_2D_file+'", rss_list = list_of_files_of_stars[i], path_star=path, date=date,grating=grating,pixel_size=pixel_size,kernel_size=kernel_size, rss_clean=rss_clean)')
                    print("\n> Running automatically calibration star in CONFIG_FILE")
                    print("  ",CONFIG_FILE_list[i]," SUCCESSFUL !!\n")
                    good_CONFIG_FILE_list.append(CONFIG_FILE_list[i])
                    good_star_names.append(running_star)
                    try: # This is for a combined cube
                        exec("stars.append("+running_star+".combined_cube)")      
                        if grating in red_gratings:
                            exec("telluric_correction_list.append("+running_star+".combined_cube.telluric_correction)")
                    except Exception: # This is when we read a cube from fits file
                        exec("stars.append("+running_star+")")      
                        if grating in red_gratings:
                            exec("telluric_correction_list.append("+running_star+".telluric_correction)")                                
                # except Exception:   
                #     print("===================================================================================")
                #     print("\n> ERROR! something wrong happened running config file {} !\n".format(CONFIG_FILE_list[i]))

    else:       # This is for the case that we have individual star cubes ALREADY calibrated in flux
        pprint = ""
        stars=[]
        good_star_names=[]
        for i in range(len(objects_auto)):
            pprint=pprint+objects_auto[i]+ "  " 
            try: # This is for a combined cube
                exec("stars.append("+objects_auto[i]+".combined_cube)")
                if grating in red_gratings:
                    exec("telluric_correction_list.append("+objects_auto[i]+".combined_cube.telluric_correction)")
            except Exception: # This is when we read a cube from fits file
                exec("stars.append("+objects_auto[i]+")") 
                if grating in red_gratings:
                    exec("telluric_correction_list.append("+objects_auto[i]+".telluric_correction)")  
            good_star_names.append(stars[i].object)
                
        print("\n> Fully calibrated star cubes provided :",pprint) 
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
        w= stars[0].wavelength
        spectrum_to_text_file(w, flux_calibration_night, filename=flux_calibration_file)

        # Similarly, provide a list with the telluric corrections and apply:            
        if grating in red_gratings:
            telluric_correction_night = obtain_telluric_correction(w,telluric_correction_list, label_stars=good_star_names, scale=abs_flux_scale)            
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
        print("\n> SUMMARY of running configuration file", CALIBRATION_NIGHT_FILE,":\n") 
    else:
        print("\n> SUMMARY of running automatic_calibration_night() :\n") 


    if len(objects_auto) > 0 and cal_from_calibrated_starcubes == False:    
        pprint = ""
        for i in range(len(objects_auto)):
            pprint=pprint+objects_auto[i]+ "  " 
        print("  Created objects for calibration stars           :",pprint) 
    
        if len(CONFIG_FILE_list) > 0:    
            print("  Variable with the flux calibration              :",flux_calibration_name)
            if grating in red_gratings:
                print("  Variable with the telluric calibration          :",telluric_correction_name)
                print(" ")
        print("  throughput_2D_file        = ",throughput_2D_file)
        if throughput_2D_variable != "" : print("  throughput_2D variable    = ",throughput_2D_variable)
    
        if sol[0] != -1 and sol[0] != 0:
            print("  The throughput_2D information HAS BEEN corrected for small wavelength variations:")
            print("  sol                       =  ["+np.str(sol[0])+","+np.str(sol[1])+","+np.str(sol[2])+"]")
    
        if skyflat_variable != "" : print("  Python object created with skyflat = ",skyflat_variable)
        
        if len(CONFIG_FILE_list) > 0:
            print('  flux_calibration_file     = "'+flux_calibration_file+'"')
            if grating in red_gratings:
                print('  telluric_correction_file  = "'+telluric_correction_file+'"')
 
    if cal_from_calibrated_starcubes:
        print("  Variable with the flux calibration              :",flux_calibration_name)
        if grating in red_gratings:
                print("  Variable with the telluric calibration          :",telluric_correction_name)
                print(" ")       
        print('  flux_calibration_file     = "'+flux_calibration_file+'"')
        if grating in red_gratings:
            print('  telluric_correction_file  = "'+telluric_correction_file+'"')
        
 
    if check_nothing_done == 2:
        print("\n> NOTHING DONE!")
              
    print("\n===================================================================================")   
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_calibration_star_data(star, path_star, grating, pk):
 
    
    description = star
    fits_file = path_star+star+"_"+grating+pk+".fits"
    response_file = path_star+star+"_"+grating+pk+"_response.dat" 
    telluric_file = path_star+star+"_"+grating+pk+"_telluric_correction.dat" 
    
    if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_star_blue.config"
    if grating in red_gratings : 
        CONFIG_FILE="CONFIG_FILES/STARS/calibration_star_red.config"
        list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140],     # DEFAULT VALUES
                                     [7080,7140,7500,7580], [7400,7580,7705,7850],
                                     [7850,8090,8450,8700] ]
    else:
        list_of_telluric_ranges = [[0]]
    
    if star in ["cd32d9927", "CD32d9927", "CD32D9927", "cd32d9927auto", "CD32d9927auto", "CD32D9927auto"] : 
        absolute_flux_file = 'FLUX_CAL/fcd32d9927_edited.dat'  
        #list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140], 
        #                             [7080,7140,7500,7580], [7400,7580,7705,7850],
        #                             [7850,8090,8450,8700] ]
        # If needed, include here particular CONFIG FILES:
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_CD32d9927_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_CD32d9927_blue.config"
    if star in ["HD49798" , "hd49798" , "HD49798auto" , "hd49798auto"] : 
        absolute_flux_file = 'FLUX_CAL/fhd49798.dat'  
        #list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140], 
        #                             [7080,7140,7500,7580], [7400,7580,7705,7850],
        #                             [7850,8090,8450,8700] ]           
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HD49798_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HD49798_blue.config"
    if star in ["HD60753", "hd60753" , "HD60753auto" ,"hd60753auto", "HD60753FLUX", "hd60753FLUX" , "HD60753FLUXauto" ,"hd60753FLUXauto" ] : 
        absolute_flux_file = 'FLUX_CAL/fhd60753.dat'  
        # list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140], 
        #                             [7080,7140,7500,7580], [7400,7580,7705,7850],
        #                             [7850,8090,8450,8700] ]    
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HD60753_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HD60753_blue.config"    
    if star in [ "H600", "Hiltner600" , "Hilt600" ,"H600auto"] : 
        absolute_flux_file = 'FLUX_CAL/fhilt600_edited.dat'  
        # list_of_telluric_ranges =  [ [6150,6240,6410,6490], [6720,6855,7080,7140], 
        #                            [7080,7140,7500,7580], [7400,7580,7705,7850],
        #                            [7850,8090,8450,8700] ] 
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_Hilt600_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/ccalibration_Hilt600_blue.config"
                          
    if star in [ "EG274" , "E274" , "eg274", "e274", "EG274auto", "E274auto" , "eg274auto", "e274auto" ] :
        absolute_flux_file = '=FLUX_CAL/feg274_edited.dat'
        list_of_telluric_ranges =  [ [6150,6245,6380,6430], [6720,6855,7080,7150], 
                                     [7080,7140,7500,7580], [7400,7580,7705,7850], 
                                     [7850,8090,8450,8700] ] 
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_EG274_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_EG274_blue.config"               
    if star in ["EG21", "eg21" , "Eg21", "EG21auto", "eg21auto" , "Eg21auto"]  : 
        absolute_flux_file = 'FLUX_CAL/feg21_edited.dat'
        list_of_telluric_ranges =  [ [6150,6245,6380,6430], [6720,6855,7080,7150], 
                                     [7080,7140,7500,7580], [7400,7580,7705,7850], 
                                     [7850,8090,8450,8700] ]
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_EG21_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_EG21_blue.config"
    if star in [ "HR3454" ,"Hr3454" , "hr3454", "HR3454auto" ,"Hr3454auto" , "hr3454auto" ]  : 
        absolute_flux_file = 'FLUX_CAL/fhr3454_edited.dat'
        list_of_telluric_ranges =  [ [6150,6245,6380,6430], [6720,6855,7080,7150], 
                                     [7080,7140,7500,7580], [7400,7580,7705,7850], 
                                     [7850,8090,8450,8700] ]
        # if grating in red_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HR3454_red.config"
        # if grating in blue_gratings : CONFIG_FILE="CONFIG_FILES/STARS/calibration_HR3454_blue.config"
    if star in [ "HR718" ,"Hr718" , "hr718", "HR718FLUX","HR718auto" ,"Hr718auto" , "hr718auto", "HR718FLUXauto"  ]  : 
        absolute_flux_file = 'FLUX_CAL/fhr718_edited.dat'
        #list_of_telluric_ranges =  [ [6150,6245,6380,6430], [6720,6855,7080,7150], 
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
                       description ="",
                       obj_name ="",
                       object_auto = "",
                       date="", grating="", 
                       pixel_size  = 0.7, 
                       kernel_size = 1.1, 
                       path_star ="",
                       rss_list=[],
                       path="",
                       reduced = False,
                       read_fits_cube = False,
                       fits_file="",
                       rss_clean = False,
                       save_rss = False,
                       save_rss_to_fits_file_list=[],
                       
                       apply_throughput = False,
                       throughput_2D_variable = "",
                       throughput_2D=[], throughput_2D_file = "",
                       throughput_2D_wavecor = False,
                       valid_wave_min = 0, valid_wave_max = 0,
                       correct_ccd_defects = False,
                       fix_wavelengths = False, sol =[0,0,0],
                       do_extinction = False,
                       sky_method ="none",
                       n_sky = 100,
                       sky_fibres =[], 
                       win_sky = 0,
                       remove_5577=False,
                       correct_negative_sky = False,     
                       order_fit_negative_sky =3, 
                       kernel_negative_sky = 51,
                       individual_check = True,
                       use_fit_for_negative_sky = False,
                       force_sky_fibres_to_zero = True,
                       low_fibres=10,
                       high_fibres=20,
                       
                       remove_negative_median_values = False ,   
                       fix_edges=False,
                       clean_extreme_negatives = False,
                       percentile_min=0.9  ,
                       clean_cosmics = False,
                       width_bl = 20.,
                       kernel_median_cosmics = 5 ,
                       cosmic_higher_than 	=	100. ,
                       extra_factor =	1.,
                                                                  
                       do_cubing = True, do_alignment=True, make_combined_cube=True,
                       edgelow = -1, edgehigh =-1,
                       ADR=False, jump = -1,
                       adr_index_fit=2, g2d = True,
                       box_x=[0,-1], box_y=[0,-1],
                       trim_cube = True, trim_values =[],
                       scale_cubes_using_integflux = False,
                       flux_ratios =[],
                       
                       do_calibration = True,
                       absolute_flux_file ="",
                       response_file="",
                       size_arcsec = [],
                       r_max=5.,
                       step_flux=10.,
                       ha_width = 0, exp_time = 0.,
                       min_wave_flux = 0, max_wave_flux = 0,
                       sky_annulus_low_arcsec = 5.,
                       sky_annulus_high_arcsec = 10.,
                       exclude_wlm=[[0,0]],
                       odd_number=0,
                       smooth =0.,
                       fit_weight=0., 
                       smooth_weight=0.,
                       
                       order_telluric = 2,
                       list_of_telluric_ranges = [[0]],                                                
                       apply_tc=True, 
                       log = True, gamma = 0,   
                       fig_size = 12,                                     
                       plot = True, warnings = True, verbose = True  ): 
    """
    Use: 
        CONFIG_FILE_H600 = "./CONFIG_FILES/calibration_star1.config"
        H600auto=run_automatic_star(CONFIG_FILE_H600)
    """

    global star_object   
    sky_fibres_print ="" 
    if object_auto == "" : print("\n> Running automatic script for processing a calibration star")
    
    rss_clean_given = rss_clean
    
# # Setting default values (now the majority as part of definition)
    
    pk = "_"+str(int(pixel_size))+"p"+str(int((abs(pixel_size)-abs(int(pixel_size)))*10))+"_"+str(int(kernel_size))+"k"+str(int(abs(kernel_size*100))-int(kernel_size)*100)   
    first_telluric_range = True 

# # if path is given, add it to the rss_list if needed
    if path_star == "" and path != "" : path_star=path
    if path == "" and path_star != "" : path=path_star
    if path != "" and len(rss_list) > 0:
        for i in range(len(rss_list)):
            rss_list[i]=full_path(rss_list[i],path)


    if CONFIG_FILE == "":
        
        # If no configuration file is given, check if name of the star provided
        if star == "":
            print("  - No name for calibration star given, asuming name = star")
            star="star"
        
        # If grating is not given, we can check reading a RSS file      
        if grating == "" :
            print("\n> No grating provided! Checking... ")
            _test_ = KOALA_RSS(rss_list[0], plot_final_rss=False, verbose = False)
            grating = _test_.grating
            print("\n> Reading file",rss_list[0], "the grating is",grating)
                
        CONFIG_FILE, description_, fits_file_, response_file_, absolute_flux_file_, list_of_telluric_ranges_ =  get_calibration_star_data (star, path_star, grating, pk)

        if description == "" : description = description_
        if fits_file == "" : fits_file = fits_file_
        if response_file == "" : response_file = response_file_
        if absolute_flux_file == "" : absolute_flux_file = absolute_flux_file_
        if list_of_telluric_ranges == "" : list_of_telluric_ranges = list_of_telluric_ranges_


        # Check if folder has throughput if not given
        if throughput_2D_file == "":
            print("\n> No throughout file provided, using default file:")
            throughput_2D_file = path_star+"throughput_2D_"+date+"_"+grating+".fits" 
            print("  ",throughput_2D_file)
        

# # Read configuration file
       
    #print("\n  CONFIG FILE: ",CONFIG_FILE)
    config_property, config_value = read_table(CONFIG_FILE, ["s", "s"] )
    
    if object_auto == "" : 
        print("\n> Reading configuration file", CONFIG_FILE,"...\n")
        if obj_name =="":
            object_auto = star+"_"+grating+"_"+date
        else:
            object_auto = obj_name
       
    for i in range(len(config_property)):
        
        if  config_property[i] == "pixel_size" : 	 pixel_size = float(config_value[i]) 
        if  config_property[i] == "kernel_size" : 	 
            kernel_size = float(config_value[i])
            pk = "_"+str(int(pixel_size))+"p"+str(int((abs(pixel_size)-abs(int(pixel_size)))*10))+"_"+str(int(kernel_size))+"k"+str(int(abs(kernel_size*100))-int(kernel_size)*100)
        if  config_property[i] == "date" : 	 date = config_value[i]
        if  config_property[i] == "grating" : 	 grating = config_value[i]
        
        if  config_property[i] == "path_star" :  
            path_star = config_value[i] 
            if path_star[-1] != "/" : path_star =path_star +"/"
        if  config_property[i] == "obj_name" :  object_auto = config_value[i]        
        if  config_property[i] == "star" : 
            star = config_value[i]
            _CONFIG_FILE_, description, fits_file, response_file, absolute_flux_file, list_of_telluric_ranges =  get_calibration_star_data (star, path_star, grating, pk)

            

        if  config_property[i] == "description" :  description = config_value[i]
        if  config_property[i] == "fits_file"   :  fits_file = full_path(config_value[i],path_star)
        if  config_property[i] == "response_file" :  response_file = full_path(config_value[i],path_star)
        if  config_property[i] == "telluric_file" :  telluric_file = full_path(config_value[i],path_star)
        
        if  config_property[i] == "rss" : rss_list.append(full_path(config_value[i],path_star))  #list_of_files_of_stars
        if  config_property[i] == "reduced" :
            if config_value[i] == "True" :  reduced = True 
        
        if  config_property[i] == "read_cube" :
            if config_value[i] == "True" : read_fits_cube = True 
                    
        # RSS Section -----------------------------

        if  config_property[i] == "rss_clean" :
            if config_value[i] == "True" : 
                rss_clean = True 
            else: rss_clean = False 
            
            if rss_clean_given  == True: rss_clean = True

        if  config_property[i] == "save_rss" :
            if config_value[i] == "True" :   save_rss = True
        
        if  config_property[i] == "apply_throughput" :
            if config_value[i] == "True" : 
                apply_throughput = True 
            else: apply_throughput = False                                                 
        if  config_property[i] == "throughput_2D_file" : throughput_2D_file = full_path(config_value[i],path_star)
        if  config_property[i] == "throughput_2D" : throughput_2D_variable =config_value[i] # full_path(config_value[i],path_star)
        
        if  config_property[i] == "correct_ccd_defects" :
            if config_value[i] == "True" : 
                correct_ccd_defects = True 
            else: correct_ccd_defects = False 
        if  config_property[i] == "fix_wavelengths":
            if config_value[i] == "True" : fix_wavelengths = True 
        if  config_property[i] == "sol" :
            sol_ = config_value[i].strip('][').split(',')
            if float(sol_[0]) == -1:
                sol = [float(sol_[0])]
            else:
                if float(sol_[0]) != -0: sol = [float(sol_[0]),float(sol_[1]),float(sol_[2])]

        if  config_property[i] == "throughput_2D_wavecor" :
            if config_value[i] == "True" : 
                throughput_2D_wavecor = True 
            else: throughput_2D_wavecor = False      

        if  config_property[i] == "do_extinction":
            if config_value[i] == "True" : 
                do_extinction = True 
            else: do_extinction = False 
            
        if  config_property[i] == "sky_method" : sky_method = config_value[i]
        if  config_property[i] == "n_sky" : n_sky=int(config_value[i])
        if  config_property[i] == "win_sky" : win_sky =  int(config_value[i])
        if  config_property[i] == "remove_5577" : 
            if config_value[i] == "True" : remove_5577 = True 
        if  config_property[i] == "correct_negative_sky" :
            if config_value[i] == "True" : correct_negative_sky = True 

        if  config_property[i] == "order_fit_negative_sky" : order_fit_negative_sky =  int(config_value[i])
        if  config_property[i] == "kernel_negative_sky" : kernel_negative_sky =  int(config_value[i])            
        if  config_property[i] == "individual_check" :
            if config_value[i] == "True" : 
                individual_check = True 
            else: individual_check = False 
        if  config_property[i] == "use_fit_for_negative_sky" :
            if config_value[i] == "True" : 
                use_fit_for_negative_sky = True 
            else: use_fit_for_negative_sky = False  

        if  config_property[i] == "force_sky_fibres_to_zero" :
            if config_value[i] == "True" : 
                force_sky_fibres_to_zero = True 
            else: force_sky_fibres_to_zero = False 
        if  config_property[i] == "high_fibres" : high_fibres =  int(config_value[i])
        if  config_property[i] == "low_fibres" : low_fibres =  int(config_value[i])

        if config_property[i] == "sky_fibres" :  
            sky_fibres_ =  config_value[i]
            if sky_fibres_ == "fibres_best_sky_100":
                sky_fibres = fibres_best_sky_100
                sky_fibres_print =  "fibres_best_sky_100"
            else:
                if sky_fibres_[0:5] == "range":
                    sky_fibres_ = sky_fibres_[6:-1].split(',')
                    sky_fibres = list(range(np.int(sky_fibres_[0]),np.int(sky_fibres_[1])))
                    sky_fibres_print = "range("+sky_fibres_[0]+","+sky_fibres_[1]+")"
                else:
                    sky_fibres_ = config_value[i].strip('][').split(',')
                    for i in range(len(sky_fibres_)):
                        sky_fibres.append(float(sky_fibres_[i]))                    
                    sky_fibres_print =  sky_fibres  

        if  config_property[i] == "remove_negative_median_values":
            if config_value[i] == "True" : 
                remove_negative_median_values = True 
            else: remove_negative_median_values = False 

        if  config_property[i] == "fix_edges" and  config_value[i] == "True" : fix_edges = True 
        if  config_property[i] == "clean_extreme_negatives" :
            if config_value[i] == "True" : clean_extreme_negatives = True 
        if  config_property[i] == "percentile_min" : percentile_min = float(config_value[i])  

        if  config_property[i] == "clean_cosmics" and config_value[i] == "True" : clean_cosmics = True 
        if  config_property[i] == "width_bl" : width_bl = float(config_value[i])  
        if  config_property[i] == "kernel_median_cosmics" : kernel_median_cosmics = int(config_value[i])  
        if  config_property[i] == "cosmic_higher_than" : cosmic_higher_than = float(config_value[i])  
        if  config_property[i] == "extra_factor" : extra_factor = float(config_value[i])  
          
        # Cubing Section ------------------------------

        if  config_property[i] == "do_cubing" : 
            if config_value[i] == "False" :  
                do_cubing = False 
                do_alignment = False
                make_combined_cube = False  # LOki
                
        if  config_property[i] == "size_arcsec" :     
            size_arcsec_ = config_value[i].strip('][').split(',')
            for i in range(len(size_arcsec_)):
                size_arcsec.append(float(size_arcsec_[i]))         
        
        if  config_property[i] == "edgelow" : edgelow =  int(config_value[i])    
        if  config_property[i] == "edgehigh" : edgehigh =  int(config_value[i]) 

        if  config_property[i] == "ADR" and config_value[i] == "True" : ADR = True 
        if  config_property[i] == "adr_index_fit": adr_index_fit = int(config_value[i])
        if  config_property[i] == "g2d": 
            if config_value[i] == "True" : 
                g2d = True
            else: g2d = False
            
        if  config_property[i] == "jump": jump = int(config_value[i])
        
        if  config_property[i] == "trim_cube" : 
            if config_value[i] == "True" : 
                trim_cube = True 
            else: trim_cube = False 
            
        if  config_property[i] == "trim_values" :     
            trim_values_ = config_value[i].strip('][').split(',')
            trim_values = [int(trim_values_[0]),int(trim_values_[1]),int(trim_values_[2]),int(trim_values_[3])]           
            
        if  config_property[i] == "scale_cubes_using_integflux" : 
            if config_value[i] == "True" : 
                scale_cubes_using_integflux = True 
            else: scale_cubes_using_integflux = False 
 
        if  config_property[i] == "flux_ratios" :
            flux_ratios_ = config_value[i].strip('][').split(',')
            flux_ratios = []
            for i in range(len(flux_ratios_)):
                flux_ratios.append(float(flux_ratios_[i]))

        # Calibration  ---------------------------------

        if  config_property[i] == "do_calibration" : 
            if config_value[i] == "False" :  do_calibration = False 
            if config_value[i] == "True" :  do_calibration = True 

        if  config_property[i] == "r_max" : r_max = float(config_value[i])

        # CHECK HOW TO OBTAIN TELLURIC CORRECTION !!! 
        if  config_property[i] == "order_telluric" : order_telluric =  int(config_value[i])
        if  config_property[i] == "telluric_range" :           
            if first_telluric_range == True : 
                list_of_telluric_ranges =[]   
                first_telluric_range   = False  
            telluric_range_string = config_value[i].strip('][').split(',')
            telluric_range_float = [float(telluric_range_string[0]),float(telluric_range_string[1]),float(telluric_range_string[2]),float(telluric_range_string[3])]
            list_of_telluric_ranges.append(telluric_range_float)     
                                  
        if  config_property[i] == "apply_tc"  :	
            if config_value[i] == "True" : 
                apply_tc = True 
            else: apply_tc = False

        if  config_property[i] == "absolute_flux_file" : absolute_flux_file = config_value[i]
        if  config_property[i] == "min_wave_flux" : min_wave_flux = float(config_value[i])   
        if  config_property[i] == "max_wave_flux" : max_wave_flux = float(config_value[i])   
        if  config_property[i] == "step_flux" : step_flux = float(config_value[i])   
        if  config_property[i] == "exp_time" : exp_time = float(config_value[i])   
        if  config_property[i] == "fit_degree_flux" : fit_degree_flux = int(config_value[i])
        if  config_property[i] == "ha_width" : ha_width = float(config_value[i])  
                
        if  config_property[i] == "sky_annulus_low_arcsec" : sky_annulus_low_arcsec = float(config_value[i]) 
        if  config_property[i] == "sky_annulus_high_arcsec" : sky_annulus_high_arcsec = float(config_value[i]) 

        if  config_property[i] == "valid_wave_min" : valid_wave_min = float(config_value[i])
        if  config_property[i] == "valid_wave_max" : valid_wave_max = float(config_value[i])
        
        if  config_property[i] == "odd_number" : odd_number = int(config_value[i])
        if  config_property[i] == "smooth" : smooth = float(config_value[i])
        if  config_property[i] == "fit_weight" : fit_weight = float(config_value[i])
        if  config_property[i] == "smooth_weight" : smooth_weight = float(config_value[i])
        
        if  config_property[i] == "exclude_wlm" :       
            exclude_wlm=[]
            exclude_wlm_string_= config_value[i].replace("]","")
            exclude_wlm_string= exclude_wlm_string_.replace("[","").split(',')
            for i in np.arange(0, len(exclude_wlm_string),2) :
                exclude_wlm.append([float(exclude_wlm_string[i]),float(exclude_wlm_string[i+1])])    

        # Plotting, printing ------------------------------
        
        if  config_property[i] == "log" :  
            if config_value[i] == "True" : 
                log = True 
            else: log = False 
        if  config_property[i] == "gamma" : gamma = float(config_value[i])            
        if  config_property[i] == "fig_size" : fig_size = float(config_value[i])
        if  config_property[i] == "plot" : 
            if config_value[i] == "True" : 
                plot = True 
            else: plot = False 
        if  config_property[i] == "plot_rss" : 
            if config_value[i] == "True" : 
                plot_rss = True 
            else: plot_rss = False 
        if  config_property[i] == "plot_weight" : 
            if config_value[i] == "True" : 
                plot_weight = True 
            else: plot_weight = False             
                           
        if  config_property[i] == "warnings" : 
            if config_value[i] == "True" : 
                warnings = True 
            else: warnings = False     
        if  config_property[i] == "verbose" : 
            if config_value[i] == "True" : 
                verbose = True 
            else: verbose = False   

    if throughput_2D_variable != "":  throughput_2D = eval(throughput_2D_variable)

    if do_cubing == False:      
        fits_file = "" 
        make_combined_cube = False  
        do_alignment = False

# # Print the summary of parameters

    print("> Parameters for processing this calibration star :\n")
    print("  star                     = ",star) 
    if object_auto != "" : 
        if reduced == True and read_fits_cube == False :
            print("  Python object            = ",object_auto,"  already created !!")   
        else:
            print("  Python object            = ",object_auto,"  to be created")
    print("  path                     = ",path_star)
    print("  description              = ",description)
    print("  date                     = ",date)
    print("  grating                  = ",grating)
    
    if reduced == False and read_fits_cube == False :  
        for rss in range(len(rss_list)):
            if rss == 0 : 
                if len(rss_list) > 1:
                    print("  rss_list                 = [",rss_list[rss],",")
                else:
                    print("  rss_list                 = [",rss_list[rss],"]")
            else:
                if rss == len(rss_list)-1:
                    print("                              ",rss_list[rss]," ]")
                else:        
                    print("                              ",rss_list[rss],",")     

        if rss_clean:
            print("  rss_clean                =  True, skipping to cubing\n")
        else:  
            if save_rss : print("  'CLEANED' RSS files will be saved automatically")

            if apply_throughput:
                if throughput_2D_variable != "" :    
                    print("  throughput_2D variable   = ",throughput_2D_variable)
                else:
                    if throughput_2D_file != "" : print("  throughput_2D_file       = ",throughput_2D_file)

            if apply_throughput and throughput_2D_wavecor:
                 print("  throughput_2D will be applied AFTER correcting CCD defects and fixing small wavelenghts")

            print("  correct_ccd_defects      = ",correct_ccd_defects)
            print("  fix_wavelengths          = ",fix_wavelengths)
            if fix_wavelengths: 
                if sol[0] == -1:
                    print("    Only using few skylines in the edges")
                else:
                    if sol[0] != -1: print("    sol                    = ",sol)
    
            print("  do_extinction            = ",do_extinction)           
            print("  sky_method               = ",sky_method)     
            if sky_method != "none" :              
                if len(sky_fibres) > 1: 
                    print("    sky_fibres             = ",sky_fibres_print)
                else:
                    print("    n_sky                  = ",n_sky)    
            if win_sky > 0 : print("    win_sky                = ",win_sky)  
            if remove_5577: print("    remove 5577 skyline    = ",remove_5577)
            print("  correct_negative_sky     = ",correct_negative_sky)
            if correct_negative_sky:
                print("    order_fit_negative_sky   = ",order_fit_negative_sky)
                print("    kernel_negative_sky      = ",kernel_negative_sky)
                print("    use_fit_for_negative_sky = ",use_fit_for_negative_sky) 
                print("    low_fibres               = ",low_fibres)
                print("    individual_check         = ",individual_check)  
                if sky_method in ["self" , "selffit"]:  print("    force_sky_fibres_to_zero = ",force_sky_fibres_to_zero)

            if fix_edges: print("  fix_edges                = ",fix_edges)          
 
            print("  clean_cosmics            = ",clean_cosmics)
            if clean_cosmics:
                print("    width_bl               = ",width_bl)
                print("    kernel_median_cosmics  = ",kernel_median_cosmics)
                print("    cosmic_higher_than     = ",cosmic_higher_than)
                print("    extra_factor           = ",extra_factor)
 
            print("  clean_extreme_negatives  = ",clean_extreme_negatives)
            if clean_extreme_negatives:
                print("    percentile_min         = ",percentile_min)    
            if remove_negative_median_values:
                print("  Negative pixels will be set to 0 when median value of spectrum is negative")
                
        if valid_wave_min > 0 : print("  valid_wave_min           = ",valid_wave_min,"A")
        if valid_wave_max > 0 : print("  valid_wave_max           = ",valid_wave_max,"A")

        if do_cubing:
            if len(size_arcsec) > 0: print("  cube_size_arcsec         = ",size_arcsec)    
    
            if edgelow != -1:  print("  edgelow for tracing      = ",edgelow,"pixels")
            if edgehigh != -1: print("  edgehigh for tracing     = ",edgehigh,"pixels")
            print("  2D Gauss for tracing     = ",g2d)
            
            print("  ADR                      = ",ADR)       
            if ADR: print("    adr_index_fit          = ",adr_index_fit)
    
            if jump != -1 :    print("    jump for ADR           = ",jump)
    
            if scale_cubes_using_integflux:
                if len(flux_ratios) == 0 :
                    print("  Scaling individual cubes using integrated flux of common region")
                else:
                    print("  Scaling individual cubes using flux_ratios = ",flux_ratios)
    
            if trim_cube: print("  Trim cube                = ",trim_cube)
            if len(trim_values) != 0: print("    Trim values            = ",trim_values)
        else:      
            print("\n> No cubing will be performed\n")
    
    if do_calibration:    
    
        if read_fits_cube:   
            print("\n> Input fits file with cube:\n ",fits_file,"\n")
        else:      
            print("  pixel_size               = ",pixel_size)
            print("  kernel_size              = ",kernel_size)    
        print("  plot                     = ",plot)
        print("  verbose                  = ",verbose)
    
        print("  warnings                 = ",warnings)
        print("  r_max                    = ",r_max, '" for extracting the star')
        if grating in red_gratings:
            #print "  telluric_file        = ",telluric_file
            print("  Parameters for obtaining the telluric correction:")
            print("    apply_tc               = ", apply_tc)
            print("    order continuum fit    = ",order_telluric)
            print("    telluric ranges        = ",list_of_telluric_ranges[0])
            for i in range(1,len(list_of_telluric_ranges)):
                print("                             ",list_of_telluric_ranges[i])    
        print("  Parameters for obtaining the absolute flux calibration:")    
        print("     absolute_flux_file    = ",absolute_flux_file)
        
        if min_wave_flux == 0 :  min_wave_flux = valid_wave_min
        if max_wave_flux == 0 :  max_wave_flux = valid_wave_max
        
        if min_wave_flux  > 0 : print("     min_wave_flux         = ",min_wave_flux)  
        if max_wave_flux  > 0 : print("     max_wave_flux         = ",max_wave_flux)   
        print("     step_flux             = ",step_flux) 
        if exp_time > 0 : 
            print("     exp_time              = ",exp_time) 
        else:
            print("     exp_time              =  reads it from .fits files")
        print("     fit_degree_flux       = ",fit_degree_flux) 
        print("     sky_annulus_low       = ",sky_annulus_low_arcsec,"arcsec")
        print("     sky_annulus_high      = ",sky_annulus_high_arcsec,"arcsec")
        if ha_width > 0 : print("     ha_width              = ",ha_width,"A")  
        if odd_number > 0 : print("     odd_number            = ",odd_number)  
        if smooth > 0 : print("     smooth                = ",smooth)  
        if fit_weight > 0 : print("     fit_weight            = ",fit_weight)  
        if smooth_weight > 0 :     print("     smooth_weight         = ",smooth_weight)  
        if exclude_wlm[0][0] != 0: print("     exclude_wlm           = ",exclude_wlm)  
        
         
        print("\n> Output files:\n")
        if read_fits_cube == "False" : print("  fits_file            =",fits_file)
        print("  integrated spectrum  =",fits_file[:-5]+"_integrated_star_flux.dat")
        if grating in red_gratings :
            print("  telluric_file        =",telluric_file) 
        print("  response_file        =",response_file)
        print(" ")

    else:
        print("\n> No calibration will be performed\n")

# # Read cube from fits file if given

    if read_fits_cube:  
        star_object = read_cube(fits_file, valid_wave_min = valid_wave_min, valid_wave_max = valid_wave_max)
        reduced = True
        exp_time = np.nanmedian(star_object.exptimes)
        
        print(" ")
        exec(object_auto+"=copy.deepcopy(star_object)", globals())
        print("> Cube saved in object", object_auto," !")


# # Running KOALA_REDUCE using rss_list
    
    if reduced == False:
        
        for rss in rss_list:
            if save_rss :
                save_rss_to_fits_file_list.append("auto")
            else:
                save_rss_to_fits_file_list.append("")
             
        if do_cubing:
            print("> Running KOALA_reduce to create combined datacube...")   
        else:
            print("> Running KOALA_reduce ONLY for processing the RSS files provided...") 
    
        star_object=KOALA_reduce(rss_list,
                           path=path,
                           fits_file=fits_file, 
                           obj_name=star,  
                           description=description,
                           save_rss_to_fits_file_list = save_rss_to_fits_file_list,
                           rss_clean=rss_clean,
                           grating = grating,
                           apply_throughput=apply_throughput, 
                           throughput_2D_file = throughput_2D_file,
                           throughput_2D = throughput_2D,
                           correct_ccd_defects = correct_ccd_defects, 
                           fix_wavelengths = fix_wavelengths, 
                           sol = sol,
                           throughput_2D_wavecor = throughput_2D_wavecor,
                           do_extinction= do_extinction,
                           sky_method=sky_method, 
                           n_sky=n_sky,
                           win_sky=win_sky,
                           remove_5577=remove_5577,
                           sky_fibres=sky_fibres,
                           correct_negative_sky = correct_negative_sky,
                           order_fit_negative_sky =order_fit_negative_sky, 
                           kernel_negative_sky = kernel_negative_sky,
                           individual_check = individual_check, 
                           use_fit_for_negative_sky = use_fit_for_negative_sky,
                           force_sky_fibres_to_zero=force_sky_fibres_to_zero,
                           low_fibres= low_fibres,
                           high_fibres=high_fibres,
                           
                           fix_edges=fix_edges,           
                           clean_extreme_negatives=clean_extreme_negatives, percentile_min=percentile_min,
                           remove_negative_median_values=remove_negative_median_values,
                           clean_cosmics = clean_cosmics,
                           width_bl = width_bl, kernel_median_cosmics = kernel_median_cosmics, 
                           cosmic_higher_than = cosmic_higher_than, extra_factor = extra_factor,
                                                    
                           do_cubing = do_cubing, do_alignment=do_alignment, make_combined_cube=make_combined_cube,
                           pixel_size_arcsec=pixel_size, 
                           kernel_size_arcsec=kernel_size, 
                           size_arcsec=size_arcsec,
                           edgelow = edgelow, edgehigh = edgehigh,
                           ADR= ADR,
                           adr_index_fit=adr_index_fit, g2d=g2d,
                           jump=jump,
                           box_x=box_x, box_y=box_y,
                           trim_values=trim_values,
                           scale_cubes_using_integflux = scale_cubes_using_integflux,
                           flux_ratios = flux_ratios,
                           valid_wave_min = valid_wave_min, 
                           valid_wave_max = valid_wave_max,
                           log=log,
                           gamma = gamma,
                           plot= plot, 
                           plot_rss=plot_rss,
                           plot_weight=plot_weight,
                           fig_size=fig_size,
                           verbose = verbose,
                           warnings=warnings ) 
    
        # Save object is given
        if object_auto != 0: # and make_combined_cube == True:
            exec(object_auto+"=copy.deepcopy(star_object)", globals())
            print("> Cube saved in object", object_auto," !")

    else:
        if read_fits_cube == False:
            print("> Python object",object_auto,"already created.")
            exec("star_object=copy.deepcopy("+object_auto+")", globals())


    #  Perform the calibration

    if do_calibration :

        # Check exposition times
        if exp_time == 0:
            different_times = False
            try:
                exptimes = star_object.combined_cube.exptimes
            except Exception:    
                exptimes = star_object.exptimes
            
            exp_time1 = exptimes[0]
            print("\n> Exposition time reading from rss1: ",exp_time1," s")
            
            exp_time_list = [exp_time1]
            for i in range(1,len(exptimes)):
                exp_time_n = exptimes [i]
                exp_time_list.append(exp_time_n)
                if exp_time_n != exp_time1:
                    print("  Exposition time reading from rss"+np.str(i)," = ", exp_time_n," s")
                    different_times = True
            
            if  different_times:
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
        if grating in red_gratings :
    
            # Extract the integrated spectrum of the star & save it
    
            print("\n> Extracting the integrated spectrum of the star...")
           
            star_cube.half_light_spectrum(r_max=r_max, plot=plot)
            spectrum_to_text_file(star_cube.wavelength,
                              star_cube.integrated_star_flux, 
                              filename=fits_file[:-5]+"_integrated_star_flux_before_TC.dat")
        
           
    
     # Find telluric correction CAREFUL WITH apply_tc=True
    
            print("\n> Finding telluric correction...")
            try:                         
                telluric_correction_star = telluric_correction_from_star(star_object,
                                                                         list_of_telluric_ranges = list_of_telluric_ranges,
                                                                         order = order_telluric,
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
                    if object_auto != 0: exec(object_auto+"=copy.deepcopy(star_object)", globals())
        
            except Exception:  
                print("\n> Finding telluric correction FAILED!")
    
    
     #Flux calibration
        
        print("\n> Finding absolute flux calibration...")
        
     # Now we read the absolute flux calibration data of the calibration star and get the response curve
     # (Response curve: correspondence between counts and physical values)
     # Include exp_time of the calibration star, as the results are given per second
     # For this BE CAREFUL WITH ABSORPTIONS (Halpha) and check behaviour in the edges of the CCD
     # Change fit_degree (3,5,7), step, min_wave, max_wave to get better fits !!!  
       
        #try: 
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
                                        exclude_wlm = exclude_wlm,
                                        odd_number=odd_number,
                                        smooth=smooth,
                                        fit_weight=fit_weight,
                                        smooth_weight=smooth_weight)
         
        
            spectrum_to_text_file(star_cube.wavelength,
                              star_cube.integrated_star_flux, 
                              filename=fits_file[:-5]+"_integrated_star_flux.dat")
        
         # Now we can save this calibration as a text file 
        
            spectrum_to_text_file(star_cube.wavelength,
                                  star_cube.response_curve, 
                                  filename=response_file, verbose = False)
            
            print('\n> Absolute flux calibration (response) saved in text file :\n  "'+response_file+'"')
            
            if object_auto != 0: exec(object_auto+"=copy.deepcopy(star_object)", globals())
                
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
    cube_list=[]
    cube_list_names = []
    save_aligned_cubes = False

    apply_throughput=False
    throughput_2D = []
    throughput_2D_file = ""
    throughput_2D_wavecor = False

    correct_ccd_defects = False
    kernel_correct_ccd_defects=51

    fix_wavelengths = False
    sol =[0,0,0]
    do_extinction = False

    do_telluric_correction = False
        
    sky_method = "none"
    sky_spectrum = []
    sky_spectrum_name = ""
 #   sky_spectrum_file = ""   #### NEEDS TO BE IMPLEMENTED
    sky_list = []    
    sky_fibres  = [1000]         
    sky_lines_file = ""

    scale_sky_1D = 1.
    auto_scale_sky 	=	False
    n_sky = 50 
    print_n_sky = False
    win_sky = 0
    remove_5577 = False    

    correct_negative_sky = False
    order_fit_negative_sky =3 
    kernel_negative_sky = 51
    individual_check = True
    use_fit_for_negative_sky = False
    force_sky_fibres_to_zero = True
    high_fibres=20
    low_fibres=10
    
    brightest_line="Ha"
    brightest_line_wavelength=0.      
    ranges_with_emission_lines = [0]
    cut_red_end = 0			
    
    id_el=False
    id_list=[0]
    cut=1.5
    broad=1.8
    plot_id_el=False
    
    clean_sky_residuals = False
    features_to_fix =[]
    sky_fibres_for_residuals = []
    sky_fibres_for_residuals_print = "Using the same n_sky fibres"

    remove_negative_median_values = False    
    fix_edges=False
    clean_extreme_negatives = False
    percentile_min=0.9  

    clean_cosmics = False
    width_bl = 20.
    kernel_median_cosmics = 5 
    cosmic_higher_than 	=	100. 
    extra_factor =	1.

    offsets=[]
    ADR = False
    ADR_cc = False
    force_ADR=False
    box_x=[]
    box_y=[]
    jump=-1
    half_size_for_centroid = 10
    ADR_x_fit_list = []
    ADR_y_fit_list = []
    adr_index_fit=2
    g2d = False
    step_tracing = 100
    plot_tracing_maps=[]
    edgelow  = -1
    edgehigh = -1


    delta_RA  = 0
    delta_DEC = 0

    trim_cube = False
    trim_values=[]
    size_arcsec=[]
    centre_deg =[]
    scale_cubes_using_integflux = True
    remove_spaxels_not_fully_covered = True
    flux_ratios =[]

    valid_wave_min=0 
    valid_wave_max=0 
            
    plot = True
    plot_rss = True
    plot_weight = False
    plot_spectra = True
    fig_size	=12.

    log = True
    gamma = 0.

    warnings	=False
    verbose=True    
              
    if path != "" : KOALA_REDUCE_FILE = full_path(KOALA_REDUCE_FILE, path)   # VR
    config_property, config_value = read_table(KOALA_REDUCE_FILE, ["s", "s"] )
    
    print("\n> Reading configuration file", KOALA_REDUCE_FILE,"...\n")
           
    for i in range(len(config_property)):
        
        if  config_property[i] == "pixel_size" : 	 pixel_size = float(config_value[i])         
        if  config_property[i] == "kernel_size" : 	 
            kernel_size = float(config_value[i])
            pk = "_"+str(int(pixel_size))+"p"+str(int((abs(pixel_size)-abs(int(pixel_size)))*10))+"_"+str(int(kernel_size))+"k"+str(int(abs(kernel_size*100))-int(kernel_size)*100)
        if  config_property[i] == "date" : 	 date = config_value[i]
        if  config_property[i] == "grating" : 	 grating = config_value[i]

        if  config_property[i] == "path" : 	 
            path = config_value[i]
       
        if  config_property[i] == "obj_name" : 
            obj_name = config_value[i]
            description = obj_name
            fits_file = path+obj_name+"_"+grating+pk+".fits"            
            Python_obj_name = obj_name + "_" + grating+pk
        if  config_property[i] == "description" :  description = config_value[i]       
        if  config_property[i] == "Python_obj_name" : Python_obj_name = config_value[i]
    
        if  config_property[i] == "flux_calibration_file" :
            flux_calibration_name = ""
            flux_calibration_file_  = config_value[i]
            if len(flux_calibration_file_.split("/")) == 1:
                flux_calibration_file = path+flux_calibration_file_
            else:
                flux_calibration_file = flux_calibration_file_                     
            flux_calibration_file_list.append(flux_calibration_file)
        if  config_property[i] == "telluric_correction_file" : 
            telluric_correction_name = ""
            telluric_correction_file_  = config_value[i]   
            if len(telluric_correction_file_.split("/")) == 1:
                telluric_correction_file = path+telluric_correction_file_
            else:
                telluric_correction_file = telluric_correction_file_               
            telluric_correction_list_.append(config_value[i])

        if  config_property[i] == "flux_calibration_name" :  
            flux_calibration_name = config_value[i]
            #flux_calibration_name_list.append(flux_calibration_name)
            
        if  config_property[i] == "telluric_correction_name" :  
            telluric_correction_name = config_value[i]

        if  config_property[i] == "fits_file"   :  
            fits_file_ = config_value[i]
            if len(fits_file_.split("/")) == 1:
                fits_file = path+fits_file_
            else:
                fits_file = fits_file_
            
        if  config_property[i] == "save_aligned_cubes" :
            if config_value[i] == "True" : save_aligned_cubes = True 
        
        if  config_property[i] == "rss_file" : 
            rss_file_ = config_value[i]
            if len(rss_file_.split("/")) == 1:
                _rss_file_ = path+rss_file_
            else:
                _rss_file_ = rss_file_    
            rss_list.append(_rss_file_)
        if  config_property[i] == "cube_file" : 
            cube_file_ = config_value[i]
            if len(cube_file_.split("/")) == 1:
                _cube_file_ = path+cube_file_
            else:
                _cube_file_ = cube_file_    
            cube_list_names.append(_cube_file_)
            cube_list.append(_cube_file_)   # I am not sure about this... 
            
        if  config_property[i] == "rss_clean" :
            if config_value[i] == "True" : 
                rss_clean = True 
            else: rss_clean = False          
        if  config_property[i] == "save_rss" :
            if config_value[i] == "True" : 
                save_rss = True 
            else: save_rss = False  
        if  config_property[i] == "do_cubing" and  config_value[i] == "False" :  do_cubing = False 

        if  config_property[i] == "apply_throughput" :
            if config_value[i] == "True" : 
                apply_throughput = True 
            else: apply_throughput = False              

        if  config_property[i] == "throughput_2D_file" : 
            throughput_2D_file_ = config_value[i]
            if len(throughput_2D_file_.split("/")) == 1:
                throughput_2D_file = path+throughput_2D_file_
            else:
                throughput_2D_file = throughput_2D_file_  
        
        if  config_property[i] == "throughput_2D" : throughput_2D_variable = config_value[i]

        if  config_property[i] == "throughput_2D_wavecor" :
            if config_value[i] == "True" : 
                throughput_2D_wavecor = True 
            else: throughput_2D_wavecor = False  

        if  config_property[i] == "correct_ccd_defects" :
            if config_value[i] == "True" : 
                correct_ccd_defects = True 
            else: correct_ccd_defects = False  
        if  config_property[i] == "kernel_correct_ccd_defects" : 	 kernel_correct_ccd_defects = float(config_value[i])     
                
        if  config_property[i] == "fix_wavelengths":
            if config_value[i] == "True" : 
                fix_wavelengths = True 
            else: fix_wavelengths = False
        if  config_property[i] == "sol" :
            sol_ = config_value[i].strip('][').split(',')
            if float(sol_[0]) == -1:
                sol = [float(sol_[0])]
            else:
                sol = [float(sol_[0]),float(sol_[1]),float(sol_[2])]
            
        if  config_property[i] == "do_extinction":
            if config_value[i] == "True" : 
                do_extinction = True 
            else: do_extinction = False            
            
        if  config_property[i] == "sky_method" : sky_method = config_value[i]
                
        if  config_property[i] == "sky_file" : sky_list.append(path+config_value[i])            
        if  config_property[i] == "n_sky" : n_sky=int(config_value[i])

        if config_property[i] == "sky_fibres" :  
            sky_fibres_ =  config_value[i]
            if sky_fibres_[0:5] == "range":
                sky_fibres_ = sky_fibres_[6:-1].split(',')
                sky_fibres = list(range(np.int(sky_fibres_[0]),np.int(sky_fibres_[1])))
                sky_fibres_print = "range("+sky_fibres_[0]+","+sky_fibres_[1]+")"
            else:
                sky_fibres_ = config_value[i].strip('][').split(',')
                for i in range(len(sky_fibres_)):
                    sky_fibres.append(float(sky_fibres_[i]))                    
                sky_fibres_print =  sky_fibres  

        if  config_property[i] == "win_sky" : win_sky =  int(config_value[i])

        if  config_property[i] == "sky_spectrum" :
            if config_value[i] != "[0]" :
                sky_spectrum_name = config_value[i]
                exec("sky_spectrum ="+sky_spectrum_name)
            else:
                sky_spectrum = []
        if  config_property[i] == "scale_sky_1D" : 	 scale_sky_1D = float(config_value[i]) 

        if  config_property[i] == "auto_scale_sky" :
            if config_value[i] == "True" : 
                auto_scale_sky = True 
            else: auto_scale_sky = False  

        if  config_property[i] == "sky_lines_file" : sky_lines_file = config_value[i]

        if  config_property[i] == "correct_negative_sky" :
            if config_value[i] == "True" : 
                correct_negative_sky = True 
            else: correct_negative_sky = False 
 
        if  config_property[i] == "order_fit_negative_sky" : order_fit_negative_sky =  int(config_value[i])
        if  config_property[i] == "kernel_negative_sky" : kernel_negative_sky =  int(config_value[i])            
        if  config_property[i] == "individual_check" :
            if config_value[i] == "True" : 
                individual_check = True 
            else: individual_check = False 
        if  config_property[i] == "use_fit_for_negative_sky" :
            if config_value[i] == "True" : 
                use_fit_for_negative_sky = True 
            else: use_fit_for_negative_sky = False  
        if  config_property[i] == "force_sky_fibres_to_zero" :
            if config_value[i] == "True" : 
                force_sky_fibres_to_zero = True 
            else: force_sky_fibres_to_zero = False 
        if  config_property[i] == "high_fibres" : high_fibres =  int(config_value[i])
        if  config_property[i] == "low_fibres" : low_fibres =  int(config_value[i])
                
        if  config_property[i] == "remove_5577" :
            if config_value[i] == "True" : 
                remove_5577 = True 
            else: remove_5577 = False             

        if  config_property[i] == "do_telluric_correction" :
            if config_value[i] == "True" : 
                do_telluric_correction = True 
            else: 
                do_telluric_correction = False  
                telluric_correction_name = ""
                telluric_correction_file = ""
            
        if  config_property[i] == "brightest_line" :  brightest_line = config_value[i]
        if  config_property[i] == "brightest_line_wavelength" : brightest_line_wavelength = float(config_value[i])
             
        if config_property[i] == "ranges_with_emission_lines":
            ranges_with_emission_lines_ = config_value[i].strip('[]').replace('],[', ',').split(',')
            ranges_with_emission_lines=[]
            for i in range(len(ranges_with_emission_lines_)):
                if i % 2 == 0 : ranges_with_emission_lines.append([float(ranges_with_emission_lines_[i]),float(ranges_with_emission_lines_[i+1])])                       
        if  config_property[i] == "cut_red_end" :  cut_red_end = config_value[i]

        # CHECK id_el
        if  config_property[i] == "id_el" : 
            if config_value[i] == "True" : 
                id_el = True 
            else: id_el = False 
        if  config_property[i] == "cut" : cut = float(config_value[i])
        if  config_property[i] == "broad" : broad = float(config_value[i])
        if  config_property[i] == "id_list":
            id_list_ = config_value[i].strip('][').split(',')
            for i in range(len(id_list_)):
                id_list.append(float(id_list_[i]))               
        if  config_property[i] == "plot_id_el" : 
            if config_value[i] == "True" : 
                plot_id_el = True 
            else: plot_id_el = False 

        if  config_property[i] == "clean_sky_residuals" and  config_value[i] == "True" : clean_sky_residuals = True 
        if  config_property[i] == "fix_edges" and  config_value[i] == "True" : fix_edges = True 

        if  config_property[i] == "feature_to_fix" :     
            feature_to_fix_ = config_value[i]  #.strip('][').split(',')
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
        
        if config_property[i] == "sky_fibres_for_residuals" :  
            sky_fibres_for_residuals_ =  config_value[i]
            if sky_fibres_for_residuals_[0:5] == "range":
                sky_fibres_for_residuals_ = sky_fibres_for_residuals_[6:-1].split(',')
                sky_fibres_for_residuals = list(range(np.int(sky_fibres_for_residuals_[0]),np.int(sky_fibres_for_residuals_[1])))
                sky_fibres_for_residuals_print = "range("+sky_fibres_for_residuals_[0]+","+sky_fibres_for_residuals_[1]+")"

            else:
                sky_fibres_for_residuals_ = config_value[i].strip('][').split(',')
                for i in range(len(sky_fibres_for_residuals_)):
                    sky_fibres_for_residuals.append(float(sky_fibres_for_residuals_[i]))                    
                sky_fibres_for_residuals_print =   sky_fibres_for_residuals  

        if  config_property[i] == "clean_cosmics" and config_value[i] == "True" : clean_cosmics = True 
        if  config_property[i] == "width_bl" : width_bl = float(config_value[i])  
        if  config_property[i] == "kernel_median_cosmics" : kernel_median_cosmics = int(config_value[i])  
        if  config_property[i] == "cosmic_higher_than" : cosmic_higher_than = float(config_value[i])  
        if  config_property[i] == "extra_factor" : extra_factor = float(config_value[i])  

        if  config_property[i] == "clean_extreme_negatives" :
            if config_value[i] == "True" : clean_extreme_negatives = True 
        if  config_property[i] == "percentile_min" : percentile_min = float(config_value[i])  

        if  config_property[i] == "remove_negative_median_values":
            if config_value[i] == "True" : 
                remove_negative_median_values = True 
            else: remove_negative_median_values = False 
     
        if  config_property[i] == "read_cube" : 
            if config_value[i] == "True" : 
                read_cube = True 
            else: read_cube = False   
   
        if  config_property[i] == "offsets" :     
            offsets_ = config_value[i].strip('][').split(',')
            for i in range(len(offsets_)):
                offsets.append(float(offsets_[i]))
                   
        if  config_property[i] == "valid_wave_min" : valid_wave_min = float(config_value[i])
        if  config_property[i] == "valid_wave_max" : valid_wave_max = float(config_value[i])


        if  config_property[i] == "half_size_for_centroid": half_size_for_centroid = int(config_value[i])

        if  config_property[i] == "box_x" :     
            box_x_ = config_value[i].strip('][').split(',')
            for i in range(len(box_x_)):
                box_x.append(int(box_x_[i]))            
        if  config_property[i] == "box_y" :     
            box_y_ = config_value[i].strip('][').split(',')
            for i in range(len(box_y_)):
                box_y.append(int(box_y_[i]))         

        if  config_property[i] == "adr_index_fit": adr_index_fit = int(config_value[i])
        if  config_property[i] == "g2d": 
            if config_value[i] == "True" : 
                g2d = True
            else: g2d = False
        if  config_property[i] == "step_tracing" : step_tracing =  int(config_value[i])

        if  config_property[i] == "plot_tracing_maps" :
            plot_tracing_maps_ = config_value[i].strip('][').split(',')
            for i in range(len(plot_tracing_maps_)):
                plot_tracing_maps.append(float(plot_tracing_maps_[i]))

        if  config_property[i] == "edgelow" : edgelow = int(config_value[i])
        if  config_property[i] == "edgehigh" : edgehigh = int(config_value[i])
        
        if  config_property[i] == "ADR" : 
            if config_value[i] == "True" : 
                ADR = True 
            else: ADR = False   
        if  config_property[i] == "ADR_cc" : 
            if config_value[i] == "True" : 
                ADR_cc = True 
            else: ADR_cc = False              
        if  config_property[i] == "force_ADR" : 
            if config_value[i] == "True" : 
                force_ADR = True 
            else: force_ADR = False  

        if  config_property[i] == "ADR_x_fit" :     
            ADR_x_fit_ = config_value[i].strip('][').split(',')
            if len(ADR_x_fit_) == 4:
                ADR_x_fit_list.append([float(ADR_x_fit_[0]),float(ADR_x_fit_[1]), float(ADR_x_fit_[2]),  float(ADR_x_fit_[3])])
            else:
                ADR_x_fit_list.append([float(ADR_x_fit_[0]),float(ADR_x_fit_[1]), float(ADR_x_fit_[2])])
            
        if  config_property[i] == "ADR_y_fit" :     
            ADR_y_fit_ = config_value[i].strip('][').split(',')
            if len(ADR_y_fit_) == 4:
                ADR_y_fit_list.append([float(ADR_y_fit_[0]),float(ADR_y_fit_[1]), float(ADR_y_fit_[2]),  float(ADR_y_fit_[3])])
            else:
                ADR_y_fit_list.append([float(ADR_y_fit_[0]),float(ADR_y_fit_[1]), float(ADR_y_fit_[2])])

        if  config_property[i] == "jump": jump = int(config_value[i])
    
        if  config_property[i] == "size_arcsec" :     
            size_arcsec_ = config_value[i].strip('][').split(',')
            for i in range(len(size_arcsec_)):
                size_arcsec.append(float(size_arcsec_[i]))    

        if  config_property[i] == "centre_deg" :
            centre_deg_ = config_value[i].strip('][').split(',')
            centre_deg = [float(centre_deg_[0]),float(centre_deg_[1])]

        if  config_property[i] == "delta_RA"  : delta_RA = float(config_value[i])
        if  config_property[i] == "delta_DEC" : delta_DEC = float(config_value[i])
         
        if  config_property[i] == "scale_cubes_using_integflux" : 
            if config_value[i] == "True" : 
                scale_cubes_using_integflux = True 
            else: scale_cubes_using_integflux = False 

        if  config_property[i] == "flux_ratios" :
            flux_ratios_ = config_value[i].strip('][').split(',')
            flux_ratios = []
            for i in range(len(flux_ratios_)):
                flux_ratios.append(float(flux_ratios_[i]))
                
        if  config_property[i] == "apply_scale" : 
            if config_value[i] == "True" : 
                apply_scale = True 
            else: apply_scale = False         


        if  config_property[i] == "trim_cube" : 
            if config_value[i] == "True" : 
                trim_cube = True 
            else: trim_cube = False 
            
        if  config_property[i] == "trim_values" :     
            trim_values_ = config_value[i].strip('][').split(',')
            trim_values = [int(trim_values_[0]),int(trim_values_[1]),int(trim_values_[2]),int(trim_values_[3])]
            
        if  config_property[i] == "remove_spaxels_not_fully_covered" : 
            if config_value[i] == "True" : 
                remove_spaxels_not_fully_covered = True 
            else: remove_spaxels_not_fully_covered = False 
            
        
        if  config_property[i] == "log" :  
            if config_value[i] == "True" : 
                log = True 
            else: log = False 

        if  config_property[i] == "gamma" : gamma = float(config_value[i])
            
        if  config_property[i] == "fig_size" : fig_size = float(config_value[i])

        if  config_property[i] == "plot" : 
            if config_value[i] == "True" : 
                plot = True 
            else: plot = False 
        if  config_property[i] == "plot_rss" : 
            if config_value[i] == "True" : 
                plot_rss = True 
            else: plot_rss = False 
        if  config_property[i] == "plot_weight" : 
            if config_value[i] == "True" : 
                plot_weight = True 
            else: plot_weight = False             
                           
        if  config_property[i] == "warnings" : 
            if config_value[i] == "True" : 
                warnings = True 
            else: warnings = False     
        if  config_property[i] == "verbose" : 
            if config_value[i] == "True" : 
                verbose = True 
            else: verbose = False             

    # Save rss list if requested:
    if save_rss : 
        for i in range(len(rss_list)):
            save_rss_list.append("auto")
 
    if len(cube_list_names) < 1 : cube_list_names=[""]   
 
    # Asign names to variables
    # If files are given, they have preference over variables!
    
    
    if telluric_correction_file != "":
        w_star,telluric_correction = read_table(telluric_correction_file, ["f", "f"] )        
    if telluric_correction_name != "" :  
        exec("telluric_correction="+telluric_correction_name)
    else:
        telluric_correction=[0]

 # Check that skyflat, flux and telluric lists are more than 1 element

    if len(skyflat_list) < 2 : skyflat_list=["","","","","","","","","",""]   # CHECK THIS
    

        
    if len(telluric_correction_list_) < 2 : 
        telluric_correction_list=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
    else:
        for i in range(len(telluric_correction_list_)):
            w_star,telluric_correction_ = read_table(telluric_correction_list_[i], ["f", "f"] ) 
            telluric_correction_list.append(telluric_correction_)
  
 # More checks

    if len(box_x) < 1 : box_x = [0,-1]
    if len(box_y) < 1 : box_y = [0,-1]
    
    if do_cubing == False:      
        fits_file = "" 
        save_aligned_cubes = False 
        make_combined_cube = False  
        do_alignment = False

    if throughput_2D_variable != "":
        exec("throughput_2D = "+throughput_2D_variable)
          
 # Print the summary of parameters read with this script
  
    print("> Parameters for processing this object :\n")
    print("  obj_name                 = ",obj_name) 
    print("  description              = ",description)
    print("  path                     = ",path)
    print("  Python_obj_name          = ",Python_obj_name)
    print("  date                     = ",date)
    print("  grating                  = ",grating)
    
    if read_cube == False:   
        for rss in range(len(rss_list)):       
            if len(rss_list) > 1:       
                if rss == 0 : 
                    print("  rss_list                 = [",rss_list[rss],",")
                else:
                    if rss == len(rss_list)-1:
                        print("                              ",rss_list[rss]," ]")
                    else:        
                        print("                              ",rss_list[rss],",")     
            else:
                print("  rss_list                 = [",rss_list[rss],"]")
                
        if rss_clean: 
            print("  rss_clean                = ",rss_clean)
            print("  plot_rss                 = ",plot_rss)
        else:    
            print("  apply_throughput         = ",apply_throughput)          
            if apply_throughput:
                if throughput_2D_variable != "" :    
                    print("    throughput_2D variable = ",throughput_2D_variable)
                else:
                    if throughput_2D_file != "" : 
                        print("    throughput_2D_file     = ",throughput_2D_file)
                    else:
                        print("    Requested but no throughput 2D information provided !!!") 
                if throughput_2D_wavecor:
                    print("    throughput_2D will be applied AFTER correcting CCD defects and fixing small wavelenghts")
    
            
            print("  correct_ccd_defects      = ",correct_ccd_defects) 
            if correct_ccd_defects: print("    kernel_correct_ccd_defects = ",kernel_correct_ccd_defects) 
            
            if fix_wavelengths:
                print("  fix_wavelengths          = ",fix_wavelengths)
                print("    sol                    = ",sol)
        
            print("  do_extinction            = ",do_extinction)
    
            if do_telluric_correction: 
                print("  do_telluric_correction   = ",do_telluric_correction)
            else: 
                if grating == "385R" or grating == "1000R" or grating == "2000R" : 
                    print("  do_telluric_correction   = ",do_telluric_correction)
      
            print("  sky_method               = ",sky_method)
            if sky_method == "1D" or sky_method == "1Dfit" or sky_method == "2D":    
                for sky in range(len(sky_list)):
                    if sky == 0 : 
                        print("    sky_list               = [",sky_list[sky],",")
                    else:
                        if sky == len(sky_list)-1:
                            print("                              ",sky_list[sky]," ]")
                        else:        
                            print("                              ",sky_list[sky],",")     
            if sky_spectrum[0] != -1  and sky_spectrum_name !="" : 
                print("    sky_spectrum_name      = ",sky_spectrum_name)
                if sky_method == "1Dfit" or sky_method == "selffit":
                    print("    ranges_with_emis_lines = ",ranges_with_emission_lines)
                    print("    cut_red_end            = ",cut_red_end)
                
            if sky_method == "1D" or sky_method == "1Dfit" : print("    scale_sky_1D           = ",scale_sky_1D)
            if sky_spectrum[0] == -1 and len(sky_list) == 0 : print_n_sky = True
            if sky_method == "self" or sky_method == "selffit": print_n_sky = True  
            if print_n_sky: 
                if len(sky_fibres) > 1: 
                    print("    sky_fibres             = ",sky_fibres_print)
                else:
                    print("    n_sky                  = ",n_sky)   
    
            if win_sky > 0 : print("    win_sky                = ",win_sky)
            if auto_scale_sky: print("    auto_scale_sky         = ",auto_scale_sky)
            if remove_5577: print("    remove 5577 skyline    = ",remove_5577)
            print("  correct_negative_sky     = ",correct_negative_sky)
            if correct_negative_sky:
                print("    order_fit_negative_sky   = ",order_fit_negative_sky)
                print("    kernel_negative_sky      = ",kernel_negative_sky)
                print("    use_fit_for_negative_sky = ",use_fit_for_negative_sky) 
                print("    low_fibres               = ",low_fibres)
                print("    individual_check         = ",individual_check)  
                if sky_method in ["self" , "selffit"]:  print("    force_sky_fibres_to_zero = ",force_sky_fibres_to_zero)
                                 
            if sky_method == "1Dfit" or sky_method == "selffit" or id_el == True:
                if sky_lines_file != "": print("    sky_lines_file         = ",sky_lines_file)
                print("    brightest_line         = ",brightest_line)
                print("    brightest_line_wav     = ",brightest_line_wavelength)
          
            
            if  id_el == True:    # NEED TO BE CHECKED
                print("  id_el                = ",id_el)
                print("    high_fibres            = ",high_fibres)
                print("    cut                    = ",cut)
                print("    broad                  = ",broad)
                print("    id_list                = ",id_list)
                print("    plot_id_el             = ",plot_id_el)
                
            if fix_edges: print("  fix_edges                = ",fix_edges)          
            print("  clean_sky_residuals      = ",clean_sky_residuals)
            if clean_sky_residuals:
                if len(features_to_fix) > 0:
                    for feature in features_to_fix:
                        print("    feature_to_fix         = ",feature)
                else:
                    print("    No list with features_to_fix provided, using default list")
                print("    sky_fibres_residuals   = ",sky_fibres_for_residuals_print)    
                       
            print("  clean_cosmics            = ",clean_cosmics)
            if clean_cosmics:
                print("    width_bl               = ",width_bl)
                print("    kernel_median_cosmics  = ",kernel_median_cosmics)
                print("    cosmic_higher_than     = ",cosmic_higher_than)
                print("    extra_factor           = ",extra_factor)
 
            print("  clean_extreme_negatives  = ",clean_extreme_negatives)
            if clean_extreme_negatives:
                print("    percentile_min         = ",percentile_min)    
            if remove_negative_median_values:
                print("  Negative pixels will be set to 0 when median value of spectrum is negative")
        
        if do_cubing:
            print(" ")
            print("  pixel_size               = ",pixel_size)
            print("  kernel_size              = ",kernel_size) 

            if len(size_arcsec) > 0:  print("  cube_size_arcsec         = ",size_arcsec)
            if len(centre_deg) > 0 :  print("  centre_deg               = ",centre_deg)

            if len(offsets) > 0 :
                print("  offsets                  = ",offsets)
            else:
                print("  offsets will be calculated automatically")


            if half_size_for_centroid > 0 : print("  half_size_for_centroid   = ",half_size_for_centroid)
            if np.nanmedian(box_x+box_y) != -0.5: print("  box_x, box_y             = ", box_x, box_y)
            print("  adr_index_fit            = ",adr_index_fit)
            print("  2D Gauss for tracing     = ",g2d)
            print("  step_tracing             = ",step_tracing)
            if len(plot_tracing_maps) > 0 : 
                print("  plot_tracing_maps        = ",plot_tracing_maps)

            if edgelow != -1: print("  edgelow for tracing      = ",edgelow)
            if edgehigh != -1:print("  edgehigh for tracing     = ",edgehigh)
            
            
            print("  ADR                      = ",ADR)
            print("  ADR in combined cube     = ",ADR_cc)
            print("  force_ADR                = ",force_ADR)
            if jump != -1 : print("  jump for ADR             = ",jump)        


            
            if len(ADR_x_fit_list)  >  0:
                print("  Fitting solution for correcting ADR provided!")
                for i in range(len(rss_list)):
                    print("                           = ",ADR_x_fit_list[i])
                    print("                           = ",ADR_y_fit_list[i])
            else:
                if ADR: print("    adr_index_fit          = ",adr_index_fit)
            
            if delta_RA+delta_DEC != 0:
                print("  delta_RA                 = ",delta_RA)
                print("  delta_DEC                = ",delta_DEC)
            
            if valid_wave_min > 0 : print("  valid_wave_min           = ",valid_wave_min)
            if valid_wave_max > 0 : print("  valid_wave_max           = ",valid_wave_max)
            if trim_cube: 
                print("  Trim cube                = ",trim_cube)
                print("     remove_spaxels_not_fully_covered = ",remove_spaxels_not_fully_covered)

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
                        print("    flux_calibration_file  = ",flux_calibration_file_list[i])
                else:    
                    if flux_calibration_file != "" : 
                        print("    flux_calibration_file  = ",flux_calibration_file)
                    else:
                        print("  No flux calibration will be applied")
            else:
                print("  Variable with the flux calibration :",flux_calibration_name)
        
        if do_telluric_correction:
            if telluric_correction_name == "":
                if np.nanmedian(telluric_correction_list) != 0:
                    print("  Each rss file has a telluric correction file:")
                    for i in range(len(telluric_correction_list)):                   
                        print("  telluric_correction_file = ",telluric_correction_list_[i])
                else:    
                    print("  telluric_correction_file = ",telluric_correction_file)        
            else:
                print("  Variable with the telluric calibration :",telluric_correction_name)
    
    else:
        print ("\n  List of ALIGNED cubes provided!")
        for cube in range(len(cube_list_names)):       
            if cube == 0 : 
                print("  cube_list                = [",cube_list_names[cube],",")
            else:
                if cube == len(cube_list_names)-1:
                    print("                              ",cube_list_names[cube]," ]")
                else:        
                    print("                              ",cube_list_names[cube],",")     
        
        print("  pixel_size               = ",pixel_size)
        print("  kernel_size              = ",kernel_size)  
        if half_size_for_centroid > 0 : print("  half_size_for_centroid   = ",half_size_for_centroid)
        if np.nanmedian(box_x+box_y) != -0.5: print("  box_x, box_y             = ", box_x, box_y)
        if jump != -1 : print("  jump for ADR             = ",jump)        
        if edgelow != -1: print("  edgelow for tracing      = ",edgelow)
        if edgehigh != -1:print("  edgehigh for tracing     = ",edgehigh)
        print("  ADR in combined cube     = ",ADR_cc)          
        if valid_wave_min > 0 : print("  valid_wave_min           = ",valid_wave_min)
        if valid_wave_max > 0 : print("  valid_wave_max           = ",valid_wave_max)
        if trim_cube: print("  Trim cube                = ",trim_cube)
        make_combined_cube = True



        

    if make_combined_cube:
        if scale_cubes_using_integflux:
            if len(flux_ratios) == 0 :
                print("  Scaling individual cubes using integrated flux of common region")
            else:
                print("  Scaling individual cubes using flux_ratios = ",flux_ratios)


    print("  plot                     = ",plot)
    if do_cubing or make_combined_cube:    
        if plot_weight: print("  plot weights             = ",plot_weight)
    #if norm != "colors.LogNorm()" :  print("  norm                     = ",norm)
    if fig_size != 12. : print("  fig_size                 = ",fig_size)
    print("  warnings                 = ",warnings)
    if verbose == False: print("  verbose                  = ",verbose)    


    print("\n> Output files:\n")
    if fits_file != "" : print("  fits file with combined cube  =  ",fits_file)
    
    if read_cube == False:
        if len(save_rss_list) > 0 and rss_clean == False:
            for rss in range(len(save_rss_list)):
                if save_rss_list[0] != "auto":
                    if rss == 0 : 
                        print("  list of saved rss files       = [",save_rss_list[rss],",")
                    else:
                        if rss == len(save_rss_list)-1:
                            print("                                   ",save_rss_list[rss]," ]")
                        else:        
                            print("                                   ",save_rss_list[rss],",")      
                else: print("  Processed rss files will be saved using automatic naming")
                    
        else:
            save_rss_list = ["","","","","","","","","",""]
        if save_aligned_cubes:
            print("  Individual cubes will be saved as fits files")

     # Last little checks...
        if len(sky_list) == 0: sky_list=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
        if fix_wavelengths == False : sol =[0,0,0]
        if len (ADR_x_fit_list) == 0 :
            for i in range(len(rss_list)): 
                ADR_x_fit_list.append([0])
                ADR_y_fit_list.append([0])
            
      # Values for improving alignment:   # TODO: CHECK THIS!!! Now I think it is wrong!!!
        if delta_RA+delta_DEC != 0:          
            for i in range(len(ADR_x_fit_list)):
                ADR_x_fit_list[i][2] = ADR_x_fit_list[i][2] + (delta_RA/2)
                ADR_y_fit_list[i][2] = ADR_y_fit_list[i][2] + (delta_DEC/2)
       
                
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
    print("\n  Python object created :",Python_obj_name)
    if fits_file != "" : print("  Fits file created     :",fits_file)
    
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
class KOALA_reduce(RSS,Interpolated_cube):                                      # TASK_KOALA_reduce
 
    def __init__(self, rss_list, fits_file="", obj_name="",  description = "", path="",
                 do_rss=True, do_cubing=True, do_alignment=True, make_combined_cube=True, rss_clean=False, 
                 save_aligned_cubes= False, save_rss_to_fits_file_list = [], #["","","","","","","","","",""],  
                 # RSS
                 flat="",
                 grating = "",
                 # skyflat_file is a RSS, skyflat and skyflat_list are the names of objects keeping the relative throughput of skyflats
                 apply_throughput=True,  
                 throughput_2D=[], throughput_2D_file="",
                 throughput_2D_wavecor = False,
                 #nskyflat=True, skyflat = "", skyflat_file ="",throughput_file ="", nskyflat_file="",
                 #skyflat_list=["","","","","","","","","",""], 
                 #This line is needed if doing FLAT when reducing (NOT recommended)
                 #plot_skyflat=False, wave_min_scale=0, wave_max_scale=0, ymin=0, ymax=0,
                 # Correct CCD defects & high cosmics
                 correct_ccd_defects = False, remove_5577 = False, kernel_correct_ccd_defects = 51, plot_suspicious_fibres=False,
                 # Correct for small shofts in wavelength
                 fix_wavelengths = False, sol = [0,0,0],
                 # Correct for extinction
                 do_extinction=True,
                 # Telluric correction                      
                 telluric_correction = [0], telluric_correction_file="",
                 telluric_correction_list=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]],                  	
                 # Sky substraction
                 sky_method="self", n_sky=50, sky_fibres=[], win_sky = 0, 
                 sky_spectrum=[], sky_rss=[0], scale_sky_rss=0, scale_sky_1D=0, 
                 sky_spectrum_file = "", sky_spectrum_file_list = ["","","","","","","","","",""],               
                 sky_list=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]],
                 ranges_with_emission_lines =  [0],
                 cut_red_end = 0,
                 
                 correct_negative_sky = False, 
                 order_fit_negative_sky = 3, kernel_negative_sky = 51, individual_check = True, use_fit_for_negative_sky = False,
                 force_sky_fibres_to_zero = True, high_fibres=20, low_fibres = 10,
                 auto_scale_sky = False,
                 brightest_line="Ha", brightest_line_wavelength = 0, sky_lines_file="", 
                 is_sky=False, sky_wave_min = 0, sky_wave_max =0, cut_sky=5., fmin=1, fmax=10,                  
                 individual_sky_substraction=False, fibre_list=[100,200,300,400,500,600,700,800,900], 
                 # Identify emission lines
                 id_el=False, cut=1.5, plot_id_el=True, broad=2.0, id_list=[0], 
                 # Clean sky residuals                    
                 fibres_to_fix=[],                                     
                 clean_sky_residuals = False, features_to_fix =[], sky_fibres_for_residuals=[],
                 remove_negative_median_values = False,
                 fix_edges = False,
                 clean_extreme_negatives = False, percentile_min = 0.5,
                 clean_cosmics = False, #show_cosmics_identification = True,                                                            
                 width_bl = 20., kernel_median_cosmics = 5, cosmic_higher_than = 100., extra_factor = 1.,                                                          

                 # CUBING
                 pixel_size_arcsec=.4, kernel_size_arcsec=1.2,
                 offsets=[],
                 ADR=False, ADR_cc = False, force_ADR= False,
                 box_x =[0,-1], box_y=[0,-1], jump = -1, half_size_for_centroid = 10,
                 ADR_x_fit_list=[],ADR_y_fit_list=[], adr_index_fit = 2, 
                 g2d = False,
                 plot_tracing_maps =[],
                 step_tracing=100,
                 edgelow=-1, edgehigh=-1,
                 flux_calibration_file="",  # this can be a single file (string) or a list of files (list of strings)
                 flux_calibration=[],       # an array
                 flux_calibration_list=[],  # a list of arrays
                 trim_cube = True, trim_values =[],
                 remove_spaxels_not_fully_covered = True,
                 size_arcsec = [],
                 centre_deg=[],
                 scale_cubes_using_integflux = True,
                 apply_scale=True,
                 flux_ratios = [],
                 cube_list_names=[""],

                 # COMMON TO RSS AND CUBING & PLOTS
                 valid_wave_min = 0, valid_wave_max = 0,
                 log = True	,		# If True and gamma = 0, use colors.LogNorm() [LOG], if False colors.Normalize() [LINEAL] 
                 gamma	= 0,
                 plot= True, plot_rss=True, plot_weight=False, plot_spectra = True, fig_size=12,
                 warnings=False, verbose = False): # norm
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
        sky_rss_list=[]
        pk = "_"+str(int(pixel_size_arcsec))+"p"+str(int((abs(pixel_size_arcsec)-abs(int(pixel_size_arcsec)))*10))+"_"+str(int(kernel_size_arcsec))+"k"+str(int(abs(kernel_size_arcsec*100))-int(kernel_size_arcsec)*100)

        if plot == False:
            plot_rss=False
            if verbose: print("No plotting anything.....\n" )
            
        #if plot_rss == False and plot == True and verbose: print(" plot_rss is false.....")
        
        print("1. Checking input values:")
        
        print("\n  - Using the following RSS files : ")
        rss_object=[]
        cube_object=[]
        cube_aligned_object=[]
        number=1
        
        for rss in range(n_files):
            rss_list[rss] =full_path(rss_list[rss],path)       
            print("    ",rss+1,". : ",rss_list[rss])
            _rss_ = "self.rss"+np.str(number)
            _cube_= "self.cube"+np.str(number)
            _cube_aligned_= "self.cube"+np.str(number)+"_aligned"
            rss_object.append(_rss_)
            cube_object.append(_cube_)
            cube_aligned_object.append(_cube_aligned_)
            number = number +1
            sky_rss_list.append([0])


        if len(save_rss_to_fits_file_list) > 0:
            try:
                if save_rss_to_fits_file_list == "auto":
                    save_rss_to_fits_file_list =[]
                    for rss in range(n_files):
                        save_rss_to_fits_file_list.append("auto")
            except Exception:
                if len(save_rss_to_fits_file_list) != len(n_files) and verbose : print("  WARNING! List of rss files to save provided does not have the same number of rss files!!!")
                       
        else:
            for rss in range(n_files):
                save_rss_to_fits_file_list.append("")
            
            
        self.rss_list=rss_list
        
        if number == 1: 
            do_alignment  =False 
            make_combined_cube= False
         
        if rss_clean:
            print("\n  - These RSS files are ready to be cubed & combined, no further process required ...")           
        else:   
            # Check throughput
            if apply_throughput:
                if len(throughput_2D) == 0 and throughput_2D_file == "" :
                    print("\n\n\n  WARNING !!!! \n\n  No 2D throughput data provided, no throughput correction will be applied.\n\n\n")
                    apply_throughput = False
                else:
                    if len(throughput_2D) > 0 :
                        print("\n  - Using the variable provided for applying the 2D throughput correction ...")
                    else:
                        print("\n  - The 2D throughput correction will be applied using the file:")
                        print("  ",throughput_2D_file)
            else:
                print("\n  - No 2D throughput correction will be applied")
            

            # sky_method = "self" "1D" "2D" "none" #1Dfit" "selffit"
              
            if sky_method == "1D" or sky_method == "1Dfit" or sky_method == "selffit":    
                if np.nanmedian(sky_spectrum) != -1 and np.nanmedian(sky_spectrum) != 0 :
                    for i in range(n_files):
                        sky_list[i] = sky_spectrum
                    print("\n  - Using same 1D sky spectrum provided for all object files") 
                else:
                    if np.nanmedian(sky_list[0]) == 0:
                        print("\n  - 1D sky spectrum requested but not found, assuming n_sky =",n_sky,"from the same files")
                        if sky_method in ["1Dfit","1D"]: sky_method = "self"
                    else:
                        print("\n  - List of 1D sky spectrum provided for each object file")

            if sky_method == "2D": 
                try:
                    if np.nanmedian(sky_list[0].intensity_corrected) != 0 :
                        print("\n  - List of 2D sky spectra provided for each object file")
                        for i in range(n_files):
                            sky_rss_list[i]=sky_list[i]
                            sky_list[i] = [0]
                except Exception: 
                    try:
                        if sky_rss == 0 :
                            print("\n  - 2D sky spectra requested but not found, assuming n_sky = 50 from the same files")
                            sky_method = "self"
                    except Exception:       
                        for i in range(n_files):
                            sky_rss_list[i]=sky_rss
                        print("\n  - Using same 2D sky spectra provided for all object files")  
                        
            if sky_method == "self": # or  sky_method == "selffit":
                for i in range(n_files):
                    sky_list[i] = []
                if n_sky == 0 : n_sky = 50
                if len(sky_fibres) == 0:
                    print("\n  - Using n_sky =",n_sky,"to create a sky spectrum")
                else:
                    print("\n  - Using n_sky =",n_sky,"and sky_fibres =",sky_fibres,"to create a sky spectrum")
                                                                                                    
            if grating in red_gratings:                                                                                                                       
                if np.nanmedian(telluric_correction) == 0 and np.nanmedian (telluric_correction_list[0]) == 0 :
                    print("\n  - No telluric correction considered")
                else: 
                    if np.nanmedian (telluric_correction_list[0]) == 0: 
                        for i in range(n_files):
                            telluric_correction_list[i] = telluric_correction
                        print("\n  - Using same telluric correction for all object files")
                    else: print("\n  - List of telluric corrections provided!")
 
        if do_rss: 
            print("\n-------------------------------------------")
            print("2. Reading the data stored in rss files ...")
                                    
            for i in range(n_files):
                     #skyflat=skyflat_list[i], plot_skyflat=plot_skyflat, throughput_file =throughput_file, nskyflat_file=nskyflat_file,\
                     # This considers the same throughput for ALL FILES !!
                exec(rss_object[i]+'= KOALA_RSS(rss_list[i], rss_clean = rss_clean, save_rss_to_fits_file = save_rss_to_fits_file_list[i], \
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
          
        if len(offsets) > 0 and len (ADR_x_fit_list) >  0 and ADR == True :
            #print("\n  Offsets values for alignment AND fitting for ADR correction have been provided, skipping cubing no-aligned rss...")
            do_cubing = False
        elif len(offsets) > 0 and ADR == False :
            #print("\n  Offsets values for alignment given AND the ADR correction is NOT requested, skipping cubing no-aligned rss...")
            do_cubing = False


        if len (ADR_x_fit_list) ==  0 :   # Check if lists with ADR values have been provided, if not create lists with 0
            ADR_x_fit_list = []
            ADR_y_fit_list = []
            for i in range (n_files):
                ADR_x_fit_list.append([0,0,0])
                ADR_y_fit_list.append([0,0,0])
                
        fcal= False
        if flux_calibration_file  != "":   # If files have been provided for the flux calibration, we read them
            fcal = True
            if type(flux_calibration_file) == str :
                if path != "": flux_calibration_file = full_path(flux_calibration_file,path)
                w_star,flux_calibration = read_table(flux_calibration_file, ["f", "f"] ) 
                for i in range(n_files):
                    flux_calibration_list.append(flux_calibration)
                
                if verbose: print("\n  - Using for all the cubes the same flux calibration provided in file:\n   ",flux_calibration_file)
            else:
                if verbose: print("\n  - Using list of files for flux calibration:")
                for i in range(n_files):
                    if path != "": flux_calibration_file[i] = full_path(flux_calibration_file[i],path)
                    print("   ",flux_calibration_file[i])
                    w_star,flux_calibration = read_table(flux_calibration_file[i], ["f", "f"] ) 
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
                exec(cube_object[i]+'=Interpolated_cube('+rss_object[i]+', pixel_size_arcsec=pixel_size_arcsec, kernel_size_arcsec=kernel_size_arcsec, plot=plot, half_size_for_centroid=half_size_for_centroid,\
                     ADR_x_fit = ADR_x_fit_list[i], ADR_y_fit = ADR_y_fit_list[i], box_x=box_x, box_y=box_y, plot_spectra=plot_spectra, \
                     adr_index_fit=adr_index_fit, g2d=g2d, plot_tracing_maps = plot_tracing_maps, step_tracing=step_tracing,  ADR=ADR, apply_ADR = False, \
                     flux_calibration=flux_calibration_list[i], edgelow=edgelow, edgehigh=edgehigh, size_arcsec=size_arcsec, centre_deg=centre_deg,warnings=warnings)')    
        else:
            if do_alignment:
                print("\n------------------------------------------------")
                if ADR == False:
                    print("3. Offsets provided, ADR correction NOT requested, cubing will be done using aligned cubes ...")   
                else:
                    print("3. Offsets AND correction for ADR provided, cubing will be done using aligned cubes ...")   

        rss_list_to_align=[]    
        cube_list = []
        for i in range(n_files):
            exec('rss_list_to_align.append('+rss_object[i]+')')
            if do_cubing:
                exec('cube_list.append('+cube_object[i]+')')
            else:
                cube_list.append([0])

        if do_alignment:   
            if len(offsets) == 0: 
                print("\n--------------------------------")
                print("4. Aligning individual cubes ...")
            else:
                print("\n-----------------------------------------------------")
                print("4. Checking offsets data provided and performing cubing ...")
                

            cube_aligned_list=align_n_cubes(rss_list_to_align, cube_list=cube_list, flux_calibration_list=flux_calibration_list, pixel_size_arcsec=pixel_size_arcsec, 
                                            kernel_size_arcsec=kernel_size_arcsec, plot=plot, plot_weight=plot_weight, 
                                            offsets=offsets, ADR=ADR, jump=jump, edgelow=edgelow, edgehigh=edgehigh, 
                                            size_arcsec=size_arcsec,centre_deg=centre_deg,
                                            half_size_for_centroid=half_size_for_centroid, box_x=box_x, box_y=box_y,
                                            ADR_x_fit_list =ADR_x_fit_list, ADR_y_fit_list = ADR_y_fit_list, 
                                            adr_index_fit=adr_index_fit, g2d=g2d, step_tracing = step_tracing, 
                                            plot_tracing_maps=plot_tracing_maps, plot_spectra=plot_spectra,
                                            force_ADR=force_ADR, warnings=warnings)      

            for i in range(n_files):             
                exec(cube_aligned_object[i]+'=cube_aligned_list[i]')
                
        else:
            
            if ADR == True and np.nanmedian(ADR_x_fit_list) ==  0:
            # If not alignment but ADR is requested
            
                print("\n--------------------------------")
                print("4. Applying ADR ...")
            
                for i in range(n_files): 
                    exec(cube_object[i]+'=Interpolated_cube('+rss_object[i]+', pixel_size_arcsec, kernel_size_arcsec, plot=plot, half_size_for_centroid=half_size_for_centroid,\
                         ADR_x_fit = cube_list[i].ADR_x_fit, ADR_y_fit = cube_list[i].ADR_y_fit, box_x=box_x, box_y=box_y, check_ADR = True, \
                         flux_calibration=flux_calibration_list[i], edgelow=edgelow, edgehigh=edgehigh, size_arcsec=size_arcsec, centre_deg=centre_deg,warnings=warnings)')    
 

       # Save aligned cubes to fits files
        if save_aligned_cubes:             
            print("\n> Saving aligned cubes to fits files ...")
            if cube_list_names[0] == "" :  
                for i in range(n_files):
                    if i < 9: 
                        replace_text = "_"+obj_name+"_aligned_cube_0"+np.str(i+1)+pk+".fits" 
                    else: replace_text = "_aligned_cube_"+np.str(i+1)+pk+".fits"                   
                    
                    aligned_cube_name=  rss_list[i].replace(".fits", replace_text)
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
                        print("ADR_x_fit  = ["+np.str(cube_list[i].ADR_x_fit[0])+","+np.str(cube_list[i].ADR_x_fit[1])+"]")
                        print("ADR_y_fit  = ["+np.str(cube_list[i].ADR_y_fit[0])+","+np.str(cube_list[i].ADR_y_fit[1])+"]")           
                    elif adr_index_fit == 2:
                        print("ADR_x_fit  = ["+np.str(cube_list[i].ADR_x_fit[0])+","+np.str(cube_list[i].ADR_x_fit[1])+","+np.str(cube_list[i].ADR_x_fit[2])+"]")
                        print("ADR_y_fit  = ["+np.str(cube_list[i].ADR_y_fit[0])+","+np.str(cube_list[i].ADR_y_fit[1])+","+np.str(cube_list[i].ADR_y_fit[2])+"]")
                    elif adr_index_fit == 3:
                        print("ADR_x_fit  = ["+np.str(cube_list[i].ADR_x_fit[0])+","+np.str(cube_list[i].ADR_x_fit[1])+","+np.str(cube_list[i].ADR_x_fit[2])+","+np.str(cube_list[i].ADR_x_fit[3])+"]")
                        print("ADR_y_fit  = ["+np.str(cube_list[i].ADR_y_fit[0])+","+np.str(cube_list[i].ADR_y_fit[1])+","+np.str(cube_list[i].ADR_y_fit[2])+","+np.str(cube_list[i].ADR_y_fit[3])+"]")

                except Exception:
                    print("  WARNING: Something wrong happened printing the ADR fit values! Results are:")
                    print("  ADR_x_fit  = ",cube_list[i].ADR_x_fit)
                    print("  ADR_y_fit  = ",cube_list[i].ADR_y_fit)

                _x_ = []
                _y_ = []
                for j in range(len(cube_list[i].ADR_x_fit)):
                    _x_.append(cube_list[i].ADR_x_fit[j])
                    _y_.append(cube_list[i].ADR_y_fit[j])   
                ADR_x_fit_list.append(_x_)
                ADR_y_fit_list.append(_y_)
   
        if obj_name == "":
            exec('obj_name = '+rss_object[0]+'.object')
            obj_name=obj_name.replace(" ", "_")
 
        if make_combined_cube and n_files > 1 :   
            print("\n---------------------------")
            print("5. Making combined cube ...")
                       
            self.combined_cube=build_combined_cube(cube_aligned_list,   obj_name=obj_name, description=description,
                                                  fits_file = fits_file, path=path,
                                                  scale_cubes_using_integflux= scale_cubes_using_integflux, 
                                                  flux_ratios = flux_ratios, apply_scale = apply_scale,
                                                  edgelow=edgelow, edgehigh=edgehigh,
                                                  ADR=ADR, ADR_cc = ADR_cc, jump = jump, pk = pk, 
                                                  ADR_x_fit_list=ADR_x_fit_list, ADR_y_fit_list=ADR_y_fit_list,
                                                  force_ADR=force_ADR,
                                                  half_size_for_centroid = half_size_for_centroid,
                                                  box_x=box_x, box_y=box_y,  
                                                  adr_index_fit=adr_index_fit, g2d=g2d,
                                                  step_tracing = step_tracing, 
                                                  plot_tracing_maps = plot_tracing_maps,
                                                  trim_cube = trim_cube,  trim_values =trim_values, 
                                                  remove_spaxels_not_fully_covered = remove_spaxels_not_fully_covered,
                                                  plot=plot, plot_weight= plot_weight, plot_spectra=plot_spectra,
                                                  verbose=verbose, say_making_combined_cube= False)
        else:
            if n_files > 1:
                if do_alignment == False and  do_cubing == False:
                    print("\n> As requested, skipping cubing...")
                else:
                    print("\n  No combined cube obtained!")
                    
            else:
                print("\n> Only one file provided, no combined cube obtained")
                # Trimming cube if requested or needed
                cube_aligned_list[0].trim_cube(trim_cube=trim_cube, trim_values=trim_values, ADR=ADR,
                               half_size_for_centroid =half_size_for_centroid, 
                               adr_index_fit=adr_index_fit, g2d=g2d, step_tracing=step_tracing,
                               plot_tracing_maps=plot_tracing_maps,
                               remove_spaxels_not_fully_covered = remove_spaxels_not_fully_covered,
                               box_x=box_x, box_y=box_y, edgelow=edgelow, edgehigh=edgehigh, 
                               plot_weight = plot_weight, fcal=fcal, plot=plot)    
                
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
def create_map(cube, line, w2 = 0., gaussian_fit = False, gf=False,
               lowlow= 50, lowhigh=10, highlow=10, highhigh = 50, no_nans = False,
               show_spaxels=[], verbose = True, description = "" ):
    
  
    if gaussian_fit or gf:
        
        if description == "" : 
            description = "{} - Gaussian fit to {}".format(cube.description, line)
            description = description+" $\mathrm{\AA}$"
    
        map_flux = np.zeros((cube.n_rows,cube.n_cols))
        map_vel  = np.zeros((cube.n_rows,cube.n_cols))
        map_fwhm = np.zeros((cube.n_rows,cube.n_cols))
        map_ew   = np.zeros((cube.n_rows,cube.n_cols))
        
        n_fits = cube.n_rows*cube.n_cols
        w = cube.wavelength
        if verbose: print("\n> Fitting emission line",line,"A in cube ( total = ",n_fits,"fits) ...")
        
        # For showing fits
        show_x=[]
        show_y=[]
        name_list=[]
        for i in range(len(show_spaxels)):
            show_x.append(show_spaxels[i][0]) 
            show_y.append(show_spaxels[i][1])
            name_list_ = "["+np.str(show_x[i])+","+np.str(show_y[i])+"]"
            name_list.append(name_list_)
        
        empty_spaxels=[]
        fit_failed_list=[]
        for x in range(cube.n_rows):
            for y in range(cube.n_cols):
                plot_fit = False
                verbose_fit = False
                warnings_fit = False

                for i in range(len(show_spaxels)):
                    if x == show_x[i] and y == show_y[i]:
                        if verbose: print("\n  - Showing fit and results for spaxel",name_list[i],":")
                        plot_fit = True
                        if verbose: verbose_fit = True
                        if verbose: warnings_fit = True
                                     
                spectrum=cube.data[:,x,y]
                
                if np.isnan(np.nanmedian(spectrum)):
                    if verbose_fit: print("  SPAXEL ",x,y," is empty! Skipping Gaussian fit...")
                    resultado = [np.nan]*10  
                    empty_spaxel = [x,y]
                    empty_spaxels.append(empty_spaxel)
                                          
                else:    
                    resultado = fluxes(w, spectrum, line, lowlow= lowlow, lowhigh=lowhigh, highlow=highlow, highhigh = highhigh, 
                                   plot=plot_fit, verbose=verbose_fit, warnings=warnings_fit)
                map_flux[x][y] = resultado[3]                      # Gaussian Flux, use 7 for integrated flux
                map_vel[x][y] = resultado[1]
                map_fwhm[x][y] = resultado[5] * C / resultado[1]   # In km/s
                map_ew[x][y] = resultado[9]                        # In \AA
                # checking than resultado[3] is NOT 0 (that means the Gaussian fit has failed!)
                if resultado[3] == 0:
                    map_flux[x][y] = np.nan                     
                    map_vel[x][y] = np.nan 
                    map_fwhm[x][y] = np.nan 
                    map_ew[x][y] = np.nan 
                    #if verbose_fit: print "  Gaussian fit has FAILED in SPAXEL ",x,y,"..."
                    fit_failed = [x,y]
                    fit_failed_list.append(fit_failed)
                    
        
        median_velocity= np.nanmedian(map_vel)
        map_vel = C*(map_vel - median_velocity) / median_velocity
        
        if verbose: 
            #print "\n> Summary of Gaussian fitting : "
            print("\n> Found ",len(empty_spaxels)," the list with empty spaxels is ",empty_spaxels)
            print("  Gaussian fit FAILED in",len(fit_failed_list)," spaxels = ",fit_failed_list)
            print("\n> Returning [map_flux, map_vel, map_fwhm, map_ew, description] ")
        return description, map_flux, map_vel, map_fwhm, map_ew 

    else:
        w1 = line
        if w2 == 0.:
            if verbose: print("\n> Creating map using channel closest to ",w1)
            interpolated_map = cube.data[np.searchsorted(cube.wavelength, w1)]
            descr = "{} - only {} ".format(cube.description, w1)
        
        else:
            if verbose: print("\n> Creating map integrating [{}-{}]".format(w1, w2))
            interpolated_map = np.nansum(cube.data[np.searchsorted(cube.wavelength, w1):np.searchsorted(cube.wavelength, w2)],axis=0)        
            descr ="{} - Integrating [{}-{}] ".format(cube.description, w1, w2)

        if description == "" : description = descr+"$\mathrm{\AA}$"
        if verbose: print("  Description =",description)
        # All 0 values should be nan
        if no_nans == False: interpolated_map[interpolated_map == 0] = np.nan
        return description,interpolated_map, w1, w2
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def create_mask(mapa, low_limit, high_limit=1E20, plot = False, verbose = True):
    
    n_rows = mapa.shape[0]
    n_cols = mapa.shape[1]
    
    mask = np.ones((n_rows,n_cols))
    
    for x in range(n_rows):
        for y in range(n_cols):
            value = mapa[x,y]                      
            if value < low_limit or value > high_limit: 
                mask[x][y] = np.nan
    
    if verbose: print("\n> Mask with good values between", low_limit,"and",high_limit,"created!")
        
    return mask,low_limit, high_limit
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

print("\n> PyKOALA",version,"read !!")