#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from synphot import observation   ########from pysynphot import observation
#from synphot import spectrum      ########from pysynphot import spectrum
from synphot import SourceSpectrum, SpectralElement
from synphot.models import Empirical1D
from scipy.signal import medfilt

developers = 'Developed by Angel Lopez-Sanchez, Pablo Corcho-Caballero,\
 Yago Ascasibar, Lluis Galbany, Barr Perez, Nathan Pidcock,\
 Diana Dalae, Giacomo Biviano, Adithya Gudalur Balasubramania,\
 Blake Staples, Taylah Beard, Matt Owers, James Tocknell et al.'


#import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import numpy as np
#import sys
#import os

from scipy import interpolate, signal
from scipy.optimize import curve_fit
import scipy.signal as sig

from random import uniform

# testing

#import datetime
import copy

#import glob
#from astropy.io import fits as pyfits 


from pykoala.plotting.plot_plot import plot_plot, basic_statistics

# Disable some annoying warnings
#import warnings

#warnings.simplefilter('ignore', np.RankWarning)
#warnings.simplefilter(action='ignore',category=FutureWarning)
#warnings.filterwarnings('ignore')
#import logging
#logging.basicConfig(level=logging.CRITICAL)
from pykoala._version import get_versions
version = get_versions()["version"]
del get_versions


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 0. Some constants
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------



pc=3.086E18    # pc in cm
C =299792.458  # c in km/s
nebula_lines = [6300.3, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15,
                6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 7751.1, 
                9068.9] 



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

# EXTRA
#    8750.47   -0.568       H-I    P12   8776
#    8862.79   -0.578       H-I    P11   8888
#    9014.91   -0.590       H-I    P10   9041


# NOTE: 18 lines for BLUE (580V) and 18 lines for RED (385R), if this changes careful with emission line detection!!!
                         
el_list_no_z = [3726.03, 3728.82,  # 3727.0, #
                3868.75, 3889.05,  3964.73, 3967.46, 3970.07,  #3966.1 #
                4068.60 , #4076.35 ,
                4101.74, 4340.47, 4363.21, 
                #4437.55, 
                4471.48, 4658.10,
                4686.00, 4861.33, 4958.91, 5006.84,
                5191.82, #5517.71,   
                #red  #  [OI]   [SIII]  [OI]   Ha+[NII]  HeI    [SII]     HeI   [ArIII]  [OII]  [ArIII]    P12 P11 P10    [SIII]         
                6300.3, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 
                6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 7751.1, 
                8750.47, 8862.79, 9014.91,
                9068.9] 

el_list_names = ["[OII]", "[OII]", 
                 "[NeIII]",  "H8",  "HeI", "[NeIII]",  "H7", # "HeI+[NeIII]"
                 "[SII]", #"[SII]",
                 "Hd","Hg", "[OIII]", 
                 #"HeI",
                 "HeI", "[FeIII]",
                 "HeII", "Hb", "[OIII]", "[OIII]",
                 "[ArIII]", # "[ClIII]",
                 #red
                "[OI]", "[SIII]", "[OI]","[NII]","Ha","[NII]","HeI","[SII]","[SII]","HeI","[ArIII]","[OII]","[OII]","[ArIII]",
                 "P12", "P11", "P10",
                 "[SIII]"]
el_list_fl = [0.322, 0.322,   
              0.291, 0.286,   0.267, 0.267, 0.266,
              0.239, #0.237,
              0.230, 0.157, 0.149,
              #0.126 ,
              0.115,  0.058,             
              0.050,  0.000, -0.032, -0.038,
              -0.081, #-0.145 ,
              #red  
              -0.263, -0.264, -0.271, -0.296,  -0.298,  -0.300, -0.313,  -0.318, -0.320, -0.364, -0.374, -0.398, -0.400, -0.455,
              -0.568,-0.578 , -0.590,
              -0.594]



# EL  dictionary 
emission_line_dictionary = {}

for i in range(len(el_list_no_z)):
    emission_line_dictionary[el_list_no_z[i]] = [el_list_names[i], el_list_fl[i]]

#  [OI]   [SIII]  [OI]   Ha+[NII]  HeI    [SII]     HeI   [ArIII]  [OII]  [ArIII]  [SIII]  
# el_low_list_no_z  =[6296.3, 6308.1, 6359.8, 6544.0, 6674.2, 6712.5, 7061.3, 
#                     7129., 7312., 7747.1,   0,0,0,   9063.9]                     ### Careful with 0,0,0 as extra added to el_list
# el_high_list_no_z =[6304.3, 6316.1, 6367.8, 6590.0, 6682.2, 6736.9, 7069.3, 
#                     7141., 7336., 7755.1,   0,0,0,   9073.9]

# =============================================================================
# Sky lines
# =============================================================================


# These are for KOALA, I should move them there....

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



#Double Skylines for fitting sky OLD VERSION - perhaps not needed anymore
dsky1_=[6257.82, 6465.34, 6828.22, 6969.70, 7239.41, 7295.81, 7711.50, 7750.56,
        7853.391, 7913.57, 7773.00, 7870.05, 8280.94, 8344.613, 9152.2, 9092.7,
        9216.5,  8827.112, 8761.2, 0] # 8760.6, 0]#
dsky2_=[6265.50, 6470.91, 6832.70, 6978.45, 7244.43, 7303.92, 7715.50, 7759.89,
        7860.662, 7921.02, 7780.43, 7879.96, 8288.34, 8352.78,  9160.9, 9102.8,
        9224.8,  8836.27 , 8767.7, 0] # 8767.2, 0] #     



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 1. Generic In / Out tasks
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

from astropy.io import fits
import datetime

def read_table(file, fmt):
    """
    Read data from and txt file (sorted by columns), the type of data 
    (string, integer or float) MUST be given in "formato".
    This routine will ONLY read the columns for which "formato" is defined.
    E.g. for a txt file with 7 data columns, using formato=["f", "f", "s"] will only read the 3 first columns.
    
    Parameters
    ----------
    file: 
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
    data_len = len(fmt)
    data = [[] for x in range(data_len)]
    for i in range (0,data_len) :
        try:
            if fmt[i] == "i" : data[i]=np.loadtxt(file, skiprows=0, unpack=True, usecols=[i], dtype=int)
            if fmt[i] == "s" : data[i]=np.loadtxt(file, skiprows=0, unpack=True, usecols=[i], dtype=str)
            if fmt[i] == "f" : data[i]=np.loadtxt(file, skiprows=0, unpack=True, usecols=[i], dtype=float) 
        except Exception:
            pass       
    return data
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def write_table_from_data(data,filename,overwrite=True, verbose_list = False, verbose = True):
    """
    Write a txt table in a file from data.

    Parameters
    ----------
    data : list
        list with the arrays of our data.
    filename : string
        Name, including path, of the file to be saved.
    overwrite : Boolean, optional
        If True, overwrite file. If False, append data to file. The default is True.
    verbose_list : Boolean, optional
        If True, it lists all the data. The default is False.
    verbose : Boolean, optional
        If True, it says what it is doing. The default is True.

    Returns
    -------
    NONE
    
    """
    if overwrite:
        f = open(filename, 'w')
        if verbose: print("Listing data overwriting file...")
    else:
        f = open(filename, 'a')
        if verbose: print("Listing data appending file...")
        
    n_variables = len(data)
    n_items = len(data[0])
    for line in range(n_items):
        escribe = ""
        for column in range (n_variables):
            if column != 0: escribe+="  "
            escribe+=str(data[column][line])  
        if verbose_list: print(escribe)
        print(escribe, file=f)

    if verbose: print("Data saved in file",filename)            
    f.close()
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def array_to_text_file(data, filename="array.dat" ):
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
        escribe = str(data[i])+" \n"
        f.write(escribe)
    f.close()
    print("\n> Array saved in text file",filename," !!")
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
        escribe = str(wavelength[i])+"  "+str(flux[i])+" \n"
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
    hdu.header['HISTORY'] = developers
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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2. RSS In / Out tasks
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
# 1D TASKS
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
# -----------------------------------------------------------------------------
def gauss(x, x0, y0, sigma):
    p = [x0, y0, sigma]
    return p[1]* np.exp(-0.5*((x-p[0])/p[2])**2)
def gauss_fix_x0(x, x0, y0, sigma):
    p = [y0, sigma]
    return p[0]* np.exp(-0.5*((x-x0)/p[1])**2)    
def gauss_flux (y0,sigma):  ### THIS DOES NOT WORK...
    return y0 * sigma * np.sqrt(2*np.pi)
#-----------------------------------------------------------------------------
def ngauss(x, xn, yn, sigman):
    n_fits = len(xn)
    if len(yn) != n_fits or len(sigman) != n_fits:
        f = None
    else:
        f = 0
        p=[]
        for i in range(n_fits):
            p_ = [xn[i], yn[i], sigman[i]]
            p = p + p_
            f = f + p_[1]* np.exp(-0.5*((x-p_[0])/p_[2])**2)
    return f
#-----------------------------------------------------------------------------
def gauss10(x, x1, y1, sigma1,
               x2=None, y2=None, sigma2=None,
               x3=None, y3=None, sigma3=None,
               x4=None, y4=None, sigma4 = None, x5=None, y5=None, sigma5 = None, x6=None, y6=None, sigma6 = None,
               x7=None, y7=None, sigma7 = None, x8=None, y8=None, sigma8 = None, x9=None, y9=None, sigma9 = None,
               x10=None, y10=None, sigma10 = None):
    p = [x1, y1, sigma1, x2, y2, sigma2,  x3, y3, sigma3, x4, y4, sigma4,  x5,  y5,  sigma5, 
         x6, y6, sigma6, x7, y7, sigma7,  x8, y8, sigma8, x9, y9, sigma9 , x10, y10, sigma10 ]
    f = 0
    j = 0
    empty = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10].count(None)
    
    for i in range(10-empty):
        f = f + p[j+1]* np.exp(-0.5*((x-p[j])/p[j+2])**2)
        j=j+3
    return f
#-----------------------------------------------------------------------------
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
# Definition introduced by Matt - used in fit_smooth_spectrum
def MAD(x):
    MAD=np.nanmedian(np.abs(x-np.nanmedian(x)))
    return MAD/0.6745
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def calculate_z(l_obs=None, l_ref=None, ref_line = None, v_rad=None, verbose=True):

    if v_rad is not None:
        z = v_rad/C
        if verbose: print("  Using v_rad = {:.2f} km/s, redshift z = {:.06}".format(v_rad,z))
    else:
        if ref_line=="[OIII]": 
            l_ref = 5006.84
        elif ref_line=="Hb": 
            l_ref = 4861.33 
        else:
            l_ref = 6562.82
            ref_line="Ha"
    
        z = l_obs/l_ref - 1.
        v_rad = z *C
        
        if verbose:
            print("  Using line {}, l_rest = {:.2f}, peak at l_obs = {:.2f}. ".format(ref_line,l_ref,l_obs))
            print("  v_rad = {:.2f} km/s, redshift z = {:.06}".format(v_rad,z))
    return z
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def apply_z(lambdas, z=None, l_obs=None, l_ref=None, ref_line = None, v_rad=None, verbose=True):
    
    if z is None: z = calculate_z(l_obs=l_obs, l_ref=l_ref, ref_line = ref_line, v_rad=v_rad, verbose=verbose)
        
    zlambdas =(z+1) * np.array(lambdas)
    
    if verbose:
        print("  Computing observed wavelengths using v_rad = {:.2f} km/s, redshift z = {:.06} :".format(v_rad,z))
        print("  REST :",lambdas)
        print("  z    :",np.round(zlambdas,2))
        
    return zlambdas
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def substract_given_gaussian(wavelength, spectrum, centre, peak=0, sigma=0,  flux=0, search_peak=False, allow_absorptions = False,
                             lowlow= 20, lowhigh=10, highlow=10, highhigh = 20, 
                             xmin=0, xmax=0, fmin=0, fmax=0, plot=True, fcal=False, verbose = True, warnings=True):
    """
    Substract a given Gaussian to a spectrum after fitting the continuum.
    #FIXME perhaps not needed...
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
        if xmin == 0 : xmin = centre-65.    # By default, +-65 A with respect to line
        if xmax == 0 : xmax = centre+65.
            
        # Extract subrange to fit
        # THIS SHOULD BE A FUNCTION  #TODO
        w_spec = []
        f_spec = []
        w_spec.extend((wavelength[i]) for i in range(len(wavelength)) if (wavelength[i] > xmin and wavelength[i] < xmax) )    
        f_spec.extend((spectrum[i]) for i in range(len(wavelength)) if (wavelength[i] > xmin and wavelength[i] < xmax) )  
     
        # Setup min and max flux values in subrange to fit
        if fmin == 0 : fmin = np.nanmin(f_spec)            
        if fmax == 0 : fmax = np.nanmax(f_spec)                                 
    
        # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to centre
        # THIS SHOULD BE A FUNCTION  #TODO
        w_cont=[]
        f_cont=[]
        w_cont.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > centre-lowlow and w_spec[i] < centre-lowhigh) or (w_spec[i] > centre+highlow and w_spec[i] < centre+highhigh)   )    
        f_cont.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > centre-lowlow and w_spec[i] < centre-lowhigh) or (w_spec[i] > centre+highlow and w_spec[i] < centre+highhigh)   )    
    
        # Linear Fit to continuum 
        # THIS SHOULD BE A FUNCTION  #TODO
        try:    
            mm,bb = np.polyfit(w_cont, f_cont, 1)
        except Exception:
            bb = np.nanmedian(spectrum)
            mm = 0.
            if warnings: 
                print("      WARNING! Impossible to get the continuum!")
                print("               Scaling the continuum to the median value") 
        continuum =   mm*np.array(w_spec)+bb  
        # c_cont = mm*np.array(w_cont)+bb  
        # rms continuum
        # rms_cont = np.nansum([ np.abs(f_cont[i] - c_cont[i])  for i in range(len(w_cont)) ]) / len(c_cont)

        if search_peak:
            # Search for index here w_spec(index) closest to line
            # THIS SHOULD BE A FUNCTION  #TODO
            try:
                min_w = np.abs(np.array(w_spec)-centre)
                mini = np.nanmin(min_w)
                peak = f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]
                flux = peak * sigma * np.sqrt(2*np.pi)   
                if verbose: print("    Using peak as f[",np.round(centre,2),"] = ",np.round(peak,2)," and sigma = ", np.round(sigma,2), "    flux = ",np.round(flux,2))
            except Exception:
                if warnings: print("    Error trying to get the peak as requested wavelength is ",np.round(centre,2),"! Ignoring this fit!")
                peak = 0.
                flux = -0.0001
    
        no_substract = False
        if flux < 0:
            if allow_absorptions == False:
                if np.isnan(centre) == False:
                    if warnings : print("    WARNING! This is an ABSORPTION Gaussian! As requested, this Gaussian is NOT substracted!")
                no_substract = True
                
        if no_substract == False:     
            # THIS SHOULD BE A FUNCTION  #TODO
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
                # THIS SHOULD BE PLOT_PLOT #TODO
                plt.figure(figsize=(10, 4))
                plt.plot(np.array(w_spec),np.array(f_spec), "b", lw=3, alpha = 0.5)
                plt.minorticks_on() 
                plt.xlabel("Wavelength [$\mathrm{\AA}$]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
                else:
                    plt.ylabel("Flux [ counts ]")
                plt.xlim(xmin,xmax)
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
                plt.xlim(xmin,xmax)
                plt.ylim(fmin,fmax)
                plt.show()
                plt.close()
        else:
            s_s = spectrum
    return s_s
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#TODO This is a very important task, but it needs to be checked
def fluxes(wavelength, s, line, lowlow= 14, lowhigh=6, highlow=6, highhigh = 14, 
           xmin=0, xmax=0, fmin=0, fmax=0, broad=2.355, 
           plot=True,  plot_sus = False, plot_residuals = True, 
           fcal = True, fit_continuum = True, median_kernel=35, 
           verbose=True, warnings = True ):   # Broad is FWHM for Gaussian sigma= 1, 
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
    xmin, xmax: float
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
    if xmin == 0 : xmin = line-65.    # By default, +-65 A with respect to line
    if xmax == 0 : xmax = line+65.
        
    # Extract subrange to fit
    # THIS SHOULD BE A FUNCTION  #TODO
    w_spec = []
    f_spec = []
    w_spec.extend((wavelength[i]) for i in range(len(wavelength)) if (wavelength[i] > xmin and wavelength[i] < xmax) )    
    f_spec.extend((s[i]) for i in range(len(wavelength)) if (wavelength[i] > xmin and wavelength[i] < xmax) )  
 
    if np.isnan(np.nanmedian(f_spec)): 
        # The data are NAN!! Nothing to do
        if verbose or warnings: print("    There is no valid data in the wavelength range [{},{}] !!".format(xmin,xmax))
        
        resultado = [0, line, 0, 0, 0, 0, 0, 0, 0, 0, 0, s  ]  

        return resultado
        
    else:    
    
        ## 20 Sep 2020
        f_spec_m=signal.medfilt(f_spec,median_kernel)    # median_kernel = 35 default
        
        # Remove nans
        median_value = np.nanmedian(f_spec)
        f_spec = [median_value if np.isnan(x) else x for x in f_spec]  
            
        # Setup min and max flux values in subrange to fit
        if fmin == 0 : fmin = np.nanmin(f_spec)            
        if fmax == 0 : fmax = np.nanmax(f_spec) 
         
        # We have to find some "guess numbers" for the Gaussian. Now guess_centre is line
        guess_centre = line
                   
        # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to guess_centre
        # THIS SHOULD BE A FUNCTION  #TODO
        w_cont=[]
        f_cont=[]
        w_cont.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-lowlow and w_spec[i] < guess_centre-lowhigh) or (w_spec[i] > guess_centre+highlow and w_spec[i] < guess_centre+highhigh)   )    
        f_cont.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-lowlow and w_spec[i] < guess_centre-lowhigh) or (w_spec[i] > guess_centre+highlow and w_spec[i] < guess_centre+highhigh)   )    
    
        if fit_continuum:
            # Linear Fit to continuum   
            # THIS SHOULD BE A FUNCTION  #TODO
            f_cont_filtered=sig.medfilt(f_cont,int(median_kernel))
            try:    
                mm,bb = np.polyfit(w_cont, f_cont_filtered, 1)
            except Exception:
                bb = np.nanmedian(f_cont_filtered)
                mm = 0.
                if warnings: 
                    print("    WARNING: Impossible to get the continuum!")
                    print("             Scaling the continuum to the median value b = ",bb,":  cont =  0 * w_spec  + ", bb)
            continuum =   mm*np.array(w_spec)+bb  
            c_cont = mm*np.array(w_cont)+bb  
    
        else:    
            # Median value in each continuum range  # NEW 15 Sep 2019
            # THIS SHOULD BE A FUNCTION  #TODO
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
            
            # THIS SHOULD BE A FUNCTION  #TODO
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
        # THIS SHOULD BE A FUNCTION  #TODO
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
        # THIS SHOULD BE A FUNCTION  #TODO
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
            if fit[0] <  guess_centre - broad  or fit[0] >  guess_centre + broad:
                if verbose: print("    Fitted center wavelength", fit[0],"is NOT in the expected range [",guess_centre - broad,",",guess_centre + broad,"]")
                 
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
                # THIS SHOULD BE with plot_plot #TODO
                plt.figure(figsize=(9.5,3.5)) #10, 4))
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
                    
                plt.xlim(xmin,xmax)    
                interval = (fmax - fmin)
                fmin = fmin - interval * 0.15 #extra_y
                fmax = fmax + interval * 0.15 #extra_y    
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
                if plot_residuals:
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
            if plot_sus: plot_plot(wavelength,[s,s_s], xmin=xmin, xmax=xmax, ymin=fmin, ymax=fmax, fcal=fcal, frameon=True, ptitle=ptitle)
          
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
                # This should be with plot_plot
                
                plt.figure(figsize=(10, 4))
                plt.plot(np.array(w_spec),np.array(f_spec), "b", lw=3, alpha = 0.5)
                plt.minorticks_on() 
                plt.xlabel("Wavelength [$\mathrm{\AA}$]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
                else:
                    plt.ylabel("Flux [ counts ]")            
                plt.xlim(xmin,xmax)
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
            xmin=0, xmax=0, fmin=0, fmax=0,
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
    xmin, xmax: float
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
    if xmin == 0 : xmin = line1-65.    # By default, +-65 A with respect to line
    if xmax == 0 : xmax = line2+65.
        
    # Extract subrange to fit
    # THIS SHOULD BE A FUNCTION  #TODO
    w_spec = []
    f_spec = []
    w_spec.extend((wavelength[i]) for i in range(len(wavelength)) if (wavelength[i] > xmin and wavelength[i] < xmax) )    
    f_spec.extend((s[i]) for i in range(len(wavelength)) if (wavelength[i] > xmin and wavelength[i] < xmax) )  
 
    
    if np.nanmedian(f_spec) == np.nan: print("  NO DATA HERE... ALL are nans!")

    
    # Setup min and max flux values in subrange to fit
    if fmin == 0 : fmin = np.nanmin(f_spec)            
    if fmax == 0 : fmax = np.nanmax(f_spec) 
     

    # We have to find some "guess numbers" for the Gaussian
    # Now guess_centre is line
    guess_centre1 = line1
    guess_centre2 = line2  
    guess_centre = (guess_centre1+guess_centre2)/2.         
    
    # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to guess_centre
    # THIS SHOULD BE A FUNCTION  #TODO
    w_cont=[]
    f_cont=[]
    w_cont.extend((w_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-lowlow and w_spec[i] < guess_centre-lowhigh) or (w_spec[i] > guess_centre+highlow and w_spec[i] < guess_centre+highhigh)   )    
    f_cont.extend((f_spec[i]) for i in range(len(w_spec)) if (w_spec[i] > guess_centre-lowlow and w_spec[i] < guess_centre-lowhigh) or (w_spec[i] > guess_centre+highlow and w_spec[i] < guess_centre+highhigh)   )    

    if fit_continuum:
        # Linear Fit to continuum  
        # THIS SHOULD BE A FUNCTION  #TODO
        f_cont_filtered=sig.medfilt(f_cont,int(median_kernel))
        try:    
            mm,bb = np.polyfit(w_cont, f_cont_filtered, 1)
        except Exception:
            bb = np.nanmedian(f_cont_filtered)
            mm = 0.
            if warnings: 
                print("  WARNING: Impossible to get the continuum!")
                print("           Scaling the continuum to the median value")          
        continuum =   mm*np.array(w_spec)+bb  
        c_cont = mm*np.array(w_cont)+bb  

    else:    
        # Median value in each continuum range  # NEW 15 Sep 2019
        # THIS SHOULD BE A FUNCTION  #TODO
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
    # THIS SHOULD BE A FUNCTION  #TODO
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
    # THIS SHOULD BE A FUNCTION  #TODO
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
            # This should be with plot_plot #TODO
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
            # This should be plot_plot #TODO
            plt.figure(figsize=(10, 4))
            plt.plot(wavelength,s, "r")
            plt.plot(wavelength,s_s, "c")
            plt.minorticks_on() 
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(xmin,xmax)
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

       # Plotting # This should be with plot_plot #TODO
        if plot :
            plt.figure(figsize=(10, 4))
            plt.plot(np.array(w_spec),np.array(f_spec), "b", lw=3, alpha = 0.5)
            plt.minorticks_on() 
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]")
            else:
                plt.ylabel("Flux [ counts ]")            
            plt.xlim(xmin,xmax)
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
def fit10gaussians(w, s, lines_to_fit, 
                   continuum = None,
                   low_low=None, low_high=None, high_low=None, high_high=None,  # for continuum
                   yn= None, sigman=None,
                   fit_degree_continuum=None, kernel_smooth_continuum=None,
                   max_wave_disp = None, min_peak_flux = None, max_sigma = None, max_peak_factor = None, min_peak_factor =None,
                   return_fit = False, return_fitted_lines = False, return_dictionary_for_fitted_lines = False,
                   xmin=None, xmax=None, ymin = None, ymax = None, extra_y = None,
                   plot_continuum = False, **kwargs): #plot = True):
    """
    Fit up to 10 Gaussians in spectrum

    Parameters
    ----------
    w : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.
    lines_to_fit : TYPE
        DESCRIPTION.
    continuum : TYPE, optional
        DESCRIPTION. The default is None.
    low_low : TYPE, optional
        DESCRIPTION. The default is None.
    low_high : TYPE, optional
        DESCRIPTION. The default is None.
    high_low : TYPE, optional
        DESCRIPTION. The default is None.
    high_high : TYPE, optional
        DESCRIPTION. The default is None.
    yn : TYPE, optional
        DESCRIPTION. The default is None.
    sigman : TYPE, optional
        DESCRIPTION. The default is None.
    fit_degree_continuum : TYPE, optional
        DESCRIPTION. The default is 2.
    max_wave_disp : TYPE, optional
        DESCRIPTION. The default is 0.9.
    min_peak_flux : TYPE, optional
        DESCRIPTION. The default is 0.
    max_sigma : TYPE, optional
        DESCRIPTION. The default is 3.2.
    max_peak_factor : TYPE, optional
        DESCRIPTION. The default is 5.
    min_peak_factor : TYPE, optional
        DESCRIPTION. The default is 5.
    return_fit : TYPE, optional
        DESCRIPTION. The default is False.
    return_fitted_lines : TYPE, optional
        DESCRIPTION. The default is False.
    return_dictionary_for_fitted_lines : TYPE, optional
        DESCRIPTION. The default is False.
    xmin : TYPE, optional
        DESCRIPTION. The default is None.
    xmax : TYPE, optional
        DESCRIPTION. The default is None.
    ymin : TYPE, optional
        DESCRIPTION. The default is None.
    ymax : TYPE, optional
        DESCRIPTION. The default is None.
    extra_y : TYPE, optional
        DESCRIPTION. The default is None.
    plot_continuum : TYPE, optional
        DESCRIPTION. The default is False.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    s_out, gaussian10_fit, dictionary_for_fitted_lines

    """
    verbose = kwargs.get('verbose', False)
    plot =  kwargs.get('plot', False)
    warnings = kwargs.get('warnings', verbose)
    if plot is False: plot_continuum = False
    
    n_lines = len(lines_to_fit)
 
    # lines_to_fit: if line is -, it is an ABSORPTION line
    absorption_lines=[x<0 for x in lines_to_fit]
    lines_to_fit = np.abs(lines_to_fit)
    
    #Provide default values, these are for EMISSION lines
    if max_wave_disp is None:  
        max_wave_disp = [0.9]*n_lines
    elif np.isscalar(max_wave_disp) : max_wave_disp =[max_wave_disp] *n_lines
            
    if min_peak_flux is None: min_peak_flux = 0    # Only emission lines by default
    if max_sigma is None: max_sigma = 3.2
    if max_peak_factor is None: max_peak_factor = 5 
    if min_peak_factor is None: min_peak_factor = 5


    
    # Prepare data
    s_ =[np.nanmedian(s) if np.isnan(x) else x for x in s] # be sure we don't have nans
    if continuum is None:
        # Get continuum in range and substract it
        if low_low is not None and low_high is not None and high_low is not None and high_high is not None:
            if fit_degree_continuum is None: fit_degree_continuum = 2
            continuum=get_continuum_in_range(w,s_, low_low, low_high, high_low, high_high, 
                                             fit_degree=fit_degree_continuum, 
                                             return_fit=True, plot=plot_continuum)
        else:
            if fit_degree_continuum is None: fit_degree_continuum = 7
            _,continuum=fit_smooth_spectrum(w, s, plot=plot_continuum, index_fit=fit_degree_continuum, 
                                            kernel_fit=kernel_smooth_continuum, auto_trim=True, verbose=False) 
    else:
        continuum=[np.nanmedian(continuum) if np.isnan(x) else x for x in continuum]  # be sure we don't have nans   
    sc = np.array(s_) - np.array(continuum) # Sustract continuum, being sure they are arrays
    # Prepare value

    #if yn is None: 
    #    yn = [sc[np.abs(w - line).argmin()] for line in lines_to_fit]
                
    if sigman is not None:
        if np.isscalar(sigman) : sigman =[sigman]*n_lines
    else:
        sigman =[2]*n_lines
    # find max yn value in data
    max_yn = []
    min_yn = []
    for i in range(n_lines):
        valid_ind=np.where((w >= lines_to_fit[i]-max_wave_disp[i]) & (w <= lines_to_fit[i]+max_wave_disp[i]) )[0]
        max_yn.append(np.nanmax(sc[valid_ind]))
        min_yn.append(np.nanmin(sc[valid_ind]))        
    # Prepare p0
    p0=[]
    for i in range(n_lines):
        if yn is not None:
            p0=p0+[lines_to_fit[i], yn[i], sigman[i]]
        else:
            if absorption_lines[i]:
                p0=p0+[lines_to_fit[i], min_yn[i], sigman[i]]
            else:
                p0=p0+[lines_to_fit[i], max_yn[i], sigman[i]]
       
    # Perform the fit to continuum substracted
    fit, pcov = curve_fit(gauss10, w, sc, p0=p0, maxfev=10000)   # If this fails, increase maxfev...
    # Check that fitted central wavelength is +-max_wave_disp A, width < max_sigma and min_peak_flux > 0
    j=0
    cvlines = ["k"] * n_lines
    plot_vlines = list(lines_to_fit)
    bad_fit_count=0
    fitted_lines =[]
    fit=fit.tolist()
    dictionary_for_fitted_lines = {} # Create dictionary if requested:
    for i in range(n_lines):
        reason=" "
        bad_fit = False
        if fit[j] < lines_to_fit[i]-max_wave_disp[i] or fit[j] > lines_to_fit[i]+max_wave_disp[i]: 
            bad_fit = True
            reason=reason+"out of wave range,"
        if fit[j+1] < min_peak_flux :                 # only emission lines if min_peak_flux = 0
            bad_fit = True 
            reason=reason+"min peak flux too low,"    
        if fit[j+1] > max_peak_factor  * max_yn[i] :  # fitting too high
            bad_fit = True   
            reason=reason+"max peak factor exceeded,"
        if absorption_lines[i] and fit[j+1] < min_peak_factor  * min_yn[i] : 
            bad_fit = True                            # fitting too low
            reason=reason+"min peak factor exceeded,"
        if fit[j+2] > max_sigma:                     # max sigma
            bad_fit = True
            reason=reason+"max sigma exceeded,"
        if bad_fit:
            if warnings: print("\n- Fitting "+str(fit[j])+" FAILED because"+reason)
            fit[j+1] = 0
            cvlines.append("r")
            plot_vlines.append(fit[j])
            bad_fit_count = bad_fit_count+1
        else:
            fitted_lines.append(lines_to_fit[i])
            cvlines.append("lime")
            plot_vlines.append(fit[j])
            if return_dictionary_for_fitted_lines: dictionary_for_fitted_lines[lines_to_fit[i]] = [fit[j], fit[j+1], fit[j+2]]
        j=j+3
    for i in range(10-n_lines):
        fit = fit+[None,None,None]
    
    # Create fitted spectrum and substract
    gaussian10_fit =  gauss10(w, fit[0], fit[1], fit[2],fit[3], fit[4], fit[5], fit[6], fit[7], fit[8], 
                          fit[9], fit[10], fit[11], fit[12], fit[13], fit[14], fit[15], fit[16], fit[17],
                          fit[18], fit[19], fit[20], fit[21], fit[22], fit[23],
                          fit[24], fit[25], fit[26], fit[27], fit[28], fit[29])
    s_out = s -gaussian10_fit
   
    # Plot fit
    if plot:
        ptitle = "Fitted "+str(n_lines-bad_fit_count)+" Gaussians (lime)"
        if bad_fit_count > 0 : ptitle = ptitle+" - "+str(bad_fit_count)+" Gaussian fits discarded (red)"
        if xmin is None: 
            if low_low is not None: 
                xmin=low_low
            else:
                xmin = np.nanmin(lines_to_fit) - 40 
        if xmax is None: 
            if high_high is not None:
                xmax=high_high
            else:
                xmax = np.nanmax(lines_to_fit) + 40 
        plot_plot(w, [s, gaussian10_fit+continuum, s_out], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, extra_y= extra_y,
                  hlines=[0], vlines =plot_vlines, cvlines=cvlines, ptitle=ptitle, **kwargs)   

    # Return values
    if return_dictionary_for_fitted_lines : return_fitted_lines = True    
    if return_fitted_lines:  return_fit = True
    if return_fit:
        if return_fitted_lines:
            if return_dictionary_for_fitted_lines:
                return s_out, gaussian10_fit, dictionary_for_fitted_lines
            else:
                return s_out, gaussian10_fit, fitted_lines
        else:
            return s_out, gaussian10_fit
    else:
        return s_out
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def find_emission_lines(w, s, continuum = None,
                        ref_line="Ha", l_ref = None,
                        sigman=2.5,  max_wave_disp=2, min_peak_flux=0, max_sigma=5, 
                        max_peak_factor=None, min_peak_factor=None,
                        index_fit_smooth_spectrum = 51, kernel_fit_smooth_spectrum =91,
                        fit_degree_continuum=None, kernel_smooth_continuum=None,
                        exclude_wlm = None,
                        **kwargs):
    
    verbose = kwargs.get('verbose', False)
    #plot =  kwargs.get('plot', False)
    #warnings = kwargs.get('warnings', verbose)
    
    if verbose: print("> Detecting emission lines in this spectrum...")
    # Find peak of emission in input spectrum        
    # The peak SHOULD be Halpha in red and Hbeta or [OIII] 5007 in blue... So far only checking red:
    index_max = s.tolist().index(np.nanmax(s))
    w_max = w[index_max]  # This should be the peak of H-alpha REDSHIFTED    
    
    if ref_line == "Ha": 
        l_ref = 6562.82
        if exclude_wlm is None:
            exclude_wlm = [[w_max-45,w_max+80]]
        else:
            exclude_wlm.append([w_max-45,w_max+80]) 
    if ref_line=="[OIII]": 
            l_ref = 5006.84
    if ref_line=="Hb": 
            l_ref = 4861.33 
    
    if verbose: print(f"  Peak of emission is at {w_max}, this should be {ref_line}")
    
    # Compute z
    z = calculate_z(l_obs=w_max, ref_line = ref_line, verbose=verbose)
    
    # Computing continuum discarding Halpha region if needed
    if continuum is None:
        _,continuum =fit_smooth_spectrum(w,s, index_fit=index_fit_smooth_spectrum, 
                                       kernel_fit=kernel_fit_smooth_spectrum, exclude_wlm=exclude_wlm, verbose = False, plot=False)
    
    # Check emission lines that lie in observed wavelength range
    _redshifted_emission_lines_ = [line*(z+1)   for line in el_list_no_z       if line*(z+1) > w[0] and line*(z+1) < w[-1]]
    _redshifted_emission_line_dictionary_ = {}
    for line in el_list_no_z:
        if line*(z+1) > w[0] and line*(z+1) < w[-1]:
            _redshifted_emission_line_dictionary_[line] =[line*(z+1), emission_line_dictionary[line][0], emission_line_dictionary[line][1]  ]    

    if kwargs.get("percentile_min") is None: kwargs["percentile_min"] = 0
    if kwargs.get("percentile_max") is None: kwargs["percentile_max"] = 97
       
    # FITTING IN BATCHES:

    # This is for range   [OI] 6300.30 - [SII] 6730.85 (red)      Using 385R or 1000R
    #                    [OII] 3726.03  -  Hd 4101.74  (blue)     Using 580V
        
    out=fit10gaussians(w, s, _redshifted_emission_lines_[0:9], continuum=continuum, 
                       sigman=sigman,  max_wave_disp=max_wave_disp, min_peak_flux=min_peak_flux, 
                       max_sigma=max_sigma, max_peak_factor=max_peak_factor, min_peak_factor=min_peak_factor, 
                       fit_degree_continuum=fit_degree_continuum, kernel_smooth_continuum=kernel_smooth_continuum,
                       return_dictionary_for_fitted_lines=True, 
                       xmin=_redshifted_emission_lines_[0]-20, 
                       xmax=_redshifted_emission_lines_[8]+20, 
                       **kwargs)
    
    emission_line_gauss_spectrum = out[1]
    _redshifted_lines_detected_dictionary_ = out[2]
    redshifted_lines_detected = [key for key in out[2]]
    
    # This is for range HeI 7065.28 - [SIII] 9068.90 (red)      Using 385R
    #                    Hg 4340.47 - [ArIII] 5191.82 (blue)    Using 580V 
    
    out=fit10gaussians(w, s, _redshifted_emission_lines_[9:], continuum=continuum, 
                       sigman=sigman,  max_wave_disp=max_wave_disp, min_peak_flux=min_peak_flux, 
                       max_sigma=max_sigma, max_peak_factor=max_peak_factor, min_peak_factor=min_peak_factor, 
                       fit_degree_continuum=fit_degree_continuum, kernel_smooth_continuum=kernel_smooth_continuum,
                       return_dictionary_for_fitted_lines=True, 
                       xmin=_redshifted_emission_lines_[9]-20, 
                       xmax=_redshifted_emission_lines_[-1]+20, 
                       **kwargs)
    
    emission_line_gauss_spectrum = emission_line_gauss_spectrum + out[1]
    _redshifted_lines_detected_dictionary_ = {**_redshifted_lines_detected_dictionary_, **out[2]}
    redshifted_lines_detected =redshifted_lines_detected + [key for key in out[2]]
    
    # Check results and build dictionary
    redshifted_lines_detected_dictionary= {}
    for line in _redshifted_emission_line_dictionary_:
        if _redshifted_emission_line_dictionary_[line][0] in  redshifted_lines_detected:
            redshifted_lines_detected_dictionary[line] ={}
            redshifted_lines_detected_dictionary[line]["ion"] =_redshifted_emission_line_dictionary_[line][1]
            redshifted_lines_detected_dictionary[line]["fl"] =_redshifted_emission_line_dictionary_[line][2]
            redshifted_lines_detected_dictionary[line]["gauss"] =_redshifted_lines_detected_dictionary_[_redshifted_emission_line_dictionary_[line][0]]  
                
    if verbose:
        z = redshifted_lines_detected_dictionary[l_ref]["gauss"][0]/l_ref -1
        print("  Identified {} emission line in this spectrum, with redshift = {:.6f} :".format(len(redshifted_lines_detected_dictionary), z))
        print("-------------------------------------------------------------------------")
        print("   line     ion       fl    Gauss center     peak     sigma         z")
        print("-------------------------------------------------------------------------")

        for line in  redshifted_lines_detected_dictionary:
            print(" {:8.2f}  {:8} {:6.3f}   {:9.2f}  {:10.2f}  {:7.3f}     {:.6f}".format(
                line, redshifted_lines_detected_dictionary[line]["ion"], 
                  redshifted_lines_detected_dictionary[line]["fl"], 
                  redshifted_lines_detected_dictionary[line]["gauss"][0],  
                  redshifted_lines_detected_dictionary[line]["gauss"][1],
                  redshifted_lines_detected_dictionary[line]["gauss"][2],
                  redshifted_lines_detected_dictionary[line]["gauss"][0]/line -1 ))
        print("-------------------------------------------------------------------------")
        
    return redshifted_lines_detected_dictionary, emission_line_gauss_spectrum
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def delete_unidentified_emission_lines(w, s, unidentified_lines=None, #ref_line="Ha", l_ref = None,
                                       continuum = None,
                                       redshifted_lines_detected_dictionary = None,
                                       sigman=2.5,  max_wave_disp=2, min_peak_flux=0, max_sigma=5, 
                                       max_peak_factor=None, min_peak_factor=None,
                                       fit_degree_continuum=None, kernel_smooth_continuum=None,
                                       index_fit_smooth_spectrum = 51, kernel_fit_smooth_spectrum =91,
                                       exclude_wlm = None, exclude_Ha=True,
                                       xmin_xmax_list = None,
                                       only_return_corrected_spectrum = True,
                                       **kwargs):
    verbose = kwargs.get('verbose', False)
    plot =  kwargs.get('plot', False)
    #warnings = kwargs.get('warnings', verbose)
    
    if unidentified_lines is None:
        unidentified_lines =[6284,6366, 6649.5, 7067,  7254, 7302,7400,  8471, 9255]   #6390 merged with good emission line,  7252,7257 merged in 7254

    if verbose: print("> Deleting unidentified emission lines:",unidentified_lines,"...")

# These are for He 2-10 in 385R  
# unidentified_lines =[6284,6366, 6649.5, 7067,  7254, 7302,7400,  8471, 9255]   #6390 merged with good emission line,  7252,7257 merged in 7254
  
#sky_residua = [7400], big telluric residual still there, telluric residua between 9000 and 9040
#others: 8471 -> OI 8446.486 ??

#plot_plot(w,fmax-fmaxfit, xmin=6250, xmax=6500, vlines = redshifted_lines_detected+[6366,6390], ymax=2000)
#plot_plot(w,fmax-fmaxfit, xmin=6500, xmax=6800, vlines = redshifted_lines_detected+[6366,6390, 6649.5], ymax=3000)
#plot_plot(w,fmax-fmaxfit, xmin=6800, xmax=7200, ymax=3000,vlines = redshifted_lines_detected+[6366,6390, 6649.5,7067])
#plot_plot(w,fmax-fmaxfit, xmin=7200, xmax=7450, ymax=3000,vlines = redshifted_lines_detected+[6366,6390, 6649.5,7067, 7252,7257,7302,7400])
#plot_plot(w,fmax-fmaxfit, xmin=7500, xmax=8300, ymax=3000,vlines = redshifted_lines_detected+[6366,6390, 6649.5,7067, 7252,7257,7302,7400])
#plot_plot(w,fmax-fmaxfit, xmin=8300, xmax=8600, ymax=3000,vlines = redshifted_lines_detected+[6366,6390, 6649.5,7067, 7252,7257,7302,7400,  8471])
#plot_plot(w,fmax-fmaxfit, xmin=8550, xmax=8900, ymax=3000,vlines = redshifted_lines_detected+[6366,6390, 6649.5,7067, 7252,7257,7302,7400,  8471,   8603, 8623,8642, 8776,8888])
#plot_plot(w,fmax-fmaxfit, xmin=8900, xmax=9320, ymax=3000,vlines = redshifted_lines_detected+[6366,6390, 6649.5,7067, 7252,7257,7302,7400,  8471,   8603, 8623,8642, 8776,8888, 9041,9255])
    

    # Computing continuum discarding Halpha region if needed
    if continuum is None:
        if exclude_Ha and exclude_wlm is None:
            Ha_wavelength =redshifted_lines_detected_dictionary[6562.82]["gauss"][0] 
            exclude_wlm = [[Ha_wavelength-45,Ha_wavelength+80]]
        
        _,continuum =fit_smooth_spectrum(w,s, index_fit=index_fit_smooth_spectrum, 
                                   kernel_fit=kernel_fit_smooth_spectrum, exclude_wlm=exclude_wlm, verbose = False, plot=False)

    if kwargs.get("percentile_min") is None: kwargs["percentile_min"] = 0
    if kwargs.get("percentile_max") is None: kwargs["percentile_max"] = 97
    out=fit10gaussians(w, s, unidentified_lines, continuum=continuum, 
                       sigman=sigman,  max_wave_disp=max_wave_disp, min_peak_flux=min_peak_flux, 
                       max_sigma=max_sigma, max_peak_factor=max_peak_factor, min_peak_factor=min_peak_factor, 
                       fit_degree_continuum=fit_degree_continuum, kernel_smooth_continuum=kernel_smooth_continuum,
                       return_dictionary_for_fitted_lines=True, **kwargs) 
    if plot:
        if redshifted_lines_detected_dictionary is not None:
            redshifted_lines_detected = [redshifted_lines_detected_dictionary[key]["gauss"][0] for key in redshifted_lines_detected_dictionary]
            vlines = redshifted_lines_detected+unidentified_lines
            cvlines=["k"]*len(redshifted_lines_detected) + ["orange"]*len(unidentified_lines)
            ptitle = "Deleted "+str(len(out[2]))+" unidentified emission lines\n (dashed = good emission lines, orange = unidentified emission lines)"
        else:
            vlines = unidentified_lines
            cvlines=["orange"]*len(unidentified_lines)
            ptitle = "Deleted "+str(len(out[2]))+" unidentified emission lines (orange)"
            
        if xmin_xmax_list is None:
            xmin_xmax_list =[[6200,6900], [6900,7500], [8400,9300]]
        for i in range(len(xmin_xmax_list)):  
            if 6562.82 > xmin_xmax_list[i][0] and 6562.82 < xmin_xmax_list[i][1]: 
                kwargs["percentile_max"] = 92
            else:
                kwargs["percentile_max"] = 97
                
            plot_plot(w,[s-continuum, out[0]-continuum], 
                      color=["r","b"], vlines = vlines,
                      ptitle = ptitle, cvlines=cvlines,
                      ylabel = "Flux - Continuum [counts]",
                      xmin=xmin_xmax_list[i][0],xmax=xmin_xmax_list[i][1], **kwargs)
    
    if only_return_corrected_spectrum:
        return out[0]
    else:
        return out  ### spectrum corrected [0] , gauss spectra [1], dictionary [2]
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def clip_spectrum_using_continuum(wavelength, spectrum, continuum=None, max_dispersion=1.5, 
                                  xmin=None, xmax=None, 
                                  interval_to_clean=None, wave_index_min = None, wave_index_max = None, 
                                  half_width_continuum = None,
                                  **kwargs): ### verbose =False, warnings = False, ):
    
    verbose = kwargs.get('verbose', False)
    plot =  kwargs.get('plot', False)
    #warnings = kwargs.get('warnings', False)
    
    if interval_to_clean is None:
        if wave_index_min is None: wave_index_min = wavelength[0]
        if wave_index_max is None: wave_index_max = wavelength[-1]
    else:
        if wave_index_min is None:      
            wave_index_min = np.abs(wavelength - interval_to_clean[0]).tolist().index(np.nanmin(np.abs(wavelength - interval_to_clean[0])))
        if wave_index_max is None:   
            wave_index_max = np.abs(wavelength - interval_to_clean[1]).tolist().index(np.nanmin(np.abs(wavelength - interval_to_clean[1]))) 
    
    if xmin is None:
        xmin =wavelength[0]
        if interval_to_clean is not None and half_width_continuum is not None:
            xmin = wavelength[int((wave_index_max+wave_index_min)/2 -  half_width_continuum)]
            if xmin < wavelength[0] : xmin= wavelength[0]        
    if xmax is None:
        xmax =wavelength[-1]
        if interval_to_clean is not None and half_width_continuum is not None:
            xmax = wavelength[int((wave_index_max+wave_index_min)/2 + half_width_continuum)]
            if xmax > wavelength[-1] : xmax= wavelength[-1] 
            
    if continuum is None:
        _,continuum= fit_smooth_spectrum(wavelength, spectrum,
                                         xmin=xmin, xmax=xmax,
                                         #mask = mask[i], #auto_trim=True, 
                                         #mask=[rss.koala_info["first_good_wave_per_fibre"][i], rss.koala_info["last_good_wave_per_fibre"][i]],
                                         #**kwargs) # 
                                         plot=False,  verbose=False)
        
    stat= basic_statistics(spectrum/continuum, x=wavelength, xmin=xmin, xmax=xmax, return_data=True, verbose = False)
    interval = spectrum[wave_index_min:wave_index_max]/continuum[wave_index_min:wave_index_max]
        
    fix_interval = [uniform(stat[1]-max_dispersion*stat[3],stat[1]+max_dispersion*stat[3]) if ( f < stat[1]-max_dispersion*stat[3] or f > stat[1]+max_dispersion*stat[3]) else f for f in interval]
    fix = copy.deepcopy(spectrum/continuum)
    fix[wave_index_min:wave_index_max] =fix_interval
    fix = fix * continuum
        
    if plot:
        if interval_to_clean is None:
            ptitle = "Clipping spectrum using max dispersion = "+str(max_dispersion)
        else:
            ptitle = "Clipping spectrum in interval ["+str(round(interval_to_clean[0],2))+" , "+str(round(interval_to_clean[1],2))+"] using max dispersion = "+str(max_dispersion)
        plot_plot(wavelength, [spectrum, fix, continuum, continuum*(1+max_dispersion*stat[3]), continuum*(1-max_dispersion*stat[3])], 
                  color=["red","blue","green","green","green"],
                  linestyle=["-","-","-","--","--"],
                  xmin=xmin, xmax=xmax, vlines=[wavelength[wave_index_min],wavelength[wave_index_max]],
                  ptitle=ptitle, extra_y=0.2) 
    
    return fix
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fit_clip(x, y, clip=0.4, index_fit = 2, kernel = 19, mask ="",                          
             xmin=None,xmax=None,ymin=None,ymax=None,percentile_min=2, percentile_max=98, extra_y = 0.1,
             ptitle=None, xlabel=None, ylabel = None, label="",
             hlines=[], vlines=[],chlines=[], cvlines=[], axvspan=[[0,0]], hwidth =1, vwidth =1,
             plot=True, verbose=True):
    """
    This tasks performs a polynomic fits of order index_fit
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    clip : TYPE, optional
        DESCRIPTION. The default is 0.4.
    index_fit : TYPE, optional
        DESCRIPTION. The default is 2.
    kernel : TYPE, optional
        DESCRIPTION. The default is 19.
    ptitle : TYPE, optional
        DESCRIPTION. The default is None.
    xlabel : TYPE, optional
        DESCRIPTION. The default is None.
    ylabel : TYPE, optional
        DESCRIPTION. The default is None.
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    fit, pp, y_fit, y_fit_clipped, x[idx], y_clipped[idx]: 
    The fit, 
    the polynomium with the fit, 
    the values of the fit for all x
    the values of the fit for only valid x
    the values of the valid x
    the values of the valid y
    """
        
    if ylabel is None: ylabel = "y (x)"
    
    # Preparing the data. Trim edges and remove nans
    x = np.array(x)
    y = np.array(y)
        
    if clip != 0 and kernel not in [0,1]:
    
        y_smooth = signal.medfilt(y, kernel)
        residuals = y - y_smooth
        residuals_std = np.std(residuals)
        
        y_nan = [np.nan if np.abs(i) > residuals_std*clip else 1. for i in residuals ] 
        y_clipped = y * y_nan
        
        idx = np.isfinite(x) & np.isfinite(y_clipped)
        
        fit  = np.polyfit(x[idx], y_clipped[idx], index_fit) 
        pp=np.poly1d(fit)
        y_fit=pp(x)
        y_fit_clipped =pp(x[idx])
   
        if verbose: 
            print("\n> Fitting a polynomium of degree",index_fit,"using clip =",clip,"* std to smoothed spectrum with kernel = ",kernel,"...")
            print("  Eliminated",len(x)-len(x[idx]),"outliers, the solution is: ",fit)
    
        if plot:
            if ptitle is None:
                ptitle = "Polyfit of degree "+str(index_fit)+" using clip = "+str(clip)+" * std to smoothed spectrum with kernel = "+str(kernel)
            if label == "":
                label =["Spectrum","Smoothed spectrum","Clipped spectrum","Fit"]
            
            plot_plot(x, [y,y_smooth, y_clipped, y_fit], psym=["+","-", "+","-"],
                      alpha=[0.5,0.5,0.8,1], color=["r","b","g","k"], label=label,
                      xlabel=xlabel, ylabel=ylabel, ptitle=ptitle, 
                      xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,percentile_min=percentile_min, percentile_max=percentile_max,extra_y=extra_y,
                      hlines=hlines, vlines=vlines,chlines=chlines, cvlines=cvlines, 
                      axvspan=axvspan, hwidth =hwidth, vwidth =vwidth)

        return fit, pp, y_fit, y_fit_clipped, x[idx], y_clipped[idx]   
    else:
        
        fit  = np.polyfit(x, y, index_fit) 
        pp=np.poly1d(fit)
        y_fit=pp(x)
        
        if verbose: print("\n> Fitting a polynomium of degree",index_fit,"...")
        
        if plot:
            if ptitle is None:
                ptitle = "Polyfit of degree "+str(index_fit)
            if label == "":
                label =["Spectrum", "Fit"]
            plot_plot(x, [y, y_fit], psym=["+","-"],
                      alpha=[0.5,1], color=["g","k"], label=label,
                      xlabel=xlabel, ylabel=ylabel, ptitle=ptitle, 
                      xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,percentile_min=percentile_min, percentile_max=percentile_max, extra_y=extra_y,
                      hlines=hlines, vlines=vlines,chlines=chlines, cvlines=cvlines, 
                      axvspan=axvspan, hwidth =hwidth, vwidth =vwidth)
        
        
        return fit, pp, y_fit, y_fit, x, y
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def trim_spectrum(w,s, edgelow=None,edgehigh=None, mask=None, auto_trim = True, 
                  exclude_wlm=None, verbose=True, plot=True):
    
    if edgelow is None or edgehigh is None : auto_trim = True
    if mask is not None:   # Mask is given as [edgelow,edgehigh]  If mask is given, use values of mask instead of edgelow edghigh
            edgelow = mask[0]
            edgehigh = len(w)-mask[1]+1
            if verbose: print("  Trimming the edges using the mask: [0:{}] and [{}:{}] ...".format(edgelow,len(w)-edgehigh, len(w)))  
    elif auto_trim:
        found = 0
        i = -1
        while found == 0:
            i=i+1
            if np.isnan(s[i]) == False: found = 1
            if i > len(s) : found =1
        edgelow = i
        i = len(s)
        found = 0
        while found == 0:
            i=i-1
            if np.isnan(s[i]) == False: found = 1
            if i == 0 : found =1
        edgehigh = len(w)-i
        if verbose: print("  Automatically trimming the edges [0:{}] and [{}:{}] ...".format(edgelow,len(w)-edgehigh, len(w)))          
    elif verbose: 
        print("  Trimming the edges [0:{}] and [{}:{}] ...".format(edgelow,len(w)-edgehigh, len(w)))  
        
    vlines=[w[edgelow], w[len(w)-edgehigh]]
    index=np.arange(len(w))
    valid_ind=np.where((index >= edgelow) & (index <= len(w)-edgehigh) & (~np.isnan(s)))[0]
    valid_w = w[valid_ind]
    valid_s = s[valid_ind] 
    
    if exclude_wlm is not None:    
        for rango in exclude_wlm :
            if verbose: print("  Trimming wavelength range [", rango[0],",", rango[1],"] ...")
            index=np.arange(len(valid_w))
            #not_valid_ind = np.where((valid_w[index] >= rango[0]) & (valid_w[index] <= rango[1]))[0]
            valid_ind = np.where((valid_w[index] <= rango[0]) | (valid_w[index] >= rango[1]))[0]  # | is OR
            valid_w = valid_w[valid_ind]
            valid_s = valid_s[valid_ind]
            vlines.append(rango[0])
            vlines.append(rango[1])
        # Recover valid_ind
        valid_ind =[]
        for i in range(len(s)):
            if w[i] in valid_w: valid_ind.append(i)
        
    if plot:
        ptitle ="Comparing original (red) and trimmed (blue) spectra"
        w_distance = w[-1]-w[0]  
        plot_plot([w,valid_w],[s,valid_s], ptitle=ptitle, vlines=vlines,
                  percentile_max=99.8,percentile_min=0.2,
                  xmin = w[0]-w_distance/50, xmax=w[-1]+w_distance/50)
    return valid_w, valid_s, valid_ind
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def force_continuum_to_zero(w,s, **kwargs):
    _,fit= fit_smooth_spectrum(w,s,  hlines=[0], **kwargs)
    return fit
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fit_smooth_spectrum(w,s, edgelow=None,edgehigh=None, mask=None, auto_trim = True,
                        kernel_correct_defects = 51, exclude_wlm=None, #[[0,0]],  #remove_nans = True,
                        kernel_fit=11, index_fit= 9,  clip_fit = 1.0, sigma_factor = 2.5,
                        maxit=10, 
                        plot_all_fits = False,
                        ptitle= "", fcal=False, **kwargs):  #verbose=True, plot=True, hlines=[1.], 
    """
    Apply f1,f2 = fit_smooth_spectrum(wl,spectrum) and returns:
    
    f1 is the smoothed spectrum, with edges 'fixed'
    f2 is the fit to the smooth spectrum
    Tasks that use this: correcting_negative_sky, plot_corrected_vs_uncorrected_spectrum
    """
    verbose = kwargs.get('verbose', True)
    plot =  kwargs.get('plot', True)
    kwargs['hlines'] = kwargs.get('hlines', [1.])

    if verbose: print('\n> Fitting an order {} polynomium to a spectrum smoothed with medfilt window of {}'.format(index_fit,kernel_fit))
    
    valid_w, valid_s, valid_ind= trim_spectrum(w,s, edgelow=edgelow,edgehigh=edgehigh, mask=mask, auto_trim = auto_trim,
                                                    exclude_wlm=exclude_wlm, verbose=verbose, plot=False)
    edgelow=valid_ind[0]
    edgehigh=valid_ind[-1]        
        
    valid_s_smooth = signal.medfilt(valid_s, kernel_fit)


    #iteratively clip and refit if requested
    if maxit > 1:
        niter=0
        stop=False
        fit_len=100# -100
        resid=0
        list_of_fits=[]
        while stop is False:
            #print '  Trying iteration ', niter,"..."
            fit_len_init=copy.deepcopy(fit_len)
            if niter == 0:
                fit_index=np.where(valid_s_smooth == valid_s_smooth)[0]
                fit_len=len(fit_index)
                sigma_resid=0.0
            if niter > 0:
                sigma_resid=MAD(resid)
                fit_index=np.where(np.abs(resid) < sigma_factor * sigma_resid)[0]  # sigma_factor was originally 4
                fit_len=len(fit_index)
            try:
                #print("  - Fitting between ", valid_w[fit_index][0],valid_w[fit_index][-1], " fit_len = ",fit_len)
                fit, pp, y_fit, y_fit_c, x_, y_c = fit_clip(valid_w[fit_index], valid_s_smooth[fit_index], index_fit=index_fit, clip = clip_fit, 
                                                            plot=False, verbose=False, kernel=kernel_fit)
                
                fx=pp(w)
                list_of_fits.append(fx)
                valid_fx=pp(valid_w)
                resid=valid_s_smooth-valid_fx
                #print niter,wl,fx, fxm
                #print "  Iteration {:2} results in RA: sigma_residual = {:.6f}, fit_len_init = {:5}  fit_len ={:5}".format(niter,sigma_resid,fit_len_init,fit_len)             
            except Exception:  
                if verbose: print('   - Skipping iteration ',niter)
            if (niter >= maxit) or (fit_len_init == fit_len): 
                if verbose: 
                    if niter >= maxit : print("   - Max iterations, {:2}, reached!".format(niter))
                    if fit_len_init == fit_len : print("  All interval fitted in iteration {} ! ".format(niter+1))
                stop = True    
            niter=niter+1
    else:
        fit, pp, y_fit, y_fit_c, x_, y_c = fit_clip(valid_w, valid_s, index_fit=index_fit, clip = clip_fit, 
                                                            plot=False, verbose=False, kernel=kernel_fit)          
        fx=pp(w)


    # reconstract smooth spectrum
    f = np.zeros_like(s)
    f[edgelow:edgehigh] = np.interp(w[edgelow:edgehigh], valid_w, valid_s_smooth)
    f[0:edgelow] = np.nan
    f[edgehigh:] = np.nan
    
    alpha=[0.1,0.3]
    if plot_all_fits and maxit > 1:
        fits_to_plot=[s,f]
        for item in list_of_fits:
            fits_to_plot.append(item)
            alpha.append(0.8)
        plot_plot(w,fits_to_plot, ptitle="All fits to the smoothed spectrum", 
                  vlines=[w[edgelow],w[edgehigh]], axvspan=exclude_wlm,
                  fcal=fcal,alpha=alpha, **kwargs)
                  #ymin=ymin, ymax=ymax)
               
    if plot:                   
        if ptitle == "" : ptitle= "Order "+str(index_fit)+" polynomium fitted to a spectrum smoothed with a "+str(kernel_fit)+" kernel window"
        kwargs['extra_y'] = kwargs.get('extra_y', 0.2)
        plot_plot(w, [s,f,fx], #ymin=ymin, ymax=ymax, 
                  color=["red","green","blue"], alpha=[0.2,0.5,0.5], 
                  label=["spectrum","smoothed","fit"], ptitle=ptitle, fcal=fcal, 
                  vlines=[w[edgelow],w[edgehigh]], axvspan=exclude_wlm, **kwargs)
      
    return f,fx
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def smooth_spectrum(wlm, s, wave_min=0, wave_max=0, step=50, exclude_wlm=[[0,0]], order=7,    
                    weight_fit_median=0.5, plot=False, verbose=False, fig_size=12): 
    """
    THIS IS NOT EXACTLY THE SAME THING THAT applying signal.medfilter()
    
    This needs to be checked, updated, and combine (if needed) with task fit_smooth_spectrum.
    The task gets the median value in steps of "step", gets an interpolated spectrum, 
    and fits a 7-order polynomy.
    
    It returns fit_median + fit_median_interpolated (each multiplied by their weights).
    
    Tasks that use this:  get_telluric_correction
    """

    if verbose: print("\n> Computing smooth spectrum...")

    if wave_min == 0 : wave_min = wlm[0]
    if wave_max == 0 : wave_max = wlm[-1]
        
    running_wave = []    
    running_step_median = []
    cuts=int( (wave_max - wave_min) /step)
   
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
def find_scale(f1, f2, x=None, step=0.0001, offset_min=0.01, offset_max=0.01,
               exclude_x = [[]], 
               fit_1D_points = 0,  clip=0.5, index_fit = 1, kernel = 21,
               xlabel="", ylabel ="", ymax="", ymin ="", 
               percentile_min="", percentile_max="", ptitle= "",
               plot = True, verbose = True):
    
    if x is not None: x_original = copy.deepcopy(x)
    
    if len(exclude_x[0]) > 0 and x is not None:
        x_, f1, ind = trim_spectrum(x,f1,  
                  exclude_wlm=exclude_x, verbose=False, plot=False, auto_trim=True)
        x, f2, ind = trim_spectrum(x,f2,  
                  exclude_wlm=exclude_x, verbose=False, plot=False, auto_trim=True)
        
    f1_m = np.nanmedian(f1)
    f2_m = np.nanmedian(f2)

    scale = f1_m / f2_m
    
    factores = np.arange(scale-offset_min, scale+offset_max, step)
    
    if fit_1D_points == 0 or x is None: 
        residua=[]    
        for factor in factores :
            residua.append(np.nansum(np.abs(f1-factor*f2)))
        
        residua_min = np.nanmin(residua)
        v = np.abs(residua-residua_min)
        index = v.tolist().index(np.nanmin(v))
        
        best_factor  = factores[index]
        
        if plot:
            if x is None: x = range(len(f1))
            if percentile_min == "" and  percentile_min == "" and ymax =="" and ymin =="":
                _y_min_ = np.nanmin([np.nanmin(f1), np.nanmin(f2*best_factor)] )
                _y_max_ = np.nanmax([np.nanmax(f1), np.nanmax(f2*best_factor)])
                rango = _y_max_-_y_min_
                ymin = _y_min_ - rango /20.
                ymax = _y_max_ + rango /20.
                
            
            round_value = -int(np.round(np.log10(step)))
            label2 = "Second spectrum scaled by "+str(np.round(best_factor,round_value))
            if ptitle == "" :ptitle = "Scaling second spectrum by "+str(np.round(best_factor,round_value))
            plot_plot(x,[f1,f2*best_factor], ylabel = ylabel, xlabel=xlabel,
                      ymin=ymin, ymax=ymax, percentile_min=percentile_min, percentile_max=percentile_max, 
                      ptitle=ptitle, label=["First spectrum", label2])
            
        if verbose: print("\n> The scale that minimize residua between f1 and f2 is",best_factor,"for f2")
        return best_factor
    
    else:
        best_factor_list =[]
        x_list =[]
        block_length = int(len(x)/fit_1D_points)
        x_divided=[]
        f1_divided =[]
        f2_divided =[]
        
        for i in range(fit_1D_points):
            x_divided.append(x[i*block_length:(i+1)*block_length])
            x_list.append(np.nanmedian(x[i*block_length:(i+1)*block_length]))
            f1_divided.append(f1[i*block_length:(i+1)*block_length])
            f2_divided.append(f2[i*block_length:(i+1)*block_length])

        for i in range(len(x_divided)) :
            residua=[]    
            for factor in factores :
                residua.append(np.nansum(np.abs(f1_divided[i]-factor*f2_divided[i])))
            
            residua_min = np.nanmin(residua)
            v = np.abs(residua-residua_min)
            index = v.tolist().index(np.nanmin(v))
            
            best_factor_list.append(factores[index])
            
        if percentile_min == "" and  percentile_min == "" and ymax =="" and ymin =="" and plot :
            _y_min_ = np.nanmin(best_factor_list)
            _y_max_ = np.nanmax(best_factor_list)
            rango = _y_max_-_y_min_
            ymin = _y_min_ - rango /20.
            ymax = _y_max_ + rango /20.
                           
        fit, pp, y_fit, y_fit_clipped, x_clipped, y_clipped = fit_clip(x_list, best_factor_list, 
                                                                       clip=clip, index_fit = index_fit, kernel = kernel, #mask ="",                          
                                                                       ymin=ymin, ymax=ymax,
                                                                       ptitle=None, xlabel=xlabel, label="",
                                                                       ylabel = "Scale factor",  
                                                                       hlines=[], vlines=[],chlines=[], cvlines=[], axvspan=[[0,0]], hwidth =1, vwidth =1,
                                                                       plot=plot, verbose=verbose)
            
        if verbose: print("\n> The 1D fit that scale the second spectra is: ",fit)
        scale_1D = pp(x_original)
        return   scale_1D          


        
        
        
        
        
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
# def scale_sky_spectrum(wlm, sky_spectrum, spectra, cut_sky=4., fmax=10, fmin=1, valid_wave_min=0, valid_wave_max=0, 
#                        fibre_list=[100,200,300,400,500,600,700,800,900], plot=True, verbose=True, warnings=True):
#     """
#     This task needs to be checked.
#     Using the continuum, the scale between 2 spectra can be determined runnning
#     auto_scale_two_spectra()
#     """    
    
# # # Read sky lines provided by 2dFdr
# #    sky_line_,flux_sky_line_ = read_table("sky_lines_2dfdr.dat", ["f", "f"] )
# # # Choose those lines in the range
# #    sky_line=[]
# #    flux_sky_line=[]
# #    valid_wave_min = 6240
# #    valid_wave_max = 7355
# #    for i in range(len(sky_line_)):
# #        if valid_wave_min < sky_line_[i] < valid_wave_max:
# #            sky_line.append(sky_line_[i])
# #            flux_sky_line.append(flux_sky_line_[i])
            
            
#     if valid_wave_min == 0: valid_wave_min = wlm[0]
#     if valid_wave_max == 0: valid_wave_max = wlm[-1]
        
#     if verbose: print("\n> Identifying sky lines using cut_sky =",cut_sky,", allowed SKY/OBJ values = [",fmin,",",fmax,"]")
#     if verbose: print("  Using fibres = ",fibre_list)

#     peaks,peaks_name,peaks_rest,continuum_limits=search_peaks(wlm,sky_spectrum, plot=plot, cut=cut_sky, fmax=fmax, only_id_lines=False, verbose=False)   

#     ratio_list=[]
#     valid_peaks=[]
        
#     if verbose: print("\n      Sky line     Gaussian ratio      Flux ratio")
#     n_sky_lines_found=0
#     for i in range(len(peaks)):
#         sky_spectrum_data=fluxes(wlm,sky_spectrum, peaks[i], fcal=False, lowlow=50,highhigh=50, plot=False, verbose=False, warnings=False)
 
#         sky_median_continuum = np.nanmedian(sky_spectrum_data[11])
               
#         object_spectrum_data_gauss=[]
#         object_spectrum_data_integrated=[] 
#         median_list=[]
#         for fibre in fibre_list:   
#             object_spectrum_flux=fluxes(wlm, spectra[fibre], peaks[i], fcal=False, lowlow=50,highhigh=50, plot=False, verbose=False, warnings=False)
#             object_spectrum_data_gauss.append(object_spectrum_flux[3])       # Gaussian flux is 3
#             object_spectrum_data_integrated.append(object_spectrum_flux[7])  # integrated flux is 7
#             median_list.append(np.nanmedian(object_spectrum_flux[11]))
#         object_spectrum_data=np.nanmedian(object_spectrum_data_gauss)
#         object_spectrum_data_i=np.nanmedian(object_spectrum_data_integrated)
        
#         object_median_continuum=np.nanmin(median_list)     
        
#         if fmin < object_spectrum_data/sky_spectrum_data[3] *  sky_median_continuum/object_median_continuum    < fmax :
#             n_sky_lines_found = n_sky_lines_found + 1
#             valid_peaks.append(peaks[i])
#             ratio_list.append(object_spectrum_data/sky_spectrum_data[3])
#             if verbose: print("{:3.0f}   {:5.3f}         {:2.3f}             {:2.3f}".format(n_sky_lines_found,peaks[i],object_spectrum_data/sky_spectrum_data[3], object_spectrum_data_i/sky_spectrum_data[7]))  


#     #print "ratio_list =", ratio_list
#     #fit = np.polyfit(valid_peaks, ratio_list, 0) # This is the same that doing an average/mean
#     #fit_line = fit[0]+0*wlm
#     fit_line =np.nanmedian(ratio_list)  # We just do a median
#     #fit_line = fit[1]+fit[0]*wlm
#     #fit_line = fit[2]+fit[1]*wlm+fit[0]*wlm**2
#     #fit_line = fit[3]+fit[2]*wlm+fit[1]*wlm**2+fit[0]*wlm**3
   
    
#     if plot:
#         plt.plot(valid_peaks,ratio_list,"+")
#         #plt.plot(wlm,fit_line)
#         plt.axhline(y=fit_line, color='k', linestyle='--')
#         plt.xlim(valid_wave_min-10, valid_wave_max+10)      
#         #if len(ratio_list) > 0:
#         plt.ylim(np.nanmin(ratio_list)-0.2,np.nanmax(ratio_list)+0.2)
#         plt.title("Scaling sky spectrum to object spectra")
#         plt.xlabel("Wavelength [$\mathrm{\AA}$]")
#         plt.ylabel("OBJECT / SKY")
#         plt.minorticks_on()
#         plt.show()
#         plt.close()
        
#         if verbose: print("  Using this fit to scale sky spectrum to object, the median value is ",np.round(fit_line,3),"...")      
    
#     sky_corrected = sky_spectrum  * fit_line

# #        plt.plot(wlm,sky_spectrum, "r", alpha=0.3)
# #        plt.plot(wlm,sky_corrected, "g", alpha=0.3)
# #        plt.show()
# #        plt.close()
    
#     return sky_corrected, np.round(fit_line,3)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------- 
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------- 
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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_continuum_in_range(w,s,low_low, low_high, high_low, high_high,
                           pmin=12,pmax=88, only_correct_negative_values = False,
                           return_fit = False,
                           fit_degree=2, plot = True, verbose = True, warnings=True)  :
    """
    This task computes the continuum of a 1D spectrum using the intervals [low_low, low_high] 
    and [high_low, high_high] and returns the spectrum but with the continuum in the range
    [low_high, high_low] (where a feature we want to remove is located).
    """
    s = [np.nanmedian(s) if np.isnan(x) else x for x in s]
    s = np.array(s)
    w = np.array(w)
    
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
            
        
        if return_fit:
            corrected_s_ = y_fitted
        else:
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
            ptitle = "Correction in range  "+str(np.round(low_low,2))+" - [ "+str(np.round(low_high,2))+" - "+str(np.round(high_low,2))+" ] - "+str(np.round(high_high,2))
            plot_plot(w_fit,[y_fit,y_fitted,y_fitted-highlimit,y_fitted-lowlimit,corrected_s_], color=["r","b", "black","black","green"], alpha=[0.3,0.7,0.2,0.2,0.5],xmin=low_low-40, xmax=high_high+40,vlines=[low_low,low_high,high_low,high_high],ptitle=ptitle, ylabel="Normalized flux")  
            #plot_plot(w,[s,corrected_s],xmin=low_low-40, xmax=high_high+40,vlines=[low_low,low_high,high_low,high_high])
    except Exception:
        if warnings: print("  Fitting the continuum failed! Nothing done.")
        corrected_s = s

    return corrected_s

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------       
# def fix_these_features(w,s,features, sky_fibres=None, sky_spectrum=None, objeto=None, plot_all = False): #objeto=test):
            
#     ff=copy.deepcopy(s)
    
#     kind_of_features = [features[i][0] for i in range(len(features))]
# #    if "g" in kind_of_features or "s" in kind_of_features:
# #        if len(sky_spectrum) == 0 :             
# #            sky_spectrum=objeto.plot_combined_spectrum(list_spectra=sky_fibres, plot=False, median=True)
#     if "s" in kind_of_features:
#         if sky_spectrum is None and sky_fibres is not None:             
#             sky_spectrum=plot_combined_spectrum(objeto, list_spectra=sky_fibres, plot=plot_all, median=True, r=True)

                        
#     for feature in features:
#         #plot_plot(w,ff,xmin=feature[1]-20,xmax=feature[4]+20)
#         if feature[0] == "l":   # Line
#             resultado = fluxes(w,ff, feature[1], lowlow=feature[2],lowhigh=feature[3],highlow=feature[4],highhigh=feature[5],broad=feature[6],plot=feature[7],verbose=feature[8])
#             ff=resultado[11] 
#         if feature[0] == "r":   # range
#             ff = get_continuum_in_range(w,ff,feature[1],feature[2],feature[3],feature[4],pmin=feature[5],pmax=feature[6],fit_degree=feature[7],plot=feature[8],verbose=feature[9])
#         if feature[0] == "g":   # gaussian           
# #            resultado = fluxes(w,sky_spectrum, feature[1], lowlow=feature[2],lowhigh=feature[3],highlow=feature[4],highhigh=feature[5],broad=feature[6],plot=feature[7],verbose=feature[8])
# #            sky_feature=sky_spectrum-resultado[11]
#             resultado = fluxes(w,s, feature[1], lowlow=feature[2],lowhigh=feature[3],highlow=feature[4],highhigh=feature[5],broad=feature[6],plot=feature[7],verbose=feature[8])
#             sky_feature=s-resultado[11]
#             ff = ff - sky_feature
#         if feature[0] == "n":    # negative values
#             ff = get_continuum_in_range(w,ff,feature[1],feature[2],feature[3],feature[4],pmin=feature[5],pmax=feature[6],fit_degree=feature[7],plot=feature[8],verbose=feature[9],only_correct_negative_values = True)
#         if feature[0] == "s":    # sustract
#             ff_low = ff[np.where(w < feature[2])]
#             ff_high = ff[np.where(w > feature[3])]
#             subs = ff - sky_spectrum
#             ff_replace = subs[np.where((w >= feature[2]) & (w <= feature[3]))]
#             ff_ = np.concatenate((ff_low,ff_replace))
#             ff_ = np.concatenate((ff_,ff_high))
            
#             c = get_continuum_in_range(w,ff_,feature[1],feature[2],feature[3],feature[4],pmin=feature[5],pmax=feature[6],fit_degree=feature[7],plot=feature[8],verbose=feature[9],only_correct_negative_values=True)

            
#             if feature[8] or plot_all : #plot
#                 vlines=[feature[1],feature[2],feature[3],feature[4]]
#                 plot_plot(w,[ff, ff_,c],xmin=feature[1]-10,xmax=feature[4]+10,vlines=vlines,alpha=[0.1,0.2,0.8],ptitle="Correcting 's'")
            
#             ff=c    

#     return ff   
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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
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
        if line_wavelength != 0 : ptitle="Cosmic identification in cut at "+str(line_wavelength)+" $\mathrm{\AA}$"        
        plot_plot(x,verde, ymin=0,ymax=200, hlines=[cosmic_higher_than], ptitle=ptitle,  ylabel="abs (cut - medfilt(cut)) - extra_factor * max_val")
 
    if verbose:
        if line_wavelength == 0:
            print("\n> Identified", len(cosmics_list),"cosmics in fibres",cosmics_list)
        else:
            print("\n> Identified", len(cosmics_list),"cosmics at",str(line_wavelength),"A in fibres",cosmics_list)
    return cosmics_list
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fit_clip_old(x, y, clip=0.4, index_fit = 2, kernel = 19, mask ="",                          
             xmin="",xmax="",ymin="",ymax="",percentile_min=2, percentile_max=98,
             ptitle=None, xlabel=None, ylabel = None, label="", 
             hlines=[], vlines=[],chlines=[], cvlines=[], axvspan=[[0,0]], hwidth =1, vwidth =1,
             plot=True, verbose=True):
    """
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    clip : TYPE, optional
        DESCRIPTION. The default is 0.4.
    index_fit : TYPE, optional
        DESCRIPTION. The default is 2.
    kernel : TYPE, optional
        DESCRIPTION. The default is 19.
    ptitle : TYPE, optional
        DESCRIPTION. The default is None.
    xlabel : TYPE, optional
        DESCRIPTION. The default is None.
    ylabel : TYPE, optional
        DESCRIPTION. The default is None.
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    fit, pp, y_fit, y_fit_clipped, x[idx], y_clipped[idx]: 
    The fit, 
    the polynomium with the fit, 
    the values of the fit for all x
    the values of the fit for only valid x
    the values of the valid x
    the values of the valid y
    """
     
    if kernel != 0:
        x = np.array(x)
        y = np.array(y)
        
        y_smooth = signal.medfilt(y, kernel)
        residuals = y - y_smooth
        residuals_std = np.std(residuals)
        
        y_nan = [np.nan if np.abs(i) > residuals_std*clip else 1. for i in residuals ] 
        y_clipped = y * y_nan
        
        idx = np.isfinite(x) & np.isfinite(y_clipped)
        
        fit  = np.polyfit(x[idx], y_clipped[idx], index_fit) 
        pp=np.poly1d(fit)
        y_fit=pp(x)
        y_fit_clipped =pp(x[idx])
   
        if verbose: 
            print("\n> Fitting a polynomium of degree",index_fit,"using clip =",clip,"* std ...")
            print("  Eliminated",len(x)-len(x[idx]),"outliers, the solution is: ",fit)
        
        if plot:
            if ylabel is None: ylabel = "y (x)"
            
            if ptitle is None:
                ptitle = "Polyfit of degree "+str(index_fit)+" using clip = "+str(clip)+" * std"
            plot_plot(x, [y,y_smooth, y_clipped, y_fit], psym=["+","-", "+","-"],
                      alpha=[0.5,0.5,0.8,1], color=["r","b","g","k"], label=label,
                      xlabel=xlabel, ylabel=ylabel, ptitle=ptitle, 
                      xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,percentile_min=percentile_min, percentile_max=percentile_max,
                      hlines=hlines, vlines=vlines,chlines=chlines, cvlines=cvlines, 
                      axvspan=axvspan, hwidth =hwidth, vwidth =vwidth)

        return fit, pp, y_fit, y_fit_clipped, x[idx], y_clipped[idx]   
    else:
        fit  = np.polyfit(x, y, index_fit) 
        pp=np.poly1d(fit)
        y_fit=pp(x)
        return fit, pp, y_fit, y_fit, x, y
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def correct_defects(spectrum, w=[], only_nans = True, 
                    kernel_correct_defects = 51,
                    plot=False, verbose=False):
    
    s = copy.deepcopy(spectrum)
    if only_nans:
        # Fix nans & inf
        s = [0 if np.isnan(x) or np.isinf(x) else x for x in s]  
    else:
        # Fix nans, inf & negative values = 0
        s = [0 if np.isnan(x) or x < 0. or np.isinf(x) else x for x in s]  
    s_smooth = medfilt(s, kernel_correct_defects)

    bad_indices = [i for i, x in enumerate(s) if x == 0]
    for index in bad_indices:
        s[index] = s_smooth[index]  # Replace 0s for median value
    
    if plot and len(w) > 0:
        plot_plot(w, [spectrum,s_smooth,s], ptitle="Comparison between old (red) and corrected (green) spectrum.\n Smoothed spectrum in blue.")
    return s
    
    