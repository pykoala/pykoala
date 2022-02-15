#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# IN / OUT TASKS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# This file has 4 sections:
#    1. Generic In / Out tasks
#    2. RSS In / Out tasks
#    3. Cube In / Out tasks
#    4. Map In / Out tasks


from astropy.io import fits
#from astropy.io import fits as pyfits 

import numpy as np
import sys
import os
import datetime

import glob

# Disable some annoying warnings
import warnings

from koala.constants import red_gratings, blue_gratings
#from koala.cube import create_mask

warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.CRITICAL)
from koala._version import get_versions
version = get_versions()["version"]
del get_versions


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
  
sys.path.append(os.path.join(parent, 'RSS'))
sys.path.append(os.path.join(parent, 'cube'))
sys.path.append(os.path.join(parent, 'automatic_scripts'))
# sys.path.append('../cube')
# sys.path.append('../automatic_scripts')

# from koala_reduce import KOALA_reduce
# from InterpolatedCube import Interpolated_cube
# from koala_rss import KOALA_RSS


developers = 'Developed by Angel Lopez-Sanchez, Pablo Corcho-Caballero,\
 Yago Ascasibar, Lluis Galbany, Barr Perez, Nathan Pidcock,\
 Diana Dalae, Giacomo Biviano, Adithya Gudalur Balasubramania,\
 Blake Staples, Taylah Beard, Matt Owers, James Tocknell et al.'


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
    if path != "": 
        if path[-1] != "/" : path = path+"/" # If path does not end in "/" it is added
        
        if len(filename.replace("/","")) == len(filename):
            if verbose: print("\n> Variable {} does not include the full path {}".format(filename,path))
            fullpath = path+filename
        else:
            if verbose: print("\n> Variable {} includes the full path {}".format(filename,path))
            fullpath = filename
    else:
        fullpath = filename
        if verbose: print("  The path has not been given, returning {} ...".format(filename))
    return fullpath
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
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
        if fmt[i] == "i" : data[i]=np.loadtxt(file, skiprows=0, unpack=True, usecols=[i], dtype=int)
        if fmt[i] == "s" : data[i]=np.loadtxt(file, skiprows=0, unpack=True, usecols=[i], dtype=str)
        if fmt[i] == "f" : data[i]=np.loadtxt(file, skiprows=0, unpack=True, usecols=[i], dtype=float)    
    return data
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
        escribe = np.str(data[i])+" \n"
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

# Some have been moved to RSS class to avoid conflicts with KOALA_RSS
# and not enter in a loop when importing

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def save_rss_fits(rss, data=[[0],[0]], fits_file="RSS_rss.fits", text="RSS data", sol=None,
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
                 
    if sol != None: # len(sol) > 0:  # sol has been provided
        fits_image_hdu.header["SOL0"] = sol[0]
        if len(sol) > 1: fits_image_hdu.header["SOL1"] = sol[1]
        if len(sol) > 2: fits_image_hdu.header["SOL2"] = sol[2]
     
    if description == "":
        description = rss.object
    fits_image_hdu.header['DESCRIP'] = description

    for item in rss.history_RSS:
        if item == "- Created fits file (this file) :":
            fits_image_hdu.header['HISTORY'] = "- Created fits file :"
        else:    
            fits_image_hdu.header['HISTORY'] = item        
    fits_image_hdu.header['FILE_IN'] = rss.filename     

    fits_image_hdu.header['HISTORY'] = '-- RSS processing using PyKOALA '+ version
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
    
    if filename[-8:] == "red.fits" :
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
            
    if filename[-8:] == "red.fits" :       
        return filename[0:-5]+clave+".fits"
    else:
        return filename[0:-15]+clave+".fits"
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
        
    
    if verbose: print("\n> Listing 2dFdr fits files in folder",path,":\n")
    
    if path[-1] != "/" : path=path+"/"
    date_ = ''  # TODO: This must be filled somehow. It is just for printing data
    files_ = glob.glob(path + '*.fits')
    if len(files_) == 0:
        raise NameError('No files found within folder '+path)
 
    # Ignore fits products from 2dFdr, darks, flats, arcs...
    files=[]
    for fitsName in sorted(files_):
        include_this = True
        if fitsName[-8:] == "tlm.fits" : include_this = False
        if fitsName[-7:] == "im.fits" : include_this = False
        if fitsName[-7:] == "ex.fits" : include_this = False  
        
        if include_this: 
            hdulist = fits.open(fitsName)
            try:
                object_class = hdulist[0].header['NDFCLASS']
                if object_class ==  "MFOBJECT": 
                    files.append(fitsName) 
                #else: print(object_class, fitsName)
            except Exception:
                nothing=0

    
    for fitsName in sorted(files):
                
        check_file = True
        if fitsName[-8:] != "red.fits" : 
            check_file = False
        if fitsName[0:8] == "combined" and check_file == False: 
            check_file = True
        for skyflat_name in skyflat_names:
            if skyflat_name in fitsName : check_file = True
        
        hdulist = fits.open(fitsName)   # it was pyfits

        object_fits = hdulist[0].header['OBJECT'].split(" ")
    
        if object_fits[0] in ["HD", "NGC", "IC"] or use2:
            try:
                if not ignore_offsets:
                    object_fits[0]=object_fits[0]+object_fits[1]
                elif object_fits[1] not in ignore_list:
                    object_fits[0]=object_fits[0]+object_fits[1]
            except Exception:
                nothing=0
        if use3:
            try:
                if not ignore_offsets:
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
            if not found:
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
                        
        print("\n  They were obtained on {} using the grating {}".format(date, grating))

    if return_list:
        return list_of_objetos, list_of_files, list_of_exptimes, date, grating
    if nothing > 10:
        print(nothing)  # Stupid thing for controlling Exceptions


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 3. Cube In / Out tasks
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# def read_cube()          # MOVED TO cube.py to avoid loops when importing

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
        
    fits_image_hdu.header['HISTORY'] =  version 
    fits_image_hdu.header['HISTORY'] =  developers    
        
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
                fits_image_hdu.header[text] = cube.rss_file_list[i]    
    else: 
        #print(" THIS IS NOT A COMBINED CUBE")
        fits_image_hdu.header["COMCUBE"] = False 
        is_combined_cube = False
        fits_image_hdu.header['COFILES'] = 1
        fits_image_hdu.header["RSS_01"] = cube.rss_file_list[0] 

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
# -----------------------------------------------------------------------------
# This is OLD and needs to be updated if ever needed!
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
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 4. Map In / Out tasks
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
def save_map(cube, mapa, fits_file, mask=[], description="", path="", verbose = True):
    
    if path != "" : fits_file=full_path(fits_file,path)

    if description == "" : description =mapa[0]

    fits_image_hdu = fits.PrimaryHDU(mapa[1])
         
    fits_image_hdu.header['HISTORY'] = 'Map created by PyKOALA'        
    fits_image_hdu.header['HISTORY'] =  version
    fits_image_hdu.header['HISTORY'] =  developers    
    
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
# def load_map(mapa_fits, description="", path="", verbose = True):
    
#     if verbose: print("\n> Reading map(s) stored in file", mapa_fits,"...")
        
#     if path != "" : mapa_fits=full_path(mapa_fits,path)
#     mapa_fits_data = fits.open(mapa_fits)  # Open file

#     if description == "" : description = mapa_fits_data[0].header['DESCRIP']    #
#     if verbose: print("- Description stored in [0]")

#     intensity_map = mapa_fits_data[0].data

#     try:
#         vel_map = mapa_fits_data[1].data
#         fwhm_map = mapa_fits_data[2].data
#         ew_map = mapa_fits_data[3].data
#         mapa = [description, intensity_map, vel_map, fwhm_map, ew_map]
#         if verbose: 
#             print("  This map comes from a Gaussian fit: ")
#             print("- Intensity map stored in [1]")
#             print("- Radial velocity map [km/s] stored in [2]")
#             print("- FWHM map [km/s] stored in [3]")
#             print("- EW map [A] stored in [4]")

#     except Exception:
#         if verbose: print("- Map stored in [1]")
#         mapa = [description, intensity_map]
    
#     fail=0
#     try:
#         for i in range(4):
#             try:
#                 mask_name1="MASK"+np.str(i+1)+"1" 
#                 mask_name2="MASK"+np.str(i+1)+"2" 
#                 mask_low_limit = mapa_fits_data[0].header[mask_name1] 
#                 mask_high_limit = mapa_fits_data[0].header[mask_name2] 
#                 _mask_ = create_mask(mapa_fits_data[i].data, low_limit=mask_low_limit,  high_limit=mask_high_limit, verbose=False)
#                 mapa.append(_mask_)
#                 if verbose: print("- Mask with good values between {} and {} created and stored in [{}]".format(mask_low_limit,mask_high_limit,len(mapa)-1))
#             except Exception:
#                 fail=fail+1                    
            
#     except Exception:
#         if verbose: print("- Map does not have any mask.")
        

#     return mapa
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------