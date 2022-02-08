#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Map TASKS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

from astropy.io import fits

import numpy as np
import sys
import os
import datetime
import copy

from scipy.ndimage.interpolation import shift
from koala.constants import red_gratings, blue_gratings, C
from koala.io import full_path, version, developers
from koala.onedspec import fluxes

# Disable some annoying warnings
import logging
import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.CRITICAL)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------  
def create_map(cube, line, w2=None, gaussian_fit = False, gf=False,
               lowlow= 50, lowhigh=10, highlow=10, highhigh = 50, no_nans = False,
               show_spaxels=[], verbose = True, description = "" ):
    """
    This function creates a map given inputs.

    Parameters
    ----------
    cube : Cube Object
        This is the inputted Cube Object.
    line : TYPE
        DESCRIPTION.
    w2 : TYPE, Float
        DESCRIPTION. The default is 0..
    gaussian_fit : Boolean, optional
        DESCRIPTION. The default is False.
    gf : Boolean, optional
        DESCRIPTION. The default is False.
    lowlow : Integer, optional
        DESCRIPTION. The default is 50.
    lowhigh : Integer, optional
        DESCRIPTION. The default is 10.
    highlow : Integer, optional
        DESCRIPTION. The default is 10.
    highhigh : Integer, optional
        DESCRIPTION. The default is 50.
    no_nans : Boolean, optional
        This is to say if there are nans or not. The default is False.
    show_spaxels : List, optional
        DESCRIPTION. The default is [].
    verbose : Boolean, optional
        Print results. The default is True.
    description : String, optional
        This is the description of the cube. The default is "".

    Returns
    -------
    description : TYPE
        This is the description of the cube.
    interpolated_map : TYPE
        DESCRIPTION.
    w1 : TYPE
        DESCRIPTION.
    w2 : TYPE
        DESCRIPTION.

    """
  
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
        if w2 is None:
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
def save_map(cube, mapa, fits_file, mask=[], description="", path="", verbose = True):
    
    if path != "" : fits_file=full_path(fits_file,path)

    if description == "" : description =mapa[0]

    fits_image_hdu = fits.PrimaryHDU(mapa[1])
         
    fits_image_hdu.header['HISTORY'] = 'Map created by PyKOALA'        
    # fits_image_hdu.header['HISTORY'] = 'Developed by Angel Lopez-Sanchez, Pablo Corcho-Caballero,'
    # fits_image_hdu.header['HISTORY'] = 'Yago Ascasibar, Lluis Galbany, Barr Perez, Nathan Pidcock'
    # fits_image_hdu.header['HISTORY'] = 'Diana Dalae, Giacomo Biviano, Adithya Gudalur Balasubramania,'
    # fits_image_hdu.header['HISTORY'] = 'Blake Staples, Taylah Beard, Matt Owers, James Tocknell et al.'
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
def create_map_mask(mapa, low_limit, high_limit=1E20, plot = False, verbose = True):
    
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
                _mask_ = create_map_mask(mapa_fits_data[i].data, low_limit=mask_low_limit,  high_limit=mask_high_limit, verbose=False)
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
def shift_map(mapa, delta_RA=0, delta_DEC=0, pixel_size_arcsec = None,
              verbose = True):
    
    if verbose: print('\n> Shifting map in RA = {:.2f}" and DEC = {:.2f}" ...'.format(delta_RA, delta_DEC))
        
    if pixel_size_arcsec is None:
        print('  - Pixel_size_arcsec is not provided, assuming 0.7"...')
        pixel_size_arcsec = 0.7
    
    mapa_shifted = copy.deepcopy(mapa) * 0.
    tmp=copy.deepcopy(mapa)
    mask=copy.deepcopy(mapa)*0.
    mask[np.where( np.isnan(tmp) == False  )]=1.      # Nans stay the same, when a good value = 1.
    tmp_nonan=np.nan_to_num(tmp, nan=np.nanmedian(tmp))  # map without nans, replaced for median value
    mapa_shifted[:,:]=mask*shift(tmp_nonan[:,:],[delta_DEC/pixel_size_arcsec, delta_RA/pixel_size_arcsec], cval=np.nan)
    
    return mapa_shifted
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------