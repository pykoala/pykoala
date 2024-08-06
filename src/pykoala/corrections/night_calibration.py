# =============================================================================
# Basics packages
# =============================================================================
import os
#from os import path
#import numpy as np
#import copy
#import sys
#from astropy.io import fits
#from scipy.ndimage import median_filter
#from scipy.ndimage import gaussian_filter
#import matplotlib.pyplot as plt
#from scipy.signal import medfilt
# =============================================================================
# Astropy and associated packages
# =============================================================================
#from astropy.io import fits
#from astropy.wcs import WCS
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala.corrections.flux_calibration import FluxCalibration
from pykoala.corrections.throughput import Throughput
from pykoala.corrections.wavelength import WavelengthShiftCorrection
from pykoala.corrections.sky import TelluricCorrection
from pykoala.instruments.koala_ifu import red_gratings #, blue_gratings
# #-----------------------------------------------------------------------------
# %% ============================================================================
# #-----------------------------------------------------------------------------
class NIGHT_CALIBRATION(object):
    def __init__(self, **kwargs):
        self.date = kwargs.get('date', None)
        self.grating = kwargs.get('grating', None)
        self.wavelength = kwargs.get('wavelength', None)
        self.throughput = kwargs.get('throughput', None)
        self.wavelength_shift_correction = kwargs.get('wavelength_shift_correction', None)
        self.telluric_correction = kwargs.get('telluric_correction', None)
        self.flux_calibration = kwargs.get('flux_calibration', None)
        self.star_list = kwargs.get('star_list', None)
        path = kwargs.get('path', None)
        verbose = kwargs.get('verbose', False)

        #self.filename = kwargs.get('filename', None)
        #if self.filename is not None:
        #    if path is not None: self.filename = os.path.join(path,self.filename)
            #if verbose: print("\n> Reading files for calibration of the night:")
        if self.throughput is not None:
            if type(self.throughput) == str:
                if path is not None: self.throughput = os.path.join(path,self.throughput)
                if verbose: print(" - Reading throughtput calibration from file",self.throughput)
                self.throughput = Throughput(path=self.throughput)
                
        if self.wavelength_shift_correction is not None:
            if str(type(self.wavelength_shift_correction))[-5:-2] == "str":  ##  [-27:-2] #"WavelengthShiftCorrection":
                if path is not None: self.wavelength_shift_correction = os.path.join(path,self.wavelength_shift_correction)
                if verbose: print(" - Reading wavelength shift correction from file",self.wavelength_shift_correction)
                self.wavelength_shift_correction = WavelengthShiftCorrection(fits_file=self.wavelength_shift_correction)
                if self.wavelength is None: self.wavelength = self.wavelength_shift_correction.wavelength
                if self.date is None: self.date = self.wavelength_shift_correction.header["UTDATE"]
                if self.grating is None: self.grating = self.wavelength_shift_correction.header["GRATID"]
        
        if self.telluric_correction is not None:
            if str(type(self.telluric_correction))[-5:-2] == "str":  
                if path is not None: self.telluric_correction = os.path.join(path,self.telluric_correction)
                if verbose: print(" - Reading telluric correction from file",self.telluric_correction)
                self.telluric_correction = TelluricCorrection(telluric_correction_file = self.telluric_correction)

                
        if self.flux_calibration is not None:
            if str(type(self.flux_calibration))[-5:-2] == "str":  
                if path is not None: self.flux_calibration = os.path.join(path,self.flux_calibration)
                if verbose: print(" - Reading flux calibration from file",self.flux_calibration)
                self.flux_calibration =  FluxCalibration (path_to_response = self.flux_calibration)
                
        #TODO: check that all are from the same day / grating and have the same wavelength
        
    def save_night_calibration(self, **kwargs):   #TODO
        verbose = kwargs.get('verbose', False)
        warnings = kwargs.get('warnings', verbose)
        
   

def create_night_calibration(data_container = None,
                             date = None,
                             grating = None,
                             wavelength = None,
                             throughput = None,
                             wavelength_shift_correction = None,
                             telluric_correction = None,
                             flux_calibration = None,
                             star_list = None, 
                             **kwargs):
    
    verbose = kwargs.get('verbose', False)
    warnings = kwargs.get('warnings', verbose)

    if verbose: print("\n> Creating object with the calibration of the night...")       
    
    if data_container is not None:
        if wavelength is None: wavelength = data_container.wavelength
        if date is None: date = data_container.koala.header["UTDATE"]
        if grating is None: grating = data_container.koala.info["aaomega_grating"]
    
    if warnings:
        if wavelength is None: print("  WARNING! No wavelength has been provided !!!")
        if date is None: print("  WARNING! No data has been provided !!!")
        if grating is None: print("  WARNING! No grating has been provided !!!")
    
    if verbose:
        if date is not None and grating is not None: print("  Date:  {}   , AAOmega grating : {}".format(date,grating))
        if wavelength is not None: print("   - Wavelength: length = {}    w[0] = {}      w[-1] = {}".format(len(wavelength), wavelength[0], wavelength[-1]))
        if throughput is not None: print("   - Saving throughput ...")
        if wavelength_shift_correction is not None: print("   - Saving wavelength_shift_correction ...")        
        if grating in red_gratings and warnings:
            if telluric_correction is None:  print("   - WARNING! No telluric_correction has been provided for this RED grating !!!")
        if telluric_correction is not None: print("   - Saving telluric_correction ...")
        if flux_calibration is not None: print("   - Saving flux_calibration ...")        
        if star_list is not None: print("   - Saving star objects provided in star_list ...")
        
    night_cal = NIGHT_CALIBRATION(data_container = data_container,
                                  date = date,
                                  grating = grating,
                                  wavelength = wavelength,
                                  throughput = throughput,
                                  wavelength_shift_correction = wavelength_shift_correction,
                                  telluric_correction = telluric_correction,
                                  flux_calibration = flux_calibration,
                                  star_list = star_list )
    
    return night_cal
