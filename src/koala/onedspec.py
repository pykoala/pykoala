#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from synphot import observation   ########from pysynphot import observation
#from synphot import spectrum      ########from pysynphot import spectrum
from synphot import SourceSpectrum, SpectralElement
from synphot.models import Empirical1D
from scipy.signal import medfilt



#import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import numpy as np
import sys
import os

from scipy import interpolate, signal
from scipy.optimize import curve_fit
import scipy.signal as sig

#import datetime
import copy

#import glob
#from astropy.io import fits as pyfits 

#import constants

# Disable some annoying warnings
import warnings

from koala.constants import C
from koala.io import read_table
from koala.plot_plot import plot_plot, basic_statistics

warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.CRITICAL)


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


#version = 'NOT IMPLEMENTED'
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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def substract_given_gaussian(wavelength, spectrum, centre, peak=0, sigma=0,  flux=0, search_peak=False, allow_absorptions = False,
                             lowlow= 20, lowhigh=10, highlow=10, highhigh = 20, 
                             lmin=0, lmax=0, fmin=0, fmax=0, plot=True, fcal=False, verbose = True, warnings=True):
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
            if verbose or warnings: 
                print("      WARNING! Impossible to get the continuum!")
                print("               Scaling the continuum to the median value") 
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
                if verbose or warnings: print("    Error trying to get the peak as requested wavelength is ",np.round(centre,2),"! Ignoring this fit!")
                peak = 0.
                flux = -0.0001
    
        no_substract = False
        if flux < 0:
            if allow_absorptions == False:
                if np.isnan(centre) == False:
                    if verbose or warnings : print("    WARNING! This is an ABSORPTION Gaussian! As requested, this Gaussian is NOT substracted!")
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
        if verbose or warnings: print("    There is no valid data in the wavelength range [{},{}] !!".format(lmin,lmax))
        
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
                    print("    WARNING: Impossible to get the continuum!")
                    print("             Scaling the continuum to the median value b = ",bb,":  cont =  0 * w_spec  + ", bb)
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
                print("  WARNING: Impossible to get the continuum!")
                print("           Scaling the continuum to the median value")          
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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fit_clip(x, y, clip=0.4, index_fit = 2, kernel = 19, mask ="",                          
             xmin="",xmax="",ymin="",ymax="",percentile_min=2, percentile_max=98,
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
     
    # Preparing the data. Trim edges and remove nans
    
    
    
    
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
                ptitle = "Polyfit of degree "+np.str(index_fit)+" using clip = "+np.str(clip)+" * std"
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
def trim_spectrum(w,s, edgelow=None,edgehigh=None, mask=None, auto_trim = True, 
                  exclude_wlm=[[0,0]], verbose=True, plot=True):
    
    if auto_trim:
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
    else:
    # If mask is given, use values of mask instead of edgelow edghigh
        if mask is not None:   # Mask is given as [edgelow,edgehigh]
            edgelow = mask[0]
            edgehigh = len(w)-mask[1]+1
            if verbose: print("  Trimming the edges using the mask: [0:{}] and [{}:{}] ...".format(edgelow,len(w)-edgehigh, len(w)))  
        else:
            if verbose: print("  Trimming the edges [0:{}] and [{}:{}] ...".format(edgelow,len(w)-edgehigh, len(w)))  
        
    vlines=[w[edgelow], w[len(w)-edgehigh]]
    index=np.arange(len(w))
    valid_ind=np.where((index >= edgelow) & (index <= len(w)-edgehigh) & (~np.isnan(s)))[0]
    valid_w = w[valid_ind]
    valid_s = s[valid_ind] 
    
    if exclude_wlm[0][0] != 0:    
        for rango in exclude_wlm :
            if verbose: print("  Trimming wavelength range [", rango[0],",", rango[1],"] ...")
            index=np.arange(len(valid_w))
            #not_valid_ind = np.where((valid_w[index] >= rango[0]) & (valid_w[index] <= rango[1]))[0]
            valid_ind = np.where((valid_w[index] <= rango[0]) | (valid_w[index] >= rango[1]))[0]  # | is OR
            valid_w = valid_w[valid_ind]
            valid_s = valid_s[valid_ind]
            vlines.append(rango[0])
            vlines.append(rango[1])
        
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
def fit_smooth_spectrum(w,s, edgelow=20,edgehigh=20, mask=None, auto_trim = False,   # BOBA
                        kernel_correct_defects = 51, exclude_wlm=[[0,0]],  #remove_nans = True,
                        kernel_fit=11, index_fit= 9,  clip_fit = 1.0, sigma_factor = 2.5,
                        maxit=10, verbose=True, plot_all_fits = False,
                        plot=True, hlines=[1.], ptitle= "", fcal=False):
    """
    Apply f1,f2 = fit_smooth_spectrum(wl,spectrum) and returns:
    
    f1 is the smoothed spectrum, with edges 'fixed'
    f2 is the fit to the smooth spectrum
    Tasks that use this: correcting_negative_sky, plot_corrected_vs_uncorrected_spectrum
    """

    if verbose: 
        print('\n> Fitting an order {} polynomium to a spectrum smoothed with medfilt window of {}'.format(index_fit,kernel_fit))
    
    if mask is not None:   # Mask is given as [edgelow,edgehigh]
        edgelow = mask[0]
        edgehigh = len(w)-mask[1]+1
    
    # Trimming edges in spectrum
    verbose_this =False
    if verbose and exclude_wlm[0][0] == 0 : verbose_this = True
    valid_w, valid_s, valid_ind= trim_spectrum(w,s, edgelow=edgelow,edgehigh=edgehigh, mask=mask, auto_trim = auto_trim, 
                                               verbose=verbose_this, plot=False)
    valid_s_smooth_all = signal.medfilt(valid_s, kernel_fit)
    valid_ind_all = valid_ind
    
    # If exclude_wlm included, run it again
    valid_w, valid_s, valid_ind= trim_spectrum(w,s, edgelow=edgelow,edgehigh=edgehigh, mask=mask, auto_trim = auto_trim, 
                  exclude_wlm=exclude_wlm, verbose=verbose, plot=False)
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

    # Smoothed spectrum only between edgelow and edgehigh        
    f = np.zeros_like(s)
    f[valid_ind_all] = valid_s_smooth_all

    if plot or plot_all_fits:
        ymin = np.nanpercentile(s[edgelow:len(s)-edgehigh],0.5)
        ymax=  np.nanpercentile(s[edgelow:len(s)-edgehigh],99.5)
        rango = (ymax-ymin)
        ymin = ymin - rango/10.
        ymax = ymax + rango/10.

    alpha=[0.1,0.3]
    if plot_all_fits and maxit > 1:
        fits_to_plot=[s,f]
        for item in list_of_fits:
            fits_to_plot.append(item)
            alpha.append(0.8)
        plot_plot(w,fits_to_plot, ptitle="All fits to the smoothed spectrum", 
                  vlines=[w[edgelow],w[-1-edgehigh]], hlines=hlines, axvspan=exclude_wlm,
                  fcal=fcal,alpha=alpha, ymin=ymin, ymax=ymax)
               
    if plot:                   
        if ptitle == "" : ptitle= "Order "+np.str(index_fit)+" polynomium fitted to a spectrum smoothed with a "+np.str(kernel_fit)+" kernel window"
        plot_plot(w, [s,f,fx], ymin=ymin, ymax=ymax, color=["red","green","blue"], alpha=[0.2,0.5,0.5], 
                  label=["spectrum","smoothed","fit"], ptitle=ptitle, fcal=fcal, 
                  vlines=[w[edgelow],w[-1-edgehigh]], hlines=hlines, axvspan=exclude_wlm)
      
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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
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
        if line_wavelength != 0 : ptitle="Cosmic identification in cut at "+np.str(line_wavelength)+" $\mathrm{\AA}$"        
        plot_plot(x,verde, ymin=0,ymax=200, hlines=[cosmic_higher_than], ptitle=ptitle,  ylabel="abs (cut - medfilt(cut)) - extra_factor * max_val")
 
    if verbose:
        if line_wavelength == 0:
            print("\n> Identified", len(cosmics_list),"cosmics in fibres",cosmics_list)
        else:
            print("\n> Identified", len(cosmics_list),"cosmics at",np.str(line_wavelength),"A in fibres",cosmics_list)
    return cosmics_list
#---------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# def basic_statistics(y, x="", xmin="", xmax="", return_data=False, verbose = True):
#     """
#     Provides basic statistics: min, median, max, std, rms, and snr"
#     """    
#     if x == "":
#         y_ = y
#     else:          
#         y_ = []      
#         if xmin == "" : xmin = x[0]
#         if xmax == "" : xmax = x[-1]          
#         y_.extend((y[j]) for j in range(len(x)) if (x[j] > xmin and x[j] < xmax) )  
     
#     median_value = np.nanmedian(y_)
#     min_value = np.nanmin(y_)
#     max_value = np.nanmax(y_)
    
#     n_ = len(y_)
#     #mean_ = np.sum(y_) / n_
#     mean_ = np.nanmean(y_)
#     #var_ = np.sum((item - mean_)**2 for item in y_) / (n_ - 1)  
#     var_ = np.nanvar(y_)

#     std = np.sqrt(var_)
#     ave_ = np.nanmean(y_)
#     disp_ =  max_value - min_value
    
#     rms_v = ((y_ - mean_) / disp_ ) **2
#     rms = disp_ * np.sqrt(np.nansum(rms_v)/ (n_-1))
#     snr = ave_ / rms
    
#     if verbose:
#         print("  min_value  = {}, median value = {}, max_value = {}".format(min_value,median_value,max_value))
#         print("  standard deviation = {}, rms = {}, snr = {}".format(std, rms, snr))
    
#     if return_data : return min_value,median_value,max_value,std, rms, snr
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
                ptitle = "Polyfit of degree "+np.str(index_fit)+" using clip = "+np.str(clip)+" * std"
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
    
    