"""
This script contains the wrapper functions to build a PyKoala RSS object from KOALA (2dfdr-reduced) data.
"""

# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import os
import sys
import copy
import glob
import datetime
from random import uniform
from scipy.signal import medfilt
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits
from astropy.wcs import WCS
#from astropy.utils.exceptions import AstropyWarning
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala.ancillary import vprint, rss_info_template  # Template to create the info variable 
from pykoala.ancillary import interpolate_image_nonfinite
from pykoala.rss import RSS
from pykoala.plotting.rss_plot import rss_image, rss_map, compare_rss_images, plot_combined_spectrum
from pykoala.plotting.plot_plot import plot_plot #basic_statistics
from pykoala.rss import rss_valid_wave_range
from pykoala.corrections.sky import SkyModel, TelluricCorrection, SkySubsCorrection, SkyFromObject
from pykoala.corrections.atmospheric_corrections import AtmosphericExtCorrection
from pykoala.corrections.throughput import ThroughputCorrection, Throughput
from pykoala.corrections.wavelength_corrections import WavelengthShiftCorrection
from pykoala.spectra.onedspec import fit_smooth_spectrum,fit10gaussians, gauss, find_emission_lines,fluxes,clip_spectrum_using_continuum,find_cosmics_in_cut #, read_table
from pykoala import __version__ as version
# =============================================================================
# Ignore warnings
# =============================================================================
import warnings
warnings.filterwarnings('ignore')
from astropy.utils.exceptions import AstropyWarning
# =============================================================================
# KOALA specifics
# =============================================================================
red_gratings = ["385R","1000R","2000R", "1000I", "1700D","1700I"]
blue_gratings = ["580V" , "1500V" ,"1700B" , "3200B" , "2500V"]   

koala_info_rss_dict= dict(rss_object_name = None,
                          path_to_file = None,
                          n_wave = None,
                          n_spectra = None,
                          spaxel_size = None,
                          aaomega_arm = None,
                          aaomega_grating = None,
                          aaomega_dichroic = None, 
                          KOALA_fov = None,
                          position_angle = None,
                          RA_centre_deg = None,
                          DEC_centre_deg = None,
                          exptime = None,
                          description = None,
                          valid_wave_min = None,
                          valid_wave_max = None,
                          valid_wave_min_index = None,
                          valid_wave_max_index = None,
                          brightest_line = None,
                          brightest_line_wavelength = None,
                          history = None,
                          )

koala_info_cube_dict= dict(cube_object_name = None,
                           path_to_file = None,
                           path_to_rss_file = None,
                           #n_wave = None,
                           #n_spectra = None,
                           #spaxel_size = None,
                           #aaomega_arm = None,
                           aaomega_grating = None,
                           aaomega_dichroic = None,
                           #KOALA_fov = None,
                           position_angle = None,
                           description = None,
                           valid_wave_min = None,
                           valid_wave_max = None,
                           valid_wave_min_index = None,
                           valid_wave_max_index = None,
                           valid_RA_spaxel_range = None,
                           valid_DEC_spaxel_range = None,
                           #brightest_line = None,
                           #brightest_line_wavelength = None,
                           history = None,
                           )

class KOALA_INFO(object):
    """
    Class KOALA_INFO. Add object rss.koala or cube.koala to store KOALA-specific information: 
    - rss.koala.info : a dictionary with info, see variables in koala_info_rss_dict
    - rss.header: header of the rss
    - rss.wcs: WCS of the rss
    - rss.fibre_table: fibre table of the rss
    - rss.integrated_fibre: flux integrated in each fibre between  valid_wave_min and valid_wave_max (in koala.info)
    - rss.integrated_fibre_sorted: list of fibres sorted from lowest integrated value [0] to highest [-1]
    - rss.negative_fibres: list of fibres for which integrated fibre is NEGATIVE
    - rss.mask: KOALA mask for EDGES
    - rss.list_fibres_all_good_values: list of fibres for which all wavelengths are valid (central part of rss)
    - rss.continuum_model_after_sky_correction: as it says
    - rss.emission_lines_gauss_spectrum: after identifying emission lines, a spectrum with the Gaussian fits
    - rss.redshifted_emission_lines_detected_dictionary: as it says
    """
    def __init__(self, **kwargs):
        self.corrections_done = kwargs.get('corrections_done', None) 
        self.history = kwargs.get('history', None) 
        data_container = kwargs.get('data_container', 'rss')
        if data_container == "cube":
            self.info = kwargs.get('info', koala_info_cube_dict.copy())
            self.integrated_map = kwargs.get('integrated_map', None)
            
            self.x_max = kwargs.get('x_max', None)
            self.y_max = kwargs.get('y_max', None)
            self.x_peaks = kwargs.get('x_peaks', None)
            self.y_peaks = kwargs.get('y_peaks', None)
            self.x_peak_median = kwargs.get('x_peak_median', None)
            self.y_peak_median = kwargs.get('y_peak_median', None)
            self.offset_from_center_x_arcsec_tracing = kwargs.get('offset_from_center_x_arcsec_tracing', None)
            self.offset_from_center_y_arcsec_tracing = kwargs.get('offset_from_center_x_arcsec_tracing', None)
            self.ADR =  kwargs.get('ADR', None)
                        
        else:            
           self.info = kwargs.get('info', koala_info_rss_dict.copy())
           self.header =  kwargs.get('header', None)
           self.wcs = kwargs.get('wcs', None)
           self.fibre_table =  kwargs.get('fibre_table', None)

           self.integrated_fibre = kwargs.get('integrated_fibre', None)
           self.integrated_fibre_sorted = kwargs.get('integrated_fibre_sorted', None)
           self.integrated_fibre_variance = kwargs.get('integrated_fibre_variance', None)
           self.negative_fibres = kwargs.get('negative_fibres', None)
           self.mask = kwargs.get('mask', None)
           self.list_fibres_all_good_values = kwargs.get('list_fibres_all_good_values', None)
           self.continuum_model_after_sky_correction= kwargs.get('continuum_model_after_sky_correction', None)
       
           self.emission_lines_gauss_spectrum = kwargs.get('emission_lines_gauss_spectrum', None)
           self.redshifted_emission_lines_detected_dictionary =  kwargs.get('redshifted_emission_lines_detected_dictionary', None)
       #self.brightest_line = kwargs.get('brightest_line', None)
       #self.brightest_line_wavelength = kwargs.get('brightest_line_wavelength', None)

# =============================================================================
# %% ==========================================================================
# =============================================================================
def py_koala_header(header):
    """
    Copy 2dfdr headers values from extensions 0 and 2 needed for the initial
    header for PyKoala. (based in the header constructed in  save_rss_fits in
    koala.io)
    """
    # To fit actual PyKoala header format
    try:                                             #!!! Angel to read a saved rss
        header.rename_keyword('CENRA', 'RACEN')
        header.rename_keyword('CENDEC', 'DECCEN')
    except Exception:
        pass

    cards = [header.cards['BITPIX'],
             header.cards["ORIGIN"],
             header.cards["TELESCOP"],
             header.cards["ALT_OBS"],
             header.cards["LAT_OBS"],
             header.cards["LONG_OBS"],
             header.cards["INSTRUME"],
             header.cards["GRATID"],
             header.cards["SPECTID"],
             header.cards["DICHROIC"],
             header.cards['OBJECT'],
             header.cards["EXPOSED"],
             header.cards["ZDSTART"],
             header.cards["ZDEND"],
             header.cards['NAXIS'],
             header.cards['NAXIS1'],
             header.cards['NAXIS2'],
             header.cards['RACEN'],
             header.cards['DECCEN'],
             header.cards['TEL_PA'],
             header.cards["CTYPE2"],
             header.cards["CUNIT2"],
             header.cards["CTYPE1"],
             header.cards["CUNIT1"],
             header.cards["CRVAL1"],
             header.cards["CDELT1"],
             header.cards["CRPIX1"],
             header.cards["CRVAL2"],
             header.cards["CDELT2"],
             header.cards["CRPIX2"],
             ]
    py_koala_header = fits.header.Header(cards=cards, copy=False)
    py_koala_header = header
    return py_koala_header
# =============================================================================
# %% ==========================================================================
# =============================================================================
def py_koala_fibre_table(fibre_table):
    """
    Generates the spaxels tables needed for PyKoala from the 2dfdr spaxels table.
    """
    # Filtering only selected (in use) fibres
    file_from_pykoala = False
    try:
        spaxels_table = fibre_table[fibre_table['SELECTED'] == 1]
    except Exception:
        spaxels_table = fibre_table
        file_from_pykoala = True

    # Defining new arrays
    arr1 = np.arange(len(spaxels_table)) + 1  # +  for starting in 1
    arr2 = np.ones(len(spaxels_table))
    arr3 = np.ones(len(spaxels_table))
    arr4 = np.ones(len(spaxels_table)) * 2048
    arr5 = np.zeros(len(spaxels_table))
    if file_from_pykoala:
        arr6 = spaxels_table['Delta_RA']
        arr7 = spaxels_table['Delta_Dec']
        arr8 = spaxels_table['Fibre_OLD']
    else:
        arr6 = spaxels_table['XPOS']
        arr7 = spaxels_table['YPOS']
        arr8 = spaxels_table['SPEC_ID']

    # Defining new columns
    col1 = fits.Column(name='Fibre', format='I', array=arr1)
    col2 = fits.Column(name='Status', format='I', array=arr2)
    col3 = fits.Column(name='Ones', format='I', array=arr3)
    col4 = fits.Column(name='Wavelengths', format='I', array=arr4)
    col5 = fits.Column(name='Zeros', format='I', array=arr5)
    col6 = fits.Column(name='Delta_RA', format='D', array=arr6)
    col7 = fits.Column(name='Delta_Dec', format='D', array=arr7)
    col8 = fits.Column(name='Fibre_OLD', format='I', array=arr8)

    # PyKoala Spaxels table
    py_koala_spaxels_table = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5, col6, col7, col8])

    return py_koala_spaxels_table
# %% ============================================================================
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
def koalaRSS(filename, rss_object_name = None, path=None, plot_map= False, **kwargs):
    """
    Tasks for creating a KOALA rss object including rss.koala.
    
    This uses koala_rss wrapper, which reads rss using read_rss

    Parameters
    ----------
    filename : string
        Name of the rss file.
    rss_object_name : string, optional
        name of the rss object
    path : string, optional
        path to the rss file
    plot_map : boolean, optional
        if True it will plot the rss map. The default is False.
    **kwargs : kwargs
        where we can find verbose, warnings...

    Returns
    -------
    rss : object
        rss KOALA object

    """
    if path is not None:
        path_to_file = os.path.join(path,filename)
    else:
        path_to_file = filename  
    
    vprint('\n> Converting KOALA+AAOmega RSS file "'+path_to_file+'" to a koala RSS object...', **kwargs)
    
    # Read rss using standard pykoala      #TODO: Ángel: I think we should merge the 2 of them (koalaRSS and koala_rss)
    rss = koala_rss(path_to_file)

    # Create rss.koala
    rss.koala = KOALA_INFO()

    # Now, we add to this rss the information for a koala file
    # Create the dictionary containing relevant information
    koala_info = koala_info_rss_dict.copy()  # Avoid missing some key
    
    koala_info["rss_object_name"] = rss_object_name    # Save the name of the rss object
    koala_info['path_to_file'] = path_to_file          # Save name of the .fits file
    koala_info['n_wave'] = len(rss.wavelength)     
    koala_info['n_spectra'] = len(rss.intensity)   

    # Check that dimensions match KOALA numbers
    if koala_info['n_wave'] != 2048 and koala_info['n_spectra'] != 986: #  1000:
        if kwargs.get("warnings") or kwargs.get("verbose"):
            print('\n  *** WARNING *** : These numbers are NOT the standard n_wave and n_spectra values for KOALA')
    
    # As now the header is not saved in koala_rss, let's do it again to save here key KOALA info
    header = fits.getheader(path_to_file, 0) + fits.getheader(path_to_file, 2)
    koala_header = py_koala_header(header)
    
    # Add history
    rss.koala.history = koala_header["HISTORY"]
    
    koala_info['RA_centre_deg'] = koala_header["RACEN"] *180/np.pi
    koala_info['DEC_centre_deg'] = koala_header["DECCEN"] *180/np.pi
    koala_info['exptime'] = koala_header["EXPOSED"]
    
    # Get AAOmega Arm & gratings
    if (koala_header['SPECTID'] == "RD"):      
        koala_info['aaomega_arm'] = "red"
    if (koala_header['SPECTID'] == "BL"):      
        koala_info['aaomega_arm'] = "blue"    
    koala_info['aaomega_grating'] = koala_header['GRATID']
    koala_info['aaomega_dichroic'] = koala_header["DICHROIC"]
    
    # Constructing Pykoala Spaxels table from 2dfdr spaxels table (data[2]) AGAIN to know WIDE/NARROW & spaxel_size
    fibre_table = fits.getdata(path_to_file, 2)
    koala_fibre_table = py_koala_fibre_table(fibre_table)
    
    fib_ra_offset = koala_fibre_table.data['Delta_RA'] 
    fib_dec_offset = koala_fibre_table.data['Delta_DEC'] 
    # Check if NARROW (spaxel_size = 0.7 arcsec) or WIDE (spaxel_size=1.25) field of view
    # (if offset_max - offset_min > 31 arcsec in both directions)
    if (np.max(fib_ra_offset) - np.min(fib_ra_offset)) > 31 or (np.max(fib_dec_offset) - np.min(fib_dec_offset)) > 31:
        spaxel_size = 1.25
        KOALA_fov = 'WIDE: 50.6" x 27.4"'
    else:
        spaxel_size = 0.7
        KOALA_fov = 'NARROW: 28.3" x 15.3"'   
    koala_info['KOALA_fov']=KOALA_fov
    koala_info['spaxel_size']=spaxel_size 
    koala_info['position_angle'] = koala_header['TEL_PA']
    
    # Check valid range (basic mask in AnPyKOALA)
    valid_wave_range_data = rss_valid_wave_range (rss)    
    koala_info['valid_wave_min'] = valid_wave_range_data[2][0]
    koala_info['valid_wave_max'] = valid_wave_range_data[2][1]
    koala_info['valid_wave_min_index'] = valid_wave_range_data[1][0]
    koala_info['valid_wave_max_index'] = valid_wave_range_data[1][1]   
    # koala_info['first_good_wave_per_fibre'] = np.array(valid_wave_range_data[0][0])
    # koala_info['last_good_wave_per_fibre'] = np.array(valid_wave_range_data[0][1])
    # koala_info['list_fibres_all_good_values'] = np.array(valid_wave_range_data[3])
    rss.koala.mask = [np.array(valid_wave_range_data[0][0]), np.array(valid_wave_range_data[0][1]) ]
    rss.koala.list_fibres_all_good_values = np.array(valid_wave_range_data[3])
        
    # Saving the info to the rss object
    rss.koala.info = koala_info
    rss.koala.header = koala_header    # Saving header
    rss.koala.fibre_table = koala_fibre_table
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        rss.koala.wcs = WCS(header)   # Saving WCS, not sure if later needed, but for now it is there
    
    # Plot RSS image if requested  #!!!
    if kwargs.get('plot'): rss_image(rss, **kwargs)
    
    # Computing integrating time and plot if requested:
    kwargs["plot"] = plot_map
    verbose = kwargs.get('verbose', False)
    kwargs["verbose"] = False
    rss=compute_integrated_fibre(rss, **kwargs)
    kwargs["verbose"] = verbose
     
    #get mask
    rss.mask =get_rss_mask(rss, make_zeros=kwargs.get('make_zeros', False))
    
    # Printing the info is requested:
    vprint('  Found {} spectra with {} wavelengths'.format(koala_info['n_spectra'], koala_info['n_wave']),
                      'between {:.2f} and {:.2f} Angstroms.'.format(rss.wavelength[0], rss.wavelength[-1]), **kwargs)
    vprint('  This RSS file uses the',koala_info['aaomega_grating'],'grating in the',koala_info['aaomega_arm'],'AAOmega arm.', **kwargs)
    vprint('  The KOALA field of view is {}, with a spaxel size of {}" and PA = {:.1f}º.'.format(KOALA_fov, spaxel_size,koala_info['position_angle']), **kwargs)
    
    if koala_info['rss_object_name'] is not None: 
        vprint('  Name of the observation = "{}",   Name of this Python RSS object = "{}".'.format(rss.info['name'],koala_info['rss_object_name']), **kwargs)
    else:
        vprint('  Name of the observation = "{}".'.format(rss.info['name']), **kwargs)    
        
    if kwargs.get("description") is not None:
        rss.koala_info["description"] = kwargs.get("description")
        vprint('  Description provided to this KOALA RSS object = "{}".'.format(koala_info['description']), **kwargs)
            
    return rss
# =============================================================================
# %% ==========================================================================
# =============================================================================
def koala_rss(path_to_file, **kwargs):
    """
    A wrapper function that converts a file (not an RSS object) to a koala RSS object
    The paramaters used to build the RSS object e.g. bad spaxels, header etc all come from the original (non PyKoala) .fits file
    """
    vprint('\n> Converting RSS file',path_to_file,'to a koala RSS object...', **kwargs) #!!!
    header = fits.getheader(path_to_file, 0) + fits.getheader(path_to_file, 2)
    koala_header = py_koala_header(header)
    # WCS
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        koala_wcs = WCS(header)   #!!!  Ángel get annoying warning:
    # WARNING: FITSFixedWarning: RADECSYS= 'FK5 ' / FK5 reference system 
    # the RADECSYS keyword is deprecated, use RADESYSa. [astropy.wcs.wcs]
    # Still not working... #!!! 
    
    # Constructing Pykoala Spaxels table from 2dfdr spaxels table (data[2])
    fibre_table = fits.getdata(path_to_file, 2)
    koala_fibre_table = py_koala_fibre_table(fibre_table)

    # List of bad spaxels from 2dfdr spaxels table
    try:
        bad_fibres_list = (fibre_table['SPEC_ID'][fibre_table['SELECTED'] == 0] - 1).tolist()
    except Exception:
        bad_fibres_list = None
    # -1 to start in 0 rather than in 1
    # Create the dictionary containing relevant information
    info = rss_info_template.copy()  # Avoid missing some key
    info['name'] = koala_header['OBJECT']
    info['exptime'] = koala_header['EXPOSED']
    info['fib_ra'] = np.rad2deg(koala_header['RACEN']) + koala_fibre_table.data['Delta_RA'] / 3600
    info['fib_dec'] = np.rad2deg(koala_header['DECCEN']) + koala_fibre_table.data['Delta_DEC'] / 3600
    info['airmass'] = airmass_from_header(koala_header)
    # Read RSS file into a PyKoala RSS object
    rss = read_rss(path_to_file, 
                   wcs=koala_wcs,                         
                   bad_fibres_list=bad_fibres_list,      
                   intensity_axis=0,
                   variance_axis=1,
                   header=koala_header,                  
                   fibre_table=koala_fibre_table,
                   info=info
                   )
    # Plot RSS image if requested
    if kwargs.get('plot'): rss_image(rss, **kwargs)
    
    return rss
# =============================================================================
# %% ==========================================================================
# =============================================================================
def clean_nan(rss, **kwargs):
    """
    Clean nans in rss using interpolation (this is needed to avoid 1D tasks failing )
    """
    rss_out = copy.deepcopy(rss)
    vprint("> Applying nearest neighbour interpolation to remove NaN values ...", **kwargs)
    rss_out.intensity = interpolate_image_nonfinite(rss.intensity)
    rss_out.variance = interpolate_image_nonfinite(rss.variance)    
    return rss_out
# =============================================================================
# %% ==========================================================================
# =============================================================================
def get_throughput_2D(file_skyflat= None,
                      path=None, #instrument=None,
                      rss = None,
                      rss_object_name = None,
                      mask = None,
                      correct_ccd_defects=True, 
                      kernel_throughput=None,
                      index_fit_throughput = None,
                      throughput_2D_file=None,  
                      also_return_skyflat=True,
                      plot_final_rss = False,
                      **kwargs ):   
                      # fix_wavelengths=False, sol=None, 
    """
    Adaptation of Angel's get_throughput_2D for 2024 PyKOALA.
    Get a 2D array with the throughput 2D using a COMBINED skyflat / domeflat.
    It is important that this is a COMBINED file, as we should not have any cosmics/nans left.
    A COMBINED flappy flat could be also used if skyflat/domeflats were not available, but 
    these are NOT as good as the dome / sky flats and should be avoided.

    Parameters
    ----------
    file_skyflat: string
        The fits file containing the skyflat/domeflat
    path: string
        Path to file
    rss: object with the skyflat/domeflat 
    rss_object_name : string, optional
        name of the rss object
    mask: list of arrays
        if a mask is given, apply the mask. If not using KOALA mask in skyflat.koala.mask    
    correct_ccd_deffects: boolean
        If True, it corrects for ccd defects when reading the skyflat fits file
    kernel_throughput: odd integer 
        If not 0, the 2D throughput will be smoothed with a this kernel
    index_fit_throughput: integer
        index of the polynomium to fit each spectrum. If none it uses index_fit_throughput = 11
    throughput_2D_file: string
        the name of the fits file to be created with the output throughput 2D
    plot_final_rss: boolean
        If True plot the final rss
    also_return_skyflat
        If True it also returns the skyflats
    
    **kwargs : kwargs
        where we can find verbose, warnings, plot...

    Returns
    -------
    if also_return_skyflat: 
        throughput, skyflat
    else:
        throughput
    """
    
    verbose = kwargs.get('verbose', False)
    #warnings = kwargs.get('warnings', verbose)
    plot =  kwargs.get('plot', False)
    if plot_final_rss is False and plot is True: plot_final_rss = True

    if verbose: print("\n> Reading a COMBINED skyflat / domeflat to get the 2D throughput...")

    # if sol[0] == 0 or sol[0] == -1:
    #     if fix_wavelengths:
    #         print("  Fix wavelength requested but not solution given, ignoring it...")
    #         fix_wavelengths = False
    # # else:
    # #    if len(sol) == 3 : fix_wavelengths = True
    
    if rss_object_name is None: rss_object_name = "skyflat"
    
    skyflat =  process_koala_rss(rss_object_name=rss_object_name,
                                 rss=rss,
                                 filename=file_skyflat, path = path,
                                 correct_ccd_defects = correct_ccd_defects, 
                                 apply_mask=True,
                                 plot_final_rss=plot_final_rss,   **kwargs )
                                 #plot_final_rss=True,warnings=False,plot = True, verbose = True)

    #skyflat = apply_mask_to_rss(skyflat, verbose=verbose)  # Already in skyflat
    
    throughput_2D_ = np.zeros_like(skyflat.intensity)
    #throughput_2D_variance_ = np.zeros_like(skyflat.variance)

    if verbose: print("\n> Getting the throughput per wavelength...")
    n_wave=len(skyflat.wavelength)
    n_spectra = len(skyflat.intensity)
    for i in range(n_wave):
        column = skyflat.intensity[:, i]
        mcolumn = column / np.nanmedian(column)
        throughput_2D_[:, i] = mcolumn

        # column = skyflat.variance[:, i]
        # mcolumn = column / np.nanmedian(column)
        # throughput_2D_variance_[:, i] = mcolumn

    if kernel_throughput is not None:
        
        if plot: rss_image(skyflat, image=throughput_2D_, cmap="binary_r", title=" 2D throughput BEFORE SMOOTHING AND FITTING A POLYNOMIUM")
        
        if mask is None: mask = skyflat.koala.mask
        if index_fit_throughput is None: index_fit_throughput = 11
        if verbose: 
            print("\n  - Applying smooth with kernel =", kernel_throughput," and using to fit a polynomium of degree =",index_fit_throughput,"..." )
            sys.stdout.write("    Working on fibre:                           ")
            sys.stdout.flush()
    
        throughput_2D = np.zeros_like(throughput_2D_)
        # throughput_2D_variance = np.zeros_like(throughput_2D_variance_)
        for fibre in range(n_spectra):
            if verbose:
                sys.stdout.write("\b" * 25)
                sys.stdout.write("{:5.0f}  ({:5.2f}% completed)".format(fibre, fibre/n_spectra*100))
                sys.stdout.flush()
            #throughput_2D[i] = medfilt(throughput_2D_[i], kernel_throughput)
            _,throughput_2D[fibre] = fit_smooth_spectrum(skyflat.wavelength,
                                                         throughput_2D_[fibre], 
                                                         mask =[mask[0][fibre], mask[1][fibre]],
                                                         kernel_fit=kernel_throughput, 
                                                         index_fit=index_fit_throughput,
                                                         verbose=False, plot=False)     
            # _,throughput_2D_variance[fibre] = fit_smooth_spectrum(skyflat.wavelength,
            #                                              throughput_2D_variance_[fibre], 
            #                                              mask =[mask[0][fibre], mask[1][fibre]],
            #                                              kernel_fit=kernel_throughput, 
            #                                              index_fit=index_fit_throughput,
            #                                              verbose=False, plot=False)  

        if verbose:
            sys.stdout.write("\b" * 51)
            sys.stdout.write("  Process completed!                                     ")
            sys.stdout.flush()
            print(" ")
        throughput_2D = throughput_2D * skyflat.mask 
        # throughput_2D_variance = throughput_2D_variance * skyflat.mask 
        #if plot: rss_image(skyflat, image=throughput_2D, chigh=1.1, clow=0.9, cmap="binary_r",  title=" 2D throughput AFTER SMOOTHING AND FITTING A POLYNOMIUM")
        #skyflat.history.append('- Throughput 2D smoothed with kernel ' + str(kernel_throughput))

    else:
        throughput_2D = throughput_2D_
        # throughput_2D_variance = throughput_2D_variance_

    #skyflat.sol = sol
    # # Saving the information of fix_wavelengths in throughput_2D[0][0]
    # if sol[0] != 0:
    #     print("\n  - The solution for fixing wavelengths has been provided")
    #     if sol[0] != -1:
    #         throughput_2D[0][
    #             0] = 1.0  # if throughput_2D[0][0] is 1.0, the throughput has been corrected for small wavelength variations
    #         skyflat.history.append('- Written data[0][0] = 1.0 for automatically identifing')   #!!! HISTORY TODO
    #         skyflat.history.append('  that the throughput 2D data has been obtained')
    #         skyflat.history.append('  AFTER correcting for small wavelength variations')

    if plot:
        x = np.arange(n_spectra)
        median_throughput = np.nanmedian(throughput_2D, axis=1)
        plot_plot(x, median_throughput, ymin=0.2, ymax=1.2, hlines=[1, 0.9, 1.1],
                  ptitle="Median value of the 2D throughput per fibre", xlabel="Fibre")
        rss_image(skyflat, image=throughput_2D, cmap="binary_r",
                          title="\n ---- 2D throughput ----")

    skyflat_corrected = skyflat.intensity / throughput_2D
    if plot: rss_image(skyflat, image=skyflat_corrected, title="\n Skyflat CORRECTED for 2D throughput")
    
    # if throughput_2D_file is not None:  ##!!! CHECK SAVE throughput
    #     save_rss_fits(skyflat, data=throughput_2D, fits_file=throughput_2D_file, text="Throughput 2D ", sol=sol)

    # Ponerlo como lo quiere Pablo
    throughput = Throughput()
    throughput.throughput_data = throughput_2D
    #throughput.throughput.throughput_error = throughput_2D_variance
    throughput.throughput_error = throughput_2D**0.5
    
    # Save fits file if requested
    if throughput_2D_file is not None:
        if path is not None: throughput_2D_file = os.path.join(path,throughput_2D_file)
        throughput.tofits(throughput_2D_file)
        
    
    if verbose: print("\n> Throughput 2D obtained!")
    if also_return_skyflat:
        return throughput, skyflat
    else:
        return throughput
# =============================================================================
# %% ==========================================================================
# =============================================================================
def compute_integrated_fibre(rss, 
                             list_spectra=None, 
                             valid_wave_min=None,
                             valid_wave_max=None, 
                             min_value_to_plot=0.01,
                             title=" - Integrated values",
                             text="...",
                             correct_negative_sky=False,
                             order_fit_negative_sky=3,
                             kernel_negative_sky=51, low_fibres=10,
                             individual_check=True,
                             use_fit_for_negative_sky=False,
                             last_check=False,
                             **kwargs): #plot=False, warnings=True, verbose=True

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
    min_value_to_plot: float (default 0)
        For values lower than min_value, we set them as min_value
    title : string
        Tittle for the plot
    text: string
        A bit of extra text
    correct_negative_sky : boolean (default = False)
        Corrects negative values making 0 the integrated flux of the lowest fibre
    last_check: boolean (default = False)
        If that is the last correction to perform, say if there is not any fibre
        with has an integrated value < 0.
    **kwargs : kwargs
        where we can find verbose, warnings, plot...

    Example
    ----------
    star1 = compute_integrated_fibre(star1,valid_wave_min=6500, valid_wave_max=6600,
    title = " - [6500,6600]", plot = True)
    """
    verbose = kwargs.get('verbose', False)
    plot =  kwargs.get('plot', False)
    warnings = kwargs.get('warnings', verbose)
    
    wavelength = rss.wavelength
    n_spectra = len(rss.intensity)
    
    if list_spectra is None:
        list_spectra = list(range(n_spectra))
    if valid_wave_min is None: 
        valid_wave_min = rss.koala.info["valid_wave_min"]
    #     valid_wave_min_index = rss.koala.info["valid_wave_min_index"]
    # else:
    #     valid_wave_min_index = np.searchsorted(wavelength, valid_wave_min)        
    if valid_wave_max is None: 
        valid_wave_max = rss.koala.info["valid_wave_max"]
    #     valid_wave_max_index = rss.koala.info["valid_wave_max_index"]
    # else:
    #     valid_wave_max_index = np.searchsorted(wavelength, valid_wave_max)
        
    vprint("\n> Computing integrated fibre values in range [ {:.2f} , {:.2f} ] {}".format(valid_wave_min, valid_wave_max, text),  **kwargs)

    ###!!!: Updated by Pablo
    #integrated_fibre = np.zeros(n_spectra)
    #integrated_fibre_variance = np.zeros(n_spectra)

    region = np.where((wavelength > valid_wave_min
                       ) & (wavelength < valid_wave_max))[0]
    waves_in_region = len(region)

    integrated_fibre = np.nansum(rss.intensity[:, region], axis=1)
    integrated_fibre_variance = np.nansum(rss.variance[:, region], axis=1)
    negative_fibres = (np.where(integrated_fibre < 0)[0]).tolist()
    n_negative_fibres = len(negative_fibres)    # n_negative_fibres = len(integrated_fibre[integrated_fibre < 0])

    if kwargs.get("verbose"):
        print("  - Median value of the integrated flux =", np.round(np.nanmedian(integrated_fibre), 2))
        print("                                    min =", np.round(np.nanmin(integrated_fibre), 2), ", max =",
              np.round(np.nanmax(integrated_fibre), 2))
        print("  - Median value per wavelength         =",
              np.round(np.nanmedian(integrated_fibre) / waves_in_region, 2))
        print("                                    min = {:9.3f} , max = {:9.3f}".format(
            np.nanmin(integrated_fibre) / waves_in_region, np.nanmax(integrated_fibre) / waves_in_region))

    if len(negative_fibres) != 0:
        if warnings or verbose: print(
            "\n> WARNING! : Number of fibres with integrated flux < 0 : {}, that is the {:5.2f} % of the total !".format(
                n_negative_fibres, n_negative_fibres * 100. / n_spectra))
        if correct_negative_sky:
            if verbose: print("  CORRECTING NEGATIVE SKY HERE NEEDS TO BE IMPLEMENTED - NOTHING ELSE DONE")
            pass  ## NEDD TO IMPLEMENT THIS #TODO
            # rss.correcting_negative_sky(plot=plot, order_fit_negative_sky=order_fit_negative_sky,
            #                              individual_check=individual_check,
            #                              kernel_negative_sky=kernel_negative_sky,
            #                              use_fit_for_negative_sky=use_fit_for_negative_sky, low_fibres=low_fibres)
        else:
            if plot and verbose:
                print(
                    "\n> Adopting integrated flux = {:5.2f} for all fibres with negative integrated flux (for presentation purposes)".format(
                        min_value_to_plot))
                print("  This value is {:5.2f} % of the median integrated flux per wavelength".format(
                    min_value_to_plot * 100. / np.nanmedian(integrated_fibre) * waves_in_region))
                integrated_fibre_plot = integrated_fibre.clip(min=min_value_to_plot, max=integrated_fibre.max())
    else:
        integrated_fibre_plot = integrated_fibre
        if last_check:
            if warnings or verbose: print("\n> There are no fibres with integrated flux < 0 !")

    integrated_fibre_sorted = np.argsort(integrated_fibre).tolist()
    if kwargs.get("plot"): rss_map(rss,  variable=integrated_fibre_plot, title=title , **kwargs) #log=log, gamma=gamma, )
    
    rss.koala.integrated_fibre = integrated_fibre
    rss.koala.integrated_fibre_variance = integrated_fibre_variance
    rss.koala.integrated_fibre_sorted = integrated_fibre_sorted
    rss.koala.negative_fibres = negative_fibres
    
    return rss
# =============================================================================
# %% ==========================================================================
# =============================================================================
def cut_wave(rss, wave, wave_index=None,  **kwargs): #plot=False,  plot=False, ymax=None):
     """
     Provides a cut at a particular wavelength. 
     
     Parameters
     ----------
     wave : float
         wavelength to be cut
     wave_index : integer
         wavelength index to be cut. The default is False. If given, wave is not used.
     **kwargs : kwargs
         where we can find verbose, warnings, plot...
         
     Returns
     -------
     corte_wave : list
         list with the values at a particular wavelength, from fibre 0 to -1.
     """
     w = rss.wavelength
     if wave_index is None:
         _w_ = np.abs(w - wave)
         w_min = np.nanmin(_w_)
         wave_index = _w_.tolist().index(w_min)
     else:
         wave = w[wave_index]
     corte_wave = rss.intensity[:, wave_index]

     plot =  kwargs.get('plot', False)
     if plot:
         x = range(len(rss.intensity))
         ptitle = "Intensity cut at " + str(wave) + " $\mathrm{\AA}$ - index =" + str(wave_index)
         plot_plot(x, corte_wave,  xlabel="Fibre", ptitle=ptitle,  **kwargs)  # ymax=ymax,
     return corte_wave
# =============================================================================
# %% ==========================================================================
# =============================================================================
def airmass_from_header(header):
    """
    Compute the airmass extracting the parameters from KOALAS's header'
    """
    # Get ZD, airmass
    ZDSTART = header['ZDSTART']
    ZDEND = header['ZDEND']
    ZD = (ZDSTART + ZDEND) / 2
    airmass = 1 / np.cos(np.radians(ZD))
    return airmass
# =============================================================================
# %% ==========================================================================
# =============================================================================
def obtain_telluric_correction(telluric_correction_file = None,
                               path_to_data = None,
                               star_list = None,
                               width_for_telluric = 30,
                               **kwargs):
    
    verbose = kwargs.get('verbose', False)
    #warnings = kwargs.get('warnings', verbose)
    plot = kwargs.get('plot', False)
    
    if telluric_correction_file is None and star_list is None: raise RuntimeError("No teluric_correction_file or star_list provided!!")
    
    
    if star_list is not None:    # If star_list given, use star_list and save it into telluric_correction_file
        # FIXME well.... so far this is done in rss
        # stars are rss objects
        
        n_stars = len(star_list)
        wavelength = star_list[0].wavelength

        # Check that all wavelengths are the same...
        # Check that were taken the same night 
        
        if verbose: print("\n> Obtaining telluric correction using {} stars...".format(n_stars))
        telluric_correction_list = []
        for i in range(n_stars):
            _telluric_correction_ = TelluricCorrection(star_list[i], verbose=verbose)
            _, fig = _telluric_correction_.telluric_from_model(plot=plot, width=width_for_telluric)
            telluric_correction_list.append(_telluric_correction_.telluric_correction)

        telluric_correction = copy.deepcopy(_telluric_correction_)
        
        telluric_correction.telluric_correction = np.nanmedian(telluric_correction_list, axis=0)
        
        ptitle = "Telluric correction combining {} stars".format(n_stars)
            
        if telluric_correction_file is not None:
            if path_to_data is not None:  telluric_correction_file = os.path.join(path_to_data,telluric_correction_file)
            telluric_correction.save(filename = telluric_correction_file)
    
    else:  # no star_list provided but telluric_correction_file is, reading it    
        if telluric_correction_file[-4:] == "fits":
            # Read it from fits file #TODO
            pass
        else:
            # Read it from txt file #TODO
            # wavelength, telluric_correction_data = read_table(telluric_correction_file, ["f", "f"] ) 
            
            ptitle = "Telluric correction from file {}".format(telluric_correction_file)

    if plot:
        plot_plot(wavelength,telluric_correction.telluric_correction, 
                  ptitle = ptitle,
                  ylabel = "Telluric correction",
                  #percentile_min=0.1, percentile_min=99.9, 
                  **kwargs)
        
    return telluric_correction
# =============================================================================
# %% ==========================================================================
# =============================================================================
def correcting_negative_sky(rss, 
                            mask= None,
                            edgelow=None, edgehigh=None, #low_fibres=10, 
                            kernel_for_negative_sky=51, order_fit_for_negative_sky=3, 
                            min_percentile_for_negative_sky = 5,
                            clip_fit_for_negative_sky = 0.8,
                            use_fit_for_negative_sky=False, 
                            individual_check=True, 
                            sky_fibres = None, 
                            check_only_sky_fibres = False,
                            force_sky_fibres_to_zero=False,
                            exclude_wlm=None,
                            show_fibres=None, 
                            plot_rss_map_for_negative_sky = False,
                            **kwargs): #fig_size=12, plot=True, verbose=True):
    """
    Corrects negative sky with a median spectrum of the lowest intensity fibres

    Parameters  #TODO: Needs to be checked
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
        Substract the fit instead of the smoothed median spectrum
    individual_check: boolean (default = True)
        Check individual fibres and correct if integrated value is negative
    exclude_wlm : list
        exclusion command to prevent large absorption lines from being affected by negative sky correction : (lower wavelength, upper wavelength)
    show_fibres : list of integers (default = [0,450,985])
        List of fibres to show
    force_sky_fibres_to_zero : boolean (default = True)
        If True, fibres defined at the sky will get an integrated value = 0
    **kwargs : kwargs
        where we can find verbose, warnings, plot...
    """
    verbose = kwargs.get('verbose', False)
    plot =  kwargs.get('plot', False)
    warnings = kwargs.get('warnings', verbose)
    rss_out = copy.deepcopy(rss)
    
    
    if verbose: 
        if individual_check: 
            print("> Individual correction of fibres with negative sky ... ")
        else:
            print("> Correcting for negative sky...")

    if sky_fibres is None: # and check_only_sky_fibres:
        try:
            sky_fibres = rss_out.skymodel.sky_fibres
        except Exception:
            if verbose or warnings: 
                if check_only_sky_fibres: print("  WARNING: check_only_sky_fibres resquested but no sky fibre list provided!")

    if sky_fibres is not None and check_only_sky_fibres: 
        if verbose: print("  Only provided sky fibres will be checked for negative sky...")
        number_fibres_to_check = len(sky_fibres)
        fibres_to_check = sky_fibres
    else:
        number_fibres_to_check = len(rss_out.intensity)
        fibres_to_check = range(len(rss_out.intensity))

    # CHECK fit_smooth_spectrum and compare with medfilt
    w = rss_out.wavelength
    # Set limits
    #if edgelow is None: edgelow = rss_out.koala.info["valid_wave_min_index"]   # Not needed if using mask
    #if edgehigh is None: edgehigh = rss_out.koala.info["valid_wave_max_index"]
    if mask is None: mask = rss_out.koala.mask
    
    # Compute integrated fibre 
    rss_out = compute_integrated_fibre(rss_out, verbose=False,plot=False)

    if show_fibres is None:
        show_fibres =[]

    if plot:
        integrated_fibre_sorted = rss_out.koala.integrated_fibre_sorted   #np.argsort(rss_out.integrated_fibre)
        brightest_fibre = integrated_fibre_sorted[-1]
        if sky_fibres is None:
            faintest_fibre = integrated_fibre_sorted[0]
        else:
            faintest_fibre = sky_fibres[0]
        if verbose: 
            if len(show_fibres) > 0:
                print(f"  Plots in fibres {show_fibres}, will be shown, including brightest fibre {brightest_fibre} and faintest fibre {faintest_fibre}.")    
            else:
                if sky_fibres is not None and check_only_sky_fibres: 
                    print(f"  The faintest sky fibre {faintest_fibre} will be shown.") 
                else:
                    print(f"  The brightest fibre {brightest_fibre} and faintest fibre {faintest_fibre} will be shown.") 
        show_fibres.append(brightest_fibre)  # Adding the brightest fibre
        show_fibres.append(faintest_fibre)  # Adding the faintest fibre
    else: 
        show_fibres =[]
        
    if individual_check:
        if force_sky_fibres_to_zero and verbose: print("  Also forcing integrated spectrum of sky_fibres = 0 ... ")
        if verbose: 
            if use_fit_for_negative_sky: 
                print("  Using fit of order",order_fit_for_negative_sky,"to smooth spectrum with kernel",kernel_for_negative_sky,"for correcting the negative skies...")
            else:
                print("  Using smooth spectrum with kernel",kernel_for_negative_sky,"for correcting the negative skies...")

        list_of_fibres_corrected =[]
        fit_list_to_sky_fibres =[]
        #corrected_not_sky_fibres = 0
        #sky_fibres_to_zero = 0
  
        output_every_few = np.sqrt(number_fibres_to_check) + 1
        next_output = -1

        for fibre in fibres_to_check: 
            if verbose:
                if fibre > next_output:
                    sys.stdout.write("\b" * 51)
                    sys.stdout.write("  Checking fibre {:4} ... ({:6.2f} % completed ) ...".format(fibre, fibre * 100. / number_fibres_to_check))
                    sys.stdout.flush()
                    next_output = fibre + output_every_few
            
            plot_this = False
            if fibre in show_fibres:
                if verbose: 
                    sys.stdout.write("\b" * 55)
                    sys.stdout.write(" " *  55)
                    print("\n- Checking fibre", fibre, ":")
                if plot: plot_this = True
            
            smooth, fit = fit_smooth_spectrum(w, rss_out.intensity[fibre], 
                                              mask =[mask[0][fibre], mask[1][fibre]],                                              
                                              edgelow=edgelow, edgehigh=edgehigh, 
                                              kernel_fit=kernel_for_negative_sky, 
                                              index_fit=order_fit_for_negative_sky, 
                                              clip_fit = clip_fit_for_negative_sky,
                                              exclude_wlm = exclude_wlm,
                                              plot=plot_this, verbose=False, hlines=[0.]) 
            
            needs_correction = False
            if np.nanpercentile(fit, min_percentile_for_negative_sky) < 0: needs_correction = True
            if sky_fibres is not None:
                if fibre in sky_fibres and force_sky_fibres_to_zero: 
                    needs_correction = True
                    fit_list_to_sky_fibres.append(fit)
            
            if needs_correction: 
                list_of_fibres_corrected.append(fibre)
                if use_fit_for_negative_sky:
                    if fibre in show_fibres and verbose: print(
                        "      Using fit to smooth spectrum for correcting the negative sky in fibre", fibre, " ...")
                    rss_out.intensity[fibre] -= fit
                    rss_out.variance[fibre] -= fit
                else:
                    if fibre in show_fibres and verbose: print(
                        "      Using smooth spectrum for correcting the negative sky in fibre", fibre, " ...")
                    rss_out.intensity[fibre] -= smooth
                    rss_out.variance[fibre] -= smooth
                    
            else:
                if fibre in show_fibres and verbose: print("      Fibre", fibre,
                                                           "does not need to be corrected for negative sky ...")
        if verbose:
            sys.stdout.write("\b" * 55)
            sys.stdout.write(" " *  55)
            sys.stdout.write("\n  Checking fibres completed!")
            sys.stdout.flush()
            print(" ")

        if force_sky_fibres_to_zero:
            integer = int(len(sky_fibres) *0.25)  # this is 8 for 25 skyfibres
            if verbose: print("  Substracting residual of forcing sky fibres to zero to all fibres combining\n  the negative fits to the",integer,"brightest sky fibres...")
            fit_median_=    np.nanmedian(fit_list_to_sky_fibres[-integer:], axis=0)
            _,fit_median= fit_smooth_spectrum(w,  fit_median_, index_fit=order_fit_for_negative_sky, kernel_fit=None, plot=plot, verbose=False, extra_y=1,
                                              hlines =[-3,-2,-1,0,1,2,3],
                                              ptitle="Order 3 polynomium fitted to small correction of the "+str(integer)+" brightest sky fibres")
            for fibre in range(len(rss_out.intensity)):
                if fibre not in sky_fibres: 
                    rss_out.intensity[fibre] -= fit_median
                    rss_out.variance[fibre] -= fit_median
            

        #corrected_sky_fibres = total_corrected - corrected_not_sky_fibres
        #if verbose:
        total_corrected = len(list_of_fibres_corrected)
            #print("\n> Corrected {} fibres (not defined as sky) and {} out of {} sky fibres !".format(
            #    corrected_not_sky_fibres, corrected_sky_fibres, len(self.sky_fibres)))
            #if force_sky_fibres_to_zero:
            #    print("  The integrated spectrum of", sky_fibres_to_zero, "sky fibres have been forced to 0.")
            #    print("  The integrated spectrum of all sky_fibres have been set to 0.")
        #self.history.append("- Individual correction of negative sky applied")
        #self.history.append("  Corrected " + str(corrected_not_sky_fibres) + " not-sky fibres")
        #if force_sky_fibres_to_zero:
        #    self.history.append("  All the " + str(len(self.sky_fibres)) + " sky fibres have been set to 0")
        #else:
        #    self.history.append("  Corrected " + str(corrected_sky_fibres) + " out of " + str(
        #        len(self.sky_fibres)) + " sky fibres")

    else:
        pass #TODO needs to be implemented?
        # # Get integrated spectrum of n_low lowest fibres and use this for ALL FIBRES
        # integrated_intensity_sorted = np.argsort(self.integrated_fibre)
        # region = integrated_intensity_sorted[0:low_fibres]
        # Ic = np.nanmedian(self.intensity_corrected[region], axis=0)

        # if verbose:
        #     print("\n> Correcting negative sky using median spectrum combining the", low_fibres,
        #           "fibres with the lowest integrated intensity")
        #     print("  which are :", region)
        #     print("  Obtaining smoothed spectrum using a {} kernel and fitting a {} order polynomium...".format(
        #         kernel_negative_sky, order_fit_negative_sky))
        # ptitle = self.object + " - " + str(low_fibres) + " fibres with lowest intensity - Fitting an order " + str(
        #     order_fit_negative_sky) + " polynomium to spectrum smoothed with a " + str(
        #     kernel_negative_sky) + " kernel window"
        # smooth, fit = fit_smooth_spectrum(self.wavelength, Ic, kernel=kernel_negative_sky, edgelow=edgelow,
        #                                   edgehigh=edgehigh, verbose=False, #mask=self.mask[],
        #                                   order=order_fit_negative_sky, plot=plot, hlines=[0.], ptitle=ptitle,
        #                                   fcal=False)
        # if use_fit_for_negative_sky:
        #     self.smooth_negative_sky = fit
        #     if verbose: print(
        #         "  Sustracting fit to smoothed spectrum of {} low intensity fibres to all fibres ...".format(
        #             low_fibres))
        # else:
        #     self.smooth_negative_sky = smooth
        #     if verbose: print(
        #         "  Sustracting smoothed spectrum of {} low intensity fibres to all fibres ...".format(low_fibres))

        # for i in range(self.n_spectra):
        #     self.intensity_corrected[i, :] = self.intensity_corrected[i, :] - self.smooth_negative_sky
        #     # self.sky_emission = self.sky_emission - self.smooth_negative_sky

        # # TODO: New implementation including variance
        # # self.intensity_corrected -= self.smooth_negative_sky[np.newaxis, :]
        # # self.variance_corrected -= self.smooth_negative_sky[np.newaxis, :]

        # if verbose: print("  This smoothed spectrum is stored in self.smooth_negative_sky")
        # self.history.append("- Correcting negative sky using smoothed spectrum of the")
        # self.history.append("  " + str(low_fibres) + " fibres with the lowest integrated value")
    
        
    if plot_rss_map_for_negative_sky:
        rss_map(rss_out, list_spectra=list_of_fibres_corrected, title = " - Fibres corrected for negative sky", log=False)
    
    if verbose:             
        print("  Corrected {} fibres for negative sky, {:.2f}% of fibres.".format(total_corrected, 100*total_corrected/len(rss_out.intensity)))
        if sky_fibres is not None and check_only_sky_fibres: 
            print("  List of fibres corrected =",list_of_fibres_corrected)
            if force_sky_fibres_to_zero: print("  The substraction of the median of the",integer,"brightest sky fibres fit to ALL non sky fibres has been also performed.")
    return rss_out
# =============================================================================
# %% ==========================================================================
# =============================================================================
def find_sky_fibres(rss, n_sky=200, sky_wave_min=None, sky_wave_max=None, **kwargs):
     """
     Identify n_sky spaxels with the LOWEST INTEGRATED VALUES and store them in self.sky_fibres

     Parameters
     ----------
     n_sky : integer (default = 200)
         number of spaxels used for identifying sky.
         200 is a good number for calibration stars
         for real objects, particularly extense objects, set n_sky = 30 - 50
     sky_wave_min, sky_wave_max : float, float (default 0, 0)
         Consider the integrated flux in the range [sky_wave_min, sky_wave_max]
         If 0, they are set to self.valid_wave_min or self.valid_wave_max
     **kwargs : kwargs
         where we can find verbose, warnings, plot...    
         if plot : plots a RSS map with sky positions
     """
     if sky_wave_min is None: sky_wave_min = rss.koala.info["valid_wave_min"]
     if sky_wave_max is None: sky_wave_max = rss.koala.info["valid_wave_max"]
     
     # Assuming cleaning of cosmics and CCD defects, we just use the spaxels with the LOWEST INTEGRATED VALUES
     rss = compute_integrated_fibre(rss, valid_wave_min=sky_wave_min, valid_wave_max=sky_wave_max, plot=False, verbose=False)# **kwargs)
     sorted_by_flux = np.argsort(rss.koala.integrated_fibre)
     vprint("> Identifying",n_sky,"sky spaxels using the lowest integrated values in the [", np.round(sky_wave_min, 2), ",",
           np.round(sky_wave_max, 2), "] range ...", **kwargs)
     
     sky_fibres = sorted_by_flux[:n_sky]
     if kwargs.get("plot"): rss_map(rss, list_spectra=sky_fibres, title=" - Sky Spaxels")  #Show their positions if requested
     return sky_fibres
# =============================================================================
# %% ==========================================================================
# =============================================================================
def SkyFrom_n_sky(rss, n_sky=50, sky_fibres=None, sky_wave_min=None, sky_wave_max=None, 
                  bright_emission_lines_to_substract_in_sky = None,
                  fix_edges = False,
                  list_of_skylines_to_fit_near_bright_emission_lines = None,
                  fit_degree_continuum=None, max_wave_disp = None, min_peak_flux = None, max_sigma = None,  # If substract Ha is needed
                  max_peak_factor = None, min_peak_factor = None,
                  **kwargs):
    """
    Creates a 1D sky model using the lowest n_sky fibres

    Parameters
    ----------
    rss : rss
        DESCRIPTION.
    n_sky : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    skymodel : TYPE
        DESCRIPTION.

    """  
    verbose = kwargs.get('verbose', False)
    plot =  kwargs.get('plot', False)

    if sky_fibres is None:
        if verbose: print("> Using",n_sky,"fibres with the lowest integrated intensity to get sky spectrum...")
        sky_fibres = find_sky_fibres(rss, n_sky=n_sky, sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, **kwargs)
    else:
        if verbose: 
            print("> Using the list of",len(sky_fibres),"fibres provided to get sky spectrum...")
            print("  sky_fibres = ", sky_fibres) 
    if verbose: print("  Combining fibres to create sky spectrum...")
    rss=apply_mask_to_rss(rss) # apply mask to have edges as nan
    sky_spectrum = plot_combined_spectrum(rss, list_spectra=sky_fibres, r=True, median = True, plot=False, verbose=verbose)                    
    sky_spectrum_variance = np.nanmedian([rss.variance[fibre] for fibre in sky_fibres], axis=0)                    
    
    # Check if we have to substract Halpha or other bright emission line
    if bright_emission_lines_to_substract_in_sky is not None:
        if list_of_skylines_to_fit_near_bright_emission_lines is None: list_of_skylines_to_fit_near_bright_emission_lines =[None] * len (bright_emission_lines_to_substract_in_sky)
        if fit_degree_continuum is None: fit_degree_continuum = [None] * len (bright_emission_lines_to_substract_in_sky)
        if max_wave_disp is None: max_wave_disp = [None] * len (bright_emission_lines_to_substract_in_sky)
        if min_peak_flux is None: min_peak_flux = [None] * len (bright_emission_lines_to_substract_in_sky)
        if max_sigma is None: max_sigma = [None] * len (bright_emission_lines_to_substract_in_sky)
        if max_peak_factor is None: max_peak_factor = [None] * len (bright_emission_lines_to_substract_in_sky)
        if min_peak_factor is None: min_peak_factor = [None] * len (bright_emission_lines_to_substract_in_sky)
        
        for i in range(len(bright_emission_lines_to_substract_in_sky)):
            sky_spectrum = substract_Ha_in_sky(w=rss.wavelength, sky_spectrum=sky_spectrum, 
                                       Ha_wavelength = bright_emission_lines_to_substract_in_sky[i],
                                       list_of_skylines_to_fit = list_of_skylines_to_fit_near_bright_emission_lines[i], 
                                       fit_degree_continuum= fit_degree_continuum[i],
                                       max_wave_disp = max_wave_disp [i],
                                       min_peak_flux = min_peak_flux[i],
                                       max_sigma = max_sigma[i],
                                       max_peak_factor = max_peak_factor[i],
                                       min_peak_factor = min_peak_factor[i],
                                       **kwargs)
        #TODO Check what to do with variance if bright emission line is substracted
    # Add continuum to the edges if needed:
    if np.isnan(np.median(sky_spectrum)) and fix_edges:   # This is True if there are nans, should be the edges  #TODO CHECK IT
        sky_spectrum_as_list = list(sky_spectrum)
        # Right edge, typically the one with problems
        # Find the first nan
        first_bad_index=None
        for j in range(rss.koala.info["valid_wave_max_index"],len(rss.wavelength)):
            if np.isnan(sky_spectrum_as_list[j]) and first_bad_index is None: first_bad_index = j

        exclude_wlm =[[rss.wavelength[0], rss.wavelength[first_bad_index]-500], [rss.wavelength[first_bad_index],rss.wavelength[-1]]]
        
        if verbose: print("  Found nans in edges of sky spectrum... fitting a polynomium in right edge...")
            
        _,fit = fit_smooth_spectrum(rss.wavelength, sky_spectrum, 
                            index_fit=3, kernel_fit=3,
                            exclude_wlm= exclude_wlm,
                            xmin=rss.wavelength[first_bad_index]-550, xmax=rss.wavelength[-1]+20, 
                            plot=plot,verbose=False)
            
        sky_spectrum_ =  sky_spectrum_as_list[:first_bad_index] + list(fit[first_bad_index:])
        sky_spectrum=np.array(sky_spectrum_)
        
        # Left edget, typically OK
        if np.isnan(sky_spectrum_as_list[0]):
            pass 
        elif verbose: print("  Left edge of sky spectrum is OK, no need to be fixed")

    try:
        valid_wave_min = rss.koala.info["valid_wave_min"]
        valid_wave_max = rss.koala.info["valid_wave_max"]
    except Exception:
        valid_wave_min = None
        valid_wave_max = None
    if plot: 
        ptitle = "Combined spectrum using requested fibres"
        plot_plot(rss.wavelength,sky_spectrum,vlines=[valid_wave_min,valid_wave_max], ptitle = ptitle)
    
    skymodel = SkyModel(wavelength=rss.wavelength, intensity = sky_spectrum, variance=sky_spectrum_variance)
    skymodel.sky_fibres = sky_fibres
    return skymodel
# =============================================================================
# %% ==========================================================================
# =============================================================================
def quick_find_brightest_line(rss, brightest_fibres_to_combine = None, lowest_fibres_to_combine = None, verbose= False):
    """
    Returns the wavelength where the peak of the brightest emission line is found.

    Parameters
    ----------
    rss : object
        rss 
    brightest_fibres_to_combine : integer, optional
        number of the brightest fibres to combine in the rss to do analysis. The default is 10.
    lowest_fibres_to_combine : integer, optional
        number of the lowest fibres to combine in the rss to do analysis. The default is 10.
    verbose : boolean, optional
        Print the result. The default is False.

    Returns
    -------
    w_max : float
        wavelegth where the peak of the brightest emission line is found.
    """
    if brightest_fibres_to_combine is None: brightest_fibres_to_combine = 10
    if lowest_fibres_to_combine is None: lowest_fibres_to_combine = 10
    fmax = np.nanmedian(rss.intensity[rss.koala.integrated_fibre_sorted[-brightest_fibres_to_combine:]], axis=0)
    fmin = np.nanmedian(rss.intensity[rss.koala.integrated_fibre_sorted[:lowest_fibres_to_combine]], axis=0)    
    s = fmax-fmin                                       
    index_max = s.tolist().index(np.nanmax(s))
    w_max = rss.wavelength[index_max] 
    if verbose: print("> Quick checking wavelength of the brightest emission line in this rss:", w_max)
    return (w_max)
# =============================================================================
# %% ==========================================================================
# =============================================================================
def model_sky_fitting_gaussians(rss, sky_spectrum, 
                                list_of_skylines_to_fit = None, sky_fibres= None,
                                #low_low_list = None, low_high_list=None, high_low_list = None, high_high_list=None,
                                plot_continuum = False,
                                **kwargs):
    
    verbose = kwargs.get('verbose', False)
    plot =  kwargs.get('plot', False)
    warnings = kwargs.get('warnings', verbose)
    
    if list_of_skylines_to_fit is None:    # This is hard-cored here and perhaps should not be.... #!!!
        lines_to_fit_1 = [9049.254, 9038.11, 9001.33, 8988.384, 8958.103, 8943.408, 8919.637, 8903.123, 8885.856, 8867.605]
        lines_to_fit_2 = [8849.70, 8827.112,8836.27,   8791.189,  8777.70,  8767.925,8761.337]   #8761.617,8761.337]
        lines_to_fit_3 = [6300.309,6363.783,   6563.4 ] #  6863.79,   ## 6563.4 seems to be rest-H-alpha??   6828.22,6832.70 a double
        lines_to_fit_4 = [8280.94, 8288.34, 8298.40,8310.725, 8352.78,8344.613, 8364.509]
        lines_to_fit_5 = [8382.402,8399.167, 8415.242,8430.20, 8452.266, 8465.374, 8493.56, 8504.78] #,    8538.690,8548.957] faint
        
        #low_low_list =  [8700, 8700, 6100, 8200, 8200]   # Not needed now
        #low_high_list = [8750, 8750, 6200, 8265, 8265]
        #high_low_list = [9060, 8880, 6620, 8560, 8560]
        #high_high_list =[9085, 9085, 6700, 8610, 8610]
        
        list_of_skylines_to_fit =[lines_to_fit_1,lines_to_fit_2, lines_to_fit_3,lines_to_fit_4, lines_to_fit_5]
    
    #TODO Check that skylines are NOT in position expected for redshifted emission lines, discard them if it is the case
    
    
    # Setting values
    w = rss.wavelength
    gaussian_model = np.zeros_like(rss.intensity)
    intensity_out = copy.deepcopy(rss.intensity)
    #n_spectra = len(rss.intensity)
    if sky_fibres is None: 
        sky_fibres = range(len(rss.intensity))
    n_fibres = len(sky_fibres)
    fitted_lines = [[] for x in range(len(rss.intensity))]

    if verbose: 
        print("\n> Modelling sky using Gaussian fits and input sky spectrum.")
        print("  - Performing Gaussian fits in input rss...")
        sys.stdout.write("    Working on fibre:                           ")
        sys.stdout.flush()
       
    ### Fitting wavelength ranges
    for fibre in sky_fibres: #range(0,n_spectra): #637
        if verbose:
            sys.stdout.write("\b" * 25)
            sys.stdout.write("{:5.0f}  ({:5.2f}% completed)".format(fibre, fibre/n_fibres*100))
            sys.stdout.flush()
        
        for i in range(len(list_of_skylines_to_fit)): 
            #print("Fibre = ",fibre,",  lista = ",i)
            _,continuum=fit_smooth_spectrum(w, intensity_out[fibre], plot=plot_continuum, auto_trim=True, verbose=False)
            try:
                intensity_out[fibre], gaussian_model_, fitted_lines_ = fit10gaussians(w, intensity_out[fibre], 
                                                                                      lines_to_fit=list_of_skylines_to_fit[i],
                                                                                      continuum=continuum,
                                                                                      #low_low=low_low_list[i],low_high=low_high_list[i],
                                                                                      #high_low=high_low_list[i],high_high=high_high_list[i], 
                                                                                      return_fit = True, return_fitted_lines = True,
                                                                                      plot=False, plot_continuum=plot_continuum)
                gaussian_model[fibre] = gaussian_model[fibre] +   gaussian_model_  
                fitted_lines[fibre] = fitted_lines[fibre] + fitted_lines_
            except Exception:
                if warnings: print("  WARNING: Gaussian fit failed in Fibre = ",fibre,",  list = ",i,"\n")
    if verbose:
        sys.stdout.write("\b" * 51)
        sys.stdout.write("  Checking fibres completed!                             ")
        sys.stdout.flush()
        print(" ")
        
    ### Fit the sky_spectrum
    if verbose: print("  - Performing Gaussian fits in sky spectrum...")
    sky_out = copy.deepcopy(sky_spectrum)
    gaussian_model_sky=np.zeros_like(w)
    fitted_lines_sky_dictionary ={}
    _,continuum_sky=fit_smooth_spectrum(w, sky_spectrum, plot=plot_continuum, auto_trim=True, verbose=False)
    for i in range(len(list_of_skylines_to_fit)):
        sky_out, gaussian_model_sky_, fitted_lines_sky_dict_ = fit10gaussians(w, sky_out, lines_to_fit=list_of_skylines_to_fit[i], 
                                                                              continuum=continuum_sky,
                                                                              #low_low=low_low_list[i],low_high=low_high_list[i],
                                                                              #high_low=high_low_list[i],high_high=high_high_list[i], 
                                                                              return_dictionary_for_fitted_lines = True,
                                                                              plot=False)
        gaussian_model_sky = gaussian_model_sky + gaussian_model_sky_
        fitted_lines_sky_dictionary.update(fitted_lines_sky_dict_)

    ### Check fibre by fibre to create an individualised model of sky        
    if verbose: print("  - Creating individualised sky model per fibre...")
    #skymodel_gaussians = np.zeros_like(rss.intensity)
    skymodel_no_gaussians = np.zeros_like(rss.intensity)
    
    for fibre in sky_fibres: #range(0,n_spectra):
        # Check valid fits in this fibre
        individual_sky_gaussian_model = np.zeros_like(w)
        for line in fitted_lines[fibre]: 
            gauss_solution = fitted_lines_sky_dictionary.get(line)
            x0 =gauss_solution[0]
            y0 =gauss_solution[1]
            sigma =gauss_solution[2]
            individual_sky_gaussian_model = individual_sky_gaussian_model + gauss(w, x0,y0,sigma)
        #skymodel_gaussians[fibre] = individual_sky_model
        skymodel_no_gaussians[fibre] = sky_spectrum - individual_sky_gaussian_model
    
    # Create sky model adding Gaussian + no-Gaussian models
    skymodel = skymodel_no_gaussians + gaussian_model
    # Plot if requested
    if plot:
        rss_image(rss, image=skymodel, greyscale =True, title ="- Sky model using Gaussian fits to skylines", add_title=True)
    # Return results
    return skymodel, gaussian_model
# =============================================================================
# %% ==========================================================================
# =============================================================================
def substract_Ha_in_sky(w, sky_spectrum,
                        Ha_wavelength = None,
                        list_of_skylines_to_fit = None,
                        #fit_degree_continuum=2, #### All of these in **kwargs
                        #max_wave_disp = None, min_peak_flux = None, max_sigma = None, max_peak_factor = None, min_peak_factor = None,
                        #return_dictionary_for_fitted_lines = False
                        xmin=None, xmax=None, ymin = None, ymax = None, extra_y = None,
                        plot_solution = None, **kwargs):
    """
    Substract a faint H-alpha residua in sky spectrum . This could be applied to any bright emission line as requested.   

    Parameters
    ----------
    w : list
        wavelength.
    sky_spectrum : list
        sky spectrum.
    Ha_wavelength : float, optional
        If given, the wavelenght of Ha.
    list_of_skylines_to_fit : list, optional
        List with sky emission lines that need to be fit with H-alpha in sky spectrum 
    xmin,xmax,ymin,ymax,extra_y : all floats, for plotting, optional
    plot_solution : Boolean, optional
        If True. or plot in kwargs is True, plot solution. The default is None.
    **kwargs : kwargs
        where we can find verbose, warnings, plot...

    Returns
    -------
    sky_spectrum_out : list
        Sky spectrum with H-alpha substracted.
    """
    verbose = kwargs.get('verbose', False)
    plot =  kwargs.get('plot', False)
    if plot_solution is None: plot_solution = plot
    #warnings = kwargs.get('warnings', verbose)
    
    if verbose: print(f"> Substracting Ha at around {Ha_wavelength} to sky spectrum...")
    
    if list_of_skylines_to_fit is None:
        lines_to_fit = [Ha_wavelength]
    else:
        lines_to_fit = [Ha_wavelength] + list_of_skylines_to_fit
    
    out=fit10gaussians(w,sky_spectrum, lines_to_fit, 
                       return_dictionary_for_fitted_lines=True, **kwargs)
    
    gaussHa = gauss(w, out[2].get(Ha_wavelength)[0], out[2].get(Ha_wavelength)[1], out[2].get(Ha_wavelength)[2])
    sky_spectrum_out  = sky_spectrum - gaussHa           

    if plot_solution:
        if xmin is None: xmin = Ha_wavelength - 30
        if xmax is None: xmax = Ha_wavelength + 30
        plot_plot(w, [sky_spectrum,sky_spectrum_out], 
                  vlines = lines_to_fit, hlines=[0], ptitle = "H-alpha substracted to sky spectrum",
                  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, extra_y=extra_y)
    
    return sky_spectrum_out
# =============================================================================
# %% ==========================================================================
# =============================================================================
def find_emission_lines_in_koala(rss,
                                 brightest_fibres_to_combine = None,
                                 brightest_line = None,
                                 **kwargs): #plot=True, verbose=True, warnings=True):

    verbose = kwargs.get('verbose', False)
    #plot =  kwargs.get('plot', False)
    #warnings = kwargs.get('warnings', False)
    
    if brightest_fibres_to_combine is None: brightest_fibres_to_combine = 6  
    if brightest_line is None:
        if rss.wavelength[0] < 6562.82 and 6562.82 < rss.wavelength[-1]: brightest_line="Ha"
        if rss.wavelength[0] < 5006.84 and 5006.84 < rss.wavelength[-1]: brightest_line="[OIII]"      
    
    if verbose: print(f"> Combining {brightest_fibres_to_combine} brightest lines to obtain object spectrum...")
    
    
    fmax = np.nanmedian(rss.intensity[rss.koala.integrated_fibre_sorted[-brightest_fibres_to_combine:]], axis=0)
    redshifted_lines_detected_dictionary, emission_line_gauss_spectrum = find_emission_lines(rss.wavelength, fmax, 
                                                                                             ref_line=brightest_line,
                                                                                             **kwargs)
    rss.koala.redshifted_emission_lines_detected_dictionary = redshifted_lines_detected_dictionary
    rss.koala.emission_lines_gauss_spectrum = emission_line_gauss_spectrum
    #corrections_done.append("emission_line_identification")
    
    # Update brightest_line_wavelength
    _ion_list_ = [redshifted_lines_detected_dictionary[i]["ion"] for i in redshifted_lines_detected_dictionary.keys()]
    _obs_wave_list = [redshifted_lines_detected_dictionary[i]["gauss"][0] for i in redshifted_lines_detected_dictionary.keys()]
    brightest_line_wavelength =_obs_wave_list[_ion_list_.index(brightest_line)] 
    rss.koala.info["brightest_line_wavelength"] =brightest_line_wavelength
    rss.koala.info["brightest_line"] =brightest_line
# =============================================================================
# %% ==========================================================================
# =============================================================================
def rss_continuum_image(rss, mask = None, **kwargs): #verbose = False, plot=False):
    """
    Creates a continuum image of a rss object.
    
    **kwargs : kwargs
        where we can find verbose, warnings, plot...
    
    """
    verbose = kwargs.get('verbose', False)
    kwargs["verbose"]=False 
    plot =  kwargs.get('plot', False)
    kwargs["plot"]=False
    w=rss.wavelength
    n_fibres = len(rss.intensity)
    continuum = []
    
    if mask is None:
        try:
            mask = [[rss.koala.mask[0][i],rss.koala.mask[1][i]] for i in range(len(rss.intensity)) ]
        except Exception:
            pass
    
    if verbose: 
        print("> Creating continuum image of this rss...")
        sys.stdout.write("    Working on fibre:                           ")
        sys.stdout.flush()
    
    for i in range(n_fibres):
        if verbose:
            sys.stdout.write("\b" * 25)
            sys.stdout.write("{:5.0f}  ({:5.2f}% completed)".format(i, i/n_fibres*100))
            sys.stdout.flush()
        #print(mask[i])
        _,cont= fit_smooth_spectrum(w, rss.intensity[i], #mask = mask[i], #auto_trim=True, 
                                    mask = mask[i],
                                    #mask=[rss.koala_info["first_good_wave_per_fibre"][i], rss.koala_info["last_good_wave_per_fibre"][i]],
                                    **kwargs) # plot=False,  verbose=False)
        continuum.append(cont)
    if verbose:
        sys.stdout.write("\b" * 51)
        sys.stdout.write("  Checking fibres completed!                             ")
        sys.stdout.flush()
    continuum=np.array(continuum)  
    if plot: rss_image(rss, image=continuum, greyscale=True, log=True, title=" - Continuum image", add_title=True)
    return continuum
# =============================================================================
# %% ==========================================================================
# =============================================================================
def clean_telluric_residuals(rss, 
                             fibre_list = None,
                             interval_to_clean = None,
                             max_dispersion = None,
                             half_width_continuum = None,
                             min_value_per_wave_for_fitting = None,
                             lines_to_fit = None,
                             sigman = None,
                             max_wave_disp=None, 
                             max_sigma=None, 
                             min_peak_flux=None, 
                             max_peak_factor = None,
                             min_peak_factor = None,
                             continuum = None,
                             xmin=None, xmax = None,
                             plot_fibre_list = None,
                             plot_fit = False, 
                             plot_individual_comparison = False,
                             verbose_fits=False,
                             **kwargs): ### verbose =False, warnings = False, ):
    
    verbose = kwargs.get('verbose', False)
    plot =  kwargs.get('plot', False)
    warnings = kwargs.get('warnings', False)
    
    # Check values
    #if min_value_per_wave_for_fitting is None: min_value_per_wave_for_fitting = 50

    if half_width_continuum is None: half_width_continuum = 70
    if interval_to_clean is None: interval_to_clean=[7590,7640]
    if lines_to_fit is None: lines_to_fit=[7600,7615,7625]
    if sigman is None: sigman =[5,5, 8]
    if max_wave_disp is None: max_wave_disp=[10,20,10],
    if max_sigma is None: max_sigma=12
    if min_peak_factor is None: min_peak_factor = 3
    if min_peak_flux is None: min_peak_flux=-1000 
    if max_peak_factor is None: max_peak_factor = 3
    
    if verbose: 
        texto = "> Cleaning the residuals of this rss in the interval "+str(interval_to_clean)
        if min_value_per_wave_for_fitting is not None:
            texto=texto+"\n  checking Gaussian fits in fibres with min_value_per_wave_for_fitting > "+str(min_value_per_wave_for_fitting)
        print(texto+" ...")
    
    # Prepare the data
    rss_out = copy.deepcopy(rss)
    if fibre_list is None: fibre_list = range(len(rss.intensity))    
    n_fibres = len(fibre_list) 
    w=rss.wavelength

    if continuum is None:  continuum= rss_continuum_image(rss)

    if plot_fibre_list is None and plot_individual_comparison is True:
        plot_fibre_list =[int(round(uniform(0, (n_fibres-1)/100),2)*100) for i in range(3)]
        if verbose: print("  Plotting comparison in 3 fibres randomly choosen :",plot_fibre_list)  
    
    if verbose: 
        sys.stdout.write("  Working on fibre:                           ")
        sys.stdout.flush()

    wave_index_min = np.abs(w - interval_to_clean[0]).tolist().index(np.nanmin(np.abs(w - interval_to_clean[0])))
    wave_index_max = np.abs(w - interval_to_clean[1]).tolist().index(np.nanmin(np.abs(w - interval_to_clean[1]))) 
    
    if xmin is None:  xmin = w[int((wave_index_max+wave_index_min)/2 -  half_width_continuum)]
    if xmax is None:  xmax = w[int((wave_index_max+wave_index_min)/2 + half_width_continuum)]
    if max_dispersion is None: max_dispersion= 1.5
        
    j=0
    for fibre in fibre_list: 
        if verbose:
            sys.stdout.write("\b" * 46)
            sys.stdout.write("  Working on fibre: {:5.0f}  ({:6.2f}% completed)".format(fibre, j/n_fibres*100))
            sys.stdout.flush()
            
        # Check if Gaussian fits needed    (is None never does it)
        if min_value_per_wave_for_fitting is not None:
            if rss_out.koala.integrated_fibre[fibre]/len(w) > min_value_per_wave_for_fitting:    
                try:
                    plot_this = False
                    if fibre in plot_fibre_list or plot_fit: plot_this = True
                    rss_out.intensity[fibre]= fit10gaussians(w, rss_out.intensity[fibre],
                                                             continuum=continuum[fibre], 
                                                             lines_to_fit=lines_to_fit,  
                                                             sigman =sigman,
                                                             max_wave_disp=max_wave_disp,
                                                             max_sigma=max_sigma, 
                                                             min_peak_factor = min_peak_factor,
                                                             min_peak_flux=min_peak_flux, 
                                                             max_peak_factor = max_peak_factor, 
                                                             plot=plot_this, verbose=verbose_fits)
                                                             #xmin=7490,xmax=7730, ymin=-100, ymax=160, extra_y = 0,
                                                             #return_dictionary_for_fitted_lines=True)
                except Exception:
                    if warnings: print(" Gaussian fits failed in fibre ",fibre)
        
        # Random reduction in interval 
        plot_comparison = False
        if plot_fibre_list is not None:    
            if fibre in plot_fibre_list:
                if verbose: print("  Comparison between before (red) and after (blue), continuum in green, in fibre",fibre)
                plot_comparison = True
        rss_out.intensity[fibre] = clip_spectrum_using_continuum(w,rss_out.intensity[fibre], 
                                                                 continuum=continuum[fibre], 
                                                                 xmin=xmin,xmax=xmax, 
                                                                 interval_to_clean=interval_to_clean, 
                                                                 max_dispersion=max_dispersion,
                                                                 verbose=False, plot=plot_comparison)
        j=j+1
               
    if verbose:
        sys.stdout.write("\b" * 46)
        sys.stdout.write("  Checking fibres completed!                             ")
        sys.stdout.flush()
        print(" ")
    if plot: 
        compare_rss_images(rss_out, image_list=[rss.intensity, rss_out.intensity, rss_out.intensity/rss.intensity],
                   cmap=["binary_r","binary_r","binary_r"], 
                   log=[True,True,False], #gamma=[0,0,0],
                   title=["BEFORE","AFTER","AFTER/BEFORE"],
                   colorbar_label = ["Intensity", "Intensity", "Ratio"],
                   clow=[None,None,0], chigh=[None,None,2],
                   percentile_min = 0, percentile_max=100,
                   fig_size=[8.5,6.5],
                   wmin=interval_to_clean[0]-100,wmax=interval_to_clean[1]+100)
        
    return rss_out 
# =============================================================================
# %% ==========================================================================
# =============================================================================
def clean_skyline(rss, skyline,
                  fibre_list= None,
                  show_fibres=None,
                  lowlow= 40, lowhigh=15, highlow=15, highhigh=40,
                  xmin = None, xmax =None,
                  broad = 2.5,
                  **kwargs):
    """
    Substract a skyline using a Gaussian fit in all spectra/fibres or in a given fibre list.

    Parameters
    ----------
    rss : object
        rss to be clean.
    skyline : float
        wavelength of the emission line to be cleaned.
    fibre_list : list, optional
        List of fibres to clean. The default is None, that is all.
    show_fibres : list, optional
        List of fibres to PLOT. The default is None.
    **kwargs : kwargs
        where we can find verbose, warnings, plot...
    
    THESE PARAMETERS FOR GAUSSIAN FITTING using task fluxes:
    lowlow : float, optional
        DESCRIPTION. The default is 40.
    lowhigh : float, optional
        DESCRIPTION. The default is 15.
    highlow : float, optional
        DESCRIPTION. The default is 15.
    highhigh : float, optional
        DESCRIPTION. The default is 40.
    xmin : float, optional
        DESCRIPTION. The default is None.
    xmax : float, optional
        DESCRIPTION. The default is None.
    broad : float, optional
        DESCRIPTION. The default is 2.5.
        
    Returns
    -------
    rss_out : object
        rss with skyline substracted.
    """
    verbose = kwargs.get('verbose', False)
    plot =  kwargs.get('plot', False)
    warnings = kwargs.get('warnings', verbose)
    
    if skyline in [5577.338, 5577]:
        skyline = 5577.338
        xmin,xmax = 5520,5640
    else:
        if xmin is None: xmin = skyline - 60
        if xmax is None: xmax = skyline + 60    
    
    if fibre_list is None:
        fibre_list = list(range(len(rss.intensity)))
        if verbose: print("\n> Correcting skyline 5577 in all fibres...")
    else:
        if verbose: print("\n> Correcting skyline 5577 in selected fibres...")

    rss_out = copy.deepcopy(rss)
    w= rss_out.wavelength
    
    if verbose:
        sys.stdout.write("  Working on fibre:                           ")
        sys.stdout.flush()
        output_every_few = np.sqrt(len(fibre_list)) + 1
        next_output = -1

    n_fibres = len(rss.intensity)
    j=0
    for fibre in fibre_list:
        if verbose:
            if fibre > next_output:
                sys.stdout.write("\b" * 46)
                sys.stdout.write("  Working on fibre: {:5.0f}  ({:6.2f}% completed)".format(fibre, j/n_fibres*100))
                sys.stdout.flush()
                next_output = fibre + output_every_few
                
        plot_fit = False
        if show_fibres is not None and plot:
            if fibre in show_fibres: plot_fit = True
        resultado = fluxes(w, rss.intensity[fibre],  skyline, 
                           lowlow=lowlow, lowhigh=lowhigh, highlow=highlow, highhigh=highhigh,
                           xmin = xmin, xmax =xmax,
                           broad=broad, plot=plot_fit, verbose=False, plot_sus=False, fcal=False,
                           warnings=warnings)  # Broad is FWHM for Gaussian sigma = 1,
        rss_out.intensity[fibre]=resultado[-1]
        j=j+1
   
    if verbose:
        sys.stdout.write("\b" * 46)
        sys.stdout.write("  Checking fibres completed!                             ")
        sys.stdout.flush()
        print(" ")
    if plot: compare_rss_images(rss_out, image_list=[rss.intensity, rss_out.intensity, rss_out.intensity/rss.intensity],
                   cmap=["binary_r","binary_r","binary_r"], 
                   log=[True,True,False], #gamma=[0,0,0],
                   title=["BEFORE","AFTER","AFTER/BEFORE"],
                   colorbar_label = ["Intensity", "Intensity", "Ratio"],
                   clow=[None,None,0], chigh=[None,None,2],
                   percentile_min = 0, percentile_max=100,
                   fig_size=[9.0,6.],
                   wmin=xmin-40,wmax=xmax+30)
    #self.history.append()   #TODO
    return rss_out   
# =============================================================================
# %% ==========================================================================
# =============================================================================
def clean_extreme_negatives(rss, fibre_list=[], percentile_min=0.3, continuum=None, **kwargs): # plot=True, verbose=True):
     """
     Remove pixels that have extreme negative values (that is below percentile_min) and replace for the median value

     Parameters
     ----------
     fibre_list : list of integers (default all)
         List of fibers to clean. The default is [], that means it will do everything.
     percentile_min : float, (default = 0.3)
         Minimum value accepted as good.
     **kwargs : kwargs
         where we can find verbose, warnings, plot...
     """
     verbose = kwargs.get('verbose', False)
     plot =  kwargs.get('plot', False)
     #warnings = kwargs.get('warnings', verbose)
     
     if fibre_list is None:
         fibre_list = list(range(len(rss.intensity)))
         if verbose: print("\n> Correcting the extreme negatives in all fibres, making any pixel below")
     else:
         if verbose: print("\n> Correcting the extreme negatives in given fibres, making any pixel below")

     rss_out = copy.deepcopy(rss)
     minimo = np.nanpercentile(rss_out.intensity, percentile_min)

     if verbose:
         print("  np.nanpercentile(intensity_corrected, ", percentile_min, ") = ", np.round(minimo, 2))
         print("  to have the median value of the fibre...")            
         sys.stdout.write("  Fixing {} spectra...       ".format(len(fibre_list)))
         sys.stdout.flush()
         output_every_few = np.sqrt(len(fibre_list)) + 1
         next_output = -1
         i = 0

     if plot:
         correction_map = np.zeros_like(rss_out.intensity)
              
     #median_value_fibre_list=[]
     for fibre in fibre_list:
          if verbose:
                i = i + 1
                if fibre > next_output:
                    sys.stdout.write("\b" * 6)
                    sys.stdout.write("{:5.2f}%".format(i * 100. / len(fibre_list)))
                    sys.stdout.flush()
                    next_output = fibre + output_every_few
         
          if continuum is None:
              median_value_fibre = np.nanmedian(rss.intensity[fibre]) 
              rss_out.intensity[fibre] = [median_value_fibre if x < minimo else x for x in rss.intensity[fibre]]
              if plot: correction_map[fibre] = [1 if x == median_value_fibre else 0 for x in rss_out.intensity[fibre]]

          else:
              n_wave = len(rss.wavelength)
              s = rss.intensity[fibre]
              rss_out.intensity[fibre] = [continuum[fibre][j] if s[j] < minimo else s[j] for j in range(n_wave)]
              if plot: correction_map[fibre] = [1 if s[j] < minimo else s[j] for j in range(n_wave)]
    

     if verbose:
         sys.stdout.write("\b" * 51)
         sys.stdout.write("  Checking fibres completed!                  ")
         sys.stdout.flush()
         print(" ")

     if plot:
         rss_image(rss_out, image=correction_map*rss_out.mask, cmap="binary", title="Extreme negatives - Correction map", clow=0,chigh=1,
                   colorbar_label="Extreme negatives = 0")

     #self.history.append(     #!!! TODO
     #    "- Extreme negatives (values below percentile " + str(np.round(percentile_min, 3)) + " = " + str(
     #        np.round(minimo, 3)) + " ) cleaned")
     return rss_out
# =============================================================================
# %% ==========================================================================
# =============================================================================
def kill_cosmics(rss, brightest_line_wavelength = None, 
                 valid_wave_min = None, valid_wave_max = None,
                 width_bl=20., fibre_list=None, max_number_of_cosmics_per_fibre=12,
                 kernel_median_cosmics=5, cosmic_higher_than=100, extra_factor=1., continuum=None,
                 plot_waves=None, plot_cosmic_image=True, plot_rss_images=False, only_plot_cosmics_cleaned = False, **kwargs):
                 #plot=True, verbose=True, warnings=True):
    """
    Kill cosmics in a RSS.

    Parameters   #!!! NEED TO BE CHECKED
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
    **kwargs : kwargs
        where we can find verbose, warnings, plot...

    Returns
    -------
    Save the corrected RSS to self.intensity_corrected

    """
    verbose = kwargs.get('verbose', False)
    plot =  kwargs.get('plot', False)
    warnings = kwargs.get('warnings', False)
    
    if only_plot_cosmics_cleaned:
        plot=False
        plot_waves = None
        plot_cosmic_image = False
        plot_rss_images = False
    
    rss_out = copy.deepcopy(rss)
    g  = rss_out.intensity
    #gv = rss_out.variance   ### ACTUALLY, we don't need to modify variance if only change values of cosmics...

    if plot is False:
        plot_rss_images = False
        plot_cosmic_image = False

    x = range(len(g))
    w = rss.wavelength
    if fibre_list is None:
    #len(fibre_list) == 0:
        fibre_list_ALL = True
        fibre_list = list(range(len(g)))
        if verbose: print("\n> Finding and killing cosmics in all fibres...")
    else:
        fibre_list_ALL = False
        if verbose: print("\n> Finding and killing cosmics in given fibres...")

    if brightest_line_wavelength is None:
        if warnings or verbose: print("\n\n  WARNING !!!!! brightest_line_wavelength is NOT given!\n")
        
        if valid_wave_min is None: valid_wave_min = rss_out.koala.info["valid_wave_min"]
        if valid_wave_max is None: valid_wave_max = rss_out.koala.info["valid_wave_max"]
     
        rss = compute_integrated_fibre(rss_out, valid_wave_min=valid_wave_min, valid_wave_max=valid_wave_max, plot=False, verbose=False)# **kwargs)
        integrated_fibre_sorted_by_flux = np.argsort(rss_out.integrated_fibre)
        

        median_spectrum = plot_combined_spectrum(rss_out, plot=plot, median=True,
                                                      list_spectra=integrated_fibre_sorted_by_flux[-11:-1],
                                                      ptitle="Combined spectrum using 10 brightest fibres",
                                                      percentile_max=99.5, percentile_min=0.5 , r=True)
        # brightest_line_wavelength=w[int(self.n_wave/2)]
        brightest_line_wavelength = w[median_spectrum.tolist().index(np.nanmax(median_spectrum))]
        
        if brightest_line_wavelength < valid_wave_min: brightest_line_wavelength = valid_wave_min
        if brightest_line_wavelength > valid_wave_max: brightest_line_wavelength = valid_wave_max

        if warnings: print(
            "  Assuming brightest_line_wavelength is the max of median spectrum of 10 brightest fibres =",
            np.round(brightest_line_wavelength,2))

    # Get the cut at the brightest_line_wavelength
    corte_wave_bl = cut_wave(rss_out, brightest_line_wavelength)
    gc_bl = medfilt(corte_wave_bl, kernel_size=kernel_median_cosmics)
    max_val = np.abs(corte_wave_bl - gc_bl)

    if plot:
        ptitle = "Intensity cut at brightest line wavelength = " + str(
            np.round(brightest_line_wavelength, 2)) + " $\mathrm{\AA}$ and extra_factor = " + str(extra_factor)
        plot_plot(x, [max_val, extra_factor * max_val], percentile_max=99, xlabel="Fibre", ptitle=ptitle,
                  ylabel="abs (f - medfilt(f))",
                  label=["intensity_cut", "intensity_cut * extra_factor"])

    # List of waves to plot:
    plot_waves_index = []
    if plot_waves is not None:
        for wave in plot_waves:
            wave_min_vector = np.abs(w - wave)
            plot_waves_index.append(wave_min_vector.tolist().index(np.nanmin(wave_min_vector)))
        if verbose: print("  List of waves to plot:", plot_waves)

    # Start loop
    lista_cosmicos = []
    cosmic_image = np.zeros_like(rss_out.intensity)
    
    if verbose:
         sys.stdout.write("  Checking {} wavelength...         ".format(len(w)))
         sys.stdout.flush()
         output_every_few = np.sqrt(len(w)) + 1
         next_output = -1
    
    for i in range(len(w)):
        if verbose:
                if i > next_output:
                    sys.stdout.write("\b" * 6)
                    sys.stdout.write("{:5.2f}%".format(i * 100. / len(w)))
                    sys.stdout.flush()
                    next_output = i + output_every_few
        
        wave = w[i]
        # Perhaps we should include here not cleaning in emission lines... #FIXME
        correct_cosmics_in_fibre = True
        if width_bl != 0:
            if wave > brightest_line_wavelength - width_bl / 2 and wave < brightest_line_wavelength + width_bl / 2:
                if warnings: print(
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
            corte_wave = cut_wave(rss_out,wave)
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


    if verbose:
        sys.stdout.write("\b" * 70)
        sys.stdout.write("  Checking spectra completed!                 ")
        sys.stdout.flush()
        print(" ")

    # Check number of cosmics found
    if plot_cosmic_image: rss_image(rss_out, image=cosmic_image, cmap="binary_r", title=" - Cosmics identification", add_title=True)
    # print(lista_cosmicos)
    if verbose: print("\n  Total number of cosmic candidates found = "+str(len(lista_cosmicos))+", correcting only if < "+str(max_number_of_cosmics_per_fibre)+" per wavelength ...")

    if plot_rss_images: rss_image(rss,cmap="binary_r", title=" - Before correcting cosmics", add_title=True)

    if fibre_list_ALL == False and verbose == True: print("  Correcting cosmics in selected fibres...")
    cosmics_cleaned = 0
    for fibre in fibre_list:
        if np.nansum(cosmic_image[fibre]) > 0:  # A cosmic is found
            # print("Fibre ",fibre," has cosmics!")
            f = g[fibre]
            #fv = gv[fibre]
            if continuum is None:
                gc = medfilt(f, kernel_size=21)
                #gc_variance =  medfilt(fv, kernel_size=21)
            else:
                gc = continuum[fibre]
                #gc_variance = np.sqrt(continuum[fibre])  #!!! CHECH
            bad_indices = [i for i, x in enumerate(cosmic_image[fibre]) if x == 1]
            if len(bad_indices) <= max_number_of_cosmics_per_fibre:
                for index in bad_indices:
                    g[fibre, index] = gc[index]
                    #gv[fibre, index] = gc_variance[index]
                    cosmics_cleaned = cosmics_cleaned + 1
            else:
                cosmic_image[fibre] = np.zeros_like(w)
                if warnings: print("  WARNING! Fibre", fibre, "has", len(bad_indices),
                                   "cosmics found, this is larger than", max_number_of_cosmics_per_fibre,
                                   "and hence is NOT corrected!")

    rss_out.intensity = copy.deepcopy(g)
    #rss_out.variance = copy.deepcopy(gv)
    if plot_rss_images: rss_image(rss_out, cmap="binary_r", title=" - After correcting cosmics", add_title=True)

    # Check number of cosmics eliminated
    #if cosmics_cleaned != len(lista_cosmicos):
    if plot_cosmic_image or only_plot_cosmics_cleaned: 
        rss_image(rss_out, image=cosmic_image, cmap="binary_r", title=" - Cosmics cleaned", add_title=True)

    if verbose: print("  Total number of cosmics cleaned = ", cosmics_cleaned)


    # self.history.append("- " + str(cosmics_cleaned) + " cosmics cleaned using:")              #TODO
    # self.history.append("  brightest_line_wavelength = " + str(brightest_line_wavelength))
    # self.history.append(
    #     "  width_bl = " + str(width_bl) + ", kernel_median_cosmics = " + str(kernel_median_cosmics))
    # self.history.append(
    #     "  cosmic_higher_than = " + str(cosmic_higher_than) + ", extra_factor = " + str(extra_factor))
    return rss_out
# =============================================================================
# %% ==========================================================================
# =============================================================================
def get_rss_mask(rss, make_zeros = False):   # THIS TASK SHOULD BE A CORRECTION CLASS, including apply() #TODO

    n_waves = len(rss.wavelength)
    n_fibres = len(rss.intensity)
    indeces_low = rss.koala.mask[0]
    indeces_high = rss.koala.mask[1] 
    if make_zeros:
        mask = [   [0 if j < indeces_low[fibre] or j > indeces_high[fibre] else 1 for j in range(n_waves)]     for fibre in range(n_fibres)]            
    else:
        mask = [   [np.nan if j < indeces_low[fibre] or j > indeces_high[fibre] else 1 for j in range(n_waves)]     for fibre in range(n_fibres)]

    return np.array(mask)
# =============================================================================
# %% ==========================================================================
# =============================================================================
def apply_mask_to_rss(rss, mask = None,   # THIS TASK SHOULD BE A CORRECTION CLASS, including apply() #TODO
                      make_zeros=False, **kwargs):    # verbose, plot, warnings should be there 
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
    verbose = kwargs.get('verbose', False)
    rss_out = copy.deepcopy(rss)
    mask_in_rss = False
    if mask is not None:
        mask_provided = True                                   # Mask provided 
        if str(type(mask)) == "<class 'str'>":   # It is a file, read mask TODO
            pass           
    else:
        mask_provided = False         # Mask NOT provided 
        try:
            if rss.mask is not None: 
                mask = rss.mask
                mask_in_rss = True
        except Exception:
            mask = get_rss_mask(rss_out)
    rss_out.intensity = rss_out.intensity * mask
    rss_out.variance = rss_out.variance * mask        
    rss_out.mask = np.array(mask)
    if verbose: 
        if mask_provided:
            print("> Mask provided has been applied to rss.") 
        else:
            if mask_in_rss: 
                if make_zeros:
                    print("> Mask included in rss applied to make 0 all bad pixels in edges.")
                else:
                    print("> Mask included in rss applied to make nan all bad pixels in edges.")    
            else:
                if make_zeros:
                    print("> Mask computed and applied to make 0 all bad pixels in edges.")
                else:
                    print("> Mask computed and applied to make nan all bad pixels in edges.")  
               
    return rss_out
# =============================================================================
# %% ==========================================================================
# =============================================================================
def process_n_koala_rss_files(filename_list = None,
                              path = None,
                              rss_list = None,
                              rss_object_name_list = None,
                              save_rss_to_fits_file_list = None,
                              #calibration_night = None,
                              # more things to add #TODO
                              **kwargs):
    
    verbose = kwargs.get('verbose', False)
    #warnings = kwargs.get('warnings', verbose)
    #plot =  kwargs.get('plot', False)
    
    number_rss_files = None
    if filename_list is not None:  number_rss_files = len(filename_list)
    if rss_list is not None:  number_rss_files = len(rss_list)
    if number_rss_files is None:
        raise RuntimeError("NO filename_list or rss_list provided!!!") 
            
    if rss_list is None: rss_list = [None] * number_rss_files
    if filename_list is None: filename_list = [None] * number_rss_files
    if rss_object_name_list is None: rss_object_name_list = [None] * number_rss_files
    if path is None: 
        path = [None] * number_rss_files
    elif np.isscalar(path):
        path = [path] * number_rss_files
    if save_rss_to_fits_file_list is None: 
        save_rss_to_fits_file_list = [None] * number_rss_files
    elif np.isscalar(save_rss_to_fits_file_list):
        save_rss_to_fits_file_list = [save_rss_to_fits_file_list] * number_rss_files
        
    processed_rss_files = []
    
    for i in range(number_rss_files):
        if verbose: print("\n> Processing rss {} of {} :    ---------------------------------------------".format(i+1,number_rss_files))

        _rss_ = process_koala_rss(filename=filename_list[i], 
                                  path = path[i], 
                                  rss = rss_list[i],
                                  rss_object_name=rss_object_name_list[i], 
                                  save_rss_to_fits_file = save_rss_to_fits_file_list[i],
                                  # more things to add #TODO
                                  **kwargs)
        processed_rss_files.append(_rss_)   

    return processed_rss_files
# =============================================================================
# %% ==========================================================================
# =============================================================================

# THIS IS ONE OF THE MOST IMPORTANT TASKS, AS IT CAN CALL EVERYTHING ELSE FOR PROCESSING / CLEANING RSS


def process_koala_rss(filename=None, path=None,
                      rss = None,  rss_object_name = None,
                      save_rss_to_fits_file = None,
                      rss_clean=False,
                      # Calibration of the night
                      calibration_night = None,
                      # MASK
                      apply_mask = False,
                      mask = None,   # This can be from a file or a mask
                      make_zeros_in_mask = False, #plot_mask=False,  # Mask if given
                      #valid_wave_min=0, valid_wave_max=0,  # These two are not needed if Mask is given
                      apply_throughput=False,    # ----------------- THROUGHPUT (T)
                      throughput = None,
                      #throughput_2D=[], throughput_2D_file="", throughput_2D_wavecor=False,
                      correct_ccd_defects=False,  # ----------------- CORRECT NANs in CCD (C)
                      #remove_5577=False, kernel_correct_ccd_defects=51, fibre_p=-1, plot_suspicious_fibres=False,
                      
                      fix_wavelengths=False,     # ----------------- FIX WAVELENGTH SHIFTS   (W)
                      wavelength_shift_correction = None, #sol=None, #[0, 0, 0],
                      sky_lines_for_wavelength_shifts=None, sky_lines_file_for_wavelength_shifts = None, n_sky_lines_for_wavelength_shifts  = 3,
                      maxima_sigma_for_wavelength_shifts = 2.5, maxima_offset_for_wavelength_shifts = 1.5,
                      median_fibres_for_wavelength_shifts = 7, 
                      index_fit_for_wavelength_shifts = 2, kernel_fit_for_wavelength_shifts= None, clip_fit_for_wavelength_shifts =0.4,
                      fibres_to_plot_for_wavelength_shifts=None,
                      show_fibres_for_wavelength_shifts=None,
                      median_offset_per_skyline_weight = 1.,     # 1 is the BLUE line (median offset per skyline), 0 is the GREEN line (median of solutions), anything between [0,1] is a combination.
                      show_skylines_for_wavelength_shifts = None,
                      plot_wavelength_shift_correction_solution = None,
                      
                      correct_for_extinction=False, # ----------------- EXTINCTION  (X)
                      
                      apply_telluric_correction = False,    # ----------------- TELLURIC  (U)
                      telluric_correction=None, 
                      #telluric_correction_file= None,
                      width_for_telluric_correction = 30,
                      clean_5577 = False,
                      
                      sky_method=None,        # ----------------- SKY (S)
                      skycorrection = None,
                      n_sky=None, 
                      sky_wave_min=None, sky_wave_max=None,
                      sky_fibres=None,  # do_sky=True
                      sky_spectrum = None,  #sky_spectrum_file=None      ### These should be together
                      bright_emission_lines_to_substract_in_sky = None,
                      list_of_skylines_to_fit_near_bright_emission_lines = None,
                      list_of_skylines_to_fit = None,
                      fibre_list=None,
                      fix_edges = None,
                      #sky_rss=[0], scale_sky_rss=0, scale_sky_1D=0.,
                      #maxima_sigma=3.,
                
                      #sky_lines_file=None, exclude_wlm=[[0, 0]], emission_line_file = None,
                      is_sky=False, win_sky=0, #auto_scale_sky=False, ranges_with_emission_lines=[0], cut_red_end=0,
                      
                      correct_negative_sky=False,      # ----------------- NEGATIVE SKY  (N)
                      order_fit_for_negative_sky=7, 
                      kernel_for_negative_sky=21, 
                      clip_fit_for_negative_sky = 0.8,
                      individual_check_for_negative_sky=True,
                      use_fit_for_negative_sky=True,
                      check_only_sky_fibres = False,
                      force_sky_fibres_to_zero=False,
                      individual_sky_substraction=True,  
                      show_fibres_for_negative_sky = None,
                      plot_rss_map_for_negative_sky = False, 
                      
                      id_el=False,     # ----------------- ID emission lines   (I)
                      brightest_line=None, # "Ha",
                      brightest_line_wavelength=None,
                      brightest_fibres_to_combine = None,    #high_fibres=20, 
                      #lowest_fibres_to_combine = None, low_fibres=10,  using n_sky if needed
                      
                      #clean_sky_residuals=False, # ----------------- CLEAN SKY RESIDUALS   (R)
                      big_telluric_residua_correction = False ,
                      max_dispersion_for_big_telluric = 1.4,
                      min_value_per_wave_for_fitting_big_telluric = 50, 
                      telluric_residua_at_6860_correction = False,
                      max_dispersion_for_6860 = 1.4,
                      min_value_per_wave_for_for_6860 = None, 
                      continuum_model_after_sky_correction = None,
                      fibres_to_fix=None,
                      #features_to_fix=[], sky_fibres_for_residuals=[],
                      #remove_negative_median_values=False,
                      #fix_edges=False,  # Not implemented
                      
                      correct_extreme_negatives=False,    # ----------- EXTREME NEGATIVES   (R)
                      percentile_min_for_extreme_negatives=0.05,
                      clean_cosmics=False,                # ----------- CLEAN COSMICS     (R)
                      width_bl=20., kernel_median_cosmics=5, cosmic_higher_than=100., extra_factor=1.,
                      max_number_of_cosmics_per_fibre=12, 
                      only_plot_cosmics_cleaned = False,
                      
                      print_summary=False, 
                      plot_final_rss=None,   # None: if a correction is done, it will plot it at the end 
                      plot_final_rss_title = None,   # log= True, gamma = 0.,fig_size=12,
                      **kwargs):    # verbose, plot, warnings should be there                  
        
    if rss_clean:  # Just read file if rss_clean = True
        apply_mask = False
        apply_throughput = False                             # T
        correct_ccd_defects = False                          # C
        fix_wavelengths = False                              # W
        #wavelength_shift_correction = None  #sol
        correct_for_extinction = False                       # X
        apply_telluric_correction = False                    # T
        clean_5577 = False                                   # R
        sky_method = None                                    # S
        correct_negative_sky = False                         # N
        id_el = False                                        # E
        #clean_sky_residuals = False
        big_telluric_residua_correction = False              # R01
        telluric_residua_at_6860_correction = False          # R02
        clean_cosmics = False                                # R04
        correct_extreme_negatives = False                    # R08
        #fix_edges = False
        #remove_negative_median_values = False
        # plot_final_rss = plot
        plot = False
        #verbose = False 
    elif calibration_night is not None:
        if throughput is None and calibration_night.throughput is not None: throughput = calibration_night.throughput
        if wavelength_shift_correction is None and calibration_night.wavelength_shift_correction is not None: wavelength_shift_correction = calibration_night.wavelength_shift_correction     
        if telluric_correction is None and calibration_night.telluric_correction is not None: telluric_correction=calibration_night.telluric_correction
        #if flux_calibration is not None
                   
    verbose = kwargs.get('verbose', False)
    #warnings = kwargs.get('warnings', verbose)
    plot =  kwargs.get('plot', False)
    #plot_all = kwargs.get('plot_all', False)
    
    if plot is False:
        only_plot_cosmics_cleaned = False
        plot_rss_map_for_negative_sky = False

    # Reading the file or the object
    if filename is not None:
        rss = koalaRSS(filename, path = path,
                       rss_object_name = rss_object_name, **kwargs)
    elif rss is None:
        raise RuntimeError("  No rss provided !!!!")
    else:  # rss is an object
        rss=copy.deepcopy(rss)
        if rss_object_name is not None:
            rss.koala.info["rss_object_name"]=rss_object_name
    
    # Get name of original file in case we need it for saving rss at the end
    if filename is None: filename = rss.koala.info["path_to_file"]   

    # Check the number of corrections to be applied
    if (apply_throughput == False and correct_ccd_defects == False and fix_wavelengths == False
        and correct_for_extinction == False and apply_telluric_correction == False and clean_5577 == False
        and sky_method == None and correct_negative_sky == False and id_el == False
        and big_telluric_residua_correction == False and telluric_residua_at_6860_correction == False #and clean_sky_residuals == False    
        and clean_cosmics == False and correct_extreme_negatives == False #and fix_edges == False # and remove_negative_median_values == False
        and is_sky == False and apply_mask == False):
        # If nothing is selected to do, we assume that the RSS file is CLEAN
        rss_clean = True
        #plot_final_rss = plot   
        #plot = False
        #verbose = False
    elif verbose:
        if filename is not None: 
            print("\n> Processing file {} as requested... ".format(filename))
        else:
            filename = rss.koala.info["path_to_file"]
            
            if rss_clean is False:
                if rss_object_name is None:
                    print("\n> Processing rss object as requested... ")
                else:
                    print("\n> Processing rss object {} as requested... ".format(rss_object_name))
                if calibration_night is not None: print("  Calibration of the night provided!")
 
    # Check wavelength range to guess brightest emission line 
    if brightest_line is None:
        if rss.wavelength[0] < 6562.82 and 6562.82 < rss.wavelength[-1]: brightest_line="Ha"
        if rss.wavelength[0] < 5006.84 and 5006.84 < rss.wavelength[-1]: brightest_line="[OIII]"
    if brightest_line_wavelength is None:
        brightest_line_wavelength=quick_find_brightest_line(rss, brightest_fibres_to_combine=brightest_fibres_to_combine, lowest_fibres_to_combine=n_sky)
              
    # Corrections:
    corrections_done = []
    
    if apply_throughput:   # -------------------------------------------------------------------  (T)
        throughput_corr = ThroughputCorrection(throughput=throughput, **kwargs)
        rss = throughput_corr.apply(rss, **kwargs) 
        corrections_done.append("apply_throughput") 
    
    if correct_ccd_defects:  # -----------------------------------------------------------------  (C)
        rss = clean_nan(rss, **kwargs)
        corrections_done.append("correct_ccd_defects")

    if fix_wavelengths:    # -------------------------------------------------------------------  (W)
        # Find correction if not provided
        if wavelength_shift_correction is None:
            wavelength_shift_correction = WavelengthShiftCorrection.wavelength_shift_using_skylines(rss, 
                                                                                                    sky_lines =sky_lines_for_wavelength_shifts,
                                                                                                    sky_lines_file = sky_lines_file_for_wavelength_shifts,
                                                                                                    n_sky_lines = n_sky_lines_for_wavelength_shifts,
                                                                                                    valid_wave_min = rss.koala.info["valid_wave_min"],
                                                                                                    valid_wave_max = rss.koala.info["valid_wave_max"],
                                                                                                    maxima_sigma=maxima_sigma_for_wavelength_shifts, 
                                                                                                    maxima_offset=maxima_offset_for_wavelength_shifts,
                                                                                                    median_fibres = median_fibres_for_wavelength_shifts,
                                                                                                    index_fit = index_fit_for_wavelength_shifts, 
                                                                                                    kernel_fit = kernel_fit_for_wavelength_shifts, 
                                                                                                    clip_fit =clip_fit_for_wavelength_shifts,
                                                                                                    fibres_to_plot=fibres_to_plot_for_wavelength_shifts,
                                                                                                    show_fibres=show_fibres_for_wavelength_shifts,
                                                                                                    **kwargs) #, plot=True, verbose =True)
        elif plot and plot_wavelength_shift_correction_solution is None: 
            plot_wavelength_shift_correction_solution = True
        else: plot_wavelength_shift_correction_solution = False
     
        # Apply solution
        rss = wavelength_shift_correction.apply(rss, 
                                                wavelength_shift_correction=wavelength_shift_correction,
                                                median_offset_per_skyline_weight = median_offset_per_skyline_weight,
                                                show_fibres=show_fibres_for_wavelength_shifts,
                                                show_skylines = show_skylines_for_wavelength_shifts,
                                                plot_solution = plot_wavelength_shift_correction_solution,
                                                **kwargs)   # verbose = True, plot=False)

        corrections_done.append("wavelength_shift_correction")
    
    if correct_for_extinction: # ---------------------------------------------------------------  (X)
        atm_ext_corr = AtmosphericExtCorrection(verbose=verbose)
        rss = atm_ext_corr.apply(rss)
        corrections_done.append("extinction_correction")
        
    if apply_telluric_correction: # ------------------------------------------------------------  (U)
        if telluric_correction is None:
            telluric_correction = TelluricCorrection(rss, verbose=verbose)
            _, fig = telluric_correction.telluric_from_model(plot=plot, width=width_for_telluric_correction)
        elif str(type(telluric_correction)) == "<class 'str'>":    # It is a file
            pass
            #telluric_correction = read_file(telluric_correction)    #### TODO

        rss = telluric_correction.apply(rss, verbose=verbose)
        corrections_done.append("telluric_correction")
        
    if rss.wavelength[0] < 5577 and rss.wavelength[-1] > 5577 and clean_5577: # ----------------  (S)
        rss = clean_skyline(rss, skyline = 5577, **kwargs)
        corrections_done.append("clean_5577")
    
    if sky_method is not None:  # --------------------------------------------------------------  (S)
    
        if verbose: print("> Correcting sky using the",sky_method,"method.")

        if skycorrection is None:  
            if sky_spectrum is not None:   # Valid for 1D, 2D
                if verbose: print("  Sky spectrum provided ...")
                skymodel = SkyModel(wavelength=rss.wavelength, intensity = sky_spectrum, variance=np.sqrt(sky_spectrum))  
            else:  # sky_method in ["self", "selffit"]: DEFAULT if sky_method is not ["2D", "1D", "1Dfit"]
                if n_sky is None:
                    if verbose: print("  Using Pablo's method for obtaining self sky spectrum ...")
                    skymodel = SkyFromObject(rss, bckgr_estimator='mad', source_mask_nsigma=5, remove_cont=False)   #TODO
                else:
                    skymodel = SkyFrom_n_sky(rss, n_sky, sky_fibres=sky_fibres, 
                                             sky_wave_min=sky_wave_min, sky_wave_max=sky_wave_max, 
                                             bright_emission_lines_to_substract_in_sky = bright_emission_lines_to_substract_in_sky,
                                             list_of_skylines_to_fit_near_bright_emission_lines = list_of_skylines_to_fit_near_bright_emission_lines,
                                             fix_edges = fix_edges,
                                             **kwargs)
                    sky_fibres = skymodel.sky_fibres
                    
            #TODO Perhaps it is best to do here the identification of emission lines to know where they are....
            
            if sky_method in ["1Dfit", "selffit"]:   
                sky_spectrum=skymodel.intensity
                sky2D, gaussian_model = model_sky_fitting_gaussians(rss, sky_spectrum, 
                                                                    list_of_skylines_to_fit = list_of_skylines_to_fit, 
                                                                    fibre_list= sky_fibres,
                                                                    plot_continuum = False, **kwargs)
                skymodel = SkyModel(wavelength=rss.wavelength, intensity = sky2D, variance=np.sqrt(sky2D)) 
                skymodel.gaussian_model = gaussian_model
            skycorrection = SkySubsCorrection(skymodel)
        
        rss, _ = skycorrection.apply(rss, verbose=verbose)
        rss.skymodel = skycorrection.skymodel
        if sky_fibres is not None: rss.skymodel.sky_fibres = sky_fibres
        corrections_done.append("sky_correction")
        
    # Correct negative sky  # ------------------------------------------------------------------  (N)
    if is_sky is False and correct_negative_sky is True:   #TODO: This has to be a correction applied to data container
    
        rss= correcting_negative_sky(rss, 
                                     order_fit_for_negative_sky = order_fit_for_negative_sky,
                                     use_fit_for_negative_sky = use_fit_for_negative_sky,
                                     kernel_for_negative_sky = kernel_for_negative_sky,
                                     clip_fit_for_negative_sky = clip_fit_for_negative_sky,
                                     individual_check = individual_check_for_negative_sky,
                                     check_only_sky_fibres = check_only_sky_fibres,
                                     sky_fibres= sky_fibres,
                                     show_fibres = show_fibres_for_negative_sky,
                                     force_sky_fibres_to_zero=force_sky_fibres_to_zero,
                                     plot_rss_map_for_negative_sky = plot_rss_map_for_negative_sky,
                                     **kwargs)  #plot=plot, verbose = verbose, 
        
        corrections_done.append("negative_sky_correction")
    
    
    # Identify emission lines     # ------------------------------------------------------------  (E)  
    if id_el:
        find_emission_lines_in_koala(rss, brightest_fibres_to_combine = brightest_fibres_to_combine, 
                                     brightest_line = brightest_line,  **kwargs)
        # Update brightest_line_wavelength
        brightest_line_wavelength = rss.koala.info["brightest_line_wavelength"]          
        corrections_done.append("emission_line_identification")
        
    # Clean telluric residua, extreme negatives & cosmics    # ---------------------------------  (R)  
    if big_telluric_residua_correction or telluric_residua_at_6860_correction or correct_extreme_negatives or clean_cosmics:
        # Get continuum image if any of these have been requested
        if continuum_model_after_sky_correction is None:
            continuum_model_after_sky_correction = rss_continuum_image(rss, **kwargs)
            rss.koala.continuum_model_after_sky_correction = continuum_model_after_sky_correction
    
    if big_telluric_residua_correction:               #TODO: This has to be a correction applied to data container
        rss = clean_telluric_residuals(rss,                                    
                                       fibre_list = fibres_to_fix,       
                                       continuum = continuum_model_after_sky_correction,
                                       max_dispersion = max_dispersion_for_big_telluric,
                                       min_value_per_wave_for_fitting = min_value_per_wave_for_fitting_big_telluric, 
                                       **kwargs)   
        corrections_done.append("big_telluric_residua_correction")

    if telluric_residua_at_6860_correction:          #TODO: This has to be a correction applied to data container
        rss =     clean_telluric_residuals (rss, continuum=continuum_model_after_sky_correction,    
                                            max_dispersion = max_dispersion_for_6860,
                                            min_value_per_wave_for_fitting = min_value_per_wave_for_for_6860, #10, #50,
                                            interval_to_clean = [6850,6876],
                                            lines_to_fit=[6857,6867],
                                            sigman = [[3.5,3.5]],
                                            max_sigma = [5,5],
                                            max_wave_disp = [15,15],
                                            #fibre_list=use_list,
                                            #plot_fibre_list=use_list,
                                            #plot_individual_comparison 
                                            **kwargs)        
        corrections_done.append("telluric_residua_at_6860_correction")
    
    if correct_extreme_negatives:        #TODO: This has to be a correction applied to data container
        rss = clean_extreme_negatives(rss,                                       
                                      fibre_list=fibres_to_fix, 
                                      percentile_min=percentile_min_for_extreme_negatives,  
                                      continuum = continuum_model_after_sky_correction,
                                      **kwargs)
        corrections_done.append("correct_extreme_negatives")

    # Clean cosmics    (R)
    if clean_cosmics:                    #TODO: This has to be a correction applied to data container
        rss=kill_cosmics(rss,                                                  
                         brightest_line_wavelength, 
                         width_bl=width_bl, 
                         kernel_median_cosmics=kernel_median_cosmics,
                         cosmic_higher_than=cosmic_higher_than, extra_factor=extra_factor,
                         max_number_of_cosmics_per_fibre=max_number_of_cosmics_per_fibre,
                         continuum = continuum_model_after_sky_correction,
                         fibre_list=fibres_to_fix, 
                         only_plot_cosmics_cleaned = only_plot_cosmics_cleaned, **kwargs)  #plot_cosmic_image=plot, plot_RSS_images=plot, verbose=verbose)   
        corrections_done.append("clean_cosmics")
    
    # Apply mask for edges if corrections applied or requested
    if apply_telluric_correction or sky_method is not None or correct_negative_sky or clean_cosmics or correct_extreme_negatives or apply_mask:
        rss = apply_mask_to_rss(rss, mask=mask, make_zeros=make_zeros_in_mask, verbose=verbose) 
        
    # Summary
    rss.koala.corrections_done = corrections_done
    rss.koala.info["history"] = corrections_done  # TODO: needs to do proper history 
    
    if plot_final_rss is None and len(corrections_done) > 0 : plot_final_rss= True
    
    if len(corrections_done) > 0: 
        if plot_final_rss:
            if plot_final_rss_title is None:
                plot_final_rss_title = rss.info['name'] + " - RSS image - "+str(len(corrections_done))+" corrections applied"
            rss_image(rss, title=plot_final_rss_title, **kwargs)
    
        if verbose or print_summary:    
            if filename is not None:
                print("\n> Summary of processing rss file", '"' + filename + '"', ":")
            elif rss_object_name is not None:
                print("\n> Summary of processing rss object", '"' + rss_object_name + '"', ":")
            else: 
                print("\n> Summary of processing this rss:")
            print('  Name of the observation = "{}",   Name of this Python RSS object = "{}".'.format(rss.info['name'],rss.koala.info['rss_object_name']))
            if len(corrections_done) > 0:
                print(f"  Corrections applied: {len(corrections_done)} in total:")
                for correction in corrections_done: print(f"  - {correction}")
            if rss.koala.info['rss_object_name'] is not None:
                print(f"\n  All applied corrections are stored in {rss.koala.info['rss_object_name']}.intensity !")
    
    if save_rss_to_fits_file is not None:
        if save_rss_to_fits_file == "auto": # These two options, "auto" and "clean", should go in task save_rss_to_fits_file
            save_rss_to_fits_file = name_keys(filename, path= path, 
                                              apply_throughput = apply_throughput,                                       # T
                                              correct_ccd_defects = correct_ccd_defects,                                 # C
                                              fix_wavelengths = fix_wavelengths,                                         # W        
                                              correct_for_extinction = correct_for_extinction,                           # X
                                              apply_telluric_correction = apply_telluric_correction,                     # T
                                              sky_method = sky_method,                                                   # S
                                              correct_negative_sky = correct_negative_sky,                               # N
                                              id_el = id_el,                                                             # E
                                              big_telluric_residua_correction = big_telluric_residua_correction,         # R01
                                              telluric_residua_at_6860_correction = telluric_residua_at_6860_correction, # R02
                                              clean_cosmics = clean_cosmics,                                             # R04
                                              correct_extreme_negatives = correct_extreme_negatives)                     # R08 
        
        if save_rss_to_fits_file == "clean": save_rss_to_fits_file = filename[:-5]+"_clean.fits"
            
        koala_rss_to_fits(rss, fits_file=save_rss_to_fits_file, path = path, verbose=verbose)
    
    return rss













# =============================================================================
# %% ==========================================================================
# =============================================================================
#
# KOALA TASKS FOR READING AND SAVING FILES
#
# =============================================================================
# %% ==========================================================================
# =============================================================================
def read_rss(file_path,              
             wcs,
             intensity_axis=0,
             variance_axis=1,
             bad_fibres_list=None,
             instrument=None,
             verbose=False,
             log=None,
             header=None,
             fibre_table=None,
             info=None
             ):
    """TODO """
    # Blank dictionary for the log
    if log is None:
        log = {'read': {'comment': None, 'index': None},
               'mask from file': {'comment': None, 'index': 0},
               'blue edge': {'comment': None, 'index': 1},
               'red edge': {'comment': None, 'index': 2},
               'cosmic': {'comment': None, 'index': 3},
               'extreme negative': {'comment': None, 'index': 4},
               'wavelength fix': {'comment': None, 'index': None, 'sol': []}}
    if header is None:
        # Blank Astropy Header object for the RSS header
        # Example how to add header value at the end
        # blank_header.append(('DARKCORR', 'OMIT', 'Dark Image Subtraction'), end=True)
        header = fits.header.Header(cards=[], copy=False)

    file_name = os.path.basename(file_path)

    vprint("\n> Reading RSS file", file_name, "created with", instrument, "...",
           verbose=verbose)

    #  Open fits file. This assumes that RSS objects are written to file as .fits.
    with fits.open(file_path) as rss_fits:
        # Read intensity using rss_fits_file[0]
        all_intensities = np.array(rss_fits[intensity_axis].data, dtype=np.float32)
        if bad_fibres_list is not None:
            intensity = np.delete(all_intensities, bad_fibres_list, 0)
            n_bad_fibres = len(bad_fibres_list)
        else:
            intensity = all_intensities
            n_bad_fibres = 0
        # Bad pixel verbose summary
        vprint("\n  Number of spectra in this RSS =", len(all_intensities),
            ",  number of good spectra =", len(intensity),
            " ,  number of bad spectra =", n_bad_fibres,
            verbose=verbose)
        if bad_fibres_list is not None:
            vprint("  Bad fibres =", bad_fibres_list, verbose=verbose)

        # Read errors if exist a dedicated axis
        if variance_axis is not None:
            all_variances = rss_fits[variance_axis].data
            if bad_fibres_list is not None:
                variance = np.delete(all_variances, bad_fibres_list, 0)
            else:
                variance = all_variances

        else:
            vprint("\n  WARNING! Variance extension not found in fits file!", verbose=verbose)
            variance = np.full_like(intensity, fill_value=np.nan)

    # Create wavelength from wcs
    nrow, ncol = wcs.array_shape
    wavelength_index = np.arange(ncol)
    wavelength = wcs.dropaxis(1).wcs_pix2world(wavelength_index, 0)[0]
    # log
    comment = ' '.join(['- RSS read from ', file_name])
    log['read']['comment'] = comment
    # First Header value added by the PyKoala routine
    header.append(('DARKCORR', 'OMIT', 'Dark Image Subtraction'), end=True)

    # Blank mask (all 0, i.e. making nothing) of the same shape of the data
    mask = np.zeros_like(intensity)

    return RSS(intensity=intensity,
               variance=variance,
               wavelength=wavelength,
               mask=mask,
               info=info,
               log=log,
               )

# =============================================================================
# =============================================================================
def name_keys(filename, path= None, 
              #apply_mask = False,
              apply_throughput = False,                             # T
              correct_ccd_defects = False,                          # C
              fix_wavelengths = False,                              # W        
              correct_for_extinction = False,                       # X
              apply_telluric_correction = False,                    # T
              sky_method = None,                                    # S
              correct_negative_sky = False,                         # N
              id_el = False,                                        # E
              big_telluric_residua_correction = False,              # R01
              telluric_residua_at_6860_correction = False,          # R02
              clean_cosmics = False,                                # R04
              correct_extreme_negatives = False):                   # R08
    """
    Task for automatically naming output rss files.
    """    
    if path is not None: filename = os.path.join(path,filename)
     
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
    if correct_for_extinction : 
        X="X"           # X = extinction corrected
    else:
        X = clave[-6]
    if apply_telluric_correction : 
        U="U"           # U = Telluric corrected
    else:
        U = clave[-5]    
    if sky_method is not None : 
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
    if big_telluric_residua_correction or telluric_residua_at_6860_correction or clean_cosmics or correct_extreme_negatives: 
        R="R"           # R = Sky and CCD residuals
    else:
        R = clave[-1]    

    clave="_"+T+C+W+X+U+S+E+N+R
            
    if filename[-8:] == "red.fits" :       
        return filename[0:-5]+clave+".fits"
    else:
        return filename[0:-15]+clave+".fits"
# =============================================================================
# =============================================================================
def list_koala_fits_files_in_folder(path, verbose = True, use2=True, use3=False, ignore_offsets=True, 
                              skyflat_names=None, ignore_list=None, return_list=False):  
    
    list_of_objetos=[]
    list_of_files=[]
    list_of_exptimes=[]
    if skyflat_names is None: skyflat_names = ["skyflat", "SKYFLAT", "SkyFlat"]

    if ignore_list is None:  ignore_list = ["a", "b", "c", "d", "e", "f", "p", "pos", "Pos",
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
    if len(files_) == 0: raise NameError('No files found within folder '+path)
 
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
                pass

    
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
                pass
        if use3:
            try:
                if not ignore_offsets:
                    object_fits[0]=object_fits[0]+object_fits[2]
                elif object_fits[2] not in ignore_list:
                    object_fits[0]=object_fits[0]+object_fits[2]
            except Exception:
                pass
            
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

    if return_list: return list_of_objetos, list_of_files, list_of_exptimes, date, grating
# =============================================================================
# =============================================================================
def koala_rss_to_fits(rss, data=None, 
                      fits_file="RSS_rss.fits", path = None,
                      text="RSS data", #sol=None,
                      description=None, **kwargs):   # verbose=True): 
    """
    Routine to save RSS data as fits

    Parameters
    ----------
    rss is the rss
    description = if you want to add a description      
    """
    verbose = kwargs.get('verbose', True)
    warnings = kwargs.get('warnings', verbose)
    
    # Check data: intensity or something else?
    if data is None:
        data = rss.intensity
        if verbose: print("> Using rss.intensity of given RSS file to create fits file...")
    else:
        if len(np.array(data).shape) != 2:
            if verbose or warnings:
                print("> The data provided are NOT valid, as they have a shape",data.shape)
                print("  Using rss.intensity_corrected instead to create a RSS fits file !")
            data = rss.intensity
        else:
            if verbose: print("> Using the data provided + structure of given RSS file to create fits file...")
    
    
    # Check the path
    if path is not None: fits_file = os.path.join(path,fits_file)
    # if save_rss_to_fits_file == "auto":                                #TODO, this can be done reading rss.koala.corrections_done
    #     save_rss_to_fits_file = name_keys(filename, path= path, 
    #                                       apply_throughput = apply_throughput,                                       # T
    #                                       correct_ccd_defects = correct_ccd_defects,                                 # C
    #                                       fix_wavelengths = fix_wavelengths,                                         # W        
    #                                       correct_for_extinction = correct_for_extinction,                           # X
    #                                       apply_telluric_correction = apply_telluric_correction,                     # T
    #                                       sky_method = sky_method,                                                   # S
    #                                       correct_negative_sky = correct_negative_sky,                               # N
    #                                       id_el = id_el,                                                             # E
    #                                       big_telluric_residua_correction = big_telluric_residua_correction,         # R01
    #                                       telluric_residua_at_6860_correction = telluric_residua_at_6860_correction, # R02
    #                                       clean_cosmics = clean_cosmics,                                             # R04
    #                                       correct_extreme_negatives = correct_extreme_negatives)                     # R08 
    
    if fits_file == "clean": fits_file = rss.koala.info["path_to_file"][:-5]+"_clean.fits"
    
    # Star building fits file
    koala_info = rss.koala.info
    fits_image_hdu = fits.PrimaryHDU(data)
    fits_image_hdu.header = copy.deepcopy(rss.koala.header)
        
    # fits_image_hdu.header['BITPIX']  =  16  
    # fits_image_hdu.header["ORIGIN"]  = 'AAO'    #    / Originating Institution                        
    # fits_image_hdu.header["TELESCOP"]= 'Anglo-Australian Telescope'    # / Telescope Name  
    # fits_image_hdu.header["ALT_OBS"] =                 1164 # / Altitude of observatory in metres              
    # fits_image_hdu.header["LAT_OBS"] =            -31.27704 # / Observatory latitude in degrees                
    # fits_image_hdu.header["LONG_OBS"]=             149.0661 # / Observatory longitude in degrees 

    # fits_image_hdu.header["INSTRUME"] = "AAOMEGA-KOALA"             # / Instrument in use  
    # fits_image_hdu.header["GRATID"]  = koala_info["aaomega_grating"]      # / Disperser ID 
    # SPECTID = "UNKNOWN"
    # if koala_info["aaomega_arm"] == "red" :  SPECTID="RD"
    # if koala_info["aaomega_arm"] == "blue" :  SPECTID="BL"
    # fits_image_hdu.header["SPECTID"] = SPECTID                        # / Spectrograph ID
    # fits_image_hdu.header["DICHROIC"]= 'X5700'                        # / Dichroic name   ---> CHANGE if using X6700!! 
    
    fits_image_hdu.header['OBJECT'] = rss.info["name"]    
    # fits_image_hdu.header["EXPOSED"] = rss.info["exptime"]
    # fits_image_hdu.header["ZDSTART"]= rss.original_koala_header["ZDSTART"]
    # fits_image_hdu.header["ZDEND"]= rss.original_koala_header["ZDEND"] 
                                       
    # fits_image_hdu.header['NAXIS']   =   2                              # / number of array dimensions                       
    # fits_image_hdu.header['NAXIS1']  =   rss.intensity.shape[0]                 
    # fits_image_hdu.header['NAXIS2']  =   rss.intensity.shape[1]                 

    fits_image_hdu.header["KOALAFOV"] = koala_info["KOALA_fov"]
    fits_image_hdu.header["SPAXSIZE"] = koala_info["spaxel_size"]
    fits_image_hdu.header['RACEN'] = koala_info["RA_centre_deg"] *np.pi/180     # / Field Centre RA (Radians) (from PTCS) 
    fits_image_hdu.header['DECCEN'] = koala_info["DEC_centre_deg"] *np.pi/180   # / Field Centre DEC (Radians) (from PTCS) 
    # fits_image_hdu.header['TEL_PA'] = rss.PA

    # fits_image_hdu.header["CTYPE2"] = 'Fibre number'          # / Label for axis 2  
    # fits_image_hdu.header["CUNIT2"] = ' '           # / Units for axis 2     
    # fits_image_hdu.header["CTYPE1"] = 'Wavelength'          # / Label for axis 2  
    # fits_image_hdu.header["CUNIT1"] = 'Angstroms'           # / Units for axis 2     
    # fits_image_hdu.header["CRVAL2"] = 5.000000000000E-01 # / Co-ordinate value of axis 2  
    # fits_image_hdu.header["CDELT2"] = 1.000000000000E+00 # / Co-ordinate increment along axis 2
    # fits_image_hdu.header["CRPIX2"] = 1.000000000000E+00 # / Reference pixel along axis 2 

    fits_image_hdu.header["CRVAL1"] = rss.wavelength[0]
    fits_image_hdu.header["CDELT1"] = (rss.wavelength[-1]-rss.wavelength[0])/(len(rss.wavelength)-1)
    fits_image_hdu.header["CRPIX1"] = 1. 

    if description is None: description = koala_info["description"]
    if description is None: description = rss.info["name"]
    fits_image_hdu.header['DESCRIP'] = description

    # HISTORY
    # First, delete HISTORY in the current header, to have everything together
    del fits_image_hdu.header['HISTORY']
    fits_image_hdu.header['HISTORY'] = "-- PREVIOUS HISTORY:"
    # Add the HISTORY in the previous file
    for item in rss.koala.history:   # rss.koala.header["HISTORY"]:
        if item == "- Created fits file (this file):":
            fits_image_hdu.header['HISTORY'] = "- Created fits file:"
        elif item == "-- ADDED NEW HISTORY:":
            pass
        else:    
            fits_image_hdu.header['HISTORY'] = item        
    fits_image_hdu.header['HISTORY'] = "-- ADDED NEW HISTORY:"   
    
    # Now, add the new HISTORY
    history = koala_info["history"]
    
    fits_image_hdu.header['HISTORY'] = '-- RSS processing using PyKOALA '+ version
    #fits_image_hdu.header['HISTORY'] = 'Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany et al.'
    #fits_image_hdu.header['HISTORY'] =  version #'Version 0.10 - 12th February 2019'    
    now=datetime.datetime.now()
    
    fits_image_hdu.header['HISTORY'] = now.strftime("File created on %d %b %Y, %H:%M:%S using input file:")
    fits_image_hdu.header['DATE'] = now.strftime("%Y-%m-%dT%H:%M:%S") #'2002-09-16T18:52:44'   # /Date of FITS file creation
    #fits_image_hdu.header['HISTORY'] = 'using input file:'
    fits_image_hdu.header['HISTORY'] = koala_info["path_to_file"]

    for item in history:  fits_image_hdu.header['HISTORY'] = item

    fits_image_hdu.header['HISTORY'] = "- Created fits file (this file):"
    fits_image_hdu.header['HISTORY'] = fits_file   
    
    fits_image_hdu.header['FILE_IN'] = koala_info["path_to_file"]
    fits_image_hdu.header['FILE_OUT'] = fits_file
    
    # VARIANCE 
    variance = rss.variance               
    variance_hdu = fits.ImageHDU(variance)
    

    # HEADER 2  with the RA and DEC info!    
    header2_all_fibres = rss.koala.fibre_table.data
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
             
    col1 = fits.Column(name='Fibre', format='I', array=np.array(header2_new_fibre))
    col2 = fits.Column(name='Status', format='I', array=np.array(header2_good_fibre))
    col3 = fits.Column(name='Ones', format='I', array=np.array(header2_good_fibre))
    col4 = fits.Column(name='Wavelengths', format='I', array=np.array(header2_2048))
    col5 = fits.Column(name='Zeros', format='I', array=np.array(header2_0))
    col6 = fits.Column(name='Delta_RA', format='D', array=np.array(header2_delta_RA))
    col7 = fits.Column(name='Delta_Dec', format='D', array=np.array(header2_delta_DEC))
    col8 = fits.Column(name='Fibre_OLD', format='I', array=np.array(header2_original_fibre))
    
    cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8])
    fibre_table_hdu = fits.BinTableHDU.from_columns(cols)
    fibre_table_hdu.header = rss.koala.fibre_table.header
    
    fibre_table_hdu.header['RACEN'] = koala_info["RA_centre_deg"] *np.pi/180     # / Field Centre RA (Radians) (from PTCS) 
    fibre_table_hdu.header['DECCEN'] = koala_info["DEC_centre_deg"] *np.pi/180   # / Field Centre DEC (Radians) (from PTCS) 

    
    # Put everything together & write
    #hdu_list = fits.HDUList([fits_image_hdu, variance_hdu, header2_hdu]) 
    hdu_list = fits.HDUList([fits_image_hdu, variance_hdu, fibre_table_hdu]) #, header2_hdu]) 

    hdu_list.writeto(fits_file, overwrite=True) 

    if verbose: print('  '+text+' saved to file "'+fits_file+'"')  
# =============================================================================
# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/ + Ángel :-)
