"""
This script contains the wrapper functions to build a PyKoala RSS object from KOALA (2dfdr-reduced) data.
"""

# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import os
import copy
from tqdm import tqdm
import sys
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits
from astropy.wcs import WCS
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala import vprint  # Template to create the info variable 
from pykoala.rss import RSS
from pykoala.ancillary import print_counter,interpolate_image_nonfinite
from pykoala.spectra.onedspec import fit_smooth_spectrum,trim_spectrum
from pykoala.corrections.throughput import Throughput,ThroughputCorrection
from pykoala.corrections.atmospheric_corrections import AtmosphericExtCorrection
from pykoala.corrections.sky import TelluricCorrection
from pykoala.corrections.wavelength import rss_valid_wave_range

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


def py_koala_header(header):
    """
    Copy 2dfdr headers values from extensions 0 and 2 needed for the initial
    header for PyKoala. (based in the header constructed in  save_rss_fits in
    koala.io)
    """

    # To fit actual PyKoala header format
    header.rename_keyword('CENRA', 'RACEN')
    header.rename_keyword('CENDEC', 'DECCEN')

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

def py_koala_fibre_table(fibre_table):
    """
    Generates the spaxels tables needed for PyKoala from the 2dfdr spaxels table.
    """
    # Filtering only selected (in use) fibres
    spaxels_table = fibre_table[fibre_table['SELECTED'] == 1]

    # Defining new arrays
    arr1 = np.arange(len(spaxels_table)) + 1  # +  for starting in 1
    arr2 = np.ones(len(spaxels_table))
    arr3 = np.ones(len(spaxels_table))
    arr4 = np.ones(len(spaxels_table)) * 2048
    arr5 = np.zeros(len(spaxels_table))
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

def read_rss(file_path,
             wcs,
             intensity_axis=0,
             variance_axis=1,
             bad_fibres_list=None,
             header=None,
             fibre_table=None,
             info=None
             ):
    """TODO."""
    if header is None:
        # Blank Astropy Header object for the RSS header
        # Example how to add header value at the end
        # blank_header.append(('DARKCORR', 'OMIT', 'Dark Image Subtraction'), end=True)
        header = fits.header.Header(cards=[], copy=False)

    file_name = os.path.basename(file_path)

    vprint(f"\n> Reading KOALA RSS file {file_name}")
    #  Open fits file. This assumes that RSS objects are written to file as .fits.
    with fits.open(file_path) as rss_fits:
        # Read intensity using rss_fits_file[0]
        all_intensities = np.array(rss_fits[intensity_axis].data,
                                   dtype=np.float32)
        intensity = np.delete(all_intensities, bad_fibres_list, 0)
        # Bad pixel verbose summary
        vprint(f"Number of fibres in this RSS ={len(all_intensities)}"
               + f"No. of good fibres = {len(intensity)}"
               + f"No. of bad fibres = {len(bad_fibres_list)}")
        if bad_fibres_list is not None:
            vprint(f"Bad fibres = {bad_fibres_list}")
        # Read errors if exist a dedicated axis
        if variance_axis is not None:
            all_variances = rss_fits[variance_axis].data
            variance = np.delete(all_variances, bad_fibres_list, 0)

        else:
            vprint("WARNING! Variance extension not found in fits file!")
            variance = np.full_like(intensity, fill_value=np.nan)

    # Create wavelength from wcs
    nrow, ncol = wcs.array_shape
    wavelength_index = np.arange(ncol)
    wavelength = wcs.dropaxis(1).wcs_pix2world(wavelength_index, 0)[0]
    # First Header value added by the PyKoala routine
    header.append(('DARKCORR', 'OMIT', 'Dark Image Subtraction'), end=True)

    rss = RSS(intensity=intensity,
               variance=variance,
               wavelength=wavelength,
               info=info)
    rss.history('read', ' '.join(['- RSS read from ', file_name]))
    return rss

def koala_rss(path_to_file, **kwargs):
    """
    A wrapper function that converts a file (not an RSS object) to a koala RSS object
    The paramaters used to build the RSS object e.g. bad spaxels, header etc all come from the original (non PyKoala) .fits file
    """
    verbose = kwargs.get('verbose', False)
    
    header = fits.getheader(path_to_file, 0) + fits.getheader(path_to_file, 2)
    koala_header = py_koala_header(header)
    # WCS
    koala_wcs = WCS(header)
    # Constructing Pykoala Spaxels table from 2dfdr spaxels table (data[2])
    fibre_table = fits.getdata(path_to_file, 2)
    koala_fibre_table = py_koala_fibre_table(fibre_table)

    # List of bad spaxels from 2dfdr spaxels table
    bad_fibres_list = (fibre_table['SPEC_ID'][fibre_table['SELECTED'] == 0] - 1).tolist()
    # -1 to start in 0 rather than in 1
    # Create the dictionary containing relevant information
    info = {}
    info['name'] = koala_header['OBJECT']
    info['exptime'] = koala_header['EXPOSED']
    info['fib_ra'] = np.rad2deg(koala_header['RACEN']) + koala_fibre_table.data['Delta_RA'] / 3600
    info['fib_dec'] = np.rad2deg(koala_header['DECCEN']) + koala_fibre_table.data['Delta_DEC'] / 3600
    info['airmass'] = airmass_from_header(koala_header)
    # Read RSS file into a PyKoala RSS object
    rss = read_rss(path_to_file, wcs=koala_wcs,
                   bad_fibres_list=bad_fibres_list,
                   intensity_axis=0,
                   variance_axis=1,
                   header=koala_header,
                   fibre_table=koala_fibre_table,
                   info=info,
                   )
    return rss


# # ----------------------------------------------------------------------------
# %% ===========================================================================
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# # ANGEL TASKS FROM HERE
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# %% ===========================================================================
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# # Import tasks
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
import glob

from pykoala.plotting.quick_plot import quick_plot
from pykoala.plotting.rss_plot import rss_image, rss_map


# =============================================================================
# Ignore warnings
# =============================================================================
import warnings
warnings.filterwarnings('ignore')
from astropy.utils.exceptions import AstropyWarning

# #-----------------------------------------------------------------------------
# %% ===========================================================================
# #-----------------------------------------------------------------------------

# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# # General tasks
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# %% ===========================================================================
# #-----------------------------------------------------------------------------
def list_koala_fits_files_in_folder(path, verbose = True, use2=True, use3=False, ignore_offsets=True, 
                              skyflat_names=None, ignore_list=None, return_list=False):  
    """
    This task just reads the fits files in a folder and prints the KOALA-related fits files.
    
    The files are shown organised by name (in header). Exposition times are also given.
    
    Option of returning the list if return_list == True.

    Parameters
    ----------
    path : string
        path to data
    verbose : Boolean, optional
        Print the list. The default is True.
    use2 : Boolean, optional
        If True, the SECOND word of the name in fit files will be used
    use3 : Boolean, optional
        If True, the THIRD word of the name in fit files will be used
    ignore_offsets : Boolean, optional
        If True it will show all the fits files with the same name but different offsets together. The default is True.
    skyflat_names : list of strings, optional
        List with the names of the skyflat in fits file
    ignore_list : list of strings, optional
        List of words to ignore. The default is None.
    return_list : Boolean, optional
        Return the list. The default is False.

    Raises
    ------
    NameError
        DESCRIPTION.

    Returns
    -------
    if return_list == True, it returns the list of files as: 
        list_of_objetos, list_of_files, list_of_exptimes, date, grating

    """
    
    #FIXME This task needs to be checked, it was used by Ángel old automatic script
    #      but it is still useful for easy printing what the user has in a particular folder.
    
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
                    vprint("  {:15s}  {}          {:.1f} s".format(list_of_objetos[i], list_of_files[i][0], list_of_exptimes[i][0]))
                else:
                    vprint("                   {}          {:.1f} s".format(list_of_files[i][j], list_of_exptimes[i][j]))
                        
        vprint("\n  They were obtained on {} using the grating {}".format(date, grating))

    if return_list: return list_of_objetos, list_of_files, list_of_exptimes, date, grating
# #-----------------------------------------------------------------------------
# %% ===========================================================================
# #-----------------------------------------------------------------------------

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
# =============================================================================
# %% ==========================================================================
# =============================================================================
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
# #-----------------------------------------------------------------------------
# %% ===========================================================================
# #-----------------------------------------------------------------------------
def koalaRSS(filename, 
             path=None, 
             rss_object_name = None, 
             plot_map= False, 
             **kwargs):
    """
    Task for creating a KOALA rss object including rss.koala.
    
    This uses koala_rss() wrapper, which reads rss using read_rss()
    
    Then, it adds .koala to the rss object building all the info, see class KOALA_INFO() for rss.
    
    It plots the image (if plot = True) and map (if plot_map = True and plot = True) if requested.
    
    Tasks / Objects that this task uses:
        - KOALA_INFO() : Object keeping the specific KOALA info
        - koala_rss() : Pablo's wrapper read rss for KOALA
        - rss_image() : plotting rss image
        - rss_valid_wave_range() : Computing the rss valid wavelength range (left and right edges for each fibre)
        - compute_integrated_fibre() : computes integrated fibre for a rss
        - get_rss_mask(): Mask for rss
        
    Dictionaries that this task uses:
        - koala_info_rss_dict
        
    Parameters
    ----------
    filename : string
        Name of the rss file.
    path : string, optional
        path to the rss file
    rss_object_name : string, optional
        name of the rss object
    plot_map : boolean, optional
        if True it will plot the rss map. The default is False.
    **kwargs : kwargs
        where we can find plot, verbose, warnings...

    Returns
    -------
    rss : object
        rss KOALA object
    """
    verbose = kwargs.get('verbose', False)
    _warnings_ = kwargs.get('warnings', False)   # We can't use this here, as using warnings.catch_warnings()
    plot =  kwargs.get('plot', False)

    
    if path is not None:
        path_to_file = os.path.join(path,filename)
    else:
        path_to_file = filename  
    
    if verbose: print('\n> Converting KOALA+AAOmega RSS file "'+path_to_file+'" to a koala RSS object...')
    
    # Read rss using standard pykoala  #TODO: Ángel: I think we should merge the 2 of them (koalaRSS and koala_rss)
    rss = koala_rss(path_to_file, verbose=False)   # Using verbose = False to dissable showing the log

    # Create rss.koala
    rss.koala = KOALA_INFO()

    # Now, we add to this rss the information for a koala file
    # Create the dictionary containing relevant information for KOALA
    koala_info = koala_info_rss_dict.copy()  # Avoid missing some key info
    
    koala_info["rss_object_name"] = rss_object_name    # Save the name of the rss object
    koala_info['path_to_file'] = path_to_file          # Save name of the .fits file
    koala_info['n_wave'] = len(rss.wavelength)         # Save n_wave
    koala_info['n_spectra'] = len(rss.intensity)       # Save n_spectra

    # Check that dimensions of fits match KOALA numbers
    if koala_info['n_wave'] != 2048 and koala_info['n_spectra'] != 986: #  1000:
        if _warnings_ or verbose:  print('\n  *** WARNING *** : These numbers are NOT the standard n_wave and n_spectra values for KOALA')
    
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
    
    # Check valid range (basic mask in PyKOALA)
    valid_wave_range_data = rss_valid_wave_range(rss)    
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
    rss.koala.header = koala_header                     # Saving header
    rss.koala.fibre_table = koala_fibre_table           # Saving original koala fibre table as needed later
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        rss.koala.wcs = WCS(header)   # Saving WCS, not sure if later needed, but for now it is there
    
    # Plot RSS image if requested  #!!!
    if plot: rss_image(rss, **kwargs)
    
    # Computing integrating time and plot map if requested (plot_map in **kwargs is used in compute_integrated_fibre):
    kwargs["plot"] = plot_map
    verbose = kwargs.get('verbose', False)
    kwargs["verbose"] = False
    rss=compute_integrated_fibre(rss, **kwargs)
    kwargs["verbose"] = verbose
     
    #get mask
    rss.mask = get_rss_mask(rss, **kwargs)
    
    # Printing the info is requested:
    if verbose:    
        print('  Found {} spectra with {} wavelengths'.format(koala_info['n_spectra'], koala_info['n_wave']),
                      'between {:.2f} and {:.2f} Angstroms.'.format(rss.wavelength[0], rss.wavelength[-1]))
        print('  This RSS file uses the',koala_info['aaomega_grating'],'grating in the',koala_info['aaomega_arm'],'AAOmega arm.')
        print('  The KOALA field of view is {}, with a spaxel size of {}" and PA = {:.1f}º.'.format(KOALA_fov, spaxel_size,koala_info['position_angle']))
    
        if koala_info['rss_object_name'] is not None: 
            print('  Name of the observation = "{}",   Name of this Python RSS object = "{}".'.format(rss.info['name'],koala_info['rss_object_name']))
        else:
            print('  Name of the observation = "{}".'.format(rss.info['name']))    
        
    if kwargs.get("description") is not None:
        rss.koala_info["description"] = kwargs.get("description")
        if verbose: print('  Description provided to this KOALA RSS object = "{}".'.format(koala_info['description']))
            
    return rss
# #-----------------------------------------------------------------------------
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
    
    if list_spectra is None: list_spectra = list(range(n_spectra))
    if valid_wave_min is None:  valid_wave_min = rss.koala.info["valid_wave_min"]        
    if valid_wave_max is None:  valid_wave_max = rss.koala.info["valid_wave_max"]

        
    if verbose: print("\n> Computing integrated fibre values in range [ {:.2f} , {:.2f} ] {}".format(valid_wave_min, valid_wave_max, text))


    region = np.where((wavelength > valid_wave_min
                       ) & (wavelength < valid_wave_max))[0]
    waves_in_region = len(region)

    integrated_fibre = np.nansum(rss.intensity[:, region], axis=1)
    integrated_fibre_variance = np.nansum(rss.variance[:, region], axis=1)
    negative_fibres = (np.where(integrated_fibre < 0)[0]).tolist()
    n_negative_fibres = len(negative_fibres)    # n_negative_fibres = len(integrated_fibre[integrated_fibre < 0])

    if verbose:
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
    if plot: rss_map(rss,  variable=integrated_fibre_plot, title=title , **kwargs)
    
    rss.koala.integrated_fibre = integrated_fibre
    rss.koala.integrated_fibre_variance = integrated_fibre_variance
    rss.koala.integrated_fibre_sorted = integrated_fibre_sorted
    rss.koala.negative_fibres = negative_fibres
    
    return rss
# #-----------------------------------------------------------------------------
# %% ===========================================================================
# #-----------------------------------------------------------------------------
def get_rss_mask(rss, **kwargs):   # THIS TASK SHOULD BE A CORRECTION CLASS, including apply() #TODO
    """
    Get easy mask for rss.    

    Parameters
    ----------
    rss : Object
        rss.
    **kwargs : kwargs
        where we can find verbose, warnings, plot...
        For this task, the option is make_zeros, if True the masked values will be 0 instead of nan.

    Returns
    -------
    Array
        Mask
    """
    make_zeros=kwargs.get('make_zeros', False)

    n_waves = len(rss.wavelength)
    n_fibres = len(rss.intensity)
    indeces_low = rss.koala.mask[0]
    indeces_high = rss.koala.mask[1] 
    if make_zeros:
        mask = [   [0 if j < indeces_low[fibre] or j > indeces_high[fibre] else 1 for j in range(n_waves)]     for fibre in range(n_fibres)]            
    else:
        mask = [   [np.nan if j < indeces_low[fibre] or j > indeces_high[fibre] else 1 for j in range(n_waves)]     for fibre in range(n_fibres)]

    return np.array(mask)
# #-----------------------------------------------------------------------------
# %% ===========================================================================
# #-----------------------------------------------------------------------------





# #-----------------------------------------------------------------------------
# %% ===========================================================================
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------    
# # PLOT tasks, originally in plotting.rss_plot
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# %% ===========================================================================
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------







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
                      wavelength_shift_correction = None, 
                      sky_lines_for_wavelength_shifts=None, 
                      sky_lines_file_for_wavelength_shifts = None, 
                      n_sky_lines_for_wavelength_shifts  = 3,
                      maxima_sigma_for_wavelength_shifts = 2.5, 
                      maxima_offset_for_wavelength_shifts = 1.5,
                      median_fibres_for_wavelength_shifts = 7, 
                      index_fit_for_wavelength_shifts = 2, 
                      kernel_fit_for_wavelength_shifts= None, 
                      clip_fit_for_wavelength_shifts =0.4,
                      fibres_to_plot_for_wavelength_shifts=None,
                      show_fibres_for_wavelength_shifts=None,
                      median_offset_per_skyline_weight = 1.,     # 1 is the BLUE line (median offset per skyline), 0 is the GREEN line (median of solutions), anything between [0,1] is a combination.
                      show_skylines_for_wavelength_shifts = None,
                      plot_wavelength_shift_correction_solution = None,
                      
                      correct_for_extinction=False, # ----------------- EXTINCTION  (X)
                      
                      apply_telluric_correction = False,    # ----------------- TELLURIC  (U)
                      telluric_correction=None, 
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
                      fix_edges = None,
                      fix_edges_wavelength_continuum = None,
                      fix_edges_index_fit=None, 
                      fix_edges_kernel_fit=None,
                      
                      scale_sky = None,
                      #sky_rss=[0], scale_sky_rss=0, scale_sky_1D=0.,
                      #maxima_sigma=3.,
                
                      #sky_lines_file=None, exclude_wlm=[[0, 0]], emission_line_file = None,
                      is_sky=False, win_sky=0, #auto_scale_sky=False, ranges_with_emission_lines=[0], cut_red_end=0,
                      
                      correct_negative_sky=False,           # ----------------- NEGATIVE SKY  (N)
                      min_percentile_for_negative_sky = 5,
                      kernel_for_negative_sky=21,
                      order_fit_for_negative_sky=7,  
                      clip_fit_for_negative_sky = 0.8,
                      individual_check_for_negative_sky=True, # NOT IMPLEMENTED YET #TODO IF NEEDED
                      use_fit_for_negative_sky=True,
                      check_only_sky_fibres = False,
                      force_sky_fibres_to_zero=False,
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
    """
    This is the most important task, as it calls the rests.
    
    It keeps almost all the parameters that are used in the different tasks. Their description are there, but it will be included here ASAP.
    
    """         
    if rss_clean:                        # Just read file if rss_clean = True
        apply_mask = False
        apply_throughput = False                             # T
        correct_ccd_defects = False                          # C
        fix_wavelengths = False                              # W
        correct_for_extinction = False                       # X
        apply_telluric_correction = False                    # T
        clean_5577 = False                                   # R
        sky_method = None                                    # S
        correct_negative_sky = False                         # N
        id_el = False                                        # E
        big_telluric_residua_correction = False              # R01
        telluric_residua_at_6860_correction = False          # R02
        clean_cosmics = False                                # R04
        correct_extreme_negatives = False                    # R08
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
        rss = throughput_corr.apply(rss) 
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
                                                show_fibres_for_wavelength_shifts=show_fibres_for_wavelength_shifts,
                                                show_skylines_for_wavelength_shifts = show_skylines_for_wavelength_shifts,
                                                plot_wavelength_shift_correction_solution = plot_wavelength_shift_correction_solution,
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
            if path is not None: telluric_correction = os.path.join(path,telluric_correction)
            if verbose: print(" - Reading telluric correction from file",telluric_correction)
            telluric_correction = TelluricCorrection(telluric_correction_file = telluric_correction)
        
            
        rss = telluric_correction.apply(rss, verbose=verbose)
        corrections_done.append("telluric_correction")
        
    if rss.wavelength[0] < 5577 and rss.wavelength[-1] > 5577 and clean_5577: # ----------------  (S)
        rss = clean_skyline(rss, skyline = 5577, **kwargs)
        corrections_done.append("clean_5577")
    
    if sky_method is not None:  # --------------------------------------------------------------  (S)
    
        #TODO: Perhaps this section should be a separated task...
    
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
                                              fix_edges_wavelength_continuum = fix_edges_wavelength_continuum,
                                              fix_edges_index_fit=fix_edges_index_fit, 
                                              fix_edges_kernel_fit=fix_edges_kernel_fit,
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
                                      min_percentile_for_negative_sky = min_percentile_for_negative_sky,
                                      individual_check_for_negative_sky = individual_check_for_negative_sky,
                                      kernel_for_negative_sky = kernel_for_negative_sky,
                                      order_fit_for_negative_sky = order_fit_for_negative_sky,
                                      clip_fit_for_negative_sky = clip_fit_for_negative_sky,
                                      use_fit_for_negative_sky = use_fit_for_negative_sky,
                                      check_only_sky_fibres = check_only_sky_fibres,
                                      force_sky_fibres_to_zero=force_sky_fibres_to_zero,
                                      sky_fibres= sky_fibres,
                                      show_fibres = show_fibres_for_negative_sky,
                                      plot_rss_map_for_negative_sky = plot_rss_map_for_negative_sky,
                                      **kwargs)  
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
            if rss.koala.continuum_model_after_sky_correction is not None:
                continuum_model_after_sky_correction = rss.koala.continuum_model_after_sky_correction
            else:
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
        rss =     clean_telluric_residuals(rss, continuum=continuum_model_after_sky_correction,    
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
        
        
    # Add corrections_done and history:    
    if rss.koala.corrections_done is None:
        rss.koala.corrections_done = corrections_done
    else:
        rss.koala.corrections_done.append(corrections_done)
    
    if rss.koala.info["history"] is None: rss.koala.info["history"] = []  # TODO: needs to do proper history 
    for item in corrections_done: 
        #print(item)
        rss.koala.info["history"].append(item)

    # Summary:
    
    if plot_final_rss is None and len(corrections_done) > 0 and plot is not False : plot_final_rss= True
    
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
    
    # if save_rss_to_fits_file is not None:
    #     if save_rss_to_fits_file == "auto": # These two options, "auto" and "clean", should go in task save_rss_to_fits_file
    #         save_rss_to_fits_file = name_keys(filename, path= path, 
    #                                           apply_throughput = apply_throughput,                                       # T
    #                                           correct_ccd_defects = correct_ccd_defects,                                 # C
    #                                           fix_wavelengths = fix_wavelengths,                                         # W        
    #                                           correct_for_extinction = correct_for_extinction,                           # X
    #                                           apply_telluric_correction = apply_telluric_correction,                     # T
    #                                           sky_method = sky_method,                                                   # S
    #                                           correct_negative_sky = correct_negative_sky,                               # N
    #                                           id_el = id_el,                                                             # E
    #                                           big_telluric_residua_correction = big_telluric_residua_correction,         # R01
    #                                           telluric_residua_at_6860_correction = telluric_residua_at_6860_correction, # R02
    #                                           clean_cosmics = clean_cosmics,                                             # R04
    #                                           correct_extreme_negatives = correct_extreme_negatives)                     # R08 
        
    #     if save_rss_to_fits_file == "clean": save_rss_to_fits_file = filename[:-5]+"_clean.fits"
            
    #     koala_rss_to_fits(rss, fits_file=save_rss_to_fits_file, path = path, verbose=verbose)
    
    return rss



def process_n_koala_rss_files(filename_list = None,
                              path = None,
                              rss_list = None,
                              rss_object_name_list = None,
                              save_rss_to_fits_file_list = None,
                              # more things to add, e.g., sky.. #TODO
                              **kwargs):
    """
    This task process several koala rss files.
    
    As **kwargs, any parameter in process_koala_rss().
    
    Note that this task will use the same parameters for ALL rss files (we will add more options, e.g., adding different sky spectra to used in each file, soon)

    Parameters
    ----------
    filename_list : list of strings, optional
        List with the names of the fits files to process
    path : string or list of strings, optional
        Path to the fits files. 
        If only 1 string is given, it assumes all files listed in filename_list are in the same folder
    rss_list : list of objects, optional
        lift of rss objects
    rss_object_name_list : list of strings, optional
        list with the names of the rss objects
    save_rss_to_fits_file_list : list of strings, optional
        if provided, the processed rss files will be saved in these files
        it will use path if provided

    Raises
    ------
    RuntimeError
        If not filename_list or rss_list is provided.

    Returns
    -------
    processed_rss_files : list of rss objects
        list with the processed rss objects.

    """
    verbose = kwargs.get('verbose', False)
    
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


def get_throughput_2D(file_skyflat= None,
                      path=None, #instrument=None,
                      rss = None,
                      rss_object_name = None,
                      mask = None,
                      correct_ccd_defects=None, 
                      kernel_throughput=None,
                      index_fit_throughput = None,
                      throughput_2D_file=None,  
                      also_return_skyflat=True,
                      plot_final_rss = False,
                      **kwargs ):   
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
    rss: RSS object
        object with the skyflat/domeflat 
    rss_object_name : string, optional
        name of the rss object
    mask: list of arrays, optional
        if a mask is given, apply the mask. If not using KOALA mask in skyflat.koala.mask    
    correct_ccd_deffects: boolean, optional
        If True, it corrects for ccd defects when reading the skyflat fits file. Default is True
    kernel_throughput: odd integer 
        If provided, the 2D throughput will be smoothed with a this kernel.
    index_fit_throughput: integer
        index of the polynomium to fit each spectrum. If None it uses index_fit_throughput = 11
    throughput_2D_file: string
        the name of the fits file to be created with the output throughput 2D
    also_return_skyflat: boolean
        If True it also returns the skyflat (default is True)
    plot_final_rss: boolean
        If True plot the final rss (default is False)

    **kwargs : kwargs
        where we can find verbose, warnings, plot, verbose_counter, Jupyter...

    PyKOALA tasks that this uses:
    ----------------------------- 
        - process_koala_rss()
        - rss_image()
        - onedspec.fit_smooth_spectrum()
        - quick_plot()

    Returns
    -------
    if also_return_skyflat: 
        throughput, skyflat
    else:
        throughput
    """
    verbose = kwargs.get('verbose', False)
    verbose_counter = kwargs.get('verbose_counter', verbose)
    Jupyter = kwargs.get('Jupyter', True)
    #warnings = kwargs.get('warnings', verbose)
    plot =  kwargs.get('plot', False)
    if plot_final_rss is False and plot is True: plot_final_rss = True

    if verbose: print("\n> Reading a COMBINED skyflat / domeflat to get the 2D throughput...")
    
    if rss_object_name is None: rss_object_name = "skyflat"
    
    if correct_ccd_defects is None:
        if kernel_throughput is not None:
            correct_ccd_defects = False
        else:
            correct_ccd_defects = True
    
    skyflat =  process_koala_rss(rss_object_name=rss_object_name,
                                 rss=rss,
                                 filename=file_skyflat, 
                                 path = path,
                                 correct_ccd_defects = correct_ccd_defects, 
                                 apply_mask=True,
                                 plot_final_rss=plot_final_rss,   
                                 **kwargs )
                                 
    throughput_2D_ = np.zeros_like(skyflat.intensity)
    #throughput_2D_variance_ = np.zeros_like(skyflat.variance)      #TODO: Do we need to consider this?

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
        if verbose: print("\n  - Applying smooth with kernel =", kernel_throughput," and using to fit a polynomium of degree =",index_fit_throughput,"..." )
        if verbose_counter:  
            if Jupyter:
                pbar = tqdm(total=n_spectra, file=sys.stdout)
            else:
                pbar = None
            next_output = print_counter(stage=1, Jupyter = Jupyter, pbar = pbar)

        throughput_2D = np.zeros_like(throughput_2D_)
        # throughput_2D_variance = np.zeros_like(throughput_2D_variance_)
        
        for fibre in range(n_spectra):
            if verbose_counter:  
                if fibre >= next_output :  next_output = print_counter(stage=2, 
                                                                       iteration = fibre, 
                                                                       total = n_spectra,
                                                                       Jupyter = Jupyter,
                                                                       pbar = pbar)
                                                                     

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

        
        if verbose_counter: 
            print_counter(stage=3, Jupyter = Jupyter, pbar = pbar, total = n_spectra,
                          next_output = next_output )
        elif verbose: 
            print("    Process completed!\n")
        if verbose or verbose_counter: print(" ")
        
        throughput_2D = throughput_2D * skyflat.mask 
        # throughput_2D_variance = throughput_2D_variance * skyflat.mask 
        #if plot: rss_image(skyflat, image=throughput_2D, chigh=1.1, clow=0.9, cmap="binary_r",  title=" 2D throughput AFTER SMOOTHING AND FITTING A POLYNOMIUM")
        #skyflat.history.append('- Throughput 2D smoothed with kernel ' + str(kernel_throughput))

    else:
        throughput_2D = throughput_2D_
        # throughput_2D_variance = throughput_2D_variance_

    if plot:
        x = np.arange(n_spectra)
        median_throughput = np.nanmedian(throughput_2D, axis=1)
        quick_plot(x, median_throughput, ymin=0.2, ymax=1.2, hlines=[1, 0.9, 1.1],
                  ptitle="Median value of the 2D throughput per fibre", xlabel="Fibre")
        rss_image(skyflat, image=throughput_2D, cmap="binary_r",
                          title="\n ---- 2D throughput ----")

    skyflat_corrected = skyflat.intensity / throughput_2D
    if plot: rss_image(skyflat, image=skyflat_corrected, title="\n Skyflat CORRECTED for 2D throughput")
    
    # Create object with the Throughput correction
    throughput = Throughput()
    throughput.throughput_data = throughput_2D
    #throughput.throughput.throughput_error = throughput_2D_variance
    throughput.throughput_error = throughput_2D**0.5
    
    # Save fits file with the Throughput correcion if requested
    if throughput_2D_file is not None:
        if path is not None: throughput_2D_file = os.path.join(path,throughput_2D_file)
        throughput.tofits(throughput_2D_file)
        
    if verbose: print("\n> Throughput 2D obtained!")
    if also_return_skyflat:
        return throughput, skyflat
    else:
        return throughput

def obtain_telluric_correction(telluric_correction_file = None,
                               path_to_data = None,
                               star_list = None,
                               width_for_telluric = 30,
                               **kwargs):
    """
    This is a wrapper for quickly obtaining the telluric correction.

    Parameters
    ----------
    telluric_correction_file : string, optional
        File with the .txt/.dat or .fits (not implemented yet) info with the telluric correction, 
        to be read (if star_list is None) or saved (if star_list is provided).
    path_to_data : string, optional
        Path to the telluric correction file  The default is None.
    star_list : list of objects, optional
        List with the rss or cube objects to be used for creating the telluric correction. The default is None.
        If star_list is provided, the solution will be saved into telluric_correction_file 
    width_for_telluric : float, optional
        Width for telluric. The default is 30.
    **kwargs : kwargs
        where we can find verbose, warnings, plot, verbose_counter, Jupyter...

    PyKOALA tasks that this uses:
    ----------------------------- 
        - TelluricCorrection()
        - telluric_from_model()

    Raises
    ------
    RuntimeError:
        When no telluric_correction_file or star_list is provided.

    Returns
    -------
    telluric_correction : object
        Object with the telluric correction.
    """
    
    verbose = kwargs.get('verbose', False)
    #warnings = kwargs.get('warnings', verbose)
    plot = kwargs.get('plot', False)
    
    if telluric_correction_file is None and star_list is None: raise RuntimeError("No telluric_correction_file or star_list provided!!")
    
    
    if star_list is not None:    # If star_list given, use star_list and save it into telluric_correction_file
        # FIXME well.... so far this is done in rss
        # stars are rss objects
        
        n_stars = len(star_list)
        wavelength = star_list[0].wavelength

        # Check that all wavelengths are the same... #TODO
        # Check that they were taken the same night... #TODO
        
        if verbose: print("\n> Obtaining telluric correction using {} stars...".format(n_stars))
        telluric_correction_list = []
        for i in range(n_stars):
            if verbose: print("Star {} : {}".format(i+1, star_list[i].info["name"]))
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
            if path_to_data is not None: telluric_correction_file = os.path.join(path_to_data, telluric_correction_file)
            if verbose: print(" - Reading telluric correction from file",telluric_correction_file)
            telluric_correction = TelluricCorrection(telluric_correction_file = telluric_correction_file)
            
        ptitle = "Telluric correction from file {}".format(telluric_correction_file)

    if plot:
        quick_plot(wavelength,telluric_correction.telluric_correction, 
                  ptitle = ptitle,
                  ylabel = "Telluric correction",
                  **kwargs)
        
    return telluric_correction

def quick_find_brightest_line(rss, 
                              brightest_fibres_to_combine = None, 
                              lowest_fibres_to_combine = None, 
                              verbose= False):
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


def apply_mask_to_rss(rss,              # THIS TASK SHOULD BE A CORRECTION CLASS, including apply() #TODO
                      mask = None,   
                      make_zeros=False, 
                      **kwargs):    
    """
    Apply a mask to a RSS. 

    Parameters
    ----------
    rss : object
        rss object where the mask will be applied.
    mask : np.array, optional
        mask. The default is None.
    make_zeros : boolean, optional
        When applying mask, put 0s instead of nans. The default is False.
    **kwargs : kwargs
        where we can find verbose, verbose_counter, warnings, plot...

    Returns
    -------
    rss_out : 
        rss with the mask applied.

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


def clean_nan(rss, **kwargs): #TODO: THIS TASK SHOULD BE A CORRECTION CLASS, including apply()
    """
    Clean nans in rss using interpolation (this is needed to avoid 1D tasks failing )
    """
    verbose = kwargs.get('verbose', False)
    if verbose: print("> Applying nearest neighbour interpolation to remove NaN values ...")
    
    rss_out = copy.deepcopy(rss)
    rss_out.intensity = interpolate_image_nonfinite(rss.intensity)
    rss_out.variance = interpolate_image_nonfinite(rss.variance)    
    return rss_out
# Mr Krtxo \(ﾟ▽ﾟ)/ + Ángel :-)
