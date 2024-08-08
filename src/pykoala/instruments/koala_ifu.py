"""
This script contains the wrapper functions to build a PyKoala RSS object from KOALA (2dfdr-reduced) data.
"""

# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import os
import copy
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits
from astropy.wcs import WCS
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala import vprint  # Template to create the info variable 
from pykoala.data_container import HistoryLog
from pykoala.rss import RSS

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
             instrument=None,
             verbose=False,
             log=None,
             header=None,
             fibre_table=None,
             info=None
             ):
    """TODO."""
    # Blank dictionary for the log
    if log is None:
        log = HistoryLog(verbose=verbose)
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
        log('read', ' '.join(['- RSS read from ', file_name]))
        # Read intensity using rss_fits_file[0]
        all_intensities = np.array(rss_fits[intensity_axis].data, dtype=np.float32)
        intensity = np.delete(all_intensities, bad_fibres_list, 0)
        # Bad pixel verbose summary
        vprint("\n  Number of spectra in this RSS =", len(all_intensities),
            ",  number of good spectra =", len(intensity),
            " ,  number of bad spectra =", len(bad_fibres_list),
            verbose=verbose)
        if bad_fibres_list is not None:
            vprint("  Bad fibres =", bad_fibres_list, verbose=verbose)

        # Read errors if exist a dedicated axis
        if variance_axis is not None:
            all_variances = rss_fits[variance_axis].data
            variance = np.delete(all_variances, bad_fibres_list, 0)

        else:
            vprint("\n  WARNING! Variance extension not found in fits file!", verbose=verbose)
            variance = np.full_like(intensity, fill_value=np.nan)

    # Create wavelength from wcs
    nrow, ncol = wcs.array_shape
    wavelength_index = np.arange(ncol)
    wavelength = wcs.dropaxis(1).wcs_pix2world(wavelength_index, 0)[0]
    # First Header value added by the PyKoala routine
    header.append(('DARKCORR', 'OMIT', 'Dark Image Subtraction'), end=True)

    # Blank mask (all 0, i.e. making nothing) of the same shape of the data
    mask = np.zeros_like(intensity)

    return RSS(intensity=intensity,
               variance=variance,
               wavelength=wavelength,
               info=info,
               log=log,
               )

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
                   verbose = verbose
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
                    print("  {:15s}  {}          {:.1f} s".format(list_of_objetos[i], list_of_files[i][0], list_of_exptimes[i][0]))
                else:
                    print("                   {}          {:.1f} s".format(list_of_files[i][j], list_of_exptimes[i][j]))
                        
        print("\n  They were obtained on {} using the grating {}".format(date, grating))

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







# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------    
# # This was in rss as it does not depend on instrument
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
def rss_valid_wave_range(rss, **kwargs):
    """
    Provides the list of wavelengths with good values (non-nan) in edges.
    
    BE SURE YOU HAVE NOT CLEANED CCD DEFECTS if you are running this!!!

    Parameters
    ----------
    rss : object
        rss object.
    **kwargs : kwargs
        where we can find plot, verbose, warnings...

    Returns
    -------
    A list of lists:
        [0][0]: mask_first_good_value_per_fibre
        [0][1]: mask_last_good_value_per_fibre
        [1][0]: mask_max
        [1][1]: mask_min
        
        [[mask_first_good_value_per_fibre, mask_last_good_value_per_fibre],
         [mask_max, mask_min],
         [w[mask_max], w[mask_min]], 
         mask_list_fibres_all_good_values] 

    """
    
    verbose = kwargs.get('verbose', False)
    warnings = kwargs.get('warnings', False)
    plot =  kwargs.get('plot', False)
    
    w = rss.wavelength
    n_spectra = len(rss.intensity)
    n_wave = len(rss.wavelength)
    x = list(range(n_spectra))
    
    #  Check if file has 0 or nans in edges
    if np.isnan(rss.intensity[0][-1]):
        no_nans = False
    else:
        no_nans = True
        if rss.intensity[0][-1] != 0:
            if verbose or warnings: print(
                "  Careful!!! pixel [0][-1], fibre = 0, wave = -1, that should be in the mask has a value that is not nan or 0 !!!!!", **kwargs)

    if verbose and plot : print("\n  - Checking the left edge of the ccd...")

    mask_first_good_value_per_fibre = []
    for fibre in range(n_spectra):
        found = 0
        j = 0
        while found < 1:
            if no_nans:
                if rss.intensity[fibre][j] == 0:
                    j = j + 1
                else:
                    mask_first_good_value_per_fibre.append(j)
                    found = 2
            else:
                if np.isnan(rss.intensity[fibre][j]):
                    j = j + 1
                else:
                    mask_first_good_value_per_fibre.append(j)
                    found = 2
            if j > 101:
                vprint((" No nan or 0 found in the fist 100 pixels, ", w[j], " for fibre", fibre), **kwargs)
                mask_first_good_value_per_fibre.append(j)
                found = 2

    mask_max = np.nanmax(mask_first_good_value_per_fibre)
    if plot:        
        quick_plot(x, mask_first_good_value_per_fibre, ymax=mask_max + 1, xlabel="Fibre",
                  ptitle="Left edge of the RSS", hlines=[mask_max], ylabel="First good pixel in RSS")

    # Right edge, important for RED
    if verbose and plot :  print("\n- Checking the right edge of the ccd...")
    mask_last_good_value_per_fibre = []
    mask_list_fibres_all_good_values = []

    for fibre in range(n_spectra):
        found = 0
        j = n_wave - 1
        while found < 1:
            if no_nans:
                if rss.intensity[fibre][j] == 0:
                    j = j - 1
                else:
                    mask_last_good_value_per_fibre.append(j)
                    if j == len(rss.intensity[0]) - 1:
                        mask_list_fibres_all_good_values.append(fibre)
                    found = 2
            else:
                if np.isnan(rss.intensity[fibre][j]):
                    j = j - 1
                else:
                    mask_last_good_value_per_fibre.append(j)
                    if j == len(rss.intensity[0]) - 1:
                        mask_list_fibres_all_good_values.append(fibre)
                    found = 2

            if j < n_wave - 1 - 300:
                if verbose: print((" No nan or 0 found in the last 300 pixels, ", w[j], " for fibre", fibre))
                mask_last_good_value_per_fibre.append(j)
                found = 2

    mask_min = np.nanmin(mask_last_good_value_per_fibre)
    if plot:
        ptitle = "Fibres with all good values in the right edge of the RSS file : " + str(
            len(mask_list_fibres_all_good_values))
        quick_plot(x, mask_last_good_value_per_fibre, ymin=np.nanmin(mask_min),
                  ymax=2050, hlines=[2047], xlabel="Fibre", ylabel="Last good pixel in RSS", ptitle=ptitle)

    if verbose: 
        print("\n  --> The valid range for this RSS is {:.2f} to {:.2f} ,  in pixels = [ {} ,{} ]".format(w[mask_max],
                                                                                                    w[mask_min],
                                                                                                    mask_max,
                                                                                                    mask_min))

    # rss.mask = [mask_first_good_value_per_fibre, mask_last_good_value_per_fibre]
    # rss.mask_good_index_range = [mask_max, mask_min]
    # rss.mask_good_wavelength_range = [w[mask_max], w[mask_min]]
    # rss.mask_list_fibres_all_good_values = mask_list_fibres_all_good_values

        print("\n> Returning [ [mask_first_good_value_per_fibre, mask_last_good_value_per_fibre], ")
        print(  "              [mask_max, mask_min], ")
        print(  "              [w[mask_max], w[mask_min]], ")
        print(  "              mask_list_fibres_all_good_values ] ")
    
    # if verbose:
        # print("\n> Mask stored in rss.mask !")
        # print("  self.mask[0] contains the left edge, self.mask[1] the right edge")
        # print("  Valid range of the data stored in self.mask_good_index_range (index)")
        # print("                             and in self.mask_good_wavelength  (wavelenghts)")
        # print("  Fibres with all good values (in right edge) in self.mask_list_fibres_all_good_values")
    
    #return [rss.mask,rss.mask_good_index_range,rss.mask_good_wavelength_range,rss.mask_list_fibres_all_good_values]
    return [[mask_first_good_value_per_fibre, mask_last_good_value_per_fibre],
            [mask_max, mask_min],
            [w[mask_max], w[mask_min]], 
            mask_list_fibres_all_good_values ] 
    # if include_history:
    #     self.history.append("- Mask obtainted using the RSS file, valid range of data:")
    #     self.history.append(
    #         "  " + str(w[mask_max]) + " to " + str(w[mask_min]) + ",  in pixels = [ " + str(
    #             mask_max) + " , " + str(mask_min) + " ]")
    #     # -----------------------------------------------------------------------------







# Mr Krtxo \(ﾟ▽ﾟ)/ + Ángel :-)
