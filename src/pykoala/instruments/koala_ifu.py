"""
This script contains the wrapper functions to build a PyKoala RSS object from KOALA (2dfdr-reduced) data.
"""

# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import os
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala import vprint
from pykoala.data_container import RSS

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


def koala_header(header):
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
    koala_header = fits.header.Header(cards=cards, copy=False)
    koala_header = header
    return koala_header

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
        vprint(f"No. of fibres in this RSS ={len(all_intensities)}"
               + f"\nNo. of good fibres = {len(intensity)}"
               + f"\nNo. of bad fibres = {len(bad_fibres_list)}")
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
    #TODO : remove to_value once units are homogeneized
    wavelength = wcs.spectral.array_index_to_world(wavelength_index).to_value("angstrom")
    # First Header value added by the PyKoala routine
    rss = RSS(intensity=intensity << u.adu,
              variance=variance << u.adu**2,
              wavelength=wavelength << u.angstrom,
              info=info,
              header=header,
              fibre_diameter=1.25 << u.arcsec,
              wcs=wcs)

    rss.history('read', ' '.join(['- RSS read from ', file_name]))
    return rss

def koala_rss(path_to_file):
    """
    A wrapper function that converts a file (not an RSS object) to a koala RSS object
    The paramaters used to build the RSS object e.g. bad spaxels, header etc all come from the original (non PyKoala) .fits file
    """
    
    header = fits.getheader(path_to_file, 0) + fits.getheader(path_to_file, 2)
    header = koala_header(header)
    # WCS
    if "RADECSYS" in header:
        header["RADECSYSa"] = header["RADECSYS"]
        del header["RADECSYS"]
    koala_wcs = WCS(header)
    # Fix the WCS information such that koala_wcs.spectra exists
    koala_wcs.wcs.ctype[0] = 'WAVE    '
    # Constructing Pykoala Spaxels table from 2dfdr spaxels table (data[2])
    fibre_table = fits.getdata(path_to_file, 2)
    koala_fibre_table = py_koala_fibre_table(fibre_table)

    # List of bad spaxels from 2dfdr spaxels table
    bad_fibres_list = (fibre_table['SPEC_ID'][fibre_table['SELECTED'] == 0] - 1).tolist()
    # -1 to start in 0 rather than in 1
    # Create the dictionary containing relevant information
    info = {}
    info['name'] = header['OBJECT']
    info['exptime'] = header['EXPOSED'] << u.second
    info['fib_ra'] = (np.rad2deg(header['RACEN'])
                      + koala_fibre_table.data['Delta_RA'] / 3600) << u.deg
    info['fib_dec'] = (np.rad2deg(header['DECCEN'])
                       + koala_fibre_table.data['Delta_DEC'] / 3600) << u.deg
    info['airmass'] = airmass_from_header(header)
    # Read RSS file into a PyKoala RSS object
    rss = read_rss(path_to_file, wcs=koala_wcs,
                   bad_fibres_list=bad_fibres_list,
                   intensity_axis=0,
                   variance_axis=1,
                   header=header,
                   info=info,
                   )
    return rss


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
    header = koala_header(header)
    
    # Add history
    rss.koala.history = header["HISTORY"]
    
    koala_info['RA_centre_deg'] = header["RACEN"] *180/np.pi
    koala_info['DEC_centre_deg'] = header["DECCEN"] *180/np.pi
    koala_info['exptime'] = header["EXPOSED"]
    
    # Get AAOmega Arm & gratings
    if (header['SPECTID'] == "RD"):      
        koala_info['aaomega_arm'] = "red"
    if (header['SPECTID'] == "BL"):      
        koala_info['aaomega_arm'] = "blue"    
    koala_info['aaomega_grating'] = header['GRATID']
    koala_info['aaomega_dichroic'] = header["DICHROIC"]
    
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
    koala_info['position_angle'] = header['TEL_PA']
    
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
    rss.koala.header = header                     # Saving header
    rss.koala.fibre_table = koala_fibre_table           # Saving original koala fibre table as needed later

    if "RADECSYS" in header:
        header["RADECSYSa"] = header["RADECSYS"]
        del header["RADECSYS"]
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



# Mr Krtxo \(ﾟ▽ﾟ)/ + Ángel :-)
