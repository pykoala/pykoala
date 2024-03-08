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
from pykoala.ancillary import vprint, rss_info_template  # Template to create the info variable 
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
               info=info,
               log=log,
               )

def koala_rss(path_to_file):
    """
    A wrapper function that converts a file (not an RSS object) to a koala RSS object
    The paramaters used to build the RSS object e.g. bad spaxels, header etc all come from the original (non PyKoala) .fits file
    """
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
    info = rss_info_template.copy()  # Avoid missing some key
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
                   info=info
                   )
    return rss

# Mr Krtxo \(ﾟ▽ﾟ)/
