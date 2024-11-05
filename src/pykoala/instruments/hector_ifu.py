"""
This script contains the wrapper functions to build a PyKoala RSS object from Hector (2dfdr-reduced) data.
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
from pykoala.instruments.koala_ifu import read_rss


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


def hector_header(header):
    """
    Copy 2dfdr headers values from extensions 0 and 2 needed for the initial
    header for PyKoala. (based in the header constructed in  save_rss_fits in
    koala.io)
    """
    # To fit actual PyKoala header format
    header.rename_keyword('CENRA', 'RACEN')
    header.rename_keyword('CENDEC', 'DECCEN')
    return header

def set_fibre_table(fibre_table, bundle=''):
    """
    Generates the spaxels tables needed for PyKoala from the 2dfdr spaxels table.
    """
    # Filtering only selected (in use) fibres
    bundle_fibres = (fibre_table['PROBENAME'] == bundle) & (fibre_table['SELECTED'] == 1)
    bad_fibres = np.where(fibre_table['PROBENAME'] != bundle)[0]
    # Update the table
    fibre_table = fibre_table[bundle_fibres]

    # Defining new arrays
    arr1 = np.arange(len(fibre_table)) + 1  # +  for starting in 1
    arr2 = np.ones(len(fibre_table))
    arr3 = np.ones(len(fibre_table))
    arr4 = np.ones(len(fibre_table)) * 2048
    arr5 = np.zeros(len(fibre_table))
    arr6 = fibre_table['XPOS']
    arr7 = fibre_table['YPOS']
    arr8 = fibre_table['SPEC_ID']

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
    fibre_table = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5, col6, col7, col8])

    return fibre_table, bad_fibres


def hector_rss(path_to_file, bundle=''):
    """
    A wrapper function that converts a file (not an RSS object) to a koala RSS object
    The paramaters used to build the RSS object e.g. bad spaxels, header etc all come from the original (non PyKoala) .fits file
    """
    header = fits.getheader(path_to_file, 0) + fits.getheader(path_to_file, 2)
    header = hector_header(header)
    # WCS
    wcs = WCS(header)
    # Constructing Pykoala Spaxels table from 2dfdr spaxels table (data[2])
    fibre_table = fits.getdata(path_to_file, 2)
    koala_fibre_table, bad_fibres = set_fibre_table(fibre_table, bundle=bundle)

    # List of bad spaxels from 2dfdr spaxels table
    # -1 to start in 0 rather than in 1
    # Create the dictionary containing relevant information
    info = {}
    info['name'] = header['OBJECT']
    info['exptime'] = header['EXPOSED']
    info['obj_ra'] = None
    info['obj_dec'] = None
    info['cen_ra'] = np.rad2deg(header['RACEN'])
    info['cen_dec'] = np.rad2deg(header['DECCEN'])
    info['pos_angle'] = header.get('TEL_PA', 0)
    info['fib_ra'] = koala_fibre_table.data['Delta_RA']
    info['fib_dec'] = koala_fibre_table.data['Delta_DEC']
    info['airmass'] = airmass_from_header(header)
    # Read RSS file into a PyKoala RSS object
    rss = read_rss(path_to_file, wcs=wcs,
                   bad_fibres_list=bad_fibres,
                   header=header,
                   fibre_table=koala_fibre_table,
                   info=info,
                   )
    return rss

# Mr Krtxo \(ﾟ▽ﾟ)/
