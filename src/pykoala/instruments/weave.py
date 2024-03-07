"""
This script contains the wrapper functions to build a PyKoala RSS object from WEAVE L1 data.
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
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala.ancillary import vprint, rss_info_template  # Template to create the info variable 
from pykoala.rss import RSS

def weave_rss(filename):
    '''Read a WEAVE "single exposure" file (i.e. row-stacked spectra for just one arm)'''

    with fits.open(filename) as hdu:
        header = hdu[0].header + hdu[1].header
        wcs = WCS(header)
        pixels = np.arange(hdu[1].data.shape[1])
        wavelength = wcs.spectral.array_index_to_world(pixels)
        intensity = hdu[3].data
        variance = np.where(hdu[4].data > 0, 1/hdu[4].data, np.nan)
        fibtable = Table.read(hdu['FIBTABLE'])

    # This is ugly :^(
    log = {'read': {'comment': None, 'index': None},
           'mask from file': {'comment': None, 'index': 0},
           'blue edge': {'comment': None, 'index': 1},
           'red edge': {'comment': None, 'index': 2},
           'cosmic': {'comment': None, 'index': 3},
           'extreme negative': {'comment': None, 'index': 4},
           'wavelength fix': {'comment': None, 'index': None, 'sol': []}}
    
    info = {}

    print(f'Targets in {filename}:')
    main_target = 'unknown'
    main_count = 0
    for name in np.unique(fibtable['TARGNAME']):
        count = np.count_nonzero(fibtable['TARGNAME'] == name)
        print(f' {name} ({count} fibres)')
        if count > main_count:
            main_target = name
    info['name'] = main_target  # Name of the object
    target_fibres = np.where(fibtable['TARGNAME'] == name)

    info['exptime'] = header['EXPTIME']  # Total rss exposure time (seconds)
    info['pos_angle'] = header['ROTSKYPA']  # Instrument position angle
    info['airmass'] = header['AIRMASS']  # Airmass

    fibres = SkyCoord(fibtable['FIBRERA'], fibtable['FIBREDEC'], unit='deg')
    centre_ra = np.nanmean(fibres.ra.deg[target_fibres])
    centre_dec = np.nanmean(fibres.dec.deg[target_fibres])
    info['obj_ra'] = centre_ra
    info['obj_dec'] = centre_dec  # Celestial coordinates of the object (deg)
    info['cen_ra'] = centre_ra
    info['cen_dec'] = centre_dec  # Celestial coordinates of the pointing (deg)

    w = WCS(naxis=2)
    w.wcs.crpix = [1, 1]
    w.wcs.cdelt = np.array([1./3600, 1./3600])
    w.wcs.crval = [centre_ra, centre_dec]
    print('centre', w.wcs.crval)
    print('ra', np.min(fibtable['FIBRERA']), np.max(fibtable['FIBRERA']))
    print('dec', np.min(fibtable['FIBREDEC']), np.max(fibtable['FIBREDEC']))
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    fibres = w.world_to_pixel(fibres)
    info['fib_ra_offset'] = fibres[0]
    info['fib_dec_offset'] = fibres[1]  # Fibres' celestial offset
    
    return RSS(intensity=intensity,
           wavelength=wavelength,
           variance=variance,
           mask=np.zeros_like(intensity),
           intensity_corrected=intensity.copy(),
           variance_corrected=variance.copy(),
           log=log,
           header=header,
           fibre_table=None,
           info=info
           )


# %%
# -----------------------------------------------------------------------------
#                                                    ... Paranoy@ Rulz! ;^D
# -----------------------------------------------------------------------------
