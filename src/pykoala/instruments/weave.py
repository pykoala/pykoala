"""
This script contains the wrapper functions to build a PyKoala RSS object from WEAVE L1 data.
"""

# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
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
from pykoala import vprint
from pykoala.rss import RSS
from pykoala.data_container import DataContainerHistory

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

    log = DataContainerHistory()
    info = {}

    vprint(f'Targets in {filename}:')
    main_target = 'unknown'
    main_count = 0
    for name in np.unique(fibtable['TARGNAME']):
        count = np.count_nonzero(fibtable['TARGNAME'] == name)
        vprint(f' {name} ({count} fibres)')
        if count > main_count:
            main_target = name
    info['name'] = main_target  # Name of the object
    info['exptime'] = header['EXPTIME']  # Total rss exposure time (seconds)
    info['airmass'] = header['AIRMASS']  # Airmass
    info['fib_ra'] = fibtable['FIBRERA']
    info['fib_dec'] = fibtable['FIBREDEC']
    
    return RSS(intensity=intensity,
           wavelength=wavelength,
           variance=variance,
           log=log,
           #header=header,
           info=info
           )


# %%
# -----------------------------------------------------------------------------
#                                                    ... Paranoy@ Rulz! ;^D
# -----------------------------------------------------------------------------
