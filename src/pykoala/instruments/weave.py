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
from pykoala.data_container import RSS, GaussianFibreLSFModel
from pykoala.data_container import DataContainerHistory

def weave_rss(filename, lsf_filename=None, lsf_wavelength_thinning=50):
    '''Read a WEAVE "single exposure" file (i.e. row-stacked spectra for just one arm)'''

    with fits.open(filename) as hdu:
        header = hdu[0].header + hdu[1].header
        wcs = WCS(header)
        pixels = np.arange(hdu[1].data.shape[1])
        wavelength = wcs.spectral.array_index_to_world(pixels)
        intensity = hdu[3].data << u.adu
        variance = np.where(hdu[4].data > 0, 1/hdu[4].data, np.nan) << u.adu**2
        sky = np.nanmedian(hdu[3].data - hdu[1].data, axis=0) << u.adu
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
    info['exptime'] = header['EXPTIME'] << u.second # Total rss exposure time (seconds)
    info['airmass'] = header['AIRMASS']  # Airmass
    info['fib_ra'] = fibtable['FIBRERA'] << u.deg
    info['fib_dec'] = fibtable['FIBREDEC'] << u.deg
    info['sky_fibres'] = np.where(fibtable['TARGUSE'] == "S")[0]
    info['sky_CASU'] = sky
    if header['OBSMODE'] == "LIFU":
        fibre_diameter = 2.6 << u.arcsec
    else:
        fibre_diameter = 1.3 << u.arcsec
    
    if lsf_filename is not None:
        nspec_list = np.arange(intensity.shape[0]) + 1
        lsf_model = casu_lsf(lsf_filename, nspec_list, wavelength[::lsf_wavelength_thinning])
    else:
        lsf_model = None
    return RSS(intensity=intensity,
               wavelength=wavelength,
               variance=variance,
               fibre_diameter = fibre_diameter,
               log=log,
               #header=header,
               info=info,
               wcs=wcs,
               lsf_model=lsf_model
           )

def _getfwhm(file, nspec_list, wl):
    """
    Evaluate FWHM for a list of nspec values across given wavelengths.
    Returns (fwhm_array, actual_nspec_array).
    fwhm_array shape: (N_nspec, N_wl) matching input order.
    """
    from scipy.interpolate import splev

    wl = np.asarray(wl)
    nspec_array = np.asarray(nspec_list).ravel()
    fwhm_list = []
    actual_nspec_list = []
    with fits.open(file) as hdul:
        data = hdul['LSF_splines'].data
        available_nspec = data['NSPEC']
        wl_min = hdul[0].header['MINWL']
        wl_max = hdul[0].header['MAXWL']
        for ns in nspec_array:
            idx = np.abs(available_nspec - ns).argmin()
            actual_nspec = int(available_nspec[idx])
            t = data['t'][idx]
            c = data['c'][idx]
            k = int(data['k'][idx])
            tck = (t, c, k)
            if actual_nspec != ns:
                print(f"Warning: nspec {ns} not found in LSF file. Using nearest: {actual_nspec}")
            below_range = wl < wl_min
            above_range = wl > wl_max
            within_range = ~below_range & ~above_range
            fwhm = np.zeros_like(wl, dtype=float)
            if np.any(within_range):
                fwhm[within_range] = splev(wl[within_range], tck)
            if np.any(below_range):
                edge_value_low = splev(wl_min, tck)
                fwhm[below_range] = edge_value_low
            if np.any(above_range):
                edge_value_high = splev(wl_max, tck)
                fwhm[above_range] = edge_value_high
            fwhm_list.append(fwhm)
            actual_nspec_list.append(actual_nspec)
    fwhm_array = np.stack(fwhm_list, axis=0) if fwhm_list else np.empty((0, len(wl)))
    actual_nspec_array = np.array(actual_nspec_list, dtype=int)
    return fwhm_array, actual_nspec_array

def casu_lsf(filename, nspec_list, wl):
    """
    Get FWHM values for given nspec_list and wavelengths from a CASU LSF file.
    Returns (fwhm_array, actual_nspec_array).
    fwhm_array shape: (N_nspec, N_wl) matching input order.
    """
    fwhm, nspec_array = _getfwhm(filename, nspec_list, wl.to_value(u.AA))
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    worst_resolution = sigma.max()
    best_resolution = sigma.min()
    delta_wl = best_resolution / 4
    n_pixels = int(np.ceil(6 * worst_resolution / delta_wl))
    lsf_wave_edges = np.linspace(-3*worst_resolution, 3*worst_resolution, n_pixels) << u.AA
    return GaussianFibreLSFModel(wl, sigma << u.AA, lsf_wave_edges)


# %%
# -----------------------------------------------------------------------------
#                                                    ... Paranoy@ Rulz! ;^D
# Mr Krtxo \(ﾟ▽ﾟ)/
# -----------------------------------------------------------------------------
