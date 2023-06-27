#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 02:09:09 2023

@author: pablo
"""

# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import copy

# =============================================================================
# KOALA packages
# =============================================================================
from koala.cubing import Cube
from scipy.special import erf

# -------------------------------------------
# Fibre Interpolation and cube reconstruction
# -------------------------------------------

def interpolate_fibre(fib_spectra, fib_variance, cube, cube_var, cube_weight,
                      offset_cols, offset_rows, pixel_size, kernel_size_pixels,
                      adr_x=None, adr_y=None, adr_pixel_frac=0.05):

    """ Interpolates fibre spectra and variance to data cube.

    Parameters
    ----------
    fib_spectra: (k,) np.array(float)
        Array containing the fibre spectra.
    fib_variance: (k,) np.array(float)
        Array containing the fibre variance.
    cube: (k, n, m) np.ndarray (float)
        Cube to interpolate fibre spectra.
    cube_var: (k, n, m) np.ndarray (float)
        Cube to interpolate fibre variance.
    cube_weight: (k, n, m) np.ndarray (float)
        Cube to store fibre spectral weights.
    offset_cols: int
        offset columns pixels (m) with respect to Cube.
    offset_rows: int
        offset rows pixels (n) with respect to Cube.
    pixel_size: float
        Cube pixel size in arcseconds.
    kernel_size_pixels: float
        Smoothing kernel size in pixels.
    adr_x: (k,) np.array(float), optional, default=None
        Atmospheric Differential Refraction (ADR) of each wavelength point along x-axis (m) expressed in pixels.
    adr_y: (k,) np.array(float), optional, default=None
        Atmospheric Differential Refraction of each wavelength point along y-axis (n) expressed in pixels.
    adr_pixel_frac: float, optional, default=0.1
        ADR Pixel fraction used to bin the spectral pixels. For each bin, the median ADR correction will be used to
        correct the range of wavelength.

    Returns
    -------
    cube:
        Original datacube with the fibre data interpolated.
    cube_var:
        Original variance with the fibre data interpolated.
    cube_weight:
        Original datacube weights with the fibre data interpolated.
    """
    if adr_x is None and adr_y is None:
        adr_x = np.zeros_like(fib_spectra)
        adr_y = np.zeros_like(fib_spectra)
        spectral_window = 0
    else:
        # Estimate spectral window
        spectral_window = int(np.min(
            (adr_pixel_frac / np.abs(adr_x[0] - adr_x[-1]),
             adr_pixel_frac / np.abs(adr_y[0] - adr_y[-1]))
        ) * fib_spectra.size)
    if spectral_window == 0:
        spectral_window = fib_spectra.size
    # Set NaNs to 0
    bad_wavelengths = ~np.isfinite(fib_spectra)
    fib_spectra[bad_wavelengths] = 0.
    ones = np.ones_like(fib_spectra)
    ones[bad_wavelengths] = 0.

    # Loop over wavelength pixels
    for wl_range in range(0, fib_spectra.size, spectral_window):
        # ADR correction for spectral window
        median_adr_x = np.nanmedian(adr_x[wl_range: wl_range + spectral_window])
        median_adr_y = np.nanmedian(adr_y[wl_range: wl_range + spectral_window])

        # Kernel for columns (x)
        kernel_centre_x = .5 * cube.shape[2] + offset_cols - median_adr_x / pixel_size
        x_min = max(int(kernel_centre_x - kernel_size_pixels), 0)
        x_max = min(int(kernel_centre_x + kernel_size_pixels) + 1, cube.shape[2] + 1)
        # Kernel for rows (y)
        n_points_x = x_max - x_min
        kernel_centre_y = .5 * cube.shape[1] + offset_rows - median_adr_y / pixel_size
        y_min = max(int(kernel_centre_y - kernel_size_pixels), 0)
        y_max = min(int(kernel_centre_y + kernel_size_pixels) + 1, cube.shape[1] + 1)
        n_points_y = y_max - y_min

        if (n_points_x < 1) | (n_points_y < 1):
            # print("OUT FOV")
            continue

        x = np.linspace(x_min - kernel_centre_x, x_max - kernel_centre_x,
                        n_points_x) / kernel_size_pixels
        y = np.linspace(y_min - kernel_centre_y, y_max - kernel_centre_y,
                        n_points_y) / kernel_size_pixels
        # Ensure weight normalization
        if x_min > 0:
            x[0] = -1.
        if x_max < cube.shape[2] + 1:
            x[-1] = 1.
        if y_min > 0:
            y[0] = -1.
        if y_max < cube.shape[1] + 1:
            y[-1] = 1.

        weight_x = kernel(x)
        weight_y = kernel(y)

        # Kernel weight matrix
        w = weight_y[np.newaxis, :, np.newaxis] * weight_x[np.newaxis, np.newaxis, :]
        # Add spectra to cube
        cube[wl_range: wl_range + spectral_window, y_min:y_max - 1, x_min:x_max - 1] += (
                fib_spectra[wl_range: wl_range + spectral_window, np.newaxis, np.newaxis]
                * w)
        cube_var[wl_range: wl_range + spectral_window, y_min:y_max - 1, x_min:x_max - 1] += (
                fib_variance[wl_range: wl_range + spectral_window, np.newaxis, np.newaxis]
                * w)
        cube_weight[wl_range: wl_range + spectral_window, y_min:y_max - 1, x_min:x_max - 1] += (
                ones[wl_range: wl_range + spectral_window, np.newaxis, np.newaxis]
                * w)
    return cube, cube_var, cube_weight


def interpolate_rss(rss, pixel_size_arcsec=0.7, kernel_size_arcsec=2.0,
                    cube_size_arcsec=None, datacube=None,
                    datacube_var=None, datacube_weight=None,
                    adr_x=None, adr_y=None):

    """ Converts a PyKoala RSS object to a datacube via interpolating fibers to cube

    Parameters
    ----------
    rss 
    pixel_size_arcsec
    kernel_size_arcsec
    cube_size_arcsec
    datacube
    datacube_var
    datacube_weight
    adr_x
    adr_y

    Returns
    -------
    datacube
    datacube_var
    datacube_weight

    """
    # Initialise cube data containers (flux, variance, fibre weights)
    if datacube is None:
        n_cols = int(cube_size_arcsec[1] / pixel_size_arcsec)
        n_rows = int(cube_size_arcsec[0] / pixel_size_arcsec)
        datacube = np.zeros((rss.wavelength.size, n_rows, n_cols))
        print("Creating new datacube with dimensions: ", datacube.shape)
    if datacube_var is None:
        datacube_var = np.zeros_like(datacube)
    if datacube_weight is None:
        datacube_weight = np.zeros_like(datacube)
    # Kernel pixel size for interpolation
    kernel_size_pixels = kernel_size_arcsec / pixel_size_arcsec
    # Loop over all RSS fibres
    for fibre in range(rss.intensity_corrected.shape[0]):
        offset_rows = rss.info['fib_dec_offset'][fibre] / pixel_size_arcsec  # pixel offset
        offset_cols = rss.info['fib_ra_offset'][fibre] / pixel_size_arcsec   # pixel offset
        # Interpolate fibre to cube
        datacube, datacube_var, datacube_weight = interpolate_fibre(
            fib_spectra=rss.intensity_corrected[fibre].copy(),
            fib_variance=rss.variance_corrected[fibre].copy(),
            cube=datacube, cube_var=datacube_var, cube_weight=datacube_weight,
            offset_cols=offset_cols, offset_rows=offset_rows, pixel_size=pixel_size_arcsec,
            kernel_size_pixels=kernel_size_pixels, adr_x=adr_x, adr_y=adr_y)
    return datacube, datacube_var, datacube_weight


def build_cube(rss_set, reference_coords, cube_size_arcsec,
               offset=(0, 0),
               reference_pa=0,
               fiber_size_arcsec=2.0,
               pixel_size_arcsec=0.7,
               adr_x_set=None, adr_y_set=None, **cube_info):
               
    """Create a Cube from a set of Raw Stacked Spectra (RSS).

    Parameters
    ----------
    rss_set: list of RSS
        List of Raw Stacked Spectra to interpolate.
    reference_coords: (2,) tuple
        Reference coordinates (RA, DEC) in *degrees* for aligning each RSS using
        RSS.info['cen_ra'], RSS.info['cen_dec'].
    offset: #TODO
    cube_size_arcsec: (2,) tuple
        Cube physical size in *arcseconds* in the form (DEC, RA).
    reference_pa: float
        Reference position angle in *degrees*.
    kernel_size_arcsec: float, default=1.1
        Interpolator kernel physical size in *arcseconds*.
    pixel_size_arcsec: float, default=0.7
        Cube pixel physical size in *arcseconds*.
    adr_x_set: # TODO
    adr_y_set: # TODO

    Returns
    -------
    cube: Cube
         Cube created by interpolating the set of RSS.
    """
    print('[Cubing] Starting cubing process')
    # Use defined cube size to generated number of spaxel columns and rows
    n_cols = int(cube_size_arcsec[1] / pixel_size_arcsec)
    n_rows = int(cube_size_arcsec[0] / pixel_size_arcsec)
    # Create empty cubes for data, variance and weights - these will be filled and returned
    datacube = np.zeros((rss_set[0].wavelength.size, n_rows, n_cols))
    datacube_var = np.zeros_like(datacube)
    datacube_weight = np.zeros_like(datacube)
    # Create an RSS mask that will contain the contribution of each RSS in the datacube
    rss_mask = np.zeros((len(rss_set), *datacube.shape))
    # "Empty" array that will be used to store exposure times
    exposure_times = np.zeros((len(rss_set)))

    
    # For each RSS two arrays containing the ADR over each axis might be provided
    # otherwise they will be set to None
    if adr_x_set is None:
        adr_x_set = [None] * len(rss_set)
    if adr_y_set is None:
        adr_y_set = [None] * len(rss_set)

    for i, rss in enumerate(rss_set):
        copy_rss = copy.deepcopy(rss)
        exposure_times[i] = copy_rss.info['exptime']
        # Offset between RSS WCS and reference frame in arcseconds
        # offset = ((copy_rss.info['cen_ra'] - reference_coords[0]) * 3600,
        #           (copy_rss.info['cen_dec'] - reference_coords[1]) * 3600)
        
        # Transform the coordinates of RSS
        cos_alpha = np.cos(np.deg2rad(copy_rss.info['pos_angle'] - reference_pa))
        sin_alpha = np.sin(np.deg2rad(copy_rss.info['pos_angle'] - reference_pa))
        
        offset = (offset[0] * cos_alpha - offset[1] * sin_alpha,
                  offset[0] * sin_alpha + offset[1] * cos_alpha)
        
        if adr_x_set[i] is not None and adr_y_set[i] is not None:
            adr_x = adr_x_set[i] * cos_alpha - adr_y_set[i] * sin_alpha
            adr_y = adr_x_set[i] * sin_alpha + adr_y_set[i] * cos_alpha
        else:
            adr_x = None
            adr_y = None
        print("{}-th RSS fibre (transformed) offset with respect reference pos: "
              .format(i+1), offset, ' arcsec')
        new_ra = (copy_rss.info['fib_ra_offset'] * cos_alpha - copy_rss.info['fib_dec_offset'] * sin_alpha
                  ) #- offset[0]
        new_dec = (copy_rss.info['fib_ra_offset'] * sin_alpha + copy_rss.info['fib_dec_offset'] * cos_alpha
                   ) #- offset[1]
        copy_rss.info['fib_ra_offset'] = new_ra
        copy_rss.info['fib_dec_offset'] = new_dec
        # Interpolate RSS to data cube
        datacube_weight_before = datacube_weight.copy()
        datacube, datacube_var, datacube_weight = interpolate_rss(
            copy_rss,
            pixel_size_arcsec=pixel_size_arcsec,
            kernel_size_arcsec=kernel_size_arcsec,
            cube_size_arcsec=cube_size_arcsec,
            datacube=datacube,
            datacube_var=datacube_var,
            datacube_weight=datacube_weight,
            adr_x=adr_x, adr_y=adr_y)
        rss_mask[i] = datacube_weight - datacube_weight_before
        rss_mask[i] /= np.nanmax(rss_mask[i])
    pixel_exptime = np.nansum(rss_mask * exposure_times[:, np.newaxis, np.newaxis, np.newaxis],
                           axis=0)
    datacube /= pixel_exptime
    datacube_var /= pixel_exptime**2
    # Create cube meta data
    info = dict(pixel_size_arcsec=pixel_size_arcsec, reference_coords=reference_coords, reference_pa=reference_pa,
                pixel_exptime=pixel_exptime, kernel_size_arcsec=kernel_size_arcsec, **cube_info)
    cube = Cube(parent_rss=rss_set, rss_mask=rss_mask, intensity=datacube, variance=datacube_var,
                wavelength=rss.wavelength, info=info)
    return cube

def build_cube_drizzling(rss_set, reference_coords,
                         cube_size_arcsec,
                         offset=(0, 0), reference_pa=0,
                         kernel_size_arcsec=2.0, pixel_size_arcsec=0.7,
                         adr_x_set=None, adr_y_set=None, **cube_info):
    """Create a Cube from a set of Raw Stacked Spectra (RSS).

    Parameters
    ----------
    rss_set: list of RSS
        List of Raw Stacked Spectra to interpolate.
    reference_coords: (2,) tuple
        Reference coordinates (RA, DEC) in *degrees* for aligning each RSS using
        RSS.info['cen_ra'], RSS.info['cen_dec'].
    offset: #TODO
    cube_size_arcsec: (2,) tuple
        Cube physical size in *arcseconds* in the form (DEC, RA).
    reference_pa: float
        Reference position angle in *degrees*.
    kernel_size_arcsec: float, default=1.1
        Interpolator kernel physical size in *arcseconds*.
    pixel_size_arcsec: float, default=0.7
        Cube pixel physical size in *arcseconds*.
    adr_x_set: # TODO
    adr_y_set: # TODO

    Returns
    -------
    cube: Cube
         Cube created by interpolating the set of RSS.
    """

# Mr Krtxo \(ﾟ▽ﾟ)/
