# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import copy

from datetime import datetime
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala import ancillary
from pykoala.data_container import Cube
from pykoala.plotting.utils import qc_cubing,qc_fibres_on_fov
from pykoala import __version__, vprint
from scipy.special import erf

class CubeStacking:
    """Collection of cubing stacking methods.
    
    Each method takes as input arguments a collection of cubes and variances,
    either in the form of a list or as an array with the first dimension corresponding
    to each cube, and additional keyword arguments.

    """
    def sigma_clipping(cubes: np.ndarray, variances: np.ndarray, **kwargs):
        """Perform cube stacking using STD clipping.
        
        Parameters
        ----------
        - cubes: np.ndarray
            An array consisting of the collection of data to combine. The first
            dimension must correspond to the individual elements (e.g. datacubes)
            that will be combined. If the size of the first dimension is 1, it
            will return `cubes[0]` withouth applying any combination.
        - variances: np.ndarray
            Array of variances associated to cubes.
        - inv_var_weight: np.ndarray, optional
            An array of weights to apply during the stacking.
        
        Returns
        -------
        - stacked_cube: np.ndarray
            The result of stacking the data in cubes along axis 0.
        - stacked_variance: np.ndarray
            The result of stacking the variances along axis 0.
        """
        if cubes.shape[0] == 1:
            return cubes[0], variances[0]

        nsigma = kwargs.get("nsigma", 3.0)
        sigma = np.nanstd(cubes, axis=0)
        mean =np.nanmean(cubes, axis=0)
        good_pixel = np.abs((cubes - mean[np.newaxis]) / sigma[np.newaxis]
                            ) < nsigma
        if kwargs.get("inv_var_weight", False):
            w = 1 / variances
        else:
            w = np.ones_like(cubes)

        stacked_cube = np.nansum(cubes * w * good_pixel, axis=0) / np.nansum(
            w * good_pixel, axis=0)
        stacked_variance = np.nansum(variances, axis=0) / cubes.shape[0]**2
        return stacked_cube, stacked_variance

    def mad_clipping(cubes: np.ndarray, variances: np.ndarray, **kwargs):
        """Perform cube stacking using MAD clipping.
        
        Parameters
        ----------
        - cubes: np.ndarray
            An array consisting of the collection of data to combine. The first
            dimension must correspond to the individual elements (e.g. datacubes)
            that will be combined. If the size of the first dimension is 1, it
            will return `cubes[0]` withouth applying any combination.
        - variances: np.ndarray
            Array of variances associated to cubes.
        - inv_var_weight: np.ndarray, optional
            An array of weights to apply during the stacking.
        
        Returns
        -------
        - stacked_cube: np.ndarray
            The result of stacking the data in cubes along axis 0.
        - stacked_variance: np.ndarray
            The result of stacking the variances along axis 0.
        """
        if cubes.shape[0] == 1:
            return cubes[0], variances[0]

        nsigma = kwargs.get("nsigma", 3.0)
        sigma = ancillary.std_from_mad(cubes, axis=0)
        median =np.nanmedian(cubes, axis=0)
        good_pixel = np.abs(
            (cubes - median[np.newaxis]) / sigma[np.newaxis]) < nsigma

        if kwargs.get("inv_var_weight", True):
            w = 1 / variances
        else:
            w = np.ones_like(cubes)

        stacked_cube = np.nansum(cubes * w * good_pixel, axis=0) / np.nansum(
            w * good_pixel, axis=0)
        stacked_variance = np.nansum(variances, axis=0) / cubes.shape[0]**2
        return stacked_cube, stacked_variance

# -------------------------------------------
# Fibre Interpolation and cube reconstruction
# -------------------------------------------

class InterpolationKernel(object):
    def __init__(self, scale, *args, **kwargs):
        self.scale = scale
        self.truncation_radius = kwargs.get("truncation_radius", 1.0)
        self.pixel_scale_arcsec = kwargs.get("pixel_scale_arcsec")

    def truncation_normalization(self):
        pass
        
    def kernel(self, z):
        pass  

class ParabolicKernel(InterpolationKernel):
    def __init__(self, scale, *args, **kwargs):
        if 'truncation_radius' in kwargs:
            del kwargs['truncation_radius']
        super().__init__(scale, truncation_radius=1, *args, **kwargs)
        
    def cmf(self, z):
        z.clip(-1, 1, out=z)
        return (3. * z - z ** 3 + 2.) / 4

    def kernel_1D(self, x_edges):
        z_edges = (x_edges / self.scale).clip(-1, 1)
        cumulative = self.cmf(z_edges)
        weights = np.diff(cumulative)
        return weights
    
    def kernel_2D(self, x_edges, y_edges):
        z_yy, z_xx  = np.meshgrid(y_edges / self.scale, x_edges / self.scale)
        z_yy = z_yy.clip(-1., 1.)
        z_xx = z_xx.clip(-1., 1.)
        cum_k = ((3. * z_xx - z_xx ** 3 + 2.) / 4
                 * (3. * z_yy - z_yy ** 3 + 2.) / 4)
        weights = np.diff(cum_k, axis=0)
        weights = np.diff(weights, axis=1)
        return weights

class GaussianKernel(InterpolationKernel):
    def __init__(self, scale, truncation_radius, *args, **kwargs):
        super().__init__(scale, truncation_radius=truncation_radius,
                         *args, **kwargs)
        self.left_norm = 0.5 * (1 + erf(-self.truncation_radius / np.sqrt(2)))
        self.right_norm = 0.5 * (1 + erf(self.truncation_radius / np.sqrt(2)))

    def cmf(self, z):
        c = (0.5 * (1 + erf(z / np.sqrt(2))) - self.left_norm) / (self.right_norm - self.left_norm)
        return c.clip(0, 1)

    def kernel_1D(self, x_edges):
        cumulative = self.cmf(x_edges / self.scale)
        weights = np.diff(cumulative)
        return weights

    def kernel_2D(self, x_edges, y_edges):
        weights = (
            self.kernel_1D(x_edges)[:, np.newaxis]
            * self.kernel_1D(y_edges)[np.newaxis, :])
        return weights

class TopHatKernel(InterpolationKernel):
    def __init__(self, scale, *args, **kwargs):
        super().__init__(scale, *args, **kwargs)

    def cmf(self, z):
        c = 0.5 * (z / self.truncation_radius + 1)
        c[z > self.truncation_radius] = 1.0
        c[z < - self.truncation_radius] =0.0
        return c
        
    def kernel_1D(self, x_edges):
        z_edges = (x_edges / self.scale).clip(
            -self.truncation_radius, self.truncation_radius)
        cumulative = self.cmf(z_edges)
        weights = np.diff(cumulative)
        return weights

    def kernel_2D(self, x_edges, y_edges):
        weights = (
            self.kernel_1D(x_edges)[:, np.newaxis]
            * self.kernel_1D(y_edges)[np.newaxis, :])
        return weights

class DrizzlingKernel(TopHatKernel):
    @ancillary.remove_units_dec
    def __init__(self, scale, *args, **kwargs):
        super().__init__(scale=scale, **kwargs)
        self.truncation_radius =1

    def kernel_1D(self, *args):
        pass

    def kernel_2D(self, x_edges, y_edges):
        weights = np.zeros((x_edges.size - 1, y_edges.size - 1))
        # x == rows, y == columns
        pix_edge_x, pix_edge_y = np.meshgrid(x_edges[:-1], y_edges[:-1])
        for i, (px, py) in enumerate(
            zip(pix_edge_x.flatten(), pix_edge_y.flatten())):
            _, area_fraction = ancillary.pixel_in_circle(
                (px, py), pixel_size=1, circle_pos=(0, 0),
                circle_radius=self.scale / 2)
            weights[np.unravel_index(i, weights.shape)] = area_fraction
        return weights

# ------------------------------------------------------------------------------
# Fibre interpolation
# ------------------------------------------------------------------------------

def interpolate_fibre(fib_spectra, fib_variance, cube, cube_var, cube_weight,
                      pix_pos_cols, pix_pos_rows,
                      adr_cols=None, adr_rows=None, adr_pixel_frac=0.05,
                      kernel=DrizzlingKernel(scale=1.0), fibre_mask=None):

    """ Interpolates fibre spectra and variance to a 3D data cube.

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
    pix_pos_cols: int
        Fibre column pixel position (m).
    pix_pos_rows: int
        Fibre row pixel position (n).
    adr_cols: (k,) np.array(float), optional, default=None
        Atmospheric Differential Refraction (ADR) of each wavelength point along x (ra)-axis (m) expressed in pixels.
    adr_rows: (k,) np.array(float), optional, default=None
        Atmospheric Differential Refraction of each wavelength point along y (dec) -axis (n) expressed in pixels.
    adr_pixel_frac: float, optional, default=0.05
        ADR Pixel fraction used to bin the spectral pixels. For each bin, the median ADR correction will be used to
        correct the range of wavelength.
    kernel: pykoala.cubing.InterpolationKernel
        Kernel object to interpolate the data. Default is drizzling.

    Returns
    -------
    cube:
        Original datacube with the fibre data interpolated.
    cube_var:
        Original variance with the fibre data interpolated.
    cube_weight:
        Original datacube weights with the fibre data interpolated.
    """
    if adr_rows is None and adr_cols is None:
        adr_rows = np.zeros(fib_spectra.size)
        adr_cols = np.zeros(fib_spectra.size)
        spectral_window = fib_spectra.size
    else:
        # Estimate spectral window
        spectral_window = int(np.min(
            (adr_pixel_frac / np.abs(adr_cols[0] - adr_cols[-1]),
             adr_pixel_frac / np.abs(adr_rows[0] - adr_rows[-1]))
        ) * fib_spectra.size)

    # Set NaNs to 0 and discard pixels
    nan_pixels = ~np.isfinite(fib_spectra) | fibre_mask
    fib_spectra[nan_pixels] = 0. << fib_spectra.unit

    pixel_weights = np.ones(fib_spectra.size)
    pixel_weights[nan_pixels] = 0.

    # Loop over wavelength pixels
    for wl_range in range(0, fib_spectra.size, spectral_window):
        wl_slice = slice(wl_range, wl_range + spectral_window)

        # Kernel along columns direction (x, ra)
        kernel_centre_cols = pix_pos_cols - np.nanmedian(adr_cols[wl_slice])
        kernel_offset = kernel.scale * kernel.truncation_radius
        cols_min = max(int(kernel_centre_cols - kernel_offset) - 2, 0)
        cols_max = min(int(kernel_centre_cols + kernel_offset) + 2,
                       cube.shape[2] - 1)
        columns_slice = slice(cols_min, cols_max + 1)
        # Kernel along rows direction (y, dec)
        kernel_centre_rows = pix_pos_rows - np.nanmedian(adr_rows[wl_slice])
        rows_min = max(int(kernel_centre_rows - kernel_offset) - 2, 0)
        rows_max = min(int(kernel_centre_rows + kernel_offset) + 2,
                       cube.shape[1] - 1)
        rows_slice = slice(rows_min, rows_max + 1)

        if (cols_max <= cols_min) | (rows_max <= rows_min):
            continue
        column_edges = np.arange(cols_min - 0.5, cols_max + 1.5, 1.0)
        row_edges = np.arange(rows_min - 0.5, rows_max + 1.5, 1.0)
        pos_col_edges = (column_edges - kernel_centre_cols)
        pos_row_edges = (row_edges - kernel_centre_rows)
        w = kernel.kernel_2D(pos_row_edges, pos_col_edges)        
        w = w[np.newaxis]
        # Add spectra to cube
        cube[wl_slice, rows_slice, columns_slice] += (
            fib_spectra[wl_slice, np.newaxis, np.newaxis] * w)
        cube_var[wl_slice, rows_slice, columns_slice] += (
                fib_variance[wl_slice, np.newaxis, np.newaxis] * w**2)
        cube_weight[wl_slice, rows_slice, columns_slice] += (
                pixel_weights[wl_slice, np.newaxis, np.newaxis] * w)

    return cube, cube_var, cube_weight


def interpolate_rss(rss, wcs, kernel,
                    datacube=None, datacube_var=None, datacube_weight=None,
                    adr_ra_arcsec=None, adr_dec_arcsec=None, mask_flags=None,
                    qc_plots=False):

    """Perform fibre interpolation using a RSS into to a 3D datacube.

    Parameters
    ----------
    rss: (RSS)
        Target RSS to be interpolated.
    pixel_size_arcsec: (float, default=0.7)
        Cube square pixel size in arcseconds.
    kernel_size_arcsec: (float, default=2.0)
        Smoothing kernel scale in arcseconds.
    cube_size_arcsec: (2-element tuple)
        Size of the cube (RA, DEC) expressed in arcseconds.
        Note: the final shape of the cube will be (wave, dec, ra).
    datacube
    datacube_var
    datacube_weight
    adr_ra
    adr_dec

    Returns
    -------
    datacube
    datacube_var
    datacube_weight

    """
    # Initialise cube data containers (intensity, variance, fibre weights)
    if datacube is None:
        datacube = np.zeros(wcs.array_shape) << rss.intensity.unit
        vprint(f"[Cubing] Creating new datacube with dimensions: {wcs.array_shape}")
    if datacube_var is None:
        datacube_var = np.zeros_like(datacube) << rss.variance.unit
    if datacube_weight is None:
        datacube_weight = np.zeros_like(datacube)
    
    if adr_dec_arcsec is not None:
        adr_dec_pixel = (adr_dec_arcsec / kernel.pixel_scale_arcsec).decompose()
    else:
        adr_dec_pixel = None
    if adr_ra_arcsec is not None:
        adr_ra_pixel = (adr_ra_arcsec / kernel.pixel_scale_arcsec).decompose()
    else:
        adr_ra_pixel = None
    # Interpolate all RSS fibres
    mask = rss.mask.get_flag_map(mask_flags)
    # Obtain fibre position in the detector (center of pixel)
    fibre_pixel_pos_cols, fibre_pixel_pos_rows = wcs.celestial.world_to_pixel(
        SkyCoord(rss.info['fib_ra'], rss.info['fib_dec']))
    if qc_plots:
        qc_fig = qc_fibres_on_fov(
            datacube.shape[1:], fibre_pixel_pos_cols, fibre_pixel_pos_rows,
            fibre_diam=getattr(rss, 'fibre_diameter',
                               1.25 / kernel.pixel_scale_arcsec))
    else:
        qc_fig = None

    for fibre in range(rss.intensity.shape[0]):
        offset_ra_pix = fibre_pixel_pos_cols[fibre]
        offset_dec_pix = fibre_pixel_pos_rows[fibre]
        # Interpolate fibre to cube
        datacube, datacube_var, datacube_weight = interpolate_fibre(
            fib_spectra=rss.intensity[fibre].copy(),
            fib_variance=rss.variance[fibre].copy(),
            cube=datacube, cube_var=datacube_var, cube_weight=datacube_weight,
            pix_pos_cols=offset_ra_pix, pix_pos_rows=offset_dec_pix,
            kernel=kernel,
            adr_cols=adr_ra_pixel, adr_rows=adr_dec_pixel,
            fibre_mask=mask[fibre])
    return datacube, datacube_var, datacube_weight, qc_fig


def build_cube(rss_set, wcs=None, wcs_params=None, kernel=GaussianKernel,
               kernel_size_arcsec=2.0, kernel_truncation_radius=2.0,
               adr_set=None, mask_flags=None, qc_plots=False, **kwargs):
               
    """Create a Cube from a set of Raw Stacked Spectra (RSS).

    Parameters
    ----------
    rss_set: list of RSS
        List of Raw Stacked Spectra to interpolate.
    cube_size_arcsec: (2-element tuple) 
        Cube physical size in *arcseconds* in the form (RA, DEC).
    kernel_size_arcsec: float, default=1.1
        Interpolator kernel physical size in *arcseconds*.
    pixel_size_arcsec: float, default=0.7
        Cube pixel physical size in *arcseconds*.
    adr_set: (list, default=None)
        List containing the ADR correction for every RSS (it can contain None)
        in the form: [(ADR_ra_1, ADR_dec_1), (ADR_ra_2, ADR_dec_2), (None, None)]

    Returns
    -------
    cube: Cube
         Cube created by interpolating the set of RSS.
    qc_plot: dictionary, optional
        Only if qc_plots is `True`. It contains a dictionary of QC plots that
        includes the individual RSS coverage maps and the final weight/exposure
        maps.
    """
    vprint('[Cubing] Starting cubing process')
    if wcs is None and wcs_params is None:
        raise ValueError("User must provide either wcs or wcs_params values.")
    if wcs is None and wcs_params is not None:
        wcs = WCS(wcs_params)
    plots = {}
    # Initialise kernel
    kernel_size_arcsec = ancillary.check_unit(kernel_size_arcsec, u.arcsec)
    pixel_size = wcs.celestial.pixel_scale_matrix.diagonal().mean() << u.deg
    kernel_scale = (kernel_size_arcsec / pixel_size).decompose()

    vprint(
        f"[Cubing] Initialising {kernel.__name__}"
        + f"\n Scale: {kernel_scale:.1f} (pixels)"
        + f"\n Truncation radius: {kernel_truncation_radius:.1f}")
    kernel = kernel(pixel_scale_arcsec=pixel_size,
                    scale=kernel_scale,
                    truncation_radius=kernel_truncation_radius)
    
    # Create empty cubes for data, variance and weights - these will be filled and returned
    vprint(f"[Cubing] Initialising new datacube with dimensions: {wcs.array_shape}")
    # Create the output data units
    if u.second in rss_set[0].intensity.unit.bases:
        output_unit = rss_set[0].intensity.unit
    else:
        output_unit = rss_set[0].intensity.unit / u.second

    all_datacubes = np.zeros((len(rss_set), *wcs.array_shape)
                             ) << output_unit
    all_var = np.zeros(all_datacubes.shape) << output_unit**2
    all_w = np.zeros(all_datacubes.shape)
    all_exp = np.zeros(all_datacubes.shape) << u.second

    # "Empty" array that will be used to store exposure times
    exposure_times = np.zeros((len(rss_set))) << u.second

    # For each RSS two arrays containing the ADR over each axis might be provided
    # otherwise they will be set to None
    if adr_set is None:
        adr_set = [(None, None)] * len(rss_set)

    for i, rss in enumerate(rss_set):
        copy_rss = copy.deepcopy(rss)
        exposure_times[i] = copy_rss.info['exptime']

        # Interpolate RSS to data cube
        datacube_i, datacube_var_i, datacube_weight_i, rss_plots = interpolate_rss(
            copy_rss,
            wcs=wcs,
            kernel=kernel,
            datacube=np.zeros(wcs.array_shape) << rss.intensity.unit,
            datacube_var=np.zeros(wcs.array_shape) << rss.variance.unit,
            datacube_weight=np.zeros(wcs.array_shape),
            adr_ra_arcsec=adr_set[i][0], adr_dec_arcsec=adr_set[i][1],
            mask_flags=mask_flags,
            qc_plots=qc_plots)

        plots[f'rss_{i+1}'] = rss_plots
        if u.second in rss.intensity.unit.bases:
            all_datacubes[i] = datacube_i
            all_var[i] = datacube_var_i
        else:
            all_datacubes[i] = datacube_i / exposure_times[i]
            all_var[i] = datacube_var_i / exposure_times[i]**2

        all_w[i] = datacube_weight_i
        all_exp[i] = datacube_weight_i * exposure_times[i]

    # Stacking
    stacking_method = kwargs.get("stack_method", CubeStacking.mad_clipping)
    stacking_args = kwargs.get("stack_method_args", {})
    vprint(f"[Cubing] Stacking individual cubes using {stacking_method.__name__}")
    vprint(f"[Cubing] Additonal arguments for stacking: {stacking_args}")
    datacube, datacube_var = stacking_method(
        all_datacubes, all_var, **stacking_args)
    info = dict(kernel_size_arcsec=kernel_size_arcsec,
                **kwargs.get('cube_info', {}))
    # Create WCS information
    cube = Cube(intensity=datacube, variance=datacube_var, wcs=wcs, info=info)
    if qc_plots:
        # Compute the fibre coverage and exposure time maps
        plots[f'weights'] = qc_cubing(all_w, all_exp)
        return cube, plots
    return cube


def build_wcs(datacube_shape, reference_position, spatial_pix_size : u.Quantity,
              spectra_pix_size : u.Quantity, radesys='ICRS    ', equinox=2000.0):
    """Create a WCS using cubing information.
    
    Integer pixel values fall at the center of pixels. 

    Parameters
    ----------
    - datacube_shape: (tuple)
        Pixel shape of the datacube (wavelength, ra, dec).
    - reference_position: (tuple)
        Values corresponding to the origin of the wavelength axis, and sky position of the central pixel.
    - spatial_pix_size: (float)
        Pixel size along the spatial direction.
    - spectra_pix_size: (float)
        Pixel size along the spectral direction.
        
    """
    wcs_dict = {
    # Spatial dimensions
    'RADECSYS': radesys, 'EQUINOX': equinox,
    'CTYPE1': 'RA---TAN', 'CUNIT1': 'deg',
    'CDELT1': spatial_pix_size.to_value("deg"), 'CRPIX1': datacube_shape[1] / 2,
    'CRVAL1': reference_position[1].to_value("deg"), 'NAXIS1': datacube_shape[1],
    'CTYPE2': 'DEC--TAN', 'CUNIT2': 'deg',
    'CDELT2': spatial_pix_size.to_value("deg"), 'CRPIX2': datacube_shape[2] / 2,
    'CRVAL2': reference_position[2].to_value("deg"), 'NAXIS2': datacube_shape[2],
    # Spectral dimension
    'CTYPE3': 'WAVE    ', 'CUNIT3': 'angstrom',
    'CDELT3': spectra_pix_size.to_value("angstrom"), 'CRPIX3': 0,
    'CRVAL3': reference_position[0].to_value("angstrom"),
    'NAXIS3': datacube_shape[0]}
    wcs = WCS(wcs_dict)
    return wcs


def build_hdul(intensity, variance, primary_header_info=None, wcs=None):
    primary = fits.PrimaryHDU()
    primary.header['HISTORY'] = 'PyKOALA creation date: {}'.format(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    if primary_header_info is not None:
        for k, v in primary_header_info.items():
            primary.header[k] = v
    # TODO: fill with relevant information
    if wcs is not None:
        data_header = wcs.to_header()
    else:
        data_header = None
    hdul = fits.HDUList([
    primary, 
    fits.ImageHDU(name='INTENSITY', data=intensity, header=data_header),
    fits.ImageHDU(name='VARIANCE', data=variance, header=data_header)])

    return hdul

def make_white_image_from_array(data_array, wavelength=None, **args):
    """Create a white image from a 3D data array."""
    vprint(f"Creating a Cube of dimensions: {data_array.shape}")
    cube = Cube(intensity=data_array, wavelength=wavelength)
    return cube.get_white_image()


def make_dummy_cube_from_rss(rss, spa_pix_arcsec=0.5, kernel_pix_arcsec=1.0):
    """Create an empty datacube array from an input RSS."""
    spa_pix_arcsec = ancillary.check_unit(spa_pix_arcsec, u.arcsec)
    kernel_pix_arcsec = ancillary.check_unit(kernel_pix_arcsec, u.arcsec)

    min_ra, max_ra = np.nanmin(rss.info['fib_ra']), np.nanmax(rss.info['fib_ra'])
    min_dec, max_dec = np.nanmin(rss.info['fib_dec']), np.nanmax(rss.info['fib_dec'])
    ra_pixels = int(((max_ra - min_ra) / spa_pix_arcsec).decompose())
    dec_pixels = int(((max_dec - min_dec) / spa_pix_arcsec).decompose())
    datacube_shape = (rss.wavelength.size, ra_pixels, dec_pixels)
    ref_position = (rss.wavelength[0], (min_ra + max_ra) / 2, (min_dec + max_dec) / 2)

    wcs = build_wcs(datacube_shape=datacube_shape,
                    reference_position=ref_position,
                    spatial_pix_size=spa_pix_arcsec,
                    spectra_pix_size=rss.wavelength[1] - rss.wavelength[0],
                )
    cube = build_cube([rss], pixel_size_arcsec=spa_pix_arcsec, wcs=wcs,
                      kernel_size_arcsec=kernel_pix_arcsec)
    return cube

# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
