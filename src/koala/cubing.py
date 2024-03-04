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
# =============================================================================
# KOALA packages
# =============================================================================
from koala.data_container import DataContainer
from koala import __version__
from scipy.special import erf

# -------------------------------------------
# Fibre Interpolation and cube reconstruction
# -------------------------------------------

def gaussian_kernel(z, norm=False):
    """Cumulative gaussian kernel function."""
    c = 0.5 * (1 + erf(z / np.sqrt(2)))
    w = np.diff(c)
    if norm:
        w /= np.sum(w)
    return w

def cubic_kernel(z, norm=False):
    c = (3. * z - z ** 3 + 2.) / 4
    w = np.diff(c)
    if norm:
        w /= np.sum(w)
    return w

def interpolate_fibre(fib_spectra, fib_variance, cube, cube_var, cube_weight,
                      offset_cols, offset_rows, pixel_size, kernel_size_pixels,
                      adr_x=None, adr_y=None, adr_pixel_frac=0.05,
                      kernel_func=cubic_kernel):

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
    adr_pixel_frac: float, optional, default=0.05
        ADR Pixel fraction used to bin the spectral pixels. For each bin, the median ADR correction will be used to
        correct the range of wavelength.
    kernel_func: (method)
        1D kernel function to interpolate the data. Default is cubic kernel.

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
    # Set NaNs to 0 and discard pixels
    nan_pixels = ~np.isfinite(fib_spectra)
    fib_spectra[nan_pixels] = 0.

    pixel_weights = np.ones_like(fib_spectra)
    pixel_weights[nan_pixels] = 0.

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

        weight_x = kernel_func(x)
        weight_y = kernel_func(y)

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
                pixel_weights[wl_range: wl_range + spectral_window, np.newaxis, np.newaxis]
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
        print("[Cubing] Creating new datacube with dimensions: ", datacube.shape)
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
    print('[Cubing] Starting cubing process')
    # Use defined cube size to generated number of spaxel columns and rows
    n_cols = int(cube_size_arcsec[1] / pixel_size_arcsec)
    n_rows = int(cube_size_arcsec[0] / pixel_size_arcsec)
    # Create empty cubes for data, variance and weights - these will be filled and returned
    datacube = np.zeros((rss_set[0].wavelength.size, n_rows, n_cols),
                        dtype=np.float32)
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
        print("[Cubing] {}-th RSS fibre (transformed) offset with respect reference pos: "
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
            datacube=datacube, datacube_var=datacube_var, datacube_weight=datacube_weight,
            adr_x=adr_x, adr_y=adr_y)
        rss_mask[i] = datacube_weight - datacube_weight_before
        rss_mask[i] /= np.nanmax(rss_mask[i])
        del datacube_weight_before, copy_rss
    pixel_exptime = np.nansum(rss_mask * exposure_times[:, np.newaxis, np.newaxis, np.newaxis],
                           axis=0)
    datacube /= pixel_exptime
    datacube_var /= pixel_exptime**2
    # Create cube meta data
    info = dict(pixel_size_arcsec=pixel_size_arcsec, reference_coords=reference_coords, reference_pa=reference_pa,
                pixel_exptime=pixel_exptime, kernel_size_arcsec=kernel_size_arcsec, **cube_info)
    cube = Cube(rss_mask=rss_mask, intensity=datacube, variance=datacube_var,
                wavelength=rss.wavelength, info=info)
    return cube


# =============================================================================
# Cube class
# =============================================================================

class Cube(DataContainer):
    """This class represent a collection of Raw Stacked Spectra (RSS) interpolated over a 2D spatial grid.
    
    parent_rss
    rss_mask
    intensity
    variance
    intensity_corrected
    variance_corrected
    wavelength
    info

    
    
    """
    n_wavelength = None
    n_cols = None
    n_rows = None
    x_size_arcsec = None
    y_size_arcsec = None

    def __init__(self, parent_rss=None, rss_mask=None, intensity=None, variance=None,
                 intensity_corrected=None, variance_corrected=None, wavelength=None,
                 info=None, **kwargs):
        if intensity_corrected is None and intensity is not None:
            intensity_corrected = intensity.copy()
        if variance_corrected is None and variance is not None:
            variance_corrected = variance.copy()

        super(Cube, self).__init__(intensity=intensity,
                                   intensity_corrected=intensity_corrected,
                                   variance=variance,
                                   variance_corrected=variance_corrected,
                                   info=info, **kwargs)
        self.parent_rss = parent_rss
        self.rss_mask = rss_mask
        self.intensity = intensity
        self.variance = variance
        self.wavelength = wavelength
        # Cube information
        self.info = info

        if self.intensity is not None:
            self.n_wavelength, self.n_rows, self.n_cols = self.intensity.shape
        if self.info is not None and 'pixel_size_arcsec' in self.info.keys():
            self.x_size_arcsec = self.n_cols * self.info['pixel_size_arcsec']
            self.y_size_arcsec = self.n_rows * self.info['pixel_size_arcsec']
            # Store the spaxel offset position
            self.info['spax_ra_offset'], self.info['spax_dec_offset'] = (
                np.arange(-self.x_size_arcsec / 2 + self.info['pixel_size_arcsec'] / 2,
                        self.x_size_arcsec / 2 + self.info['pixel_size_arcsec'] / 2,
                        self.info['pixel_size_arcsec']),
                np.arange(-self.y_size_arcsec / 2 + self.info['pixel_size_arcsec'] / 2,
                        self.y_size_arcsec / 2 + self.info['pixel_size_arcsec'] / 2,
                        self.info['pixel_size_arcsec']))
    
    def get_centre_of_mass(self, wavelength_step=1, stat=np.median, power=1.0):
        """Compute the center of mass of the data cube."""
        x = np.arange(0, self.n_cols, 1)
        y = np.arange(0, self.n_rows, 1)
        x_com = np.empty(self.n_wavelength)
        y_com = np.empty(self.n_wavelength)
        for wave_range in range(0, self.n_wavelength, wavelength_step):
            x_com[wave_range: wave_range + wavelength_step] = np.nansum(
                self.intensity[wave_range: wave_range + wavelength_step]**power * x[np.newaxis, np.newaxis, :],
                axis=(1, 2)) / np.nansum(self.intensity[wave_range: wave_range + wavelength_step]**power, axis=(1, 2))
            x_com[wave_range: wave_range + wavelength_step] = stat(x_com[wave_range: wave_range + wavelength_step])
            y_com[wave_range: wave_range + wavelength_step] = np.nansum(
                self.intensity[wave_range: wave_range + wavelength_step]**power * y[np.newaxis, :, np.newaxis],
                axis=(1, 2)) / np.nansum(self.intensity[wave_range: wave_range + wavelength_step]**power, axis=(1, 2))
            y_com[wave_range: wave_range + wavelength_step] = stat(y_com[wave_range: wave_range + wavelength_step])
        return x_com, y_com

    def get_integrated_light_frac(self, frac=0.5):
        """Compute the integrated spectra that accounts for a given fraction of the total intensity."""
        collapsed_intensity = np.nansum(self.intensity_corrected, axis=0)
        sort_intensity = np.sort(collapsed_intensity, axis=(0, 1))
        # Sort from highes to lowest luminosity
        sort_intensity = np.flip(sort_intensity, axis=(0, 1))
        cumulative_intensity = np.cumsum(sort_intensity, axis=(0, 1))
        cumulative_intensity /= np.nanmax(cumulative_intensity)
        pos = np.searchsorted(cumulative_intensity, frac)
        return cumulative_intensity[pos]

    def get_white_image(self, wave_range=None, s_clip=3.0):
        """Create a white image."""
        if wave_range is not None and self.wavelength is not None:
            wave_mask = (self.wavelength >= wave_range[0]) & (self.wavelength <= wave_range[1])
        else:
            wave_mask = np.ones(self.intensity.shape[0], dtype=bool)
        
        if s_clip is not None:
            std_dev = np.nanstd(self.intensity_corrected[wave_mask], axis=0)
            median = np.nanmedian(self.intensity_corrected[wave_mask], axis=0)

            weights = (
                (self.intensity_corrected[wave_mask] <= median[np.newaxis] + s_clip * std_dev[np.newaxis])
                & (self.intensity_corrected[wave_mask] >= median[np.newaxis] - s_clip * std_dev[np.newaxis]))
        else:
            weights = np.ones_like(self.intensity_corrected[wave_mask])

        white_image = np.nansum(self.intensity_corrected * weights, axis=0) / np.nansum(weights, axis=0)
        return white_image

    def get_wcs(self):
        """TODO..."""
        # TODO!!!!
        w = WCS(naxis=3)
        w.wcs.crpix = [self.n_rows / 2, self.n_cols / 2,
                       self.wavelength.size / 2]
        w.wcs.cdelt = np.array([
            self.info['pixel_size_arcsec'] / 3600,
            self.info['pixel_size_arcsec'] / 3600,
            np.diff(self.wavelength).mean()])
        w.wcs.crval = [*self.info['reference_coords'],
                       np.interp(
                           w.wcs.crpix[2],
                           np.arange(0, self.wavelength.size), self.wavelength)
                       ]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]
        self.wcs = w
        return self.wcs

    def wcs_metadata(self):
        """Get the Cube WCS metadata"""
        return self.get_wcs().to_header()
    
    def to_fits(self, fname=None, primary_hdr_kw=None):
        """ TODO...
        include --
           parent RSS information
           filenames
           exptimes
           effective exptime?

        """
        if fname is None:
            fname = 'cube_{}.fits.gz'.format(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        if primary_hdr_kw is None:
            primary_hdr_kw = {}
        
        # Create the PrimaryHDU with WCS information 
        primary = fits.PrimaryHDU()
        for key, val in primary_hdr_kw.items():
            primary.header[key] = val
        
        
        # Include cubing information
        primary.header['CREATED'] = datetime.now().strftime(
            "%d_%m_%Y_%H_%M_%S"), "Cube creation date"
        primary.header['KERNSIZE'] = self.info["kernel_size_arcsec"], "arcsec"
        primary.header['pykoala'] = __version__, "PyKOALA version"

        # Fill the header with the log information
        primary.header = self.dump_log_in_header(primary.header)

        # Create a list of HDU
        hdu_list = [primary]
        # Change headers for variance and flux
        hdu_list.append(fits.ImageHDU(data=self.intensity, name='FLUX', header=self.wcs_metadata()))
        hdu_list.append(fits.ImageHDU(data=self.variance, name='VARIANCE', header=hdu_list[-1].header))
        # Save fits
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(fname, overwrite=True)
        hdul.close()
        print("[Saving] Cube saved at:\n {}".format(fname))

# Mr Krtxo \(ﾟ▽ﾟ)/
