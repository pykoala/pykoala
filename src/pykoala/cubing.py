"""This module contains the main tools for building 3D datacubes by interpolating
RSS data.
"""
# =============================================================================
# Basics packages
# =============================================================================
from abc import abstractmethod
import numpy as np
from scipy.special import erf

# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

# =============================================================================
# KOALA packages
# =============================================================================
from pykoala import ancillary
from pykoala.data_container import Cube, RSS
from pykoala.plotting.utils import qc_cubing, qc_fibres_on_fov, qc_cube
from pykoala import vprint


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
        mean = np.nanmean(cubes, axis=0)

        good_pixel = np.abs((cubes - mean[np.newaxis]) / sigma[np.newaxis]) < nsigma
        w = np.where(good_pixel, 1.0, 0.0)
        if kwargs.get("inv_var_weight", True):
            w = np.where((variances > 0) & np.isfinite(variances) & good_pixel,
                         1 / variances, np.nan)
        norm = np.nansum(w, axis=0)
        illuminated_spx = norm > 0
        w = np.where(illuminated_spx, w / norm, np.nan)
        n_pix = np.sum(good_pixel, axis=0)
        stacked_cube = np.nansum(cubes * w, axis=0)
        # Do not include the pixels that are flagged as bad
        stacked_variance = np.full_like(stacked_cube, fill_value=np.nan)
        stacked_variance = np.where(n_pix > 0,
                                    np.nansum(variances * good_pixel, axis=0) / n_pix**2,
                                    np.nan)
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
        median = np.nanmedian(cubes, axis=0)
        # Spaxel weights
        good_pixel = np.abs((cubes - median[np.newaxis]) / sigma[np.newaxis]) < nsigma
        w = np.where(good_pixel, 1.0, 0.0)
        if kwargs.get("inv_var_weight", True):
            w = np.where((variances > 0) & np.isfinite(variances) & good_pixel,
                         1 / variances, np.nan)
        norm = np.nansum(w, axis=0)
        illuminated_spx = norm > 0
        w = np.where(illuminated_spx, w / norm, np.nan)
        n_pix = np.sum(good_pixel, axis=0)
        stacked_cube = np.nansum(cubes * w, axis=0)
        # Do not include the pixels that are flagged as bad
        stacked_variance = np.full_like(stacked_cube, fill_value=np.nan)
        stacked_variance = np.where(n_pix > 0,
                                    np.nansum(variances * good_pixel, axis=0) / n_pix**2,
                                    np.nan)
        return stacked_cube, stacked_variance


# -------------------------------------------
# Fibre Interpolation and cube reconstruction
# -------------------------------------------


class InterpolationKernel(object):
    """Interpolation Kernel.

    Description
    -----------
    A Kernel is a window function, :math:`K(u)`, used to perform the
    interpolation of individual fibre spectra at a given location in the sky
    :math:`(\alpha_0,\,\delta_0)` into a 3D grid.

    .. math::

    I(\alpha, \delta) = \int_{-\inf}^{\inf} I_{fib}(\alpha_0,\,\delta_0) \cdot K(\alpha - \alpha_0, \delta - \delta_0) d\alpha d\delta

    Different kernels have different domains, e.g. :math:`u\leq1` for
    Parabolic kernel, and therefore a scale parameter is sometimes required to
    renormalize the distance.

    The `truncation_radius`, expressed in units of `u`, is used with kernels
    whose domain extends all real numbers (e.g. Gaussian).
    """

    @property
    def scale(self):
        """Kernel scale size."""
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    @property
    def truncation_radius(self):
        """Maximum value of `u` beyond which the kernel is set to 0."""
        return self._truncation_radius

    @truncation_radius.setter
    def truncation_radius(self, value):
        self._truncation_radius = value

    @property
    def pixel_scale_arcsec(self):
        return self._pixel_scale_arcsec

    @pixel_scale_arcsec.setter
    def pixel_scale_arcsec(self, value):
        self._pixel_scale_arcsec = value

    def __init__(self, scale, *args, **kwargs):
        self.scale = scale
        self.truncation_radius = kwargs.get("truncation_radius", 1.0)
        self.pixel_scale_arcsec = kwargs.get("pixel_scale_arcsec")

    @abstractmethod
    def kernel_1D(self):
        pass

    @abstractmethod
    def kernel_2D(self):
        pass


class ParabolicKernel(InterpolationKernel):
    """Parabolic or Epanechnikov InterpolationKernel.

    Description
    -----------
    The parabolic kernel is defined as:

    .. math::

        K(u) = \frac{3}{4}(1 - u^2)

    With :math:`u` restricted to the range [-1, 1]. This model enforces
    `truncation_radius=1`.
    """

    def __init__(self, scale, *args, **kwargs):
        if "truncation_radius" in kwargs:
            del kwargs["truncation_radius"]
        super().__init__(scale, truncation_radius=1, *args, **kwargs)

    def cmf(self, u):
        """Cumulative mass distribution.

        Parameters
        ----------
        u : np.ndarray

        Returns
        -------
        cmf : np.ndarray
            The values of the cumulative distribution evaluated at the input
            values of u.
        """
        u_clip = u.clip(-1, 1)
        return (3.0 * u_clip - u_clip**3 + 2.0) / 4

    def kernel_1D(self, x_edges):
        """Compute the kernel weights associated to a 1D array of bins.

        Parameters
        ----------
        x_edges : np.ndarray
            Array of values defining the edges of the bins as
            `x_bin=[x_edges[i], x_edges[i+1]]`. The values must be increasing.

        Returns
        -------
        weights : np.ndarray
            Array of kernel weights within each bin. The size is `len(x_edges) - 1`.
        """
        assert (
            x_edges[1:] > x_edges[:-1]
        ).all(), "Input x_edges values must be increasing"
        u_edges = (x_edges / self.scale).clip(-1, 1)
        cumulative = self.cmf(u_edges)
        weights = np.diff(cumulative)
        return weights

    def kernel_2D(self, x_edges, y_edges):
        """Compute the map of kernel weights defined by input bin edges in x and y directions.

        Parameters
        ----------
        x_edges : np.ndarray
            Array defining the bin edges along the x axis.
        y_edges : np.ndarray
            Array defining the bin edges aling the y axis.

        Returns
        -------
        weights : np.ndarray
            Array of kernel weights within each bin. The size is
            `(len(y_edges) - 1, len(x_edges) - 1)`.
        """
        z_yy, z_xx = np.meshgrid(y_edges / self.scale, x_edges / self.scale)
        z_yy = z_yy.clip(-1.0, 1.0)
        z_xx = z_xx.clip(-1.0, 1.0)
        cum_k = (3.0 * z_xx - z_xx**3 + 2.0) / 4 * (3.0 * z_yy - z_yy**3 + 2.0) / 4
        weights = np.diff(cum_k, axis=0)
        weights = np.diff(weights, axis=1)
        return weights


class GaussianKernel(InterpolationKernel):
    """Gaussian InterpolationKernel."""

    def __init__(self, scale, truncation_radius, *args, **kwargs):
        super().__init__(scale, truncation_radius=truncation_radius, *args, **kwargs)
        self.left_norm = 0.5 * (1 + erf(-self.truncation_radius / np.sqrt(2)))
        self.right_norm = 0.5 * (1 + erf(self.truncation_radius / np.sqrt(2)))

    def cmf(self, z):
        c = (0.5 * (1 + erf(z / np.sqrt(2))) - self.left_norm) / (
            self.right_norm - self.left_norm
        )
        return c.clip(0, 1)

    def kernel_1D(self, x_edges):
        cumulative = self.cmf(x_edges / self.scale)
        weights = np.diff(cumulative)
        return weights

    def kernel_2D(self, x_edges, y_edges):
        weights = (
            self.kernel_1D(x_edges)[:, np.newaxis]
            * self.kernel_1D(y_edges)[np.newaxis, :]
        )
        return weights


class TopHatKernel(InterpolationKernel):
    def __init__(self, scale, *args, **kwargs):
        super().__init__(scale, *args, **kwargs)

    def cmf(self, z):
        c = 0.5 * (z / self.truncation_radius + 1)
        c[z > self.truncation_radius] = 1.0
        c[z < -self.truncation_radius] = 0.0
        return c

    def kernel_1D(self, x_edges):
        z_edges = (x_edges / self.scale).clip(
            -self.truncation_radius, self.truncation_radius
        )
        cumulative = self.cmf(z_edges)
        weights = np.diff(cumulative)
        return weights

    def kernel_2D(self, x_edges, y_edges):
        weights = (
            self.kernel_1D(x_edges)[:, np.newaxis]
            * self.kernel_1D(y_edges)[np.newaxis, :]
        )
        return weights


class DrizzlingKernel(TopHatKernel):
    """Drizzling InterpolationKernel."""

    @ancillary.remove_units_dec
    def __init__(self, scale, *args, **kwargs):
        super().__init__(scale=scale, **kwargs)
        self.truncation_radius = 1

    def kernel_1D(self, *args):
        raise NotImplementedError("kernel_1D has not been implemented in this class")

    def kernel_2D(self, x_edges, y_edges):
        weights = np.zeros((x_edges.size - 1, y_edges.size - 1))
        # x == rows, y == columns
        # Only the lower-left corner of every pixel is required.
        pix_edge_x, pix_edge_y = np.meshgrid(x_edges[:-1], y_edges[:-1])
        for i, pos in enumerate(zip(pix_edge_x.flatten(), pix_edge_y.flatten())):
            _, area_fraction = ancillary.pixel_in_circle(
                pos, pixel_size=1, circle_pos=(0, 0), circle_radius=self.scale / 2
            )
            weights[np.unravel_index(i, weights.shape)] = area_fraction
        return weights


# ------------------------------------------------------------------------------
# Fibre interpolation
# ------------------------------------------------------------------------------


def interpolate_fibre(
    fib_spectra,
    fib_variance,
    cube,
    cube_var,
    cube_weight,
    pix_pos_cols,
    pix_pos_rows,
    adr_cols=None,
    adr_rows=None,
    adr_pixel_frac=0.05,
    kernel=DrizzlingKernel(scale=1.0),
    fibre_mask=None,
):
    """Interpolates fibre spectra and variance to a 3D data cube.

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
        spectral_window = int(
            np.min(
                (
                    adr_pixel_frac / np.abs(adr_cols[0] - adr_cols[-1]),
                    adr_pixel_frac / np.abs(adr_rows[0] - adr_rows[-1]),
                )
            )
            * fib_spectra.size
        )

    # Set NaNs to 0 and discard pixels
    if fibre_mask is None:
        fibre_mask = np.zeros(fib_spectra.shape, dtype=bool)
    nan_pixels = ~np.isfinite(fib_spectra) | fibre_mask
    fib_spectra[nan_pixels] = 0.0 << fib_spectra.unit

    pixel_weights = np.ones(fib_spectra.size)
    pixel_weights[nan_pixels] = 0.0

    # Loop over wavelength pixels
    for wl_range in range(0, fib_spectra.size, spectral_window):
        wl_slice = slice(wl_range, wl_range + spectral_window)

        # Kernel along columns direction (x, ra)
        kernel_centre_cols = pix_pos_cols - np.nanmedian(adr_cols[wl_slice])
        kernel_offset = kernel.scale * kernel.truncation_radius
        cols_min = max(int(kernel_centre_cols - kernel_offset), 0)
        cols_max = min(int(kernel_centre_cols + kernel_offset), cube.shape[2] - 1)
        columns_slice = slice(cols_min, cols_max + 1)
        # Kernel along rows direction (y, dec)
        kernel_centre_rows = pix_pos_rows - np.nanmedian(adr_rows[wl_slice])
        rows_min = max(int(kernel_centre_rows - kernel_offset), 0)
        rows_max = min(int(kernel_centre_rows + kernel_offset), cube.shape[1] - 1)
        rows_slice = slice(rows_min, rows_max + 1)

        if (cols_max < cols_min) | (rows_max < rows_min):
            print("Col ", cols_max, cols_min)
            print("Row ", rows_max, rows_min)
            continue
        column_edges = np.arange(cols_min - 0.5, cols_max + 1.5, 1.0)
        row_edges = np.arange(rows_min - 0.5, rows_max + 1.5, 1.0)
        pos_col_edges = column_edges - kernel_centre_cols
        pos_row_edges = row_edges - kernel_centre_rows
        # Compute the kernel weight associated to each location
        w = kernel.kernel_2D(pos_row_edges, pos_col_edges)
        w = w[np.newaxis]
        # Add spectra to cube
        cube[wl_slice, rows_slice, columns_slice] = np.add(
            cube[wl_slice, rows_slice, columns_slice],
            fib_spectra[wl_slice, np.newaxis, np.newaxis] * w,
            where=np.isfinite(cube[wl_slice, rows_slice, columns_slice]))

        cube_var[wl_slice, rows_slice, columns_slice] = np.add(
            cube_var[wl_slice, rows_slice, columns_slice],
            fib_variance[wl_slice, np.newaxis, np.newaxis] * w**2,
            where=np.isfinite(cube_var[wl_slice, rows_slice, columns_slice]))

        cube_weight[wl_slice, rows_slice, columns_slice] = np.add(
            cube_weight[wl_slice, rows_slice, columns_slice],
            pixel_weights[wl_slice, np.newaxis, np.newaxis] * w,
            where=np.isfinite(cube_weight[wl_slice, rows_slice, columns_slice]))

    return cube, cube_var, cube_weight


def interpolate_rss(
    rss,
    wcs,
    kernel,
    datacube=None,
    datacube_var=None,
    datacube_weight=None,
    adr_ra_arcsec=None,
    adr_dec_arcsec=None,
    mask_flags=None,
    qc_plots=False,
):
    """Perform fibre interpolation using a RSS into to a 3D datacube.

    Parameters
    ----------
    rss : :class:`RSS`
        Target RSS to be interpolated.
    wcs : :class:`astropy.wcs.WCS`
        World Coordinate Transformation used to interpolate the RSS that defines
        the target 3D datacube.
    kernel : :class:`InterpolationKernel`
        Kernel to interpolate each fibre to the cube defined by the WCS
    datacube : u.Quantity, optional
        Array that stores the intensity associated to a datacube.
    datacube_var : u.Quantity, optional
        Array that stores the variance associated of a datacube.
    datacube_weights : np.ndarray, optional
        Array that stores the fibre weights of the datacube.
    adr_ra_arcsec : u.Quantity, optional
        Differential atmospheric refraction offset along the RA direction.
    adr_dec_arcsec : u.Quantity, optional
        Differential atmospheric refraction offset along the DEC direction.
    mask_flags : str or iterable, optional
        Flag names to be used for masking pixels
    qc_plots : bool, optional
        If `True`, produced a quality control plot showing the disposition of
        fibres across the defined FoV.

    Returns
    -------
    datacube : u.Quantity
    datacube_var : u.Quantity
    datacube_weight : u.Quantity
    qc_fig : plt.Figure or None
    """
    # Initialise cube data containers (intensity, variance, fibre weights)
    if datacube is None:
        datacube = np.zeros(wcs.array_shape) << rss.intensity.unit
        vprint(f"[Cubing] Creating new datacube with dimensions: {wcs.array_shape}")
    if datacube_var is None:
        datacube_var = np.zeros_like(datacube) << rss.variance.unit
    if datacube_weight is None:
        datacube_weight = np.zeros_like(datacube)
    # Obtain fibre position in the detector (center of pixel)
    fibre_pixel_pos_cols, fibre_pixel_pos_rows = wcs.celestial.world_to_pixel(
        SkyCoord(rss.info["fib_ra"], rss.info["fib_dec"])
    )
    # Wavelength array of target datacube
    cube_wavelength = wcs.spectral.array_index_to_world(
        np.arange(wcs.array_shape[0])
    ).to(rss.wavelength.unit)

    if adr_dec_arcsec is not None:
        adr_dec_pixel = (adr_dec_arcsec / kernel.pixel_scale_arcsec).decompose()
        if cube_wavelength.size != rss.wavelength.size or np.allclose(
            cube_wavelength, rss.wavelength
        ):
            adr_dec_pixel = np.interp(cube_wavelength, rss.wavelength, adr_dec_pixel)
    else:
        adr_dec_pixel = None
    if adr_ra_arcsec is not None:
        adr_ra_pixel = (adr_ra_arcsec / kernel.pixel_scale_arcsec).decompose()
        if cube_wavelength.size != rss.wavelength.size or np.allclose(
            cube_wavelength, rss.wavelength
        ):
            adr_ra_pixel = np.interp(cube_wavelength, rss.wavelength, adr_ra_pixel)
    else:
        adr_ra_pixel = None
    # Interpolate all RSS fibres
    if mask_flags is not None:
        mask = rss.mask.get_flag_map(mask_flags)
    else:
        mask = np.zeros(rss.intensity.shape, dtype=bool)

    if qc_plots:
        qc_fig = qc_fibres_on_fov(
            datacube.shape[1:],
            fibre_pixel_pos_cols,
            fibre_pixel_pos_rows,
            fibre_diam=getattr(rss, "fibre_diameter", 1.25 / kernel.pixel_scale_arcsec),
        )
    else:
        qc_fig = None

    for fibre in range(rss.intensity.shape[0]):
        # Spectra interpolation
        if cube_wavelength.size != rss.wavelength.size or np.allclose(
            cube_wavelength, rss.wavelength
        ):
            f_intensity = ancillary.flux_conserving_interpolation(
                cube_wavelength, rss.wavelength, rss.intensity[fibre]
            )
            f_variance = ancillary.flux_conserving_interpolation(
                cube_wavelength, rss.wavelength, rss.variance[fibre]
            )
            f_mask = np.interp(cube_wavelength, rss.wavelength, mask[fibre])
            # Mask all pixels contaminated
            f_mask = f_mask > 0
        else:
            f_intensity = rss.intensity[fibre]
            f_variance = rss.variance[fibre]
            f_mask = mask[fibre]

        offset_ra_pix = fibre_pixel_pos_cols[fibre]
        offset_dec_pix = fibre_pixel_pos_rows[fibre]
        # Interpolate fibre to cube
        datacube, datacube_var, datacube_weight = interpolate_fibre(
            fib_spectra=f_intensity,
            fib_variance=f_variance,
            cube=datacube,
            cube_var=datacube_var,
            cube_weight=datacube_weight,
            pix_pos_cols=offset_ra_pix,
            pix_pos_rows=offset_dec_pix,
            kernel=kernel,
            adr_cols=adr_ra_pixel,
            adr_rows=adr_dec_pixel,
            fibre_mask=f_mask,
        )
    return datacube, datacube_var, datacube_weight, qc_fig


def build_cube(
    rss_set,
    wcs=None,
    wcs_params=None,
    kernel=GaussianKernel,
    kernel_size_arcsec=2.0,
    kernel_truncation_radius=2.0,
    adr_set=None,
    mask_flags=None,
    qc_plots=False,
    **kwargs,
):
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
    vprint("[Cubing] Starting cubing process")
    if wcs is None and wcs_params is None:
        raise ValueError("User must provide either wcs or wcs_params values.")
    if wcs is None and wcs_params is not None:
        wcs = WCS(wcs_params)
    interm_prod = {}
    # Initialise kernel
    if isinstance(kernel, type):
        kernel_size_arcsec = ancillary.check_unit(kernel_size_arcsec, u.arcsec)
        pixel_size = wcs.celestial.pixel_scale_matrix.diagonal().mean() << u.deg
        kernel_scale = (kernel_size_arcsec / pixel_size).decompose()

        vprint(
            f"[Cubing] Initialising {kernel.__name__}"
            + f"\n Scale: {kernel_scale:.1f} (pixels)"
            + f"\n Truncation radius: {kernel_truncation_radius:.1f}"
        )
        kernel = kernel(
            pixel_scale_arcsec=pixel_size,
            scale=kernel_scale,
            truncation_radius=kernel_truncation_radius,
        )
    else:
        kernel_size_arcsec = kernel.scale * kernel.pixel_scale_arcsec
        vprint(
            f"[Cubing] User-provided interpolation kernel"
            + f"\n Scale: {kernel.scale:.1f} (pixels)"
            + f"\n Truncation radius: {kernel.truncation_radius:.1f}"
        )

    # Create the output data units
    if u.second in rss_set[0].intensity.unit.bases:
        output_unit = rss_set[0].intensity.unit
    else:
        output_unit = rss_set[0].intensity.unit / u.second
    vprint(f"[Cubing] Initialising new Cube with dimensions: {wcs.array_shape}")
    vprint(f"Output Cube units: {output_unit.to_string()}")

    all_datacubes = np.full((len(rss_set), *wcs.array_shape), fill_value=np.nan) << output_unit
    all_var = np.full(all_datacubes.shape, fill_value=np.nan) << output_unit**2
    # Total weight per spaxel
    all_w = np.full(all_datacubes.shape, fill_value=np.nan)
    # Exposure time per spaxel
    all_exp = np.full(all_datacubes.shape, fill_value=np.nan) << u.second
    # "Empty" array that will be used to store exposure times
    exposure_times = np.full((len(rss_set)), fill_value=np.nan) << u.second

    # For each RSS two arrays containing the ADR over each axis might be provided
    # otherwise they will be set to None
    if adr_set is None:
        adr_set = [(None, None)] * len(rss_set)

    for ith, rss in enumerate(rss_set):
        exposure_times[ith] = rss.info["exptime"]

        # Interpolate RSS to data cube
        datacube_i, datacube_var_i, datacube_weight_i, rss_plots = interpolate_rss(
            rss,
            wcs=wcs,
            kernel=kernel,
            datacube=np.zeros(wcs.array_shape) << rss.intensity.unit,
            datacube_var=np.zeros(wcs.array_shape) << rss.variance.unit,
            datacube_weight=np.zeros(wcs.array_shape),
            adr_ra_arcsec=adr_set[ith][0],
            adr_dec_arcsec=adr_set[ith][1],
            mask_flags=mask_flags,
            qc_plots=qc_plots,
        )

        interm_prod[f"rss_{ith + 1}"] = rss_plots
        if u.second in rss.intensity.unit.bases:
            all_datacubes[ith] = datacube_i
            all_var[ith] = datacube_var_i
        else:
            all_datacubes[ith] = datacube_i / exposure_times[ith]
            all_var[ith] = datacube_var_i / exposure_times[ith] ** 2

        all_w[ith] = datacube_weight_i
        all_exp[ith] = datacube_weight_i * exposure_times[ith]

    # Combine all cubes
    stacking_method = kwargs.get("stack_method", CubeStacking.mad_clipping)
    stacking_args = kwargs.get("stack_method_args", {})
    vprint(f"[Cubing] Stacking individual cubes using {stacking_method.__name__}")
    vprint(f"[Cubing] Additonal arguments for stacking: {stacking_args}")
    datacube, datacube_var = stacking_method(all_datacubes, all_var, **stacking_args)
    info = dict(kernel_size_arcsec=kernel_size_arcsec, **kwargs.get("cube_info", {}))
    # Create WCS information
    cube = Cube(intensity=datacube, variance=datacube_var, wcs=wcs, info=info)
    if qc_plots:
        # Compute the fibre coverage and exposure time maps
        interm_prod[f"weights"] = qc_cubing(all_w, all_exp)
        interm_prod[f"stack_cube"] = qc_cube(cube)

    if kwargs.get("keep_individual_cubes", False):
        for ith, (intensity, variance) in enumerate(zip(all_datacubes, all_var)):
            ind_cube = Cube(intensity=intensity, variance=variance,
                            wcs=wcs, info=info)
            interm_prod[f"cube_{ith + 1}"] = (ind_cube, qc_cube(ind_cube))
    return cube, interm_prod


def build_wcs(
    datacube_shape,
    reference_position,
    spatial_pix_size: u.Quantity,
    spectra_pix_size: u.Quantity,
    radesys="ICRS    ",
    equinox=2000.0,
):
    """Create a WCS using cubing information.

    Integer pixel values fall at the center of pixels.

    Parameters
    ----------
    datacube_shape: tuple
        Pixel shape of the datacube (wavelength, ra, dec).
    reference_position : tuple
        Values corresponding to the origin of the wavelength axis, and sky position of the central pixel.
    spatial_pix_size : u.Quantity
        Pixel size along the spatial direction.
    spectra_pix_size : u.Quantity
        Pixel size along the spectral direction.

    """
    wcs_dict = {
        # Spatial dimensions
        "RADECSYS": radesys,
        "EQUINOX": equinox,
        "CTYPE1": "RA---TAN",
        "CUNIT1": "deg",
        "CDELT1": spatial_pix_size.to_value("deg"),
        "CRPIX1": datacube_shape[1] / 2,
        "CRVAL1": reference_position[1].to_value("deg"),
        "NAXIS1": datacube_shape[1],
        "CTYPE2": "DEC--TAN",
        "CUNIT2": "deg",
        "CDELT2": spatial_pix_size.to_value("deg"),
        "CRPIX2": datacube_shape[2] / 2,
        "CRVAL2": reference_position[2].to_value("deg"),
        "NAXIS2": datacube_shape[2],
        # Spectral dimension
        "CTYPE3": "WAVE    ",
        "CUNIT3": "angstrom",
        "CDELT3": spectra_pix_size.to_value("angstrom"),
        "CRPIX3": 0,
        "CRVAL3": reference_position[0].to_value("angstrom"),
        "NAXIS3": datacube_shape[0],
    }
    wcs = WCS(wcs_dict)
    return wcs


def build_wcs_from_rss(
    rss_list: list,
    spatial_pix_size: u.Quantity,
    spectra_pix_size: u.Quantity,
    join_type="outer",
    **kwargs,
):
    """Compute the effective WCS resulting from combining an input list of RSS.

    Description
    -----------
    This methods creates a WCS that contains an input list RSS data. The joint
    spectral coverage and field of view, resulting from the combination of the
    individual RSS exposures, can consists of the space that contains all RSS
    (`outer`) or restricted to the region in common among all exposures (`inner`).

    Parameters
    ----------
    rss_list : list or RSS
        List of :class:`RSS`.
    spatial_pix_size : u.Quantity
        Angular spaxel size.
    spectral_pix_size : u.Quantity
        Spectral pixel size.
    join_type : str, optional, default=`outer`
        Which join method to use for creating the combined WCS (`outer` or `inner`).
    **kwargs : dict
        Additional arguments passed to :func:`build_wcs`.

    Returns
    -------
    wcs : astropy.wcs.WCS
        The WCS resulting from combining the list of RSS.
    """
    if isinstance(rss_list, RSS):
        rss_list = [rss_list]

    rss_footprint = np.zeros((len(rss_list), 4, 2)) << u.deg
    rss_spectral_range = np.zeros((len(rss_list), 2)) << u.AA
    for ith, rss in enumerate(rss_list):
        rss_footprint[ith] = rss.get_footprint().to("deg")
        rss_spectral_range[ith] = rss.wavelength[[0, -1]].to("AA")

    # Select the area the cover
    if join_type == "outer":
        max_ra, max_dec = np.nanmax(rss_footprint[:, 0], axis=0)
        min_ra, min_dec = np.nanmin(rss_footprint[:, -1], axis=0)
        min_wl, max_wl = (
            np.nanmin(rss_spectral_range[:, 0]),
            np.nanmax(rss_spectral_range[:, 1]),
        )
    elif join_type == "inner":
        max_ra, max_dec = np.nanmin(rss_footprint[:, 0], axis=0)
        min_ra, min_dec = np.nanmax(rss_footprint[:, -1], axis=0)
        min_wl, max_wl = (
            np.nanmax(rss_spectral_range[:, 0]),
            np.nanmin(rss_spectral_range[:, 1]),
        )

    ra_cen, dec_cen = (max_ra + min_ra) / 2, (max_dec + min_dec) / 2
    ra_width, dec_width = max_ra - min_ra, max_dec - min_dec
    wl_range = max_wl - min_wl

    vprint("Combined footprint center: {:.4f}, {:.4f}".format(ra_cen, dec_cen))
    vprint(
        "Combined footprint Fov: {:.2f}, {:.2f} (arcmin)".format(
            ra_width.to("arcmin"), dec_width.to("arcmin")
        )
    )
    vprint(
        "Combined footprint wavelength range: {:.1f}, {:.1f} (AA)".format(
            min_wl.to("AA"), max_wl.to("AA")
        )
    )

    datacube_shape = (
        int(np.round((wl_range / spectra_pix_size).decompose().value,
                     decimals=0)),
        int(np.round((ra_width / spatial_pix_size).decompose().value,
                     decimals=0)),
        int(np.round((dec_width / spatial_pix_size).decompose().value,
                     decimals=0)),
    )
    reference_position = (min_wl, ra_cen, dec_cen)
    vprint(f"WCS array shape: {datacube_shape} [wave, ra, dec]")
    return build_wcs(
        datacube_shape=datacube_shape,
        reference_position=reference_position,
        spatial_pix_size=spatial_pix_size,
        spectra_pix_size=spectra_pix_size,
        **kwargs,
    )


def make_white_image_from_array(data_array, wavelength=None, **kwargs):
    """Create a white image from a 3D data array.

    Parameters
    ----------
    data_array : np.ndarray
        3D data array. First axis must correspond to the spectral dimension.
    wavelength : np.ndarray
        Wavelength associated to `data_array`.
    **kwargs : dict
        Additional arguments to be passed to :func:`Cube.get_white_image`.

    Return
    ------
    white_image : np.ndarray
        White image array.
    """
    vprint(f"Creating a Cube from input array")
    cube = Cube(intensity=data_array, wavelength=wavelength)
    return cube.get_white_image(**kwargs)


def make_dummy_cube_from_rss(rss, spa_pix_arcsec=0.5, kernel_pix_arcsec=1.0):
    """Create an empty datacube array from an input RSS.

    Parameters
    ----------
    rss : :class:`RSS`
        Input RSS
    spa_pix_arcsec : u.Quantity
        Spaxel angular size.
    kernel_pix_arcsec : u.Quantity
        Kernel scale angular size.

    Return
    ------
    cube : :class:`Cube`
    """
    spa_pix_arcsec = ancillary.check_unit(spa_pix_arcsec, u.arcsec)
    kernel_pix_arcsec = ancillary.check_unit(kernel_pix_arcsec, u.arcsec)
    wcs = build_wcs_from_rss(
        rss,
        spatial_pix_size=spa_pix_arcsec,
        spectra_pix_size=rss.wavelength[1] - rss.wavelength[0],
    )
    cube = build_cube(
        [rss],
        pixel_size_arcsec=spa_pix_arcsec,
        wcs=wcs,
        kernel_size_arcsec=kernel_pix_arcsec,
    )
    return cube


# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
