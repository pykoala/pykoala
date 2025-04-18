"""
This module contains the main tools for building 3D datacubes by interpolating
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
from pykoala import vprint, VerboseMixin


class CubeStacking:
    """Collection of cubing stacking methods.

    Each method takes as input arguments a collection of cubes and variances,
    either in the form of a list or as an array with the first dimension corresponding
    to each cube, and additional keyword arguments.
    """

    @staticmethod
    def sigma_clipping(cubes: np.ndarray, variances: np.ndarray, **kwargs):
        """Perform cube stacking using STD clipping.

        Parameters
        ----------
        cubes: np.ndarray
            An array consisting of the collection of data to combine. The first
            dimension must correspond to the individual elements (e.g. datacubes)
            that will be combined. If the size of the first dimension is 1, it
            will return `cubes[0]` withouth applying any combination.
        variances: np.ndarray
            Array of variances associated to cubes.
        inv_var_weight: np.ndarray, optional
            An array of weights to apply during the stacking.

        Returns
        -------
        stacked_cube: np.ndarray
            The result of stacking the data in cubes along axis 0.
        stacked_variance: np.ndarray
            The result of stacking the variances along axis 0.
        """
        if cubes.shape[0] == 1:
            vprint("Only one cube to stack")
            return cubes[0], variances[0]

        nsigma = kwargs.get("nsigma", 3.0)
        sigma = np.nanstd(cubes, axis=0)
        mean = np.nanmean(cubes, axis=0)

        good_pixel = np.abs((cubes - mean[np.newaxis]) / sigma[np.newaxis]) < nsigma
        w = np.where(good_pixel, 1.0, 0.0)
        if kwargs.get("inv_var_weight", False):
            w = np.where(
                (variances > 0) & np.isfinite(variances) & good_pixel,
                1 / variances,
                np.nan,
            )
        norm = np.nansum(w, axis=0)
        illuminated_spx = norm > 0
        w = np.where(illuminated_spx, w / norm, np.nan)
        n_pix = np.sum(good_pixel, axis=0)
        stacked_cube = np.nansum(cubes * w, axis=0)
        # Do not include the pixels that are flagged as bad
        stacked_variance = np.full_like(stacked_cube, fill_value=np.nan)
        stacked_variance = np.where(
            n_pix > 0, np.nansum(variances * good_pixel, axis=0) / n_pix**2, np.nan
        )
        return stacked_cube, stacked_variance

    @staticmethod
    def mad_clipping(cubes: np.ndarray, variances: np.ndarray, **kwargs):
        """Perform cube stacking using MAD clipping.

        Parameters
        ----------
        cubes: np.ndarray
            An array consisting of the collection of data to combine. The first
            dimension must correspond to the individual elements (e.g. datacubes)
            that will be combined. If the size of the first dimension is 1, it
            will return `cubes[0]` withouth applying any combination.
        variances: np.ndarray
            Array of variances associated to cubes.
        inv_var_weight: np.ndarray, optional
            An array of weights to apply during the stacking.

        Returns
        -------
        stacked_cube: np.ndarray
            The result of stacking the data in cubes along axis 0.
        stacked_variance: np.ndarray
            The result of stacking the variances along axis 0.
        """
        if cubes.shape[0] == 1:
            vprint("Only one cube to stack")
            return cubes[0], variances[0]

        nsigma = kwargs.get("nsigma", 3.0)
        sigma = ancillary.std_from_mad(cubes, axis=0)
        median = np.nanmedian(cubes, axis=0)
        # Spaxel weights
        good_pixel = np.abs((cubes - median[np.newaxis]) / sigma[np.newaxis]) < nsigma
        w = np.where(good_pixel, 1.0, 0.0)
        if kwargs.get("inv_var_weight", False):
            w = np.where(
                (variances > 0) & np.isfinite(variances) & good_pixel,
                1 / variances,
                np.nan,
            )
        # Renormalize the weights using only pixels with data
        norm = np.nansum(w, axis=0)
        w = np.where(norm > 0, w / norm, np.nan)
        stacked_cube = np.nansum(cubes * w, axis=0)
        # Do not include the pixels that are flagged as bad
        stacked_variance = np.full_like(stacked_cube, fill_value=np.nan)
        n_pix = np.sum(good_pixel, axis=0)
        stacked_variance = np.where(
            n_pix > 0, np.nansum(variances * good_pixel, axis=0) / n_pix**2, np.nan
        )
        return stacked_cube, stacked_variance


# -------------------------------------------
# Fibre Interpolation and cube reconstruction
# -------------------------------------------


class InterpolationKernel(object):
    r"""Interpolation Kernel.

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
    def scale(self) -> u.Quantity:
        """Kernel scale size in pixels."""
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = ancillary.check_unit(
            scale, u.pixel, equivalencies=self.pixel_scale
        )

    @property
    def scale_arcsec(self) -> u.Quantity:
        """Kernel scale size in arcseconds."""
        return self.scale.to(u.arcsec, self.pixel_scale)

    @property
    def truncation_radius(self) -> float:
        """Maximum value of `u` beyond which the kernel is set to 0."""
        return self._truncation_radius

    @truncation_radius.setter
    def truncation_radius(self, value):
        self._truncation_radius = value

    @property
    def pixel_scale(self) -> u.Equivalency:
        """Target pixel scale in arcsec per pixel."""
        return self._pixel_scale

    @pixel_scale.setter
    @u.quantity_input(pixel_scale=u.arcsec / u.pixel)
    def pixel_scale(self, pixel_scale):
        self._pixel_scale = u.pixel_scale(pixel_scale)

    def __init__(self, pixel_scale, scale, **kwargs):
        self.pixel_scale = pixel_scale
        self.scale = scale
        self.truncation_radius = kwargs.get("truncation_radius", 1.0)

    @abstractmethod
    def kernel_1D(self):
        pass

    @abstractmethod
    def kernel_2D(self):
        pass


class ParabolicKernel(InterpolationKernel):
    r"""Parabolic or Epanechnikov InterpolationKernel.

    The parabolic kernel is defined as:

    .. math::

        K(u) = \frac{3}{4}(1 - u^2),\, u\in [-1, 1]

    This model enforces `truncation_radius=1`.
    """

    def __init__(self, pixel_scale, scale, **kwargs):
        if "truncation_radius" in kwargs:
            del kwargs["truncation_radius"]
        super().__init__(pixel_scale, scale, truncation_radius=1, **kwargs)

    def cmf(self, ker_u):
        """Cumulative mass distribution.

        Parameters
        ----------
        ker_u : np.ndarray
            Positions in units of the kernel scale.

        Returns
        -------
        cmf : np.ndarray
            The values of the cumulative distribution evaluated at the input
            values of ker_u.
        """
        ker_u_clip = np.atleast_1d(ker_u).clip(-1, 1)
        return (3.0 * ker_u_clip - ker_u_clip**3 + 2.0) / 4

    @u.quantity_input(x_edges=u.pixel)
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
        # Check that the input values are in increasing order
        assert (
            x_edges[1:] > x_edges[:-1]
        ).all(), "Input x_edges values must be increasing"
        # Convert pixels to kernel scale units
        u_edges = (x_edges / self.scale).clip(-1, 1)
        # Compute the cumulative distribution
        cumulative = self.cmf(u_edges)
        # Compute the weight on each bin
        weights = np.diff(cumulative)
        return weights

    @u.quantity_input(x_edges=u.pixel, y_edges=u.pixel)
    def kernel_2D(self, x_edges, y_edges):
        """Compute the kernel weights defined by bin edges in x and y directions.

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
        grid_yy, grid_xx = np.meshgrid(
            y_edges / self.scale, x_edges / self.scale, indexing="ij"
        )
        cum_k = self.cmf(grid_xx) * self.cmf(grid_yy)
        weights = np.diff(cum_k, axis=0)
        weights = np.diff(weights, axis=1)
        return weights


class GaussianKernel(InterpolationKernel):
    r"""Gaussian InterpolationKernel.

    The Gaussian kernel is defined as:

    .. math::

        K(u) = \frac{1}{\sqrt{2\pi}} e^{-u^2/2},\, u\in [-\infty, \infty]

    The kernel domain is restricted to the range ``[-truncation_radius, truncation_radius]``.
    By default ``truncation_radius=3``.
    """

    def __init__(self, pixel_scale, scale, truncation_radius=3.0, **kwargs):
        super().__init__(
            pixel_scale, scale, truncation_radius=truncation_radius, **kwargs
        )
        # Minimum and maximum percentiles used for renormalizing the kernel
        self.left_norm = 0.5 * (1 + erf(-self.truncation_radius / np.sqrt(2)))
        self.right_norm = 0.5 * (1 + erf(self.truncation_radius / np.sqrt(2)))

    def cmf(self, ker_u):
        """Kernel cumulative distribution function.

        Parameters
        ----------
        ker_u : np.ndarray
            Positions in units of the kernel scale.

        Returns
        -------
        cmf : np.ndarray
            The values of the cumulative distribution evaluated at the input
            values of ker_u.
        """
        cmf = (0.5 * (1 + erf(ker_u / np.sqrt(2))) - self.left_norm) / (
            self.right_norm - self.left_norm
        )
        return cmf.clip(0, 1)

    @u.quantity_input(x_edges=u.pixel)
    def kernel_1D(self, x_edges, axis=0):
        cumulative = self.cmf(x_edges / self.scale)
        weights = np.diff(cumulative, axis=axis)
        return weights

    def kernel_2D(self, x_edges, y_edges):
        weights = (
            self.kernel_1D(x_edges)[np.newaxis, :]
            * self.kernel_1D(y_edges)[:, np.newaxis]
        )
        return weights


class TopHatKernel(InterpolationKernel):
    r"""TopHat (uniform) InterpolationKernel.

    The TopHat kernel is defined as:

    .. math::

        K(u) = \frac{1}{2},\, u\in [-1, 1]

    With :math:`u` restricted to the range [-1, 1] (i.e., this model enforces
    `truncation_radius=1`).
    """

    def __init__(self, pixel_scale, scale, **kwargs):
        if "truncation_radius" in kwargs:
            del kwargs["truncation_radius"]
        super().__init__(pixel_scale, scale, truncation_radius=1, **kwargs)

    def cmf(self, ker_u):
        """Kernel cumulative distribution function.

        Parameters
        ----------
        ker_u : np.ndarray
            Positions in units of the kernel scale.

        Returns
        -------
        cmf : np.ndarray
            The values of the cumulative distribution evaluated at the input
            values of ker_u.
        """
        cmf = 0.5 * (ker_u + 1)
        return cmf.clip(0, 1)

    @u.quantity_input(x_edges=u.pixel)
    def kernel_1D(self, x_edges, axis=0):
        cumulative = self.cmf(x_edges / self.scale)
        weights = np.diff(cumulative, axis=axis)
        return weights

    def kernel_2D(self, x_edges, y_edges):
        weights = (
            self.kernel_1D(x_edges)[np.newaxis, :]
            * self.kernel_1D(y_edges)[:, np.newaxis]
        )
        return weights


class CircularTopHatKernel(TopHatKernel):
    """Circular TopHat kernel."""

    def kernel_2D(self, x_edges, y_edges):
        rad = x_edges[np.newaxis, :] ** 2 + y_edges[:, np.newaxis] ** 2
        return self.kernel_1D(rad, axis=(0, 1))


class DrizzlingKernel(TopHatKernel):
    """Drizzling InterpolationKernel.

    This kernel follows the same phylosophy as in Fruchter & Hook 1997.
    Users may define different fibre footprint shapes (``fibre_shape``) among:
    ``squared``, ``circular``, or ``hexagonal``. This will determine what is the
    overlapping fraction between a given spaxel and the fibre.
    """

    fibre_frac = {
        "circle": ancillary.pixel_in_circle,
        "hexagon": ancillary.pixel_in_hexagon,
        "square": ancillary.pixel_in_square,
    }
    """Dictionary mapping the available fibre shapes to the corresponding method."""

    def __init__(self, pixel_scale, scale, **kwargs):
        super().__init__(pixel_scale, scale, **kwargs)
        # Assume circular fibre footprint by default
        self.pixel_frac_method = self.fibre_frac[kwargs.get("fibre_shape", "circle")]

    def kernel_1D(self):
        raise NotImplementedError("kernel_1D has not been implemented in this class")

    def kernel_2D(self, x_edges, y_edges):
        weights = np.zeros((y_edges.size - 1, x_edges.size - 1))
        # y == rows, x == columns
        # Only the lower-left corner of every pixel is required.
        pix_edge_y, pix_edge_x = np.meshgrid(
            y_edges[:-1].to_value("pixel"),
            x_edges[:-1].to_value("pixel"),
            indexing="ij",
        )
        for i, pos in enumerate(zip(pix_edge_x.flatten(), pix_edge_y.flatten())):
            _, area_fraction = self.pixel_frac_method(
                pixel_pos=pos,
                pixel_size=1,
                pos=(0.0, 0.0),
                radius=self.scale.to_value("pixel") / 2,
            )
            weights[np.unravel_index(i, weights.shape)] = area_fraction
        return weights


# ------------------------------------------------------------------------------
# Fibre interpolation
# ------------------------------------------------------------------------------


class CubeInterpolator(VerboseMixin):
    """A class for combining multiple RSS into a 3D datacube.

    This class performs the combination of a set of RSS data into a 3D regular
    grid defined by a WCS.

    The basic ingredients for creating a datacube are:
    - Set of row-stacked spectra (:class:`RSS`).
    - A target :class:`WCS` that defines the final datacube dimensions.
    - An :class:`InterpolationKernel` that will determine the contribution of
    each RSS at every location.

    In addition, users may provide additional information such as:
    - A set of differential atmospheric corrections (DAR) for each RSS that
    accounts for wavelength-dependent fibre spatial variations.
    - A set of flags of the RSS masks that will be used to mask out pixels during
    the interpolation.

    To assess the quality of the interpolation, users may want to keep intermediate
    products or create plots that allow to evaluate the performance of the cubing.
    First, the variables ``all_datacubes`` and ``all_var`` store the interpolated
    fluxes and variances of each individual RSS, respectively.
    The kernel weights and net exposure times are stored in the variables
    ``all_weights`` and ``all_exp_time``.
    Furthermore, if "keep_individual_cubes=True" the variable ``rss_inter_products``
    will store each individual :class:`Cube` for each RSS and a QC plot.

    Users may set the argument ``qc_plots=True``, that will create


    Example
    -------
    >>> from pykoala.instruments.mock import mock_rss
    >>> interpolator = CubeInterpolator(rss_set=[mock_rss()])
    >>> cube = interpolator.build_cube()
    """

    @property
    def rss_set(self) -> list:
        """List of target :class:`RSS` to be combined into a :class:`Cube`."""
        return self._rss_set
    
    @rss_set.setter
    def rss_set(self, rss_set):
        self._rss_set = rss_set
    
    @property
    def adr_set(self) -> list:
        """List of ADR to apply during interpolation."""
        return self._adr_set

    @adr_set.setter
    def adr_set(self, adr_set):
        self._adr_set = adr_set

    @property
    def mask_flags(self) -> list:
        """List of flags to be included during the pixel masking."""
        return self._mask_flags

    @mask_flags.setter
    def mask_flags(self, mask_flags):
        self._mask_flags = mask_flags

    @property
    def target_wcs(self) -> WCS:
        """Target WCS into which the RSS data will be interpolated."""
        return self._target_wcs
    
    @target_wcs.setter
    def target_wcs(self, wcs):
        self._target_wcs = wcs
    
    @property
    def kernel(self) -> InterpolationKernel:
        """:class:`InterpolationKernel` used to combine the RSS data."""
        return self._kernel

    @kernel.setter
    def kernel(self, kernel):
        self._kernel = kernel

    def __init__(
        self,
        rss_set,
        wcs=None,
        kernel=GaussianKernel,
        kernel_scale=2.0 << u.arcsec,
        kernel_truncation_radius=3.0,
        adr_set=None,
        mask_flags=None,
        qc_plots=False,
        **kwargs,
    ):
        # Verbosity parameters
        self.logger = kwargs.get("logger", "CubeInterp")
        self.verbose = kwargs.get("verbose", True)
        # Set of RSS to be interpolated
        self.rss_set = rss_set
        # Differential atmospheric refraction
        if adr_set is None:
            self.adr_set = [(None, None)] * len(rss_set)
        else:
            self.adr_set = adr_set
        # Names of flags to be used for masking pixels
        self.mask_flags = mask_flags
        # WCS defining the target dimensions of the cube
        if wcs is None:
            self.vprint("Computing WCS using input list of RSS")
            self.target_wcs = build_wcs_from_rss(
                self.rss_set,
                kwargs.get("spatial_pix_size", 1 << u.arcsec),
                kwargs.get("spectra_pix_size", 1 << u.AA),
            )
        else:
            self.vprint("Using input WCS")
            self.target_wcs = wcs
        # RSS Interpolation kernel
        if isinstance(kernel, type):
            kernel_scale = ancillary.check_unit(kernel_scale, u.arcsec)
            # Assume square pixel
            pixel_scale = (
                np.abs(self.target_wcs.celestial.pixel_scale_matrix.diagonal()).mean()
                << u.deg / u.pixel
            )

            self.vprint(
                f"Initialising {kernel.__name__}"
                + f"\n Scale: {kernel_scale:.1f} (pixels)"
                + f"\n Truncation radius: {kernel_truncation_radius:.1f}"
                + f" ({kernel_scale * kernel_truncation_radius:.1f} px)"
            )
            self.kernel = kernel(
                pixel_scale=pixel_scale,
                scale=kernel_scale,
                truncation_radius=kernel_truncation_radius,
            )
        else:
            self.kernel = kernel
            kernel_scale = self.kernel.scale_arcsec
            self.vprint(
                f"User-provided interpolation kernel"
                + f"\n Scale: {self.kernel.scale:.1f} (pixels)"
                + f"\n Truncation radius: {self.kernel.truncation_radius:.1f}"
            )
        # Create the output data units
        if u.second in rss_set[0].intensity.unit.bases:
            output_unit = rss_set[0].intensity.unit
        else:
            output_unit = rss_set[0].intensity.unit / u.second
        self.vprint(
            f"Initialising new Cube with dimensions: {self.target_wcs.array_shape}"
        )
        self.vprint(f"Output Cube units: {output_unit.to_string()}")

        self.all_datacubes = (
            np.full(
                (len(self.rss_set), *self.target_wcs.array_shape), fill_value=np.nan
            )
            << output_unit
        )
        self.all_var = (
            np.full(self.all_datacubes.shape, fill_value=np.nan) << output_unit**2
        )
        # Total weight per spaxel
        self.all_weights = np.full(self.all_datacubes.shape, fill_value=np.nan)
        # Exposure time per spaxel
        self.all_exp_time = (
            np.full(self.all_datacubes.shape, fill_value=np.nan) << u.second
        )
        # "Empty" array that will be used to store exposure times
        self.exposure_times = (
            np.array([rss.info["exptime"].to_value("second") for rss in self.rss_set])
            << u.second
        )

        # Create variables to store plots and intermediate products
        self.make_qc_plots = qc_plots
        if self.make_qc_plots:
            self.vprint("QC plots will be generated during cube production")
        self.keep_individual_cubes = kwargs.get("keep_individual_cubes", False)
        if self.keep_individual_cubes:
            self.vprint("Cubes for each individual RSS will be built")
        self.rss_inter_products = {}
        self.cube_plots = {}

    def _fill_info(self, info):
        """Fill the info metadata using the RSS set.
        
        Parameters
        ----------
        info : dict
            Dictionary containing the Cube metadata.
        
        Returns
        -------
        info : dict
            Updated dictionary
        """
        if "airmass" not in info:
            info["airmass"] = np.nanmean(
                [rss.info.get("airmass", np.nan) for rss in self.rss_set])
        if "exptime" not in info:
            info["exptime"] = np.nansum(
                [rss.info.get("exptime").to_value("s") for rss in self.rss_set
                 if rss.info.get("exptime") is not None]) << u.second
        return info

    def build_cube(
        self,
        stacking_method=CubeStacking.mad_clipping,
        stacking_args=None,
        cube_info={},
    ) -> Cube:
        """Perform the interpolation of the RSS into a  :class:`Cube`.

        Parameters
        ----------
        stacking_method : func, optional
            Stacking method to use for combining the RSS (See :class:`CubeStacking`).
        stacking_args : dict, optional
            Additional arguments passed to the stacking method.
        cube_info : dict, optional
            Additional metadata to be included in the final :class:`Cube` or
            the intermediate RSS products.

        Returns
        -------
        cube : :class:`Cube`
            The resulting cube from combining the input RSS.
        """
        self.vprint("Cubing input RSS set")

        # Save intermediate products
        for ith, rss in enumerate(self.rss_set):
            # Interpolate RSS to data cube
            (
                datacube_i,
                datacube_var_i,
                datacube_weight_i,
                interp_info,
            ) = self.interpolate_rss(
                rss,
                # Initialise the empty variables
                datacube=np.zeros(self.target_wcs.array_shape) << rss.intensity.unit,
                datacube_var=np.zeros(self.target_wcs.array_shape) << rss.variance.unit,
                datacube_weight=np.zeros(self.target_wcs.array_shape),
                # Differential Atmospheric Refraction
                adr_ra_arcsec=self.adr_set[ith][0],
                adr_dec_arcsec=self.adr_set[ith][1],
            )

            zero_weight = datacube_weight_i == 0
            datacube_i[zero_weight] = np.nan
            datacube_var_i[zero_weight] = np.nan
            datacube_weight_i[zero_weight] = np.nan

            self.all_weights[ith] = datacube_weight_i
            self.all_exp_time[ith] = datacube_weight_i * self.exposure_times[ith]

            if u.second in rss.intensity.unit.bases:
                self.all_datacubes[ith] = datacube_i / self.all_weights[ith]
                self.all_var[ith] = datacube_var_i / self.all_weights[ith] ** 2
            else:
                self.all_datacubes[ith] = datacube_i / self.all_exp_time[ith]
                self.all_var[ith] = datacube_var_i / self.all_exp_time[ith] ** 2

            # Create a single-RSS cube
            if self.keep_individual_cubes:
                ind_cube = Cube(
                    intensity=self.all_datacubes[ith],
                    variance=self.all_var[ith],
                    wcs=self.target_wcs,
                    info=dict(kernel_scale=self.kernel.scale_arcsec, name=f"rss_{ith}"),
                )
                interp_info["cube"] = (ind_cube, qc_cube(ind_cube))

            self.rss_inter_products[f"rss_{ith}"] = interp_info

        # Combine all cubes
        self.vprint(f"Stacking individual cubes using {stacking_method.__name__}")
        if stacking_args:
            self.vprint(f"Additonal arguments for stacking: {stacking_args}")
        else:
            stacking_args = {}
        datacube, datacube_var = stacking_method(
            self.all_datacubes, self.all_var, **stacking_args
        )
        info = dict(kernel_scale=self.kernel.scale_arcsec, **cube_info)
        info = self._fill_info(info)
        # Create the Cube
        cube = Cube(intensity=datacube, variance=datacube_var,
                    wcs=self.target_wcs, info=info)
        if self.make_qc_plots:
            self.vprint("Producing quality assessment plots")
            # Fibre coverage and exposure time maps
            self.cube_plots["weights"] = qc_cubing(self.all_weights, self.all_exp_time)
            # QC cube maps
            self.cube_plots["stack_cube"] = qc_cube(cube)
        return cube

    def interpolate_rss(
        self,
        rss,
        datacube,
        datacube_var,
        datacube_weight,
        adr_ra_arcsec=None,
        adr_dec_arcsec=None,
    ):
        """Perform fibre interpolation using a RSS into to a 3D datacube.

        Parameters
        ----------
        rss : :class:`RSS`
            Target RSS to be interpolated.
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

        Returns
        -------
        datacube : u.Quantity
            Array contanining the interpolated intensity of the RSS into the
            target WCS.
        datacube_var : u.Quantity
            Array contanining the interpolated variance of the RSS into the
            target WCS.
        datacube_weight : u.Quantity
            Array contanining the sum of all kernels weights of all fibres.
        interm_products : dict
            Dictionary containing intermediate products such as individual
            fibre weights.
        """
        self.vprint("Interpolating RSS to cube")

        # Obtain fibre position in the detector (center of pixel)
        (
            fibre_pixel_pos_cols,
            fibre_pixel_pos_rows,
        ) = self.target_wcs.celestial.world_to_pixel(
            SkyCoord(rss.info["fib_ra"], rss.info["fib_dec"])
        )

        # Dictionary that stores additional information
        interm_products = {
            "fib_pix_col": fibre_pixel_pos_cols,
            "fib_pix_row": fibre_pixel_pos_rows,
        }

        # Wavelength array of target datacube
        cube_wavelength = self.target_wcs.spectral.array_index_to_world(
            np.arange(self.target_wcs.array_shape[0])
        ).to(rss.wavelength.unit)

        if adr_dec_arcsec is not None:
            adr_dec_pixel = adr_dec_arcsec.to(u.pixel, self.kernel.pixel_scale)
            if cube_wavelength.size != rss.wavelength.size or np.allclose(
                cube_wavelength, rss.wavelength
            ):
                adr_dec_pixel = np.interp(
                    cube_wavelength, rss.wavelength, adr_dec_pixel
                )
        else:
            adr_dec_pixel = None
        if adr_ra_arcsec is not None:
            adr_ra_pixel = adr_ra_arcsec.to(u.pixel, self.kernel.pixel_scale)
            if cube_wavelength.size != rss.wavelength.size or np.allclose(
                cube_wavelength, rss.wavelength
            ):
                adr_ra_pixel = np.interp(cube_wavelength, rss.wavelength, adr_ra_pixel)
        else:
            adr_ra_pixel = None
        # Interpolate all RSS fibres
        if self.mask_flags is not None:
            self.vprint(f"Flagging those pixels with: {self.mask_flags}")
            mask = rss.mask.get_flag_map(self.mask_flags)
        else:
            mask = np.zeros(rss.intensity.shape, dtype=bool)

        # Create a fibre coverage map
        if self.make_qc_plots:
            qc_fig = qc_fibres_on_fov(
                datacube.shape[1:],
                fibre_pixel_pos_cols,
                fibre_pixel_pos_rows,
                fibre_diam=getattr(rss, "fibre_diameter", 1.25 << u.arcsec).to(
                    u.pixel, self.kernel.pixel_scale
                ),
            )
            interm_products["qc_fibres_on_fov"] = qc_fig

        if cube_wavelength.size != rss.wavelength.size or np.allclose(
            cube_wavelength, rss.wavelength
        ):
            self.vprint("Fibres will be interpolated to new wavelength grid")
            interp_wave = True
        else:
            interp_wave = False
        interm_products["fibre_weights"] = []
        for fibre in range(rss.intensity.shape[0]):
            # Spectra interpolation
            if interp_wave:
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

            # Interpolate fibre to cube
            datacube, datacube_var, datacube_weight = self.interpolate_fibre(
                fib_spectra=f_intensity,
                fib_variance=f_variance,
                cube=datacube,
                cube_var=datacube_var,
                cube_weight=datacube_weight,
                pix_pos_cols=fibre_pixel_pos_cols[fibre] << u.pixel,
                pix_pos_rows=fibre_pixel_pos_rows[fibre] << u.pixel,
                adr_cols=adr_ra_pixel,
                adr_rows=adr_dec_pixel,
                fibre_mask=f_mask,
                interm_products=interm_products,
            )
        return datacube, datacube_var, datacube_weight, interm_products

    def interpolate_fibre(
        self,
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
        fibre_mask=None,
        interm_products=None,
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
            Atmospheric Differential Refraction (ADR) of each wavelength point
            along x (ra)-axis (m) expressed in pixels.
        adr_rows: (k,) np.array(float), optional, default=None
            Atmospheric Differential Refraction of each wavelength point along
            y (dec) -axis (n) expressed in pixels.
        adr_pixel_frac: float, optional, default=0.05
            ADR Pixel fraction used to bin the spectral pixels. For each bin,
            the median ADR correction will be used to
            correct the range of wavelength.
        fibre_mask: np.ndarray
            Boolean array containing the fibre mask.
        interm_products : dict
            Dictionary that stores intermediate products and metadata.

        Returns
        -------
        cube:
            Original datacube with the fibre data interpolated.
        cube_var:
            Original variance with the fibre data interpolated.
        cube_weight:
            Original datacube weights with the fibre data interpolated.
        interm_products : dict
            Dictionary that stores intermediate products and metadata.
        """
        if adr_rows is None and adr_cols is None:
            adr_rows = np.zeros(fib_spectra.size) << u.pixel
            adr_cols = np.zeros(fib_spectra.size) << u.pixel
            spectral_window = fib_spectra.size
        else:
            # Estimate spectral window
            adr_pixel_frac = ancillary.check_unit(adr_pixel_frac, u.pixel)
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
        fibre_weights = []
        for wl_range in range(0, fib_spectra.size, spectral_window):
            wl_slice = slice(wl_range, wl_range + spectral_window)
            # Kernel along columns direction (x, ra)
            kernel_centre_cols = pix_pos_cols - np.nanmedian(adr_cols[wl_slice])
            kernel_offset = self.kernel.scale * self.kernel.truncation_radius
            cols_min = max(int(kernel_centre_cols.value - kernel_offset.value) - 1, 0)
            cols_max = min(
                int(kernel_centre_cols.value + kernel_offset.value) + 1,
                cube.shape[2] - 1,
            )
            columns_slice = slice(cols_min, cols_max + 1, 1)
            # Kernel along rows direction (y, dec)
            kernel_centre_rows = pix_pos_rows - np.nanmedian(adr_rows[wl_slice])
            rows_min = max(int(kernel_centre_rows.value - kernel_offset.value) - 1, 0)
            rows_max = min(
                int(kernel_centre_rows.value + kernel_offset.value) + 1,
                cube.shape[1] - 1,
            )
            rows_slice = slice(rows_min, rows_max + 1, 1)

            if (cols_max < cols_min) | (rows_max < rows_min):
                continue

            # Compute the kernel weight associated to each location
            weights = self.kernel.kernel_2D(
                np.arange(cols_min - 0.5, cols_max + 1.5, 1.0) * u.pixel
                - kernel_centre_cols,
                np.arange(rows_min - 0.5, rows_max + 1.5, 1.0) * u.pixel
                - kernel_centre_rows,
            )

            fibre_weights.append((wl_slice, rows_slice, columns_slice, weights))

            weights = weights[np.newaxis]
            # Add spectra to cube
            cube[wl_slice, rows_slice, columns_slice] = np.add(
                cube[wl_slice, rows_slice, columns_slice],
                fib_spectra[wl_slice, np.newaxis, np.newaxis] * weights,
            )
            cube_var[wl_slice, rows_slice, columns_slice] = np.add(
                cube_var[wl_slice, rows_slice, columns_slice],
                fib_variance[wl_slice, np.newaxis, np.newaxis] * weights**2,
            )
            cube_weight[wl_slice, rows_slice, columns_slice] = np.add(
                cube_weight[wl_slice, rows_slice, columns_slice],
                pixel_weights[wl_slice, np.newaxis, np.newaxis] * weights,
            )

        interm_products["fibre_weights"].append(fibre_weights)
        return cube, cube_var, cube_weight


def build_wcs(
    datacube_shape,
    reference_position,
    spatial_pix_size: u.Quantity,
    spectra_pix_size: u.Quantity,
    radesys="ICRS    ",
    equinox=2000.0,
) -> WCS:
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
) -> WCS:
    """Compute the effective WCS resulting from combining an input list of RSS.

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
        int(np.round((wl_range / spectra_pix_size).decompose().value, decimals=0)),
        int(np.round((ra_width / spatial_pix_size).decompose().value, decimals=0)),
        int(np.round((dec_width / spatial_pix_size).decompose().value, decimals=0)),
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
    vprint("Creating a Cube from input array")
    cube = Cube(intensity=data_array, wavelength=wavelength)
    return cube.get_white_image(**kwargs)


def make_dummy_cube_from_rss(rss, spa_pix_arcsec=0.5, kernel_pix_arcsec=1.0
                             ) -> Cube:
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
    interpolator = CubeInterpolator(
        [rss],
        pixel_size_arcsec=spa_pix_arcsec,
        wcs=wcs,
        kernel_scale=kernel_pix_arcsec,
    )
    cube = interpolator.build_cube()
    return cube


# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
