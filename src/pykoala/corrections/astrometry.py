"""
Astrometry-related corrections.

This module provides tools to measure and apply relative astrometric offsets to
data containers (RSS and Cube). It supports centroid-based and
cross-correlation registration, as well as creating corrections from external
reference images or FITS files.
"""

# =============================================================================
# Basics packages
# =============================================================================
import numpy as np

# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils.centroids import centroid_2dg, centroid_com

from scipy.ndimage import median_filter

# =============================================================================
# KOALA packages
# =============================================================================
from pykoala import vprint
from pykoala.corrections.correction import CorrectionBase, CorrectionOffset
from pykoala.data_container import RSS, Cube
from pykoala.cubing import make_dummy_cube_from_rss
from pykoala.ancillary import interpolate_image_nonfinite, update_wcs_coords
from pykoala.plotting.utils import qc_registration_centroids
from pykoala import photometry
from pykoala.utils.math import odd_int, robust_standarisation, std_from_mad


class AstrometryCorrection(CorrectionBase):
    """
    Apply an astrometric offset to a data container.

    Attributes
    ----------
    offset : CorrectionOffset or None
        (dRA, dDEC) in angular units.
    path : str or None
        Optional path where the offset was loaded from.
    verbose : bool
        If True, prints progress messages (inherited).

    Notes
    -----
    This class does not measure offsets. Use the standalone registration
    functions in this module to obtain :class:`CorrectionOffset` instances.
    """

    name = "AstrometryCorrection"
    verbose = True
    offset = None
    path = None

    def __init__(self, offset=None, offset_path=None, **correction_kwargs):
        """
        Parameters
        ----------
        offset : CorrectionOffset, optional
            The offset to apply (dRA, dDEC).
        offset_path : str, optional
            Path used to initialise the offset (metadata only).
        **correction_kwargs
            Forwarded to :class:`CorrectionBase`.
        """
        super().__init__(**correction_kwargs)
        self.offset = offset
        self.path = offset_path

    @classmethod
    def from_fits(cls, path):
        """
        Create an instance from a FITS file containing an offset table.

        Parameters
        ----------
        path : str
            Path to a FITS file readable by :class:`CorrectionOffset`.

        Returns
        -------
        AstrometryCorrection
        """
        return cls(offset=CorrectionOffset.from_fits(path=path), offset_path=path)

    def apply(self, data_container):
        """
        Apply the stored offset to a data container.

        Parameters
        ----------
        data_container : RSS or Cube
            Target container.

        Returns
        -------
        dc_corrected : same type as input
            Shallow copy with updated coordinates/WCS.

        Raises
        ------
        ValueError
            If no offset is set.
        """
        if self.offset is None or self.offset.offset_data is None:
            raise ValueError("No offset set. Provide a CorrectionOffset first.")

        d_ra, d_dec = self.offset.offset_data
        self.vprint(f"Applying offset (dRA, dDEC): {d_ra}, {d_dec}")

        dc_out = data_container.copy()
        dc_out.update_coordinates(offset=[d_ra, d_dec])
        self.record_correction(
            dc_out,
            status="applied",
            offset=f"{d_ra.to('arcsec')}, {d_dec.to('arcsec')}",
        )
        return dc_out


def compute_offset_from_external_image(
    data_container, external_image, filter_name, **crosscorr_kwargs
):
    """
    Measure (dRA, dDEC) against an external reference image.

    Parameters
    ----------
    data_container : RSS or Cube
        Target dataset.
    external_image : dict
        Keys:
          - "ccd": astropy.nddata.CCDData
          - "pix_size": float (arcsec per pixel)
    filter_name : str
        Passband name for synthetic aperture photometry.
    **crosscorr_kwargs
        Passed to :func:`pykoala.photometry.crosscorrelate_im_apertures`.

    Returns
    -------
    offset : CorrectionOffset
        Measured (dRA, dDEC).
    results : dict
        Cross-correlation results. Includes a QC figure at key "offset_fig".
    """
    vprint("Computing astrometric offset via external reference")
    dc_phot = photometry.get_dc_aperture_flux(data_container, filter_name)
    mask = dc_phot["aperture_mask"]
    results = photometry.crosscorrelate_im_apertures(
        dc_phot["aperture_flux"][mask],
        dc_phot["coordinates"][mask],
        external_image,
        **crosscorr_kwargs,
    )
    fig = photometry.make_plot_astrometry_offset(
        data_container, dc_phot["synth_photo"], external_image, results
    )
    results["offset_fig"] = fig

    off = CorrectionOffset(
        offset_data=results["offset_min"],
        offset_error=[np.nan << u.deg, np.nan << u.deg],
    )
    return off, results


def register_dataset_centroids(
    data_set, object_name=None, qc_plot=False, **centroid_args
):
    """
    Register a sequence of DataContainers via centroiding.

    The first element is the reference. For each subsequent DC, returns the
    offset required to align it to the reference.

    Parameters
    ----------
    data_set : list of RSS or Cube
        Sequence to register (len >= 2).
    object_name : str, optional
        If provided, absolute reference position resolved by
        :meth:`astropy.coordinates.SkyCoord.from_name`. Otherwise, the centroid
        of the first DC defines the reference.
    qc_plot : bool, default: False
        If True, also returns a QC figure.
    **centroid_args
        Forwarded to :func:`find_centroid_in_dc`.

    Returns
    -------
    offsets : list of CorrectionOffset
        One per input DC. The first one is zero by construction.
    fig : matplotlib.figure.Figure or None
        QC figure if requested.

    Notes
    -----
    Offsets are defined as (dRA, dDEC) to move each DC onto the reference.
    """
    if len(data_set) <= 1:
        raise ArithmeticError("`data_set` must contain at least two elements.")

    # Reference sky position
    if object_name is not None:
        try:
            ref_pos = SkyCoord.from_name(object_name)
        except Exception as e:
            vprint(e)
            centroid_args["full_output"] = False
            ref_pos = find_centroid_in_dc(data_set[0], **centroid_args)
    else:
        centroid_args["full_output"] = False
        ref_pos = find_centroid_in_dc(data_set[0], **centroid_args)

    if qc_plot:
        centroid_args["full_output"] = True
        images_list, wcs_list = [], []

    vprint(f"[Centroids] Reference: RA={ref_pos.ra}, DEC={ref_pos.dec}")

    offsets = [
        CorrectionOffset(
            offset_data=[0.0 * u.deg, 0.0 * u.deg],
            offset_error=[np.nan * u.deg, np.nan * u.deg],
        )
    ]

    for dc in data_set[1:]:
        if qc_plot:
            cube, image, _, cen_world = find_centroid_in_dc(dc, **centroid_args)
            images_list.append(image)
            wcs_list.append(cube.wcs.celestial)
        else:
            cen_world = find_centroid_in_dc(dc, **centroid_args)

        dra, ddec = ref_pos.spherical_offsets_to(cen_world)
        # Move dc onto reference
        off = CorrectionOffset(
            offset_data=[-dra, -ddec], offset_error=[np.nan * u.deg, np.nan * u.deg]
        )
        vprint(
            "[Centroids] (dRA, dDEC): "
            f"{off.offset_data[0].to_value('arcsec'):.2f}, "
            f"{off.offset_data[1].to_value('arcsec'):.2f} arcsec"
        )
        offsets.append(off)

    fig = None
    if qc_plot:
        # Use measured offsets + reference position for the panel
        offs_arr = [[off.offset_data[0], off.offset_data[1]] for off in offsets]
        fig = qc_registration_centroids(
            images_list, wcs_list, offs_arr, ref_pos=ref_pos
        )
        fig.suptitle(
            f"Centroid-based registration ({centroid_args.get('centroider', 'com')})"
        )

    return offsets, fig


def register_dataset_crosscorr(
    data_set,
    *,
    wave_range=None,
    quick_cube_pix_size=0.5,
    qc_plot=False,
    **crosscorr_kwargs,
):
    """
    Register a sequence via white-image cross-correlation.

    Parameters
    ----------
    data_set : list of RSS or Cube
        Sequence to register (len >= 2).
    wave_range : (float, float), optional
        Wavelength range for the white image.
    quick_cube_pix_size : float, default: 0.5
        Pixel size (arcsec) for temporary cubes built from RSS.
    qc_plot : bool, default: False
        If True, also returns a QC figure.
    **crosscorr_kwargs
        Forwarded to :func:`cross_correlate_images`
        (e.g. oversample, median_filter_s, bckgr_kappa_sigma).

    Returns
    -------
    offsets : list of CorrectionOffset
        One per input DC (first one is zero).
    fig : matplotlib.figure.Figure or None
        QC figure if requested.

    Notes
    -----
    Non-finite pixels are inpainted; images are robustly standardised before
    cross-correlation. Offsets move each DC onto the *first* element.
    """
    if len(data_set) <= 1:
        raise ArithmeticError("`data_set` must contain at least two elements.")

    vprint("[Xcorr] Building white images")
    images, wcs_list = [], []
    for dc in data_set:
        if isinstance(dc, RSS):
            cube = make_dummy_cube_from_rss(
                dc,
                spa_pix_arcsec=quick_cube_pix_size,
                kernel_pix_arcsec=quick_cube_pix_size,
            )
            img = cube.get_white_image(wave_range=wave_range, s_clip=3.0)
        elif isinstance(dc, Cube):
            cube = dc
            img = cube.get_white_image(wave_range=wave_range, s_clip=3.0)
        else:
            raise TypeError("Elements of `data_set` must be RSS or Cube")

        if np.isfinite(img).any():
            img = img / np.nansum(img)
        img = interpolate_image_nonfinite(img)

        images.append(img)
        wcs_list.append(cube.wcs.celestial)

    results, ancillary = cross_correlate_images(images, **crosscorr_kwargs)
    ref_origin = wcs_list[0].pixel_to_world(0, 0)

    offsets = [
        CorrectionOffset(
            offset_data=[0.0 * u.deg, 0.0 * u.deg],
            offset_error=[np.nan * u.deg, np.nan * u.deg],
        )
    ]
    for i, (shift, _, _) in enumerate(results, start=1):
        drow, dcol = shift  # (row, col) to align moving -> ref
        vprint(f"[Xcorr] Pixel shift DC#{i}: drow={drow:.2f}, dcol={dcol:.2f}")
        moving_origin = wcs_list[i].pixel_to_world(-dcol, -drow)
        dra, ddec = ref_origin.spherical_offsets_to(moving_origin)
        off = CorrectionOffset(
            offset_data=[-dra, -ddec], offset_error=[np.nan * u.deg, np.nan * u.deg]
        )
        vprint(
            "[Xcorr] (dRA, dDEC): "
            f"{off.offset_data[0].to_value('arcsec'):.2f}, "
            f"{off.offset_data[1].to_value('arcsec'):.2f} arcsec"
        )
        offsets.append(off)

    fig = None
    if qc_plot:
        # Update WCS copies for visualisation (do not mutate originals)
        wcs_vis = [
            (
                update_wcs_coords(
                    w, ra_dec_offset=[off.offset_data[0], off.offset_data[1]]
                )
                if idx > 0
                else w
            )
            for idx, (w, off) in enumerate(zip(wcs_list, offsets))
        ]
        ref_pos = wcs_list[0].pixel_to_world(
            images[0].shape[1] / 2, images[0].shape[0] / 2
        )
        offs_arr = [[off.offset_data[0], off.offset_data[1]] for off in offsets]
        fig = qc_registration_centroids(
            images, wcs_vis, offs_arr, ref_pos=ref_pos, ancillary_info=ancillary
        )
        fig.suptitle("Cross-correlation registration")

    return offsets, fig


def find_centroid_in_dc(
    data_container,
    wave_range=None,
    median_filter_s=None,
    bckgr_kappa_sigma=None,
    centroider="com",
    com_power=1.0,
    quick_cube_pix_size=0.5,
    subbox=None,
    full_output=False,
):
    """
    Find the centre of light in a DataContainer.

    Parameters
    ----------
    data_container : RSS or Cube
    wave_range : (float, float), optional
    median_filter_s : int, optional
    bckgr_kappa_sigma : float, optional
        Clip threshold; pixels (image - median) < kappa*sigma are masked.
    centroider : {'com','gauss'}, default: 'com'
    com_power : float, default: 1.0
        Power applied to intensities before centroiding (for 'com').
    quick_cube_pix_size : float, default: 0.5
    subbox : [[int, int],[int, int]], optional
    full_output : bool, default: False

    Returns
    -------
    SkyCoord or tuple
        If `full_output=False`, returns only the sky coordinate of the centroid.
        Otherwise, returns `(cube, image, centroid_pixel, centroid_world)`.
    """
    if centroider == "com":
        centroider = centroid_com
    elif centroider == "gauss":
        centroider = centroid_2dg

    if subbox is None:
        subbox = [[None, None], [None, None]]
    if isinstance(data_container, RSS):
        vprint(
            "[Registration]  Data provided in RSS format --> creating a dummy datacube"
        )
        cube = make_dummy_cube_from_rss(data_container, quick_cube_pix_size)
        image = cube.get_white_image(wave_range=wave_range, s_clip=3.0)
    elif isinstance(data_container, Cube):
        cube = data_container
        image = cube.get_white_image(wave_range=wave_range, s_clip=3.0)
        image /= np.nansum(image)
    else:
        raise TypeError("Input DC must be an instance of Cube or RSS")
    # Select a subbox
    image = image[subbox[0][0] : subbox[0][1], subbox[1][0] : subbox[1][1]]

    if median_filter_s is not None:
        median_filter_s = odd_int(median_filter_s)
        image = median_filter(image, size=median_filter_s)
    # Mask bad values
    # image = interpolate_image_nonfinite(image)
    centroider_mask = ~np.isfinite(image)

    if bckgr_kappa_sigma is not None:
        median = np.nanmedian(image)
        std = std_from_mad(image, axis=None)
        background = image - median < bckgr_kappa_sigma * std
        ref_mask &= not background

    # Find centroid
    centroid_pixel = np.array(centroider(image.value**com_power, centroider_mask))
    centroid_world = cube.wcs.celestial.pixel_to_world(*centroid_pixel)
    if not full_output:
        return centroid_world
    else:
        return cube, image, centroid_pixel, centroid_world


def cross_correlate_images(
    list_of_images, oversample=100, median_filter_s=None, bckgr_kappa_sigma=None
):
    """
    Compute relative shifts via phase cross-correlation.

    Parameters
    ----------
    list_of_images : sequence of ndarray or Quantity
        First image is reference; shapes must match.
    oversample : int, default: 100
        Upsampling factor for subpixel accuracy.
    median_filter_s : int, optional
        Median filter size (converted to odd) applied to both reference and moving.
    bckgr_kappa_sigma : float, optional
        Background clipping threshold (sigma) for correlation masks.

    Returns
    -------
    results : list of [ndarray, float, float]
        Per moving image: [shift (drow, dcol), error, diffphase].
    ancillary_info : dict
        Standardised images and masks.

    Raises
    ------
    ImportError
        If scikit-image is not available.
    """
    try:
        from skimage.registration import phase_cross_correlation
    except Exception:
        raise ImportError(
            "For using the image crosscorrelation function you have to install scikit-image library"
        )

    ancillary_info = {}
    # Pre-processing
    ref = list_of_images[0]

    if median_filter_s is not None:
        median_filter_s = odd_int(median_filter_s)
        ancillary_info["median_filter_s"] = median_filter_s
        unit = ref.unit
        ref = median_filter(ref, size=median_filter_s) << unit
    # Standarise
    ref = robust_standarisation(ref, axis=None)

    ref_mask = np.isfinite(ref)
    # Add background clipping
    if bckgr_kappa_sigma is not None:
        std = std_from_mad(ref, axis=None)
        background = ref < bckgr_kappa_sigma * std
        ref_mask &= ~background
    # Set to none is all pixels are valid for performance
    if ref_mask.all():
        ref_mask = None
    # Store the information
    ancillary_info["standarised_ref"] = ref
    ancillary_info["ref_mask"] = ref_mask
    ancillary_info["standarised_mov"] = []
    ancillary_info["mov_mask"] = []

    results = []
    for i in range(len(list_of_images) - 1):
        # Preprocessing
        moving = list_of_images[i + 1]
        if median_filter_s is not None:
            unit = moving.unit
            moving = median_filter(moving, size=median_filter_s) << unit
        moving = robust_standarisation(moving, axis=None)
        # Masking
        mov_mask = np.isfinite(moving)

        if bckgr_kappa_sigma is not None:
            std = std_from_mad(moving, axis=None)
            background = moving < bckgr_kappa_sigma * std
            mov_mask &= ~background

        if mov_mask.all():
            mov_mask = None

        ancillary_info["standarised_mov"].append(moving)
        ancillary_info["mov_mask"].append(mov_mask)
        # The shift ordering is consistent with the input image shape
        # Shift (in pixels): register moving_image with reference_image.
        shift, error, diffphase = phase_cross_correlation(
            ref,
            moving,
            upsample_factor=oversample,
            space="real",
            reference_mask=ref_mask,
            moving_mask=mov_mask,
            disambiguate=False,
        )
        results.append([shift, error, diffphase])
    return results, ancillary_info


# Mr Krtxo \(ﾟ▽ﾟ)/
