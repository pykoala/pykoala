"""
Atmospheric extinction and refraction effects corrections.
"""

# =============================================================================
# Basics packages
# =============================================================================
from matplotlib import pyplot as plt
import numpy as np
import os
from astropy import units as u
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.centroids import centroid_2dg, centroid_com
from scipy.ndimage import median_filter, label, labeled_comprehension
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala import vprint
from pykoala.data_container import SpectraContainer, RSS
from pykoala.cubing import make_dummy_cube_from_rss
from pykoala.corrections.correction import CorrectionBase
from pykoala.ancillary import check_unit
from pykoala.utils.spectra import adaptive_spectra_snr_binning
from pykoala.utils.math import std_from_mad, poly_extrapolate_wrapper, nmad_filter


class AtmosphericExtCorrection(CorrectionBase):
    r"""Atmospheric Extinction Correction.

    This module accounts for the brightness reduction caused due to the absorption of 
    photons by the atmosphere.

    For a given observed (:math:`F_{obs}`) and intrinsic flux (:math:`F_{int}`), the extinction correction
    factor, :math:`C(\lambda)`, takes the form:

    .. math::
        F_{int}(\lambda) = C(\lambda) * F_{obs}(\lambda)
        
        C(\lambda) = 10^{0.4 \cdot airmass \cdot \eta(\lambda)}

    where :math:`\eta(\lambda)` corresponds to the wavelength-dependent extinction curve.

    Attributes
    ----------
    extinction_curve: np.ndarray, optional, default=None
        Atmospheric extinction curve.
    extinction_curve_wave: np.ndarray, optional, default=None
        Atmospheric extinction curve wavelength array.
    extinction_file: str
        Path to a text file containing a wavelength, and a extinction curve.
        If None, a default extinction model will be used, corresponding to the extinction curve at Siding Spring Observatory.
    """
    name = "AtmosphericExtinction"
    verbose = True
    default_extinction = os.path.join(os.path.dirname(__file__), '..', 'input_data',
                                      'observatory_extinction', 'ssoextinct.dat')
    def __init__(self,
                 extinction_curve=None,
                 extinction_curve_wave=None,
                 extinction_curve_file='unknown',
                 **correction_args):
        super().__init__(**correction_args)
        self.vprint("Initialising correction")

        # Initialise variables
        self.extinction_curve = extinction_curve
        self.extinction_curve_wave = check_unit(
            extinction_curve_wave, u.angstrom)
        self.extinction_curve_file = extinction_curve_file

    @classmethod
    def from_text_file(cls, path=None):
        r"""Initialise the Correction from a text file.
        
        Parameters
        ----------
        path : str, optional, default=``self.default_extinction``
            Path to the file containing the extinction curve. The first and 
            second columns of the file must contain the wavelength and the value
            of :math:`\eta(\lambda)`, respectively.
        
        Returns
        -------
        correction : AtmosphericExtCorrection
            An atmospheric extinction correction.
        """
        if path is None:
            path = cls.default_extinction
        wavelength, extinct = np.loadtxt(path, unpack=True)
        return cls(extinction_curve=extinct,
                   extinction_curve_wave=wavelength << u.angstrom,
                   extinction_curve_file=path)

    def extinction(self, wavelength : u.Quantity, airmass):
        """Compute the atmospheric extinction for a given airmass and wavelength.
        
        Parameters
        ----------
        wavelength: np.ndarray
            Input array of wavelengths where to estimate the extinction.
        airmass: float
            Target airmass at which the observation is performed.
        
        Returns
        -------
        extinction: np.ndarray
            Extinction at a given wavelength and airmass.
        """
        extinction_curve = np.interp(wavelength,
                                     self.extinction_curve_wave,
                                     self.extinction_curve,
                                     left=self.extinction_curve[0],
                                     right=self.extinction_curve[-1])
        return 10**(0.4 * airmass * extinction_curve)

    def apply(self, spectra_container, airmass=None):
        """Apply the Extinction Correction to a DataContainer.
        
        Parameters
        ----------
        airmass: float, optional
            If provided, the extinction will be computed using this value,
            otherwise the airmass stored at the `info` attribute will be used.
        
        Returns
        -------
        corrected_spectra_container: SpectraContainer
            SpectraContainer with the corrected intensity and variance.
        """
        assert isinstance(spectra_container, SpectraContainer)
        spectra_container_out = spectra_container.copy()
        if airmass is None:
            airmass = spectra_container.info.get("airmass", None)
            if airmass is None:
                raise ValueError("Airmass not provided")

        if self.extinction_curve is not None:
            self.vprint("Applying model-based extinction correction to"
                        f"Data Container ({airmass:.2f} airmass)")
            extinction = self.extinction(spectra_container_out.wavelength, airmass)
            comment = ("Atm. extinction file :" + os.path.basename(
                        self.extinction_curve_file)
                        + f"|airmass={airmass:.2f}")
        else:
            raise AttributeError("Extinction correction not provided")

        # Apply the correction
        extinction = np.expand_dims(extinction, axis=0)
        spectra_container_out.rss_intensity = (spectra_container_out.rss_intensity
                                            * extinction)
        spectra_container_out.rss_variance = (spectra_container_out.rss_variance
                                           * extinction**2)
        self.record_correction(spectra_container_out, status='applied',
                            comment=comment)
        return spectra_container_out


# =============================================================================
# Differential Atmospheric Refraction
# =============================================================================

class ADRCorrection(CorrectionBase):
    """
    Estimate and apply atmospheric differential refraction (ADR).

    The correction is empirical:
    1) Compute centroids as function of wavelength using several power weights.
    2) Median-combine centroids across powers.
    3) Fit polynomial models for RA and DEC offsets versus wavelength.
    4) Store the offsets or apply them using a spatial shifter.

    Parameters
    ----------
    max_adr : astropy Quantity
        Maximum allowed offset. Larger values are discarded. Default 0.5 arcsec.
    pol_deg : int
        Polynomial degree for the fit. Default 2.
    n_com_powers : int
        Number of center-of-mass powers to combine. Default 4.
    clip_sigma : float
        Sigma threshold for clipping outliers. Default 3.0.
    min_points : int
        Minimum valid points to attempt a fit. Default 20.
    store_key : str
        Key where offsets are stored in the container info. Default "adr_offsets".
    """

    name = "ADRCorrection"
    verbose = True

    def __init__(self,
                 max_adr=0.5 * u.arcsec,
                 pol_deg=2,
                 n_com_powers=4,
                 clip_sigma=3.0,
                 min_points=20,
                 store_key="adr_offsets",
                 **correction_args):
        super().__init__(**correction_args)
        self.max_adr = check_unit(max_adr, u.arcsec)
        self.pol_deg = int(pol_deg)
        self.n_com_powers = int(n_com_powers)
        self.clip_sigma = float(clip_sigma)
        self.min_points = int(min_points)
        self.store_key = str(store_key)
        self._poly_ra = None
        self._poly_dec = None

    def estimate(self, spectra_container, *,
                 find_source=False,
                 ref_coords=None,
                 quick_cube_pix_size=None,
                 target_bin_size=None,
                 target_bin_snr=None,
                 centroider="gauss",
                 median_filter_window=None,
                 sigma_clip=1.0,
                 plot=False):
        """
        Estimate ADR offsets from a spectra container.

        Parameters
        ----------
        spectra_container : SpectraContainer
            Input spectra container.
        plot : bool
            If True, generate a diagnostic plot.

        Returns
        -------
        tuple
            Returns polynomial fits for RA and DEC offsets.
        """
        self.vprint("Estimating ADR shift as function of wavelength")
        if not isinstance(spectra_container, SpectraContainer):
            raise TypeError("ADR can only be estimated using SpectraContainers")

        if isinstance(spectra_container, RSS):
            self.vprint(
                "Data provided in RSS format: creating a datacube"
            )
            if quick_cube_pix_size is None:
                quick_cube_pix_size = spectra_container.fibre_diameter / 2
            else:
                quick_cube_pix_size = check_unit(quick_cube_pix_size, u.arcsec)

            cube = make_dummy_cube_from_rss(spectra_container, quick_cube_pix_size)

        # Bin along spectral axis
        if target_bin_size is not None:
            target_bin_size = check_unit(target_bin_size, cube.wavelength.unit)
            bin_edges = np.arange(cube.wavelength[0].value,
                                  cube.wavelength[-1].value,
                                  target_bin_size.value) << target_bin_size.unit
            bin_edges = np.insert(bin_edges, bin_edges.size, cube.wavelength[-1])
            idx = np.searchsorted(cube.wavelength, bin_edges, side="right")
            bin_slices = [slice(low, up) for low, up in zip(idx[:-1], idx[1:])]
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            collapsed_snr = np.nansum(cube.rss_snr.value**2, axis=0)
            bin_snr = np.array([
                np.sqrt(np.nansum(collapsed_snr[s])) for s in bin_slices])
            # create bin_slices, bin_centers, bin_snr
        elif target_bin_snr is not None:
            collapsed_snr = np.nansum(cube.rss_snr**2, axis=0)
            bin_slices, centers, bin_snr = adaptive_spectra_snr_binning(
                cube.wavelength,
                collapsed_snr, target_snr=min_target_snr,
                max_bin_size=200 << u.AA)
        else:
            bin_slices = [slice(i, i + 1) for i in range(cube.wavelength.size)]
            centers = cube.wavelength.copy()
            bin_snr = np.sqrt(np.nansum(cube.rss_snr.value**2, axis=0))

        self.vprint(f"ADR will be estimated in {len(bin_slices)} wavelength bins")

        # Choose centroiding kind
        if centroider == "com":  # Moment-based
            centroider = centroid_com
        elif centroider == "gauss":  # Model fit
            centroider = centroid_2dg

        source_mask = None

        if find_source:
            self.vprint("Identifying sources on white image")
            white_image = cube.get_white_image()
            mean, median, std = sigma_clipped_stats(white_image, sigma=3.0)
            sources, num_sources = label(white_image > median + 2 * std)
            if num_sources == 0:
                self.vprint("No sources found")
            else:
                self.vprint(f"Sources found: {num_sources}")
                self.vprint("Selecting brightest source")
                lbl_id = np.arange(1, num_sources + 1)
                fluxes = labeled_comprehension(white_image, sources, lbl_id,
                                      np.nansum, out_dtype=float, default=0.0)
                brightest = np.argmax(fluxes)
                source_mask = sources == lbl_id[brightest]

                src_rows, src_cols = np.where(source_mask)
                sorted_pixels = np.argsort(white_image[src_rows, src_cols])[::-1]
                cum_image = np.nancumsum(white_image[source_mask][sorted_pixels])
                half_light_pixels = np.searchsorted(cum_image / cum_image[-1], 0.9)

                bright_src_rows = src_rows[sorted_pixels][:half_light_pixels]
                bright_src_cols = src_cols[sorted_pixels][:half_light_pixels]
                source_mask = np.zeros_like(source_mask)
                source_mask[bright_src_rows, bright_src_cols] = True
                n_val = np.count_nonzero(source_mask)
                self.vprint(f"Source mask contains {n_val} valid pixels")

        all_centroid_ra = np.full(len(bin_slices), fill_value=np.nan) << u.deg
        all_centroid_dec = np.full(len(bin_slices), fill_value=np.nan) << u.deg

        for ith, _slice in enumerate(bin_slices):
            median_image = np.nanmedian(cube.intensity[_slice], axis=0)
            if source_mask is None:
                background = np.nanmedian(median_image)
                std = std_from_mad(median_image, axis=None)
                backsub_image = np.abs(median_image - background) < std
                mask = backsub_image | ~np.isfinite(median_image) | (
                    median_image.value < 0)
            else:
                mask = ~source_mask
            if mask.all() or median_image[~mask].sum() == 0:
                continue
            try:
                centroid_pixel = centroider(median_image.value, mask=mask)
            except Exception as e:
                self.vprint("An error occurred during centroid estimation: skip")
                self.vprint(e)
                continue

            centroid_world = cube.wcs.celestial.pixel_to_world(
                *np.array(centroid_pixel))

            all_centroid_ra[ith] = centroid_world.ra
            all_centroid_dec[ith] = centroid_world.dec

        if median_filter_window is not None:
            self.vprint(
                f"Applying smoothing median filter (size={median_filter_window})")
            median_filter_extrap = poly_extrapolate_wrapper(
                median_filter, axis=-1, polyorder=1, pad_strategy="size"
                )
            median_ra =  median_filter(
                all_centroid_ra.value,
                size=median_filter_window) << all_centroid_ra.unit
            median_dec = median_filter(
                all_centroid_dec.value,
                size=median_filter_window) << all_centroid_dec.unit

            all_centroid_ra = median_ra
            all_centroid_dec = median_dec

        if ref_coords is None:
            ra_ref = np.nanmedian(all_centroid_ra)
            dec_def = np.nanmedian(all_centroid_dec)
        else:
            ra_ref = ref_coords.ra.to_value("deg")
            dec_ref = ref_coords.dec.to_value("deg")
        
        delta_ra = all_centroid_ra - ra_ref
        delta_dec = all_centroid_dec - dec_def
        mask = np.isfinite(delta_ra) & np.isfinite(delta_dec)

        if not mask.any():
            self.vprint("All RA/DEC ADR shifts contain non-finite values")
            self._poly_ra = np.poly1d([0.0])
            self._poly_dec = np.poly1d([0.0])
            return self._poly_ra, self._poly_dec, None

        # Fit along RA
        ra_polfit = np.polyfit(centers.to_value("angstrom")[mask],
                               delta_ra.to_value("arcsec")[mask],
                               deg=self.pol_deg, w=bin_snr[mask])
        self._poly_ra = np.poly1d(ra_polfit)

        # Fit along DEC
        dec_polfit = np.polyfit(centers.to_value("angstrom")[mask],
                                delta_dec.to_value("arcsec")[mask],
                                deg=self.pol_deg, w=bin_snr[mask])
        self._poly_dec = np.poly1d(dec_polfit)

        fig = None
        if plot:
            self.vprint("Generating ADR QC plot")
            ra_pix_scale, dec_pix_scale = proj_plane_pixel_scales(cube.wcs.celestial)
            fig = self._make_plot(cube.wavelength, centers, delta_ra, delta_dec,
            ra_pix_scale * 3600, dec_pix_scale * 3600)

        return self._poly_ra, self._poly_dec, fig

    def predict(self, wavelength):
        """
        Predict offsets at given wavelengths.
        Returns RA and DEC offsets in arcsec.
        """
        if self._poly_ra is None or self._poly_dec is None:
            raise RuntimeError("ADR model not estimated. Call estimate() first.")
        lam = wavelength.to_value("angstrom")
        dra = self._poly_ra(lam) << u.arcsec
        ddec = self._poly_dec(lam) << u.arcsec
        return dra, ddec

    def apply(self, spectra_container, copy=True):
        """
        Apply ADR correction.

        Parameters
        ----------
        spectra_container : SpectraContainer
            Input spectra container.
        
        Returns
        -------
        corrected_spectra_container : SpectraContainer
            Corrected spectra container.
        """
        assert isinstance(spectra_container, SpectraContainer)
        wave = spectra_container.wavelength
        dra, ddec = self.predict(wave)

        comment = f"ADR fit degree={self.pol_deg}, nCOM={self.n_com_powers}, clip_sigma={self.clip_sigma}"
        if copy:
            out = spectra_container.copy()
        else:
            out = spectra_container
        out.info[self.store_key] = {
            "dra_arcsec": dra.to("arcsec"),
            "ddec_arcsec": ddec.to("arcsec"),
        }
        self.record_correction(out, status="applied", comment=comment)
        return out

    def _make_plot(self, wave, wave_centers, delta_ra, delta_dec, ra_pix_scale,
                   dec_pix_scale):
        fig, ax = plt.subplots(constrained_layout=True)

        ra_poly = self._poly_ra(wave.to_value("AA"))
        dec_poly = self._poly_dec(wave.to_value("AA"))

        ax.set_title("ADR distortion")
        ax.step(wave_centers, delta_ra.to_value("arcsec"),
                where="mid", c="tomato", lw=1.0, label="RA shift")
        ax.plot(wave, ra_poly, c="gold",
                 lw=2.0, label="Poly fit (RA)")
        ax.step(wave_centers, delta_dec.to_value("arcsec"),
                where="mid", c="b", lw=1.0, label="DEC shift")
        ax.plot(wave, dec_poly, c="cyan",
                 lw=2.0, label="Poly fit (DEC)")
        max_y = np.nanmax((ra_poly.max(), dec_poly.max()))
        min_y = np.nanmin((ra_poly.min(), dec_poly.min()))

        ax.set_ylim(min_y * 0.9, max_y * 1.1)
        ax.set_ylabel("Centroid shift (arcsec)")
        ax.set_xlabel("Wavelength (Angstrom)")
        ax.legend(title=f"Cube pix scale ({ra_pix_scale:.2f}, {dec_pix_scale:.2f}) arcsec",
                  fontsize="x-small")

        #plt.close()
        return fig

