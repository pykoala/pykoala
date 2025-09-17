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
from astropy.stats import sigma_clip
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala import vprint
from pykoala.data_container import SpectraContainer
from pykoala.corrections.correction import CorrectionBase
from pykoala.ancillary import check_unit


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

    def estimate(self, spectra_container, plot=False):
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
        assert isinstance(spectra_container, SpectraContainer)
        wave = spectra_container.wavelength.to_value("angstrom")

        com_tracks = []
        for p in range(1, self.n_com_powers + 1):
            com = spectra_container.get_centre_of_mass(power=p)
            com_tracks.append(com)
        com_tracks = np.array(com_tracks) << com_tracks[0][0].unit
        com_tracks -= np.nanmedian(com_tracks, axis=2)[:, :, np.newaxis]
        median_com = np.nanmedian(com_tracks, axis=0)

        for k in (0, 1):
            over = np.abs(median_com[k]) > self.max_adr
            median_com[k][over] = np.nan

        dra = median_com[0].to_value("arcsec")
        ddec = median_com[1].to_value("arcsec")
        dra_clip = sigma_clip(dra, sigma=self.clip_sigma, masked=True)
        ddec_clip = sigma_clip(ddec, sigma=self.clip_sigma, masked=True)

        m_ra = (~dra_clip.mask) & np.isfinite(dra)
        m_dec = (~ddec_clip.mask) & np.isfinite(ddec)

        if np.count_nonzero(m_ra) >= self.min_points:
            self._poly_ra = np.poly1d(np.polyfit(wave[m_ra], dra[m_ra], deg=self.pol_deg))
        else:
            vprint("ADR WARNING: insufficient points for RA fit, using zeros")
            self._poly_ra = np.poly1d([0.0])

        if np.count_nonzero(m_dec) >= self.min_points:
            self._poly_dec = np.poly1d(np.polyfit(wave[m_dec], ddec[m_dec], deg=self.pol_deg))
        else:
            vprint("ADR WARNING: insufficient points for DEC fit, using zeros")
            self._poly_dec = np.poly1d([0.0])

        fig = None
        if plot:
            fig = self._make_plot(wave, com_tracks, median_com, dra, ddec)
            #plt.close(fig)

        return self._poly_ra, self._poly_dec, fig

    def predict(self, wavelength):
        """
        Predict offsets at given wavelengths.
        Returns RA and DEC offsets in arcsec.
        """
        if self._poly_ra is None or self._poly_dec is None:
            raise RuntimeError("ADR model not estimated. Call estimate() first.")
        lam = wavelength.to_value("angstrom")
        dra = self._poly_ra(lam) * u.arcsec
        ddec = self._poly_dec(lam) * u.arcsec
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

    def _make_plot(self, wave, com_tracks, median_com, dra, ddec):
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        for i in range(com_tracks.shape[0]):
            ax1.plot(wave, com_tracks[i, 0].to_value("arcsec"), lw=0.7, alpha=0.6)
        ax1.plot(wave, median_com[0].to_value("arcsec"), c="k", lw=1.0, label="Median")
        ax1.plot(wave, self._poly_ra(wave), c="fuchsia", lw=1.0, label="Poly fit")
        ax1.set_ylim(-self.max_adr.value, self.max_adr.value)
        ax1.set_ylabel("Delta RA (arcsec)")
        ax1.set_xlabel("Wavelength (Angstrom)")
        ax1.legend()

        ax2 = fig.add_subplot(122)
        for i in range(com_tracks.shape[0]):
            ax2.plot(wave, com_tracks[i, 1].to_value("arcsec"), lw=0.7, alpha=0.6)
        ax2.plot(wave, median_com[1].to_value("arcsec"), c="k", lw=1.0, label="Median")
        ax2.plot(wave, self._poly_dec(wave), c="fuchsia", lw=1.0, label="Poly fit")
        ax2.set_ylim(-self.max_adr.value, self.max_adr.value)
        ax2.set_ylabel("Delta DEC (arcsec)")
        ax2.set_xlabel("Wavelength (Angstrom)")
        ax2.legend()

        fig.subplots_adjust(wspace=0.3)
        return fig

