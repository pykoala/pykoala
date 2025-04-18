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

    For a given observed (:math:`F_{obs}`) and intrinsic flux (:math:`F_{int}`), the extinction
    :math:`E` takes the form:

    .. math::
        F_{int}(\lambda) = E(\lambda) * F_{obs}(\lambda)
        
        E(\lambda) = 10^{0.4 \cdot airmass \cdot \eta(\lambda)}

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
                                        self.extinction_curve)
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
            airmass = spectra_container.info['airmass']

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
# TODO: refactor ADR by DAR. Create a class that performs this correction.
def get_adr(spectra_container : SpectraContainer, max_adr=0.5, pol_deg=2,
            plot=False):
    """Computes the ADR for a given DataContainer.
    
    This method computes the spatial shift as function of wavelength that a
    chromatic source light experiments due to the Atmospheric Differential
    Refraction (ADR).

    Parameters
    ----------
    spectra_container: :class:`SpectraContainer`
        Target SpectraContainer.
    max_adr : float, optional, default=0.5
        Maxium ADR correction expressed in arcseconds to prevent unreliable
        results when analyzing low-SNR data.
    pol_deg : int, optional, default=2
        Polynomial order to model the dependance of the spatial shift as function
        of wavelength.

    Returns
    -------
    - ra_polfit: np.poly1d
        Callable that returns the RA offset in arcseconds as function of wavelength.
    - dec_polfit: np.poly1d
        Callable that returns the DEC offset in arcseconds as function of wavelength.
    """
    # Centre of mass using multiple power of the intensity
    max_adr = check_unit(max_adr, u.arcsec)
    com = []
    for i in range(1, 5):
        com.append(spectra_container.get_centre_of_mass(power=i))
    com = np.array(com) << com[0][0].unit
    com -= np.nanmedian(com, axis=2)[:, :, np.newaxis]
    median_com = np.nanmedian(
        com, axis=0) - np.nanmedian(com, axis=(0, 2))[:, np.newaxis]
    median_com[np.abs(median_com) > max_adr] = np.nan

    finite_mask = np.isfinite(median_com[0])

    if finite_mask.any():
        p_x = np.polyfit(
            spectra_container.wavelength[finite_mask].to_value("angstrom"),
            median_com[0][finite_mask].to_value("arcsec"), deg=pol_deg)
        polfit_x = np.poly1d(p_x)(
            spectra_container.wavelength.to_value("angstrom")) << u.arcsec
    else:
        vprint("[ADR] ERROR: Could not compute ADR-x, all NaN")
        polfit_x = np.zeros(spectra_container.wavelength.size) << u.arcsec

    finite_mask = np.isfinite(median_com[1])
    if finite_mask.any():
        p_y = np.polyfit(
            spectra_container.wavelength[finite_mask].to_value("angstrom"),
            median_com[1][finite_mask].to_value("arcsec"), deg=pol_deg)
        polfit_y = np.poly1d(p_y)(
            spectra_container.wavelength.to_value("angstrom")) << u.arcsec
    else:
        vprint("[ADR] ERROR: Could not compute ADR-y, all NaN")
        polfit_y = np.zeros_like(spectra_container.wavelength)
    

    if plot:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        ax.plot(spectra_container.wavelength, com[0, 0].to("arcsec"),
                label='COM', lw=0.7)
        ax.plot(spectra_container.wavelength, com[1, 0], label='COM2', lw=0.7)
        ax.plot(spectra_container.wavelength, com[2, 0], label='COM3', lw=0.7)
        ax.plot(spectra_container.wavelength, com[3, 0], label='COM4', lw=0.7)
        ax.plot(spectra_container.wavelength, median_com[0], c='k',
                label='Median', lw=0.7)
        ax.plot(spectra_container.wavelength, polfit_x, c='fuchsia',
                label=f'pol. fit (deg={pol_deg})', lw=0.7)
        ax.set_ylim(-max_adr, max_adr)
        ax.legend(ncol=2, bbox_to_anchor=(0.5, 1), loc='lower center')
        ax.set_ylabel(r'$\Delta RA$ (arcsec)')
        ax.set_xlabel(r'$\lambda$')

        ax = fig.add_subplot(122)
        ax.plot(spectra_container.wavelength, com[0, 1].to("arcsec"),
                label='COM', lw=0.7)
        ax.plot(spectra_container.wavelength, com[1, 1], label='COM2', lw=0.7)
        ax.plot(spectra_container.wavelength, com[2, 1], label='COM3', lw=0.7)
        ax.plot(spectra_container.wavelength, com[3, 1], label='COM4', lw=0.7)
        ax.plot(spectra_container.wavelength, median_com[1], c='k',
                label='Median', lw=0.7)
        ax.plot(spectra_container.wavelength, polfit_y, c='fuchsia',
                label=f'pol. fit (deg={pol_deg})', lw=0.7)
        ax.set_ylim(-max_adr, max_adr)
        ax.legend(ncol=2, bbox_to_anchor=(0.5, 1), loc='lower center')
        ax.set_ylabel(r'$\Delta DEC$ (arcsec)')
        ax.set_xlabel(r'$\lambda$')

        fig.subplots_adjust(wspace=0.3)
        plt.close(fig)
        return polfit_x, polfit_y, fig
    return polfit_x, polfit_y
