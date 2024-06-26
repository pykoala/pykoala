# =============================================================================
# Basics packages
# =============================================================================
from scipy import interpolate
from matplotlib import pyplot as plt
import numpy as np
import copy
import os

# =============================================================================
# Astropy and associated packages
# =============================================================================

# =============================================================================
# KOALA packages
# =============================================================================
# Modular
# from koala.ancillary import vprint
from pykoala.rss import RSS
from pykoala.cubing import Cube
from pykoala.corrections.correction import CorrectionBase


# =============================================================================
class AtmosphericExtCorrection(CorrectionBase):
    """Atmospheric Extinction Correction.

    This module accounts for the brightness reduction caused due to the absorption of 
    photons by the atmosphere.

    For a given observed (F_obs) and intrinsic flux F_int, the extinction E takes the form:
    F_obs(lambda) = E(lambda) * F_int(lambda) = 10^(0.4 * airmass * eta(lambda))
    where eta corresponds to the extinction curve that depends on the frequency.

    Attributes
    ----------
    - extinction_correction: (np.ndarray, optional, default=None)
        Atmospheric extinction correction function.
    - extinction_correction_wave: (np.ndarray, optional, default=None)
        Atmospheric extinction correction function wavelength array.
    - extinction_file: (str, default=input_data/observatory_extinction/ssoextinct.dat)
        Path to a text file containing a wavelength, and a extinction - E(lambda) - column respectively.
        If None, a default extinction model will be used, corresponding to the extinction curve at Siding Spring Observatory.
    - verbose: (bool, default=False)
    """
    name = "AtmosphericExtinction"
    verbose = False
    default_extinction = os.path.join(os.path.dirname(__file__), '..', 'input_data',
                                      'observatory_extinction', 'ssoextinct.dat')
    def __init__(self,
                 extinction_correction=None,
                 extinction_correction_wave=None,
                 extinction_file=default_extinction,
                 verbose=False):
        super().__init__(verbose)
        self.verbose = verbose
        self.corr_print("Initialising Atm ext. correction model.")

        # Initialise variables
        self.extinction_correction = extinction_correction
        self.extinction_correction_wave = extinction_correction_wave
        self.extinction_file = extinction_file

        if self.extinction_correction is None:
            self.model_from_file = True
            self.load_atmospheric_extinction(self.extinction_file)
        else:
            self.model_from_file = True

    def load_atmospheric_extinction(self, path):
        """Load data from text file."""
        self.corr_print(f"Loading atmospheric extinction model from:\n {path}")
        self.extinction_correction_wave, self.extinction_correction = np.loadtxt(path, unpack=True)

    def extinction(self, wavelength, airmass):
        """Compute the atmospheric extinction for a given airmass"""
        extinction_correction = np.interp(wavelength, self.extinction_correction_wave, self.extinction_correction)
        return 10**(0.4 * airmass * extinction_correction)

    def apply(self, data_container, force_airmass=None, plot=False):
        """Apply the Extinction Correction to a DataContainer"""
        data_container_out = copy.deepcopy(data_container)
        if force_airmass is None:
            airmass = data_container.info['airmass']
        else:
            airmass = force_airmass

        if self.extinction_correction is not None:
            self.corr_print(f"Applying model-based extinction correction to Data Container ({airmass:.2f} airmass)")
            extinction = self.extinction(data_container_out.wavelength, airmass)
            if self.model_from_file:
                comment = ' '.join(["- Data corrected for extinction using file :",
                                    self.extinction_file, f"| airmass={airmass:.2f}"])
            else:
                comment = ' '.join(["- Data corrected for extinction using user-provided model",
                                    f"| airmass={airmass:.2f}"])
        else:
            raise AttributeError("Extinction correction not provided")

        # Apply the correction
        if type(data_container) is Cube:
            extinction = np.expand_dims(extinction, axis=tuple(range(1, data_container.intensity.ndim)))
        elif type(data_container) is RSS:
            extinction = np.expand_dims(extinction, axis=0)

        data_container_out.intensity *= extinction
        data_container_out.variance *= extinction**2
        self.log_correction(data_container_out, status='applied',
                            comment=comment)
        return data_container_out


# =============================================================================
# Atmospheric Differential Refraction
# =============================================================================

def get_adr(data_container, max_adr=0.5, pol_deg=2, plot=False):
    """Computes the ADR for a given DataContainer."""
    # Centre of mass using multiple power of the intensity
    com = []
    for i in range(1, 5):
        com.append(data_container.get_centre_of_mass(power=i))
    com = np.array(com) * 3600
    com -= np.nanmedian(com, axis=2)[:, :, np.newaxis]
    median_com = np.nanmedian(
        com, axis=0) - np.nanmedian(com, axis=(0, 2))[:, np.newaxis]
    median_com[np.abs(median_com) > max_adr] = np.nan

    finite_mask = np.isfinite(median_com[0])
    if finite_mask.any():
        p_x = np.polyfit(data_container.wavelength[finite_mask],
                         median_com[0][finite_mask], deg=pol_deg)
        polfit_x = np.poly1d(p_x)(data_container.wavelength)
    else:
        print("[ADR] ERROR: Could not compute ADR-x, all NaN")
        polfit_x = np.zeros_like(data_container.wavelength)

    finite_mask = np.isfinite(median_com[1])
    if finite_mask.any():
        p_y = np.polyfit(data_container.wavelength[finite_mask],
                         median_com[1][finite_mask], deg=pol_deg)
        polfit_y = np.poly1d(p_y)(data_container.wavelength)
    else:
        print("[ADR] ERROR: Could not compute ADR-y, all NaN")
        polfit_y = np.zeros_like(data_container.wavelength)
    

    if plot:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        ax.plot(data_container.wavelength, com[0, 0], label='COM', lw=0.7)
        ax.plot(data_container.wavelength, com[1, 0], label='COM2', lw=0.7)
        ax.plot(data_container.wavelength, com[2, 0], label='COM3', lw=0.7)
        ax.plot(data_container.wavelength, com[3, 0], label='COM4', lw=0.7)
        ax.plot(data_container.wavelength, median_com[0], c='k', label='Median', lw=0.7)
        ax.plot(data_container.wavelength, polfit_x, c='fuchsia', label=f'pol. fit (deg={pol_deg})', lw=0.7)
        ax.set_ylim(-max_adr, max_adr)
        ax.legend(ncol=2, bbox_to_anchor=(0.5, 1), loc='lower center')
        ax.set_ylabel(r'$\Delta RA$ (arcsec)')
        ax.set_xlabel(r'$\lambda$')

        ax = fig.add_subplot(122)
        ax.plot(data_container.wavelength, com[0, 1], label='COM', lw=0.7)
        ax.plot(data_container.wavelength, com[1, 1], label='COM2', lw=0.7)
        ax.plot(data_container.wavelength, com[2, 1], label='COM3', lw=0.7)
        ax.plot(data_container.wavelength, com[3, 1], label='COM4', lw=0.7)
        ax.plot(data_container.wavelength, median_com[1], c='k', label='Median', lw=0.7)
        ax.plot(data_container.wavelength, polfit_y, c='fuchsia', label=f'pol. fit (deg={pol_deg})', lw=0.7)
        ax.set_ylim(-max_adr, max_adr)
        ax.legend(ncol=2, bbox_to_anchor=(0.5, 1), loc='lower center')
        ax.set_ylabel(r'$\Delta DEC$ (arcsec)')
        ax.set_xlabel(r'$\lambda$')

        fig.subplots_adjust(wspace=0.3)
        plt.close(fig)
        return polfit_x, polfit_y, fig
    return polfit_x, polfit_y
