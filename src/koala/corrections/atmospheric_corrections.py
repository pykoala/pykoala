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
from koala.rss import RSS
from koala.cubing import Cube
from koala.corrections.correction import CorrectionBase


# =============================================================================
class AtmosphericExtinction(CorrectionBase):
    """Atmospheric Extinction Correction.

    Attributes
    ----------
    - extinction_correction: (np.ndarray, optional, default=None) Atmospheric extinction
    correction function.
    - extinction_file: ()
    - observatory_extinction: ()
    - airmass: ()
    """
    name = "AtmosphericExtinction"
    verbose = False

    def __init__(self, extinction_correction=None, extinction_file=None,
                 observatory_extinction=None, airmass=None):
        self.corr_print("Initialising Atm ext. correction model.")
        # Initialise variables
        self.extinction_correction_model = None
        self.extinction_correction = extinction_correction
        self.extinction_file = extinction_file
        self.observatory_extinction = observatory_extinction
        self.airmass = airmass

        if extinction_correction is None and extinction_file is None:
            self.corr_print("No extinction provided")
            self.get_atmospheric_extinction(self.airmass, self.observatory_extinction)
        else:
            # TODO
            raise NotImplementedError("Extra extinction files are not implemented!")

    def get_atmospheric_extinction(self, airmass=None, path_to_extinction=None):
        """Create an atmospheric extinction model for a given airmass and extinction model.

        Given a sky extinction model, this method creates an extinction model (callable function)
        in terms as extinction(wavelength)

        Parameters
        ----------
        airmass: (float, default=1.0)
        path_to_extinction: (str, default=ssoextinct.dat) path to sky extinction curve model.

        Returns
        -------

        """
        if airmass is None:
            airmass = 1.0
            self.airmass = airmass
        if path_to_extinction is None:
            path_to_extinction = os.path.join(os.path.dirname(__file__),
                                              '../input_data/observatory_extinction/ssoextinct.dat')
            self.observatory_extinction = path_to_extinction
        # Compute the extinction
        self.corr_print("Computing extinction at airmass {:.1f} based on model:\n    {}".format(
            airmass, path_to_extinction))
        wave, extinction_curve = np.loadtxt(path_to_extinction, unpack=True)
        extinction = 10 ** (0.4 * airmass * extinction_curve)
        # Create the callable function
        self.extinction_correction_model = interpolate.interp1d(
            wave, extinction,
            fill_value=(extinction[0], extinction[-1]))

    def apply(self, data_container, plot=False):
        """Apply the Extinction Correction to a DataContainer"""
        data_container_out = copy.deepcopy(data_container)
        if self.extinction_correction is None:
            self.corr_print("Applying model-based extinction correction to RSS file")
            extinction_correction = self.extinction_correction_model(data_container_out.wavelength
                                                                     ).copy()
            comment = ' '.join(["- Data corrected for extinction using file :",
                                self.observatory_extinction, "Average airmass =", str(self.airmass)])
        else:
            extinction_correction = self.extinction_correction.copy()
            comment = "- Data corrected for extinction with user-provided function"
        # Apply the correction
        # TODO: REFACTOR
        if type(data_container) is Cube:
            extinction_correction = np.expand_dims(
                    extinction_correction, axis=tuple(
                        range(1, data_container.intensity_corrected.ndim)))
        elif type(data_container) is RSS:
            extinction_correction = np.expand_dims(
                extinction_correction, axis=0)

        data_container_out.intensity_corrected *= extinction_correction
        data_container_out.variance_corrected *= extinction_correction**2
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
    com = np.array(com)
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
