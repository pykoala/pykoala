"""
This module contains... #TODO
"""
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
from koala.corrections.correction import CorrectionBase


# =============================================================================
class AtmosphericExtinction(CorrectionBase):
    """Atmospheric Extinction Correction.

    # TODO: Fill doc.
    """
    name = "AtmosphericExtinction"
    target = RSS

    def __init__(self, extinction_correction=None, extinction_file=None, observatory_extinction=None, airmass=None):
        print("[AtmosphericExtinction] Initialising Atm ext. correction model.")
        # Initialise variables
        self.extinction_correction_model = None
        self.extinction_correction = extinction_correction
        self.extinction_file = extinction_file
        self.observatory_extinction = observatory_extinction
        self.airmass = airmass

        if extinction_correction is None and extinction_file is None:
            print("[AtmosphericExtinction] No extinction provided")
            self.get_atmospheric_extinction(self.airmass, self.observatory_extinction)

    def get_atmospheric_extinction(self, airmass=None, path_to_extinction=None):
        """..."""
        if airmass is None:
            airmass = 1.0
            self.airmass = airmass
        if path_to_extinction is None:
            path_to_extinction = os.path.join(os.path.dirname(__file__),
                                              '../input_data/observatory_extinction/ssoextinct.dat')
            self.observatory_extinction = path_to_extinction
        print("[AtmosphericExtinction] Computing extinction at airmass {:.1f} based on model:\n    {}".format(
            airmass, path_to_extinction))
        wave, extinction_curve = np.loadtxt(path_to_extinction, unpack=True)
        extinction = 10 ** (0.4 * airmass * extinction_curve)
        # Make fit
        self.extinction_correction_model = interpolate.interp1d(wave, extinction,
                                                                fill_value=(extinction[0], extinction[-1]))

    def apply(self, rss, plot=False):
        """"""
        super().check_target(rss)
        rss_out = copy.deepcopy(rss)
        if self.extinction_correction is None:
            print("[AtmosphericExtinction] Applying model-based extinction correction to RSS file")
            extinction_correction = self.extinction_correction_model(rss_out.wavelength)
            comment = ' '.join(["- Data corrected for extinction using file :",
                                self.observatory_extinction, "Average airmass =", str(self.airmass)])
        else:
            extinction_correction = self.extinction_correction
            comment = "- Data corrected for extinction using provided data"
        rss_out.intensity_corrected *= extinction_correction[np.newaxis, :]
        rss_out.variance_corrected *= extinction_correction[np.newaxis, :] ** 2
        rss_out.log[self.name] = comment
        return rss_out


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
    p_x = np.polyfit(data_container.wavelength[np.isfinite(median_com[0])],
                     median_com[0][np.isfinite(median_com[0])], deg=pol_deg)
    polfit_x = np.poly1d(p_x)(data_container.wavelength)

    p_y = np.polyfit(data_container.wavelength[np.isfinite(median_com[1])],
                     median_com[1][np.isfinite(median_com[1])], deg=pol_deg)
    polfit_y = np.poly1d(p_y)(data_container.wavelength)

    if plot:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        ax.plot(data_container.wavelength, com[0, 0], label='P=1')
        ax.plot(data_container.wavelength, com[1, 0], label='P=2')
        ax.plot(data_container.wavelength, com[2, 0], label='P=3')
        ax.plot(data_container.wavelength, com[3, 0], label='P=4')
        ax.plot(data_container.wavelength, median_com[0], c='k', label='Median')
        ax.plot(data_container.wavelength, polfit_x, c='fuchsia', label='Fit')
        ax.set_ylim(-max_adr, max_adr)
        ax.legend()
        ax.set_ylabel(r'$\Delta RA$ (")')
        ax.set_xlabel(r'$\lambda$')

        ax = fig.add_subplot(122)
        ax.plot(data_container.wavelength, com[0, 1], label='P=1')
        ax.plot(data_container.wavelength, com[1, 1], label='P=2')
        ax.plot(data_container.wavelength, com[2, 1], label='P=3')
        ax.plot(data_container.wavelength, com[3, 1], label='P=4')
        ax.plot(data_container.wavelength, median_com[1], c='k', label='Median')
        ax.plot(data_container.wavelength, polfit_y, c='fuchsia', label='Fit')
        ax.set_ylim(-max_adr, max_adr)
        ax.legend()
        ax.set_ylabel(r'$\Delta DEC$ (")')
        ax.set_xlabel(r'$\lambda$')
        plt.close(fig)
        return polfit_x, polfit_y, fig
    return polfit_x, polfit_y

# def ADR_correction(cube, RSS, plot=True, force_ADR=False, method="new", remove_spaxels_not_fully_covered=True,
#                    jump=-1, warnings=False, verbose=True):
#     """
#     Corrects for Atmospheric Differential Refraction (ADR)

#     Parameters
#     ----------
#     RSS : File/Object created with KOALA_RSS
#         This is the file that has the raw stacked spectra.
#     plot : Boolean, optional
#         If True generates and shows the plots. The default is True.
#     force_ADR : Boolean, optional
#         If True will correct for ADR even considoring a small correction. The default is False.
#     method : String, optional
#         DESCRIPTION. The default is "new". #TODO
#     remove_spaxels_not_fully_covered : Boolean, optional
#         DESCRIPTION. The default is True. #TODO
#     jump : Integer, optional
#         If a positive number partitions the wavelengths with step size jump, if -1 will not partition. The default is -1.
#     warnings : Boolean, optional
#         If True will show any problems that arose, else skipped. The default is False.
#     verbose : Boolean, optional
#         Print results. The default is True.

#     Returns
#     -------
#     None.

#     """

#     # Check if this is a self.combined cube or a self
#     try:
#         _x_ = np.nanmedian(self.combined_cube.data)
#         if _x_ > 0:
#             cubo = self.combined_cube
#             # data_ = np.zeros_like(cubo.weighted_I)
#             method = "old"
#             # is_combined_cube=True
#     except Exception:
#         cubo = self

#     # Check if ADR is needed (unless forced)...
#     total_ADR = np.sqrt(cubo.ADR_x_max ** 2 + cubo.ADR_y_max ** 2)

#     cubo.adrcor = True
#     if total_ADR < cubo.pixel_size_arcsec * 0.1:  # Not needed if correction < 10 % pixel size
#         if verbose:
#             print("\n> Atmospheric Differential Refraction (ADR) correction is NOT needed.")
#             print(
#                 '  The computed max ADR value, {:.3f}",  is smaller than 10% the pixel size of {:.2f} arcsec'.format(
#                     total_ADR, cubo.pixel_size_arcsec))
#         cubo.adrcor = False
#         if force_ADR:
#             cubo.adrcor = True
#             if verbose: print('  However we proceed to do the ADR correction as indicated: "force_ADR = True" ...')

#     if cubo.adrcor:
#         cubo.history.append("- Correcting for Atmospheric Differential Refraction (ADR) using:")
#         cubo.history.append("  ADR_x_fit = " + np.str(cubo.ADR_x_fit))
#         cubo.history.append("  ADR_y_fit = " + np.str(cubo.ADR_y_fit))
#         cubo.history.append("  Residua in RA  = " + np.str(np.round(cubo.ADR_x_residua, 3)) + '" ')
#         cubo.history.append("  Residua in Dec = " + np.str(np.round(cubo.ADR_y_residua, 3)) + '" ')
#         cubo.history.append("  Total residua  = " + np.str(np.round(cubo.ADR_total_residua, 3)) + '" ')

#         # New procedure 2nd April 2020
#         else:
#             if verbose:
#                 print("\n  Using the NEW method (building the cube including the ADR offsets)...")
#                 print("  Creating new cube considering the median value each ", jump, " lambdas...")
#             cubo.adrcor = True
#             cubo.data = cubo.build_cube(jump=jump, RSS=RSS)
#             cubo.get_integrated_map()
#             cubo.history.append("- New cube built considering the ADR correction using jump = " + np.str(jump))

#             # Now remove spaxels with not full wavelength if requested
#         if remove_spaxels_not_fully_covered == True:
#             if verbose: print(
#                 "\n> Removing spaxels that are not fully covered in wavelength in the valid range...")  # Barr

#             _mask_ = cubo.integrated_map / cubo.integrated_map
#             for w in range(cubo.n_wave):
#                 cubo.data[w] = cubo.data[w] * _mask_
#             cubo.history.append(
#                 "  Spaxels that are not fully covered in wavelength in the valid range have been removed")

#     else:
#         if verbose: print(" NOTHING APPLIED !!!")
