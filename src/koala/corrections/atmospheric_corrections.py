"""
This module contains... #TODO
"""
# =============================================================================
# Basics packages
# =============================================================================
from scipy import interpolate
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
from koala.ancillary import vprint
from koala.rss import RSS, detect_edge
from koala.corrections.correction import Correction

# =============================================================================
class AtmosphericExtinction(Correction):
    """#TODO"""

    def __init__(self, extinction_correction=None, extinction_file=None, observatory_extinction=None, airmass=None):
        """#TODO"""
        print("[AtmosphericExtinction] Initialising Atm ext. correction model.")
        super().__init__(target_class=RSS)
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
        rss_out.variance_corrected *= extinction_correction[np.newaxis, :]**2
        rss_out.log['extinction'] = comment
        return rss_out

        # TODO: Include some plotting
        # if plot:
        #     valid_wave_min, min_index, valid_wave_max, max_index = detect_edge(rss)
        #     cinco_por_ciento = 0.05 * (np.max(extinction_correction) - np.min(extinction_correction))
        #     plot_plot(extinction_curve_wavelenghts, extinction_corrected_airmass, xmin=np.min(rss_out.wavelength),
        #               xmax=np.max(rss_out.wavelength), ymin=np.min(extinction_correction) - cinco_por_ciento,
        #               ymax=np.max(extinction_correction) - cinco_por_ciento,
        #               vlines=[valid_wave_min, valid_wave_max],
        #               ptitle='Correction for extinction using airmass = ' + str(np.round(airmass, 3)),
        #               xlabel="Wavelength [$\mathrm{\AA}$]", ylabel="Flux correction", fig_size=12,
        #               statistics=False)

# =============================================================================
# Atmospheric Differential Refraction
# =============================================================================

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

