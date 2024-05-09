# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
#import os

# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits
# =============================================================================
# KOALA packages
# =============================================================================
# Modular
from pykoala.ancillary import vprint
from pykoala.data_container import DataContainer
from pykoala.plotting.plot_plot import plot_plot

# =============================================================================
# RSS CLASS
# =============================================================================


class RSS(DataContainer):
    """
    Data Container class for row-stacked spectra (RSS).

    Attributes
    ----------
    intensity: numpy.ndarray(float)
        Intensity :math:``I_lambda``.
        Axis 0 corresponds to spectral dimension
        Axis 1 Corresponds to fibre ID
    variance: numpy.ndarray(float)
        Variance :math:`sigma^2_lambda`.
        (note the square in the definition of the variance). Must have the
        same dimensions as `intensity`
    wavelength: numpy.ndarray(float)
        Wavelength, expressed in Angstrom. It must have the same dimensions
        as `intensity` along axis 0.
    info : dict
        Dictionary containing RSS information.
        Important dictionary keys:
            info['fib_ra'] - original RA fiber position
            info['fib_dec'] - original DEC fiber position
            info['exptime'] - exposure time in seconds
            info['airmass'] - mean airmass during observation
            info['name'] - Name reference
    log : dict
        Dictionary containing a log of the processes applied on the rss.   
    """
    def __init__(self,
                 intensity=None,
                 wavelength=None,
                 variance=None,
                 info=None,
                 mask = None,
                 log=None,
                 ):

        # Intialise base class
        super().__init__(intensity=intensity, variance=variance, info=info, mask=mask, log=log)
        # Specific RSS attributes
        self.wavelength = wavelength

    def get_centre_of_mass(self, wavelength_step=1, stat=np.nanmedian, power=1.0):
        """Compute the center of mass (COM) based on the RSS fibre positions

        Parameters
        ----------
        wavelength_step: int, default=1
            Number of wavelength points to consider for averaging the COM. When setting it to 1 it will average over
            all wavelength points.
        stat: function, default=np.median
            Function to compute the COM over each wavelength range.
        power: float (default=1.0)
            Power the intensity to compute the COM.
        Returns
        -------
        x_com: np.array(float)
            Array containing the COM in the x-axis (RA, columns).
        y_com: np.array(float)
            Array containing the COM in the y-axis (DEC, rows).
        """
        ra = self.info["fib_ra"]
        dec = self.info["fib_dec"]
        ra_com = np.empty(self.wavelength.size)
        dec_com = np.empty(self.wavelength.size)
        for wave_range in range(0, self.wavelength.size, wavelength_step):
            # Mean across all fibres
            ra_com[wave_range: wave_range + wavelength_step] = np.nansum(
                self.intensity[:, wave_range: wave_range + wavelength_step]**power * ra[:, np.newaxis],
                axis=0) / np.nansum(self.intensity[:, wave_range: wave_range + wavelength_step]**power,
                                    axis=0)
            # Statistic (e.g., median, mean) per wavelength bin
            ra_com[wave_range: wave_range + wavelength_step] = stat(ra_com[wave_range: wave_range + wavelength_step])
            dec_com[wave_range: wave_range + wavelength_step] = np.nansum(
                self.intensity[:, wave_range: wave_range + wavelength_step]**power * dec[:, np.newaxis],
                axis=0) / np.nansum(self.intensity[:, wave_range: wave_range + wavelength_step]**power,
                                    axis=0)
            dec_com[wave_range: wave_range + wavelength_step] = stat(dec_com[wave_range: wave_range + wavelength_step])
        return ra_com, dec_com

    def update_coordinates(self, new_fib_coord):
        """Update fibre coordinates.
        TODO: This should not be a method of RSS but part of the Astrometry Correction.

        For each of the parameters provided (different from None), the coordinates will be updated while the original
        ones will be stored in the info dict with a new prefix 'ori_'.

        Parameters
        ----------
        new_fib_coord: (2, n) np.array(float), default=None
            New fibre coordinates for ra and dec axis, expressed in *deg*.
        Returns
        -------
        """

        self.info['ori_fib_ra'], self.info['ori_fib_dec'] = (self.info["fib_ra"].copy(),
                                                             self.info["fib_dec"].copy())
        self.info["fib_ra"] = new_fib_coord[0]
        self.info["fib_dec"] = new_fib_coord[1]
        self.log['update_coords'] = "Offset-coords updated"
        print("[RSS] Offset-coords updated")

    # =============================================================================
    # Save an RSS object (corrections applied) as a separate .fits file
    # =============================================================================
    def to_fits(self, filename, overwrite=False, checksum=False):
        """
        Writes a RSS object to .fits
        
        Ideally this would be used for RSS objects containing corrected data that need to be saved
        during an intermediate step

        Parameters
        ----------
        name: path-like object
            File to write to. This can be a path to file written in string format with a meaningful name.
        layer: TODO
        overwrite: bool, optional
            If True, overwrite the output file if it exists. Raises an OSError if False and the output file exists. Default is False.
        checksum: bool, optional
            If True, adds both DATASUM and CHECKSUM cards to the headers of all HDU’s written to the file.

        Returns
        -------
        """
        # TODO: This needs to
        primary_hdu = fits.PrimaryHDU(data=self.intensity)
        # primary_hdu.header = self.header
        primary_hdu.verify('fix')
        primary_hdu.writeto(name=filename, overwrite=overwrite, checksum=checksum)
        primary_hdu.close()
        print(f"[RSS] File saved as {filename}")


# =============================================================================
# Combine RSS (e.g., flats, twilights)
# =============================================================================

def combine_rss(list_of_rss, combine_method='nansum'):
    """Combine an input list of DataContainers into a new DataContainer."""

    all_intensities = []
    all_variances = []
    for rss in list_of_rss:
        intensity = rss.intensity.copy()
        variance = rss.variance.copy()
        # Ensure nans
        finite_mask = np.isfinite(intensity) & np.isfinite(variance)
        intensity[~finite_mask] = np.nan
        variance[~finite_mask] = np.nan

        all_intensities.append(intensity)
        all_variances.append(variance)
    
    if hasattr(np, combine_method):
        combine_function = getattr(np, combine_method)
        new_intensity = combine_function(all_intensities, axis=0)
        new_variance = combine_function(all_variances, axis=0)
    else:
        raise NotImplementedError("Implement user-defined combining methods")
    # TODO: Update the metadata as well
    new_rss = RSS(intensity=new_intensity, variance=new_variance,
                  wavelength=rss.wavelength, log=rss.log, info=rss.info)
    return new_rss
    

# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------    
# # Ángel TASKS
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
def rss_valid_wave_range(rss, **kwargs):
    """
    BE SURE YOU HAVE NOT CLEANED CCD DEFECTS if you are running this!!!

    Parameters
    ----------
    rss : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    
    w = rss.wavelength
    n_spectra = len(rss.intensity)
    n_wave = len(rss.wavelength)
    x = list(range(n_spectra))
    
    #  Check if file has 0 or nans in edges
    if np.isnan(rss.intensity[0][-1]):
        no_nans = False
    else:
        no_nans = True
        if rss.intensity[0][-1] != 0:
            vprint(
                "  Careful!!! pixel [0][-1], fibre = 0, wave = -1, that should be in the mask has a value that is not nan or 0 !!!!!", **kwargs)

    if kwargs.get("verbose") and kwargs.get("plot") : print("\n  - Checking the left edge of the ccd...")

    #if verbose and plot: print("\n  - Checking the left edge of the ccd...")
    mask_first_good_value_per_fibre = []
    for fibre in range(n_spectra):
        found = 0
        j = 0
        while found < 1:
            if no_nans:
                if rss.intensity[fibre][j] == 0:
                    j = j + 1
                else:
                    mask_first_good_value_per_fibre.append(j)
                    found = 2
            else:
                if np.isnan(rss.intensity[fibre][j]):
                    j = j + 1
                else:
                    mask_first_good_value_per_fibre.append(j)
                    found = 2
            if j > 101:
                vprint((" No nan or 0 found in the fist 100 pixels, ", w[j], " for fibre", fibre), **kwargs)
                mask_first_good_value_per_fibre.append(j)
                found = 2

    mask_max = np.nanmax(mask_first_good_value_per_fibre)
    if kwargs.get('plot'): 
        plot_plot(x, mask_first_good_value_per_fibre, ymax=mask_max + 1, xlabel="Fibre",
                  ptitle="Left edge of the RSS", hlines=[mask_max], ylabel="First good pixel in RSS")

    # Right edge, important for RED
    if kwargs.get("verbose") and kwargs.get("plot") :  print("\n- Checking the right edge of the ccd...")
    mask_last_good_value_per_fibre = []
    mask_list_fibres_all_good_values = []

    for fibre in range(n_spectra):
        found = 0
        j = n_wave - 1
        while found < 1:
            if no_nans:
                if rss.intensity[fibre][j] == 0:
                    j = j - 1
                else:
                    mask_last_good_value_per_fibre.append(j)
                    if j == len(rss.intensity[0]) - 1:
                        mask_list_fibres_all_good_values.append(fibre)
                    found = 2
            else:
                if np.isnan(rss.intensity[fibre][j]):
                    j = j - 1
                else:
                    mask_last_good_value_per_fibre.append(j)
                    if j == len(rss.intensity[0]) - 1:
                        mask_list_fibres_all_good_values.append(fibre)
                    found = 2

            if j < n_wave - 1 - 300:
                vprint((" No nan or 0 found in the last 300 pixels, ", w[j], " for fibre", fibre), **kwargs)
                mask_last_good_value_per_fibre.append(j)
                found = 2

    mask_min = np.nanmin(mask_last_good_value_per_fibre)
    if kwargs.get('plot'): 
        ptitle = "Fibres with all good values in the right edge of the RSS file : " + str(
            len(mask_list_fibres_all_good_values))
        plot_plot(x, mask_last_good_value_per_fibre, ymin=np.nanmin(mask_min),
                  ymax=2050, hlines=[2047], xlabel="Fibre", ylabel="Last good pixel in RSS", ptitle=ptitle)

    vprint("\n  --> The valid range for this RSS is {:.2f} to {:.2f} ,  in pixels = [ {} ,{} ]".format(w[mask_max],
                                                                                                    w[mask_min],
                                                                                                    mask_max,
                                                                                                    mask_min), **kwargs)

    # rss.mask = [mask_first_good_value_per_fibre, mask_last_good_value_per_fibre]
    # rss.mask_good_index_range = [mask_max, mask_min]
    # rss.mask_good_wavelength_range = [w[mask_max], w[mask_min]]
    # rss.mask_list_fibres_all_good_values = mask_list_fibres_all_good_values

    vprint("\n> Returning [ [mask_first_good_value_per_fibre, mask_last_good_value_per_fibre], ", **kwargs)
    vprint(  "              [mask_max, mask_min], ", **kwargs)
    vprint(  "              [w[mask_max], w[mask_min]], ", **kwargs)
    vprint(  "              mask_list_fibres_all_good_values ] ", **kwargs)
    
    # if verbose:
        # print("\n> Mask stored in rss.mask !")
        # print("  self.mask[0] contains the left edge, self.mask[1] the right edge")
        # print("  Valid range of the data stored in self.mask_good_index_range (index)")
        # print("                             and in self.mask_good_wavelength  (wavelenghts)")
        # print("  Fibres with all good values (in right edge) in self.mask_list_fibres_all_good_values")
    
    #return [rss.mask,rss.mask_good_index_range,rss.mask_good_wavelength_range,rss.mask_list_fibres_all_good_values]
    return [[mask_first_good_value_per_fibre, mask_last_good_value_per_fibre],
            [mask_max, mask_min],
            [w[mask_max], w[mask_min]], 
            mask_list_fibres_all_good_values ] 
    # if include_history:
    #     self.history.append("- Mask obtainted using the RSS file, valid range of data:")
    #     self.history.append(
    #         "  " + str(w[mask_max]) + " to " + str(w[mask_min]) + ",  in pixels = [ " + str(
    #             mask_max) + " , " + str(mask_min) + " ]")
    #     # -----------------------------------------------------------------------------

# Mr Krtxo \(ﾟ▽ﾟ)/  + Ángel R. :-)

