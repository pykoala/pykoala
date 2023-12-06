# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from matplotlib import colors

# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits
from astropy.nddata import bitfield_to_boolean_mask
# =============================================================================
# KOALA packages
# =============================================================================
# Modular
from koala.ancillary import vprint
from koala.exceptions import exceptions as excp
from koala.data_container import DataContainer


# =============================================================================
# RSS CLASS
# =============================================================================


class RSS(DataContainer):
    """
    Data Container class for row-stacked spectra (RSS).

    Attributes
    ----------
    intensity: numpy.ndarray(float)
        Intensity :math:``I_lambda`` per unit wavelength.
        Axis 0 corresponds to fiber ID
        Axis 1 Corresponds to spectral dimension
    wavelength: numpy.ndarray(float)
        Wavelength, in Angstrom
    variance: numpy.ndarray(float)
        Variance :math:`sigma^2_lambda` per unit wavelength
        (note the square in the definition of the variance).
    mask : numpy.ndarray(float)
        Bit mask that records the pixels with individual corrections performed 
        by the various processes:
            Mask value      CorrectionBase
            -----------     ----------------
            1               Readed from file
            2               Blue edge
            4               Red edge
            8               NaNs mask
            16              Cosmic rays
            32              Extreme negative
            
    intensity_corrected: numpy.ndarray(float)
        Intensity with all the corresponding corrections applied (see log).
    variance_corrected: numpy.ndarray(float)
        Variance with all the corresponding corrections applied (see log).
    log : dict
        Dictionary containing a log of the processes applied on the rss.   
    header : astropy.io.fits.header.Header object 
        The header associated with data.
    fibre_table : astropy.io.fits.hdu.table.BinTableHDU object
        Bin table containing fibre metadata.
    info : dict
        Dictionary containing RSS basic information.
        Important dictionary keys:
            info['fib_ra_offset'] - original RA fiber offset 
            info['fib_dec_offset'] - original DEC fiber offset
            info['cen_ra'] - WCS RA center coordinates
            info['cen_dec'] - WCA DEC center coordinates
            info['pos_angle'] - original position angle (PA) 

            TODO - this is NOT an exhaustive list. Please update when new attributes are encountered       
    """
    def __init__(self,
                 intensity,
                 wavelength,
                 variance,
                 mask,
                 intensity_corrected,
                 variance_corrected,
                 log,
                 header,
                 fibre_table,
                 info
                 ):

        super().__init__(intensity=intensity,
                         intensity_corrected=intensity_corrected,
                         variance=variance,
                         variance_corrected=variance_corrected,
                         info=info, mask=mask, log=log)
        # Specific RSS attributes
        self.wavelength = wavelength
        self.header = header
        self.fibre_table = fibre_table

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
        x = self.info['fib_ra_offset']
        y = self.info['fib_dec_offset']
        x_com = np.empty(self.wavelength.size)
        y_com = np.empty(self.wavelength.size)
        for wave_range in range(0, self.wavelength.size, wavelength_step):
            x_com[wave_range: wave_range + wavelength_step] = np.nansum(
                self.intensity_corrected[:, wave_range: wave_range + wavelength_step]**power * x[:, np.newaxis],
                axis=0) / np.nansum(self.intensity_corrected[:, wave_range: wave_range + wavelength_step]**power,
                                    axis=0)
            x_com[wave_range: wave_range + wavelength_step] = stat(x_com[wave_range: wave_range + wavelength_step])
            y_com[wave_range: wave_range + wavelength_step] = np.nansum(
                self.intensity_corrected[:, wave_range: wave_range + wavelength_step]**power * y[:, np.newaxis],
                axis=0) / np.nansum(self.intensity_corrected[:, wave_range: wave_range + wavelength_step]**power,
                                    axis=0)
            y_com[wave_range: wave_range + wavelength_step] = stat(y_com[wave_range: wave_range + wavelength_step])
        return x_com, y_com

    def update_coordinates(self, new_fib_offset_coord=None, new_centre=None, new_pos_angle=None):
        """Update fibre coordinates.

        For each of the parameters provided (different from None), the coordinates will be updated while the original
        ones will be stored in the info dict with a new prefix 'ori_'.

        Parameters
        ----------
        new_fib_offset_coord: (2, n) np.array(float), default=None
            New fibre offset-coordinates for ra and dec axis, expressed in *arcseconds*.
        new_centre: (2,)-tuple, default=None
            New reference coordinates (RA, DEC) for the RSS, expressed in degrees.
        new_pos_angle: float
            New position angle in degrees (PA)

        Returns
        -------
        """
        if new_fib_offset_coord is not None:
            self.info['ori_fib_ra_offset'], self.info['ori_fib_dec_offset'] = (
                self.info['fib_ra_offset'].copy(), self.info['fib_dec_offset'].copy())
            self.info['fib_ra_offset'] = new_fib_offset_coord[0]
            self.info['fib_dec_offset'] = new_fib_offset_coord[1]
            self.log['update_coords'] = "Offset-coords updated"
            print("[RSS] Offset-coords updated")
        if new_centre is not None:
            self.info['ori_cen_ra'], self.info['ori_cen_dec'] = (
                self.info['cen_ra'].copy(), self.info['cen_dec'].copy())
            self.info['cen_ra'] = new_centre[0]
            self.info['cen_dec'] = new_centre[1]
            self.log['update_coords'] = "Centre coords updated"
            print("[RSS] Centre coords ({:.4f}, {:.4f}) updated to ({:.4f}, {:.4f})".format(
                self.info['ori_cen_ra'], self.info['ori_cen_dec'],
                self.info['cen_ra'], self.info['cen_dec']))
        if new_pos_angle is not None:
            self.info['ori_pos_angle'] = self.info['pos_angle'].copy()
            self.info['pos_angle'] = new_pos_angle
            self.log['update_coords'] = "Position angle {:.3} updated to {:.3}".format(self.info['ori_pos_angle'],
                                                                                       self.info['pos_angle'])
            print("[RSS] Position angle {:.3} updated to {:.3}".format(
                self.info['ori_pos_angle'], self.info['pos_angle']))

    # =============================================================================
    # Save an RSS object (corrections applied) as a separate .fits file
    # =============================================================================
    def to_fits(self, name, layer='corrected', overwrite=False, checksum=False):
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
        data = {'corrected': self.intensity_corrected, 'mask': self.mask}
        primary_hdu = fits.PrimaryHDU(data=data[layer])
        primary_hdu.header = self.header
        primary_hdu.verify('fix')
    
        if self.fibre_table is not None:
            # The cen_ra and cen_dec attributes exist in the RSS object header self.header which is copied to primary_hdu.header above
            # Essentially this step is redundant
            self.fibre_table.header['CENRA'] = self.header['RACEN'] / (180 / np.pi)  # Must be in radians
            self.fibre_table.header['CENDEC'] = self.header['DECCEN'] / (180 / np.pi)
            hdu_list = fits.HDUList([primary_hdu, self.fibre_table])

            # Write the fits using standard writeto with default settings and checksum
            hdu_list.writeto(name, overwrite=overwrite, checksum=checksum)
        else:
            # Write the fits
            primary_hdu.writeto(name=name, overwrite=overwrite, checksum=checksum)
        
        primary_hdu.close()
        print(f"[RSS] File saved as {name}")


# =============================================================================
# Reading RSS from .fits file - inverse opeation of to_fits method above
# =============================================================================

"""
TODO
"""
def read_rss(file_path,
             wcs,
             intensity_axis=0,
             variance_axis=1,
             bad_fibres_list=None,
             instrument=None,
             verbose=False,
             log=None,
             header=None,
             fibre_table=None,
             info=None
             ):
    """TODO."""
    # Blank dictionary for the log
    if log is None:
        log = {'read': {'comment': None, 'index': None},
               'mask from file': {'comment': None, 'index': 0},
               'blue edge': {'comment': None, 'index': 1},
               'red edge': {'comment': None, 'index': 2},
               'cosmic': {'comment': None, 'index': 3},
               'extreme negative': {'comment': None, 'index': 4},
               'wavelength fix': {'comment': None, 'index': None, 'sol': []}}
    if header is None:
        # Blank Astropy Header object for the RSS header
        # Example how to add header value at the end
        # blank_header.append(('DARKCORR', 'OMIT', 'Dark Image Subtraction'), end=True)
        header = fits.header.Header(cards=[], copy=False)

    file_name = os.path.basename(file_path)

    vprint("\n> Reading RSS file", file_name, "created with", instrument, "...",
           verbose=verbose)

    #  Open fits file. This assumes that RSS objects are written to file as .fits.
    with fits.open(file_path) as rss_fits:
        # Read intensity using rss_fits_file[0]
        all_intensities = np.array(rss_fits[intensity_axis].data, dtype=np.float32)
        intensity = np.delete(all_intensities, bad_fibres_list, 0)
        # Bad pixel verbose summary
        vprint("\n  Number of spectra in this RSS =", len(all_intensities),
            ",  number of good spectra =", len(intensity),
            " ,  number of bad spectra =", len(bad_fibres_list),
            verbose=verbose)
        if bad_fibres_list is not None:
            vprint("  Bad fibres =", bad_fibres_list, verbose=verbose)

        # Read errors if exist a dedicated axis
        if variance_axis is not None:
            all_variances = rss_fits[variance_axis].data
            variance = np.delete(all_variances, bad_fibres_list, 0)

        else:
            vprint("\n  WARNING! Variance extension not found in fits file!", verbose=verbose)
            variance = np.full_like(intensity, fill_value=np.nan)

    # Create wavelength from wcs
    nrow, ncol = wcs.array_shape
    wavelength_index = np.arange(ncol)
    wavelength = wcs.dropaxis(1).wcs_pix2world(wavelength_index, 0)[0]
    # log
    comment = ' '.join(['- RSS read from ', file_name])
    log['read']['comment'] = comment
    # First Header value added by the PyKoala routine
    header.append(('DARKCORR', 'OMIT', 'Dark Image Subtraction'), end=True)

    # Blank mask (all 0, i.e. making nothing) of the same shape of the data
    mask = np.zeros_like(intensity)

    # Blank corrected intensity (i.e. a copy of the data)
    intensity_corrected = intensity.copy()

    # Blank corrected variance (i.e a copy of the variance)
    variance_corrected = variance.copy()

    return RSS(intensity=intensity,
               wavelength=wavelength,
               variance=variance,
               mask=mask,
               intensity_corrected=intensity_corrected,
               variance_corrected=variance_corrected,
               log=log,
               header=header,
               fibre_table=fibre_table,
               info=info
               )

def combine_rss(list_of_rss, combine_method='nansum'):
    """Combine an input list of DataContainers into a new DataContainer."""

    all_intensities = []
    all_variances = []
    for rss in list_of_rss:
        intensity = rss.intensity_corrected.copy()
        variance = rss.variance_corrected.copy()
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
    new_rss = RSS(intensity=new_intensity, intensity_corrected=new_intensity,
                  variance=new_variance, variance_corrected=new_variance,
                  wavelength=rss.wavelength, log=rss.log, header=rss.header, fibre_table=rss.fibre_table,
                  info=rss.info)
    return new_rss
    

    

# Mr Krtxo \(ﾟ▽ﾟ)/

