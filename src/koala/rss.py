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
from koala import exceptions as excp
from koala.data_container import DataContainer


# =============================================================================
# Ancillary Functions
# =============================================================================

#This has not been implemented. Is meant to return the range of RA and DEC from centre RA and DEC
def coord_range(rss_list):
    """
    Return the spread of the R

    """
    ra = [rss.RA_centre_deg + rss.offset_RA_arcsec / 3600. for rss in rss_list]
    ra_min = np.nanmin(ra)
    ra_max = np.nanmax(ra)
    dec = [rss.DEC_centre_deg + rss.offset_DEC_arcsec / 3600. for rss in rss_list]
    dec_min = np.nanmin(dec)
    dec_max = np.nanmax(dec)
    return ra_min, ra_max, dec_min, dec_max


def detect_edge(rss):
    """
    Detect the edges of a RSS. Returns the minimum and maximum wavelength that 
    determine the maximum interval with valid (i.e. no masked) data in all the 
    spaxels.

    Parameters
    ----------
    rss : RSS object.

    Returns
    -------
    min_w : float
        The lowest value (in units of the RSS wavelength) with 
        valid data in all spaxels.
    min_index : int
        Index of min_w in the RSS wavelength variable.
    max_w : float
        The higher value (in units of the RSS wavelength) with 
        valid data in all spaxels.
    max_index : int
        Index of max_w in the RSS wavelength variable.

    """
    collapsed = np.sum(rss.intensity, 0)
    nans = np.isfinite(collapsed)
    wavelength = rss.wavelength
    min_w = wavelength[nans].min()
    min_index = wavelength.tolist().index(min_w)
    max_w = wavelength[nans].max()
    max_index = wavelength.tolist().index(max_w)
    return min_w, min_index, max_w, max_index


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
            Mask value      Correction
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

        super().__init__(intensity=intensity, intensity_corrected=intensity_corrected,
                         variance=variance, variance_corrected=variance_corrected,
                         info=info, mask=mask, log=log)
        # Specific RSS attributes
        self.wavelength = wavelength
        self.header = header
        self.fibre_table = fibre_table

    # =============================================================================
    # Mask layer
    # =============================================================================
    def mask_layer(self, index=-1, verbose=False):
        """
        Filter the bit mask according the index value:
            
        Index    Layer                Bit mask value
        ------   ----------------     --------------
         0        Read from file       1
         1        Blue edge            2                                 
         2        Red edge             4                               
         3        NaNs mask            8                                  
         4        Cosmic rays          16                                 
         5        Extreme negative     32                               

        Parameters
        ----------
        index : int or list, optional
            Layer index. The default -1 means that all the existing layers are 
            returned.
        verbose: bool, optional
            Set to True for getting information on the procedure.

        Returns
        -------
        numpy.ndarray(bool)
            Boolean mask with the layer (or layers) selected.
        """
        vprint.verbose = verbose
        mask = self.mask.astype(int)
        if type(index) is not list:
            index = [index]
        ignore_flags = [1, 2, 4, 8, 16, 32]
        for i in index:
            mask_value = 2**i
            try:
                ignore_flags.remove(mask_value)
            except excp.MaskBitError:
                raise excp.MaskBitError(ignore_flags, i)
            vprint('Layers considered: ', ignore_flags)
        return bitfield_to_boolean_mask(mask, ignore_flags=ignore_flags)

    def show_mask(self):
        colormap = colors.ListedColormap(['darkgreen', 'blue', 'red', 'darkviolet', 'black', 'orange'])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(np.log2(self.mask), vmin=0, vmax=6, cmap=colormap, interpolation='none')
        fig.show()

    # =============================================================================
    # Imshow data
    # =============================================================================
    def show(self, pmin=5, pmax=95, mask=False, **kwargs):
        """
        Simple "imshow" of the corrected data (i.e. self.intensity_corrected ).
        Accept all (matplotlib.pyplot) imshow parameters.
        
        In the future we will implement the PyKoala RSS display function here. 
        
        Parameters
        ----------
        pmin : float, optional
            Minimum percentile of the data range that the colormap will covers. 
            The default is 5.
        pmax : TYPE, optional
            Maximum percentile of the data range that the colormap will covers. 
            The default is 95.
        mask : bool, optional
            True show the image with correceted pixels masked. 
            The default is False.
        """

        if mask:
            data = np.ma.masked_array(self.intensity_corrected, self.mask)
        else:
            data = self.intensity_corrected

        vmin, vmax = np.nanpercentile(data, (pmin, pmax))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(data, vmin=vmin, vmax=vmax, **kwargs)
        fig.show()

    # =============================================================================
    # show/save formated log        
    # =============================================================================
    def formated_log(self, verbose=True, save=None):
        """TODO"""
        pretty_line = '-' * 50
        try:
            import textwrap
        except ModuleNotFoundError as err:
            raise err(textwrap)
        if verbose:
            for procedure in self.log.keys():
                comment = self.log[procedure]['comment']
                index = self.log[procedure]['index']
                applied = isinstance(comment, str)
                mask_index = {None: '',
                              0: 'Mask index: 0 (bit mask value 2**0)',
                              1: 'Mask index: 1 (bit mask value 2**1)',
                              2: 'Mask index: 2 (bit mask value 2**2)',
                              3: 'Mask index: 3 (bit mask value 2**3)',
                              4: 'Mask index: 4 (bit mask value 2**4)',
                              5: 'Mask index: 5 (bit mask value 2**5)',
                              }
                if applied:
                    print('\n' + pretty_line)
                    print("{:<49}{:<2}".format(procedure.capitalize(), mask_index[index]))
                    print(pretty_line)
                    for i in textwrap.wrap(comment + '\n', 80):
                        print(i)
                    print('\n')

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
                self.info['cen_dec'].copy(), self.info['cen_dec'].copy())
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
    # Save RSS in fits
    # =============================================================================
    def to_fits(self, name, layer='corrected', overwrite=False):
        """TODO"""
        data = {'corrected': self.intensity_corrected, 'mask': self.mask}
        primary_hdu = fits.PrimaryHDU(data=data[layer])
        primary_hdu.header = self.header
        primary_hdu.verify('fix')
        if self.fibre_table is not None:
            # TODO: Why add again this information again in the table header?
            self.fibre_table.header['CENRA'] = self.header['RACEN'] / (180 / np.pi)  # Must be in radians
            self.fibre_table.header['CENDEC'] = self.header['DECCEN'] / (180 / np.pi)
            hdu_list = fits.HDUList([primary_hdu, self.fibre_table])
            hdu_list.writeto(name, overwrite=True)
        else:
            # Write the fits
            primary_hdu.writeto(name=name, overwrite=overwrite)


# =============================================================================
# Reading rss from file
# =============================================================================
def read_rss(file_path,
             wcs,
             intensity_axis=0,
             variance_axis=None,
             bad_spaxels_list=None,
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

    vprint.verbose = verbose

    vprint("\n> Reading RSS file", file_name, "created with", instrument, "...")

    #  Open fits file. This assumes that RSS objects are written to file as .fits.
    rss_fits = fits.open(file_path)

    # Read intensity using rss_fits_file[0]
    all_intensities = rss_fits[intensity_axis].data
    intensity = np.delete(all_intensities, bad_spaxels_list, 0)

    # Bad pixel verbose summary
    vprint("\n  Number of spectra in this RSS =", len(all_intensities),
           ",  number of good spectra =", len(intensity),
           " ,  number of bad spectra =", len(bad_spaxels_list))
    if bad_spaxels_list:
        vprint("  Bad fibres =", bad_spaxels_list)

    # Read errors if exist a dedicated axis
    if variance_axis is not None:
        all_variances = rss_fits[variance_axis].data
        variance = np.delete(all_variances, bad_spaxels_list, 0)

    else:
        vprint("\n  WARNING! Variance extension not found in fits file!")
        variance = copy.deepcopy(intensity)

    # Close fits file
    rss_fits.close()

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
    intensity_corrected = copy.deepcopy(intensity)

    # Blank corrected variance (i.e a copy of the variance)
    variance_corrected = copy.deepcopy(variance)

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

# Mr Krtxo \(ﾟ▽ﾟ)/

