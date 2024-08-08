# =============================================================================
# Basics packages
# =============================================================================
import numpy as np
import os
from datetime import datetime
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.io import fits
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala import __version__
from pykoala import vprint
from pykoala.data_container import SpectraContainer


# =============================================================================
# RSS CLASS
# =============================================================================


class RSS(SpectraContainer):
    """
    Data Container class for row-stacked spectra (RSS).

    Attributes
    ----------
    intensity : numpy.ndarray(float)
        Intensity :math:``I_lambda``.
        Axis 0 corresponds to spectral dimension
        Axis 1 Corresponds to fibre ID
    variance : numpy.ndarray(float)
        Variance :math:`sigma^2_lambda`.
        (note the square in the definition of the variance). Must have the
        same dimensions as `intensity`
    wavelength : numpy.ndarray(float)
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

    @property
    def rss_intensity(self):
        return self._intensity

    @property
    def rss_variance(self):
        return self._variance

    def __init__(self, **kwargs):
        assert ('wavelength' in kwargs)
        assert ('intensity' in kwargs)
        super().__init__(**kwargs)

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
                self.intensity[:, wave_range: wave_range +
                               wavelength_step]**power * ra[:, np.newaxis],
                axis=0) / np.nansum(self.intensity[:, wave_range: wave_range + wavelength_step]**power,
                                    axis=0)
            # Statistic (e.g., median, mean) per wavelength bin
            ra_com[wave_range: wave_range + wavelength_step] = stat(
                ra_com[wave_range: wave_range + wavelength_step])
            dec_com[wave_range: wave_range + wavelength_step] = np.nansum(
                self.intensity[:, wave_range: wave_range +
                               wavelength_step]**power * dec[:, np.newaxis],
                axis=0) / np.nansum(self.intensity[:, wave_range: wave_range + wavelength_step]**power,
                                    axis=0)
            dec_com[wave_range: wave_range + wavelength_step] = stat(
                dec_com[wave_range: wave_range + wavelength_step])
        return ra_com, dec_com

    def update_coordinates(self, new_coords=None, offset=None):
        """Update fibre coordinates.

        Update the fibre sky position by providing new locations of relative
        offsets.

        Parameters
        ----------
        - new_fib_coord: (2, n) np.array(float), default=None
            New fibre coordinates for ra and dec axis, expressed in *deg*.
        - new_fib_coord_offset: np.ndarray, default=None
            Relative offset in *deg*. If `new_fib_coord` is provided, this will
            be ignored.

        Returns
        -------

        """

        self.info['ori_fib_ra'], self.info['ori_fib_dec'] = (self.info["fib_ra"].copy(),
                                                             self.info["fib_dec"].copy())
        if new_coords is not None:
            self.info["fib_ra"] = new_coords[0]
            self.info["fib_dec"] = new_coords[1]
        elif offset is not None:
            self.info["fib_ra"] += offset[0]
            self.info["fib_dec"] += offset[1]
        else:
            raise NameError(
                "Either `new_fib_coord` or `new_fib_coord_offset` must be provided")
        self.log('update_coords', "Offset-coords updated")
        print("[RSS] Offset-coords updated")

    # =============================================================================
    # Save an RSS object (corrections applied) as a separate .fits file
    # =============================================================================
    def to_fits(self, filename, primary_hdr_kw=None, overwrite=False, checksum=False):
        """
        Writes a RSS object to .fits

        Ideally this would be used for RSS objects containing corrected data that need to be saved
        during an intermediate step

        Parameters
        ----------
        name: path-like object
            File to write to. This can be a path to file written in string format with a meaningful name.
        overwrite: bool, optional
            If True, overwrite the output file if it exists. Raises an OSError if False and the output file exists. Default is False.
        checksum: bool, optional
            If True, adds both DATASUM and CHECKSUM cards to the headers of all HDU’s written to the file.
        Returns
        -------
        """
        if filename is None:
            filename = 'cube_{}.fits.gz'.format(
                datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        if primary_hdr_kw is None:
            primary_hdr_kw = {}

        # Create the PrimaryHDU with WCS information
        primary = fits.PrimaryHDU()
        for key, val in primary_hdr_kw.items():
            primary.header[key] = val

        # Include general information
        primary.header['pykoala0'] = __version__, "PyKOALA version"
        primary.header['pykoala1'] = datetime.now().strftime(
            "%d_%m_%Y_%H_%M_%S"), "creation date / last change"

        # Fill the header with the log information
        primary.header = self.log.dump_to_header(primary.header)

        # primary_hdu.header = self.header
        # Create a list of HDU
        hdu_list = [primary]
        # Change headers for variance and INTENSITY
        hdu_list.append(fits.ImageHDU(
            data=self.intensity, name='INTENSITY',
            # TODO: rescue the original header?
            # header=self.wcs.to_header()
        )
        )
        hdu_list.append(fits.ImageHDU(
            data=self.variance, name='VARIANCE', header=hdu_list[-1].header))
        # Store the mask information
        hdu_list.append(self.mask.dump_to_hdu())
        # Save fits
        hdul = fits.HDUList(hdu_list)
        hdul.verify('fix')
        hdul.writeto(filename, overwrite=overwrite, checksum=checksum)
        hdul.close()
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


# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
