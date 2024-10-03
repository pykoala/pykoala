"""
This module contains the base classes that represent the data used
during the reduction process of IFS data in `pykoala`.

In a nutshell, `pykoala` interprets all sources of data
(e.g., images, raw stacked spectra, data cubes) as a :class:`DataContainer` (DC).
All DC include a series of common properties (e.g., ``intensity`` or ``variance``)
as well as common methods for I/O, data reduction tracking history or data masking.

A special type of data containers are the :class:`SpectraContainer`, which do not only
include an ``intensity``, but also a ``wavelength`` property associated to each pixel along the
spectral dimension of the data. This can be used to represent RSS or DataCubes commonly
used in IFS, as well as, Multi Object Spectroscopy (MOS) datasets. In addition,
all spectra containers must include a method for transforming the N-dimensional 
``intensity`` and ``variance`` arrays into a 2D array (referred to as ``rss_intensity``)
where the first and second dimensions correspond to the spatial (e.g. fibres, spaxels),
and spectral dimension, respectively.
"""

from abc import ABC, abstractmethod
import numpy as np
import copy
from datetime import datetime
from astropy.io.fits import Header, ImageHDU
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

from pykoala import VerboseMixin, __version__
from pykoala import ancillary
from pykoala.plotting.utils import colour_map, new_figure, fibre_map
# =============================================================================


class HistoryRecord(object):
    """Log information unit.

    This class represents a unit of information stored in a log.

    Attributes
    ----------
    - title: (str)
        Title of the record.
    - comments: (str)
        List of strings the information to be stored.

    Methods
    -------
    - to_str:
        Return a string containing the information of
        the record.
    """

    def __init__(self, title, comments, tag=None) -> None:
        self.title = title
        self.comments = comments
        self.tag = tag

    @property
    def comments(self):
        return self._comments

    @comments.setter
    def comments(self, comments):
        if isinstance(comments, list):
            self._comments = comments
        elif isinstance(comments, str):
            self._comments = comments.split("\n")
        else:
            raise NameError(
                "Input comments must be str or list of strings, not:"
                + f" {comments.__class__}"
            )

    def to_str(self, title=True):
        """Convert the record into a string."""
        comments = "\n".join(self.comments)
        if title:
            comments = f"{self.title}: " + comments
        return comments


# =============================================================================


class DataContainerHistory(VerboseMixin):
    """Data reduction history logger class.

    This class stores the data reduction history of a DataContainer by creating
    consecutive log entries.

    Attributes
    ----------
    #TODO

    Methods
    -------
    #TODO
    """

    def __init__(self, list_of_entries=None, **kwargs):
        # Initialise the verbose logger
        self.logger = kwargs.get("logger", "pykoala")
        self.verbose = kwargs.get("verbose", True)
        self.record_entries = []
        self.tags = []
        self.verbose = kwargs.get("verbose", True)

        if list_of_entries is not None:
            self.initialise_record(list_of_entries)

    def initialise_record(self, list_of_entries):
        """Initialise the record from a set of input entries.

        This method initialises the record using a collection of entries. The
        input can be in the form of a, interable consisting of HistoryRecord objects
        or an interable containing a 2 or 3 elements iterable (title, comments)
        or (title, comments, tag).

        Parameters
        ----------
        - list_of_entries: (iterable)
            Set of entries to be recorded in the log.

        Returns
        -------
        """
        for record in list_of_entries:
            if isinstance(record, HistoryRecord):
                self.record_entries.append(record)
            elif isinstance(record, tuple) or isinstance(record, list):
                if len(record) == 2:
                    title, comments = record
                    tag = None
                elif len(record) == 3:
                    title, comments, tag = record
                else:
                    raise NameError(
                        "Input record must contain two (title, comments) or"
                        + " three (title, comments, tag) elements"
                    )
                record = HistoryRecord(title=title, comments=comments, tag=tag)
            else:
                raise NameError(f"Unrecognized input record of type {record.__class__}")
            self.record_entries.append(record)

    def log_record(self, title, comments, tag=None):
        """Include a new record in the history.

        Parameters
        ----------
        - title: (str)
            Title of the record.
        - comments: (list or str, default=None)
            List containing the comments of this record. Lines will be splited
            into different elements of the log.
        """
        if tag is not None and tag not in self.tags:
            self.tags.append(tag)
        record = HistoryRecord(title=title, comments=comments, tag=tag)
        self.record_entries.append(record)

    def is_record(self, title, comment=None):
        """Find an record that contains the input information.

        Parameters
        -----------
        - title: (str)
            Title of the record.
        - comment: (str, default=None)
            If provided, the match will require the record to contain the input
            comment.

        Returns
        -------
        - found: (bool)
            `True` if the record is found, `False` otherwise.
        """
        for record in self.record_entries:
            if record.title == title:
                if comment is not None:
                    if comment in record.comments:
                        return True
                else:
                    return True
        return False

    def find_record(self, title="", comment="", tag=""):
        """Return all entries matching a given title.

        Parameters
        ----------
        - title: (str, default='')
            record title str.
        - comment: (str, default='')
            record comment str.
        - tag: (str, default='')
            record tag str.
        Returns
        -------
        - entries: (list)
            List of entries associated to the input title, comment and tag.
        """
        return [
            record
            for record in self.record_entries
            if (title in record.title)
            and (comment in record.to_str(title=False))
            and (tag in str(record.tag))
        ]

    def dump_to_header(self, header=None):
        """Write the log into a astropy.fits.Header.

        Save the entries of the log in a Header. All entries will be saved using
        card names `PYKOALAnumber`, where number corresponds to an index ranging
        from 0 to the number of entries contained in the log.

        Parameters
        ----------

        """
        if header is None:
            header = Header()
            index = 0
        else:
            index = len(self.get_entries_from_header(header))

        for record in self.record_entries:
            header["PYKOALA" + str(index)] = (record.to_str(title=False), record.title)
            index += 1
        return header

    def dump_to_text(self, file):
        """Write the log into a text file

        Parameters
        ----------
        - file: (str)
            Output file name.

        Returns
        -------
        """
        self.vprint("Writting log into text file")
        with open(file, "w") as f:
            for record in self.record_entries:
                f.write(record.to_str() + "\n")

    def get_entries_from_header(self, header):
        """Get entries created by PyKOALA from an input FITS Header."""
        list_of_entries = []
        for title, key in zip(header.comments["PYKOALA*"], header["PYKOALA*"]):
            list_of_entries.append(HistoryRecord(title=title, comments=header[key]))
        return list_of_entries

    def load_from_header(self, header):
        """Load the Log from a FITS Header."""
        list_of_entries = self.get_entries_from_header(header)
        self.record_entries = list(list_of_entries)

    def show(self):
        for record in self.record_entries:
            print(record.to_str())

    def __call__(self, *args, **kwargs):
        self.log_record(*args, **kwargs)


# =============================================================================


class Parameter(object):
    """Class that represents some parameter and associated metadata"""

    def __init__(self) -> None:
        pass


# =============================================================================


class DataMask(object):
    """A mask to store the pixel flags of DataContainers.

    A mask to store the pixel flags of DataContainers.

    Attributes
    ----------
    - flag_map: dict, default={"CR": 2, "HP": 4, "DP": 8}
        A mapping between the flag names and their numerical values
        expressed in powers of two.
    - bitmask: (np.ndarray)
        The array containing the bit pixel mask.
    - masks: dict
        A dictionary that stores the individual mask in the form of
        boolean arrays for each flag name.

    Methods
    -------
    - flag_pixels
    - get_flag_map_from_bitmask
    - get_flag_map
    """

    def __init__(self, shape, flag_map=None):
        if flag_map is None:
            self.flag_map = {"BAD": (2, "Generic bad pixel flag")}
        else:
            self.flag_map = flag_map
        # Initialise the mask with all pixels being valid
        self.bitmask = np.zeros(shape, dtype=int)
        self.masks = {}
        for key in self.flag_map.keys():
            self.masks[key] = np.zeros(shape, dtype=bool)

    def __decode_bitmask(self, value):
        return np.bitwise_and(self.bitmask, value) > 0

    def flag_pixels(self, mask, flag_name, desc=""):
        """Add a pixel mask corresponding to a flag name.

        Add a pixel mask layer in the bitmask. If the mask already contains
        information about the same flag, it will be overriden by the
        new values.

        Parameters
        ----------
        - mask: np.ndarray
            Input pixel flag. It must have the same shape as the bitmask.
        """
        if flag_name not in self.flag_map:
            self.add_new_flag(flag_name, desc=desc)
        # Check that the bitmask does not already contain this flag
        bit_flag_map = self.get_flag_map_from_bitmask(flag_name)
        self.bitmask[bit_flag_map] -= self.flag_map[flag_name][0]
        self.bitmask[mask] += self.flag_map[flag_name][0]
        # Store the individual boolean map
        self.masks[flag_name] = mask

    def get_flag_map_from_bitmask(self, flag_name):
        """Get the boolean mask for a given flag name from the bitmask."""
        return self.__decode_bitmask(self.flag_map[flag_name][0])

    def get_flag_map(self, flag_name=None):
        """Return the boolean mask that corresponds to the input flags.

        Parameters
        ----------
        - flag_name: str or iterable, default=None
            The flags to be used for constructing the mask. It can be a single
            flag name or an iterable. If None, the mask will comprise every flag
            that is included on the bitmask.

        Returns
        -------
        - mask: np.ndarray
            An array containing the boolean values for every pixel.
        """
        if flag_name is not None:
            if type(flag_name) is str:
                return self.masks[flag_name]
            else:
                mask = np.zeros_like(self.bitmask, dtype=bool)
                for flag in flag_name:
                    mask |= self.masks[flag]
                return mask
        else:
            return self.bitmask > 0

    def add_new_flag(self, name, value=None, desc=""):
        if value is None:
            value = max([v[0] for v in self.flag_map.values()]) * 2
        self.flag_map[name] = (value, desc)
        self.masks[name] = np.zeros(self.bitmask.shape, dtype=bool)


    def dump_to_hdu(self):
        """Return a ImageHDU containig the mask information.

        Returns
        -------
        - hdu: ImageHDU
            An ImageHDU containing the bitmask information.
        """
        header = Header()
        header["COMMENT"] = "Each flag KEY is stored using the convention FLAG_KEY"
        for flag_name, value in self.flag_map.items():
            # Store the value and the description
            header[f"FLAG_{flag_name}"] = value
        header["COMMENT"] = "A value of 0 means unmasked"
        hdu = ImageHDU(name="BITMASK", data=self.bitmask, header=header)
        return hdu


# =============================================================================


class DataContainer(ABC, VerboseMixin):
    """
    Abstract class for data containers.

    This class might represent any kind of astronomical data: raw fits files,
    Row Stacked Spectra obtained after tramline extraction or datacubes.

    Attributes
    ----------
    intensity : :class:`astropy.Quantity`
        Array with the counts/surface brightness/... at each pixel.
    variance : :class:`astropy.Quantity`
        Uncertainties associated to the ``intensity`` values.
    inverse_variance : :class:`astropy.Quantity`
        Inverse variance associated to the ``intensity`` values.
    snr : :class:`astropy.Quantity`
        Signal-to-noise ratio defined as ``intensity / variance**0.5``.
    mask : DataMask
        :class:`DataMask` pixel mask.
    info : dict
        Parameters describing the data.
    log : :class:`DataContainerHistory`
        History log reporting the data reduction steps undertaken so far.
    header : :class:`astropy.fits.Header`
        FITS Header associated to the RSS.

    Methods
    -------
    # TODO
    """

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        self._intensity = value

    @intensity.deleter
    def intensity(self):
        del self._intensity

    @property
    def variance(self):
        return self._variance

    @variance.setter
    def variance(self, value):
        self._variance = value

    @variance.deleter
    def variance(self):
        del self._variance

    @property
    def inverse_variance(self):
        return 1 / self.variance

    @property
    def snr(self):
        return self.intensity / self.variance**0.5

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    @mask.deleter
    def mask(self):
        del self._mask

    @property
    def header(self):
        """FITS Header associated to the file."""
        return self._header
    
    @header.setter
    def header(self, value):
        assert isinstance(value, Header)
        self._header = value

    def __init__(self, **kwargs):
        self._intensity = kwargs["intensity"]
        self._variance = kwargs.get(
            "variance", np.full_like(self._intensity, np.nan, dtype=type(np.nan))
        )
        self._mask = kwargs.get("mask", DataMask(shape=self.intensity.shape))
        self.info = kwargs.get("info", dict())
        self.fill_info()
        # Setup datacontainer logging/verbosity and history
        self.logger = kwargs.get("logger", "pykoala.dc")
        self.verbose = kwargs.get("verbose", True)
        self.history = kwargs.get("history",
                                  DataContainerHistory(logger=self.logger,
                                                       verbose=self.verbose))
        self.header = kwargs.get("header", Header())

    def fill_info(self):
        """Check the keywords of info and fills them with placeholders."""
        if "name" not in self.info.keys():
            self.info["name"] = "N/A"

    def copy(self):
        return copy.deepcopy(self)

    def is_corrected(self, correction):
        """Check if a Correction has been applied by checking the DataContainerHistory"""
        if self.history.is_record(title=correction):
            return True
        else:
            return False


# =============================================================================


class SpectraContainer(DataContainer):
    """
    Abstract class for a `DataContainer` containing spectra (`RSS` or `Cube`).

    Attributes
    ----------
    wavelength : :class:`astropy.Quantity`
        Wavelength array, common to all spectra.
    n_wavelength : int
        Number of wavekengths in the `wavelength` array.
    n_spectra : int
        Number of spectra in the `intensity` array.
    intensity_rss : :class:`astropy.Quantity`
        `intensity` array, sorted as [`n_spectra`, `n_wavelength`].
    variance_rss : :class:`astropy.Quantity`
        Uncertainties associated to `intensity_rss`.
    """

    @property
    def wavelength(self):
        """Pixel flags."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value

    @wavelength.deleter
    def wavelength(self):
        del self._wavelength

    @property
    def n_wavelength(self):
        """Pixel flags."""
        return self._wavelength.size

    @property
    def n_spectra(self):
        """Pixel flags."""
        return self._intensity.size / self._wavelength.size

    @property
    @abstractmethod
    def rss_intensity(self):
        """
        Return an array of spectra,
        sorted as [`n_spectra`, `n_wavelength`].
        """
        pass

    @rss_intensity.setter
    @abstractmethod
    def rss_intensity(self):
        pass

    @property
    @abstractmethod
    def rss_variance(self):
        """Uncertainties associated to `intensity_rss`."""
        pass

    @rss_variance.setter
    @abstractmethod
    def rss_variance(self):
        pass

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        if "wavelength" in kwargs.keys():
            self._wavelength = kwargs["wavelength"]
        else:
            self.vprint(
                "WARNING: No `wavelength` vector supplied; creating empty `SpectraContainer`"
            )
            self._wavelength = u.Quantity([], u.AA)
        
    def get_spectra_sorted(self, wave_range=None):
        """Get the RSS-wise sorted order of the intensity.
        
        Parameters
        ----------
        - wave_range: 2-element iterable, ooptional
            Wavelength limits to compute the median intensity per spatial element.
        
        Returns
        -------
        - sorted_order:
            Sorted list of indices.
        """
        if wave_range is None:
            wave_mask = np.ones_like(self.wavelength, dtype=bool)
            wave_mask[:100] = 0
            wave_mask[-100:] = 0
        else:
            wave_mask = (self.wavelength >= wave_range[0]) & (
                self.wavelength <= wave_range[1])
        median_intensity = np.nanmedian(self.rss_intensity[:, wave_mask], axis=1)
        median_intensity = np.nan_to_num(median_intensity)
        return np.argsort(median_intensity)


class RSS(SpectraContainer):
    """Data Container class for row-stacked spectra (RSS)."""

    @property
    def rss_intensity(self):
        return self._intensity

    @rss_intensity.setter
    def rss_intensity(self, value):
        self.intensity = value

    @property
    def rss_variance(self):
        return self._variance

    @rss_variance.setter
    def rss_variance(self, value):
        self.variance = value

    @property
    def fibre_diameter(self):
        """Angular diameter of the RSS fibres."""
        return self._fibre_diameter
    
    @fibre_diameter.setter
    def fibre_diameter(self, value):
        assert isinstance(value, u.Quantity) or value is None, (
            "Fibre diameter must be a astropy.units.Quantity")
        self._fibre_diameter = value

    @property
    def sky_fibres(self):
        """Indices of the RSS sky fibres."""
        return self._sky_fibres
    
    @sky_fibres.setter
    def sky_fibres(self, value):
        self._sky_fibres = value

    @property
    def science_fibres(self):
        """Indices of fibres with a science target."""
        return np.delete(np.arange(self.intensity.shape[0]), self.sky_fibres)

    def __init__(self, **kwargs):
        assert ('wavelength' in kwargs)
        assert ('intensity' in kwargs)
        if "logger" not in kwargs:
            kwargs['logger'] = "pykoala.rss"

        self.fibre_diameter = kwargs.get("fibre_diameter", None)
        self.sky_fibres = kwargs.get("sky_fibres", [])
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
        new_fib_coord: (2, n) np.array(float), default=None
            New fibre coordinates for ra and dec axis, expressed in *deg*.
        new_fib_coord_offset: np.ndarray, default=None
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
        self.history('update_coords', "Offset-coords updated")
        self.vprint("[RSS] Offset-coords updated")

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
        primary.header = self.history.dump_to_header(primary.header)

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
        self.vprint(f"[RSS] File saved as {filename}")


    def get_integrated_fibres(self, wavelength_range=None):
        """Compute the integrated intensity of the RSS fibres.
        
        Paramters
        ---------
        wavelength_range: 2-element iterable, optional
            Wavelenght limits used to compute the integrated intensity.
        
        Returns
        -------
        integrated_fibres: 1D np.ndarray
            Array containing the integrated flux.
        integrated_variances: 1D np.ndarray
            Array containing the integrated variance associated to each fibre.
        """
        if wavelength_range is not None:
            wave_mask = (self.wavelength >= wavelength_range[0]) & (
                self.wavelength <= wavelength_range[1]
            )
        else:
            wave_mask = np.ones(self.wavelength.size, dtype=bool)

        integrated_fibres = np.nanmean(self.intensity[:, wave_mask], axis=1
                                       ) * np.count_nonzero(wave_mask)
        integrated_variances = np.nanmean(self.variance[:, wave_mask], axis=1
                                       ) * np.count_nonzero(wave_mask)
        return integrated_fibres, integrated_variances

    def plot_rss_image(self, data=None, data_label="", fig_args={}, cmap_args={},
                       fibre_range=None,
                       wavelength_range=None,
                       output_filename=None):
        """Plots a RSS image with optional data, fibre, and wavelength ranges.

        Parameters
        ----------
        data : array-like, optional
            The 2D array data to be plotted. If `None`, `intensity` is used.
        data_label : str, optional
            The color bar label for the data being plotted. Default is an empty string.
        fig_args : dict, optional
            Additional keyword arguments passed to `pykoala.plotting.utils.new_figure` for customizing the figure. 
            Default is an empty dictionary.
        cmap_args : dict, optional
            Additional keyword arguments passed to the `pykoala.plotting.utils.colour_map` function for the colormap
            and normalization.  Default is an empty dictionary.
        fibre_range : tuple of int, optional
            A tuple specifying the range of fibres to include in the plot (start, end).
            If `None`, all fibres are included. Default is `None`.
        wavelength_range : tuple of float, optional
            A tuple specifying the range of wavelengths to include in the plot (start, end). If `None`, all wavelengths are included.
            Default is `None`.
        output_filename : str, optional
            If provided, the plot is saved to the specified file path. Default is `None`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.

        Notes
        -----
        - The function uses the internal attributes `self.wavelength` and `self.intensity` to obtain default x-values 
        (wavelengths) and y-values (fibre indices) if `data` is not provided.
        - The `new_figure` function is used to create a new figure, and `colour_map` is used to plot the data.
        - If `fibre_range` or `wavelength_range` is specified, the data is sliced accordingly.
        - The plot is saved to `output_filename` if provided, otherwise the figure is returned for display or further manipulation.

        """
        x = self.wavelength
        y = np.arange(0, self.intensity.shape[0])
        if data is None:
            data = self.intensity
            data_label = "Intensity"
        if fibre_range is not None:
            fibre_range = range(*fibre_range)
            data = data[fibre_range]
            y = y[fibre_range]
        if wavelength_range is not None:
            wavelength_range = range(*np.searchsorted(self.wavelength, wavelength_range))
            data = data[:, wavelength_range]
            x = x[wavelength_range]

        fig, axs = new_figure(self.info['name'], **fig_args)
        im, cb = colour_map(fig, axs[0, 0], data_label, data,
                            x=x, y=y,
                            xlabel="Wavelength [AA]", ylabel="Fibre",
                            **cmap_args)

        if output_filename is not None:
            fig.savefig(output_filename, bbox_inches="tight")
        return fig

    def plot_mask(self, fig_args={}, cmap_args={}, output_filename=None):
        """Plots a mask image using the bitmask data.

        This method creates a plot of the bitmask data using a predefined colormap and normalization settings.
        It utilizes the `plot_rss_image` method to generate the plot.

        Parameters
        ----------
        fig_args : dict, optional
            Additional keyword arguments passed to the `new_figure` function for customizing the figure.
            Default is an empty dictionary.
        cmap_args : dict, optional
            Additional keyword arguments passed to the `colour_map` function for customizing the colormap.
            If not specified, the colormap is set to "Accent" and normalization to "Normalize". Default is an empty dictionary.
        output_filename : str, optional
            If provided, the plot is saved to the specified file path. Default is `None`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.

        See Also
        --------
        :func:`plot_rss_image`.
        """

        if "cmap" not in cmap_args:
            cmap_args["cmap"] = "Accent"
        if "norm" not in cmap_args:
            cmap_args["norm"] = "Normalize"
        fig = self.plot_rss_image(data=self.mask.bitmask, data_label="Bitmask",
                            fig_args=fig_args, cmap_args=cmap_args,
                            output_filename=output_filename)
        return fig
    
    

    def plot_fibre(self):
        # TODO: THIS SHOULD BE A METHOD OF THE PARENT CLASS
        pass

    def plot_fibre_map(self, data=None, cblabel="", fig_args={},
                       cmap_args={}, output_filename=None):
        """
        Plots a fibre map image, showing the spatial distribution of data across fibres.

        This method generates a plot that visualizes the spatial distribution of
        data across fibres, using the Right Ascension (RA) and Declination (Dec)
        of each fibre. If no data is provided, it uses the integrated fibre intensity data.

        Parameters
        ----------
        data : array-like, optional
            The data to be plotted. If `None`, the method calls `self.get_integrated_fibres()`
            to obtain the integrated fibre intensity data. Default is `None`.
        
        cblabel : str, optional
            The label for the color bar representing the data being plotted. Default is an empty string.
        
        fig_args : dict, optional
            Additional keyword arguments passed to the `new_figure` function for
            customizing the figure. Default is an empty dictionary.
        
        cmap_args : dict, optional
            Additional keyword arguments passed to the `fibre_map` function for
            customizing the colormap. Default is an empty dictionary.
        
        output_filename : str, optional
            If provided, the plot is saved to the specified file path. Default is `None`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        """
        if data is None:
            data, _ = self.get_integrated_fibres()
            cblabel = "Integrated intensity"
        fig, axs = new_figure(self.info['name'], **fig_args)
        axs[0, 0].set_aspect('auto')
        im, cb = fibre_map(fig, axs[0, 0], cblabel, data, fib_ra=self.info['fib_ra'],
                         fib_dec=self.info['fib_dec'], **cmap_args)
        if output_filename is not None:
            fig.savefig(output_filename, bbox_inches="tight")
        return fig


class Cube(SpectraContainer):
    """This class represent a 3D data cube.
    
    parent_rss
    rss_mask
    intensity
    variance
    intensity
    variance
    wavelength
    info

    """
    n_wavelength = None
    n_cols = None
    n_rows = None
    x_size_arcsec = None
    y_size_arcsec = None
    default_hdul_extensions_map = {"INTENSITY": "INTENSITY",
                                   "VARIANCE": "VARIANCE"}
    
    def __init__(self, hdul=None, hdul_extensions_map=None, **kwargs):

        self.hdul = hdul

        if hdul_extensions_map is not None:
            self.hdul_extensions_map = hdul_extensions_map
        else:
            self.hdul_extensions_map = self.default_hdul_extensions_map

        if "logger" not in kwargs:
            kwargs['logger'] = "pykoala.cube"
        super().__init__(intensity=self.intensity,
                         variance=self.variance,
                         **kwargs)
        self.get_wcs_from_header()
        self.parse_info_from_header()
        self.n_wavelength, self.n_rows, self.n_cols = self.intensity.shape
        self.get_wavelength()

    @property
    def hdul(self):
        return self._hdul

    @hdul.setter
    def hdul(self, hdul):
        print(hdul)
        assert isinstance(hdul, fits.HDUList)
        self._hdul = hdul

    @property
    def intensity(self):
        return self.hdul[self.hdul_extensions_map['INTENSITY']].data
    
    @intensity.setter
    def intensity(self, intensity_corr):
        self.vprint("[Cube] Updating HDUL INTENSITY")
        self.hdul[self.hdul_extensions_map['INTENSITY']].data = intensity_corr

    @property
    def variance(self):
        return self.hdul[self.hdul_extensions_map['VARIANCE']].data

    @variance.setter
    def variance(self, variance_corr):
        self.vprint("[Cube] Updating HDUL variance")
        self.hdul[self.hdul_extensions_map['VARIANCE']].data = variance_corr

    @property
    def rss_intensity(self):
        return np.reshape(self.intensity, (self.intensity.shape[0], self.intensity.shape[1]*self.intensity.shape[2])).T

    @rss_intensity.setter   
    def rss_intensity(self, value):
        self.intensity = value.T.reshape(self.intensity.shape)

    @property
    def rss_variance(self):
        return np.reshape(self.variance, (self.variance.shape[0], self.variance.shape[1]*self.variance.shape[2])).T

    @rss_variance.setter   
    def rss_variance(self, value):
        self.variance = value.T.reshape(self.variance.shape)

    @classmethod
    def from_fits(cls, path, hdul_extension_map=None, **kwargs):
        """Make an instance of a Cube using an input path to a FITS file.
        
        Parameters
        ----------
        - path: str
            Path to the FITS file. This file must be compliant with the pykoala
            standards.
        - hdul_extension_map: dict
            Dictionary containing the mapping to access the extensions that
            contain the intensity, and variance data
            (e.g. {'INTENSITY': 1, 'VARIANCE': 'var'}).
        - kwargs:
            Arguments passed to the Cube class (see Cube documentation)
        
        Returns
        -------
        - cube: Cube
            An instance of a `pykoala.cubing.Cube`.
        """
        with fits.open(path) as hdul:
            return cls(hdul, hdul_extension_map=hdul_extension_map, **kwargs)

    def parse_info_from_header(self):
        """Look into the primary header for pykoala information."""
        self.vprint("[Cube] Looking for information in the primary header")
        # TODO
        #self.info = {}
        #self.fill_info()
        self.history.load_from_header(self.hdul[0].header)

    def load_hdul(self, path_to_file):
        self.hdul = fits.open(path_to_file)
        pass

    def close_hdul(self):
        if self.hdul is not None:
            self.vprint(f"[Cube] Closing HDUL")
            self.hdul.close()

    def get_wcs_from_header(self):
        """Create a WCS from HDUL header."""
        self.wcs = WCS(self.hdul[self.hdul_extensions_map['INTENSITY']].header)

    def get_wavelength(self):
        self.vprint("[Cube] Constructing wavelength array")
        self.wavelength = self.wcs.spectral.array_index_to_world(
            np.arange(self.n_wavelength)).to('angstrom').value
        
    def get_centre_of_mass(self, wavelength_step=1, stat=np.median, power=1.0):
        """Compute the center of mass of the data cube."""
        x = np.arange(0, self.n_cols, 1)
        y = np.arange(0, self.n_rows, 1)
        x_com = np.empty(self.n_wavelength)
        y_com = np.empty(self.n_wavelength)
        for wave_range in range(0, self.n_wavelength, wavelength_step):
            x_com[wave_range: wave_range + wavelength_step] = np.nansum(
                self.intensity[wave_range: wave_range + wavelength_step]**power * x[np.newaxis, np.newaxis, :],
                axis=(1, 2)) / np.nansum(self.intensity[wave_range: wave_range + wavelength_step]**power, axis=(1, 2))
            x_com[wave_range: wave_range + wavelength_step] = stat(x_com[wave_range: wave_range + wavelength_step])
            y_com[wave_range: wave_range + wavelength_step] = np.nansum(
                self.intensity[wave_range: wave_range + wavelength_step]**power * y[np.newaxis, :, np.newaxis],
                axis=(1, 2)) / np.nansum(self.intensity[wave_range: wave_range + wavelength_step]**power, axis=(1, 2))
            y_com[wave_range: wave_range + wavelength_step] = stat(y_com[wave_range: wave_range + wavelength_step])
        return x_com, y_com

    def get_integrated_light_frac(self, frac=0.5):
        """Compute the integrated spectra that accounts for a given fraction of the total intensity."""
        collapsed_intensity = np.nansum(self.intensity, axis=0)
        sort_intensity = np.sort(collapsed_intensity, axis=(0, 1))
        # Sort from highes to lowest luminosity
        sort_intensity = np.flip(sort_intensity, axis=(0, 1))
        cumulative_intensity = np.cumsum(sort_intensity, axis=(0, 1))
        cumulative_intensity /= np.nanmax(cumulative_intensity)
        pos = np.searchsorted(cumulative_intensity, frac)
        return cumulative_intensity[pos]

    def get_white_image(self, wave_range=None, s_clip=3.0, frequency_density=False):
        """Create a white image."""
        if wave_range is not None and self.wavelength is not None:
            wave_mask = (self.wavelength >= wave_range[0]) & (self.wavelength <= wave_range[1])
        else:
            wave_mask = np.ones(self.wavelength.size, dtype=bool)
        
        if s_clip is not None:
            std_dev = ancillary.std_from_mad(self.intensity[wave_mask], axis=0)
            median = np.nanmedian(self.intensity[wave_mask], axis=0)

            weights = (
                (self.intensity[wave_mask] <= median[np.newaxis] + s_clip * std_dev[np.newaxis])
                & (self.intensity[wave_mask] >= median[np.newaxis] - s_clip * std_dev[np.newaxis]))
        else:
            weights = np.ones_like(self.intensity[wave_mask])

        if frequency_density:
            freq_trans = self.wavelength**2 / 3e18
        else:
            freq_trans = np.ones_like(self.wavelength)

        white_image = np.nansum(
            self.intensity[wave_mask] * freq_trans[wave_mask, np.newaxis, np.newaxis] * weights, axis=0
            ) / np.nansum(weights, axis=0)
        return white_image

    def to_fits(self, fname=None, primary_hdr_kw=None):
        """Save the Cube into a FITS file."""
        if fname is None:
            fname = 'cube_{}.fits.gz'.format(
                datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        if primary_hdr_kw is None:
            primary_hdr_kw = {}

        # Create the PrimaryHDU with WCS information 
        primary = fits.PrimaryHDU()
        for key, val in primary_hdr_kw.items():
            primary.header[key] = val
        
        
        # Include cubing information
        primary.header['pykoala0'] = __version__, "PyKOALA version"
        primary.header['pykoala1'] = datetime.now().strftime(
            "%d_%m_%Y_%H_%M_%S"), "creation date / last change"

        # Fill the header with the log information
        primary.header = self.history.dump_to_header(primary.header)

        # Create a list of HDU
        hdu_list = [primary]
        # Change headers for variance and INTENSITY
        hdu_list.append(fits.ImageHDU(
            data=self.intensity, name='INTENSITY', header=self.wcs.to_header()))
        hdu_list.append(fits.ImageHDU(
            data=self.variance, name='VARIANCE', header=hdu_list[-1].header))
        # Store the mask information
        hdu_list.append(self.mask.dump_to_hdu())
        # Save fits
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(fname, overwrite=True)
        hdul.close()
        self.vprint("[Cube] Cube saved at:\n {}".format(fname))

    def update_coordinates(self, new_coords=None, offset=None):
        """Update the celestial coordinates of the Cube"""
        updated_wcs = ancillary.update_wcs_coords(self.wcs.celestial,
                                        ra_dec_val=new_coords,
                                        ra_dec_offset=offset)
        # Update only the celestial axes
        self.vprint(f"Previous CRVAL: {self.wcs.celestial.wcs.crval}"
                    + f"\nNew CRVAL: {updated_wcs.wcs.crval}")
        self.wcs.wcs.crval[:-1] = updated_wcs.wcs.crval
        self.history('update_coords', "Offset-coords updated")

# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
