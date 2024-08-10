"""
This module contains the parent classes that represent the data used
during the reduction process.
"""

from abc import ABC, abstractmethod
import numpy as np
import copy
from astropy.io.fits import Header, ImageHDU
from astropy import units as u

# from astropy.nddata import bitmask
# from pykoala.exceptions.exceptions import NoneAttrError
from pykoala import VerboseMixin

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


class DataContainerHistory(object, VerboseMixin):
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
        self.vprint(f"Logging record > {title}:{comments}")
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

    def vprint(self, *mssg):
        if self.verbose:
            print("[Log] ", *mssg)

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

    def flag_pixels(self, mask, flag_name):
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
            raise NameError(f"Input flag name {flag_name} does not exist")
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
    intensity : astropy.Quantity
        Array with the counts/surface brightness/... at each pixel.
    variance : astropy.Quantity
        Uncertainties associated to the `intensity` values.
    mask : DataMask
        Pixel flags.
    info : dict
        Parameters describing the data.
    log : DataContainerHistory
        History log reporting the data reduction steps undertaken so far.

    Methods
    -------
    # TODO
    """

    @property
    def intensity(self):
        """Array with the counts/surface brightness/... at each pixel."""
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        self._intensity = value

    @intensity.deleter
    def intensity(self):
        del self._intensity

    @property
    def variance(self):
        """Uncertainties associated to the `intensity` values."""
        return self._variance

    @variance.setter
    def variance(self, value):
        self._variance = value

    @variance.deleter
    def variance(self):
        del self._variance

    @property
    def mask(self):
        """Pixel flags."""
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    @mask.deleter
    def mask(self):
        del self._mask

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
    wavelength : astropy.Quantity
        Wavelength array, common to all spectra.
    n_wavelength : int
        Number of wavekengths in the `wavelength` array.
    n_spectra : int
        Number of spectra in the `intensity` array.
    intensity_rss : astropy.Quantity
        `intensity` array, sorted as [`n_spectra`, `n_wavelength`].
    variance_rss : astropy.Quantity
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

    @property
    @abstractmethod
    def rss_variance(self):
        """Uncertainties associated to `intensity_rss`."""
        pass

    def __init__(self, **kwargs):
        if "wavelength" in kwargs.keys():
            self._wavelength = kwargs["wavelength"]
        else:
            print(
                "WARNING: No `wavelength` vector supplied; creating empty `SpectraContainer`"
            )
            self._wavelength = u.Quantity([], u.AA)
        super().__init__(**kwargs)


# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
