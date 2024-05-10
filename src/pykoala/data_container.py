"""
This module contains the parent class that represents the data used during the
reduction process
"""
import numpy as np

from astropy.io.fits import Header, ImageHDU
from astropy.nddata import bitmask

from pykoala.exceptions.exceptions import NoneAttrError

class LogEntry(object):
    """Log information unit.
    
    Description
    -----------
    This class represents a unit of information stored in a log.

    Attributes
    ----------
    - title: (str)
        Title of the entry.
    - comments: (str)
        List of strings the information to be stored.

    Methods
    -------
    - to_str:
        Return a string containing the information of 
        the entry.
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
            raise NameError("Input comments must be str or list of strings not:"
                            + f" {comments.__class__}")

    def to_str(self, title=True):
        """Convert the entry into a string."""
        comments = "\n".join(self.comments)
        if title:
            comments = f"{self.title}: " + comments
        return comments

class HistoryLog(object):
    """Data reduction history logger class.
    
    Description
    -----------
    This class stores the data reduction history of a DataContainer by creating
    consecutive log entries.

    Attributes
    ----------
    #TODO

    Methods
    -------
    #TODO
    """
    verbose = True
    log_entries = None
    tags = None

    def __init__(self, list_of_entries=None, **kwargs):
        self.vprint("Initialising history log")
        self.log_entries = []
        self.tags = []
        self.verbose = kwargs.get('verbose', True)

        if list_of_entries is not None:
            self.initialise_log(list_of_entries)
    
    def initialise_log(self, list_of_entries):
        """Initialise the log from a set of input entries.
        
        Description
        -----------
        This method initialises the log using a collection of entries. The
        input can be in the form of a, interable consisting of LogEntry objects
        or an interable containing a 2 or 3 elements iterable (title, comments)
        or (title, comments, tag).

        Parameters
        ----------
        - list_of_entries: (iterable)
            Set of entries to be recorded in the log.

        Returns
        -------
        """
        for entry in list_of_entries:
            if isinstance(entry, LogEntry):
                self.log_entries.append(entry)
            elif isinstance(entry, tuple) or isinstance(entry, list):
                if len(entry) == 2:
                    title, comments = entry
                    tag = None
                elif len(entry) == 3:
                    title, comments, tag = entry
                else:
                    raise NameError(
                        "Input entry must contain two (title, comments) or"
                        + " three (title, comments, tag) elements")
                entry = LogEntry(title=title, comments=comments, tag=tag)
            else:
                raise NameError(
                    "Unrecognized input entry of type {entry.__class__}")
            self.log_entries.append(entry)

    def log_entry(self, title, comments, tag=None):
        """Include a new entry in the log.
        
        Parameters
        ----------
        - title: (str)
            Title of the entry.
        - comments: (list or str, default=None)
            List containing the comments of this entry. Lines will be splited
            into different elements of the log.
        """
        self.vprint(f"Logging entry > {title}:{comments}")
        if tag is not None and tag not in self.tags:
            self.tags.append(tag)
        entry = LogEntry(title=title, comments=comments, tag=tag)
        self.log_entries.append(entry)

    def is_entry(self, title, comment=None):
        """Find an entry that contains the input information.
        
        Parameters
        -----------
        - title: (str)
            Title of the entry.
        - comment: (str, default=None)
            If provided, the match will require the entry to contain the input
            comment.

        Returns
        -------
        - found: (bool)
            `True` if the entry is found, `False` otherwise.
        """
        for entry in self.log_entries:
            if entry.title == title:
                if comment is not None:
                    if comment in entry.comments:
                        return True
                else:
                    return True
        return False

    def find_entry(self, title='', comment='', tag=''):
        """Return all entries matching a given title.
        
        Parameters
        ----------
        - title: (str, default='')
            Entry title str.
        - comment: (str, default='')
            Entry comment str.
        - tag: (str, default='')
            Entry tag str.
        Returns
        -------
        - entries: (list)
            List of entries associated to the input title, comment and tag.
        """
        return [entry for entry in self.log_entries if (title in entry.title)
         and (comment in entry.to_str(title=False)) and (tag in str(entry.tag))]


    def dump_to_header(self, header=None):
        """Write the log into a astropy.fits.Header.
        
        Description
        -----------
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

        for entry in self.log_entries:
            header['PYKOALA' + str(index)] = (
                entry.to_str(title=False), entry.title)
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
            for entry in self.log_entries:
                f.write(
                    entry.to_str() + "\n")

    def get_entries_from_header(self, header):
        """Get entries created by PyKOALA from an input FITS Header."""
        list_of_entries = []
        for title, key in zip(header.comments['PYKOALA*'],
                                  header['PYKOALA*']):
            list_of_entries.append(LogEntry(title=title, comments=header[key]))
        return list_of_entries

    def load_from_header(self, header):
        """Load the Log from a FITS Header."""
        list_of_entries = self.get_entries_from_header(header)
        self.log_entries = list(list_of_entries)

    def show(self):
        for entry in self.log_entries:
            print(entry.to_str())

    def vprint(self, *mssg):
        if self.verbose:
            print("[Log] ", *mssg)

    def __call__(self, *args, **kwargs):
        self.log_entry(*args, **kwargs)


class Parameter(object):
    """Class that represents some parameter and associated metadata"""
    def __init__(self) -> None:
        pass
    
class DataMask(object):
    """A mask to store the pixel flags of DataContainers.
    
    Description
    -----------
    A mask to store the pixel flags of DataContainers.

    Attributes
    ----------
    - flag_map: dict, default={"CR": 2, "HP": 4, "DP": 8}
        A mapping between the flag names and their numerical values expressed
        in powers of two.
    - bitmask: (np.ndarray)
        The array containing the bit pixel mask.
    - masks: dict
        A dictionary that stores the individual mask in the form of boolean arrays
        for each flag name.
    
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
        header['COMMENT'] = "Each flag KEY is stored using the convention FLAG_KEY"
        for flag_name, value in self.flag_map.items():
            # Store the value and the description
            header[f"FLAG_{flag_name}"] = value
        header['COMMENT'] = "A value of 0 means unmasked"
        hdu = ImageHDU(name='BITMASK', data=self.bitmask, header=header)
        return hdu

class DataContainer(object):
    """
    Abstract class for data containers.

    This class might represent any kind of astronomical data: raw fits files,
    Row Stacked Spectra obtained after tramline extraction or datacubes.

    Attributes
    ----------
    intensity
    variance
    intensity_units
    info
    log
    mask
    mask_map

    Methods
    -------
    # TODO
    """

    def __init__(self, **kwargs):
        # Data
        self.intensity = kwargs.get("intensity", None)
        self.variance = kwargs.get("variance", None)
        self.intensity_units = kwargs.get("intensity_units", None)
        # Information and masking
        self.info = kwargs.get("info", dict())
        if self.info is None:
            self.info = dict()
        self.log = kwargs.get("log", HistoryLog())
        self.fill_info()
        self.mask = kwargs.get("mask", DataMask(shape=self.intensity.shape))

    def fill_info(self):
        """Check the keywords of info and fills them with placeholders."""
        if 'name' not in self.info.keys():
            self.info['name'] = 'N/A'

    def is_in_info(self, key):
        """Check if a given keyword is stored in the info variable."""
        if self.info is None:
            raise NoneAttrError("info")
        if key in self.info.keys():
            return True
        else:
            return False

    def is_corrected(self, correction):
        """Check if a Correction has been applied by checking the HistoryLog."""
        if self.log.is_entry(title=correction):
            return True
        else:
            return False

    def dump_log_in_header(self, header):
        """Fill a FITS Header with the HistoryLog information."""
        return self.log.dump_to_header(header)

# Mr Krtxo \(ﾟ▽ﾟ)/
