"""
Base Correction class
"""

from abc import ABC, abstractmethod
import os
from astropy.io import fits
from pykoala import VerboseMixin, vprint

class CorrectionBase(ABC, VerboseMixin):
    """
    Abstract base class for implementing astronomical corrections on a 
    DataContainer.

    This class defines the structure and basic functionality required for 
    applying a correction to an astronomical data container. It includes
    logging capabilities to track the application of corrections.

    Attributes
    ----------
    verbose : bool
        Convenient attribute for controlling the logging output. If True, logs are set to 
        INFO level; if False, logs are set to ERROR level.
    logger : logging.Logger
        Logger instance used to log messages. If not provided, a pykoala logger child
        (see `pykoala.ancillary.pykoala_logger`) is created with a name based on the
        correction's name.
    log_filename : str, optional
        Name of the file to log messages. If provided, logs will also be saved 
        to this file.
    log_level : int, optional
        Level of logging, which overrides the `verbose` attribute if provided
        (for more details see `logging levels <https://docs.python.org/3/library/logging.html#logging-levels>`_.

    Methods
    -------
    name :
        Abstract property that needs to be defined in a subclass, representing 
        the name of the correction.
    verbose :
        Abstract property that needs to be defined in a subclass, representing 
        the verbosity level of the correction.
    apply :
        Abstract method that must be implemented in a subclass to apply the 
        correction.
    vprint(msg, level='info') :
        Logs a message at the specified level.
    record_correction(datacontainer, status='applied', **extra_comments) :
        Logs the status of the correction in the DataContainer, with additional
        information if provided.

    Examples
    --------
    To create a new correction, subclass CorrectionBase and implement the 
    ``name``, ``verbose``, and ``apply`` methods.

    Notes
    -----
    When implementing a new correction, ensure that the ``apply`` method correctly
    applies the intended transformation to the DataContainer and that the 
    operation is logged via the ``record_correction`` method.

    """

    def __init__(self, verbose=True, logger=None, log_filename=None, log_level=None) -> None:

        if logger is None:
            self.logger = f"correction.{self.name}"
        else:
            self.logger = logger

        self.verbose = verbose
        # This variable supersedes `verbose`
        if log_level is not None:
            self.logger.setLevel(log_level)

        if log_filename is not None:
            self.log_into_file(log_filename, level=log_level)

    @property
    @abstractmethod
    def name(self):
        """Name of the correction."""
        return None

    @abstractmethod
    def apply(self):
        raise NotImplementedError("Each class needs to implement the `apply` method")

    def record_correction(self, datacontainer, status='applied', **extra_comments):
        """
        Logs the status of the correction in the DataContainer.

        Whenever a correction is applied, this method logs the status of the 
        correction (e.g., 'applied' or 'failed') and any additional comments in 
        the DataContainer's log.

        Parameters
        ----------
        datacontainer : :class:`pykoala.data_container.DataContainer`
            The data container to log the correction.
        status : str, optional
            Indicates the success of the correction. Should be either 'applied' 
            or 'failed' (default is 'applied').
        **extra_comments : dict
            Additional information to log alongside the correction status.

        Raises
        ------
        KeyError
            If `status` is not 'applied' or 'failed'.
        """
        if status not in ['applied', 'failed']:
            raise KeyError("Correction log status can only be 'applied' or 'failed'")
        
        datacontainer.history(self.name, status, tag='correction')
        for k, v in extra_comments.items():
            datacontainer.history(self.name, f"{k} {v}", tag='correction')


class CorrectionOffset(object):
    """Relative Offset Data.

    This class stores a 2D relative offset.

    Attributes
    ----------
    offset_data : np.ndarray
        offset correction data.
    offset_error : np.ndarray
        Standard deviation of ``offset_data``.
    path: str
        Filename path.

    """
    offset_data = None
    offset_error = None

    def __init__(self, path=None, offset_data=None, offset_error=None):
        self.path = path
        self.offset_data = offset_data
        self.offset_error = offset_error

    def tofits(self, output_path=None):
        """Save the offset in a FITS file.
        
        Parameters
        ----------
        output_path: str, optional, default=None
            FITS file name path. If None, and ``self.path`` exists,
            the original file is overwritten.

        Notes
        -----
        The output fits file contains an empty PrimaryHDU, and two ImageHDU
        ("OFFSET", "OFFSET_ERR") containing the offset data and associated error.
        """
        if output_path is None:
            if self.path is None:
                raise NameError("Provide output path")
            else:
                output_path = self.path
        primary = fits.PrimaryHDU()
        data = fits.ImageHDU(data=self.offset_data, name='OFFSET')
        error = fits.ImageHDU(data=self.offset_error, name='OFFSET_ERR')
        hdul = fits.HDUList([primary, data, error])
        hdul.writeto(output_path, overwrite=True)
        hdul.close(verbose=True)
        vprint(f"{self.__class__.__name__} data saved at {output_path}")

    @classmethod
    def from_fits(cls, path):
        """Load the offset data from a fits file.

        Loads offset values (extension 1) and
        associated errors (extension 2) from a fits file.

        Parameters
        ----------
        path : str
            Path to the FITS file containing the offset data.

        Returns
        -------
        wavelength_offset : :class:`WavelengthOffset`
            A :class:`WavelengthOffset` initialised with the input data.
        """
        if not os.path.isfile(path):
            raise NameError(f"offset file {path} does not exist.")
        vprint(f"Loading wavelength offset from {path}")
        with fits.open(path) as hdul:
            offset_data = hdul[1].data
            offset_error = hdul[2].data
        return cls(offset_data=offset_data, offset_error=offset_error,
                   path=path)
