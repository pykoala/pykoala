"""
Base Correction class
"""

from abc import ABC, abstractmethod
from pykoala import VerboseMixin

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
        (for more details see https://docs.python.org/3/library/logging.html#logging-levels).

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
    `name`, `verbose`, and `apply` methods.

    Notes
    -----
    When implementing a new correction, ensure that the `apply` method correctly
    applies the intended transformation to the DataContainer and that the 
    operation is logged via the `record_correction` method.

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

    def corr_print(self, msg, *args):
        """Print a message."""
        if self.verbose:
            print("[Correction: {}] {}".format(self.name, msg), *args)


    @property
    @abstractmethod
    def name(self):
        """Abstract property for the name of the correction."""
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
        datacontainer : koala.DataContainer
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
        
