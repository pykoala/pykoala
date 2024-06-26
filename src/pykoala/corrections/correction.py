"""
Base Correction class
"""

from abc import ABC, abstractmethod
import sys
import logging

class CorrectionBase(ABC):
    """
    Base class of an astronomical correction to a given data (RSS or CUBE).
    """

    def __init__(self, verbose=False) -> None:
        self.verbose = verbose
        self.setup_logger()

    @property
    @abstractmethod
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
#    @abstractmethod
    def verbose(self):
        return self._verbose

#    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    @abstractmethod
    def apply(self):
        raise NotImplementedError("Each class needs to implement this method")

    def setup_logger(self):
        self.logger = logging.getLogger(self.name)
        stdout = logging.StreamHandler(stream=sys.stdout)
        fmt = logging.Formatter(
        "[PyKOALA Correction:%(name)s] %(asctime)s | %(levelname)s > %(message)s"
        )
        stdout.setFormatter(fmt)

        if (self.logger.hasHandlers()):
            self.logger.handlers.clear()
        self.logger.addHandler(stdout)

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.ERROR)

    def corr_print(self, msg, level='info'):
        """Print a message."""
        printer = getattr(self.logger, level)
        printer(msg)

    def log_correction(self, datacontainer, status='applied', **extra_comments):
        """Log in the DataContainer the correction and additional info.
        
        Whenever a correction is applied, this is logged into the DataContainer
        log. This might just inform of the status of the correction (applied/failed)
        as well as other relevant information.
        
        Parameters
        ----------
        - datacontainer: koala.DataContainer
            DC to log the correction.
        - status: str, default='applied'
           Keyword to denote the success of the correction. Can take two values
           'applied' or 'failed'.    
        """
        if status != 'applied' and status != 'failed':
            raise KeyError("Correction log status can only be 'applied' or 'failed'")
        
        datacontainer.log(self.name, status, tag='correction')
        for (k, v) in extra_comments.items():
            datacontainer.log(self.name, k + " " + str(v), tag='correction')

        
