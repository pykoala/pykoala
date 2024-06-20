"""
Parent CorrectionBase class
"""
from pykoala.exceptions.exceptions import CorrectionClassError
from abc import ABC, abstractmethod


class CorrectionBase(ABC):
    """
    Base class of an astronomical correction to a given data (RSS or CUBE).
    """

    @property
    @abstractmethod
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    @abstractmethod
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    @abstractmethod
    def apply(self):
        raise NotImplementedError("Each class needs to implement this method")

    def corr_print(self, msg, *args):
        """Print a message."""
        if self.verbose:
            print("[Correction: {}] {}".format(self.name, msg), *args)

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

        
