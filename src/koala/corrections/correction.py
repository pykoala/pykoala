"""
Parent CorrectionBase class
"""
from koala.exceptions.exceptions import CorrectionClassError
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

    @abstractmethod
    def apply(self, correction, data_container):
        raise NotImplementedError("Each class needs to implement this method")

    def corr_print(self, msg):
        """Print a message."""
        print("[Correction: {}] {}".format(self.name, msg))
