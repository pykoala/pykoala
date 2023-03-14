"""
Parent CorrectionBase class
"""
from koala.exceptions.exceptions import CorrectionClassError
from abc import ABC, abstractmethod


class CorrectionBase(ABC):
    """
    Base class of an astronomical correction to a given data (RSS or CUBE).
    """

    def __init__(self, target_class):
        self.target_class = target_class

    @property
    @abstractmethod
    def name(self):
        return self._name

    @abstractmethod
    def apply(self):
        raise NotImplementedError("Each class needs to implement this method")

    def check_target(self, target):
        if target.__class__ is not self.target_class:
            raise CorrectionClassError(self.target_class, target.__class__)
        else:
            return
