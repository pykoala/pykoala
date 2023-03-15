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

    @property
    @abstractmethod
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = value

    @abstractmethod
    def apply(self):
        raise NotImplementedError("Each class needs to implement this method")

    def check_target(self, target):
        if target.__class__ is not self.target:
            raise CorrectionClassError(self.target, target.__class__)
        else:
            return
