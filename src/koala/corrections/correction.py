"""
Parent Correction class
"""
from koala.exceptions import CorrectionClassError


class Correction(object):
    """
    This represents the abstract class of an astronomical correction to a given data (RSS or CUBE).
    """

    def __init__(self, target_class):
        self.target_class = target_class

    def check_target(self, target):
        if target.__class__ is not self.target_class:
            raise CorrectionClassError(self.target_class, target.__class__)
        else:
            return
