"""
This module contains the parent class that represents the data used during the
reduction process
"""

import numpy as np

from koala.exceptions.exceptions import NoneAttrError


class DataContainer(object):
    """
    Abstract class for data containers.

    This class might represent any kind of astronomical data: raw fits files, Row Stacked Spectra
    obtained after tramline extraction or datacubes.

    Attributes
    ----------
    intensity
    intensity_corrected
    variance
    variance_corrected
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
        self.intensity_corrected = kwargs.get("intensity_corrected", None)
        self.variance = kwargs.get("variance", None)
        self.variance_corrected = kwargs.get("variance", None)
        self.intensity_units = kwargs.get("intensity_units", None)
        # Information and masking
        self.info = kwargs.get("info", dict())
        self.log = kwargs.get("log", dict(corrections=dict()))
        self.mask = kwargs.get("mask", None)
        self.mask_map = kwargs.get("mask_map", None)

    def save_log(self):
        pass

    def save_info(self):
        pass

    def is_in_info(self, key):
        """Check if a given keyword is stored in the info variable."""
        if self.info is None:
            raise NoneAttrError("info")
        if key in self.info.keys():
            return True
        else:
            return False

    def is_corrected(self, correction):
        if correction in self.log['corrections'].keys():
            return True
        else:
            return False

# Mr Krtxo \(ﾟ▽ﾟ)/
