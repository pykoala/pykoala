# -*- coding: utf-8 -*-
from enum import IntEnum

class FitsExt(IntEnum):
    """
    Enumerated number class for improving readability of the extensions in the fits objects.
    """
    main = 0  # Contains main science image
    var = 1  # Contains variance data
    fibres_ifu = 2  # Contains information related to individual fibres
    ndf_class = 3  # not currently used in the code
    reduction_args = 4  # not currently used in the code, reduction arguments used in 2dfdr
    reduced = 5  # not currently used in the code

class FitsFibresIFUIndex(IntEnum):
    """
    Enumerated number class for accessing different index values in the fibres_ifu fits extension
    """
    # TODO: Finish description for the remaining indexes as they become known.
    spec_id = 0   # spectra ID, 1 to 1000
    quality_flag = 1  # Indicates if fibre is to be used. 0 = bad data, 1 = good data
    nspax = 2  # Unknown
    spec_len = 3  # Length of spectra, 2048 is KOALA expected value
    spec_sta = 4  # Unknown.
    ra_offset = 5  # RA offset of spectra
    dec_offset = 6  # DEC offset of spectra
    spec_group = 7  # Unknown, several spectra can be put into different groups?
    spax_id = 8  # Spaxel ID. Groups of 40 separated into 25 `elements`
    xgpos = 9  # Unknown
    ygpos = 10  # Unknown
    quality_flag_redundant = 11  # A redundancy for quality_flag. P for pass/good data, N for fail
    name = 12  # Unknown, maybe can assign each spectra a name in 2dfdr


def coord_range(rss_list):
    RA = [rss.RA_centre_deg + rss.offset_RA_arcsec / 3600.0 for rss in rss_list]
    RA_min = np.nanmin(RA)
    RA_max = np.nanmax(RA)
    DEC = [rss.DEC_centre_deg + rss.offset_DEC_arcsec / 3600.0 for rss in rss_list]
    DEC_min = np.nanmin(DEC)
    DEC_max = np.nanmax(DEC)
    return RA_min, RA_max, DEC_min, DEC_max


# Definition introduced by Matt
def median_absolute_deviation(x):
    """
    Derive the Median Absolute Deviation of an array
    Args:
        x (array): Array of numbers to find the median of

    Returns:
        float:
    """
    median_absolute_deviation = np.nanmedian(np.abs(x - np.nanmedian(x)))
    return median_absolute_deviation / 0.6745
