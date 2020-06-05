# -*- coding: utf-8 -*-
"""
Contains the functions to create the base objects of Pykoala
"""
from copy import deepcopy

import numpy as np
from astropy.io import fits
from astropy.table import Table

from koala.utils.utils import FitsExt
from koala.containers import RSSObject, SciCalibData, RefData, GenCalibObject
from koala.containers._utils import (get_spaxel_quality, get_position_offsets, get_wavelength,
    get_intensity_and_variance, get_valid_wave_range)


def rssobject_configurator(sci_file_loc, reduced=False, skyflat=False):
    """ Parse .fits file into RSSObject

    Obtains the required parameters from the input data_file to create a RSSObject class object

    Parameters
    ----------
    sci_file_loc : str
        The location of the .yaml file containing a science image, parsed from the .yaml config science_images
    reduced : bool
        Boolean indicating if the image in the .fits main extension has been reduced by pykoala
    skyflat : bool
        Boolean indicating if the image in the .fits file is a sky flat .

    Returns
    -------
    RSSObject
        The base object used in pykoala
    """

    # Initialise the data and header information from the .fits file
    data_main = fits.getdata(sci_file_loc, FitsExt.main)
    data_var = fits.getdata(sci_file_loc, FitsExt.var)
    head_main = fits.getheader(sci_file_loc, FitsExt.main)
    head_var = fits.getheader(sci_file_loc, FitsExt.var)
    fibres_ifu = Table.read(sci_file_loc, hdu=FitsExt.fibres_ifu)

    # Physical position of the telescope
    _zdstart = head_main["ZDSTART"]
    _zdend = head_main["ZDEND"]
    pa = head_main["TEL_PA"]
    zd = (_zdstart + _zdend) / 2
    airmass = 1 / np.cos(np.radians((_zdstart + _zdend) / 2))
    # and other headers required
    obs_object = head_main["OBJECT"]
    exptime = head_main["EXPOSED"]
    grating = head_main["GRATID"]

    # Things that need to be calculated
    good_spaxels, bad_spaxels, all_spaxels = get_spaxel_quality(fibres_ifu_data=fibres_ifu)
    wavelength = get_wavelength(header_main=head_main)
    n_wave = len(wavelength)
    offset_ra_arcsec, offset_dec_arcsec = get_position_offsets(
        main_data=data_main,
        var_data=data_var,
        fibres_ifu_data=fibres_ifu
    )
    intensity, variance = get_intensity_and_variance(main_data=data_main, var_data=data_var,
                                                               fibres_ifu_data=fibres_ifu, n_wave=n_wave)
    n_spectra = intensity.shape[0]
    intensity_corrected = deepcopy(intensity)

    valid_wave_min, valid_wave_max = get_valid_wave_range(wavelength, grating, 0, 0)


    return RSSObject(
        data_file=sci_file_loc,
        data_main=data_main,
        data_var=data_var,
        head_main=head_main,
        head_var=head_var,
        fibres_ifu=fibres_ifu,
        reduced=reduced,  # This is set to false for initial input data.
        skyflat=skyflat,
        all_spaxels=all_spaxels,
        good_spaxels=good_spaxels,
        bad_spaxels=bad_spaxels,
        wavelength=wavelength,
        n_wave=n_wave,
        n_spectra=n_spectra,
        offset_ra_arcsec=offset_ra_arcsec,
        offset_dec_arcsec=offset_dec_arcsec,

        object=obs_object,
        exptime=exptime,
        grating=grating,

        intensity=intensity,
        intensity_corrected=intensity_corrected,
        variance=variance,

        valid_wave_min=valid_wave_min,
        valid_wave_max=valid_wave_max,

        pa=pa,
        zd=zd,
        airmass=airmass,

    )


def scicalibdata_configurator(scicalib_loc):
    """ Transform the science_image_calibration_data into an object for Pykoala

        Obtains the required parameters from the input file to create a SciCalibData class object

        Parameters
        ----------
        scicalib_loc : dict
            The science_image_calibration_data dict containing throughput_correction, flux_calibration, and telluric_correction

        Returns
        -------
        SciCalibData
            A simple container to pass around the science_image_calibration_data
    """
    return SciCalibData(
        throughput_correction=scicalib_loc["throughput_correction"],
        flux_calibration=scicalib_loc["flux_calibration"],
        telluric_correction=scicalib_loc["telluric_correction"]
    )


def refdata_configurator(refdata_loc):
    """ Transform the pykoala_reference_data into an object for Pykoala

        Obtains the required parameters from the input file to create a RefData class object

        Parameters
        ----------
        refdata_loc : dict
            The pykoala_reference_data dict containing skyline, skyline_rst, sso_extinction, and abs_flux_stars_dir

        Returns
        -------
        RefData
            A simple container to pass around the pykoala_reference_data
    """
    return RefData(
        skyline=refdata_loc["skyline"],
        skyline_rest=refdata_loc["skyline_rest"],
        sso_extinction=refdata_loc["sso_extinction"],
        abs_flux_stars_dir=refdata_loc["abs_flux_stars_dir"]
    )


def calibstarobject_configurator(calib_star):
    """ Transform the calibration data generator into a list of StarObjects

    Parameters
    ----------
    calib_star: dict
        dictionary containing the files_for_generating_calibration_data information from the .yaml config.

    Returns
    -------
    GenCalibOject
        A container for a calibration star image.
    """

    stars = [rssobject_configurator(star_loc) for star_loc in calib_star["calib_star"]]  # List of RSSObjects

    return GenCalibObject(
        star_name=calib_star["name"],
        star_files=stars,
        abs_flux_cal=calib_star["absolute_flux_cal"],
        telluric_cal=calib_star["telluric_correction"],
        response=calib_star["response"]
    )


def cube_object_configurator():
    pass
