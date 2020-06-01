# -*- coding: utf-8 -*-
"""
This file has been created by copying almost verbatim from the __init__ and __main__ as part of the refactoring process.
This file is a work in progress.
"""
import numpy as np
from astropy.wcs import WCS


def get_spaxel_quality(fibres_ifu_data):
    """Obtain the 'good' and 'bad' spaxels which will and will not be used for science, respectively

    Parameters
    ----------
    fibres_ifu_data:

    Returns
    -------
    Three lists, list of good spaxels, bad spaxels, and all spaxels
    """
    all_spaxels = list(range(len(fibres_ifu_data)))
    quality_flag = [fibres_ifu_data for i in all_spaxels]
    good_spaxels = [i for i in all_spaxels if quality_flag[i] == 1]
    bad_spaxels = [i for i in all_spaxels if quality_flag[i] == 0]
    return good_spaxels, bad_spaxels, all_spaxels


def get_wavelength(header_main):
    """ gets the wavelength range?

    This is more or less copied verbatim

    Parameters
    ----------
    header_main: header of the main .fits extension

    Returns
    -------

    """
    wcskoala = WCS(header_main)
    index_wave = np.arange(header_main["NAXIS1"])
    wavelength = wcskoala.dropaxis(1).wcs_pix2world(index_wave, 0)[
        0
    ]  # can make 1 line, but what does it even do
    return wavelength


def get_position_offsets(main_data, var_data, fibres_ifu_data):
    """

    Parameters
    ----------
    main_data:

    var_data:

    fibres_ifu_data

    Returns
    -------

    """
    good_spaxels, bad_spaxels, all_spaxels = get_spaxel_quality(
        fibres_ifu_data=fibres_ifu_data
    )
    if len(bad_spaxels) == 0:
        offset_RA_arcsec_ = []
        offset_DEC_arcsec_ = []
        for i in range(len(good_spaxels)):
            offset_RA_arcsec_.append(fibres_ifu_data[i][5])
            offset_DEC_arcsec_.append(fibres_ifu_data[i][6])
        offset_RA_arcsec = np.array(offset_RA_arcsec_)
        offset_DEC_arcsec = np.array(offset_DEC_arcsec_)
        # TODO: I am aware that all these functions are calling themselves all the time. like things are being calculated 3 times instead of 1.

    else:
        offset_RA_arcsec = np.array([fibres_ifu_data[i][5] for i in good_spaxels])
        offset_DEC_arcsec = np.array([fibres_ifu_data[i][6] for i in good_spaxels])
        ID = np.array(
            [fibres_ifu_data[i][0] for i in good_spaxels]
        )  # These are the good fibres  # TODO: look into ID.

    return offset_RA_arcsec, offset_DEC_arcsec


def _get_intensity(main_data, fibres_ifu_data):
    """ Returns the intensity of the spaxels which are science usable

    Parameters
    ----------
    main_data:

    fibres_ifu_data:

    Returns
    -------

    """
    good_spaxels, _, _ = get_spaxel_quality(fibres_ifu_data=fibres_ifu_data)
    return main_data[good_spaxels]


def get_intensity_and_variance(main_data, var_data, fibres_ifu_data, n_wave):
    """ Obtain the variance, which is...

    Parameters
    ----------
    main_data:
    var_data:
    fibres_ifu_data:
    n_wave:

    Returns
    -------

    """
    good_spaxels, bad_spaxels, all_spaxels = get_spaxel_quality(
        fibres_ifu_data=fibres_ifu_data
    )
    intensity = _get_intensity(main_data=main_data, fibres_ifu_data=fibres_ifu_data)

    if len(bad_spaxels) == 0:
        variance = np.zeros_like(intensity)  # CHECK FOR ERRORS
    else:
        variance = var_data[good_spaxels]  # CHECK FOR ERRORS

    # Now we check the shape.
    if variance.shape != intensity.shape:
        print(
            "\n* ERROR: * the intensity and variance matrices are {} and {} respectively\n".format(
                intensity.shape, variance.shape
            )
        )
        raise ValueError
    n_dim = len(intensity.shape)
    if n_dim == 2:
        pass
    elif n_dim == 1:
        intensity = intensity.reshape((1, n_wave))
        variance = variance.reshape((1, n_wave))
    else:
        print(
            "\n* ERROR: * the intensity matrix supplied has {} dimensions\n".format(
                n_dim
            )
        )
        raise ValueError

    return intensity, variance


def get_valid_wave_range(wavelength, grating, valid_wave_min, valid_wave_max):
    """

    Parameters
    ----------
    wavelength:
    grating:
    valid_wave_min:
    valid_wave_max:

    Returns
    -------

    """
    # TODO: want to remove valid_wave_max and _min from the function entirely and the structure
    if valid_wave_min == 0 and valid_wave_max == 0:
        valid_wave_min = np.min(wavelength)
        valid_wave_max = np.max(wavelength)
    # Angel commented this out, it is to be updated and fixed for all gratings. I've updated the code to py3
    # if grating == "1000R":
    #     valid_wave_min = 6600.    # CHECK ALL OF THIS...
    #     valid_wave_max = 6800.
    #     print("  For 1000R, we use the [6200, 7400] range.")
    # elif grating == "1500V":
    #     valid_wave_min = np.min(wavelength)
    #     valid_wave_max = np.max(wavelength)
    #     print("  For 1500V, we use all the range.")
    # elif grating == "580V":
    #     valid_wave_min = 3650.
    #     valid_wave_max = 5700.
    #     print("  For 580V, we use the [3650, 5700] range.")
    # elif grating == "1500V":
    #     valid_wave_min = 4620.     #4550
    #     valid_wave_max = 5350.     #5350
    #     print("  For 1500V, we use the [4550, 5350] range.")
    else:
        valid_wave_min = valid_wave_min
        valid_wave_max = valid_wave_max
        print(
            "  As specified, we use the [",
            valid_wave_min,
            " , ",
            valid_wave_max,
            "] range.",
        )

    return valid_wave_min, valid_wave_max
