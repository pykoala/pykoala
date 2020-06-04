"""
This file contains the relevant functions for loading in data from .yml files to the base PyKoala objects.
"""
from pathlib import Path
from types import SimpleNamespace

from camel import Camel

from src.koala.containers import (
    rssobject_configurator,
    scicalibdata_configurator,
    refdata_configurator,
    calibstarobject_configurator,
)


def _read_yaml_file(config_file):
    """ Open a .yml file

    Open and read the input .yaml config_file, additionally checks the file exists.

    Parameters
    ----------
    config_file: str
        location of the .yml file to read.

    Returns
    -------
    dict
        Dictionary containing the configuration information inside the .yml file
    """
    _check_file_exists(config_file)
    with open(config_file, "r") as f:
        return Camel().load(f.read())


def _check_file_exists(file_loc):
    """ check that a given string to a file location exists.

    Uses pathlib.Path to obtain if Pykoala can find the specific file, only check it exists, not that the file is
    appropriate.

    Parameters
    ----------
    file_loc : str
        location of the .fits or .dat file being accessed.

    Returns
    -------
    nothing
        Raises SystemExit if file specified is NOT found.
    """
    # TODO: add logging.
    if not Path(str(file_loc)).exists():
        raise SystemExit("Could not find file at {}".format(file_loc))


def _check_sci_files_exist(file_loc, input_sel):
    """ check that the user strings for the file locations inside the config_science_images.yml file exist.

    Parses the dictionaries inside the config_science_images.yml file and checks that each individual file location is
    accessible, the science_image_calibration_data dictionary is allowed to have None as locations as they may be
    generated using PyKoala

    Parameters
    ----------
    file_loc : list or dict.
        location of the .fits or .dat file being accessed.
    input_sel : str
        str selecting the input of the config_science_images.yml i.e. science_images, science_image_calibration_data or,
        pykoala_reference_data. Specifics is file_loc is going to be a dict or list.

    Returns
    -------
    nothing
        Raises SystemExit if file specified is NOT found.
    """
    # TODO: add logging.
    if input_sel == "sci_image":  # List of science images
        for file_loc in file_loc:
            _check_file_exists(file_loc)
    elif input_sel == "sci_image_cal_data":  # Dict of sci_image_cal_data
        for file_loc in file_loc.values():
            if file_loc is not None:
                _check_file_exists(file_loc)
    elif input_sel == "ref_data":  # Dict reference data
        for file_loc in file_loc.values():
            _check_file_exists(file_loc)
    else:
        raise SystemExit(
            "Type {} for input_secl {} is not recognised for sanitising inputs".format(
                type(file_loc), input_sel
            )
        )


def _check_cal_stars_files_exist(file_loc):
    """checks that the file locations within the config_calibration_stars yml file exists.

    Parses the dictionaries inside the config_calibration_star.yml file and checks that each individual file location is
    accessible. Allows absolute_flux_cal, telluric_correction, and response to be None as they may need to be generated

    Parameters
    ----------
    file_loc : list of dict
        list of dictionaries containing the read config_calibration_stars.yml file

    Returns
    -------
    nothing
        Raises SystemExit if errors are found
    """
    # TODO: add logging
    for input_dict in file_loc:
        if "sky_flat" in input_dict:
            _check_file_exists(input_dict["sky_flat"])
            # "Could not find the skyflat file, I was given {}".format(str(input_dict["sky_flat"]))

        elif "calib_star" in input_dict:
            for calib_star_image in input_dict[
                "calib_star"
            ]:  # Check if every calib_star .fits file is found
                _check_file_exists(calib_star_image)
                # "Could not find calibration star image at {}".format(calib_star_image)

            # Check if the name, absolute_flux_cal, telluric_correction, and response data files are accessible or None.
            if input_dict["name"] is None:
                raise SystemExit(
                    "Please specify a name for calibration star {} in the config_calibration_stars.yml file".format(
                        input_dict["calib_star"][0]
                    )
                )
            if input_dict["absolute_flux_cal"] is not None:
                _check_file_exists(input_dict["absolute_flux_cal"])
                # "given absolute flux cal but could not find .dat image at {}".format(input_dict["absolute_flux_cal"])
            if input_dict["telluric_correction"] is not None:
                _check_file_exists(input_dict["telluric_correction"])
                # "given telluric correction but could not find .dat image at {}".format(input_dict["telluric_correction"])
            if input_dict["response"] is not None:
                _check_file_exists(input_dict["response"])
                # "given response but could not find .dat image at {}".format(input_dict["response"])
        else:
            raise SystemExit(
                "config_calibration_stars.yml file format is inconsistent within the file - please check."
            )


def _get_science_files(config):
    """ Obtain the location of the science_images files

    Reads the science_images section of the config_science_images.yml file, checks files exists, and returns the strings
    of the location of the science images.

    Parameters
    ----------
    config: dict
        Dictionary containing the read config_science_images.yml file with the science_images location inside

    Returns
    -------
    list of str
        List of the file names of the science files to use in the data reduction
    """
    _check_sci_files_exist(config["science_images"], "sci_image")
    return config["science_images"]


def _get_science_image_calibration_data(config):
    """ Obtain the location for the science_image_calibration_data

    Reads the science_image_calibration_data section of the config_science_images.yml file, checks files exists, and
    returns the strings of the location of the calibration data. File locations can be None.

    Parameters
    ----------
    config: dict
        Dictionary containing the read config_science_images.yml file with the science_image_calibration_data location inside

    Returns
    -------
    dict
        Dictionary containing the three entries for throughput_calibration, flux_calibration, and telluric_correction
    """
    _check_sci_files_exist(
        config["science_image_calibration_data"], "sci_image_cal_data"
    )
    return config["science_image_calibration_data"]


def _get_pykoala_reference_data(config):
    """ Obtain the location for the pykoala_reference_data

    Reads the pykoala_reference_data section of the config_science_images.yml file, checks files exists, and returns the
    strings of the location of the reference data.

    Parameters
    ----------
    config: dict
        Dictionary containing the read config_science_images.yml file with the pykoala_reference_data location inside

    Returns
    -------
    dict
        Dictionary containing the entries for the reference data, skyline, skyline_rest, sso_extinction, and the
        list of absolute flux stars provided.
    """
    _check_sci_files_exist(config["pykoala_reference_data"], "ref_data")
    return config["pykoala_reference_data"]


def _get_cal_stars_data(config):
    """ Obtain the location for the files for generating calibration_data

    Reads the files_for_generating_calibration_data section of the config_calibration_stars.yml file, and returns the
    list containing the sky_flat and calib_star dicts.

    Parameters
    ----------
    config: dict
        Dictionary containing the read config_science_images.yml file

    Returns
    -------
    dict
        Dictionary containing (minimum) calibration stars and a sky flat
    """
    return config["files_for_generating_calibration_data"]


def _get_cal_stars_sky_flat(cal_stars):
    """ obtain the sky flat in the generating calibration data data

    cal_stars is a list of several dictionaries. Only one dictionary has the key sky_flat, corresponding to the sky_flat
    to be used in the calibration data. Function iterates over the dictionaries and finds which dict has "sky_flat" and
    returns the value - corresponding to the str representing the location of the sky flat file.

    Parameters
    ----------
    cal_stars list:
        list of dictionaries containing the sky flat and calibration stars

    Returns
    -------
        str
            The location of the sky flat file
    """
    cal_stars_data = _get_cal_stars_data(cal_stars)
    _check_cal_stars_files_exist(cal_stars_data)
    return [dict["sky_flat"] for dict in cal_stars_data if "sky_flat" in dict][0]
    # This list will ONLY EVER have ONE element as only one dictionary contains the sky_flat file.


def _get_cal_stars_stars(cal_stars):
    """ obtain the stars in the generating calibration data data

    cal_stars is a list of several dictionaries. Only one dictionary has the key sky_flat, corresponding to the
    sky_flat file used in the calibration data. Function iterates over the dictionaries and finds which dicts do
    NOT have "sky_flat" and makes sure the dictionary has entires for calib_star - if no calibartion stars then it
    remove the dictionary.

    Parameters
    ----------
    cal_stars list:
        list of dictionaries containing the sky flat and calibration stars

    Returns
    -------
        list
            list of dictionaries containing the individual calibration star data.
    """
    cal_stars_data = _get_cal_stars_data(cal_stars)
    _check_cal_stars_files_exist(cal_stars_data)
    return [dict for dict in cal_stars_data if "sky_flat" not in dict]


def load_science_images(config_file):
    """ Parse the data contained within the .yaml config science images file for Pykoala

    Parses the specific sections of the .yaml config, that is, science_images, science_image_calibration_data, and
    pykoala_reference_data into a SimpleNamespace which is passed to PyKoala.

    Parameters
    ----------
    config_file : str
        location of yaml file containing the initialization config for the science images for pykoala.

    Returns
    -------
    data
        data is a SimpleNamespace containing the base data objects of the PyKoala pipeline
        attrs are rss_sci containing the list of rss science images, calib_data containing the calibration data,
        ref_data containing the reference data

    Examples
    --------
    >>> load_science_images("path_to_data/config_science_images.yml")
    returns <class 'types.SimpleNamespace'>
    """
    # Load the .yaml file.
    config = _read_yaml_file(config_file)
    # Load the data paths specified in the config.
    input_sci_files = _get_science_files(config)  # Returns sanitised data
    input_sci_calib = _get_science_image_calibration_data(
        config
    )  # Returns sanitised data
    input_ref_data = _get_pykoala_reference_data(config)  # Returns sanitised data

    # Create a SimpleNamespace for passing the loaded data to the PyKoala pipeline.
    data = SimpleNamespace()
    # Populate the namespace
    data.sci_rss = [
        rssobject_configurator(sci_file_loc=sci_file_loc)
        for sci_file_loc in input_sci_files
    ]
    data.sci_calib_data = scicalibdata_configurator(scicalib_loc=input_sci_calib)
    data.ref_data = refdata_configurator(refdata_loc=input_ref_data)

    return data


def load_calibaration_files(calib_file):
    """ Parse the data contained within the .yaml config calibration stars file for Pykoala

    Parses the specific sections of the .yaml config, that is, sky_flat and calib_stars into a SimpleNamespace which is
    passed to PyKoala.

    Parameters
    ----------
    calib_file : str
        location of yaml file containing the calibration star config for the science images for pykoala.

    Returns
    -------
    data
        data is a SimpleNamespace containing the base data objects of the PyKoala pipeline
        attrs are skyflat containing the master skyflat image, calib_stars being a list of calibration stars

    Examples
    --------
    >>> load_calibaration_files("path_to_data/config_calibration_stars.yml")
    returns <class 'types.SimpleNamespace'>
    """
    # Load the .yaml file.
    config = _read_yaml_file(calib_file)
    # Load the skyflat and calibration stars file locations
    skyflat_file = _get_cal_stars_sky_flat(config)
    calib_stars = _get_cal_stars_stars(config)

    # Create a SimpleNamespace for passing the loaded data to the PyKoala pipeline.
    data = SimpleNamespace()
    # Populate the namespace
    data.skyflat = rssobject_configurator(skyflat_file, skyflat=True)
    data.calib_stars = [
        calibstarobject_configurator(calib_star) for calib_star in calib_stars
    ]

    return data
