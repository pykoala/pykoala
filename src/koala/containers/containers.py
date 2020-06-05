# -*- coding: utf-8 -*-
"""
File contains the class objects used throughout PyKoala
"""
import attr

from pathlib import Path


@attr.s
class RSSObject:
    """Core object for science image .fits file.

    This class is a container for a single science .fits image used in the PyKoala pipeline. This class is populated via
    the initialiser function rssobject_configurator.

    Attributes:
    ----------
    data_file: str
        The string containing the location of the file, e.g. my_data/M31image2.fits
    data_main: .fits ext table
        The 2-dimensional raw stacked spectra science image
    data_var: .fits ext table
        The 2-dimensional raw stacked spectra image variance
    TODO: complete me
    """

    # TODO: Things passed ARE needed 100% for reduction. everything else should be removed.

    # Data and headers of fits file.
    data_file = attr.ib()  # The .fits data file location.
    data_main = attr.ib()  # Data of main fits extension in 2dfdr .fits file.
    data_var = attr.ib()  # Data of variance fits extension in 2dfdr .fits file.
    head_main = attr.ib()  # Header of main fits extension in 2dfdr .fits file
    head_var = attr.ib()  # Header of variance fits extension in 2dfdr .fits file
    fibres_ifu = (
        attr.ib()
    )  # fibres ifu table fits extension in 2dfdr .fits file

    # Object flags
    reduced = (
        attr.ib()
    )  # Boolean flag indicating if the main data of the .fits files has been reduced through pykoala
    skyflat = attr.ib()  # Boolean flag indicating if the .fits file is a sky flat file.

    # Required headers from head_main
    pa = attr.ib()  # Position angle of telescope during exposure
    # TODO: change object name as it shadows pytohn object.
    object = attr.ib()  # Name of object being observed with telescope
    exptime = attr.ib()  # Exposure time of image
    grating = attr.ib()  # Name of grating used on the exposure of the image

    # Required information calculated from headers
    zd = attr.ib()  # Average zenith angle over the exposure.
    airmass = attr.ib()  # Air mass above the telescope during observation

    # Required spaxel information
    all_spaxels = attr.ib()
    good_spaxels = attr.ib()
    bad_spaxels = attr.ib()

    wavelength = attr.ib()
    n_wave = attr.ib()
    n_spectra = attr.ib()

    # todo: intensity should be named differently
    intensity = attr.ib()
    intensity_corrected = attr.ib()
    offset_ra_arcsec = attr.ib()
    offset_dec_arcsec = attr.ib()
    variance = attr.ib()

    valid_wave_min = attr.ib()
    valid_wave_max = attr.ib()

    def reduce(self):
        # this will not be a function! or a method!.
        raise SystemExit("This will never be a class method.")

    def is_reduced(self):
        """ Check if the RSSObject file has been reduced, meaning i.e. telluric correction was applied

        Returns
        -------
        bool
            bool indicating if the image in the file has been corrected
        """
        if self.reduced:
            return True
        else:
            return False

    def is_skyflat(self):
        """ Check if the RSSObject file is a skyflat file

        Returns
        -------
        bool
            bool indicating if the image in the file is a sky flat
        """
        if self.skyflat:
            return True
        else:
            return False

    # THIS IS temp for running old code. where print(file) is used and we need file to be formatted correctly
    def __add__(self, other):
        return str(self.data_file)

    def __radd__(self, other):
        return str(self.data_file)


@attr.s
class SciCalibData:
    """ Core object for containing the calibration data used in reducing the science images.

    This class is a container for the science images calibration data given in config_science_images.yml

     Attributes:
    ----------
    throughput_correction: str
    flux_calibration: str
    telluric_correction: str
    all_files_exist: bool
    """
    skyflat = attr.ib()
    throughput_correction = attr.ib()
    flux_calibration = attr.ib()
    telluric_correction = attr.ib()

    all_files_exist = attr.ib(init=False)

    def __attrs_post_init__(self):
        if (
            self.throughput_correction is not None
            and self.flux_calibration is not None
            and self.telluric_correction is not None
        ):
            self.all_files_exist = True
        else:
            self.all_files_exist = False


@attr.s
class RefData:
    """ Core object for containing the reference data used in PyKoala

    This class is a container for the Pykoala reference data given in config_science_images.yml

    Attributes:
    ----------
    skyline: str
    skyline_rest: str
    sso_extinction: str
    abs_flux_stars_dir: str
    abs_flux_stars: list of strings
    """

    skyline = attr.ib()
    skyline_rest = attr.ib()
    sso_extinction = attr.ib()
    abs_flux_stars_dir = attr.ib()

    abs_flux_stars = attr.ib(init=False)

    def __attrs_post_init__(self):
        # Obtain the list of different stars by iterating through the directory and obtaining the .dat files.
        self.abs_flux_stars = [
            file
            for file in Path(self.abs_flux_stars_dir).iterdir()
            if file.suffix == ".dat"
        ]


@attr.s
class GenCalibObject:
    """Core object for the calibration star .fits files.

    This class is a container for a list of reference stars used in the PyKoala pipeline.

    Attributes:
    ----------
    star_name: str
        string containing the name of the reference star. e.g. H600
    star_files: list of strings
        The list contains the strings pointing to the observations of 1 reference star
    abs_flux_cal: str
        The location of the reference star .dat file containing information about the flux of the star at specific
        wavelengths
    telluric_cal: str
    response: str
    """

    star_name = attr.ib()
    star_files = attr.ib()  # These are RSSobjects
    abs_flux_cal = attr.ib()
    telluric_cal = attr.ib()
    response = attr.ib()

