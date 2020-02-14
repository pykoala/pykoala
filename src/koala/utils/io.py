from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div

from astropy.io import fits
from astropy.wcs import WCS

from pysynphot import observation
from pysynphot import spectrum

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys

from scipy import interpolate, signal, optimize
from scipy.optimize import curve_fit
import scipy.signal as sig

# from scipy.optimize import leastsq

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.ndimage.interpolation import shift

import datetime
import copy

import os.path as pth

from .._version import get_versions

version = get_versions()["version"]
del get_versions


def read_table(fichero, formato):
    """
    Read data from and txt file (sorted by columns), the type of data
    (string, integer or float) MUST be given in "formato".
    This routine will ONLY read the columns for which "formato" is defined.
    E.g. for a txt file with 7 data columns, using formato=["f", "f", "s"] will only read the 3 first columns.

    Parameters
    ----------
    fichero:
        txt file to be read
    formato:
        List with the format of each column of the data, using:\n
        "i" for a integer\n
        "f" for a float\n
        "s" for a string (text)

    Example
    -------
    >>> el_center,el_fnl,el_name = read_table("lineas_c89_python.dat", ["f", "f", "s"] )
    """

    datos_len = len(formato)
    datos = [[] for x in range(datos_len)]
    for i in range(0, datos_len):
        if formato[i] == "i":
            datos[i] = np.loadtxt(
                fichero, skiprows=0, unpack=True, usecols=[i], dtype=int
            )
        if formato[i] == "s":
            datos[i] = np.loadtxt(
                fichero, skiprows=0, unpack=True, usecols=[i], dtype=str
            )
        if formato[i] == "f":
            datos[i] = np.loadtxt(
                fichero, skiprows=0, unpack=True, usecols=[i], dtype=float
            )
    return datos


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def array_to_text_file(data, filename="array.dat"):
    """
    Write array into a text file.

    Parameters
    ----------
    data: float
        flux per wavelength
    filename: string (default = "array.dat")
        name of the text file where the data will be written.

    Example
    -------
    >>> array_to_text_file(data, filename="data.dat" )
    """
    f = open(filename, "w")
    for i in range(len(data)):
        escribe = np.str(data[i]) + " \n"
        f.write(escribe)
    f.close()
    print("\n> Array saved in text file", filename, " !!")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def spectrum_to_text_file(wavelength, flux, filename="spectrum.txt"):
    """
    Write given 1D spectrum into a text file.

    Parameters
    ----------
    wavelength: float
        wavelength.
    flux: float
        flux per wavelength
    filename: string (default = "spectrum.txt")
        name of the text file where the data will be written.

    Example
    -------
    >>> spectrum_to_text_file(wavelength, spectrum, filename="fantastic_spectrum.txt" )
    """
    f = open(filename, "w")
    for i in range(len(wavelength)):
        escribe = np.str(wavelength[i]) + "  " + np.str(flux[i]) + " \n"
        f.write(escribe)
    f.close()
    print("\n> Spectrum saved in text file", filename, " !!")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def spectrum_to_fits_file(
    wavelength,
    flux,
    filename="spectrum.fits",
    name="spectrum",
    exptime=1,
    CRVAL1_CDELT1_CRPIX1=[0, 0, 0],
):
    """
    Routine to save a given 1D spectrum into a fits file.

    If CRVAL1_CDELT1_CRPIX1 it not given, it assumes a LINEAR dispersion,
    with Delta_pix = (wavelength[-1]-wavelength[0])/(len(wavelength)-1).

    Parameters
    ----------
    wavelength: float
        wavelength.
    flux: float
        flux per wavelength
    filename: string (default = "spectrum.fits")
        name of the fits file where the data will be written.
    Example
    -------
    >>> spectrum_to_fits_file(wavelength, spectrum, filename="fantastic_spectrum.fits",
                              exptime=600,name="POX 4")
    """
    hdu = fits.PrimaryHDU()
    hdu.data = flux
    hdu.header["ORIGIN"] = "Data from KOALA Python scripts"
    # Wavelength calibration
    hdu.header["NAXIS"] = 1
    hdu.header["NAXIS1"] = len(wavelength)
    hdu.header["CTYPE1"] = "Wavelength"
    hdu.header["CUNIT1"] = "Angstroms"
    if CRVAL1_CDELT1_CRPIX1[0] == 0:
        hdu.header["CRVAL1"] = wavelength[0]
        hdu.header["CRPIX1"] = 1.0
        hdu.header["CDELT1"] = old_div((wavelength[-1] - wavelength[0]), (len(wavelength) - 1))
    else:
        hdu.header["CRVAL1"] = CRVAL1_CDELT1_CRPIX1[
            0
        ]  # 7.692370611909E+03  / Co-ordinate value of axis 1
        hdu.header["CDELT1"] = CRVAL1_CDELT1_CRPIX1[1]  # 1.575182431607E+00
        hdu.header["CRPIX1"] = CRVAL1_CDELT1_CRPIX1[
            2
        ]  # 1024. / Reference pixel along axis 1
    # Extra info
    hdu.header["OBJECT"] = name
    hdu.header["TOTALEXP"] = exptime
    hdu.header["HISTORY"] = "Spectrum derived using the KOALA Python pipeline"
    hdu.header[
        "HISTORY"
    ] = "Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany et al."
    hdu.header["HISTORY"] = version
    now = datetime.datetime.now()
    hdu.header["HISTORY"] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    hdu.header["DATE"] = now.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )  # '2002-09-16T18:52:44'   # /Date of FITS file creation

    hdu.writeto(filename, overwrite=True)
    print("\n> Spectrum saved in fits file", filename, " !!")
    if name == "spectrum":
        print("  No name given to the spectrum, named 'spectrum'.")
    if exptime == 1:
        print("  No exposition time given, assumed exptime = 1")
    if CRVAL1_CDELT1_CRPIX1[0] == 0:
        print("  CRVAL1_CDELT1_CRPIX1 values not given, using ", wavelength[0], "1", old_div((
            wavelength[-1] - wavelength[0]
        ), (len(wavelength) - 1)))





def save_bluered_fits_file(
    blue_cube,
    red_cube,
    fits_file,
    fcalb=[0],
    fcalr=[0],
    ADR=False,
    objeto="",
    description="",
    trimb=[0],
    trimr=[0],
):
    """
    Routine combine blue + red files and save result in a fits file fits file

    Parameters
    ----------
    Combined cube:
        Combined cube
    Header:
        Header
    """

    # Prepare the red+blue datacube
    print("\n> Combining blue + red datacubes...")

    if trimb[0] == 0:
        lb = blue_cube.wavelength
        b = blue_cube.data
    else:
        print("  Trimming blue cube in range [{},{}]".format(trimb[0], trimb[1]))
        index_min = np.searchsorted(blue_cube.wavelength, trimb[0])
        index_max = np.searchsorted(blue_cube.wavelength, trimb[1]) + 1
        lb = blue_cube.wavelength[index_min:index_max]
        b = blue_cube.data[index_min:index_max]
        fcalb = fcalb[index_min:index_max]

    if trimr[0] == 0:
        lr = red_cube.wavelength
        r = red_cube.data
    else:
        print("  Trimming red cube in range [{},{}]".format(trimr[0], trimr[1]))
        index_min = np.searchsorted(red_cube.wavelength, trimr[0])
        index_max = np.searchsorted(red_cube.wavelength, trimr[1]) + 1
        lr = red_cube.wavelength[index_min:index_max]
        r = red_cube.data[index_min:index_max]
        fcalr = fcalr[index_min:index_max]

    l = np.concatenate((lb, lr), axis=0)
    blue_red_datacube = np.concatenate((b, r), axis=0)

    if fcalb[0] == 0:
        print("  No absolute flux calibration included")
    else:
        flux_calibration = np.concatenate((fcalb, fcalr), axis=0)

    if objeto == "":
        description = "UNKNOWN OBJECT"

    fits_image_hdu = fits.PrimaryHDU(blue_red_datacube)
    #    errors = combined_cube.data*0  ### TO BE DONE
    #    error_hdu = fits.ImageHDU(errors)

    wavelengths_hdu = fits.ImageHDU(l)

    fits_image_hdu.header["ORIGIN"] = "Combined datacube from KOALA Python scripts"

    fits_image_hdu.header["BITPIX"] = 16
    fits_image_hdu.header["NAXIS"] = 3
    fits_image_hdu.header["NAXIS1"] = len(l)
    fits_image_hdu.header["NAXIS2"] = blue_red_datacube.shape[1]  # CHECK !!!!!!!
    fits_image_hdu.header["NAXIS2"] = blue_red_datacube.shape[2]

    fits_image_hdu.header["OBJECT"] = objeto
    fits_image_hdu.header["RAcen"] = blue_cube.RA_centre_deg
    fits_image_hdu.header["DECcen"] = blue_cube.DEC_centre_deg
    fits_image_hdu.header["PIXsize"] = blue_cube.pixel_size_arcsec
    fits_image_hdu.header["Ncols"] = blue_cube.data.shape[2]
    fits_image_hdu.header["Nrows"] = blue_cube.data.shape[1]
    fits_image_hdu.header["PA"] = blue_cube.PA
    #    fits_image_hdu.header["CTYPE1"] = 'LINEAR  '
    #    fits_image_hdu.header["CRVAL1"] = wavelength[0]
    #    fits_image_hdu.header["CRPIX1"] = 1.
    #    fits_image_hdu.header["CDELT1"] = (wavelength[-1]-wavelength[0])/len(wavelength)
    #    fits_image_hdu.header["CD1_1"]  = (wavelength[-1]-wavelength[0])/len(wavelength)
    #    fits_image_hdu.header["LTM1_1"] = 1.

    fits_image_hdu.header[
        "COFILES"
    ] = blue_cube.number_of_combined_files  # Number of combined files
    fits_image_hdu.header["OFFSETS"] = blue_cube.offsets_files  # Offsets

    fits_image_hdu.header["ADRCOR"] = np.str(ADR)

    if fcalb[0] == 0:
        fits_image_hdu.header["FCAL"] = "False"
        flux_correction_hdu = fits.ImageHDU(0 * l)
    else:
        flux_correction = flux_calibration
        flux_correction_hdu = fits.ImageHDU(flux_correction)
        fits_image_hdu.header["FCAL"] = "True"

    if description == "":
        description = flux_calibration.description
    fits_image_hdu.header["DESCRIP"] = description

    #    hdu_list = fits.HDUList([fits_image_hdu, error_hdu])
    hdu_list = fits.HDUList([fits_image_hdu, wavelengths_hdu, flux_correction_hdu])
    hdu_list.writeto(fits_file, overwrite=True)
    print("\n> Combined datacube saved to file ", fits_file)



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def save_fits_file(combined_cube, fits_file, description="", ADR=False):  # fcal=[0],
    """
    Routine to save a fits file

    Parameters
    ----------
    Combined cube:
        Combined cube
    Header:
        Header
    """
    fits_image_hdu = fits.PrimaryHDU(combined_cube.data)
    #    errors = combined_cube.data*0  ### TO BE DONE
    #    error_hdu = fits.ImageHDU(errors)

    # wavelength =  combined_cube.wavelength

    fits_image_hdu.header["HISTORY"] = "Combined datacube from KOALA Python pipeline"
    fits_image_hdu.header[
        "HISTORY"
    ] = "Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany et al."
    fits_image_hdu.header["HISTORY"] = version  # 'Version 0.10 - 12th February 2019'
    now = datetime.datetime.now()
    fits_image_hdu.header["HISTORY"] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    fits_image_hdu.header["DATE"] = now.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )  # '2002-09-16T18:52:44'   # /Date of FITS file creation

    fits_image_hdu.header["BITPIX"] = 16

    fits_image_hdu.header["ORIGIN"] = "AAO"  # / Originating Institution
    fits_image_hdu.header["TELESCOP"] = "Anglo-Australian Telescope"  # / Telescope Name
    fits_image_hdu.header["ALT_OBS"] = 1164  # / Altitude of observatory in metres
    fits_image_hdu.header["LAT_OBS"] = -31.27704  # / Observatory latitude in degrees
    fits_image_hdu.header["LONG_OBS"] = 149.0661  # / Observatory longitude in degrees

    fits_image_hdu.header["INSTRUME"] = "AAOMEGA-KOALA"  # / Instrument in use
    fits_image_hdu.header["GRATID"] = combined_cube.RSS.grating  # / Disperser ID
    if combined_cube.RSS.grating == "385R":
        SPECTID = "RD"
    if combined_cube.RSS.grating == "580V":
        SPECTID = "BD"
    if combined_cube.RSS.grating == "1000R":
        SPECTID = "RD"
    if combined_cube.RSS.grating == "1000I":
        SPECTID = "RD"
    fits_image_hdu.header["SPECTID"] = SPECTID  # / Spectrograph ID

    fits_image_hdu.header[
        "DICHROIC"
    ] = "X5700"  # / Dichroic name   ---> CHANGE if using X6700!!

    fits_image_hdu.header["OBJECT"] = combined_cube.object
    fits_image_hdu.header["TOTALEXP"] = combined_cube.total_exptime

    fits_image_hdu.header["NAXIS"] = 3  # / number of array dimensions
    fits_image_hdu.header["NAXIS1"] = combined_cube.data.shape[1]  # CHECK !!!!!!!
    fits_image_hdu.header["NAXIS2"] = combined_cube.data.shape[2]
    fits_image_hdu.header["NAXIS3"] = combined_cube.data.shape[0]

    # WCS
    fits_image_hdu.header["RADECSYS"] = "FK5"  # / FK5 reference system
    fits_image_hdu.header["EQUINOX"] = 2000  # / [yr] Equinox of equatorial coordinates
    fits_image_hdu.header["WCSAXES"] = 3  # / Number of coordinate axes

    fits_image_hdu.header["CRPIX1"] = (
        combined_cube.data.shape[1] / 2.0
    )  # / Pixel coordinate of reference point
    fits_image_hdu.header["CDELT1"] = (
        -combined_cube.pixel_size_arcsec / 3600.0
    )  # / Coordinate increment at reference point
    fits_image_hdu.header[
        "CTYPE1"
    ] = "RA--TAN"  # 'DEGREE'                               # / Coordinate type code
    fits_image_hdu.header[
        "CRVAL1"
    ] = combined_cube.RA_centre_deg  # / Coordinate value at reference point

    fits_image_hdu.header["CRPIX2"] = (
        combined_cube.data.shape[2] / 2.0
    )  # / Pixel coordinate of reference point
    fits_image_hdu.header["CDELT2"] = (
        combined_cube.pixel_size_arcsec / 3600.0
    )  # Coordinate increment at reference point
    fits_image_hdu.header[
        "CTYPE2"
    ] = "DEC--TAN"  # 'DEGREE'                               # / Coordinate type code
    fits_image_hdu.header[
        "CRVAL2"
    ] = combined_cube.DEC_centre_deg  # / Coordinate value at reference point

    fits_image_hdu.header["RAcen"] = combined_cube.RA_centre_deg
    fits_image_hdu.header["DECcen"] = combined_cube.DEC_centre_deg
    fits_image_hdu.header["PIXsize"] = combined_cube.pixel_size_arcsec
    fits_image_hdu.header["Ncols"] = combined_cube.data.shape[2]
    fits_image_hdu.header["Nrows"] = combined_cube.data.shape[1]
    fits_image_hdu.header["PA"] = combined_cube.PA

    # Wavelength calibration
    fits_image_hdu.header["CTYPE3"] = "Wavelength"  # / Label for axis 3
    fits_image_hdu.header["CUNIT3"] = "Angstroms"  # / Units for axis 3
    fits_image_hdu.header["CRVAL3"] = combined_cube.CRVAL1_CDELT1_CRPIX1[
        0
    ]  # 7.692370611909E+03  / Co-ordinate value of axis 3
    fits_image_hdu.header["CDELT3"] = combined_cube.CRVAL1_CDELT1_CRPIX1[
        1
    ]  # 1.575182431607E+00
    fits_image_hdu.header["CRPIX3"] = combined_cube.CRVAL1_CDELT1_CRPIX1[
        2
    ]  # 1024. / Reference pixel along axis 3

    fits_image_hdu.header["COFILES"] = (
        len(combined_cube.offsets_files) + 1
    )  # Number of combined files
    offsets_text = " "
    for i in range(len(combined_cube.offsets_files)):
        if i != 0:
            offsets_text = offsets_text + "  ,  "
        offsets_text = (
            offsets_text
            + np.str(np.around(combined_cube.offsets_files[i][0], 3))
            + " "
            + np.str(np.around(combined_cube.offsets_files[i][1], 3))
        )
    fits_image_hdu.header["OFFSETS"] = offsets_text  # Offsets

    fits_image_hdu.header["ADRCOR"] = np.str(ADR)

    if np.nanmedian(combined_cube.data) > 1:
        fits_image_hdu.header["FCAL"] = "False"
        fits_image_hdu.header["F_UNITS"] = "Counts"
        # flux_correction_hdu = fits.ImageHDU(0*wavelength)
    else:
        # flux_correction = fcal
        # flux_correction_hdu = fits.ImageHDU(flux_correction)
        fits_image_hdu.header["FCAL"] = "True"
        fits_image_hdu.header["F_UNITS"] = "erg s-1 cm-2 A-1"

    if description == "":
        description = combined_cube.description
    fits_image_hdu.header["DESCRIP"] = description

    for file in range(len(combined_cube.rss_list)):
        fits_image_hdu.header["HISTORY"] = (
            "RSS file " + np.str(file + 1) + ":" + combined_cube.rss_list[file]
        )

    #    hdu_list = fits.HDUList([fits_image_hdu, error_hdu])
    #    hdu_list = fits.HDUList([fits_image_hdu, wavelengths_hdu, flux_correction_hdu])
    hdu_list = fits.HDUList([fits_image_hdu])  # , flux_correction_hdu])

    hdu_list.writeto(fits_file, overwrite=True)
    print("\n> Combined datacube saved to file:", fits_file)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def save_rss_fits(
    rss, data=[[0], [0]], fits_file="RSS_rss.fits", description=""
):  # fcal=[0],     # TASK_save_rss_fits
    """
    Routine to save RSS data as fits

    Parameters
    ----------
    rss is the rss
    description = if you want to add a description
    """
    if np.nanmedian(data[0]) == 0:
        data = rss.intensity_corrected
        print("\n> Using rss.intensity_corrected of given RSS file to create fits file...")
    else:
        if len(np.array(data).shape) != 2:
            print("\n> The data provided are NOT valid, as they have a shape", data.shape)
            print("  Using rss.intensity_corrected instead to create a RSS fits file !")
            data = rss.intensity_corrected
        else:
            print("\n> Using the data provided + structure of given RSS file to create fits file...")
    fits_image_hdu = fits.PrimaryHDU(data)

    fits_image_hdu.header["HISTORY"] = "RSS from KOALA Python pipeline"
    fits_image_hdu.header[
        "HISTORY"
    ] = "Developed by Angel Lopez-Sanchez, Yago Ascasibar, Lluis Galbany et al."
    fits_image_hdu.header["HISTORY"] = version  # 'Version 0.10 - 12th February 2019'
    now = datetime.datetime.now()
    fits_image_hdu.header["HISTORY"] = now.strftime("Created on %d %b %Y, %H:%M:%S")
    fits_image_hdu.header["DATE"] = now.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )  # '2002-09-16T18:52:44'   # /Date of FITS file creation

    fits_image_hdu.header["BITPIX"] = 16

    fits_image_hdu.header["ORIGIN"] = "AAO"  # / Originating Institution
    fits_image_hdu.header["TELESCOP"] = "Anglo-Australian Telescope"  # / Telescope Name
    fits_image_hdu.header["ALT_OBS"] = 1164  # / Altitude of observatory in metres
    fits_image_hdu.header["LAT_OBS"] = -31.27704  # / Observatory latitude in degrees
    fits_image_hdu.header["LONG_OBS"] = 149.0661  # / Observatory longitude in degrees

    fits_image_hdu.header["INSTRUME"] = "AAOMEGA-KOALA"  # / Instrument in use
    fits_image_hdu.header["GRATID"] = rss.grating  # / Disperser ID
    if rss.grating == "385R":
        SPECTID = "RD"
    if rss.grating == "580V":
        SPECTID = "BD"
    if rss.grating == "1000R":
        SPECTID = "RD"
    if rss.grating == "1000I":
        SPECTID = "RD"
    fits_image_hdu.header["SPECTID"] = SPECTID  # / Spectrograph ID

    fits_image_hdu.header[
        "DICHROIC"
    ] = "X5700"  # / Dichroic name   ---> CHANGE if using X6700!!

    fits_image_hdu.header["OBJECT"] = rss.object
    fits_image_hdu.header["EXPOSED"] = rss.exptime
    fits_image_hdu.header["ZDSTART"] = rss.ZDSTART
    fits_image_hdu.header["ZDEND"] = rss.ZDEND

    fits_image_hdu.header["NAXIS"] = 2  # / number of array dimensions
    fits_image_hdu.header["NAXIS1"] = rss.intensity_corrected.shape[0]
    fits_image_hdu.header["NAXIS2"] = rss.intensity_corrected.shape[1]

    fits_image_hdu.header["RAcen"] = rss.RA_centre_deg
    fits_image_hdu.header["DECcen"] = rss.DEC_centre_deg
    fits_image_hdu.header["TEL_PA"] = rss.PA

    fits_image_hdu.header["CTYPE2"] = "Fibre number"  # / Label for axis 2
    fits_image_hdu.header["CUNIT2"] = " "  # / Units for axis 2
    fits_image_hdu.header["CTYPE1"] = "Wavelength"  # / Label for axis 2
    fits_image_hdu.header["CUNIT1"] = "Angstroms"  # / Units for axis 2

    fits_image_hdu.header["CRVAL1"] = rss.CRVAL1_CDELT1_CRPIX1[
        0
    ]  # / Co-ordinate value of axis 2
    fits_image_hdu.header["CDELT1"] = rss.CRVAL1_CDELT1_CRPIX1[1]  #
    fits_image_hdu.header["CRPIX1"] = rss.CRVAL1_CDELT1_CRPIX1[
        2
    ]  # 1024. / Reference pixel along axis 2
    fits_image_hdu.header[
        "CRVAL2"
    ] = 5.000000000000e-01  # / Co-ordinate value of axis 2
    fits_image_hdu.header[
        "CDELT2"
    ] = 1.000000000000e00  # / Co-ordinate increment along axis 2
    fits_image_hdu.header[
        "CRPIX2"
    ] = 1.000000000000e00  # / Reference pixel along axis 2

    if description == "":
        description = rss.description
    fits_image_hdu.header["DESCRIP"] = description

    # TO BE DONE
    errors = [0]  # TO BE DONE
    error_hdu = fits.ImageHDU(errors)

    # Header 2 with the RA and DEC info!

    header2_all_fibres = rss.header2_data
    header2_good_fibre = []
    header2_original_fibre = []
    header2_new_fibre = []
    header2_delta_RA = []
    header2_delta_DEC = []
    header2_2048 = []
    header2_0 = []

    fibre = 1
    for i in range(len(header2_all_fibres)):
        if header2_all_fibres[i][1] == 1:
            header2_original_fibre.append(i + 1)
            header2_new_fibre.append(fibre)
            header2_good_fibre.append(1)
            header2_delta_RA.append(header2_all_fibres[i][5])
            header2_delta_DEC.append(header2_all_fibres[i][6])
            header2_2048.append(2048)
            header2_0.append(0)
            fibre = fibre + 1

    #    header2_=[header2_new_fibre, header2_good_fibre, header2_good_fibre, header2_2048, header2_0,  header2_delta_RA,  header2_delta_DEC,  header2_original_fibre]
    #    header2 = np.array(header2_).T.tolist()
    #    header2_hdu = fits.ImageHDU(header2)

    col1 = fits.Column(name="Fibre", format="I", array=np.array(header2_new_fibre))
    col2 = fits.Column(name="Status", format="I", array=np.array(header2_good_fibre))
    col3 = fits.Column(name="Ones", format="I", array=np.array(header2_good_fibre))
    col4 = fits.Column(name="Wavelengths", format="I", array=np.array(header2_2048))
    col5 = fits.Column(name="Zeros", format="I", array=np.array(header2_0))
    col6 = fits.Column(name="Delta_RA", format="D", array=np.array(header2_delta_RA))
    col7 = fits.Column(name="Delta_Dec", format="D", array=np.array(header2_delta_DEC))
    col8 = fits.Column(
        name="Fibre_OLD", format="I", array=np.array(header2_original_fibre)
    )

    cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7, col8])
    header2_hdu = fits.BinTableHDU.from_columns(cols)

    header2_hdu.header["CENRA"] = old_div(rss.RA_centre_deg, (
        old_div(180, np.pi)
    ))  # Must be in radians
    header2_hdu.header["CENDEC"] = old_div(rss.DEC_centre_deg, (old_div(180, np.pi)))

    hdu_list = fits.HDUList(
        [fits_image_hdu, error_hdu, header2_hdu]
    )  # hdu_list = fits.HDUList([fits_image_hdu, wavelengths_hdu, flux_correction_hdu])

    hdu_list.writeto(fits_file, overwrite=True)
    print("  RSS data saved to file ", fits_file)



