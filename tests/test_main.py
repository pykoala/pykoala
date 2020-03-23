import os.path as pth

import pytest

from koala import KOALA_RSS, KOALA_reduce, DATA_PATH, read_table
from koala.cli import reduce_koala_data_main

DO_PLOTTING = False

DATE = "20180310"
GRATING = "385R"
PIXEL_SIZE = 0.6  # Just 0.1 precision
KERNEL_SIZE = 1.25
OBJECT = "POX4"
DESCRIPTION = "POX4 CUBE"

PATH_SKYFLAT = pth.join(DATA_PATH, GRATING)

SKY_FLAT_RED_FILENAME = "10mar2_combined.fits"
FLUX_CALIBRATION_RED_FILENAME = "flux_calibration_20180310_385R_0p6_1k8.dat"
TELLURIC_CORRECTION_RED_FILENAME = "telluric_correction_20180310_385R_0p6_1k25.dat"

FILE_SKY_FLAT_RED = pth.join(
    PATH_SKYFLAT, SKY_FLAT_RED_FILENAME
) # FILE NOT DIVIDED BY THE FLAT
FLUX_CAL_FILE = pth.join(DATA_PATH, FLUX_CALIBRATION_RED_FILENAME)
TELLURIC_CORRECTION_FILE = pth.join(DATA_PATH, TELLURIC_CORRECTION_RED_FILENAME)

SCIENCE_RED_2_FILENAME = pth.join(DATA_PATH, GRATING, "10mar20092red.fits")
SCIENCE_RED_3_FILENAME = pth.join(DATA_PATH, GRATING, "10mar20093red.fits")

RSS_LIST = [SCIENCE_RED_2_FILENAME, SCIENCE_RED_3_FILENAME]


def test_main(tmpdir):
    # This should test a main function, currently this is a naive copy of what
    # was in __main__.py

    skyflat_red = KOALA_RSS(
        FILE_SKY_FLAT_RED,
        flat="",
        apply_throughput=False,
        sky_method="none",
        do_extinction=False,
        correct_ccd_defects = False,
        correct_high_cosmics = False,
        clip_high = 100,
        step_ccd = 50,
        plot=DO_PLOTTING,
    )

    skyflat_red.find_relative_throughput(
        ymin=0,
        ymax=800000,
        wave_min_scale=6300,
        wave_max_scale=6500,
        plot=DO_PLOTTING,
    )

    _, flux_calibration = read_table(FLUX_CAL_FILE, ["f", "f"] )
    _, telluric_correction = read_table(TELLURIC_CORRECTION_FILE, ["f", "f"])

    sky_r3 = KOALA_RSS(
        SCIENCE_RED_3_FILENAME,
        apply_throughput=True,
        skyflat=skyflat_red,
        do_extinction=False,
        correct_ccd_defects=True,
        correct_high_cosmics=False,
        clip_high=100,
        step_ccd=50,
        sky_method="none",
        is_sky=True,
        win_sky=151,
        plot=DO_PLOTTING
    )

    sky3 = sky_r3.plot_combined_spectrum(
        list_spectra = [
            870, 871, 872, 873, 875, 876, 877, 878, 879, 880, 881, 882, 883,
            884, 885, 886, 887, 888, 889, 900
        ], median=True
    )

    sky_list = [
        sky3, sky3
    ]

    hikids_red = KOALA_reduce(
        RSS_LIST,
        obj_name=OBJECT,
        description=DESCRIPTION,
        fits_file=str(tmpdir.join("output.fits")),
        apply_throughput=True,
        skyflat=skyflat_red,
        plot_skyflat=False,
        correct_ccd_defects=True,
        correct_high_cosmics=False,
        clip_high=100,
        step_ccd=50,
        sky_method="1D",
        sky_list=sky_list,
        scale_sky_1D=1.,
        auto_scale_sky=True,
        brightest_line="Ha",
        brightest_line_wavelength = 6641.,
        id_el=False,
        high_fibres=10,
        cut=1.5,
        plot_id_el=True,
        broad=1.8,
        id_list=[
            6300.30, 6312.1, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15,
            6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66, 8750.47,
            8862.79, 9014.91, 9069.0
        ],
        telluric_correction=telluric_correction,
        do_extinction=True,
        correct_negative_sky=False,
        pixel_size_arcsec=PIXEL_SIZE,
        kernel_size_arcsec=KERNEL_SIZE,
        ADR=False,
        flux_calibration=flux_calibration,
        valid_wave_min = 6085,
        valid_wave_max = 9305,
        plot=DO_PLOTTING,
        warnings=False
    )


class TestCliFramework:
    @pytest.mark.xfail
    def test_version(self):
        reduce_koala_data_main(["--version"])

    @pytest.mark.xfail
    def test_citation(self):
        reduce_koala_data_main(["--citation"])

    @pytest.mark.xfail
    def test_default(self):
        reduce_koala_data_main([])
