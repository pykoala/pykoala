# -*- coding: utf-8 -*-
"""
File containing functions related to calculating offsets
"""
import numpy as np


def KOALA_offsets(s, pa):
    """

    Parameters
    ----------
    s
    pa

    Returns
    -------

    """
    print("\n> Offsets towards North and East between pointings," "according to KOALA manual, for pa = {} degrees".format(pa))
    pa *= np.pi/180
    print("  a -> b : {} {}".format(s * np.sin(pa), -s * np.cos(pa)))
    print("  a -> c : {} {}".format(-s * np.sin(60 - pa), -s * np.cos(60 - pa)))
    print("  b -> d : {} {}".format(-np.sqrt(3) * s * np.cos(pa), -np.sqrt(3) * s * np.sin(pa)))


def ds9_offsets(x1, y1, x2, y2, pixel_size_arc=0.6):
    """
    Print information about offsets in pixels between (x1, y1) and (x2, y2). This assumes that (x1, y1) and (x2, y2) are close on the sky and small amngle approximations are valid!

    Args:
        x1 (float): x position 1 (in pixels)
        y1 (float): y position 1 (in pixels
        x2 (float): x position 2 (in pixels)
        y2 (float): y position 2 (in pixels)
        pixel_size_arc (float, default=0.6): The pixel size in arcseconds

    Returns:
        None
    """

    delta_x = x2 - x1
    delta_y = y2 - y1

    print("\n> Offsets in pixels : {} {}".format(delta_x, delta_y))
    print("  Offsets in arcsec : {} {}".format(pixel_size_arc * delta_x, pixel_size_arc * delta_y))
    offset_RA = np.abs(pixel_size_arc * delta_x)
    if delta_x < 0:
        direction_RA = "W"
    else:
        direction_RA = "E"
    offset_DEC = np.abs(pixel_size_arc * delta_y)
    if delta_y < 0:
        direction_DEC = "N"
    else:
        direction_DEC = "S"
    print("  Assuming N up and E left, the telescope did an offset of ----> {:5.2f} {:1} {:5.2f} {:1}".format(
        offset_RA, direction_RA, offset_DEC, direction_DEC
    ))


def offset_positions(
    ra1h,
    ra1m,
    ra1s,
    dec1d,
    dec1m,
    dec1s,
    ra2h,
    ra2m,
    ra2s,
    dec2d,
    dec2m,
    dec2s,
    decimals=2,
):
    """
    Work out offsets between two sky positions and print them to the screen. This could probably be replaced with some astropy functions.
    TODO: Include arguments

    Parameters
    ----------
    ra1h
    ra1m
    ra1s
    dec1d
    dec1m
    dec1s
    ra2h
    ra2m
    ra2s
    dec2d
    dec2m
    dec2s
    decimals

    Returns
    -------
        None
    """

    ra1 = ra1h + ra1m / 60.0 + ra1s / 3600.0
    ra2 = ra2h + ra2m / 60.0 + ra2s / 3600.0

    if dec1d < 0:
        dec1 = dec1d - dec1m / 60.0 - dec1s / 3600.0
    else:
        dec1 = dec1d + dec1m / 60.0 + dec1s / 3600.0
    if dec2d < 0:
        dec2 = dec2d - dec2m / 60.0 - dec2s / 3600.0
    else:
        dec2 = dec2d + dec2m / 60.0 + dec2s / 3600.0

    avdec = (dec1 + dec2)/2

    deltadec = round(3600.0 * (dec2 - dec1), decimals)
    deltara = round(15 * 3600.0 * (ra2 - ra1) * (np.cos(np.radians(avdec))), decimals)

    tdeltadec = np.fabs(deltadec)
    tdeltara = np.fabs(deltara)

    if deltadec < 0:
        t_sign_deltadec = "South"
        t_sign_deltadec_invert = "North"

    else:
        t_sign_deltadec = "North"
        t_sign_deltadec_invert = "South"

    if deltara < 0:
        t_sign_deltara = "West"
        t_sign_deltara_invert = "East"

    else:
        t_sign_deltara = "East"
        t_sign_deltara_invert = "West"

    print("\n> POS1: RA = {:3}h {:2}min {:2.4f}sec, DEC = {:3}d {:2}m {:2.4f}s".format(
        ra1h, ra1m, ra1s, dec1d, dec1m, dec1s
    ))
    print("  POS2: RA = {:3}h {:2}min {:2.4f}sec, DEC = {:3}d {:2}m {:2.4f}s".format(
        ra2h, ra2m, ra2s, dec2d, dec2m, dec2s
    ))

    print("\n> Offset 1 -> 2 : {} {}       {} {}".format(tdeltara, t_sign_deltara, tdeltadec, t_sign_deltadec))
    print("  Offset 2 -> 1 : {} {}       {} {}".format(tdeltara, t_sign_deltara_invert, tdeltadec, t_sign_deltadec_invert))


