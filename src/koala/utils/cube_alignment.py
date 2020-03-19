# -*- coding: utf-8 -*-
"""
File contains functions relating to aligning data cubes and differences between cubes.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from .plots import plot_offset_between_cubes


def offset_between_cubes(cube1, cube2, plot=False):
    """

    Parameters
    ----------
    cube1
    cube2
    plot

    Returns
    -------

    """
    x = (
        cube2.x_peak
        - cube2.n_cols / 2.0
        + cube2.RA_centre_deg * 3600.0 / cube2.pixel_size_arcsec
    ) - (
        cube1.x_peak
        - cube1.n_cols / 2.0
        + cube1.RA_centre_deg * 3600.0 / cube1.pixel_size_arcsec
    )
    y = (
        cube2.y_peak
        - cube2.n_rows / 2.0
        + cube2.DEC_centre_deg * 3600.0 / cube2.pixel_size_arcsec
    ) - (
        cube1.y_peak
        - cube1.n_rows / 2.0
        + cube1.DEC_centre_deg * 3600.0 / cube1.pixel_size_arcsec
    )
    delta_RA_pix = np.nanmedian(x)
    delta_DEC_pix = np.nanmedian(y)
    #    weight = np.nansum(cube1.data+cube2.data, axis=(1, 2))
    #    total_weight = np.nansum(weight)
    #    print "--- lambda=", np.nansum(cube1.RSS.wavelength*weight) / total_weight
    #    delta_RA_pix = np.nansum(x*weight) / total_weight
    #    delta_DEC_pix = np.nansum(y*weight) / total_weight
    delta_RA_arcsec = delta_RA_pix * cube1.pixel_size_arcsec
    delta_DEC_arcsec = delta_DEC_pix * cube1.pixel_size_arcsec
    print("(delta_RA, delta_DEC) = ({:.3f}, {:.3f}) arcsec".format(
        delta_RA_arcsec, delta_DEC_arcsec
    ))
    #    delta_RA_headers = (cube2.RSS.RA_centre_deg - cube1.RSS.RA_centre_deg) * 3600
    #    delta_DEC_headers = (cube2.RSS.DEC_centre_deg - cube1.RSS.DEC_centre_deg) * 3600
    #    print '                        ({:.3f}, {:.3f}) arcsec according to headers!!!???' \
    #        .format(delta_RA_headers, delta_DEC_headers)
    #    print 'difference:             ({:.3f}, {:.3f}) arcsec' \
    #        .format(delta_RA-delta_RA_headers, delta_DEC-delta_DEC_headers)

    if plot:
        wl = None  # wl was unresolved. Added to fix breaking errors. plot_... takes wl as input, but not used in func
        x -= delta_RA_pix
        y -= delta_DEC_pix
        fig = plot_offset_between_cubes(cube1, delta_RA_pix, delta_DEC_pix, wl, medfilt_window=151)

    return delta_RA_arcsec, delta_DEC_arcsec

def compare_cubes(cube1, cube2, line=0):
    if line == 0:
        map1 = cube1.integrated_map
        map2 = cube2.integrated_map
    else:
        l = np.searchsorted(cube1.RSS.wavelength, line)
        map1 = cube1.data[l]
        map2 = cube2.data[l]

    scale = np.nanmedian(map1 + map2) * 3
    scatter = np.nanmedian(np.nonzero(map1 - map2))

    plt.figure(figsize=(12, 8))
    plt.imshow(
        map1 - map2, vmin=-scale, vmax=scale, cmap=plt.cm.get_cmap("RdBu")
    )  # vmin = -scale
    plt.colorbar()
    plt.contour(map1, colors="w", linewidths=2, norm=colors.LogNorm())
    plt.contour(map2, colors="k", linewidths=1, norm=colors.LogNorm())
    if line != 0:
        plt.title("{:.2f} AA".format(line))
    else:
        plt.title("Integrated Map")
    # plt.show()
    # plt.close()


# def align_3_cubes(
#         cube1,
#         cube2,
#         cube3,
#         rss1,
#         rss2,
#         rss3,
#         pixel_size_arcsec=0.3,
#         kernel_size_arcsec=1.5,
#         offsets=[1000],
#         plot=False,
#         ADR=False,
#         warnings=False,
# ):
#     """
#     (OLD) Routine to align 3 cubes.
#
#     THIS SHOULD NOT BE USED! Use "align_n_cubes" instead !!
#
#
#     Parameters
#     ----------
#     Cubes:
#         Cubes
#     pointings_RSS :
#         list with RSS files
#     pixel_size_arcsec:
#         float, default = 0.3
#     kernel_size_arcsec:
#         float, default = 1.5
#
#     """
#     print("\n> Starting alignment procedure...")
#
#     # pointings_RSS=[rss1, rss2, rss3, rss4]
#     # RA_min, RA_max, DEC_min, DEC_max = coord_range(pointings_RSS)
#
#     if offsets[0] == 1000:
#         #        print "  Using peak in integrated image to align cubes:"
#         #        x12 = cube1.offset_from_center_x_arcsec_integrated - cube2.offset_from_center_x_arcsec_integrated
#         #        y12 = cube1.offset_from_center_y_arcsec_integrated - cube2.offset_from_center_y_arcsec_integrated
#         #        x23 = cube2.offset_from_center_x_arcsec_integrated - cube3.offset_from_center_x_arcsec_integrated
#         #        y23 = cube2.offset_from_center_y_arcsec_integrated - cube3.offset_from_center_y_arcsec_integrated
#         print("  Using peak of the emission tracing all wavelengths to align cubes:")
#         x12 = (
#                 cube2.offset_from_center_x_arcsec_tracing
#                 - cube1.offset_from_center_x_arcsec_tracing
#         )
#         y12 = (
#                 cube2.offset_from_center_y_arcsec_tracing
#                 - cube1.offset_from_center_y_arcsec_tracing
#         )
#         x23 = (
#                 cube3.offset_from_center_x_arcsec_tracing
#                 - cube2.offset_from_center_x_arcsec_tracing
#         )
#         y23 = (
#                 cube3.offset_from_center_y_arcsec_tracing
#                 - cube2.offset_from_center_y_arcsec_tracing
#         )
#         x31 = (
#                 cube1.offset_from_center_x_arcsec_tracing
#                 - cube3.offset_from_center_x_arcsec_tracing
#         )
#         y31 = (
#                 cube1.offset_from_center_y_arcsec_tracing
#                 - cube3.offset_from_center_y_arcsec_tracing
#         )
#
#     else:
#         print("  Using offsets given by the user:")
#         x12 = offsets[0]
#         y12 = offsets[1]
#         x23 = offsets[2]
#         y23 = offsets[3]
#         x31 = -(offsets[0] + offsets[2])
#         y31 = -(offsets[1] + offsets[3])
#
#     rss1.ALIGNED_RA_centre_deg = cube1.RA_centre_deg
#     rss1.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg
#     rss2.ALIGNED_RA_centre_deg = cube1.RA_centre_deg - x12 / 3600.0
#     rss2.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg - y12 / 3600.0
#     rss3.ALIGNED_RA_centre_deg = cube1.RA_centre_deg + x31 / 3600.0
#     rss3.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg + y31 / 3600.0
#
#     RA_centre_deg = rss1.ALIGNED_RA_centre_deg
#     DEC_centre_deg = rss1.ALIGNED_DEC_centre_deg
#
#     print("\n  Offsets (in arcsec):")
#     print("  Offsets in x : {} {}       Total offset in x = {}".format(x12, x23, x12 + x23 + x31))
#     print("  Offsets in y : {} {}       Total offset in y = {}".format(y12, y23, y12 + y23 + y31))
#
#     print("\n>        New_RA_centre_deg       New_DEC_centre_deg       Diff respect Cube 1 (arcsec)")
#     print("  Cube 1 : {}       {}            0 0".format(rss1.ALIGNED_RA_centre_deg, rss1.ALIGNED_DEC_centre_deg))
#     print("  Cube 2 : {}       {}            {} {}".format(
#         rss2.ALIGNED_RA_centre_deg, rss2.ALIGNED_DEC_centre_deg,
#         (rss2.ALIGNED_RA_centre_deg - rss1.ALIGNED_RA_centre_deg) * 3600.0,
#         (rss2.ALIGNED_DEC_centre_deg - rss1.ALIGNED_DEC_centre_deg) * 3600.0))
#     print("  Cube 3 : {}       {}            {} {}".format(
#         rss3.ALIGNED_RA_centre_deg, rss3.ALIGNED_DEC_centre_deg,
#         (rss3.ALIGNED_RA_centre_deg - rss1.ALIGNED_RA_centre_deg) * 3600.0,
#         (rss3.ALIGNED_DEC_centre_deg - rss1.ALIGNED_DEC_centre_deg) * 3600.0))
#
#     offsets_files = [
#         [x12, y12],
#         [x23, y23],
#     ]  # For keeping in the files with self.offsets_files
#     #    RA_size_arcsec = rss1.RA_segment + np.abs(x12)+np.abs(x23) + 2*kernel_size_arcsec
#     #    DEC_size_arcsec =rss1.DEC_segment +np.abs(y12)+np.abs(y23)  +2*kernel_size_arcsec
#
#     RA_size_arcsec = rss1.RA_segment + x12 + x23 + 2 * kernel_size_arcsec
#     DEC_size_arcsec = rss1.DEC_segment + y12 + y23 + 2 * kernel_size_arcsec
#
#     #    print "  RA_centre_deg , DEC_centre_deg   = ", RA_centre_deg, DEC_centre_deg
#     print("  RA_size x DEC_size  = {:.2f} arcsec x {:.2f} arcsec".format(
#         RA_size_arcsec, DEC_size_arcsec
#     ))
#
#     #    probando=raw_input("Continue?")
#
#     cube1_aligned = Interpolated_cube(
#         rss1,
#         pixel_size_arcsec,
#         kernel_size_arcsec,
#         centre_deg=[RA_centre_deg, DEC_centre_deg],
#         size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
#         aligned_coor=True,
#         flux_calibration=cube1.flux_calibration,
#         offsets_files=offsets_files,
#         offsets_files_position=1,
#         plot=plot,
#         ADR=ADR,
#         warnings=warnings,
#     )
#     cube2_aligned = Interpolated_cube(
#         rss2,
#         pixel_size_arcsec,
#         kernel_size_arcsec,
#         centre_deg=[RA_centre_deg, DEC_centre_deg],
#         size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
#         aligned_coor=True,
#         flux_calibration=cube2.flux_calibration,
#         offsets_files=offsets_files,
#         offsets_files_position=2,
#         plot=plot,
#         ADR=ADR,
#         warnings=warnings,
#     )
#     cube3_aligned = Interpolated_cube(
#         rss3,
#         pixel_size_arcsec,
#         kernel_size_arcsec,
#         centre_deg=[RA_centre_deg, DEC_centre_deg],
#         size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
#         aligned_coor=True,
#         flux_calibration=cube3.flux_calibration,
#         offsets_files=offsets_files,
#         offsets_files_position=3,
#         plot=plot,
#         ADR=ADR,
#         warnings=warnings,
#     )
#
#     print("\n> Checking offsets of ALIGNED cubes (in arcsec):")
#
#     x12 = (
#             cube1_aligned.offset_from_center_x_arcsec_tracing
#             - cube2_aligned.offset_from_center_x_arcsec_tracing
#     )
#     y12 = (
#             cube1_aligned.offset_from_center_y_arcsec_tracing
#             - cube2_aligned.offset_from_center_y_arcsec_tracing
#     )
#     x23 = (
#             cube2_aligned.offset_from_center_x_arcsec_tracing
#             - cube3_aligned.offset_from_center_x_arcsec_tracing
#     )
#     y23 = (
#             cube2_aligned.offset_from_center_y_arcsec_tracing
#             - cube3_aligned.offset_from_center_y_arcsec_tracing
#     )
#     x31 = (
#             cube3_aligned.offset_from_center_x_arcsec_tracing
#             - cube1_aligned.offset_from_center_x_arcsec_tracing
#     )
#     y31 = (
#             cube3_aligned.offset_from_center_y_arcsec_tracing
#             - cube1_aligned.offset_from_center_y_arcsec_tracing
#     )
#
#     print("  Offsets in x : {:.3f}   {:.3f}   {:.3f}       Total offset in x = {:.3f}".format(
#         x12, x23, x31, x12 + x23 + x31
#     ))
#     print("  Offsets in y : {:.3f}   {:.3f}   {:.3f}       Total offset in y = {:.3f}".format(
#         y12, y23, y31, y12 + y23 + y31
#     ))
#
#     print("\n> Updated values for Alignment DONE")
#     return cube1_aligned, cube2_aligned, cube3_aligned
#
#
# def align_4_cubes(
#         cube1,
#         cube2,
#         cube3,
#         cube4,
#         rss1,
#         rss2,
#         rss3,
#         rss4,
#         pixel_size_arcsec=0.3,
#         kernel_size_arcsec=1.5,
#         offsets=[1000],
#         plot=False,
#         ADR=False,
#         warnings=False,
# ):
#     """
#     (OLD) Routine to align 4 cubes
#
#     THIS SHOULD NOT BE USED! Use "align_n_cubes" instead !!
#
#     Parameters
#     ----------
#     Cubes:
#         Cubes
#     pointings_RSS :
#         list with RSS files
#     pixel_size_arcsec:
#         float, default = 0.3
#     kernel_size_arcsec:
#         float, default = 1.5
#
#     """
#     print("\n> Starting alignment procedure...")
#
#     # pointings_RSS=[rss1, rss2, rss3, rss4]
#     # RA_min, RA_max, DEC_min, DEC_max = coord_range(pointings_RSS)
#
#     if offsets[0] == 1000:
#         #        print "  Using peak in integrated image to align cubes:"
#         #        x12 = cube1.offset_from_center_x_arcsec_integrated - cube2.offset_from_center_x_arcsec_integrated
#         #        y12 = cube1.offset_from_center_y_arcsec_integrated - cube2.offset_from_center_y_arcsec_integrated
#         #        x23 = cube2.offset_from_center_x_arcsec_integrated - cube3.offset_from_center_x_arcsec_integrated
#         #        y23 = cube2.offset_from_center_y_arcsec_integrated - cube3.offset_from_center_y_arcsec_integrated
#         #        x34 = cube3.offset_from_center_x_arcsec_integrated - cube4.offset_from_center_x_arcsec_integrated
#         #        y34 = cube3.offset_from_center_y_arcsec_integrated - cube4.offset_from_center_y_arcsec_integrated
#         #        x41 = cube4.offset_from_center_x_arcsec_integrated - cube1.offset_from_center_x_arcsec_integrated
#         #        y41 = cube4.offset_from_center_y_arcsec_integrated - cube1.offset_from_center_y_arcsec_integrated
#         print("  Using peak of the emission tracing all wavelengths to align cubes:")
#         x12 = (
#                 cube2.offset_from_center_x_arcsec_tracing
#                 - cube1.offset_from_center_x_arcsec_tracing
#         )
#         y12 = (
#                 cube2.offset_from_center_y_arcsec_tracing
#                 - cube1.offset_from_center_y_arcsec_tracing
#         )
#         x23 = (
#                 cube3.offset_from_center_x_arcsec_tracing
#                 - cube2.offset_from_center_x_arcsec_tracing
#         )
#         y23 = (
#                 cube3.offset_from_center_y_arcsec_tracing
#                 - cube2.offset_from_center_y_arcsec_tracing
#         )
#         x34 = (
#                 cube4.offset_from_center_x_arcsec_tracing
#                 - cube3.offset_from_center_x_arcsec_tracing
#         )
#         y34 = (
#                 cube4.offset_from_center_y_arcsec_tracing
#                 - cube3.offset_from_center_y_arcsec_tracing
#         )
#         x41 = (
#                 cube1.offset_from_center_x_arcsec_tracing
#                 - cube4.offset_from_center_x_arcsec_tracing
#         )
#         y41 = (
#                 cube1.offset_from_center_y_arcsec_tracing
#                 - cube4.offset_from_center_y_arcsec_tracing
#         )
#
#     else:
#         print("  Using offsets given by the user:")
#         x12 = offsets[0]
#         y12 = offsets[1]
#         x23 = offsets[2]
#         y23 = offsets[3]
#         x34 = offsets[4]
#         y34 = offsets[5]
#         x41 = -(offsets[0] + offsets[2] + offsets[4])
#         y41 = -(offsets[1] + offsets[3] + offsets[5])
#
#     rss1.ALIGNED_RA_centre_deg = cube1.RA_centre_deg
#     rss1.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg
#     rss2.ALIGNED_RA_centre_deg = cube1.RA_centre_deg - x12 / 3600.0
#     rss2.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg - y12 / 3600.0
#     rss3.ALIGNED_RA_centre_deg = cube1.RA_centre_deg - (x12 + x23) / 3600.0
#     rss3.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg - (y12 + y23) / 3600.0
#     rss4.ALIGNED_RA_centre_deg = cube1.RA_centre_deg + x41 / 3600.0
#     rss4.ALIGNED_DEC_centre_deg = cube1.DEC_centre_deg + y41 / 3600.0
#
#     RA_centre_deg = rss1.ALIGNED_RA_centre_deg
#     DEC_centre_deg = rss1.ALIGNED_DEC_centre_deg
#
#     print("\n  Offsets (in arcsec):")
#     print("  Offsets in x : ", x12, x23, x34, "      Total offset in x = ", x12 + x23 + x34 + x41)
#     print("  Offsets in y : ", y12, y23, y34, "      Total offset in y = ", y12 + y23 + y34 + y41)
#
#     print("\n>        New_RA_centre_deg       New_DEC_centre_deg       Diff respect Cube 1 (arcsec)")
#     print("  Cube 1 : ", rss1.ALIGNED_RA_centre_deg, "     ", rss1.ALIGNED_DEC_centre_deg, "      0,0")
#     print("  Cube 2 : ", rss2.ALIGNED_RA_centre_deg, "     ", rss2.ALIGNED_DEC_centre_deg, "    ", (
#             rss2.ALIGNED_RA_centre_deg - rss1.ALIGNED_RA_centre_deg
#     ) * 3600.0, (
#                   rss2.ALIGNED_DEC_centre_deg - rss1.ALIGNED_DEC_centre_deg
#           ) * 3600.0)
#     print("  Cube 3 : ", rss3.ALIGNED_RA_centre_deg, "     ", rss3.ALIGNED_DEC_centre_deg, "    ", (
#             rss3.ALIGNED_RA_centre_deg - rss1.ALIGNED_RA_centre_deg
#     ) * 3600.0, (
#                   rss3.ALIGNED_DEC_centre_deg - rss1.ALIGNED_DEC_centre_deg
#           ) * 3600.0)
#     print("  Cube 4 : ", rss4.ALIGNED_RA_centre_deg, "     ", rss4.ALIGNED_DEC_centre_deg, "    ", (
#             rss4.ALIGNED_RA_centre_deg - rss1.ALIGNED_RA_centre_deg
#     ) * 3600.0, (
#                   rss4.ALIGNED_DEC_centre_deg - rss1.ALIGNED_DEC_centre_deg
#           ) * 3600.0)
#
#     offsets_files = [
#         [x12, y12],
#         [x23, y23],
#         [x34, y34],
#     ]  # For keeping in the files with self.offsets_files
#     #    RA_size_arcsec = 1.1*(RA_max - RA_min)*3600.
#     #    DEC_size_arcsec = 1.1*(DEC_max - DEC_min)*3600.
#     RA_size_arcsec = rss1.RA_segment + x12 + x23 + x34 + 2 * kernel_size_arcsec
#     DEC_size_arcsec = rss1.DEC_segment + y12 + y23 + y34 + 2 * kernel_size_arcsec
#
#     #    print "  RA_centre_deg , DEC_centre_deg   = ", RA_centre_deg, DEC_centre_deg
#     print("  RA_size x DEC_size  = {:.2f} arcsec x {:.2f} arcsec".format(
#         RA_size_arcsec, DEC_size_arcsec
#     ))
#
#     #    probando=raw_input("Continue?")
#
#     cube1_aligned = Interpolated_cube(
#         rss1,
#         pixel_size_arcsec,
#         kernel_size_arcsec,
#         centre_deg=[RA_centre_deg, DEC_centre_deg],
#         size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
#         aligned_coor=True,
#         flux_calibration=cube1.flux_calibration,
#         offsets_files=offsets_files,
#         offsets_files_position=1,
#         plot=plot,
#         ADR=ADR,
#         warnings=warnings,
#     )
#     cube2_aligned = Interpolated_cube(
#         rss2,
#         pixel_size_arcsec,
#         kernel_size_arcsec,
#         centre_deg=[RA_centre_deg, DEC_centre_deg],
#         size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
#         aligned_coor=True,
#         flux_calibration=cube2.flux_calibration,
#         offsets_files=offsets_files,
#         offsets_files_position=2,
#         plot=plot,
#         ADR=ADR,
#         warnings=warnings,
#     )
#     cube3_aligned = Interpolated_cube(
#         rss3,
#         pixel_size_arcsec,
#         kernel_size_arcsec,
#         centre_deg=[RA_centre_deg, DEC_centre_deg],
#         size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
#         aligned_coor=True,
#         flux_calibration=cube3.flux_calibration,
#         offsets_files=offsets_files,
#         offsets_files_position=3,
#         plot=plot,
#         ADR=ADR,
#         warnings=warnings,
#     )
#     cube4_aligned = Interpolated_cube(
#         rss4,
#         pixel_size_arcsec,
#         kernel_size_arcsec,
#         centre_deg=[RA_centre_deg, DEC_centre_deg],
#         size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
#         aligned_coor=True,
#         flux_calibration=cube3.flux_calibration,
#         offsets_files=offsets_files,
#         offsets_files_position=4,
#         plot=plot,
#         ADR=ADR,
#         warnings=warnings,
#     )
#
#     print("\n> Checking offsets of ALIGNED cubes (in arcsec):")
#
#     x12 = (
#             cube1_aligned.offset_from_center_x_arcsec_tracing
#             - cube2_aligned.offset_from_center_x_arcsec_tracing
#     )
#     y12 = (
#             cube1_aligned.offset_from_center_y_arcsec_tracing
#             - cube2_aligned.offset_from_center_y_arcsec_tracing
#     )
#     x23 = (
#             cube2_aligned.offset_from_center_x_arcsec_tracing
#             - cube3_aligned.offset_from_center_x_arcsec_tracing
#     )
#     y23 = (
#             cube2_aligned.offset_from_center_y_arcsec_tracing
#             - cube3_aligned.offset_from_center_y_arcsec_tracing
#     )
#     x34 = (
#             cube3_aligned.offset_from_center_x_arcsec_tracing
#             - cube4_aligned.offset_from_center_x_arcsec_tracing
#     )
#     y34 = (
#             cube3_aligned.offset_from_center_y_arcsec_tracing
#             - cube4_aligned.offset_from_center_y_arcsec_tracing
#     )
#     x41 = (
#             cube4_aligned.offset_from_center_x_arcsec_tracing
#             - cube1_aligned.offset_from_center_x_arcsec_tracing
#     )
#     y41 = (
#             cube4_aligned.offset_from_center_y_arcsec_tracing
#             - cube1_aligned.offset_from_center_y_arcsec_tracing
#     )
#
#     print("  Offsets in x : {:.3f}   {:.3f}   {:.3f}   {:.3f}    Total offset in x = {:.3f}".format(
#         x12, x23, x34, x41, x12 + x23 + x34 + x41
#     ))
#     print("  Offsets in y : {:.3f}   {:.3f}   {:.3f}   {:.3f}    Total offset in y = {:.3f}".format(
#         y12, y23, y34, y41, y12 + y23 + y34 + y41
#     ))
#
#     print("\n> Updated values for Alignment DONE")
#     return cube1_aligned, cube2_aligned, cube3_aligned, cube4_aligned


def align_n_cubes(
        rss_list,
        cube_list=[0],
        flux_calibration_list=[[0]],
        pixel_size_arcsec=0.3,
        kernel_size_arcsec=1.5,
        offsets=[1000],
        plot=False,
        ADR=False,
        warnings=False,
):  # TASK_align_n_cubes
    """
    Routine to align n cubes

    Parameters
    ----------
    Cubes:
        Cubes
    pointings_RSS :
        list with RSS files
    pixel_size_arcsec:
        float, default = 0.3
    kernel_size_arcsec:
        float, default = 1.5

    """
    from koala import Interpolated_cube  # TODO: currently importing like this for workaround of circular imports
    # This file requires a class, Interpolated_Cube from koala, however from koala import Inter... will result in
    # circular import as __init__ imports specific functions from this file.

    print("\n> Starting alignment procedure...")

    n_rss = len(rss_list)

    xx = [0]  # This will have 0, x12, x23, x34, ... xn1
    yy = [0]  # This will have 0, y12, y23, y34, ... yn1

    if np.nanmedian(flux_calibration_list[0]) == 0:
        flux_calibration_list[0] = [0]
        for i in range(1, n_rss):
            flux_calibration_list.append([0])

    if offsets[0] == 1000:
        print("\n  Using peak of the emission tracing all wavelengths to align cubes:")
        n_cubes = len(cube_list)
        if n_cubes != n_rss:
            print("\n\n\n ERROR: number of cubes and number of rss files don't match!")
            print("\n\n THIS IS GOING TO FAIL ! \n\n\n")

        for i in range(n_rss - 1):
            xx.append(
                cube_list[i + 1].offset_from_center_x_arcsec_tracing
                - cube_list[i].offset_from_center_x_arcsec_tracing
            )
            yy.append(
                cube_list[i + 1].offset_from_center_y_arcsec_tracing
                - cube_list[i].offset_from_center_y_arcsec_tracing
            )
        xx.append(
            cube_list[0].offset_from_center_x_arcsec_tracing
            - cube_list[-1].offset_from_center_x_arcsec_tracing
        )
        yy.append(
            cube_list[0].offset_from_center_y_arcsec_tracing
            - cube_list[-1].offset_from_center_y_arcsec_tracing
        )

    else:
        print("\n  Using offsets given by the user:")
        for i in range(0, 2 * n_rss - 2, 2):
            xx.append(offsets[i])
            yy.append(offsets[i + 1])
        xx.append(-np.nansum(xx))  #
        yy.append(-np.nansum(yy))

    # Estimate median value of the centre of files
    list_RA_centre_deg = []
    list_DEC_centre_deg = []

    for i in range(n_rss):
        list_RA_centre_deg.append(rss_list[i].RA_centre_deg)
        list_DEC_centre_deg.append(rss_list[i].DEC_centre_deg)

    median_RA_centre_deg = np.nanmedian(list_RA_centre_deg)
    median_DEC_centre_deg = np.nanmedian(list_DEC_centre_deg)

    print("\n\n\n\n\n\n")
    print(list_RA_centre_deg, median_RA_centre_deg)
    print(list_DEC_centre_deg, median_DEC_centre_deg)
    print("\n\n\n\n\n\n")

    for i in range(n_rss):
        # print i, np.nansum(xx[1:i+1]) ,  np.nansum(yy[1:i+1])
        rss_list[i].ALIGNED_RA_centre_deg = (
                median_RA_centre_deg + np.nansum(xx[1: i + 1]) / 3600.0
        )  # CHANGE SIGN 26 Apr 2019    # ERA cube_list[0]
        rss_list[i].ALIGNED_DEC_centre_deg = (
                median_DEC_centre_deg - np.nansum(yy[1: i + 1]) / 3600.0
        )  # rss_list[0].DEC_centre_deg

        print(rss_list[i].RA_centre_deg, rss_list[i].ALIGNED_RA_centre_deg)
        print(rss_list[i].ALIGNED_DEC_centre_deg)

    RA_centre_deg = rss_list[0].ALIGNED_RA_centre_deg
    DEC_centre_deg = rss_list[0].ALIGNED_DEC_centre_deg

    print("  Offsets (in arcsec):        x             y                          ( EAST- / WEST+   NORTH- / SOUTH+) ")
    for i in range(1, len(xx) - 1):
        print("         {:2.0f} -> {:2.0f}           {:+5.3f}         {:+5.3f}".format(
            i, i + 1, xx[i], yy[i]
        ))
    print("         {:2.0f} ->  1           {:+5.3f}         {:+5.3f}".format(
        len(xx) - 1, xx[-1], yy[-1]
    ))
    print("           TOTAL:            {:5.3f}          {:5.3f}".format(
        np.nansum(xx), np.nansum(yy)
    ))

    print("\n         New_RA_centre_deg       New_DEC_centre_deg      Diff with respect Cube 1 [arcsec]")

    for i in range(0, n_rss):
        print("  Cube {:2.0f}:     {:5.8f}          {:5.8f}           {:5.3f}   ,  {:5.3f}   ".format(
            i + 1,
            rss_list[i].ALIGNED_RA_centre_deg,
            rss_list[i].ALIGNED_DEC_centre_deg,
            (rss_list[i].ALIGNED_RA_centre_deg - rss_list[0].ALIGNED_RA_centre_deg)
            * 3600.0,
            (rss_list[i].ALIGNED_DEC_centre_deg - rss_list[0].ALIGNED_DEC_centre_deg)
            * 3600.0,
        ))

    offsets_files = []
    for i in range(1, n_rss):  # For keeping in the files with self.offsets_files
        vector = [xx[i], yy[i]]
        offsets_files.append(vector)

    RA_size_arcsec = (
            rss_list[0].RA_segment + np.nansum(np.abs(xx[0:-1])) + 2 * kernel_size_arcsec
    )
    DEC_size_arcsec = (
            rss_list[0].DEC_segment + np.nansum(np.abs(yy[0:-1])) + 2 * kernel_size_arcsec
    )
    print("\n  RA_size x DEC_size  = {:.2f} arcsec x {:.2f} arcsec".format(
        RA_size_arcsec, DEC_size_arcsec
    ))

    cube_aligned_list = []
    for i in range(1, n_rss + 1):
        escribe = "cube" + np.str(i) + "_aligned"
        cube_aligned_list.append(escribe)

    for i in range(n_rss):
        print("\n> Creating aligned cube {} of a total of {} ...".format(i + 1, n_rss))
        cube_aligned_list[i] = Interpolated_cube(
            rss_list[i],
            pixel_size_arcsec,
            kernel_size_arcsec,
            centre_deg=[RA_centre_deg, DEC_centre_deg],
            size_arcsec=[RA_size_arcsec, DEC_size_arcsec],
            aligned_coor=True,
            flux_calibration=flux_calibration_list[i],
            offsets_files=offsets_files,
            offsets_files_position=i + 1,
            plot=plot,
            ADR=ADR,
            warnings=warnings,
        )

    print("\n> Checking offsets of ALIGNED cubes (in arcsec, everything should be close to 0):")
    print("  Offsets (in arcsec):        x             y                          ( EAST- / WEST+   NORTH- / SOUTH+) ")

    xxx = []
    yyy = []

    for i in range(1, n_rss):
        xxx.append(
            cube_aligned_list[i - 1].offset_from_center_x_arcsec_tracing
            - cube_aligned_list[i].offset_from_center_x_arcsec_tracing
        )
        yyy.append(
            cube_aligned_list[i - 1].offset_from_center_y_arcsec_tracing
            - cube_aligned_list[i].offset_from_center_y_arcsec_tracing
        )
    xxx.append(
        cube_aligned_list[-1].offset_from_center_x_arcsec_tracing
        - cube_aligned_list[0].offset_from_center_x_arcsec_tracing
    )
    yyy.append(
        cube_aligned_list[-1].offset_from_center_y_arcsec_tracing
        - cube_aligned_list[0].offset_from_center_y_arcsec_tracing
    )

    for i in range(1, len(xx) - 1):
        print("         {:2.0f} -> {:2.0f}           {:+5.3f}         {:+5.3f}".format(
            i, i + 1, xxx[i - 1], yyy[i - 1]
        ))
    print("         {:2.0f} ->  1           {:+5.3f}         {:+5.3f}".format(
        len(xxx), xxx[-1], yyy[-1]
    ))
    print("           TOTAL:            {:5.3f}          {:5.3f}".format(
        np.nansum(xxx), np.nansum(yyy)
    ))

    print("\n> Updated values for Alignment cubes DONE")
    return cube_aligned_list
