# -*- coding: utf-8 -*-
"""
File containing functions related to calculating Moffat distributions
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def cumulaive_Moffat(r2, L_star, alpha2, beta):
    """

    Parameters
    ----------
    r2
    L_star
    alpha2
    beta

    Returns
    -------

    """
    return L_star * (1 - np.power(1 + (r2/alpha2), -beta))


def fit_Moffat(
    r2_growth_curve, F_growth_curve, F_guess, r2_half_light, r_max, plot=False
):
    """
    Fits a Moffat profile to a flux growth curve
    as a function of radius squared,
    cutting at to r_max (in units of the half-light radius),
    provided an initial guess of the total flux and half-light radius squared.

    Parameters
    ----------
    r2_growth_curve
    F_growth_curve
    F_guess
    r2_half_light
    r_max
    plot

    Returns
    -------

    """
    index_cut = np.searchsorted(r2_growth_curve, r2_half_light * r_max ** 2)
    fit, cov = curve_fit(
        cumulaive_Moffat,
        r2_growth_curve[:index_cut],
        F_growth_curve[:index_cut],
        p0=(F_guess, r2_half_light, 1),
    )
    if plot:
        print("Best-fit: L_star = {}".format(fit[0]))
        print("          alpha = {}".format(np.sqrt(fit[1])))
        print("          beta = {}".format(fit[2]))
        r_norm = np.sqrt(np.array(r2_growth_curve)/r2_half_light)
        plt.plot(
            r_norm,
            cumulaive_Moffat(np.array(r2_growth_curve), fit[0], fit[1], fit[2])/fit[0],
            ":",
        )

    return fit
