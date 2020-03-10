# -*- coding: utf-8 -*-
"""
Functions related to flux calculations.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy import interpolate
from scipy.optimize import curve_fit

from ..constants import C
from .plots import plot_redshift_peaks
from .io import read_table


def gauss(x, x0, y0, sigma):
    p = [x0, y0, sigma]
    return p[1] * np.exp(-0.5 * ((x - p[0])/p[2]) ** 2)


def dgauss(x, x0, y0, sigma0, x1, y1, sigma1):
    p = [x0, y0, sigma0, x1, y1, sigma1]
    #         0   1    2      3    4  5
    return p[1] * np.exp(-0.5 * ((x - p[0])/p[2]) ** 2) + p[4] * np.exp(
        -0.5 * ((x - p[3])/p[5]) ** 2
    )


def gauss_fix_x0(x, x0, y0, sigma):
    """
    A Gaussian of fixed location (x0)

    Args:
        x (array): A list of x locations to make the Gaussian at
        x0 (float): Location of the Gaussian
        y0 (float): Amplitude
        sigma (float): Gaussian width
    """
    p = [y0, sigma]
    return p[0] * np.exp(-0.5 * ((x - x0)/p[1]) ** 2)


def gauss_flux(y0, sigma):  # THIS DOES NOT WORK...
    return y0 * sigma * np.sqrt(2 * np.pi)


def obtain_flux_calibration(calibration_star_cubes):
    #    print "\n> Obtaining flux calibration...\n"
    vector_wave = []
    vector_response = []
    cube_star = calibration_star_cubes[0]
    for i in range(len(cube_star.response_curve)):
        if np.isnan(cube_star.response_curve[i]) == False:
            vector_wave.append(cube_star.response_wavelength[i])
            vector_response.append(cube_star.response_curve[i])
            # print "  For wavelength = ",cube_star.response_wavelength[i], " the flux correction is = ", cube_star.response_curve[i]

    interpolated_response = interpolate.splrep(vector_wave, vector_response, s=0)
    flux_calibration = interpolate.splev(
        cube_star.wavelength, interpolated_response, der=0
    )
    #    flux_correction = flux_calibration

    print("\n> Flux calibration for all wavelengths = {}".format(flux_calibration))
    print("\n  Flux calibration obtained!")
    return flux_calibration


def fluxes(
    wavelength,
    s,
    line,
    lowlow=14,
    lowhigh=6,
    highlow=6,
    highhigh=14,
    lmin=0,
    lmax=0,
    fmin=0,
    fmax=0,
    broad=2.355,
    plot=True,
    verbose=True,
    plot_sus=False,
    fcal=True,
    fit_continuum=True,
    median_kernel=35,
    warnings=True,
):  # Broad is FWHM for Gaussian sigma= 1,
    """
    Provides integrated flux and perform a Gaussian fit to a given emission line.
    It follows the task "splot" in IRAF, with "e -> e" for integrated flux and "k -> k" for a Gaussian.

    Info from IRAF:\n
        - Integrated flux:\n
            center = sum (w(i) * (I(i)-C(i))**3/2) / sum ((I(i)-C(i))**3/2) (NOT USED HERE)\n
            continuum = C(midpoint) (NOT USED HERE) \n
            flux = sum ((I(i)-C(i)) * (w(i2) - w(i1)) / (i2 - i1)\n
            eq. width = sum (1 - I(i)/C(i))\n
        - Gaussian Fit:\n
             I(w) = cont + core * exp (-0.5*((w-center)/sigma)**2)\n
             fwhm = 2.355 * sigma\n
             flux = core * sigma * sqrt (2*pi)\n
             eq. width = abs (flux) / cont\n

    Returns
    -------

    This routine provides a list compiling the results. The list has the the following format:

        resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]

    "spectrum" in resultado[11] is the spectrum-fit (New 22 Jan 2019).

    Parameters
    ----------
    wavelength: float
        wavelength.
    spectrum: float
        flux per wavelength
    line: float
        approx. observed central wavelength of emission line to fit.
    lmin, lmax: float
        wavelength range to be analysed
    fmin, fmax: float (default = 0, 0.)
        minimum and maximum values of flux to be plotted.
        If 0 is given (i.e. defaul) the routine uses the nanmin and nanmax values of the given spectrum.
    plot: boolean (default = True)
        Plot a figure with the emission lines identifications.
    verbose: boolean (default = True)
        Print results.
    fit_continuum: boolean (default = True)
        Perform a linear fit of the continuum using all data, otherwise it just does a linear fit considering only the two median values in each continuum range.
    median_kernel: odd integer (default = 35)
        size of the median filter to be applied to the continuum.
    Example
    -------
    >>> resultado = fluxes(wavelength, spectrum, 6603, fmin=-5.0E-17, fmax=2.0E-16, plot=True, verbose=False)
    """

    # Setup wavelength limits
    if lmin == 0:
        lmin = line - 65.0  # By default, +-65 A with respect to line
    if lmax == 0:
        lmax = line + 65.0

    # Extract subrange to fit
    w_spec = []
    f_spec = []
    w_spec.extend(
        (wavelength[i])
        for i in range(len(wavelength))
        if (wavelength[i] > lmin and wavelength[i] < lmax)
    )
    f_spec.extend(
        (s[i])
        for i in range(len(wavelength))
        if (wavelength[i] > lmin and wavelength[i] < lmax)
    )

    # Setup min and max flux values in subrange to fit
    if fmin == 0:
        fmin = np.nanmin(f_spec)
    if fmax == 0:
        fmax = np.nanmax(f_spec)

    # We have to find some "guess numbers" for the Gaussian
    # Now guess_centre is line
    guess_centre = line

    # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to guess_centre
    #        lowlow   = 16.
    #        lowhigh  = 6.
    #        highlow  = 20.
    #        highhigh = 30.

    w_cont = []
    f_cont = []
    w_cont.extend(
        (w_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh)
        or (w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh)
    )
    f_cont.extend(
        (f_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh)
        or (w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh)
    )

    if fit_continuum:
        # Linear Fit to continuum
        f_cont_filtered = sig.medfilt(f_cont, np.int(median_kernel))
        # print line #f_cont
        #        if line == 8465.0:
        #            print w_cont
        #            print f_cont_filtered
        #            plt.plot(w_cont,f_cont_filtered)
        #            plt.show()
        #            plt.close()
        #            warnings=True
        try:
            mm, bb = np.polyfit(w_cont, f_cont_filtered, 1)
        except Exception:
            bb = np.nanmedian(f_cont_filtered)
            mm = 0.0
            if warnings:
                print("  Impossible to get the continuum!")
                print("  Scaling the continuum to the median value")
        continuum = mm * np.array(w_spec) + bb
        c_cont = mm * np.array(w_cont) + bb
    else:
        # Median value in each continuum range  # NEW 15 Sep 2019
        w_cont_low = []
        f_cont_low = []
        w_cont_low.extend(
            (w_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh
            )
        )
        f_cont_low.extend(
            (f_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh
            )
        )
        median_w_cont_low = np.nanmedian(w_cont_low)
        median_f_cont_low = np.nanmedian(f_cont_low)
        w_cont_high = []
        f_cont_high = []
        w_cont_high.extend(
            (w_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre + highlow
                and w_spec[i] < guess_centre + highhigh
            )
        )
        f_cont_high.extend(
            (f_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre + highlow
                and w_spec[i] < guess_centre + highhigh
            )
        )
        median_w_cont_high = np.nanmedian(w_cont_high)
        median_f_cont_high = np.nanmedian(f_cont_high)

        b = (median_f_cont_low - median_f_cont_high)/(
            median_w_cont_low - median_w_cont_high
        )
        a = median_f_cont_low - b * median_w_cont_low

        continuum = a + b * np.array(w_spec)
        c_cont = b * np.array(w_cont) + a

    # rms continuum
    rms_cont = np.nansum(
        [np.abs(f_cont[i] - c_cont[i]) for i in range(len(w_cont))]
    )/len(c_cont)

    # Search for index here w_spec(index) closest to line
    min_w = np.abs(np.array(w_spec) - line)
    mini = np.nanmin(min_w)
    #    guess_peak = f_spec[min_w.tolist().index(mini)]   # WE HAVE TO SUSTRACT CONTINUUM!!!
    guess_peak = (
        f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]
    )

    # LOW limit
    low_limit = 0
    w_fit = []
    f_fit = []
    w_fit.extend(
        (w_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre - 15 and w_spec[i] < guess_centre)
    )
    f_fit.extend(
        (f_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre - 15 and w_spec[i] < guess_centre)
    )
    if fit_continuum:
        c_fit = mm * np.array(w_fit) + bb
    else:
        c_fit = b * np.array(w_fit) + a

    fs = []
    ws = []
    for ii in range(len(w_fit) - 1, 1, -1):
        if (
            (f_fit[ii]/c_fit[ii]) < 1.05
            and (f_fit[ii - 1]/c_fit[ii - 1]) < 1.05
            and low_limit == 0
        ):
            low_limit = w_fit[ii]
        #        if f_fit[ii]/c_fit[ii] < 1.05 and low_limit == 0: low_limit = w_fit[ii]
        fs.append(f_fit[ii]/c_fit[ii])
        ws.append(w_fit[ii])
    if low_limit == 0:
        sorted_by_flux = np.argsort(fs)
        low_limit = ws[sorted_by_flux[0]]

    # HIGH LIMIT
    high_limit = 0
    w_fit = []
    f_fit = []
    w_fit.extend(
        (w_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre and w_spec[i] < guess_centre + 15)
    )
    f_fit.extend(
        (f_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre and w_spec[i] < guess_centre + 15)
    )
    if fit_continuum:
        c_fit = mm * np.array(w_fit) + bb
    else:
        c_fit = b * np.array(w_fit) + a

    fs = []
    ws = []
    for ii in range(len(w_fit) - 1):
        if (
            (f_fit[ii]/c_fit[ii]) < 1.05
            and (f_fit[ii + 1]/c_fit[ii + 1]) < 1.05
            and high_limit == 0
        ):
            high_limit = w_fit[ii]
        #        if f_fit[ii]/c_fit[ii] < 1.05 and high_limit == 0: high_limit = w_fit[ii]
        fs.append(f_fit[ii]/c_fit[ii])
        ws.append(w_fit[ii])
    if high_limit == 0:
        sorted_by_flux = np.argsort(fs)
        high_limit = ws[sorted_by_flux[0]]

    # Fit a Gaussian to data - continuum
    p0 = [
        guess_centre,
        guess_peak,
        broad / 2.355,
    ]  # broad is the Gaussian sigma, 1.0 for emission lines
    try:
        fit, pcov = curve_fit(
            gauss, w_spec, f_spec - continuum, p0=p0, maxfev=10000
        )  # If this fails, increase maxfev...
        fit_error = np.sqrt(np.diag(pcov))

        # New 28th Feb 2019: Check central value between low_limit and high_limit
        # Better: between guess_centre - broad, guess_centre + broad
        # If not, redo fit fixing central value to the peak (it does not work... just fix FWHM= (high_limit-low_limit)/2.5 )

        if verbose != False:
            print(" ----------------------------------------------------------------------------------------")
        #        if low_limit < fit[0] < high_limit:
        if fit[0] < guess_centre - broad or fit[0] > guess_centre + broad:
            #            if verbose: print "  Fitted center wavelength", fit[0],"is NOT in the range [",low_limit,",",high_limit,"]"
            if verbose:
                print("  Fitted center wavelength {} is NOT in the expected range [ {} , {} ]".format(
                    fit[0],guess_centre - broad, guess_centre + broad))

            #            print "Re-do fitting fixing center wavelength"
            #            p01 = [guess_peak, broad]
            #            fit1, pcov1 = curve_fit(gauss_fix_x0, w_spec, f_spec-continuum, p0=p01, maxfev=100000)   # If this fails, increase maxfev...
            #            fit_error1 = np.sqrt(np.diag(pcov1))
            #            fit[0]=guess_centre
            #            fit_error[0] = 0.
            #            fit[1] = fit1[0]
            #            fit_error[1] = fit_error1[0]
            #            fit[2] = fit1[1]
            #            fit_error[2] = fit_error1[1]

            fit[0] = guess_centre
            fit_error[0] = 0.000001
            fit[1] = guess_peak
            fit_error[1] = 0.000001
            fit[2] = broad / 2.355
            fit_error[2] = 0.000001
        else:
            if verbose:
                print("  Fitted center wavelength {} is NOT in the expected range [ {} , {} ]".format(
                    fit[0],guess_centre - broad,guess_centre + broad))

        # TILL HERE

        if verbose:
            print("  Fit parameters =  ", fit[0], fit[1], fit[2])
        if fit[2] == broad and warnings == True:
            print("  WARNING: Fit in", fit[
                0
            ], "failed! Using given centre wavelength (cw), peak at (cv) & sigma = broad/2.355 given.")
        gaussian_fit = gauss(w_spec, fit[0], fit[1], fit[2])

        # Estimate rms of the Gaussian fit in range [low_limit, high_limit]
        residuals = f_spec - gaussian_fit - continuum
        rms_fit = np.nansum(
            [
                ((residuals[i] ** 2)/(len(residuals) - 2)) ** 0.5
                for i in range(len(w_spec))
                if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)
            ]
        )

        # Fluxes, FWHM and Eq. Width calculations
        gaussian_flux = gauss_flux(fit[1], fit[2])
        error1 = np.abs(gauss_flux(fit[1] + fit_error[1], fit[2]) - gaussian_flux)
        error2 = np.abs(gauss_flux(fit[1], fit[2] + fit_error[2]) - gaussian_flux)
        gaussian_flux_error = (1/((1/error1 ** 2) + (1/error2 ** 2)) ** 0.5)

        fwhm = fit[2] * 2.355
        fwhm_error = fit_error[2] * 2.355
        fwhm_vel = (fwhm/fit[0]) * C
        fwhm_vel_error = (fwhm_error/fit[0]) * C

        gaussian_ew = gaussian_flux/np.nanmedian(f_cont)
        gaussian_ew_error = gaussian_ew * gaussian_flux_error/gaussian_flux

        # Integrated flux
        # IRAF: flux = sum ((I(i)-C(i)) * (w(i2) - w(i1)) / (i2 - i2)
        flux = np.nansum(
            [
                (f_spec[i] - continuum[i]) * (w_spec[i + 1] - w_spec[i])
                for i in range(len(w_spec))
                if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)
            ]
        )
        flux_error = rms_cont * (high_limit - low_limit)
        wave_resolution = (wavelength[-1] - wavelength[0])/len(wavelength)
        ew = wave_resolution * np.nansum(
            [
                (1 - (f_spec[i]/continuum[i]))
                for i in range(len(w_spec))
                if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)
            ]
        )
        ew_error = np.abs(ew * flux_error/flux)
        gauss_to_integrated = (gaussian_flux/flux) * 100.0

        # Plotting
        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(np.array(w_spec), np.array(f_spec), "b", lw=3, alpha=0.5)
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(lmin, lmax)
            plt.ylim(fmin, fmax)

            # Vertical line at guess_centre
            plt.axvline(x=guess_centre, color="r", linestyle="-", alpha=0.5)
            # Horizontal line at y = 0
            plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
            # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
            plt.axvspan(
                guess_centre + highlow,
                guess_centre + highhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            plt.axvspan(
                guess_centre - lowlow,
                guess_centre - lowhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            # Plot linear fit for continuum
            plt.plot(w_spec, continuum, "g--")
            # Plot Gaussian fit
            plt.plot(w_spec, gaussian_fit + continuum, "r-", alpha=0.8)
            # Vertical line at Gaussian center
            plt.axvline(x=fit[0], color="k", linestyle="-", alpha=0.5)
            # Vertical lines to emission line
            plt.axvline(x=low_limit, color="k", linestyle=":", alpha=0.5)
            plt.axvline(x=high_limit, color="k", linestyle=":", alpha=0.5)
            # Plot residuals
            plt.plot(w_spec, residuals, "k")
            plt.title(
                "Fit: x0={:.2f} y0={:.2e} sigma={:.2f}  flux={:.2e}  rms={:.3e}".format(
                    fit[0], fit[1], fit[2], gaussian_flux, rms_fit)
            )
            #plt.show()

        # Printing results
        if verbose:
            print("\n> Gauss and continuum fitting + integrated flux calculations:\n")
            print("rms continuum = {:.3e} erg/cm/s/A ".format(rms_cont))
            print("Gaussian Fit parameters: x0 = ( {:.2f} +- {:.2f} )  A ".format(
                fit[0],
                fit_error[0],
            ))
            print("                         y0 = ( {:.3f} +- {:.3f} )  1E-16 erg/cm2/s/A".format(
                (fit[1]/1e-16),
                (fit_error[1]/1e-16),
            ))
            print("                      sigma = ( {:.3f} +- {:.3f} )  A".format(
                fit[2],
                fit_error[2],
            ))
            print("                    rms fit = {:.3e} erg/cm2/s/A".format(rms_fit))
            print("Gaussian Flux = ( {:.2f} +- {:.2f} ) 1E-16 erg/s/cm2         (error = {:.1f} per cent)".format(
                (gaussian_flux/1e-16),
                (gaussian_flux_error/1e-16),
                (gaussian_flux_error/gaussian_flux) * 100,
            ))
            print("FWHM          = ( {:.3f} +- {:.3f} ) A    =   ( {:.1f} +- {:.1f} ) km/s ".format(
                fwhm,
                fwhm_error,
                fwhm_vel,
                fwhm_vel_error,
            ))
            print("Eq. Width     = ( {:.1f} +- {:.1f} ) A".format(
                -gaussian_ew,
                gaussian_ew_error,
            ))
            print("\nIntegrated flux  = ( {:.2f} +- {:.2f} ) 1E-16 erg/s/cm2      (error = {:.1f} per cent) ".format(
                (flux/1e-16),
                (flux_error/1e-16),
                (flux_error/flux) * 100,
            ))
            print("Eq. Width        = ( {:.1f} +- {:.1f} ) A".format(ew, ew_error))
            print("Gauss/Integrated = {:.2f per cent} ".format(gauss_to_integrated))

        # New 22 Jan 2019: sustract Gaussian fit

        index = 0
        s_s = np.zeros_like(s)
        for wave in range(len(wavelength)):
            s_s[wave] = s[wave]
            if wavelength[wave] == w_spec[0]:
                s_s[wave] = f_spec[0] - gaussian_fit[0]
                index = 1
            if wavelength[wave] > w_spec[0] and wavelength[wave] <= w_spec[-1]:
                s_s[wave] = f_spec[index] - gaussian_fit[index]
                index = index + 1
        if plot_sus:
            plt.figure(figsize=(10, 4))
            plt.plot(wavelength, s, "r")
            plt.plot(wavelength, s_s, "c")
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(lmin, lmax)
            plt.ylim(fmin, fmax)
            # plt.show()
            # plt.close()

        #                     0      1         2                3               4              5      6         7        8        9     10      11
        resultado = [
            rms_cont,
            fit[0],
            fit_error[0],
            gaussian_flux,
            gaussian_flux_error,
            fwhm,
            fwhm_error,
            flux,
            flux_error,
            ew,
            ew_error,
            s_s,
        ]
        return resultado
    except Exception:
        if verbose:
            print("  Gaussian fit failed!")
        resultado = [
            0,
            line,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            s,
        ]  # line was identified at lambda=line but Gaussian fit failed

        # NOTE: This can return the INTEGRATED FLUX although the Gaussian fit fails

        # Plotting
        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(np.array(w_spec), np.array(f_spec), "b", lw=3, alpha=0.5)
            plt.minorticks_on()
            plt.xlabel(r"Wavelength [$\AA$]")
            if fcal:
                plt.ylabel(r"Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(lmin, lmax)
            plt.ylim(fmin, fmax)

            # Vertical line at guess_centre
            plt.axvline(x=guess_centre, color="r", linestyle="-", alpha=0.5)
            # Horizontal line at y = 0
            plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
            # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
            plt.axvspan(
                guess_centre + highlow,
                guess_centre + highhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            plt.axvspan(
                guess_centre - lowlow,
                guess_centre - lowhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            # Plot linear fit for continuum
            plt.plot(w_spec, continuum, "g--")
            # Plot Gaussian fit
            #            plt.plot(w_spec, gaussian_fit+continuum, 'r-', alpha=0.8)
            # Vertical line at Gaussian center
            #            plt.axvline(x=fit[0], color='k', linestyle='-', alpha=0.5)
            # Vertical lines to emission line
            plt.axvline(x=low_limit, color="k", linestyle=":", alpha=0.5)
            plt.axvline(x=high_limit, color="k", linestyle=":", alpha=0.5)
            # Plot residuals
            #            plt.plot(w_spec, residuals, 'k')
            plt.title("No Gaussian fit obtained...")
            #plt.show()

        return resultado


def dfluxes(
    wavelength,
    s,
    line1,
    line2,
    lowlow=14,
    lowhigh=6,
    highlow=6,
    highhigh=14,
    lmin=0,
    lmax=0,
    fmin=0,
    fmax=0,
    broad1=2.355,
    broad2=2.355,
    plot=True,
    verbose=True,
    plot_sus=False,
    fcal=True,
    fit_continuum=True,
    median_kernel=35,
    warnings=True,
):  # Broad is FWHM for Gaussian sigma= 1,
    """
    Provides integrated flux and perform a Gaussian fit to a given emission line.
    It follows the task "splot" in IRAF, with "e -> e" for integrated flux and "k -> k" for a Gaussian.

    Info from IRAF:\n
        - Integrated flux:\n
            center = sum (w(i) * (I(i)-C(i))**3/2) / sum ((I(i)-C(i))**3/2) (NOT USED HERE)\n
            continuum = C(midpoint) (NOT USED HERE) \n
            flux = sum ((I(i)-C(i)) * (w(i2) - w(i1)) / (i2 - i1)\n
            eq. width = sum (1 - I(i)/C(i))\n
        - Gaussian Fit:\n
             I(w) = cont + core * exp (-0.5*((w-center)/sigma)**2)\n
             fwhm = 2.355 * sigma\n
             flux = core * sigma * sqrt (2*pi)\n
             eq. width = abs (flux) / cont\n

    Returns
    -------

    This routine provides a list compiling the results. The list has the following format:

        resultado = [rms_cont, fit[0], fit_error[0], gaussian_flux, gaussian_flux_error, fwhm, fwhm_error, flux, flux_error, ew, ew_error, spectrum  ]

    "spectrum" in resultado[11] is the spectrum-fit (New 22 Jan 2019).

    Parameters
    ----------
    wavelength: float
        wavelength.
    spectrum: float
        flux per wavelength
    line: float
        approx. observed central wavelength of emission line to fit.
    lmin, lmax: float
        wavelength range to be analysed
    fmin, fmax: float (default = 0, 0.)
        minimum and maximum values of flux to be plotted.
        If 0 is given (i.e. defaul) the routine uses the nanmin and nanmax values of the given spectrum.
    plot: boolean (default = True)
        Plot a figure with the emission lines identifications.
    verbose: boolean (default = True)
        Print results.
    fit_continuum: boolean (default = True)
        Perform a linear fit of the continuum using all data, otherwise it just does a linear fit considering only the two median values in each continuum range.
    median_kernel: odd integer (default = 35)
        size of the median filter to be applied to the continuum.

    Example
    -------
    >>> resultado = fluxes(wavelength, spectrum, 6603, fmin=-5.0E-17, fmax=2.0E-16, plot=True, verbose=False)
    """

    # Setup wavelength limits
    if lmin == 0:
        lmin = line1 - 65.0  # By default, +-65 A with respect to line
    if lmax == 0:
        lmax = line2 + 65.0

    # Extract subrange to fit
    w_spec = []
    f_spec = []
    w_spec.extend(
        (wavelength[i])
        for i in range(len(wavelength))
        if (wavelength[i] > lmin and wavelength[i] < lmax)
    )
    f_spec.extend(
        (s[i])
        for i in range(len(wavelength))
        if (wavelength[i] > lmin and wavelength[i] < lmax)
    )

    # Setup min and max flux values in subrange to fit
    if fmin == 0:
        fmin = np.nanmin(f_spec)
    if fmax == 0:
        fmax = np.nanmax(f_spec)

    # We have to find some "guess numbers" for the Gaussian
    # Now guess_centre is line
    guess_centre1 = line1
    guess_centre2 = line2
    guess_centre = (guess_centre1 + guess_centre2) / 2.0
    # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to guess_centre

    w_cont = []
    f_cont = []
    w_cont.extend(
        (w_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh)
        or (w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh)
    )
    f_cont.extend(
        (f_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh)
        or (w_spec[i] > guess_centre + highlow and w_spec[i] < guess_centre + highhigh)
    )

    if fit_continuum:
        # Linear Fit to continuum
        f_cont_filtered = sig.medfilt(f_cont, np.int(median_kernel))
        try:
            mm, bb = np.polyfit(w_cont, f_cont_filtered, 1)
        except Exception:
            bb = np.nanmedian(f_cont_filtered)
            mm = 0.0
            if warnings:
                print("  Impossible to get the continuum!")
                print("  Scaling the continuum to the median value")
        continuum = mm * np.array(w_spec) + bb
        c_cont = mm * np.array(w_cont) + bb
    else:
        # Median value in each continuum range  # NEW 15 Sep 2019
        w_cont_low = []
        f_cont_low = []
        w_cont_low.extend(
            (w_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh
            )
        )
        f_cont_low.extend(
            (f_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre - lowlow and w_spec[i] < guess_centre - lowhigh
            )
        )
        median_w_cont_low = np.nanmedian(w_cont_low)
        median_f_cont_low = np.nanmedian(f_cont_low)
        w_cont_high = []
        f_cont_high = []
        w_cont_high.extend(
            (w_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre + highlow
                and w_spec[i] < guess_centre + highhigh
            )
        )
        f_cont_high.extend(
            (f_spec[i])
            for i in range(len(w_spec))
            if (
                w_spec[i] > guess_centre + highlow
                and w_spec[i] < guess_centre + highhigh
            )
        )
        median_w_cont_high = np.nanmedian(w_cont_high)
        median_f_cont_high = np.nanmedian(f_cont_high)

        b = ((median_f_cont_low - median_f_cont_high)/(
            median_w_cont_low - median_w_cont_high
        ))
        a = median_f_cont_low - b * median_w_cont_low

        continuum = a + b * np.array(w_spec)
        c_cont = b * np.array(w_cont) + a

    # rms continuum
    rms_cont = (np.nansum(
        [np.abs(f_cont[i] - c_cont[i]) for i in range(len(w_cont))]
    )/len(c_cont))

    # Search for index here w_spec(index) closest to line
    min_w = np.abs(np.array(w_spec) - line1)
    mini = np.nanmin(min_w)
    guess_peak1 = (
        f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]
    )
    min_w = np.abs(np.array(w_spec) - line2)
    mini = np.nanmin(min_w)
    guess_peak2 = (
        f_spec[min_w.tolist().index(mini)] - continuum[min_w.tolist().index(mini)]
    )

    # Search for beginning/end of emission line, choosing line +-10
    # 28th Feb 2019: Check central value between low_limit and high_limit

    # LOW limit
    low_limit = 0
    w_fit = []
    f_fit = []
    w_fit.extend(
        (w_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre1 - 15 and w_spec[i] < guess_centre1)
    )
    f_fit.extend(
        (f_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre1 - 15 and w_spec[i] < guess_centre1)
    )
    if fit_continuum:
        c_fit = mm * np.array(w_fit) + bb
    else:
        c_fit = b * np.array(w_fit) + a

    fs = []
    ws = []
    for ii in range(len(w_fit) - 1, 1, -1):
        if (
            (f_fit[ii]/c_fit[ii]) < 1.05
            and (f_fit[ii - 1]/c_fit[ii - 1]) < 1.05
            and low_limit == 0
        ):
            low_limit = w_fit[ii]
        #        if f_fit[ii]/c_fit[ii] < 1.05 and low_limit == 0: low_limit = w_fit[ii]
        fs.append((f_fit[ii]/c_fit[ii]))
        ws.append(w_fit[ii])
    if low_limit == 0:
        sorted_by_flux = np.argsort(fs)
        low_limit = ws[sorted_by_flux[0]]

    # HIGH LIMIT
    high_limit = 0
    w_fit = []
    f_fit = []
    w_fit.extend(
        (w_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre2 and w_spec[i] < guess_centre2 + 15)
    )
    f_fit.extend(
        (f_spec[i])
        for i in range(len(w_spec))
        if (w_spec[i] > guess_centre2 and w_spec[i] < guess_centre2 + 15)
    )
    if fit_continuum:
        c_fit = mm * np.array(w_fit) + bb
    else:
        c_fit = b * np.array(w_fit) + a

    fs = []
    ws = []
    for ii in range(len(w_fit) - 1):
        if (
            (f_fit[ii]/c_fit[ii]) < 1.05
            and (f_fit[ii + 1]/c_fit[ii + 1]) < 1.05
            and high_limit == 0
        ):
            high_limit = w_fit[ii]
        #        if f_fit[ii]/c_fit[ii] < 1.05 and high_limit == 0: high_limit = w_fit[ii]
        fs.append((f_fit[ii]/c_fit[ii]))
        ws.append(w_fit[ii])
    if high_limit == 0:
        sorted_by_flux = np.argsort(fs)
        high_limit = ws[sorted_by_flux[0]]

    # Fit a Gaussian to data - continuum
    p0 = [
        guess_centre1,
        guess_peak1,
        broad1 / 2.355,
        guess_centre2,
        guess_peak2,
        broad2 / 2.355,
    ]  # broad is the Gaussian sigma, 1.0 for emission lines
    try:
        fit, pcov = curve_fit(
            dgauss, w_spec, f_spec - continuum, p0=p0, maxfev=10000
        )  # If this fails, increase maxfev...
        fit_error = np.sqrt(np.diag(pcov))

        # New 28th Feb 2019: Check central value between low_limit and high_limit
        # Better: between guess_centre - broad, guess_centre + broad
        # If not, redo fit fixing central value to the peak (it does not work... just fix FWHM= (high_limit-low_limit)/2.5 )

        if verbose != False:
            print(" ----------------------------------------------------------------------------------------")
        if (
            fit[0] < guess_centre1 - broad1
            or fit[0] > guess_centre1 + broad1
            or fit[3] < guess_centre2 - broad2
            or fit[3] > guess_centre2 + broad2
        ):
            if warnings:
                if fit[0] < guess_centre1 - broad1 or fit[0] > guess_centre1 + broad1:
                    print("  Fitted center wavelength {} is NOT in the expected range [ {} , {} ]".format(
                        fit[0], guess_centre1 - broad1, guess_centre1 + broad1))
                else:
                    print("  Fitted center wavelength {} is in the expected range [ {} , {} ]".format(
                        fit[0], guess_centre1 - broad1, guess_centre1 + broad1))
                if fit[3] < guess_centre2 - broad2 or fit[3] > guess_centre2 + broad2:
                    print("  Fitted center wavelength {} is NOT in the expected range [ {} , {} ]".format(
                        fit[3], guess_centre2 - broad2, guess_centre2 + broad2))
                else:
                    print("  Fitted center wavelength {} is in the expected range [ {} , {} ]".format(
                    fit[3], guess_centre2 - broad2, guess_centre2 + broad2))
                print("  Fit failed!")

            fit[0] = guess_centre1
            fit_error[0] = 0.000001
            fit[1] = guess_peak1
            fit_error[1] = 0.000001
            fit[2] = broad1 / 2.355
            fit_error[2] = 0.000001
            fit[3] = guess_centre2
            fit_error[3] = 0.000001
            fit[4] = guess_peak2
            fit_error[4] = 0.000001
            fit[5] = broad2 / 2.355
            fit_error[5] = 0.000001
        else:
            if warnings:
                print("  Fitted center wavelength {} is in the expected range [ {} , {} ]".format(
                    fit[0], guess_centre1 - broad1, guess_centre1 + broad1))
            if warnings:
                print("  Fitted center wavelength {} is in the expected range [ {} , {} ]".format(
                    fit[3], guess_centre2 - broad2, guess_centre2 + broad2))

        gaussian_fit = dgauss(w_spec, fit[0], fit[1], fit[2], fit[3], fit[4], fit[5])

        if warnings:
            print("  Fit parameters =  {} {} {} {} {} {}".format(fit[0], fit[1], fit[2], fit[3], fit[4], fit[5]))
        if fit[2] == broad1 and warnings == True:
            print("  WARNING: Fit in {} failed! Using given centre wavelength (cw), peak at (cv) & sigma = broad/2.355 given.".format(fit[0]))  # CHECK THIS
        # gaussian_fit =  gauss(w_spec, fit[0], fit[1], fit[2])

        # Estimate rms of the Gaussian fit in range [low_limit, high_limit]
        residuals = f_spec - gaussian_fit - continuum
        rms_fit = np.nansum(
            [
                (((residuals[i] ** 2)/(len(residuals) - 2))) ** 0.5
                for i in range(len(w_spec))
                if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)
            ]
        )

        # Fluxes, FWHM and Eq. Width calculations  # CHECK THIS
        gaussian_flux = gauss_flux(fit[1], fit[2])
        error1 = np.abs(gauss_flux(fit[1] + fit_error[1], fit[2]) - gaussian_flux)
        error2 = np.abs(gauss_flux(fit[1], fit[2] + fit_error[2]) - gaussian_flux)
        gaussian_flux_error = (1/((1/error1 ** 2) + (1/error2 ** 2)) ** 0.5)

        fwhm = fit[2] * 2.355
        fwhm_error = fit_error[2] * 2.355
        fwhm_vel = (fwhm/fit[0]) * C
        fwhm_vel_error = (fwhm_error/fit[0]) * C

        gaussian_ew = (gaussian_flux/np.nanmedian(f_cont))
        gaussian_ew_error = (gaussian_ew * gaussian_flux_error/gaussian_flux)

        # Integrated flux
        # IRAF: flux = sum ((I(i)-C(i)) * (w(i2) - w(i1)) / (i2 - i2)
        flux = np.nansum(
            [
                (f_spec[i] - continuum[i]) * (w_spec[i + 1] - w_spec[i])
                for i in range(len(w_spec))
                if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)
            ]
        )
        flux_error = rms_cont * (high_limit - low_limit)
        wave_resolution = ((wavelength[-1] - wavelength[0])/len(wavelength))
        ew = wave_resolution * np.nansum(
            [
                (1 - (f_spec[i]/continuum[i]))
                for i in range(len(w_spec))
                if (w_spec[i] >= low_limit and w_spec[i] <= high_limit)
            ]
        )
        ew_error = np.abs((ew * flux_error/flux))
        gauss_to_integrated = (gaussian_flux/flux) * 100.0

        # Plotting
        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(np.array(w_spec), np.array(f_spec), "b", lw=3, alpha=0.5)
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(((line1 + line2)/2) - 40, ((line1 + line2)/2) + 40)
            plt.ylim(fmin, fmax)

            # Vertical line at guess_centre
            plt.axvline(x=guess_centre1, color="r", linestyle="-", alpha=0.5)
            plt.axvline(x=guess_centre2, color="r", linestyle="-", alpha=0.5)

            # Horizontal line at y = 0
            plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
            # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
            plt.axvspan(
                guess_centre + highlow,
                guess_centre + highhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            plt.axvspan(
                guess_centre - lowlow,
                guess_centre - lowhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            # Plot linear fit for continuum
            plt.plot(w_spec, continuum, "g--")
            # Plot Gaussian fit
            plt.plot(w_spec, gaussian_fit + continuum, "r-", alpha=0.8)
            # Vertical line at Gaussian center
            plt.axvline(x=fit[0], color="k", linestyle="-", alpha=0.5)
            plt.axvline(x=fit[3], color="k", linestyle="-", alpha=0.5)

            # Vertical lines to emission line
            plt.axvline(x=low_limit, color="k", linestyle=":", alpha=0.5)
            plt.axvline(x=high_limit, color="k", linestyle=":", alpha=0.5)
            # Plot residuals
            plt.plot(w_spec, residuals, "k")
            plt.title(
                "Double Gaussian Fit"
            )  # Fit: x0=%.2f y0=%.2e sigma=%.2f  flux=%.2e  rms=%.3e' % (fit[0], fit[1], fit[2], gaussian_flux, rms_fit))
            #plt.show()

        # Printing results
        if verbose:
            print("\n> Gauss and continuum fitting + integrated flux calculations:\n")
            print("rms continuum = {:.3e} erg/cm/s/A ".format(rms_cont))
            print("Gaussian Fit parameters: x0 = ( {:.2f} +- {:.2f} )  A ".format(
                fit[0],
                fit_error[0],
            ))
            print("                         y0 = ( {:.3f} +- {:.3f} )  1E-16 erg/cm2/s/A".format(
                (fit[1]/1e-16),
                (fit_error[1]/1e-16),
            ))
            print("                      sigma = ( {:.3f} +- {:.3f} )  A".format(
                fit[2],
                fit_error[2],
            ))
            print("                    rms fit = {:.3e} erg/cm2/s/A".format(rms_fit))
            print("Gaussian Flux = ( {:.2f} +- {:.2f} ) 1E-16 erg/s/cm2         (error = {:.1f} per cent)".format(
                (gaussian_flux/1e-16),
                (gaussian_flux_error/1e-16),
                (gaussian_flux_error/gaussian_flux) * 100,
            ))
            print("FWHM          = ( {:.3f} +- {:.3f} ) A    =   ( {:.1f} +- {:.1f} ) km/s ".format(
                fwhm,
                fwhm_error,
                fwhm_vel,
                fwhm_vel_error,
            ))
            print("Eq. Width     = ( {:.1f} +- {:.1f} ) A".format(
                -gaussian_ew,
                gaussian_ew_error,
            ))
            print("\nIntegrated flux  = ( {:.2f} +- {:.2f} ) 1E-16 erg/s/cm2      (error = {:.1f} per cent) ".format(
                (flux/1e-16),
                (flux_error/1e-16),
                (flux_error/flux) * 100,
            ))
            print("Eq. Width        = ( {:.1f} +- {:.1f} ) A".format(ew, ew_error))
            print("Gauss/Integrated = {:.2f} per cent ".format(gauss_to_integrated))

        # New 22 Jan 2019: sustract Gaussian fit

        index = 0
        s_s = np.zeros_like(s)
        for wave in range(len(wavelength)):
            s_s[wave] = s[wave]
            if wavelength[wave] == w_spec[0]:
                s_s[wave] = f_spec[0] - gaussian_fit[0]
                index = 1
            if wavelength[wave] > w_spec[0] and wavelength[wave] <= w_spec[-1]:
                s_s[wave] = f_spec[index] - gaussian_fit[index]
                index = index + 1
        if plot_sus:
            plt.figure(figsize=(10, 4))
            plt.plot(wavelength, s, "r")
            plt.plot(wavelength, s_s, "c")
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(lmin, lmax)
            plt.ylim(fmin, fmax)
            # plt.show()
            # plt.close()

        #                     0      1         2                3               4              5      6         7        8        9     10      11   12       13      14
        resultado = [
            rms_cont,
            fit[0],
            fit_error[0],
            gaussian_flux,
            gaussian_flux_error,
            fwhm,
            fwhm_error,
            flux,
            flux_error,
            ew,
            ew_error,
            s_s,
            fit[3],
            fit[4],
            fit[5],
        ]
        return resultado
    except Exception:
        if verbose:
            print("  Gaussian fit failed!")
        resultado = [
            0,
            line1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            s,
            0,
            0,
            0,
        ]  # line was identified at lambda=line but Gaussian fit failed

        # NOTA: PUEDE DEVOLVER EL FLUJO INTEGRADO AUNQUE FALLE EL AJUSTE GAUSSIANO...

        # Plotting
        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(np.array(w_spec), np.array(f_spec), "b", lw=3, alpha=0.5)
            plt.minorticks_on()
            plt.xlabel("Wavelength [$\AA$]")
            if fcal:
                plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
            else:
                plt.ylabel("Flux [ counts ]")
            plt.xlim(lmin, lmax)
            plt.ylim(fmin, fmax)

            # Vertical line at guess_centre
            plt.axvline(x=guess_centre, color="r", linestyle="-", alpha=0.5)
            # Horizontal line at y = 0
            plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
            # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
            plt.axvspan(
                guess_centre + highlow,
                guess_centre + highhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            plt.axvspan(
                guess_centre - lowlow,
                guess_centre - lowhigh,
                facecolor="g",
                alpha=0.15,
                zorder=3,
            )
            # Plot linear fit for continuum
            plt.plot(w_spec, continuum, "g--")
            # Plot Gaussian fit
            #            plt.plot(w_spec, gaussian_fit+continuum, 'r-', alpha=0.8)
            # Vertical line at Gaussian center
            #            plt.axvline(x=fit[0], color='k', linestyle='-', alpha=0.5)
            # Vertical lines to emission line
            plt.axvline(x=low_limit, color="k", linestyle=":", alpha=0.5)
            plt.axvline(x=high_limit, color="k", linestyle=":", alpha=0.5)
            # Plot residuals
            #            plt.plot(w_spec, residuals, 'k')
            plt.title("No Gaussian fit obtained...")
            #plt.show()

        return resultado



def search_peaks(
    wavelength,
    flux,
    smooth_points=20,
    lmin=0,
    lmax=0,
    fmin=0.5,
    fmax=3.0,
    emission_line_file="lineas_c89_python.dat",
    brightest_line="Ha",
    cut=1.2,
    check_redshift=0.0003,
    only_id_lines=True,
    plot=True,
    verbose=True,
    fig_size=12,
):
    """
    Search and identify emission lines in a given spectrum.\n
    For this the routine first fits a rough estimation of the global continuum.\n
    Then it uses "flux"/"continuum" > "cut" to search for the peaks, assuming this
    is satisfied in at least 2 consecutive wavelengths.\n
    Once the peaks are found, the routine identifies the brightest peak with the given "brightest_line",
    that has to be included in the text file "emission_line_file".\n
    After that the routine checks if the rest of identified peaks agree with the emission
    lines listed in text file "emission_line_file".
    If abs(difference in wavelength) > 2.5, we don't consider the line identified.\n
    Finally, it checks if all redshifts are similar, assuming "check_redshift" = 0.0003 by default.

    Returns
    -------

    The routine returns FOUR lists:

    peaks: (float) wavelength of the peak of the detected emission lines.
        It is NOT necessarily the central wavelength of the emission lines.
    peaks_name: (string).
        name of the detected emission lines.
    peaks_rest: (float)
        rest wavelength of the detected emission lines
    continuum_limits: (float):
        provides the values for fitting the local continuum
        for each of the detected emission lines, given in the format
        [lowlow, lowhigh, highlow, highhigh]

    Parameters
    ----------
    wavelength: float
        wavelength.
    flux: float
        flux per wavelength
    smooth_points: float (default = 20)
        Number of points for a smooth spectrum to get a rough estimation of the global continuum
    lmin, lmax: float
        wavelength range to be analysed
    fmin, fmax: float (default = 0.5, 2.)
        minimum and maximum values of flux/continuum to be plotted
    emission_line_file: string (default = "lineas_c89_python.dat")
        tex file with a list of emission lines to be found.
        This text file has to have the following format per line:

        rest_wavelength  name   f(lambda) lowlow lowhigh  highlow  highhigh

        E.g.: 6300.30       [OI]  -0.263   15.0  4.0   20.0   40.0

    brightest_line: string (default="Ha")
        expected emission line in the spectrum
    cut: float (default = 1.2)
        minimum value of flux/continuum to check for emission lines
    check_redshift: float (default = 0.0003)
        check if the redshifts derived using the detected emission lines agree with that obtained for
        the brightest emission line (ref.). If abs(z - zred) > check_redshift a warning is shown.
    plot: boolean (default = True)
        Plot a figure with the emission lines identifications.
    verbose: boolean (default = True)
        Print results.
    only_id_lines: boolean (default = True)
        Provide only the list of the identified emission lines

    Example
    -------
    >>> peaks,peaks_name,peaks_rest,continuum_limits=search_peaks(wavelength, spectrum, plot=False)
    """
    # Setup wavelength limits
    if lmin == 0:
        lmin = np.nanmin(wavelength)
    if lmax == 0:
        lmax = np.nanmax(wavelength)

    # Fit a smooth continuum
    # smooth_points = 20      # Points in the interval
    step = np.int((len(wavelength)/smooth_points))  # step
    w_cont_smooth = np.zeros(smooth_points)
    f_cont_smooth = np.zeros(smooth_points)

    for j in range(smooth_points):
        w_cont_smooth[j] = np.nanmedian(
            [
                wavelength[i]
                for i in range(len(wavelength))
                if (i > step * j and i < step * (j + 1))
            ]
        )
        f_cont_smooth[j] = np.nanmedian(
            [
                flux[i]
                for i in range(len(wavelength))
                if (i > step * j and i < step * (j + 1))
            ]
        )  # / np.nanmedian(spectrum)
        # print j,w_cont_smooth[j], f_cont_smooth[j]

    interpolated_continuum_smooth = interpolate.splrep(
        w_cont_smooth, f_cont_smooth, s=0
    )
    interpolated_continuum = interpolate.splev(
        wavelength, interpolated_continuum_smooth, der=0
    )

    funcion = (flux/interpolated_continuum)

    # Searching for peaks using cut = 1.2 by default
    peaks = []
    index_low = 0
    for i in range(len(wavelength)):
        if funcion[i] > cut and funcion[i - 1] < cut:
            index_low = i
        if funcion[i] < cut and funcion[i - 1] > cut:
            index_high = i
            if index_high != 0:
                pfun = np.nanmax(
                    [
                        funcion[j]
                        for j in range(len(wavelength))
                        if (j > index_low and j < index_high + 1)
                    ]
                )
                peak = wavelength[funcion.tolist().index(pfun)]
                if (index_high - index_low) > 1:
                    peaks.append(peak)

    # Identify lines
    # Read file with data of emission lines:
    # 6300.30 [OI] -0.263    15   5    5    15
    # el_center el_name el_fnl lowlow lowhigh highlow highigh
    # Only el_center and el_name are needed
    (
        el_center,
        el_name,
        el_fnl,
        el_lowlow,
        el_lowhigh,
        el_highlow,
        el_highhigh,
    ) = read_table(emission_line_file, ["f", "s", "f", "f", "f", "f", "f"])
    # for i in range(len(el_name)):
    #    print " %8.2f  %9s  %6.3f   %4.1f %4.1f   %4.1f   %4.1f" % (el_center[i],el_name[i],el_fnl[i],el_lowlow[i], el_lowhigh[i], el_highlow[i], el_highhigh[i])
    # el_center,el_name = read_table("lineas_c89_python.dat", ["f", "s"] )

    # In case this is needed in the future...
    #    el_center = [6300.30, 6312.10, 6363.78, 6548.03, 6562.82, 6583.41, 6678.15, 6716.47, 6730.85, 7065.28, 7135.78, 7318.39, 7329.66]
    #    el_fnl    = [-0.263, -0.264, -0.271, -0.296,    -0.298, -0.300, -0.313, -0.318, -0.320,    -0.364,   -0.374, -0.398, -0.400 ]
    #    el_name   = ["[OI]", "[SIII]", "[OI]", "[NII]", "Ha", "[NII]",  "HeI", "[SII]", "[SII]",   "HeI",  "[ArIII]", "[OII]", "[OII]" ]

    # Search for the brightest line in given spectrum ("Ha" by default)
    peaks_flux = np.zeros(len(peaks))
    for i in range(len(peaks)):
        peaks_flux[i] = flux[wavelength.tolist().index(peaks[i])]
    Ha_w_obs = peaks[peaks_flux.tolist().index(np.nanmax(peaks_flux))]

    # Estimate redshift of the brightest line ( Halpha line by default)
    Ha_index_list = el_name.tolist().index(brightest_line)
    Ha_w_rest = el_center[Ha_index_list]
    Ha_redshift = ((Ha_w_obs - Ha_w_rest)/Ha_w_rest)
    if verbose:
        print("\n> Detected {:d} emission lines using {:8s} at {:8.2f} A as brightest line!!\n".format(
            len(peaks),
            brightest_line,
            Ha_w_rest,
        ))
    #    if verbose: print "  Using %8s at %8.2f A as brightest line  --> Found in %8.2f with a redshift %.6f " % (brightest_line, Ha_w_rest, Ha_w_obs, Ha_redshift)

    # Identify lines using brightest line (Halpha by default) as reference.
    # If abs(wavelength) > 2.5 we don't consider it identified.
    peaks_name = [None] * len(peaks)
    peaks_rest = np.zeros(len(peaks))
    peaks_redshift = np.zeros(len(peaks))
    peaks_lowlow = np.zeros(len(peaks))
    peaks_lowhigh = np.zeros(len(peaks))
    peaks_highlow = np.zeros(len(peaks))
    peaks_highhigh = np.zeros(len(peaks))

    for i in range(len(peaks)):
        minimo_w = np.abs((peaks[i]/(1 + Ha_redshift)) - el_center)
        if np.nanmin(minimo_w) < 2.5:
            indice = minimo_w.tolist().index(np.nanmin(minimo_w))
            peaks_name[i] = el_name[indice]
            peaks_rest[i] = el_center[indice]
            peaks_redshift[i] = ((peaks[i] - el_center[indice])/el_center[indice])
            peaks_lowlow[i] = el_lowlow[indice]
            peaks_lowhigh[i] = el_lowhigh[indice]
            peaks_highlow[i] = el_highlow[indice]
            peaks_highhigh[i] = el_highhigh[indice]
            if verbose:
                print("{:9s} {:8.2f} found in {:8.2f} at z={:.6f}   |z-zref| = {:.6f}".format(
                    peaks_name[i],
                    peaks_rest[i],
                    peaks[i],
                    peaks_redshift[i],
                    np.abs(peaks_redshift[i] - Ha_redshift),
                ))
            # print peaks_lowlow[i],peaks_lowhigh[i],peaks_highlow[i],peaks_highhigh[i]
    # Check if all redshifts are similar, assuming check_redshift = 0.0003 by default
    # If OK, add id_peaks[i]=1, if not, id_peaks[i]=0
    id_peaks = []
    for i in range(len(peaks_redshift)):
        if np.abs(peaks_redshift[i] - Ha_redshift) > check_redshift:
            if verbose:
                print("  WARNING!!! Line {:8s} in w = {:.2f} has redshift z={:.6f}, different than zref={:.6f}".format(
                    peaks_name[i],
                    peaks[i],
                    peaks_redshift[i],
                    Ha_redshift,
                ))
            id_peaks.append(0)
        else:
            id_peaks.append(1)

    if plot:
        fig = plot_redshift_peaks(fig_size,
                            funcion,
                            wavelength,
                            lmin,
                            lmax,
                            fmin,
                            fmax,
                            cut,
                            peaks,
                            peaks_name,
                            #label)  # TODO: label is unreferenced.
        )
    continuum_limits = [peaks_lowlow, peaks_lowhigh, peaks_highlow, peaks_highhigh]

    if only_id_lines:
        peaks_r = []
        peaks_name_r = []
        peaks_rest_r = []
        peaks_lowlow_r = []
        peaks_lowhigh_r = []
        peaks_highlow_r = []
        peaks_highhigh_r = []

        for i in range(len(peaks)):
            if id_peaks[i] == 1:
                peaks_r.append(peaks[i])
                peaks_name_r.append(peaks_name[i])
                peaks_rest_r.append(peaks_rest[i])
                peaks_lowlow_r.append(peaks_lowlow[i])
                peaks_lowhigh_r.append(peaks_lowhigh[i])
                peaks_highlow_r.append(peaks_highlow[i])
                peaks_highhigh_r.append(peaks_highhigh[i])
        continuum_limits_r = [
            peaks_lowlow_r,
            peaks_lowhigh_r,
            peaks_highlow_r,
            peaks_highhigh_r,
        ]

        return peaks_r, peaks_name_r, peaks_rest_r, continuum_limits_r
    else:
        return peaks, peaks_name, peaks_rest, continuum_limits


def substract_given_gaussian(
    wavelength,
    spectrum,
    centre,
    peak=0,
    sigma=0,
    flux=0,
    search_peak=False,
    allow_absorptions=False,
    lowlow=20,
    lowhigh=10,
    highlow=10,
    highhigh=20,
    lmin=0,
    lmax=0,
    fmin=0,
    fmax=0,
    plot=True,
    fcal=False,
    verbose=True,
):
    """
    Substract a give Gaussian to a spectrum after fitting the continuum.
    """
    # plot = True
    # verbose = True

    # Check that we have the numbers!
    if peak != 0 and sigma != 0:
        do_it = True

    if peak == 0 and flux != 0 and sigma != 0:
        # flux = peak * sigma * np.sqrt(2*np.pi)
        peak = flux/(sigma * np.sqrt(2 * np.pi))
        do_it = True

    if sigma == 0 and flux != 0 and peak != 0:
        # flux = peak * sigma * np.sqrt(2*np.pi)
        sigma = flux/(peak * np.sqrt(2 * np.pi))
        do_it = True

    if flux == 0 and sigma != 0 and peak != 0:
        flux = peak * sigma * np.sqrt(2 * np.pi)
        do_it = True

    if sigma != 0 and search_peak == True:
        do_it = True

    if do_it == False:
        print("> Error! We need data to proceed! Give at least two of [peak, sigma, flux], or sigma and force peak to f[centre]")
    else:
        # Setup wavelength limits
        if lmin == 0:
            lmin = centre - 65.0  # By default, +-65 A with respect to line
        if lmax == 0:
            lmax = centre + 65.0

        # Extract subrange to fit
        w_spec = []
        f_spec = []
        w_spec.extend(
            (wavelength[i])
            for i in range(len(wavelength))
            if (wavelength[i] > lmin and wavelength[i] < lmax)
        )
        f_spec.extend(
            (spectrum[i])
            for i in range(len(wavelength))
            if (wavelength[i] > lmin and wavelength[i] < lmax)
        )

        # Setup min and max flux values in subrange to fit
        if fmin == 0:
            fmin = np.nanmin(f_spec)
        if fmax == 0:
            fmax = np.nanmax(f_spec)

        # Define continuum regions: [-lowlow, -lowhigh]  and [highlow,highhigh] in Angstroms with respect to centre
        w_cont = []
        f_cont = []
        w_cont.extend(
            (w_spec[i])
            for i in range(len(w_spec))
            if (w_spec[i] > centre - lowlow and w_spec[i] < centre - lowhigh)
            or (w_spec[i] > centre + highlow and w_spec[i] < centre + highhigh)
        )
        f_cont.extend(
            (f_spec[i])
            for i in range(len(w_spec))
            if (w_spec[i] > centre - lowlow and w_spec[i] < centre - lowhigh)
            or (w_spec[i] > centre + highlow and w_spec[i] < centre + highhigh)
        )

        # Linear Fit to continuum
        try:
            mm, bb = np.polyfit(w_cont, f_cont, 1)
        except Exception:
            bb = np.nanmedian(spectrum)
            mm = 0.0
            if verbose:
                print("  Impossible to get the continuum!")
                print("  Scaling the continuum to the median value")
        continuum = mm * np.array(w_spec) + bb
        # c_cont = mm*np.array(w_cont)+bb
        # rms continuum
        # rms_cont = np.nansum([ np.abs(f_cont[i] - c_cont[i])  for i in range(len(w_cont)) ]) / len(c_cont)

        if search_peak:
            # Search for index here w_spec(index) closest to line
            try:
                min_w = np.abs(np.array(w_spec) - centre)
                mini = np.nanmin(min_w)
                peak = (
                    f_spec[min_w.tolist().index(mini)]
                    - continuum[min_w.tolist().index(mini)]
                )
                flux = peak * sigma * np.sqrt(2 * np.pi)
                if verbose:
                    print("  Using peak as f[ {} ] = {}  and sigma = {}     flux = {}".format(centre, peak, sigma, flux))
            except Exception:
                print("  Error trying to get the peak as requested wavelength is {} ! Ignoring this fit!".format(centre))
                peak = 0.0
                flux = -0.0001

        no_substract = False
        if flux < 0:
            if allow_absorptions == False:
                if verbose:
                    print("  WARNING! This is an ABSORPTION Gaussian! As requested, this Gaussian is NOT substracted!")
                    no_substract = True
        # print no_substract
        if no_substract == False:
            if verbose:
                print("  Substracting Gaussian at {:7.1f}  with peak ={:10.4f}   sigma ={:6.2f}  and flux ={:9.4f}".format(
                    centre, peak, sigma, flux
                ))

            gaussian_fit = gauss(w_spec, centre, peak, sigma)

            index = 0
            s_s = np.zeros_like(spectrum)
            for wave in range(len(wavelength)):
                s_s[wave] = spectrum[wave]
                if wavelength[wave] == w_spec[0]:
                    s_s[wave] = f_spec[0] - gaussian_fit[0]
                    index = 1
                if wavelength[wave] > w_spec[0] and wavelength[wave] <= w_spec[-1]:
                    s_s[wave] = f_spec[index] - gaussian_fit[index]
                    index = index + 1
            if plot:
                plt.figure(figsize=(10, 4))
                plt.plot(np.array(w_spec), np.array(f_spec), "b", lw=3, alpha=0.5)
                plt.minorticks_on()
                plt.xlabel("Wavelength [$\AA$]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
                else:
                    plt.ylabel("Flux [ counts ]")
                plt.xlim(lmin, lmax)
                plt.ylim(fmin, fmax)

                # Vertical line at line
                plt.axvline(x=centre, color="k", linestyle="-", alpha=0.8)
                # Horizontal line at y = 0
                plt.axhline(y=0, color="k", linestyle=":", alpha=0.5)
                # Dashed green regions for continuum, defined by [lowlow, lowhigh] and [highlow,highhigh]
                plt.axvspan(
                    centre + highlow,
                    centre + highhigh,
                    facecolor="g",
                    alpha=0.15,
                    zorder=3,
                )
                plt.axvspan(
                    centre - lowlow,
                    centre - lowhigh,
                    facecolor="g",
                    alpha=0.15,
                    zorder=3,
                )
                # Plot linear fit for continuum
                plt.plot(w_spec, continuum, "g--")
                # Plot Gaussian fit
                plt.plot(w_spec, gaussian_fit + continuum, "r-", alpha=0.8)
                # Vertical lines to emission line
                # plt.axvline(x= low_limit, color='k', linestyle=':', alpha=0.5)
                # plt.axvline(x= high_limit, color='k', linestyle=':', alpha=0.5)
                # Plot residuals
                # plt.plot(w_spec, residuals, 'k')
                # plt.title('Fit: x0=%.2f y0=%.2e sigma=%.2f  flux=%.2e  rms=%.3e' % (fit[0], fit[1], fit[2], gaussian_flux, rms_fit))
                # plt.show()
                # plt.close()

                plt.figure(figsize=(10, 4))
                plt.plot(wavelength, spectrum, "r")
                plt.plot(wavelength, s_s, "c")
                plt.minorticks_on()
                plt.xlabel("Wavelength [$\AA$]")
                if fcal:
                    plt.ylabel("Flux [ erg cm$^{-2}$ s$^{-1}$ A$^{-1}$]")
                else:
                    plt.ylabel("Flux [ counts ]")
                plt.xlim(lmin, lmax)
                plt.ylim(fmin, fmax)
                # plt.show()
                # plt.close()
        else:
            s_s = spectrum
    return s_s

