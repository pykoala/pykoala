# =============================================================================
# Basics packages
# =============================================================================
import copy
import numpy as np
from scipy.signal import medfilt
import sys
import matplotlib.pyplot as plt
import os
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy.table import Table
# =============================================================================
# KOALA packages
# =============================================================================
# Modular
from koala.ancillary import vprint
from koala.rss import detect_edge

# Original
from koala.onedspec import fluxes
from koala.onedspec import rebin_spec
from koala.onedspec import rebin_spec_shift
from koala.onedspec import fit_clip
from koala.plot_plot import plot_plot
# =============================================================================



def fix_wavelengths_edges(rss,  
                          sky_lines=[6300.309, 8430.147, 8465.374], # sky_lines =[6300.309, 7316.290, 8430.147, 8465.374],
                            # valid_ranges=[[-0.25,0.25],[-0.5,0.5],[-0.5,0.5]],
                            # valid_ranges=[[-0.4,0.3],[-0.4,0.45],[-0.5,0.5],[-0.5,0.5]], # ORIGINAL
                          valid_ranges=[[-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6]],
                          fit_order=2,
                          apply_median_filter=True,
                          kernel_median=51,
                          fibres_to_plot=[0, 100, 300, 500, 700, 850, 985],
                          show_fibres=[0, 500, 985],
                          plot_fits=False,
                          xmin=8450,
                          xmax=8475,
                          ymin=-10,
                          ymax=250,
                          check_throughput=False,
                          plot=True,
                          verbose=True,
                          warnings=True,
                          fig_size=12):
    """
    Using bright skylines, performs small wavelength corrections to each fibre
    
    Parameters:
    ----------
    sky_lines : list of floats (default = [6300.309, 8430.147, 8465.374])
        Chooses the sky lines to run calibration for
    valid_ranges : list of lists of floats (default = [[-1.2,0.6],[-1.2,0.6],[-1.2,0.6]])
        Ranges of flux offsets for each sky line
    fit_order : integer (default = 2)
        Order of polynomial for fitting
    apply_median_filter : boolean (default = True)
        Choose if we want to apply median filter to the image
    kernel_median : odd integer (default = 51)
        Length of the median filter interval
    fibres_to_plot : list of integers (default = [0,100,300,500,700,850,985])
        Choose specific fibres to visualise fitted offsets per wavelength using plots
    show_fibres : list of integers (default = [0,500,985])
        Plot the comparission between uncorrected and corrected flux per wavelength for specific fibres
    plot_fits : boolean (default = False)
        Plot the Gaussian fits for each iteration of fitting
    xmin, xmax, ymin, ymax : integers (default = 8450, 8475, -10, 250)
        Plot ranges for Gaussian fit plots, x = wavelength in Angstroms, y is flux in counts
    plot : boolean (default = True)
        Plot the resulting KOALA RSS image
    verbose : boolean (default = True)
        Print detailed description of steps being done on the image in the console as code runs
    warnings : boolean (default = True)
        Print the warnings in the console if something works incorrectly or might require attention 
    fig_size : integer (default = 12)
        Size of the image plotted          
    """
    
    # Set print verbose
    vprint.verbose = verbose
    
    
# =============================================================================
# Copy input RSS for storage the changes implemented in the task   
# =============================================================================
    rss_out = copy.deepcopy(rss)

    vprint("\n> Fixing wavelengths using skylines in edges")
    vprint("\n  Using skylines: ", sky_lines, "\n")



    # Find offsets using 6300.309 in the blue end and average of 8430.147, 8465.374 in red end
    wavelength = rss.wavelength
    nspec = rss.intensity.shape[0]

    sol_edges = []

    offset_sky_lines = []
    fitted_offset_sky_lines = []
    gauss_fluxes_sky_lines = []
    
    for sky_line in sky_lines:
        gauss_fluxes = []
        offset_ = []    
        x = []
    
    
        for i, flux in enumerate(rss.intensity_corrected):
            x.append(i * 1.)


            """
            if i in fibres_to_plot and plot_fits:
                plot_fit = True
            else:
                plot_fit = False
            if plot_fit: print(" - Plotting Gaussian fitting for skyline", sky_line, "in fibre", i, ":")
            """
        
            
            resultado = fluxes(wavelength, flux, sky_line, lowlow=80, lowhigh=20,
                               highlow=20, highhigh=80, broad=2.0,
                               fcal=False, plot=False, verbose=False)
            
            offset_.append(resultado[1])
            gauss_fluxes.append(resultado[3])
            
        offset = np.array(offset_) - sky_line  # offset_[500]
        offset_sky_lines.append(offset)

        offset_in_range = []
        x_in_range = []
        valid_range = valid_ranges[sky_lines.index(sky_line)]
        offset_m = medfilt(offset, kernel_median)
        text = ""
        
        
        if apply_median_filter:
            # xm = medfilt(x, odd_number)
            text = " applying a " + np.str(kernel_median) + " median filter"
            for i in range(len(offset_m)):
                if offset_m[i] > valid_range[0] and offset_m[i] < valid_range[1]:
                    offset_in_range.append(offset_m[i])
                    x_in_range.append(x[i])
        else:
            for i in range(len(offset)):
                if offset[i] > valid_range[0] and offset[i] < valid_range[1]:
                    offset_in_range.append(offset[i])
                    x_in_range.append(i)

        fit = np.polyfit(x_in_range, offset_in_range, fit_order)
        #fit, pp_, y_fit_, y_fit_, x_, y_ = fit_clip(x_in_range, offset_in_range, index_fit=fit_order, clip=0.4, kernel=1)
        if fit_order == 2:
            ptitle = "Fitting to skyline " + np.str(sky_line) + " : {:.3e} x$^2$  +  {:.3e} x  +  {:.3e} ".format(
                fit[0], fit[1], fit[2]) + text
        if fit_order == 1:
            ptitle = "Fitting to skyline " + np.str(sky_line) + " : {:.3e} x  +  {:.3e} ".format(fit[0],
                                                                                                 fit[1]) + text
        if fit_order > 2:
            ptitle = "Fitting an order " + np.str(fit_order) + " polinomium to skyline " + np.str(sky_line) + text

        y = np.poly1d(fit)
        fity = y(list(range(nspec)))
        fitted_offset_sky_lines.append(fity)
        sol_edges.append(fit)  

        if plot:
            plot_plot(x, [offset, offset_m, fity], ymin=valid_range[0], ymax=valid_range[1],
                      xlabel="Fibre", ylabel="$\Delta$ Offset", ptitle=ptitle)

        gauss_fluxes_sky_lines.append(gauss_fluxes)
    sky_lines_edges = [sky_lines[0], (sky_lines[-1] + sky_lines[-2]) / 2]

    nspec_vector = list(range(nspec))
    fitted_offset_sl_median = np.nanmedian(fitted_offset_sky_lines, axis=0)

    fitted_solutions = np.nanmedian(sol_edges, axis=0)
    y = np.poly1d(fitted_solutions)
    fitsol = y(list(range(nspec)))
    sol = [fitted_solutions[2], fitted_solutions[1], fitted_solutions[0]]
    
    vprint("\n> sol = [" + np.str(fitted_solutions[2]) + "," + np.str(fitted_solutions[1]) + "," + np.str(
        fitted_solutions[0]) + "]")
    
                
    # Add sol to the log
    rss_out.log['wavelenght fix']['sol'] = sol
            

    

    if plot:
        plot_plot(nspec_vector, [fitted_offset_sky_lines[0], fitted_offset_sky_lines[1], fitted_offset_sky_lines[2],
                  fitted_offset_sl_median, fitsol], color=["r", "orange", "b", "k", "g"],
                  alpha=[0.3, 0.3, 0.3, 0.5, 0.8],
                  hlines=[-0.75, -0.5, -0.25, 0, 0.25, 0.5],
                  label=[np.str(sky_lines[0]), np.str(sky_lines[1]), np.str(sky_lines[2]), "median", "median sol"],
                  ptitle="Checking fitting solutions",
                  ymin=-1, ymax=0.6, xlabel="Fibre", ylabel="Fitted offset")

    # Plot corrections
    if plot:
        plt.figure(figsize=(fig_size, fig_size / 2.5))
        for show_fibre in fibres_to_plot:
            offsets_fibre = [fitted_offset_sky_lines[0][show_fibre],
                             (fitted_offset_sky_lines[1][show_fibre] + fitted_offset_sky_lines[2][show_fibre]) / 2]
            plt.plot(sky_lines_edges, offsets_fibre, "+")
            plt.plot(sky_lines_edges, offsets_fibre, "--", label=np.str(show_fibre))
        plt.minorticks_on()
        plt.legend(frameon=False, ncol=9)
        plt.title("Small wavelength offsets per fibre")
        plt.xlabel("Wavelength [$\mathrm{\AA}$]")
        plt.ylabel("Fitted offset")
        plt.show()
        plt.close()

    
    # Apply corrections to all fibres
    # show_fibres=[0,500,985]  # plot only the spectrum of these fibres
    # intensity_wave_fixed = np.zeros_like(intensity)

    for fibre in range(nspec):  # show_fibres:
        offsets_fibre = [fitted_offset_sky_lines[0][fibre],
                         (fitted_offset_sky_lines[-1][fibre] + fitted_offset_sky_lines[-2][fibre]) / 2]
        fit_edges_offset = np.polyfit(sky_lines_edges, offsets_fibre, 1)
        y = np.poly1d(fit_edges_offset)
        w_offset = y(wavelength)
        w_fixed = wavelength - w_offset

        # Apply correction to fibre
        # intensity_wave_fixed[fibre] =rebin_spec(w_fixed, intensity[fibre], w)
        rss_out.intensity_corrected[fibre] = rebin_spec(w_fixed, rss.intensity_corrected[fibre],
                                                     wavelength)  # =copy.deepcopy(intensity_wave_fixed)
        """

        if fibre in show_fibres and plot:
            plt.figure(figsize=(fig_size, fig_size / 4.5))
            plt.plot(w, intensity[fibre], "r-", alpha=0.2, label="No corrected")
            plt.plot(w_fixed, intensity[fibre], "b-", alpha=0.2, label="No corrected - Shifted")
            plt.plot(w, self.intensity_corrected[fibre], "g-", label="Corrected after rebinning", alpha=0.6,
                     linewidth=2.)
            for line in sky_lines:
                plt.axvline(x=line, color="k", linestyle="--", alpha=0.3)
            # plt.xlim(6280,6320)
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.minorticks_on()
            ptitle = "Fibre " + np.str(fibre)
            plt.title(ptitle)
            plt.legend(frameon=False, ncol=3)
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            plt.ylabel("Flux")
            plt.show()
            plt.close()
        """
    vprint("\n> Small fixing of the wavelengths considering only the edges done!")
    
    #self.history.append("- Fixing wavelengths using skylines in the edges")
    #self.history.append("  sol (found) = " + np.str(self.sol))
    
    """
    if check_throughput:
        print("\n> As an extra, checking the Gaussian flux of the fitted skylines in all fibres:")

        vector_x = np.arange(nspec)
        vector_y = []
        label_skylines = []
        alpha = []
        for i in range(len(sky_lines)):
            med_gaussian_flux = np.nanmedian(gauss_fluxes_sky_lines[i])
            vector_y.append(gauss_fluxes_sky_lines[i] / med_gaussian_flux)
            label_skylines.append(np.str(sky_lines[i]))
            alpha.append(0.3)
            # print "  - For line ",sky_lines[i],"the median flux is",med_gaussian_flux

        vector_y.append(np.nanmedian(vector_y, axis=0))
        label_skylines.append("Median")
        alpha.append(0.5)

        for i in range(len(sky_lines)):
            ptitle = "Checking Gaussian flux of skyline " + label_skylines[i]
            plot_plot(vector_x, vector_y[i],
                      label=label_skylines[i],
                      hlines=[0.8, 0.9, 1.0, 1.1, 1.2], ylabel="Flux / Median flux", xlabel="Fibre",
                      ymin=0.7, ymax=1.3, ptitle=ptitle)

        ptitle = "Checking Gaussian flux of the fitted skylines (this should be all 1.0 in skies)"
        #        plot_plot(vector_x,vector_y,label=label_skylines,hlines=[0.9,1.0,1.1],ylabel="Flux / Median flux", xlabel="Fibre",
        #                  ymin=0.7,ymax=1.3, alpha=alpha,ptitle=ptitle)
        plot_plot(vector_x, vector_y[:-1], label=label_skylines[:-1], alpha=alpha[:-1],
                  hlines=[0.8, 0.9, 1.0, 1.1, 1.2], ylabel="Flux / Median flux", xlabel="Fibre",
                  ymin=0.7, ymax=1.3, ptitle=ptitle)

        vlines = []
        for j in vector_x:
            if vector_y[-1][j] > 1.1 or vector_y[-1][j] < 0.9:
                # print "  Fibre ",j,"  ratio value = ", vector_y[-1][j]
                vlines.append(j)
        print("\n  TOTAL = ", len(vlines), " fibres with flux differences > 10 % !!")

        plot_plot(vector_x, vector_y[-1], label=label_skylines[-1], alpha=1, vlines=vlines,
                  hlines=[0.8, 0.9, 1.0, 1.1, 1.2], ylabel="Flux / Median flux", xlabel="Fibre",
                  ymin=0.7, ymax=1.3, ptitle=ptitle)

        # CHECKING SOMETHING...
        self.throughput_extra_checking_skylines = vector_y[-1]

#        for i in range(self.n_spectra):
#            if i != 546 or i != 547:
#                self.intensity_corrected[i] = self.intensity_corrected[i] / vector_y[-1][i]


    """
    
    
    # Add solutions parameter to the header    
    rss_out.header["SOL0"] = sol[0]
    rss_out.header["SOL1"] = sol[1]
    rss_out.header["SOL2"] = sol[2]


    
    return rss_out




# -----------------------------------------------------------------------------
# Idea: take a RSS dominated by skylines. Read it (only throughput correction). For each fibre, fit Gaussians to ~10 skylines. 
# Compare with REST wavelengths. Get a median value per fibre. Perform a second-order fit to all median values.
# Correct for that using a reference fibre (1). Save results to be applied to the rest of files of the night (assuming same configuration).

def fix_wavelengths(rss,
                    grating,
                    sol=None,
                    fibre='all',
                    edges=False,
                    maxima_sigma=2.5,
                    maxima_offset=1.5,
                    sky_lines_file=None,
                    index_fit = 2,
                    kernel_fit= 19,
                    clip_fit =0.4,
                    # xmin=7740,xmax=7770, ymin="", ymax="",
                    xmin=[6270, 8315],
                    xmax=[6330, 8375],
                    ymax="",
                    fibres_to_plot=[0, 100, 400, 600, 950],
                    plot=True,
                    plot_all=False,
                    verbose=True,
                    warnings=True):
    """
    Using bright skylines, performs small wavelength corrections to each fibre
            
    Parameters:
    ----------
    sol : list of floats (default = [0,0,0])
        Specify the parameters of the second degree polynomial fit
    fibre : integer (default = -1)
        Choose a specific fibre to run correction for. If not specified, all fibres will be corrected
    maxima_sigma : float (default = 2.5)
        Maximum allowed standard deviation for Gaussian fit
    maxima_offset : float (default 1.5)
        Maximum allowed wavelength offset in Angstroms      
    xmin : list of integer (default = [6270, 8315])
        Minimum wavelength values in Angstrom for plots before and after correction
    xmax : list of integer (default = [6330, 8375])
        Maximum wavelength values in Angstrom for plots before and after correction                
    ymax : float (default = none)
        Maximum y value to be plot, if not given, will estimate it automatically
    fibres_to_plot : list of integers (default = [0,100,400,600,950])
        Plot the comparission between uncorrected and corrected flux per wavelength for specific 
    plot : boolean (default = True)
        Plot the plots
    plot_all : boolean (default = False)
        Plot the Gaussian fits for each iteration of fitting
    verbose : boolean (default = True)
        Print detailed description of steps being done on the image in the console as code runs
    warnings : boolean (default = True)
        Print the warnings in the console if something works incorrectly or might require attention        
    """
   
    # Set print verbose
    vprint.verbose = verbose

# =============================================================================
# Copy input RSS for storage the changes implemented in the task   
# =============================================================================
    rss_out = copy.deepcopy(rss)
   
    vprint("\n> Fixing wavelengths using skylines...")
    
    if grating == "580V":
        xmin = [5555]
        xmax = [5600]
        if sol is not None and sol[2] == 0:
            print("  Only using a Gaussian fit to the 5577 emission line...")
            #self.history.append("- Fixing wavelengths using Gaussian fits to skyline 5577")
            index_fit = 1
    #else:
    #    self.history.append("- Fixing wavelengths using Gaussian fits to bright skylines")

    wavelength = rss.wavelength
    n_spectra = rss.intensity.shape[0]
    xfibre = list(range(0, n_spectra))
    
    plot_this_again = True



    
# =============================================================================
# Solutions are not given
# =============================================================================
    if sol is None:  
        if sky_lines_file is None:
            # Read file with sky emission line
            
            dirname = os.path.dirname(__file__)            
            sky_lines_file = os.path.join(dirname, "./input_data/sky_lines/sky_lines_rest.dat")

        
        # Read the sky lines table
        sky_line = Table.read(sky_lines_file,format='ascii.commented_header')
        

        # Be sure the lines we are using are in the requested wavelength range        
        # if fibre != -1:
        vprint("\n  Checking the values of skylines in the file", sky_lines_file,'\n')
        if verbose:
            print('\n')
            sky_line['center', 'fnl','lowlow','lowhigh','highlow','highhigh','lmin','lmax'].pprint()  
        
        
        # Valid wavelenght range
        valid_wave_min, min_index, valid_wave_max, max_index = detect_edge(rss)
        
        vprint("\n  We only need skylines in the {:.2f} - {:.2f} range".format(np.round(valid_wave_min, 2),
                                                                            np.round(valid_wave_max, 2)))
        
        valid_range = (sky_line['center'] > valid_wave_min) & (sky_line['center'] < valid_wave_max)
        
        sky_line = sky_line[valid_range]
    
        number_sl = len(sky_line)
        
        if fibre != 'all': print(" ", sky_line['center'])


        # Fitting Gaussians to skylines...         
        vprint("\n> Performing a Gaussian fit to selected, bright skylines...")
        vprint("  (this might FAIL if RSS is NOT corrected for CCD defects...)")

        wave_median_offset = []

        if fibre != 'all':
            f_i = fibre
            f_f = fibre + 1
            vprint("  Checking fibre ", fibre," (only this fibre is corrected, use fibre = -1 for all)...")
            
            verbose_ = True
            warnings = True
            plot_all = True
        
        else:
            f_i = 0
            f_f = n_spectra
            verbose_ = False

        number_fibres_to_check = len(list(range(f_i, f_f)))
        output_every_few = np.sqrt(len(list(range(f_i, f_f)))) + 1
        next_output = -1
        
        
        #TODO: For improving SNR and make it faster, do it in jumps of few fibres (~10 or so)
        for fibre in range(f_i, f_f):  # (self.n_spectra):
            spectrum = rss.intensity_corrected[fibre]
            if verbose:
                if fibre > next_output:
                    sys.stdout.write("\b" * 51)
                    sys.stdout.write("  Checking fibre {:4} ...  ({:6.2f} % completed) ...".format(fibre,
                                                                                                    fibre * 100. / number_fibres_to_check))
                    sys.stdout.flush()
                    next_output = fibre + output_every_few

                    # Gaussian fits to the sky spectrum
            sl_gaussian_flux = []
            sl_gaussian_sigma = []
            sl_gauss_center = []
            sl_offset = []
            sl_offset_good = []

            for i in range(number_sl):
                if sky_line['fnl'][i] == 0:
                    plot_fit = False
                else:
                    plot_fit = True
                if plot_all: plot_fit = True

                resultado = fluxes(wavelength, spectrum, sky_line['center'][i], lowlow=sky_line['lowlow'][i], lowhigh=sky_line['lowhigh'][i],
                                    highlow=sky_line['highlow'][i], highhigh=sky_line['highhigh'][i], lmin=sky_line['lmin'][i], lmax=sky_line['lmax'][i],
                                    fmin=0, fmax=0,
                                    broad=2.1 * 2.355, plot=plot_fit, verbose=False, plot_sus=False, fcal=False,
                                    warnings=warnings)  # Broad is FWHM for Gaussian sigm a= 1,

                sl_gaussian_flux.append(resultado[3])
                sl_gauss_center.append(resultado[1])
                sl_gaussian_sigma.append(resultado[5] / 2.355)
                sl_offset.append(sl_gauss_center[i] - sky_line['center'][i])

                if sl_gaussian_flux[i] < 0 or np.abs(sky_line['center'][i] - sl_gauss_center[i]) > maxima_offset or \
                        sl_gaussian_sigma[i] > maxima_sigma:
                    if verbose_: print("  Bad fitting for ", sky_line['center'][i], "... ignoring this fit...")
                else:
                    sl_offset_good.append(sl_offset[i])
                    if verbose_: print(
                        "    Fitted wavelength for sky line {:8.3f}:    center = {:8.3f}     sigma = {:6.3f}    offset = {:7.3f} ".format(
                            sky_line['center'][i], sl_gauss_center[i], sl_gaussian_sigma[i], sl_offset[i]))

            median_offset_fibre = np.nanmedian(sl_offset_good)
            wave_median_offset.append(median_offset_fibre)
            if verbose_: print("\n> Median offset for fibre {:3} = {:7.3f}".format(fibre, median_offset_fibre))

        if verbose:
            sys.stdout.write("\b" * 51)
            sys.stdout.write("  Checking fibres completed!                  ")
            sys.stdout.flush()
            print(" ")

        # Second-order fit ...         
        bad_numbers = 0
        try:
            xfibre_ = []
            wave_median_offset_ = []
            for i in xfibre:
                if np.isnan(wave_median_offset[i]) == True:
                    bad_numbers = bad_numbers + 1
                else:
                    if wave_median_offset[i] == 0:
                        bad_numbers = bad_numbers + 1
                    else:
                        xfibre_.append(i)
                        wave_median_offset_.append(wave_median_offset[i])
            if bad_numbers > 0 and verbose: print("\n> Skipping {} bad points for the fit...".format(bad_numbers))
            
            fit, pp, fx_, y_fit_c, x_c, y_c  = fit_clip(xfibre_, wave_median_offset_, clip=clip_fit, plot=plot, 
                                                        xlabel="Fibre",ylabel="offset",xmin=xfibre_[0]-20,xmax=xfibre_[-1]+20,
                                                        percentile_max = 99.2, percentile_min=0.8,
                                                        index_fit = index_fit, kernel = kernel_fit, hlines=[0])
          
            if index_fit == 1:
                sol = [fit[1], fit[0], 0]
                ptitle = "Linear fit to individual offsets"
                if verbose: print("\n> Fitting a linear polynomy a0x +  a1x * fibre:")
            else:
                sol = [fit[2], fit[1], fit[0]]
                ptitle = "Second-order fit to individual offsets"
                if verbose: 
                    if index_fit != 2 and verbose : print("  A fit of order", index_fit,"was requested, but this tasks only runs with orders 1 or 2.")
                    print("\n> Fitting a second-order polynomy a0x +  a1x * fibre + a2x * fibre**2:")
            
            # Add sol to the log
            rss_out.log['wavelenght fix']['sol'] = sol
            
            if plot: plot_this_again = False
                
            #self.history.append("  sol (found) = " + np.str(sol))
        except Exception:
            if warnings:
                print("\n> Something failed doing the fit...")
                print("  These are the data:")
                print(" - xfibre =", xfibre_)
                print(" - wave_median_offset = ", wave_median_offset_)
                #plot_plot(xfibre_, wave_median_offset_)
                #ptitle = "This plot may don't have any sense..."
                
                
# =============================================================================
# Solution provided in sol = [a0,a1,a2]                
# =============================================================================
    else:
        vprint("\n> Solution to the second-order polynomy a0x +  a1x * fibre + a2x * fibre**2 has been provided:")
        # a0x = sol[0]
        # a1x = sol[1]
        # a2x = sol[2]
        #ptitle = "Second-order polynomy provided"
        #self.history.append("  sol (provided) = " + np.str(sol))

    vprint("  a0x =", sol[0], "   a1x =", sol[1], "     a2x =", sol[2])
    vprint("\n> sol = [{},{},{}]".format(sol[0], sol[1], sol[2]))


# =============================================================================
# Fitted polinom
# =============================================================================

    fx = sol[0] + sol[1] * np.array(xfibre) + sol[2] * np.array(xfibre) ** 2

    if plot:
        if sol[0] == 0:
            pf = wave_median_offset
        else:
            pf = fx
        if plot_this_again:
            if index_fit == 1: 
                ptitle = "Linear fit to individual offsets"
            else:
                ptitle = "Second-order fit to individual offsets"         
            plot_plot(xfibre, [fx, pf], ptitle=ptitle, color=['red', 'blue'], xmin=-20, xmax=1000, xlabel="Fibre",
                      ylabel="offset", hlines=[0])
    # Applying results
    vprint("\n> Applying results to all fibres...")
    for fibre in xfibre:
        f = rss.intensity_corrected[fibre]
        w_shift = fx[fibre]
        rss_out.intensity_corrected[fibre] = rebin_spec_shift(wavelength, f, w_shift)

    if plot:
        # Check results
        vprint("\n> Plotting some results after fixing wavelengths: ")

        for line in range(len(xmin)):

            xmin_ = xmin[line]
            xmax_ = xmax[line]

            plot_y = []
            plot_y_corrected = []
            ptitle = "Before corrections, fibres "
            ptitle_corrected = "After wavelength correction, fibres "
            if ymax == "": y_max_list = []
            for fibre in fibres_to_plot:
                plot_y.append(rss.intensity[fibre])
                plot_y_corrected.append(rss.intensity_corrected[fibre])
                ptitle = ptitle + np.str(fibre) + " "
                ptitle_corrected = ptitle_corrected + np.str(fibre) + " "
                if ymax == "":
                    y_max_ = []
                    y_max_.extend(
                        (rss.intensity[fibre, i]) for i in range(len(wavelength)) if (wavelength[i] > xmin_ and wavelength[i] < xmax_))
                    y_max_list.append(np.nanmax(y_max_))
            if ymax == "": ymax = np.nanmax(y_max_list) + 20  # TIGRE
            plot_plot(wavelength, plot_y, ptitle=ptitle, xmin=xmin_, xmax=xmax_, percentile_min=0.1,
                      ymax=ymax)  # ymin=ymin, ymax=ymax)
            plot_plot(wavelength, plot_y_corrected, ptitle=ptitle_corrected, xmin=xmin_, xmax=xmax_, percentile_min=0.1,
                      ymax=ymax)  # ymin=ymin, ymax=ymax)
            y_max_list = []
            ymax = ""
    vprint("\n> Small fixing of the wavelengths done!")
    
    return rss_out





def compare_fix_wavelengths(rss1, 
                            rss2,
                            verbose=False):
    
    # Set print verbose
    vprint.verbose = verbose
    vprint("\n> Comparing small fixing of wavelengths between two rss...")
    
    n_spectra = rss1.intensity.shape[0]
    
    xfibre = list(range(0, n_spectra))

    a0x, a1x, a2x = rss1.log['wavelenght fix']['sol']

    aa0x, aa1x, aa2x = rss2.log['wavelenght fix']['sol']


    fx = a0x + a1x * np.array(xfibre) + a2x * np.array(xfibre) ** 2
    fx2 = aa0x + aa1x * np.array(xfibre) + aa2x * np.array(xfibre) ** 2
    dif = fx - fx2

    plot_plot(xfibre, dif, ptitle="Fit 1 - Fit 2", xmin=-20, xmax=1000, xlabel="Fibre", ylabel="Dif")

    resolution = rss1.wavelength[1] - rss1.wavelength[0]
    error = np.nanmedian(dif) / resolution * 100.
    vprint("\n> The median rms is {:8.6f} A,  resolution = {:5.2f} A,  error = {:5.3} %".format(np.nanmedian(dif),
                                                                                                resolution, error))
