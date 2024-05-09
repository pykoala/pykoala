# =============================================================================
# Basics packages
# =============================================================================
from os import path
import numpy as np
import copy
import sys
#from astropy.io import fits
#from scipy.ndimage import median_filter
#from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.signal import medfilt
# =============================================================================
# Astropy and associated packages
# =============================================================================

# =============================================================================
# KOALA packages
# =============================================================================
# Modular
from pykoala.corrections.correction import CorrectionBase
#from pykoala.rss import RSS
from pykoala.rss import rss_valid_wave_range
#from pykoala import ancillary
from pykoala.ancillary import vprint, read_table
#from pykoala.plotting.rss_plot import rss_image
from pykoala.plotting.plot_plot import plot_plot, basic_statistics
from pykoala.spectra.onedspec import fluxes, rebin_spec_shift, fit_clip   #rebin_spec, 

# class Wavelength(object):
#     def __init__(self, 
#                  sky_lines = None,
#                  wavelength_shift_solution=None, 
#                  sol_sky_lines = None,
#                  fitted_offset_sky_lines = None,
#                  w_fixed_all_fibres =None):
#                  #path=None, wavelength_data=None, wavelength_error=None):
                
        
#         self.sky_lines = sky_lines
#         self.wavelength_shift_solution = wavelength_shift_solution
#         self.sol_sky_lines = sol_sky_lines
#         self.fitted_offset_sky_lines = fitted_offset_sky_lines
#         self.w_fixed_all_fibres = w_fixed_all_fibres
                
#         # self.path = path
#         # self.wavelength_data = wavelength_data
#         # self.wavelength_error = wavelength_error

#         # if self.path is not None and self.wavelength_data is None:
#         #     self.load_fits()
    
#     # def tofits(self, output_path):
#     #     primary = fits.PrimaryHDU()
#     #     thr = fits.ImageHDU(data=self.wavelength_data,
#     #                         name='WAVE')
#     #     thr_err = fits.ImageHDU(data=self.wavelength_error,
#     #                         name='WAVEERR')
#     #     hdul = fits.HDUList([primary, thr, thr_err])
#     #     hdul.writeto(output_path, overwrite=True)
#     #     hdul.close(verbose=True)
#     #     print(f"[Wavelength] Wavelength corrections saved at {output_path}")

#     # def load_fits(self):
#     #     """Load the throughput data from a fits file.
        
#     #     Description
#     #     -----------
#     #     Loads throughput values (extension 1) and associated errors (extension 2) from a fits
#     #     file.
#     #     """
#     #     if not path.isfile(self.path):
#     #         raise NameError(f"Throughput file {self.path} does not exists.")
#     #     print(f"[Throughput] Loading throughput from {self.path}")
#     #     with fits.open(self.path) as hdul:
#     #         self.wavelength_data = hdul[1].data
#     #        self.wavelength_error = hdul[2].data

class WavelengthShiftCorrection(CorrectionBase):
    """
    Wavelength shiftcorrection class.

    This class accounts for the small correction in wavevelength shifts.

    Attributes
    ----------
    - name
    -
    """
    name = "WavelengthShiftCorrection"
    wavelength_shift_solution = None
    verbose = False

    def __init__(self, 
                 sky_lines = None,
                 wavelength_shift_solution=None, 
                 sol_sky_lines = None,
                 fitted_offset_sky_lines = None,
                 w_fixed_all_fibres =None,
                 **kwargs):
        
        self.sky_lines = sky_lines
        self.wavelength_shift_solution = wavelength_shift_solution
        self.sol_sky_lines = sol_sky_lines
        self.fitted_offset_sky_lines = fitted_offset_sky_lines
        self.w_fixed_all_fibres = w_fixed_all_fibres
        
        self.verbose=kwargs.get("verbose", False)
        self.corr_print("Initialising wavelength correction model.")
        
        
        #super().__init__()
        

        #self.wavelength_shift_solution = kwargs.get('wavelength_shift_solution', Wavelength())
        #if type(self.throughput) is not Throughput():   #!!! ANGEL    Así me falla, tengo que llamar a str
        #if str(type(self.throughput)) != str(Throughput()):   ## TAMBIEN FALLA ????
        #    raise AttributeError("Input throughput must be an instance of Throughput class")

        # self.wavelength.path = kwargs.get('wavelength_path', None)
        # if self.wavelength.wavelength_shift_solution is None and self.wavelength.path is not None:
        #     self.wavelength.load_fits(self.wavelength_path)
# #-----------------------------------------------------------------------------
# %% ============================================================================
# #-----------------------------------------------------------------------------
    def wavelength_shift_using_edges(rss,                    # OLD, better use the other below
                                     sky_lines=None, 
                                     median_fibres = 5,
                                     fit_order=2, apply_median_filter=True, kernel_median=51,
                                     fibres_to_plot=None, #[0, 100, 300, 500, 700, 850, 985],
                                     show_fibres=None,
                                     #xmin=8450, xmax=8475, ymin=-10, ymax=250,
                                     #check_throughput=False,  # TODO Needs to be implemented?
                                     #verbose = False,
                                     **kwargs):
                                     #plot=True, verbose=True, warnings=True, fig_size=12):
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
        
        if sky_lines == None:
            # This depends of wavelength range & resolution (grating) #TODO
            # For KOALA 385R grating, Find offsets using 6300.309 in the blue end and average of 8430.147, 8465.374 in red end
            #sky_lines = [6300.309, 8430.147, 8465.374]
            sky_lines = [6300.309, 8430.147] #, 8465.374]  # only 2
            #if self.grating == "2000R": sky_lines = [6498.737, 6553.626, 6863.971]
        
        if fibres_to_plot is None:
            fibres_to_plot = [0, np.int(np.percentile(range(len(rss.intensity)),17)), np.int(np.percentile(range(len(rss.intensity)),34)),
                              np.int(np.percentile(range(len(rss.intensity)),50)),
                              np.int(np.percentile(range(len(rss.intensity)),67)),np.int(np.percentile(range(len(rss.intensity)),84)),                              
                              len(rss.intensity)-1]
        
        vprint("> Fixing wavelengths using skylines in edges...",  **kwargs)
     
        w = rss.wavelength
        nspec = len(rss.intensity)
        
        sol_sky_lines = dict()
        sol_edges = []
    
        offset_sky_lines = []
        fitted_offset_sky_lines = []
        gauss_fluxes_sky_lines = []
        valid_shift_range_lines_min=[]
        valid_shift_range_lines_max=[]
        
        if median_fibres == None:
            intensity = rss.intensity
            xfibre = list(range(nspec))
            vprint("  Performing Gaussian fitting to skylines", sky_lines, "in all spectra...", **kwargs)
        else:
            apply_median_filter = False
            intensity=[]
            xfibre=[]
            combined_fibres=[]
            last_fibre=0
            for i in range(0, nspec-median_fibres, median_fibres):
                intensity.append(np.nanmedian(rss.intensity[i:i+median_fibres], axis=0))
                xfibre.append(i+int(median_fibres/2))
                combined_fibres.append(range(i,i+median_fibres))
                last_fibre= i+median_fibres
                #print(xfibre[-1], combined_fibres[-1], last_fibre)
            intensity.append(np.nanmedian(rss.intensity[last_fibre:nspec], axis=0))  
            xfibre.append(int(np.nanmedian(range(last_fibre,nspec))))
            combined_fibres.append(range(last_fibre,nspec))
            #print(xfibre[-1], combined_fibres[-1], last_fibre)
            vprint(f"  Performing Gaussian fitting to skylines {sky_lines} in spectra, median = {median_fibres} (total = {str(len(xfibre))} spectra)...", **kwargs)
                
        for sky_line in sky_lines:
            gauss_fluxes = []
            #x = []
            offset_ = []
            for i in range(len(xfibre)): #range(nspec):
                
                #x.append(i * 1.)
                f = intensity[i] # f = rss.intensity[i]
                plot_fit = False
                if show_fibres is not None and kwargs.get("plot"): #and i in show_fibres:                        
                    if median_fibres == None and i in show_fibres:
                        vprint("  - Plotting Gaussian fitting for skyline", sky_line, "in fibre", i, ":", **kwargs)
                        plot_fit = True
                    elif median_fibres is not None:
                        if [fibre in show_fibres for fibre in combined_fibres[i]].count(True) > 0:
                            vprint("  - Plotting Gaussian fitting for skyline", sky_line, "in fibre interval", combined_fibres[i][0],"-", combined_fibres[i][-1],":", **kwargs)
                            plot_fit = True
                resultado = fluxes(w, f, sky_line, lowlow=80, lowhigh=20, highlow=20, highhigh=80, broad=2.0,
                                   fcal=False, plot=plot_fit, verbose=False)
                offset_.append(resultado[1])
                gauss_fluxes.append(resultado[3])
            offset = np.array(offset_) - sky_line 
            offset_sky_lines.append(offset)
    
            
            offset_in_range = []
            x_in_range = []
            valid_shift_min = np.min([np.nanmedian(offset[0:30]), np.nanmedian(offset[-30:])])
            valid_shift_max = np.max([np.nanmedian(offset[0:30]), np.nanmedian(offset[-30:])])
            interval = np.abs(np.nanmedian(offset[0:30]) - np.nanmedian(offset[-30:]) )
            valid_shift_range=[valid_shift_min-interval/1.5,valid_shift_max+interval/1.5]
            valid_shift_range_lines_min.append(valid_shift_range[0])
            valid_shift_range_lines_max.append(valid_shift_range[1])

            
            text = ""
            if apply_median_filter:
                offset_m = medfilt(offset, kernel_median)
                text = "\n applying a " + str(kernel_median) + " median filter"
                for i in range(len(offset_m)):
                    if offset_m[i] > valid_shift_range[0] and offset_m[i] < valid_shift_range[1]:
                        offset_in_range.append(offset_m[i])
                        x_in_range.append(xfibre[i])
            else:
                for i in range(len(offset)):
                    if offset[i] > valid_shift_range[0] and offset[i] < valid_shift_range[1]:
                        offset_in_range.append(offset[i])
                        x_in_range.append(xfibre[i])
       
            fit = np.polyfit(x_in_range, offset_in_range, fit_order)
            if fit_order == 2:
                ptitle = "Fitting to skyline " + str(sky_line) + " : {:.3e} x$^2$  +  {:.3e} x  +  {:.3e} ".format(
                    fit[0], fit[1], fit[2]) + text
            if fit_order == 1:
                ptitle = "Fitting to skyline " + str(sky_line) + " : {:.3e} x  +  {:.3e} ".format(fit[0],
                                                                                                     fit[1]) + text
            if fit_order > 2:
                ptitle = "Fitting an order " + str(fit_order) + " polinomium to skyline " + str(sky_line) + text
    
            y = np.poly1d(fit)
            fity = y(list(range(nspec)))
            fitted_offset_sky_lines.append(fity)
            sol_edges.append(fit)  
            sol_sky_lines[str(sky_line)] = fit
    
            if kwargs.get("plot"): 
                if apply_median_filter:
                    y_plot = [offset, offset_m, fity]
                    x_plot = xfibre
                else:
                    y_plot = [offset, fity]
                    x_plot = [xfibre, range(nspec)]
                    
                plot_plot(x_plot, y_plot, ymin=valid_shift_range[0], ymax=valid_shift_range[1],
                          xlabel="Fibre", ylabel="$\Delta$ Offset [$\mathrm{\AA}$]", ptitle=ptitle)
    
            gauss_fluxes_sky_lines.append(gauss_fluxes)   
            # TODO: check throughtput using Gauss fitting and add this:
            #sol_sky_lines[str(sky_line)+"_gauss_flux"] = gauss_fluxes
        
        #sky_lines_edges = [sky_lines[0], (sky_lines[-1] + sky_lines[-2]) / 2]    # FOR RED 385R in KOALA, average 2 lines in red, perhaps good idea implementing it
        
        fitted_offset_sl_median = np.nanmedian(fitted_offset_sky_lines, axis=0)
        fitted_solutions = np.nanmedian(sol_edges, axis=0)
        y = np.poly1d(fitted_solutions)
        fitsol = y(list(range(nspec)))
        wavelength_shift_solution = [fitted_solutions[2], fitted_solutions[1], fitted_solutions[0]]   # THIS WAS sol  / rss.sol

        vprint("\n> sol = [" + str(fitted_solutions[2]) + "," + str(fitted_solutions[1]) + "," + str(
            fitted_solutions[0]) + "]", **kwargs)
        vprint("  offset_min = {:.3f} A ,  offset_max = {:.3f} A,  offset_difference = {:.3f} A".format(fitsol[0], fitsol[-1],np.abs(fitsol[0]-fitsol[-1])), **kwargs)
    
        if kwargs.get("plot"): 
            ymin_ = np.min([fitsol[0], fitsol[-1]])
            ymax_ = np.max([fitsol[0], fitsol[-1]])            
            interval = np.abs(ymax_-ymin_)
            _y_ = [fitted_offset_sky_lines[i] for i in range(len(sky_lines)) ]
            if len(sky_lines)> 2: _y_.append(fitted_offset_sl_median)
            _y_.append(fitsol)
            label_lines = [str(sky_lines[i])  for i in range(len(sky_lines))  ]
            if len(sky_lines)> 2: label_lines.append("median")
            label_lines.append("MEDIAN SOL")
            linestyle_lines =["--"  for i in range(len(sky_lines))]
            linestyle_lines.append("-")
            if len(sky_lines)> 2: linestyle_lines.append("-")
            if len(sky_lines) == 4: color_lines = ["r", "orange", "b", "purple", "k", "g"]
            if len(sky_lines) == 3: color_lines = ["r", "orange", "b", "k", "g"]
            if len(sky_lines) == 2: color_lines = ["r", "orange", "g"]
            if len(sky_lines) > 4 :
                color_lines = [None]*len(sky_lines)
                color_lines.append("k")
                color_lines.append("g")
            alpha_lines = [0.3] * len(sky_lines)
            if len(sky_lines)> 2: alpha_lines.append(0.5)
            alpha_lines.append(0.8)
            nspec_vector = list(range(nspec))
            plot_plot(nspec_vector, _y_, #[fitted_offset_sky_lines[0], fitted_offset_sky_lines[1], fitted_offset_sky_lines[2],fitted_offset_sl_median, fitsol], 
                      color= color_lines, #["r", "orange", "b", "k", "g"],
                      alpha = alpha_lines, #[0.3, 0.3, 0.3, 0.5, 0.8],
                      linestyle = linestyle_lines,
                      hlines=[-1.5,-1,-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5],
                      label=label_lines, #[str(sky_lines[0]), str(sky_lines[1]), str(sky_lines[2]), "median", "median sol"],
                      ptitle="Checking fitting solutions",
                      #ymin = np.nanmin(valid_shift_range_lines_min), ymax = np.nanmax(valid_shift_range_lines_max),
                      ymin = ymin_ -interval/6, ymax = ymax_ +interval/6, 
                      #ymin=-1, ymax=0.6, 
                      xlabel="Fibre", ylabel="Fitted offset [$\mathrm{\AA}$]")
    
        # Plot corrections
        if kwargs.get("plot"): 
            fig_size=kwargs.get("fig_size", 9)
            plt.figure(figsize=(fig_size, fig_size / 2.5))
            for show_fibre in fibres_to_plot:       
                offsets_fibre  = [fitted_offset_sky_lines[i][show_fibre] for i in range(len(sky_lines)) ]
                plt.plot(sky_lines, offsets_fibre, "+")
                plt.plot(sky_lines, offsets_fibre, "--", label=str(show_fibre))
                # plt.plot(sky_lines_edges, offsets_fibre, "+")
                # plt.plot(sky_lines_edges, offsets_fibre, "--", label=str(show_fibre))
                #offsets_fibre = [fitted_offset_sky_lines[0][show_fibre],
                #                 (fitted_offset_sky_lines[1][show_fibre] + fitted_offset_sky_lines[2][show_fibre]) / 2]
                
            plt.minorticks_on()
            plt.legend(frameon=False, ncol=9)
            plt.title("Small wavelength offsets per fibre")
            plt.xlabel("Wavelength [$\mathrm{\AA}$]")
            plt.ylabel("Fitted offset [$\mathrm{\AA}$]")
            plt.show()
            plt.close()
        
        # w_fixed_all_fibres =[]   NOT NEEDED
        # for fibre in range(nspec):
        #     # offsets_fibre = [fitted_offset_sky_lines[0][fibre],
        #     #                      (fitted_offset_sky_lines[-1][fibre] + fitted_offset_sky_lines[-2][fibre]) / 2]
        #     # fit_edges_offset = np.polyfit(sky_lines_edges, offsets_fibre, 1)
        #     offsets_fibre  = [fitted_offset_sky_lines[i][fibre] for i in range(len(sky_lines)) ]
        #     fit_edges_offset = np.polyfit(sky_lines, offsets_fibre, 1)
        #     y = np.poly1d(fit_edges_offset)
        #     w_fixed = w - y(w)
        #     w_fixed_all_fibres.append(w_fixed)
        
        #     print("\n> Small fixing of the wavelengths considering only the edges done!")
#     self.history.append("- Fixing wavelengths using skylines in the edges")
#     self.history.append("  sol (found) = " + str(self.sol))

#     if check_throughput:
#         print("\n> As an extra, checking the Gaussian flux of the fitted skylines in all fibres:")

#         vector_x = np.arange(nspec)
#         vector_y = []
#         label_skylines = []
#         alpha = []
#         for i in range(len(sky_lines)):
#             med_gaussian_flux = np.nanmedian(gauss_fluxes_sky_lines[i])
#             vector_y.append(gauss_fluxes_sky_lines[i] / med_gaussian_flux)
#             label_skylines.append(str(sky_lines[i]))
#             alpha.append(0.3)
#             # print "  - For line ",sky_lines[i],"the median flux is",med_gaussian_flux

#         vector_y.append(np.nanmedian(vector_y, axis=0))
#         label_skylines.append("Median")
#         alpha.append(0.5)

#         for i in range(len(sky_lines)):
#             ptitle = "Checking Gaussian flux of skyline " + label_skylines[i]
#             plot_plot(vector_x, vector_y[i],
#                       label=label_skylines[i],
#                       hlines=[0.8, 0.9, 1.0, 1.1, 1.2], ylabel="Flux / Median flux", xlabel="Fibre",
#                       ymin=0.7, ymax=1.3, ptitle=ptitle)

#         ptitle = "Checking Gaussian flux of the fitted skylines (this should be all 1.0 in skies)"
#         #        plot_plot(vector_x,vector_y,label=label_skylines,hlines=[0.9,1.0,1.1],ylabel="Flux / Median flux", xlabel="Fibre",
#         #                  ymin=0.7,ymax=1.3, alpha=alpha,ptitle=ptitle)
#         plot_plot(vector_x, vector_y[:-1], label=label_skylines[:-1], alpha=alpha[:-1],
#                   hlines=[0.8, 0.9, 1.0, 1.1, 1.2], ylabel="Flux / Median flux", xlabel="Fibre",
#                   ymin=0.7, ymax=1.3, ptitle=ptitle)

#         vlines = []
#         for j in vector_x:
#             if vector_y[-1][j] > 1.1 or vector_y[-1][j] < 0.9:
#                 # print "  Fibre ",j,"  ratio value = ", vector_y[-1][j]
#                 vlines.append(j)
#         print("\n  TOTAL = ", len(vlines), " fibres with flux differences > 10 % !!")

#         plot_plot(vector_x, vector_y[-1], label=label_skylines[-1], alpha=1, vlines=vlines,
#                   hlines=[0.8, 0.9, 1.0, 1.1, 1.2], ylabel="Flux / Median flux", xlabel="Fibre",
#                   ymin=0.7, ymax=1.3, ptitle=ptitle)

#         # CHECKING SOMETHING...
#         self.throughput_extra_checking_skylines = vector_y[-1]

# #        for i in range(self.n_spectra):
# #            if i != 546 or i != 547:
# #                self.intensity_corrected[i] = self.intensity_corrected[i] / vector_y[-1][i]

            
            
        wavelength_shift_correction_data = WavelengthShiftCorrection(sky_lines = sky_lines,
                                                                wavelength_shift_solution = wavelength_shift_solution,
                                                                sol_sky_lines = sol_sky_lines,
                                                                fitted_offset_sky_lines = fitted_offset_sky_lines)
                                                                #w_fixed_all_fibres = w_fixed_all_fibres)
                                                      
            
        return  wavelength_shift_correction_data        
# #-----------------------------------------------------------------------------
# %% ============================================================================
# #-----------------------------------------------------------------------------
    def wavelength_shift_using_skylines(rss,  
                                        sky_lines=None, 
                                        sky_lines_file = None,
                                        n_sky_lines = 3,   #TODO need to be implemented
                                        valid_wave_min = None,
                                        valid_wave_max = None,
                                        only_fibre = None,
                                        maxima_sigma=2.5, maxima_offset=1.5,
                                        median_fibres = 5,
                                        index_fit = 2, kernel_fit= None, clip_fit =0.4,
                                        fibres_to_plot=None,
                                        show_fibres=None,
                                        #check_throughput=False,  #TODO Needs to be implemented?
                                        **kwargs): #plot=True, verbose=True, warnings=True, fig_size=12):
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
        verbose = kwargs.get('verbose', False)
        plot_all = kwargs.get('plot_all', False)
        plot =  kwargs.get('plot', False)
        warnings = kwargs.get('warnings', verbose)
        
        if verbose: print("\n> Computing small wavelength shifts using skylines...")
        
        w = rss.wavelength
        nspec = len(rss.intensity)
                
        # Check valid wavelength range # #!!!  NOTE: if valid_wave_min, valid_wave_max included in rss.info (from mask), this would not be needed here!!!        
        if valid_wave_min is None or valid_wave_max is None : valid_wave_range_data = rss_valid_wave_range (rss)    
        if valid_wave_min is None: valid_wave_min = valid_wave_range_data[2][0]   # valid_wave_min = w[0]
        if valid_wave_max is None: valid_wave_max = valid_wave_range_data[2][1]   # valid_wave_max = w[-1]    
         
        if fibres_to_plot is None:
            fibres_to_plot = [0, int(np.percentile(range(len(rss.intensity)),17)), int(np.percentile(range(len(rss.intensity)),34)),
                              int(np.percentile(range(len(rss.intensity)),50)),
                              int(np.percentile(range(len(rss.intensity)),67)),int(np.percentile(range(len(rss.intensity)),84)),                              
                              len(rss.intensity)-1]
        
        if sky_lines_file is None:
            sky_lines_file = path.join(path.dirname(__file__), '..', 'input_data',
                                      'sky_lines', 'sky_lines_rest.dat')
            
        #if sky_lines is None:
            #if self.grating == "385R": sky_lines = [6300.309, 8430.147, 8465.374]
            #if self.grating == "2000R": sky_lines = [6498.737, 6553.626, 6863.971]
    
        # Read skylines from file        
        sl_center_, sl_name_, sl_fnl_, sl_lowlow_, sl_lowhigh_, sl_highlow_, sl_highhigh_, sl_xmin_, sl_xmax_ = read_table(sky_lines_file, ["f", "s", "f", "f", "f", "f", "f", "f", "f"])

        # Be sure the lines we are using are in the requested wavelength range        
        if verbose: 
            print("  Checking the values of skylines provided in the file\n ",sky_lines_file)
            print("  ---------------------------------------------------------------------")
            print("      Center   fnl  lowlow lowhigh    highlow highhigh      xmin   xmax")
            for i in range(len(sl_center_)):
                print(
                "  {:11.3f}   {:.0f}    {:5.1f}  {:5.1f}       {:5.1f}   {:5.1f}      {:6.1f} {:6.1f}".format(sl_center_[i],
                                                                                                sl_fnl_[i],
                                                                                                sl_lowlow_[i],
                                                                                                sl_lowhigh_[i],
                                                                                                sl_highlow_[i],
                                                                                                sl_highhigh_[i],
                                                                                                sl_xmin_[i],
                                                                                                sl_xmax_[i]))
            print("  ---------------------------------------------------------------------")
            print( "  We only need skylines in the {:.2f} - {:.2f} range:".format(np.round(valid_wave_min, 2), np.round(valid_wave_max, 2)))


        skyline_in_absorption_=[True if line< 0 else False for line in sl_center_]  
        sl_center_=np.abs(sl_center_)
        if sky_lines is not None: sky_lines=np.abs(sky_lines)

        if sky_lines is None:
            valid_skylines = np.where((sl_center_ < valid_wave_max) & (sl_center_ > valid_wave_min))
        else:
            valid_skylines = [list(sl_center_).index(line) for line in sky_lines]
            
        sl_center = sl_center_[valid_skylines]
        sl_fnl = sl_fnl_[valid_skylines]
        sl_lowlow = sl_lowlow_[valid_skylines]
        sl_lowhigh = sl_lowhigh_[valid_skylines]
        sl_highlow = sl_highlow_[valid_skylines]
        sl_highhigh = sl_highhigh_[valid_skylines]
        sl_xmin = sl_xmin_[valid_skylines]
        sl_xmax = sl_xmax_[valid_skylines]
        skyline_in_absorption =[]
        for i in range(len(sl_center_)):
            if i in valid_skylines[0]:
                skyline_in_absorption.append(skyline_in_absorption_[i])        
        
        number_sl = len(sl_center)
        sky_lines = list(sl_center)
        if verbose: print("  Valid skylines: ", sky_lines, " , total =",number_sl)

        
        # Check if we perform a median
        if only_fibre is not None: median_fibres = None
        if median_fibres == None:
            intensity = rss.intensity
            xfibre = list(range(nspec))
            if kernel_fit is None: kernel_fit = 19
            vprint("  Performing Gaussian fitting to selected skylines in all spectra...", **kwargs)
        else:
            intensity=[]
            xfibre=[]
            if kernel_fit is None: kernel_fit = 3
            combined_fibres=[]
            last_fibre=0
            for i in range(0, nspec-median_fibres, median_fibres):
                intensity.append(np.nanmedian(rss.intensity[i:i+median_fibres], axis=0))
                xfibre.append(i+int(median_fibres/2))
                combined_fibres.append(range(i,i+median_fibres))
                last_fibre= i+median_fibres
            intensity.append(np.nanmedian(rss.intensity[last_fibre:nspec], axis=0))  
            xfibre.append(int(np.nanmedian(range(last_fibre,nspec))))
            combined_fibres.append(range(last_fibre,nspec))
            vprint(f"  Performing Gaussian fitting to selected skylines in spectra, median = {median_fibres} (total = {str(len(xfibre))} spectra)...", **kwargs)


        wave_median_offset = []
        all_offset_sky_lines = [[] for x in range(number_sl)]  #  [[] for i in repeat(None, number_sl)]  # List with list of offsets per skyline
        all_offset_sky_lines_fluxes = [[] for x in range(number_sl)] # List with list of gaussian flux per skyline
        #print(all_offset_sky_lines)
        

        if only_fibre is not None:
            f_i = only_fibre
            f_f = only_fibre + 1
            if verbose: print("  Checking fibre ", only_fibre,
                              " (use only_fibre = None for all)...")
            verbose_ = True
            warnings = True
            plot_all = True
            next_output = only_fibre
        else:
            f_i = 0
            f_f = len(intensity)
            verbose_ = False
            next_output = -1

        number_fibres_to_check = len(list(range(f_i, f_f)))
        output_every_few = np.sqrt(len(list(range(f_i, f_f)))) + 1
        
        for fibre in range(f_i, f_f):  
            spectrum = intensity[fibre]
            if verbose:
                if fibre > next_output:
                    sys.stdout.write("\b" * 51)
                    sys.stdout.write("  Checking fibre {:4} ...  ({:6.2f} % completed) ...".format(fibre, fibre * 100. / number_fibres_to_check))
                    sys.stdout.flush()
                    next_output = fibre + output_every_few

            # Gaussian fits to the sky spectrum
            sl_gaussian_flux = []
            sl_gaussian_sigma = []
            sl_gauss_center = []
            sl_offset = []
            sl_offset_good = []
            for skyline in range(number_sl):   #skyline here is an index
                # Check if we need to plot the fit
                plot_fit = False
                if plot:
                    if sl_fnl[skyline] == 1:  # Reading the file with lines, if this is 1 plots the fibre, careful as it will plot all fibres!
                        plot_fit = True  
                        if fibre == 0: vprint("\n  Plotting Gaussian fitting for skyline", sl_center[skyline], "in ALL fibres...", **kwargs)
                    if show_fibres is not None:
                        if median_fibres is None and skyline in show_fibres:
                            vprint("\n  - Plotting Gaussian fitting for skyline", sl_center[skyline], "in fibre", fibre, ":", **kwargs)
                            plot_fit = True
                        elif median_fibres is not None:
                            if [_fibre_ in show_fibres for _fibre_ in combined_fibres[fibre]].count(True) > 0:
                                vprint("\n  - Plotting Gaussian fitting for skyline", sl_center[skyline], "in fibre interval", combined_fibres[fibre][0],"-", combined_fibres[fibre][-1],":", **kwargs)
                                plot_fit = True
                    if plot_all: plot_fit = True
                
                # Perform Gaussian fit
                resultado = fluxes(w, spectrum, sl_center[skyline], 
                                   lowlow=sl_lowlow[skyline], lowhigh=sl_lowhigh[skyline],
                                   highlow=sl_highlow[skyline], highhigh=sl_highhigh[skyline], 
                                   xmin=sl_xmin[skyline], xmax=sl_xmax[skyline],
                                   broad=2.1 * 2.355, plot=plot_fit, verbose=False, plot_sus=False, fcal=False,
                                   warnings=warnings)  # Broad is FWHM for Gaussian sigma = 1,
                sl_gaussian_flux.append(resultado[3])
                sl_gauss_center.append(resultado[1])
                sl_gaussian_sigma.append(resultado[5] / 2.355)
                sl_offset.append(sl_gauss_center[skyline] - sl_center[skyline])

                # Check if the fit is good, using the values of maxima_offset and maxima_sigma
                if sl_gaussian_flux[skyline] < 0 and skyline_in_absorption[skyline] is False:
                    if np.abs(sl_center[skyline] - sl_gauss_center[skyline]) > maxima_offset or sl_gaussian_sigma[skyline] > maxima_sigma:
                            if verbose_:  print("\n  Bad fitting for ", sl_center[skyline], " in fibre ",xfibre[fibre], "... ignoring this fit...")
                else:
                    sl_offset_good.append(sl_offset[skyline])
                    if verbose_: print(
                        "\n    Fitted wavelength for sky line {:8.3f}:    center = {:8.3f}     sigma = {:6.3f}    offset = {:7.3f} ".format(
                            sl_center[skyline], sl_gauss_center[skyline], sl_gaussian_sigma[skyline], sl_offset[skyline]))
                all_offset_sky_lines[skyline].append(sl_gauss_center[skyline] - sl_center[skyline])
                all_offset_sky_lines_fluxes[skyline].append(resultado[3])  # Unsure if we need this, perhaps for checking the throughput with skylines?? #!!!

            median_offset_fibre = np.nanmedian(sl_offset_good)     # Median offset per fibre for this skyline
            wave_median_offset.append(median_offset_fibre)         # Append the median offset per fibre to list
            if verbose_: print("\n> Median offset for fibre {:3} = {:7.3f}".format(fibre, median_offset_fibre))

        if verbose and only_fibre is  None:
            sys.stdout.write("\b" * 51)
            sys.stdout.write("  Checking fibres completed!                  ")
            sys.stdout.flush()
            print(" ")
            print("\n")

        if only_fibre is None:

            # Checking values and performing fits
            # First, check individual fits
            sol_sky_lines = dict()        # Dictionary with skykline and fit solution (index_fit + 1 values)
            sky_line_fits_sol=[]          # list of fit solutions per skyline (same thing that sol_sky_lines, no dictionary)
            fitted_offset_sky_lines = []  # List with the nspec-values of the index_fit polynomial fit for each skyline
            for skyline in range(number_sl):   #skyline here is an index
                if verbose: print("  - Checking fit in skyline", sl_center[skyline],":")
                fit, pp, fx_, y_fit_c, x_c, y_c  = fit_clip(xfibre, all_offset_sky_lines[skyline], clip=clip_fit, 
                                                            plot=plot, verbose = verbose, 
                                                            xlabel="Fibre",ylabel="offset",
                                                            xmin=kwargs.get("xmin",xfibre[0]-20),
                                                            xmax=kwargs.get("xmax",xfibre[-1]+20),
                                                            extra_y = kwargs.get("extra_y",0.2),
                                                            index_fit = index_fit, kernel = kernel_fit, 
                                                            hlines=[0], vlines=[xfibre[0],xfibre[-1]])
                y = np.poly1d(fit)
                fitted_offset_sky_lines.append(list(y(list(range(nspec)))))
                sky_line_fits_sol.append(fit)  
                sol_sky_lines[str(sl_center[skyline])] = fit
            
            # Check combined fit
            if verbose: print("  - Checking combined fits:")
            fit, pp, fx_, y_fit_c, x_c, y_c  = fit_clip(xfibre, wave_median_offset, clip=clip_fit*2, 
                                                        plot=plot, verbose = verbose,
                                                        xlabel="Fibre",ylabel="offset",
                                                        xmin=kwargs.get("xmin",xfibre[0]-20),
                                                        xmax=kwargs.get("xmax",xfibre[-1]+20),
                                                        extra_y = kwargs.get("extra_y",0.2),
                                                        index_fit = index_fit, kernel = kernel_fit, 
                                                        hlines=[0], vlines=[xfibre[0],xfibre[-1]])
    
            # 1. fitted_offset_sl_median:  Median of the individual fits per skyline BLACK, solution = NO SOLUTION, is it a MEDIAN
            fitted_offset_sl_median = np.nanmedian(fitted_offset_sky_lines, axis=0)  # List with the nspec-values of the MEDIAN index_fit polynomial fit of all skylines
           
            # 2. fitsol: Median of solutions GREEN   solution = wavelength_shift_solution
            fitted_solutions = np.nanmedian(sky_line_fits_sol, axis=0)               # List with the median index_fit + 1 values (median of all skylines)
            y = np.poly1d(fitted_solutions)                                          # Solution considering the median of the fits
            fitsol = y(list(range(nspec)))                                           # List with the nspec-values of the solution considering the median of the fits
            
            if index_fit == 1:
                wavelength_shift_solution = [fitted_solutions[1], fitted_solutions[0], 0]
                #ptitle = "Linear fit to individual offsets"
                if verbose: print("\n> Median of solutions to the linear polynomy a0x +  a1x * fibre to all fibres (GREEN):")
            else:
                wavelength_shift_solution = [fitted_solutions[2], fitted_solutions[1], fitted_solutions[0]]
                #ptitle = "Second-order fit to individual offsets"
                if verbose: 
                    if index_fit != 2 and verbose : print("  A fit of order", index_fit,"was requested, but this tasks only runs with orders 1 or 2.")
                    print("\n> Median of solutions to the second-order polynomy a0x +  a1x * fibre + a2x * fibre**2 to all fibres (GREEN):")
            if verbose:
                print(" ", wavelength_shift_solution)
                print("  offset_min = {:.3f} A ,  offset_max = {:.3f} A,  offset_difference = {:.3f} A".format(np.nanmin(fitsol), np.nanmax(fitsol),np.nanmax(fitsol)-np.nanmin(fitsol)))
    
            # 3. fitsol_median_wave:       Fit to the median offset per wave   BLUE    solution = wavelength_shift_solution_wave
            y_median_wave =np.poly1d(fit)                            # Fit to the MEDIAN value of each fit per wave
            fitsol_median_wave = y_median_wave(list(range(nspec)))   # List with the nspec-values of the fit to the median value per wave
            wavelength_shift_solution_wave = [y_median_wave[2], y_median_wave[1], y_median_wave[0]]   
    
            if index_fit == 1:
                wavelength_shift_solution_wave = [y_median_wave[0], y_median_wave[1], 0]
                #ptitle = "Linear fit to individual offsets"
                if verbose: print("\n> Fitting a linear polynomy a0x +  a1x * fibre to the median offset per skyline (BLUE):")
            else:
                wavelength_shift_solution_wave = [y_median_wave[0], y_median_wave[1], y_median_wave[2]]
                #ptitle = "Second-order fit to individual offsets"
                if verbose: 
                    if index_fit != 2 and verbose : print("  A fit of order", index_fit,"was requested, but this tasks only runs with orders 1 or 2.")
                    print("\n> Fitting a second-order polynomy a0x +  a1x * fibre + a2x * fibre**2 to the median offset per skyline (BLUE):")
            if verbose:
                print(" ", wavelength_shift_solution_wave)
                print("  offset_min = {:.3f} A ,  offset_max = {:.3f} A,  offset_difference = {:.3f} A".format(np.nanmin(fitsol_median_wave), np.nanmax(fitsol_median_wave),np.nanmax(fitsol_median_wave)-np.nanmin(fitsol_median_wave)))
    
            # So we have:
            # fitted_offset_sky_lines :  List with the individual fits per skyline,         solution = sky_line_fits_sol
            # fitted_offset_sl_median :  Median of the individual fits per skyline: BLACK   solution = NONE
            # fitsol                  :  Median of solutions: GREEN                         solution = wavelength_shift_solution
            # fitsol_median_wave      :  Fit to the median offset per wave: BLUE            solution = wavelength_shift_solution_wave
            
            # For checking small wavelength offsets per fibre, including min,med,max, std and dipersion
            offsets_fibre=[[fitted_offset_sky_lines[i][show_fibre] for i in range(len(sky_lines)) ] for show_fibre in fibres_to_plot]
        
            # Plots
            if kwargs.get("plot"):
                #Plot fitting solutions to all skylines
                _y_ = [ y for y in fitted_offset_sky_lines ]                # Results of the individual fits per skyline      
                if len(sky_lines)> 2: _y_.append(fitted_offset_sl_median)   # Add the median of individual fits if +2: BLACK
                _y_.append(fitsol)                                          # Add the median of the fitted solution per skyline, GREEN
                _y_.append(fitsol_median_wave)                              # Add fit to the median offset per wave  BLUE 
                label_lines = [str(sky_lines[i])  for i in range(len(sky_lines))  ]
                if len(sky_lines)> 2: label_lines.append("median of fits")
                label_lines.append("median of solutions")
                label_lines.append("median offset per skyline")
                linestyle_lines =["--"  for i in range(len(sky_lines))]
                linestyle_lines.append("-")
                linestyle_lines.append("-")
                if len(sky_lines)> 2: linestyle_lines.append("-")
                color_lines = None            
                if len(sky_lines) == 4: color_lines = ["r", "orange", "cyan", "purple", "k", "g", "b"]
                if len(sky_lines) == 3: color_lines = ["r", "orange", "cyan", "k", "g", "b"]
                if len(sky_lines) == 2: color_lines = ["r", "orange", "g", "b"]
                if len(sky_lines) > 4 :
                    color_lines = [None]*len(sky_lines)
                    color_lines.append("k")
                    color_lines.append("g")
                    color_lines.append("b")
                alpha_lines = [0.7] * len(sky_lines)
                linewidth =[1]*len(sky_lines)
                if len(sky_lines)> 2: 
                    alpha_lines.append(0.9)
                    linewidth.append(2)
                alpha_lines.append(0.9)
                linewidth.append(2)
                alpha_lines.append(0.9)
                linewidth.append(2)
                nspec_vector = list(range(nspec))
                plot_plot(nspec_vector, _y_, 
                          color= color_lines, alpha = alpha_lines, linestyle = linestyle_lines, linewidth =linewidth,
                          extra_y = 0.2, loc = 3, ncol = 4, hlines=[0], 
                          label=label_lines,ptitle="Checking fitting solutions to all skylines",
                          xlabel="Fibre", ylabel="Fitted offset [$\mathrm{\AA}$]")
                
                # Plot small wavelength offsets per fibre  
                label=[str(show_fibre) for show_fibre in fibres_to_plot]
                
                # for cosa in [sky_lines, offsets_fibre,  ["--"]*len(sky_lines), [1.5]*len(sky_lines), [1]*len(sky_lines)]:
                
                #     print(len(cosa), cosa)
                
                
                plot_plot(sky_lines, offsets_fibre, 
                          linestyle = ["--"]*len(fibres_to_plot),
                          linewidth = [1.5]*len(fibres_to_plot),
                          alpha = [1]*len(fibres_to_plot),
                          loc = 3, ncol = 7, label=label,
                          ymin = -2, ymax=2,   # para que no casque plot_plot... needs to be fixed!
                          show = False)                       # Show = False as we need the second plot_plot for green crosses 
                plot_plot(sky_lines, offsets_fibre, 
                          fig_size="C",                       # fig_size="C"  to continue the figure we started
                          psym = ["+"]*len(fibres_to_plot),
                          alpha = [1]*len(fibres_to_plot),
                          color = ["green"]*len(fibres_to_plot),
                          linewidth=[1.3]*len(fibres_to_plot),
                          markersize=[12]*len(fibres_to_plot),
                          xmin=np.nanmin(w),
                          xmax=np.nanmax(w),
                          ptitle="Small wavelength offsets per fibre",
                          xlabel="Wavelength [$\mathrm{\AA}$]",
                          ylabel="Fitted offset [$\mathrm{\AA}$]",
                          extra_y = 0.22, 
                          show = True)
            if verbose: 
                print(f"\n> Small wavelength offsets for selected fibres considering the {len(sky_lines)} skylines:")
                print("  --------------------------------------------------")
                print("  Fibre        Min     Med    Max      std      disp   (all in A)")
                for i in range(len(offsets_fibre)):
                    #print(offsets_fibre[i])
                    min_value,median_value,max_value,std, rms, snr = basic_statistics(offsets_fibre[i], return_data=True, verbose = False)
                    print("   {:4.0f}     {:6.3f}  {:6.3f} {:6.3f}    {:.3f}     {:.3f}".format(fibres_to_plot[i],   min_value,median_value,max_value,std,   max_value-min_value))
                print("  --------------------------------------------------")
    
            
            # Prepare the results to be saved in an object with all the key info
            _wavelength_shift_solution_ = [np.array(wavelength_shift_solution), np.array(wavelength_shift_solution_wave)] #, sky_line_fits_sol] this one is in sol_sky_lines
            wavelength_shift_correction_data = WavelengthShiftCorrection(sky_lines = sky_lines,
                                                                         wavelength_shift_solution = _wavelength_shift_solution_,
                                                                         sol_sky_lines = sol_sky_lines,
                                                                         fitted_offset_sky_lines = fitted_offset_sky_lines)
                                                                         #w_fixed_all_fibres = w_fixed_all_fibres)
                                                                
            # Add information in History / log
            # self.history.append("  sol (found) = " + str(sol))
            if verbose: print("\n> Computing small wavelength shifts using skylines COMPLETED and saved in a WavelengthShiftCorrection object!")
    
            return   wavelength_shift_correction_data  
            

# #-----------------------------------------------------------------------------
# %% ============================================================================
# #-----------------------------------------------------------------------------

    def apply(self, rss, 
              wavelength_shift_solution = None,  
              median_offset_per_skyline_weight = 1.,   # 1 is the BLUE, 0 is the GREEN
              sky_lines = None, 
              show_fibres=None, show_skylines = None,
              plot_solution = False,
              **kwargs):                               
        """Apply a 2D wavelength correction to a RSS.

        Parameters
        ----------
        - wavelength_shift_solution : sol
        - rss: (RSS)
        - median_offset_per_skyline_weight: 
            1 is the BLUE line (median offset per skyline), 0 is the GREEN line (median of solutions), anything between [0,1] is a combination.
        """
        
        if wavelength_shift_solution is None and self.wavelength_shift_solution is not None:
            wavelength_shift_solution = self.wavelength_shift_solution
        else:
            raise RuntimeError("Wavelength correction not provided!")

        if sky_lines is None and self.sky_lines is not None: sky_lines = self.sky_lines
        
        if median_offset_per_skyline_weight < 0 or  median_offset_per_skyline_weight > 1:
            raise ValueError("median_offset_per_skyline_weight must be in the range [0 , 1]")
            
        
        # if str(type(wavelength_shift_solution)) != str(Wavelength):        #!!! ANGEL    Así me falla, tengo que llamar a str y usar != en lugar de is not
        #     raise AttributeError("Input throughput must be an instance of Throughput class")

        # # if type(rss) is not RSS:                          #!!! ANGEL    Así me falla, tengo que llamar a str y usar != en lugar de is not
        # if str(type(rss)) != str(RSS):
        #     raise ValueError("Throughput can only be applied to RSS data:\n input {}"
        #                      .format(type(rss)))
            
        # =============================================================================
        # Verbose if needed   
        # =============================================================================    
        # vprint('> Applying wavelength shift solution to all fibres in object ...', **kwargs)   
        # =============================================================================
        # Copy input RSS for storage the changes implemented in the task   
        # =============================================================================
        rss_out = copy.deepcopy(rss)
        
        w = rss.wavelength
        nspec = len(rss.intensity)
        xfibre = list(range(nspec))
        if show_fibres is None:  show_fibres=[0, int(nspec/2), nspec-1]
        if show_skylines is None: show_skylines = [self.sky_lines[0], self.sky_lines[-1]]
    
        if len (self.wavelength_shift_solution) == 2:
            wavelength_shift_solution = (1-median_offset_per_skyline_weight) * self.wavelength_shift_solution[0] + median_offset_per_skyline_weight * self.wavelength_shift_solution [1]            
            if kwargs.get("verbose"):
                if median_offset_per_skyline_weight == 1: 
                    print('> Applying wavelength shift solution to all fibres in object using median offset for skyline ...')
                elif median_offset_per_skyline_weight == 0:
                    print('> Applying wavelength shift solution to all fibres in object using median of solutions of skylines ...')
                else:
                    print(f'> Applying wavelength shift solution to all fibres in object\n  using {(1-median_offset_per_skyline_weight)} * median offset for skyline + {(median_offset_per_skyline_weight)} * median of solutions of skylines ...')
        else:
            vprint('> Applying wavelength shift solution to all fibres in object ...', **kwargs)  
            
            
        #sol = wavelength_shift_solution
        fx = wavelength_shift_solution[0] + wavelength_shift_solution[1] * np.array(xfibre) + wavelength_shift_solution[2] * np.array(xfibre) ** 2

        if plot_solution:   # Do we need to plot the correction again?     
            if len(wavelength_shift_solution) == 1: 
                ptitle = "Linear correction, y = "+str(round(wavelength_shift_solution[0],4))+" + "+str(round(wavelength_shift_solution[1],6))+"x"
            else:
                ptitle = "Second-order correction, y = {:.4f} + {:.6f}x + {:.3e}".format(wavelength_shift_solution[0],wavelength_shift_solution[1],wavelength_shift_solution[2])+"x$^{2}$"
            plot_plot(xfibre, fx, ptitle=ptitle, xlabel="Fibre", ylabel="offset [$\mathrm{\AA}$]", hlines=[0])
        
        vprint("  Polynomic solution:", wavelength_shift_solution, **kwargs)
        vprint("  offset_min = {:.3f} A ,  offset_max = {:.3f} A,  offset_difference = {:.3f} A".format(np.nanmin(fx), np.nanmax(fx),np.nanmax(fx)-np.nanmin(fx)), **kwargs)
    
        
        # Apply corrections to all fibres
        for fibre in range(nspec):  
            
            w_shift = fx[fibre]
            rss_out.intensity[fibre] = rebin_spec_shift(w, rss.intensity[fibre], w_shift)
            rss_out.variance[fibre] = rebin_spec_shift(w, rss.variance[fibre], w_shift)
            w_fixed = w - w_shift
    
            if kwargs.get("plot"):
                for line in sky_lines: 
                    if fibre in show_fibres and line in show_skylines:
                        ptitle = "Wavelength correction in Fibre " + str(fibre) + " in skyline "+str(line)
                        plot_plot([w, w_fixed, w], [rss.intensity[fibre], rss.intensity[fibre], rss_out.intensity[fibre]],
                                  xmin = line-20, xmax= line+20,
                                  xlabel="Wavelength [$\mathrm{\AA}$]",ylabel="Flux", ptitle=ptitle,
                                  vlines=[line], alpha=[0.2,0.2,0.6], 
                                  label= ["No corrected","No corrected - Shifted","Corrected after rebinning"],
                                  color=["r","b","g"],linewidth=[1,1,2],
                                  percentile_min=0.5, percentile_max=100, ncol = 3
                                  )
                
        vprint("\n> Small fixing of the wavelength shifts APPLIED!", **kwargs)
    
        #!!! TODO: log / history...
        rss.log["wavelength fix"]["index"] = len(wavelength_shift_solution)
        rss.log["wavelength fix"]["sol"] = [wavelength_shift_solution]
        #self.log_correction(rss, status='applied')      ##!!! ??????
        return rss_out
