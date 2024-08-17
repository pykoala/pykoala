import os
import numpy as np
import copy
import sys
from astropy.io import fits

from pykoala.corrections.correction import CorrectionBase
from pykoala.rss import RSS
from pykoala.ancillary import flux_conserving_interpolation
from pykoala.plotting.quick_plot import quick_plot, basic_statistics
from pykoala.spectra.onedspec import read_table, fluxes, fit_clip, rebin_spec_shift
from tqdm import tqdm
from pykoala.ancillary import print_counter

class WavelengthOffset(object):
    """Wavelength offset class.

    This class stores a 2D wavelength offset.

    Attributes
    ----------
    offset_data : wavelength offset, in pixels
    offset_error : standard deviation of `offset_data`
    """
    offset_data = None
    offset_error = None

    def __init__(self, path=None, offset_data=None, offset_error=None):
        self.path = path
        self.offset_data = offset_data
        self.offset_error = offset_error

        if self.path is not None and self.offset_data is None:
            self.load_fits()

    def tofits(self, output_path):
        primary = fits.PrimaryHDU()
        data = fits.ImageHDU(data=self.offset_data, name='OFFSET')
        error = fits.ImageHDU(data=self.offset_error, name='OFFSET_ERR')
        hdul = fits.HDUList([primary, data, error])
        hdul.writeto(output_path, overwrite=True)
        hdul.close(verbose=True)
        print(f"[wavelength offset] offset saved at {output_path}")

    def load_fits(self):
        """Load the offset data from a fits file.

        Loads offset values (extension 1) and
        associated errors (extension 2) from a fits file.
        """
        if not os.path.isfile(self.path):
            raise NameError(f"offset file {self.path} does not exist.")
        print(f"[wavelength offset] Loading offset from {self.path}")
        with fits.open(self.path) as hdul:
            self.offset_data = hdul[1].data
            self.offset_error = hdul[2].data


class WavelengthCorrection(CorrectionBase):
    """Wavelength correction class.

    This class accounts for the relative wavelength offset between fibres.

    Attributes
    ----------
    name : str
        Correction name, to be recorded in the log.
    offset : WavelengthOffset
        2D wavelength offset (n_fibres x n_wavelengths)
    verbose: bool
        False by default.
    """
    name = "WavelengthCorrection"
    offset = None
    verbose = False

    def __init__(self, offset_path=None, offset=None, **correction_kwargs):
        super().__init__(**correction_kwargs)

        path = offset_path
        if offset is not None:
            assert isinstance(offset, WavelengthOffset)
            self.offset = offset
        else:
            self.offset = WavelengthOffset(path=path)
        

    def apply(self, rss):
        """Apply a 2D wavelength offset model to a RSS.

        Parameters
        ----------
        rss : RSS
            Original Row-Stacked-Spectra object to be corrected.

        Returns
        -------
        RSS
            Corrected RSS object.
        """

        assert isinstance(rss, RSS)

        rss_out = copy.deepcopy(rss)
        x = np.arange(rss.wavelength.size)
        for i in range(rss.intensity.shape[0]):
            rss_out.intensity[i] = flux_conserving_interpolation(
                x, x - self.offset.offset_data[i], rss.intensity[i])

        self.record_correction(rss_out, status='applied')
        return rss_out


class WavelengthShiftCorrection(CorrectionBase):
    """
    Wavelength shiftcorrection class.

    This class accounts for the small correction in wavevelength shifts.

    Attributes
    ----------
    - name : Name of the correction
    - sky_lines : skylines used for deriving the correction
    - sol_sky_lines : solutions to the fits using skylines
    - wavelength: wavelength vector
    - fitted_offset_sky_lines: all parameters for the Gaussian fitting in skylines
    - w_fixed_all_fibres : wavelength fixed for all fibres
    - fits_file: fits file to read or save
    - wcs: WCS of the fits file with the wavelength information
    - header: header of the fits file
    
    """
    
    name = "WavelengthShiftCorrection"
    wavelength_shift_solution = None
    verbose = False

    def __init__(self, 
                 sky_lines = None,
                 wavelength_shift_solution=None, 
                 sol_sky_lines = None,
                 wavelength = None,
                 fitted_offset_sky_lines = None,
                 w_fixed_all_fibres =None,
                 fits_file = None,
                 wcs = None,
                 header = None,
                 **kwargs):
        
        self.sky_lines = sky_lines
        self.wavelength = wavelength
        self.wavelength_shift_solution = wavelength_shift_solution
        self.sol_sky_lines = sol_sky_lines
        self.fitted_offset_sky_lines = fitted_offset_sky_lines
        self.w_fixed_all_fibres = w_fixed_all_fibres
        self.fits_file = fits_file
        self.wcs = wcs
        self.header = header
        
        self.verbose=kwargs.get("verbose", False)
        self.corr_print("Initialising wavelength correction model.")
        
        if self.fits_file is not None and self.wavelength_shift_solution is None:
            self.load_fits(**kwargs)
        
        
    def tofits(self, fits_file, path_to_file = None ):
        """
        Save the small wavelength shift correction into a fits file.

        Parameters
        ----------
        fits_file : string
            name of the fits file to be saved.
        path_to_file : string, optional
            path to the file. The default is None.

        Returns
        -------
        None.

        """
        primary = fits.PrimaryHDU()
        WAVSHIFT = fits.ImageHDU(data=self.wavelength_shift_solution,
                            name='WAVSHIFT')
        #WSHIFTER = fits.ImageHDU(data=self.throughput_error,
        #                    name='WSHIFTER')
        FITTED = fits.ImageHDU(data=self.fitted_offset_sky_lines, name='FITTED')

        # DATE, GRATING, WAVELENGTH RANGE SHOULD BE HERE TOO           
        primary.header = self.header
        primary.header['OBJECT'] = self.name 
        
        #wcs = WCS(naxis=2)   # ÁNGEL: I CAN'T GET WCS RIGHT, MOVING ALL THE HEADER FROM RSS      #FIXME
        #wcs = self.wcs
        #primary.header = wcs.to_header()
        
        # primary.header["CRVAL1"] = self.wavelength[0]
        # primary.header["CDELT1"] = (self.wavelength[-1]-self.wavelength[0])/(len(self.wavelength)-1)
        # primary.header["CRPIX1"] = 1. 
        # primary.header["NAXIS"] = 2 
        # primary.header["NAXIS1"] = len(self.wavelength) 
        primary.header["NWAVES"] = len(self.wavelength)  #FIXME  This is for getting WCS manually...
        
        i = 1
        for item in self.sol_sky_lines:
            if i>9:
                text_skyline = "SKYLIN"+str(i)
                text_solsky  = "SOLSKY"+str(i)
            else:
                text_skyline = "SKYLIN0"+str(i)
                text_solsky  = "SOLSKY0"+str(i)
            primary.header[text_skyline] = item
            primary.header[text_solsky] = (str(self.sol_sky_lines[item]).strip("]")).strip("[")
            i=i+1
        hdul = fits.HDUList([primary, WAVSHIFT, FITTED])

        if path_to_file is not None: fits_file = os.path.join(path_to_file,fits_file)
        hdul.writeto(fits_file, overwrite=True)
        hdul.close(verbose=True)
        print(f"[WavelengthShiftCorrection] WavelengthShiftCorrection saved at {fits_file}")
        
    def load_fits(self, fits_file = None, path_to_file=None):
        """
        Load the WavelengthShiftCorrection data from a fits file.
        
        Loads WavelengthShiftCorrection correction (extension 1) from a fits file
        
        Parameters
        ----------
        fits_file : string, optional
            name of the fits file to read.
            If not provided, it uses self.fits_file
        path_to_file : string, optional
            path to the file. The default is None.

        Returns
        -------
        None.
        
        """
        if fits_file is not None: self.fits_file = fits_file
        if path_to_file is not None: self.fits_file = path.join(path_to_file,self.fits_file)
        if not os.path.isfile(self.fits_file):
            raise NameError(f"WavelengthShiftCorrection file {self.fits_file} does not exists.")
        print(f"[WavelengthShiftCorrection] Loading WavelengthShiftCorrection from {self.fits_file}")
        with fits.open(self.fits_file) as hdul:
            self.wavelength_shift_solution = hdul[1].data
            self.fitted_offset_sky_lines = hdul[2].data
            
            header = fits.getheader(self.fits_file, 0) # + fits.getheader(path_to_file, 2)
            self.header = header
            #header = hdul[1].header
            #print(header)
            
            # Create wavelength from wcs   #FIXME !!!!!!!!!!!!!!!!!
            #wcs=WCS(header)
            #print(wcs)
            #nrow, ncol = wcs.array_shape
            #wavelength_index = np.arange(ncol)
            #self.wavelength = wcs.dropaxis(1).wcs_pix2world(wavelength_index, 0)[0]
            
            wavelength_index =  np.arange(header["NWAVES"])
            CRPIX1 = header["CRPIX1"]
            CRVAL1 = header["CRVAL1"]
            CDELT1 = header["CDELT1"]
            self.wavelength = CRVAL1 - CDELT1 * (CRPIX1-1) + wavelength_index * CDELT1 
            self.sky_lines = []
            for item in header:
                if item[0:6] == "SKYLIN": self.sky_lines.append(float(header[item]))
# #-----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# %% ============================================================================
# #-----------------------------------------------------------------------------
    def wavelength_shift_using_skylines(rss,  
                                        wavelength_shift_correction_file = None,
                                        sky_lines=None, 
                                        sky_lines_file = None,
                                        #n_sky_lines = 3,   #TODO need to be implemented
                                        valid_wave_min = None,
                                        valid_wave_max = None,
                                        only_fibre = None,
                                        maxima_sigma=2.5, 
                                        maxima_offset=1.5,
                                        median_fibres = 7,
                                        index_fit = 2, 
                                        kernel_fit= None, 
                                        clip_fit =0.5,
                                        fibres_to_plot=None,
                                        show_fibres=None,
                                        #check_throughput=False,  #TODO Needs to be implemented?
                                        **kwargs): #plot=True, verbose=True, warnings=True, fig_size=12):
        """
        Using bright skylines, derive the small wavelength shift correction for each fibre.
        
        Parameters: 
        ----------
        sky_lines : list of floats
            Chooses the sky lines to run calibration for small wavelength shifts
            If None, it will take 3 default lines depending on the AAOmega grating:
                if grating == "385R": sky_lines = [6300.309, 8430.147, 8465.374]
                if grating in ["2000R", "1000R"] : sky_lines = [6498.737, 6553.626, 6863.971]
                if grating == "580V": sky_lines = [-3934, -3968, 5577.338]     
        sky_lines_file: string
            If provided, file with the skylines including the full path. Default is None, 
            for which it will use input_data.skylines.sky_lines_rest
        valid_wave_min, valid_wave_max: float
            Valid wavelenghts to check, range [valid_wave_min, valid_wave_max]
        only_fibre: integer, optional
            If given, it only check that fibre. Default is None, that is checks everything.
        maxima_sigma: float
            Max of the sigma of the Gaussian fitting for each emission line to be considered a good fit. Default is 2.5.
        maxima_offset: float
            Max of the deviation of the Gaussian center from the provided central wavelength of the skyline to be considered a good fit. Default is 1.5 (A). 
        median_fibres: odd integer
            Do the Gaussian fitting promediating this number of adjacent wavelengths. Default is 7.  
        index_fit : integer
            Degree of the polynomiun to be used for fitting shift vs wavelength. Default is 2.     
        kernel_fit: odd integer
            Smoothing of the Gaussian fit results for fitting shift vs wavelength.  
            If median_fibres is not None, default is 3.
            If median_fibres is not given, default is 19.    
        clip_fit: float
            Clipping values for the fitting shift vs wavelength. Default is 0.5
        fibres_to_plot : list of integers
            Choose specific fibres to visualise fitted offsets per wavelength using plots.
            The default None that is percentiles 0, 17, 34, 50, 67, 84, 100.
        show_fibres : list of integers (default = [0,500,985])
            Fibres for which the comparison between uncorrected and corrected flux per wavelength will be showed.
            Default is None, and it will show the first fibre, the middle fibre, and the last fibre if plot=True
        **kwargs : kwargs
             Where we can find plot, verbose, warnings, verbose_counter, Jupyter, plot_all, ...
             
        Returns:
        -------
        
        wavelength_shift_correction_data object:
        
        WavelengthShiftCorrection(sky_lines = sky_lines,
                                  wavelength_shift_solution = _wavelength_shift_solution_,
                                  sol_sky_lines = sol_sky_lines,
                                  wavelength = copy.deepcopy(w),
                                  wcs = wcs,
                                  header = header,
                                  fitted_offset_sky_lines = fitted_offset_sky_lines)
        """
    
        verbose = kwargs.get('verbose', False)
        verbose_counter = kwargs.get('verbose_counter', verbose)
        Jupyter = kwargs.get('Jupyter', True)
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
            if os.path.dirname(__file__)[-9:] == "tutorials":
                sky_lines_file = os.path.join(os.path.dirname(__file__), '..', 'src','pykoala','input_data',
                                          'sky_lines', 'sky_lines_rest.dat')
            else:
                sky_lines_file = os.path.join(os.path.dirname(__file__), '..', 'input_data',
                                          'sky_lines', 'sky_lines_rest.dat')
            
        if sky_lines is None:
            grating = rss.koala.info["aaomega_grating"]
            if grating == "385R": sky_lines = [6300.309, 8430.147, 8465.374]
            if grating in ["2000R", "1000R"] : sky_lines = [6498.737, 6553.626, 6863.971]
            if grating == "580V": sky_lines = [-3934, -3968, 5577.338] 
        elif sky_lines == "all":
            sky_lines = None
    
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
                
        if valid_wave_min < sl_center_[0]:   #FIXME
            for j in range(len(sl_center_)):
                if sl_center_[j] in valid_skylines[0]: skyline_in_absorption.append(skyline_in_absorption_[j])        
        
        number_sl = len(sl_center)
        sky_lines = list(sl_center)
        if verbose: print("  Valid skylines: ", sky_lines, " , total =",number_sl)

        # Check if we perform a median
        if only_fibre is not None: median_fibres = None
        if median_fibres == None:
            intensity = rss.intensity
            xfibre = list(range(nspec))
            if kernel_fit is None: kernel_fit = 19
            if verbose: print("  Performing Gaussian fitting to selected skylines in all spectra...")
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
            if verbose: print(f"  Performing Gaussian fitting to selected skylines in spectra, median = {median_fibres} (total = {str(len(xfibre))} spectra)...")


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
        else:
            f_i = 0
            f_f = len(intensity)
            verbose_ = False
            
        number_fibres_to_check = len(list(range(f_i, f_f)))

        if verbose_counter:  
            if Jupyter:
                pbar = tqdm(total=number_fibres_to_check, file=sys.stdout)
            else:
                pbar = None
            next_output = print_counter(stage=1, Jupyter = Jupyter, pbar = pbar)
            if only_fibre is not None:  next_output = only_fibre

    
        for fibre in range(f_i, f_f):  
            spectrum = intensity[fibre]
            if verbose_counter:
                if fibre >= next_output :  
                    next_output = print_counter(stage=2, 
                                                iteration = fibre, 
                                                total = number_fibres_to_check,
                                                Jupyter = Jupyter,
                                                pbar = pbar)
                                                                     

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
                        if fibre == 0 and verbose: print("\n  Plotting Gaussian fitting for skyline", sl_center[skyline], "in ALL fibres...")
                    if show_fibres is not None:
                        if median_fibres is None and skyline in show_fibres:
                            if verbose: print("\n  - Plotting Gaussian fitting for skyline", sl_center[skyline], "in fibre", fibre, ":")
                            plot_fit = True
                        elif median_fibres is not None:
                            if [_fibre_ in show_fibres for _fibre_ in combined_fibres[fibre]].count(True) > 0:
                                if verbose: print("\n  - Plotting Gaussian fitting for skyline", sl_center[skyline], "in fibre interval", combined_fibres[fibre][0],"-", combined_fibres[fibre][-1],":")
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
                try:
                    if sl_gaussian_flux[skyline] < 0 and skyline_in_absorption[skyline] is False:
                        if np.abs(sl_center[skyline] - sl_gauss_center[skyline]) > maxima_offset or sl_gaussian_sigma[skyline] > maxima_sigma:
                                if verbose_:  print("\n  Bad fitting for ", sl_center[skyline], " in fibre ",xfibre[fibre], "... ignoring this fit...")
                    else:
                        sl_offset_good.append(sl_offset[skyline])
                        if verbose_: print(
                            "\n    Fitted wavelength for sky line {:8.3f}:    center = {:8.3f}     sigma = {:6.3f}    offset = {:7.3f} ".format(
                                sl_center[skyline], sl_gauss_center[skyline], sl_gaussian_sigma[skyline], sl_offset[skyline]))
                except Exception:
                    if warnings: print("\n    Something has failed in the fitting, ignoring it...")
                all_offset_sky_lines[skyline].append(sl_gauss_center[skyline] - sl_center[skyline])
                all_offset_sky_lines_fluxes[skyline].append(resultado[3])  # Unsure if we need this, perhaps for checking the throughput with skylines?? #!!!

            median_offset_fibre = np.nanmedian(sl_offset_good)     # Median offset per fibre for this skyline
            wave_median_offset.append(median_offset_fibre)         # Append the median offset per fibre to list
            if verbose_: print("\n> Median offset for fibre {:3} = {:7.3f}".format(fibre, median_offset_fibre))

            
        if verbose_counter and only_fibre is None: 
                print_counter(stage=3, Jupyter = Jupyter, pbar = pbar, total = number_fibres_to_check,
                              next_output = next_output )
        elif verbose: 
                print("    Process completed!\n")
        if verbose or verbose_counter: print(" ")

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
                quick_plot(nspec_vector, _y_, 
                          color= color_lines, alpha = alpha_lines, linestyle = linestyle_lines, linewidth =linewidth,
                          extra_y = 0.2, loc = 3, ncol = 4, hlines=[0], 
                          label=label_lines,ptitle="Checking fitting solutions to all skylines",
                          xlabel="Fibre", ylabel="Fitted offset [$\mathrm{\AA}$]")
                
                # Plot small wavelength offsets per fibre  
                label=[str(show_fibre) for show_fibre in fibres_to_plot]
                
                # for cosa in [sky_lines, offsets_fibre,  ["--"]*len(sky_lines), [1.5]*len(sky_lines), [1]*len(sky_lines)]:
                
                #     print(len(cosa), cosa)
                
                
                quick_plot(sky_lines, offsets_fibre, 
                          linestyle = ["--"]*len(fibres_to_plot),
                          linewidth = [1.5]*len(fibres_to_plot),
                          alpha = [1]*len(fibres_to_plot),
                          loc = 3, ncol = 7, label=label,
                          ymin = -2, ymax=2,   # para que no casque quick_plot... needs to be fixed!
                          show = False)                       # Show = False as we need the second quick_plot for green crosses 
                quick_plot(sky_lines, offsets_fibre, 
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
    
            try:
                wcs = copy.deepcopy(rss.koala.wcs)            # FIXME WE NEED TO KEEP the WCS of the rss... 
                header = copy.deepcopy(rss.koala.header)      # Taking all header as I can't get WCS working later...
            except Exception:
                wcs = None 
                header = None
            # Prepare the results to be saved in an object with all the key info
            _wavelength_shift_solution_ = [np.array(wavelength_shift_solution), np.array(wavelength_shift_solution_wave)] #, sky_line_fits_sol] this one is in sol_sky_lines
            wavelength_shift_correction_data = WavelengthShiftCorrection(sky_lines = sky_lines,
                                                                         wavelength_shift_solution = _wavelength_shift_solution_,
                                                                         sol_sky_lines = sol_sky_lines,
                                                                         wavelength = copy.deepcopy(w),
                                                                         wcs = wcs,
                                                                         header = header,
                                                                         fitted_offset_sky_lines = fitted_offset_sky_lines)
                                                                         #w_fixed_all_fibres = w_fixed_all_fibres)
                                                                
            # Add information in History / log
            # self.history.append("  sol (found) = " + str(sol))
            if verbose: print("\n> Computing small wavelength shifts using skylines COMPLETED and saved in a WavelengthShiftCorrection object!")
    
    
            # Save file if requested
            if wavelength_shift_correction_file is not None:
                path_to_file = kwargs.get('path', None)
                wavelength_shift_correction_data.tofits(wavelength_shift_correction_file, path_to_file)
    
            return   wavelength_shift_correction_data  
            

# #-----------------------------------------------------------------------------
# %% ============================================================================
# #-----------------------------------------------------------------------------
    def apply(self, rss, 
              wavelength_shift_solution = None,  
              median_offset_per_skyline_weight = 1.,   # 1 is the BLUE, 0 is the GREEN
              sky_lines = None, 
              show_fibres_for_wavelength_shifts=None,              
              show_skylines_for_wavelength_shifts = None,          
              plot_wavelength_shift_correction_solution = False,    
              **kwargs):                               
        """
        Apply the small wavelength shift correction to a RSS.

        Parameters
        ----------
        rss: object
            rss object to apply correction
        wavelength_shift_solution : list of floats or list of 2 list of floats
            each list has the parameters of the fitting polynomium (typically order 2-3) to create the wavelength shift correction
            if only list one is given, that is the solution that will be applied.
            if two lists are provided, it will combine them accordingly to the value of median_offset_per_skyline_weight.
            The two lists are obtained running task 
        median_offset_per_skyline_weight: float between [0,1] 
            0 is the GREEN line in plot (median of solutions), 
            1 is the BLUE line in plot (median offset per skyline), 
            anything between [0,1] is a combination.
        """
        verbose = kwargs.get('verbose', False)
        plot =  kwargs.get('plot', False)
        #warnings = kwargs.get('warnings', verbose)
        
        if wavelength_shift_solution is None and self.wavelength_shift_solution is not None:
            wavelength_shift_solution = self.wavelength_shift_solution
        else:
            raise RuntimeError("Wavelength shift correction not provided!")

        if sky_lines is None and self.sky_lines is not None: sky_lines = self.sky_lines
        
        if median_offset_per_skyline_weight < 0 or  median_offset_per_skyline_weight > 1:
            raise ValueError("median_offset_per_skyline_weight must be in the range [0 , 1]")
            
        
        # if str(type(wavelength_shift_solution)) != str(Wavelength):        #!!! ANGEL    Así me falla, tengo que llamar a str y usar != en lugar de is not
        #     raise AttributeError("Input throughput must be an instance of Throughput class")

        # # if type(rss) is not RSS:                          #!!! ANGEL    Así me falla, tengo que llamar a str y usar != en lugar de is not
        # if str(type(rss)) != str(RSS):
        #     raise ValueError("Throughput can only be applied to RSS data:\n input {}"
        #                      .format(type(rss)))
            
        
        
        rss_out = copy.deepcopy(rss)
        w = rss.wavelength
        
        #TODO check that initial wavelength in wavelength_shift_solution is the same one that in rss
        
        nspec = len(rss.intensity)
        xfibre = list(range(nspec))
        if show_fibres_for_wavelength_shifts is None:  show_fibres_for_wavelength_shifts=[0, int(nspec/2), nspec-1]
        if show_skylines_for_wavelength_shifts is None: show_skylines_for_wavelength_shifts = [self.sky_lines[0], self.sky_lines[-1]]
    
        if len (self.wavelength_shift_solution) == 2:
            wavelength_shift_solution = (1-median_offset_per_skyline_weight) * self.wavelength_shift_solution[0] + median_offset_per_skyline_weight * self.wavelength_shift_solution [1]            
            if verbose:
                if median_offset_per_skyline_weight == 1: 
                    print('> Applying wavelength shift solution to all fibres in object using median offset for skyline ...')
                elif median_offset_per_skyline_weight == 0:
                    print('> Applying wavelength shift solution to all fibres in object using median of solutions of skylines ...')
                else:
                    print(f'> Applying wavelength shift solution to all fibres in object\n  using {(1-median_offset_per_skyline_weight)} * median offset for skyline + {(median_offset_per_skyline_weight)} * median of solutions of skylines ...')
        else:
            if verbose: print('> Applying wavelength shift solution to all fibres in object ...')  
            
            
        #sol = wavelength_shift_solution  # Keep this line as reference that wavelength_shift_solution was what we called "sol" in the previous version of the code
        fx = wavelength_shift_solution[0] + wavelength_shift_solution[1] * np.array(xfibre) + wavelength_shift_solution[2] * np.array(xfibre) ** 2

        if plot_wavelength_shift_correction_solution:   # Do we need to plot the correction again?     
            if len(wavelength_shift_solution) == 1: 
                ptitle = "Linear correction, y = "+str(round(wavelength_shift_solution[0],4))+" + "+str(round(wavelength_shift_solution[1],6))+"x"
            else:
                ptitle = "Second-order correction, y = {:.4f} + {:.6f}x + {:.3e}".format(wavelength_shift_solution[0],wavelength_shift_solution[1],wavelength_shift_solution[2])+"x$^{2}$"
            quick_plot(xfibre, fx, ptitle=ptitle, xlabel="Fibre", ylabel="offset [$\mathrm{\AA}$]", hlines=[0])
        
        if verbose: 
            print("  Polynomic solution:", wavelength_shift_solution)
            print("  offset_min = {:.3f} A ,  offset_max = {:.3f} A,  offset_difference = {:.3f} A".format(np.nanmin(fx), np.nanmax(fx),np.nanmax(fx)-np.nanmin(fx)))
    
        
        # Apply corrections to all fibres
        for fibre in range(nspec):  
            
            w_shift = fx[fibre]
            rss_out.intensity[fibre] = rebin_spec_shift(w, rss.intensity[fibre], w_shift)
            rss_out.variance[fibre] = rebin_spec_shift(w, rss.variance[fibre], w_shift)
            w_fixed = w - w_shift
    
            if plot:
                for line in sky_lines: 
                    if fibre in show_fibres_for_wavelength_shifts and line in show_skylines_for_wavelength_shifts:
                        ptitle = "Wavelength correction in Fibre " + str(fibre) + " in skyline "+str(line)
                        quick_plot([w, w_fixed, w], [rss.intensity[fibre], rss.intensity[fibre], rss_out.intensity[fibre]],
                                  xmin = line-20, xmax= line+20,
                                  xlabel="Wavelength [$\mathrm{\AA}$]",ylabel="Flux", ptitle=ptitle,
                                  vlines=[line], alpha=[0.2,0.2,0.6], 
                                  label= ["No corrected","No corrected - Shifted","Corrected after rebinning"],
                                  color=["r","b","g"],linewidth=[1,1,2],
                                  percentile_min=0.5, percentile_max=100, ncol = 3
                                  )
                
        if verbose: print("\n> Correction for the small wavelength shifts APPLIED!")
    
    
        #!!! TODO: log / history...
        #rss.log["wavelength fix"]["index"] = len(wavelength_shift_solution)
        #rss.log["wavelength fix"]["sol"] = [wavelength_shift_solution]
        #self.log_correction(rss, status='applied')      ##!!! ??????
        return rss_out



def rss_valid_wave_range(rss, **kwargs):
    """
    Provides the list of wavelengths with good values (non-nan) in edges.
    
    BE SURE YOU HAVE NOT CLEANED CCD DEFECTS if you are running this!!!

    Parameters
    ----------
    rss : object
        rss object.
    **kwargs : kwargs
        where we can find plot, verbose, warnings...

    Returns
    -------
    A list of lists:
        [0][0]: mask_first_good_value_per_fibre
        [0][1]: mask_last_good_value_per_fibre
        [1][0]: mask_max
        [1][1]: mask_min
        
        [[mask_first_good_value_per_fibre, mask_last_good_value_per_fibre],
         [mask_max, mask_min],
         [w[mask_max], w[mask_min]], 
         mask_list_fibres_all_good_values] 

    """
    
    verbose = kwargs.get('verbose', False)
    warnings = kwargs.get('warnings', False)
    plot =  kwargs.get('plot', False)
    
    w = rss.wavelength
    n_spectra = len(rss.intensity)
    n_wave = len(rss.wavelength)
    x = list(range(n_spectra))
    
    #  Check if file has 0 or nans in edges
    if np.isnan(rss.intensity[0][-1]):
        no_nans = False
    else:
        no_nans = True
        if rss.intensity[0][-1] != 0:
            if verbose or warnings: print(
                "  Careful!!! pixel [0][-1], fibre = 0, wave = -1, that should be in the mask has a value that is not nan or 0 !!!!!", **kwargs)

    if verbose and plot : print("\n  - Checking the left edge of the ccd...")

    mask_first_good_value_per_fibre = []
    for fibre in range(n_spectra):
        found = 0
        j = 0
        while found < 1:
            if no_nans:
                if rss.intensity[fibre][j] == 0:
                    j = j + 1
                else:
                    mask_first_good_value_per_fibre.append(j)
                    found = 2
            else:
                if np.isnan(rss.intensity[fibre][j]):
                    j = j + 1
                else:
                    mask_first_good_value_per_fibre.append(j)
                    found = 2
            if j > 101:
                vprint((" No nan or 0 found in the fist 100 pixels, ", w[j], " for fibre", fibre), **kwargs)
                mask_first_good_value_per_fibre.append(j)
                found = 2

    mask_max = np.nanmax(mask_first_good_value_per_fibre)
    if plot:        
        quick_plot(x, mask_first_good_value_per_fibre, ymax=mask_max + 1, xlabel="Fibre",
                  ptitle="Left edge of the RSS", hlines=[mask_max], ylabel="First good pixel in RSS")

    # Right edge, important for RED
    if verbose and plot :  print("\n- Checking the right edge of the ccd...")
    mask_last_good_value_per_fibre = []
    mask_list_fibres_all_good_values = []

    for fibre in range(n_spectra):
        found = 0
        j = n_wave - 1
        while found < 1:
            if no_nans:
                if rss.intensity[fibre][j] == 0:
                    j = j - 1
                else:
                    mask_last_good_value_per_fibre.append(j)
                    if j == len(rss.intensity[0]) - 1:
                        mask_list_fibres_all_good_values.append(fibre)
                    found = 2
            else:
                if np.isnan(rss.intensity[fibre][j]):
                    j = j - 1
                else:
                    mask_last_good_value_per_fibre.append(j)
                    if j == len(rss.intensity[0]) - 1:
                        mask_list_fibres_all_good_values.append(fibre)
                    found = 2

            if j < n_wave - 1 - 300:
                if verbose: print((" No nan or 0 found in the last 300 pixels, ", w[j], " for fibre", fibre))
                mask_last_good_value_per_fibre.append(j)
                found = 2

    mask_min = np.nanmin(mask_last_good_value_per_fibre)
    if plot:
        ptitle = "Fibres with all good values in the right edge of the RSS file : " + str(
            len(mask_list_fibres_all_good_values))
        quick_plot(x, mask_last_good_value_per_fibre, ymin=np.nanmin(mask_min),
                  ymax=2050, hlines=[2047], xlabel="Fibre", ylabel="Last good pixel in RSS", ptitle=ptitle)

    if verbose: 
        print("\n  --> The valid range for this RSS is {:.2f} to {:.2f} ,  in pixels = [ {} ,{} ]".format(w[mask_max],
                                                                                                    w[mask_min],
                                                                                                    mask_max,
                                                                                                    mask_min))

    # rss.mask = [mask_first_good_value_per_fibre, mask_last_good_value_per_fibre]
    # rss.mask_good_index_range = [mask_max, mask_min]
    # rss.mask_good_wavelength_range = [w[mask_max], w[mask_min]]
    # rss.mask_list_fibres_all_good_values = mask_list_fibres_all_good_values

        print("\n> Returning [ [mask_first_good_value_per_fibre, mask_last_good_value_per_fibre], ")
        print(  "              [mask_max, mask_min], ")
        print(  "              [w[mask_max], w[mask_min]], ")
        print(  "              mask_list_fibres_all_good_values ] ")
    
    # if verbose:
        # print("\n> Mask stored in rss.mask !")
        # print("  self.mask[0] contains the left edge, self.mask[1] the right edge")
        # print("  Valid range of the data stored in self.mask_good_index_range (index)")
        # print("                             and in self.mask_good_wavelength  (wavelenghts)")
        # print("  Fibres with all good values (in right edge) in self.mask_list_fibres_all_good_values")
    
    #return [rss.mask,rss.mask_good_index_range,rss.mask_good_wavelength_range,rss.mask_list_fibres_all_good_values]
    return [[mask_first_good_value_per_fibre, mask_last_good_value_per_fibre],
            [mask_max, mask_min],
            [w[mask_max], w[mask_min]], 
            mask_list_fibres_all_good_values ] 
    # if include_history:
    #     self.history.append("- Mask obtainted using the RSS file, valid range of data:")
    #     self.history.append(
    #         "  " + str(w[mask_max]) + " to " + str(w[mask_min]) + ",  in pixels = [ " + str(
    #             mask_max) + " , " + str(mask_min) + " ]")
    #     # -----------------------------------------------------------------------------





# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
