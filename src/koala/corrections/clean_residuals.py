"""

# TODO: THIS SCRIPT IS DEPRECATED.
"""

# =============================================================================
# Basics packages
# =============================================================================
import copy
import numpy as np
from scipy.signal import medfilt
# =============================================================================
# Astropy and associated packages
# =============================================================================
from ccdproc import cosmicray_lacosmic

# =============================================================================
# KOALA packages
# =============================================================================
# Modular
from koala.ancillary import vprint,interpolate_nan, nearest
from koala.mask import mask_section
from koala.rss import detect_edge

# Original
from koala.onedspec import find_cosmics_in_cut
from koala.ancillary import cut_wave # Moved provisionally to ancillary. Is a method od the RSS in the original code.
# =============================================================================


def fix_edges(rss,
              median_from=8800,
              median_to=6300,              
              verbose=False):
    """
    This function replaces the NaNs of the edges by the median value in the 
    region defined between the blue edge and the "meadian_to" and between 
    "median_from" and the red edge, respectively.
    """
    # Set print verbose
    vprint.verbose = verbose
    
    # Copy input RSS for storage the changes implemented in the task   
    rss_out = copy.deepcopy(rss)
    wavelength = rss.wavelength
    
    
    # Detect edges of the detector ()
    min_w, min_index,  max_w, max_index = detect_edge(rss)

    # median_to and median_from in index
    median_to_index = nearest(wavelength,median_to)
    median_from_index = nearest(wavelength,median_from)
    
# =============================================================================
# Blue edge
# =============================================================================
    lower = 0
    upper = min_index
    lower_mask = mask_section(rss, 
                          lower=lower,
                          upper=upper,
                          mask_value = 'nan',
                          verbose = verbose,
                          )
    
    rss_out.mask += (lower_mask * 2**1) # This mask has index 1 (2^1, i.e. maks value = 2)

    for i, f in enumerate(rss.intensity):
        median_value = np.nanmedian(f[min_index:median_to_index])
    
        rss_out.intensity_corrected[i][lower_mask[i]] = median_value
        
    comment = "\n - Blue edge has been checked up to " + str(min_w)+' Angstrom (pixel '+str(min_index)+')'
    rss_out.log['blue edge']['comment'] = comment
    vprint(comment)

        # if plot: plot_plot(w,[f,ff],vlines=[median_to,fix_to], xmax=median_to)
            
# =============================================================================
# Red edge
# =============================================================================
    lower = max_index
    upper = len(wavelength)
    upper_mask = mask_section(rss, 
                          lower=lower,
                          upper=upper,
                          mask_value = 'nan',
                          verbose = verbose,
                          )
    
    rss_out.mask += (upper_mask * 2**2) # This mask has index 2 (2^2, i.e. maks value = 4)

    for i, f in enumerate(rss.intensity):
        median_value = np.nanmedian(f[median_from_index:max_index])

        rss_out.intensity_corrected[i][upper_mask[i]] = median_value
        
        # if plot: plot_plot(w,[f,ff],vlines=[median_to,fix_to], xmax=median_to)

    comment = "\n - Red edge has been checked from " + str(max_w)+' Angstrom (pixel '+str(max_index)+')'
    rss_out.log['red edge']['comment'] = comment
    vprint(comment)
        

    return rss_out
    
"""    
    if plot:
        self.RSS_image(title=" - Before correcting edges") # PLOT BEFORE                                                                      
        self.RSS_image(title=" - After correcting edges") # PLOT AFTER


"""



def clean_nans(rss,
              verbose=False):
    # Set print verbose
    vprint.verbose = verbose
    
    # Copy input RSS for storage the changes implemented in the task   
    rss_out = copy.deepcopy(rss)
    
    rss_out.mask += np.isnan(rss.intensity_corrected) * 2**3  # This mask has index 3 (2^3, i.e. maks value = 8)
    
    for index, fiber in enumerate(rss.intensity_corrected):
        rss_out.intensity_corrected[index] = interpolate_nan(fiber)

    return rss_out





def kill_cosmics_2D(rss, 
                 verbose = False,
                 **kwargs
                 ):
    """
    """
    # Set print verbose
    vprint.verbose = verbose

    # Copy input RSS for storage the changes implemented in the task   
    rss_out = copy.deepcopy(rss)
    
    
# =============================================================================
# Construct a sky model from the 20 faintest fibers for subtracting in the 
# cosmic ray detection proccess   
# =============================================================================
    # Find 20 brigthtest fibers for constructing the sky spectra template 
    
    masked_data = np.ma.masked_array(rss.intensity,rss.mask)
    
    integrated_fibre = np.nansum(masked_data,axis=1).data # appliying the mask

    # brightest_20_fibers = (-integrated_fibre).argsort()[:20]
    # brightest_line_pixel = np.nanargmax(median_spectrum) # Take 10 brigtest and then median ??
    
    faintest_20_fibers = integrated_fibre.argsort()[:20]
                               
    # 2D sky model by scaling the 
    sky_median_model = np.nanmedian(rss.intensity_corrected[faintest_20_fibers],0)
    sky_2D_model = np.zeros_like(rss.intensity_corrected)    
    
    for index, fiber in enumerate(rss.intensity_corrected):
        proportion = sky_median_model / fiber
        scale_factor = np.nanmedian(proportion)
        sky_2D_model[index] = sky_median_model * scale_factor 
        
    #     from astropy.stats import sigma_clipped_stats
    # cut_mask = []
    #     cut = fiber[300:340]
    #     mean, median, stddev = sigma_clipped_stats(cut, sigma=3)
    #     cut_mask.append(fiber < (mean +(20*stddev)))
    # cut_mask = np.array(cut_mask)

    corrected_data, cosmic_mask = cosmicray_lacosmic(rss.intensity_corrected,
                                       gain_apply=False,
                                       inbkg=sky_2D_model,
                                       **kwargs) 

    rss_out.intensity_corrected = corrected_data    
    rss_out.mask += cosmic_mask * 2**4  # This mask has index 4 (2^4, i.e. maks value = 16)
    
    return rss_out




def kill_cosmics(rss,
                  brightest_line_wavelength,
                  width_bl=20.,
                  fibre_list=[],
                  max_number_of_cosmics_per_fibre=10,
                  kernel_median_cosmics=5,
                  cosmic_higher_than=100,
                  extra_factor=1.,
                  plot_waves=[],
                  plot_cosmic_image=True,
                  plot_RSS_images=True,
                  plot=True,
                  verbose=True,
                  warnings=True):
    """
    Kill cosmics in a RSS.

    Parameters
    ----------
    brightest_line_wavelength : float
        wavelength in A of the brightest emission line found in the RSS
    width_bl : float, optional
        broad in A of the bright emission line. The default is 20..
    fibre_list : array of integers, optional
        fibres to be modified. The default is "", that will be all
    kernel_median_cosmics : odd integer (default = 5)
        Width of the median filter
    cosmic_higher_than : float (default = 100)
        Upper boundary for pixel flux to be considered a cosmic
    extra_factor : float (default = 1.)
        Extra factor to be considered as maximum value
    plot_waves : list, optional
        list of wavelengths to plot. The default is none.
    plot_cosmic_image : boolean (default = True)
        Plot the image with the cosmic identification.
    plot_RSS_images : boolean (default = True)
        Plot comparison between RSS images before and after correcting cosmics
    plot : boolean (default = False)
        Display all the plots
    verbose: boolean (default = True)
        Print what is doing
    warnings: boolean (default = True)
        Print warnings

    Returns
    -------
    Save the corrected RSS to self.intensity_corrected

    """
    if plot == False:
        plot_cosmic_image = False

    # Set print verbose
    vprint.verbose = verbose
    
    # Copy input RSS for storage the changes implemented in the task   
    rss_out = copy.deepcopy(rss)

    # Valid wavelength range
    valid_wave_min, min_index, valid_wave_max, max_index = detect_edge(rss)




    x = range(rss.intensity.shape[0])
    w = rss.wavelength
    
    if len(fibre_list) == 0:
        fibre_list_ALL = True
        fibre_list = list(range(rss.intensity.shape[0]))
        if verbose: print("\n> Finding and killing cosmics in all fibres...")
    else:
        fibre_list_ALL = False
        if verbose: print("\n> Finding and killing cosmics in given fibres...")

    if brightest_line_wavelength == 0:
        if warnings or verbose: print("\n\n\n  WARNING !!!!! brightest_line_wavelength is NOT given!\n")


        
        integrated_fibre = np.nansum(rss.intensity_corrected,axis=1)
        brightest_10 = (-integrated_fibre).argsort()[:10]
        median_spectrum = np.nanmedian(rss.intensity_corrected[brightest_10], axis=0)


        # #self.integrated_fibre_sorted = np.argsort(self.integrated_fibre)
               
        # median_spectrum = plot_combined_spectrum(plot=plot, 
        #                                          median=True,
        #                                          list_spectra=integrated_fibre_sorted[-11:-1],
        #                                          ptitle="Combined spectrum using 10 brightest fibres",
        #                                          percentile_max=99.5, 
        #                                          percentile_min=0.5)
        
        
        
        # brightest_line_wavelength=w[np.int(self.n_wave/2)]
        brightest_line_wavelength = rss.wavelength[median_spectrum.tolist().index(np.nanmax(median_spectrum))]

        if brightest_line_wavelength < valid_wave_min: brightest_line_wavelength = valid_wave_min
        if brightest_line_wavelength > valid_wave_max: brightest_line_wavelength = valid_wave_max

        if warnings or verbose: print(
            "  Assuming brightest_line_wavelength is the max of median spectrum of 10 brightest fibres =",
            brightest_line_wavelength)




    # Get the cut at the brightest_line_wavelength
    corte_wave_bl = cut_wave(rss,brightest_line_wavelength)

    # if plot:
    #     gc_bl = medfilt(corte_wave_bl, kernel_size=kernel_median_cosmics)
    #     max_val = np.abs(corte_wave_bl - gc_bl)

    #     ptitle = "Intensity cut at brightest line wavelength = " + np.str(
    #         np.round(brightest_line_wavelength, 2)) + " $\mathrm{\AA}$ and extra_factor = " + np.str(extra_factor)
    #     plot_plot(x, [max_val, extra_factor * max_val], percentile_max=99, xlabel="Fibre", ptitle=ptitle,
    #               ylabel="abs (f - medfilt(f))",
    #               label=["intensity_cut", "intensity_cut * extra_factor"])

    # List of waves to plot:
    plot_waves_index = []
    for wave in plot_waves:
        wave_min_vector = np.abs(w - wave)
        plot_waves_index.append(wave_min_vector.tolist().index(np.nanmin(wave_min_vector)))
    if len(plot_waves) > 0: print("  List of waves to plot:", plot_waves)

    # Start loop
    lista_cosmicos = []
    cosmic_image = np.zeros_like(rss.intensity_corrected)
    for i in range(len(w)):
        wave = w[i]
        # Perhaps we should include here not cleaning in emission lines...
        correct_cosmics_in_fibre = True
        if width_bl != 0:
            if wave > brightest_line_wavelength - width_bl / 2 and wave < brightest_line_wavelength + width_bl / 2:
                if verbose: print(
                    "  Skipping {:.4f} as it is adjacent to brightest line wavelenght {:.4f}".format(wave,
                                                                                                      brightest_line_wavelength))
                correct_cosmics_in_fibre = False
        if correct_cosmics_in_fibre:
            if i in plot_waves_index:
                plot_ = True
                verbose_ = True
            else:
                plot_ = False
                verbose_ = False
            corte_wave = cut_wave(rss,wave)
            cosmics_found = find_cosmics_in_cut(x, corte_wave, corte_wave_bl * extra_factor, line_wavelength=wave,
                                                plot=plot_, verbose=verbose_, cosmic_higher_than=cosmic_higher_than)
            if len(cosmics_found) <= max_number_of_cosmics_per_fibre:
                for cosmic in cosmics_found:
                    lista_cosmicos.append([wave, cosmic])
                    cosmic_image[cosmic, i] = 1.
            else:
                if warnings: print("  WARNING! Wavelength", np.round(wave, 2), "has", len(cosmics_found),
                                    "cosmics found, this is larger than", max_number_of_cosmics_per_fibre,
                                    "and hence these are NOT corrected!")

    # Check number of cosmics found
    #if plot_cosmic_image: self.RSS_image(image=cosmic_image, cmap="binary_r", title=" - Cosmics identification")

    vprint("\n> Total number of cosmics found = ", len(lista_cosmicos), " , correcting now ...")

    #if plot_RSS_images: self.RSS_image(cmap="binary_r", title=" - Before correcting cosmics")

    if fibre_list_ALL == False and verbose == True: print("  Correcting cosmics in selected fibres...")
    cosmics_cleaned = 0
    
    
    for fibre in fibre_list:
        if np.nansum(cosmic_image[fibre]) > 0:  # A cosmic is found
            # print("Fibre ",fibre," has cosmics!")
            f = rss_out.intensity_corrected[fibre]
            gc = medfilt(f, kernel_size=21)
            bad_indices = [i for i, x in enumerate(cosmic_image[fibre]) if x == 1]
            if len(bad_indices) <= max_number_of_cosmics_per_fibre:
                for index in bad_indices:
                    rss_out.intensity_corrected[fibre, index] = gc[index]
                    rss_out.mask[fibre, index] += 2**4  # This mask has index 4 (2^4, i.e. maks value = 16)

                    
                    cosmics_cleaned = cosmics_cleaned + 1
                    
                    
            else:
                cosmic_image[fibre] = np.zeros_like(w)
                if warnings: print("  WARNING! Fibre", fibre, "has", len(bad_indices),
                                    "cosmics found, this is larger than", max_number_of_cosmics_per_fibre,
                                    "and hence is NOT corrected!")

    #if plot_RSS_images: self.RSS_image(cmap="binary_r", title=" - After correcting cosmics")

    # Check number of cosmics eliminated
    vprint("\n> Total number of cosmics cleaned = ", cosmics_cleaned)
    
    #if cosmics_cleaned != len(lista_cosmicos):
    #    if plot_cosmic_image: self.RSS_image(image=cosmic_image, cmap="binary_r", title=" - Cosmics cleaned")

    #self.history.append("- " + np.str(cosmics_cleaned) + " cosmics cleaned using:")
    #self.history.append("  brightest_line_wavelength = " + np.str(brightest_line_wavelength))
    #self.history.append("  width_bl = " + np.str(width_bl) + ", kernel_median_cosmics = " + np.str(kernel_median_cosmics))
    #self.history.append("  cosmic_higher_than = " + np.str(cosmic_higher_than) + ", extra_factor = " + np.str(extra_factor))
    
    return rss_out


def extreme_negatives(rss, 
                      fibre_list='all',
                      percentile_min=0.5,
                      plot=True,
                      verbose=True):
    """
    Remove pixels that have extreme negative values (that is below percentile_min) and replace for the median value

    Parameters
    ----------
    fibre_list : list of integers (default all)
        List of fibers to clean. The default is [], that means it will do everything.
    percentile_min : float, (default = 0.5)
        Minimum value accepted as good.
    plot : boolean (default = False)
        Display all the plots
    verbose: boolean (default = True)
        Print what is doing
    """
       
    # Set print verbose
    vprint.verbose = verbose
    

# =============================================================================
# Copy input RSS for storage the changes implemented in the task   
# =============================================================================
    rss_out = copy.deepcopy(rss)

    if fibre_list == 'all':
        fibre_list = list(range(rss_out.intensity.shape[0]))
        vprint("\n> Correcting the extreme negatives in all fibres, making any pixel below")
    else:
        vprint("\n> Correcting the extreme negatives in given fibres, making any pixel below")


    minimo = np.nanpercentile(rss_out.intensity, percentile_min)

    vprint("  np.nanpercentile(intensity_corrected, ", percentile_min, ") = ", np.round(minimo, 2))
    vprint("  to have the median value of the fibre...")

    
     # Masking by fibre 
    for fibre in fibre_list:
        fibre_mask = rss.intensity[fibre]<minimo
        rss_out.intensity_corrected[fibre][fibre_mask] = np.nanmedian(rss_out.intensity[fibre])
        rss_out.mask[fibre][fibre_mask] += 2**5 # This mask has index 4 (2^5, i.e. maks value = 32)
        
        
    comment = "- Extreme negatives (values below percentile " + str(np.round(percentile_min, 3)) + " = " + str(np.round(minimo, 3)) + " ) cleaned"    
    rss_out.log['extreme negative']['comment'] = comment
                

    """
    if plot:
        correction_map = g / self.intensity_corrected  # / g
        self.RSS_image(image=correction_map, cmap="binary_r", title=" - CorrectionBase map")
    """

    return rss_out








