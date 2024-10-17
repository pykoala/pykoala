import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pykoala.plotting.quick_plot import quick_plot #, basic_statistics

# Fuego color map
#from matplotlib import pyplot as plt
#import matplotlib.colors as colors

fuego_color_map = colors.LinearSegmentedColormap.from_list("fuego", 
                                                            ((0.25, 0, 0),  
                                                            (0.5,0,0),    
                                                            (1, 0, 0), 
                                                            (1, 0.5, 0), 
                                                            (1, 0.75, 0), 
                                                            (1, 1, 0), 
                                                            (1, 1, 1)), 
                                                            N=256, gamma=1.0)
fuego_color_map.set_bad('lightgray')  #('black')
#plt.register_cmap(cmap=fuego_color_map)  
# #-----------------------------------------------------------------------------
# %% ============================================================================
# #-----------------------------------------------------------------------------

def plot_map():
    """
    Task for plotting a map from a cube. Done in Angel's koala_cube.py but need to be implemented here. #TODO
    
    Returns
    -------
    None.

    """    
    pass
# #-----------------------------------------------------------------------------
# %% ============================================================================
# #-----------------------------------------------------------------------------
def plot_wavelength(rss, wavelength, r = False, **kwargs):
    """
    Plot cut at a particular wavelength.

    Parameters
    ----------
    rss: 
        RSS object
    wavelength: int (index) or float
        wavelength to cut. If it is an integer and wavelength < n_wave, it is considered and index.
    r: Boolean    
        If True, return cut
    
    Example
    -------
    >>> plot_wavelength(rss, 7000, r =True)
    """
    w = rss.wavelength.value
    # Check if wavelength is a index or a wavelength
    n_wave = len(rss.wavelength)
    if wavelength < n_wave and type(wavelength) is int:  # wavelength is an index
        wave_index = wavelength
    else:
        wave_index= np.searchsorted(w,wavelength)
           
    wave = w[wave_index]       
    corte_wave = rss.intensity[:, wave_index].value
    
    if kwargs.get("plot") is None: plot = True
    
    if plot:
        x = range(len(rss.intensity))
        if kwargs.get("xlabel") is None:
            kwargs["xlabel"] = "Fibre"  
        if kwargs.get("ptitle") is None:
            kwargs["ptitle"] = "Intensity cut at " + str(round(wave,2)) + " $\mathrm{\AA}$ - index =" + str(wave_index)
        quick_plot(x, corte_wave, **kwargs)
    
    if r: return corte_wave
# #-----------------------------------------------------------------------------
# %% ============================================================================
# #-----------------------------------------------------------------------------
def get_spectrum(data_container, 
                 r = True, median=False,
                 # Parameters for RSS
                 fibre = None, list_fibre=None,   
                 # Parameters for CUBE
                 spaxel = None, spaxel_list = None, show_map = False,
                 **kwargs):
    """
    This task should replace tasks plot_spectrum and plot_combined_spectrum in rss and
    task spectrum_of_cube and spectrum_of_spaxel in cube.

    Parameters
    ----------
    data_container : TYPE
        DESCRIPTION.
    r : TYPE, optional
        DESCRIPTION. The default is False.
    median : TYPE, optional
        DESCRIPTION. The default is False.
    # Parameters for RSS             fibre : TYPE, optional
        DESCRIPTION. The default is None.
    list_fibre : TYPE, optional
        DESCRIPTION. The default is None.
    # Parameters for CUBE             spaxel : TYPE, optional
        DESCRIPTION. The default is None.
    spaxel_list : TYPE, optional
        DESCRIPTION. The default is None.
    show_map : TYPE, optional
        DESCRIPTION. The default is False.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    if r is True, return the spectrum.

    """
    
    verbose = kwargs.get('verbose', False)
    #warnings = kwargs.get('warnings', verbose)
    plot = kwargs.get('plot', False)
    ptitle = kwargs.get('ptitle', None)
    
    info = data_container.info
    wavelength = data_container.wavelength.value
    intensity =  data_container.intensity.value
    
    if str(type(data_container))[-5:-2] == "RSS":     # It is a RSS

        if fibre is not None:
            spectrum = intensity[fibre]
            if ptitle is None: kwargs["ptitle"] = "Spectrum of fibre {} in {}".format(fibre, info["name"])
            
        else:
            if list_fibre is None:  list_fibre = list(range(len(intensity)))
            value_list = [intensity[fibre] for fibre in list_fibre]
            if median:
                spectrum = np.nanmedian(value_list, axis=0)
            else:
                spectrum = np.nansum(value_list, axis=0)
            if ptitle is None:
                if len(list_fibre) == list_fibre[-1] - list_fibre[0] + 1:
                    kwargs["ptitle"] = "{} - Combined spectrum in range [ {} , {} ]".format(info["name"],list_fibre[0], list_fibre[-1])
                else:
                    kwargs["ptitle"] = "Combined spectrum using requested fibres"
            if verbose: 
                if median:
                    print("\n> Median spectrum of selected {} spaxels...".format(len(value_list)))
                else:    
                    print("\n> Integrating spectrum of selected {} spaxels...".format(len(value_list)))
        
    else:               # It is a cube
 
        if spaxel is None and spaxel_list is None:          # If these are not provided, plot integrated spectrum
        
            if median:
                if verbose: print("\n> Computing the median spectrum of the cube...")
                spectrum=np.nanmedian(np.nanmedian(intensity, axis=1),axis=1)
                if ptitle is None : kwargs['ptitle'] = "Median spectrum in {}".format(info["name"])
            else:
                if verbose: print("\n> Computing the integrated spectrum of the cube...")
                spectrum=np.nansum(np.nansum(intensity, axis=1),axis=1)
                if ptitle is None: kwargs['ptitle'] = "Integrated spectrum in {}".format(info["name"])
        
        elif spaxel_list is not None:
            
            n_spaxels = len(spaxel_list)
            if verbose: 
                if median:
                    print("\n> Median spectrum of selected {} spaxels...".format(n_spaxels))
                else:    
                    print("\n> Integrating spectrum of selected {} spaxels...".format(n_spaxels))
            list_of_spectra=[]        
            for i in range(n_spaxels):
                #if verbose: print("  Adding spaxel  {} : [ {}, {} ]".format(i+1,spaxel_list[i][0],spaxel_list[i][1]))
                list_of_spectra.append( [intensity[:,spaxel_list[i][1], spaxel_list[i][0]] ])
                                     
            if median:                
                spectrum=np.nanmedian(list_of_spectra, axis=0)[0]
                if ptitle is None: kwargs['ptitle'] = "Median spectrum adding {} spaxels in {}".format(n_spaxels, info["name"])
            else:    
                spectrum=np.nansum(list_of_spectra, axis=0)[0]
                if ptitle is None: kwargs['ptitle'] = "Integrated spectrum adding {} spaxels in {}".format(n_spaxels, info["name"])
        
        else:
            x,y = spaxel[1],spaxel[0]   
            spectrum = intensity[:,y,x]
            if ptitle is None : kwargs['ptitle'] = "Spaxel [{},{}] in {}".format(x,y, info["name"])
        
        if show_map: plot_map(data_container, spaxel = spaxel, spaxel_list=spaxel_list, **kwargs)
  
    if plot:        
        quick_plot(wavelength, spectrum,
                  #ptitle = ptitle,
                  #vlines=[kinfo["valid_wave_min"],kinfo["valid_wave_max"]],
                  **kwargs)    
    
    if r: return spectrum 

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def plot_spectrum(rss, spectrum_number, r = False, **kwargs):
    """
    Plot spectrum of a particular fibre.

    Parameters
    ----------
    rss: 
        RSS object
    spectrum_number: int
        fibre to show spectrum.
    r: Boolean    
        If True, return spectrum
    
    Example
    -------
    >>> plot_spectrum(rss, 550)
    """
    wavelength = rss.wavelength
    spectrum = rss.intensity[spectrum_number]
    if kwargs.get("ptitle") is None:
        kwargs["ptitle"] = rss.info["name"] + " - Spectrum: "+str(spectrum_number)
    quick_plot(wavelength, spectrum, **kwargs)
    if r: return spectrum
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_combined_spectrum(rss, list_spectra=None,  median=False, r = False, **kwargs):
    """
    Plot combined spectrum of a list and return the combined spectrum.

    Parameters
    ----------
    rss: 
        RSS object
    list_spectra:
        spaxels to show combined spectrum. Default is all.
    median : boolean (default = False)
        if True the combined spectrum is the median spectrum
        if False the combined spectrum is the sum of the list of spectra
    r: Boolean    
        If True, return spectrum
    
    Example
    -------
    >>> star_spectrum = plot_spectrum(rss, list_spectra= [550:550], r = True)
    """
    if list_spectra is None:  list_spectra = list(range(len(rss.intensity)))

    value_list = [rss.intensity[fibre] for fibre in list_spectra]
    
    if median:
        spectrum = np.nanmedian(value_list, axis=0)
    else:
        spectrum = np.nansum(value_list, axis=0)

    plot= kwargs.get("plot", True)
 
    if plot:
        #vlines = [self.valid_wave_min, self.valid_wave_max]   #TODO  # It would be good to indicate the good wavelength range, but now these are not saved in RSS object
        if kwargs.get("ptitle") is None:
            if len(list_spectra) == list_spectra[-1] - list_spectra[0] + 1:
                kwargs["ptitle"] = "{} - Combined spectrum in range [{},{}]".format(rss.info["name"],list_spectra[0], list_spectra[-1])
            else:
                kwargs["ptitle"] = "Combined spectrum using requested fibres"
        quick_plot(rss.wavelength, spectrum,  **kwargs)
                  
    if r: return spectrum       
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def rss_image(rss=None, image=None, log=False, gamma=0,
              cmap="seismic_r", clow=None, chigh=None, percentile_min = 5, percentile_max=95,
              greyscale = False, greyscale_r = False,
              xmin = None, xmax= None, wmin = None, wmax= None, fmin=0, fmax=None, 
              title_fontsize=12, title=None, add_title = False,
              xlabel = "Wavelength vector", ylabel = "Fibre", 
              axes_label_fontsize=10, axes_ticksize=None, axes_fontsize=10, axes_thickness=0, 
              colorbar_label_fontsize = 10, colorbar_ticksize= 10, colorbar_rotation = 270,  
              colorbar_width_fraction=None, colorbar_pad=None, colorbar_labelpad=10,
              colorbar_label="Intensity [Arbitrary units]", 
              fig_size=8, save_plot_in_file = None, 
              **kwargs):
    """
    Plot image of rss coloured by variable. 
    By default the cmap is "seismic_r" using symetric scale [-max_abs, max_abs],
    where max_abs = np.nanmax([np.abs(clow), np.abs(chigh)])
    
    Try greyscale=True or cmap = "binary_r" for a nice greyscale map.

    Parameters
    ----------
    rss : object
        object with the rss information.
    image : string (default = none)
        Specify the name of saved RSS image. If not, it uses rss.intensity
    log and gamma:
        Normalization scale, default is lineal scale.
        Lineal scale: norm=colors.Normalize().
        Log scale:    norm=colors.LogNorm()
        Power law:    norm=colors.PowerNorm(gamma=1./4.)  if gamma given
    log : Boolean, optional
        Plot in log scale. The default is False.
    gamma : float, optional
        Plot using Power law if gamma > 0. The default is 0.
    cmap : string 
        Colour map for the plot. The default is "seismic_r".
    clow : float, optional
        Lower bound, if not specified, 5th percentile is chosen. The default is None.
    chigh : float, optional
        Higher bound, if not specified, 95th percentile is chosen. The default is None.
    greyscale : Boolean, optional
        Use nice cmap = "binary_r" for greyscale. The default is False.
    xmin : integer, optional
        minimum value of x to plot. The default is None, that is [0]
    xmax : integer, optional
        maximum value of x to plot. The default is None, that is [-1]
    wmin : float, optional
        minimum value of wavelength to plot. If both wmin and wmax are given, the x-axis is shown in Angstroms. The default is None.
    wmax : float, optional
        maximum value of wavelength to plot. If both wmin and wmax are given, the x-axis is shown in Angstroms. The default is None.
    fmin : integer, optional
        minimum fibre to plot. The default is 0.
    fmax : integer, optional
        maximum fibre to plot. The default is None, that is the last fibre
    title_fontsize : float, optional
        Size of the font in title. The default is 12.
    title : string, optional
        Title of the plot. The default is rss.info['description'] + " - RSS image"
    xlabel : string, optional
        Label of the x-axis. The default is "Wavelength vector", but if both wmin and wmax are given, it changes to "Wavelength [$\mathrm{\AA}$]".
    ylabel : string, optional
        Label of the y-axis. The default is "Fibre".
    axes_label_fontsize : float, optional
        Size of the font of the labels in axes. The default is 10.
    axes_ticksize : list of 4 float, optional
        Size of the ticks of the axes. The default is None, that is [10,1,5,1]
    axes_fontsize : float, optional
        Size of the font of the axes. The default is 10.
    axes_thickness : float, optional
        Thickness of the axes. The default is 0.
    colorbar_label_fontsize : float, optional
        Size of the label in the colorbar. The default is 10.
    colorbar_ticksize : float, optional
        Size of the ticks in the colorbar. The default is 14.
    colorbar_rotation : float, optional
        Rotation of the colorbar. The default is 270.
    colorbar_width_fraction : float, optional
        Fraction of the plot used for the width of the colorbar. The default is None, that is 0.03 or using aspect_ratio=len(x)/len(y)
    colorbar_pad : float, optional
        Space between main figure an colorbar, in a fraction of figure. The default is None, that is 0.01, or using aspect_ratio=len(x)/len(y)
    colorbar_labelpad : float, optional
        Space between the colorbar and its label. The default is 10.
    colorbar_label : String, optional
        Label of the colorbar. The default is "Intensity [Arbitrary units]".
    fig_size : float or string, optional
        Size of the figure. If "big" it will used a specified big size. The default is 8. 
    save_plot_in_file : string, optional
        filename of the image to be saved. If not None, the image will be saved. Include ".png", ".eps" or ".pdf" at the end to be sure it is saved in the desired format. The default is None.
    **kwargs : dictionary
        list of kwargs.

    Returns
    -------
    None.
    """
    verbose = kwargs.get('verbose', False)
    #warnings = kwargs.get('warnings', verbose)
    #plot =  kwargs.get('plot', False)
    
    if rss is None and image is None: raise RuntimeError("No rss or image provided!!")
    
    # Check if image is given
    if image is None:
        image = rss.intensity
        
    # Check if rss is given
    if rss is None:
        wavelength = np.arange(len(image[0]))
        name = "IMAGE "
    else:
        wavelength = rss.wavelength
        name = rss.info['name']
        
    # Check color visualization
    norm=colors.LogNorm()
    if log is False: norm = colors.Normalize()
    if gamma > 0: norm=colors.PowerNorm(gamma=gamma)
    if greyscale: cmap = "binary_r" 
    if greyscale_r: cmap = "binary" 

    #  Set color scale and map   
    if clow is None:
        clow = np.nanpercentile(image, percentile_min)
    if chigh is None:
        chigh = np.nanpercentile(image, percentile_max)
    if cmap == "seismic_r" and log == False:
        max_abs = np.nanmax([np.abs(clow), np.abs(chigh)])
        clow = -max_abs
        chigh = max_abs
    if log and clow <=0:
        clow_=clow
        clow = np.nanmin(np.abs(image))
        if clow == 0 : 
            if chigh > 100 : 
                clow = 1.0
            else:
                clow = 0.0001
        if verbose: print("\n> Plotting image in log but the lowest value is {:.2f}, using the minimum positive value of {:.2e} instead.".format(clow_,clow))  
    
    # Show only a subregion if requested
    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = len(wavelength) 
    if wmin is not None:
        xmin = np.searchsorted(wavelength, wmin)
        if xmin == 0 : wmin = wavelength[0]
    if wmax is not None:
        xmax = np.searchsorted(wavelength, wmax)
        if xmax == len(wavelength)-1 : wmax = wavelength[-1]
    if wmin is not None and wmax is not None:
        extent1 = wmin
        extent2 = wmax
        if xlabel == "Wavelength vector": xlabel="Wavelength [$\mathrm{\AA}$]"
    else:
        extent1 = xmin
        extent2 = xmax
        
    if fmax is None: 
        fmax = len(image)
    else:
        if fmax > len(image) - 1 : fmax = len(image)
    
    extent3 = fmax
    extent4 = fmin    
    image= image[fmin:fmax,xmin:xmax]    
    
    # Check if we are saving in file to increase fig size
    if save_plot_in_file is not None:
        if type(fig_size) is not str: fig_size = fig_size * 3
        title_fontsize = title_fontsize *2.1
        axes_label_fontsize= axes_label_fontsize *2.1
        axes_fontsize=axes_fontsize *1.7
        colorbar_label_fontsize = colorbar_label_fontsize * 1.8
        colorbar_ticksize = colorbar_ticksize * 2.8
        axes_thickness = 2
        if axes_ticksize == None: axes_ticksize = [10,1,5,1]

    if type(fig_size) is str: 
        if fig_size == "big":
            fig_size=20
            title_fontsize=22
            axes_label_fontsize=18
            axes_fontsize=15
            colorbar_label_fontsize = 15
            colorbar_ticksize = 14
            axes_ticksize=[10,1,5,1]
            axes_thickness =2
     
    if axes_ticksize == None: axes_ticksize = [5,1,2,1]
    
    # Compute aspect ratio & prepare plot
    aspect_ratio = (xmax-xmin) / (fmax-fmin)
    if aspect_ratio > 1:
        fig, ax = plt.subplots(figsize=(fig_size, fig_size*aspect_ratio))
    else:        
        fig, ax = plt.subplots(figsize=(fig_size/aspect_ratio, fig_size))        
    
    # Create plot including minorticks
    pmapa=ax.imshow(image, norm=norm, cmap=cmap, clim=(clow, chigh), extent=(extent1, extent2, extent3, extent4))    
    plt.gca().invert_yaxis()    
    plt.minorticks_on()
    
    # Include labels
    plt.xlabel(xlabel, fontsize=axes_label_fontsize)
    plt.ylabel(ylabel, fontsize=axes_label_fontsize)    
    plt.tick_params('both', length=axes_ticksize[0], width=axes_ticksize[1], which='major')
    plt.tick_params('both', length=axes_ticksize[2], width=axes_ticksize[3], which='minor')
    plt.tick_params(labelsize=axes_fontsize)
    
    # Include title
    if title is not None:
        if add_title:
            plt.title(name + str(title), fontsize=title_fontsize)
        else:
            plt.title(str(title), fontsize=title_fontsize)
    else:
        plt.title(name + " - RSS image", fontsize=title_fontsize)
        
    # Making axes thicker if requested
    if axes_thickness > 0:
        plt.axhline(y=fmin,linewidth=axes_thickness, color="k")     
        plt.axvline(x=xmin,linewidth=axes_thickness, color="k")    
        plt.axhline(y=fmax,linewidth=axes_thickness, color="k")    
        plt.axvline(x=xmax,linewidth=axes_thickness, color="k")   

    # # Include colorbar well aligned with the map    
    if 0.8  <= aspect_ratio <= 1.2:
        bth = 0.05
        gap = 0.03
    else:
        bth = 0.03
        gap = 0.02    
    if colorbar_width_fraction is None: colorbar_width_fraction = np.nanmin((0.03, np.max((aspect_ratio * bth, bth))))
    if colorbar_pad is None: colorbar_pad = np.nanmin((0.01, np.max((aspect_ratio * gap, gap))))
    cax  = ax.inset_axes((1+colorbar_pad, 0, colorbar_width_fraction , 1))
    cbar = fig.colorbar(pmapa,cax=cax) 
    cbar.ax.tick_params(labelsize=colorbar_ticksize)
    cbar.set_label(str(colorbar_label), rotation=colorbar_rotation, labelpad=colorbar_labelpad, fontsize=colorbar_label_fontsize)


    # Show or save final plot
    if save_plot_in_file is None:
            plt.show()
            plt.close() 
    else:
       #if path != "" : save_file=full_path(save_file,path)
       plt.savefig(save_plot_in_file, bbox_inches='tight')
       plt.close() 
       if verbose: print("  Figure saved in file",save_plot_in_file)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def compare_rss_images(rss, image_list=None, log=False, gamma=0,
              cmap="seismic_r", clow=None, chigh=None, percentile_min = 5, percentile_max=95,
              greyscale = False, greyscale_r = False,
              xmin = None, xmax= None, wmin = None, wmax= None, fmin=0, fmax=None, 
              title_fontsize=12, title=None, add_title = False,
              xlabel = "Wavelength vector", ylabel = "Fibre", 
              axes_label_fontsize=10, axes_ticksize=None, axes_fontsize=10, axes_thickness=0, 
              colorbar_label_fontsize = 10, colorbar_ticksize= 10, colorbar_rotation = 270,  
              colorbar_width_fraction=None, colorbar_pad=None, colorbar_labelpad=10,
              colorbar_label="Intensity [Arbitrary units]", 
              fig_size=[8,6], save_plot_in_file = None, 
              **kwargs):
    verbose = kwargs.get('verbose', False)
    #warnings = kwargs.get('warnings', verbose)
    #plot =  kwargs.get('plot', False)
    
    # Show only a subregion if requested
    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = len(rss.wavelength) 
    if wmin is not None:
        xmin = np.searchsorted(rss.wavelength, wmin)
        if xmin == 0 : wmin = rss.wavelength[0]
    if wmax is not None:
        xmax = np.searchsorted(rss.wavelength, wmax)
        if xmax == len(rss.wavelength)-1 : wmax = rss.wavelength[-1]
    if wmin is not None and wmax is not None:
        extent1 = wmin
        extent2 = wmax
        if xlabel == "Wavelength vector": xlabel="Wavelength [$\mathrm{\AA}$]"
    else:
        extent1 = xmin
        extent2 = xmax
    if fmax is None: 
        fmax = len(rss.intensity)
    else:
        if fmax > len(rss.intensity) - 1 : fmax = len(rss.intensity)
    extent3 = fmax
    extent4 = fmin    
    
    # Check if we are saving in file to increase fig size
    if save_plot_in_file is not None:
        if type(fig_size) is not str: fig_size = fig_size * 3
        title_fontsize = title_fontsize *2.1
        axes_label_fontsize= axes_label_fontsize *2.1
        axes_fontsize=axes_fontsize *1.7
        colorbar_label_fontsize = colorbar_label_fontsize * 1.8
        colorbar_ticksize = colorbar_ticksize * 2.8
        #axes_thickness = 2
        if axes_ticksize == None: axes_ticksize = [10,1,5,1]

    if type(fig_size) is str: 
        if fig_size == "big":
            fig_size=20
            title_fontsize=22
            axes_label_fontsize=18
            axes_fontsize=15
            colorbar_label_fontsize = 15
            colorbar_ticksize = 14
            axes_ticksize=[10,1,5,1]
            #axes_thickness =2
     
    if axes_ticksize == None: axes_ticksize = [5,1,2,1]


    # Check position of colorbars
    if colorbar_pad is None: colorbar_pad =0.05
    if colorbar_width_fraction is None: colorbar_width_fraction =0.15
    
    # Start figure with sublots   
    a = 1   # All horizontal
    n_plots = len (image_list)
    fig, axs = plt.subplots(a, n_plots , figsize=(fig_size[0], fig_size[1]),
                            constrained_layout=True, sharey=True,
                            sharex=True)
    
    if type(greyscale) is bool: greyscale = [greyscale]*n_plots
    if type(greyscale_r) is bool: greyscale_r = [greyscale_r]*n_plots  
    if type(gamma) is int or float : gamma = [gamma] *n_plots 
    
    i = 0  # Looping the plots
    for col in range(n_plots):
        #for row in range(a):  #### IF in the future we need to expand this...
            
        data = image_list[i][fmin:fmax,xmin:xmax]            
        
        # Check color visualization
        norm=colors.LogNorm()
        if log[i] is False: norm = colors.Normalize()
        if gamma[i] > 0: norm=colors.PowerNorm(gamma=gamma[i])
        if greyscale[i]: cmap[i] = "binary_r" 
        if greyscale_r[i]: cmap[i] = "binary" 
    
        #  Set color scale and map   
        if clow[i] is None:
            clow[i] = np.nanpercentile(data, percentile_min)
        if chigh[i] is None:
            chigh[i] = np.nanpercentile(data, percentile_max)
        if cmap[i] == "seismic_r" and log[i] == False:
            max_abs = np.nanmax([np.abs(clow[i]), np.abs(chigh[i])])
            clow[i] = -max_abs
            chigh[i] = max_abs
        if log[i] and clow[i] <=0:
            clow[i] = np.nanmin(np.abs(data))
            if clow[i] == 0 : 
                if chigh[i] > 100 : 
                    clow[i] = 1.0
                else:
                    clow[i] = 0.0001
          
        # Create subplot including minorticks
        ax = axs[col]
        pcm=ax.imshow(data, norm=norm, cmap=cmap[i], clim=(clow[i], chigh[i]), extent=(extent1, extent2, extent3, extent4))    
        plt.minorticks_on()
        plt.gca().invert_yaxis()   # Invert axis in y 
        
        plt.tick_params('both', length=axes_ticksize[0], width=axes_ticksize[1], which='major')
        plt.tick_params('both', length=axes_ticksize[2], width=axes_ticksize[3], which='minor')
        
        # Include labels
        ax.set_xlabel(xlabel,  fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13) 
        ax.set_title(title[i])  ##, fontsize=title_fontsize)
        
        # Color bar
        cax  = ax.inset_axes((1+colorbar_pad, 0, colorbar_width_fraction , 1))
        cbar = fig.colorbar(pcm,cax=cax) 
        #cbar.ax.tick_params(labelsize=colorbar_ticksize)
        if colorbar_label[i] is not None: cbar.set_label(str(colorbar_label[i])) #, rotation=colorbar_rotation, labelpad=colorbar_labelpad, fontsize=colorbar_label_fontsize)

        i=i+1 # Prepare for next plot

    
            
    # Show or save final plot
    if save_plot_in_file is None:
            plt.show()
            plt.close() 
    else:
       #if path != "" : save_file=full_path(save_file,path)
       plt.savefig(save_plot_in_file, bbox_inches='tight')
       plt.close() 
       if verbose: print("  Figure saved in file",save_plot_in_file)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def rss_map(rss, variable=None, list_spectra=None, 
            instrument = None,
            clow=None, chigh=None,
            log=True, floor = None, gamma=0, cmap = None,
            title=" - RSS map",  description = None,
            xlabel = None, ylabel = None, 
            color_bar_text="Integrated Flux [Arbitrary units]", 
            figsize = 8, grid=False,
            colorbar_labelpad = 10, colorbar_pad= None, colorbar_width_fraction= None,
            save_plot_in_file = None,
            **kwargs):
    """
    Plot map showing the offsets, coloured by variable.

    Parameters
    ----------
    norm:
        Normalization scale, default is lineal scale.
        Lineal scale: norm=colors.Normalize()
        Log scale:    norm=colors.LogNorm()
        Power law:    norm=colors.PowerNorm(gamma=1./4.)
    list_spectra : list of floats (default = none)
        List of RSS spectra
    title : string (default = " - RSS image")
        Set plot title
    color_bar_text : string (default = "Integrated Flux [Arbitrary units]")
        Colorbar's label text
    """
    verbose = kwargs.get('verbose', False)
    warnings = kwargs.get('warnings', verbose)

    if variable is None: 
        variable, _ = rss.get_integrated_fibres()

    if cmap is None:
        cmap = fuego_color_map

    if gamma > 0: 
        norm=colors.PowerNorm(gamma=gamma)
    elif log:
        norm=colors.LogNorm()     
        min_value = np.nanmin(variable)
        if min_value <= 0:
            if floor is None:
                floor = np.nanmin(np.abs(variable))
                if floor == 0: floor = 0.1
                variable = np.array([np.nan if x <= floor else x for x in variable])
                negative_spaxels = np.count_nonzero(np.isnan(variable))
                if warnings: print(f"> WARNING: The image has {negative_spaxels} spaxels with <= 0 values, min_value = {min_value}, using nan for these...")
            else:
                variable = np.array([floor if x <= floor else x for x in variable])  
                negative_spaxels = variable.tolist().count(floor)
                if warnings: print(f"> WARNING: The image has {negative_spaxels} spaxels with <= 0 values, min_value = {min_value}, using provided floor = {floor} for these...")
            if clow is None: clow = floor
    else:
        norm = colors.Normalize()
        
    if clow is None: clow = np.nanmin(variable[list_spectra])
    if chigh is None: chigh = np.nanmax(variable[list_spectra])

    if list_spectra is None: list_spectra = list(range(len(rss.intensity)))

    # Get coordinates
    fib_ra = rss.info["fib_ra"]
    fib_dec = rss.info["fib_dec"]
    
    RA_cen = fib_ra.mean()
    DEC_cen = fib_dec.mean()
    
    offset_RA_arcsec = (fib_ra - RA_cen) * 3600.
    offset_DEC_arcsec = (fib_dec - DEC_cen) * 3600.

    # Start figure
    if np.isscalar(figsize) :
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        aspect_ratio = 1.
    else:
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
        aspect_ratio = figsize[0] / figsize[1]
        figsize = figsize[0]
    
    cax= ax.scatter(offset_RA_arcsec[list_spectra],
                offset_DEC_arcsec[list_spectra],
                c=variable[list_spectra], cmap=cmap, 
                clim=(clow, chigh),
                norm=norm,
                s=20*figsize, marker="h")
    if description is None:
        ax.set_title(rss.info["name"] + title)   
    else:
        ax.set_title(str(description))   # description
    ax.set_facecolor("lightgrey")
    
    plt.xlim(np.nanmin(offset_RA_arcsec) - 0.75, np.nanmax(offset_RA_arcsec) + 0.75)
    plt.ylim(np.nanmin(offset_DEC_arcsec) - 0.75, np.nanmax(offset_DEC_arcsec) + 0.75)
    plt.gca().invert_xaxis()   # RA is NEG to the right
    if xlabel is None: xlabel ="$\Delta$ RA [arcsec]" 
    if ylabel is None: ylabel="$\Delta$ DEC [arcsec]"
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))    
    plt.minorticks_on()
    if grid: plt.grid(which='both')
    
    # Set colorbar
    if colorbar_pad is None : colorbar_pad = 0.02
    if colorbar_width_fraction is None: colorbar_width_fraction = 0.4        
    cbar = fig.colorbar(cax, pad = colorbar_pad, aspect = figsize*aspect_ratio / colorbar_width_fraction ) #, width=figsize*colorbar_width_fraction) # colorbar_width_fraction
    cbar.set_label(str(color_bar_text), rotation=90, labelpad=colorbar_labelpad)
    cbar.ax.tick_params()
    
    # Show or save final plot
    if save_plot_in_file is None:
            plt.show()
            plt.close() 
    else:
       plt.savefig(save_plot_in_file, bbox_inches='tight')
       plt.close() 
       if verbose: print("  Figure saved in file",save_plot_in_file, **kwargs)



# Ángel :-)