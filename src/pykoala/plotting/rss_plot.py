import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pykoala.ancillary import vprint, find_index_nearest  
from pykoala.plotting.plot_plot import plot_plot

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
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
    w = rss.wavelength
    # Check if wavelength is a index or a wavelength
    n_wave = len(rss.wavelength)
    if wavelength < n_wave and type(wavelength) is int:  # wavelength is an index
        wave_index = wavelength
    else:
        wave_index= find_index_nearest(wavelength, w)
           
    wave = w[wave_index]       
    corte_wave = rss.intensity[:, wave_index]
    
    if kwargs.get("plot") is None: plot = True
    
    if plot:
        x = range(len(rss.intensity))
        if kwargs.get("xlabel") is None:
            kwargs["xlabel"] = "Fibre"  
        if kwargs.get("ptitle") is None:
            kwargs["ptitle"] = "Intensity cut at " + str(round(wave,2)) + " $\mathrm{\AA}$ - index =" + str(wave_index)
        plot_plot(x, corte_wave, **kwargs)
    
    if r: return corte_wave
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
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
    plot_plot(wavelength, spectrum, **kwargs)
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
    if list_spectra is None:
        list_spectra = list(range(len(rss.intensity)))

    spectrum = np.zeros_like(rss.intensity[list_spectra[0]])
    value_list = []

    for fibre in list_spectra: value_list.append(rss.intensity[fibre])
    
    if median:
        spectrum = np.nanmedian(value_list, axis=0)
    else:
        spectrum = np.nansum(value_list, axis=0)

    if kwargs.get("plot") is None: plot = True
 
    if plot:
        #vlines = [self.valid_wave_min, self.valid_wave_max]   #TODO  # It would be good to indicate the good wavelength range, but now these are not saved in RSS object
        if kwargs.get("ptitle") is None:
            if len(list_spectra) == list_spectra[-1] - list_spectra[0] + 1:
                kwargs["ptitle"] = "{} - Combined spectrum in range [{},{}]".format(rss.info["name"],list_spectra[0], list_spectra[-1])
            else:
                kwargs["ptitle"] = "Combined spectrum using requested fibres"
        plot_plot(rss.wavelength, spectrum,  **kwargs)
                  
    if r: return spectrum    
        
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def rss_image(rss, image=None, log=False, gamma=0,
              cmap="seismic_r", clow=None, chigh=None, greyscale = False,
              xmin = None, xmax= None, wmin = None, wmax= None, fmin=0, fmax=None, 
              title_fontsize=12, title=None,
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
    # Check if image is given
    if image is None:
        image = rss.intensity

    # Check color visualization
    norm=colors.LogNorm()
    if log is False: norm = colors.Normalize()
    if gamma > 0: norm=colors.PowerNorm(gamma=gamma)
    if greyscale: cmap = "binary_r" 

    #  Set color scale and map   
    if clow is None:
        clow = np.nanpercentile(image, 5)
    if chigh is None:
        chigh = np.nanpercentile(image, 95)
    if cmap == "seismic_r" and log == False:
        max_abs = np.nanmax([np.abs(clow), np.abs(chigh)])
        clow = -max_abs
        chigh = max_abs
    if log and clow <=0:
        clow_=clow
        clow = np.nanmin(np.abs(image))
        vprint("\n> Plotting image in log but the lowest value is {:.2f}, using the minimum positive value of {:.2e} instead.".format(clow_,clow), **kwargs)  
    
    # Show only a subregion if requested
    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = len(rss.wavelength) 
    if wmin is not None:
        xmin = find_index_nearest(rss.wavelength, wmin)
        if xmin == 0 : wmin = rss.wavelength[0]
    if wmax is not None:
        xmax = find_index_nearest(rss.wavelength, wmax)  
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
        plt.title(str(title), fontsize=title_fontsize)
    else:
        plt.title(rss.info['name'] + " - RSS image", fontsize=title_fontsize)
        
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
       vprint("  Figure saved in file",save_plot_in_file, **kwargs)

# Ángel :-)