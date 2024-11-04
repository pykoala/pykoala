#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
from pykoala.ancillary import remove_units_dec

@remove_units_dec
def quick_plot(x, y,  xmin=None,xmax=None,ymin=None,ymax=None,percentile_min=2, percentile_max=98, extra_y = 0.1,
              ptitle = None, xlabel = None, ylabel = None, label="",
              #ptitle="Pretty plot", xlabel="Wavelength [$\mathrm{\AA}$]", ylabel="", 
              fcal=None, 
              psym=None, color=None, alpha=None, linewidth=1,  linestyle=None, markersize = 10,
              vlines=[], cvlines=[], svlines = [], wvlines=[], avlines=[], 
              hlines=[], chlines=[], shlines = [], whlines=[], ahlines=[],
              axvspan=None,  caxvspan = None,  aaxvspan = None,
              frameon = False, loc = 0, ncol = 6,   text=[],
              title_fontsize=12, label_axes_fontsize=10, axes_fontsize=10, tick_size=[5,1,2,1], axes_thickness =0,
              save_file=None, path=None, fig_size=9, warnings = True, verbose=True,
              show=True, statistics="", plot=True, **kwargs):
    
    
    
   """
   Plot this plot! An easy way of plotting plots in Python.
   
   Parameters
    ----------
    x,y : floats (default = None)
        Positional arguments for plotting the data
    xmin, xmax, ymin, ymax : floats (default = None)
        Plotting limits
    percentile_min : integer (default = 2)
        Lower bound percentile for filtering outliers in data
    percentile_max : integer (default = 98)
        Higher bound percentile for filtering outliers in data
    extra_y : float between 0 and 1
        Extra space up and down in y-axis, percentage of the plotting interval (ymax-ymin)
    ptitle : string (default = "Pretty plot")
        Title of the plot
    xlabel : string (default = "Wavelength [$\mathrm{\AA}$]")
        Label for x axis of the plot
    ylabel : string (default = "Flux [counts]")
        Label for y axis of the plot
    fcal : boolean (default = None)
        If True that means that flux is calibrated to CGS units, changes the y axis label to match
    psym : string or list of strings (default = "")
        Symbol marker, is given. If "" then it is a line.
    color : string (default = "blue")
        Color pallete of the plot. Default order is "red","blue","green","k","orange", "purple", "cyan", "lime"
    alpha : float or list of floats (default = 1)
        Opacity of the graph, 1 is opaque, 0 is transparent
    linewidth: float or list of floats (default=1)
        Linewidth of each line or marker
    linestyle:  string or list of strings (default="-")
        Style of the line
    markersize: float or list of floats (default = 10)
        Size of the marker
    vlines : list of floats (default = [])
        Draws vertical lines at the specified x positions
    hlines : list of floats (default = [])
        Draws horizontal lines at the specified y positions
    chlines : list of strings (default = [])
        Color of the horizontal lines
    cvlines : list of strings (default = [])
        Color of the vertical lines
    axvspan : list of floats (default = [[0,0]])
        Shades the region between the x positions specified
    caxvspan : list of strings (default = [])
        Color of the shaded regions     
    hwidth: float (default =1)
        thickness of horizontal lines
    vwidth: float (default =1)
        thickness of vertical lines
    frameon : boolean (default = False)
        Display the frame of the legend section
    loc : string or pair of floats or integer (default = 0)
        Location of the legend in pixels. See matplotlib.pyplot.legend() documentation
    ncol : integer (default = 6)
        Number of columns of the legend
    label : string or list of strings (default = "")
        Specify labels for the graphs
    title_fontsize: float (default=12)
        Size of the font of the title
    label_axes_fontsize: float (default=10)
        Size of the font of the label of the axes (e.g. Wavelength or Flux)
    axes_fontsize: float (default=10)
        Size of the font of the axes (e.g. 5000, 6000, ....)
    tick_size: list of 4 floats (default=[5,1,2,1])
        [length_major_tick_axes, thickness_major_tick_axes, length_minor_tick_axes, thickness_minor_tick_axes]   
        For defining the length and the thickness of both the major and minor ticks
    axes_thickness: float (default=0)
        Thickness of the axes
    save_file : string (default = none)
        Specify path and filename to save the plot
    path: string  (default = "")
        path of the file to be saved
    fig_size : float (default = 12)
        Size of the figure
    warnings : boolean (default = True)
        Print the warnings in the console if something works incorrectly or might require attention 
    show : boolean (default = True)
        Show the plot
    statistics : boolean (default = False)
        Print statistics of the data in the console
       
   """
   
   if color is None: color = "blue"
   if linestyle is None: linestyle ="-"
      
   if fig_size == "big":
       fig_size=20
       label_axes_fontsize=20 
       axes_fontsize=15
       title_fontsize=22
       tick_size=[10,1,5,1]
       axes_thickness =3
       if wvlines is None and vlines is not None: wvlines = [2] * len(vlines)
       if whlines is None and hlines is not None: whlines = [2] * len(hlines)

   if fig_size in ["very_big", "verybig", "vbig"]:
       fig_size=35
       label_axes_fontsize=30 
       axes_fontsize=25
       title_fontsize=28 
       tick_size=[15,2,8,2]
       axes_thickness =3
       if wvlines is None and vlines is not None: wvlines = [4] * len(vlines)
       if whlines is None and hlines is not None: whlines = [4] * len(hlines)

   if fig_size not in ["C","c","continue","Continue"]:
       if fig_size != 0 : 
           if type(fig_size) == list:
               plt.figure(figsize=(fig_size[0], fig_size[1]))
           else:     
               plt.figure(figsize=(fig_size, fig_size/2.5))
   
   if np.isscalar(x[0]) :
       xx=[]
       for i in range(len(y)):
           xx.append(x)
   else:
       xx=x

   if xmin is None : xmin = np.nanmin(xx[0])
   if xmax is None : xmax = np.nanmax(xx[0])  
   
   alpha_=alpha
   label_=label
   linewidth_=linewidth
   markersize_=markersize
   linestyle_=linestyle

   n_plots=len(y)
       
   if np.isscalar(y[0]) ==  False:
               
       if alpha_ is None:
           alpha =[0.5]*n_plots
       elif np.isscalar(alpha):
           alpha=[alpha_]*n_plots
       if psym is None: psym=[None]*n_plots
       if np.isscalar(label): label=[label_]*n_plots
       if np.isscalar(linewidth): linewidth=[linewidth_]*n_plots
       if np.isscalar(markersize):markersize=[markersize_]*n_plots
       if np.isscalar(linestyle): linestyle=[linestyle_]*n_plots
       if color == "blue" : 
           color_list = ["red","blue","green","k","orange", "purple", "cyan", "lime", "plum","teal","lightcoral", "navy"]
           color = []
           j=0
           for i in range(n_plots):
               color.append(color_list[j])
               j = j+1
               if j == 12 : j = 0
       if ymax is None: 
           y_max_list = []
       if ymin is None: y_min_list = []
              
       if fcal is None:
           if np.nanmedian(np.abs(y[0])) < 1E-10:
               fcal = True
               if np.nanmedian(y[0]) < 1E-20 and np.var(y[0]) > 0 : fcal = False
       for i in range(len(y)):
           #print("color=",color,"\n alpha=", alpha,"\n label=", label, "linewidth=", linewidth, "linestyle=", linestyle)
           if psym[i] is None:
             plt.plot(xx[i],y[i], color=color[i], alpha=alpha[i], label=label[i], linewidth=linewidth[i], linestyle=linestyle[i])
           else: 
               plt.plot(xx[i],y[i], psym[i], color=color[i], alpha=alpha[i], label=label[i], mew=linewidth[i], markersize=markersize[i])

           if ymax is None:
                    y_max_ = []                
                    y_max_.extend((y[i][j]) for j in range(len(xx[i])) if (xx[i][j] > xmin and xx[i][j] < xmax) )  
                    y_max_list.append(np.nanpercentile(y_max_, percentile_max))
           if ymin is None:
                    y_min_ = []                
                    y_min_.extend((y[i][j]) for j in range(len(xx[i])) if (xx[i][j] > xmin and xx[i][j] < xmax) )  
                    y_min_list.append(np.nanpercentile(y_min_, percentile_min))
       if ymax is None :
           ymax = np.nanmax(y_max_list)
       if ymin is None:
           ymin = np.nanmin(y_min_list)
   else:
       if alpha is None: alpha=1
       if statistics == None : statistics = True
       if fcal is None:
           if np.nanmedian(np.abs(y)) < 1E-10 : 
               fcal= True 
               if np.nanmedian(np.abs(y)) < 1E-20 and np.nanvar(np.abs(y)) > 0 : fcal = False
       if psym is None:
             plt.plot(x,y, color=color, alpha=alpha,linewidth=linewidth,  linestyle=linestyle)
       else:
           plt.plot(x,y, psym, color=color, alpha=alpha, mew=linewidth, markersize=markersize)
       if ymin is None :
           y_min_ = []                
           y_min_.extend((y[j]) for j in range(len(x)) if (x[j] > xmin and x[j] < xmax) )  
           ymin = np.nanpercentile(y_min_, percentile_min)
       if ymax is None :
           y_max_ = []                
           y_max_.extend((y[j]) for j in range(len(x)) if (x[j] > xmin and x[j] < xmax) )  
           ymax = np.nanpercentile(y_max_, percentile_max)    
       

   # Provide extra range up and down
   if extra_y is not None:
       interval = (ymax - ymin)
       ymin = ymin - interval * extra_y
       ymax = ymax + interval * extra_y
   
   # Set limits
   plt.xlim(xmin,xmax)
   plt.ylim(ymin,ymax)
   
   # Set title
   if ptitle is None: ptitle="Pretty plot"
   try:   
       plt.title(ptitle, fontsize=title_fontsize)
   except Exception:
       if warnings : print("  WARNING: Something failed when including the title of the plot")
   
   plt.minorticks_on()
   if xlabel is None: xlabel="Wavelength [$\mathrm{\AA}$]"
   plt.xlabel(xlabel, fontsize=label_axes_fontsize)
   #plt.xticks(rotation=90)
   plt.tick_params('both', length=tick_size[0], width=tick_size[1], which='major')
   plt.tick_params('both', length=tick_size[2], width=tick_size[3], which='minor')
   plt.tick_params(labelsize=axes_fontsize)
   plt.axhline(y=ymin,linewidth=axes_thickness, color="k")     # These 4 are for making the axes thicker, it works but it is not ideal...
   plt.axvline(x=xmin,linewidth=axes_thickness, color="k")    
   plt.axhline(y=ymax,linewidth=axes_thickness, color="k")    
   plt.axvline(x=xmax,linewidth=axes_thickness, color="k")    
   
   if ylabel is None: 
       if fcal: 
           ylabel = "Flux [ erg cm$^{-2}$ s$^{-1}$ $\mathrm{\AA}^{-1}$ ]"
       else:         
           ylabel = "Flux [counts]"  
 
   plt.ylabel(ylabel, fontsize=label_axes_fontsize)
   if label_ != "" : 
       plt.legend(frameon=frameon, loc=loc, ncol=ncol)
   
   # Plot horizontal lines
   if len(shlines) != len(hlines):
       for i in range(len(hlines)-len(shlines)):
           shlines.append("--")  
   if len(chlines) != len(hlines):
       for i in range(len(hlines)-len(chlines)):
           chlines.append("k")  
   if len(whlines) != len(hlines):
       for i in range(len(hlines)-len(whlines)):
           whlines.append(1)
   if len(ahlines) != len(hlines):
       for i in range(len(hlines)-len(ahlines)):
           if chlines[i] != "k":
               ahlines.append(0.8)
           else:
               ahlines.append(0.3)
   for i in range(len(hlines)):        
       plt.axhline(y=hlines[i], color=chlines[i], linestyle=shlines[i], alpha=ahlines[i], linewidth=whlines[i])
    
   # Plot vertical lines 
   if len(svlines) != len(vlines):
       for i in range(len(vlines)-len(svlines)):
           svlines.append("--")  
   if len(cvlines) != len(vlines):
       for i in range(len(vlines)-len(cvlines)):
           cvlines.append("k") 
   if len(wvlines) != len(vlines):
       for i in range(len(vlines)-len(wvlines)):
           wvlines.append(1)
   if len(avlines) != len(vlines):
       for i in range(len(vlines)-len(avlines)):
           if cvlines[i] != "k":
               avlines.append(0.8)
           else:
               avlines.append(0.3)
   for i in range(len(vlines)): 
       plt.axvline(x=vlines[i], color=cvlines[i], linestyle=svlines[i], alpha=avlines[i], linewidth=wvlines[i])
    
   # Plot horizontal ranges
   if axvspan is not None:
       if caxvspan is None:  caxvspan = ["orange" for item in axvspan]
       if aaxvspan is None:  aaxvspan = [0.15 for item in axvspan]
       for i in range(len(axvspan)):
           plt.axvspan(axvspan[i][0], axvspan[i][1], facecolor=caxvspan[i], alpha=aaxvspan[i], zorder=3)   
           
   if len(text)  > 0:
        for i in range(len(text)):
            plt.text(text[i][0],text[i][1], text[i][2], size=axes_fontsize)
                    
   if save_file == None:
       if show: 
           plt.show()
           plt.close() 
   else:
       if path != None : save_file=os.path.join(path, save_file) 
       plt.savefig(save_file,  dpi=300, bbox_inches='tight')
       plt.close() 
       if verbose: print("  Figure saved in file",save_file)
   
   if statistics == None: statistics=False
   if statistics:
       if np.isscalar(y[0]) : 
           basic_statistics(y, x=x, xmin=xmin, xmax=xmax)
       else:    
           for i in range(len(y)):
               basic_statistics(y[i], x=xx[i], xmin=xmin, xmax=xmax)



def basic_statistics(y, x=None, xmin=None, xmax=None, return_data=False, verbose = True):
    """
    Provides basic statistics: min, median, max, std, rms, and snr"
    """    
    if x is None:
        y_ = y
    else:          
        y_ = []      
        if xmin is None : xmin = x[0]
        if xmax is None : xmax = x[-1]          
        y_.extend((y[j]) for j in range(len(x)) if (x[j] > xmin and x[j] < xmax) )  
     
    median_value = np.nanmedian(y_)
    min_value = np.nanmin(y_)
    max_value = np.nanmax(y_)
    
    n_ = len(y_)
    mean_ = np.nanmean(y_)
    var_ = np.nanvar(y_)

    std = np.sqrt(var_)
    ave_ = np.nanmean(y_)
    disp_ =  max_value - min_value
    
    rms_v = ((y_ - mean_) / disp_ ) **2
    rms = disp_ * np.sqrt(np.nansum(rms_v)/ (n_-1))
    snr = ave_ / rms
    
    if verbose:
        print("  min_value  = {}, median value = {}, max_value = {}".format(min_value,median_value,max_value))
        print("  standard deviation = {}, rms = {}, snr = {}".format(std, rms, snr))
    
    if return_data : return min_value,median_value,max_value,std, rms, snr


def minimize_error(error_list, max_std_to_median_value = 0.25, verbose = False):
    """
    Providing a list of errors, check if they are similar within an intervale and if they are minimize error

    Parameters
    ----------
    error_list : list of floats
        list with errors.
    max_std_to_median_value : float, optional
        Max value of the STD/MEDIAN value to apply minimization of errors. The default is 0.25.
    verbose : Boolean, optional
        Print the errors in screen. The default is False.

    Returns
    -------
    error : float
        median or minimized error.

    """
    stat_error_list=basic_statistics(error_list, return_data=True, verbose=False)
    optimized = False
    if stat_error_list[3]/stat_error_list[1] < max_std_to_median_value:
        # Only 2 values of errors are combined: the MEDIAN value and the MAX value
        error = 1/stat_error_list[1]**2    +   1/np.nanmax(error_list)**2  
        # error=0                            # This makes the error too low!!!
        # for i in range(len(error_list)):  
        #     #eo2=(1/ea1**2 + 1/ea2**2 + 1/ea3**2 + 1/ea4**2)**(-0.5)
        #     error = error + 1 / error_list[i]**2 
        #     #print (1/np.sqrt(error))
        error = 1/np.sqrt(error)
        optimized = True
    else:
        error = stat_error_list[1]
    if verbose: print("Error list provided:",error_list,"\nEstimated error =",error, ", Optimized = ",optimized)
    return error