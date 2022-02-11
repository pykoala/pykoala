#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from koala.io import full_path

def plot_plot(x, y,  xmin="",xmax="",ymin="",ymax="",percentile_min=2, percentile_max=98,
              ptitle = None, xlabel = None, ylabel = None, label="",
              #ptitle="Pretty plot", xlabel="Wavelength [$\mathrm{\AA}$]", ylabel="", 
              fcal="", 
              psym="", color="blue", alpha="", linewidth=1,  linestyle="-", markersize = 10,
              vlines=[], hlines=[], chlines=[], cvlines=[], axvspan=[[0,0]], hwidth =1, vwidth =1,
              frameon = False, loc = 0, ncol = 6,   text=[],
              title_fontsize=12, label_axes_fontsize=10, axes_fontsize=10, tick_size=[5,1,2,1], axes_thickness =0,
              save_file="", path="", fig_size=12, warnings = True, show=True, statistics=""):
    
   """
   Plot this plot! An easy way of plotting plots in Python.
   
   Parameters
    ----------
    x,y : floats (default = none)
        Positional arguments for plotting the data
    xmin, xmax, ymin, ymax : floats (default = none)
        Plotting limits
    percentile_min : integer (default = 2)
        Lower bound percentile for filtering outliers in data
    percentile_max : integer (default = 98)
        Higher bound percentile for filtering outliers in data
    ptitle : string (default = "Pretty plot")
        Title of the plot
    xlabel : string (default = "Wavelength [$\mathrm{\AA}$]")
        Label for x axis of the plot
    ylabel : string (default = "Flux [counts]")
        Label for y axis of the plot
    fcal : boolean (default = none)
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
   
   if fig_size == "big":
       fig_size=20
       label_axes_fontsize=20 
       axes_fontsize=15
       title_fontsize=22
       tick_size=[10,1,5,1]
       axes_thickness =3
       hwidth =2
       vwidth =2

   if fig_size in ["very_big", "verybig", "vbig"]:
       fig_size=35
       label_axes_fontsize=30 
       axes_fontsize=25
       title_fontsize=28 
       tick_size=[15,2,8,2]
       axes_thickness =3
       hwidth =4 
       vwidth =4

   if fig_size != 0 : plt.figure(figsize=(fig_size, fig_size/2.5))
   
   if np.isscalar(x[0]) :
       xx=[]
       for i in range(len(y)):
           xx.append(x)
   else:
       xx=x

   if xmin == "" : xmin = np.nanmin(xx[0])
   if xmax == "" : xmax = np.nanmax(xx[0])  
   
   alpha_=alpha
   psym_=psym
   label_=label
   linewidth_=linewidth
   markersize_=markersize
   linestyle_=linestyle

   n_plots=len(y)
       
   if np.isscalar(y[0]) ==  False:
       if np.isscalar(alpha):        
           if alpha_ == "":
               alpha =[0.5]*n_plots
           else:
               alpha=[alpha_]*n_plots
       if np.isscalar(psym): psym=[psym_]*n_plots
       if np.isscalar(label): label=[label_]*n_plots
       if np.isscalar(linewidth): linewidth=[linewidth_]*n_plots
       if np.isscalar(markersize):markersize=[markersize_]*n_plots
       if np.isscalar(linestyle): linestyle=[linestyle_]*n_plots
       if color == "blue" : color = ["red","blue","green","k","orange", "purple", "cyan", "lime"]
       if ymax == "": y_max_list = []
       if ymin == "": y_min_list = []
              
       if fcal == "":
           if np.nanmedian(np.abs(y[0])) < 1E-10:
               fcal = True
               if np.nanmedian(y[0]) < 1E-20 and np.var(y[0]) > 0 : fcal = False
       for i in range(len(y)):
           if psym[i] == "":
               plt.plot(xx[i],y[i], color=color[i], alpha=alpha[i], label=label[i], linewidth=linewidth[i], linestyle=linestyle[i])
           else:
               #print(xx)
               #print(y)
               # print(psym)
               # print(color) 
               # print(alpha)
               # print(label)
               # print(linewidth)
               # print(markersize)
               # print(len(xx),len(y), len(psym), len(color), len(alpha), len(label), len(linewidth), len(markersize))
               plt.plot(xx[i],y[i], psym[i], color=color[i], alpha=alpha[i], label=label[i], mew=linewidth[i], markersize=markersize[i])
           if ymax == "":
                    y_max_ = []                
                    y_max_.extend((y[i][j]) for j in range(len(xx[i])) if (xx[i][j] > xmin and xx[i][j] < xmax) )  
                    y_max_list.append(np.nanpercentile(y_max_, percentile_max))
           if ymin == "":
                    y_min_ = []                
                    y_min_.extend((y[i][j]) for j in range(len(xx[i])) if (xx[i][j] > xmin and xx[i][j] < xmax) )  
                    y_min_list.append(np.nanpercentile(y_min_, percentile_min))
       if ymax == "":
           ymax = np.nanmax(y_max_list)
       if ymin == "":
           ymin = np.nanmin(y_min_list)
   else:
       if alpha == "": alpha=1
       if statistics =="": statistics = True
       if fcal == "":
           if np.nanmedian(np.abs(y)) < 1E-10 : 
               fcal= True 
               if np.nanmedian(np.abs(y)) < 1E-20 and np.nanvar(np.abs(y)) > 0 : fcal = False
       if psym == "":
             plt.plot(x,y, color=color, alpha=alpha,linewidth=linewidth,  linestyle=linestyle)
       else:
           plt.plot(x,y, psym, color=color, alpha=alpha, mew=linewidth, markersize=markersize)
       if ymin == "" :
           y_min_ = []                
           y_min_.extend((y[j]) for j in range(len(x)) if (x[j] > xmin and x[j] < xmax) )  
           ymin = np.nanpercentile(y_min_, percentile_min)
       if ymax == "" :
           y_max_ = []                
           y_max_.extend((y[j]) for j in range(len(x)) if (x[j] > xmin and x[j] < xmax) )  
           ymax = np.nanpercentile(y_max_, percentile_max)    
       
   plt.xlim(xmin,xmax)                    
   plt.ylim(ymin,ymax)
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
   
   if len(chlines) != len(hlines):
       for i in range(len(hlines)-len(chlines)):
           chlines.append("k")  
           
   for i in range(len(hlines)):
       if chlines[i] != "k":
           hlinestyle="-"
           halpha=0.8
       else:
           hlinestyle="--" 
           halpha=0.3
       plt.axhline(y=hlines[i], color=chlines[i], linestyle=hlinestyle, alpha=halpha, linewidth=hwidth)

    
   if len(cvlines) != len(vlines):
       for i in range(len(vlines)-len(cvlines)):
           cvlines.append("k") 
    
   for i in range(len(vlines)):
       if cvlines[i] != "k":
           vlinestyle="-"
           valpha=0.8
       else:
           vlinestyle="--" 
           valpha=0.3
       plt.axvline(x=vlines[i], color=cvlines[i], linestyle=vlinestyle, alpha=valpha, linewidth=vwidth)
    
   if label_ != "" : 
       plt.legend(frameon=frameon, loc=loc, ncol=ncol)
       
   if axvspan[0][0] != 0:
       for i in range(len(axvspan)):
           plt.axvspan(axvspan[i][0], axvspan[i][1], facecolor='orange', alpha=0.15, zorder=3)   
           
   if len(text)  > 0:
        for i in range(len(text)):
            plt.text(text[i][0],text[i][1], text[i][2], size=axes_fontsize)
                    
   if save_file == "":
       if show: 
           plt.show()
           plt.close() 
   else:
       if path != "" : save_file=full_path(save_file,path)
       plt.savefig(save_file)
       plt.close() 
       print("  Figure saved in file",save_file)
   
   if statistics == "": statistics=False
   if statistics:
       if np.isscalar(y[0]) : 
           basic_statistics(y, x=x, xmin=xmin, xmax=xmax)
       else:    
           for i in range(len(y)):
               basic_statistics(y[i], x=xx[i], xmin=xmin, xmax=xmax)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def basic_statistics(y, x="", xmin="", xmax="", return_data=False, verbose = True):
    """
    Provides basic statistics: min, median, max, std, rms, and snr"
    """    
    if len(x) == 0:
        y_ = y
    else:          
        y_ = []      
        if xmin == "" : xmin = x[0]
        if xmax == "" : xmax = x[-1]          
        y_.extend((y[j]) for j in range(len(x)) if (x[j] > xmin and x[j] < xmax) )  
     
    median_value = np.nanmedian(y_)
    min_value = np.nanmin(y_)
    max_value = np.nanmax(y_)
    
    n_ = len(y_)
    #mean_ = np.sum(y_) / n_
    mean_ = np.nanmean(y_)
    #var_ = np.sum((item - mean_)**2 for item in y_) / (n_ - 1)  
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