#!/usr/bin/python
# -*- coding: utf-8 -*-
# Python script to estimate AAT focus position using 2 FWHM values
# By Angel R. Lopez-Sanchez (AAO/MQU)
# Version 1.1 - 27 Feb 2018 

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
# Tkinter is for python 2; tkinter is for python 3
if sys.version_info[0] < 3:
    import Tkinter as tk
    import tkMessageBox, tkFileDialog

else:
    import tkinter as tk
    from tkinter import messagebox as tkMessageBox
    from tkinter import filedialog as tkFileDialog


class MainApp(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.parent.title('Focusing KOALA @ AAT')
        # call the widgets
        self.ComputeButton()
	self.FWHMButton()
	self.FWHMButton1()
	self.FWHMButton2()
	self.getdata()
        self.canvas()
	self.scale=1.0

    def fwhmKOALA(self):
    	self.scale = 1.0
	self.FWHMButton.configure(bg ="green")
	self.FWHMButton1.configure(bg ="white")
	self.FWHMButton2.configure(bg ="white")
	print "KOALA scale selected, FWHM given in arcsec"

    def fwhmBIN1(self):
    	self.scale = 0.16
	self.FWHMButton.configure(bg ="white")
	self.FWHMButton1.configure(bg ="green")
	self.FWHMButton2.configure(bg ="white")
	print "TV DIRECT binning 1 scale selected, FWHM given in 0.16 arcsec / pix "
	
    def fwhmBIN2(self):
    	self.scale = 0.32
	self.FWHMButton.configure(bg ="white")
	self.FWHMButton1.configure(bg ="white")
	self.FWHMButton2.configure(bg ="green")
	print "TV DIRECT binning 2 scale selected, FWHM given in 0.32 arcsec / pix "
	
    # Compute!	    
    def compute(self):
    	focus=[]
	fwhm1=[]
	fwhm2=[]
        try:
  	    focus.append(float(self.FOCUS1.get()))
  	    fwhm1.append(float(self.FWHM11.get()))
  	    fwhm2.append(float(self.FWHM21.get()))
        except ValueError:
            print "No data in row 1"	
        try:
  	    focus.append(float(self.FOCUS2.get()))
  	    fwhm1.append(float(self.FWHM12.get()))
  	    fwhm2.append(float(self.FWHM22.get()))
        except ValueError:
            print "No data in row 2"
        try:
  	    focus.append(float(self.FOCUS3.get()))
  	    fwhm1.append(float(self.FWHM13.get()))
  	    fwhm2.append(float(self.FWHM23.get()))
        except ValueError:
            print "No data in row 3"	    
        try:
  	    focus.append(float(self.FOCUS4.get()))
  	    fwhm1.append(float(self.FWHM14.get()))
  	    fwhm2.append(float(self.FWHM24.get()))
        except ValueError:
            print "No data in row 4"
        try:
  	    focus.append(float(self.FOCUS5.get()))
  	    fwhm1.append(float(self.FWHM15.get()))
  	    fwhm2.append(float(self.FWHM25.get()))
        except ValueError:
            print "No data in row 5"
	    
        try:
  	    focus.append(float(self.FOCUS6.get()))
  	    fwhm1.append(float(self.FWHM16.get()))
  	    fwhm2.append(float(self.FWHM26.get()))
        except ValueError:
            print "No data in row 6"	    
	    
        try:
  	    focus.append(float(self.FOCUS7.get()))
  	    fwhm1.append(float(self.FWHM17.get()))
  	    fwhm2.append(float(self.FWHM27.get()))
        except ValueError:
            print "No data in row 7"
	    
        try:
  	    focus.append(float(self.FOCUS8.get()))
  	    fwhm1.append(float(self.FWHM18.get()))
  	    fwhm2.append(float(self.FWHM28.get()))
        except ValueError:
            print "No data in row 8"	    
	    
        try:
  	    focus.append(float(self.FOCUS9.get()))
  	    fwhm1.append(float(self.FWHM19.get()))
  	    fwhm2.append(float(self.FWHM29.get()))
        except ValueError:
            print "No data in row 9"        

	try:
  	    focus.append(float(self.FOCUS10.get()))
  	    fwhm1.append(float(self.FWHM110.get()))
  	    fwhm2.append(float(self.FWHM210.get()))
        except ValueError:
            print "No data in row 10"	    

        try:
  	    focus.append(float(self.FOCUS11.get()))
  	    fwhm1.append(float(self.FWHM111.get()))
  	    fwhm2.append(float(self.FWHM211.get()))
        except ValueError:
            print "No data in row 11"
        try:
  	    focus.append(float(self.FOCUS12.get()))
  	    fwhm1.append(float(self.FWHM112.get()))
  	    fwhm2.append(float(self.FWHM212.get()))
        except ValueError:
            print "No data in row 12"	    	    

	fwhm1 = np.array(fwhm1) * self.scale
	fwhm2 = np.array(fwhm2) * self.scale
	afwhm=(fwhm1+fwhm2)/2 

	fwhm=afwhm.tolist()
	
	#fits
	a2,a1,a0 = np.polyfit(focus,fwhm,2)
	a2b,a1b,a0b = np.polyfit(focus,fwhm1,2)
	a2r,a1r,a0r = np.polyfit(focus,fwhm2,2)
	
	xmax=np.nanmax(focus)+0.4
        xmin=np.nanmin(focus)-0.4
	
	xx=np.arange(100)/100.* (xmax-xmin)+ xmin
	fit = a0 + a1*xx + a2*xx**2
	fitb = a0b + a1b*xx + a2b*xx**2
	fitr = a0r + a1r*xx + a2r*xx**2
 	
	seeing = round(np.nanmin(fit),2)	
	bestfocus=round((xx[np.where(fit ==np.nanmin(fit))[0]][0]),2)	    

	seeingr = round(np.nanmin(fitr),2)	
	bestfocusr=round((xx[np.where(fitr ==np.nanmin(fitr))[0]][0]),2)	    

	seeingb = round(np.nanmin(fitb),2)	
	bestfocusb=round((xx[np.where(fitb ==np.nanmin(fitb))[0]][0]),2)	    

  	print " Focus values =",focus
	print " FWHM values =",fwhm
	print " Best seeing    =",seeing,'"     b =',seeingb,'"     r =',seeingr,'"'
	print " Focus position =",bestfocus,"mm    b =",bestfocusb,"mm    r =",bestfocusr,"mm"
	
	result="Focus position: "
	result += str(bestfocus)
	result +=" mm              Best seeing : "
	result += str(seeing)
	result +='"'
	if self.scale == 1.0:
		result +='             Best RED seeing : '
		result += str(seeingr)
		result +='"' 	   
	tbestfocus = tk.Label(self, text=result, background='lightblue')
        tbestfocus.grid(column=3, row=14, sticky="nesw")

	
        f = Figure(figsize=(10,8))
        a = f.add_subplot(111)
        a.plot(focus,fwhm, 'o', ms=20)
	
	
	a.plot(xx,fit)
	a.plot([bestfocus],[seeing],"s", color="green",ms=5)
	a.plot([bestfocusb],[seeingb],"s", color="blue",ms=5)
	a.plot(xx,fitb, color="blue", alpha=0.3)
	a.plot([bestfocusr],[seeingr],"s", color="red",ms=5)
	a.plot(xx,fitr, color="red", alpha=0.3)
	a.set_xlabel("Focus value [mm]")
	a.set_ylabel('FWHM ["]')
	

        self.canvas = FigureCanvasTkAgg(f, master=self)
        self.canvas.get_tk_widget().grid(column=3, row=1, rowspan=13, sticky="nesw")

    ### Compute button
    def ComputeButton(self):
        self.ComputeButton = tk.Button(self, text='Compute!', command=self.compute)
        self.ComputeButton.grid(column=0, row=14, columnspan=3, sticky="nesw")


    ### FWHM size
    def FWHMButton(self):
        self.FWHMButton = tk.Button(self, text='KOALA', command=self.fwhmKOALA, bg="green")
        self.FWHMButton.grid(column=0, row=13,  sticky="nesw")

    def FWHMButton1(self):
        self.FWHMButton1 = tk.Button(self, text='TV 1x1', command=self.fwhmBIN1, bg="white")
        self.FWHMButton1.grid(column=1, row=13,  sticky="nesw")

    def FWHMButton2(self):
        self.FWHMButton2 = tk.Button(self, text='TV 2x2', command=self.fwhmBIN2, bg="white")
        self.FWHMButton2.grid(column=2, row=13,  sticky="nesw")


    # get data
    def getdata(self):
        self.textFocus = tk.Label(self, text="Focus")
        self.textFocus.grid(column=0, row=0, sticky="nesw")
	self.textFWHM1 = tk.Label(self, text="FWHM b")
        self.textFWHM1.grid(column=1, row=0, sticky="nesw")
	self.textFWHM2 = tk.Label(self, text="FWHM r")
        self.textFWHM2.grid(column=2, row=0, sticky="nesw")
	
        self.FOCUS1 = tk.Entry(self, width=6, bg="yellow")	
	self.FOCUS2 = tk.Entry(self, width=6, bg="yellow")
	self.FOCUS3 = tk.Entry(self, width=6, bg="yellow")
	self.FOCUS4 = tk.Entry(self, width=6, bg="yellow")
        self.FOCUS5 = tk.Entry(self, width=6, bg="yellow")
	self.FOCUS6 = tk.Entry(self, width=6, bg="yellow")
	self.FOCUS7 = tk.Entry(self, width=6, bg="yellow")
	self.FOCUS8 = tk.Entry(self, width=6, bg="yellow")
	self.FOCUS9 = tk.Entry(self, width=6, bg="yellow")
	self.FOCUS10 = tk.Entry(self, width=6, bg="yellow")
	self.FOCUS11 = tk.Entry(self, width=6, bg="yellow")
	self.FOCUS12 = tk.Entry(self, width=6, bg="yellow")
		
	self.FWHM11 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM12 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM13 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM14 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM15 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM16 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM17 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM18 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM19 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM110 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM111 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM112 = tk.Entry(self, width=6, bg="yellow")

	self.FWHM21 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM22 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM23 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM24 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM25 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM26 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM27 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM28 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM29 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM210 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM211 = tk.Entry(self, width=6, bg="yellow")
	self.FWHM212 = tk.Entry(self, width=6, bg="yellow")

	self.FOCUS1.grid(column=0, row=1, sticky="nesw")	
	self.FOCUS2.grid(column=0, row=2, sticky="nesw")
	self.FOCUS3.grid(column=0, row=3, sticky="nesw")
	self.FOCUS4.grid(column=0, row=4, sticky="nesw")
	self.FOCUS5.grid(column=0, row=5, sticky="nesw")
	self.FOCUS6.grid(column=0, row=6, sticky="nesw")
	self.FOCUS7.grid(column=0, row=7, sticky="nesw")
	self.FOCUS8.grid(column=0, row=8, sticky="nesw")
	self.FOCUS9.grid(column=0, row=9, sticky="nesw")
	self.FOCUS10.grid(column=0, row=10, sticky="nesw")
	self.FOCUS11.grid(column=0, row=11, sticky="nesw")
	self.FOCUS12.grid(column=0, row=12, sticky="nesw")
	
	self.FWHM11.grid(column=1, row=1, sticky="nesw")	
	self.FWHM12.grid(column=1, row=2, sticky="nesw")	
	self.FWHM13.grid(column=1, row=3, sticky="nesw")	
	self.FWHM14.grid(column=1, row=4, sticky="nesw")	
	self.FWHM15.grid(column=1, row=5, sticky="nesw")	
	self.FWHM16.grid(column=1, row=6, sticky="nesw")	
	self.FWHM17.grid(column=1, row=7, sticky="nesw")	
	self.FWHM18.grid(column=1, row=8, sticky="nesw")	
	self.FWHM19.grid(column=1, row=9, sticky="nesw")	
	self.FWHM110.grid(column=1, row=10, sticky="nesw")	
	self.FWHM111.grid(column=1, row=11, sticky="nesw")	
	self.FWHM112.grid(column=1, row=12, sticky="nesw")	
	
	self.FWHM21.grid(column=2, row=1, sticky="nesw")
	self.FWHM22.grid(column=2, row=2, sticky="nesw")
	self.FWHM23.grid(column=2, row=3, sticky="nesw")
	self.FWHM24.grid(column=2, row=4, sticky="nesw")
	self.FWHM25.grid(column=2, row=5, sticky="nesw")
	self.FWHM26.grid(column=2, row=6, sticky="nesw")
	self.FWHM27.grid(column=2, row=7, sticky="nesw")
	self.FWHM28.grid(column=2, row=8, sticky="nesw")
	self.FWHM29.grid(column=2, row=9, sticky="nesw")
	self.FWHM210.grid(column=2, row=10, sticky="nesw")
	self.FWHM211.grid(column=2, row=11, sticky="nesw")
	self.FWHM212.grid(column=2, row=12, sticky="nesw")
	

	
    # Canvas
    def canvas(self):
        self.f = Figure(figsize=(10,8))
        self.canvas = FigureCanvasTkAgg(self.f, master=self)
        self.canvas.get_tk_widget().grid(column=3, row=1, rowspan=13, sticky="nesw")

        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.parent)

if __name__ == "__main__":
    root = tk.Tk()
#    root.geometry("1000x730+10+10")
    root.resizable(0, 0)
    MainApp(root).pack(side=tk.TOP)
    root.mainloop()
