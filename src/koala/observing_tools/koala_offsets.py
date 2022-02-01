#!/usr/bin/python
# -*- coding: utf-8 -*-
# www.pythondiario.com
 
import sys
import math
try:
    from Tkinter import *
except:
    from tkinter import *


def clickwide():
 try:
  s=1.25
  ts.config(text=s)
  _pa = float(type_pa.get())
  _abN = round(s * math.sin(math.radians(_pa)), decimals)
  tabN.config(text=_abN)
  _abE = round(-s * math.cos(math.radians(_pa)), decimals) 
  tabE.config(text=_abE)

  _bcN = round(-s * (math.sin(math.radians(60-_pa)) + math.sin(math.radians(_pa))), decimals)
  tbcN.config(text=_bcN)
  _bcE = round(s * (math.cos(math.radians(_pa)) - math.cos(math.radians(60-_pa) )), decimals)
  tbcE.config(text=_bcE)
  
  _caN = round(s * (math.sin(math.radians(60-_pa))), decimals)
  tcaN.config(text=_caN)
  _caE = round(s * (math.cos(math.radians(60-_pa))), decimals)
  tcaE.config(text=_caE)

  _bdN = round(-s * math.sqrt(3) * (math.cos(math.radians(_pa))), decimals)
  tbdN.config(text=_bdN)
  _bdE = round(-s * math.sqrt(3) * (math.sin(math.radians(_pa))), decimals)
  tbdE.config(text=_bdE)
       
 except ValueError:
  tabN.config(text="This has to be a number!")


def clicknarrow():
 try:
  s=0.7
  ts.config(text=s)
  _pa = float(type_pa.get())
  _abN = round(s * math.sin(math.radians(_pa)), decimals)
  tabN.config(text=_abN)
  _abE = round(-s * math.cos(math.radians(_pa)), decimals) 
  tabE.config(text=_abE)

  _bcN = round(-s * (math.sin(math.radians(60-_pa)) + math.sin(math.radians(_pa))), decimals)
  tbcN.config(text=_bcN)
  _bcE = round(s * (math.cos(math.radians(_pa)) - math.cos(math.radians(60-_pa) )), decimals)
  tbcE.config(text=_bcE)

  _caN = round(s * (math.sin(math.radians(60-_pa))), decimals)
  tcaN.config(text=_caN)
  _caE = round(s * (math.cos(math.radians(60-_pa))), decimals)
  tcaE.config(text=_caE)

  _bdN = round(-s * math.sqrt(3) * (math.cos(math.radians(_pa))), decimals)
  tbdN.config(text=_bdN)
  _bdE = round(-s * math.sqrt(3) * (math.sin(math.radians(_pa))), decimals)
  tbdE.config(text=_bdE)
       
 except ValueError:
  tabN.config(text="This has to be a number!")


 
app = Tk()
app.title("KOALA: Offsets for a given position angle")
 
var=DoubleVar()
var.set(1.25)

# Main Window
vp = Frame(app)
vp.grid(column=0, row=0, padx=(15,15), pady=(10,10))
vp.columnconfigure(0, weight=1)
vp.rowconfigure(0, weight=1)
 
 
 
text0 = Label(vp, text="Position Angle (PA):", background='yellow')
text0.grid(column=1, row=1, sticky=(W,E))

textN = Label(vp, text="N")
textN.grid(column=2, row=2, sticky=(W,E))
textE = Label(vp, text="E")
textE.grid(column=3, row=2, sticky=(W,E))
 
 
text1 = Label(vp, text="Offset c ---> a :")
text1.grid(column=1, row=3, sticky=(W,E))
 
tcaN = Label(vp, text="-")
tcaN.grid(column=2, row=3, sticky=(W,E))

tcaE = Label(vp, text="-")
tcaE.grid(column=3, row=3, sticky=(W,E))
 

text2 = Label(vp, text="Offset a ---> b :")
text2.grid(column=1, row=4, sticky=(W,E))
 
tabN = Label(vp, text="-")
tabN.grid(column=2, row=4, sticky=(W,E))

tabE = Label(vp, text="-")
tabE.grid(column=3, row=4, sticky=(W,E))


text3 = Label(vp, text="Offset b ---> c :")
text3.grid(column=1, row=5, sticky=(W,E))
 
tbcN = Label(vp, text="-")
tbcN.grid(column=2, row=5, sticky=(W,E))

tbcE = Label(vp, text="-")
tbcE.grid(column=3, row=5, sticky=(W,E))

text4 = Label(vp, text="Offset b ---> d :")
text4.grid(column=1, row=6, sticky=(W,E))
 
tbdN = Label(vp, text="-")
tbdN.grid(column=2, row=6, sticky=(W,E))

tbdE = Label(vp, text="-")
tbdE.grid(column=3, row=6, sticky=(W,E))


boton = Button(vp, text="WIDE", command=clickwide)
boton.grid(column=6, row=1)
boton = Button(vp, text="NARROW", command=clicknarrow)
boton.grid(column=7, row=1)
ts = Label(vp, text="-")
ts.grid(column=6, row=2, sticky=(W,E))

texts2 = Label(vp, text="arcsec/spaxel")
texts2.grid(column=7, row=2, sticky=(W,E))


decimals=2


pa = ""
type_pa = Entry(vp, width=10, textvariable=pa)
type_pa.grid(column=2, row=1)
 
app.mainloop()
