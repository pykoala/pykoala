#!/usr/bin/python
# -*- coding: utf-8 -*-
# www.pythondiario.com
 
import sys
import math
#from Tkinter import *
#from tk import *
#from Tkinter import *
try:
    from Tkinter import *
except:
    from tkinter import *

def click():
 try:
  ra1h = float(type_ra1h.get())
  ra1m = float(type_ra1m.get())
  ra1s = float(type_ra1s.get())

  ra2h = float(type_ra2h.get())
  ra2m = float(type_ra2m.get())
  ra2s = float(type_ra2s.get())

  dec1d = float(type_dec1d.get())
  dec1m = float(type_dec1m.get())
  dec1s = float(type_dec1s.get())

  dec2d = float(type_dec2d.get())
  dec2m = float(type_dec2m.get())
  dec2s = float(type_dec2s.get())

  
  ra1=ra1h+ra1m/60.+ ra1s/3600.
  ra2=ra2h+ra2m/60.+ ra2s/3600.

  if dec1d < 0:        
     dec1=dec1d-dec1m/60.- dec1s/3600.
  else:
     dec1=dec1d+dec1m/60.+ dec1s/3600
  if dec2d < 0:                
     dec2=dec2d-dec2m/60.- dec2s/3600.
  else:
     dec2=dec2d+dec2m/60.+ dec2s/3600.
  
  avdec = (dec1+dec2)/2
  
  deltadec=round(3600.*(dec2-dec1), decimals)
  deltara =round(15*3600.*(ra2-ra1)*(math.cos(math.radians(avdec))) ,decimals)

  
  tdeltadec.config(text=math.fabs(deltadec))
  tdeltara.config(text=math.fabs(deltara))

  tdeltadec_inver=tdeltadec
  tdeltara_invert=tdeltara
 
  
  if deltadec < 0:
      t_sign_deltadec.config(text="South")
      t_sign_deltadec_invert.config(text="North")
  else:
      t_sign_deltadec.config(text="North")
      t_sign_deltadec_invert.config(text="South")

  if deltara < 0:
      t_sign_deltara.config(text="West")
      t_sign_deltara_invert.config(text="East")

  else:
      t_sign_deltara.config(text="East")
      t_sign_deltara_invert.config(text="West")
	
		 
       
 except ValueError:
  tdeltadec.config(text="This has to be a number!")


 
app = Tk()
#app = tk()
app.title("OFFSETS [in arcsec] between 2 positions")
 

# Main Window
vp = Frame(app)
vp.grid(column=0, row=0, padx=(15,15), pady=(10,10))
vp.columnconfigure(0, weight=1)
vp.rowconfigure(0, weight=1)
 
 
 
text = Label(vp, text="POSITION 1: ")
text.grid(column=1, row=1, sticky=(W,E))
text = Label(vp, text="RA: ")
text.grid(column=2, row=1, sticky=(W,E))
text = Label(vp, text="DEC:")
text.grid(column=6, row=1, sticky=(W,E))


text = Label(vp, text="POSITION 2:")
text.grid(column=1, row=2, sticky=(W,E))
text = Label(vp, text="RA: ")
text.grid(column=2, row=2, sticky=(W,E))
text = Label(vp, text="DEC:")
text.grid(column=6, row=2, sticky=(W,E))


text = Label(vp, text="1 -> 2 :")
text.grid(column=2, row=5, sticky=(W,E))
#text = Label(vp, text="East")
#text.grid(column=4, row=5, sticky=(W,E))
#text = Label(vp, text="North")
#text.grid(column=6, row=5, sticky=(W,E))

text = Label(vp, text="2 -> 1 :")
text.grid(column=2, row=6, sticky=(W,E))
#text = Label(vp, text="East")
#text.grid(column=4, row=6, sticky=(W,E))
#text = Label(vp, text="North")
#text.grid(column=6, row=6, sticky=(W,E))



empty = ""

type_ra1h = Entry(vp, width=5, textvariable=empty)
type_ra1h.grid(column=3, row=1)
type_ra1m = Entry(vp, width=5, textvariable=empty)
type_ra1m.grid(column=4, row=1)
type_ra1s = Entry(vp, width=7, textvariable=empty)
type_ra1s.grid(column=5, row=1)

type_dec1d = Entry(vp, width=5, textvariable=empty)
type_dec1d.grid(column=7, row=1)
type_dec1m = Entry(vp, width=5, textvariable=empty)
type_dec1m.grid(column=8, row=1)
type_dec1s = Entry(vp, width=7, textvariable=empty)
type_dec1s.grid(column=9, row=1)


type_ra2h = Entry(vp, width=5, textvariable=empty)
type_ra2h.grid(column=3, row=2)
type_ra2m = Entry(vp, width=5, textvariable=empty)
type_ra2m.grid(column=4, row=2)
type_ra2s = Entry(vp, width=7, textvariable=empty)
type_ra2s.grid(column=5, row=2)

type_dec2d = Entry(vp, width=5, textvariable=empty)
type_dec2d.grid(column=7, row=2)
type_dec2m = Entry(vp, width=5, textvariable=empty)
type_dec2m.grid(column=8, row=2)
type_dec2s = Entry(vp, width=7, textvariable=empty)
type_dec2s.grid(column=9, row=2)



boton = Button(vp, text="CALCULATE!", command=click)
boton.grid(column=1, row=3)


tdeltara = Label(vp, text="", background='yellow')
tdeltara.grid(column=3, row=5, sticky=(W,E))
t_sign_deltara=Label(vp, text=" ", background='yellow')
t_sign_deltara.grid(column=4, row=5, sticky=(W,E))

tdeltadec = Label(vp, text="", background='yellow')
tdeltadec.grid(column=5, row=5, sticky=(W,E))
t_sign_deltadec = Label(vp, text="", background='yellow')
t_sign_deltadec.grid(column=6, row=5, sticky=(W,E))



tdeltara_invert = Label(vp, text="", background='cyan')
tdeltara_invert.grid(column=3, row=6, sticky=(W,E))
t_sign_deltara_invert=Label(vp, text=" ", background='cyan')
t_sign_deltara_invert.grid(column=4, row=6, sticky=(W,E))


tdeltadec_invert = Label(vp, text="", background='cyan')
tdeltadec_invert.grid(column=5, row=6, sticky=(W,E))
t_sign_deltadec_invert = Label(vp, text="", background='cyan')
t_sign_deltadec_invert.grid(column=6, row=6, sticky=(W,E))



decimals=2



 
app.mainloop()
