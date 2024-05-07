"""
Code selected from https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service#PS1ImageCutoutService-python

----
Get PS1 stack FITS cutout images at a list of positions
 
NOTE: If you modify this script to download images in multiple threads,
please do not use more than 10 simultaneous threads for the download. 
The ps1images service is a shared resource, and too many requests from
a single user can cause the system to be unresponsive for all users. 
If you attempt to download images at an excessive rate, eventually you
will find your downloads blocked by the server.
-----
"""
 
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
import requests
import time
from io import StringIO
 
class PSQuery:

    ps1filename = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    fitscut = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
    pixelsize_arcsec = 0.25

    def getimages(tra, tdec, size=240, filters="grizy", format="fits", imagetypes="stack"):
        
        """Query ps1filenames.py service for multiple positions to get a list of images
        This adds a url column to the table to retrieve the cutout.
        
        tra, tdec = list of positions in degrees
        size = image size in pixels (0.25 arcsec/pixel)
        filters = string with filters to include
        format = data format (options are "fits", "jpg", or "png")
        imagetypes = list of any of the acceptable image types.  Default is stack;
            other common choices include warp (single-epoch images), stack.wt (weight image),
            stack.mask, stack.exp (exposure time), stack.num (number of exposures),
            warp.wt, and warp.mask.  This parameter can be a list of strings or a
            comma-separated string.
    
        Returns an astropy table with the results
        """
        
        if format not in ("jpg","png","fits"):
            raise ValueError("format must be one of jpg, png, fits")
        # if imagetypes is a list, convert to a comma-separated string
        if not isinstance(imagetypes,str):
            imagetypes = ",".join(imagetypes)
        # put the positions in an in-memory file object
        cbuf = StringIO()
        cbuf.write('\n'.join(["{} {}".format(ra, dec) for (ra, dec) in zip(tra,tdec)]))
        cbuf.seek(0)
        # use requests.post to pass in positions as a file
        r = requests.post(PSQuery.ps1filename, data=dict(filters=filters, type=imagetypes),
            files=dict(file=cbuf))
        r.raise_for_status()
        tab = Table.read(r.text, format="ascii")
    
        urlbase = "{}?size={}&format={}".format(PSQuery.fitscut,size,format)
        tab["url"] = ["{}&ra={}&dec={}&red={}".format(urlbase,ra,dec,filename)
                for (filename,ra,dec) in zip(tab["filename"],tab["ra"],tab["dec"])]
        return tab

    def getimage(ra, dec, size=240, filters="grizy", format="fits", imagetypes="stack"):
        
        """Query ps1filenames.py service for multiple positions to get a list of images
        This adds a url column to the table to retrieve the cutout.
        
        tra, tdec = list of positions in degrees
        size = image size in pixels (0.25 arcsec/pixel)
        filters = string with filters to include
        format = data format (options are "fits", "jpg", or "png")
        imagetypes = list of any of the acceptable image types.  Default is stack;
            other common choices include warp (single-epoch images), stack.wt (weight image),
            stack.mask, stack.exp (exposure time), stack.num (number of exposures),
            warp.wt, and warp.mask.  This parameter can be a list of strings or a
            comma-separated string.
    
        Returns an astropy table with the results
        """
        
        if format not in ("jpg","png","fits"):
            raise ValueError("format must be one of jpg, png, fits")
        # if imagetypes is a list, convert to a comma-separated string
        if not isinstance(imagetypes,str):
            imagetypes = ",".join(imagetypes)
        # put the positions in an in-memory file object
        cbuf = StringIO()
        cbuf.write('\n'.join(["{} {}".format(ra, dec)]))
        cbuf.seek(0)
        # use requests.post to pass in positions as a file
        r = requests.post(PSQuery.ps1filename, data=dict(filters=filters, type=imagetypes),
            files=dict(file=cbuf))
        r.raise_for_status()
        tab = Table.read(r.text, format="ascii")
    
        urlbase = "{}?size={}&format={}".format(PSQuery.fitscut,size,format)
        tab["url"] = ["{}&ra={}&dec={}&red={}".format(urlbase,ra,dec,filename)
                for (filename,ra,dec) in zip(tab["filename"],tab["ra"],tab["dec"])]
        return tab
    
    def download_image(url, fname):
        """Download a PS cutout image and write it"""
        print(f"Downloading: {url}")
        try:
            r = requests.get(url)
        except Exception as e:
            print(f"ERROR: Download unsuccessful (error: {e})")
            return None
        print(f"Saving file at: {fname}")
        with open(fname,"wb") as f:
            f.write(r.content)
        return fname

    def read_ps_fits(fname):
        """Load a PANSTARRS image."""
        print("Opening PANSTARRS fits file")
        with fits.open(fname) as hdul:
            wcs = WCS(hdul[0].header)
            zp = hdul[0].header['FPA.ZP']
            exptime = hdul[0].header['exptime']
            ps_pix_area = 0.25**2  # arcsec^2/pix
            intensity_zp = 10**(0.4 * zp)
            intensity = 3631 * hdul[0].data / intensity_zp / exptime
            #sb = -2.5 * np.log10(hdul[0].data / ps_pix_area) + zp + 2.5*np.log10(exptime)
            #intensity = 10**(-0.4 * sb) * 3631  # jy
        return intensity, wcs