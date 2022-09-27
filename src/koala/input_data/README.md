### Adding new spectrophotometric standard stars
Each new file must include at least a column with wavelength (in angstrom) and a column with the flux density per unit wavelength.

Flux units can be arbitrary but in order to provide a common framework, flux density in all default files is expressed
in units of *erg/s/cm^2/AA* 

The naming convention for each file is: "f-starname.dat", e.g. "feg274.dat".
This facilitates to find the stellar template when a given observation contains the star name.  

All default templates were selected from: https://www.eso.org/sci/observing/tools/standards/spectra.html

> This directory contains data files for all the spectrophotometric
standard stars in the Oke 1990 ( AJ, 99, 1621 ) reference, taken
from the CALOBS directory at STScI (see 
http://www.stsci.edu/hst/observatory/cdbs/astronomical_catalogs.html )
The 'f' files list wavelength ( A ), flux ( ergs/cm/cm/s/A * 10**16 )
and flux ( milli-Jy ) and bin (A). The file name consists of a prefix
'f' and the star name.
The 'm' files list wavelength ( A ), AB magnitude and bin (A). The file 
name consists of a prefix 'm' and the star name.
For more details see:
http://www.eso.org/sci/observing/tools/standards/spectra/okestandards.html
Last update 10 August 2007 by jwalsh at eso.org

