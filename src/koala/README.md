# PyKOALA: A multi-instrument tool for reducing Integral Field Spectroscopic Data

---

PyKOALA is a Python package to reduce KOALA+AAOmega integral field spectroscopy (IFS) data creating a data cube. It produces full calibrated (wavelength, flux and astrometry) data cubes ready for science.

---
## Installation

PyKOALA will be available on PyPI soon. In the meantime, it can be built from source as follows. It can then be run from the ```build``` folder by importing the relvant scripts.

1. Clone or copy the branch ```modular_version```




## Code structure

---

### *Data containers*
Abstract classes that represent the different types of data used by PyKOALA.
#### rss.RSS
#### cubing.Cube
#### Data Wrappers
Instrument-dedicated classes for reading the data
- koala_rss

---

### *Cubing*
#### cubing.interpolate_fibre
#### cubing.interpolate_rss
#### cubing.build_cube

### *Registration*
#### registration.register_stars

---


### *Corrections*
#### Fibre Throughput
#### Atmospheric corrections
- AtmosphericExtinction
- Atmospheric Differential Refraction (ADR)
#### Sky
- Sky substraction
  - Sky continuum
  - Sky emission lines
- Telluric correction
#### Flux calibration (Spectral throughput)
#### Cleaning
- Cosmics
- NaN's
- CCD edges

#### Ancillary

---

## Basic reduction procedure

---
(See examples)

### Reducing calibration star data

- Read RSS data.
- Apply fibre throughput.
- Correct data for atmospheric extinction.
- Correct data from telluric atmospheric absorption.
- Substract sky (continuum + emission lines).
- Build cube


---

## License and Acknowledgements

TODO XD