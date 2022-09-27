# PyKOALA: A multi-instrument tool for reducing Integral Field Spectroscopic Data

---

## Description

TODO...


---
## Installation

TODO...

---

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