.. koala documentation master file, created by
   sphinx-quickstart on Thu Feb 13 16:10:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyKoala's documentation!
=================================


PyKOALA is a Python package to reduce KOALA+AAOmega integral field spectroscopy (IFS) data creating a data cube. It produces full calibrated (wavelength, flux and astrometry) data cubes ready for science.

`KOALA`_, the Kilofibre Optical AAT Lenslet Array, is a wide-field, high efficiency, integral field unit used by the 
AAOmega spectrograph on the 3.9m AAT `Anglo-Australian Telescope`_ at Siding Spring Observatory. **PyKOALA** is the forthcoming data reduction pipeline for 
creating science-ready 3D data cubes using Raw Stacked Spectra (RSS) images created with `2dfdr`_.

- Code repository: https://github.com/pykoala/koala

.. _KOALA: https://aat.anu.edu.au/science/instruments/current/koala/overview
.. _Anglo-Australian Telescope: https://www.aao.gov.au/about-us/AAT
.. _2dfdr: https://aat.anu.edu.au/science/instruments/current/AAOmega/reduction
 

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    getting-started/index
    user-guide/index
    developer-guide/index
    api
    license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
