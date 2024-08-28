API Documentation
=================

The Documentation is divided into four main categories:

- Data Structures
   How `pykoala` deals with different sorts of data.
- Data Corrections
   What corrections does `pykoala` perform on the data.
- 1D Spectra manipulation
   Tools for manipulating one dimensional spectra.
- Plotting facilities
   Quick visualization and quality control plots.

Data Structures
===============

.. automodule:: pykoala.data_container
   :members:
   :undoc-members:
   :show-inheritance:

pykoala.rss module
------------------

.. automodule:: pykoala.rss
   :members:
   :undoc-members:
   :show-inheritance:

pykoala.cubing module
---------------------

.. automodule:: pykoala.cubing
   :members:
   :undoc-members:
   :show-inheritance:


Supported IFS Instruments
---------------------

.. toctree::
   :maxdepth: 3
   pykoala.instruments

The currently available instrument modules in `pykoala` are:

- :class:`pykoala.instruments.koala_ifu`


Data Corrections
================
 
This section contains the different correction modules included in `pykoala`.

.. toctree:: pykoala.corrections
   :maxdepth: 2
   :caption: Available modules


Manipulating Spectra
====================

Tools for manipulating 1D spectra.


Plotting tools
===================

Useful methods for plotting and performing quality control tests.

.. toctree:: pykoala.plotting
   :maxdepth: 2
