# PyKOALA

> A multi-instrument tool for reducing Integral Field Spectroscopic Data

---

**PyKOALA** is a Python package to reduce KOALA+AAOmega integral field spectroscopy (IFS) data creating a data cube. It produces full calibrated (wavelength, flux and astrometry) data cubes ready for science.

[KOALA][koala_website], the Kilofibre Optical AAT Lenslet Array, is a wide-field, high efficiency, integral field unit used by the 
AAOmega spectrograph on the 3.9m AAT ([Anglo-Australian Telescope][aat_website]) at Siding Spring Observatory. **PyKOALA** is the forthcoming data reduction pipeline for creating science-ready 3D data cubes using Raw Stacked Spectra (RSS) images created with [2dfdr][2dfdr_website].

[koala_website]: https://aat.anu.edu.au/science/instruments/current/koala/overview
[aat_website]: https://aat.anu.edu.au/about-us/AAT
[2dfdr_website]: https://aat.anu.edu.au/science/instruments/current/AAOmega/reduction

---
## Status
[![Documentation Status](https://readthedocs.org/projects/pykoala/badge/?version=latest)](https://pykoala.readthedocs.io/en/latest/?badge=latest)
[![test](https://github.com/pykoala/pykoala/actions/workflows/test.yml/badge.svg)](https://github.com/pykoala/pykoala/actions/workflows/test.yml)
[![Coverage Status](https://codecov.io/github/pykoala/koala/coverage.svg?branch=master)](https://codecov.io/github/pykoala/koala?branch=master)
[![License](https://img.shields.io/pypi/l/pykoala-ifs.svg)](https://pypi.python.org/pypi/pykoala-ifs/)
[![Supported versions](https://img.shields.io/pypi/pyversions/pykoala-ifs.svg)](https://pypi.python.org/pypi/pykoala-ifs/)
[![PyPI](https://img.shields.io/pypi/status/pykoala-ifs.svg)](https://pypi.python.org/pypi/pykoala-ifs/)

---
## Documentation

**PyKOALA** full documentation can be found here: https://pykoala.readthedocs.io/en/latest/index.html

---
## Download

**PyKOALA** can be obtained by cloning the repository, using git:

```bash
git clone https://github.com/pykoala/pykoala.git
```

## Installation (recommended)

We recommend the use of Python environments for the installation of **PyKOALA**. First, from the terminal enter in the downloaded **PyKOALA** package and create the Python environment:

```bash
cd pykoala
python3 -m venv venv_koala
```

To activate the environment, use:

```bash
source venv_koala/bin/activate
```

then install all required packages with:

```bash
pip install -r requirements.txt
pip install .
```

The second command will also install **PyKOALA** in the virtual environment. Once you are finished with your **PyKOALA** session, use:

```bash
deactivate
```

to deactivate the Python environment.

For more information about installation and usage of Python virtual environment, check the [documentation](https://pykoala.readthedocs.io/en/latest/getting-started/virtual-environment.html) or the oficial [Python documentation](https://docs.python.org/3/library/venv.html).


## Tutorials
---

See the list of available [tutorials](https://github.com/pykoala/pykoala-tutorials)

## License and Acknowledgements

BSD 3-Clause License

Copyright (c) 2020, pykoala All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
    

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.