# PyKOALA

> A multi-instrument tool for reducing Integral Field Spectroscopic Data

---

PyKOALA is an innovative Python-based library designed to provide a robust and flexible framework for Integral Field Spectroscopy (IFS) data reduction.
By addressing the complexities of transforming raw measurements into scientifically valuable spectra, PyKOALA simplifies the data reduction pipeline while remaining instrument-agnostic and user-friendly.

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

**PyKOALA** documentation can be found [here](https://pykoala.readthedocs.io/en/latest/index.html).

---
## Installation

### Creating a virtual environment (recommended)
To avoid dependency conflicts, we recommend the use of virtual environments for the installation of **PyKOALA**. For example, a simple way to setup a python environment is by using the [`venv`](https://docs.python.org/3/library/venv.html) module:

```bash
python3 -m venv venv_pykoala # common conventions are .venv or venv
```

To activate the environment, use:

```bash
source venv_koala/bin/activate
```

To stop using the environment:

```bash
deactivate
```

### Installing from pypi

Every release of PyKOALA is automatically avaialable through [PyPI](https://pypi.org/):

```bash
python3 -m pip install pykoala-ifs
```

You can check this [link](https://github.com/pykoala/pykoala/releases) for a complete list of previous releases.

### Installing from the source repository

**PyKOALA** can be installed by cloning this repository using:

```bash
git clone https://github.com/pykoala/pykoala.git
```

Then users can install all required packages with:

```bash
cd path/to/pykoala
python3 -m pip install -r requirements.txt  # Install the dependencies
python3 -m pip install .  # Install pykoala
```

For more further instructions about the installation and virtual environtment setup, check the [quick-start documentation](https://pykoala.readthedocs.io/en/latest/getting-started/index.html#quickstart).

Users that would like to contribute to the development of PyKOALA can follow the instructions available [here](https://pykoala.readthedocs.io/en/latest/developer-guide/index.html).

## Tutorials

We provide a comprehensive set tutorials and test data in a [dedicated repository](https://github.com/pykoala/pykoala-tutorials).

## License and Acknowledgements

BSD 3-Clause License

Copyright (c) 2020, pykoala All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
    

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
