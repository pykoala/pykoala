# KOALA
[![Build Status](https://travis-ci.com/pykoala/koala.svg?branch=master)](https://travis-ci.com/pykoala/koala)
[![Documentation Status](https://readthedocs.org/projects/pykoala/badge/?version=latest)](https://pykoala.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://codecov.io/github/pykoala/koala/coverage.svg?branch=master)](https://codecov.io/github/pykoala/koala?branch=master)
[![License](https://img.shields.io/pypi/l/pykoala-ifs.svg)](https://pypi.python.org/pypi/pykoala-ifs/)
[![Supported versions](https://img.shields.io/pypi/pyversions/pykoala-ifs.svg)](https://pypi.python.org/pypi/pykoala-ifs/)
[![PyPI](https://img.shields.io/pypi/status/pykoala-ifs.svg)](https://pypi.python.org/pypi/pykoala-ifs/)

---------------------------------------------

[KOALA][koala_website], the Kilofibre Optical AAT Lenslet Array, is a wide-field, high efficiency, integral field unit used by the 
AAOmega spectrograph on the 3.9m AAT ([Anglo-Australian Telescope][aat_website]) at Siding Spring Observatory. **PyKOALA** is the forthcoming data reduction pipeline for 
creating science-ready 3D data cubes using Raw Stacked Spectra (RSS) images created with [2dfdr][2dfdr_website].

[koala_website]: https://aat.anu.edu.au/science/instruments/current/koala/overview
[aat_website]: https://www.aao.gov.au/about-us/AAT
[2dfdr_website]: https://aat.anu.edu.au/science/instruments/current/AAOmega/reduction

Helping to develop PyKOALA 
---------------------------------------------
1. Fork koala into your github account
2. Clone your fork onto your laptop:
```
    git clone https://github.com/<your_account>/koala
```
3. Add this repository as another remote (to get the latest stuff):
```
    git remote add upstream https://github.com/pykoala/koala
```
4. Create a branch to work on the changes:
```
    git checkout -b <new_branch>
```
5. Add and commit changes
6. Push up your changes
7. Create a PR, and wait for someone to review it

Reviewing commits
---------------------------------------------
1. Look through the changes, and provide comments
2. Once the PR is ready, type bors r+, then bors will handle the merge (DON'T
   HIT THE MERGE BUTTON).
3. If the tests fail, help the proposer fix the failures
4. Use bors r+ again

You can also use bors try to try running the tests, without merging

<!---[![Version](https://img.shields.io/pypi/v/pykoala-ifs.svg)](https://pypi.python.org/pypi/pykoala-ifs/) --->
<!---[![Wheel](https://img.shields.io/pypi/wheel/pykoala-ifs.svg)](https://pypi.python.org/pypi/pykoala-ifs/) --->
<!---[![Format](https://img.shields.io/pypi/format/pykoala-ifs.svg)](https://pypi.python.org/pypi/pykoala-ifs/) --->
<!---[![Supported implemntations](https://img.shields.io/pypi/implementation/pykoala-ifs.svg)](https://pypi.python.org/pypi/pykoala-ifs/)--->
