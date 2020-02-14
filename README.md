# KOALA
[![Build Status](https://travis-ci.com/pykoala/koala.svg?branch=master)](https://travis-ci.com/pykoala/koala)
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
