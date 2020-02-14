# -*- coding: utf-8 -*-
import setuptools

import versioneer

#DESCRIPTION_FILES = ["pypi-intro.rst"]
#
#long_description = []
#import codecs
#for filename in DESCRIPTION_FILES:
#    with codecs.open(filename, 'r', 'utf-8') as f:
#        long_description.append(f.read())
#long_description = "\n".join(long_description)


setuptools.setup(
    name = "pykoala-ifs",
    version = versioneer.get_version(),
    packages = setuptools.find_packages('src'),
    package_dir = {'': 'src'},
    install_requires = [
        "astropy==2.0.16",
        "pysynphot",
        "matplotlib==2.2.3",
        "numpy==1.16.6",
        "scipy==1.2.3",
        'future',
    ],
    #python_requires = '>=3.5',
    author = u"Ángel López-Sánchez",
    author_email = "angel.lopez-sanchez@mq.edu.au",
    description = "Data reduction tools for KOALA IFU.",
    #long_description = long_description,
    license = "3-clause BSD",
    keywords = "astronomy aat koala",
    url = "https://pykoala.readthedocs.io",
    project_urls={
        'Documentation': 'https://pykoala.readthedocs.io',
        'Source': 'https://github.com/pykoala/koala/',
        'Tracker': 'https://github.com/pykoala/koala/issues',
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering',
        #'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.5',
        #'Programming Language :: Python :: 3.6',
        #'Programming Language :: Python :: 3.7',
        #'Programming Language :: Python :: 3.8',
        #'Programming Language :: Python :: 3 :: Only',
    ],
    include_package_data=True,
    cmdclass=versioneer.get_cmdclass(),
)
