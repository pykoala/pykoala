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
    version = versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)

