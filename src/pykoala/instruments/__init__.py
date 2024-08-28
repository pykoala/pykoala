"""
This module contains the currenly available IFS intruments that `pykoala` can
ingest. Each submodule includes basic routines for reading and formating both data
and metadata from instrument-native files.
"""

from . import koala_ifu
#from . import hector_ifu
from . import weave

__all__ = ["koala_ifu"]