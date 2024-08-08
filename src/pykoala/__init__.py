from ._version import get_versions
import logging
import sys

__version__ = get_versions()['version']
del get_versions

# Parent logger
pykoala_logger = logging.getLogger('pykoala')

if not (pykoala_logger.hasHandlers()):
    stdout = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
    "[%(name)s] %(asctime)s|%(levelname)s> %(message)s",
    datefmt="%Y/%m/%d %H:%M")
    stdout.setFormatter(fmt)
    pykoala_logger.addHandler(stdout)
    pykoala_logger.setLevel(logging.INFO)

def log_into_file(filename, logger_name='pykoala', level='INFO'):
    logger = logging.getLogger(logger_name)
    hdlr = logging.FileHandler(filename)
    fmt = logging.Formatter(
    "[%(name)s] %(asctime)s|%(levelname)s> %(message)s",
    datefmt="%Y/%m/%d %H:%M")
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr) 
    logger.setLevel(level)


def vprint(msg, level='info'):
    """
    Convenience function for using with the pykoala generic logger.
    """
    logger = logging.getLogger('pykoala')
    print_method = getattr(logger, level)
    print_method(msg)

__all__ = ["__version__", "pykoala_logger", "log_into_file", "vprint"]