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
    logger.setLevel(level.upper())


def vprint(*mssg, *, level="INFO", verbose=True):
    """Print a message to the logger.
    
    Parameters
    ----------
    mssg : str or any
        The message to log. Can be a string or any object that can be converted to a string.
    level : int or str, optional
        The logging level to use. Can be an integer (e.g., logging.INFO) or
        a string (e.g., 'INFO', 'DEBUG'). Default is "INFO".
    verbose : bool, optional
        If True, the message will be logged. If False, it will not log anything.
        Default is True.
    """
    if verbose:
        if isinstance(level, str): 
            level = getattr(logging, level.upper())
            if level is None:
                raise ValueError(f"Unrecognized log level: {level}")
            # Get the numeric value of the level
            level = level.numerator
        if isinstance(mssg, str):
            pykoala_logger.log(level, mssg)
        else:
            pykoala_logger.log(level, " ".join(map(str, mssg)))


class VerboseMixin():

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        if isinstance(logger, str):
            if logger == pykoala_logger.name:
                self._logger = pykoala_logger
            else:
                self._logger = pykoala_logger.getChild(logger)
        elif isinstance(logger, logging.Logger):
            self._logger = logger

    @property
    def verbose(self):
        """Abstract property for the verbosity of the correction."""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.ERROR)
        self._verbose = verbose
    
    def vprint(self, msg, level='info'):
        """
        Logs a message at the specified level.

        Parameters
        ----------
        msg : str
            The message to be logged.
        level : str, optional
            The level at which to log the message (default is 'info').
        """
        printer = getattr(self.logger, level.lower())
        printer(msg)

    def log_into_file(self, filename, level="INFO"):
        log_into_file(filename, self.logger.name, level=level)


__all__ = ["__version__", "pykoala_logger", "log_into_file", "vprint",
           "VerboseMixin"]
