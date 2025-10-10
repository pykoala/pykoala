import numpy as np
import astropy.units as u
from typing import Dict, Tuple, Optional
from scipy.ndimage import percentile_filter, gaussian_filter1d, maximum_filter, label

def odd_int(n: int) -> int:
    n = int(max(1, round(n)))
    return n if n % 2 == 1 else n + 1

def med_abs_dev(x, axis=0):
    """Compute the Median Absolute Deviation (MAD) from an input array.
    
    Parameters
    ----------
    x : :class:`np.ndarray`
        Input data.
    axis : int of tupla, optional
        Array axis along with the MAD will be computed
    
    Returns
    -------
    mad : np.ndarray
        Associated MAD to x along the chosen axes.
    """
    if axis is not None:
        mad = np.nanmedian(
            np.abs(x - np.expand_dims(np.nanmedian(x, axis=axis), axis=axis)),
            axis=axis)
    else:
        mad = np.nanmedian(np.abs(x - np.nanmedian(x)))
    return mad

def std_from_mad(x, axis=0):
    """Estimate the estandard deviation from the MAD.

    Parameters
    ----------
    x : :class:`np.ndarray`
        Input data.
    axis : int of tupla, optional
        Array axis along with the MAD will be computed

    Returns
    -------
    mad : np.ndarray
        Associated MAD to x along the chosen axes.
    
    See also
    --------
    :func:`med_abs_dev`
    """
    return 1.4826 * med_abs_dev(x, axis=axis)

def robust_standarisation(x, axis=None):
    """Standarise an input array using the median and NMAD.
    
    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    standarised : np.ndarray
        The standarised array ``(x - median) / NMAD``
    """
    median = np.nanmedian(x, axis=axis)
    nmad = std_from_mad(x, axis=axis)
    return (x - median) / nmad

def integrated_autocorr_time(x: np.ndarray, max_lag: Optional[int] = None) -> float:
    """Crude integrated autocorrelation time tau for 1D series x."""
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    n = x.size
    if max_lag is None:
        max_lag = max(1, min(200, n // 5))
    var = np.nanvar(x)
    if not np.isfinite(var) or var <= 0:
        return 1.0
    tau = 1.0
    for k in range(1, max_lag + 1):
        a = x[:-k]
        b = x[k:]
        cov = np.nanmean(a * b)
        rho = cov / var
        if rho <= 0.05:  # stop when correlation is small
            break
        tau += 2.0 * rho
    return max(1.0, tau)