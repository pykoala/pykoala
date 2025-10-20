import numpy as np
from math import factorial
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

############################ Customised Filters ###############################

# ------------------------- filter polynomial extrapolation --------------------

def _poly_extrap_pad_1d(y, pad, polyorder=1):
    """
    Extrapolate a 1D array on both ends with a polynomial fit.

    Parameters
    ----------
    y : (n,) array_like
        Input 1D signal.
    pad : int
        Number of samples to extrapolate on each side (>= 0).
    polyorder : int
        Polynomial order used for edge fitting (0 = constant, 1 = linear, ...).

    Returns
    -------
    yp : (n + 2*pad,) ndarray
        Padded signal [left_extrap, y, right_extrap].

    Notes
    -----
    Uses least-squares polynomial fits on the first/last m points, where
    m = max(polyorder + 1, min(len(y), max(3, pad))).
    """
    y = np.asarray(y)
    n = y.size
    if pad <= 0 or n == 0:
        return y.copy()
    m = max(polyorder + 1, min(n, max(3, pad)))
    x = np.arange(n, dtype=float)

    # Left fit on first m points
    xl = x[:m]
    yl = y[:m]
    # Right fit on last m points
    xr = x[-m:]
    yr = y[-m:]

    # Guard against singular fits by dropping order if needed
    def _safe_polyfit(x, y, deg):
        deg = min(deg, len(x) - 1)
        if deg < 0:
            deg = 0
        return np.polyfit(x, y, deg)

    cl = _safe_polyfit(xl, yl, polyorder)
    cr = _safe_polyfit(xr, yr, polyorder)

    # Extrapolation coordinates
    x_left = -np.arange(pad, 0, -1, dtype=float)  # -pad, ..., -1
    x_right = n + np.arange(1, pad + 1, dtype=float)  # n+1, ..., n+pad

    y_left = np.polyval(cl, x_left)
    y_right = np.polyval(cr, x_right)

    return np.concatenate([y_left, y, y_right])


def poly_extrapolate_pad(a, pad, axis=-1, polyorder=1):
    """
    Extrapolation padding along a chosen axis using polynomial fits.

    Parameters
    ----------
    a : ndarray
        Input array.
    pad : int
        Samples to add on both sides along `axis`.
    axis : int
        Axis along which padding is applied.
    polyorder : int
        Polynomial order for edge fits.

    Returns
    -------
    out : ndarray
        Padded array with shape increased by 2*pad along `axis`.
    """
    a = np.asarray(a)
    if pad <= 0:
        return a.copy()

    # Move target axis to last for easy iteration and reshape back later
    a_m = np.moveaxis(a, axis, -1)
    leading = a_m.shape[:-1]
    n = a_m.shape[-1]
    out = np.empty(leading + (n + 2 * pad,), dtype=a_m.dtype)
    it = np.nditer(np.zeros(leading), flags=["multi_index"])

    while not it.finished:
        idx = it.multi_index + (slice(None),)
        out[idx] = _poly_extrap_pad_1d(a_m[idx], pad, polyorder)
        it.iternext()

    return np.moveaxis(out, -1, axis)

def _pad_from_size_argument(size, axis, ndim):
    """
    Compute half-window pad from a `size` argument that may be:
    - scalar: same on all axes
    - sequence: per-axis sizes
    Returns the pad along the chosen axis as an integer >= 0.
    """
    if np.isscalar(size) or size is None:
        s = int(size) if size is not None else 0
    else:
        # Normalize axis to non-negative
        ax = axis if axis >= 0 else (ndim + axis)
        s = int(size[ax])
    # For odd/even sizes, define half-width as floor(size/2)
    return max(0, s // 2)

def poly_extrapolate_wrapper(
    filter_func,
    *,
    axis=-1,
    polyorder=1,
    pad_strategy="size",
):
    """
    Create a wrapper around a filtering function that pre-pads the input by
    polynomial extrapolation along `axis`, then crops the result back.

    Parameters
    ----------
    filter_func : callable
        Function with signature like `f(input, *args, **kwargs)` returning
        an ndarray of the same shape as `input` when reasonable.
        Examples: scipy.ndimage.median_filter, percentile_filter,
        gaussian_filter1d, etc.
    axis : int, optional
        Axis along which to perform polynomial extrapolation.
    polyorder : int, optional
        Polynomial order for edge extrapolation.
    pad_strategy : {'size', 'gaussian1d'} or callable, optional
        How to compute the pad (half-window in samples):
        - 'size':   infer from the `size` argument (scalar or per-axis).
        - 'gaussian1d': infer from `sigma` and `truncate` as
                        pad = ceil(truncate * sigma).
        - callable: custom function `(a, args, kwargs, axis) -> pad`.

    Returns
    -------
    wrapped : callable
        A function with the same signature as `filter_func` that applies
        polynomial extrapolation padding before filtering.

    Notes
    -----
    - For 'size' strategy, ensure you pass `size` to the filter. For even
      windows, half-width is floor(size/2).
    - For 'gaussian1d', the input must be 1D along `axis` or separable
      along other axes as per the original function's contract.
    """
    def _pad_resolver(a, args, kwargs):
        # Custom callable
        if callable(pad_strategy):
            return int(pad_strategy(a, args, kwargs, axis))

        if pad_strategy == "size":
            size = kwargs.get("size", None)
            # Try positional if not in kwargs; many ndimage functions are (input, size, ...)
            if size is None and len(args) >= 1:
                size = args[0]
            return _pad_from_size_argument(size, axis, a.ndim)

        if pad_strategy == "gaussian1d":
            # gaussian_filter1d signature: (input, sigma, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)
            sigma = kwargs.get("sigma", None)
            if sigma is None and len(args) >= 1:
                sigma = args[0]
            truncate = kwargs.get("truncate", 4.0)
            # If sigma is array-like, take along axis (rare for 1D case); else scalar
            if np.isscalar(sigma):
                sig = float(sigma)
            else:
                ax = axis if axis >= 0 else (a.ndim + axis)
                sig = float(np.asarray(sigma)[ax])
            return int(np.ceil(truncate * sig))

        raise ValueError(f"Unknown pad_strategy: {pad_strategy!r}")

    def wrapped(a, *args, **kwargs):
        a = np.asarray(a)
        pad = _pad_resolver(a, args, kwargs)
        if pad > 0:
            a_pad = poly_extrapolate_pad(a, pad=pad, axis=axis, polyorder=polyorder)
            out_pad = filter_func(a_pad, *args, **kwargs)
            # Crop back
            slicer = [slice(None)] * a.ndim
            slicer[axis] = slice(pad, pad + a.shape[axis])
            return np.asarray(out_pad)[tuple(slicer)]
        else:
            # Nothing to pad, just call directly
            return filter_func(a, *args, **kwargs)

    # Preserve metadata if available
    try:
        wrapped.__name__ = getattr(filter_func, "__name__", "wrapped_filter")
    except Exception:
        pass
    wrapped.__doc__ = (
        f"Polynomial-extrapolating wrapper around `{getattr(filter_func, '__name__', 'filter')}`.\n\n"
        f"Pads the input along axis={axis} with a degree-{polyorder} polynomial edge extrapolation\n"
        f"before applying the filter, then crops back to the original shape.\n\n"
        f"Pad strategy: {pad_strategy!r}."
    )
    return wrapped

def savgol_filter_weighted(y, window_length, polyorder, deriv=0, delta=1.0,
                           axis=-1, mode="interp", cval=0.0, w=None):
    """
    Weighted Savitzky–Golay filter with optional derivatives.

    Parameters
    ----------
    y : array_like
        Input data.
    window_length : int
        Length of the filtering window (number of coefficients). Must be a
        positive odd integer >= polyorder + 1.
    polyorder : int
        Order of the polynomial used to fit the samples. Must be < window_length.
    deriv : int, optional
        Order of the derivative to compute (default 0 = smoothing only).
    delta : float, optional
        Sample spacing of the input. For derivatives, the result is scaled by
        delta**(-deriv), as in scipy.signal.savgol_filter.
    axis : int, optional
        Axis along which to filter.
    mode : {"interp"}, optional
        How to handle edges. Currently only "interp" is supported (fits a local
        polynomial on truncated windows near the edges, as SciPy does).
    cval : float, optional
        Ignored for mode="interp". Kept for API compatibility.
    w : array_like, optional
        Non-negative weights, broadcastable to `y`. If None, all weights=1.
        Use w=0 to effectively mask points (including inside windows).

    Returns
    -------
    y_out : ndarray
        Filtered (or differentiated) output with the same shape as `y`.

    Notes
    -----
    - This is a per-position weighted least-squares fit of a degree-`polyorder`
      polynomial over a moving window. The estimate is the polynomial value (or
      derivative) at the window center.
    - With `w=None`, results match the unweighted formulation (up to FP noise).
    - Complexity is O(N * polyorder^3) along the filtered axis (polyorder is
      small in practice). If weights are constant, one could precompute
      convolution coefficients—but this implementation handles *varying* weights.

    Examples
    --------
    >>> y = np.sin(np.linspace(0, 6.28, 100)) + 0.2*np.random.randn(100)
    >>> w = np.ones_like(y); w[[3, 40]] = 0.0      # mask two outliers
    >>> ys = savgol_filter_weighted(y,  nine=9, polyorder=3, w=w)

    """
    y = np.asarray(y)
    if window_length % 2 != 1 or window_length < 1:
        raise ValueError("window_length must be a positive odd integer")
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length")
    if deriv < 0:
        raise ValueError("deriv must be >= 0")
    if mode != "interp":
        raise NotImplementedError("Only mode='interp' is implemented in this version.")

    # Move target axis to last for simple iteration
    y_m = np.moveaxis(y, axis, -1)
    out = np.empty_like(y_m, dtype=float)

    # Prepare weights
    if w is None:
        w_m = np.ones_like(y_m, dtype=float)
    else:
        w_m = np.broadcast_to(np.asarray(w, dtype=float), y.shape)
        w_m = np.moveaxis(w_m, axis, -1)
        if np.any(w_m < 0):
            raise ValueError("weights must be non-negative")

    n = y_m.shape[-1]
    half = window_length // 2
    deg = polyorder

    # Factorials for derivative scaling: y^(d)(0) = d! * coeff[d]
    
    deriv_factor = factorial(deriv) / (delta ** deriv)

    # Helper to do one weighted LS fit on a 1D window and return value/deriv at x=0
    def _fit_eval(yw, ww, x):
        # Build weighted Vandermonde: A (m x (deg+1)), weights W (m,)
        # We center x so the evaluation point is always 0 at the window center.
        A = np.vander(x, N=deg + 1, increasing=True)  # [1, x, x^2, ...]
        # Apply weights: solve (A^T W A) c = A^T W y
        # Weights may be zero for masked points.
        ww = ww.astype(float)
        if np.all(ww == 0):
            # No information: fall back to unweighted to avoid singular matrix
            ww = np.ones_like(ww)
        Aw = A * ww[:, None]
        ATA = Aw.T @ A
        ATy = Aw.T @ yw
        try:
            coeff = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if poorly conditioned
            coeff, *_ = np.linalg.lstsq(ATA, ATy, rcond=None)
        if deriv == 0:
            return coeff[0]
        else:
            # Derivative at x=0 is d! * coeff[d]
            return deriv_factor * coeff[deriv]

    # Iterate all trailing lines (vectorized over leading dims)
    leading_shape = y_m.shape[:-1]
    it = np.nditer(np.zeros(leading_shape), flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index + (slice(None),)
        yi = y_m[idx]
        wi = w_m[idx]

        # Output buffer for this 1D slice
        yo = out[idx]

        # Interior indices: use full symmetric window centered at i
        for i in range(n):
            # Determine window bounds with "interp" behaviour:
            # shift the window to stay within [0, n-1]
            start = i - half
            end = i + half + 1
            if start < 0:
                # shift right
                end += -start
                start = 0
            if end > n:
                # shift left
                start -= (end - n)
                end = n
            # Window data
            yw = yi[start:end]
            ww = wi[start:end]
            # Local x centered at the *evaluation index* i
            x_idx = np.arange(start, end)
            x = (x_idx - i).astype(float)
            yo[i] = _fit_eval(yw, ww, x)

        it.iternext()

    # Move axis back
    return np.moveaxis(out, -1, axis)
