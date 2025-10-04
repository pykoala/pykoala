"""Module containing tools for signal processing"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

from astropy.stats import biweight_location, biweight_scale


#######################################################
############ BACKGROUND ESTIMATION ####################
#######################################################


def mode_from_sym_background(x, weights=None, fig_name=None):
    """
    This function calculates the mode of an array,
    assuming that the data can be decomposed as:
    signal plus uniform background plus noise with a symmetric
    probability distribution.

    Parameters
    ----------
    x : np.ndarray
        Data.
    weigths : np.ndarray (None)
        Statistical weight of each point (defaults to None; equal weights).
    fig_name : str (None)
        Name of the figure to be plotted, if desired.

    Returns
    -------
    mode : x.dtype
        Mode of the input array.
    threshold : x.dtype
        Optimal separation between signal and noise background.
    """
    if weights is None:
        weights = np.ones_like(x).astype(float)

    valid = np.where(np.isfinite(x) & np.isfinite(weights))
    n_valid = valid[0].size
    if not n_valid > 2:
        return np.nan, np.nan

    sorted_by_x = np.argsort(x[valid].flatten())
    sorted_x = x[valid].flatten()[sorted_by_x]
    sorted_weight = weights[valid].flatten()[sorted_by_x]

    cumulative_mass = np.cumsum(sorted_weight)
    total_mass = cumulative_mass[-1]
    sorted_weight /= total_mass
    cumulative_mass /= total_mass

    nbins = int(2 + np.sqrt(n_valid))
    m_left = np.linspace(0, 0.5, nbins)[1:-1]
    m_mid = 2 * m_left
    m_right = 0.5 + m_left

    x_left = np.interp(m_left, cumulative_mass, sorted_x)
    x_mid = np.interp(m_mid, cumulative_mass, sorted_x)
    x_right = np.interp(m_right, cumulative_mass, sorted_x)

    h = np.fmin(x_right - x_mid, x_mid - x_left)
    valid = np.where(h > 0)
    if not valid[0].size > 0:
        return np.nan, np.nan
    x_mid = x_mid[valid]
    h = h[valid]
    rho = (
        (
            np.interp(x_mid + h, sorted_x, cumulative_mass)
            - np.interp(x_mid - h, sorted_x, cumulative_mass)
        )
        / 2
        / h
    )

    rho_threshold = np.nanpercentile(rho, 100 * (1 - 1 / np.sqrt(nbins)))
    peak_region = x_mid[rho >= rho_threshold]
    index_min = np.searchsorted(sorted_x, np.min(peak_region))
    index_max = np.searchsorted(sorted_x, np.max(peak_region))
    index_mode = (index_min + index_max) // 2
    # TODO: Parabolic maximum refinement?
    mode = sorted_x[index_mode]
    m_mode = cumulative_mass[index_mode]

    rho_bg = np.fmin(
        rho, np.interp(x_mid, (2 * mode - x_mid)[::-1], rho[::-1], left=0, right=0)
    )
    if m_mode <= 0.5:
        total_bg = 2 * m_mode
        threshold = sorted_x[np.clip(int(n_valid * total_bg), 0, n_valid - 1)]
        contamination = np.interp(2 * mode - threshold, sorted_x, cumulative_mass)
    else:
        total_bg = 2 * (1 - m_mode)
        threshold = sorted_x[np.clip(int(n_valid * (1 - total_bg)), 0, n_valid - 1)]
        contamination = 1 - np.interp(2 * mode - threshold, sorted_x, cumulative_mass)

    if fig_name is not None:
        plt.close(fig_name)
        fig = plt.figure(fig_name, figsize=(8, 5))
        axes = fig.subplots(
            nrows=1,
            ncols=1,
            squeeze=False,
            sharex="col",
            sharey="row",
            gridspec_kw={"hspace": 0, "wspace": 0},
        )

        ax = axes[0, 0]
        ax.set_ylabel("probability density")
        # ax.set_yscale('log')
        ax.plot(x_mid, rho - rho_bg, "b-", alpha=0.5, label="p(signal)")
        ax.plot(x_mid, rho_bg, "r-", alpha=0.5, label="p(background)")
        ax.plot(x_mid, rho, "k-", alpha=0.5, label="total")

        ax.set_xlabel("value")
        L = 5 * np.abs(threshold - mode)
        vmin = np.max([mode - L, x_mid[0]])
        vmax = np.min([mode + L, x_mid[-1]])
        ax.set_xlim(vmin, vmax)

        for ax in axes.flatten():
            ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
            ax.tick_params(which="major", direction="inout", length=8, grid_alpha=0.3)
            ax.tick_params(which="minor", direction="in", length=2, grid_alpha=0.1)
            ax.grid(True, which="both")
            ax.axvspan(sorted_x[index_min], sorted_x[index_max], color="k", alpha=0.1)
            ax.axvline(mode, c="k", ls=":", label=f"background = {mode:.4g}")
            ax.axvline(
                threshold, c="b", ls="-.", label=f"signal threshold = {threshold:.4g}"
            )
            ax.axvline(
                2 * mode - threshold,
                c="r",
                ls="-.",
                alpha=0.5,
                label=f"contamination = {100*contamination/(1-total_bg):.2f}%",
            )
            ax.legend()

        fig.suptitle(fig_name)
        fig.set_tight_layout(True)
        return mode, threshold, fig

    return mode, threshold


class BackgroundEstimator(object):
    """
    Class for estimating background and its dispersion using different statistical methods.
    """

    @staticmethod
    def mean(rss_intensity):
        """
        Compute the mean and standard deviation of the spectra.

        Parameters
        ----------
        rss_intensity : np.ndarray
            Input array (n_fibres x n_wavelength).

        Returns
        -------
        background : np.ndarray
            The computed background (mean) of the data.
        background_sigma : np.ndarray
            The dispersion (standard deviation) of the data.
        """
        background = np.nanmean(rss_intensity, axis=0)
        background_sigma = np.nanstd(rss_intensity, axis=0)
        return background, background_sigma

    @staticmethod
    def percentile(rss_intensity, percentiles=[16, 50, 84]):
        """
        Compute the background and dispersion from specified percentiles.

        Parameters
        ----------
        rss_intensity : np.ndarray
            Input array (n_fibres x n_wavelength).
        percentiles : list of float, optional
            The percentiles to use for computation. Default is [16, 50, 84].

        Returns
        -------
        background : np.ndarray
            The computed background (median) of the data.
        background_sigma : np.ndarray
            The dispersion (half the interpercentile range) of the data.
        """
        plow, background, pup = np.nanpercentile(rss_intensity, percentiles, axis=0)
        background_sigma = (pup - plow) / 2
        return background, background_sigma

    @staticmethod
    def mad(rss_intensity):
        """
        Estimate the background from the median and the dispersion from
        the Median Absolute Deviation (MAD) along the given axis.

        Parameters
        ----------
        rss_intensity : np.ndarray
            Input array (n_fibres x n_wavelength).

        Returns
        -------
        background : np.ndarray
            The computed background (median) of the data.
        background_sigma : np.ndarray
            The dispersion (scaled MAD) of the data.
        """
        background = np.nanmedian(rss_intensity, axis=0)
        mad = np.nanmedian(np.abs(rss_intensity - background[np.newaxis :]), axis=0)
        background_sigma = 1.4826 * mad
        return background, background_sigma

    @staticmethod
    def biweight(rss_intensity):
        """
        Estimate the background and dispersion using the biweight method.

        Parameters
        ----------
        rss_intensity : np.ndarray
            Input array (n_fibres x n_wavelength).

        Returns
        -------
        background : np.ndarray
            Computed background (location) of the data.
        background_sigma : np.ndarray
            Dispersion (scale) of the data.
        """
        background = biweight_location(rss_intensity, axis=0)
        background_sigma = biweight_scale(rss_intensity, axis=0)
        return background, background_sigma

    @staticmethod
    def mode(rss_intensity):
        """
        Estimate the background and dispersion using the mode.

        Parameters
        ----------
        rss_intensity : np.ndarray
            Input array (n_fibres x n_wavelength).

        Returns
        -------
        background : np.ndarray
            The computed background sky.
        background_sigma : np.ndarray
            Scaled MAD of the data behind the mode.
        """
        mode = np.empty_like(rss_intensity[0])
        sigma = np.empty_like(mode)
        for wavelength_index in range(mode.size):
            fibre_flux = rss_intensity[:, wavelength_index]
            mode[wavelength_index], dummy = mode_from_sym_background(fibre_flux)
            below_mode = fibre_flux < mode[wavelength_index]
            sigma[wavelength_index] = 1.4826 * np.nanmedian(
                mode[wavelength_index] - fibre_flux[below_mode]
            )

        return mode, sigma


#######################################################
############ CONTINUUM ESTIMATION ####################
#######################################################


class ContinuumEstimator:
    """
    Class for estimating the continuum of spectral data using different methods.

    Methods
    -------
    medfilt_continuum(data, window_size=5)
        Estimate the continuum using a median filter.

    percentile_continuum(data, percentile, window_size=5)
        Estimate the continuum using a percentile filter.

    pol_continuum(data, wavelength, pol_order=3, **polfit_kwargs)
        Estimate the continuum using polynomial fitting.

    sigma_clipped_mean_continuum(data, window_size=51, sigma=3.0, max_iters=3,
                                 weights=None, min_valid=3, boundary='reflect')
        Estimate the continuum with a sliding-window sigma-clipped (weighted) mean.
    """
    @staticmethod
    def medfilt_continuum(data, window_size=5):
        """
        Estimate the continuum using a median filter.

        Parameters
        ----------
        data : np.ndarray
            The input data array for which to compute the continuum.
        window_size : int, optional
            The size of the window over which to compute the median filter. Default is 5.

        Returns
        -------
        continuum : np.ndarray
            The estimated continuum of the input data.
        """
        continuum = scipy.signal.medfilt(data, window_size)
        return continuum

    @staticmethod
    def percentile_continuum(data, percentile, window_size=5):
        """
        Estimate the continuum using a percentile filter.

        Parameters
        ----------
        data : np.ndarray
            The input data array for which to compute the continuum.
        percentile : list or tuple
            The percentiles (0-100) to use for the continuum estimation.
        window_size : int, optional
            The size of the window over which to compute the percentile filter. Default is 5.

        Returns
        -------
        continuum : np.ndarray
            The estimated continuum of the input data.
        """
        continuum = scipy.ndimage.percentile_filter(data, percentile, window_size)
        return continuum

    @staticmethod
    def pol_continuum(data, wavelength, pol_order=3, **polfit_kwargs):
        """
        Estimate the continuum using polynomial fitting.

        Parameters
        ----------
        data : np.ndarray
            The input data array for which to compute the continuum.
        wavelength : np.ndarray
            The wavelength array corresponding to the data.
        pol_order : int, optional
            The order of the polynomial to fit. Default is 3.
        **polfit_kwargs : dict, optional
            Additional keyword arguments to pass to `np.polyfit`.

        Returns
        -------
        continuum : np.ndarray
            The estimated continuum of the input data.
        """
        fit = np.polyfit(wavelength, data, pol_order, **polfit_kwargs)
        polynomial = np.poly1d(fit)
        return polynomial(wavelength)

    default_min_separation = 10

    @classmethod
    def lower_envelope(self, x, y, min_separation=None):
        """
        #TODO --> Refactor
        Fit lower envelope of a single spectrum:
        1) Find local minima, with a minimum separation `min_separation`.
        2) Interpolate linearly between them.
        3) Add "typical" (~median) offset.
        """
        if min_separation is None:
            min_separation = self.default_min_separation
        valleys = []
        y[np.isnan(y)] = np.inf
        for i in range(min_separation, y.size - min_separation - 1):
            if (
                np.argmin(y[i - min_separation : i + min_separation + 1])
                == min_separation
            ):
                valleys.append(i)
        y[~np.isfinite(y)] = np.nan

        continuum = np.fmin(y, np.interp(x, x[valleys], y[valleys]))

        offset = y - continuum
        offset = np.nanpercentile(offset[offset > 0], np.linspace(1, 50, 51))
        density = (np.arange(offset.size) + 1) / offset
        offset = np.median(offset[density > np.max(density) / 2])

        return continuum + offset, offset

    @staticmethod
    def sigma_clipped_mean_continuum(
        data,
        window_size=11,
        kappa_sigma=3.0,
        max_iters=3,
        weights=None,
        min_valid=3,
        boundary="reflect",
    ):
        """
        Estimate the continuum using a sliding-window sigma-clipped (weighted) mean.

        For each wavelength pixel, this method takes a local window centered on
        that pixel, performs iterative sigma clipping (with optional weights),
        and returns the weighted mean of the surviving samples. Windows with
        fewer than `min_valid` surviving samples after clipping are set to NaN
        and later filled by linear interpolation along the spectrum.

        Parameters
        ----------
        data : np.ndarray, shape (n,)
            1-D array with the input spectrum. NaNs are ignored.
        window_size : int, optional
            Size of the sliding window (in pixels). If even, it is increased by 1
            to make it odd so it can be centered. Default is 11.
        kappa_sigma : float, optional
            Clipping threshold in units of the (weighted) standard deviation.
            Default is 3.0.
        max_iters : int, optional
            Maximum number of sigma-clipping iterations per window. Default is 3.
        weights : np.ndarray or None, shape (n,), optional
            Per-sample non-negative weights (e.g., inverse variance). Zeros or
            non-finite values exclude samples. If None, all valid samples are
            equally weighted. Default is None.
        min_valid : int, optional
            Minimum number of valid (unclipped) samples required to compute the
            window mean. Windows that fail this criterion yield NaN (later
            interpolated). Default is 3.
        boundary : {'reflect', 'nearest', 'wrap', 'edge', 'constant'}, optional
            Padding mode used at the edges before forming windows. 'nearest' is
            treated as an alias of 'edge'. If 'constant' is used, constant values
            of NaN are padded and thus ignored. Default is 'reflect'.

        Returns
        -------
        continuum : np.ndarray, shape (n,)
            Estimated continuum array. Any NaNs arising from insufficient valid
            samples are linearly interpolated; if all values are NaN, the output
            is all-NaN.

        Notes
        -----
        The weighted mean and standard deviation for samples x with weights w are:
        - mean = sum(w * x) / sum(w)
        - std  = sqrt( sum(w * (x - mean)^2) / sum(w) )
        """

        x = np.asarray(data, dtype=float)
        n = x.size
        if n == 0:
            return np.asarray([], dtype=float)

        if weights is None:
            w = np.ones_like(x, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != x.shape:
                raise ValueError("`weights` must have the same shape as `data`.")
            # negative/NaN/inf weights are set to 0 (ignored)
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

        # ensure odd window size >= 3
        win = int(window_size)
        if win < 3:
            raise ValueError("`window_size` must be >= 3.")
        if win % 2 == 0:
            win += 1
        half = win // 2

        # map boundary alias
        pad_mode = "edge" if boundary == "nearest" else boundary
        if pad_mode not in {"reflect", "edge", "wrap", "constant"}:
            raise ValueError("Unsupported `boundary` mode: %r" % boundary)

        # pad data and weights
        if pad_mode == "constant":
            xp = np.pad(x, (half, half), mode=pad_mode, constant_values=np.nan)
            wp = np.pad(w, (half, half), mode=pad_mode, constant_values=0.0)
        else:
            xp = np.pad(x, (half, half), mode=pad_mode)
            wp = np.pad(w, (half, half), mode=pad_mode)

        cont = np.full(n, np.nan, dtype=float)

        def wmean_wstd(vals, ww):
            denom = np.sum(ww)
            if denom <= 0:
                return np.nan, np.nan
            mu = np.sum(ww * vals) / denom
            var = np.sum(ww * (vals - mu) ** 2) / denom
            return mu, np.sqrt(var)

        # main sliding window loop
        for i in range(n):
            xi = xp[i : i + win]
            wi = wp[i : i + win]

            # initial mask: finite values with positive weight
            m = np.isfinite(xi) & (wi > 0)

            if np.count_nonzero(m) < min_valid:
                cont[i] = np.nan
                continue

            # iterative sigma clipping
            for _ in range(int(max_iters)):
                mu, sd = wmean_wstd(xi[m], wi[m])
                if not np.isfinite(mu) or not np.isfinite(sd) or sd == 0.0:
                    break
                new_m = m & (np.abs(xi - mu) <= kappa_sigma * sd)
                if new_m.sum() == m.sum():
                    # converged
                    break
                m = new_m
                if m.sum() < min_valid:
                    break

            # final mean
            if m.sum() >= min_valid:
                mu, _ = wmean_wstd(xi[m], wi[m])
                cont[i] = mu
            else:
                cont[i] = np.nan

        # fill remaining NaNs by linear interpolation along the spectrum
        idx = np.isfinite(cont)
        if np.any(idx):
            if not np.all(idx):
                cont = np.interp(np.arange(n), np.nonzero(idx)[0], cont[idx])
        # else: all NaN -> return as-is

        return cont