"""
Module for estimating a fibre throughput correction.
"""
import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.interpolate import UnivariateSpline

from pykoala import vprint
from pykoala.corrections.correction import CorrectionBase
from pykoala.data_container import RSS
from pykoala import ancillary
from pykoala.plotting import utils as plot_utils

class Throughput(object):
    """Class that represents a throughput data set.
    
    Attributes
    ----------
    throughput_data : np.ndarray
        Array containing the values of the throughput.
    throughput_error : np.ndarray
        Array containing the associated error of ``throughput_data``.
    throughput_path : str
        Path to the original file that was used to initialise the throughput.
    """
    def __init__(self, throughput_data, throughput_error,
                 throughput_file=None):
        self.throughput_data = throughput_data
        self.throughput_error = throughput_error
        self.throughput_file = throughput_file

    def to_fits(self, output_path=None):
        """Save the current throughput into a FITS file.

        Description
        -----------
        The throughput information is stored in a FITS file that consists of:

        - An empty primary HDu
        - Two ImageHDUs containing the data (extension='THROU') and associated error (extension='THROUERR') of the throughput, respectively.

        Parameters
        ----------
        output_path : str, optional, default=None
            Path to the output file where the throughput information will be stored.
            If None, and ``self.throughput_path`` is not ``None`` the original file
            will be overwritten with the new data.
        """
        if output_path is None:
            if self.throughput_file is None:
                raise NameError("Provide output file name for saving throughput")
            else:
                output_path = self.throughput_file
        primary = fits.PrimaryHDU()
        thr = fits.ImageHDU(data=self.throughput_data,
                            name='THROU')
        thr_err = fits.ImageHDU(data=self.throughput_error,
                                name='THROUERR')
        hdul = fits.HDUList([primary, thr, thr_err])
        hdul.writeto(output_path, overwrite=True)
        hdul.close(verbose=True)
        vprint(f"Throughput saved at {output_path}")

    def smooth(self, method="none", axis=1, polydeg=3, spline_s=None,
               update_error="residual", robust=True, random_state=42):
        """
        Smooth the throughput with a polynomial or a spline along one axis.

        Parameters
        ----------
        method : {"none", "polynomial", "spline"}, default "none"
            Smoothing method. "polynomial" uses per-curve polyfit.
            "spline" uses UnivariateSpline with optional smoothing factor s.
        axis : {0, 1}, default 1
            Axis along which to smooth: 0=fibre direction, 1=wavelength direction.
        polydeg : int, default 3
            Degree of the polynomial fit (for method="polynomial").
        spline_s : float or None, default None
            Smoothing factor 's' for UnivariateSpline (for method="spline").
            If None, UnivariateSpline will pick a value based on noise.
        update_error : {"residual", "keep"}, default "residual"
            How to handle throughput_error. "residual" replaces it with a robust
            scatter (NMAD) of residuals; "keep" leaves it unchanged.
        robust : bool, default True
            If True, iteratively sigma-clips residuals during fitting.
        random_state : int, default 42
            Seed used only to preserve deterministic behaviour in cases with
            stochastic tie-breaking.

        Returns
        -------
        self : Throughput
            The same object, modified in place.
        """
        if method not in {"none", "polynomial", "spline"}:
            raise ValueError("method must be 'none', 'polynomial', or 'spline'")

        if method == "none":
            return self

        rng = np.random.default_rng(random_state)
        data = self.throughput_data
        err = self.throughput_error

        if data is None:
            raise ValueError("throughput_data is None; nothing to smooth.")

        # Work on a copy to avoid partial overwrites if something fails
        smoothed = np.array(data, copy=True)
        new_err = None if err is None else np.array(err, copy=True)

        # Iterate curves along the orthogonal axis
        n_fib, n_wave = data.shape
        n_curves = n_fib if axis == 1 else n_wave
        n_pts = n_wave if axis == 1 else n_fib
        x = np.arange(n_pts, dtype=float)

        # Fit each curve independently
        for i in range(n_curves):
            y = data[i, :] if axis == 1 else data[:, i]
            mask = np.isfinite(y)
            # If too few valid points, skip smoothing for this curve
            if mask.sum() < max(polydeg + 1, 5):
                # leave as is
                continue

            xx = x[mask]
            yy = y[mask]
            # Optional robust pre-clipping loop
            if robust:
                # 2 iterations of clipping
                for _ in range(2):
                    # Provisional fit to get residuals
                    if method == "polynomial":
                        coeff = np.polyfit(xx, yy, deg=polydeg)
                        yfit = np.polyval(coeff, xx)
                    else:
                        spl = UnivariateSpline(xx, yy, s=spline_s)
                        yfit = spl(xx)
                    res = yy - yfit
                    sig = ancillary.std_from_mad(res)
                    if not np.isfinite(sig) or sig == 0:
                        break
                    clip = np.abs(res) < 3.0 * sig
                    if clip.sum() < max(polydeg + 1, 5):
                        break
                    xx, yy = xx[clip], yy[clip]

            # Final fit on clipped data
            if method == "polynomial":
                coeff = np.polyfit(xx, yy, deg=polydeg)
                yhat = np.polyval(coeff, x)
            else:
                spl = UnivariateSpline(xx, yy, s=spline_s)
                yhat = spl(x)

            # Write back
            if axis == 1:
                smoothed[i, :] = yhat
            else:
                smoothed[:, i] = yhat

            # Update error from residuals if requested
            if update_error == "residual":
                if axis == 1:
                    res = data[i, :] - yhat
                else:
                    res = data[:, i] - yhat
                # Use robust scatter and enforce a small floor
                sigma = ancillary.std_from_mad(res)
                if np.isnan(sigma) or sigma <= 0:
                    sigma = 0.0
                if new_err is None:
                    # create error if missing
                    new_err = np.zeros_like(smoothed)
                if axis == 1:
                    new_err[i, :] = sigma
                else:
                    new_err[:, i] = sigma

        # Store results
        self.throughput_data = smoothed
        if update_error == "residual":
            # If some curves were not updated, fall back to original error there
            if self.throughput_error is not None and new_err is not None:
                # replace zeros (un-updated) by original errors
                replace_mask = (new_err == 0.0)
                new_err = np.where(replace_mask, self.throughput_error, new_err)
            self.throughput_error = new_err
        # else: keep original error

        return self

    def plot(self, pct=[1, 50, 99], random_seed=50):
        """Plot the Throughput data.
        
        Description
        -----------
        The figure is composed of four panels. The top-row panels correspond
        to a image of the Throughput values and the histogram distribution.
        The panel in the middle row displays the Throughput values along the 
        spectral axis of 3 randomly selected fibres, whereas the bottom panel
        shows the Throughput values of 3 randomly selected columns along the fibre
        direction.

        Parameters
        ----------
        pct : list, optional, default=[1, 50, 99]
            List of percentiles to compute. The first element will be used
            to symmetrize the Throughput image colour bar and histogram with
            respect to 1.
        random_seed : int, optional, default=50
            Random seed to use for generating random column/row indices.
        
        Returns
        -------
        fig : :class:`matplotlib.pyplot.Figure`
            Figure containing the plots.
        """
        fig, ax = plot_utils.new_figure(
            fig_name="Throughput", figsize=(8, 6),
            ncols=1, nrows=1
            )
        ax[0, 0].axis("off")
        gs = fig.add_gridspec(3, 4, wspace=0.15, hspace=0.35)
        
        # Throughput map
        ax = fig.add_subplot(gs[0, 0:-1])
        p_values = np.nanpercentile(self.throughput_data, pct)
        im, cb = plot_utils.plot_image(fig, ax, cblabel="Throughput",
                                       xlabel="wavelength axis",
                                       ylabel="Fibre",
                                       norm=plot_utils.colors.Normalize(
                                           vmin=p_values[0],
                                           vmax=2 - p_values[0]),
                                        cmap=plot_utils.SYMMETRIC_CMAP,
                                       data=self.throughput_data)
        # Histogram
        ax = fig.add_subplot(gs[0, -1])
        ax.hist(self.throughput_data.flatten(),
                bins=self.throughput_data.size // 1000, range=[
                    p_values[0] - 0.1, 2.1 - p_values[0]],
                log=True)
        for p_name, p in zip(pct, p_values):
            ax.axvline(p, label=f"P{p_name}", ls=':', c="k")
        ax.set_ylabel("N pixels")
        ax.set_xlabel("Throughput value")
        ax.set_ylim(10, self.throughput_data.size // 100)

        ax = fig.add_subplot(gs[1, :])

        # Median throughput along wavelength
        median_wavelength_throughput = np.nanmedian(self.throughput_data, axis=0)
        std_wavelength_throughput = ancillary.std_from_mad(self.throughput_data,
                                                           axis=0)

        ax.fill_between(np.arange(0, self.throughput_data.shape[1]),
                        median_wavelength_throughput - std_wavelength_throughput,
                        median_wavelength_throughput + std_wavelength_throughput,
                        alpha=0.1, color='r', label='Median +/- NMAD')
        ax.plot(median_wavelength_throughput, label='Median',
                lw=0.7, color='r')
        
        # Select 3 random fibres
        np.random.seed(random_seed)
        fibre_idx = np.random.randint(low=0, high=self.throughput_data.shape[0],
                                      size=3)
        for idx in fibre_idx:
            ax.plot(self.throughput_data[idx], label='Fibre {}'.format(idx),
                    lw=0.8, alpha=0.8)
        ax.set_ylim(0.75, 1.25)
        ax.set_xlabel("Spectral pixel")
        ax.legend(ncol=5, fontsize='small')

        ax = fig.add_subplot(gs[-1, :])
        wl_idx = np.random.randint(low=0, high=self.throughput_data.shape[1],
                                   size=3)
        for idx in wl_idx:
            ax.plot(self.throughput_data[:, idx].squeeze(),
                    label='Wave col. {}'.format(idx), lw=0.7,
                    alpha=1.0)
        ax.set_ylim(0.75, 1.25)
        ax.set_xlabel("Fibre number")
        ax.legend(ncol=4, fontsize='small')
        
        return fig
        
    @classmethod
    def from_fits(cls, path):
        """Create a :class:`Throughput` from an input FITS file.
        
        Throughput data and associated errors must be stored in HDUL extension 1,
        and 2, respectively.

        Parameters
        ----------
        path: str
            Path to the FITS file containing the throughput data.

        Returns
        -------
        throughput : :class:`Throughput`
            A :class:`Throughput` initialised with the input data.
        """
        if not os.path.isfile(path):
            raise NameError(f"Throughput file {path} does not exists.")
        vprint(f"Loading throughput from {path}")
        with fits.open(path) as hdul:
            throughput_data = hdul[1].data
            throughput_error = hdul[2].data
        return cls(throughput_data, throughput_error, path)

class ThroughputCorrection(CorrectionBase):
    """
    Throughput correction class.

    This class accounts for the relative flux loss due to differences
    on the fibre efficiencies.

    Attributes
    ----------
    name : str
        Correction name, to be recorded in the log.
    throughput : :class:`Throughput`
        2D fibre throughput (n_fibres x n_wavelengths).
    verbose: bool
        False by default.
    """
    name = "ThroughputCorrection"
    throughput = None

    def __init__(self, throughput=None, throughput_path=None,
                 **correction_kwargs):
        super().__init__(**correction_kwargs)

        if throughput is None:
            self.throughput = Throughput()
        else:
            assert isinstance(throughput, Throughput)
            self.throughput = throughput

        self.throughput.path = throughput_path
        if self.throughput.throughput_data is None and self.throughput.path is not None:
            self.throughput.load_fits(self.throughput_path)

    @classmethod
    def from_file(cls, path):
        """Creates a `ThroughputCorrection` using an input FITS file.
        
        Parameters
        ----------
        path: str
            Path to the FITS file containing the throughput data.
        
        Returns
        -------
        throughput_correction : :class:`ThroughputCorrection`
            ThroughputCorrection initialised with the input Throughput data.
        """
        throughput = Throughput.from_fits(path)
        return cls(throughput, path)

    @classmethod
    def from_rss(cls, rss_set, clear_nan=True, statistic='median', medfilt=5,
                 pct_outliers=[5, 95], smooth_method="none",
                 smooth_axis=1,
                 smooth_polydeg=3,
                 smooth_spline_s=None,
                 smooth_update_error="residual",
                 smooth_robust=True):
        """Compute the throughput correctoin from a set of (dome/sky)flat exposures.

        Description
        -----------
        Given a set of flat exposures, this method will estimate the average
        efficiency of each fibre.

        Parameters
        ----------
        rss_set: list
            List of RSS data.
        clean_nan: bool, optional, default=True
            If True, nan values will be replaced by a
            nearest neighbour interpolation.
        statistic: str, optional, default='median'
            Set to 'median' or 'mean' to compute the throughput function.
        medfilt: float, optional, default=None
            If provided, apply a median filter to the throughput estimate.
        pct_outliers : 2-element tupla
            Percentile limits to clip outliers.
        smooth_method : {"none","polynomial","spline"}, default "none"
            Optional curve-wise smoothing of the final throughput.
        smooth_axis : {0,1}, default 1
            Axis along which to smooth (0=fibre, 1=wavelength).
        smooth_polydeg : int, default 3
            Degree for polynomial smoothing.
        smooth_spline_s : float or None, default None
            Smoothing factor 's' for UnivariateSpline (if method="spline").
        smooth_update_error : {"residual","keep"}, default "residual"
            Whether to recompute errors from residuals after smoothing.
        smooth_robust : bool, default True
            Apply light sigma-clipping before the final fit.

        Returns
        -------
        throughput_correction : :class:`ThroughputCorrection`
            ThroughputCorrection initialised with the resulting throughput
            estimation.
        """
        if statistic == 'median':
            stat_func = np.nanmedian
        elif statistic == 'mean':
            stat_func = np.nanmean

        fluxes = []
        for rss in rss_set:
            f = rss.intensity / rss.info['exptime']
            f[f <= 0] = np.nan
            fluxes.append(f)
        # Combine all RSS
        combined_flux = stat_func(fluxes, axis=0)
        combined_flux_err = np.nanstd(fluxes, axis=0) / np.sqrt(len(fluxes))

        # Normalize fibres
        reference_fibre = stat_func(combined_flux, axis=0)
        throughput_data = combined_flux / reference_fibre[np.newaxis, :]

        # Estimate the error
        throughput_error = combined_flux_err / reference_fibre[np.newaxis, :]

        p5, p95 = np.nanpercentile(throughput_data, pct_outliers, axis=1)
        outliers_mask = (throughput_data - p5[:, np.newaxis] <= 0
                         ) | (throughput_data - p95[:, np.newaxis] >= 0)
        throughput_data[outliers_mask] = np.nan

        if clear_nan:
            pix = np.arange(throughput_data.shape[1])
            for ith, (f, f_err) in enumerate(zip(throughput_data, throughput_error)):
                mask = np.isfinite(f) & np.isfinite(f_err)
                throughput_data[ith] = np.interp(pix, pix[mask], f[mask])
                throughput_error[ith] = np.interp(pix, pix[mask], f_err[mask])

        if medfilt is not None:
            print(f"Applying median filter (size={medfilt} px)")
            throughput_data = median_filter(throughput_data, size=medfilt,
                                            axes=1)
            throughput_error = median_filter(
                throughput_error**2, size=medfilt)**0.5

        throughput = Throughput(throughput_data=throughput_data,
                                throughput_error=throughput_error)
        if smooth_method != "none":
            throughput.smooth(method=smooth_method,
                              axis=smooth_axis,
                              polydeg=smooth_polydeg,
                              spline_s=smooth_spline_s,
                              update_error=smooth_update_error,
                              robust=smooth_robust)

        return cls(throughput)


    def apply(self, rss):
        """Apply a 2D throughput model to a RSS.

        Parameters
        ----------
        rss : :class:`pykoala.rss.RSS`
            Original Row-Stacked-Spectra to be corrected.

        Returns
        -------
        rss_corrected : :class:`pykoala.rss.RSS`
            Corrected copy of the input RSS.
        """
        assert isinstance(rss, RSS), "Throughput can only be applied to RSS data"
        rss_out = rss.copy()
        rss_out.intensity = rss_out.intensity / self.throughput.throughput_data
        rss_out.variance = rss_out.variance / self.throughput.throughput_data**2
        self.record_correction(rss_out, status='applied')
        return rss_out

# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
