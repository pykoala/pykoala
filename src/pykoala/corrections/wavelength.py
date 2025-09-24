"""
Module for estimating and applying wavelength offset corrections related to
inaccuracies in the original wavelength calibration.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from astropy.io import fits
from astropy import units as u
from scipy.ndimage import median_filter, gaussian_filter, percentile_filter, gaussian_filter1d
from scipy.interpolate import interp1d, splrep, BSpline
from scipy.signal import correlate, correlation_lags

from pykoala import vprint
from pykoala.corrections.correction import CorrectionBase
from pykoala.data_container import RSS, SpectraContainer
from pykoala.ancillary import flux_conserving_interpolation, vac_to_air, check_unit
from pykoala import ancillary
from pykoala.plotting.utils import plot_image


class WavelengthOffset(object):
    """Wavelength offset class.

    This class stores a 2D wavelength offset.

    Attributes
    ----------
    offset_data : :class:`astropy.units.Quantity`
        Wavelength offset expressed in pixels or wavelengths.
    offset_error : :class:`astropy.units.Quantity`
        Standard deviation of ``offset_data``.
    path: str
        Filename path.

    """

    offset_data = None
    offset_error = None

    def __init__(self, path=None, offset_data=None, offset_error=None):
        self.path = path
        # The input units can be either pixel or wavelength
        self.offset_data = check_unit(offset_data)
        self.offset_error = check_unit(offset_error)

    def to_fits(self, output_path=None):
        """Save the offset in a FITS file.
        
        Parameters
        ----------
        output_path: str, optional, default=None
            FITS file name path. If None, and ``self.path`` exists,
            the original file is overwritten.

        Notes
        -----
        The output fits file contains an empty PrimaryHDU, and two ImageHDU
        ("OFFSET", "OFFSET_ERR") containing the offset data and associated error.
        """
        if output_path is None:
            if self.path is None:
                raise NameError("Provide output path")
            else:
                output_path = self.path

        primary = fits.PrimaryHDU()
        # OFFSET
        hdr_data = fits.Header()
        if self.offset_data is None:
            raise ValueError("offset_data is None")
        hdr_data["BUNIT"] = self.offset_data.unit.to_string()
        hdu_data = fits.ImageHDU(data=self.offset_data.value, name='OFFSET', header=hdr_data)

        if self.offset_error is None:
            # create an array of NaN with same shape and unit as data
            err_values = np.full_like(self.offset_data.value, np.nan, dtype=float)
            err_unit = self.offset_data.unit
        else:
            err_values = self.offset_error.value
            err_unit = self.offset_error.unit
        hdr_err = fits.Header()
        hdr_err["BUNIT"] = err_unit.to_string()
        hdu_err = fits.ImageHDU(data=err_values, name="OFFSET_ERR", header=hdr_err)

        hdul = fits.HDUList([primary, hdu_data, hdu_err])
        hdul.writeto(output_path, overwrite=True)
        hdul.close()
        vprint(f"Wavelength offset saved at {output_path}")

    @classmethod
    def from_fits(cls, path):
        """Load the offset data from a fits file.

        Loads offset values (extension 1) and
        associated errors (extension 2) from a fits file.

        Parameters
        ----------
        path : str
            Path to the FITS file containing the offset data.

        Returns
        -------
        wavelength_offset : :class:`WavelengthOffset`
            A :class:`WavelengthOffset` initialised with the input data.
        """
        if not os.path.isfile(path):
            raise NameError(f"offset file {path} does not exist.")
        vprint(f"Loading wavelength offset from {path}")
        with fits.open(path) as hdul:
            offset_data = hdul[1].data << u.Unit(hdul[1].header.get("BUNIT", 1))
            offset_error = hdul[2].data << u.Unit(hdul[2].header.get("BUNIT", 1))
        return cls(offset_data=offset_data, offset_error=offset_error,
                   path=path)


class FibreLSFModel:
    """
    Wavelength-dependent Gaussian LSF (sigma in pixels) for each fibre.

    Attributes
    ----------
    wavelength : Quantity, shape (n_wave,)
        Wavelength grid where sigma_pix is defined.
    sigma_pix : ndarray or Quantity, shape (n_fibre, n_wave)
        Measured sigma in pixels per fibre and wavelength.
    meta : dict
        Optional metadata (e.g., origin, date, info strings).
    models : dict
        Per-fibre fitted models, filled by fit_models(). Keys are fibre index,
        value has keys: kind, coeff, knots, degree, s, etc.
    """

    def __init__(self, wavelength, sigma_pix, meta=None):
        self.wavelength = check_unit(wavelength, u.AA)
        sig = sigma_pix
        if hasattr(sig, "unit"):
            sig = sig.to(u.pixel).value
        sig = np.asarray(sig, dtype=float)
        if sig.ndim != 2:
            raise ValueError("sigma_pix must be 2D: (n_fibre, n_wave)")
        if self.wavelength.ndim != 1 or self.wavelength.size != sig.shape[1]:
            raise ValueError("wavelength length must match sigma_pix.shape[1]")
        if not np.all(np.diff(self.wavelength.to_value(self.wavelength.unit)) > 0):
            raise ValueError("wavelength must be strictly increasing")

        self.sigma_pix = sig
        self.meta = {} if meta is None else dict(meta)
        self.models = {}  # per-fibre fitted models

    def to_fits(self, path):
        n_fib, n_w = self.sigma_pix.shape
        primary = fits.PrimaryHDU()
        hdr = fits.Header()
        hdr["BUNIT"] = "pixel"
        hdr["NWAVE"] = n_w
        hdr["NFIBRE"] = n_fib
        hdr["WUNIT"] = self.wavelength.unit.to_string()
        for k, v in (self.meta or {}).items():
            key = f"META_{k[:12].upper()}"
            try:
                hdr[key] = str(v)
            except Exception:
                pass
        h_sigma = fits.ImageHDU(data=self.sigma_pix.astype(np.float32), header=hdr, name="SIGMA")
        col = fits.Column(name="WAVELENGTH",
                          array=self.wavelength.to_value(self.wavelength.unit).astype(np.float64),
                          format="D", unit=self.wavelength.unit.to_string())
        h_wave = fits.BinTableHDU.from_columns([col], name="WAVE")
        fits.HDUList([primary, h_sigma, h_wave]).writeto(path, overwrite=True)
        vprint(f"LSF saved to {path}")

    @classmethod
    def from_fits(cls, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        with fits.open(path) as hdul:
            if "SIGMA" not in hdul or "WAVE" not in hdul:
                raise ValueError("FITS must contain SIGMA image and WAVE table")
            sigma = np.array(hdul["SIGMA"].data, dtype=float)
            hdr = hdul["SIGMA"].header
            wave_unit = u.Unit(hdr.get("WUNIT", "Angstrom"))
            wave = np.array(hdul["WAVE"].data["WAVELENGTH"], dtype=float) << wave_unit
            meta = {}
            for k, v in hdr.items():
                if isinstance(k, str) and k.startswith("META_"):
                    meta[k[5:].lower()] = v
        return cls(wavelength=wave, sigma_pix=sigma, meta=meta)

    @classmethod
    def from_sparse(cls,
                    instrument_wavelength,
                    centres,
                    sigma_values,
                    kind="spline",
                    degree=3,
                    knots=None,
                    s=None,
                    extrapolation="clamp",
                    meta=None):
        """
        Build an LSF model directly from sparse per-window measurements.

        Parameters
        ----------
        instrument_wavelength : Quantity (n_wave,)
            Target wavelength grid used when sampling to a dense grid.
        centres : 1D Quantity or list of 1D Quantity
            Window centers. Either a single common array for all fibres or a list
            with one array per fibre.
        sigma_values : ndarray (n_fibre, n_points) or list of 1D arrays
            Sigma measurements per fibre corresponding to the centres. If 'centres'
            is a list with variable lengths, 'sigma_values' must be a list as well.
        kind : {"poly","spline"}
            Fit type per fibre.
        degree : int
            Polynomial degree or spline order k.
        knots : Quantity array or None
            Interior knots for spline. If None, a small uniform grid is used per fibre.
        s : float or None
            Smoothing factor for spline.
        extrapolation : {"clamp","error"}
            How to handle evaluation outside the data domain.
        meta : dict or None
            Optional metadata.

        Returns
        -------
        lsf : FibreLSFModel
            With sigma_pix pre-sampled on instrument_wavelength and per-fibre models cached.
        """
        wl = check_unit(instrument_wavelength, u.AA)
        n_wave = wl.size
        n_fibre = sigma_values.shape[0]
        # construct empty model with placeholder dense grid; we will fill sigma_pix by evaluation
        lsf = cls(wavelength=wl, sigma_pix=np.zeros((n_fibre, n_wave), dtype=float), meta=meta)
        # fit per-fibre models directly from sparse data
        lsf.fit_from_sparse(centres, sigma_values,
                            kind=kind, degree=degree, knots=knots, s=s)
        # sample to dense grid without linear interpolation
        lsf.sigma_pix = lsf.evaluate(wl, fibre=None, extrapolation=extrapolation)
        return lsf

    def fit_from_sparse(self,
                        wave_centers,
                        sigma_per_fibre,
                        kind="spline",
                        degree=3,
                        knots=None,
                        s=None):
        """
        Fit per-fibre models from sparse inputs. Stores models in self.models.

        Parameters
        ----------
        wave_centers : list of 1D Quantity
            Window centers for each fibre.
        sigma_per_fibre : list of 1D float arrays
            Sigma values aligned with centres for each fibre.
        kind : {"poly","spline"}
        degree : int
        knots : Quantity array or None
        s : float or None
        """

        n_fibre = sigma_per_fibre.shape[0]
        self.models = {}

        wave_centers = check_unit(wave_centers, self.wavelength.unit
                                  ).to_value(self.wavelength.unit)
        for i in range(n_fibre):
            fib_sigma = sigma_per_fibre[i]
            # clean and guards
            finite = np.isfinite(fib_sigma)
            cx = wave_centers[finite]
            cy = fib_sigma[finite]

            if cx.size == 0:
                # fallback: constant 0.0 pix (arbitrary)
                self.models[i] = {"kind": "poly", "degree": 0, "coeff": np.array([0.0]),
                                  "domain": (self.wavelength.value[0], self.wavelength.value[-1])}
                continue
            elif cx.size == 1:
                self.models[i] = {"kind": "poly", "degree": 0, "coeff": np.array([float(cy[0])]),
                                  "domain": (cx[0], cx[0])}
                continue
            # Model domain
            domain = (float(cx[0]), float(cx[-1]))

            if kind == "poly":
                deg_eff = min(int(degree), max(0, cx.size - 1))
                coeff = np.polyfit(cx, cy, deg_eff)
                self.models[i] = {"kind": "poly", "degree": deg_eff, "coeff": coeff,
                                  "domain": domain}
            elif kind == "spline":
                k = int(degree) if degree is not None else 3
                k = max(1, min(k, cx.size - 1))
                if knots is not None:
                    t_interior = check_unit(knots, self.wavelength.unit).to_value(self.wavelength.unit)
                    # keep only interior knots strictly inside domain
                    t_interior = t_interior[(t_interior > cx[0]) & (t_interior < cx[-1])]
                    tck = splrep(cx, cy, k=k, t=t_interior, s=s)
                else:
                    # choose a small number of interior knots (roughly sqrt of points)
                    n_int = max(0, int(np.sqrt(cx.size)) - 1)
                    if n_int > 0:
                        t_interior = np.linspace(cx[0], cx[-1], n_int + 2)[1:-1]
                        tck = splrep(cx, cy, k=k, t=t_interior, s=s)
                    else:
                        tck = splrep(cx, cy, k=min(k, cx.size - 1), s=s)
                self.models[i] = {"kind": "spline", "k": tck[2], "t": tck[0], "c": tck[1],
                                  "domain": domain}
            else:
                raise ValueError("kind must be 'poly' or 'spline'")

    def fit_models(self, kind="poly", degree=3, knots=None, s=None):
        x = self.wavelength.to_value(self.wavelength.unit)
        n_fib = self.sigma_pix.shape[0]
        self.models = {}
        if kind not in ("poly", "spline"):
            raise ValueError("kind must be 'poly' or 'spline'")
        if kind == "poly":
            deg = int(degree)
            for i in range(n_fib):
                y = self.sigma_pix[i]
                if not np.isfinite(y).any():
                    coeff = np.array([np.nan])
                    deg_eff = 0
                else:
                    deg_eff = min(deg, max(0, y.size - 1))
                    coeff = np.polyfit(x, y, deg_eff)
                self.models[i] = {"kind": "poly", "degree": deg_eff, "coeff": coeff,
                                  "domain": (x[0], x[-1])}
        else:
            if knots is None:
                n_int = 8
                t_interior = np.linspace(x[0], x[-1], n_int + 2)[1:-1]
            else:
                t_interior = check_unit(knots, self.wavelength.unit).to_value(self.wavelength.unit)
            k = int(degree) if degree is not None else 3
            k = max(1, k)
            for i in range(n_fib):
                y = self.sigma_pix[i]
                if np.count_nonzero(np.isfinite(y)) < (k + 2):
                    coeff = np.array([np.nanmedian(y)])
                    self.models[i] = {"kind": "poly", "degree": 0, "coeff": coeff,
                                      "domain": (x[0], x[-1])}
                    continue
                y_fit = np.where(np.isfinite(y) & (y > 0), y,
                                 np.nanmedian(y[y > 0]) if np.any(y > 0) else 1.0)
                t = np.r_[ [x[0]]*k, t_interior, [x[-1]]*k ]
                tck = splrep(x, y_fit, k=k, t=t_interior, s=s)
                self.models[i] = {"kind": "spline", "k": tck[2], "t": tck[0], "c": tck[1],
                                  "domain": (x[0], x[-1])}

    def evaluate(self, wavelength, fibre=None, extrapolation="clamp"):
        """
        Evaluate sigma(lambda) from fitted models.

        Parameters
        ----------
        wavelength : Quantity array
        fibre : int or None
        extrapolation : {"clamp","error"}
            "clamp": clip wavelength to the model domain per fibre.
            "error": raise if any wavelength lies outside the domain.

        Returns
        -------
        sigma_pix_eval : ndarray of floats
        """
        if not self.models:
            raise RuntimeError("No models found. Run fit_models() or fit_from_sparse().")
        lam = check_unit(wavelength, self.wavelength.unit).to_value(self.wavelength.unit)
        n_fib = self.sigma_pix.shape[0]

        def eval_one(i):
            m = self.models.get(i, None)
            x0, x1 = m.get("domain", (self.wavelength.value[0], self.wavelength.value[-1]))
            if extrapolation == "error":
                if (lam.min() < x0) or (lam.max() > x1):
                    raise ValueError("Requested wavelengths outside model domain for fibre index "
                                     + str(i))
                xx = lam
            else:
                xx = np.clip(lam, x0, x1)
            if m["kind"] == "poly":
                return np.polyval(m["coeff"], xx)
            else:
                return BSpline(m["t"], m["c"], m["k"], extrapolate=False)(xx)

        if fibre is None:
            out = np.zeros((n_fib, lam.size), dtype=float)
            for i in range(n_fib):
                out[i] = eval_one(i)
            return out
        else:
            if fibre < 0 or fibre >= n_fib:
                raise IndexError("fibre out of range")
            return eval_one(fibre)

    def to_dense(self, wavelength=None, extrapolation="clamp"):
        """
        Return sigma_pix sampled on 'wavelength' (default: self.wavelength).
        Does not modify self.sigma_pix unless you assign the result.

        Returns
        -------
        sigma_pix_dense : ndarray (n_fibre, n_eval)
        """
        wl = self.wavelength if wavelength is None else check_unit(wavelength, self.wavelength.unit)
        return self.evaluate(wl, fibre=None, extrapolation=extrapolation)

    def degrade_to_worst(self, spectra_container: SpectraContainer, use_models=True,
                         n_knots=16, min_sigma=1e-3):
        """
        Convolve each fibre with a wavelength-dependent Gaussian so that
        all fibres reach the worst resolution at every wavelength.

        Steps:
          1) Compute sigma_target(lambda) = max_fibre sigma(lambda).
          2) For each fibre, compute delta_sigma(lambda) = sqrt(max^2 - sigma_fibre^2), clipped to >= 0.
          3) Perform variable-sigma convolution using K knot sigmas and linear blending
             of pre-filtered versions produced with gaussian_filter1d.

        Parameters
        ----------
        spectra_container : SpectraContainer
            Input data to be degraded. Must share the same wavelength grid as this model.
        use_models : bool
            If True, evaluate fitted models. If False, use stored sigma_pix samples.
        n_knots : int
            Number of sigma knots for blending (trade-off of speed vs fidelity).
        min_sigma : float
            Minimum sigma to use when filtering (prevents zero-sigma numerical edge cases).

        Returns
        -------
        out : SpectraContainer
            A new container with degraded rss_intensity and rss_variance.
        """
        assert isinstance(spectra_container, SpectraContainer)
        wave = spectra_container.wavelength
        if wave.size != self.wavelength.size or not np.allclose(
            wave.to_value(self.wavelength.unit), self.wavelength.to_value(self.wavelength.unit), rtol=0, atol=0
        ):
            raise ValueError("Wavelength grid must match between SpectraContainer and LSF model.")

        n_fib, n_wave = spectra_container.rss_intensity.shape

        # sigma arrays per fibre
        if use_models and self.models:
            sigma_f = self.evaluate(wave)  # (n_fib, n_wave)
        else:
            sigma_f = np.array(self.sigma_pix, dtype=float)

        # target is worst across fibres per wavelength
        sigma_tgt = np.nanmax(sigma_f, axis=0)  # (n_wave,)
        sigma_tgt = np.where(np.isfinite(sigma_tgt) & (sigma_tgt > 0), sigma_tgt, np.nanmax(sigma_tgt[np.isfinite(sigma_tgt)]))

        # output container
        out = spectra_container.copy()

        # helper: blend filtered versions computed at knot sigmas
        idx = np.linspace(0, n_wave - 1, n_knots).astype(int)
        idx[0] = 0
        idx[-1] = n_wave - 1

        # precompute piecewise linear weights for each pixel to two nearest knots
        # map pixel p -> left knot j and right knot j+1 and blend weight t in [0,1]
        knot_pos = idx.astype(float)
        pix = np.arange(n_wave, dtype=float)

        # find for each pixel the interval in knot_pos
        # use np.searchsorted over knot_pos
        right = np.searchsorted(knot_pos, pix, side="right")
        right[right <= 0] = 1
        right[right >= len(knot_pos)] = len(knot_pos) - 1
        left = right - 1
        # linear weights
        denom = knot_pos[right] - knot_pos[left]
        denom[denom == 0] = 1.0
        tblend = (pix - knot_pos[left]) / denom

        # process each fibre
        for i in range(n_fib):
            sig_i = sigma_f[i]
            # required additional sigma at each pixel
            delta = np.sqrt(np.maximum(sigma_tgt**2 - sig_i**2, 0.0))
            # enforce minimum
            delta = np.maximum(delta, 0.0)

            # knot sigma values for this fibre
            sigma_knots = np.maximum(delta[idx], min_sigma)

            # precompute filtered versions at each knot sigma
            spec = out.rss_intensity[i].value if hasattr(out.rss_intensity[i], "value") else out.rss_intensity[i]
            var = out.rss_variance[i].value if hasattr(out.rss_variance[i], "value") else out.rss_variance[i]

            filtered = np.empty((n_knots, n_wave), dtype=float)
            filtered_var = np.empty((n_knots, n_wave), dtype=float)

            for k in range(n_knots):
                s_k = float(sigma_knots[k])
                filtered[k] = gaussian_filter1d(spec, sigma=s_k, mode="nearest")
                # variance propagation approximation: filter variance with sigma / sqrt(2)
                filtered_var[k] = gaussian_filter1d(var, sigma=max(s_k / np.sqrt(2.0), min_sigma), mode="nearest")

            # blend between neighbor knot-filtered results
            y_left = filtered[left, np.arange(n_wave)]
            y_right = filtered[right, np.arange(n_wave)]
            spec_out = (1.0 - tblend) * y_left + tblend * y_right

            v_left = filtered_var[left, np.arange(n_wave)]
            v_right = filtered_var[right, np.arange(n_wave)]
            var_out = (1.0 - tblend) * v_left + tblend * v_right

            # store back with original units
            out.rss_intensity[i] = spec_out * spectra_container.intensity.unit
            out.rss_variance[i] = var_out * spectra_container.variance.unit

        # record simple note in correction log if available
        if hasattr(out, "record_correction"):
            comment = f"LSF degraded to worst across fibres using {n_knots} knots"
            try:
                out = out  # no-op; caller's pipeline may record elsewhere
            except Exception:
                pass

        return out

    # ---------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------
    def plot_sigma_image(self, show_worst=True, figsize=(9, 4.5),
                         vmin=None, vmax=None):
        """
        Plot sigma_pix as an image (fibre vs wavelength). Optionally overlay
        the worst sigma curve.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        n_fib, n_wave = self.sigma_pix.shape
        wl = self.wavelength.to_value(self.wavelength.unit)
        fig, ax = plt.subplots(figsize=figsize)
        im, cb = plot_image(fig, ax, self.sigma_pix,
                            xlabel=f"Wavelength [{self.wavelength.unit}]",
                            ylabel="Fibre index",
                            x=wl, y=np.arange(n_fib),
                            cblabel="sigma (pix)")
        if show_worst:
            worst = np.nanmax(self.sigma_pix, axis=0)
            ax.plot(wl, (n_fib - 1) * (worst - worst.min()) / (worst.ptp() + 1e-12) - 0.5,
                    ".-", color="r", alpha=0.4, lw=1.0, label="worst (scaled)")
            ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.close(fig)
        return fig


class WavelengthCorrection(CorrectionBase):
    """Wavelength correction class.

    This class accounts for the relative wavelength offset between fibres.

    Attributes
    ----------
    name : str
        Correction name, to be recorded in the log.
    offset : :class:`WavelengthOffset`
        2D wavelength offset (n_fibres x n_wavelengths)
    """

    name = "WavelengthCorrection"
    offset = None
    verbose = False

    def __init__(self, offset_path: str=None, offset: WavelengthOffset=None,
                 **correction_kwargs):
        super().__init__(**correction_kwargs)
        self.path = offset_path
        self.offset = offset

    @classmethod
    def from_fits(cls, path: str):
        """Initialise a WavelengthOffset correction from an input FITS file.
        
        Parameters
        ----------
        path : str
            Path to the FITS file containing the offset data.

        Returns
        -------
        wave_correction : :class:`WavelengthCorrection`
            A :class:`WavelengthCorrection` initialised with the input data.
        """
        return cls(offset=WavelengthOffset.from_fits(path=path),
                   offset_path=path)

    def apply(self, rss : RSS) -> RSS:
        """Apply a 2D wavelength offset model to a RSS.

        Parameters
        ----------
        rss : :class:`pykoala.rss.RSS`
            Original Row-Stacked-Spectra object to be corrected.

        Returns
        -------
        rss_corrected : :class:`pykoala.rss.RSS`
            Corrected copy of the input RSS.
        """

        assert isinstance(rss, RSS)

        if self.offset is None or self.offset.offset_data is None:
            raise ValueError("No offset loaded")
        
        rss_out = rss.copy()
        self.vprint("Applying correction to input RSS")

        if self.offset.offset_data.unit == u.pixel:
            x = np.arange(rss.wavelength.size) << u.pixel
        elif self.offset.offset_data.unit.is_equivalent(u.AA):
            x = rss.wavelength.to(self.offset.offset_data.unit)
        else:
            raise ValueError("Offset units must be pixel or wavelength")

        # per-fibre scalar or vector offsets
        off = self.offset.offset_data
        if off.ndim == 1:
            if off.size != rss.intensity.shape[0]:
                raise ValueError("offset_data shape is invalid for RSS")
            for i in range(rss.intensity.shape[0]):
                rss_out.intensity[i] = flux_conserving_interpolation(
                    x, x - off[i], rss.intensity[i]
                )
                if hasattr(rss, "variance") and rss.variance is not None:
                    rss_out.variance[i] = flux_conserving_interpolation(
                        x, x - off[i], rss.variance[i]
                    )
        elif off.ndim == 2:
            if off.shape != rss.intensity.shape:
                raise ValueError("2D offset_data must match RSS intensity shape")
            for i in range(rss.intensity.shape[0]):
                rss_out.intensity[i] = flux_conserving_interpolation(
                    x, x - off[i], rss.intensity[i]
                )
                if hasattr(rss, "variance") and rss.variance is not None:
                    rss_out.variance[i] = flux_conserving_interpolation(
                        x, x - off[i], rss.variance[i]
                    )
        else:
            raise ValueError("offset_data must be 1D or 2D")

        comment = f"wave-offset_unit={self.offset.offset_data.unit}; shape={self.offset.offset_data.shape}"
        self.record_correction(rss_out, status="applied", comment=comment)

        return rss_out


class TelluricWavelengthCorrection(WavelengthCorrection):
    """WavelengthCorrection based on the cross-correlation of telluric lines."""

    name = "TelluricWavelengthCorrection"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_rss(cls, rss : RSS,
                 median_smooth=None, pol_fit_deg=None, oversampling=5,
                 wave_range=None, plot=False):
        """Estimate the wavelength offset from an input RSS using telluric absorption.
        
        Parameters
        ----------
        rss : :class:`RSS`
        median_smooth : int, optional
            Median filter size.
        pol_fit_deg : int, optional
            Polynomial degree to fit the resulting offset.
        oversampling : float, optional
            Oversampling factor to increase the accuracy of the cross-correlation.
        wave_range : list or tupla
            Wavelength range to fit the offset.
        plot : bool, optional
            If True, returns quality control plots.

        Returns
        -------
        WavelengthCorrection : :class:`WavelengthCorrection`
        figs : :class:`plt.Figure`
        """
        assert isinstance(rss, RSS), "Input data must be an instance of RSS"

        # Normalize each spectrum
        intensity = rss.intensity.value.copy()
        med = np.nanmedian(intensity, axis=1)
        med[~np.isfinite(med)] = 1.0
        intensity /= med[:, np.newaxis]

        # Optional wavelength range mask
        mask = np.isfinite(rss.wavelength)
        if wave_range is not None:
            lo = check_unit(wave_range[0], rss.wavelength.unit)
            hi = check_unit(wave_range[1], rss.wavelength.unit)
            mask &= (rss.wavelength >= lo) & (rss.wavelength <= hi)

        # Oversample in wavelength index domain
        new_wavelength = np.interp(
            np.arange(0, rss.wavelength.size, 1.0 / oversampling),
            np.arange(rss.wavelength.size),
            rss.wavelength,
        )
        interpolator = interp1d(rss.wavelength, intensity, axis=1)
        intensity = interpolator(new_wavelength)

        # Restrict mask to resampled grid
        res_mask = np.interp(new_wavelength, rss.wavelength, mask.astype(float)) > 0.5
        # Reference median spectrum
        median_intensity = np.nanmedian(intensity, axis=0)

        fibre_offset = np.zeros(intensity.shape[0], dtype=float)
        for ith, fibre in enumerate(intensity):
            fibre_mask = np.isfinite(fibre) & res_mask
            if not fibre_mask.any():
                continue

            corr = correlate(fibre[fibre_mask], median_intensity[fibre_mask],
                             mode="full", method="fft")
            lags = correlation_lags(fibre[fibre_mask].size,
                                    median_intensity[fibre_mask].size)
            max_corr = np.argmax(corr)
            # guard edges for parabolic interpolation
            i0 = max(max_corr - 1, 0)
            i1 = max_corr
            i2 = min(max_corr + 1, corr.size - 1)
            x3 = lags[[i0, i1, i2]]
            y3 = corr[[i0, i1, i2]]
            # if duplicates happen at edges, skip parabolic and take argmax
            if (i0 == i1) or (i1 == i2) or (i0 == i2):
                peak = lags[max_corr]
            else:
                peak = ancillary.parabolic_maximum(x3, y3)
            fibre_offset[ith] = peak / oversampling

        if median_smooth is not None:
            fibre_offset = median_filter(fibre_offset, size=median_smooth)

        if pol_fit_deg is not None:
            x = np.arange(fibre_offset.size, dtype=float)
            pol = np.polyfit(x, fibre_offset, deg=pol_fit_deg)
            fibre_offset = np.poly1d(pol)(x)

        figs = None
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(fibre_offset)
            ax.set_ylim(np.nanmin(fibre_offset), np.nanmax(fibre_offset))
            ax.set_xlabel("Fibre number")
            ax.set_ylabel("Average wavelength offset (pixel)")
            fibre_map_fig = rss.plot_fibres(data=fibre_offset)
            figs = (fig, fibre_map_fig)
            plt.close(fig)

        fibre_offset = fibre_offset << u.pixel
        offset = WavelengthOffset(
            offset_data=fibre_offset, offset_error=np.full_like(fibre_offset, fill_value=np.nan)
        )
        return cls(offset=offset), figs


class SolarCrossCorrOffset(WavelengthCorrection):
    """WavelengthCorrection based on solar spectra cross-correlation.
    
    Constructs a WavelengthOffset using a cross-correlation between a solar
    reference spectrum and a twilight exposure (dominated by solar features).

    Also implements Line Spread Function (LSF) estimation from the solar spectrum.
    """
    name = "SolarCrossCorrelationOffset"

    def __init__(self, sun_wavelength, sun_intensity, **kwargs):
        super().__init__(offset=WavelengthOffset(), **kwargs)
        self.sun_wavelength = check_unit(sun_wavelength, u.AA)
        self.sun_intensity = check_unit(sun_intensity,
                                        u.erg / u.s / u.AA / u.cm**2)
        self._lsf_knots_wave = None
        self._lsf_knots_sigma_pix = None
        self._lsf_poly_coeff = None
        self._lsf_poly_deg = None

    @classmethod
    def from_fits(cls, path=None, extension=1):
        """Initialise a WavelengthOffset correction from an input FITS file.
        
        Parameters
        ----------
        path : str, optional
            Path to the FITS file containing the reference Sun's spectra. The
            file must contain an extension with a table including a ``WAVELENGTH``
            and ``FLUX`` columns.The wavelength array must be angstrom in the
            vacuum frame.
        extension : int or str, optional
            HDU extension containing the table. Default is 1.

        Returns
        -------
        solar_offset_correction : :class:`SolarCrossCorrOffset`
            An instance of SolarCrossCorrOffset.
        """
        if path is None:
            path = os.path.join(os.path.dirname(__file__), '..',
                     'input_data', 'spectrophotometric_stars',
                     'sun_mod_001.fits')
        with fits.open(path) as hdul:
            sun_wavelength = hdul[extension].data['WAVELENGTH'] << u.AA
            sun_wavelength = vac_to_air(sun_wavelength)
            sun_intensity = hdul[extension].data['FLUX'] << u.erg / u.s / u.AA / u.cm**2
        return cls(sun_wavelength=sun_wavelength,
                   sun_intensity=sun_intensity)

    @classmethod
    def from_text_file(cls, path, loadtxt_args={}):
        """Initialise a :class:`SolarCrossCorrOffset` correction from an input text file.
        
        Parameters
        ----------
        path: str
            Path to the text file containing the reference Sun's spectra. The
            text file must contain two columns consisting of the
            vacuum wavelength array in angstrom and the solar flux or luminosity.
        loadtxt_args: dict, optional
            Additional arguments to be passed to ``numpy.loadtxt``.

        Returns
        -------
        solar_offset_correction: :class:`SolarCrossCorrOffset`
            An instance of SolarCrossCorrOffset.
        """
        sun_wavelength, sun_intensity = np.loadtxt(path, unpack=True,
                                                   usecols=(0, 1),
                                                   **loadtxt_args)
        #TODO: Handle units
        sun_wavelength = vac_to_air(sun_wavelength)
        return cls(sun_wavelength=sun_wavelength,
                   sun_intensity=sun_intensity)


    def get_solar_features(self, solar_wavelength, solar_spectra,
                            window_size_aa=20):
        """
        Estimate the regions of the solar spectrum dominated by absorption features.

        Description
        -----------
        First, a median filter is applied to estimate the upper envelope of the
        continuum. Then, the median ratio between the solar spectra and the median-filtered
        estimate is used to compute the relative weights:

        .. math::
            \\begin{equation}
                w = \\left\\|\\frac{F_\\odot}{F_{cont}} - Median(\\frac{F_\\odot}{F_{cont}})\\right\\|
            \\end{equation}

        Parameters
        ----------
        solar_wavelength: numpy.ndarray
            Solar spectra wavelengths array.
        solar_spectra: numpy.ndarray
            Array containing the flux of the solar spectra associated to a given
            wavelength.
        window_size_aa: int, optional
            Size of a spectral window in angstrom to perform a median filtering
            and estimate the underlying continuum. Default is 20 AA.

        Returns
        -------
        weights: numpy.ndarray
            Array of weights representing the prominance of an absorption feature.

        """
        self.vprint("Estimating regions of solar spectra dominated by absorption lines.")
        # convert window size to pixels over the current wavelength span
        delta_pixel = int(
            (
                check_unit(window_size_aa, u.AA)
                / (solar_wavelength[-1] - solar_wavelength[0])
                * solar_wavelength.size
            ).decompose()
        )
        if delta_pixel % 2 == 0:
            delta_pixel += 1

        solar_continuum = median_filter(solar_spectra, size=delta_pixel) << solar_spectra.unit
        ratio = solar_spectra / (solar_continuum + ancillary.EPSILON_FLOAT64 * solar_continuum.unit)
        # Detect absorption features
        median_continuum_ratio = np.nanmedian(ratio)
        weights = np.abs(ratio -  median_continuum_ratio)
        s = np.nansum(weights)
        if s > 0:
            weights = weights / s
        return weights

    def compute_grid_of_models(self, pix_shift_array, pix_std_array, pix_array,
                               sun_intensity, weights, interp_weights=True):
        """
        Compute a (shift, sigma, pixel) grid of broadened, shifted solar models
        and their corresponding line-feature weights.

        Parameters
        ----------
        pix_shift_array : Quantity or array-like, shape (n_shift,)
            Pixel shifts (in pixels). If dimensionless, interpreted as pixels.
        pix_std_array : Quantity or array-like, shape (n_sigma,)
            Gaussian LSF sigmas (in pixels). If dimensionless, interpreted as pixels.
        pix_array : Quantity, shape (n_pix,)
            Native pixel coordinate array (monotonic).
        sun_intensity : Quantity, shape (n_pix,)
            Solar spectrum sampled on ``pix_array`` grid.
        weights : ndarray or Quantity, shape (n_pix,)
            Relative line-feature weights on the same grid as ``sun_intensity``.
            Will be transported consistently with shift and sigma when
            ``interp_weights=True``.
        interp_weights : bool, optional
            If True (default), propagate ``weights`` via the same shift and
            Gaussian smoothing (truncated) applied to the model and renormalize
            per model slice. If False, reuse the same ``weights`` slice for all
            grid points.

        Returns
        -------
        models_grid : Quantity, shape (n_shift, n_sigma, n_pix)
            Broadened and shifted solar models.
        weights_grid : ndarray, shape (n_shift, n_sigma, n_pix)
            Propagated line weights for each model slice (sum≈1 per slice).
        """
        models_grid = np.zeros(
            (pix_shift_array.size, pix_std_array.size, sun_intensity.size)
            ) << sun_intensity.unit
        weights_grid = np.zeros(
            (pix_shift_array.size, pix_std_array.size, sun_intensity.size)
            )
        shift_idx, std_idx = np.indices(models_grid.shape[:-1])

        for z, (velshift, gauss_std) in enumerate(
            zip(pix_shift_array[shift_idx.flatten()], pix_std_array[std_idx.flatten()])):

                i, j = np.unravel_index(z, models_grid.shape[:-1])

                new_pixel_array = pix_array + velshift
                interp_sun_intensity = flux_conserving_interpolation(
                    new_pixel_array, pix_array, sun_intensity)
                # gaussian_filter expects sigma in pixels
                interp_sun_intensity = gaussian_filter(
                    interp_sun_intensity, gauss_std.value)
                # Restore the intensity units removed by gaussian_filter
                models_grid[i, j] = interp_sun_intensity << sun_intensity.unit
                if interp_weights:
                    # propagate weights consistently, truncate to reduce long wings
                    interp_sun_weight = flux_conserving_interpolation(
                        new_pixel_array, pix_array, weights)
                    interp_sun_weight = gaussian_filter(
                        interp_sun_weight, gauss_std.value, truncate=2.0)
                    norm = np.nansum(interp_sun_weight)
                    if norm > 0:
                        interp_sun_weight /= norm
                else:
                    interp_sun_weight = weights
                weights_grid[i, j] = interp_sun_weight

        return models_grid, weights_grid

    def _build_overlapping_windows(self, wave, n_windows, window_overlap):
        """
        Create overlapping wavelength windows over the provided 1D wave array.

        Returns
        -------
        centers : list of Quantities (wavelength units)
        slices  : list of slice objects selecting indices per window
        """
        w0 = wave[0]
        w1 = wave[-1]
        # equal-size windows in wavelength space
        centers = np.linspace(w0, w1, n_windows + 2)[1:-1]  # exclude ends
        half_span = (w1 - w0) / (n_windows * 2.0)
        half_span = half_span * (1.0 + window_overlap)

        slices = []
        for c in centers:
            lo = c - half_span
            hi = c + half_span
            # indices in wave range
            i0 = int(np.searchsorted(wave, lo))
            i1 = int(np.searchsorted(wave, hi))
            i0 = max(i0, 0)
            i1 = min(i1, wave.size)
            if i1 - i0 < 5:
                continue
            slices.append(slice(i0, i1))
        return centers, slices

    def _mask_telluric_lines(self, wave, width=30 << u.angstrom,
                             model_file=None):
        """
        Build a boolean mask that flags pixels near strong telluric features.

        Parameters
        ----------
        wave : Quantity, shape (n_wave,)
            Instrument wavelength grid.
        width : Quantity, optional
            Half-width added on each side of each telluric band from the model file.
            Default is 30 AA.
        model_file : str or None, optional
            Path to a two-column text file with start/end wavelengths for
            telluric bands (Angstrom). If None, uses the packaged default
            file under ``input_data/sky_lines/telluric_lines.txt``.

        Returns
        -------
        mask : ndarray of bool, shape (n_wave,)
            True where wavelengths are considered clean (to keep), False where
            pixels lie inside broadened telluric bands (to mask out).
        """
        if model_file is None:
            model_file = os.path.join(os.path.dirname(__file__), '..',
                                     'input_data', 'sky_lines',
                                     'telluric_lines.txt')
        
        w_l_1, w_l_2 = np.loadtxt(model_file, unpack=True, usecols=(0, 1))
        w_l_1 = w_l_1 << u.angstrom
        w_l_2 = w_l_2 << u.angstrom
        mask = np.ones(wave.size, dtype=bool)
        for b, r in zip(w_l_1, w_l_2):
            mask[(wave >= b - width) & (wave <= r + width)] = False
        return mask

    def fit_solar_spectra(
        self,
        spectra_container: SpectraContainer,
        n_windows=1,
        window_overlap=0.25,
        win_pixel_valid_frac=0.2,
        response_window_size_aa=200,
        mask_tellurics=False,
        smooth_sigma_pix=None,
        smooth_shift_pix=None,
        make_lsf_model=False,
        **fit_kwargs
    ):
        """
        Fit the reference solar spectrum to each fibre in a ``SpectraContainer``,
        window by window, returning per-window wavelength shifts and LSF sigmas.

        The method:
        1) Interpolates the stored solar spectrum to the instrument wavelength grid.
        2) Estimates and divides out the per-fibre response using a median filter.
        3) Splits the bandpass into overlapping wavelength windows.
        4) In each window and fibre, fits the solar spectrum (shift + Gaussian LSF sigma)
            using :func:`pykoala.ancillary.fit_reference_spectra`.
        5) Optionally smooths the per-window tracks and constructs a
            :class:`FibreLSFModel` from the sparse sigma measurements.

        Parameters
        ----------
        spectra_container : SpectraContainer
            Input data (RSS or cube reduced to RSS-like arrays). Must expose
            ``wavelength`` (Quantity, shape (n_wave,)),
            ``rss_intensity`` (Quantity, shape (n_fibre, n_wave)),
            and ``rss_variance`` (Quantity, shape (n_fibre, n_wave)).
        n_windows : int, optional
            Number of overlapping windows across the full wavelength range.
            If ``n_windows==1`` the full band is treated as a single window.
            Default is 1.
        window_overlap : float, optional
            Fractional half-overlap between adjacent windows in [0, 1).
            Default is 0.25.
        win_pixel_valid_frac : float, optional
            Minimum fraction of non-masked pixels per wavelength. Default is ``0.2``.
        response_window_size_aa : float or Quantity, optional
            Median-filter scale (in AA) used to estimate the response per fibre.
            Default is 200 AA.
        mask_tellurics : bool, optional
            If True, a built-in telluric mask is applied before fitting.
            Default is False.
        smooth_sigma_pix : int or None, optional
            Median filter size (in pixels along the window axis) applied to the
            resulting sigma matrix. If ``None`` or 0, no smoothing is applied.
            Default is None.
        smooth_shift_pix : int or None, optional
            Median filter size (in pixels along the window axis) applied to the
            resulting shift matrix. If ``None`` or 0, no smoothing is applied.
            Default is None.
        make_lsf_model : bool, optional
            If True, builds a :class:`FibreLSFModel` from the sparse per-window
            sigma estimates and returns it in the results. Default is False.
        **fit_kwargs
            Extra keyword arguments forwarded to
            :func:`pykoala.ancillary.fit_reference_spectra` (e.g. priors, bounds,
            optimization options).

        Returns
        -------
        results : dict
            Dictionary with fields:
            - ``"wave"`` : Quantity (n_wave,)
                Original instrument wavelength vector.
            - ``"normalized_rss"`` : Quantity (n_fibre, n_wave)
                Flux after response normalization.
            - ``"normalized_rss_var"`` : Quantity (n_fibre, n_wave)
                Variance after response normalization.
            - ``"response"`` : Quantity (n_fibre, n_wave)
                Estimated response curves.
            - ``"windows"`` : dict
                Metadata with keys:
                * ``"centers"``: list of Quantity window centers,
                * ``"slices"`` : list of Python ``slice`` objects per window.
            - ``"fibre_<i>"`` : list
                One entry per window; each entry is either ``None`` (fit skipped)
                or a dict returned by ``fit_reference_spectra`` augmented with
                scalar keys ``"shift_pix"`` and ``"sigma_pix"``.
            - ``"shift_pix_matrix"`` : ndarray (n_fibre, n_windows)
                Per-fibre, per-window best-fit shifts (pixels, floats).
            - ``"sigma_pix_matrix"`` : ndarray (n_fibre, n_windows)
                Per-fibre, per-window best-fit sigmas (pixels, floats).
            - ``"lsf_model"`` : FibreLSFModel, optional
                Present only if ``make_lsf_model=True``.

        Notes
        -----
        * Windows with fewer than ~5 valid pixels are skipped and appear as ``None``.
        * If response normalization encounters invalid values, per-fibre medians are
        used as fallbacks to keep computations finite.
        * The method is agnostic to absolute solar flux scale since the spectra
        are response-normalized prior to fitting.

        Raises
        ------
        ValueError
            If inputs are inconsistent (e.g. invalid ``n_windows`` or overlap).
        RuntimeError
            If no valid wavelength windows can be constructed.
        """
        self.vprint("Fitting reference solar spectra to SpectraContainer.")

        # Basic input checks
        if n_windows is None or int(n_windows) < 1:
            raise ValueError("n_windows must be >= 1.")
        if not (0.0 <= float(window_overlap) < 1.0):
            raise ValueError("window_overlap must be in [0, 1).")
        
        self.vprint(f"Number of windows: {n_windows}, overlap fraction: {window_overlap}")

        wave = spectra_container.wavelength
        n_fibre, n_wave = spectra_container.rss_intensity.shape
        results = {}
        # ----- Windows -----
        centers, slices = self._build_overlapping_windows(wave, n_windows,
                                                          window_overlap)
        if len(slices) == 0:
            raise RuntimeError("No valid wavelength windows were built."
                               " Check n_windows and overlap.")
        results["windows"] = {"centers": centers, "slices": slices}
        # Interpolate solar spectrum onto current wavelength grid
        sun_spectra = flux_conserving_interpolation(
            wave, self.sun_wavelength, self.sun_intensity
        )

        response = spectra_container.rss_intensity / sun_spectra[np.newaxis]
        # ----- Response normalization -----
        delta_pixel = int(
            (
                check_unit(response_window_size_aa, u.AA)
                / (wave[-1] - wave[0])
                * n_wave
            ).decompose()
        )
        if delta_pixel % 2 == 0:
            delta_pixel += 1

        self.vprint("Renormalizing spectra by solar spectrum using window size "
                    + f"{response_window_size_aa} AA ({delta_pixel} pixels).")
        # Smooth and estimate upper envelope along wavelength axis
        smooth_resp = median_filter(response, size=(1, delta_pixel)) << response.unit
        bad_resp = ~np.isfinite(smooth_resp.value) | (smooth_resp.value <= 0)
        if np.any(bad_resp):
            # Replace with per-fibre median of finite response
            for i in range(n_fibre):
                row = smooth_resp.value[i]
                bad = ~np.isfinite(row) | (row <= 0)
                if np.any(bad):
                    med = np.nanmedian(row[~bad]) if np.any(~bad) else 1.0
                    row[bad] = med
            smooth_resp = (smooth_resp.value * smooth_resp.unit)

        normalized_rss = spectra_container.rss_intensity / smooth_resp
        normalized_rss_var = spectra_container.rss_variance / (smooth_resp ** 2)
        results["response"] = smooth_resp
        results["wave"] = wave
        results["normalized_rss"] = normalized_rss
        results["normalized_rss_var"] = normalized_rss_var

        if mask_tellurics:
            self.vprint("Masking telluric lines.")
            telluric_mask = self._mask_telluric_lines(wave)
            normalized_rss.value[:, ~telluric_mask] = np.nan
            normalized_rss_var.value[:, ~telluric_mask] = np.nan

        self.vprint("Fitting solar spectrum to each fibre and window.")
        for i in range(n_fibre):
            flux = normalized_rss[i].value.copy()
            var = normalized_rss_var[i].value.copy()

            mask = np.isfinite(flux) & np.isfinite(var) & (var > 0)
            flux[~mask] = 0.0
            var[~mask] = np.inf

            results[f"fibre_{i}"] = []
            for s in slices:
                n_valid_s = np.count_nonzero(mask[s])
                n_valid_frac = n_valid_s / mask[s].size

                if n_valid_frac < win_pixel_valid_frac:
                    self.vprint(f"Fibre {i} Window {s}: Not enough valid pixels ({n_valid_s}), skipping.")
                    results[f"fibre_{i}"].append(None)
                    continue
                try:
                    res = ancillary.fit_reference_spectra(
                        wave.value[s], flux[s], flux_var=var[s],
                        ref_wave=self.sun_wavelength.value,
                        ref_flux=self.sun_intensity.value,
                        **fit_kwargs
                    )
                    res["n_valid_pix"] = n_valid_s
                    res["n_valid_frac"] = n_valid_s / mask[s].size

                except Exception as e:
                    self.vprint(f"Fibre {i} Window {s}: Fit failed: {e}")
                    res = None
                results[f"fibre_{i}"].append(res)
        
        # Build the matrix of shift and sigma values
        shift_pix = np.full((n_fibre, len(slices)), fill_value=np.nan)
        sigma_pix = np.full((n_fibre, len(slices)), fill_value=np.nan)
        for i in range(n_fibre):
            per_win = results.get(f"fibre_{i}", [])
            for j, res in enumerate(per_win):
                if res is None:
                    continue
                shift_pix[i, j] = res.get("shift_pix", np.nan)
                sigma_pix[i, j] = res.get("sigma_pix", np.nan)

        if smooth_shift_pix is not None and smooth_shift_pix > 0:
            self.vprint(f"Smoothing shift_pix with median filter ({smooth_shift_pix} pix)")
            shift_pix = median_filter(shift_pix, size=smooth_shift_pix,
                                      axes=0)
        if smooth_sigma_pix is not None and smooth_sigma_pix > 0:
            self.vprint(f"Smoothing sigma_pix with median filter ({smooth_sigma_pix} pix)")
            sigma_pix = median_filter(sigma_pix, size=smooth_sigma_pix,
                                      axes=0)

        results["shift_pix_matrix"] = shift_pix
        results["sigma_pix_matrix"] = sigma_pix

        # Compute the median shift // TODO: create a 2D offset model.
        median_shift = np.nanmedian(shift_pix, axis=1)
        if not np.isfinite(median_shift).all():
            good = np.isfinite(median_shift)
            x = np.arange(median_shift.size)
            median_shift = np.interp(x, x[good], median_shift[good])
        self.offset.offset_data = median_shift << u.pixel

        if make_lsf_model:
            lsf_model = FibreLSFModel.from_sparse(
                wave, centers, sigma_values=sigma_pix, kind="spline")
            results["lsf_model"] = lsf_model
        return results

    def plot_fit_for_fibre(
        self,
        fibre_idx,
        fit_results,
        max_windows=None,
        show_models=True,
        show_residuals=True,
        show_weights=True,
        figsize=(10, 6),
    ):
        """
        Plot per-window fits for one fibre and a summary of shift/sigma across windows.

        Parameters
        ----------
        fibre_idx : int
            RSS fibre index to display.
        fit_results : dict
            Output of fit_lsf_and_shift(...).
        max_windows : int or None
            If set, limit to first N windows for quick inspection.
        show_models : bool
            Overlay fitted model on each window panel.
        show_residuals : bool
            Add residuals panel under each window plot.
        show_weights : bool
            Overlay 1/variance as a proxy for weights (normalized per-window).
        figsize : tuple
            Figure size for each window figure.

        Returns
        -------
        figs : list[matplotlib.figure.Figure]
            One per-window figure plus a final summary figure.
        """
        # Basic checks on inputs
        if fit_results is None or not isinstance(fit_results, dict):
            raise ValueError("fit_results must be the dictionary returned by fit_lsf_and_shift.")

        windows = fit_results.get("windows", {})
        centers = windows.get("centers", None)
        slices_ = windows.get("slices", None)
        if centers is None or slices_ is None:
            raise ValueError("fit_results lacks 'windows' metadata (centers/slices).")

        wave = fit_results.get("wave", None)
        norm = fit_results.get("normalized_rss", None)
        varn = fit_results.get("normalized_rss_var", None)
        if wave is None or norm is None or varn is None:
            raise ValueError("fit_results must contain 'wave', 'normalized_rss', and 'normalized_rss_var'.")

        if fibre_idx < 0 or fibre_idx >= norm.shape[0]:
            raise IndexError("fibre_idx out of range.")

        fibre_key = f"fibre_{fibre_idx}"
        if fibre_key not in fit_results:
            raise KeyError(f"Missing per-window results for {fibre_key} in fit_results.")

        per_win = fit_results[fibre_key]
        n_win_total = len(slices_)
        if len(per_win) != n_win_total:
            # Inconsistent storage
            raise RuntimeError("Mismatch between number of windows and stored fibre results.")

        use_windows = n_win_total if max_windows is None else min(max_windows, n_win_total)
        figs = []

        # Window-by-window plots
        for w in range(use_windows):
            sl = slices_[w]
            lam = wave[sl]
            data = norm[fibre_idx, sl]
            varw = varn[fibre_idx, sl].value if hasattr(varn, "value") else varn[fibre_idx, sl]
            res = per_win[w]

            # Prepare axes
            nrows = 2 if show_residuals else 1
            fig, axes = plt.subplots(
                nrows, 1, figsize=figsize, sharex=True,
                gridspec_kw=dict(height_ratios=[2, 1] if show_residuals else [1])
            )
            ax1 = axes if nrows == 1 else axes[0]
            ax1.set_title(
                f"Fibre {fibre_idx}  Window {w+1}/{n_win_total} "
                f"center={centers[w]:.1f}  "
                f"shift={res.get('shift_pix', np.nan):.2f} pix  "
                f"sigma={res.get('sigma_pix', np.nan):.2f} pix  "
                f"Fit: {'OK' if res.get('success', False) else 'FAIL'}"
            )

            # Data
            ax1.plot(lam, data, lw=1.0, color="k", label="data")
            if show_models and ("model" in res) and (res["model"] is not None):
                ax1.plot(lam, res["model"], lw=1.0, color="r", label="model")

            # Optional weights overlay (1/variance normalized)
            if show_weights:
                invw = 1.0 / np.clip(varw, np.finfo(float).tiny, np.inf)
                if np.any(np.isfinite(invw)) and np.nansum(invw) > 0:
                    invw = invw / np.nanmax(invw)  # normalize 0..1
                    tw = ax1.twinx()
                    tw.plot(lam, invw, color="orange", alpha=0.5, label="inv var (norm)")
                    tw.set_ylabel("weight (norm)")
                    # Local legends
                    ax1.legend(loc="upper left", fontsize=8)
                    tw.legend(loc="upper right", fontsize=8)
                else:
                    ax1.legend(loc="upper left", fontsize=8)
            else:
                ax1.legend(loc="upper left", fontsize=8)

            ax1.set_ylabel("Normalized flux")

            # Residuals
            if show_residuals and ("model" in res) and (res["model"] is not None):
                ax2 = axes[1]
                # data may have units; model is unitless float. Convert safely to float for residual.
                data_v = data.value if hasattr(data, "value") else data
                resid = data_v - np.asarray(res["model"], dtype=float)
                ax2.plot(lam, resid, lw=0.9, color="tab:gray")
                ax2.axhline(0.0, color="k", lw=0.7, alpha=0.6)
                ax2.set_ylabel("Residual")
                ax2.set_xlabel(f"Wavelength [{wave.unit}]")
            else:
                ax1.set_xlabel(f"Wavelength [{wave.unit}]")

            plt.tight_layout()
            plt.close(fig)
            figs.append(fig)

        # Summary panel: shift and sigma vs window centers, with optional error bars
        centers_val = np.array([c.to_value(wave.unit) for c in centers], dtype=float)
        shift_arr = np.array([r.get("shift_pix", np.nan) for r in per_win], dtype=float)
        sigma_arr = np.array([r.get("sigma_pix", np.nan) for r in per_win], dtype=float)

        # Try to infer simple uncertainties from covariance if present
        shift_err = np.full_like(shift_arr, np.nan, dtype=float)
        sigma_err = np.full_like(sigma_arr, np.nan, dtype=float)
        for w in range(n_win_total):
            cov = per_win[w].get("cov", None)
            if cov is not None and np.ndim(cov) == 2 and cov.size >= 4:
                # We expect parameter order like [shift, sigma, ...]; pull diag
                try:
                    shift_err[w] = np.sqrt(max(cov[0, 0], 0.0))
                    sigma_err[w] = np.sqrt(max(cov[1, 1], 0.0))
                except Exception:
                    pass

        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(211)
        ax1.set_title(f"Fibre {fibre_idx}  shift(lambda) and sigma(lambda) per window")
        # Shift
        if np.any(np.isfinite(shift_err)):
            ax1.errorbar(centers_val, shift_arr, yerr=shift_err, fmt="o", ms=4, label="shift ± err")
        else:
            ax1.plot(centers_val, shift_arr, "o", ms=4, label="shift")
        ax1.set_ylabel("shift (pix)")
        ax1.legend(loc="best", fontsize=8)

        ax2 = fig.add_subplot(212, sharex=ax1)
        # Sigma
        if np.any(np.isfinite(sigma_err)):
            ax2.errorbar(centers_val, sigma_arr, yerr=sigma_err, fmt="o", ms=4, label="sigma ± err")
        else:
            ax2.plot(centers_val, sigma_arr, "o", ms=4, label="sigma")
        ax2.set_xlabel(f"Wavelength center [{wave.unit}]")
        ax2.set_ylabel("sigma (pix)")
        ax2.legend(loc="best", fontsize=8)

        plt.tight_layout()
        plt.close(fig)
        figs.append(fig)

        return figs


    def compute_shift_from_twilight(self, spectra_container,
                                    sun_window_size_aa=20, keep_features_frac=0.1,
                                    response_window_size_aa=200,
                                    wave_range=None,
                                    pix_shift_array=None,
                                    pix_std_array=None,
                                    logspace=True, use_mean=True,
                                    inspect_fibres=None):
        """Compute the wavelenght offset of between a given SpectraContainer and a reference Solar spectra.
        
        Parameters
        ----------
        spectra_container: `pykoala.data_container.SpectraContainer`
            Spectra container (RSS or Cube) to cross-correlate with the reference
            spectra.
        sun_window_size_aa: int, optional
            Size of a spectral window in angstrom to perform a median filtering
            and estimate the underlying continuum. Default is 20 AA.
            See `get_solar_features` for details.
        keep_features_frac: float, optional
            Fraction of absorption-features weights to keep. All values below
            that threshold will be set to 0. Default is 0.1.
        wave_range: list or tuple, optional
            If provided, the cross-correlation will only be done in the provided
            wavelength range. Default is None.
        pix_shift_array: 1D-np.array, optional, default=np.arange(-5, 5, 0.1)
            Array containing the wavelength offsets expressed in pixels.
        pix_std_array: 1D-np.array, optional, default=np.arange(0.1, 3, 0.1)
            Array containing the values of the gaussian LSF standard deviation
            in pixels. See `compute_grid_of_models` for details.
        logspace: bool, optional
            If True, the cross-correlation will be perform using a logarithmic
            sampling in terms of wavelength. Default is True.
        use_mean: bool, optional
            If True, the mean likelihood-weighted value of the wavelength offset
            is used to create the `WavelengthOffsetCorrection`. Otherwise, the
            best fit parameters of the input grid are used. Default is True.
        inspect_fibres: list or tuple, optional
            Iterable containing RSS-wise spectra indices. If provided, a
            quality-control plot of each fibre is produced.
        
        Returns
        -------
        results: dict
            The dictionary contains the ``best-fit`` and ``mean`` likelihood-weighted
            values of ``pix_shift_array`` and ``pix_std_array`` in a tuple, respectively.
            If ``inspect_fibres`` is not ``None``, it containes a list of figures
            for each fibre included in ``inspect_fibres``.

        """
        if pix_shift_array is None:
            pix_shift_array = np.arange(-5, 5, 0.1)
        if pix_std_array is None:
            pix_std_array = np.arange(0.1, 3, 0.1)

        pix_shift_array = check_unit(pix_shift_array, u.pixel)
        pix_std_array = check_unit(pix_std_array, u.pixel)

        if logspace:
            new_wavelength = np.geomspace(spectra_container.wavelength[0],
                                          spectra_container.wavelength[-1],
                                          spectra_container.wavelength.size)
            rss_intensity = np.array([flux_conserving_interpolation(
                new_wavelength, spectra_container.wavelength, fibre
                ) for fibre in spectra_container.rss_intensity]
                ) << spectra_container.intensity.unit
        else:
            new_wavelength = spectra_container.wavelength
            rss_intensity = spectra_container.rss_intensity
        
        # Interpolate the solar spectrum to the new grid of wavelengths
        sun_intensity = flux_conserving_interpolation(
        new_wavelength, self.sun_wavelength, self.sun_intensity)

        # Make an array of weights to focus on the absorption lines
        if wave_range is None:
            weights = self.get_solar_features(new_wavelength, sun_intensity,
                                            window_size_aa=sun_window_size_aa)
            weights[weights < np.nanpercentile(weights, 100*(1 - keep_features_frac))] = 0
            weights[:100] = 0
            weights[-100:] = 0
        else:
            weights = np.zeros(new_wavelength.size)
            weights[slice(*np.searchsorted(new_wavelength, wave_range))] = 1.0

        valid_pixels = weights > 0
        self.vprint("Number of pixels with non-zero weights: "
                    + f"{np.count_nonzero(valid_pixels)} out of {valid_pixels.size}")

        # Estimate the response curve for each individual fibre
        delta_pixel = int(
            (check_unit(response_window_size_aa, u.AA)
                        / (new_wavelength[-1] - new_wavelength[0])
                        * new_wavelength.size).decompose()
                        )
        if delta_pixel % 2 == 0:
            delta_pixel += 1

        response_spectrograph = rss_intensity / sun_intensity[np.newaxis]
        smoothed_r_spectrograph = median_filter(
            response_spectrograph, delta_pixel, axes=1) << response_spectrograph.unit
        spectrograph_upper_env = percentile_filter(
            smoothed_r_spectrograph, 95, delta_pixel, axes=1) << response_spectrograph.unit
        # Avoid regions dominated by telluric absorption
        self.vprint("Including the masking of pixels dominated by telluric absorption")
        fibre_weights =  1 / (1  + (
                spectrograph_upper_env / smoothed_r_spectrograph
                - np.nanmedian(spectrograph_upper_env / smoothed_r_spectrograph)
                )**2)

        normalized_rss_intensity = rss_intensity / smoothed_r_spectrograph
        # Generate and fit the model
        pix_array = np.arange(new_wavelength.size) << u.pixel

        models_grid, weights_grid = self.compute_grid_of_models(
            pix_shift_array, pix_std_array, pix_array, sun_intensity, weights)

        # loop over one variable to avoir memory errors
        all_chi2 = np.zeros((pix_shift_array.size,
                             pix_std_array.size,
                             rss_intensity.shape[0]))
        
        self.vprint("Performing the cross-correlation with the grid of models")
        for i in range(pix_shift_array.size):
            all_chi2[i] = np.nansum(
                (models_grid[i, :, np.newaxis]
                 - normalized_rss_intensity[np.newaxis, :, :]).value**2
                * weights_grid[i, :, np.newaxis]
                * fibre_weights[np.newaxis, :, :],
                axis=-1) / np.nansum(
                    weights_grid[i, :, np.newaxis]
                    * fibre_weights[np.newaxis, :, :],
                    axis=-1)
            
        likelihood = np.exp(- (all_chi2 - all_chi2.min())/ 2)
        likelihood /= np.nansum(likelihood, axis=(0, 1))[np.newaxis, np.newaxis, :]
        mean_pix_shift = np.sum(likelihood.sum(axis=1)
                                * pix_shift_array[:, np.newaxis], axis=0)
        mean_sigma = np.sum(likelihood.sum(axis=0)
                          * pix_std_array[:, np.newaxis], axis=0)

        best_fit_idx = np.argmax(likelihood.reshape((-1, likelihood.shape[-1])),
                                 axis=0)
        best_vel_idx, best_std_idx = np.unravel_index(
                best_fit_idx, all_chi2.shape[:-1])
        best_sigma, best_shift = (pix_std_array[best_std_idx],
                                    pix_shift_array[best_vel_idx])

        if inspect_fibres is not None:
            fibre_figures = self.inspect_fibres(
                inspect_fibres, pix_shift_array, pix_std_array,
                best_vel_idx, best_std_idx, mean_pix_shift, mean_sigma,
                likelihood, models_grid, weights_grid, normalized_rss_intensity,
                new_wavelength)
        else:
            fibre_figures= None
        if use_mean:
            self.vprint("Using mean likelihood-weighted values to compute the wavelength offset correction")
            self.offset.offset_data = - mean_pix_shift
        else:
            self.vprint("Using best fit values to compute the wavelength offset correction")
            self.offset.offset_data = - best_shift
        
        self.offset.offset_error = np.full_like(best_shift, fill_value=np.nan)

        return {"best-fit": (best_shift, best_sigma),
                "mean": (mean_pix_shift, mean_sigma),
                "fibre_figures": fibre_figures}
    
    def inspect_fibres(self, fibres, pix_shift_array, pix_std_array,
                       best_vel_idx, best_std_idx, mean_vel, mean_sigma,
                       likelihood,
                       models_grid, weights_grid,
                       normalized_rss_intensity, wavelength):
        """
        Create a quality control plot of the solar cross-correlation process of each input fibre.

        Parameters
        ----------
        fibres: iterable
            List of input fibres to check.
        pix_shift_array: 1D-np.array
            Array containing the wavelength offsets expressed in pixels.
        pix_std_array: 1D-np.array
            Array containing the values of the gaussian LSF standard deviation
            in pixels. See :func:`compute_grid_of_models` for details.
        best_vel_idx: int
            Index of ``pix_shift_array`` that correspond to the best fit.
        best_std_idx: int
            Index of ``pix_std_array`` that correspond to the best fit.
        mean_vel: float
            Mean likelihood-weighted values of ``pix_shift_array``.
        mean_sigma: float
            Mean likelihood-weighted values of ``pix_std_array``.
        likelihood: numpy.ndarray:
            Likelihood of the cross-correlation.
        models_grid: numpy.ndarray
            Grid of solar spectra models. See :func:`compute_grid_of_models` for details.
        weights_grid: numpy.ndarray
            Grid of solar spectra weights. See :func:`compute_grid_of_models` for details.
        normalized_rss_intensity: numpy.ndarray
            Array containing the RSS intensity values of a SpectraContainer including
            the correction of the spectrograph response curve.
        wavelength: np.array
            Wavelength array associated to ``normalized_rss_intensity`` and ``models_grid``.

        Returns
        -------
        fibres_figures: list
            List of figures containing a QC plot of each fibre.
        """
        fibre_figures = []
        best_sigma, best_shift = (pix_std_array[best_std_idx],
                                  pix_shift_array[best_vel_idx])
        for fibre in fibres:
            self.vprint(f"Inspecting input fibre: {fibre}")
            fig = plt.figure(constrained_layout=True, figsize=(10, 8))
            gs = GridSpec(2, 4, figure=fig, wspace=0.25, hspace=0.25)

            ax = fig.add_subplot(gs[0, 0])
            mappable = ax.pcolormesh(
                pix_std_array, pix_shift_array, likelihood[:, :, fibre],
                cmap='gnuplot',
                norm=LogNorm(vmin=likelihood.max() / 1e2, vmax=likelihood.max()))
            plt.colorbar(mappable, ax=ax,
                         label=r"$e^(-\sum_\lambda w(I - \hat{I}(s, \sigma))^2 / 2)$")
            ax.plot(best_sigma[fibre], best_shift[fibre], '+', color='cyan',
                    label=r'Best fit: $\Delta\lambda$='
                    + f'{best_shift[fibre]:.2}, ' + r'$\sigma$=' + f'{best_sigma[fibre]:.2f}')
            ax.plot(mean_sigma[fibre], mean_vel[fibre], 'o', mec='lime', mfc='none',
                    label=r'Mean value: $\Delta\lambda$='
                    + f'{mean_vel[fibre]:.2}, ' + r'$\sigma$=' + f'{mean_sigma[fibre]:.2f}')
            ax.set_xlabel(r"$\sigma$ (pix)")
            ax.set_ylabel(r"$\Delta \lambda$ (pix)")
            ax.legend(bbox_to_anchor=(0., 1.05), loc='lower left', fontsize=7)

            sun_intensity = models_grid[best_vel_idx[fibre],
                                        best_std_idx[fibre]]
            weight = weights_grid[best_vel_idx[fibre],
                                        best_std_idx[fibre]]

            ax = fig.add_subplot(gs[0, 1:])
            ax.set_title(f"Fibre: {fibre}")
            ax.plot(wavelength, sun_intensity, label='Sun Model')
            ax.plot(wavelength, normalized_rss_intensity[fibre],
                    label='Fibre', lw=2)
            twax = ax.twinx()
            twax.plot(wavelength, weight, c='fuchsia',
                    zorder=-1, alpha=0.5, label='Weight')
            ax.legend(fontsize=7)
            ax.set_ylabel("Intensity")
            ax.set_xlabel("Wavelength")
            twax.set_ylabel("Relative weight")

            ax = fig.add_subplot(gs[1, :])
            max_idx = np.argmax(weight)
            max_weight_range = range(np.max((max_idx - 80, 0)),
                  np.min((max_idx + 80, wavelength.size - 1)))
            ax.plot(wavelength[max_weight_range],
                    sun_intensity[max_weight_range], label='Model')
            ax.plot(wavelength[max_weight_range],
                    normalized_rss_intensity[fibre][max_weight_range],
                    label='Fibre', lw=2)
            
            ax.set_xlim(wavelength[max_weight_range][0],
                        wavelength[max_weight_range][-1])
            twax = ax.twinx()
            twax.plot(wavelength[max_weight_range], weight[max_weight_range],
                      c='fuchsia',
                    zorder=-1, alpha=0.5, label='Absorption-feature Weight')
            twax.axhline(0)
            twax.legend(fontsize=7)
            ax.set_ylabel("Intensity")
            ax.set_xlabel("Wavelength")
            twax.set_ylabel("Relative weight")
            fibre_figures.append(fig)
            plt.close(fig)
        return fibre_figures


# =============================================================================
# Mr Krtxo \(ﾟ▽ﾟ)/
#                                                       ... Paranoy@ Rulz! ;^D
