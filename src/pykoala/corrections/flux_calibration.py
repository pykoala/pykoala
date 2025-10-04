"""
Flux calibration utilities.

This module provides tools to derive and apply an absolute (or relative) flux
calibration by estimating and fitting an instrumental spectral response curve
as a function of wavelength.

It includes:
- StandardStar: a container for spectrophotometric reference stars
- FluxCalibration: extraction, response fitting, mastering, and application
- Helper utilities (curve_of_growth, plotting)
"""

# =============================================================================
# Basics packages
# =============================================================================
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any, List, Union

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter, percentile_filter
from scipy.interpolate import interp1d, make_smoothing_spline

import matplotlib.pyplot as plt
import os
from astropy import units as u
from astropy.visualization import quantity_support
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.io import fits
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala import vprint
from pykoala.corrections.correction import CorrectionBase
from pykoala.data_container import SpectraContainer, RSS, Cube
from pykoala.utils.spectra import estimate_continuum_and_mask_absorption
from pykoala.utils.math import std_from_mad
from pykoala.utils.io import suppress_warnings
from pykoala.ancillary import (centre_of_mass, cumulative_1d_moffat,
                               mask_lines, mask_telluric_lines,
                               flux_conserving_interpolation, check_unit)
from pykoala.plotting import utils as plt_utils

quantity_support()

@dataclass
class StandardStar:
    """
    Spectrophotometric standard star for flux calibration.

    Parameters
    ----------
    name : str
        Canonical name of the star, for example ``"feige34"``.
    catalog : str
        Source catalog identifier, for example ``"CALSPEC"``, ``"Oke1990"``, ``"ESO"``.
    wavelength : `astropy.units.Quantity`
        Wavelength grid, typically in Angstrom.
    flux : `astropy.units.Quantity`
        Reference flux density on the same grid as ``wavelength``.
    flux_err : `astropy.units.Quantity`, optional
        Flux uncertainty, same unit and length as ``flux``.
    file_path : str, optional
        Origin file path, if loaded from disk.
    bibcode : str, optional
        Reference code for citation.
    meta : dict, optional
        Additional metadata, for example date, version, link, instrument, notes.

    Notes
    -----
    On initialization, arrays are validated, nonfinite samples are dropped,
    and the wavelength axis is sorted ascending if needed.
    """
    name: str
    catalog: str
    wavelength: u.Quantity
    flux: u.Quantity
    flux_err: Optional[u.Quantity] = None
    file_path: Optional[str] = None
    bibcode: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.wavelength = check_unit(self.wavelength, u.AA)
        if not isinstance(self.flux, u.Quantity):
            raise TypeError("flux must be an astropy Quantity")
        if self.flux_err is not None and not isinstance(self.flux_err, u.Quantity):
            raise TypeError("flux_err must be an astropy Quantity or None")

        if self.wavelength.ndim != 1 or self.flux.ndim != 1:
            raise ValueError("wavelength and flux must be 1D arrays")
        if self.wavelength.shape != self.flux.shape:
            raise ValueError("wavelength and flux must have the same shape")
        if self.flux_err is not None and self.flux_err.shape != self.flux.shape:
            raise ValueError("flux_err must match flux shape")

        finite = np.isfinite(self.wavelength.value) & np.isfinite(self.flux.to_value(self.flux.unit))
        if self.flux_err is not None:
            finite &= np.isfinite(self.flux_err.to_value(self.flux.unit))
        if not np.any(finite):
            raise ValueError("no finite samples in standard star arrays")

        self.wavelength = self.wavelength[finite]
        self.flux = self.flux[finite]
        if self.flux_err is not None:
            self.flux_err = self.flux_err[finite]

        # ensure ascending order
        order = np.argsort(self.wavelength.value)
        self.wavelength = self.wavelength[order]
        self.flux = self.flux[order]
        if self.flux_err is not None:
            self.flux_err = self.flux_err[order]

    @property
    def wave_unit(self) -> u.UnitBase:
        """Wavelength unit."""
        return self.wavelength.unit

    @property
    def flux_unit(self) -> u.UnitBase:
        """Flux unit."""
        return self.flux.unit

    @property
    def wave_range(self) -> Tuple[u.Quantity, u.Quantity]:
        """Minimum and maximum wavelength."""
        return self.wavelength.min(), self.wavelength.max()

    @classmethod
    def from_ascii(cls, path: str, name: str, catalog: str = None,
                   wave_unit: u.UnitBase = u.AA,
                   flux_unit: Optional[u.UnitBase] = None,
                   has_error: bool = False,
                   header_units: bool = True,
                   **meta) -> "StandardStar":
        """
        Load a star from a two or three column ASCII file.

        Parameters
        ----------
        path : str
            File path. Columns must be wavelength, flux [, flux_err].
        name : str
            Star name.
        catalog : str
            Catalog identifier.
        wave_unit : `astropy.units.Unit`, optional
            Wavelength unit if not present in the file header.
        flux_unit : `astropy.units.Unit`, optional
            Flux unit if not present in the file header. If None, defaults to
            ``1e-16 erg s-1 cm-2 AA-1``.
        has_error : bool, optional
            Whether the table has a third column with flux errors.
        header_units : bool, optional
            If True, attempt to parse units from header comments.

        Returns
        -------
        star : StandardStar
            Loaded standard star.
        """
        data = np.loadtxt(path, dtype=float, comments="#")
        if data.ndim != 2 or data.shape[1] not in (2, 3):
            raise ValueError(f"unrecognized ASCII shape {data.shape}")

        w = data[:, 0]
        f = data[:, 1]
        fe = data[:, 2] if (has_error or data.shape[1] == 3) else None

        wunit = wave_unit
        funit = flux_unit
        if header_units:
            try:
                with open(path, "r") as fh:
                    lines = [fh.readline() for _ in range(4)]
                for line in lines:
                    if "wavelength" in line.lower() and "(" in line and ")" in line:
                        wunit = u.Unit(line.split("(")[1].split(")")[0])
                    if "flux" in line.lower() and "(" in line and ")" in line:
                        fidx = line.lower().find("flux")
                        funit = u.Unit(line[fidx:].split("(")[1].split(")")[0])
                    if "catalog" in line.lower():
                        catalog = line.strip().split("=")[1]
            except Exception:
                vprint("Could not read units from header")
                raise ValueError("Could not read units from header")
                pass

        if funit is None:
            funit = 1e-16 * u.erg / u.s / u.cm**2 / u.AA

        return cls(
            name=name,
            catalog=catalog,
            wavelength=w * wunit,
            flux=(f * funit).to("1e-16 erg / (s * cm^2 * angstrom)"),
            flux_err=(fe * funit if fe is not None else None),
            file_path=path,
            meta=meta or {},
        )

    def to_qtable(self) -> QTable:
        """
        Convert to a QTable.

        Returns
        -------
        table : `astropy.table.QTable`
            Table with columns ``wavelength``, ``flux``, and optional ``flux_err``.
        """
        tab = QTable()
        tab["wavelength"] = self.wavelength
        tab["flux"] = self.flux
        if self.flux_err is not None:
            tab["flux_err"] = self.flux_err
        tab.meta = dict(
            name=self.name,
            catalog=self.catalog,
            file_path=self.file_path,
            bibcode=self.bibcode,
            **(self.meta or {}),
        )
        return tab

    @classmethod
    def from_qtable(cls, table: QTable, name: str, catalog: str, **meta) -> "StandardStar":
        """
        Build from a QTable.

        Parameters
        ----------
        table : `astropy.table.QTable`
            Table with columns ``wavelength`` and ``flux`` (optional ``flux_err``).
        name : str
            Star name.
        catalog : str
            Catalog identifier.

        Returns
        -------
        star : StandardStar
        """
        w = table["wavelength"]
        f = table["flux"]
        fe = table["flux_err"] if "flux_err" in table.colnames else None
        return cls(
            name=name,
            catalog=catalog,
            wavelength=w,
            flux=f,
            flux_err=fe,
            meta={**table.meta, **meta},
        )

    def save_fits(self, path: str) -> None:
        """
        Save to FITS.

        Parameters
        ----------
        path : str
            Output FITS path.
        """
        self.to_qtable().write(path, format="fits", overwrite=True)

    @classmethod
    def read_fits(cls, path: str, name: str, catalog: str) -> "StandardStar":
        """
        Read from FITS created by :meth:`save_fits`.

        Parameters
        ----------
        path : str
            FITS path.
        name : str
            Star name.
        catalog : str
            Catalog identifier.

        Returns
        -------
        star : StandardStar
        """
        tab = QTable.read(path, format="fits")
        meta = dict(tab.meta)
        meta["file_loaded_from"] = path
        return cls.from_qtable(tab, name=name, catalog=catalog, **meta)

    def resample(self, new_wave: u.Quantity, conserve_flux: bool = True) -> u.Quantity:
        """
        Interpolate the reference flux onto a new wavelength grid.

        Parameters
        ----------
        new_wave : `astropy.units.Quantity`
            Target wavelength grid.
        conserve_flux : bool, optional
            If True, use flux-conserving interpolation when available.

        Returns
        -------
        ref_flux_on_new : `astropy.units.Quantity`
            Interpolated flux density on ``new_wave``.
        """
        new_wave = check_unit(new_wave, self.wave_unit)
        if conserve_flux:
            return flux_conserving_interpolation(new_wave, self.wavelength, self.flux)
        f = interp1d(self.wavelength.to_value(new_wave.unit),
                     self.flux.to_value(self.flux.unit),
                     bounds_error=False, fill_value="extrapolate")
        return f(new_wave.to_value(new_wave.unit)) << self.flux.unit

    def subset(self, wave_min: u.Quantity, wave_max: u.Quantity) -> "StandardStar":
        """
        Return a restricted wavelength slice.

        Parameters
        ----------
        wave_min, wave_max : `astropy.units.Quantity`
            Inclusive wavelength limits.

        Returns
        -------
        star : StandardStar
            New instance restricted to the provided range.
        """
        wave_min = check_unit(wave_min, self.wave_unit)
        wave_max = check_unit(wave_max, self.wave_unit)
        m = (self.wavelength >= wave_min) & (self.wavelength <= wave_max)
        return StandardStar(
            name=self.name,
            catalog=self.catalog,
            wavelength=self.wavelength[m],
            flux=self.flux[m],
            flux_err=(self.flux_err[m] if self.flux_err is not None else None),
            file_path=self.file_path,
            bibcode=self.bibcode,
            meta=dict(self.meta),
        )

    def plot_spectra(self, show: bool = False):
        """
        Plot the stellar spectrum.

        Parameters
        ----------
        show : bool, optional
            If True, display the figure instead of returning it.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
        """
        fig, axs = plt_utils.new_figure(self.name)
        ax = axs[0]
        ax.plot(self.wavelength, self.flux, label="Flux")
        if self.flux_err is not None:
            ax.plot(self.wavelength, self.flux_err, label="Flux err")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.legend()
        if show:
            plt.show()
        else:
            plt.close()
        return fig


def list_available_stars(verbose=True):
    """
    Lists all available spectrophotometric standard star files.

    Parameters
    ----------
    verbose : bool, optional
        If True, prints the list of available stars.

    Returns
    -------
    stars : list
        List of available spectrophotometric standard stars.
    """
    files = os.listdir(os.path.join(os.path.dirname(__file__), '..',
                                    'input_data', 'spectrophotometric_stars'))
    files = np.sort(files)
    names = []
    for file in files:
        names.append(file.split('.')[0].split('_')[0])
        if verbose:
            vprint(" - Standard star file: {}\n   · Name: {}"
                    .format(file, names[-1]))
    return np.array(names), files

def find_standard_star(name=None, path=None):
    """
    Find the file associated to a standard star.

    Parameters
    ----------
    name : str, optional
        Name of the calibration star.
    path : str, optional
        Path to the file containing the calibration star spectra.

    Returns
    -------
    star : StandardStar
    """
    if path is None and name is None:
        raise ValueError("Provide either a star `name` or a file `path`.")

    if path is None:
        nm = name.strip().lower()
        if nm[0] != 'f' or 'feige' in nm:
            nm = 'f' + nm
        all_names, files = list_available_stars(verbose=False)
        m = np.where(all_names == nm)[0]
        if len(m) == 0:
            raise FileNotFoundError(f"Calibration star not found: {name}")
        pick = files[m[-1]]
        path = os.path.join(os.path.dirname(__file__), '..', 'input_data',
                        'spectrophotometric_stars', pick)

    if os.path.isfile(path):
        return StandardStar.from_ascii(path, name=name, catalog="N/A")
    else:
        raise FileNotFoundError(f"File not found: {path}")

def curve_of_growth(radii : u.Quantity, data : u.Quantity,
                    ref_radii : u.Quantity, mask=None) -> u.Quantity:
    """
    Compute a curve of growth (cumulative flux versus radius) and evaluate it
    at the requested reference radii.

    The algorithm sorts samples by radius, applies the mask (if given), computes
    the cumulative sum of the data over increasing radius, and linearly
    interpolates that cumulative sum to the reference radii. The returned curve
    is non decreasing by construction when data are non negative.

    Parameters
    ----------
    radii
        Radial distance of each sample. Must be a 1D astropy Quantity. It will
        be converted to the unit of ref_radii.
    data
        Flux value of each sample. Must be a 1D astropy Quantity with the same
        length as radii. The returned curve has the same unit as data.
    ref_radii
        Radii at which to evaluate the curve of growth. Must be a 1D astropy
        Quantity. radii will be converted to this unit for sorting and
        interpolation.
    mask
        Optional boolean array with the same length as radii. Samples with
        mask False are ignored. If None, all samples are considered.

    Returns
    -------
    cog
        Curve of growth evaluated at ref_radii, as an astropy Quantity with the
        same unit as data. If no valid samples are available, zeros are
        returned.

    Notes
    -----
    1. Non finite values in radii or data are ignored.
    2. Duplicate radii are handled by taking the cumulative sum at the last
       occurrence of each unique radius.
    3. Linear interpolation is used between unique radii. Values below the
       smallest radius are set to zero. Values above the largest radius are
       set to the total cumulative flux.

    Raises
    ------
    ValueError
        If radii and data shapes differ, or if mask has an incompatible shape.
    """
    if mask is None:
        mask = np.ones(data.size, dtype=bool)

    radii = check_unit(radii, ref_radii.unit)
    r_val = radii.to_value(ref_radii.unit)
    d_val = data.value
    data_unit = data.unit

    valid = np.isfinite(r_val) & np.isfinite(d_val)
    if mask is not None:
        valid &= mask.astype(bool)

    if not np.any(valid):
        # No usable samples
        return np.zeros(ref_radii.size, dtype=float) * data_unit

    sort_idx = np.argsort(r_val)
    r_sorted = r_val[sort_idx]
    d_sorted = d_val[sort_idx]
    v_sorted = valid[sort_idx]

    # Zero out invalid samples and accumulate
    d_sorted = np.where(v_sorted, d_sorted, 0.0)
    cumsum = np.cumsum(d_sorted)
    # Map each unique radius to the last cumulative value at or below it
    u_r = np.unique(r_sorted)
    # index of last element <= each unique radius
    last_idx = np.searchsorted(r_sorted, u_r, side="right") - 1
    cog_at_u = cumsum[last_idx]
    # Enforce non decreasing curve (robust to small negative values if present)
    cog_at_u = np.maximum.accumulate(cog_at_u)
    # Interpolate to requested radii
    ref_vals = np.interp(
        ref_radii.to_value(ref_radii.unit),
        u_r,
        cog_at_u,
        left=0.0,
        right=cog_at_u[-1] if cog_at_u.size > 0 else 0.0,
    )
    return ref_vals * data_unit


class FluxCalibration(CorrectionBase):
    """
    A class to handle the extraction and application of absolute flux calibration.

    Attributes
    ----------
    name : str
        The name of the Correction.
    verbose : bool
        If True, prints additional information during execution.
    calib_spectra : None or array-like
        The calibration spectra data.
    calib_wave : None or array-like
        The calibration wavelength data.
    response : None or array-like
        The response data.
    response_wavelength : None or array-like
        The response wavelength data.
    response_units : float
        Units of the response function, default is 1e16 (erg/s/cm^2/AA).
    """
    name = "FluxCalibration"

    def __init__(self, response=None, response_err=None, response_wavelength=None,
                 response_file=None, **correction_kwargs):
        super().__init__(**correction_kwargs)
        
        self.vprint("Initialising Flux Calibration (Spectral Throughput)")

        self.response_wavelength = check_unit(response_wavelength, u.angstrom)
        self.response = check_unit(response)
        if response_err is not None:
            self.response_err = check_unit(response_err, self.response.unit)
        else:
            self.response_err = np.zeros_like(self.response)

        self.response_file = response_file

    @classmethod
    def from_text_file(cls, path: str) -> "FluxCalibration":
        """
        Load a response function from a text file.

        The file must contain two or three columns: wavelength, response [,
        response_err]. The first two commented header lines may optionally encode
        units in parentheses.

        Parameters
        ----------
        path : str
            Path to the text file.

        Returns
        -------
        flux_calibration : FluxCalibration
            Instance with ``response``, ``response_err`` and ``response_wavelength`` set.
        """
        data = np.loadtxt(path, dtype=float, comments="#")
        if data.ndim != 2 or data.shape[1] not in (2, 3):
            raise ArithmeticError(f"Unrecognized shape: {data.shape}")

        wavelength = data[:, 0]
        response = data[:, 1]
        response_err = data[:, 2] if data.shape[1] == 3 else np.zeros_like(response)  # fixed

        wave_unit = u.AA
        resp_unit = u.dimensionless_unscaled
        with open(path, "r") as f:
            _ = f.readline()
            line = f.readline()
            if line.startswith("#"):
                parts = [p.strip() for p in line[1:].split(",")]
                if len(parts) >= 2:
                    widx = parts[0].find("("), parts[0].rfind(")")
                    ridx = parts[1].find("("), parts[1].rfind(")")
                    if widx[0] != -1 and widx[1] != -1:
                        wave_unit = u.Unit(parts[0][widx[0] + 1:widx[1]])
                    if ridx[0] != -1 and ridx[1] != -1:
                        resp_unit = u.Unit(parts[1][ridx[0] + 1:ridx[1]])

        return cls(response=response * resp_unit,
                response_err=response_err * resp_unit,
                response_wavelength=wavelength * wave_unit,
                response_file=path)

    @classmethod
    def auto(cls, data: List, standards: List[Union[str, StandardStar]],
             extract_args: Optional[dict] = None,
             response_params: Optional[dict] = None,
             combine: bool = False):
        """
        Build response curves from standard star observations.

        This method accepts either a list of StandardStar instances or a list of
        star names. When names are provided, the stars are loaded using the
        internal library via :meth:`read_calibration_star` and wrapped in
        StandardStar objects.

        Parameters
        ----------
        data : list of DataContainer
            Observations of standard stars (RSS or Cube), one per star.
        standards : list of {str, StandardStar}
            Either a list of star names (to be loaded from the internal
            spectrophotometric library) or a list of StandardStar instances.
            The list length must match ``data``.
        extract_args : dict, optional
            Parameters passed to :meth:`extract_stellar_flux`.
            Default is ``{"wave_range": None, "wave_window": None, "plot": False}``.
        response_params : dict, optional
            Parameters passed to :meth:`get_response_curve`.
            Default is ``{"pol_deg": 5, "plot": False}``.
        combine : bool, optional
            If True, combine the individual responses into a master response.

        Returns
        -------
        results : dict
            Per-star results, keyed by star name. Includes extraction output,
            wavelength grid, response array, and a QA figure.
        responses : list of FluxCalibration
            Individual ``FluxCalibration`` instances, one per star.
        master : FluxCalibration or None
            Master response if ``combine`` is True, else None.

        Raises
        ------
        ValueError
            If the lengths of ``data`` and ``standards`` disagree.
        """
        if extract_args is None:
            # Default extraction arguments
            extract_args = dict(wave_range=None, wave_window=None, plot=False)
        if response_params is None:
            # Default response curve parameters
            response_params = dict(plot=False)
        
        if len(data) != len(standards):
            raise ValueError("data and standards must have the same length")

        std_stars = []
        for item in standards:
            if isinstance(item, StandardStar):
                std_stars.append(item)
            elif isinstance(item, str):
                std_stars.append(find_standard_star(item))
            else:
                raise TypeError(
                    "standards must be a list of StandardStar or str names; "
                    f"got element of type {type(item)}"
                )

        results: Dict[str, dict] = {}
        responses: List[FluxCalibration] = []

        for obs, star in zip(data, std_stars):
            vprint(f"Automatic calibration process for {star.name}")
            # Extract flux from observed std star
            vprint("Extracting stellar flux from data")
            extract = cls.extract_stellar_flux(obs.copy(), **extract_args)
            results[star.name] = dict(std_star=star, extraction=extract)

            # Compute the instrumental response with the StandardStar
            resp_fn, resp_err_fn, fig = cls.get_response_curve(
            obs_wave=obs.wavelength,
            obs_spectra=extract["stellar_flux"],
            star=star,
            **response_params,
            )
        
            results[star.name]["wavelength"] = obs.wavelength.copy()
            results[star.name]["response"] = resp_fn(obs.wavelength.copy())
            results[star.name]["response_err"] = resp_err_fn(obs.wavelength.copy())
            results[star.name]["response_fig"] = fig

            responses.append(
                cls(response=resp_fn(obs.wavelength),
                    response_wavelength=obs.wavelength)
            )

        if not responses:
            vprint("No flux calibration was created", level="warning")
            return None, None, None

        master = cls.master_flux_auto(responses) if combine else None
        return results, responses, master

    @classmethod
    def master_flux_auto(cls, flux_calibration_corrections: list,
                         combine_method: str = "median"):
        """
        Combine multiple response curves into a master response.

        Parameters
        ----------
        flux_calibration_corrections : list of FluxCalibration
            Individual responses to combine.
        combine_method : {"median", "mean", "wmean"}, optional
            Combination strategy.

        Returns
        -------
        master : FluxCalibration
            Master response with propagated uncertainty.
        """
        vprint("Mastering response function")
        if len(flux_calibration_corrections) == 1:
            return flux_calibration_corrections[0]

        # Select a reference wavelength array
        ref_wave = flux_calibration_corrections[0].response_wavelength
        ref_resp_unit = flux_calibration_corrections[0].response.unit
        # Place holder
        spectral_response = np.full((len(flux_calibration_corrections),
                                         ref_wave.size), fill_value=np.nan,
                                         dtype=float)
        spectral_response_err = np.full_like(spectral_response,
                                        fill_value=np.nan)

        spectral_response[0] = flux_calibration_corrections[0].response.to_value(ref_resp_unit)
        spectral_response_err[1] = flux_calibration_corrections[0].response_err.to_value(ref_resp_unit)

        for ith, fcal_corr in enumerate(flux_calibration_corrections[1:]):
            resp, resp_err = fcal_corr.interpolate_response(ref_wave)
            spectral_response[ith + 1] = resp.to_value(ref_resp_unit)
            spectral_response_err[ith + 1] = resp_err.to_value(ref_resp_unit)

        if combine_method == "median":
            master_resp = np.nanmedian(spectral_response, axis=0)
            mad = 1.4826 * np.nanmedian(
                np.abs(spectral_response - master_resp), axis=0)
            n_eff = np.sum(np.isfinite(spectral_response), axis=0).clip(min=1)
            master_resp_err = 1.2533 * mad / np.sqrt(n_eff)
        elif combine_method == "mean":
            master_resp = np.nanmean(spectral_response, axis=0)
            n_eff = np.sum(np.isfinite(spectral_response), axis=0).clip(min=1)
            master_resp_err = np.sqrt(
                np.nansum(spectral_response_err**2, axis=0) / n_eff)
        elif combine_method == "wmean":
            w = 1.0 / np.where(
                np.isfinite(spectral_response_err) & (spectral_response_err > 0),
                spectral_response_err**2, np.nan)
            num = np.nansum(w * spectral_response, axis=0)
            den = np.nansum(w, axis=0)
            master_resp = num / den
            master_resp_err = 1.0 / np.sqrt(den)
        else:
            raise ValueError(f"Unknown combine_method: {combine_method}")

        return FluxCalibration(response=master_resp << ref_resp_unit,
                               response_err=master_resp_err << ref_resp_unit,
                               response_wavelength=ref_wave)

    @staticmethod
    @suppress_warnings(categories=RuntimeWarning)
    def extract_stellar_flux(data_container,
                             wave_range=None, wave_window=None,
                             profile=cumulative_1d_moffat,
                             bounds: tuple=None,
                             growth_r : u.Quantity=None,
                             plot=False, **fitter_args):
        """
        Extract the stellar flux from an RSS or Cube.

        Parameters
        ----------
        data_container : DataContainer
            Source for extracting the stellar flux.
        wave_range : list, optional
            Wavelength range to use for the flux extraction.
        wave_window : int, optional
            Wavelength window size for averaging the input flux.
        profile : function, optional
            Profile function to model the cumulative light profile. Any function
            that accepts as first argument
            the square distance (r^2) and returns the cumulative profile can be
            used. Default is cumulative_1d_moffat.
        bounds : tuple, optional
            Bounds for the curve fit. If ``None``, the bounds will be estimated
            automatically.
        growth_r : :class:`astropy.units.Quantity`, optional
            Reference radial bins relative to the center of the star that will
            be used to compute the curve of growth. Default ranges from 0.5 to 
            10 arcsec in steps of 0.5 arcsec.
        plot : bool, optional
            If True, shows a plot of the fit for each wavelength step.
        fitter_args : dict, optional
            Extra arguments to be passed to scipy.optimize.curve_fit

        Returns
        -------
        result : dict
            Dictionary containing the extracted flux and other related data.
        """

        wavelength = data_container.wavelength
        wave_mask = np.ones(wavelength.size, dtype=bool)
        if wave_range is not None:
            wave_mask[
                (wavelength < check_unit(wave_range[0], wavelength.unit)
                ) | (wavelength > check_unit(wave_range[1], wavelength.unit))
                ] = False
        if wave_window is None:
            wave_window = 1
        if growth_r is None:
            growth_r = np.arange(0.5, 10, 0.5) << u.arcsec
        if bounds is None:
            bounds = "auto"

        wavelength = wavelength[wave_mask]

        # Curve of growth radial bins
        r_dummy = check_unit(growth_r, u.arcsec)

        vprint("Extracting star flux.\n"
                + " -> Wavelength range={}\n".format(wave_range)
                + " -> Wavelength window={}\n".format(wave_window))

        # Formatting the data
        if isinstance(data_container, RSS):
            vprint("Extracting flux from RSS")
            data = data_container.intensity.copy()
            variance = data_container.variance.copy()
            # Invert the matrix to get the wavelength dimension as 0.
            data, variance = data.T, variance.T
            x = data_container.info['fib_ra'].copy()
            y = data_container.info['fib_dec'].copy()
        elif isinstance(data_container, Cube):
            vprint("Extracting flux from input Cube")
            data = data_container.intensity.copy()
            data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
            variance = data_container.variance.copy()
            variance = variance.reshape((variance.shape[0], variance.shape[1]
                                         * variance.shape[2]))
            x, y = np.indices((data_container.n_rows, data_container.n_cols))
            x, y = x.flatten(), y.flatten()
            skycoords = data_container.wcs.celestial.array_index_to_world(x, y)
            x, y = skycoords.ra, skycoords.dec
        else:
            raise NameError(
                f"Unrecongnized datacontainer of type {type(data_container)}")
        # Convert x and y to relative offset position wrt the mean centre
        xy_coords = SkyCoord(ra=x, dec=y, frame="icrs")

        data = data[wave_mask, :]
        variance = variance[wave_mask, :]

        # Declare variables
        mean_residuals = np.full(wavelength.size, fill_value=np.nan) << data.unit
        star_flux = np.full_like(mean_residuals, fill_value=np.nan)
        profile_popt = []
        profile_var = []
        running_wavelength = []
        cog = []
        cog_var = []
        # Fitting tolerance and number of evaluations
        if 'ftol' not in fitter_args.keys():
            fitter_args['ftol'] = 0.001
            fitter_args['xtol'] = 0.001
            fitter_args['gtol'] = 0.001
        if 'max_nfev' not in fitter_args.keys():
            fitter_args['max_nfev'] = 1000
 
        # Loop over all spectral slices
        vprint("...Fitting wavelength chuncks...")
        for lambda_ in range(0, wavelength.size, wave_window):
            wave_slice = slice(lambda_, lambda_ + wave_window, 1)
            wave_edges = [wavelength[wave_slice][0].to_value("AA"),
                          wavelength[wave_slice][-1].to_value("AA")]
            slice_data = data[wave_slice]
            slice_var = variance[wave_slice]
            # Compute the median value only considering good values (i.e. excluding NaN)
            slice_data = np.nanmedian(slice_data, axis=0)
            slice_var = np.nanmedian(slice_var, axis=0) / np.sqrt(slice_var.shape[0])
            slice_var[slice_var <= 0] = np.inf << slice_var.unit

            mask = np.isfinite(slice_data) & np.isfinite(slice_var) & (slice_data > 0)
            if not mask.any():
                vprint("Chunk between {} to {} AA contains no useful data"
                        .format(wave_edges[0], wave_edges[1]))
                continue

            x0, y0 = centre_of_mass(slice_data[mask], x[mask], y[mask])
            # Make the growth curve
            distance = xy_coords.separation(
                SkyCoord(ra=x0, dec=y0, frame="icrs"))

            growth_c = curve_of_growth(distance, slice_data, r_dummy, mask)
            growth_c_var = curve_of_growth(distance, slice_var, r_dummy, mask)

            cog_mask = np.isfinite(growth_c) & (growth_c > 0)
            if not cog_mask.any():
                vprint("Chunk between {} to {} AA contains no useful data"
                        .format(wave_edges[0], wave_edges[1]))
                continue
            # Profile fit
            try:
                if bounds == 'auto':
                    # vprint("Automatic fit bounds")
                    p_bounds = ([0, 0, 0],
                                [growth_c[cog_mask][-1].value * 2,
                                 r_dummy.to_value("arcsec").max(), 4])
                else:
                    p_bounds = bounds
                # Initial guess
                p0=[growth_c[cog_mask][-1].value,
                    r_dummy.to_value("arcsec").mean(), 1.0]
                popt, pcov = curve_fit(
                    profile, r_dummy[cog_mask].to_value("arcsec"),
                    growth_c[cog_mask].value, bounds=p_bounds,
                    p0=p0, **fitter_args)
                model_growth_c = profile(
                    r_dummy[cog_mask].to_value("arcsec"), *popt) << growth_c.unit
            except Exception as e:
                vprint("There was a problem during the fit:\n", e)
            else:
                cog.append(growth_c)
                cog_var.append(growth_c_var)
                profile_popt.append(popt)
                profile_var.append(pcov.diagonal())
                running_wavelength.append(wave_edges)

                res = growth_c[cog_mask] - model_growth_c
                mean_residuals[wave_slice] = np.nanmean(res)
                star_flux[wave_slice] = popt[0] << growth_c.unit

        star_good = np.isfinite(star_flux)
        star_flux = np.interp(
            data_container.wavelength, wavelength[star_good],
            star_flux[star_good])
        final_residuals = np.full(data_container.wavelength.size,
                                  fill_value=np.nan) << data.unit
        final_residuals[wave_mask] = mean_residuals

        if plot:
            fig = FluxCalibration.plot_extraction(
                x.value, y.value, x0.value, y0.value,
                data.value, r_dummy.value**0.5, cog,
                data_container.wavelength, star_flux, final_residuals)
        else:
            fig = None

        result = dict(wave_edges=np.array(running_wavelength) << u.AA,
                      optimal=np.array(profile_popt),
                      variance=np.array(profile_var),
                      stellar_flux=star_flux,
                      residuals=final_residuals,
                      figure=fig)
        return result

    @staticmethod
    @suppress_warnings
    def plot_extraction(x, y, x0, y0, data, rad, cog, wavelength, star_flux,
                        residuals):
        """"""
        vprint("Making stellar flux extraction plot")
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8),
                                constrained_layout=True)
        # Plot in the sky
        ax = axs[0, 0]
        ax.set_title("Median intensity")
        mappable = ax.scatter(x, y,
                              c=np.log10(np.nanmedian(data, axis=0)),
                              cmap='nipy_spectral')
        plt.colorbar(mappable, ax=ax, label=r'$\log_{10}$(intensity)')
        ax.plot(x0, y0, '+', ms=10, color='fuchsia')
        # Plot cog
        ax = axs[1, 0]
        ax.set_title("COG shape chromatic differences")
        cog = np.array(cog)
        cog /= cog[:, -1][:, np.newaxis]
        pct_cog = np.nanpercentile(cog, [16, 50, 84], axis=0)
        for p, pct in zip([16, 50, 84], pct_cog):
            ax.plot(rad, pct, label=f'Percentile {p}')
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        ax.set_ylabel('Normalised Curve of Growth')
        ax.set_xlabel('Distance to COM (arcsec)')
        ax.grid(visible=True, which='both')
        ax.legend()
        ax = axs[0, 1]
        ax.plot(wavelength, star_flux, '-', lw=0.7, label=f'Extracted stellar flux ')
        ax.plot(wavelength, residuals, '-', lw=0.7, label=f'Mean model residuals')
        ax.legend()
        ax.set_ylabel(f'Flux ({star_flux.unit})')
        ax.set_xlabel('Wavelength')

        axs[1, 1].axis("off")
        
        plt.close(fig)
        return fig

    @staticmethod
    def get_response_curve(obs_wave: u.Quantity,
                           obs_spectra: u.Quantity,
                           star: StandardStar, *,
                           cont_percentile: float = 80.0,
                           cont_window: u.Quantity = 50 << u.AA,
                           smooth_spline: bool = True,
                           abs_kappa_sigma: float = 3.0,
                           pct_filter_n: Optional[int] = None,
                           pct: Optional[float] = None,
                           pol_deg: Optional[int] = None,
                           spline: bool = False,
                           spline_lam: Optional[float] = None,
                           mask_absorption: bool = True,
                           mask_tellurics: bool = True,
                           mask_zeros: bool = True,
                           plot: bool = False
                           ):
        """
        Compute a smooth instrumental response from an observed star and its reference.

        Parameters
        ----------
        obs_wave : `astropy.units.Quantity`
            Observed wavelength grid.
        obs_spectra : `astropy.units.Quantity`
            Observed flux density on ``obs_wave``.
        star : StandardStar
            Reference standard star.
        cont_percentile : float, optional
            Percentile for the continuum upper envelope (typical 85 to 95).
        cont_window : `astropy.units.Quantity`, optional
            Continuum window in wavelength units.
        smooth_spline : bool, optional
            If True, refine the continuum with a weighted spline.
        abs_kappa_sigma: float, optional
            Number of sigmas below the continuum to identify absorption features.
            Default is 3.0.
        pct_filter_n : int or None, optional
            If given, apply a percentile filter of length ``pct_filter_n`` to the
            response before fitting.
        pct : float or None, optional
            Percentile used when ``pct_filter_n`` is set.
        pol_deg : int or None, optional
            Polynomial degree for response fitting. If provided, overrides spline/linear.
        spline : bool, optional
            If True, fit a smoothing spline to the response (requires ``spline_lam``).
        spline_lam : float or None, optional
            Smoothing parameter for ``make_smoothing_spline`` if ``spline`` is True.
        mask_absorption : bool, optional
            If True, downweight canonical stellar absorption features.
        mask_tellurics : bool, optional
            If True, downweight telluric bands.
        mask_zeros : bool, optional
            If True, downweight nonpositive observed flux pixels.
        plot : bool, optional
            If True, return a QA figure.

        Returns
        -------
        response_curve : callable
            A function ``f(x)`` that evaluates the response at wavelength ``x``.
            Units follow ``obs_spectra / ref_spectra`` (usually dimensionless).
        response_fig : `matplotlib.figure.Figure` or None
            QA figure if ``plot`` is True, else None.
        """
        vprint("Computing spectrophotometric response")

        # Reference on observed grid
        ref_on_obs = star.resample(obs_wave, conserve_flux=True)

        # Remove absorption features from reference spectra
        res = estimate_continuum_and_mask_absorption(
            obs_wave, ref_on_obs,
            cont_percentile=cont_percentile,
            cont_window=cont_window, smooth_spline=smooth_spline,
            abs_kappa_sigma=abs_kappa_sigma)

        ref_cont, ref_cont_err, ref_abs_regions = res

        # Remove absorption feature from observed spectra
        res = estimate_continuum_and_mask_absorption(obs_wave, obs_spectra,
        cont_percentile=cont_percentile,
        cont_window=cont_window, smooth_spline=smooth_spline,
        abs_kappa_sigma=abs_kappa_sigma)

        obs_cont, obs_cont_err, obs_abs_regions = res
        # Compute raw response
        cont_response = obs_cont / ref_cont
        cont_response_err = cont_response * np.sqrt(
            (obs_cont_err / obs_cont)**2 + (ref_cont_err / ref_cont)**2)

        # Build weights
        weights = np.ones(cont_response.size, dtype=float)
        if mask_absorption:
            vprint("Masking stellar absoprtion lines")
            ref_mask = ref_abs_regions > 0
            n_masked = np.count_nonzero(ref_mask)
            vprint(f"Fraction of reference pixels masked: {n_masked / ref_mask.size}")
            weights[ref_mask] = 0.0
            obs_mask = obs_abs_regions > 0
            n_masked = np.count_nonzero(obs_mask)
            vprint(f"Fraction of observed pixels masked: {n_masked / obs_mask.size}")
            weights[obs_mask] = 0.0
        if mask_tellurics:
            vprint("Masking telluric lines")
            weights *= np.asarray(mask_telluric_lines(obs_wave), dtype=float)
        if mask_zeros:
            vprint("Masking zeros")
            weights[obs_spectra.to_value(obs_spectra.unit) < 0.0] = 0.0

        weights = np.clip(weights, 0.0, None)
        weights_norm = weights.sum()
        if weights_norm == 0 or weights_norm < weights.size * 0.1:
            vprint("Too few unmasked pixels", level="warning")

        # Optional median filtering and reweighting
        if pct_filter_n is not None and pct_filter_n >= 2 and pct is not None:
            vprint(f"Applying percentile ({pct}) filter to response function")
            filtered_cont_response = percentile_filter(
                cont_response, pct, size=pct_filter_n,
                mode="mirror") << cont_response.unit
        else:
            filtered_cont_response = cont_response.copy()

        # Interpolation (skip pixels with w=0)
        if pol_deg is not None:
            vprint(f"Response smoothing using a polynomial (deg {pol_deg})")
            coeff = np.polyfit(obs_wave.to_value("AA")[weights > 0],
                               filtered_cont_response.value[weights > 0],
                               deg=pol_deg, w=weights[weights > 0])
            response = np.poly1d(coeff)
            fit_label = f"{pol_deg}-deg polynomial"
        elif spline:
            vprint(f"Response smoothing using a spline (lambda {spline_lam})")
            response = make_smoothing_spline(
                obs_wave.to_value("AA")[weights > 0],
                filtered_cont_response.value[weights > 0],
                w=weights[weights > 0],
                lam=spline_lam)
            fit_label = f"spline lam={spline_lam}"
        else:
            vprint(f"Response linearly interpolated")
            # Linear interpolation
            response = interp1d(obs_wave.to_value("AA")[weights > 0],
                                filtered_cont_response.value[weights > 0],
                                fill_value="extrapolate", bounds_error=False)
            fit_label = "linear interp (fallback)"

        response_err = interp1d(obs_wave.to_value("AA")[weights > 0],
                                cont_response_err.value[weights > 0],
                                fill_value="extrapolate", bounds_error=False)

        def response_wrapper(x):
            if isinstance(x, u.Quantity):
                return response(x.to_value("AA")) << filtered_cont_response.unit
            else:
                return response(x) << filtered_cont_response.unit

        def response_err_wrapper(x):
            if isinstance(x, u.Quantity):
                return response_err(x.to_value("AA")) << filtered_cont_response.unit
            else:
                return response_err(x) << filtered_cont_response.unit

        final_response = check_unit(response(obs_wave.to_value("AA")),
                                    filtered_cont_response.unit)
        final_response_err = cont_response_err

        if plot:
            fig = FluxCalibration.plot_response(
            wavelength=obs_wave,
            observed=[obs_spectra, obs_cont, obs_cont_err],
            reference=[ref_on_obs, ref_cont, ref_cont_err],
            cont_response=cont_response,
            filtered_response=filtered_cont_response,
            final_response=final_response,
            final_response_err=final_response_err,
            weights=weights,
            fit_label=fit_label,
            pct_filter_n=pct_filter_n,
            )
        else:
            fig = None

        return response_wrapper, response_err_wrapper, fig

    @suppress_warnings
    def plot_response(
        *,
        wavelength,
        observed,
        reference,
        cont_response,
        filtered_response,
        final_response,
        final_response_err,
        weights,
        fit_label,
        pct_filter_n: None,
    ):
        """
        Build a three-panel QA figure for the response fit.

        The top panel shows raw and smoothed responses. The middle panel shows
        the reference spectrum and calibrated observation with weight mask overlay.
        The bottom panel shows ratios of calibrated spectra to the reference.

        Parameters
        ----------
        wavelength : `astropy.units.Quantity`
            Observed wavelength grid.
        observed : list of `astropy.units.Quantity`
            ``[obs_spectra, obs_cont, obs_cont_err]``.
        reference : list of `astropy.units.Quantity`
            ``[ref_interp, ref_cont, ref_cont_err]``.
        cont_response : `astropy.units.Quantity`
            Response from continuum ratio (Obs_cont / Ref_cont).
        filtered_response : `astropy.units.Quantity`
            Response after optional percentile prefilter.
        final_response : `astropy.units.Quantity`
            Fitted response evaluated on ``wavelength``.
        final_response_err : `astropy.units.Quantity`
            1-sigma uncertainty on the fitted response.
        weights : ndarray
            Relative weights used in the fit, in [0, 1].
        fit_label : str
            Label describing the chosen model.
        pct_filter_n : int or None
            Window size for percentile prefilter, if applied.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The assembled QA figure.
        """
        import matplotlib.pyplot as plt

        obs_spectra, obs_cont, obs_cont_err = observed
        ref_spectra, ref_cont, ref_cont_err = reference
        raw_response = obs_spectra / ref_spectra

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 8),
                                constrained_layout=True, sharex=True,
                                height_ratios=[3, 3, 1])

        # Panel 1: responses
        ax = axs[0]
        ax.set_title("Instrumental response fit")
        ax.plot(wavelength, raw_response, lw=0.7, label="R(raw)")
        ax.plot(wavelength, cont_response, lw=0.7, label="R(continuum)")
        if pct_filter_n is not None and pct_filter_n >= 2:
            ax.plot(wavelength, filtered_response, lw=0.8, label=f"R(cont filtered {pct_filter_n})")
        ax.fill_between(wavelength,
                        final_response - 2 * final_response_err,
                        final_response + 2 * final_response_err,
                        color="cornflowerblue", label="R(final) 2-sigma error", alpha=0.5)
        ax.plot(wavelength, final_response, color="b", lw=1.2,
                label=f"R(Final) [Fit: {fit_label}]")
        ax.set_ylabel(f"R(lambda) [{cont_response.unit}]",
                      fontsize="small")
        ymin, ymax = np.nanpercentile(final_response.to_value(final_response.unit),
                                      [5, 95])
        if np.isfinite(ymin) and np.isfinite(ymax):
            ax.set_ylim(ymin * 0.8, ymax * 1.2)
        ax.legend(loc="best", fontsize="small")

        # Panel 2: calibrated spectra and weights
        ax = axs[1]
        ax.plot(wavelength, ref_spectra, lw=0.7, label="Ref",
                color="r")
        ax.fill_between(wavelength, ref_cont - ref_cont_err, ref_cont + ref_cont_err,
        color="orange", alpha=0.5)
        ax.plot(wavelength, ref_cont, color="orange", label="Ref. continuum")

        # Final response
        ax.fill_between(wavelength,
                        (obs_cont - obs_cont_err) / final_response,
                        (obs_cont + obs_cont_err) / final_response,
                        color="cornflowerblue", alpha=0.5)
        ax.plot(wavelength, obs_cont / final_response, color="cornflowerblue", lw=1,
                label="Obs cont cal")
        ax.plot(wavelength, obs_spectra / final_response, lw=1.0,
                color="blue", label="Fit-based calibration")
        ax.set_ylabel(f"Flux [{ref_spectra.unit}]")
        ax.legend(loc="best", fontsize="small")
        ax.set_ylim(ref_cont.min() / 2, ref_cont.max() * 1.5)
        # Plot mask
        tw = ax.twinx()
        tw.fill_between(wavelength, 0, 1 - weights, alpha=0.2, color="grey", label="Weight mask")
        tw.set_ylabel("1 - weights")

        # Residuals
        ax = axs[2]
        continuum_ratio = (obs_cont / final_response) / ref_cont
        ratio = (obs_spectra / final_response) / ref_spectra
        median_ratio = np.nanmedian(ratio)
        nmad_ratio = std_from_mad(ratio)
        ax.axhline(1.00, ls="--", color="r", alpha=0.6)
        ax.axhline(1.10, ls=":", color="r", alpha=0.4)
        ax.axhline(0.90, ls=":", color="r", alpha=0.4)
        ax.plot(wavelength, continuum_ratio.value, lw=0.8, color="grey",
        label="Cont. ratio")
        ax.plot(wavelength, ratio.value, lw=0.8, color="k",
        label="Spec. ratio")
        ax.annotate(f"Median +/- NMAD spec.: {median_ratio:.2f} +/- {nmad_ratio:.2f}",
                    xy=(0.01, 0.92),
                    xycoords="axes fraction", ha="left", va="top",
                    fontsize="small",
                    bbox=dict(boxstyle="round",
                    fc="lightblue", ec="steelblue", lw=2))
        ax.legend(loc="best", fontsize="small")
        ax.set_ylim(0.7, 1.3)
        ax.set_ylabel("Obs cal. / Ref")
        ax.set_xlabel("Wavelength")
        plt.close(fig)
        return fig

    def interpolate_response(self, wavelength, update=True):
        """
        Interpolates the spectral response to the input wavelength array.

        Parameters
        ----------
        wavelength : array-like
            The wavelength array to which the response will be interpolated.
        update : bool, optional
            If True, updates the internal response and wavelength attributes.

        Returns
        -------
        response : array-like
            The interpolated response.
        """
        self.vprint("Interpolating spectral response to input wavelength array")
        response = np.interp(wavelength, self.response_wavelength, self.response,
                             right=0., left=0.)
        response_err = np.interp(wavelength, self.response_wavelength,
                                 self.response_err,
                                 right=np.nan, left=np.nan)
        if update:
            self.vprint("Updating response and wavelength arrays")
            self.response = response
            self.response_err = response_err
            self.response_wavelength = wavelength
        return response, response_err

    def save_response(self, fname):
        """
        Saves the response function to a file.

        Parameters
        ----------
        fname : str
            File name for saving the response function.
        response : array-like
            The response function data.
        wavelength : array-like
            The wavelength array corresponding to the response function.
        units : str, optional
            Units of the response function.

        Returns
        -------
        None
        """
        self.vprint(f"Saving response function at: {fname}")
        # TODO: Use a QTable and dump to FITS instead.
        np.savetxt(fname, np.array([self.response_wavelength, self.response,
                                    self.response_err]).T,
                   header="Spectral Response curve\n"
                    + f" wavelength ({self.response_wavelength.unit}),"
                    + f" R ({self.response.unit}), Rerr ({self.response_err.unit})" 
                   )

    def apply(self, spectra_container : SpectraContainer) -> SpectraContainer:
        """
        Applies the response curve to a given SpectraContainer.

        Parameters
        ----------
        spectra_container : :class:`SpectraContainer`
            Target object to be corrected.

        Returns
        -------
        spectra_container_out : :class:`SpectraContainer`
            Corrected version of the input :class:`SpectraContainer`.
        """
        assert isinstance(spectra_container, SpectraContainer)
        sc_out = spectra_container.copy()
        if sc_out.is_corrected(self.name):
            self.vprint("Data already calibrated")
            return sc_out

        # Check that the model is sampled in the same wavelength grid
        if not sc_out.wavelength.size == self.response_wavelength.size or not np.allclose(
            sc_out.wavelength, self.response_wavelength, equal_nan=True):
            response, response_err = self.interpolate_response(
                sc_out.wavelength, update=False)
        else:
            response, response_err = self.response, self.response_err

        response = np.where(np.isfinite(response) & (response > 0),
                            response, np.nan)
        # Apply the correction
        good = np.isfinite(response) & (response > 0)
        sc_out.rss_intensity = (
            sc_out.rss_intensity / response[np.newaxis, :])
        # Propagate the uncertainty of the flux calibration
        sc_out.rss_variance = sc_out.rss_variance / response[np.newaxis, :]**2
        sc_out.rss_variance += (
            spectra_container.rss_intensity**2
            * response_err[np.newaxis, :]**2
            / response[np.newaxis, :]**4)

        self.record_correction(sc_out, status='applied',
                               units=str(self.response.unit))
        return sc_out

# Mr Krtxo \(ﾟ▽ﾟ)/
