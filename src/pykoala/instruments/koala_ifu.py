"""
Wrappers to build a PyKoala RSS object from KOALA (2dfdr-reduced) data.

Notes
-----
- Assumes a 2dfdr KOALA RSS FITS with:
  * EXT 0: intensity (nfibres x nwave)
  * EXT 1: variance (optional; same shape as intensity)
  * EXT 2: fibre table with at least columns: SELECTED, XPOS, YPOS, SPEC_ID
- Produces a :class:`pykoala.data_container.RSS` with WCS-based wavelength.
"""

# =============================================================================
# Basics packages
# =============================================================================
from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple
import numpy as np
import os
# =============================================================================
# Astropy and associated packages
# =============================================================================
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
# =============================================================================
# KOALA packages
# =============================================================================
from pykoala import vprint
from pykoala.data_container import RSS
from pykoala.utils.io import suppress_warnings
from pykoala.utils.physics import airmass_kasten_young89

__all__ = [
    "koala_rss",
    "koala_cube",
]

SSO_ALTITUDE = 1165 << u.m


def _airmass_from_header(header: fits.Header) -> float:
    """
    Compute the airmass from the KOALA header.

    Parameters
    ----------
    header
        FITS header expected to contain either ``AIRMASS`` or ZD keywords
        ``ZDSTART`` and ``ZDEND`` (zenith distances in degrees).

    Returns
    -------
    float
        Scalar airmass value.

    Raises
    ------
    KeyError
        If neither ``AIRMASS`` nor both ``ZDSTART`` and ``ZDEND`` are present.
    """
    if "AIRMASS" in header:
        try:
            x = float(header["AIRMASS"])
            if np.isfinite(x) and 0.5 <= x <= 20:
                return x
        except Exception:
            pass

    # Get ZD, airmass
    if "ZDSTART" in header and "ZDEND" in header:
        zd = 0.5 * (float(header["ZDSTART"]) + float(header["ZDEND"]))

    altitude_m = header.get("ALT_OBS", SSO_ALTITUDE)
    return airmass_kasten_young89(
        zd=zd, pressure=None, altitude=altitude_m, apply_pressure_scaling=True
    )

def _koala_header(base: fits.Header) -> fits.Header:
    """
    Build a minimal header from a 2dfdr KOALA header.

    This selects & normalizes a subset of cards and fixes common WCS quirks.

    Parameters
    ----------
    base
        FITS header (typically EXT 0 + EXT 2 merged).

    Returns
    -------
    fits.Header
        Sanitized header suitable for :class:`astropy.wcs.WCS`.

    Notes
    -----
    - Renames ``CENRA->RACEN`` and ``CENDEC->DECCEN`` if present.
    - Keeps a curated set of cards when available; gracefully skips missing ones.
    """
    hdr = base.copy()

    # Normalize common KOALA center keywords to what downstream code expects.
    if "CENRA" in hdr and "RACEN" not in hdr:
        hdr.rename_keyword("CENRA", "RACEN")
    if "CENDEC" in hdr and "DECCEN" not in hdr:
        hdr.rename_keyword("CENDEC", "DECCEN")

    # RADESYS is the canonical keyword (some files use RADECSYS).
    if "RADECSYS" in hdr and "RADESYSa" not in hdr and "RADESYS" not in hdr:
        # Using the 'a' suffix variant is tolerated by astropy; leave original otherwise.
        hdr["RADESYSa"] = hdr["RADECSYS"]
        del hdr["RADECSYS"]

    # Select a (safe) subset if present; skip silently if missing.
    keep = [
        "BITPIX", "ORIGIN", "TELESCOP", "ALT_OBS", "LAT_OBS", "LONG_OBS",
        "INSTRUME", "GRATID", "SPECTID", "DICHROIC", "OBJECT", "EXPOSED",
        "ZDSTART", "ZDEND", "NAXIS", "NAXIS1", "NAXIS2", "RACEN", "DECCEN",
        "TEL_PA", "CTYPE1", "CUNIT1", "CRVAL1", "CDELT1", "CRPIX1",
        "CTYPE2", "CUNIT2", "CRVAL2", "CDELT2", "CRPIX2",
    ]
    filtered = fits.Header()
    for k in keep:
        if k in hdr:
            filtered[k] = hdr[k]
    # filtered.extend(hdr, update=True)
    return filtered

def _koala_fibre_table(fibre_table: np.ndarray) -> fits.BinTableHDU:
    """
    Convert a 2dfdr fibre table into a PyKoala spaxel table.

    Parameters
    ----------
    fibre_table
        Numpy record array with at least columns:
        ``SELECTED``, ``XPOS``, ``YPOS``, ``SPEC_ID``.

    Returns
    -------
    fits.BinTableHDU
        Spaxel table with columns:
        ``Fibre, Status, Ones, Wavelengths, Zeros, Delta_RA, Delta_Dec, Fibre_OLD``.
    """
    # Keep only *selected* fibres.
    spax = fibre_table[fibre_table["SELECTED"] == 1]

    n = len(spax)
    col_fibre = fits.Column(name="Fibre",        format="I", array=np.arange(n, dtype=np.int16) + 1)
    col_status = fits.Column(name="Status",      format="I", array=np.ones(n, dtype=np.int16))
    col_ones = fits.Column(name="Ones",          format="I", array=np.ones(n, dtype=np.int16))
    col_nwave = fits.Column(name="Wavelengths",  format="I", array=np.full(n, 2048, dtype=np.int16))
    col_zeros = fits.Column(name="Zeros",        format="I", array=np.zeros(n, dtype=np.int16))
    col_dra = fits.Column(name="Delta_RA",       format="D", array=np.asarray(spax["XPOS"], dtype=float))
    col_ddec = fits.Column(name="Delta_Dec",     format="D", array=np.asarray(spax["YPOS"], dtype=float))
    col_old = fits.Column(name="Fibre_OLD",      format="I", array=np.asarray(spax["SPEC_ID"], dtype=np.int16))

    return fits.BinTableHDU.from_columns(
        [col_fibre, col_status, col_ones, col_nwave, col_zeros, col_dra, col_ddec, col_old]
    )

def _ensure_wcs_for_koala(header: fits.Header) -> WCS:
    """
    Construct a spectral WCS from a KOALA header, fixing common issues.

    Parameters
    ----------
    header
        FITS header.

    Returns
    -------
    astropy.wcs.WCS
        WCS with spectral axis as axis 1 (CTYPE1='WAVE').

    Notes
    -----
    - Some KOALA headers need the first axis explicitly set to 'WAVE'.
    """
    w = WCS(header)
    # Ensure spectral axis is declared as wavelength.
    if len(w.wcs.ctype) > 0 and not str(w.wcs.ctype[0]).strip():
        w.wcs.ctype[0] = "WAVE"
    elif len(w.wcs.ctype) > 0 and str(w.wcs.ctype[0]).strip().upper() not in {"WAVE", "AWAV", "FREQ"}:
        # Force to WAVE for downstream spectral handling.
        w.wcs.ctype[0] = "WAVE"
    return w


def _safe_delete_rows(arr: np.ndarray, bad_rows: Optional[Sequence[int]]) -> np.ndarray:
    """Delete rows by index if provided; otherwise return `arr` unchanged."""
    if not bad_rows:
        return arr
    return np.delete(arr, bad_rows, axis=0)


@suppress_warnings()
def _read_rss(
    file_path: str,
    wcs: WCS,
    intensity_axis: int = 0,
    variance_axis: Optional[int] = 1,
    bad_fibres_list: Optional[Sequence[int]] = None,
    header: Optional[fits.Header] = None,
    info: Optional[dict] = None,
) -> RSS:
    """
    Read a KOALA RSS FITS into a :class:`pykoala.data_container.RSS`.

    Parameters
    ----------
    file_path
        Path to the 2dfdr KOALA RSS FITS.
    wcs
        Spectral WCS (first axis must be spectral).
    intensity_axis
        HDU index for intensity data (default: 0).
    variance_axis
        HDU index for variance data, or ``None`` if not present (default: 1).
    bad_fibres_list
        Row indices (0-based) to drop from the intensity/variance arrays.
    header
        Header for the resulting RSS; if ``None`` a minimal header is used.
    info
        Metadata dictionary to attach to the RSS.

    Returns
    -------
    RSS
        A PyKoala RSS object with intensity, variance, wavelength, WCS and metadata.
    """
    if header is None:
        header = fits.Header()

    fname = os.path.basename(file_path)
    vprint(f"\n> Reading KOALA RSS file {fname}")

    with fits.open(file_path) as hdul:
        all_int = np.asarray(hdul[intensity_axis].data, dtype=np.float32)
        intensity = _safe_delete_rows(all_int, bad_fibres_list)

        if variance_axis is not None and variance_axis < len(hdul):
            all_var = np.asarray(hdul[variance_axis].data, dtype=np.float32)
            variance = _safe_delete_rows(all_var, bad_fibres_list)
        else:
            vprint("WARNING! Variance extension not found in FITS file; filling with NaN.")
            variance = np.full_like(intensity, fill_value=np.nan, dtype=np.float32)

    # Build wavelength from WCS; size must match data's spectral length.
    nfibres, nwave = intensity.shape
    wl_idx = np.arange(nwave)
    wavelength = wcs.spectral.array_index_to_world(wl_idx).to_value("angstrom")

    rss = RSS(
        intensity=intensity << u.adu,          # keep legacy unit for compatibility
        variance=variance << (u.adu**2),
        wavelength=wavelength << u.angstrom,
        info=info,
        header=header,
        fibre_diameter=1.25 << u.arcsec,
        wcs=wcs,
    )
    rss.history("read", f"- RSS read from {fname}")
    return rss

@suppress_warnings()
def koala_rss(path_to_file: str) -> RSS:
    """
    Convert a 2dfdr KOALA RSS FITS into a PyKoala :class:`RSS`.

    Parameters
    ----------
    path_to_file
        Path to the 2dfdr KOALA RSS FITS.

    Returns
    -------
    RSS
        PyKoala RSS object.

    Notes
    -----
    - Bad fibres are inferred from the fibre table (EXT 2) where ``SELECTED == 0``.
    - The fibre RA/Dec are built from header centre plus per-fibre offsets (arcsec).
    """
    # Merge EXT 0 & EXT 2 headers (mirrors your original approach).
    header0 = fits.getheader(path_to_file, 0)
    header2 = fits.getheader(path_to_file, 2)
    merged = header0.copy()
    merged.extend(header2, update=True)

    header = _koala_header(merged)

    # WCS (ensure spectral axis declared)
    koala_wcs = _ensure_wcs_for_koala(header)

    # Original 2dfdr fibre table (EXT 2) and PyKoala spaxel table
    fibre_table = fits.getdata(path_to_file, 2)
    koala_spax_table = _koala_fibre_table(fibre_table)

    # Bad fibre list (0-based indices): those not SELECTED in the original table.
    bad_fibres_list = (fibre_table["SPEC_ID"][fibre_table["SELECTED"] == 0] - 1).tolist()

    # Info block
    info = {}
    info["name"] = header.get("OBJECT", "UNKNOWN")
    if "EXPOSED" in header:
        info["exptime"] = float(header["EXPOSED"]) * u.s

    # KOALA stores centre in degrees in most products; avoid blindly rad2deg.
    # If values look like radians (|value| < 2π), convert; otherwise assume deg.
    racen = float(header.get("RACEN", 0.0))
    deccen = float(header.get("DECCEN", 0.0))
    if abs(racen) <= 2 * np.pi and abs(deccen) <= 2 * np.pi:
        racen = np.degrees(racen)
        deccen = np.degrees(deccen)

    # Per-fibre offsets are in arcsec in the constructed spaxel table.
    info["fib_ra"] = (racen + koala_spax_table.data["Delta_RA"] / 3600.0) * u.deg
    info["fib_dec"] = (deccen + koala_spax_table.data["Delta_Dec"] / 3600.0) * u.deg

    # Airmass
    try:
        info["airmass"] = _airmass_from_header(header)
    except KeyError:
        vprint("WARNING! Could not determine airmass from header.")
        info["airmass"] = np.nan

    # Build the RSS
    rss = _read_rss(
        path_to_file,
        wcs=koala_wcs,
        bad_fibres_list=bad_fibres_list,
        intensity_axis=0,
        variance_axis=1,
        header=header,
        info=info,
    )
    return rss


def koala_cube():
    """Placeholder for a cube reader built on KOALA/2dfdr products."""
    raise NotImplementedError("koala_cube() is not implemented yet.")


# @suppress_warnings()
# def _read_rss(file_path,
#              wcs,
#              intensity_axis=0,
#              variance_axis=1,
#              bad_fibres_list=None,
#              header=None,
#              info=None
#              ):
#     """TODO."""
#     if header is None:
#         # Blank Astropy Header object for the RSS header
#         # Example how to add header value at the end
#         # blank_header.append(('DARKCORR', 'OMIT', 'Dark Image Subtraction'), end=True)
#         header = fits.header.Header(cards=[], copy=False)

#     file_name = os.path.basename(file_path)

#     vprint(f"\n> Reading KOALA RSS file {file_name}")
#     #  Open fits file. This assumes that RSS objects are written to file as .fits.
#     with fits.open(file_path) as rss_fits:
#         # Read intensity using rss_fits_file[0]
#         all_intensities = np.array(rss_fits[intensity_axis].data,
#                                    dtype=np.float32)
#         intensity = np.delete(all_intensities, bad_fibres_list, 0)
#         # Read errors if exist a dedicated axis
#         if variance_axis is not None:
#             all_variances = rss_fits[variance_axis].data
#             variance = np.delete(all_variances, bad_fibres_list, 0)

#         else:
#             vprint("WARNING! Variance extension not found in fits file!")
#             variance = np.full_like(intensity, fill_value=np.nan)

#     # Create wavelength from wcs
#     nrow, ncol = wcs.array_shape
#     wavelength_index = np.arange(ncol)
#     #TODO : remove to_value once units are homogeneized
#     wavelength = wcs.spectral.array_index_to_world(wavelength_index).to_value("angstrom")
#     # First Header value added by the PyKoala routine
#     rss = RSS(intensity=intensity << u.adu,
#               variance=variance << u.adu**2,
#               wavelength=wavelength << u.angstrom,
#               info=info,
#               header=header,
#               fibre_diameter=1.25 << u.arcsec,
#               wcs=wcs)

#     rss.history('read', ' '.join(['- RSS read from ', file_name]))
#     return rss

# @suppress_warnings()
# def koala_rss(path_to_file):
#     """
#     A wrapper function that converts a file (not an RSS object) to a koala RSS object
#     The paramaters used to build the RSS object e.g. bad spaxels, header etc all come from the original (non PyKoala) .fits file
#     """
    
#     header = fits.getheader(path_to_file, 0) + fits.getheader(path_to_file, 2)
#     header = _koala_header(header)
#     # WCS
#     if "RADECSYS" in header:
#         header["RADECSYSa"] = header["RADECSYS"]
#         del header["RADECSYS"]
#     koala_wcs = WCS(header)
#     # Fix the WCS information such that koala_wcs.spectra exists
#     koala_wcs.wcs.ctype[0] = 'WAVE    '
#     # Constructing Pykoala Spaxels table from 2dfdr spaxels table (data[2])
#     fibre_table = fits.getdata(path_to_file, 2)
#     koala_fibre_table = _koala_fibre_table(fibre_table)

#     # List of bad spaxels from 2dfdr spaxels table
#     bad_fibres_list = (fibre_table['SPEC_ID'][fibre_table['SELECTED'] == 0] - 1).tolist()
#     # -1 to start in 0 rather than in 1
#     # Create the dictionary containing relevant information
#     info = {}
#     info['name'] = header['OBJECT']
#     info['exptime'] = header['EXPOSED'] << u.second
#     info['fib_ra'] = (np.rad2deg(header['RACEN'])
#                       + koala_fibre_table.data['Delta_RA'] / 3600) << u.deg
#     info['fib_dec'] = (np.rad2deg(header['DECCEN'])
#                        + koala_fibre_table.data['Delta_DEC'] / 3600) << u.deg
#     info['airmass'] = _airmass_from_header(header)
#     # Read RSS file into a PyKoala RSS object
#     rss = _read_rss(path_to_file, wcs=koala_wcs,
#                    bad_fibres_list=bad_fibres_list,
#                    intensity_axis=0,
#                    variance_axis=1,
#                    header=header,
#                    info=info,
#                    )
#     return rss

# def koala_cube():
#     raise NotImplementedError()

# Mr Krtxo \(ﾟ▽ﾟ)/ + Ángel :-)
