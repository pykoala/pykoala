"""
Module containing physical relations
"""
# pykoala/utils/physics.py

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from astropy import units as u

from pykoala.ancillary import check_unit


__all__ = ["airmass_kasten_young89", "pressure_from_altitude_std_atm"]


def pressure_from_altitude_std_atm(
    altitude: Union[float, u.Quantity],
    p0: Union[float, u.Quantity] = 1013.25 * u.hPa,
) -> u.Quantity:
    """
    Estimate pressure from geometric altitude using the U.S. Standard Atmosphere
    tropospheric approximation (valid to ~11 km).

    Parameters
    ----------
    altitude
        Observer altitude above mean sea level (meters if unitless).
    p0
        Sea-level standard pressure.

    Returns
    -------
    astropy.units.Quantity
        Estimated pressure in hPa.

    Notes
    -----
    Formula:
        P = P0 * (1 - 2.25577e-5 * h) ** 5.25588
    """
    h = check_unit(altitude, u.m)
    p0 = check_unit(p0, u.hPa)

    # Clip negative altitudes slightly above -100 m to avoid invalid exponent
    h_val = np.clip(h.value, -100.0, 11_000.0)
    pressure = p0.value * (1.0 - 2.25577e-5 * h_val) ** 5.25588
    return pressure << u.hPa


def airmass_plane_parallel(*,
                 zd: Optional[Union[float, u.Quantity]] = None,
                 alt: Optional[Union[float, u.Quantity]] = None):
    """
    Compute the optical airmass assuming a plane-parallel atmosphere.
    
    Parameters
    ----------
    zd
        Apparent zenith distance. Degrees if unitless.
    alt
        Apparent altitude above horizon. Degrees if unitless. If provided,
        ``zd`` is ignored and computed as ``90° - alt``.
    """
    zd = check_unit(zd, u.deg)
    alt = check_unit(alt, u.deg)
    if alt is not None:
        zd = (90.0 << u.deg) - alt

    x_secz = 1 / np.cos(np.radians(zd))
    return x_secz

def airmass_kasten_young89(
    *,
    zd: Optional[Union[float, u.Quantity]] = None,
    alt: Optional[Union[float, u.Quantity]] = None,
    pressure: Optional[Union[float, u.Quantity]] = None,
    altitude: Optional[Union[float, u.Quantity]] = None,
    apply_pressure_scaling: bool = True,
    clip_horizon: bool = True,
) -> float:
    """
    Compute optical airmass via Kasten & Young (1989), optionally pressure-scaled.

    Parameters
    ----------
    zd
        Apparent zenith distance. Degrees if unitless.
    alt
        Apparent altitude above horizon. Degrees if unitless. If provided,
        ``zd`` is ignored and computed as ``90° - alt``.
    pressure
        Site pressure. hPa if unitless. If omitted and ``altitude`` is given,
        pressure is estimated with :func:`pressure_from_altitude_std_atm`.
    altitude
        Site altitude above mean sea level (meters if unitless), used only to
        estimate pressure when ``pressure`` is not provided.
    apply_pressure_scaling
        If True, return X * (P / 1013.25 hPa). If False, return geometric X.
    clip_horizon
        If True, clip zenith distance to <= 89.99° to avoid singularity.

    Returns
    -------
    float
        Airmass (dimensionless).

    Notes
    -----
    Kasten & Young (1989), Applied Optics 28(22), 4735-4738:
        X = 1 / (cos z + 0.50572 * (96.07995 - z)^(-1.6364)),  z in degrees.

    This expression is empirical for **apparent** z (refraction implicitly
    included) and remains well-behaved very close to the horizon. For
    extinction work, pressure scaling makes X proportional to Rayleigh optical
    depth.

    Examples
    --------
    >>> airmass_kasten_young89(alt=45)  # ~1.41
    1.41...
    >>> airmass_kasten_young89(zd=60 * u.deg, pressure=780 * u.hPa)
    1.33...
    >>> airmass_kasten_young89(alt=10, altitude=1165)  # estimate P from altitude
    5.6...
    """
    # Resolve zd (deg)
    
    if alt is not None:
        alt = check_unit(alt, u.deg)
        zd = (90.0 << u.deg) - alt
    elif zd is not None:
        zd = check_unit(zd, u.deg)
    else:
        raise ValueError("Provide either 'zd' (zenith distance) or 'alt' (altitude).")

    zdeg = zd.to_value("deg")

    if not np.isfinite(zdeg):
        raise ValueError(f"Invalid zenith distance: {zdeg!r}")

    if clip_horizon:
        zdeg = np.clip(zdeg, 0.0, 89.99)

    # Kasten & Young 1989 geometric airmass
    cz = np.cos(np.deg2rad(zdeg))
    X_geo = 1.0 / (cz + 0.50572 * (96.07995 - zdeg) ** (-1.6364))

    if not apply_pressure_scaling:
        return X_geo

    # Determine pressure (hPa) to scale optical depth
    if pressure is not None:
        pressure = check_unit(pressure, u.hPa)
    elif altitude is not None:
        pressure = pressure_from_altitude_std_atm(altitude)
    else:
        pressure = 1013.25 * u.hPa  # default sea-level standard pressure

    p_hpa = float(pressure.to_value(u.hPa))
    # Sanity guard (avoid pathologies)
    if not (100.0 <= p_hpa <= 1050.0):
        p_hpa = 1013.25

    return X_geo * (p_hpa / 1013.25)
