"""Solar elevation (NOAA approximation) - deterministic, known in advance for any hour."""

from __future__ import annotations

import numpy as np
import pandas as pd


def solar_elevation(when: pd.DatetimeIndex, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Sun elevation in degrees at UTC instants `when` for points (lat, lon)."""
    utc = when.tz_convert("UTC")
    doy = utc.dayofyear.to_numpy()
    hours = (utc.hour + utc.minute / 60 + utc.second / 3600).to_numpy()
    gamma = 2 * np.pi / 365 * (doy - 1 + (hours - 12) / 24)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    true_solar_min = hours * 60 + eqtime + 4 * lon
    hour_angle = np.deg2rad(true_solar_min / 4 - 180)
    phi = np.deg2rad(lat)
    cos_zen = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    elev: np.ndarray = 90 - np.rad2deg(np.arccos(np.clip(cos_zen, -1, 1)))
    return elev
