"""Unified alert-event schema shared by all sources.

One row = one alert in one location. All timestamps are tz-aware UTC.
`finished_at` is NaT for alerts still ongoing at the source snapshot time.
"""

from __future__ import annotations

import pandas as pd

EVENT_COLUMNS: tuple[str, ...] = (
    "region",  # canonical oblast id, see data.regions
    "raion",  # raion name (source language) or <NA>
    "location",  # finest location name (oblast/raion/hromada/city)
    "geo_level",  # oblast | raion | hromada | city
    "alert_type",  # air_raid | artillery_shelling | urban_fights | ...
    "alert_level",  # yellow | red | <NA> (pre-2026-09-06 / unknown)
    "started_at",
    "finished_at",
    "source",
    "source_id",
)

GEO_LEVELS = frozenset({"oblast", "raion", "hromada", "city"})


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    """Check schema invariants; return df with columns in canonical order."""
    missing = set(EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing event columns: {sorted(missing)}")
    for col in ("started_at", "finished_at"):
        if str(df[col].dt.tz) != "UTC":
            raise ValueError(f"{col} must be tz-aware UTC, got {df[col].dt.tz}")
    if df["started_at"].isna().any():
        raise ValueError("started_at must not be null")
    if df["region"].isna().any():
        raise ValueError("region must not be null (unmapped region names?)")
    bad_levels = set(df["geo_level"].unique()) - GEO_LEVELS
    if bad_levels:
        raise ValueError(f"unknown geo levels: {bad_levels}")
    if (df["finished_at"] < df["started_at"]).any():
        raise ValueError("finished_at < started_at")
    return df.loc[:, list(EVENT_COLUMNS)]
