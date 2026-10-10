"""Hourly region panel: targets and alert-minute aggregates per (region, hour).

Hours are UTC and labelled by their start: row `hour=h` covers `[h, h+1h)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uaalerts.data.regions import REGION_IDS, covers_center

HOUR = pd.Timedelta(hours=1)
PANEL_COLUMNS = [
    "region",
    "hour",
    "y_alert",
    "alert_minutes",
    "y_red",
    "red_minutes",
    "y_oblast_any",
    "oblast_minutes",
]


def merge_intervals(starts: pd.Series, ends: pd.Series) -> pd.DataFrame:
    """Union of possibly overlapping [start, end) intervals (single group)."""
    order = np.argsort(starts.to_numpy(), kind="stable")
    s = starts.to_numpy()[order]
    e = ends.to_numpy()[order]
    if len(s) == 0:
        return pd.DataFrame({"start": s, "end": e})
    running_end = np.maximum.accumulate(e)
    # A new block starts where the start is beyond everything seen so far.
    new_block = np.r_[True, s[1:] > running_end[:-1]]
    block = np.cumsum(new_block) - 1
    out = pd.DataFrame({"start": s, "end": running_end, "block": block})
    return (
        out.groupby("block").agg(start=("start", "min"), end=("end", "max")).reset_index(drop=True)
    )


def _minutes_per_hour(intervals: pd.DataFrame, hours: pd.DatetimeIndex) -> np.ndarray:
    """Minutes covered by disjoint `intervals` within each hour of `hours`."""
    minutes = np.zeros(len(hours))
    if intervals.empty:
        return minutes
    first = hours[0]
    n = len(hours)
    for start, end in zip(intervals["start"], intervals["end"], strict=True):
        lo = max(start, first)
        hi = min(end, first + n * HOUR)
        if hi <= lo:
            continue
        i0 = int((lo - first) // HOUR)
        i1 = int((hi - first - pd.Timedelta(1, "ns")) // HOUR)
        for i in range(i0, i1 + 1):
            h0 = first + i * HOUR
            overlap = min(hi, h0 + HOUR) - max(lo, h0)
            minutes[i] += overlap.total_seconds() / 60
    return minutes


def build_panel(
    events: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    regime_break: pd.Timestamp,
    regions: tuple[str, ...] = REGION_IDS,
) -> pd.DataFrame:
    """Hourly panel over [start, end) for each region.

    Targets (centre = oblast administrative centre, see `regions.covers_center`):
      y_alert / alert_minutes  air-raid alert covering the centre, any level
      y_red / red_minutes      red-level alert covering the centre; <NA> before
                               `regime_break` (levels did not exist)
    Context (not targets): y_oblast_any / oblast_minutes  any air-raid alert anywhere in
    the oblast.
    Ongoing alerts (finished_at NaT) are treated as active until `end`, which must be the
    time the source was last observed — never later.
    """
    start, end = start.floor("h"), end.floor("h")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    air = events[events["alert_type"] == "air_raid"].copy()
    fin = air["finished_at"].fillna(end)
    air["finished_at"] = fin.where(fin <= end, end)
    air = air[air["finished_at"] > air["started_at"]]
    air["center"] = [
        covers_center(r, g, loc)
        for r, g, loc in zip(air["region"], air["geo_level"], air["location"], strict=True)
    ]

    def minutes(df: pd.DataFrame) -> np.ndarray:
        return _minutes_per_hour(merge_intervals(df["started_at"], df["finished_at"]), hours)

    frames = []
    for region in regions:
        g = air[air["region"] == region]
        c = g[g["center"]]
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "hour": hours,
                    "alert_minutes": minutes(c),
                    "red_minutes": minutes(c[c["alert_level"] == "red"]),
                    "oblast_minutes": minutes(g),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    panel["y_alert"] = (panel["alert_minutes"] > 0).astype("int8")
    panel["y_oblast_any"] = (panel["oblast_minutes"] > 0).astype("int8")
    pre = panel["hour"] < regime_break
    panel["y_red"] = (panel["red_minutes"] > 0).astype("Int8")
    panel.loc[pre, "y_red"] = pd.NA
    panel.loc[pre, "red_minutes"] = np.nan
    panel["region"] = panel["region"].astype("category")
    return panel[PANEL_COLUMNS]
