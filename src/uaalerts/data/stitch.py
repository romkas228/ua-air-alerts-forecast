"""Stitching the historical (Vadimkin) and fresh (alerts.in.ua mirror) sources.

Findings (2026-10-10, see CLAUDE.md):
- Vadimkin official ends 2026-09-07 ~05:30 UTC (its upstream Telegram channel went silent)
  and is already thinning out after 2026-09-07 00:00 UTC.
- The alerts.in.ua mirror is complete only from ~2026-09-06 12:00 UTC.
- In the clean overlap [09-06 12:00, 09-07 00:00) UTC hourly `y_any` agrees on ~99% of
  region-hours (excluding Crimea/Luhansk, which Vadimkin lacks and we exclude anyway).
"""

from __future__ import annotations

import pandas as pd

from uaalerts.data.panel import build_panel
from uaalerts.data.schema import validate_events

STITCH_AT = pd.Timestamp("2026-09-06 12:00", tz="UTC")
OVERLAP = (pd.Timestamp("2026-09-06 12:00", tz="UTC"), pd.Timestamp("2026-09-07 00:00", tz="UTC"))


def stitch(
    historical: pd.DataFrame, fresh: pd.DataFrame, at: pd.Timestamp = STITCH_AT
) -> pd.DataFrame:
    """Historical events that started before `at` + fresh events that started at/after it.

    Fresh events that started before `at` but are still ongoing (finished_at NaT) are kept
    too: these are the multi-year alerts in occupied regions the historical source lacks.
    """
    hist = historical[historical["started_at"] < at]
    fresh_after = fresh[fresh["started_at"] >= at]
    fresh_ongoing = fresh[(fresh["started_at"] < at) & fresh["finished_at"].isna()]
    out = pd.concat([hist, fresh_after, fresh_ongoing], ignore_index=True)
    out = out.sort_values(["started_at", "region"], kind="stable").reset_index(drop=True)
    return validate_events(out)


def overlap_agreement(
    historical: pd.DataFrame,
    fresh: pd.DataFrame,
    regime_break: pd.Timestamp,
    window: tuple[pd.Timestamp, pd.Timestamp] = OVERLAP,
    exclude: tuple[str, ...] = ("crimea", "luhanska"),
) -> pd.DataFrame:
    """Per-region hourly `y_any` agreement of both sources inside the overlap window."""
    a = build_panel(historical, *window, regime_break=regime_break)
    b = build_panel(fresh, *window, regime_break=regime_break)
    m = a.merge(b, on=["region", "hour"], suffixes=("_hist", "_fresh"))
    m = m[~m["region"].isin(exclude)]
    m["agree"] = m["y_any_hist"] == m["y_any_fresh"]
    return (
        m.groupby("region", observed=True)
        .agg(
            hours=("agree", "size"),
            agree=("agree", "mean"),
            hist_alert_hours=("y_any_hist", "sum"),
            fresh_alert_hours=("y_any_fresh", "sum"),
        )
        .reset_index()
    )
