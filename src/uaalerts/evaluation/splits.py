"""Weekly walk-forward splits with row-level purging.

A row (origin o, lead h) describes the target hour [o+h-1, o+h); its label is known at
time T iff o + h <= T. A model used from T may train only on such rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

WEEK = pd.Timedelta(weeks=1)
HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class Fold:
    name: str
    start: pd.Timestamp  # T: model is fit with what is known at T
    end: pd.Timestamp  # origins in [start, end) are forecast with that model


def weekly_folds(
    data_end: pd.Timestamp, n_weeks: int, skip_weeks: int = 0, prefix: str = "w"
) -> list[Fold]:
    """`n_weeks` consecutive weekly folds ending `skip_weeks` weeks before `data_end`."""
    last_end = data_end - skip_weeks * WEEK
    folds = []
    for k in range(n_weeks):
        start = last_end - (n_weeks - k) * WEEK
        folds.append(Fold(f"{prefix}{k + 1}", start, start + WEEK))
    return folds


def known_at(df: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """Rows whose label is observed by time t (target hour fully elapsed)."""
    return (df["target_hour"] + HOUR <= t) & df["label"].notna()


def in_fold(df: pd.DataFrame, fold: Fold) -> pd.Series:
    """Evaluation rows of a fold: origins inside it with an observed label."""
    return (df["origin"] >= fold.start) & (df["origin"] < fold.end) & df["label"].notna()


def excluded_regions(
    panel: pd.DataFrame, target: str, before: pd.Timestamp, days: int, threshold: float
) -> list[str]:
    """Regions whose target rate in the `days` before `before` exceeds `threshold`."""
    win = panel[(panel["hour"] >= before - pd.Timedelta(days=days)) & (panel["hour"] < before)]
    rate = win.groupby("region", observed=True)[target].mean()
    return sorted(str(r) for r in rate[rate.astype(float) > threshold].index)
