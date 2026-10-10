"""The leakage gateway: what the world looked like at a forecast origin."""

from __future__ import annotations

import pandas as pd


def as_of(events: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Events as they were known at time `t`.

    Keeps alerts that started before `t`; alerts still running at `t` lose their (future)
    end time and become ongoing (NaT), exactly like in a live feed.
    """
    known = events[events["started_at"] < t].copy()
    known["finished_at"] = known["finished_at"].where(known["finished_at"] < t)
    return known
