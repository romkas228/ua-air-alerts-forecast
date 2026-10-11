"""Origin features and (origin, region, lead) dataset assembly.

Leakage contract: for an origin `t`, every feature is a function of panel hours `< t` only
(index `i = idx(t - 1h)` and earlier). Panel hours before `t` are fully observable at `t`
(an alert's activity in the past is public in real time), so this is equivalent to
building the panel from `as_of(events, t)`. `tests/test_leakage.py` checks exactly that.

Implementation: the panel is pivoted to dense (hour x region) matrices and all window
features are computed with cumulative sums, then sampled at the origins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from uaalerts.data.regions import CENTER_COORDS, NEIGHBOURS, OCCUPIED
from uaalerts.features.solar import solar_elevation

HOUR = pd.Timedelta(hours=1)
WINDOWS = (6, 24, 72, 168, 672)  # hours
PREFIXES = ("alert", "red", "oblast")
TARGETS = {"y_alert": "alert", "y_red": "red"}
LAST_YEAR_HOURS = 365 * 24
ANALOG_HALF_HOURS = 15 * 24
NEVER = 10_000.0  # "hours since" when the event never happened in the data
MIN_HISTORY_HOURS = 8 * 168  # longest look-back (8-week hour-of-week climatology)


@dataclass(frozen=True)
class Wide:
    """Dense hour x region view of the panel."""

    hours: pd.DatetimeIndex
    regions: list[str]
    m: dict[str, np.ndarray]  # column -> (T, R) float array, NaN where missing

    def idx(self, ts: pd.DatetimeIndex) -> np.ndarray:
        pos = self.hours.get_indexer(ts)
        if (pos < 0).any():
            raise KeyError("timestamps outside the panel")
        return pos


def to_wide(panel: pd.DataFrame) -> Wide:
    cols = [
        "y_alert",
        "y_red",
        "y_oblast_any",
        "alert_minutes",
        "red_minutes",
        "oblast_minutes",
        "alert_active_end",
        "red_active_end",
        "oblast_active_end",
    ]
    regions = sorted(str(r) for r in panel["region"].unique())
    m = {}
    hours = None
    for c in cols:
        w = panel.pivot(index="hour", columns="region", values=c)  # noqa: PD010 (unique keys)
        w = w.reindex(columns=regions).sort_index()
        hours = w.index if hours is None else hours
        m[c] = w.to_numpy(dtype="float64", na_value=np.nan)
    assert hours is not None
    return Wide(pd.DatetimeIndex(hours), regions, m)


def make_origins(
    start: pd.Timestamp, end: pd.Timestamp, origin_hours: list[int], tz: str
) -> pd.DatetimeIndex:
    """UTC origins in [start, end] whose local (tz) hour is in `origin_hours`."""
    local = pd.date_range(start.tz_convert(tz).ceil("h"), end.tz_convert(tz), freq="h")
    local = local[local.hour.isin(origin_hours)]
    return local.tz_convert("UTC")


def _csum(a: np.ndarray) -> np.ndarray:
    """Cumulative sum along time with a leading zero row; NaN counts as 0."""
    out = np.zeros((a.shape[0] + 1, a.shape[1]))
    np.cumsum(np.nan_to_num(a), axis=0, out=out[1:])
    return out


def _window_sum(cs: np.ndarray, i: np.ndarray, w: int) -> np.ndarray:
    """Sum over rows (i-w, i] for each origin row i -> (n_origins, R)."""
    hi = i + 1
    lo = np.maximum(hi - w, 0)
    total: np.ndarray = cs[hi] - cs[lo]
    return total


def _hours_since(flag: np.ndarray) -> np.ndarray:
    """Per row: hours since the last row where flag==1 (0 if this row). NEVER if none."""
    t = np.arange(flag.shape[0])[:, None]
    last = np.where(flag > 0, t, -1)
    last = np.maximum.accumulate(last, axis=0)
    return np.where(last >= 0, t - last, NEVER).astype("float64")


def _same_phase_mean(a: np.ndarray, period: int, n: int) -> np.ndarray:
    """Mean of the last `n` values at the same phase (row, row-period, ...), inclusive.

    NaN values (e.g. y_red before the reform) are skipped; NaN if none are available.
    """
    out = np.zeros_like(a, dtype="float64")
    cnt = np.zeros_like(a, dtype="float64")
    for k in range(n):
        lag = k * period
        if lag >= a.shape[0]:
            break
        shifted = np.full_like(a, np.nan, dtype="float64")
        shifted[lag:] = a[: a.shape[0] - lag]
        ok = ~np.isnan(shifted)
        out += np.where(ok, shifted, 0.0)
        cnt += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        mean: np.ndarray = np.where(cnt > 0, out / cnt, np.nan)
    return mean


def origin_features(
    panel: pd.DataFrame, origins: pd.DatetimeIndex, regime_break: pd.Timestamp, tz: str
) -> pd.DataFrame:
    """One row per (origin, region) with features computed from panel hours < origin."""
    wide = to_wide(panel)
    i = wide.idx(origins - HOUR)  # last fully observed hour for each origin
    if (i < MIN_HISTORY_HOURS).any():
        raise ValueError(f"origins need >= {MIN_HISTORY_HOURS} h of panel history")
    R = len(wide.regions)
    feats: dict[str, np.ndarray] = {}

    for p in PREFIXES:
        minutes = wide.m[f"{p}_minutes"]
        flag = (np.nan_to_num(minutes) > 0).astype("float64")
        cs = _csum(minutes / 60.0)
        feats[f"{p}_active_now"] = np.nan_to_num(wide.m[f"{p}_active_end"][i])
        for w in WINDOWS:
            feats[f"{p}_hours_{w}h"] = _window_sum(cs, i, w)
        feats[f"{p}_hours_since"] = _hours_since(flag)[i]
        feats[f"{p}_streak"] = _hours_since(1.0 - flag)[i]
        starts = np.vstack([flag[:1], np.clip(np.diff(flag, axis=0), 0, None)])
        cs_starts = _csum(starts)
        for w in (24, 168):
            feats[f"{p}_starts_{w}h"] = _window_sum(cs_starts, i, w)

    # Trend: recent level vs the 28-day level, and the 14-day slope of daily alert hours.
    for p in ("alert", "oblast"):
        base = feats[f"{p}_hours_672h"] / 672
        feats[f"{p}_anom_24h"] = feats[f"{p}_hours_24h"] / 24 - base
        feats[f"{p}_anom_168h"] = feats[f"{p}_hours_168h"] / 168 - base
        feats[f"{p}_logratio_24h_7d"] = np.log(
            (feats[f"{p}_hours_24h"] / 24 + 0.01) / (feats[f"{p}_hours_168h"] / 168 + 0.01)
        )
        feats[f"{p}_starts_momentum"] = feats[f"{p}_starts_24h"] - feats[f"{p}_starts_168h"] / 7
        cs = _csum(wide.m[f"{p}_minutes"] / 60.0)
        days = np.stack([_window_sum(cs, i - 24 * k, 24) for k in range(14)])  # k=0 newest
        x = -np.arange(14, dtype="float64")[:, None, None]
        xc = x - x.mean()
        feats[f"{p}_slope_14d"] = (xc * (days - days.mean(axis=0))).sum(axis=0) / (xc**2).sum()

    # National and neighbour context (occupied regions excluded from aggregates).
    free = np.array([r not in OCCUPIED for r in wide.regions])
    for name in ("alert_active_now", "alert_hours_24h", "oblast_active_now", "alert_anom_24h"):
        nat = feats[name][:, free].mean(axis=1, keepdims=True)
        feats[f"nat_{name}"] = np.repeat(nat, R, axis=1)
        pos = {r: k for k, r in enumerate(wide.regions)}
        nb = np.zeros_like(feats[name])
        for k, r in enumerate(wide.regions):
            cols = [pos[n] for n in NEIGHBOURS[r] if n in pos and n not in OCCUPIED]
            nb[:, k] = feats[name][:, cols].mean(axis=1) if cols else np.nan
        feats[f"nb_{name}"] = nb

    n = len(origins)
    local = origins.tz_convert(tz)
    out = pd.DataFrame(
        {
            "origin": np.repeat(origins, R),
            "region": np.tile(wide.regions, n),
            "origin_hod": np.repeat(local.hour, R),
            "origin_dow": np.repeat(local.dayofweek, R),
            "post_reform": np.repeat((origins >= regime_break).astype("int8"), R),
            "_i": np.repeat(i, R),
        }
    )
    for name, values in feats.items():
        out[name] = values.reshape(-1).astype("float32")
    return out


def make_dataset(
    panel: pd.DataFrame,
    ofeats: pd.DataFrame,
    leads: list[int] | np.ndarray,
    target: str,
    tz: str,
) -> pd.DataFrame:
    """Cross origin features with leads; add target-hour features and the label.

    Target hour for lead h is [t+h-1, t+h). Target-hour climatology uses the most recent
    same-hour-of-day (28 d) / same-hour-of-week (8 w) values observed before the origin.
    Rows whose target hour is beyond the panel get a NaN label (unscored).
    """
    wide = to_wide(panel)
    T = len(wide.hours)
    y = wide.m[target]
    clim_d = _same_phase_mean(y, 24, 28)
    clim_w = _same_phase_mean(y, 168, 8)
    rpos = {r: k for k, r in enumerate(wide.regions)}
    r_idx = ofeats["region"].map(rpos).to_numpy()
    i = ofeats["_i"].to_numpy()
    lat = ofeats["region"].map(lambda r: CENTER_COORDS[r][0]).to_numpy(float)
    lon = ofeats["region"].map(lambda r: CENTER_COORDS[r][1]).to_numpy(float)
    # Last-year analog: target rate over +-15 days around the same hour one year earlier.
    # Its window ends 350 days before the target hour, i.e. long before any origin.
    y_cs = _csum(y)
    n_cs = _csum((~np.isnan(y)).astype("float64"))

    frames = []
    for h in leads:
        j = i + h  # panel row of the target hour
        jd = j - 24 * np.ceil((j - i) / 24).astype(int)  # latest same-hod row <= i
        jw = j - 168 * np.ceil((j - i) / 168).astype(int)
        target_hour = ofeats["origin"] + (h - 1) * HOUR
        local = pd.DatetimeIndex(target_hour).tz_convert(tz)
        label = np.full(len(j), np.nan)
        inside = j < T
        label[inside] = y[j[inside], r_idx[inside]]
        f = ofeats.drop(columns="_i").copy()
        f["lead"] = np.int16(h)
        f["target_hour"] = target_hour.to_numpy()
        f["target_hod"] = local.hour.astype("int8")
        f["target_dow"] = local.dayofweek.astype("int8")
        f["clim_hod_28d"] = clim_d[jd, r_idx].astype("float32")
        f["clim_how_8w"] = clim_w[jw, r_idx].astype("float32")
        # Seasonal features (target hour): darkness at the centre, calendar, last-year analog.
        mid = pd.DatetimeIndex(target_hour) + pd.Timedelta(minutes=30)
        f["sun_elev"] = solar_elevation(mid, lat, lon).astype("float32")
        doy = local.dayofyear.to_numpy()
        f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
        f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
        f["month"] = local.month.astype("int8")
        lo = np.clip(j - LAST_YEAR_HOURS - ANALOG_HALF_HOURS, 0, T)
        hi = np.clip(j - LAST_YEAR_HOURS + ANALOG_HALF_HOURS + 1, 0, T)
        num = y_cs[hi, r_idx] - y_cs[lo, r_idx]
        den = n_cs[hi, r_idx] - n_cs[lo, r_idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            f["last_year_rate"] = np.where(den > 0, num / den, np.nan).astype("float32")
        f["label"] = label.astype("float32")
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


TREND_FEATURES = frozenset(
    f"{p}_{s}"
    for p in ("alert", "oblast")
    for s in ("anom_24h", "anom_168h", "logratio_24h_7d", "starts_momentum", "slope_14d")
) | {"nat_alert_anom_24h", "nb_alert_anom_24h"}
SEASON_FEATURES = frozenset({"sun_elev", "doy_sin", "doy_cos", "month", "last_year_rate"})


def feature_columns(df: pd.DataFrame, drop: frozenset[str] = frozenset()) -> list[str]:
    skip = {"origin", "target_hour", "label"} | drop
    return [c for c in df.columns if c not in skip]
