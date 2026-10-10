"""Leakage tests: features at origin t must not depend on anything at or after t."""

import numpy as np
import pandas as pd
import pytest

from uaalerts.data.panel import build_panel
from uaalerts.features.asof import as_of
from uaalerts.features.build import make_dataset, origin_features

UTC = "UTC"
TZ = "Europe/Kyiv"
START = pd.Timestamp("2026-06-01", tz=UTC)
END = pd.Timestamp("2026-09-20", tz=UTC)
BREAK = pd.Timestamp("2026-09-06", tz=UTC)
REGIONS = ["kyiv_city", "kharkivska", "dnipropetrovska", "lvivska", "kyivska", "luhanska"]
LOCS = {
    "kyiv_city": [("oblast", "м. Київ")],
    "kharkivska": [("oblast", "Харківська область"), ("raion", "Харківський район")],
    "dnipropetrovska": [("raion", "Нікопольський район"), ("city", "м. Дніпро")],
    "lvivska": [("oblast", "Львівська область")],
    "kyivska": [("raion", "Білоцерківський район"), ("raion", "Бучанський район")],
    "luhanska": [("oblast", "Луганська область")],
}


ORIGINS = [
    pd.Timestamp("2026-08-01 03:00", tz=UTC),
    pd.Timestamp("2026-09-05 21:00", tz=UTC),  # right before the regime break
    pd.Timestamp("2026-09-12 09:00", tz=UTC),
    pd.Timestamp("2026-09-19 15:00", tz=UTC),
]


def random_events(seed: int = 0, n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    span = (END - START).total_seconds()
    for k in range(n):
        region = REGIONS[rng.integers(len(REGIONS))]
        level, loc = LOCS[region][rng.integers(len(LOCS[region]))]
        start = START + pd.Timedelta(seconds=float(rng.uniform(0, span)))
        dur = pd.Timedelta(minutes=float(rng.exponential(90)))
        rows.append(
            {
                "region": region,
                "raion": pd.NA,
                "location": loc,
                "geo_level": level,
                "alert_type": "air_raid",
                "alert_level": rng.choice(["yellow", "red"]) if start >= BREAK else pd.NA,
                "started_at": start,
                "finished_at": start + dur,
                "source": "synthetic",
                "source_id": str(k),
            }
        )
    # A never-ending alert (occupied region style) and one running past the data end.
    rows.append(
        {
            **rows[0],
            "region": "luhanska",
            "location": "Луганська область",
            "geo_level": "oblast",
            "started_at": START,
            "finished_at": pd.NaT,
        }
    )
    # Alerts running through every test origin: their end time is future information.
    for k, t in enumerate(ORIGINS):
        rows.append(
            {
                **rows[0],
                "region": "kyiv_city",
                "location": "м. Київ",
                "geo_level": "oblast",
                "started_at": t - pd.Timedelta(minutes=30),
                "finished_at": t + pd.Timedelta(hours=2),
                "source_id": f"span{k}",
            }
        )
    df = pd.DataFrame(rows)
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    return df


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    return random_events()


@pytest.fixture(scope="module")
def full_panel(events: pd.DataFrame) -> pd.DataFrame:
    return build_panel(events, START, END, BREAK)


def test_as_of_hides_the_future(events: pd.DataFrame) -> None:
    t = ORIGINS[2]
    known = as_of(events, t)
    assert (known["started_at"] < t).all()
    assert (known["finished_at"].dropna() < t).all()
    running = events[(events["started_at"] < t) & (events["finished_at"] >= t)]
    assert len(running) > 0  # the fixture really exercises ongoing alerts
    assert known.loc[running.index, "finished_at"].isna().all()


@pytest.mark.parametrize("t", ORIGINS)
def test_origin_features_are_truncation_invariant(
    events: pd.DataFrame, full_panel: pd.DataFrame, t: pd.Timestamp
) -> None:
    """Features from the full history == features from history as known at t."""
    truncated_panel = build_panel(as_of(events, t), START, t, BREAK)
    origins = pd.DatetimeIndex([t])
    a = origin_features(full_panel, origins, BREAK, TZ)
    b = origin_features(truncated_panel, origins, BREAK, TZ)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("t", ORIGINS)
def test_target_hour_features_are_truncation_invariant(
    events: pd.DataFrame, full_panel: pd.DataFrame, t: pd.Timestamp
) -> None:
    truncated_panel = build_panel(as_of(events, t), START, t, BREAK)
    origins = pd.DatetimeIndex([t])
    leads = [1, 2, 23, 24, 25, 100, 167, 168]
    for target in ("y_alert", "y_red"):
        a = make_dataset(
            full_panel, origin_features(full_panel, origins, BREAK, TZ), leads, target, TZ
        )
        b = make_dataset(
            truncated_panel, origin_features(truncated_panel, origins, BREAK, TZ), leads, target, TZ
        )
        # Labels differ by construction (the truncated world has no future); nothing else may.
        assert b["label"].isna().all()
        pd.testing.assert_frame_equal(a.drop(columns="label"), b.drop(columns="label"))


def test_labels_are_the_target_hours(full_panel: pd.DataFrame) -> None:
    t = ORIGINS[2]
    ds = make_dataset(
        full_panel,
        origin_features(full_panel, pd.DatetimeIndex([t]), BREAK, TZ),
        [1, 5, 168],
        "y_alert",
        TZ,
    )
    p = full_panel.set_index(["region", "hour"])["y_alert"]
    for row in ds.itertuples():
        assert row.target_hour == t + pd.Timedelta(hours=row.lead - 1)
        assert row.label == p[(row.region, row.target_hour)]


def test_features_change_only_with_past_data(events: pd.DataFrame) -> None:
    """Perturbing events after t must not change features at t (direct sanity check)."""
    t = ORIGINS[2]
    future = events.copy()
    late = future["started_at"] >= t
    future.loc[late, "finished_at"] = future.loc[late, "finished_at"] + pd.Timedelta(hours=5)
    pa = build_panel(events, START, END, BREAK)
    pb = build_panel(future, START, END, BREAK)
    o = pd.DatetimeIndex([t])
    pd.testing.assert_frame_equal(
        origin_features(pa, o, BREAK, TZ), origin_features(pb, o, BREAK, TZ)
    )
