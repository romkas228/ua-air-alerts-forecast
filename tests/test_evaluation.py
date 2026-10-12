import itertools

import numpy as np
import pandas as pd

from uaalerts.evaluation.metrics import lead_bucket, score, score_table
from uaalerts.evaluation.splits import excluded_regions, in_fold, known_at, weekly_folds
from uaalerts.models.baselines import Climatology, Persistence

UTC = "UTC"


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


def test_weekly_folds_are_contiguous_and_end_at_data_end() -> None:
    end = ts("2026-10-10 09:00")
    folds = weekly_folds(end, 4)
    assert folds[-1].end == end
    assert all(a.end == b.start for a, b in itertools.pairwise(folds))
    tune = weekly_folds(end, 4, skip_weeks=4)
    assert tune[-1].end == folds[0].start


def test_known_at_purges_rows_whose_target_hour_is_not_over() -> None:
    t = ts("2026-09-20 00:00")
    df = pd.DataFrame(
        {
            "origin": [t - pd.Timedelta(hours=10)] * 3 + [t],
            "target_hour": [
                t - pd.Timedelta(hours=2),  # over by t
                t - pd.Timedelta(hours=1),  # ends exactly at t -> known
                t,  # in the future
                t,
            ],
            "label": [1.0, 0.0, 1.0, np.nan],
        }
    )
    assert known_at(df, t).tolist() == [True, True, False, False]


def test_in_fold_selects_origins_and_observed_labels() -> None:
    f = weekly_folds(ts("2026-10-08"), 1)[0]
    df = pd.DataFrame(
        {"origin": [f.start - pd.Timedelta(hours=6), f.start, f.end], "label": [1.0, 0.0, 1.0]}
    )
    assert in_fold(df, f).tolist() == [False, True, False]


def test_excluded_regions_by_recent_rate() -> None:
    hours = pd.date_range("2026-09-01", periods=24 * 30, freq="h", tz=UTC)
    panel = pd.DataFrame(
        {
            "region": ["a"] * len(hours) + ["b"] * len(hours),
            "hour": list(hours) * 2,
            "y_alert": [1] * len(hours) + [1, 0] * (len(hours) // 2),
        }
    )
    assert excluded_regions(panel, "y_alert", hours[-1], 28, 0.95) == ["a"]


def test_score_perfect_and_skill() -> None:
    y = np.array([0, 1, 0, 1])
    s = score(y, y.astype(float), p_ref=np.full(4, 0.5))
    assert s["brier"] == 0 and s["roc_auc"] == 1 and s["bss"] == 1


def test_score_table_and_buckets() -> None:
    df = pd.DataFrame(
        {
            "label": [0, 1, 1, 0],
            "lead": [1, 7, 30, 168],
            "p_climatology": [0.5] * 4,
            "p_x": [0.1, 0.9, 0.8, 0.2],
        }
    )
    df["lead_bucket"] = lead_bucket(df["lead"])
    assert df["lead_bucket"].astype(str).tolist() == ["1-6h", "7-24h", "1-3d", "4-7d"]
    t = score_table(df, ["climatology", "x"])
    assert t.set_index("model").loc["x", "bss"] > 0


def test_persistence_decays_to_climatology() -> None:
    df = pd.DataFrame(
        {
            "lead": [1, 168],
            "alert_active_now": [1.0, 1.0],
            "clim_hod_28d": [0.2, 0.2],
            "clim_how_8w": [0.2, 0.2],
            "label": [1.0, 0.0],
        }
    )
    m = Persistence()
    p = m.predict(df)
    assert p[0] > 0.8 and abs(p[1] - 0.2) < 1e-6
    assert (Climatology().predict(df) == 0.2).all()


def test_skill_ci_by_lead_brackets_the_point_estimate() -> None:
    from uaalerts.evaluation.metrics import skill_ci_by_lead

    rng = np.random.default_rng(1)
    n = 6000
    origin = pd.Timestamp("2026-01-01", tz=UTC) + pd.to_timedelta(rng.integers(0, 30, n), "D")
    lead = rng.integers(1, 169, n)
    y = rng.uniform(size=n) < 0.3
    good = np.where(lead <= 6, np.where(y, 0.8, 0.1), 0.3)  # skilful only at short leads
    df = pd.DataFrame(
        {
            "origin": origin,
            "lead": lead,
            "label": y.astype(float),
            "p_x": good,
            "p_climatology": 0.3,
        }
    )
    t = skill_ci_by_lead(df, "x", n_boot=200)
    assert (t["ci_low"] <= t["bss"]).all() and (t["bss"] <= t["ci_high"]).all()
    assert t.loc[t["lead_to"] <= 6, "ci_low"].min() > 0.3
    assert t.loc[t["lead_from"] > 6, "bss"].abs().max() < 1e-9


def test_skill_ci_by_lead_ignores_leads_outside_bins() -> None:
    from uaalerts.evaluation.metrics import skill_ci_by_lead

    df = pd.DataFrame(
        {
            "origin": pd.Timestamp("2026-01-01", tz=UTC),
            "lead": [1, 2, 30],
            "label": [1.0, 0.0, 1.0],
            "p_x": [0.9, 0.1, 0.5],
            "p_climatology": 0.5,
        }
    )
    t = skill_ci_by_lead(df, "x", bins=[0, 1, 2], n_boot=10)
    assert t["lead_to"].tolist() == [1, 2]
