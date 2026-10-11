"""Weekly walk-forward evaluation of several models on one target."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from uaalerts.config import ForecastConfig
from uaalerts.data.regions import OCCUPIED
from uaalerts.evaluation.splits import (
    WEEK,
    Fold,
    excluded_regions,
    in_fold,
    known_at,
    weekly_folds,
)
from uaalerts.features.build import (
    MIN_HISTORY_HOURS,
    make_dataset,
    make_origins,
    origin_features,
)

log = logging.getLogger(__name__)
HOUR = pd.Timedelta(hours=1)
TRAIN_START = pd.Timestamp("2023-01-01", tz="UTC")
BACKTEST_START, BACKTEST_END = "2024-09-01", "2026-09-01"


class Model(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> Model: ...

    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class Spec:
    """A model to evaluate: factory(target) + optional training window (days before fold)."""

    name: str
    factory: Callable[[str], Model]
    window_days: int | None = None


def folds_for(phase: str, cfg: ForecastConfig, data_end: pd.Timestamp) -> list[Fold]:
    """tune: 4 weeks before dev · dev: 4 weeks ending at the frozen dev end (the former,
    already-seen test) · prospective: weekly folds after the dev end, never seen while
    developing (the last fold may be partial)."""
    dev_end = pd.Timestamp(cfg.dev_end).tz_convert("UTC")
    if phase == "tune":
        return weekly_folds(dev_end, cfg.tune_weeks, skip_weeks=cfg.test_weeks, prefix="tune-w")
    if phase == "dev":
        return weekly_folds(dev_end, cfg.test_weeks, prefix="dev-w")
    if phase == "prospective":
        if data_end <= dev_end:
            raise ValueError("no data after the dev end yet; run `uaalerts fetch && build`")
        folds, start, k = [], dev_end, 1
        while start < data_end:
            folds.append(Fold(f"prosp-w{k}", start, min(start + WEEK, data_end)))
            start, k = start + WEEK, k + 1
        return folds
    if phase == "backtest":
        # Monthly refits over two full years before the regime break (two winters).
        months = pd.date_range(BACKTEST_START, BACKTEST_END, freq="MS", tz="UTC")
        return [
            Fold(f"bt-{a:%Y-%m}", pd.Timestamp(a), pd.Timestamp(b))
            for a, b in itertools.pairwise(months)
        ]
    raise ValueError(f"unknown phase {phase!r}")


def train_leads(max_lead: int) -> np.ndarray:
    """All leads up to 24 h, every 4th beyond (evaluation always uses every lead)."""
    leads: np.ndarray = np.concatenate([np.arange(1, 25), np.arange(28, max_lead + 1, 4)])
    return leads


def run(
    panel: pd.DataFrame,
    cfg: ForecastConfig,
    target: str,
    specs: list[Spec],
    phase: str = "dev",
    train_start: pd.Timestamp = TRAIN_START,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    regime_break = pd.Timestamp(cfg.regime_break).tz_convert("UTC")
    data_end = panel["hour"].max() + HOUR
    folds = folds_for(phase, cfg, data_end)

    first_origin = max(train_start, panel["hour"].min() + MIN_HISTORY_HOURS * HOUR + HOUR)
    origins = make_origins(first_origin, folds[-1].end - HOUR, cfg.origin_hours, cfg.timezone)
    ofeats = origin_features(panel, origins, regime_break, cfg.timezone)

    # Training keeps every non-occupied region; evaluation exclusions are decided per fold
    # from the 28 days before it (never from the fold itself).
    ofeats = ofeats[~ofeats["region"].isin(OCCUPIED)]
    excluded: dict[str, list[str]] = {
        f.name: excluded_regions(
            panel, target, f.start, cfg.exclude_window_days, cfg.exclude_if_rate_above
        )
        for f in folds
    }
    log.info("target %s: excluded per fold %s", target, excluded)

    lead_max = cfg.max_lead_hours
    train_ofeats = ofeats[ofeats["origin"] < folds[-1].start]
    train_ds = make_dataset(panel, train_ofeats, train_leads(lead_max), target, cfg.timezone)
    train_ds = train_ds[train_ds["label"].notna()]
    eval_ofeats = ofeats[ofeats["origin"] >= folds[0].start]
    eval_ds = make_dataset(panel, eval_ofeats, np.arange(1, lead_max + 1), target, cfg.timezone)

    out = []
    for fold in folds:
        test = eval_ds[in_fold(eval_ds, fold) & ~eval_ds["region"].isin(excluded[fold.name])]
        test = test.copy()
        test["fold"] = fold.name
        known = train_ds[known_at(train_ds, fold.start)]
        # Purging guard: no training label may come from at/after the fold start.
        assert (known["target_hour"] + HOUR <= fold.start).all()
        assert (test["origin"] >= fold.start).all()
        for spec in specs:
            tr = known
            if spec.window_days is not None:
                tr = tr[tr["origin"] >= fold.start - pd.Timedelta(days=spec.window_days)]
            log.info("%s %s: fit %s on %d rows", target, fold.name, spec.name, len(tr))
            model = spec.factory(target).fit(tr)
            test[f"p_{spec.name}"] = model.predict(test)
        out.append(test)
    preds = pd.concat(out, ignore_index=True)
    keep = ["origin", "region", "lead", "target_hour", "label", "fold"]
    keep += [f"p_{s.name}" for s in specs]
    every = sorted(set.intersection(*(set(v) for v in excluded.values())) | OCCUPIED)
    return preds[keep], {
        "excluded": every,
        "folds": [f.name for f in folds],
        "excluded_per_fold": excluded,
    }
