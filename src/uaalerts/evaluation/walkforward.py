"""Weekly walk-forward evaluation of several models on one target."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from uaalerts.config import ForecastConfig
from uaalerts.data.regions import OCCUPIED
from uaalerts.evaluation.splits import Fold, excluded_regions, in_fold, known_at, weekly_folds
from uaalerts.features.build import (
    MIN_HISTORY_HOURS,
    make_dataset,
    make_origins,
    origin_features,
)

log = logging.getLogger(__name__)
HOUR = pd.Timedelta(hours=1)
TRAIN_START = pd.Timestamp("2023-01-01", tz="UTC")


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


def train_leads(max_lead: int) -> np.ndarray:
    """All leads up to 24 h, every 4th beyond (evaluation always uses every lead)."""
    leads: np.ndarray = np.concatenate([np.arange(1, 25), np.arange(28, max_lead + 1, 4)])
    return leads


def run(
    panel: pd.DataFrame,
    cfg: ForecastConfig,
    target: str,
    specs: list[Spec],
    phase: str = "test",
    train_start: pd.Timestamp = TRAIN_START,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    regime_break = pd.Timestamp(cfg.regime_break).tz_convert("UTC")
    data_end = panel["hour"].max() + HOUR
    skip = cfg.test_weeks if phase == "tune" else 0
    n = cfg.tune_weeks if phase == "tune" else cfg.test_weeks
    folds: list[Fold] = weekly_folds(data_end, n, skip_weeks=skip, prefix=f"{phase}-w")

    first_origin = max(train_start, panel["hour"].min() + MIN_HISTORY_HOURS * HOUR + HOUR)
    origins = make_origins(first_origin, folds[-1].end - HOUR, cfg.origin_hours, cfg.timezone)
    ofeats = origin_features(panel, origins, regime_break, cfg.timezone)

    exclude = excluded_regions(
        panel, target, folds[0].start, cfg.exclude_window_days, cfg.exclude_if_rate_above
    )
    exclude = sorted(set(exclude) | OCCUPIED)
    ofeats = ofeats[~ofeats["region"].isin(exclude)]
    log.info("target %s: excluded regions %s", target, exclude)

    lead_max = cfg.max_lead_hours
    train_ofeats = ofeats[ofeats["origin"] < folds[-1].start]
    train_ds = make_dataset(panel, train_ofeats, train_leads(lead_max), target, cfg.timezone)
    train_ds = train_ds[train_ds["label"].notna()]
    eval_ofeats = ofeats[ofeats["origin"] >= folds[0].start]
    eval_ds = make_dataset(panel, eval_ofeats, np.arange(1, lead_max + 1), target, cfg.timezone)

    out = []
    for fold in folds:
        test = eval_ds[in_fold(eval_ds, fold)].copy()
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
    return preds[keep], {"excluded": exclude, "folds": [f.name for f in folds]}
