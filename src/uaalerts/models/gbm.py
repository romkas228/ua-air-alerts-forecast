"""Global LightGBM model over (origin, region, lead) rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from uaalerts.data.regions import REGION_IDS
from uaalerts.features.build import feature_columns
from uaalerts.models.baselines import Climatology

CLIP = 0.01

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 0,
    "seed": 0,
}


class GBM:
    """LightGBM with optional recency weighting of training rows.

    half_life_days: weight = 0.5 ** (age / half_life), age measured from the newest
    training origin; None = uniform weights.
    base: "climatology" starts boosting from logit(climatology) (init_score), so the trees
    learn only the deviation from the hour-of-day climatology; None = from the global mean.
    drop: feature names to leave out (feature-group ablations).
    """

    name = "gbm"

    def __init__(
        self,
        num_boost_round: int = 400,
        params: dict[str, Any] | None = None,
        half_life_days: float | None = None,
        base: str | None = None,
        drop: frozenset[str] = frozenset(),
    ) -> None:
        self.num_boost_round = num_boost_round
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.half_life_days = half_life_days
        self.base = base
        self.drop = drop
        self.booster: lgb.Booster | None = None
        self.features: list[str] = []

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.features].copy()
        x["region"] = pd.Categorical(x["region"], categories=REGION_IDS)
        return x

    def fit(self, train: pd.DataFrame) -> GBM:
        self.features = feature_columns(train, self.drop)
        weight = None
        if self.half_life_days is not None:
            age = (train["origin"].max() - train["origin"]).dt.total_seconds() / 86400
            weight = np.power(0.5, age.to_numpy() / self.half_life_days)
        ds = lgb.Dataset(
            self._x(train),
            label=train["label"].to_numpy(),
            weight=weight,
            init_score=self._init(train),
            categorical_feature=["region"],
            free_raw_data=True,
        )
        self.booster = lgb.train(self.params, ds, num_boost_round=self.num_boost_round)
        return self

    def _init(self, df: pd.DataFrame) -> np.ndarray | None:
        if self.base is None:
            return None
        if self.base != "climatology":
            raise ValueError(f"unknown base {self.base!r}")
        clim = np.clip(Climatology().predict(df), CLIP, 1 - CLIP)
        logit: np.ndarray = np.log(clim / (1 - clim))
        return logit

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.booster is not None, "fit first"
        raw = np.asarray(self.booster.predict(self._x(df), raw_score=True), dtype=float)
        init = self._init(df)
        if init is not None:
            raw = raw + init
        p: np.ndarray = 1 / (1 + np.exp(-raw))
        return p

    def importance(self) -> pd.Series:
        assert self.booster is not None
        return pd.Series(
            self.booster.feature_importance("gain"), index=self.booster.feature_name()
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        assert self.booster is not None
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path: Path, base: str | None = None) -> GBM:
        model = cls(base=base)
        model.booster = lgb.Booster(model_file=str(path))
        model.features = model.booster.feature_name()
        return model
