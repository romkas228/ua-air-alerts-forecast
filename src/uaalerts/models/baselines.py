"""Reference forecasters. All use only origin-time features (leakage-safe by construction)."""

from __future__ import annotations

import numpy as np
import pandas as pd


class Climatology:
    """Rate of the target at the target's hour-of-day over the 28 days before the origin."""

    name = "climatology"

    def fit(self, train: pd.DataFrame) -> Climatology:
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["clim_hod_28d"].fillna(df["clim_how_8w"]).fillna(0.0).to_numpy(float)


class Persistence:
    """The current state, decaying to climatology with lead: w(h)=exp(-h/tau).

    tau is the single parameter, chosen on the training rows by Brier score.
    """

    name = "persistence"

    def __init__(self, state_col: str = "alert_active_now") -> None:
        self.state_col = state_col
        self.tau = 6.0

    def _p(self, df: pd.DataFrame, tau: float) -> np.ndarray:
        w = np.exp(-df["lead"].to_numpy(float) / tau)
        clim = Climatology().predict(df)
        state = df[self.state_col].fillna(0.0).to_numpy(float)
        p: np.ndarray = w * state + (1 - w) * clim
        return p

    def fit(self, train: pd.DataFrame) -> Persistence:
        sample = train.sample(min(len(train), 500_000), random_state=0)
        y = sample["label"].to_numpy(float)
        grid = [0.5, 1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48]
        losses = [np.mean((self._p(sample, t) - y) ** 2) for t in grid]
        self.tau = float(grid[int(np.argmin(losses))])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._p(df, self.tau)
