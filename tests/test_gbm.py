import json
from pathlib import Path

import numpy as np
import pandas as pd

from uaalerts.forecast import load_final
from uaalerts.models.gbm import GBM


def toy(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clim = rng.uniform(0.05, 0.9, n)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(np.log(clim / (1 - clim)) + 1.5 * x)))
    return pd.DataFrame(
        {
            "origin": pd.Timestamp("2026-01-01", tz="UTC"),
            "target_hour": pd.Timestamp("2026-01-01", tz="UTC"),
            "region": rng.choice(["kyiv_city", "lvivska"], n),
            "x": x,
            "clim_hod_28d": clim,
            "clim_how_8w": clim,
            "label": (rng.uniform(size=n) < p).astype(float),
        }
    )


def test_residual_model_starts_from_climatology_and_round_trips(tmp_path: Path) -> None:
    df = toy()
    m = GBM(num_boost_round=50, base="climatology", params={"min_data_in_leaf": 20}).fit(df)
    p = m.predict(df)
    assert ((p > 0) & (p < 1)).all()
    # A (practically) empty booster == pure climatology: the model is anchored on it.
    flat = GBM(num_boost_round=1, base="climatology", params={"learning_rate": 1e-9}).fit(df)
    assert np.allclose(flat.predict(df), np.clip(df["clim_hod_28d"], 0.01, 0.99), atol=1e-6)
    # Save + load (base read from the sidecar json) gives identical predictions.
    path = tmp_path / "y_alert_gbm_resid.json"
    m.save(path.with_suffix(".txt"))
    path.write_text(json.dumps({"model": "gbm_resid", "base": "climatology"}))
    assert np.allclose(load_final(path).predict(df), p)
