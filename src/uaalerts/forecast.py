"""Final model training and the operational 7-day forecast from the latest data."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from uaalerts.config import ForecastConfig
from uaalerts.data.regions import OCCUPIED, REGION_IDS
from uaalerts.evaluation.splits import excluded_regions, known_at
from uaalerts.evaluation.walkforward import TRAIN_START, train_leads
from uaalerts.features.build import MIN_HISTORY_HOURS, make_dataset, make_origins, origin_features
from uaalerts.models.baselines import Climatology
from uaalerts.models.gbm import GBM

HOUR = pd.Timedelta(hours=1)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def latest_origin(panel: pd.DataFrame, cfg: ForecastConfig) -> pd.Timestamp:
    """Newest validated origin (00/06/12/18 Kyiv) at or before the end of the data."""
    data_end = panel["hour"].max() + HOUR
    origins = make_origins(
        data_end - pd.Timedelta(hours=24), data_end, cfg.origin_hours, cfg.timezone
    )
    return pd.Timestamp(origins[-1])


def train_final(panel: pd.DataFrame, cfg: ForecastConfig, target: str, model_dir: Path) -> Path:
    """Fit the GBM on every row whose label is known at the end of the data; save it."""
    regime_break = pd.Timestamp(cfg.regime_break).tz_convert("UTC")
    data_end = panel["hour"].max() + HOUR
    first = max(TRAIN_START, panel["hour"].min() + MIN_HISTORY_HOURS * HOUR + HOUR)
    origins = make_origins(first, data_end - HOUR, cfg.origin_hours, cfg.timezone)
    ofeats = origin_features(panel, origins, regime_break, cfg.timezone)
    ofeats = ofeats[~ofeats["region"].isin(OCCUPIED)]
    ds = make_dataset(panel, ofeats, train_leads(cfg.max_lead_hours), target, cfg.timezone)
    ds = ds[known_at(ds, data_end)]
    model = GBM().fit(ds)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{target}_gbm.txt"
    model.save(path)
    meta = {
        "target": target,
        "model": "gbm",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_end": data_end.isoformat(),
        "train_rows": len(ds),
        "git_commit": _git_commit(),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def make_forecast(
    panel: pd.DataFrame, cfg: ForecastConfig, target: str, model: GBM
) -> pd.DataFrame:
    """Probabilities for every region and lead 1..168 from the latest origin."""
    regime_break = pd.Timestamp(cfg.regime_break).tz_convert("UTC")
    origin = latest_origin(panel, cfg)
    ofeats = origin_features(panel, pd.DatetimeIndex([origin]), regime_break, cfg.timezone)
    ds = make_dataset(panel, ofeats, np.arange(1, cfg.max_lead_hours + 1), target, cfg.timezone)
    data_end = panel["hour"].max() + HOUR
    excluded = (
        set(
            excluded_regions(
                panel, target, data_end, cfg.exclude_window_days, cfg.exclude_if_rate_above
            )
        )
        | OCCUPIED
    )
    out = ds[["origin", "region", "lead", "target_hour", "label"]].copy()
    out["p_gbm"] = model.predict(ds)
    out["p_climatology"] = Climatology().predict(ds)
    out["target"] = target
    out["excluded"] = out["region"].isin(excluded)
    out["region"] = pd.Categorical(out["region"], categories=REGION_IDS)
    return out
