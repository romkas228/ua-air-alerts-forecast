"""Probabilistic forecast metrics, overall and by group."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6
LEAD_BINS = [0, 6, 24, 72, 168]
LEAD_LABELS = ["1-6h", "7-24h", "1-3d", "4-7d"]


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score(y: np.ndarray, p: np.ndarray, p_ref: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = {
        "n": float(len(y)),
        "base_rate": float(y.mean()) if len(y) else np.nan,
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
    }
    two_classes = 0 < y.sum() < len(y)
    out["roc_auc"] = float(roc_auc_score(y, p)) if two_classes else np.nan
    out["pr_auc"] = float(average_precision_score(y, p)) if two_classes else np.nan
    if p_ref is not None:
        ref = brier(y, np.asarray(p_ref, dtype=float))
        out["bss"] = 1 - out["brier"] / ref if ref > 0 else np.nan
    return out


def lead_bucket(lead: pd.Series) -> pd.Series:
    return pd.cut(lead, LEAD_BINS, labels=LEAD_LABELS)


def score_table(
    df: pd.DataFrame, models: list[str], by: list[str] | None = None, ref: str = "climatology"
) -> pd.DataFrame:
    """Long table: one row per (group..., model) with all metrics.

    `df` holds `label` and one probability column `p_<model>` per model.
    """
    groups = df.groupby(by, observed=True) if by else [((), df)]
    rows = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for m in models:
            s = score(g["label"].to_numpy(), g[f"p_{m}"].to_numpy(), g[f"p_{ref}"].to_numpy())
            rows.append({**dict(zip(by or [], key, strict=True)), "model": m, **s})
    return pd.DataFrame(rows)


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "y": y, "p": p})
    return df.groupby("bin").agg(p_mean=("p", "mean"), y_rate=("y", "mean"), n=("y", "size"))


FINE_LEAD_BINS = [0, 1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 72, 168]


def skill_ci_by_lead(
    preds: pd.DataFrame,
    model: str,
    bins: list[int] = FINE_LEAD_BINS,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Brier skill score vs climatology per lead bin with a 95% bootstrap CI.

    Resampling unit = forecast day (origins of one day), because rows of the same day are
    strongly correlated; resampling rows would make the interval far too narrow.
    """
    rng = np.random.default_rng(seed)
    d = preds.assign(
        day=preds["origin"].dt.floor("D"),
        e_m=(preds[f"p_{model}"] - preds["label"]) ** 2,
        e_c=(preds["p_climatology"] - preds["label"]) ** 2,
        bin=pd.cut(preds["lead"], bins, labels=False),
    )
    d = d[d["bin"].notna()]  # leads outside the bins
    g = d.groupby(["bin", "day"], observed=True)[["e_m", "e_c"]].sum().reset_index()
    rows = []
    for b_raw, x in g.groupby("bin"):
        b = int(float(str(b_raw)))
        m, c = x["e_m"].to_numpy(), x["e_c"].to_numpy()
        idx = rng.integers(0, len(x), (n_boot, len(x)))
        boot = 1 - m[idx].sum(axis=1) / c[idx].sum(axis=1)
        rows.append(
            {
                "lead_from": bins[b] + 1,
                "lead_to": bins[b + 1],
                "bss": 1 - m.sum() / c.sum(),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
                "days": len(x),
            }
        )
    return pd.DataFrame(rows)
