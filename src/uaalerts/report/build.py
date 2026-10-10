"""Static evaluation report: PNG figures + reports/REPORT.md (renders on GitHub)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.dates import date2num
from matplotlib.figure import Figure

from uaalerts.evaluation.metrics import brier, lead_bucket, reliability, score_table

# Reference categorical palette (dataviz skill, light mode), fixed slot order.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#b9b8b2"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]
MODEL_COLORS = {"gbm_hl90": SLOTS[0], "gbm": SLOTS[1], "persistence": SLOTS[2]}
MODEL_LABELS = {
    "gbm_hl90": "LightGBM, recency-weighted (main)",
    "gbm": "LightGBM, uniform weights",
    "persistence": "Persistence → climatology",
    "climatology": "Climatology (hour-of-day, 28 d)",
}


def _style(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=TEXT_2)


def _save(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def skill_by_lead(preds: pd.DataFrame, models: list[str], title: str, path: Path) -> None:
    rows = []
    for lead, g in preds.groupby("lead"):
        ref = brier(g["label"].to_numpy(), g["p_climatology"].to_numpy())
        for m in models:
            bs = brier(g["label"].to_numpy(), g[f"p_{m}"].to_numpy())
            rows.append({"lead": lead, "model": m, "bss": 1 - bs / ref if ref > 0 else np.nan})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhline(0, color=TEXT_2, linewidth=1)
    ax.text(168, 0.003, "climatology", color=TEXT_2, ha="right", va="bottom", fontsize=9)
    for m in models:
        s = df[df["model"] == m].set_index("lead")["bss"].rolling(6, min_periods=1).mean()
        ax.plot(s.index, s.to_numpy(), color=MODEL_COLORS[m], linewidth=2, label=MODEL_LABELS[m])
    ax.set_xlim(1, 168)
    ax.set_xticks([1, 6, 24, 48, 72, 96, 120, 144, 168])
    _style(ax, title, "lead time, hours", "Brier skill score vs climatology")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9)
    _save(fig, path)


def reliability_plot(preds: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    for m, color in (("gbm_hl90", SLOTS[0]), ("climatology", SLOTS[1])):
        r = reliability(preds["label"].to_numpy(), preds[f"p_{m}"].to_numpy())
        r = r[r["n"] >= 50]
        ax.plot(
            r["p_mean"],
            r["y_rate"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=MODEL_LABELS[m],
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax, title, "forecast probability", "observed frequency")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=8, loc="upper left")
    _save(fig, path)


def regime_plot(panel: pd.DataFrame, region: str, regime_break: pd.Timestamp, path: Path) -> None:
    p = panel[(panel["region"] == region) & (panel["hour"] >= "2025-01-01")].set_index("hour")
    weekly = p[["y_alert", "y_red"]].astype(float).resample("W").mean()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(
        weekly.index, weekly["y_alert"], color=SLOTS[0], linewidth=2, label="any level (y_alert)"
    )
    ax.plot(weekly.index, weekly["y_red"], color=SLOTS[1], linewidth=2, label="red (y_red)")
    x_break = float(date2num(regime_break.tz_convert(None)))  # type: ignore[no-untyped-call]
    ax.axvline(x_break, color=TEXT_2, linewidth=1, linestyle="--")
    ax.text(
        x_break,
        0.95,
        " yellow/red levels\n introduced",
        color=TEXT_2,
        fontsize=9,
        va="top",
        transform=ax.get_xaxis_transform(),
    )
    ax.set_ylim(0, 1)
    _style(ax, "Kyiv: weekly share of hours under alert", "", "share of hours")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper left")
    _save(fig, path)


def example_plot(preds: pd.DataFrame, region: str, title: str, path: Path) -> None:
    g = preds[preds["region"] == region]
    origin = g["origin"].max() - pd.Timedelta(days=7)
    origin = g.loc[g["origin"] <= origin, "origin"].max()
    e = g[g["origin"] == origin].sort_values("lead")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(
        e["target_hour"],
        e["label"],
        width=1 / 24,
        color=MUTED,
        align="edge",
        label="observed alert hour",
    )
    ax.plot(
        e["target_hour"],
        e["p_gbm_hl90"],
        color=SLOTS[0],
        linewidth=2,
        label=MODEL_LABELS["gbm_hl90"],
    )
    ax.set_ylim(0, 1.02)
    _style(ax, f"{title} — forecast issued {origin:%Y-%m-%d %H:%M} UTC", "", "probability")
    ax.legend(
        frameon=False,
        labelcolor=TEXT,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
    )
    _save(fig, path)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    reports_dir: Path, panel: pd.DataFrame, regime_break: pd.Timestamp, meta: dict[str, object]
) -> Path:
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    regime_plot(panel, "kyiv_city", regime_break, fig_dir / "regime_kyiv.png")
    sections = []
    for target, label in (("y_alert", "Alert of any level"), ("y_red", "Red-level alert")):
        d = reports_dir / f"{target}_dev"
        if not (d / "predictions.parquet").exists():
            continue
        preds = pd.read_parquet(d / "predictions.parquet")
        info = json.loads((d / "info.json").read_text())
        models = [c[2:] for c in preds.columns if c.startswith("p_")]
        shown: list[str] = [m for m in ("gbm_hl90", "gbm", "persistence") if m in models]
        skill_by_lead(
            preds,
            shown,
            f"{label}: skill by lead time (dev period, 4 weeks)",
            fig_dir / f"skill_{target}.png",
        )
        reliability_plot(preds, f"{label}: reliability", fig_dir / f"reliability_{target}.png")
        example_plot(
            preds, "kyiv_city", f"Kyiv, {label.lower()}", fig_dir / f"example_{target}.png"
        )
        preds["lead_bucket"] = lead_bucket(preds["lead"])
        cols = ["model", "brier", "bss", "log_loss", "roc_auc", "pr_auc"]
        overall = score_table(preds, models)[["base_rate", *cols]]
        by_lead = score_table(preds, models, by=["lead_bucket"])
        by_lead = by_lead[by_lead["model"].isin(["climatology", "persistence", "gbm_hl90"])]
        by_fold = score_table(preds, ["climatology", "gbm_hl90"], by=["fold"])
        by_region = score_table(preds, ["gbm_hl90"], by=["region"]).sort_values("bss")
        sections.append(
            f"## {label} (`{target}`)\n\n"
            f"Test origins: {preds['origin'].min():%Y-%m-%d %H:%M} → "
            f"{preds['origin'].max():%Y-%m-%d %H:%M} UTC · {len(preds):,} scored "
            f"(origin, region, lead) rows · excluded regions: {', '.join(info['excluded'])}\n\n"
            f"### Overall\n\n{_md_table(overall)}\n\n"
            f"### By lead time\n\n"
            f"{_md_table(by_lead[['lead_bucket', 'base_rate', *cols]])}\n\n"
            f"![skill](figures/skill_{target}.png)\n\n"
            f"### By test week (model refit before each week)\n\n"
            f"{_md_table(by_fold[['fold', 'base_rate', *cols]])}\n\n"
            f"### By region (main model)\n\n"
            f"{_md_table(by_region[['region', 'base_rate', 'brier', 'bss', 'roc_auc']])}\n\n"
            f"![reliability](figures/reliability_{target}.png) "
            f"![example](figures/example_{target}.png)\n"
        )
    text = (
        "# Evaluation report\n\n"
        "_Generated by `uaalerts report`. Do not edit by hand._\n\n"
        f"Data observed until **{meta.get('data_end')}** · sources stitched at "
        f"{meta.get('stitch_at')} (hourly agreement in overlap "
        f"{float(str(meta.get('overlap_agreement', 0))):.1%}).\n\n"
        "**Setup.** Forecast origins every 6 h (00/06/12/18 Kyiv); for every region and every "
        "lead 1-168 h the probability of an alert covering the oblast centre in that hour. "
        "Weekly walk-forward over the last 4 weeks: before each week the models are refit on "
        "rows whose labels were already known (origin + lead ≤ week start). The model choice "
        "(recency-weighted LightGBM) was fixed on the 4 weeks *before* the test period.\n\n"
        "**Leakage controls.** Features at origin t use panel hours < t only; "
        "`tests/test_leakage.py` asserts that features from the full history equal features "
        "built from `as_of(events, t)`, and the runner asserts purging for every fold.\n\n"
        "![regime](figures/regime_kyiv.png)\n\n"
        "The 2026-09-06 reform (yellow = drones, red = missiles/ballistics/massed attacks) "
        "roughly quadrupled alert time in Kyiv; the whole test period lies after it.\n\n"
        + "\n".join(sections)
    )
    path = reports_dir / "REPORT.md"
    path.write_text(text)
    return path
