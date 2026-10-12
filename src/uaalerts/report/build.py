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

from uaalerts.evaluation.metrics import (
    brier,
    lead_bucket,
    reliability,
    score_table,
    skill_ci_by_lead,
)

# Reference categorical palette (dataviz skill, light mode), fixed slot order.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#b9b8b2"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]
MODEL_LABELS = {
    "gbm_resid": "LightGBM on top of climatology",
    "gbm_resid_short": "LightGBM on top of climatology, trained on 1-12 h",
    "gbm_resid_season": "LightGBM on top of climatology + season",
    "gbm_trend": "LightGBM + trend features",
    "gbm_trend_season": "LightGBM + trend + season",
    "gbm_v1": "LightGBM, first feature set",
    "gbm_hl90": "LightGBM, recency-weighted",
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


def label(m: str) -> str:
    return MODEL_LABELS.get(m, m)


def skill_by_lead(preds: pd.DataFrame, models: list[str], title: str, path: Path) -> None:
    """`models` in slot order (max 3: categorical all-pairs limit)."""
    rows = []
    for lead, g in preds.groupby("lead"):
        ref = brier(g["label"].to_numpy(), g["p_climatology"].to_numpy())
        for m in models:
            bs = brier(g["label"].to_numpy(), g[f"p_{m}"].to_numpy())
            rows.append({"lead": lead, "model": m, "bss": 1 - bs / ref if ref > 0 else np.nan})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhline(0, color=TEXT_2, linewidth=1)
    ax.text(168, -0.004, "climatology", color=TEXT_2, ha="right", va="top", fontsize=9)
    for m, color in zip(models, SLOTS, strict=False):
        s = df[df["model"] == m].set_index("lead")["bss"].rolling(6, min_periods=1).mean()
        ax.plot(s.index, s.to_numpy(), color=color, linewidth=2, label=label(m))
    ax.set_xlim(1, 168)
    ax.set_xticks([1, 6, 24, 48, 72, 96, 120, 144, 168])
    _style(ax, title, "lead time, hours", "Brier skill score vs climatology")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9)
    _save(fig, path)


def reliability_plot(preds: pd.DataFrame, main: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    for m, color in ((main, SLOTS[0]), ("climatology", SLOTS[1])):
        r = reliability(preds["label"].to_numpy(), preds[f"p_{m}"].to_numpy())
        r = r[r["n"] >= 50]
        ax.plot(
            r["p_mean"],
            r["y_rate"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=label(m),
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


def example_plot(preds: pd.DataFrame, main: str, region: str, title: str, path: Path) -> None:
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
        e[f"p_{main}"],
        color=SLOTS[0],
        linewidth=2,
        label=label(main),
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


KEYS = ["origin", "region", "lead"]


def load_preds(reports_dir: Path, target: str, phase: str) -> pd.DataFrame | None:
    """Predictions of a phase merged with tagged runs of the same phase (extra models)."""
    base = reports_dir / f"{target}_{phase}" / "predictions.parquet"
    if not base.exists():
        return None
    preds = pd.read_parquet(base)
    for extra in sorted(reports_dir.glob(f"{target}_{phase}_*/predictions.parquet")):
        more = pd.read_parquet(extra)
        new = [c for c in more.columns if c.startswith("p_") and c not in preds.columns]
        if new:
            preds = preds.merge(more[KEYS + new], on=KEYS, how="inner")
    return preds


def horizon_plot(tables: dict[str, pd.DataFrame], useful: int, title: str, path: Path) -> None:
    """BSS with 95% CI per lead bin; the forecast zone (lead <= useful) is shaded."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axvspan(0.5, useful + 0.5, color=SLOTS[0], alpha=0.08, linewidth=0)
    ax.text(
        useful + 0.6,
        0.98,
        f" forecast zone: 1-{useful} h",
        color=TEXT_2,
        fontsize=9,
        va="top",
        transform=ax.get_xaxis_transform(),
    )
    ax.axhline(0, color=TEXT_2, linewidth=1)
    for (name, t), color in zip(tables.items(), SLOTS, strict=False):
        x = np.sqrt(t["lead_from"] * t["lead_to"])  # geometric bin centre (log axis)
        ax.errorbar(
            x,
            t["bss"],
            yerr=[t["bss"] - t["ci_low"], t["ci_high"] - t["bss"]],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            capsize=3,
            label=name,
        )
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 6, 12, 24, 48, 96, 168])
    ax.set_xticklabels(["1", "2", "3", "6", "12", "24", "48", "96", "168"])
    _style(ax, title, "lead time, hours (log scale)", "Brier skill score vs climatology")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper right")
    _save(fig, path)


def horizon_section(
    reports_dir: Path, fig_dir: Path, useful: dict[str, int], final: dict[str, str]
) -> str:
    out = [
        "## Useful forecast horizon\n\n"
        "Skill vs climatology per lead time with a 95% bootstrap interval (resampling whole "
        "forecast days). The forecast is shown only where the interval is clearly above zero "
        "in both the backtest and the dev period; beyond that the dashboard shows the typical "
        "(climatological) pattern, explicitly labelled as not a forecast.\n"
    ]
    for target, label_ in (("y_alert", "Alert of any level"), ("y_red", "Red-level alert")):
        main = final[target]
        tables: dict[str, pd.DataFrame] = {}
        for phase, name in (("backtest", "2-year backtest"), ("dev", "dev period, 4 weeks")):
            preds = load_preds(reports_dir, target, phase)
            if preds is None or f"p_{main}" not in preds.columns:
                continue
            t = skill_ci_by_lead(preds, main)
            t.round(4).to_csv(reports_dir / f"useful_horizon_{target}_{phase}.csv", index=False)
            tables[name] = t
        if not tables:
            continue
        horizon_plot(
            tables,
            useful[target],
            f"{label_}: skill by lead time ({main})",
            fig_dir / f"horizon_{target}.png",
        )
        rows = []
        for name, t in tables.items():
            for r in t.itertuples():
                rows.append(
                    {
                        "period": name,
                        "lead, h": f"{r.lead_from}-{r.lead_to}",
                        "bss": r.bss,
                        "95% CI": f"{r.ci_low:+.3f} … {r.ci_high:+.3f}",
                    }
                )
        out.append(
            f"### {label_} (`{target}`, model `{main}`) → forecast horizon "
            f"**{useful[target]} h**\n\n![horizon](figures/horizon_{target}.png)\n\n"
            f"{_md_table(pd.DataFrame(rows))}\n"
        )
    return "\n".join(out)


def backtest_plot(by_fold: pd.DataFrame, models: list[str], path: Path) -> None:
    w = by_fold.pivot_table(index="fold", columns="model", values="bss")
    x = pd.to_datetime(w.index.str.replace("bt-", "") + "-01")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axhline(0, color=TEXT_2, linewidth=1)
    for m, color in zip(models, SLOTS, strict=False):
        ax.plot(x, w[m], color=color, linewidth=2, marker="o", markersize=4, label=label(m))
    _style(ax, "Monthly backtest: skill vs climatology (y_alert)", "", "Brier skill score")
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="lower right")
    _save(fig, path)


def backtest_section(reports_dir: Path, main: str, fig_dir: Path) -> str:
    d = reports_dir / "y_alert_backtest"
    if not (d / "metrics_by_lead.csv").exists():
        return ""
    by_lead = pd.read_csv(d / "metrics_by_lead.csv")
    by_fold = pd.read_csv(d / "metrics_by_fold.csv")
    overall = pd.read_csv(d / "metrics_overall.csv")
    for extra in sorted(reports_dir.glob("y_alert_backtest_*")):
        for name, frame in (("by_lead", by_lead), ("by_fold", by_fold), ("overall", overall)):
            more = pd.read_csv(extra / f"metrics_{name}.csv")
            more = more[~more["model"].isin(set(frame["model"]))]  # first run wins
            merged = pd.concat([frame, more], ignore_index=True)
            if name == "by_lead":
                by_lead = merged
            elif name == "by_fold":
                by_fold = merged
            else:
                overall = merged
    compare = [m for m in (main, "gbm_resid_season") if m in set(by_fold["model"])]
    backtest_plot(by_fold, compare, fig_dir / "backtest_y_alert.png")
    w = by_fold.pivot_table(index="fold", columns="model", values="bss")
    wins = {m: int((w[m] > 0).sum()) for m in w.columns if m != "climatology"}
    cols = ["model", "base_rate", "brier", "bss", "roc_auc", "pr_auc"]
    lead = by_lead.pivot_table(index="model", columns="lead_bucket", values="bss")
    lead = lead[[c for c in ("1-6h", "7-24h", "1-3d", "4-7d") if c in lead.columns]]
    lead = lead.reset_index()
    return (
        "## Two-year monthly backtest (`y_alert`)\n\n"
        "Monthly refits from 2024-09 to 2026-08 (24 folds, two winters, all before the "
        "2026-09-06 reform). Used to decide on seasonal features, which a 4-week dev period "
        "cannot judge.\n\n"
        f"### Overall\n\n{_md_table(overall[cols])}\n\n"
        f"### Brier skill score by lead time\n\n{_md_table(lead)}\n\n"
        "Months with skill > 0: "
        + ", ".join(f"`{m}` {n}/{len(w)}" for m, n in wins.items())
        + "\n\n![backtest](figures/backtest_y_alert.png)\n\n"
        "**Decision.** Seasonal features (month, day of year, last-year analog, solar "
        "elevation) make the model worse: better than the model without them in only 5 of 24 "
        "months, and worst in winter. Darkness alone is neutral (8/24). The data show no "
        "stable annual cycle of alert hours, so the final model uses none of them.\n"
    )


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    reports_dir: Path,
    panel: pd.DataFrame,
    regime_break: pd.Timestamp,
    meta: dict[str, object],
    useful: dict[str, int],
) -> Path:
    """The final model of each target (forecast.FINAL) is highlighted against baselines."""
    from uaalerts.forecast import FINAL

    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    regime_plot(panel, "kyiv_city", regime_break, fig_dir / "regime_kyiv.png")
    sections = []
    for target, label in (("y_alert", "Alert of any level"), ("y_red", "Red-level alert")):
        d = reports_dir / f"{target}_dev"
        loaded = load_preds(reports_dir, target, "dev")
        if loaded is None:
            continue
        preds = loaded
        info = json.loads((d / "info.json").read_text())
        models = [c[2:] for c in preds.columns if c.startswith("p_")]
        main = FINAL[target]
        if main not in models:
            raise ValueError(f"{target}: model {main!r} not in predictions ({models})")
        others = [m for m in ("gbm_resid", "gbm_v1", "persistence") if m in models and m != main]
        shown: list[str] = [main, *others][:3]
        skill_by_lead(
            preds,
            shown,
            f"{label}: skill by lead time (dev period, 4 weeks)",
            fig_dir / f"skill_{target}.png",
        )
        reliability_plot(
            preds, main, f"{label}: reliability", fig_dir / f"reliability_{target}.png"
        )
        example_plot(
            preds, main, "kyiv_city", f"Kyiv, {label.lower()}", fig_dir / f"example_{target}.png"
        )
        preds["lead_bucket"] = lead_bucket(preds["lead"])
        cols = ["model", "brier", "bss", "log_loss", "roc_auc", "pr_auc"]
        overall = score_table(preds, models)[["base_rate", *cols]]
        by_lead = score_table(preds, models, by=["lead_bucket"])
        by_lead = by_lead[by_lead["model"].isin(["climatology", "persistence", *shown])]
        by_fold = score_table(preds, ["climatology", main], by=["fold"])
        by_region = score_table(preds, [main], by=["region"]).sort_values("bss")
        sections.append(
            f"## {label} (`{target}`)\n\n"
            f"Evaluation origins: {preds['origin'].min():%Y-%m-%d %H:%M} → "
            f"{preds['origin'].max():%Y-%m-%d %H:%M} UTC · {len(preds):,} scored "
            f"(origin, region, lead) rows · excluded regions: {', '.join(info['excluded'])}\n\n"
            f"### Overall\n\n{_md_table(overall)}\n\n"
            f"### By lead time\n\n"
            f"{_md_table(by_lead[['lead_bucket', 'base_rate', *cols]])}\n\n"
            f"![skill](figures/skill_{target}.png)\n\n"
            f"### By test week (model refit before each week)\n\n"
            f"{_md_table(by_fold[['fold', 'base_rate', *cols]])}\n\n"
            f"### By region (main model: `{main}`)\n\n"
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
    text += "\n" + horizon_section(reports_dir, fig_dir, useful, FINAL)
    text += "\n" + backtest_section(reports_dir, FINAL["y_alert"], fig_dir)
    text += (
        "\n## Limitations\n\n"
        "- Real skill is concentrated in the first hours (see *Useful forecast horizon*); the "
        "operational forecast is therefore limited to 6 h (any level) and 2 h (red). The "
        "original 7-day horizon from the project plan was cut after measuring it.\n"
        "- Trend features over-react to bursts: after the spikes of 24.09 and 30.09 the model "
        "expected escalation, but 01-03.10 were calm (dev week 3).\n"
        "- The red-level target has only weeks of history.\n"
        "- The target counts hours under alert, not the scale or damage of an attack.\n"
        "- Fresh data come from a daily-synced mirror, so the operational forecast can lag by "
        "up to a day.\n"
    )
    path = reports_dir / "REPORT.md"
    path.write_text(text)
    return path
