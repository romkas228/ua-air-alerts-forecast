"""Command-line entrypoint: `uaalerts <command>`."""

from __future__ import annotations

import json
import logging

import pandas as pd
import typer

from uaalerts import __version__
from uaalerts.config import get_config, get_settings
from uaalerts.data import sources
from uaalerts.data.panel import build_panel
from uaalerts.data.stitch import STITCH_AT, overlap_agreement, stitch
from uaalerts.evaluation.metrics import lead_bucket, score_table
from uaalerts.evaluation.walkforward import Spec, run
from uaalerts.features.build import SEASON_FEATURES, TARGETS, TREND_FEATURES
from uaalerts.models.baselines import Climatology, Persistence
from uaalerts.models.gbm import GBM

SPECS: dict[str, Spec] = {
    "climatology": Spec("climatology", lambda t: Climatology()),
    "persistence": Spec("persistence", lambda t: Persistence(f"{TARGETS[t]}_active_now")),
    "gbm": Spec("gbm", lambda t: GBM()),
    "gbm_365d": Spec("gbm_365d", lambda t: GBM(), window_days=365),
    "gbm_hl90": Spec("gbm_hl90", lambda t: GBM(half_life_days=90)),
    # Improvement round 2026-10-12 (feature-group ablations, residual-over-climatology).
    "gbm_v1": Spec("gbm_v1", lambda t: GBM(drop=TREND_FEATURES | SEASON_FEATURES)),
    "gbm_trend": Spec("gbm_trend", lambda t: GBM(drop=SEASON_FEATURES)),
    "gbm_resid": Spec("gbm_resid", lambda t: GBM(base="climatology", drop=SEASON_FEATURES)),
    "gbm_trend_season": Spec("gbm_trend_season", lambda t: GBM()),
    "gbm_resid_season": Spec("gbm_resid_season", lambda t: GBM(base="climatology")),
}

app = typer.Typer(help="Air raid alert forecasting pipeline.", no_args_is_help=True)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Air raid alert forecasting pipeline."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


@app.command()
def info() -> None:
    """Show version, paths and the active forecasting configuration."""
    settings = get_settings()
    cfg = get_config()
    typer.echo(f"uaalerts {__version__}")
    typer.echo(f"data dir:   {settings.data_dir}")
    typer.echo(f"origins:    {cfg.origin_hours} ({cfg.timezone})")
    typer.echo(f"max lead:   {cfg.max_lead_hours} h")
    typer.echo(f"walk-fwd:   test {cfg.test_weeks}w, tune {cfg.tune_weeks}w (weekly refits)")


@app.command()
def fetch(
    historical: bool = typer.Option(
        False, help="Also re-download the (frozen since 2026-09-07) Vadimkin dataset."
    ),
) -> None:
    """Download fresh raw snapshots into data/raw (never overwrites old ones)."""
    settings, cfg = get_settings(), get_config()
    path = sources.download_snapshot(
        cfg.sources.alerts_in_ua_mirror_url, settings.raw_dir, sources.ALERTS_IN_UA
    )
    typer.echo(f"saved {path}")
    if historical or not sources.list_snapshots(settings.raw_dir, sources.VADIMKIN):
        path = sources.download_snapshot(
            cfg.sources.vadimkin_official_url, settings.raw_dir, sources.VADIMKIN
        )
        typer.echo(f"saved {path}")


@app.command()
def build() -> None:
    """Normalize + stitch sources into events.parquet, then build the hourly panel."""
    settings, cfg = get_settings(), get_config()
    regime_break = pd.Timestamp(cfg.regime_break).tz_convert("UTC")

    hist_paths = sources.list_snapshots(settings.raw_dir, sources.VADIMKIN)
    fresh_paths = sources.list_snapshots(settings.raw_dir, sources.ALERTS_IN_UA)
    if not hist_paths or not fresh_paths:
        raise typer.BadParameter("no raw snapshots, run `uaalerts fetch` first")
    hist = sources.parse_vadimkin(hist_paths[-1])
    fresh = sources.parse_alerts_in_ua(fresh_paths)
    # The data are only known up to the newest snapshot: nothing after it may be used.
    data_end = sources.observed_until(fresh)

    agreement = overlap_agreement(hist, fresh, regime_break=regime_break)
    total = (agreement["agree"] * agreement["hours"]).sum() / agreement["hours"].sum()
    typer.echo(f"source overlap agreement (hourly y_alert): {total:.1%}")

    events = stitch(hist, fresh)
    panel = build_panel(
        events,
        start=events["started_at"].min(),
        end=data_end,
        regime_break=regime_break,
    )

    settings.interim_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(settings.interim_dir / "events.parquet", index=False)
    panel.to_parquet(settings.processed_dir / "panel.parquet", index=False)
    agreement.to_csv(settings.interim_dir / "source_overlap.csv", index=False)
    panel_hours = [str(panel["hour"].min()), str(panel["hour"].max())]
    meta = {
        "data_end": data_end.isoformat(),
        "stitch_at": STITCH_AT.isoformat(),
        "historical_snapshot": hist_paths[-1].name,
        "fresh_snapshots": [p.name for p in fresh_paths],
        "n_events": len(events),
        "panel_hours": panel_hours,
        "overlap_agreement": round(float(total), 4),
    }
    (settings.interim_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    typer.echo(f"events: {len(events):,} -> {settings.interim_dir / 'events.parquet'}")
    typer.echo(f"panel:  {len(panel):,} rows, {panel_hours[0]} .. {panel_hours[1]}")


@app.command()
def evaluate(
    target: str = typer.Option("y_alert", help="y_alert | y_red"),
    phase: str = typer.Option("dev", help="tune | dev | backtest | prospective"),
    models: str = typer.Option("climatology,persistence,gbm", help="comma-separated"),
) -> None:
    """Weekly walk-forward evaluation; writes predictions and metric tables to reports/."""
    settings, cfg = get_settings(), get_config()
    names = [m.strip() for m in models.split(",") if m.strip()]
    unknown = sorted(set(names) - set(SPECS))
    if unknown:
        raise typer.BadParameter(f"unknown models {unknown}; choose from {sorted(SPECS)}")
    if "climatology" not in names:
        names.insert(0, "climatology")  # reference for the Brier skill score
    panel = pd.read_parquet(settings.processed_dir / "panel.parquet")
    preds, info = run(panel, cfg, target, [SPECS[n] for n in names], phase=phase)

    out_dir = settings.data_dir.parent / "reports" / f"{target}_{phase}"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    preds["lead_bucket"] = lead_bucket(preds["lead"])
    tables = {
        "overall": score_table(preds, names),
        "by_lead": score_table(preds, names, by=["lead_bucket"]),
        "by_fold": score_table(preds, names, by=["fold"]),
        "by_region": score_table(preds, names, by=["region"]),
    }
    for name, table in tables.items():
        table.round(4).to_csv(out_dir / f"metrics_{name}.csv", index=False)
    (out_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    typer.echo(f"excluded in every fold: {', '.join(info['excluded'])}")
    cols = ["lead_bucket", "model", "base_rate", "brier", "bss", "roc_auc", "pr_auc"]
    typer.echo(tables["by_lead"][cols].round(3).to_string(index=False))
    typer.echo(tables["overall"][cols[1:]].round(3).to_string(index=False))
    typer.echo(f"-> {out_dir}")


@app.command()
def report(main: str = typer.Option(..., help="selected model, e.g. gbm_resid")) -> None:
    """Build reports/REPORT.md and figures from the latest test-phase evaluations."""
    from uaalerts.report.build import build_report

    settings, cfg = get_settings(), get_config()
    panel = pd.read_parquet(settings.processed_dir / "panel.parquet")
    meta = json.loads((settings.interim_dir / "meta.json").read_text())
    regime_break = pd.Timestamp(cfg.regime_break).tz_convert("UTC")
    path = build_report(settings.data_dir.parent / "reports", panel, regime_break, meta, main)
    typer.echo(f"-> {path}")


@app.command()
def train(target: str = typer.Option("all", help="y_alert | y_red | all")) -> None:
    """Fit the final GBM on everything known at the end of the data (models/)."""
    from uaalerts.forecast import train_final

    settings, cfg = get_settings(), get_config()
    panel = pd.read_parquet(settings.processed_dir / "panel.parquet")
    for t in TARGETS if target == "all" else [target]:
        path = train_final(panel, cfg, t, settings.data_dir.parent / "models")
        typer.echo(f"{t}: -> {path}")


@app.command()
def forecast(target: str = typer.Option("all", help="y_alert | y_red | all")) -> None:
    """7-day hourly forecast for every region from the latest origin."""
    from uaalerts.forecast import make_forecast
    from uaalerts.models.gbm import GBM

    settings, cfg = get_settings(), get_config()
    panel = pd.read_parquet(settings.processed_dir / "panel.parquet")
    model_dir = settings.data_dir.parent / "models"
    frames = []
    for t in TARGETS if target == "all" else [target]:
        path = model_dir / f"{t}_gbm.txt"
        if not path.exists():
            raise typer.BadParameter(f"no model at {path}; run `uaalerts train` first")
        frames.append(make_forecast(panel, cfg, t, GBM.load(path)))
    fc = pd.concat(frames, ignore_index=True)
    out = settings.processed_dir / "forecast.parquet"
    fc.to_parquet(out, index=False)
    origin = fc["origin"].iloc[0]
    typer.echo(f"origin {origin} UTC, {len(fc):,} rows -> {out}")
    top = (
        fc[(fc["target"] == "y_alert") & (fc["lead"] <= 24) & ~fc["excluded"]]
        .groupby("region", observed=True)["p_gbm"]
        .max()
        .sort_values(ascending=False)
        .head(8)
    )
    typer.echo("highest alert probability in the next 24 h:")
    typer.echo(top.round(2).to_string())


@app.command()
def update(retrain: bool = typer.Option(False, help="Also refit the final models.")) -> None:
    """fetch -> build -> (train) -> forecast: refresh everything the dashboard shows."""
    fetch(historical=False)
    build()
    if retrain:
        train(target="all")
    forecast(target="all")


if __name__ == "__main__":
    app()
