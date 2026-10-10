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


if __name__ == "__main__":
    app()
