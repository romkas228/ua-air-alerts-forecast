"""Command-line entrypoint: `uaalerts <command>`."""

from __future__ import annotations

import typer

from uaalerts import __version__
from uaalerts.config import get_config, get_settings

app = typer.Typer(help="Air raid alert forecasting pipeline.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Air raid alert forecasting pipeline."""


@app.command()
def info() -> None:
    """Show version, paths and the active forecasting configuration."""
    settings = get_settings()
    cfg = get_config()
    typer.echo(f"uaalerts {__version__}")
    typer.echo(f"data dir:   {settings.data_dir}")
    typer.echo(f"origins:    {cfg.origin_hours} ({cfg.timezone})")
    typer.echo(f"max lead:   {cfg.max_lead_hours} h")
    typer.echo(f"test/val:   {cfg.test_weeks}w / {cfg.val_weeks}w, embargo {cfg.embargo_hours} h")


if __name__ == "__main__":
    app()
