# ua-air-alerts-forecast

Mini pet-project: time series analysis & forecasting of air raid alerts in Ukraine.

> 🚧 Work in progress. Architecture and stack are being defined (see `CLAUDE.md`).

## Goal
- Forecast air raid alerts (per region) using the most recent available data.
- Evaluate on the latest period with a strict **no data leakage** policy:
  nothing used for training or as model input may be published after the forecast time.
- Optional: enrich with text/media signals (news, official statements, public Telegram channels),
  respecting the same publication-time cutoff.

## Quick start
Requires [uv](https://docs.astral.sh/uv/) (on macOS LightGBM also needs `brew install libomp`).

```bash
uv sync                 # create .venv with pinned deps
uv run uaalerts info    # show active forecasting config
uv run pytest           # tests (incl. leakage checks)
uv run pre-commit install
```

Docs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
