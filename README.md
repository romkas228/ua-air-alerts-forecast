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

## See it working

```bash
uv run uaalerts update --retrain      # fresh data -> panel -> models -> 7-day forecast (~2 min)
uv run streamlit run app/dashboard.py # opens http://localhost:8501
```

The dashboard shows the hourly alert probability for the next 7 days per region (line chart +
heatmap), a ranking of regions, the live alert state (siren.pp.ua, display only) and the model
quality on the dev period. Evaluation report: [`reports/REPORT.md`](reports/REPORT.md).

Pipeline commands: `fetch`, `build`, `evaluate --target y_alert --phase dev`, `report`,
`train`, `forecast` (`uv run uaalerts --help`).

Docs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
