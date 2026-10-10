# ua-air-alerts-forecast

Mini pet-project: time series analysis & forecasting of air raid alerts in Ukraine.

> 🚧 Work in progress. Architecture and stack are being defined (see `CLAUDE.md`).

## Goal
- Forecast air raid alerts (per region) using the most recent available data.
- Evaluate on the latest period with a strict **no data leakage** policy:
  nothing used for training or as model input may be published after the forecast time.
- Optional: enrich with text/media signals (news, official statements, public Telegram channels),
  respecting the same publication-time cutoff.
