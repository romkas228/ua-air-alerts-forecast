#!/usr/bin/env bash
# Re-run every evaluation behind reports/REPORT.md, then rebuild the report.
# Takes ~1.5-2 h on a laptop (the 2-year backtests dominate). Predictions are written to
# reports/*/predictions.parquet (not committed); metric CSVs are committed.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run uaalerts fetch
uv run uaalerts build

E="uv run uaalerts -v evaluate"
# Dev period (4 weeks before 2026-10-10 09:00 UTC) - model selection.
$E --target y_alert --phase dev \
   --models climatology,persistence,gbm_v1,gbm_trend,gbm_resid,gbm_trend_season,gbm_resid_season
$E --target y_alert --phase dev --models climatology,gbm_resid,gbm_resid_short --tag short
$E --target y_red --phase dev --models climatology,persistence,gbm_v1,gbm_resid
# 2-year monthly backtest - seasonal features and the short-horizon model.
$E --target y_alert --phase backtest --models climatology,persistence,gbm_resid,gbm_resid_season
$E --target y_alert --phase backtest --models climatology,gbm_resid_dark --tag dark
$E --target y_alert --phase backtest --models climatology,gbm_resid,gbm_resid_short --tag short
# Prospective - frozen models on data after the dev end (grows as new data arrive).
$E --target y_alert --phase prospective --models climatology,persistence,gbm_resid_short
$E --target y_red --phase prospective --models climatology,persistence

uv run uaalerts report
