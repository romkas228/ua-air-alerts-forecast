# ua-air-alerts-forecast

Short-range forecasting of air raid alerts in Ukraine, hourly and per region. The pipeline is
built to be free of data leakage and is evaluated on the most recent data.

> **In one sentence:** for each region and each of the next hours, the model gives the
> probability of an alert covering the oblast centre. It is measurably better than the
> "typical schedule" baseline for the next **6 h**: Brier skill ≈ +0.53 at 1 h, +0.14 at 2 h
> and +0.03…0.07 at 3–6 h on a 2-year backtest. The same skill held on data published
> *after* the model was frozen.

| | |
|---|---|
| Evaluation report (numbers, figures, limitations) | [`reports/REPORT.md`](reports/REPORT.md) |
| Architecture & methodology | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Decisions log (every decision, with the reason and the data behind it) | [`CLAUDE.md`](CLAUDE.md) |

## Quick start

Requires [uv](https://docs.astral.sh/uv/). On macOS LightGBM also needs `brew install libomp`.

```bash
uv sync
uv run uaalerts update --retrain       # download data -> build -> train -> forecast (~3 min)
uv run streamlit run app/dashboard.py  # http://localhost:8501
```

The dashboard has three tabs:

- **Forecast for 6 h:** a heatmap of all regions and a chart for the selected region, with
  live alert state from siren.pp.ua. The live state is only displayed; it is never used as a
  model input.
- **Typical week:** the climatological pattern for 7 days, labelled as *not* a forecast.
- **Model quality:** skill by lead time with confidence intervals.

Other commands (`uv run uaalerts --help`):

| Command | What it does |
|---|---|
| `fetch` | Download dated raw snapshots. Old snapshots are never overwritten. |
| `build` | Normalise and stitch the sources, then build the hourly region panel. |
| `evaluate --target y_alert --phase dev\|backtest\|prospective --models …` | Walk-forward evaluation. Metrics go to `reports/`. |
| `train` / `forecast` | Fit the final models and produce the operational forecast. |
| `report` | Regenerate `reports/REPORT.md` and its figures. |

Run the checks with `uv run pytest`. The same checks run in CI, together with ruff and mypy
`--strict`.

**Reproducing the report.** The committed [`reports/REPORT.md`](reports/REPORT.md) is the
reference. Its per-row predictions are large and not committed. To regenerate everything,
run `scripts/reproduce_evaluation.sh` (~1.5–2 h, mostly the 2-year backtests). Running
`uaalerts report` alone on a fresh clone gives a shortened report.

## What is forecast

- **Unit:** one of 24 regions × one hour. Crimea and Luhanska are occupied and kept out of
  the metrics.
- **Target `y_alert`:** an air-raid alert of any level covering the **oblast centre** during
  that hour.
  - Why the centre: since autumn 2025 alerts are declared per raion. Chronically alerted
    frontline raions (Nikopol is under alert 94% of the time) made "any alert in the oblast"
    almost always 1, so it carried no information.
- **Target `y_red`:** the same, red level only. Red and yellow levels exist only since the
  2026-09-06 reform.
- **Forecast origins:** every 6 h (00/06/12/18 Kyiv time).
- **Forecast horizon:** **6 h** for any level, **2 h** for red.
  - We first built forecasts up to 7 days and then *measured* how far ahead they are useful.
    Beyond about 6 h the model is statistically indistinguishable from the 28-day
    hour-of-day climatology.
  - The dashboard and the report therefore show the rest of the week as a "typical pattern",
    not as a forecast.

## Data

| Source | Period | Notes |
|---|---|---|
| [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) | 2022-03 → 2026-09-07 | Official alerts. Frozen since its upstream Telegram channel went silent. |
| [JustJimer/ukraine-air-alert-stats](https://github.com/JustJimer/ukraine-air-alert-stats) (alerts.in.ua mirror) | 2026-09-04 → now | Raion/hromada level, yellow/red levels, synced daily. |
| [siren.pp.ua](https://siren.pp.ua) | live | Shown on the dashboard only. |

- The two sources are stitched at 2026-09-06 12:00 UTC. In the overlap they agree on 99% of
  hourly values.
- Raw data are **not committed**. JustJimer has no licence, so `uaalerts fetch` downloads it.
- "End of data" is the newest timestamp *inside* the data, not the download time. The mirror
  syncs once a day.

## No data leakage

This is the core requirement of the task, and every point below is enforced by code and
tests:

- **One gateway for features.** All features at an origin `t` are computed only from panel
  hours `< t`. This is equivalent to building the panel from `as_of(events, t)`, which drops
  the end time of alerts still running at `t`.
- **Truncation-invariance test.** `tests/test_leakage.py` asserts that features computed
  from the full history are identical (up to float tolerance) to features computed from history truncated at
  `t`. It covers every feature, including ones added later.
- **Walk-forward with purging.** A model used from time `T` is trained only on rows whose
  target hour had already ended by `T` (`origin + lead ≤ T`). The runner asserts this for
  every fold.
- **Fitted parts use the past only.** Climatology, region exclusions and recency features
  use past data only, and exclusions are decided per fold.
- **Model selection never touched the final test.**
  - The first test result was poor and was reported as is. That period then became the
    *dev* period.
  - Models were chosen on dev plus a 2-year monthly backtest, then frozen (git tag
    `freeze-v2`).
  - Finally they were evaluated **prospectively**, on data that did not exist during
    development.

## Results (details in [`reports/REPORT.md`](reports/REPORT.md))

Brier skill score vs the 28-day hour-of-day climatology. Positive means better than
climatology.

| `y_alert`, final model `gbm_resid_short` | 1 h | 2 h | 3 h | 4–6 h |
|---|---|---|---|---|
| 2-year monthly backtest (24 refits) | **+0.53** | **+0.14** | +0.07 | +0.03…0.05 |
| Dev period (4 weeks before 2026-10-10) | +0.48 | +0.15 | +0.08 | +0.06…0.11 |
| **Prospective** (after the freeze, first days) | +0.23 | +0.13 | +0.10 | +0.08 |

How we got there:

1. **Baseline GBM.** The first LightGBM, trained mostly on the pre-reform regime, lost to
   climatology on the test period (BSS −0.05).
2. **Trend features and residual model.** Adding trend features and boosting from
   climatology (LightGBM learns only the *deviation* from the typical schedule) gave a model
   that is ≥ 0 at every lead.
3. **Seasonal features.** Month, day of year, last-year analog and solar elevation were
   tested on the 2-year backtest and **rejected**. They were worse in 19 of 24 months. The
   data show no stable annual cycle of alert hours (checked explicitly, including
   "are alerts more frequent in winter?" — they are not).
4. **Short horizon.** Training only on leads ≤ 12 h improved the 1–6 h forecast in 21 of 24
   backtest months.
5. **Red alerts.** `y_red` has only weeks of history, so a GBM cannot learn it. Persistence
   decaying to climatology is used instead, and the first prospective days suggest a 1 h
   horizon.

## Text / media data (optional part of the task)

This part is **not implemented** because of the deadline. The design and the groundwork are
done:

- **Source:** the official Air Force channel `@kpszsu` (drone movements per oblast with
  heading).
- **Access:** verified without an account.
- **Leakage rules:** `published_at < origin`, and edited messages are dropped.
- **Planned features** and the evaluation plan are described in the report section
  *Text data*.

## How AI was used

The project was built with **Claude Code** as the primary engineering tool:

- choosing data sources, including discovering that the official dataset had stopped
  updating and finding a fresh mirror;
- designing the leakage-safe evaluation;
- writing all code and tests;
- running experiments;
- writing the report.

The human made every decision about scope, target and evaluation. Claude asked before each
one and recorded it in [`CLAUDE.md`](CLAUDE.md), so the full reasoning trail, including
abandoned ideas and negative results, is reviewable.

## Project layout

```
src/uaalerts/
  data/        sources, stitching, hourly panel, regions (centres, adjacency), live state
  features/    as_of gateway, origin + target-hour features, solar elevation
  models/      climatology, persistence, LightGBM (residual over climatology)
  evaluation/  walk-forward splits with purging, metrics, bootstrap CIs
  report/      REPORT.md + figures
  forecast.py  final models, operational forecast
  cli.py       `uaalerts` command
app/dashboard.py   Streamlit
configs/default.yaml
tests/             incl. test_leakage.py
```
