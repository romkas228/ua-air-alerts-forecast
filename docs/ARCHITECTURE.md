# Architecture — ua-air-alerts-forecast

Status: **draft for review** (stage 2). Decisions log lives in `CLAUDE.md`.

## 1. Problem statement

At a forecast origin `t` (every 6 h: 00/06/12/18 Europe/Kyiv) predict, for every oblast `r`
and every lead `h ∈ 1..168`, the probability that the hour `[t+h-1, t+h)` contains an alert:

- **`y_any`** (primary): any active alert in oblast `r` — oblast-wide **or** in at least one
  raion/hromada of `r`, any level (yellow or red).
- **`y_red`** (experimental): a red-level alert in oblast `r`. Only exists since 2026-09-06.

Excluded: regions with a de-facto constant alert / no data (Luhansk, Crimea, …),
detected automatically from training data only.

## 2. Data sources

| Source | Period used | Role |
|---|---|---|
| Vadimkin `official_data_en.csv` | 2022-03-15 → 2026-09-07 | history |
| JustJimer `archive/alerts_in_ua.csv` (alerts.in.ua mirror) | 2026-09-04 → now | fresh data, yellow/red levels |
| siren.pp.ua `/api/v3/alerts` | live | dashboard "now" state + own logger |
| Telegram public channels (stage 3, later) | `published_at < t` | text features |

Raw files are **not committed** (licensing); `uaalerts fetch` downloads dated snapshots into `data/raw/`.
The 2026-09-04 → 09-07 overlap is used to check that both sources agree before stitching.

## 3. Pipeline

```
fetch ─► raw snapshots (csv/json, dated)
  │
normalize ─► events.parquet      one row per alert:
  │          oblast, raion, hromada, level_geo, alert_level, started_at, finished_at, source
  │
panel ─► hourly.parquet          oblast × hour (UTC): y_any, y_red, active_minutes,
  │                              n_raions_active, red_minutes, …
  │
features(t) ─► origin snapshot   state of the world AS OF t (see §4)
  │
dataset ─► rows (t, oblast, lead) = origin features ⊕ target-hour calendar ⊕ label
  │
train / evaluate / forecast / report / dashboard
```

## 4. Leakage policy (the core of the project)

1. **`as_of(events, t)`** is the only gateway from events to features. It keeps events with
   `started_at < t` and sets `finished_at = NaT` (still ongoing) when `finished_at ≥ t`.
   Features never read the raw event table directly.
2. Origin features use history `< t` only. Target-hour features are calendar-only
   (hour of day, day of week, lead), which is known in advance.
3. Climatology/baselines and region exclusion are computed from the training window only.
4. Splits are strictly chronological, with a 168 h **embargo** between train and val/test,
   so that no training label overlaps the evaluation period.
5. Scalers, calibrators and encoders are fitted on the training (or validation) part only.
6. Text features are filtered by `published_at < t` (`t` is the origin, not the target hour).
7. **Automated tests:**
   - *truncation invariance:* features computed on the full history equal features computed
     on history truncated at `t`;
   - split ordering and embargo assertions;
   - label rows never feed origin features.

## 5. Features (v1, alerts only)

- **Current state at `t`:** alert active now, minutes since the last start/end, current duration,
  number/share of raions active, red active (post-reform).
- **Recent intensity:** alert hours over 6 h / 24 h / 72 h / 7 d / 28 d; number of alert starts;
  same-hour-of-day frequency over the last 28 d.
- **Spatial:** the same aggregates for neighbouring oblasts (adjacency table) and nationwide share active.
- **Target hour:** lead `h`, hour-of-day (Kyiv), day of week, the 28-day climatology for that hour.
- **Regime:** post-2026-09-06 flag; oblast id as a categorical feature.

## 6. Models

| Model | Purpose |
|---|---|
| Persistence (`p = active at t`, decays to climatology) | naive baseline |
| Seasonal climatology (oblast × hour × dow, trailing 28 d) | strong baseline |
| **LightGBM**, one global model for all oblasts × leads | main model |
| Isotonic calibration on validation | calibrated probabilities |

Size: about 1,460 origins/yr × 24 oblasts × 168 leads ≈ 5.9M rows/yr. If needed, training
subsamples leads (all leads ≤ 24, every 3rd lead beyond 24), while evaluation always uses all leads.

## 7. Evaluation

```
──── train ────┤embargo├── val (4 w) ──┤embargo├── test = last 4 weeks ──► now
```

- Val is used for hyperparameters and calibration. After that the model is refit on train+val
  and evaluated once on test.
- **Metrics:** Brier score, Brier skill score vs climatology, log loss, ROC-AUC, PR-AUC,
  reliability diagram.
- **Breakdown:** by lead bucket (1–6 h, 7–24 h, 1–3 d, 4–7 d), by oblast, and by time of day.
- Test origins are limited so that the full 168 h window ends before the last observed hour
  (or partial horizons are scored and flagged).
- ⚠️ The test period (≈ 2026-09-12 → 10-10) lies fully after the regime break and is fully
  covered by the new source. Val straddles both, so this has to be reported.
- ⚠️ `y_red` has only ~6 days of history before test, so it gets a separate, shorter walk-forward
  scheme (TBD).

## 8. Tech stack

Python 3.12 · uv · pandas + pyarrow · LightGBM · scikit-learn · Typer (CLI) ·
pydantic-settings + YAML · Plotly · Streamlit · pytest · ruff · pre-commit · GitHub Actions.
Stage 3 (later): Telethon / web preview scraping + Claude API for structured extraction from text.

## 9. Repository layout

```
src/uaalerts/
  config.py            settings, paths, constants (origins, leads, buckets)
  data/sources/        vadimkin.py, alerts_in_ua_mirror.py, siren.py
  data/normalize.py    unified events schema, region name mapping
  data/panel.py        hourly oblast panel + targets
  features/asof.py     the leakage gateway
  features/build.py    origin features, dataset assembly
  models/baselines.py  persistence, climatology
  models/gbm.py        LightGBM + calibration
  evaluation/splits.py chronological splits + embargo
  evaluation/metrics.py
  report/              HTML/markdown report generation
  cli.py               Typer entrypoint `uaalerts`
app/dashboard.py       Streamlit
configs/default.yaml
tests/                 incl. test_leakage.py
data/ (gitignored)     raw/ interim/ processed/
```

## 10. Roadmap (stage 3)

| # | Milestone | Done when |
|---|---|---|
| M1 | Project scaffold (uv, ruff, pytest, CI) | `uv run pytest` is green in CI |
| M2 | Fetch + normalize + source-agreement check | `events.parquet` + overlap report |
| M3 | Hourly panel + targets + EDA notebook/report | plots of seasonality and the regime break |
| M4 | `as_of`, features, splits + leakage tests | truncation-invariance tests pass |
| M5 | Baselines + evaluation harness | metrics table by lead bucket |
| M6 | LightGBM + calibration | beats climatology on val/test (or an honest report) |
| M7 | Report + Streamlit dashboard + siren logger | map + 7-day forecast from "now" |
| M8 | Text features (Telegram) | ablation: with vs. without text |
