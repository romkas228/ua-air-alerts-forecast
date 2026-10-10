"""Streamlit dashboard: 7-day hourly air-alert forecast per region + live state.

Run:  uv run streamlit run app/dashboard.py
Data: `uaalerts fetch && uaalerts build && uaalerts train && uaalerts forecast`.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from uaalerts.config import get_config, get_settings
from uaalerts.data.live import current_state
from uaalerts.data.regions import REGIONS

st.set_page_config(page_title="Air alert forecast", page_icon="🛡️", layout="wide")

SERIES = {"p_gbm": ("#2a78d6", "LightGBM"), "p_climatology": ("#eb6834", "Climatology (28 d)")}
# Sequential blue ramp (dataviz reference palette), light -> dark.
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NAMES = {r.id: r.name_uk.replace(" область", " обл.") for r in REGIONS}
TARGETS = {"y_alert": "Тривога будь-якого рівня", "y_red": "Червона тривога"}
TZ = "Europe/Kyiv"

settings, cfg = get_settings(), get_config()
root = settings.data_dir.parent


@st.cache_data(ttl=300)
def load_forecast() -> pd.DataFrame:
    fc = pd.read_parquet(settings.processed_dir / "forecast.parquet")
    fc["region"] = fc["region"].astype(str)
    fc["name"] = fc["region"].map(NAMES)
    fc["local"] = pd.to_datetime(fc["target_hour"], utc=True).dt.tz_convert(TZ)
    return fc


@st.cache_data(ttl=60)
def load_live() -> pd.DataFrame | None:
    try:
        return current_state(cfg.sources.siren_api_url)
    except Exception:  # live feed is optional; the dashboard works without it
        return None


def load_json(path: str) -> dict[str, object]:
    p = root / path
    return json.loads(p.read_text()) if p.exists() else {}


if not (settings.processed_dir / "forecast.parquet").exists():
    st.error("Прогнозу ще немає. Запустіть: `uv run uaalerts fetch build train forecast`.")
    st.stop()

fc = load_forecast()
meta = load_json("data/interim/meta.json")
model_meta = load_json("models/y_alert_gbm.json")
origin = pd.Timestamp(fc["origin"].iloc[0]).tz_convert(TZ)
data_end = pd.Timestamp(str(meta.get("data_end"))).tz_convert(TZ)

st.title("Прогноз повітряних тривог · 7 днів, погодинно")
st.caption(
    f"Прогноз від **{origin:%d.%m %H:%M}** (Київ) · дані до **{data_end:%d.%m %H:%M}** · "
    f"модель `{model_meta.get('git_commit', '?')}`. Ціль — тривога, що покриває обласний "
    "центр. Модель корисна насамперед на горизонті 1-6 год; далі вона близька до "
    "кліматології (див. вкладку «Якість»)."
)

# --- controls (one row) -------------------------------------------------------------
c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
target = c1.radio("Ціль", list(TARGETS), format_func=TARGETS.get, horizontal=True)
model = c2.radio(
    "Модель", ["p_gbm", "p_climatology"], format_func=lambda m: SERIES[m][1], horizontal=True
)
regions = sorted(fc["region"].unique(), key=lambda r: NAMES[r])
region = c3.selectbox("Область", regions, index=regions.index("kyiv_city"), format_func=NAMES.get)

f = fc[fc["target"] == target]
live = load_live()

tab_now, tab_map, tab_quality = st.tabs(["Огляд", "Теплова карта 7 днів", "Якість моделі"])

with tab_now:
    # --- live state tiles ---------------------------------------------------------------
    if live is not None:
        free = live[~live["region"].isin(["crimea", "luhanska"])]
        k1, k2, k3 = st.columns(3)
        k1.metric("Центрів під тривогою зараз", int(free["center_alert"].sum()))
        k2.metric("З них червоний рівень", int(free["center_red"].sum()))
        k3.metric("Областей з будь-якою тривогою", int(free["oblast_alert"].sum()))
        st.caption(
            f"Живий стан: siren.pp.ua, {live['fetched_at'].iloc[0].tz_convert(TZ):%H:%M:%S}. "
            "Лише для відображення — у модель не потрапляє."
        )
    else:
        st.info("Живий стан недоступний (siren.pp.ua не відповідає).")

    # --- selected region: probability over the next 168 h -------------------------------
    g = f[f["region"] == region].sort_values("lead")
    fig = go.Figure()
    for col, (color, label) in SERIES.items():
        fig.add_scatter(
            x=g["local"],
            y=g[col],
            name=label,
            mode="lines",
            line={"color": color, "width": 2},
            hovertemplate="%{x|%a %d.%m %H:00}<br>%{y:.0%}<extra>" + label + "</extra>",
        )
    fig.update_layout(
        title=f"{NAMES[region]}: ймовірність тривоги по годинах",
        yaxis={"range": [0, 1], "tickformat": ".0%", "title": None},
        xaxis={"title": None},
        hovermode="x unified",
        height=380,
        legend={"orientation": "h", "y": -0.15},
        margin={"t": 50, "b": 10, "l": 10, "r": 10},
    )
    st.plotly_chart(fig, width="stretch")
    if bool(g["excluded"].iloc[0]):
        st.warning(
            "Для цього регіону частка годин з тривогою за останні 28 днів > 95%: прогноз "
            "майже завжди ≈ 1, у метриках якості регіон не враховується."
        )

    # --- ranking table ----------------------------------------------------------------
    table = (
        f.groupby("region")
        .apply(
            lambda d: pd.Series(
                {
                    "Наступні 6 год (макс.)": d.loc[d["lead"] <= 6, model].max(),
                    "Наступні 24 год (макс.)": d.loc[d["lead"] <= 24, model].max(),
                    "Середнє за 7 днів": d[model].mean(),
                    "excluded": d["excluded"].iloc[0],
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    table["Область"] = table["region"].map(NAMES)
    if live is not None:
        status = live.set_index("region")
        table["Зараз"] = table["region"].map(
            lambda r: (
                "🔴 червона"
                if status.loc[r, "center_red"]
                else ("🟡 тривога" if status.loc[r, "center_alert"] else "—")
            )
        )
    table["Примітка"] = table["excluded"].map(lambda e: "майже постійна тривога" if e else "")
    cols = [
        "Область",
        *(["Зараз"] if live is not None else []),
        "Наступні 6 год (макс.)",
        "Наступні 24 год (макс.)",
        "Середнє за 7 днів",
        "Примітка",
    ]
    st.dataframe(
        table.sort_values("Наступні 6 год (макс.)", ascending=False)[cols],
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.ProgressColumn(c, format="percent", min_value=0, max_value=1)
            for c in ("Наступні 6 год (макс.)", "Наступні 24 год (макс.)", "Середнє за 7 днів")
        },
    )

with tab_map:
    order = f.groupby("region")[model].mean().sort_values(ascending=False).index.tolist()
    pivot = f.pivot_table(index="region", columns="local", values=model).reindex(order)
    heat = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns,
            y=[NAMES[r] for r in pivot.index],
            zmin=0,
            zmax=1,
            colorscale=[[i / (len(BLUES) - 1), c] for i, c in enumerate(BLUES)],
            colorbar={"title": "P", "tickformat": ".0%"},
            xgap=1,
            ygap=1,
            hovertemplate="%{y}<br>%{x|%a %d.%m %H:00}<br>%{z:.0%}<extra></extra>",
        )
    )
    heat.update_layout(
        height=720,
        margin={"t": 30, "b": 10, "l": 10, "r": 10},
        title=f"{TARGETS[target]} · {SERIES[model][1]} · області за середньою ймовірністю",
    )
    st.plotly_chart(heat, width="stretch")

with tab_quality:
    st.markdown(
        "Оцінка — щотижневий walk-forward на dev-періоді (4 тижні до "
        f"{pd.Timestamp(cfg.dev_end).tz_convert(TZ):%d.%m %H:%M}); модель перед кожним тижнем "
        "перенавчається лише на тому, що вже було відомо. **BSS** — Brier skill score "
        "відносно кліматології (> 0 — краще за неї)."
    )
    path = root / "reports" / f"{target}_dev" / "metrics_by_lead.csv"
    if path.exists():
        m = pd.read_csv(path)
        m = m[m["model"].isin(["climatology", "persistence", "gbm"])]
        st.dataframe(
            m.pivot_table(index="model", columns="lead_bucket", values="bss", sort=False).round(3),
            width="stretch",
        )
    img = root / "reports" / "figures" / f"skill_{target}.png"
    if img.exists():
        st.image(str(img))
    st.markdown("Повний звіт: `reports/REPORT.md`.")
