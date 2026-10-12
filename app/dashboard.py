"""Streamlit dashboard: short-range air-alert forecast + typical 7-day pattern + live state.

Only the first `useful_lead_hours` are shown as a forecast (the horizon where the model is
measurably better than climatology); the rest of the week is the climatological pattern.

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

FORECAST_COLOR, TYPICAL_COLOR = "#2a78d6", "#eb6834"
MAIN_NAMES = {
    "gbm_resid": "LightGBM поверх кліматології",
    "gbm_resid_short": "LightGBM поверх кліматології (1–12 год)",
    "persistence": "«Як зараз» → кліматологія",
}
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
model_meta = load_json("models/y_alert_gbm_resid_short.json")
origin = pd.Timestamp(fc["origin"].iloc[0]).tz_convert(TZ)
data_end = pd.Timestamp(str(meta.get("data_end"))).tz_convert(TZ)

H_ALL = cfg.useful_lead_hours

st.title("Прогноз повітряних тривог")
st.caption(
    f"Прогноз від **{origin:%d.%m %H:%M}** (Київ) · дані до **{data_end:%d.%m %H:%M}** · "
    f"код моделі `{model_meta.get('git_commit', '?')}`. Ціль — тривога, що покриває обласний "
    f"центр. **Прогнозом** вважаємо лише перші {H_ALL['y_alert']} год (червона — "
    f"{H_ALL['y_red']} год): далі модель статистично не відрізняється від типового розкладу "
    "тривог, тож решту тижня показуємо саме як типовий розклад, а не прогноз."
)

c1, c2 = st.columns([1.3, 1.7])
target = c1.radio("Ціль", list(TARGETS), format_func=TARGETS.get, horizontal=True)
regions = sorted(fc["region"].unique(), key=lambda r: NAMES[r])
region = c2.selectbox("Область", regions, index=regions.index("kyiv_city"), format_func=NAMES.get)

H = H_ALL[target]
f = fc[fc["target"] == target]
near = f[f["lead"] <= H]
main_name = MAIN_NAMES.get(str(f["main_model"].iloc[0]), str(f["main_model"].iloc[0]))
live = load_live()


def heatmap(pivot: pd.DataFrame, height: int, title: str) -> go.Figure:
    fig = go.Figure(
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
    fig.update_layout(height=height, margin={"t": 40, "b": 10, "l": 10, "r": 10}, title=title)
    return fig


tab_now, tab_week, tab_quality = st.tabs(
    [f"Прогноз на {H} год", "Типовий тиждень", "Якість моделі"]
)

with tab_now:
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

    order = near.groupby("region")["p_main"].max().sort_values(ascending=False).index
    pivot = near.pivot_table(index="region", columns="local", values="p_main").reindex(order)
    st.plotly_chart(
        heatmap(pivot, 640, f"{TARGETS[target]}: прогноз на {H} год · модель: {main_name}"),
        width="stretch",
    )

    # Selected region: forecast part vs typical pattern for the whole week.
    g = f[f["region"] == region].sort_values("lead")
    gn = g[g["lead"] <= H]
    fig = go.Figure()
    fig.add_vrect(
        x0=gn["local"].min() - pd.Timedelta(minutes=30),
        x1=gn["local"].max() + pd.Timedelta(minutes=30),
        fillcolor=FORECAST_COLOR,
        opacity=0.08,
        line_width=0,
        annotation_text="прогноз",
        annotation_position="top left",
    )
    fig.add_scatter(
        x=g["local"],
        y=g["p_climatology"],
        name="Типовий розклад (28 днів), не прогноз",
        mode="lines",
        line={"color": TYPICAL_COLOR, "width": 2, "dash": "dot"},
        hovertemplate="%{x|%a %d.%m %H:00}<br>%{y:.0%}<extra>типово</extra>",
    )
    fig.add_scatter(
        x=gn["local"],
        y=gn["p_main"],
        name=f"Прогноз ({main_name})",
        mode="lines+markers",
        line={"color": FORECAST_COLOR, "width": 2},
        marker={"size": 8},
        hovertemplate="%{x|%a %d.%m %H:00}<br>%{y:.0%}<extra>прогноз</extra>",
    )
    fig.update_layout(
        title=f"{NAMES[region]}: прогноз на {H} год і типовий розклад на тиждень",
        yaxis={"range": [0, 1.05], "tickformat": ".0%", "title": None},
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

    fcol = f"Прогноз: макс. за {H} год"
    table = (
        f.groupby("region")
        .apply(
            lambda d: pd.Series(
                {
                    fcol: d.loc[d["lead"] <= H, "p_main"].max(),
                    "Типово: наступна доба": d.loc[d["lead"] <= 24, "p_climatology"].mean(),
                    "Типово: тиждень": d["p_climatology"].mean(),
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
    pct = [fcol, "Типово: наступна доба", "Типово: тиждень"]
    cols = ["Область", *(["Зараз"] if live is not None else []), *pct, "Примітка"]
    st.dataframe(
        table.sort_values(fcol, ascending=False)[cols],
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.ProgressColumn(c, format="percent", min_value=0, max_value=1)
            for c in pct
        },
    )

with tab_week:
    st.markdown(
        "**Це не прогноз.** Частка годин з тривогою в цю годину доби за останні 28 днів "
        "(кліматологія) — типовий розклад, на який варто зважати при плануванні. На горизонті "
        f"понад {H} год наша модель статистично не краща за нього."
    )
    order = f.groupby("region")["p_climatology"].mean().sort_values(ascending=False).index
    pivot = f.pivot_table(index="region", columns="local", values="p_climatology").reindex(order)
    st.plotly_chart(
        heatmap(pivot, 720, f"{TARGETS[target]}: типовий розклад на 7 днів"), width="stretch"
    )

with tab_quality:
    st.markdown(
        "**BSS** — Brier skill score відносно кліматології (> 0 — модель краща за типовий "
        "розклад). Інтервал — 95% бутстреп по днях. Звідси й межа прогнозу: "
        f"{H_ALL['y_alert']} год для будь-якої тривоги, {H_ALL['y_red']} год для червоної."
    )
    for name, title in (
        (f"useful_horizon_{target}_backtest.csv", "Бектест 24 місяці"),
        (f"useful_horizon_{target}_dev.csv", "Dev-період, 4 тижні"),
    ):
        path = root / "reports" / name
        if path.exists():
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(path).round(3), hide_index=True, width="stretch")
    img = root / "reports" / "figures" / f"horizon_{target}.png"
    if img.exists():
        st.image(str(img))
    st.markdown("Повний звіт: `reports/REPORT.md`.")
