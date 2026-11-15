import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from .data import ROOT, calendar_features, configuration, load_data, split_indices
from .models import CHECKPOINTS, LABELS, METHODS, ModelBundle

STYLE = (
    "<style>"
    ".stApp,[data-testid='stHeader']{background:#090909;color:#f1f1f1}"
    ".block-container{max-width:1280px;padding:1.4rem 2rem 2rem}"
    "h1{font-size:2rem!important;font-weight:600!important;letter-spacing:-.04em}"
    "h2,h3{font-size:1.1rem!important;font-weight:600!important;letter-spacing:-.02em}"
    "p,label{font-size:.9rem}[data-testid='stCaptionContainer']{color:#aaa}"
    "[data-testid='stToolbar']{display:none}[data-testid='stVerticalBlock']{gap:.7rem}"
    "@media(max-width:800px){.block-container{padding:1rem}"
    "[data-testid='stHorizontalBlock']{flex-wrap:wrap}"
    "[data-testid='stColumn']{min-width:min(100%,220px);flex:1 1 220px}}"
    "</style>"
)


@st.cache_data
def read_data():
    values, times, stations, stats = load_data()
    indices = split_indices(times)["test"]
    return values[indices], times[indices], stations, stats


@st.cache_resource
def model_bundle(signature):
    return ModelBundle()


def plot(figure, height=300):
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090909",
        plot_bgcolor="#090909",
        height=height,
        margin=dict(l=0, r=0, t=12, b=0),
        font=dict(color="#e8e8e8", size=12),
        legend=dict(orientation="h", y=1.15),
    )
    figure.update_xaxes(showgrid=False, zeroline=False)
    figure.update_yaxes(gridcolor="#252525", zeroline=False)
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def simulator(values, times, stations, config):
    st.subheader("Sensor failure simulator")
    controls = st.columns([1.2, 0.6, 1.8, 1])
    date = controls[0].date_input(
        "Test date",
        times[0].date(),
        min_value=times[0].date(),
        max_value=times[-1].date(),
        key="date",
    )
    hour = controls[1].selectbox("Hour", list(range(24)), index=12, key="hour")
    method = controls[2].selectbox(
        "Model",
        METHODS,
        index=METHODS.index("ensemble"),
        format_func=lambda name: LABELS[name],
        key="model",
    )
    duration = controls[3].selectbox(
        "Failure duration",
        [1, 6, 24, 72, 168],
        index=2,
        format_func=lambda value: f"{value}h",
        key="duration",
    )
    timestamp = pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=hour)
    row = int(np.searchsorted(times.asi8, timestamp.value))
    defaults = config["station_selection"]["choices"]["5"]["stations"]
    columns = st.columns(6)
    selected = []
    for index, station in enumerate(stations):
        selected.append(
            columns[index % 6].checkbox(
                station, value=station in defaults, key=f"station_{index}"
            )
        )
    operational_now = np.asarray(selected, dtype=bool)
    start = max(0, row - 24)
    observed = torch.tensor(values[start : row + 1], device="cuda")
    operational = torch.ones_like(observed, dtype=torch.bool)
    affected = np.arange(start, row + 1) >= row - duration + 1
    operational[torch.tensor(affected, device="cuda")] = torch.tensor(
        operational_now, device="cuda"
    )
    cycle = torch.tensor(calendar_features(times[start : row + 1]), device="cuda")
    signature = tuple(
        (
            name,
            (ROOT / "models" / name).stat().st_mtime_ns
            if (ROOT / "models" / name).exists()
            else None,
        )
        for name in [f"{model}.pt" for model in CHECKPOINTS]
    )
    prediction = (
        model_bundle(signature)
        .predict(method, observed, operational, cycle)[-1]
        .cpu()
        .numpy()
    )
    measured = values[row]
    available = operational_now & np.isfinite(measured)
    target = ~operational_now & np.isfinite(measured)
    st.caption(
        f"{timestamp:%Y-%m-%d %H:%M} Beijing time | {available.sum()}/12 current inputs | Failure interval ends at this hour"
    )
    if not available.any():
        st.warning(
            "No current readings available | Outside the published K=1-11 benchmark"
        )
    reveal = st.toggle("Reveal held-out measurements", key="reveal")
    figure = go.Figure()
    figure.add_bar(
        x=stations,
        y=np.where(available, measured, np.nan),
        name="Measured input",
        marker_color="#d0d0d0",
    )
    figure.add_bar(
        x=stations,
        y=np.where(~available, prediction, np.nan),
        name="Reconstruction",
        marker_color="#78a8cb",
    )
    if reveal:
        figure.add_scatter(
            x=stations,
            y=np.where(~available, measured, np.nan),
            name="Held-out truth",
            mode="markers",
            marker=dict(color="#a8c8b4", symbol="diamond", size=8),
        )
    figure.update_layout(barmode="group", yaxis_title="PM2.5")
    plot(figure, 290)
    if reveal:
        table = pd.DataFrame(
            {
                "Station": stations,
                "Input": np.where(available, measured, np.nan),
                "Reconstruction": np.where(~available, prediction, np.nan),
                "Held-out truth": np.where(~available, measured, np.nan),
                "Error": np.where(target, np.abs(prediction - measured), np.nan),
            }
        )
        if target.any():
            st.caption(
                f"Offline-target MAE {np.abs(prediction[target] - measured[target]).mean():.2f} µg/m³ | Original source gaps are never scored"
            )
        st.dataframe(table.round(2), hide_index=True, width="stretch", height=235)
    st.caption(
        "Failed readings stay hidden throughout the interval | Earlier readings remain available | No future hours"
    )


def model_results(frame):
    st.subheader("Published model results")
    names = ["ridge", "ordinary", "extreme", "saits", "temporal_mdae", "ensemble"]
    table = frame.query("scope == 'overall' and method in @names")[
        ["method", "mae", "rmse", "precision", "recall", "f1"]
    ].copy()
    table["method"] = table.method.map(LABELS)
    table.columns = ["Model", "MAE", "RMSE", "Precision", "Recall", "F1"]
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "Precision": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="%.3f"),
            "F1": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        "Same held-out targets and outage masks | SAITS and temporal MDAE use 24h history | Event threshold 187 µg/m³"
    )


def station_priority(config):
    st.subheader("Station priority")
    k = st.select_slider(
        "Operational stations", options=[3, 5, 8], value=5, key="priority_k"
    )
    choice = config["station_selection"]["choices"][str(k)]
    st.write(", ".join(choice["stations"]))
    frame = pd.read_csv(ROOT / "results/station_subsets.csv").query(
        "method == 'ordinary' and k == @k"
    )
    table = (
        frame.groupby("selection")
        .agg(MAE=("mae", "mean"), Recall=("recall", "mean"))
        .reindex(["optimized", "random", "fixed"])
        .round(3)
    )
    st.dataframe(table, width="stretch")
    st.caption(
        "Validation-selected network | Random mean over 50 subsets | Offline maintenance simulation"
    )


def continuous_failures():
    st.subheader("Continuous failures")
    frame = pd.read_csv(ROOT / "results/continuous.csv").query(
        "scope == 'endpoint' and k == 0"
    )
    figure = go.Figure()
    for method, color in [
        ("ordinary", "#a5a5a5"),
        ("temporal_mdae", "#888cb5"),
        ("saits", "#78a8cb"),
        ("ensemble", "#9ac7b2"),
    ]:
        selected = frame[frame.method == method].sort_values("duration_hours")
        figure.add_scatter(
            x=selected.duration_hours,
            y=selected.mae,
            mode="lines+markers",
            name=LABELS[method],
            line_color=color,
        )
    figure.update_layout(
        xaxis_title="Failure hours",
        yaxis_title="Endpoint MAE",
    )
    figure.update_xaxes(type="log", tickvals=[1, 6, 24, 72, 168])
    plot(figure, 270)
    st.caption(
        "Identical endpoint hours and failed stations across durations | Current-only MDAE remains at 14.66 µg/m³"
    )


def main():
    st.set_page_config(page_title="SensorGuard", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("SensorGuard")
    st.write(
        "Reconstruct missing PM2.5 readings and prioritize stations during sensor failures"
    )
    try:
        config = configuration()
        frame = pd.read_csv(ROOT / "results/metrics.csv")
        ensemble = frame.query("scope == 'overall' and method == 'ensemble'").iloc[0]
        st.markdown(
            f"**MAE {ensemble.mae:.2f} µg/m³** | **F1 {ensemble.f1:.4f}** | Beijing, 12 stations | Held-out July 2016 to February 2017"
        )
        values, times, stations, stats = read_data()
        simulator(values, times, stations, config)
        left, right = st.columns([1.5, 1])
        with left:
            model_results(frame)
        with right:
            station_priority(config)
        continuous_failures()
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        st.error(f"Cannot load the experiment: {error}")
        st.code("python -m src.evaluation")
