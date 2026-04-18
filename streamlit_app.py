"""
streamlit_app.py
----------------
Freight Rate Benchmarking Dashboard
European FTL/LTL corridor rate trends with ML forecast overlay

Run: streamlit run app/streamlit_app.py
"""

import sys
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Path setup — works whether the file lives in app/ or at repo root (Streamlit Cloud)
# Always add the repo root (the directory containing the src/ folder) to sys.path
_this_file = Path(__file__).resolve()
_repo_root = _this_file.parent if (_this_file.parent / "src").exists() else _this_file.parent.parent
sys.path.insert(0, str(_repo_root))
from src.ingest import run_ingestion, CORRIDORS
from src.clean import run_cleaning
from src.corridors import (
    get_corridor_summary, get_corridor_timeseries,
    get_corridor_benchmarks, compute_correlation_matrix
)
from src.insights import get_rate_insight, get_market_overview
from src.forecast import forecast_all_corridors, get_forecast_summary

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="European Freight Rate Benchmarking",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — professional, minimal
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0f1117; }
    [data-testid="stSidebar"] .stMarkdown { color: #c8c8d0; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 8px;
        padding: 16px;
    }

    /* Insight panel */
    .insight-box {
        background-color: #1a1d27;
        border-left: 3px solid #4a9eff;
        border-radius: 4px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .insight-box.up { border-left-color: #ff6b6b; }
    .insight-box.down { border-left-color: #51cf66; }
    .insight-box.flat { border-left-color: #fcc419; }

    /* Data source disclaimer */
    .data-note {
        font-size: 0.75rem;
        color: #666;
        font-style: italic;
        margin-top: 4px;
    }

    /* Section headers */
    h3 { font-size: 1.1rem !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading — cached for performance
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading freight rate data...")
def load_data():
    """
    Tries to load pre-processed parquet files first (from pipeline run).
    Falls back to generating data on-the-fly if files not found.
    This allows the app to run on Streamlit Cloud without a separate pipeline step.
    """
    processed_dir = _repo_root / "data" / "processed"

    if (processed_dir / "corridor_rates.parquet").exists():
        rates = pd.read_parquet(processed_dir / "corridor_rates.parquet")
        diesel = pd.read_parquet(processed_dir / "fuel_index.parquet")
        volumes = pd.read_parquet(processed_dir / "volume_index.parquet")
    else:
        # Generate on-the-fly for Streamlit Cloud
        raw = run_ingestion()
        clean = run_cleaning(raw)
        rates, diesel, volumes = clean["rates"], clean["diesel"], clean["volumes"]

    return rates, diesel, volumes


@st.cache_data(ttl=3600, show_spinner="Running forecast models (this takes ~60 seconds first time)...")
def load_forecasts(rates_df: pd.DataFrame):
    """Loads or generates forecast data."""
    fc_path = _repo_root / "data" / "processed" / "forecasts_ftl.parquet"
    if fc_path.exists():
        return pd.read_parquet(fc_path)
    try:
        return forecast_all_corridors(rates_df, rate_col="ftl_rate_eur", forecast_weeks=4)
    except Exception as e:
        st.warning(f"Forecast unavailable: {e}")
        return None


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
rates_df, diesel_df, volume_df = load_data()
summary_df = get_corridor_summary(rates_df)
benchmarks_df = get_corridor_benchmarks(rates_df)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/truck.png", width=48)
    st.title("Freight Rate\nBenchmarking")
    st.caption("European FTL/LTL — 2022–Present")
    st.divider()

    # Corridor selector
    corridor_options = {v["label"]: k for k, v in CORRIDORS.items()}
    selected_label = st.selectbox(
        "Corridor",
        options=list(corridor_options.keys()),
        index=0,
        help="Select a freight corridor to analyse",
    )
    selected_corridor = corridor_options[selected_label]

    # Rate type
    rate_type = st.radio(
        "Rate type",
        options=["FTL (full truck)", "LTL (per pallet)"],
        index=0,
        help="FTL = full truckload in EUR. LTL = per-pallet rate in EUR.",
    )
    rate_col = "ftl_rate_eur" if "FTL" in rate_type else "ltl_rate_eur_pallet"
    rate_label = "EUR (FTL)" if "FTL" in rate_type else "EUR/pallet (LTL)"

    # Date range
    min_date = rates_df["date"].min().date()
    max_date = rates_df["date"].max().date()

    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter the analysis window",
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    st.divider()

    # Show forecast toggle
    show_forecast = st.toggle("Show 4-week forecast", value=True)
    show_confidence = st.toggle("Show confidence band", value=True)

    st.divider()
    st.markdown("""
    **Data sources**
    - ECB diesel price index
    - Eurostat road freight volumes
    - Freightos Baltic Index (macro context)
    - Rate model: cost-component methodology
    """)
    st.markdown(
        '<p class="data-note">⚠ Rates are modeled from public cost-component '
        'inputs, not live market data. For methodology, see README.</p>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# --- Header ---
st.markdown("## 🚛 European Freight Rate Benchmarking Dashboard")
st.caption(
    f"Monitoring FTL/LTL rate trends across 6 European corridors | "
    f"Data through {max_date.strftime('%B %Y')}"
)

# --- Market overview banner ---
overview_text = get_market_overview(summary_df)
st.info(f"**Market overview:** {overview_text}")

# ---------------------------------------------------------------------------
# KPI metrics — top row
# ---------------------------------------------------------------------------
corridor_data = get_corridor_timeseries(
    rates_df, selected_corridor,
    start_date=str(start_date), end_date=str(end_date)
)
bench_row = benchmarks_df[benchmarks_df["corridor"] == selected_corridor].iloc[0]
summary_row = summary_df[summary_df["corridor"] == selected_corridor].iloc[0]

st.subheader(f"📍 {selected_label}")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Current FTL rate",
        value=f"€{summary_row['latest_ftl_eur']:,.0f}",
        delta=f"{summary_row['ftl_vs_90d_avg_pct']:+.1f}% vs 90-day avg",
        delta_color="inverse",  # red = above average (higher cost)
    )

with col2:
    st.metric(
        label="Current LTL rate",
        value=f"€{summary_row['latest_ltl_eur_pallet']:,.0f}/pallet",
        delta=None,
    )

with col3:
    yoy = summary_row.get("ftl_yoy_pct", None)
    if pd.notna(yoy):
        st.metric(
            label="Year-on-year",
            value=f"{yoy:+.1f}%",
            delta=None,
        )
    else:
        st.metric(label="Year-on-year", value="N/A")

with col4:
    st.metric(
        label="52-week range",
        value=f"€{bench_row['52w_low']:,.0f} – €{bench_row['52w_high']:,.0f}",
        delta=f"{bench_row['52w_percentile']:.0f}th percentile",
    )

with col5:
    st.metric(
        label="Rate per km",
        value=f"€{bench_row['rate_per_km']:.2f}/km",
        delta=f"{bench_row['distance_km']:,} km corridor",
    )

st.divider()

# ---------------------------------------------------------------------------
# Main chart — rate trend + forecast
# ---------------------------------------------------------------------------
col_chart, col_insight = st.columns([3, 1])

with col_chart:
    st.subheader("Rate trend")

    fig = go.Figure()

    # Historical rate line
    fig.add_trace(go.Scatter(
        x=corridor_data["date"],
        y=corridor_data[rate_col],
        mode="lines",
        name="Historical rate",
        line=dict(color="#4a9eff", width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Rate: €%{y:,.0f}<extra></extra>",
    ))

    # 13-week rolling average
    avg_col = "ftl_13w_avg" if rate_col == "ftl_rate_eur" else "ltl_13w_avg"
    if avg_col in corridor_data.columns:
        fig.add_trace(go.Scatter(
            x=corridor_data["date"],
            y=corridor_data[avg_col],
            mode="lines",
            name="13-week avg",
            line=dict(color="#fcc419", width=1.5, dash="dot"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>13w avg: €%{y:,.0f}<extra></extra>",
        ))

    # Forecast overlay
    if show_forecast and rate_col == "ftl_rate_eur":
        try:
            fc_df = load_forecasts(rates_df)
            if fc_df is not None:
                fc_corridor = fc_df[fc_df["corridor"] == selected_corridor]
                fc_future = fc_corridor[fc_corridor["is_forecast"]]
                fc_last_actual = fc_corridor[~fc_corridor["is_forecast"]].tail(1)

                # Connect the last actual to the forecast
                bridge = pd.concat([fc_last_actual, fc_future])

                fig.add_trace(go.Scatter(
                    x=bridge["date"],
                    y=bridge["yhat"],
                    mode="lines",
                    name="4-week forecast",
                    line=dict(color="#ff6b6b", width=2, dash="dash"),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: €%{y:,.0f}<extra></extra>",
                ))

                if show_confidence and not fc_future.empty:
                    # Confidence band
                    fig.add_trace(go.Scatter(
                        x=pd.concat([fc_future["date"], fc_future["date"].iloc[::-1]]),
                        y=pd.concat([fc_future["yhat_upper"], fc_future["yhat_lower"].iloc[::-1]]),
                        fill="toself",
                        fillcolor="rgba(255, 107, 107, 0.12)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="80% confidence band",
                        hoverinfo="skip",
                    ))

                    # Forecast period shading
                    if len(fc_future) > 0:
                        forecast_start = fc_future["date"].min()
                        fig.add_vrect(
                            x0=forecast_start,
                            x1=fc_future["date"].max(),
                            fillcolor="rgba(255, 107, 107, 0.05)",
                            layer="below",
                            line_width=0,
                            annotation_text="Forecast",
                            annotation_position="top left",
                            annotation_font_size=11,
                            annotation_font_color="#ff6b6b",
                        )
        except Exception as e:
            st.caption(f"Forecast unavailable: {e}")

    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8c8d0", size=12),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            title=f"Rate ({rate_label})",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Data source note
    st.markdown(
        '<p class="data-note">Rate data generated from ECB diesel index + Eurostat '
        'volume signals using IRU-calibrated cost model. Not live market data.</p>',
        unsafe_allow_html=True,
    )

with col_insight:
    st.subheader("📊 Insight")

    # Main insight
    insight = get_rate_insight(
        corridor_label=selected_label,
        ftl_vs_90d=summary_row["ftl_vs_90d_avg_pct"],
        yoy_pct=summary_row.get("ftl_yoy_pct", np.nan),
        pct_rank_52w=bench_row["52w_percentile"],
        latest_rate=summary_row["latest_ftl_eur"],
    )

    direction = "up" if summary_row["ftl_vs_90d_avg_pct"] > 3 else \
                "down" if summary_row["ftl_vs_90d_avg_pct"] < -3 else "flat"

    st.markdown(
        f'<div class="insight-box {direction}">{insight["headline"]}</div>',
        unsafe_allow_html=True,
    )
    if insight["detail"]:
        st.markdown(
            f'<div class="insight-box flat">{insight["detail"]}</div>',
            unsafe_allow_html=True,
        )

    # Forecast insight
    if show_forecast and rate_col == "ftl_rate_eur":
        try:
            fc_df = load_forecasts(rates_df)
            if fc_df is not None:
                fc_summary = get_forecast_summary(fc_df)
                fc_row = fc_summary[fc_summary["corridor"] == selected_corridor]
                if not fc_row.empty:
                    row = fc_row.iloc[0]
                    arrow = "↑" if row["direction"] == "up" else \
                            "↓" if row["direction"] == "down" else "→"
                    fc_text = (
                        f"**4-week forecast:** €{row['forecast_4w']:,.0f} "
                        f"{arrow} ({row['change_pct']:+.1f}%)<br>"
                        f"80% CI: €{row['forecast_lower']:,.0f} – €{row['forecast_upper']:,.0f}"
                    )
                    st.markdown(
                        f'<div class="insight-box {row["direction"]}">{fc_text}</div>',
                        unsafe_allow_html=True,
                    )
        except Exception:
            pass

    st.divider()
    st.caption("**How to read this**")
    st.caption(
        "The 90-day average is a rolling 13-week mean. "
        "The 52-week percentile tells you where current rates sit "
        "vs the past year's range. 80th+ = expensive; 20th- = cheap."
    )

# ---------------------------------------------------------------------------
# Tab section — additional views
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 All corridors", "🔥 Rate heatmap", "⛽ Cost drivers", "🔗 Corridor correlation"
])

with tab1:
    st.subheader("All corridors — current rate comparison")

    # Bar chart: current FTL rate by corridor
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=summary_df["label"],
        y=summary_df["latest_ftl_eur"],
        marker_color=summary_df["ftl_vs_90d_avg_pct"].apply(
            lambda x: "#ff6b6b" if x > 3 else "#51cf66" if x < -3 else "#4a9eff"
        ),
        text=summary_df["ftl_vs_90d_avg_pct"].apply(lambda x: f"{x:+.1f}% vs 90d"),
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "FTL rate: €%{y:,.0f}<br>"
            "<extra></extra>"
        ),
    ))
    fig_bar.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8c8d0"),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="FTL Rate (EUR)", gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("🟥 Above 90-day avg · 🟩 Below 90-day avg · 🟦 Near average")

    # Summary table
    display_cols = {
        "label": "Corridor",
        "distance_km": "Distance (km)",
        "latest_ftl_eur": "FTL Rate (€)",
        "latest_ltl_eur_pallet": "LTL (€/pallet)",
        "ftl_vs_90d_avg_pct": "vs 90-day avg (%)",
        "ftl_yoy_pct": "YoY (%)",
        "ytd_volatility": "YTD Volatility",
    }
    table_df = summary_df[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(
        table_df.style.format({
            "FTL Rate (€)": "€{:,.0f}",
            "LTL (€/pallet)": "€{:,.0f}",
            "vs 90-day avg (%)": "{:+.1f}%",
            "YoY (%)": "{:+.1f}%",
            "YTD Volatility": "€{:,.0f}",
        }).background_gradient(subset=["vs 90-day avg (%)"], cmap="RdYlGn_r"),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Rate heatmap — FTL rates by corridor × month")

    # Pivot for heatmap
    heatmap_df = rates_df.copy()
    heatmap_df["year_month"] = heatmap_df["date"].dt.strftime("%Y-%m")
    heatmap_pivot = heatmap_df.groupby(["corridor_label", "year_month"])["ftl_rate_eur"].mean().unstack()
    heatmap_pivot = heatmap_pivot.iloc[:, -24:]  # Last 24 months

    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale=[
            [0.0, "#1a3a2a"], [0.4, "#2d6a4f"], [0.6, "#fcc419"],
            [0.8, "#e67700"], [1.0, "#c92a2a"]
        ],
        hovertemplate="<b>%{y}</b><br>%{x}: €%{z:,.0f}<extra></extra>",
        colorbar=dict(title="FTL (€)", tickfont=dict(color="#c8c8d0")),
    ))
    fig_heat.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8c8d0"),
        xaxis=dict(title=""),
        yaxis=dict(title=""),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption(
        "Heatmap shows average weekly FTL rate per month. "
        "Green = lower cost / yellow-red = higher cost. "
        "Q4 seasonality and 2022 energy shock are visible."
    )

with tab3:
    st.subheader("Cost drivers — diesel price index")

    fig_diesel = go.Figure()
    fig_diesel.add_trace(go.Scatter(
        x=diesel_df["date"],
        y=diesel_df["diesel_yoy_pct"],
        mode="lines",
        name="Diesel YoY %",
        line=dict(color="#fcc419", width=2),
        fill="tozeroy",
        fillcolor="rgba(252, 196, 25, 0.08)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Diesel YoY: %{y:.1f}%<extra></extra>",
    ))

    fig_diesel.add_trace(go.Scatter(
        x=diesel_df["date"],
        y=diesel_df["diesel_3m_ma"],
        mode="lines",
        name="3-month MA",
        line=dict(color="#ff6b6b", width=1.5, dash="dot"),
        hovertemplate="<b>%{x|%b %Y}</b><br>3m MA: %{y:.1f}%<extra></extra>",
    ))

    fig_diesel.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)

    fig_diesel.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8c8d0"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(
            title="Year-on-year % change",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
        ),
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_diesel, use_container_width=True)
    st.caption(
        "Diesel cost accounts for ~30% of FTL operating cost (IRU methodology). "
        "YoY spikes in 2022 drove rate increases across all corridors. "
        "Source: ECB Statistical Data Warehouse (proxy: petroleum products PPI)."
    )

with tab4:
    st.subheader("Corridor correlation matrix")
    st.caption(
        "High correlation = corridors move together (shared demand/supply drivers). "
        "PL-DE tends to decorrelate during peak season due to different carrier pools."
    )

    corr_matrix = compute_correlation_matrix(rates_df)

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        hovertemplate="<b>%{x} × %{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Pearson r", tickfont=dict(color="#c8c8d0")),
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=11, color="#c8c8d0"),
    ))
    fig_corr.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8c8d0"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown("""
<div style="text-align: center; color: #555; font-size: 0.8rem; padding: 8px 0;">
    Freight Rate Benchmarking Dashboard · Built with Streamlit + Prophet + Plotly ·
    <a href="https://github.com/yourusername/freight-rate-benchmarking" style="color: #4a9eff;">
    GitHub</a> · Data: ECB, Eurostat, Freightos Baltic Index ·
    ⚠ Rate data is modeled, not live market data
</div>
""", unsafe_allow_html=True)
