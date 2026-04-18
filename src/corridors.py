"""
corridors.py
------------
Corridor-level aggregation, statistical summaries, and comparison metrics.
Designed so each function is independently testable and notebook-friendly.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_corridor_summary(rates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary table: one row per corridor with key statistics.
    Used in the dashboard summary panel and README metrics.
    """
    records = []
    for corridor, grp in rates_df.groupby("corridor"):
        grp = grp.sort_values("date")
        latest = grp.iloc[-1]
        ytd = grp[grp["date"].dt.year == grp["date"].dt.year.max()]

        records.append({
            "corridor": corridor,
            "label": latest["corridor_label"],
            "region": latest["region"],
            "distance_km": latest["distance_km"],
            "latest_ftl_eur": latest["ftl_rate_eur"],
            "latest_ltl_eur_pallet": latest["ltl_rate_eur_pallet"],
            "ftl_vs_90d_avg_pct": latest["ftl_vs_90d_avg_pct"],
            "ftl_yoy_pct": latest["ftl_yoy_pct"],
            "ytd_avg_ftl": ytd["ftl_rate_eur"].mean().round(2),
            "ytd_volatility": ytd["ftl_rate_eur"].std().round(2),
            "data_source": latest["data_source"],
        })

    return pd.DataFrame(records).sort_values("latest_ftl_eur", ascending=False)


def get_corridor_timeseries(
    rates_df: pd.DataFrame,
    corridor: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Returns time series for a single corridor, optionally date-filtered.
    Used directly by Streamlit chart components.
    """
    df = rates_df[rates_df["corridor"] == corridor].copy()
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    return df.sort_values("date").reset_index(drop=True)


def get_corridor_benchmarks(rates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cross-corridor benchmarks:
    - percentile rank of current rate vs 52-week distribution
    - rate per km (normalised for fair comparison across corridors)
    """
    records = []
    for corridor, grp in rates_df.groupby("corridor"):
        grp = grp.sort_values("date")
        last_52w = grp.tail(52)
        current_rate = grp.iloc[-1]["ftl_rate_eur"]
        distance = grp.iloc[-1]["distance_km"]

        pct_rank = (last_52w["ftl_rate_eur"] < current_rate).mean() * 100

        records.append({
            "corridor": corridor,
            "label": grp.iloc[-1]["corridor_label"],
            "distance_km": distance,
            "current_ftl_eur": current_rate,
            "rate_per_km": round(current_rate / distance, 3),
            "52w_percentile": round(pct_rank, 1),
            "52w_high": last_52w["ftl_rate_eur"].max(),
            "52w_low": last_52w["ftl_rate_eur"].min(),
            "52w_avg": last_52w["ftl_rate_eur"].mean().round(2),
        })

    return pd.DataFrame(records)


def get_regional_heatmap_data(rates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares data for the heatmap view: corridors vs months,
    values = avg FTL rate. Used by the Streamlit heatmap chart.
    """
    df = rates_df.copy()
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    pivot = (
        df.groupby(["corridor", "year_month"])["ftl_rate_eur"]
        .mean()
        .round(2)
        .reset_index()
    )
    return pivot


def compute_correlation_matrix(rates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes corridor-to-corridor FTL rate correlation matrix.
    High correlation = rates move together (useful for portfolio/hedging framing).
    """
    pivot = rates_df.pivot_table(
        index="date", columns="corridor", values="ftl_rate_eur"
    )
    return pivot.corr().round(3)
