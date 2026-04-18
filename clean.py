"""
clean.py
--------
Cleans, validates, and normalises raw freight rate data.
Produces a standardised DataFrame ready for corridor aggregation.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def clean_rates(rates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and cleans the rates DataFrame.

    Steps:
    1. Ensure date column is datetime
    2. Remove duplicates
    3. Clip statistical outliers (>3 IQR from median per corridor)
    4. Add derived columns: rate index (base 100 = 2022-01-01), rolling averages
    5. Flag data quality issues
    """
    logger.info("Cleaning rates data...")
    df = rates_df.copy()

    # 1. Date standardisation
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["corridor", "date"]).reset_index(drop=True)

    # 2. Deduplication
    before = len(df)
    df = df.drop_duplicates(subset=["date", "corridor"])
    if len(df) < before:
        logger.warning(f"Removed {before - len(df)} duplicate rows.")

    # 3. Outlier clipping per corridor
    for corridor in df["corridor"].unique():
        mask = df["corridor"] == corridor
        for col in ["ftl_rate_eur", "ltl_rate_eur_pallet"]:
            q1 = df.loc[mask, col].quantile(0.05)
            q3 = df.loc[mask, col].quantile(0.95)
            iqr = q3 - q1
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            outliers = ((df[col] < lower) | (df[col] > upper)) & mask
            if outliers.sum() > 0:
                logger.warning(
                    f"Clipping {outliers.sum()} outliers in {corridor}/{col}"
                )
                df.loc[outliers & mask, col] = df.loc[mask, col].clip(lower, upper)

    # 4. Derived columns
    # Rate index: 100 = average of first 4 weeks per corridor
    df["ftl_index"] = df.groupby("corridor")["ftl_rate_eur"].transform(
        lambda x: x / x.iloc[:4].mean() * 100
    )
    df["ltl_index"] = df.groupby("corridor")["ltl_rate_eur_pallet"].transform(
        lambda x: x / x.iloc[:4].mean() * 100
    )

    # Rolling 13-week (quarter) average
    df = df.sort_values(["corridor", "date"])
    df["ftl_13w_avg"] = (
        df.groupby("corridor")["ftl_rate_eur"]
        .transform(lambda x: x.rolling(13, min_periods=4).mean())
        .round(2)
    )
    df["ltl_13w_avg"] = (
        df.groupby("corridor")["ltl_rate_eur_pallet"]
        .transform(lambda x: x.rolling(13, min_periods=4).mean())
        .round(2)
    )

    # 90-day (13-week) deviation flag for insight panel
    df["ftl_vs_90d_avg_pct"] = (
        (df["ftl_rate_eur"] - df["ftl_13w_avg"]) / df["ftl_13w_avg"] * 100
    ).round(1)

    # Year-over-year change (52 weeks back)
    df["ftl_yoy_pct"] = (
        df.groupby("corridor")["ftl_rate_eur"]
        .transform(lambda x: x.pct_change(52) * 100)
        .round(1)
    )

    # Data quality flag
    df["dq_flag"] = "ok"
    df.loc[df["ftl_rate_eur"].isna(), "dq_flag"] = "missing"
    df.loc[df["data_source"] == "modeled", "dq_flag"] = df.loc[
        df["data_source"] == "modeled", "dq_flag"
    ].replace("ok", "synthetic")

    logger.info(f"Clean rates: {len(df)} rows, {df['corridor'].nunique()} corridors.")
    return df


def clean_diesel(diesel_df: pd.DataFrame) -> pd.DataFrame:
    """Normalises diesel index: forward-fill gaps, add 3m moving average."""
    df = diesel_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    df["diesel_yoy_pct"] = df["diesel_yoy_pct"].ffill()
    df["diesel_3m_ma"] = df["diesel_yoy_pct"].rolling(3, min_periods=1).mean().round(2)
    return df


def clean_volumes(volume_df: pd.DataFrame) -> pd.DataFrame:
    """Normalises volume data and adds QoQ growth."""
    df = volume_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["corridor", "date"])
    df["volume_qoq_pct"] = (
        df.groupby("corridor")["volume_ths_tonnes"]
        .transform(lambda x: x.pct_change() * 100)
        .round(1)
    )
    return df


def run_cleaning(raw_data: dict) -> dict:
    """Entry point: accepts dict of raw DataFrames, returns dict of cleaned ones."""
    return {
        "rates": clean_rates(raw_data["rates"]),
        "diesel": clean_diesel(raw_data["diesel"]),
        "volumes": clean_volumes(raw_data["volumes"]),
    }
