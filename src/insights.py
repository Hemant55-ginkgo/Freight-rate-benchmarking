"""
insights.py
-----------
Generates plain-English insight strings for the dashboard insight panel.
These are rule-based — not ML — but structured to be extensible with LLM generation later.
"""

import pandas as pd
import numpy as np


def get_rate_insight(corridor_label: str, ftl_vs_90d: float, yoy_pct: float,
                     pct_rank_52w: float, latest_rate: float) -> dict:
    """
    Returns a structured insight dict for a corridor.

    Parameters
    ----------
    corridor_label : str
    ftl_vs_90d : float    % deviation from 90-day average
    yoy_pct : float       year-on-year % change
    pct_rank_52w : float  current rate vs 52-week range (0=low, 100=high)
    latest_rate : float   current FTL EUR
    """
    # Rate level signal
    if ftl_vs_90d > 8:
        level_signal = "significantly above"
        level_colour = "red"
    elif ftl_vs_90d > 3:
        level_signal = "above"
        level_colour = "orange"
    elif ftl_vs_90d < -8:
        level_signal = "significantly below"
        level_colour = "green"
    elif ftl_vs_90d < -3:
        level_signal = "below"
        level_colour = "lightgreen"
    else:
        level_signal = "in line with"
        level_colour = "grey"

    # 52-week position
    if pct_rank_52w >= 80:
        range_signal = "near yearly highs"
    elif pct_rank_52w <= 20:
        range_signal = "near yearly lows"
    else:
        range_signal = f"at the {int(pct_rank_52w)}th percentile of the past year"

    # YoY narrative
    if pd.isna(yoy_pct):
        yoy_narrative = ""
    elif yoy_pct > 10:
        yoy_narrative = f"Rates are up {yoy_pct:.1f}% year-on-year, signalling sustained market tightness."
    elif yoy_pct < -10:
        yoy_narrative = f"Rates have fallen {abs(yoy_pct):.1f}% vs last year, reflecting softening demand."
    elif yoy_pct > 0:
        yoy_narrative = f"A modest {yoy_pct:.1f}% year-on-year increase suggests stable market conditions."
    else:
        yoy_narrative = f"Rates are {abs(yoy_pct):.1f}% below last year, with capacity likely available."

    headline = (
        f"At €{latest_rate:,.0f}, rates on {corridor_label} are "
        f"{abs(ftl_vs_90d):.1f}% {level_signal} the 90-day average — {range_signal}."
    )

    return {
        "headline": headline,
        "detail": yoy_narrative,
        "level_colour": level_colour,
        "ftl_vs_90d": ftl_vs_90d,
        "pct_rank_52w": pct_rank_52w,
    }


def get_market_overview(summary_df: pd.DataFrame) -> str:
    """
    Generates a 2-sentence market overview across all corridors.
    Used in the dashboard header.
    """
    n_corridors = len(summary_df)
    above_avg = (summary_df["ftl_vs_90d_avg_pct"] > 3).sum()
    below_avg = (summary_df["ftl_vs_90d_avg_pct"] < -3).sum()

    if above_avg > n_corridors / 2:
        tone = "Rates are elevated across the majority of monitored corridors."
    elif below_avg > n_corridors / 2:
        tone = "Rates are softening across most monitored corridors."
    else:
        tone = "The market is mixed: some corridors are above average, others below."

    best_corridor = summary_df.loc[
        summary_df["ftl_vs_90d_avg_pct"].idxmin(), "label"
    ]
    worst_corridor = summary_df.loc[
        summary_df["ftl_vs_90d_avg_pct"].idxmax(), "label"
    ]

    detail = (
        f"{best_corridor} offers the most competitive rates vs recent average; "
        f"{worst_corridor} shows the highest rate pressure."
    )
    return f"{tone} {detail}"
