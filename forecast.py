"""
forecast.py
-----------
4-week ahead FTL rate forecasting using Facebook Prophet.

WHY PROPHET?
------------
Prophet is a decomposable time series model: it fits trend + seasonality + holidays
separately, then combines them. For freight rates, this is well-suited because:
  - Rates have clear weekly and annual seasonality (Q4 peak, summer dip)
  - The trend changes over time (energy crisis shocks, demand cycles)
  - Prophet handles gaps and outliers gracefully
  - The confidence intervals are interpretable by non-statisticians

WHAT THE MODEL DOES (explain this in interviews):
  "I use Prophet to decompose freight rate history into a trend component —
   which direction rates are heading — and a seasonality component — the
   predictable weekly and annual patterns. It then projects these forward
   4 weeks. The shaded band is the 80% confidence interval: historically,
   the actual rate has fallen inside that band 8 times out of 10."

LIMITATIONS (always acknowledge these — it shows maturity):
  - No exogenous variables (diesel, capacity) in the base model
  - Assumes the future resembles the past — cannot predict demand shocks
  - Short history (2 years) limits seasonal decomposition accuracy
  - Synthetic input data means forecasts are illustrative, not operational
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _safe_prophet_import():
    """Lazy import of Prophet with a clear error message if missing."""
    try:
        from prophet import Prophet
        return Prophet
    except ImportError:
        raise ImportError(
            "Prophet not installed. Run: pip install prophet\n"
            "Note: Prophet requires pystan. On some systems: pip install prophet --no-deps"
        )


def forecast_corridor(
    rates_df: pd.DataFrame,
    corridor: str,
    rate_col: str = "ftl_rate_eur",
    forecast_weeks: int = 4,
    confidence_interval: float = 0.80,
    seasonality_mode: str = "multiplicative",
) -> pd.DataFrame:
    """
    Fits a Prophet model for one corridor and returns a combined
    historical + forecast DataFrame.

    Parameters
    ----------
    rates_df : pd.DataFrame
        Cleaned rates data (output of clean.run_cleaning).
    corridor : str
        Corridor ID e.g. "DE-NL".
    rate_col : str
        "ftl_rate_eur" or "ltl_rate_eur_pallet".
    forecast_weeks : int
        Periods to forecast ahead (default 4).
    confidence_interval : float
        Prophet uncertainty interval width (0.80 = 80%).
    seasonality_mode : str
        "multiplicative" (default, recommended for rates that scale with trend)
        or "additive".

    Returns
    -------
    pd.DataFrame with columns:
        date, y (actual), yhat (forecast), yhat_lower, yhat_upper,
        is_forecast (bool), corridor, rate_col
    """
    Prophet = _safe_prophet_import()

    corridor_df = rates_df[rates_df["corridor"] == corridor].copy()
    if len(corridor_df) < 10:
        raise ValueError(f"Insufficient data for {corridor}: {len(corridor_df)} rows.")

    # Prophet requires columns named 'ds' and 'y'
    prophet_df = corridor_df[["date", rate_col]].rename(
        columns={"date": "ds", rate_col: "y"}
    ).sort_values("ds")

    # Initialise and fit Prophet
    model = Prophet(
        interval_width=confidence_interval,
        seasonality_mode=seasonality_mode,
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        # Cap changepoints to avoid overfitting with 2 years of data
        changepoint_prior_scale=0.05,
    )

    # Suppress Prophet's verbose logging for cleaner pipeline output
    import logging as _logging
    _logging.getLogger("prophet").setLevel(_logging.WARNING)
    _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

    model.fit(prophet_df)

    # Generate future dates
    future = model.make_future_dataframe(periods=forecast_weeks, freq="W")
    forecast = model.predict(future)

    # Merge actuals and forecast
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result = result.merge(
        prophet_df.rename(columns={"y": "actual"}),
        on="ds",
        how="left",
    )

    # Label rows
    last_actual_date = prophet_df["ds"].max()
    result["is_forecast"] = result["ds"] > last_actual_date
    result["corridor"] = corridor
    result["rate_col"] = rate_col

    result = result.rename(columns={"ds": "date"})
    result[["yhat", "yhat_lower", "yhat_upper"]] = result[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].round(2)

    logger.info(
        f"Forecast complete: {corridor} | {rate_col} | "
        f"{forecast_weeks} weeks ahead | "
        f"Last actual: {last_actual_date.date()}"
    )
    return result


def forecast_all_corridors(
    rates_df: pd.DataFrame,
    rate_col: str = "ftl_rate_eur",
    forecast_weeks: int = 4,
) -> pd.DataFrame:
    """
    Runs forecast_corridor for every corridor in rates_df.
    Returns a single combined DataFrame.
    """
    results = []
    corridors = rates_df["corridor"].unique()
    logger.info(f"Forecasting {len(corridors)} corridors ({rate_col})...")

    for corridor in corridors:
        try:
            fc = forecast_corridor(rates_df, corridor, rate_col, forecast_weeks)
            results.append(fc)
        except Exception as e:
            logger.error(f"Forecast failed for {corridor}: {e}")

    if not results:
        raise RuntimeError("All corridor forecasts failed. Check Prophet installation.")

    combined = pd.concat(results, ignore_index=True)
    logger.info(f"Forecast complete: {len(combined)} total rows.")
    return combined


def get_forecast_summary(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a clean summary: corridor, current rate, 4w forecast, % change.
    Used in the dashboard insight panel.
    """
    records = []
    for corridor, grp in forecast_df.groupby("corridor"):
        actuals = grp[~grp["is_forecast"]].dropna(subset=["actual"])
        forecasts = grp[grp["is_forecast"]]

        if actuals.empty or forecasts.empty:
            continue

        current = actuals.iloc[-1]["actual"]
        future = forecasts.iloc[-1]["yhat"]
        lower = forecasts.iloc[-1]["yhat_lower"]
        upper = forecasts.iloc[-1]["yhat_upper"]
        change_pct = (future - current) / current * 100

        records.append({
            "corridor": corridor,
            "current_rate": round(current, 2),
            "forecast_4w": round(future, 2),
            "forecast_lower": round(lower, 2),
            "forecast_upper": round(upper, 2),
            "change_pct": round(change_pct, 1),
            "direction": "up" if change_pct > 1 else ("down" if change_pct < -1 else "flat"),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Fallback: ARIMA (statsmodels) — use if Prophet install fails
# ---------------------------------------------------------------------------

def forecast_corridor_arima(
    rates_df: pd.DataFrame,
    corridor: str,
    rate_col: str = "ftl_rate_eur",
    forecast_weeks: int = 4,
    order: tuple = (1, 1, 1),
) -> pd.DataFrame:
    """
    ARIMA fallback. Use this if Prophet cannot be installed.

    WHAT ARIMA DOES (for interviews):
    "ARIMA is a classical statistical forecasting model. The three parameters
    (p, d, q) control: how many past values to use, how many times to difference
    the series to make it stationary, and how many past forecast errors to
    incorporate. With (1,1,1) I'm saying: use the last observation, difference once
    to remove trend, and correct for last period's error. It's simple and
    interpretable, which matters in a business context."
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        raise ImportError("statsmodels required: pip install statsmodels")

    corridor_df = rates_df[rates_df["corridor"] == corridor].sort_values("date")
    ts = corridor_df.set_index("date")[rate_col]

    model = ARIMA(ts, order=order)
    fitted = model.fit()

    # Forecast
    forecast_result = fitted.get_forecast(steps=forecast_weeks)
    forecast_index = pd.date_range(
        start=ts.index[-1] + pd.Timedelta(weeks=1),
        periods=forecast_weeks,
        freq="W",
    )
    fc_mean = forecast_result.predicted_mean
    fc_ci = forecast_result.conf_int(alpha=0.20)  # 80% CI

    forecast_rows = pd.DataFrame({
        "date": forecast_index,
        "actual": np.nan,
        "yhat": fc_mean.values,
        "yhat_lower": fc_ci.iloc[:, 0].values,
        "yhat_upper": fc_ci.iloc[:, 1].values,
        "is_forecast": True,
        "corridor": corridor,
        "rate_col": rate_col,
    })

    historical_rows = corridor_df[["date", rate_col]].copy()
    historical_rows = historical_rows.rename(columns={rate_col: "actual"})
    historical_rows["yhat"] = fitted.fittedvalues.values[:len(historical_rows)]
    historical_rows["yhat_lower"] = np.nan
    historical_rows["yhat_upper"] = np.nan
    historical_rows["is_forecast"] = False
    historical_rows["corridor"] = corridor
    historical_rows["rate_col"] = rate_col

    return pd.concat([historical_rows, forecast_rows], ignore_index=True)
