"""
run_pipeline.py
---------------
Single entry point for the data pipeline.
Run this once to generate processed data files used by the Streamlit app.

Usage:
    python pipeline/run_pipeline.py

Output:
    data/processed/corridor_rates.parquet
    data/processed/fuel_index.parquet
    data/processed/volume_index.parquet
    data/processed/forecasts_ftl.parquet
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import run_ingestion
from src.clean import run_cleaning
from src.forecast import forecast_all_corridors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline/pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("FREIGHT RATE BENCHMARKING — DATA PIPELINE")
    logger.info("=" * 60)

    # Step 1: Ingest
    logger.info("Step 1/3: Ingesting data...")
    raw = run_ingestion()

    # Step 2: Clean
    logger.info("Step 2/3: Cleaning and normalising...")
    clean = run_cleaning(raw)

    # Step 3: Save processed data
    logger.info("Step 3/3: Saving processed data...")
    clean["rates"].to_parquet(output_dir / "corridor_rates.parquet", index=False)
    clean["diesel"].to_parquet(output_dir / "fuel_index.parquet", index=False)
    clean["volumes"].to_parquet(output_dir / "volume_index.parquet", index=False)
    logger.info("Saved: corridor_rates.parquet, fuel_index.parquet, volume_index.parquet")

    # Step 4: Forecast (optional — may take 1-2 min due to Prophet fitting)
    logger.info("Step 4/4: Running Prophet forecasts (this takes ~60-90 seconds)...")
    try:
        forecasts = forecast_all_corridors(clean["rates"], rate_col="ftl_rate_eur", forecast_weeks=4)
        forecasts.to_parquet(output_dir / "forecasts_ftl.parquet", index=False)
        logger.info("Saved: forecasts_ftl.parquet")
    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
        logger.warning("Dashboard will run in non-forecast mode. Check Prophet installation.")

    logger.info("=" * 60)
    logger.info("Pipeline complete. Run: streamlit run app/streamlit_app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
