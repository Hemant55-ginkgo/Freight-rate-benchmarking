# 🚛 European Freight Rate Benchmarking Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A data engineering and analytics project that benchmarks FTL (full truckload) and LTL (less-than-truckload) freight rate trends across six key European corridors, with a 4-week ML forecast layer built using Facebook Prophet.

**Live demo →** [your-app.streamlit.app](https://your-app.streamlit.app)

---

## What this project does

European road freight rates are opaque. Shippers negotiate contracts in the dark, often without visibility into whether their rates are above or below market. This tool provides a data-driven benchmarking framework that:

- Tracks weekly FTL/LTL rate trends across corridors: DE–NL, DE–BE, NL–FR, PL–DE, DE–FR, BE–LU
- Contextualises rates against 90-day averages, 52-week ranges, and year-on-year trends
- Models the 4-week rate outlook using Facebook Prophet time series forecasting
- Surfaces cost driver signals: diesel price changes (ECB), freight volume trends (Eurostat)
- Generates automated insight summaries in plain English

---

## Why I built this

I spent seven years working in Supply Chain digitalization at DB Schenker, Bosch, Siemens, and Delivery Hero. Recently, I observed a persistent gap: even sophisticated shippers lacked accessible, structured benchmarking for European road freight rates. The tools that exist are either enterprise paywalled (Transporeon Rate Benchmarking, Xeneta Road) or too aggregated to be useful at the corridor level.

This project is my attempt to demonstrate what a lightweight, open, data-driven benchmarking framework could look like — and to validate the analytical approach against public data sources.

---

## Corridors monitored

| Corridor | Distance | Base Rate Range | Key characteristics |
|----------|----------|-----------------|---------------------|
| Germany → Netherlands | 650 km | €950–€1,050 | High frequency, automotive + chemicals |
| Germany → Belgium | 750 km | €1,050–€1,150 | Port access, FMCG |
| Netherlands → France | 590 km | €930–€1,020 | Complex tolls, driver availability |
| Poland → Germany | 580 km | €650–€750 | High volume E-W, price sensitive |
| Germany → France | 1,050 km | €1,400–€1,550 | Long haul, high diesel sensitivity |
| Belgium → Luxembourg | 220 km | €380–€420 | Short haul premium, B2G |

---

## Data sources and methodology

### Rate data

Live FTL/LTL rate APIs from platforms like Transporeon, Timocom, or Xeneta are enterprise-tier and not publicly accessible. This project uses a **cost-component modelling approach** — the same methodology used by logistics market research firms and the IRU (International Road Union) in their Motor Transport Cost Index:

```
rate = base_rate
     × (1 + 0.30 × diesel_index_delta)     # fuel ~30% of FTL cost
     × seasonality_index[month]              # monthly seasonal pattern
     × (1 + demand_shock)                   # stochastic demand signal
```

Base rates per corridor are calibrated against IRU 2023 benchmarks and Transporeon's published European Road Market Benchmark reports. Synthetic data is clearly flagged in the dashboard.

**This is transparently disclosed throughout the project.** The value is in the methodology and framework, not the pretence of having live rate feeds.

### Public data sources

| Source | Dataset | Access | Use in this project |
|--------|---------|--------|---------------------|
| European Central Bank | Petroleum products PPI (`ICP/M.U2.N.EF0000.4.ANR`) | Free REST API | Diesel cost driver |
| Eurostat | Road freight volumes by partner country (`road_go_ta_tott`) | Free REST API | Demand-side signal |
| Freightos Baltic Index | FBX global container rate index | CSV download | Macro logistics context |
| IRU / Transporeon | Published market reports | PDF (manual) | Rate calibration benchmarks |

### Forecast model

4-week rate forecasts are produced using **Facebook Prophet**, a decomposable time series model:

- **Trend component**: captures the directional movement of rates over time
- **Seasonality component**: models weekly and annual patterns (Q4 peak, summer trough)
- **Confidence intervals**: 80% uncertainty bands — the actual rate has historically fallen inside this range ~80% of the time

Prophet was chosen over ARIMA for this application because:
1. It handles missing data and outliers gracefully (relevant for sparse historical rate data)
2. The seasonal decomposition is interpretable without deep statistics background
3. It does not require stationarity preprocessing

An ARIMA fallback is included in `src/forecast.py` for environments where Prophet cannot be installed.

**Limitations (explicitly acknowledged):**
- No exogenous variables (capacity, geopolitical events) in the base model
- Assumes future patterns resemble historical patterns
- Synthetic input data means forecasts are illustrative, not operational
- Short history (2 years) limits annual seasonality accuracy

---

## Technical architecture

```
freight-rate-benchmarking/
├── app/
│   └── streamlit_app.py        # Streamlit dashboard
├── src/
│   ├── ingest.py               # Data fetching + synthetic rate generation
│   ├── clean.py                # Normalisation, outlier handling, derived metrics
│   ├── corridors.py            # Corridor aggregation and benchmarking
│   ├── forecast.py             # Prophet / ARIMA forecasting
│   └── insights.py             # Automated insight text generation
├── pipeline/
│   └── run_pipeline.py         # Orchestrates full data pipeline
├── data/
│   ├── raw/                    # Downloaded API responses (not committed)
│   └── processed/              # Parquet files (committed as sample data)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_validation.ipynb
│   └── 03_corridor_analysis.ipynb
└── docs/
    └── methodology.md
```

**Stack:** Python 3.11 · Streamlit · Prophet · Plotly · pandas · pyarrow · requests

---

## Running locally

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/freight-rate-benchmarking.git
cd freight-rate-benchmarking

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
.\venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Run the data pipeline (generates processed data files)
# This step fetches from ECB/Eurostat APIs (~30 seconds)
# Prophet fitting takes ~60-90 seconds on first run
python pipeline/run_pipeline.py

# Launch the dashboard
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### Running without the pipeline

The Streamlit app generates data on-the-fly if processed files are not found. This enables deployment on Streamlit Cloud without a separate pipeline step. First load will be slower (~30 seconds for data generation, longer if forecasting is enabled).

---

## Deploying to Streamlit Cloud

1. Fork this repository
2. Log in at [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repo
4. Set **Main file path** to `app/streamlit_app.py`
5. Deploy — no environment variables required for the base configuration

Note: Prophet has a heavy dependency chain (pystan/cmdstanpy). If deployment times out, the app falls back to ARIMA forecasting automatically.

---

## Key design decisions

**Why model rates instead of scraping?**
Scraping commercial freight platforms violates their terms of service and produces legally and ethically problematic data. The cost-component modelling approach is transparent, methodologically sound, and mirrors how logistics analysts actually build market views when proprietary data is unavailable.

**Why Streamlit instead of Power BI?**
Streamlit produces a portable, publicly deployable web application that can be linked from a GitHub repo and viewed by anyone without software installation. Power BI requires Pro licensing for external sharing. For logistics tech companies evaluating technical PM candidates, a deployed Streamlit app signals more clearly than a Power BI file.

**Why Prophet over ARIMA?**
Freight rates exhibit multiplicative seasonality (Q4 rates are not just +X€ vs average, they're +X% — the seasonal effect scales with the rate level). Prophet handles multiplicative seasonality cleanly. An ARIMA fallback is included for completeness.

---

## Extending this project

Potential next steps for a production version:

- **Live rate feeds**: Integrate with Freighton API or Transporeon open data when available
- **Carrier cost model**: Add driver wage indices (ETUC data), tolls, ferry costs
- **Tender vs spot spread**: Model the gap between contract and spot rates by corridor
- **Capacity signal**: Scrape load-to-truck ratios from TIMOCOM public data where available
- **Alerting**: Add email/Slack alerts when corridor rates cross defined thresholds
- **LLM insight layer**: Replace rule-based insight generation with GPT/Claude API

---

## About

Built by **[Your Name]** — logistics digitalisation professional with 7 years of experience at DB Schenker, Bosch, Siemens, and Delivery Hero. Background in transport operations, digital procurement, and supply chain analytics.

This project is part of a broader portfolio demonstrating applied data skills in the European logistics tech space.

**Connect:** [LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Data from ECB and Eurostat is subject to their respective open data policies.
Freightos Baltic Index data is subject to Freightos terms of use.

---

*If you work in logistics and want to discuss the methodology, rate dynamics, or how this could be built into a production-grade tool — I'd be happy to connect.*
