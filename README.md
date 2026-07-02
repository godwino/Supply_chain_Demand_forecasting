# Retail Demand Forecasting and Inventory Replenishment with MLflow

## Project Overview
This project is an end-to-end retail demand forecasting and inventory replenishment application built to show how supply chain data science can move from model training to planner decision support. It forecasts SKU-level demand and converts forecast outputs into practical replenishment decisions such as reorder quantity, safety stock, reorder point, stockout risk, and days of cover.

## Business Problem
Retail and healthcare distributors need to maintain enough inventory to meet customer demand without tying up too much working capital in excess stock. Poor forecasts can lead to stockouts, missed service-level targets, excess inventory, waste, and poor replenishment decisions. This project forecasts SKU-level demand and converts those predictions into inventory planning recommendations that a planner can use.

## Why This Matters in Supply Chain
Demand forecasts matter only when they improve operational decisions. This project connects forecasting outputs to supply chain actions such as replenishment prioritization, inventory coverage assessment, and service-level risk management.

## Dataset
- Public dataset: UCI Online Retail
- Data source: transaction-level retail sales
- Modeling grain: daily demand by geography proxy (`store_id`) and SKU
- Realistic note: the project uses retail data, not proprietary healthcare or McKesson data

## Data Pipeline
- Download raw Excel data with `data/ingest/download_real_data.py`
- Clean transactions and aggregate to daily demand
- Retain high-signal SKU series for tractable local experimentation
- Save processed data to `data/processed/online_retail_daily.csv`

## Feature Engineering
- Demand lags: 1, 7, 14, 28 days
- Rolling means and standard deviation
- Day-of-week, week-of-year, and month features
- Price change features
- Promo proxy and inventory-related fields

## Models Used
- Naive last value
- HistGradientBoostingRegressor
- RandomForestRegressor
- LightGBM
- XGBoost

## Baseline Models
- Naive forecast
- Moving average forecast
- Seasonal naive forecast

## Model Evaluation
The project now supports:
- MAE
- RMSE
- MAPE
- SMAPE
- WAPE
- Forecast bias
- Bias percentage
- Total actual demand
- Total forecast demand

It also includes expanding-window backtesting so forecasting performance is evaluated using time-aware folds instead of random train/test splits.

## Inventory Planning Logic
- `days_of_cover = on_hand_inventory / average_daily_forecast`
- `expected_demand_during_lead_time = average_daily_forecast * lead_time_days`
- `safety_stock = z_score * demand_std * sqrt(lead_time_days)`
- `reorder_point = expected_demand_during_lead_time + safety_stock`
- `reorder_quantity` recommends replenishment when inventory falls below reorder point
- `stockout_risk` is a bounded risk score between 0 and 1

## Dashboard Features
- Planner workbench focused on store, SKU, and date
- Scenario testing for inventory, lead time, and service level
- Ranked action queue for high-priority SKUs
- SKU explorer for forecast vs actual history
- Business summary view for operational reporting

## API Endpoints
- `GET /health`
- `GET /model-info`
- `POST /forecast`
- `POST /predict`
- `POST /recommendations`

## MLflow Tracking
- Experiment comparison across models
- Run parameters and metrics
- Dataset fingerprint logging
- Feature importance artifacts
- Prediction artifacts
- Model registry support for challenger and champion workflows

## Monitoring Plan
- Recent forecast MAE and WAPE
- Forecast bias tracking
- Demand drift and prediction drift checks
- SKU mix changes
- High stockout risk counts
- API health placeholders and recommended alert thresholds

## Assumptions and Limitations
- Public data is retail transaction data, not actual McKesson or pharmaceutical data.
- On-hand inventory and lead time assumptions may be simulated or user-adjustable.
- The project demonstrates how forecast outputs can support replenishment decisions.
- It is an applied MVP, not a full enterprise production system.

## How This Applies to Pharmaceutical Distribution
Although this project uses public retail transaction data, the same forecasting and replenishment logic can support pharmaceutical distribution use cases such as stockout risk monitoring, service-level planning, days-of-supply analysis, inventory prioritization, and replenishment recommendations. In a healthcare setting, additional constraints such as expiry dates, cold-chain requirements, supplier lead time variability, regulatory handling rules, and patient/customer impact would also be considered.

## How to Run Locally
```bash
pip install -r requirements.txt
python data/ingest/download_real_data.py
python train.py --data-path data/processed/online_retail_daily.csv --horizon 28
python scripts/run_baseline_evaluation.py
python scripts/run_backtesting.py
python scripts/run_data_validation.py
python scripts/run_monitoring_report.py
streamlit run dashboard/app.py
uvicorn serve.app:app --reload
```

## How to Run with Docker
```bash
docker compose up --build
```

## Repository Structure
```text
src/
  business/
  evaluation/
  models/
  monitoring/
  supply_chain_forecasting/
  validation/
scripts/
dashboard/
serve/
data/
docs/
reports/
tests/
```

## Future Improvements
- Probabilistic forecasting
- Automated retraining
- ERP and warehouse system integration
- Expiry-aware and cold-chain-aware planning logic
- Role-based dashboard access
