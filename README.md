# Retail Demand Forecasting and Inventory Replenishment with MLflow

This project is a production-style starter for retail demand forecasting and reorder recommendation. It combines time-series feature engineering, multi-model training, MLflow experiment tracking, model registry workflows, API serving, and a planner-friendly dashboard.

## What it does

- Forecasts next `7`, `14`, or `28` days of demand at `store_id` + `sku_id` level
- Compares multiple models: naive baseline, gradient boosting, and optional `XGBoost` / `LightGBM`
- Tracks every training run in MLflow with params, metrics, artifacts, and dataset fingerprint
- Produces business outputs:
  - `forecast_units`
  - `reorder_qty`
  - `stockout_risk`
  - `days_of_cover`
- Serves predictions through FastAPI
- Shows planner-ready recommendations in Streamlit

## Repository Layout

```text
data/
  sample/                 # synthetic starter dataset
  ingest/
databricks/
  train_on_databricks.py
src/
  supply_chain_forecasting/
    config.py
    data.py
    evaluation.py
    features.py
    modeling.py
    reorder.py
train.py
evaluate.py
register.py
serve/
  app.py
dashboard/
  app.py
artifacts/
```

## Dataset Options

- Real public dataset wired into this repo: UCI Online Retail
- Recommended future upgrade for deeper hierarchy: M5 Forecasting
- Included for fallback quick start: synthetic sample in `data/sample/retail_demand_sample.csv`

To download and prepare the real public dataset:

```bash
C:\Users\Osayamwen\anaconda3\python.exe data/ingest/download_real_data.py
```

This script downloads the Excel file from the UCI repository, cleans the transactions, aggregates them to daily SKU-country demand, keeps the highest-volume series for a tractable local training set, and writes:

- `data/raw/online_retail.xlsx`
- `data/processed/online_retail_daily.csv`

Notes on realism:

- `sales`, `price`, `date`, `sku_id`, and `store_id` come from the real dataset
- `store_id` uses `Country` as a retail geography proxy because the dataset does not expose physical store IDs
- `promo`, `on_hand_inventory`, `lead_time_days`, and `unit_cost` are derived operational features so the replenishment layer can run end-to-end

To adapt M5 later, map its store-item daily demand data into this project schema:

- `date`
- `store_id`
- `sku_id`
- `sales`
- `price`
- `promo`
- `on_hand_inventory`
- `lead_time_days`
- `unit_cost`

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download and prepare the real dataset:

```bash
C:\Users\Osayamwen\anaconda3\python.exe data/ingest/download_real_data.py
```

4. Train models:

```bash
C:\Users\Osayamwen\anaconda3\python.exe train.py --data-path data/processed/online_retail_daily.csv --horizon 28
```

5. Evaluate the most recent champion-ready artifact:

```bash
python evaluate.py --predictions-path artifacts/latest_predictions.csv
```

6. Register a trained model in MLflow Model Registry:

```bash
python register.py --run-id <RUN_ID> --alias challenger
```

7. Start the API:

```bash
uvicorn serve.app:app --reload
```

8. Start the dashboard:

```bash
streamlit run dashboard/app.py
```

## Databricks Option

If you want MLflow tracking and model registry in a Databricks workspace instead of local SQLite:

```bash
set MLFLOW_TRACKING_URI=databricks
set DATABRICKS_HOST=https://<your-workspace-host>
set DATABRICKS_TOKEN=<your-personal-access-token>
set DATABRICKS_EXPERIMENT_PATH=/Shared/retail-demand-forecasting
C:\Users\Osayamwen\anaconda3\python.exe train.py --data-path data/processed/online_retail_daily.csv --horizon 28 --experiment-name /Shared/retail-demand-forecasting
```

See [databricks/README.md](C:/Users/Osayamwen/Desktop/supply_chain/databricks/README.md) for the workspace flow.

## MLflow

Default tracking URI:

```text
sqlite:///mlflow.db
```

Useful commands:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

What gets logged:

- model type and hyperparameters
- train / validation metrics
- dataset fingerprint
- feature importance artifact when available
- validation predictions
- planner reorder recommendations

## Real Dataset Source

- UCI Online Retail Excel file:
  https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx

## API Endpoints

- `GET /health`
- `GET /model-info`
- `POST /forecast`
- `POST /predict`
- `POST /recommendations`

## Example API Requests

`POST /forecast`

```json
{
  "records": [
    {
      "date": "2024-06-01",
      "store_id": "CA_1",
      "sku_id": "SKU_001",
      "sales": 18,
      "price": 8.99,
      "promo": 1,
      "on_hand_inventory": 42,
      "lead_time_days": 5,
      "unit_cost": 3.2
    }
  ],
  "horizon": 14,
  "service_level": 0.95
}
```

`POST /recommendations`

```json
{
  "items": [
    {
      "store_id": "UNITED_KINGDOM",
      "sku_id": "85123A"
    },
    {
      "store_id": "UNITED_KINGDOM",
      "sku_id": "85099B",
      "as_of_date": "2011-12-09"
    }
  ],
  "horizon": 28,
  "service_level": 0.95
}
```

## Next Iterations

- Replace synthetic data with M5 raw inputs
- Add scheduled retraining
- Add drift checks on demand, promo rate, and price distribution
- Promote `challenger` and `champion` aliases automatically after approval
- Containerize API with MLflow model serving or Docker Compose
