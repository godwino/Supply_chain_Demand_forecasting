# Databricks Setup

This project already supports Databricks-backed MLflow through environment variables.

## 1. Authenticate

Set these environment variables in your shell or `.env`:

- `MLFLOW_TRACKING_URI=databricks`
- `DATABRICKS_HOST=https://<your-workspace-host>`
- `DATABRICKS_TOKEN=<your-personal-access-token>`
- `DATABRICKS_EXPERIMENT_PATH=/Shared/retail-demand-forecasting`

## 2. Prepare local real data

```bash
C:\Users\Osayamwen\anaconda3\python.exe data/ingest/download_real_data.py
```

## 3. Train while logging to Databricks MLflow

```bash
set MLFLOW_TRACKING_URI=databricks
set DATABRICKS_HOST=https://<your-workspace-host>
set DATABRICKS_TOKEN=<your-personal-access-token>
set DATABRICKS_EXPERIMENT_PATH=/Shared/retail-demand-forecasting
C:\Users\Osayamwen\anaconda3\python.exe train.py --data-path data/processed/online_retail_daily.csv --horizon 28 --experiment-name /Shared/retail-demand-forecasting
```

## 4. Register the best run in Databricks Model Registry

```bash
C:\Users\Osayamwen\anaconda3\python.exe register.py --run-id <RUN_ID> --alias champion
```

## Optional

- Use `databricks/train_on_databricks.py` inside a Databricks notebook or repo-backed workspace.
- Upload `data/processed/online_retail_daily.csv` to a Databricks volume or DBFS path if you want training to execute inside the workspace instead of locally.
