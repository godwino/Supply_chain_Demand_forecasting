# Interview Talking Points

## 60-Second Version
I built an end-to-end retail demand forecasting and inventory replenishment project using real public retail data. It forecasts SKU-level demand, tracks experiments in MLflow, exposes results through a FastAPI service, and presents planner-oriented recommendations in a Streamlit dashboard.

## 2-Minute Technical Version
I started with raw retail transaction data, aggregated it into a daily forecasting dataset, engineered lag, rolling, and pricing features, and trained multiple models including baselines and tree-based regressors. I tracked runs in MLflow, evaluated performance with metrics such as WAPE, SMAPE, and forecast bias, and connected the forecasts to inventory planning logic like reorder point, safety stock, stockout risk, and days of cover. I also added backtesting, validation, monitoring, tests, and CI to make the project stronger from an enterprise data science perspective.

## Business Impact Explanation
The project shows how machine learning can move beyond forecast accuracy and support operational decisions that planners care about.

## How I Used MLflow
I used MLflow for experiment tracking, run comparison, artifacts, and model registry workflows.

## How The API Works
The API serves forecast and recommendation endpoints, including a planner-friendly recommendation route that returns reorder actions for selected SKU and store combinations.

## How The Dashboard Supports Planners
The dashboard lets a user inspect a SKU, review forecast vs actual, adjust scenario assumptions such as inventory and lead time, and see how the recommended reorder quantity changes.

## What I Would Improve In A Real Enterprise Environment
- Add probabilistic forecasting
- Integrate ERP or warehouse feeds
- Add automated retraining
- Support richer constraints such as expiry and supplier reliability
