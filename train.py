from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import mlflow
import mlflow.sklearn
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from supply_chain_forecasting.config import ProjectConfig, ensure_directories
from supply_chain_forecasting.data import build_synthetic_dataset, dataset_fingerprint, load_dataset, train_validation_split
from supply_chain_forecasting.evaluation import business_metrics, regression_metrics
from supply_chain_forecasting.features import create_features, make_model_matrix
from supply_chain_forecasting.modeling import available_models, feature_importance_frame, fit_and_predict
from supply_chain_forecasting.reorder import add_reorder_recommendations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train demand forecasting models with MLflow tracking.")
    parser.add_argument("--data-path", default="data/sample/retail_demand_sample.csv")
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--service-level", type=float, default=0.95)
    parser.add_argument("--experiment-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig(forecast_horizon=args.horizon, reorder_service_level=args.service_level)
    ensure_directories(config)

    data_path = Path(args.data_path)
    if not data_path.exists():
        build_synthetic_dataset(data_path)

    raw_df = load_dataset(data_path)
    feature_df = create_features(raw_df)
    if feature_df.empty:
        if "data/sample" in data_path.as_posix():
            build_synthetic_dataset(data_path)
            raw_df = load_dataset(data_path)
            feature_df = create_features(raw_df)
        if feature_df.empty:
            raise ValueError(
                "Feature generation returned no rows. Provide at least 28 days of history per store-SKU."
            )
    train_df, valid_df = train_validation_split(feature_df, horizon=args.horizon)

    X_train, y_train = make_model_matrix(train_df)
    X_valid, y_valid = make_model_matrix(valid_df)
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)

    mlflow.set_tracking_uri(config.tracking_uri)
    experiment_name = args.experiment_name or os.getenv("DATABRICKS_EXPERIMENT_PATH") or config.experiment_name
    mlflow.set_experiment(experiment_name)

    dataset_hash = dataset_fingerprint(raw_df)
    best_run = {"name": None, "rmse": float("inf"), "run_id": None}

    with mlflow.start_run(run_name=f"forecast_h{args.horizon}_comparison") as parent_run:
        mlflow.set_tags(
            {
                "project": "retail-demand-forecasting",
                "dataset_hash": dataset_hash,
                "task": "demand_forecasting",
            }
        )
        mlflow.log_params(
            {
                "forecast_horizon": args.horizon,
                "service_level": args.service_level,
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
                "feature_count": X_train.shape[1],
            }
        )

        for model_name, (estimator, model_params) in available_models().items():
            with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                trained = fit_and_predict(model_name, estimator, X_train, y_train, X_valid)
                metrics = regression_metrics(y_valid, trained.predictions)

                prediction_frame = valid_df[["date", "store_id", "sku_id", "price", "promo", "on_hand_inventory", "lead_time_days", "unit_cost", "rolling_std_7"]].copy()
                prediction_frame["actual_sales"] = y_valid.values
                prediction_frame["forecast_units"] = trained.predictions.round(2)
                recommendation_frame = add_reorder_recommendations(
                    prediction_frame,
                    service_level=args.service_level,
                    horizon=args.horizon,
                )
                biz = business_metrics(recommendation_frame)

                mlflow.set_tags({"model_type": model_name, "dataset_hash": dataset_hash})
                mlflow.log_params(model_params)
                mlflow.log_metrics(metrics | biz)

                predictions_path = config.artifact_dir / f"{model_name}_predictions.csv"
                recommendation_frame.to_csv(predictions_path, index=False)
                mlflow.log_artifact(str(predictions_path), artifact_path="predictions")

                importance = feature_importance_frame(trained.estimator, X_train.columns.tolist())
                if importance is not None:
                    importance_path = config.artifact_dir / f"{model_name}_feature_importance.csv"
                    importance.to_csv(importance_path, index=False)
                    mlflow.log_artifact(str(importance_path), artifact_path="feature_importance")

                signature_input = X_train.head(5)
                mlflow.sklearn.log_model(
                    sk_model=trained.estimator,
                    artifact_path="model",
                    input_example=signature_input,
                    registered_model_name=None,
                    serialization_format="cloudpickle",
                )

                if metrics["rmse"] < best_run["rmse"]:
                    best_run = {"name": model_name, "rmse": metrics["rmse"], "run_id": child_run.info.run_id}
                    latest_path = config.artifact_dir / "latest_predictions.csv"
                    recommendation_frame.to_csv(latest_path, index=False)

        mlflow.log_dict(best_run, "best_run_summary.json")
        print(f"Best model: {best_run['name']} | RMSE={best_run['rmse']:.4f} | run_id={best_run['run_id']}")
        print(f"Parent run_id: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()
