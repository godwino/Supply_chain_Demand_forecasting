from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from supply_chain_forecasting.evaluation import business_metrics, regression_metrics, save_metrics_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved demand forecasts.")
    parser.add_argument("--predictions-path", default="artifacts/latest_predictions.csv")
    parser.add_argument("--output-path", default="artifacts/evaluation_metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions_path)

    if "actual_sales" not in predictions.columns or "forecast_units" not in predictions.columns:
        raise ValueError("Predictions file must contain 'actual_sales' and 'forecast_units'.")

    metrics = regression_metrics(predictions["actual_sales"], predictions["forecast_units"])
    metrics.update(business_metrics(predictions))
    report_path = save_metrics_report(metrics, args.output_path)

    print("Evaluation metrics")
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.4f}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
