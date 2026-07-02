from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.forecast_metrics import calculate_forecast_metrics
from models.baselines import add_baseline_forecasts


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "processed" / "online_retail_daily.csv", parse_dates=["date"])
    frame = add_baseline_forecasts(df, group_columns=["store_id", "sku_id"], target_column="sales")
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for column in ["naive_forecast", "moving_average_forecast", "seasonal_naive_forecast"]:
        subset = frame.dropna(subset=[column]).copy()
        metrics = calculate_forecast_metrics(subset["sales"], subset[column])
        metrics["model"] = column
        rows.append(metrics)

    pd.DataFrame(rows).to_csv(reports_dir / "baseline_metrics.csv", index=False)
    print(f"Saved baseline metrics to {reports_dir / 'baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
