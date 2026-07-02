from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.forecast_metrics import calculate_forecast_metrics


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    metrics = calculate_forecast_metrics(y_true, y_pred)
    return {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "smape": metrics["smape"],
        "wape": metrics["wape"],
        "bias": metrics["forecast_bias"],
        "bias_percentage": metrics["bias_percentage"],
        "total_actual_demand": metrics["total_actual_demand"],
        "total_forecast_demand": metrics["total_forecast_demand"],
    }


def business_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    stockout_rate = float((predictions["stockout_risk"] >= 0.5).mean())
    avg_reorder = float(predictions["reorder_qty"].mean())
    avg_cover = float(predictions["days_of_cover"].mean())
    projected_revenue = float((predictions["forecast_units"] * predictions["price"]).sum())
    projected_margin = float((predictions["forecast_units"] * (predictions["price"] - predictions["unit_cost"])).sum())
    return {
        "stockout_risk_rate": stockout_rate,
        "avg_reorder_qty": avg_reorder,
        "avg_days_of_cover": avg_cover,
        "projected_revenue": projected_revenue,
        "projected_margin": projected_margin,
    }


def save_metrics_report(metrics: dict[str, float], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics).sort_index().to_csv(output, header=["value"])
    return output
