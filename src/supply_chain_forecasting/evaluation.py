from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_true) - np.asarray(y_pred)
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    mape = float(np.mean(np.abs(error) / np.clip(np.abs(y_true), 1, None)) * 100)
    bias = float(np.mean(y_pred - y_true))
    return {"mae": mae, "rmse": rmse, "mape": mape, "bias": bias}


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
