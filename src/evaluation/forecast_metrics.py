from __future__ import annotations

from typing import Iterable

import numpy as np


def _to_arrays(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(list(y_true), dtype=float)
    forecast = np.asarray(list(y_pred), dtype=float)
    if actual.shape != forecast.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return actual, forecast


def _safe_ratio(numerator: float, denominator: float) -> float:
    if np.isclose(denominator, 0.0):
        return 0.0
    return float(numerator / denominator)


def calculate_forecast_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Return standard forecasting metrics with safe divide-by-zero handling."""
    actual, forecast = _to_arrays(y_true, y_pred)
    error = forecast - actual
    abs_error = np.abs(error)

    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = np.where(np.abs(actual) > 0, np.abs(actual), np.nan)
    mape = float(np.nan_to_num(np.mean(abs_error / denom) * 100, nan=0.0))
    smape = float(
        np.nan_to_num(
            np.mean((2.0 * abs_error) / np.where((np.abs(actual) + np.abs(forecast)) > 0, np.abs(actual) + np.abs(forecast), np.nan))
            * 100,
            nan=0.0,
        )
    )
    total_actual_demand = float(np.sum(actual))
    total_forecast_demand = float(np.sum(forecast))
    bias = float(np.sum(error))
    bias_pct = _safe_ratio(bias, total_actual_demand)
    wape = _safe_ratio(float(np.sum(abs_error)), total_actual_demand)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "wape": wape,
        "forecast_bias": bias,
        "bias_percentage": bias_pct,
        "total_actual_demand": total_actual_demand,
        "total_forecast_demand": total_forecast_demand,
    }
