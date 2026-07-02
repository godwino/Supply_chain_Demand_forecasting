from __future__ import annotations

import pandas as pd

from evaluation.forecast_metrics import calculate_forecast_metrics


def recent_forecast_performance(df: pd.DataFrame, window: int = 50) -> dict[str, float]:
    recent = df.sort_values("date").tail(window)
    metrics = calculate_forecast_metrics(recent["actual_sales"], recent["forecast_units"])
    return {
        "recent_mae": metrics["mae"],
        "recent_wape": metrics["wape"],
        "recent_forecast_bias": metrics["forecast_bias"],
        "recent_bias_percentage": metrics["bias_percentage"],
    }


def prediction_drift_summary(df: pd.DataFrame) -> dict[str, float]:
    frame = df.sort_values("date")
    midpoint = len(frame) // 2
    early = frame.iloc[:midpoint]
    late = frame.iloc[midpoint:]
    return {
        "prediction_mean_shift": float(late["forecast_units"].mean() - early["forecast_units"].mean()),
        "prediction_std_shift": float(late["forecast_units"].std(ddof=0) - early["forecast_units"].std(ddof=0)),
    }


def demand_drift_summary(df: pd.DataFrame) -> dict[str, float]:
    frame = df.sort_values("date")
    midpoint = len(frame) // 2
    early = frame.iloc[:midpoint]
    late = frame.iloc[midpoint:]
    return {
        "demand_mean_shift": float(late["actual_sales"].mean() - early["actual_sales"].mean()),
        "demand_std_shift": float(late["actual_sales"].std(ddof=0) - early["actual_sales"].std(ddof=0)),
    }


def sku_mix_change(df: pd.DataFrame, top_n: int = 10) -> dict[str, str]:
    frame = df.sort_values("date")
    midpoint = len(frame) // 2
    early_top = set(frame.iloc[:midpoint].groupby("sku_id")["actual_sales"].sum().nlargest(top_n).index)
    late_top = set(frame.iloc[midpoint:].groupby("sku_id")["actual_sales"].sum().nlargest(top_n).index)
    return {"sku_mix_overlap": f"{len(early_top & late_top)}/{top_n}"}


def high_stockout_risk_count(df: pd.DataFrame, threshold: float = 0.5) -> dict[str, int]:
    return {"high_stockout_risk_count": int((df["stockout_risk"] >= threshold).sum())}


def api_style_output_summary(df: pd.DataFrame) -> dict[str, float]:
    return {
        "records_scored": float(len(df)),
        "avg_reorder_qty": float(df["reorder_qty"].mean()),
        "avg_days_of_cover": float(df["days_of_cover"].mean()),
    }
