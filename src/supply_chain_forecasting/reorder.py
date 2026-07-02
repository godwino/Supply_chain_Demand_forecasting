from __future__ import annotations

import numpy as np
import pandas as pd


def add_reorder_recommendations(
    frame: pd.DataFrame,
    service_level: float = 0.95,
    horizon: int = 28,
) -> pd.DataFrame:
    z_score = 1.65 if service_level >= 0.95 else 1.28
    result = frame.copy()

    demand_std = result.get("rolling_std_7", pd.Series(np.repeat(0.0, len(result))))
    daily_forecast = np.clip(result["forecast_units"] / max(horizon, 1), 0, None)
    safety_stock = z_score * demand_std.fillna(0) * np.sqrt(result["lead_time_days"].clip(lower=1))
    lead_time_demand = daily_forecast * result["lead_time_days"].clip(lower=1)
    reorder_point = lead_time_demand + safety_stock

    result["safety_stock"] = safety_stock.round(2)
    result["reorder_point"] = reorder_point.round(2)
    result["reorder_qty"] = np.clip(reorder_point - result["on_hand_inventory"], 0, None).round(0).astype(int)
    result["days_of_cover"] = np.where(daily_forecast > 0, result["on_hand_inventory"] / daily_forecast, np.inf)
    result["stockout_risk"] = np.clip((reorder_point - result["on_hand_inventory"]) / np.maximum(reorder_point, 1), 0, 1).round(3)
    return result
