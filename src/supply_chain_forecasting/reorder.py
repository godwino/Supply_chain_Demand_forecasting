from __future__ import annotations

import numpy as np
import pandas as pd

from business.inventory_planning import (
    days_of_cover,
    reorder_point as calc_reorder_point,
    reorder_quantity as calc_reorder_quantity,
    safety_stock as calc_safety_stock,
    stockout_risk as calc_stockout_risk,
)

def add_reorder_recommendations(
    frame: pd.DataFrame,
    service_level: float = 0.95,
    horizon: int = 28,
) -> pd.DataFrame:
    result = frame.copy()
    daily_forecast = np.clip(result["forecast_units"] / max(horizon, 1), 0, None)
    demand_std = result.get("rolling_std_7", pd.Series(np.repeat(0.0, len(result)))).fillna(0.0)

    result["safety_stock"] = [
        round(calc_safety_stock(std, lead_time, service_level), 2)
        for std, lead_time in zip(demand_std, result["lead_time_days"], strict=False)
    ]
    result["reorder_point"] = [
        round(calc_reorder_point(avg_daily, lead_time, ss), 2)
        for avg_daily, lead_time, ss in zip(daily_forecast, result["lead_time_days"], result["safety_stock"], strict=False)
    ]
    result["reorder_qty"] = [
        calc_reorder_quantity(on_hand, avg_daily, lead_time, ss, target_stock_days=horizon)
        for on_hand, avg_daily, lead_time, ss in zip(
            result["on_hand_inventory"], daily_forecast, result["lead_time_days"], result["safety_stock"], strict=False
        )
    ]
    result["days_of_cover"] = [days_of_cover(on_hand, avg_daily) for on_hand, avg_daily in zip(result["on_hand_inventory"], daily_forecast, strict=False)]
    result["stockout_risk"] = [
        round(calc_stockout_risk(on_hand, avg_daily, lead_time, ss), 3)
        for on_hand, avg_daily, lead_time, ss in zip(
            result["on_hand_inventory"], daily_forecast, result["lead_time_days"], result["safety_stock"], strict=False
        )
    ]
    return result
