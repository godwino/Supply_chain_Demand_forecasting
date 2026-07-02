from __future__ import annotations

from math import sqrt

import numpy as np


SERVICE_LEVEL_Z = {
    0.80: 0.84,
    0.85: 1.04,
    0.90: 1.28,
    0.95: 1.65,
    0.97: 1.88,
    0.99: 2.33,
}


def _closest_service_z(service_level: float) -> float:
    closest_key = min(SERVICE_LEVEL_Z, key=lambda k: abs(k - service_level))
    return SERVICE_LEVEL_Z[closest_key]


def days_of_cover(on_hand_inventory: float, average_daily_forecast: float) -> float:
    if average_daily_forecast <= 0:
        return float("inf")
    return float(on_hand_inventory / average_daily_forecast)


def safety_stock(demand_std: float, lead_time_days: float, service_level: float = 0.95) -> float:
    if demand_std <= 0 or lead_time_days <= 0:
        return 0.0
    return float(_closest_service_z(service_level) * demand_std * sqrt(lead_time_days))


def expected_demand_during_lead_time(average_daily_forecast: float, lead_time_days: float) -> float:
    return float(max(average_daily_forecast, 0.0) * max(lead_time_days, 0.0))


def reorder_point(average_daily_forecast: float, lead_time_days: float, safety_stock_value: float) -> float:
    return float(expected_demand_during_lead_time(average_daily_forecast, lead_time_days) + max(safety_stock_value, 0.0))


def reorder_quantity(
    on_hand_inventory: float,
    average_daily_forecast: float,
    lead_time_days: float,
    safety_stock_value: float,
    target_stock_days: float | None = None,
) -> int:
    point = reorder_point(average_daily_forecast, lead_time_days, safety_stock_value)
    target_level = point
    if target_stock_days is not None and target_stock_days > 0:
        target_level = max(target_level, average_daily_forecast * target_stock_days + safety_stock_value)
    if on_hand_inventory >= point:
        return 0
    return int(round(max(target_level - on_hand_inventory, 0.0)))


def stockout_risk(on_hand_inventory: float, average_daily_forecast: float, lead_time_days: float, safety_stock_value: float) -> float:
    point = reorder_point(average_daily_forecast, lead_time_days, safety_stock_value)
    if point <= 0:
        return 0.0
    risk = (point - on_hand_inventory) / point
    return float(np.clip(risk, 0.0, 1.0))
