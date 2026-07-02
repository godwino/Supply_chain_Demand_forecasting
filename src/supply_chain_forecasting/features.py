from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "price",
    "promo",
    "on_hand_inventory",
    "lead_time_days",
    "unit_cost",
    "day_of_week",
    "week_of_year",
    "month",
    "is_month_start",
    "is_month_end",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "price_change_7",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.sort_values(["store_id", "sku_id", "date"]).copy()
    group = frame.groupby(["store_id", "sku_id"])

    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["month"] = frame["date"].dt.month
    frame["is_month_start"] = frame["date"].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame["date"].dt.is_month_end.astype(int)

    for lag in (1, 7, 14, 28):
        frame[f"lag_{lag}"] = group["sales"].shift(lag)

    frame["rolling_mean_7"] = group["sales"].transform(lambda s: s.shift(1).rolling(window=7).mean())
    frame["rolling_mean_28"] = group["sales"].transform(lambda s: s.shift(1).rolling(window=28).mean())
    frame["rolling_std_7"] = group["sales"].transform(lambda s: s.shift(1).rolling(window=7).std())
    frame["price_change_7"] = group["price"].transform(lambda s: s.pct_change(periods=7))

    frame = frame.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return frame


def make_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    matrix = pd.get_dummies(df[["store_id", "sku_id", *FEATURE_COLUMNS]], columns=["store_id", "sku_id"], drop_first=False)
    target = df["sales"].astype(float)
    return matrix, target
