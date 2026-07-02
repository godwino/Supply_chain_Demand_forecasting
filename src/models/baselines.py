from __future__ import annotations

import pandas as pd


def naive_forecast(series: pd.Series) -> pd.Series:
    return series.shift(1)


def moving_average_forecast(series: pd.Series, window: int = 7) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).mean()


def seasonal_naive_forecast(series: pd.Series, seasonal_period: int = 7) -> pd.Series:
    return series.shift(seasonal_period)


def add_baseline_forecasts(
    df: pd.DataFrame,
    group_columns: list[str],
    target_column: str,
    moving_average_window: int = 7,
    seasonal_period: int = 7,
) -> pd.DataFrame:
    frame = df.sort_values(group_columns + ["date"]).copy()
    grouped = frame.groupby(group_columns)[target_column]
    frame["naive_forecast"] = grouped.transform(naive_forecast)
    frame["moving_average_forecast"] = grouped.transform(lambda s: moving_average_forecast(s, window=moving_average_window))
    frame["seasonal_naive_forecast"] = grouped.transform(lambda s: seasonal_naive_forecast(s, seasonal_period=seasonal_period))
    return frame
