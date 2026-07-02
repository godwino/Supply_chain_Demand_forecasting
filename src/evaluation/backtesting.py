from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from evaluation.forecast_metrics import calculate_forecast_metrics


@dataclass
class BacktestFoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: dict[str, float]


def rolling_backtest(
    data: pd.DataFrame,
    date_column: str,
    minimum_training_period: int,
    test_period_size: int,
    step_size: int,
    fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Run expanding-window backtesting while respecting time order."""
    if minimum_training_period <= 0 or test_period_size <= 0 or step_size <= 0:
        raise ValueError("Backtesting periods must be positive integers.")

    frame = data.sort_values(date_column).copy()
    unique_dates = pd.Index(sorted(frame[date_column].dropna().unique()))
    if len(unique_dates) < minimum_training_period + test_period_size:
        raise ValueError("Not enough history to run backtesting with the requested parameters.")

    results: list[dict[str, float | int | str]] = []
    fold = 1
    train_end_idx = minimum_training_period - 1

    while train_end_idx + test_period_size < len(unique_dates):
        train_dates = unique_dates[: train_end_idx + 1]
        test_dates = unique_dates[train_end_idx + 1 : train_end_idx + 1 + test_period_size]
        train_df = frame[frame[date_column].isin(train_dates)].copy()
        test_df = frame[frame[date_column].isin(test_dates)].copy()

        y_true, y_pred = fit_predict_fn(train_df, test_df)
        metrics = calculate_forecast_metrics(y_true, y_pred)
        results.append(
            {
                "fold": fold,
                "train_start": str(train_dates.min()),
                "train_end": str(train_dates.max()),
                "test_start": str(test_dates.min()),
                "test_end": str(test_dates.max()),
                **metrics,
            }
        )
        fold += 1
        train_end_idx += step_size

    return pd.DataFrame(results)
