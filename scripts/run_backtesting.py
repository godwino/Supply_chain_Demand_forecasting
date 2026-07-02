from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.backtesting import rolling_backtest
from supply_chain_forecasting.features import create_features, make_model_matrix


def fit_predict_hist_gbm(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    combined = pd.concat([train_df, test_df], ignore_index=True)
    feature_df = create_features(combined)
    train_dates = set(pd.to_datetime(train_df["date"]))
    test_dates = set(pd.to_datetime(test_df["date"]))
    feature_train = feature_df[feature_df["date"].isin(train_dates)].copy()
    feature_test = feature_df[feature_df["date"].isin(test_dates)].copy()
    X_train, y_train = make_model_matrix(feature_train)
    X_test, y_test = make_model_matrix(feature_test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    model = HistGradientBoostingRegressor(learning_rate=0.05, max_depth=8, max_iter=250, min_samples_leaf=20, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return y_test, predictions


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "processed" / "online_retail_daily.csv", parse_dates=["date"])
    results = rolling_backtest(
        data=df,
        date_column="date",
        minimum_training_period=120,
        test_period_size=14,
        step_size=14,
        fit_predict_fn=fit_predict_hist_gbm,
    )
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "backtesting_metrics.csv"
    results.to_csv(metrics_path, index=False)

    summary = results[["mae", "rmse", "wape", "smape", "forecast_bias"]].mean().to_dict()
    summary_lines = [
        "# Backtesting Summary",
        "",
        f"- Average MAE: {summary['mae']:.4f}",
        f"- Average RMSE: {summary['rmse']:.4f}",
        f"- Average WAPE: {summary['wape']:.4f}",
        f"- Average SMAPE: {summary['smape']:.4f}",
        f"- Average Forecast Bias: {summary['forecast_bias']:.4f}",
    ]
    (reports_dir / "backtesting_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved backtesting outputs to {reports_dir}")


if __name__ == "__main__":
    main()
