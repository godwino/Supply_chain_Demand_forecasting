from evaluation.forecast_metrics import calculate_forecast_metrics


def test_forecast_metrics_basic() -> None:
    metrics = calculate_forecast_metrics([10, 20], [12, 18])
    assert round(metrics["mae"], 4) == 2.0
    assert round(metrics["rmse"], 4) == 2.0
    assert round(metrics["wape"], 4) == round(4 / 30, 4)
    assert metrics["forecast_bias"] == 0.0


def test_forecast_metrics_divide_by_zero() -> None:
    metrics = calculate_forecast_metrics([0, 0], [0, 1])
    assert metrics["wape"] == 0.0
    assert metrics["bias_percentage"] == 0.0
    assert metrics["mape"] == 0.0
