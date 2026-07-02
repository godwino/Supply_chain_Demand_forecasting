from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monitoring.monitoring_metrics import (
    api_style_output_summary,
    demand_drift_summary,
    high_stockout_risk_count,
    prediction_drift_summary,
    recent_forecast_performance,
    sku_mix_change,
)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "app" / "dashboard_predictions.csv", parse_dates=["date"])
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    performance = recent_forecast_performance(df)
    demand_drift = demand_drift_summary(df)
    prediction_drift = prediction_drift_summary(df)
    sku_mix = sku_mix_change(df)
    stockout = high_stockout_risk_count(df)
    api_summary = api_style_output_summary(df)

    lines = [
        "# Monitoring Report",
        "",
        "## Forecast performance monitoring",
        *(f"- {k}: {v:.4f}" for k, v in performance.items()),
        "",
        "## Data drift monitoring",
        *(f"- {k}: {v:.4f}" for k, v in demand_drift.items()),
        "",
        "## Prediction drift monitoring",
        *(f"- {k}: {v:.4f}" for k, v in prediction_drift.items()),
        "",
        "## Business KPI monitoring",
        *(f"- {k}: {v}" for k, v in stockout.items()),
        *(f"- {k}: {v}" for k, v in api_summary.items()),
        *(f"- {k}: {v}" for k, v in sku_mix.items()),
        "",
        "## API health placeholder",
        "- Monitor HTTP 5xx rate, p95 latency, and request volume in production.",
        "",
        "## Recommended alert thresholds",
        "- WAPE increases by more than 20% from baseline.",
        "- Forecast bias exceeds +/-10% of total actual demand.",
        "- High stockout risk SKUs exceed an agreed operational threshold.",
        "- Input data volume drops unexpectedly.",
        "- API latency or error rate increases.",
    ]
    (reports_dir / "monitoring_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved monitoring report to {reports_dir / 'monitoring_report.md'}")


if __name__ == "__main__":
    main()
