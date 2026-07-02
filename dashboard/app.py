from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from supply_chain_forecasting.evaluation import business_metrics


st.set_page_config(page_title="Retail Demand Planner", layout="wide")
st.title("Retail Demand Forecasting and Replenishment")
st.caption("Planner-facing view for forecasted demand, stockout risk, and recommended reorders.")

artifact_path = ROOT / "artifacts" / "latest_predictions.csv"

if not artifact_path.exists():
    st.warning("No forecast artifact found yet. Run `python train.py` first.")
    st.stop()

df = pd.read_csv(artifact_path)

store_filter = st.sidebar.multiselect("Store", options=sorted(df["store_id"].unique()), default=sorted(df["store_id"].unique()))
sku_filter = st.sidebar.multiselect("SKU", options=sorted(df["sku_id"].unique()), default=sorted(df["sku_id"].unique())[:6])
risk_threshold = st.sidebar.slider("Minimum stockout risk", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

filtered = df[df["store_id"].isin(store_filter) & df["sku_id"].isin(sku_filter)]
filtered = filtered[filtered["stockout_risk"] >= risk_threshold]

metrics = business_metrics(df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Reorder Qty", f"{metrics['avg_reorder_qty']:.1f}")
col2.metric("Risk Rate", f"{metrics['stockout_risk_rate']:.1%}")
col3.metric("Avg Days Of Cover", f"{metrics['avg_days_of_cover']:.1f}")
col4.metric("Projected Margin", f"${metrics['projected_margin']:,.0f}")

st.subheader("Priority Reorder Recommendations")
st.dataframe(
    filtered.sort_values(["stockout_risk", "reorder_qty"], ascending=[False, False])[
        [
            "date",
            "store_id",
            "sku_id",
            "actual_sales",
            "forecast_units",
            "on_hand_inventory",
            "reorder_qty",
            "stockout_risk",
            "days_of_cover",
        ]
    ],
    use_container_width=True,
)

st.subheader("Forecast vs Actual")
plot_df = df.groupby("date", as_index=False)[["actual_sales", "forecast_units"]].sum()
st.line_chart(plot_df.set_index("date"))
