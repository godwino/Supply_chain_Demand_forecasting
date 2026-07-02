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
from supply_chain_forecasting.reorder import add_reorder_recommendations


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def urgency_bucket(row: pd.Series) -> str:
    if row["reorder_qty"] > 0 or row["stockout_risk"] >= 0.35:
        return "Act now"
    if row["days_of_cover"] < 180 or row["forecast_units"] > row["actual_sales"]:
        return "Watch closely"
    return "Stable"


def priority_score(row: pd.Series) -> float:
    return float(row["reorder_qty"]) + float(row["stockout_risk"]) * 1000.0 + max(0.0, 200.0 - float(row["days_of_cover"]))


st.set_page_config(page_title="Retail Demand Planner", page_icon=":bar_chart:", layout="wide")
st.title("Retail Demand Planner")
st.caption("Interactive planner workspace for forecast review, replenishment decisions, and what-if analysis.")

artifact_path = ROOT / "artifacts" / "latest_predictions.csv"
packaged_path = ROOT / "data" / "app" / "dashboard_predictions.csv"

data_path: Path | None = None
if artifact_path.exists():
    data_path = artifact_path
elif packaged_path.exists():
    data_path = packaged_path
    st.info("Using the packaged dashboard dataset for hosted viewing.")
else:
    st.warning("No forecast artifact found yet. Run `python train.py` first.")
    st.stop()

df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])
df["forecast_error"] = df["forecast_units"] - df["actual_sales"]
df["forecast_bias_pct"] = df["forecast_error"] / df["actual_sales"].clip(lower=1)
df["urgency"] = df.apply(urgency_bucket, axis=1)
df["daily_forecast_units"] = df["forecast_units"] / 28.0
df["inventory_value"] = df["on_hand_inventory"] * df["unit_cost"]
df["priority_score"] = df.apply(priority_score, axis=1)

store_options = sorted(df["store_id"].unique())
default_store = "UNITED_KINGDOM" if "UNITED_KINGDOM" in store_options else store_options[0]

st.sidebar.header("Planner Controls")
selected_store = st.sidebar.selectbox("Store", options=store_options, index=store_options.index(default_store))
store_df = df[df["store_id"] == selected_store].copy()
default_focus_row = store_df.sort_values(["priority_score", "stockout_risk", "reorder_qty"], ascending=False).iloc[0]

sku_options = sorted(store_df["sku_id"].unique())
default_sku = default_focus_row["sku_id"]
selected_sku = st.sidebar.selectbox("Focus SKU", options=sku_options, index=sku_options.index(default_sku))
sku_date_options = sorted(store_df[store_df["sku_id"] == selected_sku]["date"].dt.strftime("%Y-%m-%d").unique())
default_date = default_focus_row["date"].strftime("%Y-%m-%d") if default_focus_row["sku_id"] == selected_sku else sku_date_options[-1]
selected_date = st.sidebar.selectbox(
    "As-of date",
    options=sku_date_options,
    index=sku_date_options.index(default_date) if default_date in sku_date_options else len(sku_date_options) - 1,
)
risk_threshold = st.sidebar.slider("Minimum stockout risk", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
show_only_reorder = st.sidebar.checkbox("Only show reorder actions", value=True)

selected_row = store_df[
    (store_df["sku_id"] == selected_sku) & (store_df["date"].dt.strftime("%Y-%m-%d") == selected_date)
].tail(1)
if selected_row.empty:
    st.error("No matching record found for the selected store, SKU, and date.")
    st.stop()

selected_row = selected_row.copy()
base_service_level = 0.95
base_horizon = 28

st.sidebar.subheader("What-if Scenario")
scenario_inventory = st.sidebar.number_input(
    "Override on-hand inventory",
    min_value=0.0,
    value=float(selected_row["on_hand_inventory"].iloc[0]),
    step=10.0,
)
scenario_lead_time = st.sidebar.slider(
    "Override lead time days",
    min_value=1,
    max_value=30,
    value=int(selected_row["lead_time_days"].iloc[0]),
)
scenario_service_level = st.sidebar.slider(
    "Service level",
    min_value=0.80,
    max_value=0.99,
    value=float(base_service_level),
    step=0.01,
)

scenario_row = selected_row.copy()
scenario_row["on_hand_inventory"] = scenario_inventory
scenario_row["lead_time_days"] = scenario_lead_time
scenario_row = add_reorder_recommendations(
    scenario_row[
        [
            "date",
            "store_id",
            "sku_id",
            "price",
            "promo",
            "on_hand_inventory",
            "lead_time_days",
            "unit_cost",
            "rolling_std_7",
            "forecast_units",
        ]
    ],
    service_level=scenario_service_level,
    horizon=base_horizon,
)

metrics = business_metrics(df)
actionable_df = store_df[store_df["stockout_risk"] >= risk_threshold].copy()
if show_only_reorder:
    actionable_df = actionable_df[actionable_df["reorder_qty"] > 0]
actionable_df = actionable_df.sort_values(
    ["priority_score", "reorder_qty", "stockout_risk", "forecast_units"],
    ascending=[False, False, False, False],
)
selected_store_metrics = business_metrics(store_df)
action_sku_count = int((store_df["reorder_qty"] > 0).sum())
watch_sku_count = int((store_df["stockout_risk"] >= 0.10).sum())
top_priority = store_df.sort_values(["priority_score", "reorder_qty", "stockout_risk"], ascending=False).iloc[0]

top1, top2, top3, top4 = st.columns(4)
top1.metric("Avg Reorder Qty", f"{selected_store_metrics['avg_reorder_qty']:.1f}")
top2.metric("Risk Rate", format_percent(selected_store_metrics["stockout_risk_rate"]))
top3.metric("Avg Days Of Cover", f"{selected_store_metrics['avg_days_of_cover']:.1f}")
top4.metric("Projected Margin", f"${selected_store_metrics['projected_margin']:,.0f}")

st.markdown(
    f"**Store snapshot:** {action_sku_count} reorder actions, {watch_sku_count} SKUs above 10% stockout risk, "
    f"top priority SKU is **{top_priority['sku_id']}** on **{pd.to_datetime(top_priority['date']).strftime('%Y-%m-%d')}**."
)

hero_left, hero_right = st.columns([1.3, 1])

with hero_left:
    st.subheader("Planner Workbench")
    st.markdown(
        f"Focused on **{selected_store} / {selected_sku}** as of **{selected_date}**. "
        "Use the scenario controls to simulate inventory or lead-time changes before ordering."
    )

    card1, card2, card3, card4 = st.columns(4)
    card1.metric("Forecast Units", f"{selected_row['forecast_units'].iloc[0]:,.1f}")
    card2.metric("Actual Sales", f"{selected_row['actual_sales'].iloc[0]:,.1f}")
    card3.metric("Current Reorder Qty", f"{int(selected_row['reorder_qty'].iloc[0]):,}")
    card4.metric("Current Days Of Cover", f"{selected_row['days_of_cover'].iloc[0]:,.1f}")

    trend_df = store_df[store_df["sku_id"] == selected_sku].sort_values("date").copy()
    trend_df = trend_df.set_index("date")[["actual_sales", "forecast_units"]]
    st.line_chart(trend_df, width="stretch")

with hero_right:
    st.subheader("Scenario Recommendation")
    current = selected_row.iloc[0]
    scenario = scenario_row.iloc[0]

    s1, s2, s3 = st.columns(3)
    s1.metric("Scenario Reorder Qty", f"{int(scenario['reorder_qty']):,}", delta=int(scenario["reorder_qty"] - current["reorder_qty"]))
    s2.metric("Scenario Stockout Risk", f"{scenario['stockout_risk']:.1%}", delta=f"{scenario['stockout_risk'] - current['stockout_risk']:+.1%}")
    s3.metric("Scenario Cover", f"{scenario['days_of_cover']:,.1f}", delta=f"{scenario['days_of_cover'] - current['days_of_cover']:+.1f}")

    scenario_table = pd.DataFrame(
        [
            {"Metric": "On-hand inventory", "Current": current["on_hand_inventory"], "Scenario": scenario_inventory},
            {"Metric": "Lead time days", "Current": current["lead_time_days"], "Scenario": scenario_lead_time},
            {"Metric": "Service level", "Current": base_service_level, "Scenario": scenario_service_level},
            {"Metric": "Reorder point", "Current": current["reorder_point"], "Scenario": scenario["reorder_point"]},
            {"Metric": "Safety stock", "Current": current["safety_stock"], "Scenario": scenario["safety_stock"]},
        ]
    )
    st.dataframe(scenario_table, width="stretch", hide_index=True)

tab1, tab2, tab3 = st.tabs(["Action Queue", "SKU Explorer", "Business View"])

with tab1:
    st.subheader("Priority Action Queue")
    st.caption("Sorted to surface the most urgent replenishment opportunities for the selected store.")
    if actionable_df.empty:
        st.info("No SKUs match the current filters. Lower the risk threshold or turn off `Only show reorder actions`.")
    else:
        st.dataframe(
            actionable_df[
                [
                    "date",
                    "sku_id",
                    "urgency",
                    "actual_sales",
                    "forecast_units",
                    "on_hand_inventory",
                    "reorder_qty",
                    "stockout_risk",
                    "days_of_cover",
                    "inventory_value",
                ]
            ].head(50),
            width="stretch",
            hide_index=True,
        )

with tab2:
    st.subheader("Focused SKU Explorer")
    explorer_cols = [
        "date",
        "sku_id",
        "price",
        "promo",
        "actual_sales",
        "forecast_units",
        "forecast_error",
        "on_hand_inventory",
        "reorder_qty",
        "stockout_risk",
        "days_of_cover",
    ]
    st.dataframe(
        store_df[store_df["sku_id"] == selected_sku][explorer_cols].sort_values("date", ascending=False),
        width="stretch",
        hide_index=True,
    )

with tab3:
    st.subheader("Business View")
    agg_df = (
        store_df.groupby("sku_id", as_index=False)
        .agg(
            forecast_units=("forecast_units", "mean"),
            actual_sales=("actual_sales", "mean"),
            reorder_qty=("reorder_qty", "max"),
            stockout_risk=("stockout_risk", "max"),
            inventory_value=("inventory_value", "mean"),
        )
        .sort_values(["reorder_qty", "stockout_risk", "forecast_units"], ascending=[False, False, False])
    )
    st.dataframe(agg_df.head(25), width="stretch", hide_index=True)
