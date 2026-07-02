from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from supply_chain_forecasting.config import ProjectConfig
from supply_chain_forecasting.features import create_features, make_model_matrix
from supply_chain_forecasting.reorder import add_reorder_recommendations


class PredictionRecord(BaseModel):
    date: str
    store_id: str
    sku_id: str
    sales: float
    price: float
    promo: int = Field(ge=0, le=1)
    on_hand_inventory: float
    lead_time_days: int
    unit_cost: float


class PredictionRequest(BaseModel):
    records: list[PredictionRecord]
    horizon: int = 28
    service_level: float = 0.95
    model_uri: str | None = None


app = FastAPI(title="Retail Demand Forecasting API", version="1.0.0")
config = ProjectConfig()


def default_model_uri() -> str:
    return f"models:/{config.registered_model_name}@champion"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if not request.records:
        raise HTTPException(status_code=400, detail="At least one record is required.")

    mlflow.set_tracking_uri(config.tracking_uri)
    model_uri = request.model_uri or default_model_uri()

    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to load model '{model_uri}': {exc}") from exc

    frame = pd.DataFrame([record.model_dump() for record in request.records])
    frame["date"] = pd.to_datetime(frame["date"])
    feature_frame = create_features(frame)
    if feature_frame.empty:
        raise HTTPException(
            status_code=400,
            detail="Not enough sequential history to compute lag features. Provide at least 28 prior days per SKU-store.",
        )

    X, _ = make_model_matrix(feature_frame)
    forecast = model.predict(X)

    planner_frame = feature_frame[["date", "store_id", "sku_id", "price", "promo", "on_hand_inventory", "lead_time_days", "unit_cost", "rolling_std_7"]].copy()
    planner_frame["forecast_units"] = forecast
    planner_frame = add_reorder_recommendations(planner_frame, request.service_level, request.horizon)
    planner_frame["date"] = planner_frame["date"].astype(str)

    return {
        "model_uri": model_uri,
        "predictions": planner_frame[
            [
                "date",
                "store_id",
                "sku_id",
                "forecast_units",
                "reorder_qty",
                "stockout_risk",
                "days_of_cover",
                "reorder_point",
                "safety_stock",
            ]
        ].to_dict(orient="records"),
    }
