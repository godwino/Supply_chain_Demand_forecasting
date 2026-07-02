from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
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


class RecommendationItem(BaseModel):
    store_id: str
    sku_id: str
    as_of_date: str | None = None


class RecommendationRequest(BaseModel):
    items: list[RecommendationItem]
    horizon: int = 28
    service_level: float = 0.95
    model_uri: str | None = None
    dataset_path: str | None = None


app = FastAPI(title="Retail Demand Forecasting API", version="2.0.0")
config = ProjectConfig()


def default_model_uri() -> str:
    return f"models:/{config.registered_model_name}@champion"


def default_dataset_path() -> Path:
    return ROOT / os.getenv("SERVE_DATASET_PATH", "data/processed/online_retail_daily.csv")


@lru_cache(maxsize=4)
def load_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, parse_dates=["date"], dtype={"store_id": "string", "sku_id": "string"}, low_memory=False)
    frame["store_id"] = frame["store_id"].astype(str)
    frame["sku_id"] = frame["sku_id"].astype(str)
    return frame.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)


@lru_cache(maxsize=4)
def load_model(model_uri: str) -> Any:
    mlflow.set_tracking_uri(config.tracking_uri)
    return mlflow.pyfunc.load_model(model_uri)


def align_model_input(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    input_schema = getattr(model.metadata, "get_input_schema", lambda: None)()
    if input_schema is None:
        return X

    aligned = X.copy()
    additions: dict[str, Any] = {}
    schema_columns: list[str] = []
    for spec in input_schema.inputs:
        schema_columns.append(spec.name)
        if spec.name not in aligned.columns:
            if getattr(spec.type, "name", "") == "boolean":
                additions[spec.name] = False
            else:
                additions[spec.name] = 0.0

    if additions:
        aligned = pd.concat([aligned, pd.DataFrame(additions, index=aligned.index)], axis=1)

    aligned = aligned.reindex(columns=schema_columns, fill_value=0)
    return aligned


def dataset_history_slice(frame: pd.DataFrame, store_id: str, sku_id: str, as_of_date: str | None) -> pd.DataFrame:
    subset = frame[(frame["store_id"] == store_id) & (frame["sku_id"] == sku_id)].copy()
    if as_of_date:
        cutoff = pd.Timestamp(as_of_date)
        subset = subset[subset["date"] <= cutoff].copy()
    return subset.sort_values("date").reset_index(drop=True)


def score_feature_rows(feature_frame: pd.DataFrame, model_uri: str, horizon: int, service_level: float) -> pd.DataFrame:
    model = load_model(model_uri)
    X, _ = make_model_matrix(feature_frame)
    X = align_model_input(model, X)
    forecast = model.predict(X)

    planner_frame = feature_frame[
        ["date", "store_id", "sku_id", "price", "promo", "on_hand_inventory", "lead_time_days", "unit_cost", "rolling_std_7"]
    ].copy()
    planner_frame["forecast_units"] = forecast
    planner_frame = add_reorder_recommendations(planner_frame, service_level=service_level, horizon=horizon)
    planner_frame["date"] = planner_frame["date"].astype(str)
    return planner_frame


def ensure_model_uri(model_uri: str | None) -> str:
    return model_uri or default_model_uri()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    mlflow.set_tracking_uri(config.tracking_uri)
    client = MlflowClient(tracking_uri=config.tracking_uri)
    model_name = config.registered_model_name

    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
        return {
            "registered_model_name": model_name,
            "alias": "champion",
            "version": champion.version,
            "run_id": champion.run_id,
            "current_stage": champion.current_stage,
            "source": champion.source,
        }
    except Exception as exc:
        return {
            "registered_model_name": model_name,
            "alias": "champion",
            "detail": f"Champion alias not available: {exc}",
        }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if not request.records:
        raise HTTPException(status_code=400, detail="At least one record is required.")

    model_uri = ensure_model_uri(request.model_uri)

    try:
        frame = pd.DataFrame([record.model_dump() for record in request.records])
        frame["date"] = pd.to_datetime(frame["date"])
        feature_frame = create_features(frame)
        if feature_frame.empty:
            raise HTTPException(
                status_code=400,
                detail="Not enough sequential history to compute lag features. Provide at least 28 prior days per SKU-store.",
            )

        planner_frame = score_feature_rows(
            feature_frame=feature_frame,
            model_uri=model_uri,
            horizon=request.horizon,
            service_level=request.service_level,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "endpoint": "predict",
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


@app.post("/forecast")
def forecast(request: PredictionRequest) -> dict[str, Any]:
    response = predict(request)
    response["endpoint"] = "forecast"
    return response


@app.post("/recommendations")
def recommendations(request: RecommendationRequest) -> dict[str, Any]:
    if not request.items:
        raise HTTPException(status_code=400, detail="At least one item is required.")

    model_uri = ensure_model_uri(request.model_uri)
    dataset_path = Path(request.dataset_path) if request.dataset_path else default_dataset_path()

    try:
        frame = load_dataset(str(dataset_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to load serving dataset: {exc}") from exc

    recommendation_rows: list[dict[str, Any]] = []
    missing_items: list[dict[str, str | None]] = []

    for item in request.items:
        history = dataset_history_slice(frame, item.store_id, item.sku_id, item.as_of_date)
        if history.empty:
            missing_items.append(item.model_dump())
            continue

        feature_frame = create_features(history)
        if feature_frame.empty:
            missing_items.append(item.model_dump())
            continue

        latest_feature_row = feature_frame.tail(1).copy()
        planner_frame = score_feature_rows(
            feature_frame=latest_feature_row,
            model_uri=model_uri,
            horizon=request.horizon,
            service_level=request.service_level,
        )
        recommendation_rows.extend(
            planner_frame[
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
            ].to_dict(orient="records")
        )

    if not recommendation_rows:
        raise HTTPException(
            status_code=400,
            detail="No recommendation rows could be built. Check item keys and ensure enough history exists.",
        )

    return {
        "endpoint": "recommendations",
        "model_uri": model_uri,
        "dataset_path": str(dataset_path),
        "recommendations": recommendation_rows,
        "missing_items": missing_items,
    }
