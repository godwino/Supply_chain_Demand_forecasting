from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class ProjectConfig:
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "retail-demand-forecasting")
    registered_model_name: str = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "retail-demand-forecast")
    artifact_dir: Path = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
    forecast_horizon: int = int(os.getenv("FORECAST_HORIZON", "28"))
    reorder_service_level: float = float(os.getenv("REORDER_SERVICE_LEVEL", "0.95"))
    max_train_rows: int | None = None
    categorical_columns: list[str] = field(default_factory=lambda: ["store_id", "sku_id"])
    numeric_columns: list[str] = field(
        default_factory=lambda: [
            "sales",
            "price",
            "promo",
            "on_hand_inventory",
            "lead_time_days",
            "unit_cost",
        ]
    )


def ensure_directories(config: ProjectConfig) -> None:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
