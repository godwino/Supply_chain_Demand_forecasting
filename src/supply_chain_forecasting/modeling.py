from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


@dataclass
class TrainedModel:
    name: str
    estimator: object
    predictions: np.ndarray
    params: dict[str, object]


class NaiveLastValueModel:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveLastValueModel":
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X["lag_1"], dtype=float)


def available_models(random_state: int = 42) -> dict[str, tuple[object, dict[str, object]]]:
    models: dict[str, tuple[object, dict[str, object]]] = {
        "naive_last_value": (NaiveLastValueModel(), {}),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=8,
                max_iter=250,
                min_samples_leaf=20,
                random_state=random_state,
            ),
            {"learning_rate": 0.05, "max_depth": 8, "max_iter": 250},
        ),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=250,
                max_depth=14,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=random_state,
            ),
            {"n_estimators": 250, "max_depth": 14, "min_samples_leaf": 3},
        ),
    }

    if LGBMRegressor is not None:
        models["lightgbm"] = (
            LGBMRegressor(
                learning_rate=0.05,
                n_estimators=350,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.9,
                random_state=random_state,
            ),
            {"learning_rate": 0.05, "n_estimators": 350, "num_leaves": 31},
        )

    if XGBRegressor is not None:
        models["xgboost"] = (
            XGBRegressor(
                learning_rate=0.05,
                n_estimators=350,
                max_depth=8,
                subsample=0.85,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=random_state,
            ),
            {"learning_rate": 0.05, "n_estimators": 350, "max_depth": 8},
        )

    return models


def fit_and_predict(model_name: str, estimator: object, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame) -> TrainedModel:
    estimator.fit(X_train, y_train)
    predictions = np.clip(np.asarray(estimator.predict(X_valid), dtype=float), 0, None)
    params = getattr(estimator, "get_params", lambda: {})()
    return TrainedModel(name=model_name, estimator=estimator, predictions=predictions, params=params)


def feature_importance_frame(model: object, feature_names: list[str]) -> pd.DataFrame | None:
    if hasattr(model, "feature_importances_"):
        return (
            pd.DataFrame({"feature": feature_names, "importance": getattr(model, "feature_importances_")})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    return None
