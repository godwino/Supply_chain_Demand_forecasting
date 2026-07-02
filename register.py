from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from supply_chain_forecasting.config import ProjectConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a trained model in MLflow Model Registry.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--alias", default="challenger")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()

    mlflow.set_tracking_uri(config.tracking_uri)
    client = MlflowClient(tracking_uri=config.tracking_uri)
    model_name = args.model_name or config.registered_model_name
    model_uri = f"runs:/{args.run_id}/model"

    try:
        client.create_registered_model(model_name)
    except Exception:
        pass

    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.set_registered_model_alias(model_name, args.alias, registered.version)
    print(f"Registered model '{model_name}' version {registered.version} with alias '{args.alias}'")


if __name__ == "__main__":
    main()
