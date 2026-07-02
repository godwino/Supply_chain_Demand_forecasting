from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the same local training entrypoint, but default experiment path for Databricks.
os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", os.getenv("DATABRICKS_EXPERIMENT_PATH", "/Shared/retail-demand-forecasting"))

from train import main


if __name__ == "__main__":
    main()
