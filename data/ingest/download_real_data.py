from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from supply_chain_forecasting.data import download_and_prepare_online_retail


def main() -> None:
    output = download_and_prepare_online_retail(
        raw_output_path=ROOT / "data" / "raw" / "online_retail.xlsx",
        processed_output_path=ROOT / "data" / "processed" / "online_retail_daily.csv",
        max_series=250,
    )
    print(f"Prepared real dataset at {output}")


if __name__ == "__main__":
    main()
