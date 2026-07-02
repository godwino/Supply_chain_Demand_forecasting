from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from validation.data_checks import run_all_checks


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "processed" / "online_retail_daily.csv")
    checks = run_all_checks(df)
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Data Validation Report", ""]
    for check in checks:
        lines.extend(
            [
                f"## {check.name}",
                f"- Status: {check.status}",
                f"- Affected rows/groups: {check.affected_rows}",
                f"- Recommendation: {check.recommendation}",
                "",
            ]
        )
    (reports_dir / "data_validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved validation report to {reports_dir / 'data_validation_report.md'}")


if __name__ == "__main__":
    main()
