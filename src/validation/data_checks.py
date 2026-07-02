from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ValidationCheckResult:
    name: str
    status: str
    affected_rows: int
    recommendation: str


REQUIRED_COLUMNS = ["date", "store_id", "sku_id", "sales", "price", "promo", "on_hand_inventory", "lead_time_days", "unit_cost"]


def check_required_columns(df: pd.DataFrame) -> ValidationCheckResult:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    status = "pass" if not missing else "fail"
    return ValidationCheckResult("required_columns", status, len(missing), "Add the missing required columns before training.")


def check_missing_values(df: pd.DataFrame) -> ValidationCheckResult:
    missing_rows = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    status = "pass" if missing_rows == 0 else "warning"
    return ValidationCheckResult("missing_values", status, missing_rows, "Investigate null values in required forecasting fields.")


def check_duplicates(df: pd.DataFrame) -> ValidationCheckResult:
    duplicate_rows = int(df.duplicated().sum())
    status = "pass" if duplicate_rows == 0 else "warning"
    return ValidationCheckResult("duplicate_rows", status, duplicate_rows, "Deduplicate repeated transactional or daily rows.")


def check_negative_values(df: pd.DataFrame) -> list[ValidationCheckResult]:
    results = []
    for column in ["sales", "price", "on_hand_inventory", "lead_time_days", "unit_cost"]:
        affected = int((df[column] < 0).sum()) if column in df.columns else 0
        status = "pass" if affected == 0 else "fail"
        results.append(ValidationCheckResult(f"negative_{column}", status, affected, f"Remove or correct negative values in {column}."))
    return results


def check_invalid_dates(df: pd.DataFrame) -> ValidationCheckResult:
    parsed = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    affected = int(parsed.isna().sum())
    status = "pass" if affected == 0 else "fail"
    return ValidationCheckResult("invalid_dates", status, affected, "Fix invalid dates before time-series training.")


def check_short_history(df: pd.DataFrame, min_history: int = 60) -> ValidationCheckResult:
    counts = df.groupby(["store_id", "sku_id"]).size()
    affected = int((counts < min_history).sum())
    status = "pass" if affected == 0 else "warning"
    return ValidationCheckResult("short_history_groups", status, affected, "Consider excluding or aggregating very short series.")


def check_date_gaps(df: pd.DataFrame) -> ValidationCheckResult:
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    gap_groups = 0
    for (_, _), subset in frame.groupby(["store_id", "sku_id"]):
        dates = subset["date"].sort_values().dropna().unique()
        if len(dates) > 1 and pd.date_range(min(dates), max(dates), freq="D").difference(pd.DatetimeIndex(dates)).size > 0:
            gap_groups += 1
    status = "pass" if gap_groups == 0 else "warning"
    return ValidationCheckResult("date_gaps", status, gap_groups, "Assess whether missing demand dates should be backfilled with zeros.")


def check_outlier_demand(df: pd.DataFrame, z_threshold: float = 4.0) -> ValidationCheckResult:
    sales = df["sales"].astype(float)
    if sales.std(ddof=0) == 0:
        return ValidationCheckResult("outlier_demand", "pass", 0, "No material demand outliers detected.")
    z_scores = (sales - sales.mean()) / sales.std(ddof=0)
    affected = int((np.abs(z_scores) > z_threshold).sum())
    status = "pass" if affected == 0 else "warning"
    return ValidationCheckResult("outlier_demand", status, affected, "Review extreme demand spikes for promotions, data issues, or special events.")


def check_zero_demand(df: pd.DataFrame) -> ValidationCheckResult:
    affected = int((df["sales"].fillna(0) == 0).sum())
    status = "pass" if affected == 0 else "warning"
    return ValidationCheckResult("zero_demand", status, affected, "Review zero-demand periods to confirm whether they represent no sales or missing data.")


def run_all_checks(df: pd.DataFrame) -> list[ValidationCheckResult]:
    checks = [
        check_required_columns(df),
        check_missing_values(df),
        check_duplicates(df),
        check_invalid_dates(df),
        check_short_history(df),
        check_date_gaps(df),
        check_outlier_demand(df),
        check_zero_demand(df),
    ]
    checks.extend(check_negative_values(df))
    return checks
