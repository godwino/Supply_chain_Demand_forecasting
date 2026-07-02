from __future__ import annotations

from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request


REQUIRED_COLUMNS = {
    "date",
    "store_id",
    "sku_id",
    "sales",
    "price",
    "promo",
    "on_hand_inventory",
    "lead_time_days",
    "unit_cost",
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    df = pd.read_csv(dataset_path, parse_dates=["date"], dtype={"store_id": "string", "sku_id": "string"}, low_memory=False)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_list}")

    df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    df["promo"] = df["promo"].astype(int)
    return df


def dataset_fingerprint(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return md5(payload, usedforsecurity=False).hexdigest()


def train_validation_split(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=horizon)
    train_df = df[df["date"] <= cutoff].copy()
    valid_df = df[df["date"] > cutoff].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("Train/validation split failed. Increase dataset history or reduce horizon.")
    return train_df, valid_df


def build_synthetic_dataset(
    output_path: str | Path,
    num_days: int = 240,
    stores: tuple[str, ...] = ("CA_1", "CA_2", "TX_1"),
    skus: tuple[str, ...] = ("SKU_001", "SKU_002", "SKU_003", "SKU_004"),
    seed: int = 42,
) -> Path:
    rng = np.random.default_rng(seed)
    start_date = pd.Timestamp("2024-01-01")
    rows: list[dict[str, object]] = []

    for store in stores:
        for sku in skus:
            base_demand = rng.integers(12, 40)
            base_price = rng.uniform(4.0, 18.0)
            trend = rng.uniform(-0.01, 0.03)
            weekly_pattern = rng.normal(0, 2.0, size=7)
            inventory = rng.integers(45, 95)
            lead_time = int(rng.integers(3, 8))
            unit_cost = round(base_price * rng.uniform(0.35, 0.65), 2)

            for day in range(num_days):
                current_date = start_date + pd.Timedelta(days=day)
                promo = int(rng.random() < 0.14)
                seasonality = weekly_pattern[current_date.dayofweek]
                month_lift = 4 if current_date.month in {11, 12} else 0
                promo_lift = rng.uniform(4, 10) if promo else 0
                noise = rng.normal(0, 3.0)
                price = max(1.5, base_price * (0.93 if promo else 1.0) + rng.normal(0, 0.2))
                demand = max(0, base_demand + trend * day + seasonality + month_lift + promo_lift - 0.5 * price + noise)
                sales = int(round(demand))
                inventory = max(10, inventory + rng.integers(-6, 8) - sales)

                rows.append(
                    {
                        "date": current_date,
                        "store_id": store,
                        "sku_id": sku,
                        "sales": sales,
                        "price": round(float(price), 2),
                        "promo": promo,
                        "on_hand_inventory": inventory,
                        "lead_time_days": lead_time,
                        "unit_cost": unit_cost,
                    }
                )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def download_file(url: str, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output)
    return output


def prepare_online_retail_dataset(
    raw_input_path: str | Path,
    processed_output_path: str | Path,
    max_series: int = 250,
) -> Path:
    raw_path = Path(raw_input_path)
    df = pd.read_excel(raw_path)

    rename_map = {
        "InvoiceDate": "date",
        "StockCode": "sku_id",
        "Country": "store_id",
        "Quantity": "quantity",
        "UnitPrice": "price",
    }
    df = df.rename(columns=rename_map)
    required = {"date", "sku_id", "store_id", "quantity", "price"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Unexpected Online Retail schema. Missing columns: {sorted(missing)}")

    df = df.dropna(subset=["date", "sku_id", "store_id", "quantity", "price"])
    df = df[(df["quantity"] > 0) & (df["price"] > 0)].copy()
    df["sku_id"] = df["sku_id"].astype(str).str.strip()
    df["store_id"] = df["store_id"].astype(str).str.strip().str.upper().str.replace(" ", "_")
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    df = df[df["sku_id"].str.fullmatch(r"[A-Za-z0-9]+", na=False)]

    daily = (
        df.groupby(["date", "store_id", "sku_id"], as_index=False)
        .agg(
            sales=("quantity", "sum"),
            price=("price", "mean"),
            order_lines=("quantity", "size"),
        )
        .sort_values(["store_id", "sku_id", "date"])
        .reset_index(drop=True)
    )

    group = daily.groupby(["store_id", "sku_id"])
    rolling_price = group["price"].transform(lambda s: s.rolling(window=30, min_periods=7).median())
    rolling_sales = group["sales"].transform(lambda s: s.shift(1).rolling(window=28, min_periods=7).mean())
    rolling_std = group["sales"].transform(lambda s: s.shift(1).rolling(window=28, min_periods=7).std())

    daily["promo"] = ((daily["price"] < rolling_price.fillna(daily["price"]) * 0.97)).astype(int)
    daily["unit_cost"] = (daily["price"] * 0.58).round(2)

    lead_time_seed = pd.util.hash_pandas_object(daily["store_id"] + "::" + daily["sku_id"], index=False).astype("uint64")
    daily["lead_time_days"] = (lead_time_seed % 6 + 3).astype(int)

    base_inventory = (rolling_sales.fillna(daily["sales"]).clip(lower=1) * (daily["lead_time_days"] + 7)).round()
    variability_buffer = (rolling_std.fillna(0) * 2).round()
    daily["on_hand_inventory"] = (base_inventory + variability_buffer).clip(lower=daily["sales"]).astype(int)

    daily = daily.drop(columns=["order_lines"])
    daily = daily.dropna(subset=["sales", "price", "promo", "on_hand_inventory", "lead_time_days", "unit_cost"])

    # Focus on series with enough history for lag-based forecasting.
    history = daily.groupby(["store_id", "sku_id"]).size().rename("history_days").reset_index()
    valid_series = history[history["history_days"] >= 60][["store_id", "sku_id"]]
    prepared = daily.merge(valid_series, on=["store_id", "sku_id"], how="inner")
    series_rank = (
        prepared.groupby(["store_id", "sku_id"], as_index=False)["sales"]
        .sum()
        .rename(columns={"sales": "total_sales"})
        .sort_values("total_sales", ascending=False)
    )
    top_series = series_rank.head(max_series)[["store_id", "sku_id"]]
    prepared = prepared.merge(top_series, on=["store_id", "sku_id"], how="inner")
    prepared = prepared.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    prepared["store_id"] = prepared["store_id"].astype(str)
    prepared["sku_id"] = prepared["sku_id"].astype(str)

    output = Path(processed_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False)
    return output


def download_and_prepare_online_retail(
    raw_output_path: str | Path,
    processed_output_path: str | Path,
    url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx",
    max_series: int = 250,
) -> Path:
    raw_path = download_file(url, raw_output_path)
    return prepare_online_retail_dataset(raw_path, processed_output_path, max_series=max_series)
