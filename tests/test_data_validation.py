import pandas as pd

from validation.data_checks import check_duplicates, check_invalid_dates, check_negative_values, check_required_columns


def test_missing_required_columns() -> None:
    df = pd.DataFrame({"date": ["2024-01-01"]})
    result = check_required_columns(df)
    assert result.status == "fail"


def test_negative_quantity_detection() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "store_id": ["A"],
            "sku_id": ["B"],
            "sales": [-1],
            "price": [1.0],
            "promo": [0],
            "on_hand_inventory": [10],
            "lead_time_days": [5],
            "unit_cost": [0.5],
        }
    )
    results = check_negative_values(df)
    sales_result = [result for result in results if result.name == "negative_sales"][0]
    assert sales_result.status == "fail"


def test_duplicate_detection_and_invalid_date() -> None:
    df = pd.DataFrame(
        {
            "date": ["bad-date", "bad-date"],
            "store_id": ["A", "A"],
            "sku_id": ["B", "B"],
            "sales": [1, 1],
            "price": [1.0, 1.0],
            "promo": [0, 0],
            "on_hand_inventory": [10, 10],
            "lead_time_days": [5, 5],
            "unit_cost": [0.5, 0.5],
        }
    )
    assert check_duplicates(df).status == "warning"
    assert check_invalid_dates(df).status == "fail"
