# Ingest Notes

Primary real-data entrypoint:

```bash
C:\Users\Osayamwen\anaconda3\python.exe data/ingest/download_real_data.py
```

This downloads the public UCI Online Retail dataset and prepares a modeling table for the project.

Suggested future mappings:

- M5 `sales_train_validation.csv` -> long-format daily `sales`
- M5 `sell_prices.csv` -> `price`
- M5 `calendar.csv` -> calendar and event features

The training pipeline expects a unified table with:

- `date`
- `store_id`
- `sku_id`
- `sales`
- `price`
- `promo`
- `on_hand_inventory`
- `lead_time_days`
- `unit_cost`
