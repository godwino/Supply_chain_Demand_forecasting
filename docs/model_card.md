# Model Card

## Model Purpose
Forecast SKU-level demand and convert forecast outputs into replenishment recommendations.

## Training Data
Public UCI Online Retail transaction data aggregated to daily SKU-country demand.

## Features
- Lag demand features
- Rolling demand statistics
- Calendar features
- Price and promo signals
- Inventory-related inputs

## Target
Daily SKU-level demand (`sales`)

## Evaluation Metrics
- MAE
- RMSE
- MAPE
- SMAPE
- WAPE
- Forecast bias

## Assumptions
- Country acts as a store/geography proxy
- Inventory and lead time are simplified operational inputs
- Reorder logic is intended for demonstration of decision support

## Limitations
- Public retail data is not the same as McKesson or pharmaceutical data
- Cold-chain, expiry, regulatory handling, and patient impact are not modeled

## Monitoring Requirements
- Forecast drift
- Demand drift
- Bias monitoring
- High-risk SKU counts
- API health and latency

## Ethical / Business Risk Notes
Forecasting and replenishment outputs should support human decision-making, not replace planner judgment in high-stakes supply environments.
