from fastapi.testclient import TestClient

from serve.app import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_forecast_endpoint_structure() -> None:
    payload = {
        "records": [
            {
                "date": "2024-01-01",
                "store_id": "A",
                "sku_id": "SKU1",
                "sales": 10,
                "price": 2.0,
                "promo": 0,
                "on_hand_inventory": 20,
                "lead_time_days": 5,
                "unit_cost": 1.0,
            }
        ],
        "horizon": 28,
        "service_level": 0.95,
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code in {200, 400}
    body = response.json()
    if response.status_code == 200:
        assert "predictions" in body
    else:
        assert "detail" in body


def test_recommendation_endpoint_structure() -> None:
    payload = {"items": [{"store_id": "UNITED_KINGDOM", "sku_id": "85123A"}], "horizon": 28, "service_level": 0.95}
    response = client.post("/recommendations", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "recommendations" in body
