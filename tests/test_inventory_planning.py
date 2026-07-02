from business.inventory_planning import days_of_cover, reorder_point, reorder_quantity, safety_stock, stockout_risk


def test_days_of_cover() -> None:
    assert days_of_cover(100, 10) == 10.0
    assert days_of_cover(100, 0) == float("inf")


def test_safety_stock_and_reorder_point() -> None:
    ss = safety_stock(5, 4, 0.95)
    assert ss > 0
    rp = reorder_point(10, 4, ss)
    assert rp > 40


def test_reorder_quantity_and_risk() -> None:
    ss = safety_stock(3, 5, 0.95)
    qty = reorder_quantity(5, 10, 5, ss, target_stock_days=14)
    assert qty > 0
    risk = stockout_risk(5, 10, 5, ss)
    assert 0.0 <= risk <= 1.0
