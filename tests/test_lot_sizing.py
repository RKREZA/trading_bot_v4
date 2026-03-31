import pytest
from core.lot_calculator import LotCalculator

def test_lot_size_calculation():
    # Example XAUUSDm
    # Risk $10, SL distance 1.5 ($1.50)
    # Tick size 0.01, Tick Value 1.0 (means 1 lot moving 0.01 is $1.00)
    # So 1 lot moving 1.5 is $150.00
    # To risk $10: 10 / 150 = 0.0666 -> 0.06 or 0.07 lots.
    
    lot = LotCalculator.calculate(
        risk_amount=10.0,
        sl_distance=1.5,
        tick_size=0.01,
        tick_value=1.0,
        volume_min=0.01,
        volume_step=0.01,
        volume_max=100.0
    )
    
    # risk_per_lot = (1.5 / 0.01) * 1.0 = 150 * 1 = 150
    # lot = 10.0 / 150 = 0.06666
    # rounded to step 0.01 -> 0.07
    assert lot == 0.07

def test_lot_clamped_to_min():
    lot = LotCalculator.calculate(
        risk_amount=1.0,
        sl_distance=100.0,
        tick_size=0.01,
        tick_value=1.0
    )
    # risk_per_lot = 10000. lot = 0.0001
    assert lot == 0.01

def test_lot_clamped_to_max():
    lot = LotCalculator.calculate(
        risk_amount=100000.0,
        sl_distance=1.0,
        tick_size=0.01,
        tick_value=1.0,
        volume_max=50.0
    )
    assert lot == 50.0

def test_zero_risk_returns_min():
    lot = LotCalculator.calculate(0.0, 1.0, 0.01, 1.0)
    assert lot == 0.01
