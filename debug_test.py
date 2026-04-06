from core.risk_engine import RiskEngine

cfg = {
    'risk_governance': {'risk_per_trade_pct': 0.5},
    'symbols_config': {'XAUUSDm': {'point': 0.01, 'tick_value': 1.0, 'min_lot': 0.01, 'max_lot': 50.0, 'lot_step': 0.01}}
}
re = RiskEngine(cfg)

# Step by step
balance = 1000.0
sl_dist = 5.0   # 500 pt SL
point = 0.01
tick_value = 1.0

risk_amount = balance * (0.5 / 100)
print(f"Risk amount: {risk_amount}")

sl_points = sl_dist / point
print(f"SL in points: {sl_points}")

raw_lot = risk_amount / (sl_points * tick_value)
print(f"Raw lot: {raw_lot}")

# Check cost ratio
spread_cost = 15 * tick_value * raw_lot
print(f"Spread cost: {spread_cost}")
print(f"Cost ratio: {spread_cost / risk_amount * 100}%")
print(f"Cost threshold: 30%")

lot = re.calculate_lot_size(balance, sl_dist, point, tick_value, 'XAUUSDm')
print(f"Final lot: {lot}")
