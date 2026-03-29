def simulate_lot_calc(base_lot, ai_multi, min_lot, max_lot_broker, max_lot_config, lot_step):
    print(f"\nSimulating: base={base_lot}, ai_multi={ai_multi}, step={lot_step}")
    
    # The logic from main.py
    lot_size = base_lot * ai_multi
    
    if lot_step > 0:
        lot_size = round(lot_size / lot_step) * lot_step
    
    max_lot = min(max_lot_broker, max_lot_config)
    lot_size = max(min_lot, min(max_lot, lot_size))
    lot_size = round(lot_size, 3)
    
    print(f"Result: {lot_size}")
    
    # Validation
    is_step_valid = (round(lot_size / lot_step, 8) % 1 == 0) if lot_step > 0 else True
    is_min_valid = lot_size >= min_lot
    is_max_valid = lot_size <= max_lot
    
    print(f"Validations: Step={is_step_valid}, Min={is_min_valid}, Max={is_max_valid}")
    return lot_size

if __name__ == "__main__":
    # BTCUSDm example from user's account
    # min=0.01, max=200, step=0.01
    cfg_max = 5.0
    broker_min = 0.01
    broker_max = 200.0
    broker_step = 0.01
    
    # Case 1: AI reduces risk below minimum
    simulate_lot_calc(0.01, 0.5, broker_min, broker_max, cfg_max, broker_step) # Should be 0.01
    
    # Case 2: AI increases risk with non-step result
    simulate_lot_calc(0.1, 1.15, broker_min, broker_max, cfg_max, broker_step) # Should be 0.12 or 0.11
    
    # Case 3: Very small AI multiplier
    simulate_lot_calc(0.01, 0.1, broker_min, broker_max, cfg_max, broker_step) # Should be 0.01
    
    # Case 4: Large lot hitting config max
    simulate_lot_calc(4.0, 1.5, broker_min, broker_max, cfg_max, broker_step) # Should be 5.0
