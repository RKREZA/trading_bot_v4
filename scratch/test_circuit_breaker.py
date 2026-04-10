import os
import json
import sys
import time
from typing import Dict

# Add project root to path
sys.path.append(os.getcwd())

from core.risk.risk_guardian import RiskGuardian

def test_circuit_breaker():
    print("--- Institutional Circuit Breaker Test ---")
    
    # Setup mock config
    config = {
        "risk_governance": {
            "strategy_loss_halt_pct": 3.0
        }
    }
    
    # Initialize RiskGuardian (will load existing health if present)
    rg = RiskGuardian(config)
    sid = "TestStrategy"
    
    # 1. Reset state for clean test
    rg.strategy_performance[sid] = []
    rg.strategy_status[sid] = "OK"
    rg._save_health_state()
    
    print(f"Initial Status for {sid}: {rg.strategy_status[sid]}")
    
    # 2. Record a sequence of losses totaling > 3%
    # Partition Balance = $1000
    alloc_bal = 1000.0
    
    print("Recording losses...")
    rg.record_strategy_result(sid, -15.0, alloc_bal) # -1.5%
    rg.record_strategy_result(sid, -20.0, alloc_bal) # -2.0% (Total -3.5%)
    
    # 3. Check governance
    allowed, reason = rg.check_strategy_governance(sid)
    print(f"Governance Check: Allowed={allowed}, Reason={reason}")
    print(f"Status for {sid}: {rg.strategy_status.get(sid)}")
    
    # 4. Verify Persistence
    if os.path.exists(rg.health_file):
        with open(rg.health_file, 'r') as f:
            state = json.load(f)
            persisted_status = state.get("status", {}).get(sid)
            print(f"Persisted Status in JSON: {persisted_status}")
            
    if not allowed and rg.strategy_status.get(sid) == "HALTED":
        print("✅ CIRCUIT BREAKER VERIFIED: Strategy halted and persisted correctly.")
    else:
        print("❌ CIRCUIT BREAKER FAILURE: Strategy did not halt as expected.")

if __name__ == "__main__":
    test_circuit_breaker()
