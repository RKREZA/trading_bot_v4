import json, sys, os
sys.path.append(os.getcwd())
from dotenv import load_dotenv
load_dotenv()
from backtesting import PortfolioBacktester
from strategies import create_strategy, STRATEGY_REGISTRY
from core.portfolio_manager import PortfolioManager

with open("config.json") as f:
    config = json.load(f)

pascal_name = "TrendFollowing"
st_class = STRATEGY_REGISTRY["TRENDFOLLOWING"]
strategy_obj = st_class(pascal_name, config=config)
print(f"Strategy ID: {strategy_obj.strategy_id}")
print(f"Enabled: {strategy_obj.enabled}")

sym = "XAUUSDm"
print(f"Symbol allowed: {strategy_obj.is_symbol_allowed(sym)}")

pm = PortfolioManager(config)
bal = pm.get_strategy_balance(100.0, strategy_obj.strategy_id)
print(f"Balance for {strategy_obj.strategy_id}: {bal}")

# Check the active_strategies filter in PortfolioBacktester
bt = PortfolioBacktester(config)
strategies = [strategy_obj]
active = []
for s in strategies:
    if not getattr(s, "enabled", True):
        print(f"  BLOCKED: not enabled")
        continue
    if not s.is_symbol_allowed(sym):
        print(f"  BLOCKED: symbol not allowed")
        continue
    if bt.portfolio_manager.get_strategy_balance(100.0, s.strategy_id) <= 0:
        print(f"  BLOCKED: zero allocation")
        continue
    active.append(s)
    print(f"  ACTIVE: {s.strategy_id}")

print(f"\nActive strategies: {len(active)}")
