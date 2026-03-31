import traceback
try:
    from core.connection import MT5Connection
    print("connection OK")
except Exception as e:
    traceback.print_exc()

try:
    from core.data_fetcher import DataFetcher
    print("data_fetcher OK")
except Exception as e:
    traceback.print_exc()

try:
    from core.lot_calculator import LotCalculator
    print("lot_calculator OK")
except Exception as e:
    traceback.print_exc()

try:
    from core.strategy_engine import StrategyEngine
    print("strategy_engine OK")
except Exception as e:
    traceback.print_exc()

try:
    from core.backtester import BacktestEngine
    print("backtester OK")
except Exception as e:
    traceback.print_exc()

try:
    from core.ai_advisor import AIAdvisor
    print("ai_advisor OK")
except Exception as e:
    traceback.print_exc()

try:
    from core.risk_manager import RiskManager
    print("risk_manager OK")
except Exception as e:
    traceback.print_exc()

try:
    from dashboard import Dashboard, AnalysisLogger
    print("dashboard OK")
except Exception as e:
    traceback.print_exc()

print("--- Import test complete ---")
