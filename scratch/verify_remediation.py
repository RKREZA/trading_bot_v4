import sys
import os
sys.path.append(os.getcwd())

from comprehensive_backtest import ComprehensiveBacktestSuite

def verify_mean_reversion():
    print("Verifying SmartMeanReversion Resilience...")
    suite = ComprehensiveBacktestSuite()
    result = suite.run_single_strategy_backtest(
        strategy_name="SmartMeanReversion",
        symbol="XAUUSDm",
        market_condition="VOLATILE", # Simulate trend/volatility
        volatility=5.0
    )
    
    if result and result.get("status") == "SUCCESS":
        print(f"Status: SUCCESS")
        print(f"Sharpe: {result.get('sharpe_ratio', 0):.2f}")
        print(f"Max DD: {result.get('max_drawdown', 'N/A')}")
        print(f"Return: {result.get('cumulative_return', 'N/A')}")
    else:
        print(f"Status: {result.get('status', 'FAILED')}")
        if "error" in result:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    verify_mean_reversion()
