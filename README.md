# Trading Bot V3

**M30/M15 Rejection Pattern Strategy**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test connection first!
python test_connection.py

# Run backtest
python main.py --backtest --symbol BTCUSDm

# Run live trading
python main.py --symbol BTCUSDm
```

## Troubleshooting

If you see "CONNECTION FAILED":
1. Open MT5 terminal and login manually
2. Enable "Algo Trading" in MT5 settings
3. Check your internet connection
4. Run: python test_connection.py

## Files

- `main.py` - Main entry point
- `dashboard.py` - Rich CLI dashboard
- `test_connection.py` - Test MT5 connection
- `config.json` - Configuration
- `core/strategy_engine.py` - Strategy logic
