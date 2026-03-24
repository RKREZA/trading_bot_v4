# Trading Bot V3

**Hybrid Breakout Strategy** — Multi-timeframe analysis with session filtering.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure credentials** — copy `.env.example` to `.env` and fill in your MT5 credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
   ```

3. **Configure strategy** — edit `config.json` for symbol, risk, and strategy parameters.

## Usage

```bash
# Test MT5 connection
python test_connection.py

# Run backtest
python main.py --backtest --symbol BTCUSDm

# Run live trading
python main.py --symbol BTCUSDm
```

## Project Structure

```
trading_bot_v3/
├── main.py                  # Entry point & orchestrator
├── dashboard.py             # Rich CLI dashboard
├── config.json              # Strategy & symbol configuration
├── .env                     # MT5 credentials (not in git)
├── core/
│   ├── strategy_engine.py   # Hybrid breakout/pullback strategy
│   ├── connection.py        # MT5 connection + auto-reconnect
│   ├── data_fetcher.py      # Candle fetching with caching
│   ├── backtest.py          # Backtesting engine
│   └── logger.py            # Rotating file logger
├── tests/
│   ├── test_strategy.py     # Strategy unit tests
│   └── test_backtest.py     # Backtest unit tests
└── logs/                    # Log files (auto-created)
```

## Troubleshooting

If you see "CONNECTION FAILED":
1. Open MT5 terminal and login manually
2. Enable "Algo Trading" in MT5 settings
3. Verify your `.env` credentials
4. Run: `python test_connection.py`
