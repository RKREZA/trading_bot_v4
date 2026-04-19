# Trading Bot Scripts
=====================

This directory contains helper scripts for running the trading bot in different modes.

## Available Scripts

### Live Trading
- `start_live_trading.sh` (Linux/Mac)
- `start_live_trading.bat` (Windows)

These scripts start the live trading bot with proper environment setup.
They handle:
- Activating the Python virtual environment
- Checking for required configuration files
- Starting the main trading orchestrator

#### Usage:
```bash
# Linux/Mac
./scripts/start_live_trading.sh [SYMBOL] [STRATEGIES]

# Windows
scripts\start_live_trading.bat [SYMBOL] [STRATEGIES]
```

**Arguments:**
- `SYMBOL`: Trading symbol (default: XAUUSDm)
- `STRATEGIES`: Comma-separated list of strategies (default: LiquiditySweepBreakout,RangeBounce)

### Backtesting
- `run_backtest.sh` (Linux/Mac)

This script runs the backtesting suite with various options.

#### Usage:
```bash
./scripts/run_backtest.sh [options]
```

**Options:**
- `--symbol SYMBOL`: Trading symbol (default: XAUUSDm)
- `--from DATE`: Start date (default: 2024-01-01)
- `--to DATE`: End date (default: 2026-04-09)
- `--monte-carlo`: Run Monte Carlo simulation
- `--walk-forward`: Run Walk-Forward optimization
- `--stress-test`: Run Stress Testing suite
- `--help`: Show help message

#### Examples:
```bash
# Run standard backtest
./scripts/run_backtest.sh

# Run backtest for EURUSDm with walk-forward optimization
./scripts/run_backtest.sh --symbol EURUSDm --walk-forward

# Run full validation suite
./scripts/run_backtest.sh --monte-carlo --walk-forward --stress-test
```

## Requirements
- Python 3.11+
- Virtual environment (myenv) with dependencies installed
- Valid MT5 credentials in .env file (for live trading)
- Configuration files in config/ and configs/symbols/

## Notes
- Make sure to give execute permissions to shell scripts on Linux/Mac:
  ```bash
  chmod +x scripts/*.sh
  ```
- The Windows batch script can be run directly from Command Prompt or PowerShell.
- For live trading, ensure MetaTrader 5 is running and logged in with "Algo Trading" enabled.