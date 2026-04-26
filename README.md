# V5-INSIGNIA Institutional Trading System

**Institutional N-Pattern Grid Engine** — Featuring high-fidelity event-driven simulation, institutional risk governance, and N-Pattern impulse fusion technology.

## Key Features
- **MT5 Production Connection**: Robust terminal interface with auto-recovery and thread-safe execution.
- **Unified Risk Governance**: Shared `RiskGuardian` for both backtesting and live trading (Zero Parity Gap).
- **Institutional Execution**: Latency simulation, variable spread modeling, and direction-aware slippage.
- **N-Pattern Grid Strategy**: Advanced impulse fusion and grid scaling with 1:1 risk-to-reward logic.
- **Advanced Validation**: Monte Carlo, Walk-Forward Optimization, and Stress Testing suites.

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

### Backtesting & Audit
```bash
# Run a production benchmark
python backtest.py --symbol XAUUSDm --from 2024-01-01 --to 2026-04-09

# Run with full validation suite
python backtest.py --symbol XAUUSDm --from 2024-01-01 --to 2026-04-09 --monte-carlo --walk-forward --stress-test
```

### Live Trading
```bash
# Start the production orchestrator
python main.py --symbol XAUUSDm
```

## Project Structure

```
trading_bot_v3/
├── main.py                  # Live Trading Entry Point
├── backtest.py              # Backtest & Audit CLI
├── dashboard.py             # Institutional Telemetry TUI
├── config.json              # Global Configuration & Governance
├── core/
│   ├── risk/                # Unified Risk Governance (RiskGuardian)
│   ├── execution/           # Order Management & Simulation (OrderManager)
│   ├── common/              # Shared Types (TradeSignal, CandleArray)
│   └── strategy_orchestrator.py # Multi-Strategy Coordinator
├── strategies/              # Strategy Implementations (V5-INSIGNIA)
├── backtesting/             # High-Fidelity Simulator & Validators
├── tests/                   # Professional Verification Suite
└── logs/                    # Institutional Audit Trail
```

## Troubleshooting

1. **MT5 Sync**: Ensure "Algo Trading" is enabled in MT5 and the terminal is logged in.
2. **Data Gaps**: Ensure you have enough historical data downloaded in MT5 for all timeframes (M1 to H1).
3. **Connectivity**: Use `python test_connection.py` (if available) or check `logs/trading_bot.log`.
