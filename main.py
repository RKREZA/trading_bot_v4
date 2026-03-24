"""
TRADING BOT V3 - Main Entry Point
Orchestrates MT5 connection, data fetching, strategy analysis, and dashboard.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

from core.logger import setup_logging
from core.connection import MT5Connection
from core.data_fetcher import DataFetcher
from core.backtest import BacktestEngine
from core.strategy_engine import StrategyEngine
from dashboard import Dashboard, AnalysisLogger

# Load .env file if it exists
load_dotenv()

logger = logging.getLogger("trading_bot.main")


class TradingBot:
    """
    Main trading bot orchestrator.

    Coordinates:
    - MT5Connection for connectivity
    - DataFetcher for candle data (with caching)
    - StrategyEngine for signal generation
    - Dashboard for live CLI display
    - BacktestEngine for historical testing
    """

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.analysis_logger = AnalysisLogger(max_entries=100)
        self.strategy = StrategyEngine(self.config, self.analysis_logger)
        self.dashboard = Dashboard(self.config, self.analysis_logger)
        self.connection = MT5Connection()
        self.data_fetcher = DataFetcher()

        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.win_count = 0
        self.loss_count = 0
        self.running = False

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Config %s not found, using defaults", config_path)
            return {
                "symbol": "BTCUSDm",
                "risk_per_trade": 2.0,
                "max_daily_trades": 5,
                "daily_goal": 100.0,
                "strategy": {"min_confluence_score": 4, "min_confidence": 50, "cooldown_candles": 3},
                "backtest": {"initial_balance": 1000, "spread_pips": {"XAUUSDm": 30, "BTCUSDm": 50}, "candles": {"H4": 600, "M30": 4800, "M15": 9600}},
                "symbols_config": {
                    "XAUUSDm": {"point": 0.01, "contract_size": 100, "lot": 0.1},
                    "BTCUSDm": {"point": 0.01, "contract_size": 1, "lot": 0.01},
                },
            }

    @staticmethod
    def _get_session() -> str:
        """
        Determine current trading session based on UTC+0 hour conventions.
        Correctly handles the London/NY overlap (13:00-16:59).
        """
        hour = datetime.now().hour
        if 13 <= hour < 17:
            return "LONDON/NY"
        if 8 <= hour < 17:
            return "LONDON"
        if 17 <= hour < 22:
            return "NEW_YORK"
        if 0 <= hour < 9:
            return "TOKYO"
        return "CLOSED"

    def _update_dashboard_state(self, signal=None, analysis=None):
        """Push latest state to the dashboard."""
        if signal:
            self.dashboard.signal = {
                "direction": signal.direction,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "confidence": signal.confidence,
                "confluence_score": signal.confluence_score,
                "reasons": signal.reasons,
                "rejection_type": signal.rejection_type,
            }
        else:
            self.dashboard.signal = None
        if analysis:
            self.dashboard.h4_trend = analysis.get("h4_trend", "RANGING")
            self.dashboard.m30_structure = analysis.get("m30_structure", "NEUTRAL")
        self.dashboard.session = self._get_session()
        self.dashboard.daily_pnl = self.daily_pnl
        self.dashboard.daily_trades = self.daily_trades
        self.dashboard.win_count = self.win_count
        self.dashboard.loss_count = self.loss_count

    # ------------------------------------------------------------------
    # Live Trading
    # ------------------------------------------------------------------

    def run_live(self):
        """Run the live trading loop with error handling and auto-reconnect."""
        if not self.connection.connect():
            return

        symbol = self.config.get("symbol", "BTCUSDm")
        self.dashboard.selected_symbol = symbol
        self.analysis_logger.log(f"Starting live trading for {symbol}")
        self.running = True
        self.dashboard.running = True
        self.dashboard.account_info = self.connection.account_info
        self.dashboard.start()

        try:
            while self.running:
                start_time = time.time()

                try:
                    # Health check with auto-reconnect
                    if not self.connection.ensure_connected():
                        self.analysis_logger.log("Connection lost — waiting to reconnect...", "ERROR")
                        time.sleep(5)
                        continue

                    # Update account info from connection
                    self.dashboard.account_info = self.connection.account_info

                    symbol_info = self.data_fetcher.get_symbol_info(symbol)
                    if not symbol_info:
                        time.sleep(1)
                        continue

                    mid_price = (symbol_info["bid"] + symbol_info["ask"]) / 2
                    self.dashboard.tick = {
                        "bid": symbol_info["bid"],
                        "ask": symbol_info["ask"],
                        "price": mid_price,
                        "spread": symbol_info["spread"] * symbol_info["point"],
                        "contract_size": symbol_info["contract_size"],
                    }

                    # Fetch candles (cached per timeframe)
                    h4_candles = self.data_fetcher.fetch_candles(symbol, "H4", 250)
                    m30_candles = self.data_fetcher.fetch_candles(symbol, "M30", 1540)
                    m15_candles = self.data_fetcher.fetch_candles(symbol, "M15", 2000)

                    if h4_candles and m30_candles and m15_candles:
                        session = self._get_session()
                        signal = self.strategy.analyze(
                            symbol, h4_candles, m30_candles, m15_candles,
                            mid_price, session=session,
                        )

                        h4_trend = self.strategy._determine_trend(h4_candles)
                        analysis_state = {
                            "h4_trend": h4_trend,
                            "m30_structure": (
                                "BULLISH" if h4_trend == "BULLISH"
                                else ("BEARISH" if h4_trend == "BEARISH" else "NEUTRAL")
                            ),
                        }
                        self._update_dashboard_state(signal, analysis_state)

                except Exception as e:
                    logger.exception("Error in trading cycle: %s", e)
                    self.analysis_logger.log(f"Cycle error: {e}", "ERROR")

                cycle_ms = (time.time() - start_time) * 1000
                self.dashboard.update(cycle_ms)
                time.sleep(1)

        except KeyboardInterrupt:
            self.analysis_logger.log("Stopping (Ctrl+C)...")
            self.running = False
        finally:
            self.dashboard.stop()
            self.connection.disconnect()

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def run_backtest(self, symbol: str):
        """Fetch data from MT5 and run the backtest engine."""
        if not self.connection.connect():
            return

        logger.info("Fetching data for %s...", symbol)
        
        bt_candles = self.config.get("backtest", {}).get("candles", {"H4": 600, "M30": 4800, "M15": 9600})
        h4_candles = self.data_fetcher.fetch_candles(symbol, "H4", bt_candles.get("H4", 600))
        m30_candles = self.data_fetcher.fetch_candles(symbol, "M30", bt_candles.get("M30", 4800))
        m15_candles = self.data_fetcher.fetch_candles(symbol, "M15", bt_candles.get("M15", 9600))

        if not h4_candles or not m30_candles or not m15_candles:
            logger.error("Failed to fetch data for %s", symbol)
            self.connection.disconnect()
            return

        self.connection.disconnect()
        logger.info("MT5 connection closed — running backtest offline")

        engine = BacktestEngine(self.config, self.strategy)
        engine.run(symbol, h4_candles, m30_candles, m15_candles)


def main():
    """CLI entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Trading Bot V3")
    parser.add_argument("--backtest", action="store_true", help="Run backtest mode")
    parser.add_argument("--symbol", type=str, default="BTCUSDm", help="Trading symbol")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    args = parser.parse_args()

    bot = TradingBot(args.config)
    bot.config["symbol"] = args.symbol

    if args.backtest:
        # Suppress verbose strategy logs during backtest so they don't break the progress bar
        logging.getLogger("trading_bot.strategy").setLevel(logging.WARNING)
        bot.run_backtest(args.symbol)
    else:
        bot.run_live()


if __name__ == "__main__":
    main()
