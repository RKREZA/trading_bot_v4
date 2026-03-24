"""Trading Bot V3 - Core Module"""
from .strategy_engine import StrategyEngine, TradeSignal
from .connection import MT5Connection
from .data_fetcher import DataFetcher
from .backtest import BacktestEngine
from .logger import setup_logging

__all__ = [
    "StrategyEngine",
    "TradeSignal",
    "MT5Connection",
    "DataFetcher",
    "BacktestEngine",
    "setup_logging",
]
