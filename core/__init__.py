"""Trading Bot V3 - Core Module"""
from .strategy_engine import StrategyEngine, TradeSignal
from .connection import MT5Connection
from .data_fetcher import DataFetcher
from .backtester import BacktestEngine
from .risk_manager import RiskManager
from .performance import PerformanceMetrics
from .ai_advisor import AIAdvisor
from .validation import ValidationSuite
from .logger import setup_logging

__all__ = [
    "StrategyEngine",
    "TradeSignal",
    "MT5Connection",
    "DataFetcher",
    "BacktestEngine",
    "RiskManager",
    "PerformanceMetrics",
    "AIAdvisor",
    "ValidationSuite",
    "setup_logging",
]