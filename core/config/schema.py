from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, List, Any, Optional


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    strategy_health_file: str = "config/strategy_health.json"
    state_db_path: str = "config/regime_state_v3.db"
    shadow_fill_audit: str = "logs/shadow_fill_audit.csv"
    crash_report: str = "logs/crash_report.log"


class RiskGovernanceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 5.0
    max_drawdown_halt_pct: float = 15.0
    max_consecutive_losses: int = 10
    max_parallel_strategies: int = 4
    max_net_exposure_long: int = 3
    max_net_exposure_short: int = 3
    min_notional_value: float = 0.0
    min_confidence: float = 0.5
    min_rr: float = 2.0
    max_concurrent_trades: int = 3
    min_tick_density: int = 45

    @field_validator(
        "risk_per_trade_pct", "max_daily_loss_pct", "max_drawdown_halt_pct",
        "min_notional_value", "min_confidence", "min_rr",
        mode="before",
    )
    @classmethod
    def coerce_float(cls, v: Any) -> float:
        try:
            return float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Expected numeric value, got {type(v)}: {v}")


class PartialFillConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    threshold_lots: float = 5.0
    part1_ratio: float = 0.4
    part2_ratio: float = 0.6
    worse_slippage_multiplier: float = 2.0


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    latency_ms: int = 150
    max_spread_points: float = 500.0
    entry_slippage_points: float = 1.0
    sl_exit_slippage_points: float = 2.0
    phase1_lot_ceiling: float = 2.0
    shadow_drift_p95_threshold: float = 0.5
    max_retries: int = 3
    retry_delay_sec: float = 1.0
    slippage_deviation_points: int = 10
    partial_fill: PartialFillConfig = Field(default_factory=PartialFillConfig)


class PartialProfitConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    phase1_rr_target: float = 1.0
    phase1_close_pct: int = 50
    move_to_be_at_partial: bool = True
    session_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class TrailingStopConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    phase2_rr_threshold: float = 2.0
    phase2_be_offset_pct: float = 0.2
    phase3_trail_mult: float = 1.5
    session_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class NewsFilterConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    cache_file: str = "config/news_cache.json"
    source_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    impact_levels: List[str] = Field(default_factory=lambda: ["High"])
    buffer_before_min: int = 30
    buffer_after_min: int = 15
    auto_close_before_min: int = 5


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    deterministic: bool = True
    random_seed: int = 42
    initial_balance: float = 10000.0
    commission_per_lot: float = 7.0
    monte_carlo_iterations: int = 2500
    stress_multiplier: float = 1.0


class SymbolInfoConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    point: float = 0.01
    spread_pips: int = 6
    tick_value: float = 1.0
    contract_size: float = 100.0
    min_lot: float = 0.01
    max_lot: float = 100.0
    lot_step: float = 0.01
    commission_per_lot: float = 7.0
    backtest_timeframe: str = "M1"
    initial_balance_per_strategy: float = 10000.0


class StrategyInstanceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = ""
    symbol: str = ""
    timeframe: int = 5
    enabled: bool = False
    parameters: Dict[str, Any] = Field(default_factory=dict)
    sessions: List[str] = Field(default_factory=lambda: ["LONDON", "NEW_YORK", "LONDON/NY"])
    min_confidence: Optional[float] = None
    min_rr: Optional[float] = None


class MTAccountConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    login: int = 0
    password: str = ""
    server: str = ""
    broker_utc_offset: int = 0


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    version: str = "6.0"
    magic_number: int = 234000
    mt5: MTAccountConfig = Field(default_factory=MTAccountConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    risk_governance: RiskGovernanceConfig = Field(default_factory=RiskGovernanceConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    partial_profit: PartialProfitConfig = Field(default_factory=PartialProfitConfig)
    trailing_stop: TrailingStopConfig = Field(default_factory=TrailingStopConfig)
    news_filter: NewsFilterConfig = Field(default_factory=NewsFilterConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    strategies: List[StrategyInstanceConfig] = Field(default_factory=list)
    symbols_config: Dict[str, SymbolInfoConfig] = Field(default_factory=dict)
    portfolio_allocations: Dict[str, Any] = Field(default_factory=dict)
    data_cache_path: str = "data_cache"
