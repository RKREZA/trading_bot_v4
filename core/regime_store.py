from abc import ABC, abstractmethod
import sqlite3
import json
import os
import logging
from dataclasses import asdict
from core.regime_detector import RegimeState

class RegimeStateStore(ABC):
    """Abstract interface for Regime State Persistence."""
    @abstractmethod
    def load(self, strategy_id: str) -> RegimeState:
        pass

    @abstractmethod
    def save(self, strategy_id: str, state: RegimeState):
        pass

class SQLiteWALRegimeStore(RegimeStateStore):
    """Production-grade SQLite implementation with WAL mode and atomic commits."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger("trading_bot.regime_store")
        self._init_db()

    def _init_db(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                # Versioned schema: v3_state
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS v3_regime_state (
                        strategy_id TEXT PRIMARY KEY,
                        state_json TEXT,
                        version INTEGER DEFAULT 3,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
        except Exception as e:
            self.logger.error(f"Failed to initialize Regime SQLite DB: {e}")

    def load(self, strategy_id: str) -> RegimeState:
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT state_json FROM v3_regime_state WHERE strategy_id = ?", (strategy_id,))
                row = cursor.fetchone()
                if row:
                    data = json.loads(row[0])
                    return RegimeState(**data)
        except Exception as e:
            self.logger.error(f"Failed to load regime state for {strategy_id}: {e}")
        
        return RegimeState()

    def save(self, strategy_id: str, state: RegimeState):
        try:
            state_json = json.dumps(asdict(state))
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO v3_regime_state (strategy_id, state_json, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                    (strategy_id, state_json)
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save regime state for {strategy_id}: {e}")

class MemoryRegimeStore(RegimeStateStore):
    """Stateless in-memory store for backtesting and deterministic replay."""
    
    def __init__(self):
        self._states = {}

    def load(self, strategy_id: str) -> RegimeState:
        return self._states.get(strategy_id, RegimeState())

    def save(self, strategy_id: str, state: RegimeState):
        self._states[strategy_id] = state
