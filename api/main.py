from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any

from core.time.time_service import time_service
from core.data.mt5_service import mt5_service
from core.config.config_system import ConfigManager
from core.risk.risk_engine import RiskEngine
from core.execution.execution_engine import ExecutionEngine
from core.execution.reconciliation_engine import ReconciliationEngine
from core.strategy.smc_strategy import SMCStrategy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="Antigravity V5 API", version="5.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core
config_manager = ConfigManager("config/global_config.json")
risk_engine = RiskEngine(config_manager.current_config.risk.model_dump())
execution_engine = ExecutionEngine()
recon_engine = ReconciliationEngine(execution_engine)

# Active Strategies
active_strategies = {
    "SMC": SMCStrategy("XAUUSD")
}

class AppState:
    def __init__(self):
        self.is_running = False
        self.connected_clients: List[WebSocket] = []

state = AppState()

# --- Models ---
class SystemStatus(BaseModel):
    is_running: bool
    mt5_connected: bool
    server_time: str
    local_time: str
    drift_sec: float
    active_strategies: List[str]
    account: Dict[str, Any]

# --- Background Task: Core Loop ---
async def trading_loop():
    while True:
        if state.is_running:
            try:
                # 1. Update Time & Ticks
                # For demo, we just tick XAUUSD
                tick = mt5_service.get_tick("XAUUSD")
                if tick:
                    # 2. Run Strategies
                    # (Simplified: logic would normally be triggered on new candle)
                    pass
                
                # 3. Reconciliation
                recon_engine.reconcile()
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
        
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    # Connect MT5
    mt5_service.connect()
    # Start loop
    asyncio.create_task(trading_loop())

# --- Routes ---
@app.get("/status", response_model=SystemStatus)
async def get_status():
    server_time = time_service.get_server_time()
    local_time = datetime.now(timezone.utc)
    drift = (local_time - server_time).total_seconds()
    
    account = recon_engine.get_account_summary()
    
    return SystemStatus(
        is_running=state.is_running,
        mt5_connected=mt5_service.connected,
        server_time=server_time.isoformat(),
        local_time=local_time.isoformat(),
        drift_sec=drift,
        active_strategies=list(active_strategies.keys()),
        account=account
    )

@app.post("/control/start")
async def start_trading():
    if not mt5_service.connected:
        if not mt5_service.connect():
            raise HTTPException(status_code=500, detail="Failed to connect to MT5")
    state.is_running = True
    return {"status": "started"}

@app.post("/control/stop")
async def stop_trading():
    state.is_running = False
    return {"status": "stopped"}

@app.post("/control/kill")
async def kill_switch():
    state.is_running = False
    risk_engine.trigger_kill_switch("Manual Kill Triggered")
    # In production: send close_all orders
    return {"status": "killed"}

@app.get("/config")
async def get_config():
    return config_manager.current_config.model_dump()

@app.post("/config/update")
async def update_config(new_config: Dict[str, Any]):
    # Simplified update
    config_manager.current_config = config_manager.current_config.model_copy(update=new_config)
    config_manager.save()
    return {"status": "updated"}

# --- WebSockets ---
@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")
    state.connected_clients.append(websocket)
    try:
        while True:
            try:
                # Send real-time updates (account metrics, logs)
                account = recon_engine.get_account_summary()
                await websocket.send_json({
                    "type": "METRICS",
                    "data": account,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Error sending websocket message: {e}")
            
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {websocket.client}")
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)

# --- Backtest Models ---
class BacktestRequest(BaseModel):
    strategies: List[Dict[str, Any]]
    initial_balance: float = 10000.0
    stress_multiplier: float = 1.0
    monte_carlo_count: int = 50

@app.post("/backtest/run")
async def run_backtest(req: BacktestRequest):
    try:
        from core.backtest.engine import BacktestEngine
        from core.strategy.smc_strategy import SMCStrategy
        
        # 1. Instantiate Strategies with custom params
        strats = []
        for s_info in req.strategies:
            # For demo, we only support SMCStrategy. In production, use a Factory.
            strat = SMCStrategy(s_info.get("symbol", "XAUUSD"))
            if "parameters" in s_info:
                # Mock parameter injection - in real code, strategies would take params in __init__
                pass
            strats.append(strat)
            
        # 2. Fetch Historical Data (Mock for demo)
        # In production: mt5_service.get_historical_data(...)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="H")
        mock_prices = [2000 + i + (10 * np.random.randn()) for i in range(100)]
        df = pd.DataFrame({
            "open": mock_prices,
            "high": [p + 2 for p in mock_prices],
            "low": [p - 2 for p in mock_prices],
            "close": [p + 0.5 for p in mock_prices],
            "tick_volume": 100
        }, index=dates)
        
        data_map = {"XAUUSD": df}
        
        # 3. Run Engine
        engine = BacktestEngine(strats, req.initial_balance)
        report = engine.run(data_map, spread_multiplier=req.stress_multiplier)
        
        return report
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import numpy as np # Added for mock data
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
