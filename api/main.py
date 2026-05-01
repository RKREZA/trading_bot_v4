import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import services
from api.routers import status, trading, strategies, config, backtesting, data, accounts, logs
from api.websocket.handlers import router as ws_router
from core.common.events import event_bus
from core.observability.audit import audit_trail
from core.logger import setup_logging

setup_logging(console=True)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting trading platform API...")
    await services.initialize()
    event_bus_task = asyncio.create_task(event_bus.start())
    await audit_trail.start()
    audit_trail.log_system("Platform API started", data={"version": "6.0"})
    yield
    logger.info("Shutting down...")
    audit_trail.log_system("Platform API shutting down")
    await audit_trail.stop()
    await services.shutdown()
    event_bus_task.cancel()


app = FastAPI(
    title="Trading Platform API",
    version="6.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status.router)
app.include_router(trading.router)
app.include_router(strategies.router)
app.include_router(config.router)
app.include_router(backtesting.router)
app.include_router(data.router)
app.include_router(accounts.router)
app.include_router(logs.router)
app.include_router(ws_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
