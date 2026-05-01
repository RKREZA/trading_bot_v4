import logging
import os
import sys
import json
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler


class BrokerClock:
    """
    Simulated clock to override system time in logs during backtesting or live sync.
    """
    _simulated_time: float = None

    @classmethod
    def set_time(cls, timestamp: float):
        cls._simulated_time = timestamp

    @classmethod
    def get_time(cls) -> float:
        return cls._simulated_time if cls._simulated_time is not None else time.time()


class StructuredFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ts = BrokerClock.get_time()
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime(datefmt) if datefmt else dt.strftime("%Y-%m-%d %H:%M:%S")

    def format(self, record):
        log_data = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "trade_data"):
            log_data["trade"] = record.trade_data
        return json.dumps(log_data)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout)
            self.flush()
        except Exception:
            self.handleError(record)


def _try_configure_structlog():
    """Configure structlog processors if available. Falls back gracefully."""
    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        return True
    except ImportError:
        return False


def setup_logging(log_dir: str = "logs", level: int = logging.INFO, console: bool = False) -> logging.Logger:
    """
    Configure application-wide logging with both human-readable console output
    and structured JSON file rotation.
    """
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger("trading_bot")
    root_logger.setLevel(logging.DEBUG)

    if root_logger.handlers:
        return root_logger

    class BrokerTimeFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            ts = BrokerClock.get_time()
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return dt.strftime(datefmt) if datefmt else dt.strftime("%Y-%m-%d %H:%M:%S")

    console_formatter = BrokerTimeFormatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if console:
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    log_file = os.path.join(log_dir, "trading_bot.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )

    def log_namer(default_name):
        if not (default_name.split('.')[-1]).isdigit():
            return default_name
        base, ext = os.path.splitext(file_handler.baseFilename)
        index = default_name.split('.')[-1]
        return f"{base}_{index}{ext}"

    file_handler.namer = log_namer
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(StructuredFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    root_logger.addHandler(file_handler)

    has_structlog = _try_configure_structlog()
    root_logger.info(f"Logging initialized (structlog={'yes' if has_structlog else 'fallback'})")
    return root_logger
