"""
TRADING BOT V3 - Logging Setup
Configures console + rotating file logging
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: str = "logs", level: int = logging.INFO, console: bool = False) -> logging.Logger:
    """
    Configure application-wide logging with console and file handlers.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    root_logger = logging.getLogger("trading_bot")
    root_logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers on re-init
    if root_logger.handlers:
        return root_logger
    
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Rotating file handler
    log_file = os.path.join(log_dir, "trading_bot.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    root_logger.info("Logging initialized — file: %s", log_file)
    return root_logger
