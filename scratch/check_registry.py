import sys
import os
sys.path.append(os.getcwd())

from strategies import STRATEGY_REGISTRY
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Registry_Check")

def check_registry():
    logger.info("=" * 60)
    logger.info(" STRATEGY REGISTRY AUDIT ")
    logger.info("=" * 60)
    
    expected = {"TRENDFOLLOWING", "LIQUIDITYSWEEPBREAKOUT", "SMARTMEANREVERSION"}
    actual = set(STRATEGY_REGISTRY.keys())
    
    logger.info(f"Registered Strategies: {actual}")
    
    if actual == expected:
        logger.info("✅ Registry is precisely matched to Institutional Engine requirements.")
    else:
        logger.error(f"❌ Registry mismatch! Expected {expected}, got {actual}")
        
    logger.info("=" * 60)

if __name__ == "__main__":
    check_registry()
