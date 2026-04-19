#!/bin/bash
# Live Trading Startup Script
# ===========================
# This script starts the trading bot with proper environment setup

echo "Starting V5-INSIGNIA Live Trading System"
echo "========================================"

# Change to script directory
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -f "myenv/Scripts/activate" ]; then
    echo "Activating Python virtual environment..."
    source myenv/Scripts/activate
elif [ -f "myenv/bin/activate" ]; then
    echo "Activating Python virtual environment..."
    source myenv/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found!"
    echo "Please copy .env.example to .env and configure your MT5 credentials:"
    echo "  cp .env.example .env"
    echo "  # Edit .env with your MT5_LOGIN, MT5_PASSWORD, MT5_SERVER"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if config files exist
if [ ! -f "config/config.json" ]; then
    echo "ERROR: config/config.json not found!"
    exit 1
fi

# Start the trading bot
echo "Starting trading bot..."
echo "Press Ctrl+C to stop"
echo "---------------------"

# Default to XAUUSDm if no symbol provided
SYMBOL=${1:-XAUUSDm}
STRATEGIES=${2:-LiquiditySweepBreakout,RangeBounce}

echo "Symbol: $SYMBOL"
echo "Strategies: $STRATEGIES"
echo "---------------------"

python main.py --symbol "$SYMBOL" --strategies "$STRATEGIES"

# Deactivate virtual environment if we activated it
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi