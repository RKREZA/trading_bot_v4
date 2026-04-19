#!/bin/bash
# Backtesting Script
# ================
# This script runs the backtesting suite with various options

echo "Running V5-INSIGNIA Backtesting Suite"
echo "====================================="

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

# Default parameters
SYMBOL="XAUUSDm"
START_DATE="2024-01-01"
END_DATE="2026-04-09"
RUN_MONTE_CARLO=false
RUN_WALK_FORWARD=false
RUN_STRESS_TEST=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --from)
            START_DATE="$2"
            shift 2
            ;;
        --to)
            END_DATE="$2"
            shift 2
            ;;
        --monte-carlo)
            RUN_MONTE_CARLO=true
            shift
            ;;
        --walk-forward)
            RUN_WALK_FORWARD=true
            shift
            ;;
        --stress-test)
            RUN_STRESS_TEST=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --symbol SYMBOL       Trading symbol (default: XAUUSDm)"
            echo "  --from DATE           Start date (default: 2024-01-01)"
            echo "  --to DATE             End date (default: 2026-04-09)"
            echo "  --monte-carlo         Run Monte Carlo simulation"
            echo "  --walk-forward        Run Walk-Forward optimization"
            echo "  --stress-test         Run Stress Testing suite"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="python backtest.py --symbol $SYMBOL --from $START_DATE --to $END_DATE"
if [ "$RUN_MONTE_CARLO" = true ]; then
    CMD="$CMD --monte-carlo"
fi
if [ "$RUN_WALK_FORWARD" = true ]; then
    CMD="$CMD --walk-forward"
fi
if [ "$RUN_STRESS_TEST" = true ]; then
    CMD="$CMD --stress-test"
fi

echo "Running: $CMD"
echo "---------------------"

# Run the backtest
eval $CMD

# Deactivate virtual environment if we activated it
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi