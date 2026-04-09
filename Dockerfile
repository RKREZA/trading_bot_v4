FROM python:3.11-slim

LABEL maintainer="Trading Bot V4"
LABEL description="Institutional Trading Bot - MT5 Integration"

WORKDIR /app

# Install system dependencies for MT5
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/logs /app/data_cache /app/state /app/config

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import MetaTrader5 as mt5; mt5.initialize(); mt5.shutdown()" || exit 1

# Run the bot
CMD ["python", "main.py", "--symbol", "XAUUSDm", "--strategies", "TrendFollowing,LiquiditySweepBreakout,SmartMeanReversion"]
