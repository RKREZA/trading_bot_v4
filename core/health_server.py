"""
Production Health & Metrics Server
Provides health checks and Prometheus-compatible metrics endpoint.
"""
import logging
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("trading_bot.health")

class HealthServer:
    """
    Lightweight health and metrics server for production monitoring.
    Exposes /health and /metrics endpoints.
    """
    
    def __init__(self, port: int = 8080, bot_ref=None):
        self.port = port
        self.bot_ref = bot_ref
        self.start_time = time.time()
        self._server = None
        self._thread = None
        
        # Metrics
        self.metrics = {
            "cycles_completed": 0,
            "signals_generated": 0,
            "trades_executed": 0,
            "trades_closed": 0,
            "errors": 0,
            "last_cycle_time": None,
            "uptime_seconds": 0
        }
        self._metrics_lock = threading.Lock()

    def start(self):
        """Starts the health server in a background thread."""
        if self._server is not None:
            logger.warning("Health server already running.")
            return
            
        self._server = HTTPServer(('0.0.0.0', self.port), _HealthHandler)
        self._server.bot = self.bot_ref
        self._server.metrics_server = self
        
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Health server started on port {self.port}")

    def _run(self):
        self._server.serve_forever()

    def stop(self):
        """Stops the health server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            logger.info("Health server stopped.")

    def record_cycle(self):
        """Records a completed trading cycle."""
        with self._metrics_lock:
            self.metrics["cycles_completed"] += 1
            self.metrics["last_cycle_time"] = datetime.now().isoformat()

    def record_signal(self):
        """Records a signal generation."""
        with self._metrics_lock:
            self.metrics["signals_generated"] += 1

    def record_trade(self, direction: str):
        """Records a trade execution."""
        with self._metrics_lock:
            self.metrics["trades_executed"] += 1
            self.metrics["direction"] = direction

    def record_error(self):
        """Records an error."""
        with self._metrics_lock:
            self.metrics["errors"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Returns current metrics snapshot."""
        with self._metrics_lock:
            return {
                **self.metrics,
                "uptime_seconds": int(time.time() - self.start_time)
            }


class _HealthHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health and metrics endpoints."""
    
    def log_message(self, format, *args):
        pass  # Suppress default logging
    
    def do_GET(self):
        if self.path == '/health':
            self._handle_health()
        elif self.path == '/metrics':
            self._handle_metrics()
        elif self.path == '/ready':
            self._handle_ready()
        else:
            self.send_error(404)
    
    def _handle_health(self):
        """Liveness probe - is the process alive?"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "alive"}')
    
    def _handle_ready(self):
        """Readiness probe - is the bot ready to trade?"""
        server = self.server
        if hasattr(server, 'bot') and server.bot:
            connected = getattr(server.bot.connection, 'connected', False)
            if connected:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "ready"}')
                return
        
        self.send_response(503)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "not_ready"}')
    
    def _handle_metrics(self):
        """Prometheus-compatible metrics endpoint."""
        server = self.server
        metrics = {"status": "unknown"}
        
        if hasattr(server, 'metrics_server') and server.metrics_server:
            metrics = server.metrics_server.get_metrics()
        
        # Format as Prometheus text
        output = []
        output.append("# HELP trading_bot_uptime_seconds Bot uptime in seconds")
        output.append("# TYPE trading_bot_uptime_seconds gauge")
        output.append(f'trading_bot_uptime_seconds {metrics.get("uptime_seconds", 0)}')
        
        output.append("# HELP trading_bot_cycles_total Total trading cycles completed")
        output.append("# TYPE trading_bot_cycles_total counter")
        output.append(f'trading_bot_cycles_total {metrics.get("cycles_completed", 0)}')
        
        output.append("# HELP trading_bot_signals_total Total signals generated")
        output.append("# TYPE trading_bot_signals_total counter")
        output.append(f'trading_bot_signals_total {metrics.get("signals_generated", 0)}')
        
        output.append("# HELP trading_bot_trades_total Total trades executed")
        output.append("# TYPE trading_bot_trades_total counter")
        output.append(f'trading_bot_trades_total {metrics.get("trades_executed", 0)}')
        
        output.append("# HELP trading_bot_errors_total Total errors encountered")
        output.append("# TYPE trading_bot_errors_total counter")
        output.append(f'trading_bot_errors_total {metrics.get("errors", 0)}')
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write('\n'.join(output).encode())


if __name__ == "__main__":
    # Standalone test
    logging.basicConfig(level=logging.INFO)
    server = HealthServer(port=8080)
    server.start()
    
    server.record_cycle()
    server.record_signal()
    server.record_trade("BUY")
    
    print("Health server running. Try:")
    print("  curl http://localhost:8080/health")
    print("  curl http://localhost:8080/metrics")
    print("  curl http://localhost:8080/ready")
    
    try:
        time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()
