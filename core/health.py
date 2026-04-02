from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import json
import time
import logging

logger = logging.getLogger("trading_bot.health")

class HealthHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler for the bot's health monitoring endpoint.
    Returns current bot status, connectivity, and session performance in JSON format.
    """
    bot = None
    
    def do_GET(self):
        """
        Handles GET requests to the health endpoint.
        Aggregates live bot metrics into a JSON response.
        """
        try:
            status = {
                "alive": True,
                "connected": self.bot.connection.connected if self.bot and hasattr(self.bot, 'connection') else False,
                "daily_trades": getattr(self.bot, 'daily_trades', 0),
                "daily_pnl": getattr(self.bot, 'daily_pnl', 0.0),
                "open_positions": len(getattr(self.bot, 'position_meta', {})),
                "uptime_seconds": time.time() - getattr(self.bot, '_start_time', time.time()),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def log_message(self, *args):
        pass  # Suppress access logs to keep terminal clean

def start_health_server(bot, port=8081):
    """
    Starts a lightweight HTTP server in a background daemon thread 
    to provide an external health-check endpoint.
    
    Args:
        bot (TradingBot): The running bot instance to monitor.
        port (int): Port to bind the server to.
    """
    HealthHandler.bot = bot
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info("Health endpoint live at: http://localhost:%d", port)
