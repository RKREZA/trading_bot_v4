"""
TRADING BOT V5 - Main Entry Point
Hybrid Breakout Strategy with Proper Lot Sizing
"""

import argparse
import json
import sys
import time
from datetime import datetime
from typing import List, Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 not installed. Run: pip install MetaTrader5")
    sys.exit(1)

import numpy as np

from dashboard import Dashboard, BacktestDashboard, AnalysisLogger
from core.strategy_engine import StrategyEngine, TradeSignal


class TradingBot:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.analysis_logger = AnalysisLogger(max_entries=100)
        self.strategy = StrategyEngine(self.config, self.analysis_logger)
        self.dashboard = Dashboard(self.config, self.analysis_logger)
        self.connected = False
        self.account_info = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.win_count = 0
        self.loss_count = 0
        self.running = False

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config {config_path} not found, using defaults")
            return self._default_config()

    def _default_config(self) -> dict:
        return {
            "mt5": {"login": 413559204, "password": "Insia@483311", "server": "Exness-MT5Trial6"},
            "symbol": "BTCUSDm",
            "risk_per_trade": 2.0,
            "max_daily_trades": 5,
            "daily_goal": 100.0,
            "strategy": {"min_confluence_score": 4, "min_confidence": 50, "cooldown_candles": 3},
            "backtest": {"initial_balance": 1000, "spread_pips": {"XAUUSD": 30, "GBPUSD": 2, "BTCUSDm": 50}},
            "symbols_config": {
                "XAUUSD": {"point": 0.01, "contract_size": 100, "lot": 0.1},
                "GBPUSD": {"point": 0.00001, "contract_size": 100000, "lot": 0.05},
                "BTCUSDm": {"point": 0.01, "contract_size": 1, "lot": 0.01}
            }
        }

    def _init_mt5(self, max_retries: int = 5) -> bool:
        mt5_config = self.config.get('mt5', {})
        print("\n" + "=" * 50)
        print("MT5 CONNECTION")
        print("=" * 50)
        print(f"Server: {mt5_config.get('server', 'N/A')}")
        print(f"Login: {mt5_config.get('login', 'N/A')}")

        for attempt in range(max_retries):
            try:
                mt5.shutdown()
                time.sleep(2)
                print(f"\nAttempt {attempt + 1}/{max_retries}...")

                if not mt5.initialize(
                    login=mt5_config.get('login'),
                    password=mt5_config.get('password'),
                    server=mt5_config.get('server'),
                    timeout=30000,
                    portable=False
                ):
                    error = mt5.last_error()
                    print(f"  X Failed: {error}")
                    if error[0] == -10005:
                        print("  -> MT5 terminal is not running. Open MT5 first!")
                    elif error[0] == -10011 or "Invalid" in str(error):
                        print("  -> Invalid credentials. Check login/password/server!")
                    time.sleep(3)
                    continue

                info = mt5.account_info()
                if info is None:
                    print(f"  X Account info failed: {mt5.last_error()}")
                    mt5.shutdown()
                    time.sleep(3)
                    continue

                self.connected = True
                self.account_info = {
                    'login': info.login, 'server': info.server, 'balance': info.balance,
                    'equity': info.equity, 'profit': info.profit, 'margin': info.margin,
                    'free_margin': info.margin_free, 'margin_level': info.margin_level if info.margin > 0 else 0,
                    'positions': 0, 'connected': True,
                    'server_time': datetime.now().strftime("%H:%M:%S")
                }
                print(f"  + Connected!")
                print(f"  + Balance: ${info.balance:,.2f}")
                return True

            except Exception as e:
                print(f"  X Exception: {e}")
                time.sleep(3)

        print("\n" + "=" * 50)
        print("CONNECTION FAILED")
        print("=" * 50)
        print("\nTROUBLESHOOTING:")
        print("1. Make sure MT5 terminal is OPEN and logged in")
        print("2. Check if Algo Trading is enabled in MT5")
        print("3. Run: python test_connection.py")
        return False

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        if not self.connected:
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {'point': info.point, 'digits': info.digits, 'contract_size': info.trade_contract_size,
                'min_lot': info.volume_min, 'max_lot': info.volume_max, 'spread': info.spread, 
                'bid': info.bid, 'ask': info.ask}

    def fetch_candles(self, symbol: str, timeframe: str, count: int = 500) -> List[dict]:
        tf_map = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
                  'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1}
        if timeframe not in tf_map or not self.connected:
            return []
        rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, count)
        if rates is None or len(rates) == 0:
            return []
        candles = []
        dtype_names = rates.dtype.names if hasattr(rates, 'dtype') else None
        for r in rates:
            if dtype_names:
                candles.append({"time": int(r['time']), "open": float(r['open']), "high": float(r['high']),
                               "low": float(r['low']), "close": float(r['close']), "tick_volume": int(r['tick_volume'])})
            else:
                candles.append(dict(r))
        return candles

    def _get_session(self) -> str:
        hour = datetime.now().hour
        if 8 <= hour < 17:
            return "LONDON"
        elif 13 <= hour < 22:
            return "NEW YORK"
        elif 0 <= hour < 9:
            return "TOKYO"
        return "CLOSED"

    def _update_dashboard_state(self, signal=None, analysis=None):
        if signal:
            self.dashboard.signal = {
                'direction': signal.direction, 'entry_price': signal.entry_price, 'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit, 'confidence': signal.confidence, 'confluence_score': signal.confluence_score,
                'reasons': signal.reasons, 'rejection_type': signal.rejection_type,
            }
        else:
            self.dashboard.signal = None
        if analysis:
            self.dashboard.h4_trend = analysis.get('h4_trend', 'RANGING')
            self.dashboard.m30_structure = analysis.get('m30_structure', 'NEUTRAL')
        self.dashboard.session = self._get_session()
        self.dashboard.daily_pnl = self.daily_pnl
        self.dashboard.daily_trades = self.daily_trades
        self.dashboard.win_count = self.win_count
        self.dashboard.loss_count = self.loss_count

    def run_live(self):
        if not self._init_mt5():
            return
        symbol = self.config.get('symbol', 'BTCUSDm')
        self.dashboard.selected_symbol = symbol
        self.analysis_logger.log(f"Starting live trading for {symbol}")
        self.running = True
        self.dashboard.running = True
        self.dashboard.start()
        try:
            while self.running:
                start_time = time.time()
                symbol_info = self.get_symbol_info(symbol)
                if symbol_info:
                    mid_price = (symbol_info['bid'] + symbol_info['ask']) / 2
                    self.dashboard.tick = {'bid': symbol_info['bid'], 'ask': symbol_info['ask'], 'price': mid_price,
                                           'spread': symbol_info['spread'] * symbol_info['point'], 'contract_size': symbol_info['contract_size']}
                    self.dashboard.account_info = self.account_info
                    h4_candles = self.fetch_candles(symbol, 'H4', 250)
                    m30_candles = self.fetch_candles(symbol, 'M30', 1540)
                    m15_candles = self.fetch_candles(symbol, 'M15', 2000)
                    if h4_candles and m30_candles and m15_candles:
                        current_price = self.dashboard.tick.get('price', 0)
                        signal = self.strategy.analyze(symbol, h4_candles, m30_candles, m15_candles, current_price)
                        h4_trend = self.strategy._determine_trend(h4_candles)
                        analysis_state = {'h4_trend': h4_trend, 'm30_structure': 'BULLISH' if h4_trend == 'BULLISH' else ('BEARISH' if h4_trend == 'BEARISH' else 'NEUTRAL')}
                        self._update_dashboard_state(signal, analysis_state)
                cycle_ms = (time.time() - start_time) * 1000
                self.dashboard.update(cycle_ms)
                time.sleep(1)
        except KeyboardInterrupt:
            self.analysis_logger.log("Stopping...")
            self.running = False
        finally:
            self.dashboard.stop()
            mt5.shutdown()

    def run_backtest(self, symbol: str):
        if not self._init_mt5():
            return
        print(f"\nFetching data for {symbol}...")
        h4_candles = self.fetch_candles(symbol, 'H4', 250)
        m30_candles = self.fetch_candles(symbol, 'M30', 1540)
        m15_candles = self.fetch_candles(symbol, 'M15', 2000)
        if not h4_candles or not m30_candles or not m15_candles:
            print("Failed to fetch data")
            mt5.shutdown()
            return
        print(f"+ H4: {len(h4_candles)} candles")
        print(f"+ M30: {len(m30_candles)} candles")
        print(f"+ M15: {len(m15_candles)} candles")
        mt5.shutdown()
        print("+ MT5 connection closed")

        # Get symbol-specific config
        symbol_config = self.config.get('symbols_config', {}).get(symbol, {})
        point = symbol_config.get('point', 0.01)
        spread_pips = self.config.get('backtest', {}).get('spread_pips', {}).get(symbol, 50)
        spread = spread_pips * point
        contract_size = symbol_config.get('contract_size', 1)
        
        # Symbol-specific lot size
        if symbol == 'XAUUSD':
            lot = 0.1  # Standard for gold
        elif symbol == 'GBPUSD':
            lot = 0.05  # Standard for forex
        else:  # BTCUSDm
            lot = 0.01  # For crypto
        
        balance = self.config.get('backtest', {}).get('initial_balance', 1000)
        initial_balance = balance
        
        print(f"\nSymbol Config:")
        print(f"  Point: {point}")
        print(f"  Spread: {spread_pips} pips = {spread}")
        print(f"  Contract Size: {contract_size}")
        print(f"  Lot Size: {lot}")
        
        trades = []
        signal_count = 0
        last_trade_idx = -999
        cooldown = self.config.get('strategy', {}).get('cooldown_candles', 3)
        bt_dashboard = BacktestDashboard(self.config)
        total_candles = len(m30_candles) - 110
        print(f"\nRunning backtest on {total_candles} candles...")

        for i in range(100, len(m30_candles) - 10):
            current_time = datetime.fromtimestamp(m30_candles[i]['time']).strftime("%Y-%m-%d %H:%M")
            bt_dashboard.show_progress(i - 100, total_candles, current_time, signal_count, len(trades))
            if i - last_trade_idx < cooldown:
                continue
            current_price = m30_candles[i]['close']
            h4_data = [c for c in h4_candles if c['time'] < m30_candles[i]['time']]
            m30_data = m30_candles[:i+1]
            m15_data = [c for c in m15_candles if c['time'] <= m30_candles[i]['time']]
            if len(h4_data) < 50 or len(m30_data) < 100 or len(m15_data) < 100:
                continue
            signal = self.strategy.analyze(symbol, h4_data, m30_data, m15_data, current_price)
            if signal:
                signal_count += 1
                if signal.confidence >= self.strategy.min_confidence:
                    last_trade_idx = i
                    
                    # Apply spread
                    if signal.direction == 'BUY':
                        entry = signal.entry_price + spread
                    else:
                        entry = signal.entry_price - spread
                    
                    # Simulate trade
                    future_candles = m30_candles[i+1:i+50]
                    outcome = self._simulate_trade(signal, future_candles, entry)
                    
                    # Calculate P/L with proper pip value
                    if outcome == 'WIN':
                        # TP hit
                        pip_profit = abs(signal.take_profit - entry) / point
                        pnl = pip_profit * point * contract_size * lot
                        balance += pnl
                        result = 'TP'
                    elif outcome == 'LOSS':
                        # SL hit
                        pip_loss = abs(signal.stop_loss - entry) / point
                        pnl = -pip_loss * point * contract_size * lot
                        balance += pnl
                        result = 'SL'
                    else:
                        pnl = 0
                        result = 'OPEN'
                    
                    trades.append({
                        'time': datetime.fromtimestamp(m30_candles[i]['time']).strftime("%Y-%m-%d %H:%M"),
                        'direction': signal.direction, 
                        'entry': entry, 
                        'sl': signal.stop_loss,
                        'tp': signal.take_profit, 
                        'result': result, 
                        'pnl': pnl,
                        'rr': signal.rr_ratio
                    })

        wins = [t for t in trades if t['result'] == 'TP']
        losses = [t for t in trades if t['result'] == 'SL']
        total_profit = sum(t['pnl'] for t in wins)
        total_loss = abs(sum(t['pnl'] for t in losses))
        
        results = {
            'symbol': symbol,
            'start_date': datetime.fromtimestamp(m30_candles[100]['time']).strftime("%Y-%m-%d"),
            'end_date': datetime.fromtimestamp(m30_candles[-10]['time']).strftime("%Y-%m-%d"),
            'initial_balance': initial_balance, 
            'final_balance': balance,
            'return_pct': (balance - initial_balance) / initial_balance * 100,
            'total_trades': len(trades), 
            'winning_trades': len(wins), 
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'profit_factor': total_profit / total_loss if total_loss > 0 else 0,
            'total_profit': total_profit, 
            'total_loss': total_loss,
            'avg_win': total_profit / len(wins) if wins else 0,
            'avg_loss': total_loss / len(losses) if losses else 0,
            'rr_ratio': (total_profit / len(wins)) / (total_loss / len(losses)) if wins and losses else 0,
            'max_drawdown': self._calc_dd(trades, initial_balance),
            'max_win_streak': self._calc_streak(trades, 'TP'),
            'max_loss_streak': self._calc_streak(trades, 'SL'),
            'trades': trades,
        }
        bt_dashboard.show_results(results)

    def _simulate_trade(self, signal, future_candles, entry) -> str:
        if not future_candles:
            return 'OPEN'
        for candle in future_candles:
            if signal.direction == 'BUY':
                if candle['high'] >= signal.take_profit:
                    return 'WIN'
                if candle['low'] <= signal.stop_loss:
                    return 'LOSS'
            else:
                if candle['low'] <= signal.take_profit:
                    return 'WIN'
                if candle['high'] >= signal.stop_loss:
                    return 'LOSS'
        return 'OPEN'

    def _calc_dd(self, trades, initial):
        if not trades:
            return 0
        balance, peak, max_dd = initial, initial, 0
        for t in trades:
            balance += t['pnl']
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _calc_streak(self, trades, result_type):
        if not trades:
            return 0
        max_s, cur = 0, 0
        for t in trades:
            if t['result'] == result_type:
                cur += 1
                max_s = max(max_s, cur)
            else:
                cur = 0
        return max_s


def main():
    parser = argparse.ArgumentParser(description="Trading Bot V5")
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--symbol', type=str, default='BTCUSDm', help='Symbol')
    parser.add_argument('--config', type=str, default='config.json', help='Config')
    args = parser.parse_args()
    bot = TradingBot(args.config)
    bot.config['symbol'] = args.symbol
    if args.backtest:
        bot.run_backtest(args.symbol)
    else:
        bot.run_live()


if __name__ == "__main__":
    main()
