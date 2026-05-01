import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, Layers, ShieldAlert, History, Activity, 
  Play, Square, Zap, Wallet, TrendingUp, ArrowUpRight, 
  ArrowDownRight, BarChart3, AlertTriangle, Terminal,
  Settings, User, ChevronRight, Search
} from 'lucide-react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import './App.css';

const MOCK_EQUITY = [
  { time: '09:00', equity: 100000 },
  { time: '10:00', equity: 100250 },
  { time: '11:00', equity: 100150 },
  { time: '12:00', equity: 100600 },
  { time: '13:00', equity: 100400 },
  { time: '14:00', equity: 101200 },
  { time: '15:00', equity: 101000 },
];

function App() {
  const [activeTab, setActiveTab] = useState('Dashboard');
  const [status, setStatus] = useState({
    is_running: false,
    mt5_connected: false,
    server_time: new Date().toISOString(),
    drift_sec: 0,
    active_strategies: [],
    account: { balance: 100000, equity: 100000, profit: 0, drawdown: 0 }
  });
  const [config, setConfig] = useState(null);
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/status');
        const data = await res.json();
        setStatus(data);
      } catch (err) {}
    };

    const fetchConfig = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/config');
        const data = await res.json();
        setConfig(data);
      } catch (err) {}
    };

    fetchStatus();
    fetchConfig();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    let ws;
    let reconnectTimer;

    const connect = () => {
      ws = new WebSocket('ws://127.0.0.1:8000/ws/events');
      
      ws.onopen = () => {
        setLogs(prev => [{
          timestamp: new Date().toISOString(),
          type: 'SYSTEM',
          message: 'Real-time telemetry link established'
        }, ...prev]);
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'METRICS') {
          setStatus(prev => ({ ...prev, account: msg.data }));
        }
        setLogs(prev => [{
          timestamp: new Date().toISOString(),
          type: msg.type,
          message: msg.type === 'METRICS' ? 'Telemetry heartbeat received' : 'System event'
        }, ...prev.slice(0, 49)]);
      };

      ws.onclose = () => {
        reconnectTimer = setTimeout(connect, 3000);
      };
    };

    connect();
    return () => {
      ws?.close();
      clearTimeout(reconnectTimer);
    };
  }, []);

  const handleControl = async (action) => {
    try {
      await fetch(`http://127.0.0.1:8000/control/${action}`, { method: 'POST' });
      setLogs(prev => [{
        timestamp: new Date().toISOString(),
        type: 'ACTION',
        message: `Manual command: ${action.toUpperCase()} executed`
      }, ...prev]);
    } catch (err) {
      console.error(err);
    }
  };

  const [editingConfig, setEditingConfig] = useState(null);

  useEffect(() => {
    if (config && !editingConfig) {
      setEditingConfig(JSON.parse(JSON.stringify(config)));
    }
  }, [config]);

  const updateParam = (stratIndex, key, value) => {
    const newConfig = { ...editingConfig };
    if (key in newConfig.strategies[stratIndex]) {
      newConfig.strategies[stratIndex][key] = value;
    } else {
      newConfig.strategies[stratIndex].parameters[key] = value;
    }
    setEditingConfig(newConfig);
  };

  const saveConfig = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/config/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editingConfig)
      });
      if (res.ok) {
        setLogs(prev => [{
          timestamp: new Date().toISOString(),
          type: 'SYSTEM',
          message: 'Cluster configuration synchronized successfully'
        }, ...prev]);
        // Refresh global config
        const data = await (await fetch('http://127.0.0.1:8000/config')).json();
        setConfig(data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const updateRiskParam = (key, value) => {
    const newConfig = { ...editingConfig };
    newConfig.risk[key] = value;
    setEditingConfig(newConfig);
  };

  const renderDashboard = () => (
    <div className="tab-content fade-in">
      <div className="metrics-grid">
        <div className="metric-card glass-panel">
          <div className="metric-header">
            <Wallet size={18} className="icon-blue" />
            <span className="metric-label">Net Equity</span>
          </div>
          <span className="metric-value">${status.account.equity.toLocaleString(undefined, {minimumFractionDigits: 2})}</span>
          <div className={`metric-delta ${status.account.profit >= 0 ? 'positive' : 'negative'}`}>
            {status.account.profit >= 0 ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
            <span style={{fontWeight: 700}}>${Math.abs(status.account.profit).toFixed(2)}</span>
            <span style={{color: 'var(--text-muted)', fontSize: '11px', marginLeft: '4px'}}>Current Session</span>
          </div>
        </div>
        <div className="metric-card glass-panel">
          <div className="metric-header">
            <TrendingUp size={18} className="accent-orange" />
            <span className="metric-label">Risk Exposure</span>
          </div>
          <span className={`metric-value ${status.account.drawdown > 0.05 ? 'negative' : ''}`}>
            {(status.account.drawdown * 100).toFixed(2)}%
          </span>
          <div className="progress-bar">
            <div className="progress-fill" style={{ 
              width: `${Math.min(status.account.drawdown * 1000, 100)}%`, 
              background: status.account.drawdown > 0.05 ? 'var(--accent-red)' : 'linear-gradient(90deg, var(--accent-blue), var(--accent-cyan))',
              boxShadow: status.account.drawdown > 0.05 ? '0 0 10px var(--accent-red)' : '0 0 10px var(--accent-blue)'
            }}></div>
          </div>
        </div>
        <div className="metric-card glass-panel">
          <div className="metric-header">
            <Zap size={18} className="accent-yellow" />
            <span className="metric-label">Engine Clusters</span>
          </div>
          <span className="metric-value">{status.active_strategies.length || 0}</span>
          <div className="strat-list-mini">
            {status.active_strategies.map(s => <span key={s} className="strat-tag">{s}</span>)}
            {status.active_strategies.length === 0 && <span className="strat-tag" style={{opacity: 0.5}}>None Active</span>}
          </div>
        </div>
      </div>

      <div className="chart-panel glass-panel">
        <div className="panel-header">
          <div style={{display: 'flex', alignItems: 'center', gap: '12px'}}>
            <BarChart3 size={20} className="icon-blue" />
            <h2 style={{fontSize: '18px', fontWeight: 700}}>Institutional Equity Curve</h2>
          </div>
          <div className="chart-toggles">
            <button className="small-btn active">1D</button>
            <button className="small-btn">1W</button>
            <button className="small-btn">MTD</button>
          </div>
        </div>
        <div className="chart-container" style={{height: '350px'}}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={MOCK_EQUITY} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent-blue)" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="var(--accent-blue)" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
              <XAxis dataKey="time" stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} dy={10} />
              <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} axisLine={false} domain={['auto', 'auto']} tickFormatter={(v) => `$${v}`} dx={-10} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'var(--bg-panel-inner)', border: '1px solid var(--glass-border)', borderRadius: '12px', backdropFilter: 'blur(10px)' }}
                itemStyle={{ color: 'var(--accent-blue)', fontWeight: 700 }}
                labelStyle={{ color: 'var(--text-muted)', marginBottom: '4px' }}
              />
              <Area type="monotone" dataKey="equity" stroke="var(--accent-blue)" strokeWidth={3} fillOpacity={1} fill="url(#colorEquity)" animationDuration={1500} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );

  const renderStrategies = () => (
    <div className="tab-content fade-in">
      <div className="panel-header">
        <h2 style={{fontSize: '24px', fontWeight: 800}}>Cluster Management</h2>
        <button className="btn-action start" onClick={saveConfig}><ShieldAlert size={14}/> Sync to Terminal</button>
      </div>
      <div className="strategies-grid">
        {editingConfig?.strategies.map((strat, idx) => (
          <div key={strat.name} className="strategy-card glass-panel">
            <div className="strat-card-header">
              <div className="strat-info">
                <div className="logo-box"><Zap size={18} /></div>
                <div>
                  <h3>{strat.name}</h3>
                  <span className="symbol-tag">{strat.symbol}</span>
                </div>
              </div>
              <label className="switch">
                <input 
                  type="checkbox" 
                  checked={strat.enabled} 
                  onChange={(e) => updateParam(idx, 'enabled', e.target.checked)} 
                />
                <span className="slider round"></span>
              </label>
            </div>
            
            <div className="strat-params">
              <div className="param-field">
                <label>Timeframe (M)</label>
                <input 
                  type="number" 
                  value={strat.timeframe} 
                  onChange={(e) => updateParam(idx, 'timeframe', parseInt(e.target.value))} 
                />
              </div>
              {Object.keys(strat.parameters).map(key => (
                <div key={key} className="param-field">
                  <label>{key.replace('_', ' ')}</label>
                  <input 
                    type={typeof strat.parameters[key] === 'number' ? 'number' : 'text'} 
                    value={strat.parameters[key]} 
                    onChange={(e) => updateParam(idx, key, typeof strat.parameters[key] === 'number' ? parseFloat(e.target.value) : e.target.value)} 
                  />
                </div>
              ))}
            </div>

            <div className="strat-footer">
              <div className="strat-performance">
                <div className="p-stat">
                  <span>Win Rate</span>
                  <strong>--%</strong>
                </div>
                <div className="p-stat">
                  <span>Current P/L</span>
                  <strong className="positive">+$0.00</strong>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderRisk = () => (
    <div className="tab-content fade-in risk-container">
      <div className="glass-panel risk-config-form">
        <div className="panel-header">
          <h2 style={{fontSize: '24px', fontWeight: 800}}>Institutional Risk Profile</h2>
          <button className="btn-action start" onClick={saveConfig}><ShieldAlert size={14}/> Sync Risk Config</button>
        </div>
        
        <div className="risk-form-grid">
          <div className="input-group">
            <div className="label-row">
              <label>Global Max Drawdown</label>
              <span className="value-display">{(editingConfig?.risk.max_drawdown * 100).toFixed(1)}%</span>
            </div>
            <input 
              type="range" min="1" max="30" step="0.5" 
              value={editingConfig?.risk.max_drawdown * 100} 
              onChange={(e) => updateRiskParam('max_drawdown', parseFloat(e.target.value) / 100)} 
            />
            <div className="range-labels"><span>1%</span><span>30%</span></div>
          </div>

          <div className="input-group">
            <div className="label-row">
              <label>Daily Loss Limit</label>
              <span className="value-display">{(editingConfig?.risk.daily_loss_limit * 100).toFixed(1)}%</span>
            </div>
            <input 
              type="range" min="0.5" max="10" step="0.1" 
              value={editingConfig?.risk.daily_loss_limit * 100} 
              onChange={(e) => updateRiskParam('daily_loss_limit', parseFloat(e.target.value) / 100)} 
            />
            <div className="range-labels"><span>0.5%</span><span>10%</span></div>
          </div>

          <div className="input-group">
            <div className="label-row">
              <label>Risk Per Trade</label>
              <span className="value-display">{(editingConfig?.risk.risk_per_trade * 100).toFixed(1)}%</span>
            </div>
            <input 
              type="range" min="0.1" max="5" step="0.1" 
              value={editingConfig?.risk.risk_per_trade * 100} 
              onChange={(e) => updateRiskParam('risk_per_trade', parseFloat(e.target.value) / 100)} 
            />
            <div className="range-labels"><span>0.1%</span><span>5%</span></div>
          </div>

          <div className="input-group">
            <label>Max Concurrent Trades</label>
            <input 
              type="number" 
              className="glass-panel-inner"
              value={editingConfig?.risk.max_concurrent_trades} 
              onChange={(e) => updateRiskParam('max_concurrent_trades', parseInt(e.target.value))} 
            />
          </div>
        </div>

        <div className="risk-safety-info">
          <div className="info-box glass-panel-inner">
            <ShieldAlert size={20} className="accent-blue" />
            <div>
              <h4>Account Protection Enabled</h4>
              <p>System will automatically close all positions if equity drops below <strong>${(status.account.balance * (1 - (editingConfig?.risk.max_drawdown || 0.1))).toLocaleString()}</strong>.</p>
            </div>
          </div>
        </div>
      </div>

      <div className="glass-panel kill-switch-panel">
        <AlertTriangle size={48} color="var(--accent-red)" style={{marginBottom: '24px'}} />
        <h2 style={{fontSize: '24px', fontWeight: 800, color: 'var(--accent-red)'}}>MANUAL INTERVENTION</h2>
        <p style={{color: 'var(--text-muted)', margin: '20px 0', fontSize: '14px', lineHeight: '1.6'}}>
          Activating the Global Kill-Switch will bypass all logic and immediately send Market Close orders for every open position across all clusters. 
          <br /><br />
          <strong>This action is irreversible.</strong>
        </p>
        <button className="btn-kill" onClick={() => handleControl('kill')}>
          ACTIVATE GLOBAL KILL-SWITCH
        </button>
      </div>
    </div>
  );

  const renderMonitor = () => (
    <div className="tab-content fade-in" style={{display: 'flex', flexDirection: 'column', height: '100%'}}>
      <div className="panel-header">
        <h2 style={{fontSize: '24px', fontWeight: 800}}>System Audit Trail</h2>
        <div className="search-bar glass-panel-inner">
          <Search size={16} color="var(--text-muted)" />
          <input type="text" placeholder="Filter logs..." />
        </div>
      </div>
      <div className="logs-container">
        {logs.map((log, i) => (
          <div key={i} className="log-entry">
            <span style={{color: 'var(--text-muted)'}}>{new Date(log.timestamp).toLocaleTimeString()}</span>
            <span className={`log-cat ${log.type}`}>{log.type}</span>
            <span style={{color: log.type === 'ACTION' ? 'var(--accent-yellow)' : 'var(--text-main)'}}>{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );

  const [backtestResults, setBacktestResults] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [selectedStrats, setSelectedStrats] = useState([]);

  const toggleBacktestStrat = (stratName) => {
    setSelectedStrats(prev => 
      prev.includes(stratName) ? prev.filter(s => s !== stratName) : [...prev, stratName]
    );
  };

  const runBacktest = async () => {
    setBacktestLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/backtest/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategies: selectedStrats.map(s => ({ name: s, symbol: 'XAUUSD' })),
          initial_balance: 100000,
          stress_multiplier: 1.5,
          monte_carlo_count: 100
        })
      });
      const data = await res.json();
      setBacktestResults(data);
    } catch (err) {
      console.error(err);
    }
    setBacktestLoading(false);
  };

  const renderBacktest = () => (
    <div className="tab-content fade-in backtest-layout">
      <div className="backtest-setup glass-panel">
        <h2 style={{fontSize: '22px', fontWeight: 800, marginBottom: '24px'}}>Simulation Setup</h2>
        
        <div className="setup-section">
          <label className="section-label">Cluster Selection</label>
          <div className="multi-select-grid">
            {config?.strategies.map(s => (
              <button 
                key={s.name} 
                className={`select-btn ${selectedStrats.includes(s.name) ? 'active' : ''}`}
                onClick={() => toggleBacktestStrat(s.name)}
              >
                <Zap size={14} />
                {s.name}
              </button>
            ))}
          </div>
        </div>

        <div className="setup-section">
          <label className="section-label">Risk Vectors</label>
          <div className="risk-vectors-grid">
            <div className="v-field">
              <label>Initial Balance</label>
              <input type="number" defaultValue="100000" className="glass-panel-inner" />
            </div>
            <div className="v-field">
              <label>Stress Multiplier</label>
              <input type="number" defaultValue="1.5" step="0.1" className="glass-panel-inner" />
            </div>
          </div>
        </div>

        <div className="setup-section">
          <label className="section-label">Institutional Eval</label>
          <div className="eval-toggles">
            <label className="check-row">
              <input type="checkbox" defaultChecked />
              <span>Monte Carlo (100 Iterations)</span>
            </label>
            <label className="check-row">
              <input type="checkbox" defaultChecked />
              <span>Slippage Stress (1.5x)</span>
            </label>
          </div>
        </div>

        <button 
          className="btn-kill" 
          style={{background: 'var(--accent-blue)', boxShadow: 'var(--glow-blue)', marginTop: '20px'}}
          onClick={runBacktest}
          disabled={backtestLoading || selectedStrats.length === 0}
        >
          {backtestLoading ? 'SIMULATING...' : 'EXECUTE DETERMINISTIC BACKTEST'}
        </button>
      </div>

      <div className="backtest-results-panel">
        {backtestResults ? (
          <div className="results-scroll">
            <div className="results-summary-grid">
              <div className="res-stat glass-panel">
                <span className="res-label">Net Profit</span>
                <span className={`res-val ${backtestResults.summary.net_profit >= 0 ? 'positive' : 'negative'}`}>
                  ${backtestResults.summary.net_profit.toLocaleString(undefined, {maximumFractionDigits: 0})}
                </span>
              </div>
              <div className="res-stat glass-panel">
                <span className="res-label">Max Drawdown</span>
                <span className="res-val" style={{color: 'var(--accent-orange)'}}>
                  {(backtestResults.summary.max_drawdown * 100).toFixed(2)}%
                </span>
              </div>
              <div className="res-stat glass-panel">
                <span className="res-label">MC Pass Score</span>
                <span className={`res-val ${backtestResults.monte_carlo.institutional_pass ? 'positive' : 'negative'}`}>
                  {backtestResults.monte_carlo.institutional_pass ? 'PASS' : 'FAIL'}
                </span>
              </div>
            </div>

            <div className="chart-panel glass-panel" style={{marginTop: '24px', height: '380px'}}>
               <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={backtestResults.equity_curve.map((e, i) => ({ i, e }))}>
                  <defs>
                    <linearGradient id="resEquity" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--accent-blue)" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="var(--accent-blue)" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
                  <Tooltip />
                  <Area type="monotone" dataKey="e" stroke="var(--accent-blue)" strokeWidth={2} fill="url(#resEquity)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="mc-details glass-panel" style={{marginTop: '24px', padding: '24px'}}>
              <h3 style={{fontSize: '16px', marginBottom: '16px'}}>Monte Carlo Risk Analysis</h3>
              <div className="mc-grid">
                <div className="mc-item">
                  <span className="mc-label">95% Confidence Max DD</span>
                  <span className="mc-val">{(backtestResults.monte_carlo.max_drawdown_95pc * 100).toFixed(2)}%</span>
                </div>
                <div className="mc-item">
                  <span className="mc-label">Avg. Randomized DD</span>
                  <span className="mc-val">{(backtestResults.monte_carlo.avg_drawdown * 100).toFixed(2)}%</span>
                </div>
              </div>
              <div className={`mc-badge ${backtestResults.monte_carlo.institutional_pass ? 'pass' : 'fail'}`}>
                {backtestResults.monte_carlo.institutional_pass 
                  ? '✓ INSTITUTIONAL GRADE QUALITY: PASSED' 
                  : '✗ HIGH RISK DETECTED: FAILED MONTE CARLO'}
              </div>
            </div>
          </div>
        ) : (
          <div className="results-placeholder glass-panel">
            <History size={48} className="icon-blue" style={{opacity: 0.3, marginBottom: '20px'}} />
            <p style={{color: 'var(--text-muted)'}}>
              Awaiting simulation parameters. Select clusters and execute to generate report.
            </p>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="app-container top-nav-layout">
      <nav className="top-nav glass-panel">
        <div className="nav-left">
          <div className="nav-logo">
            <div className="logo-box">
              <Zap size={20} color="var(--accent-blue)" />
            </div>
            <span className="logo-text">ANTIGRAVITY V5</span>
          </div>

          <div className="nav-links-horizontal">
            {['Dashboard', 'Strategies', 'Risk & Safety', 'Backtesting', 'Live Monitor'].map((tab) => (
              <button
                key={tab}
                className={`nav-item-h ${activeTab === tab ? 'active' : ''}`}
                onClick={() => setActiveTab(tab)}
              >
                <span>{tab}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="nav-right">
          <div className="connection-pill">
            <div className={`pulse-dot ${status.mt5_connected ? 'active' : ''}`}></div>
            <span>{status.mt5_connected ? 'MT5: LIVE' : 'DISCONNECTED'}</span>
          </div>
          
          <div className="control-btns">
            <button 
              className={`btn-toggle ${status.is_running ? 'running' : 'idle'}`}
              onClick={() => handleControl(status.is_running ? 'stop' : 'start')}
              title={status.is_running ? 'Stop Engine' : 'Start Engine'}
            >
              {status.is_running ? <Square size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
              <span>{status.is_running ? 'STOP' : 'START'}</span>
            </button>
          </div>

          <div className="user-profile">
            <User size={18} color="var(--text-muted)" />
            <span className="user-name">ADMIN</span>
          </div>
          
          <button className="icon-btn" title="Settings">
            <Settings size={18} color="var(--text-muted)" />
          </button>
        </div>
      </nav>

      <main className="main-viewport">
        <div className="content-scroller">
          {activeTab === 'Dashboard' && renderDashboard()}
          {activeTab === 'Strategies' && renderStrategies()}
          {activeTab === 'Risk & Safety' && renderRisk()}
          {activeTab === 'Backtesting' && renderBacktest()}
          {activeTab === 'Live Monitor' && renderMonitor()}
        </div>
      </main>

      <footer className="status-bar glass-panel">
        <div className="status-group">
          <span className="status-label">ENGINE STATUS:</span>
          <span className={`status-value ${status.is_running ? 'active' : ''}`}>
            {status.is_running ? 'RUNNING' : 'IDLE'}
          </span>
        </div>
        <div className="status-group">
          <span className="status-label">SERVER TIME:</span>
          <span className="time-value">{new Date(status.server_time).toLocaleTimeString()}</span>
          <span className="drift-badge">{status.drift_sec.toFixed(3)}s DRIFT</span>
        </div>
        <div className="status-group">
          <span className="status-label">ACTIVE CLUSTERS:</span>
          <span className="status-value">{status.active_strategies.length}</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
