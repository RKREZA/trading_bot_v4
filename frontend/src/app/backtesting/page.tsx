"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

type BacktestStatus = "idle" | "pending" | "running" | "completed" | "failed";
type ResultTab = "overview" | "equity" | "trades" | "analysis" | "montecarlo" | "stress";

const TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];
const SYMBOLS = ["XAUUSDm", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "GBPJPY", "XAGUSD"];

export default function BacktestingPage() {
  const [symbol, setSymbol] = useState("XAUUSDm");
  const [strategyId, setStrategyId] = useState("SMC");
  const [timeframe, setTimeframe] = useState("M5");
  const [balance, setBalance] = useState(10000);
  const [riskPct, setRiskPct] = useState(1.0);
  const [commission, setCommission] = useState(0.0);
  const [spreadPips, setSpreadPips] = useState(0.0);
  const [slippagePoints, setSlippagePoints] = useState(0.0);
  const [mcIterations, setMcIterations] = useState(500);
  const [stressTest, setStressTest] = useState(false);
  const [walkForward, setWalkForward] = useState(false);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [sessionFilter, setSessionFilter] = useState("");
  const [seed, setSeed] = useState(42);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [status, setStatus] = useState<BacktestStatus>("idle");
  const [runId, setRunId] = useState<number | null>(null);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState("");
  const [runs, setRuns] = useState<any[]>([]);
  const [progressPct, setProgressPct] = useState(0);
  const [progressStage, setProgressStage] = useState("");
  const [resultTab, setResultTab] = useState<ResultTab>("overview");
  const [availableStrategies, setAvailableStrategies] = useState<any[]>([]);
  const [brokerSymbols, setBrokerSymbols] = useState<string[]>([]);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    api.getBacktestRuns(10).then(setRuns).catch(() => {});
    api.getAvailableStrategies().then(setAvailableStrategies).catch(() => {});
    api.getSymbols().then((syms) => { if (syms && syms.length > 0) setBrokerSymbols(syms); }).catch(() => {});
  }, []);

  const pollStatus = useCallback(
    (id: number) => {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const res = await api.getBacktestStatus(id);
          setProgressPct(res.progress_pct || 0);
          setProgressStage(res.stage || "");
          if (res.status === "COMPLETED") {
            clearInterval(pollRef.current!);
            pollRef.current = null;
            setStatus("completed");
            setProgressPct(100);
            const full = await api.getBacktestResults(id);
            setResults(full.results);
            api.getBacktestRuns(10).then(setRuns).catch(() => {});
          } else if (res.status === "FAILED") {
            clearInterval(pollRef.current!);
            pollRef.current = null;
            setStatus("failed");
            setError(res.error_message || "Backtest failed");
          }
        } catch {}
      }, 2000);
    },
    []
  );

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleRun = async () => {
    setStatus("pending");
    setResults(null);
    setError("");
    setProgressPct(0);
    setProgressStage("Starting...");
    try {
      const res = await api.runBacktest({
        symbol,
        strategy_id: strategyId,
        timeframe,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        initial_balance: balance,
        risk_per_trade_pct: riskPct,
        commission_per_lot: commission,
        spread_pips: spreadPips,
        slippage_points: slippagePoints,
        monte_carlo_iterations: mcIterations,
        stress_test: stressTest,
        walk_forward: walkForward,
        session_filter: sessionFilter,
        random_seed: seed,
      });
      setRunId(res.run_id);
      setStatus("running");
      pollStatus(res.run_id);
    } catch (e: any) {
      setStatus("failed");
      setError(e.message || "Failed to start backtest");
    }
  };

  const loadPastResult = async (id: number) => {
    try {
      const full = await api.getBacktestResults(id);
      setRunId(id);
      setResults(full.results);
      setStatus(full.status === "COMPLETED" ? "completed" : "failed");
      setError(full.error_message || "");
    } catch {}
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Backtesting</h2>

      <div className="grid grid-cols-5 gap-4">
        {/* Config Panel */}
        <Card className="col-span-1">
          <CardHeader><CardTitle>Configuration</CardTitle></CardHeader>
          <div className="space-y-3 px-1">
            <Field label="Symbol">
              <select value={symbol} onChange={(e) => setSymbol(e.target.value)} className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm">
                {(brokerSymbols.length > 0 ? brokerSymbols : SYMBOLS).map((s) => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Strategy">
              {availableStrategies.length > 0 ? (
                <select value={strategyId} onChange={(e) => setStrategyId(e.target.value)} className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm">
                  {availableStrategies.map((s) => <option key={s.id} value={s.id}>{s.id} ({s.class})</option>)}
                </select>
              ) : (
                <input value={strategyId} onChange={(e) => setStrategyId(e.target.value)} className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm" />
              )}
            </Field>
            <Field label="Timeframe">
              <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)} className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm">
                {TIMEFRAMES.map((tf) => <option key={tf}>{tf}</option>)}
              </select>
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="From">
                <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="w-full rounded bg-zinc-800 px-2 py-1.5 text-xs" />
              </Field>
              <Field label="To">
                <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="w-full rounded bg-zinc-800 px-2 py-1.5 text-xs" />
              </Field>
            </div>

            <p className="text-[10px] font-semibold uppercase text-zinc-500 pt-2">Capital & Risk</p>
            <Field label="Initial Balance ($)">
              <input type="number" value={balance} onChange={(e) => setBalance(Number(e.target.value))} className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm" />
            </Field>
            <Field label="Risk Per Trade (%)">
              <input type="number" step="0.1" value={riskPct} onChange={(e) => setRiskPct(Number(e.target.value))} className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm" />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Commission/Lot">
                <input type="number" step="0.5" value={commission} onChange={(e) => setCommission(Number(e.target.value))} className="w-full rounded bg-zinc-800 px-2 py-1.5 text-xs" />
              </Field>
              <Field label="Spread (pips)">
                <input type="number" step="0.1" value={spreadPips} onChange={(e) => setSpreadPips(Number(e.target.value))} className="w-full rounded bg-zinc-800 px-2 py-1.5 text-xs" />
              </Field>
            </div>
            <Field label="Slippage (points)">
              <input type="number" step="0.1" value={slippagePoints} onChange={(e) => setSlippagePoints(Number(e.target.value))} className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm" />
            </Field>

            <p className="text-[10px] font-semibold uppercase text-zinc-500 pt-2">Analysis</p>
            <Field label="Monte Carlo Iterations">
              <input type="number" value={mcIterations} onChange={(e) => setMcIterations(Number(e.target.value))} min={100} max={5000} className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm" />
            </Field>
            <label className="flex items-center gap-2 text-xs text-zinc-400">
              <input type="checkbox" checked={stressTest} onChange={(e) => setStressTest(e.target.checked)} className="rounded" /> Stress Test
            </label>
            <label className="flex items-center gap-2 text-xs text-zinc-400">
              <input type="checkbox" checked={walkForward} onChange={(e) => setWalkForward(e.target.checked)} className="rounded" /> Walk-Forward Validation
            </label>

            <button onClick={() => setShowAdvanced(!showAdvanced)} className="text-[10px] text-brand-400 hover:text-brand-300">
              {showAdvanced ? "Hide Advanced" : "Show Advanced"}
            </button>
            {showAdvanced && (
              <div className="space-y-2 border-t border-zinc-800 pt-2">
                <Field label="Session Filter">
                  <select value={sessionFilter} onChange={(e) => setSessionFilter(e.target.value)} className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm">
                    <option value="">All Sessions</option>
                    <option>LONDON</option>
                    <option>NEW_YORK</option>
                    <option>ASIAN</option>
                    <option>OVERLAP</option>
                  </select>
                </Field>
                <Field label="Random Seed">
                  <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm" />
                </Field>
              </div>
            )}

            <button
              onClick={handleRun}
              disabled={status === "pending" || status === "running"}
              className="mt-2 w-full rounded bg-brand-600 px-3 py-2 text-sm font-medium hover:bg-brand-500 disabled:opacity-50"
            >
              {status === "running" ? "Running..." : status === "pending" ? "Starting..." : "Run Backtest"}
            </button>

            {(status === "running" || status === "pending") && (
              <div className="space-y-1">
                <div className="h-2 w-full rounded-full bg-zinc-800 overflow-hidden">
                  <div className="h-full bg-brand-500 transition-all duration-500" style={{ width: `${progressPct}%` }} />
                </div>
                <p className="text-[10px] text-zinc-500 text-center">{progressStage} — {progressPct}%</p>
              </div>
            )}

            {runs.length > 0 && (
              <div className="mt-3 border-t border-zinc-800 pt-3">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs font-medium text-zinc-400">Recent Runs</p>
                  <button onClick={async () => { await api.deleteAllBacktestRuns(); setRuns([]); setResults(null); setStatus("idle"); }}
                    className="text-[10px] text-red-500 hover:text-red-400">Clear All</button>
                </div>
                <div className="max-h-48 space-y-1 overflow-auto">
                  {runs.map((r) => (
                    <div key={r.id} className={`flex w-full items-center justify-between rounded px-2 py-1 text-xs hover:bg-zinc-800 ${runId === r.id ? "bg-zinc-800" : ""}`}>
                      <button onClick={() => loadPastResult(r.id)} className="flex items-center gap-1 text-zinc-400">
                        #{r.id} <span className="text-zinc-600">{r.config?.timeframe}</span>
                      </button>
                      <div className="flex items-center gap-1">
                        <Badge variant={r.status === "COMPLETED" ? "success" : r.status === "FAILED" ? "danger" : "default"}>{r.status}</Badge>
                        <button onClick={async (e) => { e.stopPropagation(); await api.deleteBacktestRun(r.id); setRuns(runs.filter(x => x.id !== r.id)); if (runId === r.id) { setResults(null); setStatus("idle"); } }}
                          className="text-zinc-600 hover:text-red-400 ml-1" title="Delete">x</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* Results Panel */}
        <div className="col-span-4 space-y-4">
          {status === "failed" && error && (
            <Card><div className="p-4 text-red-400">{error}</div></Card>
          )}

          {status === "running" && (
            <Card>
              <div className="flex h-32 items-center justify-center text-zinc-400">
                <div className="text-center">
                  <div className="mb-2 text-lg">Backtest Running...</div>
                  <div className="text-xs text-zinc-500">{progressStage || `Processing ${symbol} on ${timeframe}`}</div>
                </div>
              </div>
            </Card>
          )}

          {status === "completed" && results && (
            <>
              {/* Tab Bar */}
              <div className="flex gap-1 border-b border-zinc-800 pb-1">
                {(["overview", "equity", "trades", "analysis", "montecarlo", "stress"] as ResultTab[]).map((tab) => (
                  <button key={tab} onClick={() => setResultTab(tab)}
                    className={`rounded-t px-3 py-1.5 text-xs font-medium transition ${resultTab === tab ? "bg-zinc-800 text-white" : "text-zinc-500 hover:text-zinc-300"}`}>
                    {tab === "montecarlo" ? "Monte Carlo" : tab === "stress" ? "Stress Test" : tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
                {results && runId && (
                  <a href={api.exportBacktestCsv(runId)} className="ml-auto rounded px-3 py-1.5 text-xs text-zinc-500 hover:text-white border border-zinc-700 hover:border-zinc-500">
                    Export CSV
                  </a>
                )}
              </div>

              {/* Summary Bar */}
              <div className="flex items-center gap-4 text-xs text-zinc-400">
                <span>{results.symbol} / {results.timeframe}</span>
                <span>{results.total_trades} trades</span>
                <span>{results.total_bars} bars</span>
                <span>{results.duration_days}d range</span>
                <span className={results.metrics?.net_profit >= 0 ? "text-emerald-400" : "text-red-400"}>
                  PnL: ${results.metrics?.net_profit}
                </span>
              </div>

              {resultTab === "overview" && <MetricsGrid metrics={results.metrics} analytics={results.analytics} />}
              {resultTab === "equity" && (
                <div className="grid grid-cols-2 gap-4">
                  <EquityCurveCard data={results.equity_curve} />
                  <DrawdownCurveCard data={results.analytics?.drawdown_curve} />
                  <RollingSharpeCard data={results.analytics?.rolling_sharpe} />
                  <PnlDistributionCard data={results.analytics?.pnl_distribution} />
                </div>
              )}
              {resultTab === "trades" && <TradeTable trades={results.trades} />}
              {resultTab === "analysis" && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <MonthlyReturnsCard data={results.analytics?.monthly_returns} />
                    <SessionPerfCard data={results.analytics?.session_performance} />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <HourlyDistCard data={results.analytics?.hourly_distribution} />
                    <DayDistCard data={results.analytics?.day_distribution} />
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <StreakCard data={results.analytics?.streak_analysis} />
                    <DurationCard data={results.analytics?.duration_analysis} />
                    <AdvancedMetricsCard data={results.analytics?.advanced_metrics} />
                  </div>
                </div>
              )}
              {resultTab === "montecarlo" && <MonteCarloCard mc={results.monte_carlo} />}
              {resultTab === "stress" && results.stress_test && <StressTestCard data={results.stress_test} />}
              {resultTab === "stress" && results.walk_forward && <WalkForwardCard data={results.walk_forward} />}
            </>
          )}

          {status === "idle" && (
            <Card>
              <div className="flex h-64 items-center justify-center text-zinc-500">
                Configure parameters and run a backtest to see results.
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="mb-1 block text-xs text-zinc-400">{label}</label>
      {children}
    </div>
  );
}

function MetricsGrid({ metrics, analytics }: { metrics: any; analytics?: any }) {
  if (!metrics || metrics.status === "NO_TRADES") {
    return <Card><div className="p-4 text-zinc-500">No trades executed.</div></Card>;
  }

  const adv = analytics?.advanced_metrics || {};
  const items = [
    { label: "Net Profit", value: `$${metrics.net_profit}`, accent: metrics.net_profit >= 0 },
    { label: "Final Balance", value: `$${metrics.final_balance}`, accent: metrics.final_balance > metrics.initial_balance },
    { label: "Win Rate", value: metrics.win_rate },
    { label: "Total Trades", value: metrics.total_trades },
    { label: "Profit Factor", value: metrics.profit_factor },
    { label: "Max Drawdown", value: metrics.max_drawdown },
    { label: "Sharpe Ratio", value: metrics.sharpe_ratio },
    { label: "Sortino Ratio", value: metrics.sortino_ratio },
    { label: "SQN", value: metrics.sqn },
    { label: "Expectancy", value: `$${metrics.expectancy}` },
    { label: "R:R Ratio", value: metrics.rr_ratio },
    { label: "CAGR", value: metrics.cagr },
    { label: "Calmar Ratio", value: metrics.calmar_ratio },
    { label: "Recovery Factor", value: adv.recovery_factor ?? "-" },
    { label: "Payoff Ratio", value: adv.payoff_ratio ?? "-" },
    { label: "Kelly %", value: adv.kelly_criterion != null ? `${adv.kelly_criterion}%` : "-" },
    { label: "Best Trade", value: adv.best_trade != null ? `$${adv.best_trade}` : "-", accent: true },
    { label: "Worst Trade", value: adv.worst_trade != null ? `$${adv.worst_trade}` : "-", accent: false },
    { label: "Avg Win", value: adv.avg_win != null ? `$${adv.avg_win}` : "-" },
    { label: "Avg Loss", value: adv.avg_loss != null ? `$${adv.avg_loss}` : "-" },
  ];

  return (
    <div className="grid grid-cols-5 gap-2">
      {items.map((item) => (
        <Card key={item.label}>
          <div className="p-2.5">
            <div className="text-[10px] text-zinc-500">{item.label}</div>
            <div className={`text-sm font-semibold ${item.accent !== undefined ? (item.accent ? "text-emerald-400" : "text-red-400") : ""}`}>
              {item.value}
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
}

function EquityCurveCard({ data }: { data: any[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !data || data.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    const equities = data.map((d) => d.equity);
    const min = Math.min(...equities);
    const max = Math.max(...equities);
    const range = max - min || 1;

    ctx.clearRect(0, 0, w, h);

    ctx.beginPath();
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 1.5;

    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((equities[i] - min) / range) * (h - 20) - 10;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = "#71717a";
    ctx.font = "10px monospace";
    ctx.fillText(`$${max.toFixed(0)}`, 4, 14);
    ctx.fillText(`$${min.toFixed(0)}`, 4, h - 4);
  }, [data]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Equity Curve</CardTitle>
      </CardHeader>
      <div className="px-4 pb-4">
        <canvas
          ref={canvasRef}
          className="h-48 w-full"
          style={{ display: "block" }}
        />
      </div>
    </Card>
  );
}

function MonteCarloCard({ mc }: { mc: any }) {
  if (!mc || mc.status !== "SUCCESS") {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Monte Carlo</CardTitle>
        </CardHeader>
        <div className="p-4 text-zinc-500">
          {mc?.message || "No Monte Carlo data"}
        </div>
      </Card>
    );
  }

  const summary = mc.summary;
  const score = summary.robustness_score;

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          Monte Carlo ({mc.iterations} iterations)
        </CardTitle>
      </CardHeader>
      <div className="space-y-2 px-4 pb-4 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-zinc-400">Robustness Score</span>
          <Badge
            variant={
              score >= 70 ? "success" : score >= 40 ? "warning" : "danger"
            }
          >
            {score}/100
          </Badge>
        </div>
        <StatRow
          label="Median Balance"
          value={`$${summary.median_final_balance}`}
        />
        <StatRow
          label="Worst Case (5%)"
          value={`$${summary.worst_case_balance_5pct}`}
        />
        <StatRow
          label="Best Case (95%)"
          value={`$${summary.best_case_balance_95pct}`}
        />
        <StatRow
          label="Worst DD (95%)"
          value={`${summary.worst_case_dd_95pct}%`}
        />
        <StatRow label="Median Sharpe" value={summary.median_sharpe} />
        <StatRow
          label="P(Profit)"
          value={`${summary.probability_of_profit}%`}
        />
        <StatRow
          label="P(Ruin)"
          value={`${summary.probability_of_ruin}%`}
        />

        {mc.distributions?.balance?.percentiles && (
          <div className="mt-2 border-t border-zinc-800 pt-2">
            <p className="mb-1 text-xs font-medium text-zinc-500">
              Balance Distribution
            </p>
            <PercentileBar pcts={mc.distributions.balance.percentiles} prefix="$" />
          </div>
        )}
        {mc.distributions?.drawdown?.percentiles && (
          <div className="mt-2 border-t border-zinc-800 pt-2">
            <p className="mb-1 text-xs font-medium text-zinc-500">
              Drawdown Distribution
            </p>
            <PercentileBar pcts={mc.distributions.drawdown.percentiles} suffix="%" />
          </div>
        )}
      </div>
    </Card>
  );
}

function PercentileBar({
  pcts,
  prefix = "",
  suffix = "",
}: {
  pcts: Record<string, number>;
  prefix?: string;
  suffix?: string;
}) {
  return (
    <div className="flex justify-between text-xs font-mono text-zinc-400">
      {Object.entries(pcts).map(([k, v]) => (
        <div key={k} className="text-center">
          <div className="text-zinc-600">{k}</div>
          <div>
            {prefix}
            {v}
            {suffix}
          </div>
        </div>
      ))}
    </div>
  );
}

function StatRow({ label, value }: { label: string; value: any }) {
  return (
    <div className="flex justify-between text-zinc-400">
      <span>{label}</span>
      <span className="font-mono text-white">{value}</span>
    </div>
  );
}

function TradeTable({ trades }: { trades: any[] }) {
  if (!trades || trades.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Trades ({trades.length})</CardTitle>
      </CardHeader>
      <div className="max-h-80 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-zinc-900">
            <tr className="border-b border-zinc-800 text-zinc-500">
              <th className="pb-2 text-left">Direction</th>
              <th className="pb-2 text-left">Strategy</th>
              <th className="pb-2 text-right">Entry</th>
              <th className="pb-2 text-right">Exit</th>
              <th className="pb-2 text-right">PnL</th>
              <th className="pb-2 text-left">Result</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t, i) => (
              <tr
                key={i}
                className="border-b border-zinc-800/30 hover:bg-zinc-800/30"
              >
                <td className="py-1.5">
                  <Badge variant={t.direction === "BUY" ? "success" : "danger"}>
                    {t.direction}
                  </Badge>
                </td>
                <td className="text-zinc-400">{t.strategy_id}</td>
                <td className="text-right font-mono">{t.fill_price}</td>
                <td className="text-right font-mono">{t.exit_price}</td>
                <td
                  className={`text-right font-mono ${
                    t.pnl >= 0 ? "text-emerald-400" : "text-red-400"
                  }`}
                >
                  ${t.pnl}
                </td>
                <td>
                  <Badge
                    variant={
                      t.result === "TP"
                        ? "success"
                        : t.result === "SL"
                        ? "danger"
                        : "default"
                    }
                  >
                    {t.result}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function StressTestCard({ data }: { data: any }) {
  if (!data || data.error) {
    return <Card><CardHeader><CardTitle>Stress Test</CardTitle></CardHeader><div className="p-4 text-zinc-500">{data?.error || "No data"}</div></Card>;
  }
  return (
    <Card>
      <CardHeader><CardTitle>Stress Test Results</CardTitle></CardHeader>
      <table className="w-full text-sm">
        <thead><tr className="border-b border-zinc-800 text-zinc-500">
          <th className="pb-2 text-left">Scenario</th><th className="pb-2 text-right">Retention</th><th className="pb-2 text-right">Net Profit</th><th className="pb-2 text-left">Status</th>
        </tr></thead>
        <tbody>
          {Object.entries(data).map(([name, res]: [string, any]) => (
            <tr key={name} className="border-b border-zinc-800/50">
              <td className="py-2 font-mono text-xs">{name}</td>
              <td className="text-right">{res.profit_retention?.toFixed(1) ?? "-"}%</td>
              <td className="text-right font-mono">${res.metrics?.net_profit ?? "-"}</td>
              <td><Badge variant={res.passed ? "success" : "danger"}>{res.passed ? "PASS" : "FAIL"}</Badge></td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

function DrawdownCurveCard({ data }: { data: any[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!canvasRef.current || !data || data.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr; ctx.scale(dpr, dpr);
    const dds = data.map((d) => d.drawdown);
    const maxDD = Math.max(...dds, 1);
    ctx.clearRect(0, 0, w, h);
    ctx.beginPath(); ctx.strokeStyle = "#ef4444"; ctx.lineWidth = 1.5;
    ctx.fillStyle = "rgba(239,68,68,0.08)";
    ctx.moveTo(0, 0);
    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * w;
      const y = (dds[i] / maxDD) * (h - 20) + 10;
      ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.lineTo(w, 0); ctx.lineTo(0, 0); ctx.fill();
    ctx.fillStyle = "#71717a"; ctx.font = "10px monospace";
    ctx.fillText(`-${maxDD.toFixed(1)}%`, 4, h - 4);
    ctx.fillText("0%", 4, 14);
  }, [data]);
  return (
    <Card><CardHeader><CardTitle>Drawdown Curve</CardTitle></CardHeader>
      <div className="px-4 pb-4"><canvas ref={canvasRef} className="h-48 w-full" style={{ display: "block" }} /></div>
    </Card>
  );
}

function RollingSharpeCard({ data }: { data: any[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!canvasRef.current || !data || data.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr; ctx.scale(dpr, dpr);
    const vals = data.map((d) => d.sharpe);
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const range = mx - mn || 1;
    ctx.clearRect(0, 0, w, h);
    const zeroY = h - ((-mn) / range) * (h - 20) - 10;
    ctx.strokeStyle = "#333"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(w, zeroY); ctx.stroke();
    ctx.beginPath(); ctx.strokeStyle = "#8b5cf6"; ctx.lineWidth = 1.5;
    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((vals[i] - mn) / range) * (h - 20) - 10;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = "#71717a"; ctx.font = "10px monospace";
    ctx.fillText(`${mx.toFixed(1)}`, 4, 14);
    ctx.fillText(`${mn.toFixed(1)}`, 4, h - 4);
  }, [data]);
  return (
    <Card><CardHeader><CardTitle>Rolling Sharpe (30-trade)</CardTitle></CardHeader>
      <div className="px-4 pb-4"><canvas ref={canvasRef} className="h-48 w-full" style={{ display: "block" }} /></div>
    </Card>
  );
}

function PnlDistributionCard({ data }: { data: any[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!canvasRef.current || !data || data.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr; ctx.scale(dpr, dpr);
    const maxCount = Math.max(...data.map((d) => d.count), 1);
    const barW = w / data.length;
    ctx.clearRect(0, 0, w, h);
    for (let i = 0; i < data.length; i++) {
      const barH = (data[i].count / maxCount) * (h - 20);
      const x = i * barW;
      ctx.fillStyle = data[i].bucket >= 0 ? "rgba(16,185,129,0.6)" : "rgba(239,68,68,0.6)";
      ctx.fillRect(x + 1, h - barH - 10, barW - 2, barH);
    }
    ctx.fillStyle = "#71717a"; ctx.font = "10px monospace";
    if (data.length > 0) {
      ctx.fillText(`$${data[0].bucket}`, 4, h - 2);
      ctx.fillText(`$${data[data.length - 1].bucket}`, w - 50, h - 2);
    }
  }, [data]);
  return (
    <Card><CardHeader><CardTitle>PnL Distribution</CardTitle></CardHeader>
      <div className="px-4 pb-4"><canvas ref={canvasRef} className="h-48 w-full" style={{ display: "block" }} /></div>
    </Card>
  );
}

function MonthlyReturnsCard({ data }: { data: any }) {
  if (!data || Object.keys(data).length === 0) return null;
  return (
    <Card>
      <CardHeader><CardTitle>Monthly Returns</CardTitle></CardHeader>
      <div className="max-h-64 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-zinc-900">
            <tr className="border-b border-zinc-800 text-zinc-500">
              <th className="pb-2 text-left">Month</th><th className="pb-2 text-right">PnL</th>
              <th className="pb-2 text-right">Trades</th><th className="pb-2 text-right">Win Rate</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data).map(([month, m]: [string, any]) => (
              <tr key={month} className="border-b border-zinc-800/30">
                <td className="py-1 font-mono">{month}</td>
                <td className={`text-right font-mono ${m.pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>${m.pnl}</td>
                <td className="text-right text-zinc-400">{m.trades}</td>
                <td className="text-right text-zinc-400">{m.win_rate}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function SessionPerfCard({ data }: { data: any }) {
  if (!data || Object.keys(data).length === 0) return null;
  return (
    <Card>
      <CardHeader><CardTitle>Session Performance</CardTitle></CardHeader>
      <div className="space-y-2 px-4 pb-4">
        {Object.entries(data).map(([session, s]: [string, any]) => (
          <div key={session} className="flex items-center justify-between text-xs">
            <span className="font-mono text-zinc-300">{session}</span>
            <div className="flex gap-3">
              <span className={s.pnl >= 0 ? "text-emerald-400" : "text-red-400"}>${s.pnl}</span>
              <span className="text-zinc-500">{s.trades} trades</span>
              <span className="text-zinc-500">{s.win_rate}% WR</span>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

function HourlyDistCard({ data }: { data: any }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!canvasRef.current || !data) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr; ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    const entries = Object.entries(data).sort(([a], [b]) => Number(a) - Number(b));
    const maxPnl = Math.max(...entries.map(([, v]: any) => Math.abs(v.pnl)), 1);
    const barW = w / 24;
    for (const [hour, v] of entries as any) {
      const x = Number(hour) * barW;
      const barH = (Math.abs(v.pnl) / maxPnl) * (h / 2 - 15);
      ctx.fillStyle = v.pnl >= 0 ? "rgba(16,185,129,0.6)" : "rgba(239,68,68,0.6)";
      if (v.pnl >= 0) { ctx.fillRect(x + 1, h / 2 - barH, barW - 2, barH); }
      else { ctx.fillRect(x + 1, h / 2, barW - 2, barH); }
    }
    ctx.strokeStyle = "#333"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    ctx.fillStyle = "#71717a"; ctx.font = "9px monospace";
    for (let i = 0; i < 24; i += 3) ctx.fillText(`${i}h`, i * barW + 2, h - 2);
  }, [data]);
  return (
    <Card><CardHeader><CardTitle>Hourly PnL Distribution</CardTitle></CardHeader>
      <div className="px-4 pb-4"><canvas ref={canvasRef} className="h-40 w-full" style={{ display: "block" }} /></div>
    </Card>
  );
}

function DayDistCard({ data }: { data: any }) {
  if (!data || Object.keys(data).length === 0) return null;
  return (
    <Card>
      <CardHeader><CardTitle>Day of Week</CardTitle></CardHeader>
      <div className="space-y-1.5 px-4 pb-4">
        {Object.entries(data).map(([day, d]: [string, any]) => {
          const maxPnl = Math.max(...Object.values(data).map((v: any) => Math.abs(v.pnl)), 1);
          const pct = (Math.abs(d.pnl) / maxPnl) * 100;
          return (
            <div key={day} className="space-y-0.5">
              <div className="flex justify-between text-xs">
                <span className="text-zinc-400">{day}</span>
                <span className={`font-mono ${d.pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>${d.pnl} ({d.trades}t, {d.win_rate}%)</span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-zinc-800 overflow-hidden">
                <div className={`h-full ${d.pnl >= 0 ? "bg-emerald-500" : "bg-red-500"}`} style={{ width: `${pct}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function StreakCard({ data }: { data: any }) {
  if (!data) return null;
  return (
    <Card>
      <CardHeader><CardTitle>Streaks</CardTitle></CardHeader>
      <div className="space-y-2 px-4 pb-4 text-sm">
        <StatRow label="Max Win Streak" value={data.max_win_streak} />
        <StatRow label="Max Loss Streak" value={data.max_loss_streak} />
      </div>
    </Card>
  );
}

function DurationCard({ data }: { data: any }) {
  if (!data) return null;
  return (
    <Card>
      <CardHeader><CardTitle>Trade Duration</CardTitle></CardHeader>
      <div className="space-y-2 px-4 pb-4 text-sm">
        <StatRow label="Average" value={`${data.avg_minutes} min`} />
        <StatRow label="Median" value={`${data.median_minutes} min`} />
        <StatRow label="Shortest" value={`${data.min_minutes} min`} />
        <StatRow label="Longest" value={`${data.max_minutes} min`} />
      </div>
    </Card>
  );
}

function AdvancedMetricsCard({ data }: { data: any }) {
  if (!data) return null;
  return (
    <Card>
      <CardHeader><CardTitle>Advanced</CardTitle></CardHeader>
      <div className="space-y-2 px-4 pb-4 text-sm">
        <StatRow label="Recovery Factor" value={data.recovery_factor} />
        <StatRow label="Ulcer Index" value={data.ulcer_index} />
        <StatRow label="Tail Ratio" value={data.tail_ratio} />
        <StatRow label="Payoff Ratio" value={data.payoff_ratio} />
        <StatRow label="Kelly %" value={`${data.kelly_criterion}%`} />
      </div>
    </Card>
  );
}

function WalkForwardCard({ data }: { data: any }) {
  if (!data || data.error || !Array.isArray(data)) return null;
  return (
    <Card>
      <CardHeader><CardTitle>Walk-Forward Validation</CardTitle></CardHeader>
      <div className="max-h-64 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-zinc-900">
            <tr className="border-b border-zinc-800 text-zinc-500">
              <th className="pb-2 text-left">Window</th><th className="pb-2 text-right">IS Profit</th>
              <th className="pb-2 text-right">OOS Profit</th><th className="pb-2 text-right">WFO Ratio</th>
              <th className="pb-2 text-left">Robustness</th>
            </tr>
          </thead>
          <tbody>
            {data.map((r: any, i: number) => (
              <tr key={i} className="border-b border-zinc-800/30">
                <td className="py-1 font-mono">{r.window}</td>
                <td className="text-right font-mono">${r.is_profit}</td>
                <td className={`text-right font-mono ${r.oos_profit >= 0 ? "text-emerald-400" : "text-red-400"}`}>${r.oos_profit}</td>
                <td className="text-right font-mono">{r.wfo_ratio}</td>
                <td><Badge variant={r.robustness === "EXCELLENT" || r.robustness === "GOOD" ? "success" : r.robustness === "MARGINAL" ? "warning" : "danger"}>{r.robustness}</Badge></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
