"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

type BacktestStatus = "idle" | "pending" | "running" | "completed" | "failed";

export default function BacktestingPage() {
  const [symbol, setSymbol] = useState("XAUUSDm");
  const [strategyId, setStrategyId] = useState("SMC");
  const [timeframe, setTimeframe] = useState("M5");
  const [balance, setBalance] = useState(10000);
  const [mcIterations, setMcIterations] = useState(500);
  const [stressTest, setStressTest] = useState(false);
  const [startDate, setStartDate] = useState("");

  const [status, setStatus] = useState<BacktestStatus>("idle");
  const [runId, setRunId] = useState<number | null>(null);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState("");
  const [runs, setRuns] = useState<any[]>([]);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    api.getBacktestRuns(10).then(setRuns).catch(() => {});
  }, []);

  const pollStatus = useCallback(
    (id: number) => {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const res = await api.getBacktestStatus(id);
          if (res.status === "COMPLETED") {
            clearInterval(pollRef.current!);
            pollRef.current = null;
            setStatus("completed");
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
      }, 3000);
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
    try {
      const res = await api.runBacktest({
        symbol,
        strategy_id: strategyId,
        timeframe,
        start_date: startDate || undefined,
        initial_balance: balance,
        monte_carlo_iterations: mcIterations,
        stress_test: stressTest,
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

      <div className="grid grid-cols-4 gap-4">
        {/* Config Panel */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
          </CardHeader>
          <div className="space-y-3">
            <Field label="Symbol">
              <select
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
              >
                <option>XAUUSDm</option>
                <option>EURUSD</option>
                <option>GBPUSD</option>
              </select>
            </Field>
            <Field label="Strategy">
              <input
                value={strategyId}
                onChange={(e) => setStrategyId(e.target.value)}
                className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
              />
            </Field>
            <Field label="Timeframe">
              <select
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value)}
                className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
              >
                <option>M1</option>
                <option>M5</option>
                <option>M15</option>
                <option>H1</option>
              </select>
            </Field>
            <Field label="Start Date (optional)">
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
              />
            </Field>
            <Field label="Initial Balance">
              <input
                type="number"
                value={balance}
                onChange={(e) => setBalance(Number(e.target.value))}
                className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
              />
            </Field>
            <Field label="Monte Carlo Iterations">
              <input
                type="number"
                value={mcIterations}
                onChange={(e) => setMcIterations(Number(e.target.value))}
                min={100}
                max={5000}
                className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
              />
            </Field>
            <label className="flex items-center gap-2 text-xs text-zinc-400">
              <input
                type="checkbox"
                checked={stressTest}
                onChange={(e) => setStressTest(e.target.checked)}
                className="rounded"
              />
              Run Stress Test
            </label>
            <button
              onClick={handleRun}
              disabled={status === "pending" || status === "running"}
              className="mt-2 w-full rounded bg-brand-600 px-3 py-2 text-sm font-medium hover:bg-brand-500 disabled:opacity-50"
            >
              {status === "running"
                ? "Running..."
                : status === "pending"
                ? "Starting..."
                : "Run Backtest"}
            </button>

            {runs.length > 0 && (
              <div className="mt-4 border-t border-zinc-800 pt-3">
                <p className="mb-2 text-xs font-medium text-zinc-400">
                  Recent Runs
                </p>
                <div className="max-h-40 space-y-1 overflow-auto">
                  {runs.map((r) => (
                    <button
                      key={r.id}
                      onClick={() => loadPastResult(r.id)}
                      className={`flex w-full items-center justify-between rounded px-2 py-1 text-xs hover:bg-zinc-800 ${
                        runId === r.id ? "bg-zinc-800" : ""
                      }`}
                    >
                      <span>#{r.id}</span>
                      <Badge
                        variant={
                          r.status === "COMPLETED"
                            ? "success"
                            : r.status === "FAILED"
                            ? "danger"
                            : "default"
                        }
                      >
                        {r.status}
                      </Badge>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* Results Panel */}
        <div className="col-span-3 space-y-4">
          {status === "failed" && error && (
            <Card>
              <div className="p-4 text-red-400">{error}</div>
            </Card>
          )}

          {status === "running" && (
            <Card>
              <div className="flex h-32 items-center justify-center text-zinc-400">
                <div className="text-center">
                  <div className="mb-2 text-lg">Backtest Running...</div>
                  <div className="text-xs text-zinc-500">
                    Processing {symbol} on {timeframe}. This may take several
                    minutes.
                  </div>
                </div>
              </div>
            </Card>
          )}

          {status === "completed" && results && (
            <>
              <MetricsGrid metrics={results.metrics} />
              <div className="grid grid-cols-2 gap-4">
                <EquityCurveCard data={results.equity_curve} />
                <MonteCarloCard mc={results.monte_carlo} />
              </div>
              <TradeTable trades={results.trades} />
              {results.stress_test && (
                <StressTestCard data={results.stress_test} />
              )}
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

function MetricsGrid({ metrics }: { metrics: any }) {
  if (!metrics || metrics.status === "NO_TRADES") {
    return (
      <Card>
        <div className="p-4 text-zinc-500">No trades executed.</div>
      </Card>
    );
  }

  const items = [
    { label: "Net Profit", value: `$${metrics.net_profit}`, accent: metrics.net_profit >= 0 },
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
  ];

  return (
    <div className="grid grid-cols-4 gap-3">
      {items.map((item) => (
        <Card key={item.label}>
          <div className="p-3">
            <div className="text-xs text-zinc-500">{item.label}</div>
            <div
              className={`text-lg font-semibold ${
                item.accent !== undefined
                  ? item.accent
                    ? "text-emerald-400"
                    : "text-red-400"
                  : ""
              }`}
            >
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
    return (
      <Card>
        <CardHeader>
          <CardTitle>Stress Test</CardTitle>
        </CardHeader>
        <div className="p-4 text-zinc-500">{data?.error || "No data"}</div>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Stress Test Results</CardTitle>
      </CardHeader>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 text-zinc-500">
            <th className="pb-2 text-left">Scenario</th>
            <th className="pb-2 text-right">Profit Retention</th>
            <th className="pb-2 text-right">Net Profit</th>
            <th className="pb-2 text-left">Status</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data).map(([name, res]: [string, any]) => (
            <tr key={name} className="border-b border-zinc-800/50">
              <td className="py-2 font-mono">{name}</td>
              <td className="text-right">
                {res.profit_retention?.toFixed(1) ?? "-"}%
              </td>
              <td className="text-right font-mono">
                ${res.metrics?.net_profit ?? "-"}
              </td>
              <td>
                <Badge variant={res.passed ? "success" : "danger"}>
                  {res.passed ? "PASS" : "FAIL"}
                </Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}
