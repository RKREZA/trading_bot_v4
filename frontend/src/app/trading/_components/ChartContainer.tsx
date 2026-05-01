"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";

const TIMEFRAMES = ["M1", "M5", "M15", "H1", "D1"] as const;

type ChartMode = "normal" | "set_sl" | "set_tp";

interface Props {
  symbol: string;
  symbols: string[];
  onSymbolChange: (s: string) => void;
  positions: any[];
  onSetSL?: (ticket: number, symbol: string, price: number) => void;
  onSetTP?: (ticket: number, symbol: string, price: number) => void;
}

export function ChartContainer({
  symbol,
  symbols,
  onSymbolChange,
  positions,
  onSetSL,
  onSetTP,
}: Props) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<any>(null);
  const candleSeriesRef = useRef<any>(null);
  const priceLinesRef = useRef<Map<string, any>>(new Map());
  const [timeframe, setTimeframe] = useState<string>("M5");
  const [chartMode, setChartMode] = useState<ChartMode>("normal");
  const [crosshairPrice, setCrosshairPrice] = useState<number | null>(null);

  // Build chart when symbol or timeframe changes
  useEffect(() => {
    if (!chartRef.current || !symbol) return;

    let disposed = false;
    let chart: any = null;
    const container = chartRef.current;
    container.innerHTML = "";
    priceLinesRef.current.clear();

    (async () => {
      try {
        const { createChart } = await import("lightweight-charts");
        if (disposed) return;

        chart = createChart(container, {
          layout: { background: { color: "#09090b" }, textColor: "#a1a1aa" },
          grid: {
            vertLines: { color: "#27272a" },
            horzLines: { color: "#27272a" },
          },
          crosshair: {
            mode: 0,
          },
          width: container.clientWidth,
          height: 450,
        });

        const series = chart.addCandlestickSeries({
          upColor: "#10b981",
          downColor: "#ef4444",
          wickUpColor: "#10b981",
          wickDownColor: "#ef4444",
        });

        const data = await api.getCandleData(symbol, timeframe, 300);
        if (disposed) { chart.remove(); return; }

        if (data?.candles) {
          const candles = data.candles.time.map((t: number, i: number) => ({
            time: t,
            open: data.candles.open[i],
            high: data.candles.high[i],
            low: data.candles.low[i],
            close: data.candles.close[i],
          }));
          series.setData(candles);
        }

        chart.timeScale().fitContent();

        chartInstanceRef.current = chart;
        candleSeriesRef.current = series;

        chart.subscribeCrosshairMove((param: any) => {
          if (param.point && param.seriesData?.size > 0) {
            const price = param.seriesData.values().next().value?.close;
            if (price) setCrosshairPrice(price);
          }
        });

        const ro = new ResizeObserver(() => {
          if (chart && container.clientWidth > 0) {
            chart.applyOptions({ width: container.clientWidth });
          }
        });
        ro.observe(container);

        const cleanup = () => ro.disconnect();
        (container as any).__ro_cleanup = cleanup;
      } catch {
        if (chart && !disposed) chart.remove();
      }
    })();

    return () => {
      disposed = true;
      if ((container as any).__ro_cleanup) {
        (container as any).__ro_cleanup();
        delete (container as any).__ro_cleanup;
      }
      if (chart) {
        try { chart.remove(); } catch {}
        chart = null;
      }
      chartInstanceRef.current = null;
      candleSeriesRef.current = null;
    };
  }, [symbol, timeframe]);

  // Update price lines for positions matching current symbol
  useEffect(() => {
    const series = candleSeriesRef.current;
    if (!series) return;

    const prevLines = priceLinesRef.current;
    const newKeys = new Set<string>();

    const symbolPositions = positions.filter((p) => p.symbol === symbol);

    for (const pos of symbolPositions) {
      const entryKey = `entry_${pos.ticket}`;
      const slKey = `sl_${pos.ticket}`;
      const tpKey = `tp_${pos.ticket}`;

      // Entry line
      if (pos.price_open) {
        newKeys.add(entryKey);
        const opts = {
          price: pos.price_open,
          color: "#3b82f6",
          lineWidth: 1,
          lineStyle: 2,
          axisLabelVisible: true,
          title: `#${pos.ticket} ${pos.type === 0 ? "BUY" : "SELL"}`,
        };
        if (prevLines.has(entryKey)) {
          prevLines.get(entryKey).applyOptions(opts);
        } else {
          prevLines.set(entryKey, series.createPriceLine(opts));
        }
      }

      // SL line
      if (pos.sl && pos.sl > 0) {
        newKeys.add(slKey);
        const opts = {
          price: pos.sl,
          color: "#ef4444",
          lineWidth: 1,
          lineStyle: 0,
          axisLabelVisible: true,
          title: `SL #${pos.ticket}`,
        };
        if (prevLines.has(slKey)) {
          prevLines.get(slKey).applyOptions(opts);
        } else {
          prevLines.set(slKey, series.createPriceLine(opts));
        }
      }

      // TP line
      if (pos.tp && pos.tp > 0) {
        newKeys.add(tpKey);
        const opts = {
          price: pos.tp,
          color: "#10b981",
          lineWidth: 1,
          lineStyle: 0,
          axisLabelVisible: true,
          title: `TP #${pos.ticket}`,
        };
        if (prevLines.has(tpKey)) {
          prevLines.get(tpKey).applyOptions(opts);
        } else {
          prevLines.set(tpKey, series.createPriceLine(opts));
        }
      }
    }

    // Remove stale lines
    Array.from(prevLines.entries()).forEach(([key, line]) => {
      if (!newKeys.has(key)) {
        series.removePriceLine(line);
        prevLines.delete(key);
      }
    });
  }, [positions, symbol]);

  // Chart click handler for SL/TP setting
  const handleChartClick = useCallback(() => {
    if (chartMode === "normal" || !crosshairPrice) return;

    const symbolPositions = positions.filter((p) => p.symbol === symbol);
    if (symbolPositions.length === 0) return;

    const pos = symbolPositions[0];
    if (chartMode === "set_sl" && onSetSL) {
      onSetSL(pos.ticket, pos.symbol, crosshairPrice);
      setChartMode("normal");
    } else if (chartMode === "set_tp" && onSetTP) {
      onSetTP(pos.ticket, pos.symbol, crosshairPrice);
      setChartMode("normal");
    }
  }, [chartMode, crosshairPrice, positions, symbol, onSetSL, onSetTP]);

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center gap-3 w-full">
          <CardTitle>{symbol || "Chart"}</CardTitle>

          {/* Timeframe buttons */}
          <div className="flex rounded-md border border-zinc-700 overflow-hidden">
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-2.5 py-1 text-xs font-medium transition-colors ${
                  timeframe === tf
                    ? "bg-zinc-700 text-white"
                    : "bg-zinc-900 text-zinc-500 hover:text-zinc-300"
                }`}
              >
                {tf}
              </button>
            ))}
          </div>

          {/* Chart mode */}
          <div className="flex rounded-md border border-zinc-700 overflow-hidden">
            {(["normal", "set_sl", "set_tp"] as ChartMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setChartMode(mode)}
                className={`px-2.5 py-1 text-xs font-medium transition-colors ${
                  chartMode === mode
                    ? mode === "set_sl" ? "bg-red-900/60 text-red-400"
                      : mode === "set_tp" ? "bg-emerald-900/60 text-emerald-400"
                      : "bg-zinc-700 text-white"
                    : "bg-zinc-900 text-zinc-500 hover:text-zinc-300"
                }`}
              >
                {mode === "normal" ? "Normal" : mode === "set_sl" ? "Set SL" : "Set TP"}
              </button>
            ))}
          </div>

          {crosshairPrice && chartMode !== "normal" && (
            <span className="text-xs text-zinc-400">
              Price: <span className="font-mono text-white">{crosshairPrice.toFixed(5)}</span>
            </span>
          )}

          {/* Symbol selector */}
          <select
            value={symbol}
            onChange={(e) => onSymbolChange(e.target.value)}
            className="ml-auto rounded bg-zinc-800 px-2 py-1 text-xs text-white"
          >
            {symbols.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
      </CardHeader>
      <div
        ref={chartRef}
        className="w-full cursor-crosshair"
        onClick={handleChartClick}
      />
    </Card>
  );
}
