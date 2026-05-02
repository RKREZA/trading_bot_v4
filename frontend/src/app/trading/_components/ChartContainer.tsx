"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

const TIMEFRAMES = ["M1", "M5", "M15", "H1", "D1"] as const;

interface DragState {
  active: boolean;
  lineKey: string;
  ticket: number;
  symbol: string;
  type: "sl" | "tp";
  originalPrice: number;
  currentPrice: number;
}

interface Props {
  symbol: string;
  symbols: string[];
  onSymbolChange: (s: string) => void;
  positions: any[];
  onModifySLTP?: (ticket: number, symbol: string, sl: number, tp: number) => void;
}

export function ChartContainer({
  symbol,
  symbols,
  onSymbolChange,
  positions,
  onModifySLTP,
}: Props) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<any>(null);
  const candleSeriesRef = useRef<any>(null);
  const priceLinesRef = useRef<Map<string, any>>(new Map());
  const dragRef = useRef<DragState>({
    active: false, lineKey: "", ticket: 0, symbol: "", type: "sl", originalPrice: 0, currentPrice: 0,
  });
  const crosshairPriceRef = useRef<number>(0);
  const positionsRef = useRef<any[]>([]);
  const [timeframe, setTimeframe] = useState<string>("M5");
  const [dragInfo, setDragInfo] = useState<{ type: string; price: number } | null>(null);

  positionsRef.current = positions;

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
          crosshair: { mode: 0 },
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
          if (!param.point) return;
          const priceData = param.seriesData?.get(series);
          if (priceData?.close) {
            crosshairPriceRef.current = priceData.close;
          }

          if (dragRef.current.active) {
            const price = crosshairPriceRef.current;
            dragRef.current.currentPrice = price;
            const lineKey = dragRef.current.lineKey;
            const line = priceLinesRef.current.get(lineKey);
            if (line) {
              line.applyOptions({ price });
            }
            setDragInfo({ type: dragRef.current.type.toUpperCase(), price });
          }
        });

        const ro = new ResizeObserver(() => {
          if (chart && container.clientWidth > 0) {
            chart.applyOptions({ width: container.clientWidth });
          }
        });
        ro.observe(container);
        (container as any).__ro_cleanup = () => ro.disconnect();
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

  // Price lines for ALL positions matching current symbol
  useEffect(() => {
    const series = candleSeriesRef.current;
    if (!series) return;

    const prevLines = priceLinesRef.current;
    const newKeys = new Set<string>();
    const symbolPositions = positions.filter((p) => p.symbol === symbol);

    for (const pos of symbolPositions) {
      const isBuy = pos.type === 0;
      const entryKey = `entry_${pos.ticket}`;
      const slKey = `sl_${pos.ticket}`;
      const tpKey = `tp_${pos.ticket}`;

      if (pos.price_open) {
        newKeys.add(entryKey);
        const opts = {
          price: pos.price_open,
          color: "#3b82f6",
          lineWidth: 1,
          lineStyle: 2,
          axisLabelVisible: true,
          title: `#${pos.ticket} ${isBuy ? "BUY" : "SELL"} ${pos.volume}`,
        };
        if (prevLines.has(entryKey)) {
          prevLines.get(entryKey).applyOptions(opts);
        } else {
          prevLines.set(entryKey, series.createPriceLine(opts));
        }
      }

      if (pos.sl && pos.sl > 0) {
        newKeys.add(slKey);
        const dragActive = dragRef.current.active && dragRef.current.lineKey === slKey;
        const opts = {
          price: dragActive ? dragRef.current.currentPrice : pos.sl,
          color: "#ef4444",
          lineWidth: 2,
          lineStyle: 0,
          axisLabelVisible: true,
          title: `SL #${pos.ticket} (drag to move)`,
          draggable: true,
        };
        if (prevLines.has(slKey)) {
          if (!dragActive) prevLines.get(slKey).applyOptions(opts);
        } else {
          prevLines.set(slKey, series.createPriceLine(opts));
        }
      }

      if (pos.tp && pos.tp > 0) {
        newKeys.add(tpKey);
        const dragActive = dragRef.current.active && dragRef.current.lineKey === tpKey;
        const opts = {
          price: dragActive ? dragRef.current.currentPrice : pos.tp,
          color: "#10b981",
          lineWidth: 2,
          lineStyle: 0,
          axisLabelVisible: true,
          title: `TP #${pos.ticket} (drag to move)`,
          draggable: true,
        };
        if (prevLines.has(tpKey)) {
          if (!dragActive) prevLines.get(tpKey).applyOptions(opts);
        } else {
          prevLines.set(tpKey, series.createPriceLine(opts));
        }
      }
    }

    Array.from(prevLines.entries()).forEach(([key, line]) => {
      if (!newKeys.has(key)) {
        series.removePriceLine(line);
        prevLines.delete(key);
      }
    });
  }, [positions, symbol]);

  const findNearestDraggableLine = useCallback((price: number): DragState | null => {
    const symbolPositions = positionsRef.current.filter((p) => p.symbol === symbol);
    if (symbolPositions.length === 0 || !price) return null;

    const chart = chartInstanceRef.current;
    if (!chart) return null;

    const visibleRange = chart.timeScale().getVisibleLogicalRange();
    if (!visibleRange) return null;

    const series = candleSeriesRef.current;
    if (!series) return null;

    let highPrice = 0;
    let lowPrice = Infinity;
    for (const pos of symbolPositions) {
      if (pos.price_open > highPrice) highPrice = pos.price_open;
      if (pos.sl > 0 && pos.sl < lowPrice) lowPrice = pos.sl;
      if (pos.tp > 0 && pos.tp > highPrice) highPrice = pos.tp;
      if (pos.price_open < lowPrice) lowPrice = pos.price_open;
    }
    const priceRange = highPrice - lowPrice || price * 0.01;
    const threshold = priceRange * 0.03;

    let closest: DragState | null = null;
    let closestDist = Infinity;

    for (const pos of symbolPositions) {
      if (pos.sl && pos.sl > 0) {
        const dist = Math.abs(price - pos.sl);
        if (dist < threshold && dist < closestDist) {
          closestDist = dist;
          closest = {
            active: true,
            lineKey: `sl_${pos.ticket}`,
            ticket: pos.ticket,
            symbol: pos.symbol,
            type: "sl",
            originalPrice: pos.sl,
            currentPrice: pos.sl,
          };
        }
      }
      if (pos.tp && pos.tp > 0) {
        const dist = Math.abs(price - pos.tp);
        if (dist < threshold && dist < closestDist) {
          closestDist = dist;
          closest = {
            active: true,
            lineKey: `tp_${pos.ticket}`,
            ticket: pos.ticket,
            symbol: pos.symbol,
            type: "tp",
            originalPrice: pos.tp,
            currentPrice: pos.tp,
          };
        }
      }
    }
    return closest;
  }, [symbol]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const price = crosshairPriceRef.current;
    if (!price) return;

    const nearest = findNearestDraggableLine(price);
    if (nearest) {
      e.preventDefault();
      dragRef.current = nearest;
      setDragInfo({ type: nearest.type.toUpperCase(), price: nearest.originalPrice });
    }
  }, [findNearestDraggableLine]);

  const handleMouseUp = useCallback(() => {
    const drag = dragRef.current;
    if (!drag.active) return;

    const newPrice = drag.currentPrice;
    const oldPrice = drag.originalPrice;

    dragRef.current = {
      active: false, lineKey: "", ticket: 0, symbol: "", type: "sl", originalPrice: 0, currentPrice: 0,
    };
    setDragInfo(null);

    if (Math.abs(newPrice - oldPrice) < 0.00001) return;

    const pos = positionsRef.current.find((p) => p.ticket === drag.ticket);
    if (!pos) return;

    const label = drag.type === "sl" ? "Stop Loss" : "Take Profit";
    const confirmed = confirm(
      `Move ${label} for #${drag.ticket}?\n\nFrom: ${oldPrice.toFixed(5)}\nTo: ${newPrice.toFixed(5)}`
    );

    if (confirmed && onModifySLTP) {
      const sl = drag.type === "sl" ? newPrice : (pos.sl ?? 0);
      const tp = drag.type === "tp" ? newPrice : (pos.tp ?? 0);
      onModifySLTP(drag.ticket, drag.symbol, sl, tp);
    } else {
      const line = priceLinesRef.current.get(drag.lineKey);
      if (line) {
        line.applyOptions({ price: oldPrice });
      }
    }
  }, [onModifySLTP]);

  const handleMouseLeave = useCallback(() => {
    if (dragRef.current.active) {
      const line = priceLinesRef.current.get(dragRef.current.lineKey);
      if (line) {
        line.applyOptions({ price: dragRef.current.originalPrice });
      }
      dragRef.current = {
        active: false, lineKey: "", ticket: 0, symbol: "", type: "sl", originalPrice: 0, currentPrice: 0,
      };
      setDragInfo(null);
    }
  }, []);

  const symbolPositions = positions.filter((p) => p.symbol === symbol);

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center gap-3 w-full">
          <CardTitle>{symbol || "Chart"}</CardTitle>

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

          {dragInfo && (
            <span className="text-xs font-medium animate-pulse">
              <span className={dragInfo.type === "SL" ? "text-red-400" : "text-emerald-400"}>
                Dragging {dragInfo.type}: {dragInfo.price.toFixed(5)}
              </span>
              <span className="ml-2 text-zinc-500">Release to confirm</span>
            </span>
          )}

          {symbolPositions.length > 0 && !dragInfo && (
            <span className="text-xs text-zinc-500">
              {symbolPositions.length} position{symbolPositions.length > 1 ? "s" : ""} on chart
              <span className="ml-1 text-zinc-600">- drag SL/TP lines to move</span>
            </span>
          )}

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

      {/* Trade info overlay */}
      {symbolPositions.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2 px-1">
          {symbolPositions.map((p) => (
            <div
              key={p.ticket}
              className="flex items-center gap-2 rounded border border-zinc-800 bg-zinc-900/60 px-2 py-1 text-xs"
            >
              <Badge variant={p.type === 0 ? "success" : "danger"}>
                {p.type === 0 ? "BUY" : "SELL"}
              </Badge>
              <span className="font-mono text-zinc-300">#{p.ticket}</span>
              <span className="font-mono">{p.volume}</span>
              <span className="text-zinc-500">@</span>
              <span className="font-mono text-blue-400">{p.price_open?.toFixed(5)}</span>
              {p.sl > 0 && <span className="font-mono text-red-400">SL:{p.sl.toFixed(5)}</span>}
              {p.tp > 0 && <span className="font-mono text-emerald-400">TP:{p.tp.toFixed(5)}</span>}
              <span className={`font-mono ${p.profit >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                {p.profit >= 0 ? "+" : ""}{p.profit?.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}

      <div
        ref={chartRef}
        className={`w-full ${dragRef.current?.active ? "cursor-grabbing" : "cursor-crosshair"}`}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
    </Card>
  );
}
