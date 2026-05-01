"use client";
import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useAppStore } from "@/lib/store";
import { useWebSocket } from "@/lib/ws";
import { api } from "@/lib/api";
import { formatCurrency, formatPercent } from "@/lib/utils";

export default function DashboardPage() {
  const store = useAppStore();
  const [status, setStatus] = useState<any>(null);
  const [positions, setPositions] = useState<any[]>([]);

  useWebSocket({
    onMessage: (msg) => {
      if (msg.type === "METRICS") {
        store.updateFromMetrics(msg.data);
      }
    },
  });

  useEffect(() => {
    api.getStatus().then(setStatus).catch(() => {});
    api.getPositions().then(setPositions).catch(() => {});
    const interval = setInterval(() => {
      api.getPositions().then(setPositions).catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold">Dashboard</h2>

      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader><CardTitle>Balance</CardTitle></CardHeader>
          <p className="text-2xl font-bold">{formatCurrency(store.account.balance)}</p>
        </Card>
        <Card>
          <CardHeader><CardTitle>Equity</CardTitle></CardHeader>
          <p className="text-2xl font-bold">{formatCurrency(store.account.equity)}</p>
        </Card>
        <Card>
          <CardHeader><CardTitle>Drawdown</CardTitle></CardHeader>
          <p className={`text-2xl font-bold ${store.account.drawdownPct > 5 ? "text-red-400" : ""}`}>
            {store.account.drawdownPct.toFixed(2)}%
          </p>
        </Card>
        <Card>
          <CardHeader><CardTitle>P&L</CardTitle></CardHeader>
          <p className={`text-2xl font-bold ${store.account.profit >= 0 ? "text-emerald-400" : "text-red-400"}`}>
            {formatCurrency(store.account.profit)}
          </p>
        </Card>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader><CardTitle>System Status</CardTitle></CardHeader>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-zinc-400">MT5</span>
              <Badge variant={status?.mt5_connected ? "success" : "danger"}>
                {status?.mt5_connected ? "Connected" : "Disconnected"}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">Trading</span>
              <Badge variant={store.isTrading ? "success" : "default"}>
                {store.isTrading ? "Active" : "Idle"}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">Strategies</span>
              <span>{status?.active_strategies?.length ?? 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">Server Time</span>
              <span className="font-mono text-xs">{status?.server_time ?? "-"}</span>
            </div>
          </div>
        </Card>

        <Card>
          <CardHeader><CardTitle>Open Positions ({positions.length})</CardTitle></CardHeader>
          {positions.length === 0 ? (
            <p className="text-sm text-zinc-500">No open positions</p>
          ) : (
            <div className="max-h-48 overflow-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-zinc-500">
                    <th className="text-left pb-1">Symbol</th>
                    <th className="text-left pb-1">Dir</th>
                    <th className="text-right pb-1">Volume</th>
                    <th className="text-right pb-1">Profit</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map((p: any, i: number) => (
                    <tr key={i} className="border-t border-zinc-800">
                      <td className="py-1">{p.symbol}</td>
                      <td>{p.type === 0 ? "BUY" : "SELL"}</td>
                      <td className="text-right">{p.volume}</td>
                      <td className={`text-right ${p.profit >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                        {p.profit?.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
