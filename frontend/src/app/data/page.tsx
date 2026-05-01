"use client";
import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

export default function DataPage() {
  const [syncStatus, setSyncStatus] = useState<any>({});
  const [syncing, setSyncing] = useState<string | null>(null);

  useEffect(() => {
    api.getSyncStatus().then(setSyncStatus).catch(() => {});
  }, []);

  const handleSync = async (symbol: string, tf: string) => {
    const key = `${symbol}_${tf}`;
    setSyncing(key);
    try {
      await api.triggerSync(symbol, tf);
      const updated = await api.getSyncStatus();
      setSyncStatus(updated);
    } catch {}
    setSyncing(null);
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Data Management</h2>

      {Object.keys(syncStatus).length === 0 ? (
        <p className="text-zinc-500">No sync data available</p>
      ) : (
        Object.entries(syncStatus).map(([symbol, timeframes]: [string, any]) => (
          <Card key={symbol}>
            <CardHeader><CardTitle>{symbol}</CardTitle></CardHeader>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-800 text-zinc-500">
                  <th className="pb-2 text-left">Timeframe</th>
                  <th className="pb-2 text-left">Status</th>
                  <th className="pb-2 text-right">Last Timestamp</th>
                  <th className="pb-2 text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(timeframes).map(([tf, info]: [string, any]) => (
                  <tr key={tf} className="border-b border-zinc-800/50">
                    <td className="py-2 font-mono">{tf}</td>
                    <td>
                      <Badge variant={info.has_data ? "success" : "warning"}>
                        {info.has_data ? "Cached" : "Empty"}
                      </Badge>
                    </td>
                    <td className="text-right font-mono text-xs">
                      {info.last_timestamp > 0
                        ? new Date(info.last_timestamp * 1000).toISOString()
                        : "-"}
                    </td>
                    <td className="text-right">
                      <button
                        onClick={() => handleSync(symbol, tf)}
                        disabled={syncing === `${symbol}_${tf}`}
                        className="rounded bg-zinc-800 px-2 py-1 text-xs hover:bg-zinc-700 disabled:opacity-50"
                      >
                        {syncing === `${symbol}_${tf}` ? "Syncing..." : "Sync"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        ))
      )}
    </div>
  );
}
