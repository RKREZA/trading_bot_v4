"use client";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { useAppStore } from "@/lib/store";
import { api } from "@/lib/api";

export function ConnectionControls({ onConnected }: { onConnected?: () => void }) {
  const store = useAppStore();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState("");

  const handleConnect = async () => {
    setError("");
    setLoading("connect");
    try {
      await api.connectMT5();
      store.setMt5Connected(true);
      onConnected?.();
    } catch (e: any) {
      setError(e.message || "Failed to connect");
    } finally {
      setLoading("");
    }
  };

  const handleDisconnect = async () => {
    setLoading("disconnect");
    try {
      await api.disconnectMT5();
      store.setMt5Connected(false);
      store.setTrading(false);
    } catch {} finally {
      setLoading("");
    }
  };

  const handleStart = async () => {
    setError("");
    setLoading("start");
    try {
      await api.startTrading();
      store.setTrading(true);
    } catch (e: any) {
      setError(e.message || "Failed to start");
    } finally {
      setLoading("");
    }
  };

  const handleStop = async () => {
    setLoading("stop");
    try {
      await api.stopTrading();
      store.setTrading(false);
    } catch {} finally {
      setLoading("");
    }
  };

  const handleKill = async () => {
    if (!confirm("Activate kill switch? This will halt ALL trading.")) return;
    try {
      await api.killSwitch();
      store.setTrading(false);
      store.setKillSwitch(true);
    } catch {}
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        {/* Connection group */}
        <div className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2">
          <Badge variant={store.mt5Connected ? "success" : "default"}>
            {store.mt5Connected ? "MT5 Connected" : "MT5 Offline"}
          </Badge>
          {!store.mt5Connected ? (
            <button
              onClick={handleConnect}
              disabled={loading === "connect"}
              className="rounded bg-emerald-600 px-3 py-1 text-xs font-medium hover:bg-emerald-500 disabled:opacity-50"
            >
              {loading === "connect" ? "Connecting…" : "Connect"}
            </button>
          ) : (
            <button
              onClick={handleDisconnect}
              disabled={!!loading || store.isTrading}
              className="rounded bg-zinc-700 px-3 py-1 text-xs font-medium hover:bg-zinc-600 disabled:opacity-50"
              title={store.isTrading ? "Stop trading first" : undefined}
            >
              {loading === "disconnect" ? "…" : "Disconnect"}
            </button>
          )}
        </div>

        {/* Trading group */}
        <div className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2">
          <Badge variant={store.isTrading ? "success" : "default"}>
            {store.isTrading ? "Trading" : "Idle"}
          </Badge>
          {!store.isTrading ? (
            <button
              onClick={handleStart}
              disabled={!store.mt5Connected || !!loading}
              className="rounded bg-emerald-600 px-3 py-1 text-xs font-medium hover:bg-emerald-500 disabled:opacity-50"
            >
              {loading === "start" ? "Starting…" : "Start"}
            </button>
          ) : (
            <button
              onClick={handleStop}
              disabled={!!loading}
              className="rounded bg-zinc-700 px-3 py-1 text-xs font-medium hover:bg-zinc-600 disabled:opacity-50"
            >
              {loading === "stop" ? "…" : "Stop"}
            </button>
          )}
          <button
            onClick={handleKill}
            disabled={!store.mt5Connected}
            className="rounded bg-red-700 px-3 py-1 text-xs font-medium hover:bg-red-600 disabled:opacity-50"
          >
            Kill
          </button>
          {store.killSwitch && <Badge variant="danger">KILL ACTIVE</Badge>}
        </div>
      </div>

      {error && (
        <div className="rounded bg-red-900/40 border border-red-800 px-3 py-1.5 text-xs text-red-300">
          {error}
        </div>
      )}
    </div>
  );
}
