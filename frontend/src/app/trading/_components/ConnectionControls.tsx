"use client";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { useAppStore } from "@/lib/store";
import { api } from "@/lib/api";

interface Props {
  onConnected?: () => void;
  assignments?: Record<string, string[]>;
}

export function ConnectionControls({ onConnected, assignments }: Props) {
  const store = useAppStore();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState("");

  const handleConnect = async () => {
    if (!confirm("Connect to MT5 terminal?")) return;
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
    if (!confirm("Disconnect from MT5? This will also stop trading.")) return;
    setLoading("disconnect");
    try {
      await api.disconnectMT5();
      store.setMt5Connected(false);
      store.setTrading(false);
    } catch {} finally {
      setLoading("");
    }
  };

  const validAssignments = assignments
    ? Object.fromEntries(Object.entries(assignments).filter(([, strats]) => strats.length > 0))
    : {};
  const hasValidAssignments = Object.keys(validAssignments).length > 0;

  const handleStart = async () => {
    if (!hasValidAssignments) return;
    if (!confirm("Start live trading?")) return;
    setError("");
    setLoading("start");
    try {
      await api.startTrading(validAssignments);
      store.setTrading(true);
    } catch (e: any) {
      setError(e.message || "Failed to start");
    } finally {
      setLoading("");
    }
  };

  const handleStop = async () => {
    if (!confirm("Stop live trading?")) return;
    setLoading("stop");
    try {
      await api.stopTrading();
      store.setTrading(false);
    } catch {} finally {
      setLoading("");
    }
  };

  const handleKill = async () => {
    if (!confirm("ACTIVATE KILL SWITCH?\n\nThis will immediately halt ALL trading and prevent new trades until manually reset.")) return;
    setLoading("kill");
    try {
      await api.killSwitch();
      store.setTrading(false);
      store.setKillSwitch(true);
    } catch {} finally {
      setLoading("");
    }
  };

  const handleResetKill = async () => {
    if (!confirm("Deactivate kill switch?\n\nThis will allow trading to resume.")) return;
    setLoading("resetkill");
    try {
      await api.resetKillSwitch();
      store.setKillSwitch(false);
    } catch (e: any) {
      setError(e.message || "Failed to reset");
    } finally {
      setLoading("");
    }
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
              {loading === "connect" ? "Connecting..." : "Connect"}
            </button>
          ) : (
            <button
              onClick={handleDisconnect}
              disabled={!!loading || store.isTrading}
              className="rounded bg-zinc-700 px-3 py-1 text-xs font-medium hover:bg-zinc-600 disabled:opacity-50"
              title={store.isTrading ? "Stop trading first" : undefined}
            >
              {loading === "disconnect" ? "..." : "Disconnect"}
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
              disabled={!store.mt5Connected || !!loading || store.killSwitch || !hasValidAssignments}
              className="rounded bg-emerald-600 px-3 py-1 text-xs font-medium hover:bg-emerald-500 disabled:opacity-50"
              title={
                store.killSwitch
                  ? "Reset kill switch first"
                  : !hasValidAssignments
                    ? "Select at least one symbol with a strategy"
                    : undefined
              }
            >
              {loading === "start" ? "Starting..." : "Start"}
            </button>
          ) : (
            <button
              onClick={handleStop}
              disabled={!!loading}
              className="rounded bg-zinc-700 px-3 py-1 text-xs font-medium hover:bg-zinc-600 disabled:opacity-50"
            >
              {loading === "stop" ? "..." : "Stop"}
            </button>
          )}
          {!store.killSwitch ? (
            <button
              onClick={handleKill}
              disabled={!store.mt5Connected || !!loading}
              className="rounded bg-red-700 px-3 py-1 text-xs font-medium hover:bg-red-600 disabled:opacity-50"
            >
              {loading === "kill" ? "..." : "Kill"}
            </button>
          ) : (
            <button
              onClick={handleResetKill}
              disabled={!!loading}
              className="animate-pulse rounded bg-amber-600 px-3 py-1 text-xs font-medium hover:bg-amber-500 disabled:opacity-50"
            >
              {loading === "resetkill" ? "..." : "Reset Kill Switch"}
            </button>
          )}
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
