"use client";
import { useState } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

interface Props {
  positions: any[];
  onRefresh: () => void;
}

export function PositionsTable({ positions, onRefresh }: Props) {
  const [editingTicket, setEditingTicket] = useState<number | null>(null);
  const [editSL, setEditSL] = useState("");
  const [editTP, setEditTP] = useState("");
  const [loadingTicket, setLoadingTicket] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const startEdit = (pos: any) => {
    setEditingTicket(pos.ticket);
    setEditSL(pos.sl?.toString() ?? "0");
    setEditTP(pos.tp?.toString() ?? "0");
    setError(null);
  };

  const handleModify = async (ticket: number, symbol: string) => {
    const newSL = parseFloat(editSL);
    const newTP = parseFloat(editTP);
    if (!confirm(`Modify position #${ticket}?\n\nNew SL: ${newSL.toFixed(5)}\nNew TP: ${newTP.toFixed(5)}`)) return;
    setLoadingTicket(ticket);
    setError(null);
    try {
      await api.modifyPosition(ticket, newSL, newTP, symbol);
      setEditingTicket(null);
      onRefresh();
    } catch (e: any) {
      setError(`#${ticket}: ${e.message}`);
    } finally {
      setLoadingTicket(null);
    }
  };

  const handleClose = async (ticket: number) => {
    if (!confirm(`Close position #${ticket}?`)) return;
    setLoadingTicket(ticket);
    setError(null);
    try {
      await api.closePosition(ticket);
      onRefresh();
    } catch (e: any) {
      setError(`#${ticket}: ${e.message}`);
    } finally {
      setLoadingTicket(null);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Open Positions ({positions.length})</CardTitle>
      </CardHeader>

      {error && (
        <div className="mb-3 rounded bg-red-900/40 border border-red-800 px-3 py-1.5 text-xs text-red-300">
          {error}
          <button onClick={() => setError(null)} className="ml-2 text-red-500 hover:text-red-400">×</button>
        </div>
      )}

      {positions.length === 0 ? (
        <p className="text-sm text-zinc-500">No open positions</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className="pb-2 text-left">Ticket</th>
                <th className="pb-2 text-left">Symbol</th>
                <th className="pb-2 text-left">Type</th>
                <th className="pb-2 text-right">Volume</th>
                <th className="pb-2 text-right">Open Price</th>
                <th className="pb-2 text-right">SL</th>
                <th className="pb-2 text-right">TP</th>
                <th className="pb-2 text-right">Profit</th>
                <th className="pb-2 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p: any) => {
                const isEditing = editingTicket === p.ticket;
                const isLoading = loadingTicket === p.ticket;
                return (
                  <tr key={p.ticket} className="border-b border-zinc-800/50">
                    <td className="py-2 font-mono text-xs">{p.ticket}</td>
                    <td>{p.symbol}</td>
                    <td>
                      <Badge variant={p.type === 0 ? "success" : "danger"}>
                        {p.type === 0 ? "BUY" : "SELL"}
                      </Badge>
                    </td>
                    <td className="text-right">{p.volume}</td>
                    <td className="text-right font-mono">{p.price_open?.toFixed(5)}</td>

                    {/* SL */}
                    <td className="text-right font-mono">
                      {isEditing ? (
                        <input
                          type="number"
                          step="any"
                          value={editSL}
                          onChange={(e) => setEditSL(e.target.value)}
                          className="w-24 rounded bg-zinc-800 px-1.5 py-0.5 text-right text-xs"
                        />
                      ) : (
                        <span className="text-red-400">{p.sl?.toFixed(5) ?? "—"}</span>
                      )}
                    </td>

                    {/* TP */}
                    <td className="text-right font-mono">
                      {isEditing ? (
                        <input
                          type="number"
                          step="any"
                          value={editTP}
                          onChange={(e) => setEditTP(e.target.value)}
                          className="w-24 rounded bg-zinc-800 px-1.5 py-0.5 text-right text-xs"
                        />
                      ) : (
                        <span className="text-emerald-400">{p.tp?.toFixed(5) ?? "—"}</span>
                      )}
                    </td>

                    <td
                      className={`text-right font-mono ${
                        p.profit >= 0 ? "text-emerald-400" : "text-red-400"
                      }`}
                    >
                      {p.profit?.toFixed(2)}
                    </td>

                    {/* Actions */}
                    <td className="text-right">
                      <div className="flex justify-end gap-1">
                        {isEditing ? (
                          <>
                            <button
                              onClick={() => handleModify(p.ticket, p.symbol)}
                              disabled={isLoading}
                              className="rounded bg-blue-700 px-2 py-0.5 text-xs text-blue-100 hover:bg-blue-600 disabled:opacity-50"
                            >
                              {isLoading ? "…" : "Save"}
                            </button>
                            <button
                              onClick={() => setEditingTicket(null)}
                              className="rounded bg-zinc-700 px-2 py-0.5 text-xs hover:bg-zinc-600"
                            >
                              Cancel
                            </button>
                          </>
                        ) : (
                          <>
                            <button
                              onClick={() => startEdit(p)}
                              className="rounded bg-zinc-700 px-2 py-0.5 text-xs hover:bg-zinc-600"
                            >
                              Modify
                            </button>
                            <button
                              onClick={() => handleClose(p.ticket)}
                              disabled={isLoading}
                              className="rounded bg-red-900/50 px-2 py-0.5 text-xs text-red-400 hover:bg-red-900 disabled:opacity-50"
                            >
                              {isLoading ? "…" : "Close"}
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
