"use client";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface Props {
  strategies: any[];
  symbols: string[];
  selectedSymbols: string[];
  onToggleSymbol: (symbol: string) => void;
  onToggleStrategy: (id: string) => void;
  primarySymbol: string;
  onSetPrimary: (symbol: string) => void;
}

export function SelectionPanel({
  strategies,
  symbols,
  selectedSymbols,
  onToggleSymbol,
  onToggleStrategy,
  primarySymbol,
  onSetPrimary,
}: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Strategies & Symbols</CardTitle>
      </CardHeader>
      <div className="space-y-4">
        {/* Strategies */}
        <div>
          <label className="mb-1.5 block text-xs font-medium text-zinc-500">Strategies</label>
          {strategies.length === 0 ? (
            <p className="text-xs text-zinc-600">No strategies loaded</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {strategies.map((s: any) => {
                const active = s.enabled !== false;
                return (
                  <button
                    key={s.strategy_id}
                    onClick={() => onToggleStrategy(s.strategy_id)}
                    className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${
                      active
                        ? "border-emerald-700 bg-emerald-900/40 text-emerald-400"
                        : "border-zinc-700 bg-zinc-800 text-zinc-500 hover:text-zinc-300"
                    }`}
                  >
                    {s.strategy_id}
                    {s.class && <span className="ml-1 text-zinc-600">{s.class}</span>}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Symbols */}
        <div>
          <label className="mb-1.5 block text-xs font-medium text-zinc-500">
            Symbols
            {primarySymbol && (
              <span className="ml-2 text-zinc-600">chart: {primarySymbol}</span>
            )}
          </label>
          {symbols.length === 0 ? (
            <p className="text-xs text-zinc-600">Connect MT5 to load symbols</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {symbols.map((sym) => {
                const selected = selectedSymbols.includes(sym);
                const isPrimary = sym === primarySymbol;
                return (
                  <button
                    key={sym}
                    onClick={() => onToggleSymbol(sym)}
                    onDoubleClick={() => onSetPrimary(sym)}
                    title="Click to select, double-click for chart"
                    className={`rounded-full border px-2.5 py-1 text-xs font-mono transition-colors ${
                      isPrimary
                        ? "border-blue-600 bg-blue-900/40 text-blue-400"
                        : selected
                          ? "border-emerald-700 bg-emerald-900/40 text-emerald-400"
                          : "border-zinc-700 bg-zinc-800 text-zinc-500 hover:text-zinc-300"
                    }`}
                  >
                    {sym}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Active selections summary */}
        {selectedSymbols.length > 0 && (
          <div className="border-t border-zinc-800 pt-2">
            <span className="text-xs text-zinc-500">Active: </span>
            {selectedSymbols.map((s) => (
              <Badge key={s} variant="success" className="mr-1 mb-1">{s}</Badge>
            ))}
          </div>
        )}
      </div>
    </Card>
  );
}
