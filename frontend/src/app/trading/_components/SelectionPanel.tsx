"use client";
import { useState } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Assignments = Record<string, string[]>;
type PairOptions = Record<string, Record<string, any>>;

interface TrailingStopOpts {
  enabled: boolean;
  phase2_rr_threshold: number;
  phase2_be_offset_pct: number;
  phase3_trail_mult: number;
}

interface PartialProfitOpts {
  enabled: boolean;
  phase1_rr_target: number;
  phase1_close_pct: number;
  move_to_be_at_partial: boolean;
}

const DEFAULT_TRAILING: TrailingStopOpts = {
  enabled: true,
  phase2_rr_threshold: 2.0,
  phase2_be_offset_pct: 0.2,
  phase3_trail_mult: 1.5,
};

const DEFAULT_PARTIAL: PartialProfitOpts = {
  enabled: true,
  phase1_rr_target: 1.0,
  phase1_close_pct: 50,
  move_to_be_at_partial: true,
};

interface Props {
  strategies: any[];
  symbols: string[];
  assignments: Assignments;
  onAssignmentsChange: (a: Assignments) => void;
  pairOptions: PairOptions;
  onPairOptionsChange: (po: PairOptions) => void;
  primarySymbol: string;
  onSetPrimary: (s: string) => void;
}

function getPairKey(sym: string, stratId: string) {
  return `${sym}:${stratId}`;
}

function getOpts(po: PairOptions, key: string) {
  return po[key] || {};
}

function getTrailing(po: PairOptions, key: string): TrailingStopOpts {
  const raw = getOpts(po, key).trailing_stop || {};
  return { ...DEFAULT_TRAILING, ...raw };
}

function getPartial(po: PairOptions, key: string): PartialProfitOpts {
  const raw = getOpts(po, key).partial_profit || {};
  return { ...DEFAULT_PARTIAL, ...raw };
}

function Toggle({ value, onChange, label }: { value: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className={`rounded px-2 py-0.5 text-xs font-medium border transition-colors ${
        value
          ? "border-emerald-700 bg-emerald-900/40 text-emerald-400"
          : "border-zinc-700 bg-zinc-800 text-zinc-500"
      }`}
    >
      {label}: {value ? "ON" : "OFF"}
    </button>
  );
}

function NumInput({ value, onChange, label, step = 0.1 }: {
  value: number; onChange: (v: number) => void; label: string; step?: number;
}) {
  return (
    <div>
      <label className="mb-0.5 block text-xs text-zinc-500">{label}</label>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="w-full rounded bg-zinc-800 px-2 py-1 text-xs font-mono text-white"
      />
    </div>
  );
}

export function SelectionPanel({
  strategies,
  symbols,
  assignments,
  onAssignmentsChange,
  pairOptions,
  onPairOptionsChange,
  primarySymbol,
  onSetPrimary,
}: Props) {
  const [addSymbol, setAddSymbol] = useState("");
  const [expandedPair, setExpandedPair] = useState<string | null>(null);

  const assignedSymbols = Object.keys(assignments);
  const availableSymbols = symbols.filter((s) => !assignedSymbols.includes(s));

  const handleAddSymbol = () => {
    const sym = addSymbol || availableSymbols[0];
    if (!sym || assignments[sym]) return;
    const stratIds = strategies.map((s) => s.strategy_id);
    const next = { ...assignments, [sym]: stratIds.length === 1 ? [stratIds[0]] : [] };
    onAssignmentsChange(next);
    setAddSymbol("");
    if (!primarySymbol) onSetPrimary(sym);
  };

  const handleRemoveSymbol = (sym: string) => {
    const next = { ...assignments };
    delete next[sym];
    onAssignmentsChange(next);

    const nextPo = { ...pairOptions };
    for (const key of Object.keys(nextPo)) {
      if (key.startsWith(`${sym}:`)) delete nextPo[key];
    }
    onPairOptionsChange(nextPo);

    if (primarySymbol === sym) {
      const remaining = Object.keys(next);
      onSetPrimary(remaining[0] || "");
    }
  };

  const handleToggleStrategy = (sym: string, stratId: string) => {
    const current = assignments[sym] || [];
    const next = current.includes(stratId)
      ? current.filter((s) => s !== stratId)
      : [...current, stratId];
    onAssignmentsChange({ ...assignments, [sym]: next });
  };

  const updatePairOpts = (key: string, block: string, updates: Record<string, any>) => {
    const existing = getOpts(pairOptions, key);
    const existingBlock = existing[block] || {};
    onPairOptionsChange({
      ...pairOptions,
      [key]: {
        ...existing,
        [block]: { ...existingBlock, ...updates },
      },
    });
  };

  const totalPairs = assignedSymbols.length;
  const validPairs = assignedSymbols.filter((s) => (assignments[s]?.length ?? 0) > 0).length;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Strategy Assignments</CardTitle>
        {totalPairs > 0 && (
          <span className="text-xs text-zinc-500">
            {validPairs}/{totalPairs} configured
          </span>
        )}
      </CardHeader>
      <div className="space-y-3">
        {/* Add symbol */}
        <div className="flex gap-2">
          <select
            value={addSymbol}
            onChange={(e) => setAddSymbol(e.target.value)}
            className="flex-1 rounded bg-zinc-800 px-2 py-1.5 text-xs text-white"
            disabled={availableSymbols.length === 0}
          >
            {availableSymbols.length === 0 ? (
              <option>No symbols available</option>
            ) : (
              <>
                <option value="">Select symbol...</option>
                {availableSymbols.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </>
            )}
          </select>
          <button
            onClick={handleAddSymbol}
            disabled={availableSymbols.length === 0}
            className="rounded bg-emerald-600 px-3 py-1.5 text-xs font-medium hover:bg-emerald-500 disabled:opacity-50"
          >
            Add
          </button>
        </div>

        {/* Assignment rows */}
        {assignedSymbols.length === 0 ? (
          <p className="text-xs text-zinc-600">Add symbols and assign strategies to begin trading</p>
        ) : (
          <div className="space-y-2">
            {assignedSymbols.map((sym) => {
              const isPrimary = sym === primarySymbol;
              const strats = assignments[sym] || [];
              const hasStrategies = strats.length > 0;
              return (
                <div
                  key={sym}
                  className={`rounded-lg border px-3 py-2 ${
                    isPrimary
                      ? "border-blue-700 bg-blue-950/30"
                      : hasStrategies
                        ? "border-zinc-700 bg-zinc-900"
                        : "border-amber-800/50 bg-amber-950/20"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-medium text-white">{sym}</span>
                      {isPrimary && <Badge variant="default" className="text-blue-400">chart</Badge>}
                      {!hasStrategies && <span className="text-xs text-amber-400">no strategy</span>}
                    </div>
                    <div className="flex items-center gap-1">
                      {!isPrimary && (
                        <button
                          onClick={() => onSetPrimary(sym)}
                          className="rounded px-1.5 py-0.5 text-xs text-zinc-500 hover:text-blue-400 hover:bg-zinc-800"
                          title="Set as chart symbol"
                        >
                          chart
                        </button>
                      )}
                      <button
                        onClick={() => handleRemoveSymbol(sym)}
                        className="rounded px-1.5 py-0.5 text-xs text-zinc-500 hover:text-red-400 hover:bg-zinc-800"
                      >
                        remove
                      </button>
                    </div>
                  </div>

                  {/* Strategy buttons + settings toggle */}
                  <div className="flex flex-wrap gap-1">
                    {strategies.map((s) => {
                      const active = strats.includes(s.strategy_id);
                      const pairKey = getPairKey(sym, s.strategy_id);
                      const isExpanded = expandedPair === pairKey;
                      return (
                        <div key={s.strategy_id} className="flex flex-col">
                          <div className="flex items-center gap-0.5">
                            <button
                              onClick={() => handleToggleStrategy(sym, s.strategy_id)}
                              className={`rounded-l-full border px-2 py-0.5 text-xs transition-colors ${
                                active
                                  ? "border-emerald-700 bg-emerald-900/40 text-emerald-400"
                                  : "border-zinc-700 bg-zinc-800 text-zinc-500 hover:text-zinc-300"
                              }`}
                            >
                              {s.strategy_id}
                            </button>
                            {active && (
                              <button
                                onClick={() => setExpandedPair(isExpanded ? null : pairKey)}
                                className={`rounded-r-full border border-l-0 px-1.5 py-0.5 text-xs transition-colors ${
                                  isExpanded
                                    ? "border-blue-600 bg-blue-900/40 text-blue-400"
                                    : "border-zinc-700 bg-zinc-800 text-zinc-500 hover:text-zinc-300"
                                }`}
                                title="Configure trailing stop & partial profit"
                              >
                                &#9881;
                              </button>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Expanded pair options */}
                  {strats.map((stratId) => {
                    const pairKey = getPairKey(sym, stratId);
                    if (expandedPair !== pairKey) return null;
                    const ts = getTrailing(pairOptions, pairKey);
                    const pp = getPartial(pairOptions, pairKey);
                    return (
                      <div key={pairKey} className="mt-2 rounded border border-zinc-700 bg-zinc-900/50 p-2 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium text-zinc-300">
                            {sym} &rarr; {stratId}
                          </span>
                          <button
                            onClick={() => setExpandedPair(null)}
                            className="text-xs text-zinc-500 hover:text-zinc-300"
                          >
                            close
                          </button>
                        </div>

                        {/* Trailing Stop */}
                        <div className="space-y-1.5">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-medium text-zinc-400">Trailing Stop</span>
                            <Toggle
                              value={ts.enabled}
                              label="Trailing"
                              onChange={(v) => updatePairOpts(pairKey, "trailing_stop", { enabled: v })}
                            />
                          </div>
                          {ts.enabled && (
                            <div className="grid grid-cols-3 gap-2">
                              <NumInput
                                label="BE at R:R"
                                value={ts.phase2_rr_threshold}
                                onChange={(v) => updatePairOpts(pairKey, "trailing_stop", { phase2_rr_threshold: v })}
                              />
                              <NumInput
                                label="BE Offset %"
                                value={ts.phase2_be_offset_pct}
                                onChange={(v) => updatePairOpts(pairKey, "trailing_stop", { phase2_be_offset_pct: v })}
                              />
                              <NumInput
                                label="Trail ATR Mult"
                                value={ts.phase3_trail_mult}
                                onChange={(v) => updatePairOpts(pairKey, "trailing_stop", { phase3_trail_mult: v })}
                              />
                            </div>
                          )}
                        </div>

                        {/* Partial Profit */}
                        <div className="space-y-1.5">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-medium text-zinc-400">Partial Profit</span>
                            <Toggle
                              value={pp.enabled}
                              label="Partial"
                              onChange={(v) => updatePairOpts(pairKey, "partial_profit", { enabled: v })}
                            />
                          </div>
                          {pp.enabled && (
                            <div className="grid grid-cols-3 gap-2">
                              <NumInput
                                label="R:R Target"
                                value={pp.phase1_rr_target}
                                onChange={(v) => updatePairOpts(pairKey, "partial_profit", { phase1_rr_target: v })}
                              />
                              <NumInput
                                label="Close %"
                                value={pp.phase1_close_pct}
                                step={5}
                                onChange={(v) => updatePairOpts(pairKey, "partial_profit", { phase1_close_pct: v })}
                              />
                              <div>
                                <label className="mb-0.5 block text-xs text-zinc-500">Move to BE</label>
                                <Toggle
                                  value={pp.move_to_be_at_partial}
                                  label="BE"
                                  onChange={(v) => updatePairOpts(pairKey, "partial_profit", { move_to_be_at_partial: v })}
                                />
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        )}

        {/* Summary */}
        {validPairs > 0 && (
          <div className="border-t border-zinc-800 pt-2">
            <div className="flex flex-wrap gap-1">
              {assignedSymbols
                .filter((s) => (assignments[s]?.length ?? 0) > 0)
                .map((s) => (
                  <Badge key={s} variant="success" className="font-mono">
                    {s} ({assignments[s].length})
                  </Badge>
                ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
