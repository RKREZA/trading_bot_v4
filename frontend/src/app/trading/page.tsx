"use client";
import { useEffect, useState, useCallback, useRef } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useAppStore } from "@/lib/store";
import { api } from "@/lib/api";
import { ConnectionControls } from "./_components/ConnectionControls";
import { SelectionPanel } from "./_components/SelectionPanel";
import { ChartContainer } from "./_components/ChartContainer";
import { PositionsTable } from "./_components/PositionsTable";

type Assignments = Record<string, string[]>;
type PairOptions = Record<string, Record<string, any>>;

export default function TradingPage() {
  const store = useAppStore();
  const [symbols, setSymbols] = useState<string[]>([]);
  const [assignments, setAssignments] = useState<Assignments>({});
  const [pairOptions, setPairOptions] = useState<PairOptions>({});
  const [primarySymbol, setPrimarySymbol] = useState("");
  const [positions, setPositions] = useState<any[]>([]);
  const [account, setAccount] = useState<any>(null);
  const [strategies, setStrategies] = useState<any[]>([]);
  const [configLoaded, setConfigLoaded] = useState(false);

  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const selectedSymbols = Object.keys(assignments);

  const loadSymbols = useCallback(() => {
    api.getSymbols().then((s) => {
      setSymbols(s);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    api.getTradingConfig().then((cfg) => {
      if (cfg.assignments && Object.keys(cfg.assignments).length > 0) {
        setAssignments(cfg.assignments);
      }
      if (cfg.primarySymbol) {
        setPrimarySymbol(cfg.primarySymbol);
      }
      if (cfg.pairOptions) {
        setPairOptions(cfg.pairOptions);
      }
      setConfigLoaded(true);
    }).catch(() => {
      setConfigLoaded(true);
    });
    loadSymbols();
  }, []);

  useEffect(() => {
    const poll = () => {
      api.getPositions().then(setPositions).catch(() => {});
      api.getAccount().then(setAccount).catch(() => {});
      api.getStrategies().then(setStrategies).catch(() => {});
    };
    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, []);

  const saveConfig = useCallback((a: Assignments, ps: string, po: PairOptions) => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      api.saveTradingConfig({ assignments: a, primarySymbol: ps, pairOptions: po }).catch(() => {});
    }, 500);
  }, []);

  const handleAssignmentsChange = (a: Assignments) => {
    setAssignments(a);
    saveConfig(a, primarySymbol, pairOptions);
  };

  const handlePairOptionsChange = (po: PairOptions) => {
    setPairOptions(po);
    saveConfig(assignments, primarySymbol, po);
  };

  const handleSetPrimary = (sym: string) => {
    setPrimarySymbol(sym);
    saveConfig(assignments, sym, pairOptions);
  };

  const handleToggleStrategy = async (id: string) => {
    try {
      await api.toggleStrategy(id);
      const updated = await api.getStrategies();
      setStrategies(updated);
    } catch {}
  };

  const handleModifySLTP = async (ticket: number, symbol: string, sl: number, tp: number) => {
    try {
      await api.modifyPosition(ticket, sl, tp, symbol);
      api.getPositions().then(setPositions).catch(() => {});
    } catch {}
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Trading</h2>
      </div>

      <ConnectionControls onConnected={loadSymbols} assignments={assignments} />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {/* MT5 Account */}
        <Card>
          <CardHeader><CardTitle>MT5 Account</CardTitle></CardHeader>
          {account?.connected ? (
            <div className="space-y-1.5 text-sm">
              <div className="flex justify-between"><span className="text-zinc-400">Login</span><span className="font-mono">{account.login}</span></div>
              <div className="flex justify-between"><span className="text-zinc-400">Server</span><span>{account.server}</span></div>
              <div className="flex justify-between"><span className="text-zinc-400">Name</span><span>{account.name}</span></div>
              <div className="flex justify-between"><span className="text-zinc-400">Leverage</span><span>1:{account.leverage}</span></div>
              <div className="mt-2 border-t border-zinc-800 pt-2">
                <div className="flex justify-between"><span className="text-zinc-400">Balance</span><span className="font-mono text-white">{account.balance?.toFixed(2)} {account.currency}</span></div>
                <div className="flex justify-between"><span className="text-zinc-400">Equity</span><span className="font-mono text-white">{account.equity?.toFixed(2)}</span></div>
                <div className="flex justify-between"><span className="text-zinc-400">Margin</span><span className="font-mono">{account.margin?.toFixed(2)}</span></div>
                <div className="flex justify-between"><span className="text-zinc-400">Free Margin</span><span className="font-mono">{account.free_margin?.toFixed(2)}</span></div>
                <div className="flex justify-between"><span className="text-zinc-400">Profit</span><span className={`font-mono ${(account.profit ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}>{account.profit?.toFixed(2)}</span></div>
                {account.margin_level > 0 && <div className="flex justify-between"><span className="text-zinc-400">Margin Level</span><span className="font-mono">{account.margin_level?.toFixed(1)}%</span></div>}
              </div>
            </div>
          ) : (
            <div className="text-sm text-zinc-500">
              <Badge variant="danger">Disconnected</Badge>
              <p className="mt-2">Connect MT5 to see account data</p>
            </div>
          )}
        </Card>

        {/* Selection Panel */}
        <SelectionPanel
          strategies={strategies}
          symbols={symbols}
          assignments={assignments}
          onAssignmentsChange={handleAssignmentsChange}
          pairOptions={pairOptions}
          onPairOptionsChange={handlePairOptionsChange}
          primarySymbol={primarySymbol}
          onSetPrimary={handleSetPrimary}
        />

        {/* Status */}
        <Card>
          <CardHeader><CardTitle>Status</CardTitle></CardHeader>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-zinc-400">MT5</span>
              <Badge variant={account?.connected ? "success" : "danger"}>
                {account?.connected ? "Connected" : "Disconnected"}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">Trading</span>
              <Badge variant={store.isTrading ? "success" : "default"}>
                {store.isTrading ? "Running" : "Stopped"}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">Kill Switch</span>
              <Badge variant={store.killSwitch ? "danger" : "default"}>
                {store.killSwitch ? "ACTIVE" : "Off"}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">Strategies</span>
              <span className="font-mono">{strategies.filter((s) => s.enabled !== false).length}/{strategies.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">Positions</span>
              <span className="font-mono">{positions.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-400">Symbols</span>
              <span className="font-mono">{selectedSymbols.length}</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Chart */}
      <ChartContainer
        symbol={primarySymbol}
        symbols={symbols}
        onSymbolChange={handleSetPrimary}
        positions={positions}
        onModifySLTP={handleModifySLTP}
      />

      {/* Positions */}
      <PositionsTable
        positions={positions}
        onRefresh={() => api.getPositions().then(setPositions).catch(() => {})}
      />
    </div>
  );
}
