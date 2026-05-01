"use client";
import { useAppStore } from "@/lib/store";
import { Badge } from "@/components/ui/badge";
import { formatCurrency } from "@/lib/utils";

export function Header() {
  const { isTrading, mt5Connected, killSwitch, account } = useAppStore();

  return (
    <header className="flex min-h-[3.5rem] flex-wrap items-center justify-between gap-2 border-b border-zinc-800 bg-zinc-950 px-4 py-2 pl-12 lg:pl-6">
      {/* Status badges */}
      <div className="flex items-center gap-2">
        <Badge variant={mt5Connected ? "success" : "danger"}>
          <span className="hidden sm:inline">MT5 </span>
          {mt5Connected ? "Connected" : "Offline"}
        </Badge>
        <Badge variant={isTrading ? "success" : "default"}>
          {isTrading ? "Trading" : "Idle"}
        </Badge>
        {killSwitch && <Badge variant="danger">KILL</Badge>}
      </div>

      {/* Account data */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs sm:text-sm text-zinc-400">
        <span>
          Bal: <span className="text-white">{formatCurrency(account.balance)}</span>
        </span>
        <span>
          Eq: <span className="text-white">{formatCurrency(account.equity)}</span>
        </span>
        <span className="hidden md:inline">
          Margin: <span className="text-white">{formatCurrency(account.margin)}</span>
        </span>
        <span>
          P/L:{" "}
          <span className={account.profit >= 0 ? "text-emerald-400" : "text-red-400"}>
            {formatCurrency(account.profit)}
          </span>
        </span>
        <span>
          DD:{" "}
          <span className={account.drawdownPct > 5 ? "text-red-400" : "text-white"}>
            {account.drawdownPct.toFixed(2)}%
          </span>
        </span>
      </div>
    </header>
  );
}
