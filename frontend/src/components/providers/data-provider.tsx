"use client";
import { useEffect } from "react";
import { useAppStore } from "@/lib/store";
import { api } from "@/lib/api";

export function DataProvider({ children }: { children: React.ReactNode }) {
  const store = useAppStore();

  useEffect(() => {
    const poll = async () => {
      try {
        const acc = await api.getAccount();
        store.setMt5Connected(acc.connected ?? false);
        if (acc.connected) {
          store.setAccount({
            balance: acc.balance ?? 0,
            equity: acc.equity ?? 0,
            margin: acc.margin ?? 0,
            freeMargin: acc.free_margin ?? 0,
            profit: acc.profit ?? 0,
            drawdownPct:
              acc.equity > 0 && acc.balance > 0
                ? Math.max(0, ((acc.balance - acc.equity) / acc.balance) * 100)
                : 0,
          });
        }
      } catch {}

      try {
        const status = await api.getStatus();
        store.setTrading(status.is_trading ?? false);
        if (status.kill_switch) store.setKillSwitch(true);
      } catch {}
    };

    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, []);

  return <>{children}</>;
}
