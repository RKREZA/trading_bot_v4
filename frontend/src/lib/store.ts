"use client";
import { create } from "zustand";

interface AccountState {
  balance: number;
  equity: number;
  margin: number;
  freeMargin: number;
  drawdownPct: number;
  profit: number;
}

interface AppStore {
  isTrading: boolean;
  mt5Connected: boolean;
  killSwitch: boolean;
  account: AccountState;
  strategies: any[];
  positions: any[];

  setTrading: (v: boolean) => void;
  setMt5Connected: (v: boolean) => void;
  setKillSwitch: (v: boolean) => void;
  setAccount: (a: Partial<AccountState>) => void;
  setStrategies: (s: any[]) => void;
  setPositions: (p: any[]) => void;
  updateFromMetrics: (data: any) => void;
}

export const useAppStore = create<AppStore>((set) => ({
  isTrading: false,
  mt5Connected: false,
  killSwitch: false,
  account: {
    balance: 0,
    equity: 0,
    margin: 0,
    freeMargin: 0,
    drawdownPct: 0,
    profit: 0,
  },
  strategies: [],
  positions: [],

  setTrading: (v) => set({ isTrading: v }),
  setMt5Connected: (v) => set({ mt5Connected: v }),
  setKillSwitch: (v) => set({ killSwitch: v }),
  setAccount: (a) =>
    set((state) => ({ account: { ...state.account, ...a } })),
  setStrategies: (s) => set({ strategies: s }),
  setPositions: (p) => set({ positions: p }),
  updateFromMetrics: (data) =>
    set((state) => ({
      isTrading: data.is_trading ?? state.isTrading,
      account: {
        ...state.account,
        balance: data.account?.balance ?? state.account.balance,
        equity: data.account?.equity ?? state.account.equity,
        margin: data.account?.margin ?? state.account.margin,
        freeMargin: data.account?.free_margin ?? state.account.freeMargin,
        drawdownPct: data.account?.drawdown ?? state.account.drawdownPct,
        profit: data.account?.profit ?? state.account.profit,
      },
    })),
}));
