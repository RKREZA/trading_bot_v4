const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export const api = {
  getStatus: () => request<any>("/api/status"),
  getHealth: () => request<any>("/api/health"),

  startTrading: () => request<any>("/api/control/start", { method: "POST" }),
  stopTrading: () => request<any>("/api/control/stop", { method: "POST" }),
  killSwitch: () => request<any>("/api/control/kill", { method: "POST" }),

  connectMT5: () => request<any>("/api/mt5/connect", { method: "POST" }),
  disconnectMT5: () => request<any>("/api/mt5/disconnect", { method: "POST" }),
  closePosition: (ticket: number) =>
    request<any>(`/api/positions/${ticket}/close`, { method: "POST" }),
  modifyPosition: (ticket: number, sl: number, tp: number, symbol: string) =>
    request<any>(`/api/positions/${ticket}/modify`, {
      method: "PUT",
      body: JSON.stringify({ sl, tp, symbol }),
    }),

  getPositions: () => request<any[]>("/api/positions"),
  getAccount: () => request<any>("/api/account"),
  getTrades: (limit = 50) => request<any>(`/api/trades?limit=${limit}`),

  getStrategies: () => request<any[]>("/api/strategies/"),
  toggleStrategy: (id: string) =>
    request<any>(`/api/strategies/${id}/toggle`, { method: "POST" }),
  updateStrategyParams: (id: string, params: Record<string, any>) =>
    request<any>(`/api/strategies/${id}/params`, {
      method: "PUT",
      body: JSON.stringify(params),
    }),

  getConfig: () => request<any>("/api/config/"),
  updateConfig: (updates: Record<string, any>) =>
    request<any>("/api/config/", {
      method: "POST",
      body: JSON.stringify(updates),
    }),

  getSymbols: () => request<string[]>("/api/data/symbols"),
  getCandleData: (symbol: string, tf: string, count = 200) =>
    request<any>(`/api/data/${symbol}/${tf}?count=${count}`),
  triggerSync: (symbol: string, tf: string) =>
    request<any>("/api/data/sync", {
      method: "POST",
      body: JSON.stringify({ symbol, timeframe: tf }),
    }),
  getSyncStatus: () => request<any>("/api/data/sync/status"),

  getAccounts: () => request<any[]>("/api/accounts/"),
  createAccount: (data: any) =>
    request<any>("/api/accounts/", { method: "POST", body: JSON.stringify(data) }),
  deleteAccount: (id: number) =>
    request<any>(`/api/accounts/${id}`, { method: "DELETE" }),

  getLogs: (limit = 100, offset = 0) =>
    request<any>(`/api/logs/?limit=${limit}&offset=${offset}`),

  runBacktest: (config: any) =>
    request<any>("/api/backtest/run", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  getBacktestRuns: (limit = 20) =>
    request<any[]>(`/api/backtest/runs?limit=${limit}`),
  getBacktestStatus: (id: number) => request<any>(`/api/backtest/${id}/status`),
  getBacktestResults: (id: number) => request<any>(`/api/backtest/${id}/results`),
};
