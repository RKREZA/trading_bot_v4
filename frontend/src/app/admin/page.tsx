"use client";
import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

interface AccountForm {
  login: string;
  password: string;
  server: string;
  broker_utc_offset: string;
  label: string;
  max_drawdown_pct: string;
  risk_per_trade_pct: string;
  max_daily_drawdown_pct: string;
  is_active: boolean;
}

const EMPTY_FORM: AccountForm = {
  login: "",
  password: "",
  server: "",
  broker_utc_offset: "0",
  label: "",
  max_drawdown_pct: "10",
  risk_per_trade_pct: "1",
  max_daily_drawdown_pct: "5",
  is_active: true,
};

export default function AdminPage() {
  const [accounts, setAccounts] = useState<any[]>([]);
  const [form, setForm] = useState<AccountForm>({ ...EMPTY_FORM });
  const [editingId, setEditingId] = useState<number | null>(null);
  const [error, setError] = useState("");

  const load = () => api.getAccounts().then(setAccounts).catch(() => {});

  useEffect(() => { load(); }, []);

  const set = (field: keyof AccountForm, value: string | boolean) =>
    setForm((f) => ({ ...f, [field]: value }));

  const handleCreate = async () => {
    if (!form.login || !form.server) return;
    setError("");
    try {
      await api.createAccount({
        login: parseInt(form.login),
        password: form.password,
        server: form.server,
        broker_utc_offset: parseInt(form.broker_utc_offset) || 0,
        label: form.label,
        max_drawdown_pct: parseFloat(form.max_drawdown_pct) || 10,
        risk_per_trade_pct: parseFloat(form.risk_per_trade_pct) || 1,
        max_daily_drawdown_pct: parseFloat(form.max_daily_drawdown_pct) || 5,
      });
      setForm({ ...EMPTY_FORM });
      load();
    } catch (e: any) {
      setError(e.message || "Failed to create");
    }
  };

  const handleEdit = (account: any) => {
    setEditingId(account.id);
    setForm({
      login: String(account.login),
      password: "",
      server: account.server,
      broker_utc_offset: String(account.broker_utc_offset ?? 0),
      label: account.label || "",
      max_drawdown_pct: String(account.max_drawdown_pct ?? 10),
      risk_per_trade_pct: String(account.risk_per_trade_pct ?? 1),
      max_daily_drawdown_pct: String(account.max_daily_drawdown_pct ?? 5),
      is_active: account.is_active ?? true,
    });
    setError("");
  };

  const handleUpdate = async () => {
    if (editingId === null) return;
    setError("");
    try {
      const data: any = {
        broker_utc_offset: parseInt(form.broker_utc_offset) || 0,
        label: form.label,
        is_active: form.is_active,
        max_drawdown_pct: parseFloat(form.max_drawdown_pct) || 10,
        risk_per_trade_pct: parseFloat(form.risk_per_trade_pct) || 1,
        max_daily_drawdown_pct: parseFloat(form.max_daily_drawdown_pct) || 5,
      };
      if (form.password) data.password = form.password;
      await api.updateAccount(editingId, data);
      setEditingId(null);
      setForm({ ...EMPTY_FORM });
      load();
    } catch (e: any) {
      setError(e.message || "Failed to update");
    }
  };

  const handleCancel = () => {
    setEditingId(null);
    setForm({ ...EMPTY_FORM });
    setError("");
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Delete this account?")) return;
    await api.deleteAccount(id);
    if (editingId === id) handleCancel();
    load();
  };

  const isEditing = editingId !== null;

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Admin</h2>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>MT5 Accounts</CardTitle></CardHeader>
          {accounts.length === 0 ? (
            <p className="text-sm text-zinc-500">No accounts configured</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-500">
                    <th className="pb-2 text-left">ID</th>
                    <th className="pb-2 text-left">Login</th>
                    <th className="pb-2 text-left">Server</th>
                    <th className="pb-2 text-left">Label</th>
                    <th className="pb-2 text-right">Max DD%</th>
                    <th className="pb-2 text-right">Risk/Trade%</th>
                    <th className="pb-2 text-right">Daily DD%</th>
                    <th className="pb-2 text-left">Active</th>
                    <th className="pb-2 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {accounts.map((a) => (
                    <tr
                      key={a.id}
                      className={`border-b border-zinc-800/50 ${editingId === a.id ? "bg-blue-950/20" : ""}`}
                    >
                      <td className="py-2">{a.id}</td>
                      <td className="font-mono">{a.login}</td>
                      <td>{a.server}</td>
                      <td className="text-zinc-400">{a.label || "—"}</td>
                      <td className="text-right font-mono">{a.max_drawdown_pct ?? 10}%</td>
                      <td className="text-right font-mono">{a.risk_per_trade_pct ?? 1}%</td>
                      <td className="text-right font-mono">{a.max_daily_drawdown_pct ?? 5}%</td>
                      <td>
                        <Badge variant={a.is_active ? "success" : "default"}>
                          {a.is_active ? "Yes" : "No"}
                        </Badge>
                      </td>
                      <td className="text-right">
                        <div className="flex justify-end gap-1">
                          <button
                            onClick={() => handleEdit(a)}
                            className="rounded bg-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-600"
                          >
                            Edit
                          </button>
                          <button
                            onClick={() => handleDelete(a.id)}
                            className="rounded bg-red-900/50 px-2 py-1 text-xs text-red-400 hover:bg-red-900"
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{isEditing ? "Edit Account" : "Add Account"}</CardTitle>
          </CardHeader>
          <div className="space-y-3">
            {isEditing && (
              <div className="rounded-lg border border-blue-800 bg-blue-950/30 px-3 py-2">
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-xs text-blue-400">Editing Account</span>
                    <div className="flex items-center gap-3 mt-0.5">
                      <span className="font-mono text-sm text-white">{form.login}</span>
                      <span className="text-xs text-zinc-400">{form.server}</span>
                      <Badge variant="default">ID: {editingId}</Badge>
                    </div>
                  </div>
                  <button
                    onClick={handleCancel}
                    className="rounded px-2 py-1 text-xs text-zinc-400 hover:text-white hover:bg-zinc-700"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
            {!isEditing && (
              <>
                <div>
                  <label className="mb-1 block text-xs text-zinc-400">MT5 Login</label>
                  <input
                    type="number"
                    value={form.login}
                    onChange={(e) => set("login", e.target.value)}
                    placeholder="12345678"
                    className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs text-zinc-400">Server</label>
                  <input
                    value={form.server}
                    onChange={(e) => set("server", e.target.value)}
                    placeholder="Exness-MT5Trial7"
                    className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
                  />
                </div>
              </>
            )}
            <div>
              <label className="mb-1 block text-xs text-zinc-400">
                Password {isEditing && <span className="text-zinc-600">(leave blank to keep current)</span>}
              </label>
              <input
                type="password"
                value={form.password}
                onChange={(e) => set("password", e.target.value)}
                placeholder="••••••••"
                className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="mb-1 block text-xs text-zinc-400">Broker UTC Offset</label>
                <input
                  type="number"
                  value={form.broker_utc_offset}
                  onChange={(e) => set("broker_utc_offset", e.target.value)}
                  placeholder="0"
                  className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs text-zinc-400">Label</label>
                <input
                  value={form.label}
                  onChange={(e) => set("label", e.target.value)}
                  placeholder="e.g. Exness Demo"
                  className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
                />
              </div>
            </div>

            <div className="border-t border-zinc-800 pt-3">
              <p className="mb-2 text-xs font-medium text-zinc-400">Risk Parameters</p>
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="mb-1 block text-xs text-zinc-500">Max Drawdown %</label>
                  <input
                    type="number"
                    step="0.1"
                    value={form.max_drawdown_pct}
                    onChange={(e) => set("max_drawdown_pct", e.target.value)}
                    className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs text-zinc-500">Risk/Trade %</label>
                  <input
                    type="number"
                    step="0.1"
                    value={form.risk_per_trade_pct}
                    onChange={(e) => set("risk_per_trade_pct", e.target.value)}
                    className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs text-zinc-500">Max Daily DD %</label>
                  <input
                    type="number"
                    step="0.1"
                    value={form.max_daily_drawdown_pct}
                    onChange={(e) => set("max_daily_drawdown_pct", e.target.value)}
                    className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
                  />
                </div>
              </div>
            </div>

            {isEditing && (
              <div className="flex items-center gap-2">
                <label className="text-xs text-zinc-400">Active</label>
                <button
                  onClick={() => set("is_active", !form.is_active)}
                  className={`rounded px-3 py-1 text-xs font-medium ${
                    form.is_active
                      ? "bg-emerald-900/40 text-emerald-400 border border-emerald-700"
                      : "bg-zinc-800 text-zinc-500 border border-zinc-700"
                  }`}
                >
                  {form.is_active ? "Active" : "Inactive"}
                </button>
              </div>
            )}

            {error && (
              <div className="rounded bg-red-900/40 border border-red-800 px-3 py-1.5 text-xs text-red-300">
                {error}
              </div>
            )}

            <div className="flex gap-2">
              {isEditing ? (
                <>
                  <button
                    onClick={handleUpdate}
                    className="flex-1 rounded bg-brand-600 px-3 py-2 text-sm font-medium hover:bg-brand-500"
                  >
                    Save Changes
                  </button>
                  <button
                    onClick={handleCancel}
                    className="rounded bg-zinc-700 px-3 py-2 text-sm font-medium hover:bg-zinc-600"
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <button
                  onClick={handleCreate}
                  className="w-full rounded bg-brand-600 px-3 py-2 text-sm font-medium hover:bg-brand-500"
                >
                  Add Account
                </button>
              )}
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
