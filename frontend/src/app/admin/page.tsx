"use client";
import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

export default function AdminPage() {
  const [accounts, setAccounts] = useState<any[]>([]);
  const [login, setLogin] = useState("");
  const [password, setPassword] = useState("");
  const [server, setServer] = useState("");
  const [offset, setOffset] = useState("0");
  const [label, setLabel] = useState("");

  const load = () => api.getAccounts().then(setAccounts).catch(() => {});

  useEffect(() => { load(); }, []);

  const handleCreate = async () => {
    if (!login || !server) return;
    await api.createAccount({
      login: parseInt(login),
      password,
      server,
      broker_utc_offset: parseInt(offset) || 0,
      label,
    });
    setLogin("");
    setPassword("");
    setServer("");
    setOffset("0");
    setLabel("");
    load();
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Delete this account?")) return;
    await api.deleteAccount(id);
    load();
  };

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
                  <th className="pb-2 text-left">UTC Offset</th>
                  <th className="pb-2 text-left">Label</th>
                  <th className="pb-2 text-left">Active</th>
                  <th className="pb-2 text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {accounts.map((a) => (
                  <tr key={a.id} className="border-b border-zinc-800/50">
                    <td className="py-2">{a.id}</td>
                    <td className="font-mono">{a.login}</td>
                    <td>{a.server}</td>
                    <td className="font-mono">{a.broker_utc_offset ?? 0}</td>
                    <td className="text-zinc-400">{a.label || "—"}</td>
                    <td>
                      <Badge variant={a.is_active ? "success" : "default"}>
                        {a.is_active ? "Yes" : "No"}
                      </Badge>
                    </td>
                    <td className="text-right">
                      <button
                        onClick={() => handleDelete(a.id)}
                        className="rounded bg-red-900/50 px-2 py-1 text-xs text-red-400 hover:bg-red-900"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          )}
        </Card>

        <Card>
          <CardHeader><CardTitle>Add Account</CardTitle></CardHeader>
          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-xs text-zinc-400">MT5 Login</label>
              <input
                type="number"
                value={login}
                onChange={(e) => setLogin(e.target.value)}
                placeholder="12345678"
                className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-zinc-400">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-zinc-400">Server</label>
              <input
                value={server}
                onChange={(e) => setServer(e.target.value)}
                placeholder="Exness-MT5Trial7"
                className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-zinc-400">Broker UTC Offset (hours)</label>
              <input
                type="number"
                value={offset}
                onChange={(e) => setOffset(e.target.value)}
                placeholder="0"
                className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-zinc-400">Label (optional)</label>
              <input
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="e.g. Exness Demo"
                className="w-full rounded bg-zinc-800 px-3 py-1.5 text-sm"
              />
            </div>
            <button
              onClick={handleCreate}
              className="w-full rounded bg-brand-600 px-3 py-2 text-sm font-medium hover:bg-brand-500"
            >
              Add Account
            </button>
          </div>
        </Card>
      </div>
    </div>
  );
}
