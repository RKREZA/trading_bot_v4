"use client";
import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

export default function StrategiesPage() {
  const [strategies, setStrategies] = useState<any[]>([]);

  const load = () => api.getStrategies().then(setStrategies).catch(() => {});

  useEffect(() => { load(); }, []);

  const handleToggle = async (id: string) => {
    await api.toggleStrategy(id);
    load();
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Strategies</h2>

      {strategies.length === 0 ? (
        <p className="text-zinc-500">No strategies registered</p>
      ) : (
        <div className="grid grid-cols-2 gap-4">
          {strategies.map((s) => (
            <Card key={s.strategy_id}>
              <CardHeader>
                <CardTitle>{s.strategy_id}</CardTitle>
                <Badge variant={s.enabled ? "success" : "default"}>
                  {s.enabled ? "Enabled" : "Disabled"}
                </Badge>
              </CardHeader>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between text-zinc-400">
                  <span>Class</span>
                  <span className="text-white">{s.class}</span>
                </div>
                {s.thresholds &&
                  Object.entries(s.thresholds).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-zinc-400">
                      <span>{k}</span>
                      <span className="font-mono text-white">{String(v)}</span>
                    </div>
                  ))}
                <button
                  onClick={() => handleToggle(s.strategy_id)}
                  className={`mt-2 w-full rounded px-3 py-1.5 text-xs font-medium ${
                    s.enabled
                      ? "bg-zinc-700 hover:bg-zinc-600"
                      : "bg-emerald-700 hover:bg-emerald-600"
                  }`}
                >
                  {s.enabled ? "Disable" : "Enable"}
                </button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
