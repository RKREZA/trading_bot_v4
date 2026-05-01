"use client";
import { useEffect, useRef, useCallback, useState } from "react";

type WSMessage = {
  type: string;
  data: any;
  timestamp: string;
};

type WSOptions = {
  onMessage?: (msg: WSMessage) => void;
  subscriptions?: string[];
  reconnectInterval?: number;
};

export function useWebSocket(options: WSOptions = {}) {
  const {
    onMessage,
    subscriptions = ["METRICS", "SYSTEM_STATUS"],
    reconnectInterval = 3000,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const reconnectTimer = useRef<NodeJS.Timeout>();

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = process.env.NEXT_PUBLIC_WS_URL || `${protocol}//${window.location.host}`;
    const ws = new WebSocket(`${host}/ws/events`);

    ws.onopen = () => {
      setConnected(true);
      ws.send(JSON.stringify({ action: "subscribe", events: subscriptions }));
    };

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);
        onMessage?.(msg);
      } catch {}
    };

    ws.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, reconnectInterval);
    };

    ws.onerror = () => ws.close();

    wsRef.current = ws;
  }, [onMessage, subscriptions, reconnectInterval]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { connected, send };
}
