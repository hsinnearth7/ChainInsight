import { useEffect, useRef, useCallback, useState } from 'react';
import type { WSMessage } from '../types/websocket';
import { auth } from '../api/client';

const isBrowser = typeof window !== 'undefined';

interface UseWebSocketOptions {
  url: string;
  onMessage?: (msg: WSMessage) => void;
  reconnectInterval?: number;
  enabled?: boolean;
}

export function useWebSocket({ url, onMessage, reconnectInterval = 3000, enabled = true }: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const [connected, setConnected] = useState(false);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    if (!enabled || !isBrowser || typeof WebSocket === 'undefined') return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrlBase = url.startsWith('ws') ? url : `${protocol}//${window.location.host}${url}`;
    const apiKey = auth.getApiKey();
    const wsUrl = apiKey
      ? `${wsUrlBase}${wsUrlBase.includes('?') ? '&' : '?'}api_key=${encodeURIComponent(apiKey)}`
      : wsUrlBase;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);
        onMessageRef.current?.(msg);
      } catch {
        // ignore non-JSON messages
      }
    };

    ws.onclose = (event) => {
      setConnected(false);
      wsRef.current = null;
      if (event.code !== 4003 && enabled) {
        reconnectTimer.current = setTimeout(connect, reconnectInterval);
      }
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [url, reconnectInterval, enabled]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((data: string) => {
    wsRef.current?.send(data);
  }, []);

  return { connected, send };
}
