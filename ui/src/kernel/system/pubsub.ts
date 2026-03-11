/**
 * PubSub client — topic-based publish-subscribe over a single WebSocket.
 *
 * Used for instant cross-window sync (editor, agents, etc.). The server
 * at `/v1/pubsub/ws` relays published messages to all other subscribers
 * of the same topic.
 */

import type { Disposable } from "../types.ts";

export interface PubSubClient {
  /** Subscribe to a topic. Messages from other clients arrive via onMessage. */
  subscribe(topic: string): void;
  /** Unsubscribe from a topic. */
  unsubscribe(topic: string): void;
  /** Publish data to a topic. All other subscribers receive it instantly. */
  publish(topic: string, data: unknown): void;
  /** Register a handler for messages on a specific topic. Returns Disposable. */
  onMessage(topic: string, handler: (data: unknown) => void): Disposable;
  /** Close the WebSocket connection and clean up. */
  close(): void;
}

const RECONNECT_DELAY_MS = 1000;

function toWsUrl(path: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}

/**
 * Connect to the pubsub WebSocket. Auto-reconnects on disconnect.
 * Re-subscribes to all active topics on reconnect.
 */
export function connectPubSub(): PubSubClient {
  type Handler = (data: unknown) => void;

  const handlers = new Map<string, Set<Handler>>();
  const activeTopics = new Set<string>();
  let ws: WebSocket | null = null;
  let closed = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  function send(msg: Record<string, unknown>): void {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  }

  function connect(): void {
    if (closed) return;

    ws = new WebSocket(toWsUrl("/v1/pubsub/ws"));

    ws.addEventListener("open", () => {
      // Re-subscribe to all active topics after reconnect.
      for (const topic of activeTopics) {
        send({ type: "subscribe", topic });
      }
    });

    ws.addEventListener("message", (evt) => {
      if (typeof evt.data !== "string") return;
      let msg: { type?: string; topic?: string; data?: unknown } | null = null;
      try { msg = JSON.parse(evt.data); } catch { return; }
      if (!msg || msg.type !== "message" || !msg.topic) return;

      const topicHandlers = handlers.get(msg.topic);
      if (topicHandlers) {
        for (const h of topicHandlers) {
          try { h(msg.data); } catch { /* handler error */ }
        }
      }
    });

    ws.addEventListener("close", () => {
      ws = null;
      if (!closed) {
        reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
      }
    });

    ws.addEventListener("error", () => {
      // Will trigger close event, which handles reconnect.
    });
  }

  connect();

  return {
    subscribe(topic: string): void {
      activeTopics.add(topic);
      send({ type: "subscribe", topic });
    },

    unsubscribe(topic: string): void {
      activeTopics.delete(topic);
      send({ type: "unsubscribe", topic });
    },

    publish(topic: string, data: unknown): void {
      send({ type: "publish", topic, data });
    },

    onMessage(topic: string, handler: Handler): Disposable {
      let set = handlers.get(topic);
      if (!set) {
        set = new Set();
        handlers.set(topic, set);
      }
      set.add(handler);
      return {
        dispose() {
          set!.delete(handler);
          if (set!.size === 0) {
            handlers.delete(topic);
          }
        },
      };
    },

    close(): void {
      closed = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (ws) {
        ws.close();
        ws = null;
      }
      handlers.clear();
      activeTopics.clear();
    },
  };
}
