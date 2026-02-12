/**
 * Network Facade — wraps fetch with plugin identity headers.
 *
 * Adds X-Talu-Plugin-Id header for attribution and capability token
 * as Authorization: Bearer for gated endpoints (proxy, plugin storage).
 *
 * Tracks connectivity passively: emits system.server.connected /
 * system.server.disconnected on the EventBus when fetch transitions
 * between success and failure.
 */

import type { NetworkAccess } from "../types.ts";
import type { EventBusImpl } from "./event-bus.ts";

/** Shared connectivity tracker — one instance per kernel, injected into all NetworkAccessImpl. */
export class NetworkConnectivity {
  private connected = true;
  private eventBus: EventBusImpl;

  constructor(eventBus: EventBusImpl) {
    this.eventBus = eventBus;
  }

  onSuccess(): void {
    if (!this.connected) {
      this.connected = true;
      this.eventBus.emit("system.server.connected", {});
    }
  }

  onFailure(): void {
    if (this.connected) {
      this.connected = false;
      this.eventBus.emit("system.server.disconnected", {});
    }
  }
}

export class NetworkAccessImpl implements NetworkAccess {
  private pluginId: string;
  private token: string | null;
  private connectivity: NetworkConnectivity;

  constructor(pluginId: string, token: string | null, connectivity: NetworkConnectivity) {
    this.pluginId = pluginId;
    this.token = token;
    this.connectivity = connectivity;
  }

  async fetch(url: string, init?: RequestInit): Promise<Response> {
    // Clone init to avoid mutating caller's object.
    const options: RequestInit = { ...init };
    const headers = new Headers(options.headers);

    headers.set("X-Talu-Plugin-Id", this.pluginId);
    if (this.token) {
      headers.set("Authorization", `Bearer ${this.token}`);
    }

    options.headers = headers;

    try {
      const resp = await window.fetch(url, options);
      this.connectivity.onSuccess();
      return resp;
    } catch (err) {
      this.connectivity.onFailure();
      // Wrap in a typed NetworkError for consistent error handling.
      throw new NetworkError(
        `Network request failed for plugin "${this.pluginId}": ${err instanceof Error ? err.message : String(err)}`,
        err,
      );
    }
  }
}

export class NetworkError extends Error {
  readonly cause: unknown;

  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = "NetworkError";
    this.cause = cause;
  }
}
