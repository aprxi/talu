/**
 * Storage Facade — scoped KV store per plugin via /v1/documents API.
 *
 * Maps get/set/delete/keys/clear to document CRUD with:
 *   doc_type = "plugin_storage"
 *   owner_id = pluginId (enforced by capability token on server)
 *
 * Cross-tab change notifications via BroadcastChannel.
 */

import type { Disposable, StorageAccess } from "../types.ts";
import { MAX_DOCUMENT_BYTES } from "../security/resource-caps.ts";

const CHANNEL_NAME = "talu-storage-changes";
const DOC_TYPE = "plugin_storage";

export class StorageFacadeImpl implements StorageAccess, Disposable {
  private pluginId: string;
  private token: string | null;
  private changeCallbacks = new Set<(key: string | null) => void>();
  private channel: BroadcastChannel | null = null;

  constructor(pluginId: string, token: string | null) {
    this.pluginId = pluginId;
    this.token = token;
    this.setupChannel();
  }

  private setupChannel(): void {
    try {
      this.channel = new BroadcastChannel(CHANNEL_NAME);
      this.channel.onmessage = (e) => {
        const data = e.data as { pluginId: string; key: string | null };
        if (data.pluginId === this.pluginId) {
          this.notifyChange(data.key);
        }
      };
    } catch {
      // BroadcastChannel not available — degrade silently.
    }
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = {
      "Content-Type": "application/json",
      "X-Talu-Plugin-Id": this.pluginId,
    };
    if (this.token) {
      h["Authorization"] = `Bearer ${this.token}`;
    }
    return h;
  }

  async get<T = unknown>(key: string): Promise<T | null> {
    const params = new URLSearchParams({
      type: DOC_TYPE,
      owner_id: this.pluginId,
      title: key,
    });
    const res = await fetch(`/v1/documents?${params}`, { headers: this.headers() });
    if (!res.ok) return null;
    const docs = await res.json();
    if (Array.isArray(docs) && docs.length > 0) {
      return docs[0].content as T;
    }
    return null;
  }

  async set(key: string, value: unknown): Promise<void> {
    // Enforce per-document size limit before sending.
    const payload = JSON.stringify({ content: value });
    if (payload.length > MAX_DOCUMENT_BYTES) {
      throw new Error(
        `Storage document exceeds size limit: ${payload.length} bytes > ${MAX_DOCUMENT_BYTES} bytes`,
      );
    }

    // Upsert: try to find existing, then create or update.
    const existing = await this.findDocId(key);

    if (existing) {
      await fetch(`/v1/documents/${existing}`, {
        method: "PATCH",
        headers: this.headers(),
        body: payload,
      });
    } else {
      await fetch("/v1/documents", {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({
          type: DOC_TYPE,
          owner_id: this.pluginId,
          title: key,
          content: value,
        }),
      });
    }

    this.broadcastChange(key);
    this.notifyChange(key);
  }

  async delete(key: string): Promise<void> {
    const docId = await this.findDocId(key);
    if (docId) {
      await fetch(`/v1/documents/${docId}`, {
        method: "DELETE",
        headers: this.headers(),
      });
      this.broadcastChange(key);
      this.notifyChange(key);
    }
  }

  async keys(): Promise<string[]> {
    const params = new URLSearchParams({
      type: DOC_TYPE,
      owner_id: this.pluginId,
    });
    const res = await fetch(`/v1/documents?${params}`, { headers: this.headers() });
    if (!res.ok) return [];
    const docs = await res.json();
    if (Array.isArray(docs)) {
      return docs.map((d: { title: string }) => d.title);
    }
    return [];
  }

  async clear(): Promise<void> {
    const allKeys = await this.keys();
    for (const key of allKeys) {
      await this.delete(key);
    }
    this.broadcastChange(null);
    this.notifyChange(null);
  }

  onDidChange(callback: (key: string | null) => void): Disposable {
    this.changeCallbacks.add(callback);
    return {
      dispose: () => {
        this.changeCallbacks.delete(callback);
      },
    };
  }

  private async findDocId(key: string): Promise<string | null> {
    const params = new URLSearchParams({
      type: DOC_TYPE,
      owner_id: this.pluginId,
      title: key,
    });
    const res = await fetch(`/v1/documents?${params}`, { headers: this.headers() });
    if (!res.ok) return null;
    const docs = await res.json();
    if (Array.isArray(docs) && docs.length > 0) {
      return docs[0].id ?? null;
    }
    return null;
  }

  private broadcastChange(key: string | null): void {
    try {
      this.channel?.postMessage({ pluginId: this.pluginId, key });
    } catch {
      // Ignore broadcast failures.
    }
  }

  private notifyChange(key: string | null): void {
    for (const cb of this.changeCallbacks) {
      try {
        cb(key);
      } catch (err) {
        console.error(`[kernel] Storage change callback for "${this.pluginId}" threw:`, err);
      }
    }
  }

  dispose(): void {
    this.channel?.close();
    this.changeCallbacks.clear();
  }
}

/**
 * LocalStorage-backed implementation of StorageAccess for built-in plugins.
 *
 * Keys are namespaced by pluginId to prevent cross-plugin collisions:
 *   localStorage key = `talu-storage:${pluginId}:${key}`
 *
 * Same async interface as StorageFacadeImpl so plugins are transport-agnostic.
 * Cross-tab notifications via BroadcastChannel (same as server-backed storage).
 */
export class LocalStorageFacade implements StorageAccess, Disposable {
  private pluginId: string;
  private changeCallbacks = new Set<(key: string | null) => void>();
  private channel: BroadcastChannel | null = null;

  constructor(pluginId: string) {
    this.pluginId = pluginId;
    this.setupChannel();
  }

  private prefix(key: string): string {
    return `talu-storage:${this.pluginId}:${key}`;
  }

  private setupChannel(): void {
    try {
      this.channel = new BroadcastChannel(CHANNEL_NAME);
      this.channel.onmessage = (e) => {
        const data = e.data as { pluginId: string; key: string | null };
        if (data.pluginId === this.pluginId) {
          this.notifyChange(data.key);
        }
      };
    } catch {
      // BroadcastChannel not available — degrade silently.
    }
  }

  async get<T = unknown>(key: string): Promise<T | null> {
    try {
      const raw = localStorage.getItem(this.prefix(key));
      if (raw === null) return null;
      return JSON.parse(raw) as T;
    } catch {
      return null;
    }
  }

  async set(key: string, value: unknown): Promise<void> {
    localStorage.setItem(this.prefix(key), JSON.stringify(value));
    this.broadcastChange(key);
    this.notifyChange(key);
  }

  async delete(key: string): Promise<void> {
    localStorage.removeItem(this.prefix(key));
    this.broadcastChange(key);
    this.notifyChange(key);
  }

  async keys(): Promise<string[]> {
    const prefix = `talu-storage:${this.pluginId}:`;
    const result: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k?.startsWith(prefix)) {
        result.push(k.slice(prefix.length));
      }
    }
    return result;
  }

  async clear(): Promise<void> {
    const allKeys = await this.keys();
    for (const key of allKeys) {
      localStorage.removeItem(this.prefix(key));
    }
    this.broadcastChange(null);
    this.notifyChange(null);
  }

  onDidChange(callback: (key: string | null) => void): Disposable {
    this.changeCallbacks.add(callback);
    return {
      dispose: () => {
        this.changeCallbacks.delete(callback);
      },
    };
  }

  private broadcastChange(key: string | null): void {
    try {
      this.channel?.postMessage({ pluginId: this.pluginId, key });
    } catch {
      // Ignore broadcast failures.
    }
  }

  private notifyChange(key: string | null): void {
    for (const cb of this.changeCallbacks) {
      try {
        cb(key);
      } catch (err) {
        console.error(`[kernel] Storage change callback for "${this.pluginId}" threw:`, err);
      }
    }
  }

  dispose(): void {
    this.channel?.close();
    this.changeCallbacks.clear();
  }
}
