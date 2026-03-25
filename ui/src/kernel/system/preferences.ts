/**
 * Unified UI Preferences — single KV key holding all plugin preferences.
 *
 * On boot, one GET loads everything. Writes are debounced and write-through.
 * Structure mirrors plugin IDs so each plugin owns its section.
 *
 * Key: "ui" namespace, "preferences" entry.
 */

import type { ApiClient } from "../../api.ts";

type KvApi = Pick<ApiClient, "kvGet" | "kvPut">;

const KV_NAMESPACE = "ui";
const KV_KEY = "preferences";
const DEBOUNCE_MS = 500;

interface PreferencesData {
  version: number;
  [pluginId: string]: unknown;
}

const DEFAULTS: PreferencesData = {
  version: 1,
  kernel: {
    keybindings: {},
  },
  "talu.chat": {
    thinking_expanded: false,
    collapsed_groups: [],
  },
  "talu.settings": {
    custom_themes: [],
  },
  "talu.editorops": {
    auto_save: false,
  },
  "talu.repo": {
    pinned_models: [],
  },
};

class Preferences {
  private data: PreferencesData = structuredClone(DEFAULTS);
  private api: KvApi | null = null;
  private dirty = false;
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;

  /** Load preferences from KV. Call once during boot. */
  async load(api: KvApi): Promise<void> {
    this.api = api;
    try {
      const result = await api.kvGet(KV_NAMESPACE, KV_KEY);
      if (result.ok && result.data?.value != null) {
        const parsed = JSON.parse(result.data.value) as PreferencesData;
        // Merge with defaults so new keys get populated.
        this.data = this.mergeWithDefaults(parsed);
        return;
      }
    } catch { /* fall through to defaults */ }

    // First boot or missing key — try migrating from old KV keys.
    await this.migrateFromLegacy(api);
  }

  /** Read a preference value. Synchronous — always from cache. */
  get<T = unknown>(pluginId: string, key: string): T | undefined {
    const section = this.data[pluginId] as Record<string, unknown> | undefined;
    if (!section || typeof section !== "object") return undefined;
    return section[key] as T | undefined;
  }

  /** Get a full plugin section. */
  getSection<T = Record<string, unknown>>(pluginId: string): T | undefined {
    return this.data[pluginId] as T | undefined;
  }

  /** Update a preference value. Updates cache immediately, debounced persist. */
  set(pluginId: string, key: string, value: unknown): void {
    let section = this.data[pluginId] as Record<string, unknown> | undefined;
    if (!section || typeof section !== "object") {
      section = {};
      this.data[pluginId] = section;
    }
    section[key] = value;
    this.schedulePersist();
  }

  /** Delete a preference key. */
  delete(pluginId: string, key: string): void {
    const section = this.data[pluginId] as Record<string, unknown> | undefined;
    if (section && typeof section === "object") {
      delete section[key];
      this.schedulePersist();
    }
  }

  /** Force an immediate persist (e.g. before page unload). */
  async flush(): Promise<void> {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    if (this.dirty) {
      await this.persist();
    }
  }

  private schedulePersist(): void {
    this.dirty = true;
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      this.debounceTimer = null;
      void this.persist();
    }, DEBOUNCE_MS);
  }

  private async persist(): Promise<void> {
    if (!this.api) return;
    this.dirty = false;
    try {
      await this.api.kvPut(KV_NAMESPACE, KV_KEY, JSON.stringify(this.data));
    } catch { /* best-effort */ }
  }

  /** Deep merge loaded data with defaults — ensures new keys are present. */
  private mergeWithDefaults(loaded: PreferencesData): PreferencesData {
    const result: PreferencesData = { version: loaded.version ?? DEFAULTS.version };
    for (const pluginId of Object.keys(DEFAULTS)) {
      if (pluginId === "version") continue;
      const defaultSection = DEFAULTS[pluginId] as Record<string, unknown>;
      const loadedSection = loaded[pluginId] as Record<string, unknown> | undefined;
      result[pluginId] = { ...defaultSection, ...(loadedSection ?? {}) };
    }
    // Preserve any extra plugin sections not in defaults.
    for (const pluginId of Object.keys(loaded)) {
      if (pluginId === "version" || pluginId in result) continue;
      result[pluginId] = loaded[pluginId];
    }
    return result;
  }

  /** One-time migration from old per-key KV storage. */
  private async migrateFromLegacy(api: KvApi): Promise<void> {
    let migrated = false;

    // kernel.keybindings from ui/talu.keybindings
    try {
      const r = await api.kvGet("ui", "talu.keybindings");
      if (r.ok && r.data?.value) {
        const parsed = JSON.parse(r.data.value);
        if (typeof parsed === "object" && parsed !== null) {
          (this.data.kernel as Record<string, unknown>).keybindings = parsed;
          migrated = true;
        }
      }
    } catch { /* ignore */ }

    // talu.chat.collapsed_groups from ui/talu-collapsed-groups
    try {
      const r = await api.kvGet("ui", "talu-collapsed-groups");
      if (r.ok && r.data?.value) {
        const parsed = JSON.parse(r.data.value);
        if (Array.isArray(parsed)) {
          (this.data["talu.chat"] as Record<string, unknown>).collapsed_groups = parsed;
          migrated = true;
        }
      }
    } catch { /* ignore */ }

    // talu.chat.thinking_expanded from plugin:talu.chat/thinkingExpanded
    try {
      const r = await api.kvGet("plugin:talu.chat", "thinkingExpanded");
      if (r.ok && r.data?.value) {
        (this.data["talu.chat"] as Record<string, unknown>).thinking_expanded = r.data.value === "true";
        migrated = true;
      }
    } catch { /* ignore */ }

    // talu.settings.custom_themes from plugin:talu.settings/custom_themes
    try {
      const r = await api.kvGet("plugin:talu.settings", "custom_themes");
      if (r.ok && r.data?.value) {
        const parsed = JSON.parse(r.data.value);
        if (Array.isArray(parsed)) {
          (this.data["talu.settings"] as Record<string, unknown>).custom_themes = parsed;
          migrated = true;
        }
      }
    } catch { /* ignore */ }

    // talu.editorops.auto_save from plugin:talu.editorops/editor.autoSave
    try {
      const r = await api.kvGet("plugin:talu.editorops", "editor.autoSave");
      if (r.ok && r.data?.value) {
        (this.data["talu.editorops"] as Record<string, unknown>).auto_save = r.data.value === "true";
        migrated = true;
      }
    } catch { /* ignore */ }

    // talu.repo.pinned_models from chat_models/models
    try {
      const r = await api.kvGet("chat_models", "models");
      if (r.ok && r.data?.value) {
        const parsed = JSON.parse(r.data.value);
        if (Array.isArray(parsed)) {
          (this.data["talu.repo"] as Record<string, unknown>).pinned_models = parsed;
          migrated = true;
        }
      }
    } catch { /* ignore */ }

    // Persist the unified config if we migrated anything.
    if (migrated) {
      await this.persist();
    }
  }
}

export const preferences = new Preferences();
