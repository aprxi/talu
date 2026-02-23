import type {
  PluginManifest,
  PluginContext,
  Disposable,
  HookPipeline,
  ToolRegistry,
  CommandRegistry as CommandRegistryInterface,
  EventBus,
  StorageAccess,
  UploadAccess,
  ContextAccess,
  MenuAccess,
} from "../types.ts";
import type { DisposableStore } from "./disposable.ts";
import type { EventBusImpl } from "../system/event-bus.ts";
import type { ServiceRegistry } from "../registries/services.ts";
import type { HookPipelineImpl } from "../registries/hooks.ts";
import type { ToolRegistryImpl } from "../registries/tools.ts";
import type { CommandRegistryImpl } from "../registries/commands.ts";
import type { ThemeAccessImpl } from "../ui/theme.ts";
import type { PopoverManager } from "../ui/popover.ts";
import type { RendererRegistryImpl } from "../registries/renderers.ts";
import { createLogger } from "../system/log.ts";
import { namespacedId, validateLocalId } from "./id-namespace.ts";
import { ManagedTimersImpl } from "../system/timers.ts";
import { StandardDialogsImpl } from "../ui/dialogs.ts";
import { HashRouterImpl } from "../system/router.ts";
import { createAssetResolver } from "../system/assets.ts";
import { ConfigurationAccessImpl } from "../system/configuration.ts";
import { PluginStatusImpl } from "./status.ts";
import { NotificationsImpl } from "../ui/notifications.ts";
import { ManagedObserversImpl } from "../system/observers.ts";
import { GlobalEventManager } from "../system/global-events.ts";
import { StorageFacadeImpl, LocalStorageFacade } from "../system/storage.ts";
import { NetworkAccessImpl, type NetworkConnectivity } from "../system/network.ts";
import { FormatAccessImpl } from "../system/format.ts";
import { resolveKeybinding } from "../registries/keybindings.ts";
import type { StatusBarManager } from "../ui/status-bar.ts";
import type { ViewManager } from "../ui/view-manager.ts";
import type { ModeManager } from "../ui/mode-manager.ts";
import type { ContextKeyService } from "../registries/context-keys.ts";
import type { MenuRegistry } from "../registries/menus.ts";
import { createApiClient } from "../../api.ts";

/** Shared kernel infrastructure passed to all plugin contexts. */
export interface KernelInfrastructure {
  eventBus: EventBusImpl;
  serviceRegistry: ServiceRegistry;
  hookPipeline: HookPipelineImpl;
  toolRegistry: ToolRegistryImpl;
  commandRegistry: CommandRegistryImpl;
  themeAccess: ThemeAccessImpl;
  popoverManager: PopoverManager;
  rendererRegistry: RendererRegistryImpl;
  statusBarManager: StatusBarManager;
  viewManager: ViewManager;
  modeManager: ModeManager;
  networkConnectivity: NetworkConnectivity;
  contextKeys: ContextKeyService;
  menuRegistry: MenuRegistry;
}

export function createPluginContext(
  manifest: PluginManifest,
  container: HTMLElement,
  infra: KernelInfrastructure,
  disposables: DisposableStore,
  abortController: AbortController,
  token: string | null = null,
): PluginContext {
  const pluginId = manifest.id;
  const isBuiltin = manifest.builtin === true;
  const log = createLogger(pluginId);

  // Permission gate: builtin plugins are trusted; third-party must declare permissions.
  const declaredPermissions = new Set(manifest.permissions ?? []);
  function requirePermission(name: string): void {
    if (isBuiltin) return;
    if (!declaredPermissions.has(name)) {
      throw new Error(
        `Plugin "${pluginId}" accessed "${name}" without declaring it in manifest.permissions`,
      );
    }
  }

  // --- Per-plugin instances (auto-disposed on deactivation) ---

  const timers = new ManagedTimersImpl(pluginId);
  disposables.track(timers);

  const observers = new ManagedObserversImpl(pluginId);
  disposables.track(observers);

  const globalEvt = new GlobalEventManager(pluginId);
  disposables.track(globalEvt);

  // Built-in plugins get localStorage-backed storage (no server token needed).
  // Third-party plugins get server-backed storage via /v1/db/tables/documents with a
  // capability token received from /v1/plugins during registration.
  const storage = isBuiltin
    ? new LocalStorageFacade(pluginId)
    : new StorageFacadeImpl(pluginId, token);
  disposables.track(storage);

  const config = new ConfigurationAccessImpl();

  // Permission-gated wrappers for sensitive APIs.
  const rawNetwork = new NetworkAccessImpl(pluginId, token, infra.networkConnectivity);
  const api = createApiClient((url, init) => rawNetwork.fetch(url, init));
  const networkAccess: { fetch(url: string, init?: RequestInit): Promise<Response> } = {
    fetch: (url, init) => { requirePermission("network"); return rawNetwork.fetch(url, init); },
  };

  const storageAccess: StorageAccess & Disposable = {
    get: (key) => { requirePermission("storage"); return storage.get(key); },
    set: (key, value) => { requirePermission("storage"); return storage.set(key, value); },
    delete: (key) => { requirePermission("storage"); return storage.delete(key); },
    keys: () => { requirePermission("storage"); return storage.keys(); },
    clear: () => { requirePermission("storage"); return storage.clear(); },
    onDidChange: (cb) => { requirePermission("storage"); return storage.onDidChange(cb); },
    dispose: () => storage.dispose(),
  };

  function unwrapUploadResult<T>(op: string, result: { ok: boolean; data?: T; error?: string }): T {
    if (result.ok && result.data !== undefined) {
      return result.data;
    }
    throw new Error(result.error ? `Upload ${op} failed: ${result.error}` : `Upload ${op} failed`);
  }

  const uploadAccess: UploadAccess = {
    async upload(file, purpose) {
      requirePermission("upload");
      const result = await api.uploadFile(file, purpose);
      const data = unwrapUploadResult("upload", result);
      return {
        id: data.id,
        filename: data.filename,
        bytes: data.bytes,
        createdAt: data.created_at,
        purpose: data.purpose,
      };
    },
    async get(fileId) {
      requirePermission("upload");
      const result = await api.getFile(fileId);
      const data = unwrapUploadResult("get", result);
      return {
        id: data.id,
        filename: data.filename,
        bytes: data.bytes,
        createdAt: data.created_at,
        purpose: data.purpose,
      };
    },
    async delete(fileId) {
      requirePermission("upload");
      const result = await api.deleteFile(fileId);
      if (!result.ok) {
        throw new Error(result.error ? `Upload delete failed: ${result.error}` : "Upload delete failed");
      }
    },
    async getContent(fileId) {
      requirePermission("upload");
      const result = await api.getFileContent(fileId);
      return unwrapUploadResult("getContent", result);
    },
  };

  // --- Scoped wrappers for shared singletons ---

  const hooks: HookPipeline = {
    on: (name, handler, options) => {
      requirePermission("hooks");
      return infra.hookPipeline.onScoped(
        pluginId,
        name,
        handler as (value: unknown) => unknown,
        options,
      );
    },
    run: (name, value) => {
      requirePermission("hooks");
      return infra.hookPipeline.run(name, value);
    },
  };

  const tools: ToolRegistry = {
    register: (id, definition) => {
      requirePermission("tools");
      validateLocalId(pluginId, id, isBuiltin);
      return infra.toolRegistry.registerScoped(
        pluginId,
        namespacedId(pluginId, id),
        definition,
      );
    },
    get: (id) => infra.toolRegistry.get(id),
  };

  const commands: CommandRegistryInterface = {
    register: (id, handler, options) => {
      validateLocalId(pluginId, id, isBuiltin);
      const fqId = namespacedId(pluginId, id);
      // Apply user keybinding overrides over manifest defaults.
      const effectiveKeybinding = resolveKeybinding(fqId, options?.keybinding);
      return infra.commandRegistry.registerScoped(
        pluginId,
        fqId,
        handler,
        { ...options, keybinding: effectiveKeybinding },
      );
    },
  };

  const context: ContextAccess = {
    set: (key, value) => infra.contextKeys.set(namespacedId(pluginId, key), value),
    get: (key) => infra.contextKeys.get(key),
    delete: (key) => infra.contextKeys.delete(namespacedId(pluginId, key)),
    has: (key) => infra.contextKeys.has(key),
    onChange: (key, callback) => infra.contextKeys.onChange(key, callback),
  };

  const menus: MenuAccess = {
    registerItem: (slot, item) => {
      requirePermission("menus");
      validateLocalId(pluginId, item.id, isBuiltin);
      return infra.menuRegistry.registerItem(pluginId, { ...item, slot });
    },
    renderSlot: (slot, container) => infra.menuRegistry.renderSlot(slot, container),
  };

  // --- Composed EventBus with global event support ---

  const events: EventBus = {
    on: <T = unknown>(event: string, handler: (data: T) => void) =>
      infra.eventBus.on(event, handler),
    once: <T = unknown>(event: string, handler: (data: T) => void) =>
      infra.eventBus.once(event, handler),
    emit: <T = unknown>(event: string, data: T) =>
      infra.eventBus.emit(event, data),
    onWindow: (name: string, handler: EventListener, opts?: AddEventListenerOptions) =>
      globalEvt.onWindow(name, handler, opts),
    onDocument: (name: string, handler: EventListener, opts?: AddEventListenerOptions) =>
      globalEvt.onDocument(name, handler, opts),
  };

  const ctx: PluginContext = {
    manifest: Object.freeze({ ...manifest }),
    container,
    log,
    events,
    services: infra.serviceRegistry,
    lifecycle: { signal: abortController.signal },
    subscriptions: {
      add(disposable: Disposable): void {
        disposables.track(disposable);
      },
    },

    // Extended APIs.
    layout: {
      setTitle: (title) => {
        const el = document.getElementById("topbar-title");
        if (el) el.textContent = title;
      },
      registerView: (slot, factory) => {
        const viewDef = manifest.contributes?.views?.find((v) => v.slot === slot);
        const priority = viewDef?.priority ?? 0;
        return infra.viewManager.registerView(pluginId, slot, manifest.name, isBuiltin, factory, priority);
      },
      showPopover: (options) => infra.popoverManager.showPopover(pluginId, options),
    },
    hooks,
    tools,
    commands,
    timers,
    dialogs: new StandardDialogsImpl(manifest.name),
    notifications: new NotificationsImpl(),
    router: new HashRouterImpl(pluginId),
    assets: createAssetResolver(pluginId),
    configuration: config,
    status: new PluginStatusImpl(pluginId),
    theme: infra.themeAccess,
    observe: observers,
    network: networkAccess,
    storage: storageAccess,
    renderers: {
      register: (renderer) => infra.rendererRegistry.registerScoped(pluginId, 0, renderer),
      registerPreProcessor: (fn) => infra.rendererRegistry.registerPreProcessorScoped(pluginId, fn),
      applyPreProcessors: (text) => infra.rendererRegistry.applyPreProcessors(text),
      mountPart: (partId, container, part) => infra.rendererRegistry.mountPart(partId, container, part),
      updatePart: (partId, part, isFinal) => infra.rendererRegistry.updatePart(partId, part, isFinal),
      unmountPart: (partId) => infra.rendererRegistry.unmountPart(partId),
    },
    mode: {
      getActive: () => infra.modeManager.getActiveMode(),
      switch: (mode) => infra.modeManager.switchMode(mode),
      onChange: (handler) => infra.eventBus.on("mode.changed", handler),
    },
    format: new FormatAccessImpl(),
    clipboard: {
      writeText: (text) => { requirePermission("clipboard"); return navigator.clipboard.writeText(text); },
    },
    download: {
      save: (blob, filename) => {
        requirePermission("download");
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
      },
    },
    upload: uploadAccess,
    context,
    menus,
  };

  return Object.freeze(ctx);
}
