import type { PluginDefinition, PluginContext, Disposable } from "../types.ts";
import { DisposableStore } from "./disposable.ts";
import { HealthTracker, errorBoundary, asyncErrorBoundary } from "./errors.ts";
import { EventBusImpl } from "../system/event-bus.ts";
import { ServiceRegistry } from "../registries/services.ts";
import { createPluginSlot, initSharedStylesheet } from "../ui/layout.ts";
import { createPluginContext, type KernelInfrastructure } from "./context-impl.ts";
import { HookPipelineImpl } from "../registries/hooks.ts";
import { ToolRegistryImpl } from "../registries/tools.ts";
import { CommandRegistryImpl } from "../registries/commands.ts";
import { ThemeAccessImpl } from "../ui/theme.ts";
import { installNavigationInterception } from "../security/navigation.ts";
import { installSensitiveApiInterception } from "../security/sensitive-apis.ts";
import { freezeIntrinsics } from "../security/realm-integrity.ts";
import { installCommandPalette } from "../ui/command-palette.ts";
import { isSafeMode, setLoadingFlag, clearLoadingFlag } from "../security/safe-mode.ts";
import { setupAccessibility } from "../ui/accessibility.ts";
import { initProvenance, setProvenanceAction } from "../ui/provenance.ts";
import { checkCapabilities } from "../security/capabilities.ts";
import { validateManifest } from "./manifest-validator.ts";
import { registerAliases } from "./alias.ts";
import { PopoverManager } from "../ui/popover.ts";
import { RendererRegistryImpl } from "../registries/renderers.ts";
import { loadKeybindingOverrides } from "../registries/keybindings.ts";
import { StatusBarManager } from "../ui/status-bar.ts";
import { ViewManager } from "../ui/view-manager.ts";
import { importWithWatchdog, ImportTimeoutError } from "./import-watchdog.ts";
import { partitionByActivation, topologicalSort, parseActivationEvent, type PluginDescriptor } from "./activation.ts";
import { verifyIntegrity } from "../security/integrity.ts";
import { ModeManager } from "../ui/mode-manager.ts";
import { NetworkConnectivity } from "../system/network.ts";
import { restoreThemeSync } from "../../styles/theme.ts";
import { BUILTIN_SCHEMES } from "../../styles/color-schemes.ts";
import { setupThemePicker } from "../ui/theme-picker.ts";

// --- Per-plugin state ---

interface PluginEntry {
  definition: PluginDefinition;
  health: HealthTracker;
  disposables: DisposableStore;
  abortController: AbortController;
  ctx: PluginContext;
  phase: "registered" | "running" | "deactivated";
  previousState: unknown;
}

// --- Slot host resolution ---

/** Resolve the host element for a plugin from its manifest's contributes.mode. */
function resolveHostElement(manifest: PluginDefinition["manifest"]): HTMLElement {
  const modeKey = manifest.contributes?.mode?.key;
  if (modeKey) {
    const el = document.getElementById(`${modeKey}-mode`);
    if (el) return el;
  }

  // Third-party plugins get a hidden host (they render via ctx.layout.registerView).
  const existing = document.getElementById(`plugin-host-${manifest.id}`);
  if (existing) return existing;

  const host = document.createElement("div");
  host.id = `plugin-host-${manifest.id}`;
  host.style.display = "none";
  document.body.appendChild(host);
  return host;
}

// --- PluginManager ---

class PluginManager {
  private plugins = new Map<string, PluginEntry>();
  private infra: KernelInfrastructure;

  constructor(infra: KernelInfrastructure) {
    this.infra = infra;
  }

  registerPlugin(definition: PluginDefinition, token: string | null = null): void {
    const { id } = definition.manifest;

    if (this.plugins.has(id)) {
      console.error(`[kernel] Plugin "${id}" already registered.`);
      return;
    }

    // Check for conflicts with already-loaded plugins.
    if (definition.manifest.conflicts) {
      for (const conflictId of definition.manifest.conflicts) {
        if (this.plugins.has(conflictId)) {
          console.error(
            `[kernel] Plugin "${id}" conflicts with loaded plugin "${conflictId}" — skipping.`,
          );
          return;
        }
      }
    }
    // Check reverse: already-loaded plugins that declare conflict with this one.
    for (const [loadedId, loadedEntry] of this.plugins) {
      if (loadedEntry.definition.manifest.conflicts?.includes(id)) {
        console.error(
          `[kernel] Plugin "${id}" conflicts with loaded plugin "${loadedId}" — skipping.`,
        );
        return;
      }
    }

    // Validate capabilities.
    const capCheck = checkCapabilities(definition.manifest.requiresCapabilities);
    if (!capCheck.satisfied) {
      console.warn(
        `[kernel] Plugin "${id}" requires unsupported capabilities: ${capCheck.unsatisfied.join(", ")} — skipping.`,
      );
      return;
    }

    // Validate manifest fields.
    const validation = validateManifest(definition.manifest);
    if (!validation.valid) {
      console.error(`[kernel] Plugin "${id}" manifest invalid:`, validation.errors);
      return;
    }
    if (validation.warnings.length > 0) {
      console.warn(`[kernel] Plugin "${id}" manifest warnings:`, validation.warnings);
    }

    // Register ID aliases.
    registerAliases(definition.manifest.aliases);

    // Register mode contribution from manifest.
    const modeContrib = definition.manifest.contributes?.mode;
    if (modeContrib) {
      this.infra.modeManager.registerMode(modeContrib.key, modeContrib.label, id);
    }

    const hostElement = resolveHostElement(definition.manifest);
    const disposables = new DisposableStore();
    const health = new HealthTracker();
    const abortController = new AbortController();

    const container = createPluginSlot(id, hostElement);

    const ctx = createPluginContext(
      definition.manifest,
      container,
      this.infra,
      disposables,
      abortController,
      token,
    );

    const entry: PluginEntry = {
      definition,
      health,
      disposables,
      abortController,
      ctx,
      phase: "registered",
      previousState: undefined,
    };

    // Register manifest-declared status bar items.
    const statusBarDisposable = this.infra.statusBarManager.registerFromManifest(id, definition.manifest);
    disposables.track(statusBarDisposable);

    // Call register() in error boundary. Must be synchronous.
    const result = errorBoundary(id, () => definition.register(ctx));

    // Detect async register (returns Promise — that's an error).
    if (result && typeof (result as unknown as Promise<unknown>)?.then === "function") {
      console.error(
        `[kernel] Plugin "${id}" register() returned a Promise — register must be synchronous.`,
      );
      health.recordFailure();
    } else if (result === undefined && definition.register.length > 0) {
      // errorBoundary returned undefined because register() threw.
      health.recordFailure();
    }

    this.plugins.set(id, entry);
  }

  async runAll(): Promise<void> {
    for (const [id, entry] of this.plugins) {
      if (entry.phase === "running") continue;
      if (entry.health.isDisabled) {
        console.warn(`[kernel] Plugin "${id}" is disabled — skipping run().`);
        continue;
      }

      const result = await asyncErrorBoundary(id, () =>
        entry.definition.run(entry.ctx, entry.abortController.signal, entry.previousState),
      );

      if (result === undefined) {
        entry.health.recordFailure();
      } else {
        entry.health.recordSuccess();
        entry.phase = "running";
      }
    }
  }

  deactivatePlugin(pluginId: string): void {
    const entry = this.plugins.get(pluginId);
    if (!entry) return;

    // Capture state for hot-reload restoration.
    const state = errorBoundary(pluginId, () => entry.definition.deactivate?.());
    if (state !== undefined) {
      entry.previousState = state;
    }
    entry.abortController.abort();
    entry.disposables.dispose();
    entry.phase = "deactivated";
  }

  deactivateAll(): void {
    // Reverse order.
    const ids = [...this.plugins.keys()].reverse();
    for (const id of ids) {
      this.deactivatePlugin(id);
    }
  }
}

// --- Third-party plugin loading ---

interface RemotePluginDescriptor {
  manifest: PluginDefinition["manifest"];
  entryUrl: string;
  token?: string;
}

async function loadThirdPartyPlugins(manager: PluginManager, infra: KernelInfrastructure): Promise<void> {
  let descriptors: RemotePluginDescriptor[];
  try {
    const resp = await fetch("/v1/plugins");
    if (!resp.ok) {
      if (resp.status !== 404) {
        console.warn("[kernel] Failed to fetch plugin list — skipping third-party plugins.");
      }
      return;
    }
    descriptors = await resp.json();
  } catch {
    // Server unreachable or no plugin endpoint — not an error for V1.
    return;
  }

  if (!Array.isArray(descriptors) || descriptors.length === 0) return;

  // Validate manifests and check capabilities.
  const valid: PluginDescriptor[] = [];
  for (const desc of descriptors) {
    const validation = validateManifest(desc.manifest);
    if (!validation.valid) {
      console.error(`[kernel] Plugin "${desc.manifest.id}" manifest invalid:`, validation.errors);
      continue;
    }

    const capCheck = checkCapabilities(desc.manifest.requiresCapabilities);
    if (!capCheck.satisfied) {
      console.warn(
        `[kernel] Plugin "${desc.manifest.id}" requires unsupported capabilities: ${capCheck.unsatisfied.join(", ")} — skipping.`,
      );
      continue;
    }

    valid.push({ manifest: desc.manifest, entryUrl: desc.entryUrl, token: desc.token });
  }

  // Partition into eager/lazy and sort eager by dependencies.
  const { eager, lazy } = partitionByActivation(valid);
  const sorted = topologicalSort(eager);

  // Load eager plugins.
  for (const desc of sorted) {
    try {
      // Integrity verification: if the manifest declares a hash, verify before import.
      if (desc.manifest.integrity) {
        const ok = await verifyIntegrity(desc.entryUrl, desc.manifest.integrity);
        if (!ok) {
          console.error(`[kernel] Plugin "${desc.manifest.id}" failed integrity check — refusing to load.`);
          continue;
        }
      }

      const module = await importWithWatchdog<{ default?: PluginDefinition }>(desc.entryUrl);
      const pluginExport = module.default;
      if (!pluginExport || typeof pluginExport.register !== "function") {
        console.error(`[kernel] Plugin "${desc.manifest.id}" has no valid default export.`);
        continue;
      }

      const definition: PluginDefinition = {
        manifest: desc.manifest,
        register: pluginExport.register,
        run: pluginExport.run,
        deactivate: pluginExport.deactivate,
      };
      manager.registerPlugin(definition, desc.token ?? null);
    } catch (err) {
      if (err instanceof ImportTimeoutError) {
        console.error(`[kernel] Plugin "${desc.manifest.id}" import timed out — disabling.`);
      } else {
        console.error(`[kernel] Plugin "${desc.manifest.id}" import failed:`, err);
      }
    }
  }

  // Run newly registered third-party plugins.
  await manager.runAll();

  // Register lazy plugins for deferred activation.
  for (const desc of lazy) {
    registerLazyPlugin(desc, manager, infra);
  }
}

function registerLazyPlugin(
  desc: PluginDescriptor,
  manager: PluginManager,
  infra: KernelInfrastructure,
): void {
  const events = desc.manifest.activationEvents ?? [];
  for (const event of events) {
    const parsed = parseActivationEvent(event);
    if (!parsed) continue;

    const kernelEvent =
      parsed.type === "onView" ? `view.activated.${parsed.arg}` :
      parsed.type === "onCommand" ? `command.invoked.${parsed.arg}` :
      null;

    if (!kernelEvent) continue;

    infra.eventBus.once(kernelEvent, async () => {
      try {
        // Integrity verification for lazy-loaded plugins.
        if (desc.manifest.integrity) {
          const ok = await verifyIntegrity(desc.entryUrl, desc.manifest.integrity);
          if (!ok) {
            console.error(`[kernel] Plugin "${desc.manifest.id}" failed integrity check — refusing to load.`);
            return;
          }
        }

        const module = await importWithWatchdog<{ default?: PluginDefinition }>(desc.entryUrl);
        const pluginExport = module.default;
        if (pluginExport && typeof pluginExport.register === "function") {
          const definition: PluginDefinition = {
            manifest: desc.manifest,
            register: pluginExport.register,
            run: pluginExport.run,
            deactivate: pluginExport.deactivate,
          };
          manager.registerPlugin(definition, desc.token ?? null);
          await manager.runAll();
        }
      } catch (err) {
        console.error(`[kernel] Lazy activation of "${desc.manifest.id}" failed:`, err);
      }
    });
  }
}

// --- Global Escape handler (focus-trap fallback) ---

/**
 * Capture-phase Escape listener: if no kernel overlay (dialog, popover, command
 * palette) is open, focus the activity bar as a safe escape hatch. This ensures
 * a rogue plugin can never trap keyboard focus permanently.
 */
function installGlobalEscapeHandler(): Disposable {
  const handler = (e: KeyboardEvent) => {
    if (e.key !== "Escape") return;

    // Let kernel overlays handle Escape themselves.
    const hasDialog = document.getElementById("kernel-dialog-overlay");
    const hasPopover = document.querySelector("[data-popover-owner]");
    const hasPalette = document.getElementById("command-palette-overlay");
    if (hasDialog || hasPopover || hasPalette) return;

    const activityBar = document.getElementById("activity-bar");
    if (activityBar) {
      activityBar.focus();
    }
  };

  document.addEventListener("keydown", handler, { capture: true });
  return {
    dispose: () => document.removeEventListener("keydown", handler, { capture: true } as EventListenerOptions),
  };
}

// --- Boot ---

export async function bootKernel(builtinPlugins: PluginDefinition[]): Promise<void> {
  // Restore theme synchronously before any DOM rendering to prevent FOUC.
  restoreThemeSync();

  // Safe mode check.
  const safeMode = isSafeMode();

  // Create shared kernel singletons.
  const eventBus = new EventBusImpl();
  const serviceRegistry = new ServiceRegistry(eventBus);
  const hookPipeline = new HookPipelineImpl();
  const toolRegistry = new ToolRegistryImpl(hookPipeline);
  const commandRegistry = new CommandRegistryImpl();
  const themeAccess = new ThemeAccessImpl();
  themeAccess.registerBuiltinSchemes(BUILTIN_SCHEMES);
  const popoverManager = new PopoverManager();
  const rendererRegistry = new RendererRegistryImpl();
  const statusBarManager = new StatusBarManager();
  const viewManager = new ViewManager();
  const modeManager = new ModeManager(eventBus);
  const networkConnectivity = new NetworkConnectivity(eventBus);

  const infra: KernelInfrastructure = {
    eventBus,
    serviceRegistry,
    hookPipeline,
    toolRegistry,
    commandRegistry,
    themeAccess,
    popoverManager,
    rendererRegistry,
    statusBarManager,
    viewManager,
    modeManager,
    networkConnectivity,
  };

  // Load persisted keybinding overrides before plugin registration.
  loadKeybindingOverrides();

  // Install global interceptors.
  const kernelDisposables = new DisposableStore();
  kernelDisposables.track(installNavigationInterception());
  kernelDisposables.track(installSensitiveApiInterception());
  kernelDisposables.track(freezeIntrinsics());
  kernelDisposables.track(installGlobalEscapeHandler());

  // Kernel UI chrome.
  setupAccessibility();
  kernelDisposables.track(setupThemePicker(themeAccess));
  initProvenance();
  const paletteHandle = installCommandPalette(commandRegistry);
  kernelDisposables.track(paletteHandle);
  setProvenanceAction(() => paletteHandle.open());

  // Pre-load shared stylesheet for Shadow DOM slots (before any plugins register).
  await initSharedStylesheet();

  // Set crash detection flag before plugin loading.
  if (!safeMode) {
    setLoadingFlag();
  }

  // Register built-in plugins.
  const manager = new PluginManager(infra);
  for (const plugin of builtinPlugins) {
    manager.registerPlugin(plugin);
  }

  // Run all registered plugins.
  await manager.runAll();

  // Activate mode manager after plugins are running (so they can listen for mode.changed).
  kernelDisposables.track(modeManager.installActivityBarListeners());
  modeManager.restoreLastMode();

  // Load third-party plugins (skipped in safe mode).
  if (!safeMode) {
    await loadThirdPartyPlugins(manager, infra);
  }

  // Clear crash detection flag — boot succeeded.
  clearLoadingFlag();

  // Remove the pre-JS safe mode fallback button.
  document.getElementById("safe-mode-fallback")?.remove();

  if (safeMode) {
    console.info("[kernel] Boot complete (safe mode — third-party plugins skipped).");
  } else {
    console.info("[kernel] Boot complete.");
  }
}
