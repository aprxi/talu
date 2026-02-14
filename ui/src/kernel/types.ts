/** Plugin API contract — interfaces only, no implementation. */

// --- Disposable ---

export interface Disposable {
  dispose(): void;
}

// --- Plugin identity ---

export interface PluginManifest {
  readonly id: string;
  readonly name: string;
  readonly version: string;
  readonly builtin?: boolean;
  readonly apiVersion?: string;
  readonly entry?: string;
  readonly activationEvents?: string[];
  readonly permissions?: string[];
  readonly requires?: { id: string; version: string }[];
  readonly enhances?: string[];
  readonly conflicts?: string[];
  readonly requiresCapabilities?: string[];
  readonly integrity?: string;
  readonly aliases?: Record<string, string>;
  readonly contributes?: {
    mode?: { key: string; label: string };
    views?: { slot: string; id: string; label: string; priority?: number }[];
    commands?: {
      id: string;
      label: string;
      keybinding?: string;
      when?: string;
      enabledInEditors?: boolean;
    }[];
    statusBarItems?: {
      id: string;
      label: string;
      alignment?: "left" | "right";
      priority?: number;
    }[];
    tools?: {
      id: string;
      description: string;
      parameters: JsonSchema;
    }[];
  };
  readonly configuration?: JsonSchema;
}

/** Minimal JSON Schema subset for tool parameters and configuration. */
export interface JsonSchema {
  type: string;
  properties?: Record<string, JsonSchema & { description?: string; default?: unknown; "x-ui"?: Record<string, unknown> }>;
  required?: string[];
  items?: JsonSchema;
  enum?: unknown[];
}

// --- Plugin lifecycle ---

export interface PluginDefinition {
  readonly manifest: PluginManifest;

  /**
   * Synchronous registration phase. Called on kernel boot.
   * Register services, views, event handlers. Must not do async work.
   * If this returns a Promise, the kernel treats it as an error.
   */
  register(ctx: PluginContext): void;

  /**
   * Async activation phase. Called after all plugins are registered.
   * Load data, render initial UI, subscribe to events.
   *
   * previousState is the value returned by deactivate() on the prior
   * lifecycle — used for hot-reload state restoration.
   */
  run(ctx: PluginContext, signal: AbortSignal, previousState?: unknown): Promise<void>;

  /** Optional cleanup. DisposableStore auto-disposes regardless. */
  deactivate?(): unknown | void;
}

// --- Health ---

export type PluginHealth = "healthy" | "warning" | "disabled";

// --- Content types ---

export type ContentPart =
  | { id: string; type: "text"; text: string }
  | { id: string; type: "code"; language: string; text: string }
  | { id: string; type: "tool_result"; toolName: string; mimeType: string; data: unknown }
  | { id: string; type: "image"; url: string; alt?: string };

// --- PluginContext (the facade) ---

export interface PluginContext {
  readonly manifest: PluginManifest;
  readonly container: HTMLElement;
  readonly log: Logger;

  // --- Core APIs (always available) ---
  readonly events: EventBus;
  readonly services: ServiceAccess;
  readonly lifecycle: LifecycleAccess;
  readonly subscriptions: SubscriptionAccess;

  // --- Extended APIs ---
  readonly layout: LayoutAccess;
  readonly hooks: HookPipeline;
  readonly tools: ToolRegistry;
  readonly commands: CommandRegistry;
  readonly network: NetworkAccess;
  readonly storage: StorageAccess;
  readonly configuration: ConfigurationAccess;
  readonly dialogs: StandardDialogs;
  readonly notifications: Notifications;
  readonly timers: ManagedTimers;
  readonly theme: ThemeAccess;
  readonly router: HashRouter;
  readonly assets: AssetResolver;
  readonly status: PluginStatus;
  readonly renderers: RendererRegistry;
  readonly observe: ManagedObservers;
  readonly mode: ModeAccess;
  readonly format: FormatAccess;
  readonly clipboard: ClipboardAccess;
  readonly download: DownloadAccess;
  readonly upload: UploadAccess;

}

// --- Lifecycle ---

export interface LifecycleAccess {
  /** AbortSignal cancelled on deactivation — wire into fetch, async work. */
  readonly signal: AbortSignal;
}

// --- Subscriptions ---

export interface SubscriptionAccess {
  /** Add an arbitrary Disposable for auto-cleanup on deactivation. */
  add(disposable: Disposable): void;
}

// --- EventBus ---

export interface EventBus {
  on<T = unknown>(event: string, handler: (data: T) => void): Disposable;
  once<T = unknown>(event: string, handler: (data: T) => void): Disposable;
  emit<T = unknown>(event: string, data: T): void;

  /** Attach a listener to `window`. Auto-removed on deactivation. */
  onWindow?(eventName: string, handler: EventListener, options?: AddEventListenerOptions): Disposable;
  /** Attach a listener to `document`. Auto-removed on deactivation. */
  onDocument?(eventName: string, handler: EventListener, options?: AddEventListenerOptions): Disposable;
}

// --- ServiceRegistry ---

export interface ServiceAccess {
  get<T = unknown>(id: string): T | undefined;
  provide<T = unknown>(id: string, instance: T): Disposable;
  /** Fires when the named service is registered or unregistered. */
  onDidChange?(serviceId: string, callback: (service: unknown | undefined) => void): Disposable;
}

// --- Logger ---

export interface Logger {
  debug?(message: string, ...args: unknown[]): void;
  info(message: string, ...args: unknown[]): void;
  warn(message: string, ...args: unknown[]): void;
  error(message: string, ...args: unknown[]): void;
}

// --- Hook Pipeline ---

export interface HookPipeline {
  on<T = unknown>(
    name: string,
    handler: (value: T) => T | { $block: true; reason: string } | void | Promise<T | { $block: true; reason: string } | void>,
    options?: { priority?: number },
  ): Disposable;
  run<T = unknown>(name: string, value: T): Promise<T | { $block: true; reason: string }>;
}

// --- Tool Registry ---

export interface ToolDefinition {
  description: string;
  parameters: JsonSchema;
  execute(args: Record<string, unknown>, signal: AbortSignal): Promise<ToolResult>;
  requiresUserApproval?: boolean;
  risk?: "low" | "medium" | "high";
  sideEffects?: boolean;
  maxModelBytes?: number;
}

export interface ToolResult {
  content: ContentPart[];
  modelContent?: ContentPart[];
  metadata?: unknown;
}

export interface ToolRegistry {
  register(id: string, definition: ToolDefinition): Disposable;
  get(id: string): ToolDefinition | undefined;
}

// --- Managed Timers ---

export interface ManagedTimers {
  setTimeout(callback: () => void, ms: number): Disposable;
  setInterval(callback: () => void, ms: number): Disposable;
  requestAnimationFrame(callback: FrameRequestCallback): Disposable;
}

// --- Standard Dialogs ---

export interface StandardDialogs {
  confirm(options: { title: string; message: string; destructive?: boolean }): Promise<boolean>;
  alert(options: { title: string; message: string }): Promise<void>;
  prompt(options: { title: string; message: string; defaultValue?: string }): Promise<string | null>;
  select(options: {
    title: string;
    items: { id: string; label: string; description?: string; icon?: string }[];
  }): Promise<string | null>;
}

// --- Hash Router ---

export interface HashRouter {
  getHash(): string;
  setHash(path: string, options?: { history?: "replace" | "push" }): void;
  onHashChange(callback: (hash: string) => void): Disposable;
}

// --- Asset Resolver ---

export interface AssetResolver {
  getUrl(path: string): string;
}

// --- Configuration Access ---

export interface ConfigurationAccess {
  get<T = unknown>(): T;
  onChange<T = unknown>(callback: (config: T) => void): Disposable;
}

// --- Plugin Status ---

export interface PluginStatus {
  setBusy(message?: string): void;
  setReady(): void;
}

// --- Managed Observers ---

export interface ManagedObservers {
  mutation(target: Node, callback: (records: MutationRecord[]) => void, options?: MutationObserverInit): Disposable;
  resize(target: Element, callback: (entries: ResizeObserverEntry[]) => void, options?: ResizeObserverOptions): Disposable;
  intersection(target: Element, callback: (entries: IntersectionObserverEntry[]) => void, options?: IntersectionObserverInit): Disposable;
}

// --- Notifications ---

export interface Notifications {
  info(message: string): void;
  success(message: string): void;
  warning(message: string): void;
  error(message: string): void;
}

// --- Theme Access ---

export interface ThemeAccess {
  readonly tokens: Record<string, string>;
  readonly activeThemeId: string;
  onChange(callback: () => void): Disposable;
  registerTheme(id: string, tokens: Record<string, string>): Disposable;
  getRegisteredThemes(): { id: string; name: string; category: string }[];
}

// --- Layout Access ---

export interface LayoutAccess {
  setTitle(title: string): void;
  registerView(slot: string, viewFactory: (shadowRoot: ShadowRoot) => void): Disposable;
  showPopover(options: {
    anchor: HTMLElement;
    content: HTMLElement;
    placement?: "top" | "bottom" | "left" | "right";
  }): Disposable;
}

// --- Command Registry ---

export interface CommandRegistry {
  register(id: string, handler: () => void, options?: { keybinding?: string; when?: string }): Disposable;
}

// --- Network Access ---

export interface NetworkAccess {
  fetch(url: string, init?: RequestInit): Promise<Response>;
}

// --- Storage Access ---

export interface StorageAccess {
  get<T = unknown>(key: string): Promise<T | null>;
  set(key: string, value: unknown): Promise<void>;
  delete(key: string): Promise<void>;
  keys(): Promise<string[]>;
  clear(): Promise<void>;
  onDidChange(callback: (key: string | null) => void): Disposable;
}

// --- Format Access ---

export interface FormatAccess {
  date(value: Date | string | number, style?: "short" | "medium" | "long"): string;
  dateTime(value: Date | string | number, style?: "short" | "medium" | "long"): string;
  number(value: number, options?: Intl.NumberFormatOptions): string;
  relativeTime(value: Date | string | number): string;
  duration(seconds: number): string;
}

// --- Mode Access ---

export interface ModeAccess {
  getActive(): string;
  switch(mode: string): void;
  onChange(handler: (data: { from: string; to: string }) => void): Disposable;
}

// --- Renderer Registry ---

export interface MessageRenderer {
  kinds?: ContentPart["type"][];
  canRender(part: ContentPart): number | false;
  mount(container: HTMLElement, part: ContentPart, signal: AbortSignal): RendererInstance;
}

export interface RendererInstance {
  update(part: ContentPart, isFinal: boolean): void | false;
  unmount(): void;
}

export interface RendererRegistry {
  register(renderer: MessageRenderer): Disposable;
  registerPreProcessor(fn: (text: string) => string): Disposable;
  applyPreProcessors(text: string): string;
  mountPart(partId: string, container: HTMLElement, part: ContentPart): void;
  updatePart(partId: string, part: ContentPart, isFinal: boolean): void;
  unmountPart(partId: string): void;
}

// --- Clipboard Access ---

export interface ClipboardAccess {
  writeText(text: string): Promise<void>;
}

// --- Download Access ---

export interface DownloadAccess {
  save(blob: Blob, filename: string): void;
}

// --- Upload Access ---

export interface UploadFileReference {
  id: string;
  filename: string;
  bytes: number;
  createdAt: number;
  purpose: string;
}

export interface UploadAccess {
  upload(file: File, purpose?: string): Promise<UploadFileReference>;
  get(fileId: string): Promise<UploadFileReference>;
  delete(fileId: string): Promise<void>;
  getContent(fileId: string): Promise<Blob>;
}
