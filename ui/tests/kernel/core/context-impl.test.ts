import { describe, test, expect, beforeEach, spyOn } from "bun:test";
import { createPluginContext, type KernelInfrastructure } from "../../../src/kernel/core/context-impl.ts";
import { DisposableStore } from "../../../src/kernel/core/disposable.ts";
import { EventBusImpl } from "../../../src/kernel/system/event-bus.ts";
import { ServiceRegistry } from "../../../src/kernel/registries/services.ts";
import { HookPipelineImpl } from "../../../src/kernel/registries/hooks.ts";
import { ToolRegistryImpl } from "../../../src/kernel/registries/tools.ts";
import { CommandRegistryImpl } from "../../../src/kernel/registries/commands.ts";
import { ThemeAccessImpl } from "../../../src/kernel/ui/theme.ts";
import { PopoverManager } from "../../../src/kernel/ui/popover.ts";
import { RendererRegistryImpl } from "../../../src/kernel/registries/renderers.ts";
import { StatusBarManager } from "../../../src/kernel/ui/status-bar.ts";
import { ViewManager } from "../../../src/kernel/ui/view-manager.ts";
import { ModeManager } from "../../../src/kernel/ui/mode-manager.ts";
import { NetworkConnectivity } from "../../../src/kernel/system/network.ts";
import type { PluginManifest } from "../../../src/kernel/types.ts";

function makeInfra(): KernelInfrastructure {
  const eventBus = new EventBusImpl();
  return {
    eventBus,
    serviceRegistry: new ServiceRegistry(),
    hookPipeline: new HookPipelineImpl(),
    toolRegistry: new ToolRegistryImpl(),
    commandRegistry: new CommandRegistryImpl(),
    themeAccess: new ThemeAccessImpl(),
    popoverManager: new PopoverManager(),
    rendererRegistry: new RendererRegistryImpl(),
    statusBarManager: new StatusBarManager(),
    viewManager: new ViewManager(),
    modeManager: new ModeManager(eventBus),
    networkConnectivity: new NetworkConnectivity(eventBus),
  };
}

function builtinManifest(overrides: Partial<PluginManifest> = {}): PluginManifest {
  return {
    id: "talu.test",
    name: "Test Plugin",
    version: "1.0.0",
    builtin: true,
    ...overrides,
  } as PluginManifest;
}

function thirdPartyManifest(permissions: string[] = [], overrides: Partial<PluginManifest> = {}): PluginManifest {
  return {
    id: "ext.test",
    name: "Ext Test",
    version: "1.0.0",
    apiVersion: "1",
    permissions,
    ...overrides,
  } as PluginManifest;
}

describe("createPluginContext — permission gates", () => {
  let infra: KernelInfrastructure;

  beforeEach(() => {
    document.body.innerHTML = `<div id="status-bar"></div>`;
    infra = makeInfra();
  });

  // --- Builtin: all permissions ---

  test("builtin plugin can access network without declaration", async () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(builtinManifest(), document.createElement("div"), infra, disposables, new AbortController());
    // Calling fetch should NOT throw the permission error.
    // It may fail for network/HappyDOM reasons — that's fine, we're testing the gate.
    try {
      await ctx.network.fetch("/test");
    } catch (err) {
      // Permission errors contain "manifest.permissions".
      expect((err as Error).message).not.toContain("manifest.permissions");
    }
  });

  test("builtin plugin can access storage without declaration", async () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(builtinManifest(), document.createElement("div"), infra, disposables, new AbortController());
    // Calling storage.get should NOT throw the permission error.
    // May fail for non-permission reasons (HappyDOM fetch limitations).
    try {
      await ctx.storage.get("nonexistent-key");
    } catch (err) {
      expect((err as Error).message).not.toContain("manifest.permissions");
    }
  });

  test("builtin plugin can access clipboard without declaration", () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(builtinManifest(), document.createElement("div"), infra, disposables, new AbortController());
    // Calling writeText should NOT throw permission error.
    // It may reject for other reasons (no clipboard API in HappyDOM).
    const promise = ctx.clipboard.writeText("test");
    expect(promise).toBeInstanceOf(Promise);
  });

  // --- Third-party: permission gates ---

  test("third-party without 'network' permission → throws on fetch", async () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(thirdPartyManifest([]), document.createElement("div"), infra, disposables, new AbortController());
    try {
      await ctx.network.fetch("/api/test");
      expect(true).toBe(false); // Should not reach.
    } catch (err) {
      expect((err as Error).message).toContain("network");
      expect((err as Error).message).toContain("ext.test");
    }
  });

  test("third-party with 'network' permission → fetch allowed", async () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(thirdPartyManifest(["network"]), document.createElement("div"), infra, disposables, new AbortController());
    // Should NOT throw permission error (may fail for network/HappyDOM reasons).
    try {
      await ctx.network.fetch("/test");
    } catch (err) {
      expect((err as Error).message).not.toContain("manifest.permissions");
    }
  });

  test("third-party without 'storage' permission → throws on get", async () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(thirdPartyManifest([]), document.createElement("div"), infra, disposables, new AbortController());
    try {
      await ctx.storage.get("key");
      expect(true).toBe(false);
    } catch (err) {
      expect((err as Error).message).toContain("storage");
    }
  });

  test("third-party with 'storage' permission → storage allowed", async () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(thirdPartyManifest(["storage"]), document.createElement("div"), infra, disposables, new AbortController());
    // Should NOT throw permission error. May fail for other reasons (HappyDOM fetch).
    try {
      await ctx.storage.get("key");
    } catch (err) {
      // Non-permission errors (e.g. HappyDOM URL issues) are fine.
      // Permission errors contain "without declaring it in manifest.permissions".
      expect((err as Error).message).not.toContain("manifest.permissions");
    }
  });

  test("third-party without 'clipboard' permission → throws on writeText", async () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(thirdPartyManifest([]), document.createElement("div"), infra, disposables, new AbortController());
    try {
      await ctx.clipboard.writeText("data");
      expect(true).toBe(false);
    } catch (err) {
      expect((err as Error).message).toContain("clipboard");
    }
  });

  test("third-party without 'hooks' permission → throws on hooks.on", () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(thirdPartyManifest([]), document.createElement("div"), infra, disposables, new AbortController());
    expect(() => ctx.hooks.on("test.hook", (v) => v)).toThrow(/hooks/);
  });

  test("third-party without 'hooks' permission → throws on hooks.run", async () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(thirdPartyManifest([]), document.createElement("div"), infra, disposables, new AbortController());
    expect(() => ctx.hooks.run("test.hook", { a: 1 })).toThrow(/hooks/);
  });

  test("third-party with 'hooks' permission → hooks.on allowed", () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(thirdPartyManifest(["hooks"]), document.createElement("div"), infra, disposables, new AbortController());
    const d = ctx.hooks.on("test.hook", (v) => v);
    expect(typeof d.dispose).toBe("function");
    d.dispose();
  });

  test("third-party with 'hooks' permission → hooks.run allowed", async () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(thirdPartyManifest(["hooks"]), document.createElement("div"), infra, disposables, new AbortController());
    const out = await ctx.hooks.run("test.hook", { a: 1 });
    expect(out).toEqual({ a: 1 });
  });

  test("third-party without 'tools' permission → throws on tools.register", () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(thirdPartyManifest([]), document.createElement("div"), infra, disposables, new AbortController());
    expect(() => ctx.tools.register("my-tool", {
      description: "test",
      parameters: {},
      execute: async () => ({}),
    })).toThrow(/tools/);
  });

  test("third-party without 'download' permission → throws on save", () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(thirdPartyManifest([]), document.createElement("div"), infra, disposables, new AbortController());
    expect(() => ctx.download.save(new Blob([""]), "file.txt")).toThrow(/download/);
  });

  test("third-party without 'upload' permission → throws on upload", async () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(thirdPartyManifest([]), document.createElement("div"), infra, disposables, new AbortController());
    await expect(ctx.upload.upload(new File(["x"], "x.txt", { type: "text/plain" }))).rejects.toThrow(/upload/);
  });

  test("third-party with 'upload' permission → upload allowed", async () => {
    const disposables = new DisposableStore();
    const fetchSpy = spyOn(window, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "file_1",
          object: "file",
          bytes: 1,
          created_at: 123,
          filename: "x.txt",
          purpose: "assistants",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    const ctx = createPluginContext(thirdPartyManifest(["upload"]), document.createElement("div"), infra, disposables, new AbortController());

    const uploaded = await ctx.upload.upload(new File(["x"], "x.txt", { type: "text/plain" }));
    expect(uploaded.id).toBe("file_1");
    expect(uploaded.filename).toBe("x.txt");
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    fetchSpy.mockRestore();
  });

  test("third-party with 'upload' permission → get/delete/getContent allowed", async () => {
    const disposables = new DisposableStore();
    const fetchSpy = spyOn(window, "fetch")
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            id: "file_2",
            object: "file",
            bytes: 3,
            created_at: 456,
            filename: "a.bin",
            purpose: "assistants",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
      .mockResolvedValueOnce(new Response(new Blob(["abc"]), { status: 200 }));

    const ctx = createPluginContext(thirdPartyManifest(["upload"]), document.createElement("div"), infra, disposables, new AbortController());

    const file = await ctx.upload.get("file_2");
    expect(file.id).toBe("file_2");
    expect(file.createdAt).toBe(456);

    await ctx.upload.delete("file_2");

    const content = await ctx.upload.getContent("file_2");
    expect(await content.text()).toBe("abc");
    expect(fetchSpy).toHaveBeenCalledTimes(3);
    fetchSpy.mockRestore();
  });
});

describe("createPluginContext — frozen context", () => {
  let infra: KernelInfrastructure;

  beforeEach(() => {
    document.body.innerHTML = `<div id="status-bar"></div>`;
    infra = makeInfra();
  });

  test("context object is frozen", () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(builtinManifest(), document.createElement("div"), infra, disposables, new AbortController());
    expect(Object.isFrozen(ctx)).toBe(true);
  });

  test("manifest is frozen", () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(builtinManifest(), document.createElement("div"), infra, disposables, new AbortController());
    expect(Object.isFrozen(ctx.manifest)).toBe(true);
  });
});

describe("createPluginContext — subsystem wiring", () => {
  let infra: KernelInfrastructure;

  beforeEach(() => {
    document.body.innerHTML = `<div id="status-bar"></div>`;
    infra = makeInfra();
  });

  test("events.emit and events.on work through shared bus", () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(builtinManifest(), document.createElement("div"), infra, disposables, new AbortController());
    let received: unknown;
    ctx.events.on("test.event", (data) => { received = data; });
    ctx.events.emit("test.event", { hello: "world" });
    expect(received).toEqual({ hello: "world" });
  });

  test("lifecycle.signal is from provided AbortController", () => {
    const disposables = new DisposableStore();
    const ac = new AbortController();
    const ctx = createPluginContext(builtinManifest(), document.createElement("div"), infra, disposables, ac);
    expect(ctx.lifecycle.signal).toBe(ac.signal);
  });

  test("subscriptions.add tracks disposables", () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(builtinManifest(), document.createElement("div"), infra, disposables, new AbortController());
    let disposed = false;
    ctx.subscriptions.add({ dispose: () => { disposed = true; } });
    disposables.dispose();
    expect(disposed).toBe(true);
  });

  test("commands.register namespaces command ID", () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(builtinManifest({ id: "talu.chat" }), document.createElement("div"), infra, disposables, new AbortController());
    let executed = false;
    ctx.commands.register("send", () => { executed = true; });
    // Should be registered with namespaced ID.
    infra.commandRegistry.execute("talu.chat.send");
    expect(executed).toBe(true);
  });

  test("status starts not busy", () => {
    const disposables = new DisposableStore();
    const ctx = createPluginContext(builtinManifest(), document.createElement("div"), infra, disposables, new AbortController());
    expect(ctx.status.isBusy).toBe(false);
  });
});
