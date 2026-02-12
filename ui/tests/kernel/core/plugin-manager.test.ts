import { describe, test, expect, beforeEach, afterEach, spyOn } from "bun:test";
import { bootKernel } from "../../../src/kernel/core/plugin-manager.ts";
import type { PluginDefinition, PluginContext } from "../../../src/kernel/types.ts";

/**
 * Integration tests for bootKernel — the kernel's top-level orchestrator.
 *
 * These test the full boot lifecycle: plugin registration, run, safe mode,
 * conflict detection, and deactivation — through the public bootKernel API.
 */

/** Minimal kernel chrome DOM required by bootKernel. */
function setupKernelDOM(): void {
  document.body.innerHTML = `
    <div id="activity-bar">
      <button class="activity-btn active" data-mode="chat">Chat</button>
    </div>
    <div class="app-layout" data-mode="chat">
      <div class="app-content">
        <div id="chat-mode" class="mode-view"></div>
      </div>
    </div>
    <div id="status-bar"></div>
    <div class="topbar"><span id="topbar-title"></span></div>
    <div id="toast-container"></div>
    <div id="safe-mode-fallback"></div>
  `;
}

/** Create a minimal valid builtin plugin definition. */
function makePlugin(
  id: string,
  overrides: {
    register?: (ctx: PluginContext) => void;
    run?: (ctx: PluginContext, signal: AbortSignal) => Promise<void>;
    deactivate?: () => unknown;
    manifest?: Partial<PluginDefinition["manifest"]>;
  } = {},
): PluginDefinition {
  return {
    manifest: {
      id,
      name: id,
      version: "1.0.0",
      builtin: true,
      ...(overrides.manifest ?? {}),
    },
    register: overrides.register ?? (() => {}),
    run: overrides.run ?? (async () => {}),
    deactivate: overrides.deactivate,
  } as PluginDefinition;
}

let originalFetch: typeof window.fetch;
let warnSpy: ReturnType<typeof spyOn>;
let infoSpy: ReturnType<typeof spyOn>;
let errorSpy: ReturnType<typeof spyOn>;

describe("bootKernel", () => {
  beforeEach(() => {
    setupKernelDOM();
    originalFetch = window.fetch;
    // Mock fetch: stylesheet returns empty CSS, plugin list returns 404.
    window.fetch = async (url: RequestInfo | URL) => {
      const u = String(url);
      if (u === "/assets/style.css") return new Response("", { status: 200 });
      if (u === "/v1/plugins") return new Response("", { status: 404 });
      return new Response("", { status: 404 });
    };
    // Suppress console noise from boot.
    warnSpy = spyOn(console, "warn").mockImplementation(() => {});
    infoSpy = spyOn(console, "info").mockImplementation(() => {});
    errorSpy = spyOn(console, "error").mockImplementation(() => {});
    // HappyDOM lacks window.print — polyfill so sensitive-apis doesn't crash.
    if (!window.print) {
      (window as { print?: () => void }).print = () => {};
    }
    localStorage.removeItem("theme");
    sessionStorage.removeItem("talu.kernel.loading");
  });

  afterEach(() => {
    window.fetch = originalFetch;
    warnSpy.mockRestore();
    infoSpy.mockRestore();
    errorSpy.mockRestore();
  });

  test("boots with no plugins", async () => {
    await bootKernel([]);
    // Boot complete message logged.
    expect(infoSpy.mock.calls.some(
      (call) => typeof call[0] === "string" && call[0].includes("Boot complete"),
    )).toBe(true);
  });

  test("calls register and run on builtin plugins", async () => {
    let registered = false;
    let ran = false;
    const plugin = makePlugin("talu.test", {
      register: () => { registered = true; },
      run: async () => { ran = true; },
    });

    await bootKernel([plugin]);
    expect(registered).toBe(true);
    expect(ran).toBe(true);
  });

  test("register receives a PluginContext with correct manifest", async () => {
    let receivedCtx: PluginContext | null = null;
    const plugin = makePlugin("talu.myplug", {
      register: (ctx) => { receivedCtx = ctx; },
    });

    await bootKernel([plugin]);
    expect(receivedCtx).not.toBeNull();
    expect(receivedCtx!.manifest.id).toBe("talu.myplug");
    expect(Object.isFrozen(receivedCtx!)).toBe(true);
  });

  test("run receives abort signal", async () => {
    let receivedSignal: AbortSignal | null = null;
    const plugin = makePlugin("talu.test", {
      run: async (_ctx, signal) => { receivedSignal = signal; },
    });

    await bootKernel([plugin]);
    expect(receivedSignal).not.toBeNull();
    expect(receivedSignal!.aborted).toBe(false);
  });

  test("multiple plugins are registered in order", async () => {
    const order: string[] = [];
    const pluginA = makePlugin("talu.a", {
      register: () => { order.push("a"); },
    });
    const pluginB = makePlugin("talu.b", {
      register: () => { order.push("b"); },
    });

    await bootKernel([pluginA, pluginB]);
    expect(order).toEqual(["a", "b"]);
  });

  test("register error does not prevent other plugins from loading", async () => {
    let bRegistered = false;
    const pluginA = makePlugin("talu.a", {
      register: () => { throw new Error("register boom"); },
    });
    const pluginB = makePlugin("talu.b", {
      register: () => { bRegistered = true; },
    });

    await bootKernel([pluginA, pluginB]);
    expect(bRegistered).toBe(true);
  });

  test("run error does not prevent other plugins from running", async () => {
    let bRan = false;
    const pluginA = makePlugin("talu.a", {
      run: async () => { throw new Error("run boom"); },
    });
    const pluginB = makePlugin("talu.b", {
      run: async () => { bRan = true; },
    });

    await bootKernel([pluginA, pluginB]);
    expect(bRan).toBe(true);
  });

  test("clears crash detection flag on successful boot", async () => {
    sessionStorage.setItem("talu.kernel.loading", "crashed");
    // isSafeMode() will consume the flag, but setLoadingFlag re-sets it,
    // and clearLoadingFlag at end should remove it.
    await bootKernel([]);
    expect(sessionStorage.getItem("talu.kernel.loading")).toBeNull();
  });

  test("removes safe-mode-fallback element after boot", async () => {
    expect(document.getElementById("safe-mode-fallback")).not.toBeNull();
    await bootKernel([]);
    expect(document.getElementById("safe-mode-fallback")).toBeNull();
  });

  test("sets up accessibility on activity bar", async () => {
    await bootKernel([]);
    const bar = document.getElementById("activity-bar")!;
    expect(bar.getAttribute("role")).toBe("tablist");
  });

  test("sets up status bar ARIA", async () => {
    await bootKernel([]);
    const bar = document.getElementById("status-bar")!;
    expect(bar.getAttribute("role")).toBe("status");
  });
});
