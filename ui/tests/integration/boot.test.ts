import { describe, test, expect, beforeEach } from "bun:test";
import { EventBusImpl } from "../../src/kernel/system/event-bus.ts";
import { ServiceRegistry } from "../../src/kernel/registries/services.ts";
import { DisposableStore } from "../../src/kernel/core/disposable.ts";
import { settingsPlugin } from "../../src/plugins/settings/index.ts";
import { chatPlugin } from "../../src/plugins/chat/index.ts";
import type { PluginContext, Disposable } from "../../src/kernel/types.ts";

/**
 * Integration tests — verify cross-plugin wiring using real kernel
 * singletons (EventBus, ServiceRegistry, DisposableStore).
 *
 * Does NOT call bootKernel (installs security interceptors, freezes
 * intrinsics). Instead, directly calls plugin register() to verify
 * that service IDs, event names, and lifecycle contracts match.
 */

// -- Helpers -----------------------------------------------------------------

function createMinimalCtx(overrides: Record<string, unknown> = {}): PluginContext {
  return {
    services: { get: () => undefined, provide: () => ({ dispose() {} }) },
    subscriptions: { add: () => {} },
    log: { info: () => {}, warn: () => {}, error: () => {}, debug: () => {} },
    renderers: {
      register: () => ({ dispose: () => {} }),
      registerPreProcessor: () => ({ dispose: () => {} }),
    },
    ...overrides,
  } as unknown as PluginContext;
}

// ── Cross-plugin service registration ────────────────────────────────────────

describe("Cross-plugin service registration", () => {
  let eventBus: EventBusImpl;
  let services: ServiceRegistry;

  beforeEach(() => {
    eventBus = new EventBusImpl();
    services = new ServiceRegistry(eventBus);
  });

  test("settingsPlugin.register() provides talu.models with correct interface", () => {
    const ctx = createMinimalCtx({
      services,
      subscriptions: { add: () => {} },
    });
    settingsPlugin.register(ctx);

    const models = services.get<any>("talu.models");
    expect(models).toBeDefined();
    expect(typeof models.getActiveModel).toBe("function");
    expect(typeof models.getAvailableModels).toBe("function");
    expect(typeof models.setActiveModel).toBe("function");
    expect(typeof models.onChange).toBe("function");
  });

  test("chatPlugin.register() provides talu.chat with correct interface", () => {
    const ctx = createMinimalCtx({ services });
    chatPlugin.register(ctx);

    const chat = services.get<any>("talu.chat");
    expect(chat).toBeDefined();
    expect(typeof chat.selectChat).toBe("function");
    expect(typeof chat.startNewConversation).toBe("function");
    expect(typeof chat.getSessions).toBe("function");
    expect(typeof chat.getActiveSessionId).toBe("function");
  });

  test("both plugins' services coexist in same registry", () => {
    const settingsCtx = createMinimalCtx({ services });
    settingsPlugin.register(settingsCtx);

    const chatCtx = createMinimalCtx({ services });
    chatPlugin.register(chatCtx);

    expect(services.get("talu.models")).toBeDefined();
    expect(services.get("talu.chat")).toBeDefined();
  });

  test("disposing tracked service removes it from registry", () => {
    const tracked: Disposable[] = [];
    const ctx = createMinimalCtx({
      services,
      subscriptions: { add: (d: Disposable) => tracked.push(d) },
    });
    settingsPlugin.register(ctx);
    expect(services.get("talu.models")).toBeDefined();

    // Simulate plugin deactivation.
    for (const d of tracked) d.dispose();
    expect(services.get("talu.models")).toBeUndefined();
  });

  test("duplicate registration is rejected (no overwrite)", () => {
    const ctx = createMinimalCtx({ services });
    settingsPlugin.register(ctx);
    const firstService = services.get("talu.models");

    // Second registration attempt.
    settingsPlugin.register(ctx);
    expect(services.get("talu.models")).toBe(firstService);
  });
});

// ── Cross-plugin event contracts ─────────────────────────────────────────────

describe("Cross-plugin event contracts", () => {
  test("model.changed flows from settings emitter to chat listener", () => {
    const eventBus = new EventBusImpl();
    const received: unknown[] = [];

    // Chat would subscribe in run().
    eventBus.on("model.changed", (data) => received.push(data));

    // Settings emits after model change.
    eventBus.emit("model.changed", {
      modelId: "gpt-4",
      availableModels: [{ id: "gpt-4", defaults: {}, overrides: {} }],
    });

    expect(received.length).toBe(1);
    expect((received[0] as any).modelId).toBe("gpt-4");
    expect((received[0] as any).availableModels.length).toBe(1);
  });

  test("prompts.changed flows to chat listener", () => {
    const eventBus = new EventBusImpl();
    const received: unknown[] = [];

    eventBus.on("prompts.changed", (data) => received.push(data));
    eventBus.emit("prompts.changed", {
      prompts: [{ id: "p1", name: "Default" }],
      defaultId: "p1",
    });

    expect(received.length).toBe(1);
    expect((received[0] as any).prompts[0].name).toBe("Default");
    expect((received[0] as any).defaultId).toBe("p1");
  });

  test("event reaches multiple listeners (fan-out)", () => {
    const eventBus = new EventBusImpl();
    const calls: string[] = [];

    eventBus.on("model.changed", () => calls.push("chat"));
    eventBus.on("model.changed", () => calls.push("browser"));
    eventBus.on("model.changed", () => calls.push("custom"));

    eventBus.emit("model.changed", {});
    expect(calls).toEqual(["chat", "browser", "custom"]);
  });

  test("disposed listener stops receiving events", () => {
    const eventBus = new EventBusImpl();
    const calls: string[] = [];

    const sub = eventBus.on("model.changed", () => calls.push("received"));
    eventBus.emit("model.changed", {});
    expect(calls.length).toBe(1);

    sub.dispose();
    eventBus.emit("model.changed", {});
    expect(calls.length).toBe(1);
  });

  test("once listener fires exactly once", () => {
    const eventBus = new EventBusImpl();
    const calls: string[] = [];

    eventBus.once("model.changed", () => calls.push("once"));
    eventBus.emit("model.changed", {});
    eventBus.emit("model.changed", {});

    expect(calls).toEqual(["once"]);
  });
});

// ── Service change notifications ────────────────────────────────────────────

describe("Service change notifications", () => {
  test("onDidChange fires when service is registered", () => {
    const eventBus = new EventBusImpl();
    const services = new ServiceRegistry(eventBus);
    const changes: unknown[] = [];

    services.onDidChange("talu.models", (svc) => changes.push(svc));

    const impl = { getActiveModel: () => "test" };
    services.provide("talu.models", impl);

    expect(changes.length).toBe(1);
    expect(changes[0]).toBe(impl);
  });

  test("onDidChange fires with undefined when service is disposed", () => {
    const eventBus = new EventBusImpl();
    const services = new ServiceRegistry(eventBus);
    const changes: unknown[] = [];

    services.onDidChange("talu.models", (svc) => changes.push(svc));
    const disposable = services.provide("talu.models", { test: true });
    disposable.dispose();

    expect(changes.length).toBe(2);
    expect(changes[1]).toBeUndefined();
  });
});

// ── DisposableStore lifecycle integration ────────────────────────────────────

describe("DisposableStore lifecycle integration", () => {
  test("simulates full plugin lifecycle: register → consume → deactivate", () => {
    const eventBus = new EventBusImpl();
    const services = new ServiceRegistry(eventBus);
    const pluginStore = new DisposableStore();
    const received: unknown[] = [];

    // Phase 1: Register — provide service and subscribe to event.
    const serviceDisp = services.provide("test.models", { active: "gpt-4" });
    pluginStore.track(serviceDisp);

    const eventDisp = eventBus.on("test.event", (d) => received.push(d));
    pluginStore.track(eventDisp);

    // Phase 2: Consume — both work.
    expect(services.get("test.models")).toBeDefined();
    eventBus.emit("test.event", "hello");
    expect(received.length).toBe(1);

    // Phase 3: Deactivate — everything cleaned up.
    pluginStore.dispose();
    expect(services.get("test.models")).toBeUndefined();

    eventBus.emit("test.event", "world");
    expect(received.length).toBe(1); // listener gone
  });

  test("double dispose is safe (idempotent)", () => {
    const store = new DisposableStore();
    let disposeCount = 0;
    store.track({ dispose: () => disposeCount++ });

    store.dispose();
    store.dispose();
    expect(disposeCount).toBe(1);
  });

  test("tracking after dispose immediately disposes the item", () => {
    const store = new DisposableStore();
    store.dispose();

    let disposed = false;
    store.track({ dispose: () => { disposed = true; } });
    expect(disposed).toBe(true);
  });
});
