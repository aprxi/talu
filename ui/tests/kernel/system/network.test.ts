import { describe, test, expect, spyOn, beforeEach, afterEach } from "bun:test";
import { EventBusImpl } from "../../../src/kernel/system/event-bus.ts";
import { NetworkConnectivity, NetworkAccessImpl, NetworkError } from "../../../src/kernel/system/network.ts";

describe("NetworkError", () => {
  test("has correct name", () => {
    const err = new NetworkError("fail");
    expect(err.name).toBe("NetworkError");
  });

  test("preserves cause", () => {
    const cause = new TypeError("fetch failed");
    const err = new NetworkError("fail", cause);
    expect(err.cause).toBe(cause);
  });

  test("is instanceof Error", () => {
    expect(new NetworkError("fail") instanceof Error).toBe(true);
  });
});

describe("NetworkConnectivity", () => {
  test("emits system.server.disconnected on first failure", () => {
    const bus = new EventBusImpl();
    const connectivity = new NetworkConnectivity(bus);
    let emitted = false;
    bus.on("system.server.disconnected", () => { emitted = true; });
    connectivity.onFailure();
    expect(emitted).toBe(true);
  });

  test("does not emit duplicate disconnected events", () => {
    const bus = new EventBusImpl();
    const connectivity = new NetworkConnectivity(bus);
    let count = 0;
    bus.on("system.server.disconnected", () => { count++; });
    connectivity.onFailure();
    connectivity.onFailure();
    expect(count).toBe(1);
  });

  test("emits system.server.connected on recovery", () => {
    const bus = new EventBusImpl();
    const connectivity = new NetworkConnectivity(bus);
    let emitted = false;
    connectivity.onFailure(); // Go offline.
    bus.on("system.server.connected", () => { emitted = true; });
    connectivity.onSuccess(); // Recover.
    expect(emitted).toBe(true);
  });

  test("does not emit connected when already connected", () => {
    const bus = new EventBusImpl();
    const connectivity = new NetworkConnectivity(bus);
    let count = 0;
    bus.on("system.server.connected", () => { count++; });
    connectivity.onSuccess();
    connectivity.onSuccess();
    expect(count).toBe(0);
  });
});

describe("NetworkAccessImpl", () => {
  let originalFetch: typeof window.fetch;

  beforeEach(() => {
    originalFetch = window.fetch;
  });

  afterEach(() => {
    window.fetch = originalFetch;
  });

  test("injects X-Talu-Plugin-Id header", async () => {
    let capturedHeaders: Headers | undefined;
    window.fetch = async (_url: RequestInfo | URL, init?: RequestInit) => {
      capturedHeaders = new Headers(init?.headers);
      return new Response("ok");
    };

    const bus = new EventBusImpl();
    const connectivity = new NetworkConnectivity(bus);
    const net = new NetworkAccessImpl("my.plugin", null, connectivity);
    await net.fetch("/api/test");

    expect(capturedHeaders!.get("X-Talu-Plugin-Id")).toBe("my.plugin");
  });

  test("injects Authorization header when token provided", async () => {
    let capturedHeaders: Headers | undefined;
    window.fetch = async (_url: RequestInfo | URL, init?: RequestInit) => {
      capturedHeaders = new Headers(init?.headers);
      return new Response("ok");
    };

    const bus = new EventBusImpl();
    const connectivity = new NetworkConnectivity(bus);
    const net = new NetworkAccessImpl("my.plugin", "secret-token", connectivity);
    await net.fetch("/api/test");

    expect(capturedHeaders!.get("Authorization")).toBe("Bearer secret-token");
  });

  test("does not inject Authorization when token is null", async () => {
    let capturedHeaders: Headers | undefined;
    window.fetch = async (_url: RequestInfo | URL, init?: RequestInit) => {
      capturedHeaders = new Headers(init?.headers);
      return new Response("ok");
    };

    const bus = new EventBusImpl();
    const connectivity = new NetworkConnectivity(bus);
    const net = new NetworkAccessImpl("my.plugin", null, connectivity);
    await net.fetch("/api/test");

    expect(capturedHeaders!.has("Authorization")).toBe(false);
  });

  test("wraps fetch errors in NetworkError", async () => {
    window.fetch = async () => { throw new TypeError("Failed to fetch"); };

    const bus = new EventBusImpl();
    const connectivity = new NetworkConnectivity(bus);
    const net = new NetworkAccessImpl("my.plugin", null, connectivity);

    try {
      await net.fetch("/api/test");
      expect(true).toBe(false); // Should not reach here.
    } catch (err) {
      expect(err instanceof NetworkError).toBe(true);
      expect((err as NetworkError).message).toContain("my.plugin");
    }
  });

  test("tracks connectivity on success", async () => {
    window.fetch = async () => new Response("ok");

    const bus = new EventBusImpl();
    const connectivity = new NetworkConnectivity(bus);
    connectivity.onFailure(); // Go offline first.
    let reconnected = false;
    bus.on("system.server.connected", () => { reconnected = true; });

    const net = new NetworkAccessImpl("my.plugin", null, connectivity);
    await net.fetch("/api/test");

    expect(reconnected).toBe(true);
  });

  test("tracks connectivity on failure", async () => {
    window.fetch = async () => { throw new Error("network down"); };

    const bus = new EventBusImpl();
    const connectivity = new NetworkConnectivity(bus);
    let disconnected = false;
    bus.on("system.server.disconnected", () => { disconnected = true; });

    const net = new NetworkAccessImpl("my.plugin", null, connectivity);
    try { await net.fetch("/api/test"); } catch { /* expected */ }

    expect(disconnected).toBe(true);
  });

  test("does not mutate caller's init object", async () => {
    window.fetch = async () => new Response("ok");

    const bus = new EventBusImpl();
    const connectivity = new NetworkConnectivity(bus);
    const net = new NetworkAccessImpl("my.plugin", null, connectivity);
    const init: RequestInit = { method: "POST" };
    await net.fetch("/api/test", init);

    // Original init should not have headers added.
    expect(init.headers).toBeUndefined();
  });
});
