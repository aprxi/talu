import { describe, test, expect, spyOn, beforeEach, afterEach } from "bun:test";
import { StorageFacadeImpl } from "../../../src/kernel/system/storage.ts";
import { MAX_DOCUMENT_BYTES } from "../../../src/kernel/security/resource-caps.ts";

/**
 * Tests for StorageFacadeImpl — the server-backed storage used by third-party
 * plugins. All fetch calls are mocked to test REST API mapping, headers,
 * size enforcement, and change notifications in isolation.
 */

let storage: StorageFacadeImpl;
let originalFetch: typeof window.fetch;
let fetchCalls: { url: string; init?: RequestInit }[];

/** Track all fetch calls and return a configurable response. */
let fetchHandler: (url: string, init?: RequestInit) => Response;

beforeEach(() => {
  originalFetch = window.fetch;
  fetchCalls = [];

  // Default: return empty doc list for GET, 200 for mutations.
  fetchHandler = () => new Response("[]", { status: 200 });

  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    fetchCalls.push({ url, init });
    return fetchHandler(url, init);
  };

  storage = new StorageFacadeImpl("ext.myplugin", "test-token-123");
});

afterEach(() => {
  storage.dispose();
  window.fetch = originalFetch;
});

// ── Headers ─────────────────────────────────────────────────────────────────

describe("StorageFacadeImpl — headers", () => {
  test("includes X-Talu-Plugin-Id on every request", async () => {
    await storage.get("key1");
    expect(fetchCalls.length).toBeGreaterThanOrEqual(1);
    const headers = fetchCalls[0]!.init?.headers as Record<string, string>;
    expect(headers["X-Talu-Plugin-Id"]).toBe("ext.myplugin");
  });

  test("includes Authorization Bearer token when token is provided", async () => {
    await storage.get("key1");
    const headers = fetchCalls[0]!.init?.headers as Record<string, string>;
    expect(headers["Authorization"]).toBe("Bearer test-token-123");
  });

  test("omits Authorization when token is null", async () => {
    storage.dispose();
    storage = new StorageFacadeImpl("ext.noauth", null);
    await storage.get("key1");
    const headers = fetchCalls[0]!.init?.headers as Record<string, string>;
    expect(headers["Authorization"]).toBeUndefined();
    expect(headers["X-Talu-Plugin-Id"]).toBe("ext.noauth");
  });

  test("includes Content-Type application/json", async () => {
    await storage.get("key1");
    const headers = fetchCalls[0]!.init?.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
  });
});

// ── GET ─────────────────────────────────────────────────────────────────────

describe("StorageFacadeImpl — get", () => {
  test("fetches correct URL with query params", async () => {
    await storage.get("mykey");
    const url = fetchCalls[0]!.url;
    expect(url).toContain("/v1/db/tables/documents?");
    expect(url).toContain("type=plugin_storage");
    expect(url).toContain("owner_id=ext.myplugin");
    expect(url).toContain("title=mykey");
  });

  test("returns content from first matching document", async () => {
    fetchHandler = () =>
      new Response(JSON.stringify([{ id: "doc1", content: { hello: "world" } }]), { status: 200 });
    const result = await storage.get("mykey");
    expect(result).toEqual({ hello: "world" });
  });

  test("returns null when no documents match", async () => {
    fetchHandler = () => new Response("[]", { status: 200 });
    const result = await storage.get("missing");
    expect(result).toBeNull();
  });

  test("returns null on non-ok response", async () => {
    fetchHandler = () => new Response("", { status: 500 });
    const result = await storage.get("key");
    expect(result).toBeNull();
  });
});

// ── SET ─────────────────────────────────────────────────────────────────────

describe("StorageFacadeImpl — set", () => {
  test("creates new document via POST when key does not exist", async () => {
    // findDocId returns empty → POST.
    fetchHandler = (url, init) => {
      if (!init?.method) return new Response("[]", { status: 200 }); // GET (findDocId)
      return new Response("", { status: 201 }); // POST
    };

    await storage.set("newkey", { data: 42 });

    // First call: findDocId (GET), second call: POST.
    const postCall = fetchCalls.find((c) => c.init?.method === "POST");
    expect(postCall).not.toBeUndefined();
    expect(postCall!.url).toBe("/v1/db/tables/documents");
    const body = JSON.parse(postCall!.init!.body as string);
    expect(body.type).toBe("plugin_storage");
    expect(body.owner_id).toBe("ext.myplugin");
    expect(body.title).toBe("newkey");
    expect(body.content).toEqual({ data: 42 });
  });

  test("updates existing document via PATCH when key exists", async () => {
    fetchHandler = (url, init) => {
      if (!init?.method) {
        // GET (findDocId) — return existing doc.
        return new Response(JSON.stringify([{ id: "existing-id" }]), { status: 200 });
      }
      return new Response("", { status: 200 }); // PATCH
    };

    await storage.set("existingkey", "updated");

    const patchCall = fetchCalls.find((c) => c.init?.method === "PATCH");
    expect(patchCall).not.toBeUndefined();
    expect(patchCall!.url).toBe("/v1/db/tables/documents/existing-id");
    const body = JSON.parse(patchCall!.init!.body as string);
    expect(body.content).toBe("updated");
  });

  test("throws when payload exceeds MAX_DOCUMENT_BYTES", async () => {
    // Create a string that will exceed the limit when JSON-serialized.
    const oversized = "x".repeat(MAX_DOCUMENT_BYTES + 1);
    try {
      await storage.set("big", oversized);
      expect(true).toBe(false); // Should not reach.
    } catch (err) {
      expect((err as Error).message).toContain("size limit");
      expect((err as Error).message).toContain(String(MAX_DOCUMENT_BYTES));
    }
    // Verify no fetch was made (size check is pre-flight).
    expect(fetchCalls.length).toBe(0);
  });

  test("allows payload exactly at MAX_DOCUMENT_BYTES", async () => {
    // JSON.stringify({ content: value }) — we need the total to be <= MAX_DOCUMENT_BYTES.
    // The overhead is '{"content":""}' = 14 bytes, so value can be MAX - 14 chars.
    const value = "x".repeat(MAX_DOCUMENT_BYTES - 14);
    fetchHandler = () => new Response("[]", { status: 200 });
    // Should not throw.
    await storage.set("exact", value);
    expect(fetchCalls.length).toBeGreaterThanOrEqual(1);
  });
});

// ── DELETE ───────────────────────────────────────────────────────────────────

describe("StorageFacadeImpl — delete", () => {
  test("sends DELETE request for existing document", async () => {
    fetchHandler = (url, init) => {
      if (!init?.method) {
        return new Response(JSON.stringify([{ id: "del-id" }]), { status: 200 });
      }
      return new Response("", { status: 200 });
    };

    await storage.delete("mykey");

    const deleteCall = fetchCalls.find((c) => c.init?.method === "DELETE");
    expect(deleteCall).not.toBeUndefined();
    expect(deleteCall!.url).toBe("/v1/db/tables/documents/del-id");
  });

  test("no DELETE sent when key does not exist", async () => {
    fetchHandler = () => new Response("[]", { status: 200 });
    await storage.delete("missing");
    const deleteCall = fetchCalls.find((c) => c.init?.method === "DELETE");
    expect(deleteCall).toBeUndefined();
  });

  test("no change notification when key does not exist", async () => {
    fetchHandler = () => new Response("[]", { status: 200 });
    let notified = false;
    storage.onDidChange(() => { notified = true; });
    await storage.delete("nonexistent");
    expect(notified).toBe(false);
  });
});

// ── KEYS ────────────────────────────────────────────────────────────────────

describe("StorageFacadeImpl — keys", () => {
  test("returns titles from document list", async () => {
    fetchHandler = () =>
      new Response(
        JSON.stringify([{ title: "a" }, { title: "b" }, { title: "c" }]),
        { status: 200 },
      );
    const keys = await storage.keys();
    expect(keys).toEqual(["a", "b", "c"]);
  });

  test("fetches with correct query params (no title filter)", async () => {
    await storage.keys();
    const url = fetchCalls[0]!.url;
    expect(url).toContain("type=plugin_storage");
    expect(url).toContain("owner_id=ext.myplugin");
    expect(url).not.toContain("title=");
  });

  test("returns empty array on non-ok response", async () => {
    fetchHandler = () => new Response("", { status: 500 });
    const keys = await storage.keys();
    expect(keys).toEqual([]);
  });

  test("returns empty array when response is not an array", async () => {
    fetchHandler = () => new Response("{}", { status: 200 });
    const keys = await storage.keys();
    expect(keys).toEqual([]);
  });
});

// ── CLEAR ───────────────────────────────────────────────────────────────────

describe("StorageFacadeImpl — clear", () => {
  test("deletes each key individually", async () => {
    let callIndex = 0;
    fetchHandler = (url, init) => {
      if (!init?.method) {
        callIndex++;
        // First GET (keys): return list of docs.
        if (callIndex === 1) {
          return new Response(
            JSON.stringify([{ title: "a" }, { title: "b" }]),
            { status: 200 },
          );
        }
        // Subsequent GETs (findDocId for each delete): return doc with id.
        return new Response(
          JSON.stringify([{ id: `id-${callIndex}` }]),
          { status: 200 },
        );
      }
      return new Response("", { status: 200 });
    };

    await storage.clear();

    const deleteCalls = fetchCalls.filter((c) => c.init?.method === "DELETE");
    expect(deleteCalls.length).toBe(2);
  });

  test("clear with no keys still notifies with null", async () => {
    fetchHandler = () => new Response("[]", { status: 200 });
    let receivedKey: string | null | undefined = "unset";
    storage.onDidChange((key) => { receivedKey = key; });
    await storage.clear();
    // Source lines 135-136: broadcastChange(null) + notifyChange(null) fire unconditionally.
    expect(receivedKey).toBeNull();
  });
});

// ── Change notifications ────────────────────────────────────────────────────

describe("StorageFacadeImpl — change notifications", () => {
  test("set notifies onDidChange with key", async () => {
    fetchHandler = () => new Response("[]", { status: 200 });
    let receivedKey: string | null | undefined;
    storage.onDidChange((key) => { receivedKey = key; });
    await storage.set("mykey", "val");
    expect(receivedKey).toBe("mykey");
  });

  test("delete notifies onDidChange with key", async () => {
    fetchHandler = (url, init) => {
      if (!init?.method) return new Response(JSON.stringify([{ id: "d1" }]), { status: 200 });
      return new Response("", { status: 200 });
    };
    let receivedKey: string | null | undefined;
    storage.onDidChange((key) => { receivedKey = key; });
    await storage.delete("mykey");
    expect(receivedKey).toBe("mykey");
  });

  test("clear notifies onDidChange with null", async () => {
    fetchHandler = () => new Response("[]", { status: 200 });
    let receivedKey: string | null | undefined = "unset";
    storage.onDidChange((key) => { receivedKey = key; });
    await storage.clear();
    expect(receivedKey).toBeNull();
  });

  test("onDidChange dispose stops notifications", async () => {
    fetchHandler = () => new Response("[]", { status: 200 });
    let callCount = 0;
    const d = storage.onDidChange(() => { callCount++; });
    await storage.set("a", 1);
    expect(callCount).toBe(1);
    d.dispose();
    await storage.set("b", 2);
    expect(callCount).toBe(1);
  });

  test("change callback error does not break storage", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    fetchHandler = () => new Response("[]", { status: 200 });
    storage.onDidChange(() => { throw new Error("cb boom"); });
    await storage.set("a", 1);
    // Error is caught and logged, not propagated.
    expect(spy).toHaveBeenCalled();
    expect(spy.mock.calls[0]![0]).toContain("ext.myplugin");
    spy.mockRestore();
  });
});

// ── Lifecycle ───────────────────────────────────────────────────────────────

describe("StorageFacadeImpl — lifecycle", () => {
  test("dispose clears callbacks", async () => {
    let callCount = 0;
    storage.onDidChange(() => { callCount++; });
    storage.dispose();
    expect(callCount).toBe(0);
  });
});
