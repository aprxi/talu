import { describe, test, expect, beforeEach } from "bun:test";
import { createApiClient, type ApiClient } from "../src/api.ts";

/**
 * Tests for createApiClient — the gateway for all data IO.
 *
 * Strategy: mock fetchFn, verify URL paths, query params, HTTP methods,
 * headers, bodies, error handling, and the createResponse special case.
 */

let client: ApiClient;
let calls: { url: string; init?: RequestInit }[];
let mockResponse: Response;

function mockFetch(url: string, init?: RequestInit): Promise<Response> {
  calls.push({ url, init });
  return Promise.resolve(mockResponse);
}

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    statusText: status === 200 ? "OK" : "Error",
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  calls = [];
  mockResponse = jsonResponse({});
  client = createApiClient(mockFetch);
});

// ── URL construction & HTTP methods ─────────────────────────────────────────

describe("ApiClient — URL paths and methods", () => {
  test("listConversations → GET /v1/conversations with limit", async () => {
    await client.listConversations();
    expect(calls[0]!.url).toContain("/v1/conversations?");
    expect(calls[0]!.url).toContain("limit=20");
    expect(calls[0]!.init?.method).toBe("GET");
  });

  test("listConversations with cursor and custom limit", async () => {
    await client.listConversations("abc123", 50);
    expect(calls[0]!.url).toContain("cursor=abc123");
    expect(calls[0]!.url).toContain("limit=50");
  });

  test("listConversations without cursor omits cursor param", async () => {
    await client.listConversations(null);
    expect(calls[0]!.url).not.toContain("cursor");
  });

  test("search → POST /v1/search with body", async () => {
    const req = { scope: "conversations" as const, text: "hello" };
    await client.search(req);
    expect(calls[0]!.url).toBe("/v1/search");
    expect(calls[0]!.init?.method).toBe("POST");
    expect(JSON.parse(calls[0]!.init!.body as string)).toEqual(req);
  });

  test("getConversation → GET with URL-encoded id", async () => {
    await client.getConversation("id/with special&chars");
    expect(calls[0]!.url).toBe(`/v1/conversations/${encodeURIComponent("id/with special&chars")}`);
    expect(calls[0]!.init?.method).toBe("GET");
  });

  test("patchConversation → PATCH with body", async () => {
    const patch = { title: "new title" };
    await client.patchConversation("conv-1", patch);
    expect(calls[0]!.url).toBe("/v1/conversations/conv-1");
    expect(calls[0]!.init?.method).toBe("PATCH");
    expect(JSON.parse(calls[0]!.init!.body as string)).toEqual(patch);
  });

  test("deleteConversation → DELETE", async () => {
    mockResponse = new Response(null, { status: 204 });
    await client.deleteConversation("conv-1");
    expect(calls[0]!.init?.method).toBe("DELETE");
  });

  test("batchConversations → POST /v1/conversations/batch", async () => {
    const req = { ids: ["a", "b"], action: "delete" };
    await client.batchConversations(req as any);
    expect(calls[0]!.url).toBe("/v1/conversations/batch");
    expect(calls[0]!.init?.method).toBe("POST");
  });

  test("forkConversation → POST /v1/conversations/:id/fork", async () => {
    await client.forkConversation("conv-1", { target_item_id: 5 } as any);
    expect(calls[0]!.url).toBe("/v1/conversations/conv-1/fork");
    expect(calls[0]!.init?.method).toBe("POST");
  });

  test("getSettings → GET /v1/settings", async () => {
    await client.getSettings();
    expect(calls[0]!.url).toBe("/v1/settings");
    expect(calls[0]!.init?.method).toBe("GET");
  });

  test("patchSettings → PATCH /v1/settings", async () => {
    await client.patchSettings({ model: "gpt-4" } as any);
    expect(calls[0]!.init?.method).toBe("PATCH");
  });

  test("resetModelOverrides → DELETE /v1/settings/models/:id", async () => {
    mockResponse = new Response(null, { status: 204 });
    await client.resetModelOverrides("gpt-4");
    expect(calls[0]!.url).toBe("/v1/settings/models/gpt-4");
    expect(calls[0]!.init?.method).toBe("DELETE");
  });

  test("listDocuments without type → GET /v1/documents", async () => {
    await client.listDocuments();
    expect(calls[0]!.url).toBe("/v1/documents");
  });

  test("listDocuments with type → GET /v1/documents?type=...", async () => {
    await client.listDocuments("prompt");
    expect(calls[0]!.url).toBe("/v1/documents?type=prompt");
  });

  test("getDocument → GET with URL-encoded id", async () => {
    await client.getDocument("doc-1");
    expect(calls[0]!.url).toBe("/v1/documents/doc-1");
  });

  test("createDocument → POST /v1/documents", async () => {
    await client.createDocument({ type: "prompt", title: "Test" } as any);
    expect(calls[0]!.url).toBe("/v1/documents");
    expect(calls[0]!.init?.method).toBe("POST");
  });

  test("updateDocument → PATCH /v1/documents/:id", async () => {
    await client.updateDocument("doc-1", { title: "Updated" } as any);
    expect(calls[0]!.url).toBe("/v1/documents/doc-1");
    expect(calls[0]!.init?.method).toBe("PATCH");
  });

  test("deleteDocument → DELETE /v1/documents/:id", async () => {
    mockResponse = new Response(null, { status: 204 });
    await client.deleteDocument("doc-1");
    expect(calls[0]!.url).toBe("/v1/documents/doc-1");
    expect(calls[0]!.init?.method).toBe("DELETE");
  });
});

// ── Headers ─────────────────────────────────────────────────────────────────

describe("ApiClient — headers", () => {
  test("POST/PATCH sends Content-Type application/json", async () => {
    await client.search({ scope: "conversations" });
    const headers = calls[0]!.init?.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
  });

  test("GET does not send Content-Type", async () => {
    await client.getSettings();
    expect(calls[0]!.init?.headers).toBeUndefined();
  });
});

// ── Response parsing ────────────────────────────────────────────────────────

describe("ApiClient — response parsing", () => {
  test("200 with JSON body → ok: true with data", async () => {
    mockResponse = jsonResponse({ id: "conv-1", title: "Hello" });
    const result = await client.getConversation("conv-1");
    expect(result.ok).toBe(true);
    expect(result.data).toEqual({ id: "conv-1", title: "Hello" });
  });

  test("204 no content → ok: true without data", async () => {
    mockResponse = new Response(null, { status: 204 });
    const result = await client.deleteConversation("conv-1");
    expect(result.ok).toBe(true);
    expect(result.data).toBeUndefined();
  });

  test("error with JSON body → extracts error.message", async () => {
    mockResponse = jsonResponse({ error: { message: "Not found" } }, 404);
    const result = await client.getConversation("missing");
    expect(result.ok).toBe(false);
    expect(result.error).toBe("Not found");
  });

  test("error with non-JSON body → falls back to status text", async () => {
    mockResponse = new Response("Internal Server Error", {
      status: 500,
      statusText: "Internal Server Error",
    });
    const result = await client.getSettings();
    expect(result.ok).toBe(false);
    expect(result.error).toBe("500 Internal Server Error");
  });

  test("fetch network error → captures error message", async () => {
    const failClient = createApiClient(() => Promise.reject(new Error("Network failure")));
    const result = await failClient.getSettings();
    expect(result.ok).toBe(false);
    expect(result.error).toBe("Network failure");
  });

  test("fetch non-Error throw → stringifies", async () => {
    const failClient = createApiClient(() => Promise.reject("string error"));
    const result = await failClient.getSettings();
    expect(result.ok).toBe(false);
    expect(result.error).toBe("string error");
  });

  test("error with non-standard JSON shape → falls back to status text", async () => {
    // Server returns JSON but not { error: { message } } shape.
    mockResponse = jsonResponse({ message: "Bad request" }, 400);
    const result = await client.getConversation("bad");
    expect(result.ok).toBe(false);
    // err?.error?.message is undefined → falls through to status text.
    expect(result.error).toBe("400 Error");
  });

  test("resetModelOverrides does not URL-encode modelId", async () => {
    // Unlike other endpoints, resetModelOverrides passes modelId without encodeURIComponent.
    // This documents current behavior — IDs with / or special chars will produce wrong URLs.
    mockResponse = new Response(null, { status: 204 });
    await client.resetModelOverrides("org/model-name");
    expect(calls[0]!.url).toBe("/v1/settings/models/org/model-name");
  });
});

// ── createResponse special case ─────────────────────────────────────────────

describe("ApiClient — createResponse", () => {
  test("injects stream: true and store: true into body", async () => {
    const body = { model: "gpt-4", input: "hello" };
    await client.createResponse(body as any);
    const sent = JSON.parse(calls[0]!.init!.body as string);
    expect(sent.stream).toBe(true);
    expect(sent.store).toBe(true);
    expect(sent.model).toBe("gpt-4");
    expect(sent.input).toBe("hello");
  });

  test("sends POST to /v1/responses", async () => {
    await client.createResponse({ model: "gpt-4" } as any);
    expect(calls[0]!.url).toBe("/v1/responses");
    expect(calls[0]!.init?.method).toBe("POST");
  });

  test("returns raw Response (not ApiResult)", async () => {
    mockResponse = new Response("streaming data", { status: 200 });
    const resp = await client.createResponse({ model: "gpt-4" } as any);
    expect(resp).toBeInstanceOf(Response);
    expect(await resp.text()).toBe("streaming data");
  });

  test("forwards AbortSignal", async () => {
    const ac = new AbortController();
    await client.createResponse({ model: "gpt-4" } as any, ac.signal);
    expect(calls[0]!.init?.signal).toBe(ac.signal);
  });

  test("stream/store override user-provided values", async () => {
    const body = { model: "gpt-4", stream: false, store: false };
    await client.createResponse(body as any);
    const sent = JSON.parse(calls[0]!.init!.body as string);
    // Spread order: { ...body, stream: true, store: true } → overrides.
    expect(sent.stream).toBe(true);
    expect(sent.store).toBe(true);
  });
});
