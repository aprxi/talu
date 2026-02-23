import { describe, test, expect, beforeEach } from "bun:test";
import {
  loadModels,
  searchHub,
  deleteModel,
  pinModel,
  unpinModel,
  downloadModel,
  cancelDownload,
  deleteSelectedModels,
  pinSelectedModels,
} from "../../../src/plugins/repo/data.ts";
import { repoState } from "../../../src/plugins/repo/state.ts";
import { initRepoDeps } from "../../../src/plugins/repo/deps.ts";
import { initRepoDom, getRepoDom } from "../../../src/plugins/repo/dom.ts";
import { createDomRoot, REPO_DOM_IDS, REPO_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications, flushAsync } from "../../helpers/mocks.ts";

/**
 * Tests for repo plugin data operations — loading models, hub search,
 * download (SSE), delete, pin/unpin, bulk operations, and race-condition guard.
 */

// -- Mock state --------------------------------------------------------------

let apiCalls: { method: string; args: unknown[] }[];
let notif: ReturnType<typeof mockNotifications>;
let confirmResult: boolean;
let statusBusy: string | undefined;
let statusReady: boolean;

let listRepoModelsResult: any;
let searchRepoModelsResult: any;
let deleteRepoModelResult: any;
let pinRepoModelResult: any;
let unpinRepoModelResult: any;

beforeEach(() => {
  apiCalls = [];
  notif = mockNotifications();
  confirmResult = true;
  statusBusy = undefined;
  statusReady = false;

  listRepoModelsResult = { ok: true, data: { models: [], total_size_bytes: 0 } };
  searchRepoModelsResult = { ok: true, data: { results: [] } };
  deleteRepoModelResult = { ok: true };
  pinRepoModelResult = { ok: true };
  unpinRepoModelResult = { ok: true };

  // Reset state.
  repoState.tab = "discover";
  repoState.localSourceFilter = "all";
  repoState.models = [];
  repoState.totalSizeBytes = 0;
  repoState.searchResults = [];
  repoState.searchQuery = "";
  repoState.isLoading = false;
  repoState.activeDownloads.clear();
  repoState.selectedIds.clear();
  repoState.sortBy = "name";
  repoState.sortDir = "asc";
  repoState.searchGeneration = 0;
  repoState.discoverSort = "trending";
  repoState.discoverSize = "8";
  repoState.discoverTask = "text-generation";
  repoState.discoverLibrary = "safetensors";

  // DOM.
  initRepoDom(createDomRoot(REPO_DOM_IDS, undefined, REPO_DOM_TAGS));

  // Deps.
  initRepoDeps({
    api: {
      listRepoModels: async (query?: string) => {
        apiCalls.push({ method: "listRepoModels", args: [query] });
        return listRepoModelsResult;
      },
      searchRepoModels: async (query: string, opts?: any) => {
        apiCalls.push({ method: "searchRepoModels", args: [query, opts] });
        return searchRepoModelsResult;
      },
      fetchRepoModel: async (body: any, signal?: AbortSignal) => {
        apiCalls.push({ method: "fetchRepoModel", args: [body, signal] });
        // Return a simple done SSE stream.
        const text = 'data: {"event":"done"}\n\n';
        return new Response(text, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      },
      deleteRepoModel: async (id: string) => {
        apiCalls.push({ method: "deleteRepoModel", args: [id] });
        return deleteRepoModelResult;
      },
      pinRepoModel: async (id: string) => {
        apiCalls.push({ method: "pinRepoModel", args: [id] });
        return pinRepoModelResult;
      },
      unpinRepoModel: async (id: string) => {
        apiCalls.push({ method: "unpinRepoModel", args: [id] });
        return unpinRepoModelResult;
      },
    } as any,
    notifications: notif.mock as any,
    dialogs: { confirm: async () => confirmResult } as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    timers: mockTimers(),
    format: {
      date: () => "",
      dateTime: () => "Jan 1, 2025",
      relativeTime: () => "",
      duration: () => "",
      number: () => "",
    } as any,
    status: {
      setBusy: (msg?: string) => { statusBusy = msg; },
      setReady: () => { statusReady = true; },
    } as any,
  });
});

// -- Helpers -----------------------------------------------------------------

function makeModel(id: string, opts?: Partial<{ size_bytes: number; pinned: boolean; mtime: number }>): any {
  return {
    id,
    path: `/models/${id}`,
    source: "hub",
    size_bytes: opts?.size_bytes ?? 1024,
    mtime: opts?.mtime ?? 1700000000,
    architecture: "llama",
    quant_scheme: "Q4_K_M",
    pinned: opts?.pinned ?? false,
  };
}

function makeSearchResult(id: string, params = 7_000_000_000): any {
  return {
    model_id: id,
    downloads: 50000,
    likes: 1200,
    last_modified: "2025-01-15T00:00:00Z",
    params_total: params,
  };
}

// ── loadModels ──────────────────────────────────────────────────────────────

describe("loadModels", () => {
  test("calls API and populates state on success", async () => {
    listRepoModelsResult = {
      ok: true,
      data: { models: [makeModel("m1"), makeModel("m2")], total_size_bytes: 2048 },
    };

    await loadModels();

    expect(apiCalls[0]!.method).toBe("listRepoModels");
    expect(repoState.models.length).toBe(2);
    expect(repoState.totalSizeBytes).toBe(2048);
  });

  test("always loads all models (no query filter)", async () => {
    repoState.tab = "local";
    await loadModels();

    expect(apiCalls[0]!.args[0]).toBe("");
  });

  test("clears models on API failure", async () => {
    repoState.models = [makeModel("old")];
    listRepoModelsResult = { ok: false, error: "server error" };

    await loadModels();

    expect(repoState.models.length).toBe(0);
    expect(repoState.totalSizeBytes).toBe(0);
  });

  test("sets and clears isLoading flag", async () => {
    let wasLoading = false;
    initRepoDeps({
      api: {
        listRepoModels: async () => {
          wasLoading = repoState.isLoading;
          return listRepoModelsResult;
        },
      } as any,
      notifications: notif.mock as any,
      dialogs: {} as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      timers: mockTimers(),
      format: { dateTime: () => "" } as any,
      status: { setBusy: () => {}, setReady: () => {} } as any,
    });

    await loadModels();

    expect(wasLoading).toBe(true);
    expect(repoState.isLoading).toBe(false);
  });
});

// ── searchHub ───────────────────────────────────────────────────────────────

describe("searchHub", () => {
  test("empty query calls API for trending models", async () => {
    searchRepoModelsResult = {
      ok: true,
      data: { results: [makeSearchResult("trending-model")] },
    };
    await searchHub("");

    expect(apiCalls.length).toBe(1);
    expect(apiCalls[0]!.method).toBe("searchRepoModels");
    expect(repoState.searchResults.length).toBe(1);
    expect(repoState.searchResults[0]!.model_id).toBe("trending-model");
  });

  test("passes query and filter state to API", async () => {
    repoState.discoverSort = "downloads";
    repoState.discoverTask = "image-to-text";
    repoState.discoverLibrary = "transformers";

    await searchHub("llama");

    expect(apiCalls[0]!.method).toBe("searchRepoModels");
    const [query, opts] = apiCalls[0]!.args as [string, any];
    expect(query).toBe("llama");
    expect(opts.sort).toBe("downloads");
    expect(opts.filter).toBe("image-to-text");
    expect(opts.library).toBe("transformers");
    expect(opts.limit).toBe(50);
  });

  test("empty task/library filters are omitted", async () => {
    repoState.discoverTask = "";
    repoState.discoverLibrary = "";

    await searchHub("test");

    const [, opts] = apiCalls[0]!.args as [string, any];
    expect(opts.filter).toBeUndefined();
    expect(opts.library).toBeUndefined();
  });

  test("populates searchResults on success", async () => {
    searchRepoModelsResult = {
      ok: true,
      data: { results: [makeSearchResult("r1"), makeSearchResult("r2")] },
    };

    await searchHub("test");

    expect(repoState.searchResults.length).toBe(2);
    expect(repoState.searchResults[0]!.model_id).toBe("r1");
  });

  test("shows error notification on failure", async () => {
    searchRepoModelsResult = { ok: false, error: "timeout" };

    await searchHub("test");

    expect(repoState.searchResults.length).toBe(0);
    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("timeout"))).toBe(true);
  });

  test("stale search is discarded (searchGeneration guard)", async () => {
    // Set up a slow search that will be superseded.
    let resolveFirst: (v: any) => void;
    const firstPromise = new Promise((r) => { resolveFirst = r; });

    let callCount = 0;
    initRepoDeps({
      api: {
        searchRepoModels: async (query: string) => {
          callCount++;
          if (callCount === 1) {
            await firstPromise;
            return { ok: true, data: { results: [makeSearchResult("stale")] } };
          }
          return { ok: true, data: { results: [makeSearchResult("fresh")] } };
        },
      } as any,
      notifications: notif.mock as any,
      dialogs: {} as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      timers: mockTimers(),
      format: { dateTime: () => "" } as any,
      status: { setBusy: () => {}, setReady: () => {} } as any,
    });

    // Fire first search (will block).
    const first = searchHub("Llama");
    // Fire second search (will complete immediately).
    const second = searchHub("Llama 3");

    await second;

    expect(repoState.searchResults[0]!.model_id).toBe("fresh");

    // Now resolve the first — it should be discarded.
    resolveFirst!(undefined);
    await first;

    // Still shows the second result, not the stale first.
    expect(repoState.searchResults[0]!.model_id).toBe("fresh");
  });
});

// ── deleteModel ─────────────────────────────────────────────────────────────

describe("deleteModel", () => {
  test("calls confirm then API on success", async () => {
    repoState.models = [makeModel("m1"), makeModel("m2")];
    await deleteModel("m1");

    expect(apiCalls[0]!.method).toBe("deleteRepoModel");
    expect(apiCalls[0]!.args[0]).toBe("m1");
    expect(repoState.models.length).toBe(1);
    expect(notif.messages.some((m) => m.type === "success")).toBe(true);
  });

  test("does nothing when user cancels confirm", async () => {
    confirmResult = false;
    await deleteModel("m1");

    expect(apiCalls.length).toBe(0);
  });

  test("shows error notification on API failure", async () => {
    repoState.models = [makeModel("m1")];
    deleteRepoModelResult = { ok: false, error: "permission denied" };
    await deleteModel("m1");

    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("permission denied"))).toBe(true);
    // Model should NOT be removed from state on failure.
    expect(repoState.models.length).toBe(1);
  });

  test("removes model from selectedIds", async () => {
    repoState.models = [makeModel("m1")];
    repoState.selectedIds.add("m1");
    await deleteModel("m1");

    expect(repoState.selectedIds.has("m1")).toBe(false);
  });

  test("emits repo.models.changed after successful deletion", async () => {
    let emittedEvent = "";
    initRepoDeps({
      api: {
        deleteRepoModel: async () => ({ ok: true }),
      } as any,
      notifications: notif.mock as any,
      dialogs: { confirm: async () => true } as any,
      events: { emit: (name: string) => { emittedEvent = name; }, on: () => ({ dispose() {} }) } as any,
      timers: mockTimers(),
      format: { dateTime: () => "" } as any,
      status: { setBusy: () => {}, setReady: () => {} } as any,
    });

    repoState.models = [makeModel("m1")];
    await deleteModel("m1");

    expect(emittedEvent).toBe("repo.models.changed");
  });

  test("does not emit event when deletion fails", async () => {
    let emitted = false;
    initRepoDeps({
      api: {
        deleteRepoModel: async () => ({ ok: false, error: "fail" }),
      } as any,
      notifications: notif.mock as any,
      dialogs: { confirm: async () => true } as any,
      events: { emit: () => { emitted = true; }, on: () => ({ dispose() {} }) } as any,
      timers: mockTimers(),
      format: { dateTime: () => "" } as any,
      status: { setBusy: () => {}, setReady: () => {} } as any,
    });

    repoState.models = [makeModel("m1")];
    await deleteModel("m1");

    expect(emitted).toBe(false);
  });

  test("re-renders discover results when on discover tab", async () => {
    repoState.tab = "discover";
    repoState.models = [makeModel("m1")];
    repoState.searchResults = [{ model_id: "m1", downloads: 100, likes: 10, last_modified: "2025-01-01T00:00:00Z", params_total: 1000 }];

    await deleteModel("m1");

    // After deletion, model is removed from state.
    expect(repoState.models.length).toBe(0);

    // The discover card for m1 should now show Download (not Delete),
    // because the model was removed from repoState.models.
    const dom = getRepoDom();
    const btn = dom.discoverResults.querySelector<HTMLElement>("[data-action]");
    expect(btn?.dataset["action"]).toBe("download");
  });
});

// ── pinModel / unpinModel ───────────────────────────────────────────────────

describe("pinModel", () => {
  test("calls API and sets pinned=true on success", async () => {
    repoState.models = [makeModel("m1", { pinned: false })];
    await pinModel("m1");

    expect(apiCalls[0]!.method).toBe("pinRepoModel");
    expect(repoState.models[0]!.pinned).toBe(true);
  });

  test("shows error on failure", async () => {
    repoState.models = [makeModel("m1")];
    pinRepoModelResult = { ok: false, error: "failed" };
    await pinModel("m1");

    expect(notif.messages.some((m) => m.type === "error")).toBe(true);
  });
});

describe("unpinModel", () => {
  test("calls API and sets pinned=false on success", async () => {
    repoState.models = [makeModel("m1", { pinned: true })];
    await unpinModel("m1");

    expect(apiCalls[0]!.method).toBe("unpinRepoModel");
    expect(repoState.models[0]!.pinned).toBe(false);
  });

  test("shows error on failure", async () => {
    repoState.models = [makeModel("m1", { pinned: true })];
    unpinRepoModelResult = { ok: false, error: "unpin failed" };
    await unpinModel("m1");

    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("Unpin failed"))).toBe(true);
    // Model should remain pinned.
    expect(repoState.models[0]!.pinned).toBe(true);
  });
});

// ── downloadModel ───────────────────────────────────────────────────────────

describe("downloadModel", () => {
  test("sets status bar busy and tracks active download", async () => {
    await downloadModel("org/model");

    expect(statusBusy).toBe("Downloading org/model...");
    // After completion, all downloads are cleared → setReady.
    expect(statusReady).toBe(true);
    expect(repoState.activeDownloads.size).toBe(0);
  });

  test("calls fetchRepoModel with model_id and abort signal", async () => {
    await downloadModel("org/model");

    expect(apiCalls[0]!.method).toBe("fetchRepoModel");
    expect(apiCalls[0]!.args[0]).toEqual({ model_id: "org/model" });
    expect(apiCalls[0]!.args[1]).toBeInstanceOf(AbortSignal);
  });

  test("emits repo.models.changed after download", async () => {
    let emitted = false;
    initRepoDeps({
      api: {
        fetchRepoModel: async () => new Response('data: {"event":"done"}\n\n', {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        }),
        listRepoModels: async () => ({ ok: true, data: { models: [], total_size_bytes: 0 } }),
      } as any,
      notifications: notif.mock as any,
      dialogs: {} as any,
      events: { emit: (name: string) => { if (name === "repo.models.changed") emitted = true; }, on: () => ({ dispose() {} }) } as any,
      timers: mockTimers(),
      format: { dateTime: () => "" } as any,
      status: { setBusy: () => {}, setReady: () => {} } as any,
    });

    await downloadModel("org/model");
    expect(emitted).toBe(true);
  });

  test("handles download error gracefully", async () => {
    initRepoDeps({
      api: {
        fetchRepoModel: async () => new Response("", { status: 500 }),
        listRepoModels: async () => ({ ok: true, data: { models: [], total_size_bytes: 0 } }),
      } as any,
      notifications: notif.mock as any,
      dialogs: {} as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      timers: mockTimers(),
      format: { dateTime: () => "" } as any,
      status: { setBusy: () => {}, setReady: () => {} } as any,
    });

    await downloadModel("org/model");

    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("Download failed"))).toBe(true);
    expect(repoState.activeDownloads.size).toBe(0);
  });

  test("SSE progress events update activeDownloads state", async () => {
    // Stream with progress events followed by done.
    const sseBody = [
      'data: {"current":25,"total":100,"label":"25%"}\n\n',
      'data: {"current":75,"total":100,"label":"75%"}\n\n',
      'data: {"event":"done"}\n\n',
    ].join("");

    initRepoDeps({
      api: {
        fetchRepoModel: async () => new Response(sseBody, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        }),
        listRepoModels: async () => ({ ok: true, data: { models: [], total_size_bytes: 0 } }),
      } as any,
      notifications: notif.mock as any,
      dialogs: {} as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      timers: mockTimers(),
      format: { dateTime: () => "" } as any,
      status: { setBusy: () => {}, setReady: () => {} } as any,
    });

    await downloadModel("org/model");

    expect(notif.messages.some((m) => m.type === "success" && m.msg.includes("org/model"))).toBe(true);
    expect(repoState.activeDownloads.size).toBe(0);
  });

  test("SSE error event shows error notification", async () => {
    const sseBody = 'data: {"event":"error","message":"disk full"}\n\n';
    initRepoDeps({
      api: {
        fetchRepoModel: async () => new Response(sseBody, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        }),
        listRepoModels: async () => ({ ok: true, data: { models: [], total_size_bytes: 0 } }),
      } as any,
      notifications: notif.mock as any,
      dialogs: {} as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      timers: mockTimers(),
      format: { dateTime: () => "" } as any,
      status: { setBusy: () => {}, setReady: () => {} } as any,
    });

    await downloadModel("org/model");

    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("disk full"))).toBe(true);
  });

  test("SSE parser handles multi-chunk buffering", async () => {
    // Simulate a chunk split mid-line: first chunk ends without newline.
    const chunk1 = 'data: {"current":50,"total":10';
    const chunk2 = '0,"label":"50%"}\ndata: {"event":"done"}\n\n';

    const encoder = new TextEncoder();
    const chunks = [encoder.encode(chunk1), encoder.encode(chunk2)];

    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk);
        controller.close();
      },
    });

    initRepoDeps({
      api: {
        fetchRepoModel: async () => new Response(stream, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        }),
        listRepoModels: async () => ({ ok: true, data: { models: [], total_size_bytes: 0 } }),
      } as any,
      notifications: notif.mock as any,
      dialogs: {} as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      timers: mockTimers(),
      format: { dateTime: () => "" } as any,
      status: { setBusy: () => {}, setReady: () => {} } as any,
    });

    await downloadModel("org/model");

    // The split data line should be reassembled and parsed as a progress event,
    // and the done event should fire the success notification.
    expect(notif.messages.some((m) => m.type === "success" && m.msg.includes("org/model"))).toBe(true);
  });

  test("SSE parser ignores malformed JSON lines", async () => {
    const sseBody = 'data: {invalid json}\ndata: {"event":"done"}\n\n';
    initRepoDeps({
      api: {
        fetchRepoModel: async () => new Response(sseBody, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        }),
        listRepoModels: async () => ({ ok: true, data: { models: [], total_size_bytes: 0 } }),
      } as any,
      notifications: notif.mock as any,
      dialogs: {} as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      timers: mockTimers(),
      format: { dateTime: () => "" } as any,
      status: { setBusy: () => {}, setReady: () => {} } as any,
    });

    // Should not throw — malformed line is skipped, done event still processed.
    await downloadModel("org/model");

    expect(notif.messages.some((m) => m.type === "success" && m.msg.includes("org/model"))).toBe(true);
  });

  test("cancelDownload aborts the download without error notification", async () => {
    let emitted = false;
    initRepoDeps({
      api: {
        fetchRepoModel: async (_body: any, signal?: AbortSignal) => {
          // Return a stream that blocks until aborted.
          const stream = new ReadableStream<Uint8Array>({
            pull() {
              return new Promise<void>((_resolve, reject) => {
                signal?.addEventListener("abort", () => {
                  reject(new DOMException("Aborted", "AbortError"));
                });
              });
            },
          });
          return new Response(stream, {
            status: 200,
            headers: { "Content-Type": "text/event-stream" },
          });
        },
        listRepoModels: async () => ({ ok: true, data: { models: [], total_size_bytes: 0 } }),
      } as any,
      notifications: notif.mock as any,
      dialogs: {} as any,
      events: { emit: () => { emitted = true; }, on: () => ({ dispose() {} }) } as any,
      timers: mockTimers(),
      format: { dateTime: () => "" } as any,
      status: { setBusy: () => {}, setReady: () => {} } as any,
    });

    const dlPromise = downloadModel("org/model");
    await flushAsync(); // Let download reach reader.read()

    cancelDownload("org/model");
    await dlPromise;

    // No error notification for cancel.
    expect(notif.messages.filter((m) => m.type === "error").length).toBe(0);
    expect(repoState.activeDownloads.size).toBe(0);
    // Does not emit repo.models.changed or call loadModels.
    expect(emitted).toBe(false);
  });

  test("cancelDownload on non-existent download is a no-op", () => {
    // Should not throw.
    cancelDownload("non-existent");
    expect(repoState.activeDownloads.size).toBe(0);
  });
});

// ── Bulk operations ─────────────────────────────────────────────────────────

describe("deleteSelectedModels", () => {
  test("deletes all selected models concurrently", async () => {
    repoState.selectedIds.add("m1");
    repoState.selectedIds.add("m2");
    repoState.selectedIds.add("m3");

    await deleteSelectedModels();

    expect(apiCalls.filter((c) => c.method === "deleteRepoModel").length).toBe(3);
    expect(repoState.selectedIds.size).toBe(0);
    expect(notif.messages.some((m) => m.type === "success" && m.msg.includes("3"))).toBe(true);
  });

  test("does nothing when no models selected", async () => {
    await deleteSelectedModels();
    expect(apiCalls.length).toBe(0);
  });

  test("does nothing when user cancels confirm", async () => {
    repoState.selectedIds.add("m1");
    confirmResult = false;
    await deleteSelectedModels();
    expect(apiCalls.filter((c) => c.method === "deleteRepoModel").length).toBe(0);
  });

  test("counts only successful deletions", async () => {
    repoState.selectedIds.add("m1");
    repoState.selectedIds.add("m2");

    let callIdx = 0;
    initRepoDeps({
      api: {
        deleteRepoModel: async () => {
          apiCalls.push({ method: "deleteRepoModel", args: [] });
          return ++callIdx === 1 ? { ok: true } : { ok: false, error: "fail" };
        },
        listRepoModels: async () => ({ ok: true, data: { models: [], total_size_bytes: 0 } }),
      } as any,
      notifications: notif.mock as any,
      dialogs: { confirm: async () => true } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      timers: mockTimers(),
      format: { dateTime: () => "" } as any,
      status: { setBusy: () => {}, setReady: () => {} } as any,
    });

    await deleteSelectedModels();

    expect(notif.messages.some((m) => m.type === "success" && m.msg.includes("1 model"))).toBe(true);
  });
});

describe("pinSelectedModels", () => {
  test("pins all selected models concurrently", async () => {
    repoState.models = [
      makeModel("m1", { pinned: false }),
      makeModel("m2", { pinned: false }),
    ];
    repoState.selectedIds.add("m1");
    repoState.selectedIds.add("m2");

    await pinSelectedModels();

    expect(apiCalls.filter((c) => c.method === "pinRepoModel").length).toBe(2);
    expect(repoState.models[0]!.pinned).toBe(true);
    expect(repoState.models[1]!.pinned).toBe(true);
    expect(repoState.selectedIds.size).toBe(0);
  });

  test("does nothing when no models selected", async () => {
    await pinSelectedModels();
    expect(apiCalls.length).toBe(0);
  });

  test("partial failure leaves failed models unpinned", async () => {
    repoState.models = [
      makeModel("m1", { pinned: false }),
      makeModel("m2", { pinned: false }),
    ];
    repoState.selectedIds.add("m1");
    repoState.selectedIds.add("m2");

    let callIdx = 0;
    initRepoDeps({
      api: {
        pinRepoModel: async () => {
          apiCalls.push({ method: "pinRepoModel", args: [] });
          return ++callIdx === 1 ? { ok: true } : { ok: false, error: "fail" };
        },
      } as any,
      notifications: notif.mock as any,
      dialogs: {} as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      timers: mockTimers(),
      format: { dateTime: () => "" } as any,
      status: { setBusy: () => {}, setReady: () => {} } as any,
    });

    await pinSelectedModels();

    // First model pinned, second failed — should remain unpinned.
    expect(repoState.models[0]!.pinned).toBe(true);
    expect(repoState.models[1]!.pinned).toBe(false);
    expect(repoState.selectedIds.size).toBe(0);
  });
});
