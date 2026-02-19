import { describe, test, expect, beforeEach } from "bun:test";
import {
  handleBrowserDelete,
  handleBrowserExport,
  handleBrowserArchive,
  handleBrowserBulkRestore,
  handleCardRestore,
} from "../../../src/plugins/browser/actions.ts";
import { bState } from "../../../src/plugins/browser/state.ts";
import { initBrowserDom } from "../../../src/plugins/browser/dom.ts";
import { initBrowserDeps } from "../../../src/plugins/browser/deps.ts";
import { createDomRoot, BROWSER_DOM_IDS, BROWSER_DOM_EXTRAS, BROWSER_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import type { Conversation } from "../../../src/types.ts";

/**
 * Tests for browser bulk actions — delete, export, archive, restore.
 */

let notifs: ReturnType<typeof mockNotifications>;
let apiCalls: { method: string; args: unknown[] }[];
let confirmResult: boolean;
let savedBlobs: { blob: Blob; filename: string }[];
let sidebarRefreshed: boolean;

function makeConvo(id: string, marker = ""): Conversation {
  return {
    id,
    object: "conversation",
    title: `Convo ${id}`,
    created_at: 1700000000,
    updated_at: 1700000000,
    model: "gpt-4",
    items: [],
    metadata: {},
    marker,
  } as Conversation;
}

beforeEach(() => {
  notifs = mockNotifications();
  apiCalls = [];
  confirmResult = true;
  savedBlobs = [];
  sidebarRefreshed = false;

  bState.selectedIds.clear();
  bState.conversations = [];
  bState.tab = "all";
  bState.isLoading = false;
  bState.pagination = { currentPage: 1, pageSize: 50, totalItems: 0 };

  initBrowserDom(createDomRoot(BROWSER_DOM_IDS, BROWSER_DOM_EXTRAS, BROWSER_DOM_TAGS));

  initBrowserDeps({
    api: {
      batchConversations: async (req: any) => {
        apiCalls.push({ method: "batchConversations", args: [req] });
        return { ok: true };
      },
      patchConversation: async (id: string, patch: any) => {
        apiCalls.push({ method: "patchConversation", args: [id, patch] });
        return { ok: true, data: makeConvo(id) };
      },
      listConversations: async () => ({
        ok: true,
        data: { data: [], total: 0, has_more: false },
      }),
      search: async () => ({ ok: true, data: { aggregations: { tags: [] } } }),
    } as any,
    notify: notifs.mock,
    dialogs: {
      confirm: async () => confirmResult,
    } as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    chatService: {
      refreshSidebar: async () => { sidebarRefreshed = true; },
      selectChat: async () => {},
      getSessions: () => [],
    } as any,
    download: {
      save: (blob: Blob, filename: string) => { savedBlobs.push({ blob, filename }); },
    } as any,
    timers: mockTimers(),
    menus: {
      registerItem: () => ({ dispose() {} }),
      renderSlot: () => ({ dispose() {} }),
    } as any,
  });
});

// ── Delete ───────────────────────────────────────────────────────────────────

describe("handleBrowserDelete", () => {
  test("no-op when no selection", async () => {
    await handleBrowserDelete();
    expect(apiCalls.length).toBe(0);
  });

  test("confirms before deleting", async () => {
    bState.selectedIds.add("c1");
    confirmResult = false;
    await handleBrowserDelete();
    expect(apiCalls.length).toBe(0);
  });

  test("calls batch delete API and clears selection", async () => {
    bState.selectedIds.add("c1");
    bState.selectedIds.add("c2");
    await handleBrowserDelete();
    expect(apiCalls[0]!.method).toBe("batchConversations");
    expect((apiCalls[0]!.args[0] as any).action).toBe("delete");
    expect((apiCalls[0]!.args[0] as any).ids).toEqual(expect.arrayContaining(["c1", "c2"]));
    expect(bState.selectedIds.size).toBe(0);
    expect(notifs.messages.some((m) => m.msg.includes("Deleted 2"))).toBe(true);
    expect(sidebarRefreshed).toBe(true);
  });

  test("shows error on API failure", async () => {
    bState.selectedIds.add("c1");
    // Override to fail
    initBrowserDeps({
      api: {
        batchConversations: async () => ({ ok: false, error: "Server error" }),
        listConversations: async () => ({
          ok: true,
          data: { data: [], total: 0, has_more: false },
        }),
      } as any,
      notify: notifs.mock,
      dialogs: { confirm: async () => true } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      chatService: { refreshSidebar: async () => {}, selectChat: async () => {}, getSessions: () => [] } as any,
      download: {} as any,
      timers: mockTimers(),
      menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
    });
    await handleBrowserDelete();
    expect(notifs.messages.some((m) => m.type === "error")).toBe(true);
  });
});

// ── Export ────────────────────────────────────────────────────────────────────

describe("handleBrowserExport", () => {
  test("no-op when no selection", () => {
    handleBrowserExport();
    expect(savedBlobs.length).toBe(0);
  });

  test("exports selected conversations as JSON", () => {
    bState.conversations = [makeConvo("c1"), makeConvo("c2"), makeConvo("c3")];
    bState.selectedIds.add("c1");
    bState.selectedIds.add("c3");
    handleBrowserExport();
    expect(savedBlobs.length).toBe(1);
    expect(savedBlobs[0]!.filename).toContain("talu-export-2-conversations.json");
    expect(savedBlobs[0]!.blob.type).toBe("application/json");
    expect(notifs.messages.some((m) => m.msg.includes("Exported 2"))).toBe(true);
  });
});

// ── Archive ──────────────────────────────────────────────────────────────────

describe("handleBrowserArchive", () => {
  test("no-op when no selection", async () => {
    await handleBrowserArchive();
    expect(apiCalls.length).toBe(0);
  });

  test("calls batch archive and clears selection", async () => {
    bState.selectedIds.add("c1");
    await handleBrowserArchive();
    expect(apiCalls[0]!.method).toBe("batchConversations");
    expect((apiCalls[0]!.args[0] as any).action).toBe("archive");
    expect(bState.selectedIds.size).toBe(0);
    expect(notifs.messages.some((m) => m.msg.includes("Archived 1"))).toBe(true);
  });
});

// ── Bulk Restore ─────────────────────────────────────────────────────────────

describe("handleBrowserBulkRestore", () => {
  test("no-op when no selection", async () => {
    await handleBrowserBulkRestore();
    expect(apiCalls.length).toBe(0);
  });

  test("calls batch unarchive and clears selection", async () => {
    bState.selectedIds.add("c1");
    bState.selectedIds.add("c2");
    await handleBrowserBulkRestore();
    expect(apiCalls[0]!.method).toBe("batchConversations");
    expect((apiCalls[0]!.args[0] as any).action).toBe("unarchive");
    expect(bState.selectedIds.size).toBe(0);
    expect(notifs.messages.some((m) => m.msg.includes("Restored 2"))).toBe(true);
  });
});

// ── Card Restore (single) ────────────────────────────────────────────────────

describe("handleCardRestore", () => {
  test("patches conversation marker and refreshes", async () => {
    await handleCardRestore("c1");
    expect(apiCalls[0]!.method).toBe("patchConversation");
    expect(apiCalls[0]!.args).toEqual(["c1", { marker: "" }]);
    expect(notifs.messages.some((m) => m.msg === "Restored conversation")).toBe(true);
    expect(sidebarRefreshed).toBe(true);
  });

  test("shows error on API failure", async () => {
    initBrowserDeps({
      api: {
        patchConversation: async () => ({ ok: false, error: "Not found" }),
        listConversations: async () => ({
          ok: true,
          data: { data: [], total: 0, has_more: false },
        }),
      } as any,
      notify: notifs.mock,
      dialogs: {} as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      chatService: { refreshSidebar: async () => {}, selectChat: async () => {}, getSessions: () => [] } as any,
      download: {} as any,
      timers: mockTimers(),
      menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
    });
    await handleCardRestore("missing");
    expect(notifs.messages.some((m) => m.type === "error")).toBe(true);
  });
});
