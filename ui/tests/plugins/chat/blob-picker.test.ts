import { describe, test, expect, beforeEach } from "bun:test";
import { openBlobPicker } from "../../../src/plugins/chat/blob-picker.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import type { FileObject } from "../../../src/types.ts";

/**
 * Tests for blob-picker — modal overlay for selecting files from library.
 * Covers: open/dismiss lifecycle, search filtering, sort toggling,
 * row click preview, keyboard navigation, and confirm/cancel flows.
 */

let notifs: ReturnType<typeof mockNotifications>;
let fileList: FileObject[];

function makeFile(id: string, overrides: Partial<FileObject> = {}): FileObject {
  return {
    id,
    filename: `${id}.txt`,
    bytes: 1024,
    kind: "text",
    mime_type: "text/plain",
    created_at: 1700000000,
    ...overrides,
  } as FileObject;
}

function makeImageFile(id: string): FileObject {
  return makeFile(id, {
    filename: `${id}.png`,
    kind: "image",
    mime_type: "image/png",
    bytes: 50000,
    image: { width: 800, height: 600, format: "png" },
  } as any);
}

beforeEach(() => {
  notifs = mockNotifications();
  fileList = [makeFile("f1"), makeFile("f2"), makeImageFile("img1")];

  initChatDeps({
    api: {
      listFiles: async () => ({ ok: true, data: { data: fileList } }),
    } as any,
    notifications: notifs.mock,
    timers: mockTimers(),
    services: {} as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    hooks: { run: async (_name: string, payload: any) => payload } as any,
    clipboard: { writeText: async () => {} } as any,
    download: {} as any,
    observe: { onResize: () => ({ dispose() {} }) } as any,
    format: { dateTime: () => "2024-01-01" } as any,
    upload: { upload: async () => ({}) } as any,
    menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
  });
});

/** Flush the async API call that happens on open. */
async function flushOpen(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
}

// ── Open and dismiss ────────────────────────────────────────────────────────

describe("openBlobPicker — lifecycle", () => {
  test("creates overlay in document.body", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    const overlay = document.querySelector(".blob-picker-overlay");
    expect(overlay).not.toBeNull();
    // Clean up by pressing Escape
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await promise;
  });

  test("Escape dismisses and resolves empty", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    const result = await promise;
    expect(result).toEqual([]);
  });

  test("Cancel button dismisses and resolves empty", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    const cancelBtn = document.querySelector<HTMLButtonElement>(".blob-picker .btn-ghost")!;
    cancelBtn.click();
    const result = await promise;
    expect(result).toEqual([]);
  });

  test("clicking overlay backdrop dismisses", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    const overlay = document.querySelector<HTMLElement>(".blob-picker-overlay")!;
    overlay.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    const result = await promise;
    expect(result).toEqual([]);
  });

  test("overlay removed from DOM after dismiss", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await promise;
    expect(document.querySelector(".blob-picker-overlay")).toBeNull();
  });
});

// ── File table rendering ────────────────────────────────────────────────────

describe("openBlobPicker — table", () => {
  test("renders table rows for fetched files", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    const rows = document.querySelectorAll(".files-row");
    expect(rows.length).toBe(3);
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await promise;
  });

  test("shows file count in footer", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    const count = document.querySelector(".blob-picker-count")!;
    expect(count.textContent).toContain("3 files");
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await promise;
  });

  test("shows empty state when no files returned", async () => {
    fileList = [];
    initChatDeps({
      api: { listFiles: async () => ({ ok: true, data: { data: [] } }) } as any,
      notifications: notifs.mock,
      timers: mockTimers(),
      services: {} as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      hooks: { run: async (_name: string, payload: any) => payload } as any,
      clipboard: { writeText: async () => {} } as any,
      download: {} as any,
      observe: { onResize: () => ({ dispose() {} }) } as any,
      format: { dateTime: () => "2024-01-01" } as any,
      upload: { upload: async () => ({}) } as any,
      menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
    });
    const promise = openBlobPicker();
    await flushOpen();
    const empty = document.querySelector(".blob-picker-empty");
    expect(empty).not.toBeNull();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await promise;
  });
});

// ── Row click preview ───────────────────────────────────────────────────────

describe("openBlobPicker — preview", () => {
  test("clicking a row shows preview pane", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    const row = document.querySelector<HTMLElement>(".files-row")!;
    row.click();
    const preview = document.querySelector<HTMLElement>(".blob-picker-preview")!;
    expect(preview.classList.contains("hidden")).toBe(false);
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await promise;
  });

  test("preview shows metadata fields", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    const row = document.querySelector<HTMLElement>(".files-row")!;
    row.click();
    const meta = document.querySelector(".files-preview-meta")!;
    expect(meta.textContent).toContain("Filename");
    expect(meta.textContent).toContain("Size");
    expect(meta.textContent).toContain("Kind");
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await promise;
  });

  test("image file preview shows img element", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    // Find image row (3rd row)
    const rows = document.querySelectorAll<HTMLElement>(".files-row");
    const imgRow = Array.from(rows).find((r) => r.dataset["id"] === "img1")!;
    imgRow.click();
    const img = document.querySelector<HTMLImageElement>(".files-preview-img");
    expect(img).not.toBeNull();
    expect(img!.src).toContain("img1");
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await promise;
  });

  test("clicked row gets selected class", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    const row = document.querySelector<HTMLElement>(".files-row")!;
    row.click();
    expect(row.classList.contains("files-row-selected")).toBe(true);
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await promise;
  });
});

// ── Confirm ─────────────────────────────────────────────────────────────────

describe("openBlobPicker — confirm", () => {
  test("Enter key confirms previewed file", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    const row = document.querySelector<HTMLElement>(".files-row")!;
    row.click();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    const result = await promise;
    expect(result.length).toBe(1);
    expect(result[0]!.id).toBe("f1");
  });

  test("Enter without preview does not confirm", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    // Don't click any row — no preview
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    // Should still be open, dismiss
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    const result = await promise;
    expect(result).toEqual([]);
  });

  test("Add button confirms previewed file", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    const row = document.querySelector<HTMLElement>(".files-row")!;
    row.click();
    const addBtn = document.querySelector<HTMLButtonElement>(".blob-picker-add-btn")!;
    addBtn.click();
    const result = await promise;
    expect(result.length).toBe(1);
  });
});

// ── Search filtering ────────────────────────────────────────────────────────

describe("openBlobPicker — search", () => {
  test("filtering narrows visible rows", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    const search = document.querySelector<HTMLInputElement>(".blob-picker-search")!;
    search.value = "img1";
    search.dispatchEvent(new Event("input"));
    // Wait for debounce (150ms)
    await new Promise((r) => setTimeout(r, 200));
    const rows = document.querySelectorAll(".files-row");
    expect(rows.length).toBe(1);
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await promise;
  });
});

// ── Sort toggling ───────────────────────────────────────────────────────────

describe("openBlobPicker — sort", () => {
  test("clicking column header toggles sort", async () => {
    const promise = openBlobPicker();
    await flushOpen();
    const sizeHeader = document.querySelector<HTMLElement>('[data-sort="size"]')!;
    sizeHeader.click();
    // renderTable() rebuilds the DOM, so re-query after click
    const updatedHeader = document.querySelector<HTMLElement>('[data-sort="size"]')!;
    expect(updatedHeader.classList.contains("files-th-sorted")).toBe(true);
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await promise;
  });
});

// ── Error handling ──────────────────────────────────────────────────────────

describe("openBlobPicker — errors", () => {
  test("API failure shows error and dismisses", async () => {
    initChatDeps({
      api: { listFiles: async () => ({ ok: false, error: "Network error" }) } as any,
      notifications: notifs.mock,
      timers: mockTimers(),
      services: {} as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      hooks: { run: async (_name: string, payload: any) => payload } as any,
      clipboard: { writeText: async () => {} } as any,
      download: {} as any,
      observe: { onResize: () => ({ dispose() {} }) } as any,
      format: { dateTime: () => "2024-01-01" } as any,
      upload: { upload: async () => ({}) } as any,
      menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
    });
    const result = await openBlobPicker();
    expect(result).toEqual([]);
    expect(notifs.messages.some((m) => m.type === "error")).toBe(true);
  });
});
