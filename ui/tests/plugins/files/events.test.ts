import { describe, test, expect, beforeEach } from "bun:test";
import { wireFileEvents } from "../../../src/plugins/files/events.ts";
import { fState } from "../../../src/plugins/files/state.ts";
import { initFilesDom, getFilesDom } from "../../../src/plugins/files/dom.ts";
import { initFilesDeps } from "../../../src/plugins/files/deps.ts";
import { createDomRoot, FILES_DOM_IDS, FILES_DOM_EXTRAS, FILES_DOM_TAGS } from "../../helpers/dom.ts";
import { mockControllableTimers, flushAsync } from "../../helpers/mocks.ts";

/**
 * Tests for files event wiring — search debouncing, tab switching,
 * select-all toggle, cancel, drag-and-drop, and table row interactions.
 *
 * Strategy: wire events with a controllable timer mock, then dispatch DOM
 * events and verify state mutations. API is mocked to return empty results.
 */

// -- Mock state --------------------------------------------------------------

let ct: ReturnType<typeof mockControllableTimers>;
let apiCalls: { method: string; args: unknown[] }[];
let uploadCalls: { file: File; purpose: string }[];

beforeEach(() => {
  ct = mockControllableTimers();
  apiCalls = [];
  uploadCalls = [];

  // Reset state.
  fState.files = [];
  fState.isLoading = false;
  fState.searchQuery = "";
  fState.selectedFileId = null;
  fState.editingFileId = null;
  fState.selectedIds.clear();
  fState.tab = "all";
  fState.sortBy = "name";
  fState.sortDir = "asc";
  fState.pagination = { currentPage: 1, pageSize: 50, totalItems: 0 };

  // DOM.
  initFilesDom(createDomRoot(FILES_DOM_IDS, FILES_DOM_EXTRAS, FILES_DOM_TAGS));

  // Deps with controllable timer.
  initFilesDeps({
    api: {
      listFiles: async (opts?: any) => {
        apiCalls.push({ method: "listFiles", args: [opts] });
        return { ok: true, data: { data: [], total: 0 } };
      },
      updateFile: async (id: string, patch: any) => {
        apiCalls.push({ method: "updateFile", args: [id, patch] });
        return { ok: true, data: { id, filename: "updated" } };
      },
      deleteFile: async (id: string) => {
        apiCalls.push({ method: "deleteFile", args: [id] });
        return { ok: true };
      },
    } as any,
    notify: { info: () => {}, error: () => {}, warn: () => {}, success: () => {} } as any,
    dialogs: { confirm: async () => true } as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    upload: {
      upload: async (file: File, purpose: string) => {
        uploadCalls.push({ file, purpose });
        return { id: `uploaded-${file.name}` };
      },
    } as any,
    download: {} as any,
    timers: ct.timers,
    format: {
      date: () => "", dateTime: () => "Jan 1, 2025", relativeTime: () => "",
      duration: () => "", number: () => "",
    } as any,
  });
});

// -- Helpers -----------------------------------------------------------------

function makeFile(id: string, name = "test.txt", bytes = 1024): any {
  return {
    id, object: "file", bytes, created_at: 1700000000,
    filename: name, purpose: "assistants", mime_type: "text/plain",
    kind: "text", marker: "active",
  };
}

/** Create a mock FileList from an array of Files. */
function makeFileList(...files: File[]): globalThis.FileList {
  const list: any = {
    length: files.length,
    item: (i: number) => files[i] ?? null,
    [Symbol.iterator]: function* () { yield* files; },
  };
  for (let i = 0; i < files.length; i++) list[i] = files[i];
  return list as globalThis.FileList;
}

// Tab switching is now handled by the subnav event bus (subnav.tab) in index.ts,
// not by wireFileEvents(). Those tests have been removed.

// ── Search debouncing ────────────────────────────────────────────────────────

describe("Search debouncing", () => {
  test("input schedules 200ms debounce", () => {
    wireFileEvents();
    const dom = getFilesDom();
    (dom.searchInput as HTMLInputElement).value = "test";
    dom.searchInput.dispatchEvent(new Event("input"));

    expect(ct.pending.length).toBe(1);
    expect(ct.pending[0]!.ms).toBe(200);
  });

  test("rapid typing cancels previous timer", () => {
    wireFileEvents();
    const dom = getFilesDom();

    (dom.searchInput as HTMLInputElement).value = "t";
    dom.searchInput.dispatchEvent(new Event("input"));
    (dom.searchInput as HTMLInputElement).value = "te";
    dom.searchInput.dispatchEvent(new Event("input"));
    (dom.searchInput as HTMLInputElement).value = "tes";
    dom.searchInput.dispatchEvent(new Event("input"));

    expect(ct.pending[0]!.disposed).toBe(true);
    expect(ct.pending[1]!.disposed).toBe(true);
    expect(ct.pending[2]!.disposed).toBe(false);
  });

  test("debounce callback updates search query", () => {
    wireFileEvents();
    const dom = getFilesDom();
    (dom.searchInput as HTMLInputElement).value = "new query";
    dom.searchInput.dispatchEvent(new Event("input"));

    // Fire the debounced callback.
    ct.pending[0]!.fn();
    expect(fState.searchQuery).toBe("new query");
  });

  test("debounce no-op when query unchanged", () => {
    fState.searchQuery = "same";
    wireFileEvents();
    const dom = getFilesDom();
    (dom.searchInput as HTMLInputElement).value = "same";
    dom.searchInput.dispatchEvent(new Event("input"));

    ct.pending[0]!.fn();
    // searchQuery should still be "same" — no render cycle triggered.
    expect(fState.searchQuery).toBe("same");
  });

  test("clear button shows when input has text", () => {
    wireFileEvents();
    const dom = getFilesDom();
    (dom.searchInput as HTMLInputElement).value = "text";
    dom.searchInput.dispatchEvent(new Event("input"));

    expect(dom.searchClear.classList.contains("hidden")).toBe(false);
  });

  test("clear button hides when input is empty", () => {
    wireFileEvents();
    const dom = getFilesDom();
    dom.searchClear.classList.remove("hidden"); // start visible
    (dom.searchInput as HTMLInputElement).value = "";
    dom.searchInput.dispatchEvent(new Event("input"));

    expect(dom.searchClear.classList.contains("hidden")).toBe(true);
  });

  test("clear button resets input and state", () => {
    fState.searchQuery = "query";
    wireFileEvents();
    const dom = getFilesDom();
    (dom.searchInput as HTMLInputElement).value = "query";

    dom.searchClear.dispatchEvent(new Event("click"));

    expect((dom.searchInput as HTMLInputElement).value).toBe("");
    expect(fState.searchQuery).toBe("");
  });

  test("clear button clears preview selection", () => {
    fState.selectedFileId = "f1";
    wireFileEvents();
    getFilesDom().searchClear.dispatchEvent(new Event("click"));

    expect(fState.selectedFileId).toBeNull();
  });
});

// ── Select All / Cancel ──────────────────────────────────────────────────────

describe("Select All / Cancel", () => {
  test("select all adds all filtered files", () => {
    fState.files = [makeFile("f1"), makeFile("f2"), makeFile("f3")];
    wireFileEvents();
    getFilesDom().selectAllBtn.dispatchEvent(new Event("click"));
    expect(fState.selectedIds.size).toBe(3);
  });

  test("select all deselects when all already selected", () => {
    fState.files = [makeFile("f1"), makeFile("f2")];
    fState.selectedIds.add("f1");
    fState.selectedIds.add("f2");
    wireFileEvents();
    getFilesDom().selectAllBtn.dispatchEvent(new Event("click"));
    expect(fState.selectedIds.size).toBe(0);
  });

  test("cancel clears all selections", () => {
    fState.selectedIds.add("f1");
    fState.selectedIds.add("f2");
    wireFileEvents();
    getFilesDom().cancelBtn.dispatchEvent(new Event("click"));
    expect(fState.selectedIds.size).toBe(0);
  });

  test("select all respects server-filtered files", () => {
    // Search is server-side: fState.files already contains only matching results.
    fState.files = [makeFile("f1", "alpha.txt")];
    fState.searchQuery = "alpha";
    wireFileEvents();
    getFilesDom().selectAllBtn.dispatchEvent(new Event("click"));
    expect(fState.selectedIds.size).toBe(1);
    expect(fState.selectedIds.has("f1")).toBe(true);
  });
});

// ── Table row clicks ─────────────────────────────────────────────────────────

describe("Table row clicks", () => {
  test("toggle button adds file to selection", () => {
    wireFileEvents();
    const dom = getFilesDom();
    const btn = document.createElement("button");
    btn.dataset["action"] = "toggle";
    btn.dataset["id"] = "f1";
    dom.tbody.appendChild(btn);

    btn.dispatchEvent(new Event("click", { bubbles: true }));
    expect(fState.selectedIds.has("f1")).toBe(true);
  });

  test("toggle button removes file from selection", () => {
    fState.selectedIds.add("f1");
    wireFileEvents();
    const dom = getFilesDom();
    const btn = document.createElement("button");
    btn.dataset["action"] = "toggle";
    btn.dataset["id"] = "f1";
    dom.tbody.appendChild(btn);

    btn.dispatchEvent(new Event("click", { bubbles: true }));
    expect(fState.selectedIds.has("f1")).toBe(false);
  });

  test("row click sets preview selection", () => {
    wireFileEvents();
    const dom = getFilesDom();
    const row = document.createElement("tr");
    row.className = "files-row";
    row.dataset["id"] = "f1";
    const cell = document.createElement("td");
    row.appendChild(cell);
    dom.tbody.appendChild(row);

    cell.dispatchEvent(new Event("click", { bubbles: true }));
    expect(fState.selectedFileId).toBe("f1");
  });
});

// ── Drag-and-drop ────────────────────────────────────────────────────────────

describe("Drag-and-drop", () => {
  test("dragenter shows overlay", () => {
    wireFileEvents();
    const dom = getFilesDom();
    dom.dropOverlay.classList.add("hidden");

    dom.mainDrop.dispatchEvent(new DragEvent("dragenter", { bubbles: true }));
    expect(dom.dropOverlay.classList.contains("hidden")).toBe(false);
  });

  test("dragleave hides overlay when counter reaches zero", () => {
    wireFileEvents();
    const dom = getFilesDom();

    // Enter twice (nested elements).
    dom.mainDrop.dispatchEvent(new DragEvent("dragenter", { bubbles: true }));
    dom.mainDrop.dispatchEvent(new DragEvent("dragenter", { bubbles: true }));
    // Leave once — still nested.
    dom.mainDrop.dispatchEvent(new DragEvent("dragleave", { bubbles: true }));
    expect(dom.dropOverlay.classList.contains("hidden")).toBe(false);
    // Leave again — counter hits 0.
    dom.mainDrop.dispatchEvent(new DragEvent("dragleave", { bubbles: true }));
    expect(dom.dropOverlay.classList.contains("hidden")).toBe(true);
  });

  test("drop resets overlay and counter", () => {
    wireFileEvents();
    const dom = getFilesDom();

    dom.mainDrop.dispatchEvent(new DragEvent("dragenter", { bubbles: true }));
    dom.mainDrop.dispatchEvent(new DragEvent("dragenter", { bubbles: true }));

    const dropEvent = new DragEvent("drop", { bubbles: true });
    dom.mainDrop.dispatchEvent(dropEvent);

    expect(dom.dropOverlay.classList.contains("hidden")).toBe(true);
  });

  test("drop with files triggers upload", async () => {
    wireFileEvents();
    const dom = getFilesDom();

    const file = new File(["hello"], "dropped.txt", { type: "text/plain" });
    const dropEvent = new DragEvent("drop", { bubbles: true });
    // HappyDOM doesn't pass dataTransfer through the constructor —
    // inject it manually to exercise the upload path.
    Object.defineProperty(dropEvent, "dataTransfer", {
      value: { files: makeFileList(file) },
    });

    dom.mainDrop.dispatchEvent(dropEvent);
    await flushAsync();

    expect(uploadCalls.length).toBe(1);
    expect(uploadCalls[0]!.file.name).toBe("dropped.txt");
    expect(uploadCalls[0]!.purpose).toBe("assistants");
  });
});

// ── Rename interactions ──────────────────────────────────────────────────────

describe("Rename interactions", () => {
  test("double-click on name cell enters edit mode", () => {
    fState.files = [makeFile("f1", "test.txt")];
    wireFileEvents();
    const dom = getFilesDom();

    const row = document.createElement("tr");
    row.className = "files-row";
    row.dataset["id"] = "f1";
    const nameCell = document.createElement("td");
    nameCell.className = "files-cell-name";
    row.appendChild(nameCell);
    dom.tbody.appendChild(row);

    nameCell.dispatchEvent(new Event("dblclick", { bubbles: true }));
    expect(fState.editingFileId).toBe("f1");
  });

  test("Escape key cancels editing", () => {
    fState.editingFileId = "f1";
    wireFileEvents();
    const dom = getFilesDom();

    const input = document.createElement("input");
    input.className = "files-name-input";
    input.dataset["id"] = "f1";
    input.value = "new-name.txt";
    dom.tbody.appendChild(input);

    const event = new KeyboardEvent("keydown", { key: "Escape", bubbles: true });
    input.dispatchEvent(event);

    expect(fState.editingFileId).toBeNull();
  });

  test("Enter key triggers rename", async () => {
    wireFileEvents();
    const dom = getFilesDom();

    const input = document.createElement("input");
    input.className = "files-name-input";
    input.dataset["id"] = "f1";
    input.value = "renamed.txt";
    dom.tbody.appendChild(input);

    const event = new KeyboardEvent("keydown", { key: "Enter", bubbles: true });
    input.dispatchEvent(event);
    await flushAsync();

    expect(apiCalls.some((c) => c.method === "updateFile")).toBe(true);
  });
});

// ── Upload button ────────────────────────────────────────────────────────────

describe("Upload button", () => {
  test("upload button triggers file input click", () => {
    wireFileEvents();
    const dom = getFilesDom();
    let inputClicked = false;
    dom.fileInput.addEventListener("click", () => { inputClicked = true; });

    dom.uploadBtn.dispatchEvent(new Event("click"));
    expect(inputClicked).toBe(true);
  });
});
