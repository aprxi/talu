import { describe, test, expect, beforeEach } from "bun:test";
import {
  renderFilesTable,
  renderStats,
  renderPreview,
  syncFilesTabs,
  updateFilesToolbar,
} from "../../../src/plugins/files/render.ts";
import { fState } from "../../../src/plugins/files/state.ts";
import { initFilesDeps } from "../../../src/plugins/files/deps.ts";
import { initFilesDom, getFilesDom } from "../../../src/plugins/files/dom.ts";
import { createDomRoot, FILES_DOM_IDS, FILES_DOM_EXTRAS, FILES_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers } from "../../helpers/mocks.ts";

/**
 * Tests for files plugin rendering — table structure, stats formatting,
 * preview panel, tab sync, and toolbar state.
 *
 * Strategy: set fState, call render function, inspect resulting DOM nodes.
 */

beforeEach(() => {
  // Reset state.
  fState.files = [];
  fState.isLoading = false;
  fState.searchQuery = "";
  fState.selectedFileId = null;
  fState.editingFileId = null;
  fState.selectedIds.clear();
  fState.tab = "all";

  // DOM.
  const root = createDomRoot(FILES_DOM_IDS, FILES_DOM_EXTRAS, FILES_DOM_TAGS);
  // Add toolbar container that updateFilesToolbar() expects.
  const toolbar = root.querySelector("#fp-toolbar") ?? root.querySelector("#fp-select-all")?.parentElement;
  if (toolbar && !toolbar.classList.contains("files-toolbar")) {
    toolbar.classList.add("files-toolbar");
  }
  initFilesDom(root);

  // Deps.
  initFilesDeps({
    api: {
      listFiles: async () => ({ ok: true, data: { data: [], has_more: false } }),
      updateFile: async () => ({ ok: true }),
      deleteFile: async () => ({ ok: true }),
    } as any,
    notify: { info: () => {}, error: () => {}, warn: () => {}, success: () => {} } as any,
    dialogs: { confirm: async () => true } as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    upload: { upload: async () => ({}) } as any,
    download: {} as any,
    timers: mockTimers(),
    format: {
      date: () => "Jan 1",
      dateTime: (_ms: number, fmt?: string) => fmt === "short" ? "01/01/25" : "Jan 1, 2025, 12:00 PM",
      relativeTime: () => "1d ago",
      duration: () => "",
      number: () => "",
    } as any,
  });
});

// -- Helpers -----------------------------------------------------------------

function makeFile(id: string, name = "test.txt", bytes = 1024, overrides: any = {}): any {
  return {
    id, object: "file", bytes, created_at: 1700000000,
    filename: name, purpose: "assistants",
    mime_type: "text/plain", kind: "text", marker: "active",
    ...overrides,
  };
}

// ── renderFilesTable ─────────────────────────────────────────────────────────

describe("renderFilesTable", () => {
  test("renders correct number of rows", () => {
    fState.files = [makeFile("f1"), makeFile("f2"), makeFile("f3")];
    renderFilesTable();

    const dom = getFilesDom();
    const rows = dom.tbody.querySelectorAll(".files-row");
    expect(rows.length).toBe(3);
  });

  test("renders empty state when no files", () => {
    fState.files = [];
    renderFilesTable();

    const dom = getFilesDom();
    const empty = dom.tbody.querySelector("[data-empty-state]");
    expect(empty).not.toBeNull();
    expect(empty!.textContent).toBe("No files uploaded");
  });

  test("renders 'No archived files' empty state on archived tab", () => {
    fState.files = [];
    fState.tab = "archived";
    renderFilesTable();

    const dom = getFilesDom();
    const empty = dom.tbody.querySelector("[data-empty-state]");
    expect(empty!.textContent).toBe("No archived files");
  });

  test("renders 'No matching files' when search has no results", () => {
    fState.files = [];
    fState.searchQuery = "nonexistent";
    renderFilesTable();

    const dom = getFilesDom();
    const empty = dom.tbody.querySelector("[data-empty-state]");
    expect(empty!.textContent).toBe("No matching files");
  });

  test("row contains filename in name cell", () => {
    fState.files = [makeFile("f1", "myfile.txt")];
    renderFilesTable();

    const dom = getFilesDom();
    const nameSpan = dom.tbody.querySelector(".files-name-text");
    expect(nameSpan).not.toBeNull();
    expect(nameSpan!.textContent).toBe("myfile.txt");
  });

  test("row has data-id attribute", () => {
    fState.files = [makeFile("f1")];
    renderFilesTable();

    const dom = getFilesDom();
    const row = dom.tbody.querySelector<HTMLElement>(".files-row");
    expect(row!.dataset["id"]).toBe("f1");
  });

  test("selected row has files-row-selected class", () => {
    fState.files = [makeFile("f1")];
    fState.selectedIds.add("f1");
    renderFilesTable();

    const dom = getFilesDom();
    const row = dom.tbody.querySelector(".files-row");
    expect(row!.classList.contains("files-row-selected")).toBe(true);
  });

  test("previewed row has files-row-previewed class", () => {
    fState.files = [makeFile("f1")];
    fState.selectedFileId = "f1";
    renderFilesTable();

    const dom = getFilesDom();
    const row = dom.tbody.querySelector(".files-row");
    expect(row!.classList.contains("files-row-previewed")).toBe(true);
  });

  test("editing row shows input instead of text span", () => {
    fState.files = [makeFile("f1", "original.txt")];
    fState.editingFileId = "f1";
    renderFilesTable();

    const dom = getFilesDom();
    const input = dom.tbody.querySelector<HTMLInputElement>(".files-name-input");
    expect(input).not.toBeNull();
    expect(input!.value).toBe("original.txt");
    expect(input!.dataset["id"]).toBe("f1");
  });

  test("updates file count display", () => {
    fState.files = [makeFile("f1"), makeFile("f2")];
    renderFilesTable();

    const dom = getFilesDom();
    expect(dom.countEl.textContent).toBe("2 files");
  });

  test("singular count for single file", () => {
    fState.files = [makeFile("f1")];
    renderFilesTable();

    const dom = getFilesDom();
    expect(dom.countEl.textContent).toBe("1 file");
  });

  test("row has checkbox toggle button", () => {
    fState.files = [makeFile("f1")];
    renderFilesTable();

    const dom = getFilesDom();
    const checkBtn = dom.tbody.querySelector<HTMLElement>("[data-action='toggle']");
    expect(checkBtn).not.toBeNull();
    expect(checkBtn!.dataset["id"]).toBe("f1");
  });

  test("row has delete button", () => {
    fState.files = [makeFile("f1")];
    renderFilesTable();

    const dom = getFilesDom();
    const deleteBtn = dom.tbody.querySelector<HTMLElement>("[data-action='delete']");
    expect(deleteBtn).not.toBeNull();
    expect(deleteBtn!.dataset["id"]).toBe("f1");
  });

  test("row has download link", () => {
    fState.files = [makeFile("f1", "test.txt")];
    renderFilesTable();

    const dom = getFilesDom();
    const link = dom.tbody.querySelector<HTMLAnchorElement>("[data-action='download']");
    expect(link).not.toBeNull();
    expect(link!.download).toBe("test.txt");
  });

  test("kind badge has correct class for image files", () => {
    fState.files = [makeFile("f1", "photo.jpg", 2048, { kind: "image", mime_type: "image/jpeg" })];
    renderFilesTable();

    const dom = getFilesDom();
    const badge = dom.tbody.querySelector(".files-kind-image");
    expect(badge).not.toBeNull();
    expect(badge!.textContent).toBe("image");
  });

  test("kind badge has correct class for binary files", () => {
    fState.files = [makeFile("f1", "data.bin", 2048, { kind: undefined, mime_type: "application/octet-stream" })];
    renderFilesTable();

    const dom = getFilesDom();
    const badge = dom.tbody.querySelector(".files-kind-binary");
    expect(badge).not.toBeNull();
  });

  test("respects search filter", () => {
    fState.files = [makeFile("f1", "readme.md"), makeFile("f2", "index.ts")];
    fState.searchQuery = "readme";
    renderFilesTable();

    const dom = getFilesDom();
    const rows = dom.tbody.querySelectorAll(".files-row");
    expect(rows.length).toBe(1);
  });
});

// ── renderStats ──────────────────────────────────────────────────────────────

describe("renderStats", () => {
  test("shows file count and total size", () => {
    fState.files = [makeFile("f1", "a.txt", 512), makeFile("f2", "b.txt", 512)];
    renderStats();

    const dom = getFilesDom();
    expect(dom.statsEl.textContent).toContain("2 files");
    expect(dom.statsEl.textContent).toContain("1.0 KB");
  });

  test("shows 0 files when empty", () => {
    fState.files = [];
    renderStats();

    const dom = getFilesDom();
    expect(dom.statsEl.textContent).toContain("0 files");
    expect(dom.statsEl.textContent).toContain("0 B");
  });

  test("formats large sizes correctly", () => {
    fState.files = [makeFile("f1", "big.bin", 1024 * 1024 * 2.5)];
    renderStats();

    const dom = getFilesDom();
    expect(dom.statsEl.textContent).toContain("2.5 MB");
  });
});

// ── renderPreview ────────────────────────────────────────────────────────────

describe("renderPreview", () => {
  test("hides panel when no file selected", () => {
    fState.selectedFileId = null;
    renderPreview();

    const dom = getFilesDom();
    expect(dom.previewPanel.classList.contains("hidden")).toBe(true);
    expect(dom.previewContent.innerHTML).toBe("");
  });

  test("shows panel when file is selected", () => {
    fState.files = [makeFile("f1", "test.txt")];
    fState.selectedFileId = "f1";
    renderPreview();

    const dom = getFilesDom();
    expect(dom.previewPanel.classList.contains("hidden")).toBe(false);
  });

  test("shows metadata for selected file", () => {
    fState.files = [makeFile("f1", "test.txt", 2048)];
    fState.selectedFileId = "f1";
    renderPreview();

    const dom = getFilesDom();
    const meta = dom.previewContent.querySelector(".files-preview-meta");
    expect(meta).not.toBeNull();
    expect(meta!.textContent).toContain("test.txt");
    expect(meta!.textContent).toContain("2.0 KB");
  });

  test("shows image preview for image files", () => {
    fState.files = [makeFile("f1", "photo.jpg", 2048, {
      kind: "image", mime_type: "image/jpeg",
      image: { format: "jpeg", width: 800, height: 600, exif_orientation: 1, aspect_ratio: 1.333 },
    })];
    fState.selectedFileId = "f1";
    renderPreview();

    const dom = getFilesDom();
    const img = dom.previewContent.querySelector<HTMLImageElement>(".files-preview-img");
    expect(img).not.toBeNull();
    expect(img!.alt).toBe("photo.jpg");
  });

  test("shows text preview placeholder for text files", () => {
    fState.files = [makeFile("f1", "readme.md", 512, { kind: "text", mime_type: "text/plain" })];
    fState.selectedFileId = "f1";
    renderPreview();

    const dom = getFilesDom();
    const pre = dom.previewContent.querySelector(".files-preview-code");
    expect(pre).not.toBeNull();
    expect(pre!.textContent).toBe("Loading...");
  });

  test("shows no-preview for binary files", () => {
    fState.files = [makeFile("f1", "data.bin", 2048, { kind: "binary", mime_type: "application/octet-stream" })];
    fState.selectedFileId = "f1";
    renderPreview();

    const dom = getFilesDom();
    const empty = dom.previewContent.querySelector(".files-preview-empty");
    expect(empty).not.toBeNull();
    expect(empty!.textContent).toBe("No preview available");
  });

  test("shows image dimensions in metadata", () => {
    fState.files = [makeFile("f1", "photo.jpg", 2048, {
      kind: "image", mime_type: "image/jpeg",
      image: { format: "jpeg", width: 1920, height: 1080, exif_orientation: 1, aspect_ratio: 1.778 },
    })];
    fState.selectedFileId = "f1";
    renderPreview();

    const dom = getFilesDom();
    const meta = dom.previewContent.querySelector(".files-preview-meta");
    expect(meta!.textContent).toContain("1920\u00d71080");
    expect(meta!.textContent).toContain("jpeg");
  });

  test("shows MIME type in metadata when present", () => {
    fState.files = [makeFile("f1", "test.txt", 512, { mime_type: "text/plain" })];
    fState.selectedFileId = "f1";
    renderPreview();

    const dom = getFilesDom();
    const meta = dom.previewContent.querySelector(".files-preview-meta");
    expect(meta!.textContent).toContain("text/plain");
  });

  test("preview actions include rename and delete buttons", () => {
    fState.files = [makeFile("f1", "test.txt")];
    fState.selectedFileId = "f1";
    renderPreview();

    const dom = getFilesDom();
    const renameBtn = dom.previewContent.querySelector<HTMLElement>("[data-action='rename']");
    const deleteBtn = dom.previewContent.querySelector<HTMLElement>("[data-action='delete']");
    expect(renameBtn).not.toBeNull();
    expect(deleteBtn).not.toBeNull();
    expect(renameBtn!.dataset["id"]).toBe("f1");
    expect(deleteBtn!.dataset["id"]).toBe("f1");
  });

  test("hides when selected file not found in state", () => {
    fState.files = [];
    fState.selectedFileId = "nonexistent";
    renderPreview();

    const dom = getFilesDom();
    expect(dom.previewPanel.classList.contains("hidden")).toBe(true);
  });
});

// ── syncFilesTabs ────────────────────────────────────────────────────────────

describe("syncFilesTabs", () => {
  test("all tab gets active class when tab is all", () => {
    fState.tab = "all";
    syncFilesTabs();

    const dom = getFilesDom();
    expect(dom.tabAll.className).toContain("active");
    expect(dom.tabArchived.className).not.toContain("active");
  });

  test("archived tab gets active class when tab is archived", () => {
    fState.tab = "archived";
    syncFilesTabs();

    const dom = getFilesDom();
    expect(dom.tabArchived.className).toContain("active");
    expect(dom.tabAll.className).not.toContain("active");
  });

  test("archive button visible on all tab", () => {
    fState.tab = "all";
    syncFilesTabs();

    const dom = getFilesDom();
    expect(dom.archiveBtn.classList.contains("hidden")).toBe(false);
    expect(dom.restoreBtn.classList.contains("hidden")).toBe(true);
  });

  test("restore button visible on archived tab", () => {
    fState.tab = "archived";
    syncFilesTabs();

    const dom = getFilesDom();
    expect(dom.archiveBtn.classList.contains("hidden")).toBe(true);
    expect(dom.restoreBtn.classList.contains("hidden")).toBe(false);
  });
});

// ── updateFilesToolbar ───────────────────────────────────────────────────────

describe("updateFilesToolbar", () => {
  test("disables action buttons when no selection", () => {
    fState.files = [makeFile("f1")];
    fState.selectedIds.clear();
    updateFilesToolbar();

    const dom = getFilesDom();
    expect(dom.deleteBtn.disabled).toBe(true);
    expect(dom.archiveBtn.disabled).toBe(true);
  });

  test("enables action buttons when files selected", () => {
    fState.files = [makeFile("f1")];
    fState.selectedIds.add("f1");
    updateFilesToolbar();

    const dom = getFilesDom();
    expect(dom.deleteBtn.disabled).toBe(false);
    expect(dom.archiveBtn.disabled).toBe(false);
  });

  test("shows cancel button when files selected", () => {
    fState.files = [makeFile("f1")];
    fState.selectedIds.add("f1");
    updateFilesToolbar();

    const dom = getFilesDom();
    expect(dom.cancelBtn.classList.contains("hidden")).toBe(false);
  });

  test("hides cancel button when no selection", () => {
    fState.selectedIds.clear();
    updateFilesToolbar();

    const dom = getFilesDom();
    expect(dom.cancelBtn.classList.contains("hidden")).toBe(true);
  });

  test("select all button shows 'Deselect All' when all selected", () => {
    fState.files = [makeFile("f1"), makeFile("f2")];
    fState.selectedIds.add("f1");
    fState.selectedIds.add("f2");
    updateFilesToolbar();

    const dom = getFilesDom();
    expect(dom.selectAllBtn.textContent).toBe("Deselect All");
  });

  test("select all button shows 'Select All' when not all selected", () => {
    fState.files = [makeFile("f1"), makeFile("f2")];
    fState.selectedIds.add("f1");
    updateFilesToolbar();

    const dom = getFilesDom();
    expect(dom.selectAllBtn.textContent).toBe("Select All");
  });
});
