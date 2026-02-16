import { describe, test, expect, beforeEach } from "bun:test";
import {
  loadFiles,
  getFilteredFiles,
  renameFile,
  deleteFile,
  uploadFiles,
  archiveFiles,
  restoreFiles,
  deleteFiles,
} from "../../../src/plugins/files/data.ts";
import { fState } from "../../../src/plugins/files/state.ts";
import { initFilesDeps } from "../../../src/plugins/files/deps.ts";
import { initFilesDom } from "../../../src/plugins/files/dom.ts";
import { createDomRoot, FILES_DOM_IDS, FILES_DOM_EXTRAS, FILES_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";

/**
 * Tests for files plugin data operations — loading, filtering, rename,
 * delete, upload, and bulk archive/restore/delete.
 *
 * Strategy: mock API (records calls), upload service, dialogs, and
 * notifications. DOM is a minimal root with expected element IDs.
 */

// -- Mock state --------------------------------------------------------------

let apiCalls: { method: string; args: unknown[] }[];
let uploadCalls: { file: File; purpose: string }[];
let notif: ReturnType<typeof mockNotifications>;
let confirmResult: boolean;

let listFilesResult: any;
let updateFileResult: any;
let deleteFileResult: any;

beforeEach(() => {
  apiCalls = [];
  uploadCalls = [];
  notif = mockNotifications();
  confirmResult = true;

  listFilesResult = { ok: true, data: { data: [], has_more: false } };
  updateFileResult = { ok: true, data: makeFile("f1", "renamed.txt") };
  deleteFileResult = { ok: true };

  // Reset state.
  fState.files = [];
  fState.isLoading = false;
  fState.searchQuery = "";
  fState.selectedFileId = null;
  fState.editingFileId = null;
  fState.selectedIds.clear();
  fState.tab = "all";

  // DOM.
  initFilesDom(createDomRoot(FILES_DOM_IDS, FILES_DOM_EXTRAS, FILES_DOM_TAGS));

  // Deps.
  initFilesDeps({
    api: {
      listFiles: async (limit: number, marker: string) => {
        apiCalls.push({ method: "listFiles", args: [limit, marker] });
        return listFilesResult;
      },
      updateFile: async (id: string, patch: any) => {
        apiCalls.push({ method: "updateFile", args: [id, patch] });
        return updateFileResult;
      },
      deleteFile: async (id: string) => {
        apiCalls.push({ method: "deleteFile", args: [id] });
        return deleteFileResult;
      },
    } as any,
    notify: notif.mock as any,
    dialogs: {
      confirm: async () => confirmResult,
    } as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    upload: {
      upload: async (file: File, purpose: string) => {
        uploadCalls.push({ file, purpose });
        return { id: `uploaded-${file.name}` };
      },
    } as any,
    download: {} as any,
    timers: mockTimers(),
    format: {
      date: () => "",
      dateTime: () => "Jan 1, 2025",
      relativeTime: () => "",
      duration: () => "",
      number: () => "",
    } as any,
  });
});

// -- Helpers -----------------------------------------------------------------

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

function makeFile(id: string, name = "test.txt", bytes = 1024): any {
  return {
    id,
    object: "file",
    bytes,
    created_at: 1700000000,
    filename: name,
    purpose: "assistants",
    mime_type: "text/plain",
    kind: "text",
    marker: "active",
  };
}

// ── loadFiles ────────────────────────────────────────────────────────────────

describe("loadFiles", () => {
  test("calls API with active marker on all tab", async () => {
    fState.tab = "all";
    await loadFiles();

    expect(apiCalls[0]!.method).toBe("listFiles");
    expect(apiCalls[0]!.args[1]).toBe("active");
  });

  test("calls API with archived marker on archived tab", async () => {
    fState.tab = "archived";
    await loadFiles();

    expect(apiCalls[0]!.method).toBe("listFiles");
    expect(apiCalls[0]!.args[1]).toBe("archived");
  });

  test("populates fState.files on success", async () => {
    listFilesResult = {
      ok: true,
      data: { data: [makeFile("f1"), makeFile("f2")], has_more: false },
    };
    await loadFiles();

    expect(fState.files.length).toBe(2);
    expect(fState.files[0]!.id).toBe("f1");
  });

  test("clears files and notifies on API failure", async () => {
    fState.files = [makeFile("old")];
    listFilesResult = { ok: false, error: "Server error" };
    await loadFiles();

    expect(fState.files.length).toBe(0);
    expect(notif.messages.some((m) => m.type === "error")).toBe(true);
  });

  test("sets and clears isLoading flag", async () => {
    let wasLoading = false;
    const origListFiles = listFilesResult;
    initFilesDeps({
      api: {
        listFiles: async () => {
          wasLoading = fState.isLoading;
          return origListFiles;
        },
        updateFile: async () => updateFileResult,
        deleteFile: async () => deleteFileResult,
      } as any,
      notify: notif.mock as any,
      dialogs: { confirm: async () => confirmResult } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      upload: { upload: async () => ({}) } as any,
      download: {} as any,
      timers: mockTimers(),
      format: { date: () => "", dateTime: () => "", relativeTime: () => "", duration: () => "", number: () => "" } as any,
    });

    await loadFiles();
    expect(wasLoading).toBe(true);
    expect(fState.isLoading).toBe(false);
  });
});

// ── getFilteredFiles ─────────────────────────────────────────────────────────

describe("getFilteredFiles", () => {
  test("returns all files when query is empty", () => {
    fState.files = [makeFile("f1", "alpha.txt"), makeFile("f2", "beta.txt")];
    fState.searchQuery = "";
    expect(getFilteredFiles().length).toBe(2);
  });

  test("filters by case-insensitive substring", () => {
    fState.files = [
      makeFile("f1", "README.md"),
      makeFile("f2", "index.ts"),
      makeFile("f3", "readme.txt"),
    ];
    fState.searchQuery = "readme";
    const result = getFilteredFiles();
    expect(result.length).toBe(2);
    expect(result.map((f) => f.id)).toEqual(["f1", "f3"]);
  });

  test("returns empty array when no match", () => {
    fState.files = [makeFile("f1", "test.txt")];
    fState.searchQuery = "nonexistent";
    expect(getFilteredFiles().length).toBe(0);
  });

  test("trims whitespace from query", () => {
    fState.files = [makeFile("f1", "test.txt")];
    fState.searchQuery = "  test  ";
    expect(getFilteredFiles().length).toBe(1);
  });
});

// ── renameFile ──────────────────────────────────────────────────────────────

describe("renameFile", () => {
  test("calls updateFile API with trimmed name", async () => {
    fState.files = [makeFile("f1", "old.txt")];
    await renameFile("f1", "  new.txt  ");

    expect(apiCalls[0]!.method).toBe("updateFile");
    expect(apiCalls[0]!.args).toEqual(["f1", { filename: "new.txt" }]);
  });

  test("updates local state on success", async () => {
    fState.files = [makeFile("f1", "old.txt")];
    updateFileResult = { ok: true, data: makeFile("f1", "new.txt") };
    await renameFile("f1", "new.txt");

    expect(fState.files[0]!.filename).toBe("new.txt");
  });

  test("shows success notification", async () => {
    fState.files = [makeFile("f1", "old.txt")];
    await renameFile("f1", "new.txt");

    expect(notif.messages.some((m) => m.type === "success" && m.msg.includes("renamed"))).toBe(true);
  });

  test("no-op when name is empty or whitespace", async () => {
    await renameFile("f1", "   ");
    expect(apiCalls.length).toBe(0);
  });

  test("clears editing state after rename", async () => {
    fState.editingFileId = "f1";
    fState.files = [makeFile("f1", "old.txt")];
    await renameFile("f1", "new.txt");

    expect(fState.editingFileId).toBeNull();
  });

  test("shows error on API failure", async () => {
    fState.files = [makeFile("f1", "old.txt")];
    updateFileResult = { ok: false, error: "Permission denied" };
    await renameFile("f1", "new.txt");

    expect(notif.messages.some((m) => m.type === "error")).toBe(true);
  });
});

// ── deleteFile ──────────────────────────────────────────────────────────────

describe("deleteFile", () => {
  test("shows confirmation dialog", async () => {
    fState.files = [makeFile("f1", "test.txt")];
    await deleteFile("f1");

    expect(apiCalls[0]!.method).toBe("deleteFile");
  });

  test("aborts if user cancels dialog", async () => {
    fState.files = [makeFile("f1", "test.txt")];
    confirmResult = false;
    await deleteFile("f1");

    expect(apiCalls.length).toBe(0);
  });

  test("removes file from state on success", async () => {
    fState.files = [makeFile("f1"), makeFile("f2")];
    await deleteFile("f1");

    expect(fState.files.length).toBe(1);
    expect(fState.files[0]!.id).toBe("f2");
  });

  test("clears preview when deleted file was previewed", async () => {
    fState.files = [makeFile("f1")];
    fState.selectedFileId = "f1";
    await deleteFile("f1");

    expect(fState.selectedFileId).toBeNull();
  });

  test("removes from selectedIds", async () => {
    fState.files = [makeFile("f1")];
    fState.selectedIds.add("f1");
    await deleteFile("f1");

    expect(fState.selectedIds.has("f1")).toBe(false);
  });

  test("shows error on API failure", async () => {
    fState.files = [makeFile("f1")];
    deleteFileResult = { ok: false, error: "Not found" };
    await deleteFile("f1");

    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("Not found"))).toBe(true);
    // File should NOT be removed from state.
    expect(fState.files.length).toBe(1);
  });
});

// ── uploadFiles ─────────────────────────────────────────────────────────────

describe("uploadFiles", () => {
  test("calls upload service for each file", async () => {
    const files = makeFileList(
      new File(["content1"], "a.txt"),
      new File(["content2"], "b.txt"),
    );

    await uploadFiles(files);

    expect(uploadCalls.length).toBe(2);
    expect(uploadCalls[0]!.purpose).toBe("assistants");
    expect(uploadCalls[1]!.purpose).toBe("assistants");
  });

  test("shows success notification with count", async () => {
    const files = makeFileList(new File(["content"], "test.txt"));

    await uploadFiles(files);

    expect(notif.messages.some((m) => m.type === "success" && m.msg.includes("1 file"))).toBe(true);
  });

  test("shows error when upload fails", async () => {
    initFilesDeps({
      api: {
        listFiles: async () => listFilesResult,
        updateFile: async () => updateFileResult,
        deleteFile: async () => deleteFileResult,
      } as any,
      notify: notif.mock as any,
      dialogs: { confirm: async () => confirmResult } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      upload: {
        upload: async () => { throw new Error("Upload failed"); },
      } as any,
      download: {} as any,
      timers: mockTimers(),
      format: { date: () => "", dateTime: () => "", relativeTime: () => "", duration: () => "", number: () => "" } as any,
    });

    const files = makeFileList(new File(["content"], "test.txt"));

    await uploadFiles(files);

    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("Upload failed"))).toBe(true);
  });
});

// ── archiveFiles ────────────────────────────────────────────────────────────

describe("archiveFiles", () => {
  test("archives all selected files", async () => {
    fState.selectedIds.add("f1");
    fState.selectedIds.add("f2");
    await archiveFiles();

    const archiveCalls = apiCalls.filter((c) => c.method === "updateFile");
    expect(archiveCalls.length).toBe(2);
    expect((archiveCalls[0]!.args[1] as any).marker).toBe("archived");
  });

  test("clears selection after archive", async () => {
    fState.selectedIds.add("f1");
    await archiveFiles();

    expect(fState.selectedIds.size).toBe(0);
    expect(fState.selectedFileId).toBeNull();
  });

  test("no-op when no selection", async () => {
    await archiveFiles();
    expect(apiCalls.length).toBe(0);
  });

  test("shows success notification", async () => {
    fState.selectedIds.add("f1");
    await archiveFiles();

    expect(notif.messages.some((m) => m.type === "success" && m.msg.includes("Archived"))).toBe(true);
  });
});

// ── restoreFiles ────────────────────────────────────────────────────────────

describe("restoreFiles", () => {
  test("restores all selected files", async () => {
    fState.selectedIds.add("f1");
    fState.selectedIds.add("f2");
    await restoreFiles();

    const restoreCalls = apiCalls.filter((c) => c.method === "updateFile");
    expect(restoreCalls.length).toBe(2);
    expect((restoreCalls[0]!.args[1] as any).marker).toBe("active");
  });

  test("clears selection after restore", async () => {
    fState.selectedIds.add("f1");
    await restoreFiles();

    expect(fState.selectedIds.size).toBe(0);
  });

  test("no-op when no selection", async () => {
    await restoreFiles();
    expect(apiCalls.length).toBe(0);
  });
});

// ── deleteFiles (bulk) ─────────────────────────────────────────────────────

describe("deleteFiles", () => {
  test("deletes all selected files after confirmation", async () => {
    fState.selectedIds.add("f1");
    fState.selectedIds.add("f2");
    await deleteFiles();

    const delCalls = apiCalls.filter((c) => c.method === "deleteFile");
    expect(delCalls.length).toBe(2);
  });

  test("aborts if user cancels dialog", async () => {
    fState.selectedIds.add("f1");
    confirmResult = false;
    await deleteFiles();

    expect(apiCalls.length).toBe(0);
  });

  test("clears selection after delete", async () => {
    fState.selectedIds.add("f1");
    await deleteFiles();

    expect(fState.selectedIds.size).toBe(0);
  });

  test("no-op when no selection", async () => {
    await deleteFiles();
    expect(apiCalls.length).toBe(0);
  });

  test("shows success notification with count", async () => {
    fState.selectedIds.add("f1");
    fState.selectedIds.add("f2");
    await deleteFiles();

    expect(notif.messages.some((m) => m.type === "success" && m.msg.includes("2 files"))).toBe(true);
  });
});
