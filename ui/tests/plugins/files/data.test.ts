import { describe, test, expect, beforeEach } from "bun:test";
import {
  loadFiles,
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
let batchFilesResult: any;

beforeEach(() => {
  apiCalls = [];
  uploadCalls = [];
  notif = mockNotifications();
  confirmResult = true;

  listFilesResult = { ok: true, data: { data: [], total: 0 } };
  updateFileResult = { ok: true, data: makeFile("f1", "renamed.txt") };
  deleteFileResult = { ok: true };
  batchFilesResult = { ok: true };

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

  // Deps.
  initFilesDeps({
    api: {
      listFiles: async (opts?: any) => {
        apiCalls.push({ method: "listFiles", args: [opts] });
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
      batchFiles: async (opts: any) => {
        apiCalls.push({ method: "batchFiles", args: [opts] });
        return batchFilesResult;
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
    expect((apiCalls[0]!.args[0] as any).marker).toBe("active");
  });

  test("calls API with archived marker on archived tab", async () => {
    fState.tab = "archived";
    await loadFiles();

    expect(apiCalls[0]!.method).toBe("listFiles");
    expect((apiCalls[0]!.args[0] as any).marker).toBe("archived");
  });

  test("populates fState.files on success", async () => {
    listFilesResult = {
      ok: true,
      data: { data: [makeFile("f1"), makeFile("f2")], total: 2 },
    };
    await loadFiles();

    expect(fState.files.length).toBe(2);
    expect(fState.files[0]!.id).toBe("f1");
  });

  test("keeps existing files and notifies on API failure", async () => {
    fState.files = [makeFile("old")];
    listFilesResult = { ok: false, error: "Server error" };
    await loadFiles();

    // Files are preserved on failure (server-side pagination — don't discard cached data).
    expect(fState.files.length).toBe(1);
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

  test("partial batch failure — succeeds around the failure", async () => {
    let callIndex = 0;
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
        upload: async (file: File) => {
          const idx = callIndex++;
          if (idx === 1) throw new Error("Disk full");
          uploadCalls.push({ file, purpose: "assistants" });
          return { id: `uploaded-${file.name}` };
        },
      } as any,
      download: {} as any,
      timers: mockTimers(),
      format: { date: () => "", dateTime: () => "", relativeTime: () => "", duration: () => "", number: () => "" } as any,
    });

    const files = makeFileList(
      new File(["a"], "first.txt"),
      new File(["b"], "second.txt"),
      new File(["c"], "third.txt"),
    );

    await uploadFiles(files);

    // 1st and 3rd succeed; 2nd fails.
    expect(uploadCalls.length).toBe(2);
    expect(uploadCalls[0]!.file.name).toBe("first.txt");
    expect(uploadCalls[1]!.file.name).toBe("third.txt");

    // Error notification for the failed upload.
    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("Disk full"))).toBe(true);
    // Success notification for the 2 that completed.
    expect(notif.messages.some((m) => m.type === "success" && m.msg.includes("2 file"))).toBe(true);
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
  test("archives all selected files via batch API", async () => {
    fState.selectedIds.add("f1");
    fState.selectedIds.add("f2");
    await archiveFiles();

    const batchCalls = apiCalls.filter((c) => c.method === "batchFiles");
    expect(batchCalls.length).toBe(1);
    expect((batchCalls[0]!.args[0] as any).action).toBe("archive");
    expect((batchCalls[0]!.args[0] as any).ids).toContain("f1");
    expect((batchCalls[0]!.args[0] as any).ids).toContain("f2");
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
  test("restores all selected files via batch API", async () => {
    fState.selectedIds.add("f1");
    fState.selectedIds.add("f2");
    await restoreFiles();

    const batchCalls = apiCalls.filter((c) => c.method === "batchFiles");
    expect(batchCalls.length).toBe(1);
    expect((batchCalls[0]!.args[0] as any).action).toBe("unarchive");
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
  test("deletes all selected files after confirmation via batch API", async () => {
    fState.selectedIds.add("f1");
    fState.selectedIds.add("f2");
    await deleteFiles();

    const batchCalls = apiCalls.filter((c) => c.method === "batchFiles");
    expect(batchCalls.length).toBe(1);
    expect((batchCalls[0]!.args[0] as any).action).toBe("delete");
    expect((batchCalls[0]!.args[0] as any).ids).toContain("f1");
    expect((batchCalls[0]!.args[0] as any).ids).toContain("f2");
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
