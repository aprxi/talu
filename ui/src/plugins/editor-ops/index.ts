/**
 * Editor plugin — interactive file browser + text editor.
 *
 * Browse the agent filesystem, open files for editing, save manually or
 * with auto-save.  Open file path is encoded in the URL hash so you can
 * open the same file in multiple tabs.  Instant cross-window sync via
 * WebSocket pubsub relay.  Stat-polling for external/agent disk changes.
 */

import type {
  AgentLsEntry,
  Disposable,
  PluginContext,
  PluginDefinition,
} from "../../kernel/types.ts";
import { navigate, onRouteChange, parseHash } from "../../kernel/system/router.ts";
import { connectPubSub, type PubSubClient } from "../../kernel/system/pubsub.ts";

/* ------------------------------------------------------------------ */
/*  State                                                              */
/* ------------------------------------------------------------------ */

interface EditorState {
  cwd: string | null;
  entries: AgentLsEntry[];
  openFile: string | null;
  dirty: boolean;
  autoSave: boolean;
  saving: boolean;
  lastModifiedAt: number;
}

const STORAGE_AUTO_SAVE = "editor.autoSave";
const AUTO_SAVE_DELAY_MS = 800;
const STAT_POLL_INTERVAL_MS = 2000;

/* ------------------------------------------------------------------ */
/*  Icons (inline SVG)                                                 */
/* ------------------------------------------------------------------ */

const FOLDER_ICON = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>`;
const FILE_ICON = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`;

/* ------------------------------------------------------------------ */
/*  DOM builder                                                        */
/* ------------------------------------------------------------------ */

function buildEditorDom(root: HTMLElement): void {
  root.innerHTML = `
    <div id="eop-root" style="display:flex;flex-direction:column;height:100%;overflow:hidden;">
      <!-- Nav bar -->
      <div style="display:flex;gap:6px;padding:8px 12px;border-bottom:1px solid var(--border);align-items:center;">
        <button id="eop-up-btn" class="btn btn-ghost btn-sm" title="Parent directory" style="padding:2px 6px;">../</button>
        <input id="eop-path" class="form-input form-input-sm" type="text" value="." style="flex:1;min-width:0;" placeholder="Directory path">
        <button id="eop-ls-btn" class="btn btn-ghost btn-sm">List</button>
      </div>
      <!-- Main area: file list + editor -->
      <div style="display:flex;flex:1;min-height:0;overflow:hidden;">
        <!-- File list -->
        <div id="eop-file-list" style="width:220px;min-width:140px;max-width:320px;overflow-y:auto;border-right:1px solid var(--border);padding:4px 0;font-size:13px;"></div>
        <!-- Editor pane -->
        <div style="flex:1;display:flex;flex-direction:column;min-width:0;">
          <!-- Toolbar -->
          <div id="eop-toolbar" style="display:none;padding:6px 12px;border-bottom:1px solid var(--border);align-items:center;gap:8px;font-size:12px;">
            <span id="eop-filename" style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:var(--font-mono);color:var(--text);"></span>
            <button id="eop-save-btn" class="btn btn-primary btn-sm" disabled>Save</button>
            <label style="display:flex;align-items:center;gap:4px;color:var(--text-muted);cursor:pointer;white-space:nowrap;">
              <input id="eop-autosave-cb" type="checkbox">
              <span>Auto-save</span>
            </label>
            <span id="eop-status" style="color:var(--text-muted);white-space:nowrap;min-width:60px;text-align:right;"></span>
          </div>
          <!-- Textarea -->
          <textarea id="eop-editor" class="form-textarea mono" style="flex:1;resize:none;border:none;border-radius:0;margin:0;font-size:13px;line-height:1.5;tab-size:4;" placeholder="Select a file to edit" disabled></textarea>
        </div>
      </div>
    </div>
  `;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function findEl<T extends HTMLElement>(root: HTMLElement, id: string): T {
  const found = root.querySelector<T>(`#${id}`);
  if (!found) throw new Error(`editor: missing #${id}`);
  return found;
}

function parentDir(path: string): string {
  const parts = path.replace(/\/+$/, "").split("/");
  parts.pop();
  return parts.length > 0 ? parts.join("/") : ".";
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} K`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} M`;
}

function encodeFilePath(path: string): string {
  return btoa(unescape(encodeURIComponent(path)));
}

function decodeFilePath(encoded: string): string | null {
  try {
    return decodeURIComponent(escape(atob(encoded)));
  } catch {
    return null;
  }
}

function editorTopic(path: string): string {
  return `editor:${path}`;
}

/* ------------------------------------------------------------------ */
/*  Plugin                                                             */
/* ------------------------------------------------------------------ */

export const editorOpsPlugin: PluginDefinition = {
  manifest: {
    id: "talu.editorops",
    name: "Editor",
    version: "0.4.0",
    builtin: true,
    permissions: ["filesystem"],
    requiresCapabilities: ["filesystem"],
    contributes: {
      mode: { key: "editor", label: "Editor" },
    },
  },

  register(_ctx: PluginContext): void {},

  async run(ctx: PluginContext): Promise<void> {
    buildEditorDom(ctx.container);

    /* --- DOM refs --- */
    const pathInput = findEl<HTMLInputElement>(ctx.container, "eop-path");
    const lsBtn = findEl<HTMLButtonElement>(ctx.container, "eop-ls-btn");
    const upBtn = findEl<HTMLButtonElement>(ctx.container, "eop-up-btn");
    const fileList = findEl<HTMLDivElement>(ctx.container, "eop-file-list");
    const toolbar = findEl<HTMLDivElement>(ctx.container, "eop-toolbar");
    const filenameEl = findEl<HTMLSpanElement>(ctx.container, "eop-filename");
    const saveBtn = findEl<HTMLButtonElement>(ctx.container, "eop-save-btn");
    const autoSaveCb = findEl<HTMLInputElement>(ctx.container, "eop-autosave-cb");
    const statusEl = findEl<HTMLSpanElement>(ctx.container, "eop-status");
    const editorEl = findEl<HTMLTextAreaElement>(ctx.container, "eop-editor");

    /* --- State --- */
    const state: EditorState = {
      cwd: null,
      entries: [],
      openFile: null,
      dirty: false,
      autoSave: false,
      saving: false,
      lastModifiedAt: 0,
    };

    let autoSaveTimer: Disposable | null = null;
    let statPollTimer: Disposable | null = null;
    let suppressPubsub = false;

    /* --- PubSub for instant cross-window sync --- */
    const pubsub: PubSubClient = connectPubSub();
    let pubsubSub: Disposable | null = null; // current topic message handler

    function subscribeTopic(path: string): void {
      // Unsubscribe from previous topic.
      unsubscribeTopic();
      const topic = editorTopic(path);
      pubsub.subscribe(topic);
      pubsubSub = pubsub.onMessage(topic, (data) => {
        const msg = data as { type?: string; text?: string } | null;
        if (!msg) return;

        if (msg.type === "saved" && !state.dirty) {
          void reloadOpenFile();
          return;
        }

        if (msg.text != null) {
          suppressPubsub = true;
          editorEl.value = msg.text;
          suppressPubsub = false;
          setStatus("Live");
        }
      });
    }

    function unsubscribeTopic(): void {
      if (pubsubSub) {
        pubsubSub.dispose();
        pubsubSub = null;
      }
      if (state.openFile) {
        pubsub.unsubscribe(editorTopic(state.openFile));
      }
    }

    /* --- Render helpers --- */

    function setStatus(text: string): void {
      statusEl.textContent = text;
    }

    function updateSaveBtn(): void {
      saveBtn.disabled = !state.dirty || state.saving;
    }

    function isNoWorkdirError(err: unknown): boolean {
      if (!(err instanceof Error)) return false;
      return err.message.includes("no_workdir") || err.message.includes("no workdir was passed");
    }

    function showNoWorkdirState(): void {
      state.cwd = null;
      state.entries = [];
      state.openFile = null;
      state.dirty = false;
      state.saving = false;
      state.lastModifiedAt = 0;
      unsubscribeTopic();
      stopStatPoll();
      toolbar.style.display = "none";
      filenameEl.textContent = "";
      filenameEl.title = "";
      pathInput.value = "";
      pathInput.placeholder = "No workdir was passed";
      pathInput.disabled = true;
      lsBtn.disabled = true;
      upBtn.disabled = true;
      editorEl.value = "";
      editorEl.placeholder = "No workdir was passed";
      editorEl.disabled = true;
      setStatus("No workdir");
      updateSaveBtn();
      renderFileList();
    }

    function renderFileList(): void {
      fileList.innerHTML = "";
      if (state.cwd === null) {
        const empty = document.createElement("div");
        empty.style.cssText = "padding:12px;color:var(--text-muted);font-size:12px;text-align:center;";
        empty.textContent = "No workdir was passed";
        fileList.appendChild(empty);
        return;
      }
      const sorted = [...state.entries].sort((a, b) => {
        if (a.isDir !== b.isDir) return a.isDir ? -1 : 1;
        return a.name.localeCompare(b.name);
      });

      for (const entry of sorted) {
        const row = document.createElement("div");
        row.style.cssText =
          "display:flex;align-items:center;gap:6px;padding:3px 10px;cursor:pointer;" +
          "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
        row.addEventListener("mouseenter", () => { row.style.background = "var(--bg-hover)"; });
        row.addEventListener("mouseleave", () => { row.style.background = ""; });

        const icon = document.createElement("span");
        icon.style.cssText = "flex-shrink:0;display:flex;color:var(--text-muted);";
        icon.innerHTML = entry.isDir ? FOLDER_ICON : FILE_ICON;

        const name = document.createElement("span");
        name.style.cssText = "flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;color:var(--text);";
        name.textContent = entry.isDir ? entry.name + "/" : entry.name;

        row.append(icon, name);

        if (!entry.isDir) {
          const size = document.createElement("span");
          size.style.cssText = "flex-shrink:0;font-size:11px;color:var(--text-muted);";
          size.textContent = formatSize(entry.size);
          row.appendChild(size);
        }

        // Build full path: cwd + entry name (entry.path is just the name for non-recursive ls).
        const fullPath = state.cwd === "." ? entry.name : `${state.cwd}/${entry.name}`;

        if (entry.isDir) {
          row.addEventListener("click", () => navigateDir(fullPath));
        } else {
          row.addEventListener("click", () => { void openFile(fullPath); });
        }

        fileList.appendChild(row);
      }

      if (sorted.length === 0) {
        const empty = document.createElement("div");
        empty.style.cssText = "padding:12px;color:var(--text-muted);font-size:12px;text-align:center;";
        empty.textContent = "Empty directory";
        fileList.appendChild(empty);
      }
    }

    /* --- Actions --- */

    async function listDir(path: string): Promise<void> {
      if (state.cwd === null && ctx.agent.cwd === null) {
        showNoWorkdirState();
        return;
      }
      try {
        const result = await ctx.agent.fs.ls(path, { recursive: false, limit: 500 });
        state.cwd = path;
        state.entries = result.entries;
        pathInput.disabled = false;
        lsBtn.disabled = false;
        upBtn.disabled = false;
        pathInput.value = path;
        pathInput.placeholder = "Directory path";
        renderFileList();
      } catch (err) {
        if (isNoWorkdirError(err)) {
          showNoWorkdirState();
          return;
        }
        ctx.notifications.error(`List failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    function navigateDir(path: string): void {
      void listDir(path);
    }

    async function openFile(path: string): Promise<void> {
      // Update URL and subscribe to pubsub immediately (before async I/O).
      state.openFile = path;
      navigate({ mode: "editor", sub: encodeFilePath(path), resource: null }, { replace: true });
      subscribeTopic(path);
      toolbar.style.display = "flex";
      filenameEl.textContent = path;
      filenameEl.title = path;
      setStatus("Loading...");

      try {
        const result = await ctx.agent.fs.readFile(path, { encoding: "utf-8", maxBytes: 1_048_576 });
        state.dirty = false;
        editorEl.value = result.content;
        editorEl.disabled = false;
        updateSaveBtn();

        if (result.truncated) {
          setStatus("Truncated!");
          ctx.notifications.warning(`File truncated at ${formatSize(result.size)}`);
        } else {
          setStatus("Loaded");
        }

        // Record mtime for stat-polling.
        const stat = await ctx.agent.fs.stat(path).catch(() => null);
        state.lastModifiedAt = stat?.modifiedAt ?? 0;

        // Start stat-polling for external/agent disk changes.
        startStatPoll();
      } catch (err) {
        if (isNoWorkdirError(err)) {
          showNoWorkdirState();
          return;
        }
        setStatus("Error");
        ctx.notifications.error(`Open failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    async function reloadOpenFile(): Promise<void> {
      if (!state.openFile) return;
      try {
        const result = await ctx.agent.fs.readFile(state.openFile, { encoding: "utf-8", maxBytes: 1_048_576 });
        editorEl.value = result.content;
        state.dirty = false;
        updateSaveBtn();
        setStatus("Reloaded");

        const stat = await ctx.agent.fs.stat(state.openFile).catch(() => null);
        state.lastModifiedAt = stat?.modifiedAt ?? 0;
      } catch {
        setStatus("Reload failed");
      }
    }

    async function saveFile(): Promise<void> {
      if (!state.openFile || state.saving) return;
      state.saving = true;
      updateSaveBtn();
      setStatus("Saving...");

      try {
        await ctx.agent.fs.writeFile(state.openFile, editorEl.value, { encoding: "utf-8" });
        state.dirty = false;
        state.saving = false;
        updateSaveBtn();
        setStatus("Saved");

        // Update mtime so stat-poll doesn't trigger a reload for our own save.
        const stat = await ctx.agent.fs.stat(state.openFile).catch(() => null);
        state.lastModifiedAt = stat?.modifiedAt ?? 0;

        // Notify other windows via pubsub.
        pubsub.publish(editorTopic(state.openFile), { type: "saved" });
      } catch (err) {
        if (isNoWorkdirError(err)) {
          showNoWorkdirState();
          return;
        }
        state.saving = false;
        updateSaveBtn();
        setStatus("Save failed");
        ctx.notifications.error(`Save failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    function scheduleAutoSave(): void {
      if (autoSaveTimer) {
        autoSaveTimer.dispose();
        autoSaveTimer = null;
      }
      if (!state.autoSave || !state.dirty) return;
      autoSaveTimer = ctx.timers.setTimeout(() => {
        autoSaveTimer = null;
        void saveFile();
      }, AUTO_SAVE_DELAY_MS);
    }

    /* --- Stat polling for external / agent disk changes --- */

    function startStatPoll(): void {
      stopStatPoll();
      statPollTimer = ctx.timers.setInterval(() => {
        void checkExternalChange();
      }, STAT_POLL_INTERVAL_MS);
    }

    function stopStatPoll(): void {
      if (statPollTimer) {
        statPollTimer.dispose();
        statPollTimer = null;
      }
    }

    async function checkExternalChange(): Promise<void> {
      if (!state.openFile || state.saving) return;
      try {
        const stat = await ctx.agent.fs.stat(state.openFile);
        if (stat.modifiedAt && stat.modifiedAt !== state.lastModifiedAt && state.lastModifiedAt > 0) {
          state.lastModifiedAt = stat.modifiedAt;
          if (!state.dirty) {
            void reloadOpenFile();
          } else {
            setStatus("Changed on disk!");
          }
        }
      } catch {
        // File may have been deleted; ignore.
      }
    }

    /* --- Event wiring --- */

    lsBtn.addEventListener("click", () => {
      const path = pathInput.value.trim() || ".";
      void listDir(path);
    });

    pathInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        const path = pathInput.value.trim() || ".";
        void listDir(path);
      }
    });

    upBtn.addEventListener("click", () => {
      navigateDir(parentDir(state.cwd));
    });

    saveBtn.addEventListener("click", () => {
      void saveFile();
    });

    autoSaveCb.addEventListener("change", () => {
      state.autoSave = autoSaveCb.checked;
      ctx.storage.set(STORAGE_AUTO_SAVE, state.autoSave).catch(() => {});
      if (state.autoSave && state.dirty) {
        scheduleAutoSave();
      }
    });

    editorEl.addEventListener("input", () => {
      if (!state.openFile) return;
      state.dirty = true;
      updateSaveBtn();
      setStatus("Modified");
      scheduleAutoSave();

      // Publish live content to other windows instantly via WebSocket.
      if (!suppressPubsub) {
        pubsub.publish(editorTopic(state.openFile), { text: editorEl.value });
      }
    });

    // Ctrl+S / Cmd+S to save.
    editorEl.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        if (state.openFile && state.dirty) {
          void saveFile();
        }
      }
    });

    // Route changes: open file from URL (Back/Forward, deep links, other tabs).
    ctx.subscriptions.add(onRouteChange((route) => {
      if (route.mode !== "editor") return;
      if (ctx.agent.cwd === null) {
        showNoWorkdirState();
        return;
      }
      if (route.sub) {
        const path = decodeFilePath(route.sub);
        if (path && path !== state.openFile) {
          void listDir(parentDir(path)).then(() => openFile(path));
        }
      }
    }));

    /* --- Restore persisted state --- */

    const savedAutoSave = await ctx.storage.get<boolean>(STORAGE_AUTO_SAVE).catch(() => null);
    if (savedAutoSave === true) {
      state.autoSave = true;
      autoSaveCb.checked = true;
    }

    // Check URL for a file to open (deep link / refresh).
    // Format: #/editor/<base64-filepath>
    const initialRoute = parseHash(window.location.hash);
    const fileFromUrl = initialRoute.mode === "editor" && initialRoute.sub
      ? decodeFilePath(initialRoute.sub)
      : null;

    if (ctx.agent.cwd === null) {
      showNoWorkdirState();
    } else if (fileFromUrl) {
      await listDir(parentDir(fileFromUrl));
      void openFile(fileFromUrl);
    } else {
      await listDir(ctx.agent.cwd);
    }

    ctx.log.info("Editor plugin ready.");
  },
};
