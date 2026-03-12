/**
 * Editor plugin — interactive file browser + text editor.
 *
 * Browse the agent filesystem, open files for editing, save manually or
 * with auto-save.  Open file path is encoded in the URL hash so you can
 * open the same file in multiple tabs.  Live buffer state flows through
 * `/v1/collab/resources/*`, with stat-polling retained only for out-of-band
 * local disk edits that bypass the collab API.
 */

import type {
  AgentLsEntry,
  Disposable,
  PluginContext,
  PluginDefinition,
} from "../../kernel/types.ts";
import { navigate, onRouteChange, parseHash } from "../../kernel/system/router.ts";

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
const LIVE_SYNC_DELAY_MS = 500;
const COLLAB_RESOURCE_KIND = "workdir_file";

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

function encodeUtf8Base64(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function decodeUtf8Base64(base64: string): string {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new TextDecoder().decode(bytes);
}

function commonPrefixLength(left: string, right: string): number {
  const max = Math.min(left.length, right.length);
  let idx = 0;
  while (idx < max && left[idx] === right[idx]) idx += 1;
  return idx;
}

function commonSuffixLength(left: string, right: string, prefixLength: number): number {
  const max = Math.min(left.length, right.length) - prefixLength;
  let idx = 0;
  while (
    idx < max &&
    left[left.length - 1 - idx] === right[right.length - 1 - idx]
  ) {
    idx += 1;
  }
  return idx;
}

function remapSelectionIndex(index: number, oldText: string, newText: string): number {
  const prefixLength = commonPrefixLength(oldText, newText);
  const suffixLength = commonSuffixLength(oldText, newText, prefixLength);
  const oldChangedEnd = oldText.length - suffixLength;
  const newChangedEnd = newText.length - suffixLength;

  if (index < prefixLength) return index;
  if (index >= oldChangedEnd) {
    return newChangedEnd + (index - oldChangedEnd);
  }
  return newChangedEnd;
}

function buildCollabWebSocketUrl(path: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/v1/collab/resources/${encodeURIComponent(COLLAB_RESOURCE_KIND)}/${encodeURIComponent(path)}/ws`;
}

function randomParticipantId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `human:ui:${crypto.randomUUID()}`;
  }
  return `human:ui:${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
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
    let liveSyncTimer: Disposable | null = null;
    let statPollTimer: Disposable | null = null;
    let suppressLiveSync = false;
    let collabSocket: WebSocket | null = null;
    let collabSocketPath: string | null = null;
    const participantId = randomParticipantId();
    let actorSeq = 0;
    let collabEnabled = false;
    let inflightLiveSnapshot: { path: string; opId: string; text: string } | null = null;
    let queuedLiveSnapshot: { path: string; text: string; opType: string } | null = null;

    function applyIncomingSnapshot(path: string, snapshotBase64: string | null, status: string): void {
      if (state.openFile !== path || snapshotBase64 === null) return;
      const snapshot = decodeUtf8Base64(snapshotBase64);
      const previousValue = editorEl.value;
      if (snapshot === previousValue) return;
      const selectionStart = editorEl.selectionStart ?? previousValue.length;
      const selectionEnd = editorEl.selectionEnd ?? previousValue.length;
      const scrollTop = editorEl.scrollTop;
      const remappedSelectionStart = remapSelectionIndex(selectionStart, previousValue, snapshot);
      const remappedSelectionEnd = remapSelectionIndex(selectionEnd, previousValue, snapshot);
      suppressLiveSync = true;
      editorEl.value = snapshot;
      editorEl.setSelectionRange(remappedSelectionStart, remappedSelectionEnd);
      editorEl.scrollTop = scrollTop;
      suppressLiveSync = false;
      state.dirty = false;
      updateSaveBtn();
      setStatus(status);
    }

    function stopCollabSocket(): void {
      if (collabSocket) {
        collabSocket.close();
        collabSocket = null;
      }
      collabSocketPath = null;
      inflightLiveSnapshot = null;
      queuedLiveSnapshot = null;
    }

    function openCollabSocketForPath(path: string): Promise<string | null> {
      stopCollabSocket();
      return new Promise((resolve, reject) => {
        const socket = new WebSocket(buildCollabWebSocketUrl(path));
        let settled = false;
        collabSocket = socket;
        collabSocketPath = path;

        const finishReady = (snapshotBase64: string | null): void => {
          if (settled) return;
          settled = true;
          resolve(snapshotBase64 === null ? null : decodeUtf8Base64(snapshotBase64));
        };

        const failReady = (message: string): void => {
          if (settled) return;
          settled = true;
          reject(new Error(message));
        };

        socket.addEventListener("open", () => {
          socket.send(JSON.stringify({
            type: "open",
            participant_id: participantId,
            participant_kind: "human",
            role: "editor",
          }));
        });

        socket.addEventListener("message", (event) => {
          const message = event as MessageEvent<string>;
          let payload: {
            type?: string;
            snapshot_base64?: string | null;
            message?: string;
            op_id?: string;
            key?: string;
          } | null = null;
          try {
            payload = JSON.parse(message.data) as typeof payload;
          } catch {
            payload = null;
          }
          if (!payload?.type) return;
          if (payload.type === "ready") {
            finishReady(payload.snapshot_base64 ?? null);
            return;
          }
          if (payload.type === "snapshot") {
            if (payload.key && payload.key.startsWith(`ops/${participantId}:`)) {
              return;
            }
            if (state.dirty || inflightLiveSnapshot !== null || queuedLiveSnapshot !== null) {
              return;
            }
            applyIncomingSnapshot(path, payload.snapshot_base64 ?? null, "Live");
            return;
          }
          if (payload.type === "ack") {
            if (inflightLiveSnapshot && (!payload.op_id || payload.op_id === inflightLiveSnapshot.opId)) {
              inflightLiveSnapshot = null;
            }
            if (queuedLiveSnapshot) {
              void flushQueuedLiveSnapshot().catch((err) => {
                setStatus("Live sync failed");
                ctx.notifications.warning(`Live sync failed: ${err instanceof Error ? err.message : String(err)}`);
              });
            } else {
              setStatus("Live");
            }
            return;
          }
          if (payload.type === "error") {
            if (!settled) {
              failReady(payload.message ?? "collab websocket failed");
            } else {
              collabEnabled = false;
              inflightLiveSnapshot = null;
              setStatus("Live error");
              ctx.notifications.warning(`Live sync failed: ${payload.message ?? "unknown error"}`);
            }
          }
        });

        socket.addEventListener("error", () => {
          failReady("failed to connect collab websocket");
        });

        socket.addEventListener("close", () => {
          if (collabSocket === socket) {
            collabSocket = null;
            collabSocketPath = null;
            collabEnabled = false;
            inflightLiveSnapshot = null;
            queuedLiveSnapshot = null;
            if (state.openFile === path) {
              setStatus("Live disconnected");
            }
          }
          failReady("collab websocket closed before ready");
        });
      });
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
      collabEnabled = false;
      stopCollabSocket();
      if (liveSyncTimer) {
        liveSyncTimer.dispose();
        liveSyncTimer = null;
      }
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

    async function flushQueuedLiveSnapshot(): Promise<void> {
      if (!queuedLiveSnapshot) return;
      if (inflightLiveSnapshot) return;
      if (!collabSocket || collabSocketPath !== queuedLiveSnapshot.path || collabSocket.readyState !== WebSocket.OPEN) {
        throw new Error("collab websocket is not connected");
      }
      const next = queuedLiveSnapshot;
      queuedLiveSnapshot = null;
      actorSeq += 1;
      const opId = `${next.opType}-${actorSeq}`;
      const payload = encodeUtf8Base64(JSON.stringify({
        type: next.opType,
        source: "ui.editor",
        path: next.path,
        bytes: new TextEncoder().encode(next.text).length,
      }));
      inflightLiveSnapshot = {
        path: next.path,
        opId,
        text: next.text,
      };
      setStatus("Live sync");
      collabSocket.send(JSON.stringify({
        type: "submit_op",
        actor_seq: actorSeq,
        op_id: opId,
        payload_base64: payload,
        snapshot_base64: encodeUtf8Base64(next.text),
        durability: "batched",
      }));
    }

    function queueLiveSnapshot(path: string, text: string, opType: string): void {
      queuedLiveSnapshot = { path, text, opType };
    }

    function scheduleLiveSync(): void {
      if (liveSyncTimer) {
        liveSyncTimer.dispose();
        liveSyncTimer = null;
      }
      if (!state.openFile || suppressLiveSync) return;
      if (!collabEnabled) return;
      liveSyncTimer = ctx.timers.setTimeout(() => {
        liveSyncTimer = null;
        if (!state.openFile) return;
        queueLiveSnapshot(state.openFile, editorEl.value, "ui_live");
        void flushQueuedLiveSnapshot().catch((err) => {
          setStatus("Live sync failed");
          ctx.notifications.warning(`Live sync failed: ${err instanceof Error ? err.message : String(err)}`);
        });
      }, LIVE_SYNC_DELAY_MS);
    }

    async function openFile(path: string): Promise<void> {
      state.openFile = path;
      navigate({ mode: "editor", sub: encodeFilePath(path), resource: null }, { replace: true });
      stopCollabSocket();
      if (liveSyncTimer) {
        liveSyncTimer.dispose();
        liveSyncTimer = null;
      }
      collabEnabled = false;
      toolbar.style.display = "flex";
      filenameEl.textContent = path;
      filenameEl.title = path;
      setStatus("Loading...");

      try {
        const collabSnapshot = await openCollabSocketForPath(path);
        collabEnabled = true;
        const result = collabSnapshot === null
          ? await ctx.agent.fs.readFile(path, { encoding: "utf-8", maxBytes: 1_048_576 })
          : null;
        state.dirty = false;
        editorEl.value = collabSnapshot ?? result!.content;
        editorEl.disabled = false;
        updateSaveBtn();

        if (result?.truncated) {
          setStatus("Truncated!");
          ctx.notifications.warning(`File truncated at ${formatSize(result.size)}`);
        } else {
          setStatus("Loaded");
        }

        const stat = await ctx.agent.fs.stat(path).catch(() => null);
        state.lastModifiedAt = stat?.modifiedAt ?? 0;
        startStatPoll();
      } catch (err) {
        if (isNoWorkdirError(err)) {
          showNoWorkdirState();
          return;
        }
        collabEnabled = false;
        setStatus("Error");
        ctx.notifications.error(`Open failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    async function reloadOpenFile(syncCollab: boolean): Promise<void> {
      if (!state.openFile) return;
      try {
        const result = await ctx.agent.fs.readFile(state.openFile, { encoding: "utf-8", maxBytes: 1_048_576 });
        editorEl.value = result.content;
        state.dirty = false;
        updateSaveBtn();
        setStatus("Reloaded");

        const stat = await ctx.agent.fs.stat(state.openFile).catch(() => null);
        state.lastModifiedAt = stat?.modifiedAt ?? 0;
        if (syncCollab && collabEnabled) {
          queueLiveSnapshot(state.openFile, result.content, "disk_reload");
          await flushQueuedLiveSnapshot().catch(() => {});
        }
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
            void reloadOpenFile(true);
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
      scheduleLiveSync();
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
    ctx.subscriptions.add({
      dispose() {
        stopCollabSocket();
        stopStatPoll();
        autoSaveTimer?.dispose();
        liveSyncTimer?.dispose();
      },
    });

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
