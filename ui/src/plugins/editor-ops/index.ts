/**
 * Editor Ops plugin — experimental Write/Edit and Files tools.
 *
 * These tools provide direct filesystem operations (write, edit, ls, read)
 * through the agent interface. This is an early experiment toward a future
 * integrated editor experience.
 */

import type {
  PluginContext,
  PluginDefinition,
} from "../../kernel/types.ts";

function formatJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function parseErrorDetails(err: unknown): { code: string | null; message: string } {
  const raw = err instanceof Error ? err.message : String(err);

  const wrappedCodeMatch = raw.match(/^Agent\s+[a-z.]+\s+failed:\s*([a-zA-Z0-9_]+):\s*(.+)$/);
  if (wrappedCodeMatch) {
    return { code: wrappedCodeMatch[1] ?? null, message: wrappedCodeMatch[2] ?? raw };
  }

  const bracketMatch = raw.match(/^\[([a-zA-Z0-9_]+)\]\s*(.+)$/);
  if (bracketMatch) {
    return { code: bracketMatch[1] ?? null, message: bracketMatch[2] ?? raw };
  }

  const colonMatch = raw.match(/([a-z][a-z0-9_]+):\s*(.+)$/i);
  if (colonMatch) {
    return { code: colonMatch[1] ?? null, message: colonMatch[2] ?? raw };
  }

  return { code: null, message: raw };
}

type StatusState = "idle" | "running" | "success" | "error";

function setStatus(statusEl: HTMLElement, state: StatusState, message: string): void {
  statusEl.dataset["state"] = state;
  statusEl.textContent = message;
}

function renderError(outputEl: HTMLElement, operation: string, err: unknown): void {
  const details = parseErrorDetails(err);
  outputEl.textContent = formatJson({
    operation,
    ok: false,
    error: {
      code: details.code,
      message: details.message,
    },
  });
}

function requireFilePath(pathInput: HTMLInputElement): string {
  const path = pathInput.value.trim();
  if (path.length === 0 || path === "." || path === "/") {
    throw new Error("invalid_request: file path is required (example: notes/todo.txt)");
  }
  if (path.endsWith("/")) {
    throw new Error("invalid_request: path must target a file, not a directory");
  }
  return path;
}

function buildEditorOpsDom(root: HTMLElement): void {
  root.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:12px;padding:12px;height:100%;overflow:auto;">
      <div class="panel-section" style="display:flex;flex-direction:column;gap:8px;">
        <div class="panel-heading">Write / Edit</div>
        <textarea id="eop-content" class="form-textarea mono" placeholder="Content for writeFile"></textarea>
        <textarea id="eop-old-text" class="form-textarea mono" placeholder="Old text (for editFile)"></textarea>
        <textarea id="eop-new-text" class="form-textarea mono" placeholder="New text (for editFile)"></textarea>
        <div style="display:flex;gap:8px;">
          <button id="eop-write-btn" class="btn btn-primary">Write</button>
          <button id="eop-edit-btn" class="btn btn-ghost">Edit</button>
        </div>
      </div>

      <div class="panel-section" style="display:flex;flex-direction:column;gap:8px;">
        <div class="panel-heading">Files</div>
        <div style="display:grid;grid-template-columns:1fr auto auto;gap:8px;">
          <input id="eop-path" class="form-input" type="text" value="." placeholder="Path (e.g. ., src/main.ts)">
          <button id="eop-ls-btn" class="btn btn-ghost">List</button>
          <button id="eop-read-btn" class="btn btn-ghost">Read</button>
        </div>
      </div>

      <div class="panel-section" style="display:flex;flex-direction:column;gap:8px;min-height:0;">
        <div id="eop-status" data-state="idle" style="font-size:12px;color:var(--text-muted);">idle</div>
        <pre id="eop-output" style="margin:0;padding:12px;border:1px solid var(--border);border-radius:8px;background:var(--bg-code);color:var(--text);white-space:pre-wrap;word-break:break-word;min-height:180px;max-height:360px;overflow:auto;"></pre>
      </div>
    </div>
  `;
}

function requiredElement<T extends HTMLElement>(root: HTMLElement, id: string): T {
  const el = root.querySelector<T>(`#${id}`);
  if (!el) {
    throw new Error(`editor-ops missing DOM element #${id}`);
  }
  return el;
}

export const editorOpsPlugin: PluginDefinition = {
  manifest: {
    id: "talu.editorops",
    name: "Editor",
    version: "0.1.0",
    builtin: true,
    permissions: ["filesystem"],
    requiresCapabilities: ["filesystem"],
    contributes: {
      mode: { key: "editor", label: "Editor" },
    },
  },

  register(_ctx: PluginContext): void {
    // no-op
  },

  async run(ctx: PluginContext, _signal: AbortSignal): Promise<void> {
    buildEditorOpsDom(ctx.container);

    const pathInput = requiredElement<HTMLInputElement>(ctx.container, "eop-path");
    const lsBtn = requiredElement<HTMLButtonElement>(ctx.container, "eop-ls-btn");
    const readBtn = requiredElement<HTMLButtonElement>(ctx.container, "eop-read-btn");
    const contentInput = requiredElement<HTMLTextAreaElement>(ctx.container, "eop-content");
    const oldTextInput = requiredElement<HTMLTextAreaElement>(ctx.container, "eop-old-text");
    const newTextInput = requiredElement<HTMLTextAreaElement>(ctx.container, "eop-new-text");
    const writeBtn = requiredElement<HTMLButtonElement>(ctx.container, "eop-write-btn");
    const editBtn = requiredElement<HTMLButtonElement>(ctx.container, "eop-edit-btn");
    const statusEl = requiredElement<HTMLElement>(ctx.container, "eop-status");
    const outputEl = requiredElement<HTMLElement>(ctx.container, "eop-output");

    const runAction = async (operation: string, fn: () => Promise<unknown>): Promise<void> => {
      setStatus(statusEl, "running", `running ${operation}...`);
      try {
        const result = await fn();
        setStatus(statusEl, "success", `${operation} succeeded`);
        outputEl.textContent = formatJson({ operation, ok: true, data: result });
      } catch (err) {
        setStatus(statusEl, "error", `${operation} failed`);
        renderError(outputEl, operation, err);
      }
    };

    lsBtn.addEventListener("click", () => {
      void runAction("fs.ls", async () => {
        const path = pathInput.value.trim() || ".";
        return ctx.agent.fs.ls(path, { recursive: false, limit: 200 });
      });
    });

    readBtn.addEventListener("click", () => {
      void runAction("fs.readFile", async () => {
        const path = requireFilePath(pathInput);
        return ctx.agent.fs.readFile(path, { encoding: "utf-8", maxBytes: 262_144 });
      });
    });

    writeBtn.addEventListener("click", () => {
      void runAction("fs.writeFile", async () => {
        const path = requireFilePath(pathInput);
        return ctx.agent.fs.writeFile(path, contentInput.value, { encoding: "utf-8", mkdir: true });
      });
    });

    editBtn.addEventListener("click", () => {
      void runAction("fs.editFile", async () => {
        const path = requireFilePath(pathInput);
        return ctx.agent.fs.editFile(path, oldTextInput.value, newTextInput.value, { replaceAll: false });
      });
    });

    ctx.log.info("Editor ops plugin ready.");
  },
};
