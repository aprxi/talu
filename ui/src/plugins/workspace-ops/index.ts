import type {
  AgentShellSession,
  Disposable,
  PluginContext,
  PluginDefinition,
} from "../../kernel/types.ts";

type StatusState = "idle" | "running" | "success" | "error";

function setStatus(statusEl: HTMLElement, state: StatusState, message: string): void {
  statusEl.dataset["state"] = state;
  statusEl.textContent = message;
}

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

function buildWorkspaceOpsDom(root: HTMLElement): void {
  root.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:12px;padding:12px;height:100%;overflow:auto;">
      <div class="panel-section" style="display:flex;flex-direction:column;gap:8px;">
        <div class="panel-heading">Terminal</div>
        <div id="wop-shell-state" style="font-size:12px;color:var(--text-muted);">disconnected</div>
        <pre id="wop-terminal-output" style="margin:0;padding:12px;border:1px solid var(--border);border-radius:8px;background:var(--bg-code);color:var(--text);white-space:pre-wrap;word-break:break-word;min-height:180px;max-height:280px;overflow:auto;"></pre>
        <div style="display:grid;grid-template-columns:1fr auto;gap:8px;">
          <input id="wop-terminal-input" class="form-input mono" type="text" placeholder="Type a command and press Send">
          <button id="wop-terminal-send-btn" class="btn btn-primary">Send >_</button>
        </div>
      </div>

      <div class="panel-section" style="display:flex;flex-direction:column;gap:8px;">
        <div class="panel-heading">Write / Edit</div>
        <textarea id="wop-content" class="form-textarea mono" placeholder="Content for writeFile"></textarea>
        <textarea id="wop-old-text" class="form-textarea mono" placeholder="Old text (for editFile)"></textarea>
        <textarea id="wop-new-text" class="form-textarea mono" placeholder="New text (for editFile)"></textarea>
        <div style="display:flex;gap:8px;">
          <button id="wop-write-btn" class="btn btn-primary">Write</button>
          <button id="wop-edit-btn" class="btn btn-ghost">Edit</button>
        </div>
      </div>

      <div class="panel-section" style="display:flex;flex-direction:column;gap:8px;">
        <div class="panel-heading">Files</div>
        <div style="display:grid;grid-template-columns:1fr auto auto;gap:8px;">
          <input id="wop-path" class="form-input" type="text" value="." placeholder="Path (e.g. ., src/main.ts)">
          <button id="wop-ls-btn" class="btn btn-ghost">List</button>
          <button id="wop-read-btn" class="btn btn-ghost">Read</button>
        </div>
      </div>

      <div class="panel-section" style="display:flex;flex-direction:column;gap:8px;min-height:0;">
        <div id="wop-status" data-state="idle" style="font-size:12px;color:var(--text-muted);">idle</div>
        <pre id="wop-output" style="margin:0;padding:12px;border:1px solid var(--border);border-radius:8px;background:var(--bg-code);color:var(--text);white-space:pre-wrap;word-break:break-word;min-height:180px;max-height:360px;overflow:auto;"></pre>
      </div>
    </div>
  `;
}

function requiredElement<T extends HTMLElement>(root: HTMLElement, id: string): T {
  const el = root.querySelector<T>(`#${id}`);
  if (!el) {
    throw new Error(`workspace-ops missing DOM element #${id}`);
  }
  return el;
}

export const workspaceOpsPlugin: PluginDefinition = {
  manifest: {
    id: "talu.workspaceops",
    name: "Terminal",
    version: "0.1.0",
    builtin: true,
    permissions: ["filesystem", "exec"],
    requiresCapabilities: ["filesystem", "exec"],
    contributes: {
      mode: { key: "workspace", label: "Terminal" },
    },
  },

  register(_ctx: PluginContext): void {
    // no-op
  },

  async run(ctx: PluginContext, signal: AbortSignal): Promise<void> {
    buildWorkspaceOpsDom(ctx.container);

    const pathInput = requiredElement<HTMLInputElement>(ctx.container, "wop-path");
    const lsBtn = requiredElement<HTMLButtonElement>(ctx.container, "wop-ls-btn");
    const readBtn = requiredElement<HTMLButtonElement>(ctx.container, "wop-read-btn");
    const contentInput = requiredElement<HTMLTextAreaElement>(ctx.container, "wop-content");
    const oldTextInput = requiredElement<HTMLTextAreaElement>(ctx.container, "wop-old-text");
    const newTextInput = requiredElement<HTMLTextAreaElement>(ctx.container, "wop-new-text");
    const writeBtn = requiredElement<HTMLButtonElement>(ctx.container, "wop-write-btn");
    const editBtn = requiredElement<HTMLButtonElement>(ctx.container, "wop-edit-btn");
    const shellState = requiredElement<HTMLElement>(ctx.container, "wop-shell-state");
    const shellOutput = requiredElement<HTMLElement>(ctx.container, "wop-terminal-output");
    const shellInput = requiredElement<HTMLInputElement>(ctx.container, "wop-terminal-input");
    const shellSendBtn = requiredElement<HTMLButtonElement>(ctx.container, "wop-terminal-send-btn");
    const statusEl = requiredElement<HTMLElement>(ctx.container, "wop-status");
    const outputEl = requiredElement<HTMLElement>(ctx.container, "wop-output");
    let shell: AgentShellSession | null = null;
    let shellEventSub: Disposable | null = null;

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

    const appendTerminal = (text: string): void => {
      shellOutput.textContent = `${shellOutput.textContent ?? ""}${text}`;
      shellOutput.scrollTop = shellOutput.scrollHeight;
    };

    const resetShell = (): void => {
      shellEventSub?.dispose();
      shellEventSub = null;
      shell = null;
      shellState.textContent = "disconnected";
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

    const openShell = async (): Promise<{ shellId: string; cols: number; rows: number; cwd: string | null } | { shellId: string; status: string }> => {
      if (shell) {
        return { shellId: shell.id, status: "already_open" };
      }

      shellOutput.textContent = "";
      const opened = await ctx.agent.shell.open({ cwd: ctx.agent.cwd, cols: 120, rows: 32 });
      shell = opened;
      shellState.textContent = `connected (${opened.id})`;
      shellEventSub = opened.onEvent((event) => {
        if (event.type === "data") {
          appendTerminal(event.data ?? "");
        } else if (event.type === "error") {
          appendTerminal(`\n[error] ${event.message ?? "shell error"}\n`);
        } else if (event.type === "exit") {
          appendTerminal(`\n[exit] code=${event.code ?? "unknown"}\n`);
          resetShell();
        }
      });

      return { shellId: opened.id, cols: opened.cols, rows: opened.rows, cwd: opened.cwd };
    };

    shellSendBtn.addEventListener("click", () => {
      void runAction("shell.send", async () => {
        const line = shellInput.value;
        if (line.trim().length === 0) {
          throw new Error("invalid_request: command is required");
        }

        await openShell();
        if (!shell) {
          throw new Error("shell_not_open: open a terminal session first");
        }

        shell.send(`${line}\n`);
        shellInput.value = "";
        return { sent: true };
      });
    });

    shellInput.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      shellSendBtn.click();
    });

    signal.addEventListener("abort", () => {
      if (!shell) return;
      const active = shell;
      resetShell();
      void active.close().catch(() => {});
    });

    if (!signal.aborted) {
      void runAction("shell.open", openShell);
    }

    ctx.log.info("Workspace ops plugin ready.");
  },
};
