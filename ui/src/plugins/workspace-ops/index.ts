import type {
  AgentShellSession,
  Disposable,
  PluginContext,
  PluginDefinition,
} from "../../kernel/types.ts";

type StatusState = "idle" | "running" | "success" | "error";

interface TerminalHandle {
  write(text: string): void;
  writeln(text: string): void;
  focus(): void;
  fit(): void;
  getCols(): number;
  getRows(): number;
  onData(handler: (data: string) => void): Disposable;
  dispose(): void;
}

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

function createFallbackTerminal(host: HTMLElement): TerminalHandle {
  host.innerHTML = "";
  host.tabIndex = 0;

  const pre = document.createElement("pre");
  pre.style.margin = "0";
  pre.style.padding = "12px";
  pre.style.whiteSpace = "pre-wrap";
  pre.style.wordBreak = "break-word";
  pre.style.minHeight = "220px";
  pre.style.maxHeight = "320px";
  pre.style.overflow = "auto";
  host.appendChild(pre);

  const listeners = new Set<(data: string) => void>();
  const emitData = (data: string): void => {
    for (const listener of listeners) {
      listener(data);
    }
  };

  const keyHandler = (event: KeyboardEvent): void => {
    if (event.key === "Tab") {
      event.preventDefault();
      emitData("\t");
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      emitData("\r");
      return;
    }
    if (event.key === "Backspace") {
      emitData("\u007f");
      return;
    }
    if (event.ctrlKey && (event.key === "c" || event.key === "C")) {
      event.preventDefault();
      emitData("\u0003");
      return;
    }
    if (event.key.length === 1 && !event.metaKey && !event.ctrlKey && !event.altKey) {
      emitData(event.key);
    }
  };

  host.addEventListener("keydown", keyHandler);

  return {
    write(text: string): void {
      pre.textContent = `${pre.textContent ?? ""}${text}`;
      pre.scrollTop = pre.scrollHeight;
    },
    writeln(text: string): void {
      pre.textContent = `${pre.textContent ?? ""}${text}\n`;
      pre.scrollTop = pre.scrollHeight;
    },
    focus(): void {
      host.focus();
    },
    fit(): void {
      // no-op
    },
    getCols(): number {
      return 120;
    },
    getRows(): number {
      return 32;
    },
    onData(handler: (data: string) => void): Disposable {
      listeners.add(handler);
      return {
        dispose(): void {
          listeners.delete(handler);
        },
      };
    },
    dispose(): void {
      host.removeEventListener("keydown", keyHandler);
      listeners.clear();
      host.innerHTML = "";
    },
  };
}

async function createTerminal(host: HTMLElement): Promise<TerminalHandle> {
  try {
    const [{ Terminal }, { FitAddon }] = await Promise.all([
      import("xterm"),
      import("xterm-addon-fit"),
    ]);

    host.innerHTML = "";
    const terminal = new Terminal({
      cursorBlink: true,
      scrollback: 10_000,
      fontSize: 13,
      fontFamily:
        "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace",
    });
    const fitAddon = new FitAddon();
    terminal.loadAddon(fitAddon);
    terminal.open(host);
    fitAddon.fit();

    return {
      write(text: string): void {
        terminal.write(text);
      },
      writeln(text: string): void {
        terminal.writeln(text);
      },
      focus(): void {
        terminal.focus();
      },
      fit(): void {
        fitAddon.fit();
      },
      getCols(): number {
        return terminal.cols;
      },
      getRows(): number {
        return terminal.rows;
      },
      onData(handler: (data: string) => void): Disposable {
        const sub = terminal.onData(handler);
        return {
          dispose(): void {
            sub.dispose();
          },
        };
      },
      dispose(): void {
        terminal.dispose();
      },
    };
  } catch (error) {
    console.warn("[workspace-ops] xterm unavailable, using fallback terminal", error);
    return createFallbackTerminal(host);
  }
}

function buildWorkspaceOpsDom(root: HTMLElement): void {
  root.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:12px;padding:12px;height:100%;overflow:auto;">
      <div class="panel-section" style="display:flex;flex-direction:column;gap:8px;">
        <div class="panel-heading">Terminal</div>
        <div id="wop-shell-state" style="font-size:12px;color:var(--text-muted);">disconnected</div>
        <div id="wop-terminal-host" style="border:1px solid var(--border);border-radius:8px;background:var(--bg-code);min-height:360px;max-height:65vh;overflow:auto;"></div>
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
    const terminalHost = requiredElement<HTMLElement>(ctx.container, "wop-terminal-host");
    const statusEl = requiredElement<HTMLElement>(ctx.container, "wop-status");
    const outputEl = requiredElement<HTMLElement>(ctx.container, "wop-output");

    const terminal = await createTerminal(terminalHost);
    terminal.focus();

    let shell: AgentShellSession | null = null;
    let shellEventSub: Disposable | null = null;
    let openInFlight:
      | Promise<{ shellId: string; cols: number; rows: number; cwd: string | null } | { shellId: string; status: string }>
      | null = null;

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
      if (openInFlight) {
        return openInFlight;
      }

      openInFlight = (async () => {
        terminal.fit();
        const cols = terminal.getCols() > 0 ? terminal.getCols() : 120;
        const rows = terminal.getRows() > 0 ? terminal.getRows() : 32;
        const opened = await ctx.agent.shell.open({ cwd: ctx.agent.cwd, cols, rows });
        shell = opened;
        shellState.textContent = `connected (${opened.id})`;
        shellEventSub = opened.onEvent((event) => {
          if (event.type === "data") {
            terminal.write(event.data ?? "");
          } else if (event.type === "error") {
            terminal.writeln(`[error] ${event.message ?? "shell error"}`);
          } else if (event.type === "exit") {
            terminal.writeln(`[exit] code=${event.code ?? "unknown"}`);
            resetShell();
          }
        });
        terminal.focus();
        return { shellId: opened.id, cols: opened.cols, rows: opened.rows, cwd: opened.cwd };
      })();

      try {
        return await openInFlight;
      } finally {
        openInFlight = null;
      }
    };

    const sendShellInput = (data: string): void => {
      if (!shell) return;
      try {
        shell.send(data);
      } catch (err) {
        setStatus(statusEl, "error", "shell.input failed");
        renderError(outputEl, "shell.input", err);
      }
    };

    const terminalInputSub = terminal.onData((data) => {
      sendShellInput(data);
    });

    const tabKeyHandler = (event: KeyboardEvent): void => {
      if (event.key !== "Tab") return;
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      event.preventDefault();
      event.stopPropagation();
      sendShellInput("\t");
    };
    terminalHost.addEventListener("keydown", tabKeyHandler, { capture: true });

    const resizeObserver = new ResizeObserver(() => {
      terminal.fit();
      if (!shell) return;
      const cols = terminal.getCols();
      const rows = terminal.getRows();
      if (cols <= 0 || rows <= 0) return;
      try {
        shell.resize(cols, rows);
      } catch (err) {
        setStatus(statusEl, "error", "shell.resize failed");
        renderError(outputEl, "shell.resize", err);
      }
    });
    resizeObserver.observe(terminalHost);

    signal.addEventListener("abort", () => {
      terminalHost.removeEventListener("keydown", tabKeyHandler, { capture: true } as EventListenerOptions);
      terminalInputSub.dispose();
      resizeObserver.disconnect();
      terminal.dispose();
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
