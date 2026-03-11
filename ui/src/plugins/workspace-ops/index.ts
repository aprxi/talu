import type {
  AgentShellSession,
  Disposable,
  PluginContext,
  PluginDefinition,
} from "../../kernel/types.ts";

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

interface HostEntry {
  id: string;
  label: string;
  primary: boolean;
  terminal: TerminalHandle | null;
  terminalHost: HTMLElement;
  shell: AgentShellSession | null;
  shellEventSub: Disposable | null;
  terminalInputSub: Disposable | null;
  resizeObserver: ResizeObserver | null;
  tabKeyHandler: ((e: KeyboardEvent) => void) | null;
  status: "disconnected" | "connecting" | "connected" | "error";
  openInFlight: Promise<unknown> | null;
}

// ---------------------------------------------------------------------------
// Terminal creation
// ---------------------------------------------------------------------------

function createFallbackTerminal(host: HTMLElement): TerminalHandle {
  host.innerHTML = "";
  host.tabIndex = 0;

  const pre = document.createElement("pre");
  pre.style.margin = "0";
  pre.style.padding = "12px";
  pre.style.whiteSpace = "pre-wrap";
  pre.style.wordBreak = "break-word";
  pre.style.height = "100%";
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

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

function buildHostsDom(root: HTMLElement): void {
  root.style.display = "flex";
  root.style.height = "100%";
  root.style.overflow = "hidden";

  root.innerHTML = `
    <aside class="sidebar">
      <div class="sidebar-header" style="padding:0.75rem;">
        <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-muted);">Hosts</div>
      </div>
      <div class="sidebar-content scroll-thin">
        <div id="wop-host-list"></div>
      </div>
    </aside>
    <div style="display:flex;flex-direction:column;flex:1;min-width:0;overflow:hidden;">
      <div id="wop-shell-state" style="font-size:11px;color:var(--text-muted);padding:4px 12px;flex-shrink:0;border-bottom:1px solid var(--border);">disconnected</div>
      <div id="wop-terminal-area" style="flex:1;min-height:0;position:relative;"></div>
    </div>
  `;
}

function renderHostList(
  listEl: HTMLElement,
  hosts: HostEntry[],
  selectedId: string,
  onSelect: (id: string) => void,
): void {
  listEl.innerHTML = "";
  for (const host of hosts) {
    const item = document.createElement("button");
    item.className = "sidebar-item";
    item.style.cssText =
      "display:flex;align-items:center;gap:0.5rem;width:100%;padding:0.5rem 0.75rem;" +
      "font-size:13px;font-family:inherit;border:none;border-radius:6px;cursor:pointer;" +
      "text-align:left;transition:background 0.15s,color 0.15s;";

    const isSelected = host.id === selectedId;
    item.style.background = isSelected
      ? "color-mix(in srgb, var(--primary) 12%, transparent)"
      : "transparent";
    item.style.color = isSelected ? "var(--text)" : "var(--text-muted)";

    // Status dot
    const dot = document.createElement("span");
    dot.style.cssText = "width:8px;height:8px;border-radius:50%;flex-shrink:0;";
    if (host.status === "connected") {
      dot.style.background = "var(--success, #22c55e)";
    } else if (host.status === "connecting") {
      dot.style.background = "var(--warning, #eab308)";
    } else if (host.status === "error") {
      dot.style.background = "var(--error, #ef4444)";
    } else {
      dot.style.background = "var(--text-muted)";
      dot.style.opacity = "0.4";
    }

    const label = document.createElement("span");
    label.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    label.textContent = host.label;

    if (host.primary) {
      const badge = document.createElement("span");
      badge.style.cssText =
        "font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.03em;" +
        "padding:1px 4px;border-radius:3px;flex-shrink:0;margin-left:auto;" +
        "background:color-mix(in srgb, var(--primary) 15%, transparent);color:var(--primary);";
      badge.textContent = "primary";
      item.append(dot, label, badge);
    } else {
      item.append(dot, label);
    }

    item.addEventListener("click", () => onSelect(host.id));
    listEl.appendChild(item);
  }
}

function requiredElement<T extends HTMLElement>(root: HTMLElement, id: string): T {
  const el = root.querySelector<T>(`#${id}`);
  if (!el) {
    throw new Error(`workspace-ops missing DOM element #${id}`);
  }
  return el;
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

export const workspaceOpsPlugin: PluginDefinition = {
  manifest: {
    id: "talu.workspaceops",
    name: "Hosts",
    version: "0.1.0",
    builtin: true,
    permissions: ["filesystem", "exec"],
    requiresCapabilities: ["filesystem", "exec"],
    contributes: {
      mode: { key: "hosts", label: "Hosts" },
    },
  },

  register(_ctx: PluginContext): void {
    // no-op
  },

  async run(ctx: PluginContext, signal: AbortSignal): Promise<void> {
    buildHostsDom(ctx.container);

    const hostListEl = requiredElement<HTMLElement>(ctx.container, "wop-host-list");
    const shellStateEl = requiredElement<HTMLElement>(ctx.container, "wop-shell-state");
    const terminalArea = requiredElement<HTMLElement>(ctx.container, "wop-terminal-area");

    // --- Host state ---

    const primaryHostname = window.location.hostname || "localhost";

    const hosts: HostEntry[] = [
      {
        id: "primary",
        label: primaryHostname,
        primary: true,
        terminal: null,
        terminalHost: document.createElement("div"),
        shell: null,
        shellEventSub: null,
        terminalInputSub: null,
        resizeObserver: null,
        tabKeyHandler: null,
        status: "disconnected",
        openInFlight: null,
      },
    ];

    // Set up the terminal host container for the primary host.
    hosts[0]!.terminalHost.style.cssText =
      "position:absolute;inset:0;background:var(--bg-code);overflow:hidden;";
    terminalArea.appendChild(hosts[0]!.terminalHost);

    let selectedHostId = "primary";

    // --- Rendering ---

    const syncUI = (): void => {
      const host = hosts.find((h) => h.id === selectedHostId);
      renderHostList(hostListEl, hosts, selectedHostId, selectHost);
      if (host) {
        shellStateEl.textContent =
          host.status === "connected" && host.shell
            ? `connected (${host.shell.id})`
            : host.status;
      }
    };

    // --- Per-host terminal + shell wiring ---

    const wireTerminal = (host: HostEntry): void => {
      if (!host.terminal) return;

      const sendShellInput = (data: string): void => {
        if (!host.shell) return;
        host.shell.send(data);
      };

      host.terminalInputSub = host.terminal.onData((data) => {
        sendShellInput(data);
      });

      const tabHandler = (event: KeyboardEvent): void => {
        if (event.key !== "Tab") return;
        if (event.altKey || event.ctrlKey || event.metaKey) return;
        event.preventDefault();
        event.stopPropagation();
        sendShellInput("\t");
      };
      host.tabKeyHandler = tabHandler;
      host.terminalHost.addEventListener("keydown", tabHandler, { capture: true });

      const observer = new ResizeObserver(() => {
        host.terminal?.fit();
        if (!host.shell) return;
        const cols = host.terminal?.getCols() ?? 0;
        const rows = host.terminal?.getRows() ?? 0;
        if (cols <= 0 || rows <= 0) return;
        host.shell.resize(cols, rows);
      });
      observer.observe(host.terminalHost);
      host.resizeObserver = observer;
    };

    const openShellForHost = async (host: HostEntry): Promise<void> => {
      if (host.shell || host.openInFlight) return;

      host.status = "connecting";
      syncUI();

      host.openInFlight = (async () => {
        // Create terminal if not yet initialized.
        if (!host.terminal) {
          host.terminal = await createTerminal(host.terminalHost);
          wireTerminal(host);
        }

        host.terminal.fit();
        const cols = host.terminal.getCols() > 0 ? host.terminal.getCols() : 120;
        const rows = host.terminal.getRows() > 0 ? host.terminal.getRows() : 32;
        const opened = await ctx.agent.shell.open({ cwd: ctx.agent.cwd ?? undefined, cols, rows });
        host.shell = opened;
        host.status = "connected";

        host.shellEventSub = opened.onEvent((event) => {
          if (event.type === "data") {
            host.terminal?.write(event.data ?? "");
          } else if (event.type === "error") {
            host.terminal?.writeln(`[error] ${event.message ?? "shell error"}`);
          } else if (event.type === "exit") {
            host.terminal?.writeln(`[exit] code=${event.code ?? "unknown"}`);
            host.shellEventSub?.dispose();
            host.shellEventSub = null;
            host.shell = null;
            host.status = "disconnected";
            syncUI();
          }
        });

        host.terminal.focus();
        syncUI();
      })();

      try {
        await host.openInFlight;
      } catch (err) {
        host.status = "error";
        host.terminal?.writeln(`[error] ${err instanceof Error ? err.message : String(err)}`);
        syncUI();
      } finally {
        host.openInFlight = null;
      }
    };

    // --- Host selection ---

    const selectHost = (hostId: string): void => {
      if (selectedHostId === hostId) return;
      selectedHostId = hostId;

      // Show/hide terminal containers.
      for (const h of hosts) {
        h.terminalHost.style.display = h.id === hostId ? "" : "none";
      }

      const host = hosts.find((h) => h.id === hostId);
      if (host) {
        // Lazily initialize terminal + shell.
        if (!host.terminal) {
          void openShellForHost(host);
        } else {
          host.terminal.fit();
          host.terminal.focus();
        }
      }

      syncUI();
    };

    // --- Cleanup ---

    signal.addEventListener("abort", () => {
      for (const host of hosts) {
        if (host.tabKeyHandler) {
          host.terminalHost.removeEventListener("keydown", host.tabKeyHandler, {
            capture: true,
          } as EventListenerOptions);
        }
        host.terminalInputSub?.dispose();
        host.resizeObserver?.disconnect();
        host.terminal?.dispose();
        if (host.shell) {
          const active = host.shell;
          host.shellEventSub?.dispose();
          host.shell = null;
          void active.close().catch(() => {});
        }
      }
    });

    // --- Boot: initialize primary host ---

    syncUI();

    if (!signal.aborted) {
      void openShellForHost(hosts[0]!);
    }

    ctx.log.info("Hosts plugin ready.");
  },
};
