/**
 * Terminal management for hosts in the Router view.
 *
 * Opens a terminal sub-page within the Router when inspecting a host.
 */

import type { AgentShellSession, Disposable, PluginContext } from "../../kernel/types.ts";
import type { TerminalHandle } from "../../lib/terminal.ts";
import { createTerminal } from "../../lib/terminal.ts";
import { repoState } from "./state.ts";
import { getRepoDom } from "./dom.ts";
import { syncRepoTabs } from "./render.ts";

interface ActiveTerminal {
  hostId: string;
  terminal: TerminalHandle;
  shell: AgentShellSession | null;
  shellEventSub: Disposable | null;
  inputSub: Disposable;
  resizeObserver: ResizeObserver;
  tabKeyHandler: (e: KeyboardEvent) => void;
}

let active: ActiveTerminal | null = null;

function cleanupActiveTerminal(): void {
  if (!active) return;
  const a = active;
  active = null;

  const dom = getRepoDom();
  dom.terminalHost.removeEventListener("keydown", a.tabKeyHandler, { capture: true } as EventListenerOptions);
  a.inputSub.dispose();
  a.resizeObserver.disconnect();
  a.terminal.dispose();

  if (a.shell) {
    a.shellEventSub?.dispose();
    const shell = a.shell;
    void shell.close().catch(() => {});
  }

  // Clear the terminal host container.
  dom.terminalHost.innerHTML = "";
}

export function openTerminalForHost(hostId: string, ctx: PluginContext): void {
  const host = repoState.hosts.find((h) => h.id === hostId);
  if (!host) return;

  // Clean up any existing terminal.
  cleanupActiveTerminal();

  repoState.activeTerminalHostId = hostId;
  repoState.subPage = "terminal";

  const dom = getRepoDom();

  // Switch to terminal sub-page.
  dom.routingMain.classList.add("hidden");
  dom.manageLocal.classList.add("hidden");
  dom.terminalPage.classList.remove("hidden");
  dom.terminalTitle.textContent = host.label;

  // Create terminal asynchronously.
  void (async () => {
    const terminal = await createTerminal(dom.terminalHost);
    terminal.focus();

    const sendInput = (data: string): void => {
      if (!active?.shell) return;
      active.shell.send(data);
    };

    const inputSub = terminal.onData((data) => sendInput(data));

    const tabHandler = (event: KeyboardEvent): void => {
      if (event.key !== "Tab") return;
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      event.preventDefault();
      event.stopPropagation();
      sendInput("\t");
    };
    dom.terminalHost.addEventListener("keydown", tabHandler, { capture: true });

    const observer = new ResizeObserver(() => {
      terminal.fit();
      if (!active?.shell) return;
      const cols = terminal.getCols();
      const rows = terminal.getRows();
      if (cols <= 0 || rows <= 0) return;
      active.shell.resize(cols, rows);
    });
    observer.observe(dom.terminalHost);

    active = {
      hostId,
      terminal,
      shell: null,
      shellEventSub: null,
      inputSub,
      resizeObserver: observer,
      tabKeyHandler: tabHandler,
    };

    // Open shell.
    try {
      terminal.fit();
      const cols = terminal.getCols() > 0 ? terminal.getCols() : 120;
      const rows = terminal.getRows() > 0 ? terminal.getRows() : 32;
      const opened = await ctx.agent.shell.open({ cwd: ctx.agent.cwd, cols, rows });

      if (!active || active.hostId !== hostId) {
        // User navigated away — close the shell we just opened.
        void opened.close().catch(() => {});
        return;
      }

      active.shell = opened;
      active.shellEventSub = opened.onEvent((event) => {
        if (event.type === "data") {
          active?.terminal.write(event.data ?? "");
        } else if (event.type === "error") {
          active?.terminal.writeln(`[error] ${event.message ?? "shell error"}`);
        } else if (event.type === "exit") {
          active?.terminal.writeln(`[exit] code=${event.code ?? "unknown"}`);
          if (active) {
            active.shellEventSub?.dispose();
            active.shellEventSub = null;
            active.shell = null;
          }
        }
      });

      terminal.focus();
    } catch (err) {
      terminal.writeln(`[error] ${err instanceof Error ? err.message : String(err)}`);
    }
  })();
}

export function closeTerminal(): void {
  cleanupActiveTerminal();
  repoState.activeTerminalHostId = null;
  repoState.subPage = null;

  const dom = getRepoDom();
  dom.terminalPage.classList.add("hidden");
  dom.routingMain.classList.remove("hidden");
  syncRepoTabs();
}

export function wireTerminalEvents(): void {
  const dom = getRepoDom();
  dom.terminalBackBtn.addEventListener("click", closeTerminal);
}

export function disposeTerminals(): void {
  cleanupActiveTerminal();
}
