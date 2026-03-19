import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import type { AgentShellSession, Disposable, PluginContext } from "../../../src/kernel/types.ts";
import type { TerminalHandle } from "../../../src/lib/terminal.ts";
import * as terminalModule from "../../../src/lib/terminal.ts";
import * as routerModule from "../../../src/kernel/system/router.ts";
import { openTerminalForHost, closeTerminal, disposeTerminals, wireTerminalEvents } from "../../../src/plugins/repo/hosts-terminal.ts";
import { initRepoDom, getRepoDom } from "../../../src/plugins/repo/dom.ts";
import { repoState } from "../../../src/plugins/repo/state.ts";
import { createDomRoot, REPO_DOM_EXTRAS, REPO_DOM_IDS, REPO_DOM_TAGS } from "../../helpers/dom.ts";
import { flushAsync } from "../../helpers/mocks.ts";

function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

class FakeResizeObserver {
  static instances: FakeResizeObserver[] = [];

  callback: ResizeObserverCallback;
  observed: Element[] = [];
  disconnectCalls = 0;

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
    FakeResizeObserver.instances.push(this);
  }

  observe(target: Element): void {
    this.observed.push(target);
  }

  disconnect(): void {
    this.disconnectCalls++;
  }

  trigger(entries: ResizeObserverEntry[] = []): void {
    this.callback(entries, this as unknown as ResizeObserver);
  }

  static reset(): void {
    FakeResizeObserver.instances.length = 0;
  }
}

function createFakeTerminal(cols = 120, rows = 32) {
  let dataHandler: ((data: string) => void) | null = null;
  const state = {
    writes: [] as string[],
    writelns: [] as string[],
    fitCalls: 0,
    focusCalls: 0,
    disposeCalls: 0,
    inputDisposeCalls: 0,
  };
  const terminal: TerminalHandle = {
    write(text: string): void {
      state.writes.push(text);
    },
    writeln(text: string): void {
      state.writelns.push(text);
    },
    focus(): void {
      state.focusCalls++;
    },
    fit(): void {
      state.fitCalls++;
    },
    getCols(): number {
      return cols;
    },
    getRows(): number {
      return rows;
    },
    onData(handler: (data: string) => void): Disposable {
      dataHandler = handler;
      return {
        dispose(): void {
          state.inputDisposeCalls++;
          dataHandler = null;
        },
      };
    },
    dispose(): void {
      state.disposeCalls++;
      dataHandler = null;
    },
  };
  return {
    terminal,
    state,
    emitInput(data: string): void {
      dataHandler?.(data);
    },
  };
}

function createFakeShell(id = "shell-1") {
  let eventHandler: ((event: { type: "data" | "error" | "exit"; data?: string; message?: string; code?: number }) => void) | null = null;
  const state = {
    sends: [] as string[],
    resizes: [] as Array<{ cols: number; rows: number }>,
    closeCalls: 0,
    eventDisposeCalls: 0,
  };
  const shell: AgentShellSession = {
    id,
    cols: 120,
    rows: 32,
    cwd: ".",
    send(data: string): void {
      state.sends.push(data);
    },
    resize(cols: number, rows: number): void {
      state.resizes.push({ cols, rows });
    },
    signal(): void {},
    onEvent(handler): Disposable {
      eventHandler = handler;
      return {
        dispose(): void {
          state.eventDisposeCalls++;
          eventHandler = null;
        },
      };
    },
    async close(): Promise<void> {
      state.closeCalls++;
    },
  };
  return {
    shell,
    state,
    emit(event: { type: "data" | "error" | "exit"; data?: string; message?: string; code?: number }): void {
      eventHandler?.(event);
    },
  };
}

function createContext(openImpl: (opts?: { cwd?: string; cols?: number; rows?: number }) => Promise<AgentShellSession>): PluginContext {
  return {
    agent: {
      cwd: ".",
      shell: { open: openImpl },
    },
  } as unknown as PluginContext;
}

let originalResizeObserver: typeof ResizeObserver;

beforeEach(() => {
  const root = createDomRoot(REPO_DOM_IDS, REPO_DOM_EXTRAS, REPO_DOM_TAGS);
  root.querySelector<HTMLElement>("#rp-terminal-page")!.classList.add("hidden");
  root.querySelector<HTMLElement>("#rp-manage-local")!.classList.add("hidden");
  document.body.innerHTML = "";
  document.body.appendChild(root);
  initRepoDom(root);

  repoState.hosts = [
    { id: "host-a", label: "alpha.local", primary: true },
    { id: "host-b", label: "beta.local", primary: false },
  ];
  repoState.activeTerminalHostId = null;
  repoState.subPage = null;
  repoState.tab = "providers";
  repoState.manageLocalTab = "local";

  FakeResizeObserver.reset();
  originalResizeObserver = globalThis.ResizeObserver;
  globalThis.ResizeObserver = FakeResizeObserver as unknown as typeof ResizeObserver;
});

afterEach(() => {
  disposeTerminals();
  globalThis.ResizeObserver = originalResizeObserver;
});

describe("repo hosts terminal", () => {
  test("opens terminal, forwards input, resizes shell, and renders shell events", async () => {
    const terminal = createFakeTerminal(90, 28);
    const shell = createFakeShell("shell-main");
    const createTerminalSpy = spyOn(terminalModule, "createTerminal").mockResolvedValue(terminal.terminal);
    const navigateSpy = spyOn(routerModule, "navigate").mockImplementation(() => {});
    const openCalls: Array<{ cwd?: string; cols?: number; rows?: number }> = [];
    const ctx = createContext(async (opts) => {
      openCalls.push(opts ?? {});
      return shell.shell;
    });

    openTerminalForHost("host-a", ctx);
    await flushAsync();

    const dom = getRepoDom();
    expect(createTerminalSpy).toHaveBeenCalledWith(dom.terminalHost);
    expect(repoState.activeTerminalHostId).toBe("host-a");
    expect(repoState.subPage).toBe("terminal");
    expect(dom.routingMain.classList.contains("hidden")).toBe(true);
    expect(dom.manageLocal.classList.contains("hidden")).toBe(true);
    expect(dom.terminalPage.classList.contains("hidden")).toBe(false);
    expect(dom.terminalTitle.textContent).toBe("alpha.local");
    expect(navigateSpy.mock.calls[0]![0]).toEqual({ mode: "routing", sub: "terminal", resource: "host-a" });
    expect(openCalls).toEqual([{ cwd: ".", cols: 90, rows: 28 }]);
    expect(terminal.state.fitCalls).toBeGreaterThanOrEqual(1);
    expect(terminal.state.focusCalls).toBeGreaterThanOrEqual(2);

    terminal.emitInput("ls\n");
    expect(shell.state.sends).toEqual(["ls\n"]);

    let prevented = false;
    let stopped = false;
    const tabEvent = new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true });
    Object.defineProperty(tabEvent, "preventDefault", { value: () => { prevented = true; } });
    Object.defineProperty(tabEvent, "stopPropagation", { value: () => { stopped = true; } });
    dom.terminalHost.dispatchEvent(tabEvent);
    expect(prevented).toBe(true);
    expect(stopped).toBe(true);
    expect(shell.state.sends).toEqual(["ls\n", "\t"]);

    expect(FakeResizeObserver.instances).toHaveLength(1);
    FakeResizeObserver.instances[0]!.trigger();
    expect(terminal.state.fitCalls).toBeGreaterThanOrEqual(2);
    expect(shell.state.resizes).toEqual([{ cols: 90, rows: 28 }]);

    shell.emit({ type: "data", data: "hello" });
    shell.emit({ type: "error", message: "bad" });
    shell.emit({ type: "exit", code: 7 });
    expect(terminal.state.writes).toEqual(["hello"]);
    expect(terminal.state.writelns).toEqual(["[error] bad", "[exit] code=7"]);
    expect(shell.state.eventDisposeCalls).toBe(1);

    FakeResizeObserver.instances[0]!.trigger();
    expect(shell.state.resizes).toEqual([{ cols: 90, rows: 28 }]);

    createTerminalSpy.mockRestore();
    navigateSpy.mockRestore();
  });

  test("opening a different host cleans up the existing terminal and shell", async () => {
    const terminalA = createFakeTerminal();
    const terminalB = createFakeTerminal(100, 40);
    const shellA = createFakeShell("shell-a");
    const shellB = createFakeShell("shell-b");
    const terminals = [terminalA.terminal, terminalB.terminal];
    const shells = [shellA.shell, shellB.shell];
    const createTerminalSpy = spyOn(terminalModule, "createTerminal").mockImplementation(async () => terminals.shift()!);
    const navigateSpy = spyOn(routerModule, "navigate").mockImplementation(() => {});
    const ctx = createContext(async () => shells.shift()!);

    openTerminalForHost("host-a", ctx);
    await flushAsync();
    expect(FakeResizeObserver.instances).toHaveLength(1);

    openTerminalForHost("host-b", ctx);
    await flushAsync();

    expect(terminalA.state.inputDisposeCalls).toBe(1);
    expect(terminalA.state.disposeCalls).toBe(1);
    expect(shellA.state.eventDisposeCalls).toBe(1);
    expect(shellA.state.closeCalls).toBe(1);
    expect(FakeResizeObserver.instances[0]!.disconnectCalls).toBe(1);
    expect(repoState.activeTerminalHostId).toBe("host-b");
    expect(getRepoDom().terminalTitle.textContent).toBe("beta.local");
    expect(navigateSpy).toHaveBeenCalledTimes(2);

    createTerminalSpy.mockRestore();
    navigateSpy.mockRestore();
  });

  test("closing before shell.open resolves closes the late shell and restores the routing view", async () => {
    const terminal = createFakeTerminal();
    const deferredShell = createDeferred<AgentShellSession>();
    const lateShell = createFakeShell("late-shell");
    const createTerminalSpy = spyOn(terminalModule, "createTerminal").mockResolvedValue(terminal.terminal);
    const navigateSpy = spyOn(routerModule, "navigate").mockImplementation(() => {});
    const ctx = createContext(async () => deferredShell.promise);

    openTerminalForHost("host-a", ctx);
    await flushAsync();
    closeTerminal();

    expect(repoState.activeTerminalHostId).toBeNull();
    expect(repoState.subPage).toBeNull();
    expect(getRepoDom().terminalPage.classList.contains("hidden")).toBe(true);
    expect(getRepoDom().routingMain.classList.contains("hidden")).toBe(false);

    deferredShell.resolve(lateShell.shell);
    await flushAsync();

    expect(lateShell.state.closeCalls).toBe(1);
    expect(terminal.state.disposeCalls).toBe(1);
    expect(navigateSpy.mock.calls.at(-1)?.[0]).toEqual({ mode: "routing", sub: null, resource: null });

    createTerminalSpy.mockRestore();
    navigateSpy.mockRestore();
  });

  test("wireTerminalEvents closes the terminal via the back button", async () => {
    const terminal = createFakeTerminal();
    const shell = createFakeShell("shell-back");
    const createTerminalSpy = spyOn(terminalModule, "createTerminal").mockResolvedValue(terminal.terminal);
    const navigateSpy = spyOn(routerModule, "navigate").mockImplementation(() => {});
    const ctx = createContext(async () => shell.shell);

    wireTerminalEvents();
    openTerminalForHost("host-a", ctx);
    await flushAsync();

    getRepoDom().terminalBackBtn.click();

    expect(repoState.activeTerminalHostId).toBeNull();
    expect(repoState.subPage).toBeNull();
    expect(shell.state.closeCalls).toBe(1);
    expect(terminal.state.disposeCalls).toBe(1);
    expect(getRepoDom().terminalPage.classList.contains("hidden")).toBe(true);
    expect(getRepoDom().routingMain.classList.contains("hidden")).toBe(false);
    expect(navigateSpy.mock.calls.at(-1)?.[0]).toEqual({ mode: "routing", sub: null, resource: null });

    createTerminalSpy.mockRestore();
    navigateSpy.mockRestore();
  });
});
