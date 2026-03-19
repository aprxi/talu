import { beforeEach, describe, expect, test } from "bun:test";
import type { AgentShellSession, PluginContext } from "../../../src/kernel/types.ts";
import { workspaceOpsPlugin } from "../../../src/plugins/workspace-ops/index.ts";
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

function createContext(overrides: Partial<PluginContext> = {}): PluginContext {
  const container = document.createElement("div");
  const baseAgent = {
    cwd: ".",
    shell: {
      open: async () => ({
        id: "shell-default",
        cols: 120,
        rows: 32,
        cwd: ".",
        send(): void {},
        resize(): void {},
        signal(): void {},
        onEvent(): { dispose(): void } {
          return { dispose() {} };
        },
        async close(): Promise<void> {},
      }),
    },
  };

  const overrideAgent = overrides.agent as any;
  const { agent: _agentOverride, ...restOverrides } = overrides as Partial<PluginContext> & {
    agent?: unknown;
  };

  const ctx = {
    container,
    log: {
      info() {},
      warn() {},
      error() {},
    },
    events: {
      on: () => ({ dispose() {} }),
      once: () => ({ dispose() {} }),
      emit: () => {},
    },
    agent: {
      ...baseAgent,
      ...(overrideAgent ?? {}),
      shell: {
        ...baseAgent.shell,
        ...(overrideAgent?.shell ?? {}),
      },
    },
    ...restOverrides,
  };

  return ctx as unknown as PluginContext;
}

beforeEach(() => {
  document.body.innerHTML = "";
});

describe("workspaceOpsPlugin", () => {
  test("registers and renders the current hosts shell UI", async () => {
    expect(workspaceOpsPlugin.manifest.permissions).toEqual(["filesystem", "exec"]);
    expect(workspaceOpsPlugin.manifest.requiresCapabilities).toEqual(["filesystem", "exec"]);

    const ctx = createContext();
    const expectedHostname = window.location.hostname || "localhost";
    workspaceOpsPlugin.register(ctx);
    await workspaceOpsPlugin.run(ctx, new AbortController().signal);

    expect(ctx.container.querySelectorAll("#wop-host-list .sidebar-item")).toHaveLength(1);
    const hostButton = ctx.container.querySelector<HTMLButtonElement>("#wop-host-list .sidebar-item")!;
    const spans = hostButton.querySelectorAll("span");
    expect(ctx.container.querySelector("#wop-shell-state")?.textContent).toBe("connecting");
    expect(ctx.container.querySelector("#wop-terminal-area")).not.toBeNull();
    expect(spans).toHaveLength(3);
    expect(spans[1]!.textContent).toBe(expectedHostname);
    expect(spans[2]!.textContent).toBe("primary");
  });

  test("auto-opens the primary shell and reacts to shell events", async () => {
    const openArgs: Array<{ cwd?: string; cols?: number; rows?: number }> = [];
    let shellEventHandler: ((event: { type: "data" | "exit" | "error"; data?: string; code?: number; message?: string }) => void) | null = null;
    const openRequested = createDeferred<void>();
    const releaseShell = createDeferred<AgentShellSession>();

    const ctx = createContext({
      agent: {
        shell: {
          open: async (opts?: { cwd?: string; cols?: number; rows?: number }) => {
            openArgs.push(opts ?? {});
            openRequested.resolve();
            return releaseShell.promise;
          },
        },
      },
    } as unknown as PluginContext);

    workspaceOpsPlugin.register(ctx);
    const runPromise = workspaceOpsPlugin.run(ctx, new AbortController().signal);
    await openRequested.promise;

    const shellState = ctx.container.querySelector<HTMLElement>("#wop-shell-state")!;
    expect(openArgs[0]!.cwd).toBe(".");
    expect(openArgs[0]!.cols).toBeGreaterThan(0);
    expect(openArgs[0]!.rows).toBeGreaterThan(0);
    expect(shellState.textContent).toBe("connecting");

    releaseShell.resolve({
      id: "shell-1",
      cols: 120,
      rows: 32,
      cwd: ".",
      send(): void {},
      resize(): void {},
      signal(): void {},
      onEvent(handler) {
        shellEventHandler = handler;
        return { dispose() {} };
      },
      async close(): Promise<void> {},
    });

    await flushAsync();
    await runPromise;

    expect(shellState.textContent).toBe("connected (shell-1)");
    shellEventHandler?.({ type: "exit", code: 0 });
    expect(shellState.textContent).toBe("disconnected");
  });

  test("reports shell open failure in the status line", async () => {
    const ctx = createContext({
      agent: {
        shell: {
          open: async () => {
            throw new Error("open_failed: shell unavailable");
          },
        },
      },
    } as unknown as PluginContext);

    workspaceOpsPlugin.register(ctx);
    await workspaceOpsPlugin.run(ctx, new AbortController().signal);
    await flushAsync();

    const shellState = ctx.container.querySelector<HTMLElement>("#wop-shell-state")!;
    const terminalArea = ctx.container.querySelector<HTMLElement>("#wop-terminal-area")!;
    expect(shellState.textContent).toBe("error");
    expect(terminalArea.querySelector(".xterm, pre")).not.toBeNull();
  });
});
