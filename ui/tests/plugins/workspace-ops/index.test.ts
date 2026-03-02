import { beforeEach, describe, expect, test } from "bun:test";
import type { PluginContext } from "../../../src/kernel/types.ts";
import { workspaceOpsPlugin } from "../../../src/plugins/workspace-ops/index.ts";
import { flushAsync } from "../../helpers/mocks.ts";

function createContext(overrides: Partial<PluginContext> = {}): PluginContext {
  const container = document.createElement("div");
  const baseAgent = {
    cwd: ".",
    fs: {
      readFile: async () => ({ path: "notes.txt", content: "hello", encoding: "utf-8", size: 5, truncated: false }),
      writeFile: async () => ({ path: "notes.txt", bytesWritten: 5 }),
      editFile: async () => ({ path: "notes.txt", replacements: 1 }),
      stat: async () => ({
        path: "notes.txt",
        exists: true,
        isFile: true,
        isDir: false,
        isSymlink: false,
        size: 5,
        mode: "644",
        modifiedAt: 0,
        createdAt: 0,
      }),
      ls: async () => ({
        path: ".",
        entries: [{ name: "a.txt", path: "a.txt", isDir: false, isSymlink: false, size: 1, modifiedAt: 0 }],
        truncated: false,
      }),
      rm: async () => {},
      mkdir: async () => {},
      rename: async () => {},
    },
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
      fs: {
        ...baseAgent.fs,
        ...(overrideAgent?.fs ?? {}),
      },
      shell: {
        ...baseAgent.shell,
        ...(overrideAgent?.shell ?? {}),
      },
    },
    ...overrides,
  };

  return ctx as unknown as PluginContext;
}

beforeEach(() => {
  document.body.innerHTML = "";
});

describe("workspaceOpsPlugin", () => {
  test("registers and activates with expected manifest capabilities", async () => {
    expect(workspaceOpsPlugin.manifest.permissions).toEqual(["filesystem", "exec"]);
    expect(workspaceOpsPlugin.manifest.requiresCapabilities).toEqual(["filesystem", "exec"]);

    const ctx = createContext();
    workspaceOpsPlugin.register(ctx);
    await workspaceOpsPlugin.run(ctx, new AbortController().signal);

    expect(ctx.container.querySelector("#wop-path")).not.toBeNull();
    expect(ctx.container.querySelector("#wop-ls-btn")).not.toBeNull();
    expect(ctx.container.querySelector("#wop-read-btn")).not.toBeNull();
    expect(ctx.container.querySelector("#wop-write-btn")).not.toBeNull();
    expect(ctx.container.querySelector("#wop-edit-btn")).not.toBeNull();
    expect(ctx.container.querySelector("#wop-terminal-send-btn")).not.toBeNull();
    const text = ctx.container.textContent ?? "";
    expect(text).toContain("Terminal");
    expect(text.indexOf("Terminal")).toBeLessThan(text.indexOf("Write / Edit"));
  });

  test("fs.ls success path renders output", async () => {
    const ctx = createContext({
      agent: {
        fs: {
          ls: async () => ({
            path: ".",
            entries: [{ name: "src", path: "src", isDir: true, isSymlink: false, size: 0, modifiedAt: 1 }],
            truncated: false,
          }),
        },
      },
    } as unknown as PluginContext);

    await workspaceOpsPlugin.run(ctx, new AbortController().signal);

    const pathInput = ctx.container.querySelector<HTMLInputElement>("#wop-path")!;
    const lsBtn = ctx.container.querySelector<HTMLButtonElement>("#wop-ls-btn")!;
    const output = ctx.container.querySelector<HTMLElement>("#wop-output")!;
    const status = ctx.container.querySelector<HTMLElement>("#wop-status")!;

    pathInput.value = ".";
    lsBtn.click();
    await flushAsync();

    expect(status.dataset["state"]).toBe("success");
    expect(output.textContent).toContain("\"operation\": \"fs.ls\"");
    expect(output.textContent).toContain("\"name\": \"src\"");
  });

  test("fs.readFile error path renders structured code and message", async () => {
    const ctx = createContext({
      agent: {
        fs: {
          readFile: async () => {
            throw new Error("policy_denied_file_read: file read blocked");
          },
        },
      },
    } as unknown as PluginContext);

    await workspaceOpsPlugin.run(ctx, new AbortController().signal);

    const pathInput = ctx.container.querySelector<HTMLInputElement>("#wop-path")!;
    const readBtn = ctx.container.querySelector<HTMLButtonElement>("#wop-read-btn")!;
    const output = ctx.container.querySelector<HTMLElement>("#wop-output")!;

    pathInput.value = "notes.txt";
    readBtn.click();
    await flushAsync();

    const payload = JSON.parse(output.textContent ?? "{}");
    expect(payload.ok).toBe(false);
    expect(payload.error.code).toBe("policy_denied_file_read");
    expect(payload.error.message).toBe("file read blocked");
  });

  test("fs.writeFile on directory path returns invalid_request before network call", async () => {
    const writeSpy = { called: 0 };
    const ctx = createContext({
      agent: {
        fs: {
          writeFile: async () => {
            writeSpy.called += 1;
            return { path: "x", bytesWritten: 1 };
          },
        },
      },
    } as unknown as PluginContext);

    await workspaceOpsPlugin.run(ctx, new AbortController().signal);

    const pathInput = ctx.container.querySelector<HTMLInputElement>("#wop-path")!;
    const writeBtn = ctx.container.querySelector<HTMLButtonElement>("#wop-write-btn")!;
    const output = ctx.container.querySelector<HTMLElement>("#wop-output")!;

    pathInput.value = ".";
    writeBtn.click();
    await flushAsync();

    const payload = JSON.parse(output.textContent ?? "{}");
    expect(writeSpy.called).toBe(0);
    expect(payload.ok).toBe(false);
    expect(payload.error.code).toBe("invalid_request");
  });

  test("terminal auto-opens and send streams output", async () => {
    const sent: string[] = [];
    let onEventHandler: ((event: { type: "data" | "exit" | "error"; data?: string; code?: number; message?: string }) => void) | null = null;

    const ctx = createContext({
      agent: {
        shell: {
          open: async () => ({
            id: "shell-1",
            cols: 120,
            rows: 32,
            cwd: ".",
            send(data: string): void {
              sent.push(data);
            },
            resize(): void {},
            signal(): void {},
            onEvent(handler: (event: { type: "data" | "exit" | "error"; data?: string; code?: number; message?: string }) => void) {
              onEventHandler = handler;
              return { dispose() {} };
            },
            async close(): Promise<void> {},
          }),
        },
      },
    } as unknown as PluginContext);

    await workspaceOpsPlugin.run(ctx, new AbortController().signal);
    await flushAsync();

    const input = ctx.container.querySelector<HTMLInputElement>("#wop-terminal-input")!;
    const sendBtn = ctx.container.querySelector<HTMLButtonElement>("#wop-terminal-send-btn")!;
    const shellState = ctx.container.querySelector<HTMLElement>("#wop-shell-state")!;
    const terminalOutput = ctx.container.querySelector<HTMLElement>("#wop-terminal-output")!;
    const output = ctx.container.querySelector<HTMLElement>("#wop-output")!;
    const status = ctx.container.querySelector<HTMLElement>("#wop-status")!;

    expect(shellState.textContent).toContain("connected");
    expect(status.dataset["state"]).toBe("success");
    expect(JSON.parse(output.textContent ?? "{}").operation).toBe("shell.open");

    input.value = "ls";
    sendBtn.click();
    await flushAsync();

    expect(sent).toEqual(["ls\n"]);
    expect(status.dataset["state"]).toBe("success");
    expect(JSON.parse(output.textContent ?? "{}").operation).toBe("shell.send");
    expect(terminalOutput.textContent).toBe("");

    onEventHandler?.({ type: "data", data: "file-a\n" });
    expect(terminalOutput.textContent).toContain("file-a");
  });

  test("terminal send reports open failure when shell cannot be created", async () => {
    const ctx = createContext({
      agent: {
        shell: {
          open: async () => {
            throw new Error("open_failed: shell unavailable");
          },
        },
      },
    } as unknown as PluginContext);
    await workspaceOpsPlugin.run(ctx, new AbortController().signal);
    await flushAsync();

    const input = ctx.container.querySelector<HTMLInputElement>("#wop-terminal-input")!;
    const sendBtn = ctx.container.querySelector<HTMLButtonElement>("#wop-terminal-send-btn")!;
    const output = ctx.container.querySelector<HTMLElement>("#wop-output")!;
    const status = ctx.container.querySelector<HTMLElement>("#wop-status")!;

    input.value = "pwd";
    sendBtn.click();
    await flushAsync();

    const payload = JSON.parse(output.textContent ?? "{}");
    expect(status.dataset["state"]).toBe("error");
    expect(payload.ok).toBe(false);
    expect(payload.error.code).toBe("open_failed");
  });
});
