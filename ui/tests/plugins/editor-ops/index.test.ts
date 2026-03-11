import { beforeEach, describe, expect, test } from "bun:test";
import type { PluginContext, Disposable } from "../../../src/kernel/types.ts";
import { editorOpsPlugin } from "../../../src/plugins/editor-ops/index.ts";
import { mockNotifications, mockTimers } from "../../helpers/mocks.ts";

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readyState = FakeWebSocket.OPEN;

  addEventListener(): void {}
  send(): void {}
  close(): void {
    this.readyState = FakeWebSocket.CLOSED;
  }
}

function createContext(agentCwd: string | null): PluginContext {
  const container = document.createElement("div");
  const { mock: notifications, messages } = mockNotifications();
  const timers = mockTimers();
  const disposables: Disposable[] = [];
  const fsCalls = { ls: 0 };

  const ctx = {
    container,
    agent: {
      cwd: agentCwd,
      fs: {
        async readFile() {
          throw new Error("readFile should not be called");
        },
        async writeFile() {
          throw new Error("writeFile should not be called");
        },
        async editFile() {
          throw new Error("editFile should not be called");
        },
        async stat() {
          throw new Error("stat should not be called");
        },
        async ls() {
          fsCalls.ls += 1;
          throw new Error("ls should not be called");
        },
        async rm() {},
        async mkdir() {},
        async rename() {},
      },
      shell: {} as PluginContext["agent"]["shell"],
      process: {} as PluginContext["agent"]["process"],
    },
    storage: {
      async get() {
        return null;
      },
      async set() {},
      async delete() {},
      async clear() {},
      async keys() {
        return [];
      },
    },
    timers,
    notifications,
    subscriptions: {
      add(disposable: Disposable): void {
        disposables.push(disposable);
      },
    },
    log: {
      info() {},
      warn() {},
      error() {},
    },
  } as unknown as PluginContext & {
    __messages: typeof messages;
    __fsCalls: typeof fsCalls;
    __disposables: Disposable[];
  };

  ctx.__messages = messages;
  ctx.__fsCalls = fsCalls;
  ctx.__disposables = disposables;
  return ctx;
}

beforeEach(() => {
  document.body.innerHTML = "";
  window.location.hash = "";
  (globalThis as { WebSocket: typeof WebSocket }).WebSocket = FakeWebSocket as unknown as typeof WebSocket;
});

describe("editorOpsPlugin", () => {
  test("shows explicit empty state when no workdir was passed", async () => {
    const ctx = createContext(null);

    await editorOpsPlugin.run(ctx);

    const pathInput = ctx.container.querySelector<HTMLInputElement>("#eop-path");
    const editor = ctx.container.querySelector<HTMLTextAreaElement>("#eop-editor");
    const fileList = ctx.container.querySelector<HTMLElement>("#eop-file-list");
    const status = ctx.container.querySelector<HTMLElement>("#eop-status");

    expect(pathInput).not.toBeNull();
    expect(editor).not.toBeNull();
    expect(fileList).not.toBeNull();
    expect(status).not.toBeNull();
    expect(pathInput!.disabled).toBe(true);
    expect(editor!.disabled).toBe(true);
    expect(pathInput!.placeholder).toBe("No workdir was passed");
    expect(editor!.placeholder).toBe("No workdir was passed");
    expect(fileList!.textContent).toContain("No workdir was passed");
    expect(status!.textContent).toBe("No workdir");
    expect(ctx.__fsCalls.ls).toBe(0);
    expect(ctx.__messages).toHaveLength(0);

    for (const disposable of ctx.__disposables) {
      disposable.dispose();
    }
  });
});
