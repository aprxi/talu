import { beforeEach, describe, expect, test } from "bun:test";
import type { PluginContext, Disposable } from "../../../src/kernel/types.ts";
import { editorOpsPlugin } from "../../../src/plugins/editor-ops/index.ts";
import { flushAsync, mockControllableTimers, mockNotifications, mockTimers } from "../../helpers/mocks.ts";

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readyState = FakeWebSocket.CONNECTING;
  sent: string[] = [];
  private listeners = new Map<string, Array<(event: unknown) => void>>();

  constructor(public readonly url: string) {
    FakeWebSocket.instances.push(this);
    queueMicrotask(() => {
      this.readyState = FakeWebSocket.OPEN;
      this.emit("open", { type: "open" });
    });
  }

  addEventListener(type: string, listener: (event: unknown) => void): void {
    const list = this.listeners.get(type) ?? [];
    list.push(listener);
    this.listeners.set(type, list);
  }

  send(data: string): void {
    this.sent.push(data);
    let payload: { type?: string } | null = null;
    try {
      payload = JSON.parse(data) as { type?: string };
    } catch {
      payload = null;
    }
    if (payload?.type === "open") {
      this.emitJson({
        type: "ready",
        snapshot_base64: null,
      });
    }
  }

  close(): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.emit("close", { type: "close" });
  }

  private emit(type: string, event: unknown): void {
    for (const listener of this.listeners.get(type) ?? []) {
      listener(event);
    }
  }

  emitJson(payload: unknown): void {
    this.emit("message", { data: JSON.stringify(payload) });
  }
}

function createContext(agentCwd: string | null, options?: {
  readFileContent?: string;
  fetch?: (url: string, init?: RequestInit) => Promise<Response>;
  timers?: PluginContext["timers"];
}): PluginContext {
  const container = document.createElement("div");
  const { mock: notifications, messages } = mockNotifications();
  const timers = options?.timers ?? mockTimers();
  const disposables: Disposable[] = [];
  const fsCalls = { ls: 0 };

  const ctx = {
    container,
    agent: {
      cwd: agentCwd,
      fs: {
        async readFile() {
          return {
            path: "notes.txt",
            content: options?.readFileContent ?? "",
            encoding: "utf-8",
            size: (options?.readFileContent ?? "").length,
            truncated: false,
          };
        },
        async writeFile() {
          throw new Error("writeFile should not be called");
        },
        async editFile() {
          throw new Error("editFile should not be called");
        },
        async stat() {
          return {
            path: "notes.txt",
            exists: true,
            isFile: true,
            isDir: false,
            isSymlink: false,
            size: 0,
            mode: "0644",
            modifiedAt: 1,
            createdAt: 1,
          };
        },
        async ls() {
          fsCalls.ls += 1;
          return {
            path: ".",
            truncated: false,
            entries: [],
          };
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
    network: {
      async fetch(url: string, init?: RequestInit) {
        if (options?.fetch) return options.fetch(url, init);
        throw new Error("network fetch should not be called");
      },
    },
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
  FakeWebSocket.instances.length = 0;
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

  test("opens a collab websocket for the selected workdir file", async () => {
    window.location.hash = "#/editor/" + encodeURIComponent(btoa(unescape(encodeURIComponent("notes.txt"))));
    const ctx = createContext(".", {
      readFileContent: "hello",
    });

    await editorOpsPlugin.run(ctx);
    await flushAsync();

    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(FakeWebSocket.instances[0]!.url).toContain(
      "/v1/collab/resources/workdir_file/notes.txt/ws",
    );
    expect(JSON.parse(FakeWebSocket.instances[0]!.sent[0]!)).toMatchObject({
      type: "open",
      participant_kind: "human",
      role: "editor",
    });
  });

  test("queues one latest live snapshot while a prior op is unacked", async () => {
    const controlled = mockControllableTimers();
    window.location.hash = "#/editor/" + encodeURIComponent(btoa(unescape(encodeURIComponent("notes.txt"))));
    const ctx = createContext(".", {
      readFileContent: "hello",
      timers: controlled.timers,
    });

    await editorOpsPlugin.run(ctx);
    await flushAsync();

    const socket = FakeWebSocket.instances[0]!;
    const editor = ctx.container.querySelector<HTMLTextAreaElement>("#eop-editor")!;
    expect(socket.sent).toHaveLength(1);

    editor.value = "one";
    editor.dispatchEvent(new Event("input"));
    controlled.advance(500);

    editor.value = "two";
    editor.dispatchEvent(new Event("input"));
    controlled.advance(500);

    expect(socket.sent).toHaveLength(2);

    socket.emitJson({
      type: "ack",
      op_id: JSON.parse(socket.sent[1]!).op_id,
    });
    await flushAsync();

    expect(socket.sent).toHaveLength(3);
    const secondSubmit = JSON.parse(socket.sent[2]!);
    expect(secondSubmit.type).toBe("submit_op");
    expect(secondSubmit.op_id).not.toBe(JSON.parse(socket.sent[1]!).op_id);
    expect(secondSubmit.snapshot_base64).toBe(btoa("two"));
  });

  test("preserves textarea selection when applying a remote snapshot", async () => {
    window.location.hash = "#/editor/" + encodeURIComponent(btoa(unescape(encodeURIComponent("notes.txt"))));
    const ctx = createContext(".", {
      readFileContent: "hello world",
    });

    await editorOpsPlugin.run(ctx);
    await flushAsync();

    const socket = FakeWebSocket.instances[0]!;
    const editor = ctx.container.querySelector<HTMLTextAreaElement>("#eop-editor")!;
    editor.focus();
    editor.setSelectionRange(6, 11);

    socket.emitJson({
      type: "snapshot",
      snapshot_base64: btoa("hello brave world"),
    });
    await flushAsync();

    expect(editor.value).toBe("hello brave world");
    expect(editor.selectionStart).toBe(12);
    expect(editor.selectionEnd).toBe(17);
  });
});
