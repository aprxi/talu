import { beforeEach, describe, expect, test } from "bun:test";
import type { ApiClient } from "../../../src/api.ts";
import { createAgentAccess } from "../../../src/kernel/system/agent.ts";

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readonly url: string;
  readyState = FakeWebSocket.OPEN;
  binaryType: BinaryType = "blob";
  sent: Array<string | ArrayBufferLike | Blob | ArrayBufferView> = [];
  private listeners = new Map<string, Array<(event: MessageEvent | Event) => void>>();

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  addEventListener(type: string, listener: (event: MessageEvent | Event) => void): void {
    const arr = this.listeners.get(type) ?? [];
    arr.push(listener);
    this.listeners.set(type, arr);
  }

  send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.emit("close", new Event("close"));
  }

  emit(type: string, event: MessageEvent | Event): void {
    const arr = this.listeners.get(type) ?? [];
    for (const listener of arr) {
      listener(event);
    }
  }
}

function makeApi(overrides: Partial<ApiClient>): ApiClient {
  return {
    ...overrides,
  } as unknown as ApiClient;
}

describe("createAgentAccess", () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    (globalThis as { WebSocket: typeof WebSocket }).WebSocket = FakeWebSocket as unknown as typeof WebSocket;
  });

  test("fs.readFile enforces filesystem permission and maps maxBytes", async () => {
    let fsPermissionChecks = 0;
    let receivedMaxBytes: number | undefined;

    const api = makeApi({
      agentFsRead: async (body) => {
        receivedMaxBytes = body.max_bytes;
        return {
          ok: true,
          data: {
            path: body.path,
            content: "hello",
            encoding: "utf-8",
            size: 5,
            truncated: false,
          },
        };
      },
    });

    const agent = createAgentAccess({
      api,
      requirePermission(name) {
        if (name === "filesystem") fsPermissionChecks += 1;
      },
    });

    const result = await agent.fs.readFile("notes.txt", { maxBytes: 64 });
    expect(result.content).toBe("hello");
    expect(receivedMaxBytes).toBe(64);
    expect(fsPermissionChecks).toBe(1);
  });

  test("shell.exec uses default cwd and parses stdout+exit SSE", async () => {
    let execPermissionChecks = 0;
    let receivedCwd: string | undefined;

    const api = makeApi({
      agentExec: async (body) => {
        receivedCwd = body.cwd;
        return new Response(
          "data: {\"type\":\"stdout\",\"data\":\"ok\"}\n\n" +
          "data: {\"type\":\"exit\",\"code\":0}\n\n",
          {
            status: 200,
            headers: { "Content-Type": "text/event-stream" },
          },
        );
      },
    });

    const agent = createAgentAccess({
      api,
      requirePermission(name) {
        if (name === "exec") execPermissionChecks += 1;
      },
      defaultCwd: null,
    });

    const result = await agent.shell.exec("echo ok");
    expect(result.stdout).toBe("ok");
    expect(result.exitCode).toBe(0);
    expect(receivedCwd).toBeUndefined();
    expect(execPermissionChecks).toBe(1);
  });

  test("shell.exec throws on SSE error event", async () => {
    const api = makeApi({
      agentExec: async () =>
        new Response("data: {\"type\":\"error\",\"message\":\"denied\"}\n\n", {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        }),
    });

    const agent = createAgentAccess({
      api,
      requirePermission() {},
    });

    await expect(agent.shell.exec("echo denied")).rejects.toThrow(/denied/);
  });

  test("shell.open creates websocket session and forwards control messages", async () => {
    let deletedShellId: string | null = null;
    const api = makeApi({
      agentShellCreate: async () => ({
        ok: true,
        data: {
          shell_id: "shell-1",
          cols: 120,
          rows: 40,
          cwd: ".",
          attached_clients: 0,
        },
      }),
      agentShellDelete: async (id) => {
        deletedShellId = id;
        return { ok: true, data: { shell_id: id, terminated: true } };
      },
    });

    const agent = createAgentAccess({
      api,
      requirePermission() {},
    });

    const session = await agent.shell.open();
    expect(session.id).toBe("shell-1");
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(FakeWebSocket.instances[0]!.url).toContain("/v1/agent/shells/shell-1/ws");

    const ws = FakeWebSocket.instances[0]!;
    const events: Array<{ type: string; data?: string }> = [];
    session.onEvent((event) => {
      events.push({ type: event.type, data: event.data });
    });

    ws.emit(
      "message",
      new MessageEvent("message", {
        data: new TextEncoder().encode("pwd\n").buffer,
      }),
    );
    session.send("ls\n");
    session.resize(100, 30);
    session.signal("sigint");

    expect(ws.sent).toHaveLength(3);
    expect(ws.sent[0]).toBeInstanceOf(Uint8Array);
    expect(new TextDecoder().decode(ws.sent[0] as Uint8Array)).toBe("ls\n");
    expect(ws.sent[1]).toBe(JSON.stringify({ type: "resize", cols: 100, rows: 30 }));
    expect(ws.sent[2]).toBe(JSON.stringify({ type: "signal", signal: "sigint" }));
    expect(events.some((e) => e.type === "data" && e.data === "pwd\n")).toBe(true);

    await session.close();
    expect(deletedShellId).toBe("shell-1");
  });
});
