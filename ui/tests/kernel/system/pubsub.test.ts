import { beforeEach, describe, expect, test } from "bun:test";
import { connectPubSub } from "../../../src/kernel/system/pubsub.ts";

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

describe("connectPubSub", () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    (globalThis as { WebSocket: typeof WebSocket }).WebSocket = FakeWebSocket as unknown as typeof WebSocket;
  });

  test("connects to the collab pubsub websocket", () => {
    const client = connectPubSub();

    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(FakeWebSocket.instances[0]!.url).toContain("/v1/collab/pubsub/ws");

    client.close();
  });
});
