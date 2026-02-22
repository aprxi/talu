import { describe, test, expect, beforeEach, spyOn } from "bun:test";
import { readSSEStream, setStreamRenderers, type SSECallbacks } from "../../../src/plugins/chat/streaming.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { initChatDom } from "../../../src/plugins/chat/dom.ts";
import type { RendererRegistry, ContentPart } from "../../../src/kernel/types.ts";
import { createDomRoot, CHAT_DOM_IDS } from "../../helpers/dom.ts";
import { mockTimers } from "../../helpers/mocks.ts";

/**
 * Tests for readSSEStream — the SSE parser that drives the real-time chat UI.
 *
 * Strategy: build a mock Response with a controlled ReadableStream, mock the
 * renderer pipeline and deps, then verify:
 *   - SSE line parsing (event/data extraction)
 *   - Split-chunk handling (buffer across network packets)
 *   - Event type routing (text delta, reasoning, complete, failed)
 *   - State mutations (chatState.lastResponseId, activeSessionId)
 *   - Usage stats calculation
 *   - Error resilience (malformed JSON, missing body)
 */

// -- Helpers -----------------------------------------------------------------

/** Encode text chunks as a ReadableStream (simulating network I/O). */
function sseResponse(...chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
  return new Response(stream);
}

/** Build SSE text for a single event. */
function sseEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

// -- Mocks -------------------------------------------------------------------

let mountedParts: { partId: string; part: ContentPart }[];
let updatedParts: { partId: string; part: ContentPart; isFinal: boolean }[];
let preProcessorCalled: string[];

const mockRenderers: RendererRegistry = {
  register: () => ({ dispose() {} }),
  registerPreProcessor: () => ({ dispose() {} }),
  applyPreProcessors(text: string) {
    preProcessorCalled.push(text);
    return text;
  },
  mountPart(partId, _container, part) {
    mountedParts.push({ partId, part });
  },
  updatePart(partId, part, isFinal) {
    updatedParts.push({ partId, part, isFinal });
  },
  unmountPart() {},
};

let errorMessages: string[];

beforeEach(() => {
  chatState.lastResponseId = null;
  chatState.activeSessionId = null;
  chatState.isGenerating = false;
  chatState.streamAbort = null;

  mountedParts = [];
  updatedParts = [];
  preProcessorCalled = [];
  errorMessages = [];

  setStreamRenderers(mockRenderers);

  initChatDom(createDomRoot(CHAT_DOM_IDS));

  // Initialize deps with synchronous requestAnimationFrame.
  initChatDeps({
    api: {} as any,
    notifications: {
      info: () => {},
      error: (msg: string) => { errorMessages.push(msg); },
      warn: () => {},
    } as any,
    services: { get: () => undefined } as any,
    events: {} as any,
    layout: {} as any,
    clipboard: {} as any,
    download: {} as any,
    upload: {} as any,
    hooks: {
      on: () => ({ dispose() {} }),
      run: async <T>(_name: string, value: T) => value,
    } as any,
    timers: mockTimers(),
    observe: {} as any,
    format: {} as any,
    menus: {
      registerItem: () => ({ dispose() {} }),
      renderSlot: () => ({ dispose() {} }),
    } as any,
  });
});

// -- DOM stubs ---------------------------------------------------------------

function makeDomEls(): { bodyEl: HTMLElement; textEl: HTMLElement } {
  const bodyEl = document.createElement("div");
  const textEl = document.createElement("div");
  bodyEl.appendChild(textEl);
  return { bodyEl, textEl };
}

// ── No body / empty stream ─────────────────────────────────────────────────

describe("readSSEStream — edge cases", () => {
  test("returns null when response has no body", async () => {
    const resp = new Response(null);
    const { bodyEl, textEl } = makeDomEls();
    const result = await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(result).toEqual({ usage: null, sessionId: null });
  });

  test("returns null usage for empty stream (no events)", async () => {
    const resp = sseResponse("");
    const { bodyEl, textEl } = makeDomEls();
    const result = await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(result).toEqual({ usage: null, sessionId: null });
  });
});

// ── Text delta ─────────────────────────────────────────────────────────────

describe("readSSEStream — text delta", () => {
  test("accumulates text deltas and mounts via renderer", async () => {
    const resp = sseResponse(
      sseEvent("response.output_text.delta", { delta: "Hello" }),
      sseEvent("response.output_text.delta", { delta: " World" }),
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);

    // First delta mounts, second updates.
    expect(mountedParts.length).toBe(1);
    expect(mountedParts[0]!.part.type).toBe("text");
    expect(updatedParts.length).toBeGreaterThanOrEqual(1);
  });

  test("text passes through applyPreProcessors", async () => {
    const resp = sseResponse(
      sseEvent("response.output_text.delta", { delta: "processed text" }),
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(preProcessorCalled.length).toBeGreaterThanOrEqual(1);
    expect(preProcessorCalled.some((t) => t.includes("processed text"))).toBe(true);
  });

  test("non-string delta is ignored", async () => {
    const resp = sseResponse(
      sseEvent("response.output_text.delta", { delta: 42 }),
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(mountedParts.length).toBe(0);
  });
});

// ── Reasoning delta ────────────────────────────────────────────────────────

describe("readSSEStream — reasoning delta", () => {
  test("creates reasoning details element on first reasoning delta", async () => {
    const resp = sseResponse(
      sseEvent("response.reasoning.delta", { delta: "Let me think..." }),
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);

    const details = bodyEl.querySelector("details.reasoning-block");
    expect(details).not.toBeNull();
    const summary = details!.querySelector("summary");
    expect(summary!.textContent).toBe("Thought process");
  });

  test("reasoning content is rendered as markdown", async () => {
    const resp = sseResponse(
      sseEvent("response.reasoning.delta", { delta: "**bold reasoning**" }),
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);

    const reasoningBody = bodyEl.querySelector(".reasoning-body");
    expect(reasoningBody).not.toBeNull();
    expect(reasoningBody!.innerHTML).toContain("<strong>");
  });

  test("non-string reasoning delta is ignored", async () => {
    const resp = sseResponse(
      sseEvent("response.reasoning.delta", { delta: null }),
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    const details = bodyEl.querySelector("details.reasoning-block");
    expect(details).toBeNull();
  });
});

// ── Response state mutations ───────────────────────────────────────────────

describe("readSSEStream — state mutations", () => {
  test("response.created sets lastResponseId and activeSessionId", async () => {
    const resp = sseResponse(
      sseEvent("response.created", {
        response: {
          id: "resp-123",
          metadata: { session_id: "sess-456" },
        },
      }),
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(chatState.lastResponseId).toBe("resp-123");
    expect(chatState.activeSessionId).toBe("sess-456");
  });

  test("response.in_progress updates state", async () => {
    const resp = sseResponse(
      sseEvent("response.in_progress", {
        response: { id: "resp-789", metadata: { session_id: "sess-012" } },
      }),
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(chatState.lastResponseId).toBe("resp-789");
    expect(chatState.activeSessionId).toBe("sess-012");
  });

  test("response.completed also updates state", async () => {
    const resp = sseResponse(
      sseEvent("response.completed", {
        response: {
          id: "resp-final",
          metadata: { session_id: "sess-final" },
          usage: { input_tokens: 10, output_tokens: 20, total_tokens: 30 },
        },
      }),
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(chatState.lastResponseId).toBe("resp-final");
    expect(chatState.activeSessionId).toBe("sess-final");
  });

  test("missing metadata does not crash", async () => {
    const resp = sseResponse(
      sseEvent("response.created", { response: { id: "resp-no-meta" } }),
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(chatState.lastResponseId).toBe("resp-no-meta");
    expect(chatState.activeSessionId).toBeNull();
  });
});

// ── Usage stats ────────────────────────────────────────────────────────────

describe("readSSEStream — usage stats", () => {
  test("response.completed returns usage stats with timing", async () => {
    const resp = sseResponse(
      sseEvent("response.completed", {
        response: {
          id: "r1",
          usage: { input_tokens: 100, output_tokens: 50, total_tokens: 150 },
        },
      }),
    );
    const { bodyEl, textEl } = makeDomEls();
    const result = await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(result.usage).not.toBeNull();
    expect(result.usage!.input_tokens).toBe(100);
    expect(result.usage!.output_tokens).toBe(50);
    expect(result.usage!.total_tokens).toBe(150);
    expect(typeof result.usage!.duration_ms).toBe("number");
    expect(result.usage!.duration_ms).toBeGreaterThanOrEqual(0);
    expect(typeof result.usage!.tokens_per_second).toBe("number");
  });

  test("response.completed without usage returns null stats", async () => {
    const resp = sseResponse(
      sseEvent("response.completed", { response: { id: "r2" } }),
    );
    const { bodyEl, textEl } = makeDomEls();
    const result = await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(result.usage).toBeNull();
  });

  test("response.incomplete also returns usage stats", async () => {
    const resp = sseResponse(
      sseEvent("response.incomplete", {
        response: {
          id: "r3",
          usage: { input_tokens: 5, output_tokens: 3, total_tokens: 8 },
        },
      }),
    );
    const { bodyEl, textEl } = makeDomEls();
    const result = await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(result.usage).not.toBeNull();
    expect(result.usage!.output_tokens).toBe(3);
  });
});

// ── Error handling ─────────────────────────────────────────────────────────

describe("readSSEStream — error handling", () => {
  test("response.failed shows error notification", async () => {
    const resp = sseResponse(
      sseEvent("response.failed", {
        response: { error: { message: "Rate limited" } },
      }),
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(errorMessages).toContain("Rate limited");
  });

  test("malformed JSON data is silently skipped", async () => {
    const resp = sseResponse(
      "event: response.output_text.delta\ndata: {invalid json\n\n" +
      sseEvent("response.output_text.delta", { delta: "valid" }),
    );
    const { bodyEl, textEl } = makeDomEls();
    // Should not throw.
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    // Only the valid delta should have been processed.
    expect(mountedParts.length).toBe(1);
  });

  test("unknown event type is silently ignored", async () => {
    const resp = sseResponse(
      sseEvent("response.unknown_event", { foo: "bar" }),
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    // No crash, no side effects.
    expect(mountedParts.length).toBe(0);
    expect(errorMessages.length).toBe(0);
  });
});

// ── SSE chunk splitting ────────────────────────────────────────────────────

describe("readSSEStream — chunk splitting", () => {
  test("event split across two chunks is reassembled", async () => {
    // Split in the middle of the data line.
    const full = sseEvent("response.output_text.delta", { delta: "split-test" });
    const mid = Math.floor(full.length / 2);
    const resp = sseResponse(full.slice(0, mid), full.slice(mid));
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(mountedParts.length).toBe(1);
    expect(preProcessorCalled.some((t) => t.includes("split-test"))).toBe(true);
  });

  test("multiple events in a single chunk are all processed", async () => {
    const combined =
      sseEvent("response.output_text.delta", { delta: "A" }) +
      sseEvent("response.output_text.delta", { delta: "B" });
    const resp = sseResponse(combined);
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    // Both deltas processed — first mounts, second updates.
    expect(mountedParts.length).toBe(1);
    expect(updatedParts.length).toBeGreaterThanOrEqual(1);
  });

  test("chunk boundary in the middle of 'event:' keyword", async () => {
    // Split right after "even" in "event: ..."
    const eventLine = "event: response.output_text.delta\n";
    const dataLine = 'data: {"delta":"boundary"}\n\n';
    const resp = sseResponse(
      eventLine.slice(0, 4),        // "even"
      eventLine.slice(4) + dataLine, // "t: response...\ndata: ...\n\n"
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(mountedParts.length).toBe(1);
    expect(preProcessorCalled.some((t) => t.includes("boundary"))).toBe(true);
  });

  test("trailing data without newline is flushed on stream end", async () => {
    // Missing final \n\n — buffer flush on done should still parse.
    const resp = sseResponse(
      "event: response.output_text.delta\ndata: {\"delta\":\"flush\"}\n",
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);
    expect(mountedParts.length).toBe(1);
    expect(preProcessorCalled.some((t) => t.includes("flush"))).toBe(true);
  });
});

// ── Full conversation flow ─────────────────────────────────────────────────

describe("readSSEStream — full flow", () => {
  test("created → deltas → completed produces stats and rendered text", async () => {
    const resp = sseResponse(
      sseEvent("response.created", {
        response: { id: "r-flow", metadata: { session_id: "s-flow" } },
      }),
      sseEvent("response.output_text.delta", { delta: "Hello" }),
      sseEvent("response.output_text.delta", { delta: " World" }),
      sseEvent("response.completed", {
        response: {
          id: "r-flow",
          usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12 },
        },
      }),
    );
    const { bodyEl, textEl } = makeDomEls();
    const result = await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);

    expect(chatState.lastResponseId).toBe("r-flow");
    expect(chatState.activeSessionId).toBe("s-flow");
    expect(result.usage).not.toBeNull();
    expect(result.usage!.total_tokens).toBe(12);
    // Text was mounted and updated.
    expect(mountedParts.length).toBe(1);
    // Final render triggers an update with isFinal=true.
    const finalUpdate = updatedParts.find((u) => u.isFinal);
    expect(finalUpdate).not.toBeUndefined();
  });

  test("reasoning + text deltas render both blocks", async () => {
    const resp = sseResponse(
      sseEvent("response.reasoning.delta", { delta: "Thinking..." }),
      sseEvent("response.output_text.delta", { delta: "Answer" }),
    );
    const { bodyEl, textEl } = makeDomEls();
    await readSSEStream(resp, bodyEl, textEl, chatState.activeViewId);

    // Reasoning block created.
    expect(bodyEl.querySelector("details.reasoning-block")).not.toBeNull();
    // Text mounted via renderer.
    expect(mountedParts.length).toBe(1);
  });
});
