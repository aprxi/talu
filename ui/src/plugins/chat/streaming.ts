import { chatState } from "./state.ts";
import { notifications, timers } from "./deps.ts";
import { sanitizedMarkdown } from "../../render/markdown.ts";
import { isThinkingExpanded } from "../../render/helpers.ts";
import { scrollToBottomIfNear, removeProgressBar } from "./messages.ts";
import type { UsageStats } from "../../types.ts";
import type { RendererRegistry, ContentPart } from "../../kernel/types.ts";

export interface DeltaMetrics {
  tokens_generated: number;  // cumulative tokens generated so far (from engine)
  elapsed_ms: number;        // ms elapsed since first token (from engine)
}

export interface SSECallbacks {
  onTextDelta: (text: string, metrics: DeltaMetrics) => void;
  onReasoningDelta: (text: string, metrics: DeltaMetrics) => void;
  onComplete?: (usage: UsageStats | null) => void;
}

export interface StreamResult {
  usage: UsageStats | null;
  sessionId: string | null;
}

// --- Renderer pipeline bridge ---
// Set by the chat plugin during run() to route text through the kernel renderer pipeline.

let renderers: RendererRegistry | null = null;

export function setStreamRenderers(r: RendererRegistry): void {
  renderers = r;
}

let partIdCounter = 0;

export async function readSSEStream(
  resp: Response,
  bodyEl: HTMLElement,
  textEl: HTMLElement,
  viewId: number,
  onSessionDiscovered?: (sessionId: string) => void,
  onResponseDiscovered?: (responseId: string) => void,
): Promise<StreamResult> {
  const reader = resp.body?.getReader();
  if (!reader) return { usage: null, sessionId: null };

  const isActive = () => viewId === chatState.activeViewId;
  const streamMeta: StreamMeta = { sessionId: null, lastResponseId: null };

  const startTime = performance.now();
  const decoder = new TextDecoder();
  let buffer = "";
  let textAccumulated = "";
  let reasoningAccumulated = "";
  let currentEvent = "";
  let usageStats: UsageStats | null = null;

  // Real-time tok/s — cumulative division from engine-level metrics.
  let speedEl: HTMLElement | null = null;

  // Renderer pipeline state.
  const partId = `stream-text-${++partIdCounter}`;
  let textMounted = false;

  // Lazily created reasoning <details>
  let reasoningDetails: HTMLDetailsElement | null = null;
  let reasoningBody: HTMLElement | null = null;

  // Batched rendering with requestAnimationFrame for smooth live markdown
  let textRenderPending = false;
  let reasoningRenderPending = false;
  let reasoningRawTail: HTMLSpanElement | null = null;
  let reasoningSlowTimer: ReturnType<typeof setTimeout> | null = null;
  let reasoningLastRenderedLen = 0;
  let progressBarRemoved = false;

  function ensureReasoningEl(): void {
    if (reasoningDetails) return;
    reasoningDetails = document.createElement("details");
    reasoningDetails.className = "reasoning-block";
    if (isThinkingExpanded()) reasoningDetails.open = true;
    const summary = document.createElement("summary");
    summary.className = "reasoning-summary";
    summary.textContent = "Thought process";
    reasoningDetails.appendChild(summary);
    reasoningBody = document.createElement("div");
    reasoningBody.className = "reasoning-body";
    reasoningDetails.appendChild(reasoningBody);
    bodyEl.insertBefore(reasoningDetails, textEl);
  }

  function fullRenderReasoning(): void {
    if (reasoningRawTail) { reasoningRawTail.remove(); reasoningRawTail = null; }
    reasoningBody!.innerHTML = sanitizedMarkdown(reasoningAccumulated);
    reasoningLastRenderedLen = reasoningAccumulated.length;
  }

  function renderText(isFinal: boolean): void {
    // Pre-processors (e.g., PII redaction) are correctness features for the final
    // text. During streaming, text is ephemeral and re-rendered on final — skip
    // the O(N) pre-processor pass on every frame.
    const processed = isFinal ? renderers!.applyPreProcessors(textAccumulated) : textAccumulated;
    const part: ContentPart = { id: partId, type: "text", text: processed };
    if (!textMounted) {
      renderers!.mountPart(partId, textEl, part);
      textMounted = true;
    } else {
      renderers!.updatePart(partId, part, isFinal);
    }
  }

  let lastMetrics: DeltaMetrics = { tokens_generated: 0, elapsed_ms: 0 };
  let lastSpeedUpdate = 0;

  function updateSpeed(m: DeltaMetrics): void {
    if (m.tokens_generated <= 0 || m.elapsed_ms <= 0) return;
    lastMetrics = m;
    const now = performance.now();
    if (now - lastSpeedUpdate < 200) return;  // ~5 updates/sec max
    lastSpeedUpdate = now;
    if (!speedEl) {
      speedEl = document.createElement("div");
      speedEl.className = "stream-speed";
      bodyEl.appendChild(speedEl);
    }
    const tokPerSec = m.tokens_generated / (m.elapsed_ms / 1000);
    speedEl.textContent = `${tokPerSec.toFixed(1)} tok/s`;
  }

  const callbacks: SSECallbacks = {
    onTextDelta(t, metrics) {
      if (!progressBarRemoved) { removeProgressBar(); progressBarRemoved = true; }
      textAccumulated += t;
      updateSpeed(metrics);
      if (!textRenderPending) {
        textRenderPending = true;
        timers.requestAnimationFrame(() => {
          renderText(false);
          textRenderPending = false;
          if (isActive()) scrollToBottomIfNear();
        });
      }
    },
    onReasoningDelta(t, metrics) {
      if (!progressBarRemoved) { removeProgressBar(); progressBarRemoved = true; }
      ensureReasoningEl();
      reasoningAccumulated += t;
      updateSpeed(metrics);
      if (!reasoningRenderPending) {
        reasoningRenderPending = true;
        timers.requestAnimationFrame(() => {
          // Fast path: append only new delta as raw text — O(delta)
          const delta = reasoningAccumulated.slice(reasoningLastRenderedLen);
          if (delta) {
            if (!reasoningRawTail) {
              reasoningRawTail = document.createElement("span");
              reasoningRawTail.className = "streaming-raw";
              reasoningBody!.appendChild(reasoningRawTail);
            }
            reasoningRawTail.textContent += delta;
            reasoningLastRenderedLen = reasoningAccumulated.length;
          }
          // Slow path: throttled full markdown re-render
          if (!reasoningSlowTimer) {
            reasoningSlowTimer = setTimeout(() => {
              reasoningSlowTimer = null;
              fullRenderReasoning();
            }, 150);
          }
          reasoningRenderPending = false;
          if (isActive()) scrollToBottomIfNear();
        });
      }
    },
    onComplete(usage) {
      // Final render — flush any pending text through the pipeline.
      // Runs even when backgrounded so detached DOM elements get the final content.
      if (textAccumulated) {
        renderText(true);
      }
      if (reasoningAccumulated && reasoningBody) {
        if (reasoningSlowTimer) { clearTimeout(reasoningSlowTimer); reasoningSlowTimer = null; }
        fullRenderReasoning();
      }

      if (usage) {
        const durationMs = performance.now() - startTime;
        // Use engine-level metrics for tok/s (decode throughput only).
        // Client-measured durationMs includes prefill + network overhead,
        // which would undercount the actual generation speed.
        const tokPerSec = lastMetrics.elapsed_ms > 0
          ? Math.round((lastMetrics.tokens_generated / (lastMetrics.elapsed_ms / 1000)) * 10) / 10
          : (durationMs > 0 ? Math.round((usage.output_tokens / (durationMs / 1000)) * 10) / 10 : 0);
        usageStats = {
          ...usage,
          duration_ms: Math.round(durationMs),
          tokens_per_second: tokPerSec,
        };
        if (speedEl) {
          speedEl.textContent = `${tokPerSec} tok/s`;
          speedEl.classList.add("stream-speed-final");
        }
      } else if (speedEl) {
        speedEl.remove();
        speedEl = null;
      }
    },
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      // Flush remaining buffer
      buffer += decoder.decode();
    } else {
      buffer += decoder.decode(value, { stream: true });
    }

    // Parse SSE lines
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const data = line.slice(6);
        handleSSEEvent(
          currentEvent,
          data,
          callbacks,
          isActive,
          streamMeta,
          onSessionDiscovered,
          onResponseDiscovered,
        );
      }
      if (line === "") {
        currentEvent = "";
      }
    }

    if (done) break;
  }
  return { usage: usageStats, sessionId: streamMeta.sessionId };
}

interface StreamMeta {
  sessionId: string | null;
  lastResponseId: string | null;
}

function handleSSEEvent(
  event: string,
  data: string,
  callbacks: SSECallbacks,
  isActive: () => boolean,
  streamMeta: StreamMeta,
  onSessionDiscovered?: (sessionId: string) => void,
  onResponseDiscovered?: (responseId: string) => void,
): void {
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(data);
  } catch {
    return;
  }

  switch (event) {
    case "response.queued": {
      const response = parsed["response"] as Record<string, unknown> | undefined;
      if (response && typeof response["id"] === "string") {
        const responseId = response["id"] as string;
        streamMeta.lastResponseId = responseId;
        onResponseDiscovered?.(responseId);
        if (isActive()) chatState.lastResponseId = responseId;
      }
      break;
    }
    case "response.created":
    case "response.in_progress": {
      const response = parsed["response"] as Record<string, unknown> | undefined;
      if (response && typeof response["id"] === "string") {
        const responseId = response["id"] as string;
        streamMeta.lastResponseId = responseId;
        onResponseDiscovered?.(responseId);
        if (isActive()) chatState.lastResponseId = responseId;
      }
      const meta = response?.["metadata"] as Record<string, unknown> | undefined;
      if (meta && typeof meta["session_id"] === "string") {
        const sid = meta["session_id"] as string;
        if (streamMeta.sessionId !== sid) {
          streamMeta.sessionId = sid;
          onSessionDiscovered?.(sid);
        }
        if (isActive()) chatState.activeSessionId = sid;
      }
      // Pick up project_id from response metadata and propagate to the active session.
      if (meta && typeof meta["project_id"] === "string" && isActive()) {
        const projectId = meta["project_id"] as string;
        if (chatState.activeChat) {
          chatState.activeChat.project_id = projectId;
        }
        const session = chatState.sessions.find(s => s.id === chatState.activeSessionId);
        if (session) {
          session.project_id = projectId;
        }
      }
      break;
    }
    case "response.output_text.delta": {
      const delta = parsed["delta"];
      if (typeof delta === "string") {
        const metrics: DeltaMetrics = {
          tokens_generated: typeof parsed["tokens_generated"] === "number" ? parsed["tokens_generated"] as number : 0,
          elapsed_ms: typeof parsed["elapsed_ms"] === "number" ? parsed["elapsed_ms"] as number : 0,
        };
        callbacks.onTextDelta(delta, metrics);
      }
      break;
    }
    case "response.reasoning.delta": {
      const delta = parsed["delta"];
      if (typeof delta === "string") {
        const metrics: DeltaMetrics = {
          tokens_generated: typeof parsed["tokens_generated"] === "number" ? parsed["tokens_generated"] as number : 0,
          elapsed_ms: typeof parsed["elapsed_ms"] === "number" ? parsed["elapsed_ms"] as number : 0,
        };
        callbacks.onReasoningDelta(delta, metrics);
      }
      break;
    }
    case "response.completed":
    case "response.incomplete": {
      const response = parsed["response"] as Record<string, unknown> | undefined;
      if (response && typeof response["id"] === "string") {
        const responseId = response["id"] as string;
        streamMeta.lastResponseId = responseId;
        onResponseDiscovered?.(responseId);
        if (isActive()) chatState.lastResponseId = responseId;
      }
      const meta = response?.["metadata"] as Record<string, unknown> | undefined;
      if (meta && typeof meta["session_id"] === "string") {
        const sid = meta["session_id"] as string;
        if (streamMeta.sessionId !== sid) {
          streamMeta.sessionId = sid;
          onSessionDiscovered?.(sid);
        }
        if (isActive()) chatState.activeSessionId = sid;
      }
      // Pick up project_id from completed response metadata.
      if (meta && typeof meta["project_id"] === "string" && isActive()) {
        const projectId = meta["project_id"] as string;
        if (chatState.activeChat) {
          chatState.activeChat.project_id = projectId;
        }
        const session = chatState.sessions.find(s => s.id === chatState.activeSessionId);
        if (session) {
          session.project_id = projectId;
        }
      }
      const usage = response?.["usage"] as Record<string, unknown> | undefined;
      if (usage && callbacks.onComplete) {
        const stats: UsageStats = {
          input_tokens: typeof usage["input_tokens"] === "number" ? usage["input_tokens"] : 0,
          output_tokens: typeof usage["output_tokens"] === "number" ? usage["output_tokens"] : 0,
          total_tokens: typeof usage["total_tokens"] === "number" ? usage["total_tokens"] : 0,
        };
        callbacks.onComplete(stats);
      } else if (callbacks.onComplete) {
        callbacks.onComplete(null);
      }
      break;
    }
    case "response.failed": {
      if (!isActive()) break;
      const response = parsed["response"] as Record<string, unknown> | undefined;
      const error = response?.["error"] as Record<string, unknown> | undefined;
      if (error && typeof error["message"] === "string") {
        notifications.error(error["message"] as string);
      }
      break;
    }
  }
}
