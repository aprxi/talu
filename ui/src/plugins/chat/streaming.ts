import { chatState } from "./state.ts";
import { notifications, timers } from "./deps.ts";
import { sanitizedMarkdown } from "../../render/markdown.ts";
import { isThinkingExpanded } from "../../render/helpers.ts";
import { scrollToBottomIfNear, updateProgressBar, removeProgressBar } from "./messages.ts";
import type { UsageStats } from "../../types.ts";
import type { RendererRegistry, ContentPart } from "../../kernel/types.ts";

export interface SSECallbacks {
  onTextDelta: (text: string) => void;
  onReasoningDelta: (text: string) => void;
  onProgress?: (phase: string, current: number, total: number) => void;
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

  // Renderer pipeline state.
  const partId = `stream-text-${++partIdCounter}`;
  let textMounted = false;

  // Lazily created reasoning <details>
  let reasoningDetails: HTMLDetailsElement | null = null;
  let reasoningBody: HTMLElement | null = null;

  // Batched rendering with requestAnimationFrame for smooth live markdown
  let textRenderPending = false;
  let reasoningRenderPending = false;

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

  function renderText(isFinal: boolean): void {
    // Run pre-processors (e.g., PII redaction) before creating the ContentPart.
    const processed = renderers!.applyPreProcessors(textAccumulated);
    const part: ContentPart = { id: partId, type: "text", text: processed };
    if (!textMounted) {
      renderers!.mountPart(partId, textEl, part);
      textMounted = true;
    } else {
      renderers!.updatePart(partId, part, isFinal);
    }
  }

  let progressVisible = false;

  const callbacks: SSECallbacks = {
    onProgress(phase, current, total) {
      if (!isActive()) return;
      progressVisible = true;
      updateProgressBar(phase, current, total);
    },
    onTextDelta(t) {
      if (isActive() && progressVisible) {
        progressVisible = false;
        removeProgressBar();
      }
      textAccumulated += t;
      if (!textRenderPending) {
        textRenderPending = true;
        timers.requestAnimationFrame(() => {
          renderText(false);
          textRenderPending = false;
          if (isActive()) scrollToBottomIfNear();
        });
      }
    },
    onReasoningDelta(t) {
      if (isActive() && progressVisible) {
        progressVisible = false;
        removeProgressBar();
      }
      ensureReasoningEl();
      reasoningAccumulated += t;
      if (!reasoningRenderPending) {
        reasoningRenderPending = true;
        timers.requestAnimationFrame(() => {
          reasoningBody!.innerHTML = sanitizedMarkdown(reasoningAccumulated);
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

      if (usage) {
        const durationMs = performance.now() - startTime;
        usageStats = {
          ...usage,
          duration_ms: Math.round(durationMs),
          tokens_per_second: durationMs > 0 ? Math.round((usage.output_tokens / (durationMs / 1000)) * 10) / 10 : 0,
        };
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
        handleSSEEvent(currentEvent, data, callbacks, isActive, streamMeta, onSessionDiscovered);
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
        streamMeta.lastResponseId = response["id"] as string;
        if (isActive()) chatState.lastResponseId = response["id"] as string;
      }
      callbacks.onProgress?.("Queued", 0, 1);
      break;
    }
    case "response.created":
    case "response.in_progress": {
      const response = parsed["response"] as Record<string, unknown> | undefined;
      if (response && typeof response["id"] === "string") {
        streamMeta.lastResponseId = response["id"] as string;
        if (isActive()) chatState.lastResponseId = response["id"] as string;
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
      if (event === "response.in_progress") {
        callbacks.onProgress?.("Generating", 0, 1);
      }
      break;
    }
    case "response.progress": {
      const phase = parsed["phase"];
      const current = parsed["current"];
      const total = parsed["total"];
      if (typeof phase === "string" && typeof current === "number" && typeof total === "number") {
        callbacks.onProgress?.(phase, current, total);
      }
      break;
    }
    case "response.output_text.delta": {
      const delta = parsed["delta"];
      if (typeof delta === "string") {
        callbacks.onTextDelta(delta);
      }
      break;
    }
    case "response.reasoning.delta": {
      const delta = parsed["delta"];
      if (typeof delta === "string") {
        callbacks.onReasoningDelta(delta);
      }
      break;
    }
    case "response.completed":
    case "response.incomplete": {
      const response = parsed["response"] as Record<string, unknown> | undefined;
      if (response && typeof response["id"] === "string") {
        streamMeta.lastResponseId = response["id"] as string;
        if (isActive()) chatState.lastResponseId = response["id"] as string;
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
