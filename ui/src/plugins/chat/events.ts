import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { api } from "./deps.ts";
import { updateProgressBar } from "./messages.ts";

const MAX_EVENT_LINES = 160;

interface EventEnvelope {
  ts_ms: number;
  level: string;
  topic: string;
  event_class: string;
  message: string;
  data?: Record<string, unknown> | null;
}

function addPlaceholderIfEmpty(): void {
  const dom = getChatDom();
  if (dom.panelEventsLog.childElementCount > 0) return;
  const empty = document.createElement("div");
  empty.className = "chat-events-empty";
  empty.textContent = "No events yet.";
  dom.panelEventsLog.appendChild(empty);
}

function appendEventsLine(text: string, level: string = "info"): void {
  const dom = getChatDom();
  const empty = dom.panelEventsLog.querySelector(".chat-events-empty");
  if (empty) empty.remove();

  const line = document.createElement("div");
  line.className = `chat-events-line chat-events-level-${level.toLowerCase()}`;
  line.textContent = text;
  dom.panelEventsLog.appendChild(line);

  while (dom.panelEventsLog.childElementCount > MAX_EVENT_LINES) {
    dom.panelEventsLog.firstElementChild?.remove();
  }
  dom.panelEventsLog.scrollTop = dom.panelEventsLog.scrollHeight;
}

function formatTime(tsMs: number): string {
  const dt = new Date(tsMs);
  return dt.toLocaleTimeString([], { hour12: false });
}

function toProgressParts(data: Record<string, unknown> | null | undefined): { phase: string; current: number; total: number } | null {
  if (!data) return null;
  const phase = typeof data["phase"] === "string" ? data["phase"] : "Progress";
  const current = typeof data["current"] === "number" ? data["current"] : null;
  const total = typeof data["total"] === "number" ? data["total"] : null;
  if (current != null && total != null && Number.isFinite(current) && Number.isFinite(total) && total > 0) {
    return { phase, current, total };
  }

  const pct = typeof data["pct"] === "number" ? data["pct"] : null;
  if (pct != null && Number.isFinite(pct) && pct >= 0) {
    return { phase, current: pct, total: 100 };
  }
  return null;
}

function handleEventEnvelope(envelope: EventEnvelope, isActive: () => boolean): void {
  const time = formatTime(typeof envelope.ts_ms === "number" ? envelope.ts_ms : Date.now());
  const level = typeof envelope.level === "string" ? envelope.level : "info";
  const topic = typeof envelope.topic === "string" ? envelope.topic : "events.unknown";
  const message = typeof envelope.message === "string" ? envelope.message : "";

  appendEventsLine(`${time} ${level.toUpperCase()} ${topic} ${message}`, level);

  if (envelope.event_class !== "progress" || topic !== "inference.progress" || !isActive()) return;
  const parts = toProgressParts(envelope.data);
  if (!parts) return;
  updateProgressBar(parts.phase, parts.current, parts.total);
}

function handleNonEnvelopeEvent(eventName: string, data: string): void {
  if (!eventName) return;
  let message = data;
  let level = "warn";
  try {
    const parsed = JSON.parse(data) as Record<string, unknown>;
    if (typeof parsed["message"] === "string") message = parsed["message"];
    if (eventName === "done") level = "info";
  } catch {
    // Keep raw text payload.
  }
  const time = formatTime(Date.now());
  appendEventsLine(`${time} ${eventName.toUpperCase()} ${message}`, level);
}

async function runResponseEventsStream(
  responseId: string,
  verbosity: 1 | 2 | 3,
  viewId: number,
  signal: AbortSignal,
): Promise<void> {
  let resp: Response;
  try {
    resp = await api.streamEvents(
      { verbosity, domains: "inference", response_id: responseId },
      signal,
    );
  } catch (err) {
    if (!signal.aborted) {
      const msg = err instanceof Error ? err.message : String(err);
      appendEventsLine(`events stream error: ${msg}`, "error");
    }
    return;
  }

  if (!resp.ok) {
    appendEventsLine(`events stream failed: ${resp.status} ${resp.statusText}`, "error");
    return;
  }

  const reader = resp.body?.getReader();
  if (!reader) {
    appendEventsLine("events stream unavailable (empty body)", "warn");
    return;
  }

  const isActive = () => viewId === chatState.activeViewId;

  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "";

  while (!signal.aborted) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const raw of lines) {
      const line = raw.trimEnd();
      if (line.startsWith(":")) continue;
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const data = line.slice(6);
        if (currentEvent === "event") {
          try {
            const envelope = JSON.parse(data) as EventEnvelope;
            handleEventEnvelope(envelope, isActive);
          } catch {
            // Ignore malformed event payloads.
          }
        } else {
          handleNonEnvelopeEvent(currentEvent, data);
        }
      } else if (line === "") {
        currentEvent = "";
      }
    }
  }
}

export function setupEventsPanelEvents(): void {
  const dom = getChatDom();
  dom.panelEventsVerbosity.value = String(chatState.eventsVerbosity);
  dom.panelEventsVerbosity.addEventListener("change", () => {
    const v = Number.parseInt(dom.panelEventsVerbosity.value, 10);
    chatState.eventsVerbosity = v === 2 ? 2 : v === 3 ? 3 : 1;
  });
  dom.panelEventsClear.addEventListener("click", clearEventsLog);
  addPlaceholderIfEmpty();
}

export function clearEventsLog(): void {
  const dom = getChatDom();
  dom.panelEventsLog.innerHTML = "";
  addPlaceholderIfEmpty();
}

export function stopResponseEventsStream(): void {
  chatState.eventsAbort?.abort();
  chatState.eventsAbort = null;
  chatState.eventsResponseId = null;
}

export function startResponseEventsStream(responseId: string, viewId: number): void {
  if (!responseId) return;
  if (chatState.eventsAbort && chatState.eventsResponseId === responseId) return;

  stopResponseEventsStream();
  const controller = new AbortController();
  chatState.eventsAbort = controller;
  chatState.eventsResponseId = responseId;
  appendEventsLine(`subscribed to ${responseId}`, "debug");

  void runResponseEventsStream(
    responseId,
    chatState.eventsVerbosity,
    viewId,
    controller.signal,
  );
}
