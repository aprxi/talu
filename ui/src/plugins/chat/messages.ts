import { getChatDom } from "./dom.ts";
import { createUserActionButtons, createAssistantActionButtons, imageUrlToSrc } from "../../render/transcript.ts";
import type { Conversation, InputContentItem, UsageStats, GenerationSettings } from "../../types.ts";

/** Scroll the transcript container to the bottom (unconditionally). */
export function scrollToBottom(): void {
  const el = getChatDom().transcriptContainer;
  el.scrollTop = el.scrollHeight;
}

/** Scroll to bottom only if the user hasn't scrolled up. */
export function scrollToBottomIfNear(): void {
  const el = getChatDom().transcriptContainer;
  const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
  if (distanceFromBottom < 80) {
    el.scrollTop = el.scrollHeight;
  }
}

/** Append a muted "Stopped" label below the assistant message text. */
export function appendStoppedIndicator(textEl: HTMLElement): void {
  const indicator = document.createElement("div");
  indicator.className = "stopped-indicator";
  indicator.textContent = "Stopped";
  textEl.parentElement?.appendChild(indicator);
}

export function appendUserMessage(text: string, input?: string | InputContentItem[]): void {
  const tc = getChatDom().transcriptContainer;

  // Remove empty state placeholder if present
  const emptyState = tc.querySelector("[data-empty-state]");
  if (emptyState) emptyState.remove();

  // Ensure message container exists
  let items = tc.querySelector("[data-transcript-items]") as HTMLElement | null;
  if (!items) {
    items = document.createElement("div");
    items.className = "transcript-messages";
    items.dataset["transcriptItems"] = "";
    tc.appendChild(items);
  }

  // Right-aligned user bubble with action buttons
  const wrapper = document.createElement("div");
  wrapper.className = "user-msg";

  const bubble = document.createElement("div");
  bubble.className = "user-bubble";

  // Render inline images/files from structured input.
  if (Array.isArray(input)) {
    for (const item of input) {
      for (const part of item.content) {
        if (part.type === "input_image") {
          const container = document.createElement("div");
          container.className = "msg-image";
          const img = document.createElement("img");
          img.src = imageUrlToSrc(part.image_url);
          img.alt = "Uploaded image";
          img.loading = "lazy";
          container.appendChild(img);
          bubble.appendChild(container);
        } else if (part.type === "input_file") {
          const pill = document.createElement("span");
          pill.className = "msg-file-pill";
          pill.textContent = part.filename ?? "Attached file";
          bubble.appendChild(pill);
        }
      }
    }
  }

  const msgText = document.createElement("div");
  msgText.textContent = text;
  bubble.appendChild(msgText);

  wrapper.appendChild(bubble);
  wrapper.appendChild(createUserActionButtons());
  items.appendChild(wrapper);
  scrollToBottom();
}

export function appendAssistantPlaceholder(): { wrapper: HTMLElement; body: HTMLElement; textEl: HTMLElement } {
  const tc = getChatDom().transcriptContainer;

  // Ensure message container exists
  let items = tc.querySelector("[data-transcript-items]") as HTMLElement | null;
  if (!items) {
    items = document.createElement("div");
    items.className = "transcript-messages";
    items.dataset["transcriptItems"] = "";
    tc.appendChild(items);
  }

  const wrapper = document.createElement("div");
  wrapper.className = "assistant-msg";

  const body = document.createElement("div");
  body.className = "assistant-body";
  const textEl = document.createElement("div");
  textEl.className = "markdown-body";
  body.appendChild(textEl);

  wrapper.appendChild(body);
  items.appendChild(wrapper);
  scrollToBottom();
  return { wrapper, body, textEl };
}

/** Create or update a progress bar at the bottom of the transcript container. */
export function updateProgressBar(phase: string, current: number, total: number): void {
  const tc = getChatDom().transcriptContainer;
  let bar = tc.querySelector(".chat-progress") as HTMLElement | null;
  if (!bar) {
    bar = document.createElement("div");
    bar.className = "chat-progress";
    bar.innerHTML =
      '<span class="chat-progress-label"></span>' +
      '<div class="chat-progress-track"><div class="chat-progress-fill"></div></div>' +
      '<span class="chat-progress-pct"></span>';
    tc.appendChild(bar);
  }
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;
  (bar.querySelector(".chat-progress-label") as HTMLElement).textContent = phase;
  (bar.querySelector(".chat-progress-fill") as HTMLElement).style.width = `${pct}%`;
  (bar.querySelector(".chat-progress-pct") as HTMLElement).textContent = `${pct}%`;
}

/** Remove the progress bar from the transcript container. */
export function removeProgressBar(): void {
  getChatDom().transcriptContainer.querySelector(".chat-progress")?.remove();
}

/** Append a "Generating response…" indicator to the given container. */
export function appendGeneratingIndicator(container: HTMLElement): void {
  removeGeneratingIndicator(container);
  const el = document.createElement("div");
  el.className = "chat-generating";
  el.textContent = "Generating response\u2026";
  container.appendChild(el);
}

/** Remove a generating indicator from the given container. */
export function removeGeneratingIndicator(container: HTMLElement): void {
  container.querySelector(".chat-generating")?.remove();
}

/** Add action buttons (copy, stats) to a streamed assistant message after generation completes. */
export function addAssistantActionButtons(
  wrapper: HTMLElement,
  chat: Conversation,
  usage?: UsageStats | null,
): void {
  // Find the last assistant message's generation data
  const items = chat.items ?? [];
  let generation: GenerationSettings | undefined;
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i];
    if (item && item.type === "message" && item.role === "assistant") {
      generation = item.generation;
      break;
    }
  }

  wrapper.appendChild(createAssistantActionButtons({ generation, usage }));
}
