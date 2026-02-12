import { getChatDom } from "./dom.ts";
import { createUserActionButtons, createAssistantActionButtons } from "../../render/transcript.ts";
import type { Conversation, UsageStats, GenerationSettings } from "../../types.ts";

/** Scroll the transcript container to the bottom. */
export function scrollToBottom(): void {
  getChatDom().transcriptContainer.scrollTop = getChatDom().transcriptContainer.scrollHeight;
}

/** Append a muted "Stopped" label below the assistant message text. */
export function appendStoppedIndicator(textEl: HTMLElement): void {
  const indicator = document.createElement("div");
  indicator.className = "stopped-indicator";
  indicator.textContent = "Stopped";
  textEl.parentElement?.appendChild(indicator);
}

export function appendUserMessage(text: string): void {
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
