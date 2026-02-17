import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { api, hooks, notifications, getModelsService, getPromptsService } from "./deps.ts";
import { getSamplingParams } from "./panel-params.ts";
import { refreshSidebar } from "./sidebar-list.ts";
import { setInputEnabled, showInputBar, hideWelcome } from "./welcome.ts";
import { appendUserMessage, appendAssistantPlaceholder, addAssistantActionButtons, appendStoppedIndicator, scrollToBottom } from "./messages.ts";
import { readSSEStream } from "./streaming.ts";
import { ensureChatHeader } from "./selection.ts";
import {
  clearAttachments,
  composeUserInputWithAttachments,
  hasAttachments,
  isAttachmentUploadInProgress,
} from "./attachments.ts";
import type { Conversation, CreateResponseRequest, UsageStats } from "../../types.ts";

export function setupInputEvents(): void {
  const dom = getChatDom();

  // Bottom bar input (active conversation)
  dom.inputSend.addEventListener("click", () => {
    if (chatState.isGenerating) { cancelGeneration(); } else { handleSend(); }
  });
  dom.inputText.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });
  dom.inputText.addEventListener("input", () => {
    dom.inputText.style.height = "auto";
    dom.inputText.style.height = Math.min(dom.inputText.scrollHeight, 200) + "px";
  });

  // Welcome state input (no conversation selected)
  dom.welcomeSend.addEventListener("click", () => handleWelcomeSend());
  dom.welcomeInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleWelcomeSend();
    }
  });
  dom.welcomeInput.addEventListener("input", () => {
    dom.welcomeInput.style.height = "auto";
    dom.welcomeInput.style.height = Math.min(dom.welcomeInput.scrollHeight, 200) + "px";
  });
}

export function cancelGeneration(): void {
  if (chatState.streamAbort) {
    chatState.streamAbort.abort();
    chatState.streamAbort = null;
  }
}

export interface StreamOptions {
  text: string;
  promptId?: string | null;
  scrollAfterPlaceholder?: boolean;
  discoverSession?: boolean;
  afterResponse?: (chat: Conversation) => void;
}

/** Shared streaming pipeline used by send, welcome-send, and rerun. */
export async function streamResponse(opts: StreamOptions): Promise<void> {
  chatState.streamAbort = new AbortController();
  chatState.isGenerating = true;
  setInputEnabled(false);

  appendUserMessage(opts.text);
  const { wrapper, body, textEl } = appendAssistantPlaceholder();
  if (opts.scrollAfterPlaceholder) scrollToBottom();

  let userCancelled = false;
  let usageStats: UsageStats | null = null;
  try {
    let requestBody: CreateResponseRequest = {
      model: getModelsService()?.getActiveModel() ?? "",
      input: opts.text,
      previous_response_id: chatState.lastResponseId,
      session_id: chatState.activeSessionId,
      prompt_id: opts.promptId ?? undefined,
      ...getSamplingParams(),
    };

    // Run chat.send.before hook — plugins can modify the request or block it.
    const hookResult = await hooks.run<CreateResponseRequest>("chat.send.before", requestBody);
    if (hookResult && typeof hookResult === "object" && "$block" in hookResult) {
      notifications.warning((hookResult as { reason: string }).reason ?? "Send blocked by plugin");
      chatState.isGenerating = false;
      chatState.streamAbort = null;
      setInputEnabled(true);
      return;
    }
    requestBody = hookResult as CreateResponseRequest;

    const resp = await api.createResponse(requestBody, chatState.streamAbort.signal);

    if (!resp.ok) {
      const err = await resp.json().catch(() => null);
      const msg = err?.error?.message ?? `${resp.status} ${resp.statusText}`;
      textEl.textContent = `Error: ${msg}`;
      textEl.classList.add("text-danger");
      notifications.error(msg);
      chatState.isGenerating = false;
      chatState.streamAbort = null;
      setInputEnabled(true);
      return;
    }

    usageStats = await readSSEStream(resp, body, textEl);
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      userCancelled = true;
    } else {
      const msg = e instanceof Error ? e.message : String(e);
      textEl.textContent = `Error: ${msg}`;
      notifications.error(msg);
    }
  }

  chatState.isGenerating = false;
  chatState.streamAbort = null;
  setInputEnabled(true);

  if (userCancelled) {
    appendStoppedIndicator(textEl);
  }

  if (opts.discoverSession) {
    await discoverSessionId();
  }

  if (chatState.activeSessionId) {
    const updated = await api.getConversation(chatState.activeSessionId);
    if (updated.ok && updated.data) {
      // Run chat.receive.after hook — plugins can inspect/transform the response.
      const afterResult = await hooks.run<Conversation>("chat.receive.after", updated.data);
      const conversation = (afterResult && typeof afterResult === "object" && "$block" in afterResult)
        ? updated.data // Ignore $block on receive — the response already happened.
        : afterResult as Conversation;
      chatState.activeChat = conversation;
      opts.afterResponse?.(conversation);
      addAssistantActionButtons(wrapper, conversation, usageStats);
    }
  }

  await refreshSidebar();
}

interface SendOptions {
  text: string;
  promptId?: string | null;
  beforeSend?: () => void;
  afterResponse?: (chat: Conversation) => void;
}

async function sendAndStream(opts: SendOptions): Promise<void> {
  const dom = getChatDom();
  opts.beforeSend?.();
  dom.inputText.value = "";
  dom.inputText.style.height = "auto";
  dom.welcomeInput.value = "";
  dom.welcomeInput.style.height = "auto";
  clearAttachments();
  await streamResponse({
    text: opts.text,
    promptId: opts.promptId,
    discoverSession: true,
    afterResponse: opts.afterResponse,
  });
}

/** Handle send from the centered welcome input — transitions to active conversation view. */
async function handleWelcomeSend(): Promise<void> {
  const dom = getChatDom();
  const text = dom.welcomeInput.value;
  if (chatState.isGenerating || isAttachmentUploadInProgress()) return;
  if (!text.trim() && !hasAttachments()) return;

  const promptId = getPromptsService()?.getSelectedPromptId() ?? null;
  const composedText = composeUserInputWithAttachments(text);

  await sendAndStream({
    text: composedText,
    promptId,
    beforeSend() {
      hideWelcome();
      showInputBar();
      dom.transcriptContainer.innerHTML = "";
    },
    afterResponse: ensureChatHeader,
  });
}

async function handleSend(): Promise<void> {
  const dom = getChatDom();
  const text = dom.inputText.value;
  if (chatState.isGenerating || isAttachmentUploadInProgress()) return;
  if (!text.trim() && !hasAttachments()) return;

  await sendAndStream({ text: composeUserInputWithAttachments(text) });
}

/**
 * If the active session ID is still unknown after a send (SSE metadata didn't
 * include it), discover it by fetching the most recently updated conversation.
 */
async function discoverSessionId(): Promise<void> {
  if (chatState.activeSessionId) return;
  const list = await api.listConversations(null, 1);
  if (list.ok && list.data && list.data.data.length > 0) {
    chatState.activeSessionId = list.data.data[0]!.id;
  }
}
