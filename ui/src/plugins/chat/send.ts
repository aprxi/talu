import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { api, hooks, notifications, timers, getModelsService, getPromptsService } from "./deps.ts";
import { getSamplingParams } from "./panel-params.ts";
import { refreshSidebar, renderSidebar } from "./sidebar-list.ts";
import { setInputEnabled, showInputBar, hideWelcome } from "./welcome.ts";
import { appendUserMessage, appendAssistantPlaceholder, addAssistantActionButtons, appendStoppedIndicator, scrollToBottom, removeGeneratingIndicator, removeProgressBar } from "./messages.ts";
import { readSSEStream } from "./streaming.ts";
import { clearEventsLog, startResponseEventsStream, stopResponseEventsStream } from "./events.ts";
import { ensureChatHeader, renderChatView } from "./selection.ts";
import {
  clearAttachments,
  composeUserInput,
  hasAttachments,
  isAttachmentUploadInProgress,
  uploadFiles,
} from "./attachments.ts";
import type { Conversation, CreateResponseRequest, InputContentItem, UsageStats } from "../../types.ts";

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

  // Clipboard paste: upload pasted files/images as attachments.
  const handlePaste = (e: ClipboardEvent) => {
    const files = e.clipboardData?.files;
    if (files && files.length > 0) {
      e.preventDefault();
      void uploadFiles(files);
    }
  };
  dom.inputText.addEventListener("paste", handlePaste);
  dom.welcomeInput.addEventListener("paste", handlePaste);
}

export function cancelGeneration(): void {
  if (chatState.streamAbort) {
    chatState.streamAbort.abort();
    chatState.streamAbort = null;
  }
}

export interface StreamOptions {
  text: string;
  input: string | InputContentItem[];
  promptId?: string | null;
  scrollAfterPlaceholder?: boolean;
  discoverSession?: boolean;
  afterResponse?: (chat: Conversation) => void;
}

function resolveInstructions(promptId?: string | null): string | null {
  if (!promptId) return null;
  return getPromptsService()?.getPromptContentById(promptId) ?? null;
}

/** Shared streaming pipeline used by send, welcome-send, and rerun. */
export async function streamResponse(opts: StreamOptions): Promise<void> {
  const myViewId = chatState.activeViewId;
  const isActive = () => myViewId === chatState.activeViewId;

  stopResponseEventsStream();
  clearEventsLog();
  chatState.streamAbort = new AbortController();
  chatState.isGenerating = true;
  setInputEnabled(false);

  appendUserMessage(opts.text, opts.input);
  const { wrapper, body, textEl } = appendAssistantPlaceholder();
  if (opts.scrollAfterPlaceholder) scrollToBottom();

  let userCancelled = false;
  let usageStats: UsageStats | null = null;
  let streamSessionId: string | null = chatState.activeSessionId;
  let sidebarRefreshed = false;

  const onSessionDiscovered = (sid: string) => {
    streamSessionId = sid;
    if (isActive()) {
      chatState.activeSessionId = sid;
    } else {
      chatState.backgroundStreamSessions.add(sid);
    }
    // Optimistically add to the sidebar so the session is navigable immediately,
    // without waiting for the full API refresh round-trip.
    if (!chatState.sessions.some(s => s.id === sid)) {
      const now = Math.floor(Date.now() / 1000);
      chatState.sessions.unshift({
        id: sid,
        object: "session",
        created_at: now,
        updated_at: now,
        model: getModelsService()?.getActiveModel() ?? "",
        title: opts.text.slice(0, 47) || null,
        marker: "active",
        group_id: null,
        parent_session_id: null,
        source_doc_id: null,
        project_id: undefined,
        metadata: {},
      });
      renderSidebar();
    }
    if (!sidebarRefreshed) {
      sidebarRefreshed = true;
      void refreshSidebar();
    }
  };

  try {
    let requestBody: CreateResponseRequest = {
      model: getModelsService()?.getActiveModel() ?? "",
      input: opts.input,
      previous_response_id: chatState.lastResponseId,
      instructions: resolveInstructions(opts.promptId),
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
      if (isActive()) {
        const err = await resp.json().catch(() => null);
        const msg = err?.error?.message ?? `${resp.status} ${resp.statusText}`;
        textEl.textContent = `Error: ${msg}`;
        textEl.classList.add("text-danger");
        notifications.error(msg);
        chatState.isGenerating = false;
        chatState.streamAbort = null;
        setInputEnabled(true);
      }
      if (streamSessionId) {
        chatState.backgroundStreamSessions.delete(streamSessionId);
      }
      return;
    }

    const result = await readSSEStream(
      resp,
      body,
      textEl,
      myViewId,
      onSessionDiscovered,
      (responseId) => startResponseEventsStream(responseId, myViewId),
    );
    usageStats = result.usage;
    streamSessionId = result.sessionId ?? streamSessionId;
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      userCancelled = true;
    } else if (isActive()) {
      const msg = e instanceof Error ? e.message : String(e);
      textEl.textContent = `Error: ${msg}`;
      notifications.error(msg);
    }
  } finally {
    stopResponseEventsStream();
    if (isActive()) removeProgressBar();
  }

  if (streamSessionId) {
    chatState.backgroundStreamSessions.delete(streamSessionId);
    chatState.backgroundStreamDom.delete(streamSessionId);
  }

  if (isActive()) {
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

    // Pick up auto-generated title (runs server-side after generation).
    if (opts.discoverSession && chatState.activeSessionId) {
      timers.setTimeout(() => refreshSidebar(), 3000);
    }
  } else {
    // Background stream completed — refresh sidebar to update status.
    await refreshSidebar();

    // If the user is currently viewing this chat, re-render with full data.
    if (streamSessionId && chatState.activeSessionId === streamSessionId) {
      const dom = getChatDom();
      const updated = await api.getConversation(streamSessionId);
      if (updated.ok && updated.data && chatState.activeSessionId === streamSessionId) {
        chatState.activeChat = updated.data;
        renderChatView(updated.data);
        removeGeneratingIndicator(dom.transcriptContainer);
        showInputBar();
      }
    }
  }
}

interface SendOptions {
  text: string;
  input: string | InputContentItem[];
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
    input: opts.input,
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

  // Explicit selection from dropdown; if "None", auto-apply default when enabled.
  let promptId = dom.welcomePrompt.value || null;
  if (!promptId && chatState.systemPromptEnabled) {
    promptId = getPromptsService()?.getDefaultPromptId() ?? null;
  }
  const input = composeUserInput(text);
  const displayText = typeof input === "string" ? input : text.trim() || "Describe the attached file.";

  await sendAndStream({
    text: displayText,
    input,
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

  const input = composeUserInput(text);
  const displayText = typeof input === "string" ? input : text.trim() || "Describe the attached file.";
  await sendAndStream({ text: displayText, input });
}

/**
 * If the active session ID is still unknown after a send (SSE metadata didn't
 * include it), discover it by fetching the most recently updated conversation.
 */
async function discoverSessionId(): Promise<void> {
  if (chatState.activeSessionId) return;
  const list = await api.listConversations({ limit: 1 });
  if (list.ok && list.data && list.data.data.length > 0) {
    chatState.activeSessionId = list.data.data[0]!.id;
  }
}
