import type {
  Conversation,
  Item,
  MessageItem,
  ReasoningItem,
  FunctionCallItem,
  FunctionCallOutputItem,
  GenerationSettings,
  UsageStats,
  InputImagePart,
  InputFilePart,
} from "../types.ts";
import { el, isArchived, getTags, isThinkingExpanded, formatDate } from "./helpers.ts";
import { sanitizedMarkdown, CODE_COPY_ICON, CODE_CHECK_ICON } from "./markdown.ts";
import { RERUN_ICON, EDIT_ICON, COPY_ICON, EXPORT_ICON, ARCHIVE_ICON, UNARCHIVE_ICON, FORK_ICON, THINKING_ICON, SETTINGS_ICON, STATS_ICON, PLUS_ICON, CLOSE_SMALL_ICON } from "../icons.ts";
import type { ClipboardAccess, ManagedTimers } from "../kernel/types.ts";

let moduleClipboard: ClipboardAccess | null = null;
let moduleTimers: ManagedTimers | null = null;

/** Options for rendering the transcript header. */
export interface HeaderOptions {
  displayModel?: string;
}

// -- Code block rendering -----------------------------------------------------

/** Initialize click handler for copy buttons in code blocks. Attaches to the given root. */
export function initCodeBlockCopyHandler(
  root: HTMLElement | Document,
  clipboard: ClipboardAccess,
  timers: ManagedTimers,
): void {
  moduleClipboard = clipboard;
  moduleTimers = timers;

  root.addEventListener("click", ((e: Event) => {
    if (!(e.target instanceof Element)) return;
    const btn = e.target.closest(".code-block .code-copy");
    if (!btn) return;

    const block = btn.closest(".code-block") as HTMLElement | null;
    if (!block) return;

    const code = block.dataset.code ?? block.querySelector("code")?.textContent ?? "";

    moduleClipboard!.writeText(code).then(() => {
      btn.innerHTML = CODE_CHECK_ICON;
      btn.classList.add("copied");
      moduleTimers!.setTimeout(() => {
        btn.innerHTML = CODE_COPY_ICON;
        btn.classList.remove("copied");
      }, 1500);
    });
  }) as EventListener);
}

/**
 * Render a single code block with header (language + copy button) and content.
 */
function renderCodeBlock(language: string, code: string, complete: boolean): HTMLElement {
  const container = el("div", "code-block");

  const header = el("div", "code-header");

  const langLabel = el("span", "code-lang", language || "code");
  header.appendChild(langLabel);

  if (!complete) {
    const incomplete = el("span", "code-incomplete", "...");
    incomplete.title = "Code block is incomplete";
    header.appendChild(incomplete);
  }

  const copyBtn = el("button", "code-copy");
  copyBtn.innerHTML = CODE_COPY_ICON;
  copyBtn.title = "Copy code";
  copyBtn.addEventListener("click", () => {
    moduleClipboard!.writeText(code).then(() => {
      copyBtn.innerHTML = CODE_CHECK_ICON;
      copyBtn.classList.add("copied");
      moduleTimers!.setTimeout(() => {
        copyBtn.innerHTML = CODE_COPY_ICON;
        copyBtn.classList.remove("copied");
      }, 1500);
    });
  });
  header.appendChild(copyBtn);

  container.appendChild(header);

  const pre = document.createElement("pre");
  const codeEl = document.createElement("code");
  codeEl.textContent = code;
  pre.appendChild(codeEl);
  container.appendChild(pre);

  return container;
}

/**
 * Render output text, using talu_code_blocks metadata if available.
 * Falls back to markdown rendering if no code blocks provided.
 */
export function renderOutputText(
  text: string,
  codeBlocks?: import("../types.ts").CodeBlock[],
): HTMLElement {
  const container = el("div", "markdown-body");

  if (!codeBlocks || codeBlocks.length === 0) {
    container.innerHTML = sanitizedMarkdown(text);
    return container;
  }

  const sorted = [...codeBlocks].sort((a, b) => a.fence_start - b.fence_start);

  let cursor = 0;

  for (const block of sorted) {
    if (block.fence_start > cursor) {
      const prose = text.slice(cursor, block.fence_start);
      if (prose.trim()) {
        const proseDiv = el("div");
        proseDiv.innerHTML = sanitizedMarkdown(prose);
        container.appendChild(proseDiv);
      }
    }

    const lang = text.slice(block.language_start, block.language_end);
    const code = text.slice(block.content_start, block.content_end);

    container.appendChild(renderCodeBlock(lang, code, block.complete));

    cursor = block.fence_end;
  }

  if (cursor < text.length) {
    const remaining = text.slice(cursor);
    if (remaining.trim()) {
      const proseDiv = el("div");
      proseDiv.innerHTML = sanitizedMarkdown(remaining);
      container.appendChild(proseDiv);
    }
  }

  return container;
}

// -- Shared message action button builders ------------------------------------

/** Create user message action buttons (rerun, edit, copy). */
export function createUserActionButtons(): HTMLElement {
  const actions = el("div", "user-msg-actions");

  const rerunBtn = el("button", "user-action-btn");
  rerunBtn.dataset["action"] = "rerun";
  rerunBtn.title = "Re-run";
  rerunBtn.innerHTML = RERUN_ICON;

  const editBtn = el("button", "user-action-btn");
  editBtn.dataset["action"] = "edit";
  editBtn.title = "Edit";
  editBtn.innerHTML = EDIT_ICON;

  const copyBtn = el("button", "user-action-btn");
  copyBtn.dataset["action"] = "copy";
  copyBtn.title = "Copy";
  copyBtn.innerHTML = COPY_ICON;

  actions.appendChild(rerunBtn);
  actions.appendChild(editBtn);
  actions.appendChild(copyBtn);
  return actions;
}

/** Create assistant message action buttons (copy + optional stats). */
export function createAssistantActionButtons(opts?: {
  generation?: GenerationSettings;
  usage?: UsageStats | null;
}): HTMLElement {
  const actions = el("div", "assistant-msg-actions");

  const copyBtn = el("button", "assistant-action-btn");
  copyBtn.dataset["action"] = "copy";
  copyBtn.title = "Copy";
  copyBtn.innerHTML = COPY_ICON;
  actions.appendChild(copyBtn);

  const generation = opts?.generation;
  const usage = opts?.usage;
  if (generation || usage) {
    const statsBtn = el("button", "assistant-action-btn");
    statsBtn.dataset["action"] = "show-message-details";
    statsBtn.title = "Message details";
    statsBtn.innerHTML = STATS_ICON;
    if (generation) statsBtn.dataset["generation"] = JSON.stringify(generation);
    if (usage) statsBtn.dataset["usage"] = JSON.stringify(usage);
    actions.appendChild(statsBtn);
  }

  return actions;
}

// -- Transcript header --------------------------------------------------------

export function renderTranscriptHeader(chat: Conversation, options?: HeaderOptions): HTMLElement {
  const header = el("div", "transcript-header");
  header.dataset["chatId"] = chat.id;

  const container = el("div", "transcript-header-container");

  const row = el("div", "transcript-header-row");

  // LEFT: Date + tags
  const metaGroup = el("div", "transcript-meta");

  const date = el("span", "transcript-date", formatDate(chat.created_at));
  metaGroup.appendChild(date);

  // TAGS
  const tagsContainer = el("div", "transcript-tags");
  tagsContainer.dataset["chatId"] = chat.id;

  const tags = getTags(chat);
  for (const tag of tags) {
    tagsContainer.appendChild(renderTag(tag));
  }

  if (tags.length < 5) {
    const addBtn = el("button", "add-tag-btn");
    addBtn.innerHTML = `${PLUS_ICON} Tag`;
    addBtn.dataset["action"] = "add-tag";
    tagsContainer.appendChild(addBtn);
  }

  metaGroup.appendChild(tagsContainer);
  row.appendChild(metaGroup);

  // RIGHT: All actions together
  const actions = el("div", "transcript-actions");

  const thinkingBtn = el("button", isThinkingExpanded() ? "chat-action active" : "chat-action");
  thinkingBtn.id = "chat-thinking-toggle";
  thinkingBtn.innerHTML = THINKING_ICON;
  thinkingBtn.title = isThinkingExpanded() ? "Collapse thoughts" : "Expand thoughts";
  thinkingBtn.dataset["action"] = "toggle-thinking";
  actions.appendChild(thinkingBtn);

  actions.appendChild(el("div", "actions-separator"));

  const copyBtn = el("button", "chat-action");
  copyBtn.innerHTML = COPY_ICON;
  copyBtn.title = "Copy conversation";
  copyBtn.dataset["action"] = "copy";
  actions.appendChild(copyBtn);

  const exportBtn = el("button", "chat-action");
  exportBtn.innerHTML = EXPORT_ICON;
  exportBtn.title = "Export conversation";
  exportBtn.dataset["action"] = "export";
  actions.appendChild(exportBtn);

  const forkBtn = el("button", "chat-action");
  forkBtn.innerHTML = FORK_ICON;
  forkBtn.title = "Fork conversation";
  forkBtn.dataset["action"] = "fork";
  actions.appendChild(forkBtn);

  // Archive/Unarchive
  const archived = isArchived(chat);
  const archiveBtn = el("button", "chat-action");
  archiveBtn.innerHTML = archived ? UNARCHIVE_ICON : ARCHIVE_ICON;
  archiveBtn.title = archived ? "Restore" : "Archive";
  archiveBtn.dataset["action"] = archived ? "unarchive" : "archive";
  actions.appendChild(archiveBtn);

  // Settings panel toggle
  const tuningBtn = el("button", "chat-action");
  tuningBtn.innerHTML = SETTINGS_ICON;
  tuningBtn.title = "Settings";
  tuningBtn.dataset["action"] = "toggle-tuning";
  actions.appendChild(tuningBtn);

  // Slot for contributed menu items.
  const menuSlot = el("div", "menu-slot");
  menuSlot.dataset["slot"] = "chat:transcript-actions";
  actions.appendChild(menuSlot);

  row.appendChild(actions);
  container.appendChild(row);

  header.appendChild(container);

  return header;
}

/** Render a single tag pill with remove button. */
function renderTag(tag: string): HTMLElement {
  const pill = el("span", "tag-pill");
  pill.dataset["tag"] = tag;

  const label = el("span", undefined, tag);
  pill.appendChild(label);

  const removeBtn = el("button", "tag-remove");
  removeBtn.dataset["tag"] = tag;
  removeBtn.innerHTML = CLOSE_SMALL_ICON;
  removeBtn.title = "Remove tag";
  pill.appendChild(removeBtn);

  return pill;
}

// -- Transcript messages ------------------------------------------------------

/**
 * Normalize conversation items for display:
 * - system/developer messages containing user input → promote to user role
 * - system+empty-user pairs → merge into one user message, skip the empty user
 * - system/developer messages that are pure prompts (first item, before any user turn) → drop
 */
function normalizeItems(items: Item[]): Item[] {
  const result: Item[] = [];
  for (let i = 0; i < items.length; i++) {
    const item = items[i]!;
    if (item.type !== "message") {
      result.push(item);
      continue;
    }
    if (item.role === "system" || item.role === "developer") {
      const next = items[i + 1];

      // Pattern: system + empty user → merge into one user message
      if (
        next?.type === "message" &&
        next.role === "user" &&
        !next.content.some((p) => "text" in p && (p as { text: string }).text.trim())
      ) {
        // Merge content from both (system text + user attachments)
        const merged = { ...item, role: "user" as const, content: [...item.content, ...next.content] };
        result.push(merged);
        i++; // skip the empty user message
        continue;
      }

      // Pattern: system followed by assistant/reasoning/function_call → user input stored as system
      if (
        !next ||
        next.type !== "message" ||
        next.role !== "user"
      ) {
        // Has text content → treat as user input
        const hasText = item.content.some((p) => "text" in p && (p as { text: string }).text.trim());
        if (hasText) {
          result.push({ ...item, role: "user" });
          continue;
        }
      }

      // Pure system/developer prompt with no user-visible text, or followed by user with text → drop
      continue;
    }
    result.push(item);
  }
  return result;
}

export function renderTranscript(items: Item[]): HTMLElement {
  const container = el("div", "transcript-messages");
  container.dataset["transcriptItems"] = "";
  const normalized = normalizeItems(items);
  for (const item of normalized) {
    container.appendChild(renderItem(item));
  }
  // Show "Stopped" after the last item if generation was cancelled
  const last = normalized[normalized.length - 1];
  if (
    last &&
    (last.type === "message" || last.type === "reasoning") &&
    last.finish_reason === "cancelled"
  ) {
    const stopped = el("div", undefined, "Stopped");
    stopped.style.marginTop = "0.25rem";
    stopped.style.fontSize = "12px";
    stopped.style.color = "var(--text-muted)";
    stopped.style.fontStyle = "italic";
    container.appendChild(stopped);
  }
  return container;
}

function renderItem(item: Item): HTMLElement {
  switch (item.type) {
    case "message":
      return renderMessage(item);
    case "reasoning":
      return renderReasoning(item);
    case "function_call":
      return renderFunctionCall(item);
    case "function_call_output":
      return renderFunctionOutput(item);
    default:
      const unknown = el("div", "card", `Unknown item type: ${(item as Item).type}`);
      unknown.style.fontSize = "12px";
      unknown.style.color = "var(--text-muted)";
      return unknown;
  }
}

/**
 * Convert an image_url from the conversation to a browser-accessible src.
 *
 * Handles:
 *   - file:// blob paths → /v1/blobs/{sha256hex}
 *   - file_* IDs → /v1/files/{id}/content
 *   - data: URIs and http(s) URLs → pass through
 */
export function imageUrlToSrc(url: string): string {
  if (url.startsWith("file://")) {
    const path = url.slice("file://".length);
    const parts = path.split("/");
    const hash = parts[parts.length - 1] ?? "";
    if (hash.length === 64) return `/v1/blobs/${hash}`;
  }
  if (url.startsWith("file_")) return `/v1/files/${url}/content`;
  return url;
}

function renderImagePart(part: InputImagePart): HTMLElement {
  const container = el("div", "msg-image");
  const img = document.createElement("img");
  img.src = imageUrlToSrc(part.image_url);
  img.alt = "Uploaded image";
  img.loading = "lazy";
  container.appendChild(img);
  return container;
}

function renderFilePart(part: InputFilePart): HTMLElement {
  const pill = el("span", "msg-file-pill");
  pill.textContent = part.filename ?? "Attached file";
  return pill;
}

function renderContentParts(parts: MessageItem["content"], container: HTMLElement): void {
  for (const part of parts) {
    if (part.type === "input_text") {
      const p = el("div");
      p.textContent = part.text;
      container.appendChild(p);
    } else if (part.type === "output_text") {
      container.appendChild(renderOutputText(part.text, part.talu_code_blocks));
    } else if (part.type === "input_image") {
      container.appendChild(renderImagePart(part));
    } else if (part.type === "input_file") {
      container.appendChild(renderFilePart(part));
    }
  }
}

function renderMessage(item: MessageItem): HTMLElement {
  const isUser = item.role === "user";

  if (isUser) {
    const wrapper = el("div", "user-msg");
    const bubble = el("div", "user-bubble");
    renderContentParts(item.content, bubble);
    wrapper.appendChild(bubble);
    wrapper.appendChild(createUserActionButtons());
    return wrapper;
  }

  // Assistant: left-aligned, clean flow
  const wrapper = el("div", "assistant-msg");
  const body = el("div", "assistant-body");
  renderContentParts(item.content, body);
  wrapper.appendChild(body);
  wrapper.appendChild(createAssistantActionButtons({ generation: item.generation }));
  return wrapper;
}

function renderReasoning(item: ReasoningItem): HTMLElement {
  const details = document.createElement("details");
  details.className = "reasoning-block";
  if (isThinkingExpanded()) details.open = true;

  const summary = el("summary", "reasoning-summary", "Thought process");
  details.appendChild(summary);

  const body = el("div", "reasoning-body");

  const parts = item.content?.length ? item.content : item.summary;
  if (parts) {
    for (const part of parts) {
      const div = el("div");
      div.innerHTML = sanitizedMarkdown(part.text);
      body.appendChild(div);
    }
  }
  details.appendChild(body);
  return details;
}

function renderFunctionCall(item: FunctionCallItem): HTMLElement {
  const details = document.createElement("details");
  details.className = "function-block";

  const summary = el("summary", "tool-summary");
  summary.textContent = `\u{1F527} ${item.name}`;
  details.appendChild(summary);

  const content = el("div", "tool-body");
  const pre = el("pre", "tool-pre");
  try {
    pre.textContent = JSON.stringify(JSON.parse(item.arguments), null, 2);
  } catch {
    pre.textContent = item.arguments;
  }
  content.appendChild(pre);
  details.appendChild(content);
  return details;
}

function renderFunctionOutput(item: FunctionCallOutputItem): HTMLElement {
  const details = document.createElement("details");
  details.className = "function-block";

  const summary = el("summary", "tool-summary", "Tool Output");
  summary.style.color = "var(--text-secondary)";
  details.appendChild(summary);

  const content = el("div", "tool-body");
  const pre = el("pre", "tool-pre");
  pre.textContent = item.output;
  content.appendChild(pre);
  details.appendChild(content);
  return details;
}
