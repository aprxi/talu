import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { notifications, clipboard } from "./deps.ts";
import { getUserMessageInfo } from "./rerun.ts";
import { doRerun } from "./rerun.ts";
import type { InputContentItem, InputTextPart, OutputTextPart } from "../../types.ts";

/** Copy user message text to clipboard. */
export function handleCopyUserMessage(msgIndex: number): void {
  const info = getUserMessageInfo(msgIndex);
  if (info) {
    clipboard.writeText(info.text).then(
      () => notifications.info("Copied to clipboard"),
      () => notifications.error("Failed to copy"),
    );
  }
}

/** Copy assistant message text to clipboard. */
export function handleCopyAssistantMessage(msgIndex: number): void {
  const text = getAssistantMessageText(msgIndex);
  if (text) {
    clipboard.writeText(text).then(
      () => notifications.info("Copied to clipboard"),
      () => notifications.error("Failed to copy"),
    );
  }
}

/** Get assistant message text by display index. */
function getAssistantMessageText(msgIndex: number): string | null {
  if (!chatState.activeChat?.items) return null;

  const items = chatState.activeChat.items;
  let assistantMsgCount = 0;
  for (const item of items) {
    if (item.type === "message" && item.role === "assistant") {
      if (assistantMsgCount === msgIndex) {
        return item.content
          .filter((p): p is OutputTextPart => p.type === "output_text")
          .map((p) => p.text)
          .join("\n");
      }
      assistantMsgCount++;
    }
  }
  return null;
}

/** Handle edit user message - show inline editor, then re-run with edited text. */
export function handleEditUserMessage(msgIndex: number): void {
  const info = getUserMessageInfo(msgIndex);
  if (!info) return;

  const tc = getChatDom().transcriptContainer;
  const items = tc.querySelector("[data-transcript-items]");
  if (!items) return;

  const userMsgs = items.querySelectorAll(".user-msg");
  const userMsgEl = userMsgs[msgIndex] as HTMLElement | undefined;
  if (!userMsgEl) return;

  // Check if already editing
  if (userMsgEl.querySelector(".edit-textarea")) return;

  const bubble = userMsgEl.querySelector("div:first-child") as HTMLElement;
  if (!bubble) return;

  const originalContent = bubble.innerHTML;

  // Keep images/files visible, only replace text nodes with textarea.
  // Remove plain text divs (children that aren't .msg-image or .msg-file-pill).
  for (const child of Array.from(bubble.children)) {
    const el = child as HTMLElement;
    if (!el.classList.contains("msg-image") && !el.classList.contains("msg-file-pill")) {
      el.remove();
    }
  }

  const textarea = document.createElement("textarea");
  textarea.className = "edit-textarea";
  textarea.value = info.text;
  textarea.rows = Math.max(2, info.text.split("\n").length);
  bubble.appendChild(textarea);

  const actions = document.createElement("div");
  actions.className = "edit-actions";

  const saveBtn = document.createElement("button");
  saveBtn.className = "btn btn-primary btn-sm";
  saveBtn.textContent = "Save & Submit";

  const cancelBtn = document.createElement("button");
  cancelBtn.className = "btn btn-ghost btn-sm";
  cancelBtn.textContent = "Cancel";

  actions.appendChild(saveBtn);
  actions.appendChild(cancelBtn);
  bubble.appendChild(actions);

  textarea.focus();
  textarea.select();

  const autoResize = () => {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 300) + "px";
  };
  textarea.addEventListener("input", autoResize);
  autoResize();

  const cancelEdit = () => {
    bubble.innerHTML = originalContent;
  };

  cancelBtn.addEventListener("click", cancelEdit);

  saveBtn.addEventListener("click", () => {
    const newText = textarea.value.trim();
    if (!newText) {
      cancelEdit();
      return;
    }
    bubble.innerHTML = originalContent;
    // Replace text in the input with the edited version, preserving images/files.
    const editedInput = replaceInputText(info.input, newText);
    doRerun(newText, info.forkBeforeIndex, editedInput);
  });

  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      e.preventDefault();
      cancelEdit();
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      saveBtn.click();
    }
  });
}

/** Replace text parts in structured input, preserving images/files. */
function replaceInputText(
  input: string | InputContentItem[],
  newText: string,
): string | InputContentItem[] {
  if (typeof input === "string") return newText;
  return input.map((item) => {
    if (item.type !== "message" || item.role !== "user") return item;
    const nonText = item.content.filter((p) => p.type !== "input_text");
    return { ...item, content: [{ type: "input_text" as const, text: newText }, ...nonText] };
  });
}
