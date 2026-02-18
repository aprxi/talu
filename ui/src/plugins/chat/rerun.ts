import { chatState } from "./state.ts";
import { api, notifications } from "./deps.ts";
import { showInputBar } from "./welcome.ts";
import { streamResponse } from "./send.ts";
import { renderChatView } from "./selection.ts";
import type {
  ContentPart,
  InputContentItem,
  InputContentPart,
  InputTextPart,
  Item,
} from "../../types.ts";

/** Handle re-run from a specific user message. */
export function handleRerunFromMessage(msgIndex: number): void {
  const info = getUserMessageInfo(msgIndex);
  if (info) {
    doRerun(info.text, info.forkBeforeIndex, info.input);
  }
}

/** Execute re-run with given text, forking from the given item index. */
export async function doRerun(
  text: string,
  itemIndex: number,
  input?: string | InputContentItem[],
): Promise<void> {
  if (!chatState.activeSessionId || chatState.isGenerating) return;

  const forkRes = await api.forkConversation(chatState.activeSessionId, { target_item_id: itemIndex });
  if (!forkRes.ok || !forkRes.data) {
    notifications.error(forkRes.error ?? "Failed to fork conversation");
    return;
  }

  const forkedId = forkRes.data.id;
  chatState.activeSessionId = forkedId;
  chatState.lastResponseId = null;

  const forkedFull = await api.getConversation(forkedId);
  if (!forkedFull.ok || !forkedFull.data) {
    notifications.error("Failed to load forked conversation");
    return;
  }
  chatState.activeChat = forkedFull.data;

  renderChatView(forkedFull.data);
  showInputBar();

  await streamResponse({ text, input: input ?? text, scrollAfterPlaceholder: true });
}

/** Build structured input from content parts when multimodal content is present. */
function buildInputFromParts(parts: ContentPart[]): string | InputContentItem[] {
  const hasMultimodal = parts.some(
    (p) => p.type === "input_image" || p.type === "input_file",
  );
  if (!hasMultimodal) {
    return parts
      .filter((p): p is InputTextPart => p.type === "input_text")
      .map((p) => p.text)
      .join("\n");
  }

  const contentParts: InputContentPart[] = [];
  for (const part of parts) {
    if (part.type === "input_text") {
      contentParts.push({ type: "input_text", text: part.text });
    } else if (part.type === "input_image") {
      contentParts.push({ type: "input_image", image_url: part.image_url });
    } else if (part.type === "input_file") {
      const filePart: InputContentPart = { type: "input_file", file_url: part.file_data ?? "" };
      if (part.filename) (filePart as { filename?: string }).filename = part.filename;
      contentParts.push(filePart);
    }
  }
  return [{ type: "message", role: "user", content: contentParts }];
}

/** Find the last user message text from the conversation items. */
export function findLastUserMessage(items: Item[]): { text: string; itemIndex: number } | null {
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i]!;
    if (item.type !== "message") continue;
    if (item.role === "user") {
      const text = item.content
        .filter((p): p is InputTextPart => p.type === "input_text")
        .map((p) => p.text)
        .join("\n");
      if (text.trim()) return { text, itemIndex: i };

      // Workaround: server stores `input` as system message, leaving user message empty.
      if (i > 0) {
        const prev = items[i - 1]!;
        if (prev.type === "message" && prev.role === "system") {
          const sysText = prev.content
            .filter((p): p is InputTextPart => p.type === "input_text")
            .map((p) => p.text)
            .join("\n");
          if (sysText.trim()) return { text: sysText, itemIndex: i };
        }
      }
    }
  }
  return null;
}

/** Get user message info by display index (returns text, structured input, and fork point). */
export function getUserMessageInfo(
  msgIndex: number,
): { text: string; input: string | InputContentItem[]; forkBeforeIndex: number } | null {
  if (!chatState.activeChat?.items) return null;

  const items = chatState.activeChat.items;
  const userMsgPositions: { itemIndex: number; text: string; content: ContentPart[] }[] = [];

  for (let i = 0; i < items.length; i++) {
    const item = items[i]!;
    if (item.type !== "message") continue;

    if (item.role === "system" || item.role === "developer") {
      const next = items[i + 1];
      if (
        next?.type === "message" &&
        next.role === "user" &&
        !next.content.some((p) => "text" in p && (p as InputTextPart).text.trim())
      ) {
        const sysText = item.content
          .filter((p): p is InputTextPart => p.type === "input_text")
          .map((p) => p.text)
          .join("\n");
        // Merge content from both the system message and the user message
        const mergedContent = [...item.content, ...next.content];
        userMsgPositions.push({ itemIndex: i, text: sysText, content: mergedContent });
        i++; // Skip the empty user message
        continue;
      }
      continue;
    }

    if (item.role === "user") {
      const text = item.content
        .filter((p): p is InputTextPart => p.type === "input_text")
        .map((p) => p.text)
        .join("\n");
      userMsgPositions.push({ itemIndex: i, text, content: item.content });
    }
  }

  if (msgIndex < 0 || msgIndex >= userMsgPositions.length) return null;

  const pos = userMsgPositions[msgIndex]!;
  if (!pos.text.trim()) return null;

  const forkBeforeIndex = pos.itemIndex > 0 ? pos.itemIndex : 0;
  const input = buildInputFromParts(pos.content);
  return { text: pos.text, input, forkBeforeIndex };
}
