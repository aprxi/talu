import type { Conversation, Item } from "../types.ts";
import { el, isPinned, isArchived, getTags, PIN_SVG, FORK_SVG, relativeTime, formatDate } from "./helpers.ts";
import { RESTORE_ICON_LG, OPEN_EXTERNAL_ICON } from "../icons.ts";

/** Extract a text preview from items (first ~120 chars of user or assistant text). */
function extractPreview(items?: Item[]): string {
  if (!items || items.length === 0) return "";
  for (const item of items) {
    if (item.type === "message" && (item.role === "user" || item.role === "assistant")) {
      for (const part of item.content) {
        const text = ("text" in part) ? part.text.trim() : "";
        if (text) return text.length > 120 ? text.slice(0, 120) + "\u2026" : text;
      }
    }
  }
  return "";
}

/** Search conversation items for the first occurrence of `query` and return a snippet */
function findSnippetInItems(items: Item[] | undefined, query: string): string | null {
  if (!items || !query) return null;
  const lowerQuery = query.toLowerCase();
  for (const item of items) {
    if (item.type !== "message") continue;
    if (item.role !== "user" && item.role !== "assistant") continue;
    for (const part of item.content) {
      if (!("text" in part)) continue;
      const text = part.text;
      const idx = text.toLowerCase().indexOf(lowerQuery);
      if (idx === -1) continue;
      const leadIn = 30;
      const start = Math.max(0, idx - leadIn);
      const end = Math.min(text.length, start + 200);
      let snippet = text.slice(start, end).trim();
      if (start > 0) snippet = "\u2026" + snippet;
      if (end < text.length) snippet += "\u2026";
      return snippet;
    }
  }
  return null;
}

/** Count user + assistant messages in items. */
function countMessages(items?: Item[]): number {
  if (!items) return 0;
  return items.filter((i) => i.type === "message" && (i.role === "user" || i.role === "assistant")).length;
}

/** Build a DocumentFragment with the query substring wrapped in <mark>. */
function renderHighlightedText(text: string, query: string): DocumentFragment {
  const frag = document.createDocumentFragment();
  if (!query) {
    frag.appendChild(document.createTextNode(text));
    return frag;
  }
  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase();
  let cursor = 0;
  while (cursor < text.length) {
    const idx = lowerText.indexOf(lowerQuery, cursor);
    if (idx === -1) {
      frag.appendChild(document.createTextNode(text.slice(cursor)));
      break;
    }
    if (idx > cursor) {
      frag.appendChild(document.createTextNode(text.slice(cursor, idx)));
    }
    const mark = document.createElement("mark");
    mark.textContent = text.slice(idx, idx + query.length);
    frag.appendChild(mark);
    cursor = idx + query.length;
  }
  return frag;
}

export function renderBrowserCard(
  chat: Conversation,
  isSelected: boolean,
  searchQuery?: string,
  activeTagFilters: string[] = [],
  showStatusBadge = false,
): HTMLElement {
  const pinned = isPinned(chat);
  const isForked = chat.parent_session_id != null;
  const archived = isArchived(chat);

  let cardCls = "card browser-card";
  if (isSelected) cardCls += " selected";
  if (isForked) cardCls += " forked";
  if (showStatusBadge && archived) cardCls += " archived";

  const card = el("div", cardCls);
  card.dataset["id"] = chat.id;

  // Top row: checkbox + title + pin icon + model badge + open chevron
  const topRow = el("div", "browser-card-top");

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = isSelected;
  checkbox.className = "browser-checkbox";
  topRow.appendChild(checkbox);

  const titleArea = el("div", "browser-card-title-area");
  const titleRow = el("div", "browser-card-title-row");
  if (pinned) {
    const pinIcon = el("span", "icon-pin shrink-0");
    pinIcon.innerHTML = PIN_SVG;
    titleRow.appendChild(pinIcon);
  }
  if (isForked) {
    const forkIcon = el("span", "icon-fork shrink-0");
    forkIcon.innerHTML = FORK_SVG;
    forkIcon.title = "Forked conversation";
    titleRow.appendChild(forkIcon);
  }
  const titleSpan = el("span", "browser-card-title");
  if (searchQuery) {
    titleSpan.appendChild(renderHighlightedText(chat.title || "Untitled", searchQuery));
  } else {
    titleSpan.textContent = chat.title || "Untitled";
  }
  titleRow.appendChild(titleSpan);
  titleArea.appendChild(titleRow);

  // Badges row (model + project + archived status)
  const badgesRow = el("div", "browser-card-badges");
  if (chat.model) {
    badgesRow.appendChild(el("span", "browser-card-badge", chat.model));
  }
  const projectId = chat.project_id ?? (chat.metadata?.project_id as string | undefined);
  if (projectId) {
    badgesRow.appendChild(el("span", "browser-card-badge project", projectId));
  }
  if (showStatusBadge && archived) {
    badgesRow.appendChild(el("span", "browser-card-badge archived", "Archived"));
  }
  if (badgesRow.children.length > 0) {
    titleArea.appendChild(badgesRow);
  }
  topRow.appendChild(titleArea);

  // Action button: "Open" for normal chats, "Restore" for archived
  const actionBtn = el("button", "browser-card-action");
  if (isArchived(chat)) {
    actionBtn.title = "Restore";
    actionBtn.dataset["action"] = "restore";
    actionBtn.innerHTML = RESTORE_ICON_LG;
  } else {
    actionBtn.title = "Open chat";
    actionBtn.dataset["action"] = "open";
    actionBtn.innerHTML = OPEN_EXTERNAL_ICON;
  }
  topRow.appendChild(actionBtn);

  card.appendChild(topRow);

  // Preview text
  const preview = searchQuery
    ? (chat.search_snippet || findSnippetInItems(chat.items, searchQuery) || extractPreview(chat.items))
    : extractPreview(chat.items);
  if (preview) {
    const previewEl = el("div", searchQuery ? "browser-card-preview full" : "browser-card-preview");
    if (searchQuery) {
      previewEl.appendChild(renderHighlightedText(preview, searchQuery));
    } else {
      previewEl.textContent = preview;
    }
    card.appendChild(previewEl);
  }

  // Tags row
  const tags = getTags(chat);
  if (tags.length > 0) {
    const tagsRow = el("div", "browser-card-tags");

    const sortedTags = [...tags].sort((a, b) => {
      const aActive = activeTagFilters.includes(a);
      const bActive = activeTagFilters.includes(b);
      if (aActive && !bActive) return -1;
      if (!aActive && bActive) return 1;
      return 0;
    });

    for (const tagText of sortedTags) {
      const isActiveFilter = activeTagFilters.includes(tagText);
      let tagCls = "tag-chip";
      if (isActiveFilter) {
        tagCls += " active";
      } else if (activeTagFilters.length > 0) {
        tagCls += " dimmed";
      }

      const tag = el("span", tagCls);
      tag.dataset["tag"] = tagText;
      if (searchQuery && tagText.toLowerCase().includes(searchQuery.toLowerCase())) {
        tag.appendChild(renderHighlightedText(tagText, searchQuery));
      } else {
        tag.textContent = tagText;
      }
      if (isActiveFilter) {
        tag.title = "Click to remove from filter";
      } else if (activeTagFilters.length > 0) {
        tag.title = "Click to add to filter";
      }
      tagsRow.appendChild(tag);
    }
    card.appendChild(tagsRow);
  }

  // Bottom meta row: created, updated, message count
  const meta = el("div", "browser-card-meta");
  meta.appendChild(el("span", undefined, formatDate(chat.created_at)));
  if (chat.updated_at !== chat.created_at) {
    meta.appendChild(el("span", "browser-card-meta-separator", "\u00B7"));
    meta.appendChild(el("span", undefined, "edited " + relativeTime(chat.updated_at)));
  }
  const msgCount = countMessages(chat.items);
  if (msgCount > 0) {
    meta.appendChild(el("span", "browser-card-meta-separator", "\u00B7"));
    meta.appendChild(el("span", undefined, `${msgCount} message${msgCount !== 1 ? "s" : ""}`));
  }
  card.appendChild(meta);

  return card;
}
