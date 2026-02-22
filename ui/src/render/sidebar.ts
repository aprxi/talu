import type { Conversation } from "../types.ts";
import { el, isPinned, getTags, PIN_SVG, FORK_SVG, relativeTime } from "./helpers.ts";

export function renderSidebarItem(
  session: Conversation,
  isActive: boolean,
  isGenerating: boolean = false,
): HTMLElement {
  const pinned = isPinned(session);
  const isForked = session.parent_session_id != null;

  let cls = "sidebar-item";
  if (isActive) cls += " active";
  if (isForked) cls += " forked";
  if (isGenerating) cls += " generating";

  const item = el("div", cls);
  item.dataset["id"] = session.id;

  const content = el("div", "sidebar-item-content");

  // Title row with icons
  const titleRow = el("div", "sidebar-item-title-row");
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
  if (isGenerating) {
    titleRow.appendChild(el("span", "sidebar-generating-dot"));
  }
  const title = el("span", "sidebar-item-title truncate", session.title || "Untitled");
  titleRow.appendChild(title);
  content.appendChild(titleRow);

  // Meta row: model + time
  const meta = el("div", "sidebar-item-meta");
  if (session.model) {
    meta.appendChild(el("span", "truncate", session.model));
  }
  meta.appendChild(el("span", "shrink-0", relativeTime(session.updated_at)));
  content.appendChild(meta);

  // Tags row (compact, max 2 visible)
  const tags = getTags(session);
  if (tags.length > 0) {
    const tagsRow = el("div", "sidebar-item-tags");
    const maxVisible = 2;
    for (let i = 0; i < Math.min(tags.length, maxVisible); i++) {
      const tag = el("span", "sidebar-item-tag", tags[i]);
      tagsRow.appendChild(tag);
    }
    if (tags.length > maxVisible) {
      const more = el("span", "sidebar-item-tag", `+${tags.length - maxVisible}`);
      tagsRow.appendChild(more);
    }
    content.appendChild(tagsRow);
  }

  item.appendChild(content);

  // Pin/unpin button on hover
  const pinBtn = el("button", pinned ? "pin-btn pinned" : "pin-btn");
  pinBtn.innerHTML = PIN_SVG;
  pinBtn.title = pinned ? "Unpin" : "Pin";
  pinBtn.dataset["pin"] = session.id;
  item.appendChild(pinBtn);

  return item;
}

export function renderSectionLabel(text: string): HTMLElement {
  return el("div", "sidebar-section-label", text);
}
