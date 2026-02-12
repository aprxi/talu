import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { api, notifications, timers } from "./deps.ts";
import { getTags } from "../../render/helpers.ts";
import { escapeHtml } from "../../utils/helpers.ts";
import { renderSidebar } from "./sidebar-list.ts";
import { PLUS_ICON } from "../../icons.ts";
import type { Conversation } from "../../types.ts";

/** Show inline input for adding a tag. */
export function handleAddTagPrompt(): void {
  if (!chatState.activeSessionId || !chatState.activeChat) {
    notifications.error("No active conversation");
    return;
  }

  const tagsContainer = getChatDom().transcriptContainer.querySelector<HTMLElement>(".transcript-tags");
  if (!tagsContainer) return;

  // Check if input already exists
  if (tagsContainer.querySelector(".tag-input-wrapper")) return;

  // Hide the add button
  const addBtn = tagsContainer.querySelector(".add-tag-btn");
  if (addBtn) addBtn.classList.add("hidden");

  // Create inline input
  const wrapper = document.createElement("div");
  wrapper.className = "tag-input-wrapper";

  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "tag name";
  input.className = "tag-input";
  input.maxLength = 50;

  const saveBtn = document.createElement("button");
  saveBtn.className = "tag-input-save";
  saveBtn.textContent = "Add";

  const cancelBtn = document.createElement("button");
  cancelBtn.className = "tag-input-cancel";
  cancelBtn.textContent = "\u00d7";

  wrapper.appendChild(input);
  wrapper.appendChild(saveBtn);
  wrapper.appendChild(cancelBtn);
  tagsContainer.appendChild(wrapper);

  input.focus();

  const cleanup = () => {
    wrapper.remove();
    if (addBtn) addBtn.classList.remove("hidden");
  };

  const submit = () => {
    const tag = input.value.trim();
    if (tag) {
      addTagToChat(tag);
    }
    cleanup();
  };

  saveBtn.addEventListener("click", submit);
  cancelBtn.addEventListener("click", cleanup);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      submit();
    } else if (e.key === "Escape") {
      cleanup();
    }
  });
  input.addEventListener("blur", () => {
    // Delay to allow button clicks
    timers.setTimeout(() => {
      if (!wrapper.contains(document.activeElement)) {
        cleanup();
      }
    }, 150);
  });
}

/** Add a tag to the active chat. */
export async function addTagToChat(tag: string): Promise<void> {
  if (!chatState.activeSessionId || !chatState.activeChat) {
    notifications.error("No active conversation");
    return;
  }

  const normalized = tag.trim().toLowerCase();
  if (!normalized || normalized.length > 50) {
    notifications.error("Tag must be 1-50 characters");
    return;
  }

  if (!/^[a-z0-9_-]+$/.test(normalized)) {
    notifications.error("Tags can only contain letters, numbers, hyphens, and underscores");
    return;
  }

  const currentTags = getTags(chatState.activeChat);
  if (currentTags.includes(normalized)) {
    notifications.error("Tag already exists");
    return;
  }

  if (currentTags.length >= 20) {
    notifications.error("Maximum 20 tags per conversation");
    return;
  }

  const newTags = [...currentTags, normalized];
  const patchResult = await api.patchConversation(chatState.activeSessionId, {
    metadata: { ...chatState.activeChat.metadata, tags: newTags },
  });

  if (!patchResult.ok) {
    notifications.error(patchResult.error ?? "Failed to add tag");
    return;
  }

  // Update local state directly (backend GET doesn't return metadata correctly)
  chatState.activeChat.metadata = { ...chatState.activeChat.metadata, tags: newTags };
  updateSessionInList(chatState.activeChat);
  updateHeaderTags(chatState.activeChat);
  renderSidebar();
  notifications.info("Tag added");
}

/** Remove a tag from the active chat. */
export async function removeTagFromChat(tag: string): Promise<void> {
  if (!chatState.activeSessionId || !chatState.activeChat) return;

  const currentTags = getTags(chatState.activeChat);
  const newTags = currentTags.filter((t) => t !== tag);

  const result = await api.patchConversation(chatState.activeSessionId, {
    metadata: { ...chatState.activeChat.metadata, tags: newTags },
  });

  if (!result.ok) {
    notifications.error(result.error ?? "Failed to remove tag");
    return;
  }

  // Update local state directly (backend GET doesn't return metadata correctly)
  chatState.activeChat.metadata = { ...chatState.activeChat.metadata, tags: newTags };
  updateSessionInList(chatState.activeChat);
  updateHeaderTags(chatState.activeChat);
  renderSidebar();
}

/** Update the tags display in the chat header. */
export function updateHeaderTags(chat: Conversation): void {
  const tagsContainer = getChatDom().transcriptContainer.querySelector<HTMLElement>(".transcript-tags");
  if (!tagsContainer) return;

  // Clear existing tags
  tagsContainer.innerHTML = "";

  const tags = getTags(chat);
  for (const tag of tags) {
    const chip = document.createElement("span");
    chip.className = "tag-pill";
    chip.dataset["tag"] = tag;
    chip.innerHTML = `
      <span>${escapeHtml(tag)}</span>
      <button class="tag-remove" data-tag="${escapeHtml(tag)}" title="Remove tag">
        <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
      </button>
    `;
    tagsContainer.appendChild(chip);
  }

  // Add tag button (if under 5 tags)
  if (tags.length < 5) {
    const addBtn = document.createElement("button");
    addBtn.className = "add-tag-btn";
    addBtn.innerHTML = `${PLUS_ICON} Tag`;
    addBtn.dataset["action"] = "add-tag";
    tagsContainer.appendChild(addBtn);
  }
}

/** Update a session in the sessions list. */
export function updateSessionInList(updated: Conversation): void {
  const idx = chatState.sessions.findIndex((s) => s.id === updated.id);
  if (idx !== -1) {
    chatState.sessions[idx] = updated;
  }
}
