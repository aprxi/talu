/** Render the Chat Models section in the providers tab. */

import { CLOSE_ICON, GRIP_ICON } from "../../icons.ts";
import { el } from "../../render/helpers.ts";
import { renderEmptyState } from "../../render/common.ts";
import { getRepoDom } from "./dom.ts";
import { repoState } from "./state.ts";
import type { CachedModel } from "./state.ts";
import { removeChatModelFamily, reorderFamily, buildFamilyOrder } from "./chat-models-data.ts";
import { formatBytes } from "./render.ts";
import { events } from "./deps.ts";
import { unpinModel } from "./data.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ModelFamily {
  familyId: string;
  displayName: string;
  variants: CachedModel[];
  isRemote: boolean;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

export function renderChatModels(): void {
  const list = getRepoDom().chatModelsList;
  list.innerHTML = "";

  if (repoState.chatModels.length === 0) {
    list.appendChild(renderEmptyState("No models pinned. Pin models in Manage to add them here."));
    return;
  }

  const families = buildFamilies();
  for (const family of families) {
    list.appendChild(buildFamilyRow(family));
  }
}

/** Group chatModels into families using source_model_id. */
function buildFamilies(): ModelFamily[] {
  const blocks = buildFamilyOrder();
  return blocks.map((block) => {
    const firstId = block.modelIds[0]!;
    const sep = firstId.indexOf("::");
    const isRemote = sep >= 0;

    if (isRemote) {
      return {
        familyId: block.familyId,
        displayName: firstId.substring(sep + 2),
        variants: [],
        isRemote: true,
      };
    }

    const variants = block.modelIds
      .map((id) => repoState.models.find((m) => m.id === id))
      .filter((m): m is CachedModel => m !== undefined);

    // Display name: use the source_model_id (family key) for a cleaner name
    const displayName = block.familyId;

    return { familyId: block.familyId, displayName, variants, isRemote };
  });
}

// ---------------------------------------------------------------------------
// Family row builder
// ---------------------------------------------------------------------------

function buildFamilyRow(family: ModelFamily): HTMLElement {
  const row = el("div", "repo-chat-model-family");
  row.dataset["familyId"] = family.familyId;

  // Top line: grip + name + remove
  const top = el("div", "repo-chat-model-top");

  const grip = el("div", "repo-chat-model-grip");
  grip.innerHTML = GRIP_ICON;
  grip.title = "Drag to reorder";
  top.appendChild(grip);

  if (family.isRemote) {
    // Remote: show provider badge
    const providerName = family.familyId.substring(0, family.familyId.indexOf("::"));
    const badge = el("span", "repo-chat-model-badge", providerName);
    top.appendChild(badge);
  }

  const nameEl = el("span", "repo-chat-model-name", family.displayName);
  nameEl.title = family.familyId;
  top.appendChild(nameEl);

  const removeBtn = el("button", "btn btn-ghost repo-chat-model-remove");
  removeBtn.innerHTML = CLOSE_ICON;
  removeBtn.title = "Remove";
  removeBtn.dataset["action"] = "cm-remove-family";
  top.appendChild(removeBtn);

  row.appendChild(top);

  // Variant pills (local families with variants)
  if (!family.isRemote && family.variants.length > 0) {
    const variantsRow = el("div", "repo-chat-model-variants");

    for (const variant of family.variants) {
      const label = formatVariantLabel(variant);
      const pill = el("button", "repo-chat-model-variant", label);
      pill.dataset["action"] = "cm-select-variant";
      pill.dataset["variantId"] = variant.id;
      pill.title = variant.id;
      variantsRow.appendChild(pill);
    }

    row.appendChild(variantsRow);
  }

  return row;
}

function formatVariantLabel(model: CachedModel): string {
  const parts: string[] = [];
  if (model.quant_scheme) {
    parts.push(model.quant_scheme);
  }
  if (model.size_bytes > 0) {
    parts.push(formatBytes(model.size_bytes));
  }
  return parts.length > 0 ? parts.join(" · ") : model.id;
}

// ---------------------------------------------------------------------------
// Events: click delegation + pointer-event drag-to-reorder
// ---------------------------------------------------------------------------

export function wireChatModelEvents(container: HTMLElement): void {
  container.addEventListener("click", (e) => {
    const target = e.target as HTMLElement;
    const actionEl = target.closest<HTMLElement>("[data-action]");
    if (!actionEl) return;
    const action = actionEl.dataset["action"];

    if (action === "cm-remove-family") {
      const familyRow = target.closest<HTMLElement>("[data-family-id]");
      if (!familyRow) return;
      const familyId = familyRow.dataset["familyId"]!;
      // Unpin the family on the server, then remove from chat models
      unpinModel(familyId);
      removeChatModelFamily(familyId);
      return;
    }

    if (action === "cm-select-variant") {
      const variantId = actionEl.dataset["variantId"];
      if (variantId) {
        events.emit("repo.selectModel", { modelId: variantId });
        events.emit("repo.openChat", {});
      }
      return;
    }
  });

  setupDragReorder(container);
}

// ---------------------------------------------------------------------------
// Drag-to-reorder (family-level)
// ---------------------------------------------------------------------------

function setupDragReorder(container: HTMLElement): void {
  let dragItem: HTMLElement | null = null;
  let startY = 0;
  let itemHeight = 0;
  let originalIndex = 0;
  let currentIndex = 0;
  let items: HTMLElement[] = [];

  container.addEventListener("pointerdown", (e) => {
    const grip = (e.target as HTMLElement).closest<HTMLElement>(".repo-chat-model-grip");
    if (!grip) return;
    const row = grip.closest<HTMLElement>(".repo-chat-model-family");
    if (!row) return;

    e.preventDefault();
    grip.setPointerCapture(e.pointerId);

    dragItem = row;
    startY = e.clientY;

    items = Array.from(container.querySelectorAll<HTMLElement>(".repo-chat-model-family"));
    originalIndex = items.indexOf(row);
    currentIndex = originalIndex;
    itemHeight = row.getBoundingClientRect().height;

    row.classList.add("dragging");
    for (const item of items) {
      if (item !== row) item.classList.add("displaced");
    }
  });

  container.addEventListener("pointermove", (e) => {
    if (!dragItem) return;
    e.preventDefault();

    const deltaY = e.clientY - startY;
    dragItem.style.transform = `translateY(${deltaY}px)`;

    const rawIndex = originalIndex + Math.round(deltaY / itemHeight);
    const newIndex = Math.max(0, Math.min(rawIndex, items.length - 1));

    if (newIndex !== currentIndex) {
      currentIndex = newIndex;
      for (let i = 0; i < items.length; i++) {
        if (items[i] === dragItem) continue;
        let shift = 0;
        if (originalIndex < currentIndex) {
          if (i > originalIndex && i <= currentIndex) shift = -itemHeight;
        } else {
          if (i >= currentIndex && i < originalIndex) shift = itemHeight;
        }
        items[i]!.style.transform = shift ? `translateY(${shift}px)` : "";
      }
    }
  });

  const endDrag = (e: PointerEvent) => {
    if (!dragItem) return;
    e.preventDefault();

    dragItem.classList.remove("dragging");
    dragItem.style.transform = "";
    for (const item of items) {
      item.classList.remove("displaced");
      item.style.transform = "";
    }

    const familyId = dragItem.dataset["familyId"]!;
    const finalIndex = currentIndex;
    dragItem = null;
    items = [];

    if (finalIndex !== originalIndex) {
      reorderFamily(familyId, finalIndex);
    }
  };

  container.addEventListener("pointerup", endDrag);
  container.addEventListener("pointercancel", endDrag);
}
