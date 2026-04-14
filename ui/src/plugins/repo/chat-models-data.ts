/** Chat models data operations: load/save from preferences, add/remove/reorder. */

import { api, events, notifications } from "./deps.ts";
import { repoState, inferFamilyKey } from "./state.ts";
import { renderChatModels } from "./chat-models-render.ts";
import { preferences } from "../../kernel/system/preferences.ts";

function emitChanged(): void {
  // Build family→defaultVariant mapping with variant lists so the model
  // selector shows family names and the advanced section offers quant choice.
  const familyMap = new Map<string, { defaultVariant: string; variants: { id: string; label: string; size_bytes: number }[] }>();
  for (const id of repoState.chatModels) {
    if (id.includes("::")) continue; // remote models stay as-is
    const model = repoState.models.find((m) => m.id === id);
    const key = model ? inferFamilyKey(model) : id;
    if (!familyMap.has(key)) {
      familyMap.set(key, { defaultVariant: id, variants: [] });
    }
    // Use quant_scheme for label (e.g. "TQ4"), fall back to suffix.
    const suffix = key !== id ? id.slice(key.length + 1) : (id.split("/").pop() ?? id);
    const label = model?.quant_scheme || suffix;
    familyMap.get(key)!.variants.push({ id, label, size_bytes: model?.size_bytes ?? 0 });
  }
  events.emit("repo.chatModels.changed", {
    models: [...repoState.chatModels],
    families: [...familyMap.entries()].map(([familyId, data]) => ({
      familyId,
      defaultVariant: data.defaultVariant,
      variants: data.variants,
    })),
  });
}

export async function loadChatModels(): Promise<void> {
  const stored = preferences.get<string[]>("talu.repo", "pinned_models");
  if (stored && Array.isArray(stored)) {
    repoState.chatModels = stored.filter((x): x is string => typeof x === "string");
  }
  emitChanged();
}

function saveChatModels(): void {
  preferences.set("talu.repo", "pinned_models", repoState.chatModels);
  emitChanged();
}

export async function addChatModel(modelId: string): Promise<void> {
  if (repoState.chatModels.includes(modelId)) return;
  repoState.chatModels.push(modelId);
  renderChatModels();
  await saveChatModels();
}

export async function removeChatModel(modelId: string): Promise<void> {
  const idx = repoState.chatModels.indexOf(modelId);
  if (idx < 0) return;
  repoState.chatModels.splice(idx, 1);
  renderChatModels();
  await saveChatModels();
}

/** Remove all models belonging to a family from the chat models list. */
export async function removeChatModelFamily(familyId: string): Promise<void> {
  const before = repoState.chatModels.length;
  repoState.chatModels = repoState.chatModels.filter((id) => {
    const model = repoState.models.find((m) => m.id === id);
    if (!model) return id !== familyId; // keep unknowns unless exact match
    return inferFamilyKey(model) !== familyId;
  });
  if (repoState.chatModels.length === before) return;
  renderChatModels();
  await saveChatModels();
}

/** Reorder a family block to a new position among families. */
export async function reorderFamily(familyId: string, newFamilyIndex: number): Promise<void> {
  // Build ordered family list to find positions
  const families = buildFamilyOrder();
  const oldIdx = families.findIndex((f) => f.familyId === familyId);
  if (oldIdx < 0 || oldIdx === newFamilyIndex) return;
  const clamped = Math.max(0, Math.min(newFamilyIndex, families.length - 1));

  // Move the family
  const [moved] = families.splice(oldIdx, 1);
  families.splice(clamped, 0, moved!);

  // Rebuild flat list from reordered families
  repoState.chatModels = families.flatMap((f) => f.modelIds);
  renderChatModels();
  await saveChatModels();
}

export async function browseProviderModels(name: string): Promise<{ id: string }[]> {
  const cached = repoState.browseModels.get(name);
  if (cached) return cached;

  const res = await api.listProviderModels(name);
  if (res.ok && res.data?.models) {
    const models = res.data.models.map((m) => ({ id: m.id }));
    repoState.browseModels.set(name, models);
    return models;
  }
  if (res.error) notifications.error(`Failed to list models for ${name}: ${res.error}`);
  return [];
}

/**
 * Sync pinned models into the chat models list.
 * Groups pinned models by family (siblings adjacent).
 * Keeps any remote provider entries (contain "::").
 */
export async function syncPinnedToChatModels(): Promise<void> {
  const remoteModels = repoState.chatModels.filter((id) => id.includes("::"));

  // Group pinned models by family so siblings are adjacent.
  const familyMap = new Map<string, string[]>();
  for (const m of repoState.models) {
    if (!m.pinned) continue;
    const key = inferFamilyKey(m);
    if (!familyMap.has(key)) familyMap.set(key, []);
    familyMap.get(key)!.push(m.id);
  }
  const pinnedGrouped = [...familyMap.values()].flat();

  const merged = [...pinnedGrouped, ...remoteModels];
  if (
    merged.length === repoState.chatModels.length &&
    merged.every((id, i) => repoState.chatModels[i] === id)
  ) {
    // Still emit so families are rebuilt with freshly-loaded model data.
    emitChanged();
    return;
  }
  repoState.chatModels = merged;
  renderChatModels();
  await saveChatModels();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface FamilyBlock { familyId: string; modelIds: string[] }

/** Build ordered family blocks from the current chatModels list. */
function buildFamilyOrder(): FamilyBlock[] {
  const families: FamilyBlock[] = [];
  const seen = new Set<string>();

  for (const id of repoState.chatModels) {
    const model = repoState.models.find((m) => m.id === id);
    const familyId = model ? inferFamilyKey(model) : id;
    if (seen.has(familyId)) {
      // Add to existing family block
      const block = families.find((f) => f.familyId === familyId);
      if (block) block.modelIds.push(id);
      continue;
    }
    seen.add(familyId);
    families.push({ familyId, modelIds: [id] });
  }
  return families;
}

export { buildFamilyOrder };
