/** Chat models data operations: load/save from KV, add/remove/reorder. */

import { api, events, notifications } from "./deps.ts";
import { repoState } from "./state.ts";
import { renderChatModels } from "./chat-models-render.ts";

const KV_NS = "chat_models";
const KV_KEY = "models";

function emitChanged(): void {
  events.emit("repo.chatModels.changed", { models: [...repoState.chatModels] });
}

export async function loadChatModels(): Promise<void> {
  const res = await api.kvGet(KV_NS, KV_KEY);
  if (res.ok && res.data?.value) {
    try {
      const parsed = JSON.parse(res.data.value);
      if (Array.isArray(parsed)) {
        repoState.chatModels = parsed.filter((x): x is string => typeof x === "string");
      }
    } catch { /* ignore parse errors */ }
  }
  emitChanged();
}

async function saveChatModels(): Promise<void> {
  await api.kvPut(KV_NS, KV_KEY, JSON.stringify(repoState.chatModels));
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

export async function moveChatModel(modelId: string, direction: "up" | "down"): Promise<void> {
  const idx = repoState.chatModels.indexOf(modelId);
  if (idx < 0) return;
  const target = direction === "up" ? idx - 1 : idx + 1;
  if (target < 0 || target >= repoState.chatModels.length) return;
  [repoState.chatModels[idx], repoState.chatModels[target]] =
    [repoState.chatModels[target]!, repoState.chatModels[idx]!];
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

export async function browseLocalModels(): Promise<{ id: string }[]> {
  const cached = repoState.browseModels.get("local");
  if (cached) return cached;

  const res = await api.listRepoModels();
  if (res.ok && res.data?.models) {
    const models = res.data.models
      .filter((m) => m.source === "managed")
      .map((m) => ({ id: m.id }));
    repoState.browseModels.set("local", models);
    return models;
  }
  return [];
}
