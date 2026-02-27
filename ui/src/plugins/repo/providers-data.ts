/** Data loading and mutation for the providers tab. */

import { api, notifications, events } from "./deps.ts";
import { repoState } from "./state.ts";
import { renderProviders } from "./providers-render.ts";

export async function loadProviders(): Promise<void> {
  const res = await api.listProviders();
  if (res.ok && res.data) {
    repoState.providers = res.data.providers ?? [];
  } else {
    repoState.providers = [];
    if (res.error) notifications.error(`Failed to load providers: ${res.error}`);
  }
  renderProviders();
}

export async function addProvider(name: string): Promise<void> {
  await updateProvider(name, { enabled: true });
}

export async function removeProvider(name: string): Promise<void> {
  await updateProvider(name, { enabled: false });
}

/** Returns { ok, model_count?, error? }. */
export async function testProvider(name: string): Promise<{ ok: boolean; error?: string }> {
  const res = await api.testProvider(name);
  if (res.ok && res.data) {
    return res.data;
  }
  return { ok: false, error: res.error ?? "Request failed" };
}

export async function updateProvider(
  name: string,
  patch: { enabled?: boolean; api_key?: string | null; base_url?: string | null },
): Promise<void> {
  const res = await api.updateProvider(name, patch);
  if (res.ok && res.data) {
    repoState.providers = res.data.providers ?? [];
    notifications.success(`Updated ${name}`);
    events.emit("repo.providers.changed", {});
  } else {
    notifications.error(`Failed to update ${name}: ${res.error ?? "unknown"}`);
  }
  renderProviders();
}
