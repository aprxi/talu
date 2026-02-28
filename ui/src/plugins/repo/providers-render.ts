/** Render provider configuration as a compact list of enabled providers. */

import { CLOSE_ICON } from "../../icons.ts";
import { el } from "../../render/helpers.ts";
import { renderEmptyState } from "../../render/common.ts";
import { getRepoDom } from "./dom.ts";
import { repoState } from "./state.ts";
import { addProvider, removeProvider, updateProvider, testProvider } from "./providers-data.ts";
import { addChatModel, browseProviderModels, browseLocalModels } from "./chat-models-data.ts";
import { renderChatModels } from "./chat-models-render.ts";
import { syncRepoTabs, updateRepoToolbar } from "./render.ts";
import { loadModels } from "./data.ts";
import type { ProviderEntry } from "../../types.ts";

// ---------------------------------------------------------------------------
// Render (safe to call repeatedly — rebuilds DOM, no listener attachment)
// ---------------------------------------------------------------------------

export function renderProviders(): void {
  const dom = getRepoDom();
  const list = dom.providersList;
  list.innerHTML = "";

  const { providers } = repoState;
  const enabled = providers.filter((p) => p.enabled);
  const disabled = providers.filter((p) => !p.enabled);

  // Populate "Add provider" dropdown with disabled providers.
  const addSelect = dom.addProviderSelect;
  addSelect.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "+ Add provider\u2026";
  placeholder.selected = true;
  placeholder.disabled = true;
  addSelect.appendChild(placeholder);

  for (const p of disabled) {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = p.name;
    addSelect.appendChild(opt);
  }
  addSelect.classList.toggle("hidden", disabled.length === 0);

  // Always show "local" as the first provider.
  list.appendChild(buildLocalRow());

  for (const p of enabled) {
    list.appendChild(buildProviderRow(p));
  }

  if (enabled.length === 0) {
    // Show a hint if no remote providers are configured.
    const hint = el("div", "repo-provider-hint", "Add a remote provider above to access cloud models.");
    list.appendChild(hint);
  }
}

// ---------------------------------------------------------------------------
// "Local" pseudo-provider row
// ---------------------------------------------------------------------------

function buildLocalRow(): HTMLElement {
  const row = el("div", "repo-provider-row");
  row.dataset["provider"] = "local";

  const top = el("div", "repo-provider-row-top");
  const nameEl = el("span", "repo-provider-name", "local");
  top.appendChild(nameEl);

  const actions = el("div", "repo-provider-row-actions");

  const browseBtn = el("button", "btn btn-ghost btn-sm", "Browse");
  browseBtn.dataset["action"] = "browse";
  actions.appendChild(browseBtn);

  const manageBtn = el("button", "btn btn-ghost btn-sm", "Manage");
  manageBtn.dataset["action"] = "manage-local";
  actions.appendChild(manageBtn);

  top.appendChild(actions);
  row.appendChild(top);

  const meta = el("div", "repo-provider-row-meta", "Cached managed models");
  row.appendChild(meta);

  // Browse expansion (hidden).
  const browseList = el("div", "repo-browse-list hidden");
  browseList.dataset["browseFor"] = "local";
  row.appendChild(browseList);

  return row;
}

// ---------------------------------------------------------------------------
// Row builder
// ---------------------------------------------------------------------------

function buildProviderRow(p: ProviderEntry): HTMLElement {
  const row = el("div", "repo-provider-row");
  row.dataset["provider"] = p.name;

  // Top line: name + actions
  const top = el("div", "repo-provider-row-top");
  const nameEl = el("span", "repo-provider-name", p.name);
  top.appendChild(nameEl);

  const actions = el("div", "repo-provider-row-actions");

  // Key status badge (only shown if the provider uses an API key).
  if (p.api_key_env) {
    const badge = el("span", "repo-provider-key-badge");
    if (p.has_api_key) {
      badge.classList.add("key-set");
      badge.textContent = "Key set";
    } else {
      badge.classList.add("key-missing");
      badge.textContent = "No key";
    }
    actions.appendChild(badge);
  }

  const browseBtn = el("button", "btn btn-ghost btn-sm", "Browse");
  browseBtn.dataset["action"] = "browse";
  actions.appendChild(browseBtn);

  const testBtn = el("button", "btn btn-ghost btn-sm", "Test");
  testBtn.dataset["action"] = "test";
  actions.appendChild(testBtn);

  const editBtn = el("button", "btn btn-ghost btn-sm", "Edit");
  editBtn.dataset["action"] = "expand";
  actions.appendChild(editBtn);

  const removeBtn = el("button", "btn btn-ghost btn-sm repo-provider-remove-btn");
  removeBtn.innerHTML = CLOSE_ICON;
  removeBtn.title = "Remove provider";
  removeBtn.dataset["action"] = "remove";
  actions.appendChild(removeBtn);

  top.appendChild(actions);
  row.appendChild(top);

  // Meta line: effective endpoint.
  const meta = el("div", "repo-provider-row-meta", p.effective_endpoint);
  row.appendChild(meta);

  // Collapsible form (hidden by default).
  const form = el("div", "repo-provider-form hidden");

  // API key input.
  const keyGroup = el("div", "repo-provider-field");
  const keyLabel = el("label", "", "API Key");
  const keyInput = document.createElement("input");
  keyInput.type = "password";
  keyInput.placeholder = p.has_api_key ? "********" : "Enter API key\u2026";
  keyInput.dataset["field"] = "api_key";
  keyInput.className = "repo-provider-input";
  keyGroup.appendChild(keyLabel);
  keyGroup.appendChild(keyInput);
  form.appendChild(keyGroup);

  // Base URL input.
  const urlGroup = el("div", "repo-provider-field");
  const urlLabel = el("label", "", "Base URL");
  const urlInput = document.createElement("input");
  urlInput.type = "text";
  urlInput.placeholder = p.default_endpoint;
  urlInput.value = p.base_url_override ?? "";
  urlInput.dataset["field"] = "base_url";
  urlInput.className = "repo-provider-input";
  urlGroup.appendChild(urlLabel);
  urlGroup.appendChild(urlInput);
  form.appendChild(urlGroup);

  // Save button.
  const formActions = el("div", "repo-provider-form-actions");
  const saveBtn = el("button", "btn btn-primary btn-sm", "Save");
  saveBtn.dataset["action"] = "save";
  formActions.appendChild(saveBtn);
  form.appendChild(formActions);

  row.appendChild(form);

  // Browse expansion (hidden).
  const browseList = el("div", "repo-browse-list hidden");
  browseList.dataset["browseFor"] = p.name;
  row.appendChild(browseList);

  return row;
}

// ---------------------------------------------------------------------------
// Browse model list rendering (inline within provider row)
// ---------------------------------------------------------------------------

function renderBrowseModels(container: HTMLElement, providerName: string, models: { id: string }[]): void {
  container.innerHTML = "";
  if (models.length === 0) {
    container.appendChild(el("div", "repo-browse-empty", "No models found."));
    return;
  }

  for (const m of models) {
    const fullId = providerName === "local" ? m.id : `${providerName}::${m.id}`;
    const item = el("div", "repo-browse-item");
    item.dataset["modelId"] = fullId;

    const nameEl = el("span", "repo-browse-name", m.id);
    item.appendChild(nameEl);

    if (repoState.chatModels.includes(fullId)) {
      const addedEl = el("span", "repo-browse-added", "\u2713 Added");
      item.appendChild(addedEl);
    } else {
      const addBtn = el("button", "btn btn-ghost btn-sm", "+ Add");
      addBtn.dataset["action"] = "add-model";
      item.appendChild(addBtn);
    }

    container.appendChild(item);
  }
}

// ---------------------------------------------------------------------------
// Events (called once from index.ts — delegates on the persistent container)
// ---------------------------------------------------------------------------

export function wireProviderEvents(container: HTMLElement): void {
  const dom = getRepoDom();

  // "Add provider" dropdown.
  dom.addProviderSelect.addEventListener("change", () => {
    const name = dom.addProviderSelect.value;
    if (!name) return;
    addProvider(name);
    dom.addProviderSelect.value = "";
  });

  // Delegated click events on the provider list.
  container.addEventListener("click", (e) => {
    const target = e.target as HTMLElement;
    const actionEl = target.closest<HTMLElement>("[data-action]");
    if (!actionEl) return;
    const action = actionEl.dataset["action"];

    const row = target.closest<HTMLElement>("[data-provider]");
    if (!row) return;
    const name = row.dataset["provider"]!;

    if (action === "expand") {
      const form = row.querySelector<HTMLElement>(".repo-provider-form");
      if (form) {
        const isHidden = form.classList.contains("hidden");
        form.classList.toggle("hidden");
        actionEl.textContent = isHidden ? "Cancel" : "Edit";
      }
      return;
    }

    if (action === "manage-local") {
      repoState.subPage = "manage-local";
      repoState.manageLocalTab = "local";
      repoState.selectedIds.clear();
      repoState.searchQuery = "";
      const dom = getRepoDom();
      dom.search.value = "";
      dom.searchClear.classList.add("hidden");
      syncRepoTabs();
      updateRepoToolbar();
      loadModels();
      return;
    }

    if (action === "browse") {
      const browseList = row.querySelector<HTMLElement>(".repo-browse-list");
      if (!browseList) return;
      const isHidden = browseList.classList.contains("hidden");
      if (!isHidden) {
        browseList.classList.add("hidden");
        actionEl.textContent = "Browse";
        return;
      }
      browseList.classList.remove("hidden");
      actionEl.textContent = "Close";
      browseList.innerHTML = "";
      browseList.appendChild(el("div", "repo-browse-loading", "Loading\u2026"));

      const fetchModels = name === "local" ? browseLocalModels() : browseProviderModels(name);
      fetchModels.then((models) => {
        renderBrowseModels(browseList, name, models);
      });
      return;
    }

    if (action === "add-model") {
      const browseItem = target.closest<HTMLElement>("[data-model-id]");
      if (!browseItem) return;
      const modelId = browseItem.dataset["modelId"]!;
      addChatModel(modelId);
      // Replace button with "Added" label.
      actionEl.remove();
      const addedEl = el("span", "repo-browse-added", "\u2713 Added");
      browseItem.appendChild(addedEl);
      return;
    }

    if (action === "test") {
      actionEl.textContent = "\u2026";
      actionEl.setAttribute("disabled", "");
      testProvider(name).then((res) => {
        actionEl.removeAttribute("disabled");
        if (res.ok) {
          actionEl.textContent = "OK";
          actionEl.classList.add("test-ok");
        } else {
          actionEl.textContent = "Fail";
          actionEl.classList.add("test-fail");
          actionEl.title = res.error ?? "Connection failed";
        }
        setTimeout(() => {
          actionEl.textContent = "Test";
          actionEl.classList.remove("test-ok", "test-fail");
          actionEl.title = "";
        }, 3000);
      });
      return;
    }

    if (action === "remove") {
      removeProvider(name);
      return;
    }

    if (action === "save") {
      const keyInput = row.querySelector<HTMLInputElement>("[data-field='api_key']");
      const urlInput = row.querySelector<HTMLInputElement>("[data-field='base_url']");
      const apiKey = keyInput?.value?.trim() || null;
      const baseUrl = urlInput?.value?.trim() || null;
      updateProvider(name, { enabled: true, api_key: apiKey, base_url: baseUrl });
      return;
    }
  });
}
