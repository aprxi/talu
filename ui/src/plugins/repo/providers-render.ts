/** Render provider configuration as a compact list of enabled providers. */

import { CLOSE_ICON } from "../../icons.ts";
import { el } from "../../render/helpers.ts";
import { renderEmptyState } from "../../render/common.ts";
import { getRepoDom } from "./dom.ts";
import { repoState } from "./state.ts";
import { addProvider, removeProvider, updateProvider, testProvider } from "./providers-data.ts";
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

  if (enabled.length === 0) {
    list.appendChild(renderEmptyState("No providers configured. Add one above."));
    return;
  }

  for (const p of enabled) {
    list.appendChild(buildProviderRow(p));
  }
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
  return row;
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
