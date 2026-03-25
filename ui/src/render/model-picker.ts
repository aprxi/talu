/**
 * Custom dropdown for model selection with integrated settings access.
 *
 * UX model:
 *  - Click trigger label → open sampling settings for current model
 *  - Click chevron → open model list dropdown
 *  - Click anywhere outside → close dropdown and settings panels
 *  - Sliders icon on model row → select model + open settings (always open, never toggle)
 */

import { navigate } from "../kernel/system/router.ts";

const NAV_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>`;
const SLIDERS_ICON_SM = `<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/><line x1="18" x2="22" y1="16" y2="16"/></svg>`;
const SLIDERS_ICON_LG = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/><line x1="18" x2="22" y1="16" y2="16"/></svg>`;
const CHEVRON = `<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>`;

export interface PickerOptions {
  /** Called when user wants to open sampling settings for a model. */
  onSettings?: (modelId: string) => void;
  /** Called when user clicks outside — use to close settings panels. */
  onDismiss?: () => void;
}

interface PickerState {
  trigger: HTMLElement;
  dropdown: HTMLElement;
  open: boolean;
  options: PickerOptions;
  onOutsideClick: (e: MouseEvent) => void;
}

const pickers = new WeakMap<HTMLSelectElement, PickerState>();

/** Create the custom dropdown UI around a hidden <select>. */
export function initModelPicker(sel: HTMLSelectElement, opts?: PickerOptions): void {
  if (pickers.has(sel)) return;

  sel.style.display = "none";

  // Trigger: [sliders] [model name] [chevron]
  const trigger = document.createElement("div");
  trigger.className = "model-picker-trigger";

  const settingsBtn = document.createElement("span");
  settingsBtn.className = "model-picker-trigger-settings";
  settingsBtn.innerHTML = SLIDERS_ICON_LG;
  settingsBtn.title = "Model settings";
  trigger.appendChild(settingsBtn);

  const label = document.createElement("span");
  label.className = "model-picker-trigger-label";
  label.textContent = sel.options[sel.selectedIndex]?.textContent ?? "Select model";
  trigger.appendChild(label);

  const chevron = document.createElement("span");
  chevron.className = "model-picker-chevron";
  chevron.innerHTML = CHEVRON;
  trigger.appendChild(chevron);

  sel.insertAdjacentElement("afterend", trigger);

  // Dropdown panel.
  const dropdown = document.createElement("div");
  dropdown.className = "model-picker-dropdown hidden";
  trigger.insertAdjacentElement("afterend", dropdown);

  const state: PickerState = {
    trigger,
    dropdown,
    open: false,
    options: opts ?? {},
    onOutsideClick: (e: MouseEvent) => {
      if (!trigger.contains(e.target as Node) && !dropdown.contains(e.target as Node)) {
        closeDropdown(state);
      }
    },
  };
  pickers.set(sel, state);

  // All handlers use mousedown for consistency with outside-click handler.

  // Sliders icon → toggle model settings.
  settingsBtn.addEventListener("mousedown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (state.open) closeDropdown(state);
    const modelId = sel.value;
    if (modelId && modelId !== "__add_model__" && state.options.onSettings) {
      state.options.onSettings(modelId);
    }
  });

  // Label or chevron → toggle dropdown (dismiss settings if open).
  const toggleDropdown = (e: Event) => {
    e.preventDefault();
    e.stopPropagation();
    if (state.open) {
      closeDropdown(state);
    } else {
      state.options.onDismiss?.();
      openDropdown(sel, state);
    }
  };
  label.addEventListener("mousedown", toggleDropdown);
  chevron.addEventListener("mousedown", toggleDropdown);

  // Sync from initial state.
  syncModelPicker(sel);
}

/** Rebuild the dropdown items from the current <select> options. */
export function syncModelPicker(sel: HTMLSelectElement): void {
  const state = pickers.get(sel);
  if (!state) return;

  const { dropdown } = state;
  dropdown.innerHTML = "";

  // Update trigger label.
  const selectedOpt = sel.options[sel.selectedIndex];
  const label = state.trigger.querySelector(".model-picker-trigger-label");
  if (label) label.textContent = selectedOpt?.textContent ?? "Select model";

  // Build items from select options/optgroups.
  for (const child of Array.from(sel.children)) {
    if (child instanceof HTMLOptGroupElement) {
      if (child.label.startsWith("──")) {
        const sep = document.createElement("div");
        sep.className = "model-picker-sep";
        dropdown.appendChild(sep);
        for (const opt of Array.from(child.querySelectorAll("option"))) {
          dropdown.appendChild(makeItem(sel, state, opt as HTMLOptionElement));
        }
      } else {
        const groupLabel = document.createElement("div");
        groupLabel.className = "model-picker-group";
        groupLabel.textContent = child.label;
        dropdown.appendChild(groupLabel);
        for (const opt of Array.from(child.querySelectorAll("option"))) {
          dropdown.appendChild(makeItem(sel, state, opt as HTMLOptionElement));
        }
      }
    } else if (child instanceof HTMLOptionElement) {
      dropdown.appendChild(makeItem(sel, state, child));
    }
  }
}

function makeItem(sel: HTMLSelectElement, state: PickerState, opt: HTMLOptionElement): HTMLElement {
  const item = document.createElement("div");

  if (opt.value === "__add_model__") {
    item.className = "model-picker-item model-picker-action";
    item.innerHTML = `<span>Add model</span>${NAV_ICON}`;
    item.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      closeDropdown(state);
      navigate({ mode: "routing", sub: null, resource: null });
    });
    return item;
  }

  item.className = "model-picker-item";
  if (opt.disabled) item.classList.add("disabled");
  if (opt.value === sel.value) item.classList.add("active");
  item.dataset.value = opt.value;

  if (opt.disabled) {
    item.textContent = opt.textContent;
    return item;
  }

  // Layout: [name] [settings icon]
  const nameLabel = document.createElement("span");
  nameLabel.className = "model-picker-label";
  nameLabel.textContent = opt.textContent;
  item.appendChild(nameLabel);

  if (state.options.onSettings) {
    const settingsBtn = document.createElement("span");
    settingsBtn.className = "model-picker-item-settings";
    settingsBtn.innerHTML = SLIDERS_ICON_SM;
    settingsBtn.title = "Model settings";
    const onSettings = state.options.onSettings;
    settingsBtn.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      selectModel(sel, state, opt);
      closeDropdown(state);
      onSettings(opt.value);
    });
    item.appendChild(settingsBtn);
  }

  // Click on name → select model + close dropdown.
  nameLabel.addEventListener("mousedown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    selectModel(sel, state, opt);
    closeDropdown(state);
  });

  return item;
}

function selectModel(sel: HTMLSelectElement, state: PickerState, opt: HTMLOptionElement): void {
  sel.value = opt.value;
  sel.dispatchEvent(new Event("change", { bubbles: true }));
  const label = state.trigger.querySelector(".model-picker-trigger-label");
  if (label) label.textContent = opt.textContent;
  for (const el of state.dropdown.querySelectorAll(".model-picker-item.active")) {
    el.classList.remove("active");
  }
  const item = state.dropdown.querySelector(`[data-value="${opt.value}"]`);
  item?.classList.add("active");
}

function openDropdown(sel: HTMLSelectElement, state: PickerState): void {
  syncModelPicker(sel);
  state.dropdown.classList.remove("hidden");
  state.trigger.classList.add("open");
  state.open = true;
  document.addEventListener("mousedown", state.onOutsideClick);
}

function closeDropdown(state: PickerState): void {
  state.dropdown.classList.add("hidden");
  state.trigger.classList.remove("open");
  state.open = false;
  document.removeEventListener("mousedown", state.onOutsideClick);
}
