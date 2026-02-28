/**
 * Theme editor — custom theme CRUD, color editor with live preview,
 * import/export as JSON. Editor renders in the app panel overlay.
 *
 * Mode slot system: dark/light toggle maps to two user-configurable
 * theme slots stored in localStorage. Settings shows two selects
 * (one per mode). The editor panel has Save/Reset + assign-to-mode.
 */

import { COLOR_SCHEME_TOKENS, DARK_SCHEME_ID, LIGHT_SCHEME_ID } from "../../styles/color-schemes.ts";
import { getThemeMode, getSlotTheme, setSlotTheme } from "../../styles/theme.ts";
import type { Disposable } from "../../kernel/types.ts";
import { theme, layout, storage, download, notifications, dialogs } from "./deps.ts";
import { settingsState, type CustomTheme } from "./state.ts";
import { getSettingsDom } from "./dom.ts";

const STORAGE_KEY = "custom_themes";

// ── Token grouping for the editor ────────────────────────────────────────────

const TOKEN_GROUPS: { label: string; tokens: string[] }[] = [
  { label: "Background", tokens: ["--bg", "--bg-sidebar", "--bg-code", "--bg-hover"] },
  { label: "Borders", tokens: ["--border", "--border-bright", "--border-dark"] },
  { label: "Accent", tokens: ["--primary", "--secondary", "--tertiary", "--accent"] },
  { label: "Text", tokens: ["--text", "--text-muted", "--text-secondary", "--text-neutral"] },
  { label: "Status", tokens: ["--green", "--danger", "--success"] },
  {
    label: "Syntax",
    tokens: (COLOR_SCHEME_TOKENS as readonly string[]).filter((t) => t.startsWith("--syntax-")),
  },
];

// ── Helpers ──────────────────────────────────────────────────────────────────

function generateId(): string {
  return `custom-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}

/** Try to parse a CSS color string as a hex value for <input type="color">. */
function toHexForPicker(value: string): string | null {
  const trimmed = value.trim();
  if (/^#[0-9a-fA-F]{6}$/.test(trimmed)) return trimmed;
  if (/^#[0-9a-fA-F]{3}$/.test(trimmed)) {
    const r = trimmed[1]!, g = trimmed[2]!, b = trimmed[3]!;
    return `#${r}${r}${g}${g}${b}${b}`;
  }
  return null;
}

/** Prettify a CSS token name for display. */
function tokenLabel(token: string): string {
  return token
    .replace(/^--/, "")
    .replace(/-/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function isCustomTheme(id: string): boolean {
  return settingsState.customThemes.some((t) => t.id === id);
}

/** Generate a unique theme name ("My Theme", "My Theme 2", ...). */
function uniqueThemeName(): string {
  const base = "My Theme";
  const names = new Set(settingsState.customThemes.map((t) => t.name));
  if (!names.has(base)) return base;
  for (let i = 2; ; i++) {
    const candidate = `${base} ${i}`;
    if (!names.has(candidate)) return candidate;
  }
}

// ── HSL color conversion ────────────────────────────────────────────────────

function hexToHsl(hex: string): { h: number; s: number; l: number } {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return { h: 0, s: 0, l: Math.round(l * 100) };
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h = 0;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
  else if (max === g) h = ((b - r) / d + 2) / 6;
  else h = ((r - g) / d + 4) / 6;
  return { h: Math.round(h * 360), s: Math.round(s * 100), l: Math.round(l * 100) };
}

function hslToHex(h: number, s: number, l: number): string {
  const s1 = s / 100, l1 = l / 100;
  const a = s1 * Math.min(l1, 1 - l1);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    const c = l1 - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(c * 255).toString(16).padStart(2, "0");
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

// ── Color picker (persistent draggable popup, reused across tokens) ─────────

const QUICK_COLORS = [
  "#ef4444", "#f97316", "#f59e0b", "#eab308", "#84cc16", "#22c55e",
  "#14b8a6", "#06b6d4", "#3b82f6", "#6366f1", "#8b5cf6", "#a855f7",
  "#d946ef", "#ec4899", "#f43f5e", "#ffffff", "#a1a1aa", "#000000",
];

interface PickerState {
  el: HTMLElement;
  cleanup: () => void;
  onChange: (hex: string) => void;
  setColor: (hex: string) => void;
}

let picker: PickerState | null = null;

function closeActivePicker(): void {
  if (!picker) return;
  picker.el.remove();
  picker.cleanup();
  picker = null;
}

/** Create the picker popup once, centered on screen. */
function ensurePicker(): PickerState {
  if (picker) return picker;

  let h = 0, s = 0, l = 0;
  let onChangeFn: (hex: string) => void = () => {};

  const popup = document.createElement("div");
  popup.className = "color-picker-popup";

  // Drag header.
  const dragBar = document.createElement("div");
  dragBar.className = "color-picker-titlebar";

  const titlePreview = document.createElement("div");
  titlePreview.className = "color-picker-preview-sm";
  dragBar.appendChild(titlePreview);

  const titleText = document.createElement("span");
  titleText.className = "color-picker-title";
  titleText.textContent = "Color";
  dragBar.appendChild(titleText);

  const spacer = document.createElement("div");
  spacer.style.flex = "1";
  dragBar.appendChild(spacer);

  const closeBtn = document.createElement("button");
  closeBtn.className = "btn btn-ghost btn-icon color-picker-close";
  closeBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>';
  closeBtn.addEventListener("click", () => closeActivePicker());
  dragBar.appendChild(closeBtn);
  popup.appendChild(dragBar);

  // Dragging logic.
  let dragOffsetX = 0, dragOffsetY = 0;
  const onDragMove = (e: MouseEvent) => {
    popup.style.left = `${e.clientX - dragOffsetX}px`;
    popup.style.top = `${e.clientY - dragOffsetY}px`;
  };
  const onDragUp = () => {
    document.removeEventListener("mousemove", onDragMove);
    document.removeEventListener("mouseup", onDragUp);
  };
  dragBar.addEventListener("mousedown", (e) => {
    if ((e.target as HTMLElement).closest("button")) return;
    e.preventDefault();
    const rect = popup.getBoundingClientRect();
    dragOffsetX = e.clientX - rect.left;
    dragOffsetY = e.clientY - rect.top;
    document.addEventListener("mousemove", onDragMove);
    document.addEventListener("mouseup", onDragUp);
  });

  // Body.
  const body = document.createElement("div");
  body.className = "color-picker-body";

  const hexInput = document.createElement("input");
  hexInput.type = "text";
  hexInput.className = "form-input form-input-sm";
  hexInput.style.marginBottom = "0.625rem";
  body.appendChild(hexInput);

  // Quick color swatches.
  const swatchRow = document.createElement("div");
  swatchRow.className = "color-picker-swatches";
  for (const color of QUICK_COLORS) {
    const btn = document.createElement("button");
    btn.className = "color-picker-quick";
    btn.style.background = color;
    btn.title = color;
    btn.addEventListener("click", () => {
      applyHex(color);
    });
    swatchRow.appendChild(btn);
  }
  body.appendChild(swatchRow);

  // Slider factory.
  function createSlider(label: string, min: number, max: number): { track: HTMLElement; input: HTMLInputElement } {
    const group = document.createElement("div");
    group.className = "color-picker-slider-group";

    const lbl = document.createElement("div");
    lbl.className = "color-picker-slider-label";
    lbl.textContent = label;
    group.appendChild(lbl);

    const track = document.createElement("div");
    track.className = "color-picker-track";

    const input = document.createElement("input");
    input.type = "range";
    input.className = "color-picker-slider";
    input.min = String(min);
    input.max = String(max);
    track.appendChild(input);
    group.appendChild(track);
    body.appendChild(group);
    return { track, input };
  }

  const hueSlider = createSlider("Hue", 0, 360);
  const satSlider = createSlider("Saturation", 0, 100);
  const litSlider = createSlider("Lightness", 0, 100);
  popup.appendChild(body);

  hueSlider.track.style.background = "linear-gradient(to right, #f00, #ff0, #0f0, #0ff, #00f, #f0f, #f00)";

  function updateGradients(): void {
    satSlider.track.style.background = `linear-gradient(to right, hsl(${h}, 0%, ${l}%), hsl(${h}, 100%, ${l}%))`;
    litSlider.track.style.background = `linear-gradient(to right, #000, hsl(${h}, ${s}%, 50%), #fff)`;
  }

  function emitChange(): void {
    const hex = hslToHex(h, s, l);
    titlePreview.style.background = hex;
    hexInput.value = hex;
    updateGradients();
    onChangeFn(hex);
  }

  hueSlider.input.addEventListener("input", () => { h = Number(hueSlider.input.value); emitChange(); });
  satSlider.input.addEventListener("input", () => { s = Number(satSlider.input.value); emitChange(); });
  litSlider.input.addEventListener("input", () => { l = Number(litSlider.input.value); emitChange(); });

  hexInput.addEventListener("input", () => {
    const parsed = toHexForPicker(hexInput.value);
    if (parsed) applyHex(parsed);
  });

  /** Set color from hex, update all controls, emit. */
  function applyHex(hex: string): void {
    const hsl = hexToHsl(hex);
    h = hsl.h; s = hsl.s; l = hsl.l;
    hueSlider.input.value = String(h);
    satSlider.input.value = String(s);
    litSlider.input.value = String(l);
    titlePreview.style.background = hex;
    hexInput.value = hex;
    updateGradients();
    onChangeFn(hex);
  }

  /** Load a new color without emitting (used when switching tokens). */
  function setColor(hex: string): void {
    const parsed = toHexForPicker(hex) ?? "#000000";
    const hsl = hexToHsl(parsed);
    h = hsl.h; s = hsl.s; l = hsl.l;
    hueSlider.input.value = String(h);
    satSlider.input.value = String(s);
    litSlider.input.value = String(l);
    titlePreview.style.background = parsed;
    hexInput.value = parsed;
    updateGradients();
  }

  // Center on screen.
  document.body.appendChild(popup);
  const popupRect = popup.getBoundingClientRect();
  popup.style.left = `${(window.innerWidth - popupRect.width) / 2}px`;
  popup.style.top = `${(window.innerHeight - popupRect.height) / 2}px`;

  // Escape closes (capture phase to prevent panel from closing too).
  const onKeydown = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      e.stopPropagation();
      closeActivePicker();
    }
  };
  document.addEventListener("keydown", onKeydown, true);

  picker = {
    el: popup,
    cleanup: () => {
      document.removeEventListener("keydown", onKeydown, true);
      document.removeEventListener("mousemove", onDragMove);
      document.removeEventListener("mouseup", onDragUp);
    },
    set onChange(fn: (hex: string) => void) { onChangeFn = fn; },
    get onChange() { return onChangeFn; },
    setColor,
  };

  return picker;
}

function openColorPicker(
  initialValue: string,
  onChange: (hex: string) => void,
): void {
  const p = ensurePicker();
  p.setColor(initialValue);
  p.onChange = onChange;
}

// ── Persistence ──────────────────────────────────────────────────────────────

async function persistCustomThemes(): Promise<void> {
  try {
    await storage.set(STORAGE_KEY, settingsState.customThemes);
  } catch {
    notifications.error("Failed to save custom themes");
  }
}

// ── Populate theme selects (one per mode) ────────────────────────────────────

export function populateThemeSelects(): void {
  const dom = getSettingsDom();
  const registered = theme.getRegisteredThemes();

  const dark = registered.filter((t) => t.category === "dark");
  const light = registered.filter((t) => t.category === "light");

  const darkSlot = getSlotTheme("dark");
  const lightSlot = getSlotTheme("light");

  // Dark select.
  dom.themeDarkSelect.innerHTML = "";
  for (const t of dark) {
    const opt = document.createElement("option");
    opt.value = t.id;
    opt.textContent = t.name;
    if (t.id === darkSlot) opt.selected = true;
    dom.themeDarkSelect.appendChild(opt);
  }

  // Light select.
  dom.themeLightSelect.innerHTML = "";
  for (const t of light) {
    const opt = document.createElement("option");
    opt.value = t.id;
    opt.textContent = t.name;
    if (t.id === lightSlot) opt.selected = true;
    dom.themeLightSelect.appendChild(opt);
  }
}

// ── Mode slot change handlers ────────────────────────────────────────────────

export function handleDarkSlotChange(id: string): void {
  setSlotTheme("dark", id);
  if (getThemeMode() === "dark") {
    theme.setActiveTheme(id);
  }
  if (isCustomTheme(id)) {
    renderEditor(id);
  } else {
    closeEditor();
  }
}

export function handleLightSlotChange(id: string): void {
  setSlotTheme("light", id);
  if (getThemeMode() === "light") {
    theme.setActiveTheme(id);
  }
  if (isCustomTheme(id)) {
    renderEditor(id);
  } else {
    closeEditor();
  }
}

// ── Create custom theme ──────────────────────────────────────────────────────

export async function createCustomTheme(): Promise<void> {
  const id = generateId();
  const currentTokens = theme.tokens;
  const currentMode = getThemeMode();
  const category = currentMode;

  const custom: CustomTheme = {
    id,
    name: uniqueThemeName(),
    category,
    tokens: { ...currentTokens },
  };

  settingsState.customThemes.push(custom);
  const disposable = theme.registerTheme(id, custom.name, custom.category, custom.tokens);
  settingsState.themeDisposables.set(id, disposable);

  // Assign to current mode slot and apply.
  setSlotTheme(currentMode, id);
  theme.setActiveTheme(id);

  await persistCustomThemes();

  populateThemeSelects();
  renderEditor(id);
  notifications.success(`Created "${custom.name}"`);
}

// ── Import ───────────────────────────────────────────────────────────────────

export function handleImport(): void {
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".json";

  fileInput.addEventListener("change", async () => {
    const files = fileInput.files;
    if (!files || files.length === 0) return;

    try {
      const text = await files[0]!.text();
      const data = JSON.parse(text) as Record<string, unknown>;

      // Validate structure.
      if (!data.name || typeof data.name !== "string") throw new Error("Missing 'name' field");
      if (!data.tokens || typeof data.tokens !== "object") throw new Error("Missing 'tokens' field");

      const tokens = data.tokens as Record<string, string>;
      const validTokens: Record<string, string> = {};
      const schema = COLOR_SCHEME_TOKENS as readonly string[];

      for (const [key, value] of Object.entries(tokens)) {
        if (schema.includes(key) && typeof value === "string") {
          validTokens[key] = value;
        }
      }

      if (Object.keys(validTokens).length === 0) {
        throw new Error("No valid theme tokens found");
      }

      const id = typeof data.id === "string" && data.id ? data.id : generateId();
      const category = data.category === "light" ? "light" as const : "dark" as const;

      // Check for ID collision with existing themes.
      const finalId = settingsState.customThemes.some((t) => t.id === id)
        ? generateId()
        : id;

      const custom: CustomTheme = {
        id: finalId,
        name: data.name as string,
        category,
        tokens: validTokens,
      };

      settingsState.customThemes.push(custom);
      const disposable = theme.registerTheme(finalId, custom.name, custom.category, custom.tokens);
      settingsState.themeDisposables.set(finalId, disposable);

      // Assign to matching mode slot and apply.
      setSlotTheme(category, finalId);
      theme.setActiveTheme(finalId);

      await persistCustomThemes();

      populateThemeSelects();
      renderEditor(finalId);
      notifications.success(`Imported "${custom.name}"`);
    } catch (err) {
      notifications.error(`Import failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  });

  fileInput.click();
}

// ── Export ────────────────────────────────────────────────────────────────────

export function handleExport(id: string): void {
  const custom = settingsState.customThemes.find((t) => t.id === id);
  if (!custom) return;

  const data = {
    id: custom.id,
    name: custom.name,
    category: custom.category,
    tokens: custom.tokens,
  };

  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const filename = `${custom.name.replace(/[^a-zA-Z0-9_-]/g, "-").toLowerCase()}.json`;
  download.save(blob, filename);
  notifications.info(`Exported "${custom.name}"`);
}

// ── Delete ───────────────────────────────────────────────────────────────────

export async function handleDelete(id: string): Promise<void> {
  const custom = settingsState.customThemes.find((t) => t.id === id);
  if (!custom) return;

  const confirmed = await dialogs.confirm({
    title: "Delete Theme",
    message: `Delete "${custom.name}"? This cannot be undone.`,
    destructive: true,
  });
  if (!confirmed) return;

  // Dispose kernel registration (removes <style> tag + metadata).
  settingsState.themeDisposables.get(id)?.dispose();
  settingsState.themeDisposables.delete(id);

  settingsState.customThemes = settingsState.customThemes.filter((t) => t.id !== id);
  await persistCustomThemes();

  // If the deleted theme was in a mode slot, reset that slot to the built-in default.
  if (getSlotTheme("dark") === id) {
    setSlotTheme("dark", DARK_SCHEME_ID);
  }
  if (getSlotTheme("light") === id) {
    setSlotTheme("light", LIGHT_SCHEME_ID);
  }

  // Switch to the current mode's slot theme.
  const mode = getThemeMode();
  theme.setActiveTheme(getSlotTheme(mode));

  closeEditor();
  populateThemeSelects();
  notifications.info(`Deleted "${custom.name}"`);
}

// ── Load from storage ────────────────────────────────────────────────────────

export async function loadCustomThemes(): Promise<void> {
  try {
    const stored = await storage.get<CustomTheme[]>(STORAGE_KEY);
    if (!stored || !Array.isArray(stored)) return;

    for (const ct of stored) {
      if (!ct.id || !ct.name || !ct.tokens) continue;
      settingsState.customThemes.push(ct);
      const disposable = theme.registerTheme(ct.id, ct.name, ct.category, ct.tokens);
      settingsState.themeDisposables.set(ct.id, disposable);
    }

    // Re-apply the user's stored theme if it matches a custom theme.
    // restoreThemeSync() ran before plugins loaded, so it would have
    // fallen back to dark if the stored ID was a custom one.
    const storedId = localStorage.getItem("theme");
    if (storedId && settingsState.customThemes.some((t) => t.id === storedId)) {
      theme.setActiveTheme(storedId);
    }
  } catch {
    // Storage not available or corrupted — proceed without custom themes.
  }
}

// ── Color editor (renders into app panel) ───────────────────────────────────

/** Snapshot of theme state at the time the editor opened (for Reset). */
let savedState: { name: string; category: "dark" | "light"; tokens: Record<string, string> } | null = null;
let panelDisposable: Disposable | null = null;
let editingThemeId: string | null = null;

function closeEditor(): void {
  closeActivePicker();
  panelDisposable?.dispose();
  panelDisposable = null;
  editingThemeId = null;
  savedState = null;
}

export function renderEditor(themeId?: string): void {
  closeActivePicker();

  const dom = getSettingsDom();
  dom.themeEditorHost.innerHTML = "";

  const targetId = themeId ?? theme.activeThemeId;
  const custom = settingsState.customThemes.find((t) => t.id === targetId);
  if (!custom) {
    closeEditor();
    return;
  }

  editingThemeId = custom.id;

  // Take snapshot for Reset (only on fresh open, not re-renders from within the editor).
  if (!savedState || savedState !== (custom as unknown)) {
    savedState = {
      name: custom.name,
      category: custom.category,
      tokens: structuredClone(custom.tokens),
    };
  }

  // Build the editor DOM into a detached container.
  const editorEl = document.createElement("div");

  // Theme name input.
  const nameRow = document.createElement("div");
  nameRow.className = "settings-row";
  nameRow.style.marginBottom = "0.75rem";

  const nameInfo = document.createElement("div");
  nameInfo.className = "settings-row-info";
  const nameLabel = document.createElement("div");
  nameLabel.className = "settings-row-label";
  nameLabel.textContent = "Theme Name";
  nameInfo.appendChild(nameLabel);
  nameRow.appendChild(nameInfo);

  const nameControl = document.createElement("div");
  nameControl.className = "settings-row-control";
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.className = "form-input";
  nameInput.value = custom.name;
  nameInput.addEventListener("input", () => {
    custom.name = nameInput.value.trim() || "Untitled";
    // Update the kernel registration metadata for live display.
    settingsState.themeDisposables.get(custom.id)?.dispose();
    const disposable = theme.registerTheme(custom.id, custom.name, custom.category, custom.tokens);
    settingsState.themeDisposables.set(custom.id, disposable);
    theme.setActiveTheme(custom.id);
    populateThemeSelects();
  });
  nameControl.appendChild(nameInput);
  nameRow.appendChild(nameControl);
  editorEl.appendChild(nameRow);

  // Category toggle.
  const catRow = document.createElement("div");
  catRow.className = "settings-row";
  catRow.style.marginBottom = "0.75rem";

  const catInfo = document.createElement("div");
  catInfo.className = "settings-row-info";
  const catLabel = document.createElement("div");
  catLabel.className = "settings-row-label";
  catLabel.textContent = "Category";
  const catDesc = document.createElement("div");
  catDesc.className = "settings-row-desc";
  catDesc.textContent = "Affects color-scheme (scrollbar appearance)";
  catInfo.appendChild(catLabel);
  catInfo.appendChild(catDesc);
  catRow.appendChild(catInfo);

  const catControl = document.createElement("div");
  catControl.className = "settings-row-control";
  const catSelect = document.createElement("select");
  catSelect.className = "form-select";
  for (const cat of ["dark", "light"] as const) {
    const opt = document.createElement("option");
    opt.value = cat;
    opt.textContent = cat.charAt(0).toUpperCase() + cat.slice(1);
    if (cat === custom.category) opt.selected = true;
    catSelect.appendChild(opt);
  }
  catSelect.addEventListener("change", () => {
    custom.category = catSelect.value as "dark" | "light";
    // Re-register so the theme appears in the correct category dropdown.
    settingsState.themeDisposables.get(custom.id)?.dispose();
    const disposable = theme.registerTheme(custom.id, custom.name, custom.category, custom.tokens);
    settingsState.themeDisposables.set(custom.id, disposable);
    theme.setActiveTheme(custom.id);
    populateThemeSelects();
  });
  catControl.appendChild(catSelect);
  catRow.appendChild(catControl);
  editorEl.appendChild(catRow);

  // "Assign to mode" button.
  const assignRow = document.createElement("div");
  assignRow.style.cssText = "margin-bottom: 1rem;";
  const assignBtn = document.createElement("button");
  assignBtn.className = "btn btn-ghost btn-sm";
  assignBtn.textContent = `Use as ${custom.category === "dark" ? "Dark" : "Light"} Mode Theme`;
  assignBtn.addEventListener("click", () => {
    const mode = catSelect.value as "dark" | "light";
    setSlotTheme(mode, custom.id);
    if (getThemeMode() === mode) {
      theme.setActiveTheme(custom.id);
    }
    populateThemeSelects();
    notifications.success(`Set as ${mode} mode theme`);
  });
  assignRow.appendChild(assignBtn);
  editorEl.appendChild(assignRow);

  // Color editor groups.
  for (const group of TOKEN_GROUPS) {
    const section = document.createElement("div");
    section.style.marginBottom = "1rem";

    const heading = document.createElement("div");
    heading.className = "form-label";
    heading.style.marginBottom = "0.5rem";
    heading.textContent = group.label;
    section.appendChild(heading);

    for (const token of group.tokens) {
      const value = custom.tokens[token] ?? "";

      const row = document.createElement("div");
      row.className = "theme-editor-row";

      const label = document.createElement("label");
      label.className = "theme-editor-label";
      label.textContent = tokenLabel(token);
      row.appendChild(label);

      // Clickable color swatch — opens inline HSL picker.
      const swatch = document.createElement("button");
      swatch.className = "theme-editor-swatch";
      swatch.style.background = toHexForPicker(value) ?? "var(--bg)";

      // Text input (accepts any CSS color).
      const textInput = document.createElement("input");
      textInput.type = "text";
      textInput.className = "form-input form-input-sm theme-editor-text";
      textInput.value = value;
      textInput.placeholder = token;

      swatch.addEventListener("click", () => {
        openColorPicker(textInput.value || "#000000", (hex) => {
          textInput.value = hex;
          swatch.style.background = hex;
          custom.tokens[token] = hex;
          theme.updateThemeTokens(custom.id, custom.tokens);
        });
      });

      row.appendChild(swatch);
      row.appendChild(textInput);

      // Sync: text → swatch + live preview.
      textInput.addEventListener("input", () => {
        custom.tokens[token] = textInput.value;
        const newHex = toHexForPicker(textInput.value);
        if (newHex) swatch.style.background = newHex;
        theme.updateThemeTokens(custom.id, custom.tokens);
      });

      section.appendChild(row);
    }

    editorEl.appendChild(section);
  }

  // Footer: Save | Reset | Export | Delete.
  const footer = document.createElement("div");
  footer.style.cssText = "display: flex; gap: 0.5rem; flex-wrap: wrap; padding: 0.75rem 0; border-top: 1px solid var(--border);";

  const saveBtn = document.createElement("button");
  saveBtn.className = "btn btn-primary btn-sm";
  saveBtn.textContent = "Save";
  saveBtn.addEventListener("click", () => {
    savedState = {
      name: custom.name,
      category: custom.category,
      tokens: structuredClone(custom.tokens),
    };
    void persistCustomThemes();
    notifications.success("Theme saved");
  });
  footer.appendChild(saveBtn);

  const resetBtn = document.createElement("button");
  resetBtn.className = "btn btn-ghost btn-sm";
  resetBtn.textContent = "Reset";
  resetBtn.addEventListener("click", () => {
    if (!savedState) return;
    custom.name = savedState.name;
    custom.category = savedState.category;
    custom.tokens = structuredClone(savedState.tokens);
    // Re-register with restored metadata.
    settingsState.themeDisposables.get(custom.id)?.dispose();
    const disposable = theme.registerTheme(custom.id, custom.name, custom.category, custom.tokens);
    settingsState.themeDisposables.set(custom.id, disposable);
    theme.setActiveTheme(custom.id);
    populateThemeSelects();
    renderEditor(custom.id);
    notifications.info("Theme reset to last save");
  });
  footer.appendChild(resetBtn);

  const exportBtn = document.createElement("button");
  exportBtn.className = "btn btn-ghost btn-sm";
  exportBtn.textContent = "Export";
  exportBtn.addEventListener("click", () => handleExport(custom.id));
  footer.appendChild(exportBtn);

  const deleteBtn = document.createElement("button");
  deleteBtn.className = "btn btn-danger btn-sm";
  deleteBtn.textContent = "Delete";
  deleteBtn.addEventListener("click", () => void handleDelete(custom.id));
  footer.appendChild(deleteBtn);

  editorEl.appendChild(footer);

  // Show the editor in the app panel.
  panelDisposable?.dispose();
  panelDisposable = layout.showPanel({
    title: "Theme Editor",
    content: editorEl,
    owner: "theme-editor",
    onHide: () => {
      // Revert unsaved changes when panel is closed.
      if (savedState && editingThemeId) {
        const ct = settingsState.customThemes.find((t) => t.id === editingThemeId);
        if (ct) {
          ct.name = savedState.name;
          ct.category = savedState.category;
          ct.tokens = structuredClone(savedState.tokens);
          settingsState.themeDisposables.get(ct.id)?.dispose();
          const disposable = theme.registerTheme(ct.id, ct.name, ct.category, ct.tokens);
          settingsState.themeDisposables.set(ct.id, disposable);
          theme.setActiveTheme(ct.id);
          populateThemeSelects();
        }
      }
      editingThemeId = null;
      savedState = null;
      panelDisposable = null;
    },
  });
}
