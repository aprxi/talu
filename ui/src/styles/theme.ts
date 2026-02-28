/**
 * Theme operations — apply/restore color schemes.
 *
 * CSS (colors.css) is the source of truth for built-in themes.
 * Custom themes inject <style> tags via the kernel. This module
 * handles the low-level class toggle + localStorage persistence.
 */

import { BUILTIN_THEMES, DARK_SCHEME_ID, LIGHT_SCHEME_ID } from "./color-schemes.ts";

/** Set of all known theme IDs — expanded at runtime when custom themes register. */
const knownThemeIds = new Set(BUILTIN_THEMES.map((t) => t.id));

/** Currently applied theme class on :root (tracked so we can remove it). */
let activeThemeClass: string | null = null;

/** Register a theme ID so normalizeThemeId accepts it. */
export function registerThemeId(id: string): void {
  knownThemeIds.add(id);
}

/** Unregister a theme ID (e.g. when a custom theme is deleted). */
export function unregisterThemeId(id: string): void {
  knownThemeIds.delete(id);
}

function normalizeThemeId(raw: string | null): string {
  if (!raw) return DARK_SCHEME_ID;
  return knownThemeIds.has(raw) ? raw : DARK_SCHEME_ID;
}

// ── Core operations ──────────────────────────────────────────────────────────

/** Synchronous theme restore — call before any DOM rendering to prevent FOUC. */
export function restoreThemeSync(): void {
  const raw = localStorage.getItem("theme");
  const theme = normalizeThemeId(raw);
  const root = document.documentElement;
  if (activeThemeClass) root.classList.remove(activeThemeClass);
  root.classList.add(theme);
  activeThemeClass = theme;
  // Don't overwrite localStorage — if raw was a custom theme ID that hasn't
  // been registered yet (plugin not loaded), preserve it so the settings
  // plugin can re-apply it after loading custom themes from storage.
}

/** Set the active color scheme class on :root. */
export function setTheme(theme: string): void {
  const normalized = normalizeThemeId(theme);
  const root = document.documentElement;
  if (activeThemeClass) root.classList.remove(activeThemeClass);
  root.classList.add(normalized);
  activeThemeClass = normalized;
  localStorage.setItem("theme", normalized);
}

/** Get the current theme ID. */
export function getCurrentTheme(): string {
  return normalizeThemeId(localStorage.getItem("theme"));
}

// ── Mode slot helpers ────────────────────────────────────────────────────────

/** Get the current mode (dark or light). */
export function getThemeMode(): "dark" | "light" {
  return localStorage.getItem("theme-mode") === "light" ? "light" : "dark";
}

/** Set the current mode. Does not apply a theme — call setTheme() separately. */
export function setThemeMode(mode: "dark" | "light"): void {
  localStorage.setItem("theme-mode", mode);
}

/** Get the preferred theme ID for a mode slot. */
export function getSlotTheme(mode: "dark" | "light"): string {
  const stored = localStorage.getItem(`theme-${mode}`);
  if (stored && knownThemeIds.has(stored)) return stored;
  return mode === "dark" ? DARK_SCHEME_ID : LIGHT_SCHEME_ID;
}

/** Assign a theme ID to a mode slot. */
export function setSlotTheme(mode: "dark" | "light", themeId: string): void {
  localStorage.setItem(`theme-${mode}`, themeId);
}
