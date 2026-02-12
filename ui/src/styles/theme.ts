/**
 * Theme operations — apply/restore color schemes from TypeScript data.
 *
 * Color scheme data lives in color-schemes.ts. This module provides the
 * low-level apply/restore/query functions used by the kernel.
 */

import { BUILTIN_SCHEMES, DARK_SCHEME_ID, LIGHT_SCHEME_ID } from "./color-schemes.ts";

const BUILTIN_THEME_IDS = new Set(BUILTIN_SCHEMES.map((s) => s.id));
const CLASS_THEMES = [DARK_SCHEME_ID, LIGHT_SCHEME_ID];

function normalizeThemeId(raw: string | null): string {
  if (!raw) return DARK_SCHEME_ID;
  return BUILTIN_THEME_IDS.has(raw) ? raw : DARK_SCHEME_ID;
}

// ── Core operations ──────────────────────────────────────────────────────────

/** Synchronous theme restore — call before any DOM rendering to prevent FOUC. */
export function restoreThemeSync(): void {
  const theme = normalizeThemeId(localStorage.getItem("theme"));
  const root = document.documentElement;
  root.classList.remove(...CLASS_THEMES);
  root.classList.add(theme);
  localStorage.setItem("theme", theme);
}

/** Set the active built-in color scheme class on :root. */
export function setTheme(theme: string): void {
  const normalized = normalizeThemeId(theme);
  const root = document.documentElement;
  root.classList.remove(...CLASS_THEMES);
  root.classList.add(normalized);
  localStorage.setItem("theme", normalized);
}

/** Get the current theme ID. */
export function getCurrentTheme(): string {
  return normalizeThemeId(localStorage.getItem("theme"));
}
