import { describe, test, expect, beforeEach } from "bun:test";
import { restoreThemeSync, setTheme, getCurrentTheme } from "../../src/styles/theme.ts";
import { DARK_SCHEME_ID, LIGHT_SCHEME_ID } from "../../src/styles/color-schemes.ts";

/**
 * Tests for theme operations — restoreThemeSync, setTheme, getCurrentTheme.
 */

beforeEach(() => {
  localStorage.clear();
  document.documentElement.classList.remove(DARK_SCHEME_ID, LIGHT_SCHEME_ID);
});

// ── restoreThemeSync ────────────────────────────────────────────────────────

describe("restoreThemeSync", () => {
  test("falls back to DARK_SCHEME_ID when localStorage is empty", () => {
    restoreThemeSync();
    expect(document.documentElement.classList.contains(DARK_SCHEME_ID)).toBe(true);
    expect(localStorage.getItem("theme")).toBe(DARK_SCHEME_ID);
  });

  test("restores dark theme from localStorage", () => {
    localStorage.setItem("theme", DARK_SCHEME_ID);
    restoreThemeSync();
    expect(document.documentElement.classList.contains(DARK_SCHEME_ID)).toBe(true);
  });

  test("restores light theme from localStorage", () => {
    localStorage.setItem("theme", LIGHT_SCHEME_ID);
    restoreThemeSync();
    expect(document.documentElement.classList.contains(LIGHT_SCHEME_ID)).toBe(true);
    expect(document.documentElement.classList.contains(DARK_SCHEME_ID)).toBe(false);
  });

  test("invalid theme in localStorage falls back to dark", () => {
    localStorage.setItem("theme", "invalid-theme-xyz");
    restoreThemeSync();
    expect(document.documentElement.classList.contains(DARK_SCHEME_ID)).toBe(true);
    expect(localStorage.getItem("theme")).toBe(DARK_SCHEME_ID);
  });

  test("removes previous theme class before applying new one", () => {
    document.documentElement.classList.add(LIGHT_SCHEME_ID);
    localStorage.setItem("theme", DARK_SCHEME_ID);
    restoreThemeSync();
    expect(document.documentElement.classList.contains(LIGHT_SCHEME_ID)).toBe(false);
    expect(document.documentElement.classList.contains(DARK_SCHEME_ID)).toBe(true);
  });
});

// ── setTheme ────────────────────────────────────────────────────────────────

describe("setTheme", () => {
  test("sets light theme class on root", () => {
    setTheme(LIGHT_SCHEME_ID);
    expect(document.documentElement.classList.contains(LIGHT_SCHEME_ID)).toBe(true);
  });

  test("persists to localStorage", () => {
    setTheme(LIGHT_SCHEME_ID);
    expect(localStorage.getItem("theme")).toBe(LIGHT_SCHEME_ID);
  });

  test("removes previous theme class", () => {
    setTheme(LIGHT_SCHEME_ID);
    setTheme(DARK_SCHEME_ID);
    expect(document.documentElement.classList.contains(LIGHT_SCHEME_ID)).toBe(false);
    expect(document.documentElement.classList.contains(DARK_SCHEME_ID)).toBe(true);
  });

  test("normalizes invalid theme to dark", () => {
    setTheme("nonexistent");
    expect(document.documentElement.classList.contains(DARK_SCHEME_ID)).toBe(true);
    expect(localStorage.getItem("theme")).toBe(DARK_SCHEME_ID);
  });
});

// ── getCurrentTheme ─────────────────────────────────────────────────────────

describe("getCurrentTheme", () => {
  test("returns dark when localStorage is empty", () => {
    expect(getCurrentTheme()).toBe(DARK_SCHEME_ID);
  });

  test("returns stored theme", () => {
    localStorage.setItem("theme", LIGHT_SCHEME_ID);
    expect(getCurrentTheme()).toBe(LIGHT_SCHEME_ID);
  });

  test("normalizes invalid theme to dark", () => {
    localStorage.setItem("theme", "garbage");
    expect(getCurrentTheme()).toBe(DARK_SCHEME_ID);
  });
});
