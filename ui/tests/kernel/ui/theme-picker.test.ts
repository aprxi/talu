import { describe, test, expect, beforeEach } from "bun:test";
import { setupThemePicker } from "../../../src/kernel/ui/theme-picker.ts";
import { ThemeAccessImpl } from "../../../src/kernel/ui/theme.ts";
import { BUILTIN_SCHEMES, DARK_SCHEME_ID, LIGHT_SCHEME_ID } from "../../../src/styles/color-schemes.ts";

describe("setupThemePicker", () => {
  let themeAccess: ThemeAccessImpl;

  beforeEach(() => {
    themeAccess = new ThemeAccessImpl();
    themeAccess.registerBuiltinSchemes(BUILTIN_SCHEMES);
    document.body.innerHTML = `
      <button class="theme-toggle-btn" id="theme-toggle" title="Toggle theme" aria-label="Toggle theme">
        <span class="theme-preview-dot"></span>
      </button>
    `;
    localStorage.removeItem("theme");
    document.documentElement.classList.remove(DARK_SCHEME_ID, LIGHT_SCHEME_ID);
  });

  test("returns disposable", () => {
    const d = setupThemePicker(themeAccess);
    expect(typeof d.dispose).toBe("function");
    d.dispose();
  });

  test("initializes button state from current theme", () => {
    localStorage.setItem("theme", DARK_SCHEME_ID);
    const d = setupThemePicker(themeAccess);
    const btn = document.getElementById("theme-toggle")!;
    expect(btn.getAttribute("aria-label")).toContain("light");
    d.dispose();
  });

  test("click toggles from dark to light", () => {
    localStorage.setItem("theme", DARK_SCHEME_ID);
    const d = setupThemePicker(themeAccess);
    const btn = document.getElementById("theme-toggle") as HTMLButtonElement;
    btn.click();
    expect(localStorage.getItem("theme")).toBe(LIGHT_SCHEME_ID);
    expect(document.documentElement.classList.contains(LIGHT_SCHEME_ID)).toBe(true);
    d.dispose();
  });

  test("click toggles from light to dark", () => {
    localStorage.setItem("theme", LIGHT_SCHEME_ID);
    document.documentElement.classList.add(LIGHT_SCHEME_ID);
    const d = setupThemePicker(themeAccess);
    const btn = document.getElementById("theme-toggle") as HTMLButtonElement;
    btn.click();
    expect(localStorage.getItem("theme")).toBe(DARK_SCHEME_ID);
    expect(document.documentElement.classList.contains(DARK_SCHEME_ID)).toBe(true);
    d.dispose();
  });

  test("no-op when toggle button is missing", () => {
    document.body.innerHTML = "";
    const d = setupThemePicker(themeAccess);
    expect(() => d.dispose()).not.toThrow();
  });
});
