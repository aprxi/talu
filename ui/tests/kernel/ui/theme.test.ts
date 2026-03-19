import { describe, test, expect, spyOn, beforeEach } from "bun:test";
import { ThemeAccessImpl } from "../../../src/kernel/ui/theme.ts";
import { COLOR_SCHEME_TOKENS } from "../../../src/styles/color-schemes.ts";

describe("ThemeAccessImpl", () => {
  beforeEach(() => {
    localStorage.removeItem("theme");
    document.documentElement.removeAttribute("style");
  });

  test("activeThemeId defaults to 'talu'", () => {
    const theme = new ThemeAccessImpl();
    expect(theme.activeThemeId).toBe("talu");
  });

  test("activeThemeId reads from localStorage", () => {
    localStorage.setItem("theme", "dracula");
    const theme = new ThemeAccessImpl();
    expect(theme.activeThemeId).toBe("dracula");
  });

  test("onChange registers callback and returns disposable", () => {
    const theme = new ThemeAccessImpl();
    let called = false;
    const d = theme.onChange(() => {
      called = true;
    });
    theme.notifyChange();
    expect(called).toBe(true);
    d.dispose();
  });

  test("onChange dispose stops notifications", () => {
    const theme = new ThemeAccessImpl();
    let count = 0;
    const d = theme.onChange(() => {
      count++;
    });
    theme.notifyChange();
    d.dispose();
    theme.notifyChange();
    expect(count).toBe(1);
  });

  test("onChange callback error does not break others", () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const theme = new ThemeAccessImpl();
    let secondCalled = false;
    theme.onChange(() => {
      throw new Error("cb boom");
    });
    theme.onChange(() => {
      secondCalled = true;
    });
    theme.notifyChange();
    expect(secondCalled).toBe(true);
    spy.mockRestore();
  });

  test("registerBuiltinThemes stores themes", () => {
    const theme = new ThemeAccessImpl();
    theme.registerBuiltinThemes([
      { id: "dark1", name: "Dark One", category: "dark" },
      { id: "light1", name: "Light One", category: "light" },
    ]);
    const registered = theme.getRegisteredThemes();
    expect(registered.length).toBe(2);
    expect(registered.map((t) => t.id).sort()).toEqual(["dark1", "light1"]);
  });

  test("setActiveTheme updates storage and notifies listeners", () => {
    const theme = new ThemeAccessImpl();
    let called = false;
    theme.onChange(() => {
      called = true;
    });
    theme.setActiveTheme("light-talu");
    expect(localStorage.getItem("theme")).toBe("light-talu");
    expect(called).toBe(true);
  });

  test("registerTheme injects a style tag and preserves token values", () => {
    const theme = new ThemeAccessImpl();
    const d = theme.registerTheme("custom", "Custom", "dark", {
      "--bg": "#123456",
      "--unknown-token": "ignored",
    });
    const styleEl = document.getElementById("theme-style-custom") as HTMLStyleElement | null;
    expect(styleEl).not.toBeNull();
    expect(styleEl!.textContent).toContain("--bg: #123456;");
    expect(styleEl!.textContent).not.toContain("--unknown-token");
    d.dispose();
  });

  test("updateThemeTokens refreshes the injected style tag", () => {
    const theme = new ThemeAccessImpl();
    const d = theme.registerTheme("custom", "Custom", "dark", { "--bg": "#111" });
    theme.updateThemeTokens("custom", { "--bg": "#222", "--text": "#eee" });
    const styleEl = document.getElementById("theme-style-custom") as HTMLStyleElement | null;
    expect(styleEl).not.toBeNull();
    expect(styleEl!.textContent).toContain("--bg: #222;");
    expect(styleEl!.textContent).toContain("--text: #eee;");
    d.dispose();
  });

  test("registerTheme dispose removes theme", () => {
    const theme = new ThemeAccessImpl();
    const d = theme.registerTheme("custom", "Custom", "dark", { "--bg": "#111" });
    expect(theme.getRegisteredThemes().find((t) => t.id === "custom")).toBeDefined();
    expect(document.getElementById("theme-style-custom")).not.toBeNull();
    d.dispose();
    expect(theme.getRegisteredThemes().find((t) => t.id === "custom")).toBeUndefined();
    expect(document.getElementById("theme-style-custom")).toBeNull();
  });

  test("tokens getter reflects computed CSS variables", () => {
    const theme = new ThemeAccessImpl();
    document.documentElement.style.setProperty("--bg", "#222");
    document.documentElement.style.setProperty("--text", "#eee");
    const tokens = theme.tokens;
    expect(tokens["--bg"]).toBe("#222");
    expect(tokens["--text"]).toBe("#eee");
    expect(COLOR_SCHEME_TOKENS.every((token) => token in tokens)).toBe(true);
  });
});
