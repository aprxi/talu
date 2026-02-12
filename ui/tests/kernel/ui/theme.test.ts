import { describe, test, expect, spyOn, beforeEach } from "bun:test";
import { ThemeAccessImpl } from "../../../src/kernel/ui/theme.ts";
import { COLOR_SCHEME_TOKENS } from "../../../src/styles/color-schemes.ts";

describe("ThemeAccessImpl", () => {
  beforeEach(() => {
    localStorage.removeItem("theme");
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
    const d = theme.onChange(() => { called = true; });
    theme.notifyChange();
    expect(called).toBe(true);
    d.dispose();
  });

  test("onChange dispose stops notifications", () => {
    const theme = new ThemeAccessImpl();
    let count = 0;
    const d = theme.onChange(() => { count++; });
    theme.notifyChange();
    d.dispose();
    theme.notifyChange();
    expect(count).toBe(1);
  });

  test("onChange callback error does not break others", () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const theme = new ThemeAccessImpl();
    let secondCalled = false;
    theme.onChange(() => { throw new Error("cb boom"); });
    theme.onChange(() => { secondCalled = true; });
    theme.notifyChange();
    expect(secondCalled).toBe(true);
    spy.mockRestore();
  });

  test("registerBuiltinSchemes stores schemes", () => {
    const theme = new ThemeAccessImpl();
    theme.registerBuiltinSchemes([
      { id: "dark1", name: "Dark One", category: "dark", tokens: { "--bg": "#000" } },
      { id: "light1", name: "Light One", category: "light", tokens: { "--bg": "#fff" } },
    ]);
    const registered = theme.getRegisteredThemes();
    expect(registered.length).toBe(2);
    expect(registered.map((t) => t.id).sort()).toEqual(["dark1", "light1"]);
  });

  test("registerTheme validates token values", () => {
    const theme = new ThemeAccessImpl();
    // Valid token.
    const d = theme.registerTheme("custom", { "--bg": "#123456" });
    expect(d).toBeDefined();
    d.dispose();
  });

  test("registerTheme rejects url() in token values", () => {
    const theme = new ThemeAccessImpl();
    expect(() => theme.registerTheme("evil", { "--bg": "url(https://evil.com)" })).toThrow(
      /validation failed/,
    );
  });

  test("registerTheme rejects expression() in token values", () => {
    const theme = new ThemeAccessImpl();
    expect(() => theme.registerTheme("evil", { "--bg": "expression(alert(1))" })).toThrow(
      /validation failed/,
    );
  });

  test("registerTheme warns on unknown token names", () => {
    const spy = spyOn(console, "warn").mockImplementation(() => {});
    const theme = new ThemeAccessImpl();
    const d = theme.registerTheme("custom", { "--unknown-token": "#fff" });
    expect(spy).toHaveBeenCalled();
    d.dispose();
    spy.mockRestore();
  });

  test("registerTheme dispose removes theme", () => {
    const theme = new ThemeAccessImpl();
    const d = theme.registerTheme("custom", { "--bg": "#111" });
    expect(theme.getRegisteredThemes().find((t) => t.id === "custom")).toBeDefined();
    d.dispose();
    expect(theme.getRegisteredThemes().find((t) => t.id === "custom")).toBeUndefined();
  });

  test("getSchemeToken returns token value", () => {
    const theme = new ThemeAccessImpl();
    theme.registerBuiltinSchemes([
      { id: "test", name: "Test", category: "dark", tokens: { "--bg": "#222", "--text": "#eee" } },
    ]);
    expect(theme.getSchemeToken("test", "--bg")).toBe("#222");
    expect(theme.getSchemeToken("test", "--text")).toBe("#eee");
  });

  test("getSchemeToken returns undefined for missing scheme", () => {
    const theme = new ThemeAccessImpl();
    expect(theme.getSchemeToken("nonexistent", "--bg")).toBeUndefined();
  });

  test("valid color formats accepted", () => {
    const theme = new ThemeAccessImpl();
    // hex
    expect(() => theme.registerTheme("a", { "--bg": "#abc" }).dispose()).not.toThrow();
    // rgba
    expect(() => theme.registerTheme("b", { "--bg": "rgba(0,0,0,0.5)" }).dispose()).not.toThrow();
    // hsl
    expect(() => theme.registerTheme("c", { "--bg": "hsl(120,50%,50%)" }).dispose()).not.toThrow();
    // named color
    expect(() => theme.registerTheme("d", { "--bg": "red" }).dispose()).not.toThrow();
  });
});
