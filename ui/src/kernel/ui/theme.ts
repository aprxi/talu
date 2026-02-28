/**
 * Theme Access — registry, CSS injection, and change notifications.
 *
 * Built-in themes are CSS classes on :root (defined in colors.css).
 * Custom themes are JSON token dictionaries compiled to injected <style> tags.
 * Both use the same CSS custom property mechanism at runtime.
 */

import type { Disposable, ThemeAccess } from "../types.ts";
import { COLOR_SCHEME_TOKENS, DARK_SCHEME_ID } from "../../styles/color-schemes.ts";
import type { ThemeMetadata } from "../../styles/color-schemes.ts";
import { setTheme, registerThemeId, unregisterThemeId } from "../../styles/theme.ts";

interface RegisteredTheme {
  id: string;
  name: string;
  category: string;
}

export class ThemeAccessImpl implements ThemeAccess {
  private changeCallbacks = new Set<() => void>();
  private themes = new Map<string, RegisteredTheme>();

  get activeThemeId(): string {
    return localStorage.getItem("theme") || DARK_SCHEME_ID;
  }

  get tokens(): Record<string, string> {
    const result: Record<string, string> = {};
    const style = getComputedStyle(document.documentElement);
    for (const token of COLOR_SCHEME_TOKENS) {
      result[token] = style.getPropertyValue(token).trim();
    }
    return result;
  }

  setActiveTheme(id: string): void {
    setTheme(id);
    this.notifyChange();
  }

  onChange(callback: () => void): Disposable {
    this.changeCallbacks.add(callback);
    return {
      dispose: () => this.changeCallbacks.delete(callback),
    };
  }

  /** Bulk-register built-in themes (metadata only — CSS is in colors.css). */
  registerBuiltinThemes(themes: ThemeMetadata[]): void {
    for (const t of themes) {
      this.themes.set(t.id, { id: t.id, name: t.name, category: t.category });
    }
  }

  /**
   * Register a theme.
   * Without tokens: metadata-only (builtins).
   * With tokens: metadata + inject <style> tag (custom themes).
   */
  registerTheme(id: string, name: string, category: string, tokens?: Record<string, string>): Disposable {
    this.themes.set(id, { id, name, category });
    registerThemeId(id);

    if (tokens) {
      this.injectThemeStyle(id, tokens);
    }

    return {
      dispose: () => {
        this.themes.delete(id);
        unregisterThemeId(id);
        this.removeThemeStyle(id);
      },
    };
  }

  /** Update the injected <style> tag for a custom theme (live editor preview). */
  updateThemeTokens(id: string, tokens: Record<string, string>): void {
    this.injectThemeStyle(id, tokens);
  }

  /** List all registered themes (for picker UI). */
  getRegisteredThemes(): { id: string; name: string; category: string }[] {
    return [...this.themes.values()].map(({ id, name, category }) => ({ id, name, category }));
  }

  /** Notify all listeners of a theme change. */
  notifyChange(): void {
    for (const cb of this.changeCallbacks) {
      try {
        cb();
      } catch (err) {
        console.error("[kernel] Theme onChange callback threw:", err);
      }
    }
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  /** Compile a token dictionary to a CSS rule and inject/update a <style> tag. */
  private injectThemeStyle(id: string, tokens: Record<string, string>): void {
    const styleId = `theme-style-${id}`;
    let styleEl = document.getElementById(styleId) as HTMLStyleElement | null;
    if (!styleEl) {
      styleEl = document.createElement("style");
      styleEl.id = styleId;
      document.head.appendChild(styleEl);
    }
    styleEl.textContent = this.compileThemeCSS(id, tokens);
  }

  /** Remove an injected <style> tag for a custom theme. */
  private removeThemeStyle(id: string): void {
    const styleEl = document.getElementById(`theme-style-${id}`);
    if (styleEl) styleEl.remove();
  }

  /** Compile a token dictionary to a CSS rule string. */
  private compileThemeCSS(id: string, tokens: Record<string, string>): string {
    const entries = Object.entries(tokens)
      .filter(([k]) => (COLOR_SCHEME_TOKENS as readonly string[]).includes(k))
      .map(([k, v]) => `  ${k}: ${v};`)
      .join("\n");
    return `:root.${CSS.escape(id)} {\n${entries}\n}`;
  }
}
