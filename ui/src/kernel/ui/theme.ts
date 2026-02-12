/**
 * Theme Access — CSS custom property management and plugin-provided themes.
 *
 * Theme tokens are defined on :root and inherit through shadow boundaries.
 * ctx.theme provides programmatic access and change notifications.
 */

import type { Disposable, ThemeAccess } from "../types.ts";
import { COLOR_SCHEME_TOKENS, DARK_SCHEME_ID, type ColorScheme } from "../../styles/color-schemes.ts";

/** Validate a theme token value — all tokens are colors. */
function validateTokenValue(_name: string, value: string): boolean {
  const dangerous = /url\s*\(|expression\s*\(/i;
  if (dangerous.test(value)) return false;
  return /^(#[0-9a-fA-F]{3,8}|rgba?\(|hsla?\(|[a-zA-Z]+)/.test(value);
}

interface RegisteredScheme {
  id: string;
  name: string;
  category: string;
  tokens: Record<string, string>;
}

export class ThemeAccessImpl implements ThemeAccess {
  private changeCallbacks = new Set<() => void>();
  private schemes = new Map<string, RegisteredScheme>();

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

  onChange(callback: () => void): Disposable {
    this.changeCallbacks.add(callback);
    return {
      dispose: () => this.changeCallbacks.delete(callback),
    };
  }

  /** Bulk-register built-in color schemes. */
  registerBuiltinSchemes(schemes: ColorScheme[]): void {
    for (const s of schemes) {
      this.schemes.set(s.id, { id: s.id, name: s.name, category: s.category, tokens: s.tokens });
    }
  }

  /** Register a single theme (for plugins). */
  registerTheme(id: string, tokens: Record<string, string>): Disposable {
    const errors: string[] = [];
    for (const [name, value] of Object.entries(tokens)) {
      if (!COLOR_SCHEME_TOKENS.includes(name as (typeof COLOR_SCHEME_TOKENS)[number])) {
        console.warn(`[kernel] Unknown theme token "${name}" — ignored.`);
        continue;
      }
      if (!validateTokenValue(name, value)) {
        errors.push(`Invalid value for ${name}: "${value}"`);
      }
    }

    if (errors.length > 0) {
      throw new Error(`Theme "${id}" validation failed:\n${errors.join("\n")}`);
    }

    this.schemes.set(id, { id, name: id, category: "dark", tokens });

    return {
      dispose: () => {
        this.schemes.delete(id);
      },
    };
  }

  /** List all registered themes (for picker UI). */
  getRegisteredThemes(): { id: string; name: string; category: string }[] {
    return [...this.schemes.values()].map(({ id, name, category }) => ({ id, name, category }));
  }

  /** Get a specific token value from a registered scheme. */
  getSchemeToken(id: string, token: string): string | undefined {
    return this.schemes.get(id)?.tokens[token];
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
}
