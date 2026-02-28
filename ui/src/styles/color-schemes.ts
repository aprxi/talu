/**
 * Theme metadata and token schema.
 *
 * Built-in themes are defined in CSS (colors.css). This module provides
 * metadata for the registry and the shared token schema used by the
 * kernel's JSON→CSS compiler, the theme editor, and the live token reader.
 */

export interface ThemeMetadata {
  id: string;
  name: string;
  category: "dark" | "light";
}

export const DARK_SCHEME_ID = "talu";
export const LIGHT_SCHEME_ID = "light-talu";

export const BUILTIN_THEMES: ThemeMetadata[] = [
  { id: DARK_SCHEME_ID, name: "Talu Dark", category: "dark" },
  { id: LIGHT_SCHEME_ID, name: "Talu Light", category: "light" },
];

/**
 * All theme tokens supported by the shared UI/docs CSS base.
 * This is the contract between the editor, the kernel compiler, and
 * the live token reader (getComputedStyle).
 */
export const COLOR_SCHEME_TOKENS = [
  "--bg", "--bg-sidebar", "--bg-code", "--bg-hover",
  "--border", "--border-bright", "--border-dark",
  "--primary", "--secondary", "--tertiary", "--accent",
  "--text", "--text-muted", "--text-secondary", "--text-neutral",
  "--green", "--danger", "--success",
  "--syntax-class-keyword", "--syntax-enum-keyword", "--syntax-type-keyword",
  "--syntax-class-name", "--syntax-enum-name", "--syntax-type-name",
  "--syntax-keyword", "--syntax-keyword-type", "--syntax-keyword-declaration", "--syntax-keyword-namespace",
  "--syntax-string", "--syntax-number", "--syntax-function",
  "--syntax-class", "--syntax-enum", "--syntax-type", "--syntax-namespace",
  "--syntax-property", "--syntax-constant", "--syntax-builtin", "--syntax-decorator",
  "--syntax-operator", "--syntax-punctuation", "--syntax-comment",
] as const;
