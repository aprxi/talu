/**
 * Built-in color schemes.
 *
 * Default runtime supports only two base schemes: `talu` (dark)
 * and `light-talu` (light). Additional schemes can be supplied later
 * by plugins via ThemeAccess.registerTheme().
 */

export interface ColorScheme {
  id: string;
  name: string;
  category: "dark" | "light";
  tokens: Record<string, string>;
}

export const DARK_SCHEME_ID = "talu";
export const LIGHT_SCHEME_ID = "light-talu";

/**
 * Scheme ID represented by CSS defaults/classes (no inline overrides required).
 */
export const CSS_DEFAULT_SCHEME_ID = DARK_SCHEME_ID;

/**
 * All theme tokens supported by the shared UI/docs CSS base.
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

const dark = (id: string, name: string, tokens: Record<string, string>): ColorScheme =>
  ({ id, name, category: "dark", tokens });

const light = (id: string, name: string, tokens: Record<string, string>): ColorScheme =>
  ({ id, name, category: "light", tokens });

export const BUILTIN_SCHEMES: ColorScheme[] = [
  dark(DARK_SCHEME_ID, "Talu Dark", {
    "--bg": "#0c0c10",
    "--bg-sidebar": "#08080c",
    "--bg-code": "#181820",
    "--bg-hover": "#27272a",
    "--border": "#1e1e26",
    "--border-bright": "rgba(96, 165, 250, 0.2)",
    "--border-dark": "#3f3f46",
    "--primary": "#60a5fa",
    "--secondary": "#e4cb81",
    "--tertiary": "#e4cb81",
    "--accent": "#38bdf8",
    "--text": "#f0f4f8",
    "--text-muted": "#94a3b8",
    "--text-secondary": "#a1a1aa",
    "--text-neutral": "#ffffff",
    "--green": "#38bdf8",
    "--danger": "#f87171",
    "--success": "#4ade80",
    "--syntax-class-keyword": "#fb7185",
    "--syntax-enum-keyword": "#f472b6",
    "--syntax-type-keyword": "#f472b6",
    "--syntax-class-name": "#f59e0b",
    "--syntax-enum-name": "#eab308",
    "--syntax-type-name": "#93c5fd",
    "--syntax-keyword": "#e23670",
    "--syntax-keyword-type": "#f472b6",
    "--syntax-keyword-declaration": "#fb7185",
    "--syntax-keyword-namespace": "#a78bfa",
    "--syntax-string": "#4ade80",
    "--syntax-number": "#22d3ee",
    "--syntax-function": "#38bdf8",
    "--syntax-class": "#f59e0b",
    "--syntax-enum": "#eab308",
    "--syntax-type": "#93c5fd",
    "--syntax-namespace": "#8b5cf6",
    "--syntax-property": "#f0f4f8",
    "--syntax-constant": "#fb923c",
    "--syntax-builtin": "#06b6d4",
    "--syntax-decorator": "#c084fc",
    "--syntax-operator": "#f0f4f8",
    "--syntax-punctuation": "#a1a1aa",
    "--syntax-comment": "#4a5568",
  }),
  light(LIGHT_SCHEME_ID, "Talu Light", {
    "--bg": "#f8f9fb",
    "--bg-sidebar": "#eff1f5",
    "--bg-code": "#e4e6ec",
    "--bg-hover": "#e8ebf0",
    "--border": "#d0d6e0",
    "--border-bright": "rgba(37, 99, 235, 0.15)",
    "--border-dark": "#c8cdd6",
    "--primary": "#2563eb",
    "--secondary": "#92700a",
    "--tertiary": "#92700a",
    "--accent": "#0284c7",
    "--text": "#1a1d23",
    "--text-muted": "#5c6370",
    "--text-secondary": "#3d4451",
    "--text-neutral": "#0f1114",
    "--green": "#0284c7",
    "--danger": "#dc2626",
    "--success": "#15803d",
    "--syntax-class-keyword": "#be185d",
    "--syntax-enum-keyword": "#9f1239",
    "--syntax-type-keyword": "#9f1239",
    "--syntax-class-name": "#b45309",
    "--syntax-enum-name": "#a16207",
    "--syntax-type-name": "#1d4ed8",
    "--syntax-keyword": "#be123c",
    "--syntax-keyword-type": "#9f1239",
    "--syntax-keyword-declaration": "#be185d",
    "--syntax-keyword-namespace": "#6d28d9",
    "--syntax-string": "#15803d",
    "--syntax-number": "#0369a1",
    "--syntax-function": "#0369a1",
    "--syntax-class": "#b45309",
    "--syntax-enum": "#a16207",
    "--syntax-type": "#1d4ed8",
    "--syntax-namespace": "#7c3aed",
    "--syntax-property": "#1a1d23",
    "--syntax-constant": "#c2410c",
    "--syntax-builtin": "#0e7490",
    "--syntax-decorator": "#7e22ce",
    "--syntax-operator": "#1a1d23",
    "--syntax-punctuation": "#5c6370",
    "--syntax-comment": "#8b919a",
  }),
];
