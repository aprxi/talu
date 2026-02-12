/**
 * Theme toggle wiring for the kernel top bar.
 *
 * Built-in runtime exposes only dark/light (`talu` / `light-talu`).
 * Additional overrides can still be registered later by plugins.
 */

import type { Disposable } from "../types.ts";
import type { ThemeAccessImpl } from "./theme.ts";
import { setTheme, getCurrentTheme } from "../../styles/theme.ts";
import { DARK_SCHEME_ID, LIGHT_SCHEME_ID } from "../../styles/color-schemes.ts";

export function setupThemePicker(themeAccess: ThemeAccessImpl): Disposable {
  const toggleBtn = document.querySelector(".theme-toggle-btn") as HTMLButtonElement | null;
  if (!toggleBtn) return { dispose() {} };

  function updateThemeUI(theme: string): void {
    const next = theme === LIGHT_SCHEME_ID ? DARK_SCHEME_ID : LIGHT_SCHEME_ID;
    const label = next === LIGHT_SCHEME_ID ? "Switch to light mode" : "Switch to dark mode";
    toggleBtn.setAttribute("aria-label", label);
    toggleBtn.setAttribute("title", label);
    toggleBtn.setAttribute("aria-pressed", theme === LIGHT_SCHEME_ID ? "true" : "false");
  }

  const clickHandler = () => {
    const current = getCurrentTheme();
    const next = current === LIGHT_SCHEME_ID ? DARK_SCHEME_ID : LIGHT_SCHEME_ID;
    setTheme(next);
    themeAccess.notifyChange();
    updateThemeUI(next);
  };
  toggleBtn.addEventListener("click", clickHandler);

  // Sync initial state.
  updateThemeUI(getCurrentTheme());

  return {
    dispose() {
      toggleBtn.removeEventListener("click", clickHandler);
    },
  };
}
