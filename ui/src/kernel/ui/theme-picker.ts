/**
 * Theme toggle wiring for the kernel top bar.
 *
 * Flips between "dark" and "light" mode slots. Each slot maps to a
 * user-configurable theme ID stored in localStorage.
 */

import type { Disposable } from "../types.ts";
import type { ThemeAccessImpl } from "./theme.ts";
import { getThemeMode, setThemeMode, getSlotTheme } from "../../styles/theme.ts";

export function setupThemePicker(themeAccess: ThemeAccessImpl): Disposable {
  const toggleBtn = document.querySelector(".theme-toggle-btn") as HTMLButtonElement | null;
  if (!toggleBtn) return { dispose() {} };

  function updateThemeUI(mode: "dark" | "light"): void {
    const nextMode = mode === "dark" ? "light" : "dark";
    const label = `Switch to ${nextMode} mode`;
    toggleBtn.setAttribute("aria-label", label);
    toggleBtn.setAttribute("title", label);
    toggleBtn.setAttribute("aria-pressed", mode === "light" ? "true" : "false");
  }

  const clickHandler = () => {
    const currentMode = getThemeMode();
    const newMode = currentMode === "dark" ? "light" : "dark";
    setThemeMode(newMode);
    const themeId = getSlotTheme(newMode);
    themeAccess.setActiveTheme(themeId);
    updateThemeUI(newMode);
  };
  toggleBtn.addEventListener("click", clickHandler);

  // Sync initial state.
  updateThemeUI(getThemeMode());

  return {
    dispose() {
      toggleBtn.removeEventListener("click", clickHandler);
    },
  };
}
