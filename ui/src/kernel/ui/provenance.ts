/**
 * View Provenance Indicator — Kernel-owned topbar label showing the
 * active main-slot plugin name (+ ID for third-party).
 *
 * Updates on slot switch. Clicking opens the command palette filtered
 * to plugin commands. Third-party plugins get a visual "ext" badge.
 */

let indicator: HTMLElement | null = null;
let commandPaletteOpener: (() => void) | null = null;

/**
 * Initialize the provenance indicator in the topbar.
 * Looks for an element with id="view-provenance" or creates one.
 */
export function initProvenance(): void {
  indicator = document.getElementById("view-provenance");
  if (!indicator) {
    const topbar = document.querySelector(".topbar");
    if (topbar) {
      indicator = document.createElement("span");
      indicator.id = "view-provenance";
      indicator.style.cssText =
        "font-size:0.6875rem;color:var(--text-subtle,#6c7086);margin-left:0.5rem;cursor:pointer;user-select:none;";
      indicator.title = "Click for plugin info";
      topbar.appendChild(indicator);
    }
  }

  if (indicator) {
    indicator.addEventListener("click", () => {
      if (commandPaletteOpener) commandPaletteOpener();
    });
  }
}

/** Set the callback for clicking the provenance indicator. */
export function setProvenanceAction(opener: () => void): void {
  commandPaletteOpener = opener;
}

/**
 * Update the provenance indicator with the active plugin info.
 * @param name - Plugin display name.
 * @param id - Plugin ID.
 * @param builtin - Whether the plugin is built-in.
 */
export function updateProvenance(name: string, id: string, builtin: boolean): void {
  if (!indicator) return;

  indicator.textContent = "";

  const nameSpan = document.createElement("span");
  nameSpan.textContent = name;
  indicator.appendChild(nameSpan);

  if (!builtin) {
    // Third-party visual distinction: show ID + "ext" badge.
    const idSpan = document.createElement("span");
    idSpan.textContent = ` (${id})`;
    idSpan.style.opacity = "0.7";
    indicator.appendChild(idSpan);

    const badge = document.createElement("span");
    badge.textContent = "ext";
    badge.style.cssText =
      "margin-left:0.375rem;padding:0.0625rem 0.25rem;font-size:0.5625rem;font-weight:600;" +
      "background:var(--accent,#6366f1);color:var(--bg,#09090b);border-radius:2px;vertical-align:middle;";
    indicator.appendChild(badge);
  }
}
