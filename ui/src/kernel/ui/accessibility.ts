/**
 * Accessibility — ARIA roles on Kernel chrome, screen reader support.
 *
 * Sets up accessibility attributes on kernel-owned elements:
 * - Activity bar: role=tablist, tab items with aria-selected
 * - Status bar: role=status, aria-live=polite
 * - Shadow hosts: role=tabpanel, aria-labelledby
 * - Notifications: role=alert / aria-live
 */

/**
 * Apply accessibility attributes to kernel chrome elements.
 * Call once during kernel boot after DOM is ready.
 */
export function setupAccessibility(): void {
  // Activity bar.
  const activityBar = document.getElementById("activity-bar");
  if (activityBar) {
    activityBar.setAttribute("role", "tablist");
    activityBar.setAttribute("aria-label", "Plugin views");

    // Activity bar items.
    const items = activityBar.querySelectorAll<HTMLElement>("[data-mode]");
    for (const item of items) {
      item.setAttribute("role", "tab");
      item.setAttribute("aria-selected", "false");
      const mode = item.dataset["mode"];
      if (mode) {
        item.setAttribute("aria-label", mode.replace(/-/g, " "));
      }
    }
  }

  // Status bar.
  const statusBar = document.getElementById("status-bar");
  if (statusBar) {
    statusBar.setAttribute("role", "status");
    statusBar.setAttribute("aria-live", "polite");
  }
}

/**
 * Set up accessibility on a shadow host element for a plugin view.
 * @param host - The shadow host element (has data-plugin-id).
 * @param pluginName - The plugin's display name.
 * @param tabId - The activity bar tab element ID for aria-labelledby.
 */
export function setupPluginViewAccessibility(
  host: HTMLElement,
  pluginName: string,
  tabId?: string,
): void {
  host.setAttribute("role", "tabpanel");
  host.setAttribute("aria-label", pluginName);
  if (tabId) {
    host.setAttribute("aria-labelledby", tabId);
  }
}

/**
 * Update aria-selected on activity bar tabs when the active view changes.
 * @param activeMode - The mode ID of the newly active view.
 */
export function updateActiveTab(activeMode: string): void {
  const activityBar = document.getElementById("activity-bar");
  if (!activityBar) return;

  const items = activityBar.querySelectorAll<HTMLElement>("[data-mode]");
  for (const item of items) {
    item.setAttribute("aria-selected", item.dataset["mode"] === activeMode ? "true" : "false");
  }
}
