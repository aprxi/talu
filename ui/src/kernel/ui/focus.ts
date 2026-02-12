/**
 * Focus Management — track and restore focus across slot transitions.
 *
 * Recursive shadowRoot.activeElement traversal finds the actually focused
 * element across nested shadow boundaries.
 */

/**
 * Get the deepest actively focused element, traversing shadow roots.
 * Returns the innermost focused element, or null if nothing is focused.
 */
export function getDeepActiveElement(): Element | null {
  let active: Element | null = document.activeElement;
  while (active?.shadowRoot?.activeElement) {
    active = active.shadowRoot.activeElement;
  }
  return active;
}

/**
 * Determine which plugin view currently has focus by finding
 * the shadow host with data-plugin-id in the focus chain.
 */
export function getFocusedViewId(): string | null {
  let el: Element | null = document.activeElement;
  while (el) {
    if (el instanceof HTMLElement && el.dataset["pluginId"]) {
      return el.dataset["pluginId"];
    }
    // Walk up: check parent, then host element of shadow root.
    const parent = el.parentElement;
    if (parent) {
      el = parent;
    } else {
      // If we're at a shadow root boundary, get the host.
      const root = el.getRootNode();
      if (root instanceof ShadowRoot) {
        el = root.host;
      } else {
        break;
      }
    }
  }
  return null;
}

/** Saved focus state for restoring after view transitions. */
interface SavedFocus {
  viewId: string;
  element: Element | null;
}

let savedFocus: SavedFocus | null = null;

/** Save the current focus state for later restoration. */
export function saveFocus(): void {
  const viewId = getFocusedViewId();
  if (viewId) {
    savedFocus = {
      viewId,
      element: getDeepActiveElement(),
    };
  }
}

/**
 * Restore previously saved focus, or focus the default element in a view.
 * @param viewId - The view to restore focus to.
 * @param defaultSelector - CSS selector for the default focus target within the shadow root.
 */
export function restoreFocus(viewId: string, defaultSelector?: string): void {
  // Try restoring saved focus.
  if (savedFocus?.viewId === viewId && savedFocus.element) {
    if (savedFocus.element instanceof HTMLElement) {
      savedFocus.element.focus();
      savedFocus = null;
      return;
    }
  }

  // Fall back to default focus selector.
  if (defaultSelector) {
    const host = document.querySelector<HTMLElement>(`[data-plugin-id="${viewId}"]`);
    if (host?.shadowRoot) {
      const target = host.shadowRoot.querySelector<HTMLElement>(defaultSelector);
      target?.focus();
      return;
    }
  }

  // Last resort: focus the first focusable element in the shadow root.
  const host = document.querySelector<HTMLElement>(`[data-plugin-id="${viewId}"]`);
  if (host?.shadowRoot) {
    const first = host.shadowRoot.querySelector<HTMLElement>(
      'button, input, select, textarea, a[href], [tabindex]:not([tabindex="-1"])',
    );
    first?.focus();
  }
}
