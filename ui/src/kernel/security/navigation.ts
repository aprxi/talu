/**
 * Navigation Interception — prevents data exfiltration via URL navigation.
 *
 * Capture-phase click handler on document detects non-self anchor navigations
 * (including inside shadow roots via composedPath). Monkey-patches window.open
 * and location.assign/replace/href. Always active (not dev-mode only).
 *
 * Shows a Kernel confirm dialog before allowing external navigation.
 */

import type { Disposable } from "../types.ts";
import { StandardDialogsImpl } from "../ui/dialogs.ts";

let installed = false;
const selfOrigin = window.location.origin;
let originalOpen: typeof window.open;
const kernelDialogs = new StandardDialogsImpl("Kernel");

/**
 * Install navigation interception. Call once during kernel boot.
 * Returns a Disposable to remove the listener.
 */
export function installNavigationInterception(): Disposable {
  if (installed) return { dispose() {} };
  installed = true;

  // Capture-phase click handler.
  const clickHandler = (e: MouseEvent) => {
    const path = e.composedPath();
    for (const el of path) {
      if (el instanceof HTMLAnchorElement && el.href) {
        try {
          const url = new URL(el.href, selfOrigin);
          if (url.origin !== selfOrigin) {
            e.preventDefault();
            e.stopPropagation();
            showNavigationConfirm(url.href);
            return;
          }
        } catch {
          // Invalid URL — let the browser handle it.
        }
      }
    }
  };

  document.addEventListener("click", clickHandler, true);

  // Monkey-patch window.open.
  originalOpen = window.open.bind(window);
  window.open = function (url?: string | URL, ...rest: unknown[]): WindowProxy | null {
    if (url) {
      try {
        const resolved = new URL(String(url), selfOrigin);
        if (resolved.origin !== selfOrigin) {
          showNavigationConfirm(resolved.href);
          return null;
        }
      } catch {
        // Invalid URL — block it.
        return null;
      }
    }
    return originalOpen(url, ...rest as [string?, string?]);
  };

  return {
    dispose() {
      document.removeEventListener("click", clickHandler, true);
      window.open = originalOpen;
      installed = false;
    },
  };
}

async function showNavigationConfirm(url: string): Promise<void> {
  const origin = new URL(url).origin;
  const allowed = await kernelDialogs.confirm({
    title: "External Navigation",
    message: `A plugin is trying to navigate to ${origin}. Allow this navigation?`,
  });
  if (allowed) {
    originalOpen(url, "_blank", "noopener,noreferrer");
  }
}
