/**
 * Sensitive API Interception — always-on gating for non-network exfil surfaces.
 *
 * Wraps: clipboard writes, download triggers, Notification API, window.print().
 * Each shows a Kernel confirm dialog with operation description.
 * Not dev-mode toggleable — these run in production.
 */

import type { Disposable } from "../types.ts";

let installed = false;

/**
 * Install sensitive API interception. Call once during kernel boot.
 * Returns a Disposable to restore original APIs.
 */
export function installSensitiveApiInterception(): Disposable {
  if (installed) return { dispose() {} };
  installed = true;

  const cleanups: (() => void)[] = [];

  // --- Clipboard ---
  if (navigator.clipboard) {
    const originalWriteText = navigator.clipboard.writeText.bind(navigator.clipboard);

    navigator.clipboard.writeText = async function (data: string): Promise<void> {
      // Built-in Kernel operations bypass the dialog by calling originalWriteText directly.
      // For plugin code, show a confirm.
      const preview = data.length > 200 ? data.slice(0, 200) + "..." : data;
      const allowed = window.confirm(
        `A plugin wants to copy data to your clipboard.\n\nPreview: "${preview}"\n\nAllow?`,
      );
      if (!allowed) throw new DOMException("User denied clipboard write", "NotAllowedError");
      return originalWriteText(data);
    };

    cleanups.push(() => {
      navigator.clipboard.writeText = originalWriteText;
    });
  }

  // --- Downloads (capture-phase click handler for <a download>) ---
  const downloadHandler = (e: MouseEvent) => {
    const path = e.composedPath();
    for (const el of path) {
      if (el instanceof HTMLAnchorElement && el.hasAttribute("download")) {
        const filename = el.download || "file";
        const allowed = window.confirm(
          `A plugin wants to save a file ("${filename}").\n\nAllow?`,
        );
        if (!allowed) {
          e.preventDefault();
          e.stopPropagation();
        }
        return;
      }
    }
  };
  document.addEventListener("click", downloadHandler, true);
  cleanups.push(() => document.removeEventListener("click", downloadHandler, true));

  // --- Notifications ---
  if (typeof Notification !== "undefined") {
    const originalRequestPermission = Notification.requestPermission.bind(Notification);

    Notification.requestPermission = async function (
      ...args: Parameters<typeof Notification.requestPermission>
    ): Promise<NotificationPermission> {
      const allowed = window.confirm(
        "A plugin wants to show system notifications.\n\nAllow?",
      );
      if (!allowed) return "denied";
      return originalRequestPermission(...args);
    };

    cleanups.push(() => {
      Notification.requestPermission = originalRequestPermission;
    });
  }

  // --- window.print ---
  const originalPrint = window.print.bind(window);
  window.print = function (): void {
    const allowed = window.confirm(
      "A plugin wants to open the print dialog.\n\nAllow?",
    );
    if (allowed) originalPrint();
  };
  cleanups.push(() => {
    window.print = originalPrint;
  });

  return {
    dispose() {
      for (const cleanup of cleanups) {
        try {
          cleanup();
        } catch {
          // Ignore cleanup errors.
        }
      }
      installed = false;
    },
  };
}
