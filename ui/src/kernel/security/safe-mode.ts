/**
 * Safe Mode — disables all third-party plugins.
 *
 * Entry points:
 * 1. ?safe=true query parameter
 * 2. Shift held during page load
 * 3. Auto-safe-mode after crash (sessionStorage flag)
 * 4. "Enter Safe Mode" button in Plugins panel
 * 5. Static HTML <a href="?safe=true"> in loading indicator
 */

const CRASH_FLAG = "talu.kernel.loading";

// --- Shift detection (self-installing, runs at module parse time) ---

let shiftHeldOnLoad = false;

function detectShiftOnLoad(): void {
  const handler = (e: KeyboardEvent) => {
    if (e.key === "Shift") shiftHeldOnLoad = true;
  };
  document.addEventListener("keydown", handler, { capture: true, once: true });
  window.addEventListener(
    "DOMContentLoaded",
    () => document.removeEventListener("keydown", handler, { capture: true } as EventListenerOptions),
    { once: true },
  );
}
detectShiftOnLoad();

// --- Public API ---

/** Check if safe mode should be active. */
export function isSafeMode(): boolean {
  // Check query parameter.
  const params = new URLSearchParams(window.location.search);
  if (params.get("safe") === "true") return true;

  // Check Shift held during page load.
  if (shiftHeldOnLoad) {
    console.warn("[kernel] Safe mode activated — Shift key held during page load.");
    return true;
  }

  // Check auto-safe-mode (previous session crashed during loading).
  if (sessionStorage.getItem(CRASH_FLAG) === "crashed") {
    sessionStorage.removeItem(CRASH_FLAG);
    console.warn("[kernel] Safe mode activated — previous session crashed during plugin loading.");
    return true;
  }

  return false;
}

/**
 * Set the crash detection flag. Call before loading third-party plugins.
 * If the page unloads during loading (crash), the flag persists
 * and triggers auto-safe-mode on next load.
 */
export function setLoadingFlag(): void {
  sessionStorage.setItem(CRASH_FLAG, "crashed");
}

/** Clear the crash detection flag. Call after successful boot. */
export function clearLoadingFlag(): void {
  sessionStorage.removeItem(CRASH_FLAG);
}

/** Navigate to safe mode. */
export function enterSafeMode(): void {
  const url = new URL(window.location.href);
  url.searchParams.set("safe", "true");
  window.location.href = url.toString();
}
