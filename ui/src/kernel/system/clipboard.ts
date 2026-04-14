/**
 * Clipboard write with graceful fallback for environments where
 * `navigator.clipboard` is unavailable (e.g. insecure origins, WebViews).
 */
export async function writeClipboardText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  if (typeof document === "undefined" || typeof document.execCommand !== "function") {
    throw new Error("Clipboard API is unavailable");
  }

  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.left = "-9999px";
  ta.style.opacity = "0";

  const prevSelection = document.getSelection();
  const prevRange = prevSelection && prevSelection.rangeCount > 0
    ? prevSelection.getRangeAt(0).cloneRange()
    : null;
  const prevFocused = document.activeElement as HTMLElement | null;

  document.body.appendChild(ta);
  ta.focus();
  ta.select();

  let copied = false;
  try {
    copied = document.execCommand("copy");
  } finally {
    ta.remove();
    if (prevSelection) {
      prevSelection.removeAllRanges();
      if (prevRange) prevSelection.addRange(prevRange);
    }
    prevFocused?.focus();
  }

  if (!copied) {
    throw new Error("Clipboard copy command failed");
  }
}
