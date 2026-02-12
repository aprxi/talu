import { el } from "./helpers.ts";

export function renderEmptyState(text: string): HTMLElement {
  const node = el("div", "empty-state", text);
  node.style.minHeight = "200px";
  node.dataset["emptyState"] = "";
  return node;
}

export function renderLoadingSpinner(): HTMLElement {
  const wrapper = el("div", "empty-state");
  wrapper.innerHTML = `<div class="spinner"></div>`;
  return wrapper;
}

export function renderToast(
  message: string,
  type: "error" | "success" | "info" | "warning",
): HTMLElement {
  const toast = el("div", "toast", message);
  toast.setAttribute("role", "alert");
  toast.setAttribute("aria-live", "assertive");

  const styles: Record<string, { bg: string; color: string }> = {
    error:   { bg: "color-mix(in srgb, var(--danger) 90%, transparent)", color: "white" },
    success: { bg: "color-mix(in srgb, var(--success) 90%, transparent)", color: "var(--bg)" },
    warning: { bg: "color-mix(in srgb, #f59e0b 90%, transparent)", color: "var(--bg)" },
    info:    { bg: "color-mix(in srgb, var(--accent) 90%, transparent)", color: "white" },
  };
  const s = styles[type]!;
  toast.style.background = s.bg;
  toast.style.color = s.color;
  return toast;
}
