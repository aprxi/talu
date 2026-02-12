/**
 * Standard Dialogs — Kernel-managed confirm/alert/prompt/select overlays.
 *
 * Rendered at document root level (outside shadow roots) so they are not
 * clipped by plugin slot containers. FIFO queue — one dialog at a time.
 * Focus trapped in dialog. z-index: 2000 (above popovers, below notifications).
 */

import type { StandardDialogs } from "../types.ts";

interface DialogOptions {
  title: string;
  message: string;
  destructive?: boolean;
  defaultValue?: string;
  items?: { id: string; label: string; description?: string }[];
}

type DialogResolver = (value: unknown) => void;

interface QueueEntry {
  options: DialogOptions;
  type: "confirm" | "alert" | "prompt" | "select";
  resolve: DialogResolver;
}

let overlay: HTMLElement | null = null;
const queue: QueueEntry[] = [];
let active = false;

function getOverlay(): HTMLElement {
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "kernel-dialog-overlay";
    overlay.style.cssText =
      "position:fixed;inset:0;z-index:2000;display:flex;align-items:center;" +
      "justify-content:center;background:rgba(0,0,0,0.5);";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
  }
  return overlay;
}

function createDialogDOM(options: DialogOptions, type: string): HTMLElement {
  const dialog = document.createElement("div");
  dialog.style.cssText =
    "background:var(--bg-secondary,#1e1e2e);color:var(--text,#cdd6f4);" +
    "border:1px solid var(--border,#45475a);border-radius:8px;padding:1.5rem;" +
    "min-width:320px;max-width:480px;font-family:var(--font-family,system-ui);";

  const title = document.createElement("h3");
  title.textContent = options.title;
  title.style.cssText = "margin:0 0 0.75rem;font-size:1rem;font-weight:600;";
  dialog.appendChild(title);

  const msg = document.createElement("p");
  msg.textContent = options.message;
  msg.style.cssText = "margin:0 0 1.25rem;font-size:0.875rem;color:var(--text-muted,#a6adc8);";
  dialog.appendChild(msg);

  if (type === "prompt") {
    const input = document.createElement("input");
    input.type = "text";
    input.value = options.defaultValue ?? "";
    input.className = "form-input";
    input.id = "kernel-dialog-input";
    input.style.cssText = "width:100%;margin-bottom:1rem;padding:0.5rem;border-radius:4px;" +
      "border:1px solid var(--border,#45475a);background:var(--bg,#1e1e2e);color:var(--text,#cdd6f4);";
    dialog.appendChild(input);
  }

  if (type === "select" && options.items) {
    const list = document.createElement("div");
    list.id = "kernel-dialog-select-list";
    list.style.cssText = "max-height:200px;overflow-y:auto;margin-bottom:1rem;";
    for (const item of options.items) {
      const btn = document.createElement("button");
      btn.dataset["selectId"] = item.id;
      btn.textContent = item.label;
      btn.style.cssText =
        "display:block;width:100%;text-align:left;padding:0.5rem;border:none;" +
        "background:transparent;color:var(--text,#cdd6f4);cursor:pointer;border-radius:4px;";
      btn.addEventListener("mouseenter", () => { btn.style.background = "var(--bg-hover,#313244)"; });
      btn.addEventListener("mouseleave", () => { btn.style.background = "transparent"; });
      list.appendChild(btn);
    }
    dialog.appendChild(list);
  }

  return dialog;
}

function createButtons(
  dialog: HTMLElement,
  type: string,
  destructive: boolean,
): { ok: HTMLButtonElement; cancel?: HTMLButtonElement } {
  const row = document.createElement("div");
  row.style.cssText = "display:flex;gap:0.5rem;justify-content:flex-end;";

  let cancel: HTMLButtonElement | undefined;
  if (type !== "alert") {
    cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    cancel.style.cssText =
      "padding:0.375rem 0.75rem;border-radius:4px;border:1px solid var(--border,#45475a);" +
      "background:transparent;color:var(--text,#cdd6f4);cursor:pointer;";
    row.appendChild(cancel);
  }

  const ok = document.createElement("button");
  ok.textContent = type === "alert" ? "OK" : type === "confirm" ? (destructive ? "Delete" : "Confirm") : "OK";
  ok.style.cssText =
    `padding:0.375rem 0.75rem;border-radius:4px;border:none;cursor:pointer;color:#fff;` +
    `background:${destructive ? "var(--danger,#f38ba8)" : "var(--accent,#89b4fa)"};`;
  row.appendChild(ok);

  dialog.appendChild(row);
  return { ok, cancel };
}

function trapFocus(container: HTMLElement): (e: KeyboardEvent) => void {
  return (e: KeyboardEvent) => {
    if (e.key !== "Tab") return;
    const focusable = container.querySelectorAll<HTMLElement>(
      'button, input, select, textarea, [tabindex]:not([tabindex="-1"])',
    );
    if (focusable.length === 0) return;
    const first = focusable[0]!;
    const last = focusable[focusable.length - 1]!;
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };
}

function processQueue(): void {
  if (active || queue.length === 0) return;
  active = true;

  const entry = queue.shift()!;
  const ov = getOverlay();
  ov.innerHTML = "";

  const dialog = createDialogDOM(entry.options, entry.type);
  const { ok, cancel } = createButtons(dialog, entry.type, entry.options.destructive ?? false);
  ov.appendChild(dialog);
  document.body.appendChild(ov);

  const focusTrap = trapFocus(dialog);
  document.addEventListener("keydown", focusTrap);

  function cleanup(): void {
    document.removeEventListener("keydown", focusTrap);
    ov.remove();
    active = false;
    processQueue();
  }

  // Handle Escape.
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      document.removeEventListener("keydown", onKey);
      cleanup();
      if (entry.type === "confirm") entry.resolve(false);
      else if (entry.type === "prompt" || entry.type === "select") entry.resolve(null);
      else entry.resolve(undefined);
    }
  };
  document.addEventListener("keydown", onKey);

  // Handle select list clicks.
  if (entry.type === "select") {
    const list = dialog.querySelector("#kernel-dialog-select-list");
    list?.addEventListener("click", (e) => {
      const target = (e.target as HTMLElement).closest<HTMLElement>("[data-select-id]");
      if (target) {
        document.removeEventListener("keydown", onKey);
        cleanup();
        entry.resolve(target.dataset["selectId"]!);
      }
    });
  }

  ok.addEventListener("click", () => {
    document.removeEventListener("keydown", onKey);
    if (entry.type === "confirm") {
      cleanup();
      entry.resolve(true);
    } else if (entry.type === "prompt") {
      const input = dialog.querySelector<HTMLInputElement>("#kernel-dialog-input");
      cleanup();
      entry.resolve(input?.value ?? null);
    } else {
      cleanup();
      entry.resolve(undefined);
    }
  });

  cancel?.addEventListener("click", () => {
    document.removeEventListener("keydown", onKey);
    cleanup();
    if (entry.type === "confirm") entry.resolve(false);
    else entry.resolve(null);
  });

  // Focus first button.
  (cancel ?? ok).focus();
}

export class StandardDialogsImpl implements StandardDialogs {
  private pluginName: string;

  constructor(pluginName: string) {
    this.pluginName = pluginName;
  }

  confirm(options: { title: string; message: string; destructive?: boolean }): Promise<boolean> {
    return new Promise((resolve) => {
      const title = options.destructive
        ? `${this.pluginName}: ${options.title}`
        : options.title;
      queue.push({ options: { ...options, title }, type: "confirm", resolve: resolve as DialogResolver });
      processQueue();
    });
  }

  alert(options: { title: string; message: string }): Promise<void> {
    return new Promise((resolve) => {
      queue.push({ options, type: "alert", resolve: resolve as DialogResolver });
      processQueue();
    });
  }

  prompt(options: { title: string; message: string; defaultValue?: string }): Promise<string | null> {
    return new Promise((resolve) => {
      queue.push({ options, type: "prompt", resolve: resolve as DialogResolver });
      processQueue();
    });
  }

  select(options: {
    title: string;
    items: { id: string; label: string; description?: string }[];
  }): Promise<string | null> {
    return new Promise((resolve) => {
      queue.push({ options: { ...options, message: "" }, type: "select", resolve: resolve as DialogResolver });
      processQueue();
    });
  }
}
