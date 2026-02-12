/**
 * Notifications — non-blocking toast messages with four severity levels.
 *
 * Rendered at z-index 3000 (above dialogs at 2000, above popovers at 1000).
 */

import type { Notifications } from "../types.ts";
import { renderToast } from "../../render/common.ts";

function showToast(message: string, type: "error" | "success" | "info" | "warning"): void {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = renderToast(message, type);
  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("visible"));
  setTimeout(() => {
    toast.classList.remove("visible");
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

export class NotificationsImpl implements Notifications {
  info(message: string): void {
    showToast(message, "info");
  }

  success(message: string): void {
    showToast(message, "success");
  }

  warning(message: string): void {
    showToast(message, "warning");
  }

  error(message: string): void {
    showToast(message, "error");
  }
}
