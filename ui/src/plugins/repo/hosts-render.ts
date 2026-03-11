/** Render the Hosts section in the Router view. */

import { repoState } from "./state.ts";
import { getRepoDom } from "./dom.ts";

const TERMINAL_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m4 17 6-6-6-6"/><path d="M12 19h8"/></svg>`;

const SERVER_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="8" x="2" y="2" rx="2" ry="2"/><rect width="20" height="8" x="2" y="14" rx="2" ry="2"/><line x1="6" x2="6.01" y1="6" y2="6"/><line x1="6" x2="6.01" y1="18" y2="18"/></svg>`;

export function renderHosts(): void {
  const dom = getRepoDom();
  dom.hostsList.innerHTML = "";

  for (const host of repoState.hosts) {
    const row = document.createElement("div");
    row.className = "repo-host-row";
    row.style.cssText =
      "display:inline-flex;align-items:center;gap:0.5rem;padding:0.375rem 0.625rem;" +
      "border:1px solid var(--border);border-radius:6px;margin-bottom:0.375rem;margin-right:0.375rem;";

    // Server icon
    const icon = document.createElement("span");
    icon.style.cssText = "color:var(--text-muted);flex-shrink:0;display:flex;";
    icon.innerHTML = SERVER_ICON;

    // Label
    const label = document.createElement("span");
    label.style.cssText = "font-size:13px;color:var(--text);font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;";
    label.textContent = host.label;

    row.append(icon, label);

    // Primary badge
    if (host.primary) {
      const badge = document.createElement("span");
      badge.style.cssText =
        "font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.03em;" +
        "padding:1px 5px;border-radius:3px;flex-shrink:0;" +
        "background:color-mix(in srgb, var(--primary) 15%, transparent);color:var(--primary);";
      badge.textContent = "primary";
      row.appendChild(badge);
    }

    // Terminal button — right next to the label
    const termBtn = document.createElement("button");
    termBtn.className = "btn btn-ghost btn-icon";
    termBtn.title = "Terminal";
    termBtn.style.cssText = "padding:0.125rem;color:var(--text-muted);min-width:0;";
    termBtn.dataset["action"] = "open-terminal";
    termBtn.dataset["hostId"] = host.id;
    termBtn.innerHTML = TERMINAL_ICON;
    row.appendChild(termBtn);

    dom.hostsList.appendChild(row);
  }

  // Test bench button for process WebSocket.
  const testRow = document.createElement("div");
  testRow.style.cssText = "margin-top:0.5rem;";
  const testBtn = document.createElement("button");
  testBtn.className = "btn btn-ghost btn-sm";
  testBtn.textContent = "Test Process WS";
  testBtn.dataset["action"] = "test-process-ws";
  testRow.appendChild(testBtn);
  dom.hostsList.appendChild(testRow);
}

export function wireHostEvents(
  container: HTMLElement,
  onOpenTerminal: (hostId: string) => void,
  onTestProcessWs?: () => void,
): void {
  container.addEventListener("click", (e) => {
    const target = (e.target as Element).closest<HTMLElement>("[data-action]");
    if (!target) return;

    const action = target.dataset["action"];
    const hostId = target.dataset["hostId"];
    if (action === "open-terminal" && hostId) {
      onOpenTerminal(hostId);
    } else if (action === "test-process-ws" && onTestProcessWs) {
      onTestProcessWs();
    }
  });
}
