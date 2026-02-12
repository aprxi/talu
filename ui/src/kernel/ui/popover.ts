/**
 * Popover Layer — Kernel-managed popovers rendered at document root level.
 *
 * z-index 1000 (below dialogs at 2000, below notifications at 3000).
 * One popover per plugin — showing a new one dismisses the previous.
 * Content wrapped in Shadow DOM for isolation.
 */

import type { Disposable } from "../types.ts";
import { getSharedStylesheet } from "./layout.ts";

interface PopoverOptions {
  anchor: HTMLElement;
  content: HTMLElement;
  placement?: "top" | "bottom" | "left" | "right";
}

/** Track active popovers per plugin (one at a time). */
const activePopovers = new Map<string, Disposable>();

const GAP = 4;

export class PopoverManager {
  showPopover(pluginId: string, options: PopoverOptions): Disposable {
    // Dismiss existing popover for this plugin.
    activePopovers.get(pluginId)?.dispose();

    // Root-level wrapper.
    const wrapper = document.createElement("div");
    wrapper.dataset["popoverOwner"] = pluginId;
    wrapper.style.cssText = "position:fixed;z-index:1000;";

    // Shadow DOM container for content isolation.
    const shadowHost = document.createElement("div");
    wrapper.appendChild(shadowHost);
    const shadowRoot = shadowHost.attachShadow({ mode: "open" });

    const sheet = getSharedStylesheet();
    if (sheet) {
      shadowRoot.adoptedStyleSheets = [sheet];
    }
    shadowRoot.appendChild(options.content);

    document.body.appendChild(wrapper);

    // Positioning.
    const placement = options.placement ?? "bottom";

    const reposition = () => {
      const anchorRect = options.anchor.getBoundingClientRect();
      positionPopover(wrapper, anchorRect, placement);
    };
    reposition();

    // Reposition on scroll/resize.
    const onScroll = () => reposition();
    const onResize = () => reposition();
    window.addEventListener("scroll", onScroll, { capture: true, passive: true });
    window.addEventListener("resize", onResize, { passive: true });

    // Click-outside dismissal (delayed to avoid same-click dismiss).
    let clickOutsideHandler: ((e: MouseEvent) => void) | null = null;
    requestAnimationFrame(() => {
      clickOutsideHandler = (e: MouseEvent) => {
        const target = e.target as Node;
        if (!wrapper.contains(target) && !options.anchor.contains(target)) {
          disposable.dispose();
        }
      };
      document.addEventListener("click", clickOutsideHandler, { capture: true });
    });

    // Escape dismissal.
    const onEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        // Only dismiss if no dialog is open (dialogs handle Escape at z-index 2000).
        const hasDialog = document.querySelector("[role='dialog']");
        if (!hasDialog) {
          disposable.dispose();
        }
      }
    };
    document.addEventListener("keydown", onEscape);

    const disposable: Disposable = {
      dispose: () => {
        wrapper.remove();
        window.removeEventListener("scroll", onScroll, { capture: true } as EventListenerOptions);
        window.removeEventListener("resize", onResize);
        if (clickOutsideHandler) {
          document.removeEventListener("click", clickOutsideHandler, { capture: true } as EventListenerOptions);
        }
        document.removeEventListener("keydown", onEscape);
        if (activePopovers.get(pluginId) === disposable) {
          activePopovers.delete(pluginId);
        }
      },
    };

    activePopovers.set(pluginId, disposable);
    return disposable;
  }
}

function positionPopover(
  wrapper: HTMLElement,
  anchorRect: DOMRect,
  placement: "top" | "bottom" | "left" | "right",
): void {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  let top: number;
  let left: number;

  switch (placement) {
    case "bottom":
      top = anchorRect.bottom + GAP;
      left = anchorRect.left;
      break;
    case "top":
      top = anchorRect.top - wrapper.offsetHeight - GAP;
      left = anchorRect.left;
      break;
    case "right":
      top = anchorRect.top;
      left = anchorRect.right + GAP;
      break;
    case "left":
      top = anchorRect.top;
      left = anchorRect.left - wrapper.offsetWidth - GAP;
      break;
  }

  // Clamp to viewport.
  top = Math.max(0, Math.min(top, vh - wrapper.offsetHeight));
  left = Math.max(0, Math.min(left, vw - wrapper.offsetWidth));

  wrapper.style.top = `${top}px`;
  wrapper.style.left = `${left}px`;
}
