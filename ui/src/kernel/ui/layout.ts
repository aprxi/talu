/**
 * Shadow DOM slot creation for plugin view containers.
 *
 * CSS custom properties (--text, --bg, --accent, etc.) inherit through shadow
 * boundaries automatically. Class-based selectors are shared via a constructed
 * stylesheet applied through adoptedStyleSheets (synchronous, no FOUC).
 */

let sharedSheet: CSSStyleSheet | null = null;

/** Access the pre-loaded shared stylesheet (for popover/renderer shadow roots). */
export function getSharedStylesheet(): CSSStyleSheet | null {
  return sharedSheet;
}

/**
 * Pre-load the shared stylesheet text and create a constructed CSSStyleSheet.
 * Must be called once during bootKernel() before any plugins register.
 */
export async function initSharedStylesheet(): Promise<void> {
  try {
    const resp = await fetch("/assets/style.css");
    if (!resp.ok) {
      console.warn("[kernel] Failed to load shared stylesheet — shadow roots will lack styles.");
      return;
    }
    const text = await resp.text();
    sharedSheet = new CSSStyleSheet();
    sharedSheet.replaceSync(text);
  } catch {
    console.warn("[kernel] Failed to load shared stylesheet — shadow roots will lack styles.");
  }
}

/**
 * Create a Shadow DOM container inside `hostElement` for a plugin to render into.
 * Clears existing content, attaches shadow root, injects stylesheet, returns container.
 */
export function createPluginSlot(pluginId: string, hostElement: HTMLElement): HTMLElement {
  // Clear static HTML (e.g., the settings form from index.html).
  hostElement.innerHTML = "";

  // Create a child div that owns the shadow root.
  const shadowHost = document.createElement("div");
  shadowHost.dataset["pluginId"] = pluginId;
  shadowHost.style.flex = "1";
  shadowHost.style.minWidth = "0";
  shadowHost.style.minHeight = "0";
  shadowHost.style.overflow = "hidden";
  hostElement.appendChild(shadowHost);

  const shadowRoot = shadowHost.attachShadow({ mode: "open" });

  // Apply the shared stylesheet via adoptedStyleSheets (synchronous, no network request).
  if (sharedSheet) {
    shadowRoot.adoptedStyleSheets = [sharedSheet];
  }

  // Inner container that the plugin renders into — fills the shadow host.
  const container = document.createElement("div");
  container.className = "plugin-container";
  container.style.width = "100%";
  container.style.height = "100%";
  shadowRoot.appendChild(container);

  return container;
}
