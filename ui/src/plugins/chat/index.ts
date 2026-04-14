/**
 * Chat plugin — core conversation experience.
 *
 * Renders into a Shadow DOM container like all other plugins. Builds its
 * own DOM (sidebar, transcript, input, right panel) programmatically.
 *
 * This plugin owns ALL conversation state, the sidebar, the right panel,
 * transcript events, and input handling. External plugins communicate via
 * the `talu.chat` service and kernel EventBus events.
 */

import type { PluginDefinition, PluginContext } from "../../kernel/types.ts";
import type { ModelEntry } from "../../types.ts";
import { createApiClient } from "../../api.ts";
import { sanitizedMarkdown } from "../../render/markdown.ts";
import { highlightCodeBlocks } from "../../render/highlight.ts";
import { initCodeBlockCopyHandler } from "../../render/transcript.ts";
import { initThinkingState, populateModelSelect } from "../../render/helpers.ts";
import { initChatDeps } from "./deps.ts";
import { buildChatDOM } from "./build-dom.ts";
import { initChatDom, getChatDom } from "./dom.ts";
import { setupInputEvents, cancelGeneration } from "./send.ts";
import { setupEventsPanelEvents } from "./events.ts";
import { setupAttachmentEvents } from "./attachments.ts";
import { setupTranscriptEvents } from "./transcript-events.ts";
import { showWelcome, startNewConversation } from "./welcome.ts";
import { selectChat } from "./selection.ts";
import { setStreamRenderers } from "./streaming.ts";
import { setupSidebarEvents } from "./sidebar-events.ts";
import { setupInfiniteScroll, loadSessions, refreshSidebar } from "./sidebar-list.ts";
import { syncRightPanelParams, setupPanelEvents } from "./panel-params.ts";
import { hideChatPanel } from "./panel-readonly.ts";
import { getChatPanelDom } from "./chat-panel-dom.ts";
import { chatState, getActiveProjectId, loadCollapsedGroups } from "./state.ts";
import { getModelsService, getPromptsService } from "./deps.ts";
import { initProjectStore, loadApiProjects, migrateLocalStorageProjects } from "../../render/project-combo.ts";
import { navigate, onRouteChange } from "../../kernel/system/router.ts";
import { initModelPicker } from "../../render/model-picker.ts";
import { preferences } from "../../kernel/system/preferences.ts";

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const val = bytes / Math.pow(1024, i);
  return `${val < 10 ? val.toFixed(1) : Math.round(val)} ${units[i]}`;
}

function populatePromptSelect(
  sel: HTMLSelectElement,
  prompts: { id: string; name: string }[],
  defaultId?: string | null,
): void {
  sel.innerHTML = "";
  for (const p of prompts) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.name;
    sel.appendChild(opt);
  }
  if (defaultId) sel.value = defaultId;
}

export const chatPlugin: PluginDefinition = {
  manifest: {
    id: "talu.chat",
    name: "Chat",
    version: "0.1.0",
    builtin: true,
    contributes: { mode: { key: "chat", label: "Chat" } },
  },

  register(ctx: PluginContext) {
    ctx.services.provide("talu.chat", {
      selectChat,
      startNewConversation,
      showWelcome,
      cancelGeneration,
      refreshSidebar,
      getSessions: () => chatState.sessions,
      getActiveSessionId: () => chatState.activeSessionId,
    });

    // Register the default text renderer using sanitizedMarkdown.
    // Third-party renderers scoring higher will override this.
    //
    // Two-tier rendering for streaming performance:
    //   Fast path (every rAF): append new delta as raw textContent span — O(delta)
    //   Slow path (throttled ~150ms): full sanitizedMarkdown re-render — O(N) but ~7/sec
    //   Final (isFinal=true): clean sanitizedMarkdown + highlightCodeBlocks
    ctx.renderers.register({
      kinds: ["text"],
      canRender(part) {
        return part.type === "text" ? 1 : false;
      },
      mount(container, part) {
        const text = part.type === "text" ? part.text : "";
        container.innerHTML = sanitizedMarkdown(text);
        highlightCodeBlocks(container);

        // Streaming state for two-tier rendering.
        // `currentText` is the mutable ref the slow-path timer reads so it
        // always renders the freshest accumulated text, not a stale capture.
        let currentText = text;
        let lastRenderedLen = text.length;
        let rawTail: HTMLSpanElement | null = null;
        let slowTimer: ReturnType<typeof setTimeout> | null = null;

        const fullRender = (t: string) => {
          if (rawTail) { rawTail.remove(); rawTail = null; }
          container.innerHTML = sanitizedMarkdown(t);
          lastRenderedLen = t.length;
        };

        return {
          update(p, isFinal) {
            const t = p.type === "text" ? p.text : "";
            currentText = t;
            if (isFinal) {
              if (slowTimer) { clearTimeout(slowTimer); slowTimer = null; }
              fullRender(t);
              highlightCodeBlocks(container);
              return;
            }
            // Fast path: append only the new delta as raw text.
            const delta = t.slice(lastRenderedLen);
            if (delta) {
              if (!rawTail) {
                rawTail = document.createElement("span");
                rawTail.className = "streaming-raw";
                container.appendChild(rawTail);
              }
              rawTail.textContent += delta;
              lastRenderedLen = t.length;
            }
            // Slow path: schedule a full markdown re-render.
            // Reads `currentText` at fire time so it always has the latest.
            if (!slowTimer) {
              slowTimer = setTimeout(() => {
                slowTimer = null;
                fullRender(currentText);
              }, 150);
            }
          },
          unmount() {
            if (slowTimer) { clearTimeout(slowTimer); slowTimer = null; }
            container.innerHTML = "";
          },
        };
      },
    });
  },

  async run(ctx: PluginContext, _signal: AbortSignal) {
    // Build DOM into shadow root container.
    buildChatDOM(ctx.container);
    initChatDom(ctx.container);

    // Initialize shared dependencies for all chat modules.
    const api = createApiClient((url, init) => ctx.network.fetch(url, init));
    initProjectStore(api);
    initChatDeps({
      api,
      notifications: ctx.notifications,
      services: ctx.services,
      events: ctx.events,
      layout: ctx.layout,
      clipboard: ctx.clipboard,
      download: ctx.download,
      timers: ctx.timers,
      observe: ctx.observe,
      format: ctx.format,
      upload: ctx.upload,
      hooks: ctx.hooks,
      menus: ctx.menus,
    });

    // Load collapsed sidebar groups from KV before first render.
    await loadCollapsedGroups();

    // Initialize thinking state from unified preferences (before any rendering).
    const thinkingExpanded = preferences.get<boolean>("talu.chat", "thinking_expanded") ?? false;
    initThinkingState(thinkingExpanded, (v) => {
      preferences.set("talu.chat", "thinking_expanded", v);
    });

    // Wire renderer pipeline to the streaming layer.
    setStreamRenderers(ctx.renderers);

    // Set up all event handlers.
    setupInputEvents();
    setupEventsPanelEvents();
    // Wire up HTTP panel copy button.
    const pd = getChatPanelDom();
    pd.panelHttpCopy.addEventListener("click", () => {
      const text = pd.panelHttpCurl.textContent ?? "";
      if (text && text !== "No request yet") {
        void ctx.clipboard.writeText(text).then(
          () => {
            pd.panelHttpCopy.title = "Copied!";
            setTimeout(() => { pd.panelHttpCopy.title = "Copy curl"; }, 1500);
          },
          () => {
            pd.panelHttpCopy.title = "Copy failed";
            setTimeout(() => { pd.panelHttpCopy.title = "Copy curl"; }, 1500);
          },
        );
      }
    });
    setupAttachmentEvents();
    setupTranscriptEvents();
    setupSidebarEvents();
    setupPanelEvents();
    setupInfiniteScroll();
    initCodeBlockCopyHandler(ctx.container, ctx.clipboard, ctx.timers);

    // Welcome settings gear → toggle inline advanced options.
    // The advanced section is absolutely positioned below the input so it
    // doesn't shift the centered input container.
    {
      const dom = getChatDom();
      const positionAdvanced = () => {
        const inputContainer = dom.welcomeInput.closest(".input-container");
        if (!inputContainer) return;
        const inputRect = inputContainer.getBoundingClientRect();
        const parentRect = dom.welcomeState.getBoundingClientRect();
        dom.welcomeAdvanced.style.top = `${inputRect.bottom - parentRect.top + 8}px`;
      };
      const showAdvancedPanel = (panel: "sampling" | "generation", toggle = false) => {
        const showPanel = panel === "sampling" ? dom.welcomePanelSampling : dom.welcomePanelGeneration;
        const hidePanel = panel === "sampling" ? dom.welcomePanelGeneration : dom.welcomePanelSampling;

        const containerHidden = dom.welcomeAdvanced.classList.contains("hidden");
        const panelAlreadyVisible = !showPanel.classList.contains("hidden");

        if (toggle && !containerHidden && panelAlreadyVisible) {
          dom.welcomeAdvanced.classList.add("hidden");
          dom.welcomeGeneration.classList.remove("active");
          return;
        }

        showPanel.classList.remove("hidden");
        hidePanel.classList.add("hidden");
        dom.welcomeGeneration.classList.toggle("active", panel === "generation");
        dom.welcomeAdvanced.classList.remove("hidden");
        positionAdvanced();
      };
      const dismissAdvanced = () => {
        dom.welcomeAdvanced.classList.add("hidden");
        dom.welcomeGeneration.classList.remove("active");
      };
      dom.welcomeGeneration.addEventListener("mousedown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        showAdvancedPanel("generation", true);
      });
      window.addEventListener("resize", () => {
        if (!dom.welcomeAdvanced.classList.contains("hidden")) positionAdvanced();
      });

      // Clicks inside the advanced panel should not dismiss it.
      dom.welcomeAdvanced.addEventListener("mousedown", (e) => {
        e.stopPropagation();
      });

      // Close settings when clicking outside the advanced panel and trigger area.
      document.addEventListener("mousedown", (e) => {
        if (dom.welcomeAdvanced.classList.contains("hidden")) return;
        const target = e.target as Node;
        if (dom.welcomeGeneration.contains(target)) return;
        // Don't dismiss if clicking inside the model picker (it has its own dismiss).
        const wrap = dom.welcomeModel.closest(".model-select-wrap");
        if (wrap?.contains(target)) return;
        dismissAdvanced();
      });

      // Initialize custom model picker — settings icon opens sampling panel.
      initModelPicker(dom.welcomeModel, {
        onSettings: () => showAdvancedPanel("sampling", true),
        onDismiss: dismissAdvanced,
      });
    }

    // Gray out sampling controls when no model is selected.
    const syncSamplingDisabled = (hasModel: boolean) => {
      const controls = getChatDom().welcomeSamplingControls;
      if (hasModel) {
        controls.classList.remove("disabled");
      } else {
        controls.classList.add("disabled");
      }
    };

    // Listen for cross-plugin events.
    ctx.events.on<{ modelId: string; availableModels: ModelEntry[] }>("model.changed", ({ modelId, availableModels }) => {
      const dom = getChatDom();
      const pd = getChatPanelDom();
      populateModelSelect(dom.welcomeModel, availableModels, modelId);
      populateModelSelect(pd.panelModel, availableModels, modelId);
      syncRightPanelParams(modelId);
      syncSamplingDisabled(!!modelId);

      // Render variant pills for the active model's family.
      const entry = availableModels.find((m) => m.id === modelId)
        ?? availableModels.find((m) => m.variants?.some((v) => v.id === modelId));
      if (entry?.variants && entry.variants.length > 0) {
        dom.welcomeVariantRow.classList.remove("hidden");
        dom.welcomeVariantPills.innerHTML = "";
        const multiVariant = entry.variants.length > 1;
        for (const v of entry.variants) {
          const pill = document.createElement("button");
          pill.className = "welcome-variant-pill";
          // Format: "GAF4 · 2.5 GB" or just label
          let text = v.label;
          if (v.size_bytes && v.size_bytes > 0) {
            text += ` \u00b7 ${formatBytes(v.size_bytes)}`;
          }
          pill.textContent = text;
          pill.title = v.id;
          if (v.id === modelId) pill.classList.add("active");
          if (multiVariant) {
            pill.addEventListener("click", () => {
              ctx.events.emit("repo.selectModel", { modelId: v.id });
              // Update active pill visually.
              for (const p of dom.welcomeVariantPills.children) {
                (p as HTMLElement).classList.toggle("active", p === pill);
              }
            });
          }
          dom.welcomeVariantPills.appendChild(pill);
        }
      } else {
        dom.welcomeVariantRow.classList.add("hidden");
      }
    });

    ctx.events.on<{ sessionId: string }>("sessions.selected", ({ sessionId }) => {
      ctx.log.debug?.(`Session selected from browser: ${sessionId}`);
    });

    ctx.events.on<{ prompts: { id: string; name: string }[]; defaultId: string | null }>("prompts.changed", ({ prompts, defaultId }) => {
      populatePromptSelect(getChatDom().welcomePrompt, prompts, defaultId);
    });

    ctx.events.on<{ enabled: boolean }>("settings.system_prompt_enabled", ({ enabled }) => {
      chatState.systemPromptEnabled = enabled;
      getChatDom().welcomePromptEnabled.checked = enabled;
      getChatDom().welcomePrompt.disabled = !enabled;
    });

    // Variant pill in repo plugin → open a new chat with that model.
    ctx.events.on("repo.openChat", () => {
      ctx.mode.switch("chat");
      startNewConversation(getActiveProjectId());
    });

    // Close the chat panel when leaving chat mode.
    ctx.mode.onChange(({ from }) => {
      if (from === "chat") hideChatPanel();
    });

    getChatPanelDom().panelModelEdit.addEventListener("click", () => {
      navigate({ mode: "routing", sub: null, resource: null });
    });
    getChatDom().welcomePromptEdit.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      navigate({ mode: "conversations", sub: "context", resource: null });
    });

    // Sync system prompt toggle with global setting.
    getChatDom().welcomePromptEnabled.checked = chatState.systemPromptEnabled;
    getChatDom().welcomePrompt.disabled = !chatState.systemPromptEnabled;
    getChatDom().welcomePromptEnabled.addEventListener("change", () => {
      const enabled = getChatDom().welcomePromptEnabled.checked;
      chatState.systemPromptEnabled = enabled;
      getChatDom().welcomePrompt.disabled = !enabled;
    });

    // Pull initial state from services (events fired before our listeners existed).
    const models = getModelsService();
    if (models) {
      const modelId = models.getActiveModel();
      const available = models.getAvailableModels();
      populateModelSelect(getChatDom().welcomeModel, available, modelId);
      populateModelSelect(getChatPanelDom().panelModel, available, modelId);
      syncRightPanelParams(modelId);
      syncSamplingDisabled(!!modelId);
    } else {
      syncSamplingDisabled(false);
    }

    const promptsSvc = getPromptsService();
    if (promptsSvc) {
      populatePromptSelect(getChatDom().welcomePrompt, promptsSvc.getAll());
    }

    // Click Chat icon when already on chat mode → new conversation in current project.
    const activityBar = document.getElementById("activity-bar");
    if (activityBar) {
      activityBar.addEventListener("click", (e) => {
        const btn = (e.target as Element).closest<HTMLElement>(".activity-btn");
        if (btn?.getAttribute("data-mode") === "chat" && ctx.mode.getActive() === "chat") {
          startNewConversation(getActiveProjectId());
        }
      });
    }

    // Prime project cache from API so empty projects appear in sidebar.
    await loadApiProjects().catch(() => {});

    // Load initial data and show welcome state.
    showWelcome();
    await loadSessions();

    // Route-driven session selection (deep links and Back/Forward).
    ctx.subscriptions.add(onRouteChange((route) => {
      if (route.mode !== "chat") return;
      const sessionId = route.sub;
      if (!sessionId) {
        // #/chat → show welcome (new conversation).
        if (chatState.activeSessionId) {
          startNewConversation(getActiveProjectId());
        }
      } else if (sessionId !== chatState.activeSessionId && !sessionId.startsWith("__pending_")) {
        // #/chat/<id> → select that session.
        selectChat(sessionId);
      }
    }));

    // Migrate localStorage projects → API (one-time, non-blocking).
    migrateLocalStorageProjects().catch(() => {});

    ctx.log.info("Chat plugin ready.");
  },
};
