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
import { chatState, getActiveProjectId } from "./state.ts";
import { getModelsService, getPromptsService } from "./deps.ts";

function populatePromptSelect(
  sel: HTMLSelectElement,
  prompts: { id: string; name: string }[],
  defaultId?: string | null,
): void {
  sel.innerHTML = "";
  const noneOpt = document.createElement("option");
  noneOpt.value = "";
  noneOpt.textContent = "None";
  sel.appendChild(noneOpt);
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
    ctx.renderers.register({
      kinds: ["text"],
      canRender(part) {
        return part.type === "text" ? 1 : false;
      },
      mount(container, part) {
        const text = part.type === "text" ? part.text : "";
        container.innerHTML = sanitizedMarkdown(text);
        highlightCodeBlocks(container);
        return {
          update(p, isFinal) {
            const t = p.type === "text" ? p.text : "";
            container.innerHTML = sanitizedMarkdown(t);
            if (isFinal) highlightCodeBlocks(container);
          },
          unmount() {
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

    // Initialize thinking state from storage (before any rendering).
    const thinkingExpanded = await ctx.storage.get<boolean>("thinkingExpanded") ?? false;
    initThinkingState(thinkingExpanded, (v) => {
      ctx.storage.set("thinkingExpanded", v).catch(() => ctx.log.warn("Failed to save thinking preference"));
    });

    // Wire renderer pipeline to the streaming layer.
    setStreamRenderers(ctx.renderers);

    // Set up all event handlers.
    setupInputEvents();
    setupEventsPanelEvents();
    setupAttachmentEvents();
    setupTranscriptEvents();
    setupSidebarEvents();
    setupPanelEvents();
    setupInfiniteScroll();
    initCodeBlockCopyHandler(ctx.container, ctx.clipboard, ctx.timers);

    // Listen for cross-plugin events.
    ctx.events.on<{ modelId: string; availableModels: ModelEntry[] }>("model.changed", ({ modelId, availableModels }) => {
      const dom = getChatDom();
      populateModelSelect(dom.welcomeModel, availableModels, modelId);
      populateModelSelect(dom.panelModel, availableModels, modelId);
      syncRightPanelParams(modelId);
    });

    ctx.events.on<{ sessionId: string }>("sessions.selected", ({ sessionId }) => {
      ctx.log.debug?.(`Session selected from browser: ${sessionId}`);
    });

    ctx.events.on<{ prompts: { id: string; name: string }[]; defaultId: string | null }>("prompts.changed", ({ prompts, defaultId }) => {
      populatePromptSelect(getChatDom().welcomePrompt, prompts, defaultId);
    });

    ctx.events.on<{ enabled: boolean }>("settings.system_prompt_enabled", ({ enabled }) => {
      chatState.systemPromptEnabled = enabled;
    });

    // Pull initial state from services (events fired before our listeners existed).
    const models = getModelsService();
    if (models) {
      const modelId = models.getActiveModel();
      const available = models.getAvailableModels();
      const dom = getChatDom();
      populateModelSelect(dom.welcomeModel, available, modelId);
      populateModelSelect(dom.panelModel, available, modelId);
      syncRightPanelParams(modelId);
    }

    const promptsSvc = getPromptsService();
    if (promptsSvc) {
      populatePromptSelect(getChatDom().welcomePrompt, promptsSvc.getAll());
    }

    // Click Chat icon when already on chat mode → new conversation.
    // - From an active chat in project X → new chat in project X
    // - Already composing a draft in project X → switch to default
    // - Already composing a draft in default → stay in default (reset)
    const activityBar = document.getElementById("activity-bar");
    if (activityBar) {
      activityBar.addEventListener("click", (e) => {
        const btn = (e.target as Element).closest<HTMLElement>(".activity-btn");
        if (btn?.getAttribute("data-mode") === "chat" && ctx.mode.getActive() === "chat") {
          const isDraft = chatState.draftSession !== null && chatState.activeSessionId === null;
          const currentProject = getActiveProjectId();
          if (isDraft && currentProject) {
            // Already drafting in a project → cycle to default.
            startNewConversation(null);
          } else {
            // Active chat or default draft → new chat in current project (or default).
            startNewConversation(currentProject);
          }
        }
      });
    }

    // Load initial data and show welcome state.
    showWelcome();
    await loadSessions();

    ctx.log.info("Chat plugin ready.");
  },
};
