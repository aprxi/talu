import { describe, test, expect, beforeEach } from "bun:test";
import {
  showReadOnlyParams,
  restoreEditableParams,
  hideRightPanel,
  handleToggleTuning,
  isPanelReadOnly,
} from "../../../src/plugins/chat/panel-readonly.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { createDomRoot, CHAT_DOM_IDS, CHAT_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import type { GenerationSettings, UsageStats } from "../../../src/types.ts";

/**
 * Tests for panel-readonly — read-only mode for viewing past message
 * generation parameters and usage stats.
 */

let domRoot: HTMLElement;

beforeEach(() => {
  const notifs = mockNotifications();
  domRoot = createDomRoot(CHAT_DOM_IDS, undefined, CHAT_DOM_TAGS);

  // Add a tuning button inside transcript for hideRightPanel/handleToggleTuning
  const transcript = domRoot.querySelector("#transcript")!;
  const tuningBtn = document.createElement("button");
  tuningBtn.dataset["action"] = "toggle-tuning";
  transcript.appendChild(tuningBtn);

  initChatDom(domRoot);

  chatState.activeChat = null;
  chatState.activeSessionId = null;
  chatState.sessions = [];

  initChatDeps({
    api: {} as any,
    notifications: notifs.mock,
    timers: mockTimers(),
    services: {
      get: () => ({
        getActiveModel: () => "gpt-4",
        getAvailableModels: () => [],
        getPromptNameById: () => null,
      }),
    } as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    hooks: { run: async (_name: string, payload: any) => payload } as any,
    clipboard: { writeText: async () => {} } as any,
    download: {} as any,
    observe: { onResize: () => ({ dispose() {} }) } as any,
    format: { dateTime: () => "2024-01-01" } as any,
    upload: { upload: async () => ({}) } as any,
    menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
  });

  // Reset read-only mode between tests
  if (isPanelReadOnly()) restoreEditableParams();
});

// ── showReadOnlyParams ──────────────────────────────────────────────────────

describe("showReadOnlyParams", () => {
  test("no-op when gen is null", () => {
    showReadOnlyParams(null, null);
    expect(isPanelReadOnly()).toBe(false);
  });

  test("sets panelReadOnlyMode to true", () => {
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, null);
    expect(isPanelReadOnly()).toBe(true);
  });

  test("opens right panel if hidden", () => {
    const dom = getChatDom();
    dom.rightPanel.classList.add("hidden");
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, null);
    expect(dom.rightPanel.classList.contains("hidden")).toBe(false);
    expect(dom.rightPanel.classList.contains("flex")).toBe(true);
  });

  test("adds read-only class to right panel", () => {
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, null);
    const dom = getChatDom();
    expect(dom.rightPanel.classList.contains("read-only")).toBe(true);
  });

  test("populates temperature value", () => {
    showReadOnlyParams({ temperature: 0.7 } as GenerationSettings, null);
    expect(getChatDom().panelTemperature.value).toBe("0.7");
  });

  test("populates top_p value", () => {
    showReadOnlyParams({ top_p: 0.9 } as GenerationSettings, null);
    expect(getChatDom().panelTopP.value).toBe("0.9");
  });

  test("populates seed value", () => {
    showReadOnlyParams({ seed: 42 } as GenerationSettings, null);
    expect(getChatDom().panelSeed.value).toBe("42");
  });

  test("disables all parameter inputs", () => {
    showReadOnlyParams({ model: "gpt-4", temperature: 0.5 } as GenerationSettings, null);
    const dom = getChatDom();
    expect(dom.panelTemperature.disabled).toBe(true);
    expect(dom.panelTopP.disabled).toBe(true);
    expect(dom.panelTopK.disabled).toBe(true);
    expect(dom.panelMinP.disabled).toBe(true);
    expect(dom.panelMaxOutputTokens.disabled).toBe(true);
    expect(dom.panelRepetitionPenalty.disabled).toBe(true);
    expect(dom.panelSeed.disabled).toBe(true);
    expect(dom.panelModel.disabled).toBe(true);
  });

  test("renders usage stats when provided", () => {
    const usage: UsageStats = {
      output_tokens: 150,
      input_tokens: 50,
      tokens_per_second: 30,
      duration_ms: 5000,
    } as UsageStats;
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, usage);
    const info = getChatDom().panelChatInfo;
    expect(info.textContent).toContain("150");
    expect(info.textContent).toContain("50");
    expect(info.textContent).toContain("30 tok/s");
    expect(info.textContent).toContain("5.00s");
  });

  test("renders only output tokens when other usage fields missing", () => {
    const usage = { output_tokens: 100 } as UsageStats;
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, usage);
    const info = getChatDom().panelChatInfo;
    expect(info.textContent).toContain("100");
    expect(info.textContent).not.toContain("tok/s");
  });
});

// ── restoreEditableParams ───────────────────────────────────────────────────

describe("restoreEditableParams", () => {
  test("no-op when not in read-only mode", () => {
    const dom = getChatDom();
    dom.panelTemperature.disabled = true;
    restoreEditableParams();
    // Should not have changed anything
    expect(dom.panelTemperature.disabled).toBe(true);
  });

  test("re-enables all inputs", () => {
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, null);
    restoreEditableParams();
    const dom = getChatDom();
    expect(dom.panelTemperature.disabled).toBe(false);
    expect(dom.panelTopP.disabled).toBe(false);
    expect(dom.panelTopK.disabled).toBe(false);
    expect(dom.panelMinP.disabled).toBe(false);
    expect(dom.panelMaxOutputTokens.disabled).toBe(false);
    expect(dom.panelRepetitionPenalty.disabled).toBe(false);
    expect(dom.panelSeed.disabled).toBe(false);
    expect(dom.panelModel.disabled).toBe(false);
  });

  test("removes read-only class", () => {
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, null);
    restoreEditableParams();
    expect(getChatDom().rightPanel.classList.contains("read-only")).toBe(false);
  });

  test("resets isPanelReadOnly to false", () => {
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, null);
    expect(isPanelReadOnly()).toBe(true);
    restoreEditableParams();
    expect(isPanelReadOnly()).toBe(false);
  });
});

// ── hideRightPanel ──────────────────────────────────────────────────────────

describe("hideRightPanel", () => {
  test("hides the right panel", () => {
    const dom = getChatDom();
    dom.rightPanel.classList.remove("hidden");
    dom.rightPanel.classList.add("flex");
    hideRightPanel();
    expect(dom.rightPanel.classList.contains("hidden")).toBe(true);
    expect(dom.rightPanel.classList.contains("flex")).toBe(false);
  });

  test("removes active class from tuning button", () => {
    const tuningBtn = domRoot.querySelector<HTMLButtonElement>('[data-action="toggle-tuning"]')!;
    tuningBtn.classList.add("active");
    hideRightPanel();
    expect(tuningBtn.classList.contains("active")).toBe(false);
  });
});

// ── handleToggleTuning ──────────────────────────────────────────────────────

describe("handleToggleTuning", () => {
  test("opens the right panel", () => {
    const dom = getChatDom();
    dom.rightPanel.classList.add("hidden");
    const btn = document.createElement("button");
    handleToggleTuning(btn);
    expect(dom.rightPanel.classList.contains("hidden")).toBe(false);
    expect(dom.rightPanel.classList.contains("flex")).toBe(true);
  });

  test("adds active class to button", () => {
    const btn = document.createElement("button");
    handleToggleTuning(btn);
    expect(btn.classList.contains("active")).toBe(true);
  });

  test("restores editable mode if currently read-only", () => {
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, null);
    expect(isPanelReadOnly()).toBe(true);
    const btn = document.createElement("button");
    handleToggleTuning(btn);
    expect(isPanelReadOnly()).toBe(false);
    expect(getChatDom().panelTemperature.disabled).toBe(false);
  });
});
