import { describe, test, expect, beforeEach } from "bun:test";
import {
  showReadOnlyParams,
  restoreEditableParams,
  hideChatPanel,
  handleToggleTuning,
  isPanelReadOnly,
} from "../../../src/plugins/chat/panel-readonly.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { getChatPanelDom } from "../../../src/plugins/chat/chat-panel-dom.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { createDomRoot, CHAT_DOM_IDS, CHAT_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import type { GenerationSettings, UsageStats } from "../../../src/types.ts";

/**
 * Tests for panel-readonly — read-only mode for viewing past message
 * generation parameters and usage stats.
 */

let domRoot: HTMLElement;
let showPanelCalls: Array<{
  title: string;
  content: HTMLElement;
  owner?: string;
  onHide?: () => void;
}>;
let hidePanelCalls: Array<string | undefined>;

beforeEach(() => {
  const notifs = mockNotifications();
  domRoot = createDomRoot(CHAT_DOM_IDS, undefined, CHAT_DOM_TAGS);
  showPanelCalls = [];
  hidePanelCalls = [];

  // Add a tuning button inside transcript for hideChatPanel/handleToggleTuning.
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
    layout: {
      setTitle: () => {},
      showPanel: (options: {
        title: string;
        content: HTMLElement;
        owner?: string;
        onHide?: () => void;
      }) => {
        showPanelCalls.push(options);
        return { dispose() {} };
      },
      hidePanel: (owner?: string) => {
        hidePanelCalls.push(owner);
      },
    } as any,
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
    expect(showPanelCalls).toHaveLength(0);
  });

  test("sets panelReadOnlyMode to true", () => {
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, null);
    expect(isPanelReadOnly()).toBe(true);
    expect(showPanelCalls).toHaveLength(1);
  });

  test("opens the chat panel through layout.showPanel", () => {
    const panelDom = getChatPanelDom();
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, null);
    const call = showPanelCalls[0]!;
    expect(showPanelCalls).toHaveLength(1);
    expect(call.title).toBe("Chat");
    expect(call.owner).toBe("chat");
    expect(call.content).toBe(panelDom.root);
    expect(typeof call.onHide).toBe("function");
  });

  test("adds read-only class to the panel root", () => {
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, null);
    expect(getChatPanelDom().root.classList.contains("read-only")).toBe(true);
  });

  test("populates temperature value", () => {
    showReadOnlyParams({ temperature: 0.7 } as GenerationSettings, null);
    expect(getChatPanelDom().panelTemperature.value).toBe("0.7");
  });

  test("populates top_p value", () => {
    showReadOnlyParams({ top_p: 0.9 } as GenerationSettings, null);
    expect(getChatPanelDom().panelTopP.value).toBe("0.9");
  });

  test("populates seed value", () => {
    showReadOnlyParams({ seed: 42 } as GenerationSettings, null);
    expect(getChatPanelDom().panelSeed.value).toBe("42");
  });

  test("disables all parameter inputs", () => {
    showReadOnlyParams({ model: "gpt-4", temperature: 0.5 } as GenerationSettings, null);
    const dom = getChatPanelDom();
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
    const info = getChatPanelDom().panelChatInfo;
    expect(info.textContent).toContain("Output tokens");
    expect(info.textContent).toContain("Input tokens");
    expect(info.textContent).toContain("Speed");
    expect(info.textContent).toContain("Duration");
    expect(info.textContent).toContain("150");
    expect(info.textContent).toContain("50");
    expect(info.textContent).toContain("30 tok/s");
    expect(info.textContent).toContain("5.00s");
  });

  test("renders only output tokens when other usage fields missing", () => {
    const usage = { output_tokens: 100 } as UsageStats;
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, usage);
    const info = getChatPanelDom().panelChatInfo;
    expect(info.textContent).toContain("100");
    expect(info.textContent).toContain("Output tokens");
    expect(info.textContent).not.toContain("Input tokens");
    expect(info.textContent).not.toContain("tok/s");
    expect(info.textContent).not.toContain("Duration");
  });
});

// ── restoreEditableParams ───────────────────────────────────────────────────

describe("restoreEditableParams", () => {
  test("no-op when not in read-only mode", () => {
    const dom = getChatPanelDom();
    dom.panelTemperature.disabled = true;
    restoreEditableParams();
    // Should not have changed anything
    expect(dom.panelTemperature.disabled).toBe(true);
  });

  test("re-enables all inputs", () => {
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, null);
    restoreEditableParams();
    const dom = getChatPanelDom();
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
    expect(getChatPanelDom().root.classList.contains("read-only")).toBe(false);
  });

  test("resets isPanelReadOnly to false", () => {
    showReadOnlyParams({ model: "gpt-4" } as GenerationSettings, null);
    expect(isPanelReadOnly()).toBe(true);
    restoreEditableParams();
    expect(isPanelReadOnly()).toBe(false);
  });
});

// ── hideChatPanel ───────────────────────────────────────────────────────────

describe("hideChatPanel", () => {
  test("hides the chat panel through layout.hidePanel", () => {
    hideChatPanel();
    expect(hidePanelCalls).toEqual(["chat"]);
  });

  test("removes active class from tuning button", () => {
    const tuningBtn = domRoot.querySelector<HTMLButtonElement>('[data-action="toggle-tuning"]')!;
    tuningBtn.classList.add("active");
    hideChatPanel();
    expect(tuningBtn.classList.contains("active")).toBe(false);
  });
});

// ── handleToggleTuning ──────────────────────────────────────────────────────

describe("handleToggleTuning", () => {
  test("opens the chat panel", () => {
    const btn = document.createElement("button");
    handleToggleTuning(btn);
    expect(showPanelCalls).toHaveLength(1);
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
    expect(getChatPanelDom().panelTemperature.disabled).toBe(false);
    expect(showPanelCalls).toHaveLength(2);
  });

  test("keeps panel manager payload aligned with the panel root", () => {
    const panelDom = getChatPanelDom();
    const btn = document.createElement("button");
    handleToggleTuning(btn);
    const call = showPanelCalls[0]!;
    expect(call.content).toBe(panelDom.root);
    expect(call.owner).toBe("chat");
  });
});
