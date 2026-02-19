import { describe, test, expect, beforeEach } from "bun:test";
import {
  getSamplingParams,
  syncRightPanelParams,
  updatePanelChatInfo,
} from "../../../src/plugins/chat/panel-params.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { createDomRoot, CHAT_DOM_IDS, CHAT_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import type { Conversation } from "../../../src/types.ts";

/**
 * Tests for panel-params — sampling param extraction, sync, and chat info display.
 */

let notifs: ReturnType<typeof mockNotifications>;

beforeEach(() => {
  notifs = mockNotifications();
  initChatDom(createDomRoot(CHAT_DOM_IDS, undefined, CHAT_DOM_TAGS));
  initChatDeps({
    api: {} as any,
    notifications: notifs.mock,
    timers: mockTimers(),
    services: {
      get: () => ({
        getActiveModel: () => "gpt-4",
        getAvailableModels: () => [
          {
            id: "gpt-4",
            defaults: { temperature: 1.0, top_p: 1.0, top_k: 50 },
            overrides: { temperature: 0.7, top_p: null, top_k: null },
          },
        ],
        getPromptNameById: () => null,
      }),
    } as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    hooks: { run: async (_name: string, payload: any) => payload } as any,
    clipboard: { writeText: async () => {} } as any,
    download: {} as any,
    observe: { onResize: () => ({ dispose() {} }) } as any,
    format: { dateTime: (ms: number) => new Date(ms * 1000).toISOString() } as any,
    upload: { upload: async () => ({}) } as any,
    layout: { setTitle: () => {} } as any,
    menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
  });
});

// ── getSamplingParams ───────────────────────────────────────────────────────

describe("getSamplingParams", () => {
  test("returns empty object when all inputs are empty", () => {
    const params = getSamplingParams();
    expect(params).toEqual({});
  });

  test("parses temperature as float", () => {
    getChatDom().panelTemperature.value = "0.7";
    const params = getSamplingParams();
    expect(params.temperature).toBe(0.7);
  });

  test("parses top_p as float", () => {
    getChatDom().panelTopP.value = "0.95";
    const params = getSamplingParams();
    expect(params.top_p).toBe(0.95);
  });

  test("parses top_k as integer", () => {
    getChatDom().panelTopK.value = "40";
    const params = getSamplingParams();
    expect(params.top_k).toBe(40);
  });

  test("parses min_p as float", () => {
    getChatDom().panelMinP.value = "0.05";
    const params = getSamplingParams();
    expect(params.min_p).toBe(0.05);
  });

  test("parses max_output_tokens as integer", () => {
    getChatDom().panelMaxOutputTokens.value = "4096";
    const params = getSamplingParams();
    expect(params.max_output_tokens).toBe(4096);
  });

  test("parses repetition_penalty as float", () => {
    getChatDom().panelRepetitionPenalty.value = "1.1";
    const params = getSamplingParams();
    expect(params.repetition_penalty).toBe(1.1);
  });

  test("parses seed as integer", () => {
    getChatDom().panelSeed.value = "42";
    const params = getSamplingParams();
    expect(params.seed).toBe(42);
  });

  test("skips whitespace-only fields", () => {
    getChatDom().panelTemperature.value = "   ";
    const params = getSamplingParams();
    expect(params.temperature).toBeUndefined();
  });

  test("returns multiple params when set", () => {
    getChatDom().panelTemperature.value = "0.5";
    getChatDom().panelTopK.value = "20";
    getChatDom().panelSeed.value = "123";
    const params = getSamplingParams();
    expect(params.temperature).toBe(0.5);
    expect(params.top_k).toBe(20);
    expect(params.seed).toBe(123);
    expect(params.top_p).toBeUndefined();
  });
});

// ── syncRightPanelParams ────────────────────────────────────────────────────

describe("syncRightPanelParams", () => {
  test("populates temperature from override", () => {
    syncRightPanelParams("gpt-4");
    const dom = getChatDom();
    expect(dom.panelTemperature.value).toBe("0.7");
    expect(dom.panelTemperature.placeholder).toBe("1");
  });

  test("shows default hint text", () => {
    syncRightPanelParams("gpt-4");
    const dom = getChatDom();
    expect(dom.panelTemperatureDefault.textContent).toBe("Default: 1");
  });

  test("leaves value empty when override is null", () => {
    syncRightPanelParams("gpt-4");
    const dom = getChatDom();
    expect(dom.panelTopP.value).toBe("");
    expect(dom.panelTopP.placeholder).toBe("1");
  });

  test("no-op for unknown model", () => {
    getChatDom().panelTemperature.value = "original";
    syncRightPanelParams("unknown-model");
    expect(getChatDom().panelTemperature.value).toBe("original");
  });
});

// ── updatePanelChatInfo ─────────────────────────────────────────────────────

describe("updatePanelChatInfo", () => {
  test("no-op when chat is null", () => {
    getChatDom().panelInfoCreated.textContent = "original";
    updatePanelChatInfo(null);
    expect(getChatDom().panelInfoCreated.textContent).toBe("original");
  });

  test("displays created date", () => {
    const chat = { created_at: 1700000 } as Conversation;
    updatePanelChatInfo(chat);
    expect(getChatDom().panelInfoCreated.textContent).not.toBe("");
  });

  test("shows forked row when parent_session_id exists", () => {
    const dom = getChatDom();
    dom.panelInfoForkedRow.classList.add("hidden");
    const chat = { created_at: 0, parent_session_id: "abcdef1234567890" } as Conversation;
    updatePanelChatInfo(chat);
    expect(dom.panelInfoForkedRow.classList.contains("hidden")).toBe(false);
    expect(dom.panelInfoForked.textContent).toBe("abcdef12...");
  });

  test("hides forked row when no parent_session_id", () => {
    const dom = getChatDom();
    dom.panelInfoForkedRow.classList.remove("hidden");
    const chat = { created_at: 0 } as Conversation;
    updatePanelChatInfo(chat);
    expect(dom.panelInfoForkedRow.classList.contains("hidden")).toBe(true);
  });
});
