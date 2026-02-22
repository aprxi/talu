import { describe, test, expect, beforeEach } from "bun:test";
import {
  populateForm,
  updateSystemPromptDisplay,
  showModelParams,
  saveTopLevelSettings,
  saveModelOverrides,
  handleResetModelOverrides,
  handleModelChange,
} from "../../../src/plugins/settings/form.ts";
import { wireEvents } from "../../../src/plugins/settings/events.ts";
import { settingsState, notifyChange, emitModelChanged } from "../../../src/plugins/settings/state.ts";
import { initSettingsDeps } from "../../../src/plugins/settings/deps.ts";
import { initSettingsDom, getSettingsDom } from "../../../src/plugins/settings/dom.ts";
import { createDomRoot, SETTINGS_DOM_IDS, SETTINGS_DOM_TAGS } from "../../helpers/dom.ts";
import type { Disposable } from "../../../src/kernel/types.ts";

/**
 * Tests for settings form — population, auto-save debouncing, model
 * switching, and state management.
 *
 * Strategy: mock API to record calls and return controllable results.
 * Timer mock captures callbacks without firing them so debounce behavior
 * can be verified precisely.
 */

// -- Mock state --------------------------------------------------------------

let apiCalls: { method: string; args: unknown[] }[];
let emittedEvents: { event: string; data: unknown }[];
let timerCallbacks: { fn: () => void; ms: number; disposed: boolean }[];
let patchSettingsResult: any;
let resetOverridesResult: any;

beforeEach(() => {
  apiCalls = [];
  emittedEvents = [];
  timerCallbacks = [];
  patchSettingsResult = { ok: true, data: { model: "gpt-4" } };
  resetOverridesResult = { ok: true, data: { available_models: [] } };

  // Reset state.
  settingsState.activeModel = "";
  settingsState.availableModels = [];
  settingsState.changeHandlers.clear();
  settingsState.systemPromptEnabled = true;

  // DOM with typed elements (select, input, button).
  initSettingsDom(createDomRoot(SETTINGS_DOM_IDS, undefined, SETTINGS_DOM_TAGS));

  // Deps with controllable timer (does NOT auto-fire).
  initSettingsDeps({
    api: {
      patchSettings: async (patch: any) => {
        apiCalls.push({ method: "patchSettings", args: [patch] });
        return patchSettingsResult;
      },
      resetModelOverrides: async (modelId: string) => {
        apiCalls.push({ method: "resetModelOverrides", args: [modelId] });
        return resetOverridesResult;
      },
    } as any,
    events: {
      emit: (event: string, data: unknown) => { emittedEvents.push({ event, data }); },
      on: () => ({ dispose() {} }),
    } as any,
    timers: {
      setTimeout(fn: () => void, ms: number): Disposable {
        const entry = { fn, ms, disposed: false };
        timerCallbacks.push(entry);
        return { dispose() { entry.disposed = true; } };
      },
      setInterval() { return { dispose() {} }; },
      requestAnimationFrame(fn: () => void) { fn(); return { dispose() {} }; },
    } as any,
    mode: {
      switch: () => {},
      getActive: () => "settings",
      onChange: () => ({ dispose() {} }),
    } as any,
  });
});

// -- Helpers -----------------------------------------------------------------

function makeModelEntry(id: string, defaults: any = {}, overrides: any = {}) {
  return {
    id,
    defaults: { temperature: 1.0, top_p: 1.0, top_k: 50, ...defaults },
    overrides,
  };
}

// ── populateForm ──────────────────────────────────────────────────────────────

describe("populateForm", () => {
  test("sets system prompt enabled checkbox", () => {
    populateForm({ system_prompt_enabled: true, auto_title: true } as any);
    expect(getSettingsDom().systemPromptEnabled.checked).toBe(true);
    populateForm({ system_prompt_enabled: false, auto_title: true } as any);
    expect(getSettingsDom().systemPromptEnabled.checked).toBe(false);
  });

  test("sets max output tokens value", () => {
    populateForm({ max_output_tokens: 2048, auto_title: true } as any);
    expect(getSettingsDom().maxOutputTokens.value).toBe("2048");
  });

  test("sets context length value", () => {
    populateForm({ context_length: 4096, auto_title: true } as any);
    expect(getSettingsDom().contextLength.value).toBe("4096");
  });

  test("null max_output_tokens → empty string", () => {
    populateForm({ max_output_tokens: null, auto_title: true } as any);
    expect(getSettingsDom().maxOutputTokens.value).toBe("");
  });

  test("null context_length → empty string", () => {
    populateForm({ context_length: null, auto_title: true } as any);
    expect(getSettingsDom().contextLength.value).toBe("");
  });

  test("sets auto_title checkbox", () => {
    populateForm({ auto_title: false } as any);
    expect((getSettingsDom().autoTitle as HTMLInputElement).checked).toBe(false);
    populateForm({ auto_title: true } as any);
    expect((getSettingsDom().autoTitle as HTMLInputElement).checked).toBe(true);
  });
});

// ── updateSystemPromptDisplay ────────────────────────────────────────────────

describe("updateSystemPromptDisplay", () => {
  test("shows prompt name when custom default exists", () => {
    updateSystemPromptDisplay(
      [{ id: "p1", name: "My Prompt" }, { id: "p2", name: "Other" }],
      "p1",
      false,
    );
    expect(getSettingsDom().systemPromptName.textContent).toBe("My Prompt");
  });

  test("shows 'Default' when isBuiltinDefault is true", () => {
    updateSystemPromptDisplay(
      [{ id: "p1", name: "My Prompt" }],
      "builtin-1",
      true,
    );
    expect(getSettingsDom().systemPromptName.textContent).toBe("Default");
  });

  test("shows 'Default' when no default and not built-in", () => {
    updateSystemPromptDisplay(
      [{ id: "p1", name: "My Prompt" }],
      null,
      false,
    );
    expect(getSettingsDom().systemPromptName.textContent).toBe("Default");
  });

  test("shows 'Default' when default not in list", () => {
    updateSystemPromptDisplay(
      [{ id: "p1", name: "A" }],
      "nonexistent",
      false,
    );
    expect(getSettingsDom().systemPromptName.textContent).toBe("Default");
  });

  test("shows 'Default' when prompts list is empty", () => {
    updateSystemPromptDisplay([], "p1", false);
    expect(getSettingsDom().systemPromptName.textContent).toBe("Default");
  });
});

// ── showModelParams ───────────────────────────────────────────────────────────

describe("showModelParams", () => {
  test("sets model label", () => {
    settingsState.availableModels = [makeModelEntry("gpt-4")];
    showModelParams("gpt-4");
    expect(getSettingsDom().modelLabel.textContent).toBe("(gpt-4)");
  });

  test("empty label when modelId is empty", () => {
    showModelParams("");
    expect(getSettingsDom().modelLabel.textContent).toBe("");
  });

  test("sets placeholder from defaults", () => {
    settingsState.availableModels = [makeModelEntry("gpt-4", { temperature: 0.7, top_p: 0.9, top_k: 40 })];
    showModelParams("gpt-4");
    const dom = getSettingsDom();
    expect(dom.temperature.placeholder).toBe("0.7");
    expect(dom.topP.placeholder).toBe("0.9");
    expect(dom.topK.placeholder).toBe("40");
  });

  test("sets value from overrides", () => {
    settingsState.availableModels = [makeModelEntry("gpt-4", {}, { temperature: 0.5, top_p: 0.8, top_k: 30 })];
    showModelParams("gpt-4");
    const dom = getSettingsDom();
    expect(dom.temperature.value).toBe("0.5");
    expect(dom.topP.value).toBe("0.8");
    expect(dom.topK.value).toBe("30");
  });

  test("empty value when override is null", () => {
    settingsState.availableModels = [makeModelEntry("gpt-4")];
    showModelParams("gpt-4");
    const dom = getSettingsDom();
    expect(dom.temperature.value).toBe("");
    expect(dom.topP.value).toBe("");
    expect(dom.topK.value).toBe("");
  });

  test("shows default hint text", () => {
    settingsState.availableModels = [makeModelEntry("gpt-4", { temperature: 1.0, top_p: 0.95, top_k: 50 })];
    showModelParams("gpt-4");
    const dom = getSettingsDom();
    expect(dom.temperatureDefault.textContent).toBe("default: 1");
    expect(dom.topPDefault.textContent).toBe("default: 0.95");
    expect(dom.topKDefault.textContent).toBe("default: 50");
  });

  test("no-op on params when model not found (label still set)", () => {
    settingsState.availableModels = [];
    showModelParams("nonexistent");
    const dom = getSettingsDom();
    expect(dom.modelLabel.textContent).toBe("(nonexistent)");
    // Params untouched — still initial empty values.
    expect(dom.temperature.placeholder).toBe("");
  });
});

// ── saveTopLevelSettings ──────────────────────────────────────────────────────

describe("saveTopLevelSettings", () => {
  test("shows Saving... status before API resolves", async () => {
    const p = saveTopLevelSettings();
    expect(getSettingsDom().status.textContent).toBe("Saving...");
    expect(getSettingsDom().status.className).toBe("text-xs text-text-subtle");
    await p;
  });

  test("sends correct patch from form values", async () => {
    const dom = getSettingsDom();
    dom.systemPromptEnabled.checked = true;
    dom.maxOutputTokens.value = "1024";
    dom.contextLength.value = "8192";
    (dom.autoTitle as HTMLInputElement).checked = true;
    await saveTopLevelSettings();

    const patch = apiCalls[0]!.args[0] as any;
    expect(patch.system_prompt_enabled).toBe(true);
    expect(patch.max_output_tokens).toBe(1024);
    expect(patch.context_length).toBe(8192);
    expect(patch.auto_title).toBe(true);
  });

  test("sends null for empty numeric fields", async () => {
    const dom = getSettingsDom();
    dom.maxOutputTokens.value = "";
    dom.contextLength.value = "";
    await saveTopLevelSettings();

    const patch = apiCalls[0]!.args[0] as any;
    expect(patch.max_output_tokens).toBeNull();
    expect(patch.context_length).toBeNull();
  });

  test("shows Saved on success", async () => {
    await saveTopLevelSettings();
    const dom = getSettingsDom();
    expect(dom.status.textContent).toBe("Saved");
    expect(dom.status.className).toContain("text-success");
  });

  test("updates activeModel from response", async () => {
    patchSettingsResult = { ok: true, data: { model: "claude-3" } };
    await saveTopLevelSettings();
    expect(settingsState.activeModel).toBe("claude-3");
  });

  test("shows error on API failure", async () => {
    patchSettingsResult = { ok: false, error: "Server error" };
    await saveTopLevelSettings();
    const dom = getSettingsDom();
    expect(dom.status.textContent).toBe("Server error");
    expect(dom.status.className).toContain("text-danger");
  });

  test("shows default error when error field is empty", async () => {
    patchSettingsResult = { ok: false };
    await saveTopLevelSettings();
    expect(getSettingsDom().status.textContent).toBe("Failed to save");
  });

  test("schedules 1500ms timer to clear status on success", async () => {
    await saveTopLevelSettings();
    expect(timerCallbacks.length).toBe(1);
    // Fire the clear-status timer.
    timerCallbacks[0]!.fn();
    expect(getSettingsDom().status.textContent).toBe("");
  });

  test("emits settings.system_prompt_enabled when toggle changes", async () => {
    settingsState.systemPromptEnabled = true;
    getSettingsDom().systemPromptEnabled.checked = false;
    await saveTopLevelSettings();
    const event = emittedEvents.find((e) => e.event === "settings.system_prompt_enabled");
    expect(event).toBeDefined();
    expect((event!.data as any).enabled).toBe(false);
  });

  test("does not emit event when system prompt enabled unchanged", async () => {
    settingsState.systemPromptEnabled = true;
    getSettingsDom().systemPromptEnabled.checked = true;
    await saveTopLevelSettings();
    const event = emittedEvents.find((e) => e.event === "settings.system_prompt_enabled");
    expect(event).toBeUndefined();
  });
});

// ── saveModelOverrides ────────────────────────────────────────────────────────

describe("saveModelOverrides", () => {
  test("sends model_overrides patch with parsed values", async () => {
    const dom = getSettingsDom();
    dom.temperature.value = "0.8";
    dom.topP.value = "0.95";
    dom.topK.value = "40";
    await saveModelOverrides();

    const patch = apiCalls[0]!.args[0] as any;
    expect(patch.model_overrides.temperature).toBe(0.8);
    expect(patch.model_overrides.top_p).toBe(0.95);
    expect(patch.model_overrides.top_k).toBe(40);
  });

  test("sends null for empty override fields", async () => {
    await saveModelOverrides();
    const patch = apiCalls[0]!.args[0] as any;
    expect(patch.model_overrides.temperature).toBeNull();
    expect(patch.model_overrides.top_p).toBeNull();
    expect(patch.model_overrides.top_k).toBeNull();
  });

  test("updates availableModels from response", async () => {
    const models = [makeModelEntry("gpt-4"), makeModelEntry("claude-3")];
    patchSettingsResult = { ok: true, data: { available_models: models } };
    await saveModelOverrides();
    expect(settingsState.availableModels).toEqual(models);
  });

  test("emits model.changed on success", async () => {
    patchSettingsResult = { ok: true, data: { available_models: [] } };
    await saveModelOverrides();
    expect(emittedEvents.some((e) => e.event === "model.changed")).toBe(true);
  });

  test("does not emit model.changed on failure", async () => {
    patchSettingsResult = { ok: false, error: "fail" };
    await saveModelOverrides();
    expect(emittedEvents.some((e) => e.event === "model.changed")).toBe(false);
  });

  test("shows error on failure", async () => {
    patchSettingsResult = { ok: false, error: "Bad request" };
    await saveModelOverrides();
    expect(getSettingsDom().status.textContent).toBe("Bad request");
  });
});

// ── handleResetModelOverrides ─────────────────────────────────────────────────

describe("handleResetModelOverrides", () => {
  test("no-op when no activeModel", async () => {
    settingsState.activeModel = "";
    await handleResetModelOverrides();
    expect(apiCalls.length).toBe(0);
  });

  test("shows Resetting... status", async () => {
    settingsState.activeModel = "gpt-4";
    const p = handleResetModelOverrides();
    expect(getSettingsDom().status.textContent).toBe("Resetting...");
    await p;
  });

  test("calls resetModelOverrides API with activeModel", async () => {
    settingsState.activeModel = "gpt-4";
    await handleResetModelOverrides();
    expect(apiCalls[0]!.method).toBe("resetModelOverrides");
    expect(apiCalls[0]!.args[0]).toBe("gpt-4");
  });

  test("updates availableModels from response", async () => {
    const models = [makeModelEntry("gpt-4")];
    resetOverridesResult = { ok: true, data: { available_models: models } };
    settingsState.activeModel = "gpt-4";
    await handleResetModelOverrides();
    expect(settingsState.availableModels).toEqual(models);
  });

  test("re-shows model params after reset", async () => {
    const models = [makeModelEntry("gpt-4", { temperature: 0.9 })];
    resetOverridesResult = { ok: true, data: { available_models: models } };
    settingsState.activeModel = "gpt-4";
    await handleResetModelOverrides();
    expect(getSettingsDom().temperature.placeholder).toBe("0.9");
  });

  test("emits model.changed on success", async () => {
    settingsState.activeModel = "gpt-4";
    resetOverridesResult = { ok: true, data: { available_models: [] } };
    await handleResetModelOverrides();
    expect(emittedEvents.some((e) => e.event === "model.changed")).toBe(true);
  });

  test("shows error on failure", async () => {
    settingsState.activeModel = "gpt-4";
    resetOverridesResult = { ok: false, error: "Not found" };
    await handleResetModelOverrides();
    expect(getSettingsDom().status.textContent).toBe("Not found");
  });
});

// ── handleModelChange ─────────────────────────────────────────────────────────

describe("handleModelChange", () => {
  test("no-op when empty string", () => {
    handleModelChange("");
    expect(apiCalls.length).toBe(0);
    expect(emittedEvents.length).toBe(0);
  });

  test("updates activeModel in state", () => {
    settingsState.availableModels = [makeModelEntry("gpt-4")];
    handleModelChange("gpt-4");
    expect(settingsState.activeModel).toBe("gpt-4");
  });

  test("shows model params for new model", () => {
    settingsState.availableModels = [makeModelEntry("gpt-4", { temperature: 0.7 })];
    handleModelChange("gpt-4");
    expect(getSettingsDom().modelLabel.textContent).toBe("(gpt-4)");
    expect(getSettingsDom().temperature.placeholder).toBe("0.7");
  });

  test("emits model.changed event with payload", () => {
    settingsState.availableModels = [makeModelEntry("gpt-4")];
    handleModelChange("gpt-4");
    const event = emittedEvents.find((e) => e.event === "model.changed")!;
    expect(event).toBeDefined();
    expect((event.data as any).modelId).toBe("gpt-4");
  });

  test("notifies change handlers", () => {
    const calls: string[] = [];
    settingsState.changeHandlers.add(() => calls.push("handler-a"));
    settingsState.availableModels = [makeModelEntry("gpt-4")];
    handleModelChange("gpt-4");
    expect(calls).toEqual(["handler-a"]);
  });

  test("calls patchSettings API with model", () => {
    settingsState.availableModels = [makeModelEntry("gpt-4")];
    handleModelChange("gpt-4");
    expect(apiCalls[0]!.method).toBe("patchSettings");
    expect((apiCalls[0]!.args[0] as any).model).toBe("gpt-4");
  });
});

// ── wireEvents — debouncing ──────────────────────────────────────────────────

describe("wireEvents — debouncing", () => {
  test("max output tokens input schedules 400ms debounce", () => {
    wireEvents();
    getSettingsDom().maxOutputTokens.dispatchEvent(new Event("input"));
    expect(timerCallbacks.length).toBe(1);
    expect(timerCallbacks[0]!.ms).toBe(400);
  });

  test("context length input schedules 400ms debounce", () => {
    wireEvents();
    getSettingsDom().contextLength.dispatchEvent(new Event("input"));
    expect(timerCallbacks.length).toBe(1);
    expect(timerCallbacks[0]!.ms).toBe(400);
  });

  test("temperature input schedules 400ms override save", () => {
    wireEvents();
    getSettingsDom().temperature.dispatchEvent(new Event("input"));
    expect(timerCallbacks.length).toBe(1);
    expect(timerCallbacks[0]!.ms).toBe(400);
  });

  test("system prompt enabled change triggers immediate save", async () => {
    wireEvents();
    getSettingsDom().systemPromptEnabled.dispatchEvent(new Event("change"));
    await new Promise((r) => setTimeout(r, 10));
    expect(apiCalls[0]!.method).toBe("patchSettings");
  });

  test("model change triggers immediate API call (no debounce)", () => {
    settingsState.availableModels = [makeModelEntry("gpt-4")];
    wireEvents();
    const select = getSettingsDom().model as HTMLSelectElement;
    const opt = document.createElement("option");
    opt.value = "gpt-4";
    select.appendChild(opt);
    select.value = "gpt-4";
    select.dispatchEvent(new Event("change"));
    expect(apiCalls[0]!.method).toBe("patchSettings");
    expect((apiCalls[0]!.args[0] as any).model).toBe("gpt-4");
  });

  test("rapid typing cancels previous debounce timers", () => {
    wireEvents();
    const dom = getSettingsDom();
    dom.maxOutputTokens.dispatchEvent(new Event("input"));
    dom.maxOutputTokens.dispatchEvent(new Event("input"));
    dom.maxOutputTokens.dispatchEvent(new Event("input"));

    expect(timerCallbacks[0]!.disposed).toBe(true);
    expect(timerCallbacks[1]!.disposed).toBe(true);
    expect(timerCallbacks[2]!.disposed).toBe(false);
  });

  test("settings and overrides share debounce handle", () => {
    wireEvents();
    const dom = getSettingsDom();
    dom.maxOutputTokens.dispatchEvent(new Event("input"));  // settings (400ms)
    dom.temperature.dispatchEvent(new Event("input"));       // overrides (400ms) cancels settings

    expect(timerCallbacks[0]!.disposed).toBe(true);
    expect(timerCallbacks[1]!.disposed).toBe(false);
  });

  test("debounce callback triggers actual save", async () => {
    wireEvents();
    const dom = getSettingsDom();
    dom.maxOutputTokens.value = "512";
    dom.maxOutputTokens.dispatchEvent(new Event("input"));
    // Fire the debounced callback manually.
    timerCallbacks[0]!.fn();
    await new Promise((r) => setTimeout(r, 10));
    expect(apiCalls[0]!.method).toBe("patchSettings");
  });

  test("reset button triggers immediate reset", async () => {
    settingsState.activeModel = "gpt-4";
    wireEvents();
    getSettingsDom().resetModel.dispatchEvent(new Event("click"));
    await new Promise((r) => setTimeout(r, 10));
    expect(apiCalls[0]!.method).toBe("resetModelOverrides");
  });
});

// ── notifyChange ──────────────────────────────────────────────────────────────

describe("notifyChange", () => {
  test("calls all registered handlers", () => {
    const calls: string[] = [];
    settingsState.changeHandlers.add(() => calls.push("a"));
    settingsState.changeHandlers.add(() => calls.push("b"));
    notifyChange();
    expect(calls).toEqual(["a", "b"]);
  });

  test("swallows handler errors without breaking others", () => {
    settingsState.changeHandlers.add(() => { throw new Error("boom"); });
    const calls: string[] = [];
    settingsState.changeHandlers.add(() => calls.push("survived"));
    notifyChange();
    expect(calls).toEqual(["survived"]);
  });
});

// ── emitModelChanged ──────────────────────────────────────────────────────────

describe("emitModelChanged", () => {
  test("emits model.changed with current state", () => {
    settingsState.activeModel = "claude-3";
    settingsState.availableModels = [makeModelEntry("claude-3")];
    emitModelChanged();

    const event = emittedEvents.find((e) => e.event === "model.changed")!;
    expect(event).toBeDefined();
    expect((event.data as any).modelId).toBe("claude-3");
    expect((event.data as any).availableModels).toEqual(settingsState.availableModels);
  });
});
