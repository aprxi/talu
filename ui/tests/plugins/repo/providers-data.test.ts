import { beforeEach, describe, expect, test } from "bun:test";
import { addProvider, loadProviders, removeProvider, testProvider, updateProvider } from "../../../src/plugins/repo/providers-data.ts";
import { initRepoDeps } from "../../../src/plugins/repo/deps.ts";
import { initRepoDom, getRepoDom } from "../../../src/plugins/repo/dom.ts";
import { repoState } from "../../../src/plugins/repo/state.ts";
import { createDomRoot, REPO_DOM_EXTRAS, REPO_DOM_IDS, REPO_DOM_TAGS } from "../../helpers/dom.ts";
import { mockNotifications, mockTimers } from "../../helpers/mocks.ts";

let apiCalls: { method: string; args: unknown[] }[];
let notif: ReturnType<typeof mockNotifications>;
let emitted: { name: string; data: unknown }[];

let listProvidersResult: any;
let updateProviderResult: any;
let testProviderResult: any;

function makeProvider(name: string, overrides?: Partial<any>): any {
  return {
    name,
    default_endpoint: `https://${name}.example/v1`,
    api_key_env: `${name.toUpperCase()}_API_KEY`,
    enabled: true,
    has_api_key: true,
    base_url_override: null,
    effective_endpoint: `https://${name}.example/v1`,
    ...overrides,
  };
}

beforeEach(() => {
  apiCalls = [];
  notif = mockNotifications();
  emitted = [];

  listProvidersResult = { ok: true, data: { providers: [] } };
  updateProviderResult = { ok: true, data: { providers: [] } };
  testProviderResult = { ok: true, data: { ok: true } };

  repoState.providers = [];

  initRepoDom(createDomRoot(REPO_DOM_IDS, REPO_DOM_EXTRAS, REPO_DOM_TAGS));
  initRepoDeps({
    api: {
      listProviders: async () => {
        apiCalls.push({ method: "listProviders", args: [] });
        return listProvidersResult;
      },
      updateProvider: async (name: string, patch: any) => {
        apiCalls.push({ method: "updateProvider", args: [name, patch] });
        return updateProviderResult;
      },
      testProvider: async (name: string) => {
        apiCalls.push({ method: "testProvider", args: [name] });
        return testProviderResult;
      },
    } as any,
    events: { emit: (name: string, data: unknown) => emitted.push({ name, data }), on: () => ({ dispose() {} }) } as any,
    notifications: notif.mock as any,
    dialogs: {} as any,
    timers: mockTimers(),
    format: {} as any,
    status: { setBusy: () => {}, setReady: () => {} } as any,
  });
});

describe("providers-data", () => {
  test("loadProviders populates state and disabled-provider dropdown", async () => {
    listProvidersResult = {
      ok: true,
      data: {
        providers: [
          makeProvider("openai"),
          makeProvider("anthropic", { enabled: false, has_api_key: false }),
        ],
      },
    };

    await loadProviders();

    expect(repoState.providers.map((p) => p.name)).toEqual(["openai", "anthropic"]);
    const addOptions = Array.from(getRepoDom().addProviderSelect.querySelectorAll("option")).map((opt) => ({
      value: opt.value,
      text: opt.textContent,
    }));
    expect(addOptions).toEqual([
      { value: "", text: "+ Add provider…" },
      { value: "anthropic", text: "anthropic" },
    ]);
    expect(getRepoDom().providersList.textContent).toContain("local");
    expect(getRepoDom().providersList.textContent).toContain("openai");
  });

  test("loadProviders failure clears state and shows error", async () => {
    repoState.providers = [makeProvider("openai")];
    listProvidersResult = { ok: false, error: "boom" };

    await loadProviders();

    expect(repoState.providers).toEqual([]);
    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("Failed to load providers: boom"))).toBe(true);
  });

  test("updateProvider success updates state, emits change, and re-renders", async () => {
    updateProviderResult = {
      ok: true,
      data: { providers: [makeProvider("openai"), makeProvider("anthropic")] },
    };

    await updateProvider("openai", { enabled: true, api_key: "sk-test", base_url: "https://override.example" });

    expect(apiCalls).toContainEqual({
      method: "updateProvider",
      args: ["openai", { enabled: true, api_key: "sk-test", base_url: "https://override.example" }],
    });
    expect(repoState.providers.map((p) => p.name)).toEqual(["openai", "anthropic"]);
    expect(emitted).toContainEqual({ name: "repo.providers.changed", data: {} });
    expect(notif.messages).toContainEqual({ type: "success", msg: "Updated openai" });
    expect(getRepoDom().providersList.textContent).toContain("anthropic");
  });

  test("addProvider and removeProvider send the expected enabled patch", async () => {
    await addProvider("openai");
    await removeProvider("openai");

    expect(apiCalls[0]).toEqual({ method: "updateProvider", args: ["openai", { enabled: true }] });
    expect(apiCalls[1]).toEqual({ method: "updateProvider", args: ["openai", { enabled: false }] });
  });

  test("testProvider returns request fallback error when API call fails", async () => {
    testProviderResult = { ok: false };

    const result = await testProvider("openai");

    expect(result).toEqual({ ok: false, error: "Request failed" });
  });
});
