import { beforeEach, afterEach, describe, expect, spyOn, test } from "bun:test";
import { renderProviders, wireProviderEvents } from "../../../src/plugins/repo/providers-render.ts";
import { initRepoDeps } from "../../../src/plugins/repo/deps.ts";
import { initRepoDom, getRepoDom } from "../../../src/plugins/repo/dom.ts";
import { repoState } from "../../../src/plugins/repo/state.ts";
import { createDomRoot, REPO_DOM_EXTRAS, REPO_DOM_IDS, REPO_DOM_TAGS } from "../../helpers/dom.ts";
import { mockNotifications, mockTimers } from "../../helpers/mocks.ts";

type ProviderLike = {
  name: string;
  default_endpoint: string;
  api_key_env: string | null;
  enabled: boolean;
  has_api_key: boolean;
  base_url_override: string | null;
  effective_endpoint: string;
};

let apiCalls: { method: string; args: unknown[] }[];
let notif: ReturnType<typeof mockNotifications>;
let listRepoModelsResult: any;
let listProviderModelsResult: any;
let kvPutResult: any;
let updateProviderResult: any;
let testProviderResult: any;
let restoreTimeouts: (() => void) | null;

function makeProvider(name: string, overrides: Partial<ProviderLike> = {}): ProviderLike {
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

function flushMicrotasks(): Promise<void> {
  return Promise.resolve().then(() => Promise.resolve());
}

beforeEach(() => {
  apiCalls = [];
  notif = mockNotifications();
  listRepoModelsResult = { ok: true, data: { models: [], total_size_bytes: 0 } };
  listProviderModelsResult = { ok: true, data: { models: [] } };
  kvPutResult = { ok: true };
  updateProviderResult = { ok: true, data: { providers: [] } };
  testProviderResult = { ok: true, data: { ok: true } };
  restoreTimeouts = null;

  repoState.providers = [];
  repoState.chatModels = [];
  repoState.browseModels.clear();
  repoState.models = [];
  repoState.subPage = null;
  repoState.manageLocalTab = "local";
  repoState.searchQuery = "";
  repoState.selectedIds.clear();
  repoState.isLoading = false;

  initRepoDom(createDomRoot(REPO_DOM_IDS, REPO_DOM_EXTRAS, REPO_DOM_TAGS));
  initRepoDeps({
    api: {
      listRepoModels: async (query?: string) => {
        apiCalls.push({ method: "listRepoModels", args: [query] });
        return listRepoModelsResult;
      },
      listProviderModels: async (name: string) => {
        apiCalls.push({ method: "listProviderModels", args: [name] });
        return listProviderModelsResult;
      },
      kvPut: async (ns: string, key: string, value: string) => {
        apiCalls.push({ method: "kvPut", args: [ns, key, value] });
        return kvPutResult;
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
    notifications: notif.mock as any,
    dialogs: {} as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    timers: mockTimers(),
    format: {
      date: () => "",
      dateTime: () => "",
      relativeTime: () => "",
      duration: () => "",
      number: () => "",
    } as any,
    status: { setBusy: () => {}, setReady: () => {} } as any,
  });
});

afterEach(() => {
  restoreTimeouts?.();
  restoreTimeouts = null;
});

describe("providers-render", () => {
  test("renderProviders shows local row, enabled providers, and disabled add-provider entries", () => {
    repoState.providers = [
      makeProvider("openai", { effective_endpoint: "https://openai.example/v1" }),
      makeProvider("anthropic", {
        enabled: false,
        has_api_key: false,
        effective_endpoint: "https://anthropic.example/v1",
      }),
    ];

    renderProviders();

    const dom = getRepoDom();
    expect(dom.providersList.querySelectorAll('[data-provider="local"]')).toHaveLength(1);
    expect(dom.providersList.querySelectorAll('[data-provider="openai"]')).toHaveLength(1);
    expect(dom.providersList.querySelector('[data-provider="anthropic"]')).toBeNull();
    expect(dom.providersList.textContent).toContain("Pin models to add them to chat");
    expect(dom.providersList.textContent).toContain("Key set");

    const addOptions = Array.from(dom.addProviderSelect.querySelectorAll("option")).map((opt) => ({
      value: opt.value,
      text: opt.textContent,
    }));
    expect(addOptions).toEqual([
      { value: "", text: "+ Add provider…" },
      { value: "anthropic", text: "anthropic" },
    ]);
    expect(dom.addProviderSelect.classList.contains("hidden")).toBe(false);

    const openaiRow = dom.providersList.querySelector('[data-provider="openai"]')!;
    expect(Array.from(openaiRow.querySelectorAll("[data-action]")).map((el) => el.getAttribute("data-action"))).toEqual([
      "browse",
      "test",
      "expand",
      "remove",
      "save",
    ]);
    expect(openaiRow.querySelector('[data-action="browse"]')!.textContent).toBe("Browse");
    expect(openaiRow.querySelector('[data-action="test"]')!.textContent).toBe("Test");
    expect(openaiRow.querySelector('[data-action="expand"]')!.textContent).toBe("Edit");
    expect(openaiRow.querySelector('[data-action="remove"]')).not.toBeNull();
  });

  test("browse action loads models, renders browse items, and add-model adds chat model", async () => {
    repoState.providers = [makeProvider("openai")];
    listProviderModelsResult = {
      ok: true,
      data: { models: [{ id: "gpt-4.1", object: "model", created: 1, owned_by: "openai" }] },
    };

    renderProviders();
    const dom = getRepoDom();
    wireProviderEvents(dom.providersList);

    const row = dom.providersList.querySelector('[data-provider="openai"]')!;
    const browseBtn = row.querySelector<HTMLButtonElement>('[data-action="browse"]')!;

    browseBtn.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await flushMicrotasks();

    const browseList = row.querySelector<HTMLElement>(".repo-browse-list")!;
    expect(browseBtn.textContent).toBe("Close");
    expect(browseList.classList.contains("hidden")).toBe(false);
    expect(row.querySelectorAll(".repo-browse-item")).toHaveLength(1);
    expect(row.querySelector('[data-action="add-model"]')).not.toBeNull();
    expect(apiCalls).toContainEqual({ method: "listProviderModels", args: ["openai"] });

    const addBtn = row.querySelector<HTMLButtonElement>('[data-action="add-model"]')!;
    addBtn.dispatchEvent(new MouseEvent("click", { bubbles: true }));

    expect(repoState.chatModels).toEqual(["openai::gpt-4.1"]);
    expect(apiCalls).toContainEqual({
      method: "kvPut",
      args: ["chat_models", "models", JSON.stringify(["openai::gpt-4.1"])],
    });
    expect(row.querySelector(".repo-browse-added")!.textContent).toBe("✓ Added");
    expect(row.querySelector('[data-action="add-model"]')).toBeNull();
  });

  test("save action trims inputs, persists the provider, and rerenders on success", async () => {
    repoState.providers = [makeProvider("openai", { effective_endpoint: "https://openai.example/v1" })];
    updateProviderResult = {
      ok: true,
      data: {
        providers: [
          makeProvider("openai", {
            base_url_override: "https://override.example",
            effective_endpoint: "https://override.example",
          }),
        ],
      },
    };

    renderProviders();
    const dom = getRepoDom();
    wireProviderEvents(dom.providersList);

    const row = dom.providersList.querySelector('[data-provider="openai"]')!;
    const apiKeyInput = row.querySelector<HTMLInputElement>("[data-field='api_key']")!;
    const baseUrlInput = row.querySelector<HTMLInputElement>("[data-field='base_url']")!;
    apiKeyInput.value = "  sk-test  ";
    baseUrlInput.value = "  https://override.example  ";

    row.querySelector<HTMLButtonElement>('[data-action="save"]')!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await flushMicrotasks();

    expect(apiCalls).toContainEqual({
      method: "updateProvider",
      args: ["openai", { enabled: true, api_key: "sk-test", base_url: "https://override.example" }],
    });
    expect(repoState.providers[0]!.effective_endpoint).toBe("https://override.example");
    expect(notif.messages).toContainEqual({ type: "success", msg: "Updated openai" });
    expect(getRepoDom().providersList.textContent).toContain("https://override.example");
  });

  test("test action updates button state and resets after the timer fires", async () => {
    const resetCallbacks: (() => void)[] = [];
    const setTimeoutSpy = spyOn(globalThis, "setTimeout").mockImplementation(((cb: TimerHandler) => {
      resetCallbacks.push(cb as () => void);
      return 1 as any;
    }) as typeof setTimeout);
    restoreTimeouts = () => {
      setTimeoutSpy.mockRestore();
    };

    repoState.providers = [makeProvider("openai")];
    testProviderResult = { ok: true, data: { ok: true } };

    renderProviders();
    const dom = getRepoDom();
    wireProviderEvents(dom.providersList);

    const row = dom.providersList.querySelector('[data-provider="openai"]')!;
    const testBtn = row.querySelector<HTMLButtonElement>('[data-action="test"]')!;
    testBtn.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await flushMicrotasks();

    expect(apiCalls).toContainEqual({ method: "testProvider", args: ["openai"] });
    expect(testBtn.textContent).toBe("OK");
    expect(testBtn.classList.contains("test-ok")).toBe(true);
    expect(testBtn.hasAttribute("disabled")).toBe(false);
    expect(resetCallbacks).toHaveLength(1);

    resetCallbacks[0]!();

    expect(testBtn.textContent).toBe("Test");
    expect(testBtn.classList.contains("test-ok")).toBe(false);
    expect(testBtn.classList.contains("test-fail")).toBe(false);
    expect(testBtn.title).toBe("");
  });

  test("test action shows failure state before reset when the provider health check fails", async () => {
    const resetCallbacks: (() => void)[] = [];
    const setTimeoutSpy = spyOn(globalThis, "setTimeout").mockImplementation(((cb: TimerHandler) => {
      resetCallbacks.push(cb as () => void);
      return 1 as any;
    }) as typeof setTimeout);
    restoreTimeouts = () => {
      setTimeoutSpy.mockRestore();
    };

    repoState.providers = [makeProvider("openai")];
    testProviderResult = { ok: false, error: "No route to provider" };

    renderProviders();
    const dom = getRepoDom();
    wireProviderEvents(dom.providersList);

    const row = dom.providersList.querySelector('[data-provider="openai"]')!;
    const testBtn = row.querySelector<HTMLButtonElement>('[data-action="test"]')!;
    testBtn.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await flushMicrotasks();

    expect(testBtn.textContent).toBe("Fail");
    expect(testBtn.classList.contains("test-fail")).toBe(true);
    expect(testBtn.title).toBe("No route to provider");
    expect(resetCallbacks).toHaveLength(1);

    resetCallbacks[0]!();

    expect(testBtn.textContent).toBe("Test");
    expect(testBtn.classList.contains("test-fail")).toBe(false);
    expect(testBtn.title).toBe("");
  });

  test("manage-local action switches sub-page, clears selection, and loads models", async () => {
    repoState.providers = [makeProvider("openai")];
    repoState.searchQuery = "stale query";
    repoState.selectedIds.add("m1");
    repoState.manageLocalTab = "discover";
    listRepoModelsResult = {
      ok: true,
      data: {
        models: [{
          id: "local-model",
          path: "/models/local-model",
          source: "managed",
          size_bytes: 1024,
          mtime: 1700000000,
          architecture: "llama",
          quant_scheme: "Q4_K_M",
          pinned: false,
        }],
        total_size_bytes: 1024,
      },
    };

    renderProviders();
    const dom = getRepoDom();
    wireProviderEvents(dom.providersList);

    dom.search.value = "stale query";
    dom.searchClear.classList.remove("hidden");

    const localBtn = dom.providersList.querySelector<HTMLButtonElement>('[data-action="manage-local"]')!;
    localBtn.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await flushMicrotasks();

    expect(repoState.subPage).toBe("manage-local");
    expect(repoState.manageLocalTab).toBe("local");
    expect(repoState.searchQuery).toBe("");
    expect(repoState.selectedIds.size).toBe(0);
    expect(dom.search.value).toBe("");
    expect(dom.searchClear.classList.contains("hidden")).toBe(true);
    expect(apiCalls).toContainEqual({ method: "listRepoModels", args: [""] });
    expect(getRepoDom().localTbody.querySelectorAll(".files-row")).toHaveLength(1);
  });
});
