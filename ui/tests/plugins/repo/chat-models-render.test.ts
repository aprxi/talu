import { beforeEach, describe, expect, test } from "bun:test";
import { wireChatModelEvents, renderChatModels } from "../../../src/plugins/repo/chat-models-render.ts";
import { initRepoDeps } from "../../../src/plugins/repo/deps.ts";
import { initRepoDom, getRepoDom } from "../../../src/plugins/repo/dom.ts";
import { repoState } from "../../../src/plugins/repo/state.ts";
import { createDomRoot, REPO_DOM_EXTRAS, REPO_DOM_IDS, REPO_DOM_TAGS } from "../../helpers/dom.ts";
import { flushAsync, mockNotifications, mockTimers } from "../../helpers/mocks.ts";

type ApiCall = { method: string; args: unknown[] };
type EventCall = { name: string; payload: unknown };

let apiCalls: ApiCall[];
let eventCalls: EventCall[];
let notifications: ReturnType<typeof mockNotifications>;

beforeEach(() => {
  apiCalls = [];
  eventCalls = [];
  notifications = mockNotifications();

  repoState.tab = "providers";
  repoState.subPage = null;
  repoState.manageLocalTab = "local";
  repoState.localSourceFilter = "all";
  repoState.models = [];
  repoState.totalSizeBytes = 0;
  repoState.searchResults = [];
  repoState.searchQuery = "";
  repoState.isLoading = false;
  repoState.activeDownloads.clear();
  repoState.selectedIds.clear();
  repoState.sortBy = "name";
  repoState.sortDir = "asc";
  repoState.searchGeneration = 0;
  repoState.discoverSort = "trending";
  repoState.discoverSize = "8";
  repoState.discoverTask = "text-generation";
  repoState.discoverLibrary = "safetensors";
  repoState.providers = [];
  repoState.chatModels = [];
  repoState.browseModels.clear();
  repoState.activeTerminalHostId = null;

  initRepoDom(createDomRoot(REPO_DOM_IDS, REPO_DOM_EXTRAS, REPO_DOM_TAGS));
  initRepoDeps({
    api: {
      kvPut: async (namespace: string, key: string, value: string) => {
        apiCalls.push({ method: "kvPut", args: [namespace, key, value] });
        return { ok: true };
      },
      unpinRepoModel: async (modelId: string) => {
        apiCalls.push({ method: "unpinRepoModel", args: [modelId] });
        return { ok: true };
      },
    } as any,
    events: {
      emit: (name: string, payload: unknown) => {
        eventCalls.push({ name, payload });
      },
      on: () => ({ dispose() {} }),
    } as any,
    notifications: notifications.mock as any,
    dialogs: {} as any,
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

function makeModel(
  id: string,
  opts: Partial<{
    source: string;
    sourceModelId: string;
    quantScheme: string;
    sizeBytes: number;
    pinned: boolean;
  }> = {},
): any {
  return {
    id,
    path: `/models/${id}`,
    source: opts.source ?? "managed",
    size_bytes: opts.sizeBytes ?? 1024,
    mtime: 1_700_000_000,
    architecture: "llama",
    quant_scheme: opts.quantScheme ?? "Q4_K_M",
    source_model_id: opts.sourceModelId,
    pinned: opts.pinned ?? false,
  };
}

function renderAndWire(): HTMLElement {
  renderChatModels();
  const dom = getRepoDom();
  wireChatModelEvents(dom.chatModelsList);
  return dom.chatModelsList;
}

function dispatchPointerEvent(
  target: EventTarget,
  type: string,
  init: { clientY: number; pointerId?: number } = { clientY: 0 },
): void {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperties(event, {
    clientY: { value: init.clientY, enumerable: true },
    pointerId: { value: init.pointerId ?? 1, enumerable: true },
  });
  target.dispatchEvent(event);
}

describe("chat-models-render", () => {
  test("remove-family click removes the whole family, unpins it, and persists the new order", async () => {
    repoState.models = [
      makeModel("alpha", { sourceModelId: "alpha", pinned: true, quantScheme: "Q4_K_M" }),
      makeModel("beta", { sourceModelId: "beta", pinned: true, quantScheme: "Q4_K_M" }),
    ];
    repoState.chatModels = ["alpha", "beta"];

    const chatModelsList = renderAndWire();
    expect(chatModelsList.querySelectorAll(".repo-chat-model-family")).toHaveLength(2);

    const removeBtn = chatModelsList
      .querySelector('[data-family-id="alpha"] [data-action="cm-remove-family"]') as HTMLButtonElement;
    removeBtn.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await flushAsync();

    expect(repoState.chatModels).toEqual(["beta"]);
    expect(Array.from(getRepoDom().chatModelsList.querySelectorAll(".repo-chat-model-family")).map((row) => row.dataset["familyId"]))
      .toEqual(["beta"]);
    expect(apiCalls).toContainEqual({ method: "unpinRepoModel", args: ["alpha"] });
    expect(apiCalls).toContainEqual({
      method: "kvPut",
      args: ["chat_models", "models", JSON.stringify(["beta"])],
    });
    expect(eventCalls.some((call) => call.name === "repo.chatModels.changed")).toBe(true);
  });

  test("select-variant click emits selectModel followed by openChat", () => {
    repoState.models = [
      makeModel("alpha-q4", { sourceModelId: "alpha", quantScheme: "Q4_K_M" }),
      makeModel("alpha-q8", { sourceModelId: "alpha", quantScheme: "Q8_0" }),
    ];
    repoState.chatModels = ["alpha-q4", "alpha-q8"];

    const chatModelsList = renderAndWire();
    const variantBtn = chatModelsList.querySelector(
      '[data-family-id="alpha"] [data-action="cm-select-variant"][data-variant-id="alpha-q8"]',
    ) as HTMLButtonElement;

    variantBtn.dispatchEvent(new MouseEvent("click", { bubbles: true }));

    expect(eventCalls).toEqual([
      { name: "repo.selectModel", payload: { modelId: "alpha-q8" } },
      { name: "repo.openChat", payload: {} },
    ]);
  });

  test("dragging a family grip reorders the family block and clears drag styles on release", async () => {
    repoState.models = [
      makeModel("alpha-q4", { sourceModelId: "alpha", quantScheme: "Q4_K_M" }),
      makeModel("alpha-q8", { sourceModelId: "alpha", quantScheme: "Q8_0" }),
      makeModel("beta-q4", { sourceModelId: "beta", quantScheme: "Q4_K_M" }),
    ];
    repoState.chatModels = ["alpha-q4", "alpha-q8", "beta-q4"];

    const chatModelsList = renderAndWire();
    const rows = Array.from(chatModelsList.querySelectorAll<HTMLElement>(".repo-chat-model-family"));
    expect(rows).toHaveLength(2);

    const alphaRow = rows[0]!;
    const betaRow = rows[1]!;
    Object.defineProperty(alphaRow, "getBoundingClientRect", {
      value: () => ({ height: 100 }),
      configurable: true,
    });

    const grip = alphaRow.querySelector<HTMLElement>(".repo-chat-model-grip")!;
    Object.defineProperty(grip, "setPointerCapture", {
      value: () => {},
      configurable: true,
    });

    dispatchPointerEvent(grip, "pointerdown", { clientY: 10, pointerId: 7 });
    expect(alphaRow.classList.contains("dragging")).toBe(true);
    expect(betaRow.classList.contains("displaced")).toBe(true);

    dispatchPointerEvent(chatModelsList, "pointermove", { clientY: 130, pointerId: 7 });
    expect(alphaRow.style.transform).toBe("translateY(120px)");
    expect(betaRow.style.transform).toBe("translateY(-100px)");

    dispatchPointerEvent(chatModelsList, "pointerup", { clientY: 130, pointerId: 7 });
    await flushAsync();

    expect(alphaRow.classList.contains("dragging")).toBe(false);
    expect(betaRow.classList.contains("displaced")).toBe(false);
    expect(repoState.chatModels).toEqual(["beta-q4", "alpha-q4", "alpha-q8"]);
    expect(Array.from(getRepoDom().chatModelsList.querySelectorAll(".repo-chat-model-family")).map((row) => row.dataset["familyId"]))
      .toEqual(["beta", "alpha"]);
    expect(apiCalls).toContainEqual({
      method: "kvPut",
      args: ["chat_models", "models", JSON.stringify(["beta-q4", "alpha-q4", "alpha-q8"])],
    });
  });
});
