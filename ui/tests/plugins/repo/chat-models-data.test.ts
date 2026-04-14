import { beforeEach, describe, expect, test } from "bun:test";
import {
  addChatModel,
  browseProviderModels,
  buildFamilyOrder,
  loadChatModels,
  removeChatModelFamily,
  reorderFamily,
  syncPinnedToChatModels,
} from "../../../src/plugins/repo/chat-models-data.ts";
import { initRepoDeps } from "../../../src/plugins/repo/deps.ts";
import { initRepoDom, getRepoDom } from "../../../src/plugins/repo/dom.ts";
import { repoState } from "../../../src/plugins/repo/state.ts";
import { createDomRoot, REPO_DOM_EXTRAS, REPO_DOM_IDS, REPO_DOM_TAGS } from "../../helpers/dom.ts";
import { mockNotifications, mockTimers } from "../../helpers/mocks.ts";

let apiCalls: { method: string; args: unknown[] }[];
let notif: ReturnType<typeof mockNotifications>;
let emitted: { name: string; data: any }[];

let kvGetResult: any;
let kvPutResult: any;
let listProviderModelsResult: any;

function makeModel(id: string, overrides?: Partial<any>): any {
  return {
    id,
    path: `/models/${id}`,
    source: "managed",
    size_bytes: 1024,
    mtime: 1700000000,
    architecture: "llama",
    quant_scheme: "Q4_K_M",
    pinned: false,
    source_model_id: null,
    ...overrides,
  };
}

beforeEach(() => {
  apiCalls = [];
  notif = mockNotifications();
  emitted = [];

  kvGetResult = { ok: true, data: { value: null } };
  kvPutResult = { ok: true };
  listProviderModelsResult = { ok: true, data: { models: [] } };

  repoState.models = [];
  repoState.chatModels = [];
  repoState.browseModels.clear();

  initRepoDom(createDomRoot(REPO_DOM_IDS, REPO_DOM_EXTRAS, REPO_DOM_TAGS));
  initRepoDeps({
    api: {
      kvGet: async (ns: string, key: string) => {
        apiCalls.push({ method: "kvGet", args: [ns, key] });
        return kvGetResult;
      },
      kvPut: async (ns: string, key: string, value: string) => {
        apiCalls.push({ method: "kvPut", args: [ns, key, value] });
        return kvPutResult;
      },
      listProviderModels: async (name: string) => {
        apiCalls.push({ method: "listProviderModels", args: [name] });
        return listProviderModelsResult;
      },
    } as any,
    events: { emit: (name: string, data: any) => emitted.push({ name, data }), on: () => ({ dispose() {} }) } as any,
    notifications: notif.mock as any,
    dialogs: {} as any,
    timers: mockTimers(),
    format: { dateTime: () => "" } as any,
    status: { setBusy: () => {}, setReady: () => {} } as any,
  });
});

describe("chat-models-data", () => {
  test("loadChatModels filters non-string entries and emits grouped families", async () => {
    repoState.models = [
      makeModel("llama-3-TQ4", { source_model_id: "llama-3", quant_scheme: "TQ4" }),
      makeModel("llama-3-TQ8", { source_model_id: "llama-3", quant_scheme: "TQ8" }),
    ];
    kvGetResult = {
      ok: true,
      data: { value: JSON.stringify(["llama-3-TQ4", 123, "llama-3-TQ8", null]) },
    };

    await loadChatModels();

    expect(repoState.chatModels).toEqual(["llama-3-TQ4", "llama-3-TQ8"]);
    expect(emitted).toHaveLength(1);
    expect(emitted[0]!.name).toBe("repo.chatModels.changed");
    expect(emitted[0]!.data.families).toEqual([
      {
        familyId: "llama-3",
        defaultVariant: "llama-3-TQ4",
        variants: [
          { id: "llama-3-TQ4", label: "TQ4", size_bytes: 1024 },
          { id: "llama-3-TQ8", label: "TQ8", size_bytes: 1024 },
        ],
      },
    ]);
  });

  test("addChatModel persists and renders newly added family", async () => {
    repoState.models = [makeModel("llama-3-TQ4", { source_model_id: "llama-3" })];

    await addChatModel("llama-3-TQ4");

    expect(repoState.chatModels).toEqual(["llama-3-TQ4"]);
    expect(apiCalls).toContainEqual({
      method: "kvPut",
      args: ["chat_models", "models", JSON.stringify(["llama-3-TQ4"])],
    });
    expect(getRepoDom().chatModelsList.textContent).toContain("llama-3");
  });

  test("removeChatModelFamily removes all family variants but preserves remote models", async () => {
    repoState.models = [
      makeModel("llama-3-TQ4", { source_model_id: "llama-3" }),
      makeModel("llama-3-TQ8", { source_model_id: "llama-3" }),
    ];
    repoState.chatModels = ["llama-3-TQ4", "openai::gpt-4o", "llama-3-TQ8"];

    await removeChatModelFamily("llama-3");

    expect(repoState.chatModels).toEqual(["openai::gpt-4o"]);
    expect(apiCalls).toContainEqual({
      method: "kvPut",
      args: ["chat_models", "models", JSON.stringify(["openai::gpt-4o"])],
    });
  });

  test("reorderFamily moves a family block as a unit", async () => {
    repoState.models = [
      makeModel("alpha-q4", { source_model_id: "alpha" }),
      makeModel("alpha-q8", { source_model_id: "alpha" }),
      makeModel("beta-q4", { source_model_id: "beta" }),
    ];
    repoState.chatModels = ["alpha-q4", "alpha-q8", "beta-q4"];

    await reorderFamily("beta", 0);

    expect(repoState.chatModels).toEqual(["beta-q4", "alpha-q4", "alpha-q8"]);
    expect(buildFamilyOrder()).toEqual([
      { familyId: "beta", modelIds: ["beta-q4"] },
      { familyId: "alpha", modelIds: ["alpha-q4", "alpha-q8"] },
    ]);
  });

  test("browseProviderModels caches successful responses", async () => {
    listProviderModelsResult = {
      ok: true,
      data: { models: [{ id: "gpt-4.1" }, { id: "gpt-4o" }] },
    };

    const first = await browseProviderModels("openai");
    const second = await browseProviderModels("openai");

    expect(first).toEqual([{ id: "gpt-4.1" }, { id: "gpt-4o" }]);
    expect(second).toEqual(first);
    expect(apiCalls.filter((c) => c.method === "listProviderModels")).toHaveLength(1);
  });

  test("syncPinnedToChatModels groups local pinned families and preserves remote entries", async () => {
    repoState.models = [
      makeModel("beta-q4", { source_model_id: "beta", pinned: true }),
      makeModel("alpha-q4", { source_model_id: "alpha", pinned: true }),
      makeModel("alpha-q8", { source_model_id: "alpha", pinned: true }),
      makeModel("gamma-q4", { source_model_id: "gamma", pinned: false }),
    ];
    repoState.chatModels = ["openai::gpt-4o"];

    await syncPinnedToChatModels();

    expect(repoState.chatModels).toEqual(["beta-q4", "alpha-q4", "alpha-q8", "openai::gpt-4o"]);
    expect(apiCalls).toContainEqual({
      method: "kvPut",
      args: ["chat_models", "models", JSON.stringify(["beta-q4", "alpha-q4", "alpha-q8", "openai::gpt-4o"])],
    });
  });
});
