import {
  CHECK_ICON as ICON_CHECK,
  COPY_ICON as ICON_COPY,
  DELETE_ICON as ICON_DELETE,
  SEARCH_ICON as ICON_SEARCH,
  SETTINGS_ICON as ICON_SETTINGS,
  CHEVRON_DOWN_ICON,
  CHEVRON_RIGHT_ICON,
} from "../icons.ts";
import { writeClipboardText } from "../kernel/system/clipboard.ts";
import { populateModelSelect, relativeTime } from "../render/helpers.ts";
import type { ModelEntry } from "../types.ts";

type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

type MajorGroupId = "responses" | "db";

const SCOPE_ORDER = [
  "responses/perf",
  "responses/evals/mmlu",
  "responses/evals/gpqa",
  "responses/evals/ifeval",
  "responses/evals/bfcl",
  "responses/evals/mmmu",
  "db/sql",
  "db/kv",
  "db/tables",
  "db/vectors",
  "db/blobs",
  "db/ops",
] as const;

type BenchScopeId = (typeof SCOPE_ORDER)[number];

type ScenarioContext = {
  baseUrl: string;
  round: number;
  state: Record<string, unknown>;
};

type RunConfig = {
  requests: number;
  concurrency: number;
  rounds: number;
  kvBatchSize: number;
  tableSeedRows: number;
  vectorDims: number;
  responsesModel: string;
  responsesMaxOutputTokens: number;
};

type LoadMetrics = {
  requests: number;
  ok: number;
  errors: number;
  wallSeconds: number;
  rps: number;
  avgMs: number;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  inputTokens: number;
  outputTokens: number;
  avgPrefillTokS: number;
  avgGenTokS: number;
  avgTtftMs: number;
};

type RunControlState = {
  paused: boolean;
  stopRequested: boolean;
};

type RunProgressTone = "idle" | "running" | "ok" | "error";

type RunProgressState = {
  scenarioId: string;
  round: number;
  totalRounds: number;
  completedRequests: number;
  totalRequests: number;
  phase: string;
  note: string;
  tone: RunProgressTone;
};

type BenchEventEnvelope = {
  ts_ms?: number;
  level?: string;
  topic?: string;
  event_class?: string;
  message?: string;
  data?: Record<string, unknown> | null;
};

const BENCH_EVENT_LEVELS = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"] as const;
type BenchEventLevel = (typeof BENCH_EVENT_LEVELS)[number];

type BenchEventLine = {
  level: BenchEventLevel;
  topic: string;
  message: string;
  text: string;
};

type BenchScenario = {
  id: string;
  title: string;
  scope: BenchScopeId;
  apiGroup: string;
  method: HttpMethod;
  pathTemplate: string;
  description: string;
  pythonName?: string;
  defaults?: Partial<RunConfig>;
  classifySuccess?: (
    response: Response,
    bodyText: string,
    index: number,
    ctx: ScenarioContext,
    cfg: RunConfig,
  ) => boolean | Promise<boolean>;
  prepare?: (ctx: ScenarioContext, cfg: RunConfig) => Promise<void>;
  request: (
    ctx: ScenarioContext,
    index: number,
    cfg: RunConfig,
  ) => Promise<Response>;
  cleanup?: (ctx: ScenarioContext, cfg: RunConfig) => Promise<void>;
};

type ScopeMeta = {
  id: BenchScopeId;
  title: string;
  description: string;
  major: MajorGroupId;
  apiGroup: string;
};

type PageScope = {
  mode: "exact" | "prefix";
  value: string;
};

type PageDef = {
  slug: string;
  title: string;
  subtitle: string;
  description: string;
  parent: string | null;
  children: string[];
  scope?: PageScope;
};

type ResolvedBenchPage =
  | { status: "ok"; requestedSlug: string; page: PageDef }
  | { status: "not_found"; requestedSlug: string };

const TEXT_ENCODER = new TextEncoder();
const MAX_BENCH_EVENT_LINES = 180;
const benchEventLines: BenchEventLine[] = [];
const benchEventSelectedLevels = new Set<BenchEventLevel>(BENCH_EVENT_LEVELS);
const benchEventDiscoveredTopics: string[] = [];
const benchEventSelectedTopics = new Set<string>();
let benchEventSearchText = "";
let benchEventsAbort: AbortController | null = null;
let benchAvailableModels: ModelEntry[] = [];
let benchSelectedVariant = "";
const BENCH_TREE_OPEN_KEY = "bench:tree-open:v1";
const BENCH_PENDING_SCENARIO_KEY = "bench:pending-scenario:v1";

function deduplicateByFamily(models: ModelEntry[]): ModelEntry[] {
  const families = new Map<string, { entry: ModelEntry; variants: { id: string; label: string; size_bytes?: number }[] }>();
  for (const m of models) {
    let familyKey = m.id;
    if (m.source === "managed") {
      const stripped = m.id.replace(/-GAF\d+(-G\d+)?$/, "");
      if (stripped !== m.id) familyKey = stripped;
    }
    const label = familyKey !== m.id
      ? m.id.slice(familyKey.length + 1)
      : (m.id.split("/").pop() ?? m.id);
    if (!families.has(familyKey)) {
      families.set(familyKey, {
        entry: { ...m, display_name: familyKey !== m.id ? familyKey : undefined },
        variants: [],
      });
    }
    families.get(familyKey)!.variants.push({ id: m.id, label });
  }
  return [...families.values()].map(({ entry, variants }) => ({
    ...entry,
    variants,
  }));
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const val = bytes / Math.pow(1024, i);
  return `${val < 10 ? val.toFixed(1) : Math.round(val)} ${units[i]}`;
}

function renderBenchModelVariants(entry: ModelEntry | undefined): void {
  const container = document.getElementById("bench-model-variants");
  if (!container) return;
  container.innerHTML = "";
  if (!entry?.variants || entry.variants.length === 0) {
    container.classList.add("hidden");
    return;
  }
  container.classList.remove("hidden");
  const multiVariant = entry.variants.length > 1;
  for (const v of entry.variants) {
    const pill = document.createElement("button");
    pill.type = "button";
    pill.className = "bench-model-variant";
    let text = v.label;
    if (v.size_bytes && v.size_bytes > 0) {
      text += ` \u00b7 ${formatBytes(v.size_bytes)}`;
    }
    pill.textContent = text;
    pill.title = v.id;
    if (v.id === benchSelectedVariant) pill.classList.add("active");
    if (multiVariant) {
      pill.addEventListener("click", () => {
        benchSelectedVariant = v.id;
        for (const p of container.children) {
          (p as HTMLElement).classList.toggle("active", p === pill);
        }
      });
    }
    container.appendChild(pill);
  }
}

async function fetchBenchModels(): Promise<void> {
  const baseUrl = window.location.origin.replace(/\/$/, "");
  try {
    // 1. Fetch settings for active model and the full managed model catalog.
    const settingsResp = await request(baseUrl, "GET", "/v1/settings");
    if (!settingsResp.ok) return;
    const settings = (await settingsResp.json()) as { model?: string; available_models?: ModelEntry[] };
    const allManaged = Array.isArray(settings.available_models) ? settings.available_models : [];
    const activeModel = typeof settings.model === "string" ? settings.model.trim() : "";

    // 2. Read user-curated chat models list from KV (same source as Chat > Router).
    //    The KV API returns { value_hex: "..." } — hex-encoded UTF-8.
    let chatModelIds: string[] = [];
    try {
      const kvResp = await request(baseUrl, "GET", "/v1/db/kv/namespaces/chat_models/entries/models");
      if (kvResp.ok) {
        const kvPayload = (await kvResp.json()) as { value_hex?: string };
        if (kvPayload.value_hex) {
          const bytes = new Uint8Array(kvPayload.value_hex.length / 2);
          for (let i = 0; i < kvPayload.value_hex.length; i += 2) {
            bytes[i / 2] = parseInt(kvPayload.value_hex.substring(i, i + 2), 16);
          }
          const decoded = new TextDecoder().decode(bytes);
          const parsed = JSON.parse(decoded);
          if (Array.isArray(parsed)) {
            chatModelIds = parsed.filter((x): x is string => typeof x === "string");
          }
        }
      }
    } catch { /* KV not available — fall back to all managed models */ }

    // 3. Build model list: if user curated a chat models list, use exactly that.
    //    Otherwise fall back to all managed models (same as chat with no curation).
    if (chatModelIds.length > 0) {
      // Build family map for local models (mirrors chat-models-data.ts emitChanged).
      const familyMap = new Map<string, { defaultVariant: string; variants: { id: string; label: string; size_bytes?: number }[] }>();
      for (const id of chatModelIds) {
        if (id.includes("::")) continue; // remote models handled separately
        const managed = allManaged.find((m) => m.id === id);
        // Infer family key: strip -GAF\d+(-G\d+)? suffix.
        let familyKey = id;
        const stripped = id.replace(/-GAF\d+(-G\d+)?$/, "");
        if (stripped !== id) familyKey = stripped;
        if (!familyMap.has(familyKey)) {
          familyMap.set(familyKey, { defaultVariant: id, variants: [] });
        }
        const label = familyKey !== id ? id.slice(familyKey.length + 1) : (id.split("/").pop() ?? id);
        familyMap.get(familyKey)!.variants.push({
          id,
          label: managed?.variants?.find((v) => v.id === id)?.label ?? label,
          size_bytes: managed?.variants?.find((v) => v.id === id)?.size_bytes,
        });
      }

      // Local family entries.
      const localEntries: ModelEntry[] = [...familyMap.entries()].map(([familyId, data]) => ({
        id: data.defaultVariant,
        display_name: familyId !== data.defaultVariant ? familyId : undefined,
        source: "managed" as const,
        defaults: { temperature: 1.0, top_k: 50, top_p: 1.0, do_sample: true },
        overrides: {},
        variants: data.variants,
      }));
      // Remote model entries (e.g. vllm::model-name).
      const remoteEntries: ModelEntry[] = chatModelIds
        .filter((id) => id.includes("::"))
        .map((id) => ({
          id,
          source: "hub" as const,
          defaults: { temperature: 1.0, top_k: 50, top_p: 1.0, do_sample: true },
          overrides: {},
        }));
      benchAvailableModels = [...localEntries, ...remoteEntries];
    } else {
      // No curated list — use all managed models (same as chat default).
      benchAvailableModels = deduplicateByFamily(allManaged);
    }

    // 4. Populate the <select> and variant pills.
    const sel = document.getElementById("bench-responses-model") as HTMLSelectElement | null;
    if (!sel) return;

    if (!benchSelectedVariant && activeModel) {
      benchSelectedVariant = activeModel;
    }
    populateModelSelect(sel, benchAvailableModels, benchSelectedVariant || activeModel);

    if (!benchSelectedVariant) {
      if (activeModel) {
        benchSelectedVariant = activeModel;
      } else if (benchAvailableModels.length > 0) {
        const first = benchAvailableModels[0]!;
        benchSelectedVariant = first.variants?.[0]?.id ?? first.id;
      }
    }

    const selectedEntry = benchAvailableModels.find((m) => m.id === sel.value)
      ?? benchAvailableModels.find((m) => m.variants?.some((v) => v.id === benchSelectedVariant));
    renderBenchModelVariants(selectedEntry);
  } catch (_) {
    // Silently fail — the model can still be resolved at run time.
  }
}

function toBase64(text: string): string {
  return btoa(text);
}

function uniqueName(prefix: string): string {
  const rand = Math.random().toString(36).slice(2, 8);
  return `${prefix}_${Date.now()}_${rand}`;
}

async function request(
  baseUrl: string,
  method: HttpMethod,
  path: string,
  options?: {
    json?: unknown;
    body?: BodyInit;
    headers?: Record<string, string>;
  },
): Promise<Response> {
  const headers = new Headers(options?.headers ?? {});
  let body: BodyInit | undefined = options?.body;
  if (options?.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.json);
  }
  return fetch(`${baseUrl}${path}`, { method, headers, body });
}

async function requestJson(
  baseUrl: string,
  method: HttpMethod,
  path: string,
  json: unknown,
): Promise<Response> {
  return request(baseUrl, method, path, { json });
}

async function seedKvNamespace(baseUrl: string, ns: string, count: number): Promise<void> {
  const entries = Array.from({ length: count }, (_, i) => ({
    key: `seed_${i}`,
    value_base64: toBase64(`value_${i}`),
    durability: "batched",
  }));
  const resp = await requestJson(
    baseUrl,
    "POST",
    `/v1/db/kv/namespaces/${encodeURIComponent(ns)}/batch`,
    { entries },
  );
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`kv seed failed: ${resp.status} ${txt}`);
  }
}

async function seedTableRows(baseUrl: string, ns: string, count: number): Promise<void> {
  for (let i = 0; i < count; i++) {
    const body = {
      schema_id: 10,
      columns: [
        { column_id: 1, type: "scalar_u64", value: i + 1 },
        { column_id: 2, type: "scalar_i64", value: 1_700_000_000_000 + i },
        { column_id: 20, type: "string", value: `seed-${i}` },
      ],
    };
    const resp = await requestJson(baseUrl, "POST", `/v1/db/tables/${ns}/rows`, body);
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`table seed failed: ${resp.status} ${txt}`);
    }
  }
}

async function createVectorCollection(
  baseUrl: string,
  collection: string,
  dims: number,
): Promise<void> {
  const resp = await requestJson(baseUrl, "POST", "/v1/db/vectors/collections", {
    name: collection,
    dims,
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`vector create failed: ${resp.status} ${txt}`);
  }
}

async function appendVectorPoints(
  baseUrl: string,
  collection: string,
  dims: number,
  count: number,
  idOffset: number = 1,
): Promise<void> {
  const vectors = Array.from({ length: count }, (_, i) => ({
    id: idOffset + i,
    values: Array.from({ length: dims }, (_, j) => (j === i % dims ? 1.0 : 0.0)),
  }));
  const resp = await requestJson(
    baseUrl,
    "POST",
    `/v1/db/vectors/collections/${encodeURIComponent(collection)}/points/append`,
    { vectors },
  );
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`vector append seed failed: ${resp.status} ${txt}`);
  }
}

function buildResponsesRequestBody(
  cfg: RunConfig,
  input: string,
  options?: {
    maxOutputTokens?: number;
    temperature?: number;
    topP?: number;
  },
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    input,
    stream: false,
    max_output_tokens: options?.maxOutputTokens ?? cfg.responsesMaxOutputTokens,
    temperature: options?.temperature ?? 0.2,
    top_p: options?.topP ?? 0.95,
  };
  if (cfg.responsesModel.trim().length > 0) {
    body.model = cfg.responsesModel.trim();
  }
  return body;
}

const PREFILL_WORD_BLOCK = "one two three four five six seven eight nine ten ";
const PREFILL_TOKEN_CALIBRATION: Record<number, { reps: number; extra: number }> = {
  512: { reps: 49, extra: 4 },
  1024: { reps: 100, extra: 6 },
  2048: { reps: 203, extra: 0 },
  4096: { reps: 407, extra: 8 },
};
const DECODE_PROMPT =
  "Write a detailed story about a knight exploring a vast underground kingdom. Describe the caverns, creatures, ruins, and a mysterious queen in vivid detail.";
const DECODE_INSTRUCTIONS =
  "You are a novelist. Write in flowing prose, no bullets, no summaries.";

function makePrefillPrompt(targetTokens: number, requestIndex: number): string {
  const calibrated = PREFILL_TOKEN_CALIBRATION[targetTokens];
  const reps = calibrated?.reps ?? Math.max(8, Math.floor(targetTokens / 10));
  const extra = calibrated?.extra ?? 0;
  const base = PREFILL_WORD_BLOCK.repeat(reps);
  const tail = extra > 0 ? PREFILL_WORD_BLOCK.split(" ").slice(0, extra).join(" ") : "";
  return `${base}${tail}\n\nindex=${requestIndex} target=${targetTokens}`;
}

function extractResponseOutputText(payload: unknown): string {
  if (!payload || typeof payload !== "object") {
    return "";
  }
  const root = payload as Record<string, unknown>;
  const direct = root.output_text;
  if (typeof direct === "string" && direct.length > 0) {
    return direct;
  }

  const output = root.output;
  if (!Array.isArray(output)) {
    return "";
  }

  const chunks: string[] = [];
  for (const item of output) {
    if (!item || typeof item !== "object") continue;
    const content = (item as Record<string, unknown>).content;
    if (!Array.isArray(content)) continue;
    for (const part of content) {
      if (!part || typeof part !== "object") continue;
      const partObj = part as Record<string, unknown>;
      const partType = partObj.type;
      const text = partObj.text;
      if (
        (partType === "output_text" || partType === "input_text") &&
        typeof text === "string"
      ) {
        chunks.push(text);
      }
    }
  }
  return chunks.join("\n").trim();
}

function normalizeEvalText(text: string): string {
  return text.trim().toLowerCase().replace(/\s+/g, " ");
}

function stripMarkdownCodeFence(text: string): string {
  const trimmed = text.trim();
  const fence = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  return fence ? fence[1]!.trim() : trimmed;
}

function parseObjectFromText(text: string): Record<string, unknown> | null {
  const cleaned = stripMarkdownCodeFence(text);
  const candidates = [cleaned];

  const first = cleaned.indexOf("{");
  const last = cleaned.lastIndexOf("}");
  if (first >= 0 && last > first) {
    candidates.push(cleaned.slice(first, last + 1));
  }

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch (_err) {
      // keep trying
    }
  }
  return null;
}

function parseResponsePayload(bodyText: string): unknown | null {
  try {
    return JSON.parse(bodyText) as unknown;
  } catch (_err) {
    return null;
  }
}

function toFiniteNumber(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return 0;
}

function extractResponseUsage(bodyText: string): {
  inputTokens: number;
  outputTokens: number;
  prefillTokS: number;
  genTokS: number;
  ttftMs: number;
} {
  const payload = parseResponsePayload(bodyText);
  if (!payload || typeof payload !== "object") {
    return { inputTokens: 0, outputTokens: 0, prefillTokS: 0, genTokS: 0, ttftMs: 0 };
  }
  const usageRaw = (payload as Record<string, unknown>).usage;
  const usage =
    usageRaw && typeof usageRaw === "object" && !Array.isArray(usageRaw)
      ? (usageRaw as Record<string, unknown>)
      : {};
  const inputTokens = toFiniteNumber(usage.input_tokens);
  const outputTokens = toFiniteNumber(usage.output_tokens);
  const prefillMs = toFiniteNumber(usage.prefill_ms);
  const generationMs = toFiniteNumber(usage.generation_ms);
  const ttftMs = toFiniteNumber(usage.ttft_ms);
  const prefillTokS = prefillMs > 0 ? inputTokens / (prefillMs / 1000) : 0;
  const genTokS = generationMs > 0 ? outputTokens / (generationMs / 1000) : 0;
  return { inputTokens, outputTokens, prefillTokS, genTokS, ttftMs };
}

function extractChoiceLetter(text: string): string | null {
  const trimmed = text.trim().toUpperCase();
  if (trimmed === "A" || trimmed === "B" || trimmed === "C" || trimmed === "D") {
    return trimmed;
  }
  const match = trimmed.match(/\b([ABCD])\b/);
  return match?.[1] ?? null;
}

function deepEqualJson(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== typeof b) return false;
  if (a === null || b === null) return a === b;
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    return a.every((item, idx) => deepEqualJson(item, b[idx]));
  }
  if (typeof a === "object" && typeof b === "object") {
    const aObj = a as Record<string, unknown>;
    const bObj = b as Record<string, unknown>;
    const aKeys = Object.keys(aObj).sort();
    const bKeys = Object.keys(bObj).sort();
    if (!deepEqualJson(aKeys, bKeys)) return false;
    return aKeys.every((key) => deepEqualJson(aObj[key], bObj[key]));
  }
  return false;
}

function extractResponseToolCalls(payload: unknown): Array<{ name: string; arguments: Record<string, unknown> }> {
  if (!payload || typeof payload !== "object") {
    return [];
  }
  const root = payload as Record<string, unknown>;
  const output = root.output;
  if (!Array.isArray(output)) {
    return [];
  }

  const calls: Array<{ name: string; arguments: Record<string, unknown> }> = [];
  for (const item of output) {
    if (!item || typeof item !== "object") continue;
    const obj = item as Record<string, unknown>;
    if (obj.type !== "function_call") continue;
    const name = typeof obj.name === "string" ? obj.name : "";
    let args: Record<string, unknown> = {};
    if (typeof obj.arguments === "string") {
      try {
        const parsed = JSON.parse(obj.arguments) as unknown;
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          args = parsed as Record<string, unknown>;
        }
      } catch (_err) {
        args = {};
      }
    } else if (obj.arguments && typeof obj.arguments === "object" && !Array.isArray(obj.arguments)) {
      args = obj.arguments as Record<string, unknown>;
    }
    calls.push({ name, arguments: args });
  }
  return calls;
}

function scoreBfclCalls(
  actual: Array<{ name: string; arguments: Record<string, unknown> }>,
  expected: Array<{ name: string; arguments: Record<string, unknown> }>,
  mode: "simple" | "parallel" | "irrelevance",
): boolean {
  if (mode === "irrelevance") {
    return actual.length === 0;
  }

  const callMatches = (
    a: { name: string; arguments: Record<string, unknown> },
    e: { name: string; arguments: Record<string, unknown> },
  ): boolean => {
    if (a.name !== e.name) return false;
    for (const [key, value] of Object.entries(e.arguments)) {
      if (!deepEqualJson(a.arguments[key], value)) {
        return false;
      }
    }
    return true;
  };

  if (mode === "simple") {
    return actual.length >= 1 && callMatches(actual[0], expected[0]!);
  }

  if (actual.length < expected.length) {
    return false;
  }
  const used = new Set<number>();
  for (const expectedCall of expected) {
    let found = false;
    for (let i = 0; i < actual.length; i++) {
      if (used.has(i)) continue;
      if (callMatches(actual[i]!, expectedCall)) {
        used.add(i);
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

function scoreIfevalLite(
  outputText: string,
  rule: {
    mustInclude?: string[];
    forbidden?: string[];
    exactText?: string;
    jsonExpected?: Record<string, unknown>;
    maxWords?: number;
    endsWith?: string;
  },
): boolean {
  const normalized = normalizeEvalText(outputText);
  if (rule.exactText && normalized !== normalizeEvalText(rule.exactText)) {
    return false;
  }
  if (rule.mustInclude) {
    for (const token of rule.mustInclude) {
      if (!normalized.includes(normalizeEvalText(token))) {
        return false;
      }
    }
  }
  if (rule.forbidden) {
    for (const token of rule.forbidden) {
      if (normalized.includes(normalizeEvalText(token))) {
        return false;
      }
    }
  }
  if (rule.endsWith) {
    if (!outputText.trim().toLowerCase().endsWith(rule.endsWith.trim().toLowerCase())) {
      return false;
    }
  }
  if (typeof rule.maxWords === "number") {
    const words = outputText.trim().split(/\s+/).filter((token) => token.length > 0);
    if (words.length > rule.maxWords) {
      return false;
    }
  }
  if (rule.jsonExpected) {
    const parsed = parseObjectFromText(outputText);
    if (!parsed) return false;
    for (const [key, expectedValue] of Object.entries(rule.jsonExpected)) {
      if (!deepEqualJson(parsed[key], expectedValue)) {
        return false;
      }
    }
  }
  return true;
}

async function uploadSvgImageFile(baseUrl: string, filename: string, svg: string): Promise<string | null> {
  try {
    const formData = new FormData();
    const blob = new Blob([svg], { type: "image/svg+xml" });
    formData.append("file", blob, filename);
    const resp = await fetch(`${baseUrl}/v1/files`, {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) {
      return null;
    }
    const payload = (await resp.json()) as { id?: string };
    return typeof payload.id === "string" && payload.id.length > 0 ? payload.id : null;
  } catch (_err) {
    return null;
  }
}

async function uploadBenchArtifact(baseUrl: string, filename: string, content: string): Promise<string | null> {
  try {
    const formData = new FormData();
    const blob = new Blob([content], { type: "application/x-ndjson" });
    formData.append("file", blob, filename);
    const resp = await fetch(`${baseUrl}/v1/files`, {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) return null;
    const payload = (await resp.json()) as { id?: string };
    return typeof payload.id === "string" && payload.id.length > 0 ? payload.id : null;
  } catch (_err) {
    return null;
  }
}

async function persistBenchRun(
  baseUrl: string,
  scenario: BenchScenario,
  cfg: RunConfig,
  rows: Array<{ round: number; metrics: LoadMetrics }>,
  stopped: boolean,
): Promise<{ fileId: string | null; docId: string | null }> {
  const ts = new Date();
  const stamp = ts.toISOString().replace(/[:.]/g, "-");

  const meta = {
    type: "meta",
    suite: "ui-bench",
    scenario: scenario.id,
    python_scenario: scenario.pythonName ?? null,
    scope: scenario.scope,
    api_group: scenario.apiGroup,
    stopped,
    config: cfg,
    created_at: ts.toISOString(),
  };
  const entries = rows.map((item) => ({
    type: "round",
    scenario: scenario.id,
    round: item.round,
    ...item.metrics,
  }));
  const totalReq = rows.reduce((sum, item) => sum + item.metrics.requests, 0);
  const totalOk = rows.reduce((sum, item) => sum + item.metrics.ok, 0);
  const successPct = totalReq > 0 ? (100 * totalOk) / totalReq : 0;
  const avgRps = rows.length
    ? rows.reduce((sum, item) => sum + item.metrics.rps, 0) / rows.length
    : 0;
  const avgP95 = rows.length
    ? rows.reduce((sum, item) => sum + item.metrics.p95Ms, 0) / rows.length
    : 0;
  const avgP99 = rows.length
    ? rows.reduce((sum, item) => sum + item.metrics.p99Ms, 0) / rows.length
    : 0;
  const avgWall = rows.length
    ? rows.reduce((sum, item) => sum + item.metrics.wallSeconds, 0) / rows.length
    : 0;
  const totalInputTokens = rows.reduce((sum, item) => sum + item.metrics.inputTokens, 0);
  const totalOutputTokens = rows.reduce((sum, item) => sum + item.metrics.outputTokens, 0);
  const avgPrefillTokS = rows.length
    ? rows.reduce((sum, item) => sum + item.metrics.avgPrefillTokS, 0) / rows.length
    : 0;
  const avgGenTokS = rows.length
    ? rows.reduce((sum, item) => sum + item.metrics.avgGenTokS, 0) / rows.length
    : 0;
  const avgTtftMs = rows.length
    ? rows.reduce((sum, item) => sum + item.metrics.avgTtftMs, 0) / rows.length
    : 0;

  const jsonl = [meta, ...entries].map((entry) => JSON.stringify(entry)).join("\n");
  const filename = `bench_${scenario.id}_${stamp}.jsonl`;
  const fileId = await uploadBenchArtifact(baseUrl, filename, `${jsonl}\n`);

  let docId: string | null = null;
  try {
    const content: Record<string, unknown> = {
      scenario: scenario.id,
      python_scenario: scenario.pythonName ?? null,
      scope: scenario.scope,
      stopped,
      rounds: rows.length,
      summary: {
        success_pct: round2(successPct),
        avg_rps: round2(avgRps),
        avg_p95_ms: round2(avgP95),
        avg_p99_ms: round2(avgP99),
        avg_wall_s: round3(avgWall),
        total_input_tokens: Math.round(totalInputTokens),
        total_output_tokens: Math.round(totalOutputTokens),
        avg_prefill_tok_s: round2(avgPrefillTokS),
        avg_gen_tok_s: round2(avgGenTokS),
        avg_ttft_ms: round2(avgTtftMs),
      },
      cfg,
      file_id: fileId,
      file_name: filename,
      generated_at: ts.toISOString(),
    };
    const docResp = await requestJson(baseUrl, "POST", "/v1/db/tables/documents", {
      type: "bench_run",
      title: scenario.id,
      content,
      marker: scenario.scope,
      group_id: scenario.scope,
      tags_text: `bench,${scenario.scope.replace("/", "_")}`,
      meta_f1: parseFloat(round2(avgGenTokS)) || undefined,
      meta_f2: parseFloat(round2(avgTtftMs)) || undefined,
      meta_f3: parseFloat(round2(avgPrefillTokS)) || undefined,
      meta_f4: parseFloat(round2(avgRps)) || undefined,
      meta_f5: parseFloat(round2(successPct)) || undefined,
      meta_i1: Math.round(totalInputTokens) || undefined,
      meta_i2: Math.round(totalOutputTokens) || undefined,
      meta_i3: rows.length || undefined,
    });
    if (docResp.ok) {
      const payload = (await docResp.json()) as { id?: string };
      docId = typeof payload.id === "string" ? payload.id : null;
    }
  } catch (_err) {
    docId = null;
  }

  return { fileId, docId };
}

const RESPONSES_PREFILL_SCENARIOS: BenchScenario[] = [512, 1024, 2048, 4096].map(
  (targetTokens) => ({
    id: `responses/perf/pp${targetTokens}`,
    title: `Prefill pp${targetTokens}`,
    scope: "responses/perf",
    apiGroup: "/v1/responses/perf",
    method: "POST",
    pathTemplate: "/v1/responses",
    description: `Prefill-focused workload targeting ~${targetTokens} input tokens with 1 output token.`,
    pythonName: `responses/perf/pp${targetTokens}`,
    defaults: {
      requests: 1,
      concurrency: 1,
      rounds: 1,
      responsesMaxOutputTokens: 1,
    },
    request: (ctx, idx, cfg) =>
      requestJson(
        ctx.baseUrl,
        "POST",
        "/v1/responses",
        buildResponsesRequestBody(cfg, makePrefillPrompt(targetTokens, idx), {
          maxOutputTokens: 1,
          temperature: 0.0,
          topP: 1.0,
        }),
      ),
  }),
);

const RESPONSES_DECODE_SCENARIOS: BenchScenario[] = [128, 256, 512].map(
  (maxOutputTokens) => ({
    id: `responses/perf/tg${maxOutputTokens}`,
    title: `Decode tg${maxOutputTokens}`,
    scope: "responses/perf",
    apiGroup: "/v1/responses/perf",
    method: "POST",
    pathTemplate: "/v1/responses",
    description: `Decode throughput with max_output_tokens=${maxOutputTokens}. Tune parallelism via run settings.`,
    pythonName: `responses/perf/tg${maxOutputTokens}`,
    defaults: {
      requests: 1,
      concurrency: 1,
      rounds: 1,
      responsesMaxOutputTokens: maxOutputTokens,
    },
    request: (ctx, idx, cfg) => {
      const body: Record<string, unknown> = {
        input: `${DECODE_PROMPT}\nrequest=${idx}`,
        instructions: DECODE_INSTRUCTIONS,
        stream: false,
        max_output_tokens: maxOutputTokens,
        temperature: 0.2,
        top_p: 0.95,
      };
      if (cfg.responsesModel.trim().length > 0) {
        body.model = cfg.responsesModel.trim();
      }
      return requestJson(ctx.baseUrl, "POST", "/v1/responses", body);
    },
  }),
);

const SCENARIOS: BenchScenario[] = [
  ...RESPONSES_PREFILL_SCENARIOS,
  ...RESPONSES_DECODE_SCENARIOS,
  {
    id: "responses_evals_mmlu_lite",
    title: "MMLU (Lite)",
    scope: "responses/evals/mmlu",
    apiGroup: "/v1/responses/evals/mmlu",
    method: "POST",
    pathTemplate: "/v1/responses",
    description: "MCQ broad-knowledge style checks, mirroring MMLU answer format.",
    pythonName: "responses/evals/mmlu",
    defaults: { requests: 1, concurrency: 1, rounds: 1, responsesMaxOutputTokens: 8 },
    prepare: async (ctx) => {
      ctx.state.evalCases = [
        {
          prompt:
            "Subject: high_school_mathematics\nQuestion: What is the derivative of x^2?\nA. x\nB. 2x\nC. x^3\nD. 2",
          expected: "B",
        },
        {
          prompt:
            "Subject: college_biology\nQuestion: Which organelle is primarily responsible for ATP production?\nA. Nucleus\nB. Golgi apparatus\nC. Mitochondrion\nD. Lysosome",
          expected: "C",
        },
        {
          prompt:
            "Subject: formal_logic\nQuestion: Which statement is logically equivalent to NOT (P AND Q)?\nA. (NOT P) OR (NOT Q)\nB. (NOT P) AND (NOT Q)\nC. P OR Q\nD. P AND (NOT Q)",
          expected: "A",
        },
      ];
    },
    request: (ctx, idx, cfg) => {
      const cases = (ctx.state.evalCases as Array<{ prompt: string; expected: string }>) ?? [];
      const current = cases[idx % Math.max(1, cases.length)] ?? {
        prompt: "Question: 1+1?\nA. 1\nB. 2\nC. 3\nD. 4",
      };
      return requestJson(
        ctx.baseUrl,
        "POST",
        "/v1/responses",
        buildResponsesRequestBody(
          cfg,
          `${current.prompt}\n\nRespond with ONLY the letter of the correct answer (A, B, C, or D).`,
          {
            maxOutputTokens: 8,
            temperature: 0.0,
            topP: 1.0,
          },
        ),
      );
    },
    classifySuccess: (_resp, bodyText, idx, ctx) => {
      const cases = (ctx.state.evalCases as Array<{ prompt: string; expected: string }>) ?? [];
      const expected = (cases[idx % Math.max(1, cases.length)] ?? { expected: "B" }).expected;
      const payload = parseResponsePayload(bodyText);
      if (!payload) return false;
      const output = extractResponseOutputText(payload);
      const predicted = extractChoiceLetter(output);
      return predicted === expected;
    },
  },
  {
    id: "responses_evals_gpqa_lite",
    title: "GPQA (Lite)",
    scope: "responses/evals/gpqa",
    apiGroup: "/v1/responses/evals/gpqa",
    method: "POST",
    pathTemplate: "/v1/responses",
    description: "Graduate-level science MCQ prompts with strict letter-only scoring.",
    pythonName: "responses/evals/gpqa",
    defaults: { requests: 1, concurrency: 1, rounds: 1, responsesMaxOutputTokens: 8 },
    prepare: async (ctx) => {
      ctx.state.evalCases = [
        {
          prompt:
            "Question: In thermodynamics, for a spontaneous process in an isolated system, which quantity must increase?\nA. Gibbs free energy\nB. Enthalpy\nC. Entropy\nD. Pressure",
          expected: "C",
        },
        {
          prompt:
            "Question: Which quantum number determines orbital angular momentum?\nA. principal quantum number n\nB. azimuthal quantum number l\nC. magnetic quantum number m\nD. spin quantum number s",
          expected: "B",
        },
        {
          prompt:
            "Question: In population genetics, the Hardy-Weinberg equilibrium assumes which condition?\nA. Strong selection\nB. Very small population\nC. Random mating\nD. Continuous migration",
          expected: "C",
        },
      ];
    },
    request: (ctx, idx, cfg) => {
      const cases = (ctx.state.evalCases as Array<{ prompt: string; expected: string }>) ?? [];
      const current = cases[idx % Math.max(1, cases.length)] ?? {
        prompt: "Question: Pick the best answer.\nA. A\nB. B\nC. C\nD. D",
      };
      return requestJson(
        ctx.baseUrl,
        "POST",
        "/v1/responses",
        buildResponsesRequestBody(
          cfg,
          `${current.prompt}\n\nRespond with ONLY one letter: A, B, C, or D.`,
          {
            maxOutputTokens: 8,
            temperature: 0.0,
            topP: 1.0,
          },
        ),
      );
    },
    classifySuccess: (_resp, bodyText, idx, ctx) => {
      const cases = (ctx.state.evalCases as Array<{ prompt: string; expected: string }>) ?? [];
      const expected = (cases[idx % Math.max(1, cases.length)] ?? { expected: "A" }).expected;
      const payload = parseResponsePayload(bodyText);
      if (!payload) return false;
      const output = extractResponseOutputText(payload);
      return extractChoiceLetter(output) === expected;
    },
  },
  {
    id: "responses_evals_ifeval_lite",
    title: "IFEval (Lite)",
    scope: "responses/evals/ifeval",
    apiGroup: "/v1/responses/evals/ifeval",
    method: "POST",
    pathTemplate: "/v1/responses",
    description: "Instruction-following checks with strict rule validation.",
    pythonName: "responses/evals/ifeval",
    defaults: { requests: 1, concurrency: 1, rounds: 1, responsesMaxOutputTokens: 16 },
    prepare: async (ctx) => {
      ctx.state.evalCases = [
        {
          prompt: "Use no more than 2 words and include both words: latency and budget.",
          rule: { mustInclude: ["latency", "budget"], maxWords: 2 },
        },
        {
          prompt: 'Return only JSON object with keys status and code: {"status":"ok","code":7}',
          rule: { jsonExpected: { status: "ok", code: 7 } },
        },
        {
          prompt: "End your reply with END-42 and do not use the word sorry.",
          rule: { endsWith: "END-42", forbidden: ["sorry"] },
        },
      ];
    },
    request: (ctx, idx, cfg) => {
      const cases =
        (ctx.state.evalCases as Array<{ prompt: string; rule: Record<string, unknown> }>) ?? [];
      const current = cases[idx % Math.max(1, cases.length)] ?? {
        prompt: "Reply with exactly: ready",
      };
      return requestJson(
        ctx.baseUrl,
        "POST",
        "/v1/responses",
        buildResponsesRequestBody(cfg, current.prompt, {
          maxOutputTokens: Math.max(16, cfg.responsesMaxOutputTokens),
          temperature: 0.0,
          topP: 1.0,
        }),
      );
    },
    classifySuccess: (_resp, bodyText, idx, ctx) => {
      const cases =
        (ctx.state.evalCases as Array<{
          prompt: string;
          rule: {
            mustInclude?: string[];
            forbidden?: string[];
            exactText?: string;
            jsonExpected?: Record<string, unknown>;
            maxWords?: number;
            endsWith?: string;
          };
        }>) ?? [];
      const current = cases[idx % Math.max(1, cases.length)] ?? {
        rule: { exactText: "ready" },
      };
      const payload = parseResponsePayload(bodyText);
      if (!payload) return false;
      const output = extractResponseOutputText(payload);
      return scoreIfevalLite(output, current.rule);
    },
  },
  {
    id: "responses_evals_bfcl_lite",
    title: "BFCL (Lite)",
    scope: "responses/evals/bfcl",
    apiGroup: "/v1/responses/evals/bfcl",
    method: "POST",
    pathTemplate: "/v1/responses",
    description: "Function-calling checks with simple/parallel/irrelevance cases.",
    pythonName: "responses/evals/bfcl",
    defaults: { requests: 1, concurrency: 1, rounds: 1, responsesMaxOutputTokens: 32 },
    prepare: async (ctx) => {
      ctx.state.evalCases = [
        {
          prompt: "Call get_weather with city='Paris' and unit='celsius'.",
          mode: "simple",
          tools: [
            {
              type: "function",
              name: "get_weather",
              description: "Get weather for a city",
              parameters: {
                type: "object",
                properties: {
                  city: { type: "string" },
                  unit: { type: "string", enum: ["celsius", "fahrenheit"] },
                },
                required: ["city", "unit"],
              },
            },
          ],
          expectedCalls: [{ name: "get_weather", arguments: { city: "Paris", unit: "celsius" } }],
        },
        {
          prompt: "Call add_task twice: first id=1,title='alpha'; second id=2,title='beta'.",
          mode: "parallel",
          tools: [
            {
              type: "function",
              name: "add_task",
              description: "Create a task item",
              parameters: {
                type: "object",
                properties: { id: { type: "integer" }, title: { type: "string" } },
                required: ["id", "title"],
              },
            },
          ],
          expectedCalls: [
            { name: "add_task", arguments: { id: 1, title: "alpha" } },
            { name: "add_task", arguments: { id: 2, title: "beta" } },
          ],
        },
        {
          prompt: "Say exactly hello world and do not call any tools.",
          mode: "irrelevance",
          tools: [
            {
              type: "function",
              name: "noop_tool",
              description: "No-op tool",
              parameters: {
                type: "object",
                properties: { value: { type: "string" } },
              },
            },
          ],
          expectedCalls: [],
        },
      ];
    },
    request: (ctx, idx, cfg) => {
      const cases =
        (ctx.state.evalCases as Array<{
          prompt: string;
          mode: "simple" | "parallel" | "irrelevance";
          tools: Array<Record<string, unknown>>;
          expectedCalls: Array<{ name: string; arguments: Record<string, unknown> }>;
        }>) ?? [];
      const current = cases[idx % Math.max(1, cases.length)] ?? {
        prompt: "Do nothing.",
        mode: "irrelevance" as const,
        tools: [],
        expectedCalls: [],
      };
      const body: Record<string, unknown> = {
        input: current.prompt,
        stream: false,
        max_output_tokens: Math.max(32, cfg.responsesMaxOutputTokens),
        temperature: 0.0,
        top_p: 1.0,
        tools: current.tools,
        tool_choice: "auto",
      };
      if (cfg.responsesModel.trim().length > 0) {
        body.model = cfg.responsesModel.trim();
      }
      return requestJson(ctx.baseUrl, "POST", "/v1/responses", body);
    },
    classifySuccess: (_resp, bodyText, idx, ctx) => {
      const cases =
        (ctx.state.evalCases as Array<{
          prompt: string;
          mode: "simple" | "parallel" | "irrelevance";
          tools: Array<Record<string, unknown>>;
          expectedCalls: Array<{ name: string; arguments: Record<string, unknown> }>;
        }>) ?? [];
      const current = cases[idx % Math.max(1, cases.length)] ?? {
        mode: "irrelevance" as const,
        expectedCalls: [],
      };
      const payload = parseResponsePayload(bodyText);
      if (!payload) return false;
      const calls = extractResponseToolCalls(payload);
      return scoreBfclCalls(calls, current.expectedCalls, current.mode);
    },
  },
  {
    id: "responses_evals_mmmu_lite",
    title: "MMMU (Lite)",
    scope: "responses/evals/mmmu",
    apiGroup: "/v1/responses/evals/mmmu",
    method: "POST",
    pathTemplate: "/v1/responses",
    description: "Multimodal MCQ flow with uploaded image files (browser-safe sample set).",
    pythonName: "responses/evals/mmmu",
    defaults: { requests: 1, concurrency: 1, rounds: 1, responsesMaxOutputTokens: 8 },
    prepare: async (ctx) => {
      const baseCases = [
        {
          prompt:
            "Image question: what color is the large square?\nA. Red\nB. Blue\nC. Green\nD. Yellow",
          expected: "A",
          svg: "<svg xmlns='http://www.w3.org/2000/svg' width='160' height='110'><rect width='160' height='110' fill='white'/><rect x='25' y='20' width='70' height='70' fill='red'/><circle cx='125' cy='55' r='25' fill='blue'/></svg>",
        },
        {
          prompt:
            "Image question: which shape is blue?\nA. Triangle\nB. Circle\nC. Square\nD. Rectangle",
          expected: "B",
          svg: "<svg xmlns='http://www.w3.org/2000/svg' width='160' height='110'><rect width='160' height='110' fill='white'/><polygon points='20,90 55,25 90,90' fill='green'/><circle cx='125' cy='55' r='25' fill='blue'/></svg>",
        },
      ];

      const uploaded = await Promise.all(
        baseCases.map(async (item, idx) => ({
          ...item,
          fileId: await uploadSvgImageFile(ctx.baseUrl, `bench_mmmu_${idx + 1}.svg`, item.svg),
        })),
      );
      ctx.state.evalCases = uploaded;
    },
    request: (ctx, idx, cfg) => {
      const cases =
        (ctx.state.evalCases as Array<{ prompt: string; expected: string; fileId: string | null }>) ??
        [];
      const current = cases[idx % Math.max(1, cases.length)] ?? {
        prompt: "A. A\nB. B\nC. C\nD. D",
        fileId: null,
      };

      const content: Array<Record<string, unknown>> = [
        { type: "input_text", text: `${current.prompt}\n\nRespond with only A, B, C, or D.` },
      ];
      if (current.fileId) {
        content.push({
          type: "input_image",
          image_url: `${ctx.baseUrl}/v1/files/${encodeURIComponent(current.fileId)}/content`,
        });
      }

      const body: Record<string, unknown> = {
        input: [{ type: "message", role: "user", content }],
        stream: false,
        max_output_tokens: 8,
        temperature: 0.0,
        top_p: 1.0,
      };
      if (cfg.responsesModel.trim().length > 0) {
        body.model = cfg.responsesModel.trim();
      }
      return requestJson(ctx.baseUrl, "POST", "/v1/responses", body);
    },
    classifySuccess: (_resp, bodyText, idx, ctx) => {
      const cases =
        (ctx.state.evalCases as Array<{ prompt: string; expected: string; fileId: string | null }>) ??
        [];
      const expected = (cases[idx % Math.max(1, cases.length)] ?? { expected: "A" }).expected;
      const payload = parseResponsePayload(bodyText);
      if (!payload) return false;
      const output = extractResponseOutputText(payload);
      return extractChoiceLetter(output) === expected;
    },
  },
  {
    id: "sql_query",
    title: "SQL Query",
    scope: "db/sql",
    apiGroup: "/v1/db/sql",
    method: "POST",
    pathTemplate: "/v1/db/sql/query",
    description: "POST /v1/db/sql/query",
    pythonName: "db/perf/sql_select1",
    defaults: { requests: 1, concurrency: 1, rounds: 1 },
    request: (ctx) =>
      requestJson(ctx.baseUrl, "POST", "/v1/db/sql/query", { query: "SELECT 1 AS v" }),
  },
  {
    id: "sql_explain",
    title: "SQL Explain",
    scope: "db/sql",
    apiGroup: "/v1/db/sql",
    method: "POST",
    pathTemplate: "/v1/db/sql/explain",
    description: "POST /v1/db/sql/explain",
    request: (ctx) =>
      requestJson(ctx.baseUrl, "POST", "/v1/db/sql/explain", { query: "SELECT 1 AS v" }),
  },
  {
    id: "kv_put_entry",
    title: "KV Put Entry",
    scope: "db/kv",
    apiGroup: "/v1/db/kv",
    method: "PUT",
    pathTemplate: "/v1/db/kv/namespaces/{namespace}/entries/{key}",
    description: "PUT /v1/db/kv/namespaces/{ns}/entries/{key}",
    prepare: async (ctx) => {
      ctx.state.ns = uniqueName("bench_kv_put");
    },
    request: (ctx, idx) => {
      const ns = String(ctx.state.ns);
      const key = `k_${ctx.round}_${idx}`;
      const body = TEXT_ENCODER.encode(`value_${idx}`);
      return request(
        ctx.baseUrl,
        "PUT",
        `/v1/db/kv/namespaces/${encodeURIComponent(ns)}/entries/${encodeURIComponent(key)}`,
        {
          body,
          headers: { "Content-Type": "application/octet-stream" },
        },
      );
    },
  },
  {
    id: "kv_get_entry",
    title: "KV Get Entry",
    scope: "db/kv",
    apiGroup: "/v1/db/kv",
    method: "GET",
    pathTemplate: "/v1/db/kv/namespaces/{namespace}/entries/{key}",
    description: "GET /v1/db/kv/namespaces/{ns}/entries/{key}",
    prepare: async (ctx) => {
      const ns = uniqueName("bench_kv_get");
      const key = "seed";
      const putResp = await request(
        ctx.baseUrl,
        "PUT",
        `/v1/db/kv/namespaces/${encodeURIComponent(ns)}/entries/${key}`,
        {
          body: TEXT_ENCODER.encode("seed-value"),
          headers: { "Content-Type": "application/octet-stream" },
        },
      );
      if (!putResp.ok) {
        const txt = await putResp.text();
        throw new Error(`kv get setup failed: ${putResp.status} ${txt}`);
      }
      ctx.state.ns = ns;
    },
    request: (ctx) =>
      request(
        ctx.baseUrl,
        "GET",
        `/v1/db/kv/namespaces/${encodeURIComponent(String(ctx.state.ns))}/entries/seed`,
      ),
  },
  {
    id: "kv_batch",
    title: "KV Batch",
    scope: "db/kv",
    apiGroup: "/v1/db/kv",
    method: "POST",
    pathTemplate: "/v1/db/kv/namespaces/{namespace}/batch",
    description: "POST /v1/db/kv/namespaces/{ns}/batch",
    pythonName: "db/perf/kv_batch",
    defaults: { requests: 1, concurrency: 1, rounds: 1, kvBatchSize: 1 },
    prepare: async (ctx) => {
      ctx.state.ns = uniqueName("bench_kv_batch");
    },
    request: (ctx, idx, cfg) => {
      const entries = Array.from({ length: cfg.kvBatchSize }, (_, j) => ({
        key: `k_${idx}_${j}`,
        value_base64: toBase64(`v_${idx}_${j}`),
        durability: "batched",
      }));
      return requestJson(
        ctx.baseUrl,
        "POST",
        `/v1/db/kv/namespaces/${encodeURIComponent(String(ctx.state.ns))}/batch`,
        { entries },
      );
    },
  },
  {
    id: "kv_list",
    title: "KV List Entries",
    scope: "db/kv",
    apiGroup: "/v1/db/kv",
    method: "GET",
    pathTemplate: "/v1/db/kv/namespaces/{namespace}/entries",
    description: "GET /v1/db/kv/namespaces/{ns}/entries",
    prepare: async (ctx) => {
      const ns = uniqueName("bench_kv_list");
      await seedKvNamespace(ctx.baseUrl, ns, 256);
      ctx.state.ns = ns;
    },
    request: (ctx) =>
      request(
        ctx.baseUrl,
        "GET",
        `/v1/db/kv/namespaces/${encodeURIComponent(String(ctx.state.ns))}/entries`,
      ),
  },
  {
    id: "kv_stats",
    title: "KV Stats",
    scope: "db/kv",
    apiGroup: "/v1/db/kv",
    method: "GET",
    pathTemplate: "/v1/db/kv/namespaces/{namespace}/stats",
    description: "GET /v1/db/kv/namespaces/{ns}/stats",
    prepare: async (ctx) => {
      const ns = uniqueName("bench_kv_stats");
      await seedKvNamespace(ctx.baseUrl, ns, 64);
      ctx.state.ns = ns;
    },
    request: (ctx) =>
      request(
        ctx.baseUrl,
        "GET",
        `/v1/db/kv/namespaces/${encodeURIComponent(String(ctx.state.ns))}/stats`,
      ),
  },
  {
    id: "table_rows_write",
    title: "Table Rows Write",
    scope: "db/tables",
    apiGroup: "/v1/db/tables",
    method: "POST",
    pathTemplate: "/v1/db/tables/{ns}/rows",
    description: "POST /v1/db/tables/{ns}/rows",
    pythonName: "db/perf/rows_write",
    defaults: { requests: 1, concurrency: 1, rounds: 1, tableSeedRows: 1 },
    prepare: async (ctx) => {
      ctx.state.ns = uniqueName("bench_rows_write");
    },
    request: (ctx, idx) =>
      requestJson(
        ctx.baseUrl,
        "POST",
        `/v1/db/tables/${encodeURIComponent(String(ctx.state.ns))}/rows`,
        {
          schema_id: 10,
          columns: [
            { column_id: 1, type: "scalar_u64", value: idx + 1 },
            { column_id: 2, type: "scalar_i64", value: 1_700_000_000_000 + idx },
            { column_id: 20, type: "string", value: `payload-${idx}` },
          ],
        },
      ),
  },
  {
    id: "table_rows_scan_get",
    title: "Table Rows Scan (GET)",
    scope: "db/tables",
    apiGroup: "/v1/db/tables",
    method: "GET",
    pathTemplate: "/v1/db/tables/{ns}/rows",
    description: "GET /v1/db/tables/{ns}/rows?schema_id=...",
    pythonName: "db/perf/rows_scan",
    defaults: { requests: 1, concurrency: 1, rounds: 1, tableSeedRows: 1 },
    prepare: async (ctx, cfg) => {
      const ns = uniqueName("bench_rows_scan");
      await seedTableRows(ctx.baseUrl, ns, cfg.tableSeedRows);
      ctx.state.ns = ns;
    },
    request: (ctx) =>
      request(
        ctx.baseUrl,
        "GET",
        `/v1/db/tables/${encodeURIComponent(String(ctx.state.ns))}/rows?schema_id=10&limit=100`,
      ),
  },
  {
    id: "docs_create",
    title: "Docs Create",
    scope: "db/tables",
    apiGroup: "/v1/db/tables",
    method: "POST",
    pathTemplate: "/v1/db/tables/{table}",
    description: "POST /v1/db/tables/{table}",
    prepare: async (ctx) => {
      ctx.state.table = uniqueName("bench_docs");
    },
    request: (ctx, idx) =>
      requestJson(
        ctx.baseUrl,
        "POST",
        `/v1/db/tables/${encodeURIComponent(String(ctx.state.table))}`,
        {
          type: "prompt",
          title: `Doc ${idx}`,
          content: { text: `hello ${idx}` },
        },
      ),
  },
  {
    id: "docs_search",
    title: "Docs Search",
    scope: "db/tables",
    apiGroup: "/v1/db/tables",
    method: "POST",
    pathTemplate: "/v1/db/tables/{table}/search",
    description: "POST /v1/db/tables/{table}/search",
    prepare: async (ctx) => {
      const table = uniqueName("bench_docs_search");
      for (let i = 0; i < 48; i++) {
        const resp = await requestJson(
          ctx.baseUrl,
          "POST",
          `/v1/db/tables/${encodeURIComponent(table)}`,
          { type: "prompt", title: `doc-${i}`, content: { text: `alpha token ${i}` } },
        );
        if (!resp.ok) {
          const txt = await resp.text();
          throw new Error(`docs search setup failed: ${resp.status} ${txt}`);
        }
      }
      ctx.state.table = table;
    },
    request: (ctx) =>
      requestJson(
        ctx.baseUrl,
        "POST",
        `/v1/db/tables/${encodeURIComponent(String(ctx.state.table))}/search`,
        { query: "alpha", limit: 10 },
      ),
  },
  {
    id: "vector_create_collection",
    title: "Vector Create Collection",
    scope: "db/vectors",
    apiGroup: "/v1/db/vectors",
    method: "POST",
    pathTemplate: "/v1/db/vectors/collections",
    description: "POST /v1/db/vectors/collections",
    pythonName: "db/perf/vector_create_collection",
    defaults: { requests: 1, concurrency: 1, rounds: 1, vectorDims: 8 },
    request: (ctx, idx, cfg) =>
      requestJson(ctx.baseUrl, "POST", "/v1/db/vectors/collections", {
        name: uniqueName(`bench_vec_create_r${ctx.round}_${idx}`),
        dims: cfg.vectorDims,
      }),
  },
  {
    id: "vector_append",
    title: "Vector Append",
    scope: "db/vectors",
    apiGroup: "/v1/db/vectors",
    method: "POST",
    pathTemplate: "/v1/db/vectors/collections/{name}/points/append",
    description: "POST /v1/db/vectors/collections/{name}/points/append",
    pythonName: "db/perf/vector_append_points",
    defaults: { requests: 1, concurrency: 1, rounds: 1, vectorDims: 8 },
    prepare: async (ctx, cfg) => {
      const collection = uniqueName("bench_vec_append");
      await createVectorCollection(ctx.baseUrl, collection, cfg.vectorDims);
      ctx.state.collection = collection;
      ctx.state.nextId = 1;
    },
    request: async (ctx, idx, cfg) => {
      const collection = String(ctx.state.collection);
      const id = Number(ctx.state.nextId ?? 1) + idx;
      return requestJson(
        ctx.baseUrl,
        "POST",
        `/v1/db/vectors/collections/${encodeURIComponent(collection)}/points/append`,
        {
          vectors: [
            {
              id,
              values: Array.from(
                { length: cfg.vectorDims },
                (_, j) => (j === id % cfg.vectorDims ? 1.0 : 0.0),
              ),
            },
          ],
        },
      );
    },
  },
  {
    id: "vector_query",
    title: "Vector Query",
    scope: "db/vectors",
    apiGroup: "/v1/db/vectors",
    method: "POST",
    pathTemplate: "/v1/db/vectors/collections/{name}/points/query",
    description: "POST /v1/db/vectors/collections/{name}/points/query",
    pythonName: "db/perf/vector_query_points",
    defaults: { requests: 1, concurrency: 1, rounds: 1, vectorDims: 8 },
    prepare: async (ctx, cfg) => {
      const collection = uniqueName("bench_vec_query");
      await createVectorCollection(ctx.baseUrl, collection, cfg.vectorDims);
      await appendVectorPoints(ctx.baseUrl, collection, cfg.vectorDims, 256);
      ctx.state.collection = collection;
    },
    request: (ctx, idx, cfg) =>
      requestJson(
        ctx.baseUrl,
        "POST",
        `/v1/db/vectors/collections/${encodeURIComponent(String(ctx.state.collection))}/points/query`,
        {
          vector: Array.from(
            { length: cfg.vectorDims },
            (_, j) => (j === idx % cfg.vectorDims ? 1.0 : 0.0),
          ),
          top_k: 10,
        },
      ),
  },
  {
    id: "vector_fetch",
    title: "Vector Fetch",
    scope: "db/vectors",
    apiGroup: "/v1/db/vectors",
    method: "POST",
    pathTemplate: "/v1/db/vectors/collections/{name}/points/fetch",
    description: "POST /v1/db/vectors/collections/{name}/points/fetch",
    pythonName: "db/perf/vector_fetch_points",
    defaults: { requests: 1, concurrency: 1, rounds: 1, vectorDims: 8 },
    prepare: async (ctx, cfg) => {
      const collection = uniqueName("bench_vec_fetch");
      await createVectorCollection(ctx.baseUrl, collection, cfg.vectorDims);
      await appendVectorPoints(ctx.baseUrl, collection, cfg.vectorDims, 1, 42);
      ctx.state.collection = collection;
    },
    request: (ctx) =>
      requestJson(
        ctx.baseUrl,
        "POST",
        `/v1/db/vectors/collections/${encodeURIComponent(String(ctx.state.collection))}/points/fetch`,
        { ids: [42], include_values: false },
      ),
  },
  {
    id: "vector_upsert",
    title: "Vector Upsert",
    scope: "db/vectors",
    apiGroup: "/v1/db/vectors",
    method: "POST",
    pathTemplate: "/v1/db/vectors/collections/{name}/points/upsert",
    description: "POST /v1/db/vectors/collections/{name}/points/upsert",
    pythonName: "db/perf/vector_upsert_points",
    defaults: { requests: 1, concurrency: 1, rounds: 1, vectorDims: 8 },
    prepare: async (ctx, cfg) => {
      const collection = uniqueName("bench_vec_upsert");
      await createVectorCollection(ctx.baseUrl, collection, cfg.vectorDims);
      await appendVectorPoints(ctx.baseUrl, collection, cfg.vectorDims, 1, 1);
      ctx.state.collection = collection;
    },
    request: (ctx, idx, cfg) =>
      requestJson(
        ctx.baseUrl,
        "POST",
        `/v1/db/vectors/collections/${encodeURIComponent(String(ctx.state.collection))}/points/upsert`,
        {
          vectors: [
            {
              id: 1,
              values: Array.from(
                { length: cfg.vectorDims },
                (_, j) => (j === idx % cfg.vectorDims ? 1.0 : 0.0),
              ),
            },
          ],
        },
      ),
  },
  {
    id: "vector_delete",
    title: "Vector Delete",
    scope: "db/vectors",
    apiGroup: "/v1/db/vectors",
    method: "POST",
    pathTemplate: "/v1/db/vectors/collections/{name}/points/delete",
    description: "POST /v1/db/vectors/collections/{name}/points/delete",
    pythonName: "db/perf/vector_delete_points",
    defaults: { requests: 1, concurrency: 1, rounds: 1, vectorDims: 8 },
    prepare: async (ctx, cfg) => {
      const collection = uniqueName("bench_vec_delete");
      await createVectorCollection(ctx.baseUrl, collection, cfg.vectorDims);
      await appendVectorPoints(ctx.baseUrl, collection, cfg.vectorDims, Math.max(1, cfg.requests), 1);
      ctx.state.collection = collection;
    },
    request: (ctx, idx) =>
      requestJson(
        ctx.baseUrl,
        "POST",
        `/v1/db/vectors/collections/${encodeURIComponent(String(ctx.state.collection))}/points/delete`,
        { ids: [idx + 1] },
      ),
  },
  {
    id: "vector_stats",
    title: "Vector Stats",
    scope: "db/vectors",
    apiGroup: "/v1/db/vectors",
    method: "GET",
    pathTemplate: "/v1/db/vectors/collections/{name}/stats",
    description: "GET /v1/db/vectors/collections/{name}/stats",
    pythonName: "db/perf/vector_stats",
    defaults: { requests: 1, concurrency: 1, rounds: 1, vectorDims: 8 },
    prepare: async (ctx, cfg) => {
      const collection = uniqueName("bench_vec_stats");
      await createVectorCollection(ctx.baseUrl, collection, cfg.vectorDims);
      await appendVectorPoints(ctx.baseUrl, collection, cfg.vectorDims, 64);
      ctx.state.collection = collection;
    },
    request: (ctx) =>
      request(
        ctx.baseUrl,
        "GET",
        `/v1/db/vectors/collections/${encodeURIComponent(String(ctx.state.collection))}/stats`,
      ),
  },
  {
    id: "vector_changes",
    title: "Vector Changes",
    scope: "db/vectors",
    apiGroup: "/v1/db/vectors",
    method: "GET",
    pathTemplate: "/v1/db/vectors/collections/{name}/changes",
    description: "GET /v1/db/vectors/collections/{name}/changes",
    pythonName: "db/perf/vector_changes",
    defaults: { requests: 1, concurrency: 1, rounds: 1, vectorDims: 8 },
    prepare: async (ctx, cfg) => {
      const collection = uniqueName("bench_vec_changes");
      await createVectorCollection(ctx.baseUrl, collection, cfg.vectorDims);
      await appendVectorPoints(ctx.baseUrl, collection, cfg.vectorDims, 2);
      ctx.state.collection = collection;
    },
    request: (ctx) =>
      request(
        ctx.baseUrl,
        "GET",
        `/v1/db/vectors/collections/${encodeURIComponent(String(ctx.state.collection))}/changes?since=0&limit=32`,
      ),
  },
  {
    id: "vector_compact",
    title: "Vector Compact",
    scope: "db/vectors",
    apiGroup: "/v1/db/vectors",
    method: "POST",
    pathTemplate: "/v1/db/vectors/collections/{name}/compact",
    description: "POST /v1/db/vectors/collections/{name}/compact",
    pythonName: "db/perf/vector_compact",
    defaults: { requests: 1, concurrency: 1, rounds: 1, vectorDims: 8 },
    prepare: async (ctx, cfg) => {
      const collection = uniqueName("bench_vec_compact");
      await createVectorCollection(ctx.baseUrl, collection, cfg.vectorDims);
      await appendVectorPoints(ctx.baseUrl, collection, cfg.vectorDims, 2);
      const delResp = await requestJson(
        ctx.baseUrl,
        "POST",
        `/v1/db/vectors/collections/${encodeURIComponent(collection)}/points/delete`,
        { ids: [2] },
      );
      if (!delResp.ok) {
        const txt = await delResp.text();
        throw new Error(`vector compact setup failed: ${delResp.status} ${txt}`);
      }
      ctx.state.collection = collection;
    },
    request: (ctx) =>
      requestJson(
        ctx.baseUrl,
        "POST",
        `/v1/db/vectors/collections/${encodeURIComponent(String(ctx.state.collection))}/compact`,
        {},
      ),
  },
  {
    id: "vector_indexes_build",
    title: "Vector Indexes Build",
    scope: "db/vectors",
    apiGroup: "/v1/db/vectors",
    method: "POST",
    pathTemplate: "/v1/db/vectors/collections/{name}/indexes/build",
    description: "POST /v1/db/vectors/collections/{name}/indexes/build",
    pythonName: "db/perf/vector_indexes_build",
    defaults: { requests: 1, concurrency: 1, rounds: 1, vectorDims: 8 },
    prepare: async (ctx, cfg) => {
      const collection = uniqueName("bench_vec_indexes");
      await createVectorCollection(ctx.baseUrl, collection, cfg.vectorDims);
      await appendVectorPoints(ctx.baseUrl, collection, cfg.vectorDims, 8);
      ctx.state.collection = collection;
    },
    request: async (ctx) => {
      const collection = encodeURIComponent(String(ctx.state.collection));
      const statsResp = await request(
        ctx.baseUrl,
        "GET",
        `/v1/db/vectors/collections/${collection}/stats`,
      );
      if (!statsResp.ok) {
        const txt = await statsResp.text();
        throw new Error(`vector indexes setup failed: ${statsResp.status} ${txt}`);
      }
      const stats = (await statsResp.json()) as { manifest_generation?: number };
      const expectedGeneration =
        typeof stats.manifest_generation === "number" && Number.isFinite(stats.manifest_generation)
          ? stats.manifest_generation
          : 0;
      return requestJson(
        ctx.baseUrl,
        "POST",
        `/v1/db/vectors/collections/${collection}/indexes/build`,
        { max_segments: 8, expected_generation: expectedGeneration },
      );
    },
  },
  {
    id: "blob_list",
    title: "Blob List",
    scope: "db/blobs",
    apiGroup: "/v1/db/blobs",
    method: "GET",
    pathTemplate: "/v1/db/blobs",
    description: "GET /v1/db/blobs",
    request: (ctx) => request(ctx.baseUrl, "GET", "/v1/db/blobs?limit=100"),
  },
  {
    id: "ops_compact",
    title: "Ops Compact",
    scope: "db/ops",
    apiGroup: "/v1/db/ops",
    method: "POST",
    pathTemplate: "/v1/db/ops/compact",
    description: "POST /v1/db/ops/compact",
    prepare: async (ctx, cfg) => {
      const collection = uniqueName("bench_ops");
      await createVectorCollection(ctx.baseUrl, collection, cfg.vectorDims);
      await appendVectorPoints(ctx.baseUrl, collection, cfg.vectorDims, 64);
      const delResp = await requestJson(
        ctx.baseUrl,
        "POST",
        `/v1/db/vectors/collections/${encodeURIComponent(collection)}/points/delete`,
        { ids: [1, 2, 3, 4] },
      );
      if (!delResp.ok) {
        const txt = await delResp.text();
        throw new Error(`ops compact setup failed: ${delResp.status} ${txt}`);
      }
      ctx.state.collection = collection;
    },
    request: (ctx, _idx, cfg) =>
      requestJson(ctx.baseUrl, "POST", "/v1/db/ops/compact", {
        collection: String(ctx.state.collection),
        dims: cfg.vectorDims,
      }),
  },
];

const SCOPE_META: Record<BenchScopeId, ScopeMeta> = {
  "responses/perf": {
    id: "responses/perf",
    title: "Responses Perf",
    description: "Throughput and latency for prefill/decode workloads.",
    major: "responses",
    apiGroup: "/v1/responses/perf",
  },
  "responses/evals/mmlu": {
    id: "responses/evals/mmlu",
    title: "Eval: MMLU",
    description: "Broad-knowledge MCQ evaluation (57-subject style).",
    major: "responses",
    apiGroup: "/v1/responses/evals/mmlu",
  },
  "responses/evals/gpqa": {
    id: "responses/evals/gpqa",
    title: "Eval: GPQA",
    description: "Graduate-level science MCQ reasoning evaluation.",
    major: "responses",
    apiGroup: "/v1/responses/evals/gpqa",
  },
  "responses/evals/ifeval": {
    id: "responses/evals/ifeval",
    title: "Eval: IFEval",
    description: "Instruction-following strict rule compliance evaluation.",
    major: "responses",
    apiGroup: "/v1/responses/evals/ifeval",
  },
  "responses/evals/bfcl": {
    id: "responses/evals/bfcl",
    title: "Eval: BFCL",
    description: "Function-calling accuracy evaluation with call matching.",
    major: "responses",
    apiGroup: "/v1/responses/evals/bfcl",
  },
  "responses/evals/mmmu": {
    id: "responses/evals/mmmu",
    title: "Eval: MMMU",
    description: "Multimodal MCQ evaluation with image understanding.",
    major: "responses",
    apiGroup: "/v1/responses/evals/mmmu",
  },
  "db/sql": {
    id: "db/sql",
    title: "DB SQL",
    description: "SQL query and explain endpoint performance.",
    major: "db",
    apiGroup: "/v1/db/sql",
  },
  "db/kv": {
    id: "db/kv",
    title: "DB KV",
    description: "Namespace key-value read/write and batch ops.",
    major: "db",
    apiGroup: "/v1/db/kv",
  },
  "db/tables": {
    id: "db/tables",
    title: "DB Tables",
    description: "Rows and docs create/search workloads.",
    major: "db",
    apiGroup: "/v1/db/tables",
  },
  "db/vectors": {
    id: "db/vectors",
    title: "DB Vectors",
    description: "Collection append/query/fetch/stats.",
    major: "db",
    apiGroup: "/v1/db/vectors",
  },
  "db/blobs": {
    id: "db/blobs",
    title: "DB Blobs",
    description: "Blob listing and access workloads.",
    major: "db",
    apiGroup: "/v1/db/blobs",
  },
  "db/ops": {
    id: "db/ops",
    title: "DB Ops",
    description: "Operational tasks like compact.",
    major: "db",
    apiGroup: "/v1/db/ops",
  },
};

const PAGE_DEFS: Record<string, PageDef> = {
  "": {
    slug: "",
    title: "Bench Suite",
    subtitle: "Benchmark Hub",
    description: "Pick a category page and run only the scenarios that match your team scope.",
    parent: null,
    children: ["responses", "db", "results"],
  },
  responses: {
    slug: "responses",
    title: "Responses Bench",
    subtitle: "/v1/responses/*",
    description: "Performance and eval benches for responses endpoints.",
    parent: "",
    children: ["responses/perf", "responses/evals"],
    scope: { mode: "prefix", value: "responses/" },
  },
  "responses/perf": {
    slug: "responses/perf",
    title: "Responses Perf",
    subtitle: "/v1/responses/perf",
    description: "Prompt prefill and decode performance workloads.",
    parent: "responses",
    children: [],
    scope: { mode: "exact", value: "responses/perf" },
  },
  "responses/evals": {
    slug: "responses/evals",
    title: "Responses Evals",
    subtitle: "/v1/responses/evals",
    description: "Evaluation suites aligned with the Python bench harness.",
    parent: "responses",
    children: [
      "responses/evals/mmlu",
      "responses/evals/gpqa",
      "responses/evals/ifeval",
      "responses/evals/bfcl",
      "responses/evals/mmmu",
    ],
    scope: { mode: "prefix", value: "responses/evals/" },
  },
  "responses/evals/mmlu": {
    slug: "responses/evals/mmlu",
    title: "Eval Suite: MMLU",
    subtitle: "responses/evals/mmlu",
    description: "Broad-knowledge multiple-choice accuracy suite.",
    parent: "responses/evals",
    children: [],
    scope: { mode: "exact", value: "responses/evals/mmlu" },
  },
  "responses/evals/gpqa": {
    slug: "responses/evals/gpqa",
    title: "Eval Suite: GPQA",
    subtitle: "responses/evals/gpqa",
    description: "Graduate-level science reasoning multiple-choice suite.",
    parent: "responses/evals",
    children: [],
    scope: { mode: "exact", value: "responses/evals/gpqa" },
  },
  "responses/evals/ifeval": {
    slug: "responses/evals/ifeval",
    title: "Eval Suite: IFEval",
    subtitle: "responses/evals/ifeval",
    description: "Instruction-following strict compliance suite.",
    parent: "responses/evals",
    children: [],
    scope: { mode: "exact", value: "responses/evals/ifeval" },
  },
  "responses/evals/bfcl": {
    slug: "responses/evals/bfcl",
    title: "Eval Suite: BFCL",
    subtitle: "responses/evals/bfcl",
    description: "Function-calling correctness suite.",
    parent: "responses/evals",
    children: [],
    scope: { mode: "exact", value: "responses/evals/bfcl" },
  },
  "responses/evals/mmmu": {
    slug: "responses/evals/mmmu",
    title: "Eval Suite: MMMU",
    subtitle: "responses/evals/mmmu",
    description: "Multimodal reasoning accuracy suite.",
    parent: "responses/evals",
    children: [],
    scope: { mode: "exact", value: "responses/evals/mmmu" },
  },
  db: {
    slug: "db",
    title: "DB Bench",
    subtitle: "/v1/db/*",
    description: "Database endpoint benches by API area.",
    parent: "",
    children: ["db/sql", "db/kv", "db/tables", "db/vectors", "db/blobs", "db/ops"],
    scope: { mode: "prefix", value: "db/" },
  },
  "db/sql": {
    slug: "db/sql",
    title: "DB SQL",
    subtitle: "/v1/db/sql",
    description: "SQL query and explain benches.",
    parent: "db",
    children: [],
    scope: { mode: "exact", value: "db/sql" },
  },
  "db/kv": {
    slug: "db/kv",
    title: "DB KV",
    subtitle: "/v1/db/kv",
    description: "KV namespace, entry, and batch benches.",
    parent: "db",
    children: [],
    scope: { mode: "exact", value: "db/kv" },
  },
  "db/tables": {
    slug: "db/tables",
    title: "DB Tables",
    subtitle: "/v1/db/tables",
    description: "Rows and docs benches.",
    parent: "db",
    children: [],
    scope: { mode: "exact", value: "db/tables" },
  },
  "db/vectors": {
    slug: "db/vectors",
    title: "DB Vectors",
    subtitle: "/v1/db/vectors",
    description: "Vector collection and point benches.",
    parent: "db",
    children: [],
    scope: { mode: "exact", value: "db/vectors" },
  },
  "db/blobs": {
    slug: "db/blobs",
    title: "DB Blobs",
    subtitle: "/v1/db/blobs",
    description: "Blob API benches.",
    parent: "db",
    children: [],
    scope: { mode: "exact", value: "db/blobs" },
  },
  "db/ops": {
    slug: "db/ops",
    title: "DB Ops",
    subtitle: "/v1/db/ops",
    description: "Operational endpoint benches.",
    parent: "db",
    children: [],
    scope: { mode: "exact", value: "db/ops" },
  },
  results: {
    slug: "results",
    title: "Results",
    subtitle: "bench/results",
    description: "Browse and compare historical benchmark runs.",
    parent: "",
    children: [],
  },
};

const SLUG_ALIASES: Record<string, string> = {
  "db/vector": "db/vectors",
  "db/vec": "db/vectors",
  "responses/eval": "responses/evals",
  "responses/evals/math": "responses/evals/mmlu",
  "responses/evals/instruction": "responses/evals/ifeval",
  "responses/evals/format": "responses/evals/bfcl",
  "responses/evals/grounding": "responses/evals/gpqa",
};

function toBenchPath(slug: string): string {
  return slug.length ? `/bench/#/${slug}/` : "/bench/";
}

function currentBenchPathForResolve(): string {
  const rawHash = window.location.hash.replace(/^#/, "").trim();
  if (!rawHash || rawHash === "/") {
    return window.location.pathname;
  }
  const hashPath = rawHash.startsWith("/") ? rawHash : `/${rawHash}`;
  if (hashPath === "/bench" || hashPath.startsWith("/bench/")) {
    return hashPath;
  }
  return `/bench${hashPath}`;
}

function normalizeBenchSlug(pathname: string): string {
  const noHash = pathname.split("#")[0] ?? pathname;
  const noQuery = noHash.split("?")[0] ?? noHash;
  const normalized = noQuery.replace(/\/+/g, "/").replace(/\/$/, "");
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length === 0) {
    return "";
  }
  if (parts[0] !== "bench") {
    return "__outside__";
  }
  return parts.slice(1).join("/");
}

export function resolveBenchPage(pathname: string): ResolvedBenchPage {
  const requested = normalizeBenchSlug(pathname);
  if (requested === "__outside__") {
    return { status: "not_found", requestedSlug: requested };
  }
  const canonical = SLUG_ALIASES[requested] ?? requested;
  const page = PAGE_DEFS[canonical];
  if (!page) {
    return { status: "not_found", requestedSlug: requested };
  }
  return { status: "ok", requestedSlug: requested, page };
}

function scenariosForPage(page: PageDef): BenchScenario[] {
  if (!page.scope) {
    return [];
  }
  if (page.scope.mode === "exact") {
    return SCENARIOS.filter((scenario) => scenario.scope === page.scope!.value);
  }
  return SCENARIOS.filter((scenario) => scenario.scope.startsWith(page.scope!.value));
}

function scenarioCountForPage(page: PageDef): number {
  return scenariosForPage(page).length;
}

function renderTopNav(current: PageDef): string {
  const items = ["responses", "db", "results"];
  const labels: Record<string, string> = {
    responses: "/responses",
    db: "/db",
    results: "/results",
  };
  return `
    <nav class="bench-topnav" aria-label="Bench sections">
      ${items
        .map((slug) => {
          const page = PAGE_DEFS[slug];
          const active = current.slug === slug || current.slug.startsWith(`${slug}/`);
          return `<a class="bench-topnav-link${active ? " is-active" : ""}" href="${toBenchPath(slug)}">${labels[slug] ?? page?.title ?? slug}</a>`;
        })
        .join("")}
    </nav>
  `;
}

function sectionRootForPage(page: PageDef): string | null {
  if (page.slug.startsWith("responses")) return "responses";
  if (page.slug.startsWith("db")) return "db";
  if (page.slug.startsWith("results")) return "results";
  return null;
}

type BenchTreeOpenState = Record<string, boolean>;

function readBenchTreeOpenState(): BenchTreeOpenState {
  try {
    const raw = localStorage.getItem(BENCH_TREE_OPEN_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    const out: BenchTreeOpenState = {};
    for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof value === "boolean") {
        out[key] = value;
      }
    }
    return out;
  } catch (_err) {
    return {};
  }
}

function writeBenchTreeOpenState(state: BenchTreeOpenState): void {
  try {
    localStorage.setItem(BENCH_TREE_OPEN_KEY, JSON.stringify(state));
  } catch (_err) {
    // ignore storage failures
  }
}

function setBenchTreeNodeOpen(slug: string, open: boolean): void {
  const state = readBenchTreeOpenState();
  state[slug] = open;
  writeBenchTreeOpenState(state);
}

type BenchPendingScenario = {
  page: string;
  scenarioId: string;
  createdAt: number;
};

function writePendingScenarioSelection(page: string, scenarioId: string): void {
  try {
    const payload: BenchPendingScenario = { page, scenarioId, createdAt: Date.now() };
    localStorage.setItem(BENCH_PENDING_SCENARIO_KEY, JSON.stringify(payload));
  } catch (_err) {
    // ignore storage failures
  }
}

function readPendingScenarioSelection(): BenchPendingScenario | null {
  try {
    const raw = localStorage.getItem(BENCH_PENDING_SCENARIO_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null;
    }
    const obj = parsed as Record<string, unknown>;
    const page = typeof obj.page === "string" ? obj.page : "";
    const scenarioId = typeof obj.scenarioId === "string" ? obj.scenarioId : "";
    const createdAt = toFiniteNumber(obj.createdAt);
    if (!page || !scenarioId || !Number.isFinite(createdAt) || createdAt <= 0) {
      return null;
    }
    if (Date.now() - createdAt > 1000 * 60 * 15) {
      localStorage.removeItem(BENCH_PENDING_SCENARIO_KEY);
      return null;
    }
    return { page, scenarioId, createdAt };
  } catch (_err) {
    return null;
  }
}

function clearPendingScenarioSelection(): void {
  try {
    localStorage.removeItem(BENCH_PENDING_SCENARIO_KEY);
  } catch (_err) {
    // ignore storage failures
  }
}

function renderSectionSidebar(current: PageDef, scenarios: BenchScenario[]): string {
  void scenarios;
  const rootSlug = sectionRootForPage(current);
  if (!rootSlug) {
    return "";
  }
  const root = PAGE_DEFS[rootSlug];
  if (!root) {
    return "";
  }
  const openState = readBenchTreeOpenState();
  const isOpen = (slug: string, fallback: boolean): boolean => {
    if (Object.prototype.hasOwnProperty.call(openState, slug)) {
      return openState[slug] === true;
    }
    return fallback;
  };

  const renderEntry = (slug: string): string => {
    const page = PAGE_DEFS[slug];
    if (!page) return "";
    const isActive = current.slug === page.slug;
    const isBranch = !isActive && current.slug.startsWith(`${page.slug}/`);
    const pageScenarios = scenariosForPage(page);

    if (page.children.length > 0) {
      const open = isOpen(page.slug, isActive || isBranch);
      return `
        <li class="bench-tree-node bench-tree-node-branch${isActive ? " is-current" : ""}${isBranch ? " is-branch" : ""}">
          <div class="bench-tree-head">
            <button type="button" class="bench-tree-toggle${open ? " is-open" : ""}" data-tree-toggle="${page.slug}" aria-expanded="${open ? "true" : "false"}" aria-label="${open ? "Collapse" : "Expand"} ${page.title}">
              <span class="bench-tree-caret" aria-hidden="true"></span>
            </button>
            <a class="bench-tree-link${isActive ? " is-active" : ""}" href="${toBenchPath(page.slug)}">
              ${page.title}
            </a>
          </div>
          <ul class="bench-tree-children${open ? " is-open" : ""}" data-tree-node="${page.slug}">
            ${page.children.map((childSlug) => renderEntry(childSlug)).join("")}
          </ul>
        </li>
      `;
    }

    const scenarioChildren = pageScenarios.length
      ? `
        <ul class="bench-tree-children bench-tree-scenario-list${isOpen(page.slug, isActive) ? " is-open" : ""}" data-tree-node="${page.slug}">
          ${pageScenarios
            .map((scenario, index) => {
              if (isActive) {
                return `
              <li>
                <button
                  type="button"
                  class="bench-tree-scenario bench-scenario-item${index === 0 ? " is-active" : ""}"
                  data-scenario-id="${scenario.id}"
                >
                  ${scenario.title}
                </button>
              </li>`;
              }
              return `
              <li>
                <button
                  type="button"
                  class="bench-tree-scenario bench-tree-scenario-nav"
                  data-nav-scenario-id="${scenario.id}"
                  data-nav-scenario-page="${page.slug}"
                >
                  ${scenario.title}
                </button>
              </li>`;
            })
            .join("")}
        </ul>
      `
      : "";

    const toggle = pageScenarios.length
      ? `<button type="button" class="bench-tree-toggle${isOpen(page.slug, isActive) ? " is-open" : ""}" data-tree-toggle="${page.slug}" aria-expanded="${isOpen(page.slug, isActive) ? "true" : "false"}" aria-label="${isOpen(page.slug, isActive) ? "Collapse" : "Expand"} ${page.title}">
          <span class="bench-tree-caret" aria-hidden="true"></span>
        </button>`
      : `<span class="bench-tree-toggle bench-tree-toggle-spacer" aria-hidden="true"></span>`;

    return `
      <li class="bench-tree-node bench-tree-node-leaf${isActive ? " is-current" : ""}">
        <div class="bench-tree-head">
          ${toggle}
          <a class="bench-tree-link${isActive ? " is-active" : ""}" href="${toBenchPath(page.slug)}">
            ${page.title}
          </a>
        </div>
        ${scenarioChildren}
      </li>
    `;
  };

  const tree = root.children.map((slug) => renderEntry(slug)).join("");

  return `
    <aside class="bench-sidebar">
      <h2 class="bench-sidebar-title">${root.title}</h2>
      <nav class="bench-sidebar-nav" aria-label="${root.title}">
        <ul class="bench-sidebar-tree">
          ${tree}
        </ul>
      </nav>
    </aside>
  `;
}

function renderChildCards(page: PageDef): string {
  if (page.children.length === 0) {
    return "";
  }

  const renderNode = (slug: string): string => {
    const child = PAGE_DEFS[slug];
    if (!child) {
      return "";
    }
    const count = scenarioCountForPage(child);
    const isActive = page.slug === child.slug;
    const expanded = page.slug === child.slug || page.slug.startsWith(`${child.slug}/`);
    const children = child.children
      .map((childSlug) => renderNode(childSlug))
      .filter((chunk) => chunk.length > 0)
      .join("");

    return `
      <li>
        <a class="bench-nav-tree-link${isActive ? " is-active" : ""}" href="${toBenchPath(child.slug)}">
          <span class="bench-nav-tree-main">
            <span class="bench-nav-tree-title">${child.title}</span>
            <span class="bench-nav-tree-sub">${child.subtitle}</span>
          </span>
          <span class="bench-nav-tree-count">${count}</span>
        </a>
        ${children.length > 0 ? `<ul class="bench-nav-tree-children${expanded ? " is-open" : ""}">${children}</ul>` : ""}
      </li>
    `;
  };

  const tree = page.children
    .map((slug) => renderNode(slug))
    .filter((chunk) => chunk.length > 0)
    .join("");

  return `
    <section id="bench-tree" class="bench-nav-tree-wrap">
      <h2 class="bench-section-title">Tree</h2>
      <ul class="bench-nav-tree">${tree}</ul>
    </section>
  `;
}

function renderRunner(page: PageDef, scenarios: BenchScenario[]): string {
  if (scenarios.length === 0 || page.children.length > 0) {
    return "";
  }

  return `
    <section id="bench-runner" class="bench-runner">
      <section class="bench-step-card bench-output">
        <div class="bench-run-bar">
          <select id="bench-responses-model" class="form-select form-select-inline" data-param="responses">
            <option value="">Loading...</option>
          </select>
          <button id="bench-settings-toggle" class="btn btn-ghost btn-icon" title="Settings">${ICON_SETTINGS}</button>
          <div class="flex-1"></div>
          <button id="bench-pause" class="btn btn-ghost btn-sm" disabled>Pause</button>
          <button id="bench-run" class="btn btn-primary btn-sm bench-run-btn">Start</button>
        </div>
        <div id="bench-model-variants" class="bench-model-variants hidden" data-param="responses"></div>
        <div id="bench-run-selected" class="bench-run-selected"></div>
        <div id="bench-settings-panel" class="bench-inline-settings-panel" style="display: none;">
          <div class="bench-grid bench-grid-primary">
            <label>Requests <input id="bench-requests" class="form-input form-input-sm bench-input-num" type="number" min="1" value="1" /></label>
            <label>Concurrency <input id="bench-concurrency" class="form-input form-input-sm bench-input-num" type="number" min="1" value="1" /></label>
            <label>Rounds <input id="bench-rounds" class="form-input form-input-sm bench-input-num" type="number" min="1" value="1" /></label>
            <label data-param="responses">Max Tokens <input id="bench-responses-max-output-tokens" class="form-input form-input-sm bench-input-num" type="number" min="1" value="1" /></label>
          </div>
          <div class="bench-grid bench-grid-advanced">
            <label class="bench-param" data-param="kv">KV Batch Size <input id="bench-kv-batch-size" class="form-input form-input-sm bench-input-num" type="number" min="1" value="1" /></label>
            <label class="bench-param" data-param="tables">Seed Rows <input id="bench-table-seed-rows" class="form-input form-input-sm bench-input-num" type="number" min="1" value="1" /></label>
            <label class="bench-param" data-param="vectors">Vector Dims <input id="bench-vector-dims" class="form-input form-input-sm bench-input-num" type="number" min="1" value="1" /></label>
          </div>
        </div>
        <div class="bench-run-toolbar">
          <div id="bench-run-status" class="bench-run-status">Idle</div>
          <div class="flex-1"></div>
          <button id="bench-clear" class="btn btn-ghost btn-icon bench-run-clear-icon" type="button" title="Clear run output">${ICON_DELETE}</button>
        </div>
        <section id="bench-progress-strip" class="bench-progress-strip" data-state="idle">
          <div class="bench-progress-strip-head">
            <strong id="bench-progress-phase">Idle</strong>
            <span id="bench-progress-round">Round 0 / 0</span>
            <span id="bench-progress-requests">Requests 0 / 0</span>
          </div>
          <div class="bench-progress-bar" aria-hidden="true">
            <div id="bench-progress-fill" class="bench-progress-fill"></div>
          </div>
          <div id="bench-progress-note" class="bench-progress-note">Waiting for a scenario run.</div>
        </section>
        <section id="bench-summary" class="bench-summary">
          <article class="bench-summary-card">
            <h3>Success</h3>
            <div id="bench-summary-success" class="bench-summary-value">—</div>
          </article>
          <article class="bench-summary-card">
            <h3>RPS avg</h3>
            <div id="bench-summary-rps" class="bench-summary-value">—</div>
          </article>
          <article class="bench-summary-card">
            <h3>p95 avg</h3>
            <div id="bench-summary-p95" class="bench-summary-value">—</div>
          </article>
          <article class="bench-summary-card">
            <h3>p99 avg</h3>
            <div id="bench-summary-p99" class="bench-summary-value">—</div>
          </article>
        </section>

        <section class="bench-results-wrap">
          <table class="bench-results">
            <thead>
              <tr>
                <th>Scenario</th>
                <th class="bench-num">Round</th>
                <th class="bench-num">Req</th>
                <th class="bench-num">Conc</th>
                <th class="bench-num">OK %</th>
                <th class="bench-num">RPS</th>
                <th class="bench-num">p50 ms</th>
                <th class="bench-num">p95 ms</th>
                <th class="bench-num">p99 ms</th>
              </tr>
            </thead>
            <tbody id="bench-results-body"></tbody>
          </table>
        </section>

        <section class="bench-log-wrap">
          <div class="bench-log-header">
            <button id="bench-log-copy" type="button" class="btn btn-ghost btn-icon bench-events-toolbar-btn" title="Copy log">${ICON_COPY}</button>
          </div>
          <pre id="bench-log" class="bench-log"></pre>
        </section>

        <section id="bench-events-wrap" class="bench-events-wrap">
          <header class="bench-events-header">
            <div class="bench-events-header-top">
              <h2>Server Events</h2>
              <button id="bench-events-clear" type="button" class="btn btn-ghost btn-icon bench-events-toolbar-btn" title="Clear">${ICON_DELETE}</button>
            </div>
            <div class="search-wrapper">
              ${ICON_SEARCH}
              <input id="bench-events-search" type="text" class="search-input" placeholder="Search events\u2026" />
            </div>
          </header>
          <div class="bench-events-log-wrap">
            <pre id="bench-events-log" class="bench-events-log">No events yet.</pre>
            <button id="bench-events-copy" type="button" class="btn btn-ghost btn-icon bench-events-copy-btn" title="Copy">${ICON_COPY}</button>
          </div>
          <div id="bench-events-levels-row" class="bench-events-filter-row is-dimmed">
            <input id="bench-events-levels-toggle" type="checkbox" class="bench-events-filter-master" title="Toggle all levels" />
            <div class="bench-events-filter-pills">
              ${BENCH_EVENT_LEVELS.map(
                (level) => `<button type="button" class="bench-events-pill" data-event-level="${level}" aria-pressed="false">${level} <span class="bench-events-pill-count">0</span></button>`,
              ).join("")}
            </div>
          </div>
          <div id="bench-events-topics-row" class="bench-events-filter-row" hidden>
            <input id="bench-events-topics-toggle" type="checkbox" class="bench-events-filter-master" title="Toggle all topics" />
            <div id="bench-events-topics-list" class="bench-events-filter-pills"></div>
          </div>
        </section>
      </section>
    </section>
  `;
}

function renderResultsBrowser(page: PageDef): string {
  if (page.slug !== "results") {
    return "";
  }

  return `
    <section id="bench-results-browser" class="bench-results-browser" data-results-scope="">
      <section class="bench-layout bench-results-layout">
        <aside class="bench-sidebar bench-results-sidebar">
          <div class="bench-results-toolbar">
            <input id="bench-results-search" class="form-input form-input-sm" placeholder="Filter..." />
            <span id="bench-results-count" class="bench-results-count">0</span>
          </div>
          <div id="bench-results-list" class="bench-results-list"></div>
          <div class="bench-results-sidebar-actions">
            <button id="bench-results-refresh" class="btn btn-ghost btn-sm">Refresh</button>
            <button id="bench-results-clear-filter" class="btn btn-ghost btn-sm" type="button">Clear</button>
            <button id="bench-results-delete-all" class="btn btn-ghost btn-sm bench-results-delete-all-btn">Delete All</button>
          </div>
        </aside>
        <div class="bench-main">
          <article id="bench-results-detail" class="bench-results-detail">
            <p class="bench-results-placeholder">Select a run to view details.</p>
          </article>
        </div>
      </section>
    </section>
  `;
}

function renderStartGuide(page: PageDef, scenarios: BenchScenario[]): string {
  void page;
  void scenarios;
  return "";
}

function renderNotFound(requestedSlug: string): string {
  return `
    <main class="bench-shell">
      <header class="bench-header">
        <h1>Bench Route Not Found</h1>
        <p>No bench page matches <code>${requestedSlug || "/"}</code>.</p>
      </header>
      <section class="bench-cards-wrap">
        <div class="bench-cards">
          <a class="bench-card" href="${toBenchPath("")}">
            <div class="bench-card-head"><h3>Bench Hub</h3><span>Home</span></div>
            <p>Open the main bench navigation.</p>
            <div class="bench-card-subtitle">/bench/</div>
          </a>
          <a class="bench-card" href="${toBenchPath("responses")}">
            <div class="bench-card-head"><h3>Responses</h3><span>Category</span></div>
            <p>Responses perf and eval suites.</p>
            <div class="bench-card-subtitle">/bench/responses/</div>
          </a>
          <a class="bench-card" href="${toBenchPath("db")}">
            <div class="bench-card-head"><h3>DB</h3><span>Category</span></div>
            <p>DB subcategory benches.</p>
            <div class="bench-card-subtitle">/bench/db/</div>
          </a>
          <a class="bench-card" href="${toBenchPath("results")}">
            <div class="bench-card-head"><h3>Results</h3><span>History</span></div>
            <p>Browse and compare saved benchmark runs.</p>
            <div class="bench-card-subtitle">/bench/results/</div>
          </a>
        </div>
      </section>
    </main>
  `;
}

const DEFAULT_CHILD_PAGE: Record<string, string> = {
  "": "responses/perf",
  responses: "responses/perf",
  "responses/evals": "responses/evals/mmlu",
  db: "db/kv",
};

function render(): { page: PageDef | null; scenarios: BenchScenario[] } {
  const existing = document.getElementById("bench-root");
  const root =
    existing ??
    (() => {
      document.body.innerHTML = "";
      const element = document.createElement("div");
      element.id = "bench-root";
      document.body.appendChild(element);
      return element;
    })();

  const resolved = resolveBenchPage(currentBenchPathForResolve());
  if (resolved.status === "not_found") {
    root.innerHTML = renderNotFound(resolved.requestedSlug);
    return { page: null, scenarios: [] };
  }

  const page = resolved.page;
  const embedded = window.self !== window.top;
  const defaultChild = DEFAULT_CHILD_PAGE[page.slug];
  if (defaultChild) {
    window.location.replace(toBenchPath(defaultChild));
    return { page: null, scenarios: [] };
  }

  const scenarios = scenariosForPage(page);
  const isResultsPage = page.slug.startsWith("results");
  const isRunnerPage = !isResultsPage && page.children.length === 0 && scenarios.length > 0;
  const showSidebar = !isResultsPage && sectionRootForPage(page) !== null;

  root.innerHTML = `
    <main class="bench-shell${isRunnerPage ? " is-runner" : ""}${embedded ? " is-embedded" : ""}">
      ${renderTopNav(page)}
      ${embedded
        ? ""
        : `<header class="bench-header">
        <h1>${page.title}</h1>
        <p>${page.description}</p>
      </header>`}

      ${renderStartGuide(page, scenarios)}
      ${page.slug === "" ? renderChildCards(page) : ""}
      ${showSidebar ? `
      <section class="bench-layout">
        ${renderSectionSidebar(page, scenarios)}
        <div class="bench-main">
          ${renderRunner(page, scenarios)}
        </div>
      </section>` : `
      ${renderResultsBrowser(page)}
      ${renderRunner(page, scenarios)}
      `}
    </main>
  `;

  return { page, scenarios };
}

function parsePositiveInt(id: string, fallback: number): number {
  const input = document.getElementById(id) as HTMLInputElement | null;
  const parsed = Number.parseInt(input?.value ?? "", 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }
  return parsed;
}

function parseString(id: string, fallback: string): string {
  const input = document.getElementById(id) as HTMLInputElement | null;
  const value = (input?.value ?? "").trim();
  return value.length > 0 ? value : fallback;
}

async function resolveResponsesModel(baseUrl: string, explicitModel: string): Promise<string> {
  const direct = explicitModel.trim();
  if (direct.length > 0) {
    return direct;
  }

  const resp = await request(baseUrl, "GET", "/v1/settings");
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`unable to resolve active model: ${resp.status} ${txt}`);
  }
  const payload = (await resp.json()) as Record<string, unknown>;
  const model = typeof payload.model === "string" ? payload.model.trim() : "";
  if (model.length > 0) {
    return model;
  }

  const modelsResp = await request(baseUrl, "GET", "/v1/models");
  if (!modelsResp.ok) {
    const txt = await modelsResp.text();
    throw new Error(`unable to resolve model list: ${modelsResp.status} ${txt}`);
  }
  const modelsPayload = (await modelsResp.json()) as { data?: Array<{ id?: string }> };
  const fallback = Array.isArray(modelsPayload.data)
    ? modelsPayload.data.find((item) => typeof item.id === "string" && item.id.trim().length > 0)
    : null;
  if (!fallback?.id) {
    throw new Error("responses benches need an available model; configure Settings > Model first");
  }
  return fallback.id;
}

function appendLog(message: string): void {
  const el = document.getElementById("bench-log");
  if (!el) return;
  const stamp = new Date().toISOString().slice(11, 19);
  el.textContent += `[${stamp}] ${message}\n`;
  el.scrollTop = el.scrollHeight;
}

function normalizeBenchEventLevel(value: string | null | undefined): BenchEventLevel {
  const upper = String(value ?? "").trim().toUpperCase();
  if (upper === "TRACE" || upper === "DEBUG" || upper === "INFO" || upper === "WARN" || upper === "ERROR") {
    return upper;
  }
  if (upper === "WARNING") {
    return "WARN";
  }
  return "INFO";
}

function renderBenchEventsLog(): void {
  const el = document.getElementById("bench-events-log");
  if (!el) return;

  if (benchEventLines.length === 0) {
    el.textContent = "No events yet.";
    return;
  }

  const needle = benchEventSearchText.toLowerCase();
  const visible = benchEventLines.filter(
    (line) =>
      benchEventSelectedLevels.has(line.level) &&
      benchEventSelectedTopics.has(line.topic) &&
      (needle.length === 0 || line.message.toLowerCase().includes(needle)),
  );
  if (visible.length === 0) {
    el.textContent = "No events match current filters.";
    return;
  }

  let topicW = 0;
  for (const line of visible) {
    if (line.topic.length > topicW) topicW = line.topic.length;
  }

  el.textContent = visible
    .map((line) => {
      const time = line.text.slice(0, 8);
      const lvl = line.level.padEnd(5);
      const topic = line.topic.padEnd(topicW);
      return `${time}  ${lvl}  ${topic}  ${line.message}`;
    })
    .join("\n");
  el.scrollTop = el.scrollHeight;
  updatePillCounts();
}

function updatePillCounts(): void {
  const levelCounts = new Map<string, number>();
  const topicCounts = new Map<string, number>();
  for (const line of benchEventLines) {
    levelCounts.set(line.level, (levelCounts.get(line.level) ?? 0) + 1);
    topicCounts.set(line.topic, (topicCounts.get(line.topic) ?? 0) + 1);
  }
  document.querySelectorAll<HTMLButtonElement>(".bench-events-pill[data-event-level]").forEach((btn) => {
    const countEl = btn.querySelector(".bench-events-pill-count");
    if (countEl) countEl.textContent = String(levelCounts.get(btn.dataset.eventLevel ?? "") ?? 0);
  });
  document.querySelectorAll<HTMLButtonElement>(".bench-events-pill[data-event-topic]").forEach((btn) => {
    const countEl = btn.querySelector(".bench-events-pill-count");
    if (countEl) countEl.textContent = String(topicCounts.get(btn.dataset.eventTopic ?? "") ?? 0);
  });
}

function clearBenchEventsLog(): void {
  benchEventLines.length = 0;
  benchEventDiscoveredTopics.length = 0;
  benchEventSelectedTopics.clear();
  benchEventSearchText = "";
  const searchInput = document.getElementById("bench-events-search") as HTMLInputElement | null;
  if (searchInput) searchInput.value = "";
  document.getElementById("bench-events-topics-row")?.setAttribute("hidden", "");
  document.getElementById("bench-events-levels-row")?.classList.add("is-dimmed");
  renderBenchEventsLog();
}

async function copyBenchEventsLog(): Promise<boolean> {
  const el = document.getElementById("bench-events-log");
  const text = el?.textContent?.trim() ?? "";
  if (!text || text === "No events yet." || text === "No events match current filters.") {
    return false;
  }
  try {
    await writeClipboardText(text);
    return true;
  } catch (_err) {
    return false;
  }
}

function discoverTopic(topic: string): void {
  if (benchEventDiscoveredTopics.includes(topic)) return;
  benchEventDiscoveredTopics.push(topic);
  benchEventSelectedTopics.add(topic);
  renderBenchEventsTopics();
}

function renderBenchEventsTopics(): void {
  const row = document.getElementById("bench-events-topics-row");
  const levelsRow = document.getElementById("bench-events-levels-row");
  const list = document.getElementById("bench-events-topics-list");
  if (!list) return;
  if (benchEventDiscoveredTopics.length === 0) {
    row?.setAttribute("hidden", "");
    levelsRow?.classList.add("is-dimmed");
    list.innerHTML = "";
    return;
  }
  row?.removeAttribute("hidden");
  levelsRow?.classList.remove("is-dimmed");
  syncLevelPills();
  list.innerHTML = benchEventDiscoveredTopics
    .map((t) => {
      const active = benchEventSelectedTopics.has(t);
      return `<button type="button" class="bench-events-pill${active ? " is-active" : ""}" data-event-topic="${t}" aria-pressed="${active}">${t} <span class="bench-events-pill-count">0</span></button>`;
    })
    .join("");
  list.querySelectorAll<HTMLButtonElement>("[data-event-topic]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const topic = btn.dataset.eventTopic ?? "";
      if (benchEventSelectedTopics.has(topic)) {
        benchEventSelectedTopics.delete(topic);
      } else {
        benchEventSelectedTopics.add(topic);
      }
      btn.classList.toggle("is-active", benchEventSelectedTopics.has(topic));
      btn.setAttribute("aria-pressed", benchEventSelectedTopics.has(topic) ? "true" : "false");
      syncTopicCheckbox();
      renderBenchEventsLog();
    });
  });
  syncTopicCheckbox();
}

function appendBenchEventLine(text: string, level?: string, topic?: string, message?: string): void {
  const resolvedTopic = topic ?? "events.log";
  const wasEmpty = benchEventLines.length === 0;
  discoverTopic(resolvedTopic);
  if (wasEmpty) {
    syncLevelPills();
  }
  benchEventLines.push({
    level: normalizeBenchEventLevel(level),
    topic: resolvedTopic,
    message: message ?? text,
    text,
  });
  while (benchEventLines.length > MAX_BENCH_EVENT_LINES) {
    benchEventLines.shift();
  }
  renderBenchEventsLog();
}

function syncLevelCheckbox(): void {
  const cb = document.getElementById("bench-events-levels-toggle") as HTMLInputElement | null;
  if (!cb) return;
  const selected = benchEventSelectedLevels.size;
  const total = BENCH_EVENT_LEVELS.length;
  cb.checked = selected === total;
  cb.indeterminate = selected > 0 && selected < total;
}

function syncLevelPills(): void {
  document.querySelectorAll<HTMLButtonElement>(".bench-events-pill[data-event-level]").forEach((btn) => {
    const level = normalizeBenchEventLevel(btn.dataset.eventLevel ?? "");
    const active = benchEventSelectedLevels.has(level);
    btn.classList.toggle("is-active", active);
    btn.setAttribute("aria-pressed", active ? "true" : "false");
  });
  syncLevelCheckbox();
}

function wireBenchEventLevelFilters(): void {
  const buttons = Array.from(
    document.querySelectorAll<HTMLButtonElement>(".bench-events-pill[data-event-level]"),
  );
  const cb = document.getElementById("bench-events-levels-toggle") as HTMLInputElement | null;
  if (buttons.length === 0) return;

  buttons.forEach((button) => {
    const level = normalizeBenchEventLevel(button.dataset.eventLevel ?? "");
    button.classList.toggle("is-active", benchEventSelectedLevels.has(level));
    button.addEventListener("click", () => {
      if (benchEventSelectedLevels.has(level)) {
        benchEventSelectedLevels.delete(level);
      } else {
        benchEventSelectedLevels.add(level);
      }
      button.classList.toggle("is-active", benchEventSelectedLevels.has(level));
      button.setAttribute("aria-pressed", benchEventSelectedLevels.has(level) ? "true" : "false");
      syncLevelCheckbox();
      renderBenchEventsLog();
    });
  });

  cb?.addEventListener("change", () => {
    benchEventSelectedLevels.clear();
    if (cb.checked) {
      BENCH_EVENT_LEVELS.forEach((l) => benchEventSelectedLevels.add(l));
    }
    cb.indeterminate = false;
    syncLevelPills();
    renderBenchEventsLog();
  });

  syncLevelCheckbox();
  renderBenchEventsLog();
}

function syncTopicCheckbox(): void {
  const cb = document.getElementById("bench-events-topics-toggle") as HTMLInputElement | null;
  if (!cb) return;
  const selected = benchEventSelectedTopics.size;
  const total = benchEventDiscoveredTopics.length;
  cb.checked = total > 0 && selected === total;
  cb.indeterminate = selected > 0 && selected < total;
}

function wireBenchEventTopicFilters(): void {
  const cb = document.getElementById("bench-events-topics-toggle") as HTMLInputElement | null;
  cb?.addEventListener("change", () => {
    benchEventSelectedTopics.clear();
    if (cb.checked) {
      benchEventDiscoveredTopics.forEach((t) => benchEventSelectedTopics.add(t));
    }
    cb.indeterminate = false;
    renderBenchEventsTopics();
    renderBenchEventsLog();
  });
}

function wireBenchEventSearch(): void {
  const input = document.getElementById("bench-events-search") as HTMLInputElement | null;
  if (!input) return;
  input.addEventListener("input", () => {
    benchEventSearchText = input.value.trim();
    renderBenchEventsLog();
  });
}

function formatEventTime(tsMs?: number): string {
  const ts = typeof tsMs === "number" && Number.isFinite(tsMs) ? tsMs : Date.now();
  return new Date(ts).toLocaleTimeString([], { hour12: false });
}

function stopBenchEventsStream(): void {
  benchEventsAbort?.abort();
  benchEventsAbort = null;
}

async function runBenchEventsStream(baseUrl: string, signal: AbortSignal): Promise<void> {
  let resp: Response;
  try {
    resp = await fetch(`${baseUrl}/v1/events/stream?verbosity=2`, {
      method: "GET",
      headers: { Accept: "text/event-stream" },
      signal,
    });
  } catch (err) {
    if (!signal.aborted) {
      appendBenchEventLine(
        `${formatEventTime()} ERROR events stream: ${err instanceof Error ? err.message : String(err)}`,
        "ERROR",
      );
    }
    return;
  }

  if (!resp.ok) {
    appendBenchEventLine(
      `${formatEventTime()} WARN events stream failed: ${resp.status} ${resp.statusText}`,
      "WARN",
    );
    return;
  }

  const reader = resp.body?.getReader();
  if (!reader) {
    appendBenchEventLine(`${formatEventTime()} WARN events stream unavailable (empty body)`, "WARN");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "";

  try {
    while (!signal.aborted) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const raw of lines) {
        const line = raw.trimEnd();
        if (line.startsWith(":")) continue;
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7).trim();
          continue;
        }
        if (line.startsWith("data: ")) {
          const payload = line.slice(6);
          if (currentEvent === "event") {
            try {
              const envelope = JSON.parse(payload) as BenchEventEnvelope;
              const time = formatEventTime(envelope.ts_ms);
              const level = normalizeBenchEventLevel(envelope.level);
              const topic = typeof envelope.topic === "string" ? envelope.topic : "events.log";
              const message = typeof envelope.message === "string" ? envelope.message : "";
              appendBenchEventLine(`${time} ${level} ${topic} ${message}`.trim(), level, topic, message);
            } catch {
              // ignore malformed payloads
            }
          } else {
            const level = normalizeBenchEventLevel(currentEvent);
            appendBenchEventLine(`${formatEventTime()} ${currentEvent || "EVENT"} ${payload}`, level);
          }
          continue;
        }
        if (line === "") {
          currentEvent = "";
        }
      }
    }
  } catch (err) {
    if (!(err instanceof DOMException && err.name === "AbortError")) {
      appendBenchEventLine(
        `${formatEventTime()} ERROR events stream: ${err instanceof Error ? err.message : String(err)}`,
        "ERROR",
      );
    }
  }
}

function startBenchEventsStream(baseUrl: string): void {
  stopBenchEventsStream();
  const controller = new AbortController();
  benchEventsAbort = controller;
  appendBenchEventLine(`${formatEventTime()} INFO subscribed to /v1/events/stream`, "INFO");
  void runBenchEventsStream(baseUrl, controller.signal);
}

function appendResultRow(
  scenario: BenchScenario,
  round: number,
  cfg: RunConfig,
  metrics: LoadMetrics,
): void {
  const body = document.getElementById("bench-results-body");
  if (!body) return;
  const okPct = metrics.requests > 0 ? (100 * metrics.ok) / metrics.requests : 0;
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td>
      <div>${scenario.id}</div>
      <div class="bench-result-route">
        <span class="bench-method method-${scenario.method.toLowerCase()}">${scenario.method}</span>
        ${scenario.pathTemplate}
      </div>
    </td>
    <td class="bench-num">${round}</td>
    <td class="bench-num">${metrics.requests}</td>
    <td class="bench-num">${cfg.concurrency}</td>
    <td class="bench-num">${round2(okPct)}</td>
    <td class="bench-num">${round2(metrics.rps)}</td>
    <td class="bench-num">${round2(metrics.p50Ms)}</td>
    <td class="bench-num">${round2(metrics.p95Ms)}</td>
    <td class="bench-num">${round2(metrics.p99Ms)}</td>
  `;
  body.appendChild(tr);
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const rank = Math.ceil((p / 100) * sorted.length);
  const idx = Math.min(sorted.length - 1, Math.max(0, rank - 1));
  return sorted[idx];
}

async function runLoad(
  totalRequests: number,
  concurrency: number,
  requestFn: (index: number) => Promise<Response>,
  classifySuccess?: (
    response: Response,
    bodyText: string,
    index: number,
  ) => boolean | Promise<boolean>,
  control?: RunControlState,
  onProgress?: (completed: number, total: number) => void,
  onError?: (status: number, body: string) => void,
): Promise<LoadMetrics> {
  let next = 0;
  let ok = 0;
  let errors = 0;
  let completed = 0;
  const latencies: number[] = [];
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let totalPrefillTokS = 0;
  let totalGenTokS = 0;
  let totalTtftMs = 0;
  let usageSamples = 0;

  const t0 = performance.now();

  async function worker(): Promise<void> {
    while (true) {
      if (control?.stopRequested) {
        return;
      }
      while (control?.paused && !control.stopRequested) {
        await new Promise((resolve) => setTimeout(resolve, 120));
      }
      if (control?.stopRequested) {
        return;
      }

      const index = next;
      next += 1;
      if (index >= totalRequests) {
        return;
      }

      const r0 = performance.now();
      try {
        const resp = await requestFn(index);
        const bodyText = await resp.text();
        const usage = extractResponseUsage(bodyText);
        if (
          usage.inputTokens > 0 ||
          usage.outputTokens > 0 ||
          usage.prefillTokS > 0 ||
          usage.genTokS > 0 ||
          usage.ttftMs > 0
        ) {
          totalInputTokens += usage.inputTokens;
          totalOutputTokens += usage.outputTokens;
          totalPrefillTokS += usage.prefillTokS;
          totalGenTokS += usage.genTokS;
          totalTtftMs += usage.ttftMs;
          usageSamples += 1;
        }
        let isOk = resp.ok;
        if (isOk && classifySuccess) {
          isOk = await classifySuccess(resp, bodyText, index);
        }
        if (isOk) {
          ok += 1;
        } else {
          if (errors === 0) onError?.(resp.status, bodyText);
          errors += 1;
        }
      } catch (_err) {
        errors += 1;
      } finally {
        completed += 1;
        latencies.push(performance.now() - r0);
        onProgress?.(completed, totalRequests);
      }
    }
  }

  const workers = Array.from({ length: concurrency }, () => worker());
  await Promise.all(workers);

  const wallMs = Math.max(1, performance.now() - t0);
  latencies.sort((a, b) => a - b);
  const avgMs = latencies.length
    ? latencies.reduce((sum, value) => sum + value, 0) / latencies.length
    : 0;

  return {
    requests: completed,
    ok,
    errors,
    wallSeconds: wallMs / 1000,
    rps: completed / (wallMs / 1000),
    avgMs,
    p50Ms: percentile(latencies, 50),
    p95Ms: percentile(latencies, 95),
    p99Ms: percentile(latencies, 99),
    inputTokens: totalInputTokens,
    outputTokens: totalOutputTokens,
    avgPrefillTokS: usageSamples > 0 ? totalPrefillTokS / usageSamples : 0,
    avgGenTokS: usageSamples > 0 ? totalGenTokS / usageSamples : 0,
    avgTtftMs: usageSamples > 0 ? totalTtftMs / usageSamples : 0,
  };
}

function round3(value: number): string {
  return value.toFixed(3);
}

function round2(value: number): string {
  return value.toFixed(2);
}

function defaultRunProgressState(): RunProgressState {
  return {
    scenarioId: "",
    round: 0,
    totalRounds: 0,
    completedRequests: 0,
    totalRequests: 0,
    phase: "Idle",
    note: "Waiting for a scenario run.",
    tone: "idle",
  };
}

function setRunProgressUi(progress: RunProgressState): void {
  const strip = document.getElementById("bench-progress-strip");
  const phaseEl = document.getElementById("bench-progress-phase");
  const roundEl = document.getElementById("bench-progress-round");
  const requestsEl = document.getElementById("bench-progress-requests");
  const noteEl = document.getElementById("bench-progress-note");
  const fillEl = document.getElementById("bench-progress-fill");
  if (!strip || !phaseEl || !roundEl || !requestsEl || !noteEl || !fillEl) {
    return;
  }

  const totalRounds = Math.max(0, progress.totalRounds);
  const totalRequests = Math.max(0, progress.totalRequests);
  const roundFraction =
    totalRounds > 0
      ? ((Math.max(0, progress.round - 1) +
          (totalRequests > 0 ? progress.completedRequests / totalRequests : 0)) /
          totalRounds) *
        100
      : 0;
  const pct = progress.tone === "ok" ? 100 : Math.max(0, Math.min(100, roundFraction));

  strip.dataset.state = progress.tone;
  phaseEl.textContent = progress.phase;
  roundEl.textContent = `Round ${progress.round} / ${progress.totalRounds}`;
  requestsEl.textContent = `Requests ${progress.completedRequests} / ${progress.totalRequests}`;
  noteEl.textContent = progress.note;
  fillEl.style.width = `${pct}%`;
}

async function runScenario(
  scenario: BenchScenario,
  cfg: RunConfig,
  baseUrl: string,
  control: RunControlState,
  onProgress?: (progress: RunProgressState) => void,
): Promise<{ rounds: Array<{ round: number; metrics: LoadMetrics }>; stopped: boolean }> {
  const roundResults: Array<{ round: number; metrics: LoadMetrics }> = [];
  for (let round = 1; round <= cfg.rounds; round++) {
    if (control.stopRequested) {
      appendLog(`scenario=${scenario.id} stop requested before round ${round}`);
      break;
    }
    onProgress?.({
      scenarioId: scenario.id,
      round,
      totalRounds: cfg.rounds,
      completedRequests: 0,
      totalRequests: cfg.requests,
      phase: "Preparing",
      note: `Preparing round ${round} resources and fixtures.`,
      tone: "running",
    });
    appendLog(`scenario=${scenario.id} round=${round} prepare`);
    const ctx: ScenarioContext = { baseUrl, round, state: {} };
    try {
      if (scenario.prepare) {
        await scenario.prepare(ctx, cfg);
      }
      appendLog(`scenario=${scenario.id} round=${round} run`);
      const metrics = await runLoad(
        cfg.requests,
        cfg.concurrency,
        (index) => scenario.request(ctx, index, cfg),
        scenario.classifySuccess
          ? (resp, bodyText, index) =>
              scenario.classifySuccess!(resp, bodyText, index, ctx, cfg)
          : undefined,
        control,
        (completed, total) => {
          onProgress?.({
            scenarioId: scenario.id,
            round,
            totalRounds: cfg.rounds,
            completedRequests: completed,
            totalRequests: total,
            phase: control.paused ? "Paused" : "Running",
            note:
              completed >= total
                ? `Round ${round} complete. Summarizing metrics.`
                : `Round ${round} in flight: ${completed} of ${total} requests complete.`,
            tone: control.stopRequested ? "error" : "running",
          });
        },
        (status, body) => {
          console.error(`[bench] ${scenario.id} round=${round} HTTP ${status}:`, body);
          appendLog(`error: ${status} ${body.slice(0, 300)}`);
        },
      );
      appendResultRow(scenario, round, cfg, metrics);
      roundResults.push({ round, metrics });
      onProgress?.({
        scenarioId: scenario.id,
        round,
        totalRounds: cfg.rounds,
        completedRequests: metrics.requests,
        totalRequests: cfg.requests,
        phase: "Round Complete",
        note: `Round ${round} finished at ${round2(metrics.rps)} RPS with p95 ${round2(metrics.p95Ms)} ms.`,
        tone: "running",
      });
      appendLog(
        `scenario=${scenario.id} round=${round} ok=${metrics.ok}/${metrics.requests} rps=${round2(metrics.rps)} p95=${round2(metrics.p95Ms)}ms${
          metrics.avgGenTokS > 0 ? ` gen_tps=${round2(metrics.avgGenTokS)}` : ""
        }${metrics.avgPrefillTokS > 0 ? ` prefill_tps=${round2(metrics.avgPrefillTokS)}` : ""}`,
      );
    } catch (err) {
      onProgress?.({
        scenarioId: scenario.id,
        round,
        totalRounds: cfg.rounds,
        completedRequests: 0,
        totalRequests: cfg.requests,
        phase: "Failed",
        note: err instanceof Error ? err.message : String(err),
        tone: "error",
      });
      appendLog(
        `scenario=${scenario.id} round=${round} failed=${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      if (scenario.cleanup) {
        try {
          await scenario.cleanup(ctx, cfg);
        } catch (_err) {
          // Cleanup failures should not stop suite progress.
        }
      }
    }
  }
  return { rounds: roundResults, stopped: control.stopRequested };
}

function selectedScenarioId(): string | null {
  const selected = document.querySelector<HTMLElement>(".bench-scenario-item.is-active");
  return selected?.dataset.scenarioId ?? null;
}

function setParamVisibility(name: string, visible: boolean): void {
  document.querySelectorAll<HTMLElement>(`[data-param="${name}"]`).forEach((el) => {
    el.style.display = visible ? "" : "none";
  });
}

function updateActiveScenarioUi(scenario: BenchScenario | null): void {
  const runSelected = document.getElementById("bench-run-selected");
  if (!scenario) {
    setParamVisibility("responses", false);
    setParamVisibility("kv", false);
    setParamVisibility("tables", false);
    setParamVisibility("vectors", false);
    if (runSelected) {
      runSelected.textContent = "No scenario selected.";
    }
    return;
  }

  const needsResponses = scenario.scope.startsWith("responses/");
  const needsKv = scenario.scope === "db/kv";
  const needsTables = scenario.scope === "db/tables";
  const needsVectors = scenario.scope === "db/vectors" || scenario.scope === "db/ops";

  setParamVisibility("responses", needsResponses);
  setParamVisibility("kv", needsKv);
  setParamVisibility("tables", needsTables);
  setParamVisibility("vectors", needsVectors);

  // Show/hide the "More Settings" details — only when it has visible params.
  const details = document.querySelector<HTMLDetailsElement>(".bench-advanced");
  if (details) {
    const hasAdvanced = needsKv || needsTables || needsVectors;
    details.style.display = hasAdvanced ? "" : "none";
    if (hasAdvanced) details.open = true;
  }

  if (runSelected) {
    runSelected.textContent = "";
  }
}

function setRunStatus(message: string, tone: "idle" | "running" | "ok" | "error"): void {
  const el = document.getElementById("bench-run-status");
  if (!el) return;
  el.textContent = message;
  el.dataset.state = tone;
}

function setSummaryValue(id: string, value: string): void {
  const el = document.getElementById(id);
  if (el) {
    el.textContent = value;
  }
}

function clearSummaryUi(): void {
  setSummaryValue("bench-summary-success", "—");
  setSummaryValue("bench-summary-rps", "—");
  setSummaryValue("bench-summary-p95", "—");
  setSummaryValue("bench-summary-p99", "—");
  setSummaryValue("bench-summary-wall", "—");
  setSummaryValue("bench-summary-input-tok", "—");
  setSummaryValue("bench-summary-output-tok", "—");
  setSummaryValue("bench-summary-prefill-tps", "—");
  setSummaryValue("bench-summary-gen-tps", "—");
  setSummaryValue("bench-summary-ttft", "—");
}

function updateSummaryUi(rounds: Array<{ round: number; metrics: LoadMetrics }>): void {
  if (rounds.length === 0) {
    clearSummaryUi();
    return;
  }

  const totalReq = rounds.reduce((sum, item) => sum + item.metrics.requests, 0);
  const totalOk = rounds.reduce((sum, item) => sum + item.metrics.ok, 0);
  const success = totalReq > 0 ? (100 * totalOk) / totalReq : 0;
  const avgRps = rounds.reduce((sum, item) => sum + item.metrics.rps, 0) / rounds.length;
  const avgP95 = rounds.reduce((sum, item) => sum + item.metrics.p95Ms, 0) / rounds.length;
  const avgP99 = rounds.reduce((sum, item) => sum + item.metrics.p99Ms, 0) / rounds.length;
  const avgWall =
    rounds.reduce((sum, item) => sum + item.metrics.wallSeconds, 0) / rounds.length;
  const totalInputTokens = rounds.reduce((sum, item) => sum + item.metrics.inputTokens, 0);
  const totalOutputTokens = rounds.reduce((sum, item) => sum + item.metrics.outputTokens, 0);
  const prefillSamples = rounds
    .map((item) => item.metrics.avgPrefillTokS)
    .filter((value) => value > 0);
  const genSamples = rounds.map((item) => item.metrics.avgGenTokS).filter((value) => value > 0);
  const ttftSamples = rounds.map((item) => item.metrics.avgTtftMs).filter((value) => value > 0);
  const avgPrefillTps = prefillSamples.length
    ? prefillSamples.reduce((sum, value) => sum + value, 0) / prefillSamples.length
    : 0;
  const avgGenTps = genSamples.length
    ? genSamples.reduce((sum, value) => sum + value, 0) / genSamples.length
    : 0;
  const avgTtft = ttftSamples.length
    ? ttftSamples.reduce((sum, value) => sum + value, 0) / ttftSamples.length
    : 0;

  setSummaryValue("bench-summary-success", `${round2(success)}%`);
  setSummaryValue("bench-summary-rps", round2(avgRps));
  setSummaryValue("bench-summary-p95", `${round2(avgP95)} ms`);
  setSummaryValue("bench-summary-p99", `${round2(avgP99)} ms`);
  setSummaryValue("bench-summary-wall", `${round3(avgWall)} s`);
  setSummaryValue(
    "bench-summary-input-tok",
    totalInputTokens > 0 ? `${Math.round(totalInputTokens)}` : "—",
  );
  setSummaryValue(
    "bench-summary-output-tok",
    totalOutputTokens > 0 ? `${Math.round(totalOutputTokens)}` : "—",
  );
  setSummaryValue(
    "bench-summary-prefill-tps",
    avgPrefillTps > 0 ? `${round2(avgPrefillTps)}` : "—",
  );
  setSummaryValue("bench-summary-gen-tps", avgGenTps > 0 ? `${round2(avgGenTps)}` : "—");
  setSummaryValue("bench-summary-ttft", avgTtft > 0 ? `${round2(avgTtft)} ms` : "—");
}

type PersistedBenchRun = {
  id: string;
  title: string;
  updatedAt: number;
  createdAt: number;
  scenario: string;
  pythonScenario: string | null;
  scope: string;
  stopped: boolean;
  rounds: number;
  summary: {
    successPct: number;
    avgRps: number;
    avgP95Ms: number;
    avgP99Ms: number;
    avgWallS: number;
    totalInputTokens: number;
    totalOutputTokens: number;
    avgPrefillTokS: number;
    avgGenTokS: number;
    avgTtftMs: number;
  };
  fileId: string | null;
  fileName: string | null;
  generatedAt: string | null;
  cfg: Record<string, unknown>;
};

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatRunTimestamp(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return "n/a";
  return new Date(ms).toISOString().replace("T", " ").slice(0, 19);
}

function numberFromUnknown(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return 0;
}

const BENCH_RERUN_KEY = "bench:rerun-seed";

type BenchRerunSeed = {
  scenario: string;
  scope: string;
  cfg: Record<string, unknown>;
  createdAt: number;
};

function toScenarioPage(scope: string): string {
  if (PAGE_DEFS[scope]) {
    return scope;
  }
  if (scope.startsWith("responses/evals/")) {
    return scope;
  }
  if (scope.startsWith("responses/")) {
    return "responses/perf";
  }
  if (scope.startsWith("db/")) {
    return scope;
  }
  return "";
}

function writeRerunSeed(run: PersistedBenchRun): void {
  const payload: BenchRerunSeed = {
    scenario: run.scenario,
    scope: run.scope,
    cfg: run.cfg,
    createdAt: Date.now(),
  };
  try {
    localStorage.setItem(BENCH_RERUN_KEY, JSON.stringify(payload));
  } catch (_err) {
    // ignore storage failures
  }
}

function readRerunSeed(): BenchRerunSeed | null {
  try {
    const raw = localStorage.getItem(BENCH_RERUN_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null;
    }
    const obj = parsed as Record<string, unknown>;
    const scenario = typeof obj.scenario === "string" ? obj.scenario : "";
    const scope = typeof obj.scope === "string" ? obj.scope : "";
    const cfgRaw = obj.cfg;
    const cfg =
      cfgRaw && typeof cfgRaw === "object" && !Array.isArray(cfgRaw)
        ? (cfgRaw as Record<string, unknown>)
        : {};
    const createdAt = toFiniteNumber(obj.createdAt);
    if (!scenario || !scope || !Number.isFinite(createdAt) || createdAt <= 0) {
      return null;
    }
    if (Date.now() - createdAt > 1000 * 60 * 30) {
      localStorage.removeItem(BENCH_RERUN_KEY);
      return null;
    }
    return { scenario, scope, cfg, createdAt };
  } catch (_err) {
    return null;
  }
}

function clearRerunSeed(): void {
  try {
    localStorage.removeItem(BENCH_RERUN_KEY);
  } catch (_err) {
    // ignore storage failures
  }
}

const EMPTY_SUMMARY = {
  successPct: 0, avgRps: 0, avgP95Ms: 0, avgP99Ms: 0, avgWallS: 0,
  totalInputTokens: 0, totalOutputTokens: 0, avgPrefillTokS: 0,
  avgGenTokS: 0, avgTtftMs: 0,
};

/** Load bench runs with full content in a single request. */
async function loadPersistedBenchRuns(
  baseUrl: string,
  scopePrefix: string,
): Promise<PersistedBenchRun[]> {
  const listResp = await request(
    baseUrl, "GET", "/v1/db/tables/documents?include=content&type=bench_run&limit=120",
  );
  if (!listResp.ok) return [];
  let listPayload: { data?: Array<Record<string, unknown>> } = {};
  try {
    listPayload = (await listResp.json()) as { data?: Array<Record<string, unknown>> };
  } catch (_err) {
    return [];
  }
  const docs = Array.isArray(listPayload.data) ? listPayload.data : [];
  const details: PersistedBenchRun[] = [];

  for (const doc of docs) {
    const docId = typeof doc.id === "string" ? doc.id : "";
    if (!docId) continue;
    const marker = typeof doc.marker === "string" ? doc.marker : "";
    const scope = marker && marker !== "active" ? marker : "";
    if (scopePrefix.length > 0 && !scope.startsWith(scopePrefix)) continue;
    const title = typeof doc.title === "string" ? doc.title : "Bench Run";

    // Parse content (available via ?include=content).
    const contentRaw = doc.content;
    const contentObj =
      contentRaw && typeof contentRaw === "object" && !Array.isArray(contentRaw)
        ? (contentRaw as Record<string, unknown>)
        : {};
    const summaryRaw = contentObj.summary;
    const summaryObj =
      summaryRaw && typeof summaryRaw === "object" && !Array.isArray(summaryRaw)
        ? (summaryRaw as Record<string, unknown>)
        : {};
    const cfgRaw = contentObj.cfg;

    details.push({
      id: docId,
      title,
      updatedAt: numberFromUnknown(doc.updated_at),
      createdAt: numberFromUnknown(doc.created_at),
      scenario: typeof contentObj.scenario === "string" ? contentObj.scenario : title,
      pythonScenario: typeof contentObj.python_scenario === "string" ? contentObj.python_scenario : null,
      scope: typeof contentObj.scope === "string" ? contentObj.scope : scope,
      stopped: Boolean(contentObj.stopped),
      rounds: numberFromUnknown(contentObj.rounds),
      summary: {
        successPct: numberFromUnknown(summaryObj.success_pct),
        avgRps: numberFromUnknown(summaryObj.avg_rps),
        avgP95Ms: numberFromUnknown(summaryObj.avg_p95_ms),
        avgP99Ms: numberFromUnknown(summaryObj.avg_p99_ms),
        avgWallS: numberFromUnknown(summaryObj.avg_wall_s),
        totalInputTokens: numberFromUnknown(summaryObj.total_input_tokens),
        totalOutputTokens: numberFromUnknown(summaryObj.total_output_tokens),
        avgPrefillTokS: numberFromUnknown(summaryObj.avg_prefill_tok_s),
        avgGenTokS: numberFromUnknown(summaryObj.avg_gen_tok_s),
        avgTtftMs: numberFromUnknown(summaryObj.avg_ttft_ms),
      },
      fileId: typeof contentObj.file_id === "string" ? contentObj.file_id : null,
      fileName: typeof contentObj.file_name === "string" ? contentObj.file_name : null,
      generatedAt: typeof contentObj.generated_at === "string" ? contentObj.generated_at : null,
      cfg: cfgRaw && typeof cfgRaw === "object" && !Array.isArray(cfgRaw)
        ? (cfgRaw as Record<string, unknown>)
        : {},
    });
  }

  details.sort((a, b) => (b.updatedAt || b.createdAt) - (a.updatedAt || a.createdAt));
  return details;
}

/** Delete documents by ID in parallel. Returns count of successful deletions. */
async function deleteRuns(baseUrl: string, ids: string[]): Promise<number> {
  const results = await Promise.all(
    ids.map((id) =>
      request(baseUrl, "DELETE", `/v1/db/tables/documents/${encodeURIComponent(id)}`)
        .then((r) => r.ok)
        .catch(() => false),
    ),
  );
  return results.filter(Boolean).length;
}

function wireResultsUi(): void {
  const browser = document.getElementById("bench-results-browser");
  if (!browser) {
    return;
  }
  const scopePrefix = browser.getAttribute("data-results-scope") ?? "";
  const listEl = document.getElementById("bench-results-list");
  const countEl = document.getElementById("bench-results-count");
  const detailEl = document.getElementById("bench-results-detail");
  const searchInput = document.getElementById("bench-results-search") as HTMLInputElement | null;
  const refreshBtn = document.getElementById("bench-results-refresh") as HTMLButtonElement | null;
  const clearFilterBtn = document.getElementById("bench-results-clear-filter") as HTMLButtonElement | null;
  if (!listEl || !countEl || !detailEl) {
    return;
  }

  const baseUrl = window.location.origin.replace(/\/$/, "");
  let runs: PersistedBenchRun[] = [];
  let activeId: string | null = null;
  let searchText = "";
  const collapsedGroups = new Set<string>();
  let initialCollapseApplied = false;

  const filteredRuns = (): PersistedBenchRun[] => {
    const query = searchText.trim().toLowerCase();
    if (!query) {
      return runs;
    }
    return runs.filter((run) => {
      const hay = `${run.scenario} ${run.pythonScenario ?? ""} ${run.scope} ${run.title}`.toLowerCase();
      return hay.includes(query);
    });
  };

  const showDetail = (run: PersistedBenchRun): void => {
    const cfgRows = Object.entries(run.cfg)
      .map(([key, value]) => `<div><span>${escapeHtml(key)}</span><span>${escapeHtml(String(value))}</span></div>`)
      .join("");
    detailEl.innerHTML = `
      <div class="bench-results-detail-head">
        <h3>${escapeHtml(run.title)}</h3>
        <div class="bench-results-detail-actions">
          <button class="bench-rerun-btn btn btn-ghost btn-sm" data-run-id="${run.id}">Re-run</button>
          ${run.fileId ? `<a class="btn btn-ghost btn-sm" href="/v1/files/${encodeURIComponent(run.fileId)}/content" target="_blank" rel="noopener noreferrer">JSONL</a>` : ""}
          <button class="bench-delete-run-btn btn btn-ghost btn-sm">Delete</button>
        </div>
      </div>
      <div class="bench-results-meta-grid">
        <div><span>Scenario</span><span>${escapeHtml(run.scenario)}</span></div>
        <div><span>Scope</span><span>${escapeHtml(run.scope)}</span></div>
        <div><span>Status</span><span>${run.stopped ? "Stopped" : "Complete"}</span></div>
        <div><span>Timestamp</span><span>${escapeHtml(formatRunTimestamp(run.updatedAt || run.createdAt))}</span></div>
        <div><span>Success %</span><span>${round2(run.summary.successPct)}</span></div>
        <div><span>RPS avg</span><span>${round2(run.summary.avgRps)}</span></div>
        <div><span>p95 ms</span><span>${round2(run.summary.avgP95Ms)}</span></div>
        <div><span>p99 ms</span><span>${round2(run.summary.avgP99Ms)}</span></div>
        <div><span>Wall s</span><span>${round3(run.summary.avgWallS)}</span></div>
        <div><span>Input tok</span><span>${Math.round(run.summary.totalInputTokens)}</span></div>
        <div><span>Output tok</span><span>${Math.round(run.summary.totalOutputTokens)}</span></div>
        <div><span>Prefill t/s</span><span>${round2(run.summary.avgPrefillTokS)}</span></div>
        <div><span>Gen t/s</span><span>${round2(run.summary.avgGenTokS)}</span></div>
        <div><span>TTFT ms</span><span>${round2(run.summary.avgTtftMs)}</span></div>
      </div>
      ${cfgRows.length > 0 ? `<div class="bench-results-cfg">${cfgRows}</div>` : ""}
    `;
    detailEl.querySelector<HTMLButtonElement>(".bench-rerun-btn")?.addEventListener("click", () => {
      const targetSlug = toScenarioPage(run.scope);
      if (!targetSlug) return;
      writeRerunSeed(run);
      window.location.href = toBenchPath(targetSlug);
    });
    detailEl.querySelector<HTMLButtonElement>(".bench-delete-run-btn")?.addEventListener("click", async () => {
      if (!window.confirm(`Delete run "${run.title}"?`)) return;
      const deleted = await deleteRuns(baseUrl, [run.id]);
      if (deleted > 0) {
        runs = runs.filter((r) => r.id !== run.id);
        activeId = null;
        renderList();
        renderDetail();
      }
    });
  };

  const renderDetail = (): void => {
    if (!activeId) {
      detailEl.innerHTML = '<p class="bench-results-placeholder">Select a run to view details.</p>';
      return;
    }
    const run = runs.find((item) => item.id === activeId);
    if (!run) {
      detailEl.innerHTML = '<p class="bench-results-placeholder">Select a run to view details.</p>';
      return;
    }
    showDetail(run);
  };

  const renderList = (): void => {
    const visibleRuns = filteredRuns();
    countEl.textContent =
      visibleRuns.length === runs.length
        ? String(visibleRuns.length)
        : `${visibleRuns.length}/${runs.length}`;
    if (visibleRuns.length === 0) {
      listEl.innerHTML = runs.length
        ? '<div class="bench-results-empty">No runs match the current filter.</div>'
        : '<div class="bench-results-empty">No saved runs yet.</div>';
      detailEl.innerHTML = '<p class="bench-results-placeholder">No runs to display.</p>';
      return;
    }
    if (activeId && !visibleRuns.some((item) => item.id === activeId)) {
      activeId = null;
    }

    // Group by scope (benchmark category).
    const groupMap = new Map<string, PersistedBenchRun[]>();
    for (const run of visibleRuns) {
      const key = run.scope || "other";
      let group = groupMap.get(key);
      if (!group) {
        group = [];
        groupMap.set(key, group);
      }
      group.push(run);
    }

    // Sort groups by most recent activity (like chat sidebar).
    const groupMaxUpdated = new Map<string, number>();
    for (const [key, items] of groupMap) {
      groupMaxUpdated.set(key, Math.max(...items.map((r) => r.updatedAt || r.createdAt)));
    }
    const sortedKeys = [...groupMap.keys()].sort((a, b) => {
      return (groupMaxUpdated.get(b) ?? 0) - (groupMaxUpdated.get(a) ?? 0);
    });

    const multiGroup = sortedKeys.length > 1;

    // Collapse all groups by default except the most recent.
    if (!initialCollapseApplied && multiGroup) {
      for (const key of sortedKeys.slice(1)) {
        collapsedGroups.add(key);
      }
      initialCollapseApplied = true;
    }

    // Build DOM (same pattern as chat sidebar-list.ts).
    listEl.innerHTML = "";

    for (const scope of sortedKeys) {
      const items = groupMap.get(scope)!;
      const isOpen = !collapsedGroups.has(scope);

      // -- Group header: [chevron] name [time] --
      const label = document.createElement("div");
      label.className = "sidebar-group-label";

      if (multiGroup) {
        const chevron = document.createElement("button");
        chevron.className = "sidebar-group-collapse";
        chevron.innerHTML = isOpen ? CHEVRON_DOWN_ICON : CHEVRON_RIGHT_ICON;
        chevron.title = isOpen ? "Collapse" : "Expand";
        chevron.addEventListener("click", (e) => {
          e.stopPropagation();
          if (isOpen) {
            collapsedGroups.add(scope);
          } else {
            collapsedGroups.delete(scope);
          }
          renderList();
        });
        label.appendChild(chevron);
      }

      const nameSpan = document.createElement("span");
      nameSpan.className = "sidebar-group-name";
      nameSpan.textContent = scope;
      label.appendChild(nameSpan);

      const maxUpdated = groupMaxUpdated.get(scope);
      if (maxUpdated) {
        const timeSpan = document.createElement("span");
        timeSpan.className = "sidebar-group-time";
        timeSpan.textContent = relativeTime(maxUpdated);
        label.appendChild(timeSpan);
      }

      const clearBtn = document.createElement("button");
      clearBtn.className = "bench-results-group-clear";
      clearBtn.title = `Delete all runs in ${scope}`;
      clearBtn.innerHTML = ICON_DELETE;
      clearBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const ids = items.map((r) => r.id);
        if (!window.confirm(`Delete all ${ids.length} run${ids.length > 1 ? "s" : ""} in "${scope}"?`)) return;
        const deleted = await deleteRuns(baseUrl, ids);
        if (deleted > 0) {
          const deletedSet = new Set(ids);
          runs = runs.filter((r) => !deletedSet.has(r.id));
          if (activeId && deletedSet.has(activeId)) activeId = null;
          renderList();
          renderDetail();
        }
      });
      label.appendChild(clearBtn);

      // Click header to expand if collapsed.
      label.style.cursor = "pointer";
      label.addEventListener("click", () => {
        if (!isOpen) {
          collapsedGroups.delete(scope);
          renderList();
        }
      });

      listEl.appendChild(label);

      // -- Items (skip if collapsed) --
      if (!isOpen) continue;

      for (const run of items) {
        const item = document.createElement("a");
        item.className = "sidebar-item" + (activeId === run.id ? " active" : "");
        item.dataset.runId = run.id;

        const content = document.createElement("div");
        content.className = "sidebar-item-content";

        const titleRow = document.createElement("div");
        titleRow.className = "sidebar-item-title-row";
        if (run.stopped) {
          const dot = document.createElement("span");
          dot.className = "bench-results-stopped-dot";
          titleRow.appendChild(dot);
        }
        const title = document.createElement("span");
        title.className = "sidebar-item-title truncate";
        title.textContent = run.scenario || run.title;
        titleRow.appendChild(title);
        content.appendChild(titleRow);

        const meta = document.createElement("div");
        meta.className = "sidebar-item-meta";
        const ago = document.createElement("span");
        ago.className = "shrink-0";
        ago.textContent = relativeTime(run.updatedAt || run.createdAt);
        meta.appendChild(ago);
        content.appendChild(meta);

        if (run.summary.avgGenTokS > 0 || run.summary.avgTtftMs > 0) {
          const metrics = document.createElement("div");
          metrics.className = "sidebar-item-metrics";
          const parts: string[] = [];
          if (run.summary.avgGenTokS > 0) parts.push(`${round2(run.summary.avgGenTokS)} t/s`);
          if (run.summary.avgTtftMs > 0) parts.push(`${round2(run.summary.avgTtftMs)}ms TTFT`);
          if (run.summary.avgPrefillTokS > 0) parts.push(`${round2(run.summary.avgPrefillTokS)} pp/s`);
          metrics.textContent = parts.join(" \u00b7 ");
          content.appendChild(metrics);
        }

        item.appendChild(content);

        item.addEventListener("click", (e) => {
          e.preventDefault();
          activeId = run.id;
          renderList();
          renderDetail();
        });

        listEl.appendChild(item);
      }
    }
  };

  const refresh = async (): Promise<void> => {
    if (refreshBtn) refreshBtn.disabled = true;
    if (clearFilterBtn) clearFilterBtn.disabled = true;
    listEl.innerHTML = '<div class="bench-results-empty">Loading saved runs…</div>';
    try {
      runs = await loadPersistedBenchRuns(baseUrl, scopePrefix);
      renderList();
      renderDetail();
    } catch (_err) {
      runs = [];
      listEl.innerHTML = '<div class="bench-results-empty">Unable to load saved runs.</div>';
      detailEl.innerHTML = '<p class="bench-results-placeholder">Failed to load run history.</p>';
    }
    if (refreshBtn) refreshBtn.disabled = false;
    if (clearFilterBtn) clearFilterBtn.disabled = false;
  };

  refreshBtn?.addEventListener("click", () => {
    refresh();
  });
  searchInput?.addEventListener("input", () => {
    searchText = searchInput.value;
    renderList();
    renderDetail();
  });
  clearFilterBtn?.addEventListener("click", () => {
    if (searchInput) {
      searchInput.value = "";
    }
    searchText = "";
    renderList();
    renderDetail();
  });

  const deleteAllBtn = document.getElementById("bench-results-delete-all") as HTMLButtonElement | null;
  deleteAllBtn?.addEventListener("click", async () => {
    if (runs.length === 0) return;
    if (!window.confirm(`Delete all ${runs.length} saved run${runs.length > 1 ? "s" : ""}? This cannot be undone.`)) return;
    deleteAllBtn.disabled = true;
    const ids = runs.map((r) => r.id);
    await deleteRuns(baseUrl, ids);
    runs = [];
    activeId = null;
    renderList();
    renderDetail();
    deleteAllBtn.disabled = false;
  });

  refresh();
}

function wireSidebarUi(): void {
  document.querySelectorAll<HTMLButtonElement>(".bench-tree-toggle[data-tree-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const slug = button.dataset.treeToggle ?? "";
      if (!slug) return;
      const target = document.querySelector<HTMLElement>(`.bench-tree-children[data-tree-node="${slug}"]`);
      if (!target) return;
      const open = !target.classList.contains("is-open");
      target.classList.toggle("is-open", open);
      button.classList.toggle("is-open", open);
      button.setAttribute("aria-expanded", open ? "true" : "false");
      setBenchTreeNodeOpen(slug, open);
    });
  });

  document.querySelectorAll<HTMLButtonElement>(".bench-tree-scenario-nav").forEach((button) => {
    button.addEventListener("click", () => {
      const scenarioId = button.dataset.navScenarioId ?? "";
      const page = button.dataset.navScenarioPage ?? "";
      if (!scenarioId || !page) return;

      const activeScenarioButton = Array.from(
        document.querySelectorAll<HTMLButtonElement>(".bench-scenario-item"),
      ).find((candidate) => candidate.dataset.scenarioId === scenarioId);
      if (activeScenarioButton) {
        activeScenarioButton.click();
        return;
      }

      writePendingScenarioSelection(page, scenarioId);
      window.location.href = toBenchPath(page);
    });
  });
}

function wireUi(activeScenarios: BenchScenario[]): void {
  const runBtn = document.getElementById("bench-run") as HTMLButtonElement | null;
  const pauseBtn = document.getElementById("bench-pause") as HTMLButtonElement | null;
  const clearBtn = document.getElementById("bench-clear") as HTMLButtonElement | null;
  const settingsToggleBtn = document.getElementById("bench-settings-toggle") as HTMLButtonElement | null;
  const settingsPanel = document.getElementById("bench-settings-panel");
  const eventsClearBtn = document.getElementById("bench-events-clear") as HTMLButtonElement | null;
  const eventsCopyBtn = document.getElementById("bench-events-copy") as HTMLButtonElement | null;
  if (!runBtn) {
    return;
  }

  let activeControl: RunControlState | null = null;
  let progressState = defaultRunProgressState();
  let settingsOpen = false;

  const setSettingsOpen = (open: boolean): void => {
    settingsOpen = open;
    if (settingsPanel) settingsPanel.style.display = open ? "" : "none";
    settingsToggleBtn?.classList.toggle("is-active", open);
  };

  const updateRunButtons = (): void => {
    const running = activeControl !== null;
    runBtn.disabled = running || !selectedScenarioId();
    runBtn.textContent = running ? "Running..." : "Start";
    scenarioButtons.forEach((button) => {
      button.disabled = running;
    });
    if (pauseBtn) {
      pauseBtn.disabled = !running;
      pauseBtn.textContent = running && activeControl?.paused ? "Resume" : "Pause";
    }
  };

  const scenarioById = new Map(activeScenarios.map((scenario) => [scenario.id, scenario]));
  const scenarioButtons = Array.from(
    document.querySelectorAll<HTMLButtonElement>(".bench-scenario-item"),
  );
  const selectScenario = (id: string | null): BenchScenario | null => {
    if (!id) return null;
    return scenarioById.get(id) ?? null;
  };
  const syncProgress = (patch?: Partial<RunProgressState>): void => {
    progressState = patch ? { ...progressState, ...patch } : defaultRunProgressState();
    setRunProgressUi(progressState);
  };

  const setInputValue = (id: string, value: unknown): void => {
    const input = document.getElementById(id) as HTMLInputElement | null;
    if (!input) return;
    input.value = String(value);
  };

  const clearRunOutput = (): void => {
    const tableBody = document.getElementById("bench-results-body");
    if (tableBody) {
      tableBody.innerHTML = "";
    }
    const logEl = document.getElementById("bench-log");
    if (logEl) {
      logEl.textContent = "";
    }
    clearBenchEventsLog();
    clearSummaryUi();
    syncProgress();
    setRunStatus("Idle", "idle");
  };

  const applyConfigToForm = (cfg: Partial<RunConfig>): void => {
    if (cfg.requests !== undefined) setInputValue("bench-requests", cfg.requests);
    if (cfg.concurrency !== undefined) setInputValue("bench-concurrency", cfg.concurrency);
    if (cfg.rounds !== undefined) setInputValue("bench-rounds", cfg.rounds);
    if (cfg.kvBatchSize !== undefined) setInputValue("bench-kv-batch-size", cfg.kvBatchSize);
    if (cfg.tableSeedRows !== undefined) setInputValue("bench-table-seed-rows", cfg.tableSeedRows);
    if (cfg.vectorDims !== undefined) setInputValue("bench-vector-dims", cfg.vectorDims);
    if (cfg.responsesModel !== undefined) {
      // If the model matches a known variant, select the family and variant.
      const variantEntry = benchAvailableModels.find((m) => m.variants?.some((v) => v.id === cfg.responsesModel));
      if (variantEntry) {
        setInputValue("bench-responses-model", variantEntry.id);
        benchSelectedVariant = cfg.responsesModel;
      } else {
        setInputValue("bench-responses-model", cfg.responsesModel);
        benchSelectedVariant = cfg.responsesModel ?? "";
      }
      renderBenchModelVariants(variantEntry ?? benchAvailableModels.find((m) => m.id === cfg.responsesModel));
    }
    if (cfg.responsesMaxOutputTokens !== undefined) {
      setInputValue("bench-responses-max-output-tokens", cfg.responsesMaxOutputTokens);
    }
  };

  const applyScenarioDefaults = (scenario: BenchScenario): void => {
    const baseline: Partial<RunConfig> = {
      requests: 1,
      concurrency: 1,
      rounds: 1,
      kvBatchSize: 1,
      tableSeedRows: 1,
      vectorDims: 8,
      responsesModel: "",
      responsesMaxOutputTokens: 1,
    };
    applyConfigToForm({ ...baseline, ...(scenario.defaults ?? {}) });
  };

  const refreshActiveScenario = (opts?: { applyDefaults?: boolean }): BenchScenario | null => {
    const selectedId = selectedScenarioId();
    const scenario = selectScenario(selectedId);
    if (scenario && opts?.applyDefaults !== false) {
      applyScenarioDefaults(scenario);
    }
    updateActiveScenarioUi(scenario);
    updateRunButtons();
    return scenario;
  };

  scenarioButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (activeControl) {
        return;
      }
      scenarioButtons.forEach((item) => item.classList.remove("is-active"));
      button.classList.add("is-active");
      const selected = selectScenario(button.dataset.scenarioId ?? null);
      if (selected) {
        applyScenarioDefaults(selected);
      }
      updateActiveScenarioUi(selected);
      updateRunButtons();
      clearRunOutput();
      if (selected) {
        setRunStatus(`Ready · ${selected.id}`, "idle");
      }
    });
  });
  settingsToggleBtn?.addEventListener("click", () => {
    setSettingsOpen(!settingsOpen);
  });
  eventsClearBtn?.addEventListener("click", () => {
    clearBenchEventsLog();
  });
  eventsCopyBtn?.addEventListener("click", async () => {
    await copyBenchEventsLog();
  });
  document.getElementById("bench-log-copy")?.addEventListener("click", async () => {
    const el = document.getElementById("bench-log");
    const text = el?.textContent?.trim() ?? "";
    if (text) {
      try { await writeClipboardText(text); } catch (_) { /* noop */ }
    }
  });
  wireBenchEventLevelFilters();
  wireBenchEventTopicFilters();
  wireBenchEventSearch();

  // Wire model select + variant pills.
  const modelSelect = document.getElementById("bench-responses-model") as HTMLSelectElement | null;
  modelSelect?.addEventListener("change", () => {
    const entry = benchAvailableModels.find((m) => m.id === modelSelect.value);
    benchSelectedVariant = entry?.variants?.[0]?.id ?? modelSelect.value;
    renderBenchModelVariants(entry);
  });
  fetchBenchModels();

  pauseBtn?.addEventListener("click", () => {
    if (!activeControl) {
      return;
    }
    activeControl.paused = !activeControl.paused;
    updateRunButtons();
    if (activeControl.paused) {
      appendLog("pause requested: no new requests will be dispatched");
      syncProgress({ phase: "Paused", note: "Dispatch paused. In-flight requests can still complete.", tone: "idle" });
      setRunStatus("Paused", "idle");
    } else {
      appendLog("resumed");
      syncProgress({ phase: "Running", note: "Dispatch resumed for the active round.", tone: "running" });
      setRunStatus("Running", "running");
    }
  });

  clearBtn?.addEventListener("click", () => {
    if (activeControl) {
      activeControl.stopRequested = true;
      activeControl.paused = false;
      updateRunButtons();
      appendLog("clear requested while running: stopping active scenario first");
      syncProgress({
        phase: "Stopping",
        note: "Clear requested. Waiting for in-flight requests to finish.",
        tone: "error",
      });
      setRunStatus("Stopping...", "error");
      return;
    }
    clearRunOutput();
    appendLog("cleared run output");
  });

  if (
    scenarioButtons.length > 0 &&
    !scenarioButtons.some((button) => button.classList.contains("is-active"))
  ) {
    scenarioButtons[0]!.classList.add("is-active");
  }
  refreshActiveScenario({ applyDefaults: true });
  const rerunSeed = readRerunSeed();
  if (rerunSeed) {
    const seedScenario = selectScenario(rerunSeed.scenario);
    if (seedScenario) {
      const targetButton = scenarioButtons.find(
        (button) => button.dataset.scenarioId === rerunSeed.scenario,
      );
      if (targetButton) {
        scenarioButtons.forEach((button) => button.classList.remove("is-active"));
        targetButton.classList.add("is-active");
      }
      applyConfigToForm(rerunSeed.cfg as Partial<RunConfig>);
      updateActiveScenarioUi(seedScenario);
      appendLog(`loaded rerun seed for scenario=${rerunSeed.scenario}`);
    }
    clearRerunSeed();
  }
  const pendingScenario = readPendingScenarioSelection();
  if (pendingScenario) {
    const pendingButton = scenarioButtons.find(
      (button) => button.dataset.scenarioId === pendingScenario.scenarioId,
    );
    if (pendingButton) {
      scenarioButtons.forEach((button) => button.classList.remove("is-active"));
      pendingButton.classList.add("is-active");
      const selected = selectScenario(pendingScenario.scenarioId);
      if (selected) {
        updateActiveScenarioUi(selected);
        setRunStatus(`Ready · ${selected.id}`, "idle");
      }
    }
    clearPendingScenarioSelection();
  }
  clearRunOutput();
  updateRunButtons();

  runBtn.addEventListener("click", async () => {
    activeControl = { paused: false, stopRequested: false };
    updateRunButtons();
    try {
      const scenario = refreshActiveScenario({ applyDefaults: false });
      const scenarioResolved = scenario ?? refreshActiveScenario({ applyDefaults: false });
      if (!scenarioResolved) {
        appendLog("no scenario selected");
        activeControl = null;
        updateRunButtons();
        return;
      }
      const scenarioId = scenarioResolved.id;

      const baseUrl = window.location.origin.replace(/\/$/, "");
      clearRunOutput();
      startBenchEventsStream(baseUrl);
      const cfg: RunConfig = {
        requests: parsePositiveInt("bench-requests", 1),
        concurrency: parsePositiveInt("bench-concurrency", 1),
        rounds: parsePositiveInt("bench-rounds", 1),
        kvBatchSize: parsePositiveInt("bench-kv-batch-size", 1),
        tableSeedRows: parsePositiveInt("bench-table-seed-rows", 1),
        vectorDims: parsePositiveInt("bench-vector-dims", 1),
        responsesModel: benchSelectedVariant || parseString("bench-responses-model", ""),
        responsesMaxOutputTokens: parsePositiveInt("bench-responses-max-output-tokens", 1),
      };
      if (scenarioResolved.scope.startsWith("responses/")) {
        cfg.responsesModel = await resolveResponsesModel(baseUrl, cfg.responsesModel);
      }

      syncProgress({
        scenarioId,
        round: 0,
        totalRounds: cfg.rounds,
        completedRequests: 0,
        totalRequests: cfg.requests,
        phase: "Queued",
        note: `Starting ${scenarioId} with ${cfg.rounds} round${cfg.rounds === 1 ? "" : "s"}.`,
        tone: "running",
      });
      setRunStatus(`Running ${scenarioId}...`, "running");
      const runStartedAt = performance.now();
      appendLog(
        `starting scenario id=${scenarioId} base=${baseUrl} requests=${cfg.requests} concurrency=${cfg.concurrency} rounds=${cfg.rounds} model=${cfg.responsesModel}`,
      );
      const runResult = await runScenario(scenarioResolved, cfg, baseUrl, activeControl, (progress) => {
        progressState = progress;
        setRunProgressUi(progressState);
      });
      const elapsedMs = performance.now() - runStartedAt;
      if (elapsedMs < 420) {
        await new Promise((resolve) => setTimeout(resolve, Math.ceil(420 - elapsedMs)));
      }
      updateSummaryUi(runResult.rounds);
      const totalExecuted = runResult.rounds.reduce(
        (sum, item) => sum + item.metrics.requests,
        0,
      );
      const totalOk = runResult.rounds.reduce((sum, item) => sum + item.metrics.ok, 0);
      const totalErrors = runResult.rounds.reduce((sum, item) => sum + item.metrics.errors, 0);
      if (!runResult.stopped && runResult.rounds.length === 0) {
        setRunStatus(`No successful rounds · ${scenarioId}`, "error");
        appendLog("no successful rounds recorded (check server logs / endpoint support)");
      } else if (!runResult.stopped && totalExecuted <= 1) {
        appendLog("quick smoke run completed (increase requests/rounds for meaningful perf numbers)");
      }
      if (runResult.stopped) {
        appendLog("scenario stopped by operator");
        syncProgress({
          phase: "Stopped",
          completedRequests: progressState.totalRequests,
          note: `Stopped ${scenarioId} after ${runResult.rounds.length} completed round${runResult.rounds.length === 1 ? "" : "s"}.`,
          tone: "error",
        });
        setRunStatus(`Stopped · ${scenarioId}`, "idle");
      } else if (totalExecuted > 0 && totalOk === 0 && totalErrors > 0) {
        appendLog("scenario finished with all requests failing");
        syncProgress({
          phase: "Failed",
          round: cfg.rounds,
          totalRounds: cfg.rounds,
          completedRequests: totalExecuted,
          totalRequests: totalExecuted,
          note: `0 of ${totalExecuted} requests succeeded. Check endpoint availability and server logs.`,
          tone: "error",
        });
        setRunStatus(`Failed · ${scenarioId} (0/${totalExecuted} OK)`, "error");
      } else if (totalErrors > 0) {
        appendLog(`scenario completed with partial errors ok=${totalOk} errors=${totalErrors}`);
        syncProgress({
          phase: "Complete with errors",
          round: cfg.rounds,
          totalRounds: cfg.rounds,
          completedRequests: totalExecuted,
          totalRequests: totalExecuted,
          note: `${totalOk} OK, ${totalErrors} failed. Review logs before trusting these numbers.`,
          tone: "error",
        });
        setRunStatus(`Complete with errors · ${scenarioId}`, "error");
      } else {
        appendLog("scenario complete");
        syncProgress({
          phase: "Complete",
          round: cfg.rounds,
          totalRounds: cfg.rounds,
          completedRequests: cfg.requests,
          totalRequests: cfg.requests,
          note: `All ${cfg.rounds} round${cfg.rounds === 1 ? "" : "s"} finished for ${scenarioId}.`,
          tone: "ok",
        });
        setRunStatus(`Complete · ${scenarioId}`, "ok");
      }

      if (runResult.rounds.length > 0) {
        const persisted = await persistBenchRun(baseUrl, scenarioResolved, cfg, runResult.rounds, runResult.stopped);
        if (persisted.fileId || persisted.docId) {
          appendLog(
            `persisted results file_id=${persisted.fileId ?? "<none>"} doc_id=${persisted.docId ?? "<none>"}`,
          );
        } else {
          appendLog("persistence skipped: no storage artifact returned");
        }
      }
    } catch (err) {
      syncProgress({
        phase: "Failed",
        note: err instanceof Error ? err.message : String(err),
        tone: "error",
      });
      setRunStatus("Failed", "error");
      appendLog(`run failed=${err instanceof Error ? err.message : String(err)}`);
    } finally {
      stopBenchEventsStream();
      activeControl = null;
      updateRunButtons();
    }
  });
}

export function inspectBenchPage(pathname: string): {
  status: "ok" | "not_found";
  slug: string;
  childSlugs: string[];
  scenarioIds: string[];
} {
  const resolved = resolveBenchPage(pathname);
  if (resolved.status !== "ok") {
    return { status: "not_found", slug: resolved.requestedSlug, childSlugs: [], scenarioIds: [] };
  }
  const scenarios = scenariosForPage(resolved.page).map((scenario) => scenario.id);
  return {
    status: "ok",
    slug: resolved.page.slug,
    childSlugs: resolved.page.children,
    scenarioIds: scenarios,
  };
}

let benchRouteListenerBound = false;

function mountBenchApp(logReady: boolean): void {
  stopBenchEventsStream();
  const { scenarios } = render();
  wireSidebarUi();
  wireUi(scenarios);
  wireResultsUi();
  if (logReady) {
    appendLog("bench runner ready");
  }
}

let benchActiveThemeClass = "";

/** Sync the theme class from localStorage to this document's :root.
 *  The bench runs in an iframe — the parent sets the class on its own :root,
 *  but the iframe has a separate document that needs the same class. */
function syncBenchTheme(): void {
  const theme = localStorage.getItem("theme") || "talu";
  if (theme === benchActiveThemeClass) return;
  const root = document.documentElement;
  if (benchActiveThemeClass) root.classList.remove(benchActiveThemeClass);
  root.classList.add(theme);
  benchActiveThemeClass = theme;
}

export function bootBenchApp(): void {
  syncBenchTheme();
  window.addEventListener("storage", (e) => {
    if (e.key === "theme") syncBenchTheme();
  });
  mountBenchApp(true);
  if (!benchRouteListenerBound) {
    window.addEventListener("hashchange", () => {
      mountBenchApp(false);
    });
    benchRouteListenerBound = true;
  }
}
