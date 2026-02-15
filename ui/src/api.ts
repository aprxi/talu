import type { ApiResult, Conversation, ConversationList, ConversationPatch, ForkRequest, CreateResponseRequest, Settings, SettingsPatch, SearchRequest, SearchResponse, BatchRequest, Document, DocumentList, CreateDocumentRequest, UpdateDocumentRequest, FileObject, FileInspection } from "./types.ts";

const BASE = "";

// ---------------------------------------------------------------------------
// Factory — creates an API client bound to a specific fetch implementation.
// Plugins use this with ctx.network.fetch for attributed requests.
// ---------------------------------------------------------------------------

type FetchFn = (url: string, init?: RequestInit) => Promise<Response>;

export interface ApiClient {
  listConversations(cursor?: string | null, limit?: number): Promise<ApiResult<ConversationList>>;
  search(req: SearchRequest): Promise<ApiResult<SearchResponse>>;
  getConversation(id: string): Promise<ApiResult<Conversation>>;
  patchConversation(id: string, patch: ConversationPatch): Promise<ApiResult<Conversation>>;
  deleteConversation(id: string): Promise<ApiResult<void>>;
  batchConversations(req: BatchRequest): Promise<ApiResult<void>>;
  forkConversation(id: string, body: ForkRequest): Promise<ApiResult<Conversation>>;
  getSettings(): Promise<ApiResult<Settings>>;
  patchSettings(patch: SettingsPatch): Promise<ApiResult<Settings>>;
  resetModelOverrides(modelId: string): Promise<ApiResult<Settings>>;
  createResponse(body: CreateResponseRequest, signal?: AbortSignal): Promise<Response>;
  listDocuments(type?: string): Promise<ApiResult<DocumentList>>;
  getDocument(id: string): Promise<ApiResult<Document>>;
  createDocument(doc: CreateDocumentRequest): Promise<ApiResult<Document>>;
  updateDocument(id: string, doc: UpdateDocumentRequest): Promise<ApiResult<Document>>;
  deleteDocument(id: string): Promise<ApiResult<void>>;
  uploadFile(file: File, purpose?: string): Promise<ApiResult<FileObject>>;
  getFile(id: string): Promise<ApiResult<FileObject>>;
  deleteFile(id: string): Promise<ApiResult<void>>;
  getFileContent(id: string): Promise<ApiResult<Blob>>;
  inspectFile(file: File): Promise<ApiResult<FileInspection>>;
  transformFile(file: File, opts?: { resize?: string; fit?: string; format?: string; quality?: number }): Promise<ApiResult<Blob>>;
}

export function createApiClient(fetchFn: FetchFn): ApiClient {
  async function parseErrorMessage(resp: Response): Promise<string> {
    const err = await resp.json().catch(() => null);
    return err?.error?.message ?? `${resp.status} ${resp.statusText}`;
  }

  async function requestJson<T>(method: string, path: string, body?: unknown): Promise<ApiResult<T>> {
    try {
      const opts: RequestInit = {
        method,
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      };
      const resp = await fetchFn(`${BASE}${path}`, opts);

      if (resp.status === 204) {
        return { ok: true };
      }

      if (!resp.ok) {
        const msg = await parseErrorMessage(resp);
        return { ok: false, error: msg };
      }

      const data = (await resp.json()) as T;
      return { ok: true, data };
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) };
    }
  }

  async function requestFormData<T>(method: string, path: string, body: FormData): Promise<ApiResult<T>> {
    try {
      const resp = await fetchFn(`${BASE}${path}`, { method, body });

      if (resp.status === 204) {
        return { ok: true };
      }

      if (!resp.ok) {
        const msg = await parseErrorMessage(resp);
        return { ok: false, error: msg };
      }

      const data = (await resp.json()) as T;
      return { ok: true, data };
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) };
    }
  }

  async function requestBlob(method: string, path: string): Promise<ApiResult<Blob>> {
    try {
      const resp = await fetchFn(`${BASE}${path}`, { method });

      if (!resp.ok) {
        const msg = await parseErrorMessage(resp);
        return { ok: false, error: msg };
      }

      const data = await resp.blob();
      return { ok: true, data };
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) };
    }
  }

  return {
    listConversations(cursor?: string | null, limit = 20) {
      const params = new URLSearchParams({ limit: String(limit) });
      if (cursor) params.set("cursor", cursor);
      return requestJson<ConversationList>("GET", `/v1/conversations?${params}`);
    },
    search: (req) => requestJson<SearchResponse>("POST", "/v1/search", req),
    getConversation: (id) => requestJson<Conversation>("GET", `/v1/conversations/${encodeURIComponent(id)}`),
    patchConversation: (id, patch) => requestJson<Conversation>("PATCH", `/v1/conversations/${encodeURIComponent(id)}`, patch),
    deleteConversation: (id) => requestJson<void>("DELETE", `/v1/conversations/${encodeURIComponent(id)}`),
    batchConversations: (req) => requestJson<void>("POST", "/v1/conversations/batch", req),
    forkConversation: (id, body) => requestJson<Conversation>("POST", `/v1/conversations/${encodeURIComponent(id)}/fork`, body),
    getSettings: () => requestJson<Settings>("GET", "/v1/settings"),
    patchSettings: (patch) => requestJson<Settings>("PATCH", "/v1/settings", patch),
    resetModelOverrides: (modelId) => requestJson<Settings>("DELETE", `/v1/settings/models/${modelId}`),
    createResponse: (body, signal) => fetchFn(`${BASE}/v1/responses`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, stream: true, store: true }),
      signal,
    }),
    listDocuments(type?: string) {
      const params = new URLSearchParams();
      if (type) params.set("type", type);
      const query = params.toString();
      return requestJson<DocumentList>("GET", `/v1/documents${query ? `?${query}` : ""}`);
    },
    getDocument: (id) => requestJson<Document>("GET", `/v1/documents/${encodeURIComponent(id)}`),
    createDocument: (doc) => requestJson<Document>("POST", "/v1/documents", doc),
    updateDocument: (id, doc) => requestJson<Document>("PATCH", `/v1/documents/${encodeURIComponent(id)}`, doc),
    deleteDocument: (id) => requestJson<void>("DELETE", `/v1/documents/${encodeURIComponent(id)}`),
    uploadFile: (file, purpose = "assistants") => {
      const form = new FormData();
      form.append("file", file, file.name);
      form.append("purpose", purpose);
      return requestFormData<FileObject>("POST", "/v1/files", form);
    },
    getFile: (id) => requestJson<FileObject>("GET", `/v1/files/${encodeURIComponent(id)}`),
    deleteFile: (id) => requestJson<void>("DELETE", `/v1/files/${encodeURIComponent(id)}`),
    getFileContent: (id) => requestBlob("GET", `/v1/files/${encodeURIComponent(id)}/content`),
    inspectFile: (file) => {
      const form = new FormData();
      form.append("file", file, file.name);
      return requestFormData<FileInspection>("POST", "/v1/file/inspect", form);
    },
    transformFile: async (file, opts = {}) => {
      try {
        const form = new FormData();
        form.append("file", file, file.name);
        if (opts.resize) form.append("resize", opts.resize);
        if (opts.fit) form.append("fit", opts.fit);
        if (opts.format) form.append("format", opts.format);
        if (opts.quality !== undefined) form.append("quality", String(opts.quality));

        const resp = await fetchFn(`${BASE}/v1/file/transform`, { method: "POST", body: form });
        if (!resp.ok) {
          const msg = await parseErrorMessage(resp);
          return { ok: false, error: msg };
        }
        const data = await resp.blob();
        return { ok: true, data };
      } catch (e) {
        return { ok: false, error: e instanceof Error ? e.message : String(e) };
      }
    },
  };
}
