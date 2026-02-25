import type { ApiResult, Conversation, ConversationList, ConversationPatch, ConversationTag, ForkRequest, CreateResponseRequest, Settings, SettingsPatch, SearchRequest, SearchResponse, BatchRequest, Document, DocumentList, CreateDocumentRequest, UpdateDocumentRequest, FileObject, FileList, FileBatchRequest, FileInspection, RepoModelList, RepoSearchResponse, Project, ProjectList } from "./types.ts";

const BASE = "";

// ---------------------------------------------------------------------------
// Factory — creates an API client bound to a specific fetch implementation.
// Plugins use this with ctx.network.fetch for attributed requests.
// ---------------------------------------------------------------------------

type FetchFn = (url: string, init?: RequestInit) => Promise<Response>;

export interface ApiClient {
  listConversations(opts?: { offset?: number; limit?: number; marker?: string; search?: string; tags_any?: string; project_id?: string }): Promise<ApiResult<ConversationList>>;
  search(req: SearchRequest): Promise<ApiResult<SearchResponse>>;
  getConversation(id: string): Promise<ApiResult<Conversation>>;
  patchConversation(id: string, patch: ConversationPatch): Promise<ApiResult<Conversation>>;
  deleteConversation(id: string): Promise<ApiResult<void>>;
  batchConversations(req: BatchRequest): Promise<ApiResult<void>>;
  forkConversation(id: string, body: ForkRequest): Promise<ApiResult<Conversation>>;
  addConversationTags(id: string, tags: string[]): Promise<ApiResult<{ tags: ConversationTag[] }>>;
  removeConversationTags(id: string, tags: string[]): Promise<ApiResult<{ tags: ConversationTag[] }>>;
  getSettings(): Promise<ApiResult<Settings>>;
  patchSettings(patch: SettingsPatch): Promise<ApiResult<Settings>>;
  resetModelOverrides(modelId: string): Promise<ApiResult<Settings>>;
  createResponse(body: CreateResponseRequest, signal?: AbortSignal): Promise<Response>;
  streamEvents(
    opts: {
      verbosity?: 1 | 2 | 3;
      domains?: string;
      topics?: string;
      event_class?: string;
      response_id?: string;
      session_id?: string;
      cursor?: string;
    },
    signal?: AbortSignal,
  ): Promise<Response>;
  listDocuments(type?: string): Promise<ApiResult<DocumentList>>;
  getDocument(id: string): Promise<ApiResult<Document>>;
  createDocument(doc: CreateDocumentRequest): Promise<ApiResult<Document>>;
  updateDocument(id: string, doc: UpdateDocumentRequest): Promise<ApiResult<Document>>;
  deleteDocument(id: string): Promise<ApiResult<void>>;
  listFiles(opts?: { limit?: number; marker?: string; offset?: number; sort?: string; order?: string; search?: string }): Promise<ApiResult<FileList>>;
  batchFiles(req: FileBatchRequest): Promise<ApiResult<void>>;
  uploadFile(file: File, purpose?: string): Promise<ApiResult<FileObject>>;
  getFile(id: string): Promise<ApiResult<FileObject>>;
  updateFile(id: string, patch: { filename?: string; marker?: string }): Promise<ApiResult<FileObject>>;
  deleteFile(id: string): Promise<ApiResult<void>>;
  getFileContent(id: string): Promise<ApiResult<Blob>>;
  inspectFile(file: File): Promise<ApiResult<FileInspection>>;
  transformFile(file: File, opts?: { resize?: string; fit?: string; format?: string; quality?: number }): Promise<ApiResult<Blob>>;
  listRepoModels(query?: string): Promise<ApiResult<RepoModelList>>;
  searchRepoModels(query: string, opts?: { sort?: string; filter?: string; library?: string; limit?: number }): Promise<ApiResult<RepoSearchResponse>>;
  fetchRepoModel(body: { model_id: string }, signal?: AbortSignal): Promise<Response>;
  deleteRepoModel(modelId: string): Promise<ApiResult<void>>;
  pinRepoModel(modelId: string): Promise<ApiResult<void>>;
  unpinRepoModel(modelId: string): Promise<ApiResult<void>>;
  listProjects(opts?: { limit?: number; search?: string }): Promise<ApiResult<ProjectList>>;
  createProject(body: { name: string; description?: string }): Promise<ApiResult<Project>>;
  updateProject(id: string, body: { name?: string; description?: string }): Promise<ApiResult<Project>>;
  deleteProject(id: string): Promise<ApiResult<void>>;
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
    listConversations(opts?: { offset?: number; limit?: number; marker?: string; search?: string; tags_any?: string; project_id?: string }) {
      const params = new URLSearchParams({ limit: String(opts?.limit ?? 50) });
      if (opts?.offset !== undefined) params.set("offset", String(opts.offset));
      if (opts?.marker) params.set("marker", opts.marker);
      if (opts?.search) params.set("search", opts.search);
      if (opts?.tags_any) params.set("tags_any", opts.tags_any);
      if (opts?.project_id) params.set("project_id", opts.project_id);
      return requestJson<ConversationList>("GET", `/v1/chat/sessions?${params}`);
    },
    search: (req) => requestJson<SearchResponse>("POST", "/v1/search", req),
    getConversation: (id) => requestJson<Conversation>("GET", `/v1/chat/sessions/${encodeURIComponent(id)}`),
    patchConversation: (id, patch) => requestJson<Conversation>("PATCH", `/v1/chat/sessions/${encodeURIComponent(id)}`, patch),
    deleteConversation: (id) => requestJson<void>("DELETE", `/v1/chat/sessions/${encodeURIComponent(id)}`),
    batchConversations: (req) => requestJson<void>("POST", "/v1/chat/sessions/batch", req),
    forkConversation: (id, body) => requestJson<Conversation>("POST", `/v1/chat/sessions/${encodeURIComponent(id)}/fork`, body),
    addConversationTags: (id, tags) => requestJson<{ tags: ConversationTag[] }>("POST", `/v1/chat/sessions/${encodeURIComponent(id)}/tags`, { tags }),
    removeConversationTags: (id, tags) => requestJson<{ tags: ConversationTag[] }>("DELETE", `/v1/chat/sessions/${encodeURIComponent(id)}/tags`, { tags }),
    getSettings: () => requestJson<Settings>("GET", "/v1/settings"),
    patchSettings: (patch) => requestJson<Settings>("PATCH", "/v1/settings", patch),
    resetModelOverrides: (modelId) => requestJson<Settings>("DELETE", `/v1/settings/models/${modelId}`),
    createResponse: (body, signal) => fetchFn(`${BASE}/v1/responses`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, stream: true, store: true }),
      signal,
    }),
    streamEvents: (opts, signal) => {
      const params = new URLSearchParams();
      if (opts.verbosity) params.set("verbosity", String(opts.verbosity));
      if (opts.domains) params.set("domains", opts.domains);
      if (opts.topics) params.set("topics", opts.topics);
      if (opts.event_class) params.set("event_class", opts.event_class);
      if (opts.response_id) params.set("response_id", opts.response_id);
      if (opts.session_id) params.set("session_id", opts.session_id);
      if (opts.cursor) params.set("cursor", opts.cursor);
      const query = params.toString();
      const path = query ? `/v1/events/stream?${query}` : "/v1/events/stream";
      return fetchFn(`${BASE}${path}`, {
        method: "GET",
        headers: { "Accept": "text/event-stream" },
        signal,
      });
    },
    listDocuments(type?: string) {
      const params = new URLSearchParams();
      if (type) params.set("type", type);
      const query = params.toString();
      return requestJson<DocumentList>("GET", `/v1/db/tables/documents${query ? `?${query}` : ""}`);
    },
    getDocument: (id) => requestJson<Document>("GET", `/v1/db/tables/documents/${encodeURIComponent(id)}`),
    createDocument: (doc) => requestJson<Document>("POST", "/v1/db/tables/documents", doc),
    updateDocument: (id, doc) => requestJson<Document>("PATCH", `/v1/db/tables/documents/${encodeURIComponent(id)}`, doc),
    deleteDocument: (id) => requestJson<void>("DELETE", `/v1/db/tables/documents/${encodeURIComponent(id)}`),
    listFiles: (opts?: { limit?: number; marker?: string; offset?: number; sort?: string; order?: string; search?: string }) => {
      const params = new URLSearchParams();
      params.set("limit", String(opts?.limit ?? 100));
      if (opts?.marker) params.set("marker", opts.marker);
      if (opts?.offset !== undefined) params.set("offset", String(opts.offset));
      if (opts?.sort) params.set("sort", opts.sort);
      if (opts?.order) params.set("order", opts.order);
      if (opts?.search) params.set("search", opts.search);
      return requestJson<FileList>("GET", `/v1/files?${params}`);
    },
    batchFiles: (req) => requestJson<void>("POST", "/v1/files/batch", req),
    uploadFile: (file, purpose = "assistants") => {
      const form = new FormData();
      form.append("file", file, file.name);
      form.append("purpose", purpose);
      return requestFormData<FileObject>("POST", "/v1/files", form);
    },
    getFile: (id) => requestJson<FileObject>("GET", `/v1/files/${encodeURIComponent(id)}`),
    updateFile: (id, patch) =>
      requestJson<FileObject>("PATCH", `/v1/files/${encodeURIComponent(id)}`, patch),
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
    listRepoModels(query?: string) {
      const path = query ? `/v1/repo/models${query}` : "/v1/repo/models";
      return requestJson<RepoModelList>("GET", path);
    },
    searchRepoModels(query: string, opts?: { sort?: string; filter?: string; library?: string; limit?: number }) {
      const params = new URLSearchParams({ query, limit: String(opts?.limit ?? 50) });
      if (opts?.sort) params.set("sort", opts.sort);
      if (opts?.filter) params.set("filter", opts.filter);
      if (opts?.library) params.set("library", opts.library);
      return requestJson<RepoSearchResponse>("GET", `/v1/repo/search?${params}`);
    },
    fetchRepoModel: (body, signal) => fetchFn(`${BASE}/v1/repo/models`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    }),
    deleteRepoModel: (modelId) => requestJson<void>("DELETE", `/v1/repo/models/${encodeURIComponent(modelId)}`),
    pinRepoModel: (modelId) => requestJson<void>("POST", `/v1/repo/pins`, { model_id: modelId }),
    unpinRepoModel: (modelId) => requestJson<void>("DELETE", `/v1/repo/pins/${encodeURIComponent(modelId)}`),
    listProjects(opts?: { limit?: number; search?: string }) {
      const params = new URLSearchParams({ limit: String(opts?.limit ?? 100) });
      if (opts?.search) params.set("search", opts.search);
      return requestJson<ProjectList>("GET", `/v1/projects?${params}`);
    },
    createProject: (body) => requestJson<Project>("POST", "/v1/projects", body),
    updateProject: (id, body) => requestJson<Project>("PATCH", `/v1/projects/${encodeURIComponent(id)}`, body),
    deleteProject: (id) => requestJson<void>("DELETE", `/v1/projects/${encodeURIComponent(id)}`),
  };
}
