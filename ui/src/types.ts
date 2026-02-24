import type { Disposable } from "./kernel/types.ts";

/** Tag associated with a conversation (from the relational tags table). */
export interface ConversationTag {
  id: string;
  name: string;
  color?: string | null;
}

/** Conversation session summary (from list) or full (from get, with items). */
export interface Conversation {
  id: string;
  object: "session";
  created_at: number;
  updated_at: number;
  model: string;
  title: string | null;
  marker: string;
  group_id: string | null;
  parent_session_id: string | null;
  /** Source document ID (e.g., prompt) that created this conversation. */
  source_doc_id: string | null;
  /** Project this session belongs to (null if unassigned). */
  project_id?: string | null;
  metadata: Record<string, unknown>;
  /** Tags from the relational tags table (source of truth for search). */
  tags?: ConversationTag[];
  search_snippet?: string | null;
  items?: Item[];
}

/** Paginated list response from GET /v1/chat/sessions. */
export interface ConversationList {
  object: "list";
  data: Conversation[];
  has_more: boolean;
  cursor: string | null;
  total: number;
}

/** Patch body for PATCH /v1/chat/sessions/{id}. */
export interface ConversationPatch {
  title?: string;
  marker?: string;
  metadata?: Record<string, unknown>;
  project_id?: string | null;
}

/** Fork body for POST /v1/chat/sessions/{id}/fork. */
export interface ForkRequest {
  target_item_id?: number;
}

/** Discriminated union of all item types in a conversation. */
export type Item =
  | MessageItem
  | ReasoningItem
  | FunctionCallItem
  | FunctionCallOutputItem;

/** Generation settings used to produce a response. */
export interface GenerationSettings {
  model?: string;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  max_output_tokens?: number;
  repetition_penalty?: number;
  seed?: number;
}

/** Token usage statistics from a response. */
export interface UsageStats {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  /** Generation time in milliseconds (client-side measured). */
  duration_ms?: number;
  /** Tokens per second (output_tokens / duration). */
  tokens_per_second?: number;
}

export interface MessageItem {
  type: "message";
  id?: string;
  role: "user" | "assistant" | "system" | "developer";
  status?: string;
  finish_reason?: string;
  content: ContentPart[];
  /** Generation settings used to produce this message (assistant messages only). */
  generation?: GenerationSettings;
}

export type ContentPart = InputTextPart | OutputTextPart | InputImagePart | InputFilePart;

export interface InputTextPart {
  type: "input_text";
  text: string;
}

export interface OutputTextPart {
  type: "output_text";
  text: string;
  annotations?: unknown[];
  /** Code block metadata from core (talu_code_blocks). */
  talu_code_blocks?: CodeBlock[];
}

export interface InputImagePart {
  type: "input_image";
  image_url: string;
}

export interface InputFilePart {
  type: "input_file";
  file_data?: string;
  filename?: string;
}

/** Code block metadata detected by core. Positions are byte offsets. */
export interface CodeBlock {
  /** Sequential index of this block (0-based). */
  index: number;
  /** Byte offset where opening fence begins. */
  fence_start: number;
  /** Byte offset after closing fence (or current position if incomplete). */
  fence_end: number;
  /** Byte offset where language identifier begins. */
  language_start: number;
  /** Byte offset after language identifier. */
  language_end: number;
  /** Byte offset where code content begins. */
  content_start: number;
  /** Byte offset where code content ends. */
  content_end: number;
  /** True if closing fence was found. */
  complete: boolean;
}

export interface ReasoningTextPart {
  type: "reasoning_text";
  text: string;
}

export interface ReasoningItem {
  type: "reasoning";
  id?: string;
  content?: ReasoningTextPart[];
  summary?: SummaryPart[];
  finish_reason?: string;
}

export interface SummaryPart {
  type: "summary_text";
  text: string;
}

export interface FunctionCallItem {
  type: "function_call";
  id?: string;
  name: string;
  arguments: string;
  call_id: string;
  status?: string;
}

export interface FunctionCallOutputItem {
  type: "function_call_output";
  call_id: string;
  output: string;
}

/** Structured input content item for multimodal requests. */
export type InputContentItem = {
  type: "message";
  role: "user";
  content: InputContentPart[];
};

/** A single content part within a structured input message. */
export type InputContentPart =
  | { type: "input_text"; text: string }
  | { type: "input_image"; image_url: string }
  | { type: "input_file"; file_url: string; filename?: string };

/** Request body for POST /v1/responses. */
export interface CreateResponseRequest {
  model: string;
  input?: string | InputContentItem[];
  previous_response_id?: string | null;
  /** OpenResponses instructions field. */
  instructions?: string | null;
  /** Sampling parameters supported by OpenResponses. */
  temperature?: number;
  top_p?: number;
  max_output_tokens?: number;
  /** Optional tool configuration passthrough. */
  tools?: unknown;
  tool_choice?: unknown;
  metadata?: Record<string, unknown>;
  store?: boolean;
  stream?: boolean;
}

/** Model generation_config.json defaults. */
export interface ModelDefaults {
  temperature: number;
  top_k: number;
  top_p: number;
  do_sample: boolean;
}

/** Per-model sampling parameter overrides (model-specific). */
export interface ModelOverrides {
  temperature?: number | null;
  top_p?: number | null;
  top_k?: number | null;
  min_p?: number | null;
  max_output_tokens?: number | null;
  repetition_penalty?: number | null;
  seed?: number | null;
}

/** Enriched model entry returned by GET /v1/settings. */
export interface ModelEntry {
  id: string;
  source: "managed" | "hub";
  defaults: ModelDefaults;
  overrides: ModelOverrides;
}

/** Bucket settings from GET /v1/settings. */
export interface Settings {
  model: string | null;
  system_prompt: string | null;
  max_output_tokens: number | null;
  context_length: number | null;
  auto_title: boolean;
  default_prompt_id: string | null;
  system_prompt_enabled: boolean;
  available_models: ModelEntry[];
}

/** Partial update body for PATCH /v1/settings. */
export interface SettingsPatch {
  model?: string | null;
  system_prompt?: string | null;
  default_prompt_id?: string | null;
  max_output_tokens?: number | null;
  context_length?: number | null;
  auto_title?: boolean;
  system_prompt_enabled?: boolean;
  model_overrides?: ModelOverrides;
}

/** Uniform API result wrapper. */
export interface ApiResult<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

// ---------------------------------------------------------------------------
// Search API types
// ---------------------------------------------------------------------------

/** Search request body for POST /v1/search. */
export interface SearchRequest {
  /** Scope: "sessions" or "items" */
  scope: "sessions" | "items";
  /** Text search (case-insensitive substring) */
  text?: string;
  /** Structured filters */
  filters?: SearchFilters;
  /** Aggregations to compute: "tags", "models", "markers" */
  aggregations?: string[];
  /** Max results (default 20, max 100) */
  limit?: number;
  /** Pagination cursor */
  cursor?: string;
  /** Include conversation items in response */
  include_items?: boolean;
  /** Include match snippets/highlights */
  highlight?: boolean;
}

/** Structured filters for search. */
export interface SearchFilters {
  /** Tags (AND logic) - must have ALL */
  tags?: string[];
  /** Tags (OR logic) - must have ANY */
  tags_any?: string[];
  /** Model filter (supports wildcards like "qwen*") */
  model?: string;
  /** Created after timestamp (ms) */
  created_after?: number;
  /** Created before timestamp (ms) */
  created_before?: number;
  /** Updated after timestamp (ms) */
  updated_after?: number;
  /** Updated before timestamp (ms) */
  updated_before?: number;
  /** Marker exact match */
  marker?: string;
  /** Marker any (OR logic) */
  marker_any?: string[];
  /** Has any tags */
  has_tags?: boolean;
  /** Group ID (multi-tenant filter) */
  group_id?: string;
  /** Project ID filter */
  project_id?: string;
}

/** Search response from POST /v1/search. */
export interface SearchResponse {
  data: Conversation[];
  aggregations?: SearchAggregations;
  cursor?: string | null;
  has_more: boolean;
  total?: number;
}

/** Aggregation results. */
export interface SearchAggregations {
  tags?: TagAggregation[];
  models?: ValueAggregation[];
  markers?: ValueAggregation[];
  projects?: ValueAggregation[];
}

export interface TagAggregation {
  id: string;
  name: string;
  count: number;
}

export interface ValueAggregation {
  value: string;
  count: number;
}

// ---------------------------------------------------------------------------
// Batch Operations API types
// ---------------------------------------------------------------------------

/** Batch operation request for POST /v1/chat/sessions/batch. */
export interface BatchRequest {
  action: "delete" | "archive" | "unarchive" | "add_tags" | "remove_tags";
  ids: string[];
  tags?: string[]; // Required for add_tags/remove_tags
}

// ---------------------------------------------------------------------------
// Documents API types
// ---------------------------------------------------------------------------

/** Document record from GET /v1/db/tables/documents/:id. */
export interface Document {
  id: string;
  type: string;
  title: string;
  content?: DocumentContent;
  tags_text?: string | null;
  parent_id?: string | null;
  marker?: string | null;
  group_id?: string | null;
  owner_id?: string | null;
  created_at: number;
  updated_at: number;
  expires_at?: number | null;
}

/** Document content (JSON payload). */
export interface DocumentContent {
  system?: string;
  model?: string;
  temperature?: number;
  [key: string]: unknown;
}

/** Document summary from list. */
export interface DocumentSummary {
  id: string;
  type: string;
  title: string;
  marker?: string | null;
  created_at: number;
  updated_at: number;
}

/** Document list response. */
export interface DocumentList {
  data: DocumentSummary[];
  has_more: boolean;
}

/** Create document request. */
export interface CreateDocumentRequest {
  type: string;
  title: string;
  content: DocumentContent;
  tags_text?: string;
  parent_id?: string;
  marker?: string;
  group_id?: string;
  owner_id?: string;
}

/** Update document request. */
export interface UpdateDocumentRequest {
  title?: string;
  content?: DocumentContent;
  tags_text?: string;
  marker?: string;
}

// ---------------------------------------------------------------------------
// Files API types
// ---------------------------------------------------------------------------

/** File record from /v1/files endpoints. */
export interface FileObject {
  id: string;
  object: "file";
  bytes: number;
  created_at: number;
  filename: string;
  purpose: string;
  /** Detected MIME type (from inspection, may differ from browser-supplied). */
  mime_type?: string;
  /** High-level file kind: "image", "text", or "binary". */
  kind?: string;
  /** Image-specific metadata (present only for image files). */
  image?: FileImageInfo;
  /** Marker: "active" or "archived". */
  marker?: string;
}

/** Paginated file list from GET /v1/files. */
export interface FileList {
  object: "list";
  data: FileObject[];
  has_more: boolean;
  cursor: string | null;
  total: number;
}

/** Batch operation request for POST /v1/files/batch. */
export interface FileBatchRequest {
  action: "delete" | "archive" | "unarchive";
  ids: string[];
}

// ---------------------------------------------------------------------------
// File Inspect/Transform API types (stateless /v1/file namespace)
// ---------------------------------------------------------------------------

/** Image metadata from file inspection. */
export interface FileImageInfo {
  format: string;
  width: number;
  height: number;
  exif_orientation: number;
  aspect_ratio: number;
}

/** Response from POST /v1/file/inspect. */
export interface FileInspection {
  kind: string;
  mime: string;
  description: string;
  size: number;
  image: FileImageInfo | null;
}

// ---------------------------------------------------------------------------
// Repo API types
// ---------------------------------------------------------------------------

/** Cached model entry from GET /v1/repo/models. */
export interface RepoModel {
  id: string;
  path: string;
  source: string;
  size_bytes: number;
  mtime: number;
  architecture?: string;
  quant_scheme?: string;
  pinned: boolean;
}

/** Response from GET /v1/repo/models. */
export interface RepoModelList {
  models: RepoModel[];
  total_size_bytes: number;
}

/** Search result from GET /v1/repo/search. */
export interface RepoSearchResult {
  model_id: string;
  downloads: number;
  likes: number;
  last_modified: string;
  params_total: number;
}

/** Response from GET /v1/repo/search. */
export interface RepoSearchResponse {
  results: RepoSearchResult[];
}

// ---------------------------------------------------------------------------
// Plugin service contracts
// ---------------------------------------------------------------------------

/** Service contract for the models/settings plugin ("talu.models"). */
export interface ModelsService {
  getActiveModel(): string;
  getAvailableModels(): ModelEntry[];
  setActiveModel(id: string): void;
  onChange(handler: () => void): Disposable;
}

/** Service contract for the prompts plugin ("talu.prompts"). */
export interface PromptsService {
  getSelectedPromptId(): string | null;
  getDefaultPromptId(): string | null;
  getPromptNameById(id: string): string | null;
  getPromptContentById(id: string): string | null;
  getAll(): { id: string; name: string }[];
}

/** Service contract for the chat plugin ("talu.chat"). */
export interface ChatService {
  selectChat(id: string): Promise<void>;
  startNewConversation(): void;
  showWelcome(): void;
  cancelGeneration(): void;
  refreshSidebar(): Promise<void>;
  getSessions(): Conversation[];
  getActiveSessionId(): string | null;
}
