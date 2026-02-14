import type { Disposable } from "./kernel/types.ts";

/** Conversation session summary (from list) or full (from get, with items). */
export interface Conversation {
  id: string;
  object: "conversation";
  created_at: number;
  updated_at: number;
  model: string;
  title: string | null;
  marker: string;
  group_id: string | null;
  parent_session_id: string | null;
  /** Source document ID (e.g., prompt) that created this conversation. */
  source_doc_id: string | null;
  metadata: Record<string, unknown>;
  search_snippet?: string | null;
  items?: Item[];
}

/** Paginated list response from GET /v1/conversations. */
export interface ConversationList {
  object: "list";
  data: Conversation[];
  has_more: boolean;
  cursor: string | null;
}

/** Patch body for PATCH /v1/conversations/{id}. */
export interface ConversationPatch {
  title?: string;
  marker?: string;
  metadata?: Record<string, unknown>;
}

/** Fork body for POST /v1/conversations/{id}/fork. */
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

export type ContentPart = InputTextPart | OutputTextPart;

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

/** Request body for POST /v1/responses. */
export interface CreateResponseRequest {
  model: string;
  input?: string;
  previous_response_id?: string | null;
  /** TaluDB session ID — used to continue an existing stored conversation. */
  session_id?: string | null;
  /** Prompt document ID to use as system prompt. */
  prompt_id?: string | null;
  /** Sampling parameters */
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  max_output_tokens?: number;
  repetition_penalty?: number;
  seed?: number;
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
  available_models: ModelEntry[];
}

/** Partial update body for PATCH /v1/settings. */
export interface SettingsPatch {
  model?: string | null;
  system_prompt?: string | null;
  max_output_tokens?: number | null;
  context_length?: number | null;
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
  /** Scope: "conversations" or "items" */
  scope: "conversations" | "items";
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

/** Batch operation request for POST /v1/conversations/batch. */
export interface BatchRequest {
  action: "delete" | "archive" | "unarchive" | "add_tags" | "remove_tags";
  ids: string[];
  tags?: string[]; // Required for add_tags/remove_tags
}

// ---------------------------------------------------------------------------
// Documents API types
// ---------------------------------------------------------------------------

/** Document record from GET /v1/documents/:id. */
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
  getPromptNameById(id: string): string | null;
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
