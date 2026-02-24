/**
 * Chat plugin local state — replaces the ctx.state and ctx top-level
 * properties previously owned by context.ts.
 *
 * All chat/, sidebar/, and panel/ modules read/write this state
 * instead of the global ctx object.
 */

import type { Conversation } from "../../types.ts";
import type { UploadFileReference } from "../../kernel/types.ts";

export interface ChatAttachment {
  file: UploadFileReference;
  mimeType: string | null;
}

export interface ChatState {
  sessions: Conversation[];
  activeSessionId: string | null;
  activeChat: Conversation | null;
  lastResponseId: string | null;
  attachments: ChatAttachment[];
  isUploadingAttachments: boolean;
  isGenerating: boolean;
  streamAbort: AbortController | null;
  eventsAbort: AbortController | null;
  eventsResponseId: string | null;
  eventsVerbosity: 1 | 2 | 3;
  /** Incremented on each navigation (selectChat, startNewConversation). Streams
   *  capture this at start and only write to global state while it matches. */
  activeViewId: number;
  /** Session IDs with active background streams (user navigated away during generation). */
  backgroundStreamSessions: Set<string>;
  /** Saved transcript DOM fragments for background-streaming sessions (keyed by session ID).
   *  Streams continue writing to these detached elements; restored when user navigates back. */
  backgroundStreamDom: Map<string, DocumentFragment>;
  /** Whether system prompts are enabled (from settings). */
  systemPromptEnabled: boolean;
  sidebarSearchQuery: string;
  pagination: {
    offset: number;
    hasMore: boolean;
    isLoading: boolean;
  };
}

export const chatState: ChatState = {
  sessions: [],
  activeSessionId: null,
  activeChat: null,
  lastResponseId: null,
  attachments: [],
  isUploadingAttachments: false,
  isGenerating: false,
  streamAbort: null,
  eventsAbort: null,
  eventsResponseId: null,
  eventsVerbosity: 1,
  activeViewId: 0,
  backgroundStreamSessions: new Set(),
  backgroundStreamDom: new Map(),
  systemPromptEnabled: true,
  sidebarSearchQuery: "",
  pagination: {
    offset: 0,
    hasMore: true,
    isLoading: false,
  },
};
