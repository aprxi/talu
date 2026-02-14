/**
 * Chat plugin local state — replaces the ctx.state and ctx top-level
 * properties previously owned by context.ts.
 *
 * All chat/, sidebar/, and panel/ modules read/write this state
 * instead of the global ctx object.
 */

import type { Conversation, FileObject } from "../../types.ts";

export interface ChatAttachment {
  file: FileObject;
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
  pagination: {
    cursor: string | null;
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
  pagination: {
    cursor: null,
    hasMore: true,
    isLoading: false,
  },
};
