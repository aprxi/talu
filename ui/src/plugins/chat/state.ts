/**
 * Chat plugin local state — replaces the ctx.state and ctx top-level
 * properties previously owned by context.ts.
 *
 * All chat/, sidebar/, and panel/ modules read/write this state
 * instead of the global ctx object.
 */

import type { Conversation } from "../../types.ts";

export interface ChatState {
  sessions: Conversation[];
  activeSessionId: string | null;
  activeChat: Conversation | null;
  lastResponseId: string | null;
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
  isGenerating: false,
  streamAbort: null,
  pagination: {
    cursor: null,
    hasMore: true,
    isLoading: false,
  },
};
