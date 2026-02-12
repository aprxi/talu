/**
 * Prompts plugin shared dependencies — initialized once by run(),
 * imported by all internal modules.
 */

import type { ApiClient } from "../../api.ts";
import type { Notifications, EventBus, StorageAccess, ClipboardAccess, ManagedTimers, Logger } from "../../kernel/types.ts";

export let api: ApiClient;
export let events: EventBus;
export let storage: StorageAccess;
export let clipboard: ClipboardAccess;
export let timers: ManagedTimers;
export let notifications: Notifications;
export let log: Logger;

export function initPromptsDeps(deps: {
  api: ApiClient;
  events: EventBus;
  storage: StorageAccess;
  clipboard: ClipboardAccess;
  timers: ManagedTimers;
  notifications: Notifications;
  log: Logger;
}): void {
  api = deps.api;
  events = deps.events;
  storage = deps.storage;
  clipboard = deps.clipboard;
  timers = deps.timers;
  notifications = deps.notifications;
  log = deps.log;
}
