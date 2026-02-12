/**
 * Browser plugin shared dependencies — initialized once in run().
 */

import type { ApiClient } from "../../api.ts";
import type { Notifications, StandardDialogs, EventBus, DownloadAccess, ManagedTimers } from "../../kernel/types.ts";
import type { ChatService } from "../../types.ts";

export let api: ApiClient;
export let notify: Notifications;
export let dialogs: StandardDialogs;
export let pluginEvents: EventBus;
export let chatService: ChatService;
export let pluginDownload: DownloadAccess;
export let pluginTimers: ManagedTimers;

export function initBrowserDeps(deps: {
  api: ApiClient;
  notify: Notifications;
  dialogs: StandardDialogs;
  events: EventBus;
  chatService: ChatService;
  download: DownloadAccess;
  timers: ManagedTimers;
}): void {
  api = deps.api;
  notify = deps.notify;
  dialogs = deps.dialogs;
  pluginEvents = deps.events;
  chatService = deps.chatService;
  pluginDownload = deps.download;
  pluginTimers = deps.timers;
}
