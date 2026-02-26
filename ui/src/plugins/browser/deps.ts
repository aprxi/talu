/**
 * Browser plugin shared dependencies — initialized once in run().
 */

import type { ApiClient } from "../../api.ts";
import type { Notifications, StandardDialogs, EventBus, DownloadAccess, ManagedTimers, MenuAccess, LayoutAccess, ModeAccess } from "../../kernel/types.ts";
import type { ChatService } from "../../types.ts";

export let api: ApiClient;
export let notify: Notifications;
export let dialogs: StandardDialogs;
export let pluginEvents: EventBus;
export let chatService: ChatService;
export let pluginDownload: DownloadAccess;
export let pluginTimers: ManagedTimers;
export let menus: MenuAccess;
export let layout: LayoutAccess;
export let mode: ModeAccess;

export function initBrowserDeps(deps: {
  api: ApiClient;
  notify: Notifications;
  dialogs: StandardDialogs;
  events: EventBus;
  chatService: ChatService;
  download: DownloadAccess;
  timers: ManagedTimers;
  menus: MenuAccess;
  layout: LayoutAccess;
  mode: ModeAccess;
}): void {
  api = deps.api;
  notify = deps.notify;
  dialogs = deps.dialogs;
  pluginEvents = deps.events;
  chatService = deps.chatService;
  pluginDownload = deps.download;
  pluginTimers = deps.timers;
  menus = deps.menus;
  layout = deps.layout;
  mode = deps.mode;
}
