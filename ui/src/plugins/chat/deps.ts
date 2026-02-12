/**
 * Chat plugin shared dependencies — initialized once by the chat plugin's
 * run() function, imported by all internal modules.
 */

import type { ApiClient } from "../../api.ts";
import type { ModelsService, PromptsService } from "../../types.ts";
import type { Notifications, ServiceAccess, EventBus, LayoutAccess, ClipboardAccess, DownloadAccess, ManagedTimers, ManagedObservers, FormatAccess } from "../../kernel/types.ts";

export let api: ApiClient;
export let notifications: Notifications;
export let services: ServiceAccess;
export let events: EventBus;
export let layout: LayoutAccess;
export let clipboard: ClipboardAccess;
export let download: DownloadAccess;
export let timers: ManagedTimers;
export let observe: ManagedObservers;
export let format: FormatAccess;

export function initChatDeps(deps: {
  api: ApiClient;
  notifications: Notifications;
  services: ServiceAccess;
  events: EventBus;
  layout: LayoutAccess;
  clipboard: ClipboardAccess;
  download: DownloadAccess;
  timers: ManagedTimers;
  observe: ManagedObservers;
  format: FormatAccess;
}): void {
  api = deps.api;
  notifications = deps.notifications;
  services = deps.services;
  events = deps.events;
  layout = deps.layout;
  clipboard = deps.clipboard;
  download = deps.download;
  timers = deps.timers;
  observe = deps.observe;
  format = deps.format;
}

export function getModelsService(): ModelsService | undefined {
  return services.get<ModelsService>("talu.models");
}

export function getPromptsService(): PromptsService | undefined {
  return services.get<PromptsService>("talu.prompts");
}
