/** Shared dependencies for the repo plugin, initialised once in run(). */

import type { ApiClient } from "../../api.ts";
import type {
  EventBus,
  Notifications,
  StandardDialogs,
  ManagedTimers,
  FormatAccess,
  PluginStatus,
} from "../../kernel/types.ts";

export let api: ApiClient;
export let events: EventBus;
export let notifications: Notifications;
export let dialogs: StandardDialogs;
export let timers: ManagedTimers;
export let format: FormatAccess;
export let status: PluginStatus;

export function initRepoDeps(deps: {
  api: ApiClient;
  events: EventBus;
  notifications: Notifications;
  dialogs: StandardDialogs;
  timers: ManagedTimers;
  format: FormatAccess;
  status: PluginStatus;
}): void {
  api = deps.api;
  events = deps.events;
  notifications = deps.notifications;
  dialogs = deps.dialogs;
  timers = deps.timers;
  format = deps.format;
  status = deps.status;
}
