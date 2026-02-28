/**
 * Settings plugin shared dependencies — initialized once by run(),
 * imported by all internal modules.
 */

import type { ApiClient } from "../../api.ts";
import type {
  EventBus,
  ManagedTimers,
  ModeAccess,
  LayoutAccess,
  ThemeAccess,
  StorageAccess,
  DownloadAccess,
  Notifications,
  StandardDialogs,
} from "../../kernel/types.ts";

export let api: ApiClient;
export let events: EventBus;
export let timers: ManagedTimers;
export let mode: ModeAccess;
export let layout: LayoutAccess;
export let theme: ThemeAccess;
export let storage: StorageAccess;
export let download: DownloadAccess;
export let notifications: Notifications;
export let dialogs: StandardDialogs;

export function initSettingsDeps(deps: {
  api: ApiClient;
  events: EventBus;
  timers: ManagedTimers;
  mode: ModeAccess;
  layout: LayoutAccess;
  theme: ThemeAccess;
  storage: StorageAccess;
  download: DownloadAccess;
  notifications: Notifications;
  dialogs: StandardDialogs;
}): void {
  api = deps.api;
  events = deps.events;
  timers = deps.timers;
  mode = deps.mode;
  layout = deps.layout;
  theme = deps.theme;
  storage = deps.storage;
  download = deps.download;
  notifications = deps.notifications;
  dialogs = deps.dialogs;
}
