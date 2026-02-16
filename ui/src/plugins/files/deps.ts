/**
 * Files plugin shared dependencies — initialized once in run().
 */

import type { ApiClient } from "../../api.ts";
import type {
  Notifications,
  StandardDialogs,
  EventBus,
  UploadAccess,
  DownloadAccess,
  ManagedTimers,
  FormatAccess,
} from "../../kernel/types.ts";

export let api: ApiClient;
export let notify: Notifications;
export let dialogs: StandardDialogs;
export let pluginEvents: EventBus;
export let upload: UploadAccess;
export let download: DownloadAccess;
export let timers: ManagedTimers;
export let format: FormatAccess;

export function initFilesDeps(deps: {
  api: ApiClient;
  notify: Notifications;
  dialogs: StandardDialogs;
  events: EventBus;
  upload: UploadAccess;
  download: DownloadAccess;
  timers: ManagedTimers;
  format: FormatAccess;
}): void {
  api = deps.api;
  notify = deps.notify;
  dialogs = deps.dialogs;
  pluginEvents = deps.events;
  upload = deps.upload;
  download = deps.download;
  timers = deps.timers;
  format = deps.format;
}
