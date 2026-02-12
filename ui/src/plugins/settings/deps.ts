/**
 * Settings plugin shared dependencies — initialized once by run(),
 * imported by all internal modules.
 */

import type { ApiClient } from "../../api.ts";
import type { EventBus, ManagedTimers } from "../../kernel/types.ts";

export let api: ApiClient;
export let events: EventBus;
export let timers: ManagedTimers;

export function initSettingsDeps(deps: {
  api: ApiClient;
  events: EventBus;
  timers: ManagedTimers;
}): void {
  api = deps.api;
  events = deps.events;
  timers = deps.timers;
}
