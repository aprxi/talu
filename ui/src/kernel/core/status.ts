/**
 * Plugin Status — initialization progress signaling.
 *
 * setBusy(msg?) signals the kernel that the plugin is still initializing.
 * setReady() clears the indicator. Auto-set when run() settles.
 */

import type { PluginStatus } from "../types.ts";

export class PluginStatusImpl implements PluginStatus {
  private pluginId: string;
  private _busy = false;
  private _message: string | undefined;

  constructor(pluginId: string) {
    this.pluginId = pluginId;
  }

  get isBusy(): boolean {
    return this._busy;
  }

  get message(): string | undefined {
    return this._message;
  }

  setBusy(message?: string): void {
    this._busy = true;
    this._message = message;
    // Future: update status bar indicator.
  }

  setReady(): void {
    this._busy = false;
    this._message = undefined;
  }
}
