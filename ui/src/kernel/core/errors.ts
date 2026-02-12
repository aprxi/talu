import type { PluginHealth } from "../types.ts";

const DISABLE_THRESHOLD = 3;

/** Wraps a synchronous function in a try/catch. Returns undefined on failure. */
export function errorBoundary<T>(pluginId: string, fn: () => T): T | undefined {
  try {
    return fn();
  } catch (err) {
    console.error(`[kernel] Plugin "${pluginId}" threw:`, err);
    return undefined;
  }
}

/** Wraps an async function in a try/catch. Returns undefined on failure. */
export async function asyncErrorBoundary<T>(
  pluginId: string,
  fn: () => Promise<T>,
): Promise<T | undefined> {
  try {
    return await fn();
  } catch (err) {
    console.error(`[kernel] Plugin "${pluginId}" threw:`, err);
    return undefined;
  }
}

/** Per-plugin health state machine with three-strike escalation. */
export class HealthTracker {
  private strikes = 0;
  private _state: PluginHealth = "healthy";

  get state(): PluginHealth {
    return this._state;
  }

  get isDisabled(): boolean {
    return this._state === "disabled";
  }

  recordFailure(): void {
    if (this._state === "disabled") return;
    this.strikes++;
    if (this.strikes >= DISABLE_THRESHOLD) {
      this._state = "disabled";
    } else if (this.strikes >= DISABLE_THRESHOLD - 1) {
      this._state = "warning";
    }
  }

  recordSuccess(): void {
    if (this._state === "disabled") return;
    this.strikes = 0;
    this._state = "healthy";
  }
}
