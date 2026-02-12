/**
 * Hook Pipeline — ordered interception system.
 *
 * Hooks are distinct from events: they form a pipeline where each handler
 * can inspect, modify, or block an operation. Priority-based ordering,
 * structuredClone between hooks, error boundary per handler.
 */

import type { Disposable, HookPipeline } from "../types.ts";

interface HookHandler {
  pluginId: string;
  priority: number;
  handler: (value: unknown) => unknown;
}

export class HookPipelineImpl implements HookPipeline {
  private hooks = new Map<string, HookHandler[]>();

  /** Register a hook handler scoped to a plugin. */
  onScoped(
    pluginId: string,
    name: string,
    handler: (value: unknown) => unknown,
    options?: { priority?: number },
  ): Disposable {
    const entry: HookHandler = {
      pluginId,
      priority: options?.priority ?? 0,
      handler,
    };

    let list = this.hooks.get(name);
    if (!list) {
      list = [];
      this.hooks.set(name, list);
    }
    list.push(entry);

    // Sort: higher priority first for *.before hooks, last for *.after hooks.
    // For simplicity, always sort higher priority first — the convention is that
    // security hooks use high priority to run first on .before and last on .after.
    list.sort((a, b) => b.priority - a.priority);

    return {
      dispose: () => {
        const idx = list!.indexOf(entry);
        if (idx >= 0) list!.splice(idx, 1);
        if (list!.length === 0) this.hooks.delete(name);
      },
    };
  }

  /** Facade method — delegates to onScoped. Used by PluginContext. */
  on<T = unknown>(
    name: string,
    handler: (value: T) => T | { $block: true; reason: string } | void | Promise<T | { $block: true; reason: string } | void>,
    options?: { priority?: number },
  ): Disposable {
    // When called through PluginContext, the pluginId is bound externally.
    return this.onScoped("unknown", name, handler as (value: unknown) => unknown, options);
  }

  /**
   * Run a hook pipeline. Each handler receives the (cloned) output of the
   * previous handler. Returns the final value, or a block sentinel.
   */
  async run<T>(name: string, initialValue: T): Promise<T | { $block: true; reason: string }> {
    const list = this.hooks.get(name);
    if (!list || list.length === 0) return initialValue;

    let current: unknown = initialValue;

    for (const entry of list) {
      // Clone between hooks to prevent shared-reference mutation.
      const input = structuredClone(current);
      const frozen = Object.freeze(input);

      let result: unknown;
      try {
        result = await entry.handler(frozen);
      } catch (err) {
        console.error(`[kernel] Hook "${name}" handler from "${entry.pluginId}" threw:`, err);
        // Skip this handler — pipeline continues with previous value.
        continue;
      }

      // Check for block sentinel.
      if (result && typeof result === "object" && "$block" in result && (result as Record<string, unknown>).$block === true) {
        return result as { $block: true; reason: string };
      }

      // undefined/void means pass-through.
      if (result !== undefined) {
        current = result;
      }
    }

    return current as T;
  }
}
