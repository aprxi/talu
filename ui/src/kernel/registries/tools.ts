/**
 * Tool Registry — register/get LLM-callable tools by namespaced ID.
 *
 * Tools are distinct from commands (user-triggered) and services (inter-plugin).
 * First-registrant-wins collision rule. Tools are disposables.
 */

import type { Disposable, ToolDefinition, ToolResult, ToolRegistry } from "../types.ts";
import type { HookPipelineImpl } from "./hooks.ts";
import { resolveAlias } from "../core/alias.ts";
import { validateArgs } from "../core/schema-validator.ts";
import { StandardDialogsImpl } from "../ui/dialogs.ts";

export class ToolRegistryImpl implements ToolRegistry {
  private tools = new Map<string, { definition: ToolDefinition; pluginId: string }>();
  private hookPipeline: HookPipelineImpl;

  constructor(hookPipeline: HookPipelineImpl) {
    this.hookPipeline = hookPipeline;
  }

  /** Register a tool. Called via scoped wrapper in PluginContext. */
  registerScoped(pluginId: string, fqId: string, definition: ToolDefinition): Disposable {
    if (this.tools.has(fqId)) {
      console.warn(`[kernel] Tool "${fqId}" already registered — ignoring duplicate.`);
      return { dispose() {} };
    }

    this.tools.set(fqId, { definition, pluginId });

    return {
      dispose: () => {
        const entry = this.tools.get(fqId);
        if (entry && entry.definition === definition) {
          this.tools.delete(fqId);
        }
      },
    };
  }

  register(id: string, definition: ToolDefinition): Disposable {
    return this.registerScoped("unknown", id, definition);
  }

  get(id: string): ToolDefinition | undefined {
    return this.tools.get(resolveAlias(id))?.definition;
  }

  /** Execute a tool through the hook pipeline. */
  async execute(
    toolId: string,
    args: Record<string, unknown>,
    signal: AbortSignal,
  ): Promise<ToolResult> {
    const resolvedId = resolveAlias(toolId);
    const entry = this.tools.get(resolvedId);
    if (!entry) throw new Error(`Tool "${toolId}" not found.`);

    // Step 1: Validate initial args BEFORE hooks (security: reject malformed input early).
    const initialValidation = validateArgs(entry.definition.parameters, args);
    if (!initialValidation.valid) {
      throw new Error(
        `Tool "${toolId}" argument validation failed: ${initialValidation.errors.join("; ")}`,
      );
    }

    // Step 2: Run before hooks (may transform args).
    const beforeResult = await this.hookPipeline.run("tool.execute.before", {
      toolId,
      args,
    });

    if (
      beforeResult &&
      typeof beforeResult === "object" &&
      "$block" in beforeResult
    ) {
      throw new Error(
        `Tool "${toolId}" blocked: ${(beforeResult as { reason: string }).reason}`,
      );
    }

    const finalArgs = (beforeResult as { args: Record<string, unknown> }).args ?? args;

    // Step 3: Validate post-hook args (hooks may have mutated them).
    if (finalArgs !== args) {
      const postHookValidation = validateArgs(entry.definition.parameters, finalArgs);
      if (!postHookValidation.valid) {
        throw new Error(
          `Tool "${toolId}" argument validation failed after hooks: ${postHookValidation.errors.join("; ")}`,
        );
      }
    }

    // Step 4: User approval gate for sensitive tools.
    if (entry.definition.requiresUserApproval) {
      const dialogs = new StandardDialogsImpl(entry.pluginId);
      const allowed = await dialogs.confirm({
        title: "Tool Execution",
        message: `"${resolvedId}" requires approval to execute. Allow?`,
      });
      if (!allowed) {
        throw new Error(`Tool "${toolId}" execution denied by user.`);
      }
    }

    // Step 5: Execute the tool (error-bounded).
    let result: ToolResult;
    try {
      result = await entry.definition.execute(finalArgs, signal);
    } catch (err) {
      console.error(`[kernel] Tool "${resolvedId}" from "${entry.pluginId}" threw:`, err);
      throw new Error(
        `Tool "${toolId}" execution failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }

    // Step 6: Validate result structure.
    assertValidToolResult(resolvedId, result);

    // Step 7: Run after hooks.
    const afterResult = await this.hookPipeline.run("tool.execute.after", {
      toolId,
      args: finalArgs,
      result,
    });

    return ((afterResult as { result: ToolResult }).result ?? result);
  }
}

/** Validate that a tool's return value conforms to the ToolResult contract. */
function assertValidToolResult(toolId: string, result: unknown): asserts result is ToolResult {
  if (!result || typeof result !== "object") {
    throw new Error(`Tool "${toolId}" returned invalid result: expected object with content array.`);
  }
  const r = result as Record<string, unknown>;
  if (!Array.isArray(r.content)) {
    throw new Error(`Tool "${toolId}" returned invalid result: content must be an array.`);
  }
  for (let i = 0; i < r.content.length; i++) {
    const part = r.content[i];
    if (!part || typeof part !== "object" || !("id" in part) || !("type" in part)) {
      throw new Error(`Tool "${toolId}" result.content[${i}]: missing required fields (id, type).`);
    }
  }
}
