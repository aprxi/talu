/**
 * Manifest Runtime Validation — validate structure, format, and safety of
 * plugin manifests before registration.
 *
 * Called during registerPlugin() before plugin loading.
 */

import type { PluginManifest } from "../types.ts";

export interface ManifestValidationResult {
  valid: boolean;
  warnings: string[];
  errors: string[];
}

/** Current kernel API version. Third-party plugins must declare this. */
export const KERNEL_API_VERSION = "1";

/** Known permission strings that plugins can declare. */
export const KNOWN_PERMISSIONS = new Set([
  "network",    // ctx.network — HTTP requests
  "storage",    // ctx.storage — persistent key-value storage
  "clipboard",  // ctx.clipboard — write to system clipboard
  "download",   // ctx.download — trigger file downloads
  "upload",     // ctx.upload — file upload/download APIs
  "hooks",      // ctx.hooks — intercept/transform kernel operations
  "tools",      // ctx.tools — register LLM-callable tools
  "menus",      // ctx.menus.registerItem — contribute UI actions to slots
]);

const MAX_NAME = 64;
const MAX_LABEL = 128;
const MAX_DESCRIPTION = 512;

/** ASCII control characters except newline. */
const CONTROL_CHARS = /[\x00-\x09\x0b-\x1f]/g;

/** Unsafe icon patterns (absolute URLs, data URIs). */
const UNSAFE_ICON = /^(https?:|data:|\/\/)/i;

/** Reverse-domain ID: lowercase segments separated by dots (e.g. talu.chat, com.example.foo). */
const VALID_ID = /^[a-z][a-z0-9]*(\.[a-z][a-z0-9]*)*$/;

/** Basic semver: digits.digits.digits (e.g. 0.1.0, 1.2.3). */
const VALID_VERSION = /^\d+\.\d+\.\d+$/;

/** Mode keys: lowercase alphanumeric + hyphens (e.g. chat, conversations, my-mode). */
const VALID_MODE_KEY = /^[a-z][a-z0-9-]*$/;

function checkString(
  value: string | undefined,
  fieldName: string,
  maxLength: number,
  allowNewlines: boolean,
  result: ManifestValidationResult,
): void {
  if (!value) return;

  if (value.length > maxLength) {
    result.errors.push(`${fieldName} exceeds ${maxLength} characters (${value.length})`);
  }

  const pattern = allowNewlines ? CONTROL_CHARS : /[\x00-\x1f]/g;
  if (pattern.test(value)) {
    result.warnings.push(`${fieldName} contains control characters (stripped)`);
  }
}

export function validateManifest(manifest: PluginManifest): ManifestValidationResult {
  const result: ManifestValidationResult = { valid: true, warnings: [], errors: [] };

  // Required fields.
  if (!manifest.id || typeof manifest.id !== "string") {
    result.errors.push("id is required");
  } else if (!VALID_ID.test(manifest.id)) {
    result.errors.push(`id "${manifest.id}" must be reverse-domain format (e.g. com.example.plugin)`);
  }

  if (!manifest.name || typeof manifest.name !== "string") {
    result.errors.push("name is required");
  }

  if (!manifest.version || typeof manifest.version !== "string") {
    result.errors.push("version is required");
  } else if (!VALID_VERSION.test(manifest.version)) {
    result.errors.push(`version "${manifest.version}" must be semver format (e.g. 1.0.0)`);
  }

  // apiVersion enforcement: required for non-builtin plugins.
  if (!manifest.builtin) {
    if (!manifest.apiVersion) {
      result.errors.push("apiVersion is required for third-party plugins");
    } else if (manifest.apiVersion !== KERNEL_API_VERSION) {
      result.errors.push(`apiVersion "${manifest.apiVersion}" is not supported (expected "${KERNEL_API_VERSION}")`);
    }
  }

  // Validate declared permissions.
  if (manifest.permissions) {
    for (const perm of manifest.permissions) {
      if (!KNOWN_PERMISSIONS.has(perm)) {
        result.errors.push(`unknown permission "${perm}" (known: ${[...KNOWN_PERMISSIONS].join(", ")})`);
      }
    }
  }

  // String length / control character checks.
  checkString(manifest.name, "name", MAX_NAME, false, result);

  // Mode key format.
  const modeKey = manifest.contributes?.mode?.key;
  if (modeKey && !VALID_MODE_KEY.test(modeKey)) {
    result.errors.push(`contributes.mode.key "${modeKey}" must be lowercase alphanumeric with hyphens`);
  }

  // Contributed views.
  if (manifest.contributes?.views) {
    for (const view of manifest.contributes.views) {
      if (!view.id) result.errors.push("contributes.views[].id must be non-empty");
      checkString(view.label, `contributes.views[${view.id}].label`, MAX_LABEL, false, result);
    }
  }

  // Contributed commands.
  if (manifest.contributes?.commands) {
    for (const cmd of manifest.contributes.commands) {
      if (!cmd.id) result.errors.push("contributes.commands[].id must be non-empty");
      checkString(cmd.label, `contributes.commands[${cmd.id}].label`, MAX_LABEL, false, result);
    }
  }

  // Contributed status bar items.
  if (manifest.contributes?.statusBarItems) {
    for (const item of manifest.contributes.statusBarItems) {
      if (!item.id) result.errors.push("contributes.statusBarItems[].id must be non-empty");
      checkString(item.label, `contributes.statusBarItems[${item.id}].label`, MAX_LABEL, false, result);
    }
  }

  // Contributed menus.
  if (manifest.contributes?.menus) {
    for (const item of manifest.contributes.menus) {
      if (!item.id) result.errors.push("contributes.menus[].id must be non-empty");
      if (!item.slot) result.errors.push("contributes.menus[].slot must be non-empty");
      if (!item.command) result.errors.push("contributes.menus[].command must be non-empty");
      checkString(item.label, `contributes.menus[${item.id}].label`, MAX_LABEL, false, result);
    }
  }

  // Contributed tools.
  if (manifest.contributes?.tools) {
    for (const tool of manifest.contributes.tools) {
      if (!tool.id) result.errors.push("contributes.tools[].id must be non-empty");
      checkString(
        tool.description,
        `contributes.tools[${tool.id}].description`,
        MAX_DESCRIPTION,
        true,
        result,
      );
    }
  }

  result.valid = result.errors.length === 0;
  return result;
}

/** Strip control characters from a manifest string for safe UI display. */
export function sanitizeManifestString(value: string, maxLength: number, allowNewlines?: boolean): string {
  const pattern = allowNewlines ? CONTROL_CHARS : /[\x00-\x1f]/g;
  let sanitized = value.replace(pattern, "");
  if (sanitized.length > maxLength) {
    sanitized = sanitized.slice(0, maxLength - 1) + "\u2026";
  }
  return sanitized;
}
