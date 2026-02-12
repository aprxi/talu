import type { Logger } from "../types.ts";

/**
 * Production mode: hash string arguments to prevent PII leaking into browser
 * console history. Structured objects (Error, numbers, booleans) pass through.
 * Development mode (localhost / file://): full passthrough for debugging.
 */

const isDev =
  typeof location !== "undefined" &&
  (location.hostname === "localhost" || location.hostname === "127.0.0.1" || location.protocol === "file:");

async function hashString(s: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  const hex = Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  return `string(sha256-${hex.slice(0, 12)})`;
}

/** Sanitize a single log argument: strings → hashed, others → passthrough. */
function sanitizeArg(arg: unknown): unknown | Promise<unknown> {
  if (typeof arg === "string") return hashString(arg);
  return arg;
}

/** Resolve all arguments (some may be async from hashing). */
async function sanitizeArgs(args: unknown[]): Promise<unknown[]> {
  return Promise.all(args.map(sanitizeArg));
}

export function createLogger(pluginId: string): Logger {
  const prefix = `[${pluginId}]`;

  if (isDev) {
    return {
      info: (msg, ...args) => console.info(prefix, msg, ...args),
      warn: (msg, ...args) => console.warn(prefix, msg, ...args),
      error: (msg, ...args) => console.error(prefix, msg, ...args),
    };
  }

  return {
    info: (msg, ...args) =>
      void sanitizeArgs([msg, ...args]).then((safe) => console.info(prefix, ...safe)),
    warn: (msg, ...args) =>
      void sanitizeArgs([msg, ...args]).then((safe) => console.warn(prefix, ...safe)),
    error: (msg, ...args) =>
      void sanitizeArgs([msg, ...args]).then((safe) => console.error(prefix, ...safe)),
  };
}
