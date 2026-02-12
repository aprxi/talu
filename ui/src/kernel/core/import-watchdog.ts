/**
 * Import Watchdog — timeout wrapper for dynamic import().
 *
 * Catches top-level `await` hangs in plugin entry points.
 * Synchronous hangs (while(true)) freeze the main thread before
 * the timeout can fire, but the error is visible after reload.
 */

const DEFAULT_TIMEOUT_MS = 10_000;

export class ImportTimeoutError extends Error {
  readonly url: string;
  constructor(url: string) {
    super(`Plugin import timed out after ${DEFAULT_TIMEOUT_MS}ms: ${url}`);
    this.name = "ImportTimeoutError";
    this.url = url;
  }
}

/**
 * Dynamic import with timeout watchdog.
 * Rejects with ImportTimeoutError if the import doesn't resolve within the deadline.
 */
export async function importWithWatchdog<T = unknown>(
  url: string,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const importPromise = import(/* @vite-ignore */ url) as Promise<T>;
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new ImportTimeoutError(url)), timeoutMs);
  });
  return Promise.race([importPromise, timeoutPromise]);
}
