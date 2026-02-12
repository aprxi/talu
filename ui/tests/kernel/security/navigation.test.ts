import { describe, test, expect, beforeEach, afterEach, spyOn } from "bun:test";
import { installNavigationInterception } from "../../../src/kernel/security/navigation.ts";
import type { Disposable } from "../../../src/kernel/types.ts";

describe("installNavigationInterception", () => {
  let disposable: Disposable;

  beforeEach(() => {
    // Reset the module-level `installed` flag in case a prior test (e.g. bootKernel)
    // left it set. Install + dispose resets it.
    installNavigationInterception().dispose();
  });

  afterEach(() => {
    disposable?.dispose();
  });

  test("returns a disposable", () => {
    disposable = installNavigationInterception();
    expect(typeof disposable.dispose).toBe("function");
  });

  test("idempotent — second call returns no-op", () => {
    disposable = installNavigationInterception();
    const second = installNavigationInterception();
    expect(() => second.dispose()).not.toThrow();
    // First dispose restores.
    disposable.dispose();
    // Re-install works after dispose.
    disposable = installNavigationInterception();
  });

  test("blocks external URLs via window.open", () => {
    disposable = installNavigationInterception();
    // External URL returns null (blocked).
    const result = window.open("https://external.example.com");
    expect(result).toBeNull();
  });

  test("dispose resets installed flag", () => {
    disposable = installNavigationInterception();
    disposable.dispose();
    // Can re-install after dispose.
    disposable = installNavigationInterception();
    expect(typeof disposable.dispose).toBe("function");
  });

  test("window.open with external URL returns null", () => {
    disposable = installNavigationInterception();
    // External URL should be blocked (shows async confirm dialog, returns null synchronously).
    const result = window.open("https://evil.com/steal");
    expect(result).toBeNull();
  });

  test("window.open with same-origin URL passes through", () => {
    disposable = installNavigationInterception();
    // Same-origin URL should pass through.
    // The result type depends on the environment but should not throw.
    expect(() => window.open("/internal/page")).not.toThrow();
  });
});
