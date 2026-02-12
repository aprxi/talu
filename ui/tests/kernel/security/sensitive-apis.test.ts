import { describe, test, expect, spyOn, beforeEach, afterEach } from "bun:test";
import { installSensitiveApiInterception } from "../../../src/kernel/security/sensitive-apis.ts";
import type { Disposable } from "../../../src/kernel/types.ts";

describe("installSensitiveApiInterception", () => {
  let disposable: Disposable;

  beforeEach(() => {
    // HappyDOM lacks window.print — polyfill so the source code doesn't crash.
    if (!window.print) {
      (window as { print?: () => void }).print = () => {};
    }
    // Reset the module-level `installed` flag in case a prior test (e.g. bootKernel)
    // left it set. Install + dispose resets it.
    installSensitiveApiInterception().dispose();
  });

  afterEach(() => {
    disposable?.dispose();
  });

  test("returns a disposable", () => {
    disposable = installSensitiveApiInterception();
    expect(typeof disposable.dispose).toBe("function");
  });

  test("idempotent — second call returns no-op", () => {
    disposable = installSensitiveApiInterception();
    const second = installSensitiveApiInterception();
    expect(() => second.dispose()).not.toThrow();
    disposable.dispose();
    disposable = installSensitiveApiInterception();
  });

  test("window.print denied → confirm shows print dialog message", () => {
    disposable = installSensitiveApiInterception();
    const confirmSpy = spyOn(window, "confirm").mockReturnValue(false);
    window.print();
    expect(confirmSpy).toHaveBeenCalled();
    // Verify the confirm message mentions "print".
    const msg = confirmSpy.mock.calls[0]![0] as string;
    expect(msg.toLowerCase()).toContain("print");
    confirmSpy.mockRestore();
  });

  test("window.print allowed → confirm shows print dialog message", () => {
    disposable = installSensitiveApiInterception();
    const confirmSpy = spyOn(window, "confirm").mockReturnValue(true);
    window.print();
    expect(confirmSpy).toHaveBeenCalled();
    const msg = confirmSpy.mock.calls[0]![0] as string;
    expect(msg.toLowerCase()).toContain("print");
    confirmSpy.mockRestore();
  });

  test("dispose resets installed flag", () => {
    disposable = installSensitiveApiInterception();
    disposable.dispose();
    // Can re-install after dispose.
    disposable = installSensitiveApiInterception();
    expect(typeof disposable.dispose).toBe("function");
  });

  test("clipboard.writeText gated by confirm — denied", async () => {
    if (!navigator.clipboard) return;
    disposable = installSensitiveApiInterception();
    const confirmSpy = spyOn(window, "confirm").mockReturnValue(false);
    try {
      await navigator.clipboard.writeText("secret");
      expect(true).toBe(false); // Should not reach.
    } catch (err) {
      expect((err as DOMException).name).toBe("NotAllowedError");
    }
    confirmSpy.mockRestore();
  });
});
