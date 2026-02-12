import { describe, test, expect } from "bun:test";
import { freezeIntrinsics } from "../../../src/kernel/security/realm-integrity.ts";

describe("freezeIntrinsics", () => {
  // NOTE: We ONLY test the disabled path here. Calling freezeIntrinsics()
  // with enabled=true permanently freezes Object.prototype etc., which would
  // break the test runner and all subsequent tests in this process.

  test("disabled: returns no-op with empty checkTamper", () => {
    const result = freezeIntrinsics({ enabled: false });
    expect(result.checkTamper()).toEqual([]);
    expect(() => result.dispose()).not.toThrow();
  });

  test("disabled: dispose is callable multiple times", () => {
    const result = freezeIntrinsics({ enabled: false });
    result.dispose();
    result.dispose();
    expect(result.checkTamper()).toEqual([]);
  });

  test("returns object with correct shape", () => {
    const result = freezeIntrinsics({ enabled: false });
    expect(typeof result.dispose).toBe("function");
    expect(typeof result.checkTamper).toBe("function");
  });
});
