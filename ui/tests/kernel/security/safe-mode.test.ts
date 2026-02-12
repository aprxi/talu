import { describe, test, expect, beforeEach, spyOn } from "bun:test";
import { isSafeMode, setLoadingFlag, clearLoadingFlag } from "../../../src/kernel/security/safe-mode.ts";

const CRASH_FLAG = "talu.kernel.loading";

describe("safe-mode loading flag", () => {
  beforeEach(() => {
    sessionStorage.removeItem(CRASH_FLAG);
  });

  test("setLoadingFlag sets crash flag in sessionStorage", () => {
    setLoadingFlag();
    expect(sessionStorage.getItem(CRASH_FLAG)).toBe("crashed");
  });

  test("clearLoadingFlag removes crash flag", () => {
    setLoadingFlag();
    clearLoadingFlag();
    expect(sessionStorage.getItem(CRASH_FLAG)).toBeNull();
  });

  test("clearLoadingFlag is idempotent", () => {
    clearLoadingFlag();
    clearLoadingFlag();
    expect(sessionStorage.getItem(CRASH_FLAG)).toBeNull();
  });
});

describe("isSafeMode", () => {
  beforeEach(() => {
    sessionStorage.removeItem(CRASH_FLAG);
  });

  test("returns false by default", () => {
    // No query param, no shift key, no crash flag.
    expect(isSafeMode()).toBe(false);
  });

  test("returns true when crash flag is set", () => {
    const spy = spyOn(console, "warn").mockImplementation(() => {});
    sessionStorage.setItem(CRASH_FLAG, "crashed");
    expect(isSafeMode()).toBe(true);
    // Flag should be cleared after detection.
    expect(sessionStorage.getItem(CRASH_FLAG)).toBeNull();
    spy.mockRestore();
  });

  test("crash flag cleared after isSafeMode returns true", () => {
    const spy = spyOn(console, "warn").mockImplementation(() => {});
    sessionStorage.setItem(CRASH_FLAG, "crashed");
    isSafeMode(); // Consumes the flag.
    expect(isSafeMode()).toBe(false); // Second call: no flag.
    spy.mockRestore();
  });
});
