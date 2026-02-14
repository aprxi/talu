import { describe, test, expect } from "bun:test";
import {
  checkCapabilities,
  getKnownCapabilities,
} from "../../../src/kernel/security/capabilities.ts";

describe("checkCapabilities", () => {
  test("undefined requirements → satisfied", () => {
    const result = checkCapabilities(undefined);
    expect(result.satisfied).toBe(true);
    expect(result.unsatisfied).toHaveLength(0);
  });

  test("empty array → satisfied", () => {
    const result = checkCapabilities([]);
    expect(result.satisfied).toBe(true);
    expect(result.unsatisfied).toHaveLength(0);
  });

  test("known capabilities → satisfied", () => {
    const result = checkCapabilities(["hooks", "tools", "storage"]);
    expect(result.satisfied).toBe(true);
    expect(result.unsatisfied).toHaveLength(0);
  });

  test("unknown capability → unsatisfied with list", () => {
    const result = checkCapabilities(["teleport"]);
    expect(result.satisfied).toBe(false);
    expect(result.unsatisfied).toEqual(["teleport"]);
  });

  test("mix of known and unknown → unsatisfied with unknown listed", () => {
    const result = checkCapabilities(["hooks", "teleport", "storage", "warp"]);
    expect(result.satisfied).toBe(false);
    expect(result.unsatisfied).toContain("teleport");
    expect(result.unsatisfied).toContain("warp");
    expect(result.unsatisfied).not.toContain("hooks");
    expect(result.unsatisfied).not.toContain("storage");
  });
});

describe("getKnownCapabilities", () => {
  test("returns a non-empty array", () => {
    const caps = getKnownCapabilities();
    expect(caps.length).toBeGreaterThan(0);
  });

  test("includes core capabilities", () => {
    const caps = getKnownCapabilities();
    expect(caps).toContain("hooks");
    expect(caps).toContain("tools");
    expect(caps).toContain("commands");
    expect(caps).toContain("storage");
    expect(caps).toContain("upload");
  });
});
