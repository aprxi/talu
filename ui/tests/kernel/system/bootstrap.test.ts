import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { getBootstrapWorkdir } from "../../../src/kernel/system/bootstrap.ts";

beforeEach(() => {
  (globalThis as { __TALU_BOOTSTRAP__?: unknown }).__TALU_BOOTSTRAP__ = undefined;
});

afterEach(() => {
  (globalThis as { __TALU_BOOTSTRAP__?: unknown }).__TALU_BOOTSTRAP__ = undefined;
});

describe("getBootstrapWorkdir", () => {
  test("returns null when bootstrap payload is missing", () => {
    expect(getBootstrapWorkdir()).toBeNull();
  });

  test("returns null when bootstrap payload is not an object", () => {
    (globalThis as { __TALU_BOOTSTRAP__?: unknown }).__TALU_BOOTSTRAP__ = "not-an-object";

    expect(getBootstrapWorkdir()).toBeNull();
  });

  test("returns null for missing, null, empty, or non-string workdir values", () => {
    const cases = [
      {},
      { workdir: null },
      { workdir: "" },
      { workdir: 42 },
      { workdir: false },
    ];

    for (const value of cases) {
      (globalThis as { __TALU_BOOTSTRAP__?: unknown }).__TALU_BOOTSTRAP__ = value;
      expect(getBootstrapWorkdir()).toBeNull();
    }
  });

  test("returns the configured bootstrap workdir string", () => {
    (globalThis as { __TALU_BOOTSTRAP__?: unknown }).__TALU_BOOTSTRAP__ = {
      workdir: "/tmp/project",
    };

    expect(getBootstrapWorkdir()).toBe("/tmp/project");
  });

  test("ignores unrelated bootstrap fields when reading workdir", () => {
    (globalThis as { __TALU_BOOTSTRAP__?: unknown }).__TALU_BOOTSTRAP__ = {
      workdir: "/repo/ui",
      other: "value",
      nested: { ok: true },
    };

    expect(getBootstrapWorkdir()).toBe("/repo/ui");
  });
});
