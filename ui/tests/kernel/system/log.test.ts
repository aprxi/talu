import { describe, test, expect, spyOn } from "bun:test";
import { createLogger } from "../../../src/kernel/system/log.ts";

// log.ts checks `location.hostname` at module parse time.
// In HappyDOM test env, hostname is not localhost → production mode (async hashing).
// Production mode: strings are SHA-256 hashed, numbers/booleans/Errors pass through.

/** Flush the microtask queue so async logging completes. */
const flush = () => new Promise((r) => setTimeout(r, 10));

describe("createLogger (production mode)", () => {
  test("info logs with plugin prefix", async () => {
    const spy = spyOn(console, "info").mockImplementation(() => {});
    const log = createLogger("my.plugin");
    log.info("hello");
    await flush();
    expect(spy).toHaveBeenCalled();
    expect(spy.mock.calls[0]![0]).toBe("[my.plugin]");
    spy.mockRestore();
  });

  test("warn logs with plugin prefix", async () => {
    const spy = spyOn(console, "warn").mockImplementation(() => {});
    const log = createLogger("my.plugin");
    log.warn("caution");
    await flush();
    expect(spy).toHaveBeenCalled();
    expect(spy.mock.calls[0]![0]).toBe("[my.plugin]");
    spy.mockRestore();
  });

  test("error logs with plugin prefix", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const log = createLogger("my.plugin");
    log.error("failed");
    await flush();
    expect(spy).toHaveBeenCalled();
    expect(spy.mock.calls[0]![0]).toBe("[my.plugin]");
    spy.mockRestore();
  });

  test("strings are hashed in production mode", async () => {
    const spy = spyOn(console, "info").mockImplementation(() => {});
    const log = createLogger("test");
    log.info("secret data");
    await flush();
    const args = spy.mock.calls[0]!;
    // String args become "string(sha256-...)" format.
    expect(args[1]).toMatch(/^string\(sha256-[0-9a-f]{12}\)$/);
    spy.mockRestore();
  });

  test("numbers pass through unhashed", async () => {
    const spy = spyOn(console, "info").mockImplementation(() => {});
    const log = createLogger("test");
    log.info("msg", 42);
    await flush();
    const args = spy.mock.calls[0]!;
    expect(args[2]).toBe(42);
    spy.mockRestore();
  });

  test("booleans pass through unhashed", async () => {
    const spy = spyOn(console, "info").mockImplementation(() => {});
    const log = createLogger("test");
    log.info("msg", true);
    await flush();
    const args = spy.mock.calls[0]!;
    expect(args[2]).toBe(true);
    spy.mockRestore();
  });

  test("Error objects pass through unhashed", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const log = createLogger("test");
    const err = new Error("boom");
    log.error("msg", err);
    await flush();
    const args = spy.mock.calls[0]!;
    expect(args[2]).toBe(err);
    spy.mockRestore();
  });

  test("different loggers have different prefixes", async () => {
    const spy = spyOn(console, "info").mockImplementation(() => {});
    const log1 = createLogger("plugin.a");
    const log2 = createLogger("plugin.b");
    log1.info("msg1");
    log2.info("msg2");
    await flush();
    expect(spy.mock.calls[0]![0]).toBe("[plugin.a]");
    expect(spy.mock.calls[1]![0]).toBe("[plugin.b]");
    spy.mockRestore();
  });

  test("hash is deterministic — same input produces same hash", async () => {
    const spy = spyOn(console, "info").mockImplementation(() => {});
    const log = createLogger("test");
    log.info("deterministic input");
    log.info("deterministic input");
    await flush();
    const hash1 = spy.mock.calls[0]![1];
    const hash2 = spy.mock.calls[1]![1];
    expect(hash1).toBe(hash2);
    expect(hash1).toMatch(/^string\(sha256-[0-9a-f]{12}\)$/);
    spy.mockRestore();
  });

  test("different strings produce different hashes", async () => {
    const spy = spyOn(console, "info").mockImplementation(() => {});
    const log = createLogger("test");
    log.info("input A");
    log.info("input B");
    await flush();
    const hashA = spy.mock.calls[0]![1];
    const hashB = spy.mock.calls[1]![1];
    expect(hashA).not.toBe(hashB);
    spy.mockRestore();
  });
});
