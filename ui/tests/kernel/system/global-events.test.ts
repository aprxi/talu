import { describe, test, expect, spyOn } from "bun:test";
import { GlobalEventManager } from "../../../src/kernel/system/global-events.ts";

describe("GlobalEventManager", () => {
  test("onDocument registers and fires handler", () => {
    const mgr = new GlobalEventManager("test.plugin");
    let received = false;
    mgr.onDocument("click", () => { received = true; });
    document.dispatchEvent(new Event("click"));
    expect(received).toBe(true);
    mgr.dispose();
  });

  test("onWindow registers and fires handler", () => {
    const mgr = new GlobalEventManager("test.plugin");
    let received = false;
    mgr.onWindow("resize", () => { received = true; });
    window.dispatchEvent(new Event("resize"));
    expect(received).toBe(true);
    mgr.dispose();
  });

  test("dispose removes all listeners", () => {
    const mgr = new GlobalEventManager("test.plugin");
    let count = 0;
    mgr.onDocument("click", () => { count++; });
    mgr.onWindow("resize", () => { count++; });
    mgr.dispose();
    document.dispatchEvent(new Event("click"));
    window.dispatchEvent(new Event("resize"));
    expect(count).toBe(0);
  });

  test("individual dispose removes specific listener", () => {
    const mgr = new GlobalEventManager("test.plugin");
    let count = 0;
    const d = mgr.onDocument("click", () => { count++; });
    d.dispose();
    document.dispatchEvent(new Event("click"));
    expect(count).toBe(0);
    mgr.dispose();
  });

  test("handler error is caught and logged", () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const mgr = new GlobalEventManager("test.plugin");
    mgr.onDocument("click", () => { throw new Error("handler boom"); });
    document.dispatchEvent(new Event("click"));
    // Dispose ASAP — other test files share this document and their clicks
    // would trigger our throwing handler in the parallel bun test runner.
    mgr.dispose();
    expect(spy).toHaveBeenCalled();
    // Cross-file console.error calls may precede ours in the spy log;
    // search all calls instead of assuming index [0].
    const found = spy.mock.calls.some(
      (args) => typeof args[0] === "string" && args[0].includes("test.plugin"),
    );
    expect(found).toBe(true);
    spy.mockRestore();
  });
});
