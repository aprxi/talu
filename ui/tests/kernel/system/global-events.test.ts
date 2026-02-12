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
    expect(spy).toHaveBeenCalled();
    expect(spy.mock.calls[0][0]).toContain("test.plugin");
    spy.mockRestore();
    mgr.dispose();
  });
});
