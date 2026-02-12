import { describe, test, expect, spyOn } from "bun:test";
import { HashRouterImpl } from "../../../src/kernel/system/router.ts";

describe("HashRouterImpl", () => {
  test("getHash returns empty string when no hash set", () => {
    const router = new HashRouterImpl("test.router.empty");
    expect(router.getHash()).toBe("");
  });

  test("setHash + getHash roundtrip", () => {
    const router = new HashRouterImpl("test.router.roundtrip");
    router.setHash("settings/general");
    expect(router.getHash()).toBe("settings/general");
  });

  test("plugins have isolated hash segments", () => {
    const routerA = new HashRouterImpl("test.router.a");
    const routerB = new HashRouterImpl("test.router.b");
    routerA.setHash("page-a");
    routerB.setHash("page-b");
    expect(routerA.getHash()).toBe("page-a");
    expect(routerB.getHash()).toBe("page-b");
  });

  test("setHash notifies own onHashChange listeners", () => {
    const router = new HashRouterImpl("test.router.notify");
    let received: string | undefined;
    router.onHashChange((hash) => { received = hash; });
    router.setHash("new-path");
    expect(received).toBe("new-path");
  });

  test("onHashChange dispose stops notifications", () => {
    const router = new HashRouterImpl("test.router.dispose");
    let callCount = 0;
    const d = router.onHashChange(() => { callCount++; });
    router.setHash("a");
    expect(callCount).toBe(1);
    d.dispose();
    router.setHash("b");
    expect(callCount).toBe(1);
  });

  test("setting empty hash clears the segment", () => {
    const router = new HashRouterImpl("test.router.clear");
    router.setHash("something");
    router.setHash("");
    expect(router.getHash()).toBe("");
  });

  test("setHash with URL-encoded special characters round-trips", () => {
    const router = new HashRouterImpl("test.router.encoded");
    router.setHash("path/with spaces & symbols=yes");
    expect(router.getHash()).toBe("path/with spaces & symbols=yes");
  });

  test("setHash with semicolons in value does not corrupt other segments", () => {
    const routerA = new HashRouterImpl("test.router.semi.a");
    const routerB = new HashRouterImpl("test.router.semi.b");
    // Semicolons inside a segment value are URL-encoded by serializeCompositeHash.
    routerA.setHash("val;with;semis");
    routerB.setHash("clean");
    expect(routerA.getHash()).toBe("val;with;semis");
    expect(routerB.getHash()).toBe("clean");
  });

  test("setHash with colons in value round-trips", () => {
    const router = new HashRouterImpl("test.router.colon");
    router.setHash("key:value:pair");
    expect(router.getHash()).toBe("key:value:pair");
  });

  test("setHash with history: 'push' calls pushState", () => {
    const router = new HashRouterImpl("test.router.push");
    const spy = spyOn(window.history, "pushState");
    router.setHash("pushed", { history: "push" });
    expect(spy).toHaveBeenCalledTimes(1);
    expect(router.getHash()).toBe("pushed");
    spy.mockRestore();
  });

  test("setHash with history: 'replace' calls replaceState (default)", () => {
    const router = new HashRouterImpl("test.router.replace");
    const spy = spyOn(window.history, "replaceState");
    router.setHash("replaced", { history: "replace" });
    expect(spy).toHaveBeenCalled();
    expect(router.getHash()).toBe("replaced");
    spy.mockRestore();
  });

  test("setHash without options defaults to replaceState", () => {
    const router = new HashRouterImpl("test.router.default");
    const pushSpy = spyOn(window.history, "pushState");
    const replaceSpy = spyOn(window.history, "replaceState");
    router.setHash("default-behavior");
    // Default is "replace", not "push".
    expect(pushSpy).not.toHaveBeenCalled();
    expect(replaceSpy).toHaveBeenCalled();
    pushSpy.mockRestore();
    replaceSpy.mockRestore();
  });

  test("handler error does not break other listeners", () => {
    const router = new HashRouterImpl("test.router.error");
    let secondCalled = false;
    router.onHashChange(() => { throw new Error("handler boom"); });
    router.onHashChange(() => { secondCalled = true; });
    // setHash notifies own listeners — first throws, second should still fire.
    const spy = spyOn(console, "error").mockImplementation(() => {});
    router.setHash("trigger");
    expect(secondCalled).toBe(true);
    spy.mockRestore();
  });
});
