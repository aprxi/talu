import { describe, test, expect, spyOn } from "bun:test";
import {
  parseHash,
  serializeRoute,
  navigate,
  onRouteChange,
  initRouter,
  getCurrentRoute,
} from "../../../src/kernel/system/router.ts";

describe("router helpers", () => {
  test("parseHash normalizes empty and nested routes", () => {
    expect(parseHash("")).toEqual({ mode: "chat", sub: null, resource: null });
    expect(parseHash("#/settings/appearance/palette")).toEqual({
      mode: "settings",
      sub: "appearance",
      resource: "palette",
    });
  });

  test("serializeRoute encodes route segments", () => {
    expect(serializeRoute({ mode: "routing", sub: "terminal & logs", resource: "x/y" })).toBe(
      "#/routing/terminal%20%26%20logs/x%2Fy",
    );
  });

  test("navigate uses pushState by default and notifies listeners", () => {
    const pushSpy = spyOn(window.history, "pushState");
    navigate({ mode: "chat", sub: null, resource: null }, { replace: true });
    const events: Array<{ route: unknown; previous: unknown }> = [];
    const d = onRouteChange((route, previous) => {
      events.push({ route, previous });
    });
    const beforePush = pushSpy.mock.calls.length;
    navigate({ mode: "settings", sub: "appearance", resource: null });

    expect(pushSpy.mock.calls.length - beforePush).toBe(1);
    expect(getCurrentRoute()).toEqual({ mode: "settings", sub: "appearance", resource: null });
    expect(events).toEqual([
      {
        route: { mode: "settings", sub: "appearance", resource: null },
        previous: { mode: "chat", sub: null, resource: null },
      },
    ]);

    d.dispose();
    pushSpy.mockRestore();
  });

  test("navigate with replace uses replaceState", () => {
    const replaceSpy = spyOn(window.history, "replaceState");
    navigate({ mode: "chat", sub: null, resource: null }, { replace: true });
    const beforeReplace = replaceSpy.mock.calls.length;
    navigate({ mode: "files", sub: "archived", resource: null }, { replace: true });

    expect(replaceSpy.mock.calls.length - beforeReplace).toBe(1);
    expect(getCurrentRoute()).toEqual({ mode: "files", sub: "archived", resource: null });

    replaceSpy.mockRestore();
  });

  test("initRouter reacts to hashchange events", () => {
    navigate({ mode: "chat", sub: null, resource: null }, { replace: true });
    const received: Array<{ mode: string; sub: string | null; resource: string | null }> = [];
    const d = onRouteChange((route) => {
      received.push(route);
    });
    const router = initRouter();

    window.location.hash = "#/settings/general";
    window.dispatchEvent(new Event("hashchange"));

    expect(received).toEqual([{ mode: "settings", sub: "general", resource: null }]);
    expect(getCurrentRoute()).toEqual({ mode: "settings", sub: "general", resource: null });

    router.dispose();
    d.dispose();
  });
});
