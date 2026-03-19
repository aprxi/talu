import { describe, test, expect, beforeEach } from "bun:test";
import { ModeManager } from "../../../src/kernel/ui/mode-manager.ts";
import { EventBusImpl } from "../../../src/kernel/system/event-bus.ts";
import { navigate } from "../../../src/kernel/system/router.ts";

describe("ModeManager", () => {
  let eventBus: EventBusImpl;

  beforeEach(() => {
    eventBus = new EventBusImpl();
    document.body.innerHTML = `
      <div id="activity-bar">
        <button class="activity-btn active" data-mode="chat">Chat</button>
        <button class="activity-btn" data-mode="settings">Settings</button>
      </div>
      <div class="app-layout" data-mode="chat">
        <div class="app-content">
          <div id="chat-mode" class="mode-view"></div>
          <div id="settings-mode" class="mode-view hidden"></div>
        </div>
      </div>
    `;
    navigate({ mode: "chat", sub: null, resource: null }, { replace: true });
  });

  test("initial mode from active button in DOM", () => {
    const mgr = new ModeManager(eventBus);
    expect(mgr.getActiveMode()).toBe("chat");
  });

  test("switchMode changes active mode", () => {
    const mgr = new ModeManager(eventBus);
    mgr.switchMode("settings");
    expect(mgr.getActiveMode()).toBe("settings");
  });

  test("switchMode is no-op for current mode", () => {
    const mgr = new ModeManager(eventBus);
    let emitted = false;
    eventBus.on("mode.changed", () => { emitted = true; });
    mgr.switchMode("chat");
    expect(emitted).toBe(false);
  });

  test("switchMode emits mode.changed event", () => {
    const mgr = new ModeManager(eventBus);
    let payload: unknown;
    eventBus.on("mode.changed", (data: unknown) => { payload = data; });
    mgr.switchMode("settings");
    expect(payload).toEqual({ from: "chat", to: "settings" });
  });

  test("switchMode updates activity bar button states", () => {
    const mgr = new ModeManager(eventBus);
    mgr.switchMode("settings");
    const btns = document.querySelectorAll(".activity-btn");
    expect(btns[0]!.classList.contains("active")).toBe(false);
    expect(btns[1]!.classList.contains("active")).toBe(true);
  });

  test("switchMode updates app-layout data-mode", () => {
    const mgr = new ModeManager(eventBus);
    mgr.switchMode("settings");
    const layout = document.querySelector(".app-layout")!;
    expect(layout.getAttribute("data-mode")).toBe("settings");
  });

  test("initFromRoute switches to a registered route mode", () => {
    const mgr = new ModeManager(eventBus);
    mgr.registerMode("settings", "Settings", "talu.settings");
    navigate({ mode: "settings", sub: null, resource: null }, { replace: true });
    const d = mgr.initFromRoute();
    expect(mgr.getActiveMode()).toBe("settings");
    d.dispose();
  });

  test("initFromRoute falls back to chat for unknown route modes", () => {
    document.querySelector('.activity-btn[data-mode="chat"]')?.classList.remove("active");
    document.querySelector('.activity-btn[data-mode="settings"]')?.classList.add("active");
    document.querySelector(".app-layout")?.setAttribute("data-mode", "settings");
    const mgr = new ModeManager(eventBus);
    mgr.registerMode("settings", "Settings", "talu.settings");
    navigate({ mode: "unknown", sub: null, resource: null }, { replace: true });
    const d = mgr.initFromRoute();
    expect(mgr.getActiveMode()).toBe("chat");
    d.dispose();
  });

  test("installActivityBarListeners handles clicks", () => {
    const mgr = new ModeManager(eventBus);
    mgr.registerMode("settings", "Settings", "talu.settings");
    const d = mgr.installActivityBarListeners();

    const settingsBtn = document.querySelectorAll(".activity-btn")[1]! as HTMLElement;
    settingsBtn.click();
    expect(mgr.getActiveMode()).toBe("settings");

    d.dispose();
  });

  test("installActivityBarListeners dispose stops handling", () => {
    const mgr = new ModeManager(eventBus);
    mgr.registerMode("settings", "Settings", "talu.settings");
    const d = mgr.installActivityBarListeners();
    d.dispose();

    const settingsBtn = document.querySelectorAll(".activity-btn")[1]! as HTMLElement;
    settingsBtn.click();
    expect(mgr.getActiveMode()).toBe("chat");
  });
});
