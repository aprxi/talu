import { describe, test, expect, beforeEach } from "bun:test";
import { ModeManager } from "../../../src/kernel/ui/mode-manager.ts";
import { EventBusImpl } from "../../../src/kernel/system/event-bus.ts";

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
    localStorage.removeItem("talu-last-active-mode");
  });

  test("initial mode from active button in DOM", () => {
    const mgr = new ModeManager(eventBus);
    expect(mgr.getActiveMode()).toBe("chat");
  });

  test("registerMode stores mode info", () => {
    const mgr = new ModeManager(eventBus);
    mgr.registerMode("chat", "Chat", "talu.chat");
    mgr.registerMode("settings", "Settings", "talu.settings");
    // No direct getter for modes, but restoreLastMode uses it.
    expect(mgr.getActiveMode()).toBe("chat");
  });

  test("switchMode changes active mode", () => {
    const mgr = new ModeManager(eventBus);
    mgr.registerMode("settings", "Settings", "talu.settings");
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

  test("switchMode persists to localStorage", () => {
    const mgr = new ModeManager(eventBus);
    mgr.switchMode("settings");
    expect(localStorage.getItem("talu-last-active-mode")).toBe("settings");
  });

  test("restoreLastMode restores saved mode", () => {
    const mgr = new ModeManager(eventBus);
    mgr.registerMode("settings", "Settings", "talu.settings");
    localStorage.setItem("talu-last-active-mode", "settings");
    mgr.restoreLastMode();
    expect(mgr.getActiveMode()).toBe("settings");
  });

  test("restoreLastMode ignores unknown mode", () => {
    const mgr = new ModeManager(eventBus);
    localStorage.setItem("talu-last-active-mode", "nonexistent");
    mgr.restoreLastMode();
    expect(mgr.getActiveMode()).toBe("chat");
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
    // Mode should NOT change after dispose.
    expect(mgr.getActiveMode()).toBe("chat");
  });
});
