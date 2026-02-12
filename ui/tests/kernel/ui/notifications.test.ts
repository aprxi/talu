import { describe, test, expect, beforeEach } from "bun:test";
import { NotificationsImpl } from "../../../src/kernel/ui/notifications.ts";

describe("NotificationsImpl", () => {
  beforeEach(() => {
    document.body.innerHTML = `<div id="toast-container"></div>`;
  });

  test("info toast contains message text and ARIA role", () => {
    const notif = new NotificationsImpl();
    notif.info("Test message");
    const container = document.getElementById("toast-container")!;
    expect(container.children.length).toBe(1);
    const toast = container.children[0] as HTMLElement;
    expect(toast.textContent).toBe("Test message");
    expect(toast.getAttribute("role")).toBe("alert");
    expect(toast.getAttribute("aria-live")).toBe("assertive");
  });

  test("success toast contains message text", () => {
    const notif = new NotificationsImpl();
    notif.success("Saved!");
    const container = document.getElementById("toast-container")!;
    const toast = container.children[0] as HTMLElement;
    expect(toast.textContent).toBe("Saved!");
  });

  test("warning toast contains message text", () => {
    const notif = new NotificationsImpl();
    notif.warning("Be careful");
    const container = document.getElementById("toast-container")!;
    const toast = container.children[0] as HTMLElement;
    expect(toast.textContent).toBe("Be careful");
  });

  test("error toast contains message text", () => {
    const notif = new NotificationsImpl();
    notif.error("Something failed");
    const container = document.getElementById("toast-container")!;
    const toast = container.children[0] as HTMLElement;
    expect(toast.textContent).toBe("Something failed");
  });

  test("multiple notifications stack with correct content", () => {
    const notif = new NotificationsImpl();
    notif.info("one");
    notif.warning("two");
    notif.error("three");
    const container = document.getElementById("toast-container")!;
    expect(container.children.length).toBe(3);
    expect((container.children[0] as HTMLElement).textContent).toBe("one");
    expect((container.children[1] as HTMLElement).textContent).toBe("two");
    expect((container.children[2] as HTMLElement).textContent).toBe("three");
  });

  test("no-op when toast container missing", () => {
    document.body.innerHTML = "";
    const notif = new NotificationsImpl();
    expect(() => notif.info("message")).not.toThrow();
  });
});
