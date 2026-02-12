import { describe, test, expect, beforeEach } from "bun:test";
import { StandardDialogsImpl } from "../../../src/kernel/ui/dialogs.ts";

describe("StandardDialogsImpl", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
  });

  test("confirm resolves true on OK click", async () => {
    const dialogs = new StandardDialogsImpl("TestPlugin");
    const promise = dialogs.confirm({ title: "Delete?", message: "Are you sure?" });

    // Dialog should be in DOM.
    const overlay = document.getElementById("kernel-dialog-overlay");
    expect(overlay).not.toBeNull();

    // Click OK.
    const buttons = overlay!.querySelectorAll("button");
    const okBtn = [...buttons].find((b) => b.textContent === "Confirm")!;
    okBtn.click();

    expect(await promise).toBe(true);
  });

  test("confirm resolves false on Cancel click", async () => {
    const dialogs = new StandardDialogsImpl("TestPlugin");
    const promise = dialogs.confirm({ title: "Delete?", message: "Sure?" });

    const overlay = document.getElementById("kernel-dialog-overlay")!;
    const cancelBtn = [...overlay.querySelectorAll("button")].find((b) => b.textContent === "Cancel")!;
    cancelBtn.click();

    expect(await promise).toBe(false);
  });

  test("confirm resolves false on Escape", async () => {
    const dialogs = new StandardDialogsImpl("TestPlugin");
    const promise = dialogs.confirm({ title: "T", message: "M" });

    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));

    expect(await promise).toBe(false);
  });

  test("destructive confirm prepends plugin name to title", async () => {
    const dialogs = new StandardDialogsImpl("MyPlugin");
    const promise = dialogs.confirm({ title: "Delete all", message: "Sure?", destructive: true });

    const overlay = document.getElementById("kernel-dialog-overlay")!;
    const title = overlay.querySelector("h3")!;
    expect(title.textContent).toBe("MyPlugin: Delete all");

    // Has red "Delete" button.
    const okBtn = [...overlay.querySelectorAll("button")].find((b) => b.textContent === "Delete")!;
    expect(okBtn).toBeDefined();
    okBtn.click();

    expect(await promise).toBe(true);
  });

  test("alert resolves on OK click", async () => {
    const dialogs = new StandardDialogsImpl("TestPlugin");
    const promise = dialogs.alert({ title: "Info", message: "Done." });

    const overlay = document.getElementById("kernel-dialog-overlay")!;
    const okBtn = [...overlay.querySelectorAll("button")].find((b) => b.textContent === "OK")!;
    expect(okBtn).toBeDefined();
    // No Cancel button for alert.
    const cancelBtn = [...overlay.querySelectorAll("button")].find((b) => b.textContent === "Cancel");
    expect(cancelBtn).toBeUndefined();

    okBtn.click();
    await promise;
  });

  test("prompt returns input value on OK", async () => {
    const dialogs = new StandardDialogsImpl("TestPlugin");
    const promise = dialogs.prompt({ title: "Name", message: "Enter name", defaultValue: "Alice" });

    const overlay = document.getElementById("kernel-dialog-overlay")!;
    const input = overlay.querySelector<HTMLInputElement>("#kernel-dialog-input")!;
    expect(input.value).toBe("Alice");
    input.value = "Bob";

    const okBtn = [...overlay.querySelectorAll("button")].find((b) => b.textContent === "OK")!;
    okBtn.click();

    expect(await promise).toBe("Bob");
  });

  test("prompt returns null on Cancel", async () => {
    const dialogs = new StandardDialogsImpl("TestPlugin");
    const promise = dialogs.prompt({ title: "Name", message: "Enter" });

    const overlay = document.getElementById("kernel-dialog-overlay")!;
    const cancelBtn = [...overlay.querySelectorAll("button")].find((b) => b.textContent === "Cancel")!;
    cancelBtn.click();

    expect(await promise).toBeNull();
  });

  test("select returns item id on click", async () => {
    const dialogs = new StandardDialogsImpl("TestPlugin");
    const promise = dialogs.select({
      title: "Choose",
      items: [
        { id: "opt1", label: "Option 1" },
        { id: "opt2", label: "Option 2" },
      ],
    });

    const overlay = document.getElementById("kernel-dialog-overlay")!;
    const selectList = overlay.querySelector("#kernel-dialog-select-list")!;
    const buttons = selectList.querySelectorAll("button");
    expect(buttons.length).toBe(2);
    expect(buttons[0]!.textContent).toBe("Option 1");

    buttons[1]!.click();
    expect(await promise).toBe("opt2");
  });

  test("select returns null on Cancel", async () => {
    const dialogs = new StandardDialogsImpl("TestPlugin");
    const promise = dialogs.select({
      title: "Choose",
      items: [{ id: "a", label: "A" }],
    });

    const overlay = document.getElementById("kernel-dialog-overlay")!;
    const cancelBtn = [...overlay.querySelectorAll("button")].find((b) => b.textContent === "Cancel")!;
    cancelBtn.click();

    expect(await promise).toBeNull();
  });

  test("overlay is removed after dialog closes", async () => {
    const dialogs = new StandardDialogsImpl("TestPlugin");
    const promise = dialogs.confirm({ title: "T", message: "M" });

    expect(document.getElementById("kernel-dialog-overlay")).not.toBeNull();

    const overlay = document.getElementById("kernel-dialog-overlay")!;
    const okBtn = [...overlay.querySelectorAll("button")].find((b) => b.textContent === "Confirm")!;
    okBtn.click();
    await promise;

    expect(document.getElementById("kernel-dialog-overlay")).toBeNull();
  });

  test("FIFO queue shows dialogs one at a time", async () => {
    const dialogs = new StandardDialogsImpl("TestPlugin");
    const p1 = dialogs.confirm({ title: "First", message: "1" });
    const p2 = dialogs.confirm({ title: "Second", message: "2" });

    // Only first dialog visible.
    let overlay = document.getElementById("kernel-dialog-overlay")!;
    expect(overlay.querySelector("h3")!.textContent).toBe("First");

    // Resolve first.
    let okBtn = [...overlay.querySelectorAll("button")].find((b) => b.textContent === "Confirm")!;
    okBtn.click();
    await p1;

    // Second dialog should now be visible.
    overlay = document.getElementById("kernel-dialog-overlay")!;
    expect(overlay.querySelector("h3")!.textContent).toBe("Second");

    okBtn = [...overlay.querySelectorAll("button")].find((b) => b.textContent === "Confirm")!;
    okBtn.click();
    await p2;
  });
});
