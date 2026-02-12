import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { installCommandPalette, type CommandPaletteHandle } from "../../../src/kernel/ui/command-palette.ts";
import { CommandRegistryImpl } from "../../../src/kernel/registries/commands.ts";

describe("installCommandPalette", () => {
  let registry: CommandRegistryImpl;
  let handle: CommandPaletteHandle;

  beforeEach(() => {
    document.body.innerHTML = "";
    registry = new CommandRegistryImpl();
    registry.registerScoped("talu.test", "talu.test.hello", () => {}, { label: "Hello" });
    registry.registerScoped("talu.test", "talu.test.world", () => {}, { label: "World" });
  });

  afterEach(() => {
    handle?.dispose();
  });

  test("returns handle with open and dispose", () => {
    handle = installCommandPalette(registry);
    expect(typeof handle.open).toBe("function");
    expect(typeof handle.dispose).toBe("function");
  });

  test("open creates overlay in DOM", () => {
    handle = installCommandPalette(registry);
    handle.open();
    expect(document.getElementById("command-palette-overlay")).not.toBeNull();
  });

  test("dispose removes overlay and listener", () => {
    handle = installCommandPalette(registry);
    handle.open();
    handle.dispose();
    expect(document.getElementById("command-palette-overlay")).toBeNull();
  });

  test("opening twice is idempotent", () => {
    handle = installCommandPalette(registry);
    handle.open();
    handle.open();
    const overlays = document.querySelectorAll("#command-palette-overlay");
    expect(overlays.length).toBe(1);
  });

  test("lists registered commands", () => {
    handle = installCommandPalette(registry);
    handle.open();
    const overlay = document.getElementById("command-palette-overlay")!;
    // The list is the second child of the container div (after input).
    const container = overlay.firstElementChild!;
    const list = container.children[1]!;
    expect(list.children.length).toBeGreaterThanOrEqual(2);
  });

  test("has search input", () => {
    handle = installCommandPalette(registry);
    handle.open();
    const overlay = document.getElementById("command-palette-overlay")!;
    const input = overlay.querySelector("input");
    expect(input).not.toBeNull();
    expect(input!.placeholder).toBe("Type a command...");
  });

  test("escape closes palette", () => {
    handle = installCommandPalette(registry);
    handle.open();
    const overlay = document.getElementById("command-palette-overlay")!;
    const input = overlay.querySelector("input")!;
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    expect(document.getElementById("command-palette-overlay")).toBeNull();
  });

  test("clicking overlay background closes palette", () => {
    handle = installCommandPalette(registry);
    handle.open();
    const overlay = document.getElementById("command-palette-overlay")!;
    // Simulate click on the overlay itself (not on the dialog).
    overlay.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(document.getElementById("command-palette-overlay")).toBeNull();
  });

  test("clicking a command row executes the command and closes palette", () => {
    let executed = false;
    registry.registerScoped("talu.test", "talu.test.action", () => { executed = true; }, { label: "Action" });
    handle = installCommandPalette(registry);
    handle.open();
    const overlay = document.getElementById("command-palette-overlay")!;
    const container = overlay.firstElementChild!;
    const list = container.children[1]!;
    // Find the row for "Action" and click it.
    const rows = Array.from(list.children) as HTMLElement[];
    const actionRow = rows.find((r) => r.textContent?.includes("Action"));
    expect(actionRow).not.toBeUndefined();
    actionRow!.click();
    expect(executed).toBe(true);
    // Palette should close after execution.
    expect(document.getElementById("command-palette-overlay")).toBeNull();
  });

  test("search input filters commands", () => {
    handle = installCommandPalette(registry);
    handle.open();
    const overlay = document.getElementById("command-palette-overlay")!;
    const input = overlay.querySelector("input")!;
    const container = overlay.firstElementChild!;
    const list = container.children[1]!;
    // Initially shows all commands (at least 2).
    const initialCount = list.children.length;
    expect(initialCount).toBeGreaterThanOrEqual(2);
    // Type "hello" to filter.
    input.value = "hello";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    // Should only show the "Hello" command.
    expect(list.children.length).toBe(1);
    expect(list.children[0]!.textContent).toContain("Hello");
  });

  test("Enter key on selected command executes it", () => {
    let executed = false;
    registry.registerScoped("talu.test", "talu.test.enter", () => { executed = true; }, { label: "Enter Cmd" });
    handle = installCommandPalette(registry);
    handle.open();
    const overlay = document.getElementById("command-palette-overlay")!;
    const input = overlay.querySelector("input")!;
    // Filter to just our command.
    input.value = "enter cmd";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    // Press Enter.
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    expect(executed).toBe(true);
  });
});
