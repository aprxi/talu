import { describe, test, expect } from "bun:test";
import { PluginStatusImpl } from "../../../src/kernel/core/status.ts";

describe("PluginStatusImpl", () => {
  test("starts not busy", () => {
    const status = new PluginStatusImpl("test.plugin");
    expect(status.isBusy).toBe(false);
    expect(status.message).toBeUndefined();
  });

  test("setBusy sets busy flag", () => {
    const status = new PluginStatusImpl("test.plugin");
    status.setBusy();
    expect(status.isBusy).toBe(true);
  });

  test("setBusy with message", () => {
    const status = new PluginStatusImpl("test.plugin");
    status.setBusy("Loading data...");
    expect(status.isBusy).toBe(true);
    expect(status.message).toBe("Loading data...");
  });

  test("setReady clears busy and message", () => {
    const status = new PluginStatusImpl("test.plugin");
    status.setBusy("Working...");
    status.setReady();
    expect(status.isBusy).toBe(false);
    expect(status.message).toBeUndefined();
  });

  test("setBusy overwrites previous message", () => {
    const status = new PluginStatusImpl("test.plugin");
    status.setBusy("first");
    status.setBusy("second");
    expect(status.message).toBe("second");
  });

  test("setBusy without message clears previous message", () => {
    const status = new PluginStatusImpl("test.plugin");
    status.setBusy("has message");
    status.setBusy();
    expect(status.isBusy).toBe(true);
    expect(status.message).toBeUndefined();
  });
});
