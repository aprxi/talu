import { describe, test, expect } from "bun:test";
import { createAssetResolver } from "../../../src/kernel/system/assets.ts";

describe("createAssetResolver", () => {
  test("resolves simple path", () => {
    const resolver = createAssetResolver("my.plugin");
    expect(resolver.getUrl("icon.png")).toBe("/v1/plugins/my.plugin/icon.png");
  });

  test("strips leading ./", () => {
    const resolver = createAssetResolver("my.plugin");
    expect(resolver.getUrl("./styles/main.css")).toBe("/v1/plugins/my.plugin/styles/main.css");
  });

  test("strips leading /", () => {
    const resolver = createAssetResolver("my.plugin");
    expect(resolver.getUrl("/images/logo.svg")).toBe("/v1/plugins/my.plugin/images/logo.svg");
  });

  test("encodes plugin ID", () => {
    const resolver = createAssetResolver("my plugin/special");
    expect(resolver.getUrl("file.txt")).toBe("/v1/plugins/my%20plugin%2Fspecial/file.txt");
  });

  test("handles nested paths", () => {
    const resolver = createAssetResolver("test");
    expect(resolver.getUrl("a/b/c/d.js")).toBe("/v1/plugins/test/a/b/c/d.js");
  });

  test("handles empty path", () => {
    const resolver = createAssetResolver("test");
    expect(resolver.getUrl("")).toBe("/v1/plugins/test/");
  });
});
