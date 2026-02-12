import { describe, test, expect, spyOn } from "bun:test";
import { RendererRegistryImpl } from "../../../src/kernel/registries/renderers.ts";
import type { ContentPart, MessageRenderer, RendererInstance } from "../../../src/kernel/types.ts";

function textPart(text: string): ContentPart {
  return { id: "p1", type: "text", text } as ContentPart;
}

function toolResultPart(data: string): ContentPart {
  return { id: "p1", type: "tool_result", data } as ContentPart;
}

/** Create a simple renderer that claims a fixed score. */
function simpleRenderer(score: number, overrides: Partial<MessageRenderer> = {}): MessageRenderer {
  return {
    canRender: () => score,
    mount: (container, part) => {
      container.textContent = `rendered:${(part as { text?: string }).text ?? ""}`;
      return {
        update: () => true,
        unmount: () => { container.textContent = ""; },
      };
    },
    ...overrides,
  };
}

describe("RendererRegistryImpl registration", () => {
  test("register and dispose", () => {
    const reg = new RendererRegistryImpl();
    const d = reg.registerScoped("p", 0, simpleRenderer(10));
    expect(() => d.dispose()).not.toThrow();
  });

  test("disposed renderer no longer participates in scoring", () => {
    const reg = new RendererRegistryImpl();
    const d = reg.registerScoped("p", 0, simpleRenderer(10));
    const container = document.createElement("div");
    reg.mountPart("p1", container, textPart("hello"));
    expect(container.textContent).toBe("rendered:hello");
    d.dispose();
    reg.mountPart("p2", container, textPart("world"));
    // With no renderers, falls back to default (textContent = text).
    expect(container.textContent).toBe("world");
  });
});

describe("RendererRegistryImpl scoring", () => {
  test("highest score wins", () => {
    const reg = new RendererRegistryImpl();
    reg.registerScoped("low", 0, simpleRenderer(1, {
      mount: (c) => { c.textContent = "low"; return { update: () => true, unmount() {} }; },
    }));
    reg.registerScoped("high", 0, simpleRenderer(10, {
      mount: (c) => { c.textContent = "high"; return { update: () => true, unmount() {} }; },
    }));
    const container = document.createElement("div");
    reg.mountPart("p1", container, textPart("test"));
    expect(container.textContent).toBe("high");
  });

  test("score tie → higher manifest priority wins", () => {
    const reg = new RendererRegistryImpl();
    reg.registerScoped("low-pri", 1, simpleRenderer(10, {
      mount: (c) => { c.textContent = "low-pri"; return { update: () => true, unmount() {} }; },
    }));
    reg.registerScoped("high-pri", 10, simpleRenderer(10, {
      mount: (c) => { c.textContent = "high-pri"; return { update: () => true, unmount() {} }; },
    }));
    const container = document.createElement("div");
    reg.mountPart("p1", container, textPart("test"));
    expect(container.textContent).toBe("high-pri");
  });

  test("score tie + priority tie → lexicographic pluginId wins", () => {
    const reg = new RendererRegistryImpl();
    reg.registerScoped("beta", 0, simpleRenderer(10, {
      mount: (c) => { c.textContent = "beta"; return { update: () => true, unmount() {} }; },
    }));
    reg.registerScoped("alpha", 0, simpleRenderer(10, {
      mount: (c) => { c.textContent = "alpha"; return { update: () => true, unmount() {} }; },
    }));
    const container = document.createElement("div");
    reg.mountPart("p1", container, textPart("test"));
    expect(container.textContent).toBe("alpha");
  });

  test("canRender returning 0 or false → excluded", () => {
    const reg = new RendererRegistryImpl();
    reg.registerScoped("zero", 0, simpleRenderer(0));
    const container = document.createElement("div");
    reg.mountPart("p1", container, textPart("hello"));
    // Falls back to default.
    expect(container.textContent).toBe("hello");
  });

  test("no renderers → default text rendering", () => {
    const reg = new RendererRegistryImpl();
    const container = document.createElement("div");
    reg.mountPart("p1", container, textPart("fallback"));
    expect(container.textContent).toBe("fallback");
  });
});

describe("RendererRegistryImpl pre-processors", () => {
  test("pre-processor transforms text", () => {
    const reg = new RendererRegistryImpl();
    reg.registerPreProcessorScoped("p", (text) => text.toUpperCase());
    expect(reg.applyPreProcessors("hello")).toBe("HELLO");
  });

  test("multiple pre-processors chain", () => {
    const reg = new RendererRegistryImpl();
    reg.registerPreProcessorScoped("a", (text) => text + "-a");
    reg.registerPreProcessorScoped("b", (text) => text + "-b");
    expect(reg.applyPreProcessors("start")).toBe("start-a-b");
  });

  test("pre-processor error is skipped", () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const reg = new RendererRegistryImpl();
    reg.registerPreProcessorScoped("bad", () => { throw new Error("boom"); });
    reg.registerPreProcessorScoped("good", (text) => text + "-ok");
    expect(reg.applyPreProcessors("start")).toBe("start-ok");
    spy.mockRestore();
  });

  test("pre-processor dispose removes it", () => {
    const reg = new RendererRegistryImpl();
    const d = reg.registerPreProcessorScoped("p", (text) => text + "-added");
    expect(reg.applyPreProcessors("x")).toBe("x-added");
    d.dispose();
    expect(reg.applyPreProcessors("x")).toBe("x");
  });
});

describe("RendererRegistryImpl lifecycle", () => {
  test("mountPart unmounts previous renderer for same partId", () => {
    const reg = new RendererRegistryImpl();
    let unmountCount = 0;
    reg.registerScoped("p", 0, {
      canRender: () => 10,
      mount: (c) => {
        c.textContent = "mounted";
        return {
          update: () => true,
          unmount: () => { unmountCount++; },
        };
      },
    });
    const container = document.createElement("div");
    reg.mountPart("p1", container, textPart("first"));
    reg.mountPart("p1", container, textPart("second"));
    expect(unmountCount).toBe(1);
  });

  test("unmountPart cleans up", () => {
    const reg = new RendererRegistryImpl();
    let unmounted = false;
    reg.registerScoped("p", 0, {
      canRender: () => 10,
      mount: () => ({
        update: () => true,
        unmount: () => { unmounted = true; },
      }),
    });
    const container = document.createElement("div");
    reg.mountPart("p1", container, textPart("test"));
    reg.unmountPart("p1");
    expect(unmounted).toBe(true);
  });

  test("unmountAll cleans up all mounts", () => {
    const reg = new RendererRegistryImpl();
    let unmountCount = 0;
    reg.registerScoped("p", 0, {
      canRender: () => 10,
      mount: () => ({
        update: () => true,
        unmount: () => { unmountCount++; },
      }),
    });
    reg.mountPart("p1", document.createElement("div"), textPart("a"));
    reg.mountPart("p2", document.createElement("div"), textPart("b"));
    reg.unmountAll();
    expect(unmountCount).toBe(2);
  });

  test("mount failure falls through to next candidate", () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const reg = new RendererRegistryImpl();
    reg.registerScoped("broken", 10, {
      canRender: () => 20,
      mount: () => { throw new Error("mount failed"); },
    });
    reg.registerScoped("fallback", 0, simpleRenderer(10, {
      mount: (c) => { c.textContent = "fallback-won"; return { update: () => true, unmount() {} }; },
    }));
    const container = document.createElement("div");
    reg.mountPart("p1", container, textPart("test"));
    expect(container.textContent).toBe("fallback-won");
    spy.mockRestore();
  });
});

describe("RendererRegistryImpl self-release", () => {
  test("update returning false triggers re-mount", () => {
    const reg = new RendererRegistryImpl();
    let mountCount = 0;

    reg.registerScoped("flaky", 0, {
      canRender: () => 10,
      mount: (c) => {
        mountCount++;
        c.textContent = `mount-${mountCount}`;
        return {
          update: () => false, // Self-release.
          unmount: () => {},
        };
      },
    });

    const container = document.createElement("div");
    reg.mountPart("p1", container, textPart("test"));
    expect(mountCount).toBe(1);

    reg.updatePart("p1", textPart("u1"), false);
    // Self-release → re-score → re-mount.
    expect(mountCount).toBe(2);
  });
});
