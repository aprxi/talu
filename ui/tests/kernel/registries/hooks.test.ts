import { describe, test, expect, spyOn } from "bun:test";
import { HookPipelineImpl } from "../../../src/kernel/registries/hooks.ts";

describe("HookPipelineImpl", () => {
  test("empty pipeline returns initial value", async () => {
    const pipeline = new HookPipelineImpl();
    const result = await pipeline.run("test.hook", { value: 42 });
    expect(result).toEqual({ value: 42 });
  });

  test("single handler transforms value", async () => {
    const pipeline = new HookPipelineImpl();
    pipeline.onScoped("pluginA", "transform", (v) => {
      const obj = v as { count: number };
      return { count: obj.count + 1 };
    });
    const result = await pipeline.run("transform", { count: 10 });
    expect(result).toEqual({ count: 11 });
  });

  test("handlers execute in priority order (higher first)", async () => {
    const pipeline = new HookPipelineImpl();
    const order: string[] = [];

    pipeline.onScoped("low", "ordered", () => {
      order.push("low");
      return undefined;
    }, { priority: 1 });

    pipeline.onScoped("high", "ordered", () => {
      order.push("high");
      return undefined;
    }, { priority: 10 });

    pipeline.onScoped("mid", "ordered", () => {
      order.push("mid");
      return undefined;
    }, { priority: 5 });

    await pipeline.run("ordered", {});
    expect(order).toEqual(["high", "mid", "low"]);
  });

  test("handler error → skipped, pipeline continues", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const pipeline = new HookPipelineImpl();

    pipeline.onScoped("good1", "resilient", (v) => {
      return { ...(v as object), step1: true };
    });

    pipeline.onScoped("bad", "resilient", () => {
      throw new Error("handler exploded");
    });

    pipeline.onScoped("good2", "resilient", (v) => {
      return { ...(v as object), step3: true };
    });

    const result = await pipeline.run("resilient", {});
    expect(result).toEqual({ step1: true, step3: true });
    expect(spy).toHaveBeenCalledTimes(1);
    spy.mockRestore();
  });

  test("handler returning undefined → pass-through", async () => {
    const pipeline = new HookPipelineImpl();
    pipeline.onScoped("noop", "passthrough", () => undefined);
    const result = await pipeline.run("passthrough", { original: true });
    expect(result).toEqual({ original: true });
  });

  test("handler returning $block sentinel → pipeline stops", async () => {
    const pipeline = new HookPipelineImpl();

    pipeline.onScoped("blocker", "blockable", () => {
      return { $block: true, reason: "denied" };
    });

    pipeline.onScoped("after", "blockable", (v) => {
      return { ...(v as object), afterBlock: true };
    });

    const result = await pipeline.run("blockable", {});
    expect(result).toEqual({ $block: true, reason: "denied" });
  });

  test("structuredClone between handlers prevents mutation leaks", async () => {
    const pipeline = new HookPipelineImpl();
    let capturedInput: unknown;

    pipeline.onScoped("first", "clone", (v) => {
      const obj = v as { items: string[] };
      return { items: [...obj.items, "added"] };
    });

    pipeline.onScoped("second", "clone", (v) => {
      capturedInput = v;
      return v;
    });

    const original = { items: ["a"] };
    await pipeline.run("clone", original);

    // The second handler should receive a clone, not the original reference.
    expect(capturedInput).toEqual({ items: ["a", "added"] });
    // Original should not be mutated.
    expect(original.items).toEqual(["a"]);
  });

  test("multiple handlers compose correctly", async () => {
    const pipeline = new HookPipelineImpl();

    pipeline.onScoped("a", "compose", (v) => {
      return { value: (v as { value: number }).value * 2 };
    });

    pipeline.onScoped("b", "compose", (v) => {
      return { value: (v as { value: number }).value + 3 };
    });

    const result = await pipeline.run("compose", { value: 5 });
    // 5 * 2 = 10, 10 + 3 = 13
    expect(result).toEqual({ value: 13 });
  });

  test("dispose removes handler from pipeline", async () => {
    const pipeline = new HookPipelineImpl();

    const disposable = pipeline.onScoped("removable", "disposable", () => {
      return { transformed: true };
    });

    // Before dispose: handler runs.
    const before = await pipeline.run("disposable", { original: true });
    expect(before).toEqual({ transformed: true });

    disposable.dispose();

    // After dispose: pipeline is empty, returns initial value.
    const after = await pipeline.run("disposable", { original: true });
    expect(after).toEqual({ original: true });
  });

  test("disposing last handler cleans up the hook name", async () => {
    const pipeline = new HookPipelineImpl();

    const d1 = pipeline.onScoped("a", "cleanup", () => ({ a: true }));
    const d2 = pipeline.onScoped("b", "cleanup", () => ({ b: true }));

    d1.dispose();
    d2.dispose();

    // After all handlers disposed, the pipeline for this name is empty.
    const result = await pipeline.run("cleanup", { empty: true });
    expect(result).toEqual({ empty: true });
  });
});
