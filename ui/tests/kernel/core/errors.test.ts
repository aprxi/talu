import { describe, test, expect, spyOn } from "bun:test";
import {
  errorBoundary,
  asyncErrorBoundary,
  HealthTracker,
} from "../../../src/kernel/core/errors.ts";

describe("errorBoundary", () => {
  test("returns fn result on success", () => {
    const result = errorBoundary("test", () => 42);
    expect(result).toBe(42);
  });

  test("returns undefined on throw", () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const result = errorBoundary("test", () => {
      throw new Error("boom");
    });
    expect(result).toBeUndefined();
    spy.mockRestore();
  });

  test("logs error to console", () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    errorBoundary("my.plugin", () => {
      throw new Error("fail");
    });
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0][0]).toContain("my.plugin");
    spy.mockRestore();
  });
});

describe("asyncErrorBoundary", () => {
  test("returns fn result on success", async () => {
    const result = await asyncErrorBoundary("test", async () => 42);
    expect(result).toBe(42);
  });

  test("returns undefined on throw", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const result = await asyncErrorBoundary("test", async () => {
      throw new Error("async boom");
    });
    expect(result).toBeUndefined();
    spy.mockRestore();
  });

  test("logs error to console", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    await asyncErrorBoundary("async.plugin", async () => {
      throw new Error("fail");
    });
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0][0]).toContain("async.plugin");
    spy.mockRestore();
  });
});

describe("HealthTracker", () => {
  test("starts healthy", () => {
    const tracker = new HealthTracker();
    expect(tracker.state).toBe("healthy");
    expect(tracker.isDisabled).toBe(false);
  });

  test("1 failure → still healthy", () => {
    const tracker = new HealthTracker();
    tracker.recordFailure();
    expect(tracker.state).toBe("healthy");
    expect(tracker.isDisabled).toBe(false);
  });

  test("2 failures → warning", () => {
    const tracker = new HealthTracker();
    tracker.recordFailure();
    tracker.recordFailure();
    expect(tracker.state).toBe("warning");
    expect(tracker.isDisabled).toBe(false);
  });

  test("3 failures → disabled", () => {
    const tracker = new HealthTracker();
    tracker.recordFailure();
    tracker.recordFailure();
    tracker.recordFailure();
    expect(tracker.state).toBe("disabled");
    expect(tracker.isDisabled).toBe(true);
  });

  test("recordSuccess resets strikes back to healthy", () => {
    const tracker = new HealthTracker();
    tracker.recordFailure();
    tracker.recordFailure(); // warning
    tracker.recordSuccess();
    expect(tracker.state).toBe("healthy");
  });

  test("recordSuccess does not recover from disabled", () => {
    const tracker = new HealthTracker();
    tracker.recordFailure();
    tracker.recordFailure();
    tracker.recordFailure(); // disabled
    tracker.recordSuccess();
    expect(tracker.state).toBe("disabled");
    expect(tracker.isDisabled).toBe(true);
  });

  test("recordFailure is no-op once disabled", () => {
    const tracker = new HealthTracker();
    tracker.recordFailure();
    tracker.recordFailure();
    tracker.recordFailure(); // disabled
    tracker.recordFailure(); // no-op
    tracker.recordFailure(); // no-op
    expect(tracker.state).toBe("disabled");
  });
});
