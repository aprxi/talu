import { describe, test, expect } from "bun:test";
import { FormatAccessImpl } from "../../../src/kernel/system/format.ts";

const fmt = new FormatAccessImpl();

describe("FormatAccessImpl.duration", () => {
  test("0 seconds → '0s'", () => {
    expect(fmt.duration(0)).toBe("0s");
  });

  test("seconds only", () => {
    expect(fmt.duration(45)).toBe("45s");
  });

  test("minutes and seconds", () => {
    expect(fmt.duration(125)).toBe("2m 5s");
  });

  test("hours, minutes, seconds", () => {
    expect(fmt.duration(3661)).toBe("1h 1m 1s");
  });

  test("exact hours (no trailing 0m 0s)", () => {
    expect(fmt.duration(3600)).toBe("1h");
  });

  test("exact minutes (no trailing 0s)", () => {
    expect(fmt.duration(120)).toBe("2m");
  });

  test("negative seconds use absolute value", () => {
    expect(fmt.duration(-45)).toBe("45s");
  });
});

describe("FormatAccessImpl.number", () => {
  test("formats integers with locale grouping", () => {
    const result = fmt.number(1234);
    // Must produce the expected Intl output (locale-dependent separator).
    const expected = new Intl.NumberFormat().format(1234);
    expect(result).toBe(expected);
  });

  test("respects percent style option", () => {
    const result = fmt.number(0.5, { style: "percent" });
    const expected = new Intl.NumberFormat(undefined, { style: "percent" }).format(0.5);
    expect(result).toBe(expected);
  });

  test("formats zero", () => {
    expect(fmt.number(0)).toBe(new Intl.NumberFormat().format(0));
  });

  test("formats negative numbers", () => {
    const result = fmt.number(-42);
    expect(result).toBe(new Intl.NumberFormat().format(-42));
  });
});

describe("FormatAccessImpl.date", () => {
  test("accepts Date object — medium style matches Intl output", () => {
    const d = new Date(2024, 0, 15); // Jan 15, 2024
    const result = fmt.date(d, "medium");
    const expected = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" }).format(d);
    expect(result).toBe(expected);
  });

  test("accepts timestamp number — epoch is a valid date", () => {
    const result = fmt.date(0, "short");
    // Epoch date: Jan 1 1970 — must contain the day.
    const expected = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(0));
    expect(result).toBe(expected);
  });

  test("accepts ISO string — long style matches Intl output", () => {
    const result = fmt.date("2024-06-15T12:00:00Z", "long");
    const expected = new Intl.DateTimeFormat(undefined, { weekday: "long", month: "long", day: "numeric", year: "numeric" }).format(new Date("2024-06-15T12:00:00Z"));
    expect(result).toBe(expected);
  });
});

describe("FormatAccessImpl.dateTime", () => {
  test("includes time components for short style", () => {
    const d = new Date(2024, 0, 15, 14, 30);
    const result = fmt.dateTime(d, "short");
    // dateTime short = short date + hour:minute.
    const expected = new Intl.DateTimeFormat(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    }).format(d);
    expect(result).toBe(expected);
  });

  test("medium style matches Intl output", () => {
    const d = new Date(2024, 5, 20, 9, 15);
    const result = fmt.dateTime(d, "medium");
    const expected = new Intl.DateTimeFormat(undefined, {
      month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit",
    }).format(d);
    expect(result).toBe(expected);
  });
});

describe("FormatAccessImpl.relativeTime", () => {
  test("now → 'now' or '0 seconds ago'", () => {
    const result = fmt.relativeTime(new Date());
    // Intl.RelativeTimeFormat with numeric:"auto" formats 0 seconds as "now".
    const expected = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" }).format(0, "second");
    expect(result).toBe(expected);
  });

  test("past date produces 'ago' or past-relative text", () => {
    const past = new Date(Date.now() - 3600 * 1000); // 1 hour ago
    const result = fmt.relativeTime(past);
    // Should produce something like "1 hour ago".
    const expected = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" }).format(-1, "hour");
    expect(result).toBe(expected);
  });

  test("future date produces future-relative text", () => {
    const future = new Date(Date.now() + 2 * 24 * 3600 * 1000); // ~2 days from now
    const result = fmt.relativeTime(future);
    // Should produce something like "in 2 days".
    const expected = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" }).format(2, "day");
    expect(result).toBe(expected);
  });
});
