import { describe, test, expect } from "bun:test";
import { computePagination, renderPagination } from "../../src/render/pagination.ts";
import type { PaginationState } from "../../src/render/pagination.ts";

/**
 * Tests for computePagination — page math for offset-based pagination.
 *
 * The pageNumbers helper (internal) is tested indirectly via renderPagination,
 * but computePagination's clamping and rounding logic is tested directly here.
 */

describe("computePagination", () => {
  test("single page when totalItems <= pageSize", () => {
    const state = computePagination(10, 50, 1);
    expect(state.totalPages).toBe(1);
    expect(state.currentPage).toBe(1);
    expect(state.totalItems).toBe(10);
    expect(state.pageSize).toBe(50);
  });

  test("zero items yields 1 page", () => {
    const state = computePagination(0, 50, 1);
    expect(state.totalPages).toBe(1);
    expect(state.currentPage).toBe(1);
  });

  test("exact multiple of pageSize", () => {
    const state = computePagination(100, 50, 1);
    expect(state.totalPages).toBe(2);
  });

  test("non-exact rounds up", () => {
    const state = computePagination(101, 50, 1);
    expect(state.totalPages).toBe(3);
  });

  test("currentPage clamped to totalPages", () => {
    const state = computePagination(100, 50, 999);
    expect(state.currentPage).toBe(2);
  });

  test("currentPage clamped to 1 when below", () => {
    const state = computePagination(100, 50, 0);
    expect(state.currentPage).toBe(1);
  });

  test("negative currentPage clamped to 1", () => {
    const state = computePagination(100, 50, -5);
    expect(state.currentPage).toBe(1);
  });

  test("middle page preserved", () => {
    const state = computePagination(500, 50, 5);
    expect(state.totalPages).toBe(10);
    expect(state.currentPage).toBe(5);
  });

  test("pageSize of 1 with many items", () => {
    const state = computePagination(100, 1, 50);
    expect(state.totalPages).toBe(100);
    expect(state.currentPage).toBe(50);
  });

  test("last page when totalItems not divisible", () => {
    const state = computePagination(51, 50, 2);
    expect(state.totalPages).toBe(2);
    expect(state.currentPage).toBe(2);
  });
});

// ── renderPagination — full mode ────────────────────────────────────────────

function makeState(overrides: Partial<PaginationState> = {}): PaginationState {
  return { currentPage: 1, totalPages: 5, totalItems: 250, pageSize: 50, ...overrides };
}

describe("renderPagination — full mode", () => {
  test("renders Prev and Next buttons", () => {
    const el = renderPagination(makeState(), () => {});
    const buttons = el.querySelectorAll<HTMLButtonElement>(".pagination-btn");
    expect(buttons[0]!.textContent).toBe("Prev");
    expect(buttons[buttons.length - 1]!.textContent).toBe("Next");
  });

  test("Prev disabled on first page", () => {
    const el = renderPagination(makeState({ currentPage: 1 }), () => {});
    const prev = el.querySelector<HTMLButtonElement>(".pagination-btn")!;
    expect(prev.disabled).toBe(true);
  });

  test("Prev enabled on later pages", () => {
    const el = renderPagination(makeState({ currentPage: 3 }), () => {});
    const prev = el.querySelector<HTMLButtonElement>(".pagination-btn")!;
    expect(prev.disabled).toBe(false);
  });

  test("Next disabled on last page", () => {
    const el = renderPagination(makeState({ currentPage: 5 }), () => {});
    const buttons = el.querySelectorAll<HTMLButtonElement>(".pagination-btn");
    const next = buttons[buttons.length - 1]!;
    expect(next.disabled).toBe(true);
  });

  test("Next enabled on non-last page", () => {
    const el = renderPagination(makeState({ currentPage: 3 }), () => {});
    const buttons = el.querySelectorAll<HTMLButtonElement>(".pagination-btn");
    const next = buttons[buttons.length - 1]!;
    expect(next.disabled).toBe(false);
  });

  test("active page button has active class", () => {
    const el = renderPagination(makeState({ currentPage: 3 }), () => {});
    const active = el.querySelector<HTMLButtonElement>(".pagination-btn.active")!;
    expect(active.textContent).toBe("3");
  });

  test("clicking page button calls onPageChange", () => {
    let clicked: number | null = null;
    const el = renderPagination(makeState({ currentPage: 1 }), (p) => { clicked = p; });
    // Find button labeled "3"
    const buttons = el.querySelectorAll<HTMLButtonElement>(".pagination-btn");
    const btn3 = Array.from(buttons).find((b) => b.textContent === "3")!;
    btn3.click();
    expect(clicked).toBe(3);
  });

  test("info text shows correct range", () => {
    const el = renderPagination(makeState({ currentPage: 2, totalItems: 234, pageSize: 50 }), () => {});
    const info = el.querySelector(".pagination-info")!;
    expect(info.textContent).toContain("51");
    expect(info.textContent).toContain("100");
    expect(info.textContent).toContain("234");
  });

  test("last page info text clamps end to totalItems", () => {
    const el = renderPagination(makeState({ currentPage: 5, totalPages: 5, totalItems: 234, pageSize: 50 }), () => {});
    const info = el.querySelector(".pagination-info")!;
    expect(info.textContent).toContain("234");
  });
});

// ── renderPagination — compact mode ─────────────────────────────────────────

describe("renderPagination — compact mode", () => {
  test("renders compact layout with page info", () => {
    const el = renderPagination(makeState({ currentPage: 3, totalPages: 5 }), () => {}, true);
    expect(el.classList.contains("compact")).toBe(true);
    const info = el.querySelector(".pagination-info")!;
    expect(info.textContent).toBe("3 / 5");
  });

  test("prev disabled on first page in compact", () => {
    const el = renderPagination(makeState({ currentPage: 1, totalPages: 5 }), () => {}, true);
    const buttons = el.querySelectorAll<HTMLButtonElement>(".pagination-btn");
    expect(buttons[0]!.disabled).toBe(true);
  });

  test("next disabled on last page in compact", () => {
    const el = renderPagination(makeState({ currentPage: 5, totalPages: 5 }), () => {}, true);
    const buttons = el.querySelectorAll<HTMLButtonElement>(".pagination-btn");
    expect(buttons[1]!.disabled).toBe(true);
  });

  test("compact prev click calls onPageChange", () => {
    let clicked: number | null = null;
    const el = renderPagination(makeState({ currentPage: 3, totalPages: 5 }), (p) => { clicked = p; }, true);
    const buttons = el.querySelectorAll<HTMLButtonElement>(".pagination-btn");
    buttons[0]!.click();
    expect(clicked).toBe(2);
  });
});

// ── pageNumbers (tested indirectly via renderPagination) ────────────────────

describe("pageNumbers — via renderPagination", () => {
  test("shows all pages when total <= 7", () => {
    const el = renderPagination(makeState({ currentPage: 1, totalPages: 5, totalItems: 250, pageSize: 50 }), () => {});
    const pageButtons = Array.from(el.querySelectorAll<HTMLButtonElement>(".pagination-btn"))
      .filter((b) => b.textContent !== "Prev" && b.textContent !== "Next");
    expect(pageButtons.map((b) => b.textContent)).toEqual(["1", "2", "3", "4", "5"]);
  });

  test("inserts ellipsis for many pages when on first page", () => {
    const state = computePagination(500, 50, 1);
    const el = renderPagination(state, () => {});
    const ellipses = el.querySelectorAll(".pagination-ellipsis");
    expect(ellipses.length).toBeGreaterThan(0);
    expect(ellipses[0]!.textContent).toBe("\u2026");
  });

  test("shows first and last page always", () => {
    const state = computePagination(500, 50, 5);
    const el = renderPagination(state, () => {});
    const pageButtons = Array.from(el.querySelectorAll<HTMLButtonElement>(".pagination-btn"))
      .filter((b) => b.textContent !== "Prev" && b.textContent !== "Next");
    const labels = pageButtons.map((b) => b.textContent);
    expect(labels[0]).toBe("1");
    expect(labels[labels.length - 1]).toBe("10");
  });

  test("current page neighbors are visible", () => {
    const state = computePagination(500, 50, 5);
    const el = renderPagination(state, () => {});
    const pageButtons = Array.from(el.querySelectorAll<HTMLButtonElement>(".pagination-btn"))
      .filter((b) => b.textContent !== "Prev" && b.textContent !== "Next");
    const labels = pageButtons.map((b) => b.textContent);
    expect(labels).toContain("3");
    expect(labels).toContain("4");
    expect(labels).toContain("5");
    expect(labels).toContain("6");
    expect(labels).toContain("7");
  });
});
