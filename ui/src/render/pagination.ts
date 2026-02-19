/**
 * Shared pagination component — page-based navigation controls
 * used by both the files table (full mode) and conversations sidebar (compact mode).
 */

import { el } from "./helpers.ts";

export interface PaginationState {
  currentPage: number; // 1-indexed
  totalPages: number;
  totalItems: number;
  pageSize: number;
}

export function computePagination(
  totalItems: number,
  pageSize: number,
  currentPage: number,
): PaginationState {
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
  return {
    currentPage: Math.max(1, Math.min(currentPage, totalPages)),
    totalPages,
    totalItems,
    pageSize,
  };
}

/**
 * Render pagination controls.
 *
 * Full mode: [Prev] [1] [2] ... [10] [Next]  Showing 1–50 of 234
 * Compact mode: [<] 1 / 5 [>]
 */
export function renderPagination(
  state: PaginationState,
  onPageChange: (page: number) => void,
  compact?: boolean,
): HTMLElement {
  const container = el("div", compact ? "pagination compact" : "pagination");

  const btn = (label: string, page: number, disabled: boolean, active?: boolean): HTMLButtonElement => {
    const b = el("button", "pagination-btn", label);
    if (active) b.classList.add("active");
    b.disabled = disabled;
    if (!disabled) {
      b.addEventListener("click", (e) => {
        e.stopPropagation();
        onPageChange(page);
      });
    }
    return b;
  };

  if (compact) {
    container.appendChild(btn("\u2039", state.currentPage - 1, state.currentPage <= 1));
    container.appendChild(el("span", "pagination-info", `${state.currentPage} / ${state.totalPages}`));
    container.appendChild(btn("\u203a", state.currentPage + 1, state.currentPage >= state.totalPages));
    return container;
  }

  // Full mode
  container.appendChild(btn("Prev", state.currentPage - 1, state.currentPage <= 1));

  const pages = pageNumbers(state.currentPage, state.totalPages);
  for (const p of pages) {
    if (p === 0) {
      container.appendChild(el("span", "pagination-ellipsis", "\u2026"));
    } else {
      container.appendChild(btn(String(p), p, false, p === state.currentPage));
    }
  }

  container.appendChild(btn("Next", state.currentPage + 1, state.currentPage >= state.totalPages));

  // Info text
  const start = (state.currentPage - 1) * state.pageSize + 1;
  const end = Math.min(state.currentPage * state.pageSize, state.totalItems);
  container.appendChild(
    el("span", "pagination-info", `Showing ${start}\u2013${end} of ${state.totalItems}`),
  );

  return container;
}

/**
 * Compute which page numbers to show. Returns an array where 0 = ellipsis.
 * Always shows first, last, and current ±2 neighbors. Max 7 entries.
 */
function pageNumbers(current: number, total: number): number[] {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i + 1);
  }

  const pages = new Set<number>();
  pages.add(1);
  pages.add(total);
  for (let i = Math.max(1, current - 2); i <= Math.min(total, current + 2); i++) {
    pages.add(i);
  }

  const sorted = [...pages].sort((a, b) => a - b);
  const result: number[] = [];
  for (let i = 0; i < sorted.length; i++) {
    if (i > 0 && sorted[i]! - sorted[i - 1]! > 1) {
      result.push(0); // ellipsis
    }
    result.push(sorted[i]!);
  }
  return result;
}
