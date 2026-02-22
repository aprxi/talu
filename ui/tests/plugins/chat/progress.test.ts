import { describe, test, expect, beforeEach } from "bun:test";
import { updateProgressBar, removeProgressBar } from "../../../src/plugins/chat/messages.ts";
import { initChatDom } from "../../../src/plugins/chat/dom.ts";

/**
 * Tests for the chat progress bar displayed in the transcript container
 * during model loading / prefill.
 */

let transcriptEl: HTMLElement;

beforeEach(() => {
  // Build minimal DOM structure expected by getChatDom().
  const container = document.createElement("div");
  transcriptEl = document.createElement("div");
  transcriptEl.id = "transcript";
  transcriptEl.className = "transcript";
  container.appendChild(transcriptEl);
  document.body.appendChild(container);
  initChatDom(container);
});

// ── updateProgressBar ────────────────────────────────────────────────────────

describe("updateProgressBar", () => {
  test("creates .chat-progress element in transcript container", () => {
    updateProgressBar("Loading", 0, 100);
    const bar = transcriptEl.querySelector(".chat-progress");
    expect(bar).not.toBeNull();
    expect(bar!.querySelector(".chat-progress-label")!.textContent).toBe("Loading");
    expect(bar!.querySelector(".chat-progress-pct")!.textContent).toBe("0%");
  });

  test("appends progress bar as child of transcript (not inside messages)", () => {
    const messages = document.createElement("div");
    messages.className = "transcript-messages";
    transcriptEl.appendChild(messages);

    updateProgressBar("Loading", 10, 100);
    // Bar should be a direct child of transcript, not inside messages.
    expect(transcriptEl.querySelector(":scope > .chat-progress")).not.toBeNull();
  });

  test("updates label, fill width, and percentage", () => {
    updateProgressBar("Loading", 25, 100);
    expect(transcriptEl.querySelector(".chat-progress-label")!.textContent).toBe("Loading");
    expect((transcriptEl.querySelector(".chat-progress-fill") as HTMLElement).style.width).toBe("25%");
    expect(transcriptEl.querySelector(".chat-progress-pct")!.textContent).toBe("25%");

    updateProgressBar("Preparing", 50, 100);
    expect(transcriptEl.querySelector(".chat-progress-label")!.textContent).toBe("Preparing");
    expect((transcriptEl.querySelector(".chat-progress-fill") as HTMLElement).style.width).toBe("50%");
    expect(transcriptEl.querySelector(".chat-progress-pct")!.textContent).toBe("50%");
  });

  test("reuses existing element on subsequent calls", () => {
    updateProgressBar("Loading", 0, 50);
    updateProgressBar("Loading", 25, 50);
    const bars = transcriptEl.querySelectorAll(".chat-progress");
    expect(bars.length).toBe(1);
    expect(transcriptEl.querySelector(".chat-progress-pct")!.textContent).toBe("50%");
  });

  test("handles zero total gracefully", () => {
    updateProgressBar("Loading", 0, 0);
    expect(transcriptEl.querySelector(".chat-progress-pct")!.textContent).toBe("0%");
    expect((transcriptEl.querySelector(".chat-progress-fill") as HTMLElement).style.width).toBe("0%");
  });

  test("rounds percentage to nearest integer", () => {
    updateProgressBar("Prefill", 1, 3);
    expect(transcriptEl.querySelector(".chat-progress-pct")!.textContent).toBe("33%");
  });

  test("clamps at 100%", () => {
    updateProgressBar("Prefill", 100, 100);
    expect(transcriptEl.querySelector(".chat-progress-pct")!.textContent).toBe("100%");
    expect((transcriptEl.querySelector(".chat-progress-fill") as HTMLElement).style.width).toBe("100%");
  });
});

// ── removeProgressBar ────────────────────────────────────────────────────────

describe("removeProgressBar", () => {
  test("removes existing progress bar", () => {
    updateProgressBar("Loading", 50, 100);
    expect(transcriptEl.querySelector(".chat-progress")).not.toBeNull();
    removeProgressBar();
    expect(transcriptEl.querySelector(".chat-progress")).toBeNull();
  });

  test("is a no-op when no progress bar exists", () => {
    const childCount = transcriptEl.children.length;
    removeProgressBar();
    expect(transcriptEl.children.length).toBe(childCount);
  });
});
