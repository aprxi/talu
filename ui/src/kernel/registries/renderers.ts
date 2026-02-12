/**
 * Renderer Pipeline — scoring-based renderer selection, stateful lifecycle,
 * self-release with thrash guard, pre-processors, error boundaries.
 *
 * Renderers are registered by plugins via ctx.renderers.register().
 * The kernel calls mountPart/updatePart/unmountPart during chat streaming.
 *
 * Scoring: canRender() → highest score wins. Ties broken by manifest
 * priority, then pluginId lexicographic. Deterministic and auditable.
 *
 * Lifecycle: mount() once → update() on each delta → unmount() on cleanup.
 * Self-release: update() returns false → re-score → mount new renderer.
 * Thrash guard: 3 self-releases locks to default; lifted on isFinal.
 */

import type { Disposable, ContentPart, MessageRenderer, RendererInstance } from "../types.ts";
import { errorBoundary } from "../core/errors.ts";
import { checkRendererNodeCap, truncateToolResult } from "../security/resource-caps.ts";

interface RegisteredRenderer {
  pluginId: string;
  manifestPriority: number;
  renderer: MessageRenderer;
}

interface ActiveMount {
  partId: string;
  container: HTMLElement;
  entry: RegisteredRenderer;
  instance: RendererInstance;
  abortController: AbortController;
  selfReleaseCount: number;
  locked: boolean;
}

const THRASH_LIMIT = 3;

export class RendererRegistryImpl {
  private renderers: RegisteredRenderer[] = [];
  private preProcessors: Array<{ pluginId: string; fn: (text: string) => string }> = [];
  private activeMounts = new Map<string, ActiveMount>();

  // --- Registration ---

  registerScoped(pluginId: string, manifestPriority: number, renderer: MessageRenderer): Disposable {
    const entry: RegisteredRenderer = { pluginId, manifestPriority, renderer };
    this.renderers.push(entry);
    return {
      dispose: () => {
        const idx = this.renderers.indexOf(entry);
        if (idx >= 0) this.renderers.splice(idx, 1);
      },
    };
  }

  registerPreProcessorScoped(pluginId: string, fn: (text: string) => string): Disposable {
    const entry = { pluginId, fn };
    this.preProcessors.push(entry);
    return {
      dispose: () => {
        const idx = this.preProcessors.indexOf(entry);
        if (idx >= 0) this.preProcessors.splice(idx, 1);
      },
    };
  }

  // --- Pre-processing ---

  /** Run all pre-processors synchronously. Skips on error. */
  applyPreProcessors(text: string): string {
    let result = text;
    for (const pp of this.preProcessors) {
      const out = errorBoundary(pp.pluginId, () => pp.fn(result));
      if (typeof out === "string") result = out;
    }
    return result;
  }

  // --- Scoring ---

  /**
   * Score all renderers for a content part.
   * Returns sorted candidates (highest score first).
   */
  private scoreRenderers(part: ContentPart): RegisteredRenderer[] {
    const scored: Array<{ entry: RegisteredRenderer; score: number }> = [];

    for (const entry of this.renderers) {
      // Skip renderers that declared kinds but don't include this type.
      if (entry.renderer.kinds && !entry.renderer.kinds.includes(part.type)) continue;

      const score = errorBoundary(entry.pluginId, () => entry.renderer.canRender(part));
      if (score === undefined || score === false || (typeof score === "number" && score <= 0)) continue;

      scored.push({ entry, score: score as number });
    }

    // Sort: highest score first, then manifest priority, then pluginId.
    scored.sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      if (b.entry.manifestPriority !== a.entry.manifestPriority) {
        return b.entry.manifestPriority - a.entry.manifestPriority;
      }
      return a.entry.pluginId.localeCompare(b.entry.pluginId);
    });

    return scored.map((s) => s.entry);
  }

  // --- Part sanitization ---

  /** Truncate tool_result data before passing to renderers to protect the DOM. */
  private sanitizePart(part: ContentPart): ContentPart {
    if (part.type === "tool_result" && typeof part.data === "string") {
      return { ...part, data: truncateToolResult(part.data) };
    }
    return part;
  }

  // --- Mount lifecycle (kernel-internal, called by chat rendering layer) ---

  /** Render a content part into a container. */
  mountPart(partId: string, container: HTMLElement, part: ContentPart): void {
    this.unmountPart(partId);
    const safePart = this.sanitizePart(part);

    const candidates = this.scoreRenderers(safePart);
    if (candidates.length === 0) {
      this.mountDefault(partId, container, safePart);
      return;
    }

    this.mountCandidate(partId, container, safePart, candidates, 0);
  }

  private mountCandidate(
    partId: string,
    container: HTMLElement,
    part: ContentPart,
    candidates: RegisteredRenderer[],
    candidateIndex: number,
  ): void {
    const entry = candidates[candidateIndex];
    if (!entry) {
      this.mountDefault(partId, container, part);
      return;
    }

    const abortController = new AbortController();
    const instance = errorBoundary(entry.pluginId, () =>
      entry.renderer.mount(container, part, abortController.signal),
    );

    if (!instance) {
      // mount() failed — try next candidate.
      this.mountCandidate(partId, container, part, candidates, candidateIndex + 1);
      return;
    }

    this.activeMounts.set(partId, {
      partId,
      container,
      entry,
      instance,
      abortController,
      selfReleaseCount: 0,
      locked: false,
    });
  }

  private mountDefault(partId: string, container: HTMLElement, part: ContentPart): void {
    container.textContent =
      "text" in part && typeof part.text === "string"
        ? part.text
        : JSON.stringify(part);
  }

  /** Update a mounted part with new data (streaming delta). */
  updatePart(partId: string, part: ContentPart, isFinal: boolean): void {
    const safePart = this.sanitizePart(part);
    const mount = this.activeMounts.get(partId);
    if (!mount) {
      // No active mount — part is using default renderer. Update directly.
      return;
    }

    if (mount.locked) {
      // Thrash-locked to default. On isFinal, allow one final re-score.
      if (isFinal) {
        this.handleFinalRescore(mount, safePart);
      }
      return;
    }

    const result = errorBoundary(mount.entry.pluginId, () =>
      mount.instance.update(safePart, isFinal),
    );

    if (result === undefined || result === false) {
      // Error or explicit self-release.
      this.handleSelfRelease(mount, safePart, isFinal);
      return;
    }

    // Enforce DOM node cap — collapse to default if renderer exceeds limit.
    const rootNode = mount.container.getRootNode();
    if (rootNode instanceof ShadowRoot) {
      const exceeded = checkRendererNodeCap(rootNode);
      if (exceeded > 0) {
        console.warn(
          `[kernel] Renderer "${mount.entry.pluginId}" exceeded DOM node cap (${exceeded} nodes) for part "${partId}" — collapsing to default.`,
        );
        errorBoundary(mount.entry.pluginId, () => mount.instance.unmount());
        mount.abortController.abort();
        this.activeMounts.delete(partId);
        this.mountDefault(partId, mount.container, safePart);
      }
    }
  }

  private handleSelfRelease(mount: ActiveMount, part: ContentPart, isFinal: boolean): void {
    mount.selfReleaseCount++;

    // Unmount current, abort signal.
    errorBoundary(mount.entry.pluginId, () => mount.instance.unmount());
    mount.abortController.abort();

    if (mount.selfReleaseCount >= THRASH_LIMIT && !isFinal) {
      // Thrash guard: lock to default renderer.
      mount.locked = true;
      this.activeMounts.delete(mount.partId);
      this.mountDefault(mount.partId, mount.container, part);
      console.warn(
        `[kernel] Renderer thrash guard: part "${mount.partId}" locked to default after ${THRASH_LIMIT} self-releases.`,
      );
      return;
    }

    // Re-score with full accumulated content.
    this.activeMounts.delete(mount.partId);
    const candidates = this.scoreRenderers(part);

    if (candidates.length > 0) {
      this.mountCandidate(mount.partId, mount.container, part, candidates, 0);
    } else {
      this.mountDefault(mount.partId, mount.container, part);
    }
  }

  private handleFinalRescore(mount: ActiveMount, part: ContentPart): void {
    // Lift thrash lock on isFinal — allow one final re-score.
    mount.locked = false;
    mount.selfReleaseCount = 0;
    this.activeMounts.delete(mount.partId);

    const candidates = this.scoreRenderers(part);
    if (candidates.length > 0) {
      this.mountCandidate(mount.partId, mount.container, part, candidates, 0);
    } else {
      this.mountDefault(mount.partId, mount.container, part);
    }
  }

  /** Unmount a part (cleanup). */
  unmountPart(partId: string): void {
    const mount = this.activeMounts.get(partId);
    if (!mount) return;

    mount.abortController.abort();
    errorBoundary(mount.entry.pluginId, () => mount.instance.unmount());
    this.activeMounts.delete(partId);
  }

  /** Unmount all parts (kernel shutdown). */
  unmountAll(): void {
    for (const partId of [...this.activeMounts.keys()]) {
      this.unmountPart(partId);
    }
  }
}
