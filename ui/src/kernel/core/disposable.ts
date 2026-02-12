import type { Disposable } from "../types.ts";

/** Collects Disposable instances and disposes all on dispose(). */
export class DisposableStore implements Disposable {
  private items = new Set<Disposable>();
  private disposed = false;

  get isDisposed(): boolean {
    return this.disposed;
  }

  track<T extends Disposable>(disposable: T): T {
    if (this.disposed) {
      // Already disposed — immediately dispose the new item.
      try {
        disposable.dispose();
      } catch {
        // Swallow — callers should not need to handle this.
      }
      return disposable;
    }
    this.items.add(disposable);
    return disposable;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const item of this.items) {
      try {
        item.dispose();
      } catch (err) {
        console.error("[kernel] Disposable.dispose() threw:", err);
      }
    }
    this.items.clear();
  }
}
