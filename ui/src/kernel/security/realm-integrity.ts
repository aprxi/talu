/**
 * Same-Realm Integrity — freeze critical intrinsics to prevent prototype pollution.
 *
 * Called after Kernel init but before third-party plugin loading.
 * Enabled by default — third-party plugins must not patch prototypes.
 * Pass { enabled: false } to disable (e.g. during tests).
 *
 * Dev-mode tamper detector: snapshots descriptors, compares post-load.
 */

import type { Disposable } from "../types.ts";

interface DescriptorSnapshot {
  target: object;
  name: string;
  descriptors: Record<string, PropertyDescriptor>;
}

/**
 * Freeze critical intrinsics. Returns a Disposable that provides
 * a tamper detection check (does not unfreeze — that's impossible).
 */
export function freezeIntrinsics(options?: { enabled?: boolean }): Disposable & { checkTamper(): string[] } {
  const snapshots: DescriptorSnapshot[] = [];
  const warnings: string[] = [];

  if (options?.enabled === false) {
    return {
      dispose() {},
      checkTamper() { return []; },
    };
  }

  // Minimal freeze set for V1: focus on the APIs that security interception depends on.
  const targets: [string, object][] = [
    ["Object.prototype", Object.prototype],
    ["Array.prototype", Array.prototype],
    ["Function.prototype", Function.prototype],
    ["Promise.prototype", Promise.prototype],
  ];

  for (const [name, target] of targets) {
    // Snapshot before freeze.
    const descriptors = Object.getOwnPropertyDescriptors(target);
    snapshots.push({ target, name, descriptors });

    try {
      Object.freeze(target);
    } catch (err) {
      console.warn(`[kernel] Failed to freeze ${name}:`, err);
    }
  }

  return {
    /** Check if any frozen intrinsics were tampered with after freeze. */
    checkTamper(): string[] {
      warnings.length = 0;
      for (const snap of snapshots) {
        const current = Object.getOwnPropertyDescriptors(snap.target);
        for (const key of Object.keys(current)) {
          if (!(key in snap.descriptors)) {
            warnings.push(`${snap.name}.${key} was added after freeze.`);
          }
        }
        for (const key of Object.keys(snap.descriptors)) {
          if (!(key in current)) {
            warnings.push(`${snap.name}.${key} was removed after freeze.`);
          }
        }
      }
      if (warnings.length > 0) {
        for (const w of warnings) {
          console.warn(`[kernel] Global tamper detected: ${w}`);
        }
      }
      return warnings;
    },

    dispose() {
      // Object.freeze is irreversible — nothing to undo.
      snapshots.length = 0;
    },
  };
}
