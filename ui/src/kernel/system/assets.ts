/**
 * Asset Resolution — resolve relative paths to plugin static resource URLs.
 *
 * Necessary because plugins render inside Shadow DOM containers —
 * relative paths in HTML attributes resolve against the document origin,
 * not the plugin's script location.
 */

import type { AssetResolver } from "../types.ts";

export function createAssetResolver(pluginId: string): AssetResolver {
  return {
    getUrl(path: string): string {
      // Strip leading ./ or / for consistency.
      const clean = path.replace(/^\.?\//, "");
      return `/v1/plugins/${encodeURIComponent(pluginId)}/${clean}`;
    },
  };
}
