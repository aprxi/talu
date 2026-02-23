/** Constructs the repo plugin DOM into the given shadow container. */

import { SEARCH_ICON as ICON_SEARCH, CLOSE_ICON as ICON_CLEAR } from "../../icons.ts";

const ICON_DB_SM = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/></svg>`;
const ICON_HUB = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>`;

const TABLE_HEADER = `<tr>
  <th class="files-th files-th-check"></th>
  <th class="files-th" style="min-width:200px" data-sort="name">Model</th>
  <th class="files-th" style="width:100px">Arch</th>
  <th class="files-th" style="width:80px">Quant</th>
  <th class="files-th" style="width:80px" data-sort="size">Size</th>
  <th class="files-th" style="width:100px" data-sort="date">Modified</th>
  <th class="files-th" style="width:50px"></th>
  <th class="files-th" style="width:50px"></th>
</tr>`;

export function buildRepoDOM(container: HTMLElement): void {
  container.innerHTML = `
<div style="display: flex; height: 100%; overflow: hidden;">
  <!-- Left sidebar -->
  <div class="browser-sidebar">
    <div class="panel-section">
      <div class="panel-heading">View</div>
      <div>
        <button id="rp-tab-discover" class="browser-tab active" data-tab="discover">${ICON_HUB} Discover</button>
        <button id="rp-tab-local" class="browser-tab" data-tab="local">${ICON_DB_SM} Local</button>
      </div>
    </div>
    <div class="panel-section">
      <div class="panel-heading">Storage</div>
      <div id="rp-stats" class="files-stats">Loading...</div>
    </div>
    <div class="flex-1"></div>
  </div>

  <!-- Main content -->
  <div class="browser-main">
    <!-- Header toolbar -->
    <div class="browser-header">
      <div class="search-wrapper">
        ${ICON_SEARCH}
        <input id="rp-search" type="text" placeholder="Search models..." class="search-input">
        <button id="rp-search-clear" class="browser-clear-btn hidden">${ICON_CLEAR}</button>
      </div>
      <div class="flex-1"></div>
      <!-- Discover-only: filter selects -->
      <div id="rp-discover-toolbar" class="repo-toolbar-group">
        <select id="rp-sort" class="form-select-sm">
          <option value="trending" selected>Trending</option>
          <option value="downloads">Downloads</option>
          <option value="likes">Likes</option>
          <option value="last_modified">Recent</option>
        </select>
        <select id="rp-size-filter" class="form-select-sm">
          <option value="1">\u22641B</option>
          <option value="2">\u22642B</option>
          <option value="4">\u22644B</option>
          <option value="8" selected>\u22648B</option>
          <option value="16">\u226416B</option>
          <option value="32">\u226432B</option>
          <option value="64">\u226464B</option>
          <option value="128">\u2264128B</option>
          <option value="512">\u2264512B</option>
          <option value="any">Any size</option>
        </select>
        <select id="rp-task-filter" class="form-select-sm">
          <option value="text-generation" selected>Text Generation</option>
          <option value="image-text-to-text">Multimodal</option>
          <option value="image-to-text">Image to Text</option>
          <option value="text-to-image">Text to Image</option>
          <option value="text-to-speech">Text to Speech</option>
          <option value="sentence-similarity">Sentence Similarity</option>
          <option value="">Any task</option>
        </select>
        <select id="rp-library-filter" class="form-select-sm">
          <option value="safetensors" selected>safetensors</option>
          <option value="transformers">transformers</option>
          <option value="mlx">mlx</option>
          <option value="sentence-transformers">sentence-transformers</option>
          <option value="">Any library</option>
        </select>
      </div>
      <!-- Local-only: source toggle + bulk actions -->
      <div id="rp-local-toolbar" class="repo-toolbar-group hidden">
        <div class="repo-source-toggle">
          <button id="rp-source-all" class="active" data-source="all">All</button>
          <button id="rp-source-hub" data-source="hub">HuggingFace</button>
          <button id="rp-source-managed" data-source="managed">Managed</button>
        </div>
        <button id="rp-select-all" class="btn btn-ghost btn-sm">Select All</button>
        <div id="rp-bulk-actions" class="browser-bulk-actions">
          <button id="rp-pin-all" class="btn btn-ghost btn-sm" disabled>Pin</button>
          <button id="rp-delete" class="btn btn-danger btn-sm" disabled>Delete</button>
        </div>
        <button id="rp-cancel" class="btn btn-ghost btn-sm hidden">Cancel</button>
        <span id="rp-count" class="files-count"></span>
      </div>
    </div>

    <!-- Active downloads strip -->
    <div id="rp-downloads" class="repo-downloads hidden"></div>

    <!-- ═══ Discover view: hub search results ═══ -->
    <div id="rp-discover-view" style="flex:1; overflow-y:auto;">
      <div id="rp-discover-container">
        <div id="rp-discover-results" class="repo-discover-list"></div>
      </div>
    </div>

    <!-- ═══ Local view: all cached models table ═══ -->
    <div id="rp-local-view" class="hidden" style="flex:1; overflow-y:auto;">
      <div id="rp-local-table-container" class="files-table-container scroll-thin">
        <table class="files-table">
          <thead id="rp-local-thead" class="files-thead">${TABLE_HEADER}</thead>
          <tbody id="rp-local-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>`;

}
