/**
 * Files plugin DOM construction — sidebar + table + preview panel.
 */

import {
  SEARCH_ICON as ICON_SEARCH,
  CLOSE_ICON as ICON_CLEAR,
  EXPORT_ICON as ICON_UPLOAD,
} from "../../icons.ts";

export function buildFilesDOM(root: HTMLElement): void {
  root.innerHTML = `
    <div style="display: flex; flex: 1; min-height: 0; overflow: hidden;">
      <!-- Main content -->
      <div class="browser-main files-main-drop">
        <!-- Top bar -->
        <div class="browser-header">
          <div class="search-wrapper">
            ${ICON_SEARCH}
            <input id="fp-search" type="text" placeholder="Search files..." class="search-input">
            <button id="fp-search-clear" class="browser-clear-btn hidden">${ICON_CLEAR}</button>
          </div>
          <div class="flex-1"></div>
          <button id="fp-select-all" class="btn btn-ghost btn-sm">Select All</button>
          <div id="fp-bulk-actions" class="browser-bulk-actions">
            <button id="fp-archive" class="btn btn-ghost btn-sm" disabled>Archive</button>
            <button id="fp-restore" class="btn btn-ghost btn-sm hidden" disabled>Restore</button>
            <button id="fp-delete" class="btn btn-danger btn-sm" disabled>Delete</button>
          </div>
          <button id="fp-cancel" class="btn btn-ghost btn-sm hidden">Cancel</button>
          <span id="fp-count" class="files-count"></span>
          <button id="fp-upload" class="btn btn-ghost btn-sm">
            ${ICON_UPLOAD} Upload
          </button>
          <input id="fp-file-input" type="file" multiple style="display:none">
          <span id="fp-stats" class="files-stats"></span>
        </div>

        <!-- Table -->
        <div id="fp-table-container" class="files-table-container scroll-thin">
          <table class="files-table">
            <thead id="fp-thead" class="files-thead">
              <tr>
                <th class="files-th files-th-check"></th>
                <th class="files-th files-th-name" data-sort="name">Name</th>
                <th class="files-th files-th-kind" data-sort="kind">Kind</th>
                <th class="files-th files-th-size" data-sort="size">Size</th>
                <th class="files-th files-th-date" data-sort="date">Date</th>
                <th class="files-th files-th-actions"></th>
              </tr>
            </thead>
            <tbody id="fp-tbody"></tbody>
          </table>
        </div>
        <div id="fp-pagination"></div>

        <!-- Drop overlay (hidden by default, shown via CSS) -->
        <div id="fp-drop-overlay" class="files-drop-overlay hidden">Drop files to upload</div>
      </div>

      <!-- Preview panel -->
      <div id="fp-preview" class="files-preview hidden">
        <div class="panel-section">
          <div class="panel-heading">Preview</div>
        </div>
        <div id="fp-preview-content" class="files-preview-body"></div>
      </div>
      </div>
  `;

}
