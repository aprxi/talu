/**
 * Files plugin DOM construction — sidebar + table + preview panel.
 */

import {
  SEARCH_ICON as ICON_SEARCH,
  CLOSE_ICON as ICON_CLEAR,
  EXPORT_ICON as ICON_UPLOAD,
  ARCHIVE_BOX_ICON as ICON_ARCHIVE,
} from "../../icons.ts";

const ICON_FILE = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>`;

export function buildFilesDOM(root: HTMLElement): void {
  root.innerHTML = `
    <div style="display: flex; height: 100%; overflow: hidden;">
      <!-- Left sidebar -->
      <div class="browser-sidebar">
        <div class="panel-section">
          <div class="search-wrapper">
            ${ICON_SEARCH}
            <input id="fp-search" type="text" placeholder="Search files..." class="search-input">
          </div>
        </div>

        <!-- Status tabs -->
        <div class="panel-section">
          <div class="panel-heading">Status</div>
          <div>
            <button id="fp-tab-all" class="browser-tab active" data-tab="all">
              ${ICON_FILE}
              Active
            </button>
            <button id="fp-tab-archived" class="browser-tab" data-tab="archived">
              ${ICON_ARCHIVE}
              Archived
            </button>
          </div>
        </div>

        <div class="panel-section">
          <div class="panel-heading">Storage</div>
          <div id="fp-stats" class="files-stats">Loading...</div>
        </div>

        <div class="flex-1"></div>
      </div>

      <!-- Main content -->
      <div class="browser-main files-main-drop">
        <!-- Top bar -->
        <div class="browser-header">
          <button id="fp-select-all" class="btn btn-ghost btn-sm">Select All</button>
          <div id="fp-bulk-actions" class="browser-bulk-actions">
            <button id="fp-archive" class="btn btn-ghost btn-sm" disabled>Archive</button>
            <button id="fp-restore" class="btn btn-ghost btn-sm hidden" disabled>Restore</button>
            <button id="fp-delete" class="btn btn-danger btn-sm" disabled>Delete</button>
          </div>
          <button id="fp-cancel" class="btn btn-ghost btn-sm hidden">Cancel</button>
          <div class="flex-1"></div>
          <span id="fp-count" class="files-count"></span>
          <button id="fp-upload" class="btn btn-ghost btn-sm">
            ${ICON_UPLOAD} Upload
          </button>
          <input id="fp-file-input" type="file" multiple style="display:none">
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

  // Create clear button for search.
  const searchInput = root.querySelector("#fp-search")!;
  const clearBtn = document.createElement("button");
  clearBtn.className = "browser-clear-btn hidden";
  clearBtn.id = "fp-search-clear";
  clearBtn.innerHTML = ICON_CLEAR;
  searchInput.parentElement!.appendChild(clearBtn);
}
