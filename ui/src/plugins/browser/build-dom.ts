/**
 * Browser plugin DOM construction — builds the full layout HTML.
 */

import {
  SEARCH_ICON as ICON_SEARCH,
  CLOSE_ICON as ICON_CLEAR,
  CHAT_ICON as ICON_CHAT,
  ARCHIVE_BOX_ICON as ICON_ARCHIVE,
} from "../../icons.ts";

export function buildBrowserDOM(root: HTMLElement): void {
  root.innerHTML = `
    <div style="display: flex; height: 100%; overflow: hidden;">
      <!-- Left sidebar: filters -->
      <div class="browser-sidebar">
        <!-- Status tabs -->
        <div class="panel-section">
          <div class="panel-heading">Status</div>
          <div>
            <button id="bp-tab-all" class="browser-tab active" data-tab="all">
              ${ICON_CHAT}
              Active
            </button>
            <button id="bp-tab-archived" class="browser-tab" data-tab="archived">
              ${ICON_ARCHIVE}
              Archived
            </button>
          </div>
        </div>

        <!-- Tags section -->
        <div id="bp-tags-section" class="panel-section hidden">
          <div class="panel-heading">Tags</div>
          <div id="bp-tags"></div>
        </div>

        <!-- Spacer -->
        <div class="flex-1"></div>
      </div>

      <!-- Main content -->
      <div class="browser-main">
        <!-- Top bar: actions -->
        <div class="browser-header" id="bp-toolbar">
          <div class="search-wrapper">
            ${ICON_SEARCH}
            <input id="bp-search" type="text" placeholder="Search..." class="search-input">
            <button id="bp-search-clear" class="browser-clear-btn hidden">${ICON_CLEAR}</button>
          </div>
          <div class="flex-1"></div>
          <button id="bp-select-all" class="btn btn-ghost btn-sm">Select All</button>
          <div id="bp-bulk-actions" class="browser-bulk-actions">
            <button id="bp-export" class="btn btn-ghost btn-sm" disabled>Export</button>
            <button id="bp-archive" class="btn btn-ghost btn-sm" disabled>Archive</button>
            <button id="bp-restore" class="btn btn-ghost btn-sm hidden" disabled>Restore</button>
            <button id="bp-delete" class="btn btn-danger btn-sm" disabled>Delete</button>
          </div>
          <button id="bp-cancel" class="btn btn-ghost btn-sm hidden">Cancel</button>
          <div data-slot="browser:toolbar" class="menu-slot"></div>
        </div>

        <!-- Cards grid -->
        <div id="bp-cards" class="card-grid browser-cards scroll-thin"></div>

        <!-- Pagination -->
        <div id="bp-pagination"></div>
      </div>
    </div>
  `;

}
