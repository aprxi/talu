/**
 * Browser plugin DOM construction — builds the full layout HTML.
 */

import {
  SEARCH_ICON as ICON_SEARCH,
  CLOSE_ICON as ICON_CLEAR,
  CHAT_ICON as ICON_CHAT,
  ARCHIVE_BOX_ICON as ICON_ARCHIVE,
} from "../../icons.ts";

const BACK_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m15 18-6-6 6-6"/></svg>`;

const DOCUMENT_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>`;

export function buildBrowserDOM(root: HTMLElement): void {
  root.innerHTML = `
    <div style="display: flex; flex-direction: column; height: 100%; overflow: hidden;">
      <!-- Conversations view (default) -->
      <div id="bp-conversations-view" style="display: flex; height: 100%; overflow: hidden;">
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

          <!-- Projects section -->
          <div id="bp-projects-section" class="panel-section">
            <div class="panel-heading">Project</div>
            <div id="bp-project-combo"></div>
          </div>

          <!-- Tags section -->
          <div id="bp-tags-section" class="panel-section hidden">
            <div class="panel-heading">Tags</div>
            <div id="bp-tags"></div>
          </div>

          <!-- Context section -->
          <div id="bp-context-section" class="panel-section">
            <div class="panel-heading">Context</div>
            <button id="bp-context-btn" class="browser-tab">
              ${DOCUMENT_ICON}
              System Prompts
            </button>
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

      <!-- Context view (hidden — full prompts management UI) -->
      <div id="bp-context-view" class="hidden" style="display: flex; flex-direction: column; height: 100%; overflow: hidden;">
        <div class="bp-context-header">
          <button id="bp-context-back" class="btn btn-ghost btn-sm rp-back-btn">
            ${BACK_ICON} Back
          </button>
          <span class="bp-context-title">System Prompts</span>
        </div>
        <div id="bp-prompts-host" style="flex: 1; overflow: hidden;"></div>
      </div>
    </div>
  `;
}
