/**
 * Prompts plugin DOM construction — builds the full layout HTML.
 */

import {
  PLUS_ICON as ICON_PLUS,
  COPY_ICON_LG as ICON_COPY,
  DELETE_ICON as ICON_DELETE,
  EMPTY_DOCUMENT_ICON as ICON_EMPTY,
} from "../../icons.ts";

export function buildPromptsDOM(root: HTMLElement): void {
  root.innerHTML = `
    <div style="display: flex; height: 100%;">
      <!-- Left sidebar: prompt list -->
      <div class="browser-sidebar">
        <div class="panel-section">
          <button id="pp-new-btn" class="btn btn-full">
            ${ICON_PLUS}
            New Prompt
          </button>
        </div>
        <div id="pp-list" class="sidebar-content scroll-thin"></div>
      </div>

      <!-- Main content -->
      <div class="browser-main">
        <!-- Editor -->
        <div id="pp-editor" class="prompts-editor hidden scroll-thin">
          <div class="prompts-editor-content">
            <div class="prompts-editor-header">
              <div class="flex-1"></div>
              <button id="pp-copy-btn" class="btn btn-ghost btn-icon" title="Copy to clipboard">${ICON_COPY}</button>
              <button id="pp-delete-btn" class="btn btn-ghost btn-icon" title="Delete prompt">${ICON_DELETE}</button>
            </div>
            <div class="form-group">
              <label for="pp-name" class="form-label">Name</label>
              <input id="pp-name" type="text" placeholder="e.g., Code Assistant" class="form-input">
            </div>
            <div class="form-group">
              <label for="pp-content" class="form-label">System Prompt</label>
              <textarea id="pp-content" rows="16" placeholder="You are a helpful assistant..." class="form-textarea mono"></textarea>
            </div>
            <div class="prompts-editor-actions">
              <button id="pp-save-btn" class="btn btn-primary">Save</button>
            </div>
          </div>
        </div>

        <!-- Empty state -->
        <div id="pp-empty" class="empty-state">
          ${ICON_EMPTY}
          <p class="empty-state-title">No prompt selected</p>
          <p class="empty-state-desc">Select a prompt from the list or create a new one</p>
        </div>
      </div>
    </div>`;
}
