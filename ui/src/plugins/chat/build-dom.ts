import { SEARCH_ICON as ICON_SEARCH, CLOSE_ICON as ICON_CLEAR, FOLDER_PLUS_ICON, COLLAPSE_ALL_ICON, SORT_RECENT_ICON, SORT_CREATED_ICON, SETTINGS_ICON, EDIT_ICON } from "../../icons.ts";

/** Build the full chat DOM into the plugin's shadow root container. */
export function buildChatDOM(container: HTMLElement): void {
  // Apply mode-view layout directly on the container since the
  // .mode-view class is on the host element outside the shadow root.
  container.style.display = "flex";
  container.style.height = "100%";
  container.style.overflow = "hidden";

  container.innerHTML = `
    <aside class="sidebar">
      <div class="sidebar-search">
        <div class="search-wrapper">
          ${ICON_SEARCH}
          <input id="sidebar-search" type="text" placeholder="Search..." class="search-input">
          <button id="sidebar-search-clear" class="browser-clear-btn hidden">${ICON_CLEAR}</button>
        </div>
      </div>
      <div class="sidebar-toolbar">
        <button id="sidebar-new-project-btn" class="sidebar-toolbar-btn" title="New project">${FOLDER_PLUS_ICON}</button>
        <div class="flex-1"></div>
        <button id="sidebar-collapse-all-btn" class="sidebar-toolbar-btn" title="Collapse all">${COLLAPSE_ALL_ICON}</button>
        <button id="sidebar-sort-btn" class="sidebar-toolbar-btn" title="Sorted by recent activity">${SORT_RECENT_ICON}</button>
      </div>
      <div class="sidebar-content scroll-thin">
        <div id="sidebar-list">
          <div id="loader-sentinel" class="empty-state hidden">
            <div class="spinner"></div>
            <a href="?safe=true" style="display:block;margin-top:0.75rem;font-size:12px;color:var(--text-muted,#6c7086);text-decoration:underline;">Safe mode</a>
          </div>
        </div>
      </div>
    </aside>

    <div class="chat-column">
      <div class="content-area">
        <div id="generation-bar" class="generation-bar hidden">
          <div class="generation-bar-dots">
            <span></span><span></span><span></span>
          </div>
          <button id="generation-stop" class="generation-bar-stop" title="Stop generating">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><rect x="9" y="9" width="6" height="6" rx="1" fill="currentColor" stroke="none"/></svg>
          </button>
        </div>
        <div id="transcript" class="transcript scroll-thin"></div>

        <div id="welcome-state" class="welcome-state">
          <span id="welcome-project" class="welcome-project"></span>
          <h2 class="welcome-title">Ask anything</h2>
          <span class="welcome-brand">TALU</span>
          <div class="input-container">
            <button id="welcome-generation" class="welcome-gen-icon" title="Settings">
              ${SETTINGS_ICON}
            </button>
            <div id="welcome-attachment-list" class="attachment-list hidden"></div>
            <textarea id="welcome-input" rows="1" placeholder="Send a message..." class="input-textarea"></textarea>
            <div class="input-footer">
              <span class="model-select-wrap">
                <select id="welcome-model" class="form-select form-select-inline">
                  <option value="">Loading...</option>
                </select>
              </span>
              <div class="flex-1"></div>
              <button id="welcome-attach" class="btn btn-ghost btn-icon" title="Upload file">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.48l9.2-9.2a4 4 0 0 1 5.65 5.66l-9.2 9.19a2 2 0 0 1-2.82-2.82l8.49-8.48"/></svg>
              </button>
              <button id="welcome-library" class="btn btn-ghost btn-icon" title="Choose from library">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/></svg>
              </button>
              <button id="welcome-send" class="btn btn-primary btn-send">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
              </button>
            </div>
          </div>
          <div id="welcome-advanced" class="welcome-advanced hidden">
            <div id="welcome-panel-sampling" class="welcome-advanced-panel">
              <div id="welcome-variant-row" class="welcome-variant-row hidden">
                <div id="welcome-variant-pills" class="welcome-variant-pills"></div>
              </div>
              <div id="welcome-sampling-controls" class="welcome-advanced-grid">
                <div class="welcome-advanced-field">
                  <label class="form-label form-label-sm" for="welcome-temperature">Temperature</label>
                  <input id="welcome-temperature" type="number" class="form-input form-input-sm" step="0.1" min="0" max="2" placeholder="1.0">
                </div>
                <div class="welcome-advanced-field">
                  <label class="form-label form-label-sm" for="welcome-top-p">Top P</label>
                  <input id="welcome-top-p" type="number" class="form-input form-input-sm" step="0.05" min="0" max="1" placeholder="1.0">
                </div>
                <div class="welcome-advanced-field">
                  <label class="form-label form-label-sm" for="welcome-top-k">Top K</label>
                  <input id="welcome-top-k" type="number" class="form-input form-input-sm" step="1" min="0" placeholder="50">
                </div>
              </div>
            </div>
            <div id="welcome-panel-generation" class="welcome-advanced-panel hidden">
              <div class="welcome-gen-row">
                <div class="welcome-advanced-field welcome-gen-field-compact">
                  <label class="form-label form-label-sm" for="welcome-max-tokens">Max Tokens</label>
                  <input id="welcome-max-tokens" type="number" class="form-input form-input-sm" step="1" min="1" placeholder="2048">
                </div>
                <div class="welcome-advanced-field welcome-gen-field-compact">
                  <label class="form-label form-label-sm" for="welcome-context-length">Context Length</label>
                  <input id="welcome-context-length" type="number" class="form-input form-input-sm" step="1" min="1" placeholder="4096">
                </div>
              </div>
              <div class="welcome-gen-prompt-row">
                <label class="form-label form-label-sm">System Prompt</label>
                <div class="welcome-gen-prompt-controls">
                  <span class="model-select-wrap">
                    <button id="welcome-prompt-edit" class="model-select-icon" title="Edit prompts">
                      ${EDIT_ICON}
                    </button>
                    <select id="welcome-prompt" class="form-select form-select-sm model-select-has-icon welcome-gen-prompt-select">
                    </select>
                  </span>
                  <label class="toggle toggle-sm" title="Enable system prompt">
                    <input type="checkbox" id="welcome-prompt-enabled" checked>
                    <span class="toggle-track"></span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div id="input-bar" class="input-bar hidden">
          <div class="input-container">
            <div id="input-attachment-list" class="attachment-list hidden"></div>
            <textarea id="input-text" rows="1" placeholder="Send a message..." class="input-textarea"></textarea>
            <div class="input-footer">
              <div class="flex-1"></div>
              <button id="input-attach" class="btn btn-ghost btn-icon" title="Upload file">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.48l9.2-9.2a4 4 0 0 1 5.65 5.66l-9.2 9.19a2 2 0 0 1-2.82-2.82l8.49-8.48"/></svg>
              </button>
              <button id="input-library" class="btn btn-ghost btn-icon" title="Choose from library">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/></svg>
              </button>
              <button id="input-send" class="btn btn-primary btn-send">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>`;

  const fileInput = document.createElement("input");
  fileInput.id = "chat-file-input";
  fileInput.type = "file";
  fileInput.multiple = true;
  fileInput.className = "hidden";
  container.appendChild(fileInput);
}
