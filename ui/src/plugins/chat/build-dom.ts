import { SEARCH_ICON as ICON_SEARCH, CLOSE_ICON as ICON_CLEAR, FOLDER_PLUS_ICON, COLLAPSE_ALL_ICON, SORT_RECENT_ICON, SORT_CREATED_ICON } from "../../icons.ts";

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
        <div id="transcript" class="transcript scroll-thin"></div>

        <div id="welcome-state" class="welcome-state">
          <span id="welcome-project" class="welcome-project"></span>
          <h2 class="welcome-title">Ask anything</h2>
          <span class="welcome-brand">TALU</span>
          <div class="input-container">
            <div id="welcome-attachment-list" class="attachment-list hidden"></div>
            <textarea id="welcome-input" rows="1" placeholder="Send a message..." class="input-textarea"></textarea>
            <div class="input-footer">
              <select id="welcome-model" class="form-select form-select-inline">
                <option value="">Loading...</option>
              </select>
              <select id="welcome-prompt" class="form-select form-select-inline" title="System prompt">
                <option value="">No prompt</option>
              </select>
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
