/**
 * Settings plugin DOM construction — builds the full layout HTML.
 */

export function buildSettingsDOM(root: HTMLElement): void {
  root.innerHTML = `
    <div class="settings-content scroll-thin">
      <div class="settings-page">

        <div class="settings-section">
          <div class="settings-section-title">Appearance</div>
          <div class="settings-row">
            <div class="settings-row-info">
              <div class="settings-row-label">Dark Theme</div>
            </div>
            <div class="settings-row-control">
              <select id="sp-theme-dark" class="form-select"></select>
            </div>
          </div>
          <div class="settings-row">
            <div class="settings-row-info">
              <div class="settings-row-label">Light Theme</div>
            </div>
            <div class="settings-row-control">
              <select id="sp-theme-light" class="form-select"></select>
            </div>
          </div>
          <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; padding: 0.25rem 0;">
            <button id="sp-theme-new" class="btn btn-ghost btn-sm">New Theme</button>
            <button id="sp-theme-import" class="btn btn-ghost btn-sm">Import</button>
          </div>
          <div id="sp-theme-editor"></div>
        </div>

        <div class="settings-section">
          <div class="settings-section-title">Model</div>
          <div style="padding: 0.75rem 0;">
            <select id="sp-model" class="form-select">
              <option value="">Loading models...</option>
            </select>
          </div>
          <details id="sp-model-sampling">
            <summary class="settings-sampling-toggle">
              Sampling parameters
              <span id="sp-model-label"></span>
            </summary>
            <div class="settings-sampling-grid">
              <div>
                <label for="sp-temperature" class="form-label form-label-sm">Temperature</label>
                <input id="sp-temperature" type="number" step="0.1" min="0" max="2" placeholder="1.0" class="form-input form-input-sm">
                <div id="sp-temperature-default" class="form-hint"></div>
              </div>
              <div>
                <label for="sp-top-p" class="form-label form-label-sm">Top P</label>
                <input id="sp-top-p" type="number" step="0.05" min="0" max="1" placeholder="1.0" class="form-input form-input-sm">
                <div id="sp-top-p-default" class="form-hint"></div>
              </div>
              <div>
                <label for="sp-top-k" class="form-label form-label-sm">Top K</label>
                <input id="sp-top-k" type="number" step="1" min="0" placeholder="50" class="form-input form-input-sm">
                <div id="sp-top-k-default" class="form-hint"></div>
              </div>
            </div>
            <div style="text-align: right; margin-top: 0.25rem;">
              <button id="sp-reset-model" class="btn btn-ghost btn-sm" style="font-size: 11px;">Reset to defaults</button>
            </div>
          </details>
        </div>

        <div class="settings-section">
          <div class="settings-section-title">Generation</div>
          <div class="settings-row">
            <div class="settings-row-info">
              <div class="settings-row-label">System Prompt</div>
              <div class="settings-row-desc">
                <span id="sp-system-prompt-name">Default</span>
                <span class="settings-separator">&middot;</span>
                <button id="sp-open-prompts" class="settings-link">Edit in Prompts</button>
              </div>
            </div>
            <label class="toggle">
              <input type="checkbox" id="sp-system-prompt-enabled" checked>
              <span class="toggle-track"></span>
            </label>
          </div>
          <div class="settings-row">
            <div class="settings-row-info">
              <div class="settings-row-label">Max Output Tokens</div>
              <div class="settings-row-desc">Maximum number of tokens per response</div>
            </div>
            <div class="settings-row-control">
              <input id="sp-max-output-tokens" type="number" step="1" min="1" placeholder="2048" class="form-input">
            </div>
          </div>
          <div class="settings-row">
            <div class="settings-row-info">
              <div class="settings-row-label">Context Length</div>
              <div class="settings-row-desc">Maximum context window size</div>
            </div>
            <div class="settings-row-control">
              <input id="sp-context-length" type="number" step="1" min="1" placeholder="4096" class="form-input">
            </div>
          </div>
        </div>

        <div class="settings-section">
          <div class="settings-section-title">Features</div>
          <div class="settings-row">
            <div class="settings-row-info">
              <div class="settings-row-label">Auto-generate titles</div>
              <div class="settings-row-desc">Automatically name new conversations</div>
            </div>
            <label class="toggle">
              <input type="checkbox" id="sp-auto-title" checked>
              <span class="toggle-track"></span>
            </label>
          </div>
        </div>

        <div id="sp-status" class="settings-status"></div>
      </div>
    </div>`;
}
