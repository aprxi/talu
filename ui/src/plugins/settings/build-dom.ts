/**
 * Settings plugin DOM construction — builds the full layout HTML.
 */

export function buildSettingsDOM(root: HTMLElement): void {
  root.innerHTML = `
    <div class="settings-content scroll-thin">
      <div style="max-width: 672px; margin: 0 auto; padding: 2rem;">
        <h2 style="font-size: 1.125rem; font-weight: 600; color: var(--text); margin-bottom: 1.5rem;">Settings</h2>

        <div class="section-label">General</div>
        <div class="card" style="margin-bottom: 1.5rem;">
          <div class="form-group">
            <label for="sp-model" class="form-label">Model</label>
            <select id="sp-model" class="form-select">
              <option value="">Loading models...</option>
            </select>
          </div>
          <div class="form-group">
            <label for="sp-system-prompt" class="form-label">System Prompt</label>
            <textarea id="sp-system-prompt" rows="3" placeholder="You are a helpful assistant..." class="form-textarea"></textarea>
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div>
              <label for="sp-max-output-tokens" class="form-label">Max Output Tokens</label>
              <input id="sp-max-output-tokens" type="number" step="1" min="1" placeholder="2048" class="form-input">
            </div>
            <div>
              <label for="sp-context-length" class="form-label">Context Length</label>
              <input id="sp-context-length" type="number" step="1" min="1" placeholder="4096" class="form-input">
            </div>
          </div>
        </div>

        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
          <span class="section-label" style="margin-bottom: 0;">Model Sampling</span>
          <span id="sp-model-label" style="font-size: 12px; color: var(--text-muted);"></span>
          <div class="flex-1"></div>
          <button id="sp-reset-model" class="btn btn-ghost btn-sm">Reset to Defaults</button>
        </div>
        <div class="card" style="margin-bottom: 1.5rem;">
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
            <div>
              <label for="sp-temperature" class="form-label">Temperature</label>
              <input id="sp-temperature" type="number" step="0.1" min="0" max="2" placeholder="1.0" class="form-input">
              <div id="sp-temperature-default" class="form-hint"></div>
            </div>
            <div>
              <label for="sp-top-p" class="form-label">Top P</label>
              <input id="sp-top-p" type="number" step="0.05" min="0" max="1" placeholder="1.0" class="form-input">
              <div id="sp-top-p-default" class="form-hint"></div>
            </div>
            <div>
              <label for="sp-top-k" class="form-label">Top K</label>
              <input id="sp-top-k" type="number" step="1" min="0" placeholder="50" class="form-input">
              <div id="sp-top-k-default" class="form-hint"></div>
            </div>
          </div>
        </div>

        <div style="height: 1.5rem;">
          <span id="sp-status" style="font-size: 12px; color: var(--text-muted);"></span>
        </div>
      </div>
    </div>`;
}
