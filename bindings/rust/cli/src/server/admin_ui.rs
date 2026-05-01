//! Built-in Talu admin web interface for model serving operations.
//!
//! This page is served directly by the Talu server and consumes `/v1/repo/*`
//! APIs from the same origin.

pub fn html() -> String {
    r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Talu Admin</title>
<style>
:root {
  --bg: #f5f6f7;
  --panel: #ffffff;
  --ink: #1f2937;
  --muted: #6b7280;
  --line: #e5e7eb;
  --accent: #2563eb;
  --accent-ink: #ffffff;
  --warn: #92400e;
  --danger: #b91c1c;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--bg);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.shell {
  max-width: 1180px;
  margin: 0 auto;
  padding: 1rem;
}
.mast {
  display: flex;
  justify-content: space-between;
  gap: 0.8rem;
  align-items: end;
  margin-bottom: 0.9rem;
  border-bottom: 1px solid var(--line);
  padding-bottom: 0.7rem;
}
.brand h1 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
}
.brand p {
  margin: 0.2rem 0 0;
  color: var(--muted);
  font-size: 0.87rem;
}
.top-actions {
  display: flex;
  gap: 0.45rem;
}
button, input {
  font: inherit;
}
button {
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 8px;
  padding: 0.42rem 0.7rem;
  cursor: pointer;
}
button:hover { background: #f8fafc; }
button.primary {
  background: var(--accent);
  color: var(--accent-ink);
  border-color: #1d4ed8;
}
button.primary:hover { background: #1d4ed8; }
button.danger {
  border-color: #fecaca;
  color: var(--danger);
}
button.warn {
  border-color: #fcd9b6;
  color: var(--warn);
}
.icon-btn {
  width: 2.1rem;
  height: 2.1rem;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.icon-btn svg {
  width: 1rem;
  height: 1rem;
  stroke: currentColor;
  fill: none;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.icon-btn.busy {
  opacity: 0.72;
  cursor: default;
}
.spin {
  animation: talu-spin 0.95s linear infinite;
  transform-origin: center;
}
@keyframes talu-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
.workspace {
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
  gap: 0.9rem;
  align-items: start;
}
.side-nav {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.7rem;
}
.side-nav-title {
  margin: 0 0 0.55rem;
  font-size: 0.84rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.deck-nav {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}
.deck-tab {
  text-align: left;
  width: 100%;
  padding: 0.55rem 0.62rem;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: #fff;
}
.deck-tab.active {
  border-color: #93c5fd;
  background: #eff6ff;
}
.content {
  display: grid;
  gap: 0.75rem;
}
.panel {
  display: none;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.85rem;
}
.panel.active { display: block; }
.cards {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.5rem;
}
.card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.62rem;
  background: #fff;
}
.card .k { color: var(--muted); font-size: 0.84rem; }
.card .v { font-size: 1.2rem; margin-top: 0.16rem; font-weight: 600; }
.tool-row {
  margin-top: 0.75rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  align-items: center;
}
.tool-row input, .tool-row select {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.45rem 0.55rem;
  background: #fff;
  color: var(--ink);
}
.tool-row select {
  width: auto;
}
.segmented {
  display: inline-flex;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}
.seg-btn {
  border: 0;
  border-right: 1px solid var(--line);
  border-radius: 0;
  padding: 0.42rem 0.68rem;
  background: #fff;
  color: var(--muted);
}
.seg-btn:last-child { border-right: 0; }
.seg-btn.active {
  background: #eff6ff;
  color: var(--ink);
}
.settings-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.6rem;
}
.field {
  display: grid;
  gap: 0.28rem;
}
.field label {
  font-size: 0.8rem;
  color: var(--muted);
}
.field input {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.45rem 0.55rem;
}
.flag-row {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  align-items: center;
}
.flag {
  display: inline-flex;
  gap: 0.35rem;
  align-items: center;
  font-size: 0.88rem;
}
.table-wrap {
  margin-top: 0.7rem;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: auto;
}
table { width: 100%; border-collapse: collapse; min-width: 680px; }
th, td {
  text-align: left;
  padding: 0.52rem 0.6rem;
  border-bottom: 1px solid #edf0f2;
  font-size: 0.88rem;
  vertical-align: top;
}
th { background: #f9fafb; }
.th-sort {
  border: 0;
  background: transparent;
  padding: 0;
  display: inline-flex;
  align-items: center;
  gap: 0.2rem;
  color: inherit;
  font-weight: 600;
}
.th-sort:hover { background: transparent; }
.sort-ind {
  width: 0.9rem;
  text-align: center;
  color: var(--muted);
}
tr:last-child td { border-bottom: none; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.muted { color: var(--muted); }
.status {
  display: inline-block;
  border-radius: 999px;
  padding: 0.08rem 0.45rem;
  font-size: 0.75rem;
  border: 1px solid #d1d5db;
  background: #f9fafb;
}
.err { color: var(--danger); }
.ok { color: #166534; }
.empty-row td {
  color: var(--muted);
  text-align: center;
  padding: 1rem 0.6rem;
}
#flash {
  min-height: 1.1rem;
  color: var(--muted);
  font-size: 0.86rem;
}
body.dark {
  --bg: #0f172a;
  --panel: #111827;
  --ink: #e5e7eb;
  --muted: #9ca3af;
  --line: #374151;
}
body.dark button {
  background: #1f2937;
  color: var(--ink);
}
body.dark button:hover {
  background: #253142;
}
body.dark button.primary {
  background: #2563eb;
  color: #fff;
}
body.dark .deck-tab {
  background: #111827;
}
body.dark .deck-tab.active {
  background: #1e3a8a;
  border-color: #60a5fa;
}
body.dark .card {
  background: #111827;
}
body.dark th {
  background: #1f2937;
}
body.dark .status {
  background: #1f2937;
  border-color: #4b5563;
}
body.dark .tool-row input,
body.dark .tool-row select {
  background: #111827;
  color: var(--ink);
}
body.dark .seg-btn {
  background: #111827;
}
body.dark .seg-btn.active {
  background: #1e3a8a;
}
body.dark .field input {
  background: #111827;
  color: var(--ink);
}
@media (max-width: 980px) {
  .workspace { grid-template-columns: 1fr; }
  .deck-nav {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .settings-grid { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .mast { flex-direction: column; align-items: stretch; }
  .deck-nav { grid-template-columns: 1fr; }
  .cards { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<main class="shell">
  <header class="mast">
    <div class="brand">
      <h1>Talu</h1>
    </div>
    <div class="top-actions">
      <button id="toggle-theme" class="icon-btn" title="Toggle dark mode" aria-label="Toggle dark mode"></button>
    </div>
  </header>

  <div class="workspace">
    <aside class="side-nav">
      <h2 class="side-nav-title">Manage</h2>
      <nav class="deck-nav" aria-label="Admin panels">
        <button class="deck-tab active" data-panel="overview"><strong>Overview</strong></button>
        <button class="deck-tab" data-panel="discovery"><strong>Discovery</strong></button>
        <button class="deck-tab" data-panel="local"><strong>Local Models</strong></button>
        <button class="deck-tab" data-panel="downloads"><strong>Queue</strong></button>
        <button class="deck-tab" data-panel="settings"><strong>Settings</strong></button>
        <button class="deck-tab" data-panel="api"><strong>API</strong></button>
      </nav>
    </aside>
    <section class="content">

  <section id="panel-overview" class="panel active">
    <div class="cards">
      <article class="card"><div class="k">Total models</div><div class="v" id="ov-total-models">-</div></article>
      <article class="card"><div class="k">Total bytes</div><div class="v" id="ov-total-bytes">-</div></article>
      <article class="card"><div class="k">Managed / Hub</div><div class="v" id="ov-managed-hub">-</div></article>
      <article class="card"><div class="k">Enabled / Ready</div><div class="v" id="ov-enabled-ready">-</div></article>
    </div>
  </section>

  <section id="panel-discovery" class="panel">
    <div class="tool-row">
      <input id="search-query" placeholder="Search models (empty = trending)" aria-label="Search models"/>
      <button id="search-run" class="primary icon-btn" title="Search" aria-label="Search">
        <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"></circle><path d="m21 21-4.35-4.35"></path></svg>
      </button>
      <button id="search-clear" class="icon-btn" title="Clear search" aria-label="Clear search">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"></path></svg>
      </button>
    </div>
    <div class="tool-row">
      <select id="search-sort" aria-label="Search sort">
        <option value="trending">Trending</option>
        <option value="downloads">Downloads</option>
        <option value="likes">Likes</option>
        <option value="last_modified">Modified</option>
      </select>
      <select id="search-direction" aria-label="Search direction">
        <option value="descending">Desc</option>
        <option value="ascending">Asc</option>
      </select>
      <select id="search-limit" aria-label="Search result limit">
        <option value="20">20</option>
        <option value="40" selected>40</option>
        <option value="80">80</option>
      </select>
      <select id="search-task" aria-label="Task filter">
        <option value="text-generation" selected>Text Gen</option>
        <option value="image-text-to-text">Multimodal</option>
        <option value="image-to-text">Image to Text</option>
        <option value="text-to-image">Text to Image</option>
        <option value="text-to-speech">Text to Speech</option>
        <option value="sentence-similarity">Sentence Similarity</option>
        <option value="">Any task</option>
      </select>
      <select id="search-library" aria-label="Library filter">
        <option value="safetensors" selected>safetensors</option>
        <option value="transformers">transformers</option>
        <option value="mlx">mlx</option>
        <option value="sentence-transformers">sentence-transformers</option>
        <option value="">Any library</option>
      </select>
      <select id="search-size" aria-label="Model size filter">
        <option value="1">≤1B</option>
        <option value="2">≤2B</option>
        <option value="4">≤4B</option>
        <option value="8" selected>≤8B</option>
        <option value="16">≤16B</option>
        <option value="32">≤32B</option>
        <option value="64">≤64B</option>
        <option value="128">≤128B</option>
        <option value="512">≤512B</option>
        <option value="any">Any size</option>
      </select>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th><button class="th-sort" data-sort-table="search" data-sort-key="model_id">Model ID<span class="sort-ind"></span></button></th>
            <th><button class="th-sort" data-sort-table="search" data-sort-key="downloads">Downloads<span class="sort-ind"></span></button></th>
            <th><button class="th-sort" data-sort-table="search" data-sort-key="likes">Likes<span class="sort-ind"></span></button></th>
            <th><button class="th-sort" data-sort-table="search" data-sort-key="params_total">Params<span class="sort-ind"></span></button></th>
            <th><button class="th-sort" data-sort-table="search" data-sort-key="last_modified">Modified<span class="sort-ind"></span></button></th>
            <th></th>
          </tr>
        </thead>
        <tbody id="search-rows"></tbody>
      </table>
    </div>
  </section>

  <section id="panel-local" class="panel">
    <div class="tool-row">
      <input id="local-query" placeholder="Filter local models" aria-label="Filter local models"/>
      <div class="segmented" aria-label="Local model source filter">
        <button class="seg-btn active" id="local-source-all" data-source="all" aria-label="All sources">All</button>
        <button class="seg-btn" id="local-source-hub" data-source="hub" aria-label="Hub source only">Hub</button>
        <button class="seg-btn" id="local-source-managed" data-source="managed" aria-label="Managed source only">Managed</button>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th><button class="th-sort" data-sort-table="local" data-sort-key="id">Model ID<span class="sort-ind"></span></button></th>
            <th><button class="th-sort" data-sort-table="local" data-sort-key="source">Source<span class="sort-ind"></span></button></th>
            <th><button class="th-sort" data-sort-table="local" data-sort-key="size_bytes">Size<span class="sort-ind"></span></button></th>
            <th><button class="th-sort" data-sort-table="local" data-sort-key="mtime">Modified<span class="sort-ind"></span></button></th>
            <th><button class="th-sort" data-sort-table="local" data-sort-key="quant_scheme">Quant<span class="sort-ind"></span></button></th>
            <th><button class="th-sort" data-sort-table="local" data-sort-key="architecture">Arch<span class="sort-ind"></span></button></th>
            <th></th>
          </tr>
        </thead>
        <tbody id="local-rows"></tbody>
      </table>
    </div>
  </section>

  <section id="panel-settings" class="panel">
    <div class="settings-grid">
      <div class="field">
        <label for="settings-endpoint">HuggingFace endpoint (optional)</label>
        <input id="settings-endpoint" placeholder="https://huggingface.co" aria-label="HuggingFace endpoint"/>
      </div>
      <div class="field">
        <label for="settings-token">HuggingFace token (optional)</label>
        <input id="settings-token" placeholder="hf_..." aria-label="HuggingFace token"/>
      </div>
    </div>
    <div class="tool-row">
      <div class="flag-row">
        <label class="flag"><input id="settings-force" type="checkbox"/> Force re-download</label>
        <label class="flag"><input id="settings-skip-weights" type="checkbox"/> Skip weight files</label>
      </div>
    </div>
  </section>

  <section id="panel-downloads" class="panel">
    <div class="tool-row">
      <button id="queue-clear-finished" class="icon-btn" title="Clear finished" aria-label="Clear finished">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18"></path><path d="M8 6V4h8v2"></path><path d="M6 6l1 14h10l1-14"></path><path d="M10 11v6M14 11v6"></path></svg>
      </button>
      <button id="queue-cancel-all" class="icon-btn warn" title="Cancel all active/queued" aria-label="Cancel all active/queued">
        <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"></circle><path d="M9 9h6v6H9z"></path></svg>
      </button>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th><button class="th-sort" data-sort-table="downloads" data-sort-key="id">ID<span class="sort-ind"></span></button></th>
            <th><button class="th-sort" data-sort-table="downloads" data-sort-key="model_id">Model<span class="sort-ind"></span></button></th>
            <th><button class="th-sort" data-sort-table="downloads" data-sort-key="status">Status<span class="sort-ind"></span></button></th>
            <th><button class="th-sort" data-sort-table="downloads" data-sort-key="progress">Progress<span class="sort-ind"></span></button></th>
            <th><button class="th-sort" data-sort-table="downloads" data-sort-key="updated_at">Updated<span class="sort-ind"></span></button></th>
            <th></th>
          </tr>
        </thead>
        <tbody id="download-rows"></tbody>
      </table>
    </div>
  </section>

  <section id="panel-api" class="panel">
    <div class="table-wrap" style="overflow:hidden;">
      <iframe id="api-frame" title="Talu API docs" src="about:blank" style="width:100%;height:72vh;border:0;"></iframe>
    </div>
  </section>

  <div id="flash" role="status" aria-live="polite"></div>
    </section>
  </div>
</main>

<script>
const $ = (id) => document.getElementById(id);
const state = { localModels: [], searchResults: [], downloads: [], localSource: "all" };
const panelIds = ["overview", "discovery", "local", "downloads", "settings", "api"];
const THEME_KEY = "talu.admin.theme";
const SETTINGS_KEY = "talu.admin.repo.settings";
const sortState = {
  search: { key: "downloads", dir: "desc" },
  local: { key: "size_bytes", dir: "desc" },
  downloads: { key: "updated_at", dir: "desc" },
};
const MAX_PARAMS_BY_SIZE = {
  "1": 1_500_000_000,
  "2": 2_500_000_000,
  "4": 4_500_000_000,
  "8": 8_500_000_000,
  "16": 16_500_000_000,
  "32": 32_500_000_000,
  "64": 65_000_000_000,
  "128": 130_000_000_000,
  "512": 520_000_000_000,
  any: null,
};
const ICON_SUN = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"></path></svg>';
const ICON_MOON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 1 0 9.8 9.8z"></path></svg>';
const ICON_DOWNLOAD = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12"></path><path d="m7 10 5 5 5-5"></path><path d="M4 21h16"></path></svg>';
const ICON_BUSY = '<svg class="spin" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12a9 9 0 1 1-2.64-6.36"></path></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 6v12M14 6v12"></path></svg>';
const ICON_RESUME = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 6v12l10-6z"></path></svg>';
const ICON_CANCEL = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"></path></svg>';
let searchTimer = null;
let searchAbort = null;
let queuePollTimer = null;
const enqueueingModels = new Set();

function formatCompact(v) {
  const n = Number(v || 0);
  if (!Number.isFinite(n) || n <= 0) return "-";
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function estimateParamsFromModelId(modelId) {
  const m = /[\-_](\d+(?:\.\d+)?)[Bb]\b/.exec(String(modelId || ""));
  if (!m) return 0;
  return Math.floor(Number(m[1]) * 1_000_000_000);
}

function effectiveParams(hit) {
  const raw = Number(hit?.params_total || 0);
  if (Number.isFinite(raw) && raw > 0) return raw;
  return estimateParamsFromModelId(hit?.model_id || "");
}

function currentSettings() {
  return {
    endpoint_url: $("settings-endpoint").value.trim() || null,
    token: $("settings-token").value.trim() || null,
    force: $("settings-force").checked,
    skip_weights: $("settings-skip-weights").checked,
  };
}

function persistSettings() {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(currentSettings()));
}

function loadSettings() {
  const raw = localStorage.getItem(SETTINGS_KEY);
  if (!raw) return;
  try {
    const parsed = JSON.parse(raw);
    $("settings-endpoint").value = parsed?.endpoint_url || "";
    $("settings-token").value = parsed?.token || "";
    $("settings-force").checked = Boolean(parsed?.force);
    $("settings-skip-weights").checked = Boolean(parsed?.skip_weights);
  } catch (_) {}
}

function flash(message, isError = false) {
  const node = $("flash");
  node.textContent = message;
  node.className = isError ? "err" : "muted";
}

function applyTheme(theme) {
  const dark = theme === "dark";
  document.body.classList.toggle("dark", dark);
  $("toggle-theme").innerHTML = dark ? ICON_SUN : ICON_MOON;
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  const systemPrefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(saved || (systemPrefersDark ? "dark" : "light"));
}

function toggleTheme() {
  const next = document.body.classList.contains("dark") ? "light" : "dark";
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
}

function humanBytes(v) {
  const value = Number(v || 0);
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let n = value;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

function toEpochSeconds(value) {
  if (!value) return 0;
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? Math.floor(ms / 1000) : 0;
}

function formatTimestamp(value) {
  const ts = Number(value || 0);
  if (!Number.isFinite(ts) || ts <= 0) return "-";
  return new Date(ts * 1000).toLocaleString();
}

function sortRows(table, rows) {
  const cfg = sortState[table];
  const factor = cfg.dir === "asc" ? 1 : -1;
  const get = (row) => {
    if (table === "search") {
      if (cfg.key === "downloads") return Number(row.downloads || 0);
      if (cfg.key === "likes") return Number(row.likes || 0);
      if (cfg.key === "params_total") return effectiveParams(row);
      if (cfg.key === "last_modified") return toEpochSeconds(row.last_modified);
      return String(row.model_id || "").toLowerCase();
    }
    if (table === "local") {
      if (cfg.key === "size_bytes") return Number(row.size_bytes || 0);
      if (cfg.key === "mtime") return Number(row.mtime || 0);
      if (cfg.key === "source") return String(row.source || "").toLowerCase();
      if (cfg.key === "quant_scheme") return String(row.quant_scheme || "").toLowerCase();
      if (cfg.key === "architecture") return String(row.architecture || "").toLowerCase();
      return String(row.id || "").toLowerCase();
    }
    if (cfg.key === "updated_at") return Number(row.updated_at || 0);
    if (cfg.key === "progress") return Number(row.total || 0) > 0 ? Number(row.current || 0) / Number(row.total) : 0;
    if (cfg.key === "status") return String(row.status || "").toLowerCase();
    if (cfg.key === "model_id") return String(row.model_id || "").toLowerCase();
    return String(row.id || "").toLowerCase();
  };
  return [...rows].sort((a, b) => {
    const av = get(a);
    const bv = get(b);
    if (typeof av === "string" && typeof bv === "string") return av.localeCompare(bv) * factor;
    return ((Number(av) || 0) - (Number(bv) || 0)) * factor;
  });
}

function updateSortIndicators() {
  document.querySelectorAll("button[data-sort-table][data-sort-key]").forEach((btn) => {
    const table = btn.getAttribute("data-sort-table");
    const key = btn.getAttribute("data-sort-key");
    const ind = btn.querySelector(".sort-ind");
    if (!table || !key || !ind) return;
    const cfg = sortState[table];
    ind.textContent = cfg.key === key ? (cfg.dir === "asc" ? "↑" : "↓") : "";
  });
}

function showPanel(name) {
  if (name === "api") ensureApiPanel();
  for (const id of panelIds) {
    const active = id === name;
    $(`panel-${id}`).classList.toggle("active", active);
    document.querySelector(`.deck-tab[data-panel="${id}"]`)?.classList.toggle("active", active);
  }
}

function ensureApiPanel() {
  const frame = $("api-frame");
  if (frame && frame.src !== `${window.location.origin}/docs`) {
    frame.src = "/docs";
  }
}

async function json(path, options = undefined) {
  const resp = await fetch(path, options);
  const payload = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const message = payload?.error?.message || payload?.message || `HTTP ${resp.status}`;
    throw new Error(message);
  }
  return payload;
}

function renderOverview(stats) {
  $("ov-total-models").textContent = String(stats.total_models ?? 0);
  $("ov-total-bytes").textContent = humanBytes(stats.total_size_bytes ?? 0);
  $("ov-managed-hub").textContent = `${stats.managed_models ?? 0} / ${stats.hub_models ?? 0}`;
  $("ov-enabled-ready").textContent = `${stats.enabled_models ?? 0} / ${stats.ready_models ?? 0}`;
}

function setLocalSource(source) {
  state.localSource = source;
  ["all", "hub", "managed"].forEach((name) => {
    const btn = $(`local-source-${name}`);
    if (!btn) return;
    btn.classList.toggle("active", source === name);
  });
}

function renderLocal() {
  const body = $("local-rows");
  body.innerHTML = "";
  const query = $("local-query").value.trim().toLowerCase();
  const source = state.localSource;
  const filtered = state.localModels.filter((row) => {
    const sourceOk = source === "all" || String(row.source || "") === source;
    if (!sourceOk) return false;
    if (!query) return true;
    const id = String(row.id || "").toLowerCase();
    const quant = String(row.quant_scheme || "").toLowerCase();
    const arch = String(row.architecture || "").toLowerCase();
    return id.includes(query) || quant.includes(query) || arch.includes(query);
  });
  const rows = sortRows("local", filtered);
  if (rows.length === 0) {
    body.innerHTML = '<tr class="empty-row"><td colspan="7">No local models.</td></tr>';
    updateSortIndicators();
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    const quant = row.quant_scheme || "-";
    const arch = row.architecture || "-";
    tr.innerHTML = `
      <td class="mono">${row.id}</td>
      <td>${row.source}</td>
      <td>${humanBytes(row.size_bytes)}</td>
      <td>${formatTimestamp(row.mtime)}</td>
      <td>${quant}</td>
      <td>${arch}</td>
      <td><button class="danger" data-action="delete-model" data-id="${row.id}">Remove</button></td>
    `;
    body.appendChild(tr);
  }
  updateSortIndicators();
}

function latestDownloadForModel(modelId) {
  let selected = null;
  for (const item of state.downloads) {
    if (item?.model_id !== modelId) continue;
    if (!selected || Number(item.updated_at || 0) > Number(selected.updated_at || 0)) {
      selected = item;
    }
  }
  return selected;
}

function renderSearch() {
  const body = $("search-rows");
  body.innerHTML = "";
  const sizeKey = $("search-size").value || "any";
  const maxParams = MAX_PARAMS_BY_SIZE[sizeKey] ?? null;
  const filtered = state.searchResults.filter((hit) => {
    if (maxParams === null) return true;
    const params = effectiveParams(hit);
    return params === 0 || params <= maxParams;
  });
  const rows = sortRows("search", filtered);
  if (rows.length === 0) {
    body.innerHTML = '<tr class="empty-row"><td colspan="6">No matching models.</td></tr>';
    updateSortIndicators();
    return;
  }
  for (const hit of rows) {
    const modelId = String(hit.model_id || "");
    const params = effectiveParams(hit);
    const status = String(latestDownloadForModel(modelId)?.status || "");
    const isDownloading = enqueueingModels.has(modelId) || status === "queued" || status === "active";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${modelId}</td>
      <td>${hit.downloads ?? 0}</td>
      <td>${hit.likes ?? 0}</td>
      <td>${formatCompact(params)}</td>
      <td>${formatTimestamp(toEpochSeconds(hit.last_modified))}</td>
      <td>${isDownloading
        ? `<button class="icon-btn busy" disabled title="Downloading" aria-label="Downloading">${ICON_BUSY}</button>`
        : `<button class="primary icon-btn" data-action="queue-model" data-id="${modelId}" title="Download" aria-label="Download">${ICON_DOWNLOAD}</button>`}</td>
    `;
    body.appendChild(tr);
  }
  updateSortIndicators();
}

function flattenDownloads(data) {
  const merged = [];
  for (const key of ["active", "queued", "paused", "recent"]) {
    for (const item of (data[key] || [])) merged.push(item);
  }
  const seen = new Set();
  return merged.filter((entry) => {
    if (!entry || !entry.id || seen.has(entry.id)) return false;
    seen.add(entry.id);
    return true;
  });
}

function renderDownloads() {
  const body = $("download-rows");
  body.innerHTML = "";
  const rows = sortRows("downloads", state.downloads);
  if (rows.length === 0) {
    body.innerHTML = '<tr class="empty-row"><td colspan="6">No downloads yet.</td></tr>';
    updateSortIndicators();
    return;
  }
  for (const item of rows) {
    const pct = item.total > 0 ? Math.floor((item.current / item.total) * 100) : 0;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${item.id}</td>
      <td class="mono">${item.model_id}</td>
      <td><span class="status">${item.status || "-"}</span></td>
      <td>${humanBytes(item.current)} / ${humanBytes(item.total)} (${pct}%)</td>
      <td>${formatTimestamp(item.updated_at)}</td>
      <td>
        <button class="icon-btn warn" data-action="pause-download" data-id="${item.id}" title="Pause" aria-label="Pause">${ICON_PAUSE}</button>
        <button class="icon-btn" data-action="resume-download" data-id="${item.id}" title="Resume" aria-label="Resume">${ICON_RESUME}</button>
        <button class="icon-btn danger" data-action="cancel-download" data-id="${item.id}" title="Cancel" aria-label="Cancel">${ICON_CANCEL}</button>
      </td>
    `;
    body.appendChild(tr);
  }
  updateSortIndicators();
}

async function refreshStats() {
  const stats = await json("/v1/repo/stats");
  renderOverview(stats);
}

async function refreshLocal() {
  const data = await json("/v1/repo/models");
  state.localModels = data.models || [];
  renderLocal();
}

async function refreshSearch(options = undefined) {
  const query = $("search-query").value.trim();
  const sort = $("search-sort").value;
  const direction = $("search-direction").value;
  const limit = $("search-limit").value;
  const filter = $("search-task").value.trim();
  const library = $("search-library").value.trim();
  const settings = currentSettings();
  const params = new URLSearchParams();
  if (query) params.set("query", query);
  params.set("limit", limit || "40");
  params.set("sort", sort || "trending");
  params.set("direction", direction || "descending");
  if (filter) params.set("filter", filter);
  if (library) params.set("library", library);
  if (settings.endpoint_url) params.set("endpoint_url", settings.endpoint_url);
  if (settings.token) params.set("token", settings.token);
  const path = `/v1/repo/search?${params.toString()}`;
  const data = await json(path, options);
  state.searchResults = data.results || [];
  renderSearch();
  flash("");
}

function triggerLiveSearch(delayMs = 120) {
  if (searchTimer) clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    if (searchAbort) searchAbort.abort();
    searchAbort = new AbortController();
    try {
      await refreshSearch({ signal: searchAbort.signal });
    } catch (err) {
      if (err?.name === "AbortError") return;
      flash(err.message || String(err), true);
    }
  }, delayMs);
}

async function refreshDownloads() {
  const data = await json("/v1/repo/downloads");
  state.downloads = flattenDownloads(data);
  renderDownloads();
  renderSearch();
  restartQueuePolling();
}

function hasPendingQueue() {
  return state.downloads.some((item) => {
    const status = String(item?.status || "");
    return status === "active" || status === "queued" || status === "paused";
  }) || enqueueingModels.size > 0;
}

function restartQueuePolling() {
  if (queuePollTimer) clearInterval(queuePollTimer);
  queuePollTimer = null;
  if (!hasPendingQueue()) return;
  queuePollTimer = setInterval(() => {
    void refreshDownloads().catch((err) => flash(err.message || String(err), true));
  }, 1200);
}

async function refreshAll() {
  try {
    await Promise.all([refreshStats(), refreshLocal(), refreshDownloads()]);
  } catch (err) {
    flash(err.message || String(err), true);
  }
}

async function queueModel(modelId) {
  enqueueingModels.add(modelId);
  renderSearch();
  try {
    const settings = currentSettings();
    await json("/v1/repo/downloads", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model_id: modelId,
        token: settings.token,
        endpoint_url: settings.endpoint_url,
        force: settings.force,
        skip_weights: settings.skip_weights,
      }),
    });
  } finally {
    enqueueingModels.delete(modelId);
  }
  await refreshDownloads();
}

async function deleteModel(modelId) {
  await json(`/v1/repo/models/${encodeURIComponent(modelId)}`, { method: "DELETE" });
  await refreshAll();
}

async function controlDownload(id, action) {
  await json(`/v1/repo/downloads/${encodeURIComponent(id)}/${action}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  });
  await refreshDownloads();
}

async function clearFinishedQueue() {
  await json("/v1/repo/downloads/clear/finished", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  });
  await refreshDownloads();
}

async function cancelAllQueue() {
  await json("/v1/repo/downloads/cancel/all", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  });
  await refreshDownloads();
}

document.querySelectorAll(".deck-tab").forEach((node) => {
  node.addEventListener("click", () => showPanel(node.getAttribute("data-panel")));
});

$("toggle-theme").addEventListener("click", toggleTheme);
$("local-query").addEventListener("input", () => renderLocal());
["all", "hub", "managed"].forEach((source) => {
  $(`local-source-${source}`).addEventListener("click", () => {
    setLocalSource(source);
    renderLocal();
  });
});
$("queue-clear-finished").addEventListener("click", () => {
  void clearFinishedQueue().catch((err) => flash(err.message || String(err), true));
});
$("queue-cancel-all").addEventListener("click", () => {
  void cancelAllQueue().catch((err) => flash(err.message || String(err), true));
});

$("search-run").addEventListener("click", async () => {
  if (searchTimer) clearTimeout(searchTimer);
  if (searchAbort) searchAbort.abort();
  searchAbort = null;
  try {
    await refreshSearch();
  } catch (err) {
    flash(err.message || String(err), true);
  }
});
$("search-clear").addEventListener("click", () => {
  $("search-query").value = "";
  triggerLiveSearch(0);
});
$("search-query").addEventListener("input", () => {
  triggerLiveSearch(120);
});
$("search-sort").addEventListener("change", () => { triggerLiveSearch(0); });
$("search-direction").addEventListener("change", () => { triggerLiveSearch(0); });
$("search-limit").addEventListener("change", () => { triggerLiveSearch(0); });
$("search-task").addEventListener("change", () => { triggerLiveSearch(0); });
$("search-library").addEventListener("change", () => { triggerLiveSearch(0); });
$("search-size").addEventListener("change", () => { renderSearch(); });
$("search-query").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    $("search-run").click();
  }
});
["settings-endpoint", "settings-token"].forEach((id) => {
  $(id).addEventListener("change", () => {
    persistSettings();
    triggerLiveSearch(0);
  });
});
["settings-force", "settings-skip-weights"].forEach((id) => {
  $(id).addEventListener("change", () => persistSettings());
});

document.addEventListener("click", (e) => {
  const sortBtn = e.target.closest("button[data-sort-table][data-sort-key]");
  if (!sortBtn) return;
  const table = sortBtn.getAttribute("data-sort-table");
  const key = sortBtn.getAttribute("data-sort-key");
  if (!table || !key || !sortState[table]) return;
  const cfg = sortState[table];
  cfg.dir = cfg.key === key && cfg.dir === "desc" ? "asc" : "desc";
  cfg.key = key;
  if (table === "search") renderSearch();
  else if (table === "local") renderLocal();
  else renderDownloads();
});

document.body.addEventListener("click", (e) => {
  const target = e.target.closest("button[data-action]");
  if (!target) return;
  const action = target.getAttribute("data-action");
  const id = target.getAttribute("data-id");
  if (!action || !id) return;
  void (async () => {
    try {
      if (action === "queue-model") await queueModel(id);
      if (action === "delete-model") await deleteModel(id);
      if (action === "pause-download") await controlDownload(id, "pause");
      if (action === "resume-download") await controlDownload(id, "resume");
      if (action === "cancel-download") await controlDownload(id, "cancel");
    } catch (err) {
      flash(err.message || String(err), true);
    }
  })();
});

void (async () => {
  initTheme();
  loadSettings();
  setLocalSource("all");
  await refreshAll();
  await refreshSearch().catch((err) => flash(err.message || String(err), true));
})();
</script>
</body>
</html>
"##
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::html;

    #[test]
    fn html_contains_control_deck_shell() {
        let page = html();
        assert!(page.contains("<h1>Talu</h1>"));
        assert!(page.contains("data-panel=\"discovery\""));
        assert!(page.contains("data-panel=\"settings\""));
        assert!(page.contains("id=\"search-size\""));
        assert!(page.contains("/v1/repo/downloads"));
    }
}
