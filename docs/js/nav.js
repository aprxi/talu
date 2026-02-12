// ══════════════════════════════════════════════════════════════════════════
// Dark / Light Mode Toggle
// ══════════════════════════════════════════════════════════════════════════

const THEMES = ['talu', 'light-talu'];

function getMode() {
  return localStorage.getItem('docs-mode') || 'dark';
}

function setMode(mode) {
  THEMES.forEach(t => document.documentElement.classList.remove(t));
  document.documentElement.classList.add(mode === 'light' ? 'light-talu' : 'talu');
  document.documentElement.removeAttribute('style');
  localStorage.setItem('docs-mode', mode);
}

// Topbar toggle button
const modeBtn = document.getElementById('mode-toggle');
if (modeBtn) {
  modeBtn.addEventListener('click', () => {
    const newMode = getMode() === 'dark' ? 'light' : 'dark';
    setMode(newMode);
  });
}

// Navigation toggle
document.querySelectorAll('.nav-toggle').forEach(toggle => {
  toggle.addEventListener('click', () => {
    toggle.parentElement.classList.toggle('open');
    saveState();
  });
});

function saveState() {
  const open = [];
  document.querySelectorAll('.sidebar nav li.open > a').forEach(a => {
    open.push(a.getAttribute('href'));
  });
  localStorage.setItem('nav-open', JSON.stringify(open));
}

function restoreState() {
  try {
    const open = JSON.parse(localStorage.getItem('nav-open') || '[]');
    open.forEach(href => {
      const a = document.querySelector(`.sidebar nav a[href="${href}"]`);
      if (a) a.parentElement.classList.add('open');
    });
  } catch (e) {}
}

restoreState();

// Icons
const copyIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
const checkIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';
const docIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>';

// Shared copy feedback: swap icon to checkmark, then restore.
// Available globally for page-specific scripts.
window.showCopied = function(btn) {
  var orig = btn.innerHTML;
  btn.innerHTML = checkIcon;
  setTimeout(function() { btn.innerHTML = orig; }, 1500);
};

// Copy buttons for code blocks
document.querySelectorAll('.highlight').forEach(block => {
  const btn = document.createElement('button');
  btn.className = 'copy-btn';
  btn.innerHTML = copyIcon;
  btn.title = 'Copy';
  btn.addEventListener('click', () => {
    const code = block.querySelector('code').textContent;
    navigator.clipboard.writeText(code).then(() => showCopied(btn));
  });
  block.appendChild(btn);
});

// Article copy button (copy original markdown source)
const pageMarkdown = document.getElementById('page-markdown');
const articleCopyBtn = document.querySelector('.article-actions .copy-btn');
if (pageMarkdown && articleCopyBtn) {
  articleCopyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(pageMarkdown.textContent).then(() => showCopied(articleCopyBtn));
  });
}

// Anchor links on headings
document.querySelectorAll('h2[id], h3[id], h4[id]').forEach(heading => {
  const link = document.createElement('a');
  link.href = '#' + heading.id;
  link.className = 'anchor-link';
  link.textContent = '#';
  link.onclick = (e) => {
    e.preventDefault();
    navigator.clipboard.writeText(window.location.origin + window.location.pathname + '#' + heading.id);
    history.pushState(null, null, '#' + heading.id);
  };
  heading.appendChild(link);
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  // Ignore if typing in input (except for search-specific keys handled below)
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
    if (e.target.id === 'search-input') {
      // Search input handles its own keys (Escape, arrows, Enter) via initSearch
    }
    return;
  }

  switch(e.key) {
    case '/': // Focus search
      e.preventDefault();
      const searchInput = document.getElementById('search-input');
      if (searchInput) searchInput.focus();
      break;
    case 'j': window.scrollBy(0, 100); break;
    case 'k': window.scrollBy(0, -100); break;
    case 'n': // Next page
      const next = document.querySelector('.page-nav-next');
      if (next) window.location = next.href;
      break;
    case 'p': // Previous page
      const prev = document.querySelector('.page-nav-prev');
      if (prev) window.location = prev.href;
      break;
    case 'c': // Copy page (with Ctrl/Cmd for accessibility)
      if (e.ctrlKey || e.metaKey) {
        const copyBtn = document.querySelector('.article-actions .copy-btn');
        if (copyBtn) copyBtn.click();
      }
      break;
  }
});

// ══════════════════════════════════════════════════════════════════════════
// Search
// ══════════════════════════════════════════════════════════════════════════

(function initSearch() {
  const input = document.getElementById('search-input');
  const resultsContainer = document.getElementById('search-results');
  if (!input || !resultsContainer) return;

  let searchIndex = null;
  let activeIndex = -1;

  // Compute base path for search result URLs
  // nav.js is always at root, so asset_prefix tells us our depth
  const scripts = document.querySelectorAll('script[src]');
  let basePath = '';
  for (const s of scripts) {
    const src = s.getAttribute('src');
    if (src && src.endsWith('nav.js')) {
      basePath = src.replace('nav.js', '');
      break;
    }
  }

  // Lazy-load the search index on first focus
  async function ensureIndex() {
    if (searchIndex !== null) return;
    try {
      const resp = await fetch(basePath + 'search-index.json');
      searchIndex = await resp.json();
    } catch (e) {
      searchIndex = [];
    }
  }

  function showResults(items) {
    activeIndex = -1;
    if (items.length === 0) {
      resultsContainer.innerHTML = '<div class="search-no-results">No results</div>';
      resultsContainer.classList.remove('hidden');
      return;
    }

    resultsContainer.innerHTML = items.slice(0, 12).map((item, i) =>
      `<a href="${basePath}${item.url}" class="search-result-item" data-index="${i}">
        <div><span class="search-result-name">${item.name}</span><span class="search-result-type">${item.type}</span></div>
        ${item.doc ? `<div class="search-result-doc">${escapeHtml(item.doc)}</div>` : ''}
      </a>`
    ).join('');
    resultsContainer.classList.remove('hidden');
  }

  function hideResults() {
    resultsContainer.classList.add('hidden');
    activeIndex = -1;
  }

  function escapeHtml(text) {
    const el = document.createElement('span');
    el.textContent = text;
    return el.innerHTML;
  }

  function setActive(index) {
    const items = resultsContainer.querySelectorAll('.search-result-item');
    items.forEach(el => el.classList.remove('active'));
    if (index >= 0 && index < items.length) {
      activeIndex = index;
      items[index].classList.add('active');
      items[index].scrollIntoView({ block: 'nearest' });
    } else {
      activeIndex = -1;
    }
  }

  // Search on input
  input.addEventListener('input', async () => {
    await ensureIndex();
    const query = input.value.trim().toLowerCase();
    if (!query) {
      hideResults();
      return;
    }

    const matches = searchIndex.filter(item =>
      item.name.toLowerCase().includes(query) ||
      item.doc.toLowerCase().includes(query)
    );
    showResults(matches);
  });

  // Focus: load index
  input.addEventListener('focus', () => {
    ensureIndex();
    if (input.value.trim()) {
      input.dispatchEvent(new Event('input'));
    }
  });

  // Keyboard navigation within search
  input.addEventListener('keydown', (e) => {
    const items = resultsContainer.querySelectorAll('.search-result-item');
    const count = items.length;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (count > 0) setActive(activeIndex < count - 1 ? activeIndex + 1 : 0);
        break;
      case 'ArrowUp':
        e.preventDefault();
        if (count > 0) setActive(activeIndex > 0 ? activeIndex - 1 : count - 1);
        break;
      case 'Enter':
        e.preventDefault();
        if (activeIndex >= 0 && items[activeIndex]) {
          window.location = items[activeIndex].href;
        }
        break;
      case 'Escape':
        hideResults();
        input.blur();
        break;
    }
  });

  // Click outside closes results
  document.addEventListener('click', (e) => {
    if (!input.contains(e.target) && !resultsContainer.contains(e.target)) {
      hideResults();
    }
  });
})();
