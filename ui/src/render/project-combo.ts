/**
 * Reusable project list — clickable buttons showing available projects.
 *
 * Displays "All", "Default" (sessions with no project), and named projects
 * with counts. A "+" button reveals an inline input for creating new projects.
 *
 * User-created project names are persisted in localStorage so they survive
 * page refreshes even before any sessions are assigned to them.
 */

import { el } from "./helpers.ts";

const STORAGE_KEY = "talu-user-projects";

/** Read user-created project names from localStorage. */
function loadUserProjects(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

/** Save a user-created project name to localStorage. */
export function addUserProject(name: string): void {
  const list = loadUserProjects();
  if (!list.includes(name)) {
    list.push(name);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  }
}

/** Remove a user-created project name from localStorage. */
export function removeUserProject(name: string): void {
  const list = loadUserProjects().filter((n) => n !== name);
  if (list.length > 0) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } else {
    localStorage.removeItem(STORAGE_KEY);
  }
}

export interface ProjectListOptions {
  /** Currently selected project IDs (empty = All). */
  currentValues: string[];
  /** Available projects from search aggregation (includes __default__). */
  projects: { value: string; count: number }[];
  /** Called with the updated selection array after a toggle. */
  onSelect: (projectIds: string[]) => void;
  /** Called when user creates a new project via the "+" input. */
  onCreate: (name: string) => void;
}

export function renderProjectList(options: ProjectListOptions): HTMLElement {
  const { currentValues, projects, onSelect, onCreate } = options;
  const root = el("div", "project-list");

  // Header: "Projects" label + "+" button.
  const header = el("div", "project-list-header");
  const label = el("span", undefined, "Projects");
  header.appendChild(label);

  if (currentValues.length > 0) {
    const clearBtn = el("button", "project-list-clear", "\u00d7");
    clearBtn.title = "Clear selection";
    clearBtn.addEventListener("click", () => onSelect([]));
    header.appendChild(clearBtn);
  }

  const addBtn = el("button", "project-list-add", "+");
  addBtn.title = "New project";
  header.appendChild(addBtn);
  root.appendChild(header);

  // Items container.
  const items = el("div", "project-list-items");

  // Collect project names already in the aggregation.
  const aggValues = new Set(projects.map((p) => p.value));

  // Render __default__ and named projects from aggregation.
  for (const p of projects) {
    const isActive = currentValues.includes(p.value);
    const btn = el("button", isActive ? "project-list-item active" : "project-list-item");
    btn.dataset["value"] = p.value;

    const displayName = p.value === "__default__" ? "Default" : p.value;
    const nameText = document.createTextNode(displayName);
    btn.appendChild(nameText);

    const count = el("span", "count", String(p.count));
    btn.appendChild(count);

    items.appendChild(btn);
  }

  // Merge user-created projects that aren't yet in the aggregation (0 sessions).
  for (const name of loadUserProjects()) {
    if (aggValues.has(name)) continue;
    const isActive = currentValues.includes(name);
    const btn = el("button", isActive ? "project-list-item active" : "project-list-item");
    btn.dataset["value"] = name;
    btn.appendChild(document.createTextNode(name));
    btn.appendChild(el("span", "count", "0"));
    items.appendChild(btn);
  }

  root.appendChild(items);

  // Create input (hidden by default).
  const createSection = el("div", "project-list-create hidden");
  const createInput = el("input", undefined);
  createInput.type = "text";
  createInput.placeholder = "Project name\u2026";
  createSection.appendChild(createInput);
  root.appendChild(createSection);

  // -- Events --

  items.addEventListener("click", (e) => {
    const target = (e.target as HTMLElement).closest<HTMLElement>(".project-list-item");
    if (!target || target.dataset["value"] == null) return;

    const value = target.dataset["value"]!;
    // Toggle: add if not present, remove if present.
    const next = [...currentValues];
    const idx = next.indexOf(value);
    if (idx >= 0) {
      next.splice(idx, 1);
    } else {
      next.push(value);
    }
    onSelect(next);
  });

  addBtn.addEventListener("click", () => {
    createSection.classList.toggle("hidden");
    if (!createSection.classList.contains("hidden")) {
      createInput.value = "";
      createInput.focus();
    }
  });

  createInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const name = createInput.value.trim();
      if (name) {
        createSection.classList.add("hidden");
        addUserProject(name);
        onCreate(name);
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      createSection.classList.add("hidden");
    }
  });

  createInput.addEventListener("blur", () => {
    // Small delay so that Enter can fire before blur hides the input.
    setTimeout(() => createSection.classList.add("hidden"), 150);
  });

  return root;
}

// Keep old name as alias for any stale imports during transition.
export { renderProjectList as renderProjectCombo };
