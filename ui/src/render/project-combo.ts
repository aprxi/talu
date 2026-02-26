/**
 * Reusable project list — clickable buttons showing available projects.
 *
 * Displays "All", "Default" (sessions with no project), and named projects
 * with counts. A "+" button reveals an inline input for creating new projects.
 *
 * Projects are persisted via the /v1/projects API. A module-level cache
 * avoids blocking renders; call `loadApiProjects()` to prime it.
 */

import { el } from "./helpers.ts";
import type { ApiClient } from "../api.ts";
import type { Project } from "../types.ts";

// ---------------------------------------------------------------------------
// Module-level project cache (loaded from API, used synchronously at render)
// ---------------------------------------------------------------------------

let cachedProjects: Project[] = [];
let apiRef: ApiClient | null = null;

/** Set the API client used for project CRUD. Call once at boot. */
export function initProjectStore(api: ApiClient): void {
  apiRef = api;
}

/** Load projects from the API into the cache. Returns the project list. */
export async function loadApiProjects(): Promise<Project[]> {
  if (!apiRef) return cachedProjects;
  const result = await apiRef.listProjects({ limit: 100 });
  if (result.ok && result.data) {
    cachedProjects = result.data.data;
  }
  return cachedProjects;
}

/** Get cached project list (synchronous — call loadApiProjects first). */
export function getCachedProjects(): Project[] {
  return cachedProjects;
}

/** Create a project via the API and add to cache. Returns the new project name. */
export async function createApiProject(name: string): Promise<string> {
  if (!apiRef) return name;
  const result = await apiRef.createProject({ name });
  if (result.ok && result.data) {
    // Avoid duplicates in cache.
    if (!cachedProjects.some((p) => p.id === result.data!.id)) {
      cachedProjects.push(result.data);
    }
  }
  return name;
}

/** Delete a project by name via the API and remove from cache. */
export async function deleteApiProject(name: string): Promise<void> {
  if (!apiRef) return;
  const project = cachedProjects.find((p) => p.name === name);
  if (project) {
    await apiRef.deleteProject(project.id);
    cachedProjects = cachedProjects.filter((p) => p.id !== project.id);
  }
}

/** One-time migration: move localStorage projects to the API. */
export async function migrateLocalStorageProjects(): Promise<void> {
  const STORAGE_KEY = "talu-user-projects";
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const names: string[] = JSON.parse(raw);
    if (!Array.isArray(names) || names.length === 0) return;

    // Load existing API projects first to avoid duplicates.
    await loadApiProjects();
    const existingNames = new Set(cachedProjects.map((p) => p.name));

    for (const name of names) {
      if (name && !existingNames.has(name)) {
        await createApiProject(name);
      }
    }

    // Clear localStorage after successful migration.
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Migration is best-effort; don't block startup.
  }
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

const EDIT_ICON_SM = `<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg>`;
const DELETE_ICON_SM = `<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>`;

export interface ProjectListOptions {
  /** Currently selected project IDs (empty = All). */
  currentValues: string[];
  /** Available projects from search aggregation (includes __default__). */
  projects: { value: string; count: number }[];
  /** Called with the updated selection array after a toggle. */
  onSelect: (projectIds: string[]) => void;
  /** Called when user creates a new project via the "+ New" input. */
  onCreate: (name: string) => void;
  /** Called after inline rename is committed. Receives old and new name. */
  onRename?: (oldName: string, newName: string) => void;
  /** Called when user clicks the delete action on a project row. */
  onDelete?: (projectName: string) => void;
}

export function renderProjectList(options: ProjectListOptions): HTMLElement {
  const { currentValues, projects, onSelect, onCreate } = options;
  const root = el("div", "project-list");

  // Header: "Projects" label + "+ New" button.
  const header = el("div", "project-list-header");
  header.appendChild(el("span", undefined, "Projects"));

  const addBtn = el("button", "project-list-add", "+ New");
  addBtn.title = "New project";
  header.appendChild(addBtn);
  root.appendChild(header);

  // Items container.
  const items = el("div", "project-list-items");

  // "All" row — clears filter.
  let totalCount = 0;
  for (const p of projects) totalCount += p.count;
  const isAllActive = currentValues.length === 0;
  const allRow = el("div", isAllActive ? "project-list-row active" : "project-list-row");
  allRow.appendChild(el("span", "project-list-row-name", "All"));
  allRow.appendChild(el("span", "project-list-row-count", String(totalCount)));
  allRow.addEventListener("click", () => onSelect([]));
  items.appendChild(allRow);

  // Collect project names already in the aggregation.
  const aggValues = new Set(projects.map((p) => p.value));

  // Render __default__ and named projects from aggregation.
  for (const p of projects) {
    items.appendChild(buildProjectRow(p.value, p.count, currentValues, onSelect, options));
  }

  // Merge API-cached projects that aren't yet in the aggregation (0 sessions).
  for (const project of cachedProjects) {
    if (aggValues.has(project.name)) continue;
    items.appendChild(buildProjectRow(project.name, 0, currentValues, onSelect, options));
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
        void createApiProject(name).then(() => onCreate(name));
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      createSection.classList.add("hidden");
    }
  });

  createInput.addEventListener("blur", () => {
    setTimeout(() => createSection.classList.add("hidden"), 150);
  });

  return root;
}

function buildProjectRow(
  value: string,
  count: number,
  currentValues: string[],
  onSelect: (ids: string[]) => void,
  options: ProjectListOptions,
): HTMLElement {
  const isActive = currentValues.includes(value);
  const row = el("div", isActive ? "project-list-row active" : "project-list-row");
  row.dataset["value"] = value;

  const displayName = value === "__default__" ? "Default" : value;
  const nameSpan = el("span", "project-list-row-name", displayName);
  row.appendChild(nameSpan);
  row.appendChild(el("span", "project-list-row-count", String(count)));

  // Named projects (not __default__) get edit/delete actions on hover.
  const isManageable = value !== "__default__" && (options.onRename || options.onDelete);
  if (isManageable) {
    const actions = el("span", "project-list-row-actions");
    if (options.onRename) {
      const editBtn = el("button", "project-list-row-btn");
      editBtn.dataset["action"] = "edit";
      editBtn.title = "Rename";
      editBtn.innerHTML = EDIT_ICON_SM;
      editBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        beginRowRename(nameSpan, value, options.onRename!);
      });
      actions.appendChild(editBtn);
    }
    if (options.onDelete) {
      const delBtn = el("button", "project-list-row-btn");
      delBtn.dataset["action"] = "delete";
      delBtn.title = "Delete";
      delBtn.innerHTML = DELETE_ICON_SM;
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        options.onDelete!(value);
      });
      actions.appendChild(delBtn);
    }
    row.appendChild(actions);
  }

  row.addEventListener("click", () => {
    const next = [...currentValues];
    const idx = next.indexOf(value);
    if (idx >= 0) {
      next.splice(idx, 1);
    } else {
      next.push(value);
    }
    onSelect(next);
  });

  return row;
}

/** Make a project row name editable inline. */
function beginRowRename(
  nameSpan: HTMLElement,
  oldName: string,
  onCommit: (oldName: string, newName: string) => void,
): void {
  nameSpan.contentEditable = "plaintext-only";
  nameSpan.focus();

  const range = document.createRange();
  range.selectNodeContents(nameSpan);
  const sel = window.getSelection();
  sel?.removeAllRanges();
  sel?.addRange(range);

  let committed = false;
  const commit = () => {
    if (committed) return;
    committed = true;
    nameSpan.removeEventListener("keydown", onKey);
    nameSpan.contentEditable = "false";
    const newName = (nameSpan.textContent ?? "").trim();
    if (!newName || newName === oldName) {
      nameSpan.textContent = oldName;
      return;
    }
    onCommit(oldName, newName);
  };

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); nameSpan.blur(); }
    else if (e.key === "Escape") { e.preventDefault(); nameSpan.textContent = oldName; nameSpan.blur(); }
  };

  nameSpan.addEventListener("keydown", onKey);
  nameSpan.addEventListener("blur", commit, { once: true });
}

// Keep old name as alias for any stale imports during transition.
export { renderProjectList as renderProjectCombo };
