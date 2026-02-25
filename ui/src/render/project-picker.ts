/**
 * Project picker popover — for right-click "Move to project" context menus.
 *
 * Shows a filterable list of existing projects + "Create" and "Remove" options.
 * Used inside `layout.showPopover()` anchored to a sidebar item or browser card.
 */

import { el } from "./helpers.ts";
import { createApiProject } from "./project-combo.ts";

export interface ProjectPickerOptions {
  /** Current project_id of the session (null if unassigned). */
  currentProjectId: string | null;
  /** Available projects from search aggregation. */
  projects: { value: string; count: number }[];
  /** Called when user picks a project (null = remove from project). */
  onSelect: (projectId: string | null) => void;
}

export function renderProjectPicker(options: ProjectPickerOptions): HTMLElement {
  const { currentProjectId, projects, onSelect } = options;
  const root = el("div", "project-picker");

  const header = el("div", "project-picker-header", "Move to project");
  root.appendChild(header);

  const input = el("input", "project-picker-input");
  input.type = "text";
  input.placeholder = "Type or select\u2026";
  input.autocomplete = "off";
  input.spellcheck = false;
  root.appendChild(input);

  const list = el("div", "project-picker-list");
  root.appendChild(list);

  const createRow = el("div", "project-picker-create hidden");
  const removeRow = el("div", "project-picker-remove");
  removeRow.textContent = "Remove from project";
  if (!currentProjectId) removeRow.classList.add("hidden");

  function renderList(filter: string): void {
    list.innerHTML = "";
    const lower = filter.toLowerCase();

    // "Default" option — move session to no project.
    if (!lower || "default".includes(lower)) {
      const defItem = el("div", "project-picker-item");
      defItem.dataset["value"] = "__default__";
      defItem.textContent = "Default";
      if (!currentProjectId) defItem.classList.add("active");
      list.appendChild(defItem);
    }

    let hasExactMatch = false;
    for (const p of projects) {
      // Skip __default__ entries from aggregation — we render our own above.
      if (p.value === "__default__") continue;
      if (lower && !p.value.toLowerCase().includes(lower)) continue;
      if (p.value.toLowerCase() === lower) hasExactMatch = true;

      const item = el("div", "project-picker-item");
      item.dataset["value"] = p.value;
      item.textContent = p.value;
      if (p.value === currentProjectId) item.classList.add("active");
      list.appendChild(item);
    }

    if (filter.trim() && !hasExactMatch) {
      createRow.textContent = `Create "${filter.trim()}" \u21B5`;
      createRow.dataset["value"] = filter.trim();
      createRow.classList.remove("hidden");
      list.appendChild(createRow);
    } else {
      createRow.classList.add("hidden");
    }
  }

  renderList("");
  root.appendChild(removeRow);

  // -- Events --

  input.addEventListener("input", () => renderList(input.value));

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const val = input.value.trim();
      if (val) {
        // Create project via API if it's a new name.
        const isNew = !projects.some((p) => p.value === val);
        if (isNew) void createApiProject(val);
        onSelect(val);
      }
    }
  });

  list.addEventListener("click", (e) => {
    const target = (e.target as HTMLElement).closest<HTMLElement>(
      ".project-picker-item, .project-picker-create",
    );
    if (target?.dataset["value"] != null) {
      // __default__ means "remove from project" (set project_id to null).
      const val = target.dataset["value"];
      // Create via API if this is the "Create" row.
      if (target.classList.contains("project-picker-create")) {
        void createApiProject(val);
      }
      onSelect(val === "__default__" ? null : val);
    }
  });

  removeRow.addEventListener("click", () => onSelect(null));

  // Auto-focus input after popover is shown.
  requestAnimationFrame(() => input.focus());

  return root;
}
