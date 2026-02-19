/**
 * Blob file picker — modal overlay for selecting existing files from the
 * file store as chat attachments. Mirrors the file manager table layout.
 */

import { api, notifications, format } from "./deps.ts";
import type { FileObject } from "../../types.ts";

// -- Helpers (matching file manager render.ts) --------------------------------

type SortColumn = "name" | "kind" | "size" | "date";
type SortDir = "asc" | "desc";

const SORT_ASC = " \u25b2";   // ▲
const SORT_DESC = " \u25bc";  // ▼

const SORT_LABELS: Record<SortColumn, string> = {
  name: "Name",
  kind: "Kind",
  size: "Size",
  date: "Date",
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function kindBadgeClass(kind?: string, mimeType?: string): string {
  if (kind === "image" || mimeType?.startsWith("image/")) return "files-kind-badge files-kind-image";
  if (kind === "text" || mimeType?.startsWith("text/")) return "files-kind-badge files-kind-text";
  return "files-kind-badge files-kind-binary";
}

function kindLabel(kind?: string, mimeType?: string): string {
  if (kind) return kind;
  if (mimeType?.startsWith("image/")) return "image";
  if (mimeType?.startsWith("text/")) return "text";
  if (mimeType?.startsWith("application/pdf")) return "document";
  return "binary";
}

// -- Picker -------------------------------------------------------------------

/** Open the blob picker and return selected files (empty on cancel). */
export function openBlobPicker(): Promise<FileObject[]> {
  return new Promise((resolve) => {
    let allFiles: FileObject[] = [];
    let previewedFile: FileObject | null = null;
    let searchQuery = "";
    let sortBy: SortColumn = "name";
    let sortDir: SortDir = "asc";

    // --- Overlay ---
    const overlay = document.createElement("div");
    overlay.className = "blob-picker-overlay";

    const modal = document.createElement("div");
    modal.className = "blob-picker";

    // --- Header ---
    const header = document.createElement("div");
    header.className = "blob-picker-header";

    const title = document.createElement("span");
    title.textContent = "Choose from library";
    header.appendChild(title);

    const searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.className = "blob-picker-search";
    searchInput.placeholder = "Search files...";
    header.appendChild(searchInput);

    modal.appendChild(header);

    // --- Body (table + preview side by side) ---
    const body = document.createElement("div");
    body.className = "blob-picker-body";
    body.innerHTML = '<div class="blob-picker-loading"><div class="spinner"></div></div>';

    const previewPane = document.createElement("div");
    previewPane.className = "blob-picker-preview hidden";

    const bodyWrap = document.createElement("div");
    bodyWrap.className = "blob-picker-body-wrap";
    bodyWrap.appendChild(body);
    bodyWrap.appendChild(previewPane);
    modal.appendChild(bodyWrap);

    // --- Footer ---
    const footer = document.createElement("div");
    footer.className = "blob-picker-footer";

    const openFmBtn = document.createElement("button");
    openFmBtn.className = "btn btn-ghost btn-sm";
    openFmBtn.textContent = "Open file manager";
    footer.appendChild(openFmBtn);

    const countEl = document.createElement("span");
    countEl.className = "blob-picker-count";
    footer.appendChild(countEl);

    const spacer = document.createElement("div");
    spacer.className = "flex-1";
    footer.appendChild(spacer);

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "btn btn-ghost btn-sm";
    cancelBtn.textContent = "Cancel";

    const addBtn = document.createElement("button");
    addBtn.className = "btn btn-primary blob-picker-add-btn";
    addBtn.textContent = "Add";

    footer.appendChild(cancelBtn);
    modal.appendChild(footer);
    overlay.appendChild(modal);

    // --- Helpers ---
    function getFiltered(): FileObject[] {
      const q = searchQuery.toLowerCase().trim();
      const filtered = q
        ? allFiles.filter((f) => f.filename.toLowerCase().includes(q))
        : [...allFiles];

      const dir = sortDir === "asc" ? 1 : -1;
      filtered.sort((a, b) => {
        switch (sortBy) {
          case "name":
            return dir * a.filename.localeCompare(b.filename);
          case "kind":
            return dir * (a.kind ?? "").localeCompare(b.kind ?? "");
          case "size":
            return dir * (a.bytes - b.bytes);
          case "date":
            return dir * (a.created_at - b.created_at);
          default:
            return 0;
        }
      });

      return filtered;
    }

    function isImageFile(f: FileObject): boolean {
      return f.kind === "image" || (f.mime_type?.startsWith("image/") ?? false);
    }

    function showPreview(file: FileObject) {
      previewedFile = file;
      addBtn.disabled = false;
      addBtn.textContent = "Add";

      previewPane.innerHTML = "";

      // Image preview
      if (isImageFile(file)) {
        const img = document.createElement("img");
        img.className = "files-preview-img";
        img.src = `/v1/files/${encodeURIComponent(file.id)}/content`;
        img.alt = file.filename;
        if (file.image) {
          img.style.aspectRatio = `${file.image.width} / ${file.image.height}`;
        }
        previewPane.appendChild(img);
      }

      // Metadata (same layout as file manager)
      const meta = document.createElement("dl");
      meta.className = "files-preview-meta";
      const addField = (label: string, value: string) => {
        const dt = document.createElement("dt");
        dt.textContent = label;
        meta.appendChild(dt);
        const dd = document.createElement("dd");
        dd.textContent = value;
        meta.appendChild(dd);
      };
      addField("Filename", file.filename);
      addField("Size", formatSize(file.bytes));
      addField("Kind", kindLabel(file.kind, file.mime_type));
      if (file.mime_type) addField("MIME", file.mime_type);
      if (file.image) {
        addField("Dimensions", `${file.image.width}\u00d7${file.image.height}`);
        addField("Format", file.image.format);
      }
      addField("Uploaded", format.dateTime(file.created_at * 1000));
      previewPane.appendChild(meta);

      previewPane.appendChild(addBtn);
      previewPane.classList.remove("hidden");

      // Highlight the active row
      const rows = body.querySelectorAll<HTMLElement>(".files-row");
      for (const r of rows) {
        r.classList.toggle("files-row-selected", r.dataset["id"] === file.id);
      }
    }

    function confirmFile(file: FileObject) {
      cleanup();
      resolve([file]);
    }

    function cleanup() {
      document.removeEventListener("keydown", onKey);
      overlay.remove();
    }

    function dismiss() {
      cleanup();
      resolve([]);
    }

    function confirm() {
      if (previewedFile) {
        confirmFile(previewedFile);
      }
    }

    function updateSortHeaders(thead: HTMLElement) {
      const headers = thead.querySelectorAll<HTMLElement>("[data-sort]");
      for (const th of headers) {
        const col = th.dataset["sort"] as SortColumn;
        const label = SORT_LABELS[col] ?? col;
        if (col === sortBy) {
          th.textContent = label + (sortDir === "asc" ? SORT_ASC : SORT_DESC);
          th.classList.add("files-th-sorted");
        } else {
          th.textContent = label;
          th.classList.remove("files-th-sorted");
        }
      }
    }

    // --- Render table ---
    function renderTable() {
      body.innerHTML = "";
      const filtered = getFiltered();
      countEl.textContent = `${filtered.length} file${filtered.length !== 1 ? "s" : ""}`;

      if (filtered.length === 0) {
        const empty = document.createElement("div");
        empty.className = "blob-picker-empty";
        empty.textContent = allFiles.length === 0
          ? "No files in library. Upload files via the Files tab."
          : "No matching files.";
        body.appendChild(empty);
        return;
      }

      const tableContainer = document.createElement("div");
      tableContainer.className = "files-table-container";

      const table = document.createElement("table");
      table.className = "files-table";

      // Thead
      const thead = document.createElement("thead");
      thead.className = "files-thead";
      thead.innerHTML = `<tr>
        <th class="files-th files-th-name" data-sort="name">Name</th>
        <th class="files-th files-th-kind" data-sort="kind">Kind</th>
        <th class="files-th files-th-size" data-sort="size">Size</th>
        <th class="files-th files-th-date" data-sort="date">Date</th>
      </tr>`;
      updateSortHeaders(thead);

      thead.addEventListener("click", (e) => {
        const th = (e.target as HTMLElement).closest<HTMLElement>("[data-sort]");
        if (!th) return;
        const col = th.dataset["sort"] as SortColumn;
        if (sortBy === col) {
          sortDir = sortDir === "asc" ? "desc" : "asc";
        } else {
          sortBy = col;
          sortDir = "asc";
        }
        renderTable();
      });

      table.appendChild(thead);

      // Tbody
      const tbody = document.createElement("tbody");

      for (const file of filtered) {
        const row = document.createElement("tr");
        row.className = `files-row${previewedFile?.id === file.id ? " files-row-selected" : ""}`;
        row.dataset["id"] = file.id;

        // Name
        const nameCell = document.createElement("td");
        nameCell.className = "files-cell files-cell-name";
        nameCell.textContent = file.filename;
        row.appendChild(nameCell);

        // Kind
        const kindCell = document.createElement("td");
        kindCell.className = "files-cell";
        const badge = document.createElement("span");
        badge.className = kindBadgeClass(file.kind, file.mime_type);
        badge.textContent = kindLabel(file.kind, file.mime_type);
        kindCell.appendChild(badge);
        row.appendChild(kindCell);

        // Size
        const sizeCell = document.createElement("td");
        sizeCell.className = "files-cell files-cell-mono";
        sizeCell.textContent = formatSize(file.bytes);
        row.appendChild(sizeCell);

        // Date
        const dateCell = document.createElement("td");
        dateCell.className = "files-cell files-cell-date";
        dateCell.textContent = format.dateTime(file.created_at * 1000, "short");
        row.appendChild(dateCell);

        // Click → preview
        row.addEventListener("click", () => showPreview(file));

        tbody.appendChild(row);
      }

      table.appendChild(tbody);
      tableContainer.appendChild(table);
      body.appendChild(tableContainer);
    }

    // --- Events ---
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        dismiss();
      }
      if (e.key === "Enter" && previewedFile) {
        e.preventDefault();
        confirmFile(previewedFile);
      }
    };
    document.addEventListener("keydown", onKey);

    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) dismiss();
    });

    cancelBtn.addEventListener("click", dismiss);
    addBtn.addEventListener("click", confirm);

    openFmBtn.addEventListener("click", () => {
      cleanup();
      resolve([]);
      // Switch to files mode by clicking the activity bar button.
      document.querySelector<HTMLElement>('[data-mode="files"]')?.click();
    });

    let searchTimer: ReturnType<typeof setTimeout> | null = null;
    searchInput.addEventListener("input", () => {
      if (searchTimer) clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        searchQuery = searchInput.value;
        previewedFile = null;
        previewPane.classList.add("hidden");
        addBtn.disabled = true;
        addBtn.textContent = "Add";
        renderTable();
      }, 150);
    });

    // --- Fetch and display ---
    document.body.appendChild(overlay);
    searchInput.focus();

    api.listFiles({ limit: 100, marker: "active" }).then((res) => {
      if (!res.ok || !res.data) {
        notifications.error(res.error ?? "Failed to load files");
        dismiss();
        return;
      }
      allFiles = res.data.data;
      renderTable();
    }).catch(() => {
      notifications.error("Failed to load files");
      dismiss();
    });
  });
}
