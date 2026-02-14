import { getChatDom } from "./dom.ts";
import { api, notifications } from "./deps.ts";
import { chatState, type ChatAttachment } from "./state.ts";

function escapeHtml(input: string): string {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderAttachmentList(container: HTMLElement): void {
  if (chatState.attachments.length === 0) {
    container.innerHTML = "";
    container.classList.add("hidden");
    return;
  }

  const chips = chatState.attachments.map((attachment) => {
    const name = escapeHtml(attachment.file.filename);
    const bytes = attachment.file.bytes.toLocaleString();
    return `
      <span class="attachment-pill" title="${name}">
        <span class="attachment-name">${name}</span>
        <span class="attachment-meta">${bytes} bytes</span>
        <button class="attachment-remove" data-remove-attachment="${attachment.file.id}" title="Remove attachment">×</button>
      </span>
    `;
  }).join("");

  container.innerHTML = `<div class="attachment-list-inner">${chips}</div>`;
  container.classList.remove("hidden");
}

function renderAttachmentLists(): void {
  const dom = getChatDom();
  renderAttachmentList(dom.welcomeAttachmentList);
  renderAttachmentList(dom.inputAttachmentList);
}

function setUploadBusy(busy: boolean): void {
  chatState.isUploadingAttachments = busy;
  const dom = getChatDom();
  dom.welcomeAttach.disabled = busy || chatState.isGenerating;
  dom.inputAttach.disabled = busy || chatState.isGenerating;
  dom.fileInput.disabled = busy || chatState.isGenerating;
  if (!chatState.isGenerating) {
    dom.welcomeSend.disabled = busy;
    dom.inputSend.disabled = busy;
  }
}

async function uploadFiles(files: FileList): Promise<void> {
  if (files.length === 0) return;
  if (chatState.isUploadingAttachments) return;

  setUploadBusy(true);
  let uploaded = 0;

  try {
    for (const file of Array.from(files)) {
      const result = await api.uploadFile(file);
      if (!result.ok || !result.data) {
        notifications.error(result.error ?? `Failed to upload ${file.name}`);
        continue;
      }

      const attachment: ChatAttachment = {
        file: result.data,
        mimeType: file.type || null,
      };
      chatState.attachments.push(attachment);
      uploaded += 1;
    }
    if (uploaded > 0) {
      notifications.info(`Attached ${uploaded} file${uploaded === 1 ? "" : "s"}`);
    }
  } finally {
    setUploadBusy(false);
    renderAttachmentLists();
  }
}

function removeAttachment(fileId: string): void {
  chatState.attachments = chatState.attachments.filter((item) => item.file.id !== fileId);
  renderAttachmentLists();
}

function onAttachmentClick(event: Event): void {
  const target = event.target as HTMLElement | null;
  const button = target?.closest<HTMLButtonElement>("button[data-remove-attachment]");
  const fileId = button?.dataset["removeAttachment"];
  if (!fileId) return;
  removeAttachment(fileId);
}

export function setupAttachmentEvents(): void {
  const dom = getChatDom();

  dom.welcomeAttach.addEventListener("click", () => {
    if (chatState.isUploadingAttachments || chatState.isGenerating) return;
    dom.fileInput.value = "";
    dom.fileInput.click();
  });

  dom.inputAttach.addEventListener("click", () => {
    if (chatState.isUploadingAttachments || chatState.isGenerating) return;
    dom.fileInput.value = "";
    dom.fileInput.click();
  });

  dom.fileInput.addEventListener("change", () => {
    if (!dom.fileInput.files) return;
    void uploadFiles(dom.fileInput.files);
  });

  dom.welcomeAttachmentList.addEventListener("click", onAttachmentClick);
  dom.inputAttachmentList.addEventListener("click", onAttachmentClick);

  renderAttachmentLists();
}

export function hasAttachments(): boolean {
  return chatState.attachments.length > 0;
}

export function isAttachmentUploadInProgress(): boolean {
  return chatState.isUploadingAttachments;
}

export function clearAttachments(): void {
  chatState.attachments = [];
  renderAttachmentLists();
}

export function buildAttachmentReferencePrefix(attachments: ChatAttachment[]): string {
  return attachments
    .map((attachment) => {
      const attrs = [
        `id=${attachment.file.id}`,
        `name=${attachment.file.filename}`,
        `bytes=${attachment.file.bytes}`,
      ];
      if (attachment.mimeType) attrs.push(`mime=${attachment.mimeType}`);
      return `[Attachment ${attrs.join(", ")}]`;
    })
    .join("\n");
}

export function composeUserInputWithAttachments(text: string): string {
  const trimmed = text.trim();
  if (chatState.attachments.length === 0) return trimmed;

  const prefix = buildAttachmentReferencePrefix(chatState.attachments);
  if (!trimmed) return `${prefix}\n\nPlease use the attached files above.`;
  return `${prefix}\n\n${trimmed}`;
}

