import { getChatDom } from "./dom.ts";
import { hooks, notifications, upload } from "./deps.ts";
import { chatState, type ChatAttachment } from "./state.ts";
import type { InputContentItem, InputContentPart } from "../../types.ts";

interface ChatUploadBeforePayload {
  filename: string;
  mimeType: string | null;
  size: number;
  purpose: string;
}

function escapeHtml(input: string): string {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function isBlocked(value: unknown): value is { $block: true; reason: string } {
  return Boolean(
    value
      && typeof value === "object"
      && "$block" in value
      && (value as { $block?: unknown }).$block === true,
  );
}

function isValidAttachment(value: unknown): value is ChatAttachment {
  if (!value || typeof value !== "object") return false;
  const candidate = value as { file?: unknown; mimeType?: unknown };
  if (!candidate.file || typeof candidate.file !== "object") return false;

  const file = candidate.file as {
    id?: unknown;
    filename?: unknown;
    bytes?: unknown;
    createdAt?: unknown;
    purpose?: unknown;
  };

  return (
    typeof file.id === "string"
    && typeof file.filename === "string"
    && typeof file.bytes === "number"
    && typeof file.createdAt === "number"
    && typeof file.purpose === "string"
    && (candidate.mimeType === null || typeof candidate.mimeType === "string")
  );
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
      try {
        const beforeUpload = await hooks.run<ChatUploadBeforePayload>("chat.upload.before", {
          filename: file.name,
          mimeType: file.type || null,
          size: file.size,
          purpose: "assistants",
        });
        if (isBlocked(beforeUpload)) {
          notifications.warning(beforeUpload.reason || `Upload blocked for ${file.name}`);
          continue;
        }

        const purposeOverride = (
          beforeUpload
          && typeof beforeUpload === "object"
          && "purpose" in beforeUpload
          && typeof (beforeUpload as { purpose?: unknown }).purpose === "string"
        )
          ? (beforeUpload as { purpose: string }).purpose.trim()
          : "";
        const purpose = purposeOverride
          ? purposeOverride
          : "assistants";
        const fileRef = await upload.upload(file, purpose);

        let attachment: ChatAttachment = {
          file: fileRef,
          mimeType: file.type || null,
        };
        const afterUpload = await hooks.run("chat.upload.after", attachment);
        if (isBlocked(afterUpload)) {
          void upload.delete(fileRef.id).catch(() => {});
          notifications.warning(afterUpload.reason || `Upload blocked for ${file.name}`);
          continue;
        }
        if (isValidAttachment(afterUpload)) {
          attachment = afterUpload;
        }

        chatState.attachments.push(attachment);
        uploaded += 1;
      } catch (err) {
        const msg = err instanceof Error ? err.message : `Failed to upload ${file.name}`;
        notifications.error(msg);
      }
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

/** Build structured input content parts from attachments. */
function buildAttachmentContentParts(attachments: ChatAttachment[]): InputContentPart[] {
  return attachments.map((attachment): InputContentPart => {
    const mime = attachment.mimeType ?? "";
    if (mime.startsWith("image/")) {
      return { type: "input_image", image_url: attachment.file.id };
    }
    return {
      type: "input_file",
      file_url: attachment.file.id,
      filename: attachment.file.filename,
    };
  });
}

/**
 * Build structured input for the /v1/responses endpoint.
 *
 * When attachments are present, returns an array of InputContentItem objects
 * so the server can resolve file references into actual multimodal content.
 * When no attachments exist, returns the plain text string.
 */
export function composeUserInput(text: string): string | InputContentItem[] {
  const trimmed = text.trim();
  if (chatState.attachments.length === 0) return trimmed;

  const parts: InputContentPart[] = buildAttachmentContentParts(chatState.attachments);
  const textContent = trimmed || "Describe the attached file.";
  parts.push({ type: "input_text", text: textContent });

  return [{ type: "message", role: "user", content: parts }];
}
