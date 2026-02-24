import { getChatDom } from "./dom.ts";
import { hooks, notifications, upload } from "./deps.ts";
import { chatState, type ChatAttachment } from "./state.ts";
import { openBlobPicker } from "./blob-picker.ts";
import type { FileObject, InputContentItem, InputContentPart } from "../../types.ts";

interface ChatUploadBeforePayload {
  filename: string;
  mimeType: string | null;
  size: number;
  purpose: string;
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

  container.innerHTML = "";
  const inner = document.createElement("div");
  inner.className = "attachment-list-inner";

  for (const attachment of chatState.attachments) {
    const isImage = attachment.mimeType?.startsWith("image/") ?? false;

    if (isImage) {
      const thumb = document.createElement("div");
      thumb.className = "attachment-thumb";

      const img = document.createElement("img");
      img.src = `/v1/files/${encodeURIComponent(attachment.file.id)}/content`;
      img.alt = attachment.file.filename;
      img.draggable = false;
      thumb.appendChild(img);

      const removeBtn = document.createElement("button");
      removeBtn.className = "attachment-thumb-remove";
      removeBtn.dataset["removeAttachment"] = attachment.file.id;
      removeBtn.title = "Remove";
      removeBtn.textContent = "\u00d7";
      thumb.appendChild(removeBtn);

      inner.appendChild(thumb);
    } else {
      const pill = document.createElement("span");
      pill.className = "attachment-pill";
      pill.title = attachment.file.filename;

      const nameSpan = document.createElement("span");
      nameSpan.className = "attachment-name";
      nameSpan.textContent = attachment.file.filename;
      pill.appendChild(nameSpan);

      const metaSpan = document.createElement("span");
      metaSpan.className = "attachment-meta";
      metaSpan.textContent = `${attachment.file.bytes.toLocaleString()} bytes`;
      pill.appendChild(metaSpan);

      const removeBtn = document.createElement("button");
      removeBtn.className = "attachment-remove";
      removeBtn.dataset["removeAttachment"] = attachment.file.id;
      removeBtn.title = "Remove attachment";
      removeBtn.textContent = "\u00d7";
      pill.appendChild(removeBtn);

      inner.appendChild(pill);
    }
  }

  container.appendChild(inner);
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

export async function uploadFiles(files: FileList): Promise<void> {
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

  dom.welcomeLibrary.addEventListener("click", () => {
    if (chatState.isUploadingAttachments || chatState.isGenerating) return;
    void openBlobPicker().then(addBlobAttachments);
  });

  dom.inputLibrary.addEventListener("click", () => {
    if (chatState.isUploadingAttachments || chatState.isGenerating) return;
    void openBlobPicker().then(addBlobAttachments);
  });

  dom.fileInput.addEventListener("change", () => {
    if (!dom.fileInput.files) return;
    void uploadFiles(dom.fileInput.files);
  });

  dom.welcomeAttachmentList.addEventListener("click", onAttachmentClick);
  dom.inputAttachmentList.addEventListener("click", onAttachmentClick);

  renderAttachmentLists();
}

/** Convert selected FileObjects from the blob picker into chat attachments. */
function addBlobAttachments(files: FileObject[]): void {
  if (files.length === 0) return;

  for (const file of files) {
    // Skip if already attached.
    if (chatState.attachments.some((a) => a.file.id === file.id)) continue;

    chatState.attachments.push({
      file: {
        id: file.id,
        filename: file.filename,
        bytes: file.bytes,
        createdAt: file.created_at,
        purpose: file.purpose,
      },
      mimeType: file.mime_type ?? null,
    });
  }

  notifications.info(`Attached ${files.length} file${files.length === 1 ? "" : "s"}`);
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
 * Build structured input for the /v1/chat/generate endpoint.
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
