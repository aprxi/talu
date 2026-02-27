import { beforeEach, describe, expect, test } from "bun:test";
import {
  composeUserInput,
  hasAttachments,
} from "../../../src/plugins/chat/attachments.ts";
import { chatState, type ChatAttachment } from "../../../src/plugins/chat/state.ts";

function makeAttachment(
  id: string,
  filename: string,
  bytes: number,
  mimeType: string | null,
): ChatAttachment {
  return {
    file: {
      id,
      bytes,
      filename,
      createdAt: 123,
      purpose: "assistants",
    },
    mimeType,
  };
}

beforeEach(() => {
  chatState.attachments = [];
});

describe("chat attachments helpers", () => {
  test("hasAttachments reflects attachment state", () => {
    expect(hasAttachments()).toBe(false);
    chatState.attachments = [makeAttachment("file_1", "a.txt", 10, "text/plain")];
    expect(hasAttachments()).toBe(true);
  });

  test("composeUserInput returns raw trimmed text when no attachments", () => {
    const result = composeUserInput("  hello world  ");
    expect(result).toBe("hello world");
  });

  test("composeUserInput returns structured input with file parts when attachments present", () => {
    chatState.attachments = [makeAttachment("file_1", "manual.pdf", 12345, "application/pdf")];
    const result = composeUserInput("Please summarize this.");
    expect(Array.isArray(result)).toBe(true);
    const items = result as any[];
    expect(items.length).toBe(1);
    expect(items[0].type).toBe("message");
    expect(items[0].role).toBe("user");
    const parts = items[0].content;
    expect(parts.some((p: any) => p.type === "input_file" && p.file_url === "file_1")).toBe(true);
    expect(parts.some((p: any) => p.type === "input_text" && p.text === "Please summarize this.")).toBe(true);
  });

  test("composeUserInput uses fallback text when input is empty", () => {
    chatState.attachments = [makeAttachment("file_1", "manual.pdf", 12345, "application/pdf")];
    const result = composeUserInput("   ");
    expect(Array.isArray(result)).toBe(true);
    const items = result as any[];
    const parts = items[0].content;
    expect(parts.some((p: any) => p.type === "input_text" && p.text === "Describe the attached file.")).toBe(true);
  });

  test("composeUserInput creates input_image parts for image attachments", () => {
    chatState.attachments = [makeAttachment("img_1", "photo.png", 5000, "image/png")];
    const result = composeUserInput("Describe this image.");
    expect(Array.isArray(result)).toBe(true);
    const parts = (result as any[])[0].content;
    expect(parts.some((p: any) => p.type === "input_image" && p.image_url === "img_1")).toBe(true);
  });
});
