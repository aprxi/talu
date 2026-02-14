import { beforeEach, describe, expect, test } from "bun:test";
import {
  buildAttachmentReferencePrefix,
  composeUserInputWithAttachments,
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

  test("buildAttachmentReferencePrefix includes id, name, size, and mime when present", () => {
    const prefix = buildAttachmentReferencePrefix([
      makeAttachment("file_1", "report.pdf", 2048, "application/pdf"),
      makeAttachment("file_2", "raw.bin", 99, null),
    ]);

    expect(prefix).toContain("<attachments>");
    expect(prefix).toContain('<file_ref id="file_1" name="report.pdf" bytes="2048" mime_type="application/pdf" />');
    expect(prefix).toContain('<file_ref id="file_2" name="raw.bin" bytes="99" />');
    expect(prefix).toContain("</attachments>");
  });

  test("buildAttachmentReferencePrefix escapes XML attributes", () => {
    const prefix = buildAttachmentReferencePrefix([
      makeAttachment("file_1", 'a "b" <c>.txt', 12, 'text/plain; charset="utf-8"'),
    ]);

    expect(prefix).toContain('name="a &quot;b&quot; &lt;c&gt;.txt"');
    expect(prefix).toContain('mime_type="text/plain; charset=&quot;utf-8&quot;"');
  });

  test("composeUserInputWithAttachments returns raw trimmed text when no attachments", () => {
    const result = composeUserInputWithAttachments("  hello world  ");
    expect(result).toBe("hello world");
  });

  test("composeUserInputWithAttachments prefixes attachments before user text", () => {
    chatState.attachments = [makeAttachment("file_1", "manual.pdf", 12345, "application/pdf")];
    const result = composeUserInputWithAttachments("Please summarize this.");
    expect(result).toContain('<file_ref id="file_1" name="manual.pdf" bytes="12345" mime_type="application/pdf" />');
    expect(result).toContain("Please summarize this.");
  });

  test("composeUserInputWithAttachments emits instruction when text is empty", () => {
    chatState.attachments = [makeAttachment("file_1", "manual.pdf", 12345, "application/pdf")];
    const result = composeUserInputWithAttachments("   ");
    expect(result).toContain("Please use the attached files above.");
  });
});
