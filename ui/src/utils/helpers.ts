/**
 * Utility helper functions.
 */

/**
 * Escape HTML special characters (&, <, >, ").
 * Safe for use in both text content and attribute values.
 */
export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
