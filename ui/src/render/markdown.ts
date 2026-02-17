import { marked } from "marked";
import DOMPurify from "dompurify";
import { COPY_ICON as CODE_COPY_ICON, CHECK_ICON as CODE_CHECK_ICON } from "../icons.ts";
import { escapeHtml } from "../utils/helpers.ts";

export { CODE_COPY_ICON, CODE_CHECK_ICON };

marked.use({
  renderer: {
    code(token) {
      const lang = token.lang || "code";
      const escaped = escapeHtml(token.text);
      // data-code attribute holds the raw code for copy button
      return `<div class="code-block" data-code="${escapeHtml(token.text)}">
<div class="code-header">
<span class="code-lang">${lang}</span>
<button class="code-copy" title="Copy code">${CODE_COPY_ICON}</button>
</div>
<pre><code class="language-${lang}">${escaped}</code></pre>
</div>`;
    },
  },
});

export function sanitizedMarkdown(text: string): string {
  const raw = marked(text, { async: false }) as string;
  const clean = DOMPurify.sanitize(raw);
  // DOMPurify's parse→serialize round-trip decodes &lt;/&gt; in attribute
  // values (they're valid in double-quoted HTML attributes).  Re-escape
  // < and > inside data-code so the attribute stays safely encoded.
  return clean.replace(
    /data-code="([^"]*)"/g,
    (_, val: string) =>
      `data-code="${val.replace(/</g, "&lt;").replace(/>/g, "&gt;")}"`,
  );
}
