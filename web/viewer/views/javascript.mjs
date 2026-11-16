/**
 * JavaScript View - JavaScript code renderer with syntax highlighting
 */

import { BaseView, escapeHtml } from "./base_view.mjs";

class JavaScriptView extends BaseView {
  static id = "javascript";
  static displayName = "JavaScript";
  static priority = 55;

  static detect(content) {
    const trimmed = content.trim();
    let score = 0;

    const jsPatterns = [
      [/^function\s+\w+\s*\(/m, 3],
      [/^const\s+\w+\s*=/m, 3],
      [/^let\s+\w+\s*=/m, 3],
      [/^var\s+\w+\s*=/m, 2],
      [/=>\s*\{/, 2],
      [/^import\s+\{/m, 3],
      [/^export\s+(default\s+)?/m, 3],
      [/document\.\w+/, 2],
      [/console\.\w+/, 2],
      [/\.addEventListener\(/, 2],
      [/===|!==/, 1],
      [/\?\.\w+/, 2],
    ];
    for (const [pattern, weight] of jsPatterns) {
      if (pattern.test(trimmed)) score += weight;
    }

    return score;
  }

  static render(content, theme) {
    return `<pre><code class="language-javascript">${escapeHtml(content)}</code></pre>`;
  }

  static getStyles(theme) {
    return "";
  }

  static needsPrism() {
    return true;
  }
}

export default JavaScriptView;
