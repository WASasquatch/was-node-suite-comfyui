/**
 * CSS View - CSS code renderer with syntax highlighting
 */

import { BaseView, escapeHtml } from "./base_view.mjs";

class CssView extends BaseView {
  static id = "css";
  static displayName = "CSS";
  static priority = 45;

  static detect(content) {
    const trimmed = content.trim();
    let score = 0;

    const cssPatterns = [
      [/[.#][\w-]+\s*\{/, 2],
      [/:\s*[\w-]+\s*;/, 2],
      [/@media\s/, 3],
      [/@import\s/, 2],
      [/@keyframes\s/, 3],
      [/^\s*[\w-]+:\s*[^;]+;/m, 1],
    ];
    for (const [pattern, weight] of cssPatterns) {
      if (pattern.test(trimmed)) score += weight;
    }

    return score;
  }

  static render(content, theme) {
    return `<pre><code class="language-css">${escapeHtml(content)}</code></pre>`;
  }

  static getStyles(theme) {
    return "";
  }

  static needsPrism() {
    return true;
  }
}

export default CssView;
