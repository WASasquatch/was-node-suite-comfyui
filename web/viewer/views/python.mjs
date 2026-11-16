/**
 * Python View - Python code renderer with syntax highlighting
 */

import { BaseView, escapeHtml } from "./base_view.mjs";

class PythonView extends BaseView {
  static id = "python";
  static displayName = "Python";
  static priority = 60;

  static detect(content) {
    const trimmed = content.trim();
    let score = 0;

    const pyPatterns = [
      [/^import\s+\w+/m, 3],
      [/^from\s+\w+\s+import/m, 3],
      [/^def\s+\w+\s*\(/m, 3],
      [/^class\s+\w+/m, 3],
      [/self\.\w+/, 2],
      [/__init__/, 2],
      [/__name__/, 2],
      [/:\s*$/m, 1],
      [/^\s+return\s+/m, 1],
      [/^\s+if\s+.+:/m, 1],
      [/^\s+for\s+.+:/m, 1],
      [/^\s+elif\s+/m, 2],
      [/True|False|None/, 1],
    ];
    for (const [pattern, weight] of pyPatterns) {
      if (pattern.test(trimmed)) score += weight;
    }

    return score;
  }

  static render(content, theme) {
    return `<pre><code class="language-python">${escapeHtml(content)}</code></pre>`;
  }

  static getStyles(theme) {
    return "";
  }

  static needsPrism() {
    return true;
  }
}

export default PythonView;
