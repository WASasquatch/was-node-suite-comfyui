/**
 * YAML View - YAML code renderer with custom syntax highlighting
 */

import { BaseView, escapeHtml, highlightSyntax } from "./base_view.mjs";

class YamlView extends BaseView {
  static id = "yaml";
  static displayName = "YAML";
  static priority = 40;

  static highlightRules = [
    // Comments (highest priority)
    { name: "comment", pattern: /#.*$/m, className: "yaml-comment" },
    // Document separators
    { name: "separator", pattern: /^---\s*$/m, className: "yaml-separator" },
    { name: "separator-end", pattern: /^\.\.\.\s*$/m, className: "yaml-separator" },
    // List item markers
    { name: "list-marker", pattern: /^(\s*)-(?=\s|$)/m, className: "yaml-list-marker" },
    // Keys (word before colon)
    { name: "key", pattern: /^(\s*)([^\s#:][^:#]*?)(?=:\s|:$)/m, className: "yaml-key" },
    // Quoted strings
    { name: "string-double", pattern: /"(?:[^"\\]|\\.)*"/g, className: "yaml-string" },
    { name: "string-single", pattern: /'(?:[^'\\]|\\.)*'/g, className: "yaml-string" },
    // Booleans
    { name: "boolean", pattern: /\b(true|false|yes|no|on|off)\b/gi, className: "yaml-boolean" },
    // Null
    { name: "null", pattern: /\b(null|~)\b/gi, className: "yaml-null" },
    // Numbers
    { name: "number", pattern: /\b-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\b/g, className: "yaml-number" },
    // Anchors and aliases
    { name: "anchor", pattern: /&[\w-]+/g, className: "yaml-anchor" },
    { name: "alias", pattern: /\*[\w-]+/g, className: "yaml-alias" },
    // Tags
    { name: "tag", pattern: /![\w!\/.-]*/g, className: "yaml-tag" },
  ];

  static detect(content) {
    const trimmed = content.trim();
    let score = 0;

    const yamlPatterns = [
      [/^[\w-]+:\s*.+$/m, 2],
      [/^[\w-]+:\s*$/m, 2],
      [/^\s+-\s+.+$/m, 1],
      [/^---\s*$/m, 3],
    ];
    for (const [pattern, weight] of yamlPatterns) {
      if (pattern.test(trimmed)) score += weight;
    }
    if (trimmed.includes(": ") && !trimmed.includes("{") && !trimmed.includes("<")) {
      score += 1;
    }

    return score;
  }

  static render(content, theme) {
    const highlighted = highlightSyntax(content, this.highlightRules);
    return `<pre class="yaml-code"><code>${highlighted}</code></pre>`;
  }

  static getStyles(theme) {
    return `
      .yaml-code {
        margin: 0;
        font-size: 13px;
        line-height: 1.5;
      }
      .yaml-comment { color: #6a9955; font-style: italic; }
      .yaml-separator { color: #569cd6; }
      .yaml-list-marker { color: #c586c0; font-weight: bold; }
      .yaml-key { color: #9cdcfe; }
      .yaml-string { color: #ce9178; }
      .yaml-boolean { color: #569cd6; }
      .yaml-null { color: #569cd6; font-style: italic; }
      .yaml-number { color: #b5cea8; }
      .yaml-anchor { color: #dcdcaa; }
      .yaml-alias { color: #dcdcaa; font-style: italic; }
      .yaml-tag { color: #4ec9b0; }
    `;
  }

  static needsPrism() {
    return false;
  }
}

export default YamlView;
