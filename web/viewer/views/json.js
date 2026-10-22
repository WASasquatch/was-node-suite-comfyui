/**
 * JSON View - JSON tree viewer with collapsible nodes
 */

import { BaseView, escapeHtml } from "./base_view.js";

class JsonView extends BaseView {
  static id = "json";
  static displayName = "JSON";
  static priority = 90;

  static detect(content) {
    const trimmed = content.trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      try {
        JSON.parse(trimmed);
        return 100;
      } catch {}
    }
    return 0;
  }

  static renderTree(data, depth = 0) {
    const indent = "  ".repeat(depth);
    const childIndent = "  ".repeat(depth + 1);
    
    if (data === null) return `<span class="json-null">null</span>`;
    if (typeof data === "boolean") return `<span class="json-boolean">${data}</span>`;
    if (typeof data === "number") return `<span class="json-number">${data}</span>`;
    if (typeof data === "string") return `<span class="json-string">"${escapeHtml(data)}"</span>`;
    
    if (Array.isArray(data)) {
      if (data.length === 0) return `<span class="json-bracket">[]</span>`;
      const id = `json-${Math.random().toString(36).slice(2, 9)}`;
      const items = data.map((item, i) => {
        const comma = i < data.length - 1 ? "," : "";
        return `${childIndent}${this.renderTree(item, depth + 1)}${comma}`;
      }).join("\n");
      return `<span class="json-node"><span class="json-toggle" data-json-toggle="${id}"><span class="json-arrow">▼</span><span class="json-bracket">[</span><span class="json-collapsed-preview">${data.length} items</span></span><span id="${id}" class="json-collapsible">
${items}
${indent}<span class="json-bracket">]</span></span><span class="json-collapsed-bracket">]</span></span>`;
    }
    
    if (typeof data === "object") {
      const keys = Object.keys(data);
      if (keys.length === 0) return `<span class="json-bracket">{}</span>`;
      const id = `json-${Math.random().toString(36).slice(2, 9)}`;
      const items = keys.map((key, i) => {
        const comma = i < keys.length - 1 ? "," : "";
        return `${childIndent}<span class="json-key">"${escapeHtml(key)}"</span>: ${this.renderTree(data[key], depth + 1)}${comma}`;
      }).join("\n");
      return `<span class="json-node"><span class="json-toggle" data-json-toggle="${id}"><span class="json-arrow">▼</span><span class="json-bracket">{</span><span class="json-collapsed-preview">${keys.length} keys</span></span><span id="${id}" class="json-collapsible">
${items}
${indent}<span class="json-bracket">}</span></span><span class="json-collapsed-bracket">}</span></span>`;
    }
    
    return escapeHtml(String(data));
  }

  static render(content, theme) {
    try {
      const parsed = JSON.parse(content);
      return `<div class="json-tree"><pre>${this.renderTree(parsed)}</pre></div>`;
    } catch {
      return `<pre><code class="language-javascript">${escapeHtml(content)}</code></pre>`;
    }
  }

  static getStyles(theme) {
    return `
      .json-tree pre {
        margin: 0;
        white-space: pre-wrap;
        font-size: 13px;
        line-height: 1.4;
      }
      .json-key { color: #9cdcfe; }
      .json-string { color: #ce9178; }
      .json-number { color: #b5cea8; }
      .json-boolean { color: #569cd6; }
      .json-null { color: #569cd6; }
      .json-bracket { color: ${theme.fg}; }
      .json-node { display: inline; }
      .json-toggle {
        cursor: pointer;
        user-select: none;
      }
      .json-toggle:hover { opacity: 0.7; }
      .json-arrow {
        display: inline-block;
        width: 1em;
        font-size: 10px;
        transition: transform 0.15s;
        color: #888;
      }
      .json-collapsed-preview {
        color: #6a9955;
        font-size: 11px;
        font-style: italic;
        display: none;
      }
      .json-collapsed-bracket {
        display: none;
      }
      .json-collapsible.collapsed {
        display: none;
      }
      .json-collapsible.collapsed + .json-collapsed-bracket {
        display: inline;
      }
      .json-node:has(> .json-collapsible.collapsed) .json-collapsed-preview {
        display: inline;
        margin-left: 4px;
      }
      .json-node:has(> .json-collapsible.collapsed) .json-arrow {
        transform: rotate(-90deg);
      }
    `;
  }

  static getScripts() {
    return `
      <script>
        document.addEventListener('click', function (event) {
          var toggle = event.target && event.target.closest ? event.target.closest('[data-json-toggle]') : null;
          if (!toggle) return;
          var el = document.getElementById(toggle.getAttribute('data-json-toggle'));
          if (el) el.classList.toggle('collapsed');
        });
      <\/script>
    `;
  }
}

export default JsonView;
