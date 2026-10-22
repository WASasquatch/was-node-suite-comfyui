/**
 * Markdown View - Markdown renderer with Mermaid diagrams and KaTeX math support
 */

import { BaseView, escapeHtml, loadScriptText } from "./base_view.js";
import { triggerViewsRefresh } from "./view_loader.js";

const MERMAID_FILE = "mermaid.min.txt";
const KATEX_FILES = ["katex.min.txt", "katex-auto-render.min.txt"];
const KATEX_CSS_FILE = "katex-with-fonts.min.css.txt";

/** Schemes a link may name. An address with no scheme is relative to the frame. */
const LINK_SCHEMES = new Set(["http", "https", "mailto"]);

/** Schemes a picture may name. A `data:` address is a picture only. */
const IMAGE_SCHEMES = new Set(["http", "https", "data", "blob"]);

/**
 * Text written inside a double-quoted attribute value.
 * @param {string} value
 * @returns {string} `value` with `&`, `"`, `'`, `<` and `>` as entities, and the characters
 *   later markdown passes match (`*`, `_`, `~`, `|`, `[`, `]`, `(`, `)`) as numeric entities.
 */
function attribute(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/[*_~|[\]()]/g, (c) => `&#${c.charCodeAt(0)};`);
}

/**
 * An address from markdown text, or null when its scheme is not one offered.
 * @param {string} value - The text between the parentheses.
 * @param {Set<string>} schemes - Schemes accepted.
 * @returns {string|null} The trimmed address, or null.
 */
function address(value, schemes) {
  const text = String(value).trim();
  // Control characters and spaces are dropped before the scheme is read.
  const bare = text.replace(/[\x00-\x20\x7f]/g, "").toLowerCase();
  const scheme = /^([a-z][a-z0-9+.-]*):/.exec(bare);
  if (!scheme) return text;
  if (!schemes.has(scheme[1])) return null;
  if (scheme[1] === "data" && !bare.startsWith("data:image/")) return null;
  return text;
}

class MarkdownView extends BaseView {
  static id = "markdown";
  static displayName = "Markdown";
  static priority = 70;

  static scripts = {
    mermaid: null,
    katexCore: null,
    katexAutoRender: null,
    katexCss: null,
  };

  static detect(content) {
    const trimmed = content.trim();
    let score = 0;

    const mdPatterns = [
      [/^#{1,6}\s+.+/m, 3],
      [/^\s*[-*+]\s+.+/m, 1],
      [/^\s*\d+\.\s+.+/m, 1],
      [/\[[^\]]+\]\([^)]+\)/, 2],
      [/^\s*```/m, 3],
      [/^\s*>/m, 1],
      [/\*\*[^*]+\*\*/, 1],
      [/~~[^~]+~~/, 1],
      [/^\s*\|.+\|.+\|/m, 2],
      [/!\[[^\]]*\]\([^)]+\)/, 2],
    ];
    for (const [pattern, weight] of mdPatterns) {
      if (pattern.test(trimmed)) score += weight;
    }

    return score;
  }

  static async loadScripts(basePath) {
    const promises = [];

    if (!this.scripts.mermaid) {
      promises.push(
        loadScriptText(`${basePath}/views/markdown_scripts/${MERMAID_FILE}`)
          .then(script => {
            if (script && script.trim().startsWith('<')) {
              console.warn('[Markdown] Mermaid script appears to be HTML, not JS - likely 404');
              this.scripts.mermaid = null;
            } else {
              this.scripts.mermaid = script;
              if (script) triggerViewsRefresh();
            }
          })
      );
    }

    if (!this.scripts.katexCore || !this.scripts.katexAutoRender || !this.scripts.katexCss) {
      promises.push(
        Promise.all([
          loadScriptText(`${basePath}/views/markdown_scripts/${KATEX_FILES[0]}`),
          loadScriptText(`${basePath}/views/markdown_scripts/${KATEX_FILES[1]}`),
          loadScriptText(`${basePath}/views/markdown_scripts/${KATEX_CSS_FILE}`),
        ]).then(([katexCore, autoRender, katexCss]) => {
          // Validate scripts aren't HTML error pages
          this.scripts.katexCore = (katexCore && !katexCore.trim().startsWith('<')) ? katexCore : null;
          this.scripts.katexAutoRender = (autoRender && !autoRender.trim().startsWith('<')) ? autoRender : null;
          this.scripts.katexCss = katexCss; // CSS can start with < in comments
          if (this.scripts.katexCore && this.scripts.katexCss) triggerViewsRefresh();
        })
      );
    }

    await Promise.all(promises);
    return true;
  }

  static isReady() {
    return true;
  }

  static parseNestedList(match, listTag, itemTag) {
    const regex = new RegExp(`<${itemTag}[^>]*data-level="(\\d+)"[^>]*(?:class="([^"]*)")?[^>]*>(.*?)<\\/${itemTag}>`, 'g');
    const items = [];
    let m;
    while ((m = regex.exec(match)) !== null) {
      items.push({ level: parseInt(m[1]), className: m[2] || '', content: m[3] });
    }
    
    if (items.length === 0) return match;
    
    let result = '';
    let stack = [];
    
    for (const item of items) {
      while (stack.length > 0 && stack[stack.length - 1] > item.level) {
        result += `</li></${listTag}>`;
        stack.pop();
      }
      
      if (stack.length === 0 || stack[stack.length - 1] < item.level) {
        result += `<${listTag}>`;
        stack.push(item.level);
      } else if (stack[stack.length - 1] === item.level && result.length > 0) {
        result += '</li>';
      }
      
      const classAttr = item.className ? ` class="${item.className}"` : '';
      result += `<li${classAttr}>${item.content}`;
    }
    
    while (stack.length > 0) {
      result += `</li></${listTag}>`;
      stack.pop();
    }
    
    return result;
  }

  static parseMarkdown(md) {
    let html = md;

    // STEP 1: Extract and protect code blocks FIRST (before any other processing)
    // This prevents LaTeX, emphasis, etc. from being parsed inside code
    const codeBlocks = [];
    const PLACEHOLDER = '\x00CODE_BLOCK_';
    
    // Fenced code blocks (mermaid) - allow leading whitespace for indented blocks
    html = html.replace(/^[ \t]*```(?:mermaid|flow|flex)(?:\[([^\]]*)\])?\n([\s\S]*?)^[ \t]*```$/gm, (_, options, code) => {
      const centered = options && options.toLowerCase().includes('center');
      const style = centered ? 'text-align:center;' : 'text-align:left;';
      const rendered = `<div class="mermaid" style="${style}">${escapeHtml(code.trim())}</div>`;
      codeBlocks.push(rendered);
      return `${PLACEHOLDER}${codeBlocks.length - 1}\x00`;
    });

    // Fenced code blocks (regular) - allow leading whitespace for indented blocks
    html = html.replace(/^[ \t]*```(\w*)\n([\s\S]*?)^[ \t]*```$/gm, (_, lang, code) => {
      const rendered = `<pre><code class="language-${lang || "text"}">${escapeHtml(code.trim())}</code></pre>`;
      codeBlocks.push(rendered);
      return `${PLACEHOLDER}${codeBlocks.length - 1}\x00`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, (_, code) => {
      const rendered = `<code>${escapeHtml(code)}</code>`;
      codeBlocks.push(rendered);
      return `${PLACEHOLDER}${codeBlocks.length - 1}\x00`;
    });

    // STEP 2: Now process LaTeX (code is already protected)
    html = html.replace(/\$\$\s*\\\(([^]*?)\\\)\s*\$\$/g, '$$$$$1$$$$');
    html = html.replace(/\$\$\s*\\\[([^]*?)\\\]\s*\$\$/g, '$$$$$1$$$$');
    html = html.replace(/([^\n$\x00])\$\$([^$\n]+)\$\$([^\n$])/g, '$1$$$2$$$3');

    html = html.replace(/^(.+)\n=+\s*$/gm, "<h1>$1</h1>");
    html = html.replace(/^(.+)\n-+\s*$/gm, "<h2>$1</h2>");

    const slugify = (text) => text.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-');
    html = html.replace(/^######\s*(.+)$/gm, (_, t) => `<h6 id="${slugify(t)}">${t}</h6>`);
    html = html.replace(/^#####\s*(.+)$/gm, (_, t) => `<h5 id="${slugify(t)}">${t}</h5>`);
    html = html.replace(/^####\s*(.+)$/gm, (_, t) => `<h4 id="${slugify(t)}">${t}</h4>`);
    html = html.replace(/^###\s*(.+)$/gm, (_, t) => `<h3 id="${slugify(t)}">${t}</h3>`);
    html = html.replace(/^##\s*(.+)$/gm, (_, t) => `<h2 id="${slugify(t)}">${t}</h2>`);
    html = html.replace(/^#\s*(.+)$/gm, (_, t) => `<h1 id="${slugify(t)}">${t}</h1>`);

    html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, src) => {
      const source = address(src, IMAGE_SCHEMES);
      if (source === null) return attribute(alt);
      return `<img src="${attribute(source)}" alt="${attribute(alt)}" style="max-width:100%;">`;
    });
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, href) => {
      const link = address(href, LINK_SCHEMES);
      if (link === null) return text;
      const target = link.startsWith('#') ? '' : ' target="_blank" rel="noopener noreferrer"';
      return `<a href="${attribute(link)}"${target}>${text}</a>`;
    });

    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/__(.+?)__/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    html = html.replace(/(?<![a-zA-Z0-9\/_])_([^_\s][^_]*)_(?![a-zA-Z0-9\/_])/g, "<em>$1</em>");
    html = html.replace(/~~(.+?)~~/g, "<del>$1</del>");

    html = html.replace(/^>\s+(.*)$/gm, "<blockquote>$1</blockquote>");

    html = html.replace(/^(\s*)- \[x\]\s+(.*)$/gim, (_, indent, text) => {
      const level = Math.floor(indent.length / 2);
      return `<uli data-level="${level}" class="task-item checked"><input type="checkbox" checked disabled> ${text}</uli>`;
    });
    html = html.replace(/^(\s*)- \[ \]\s+(.*)$/gm, (_, indent, text) => {
      const level = Math.floor(indent.length / 2);
      return `<uli data-level="${level}" class="task-item"><input type="checkbox" disabled> ${text}</uli>`;
    });

    html = html.replace(/^(\s*)\d+\.\s+(.*)$/gm, (_, indent, text) => {
      const level = Math.floor(indent.length / 2);
      return `<oli data-level="${level}">${text}</oli>`;
    });

    html = html.replace(/^(\s*)[-*+]\s+(.*)$/gm, (_, indent, text) => {
      const level = Math.floor(indent.length / 2);
      return `<uli data-level="${level}">${text}</uli>`;
    });

    html = html.replace(/(<oli[^>]*>.*?<\/oli>\s*)+/g, (match) => {
      return this.parseNestedList(match, 'ol', 'oli');
    });

    html = html.replace(/(<uli[^>]*>.*?<\/uli>\s*)+/g, (match) => {
      return this.parseNestedList(match, 'ul', 'uli');
    });

    html = html.replace(/^---$/gm, "<hr>");
    html = html.replace(/^\*\*\*$/gm, "<hr>");

    html = html.replace(/^(\|.+\|)\n(\|[-:\s|]+\|)\n((?:\|.+\|\n?)+)/gm, (match, headerRow, separatorRow, bodyRows) => {
      const parseRow = (row, cellTag) => {
        const cells = row.split("|").slice(1, -1);
        return `<tr>${cells.map(c => `<${cellTag}>${c.trim()}</${cellTag}>`).join("")}</tr>`;
      };
      const alignments = separatorRow.split("|").slice(1, -1).map(cell => {
        const trimmed = cell.trim();
        if (trimmed.startsWith(":") && trimmed.endsWith(":")) return "center";
        if (trimmed.endsWith(":")) return "right";
        return "left";
      });
      const headerCells = headerRow.split("|").slice(1, -1);
      const theadRow = `<tr>${headerCells.map((c, i) => `<th style="text-align:${alignments[i] || "left"}">${c.trim()}</th>`).join("")}</tr>`;
      const bodyRowsArray = bodyRows.trim().split("\n");
      const tbodyRows = bodyRowsArray.map(row => {
        const cells = row.split("|").slice(1, -1);
        return `<tr>${cells.map((c, i) => `<td style="text-align:${alignments[i] || "left"}">${c.trim()}</td>`).join("")}</tr>`;
      }).join("");
      return `<table><thead>${theadRow}</thead><tbody>${tbodyRows}</tbody></table>`;
    });

    html = html.replace(/\n\n/g, "</p><p>");
    html = "<p>" + html + "</p>";

    // STEP 3: Restore protected code blocks BEFORE paragraph cleanup
    // (so <pre> tags can be properly unwrapped from <p> tags)
    html = html.replace(/\x00CODE_BLOCK_(\d+)\x00/g, (_, idx) => codeBlocks[parseInt(idx)] || '');

    // STEP 4: Clean up paragraph tags around block elements
    html = html.replace(/<p>\s*<(h[1-6]|ul|ol|pre|blockquote|hr|table|div)/g, "<$1");
    html = html.replace(/<\/(h[1-6]|ul|ol|pre|blockquote|table|div)>\s*<\/p>/g, "</$1>");
    html = html.replace(/<p>\s*<\/p>/g, "");

    return html;
  }

  static render(content, theme) {
    return this.parseMarkdown(content);
  }

  static getStyles(theme) {
    return `
      .mermaid {
        background: transparent;
        text-align: center;
        padding: 16px;
      }
      .math-block {
        text-align: center;
        margin: 16px 0;
        overflow-x: auto;
      }
      .task-item {
        list-style: none;
        margin-left: -20px;
      }
      .task-item input[type="checkbox"] {
        margin-right: 8px;
        accent-color: ${theme.accent};
      }
      .task-item.checked {
        color: ${theme.fg};
      }
    `;
  }

  static hasMermaid(content) {
    return /```(?:mermaid|flow|flex)/.test(content) || content.includes("<div class=\"mermaid\">");
  }

  static hasLatex(content) {
    return content.includes("$$") || content.includes("\\(") || content.includes("\\[") || /\$[^$\n]+\$/.test(content);
  }

  static getScripts(content) {
    // Legacy method - returns empty, use getScriptData instead for postMessage injection
    return "";
  }

  static getScriptData(content) {
    const scripts = [];

    const hasMermaid = this.hasMermaid(content);
    // Only inject mermaid if it's valid JS (not an HTML error page)
    if (hasMermaid && this.scripts.mermaid && !this.scripts.mermaid.trim().startsWith('<')) {
      scripts.push({
        code: this.scripts.mermaid,
        // The frame has already loaded by the time this is injected, so the diagrams are
        // rendered by hand rather than by startOnLoad.
        init: `mermaid.initialize({ startOnLoad: false, theme: 'dark' });
        (function () {
          const blocks = document.querySelectorAll('.mermaid');
          if (!blocks.length) return;
          if (typeof mermaid.run === 'function') {
            mermaid.run({ nodes: blocks });
          } else if (typeof mermaid.init === 'function') {
            mermaid.init(undefined, blocks);
          }
        })();`
      });
    }

    const hasLatex = this.hasLatex(content);
    // Only inject KaTeX if it's valid JS (not an HTML error page)
    const katexCoreValid = this.scripts.katexCore && !this.scripts.katexCore.trim().startsWith('<');
    const katexAutoRenderValid = this.scripts.katexAutoRender && !this.scripts.katexAutoRender.trim().startsWith('<');
    if (hasLatex && katexCoreValid && katexAutoRenderValid) {
      scripts.push({
        code: this.scripts.katexCore
      });
      scripts.push({
        code: this.scripts.katexAutoRender,
        init: `if (typeof renderMathInElement === 'function') {
          renderMathInElement(document.body, {
            delimiters: [
              {left: '$$', right: '$$', display: true},
              {left: '$', right: '$', display: false},
              {left: '\\\\(', right: '\\\\)', display: false},
              {left: '\\\\[', right: '\\\\]', display: true}
            ],
            ignoredTags: ['script', 'noscript', 'style', 'textarea', 'option', 'code', 'pre'],
            throwOnError: false
          });
        }`
      });
    }

    return scripts;
  }

  static getKatexCss() {
    return this.scripts.katexCss || "";
  }

  static needsPrism() {
    return true;
  }
}

export default MarkdownView;
