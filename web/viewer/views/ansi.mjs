/**
 * ANSI View - Terminal output renderer with ANSI escape code support
 */

import { BaseView, escapeHtml } from "./base_view.mjs";

class AnsiView extends BaseView {
  static id = "ansi";
  static displayName = "Terminal";
  static priority = 80;

  static detect(content) {
    const trimmed = content.trim();
    // Detect actual escape bytes
    if (/\x1b\[[\d;]*m/.test(trimmed)) return 100;
    // Detect literal escape sequences (text like \033[ or \x1b[)
    if (/\\033\[[\d;]*m/.test(trimmed) || /\\x1b\[[\d;]*m/i.test(trimmed)) return 100;
    return 0;
  }

  static normalizeEscapes(content) {
    // Convert literal escape sequences to actual escape character
    return content
      .replace(/\\033/g, "\x1b")
      .replace(/\\x1b/gi, "\x1b");
  }

  static render(content, theme) {
    const ansiColors = {
      "30": "#000", "31": "#e74c3c", "32": "#2ecc71", "33": "#f39c12",
      "34": "#3498db", "35": "#9b59b6", "36": "#1abc9c", "37": "#ecf0f1",
      "90": "#7f8c8d", "91": "#ff6b6b", "92": "#69db7c", "93": "#ffd43b",
      "94": "#74c0fc", "95": "#da77f2", "96": "#63e6be", "97": "#fff",
    };
    const bgColors = {
      "40": "#000", "41": "#e74c3c", "42": "#2ecc71", "43": "#f39c12",
      "44": "#3498db", "45": "#9b59b6", "46": "#1abc9c", "47": "#ecf0f1",
    };
    
    // Normalize literal escapes to actual escape bytes first
    const normalized = this.normalizeEscapes(content);
    
    // Process ANSI codes - parse before escaping HTML
    let result = '';
    let lastIndex = 0;
    const regex = /\x1b\[([\d;]*)m/g;
    let match;
    
    while ((match = regex.exec(normalized)) !== null) {
      // Add escaped text before this match
      result += escapeHtml(normalized.slice(lastIndex, match.index));
      
      const codes = match[1];
      if (!codes || codes === "0") {
        result += "</span>";
      } else {
        const parts = codes.split(";");
        let style = "";
        for (const code of parts) {
          if (ansiColors[code]) style += `color:${ansiColors[code]};`;
          if (bgColors[code]) style += `background:${bgColors[code]};`;
          if (code === "1") style += "font-weight:bold;";
          if (code === "3") style += "font-style:italic;";
          if (code === "4") style += "text-decoration:underline;";
        }
        if (style) result += `<span style="${style}">`;
      }
      lastIndex = regex.lastIndex;
    }
    // Add remaining text
    result += escapeHtml(normalized.slice(lastIndex));
    
    return `<pre class="ansi-output">${result}</pre>`;
  }

  static getStyles(theme) {
    return `
      .ansi-output {
        background: #1e1e1e;
        padding: 12px;
        border-radius: 6px;
        font-family: "Fira Code", Consolas, Monaco, monospace;
        font-size: 13px;
        line-height: 1.4;
      }
    `;
  }
}

export default AnsiView;
