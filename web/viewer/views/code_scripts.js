/**
 * Code Scripts Loader - Loads Prism.js for syntax highlighting
 * Shared across code view modules (Python, JavaScript, CSS, YAML, Markdown)
 */

import { loadScriptText } from "./base_view.js";
import { triggerViewsRefresh } from "./view_loader.js";

const PRISM_FILES = [
  "prism.min.txt",
  "prism-python.min.txt",
  "prism-javascript.min.txt",
  "prism-css.min.txt",
  "prism-markup.min.txt",
  "prism-json.min.txt",
  "prism-bash.min.txt",
];

let prismScripts = null;
let loadPromise = null;

/**
 * Load all Prism scripts
 * @param {string} basePath - Base path for scripts (web folder)
 * @returns {Promise<string>}
 */
export async function loadPrismScripts(basePath) {
  if (prismScripts) return prismScripts;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    try {
      const scripts = await Promise.all(
        PRISM_FILES.map(file => loadScriptText(`${basePath}/views/code_scripts/${file}`))
      );
      prismScripts = scripts.join("\n");
      if (prismScripts) {
        triggerViewsRefresh();
      }
      return prismScripts;
    } catch (e) {
      console.error("[WAS Viewer] Failed to load Prism scripts:", e);
      return "";
    }
  })();

  return loadPromise;
}

/**
 * Get loaded Prism scripts
 * @returns {string|null}
 */
export function getPrismScripts() {
  return prismScripts;
}

/**
 * Check if Prism scripts are loaded
 * @returns {boolean}
 */
export function isPrismReady() {
  return prismScripts !== null && prismScripts.length > 0;
}

/**
 * Get Prism theme styles
 * @param {object} theme - Theme tokens
 * @returns {string}
 */
export function getPrismTheme(theme) {
  return `
    code[class*="language-"], pre[class*="language-"] {
      color: ${theme.fg};
      text-shadow: none;
      font-family: "Fira Code", Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
      font-size: 13px;
      text-align: left;
      white-space: pre-wrap;
      word-spacing: normal;
      word-break: normal;
      word-wrap: break-word;
      line-height: 1.5;
      tab-size: 4;
      hyphens: none;
    }
    pre[class*="language-"] {
      padding: 12px;
      margin: 0;
      overflow: auto;
      border-radius: 6px;
      background: rgba(0,0,0,0.3);
    }
    :not(pre) > code[class*="language-"] {
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(0,0,0,0.2);
    }
    .token.comment, .token.prolog, .token.doctype, .token.cdata { color: #6a9955; }
    .token.punctuation { color: ${theme.fg}; }
    .token.namespace { opacity: .7; }
    .token.property, .token.tag, .token.boolean, .token.number, .token.constant, .token.symbol, .token.deleted { color: #b5cea8; }
    .token.selector, .token.attr-name, .token.string, .token.char, .token.builtin, .token.inserted { color: #ce9178; }
    .token.operator, .token.entity, .token.url, .language-css .token.string, .style .token.string { color: #d4d4d4; }
    .token.atrule, .token.attr-value, .token.keyword { color: #569cd6; }
    .token.function, .token.class-name { color: #dcdcaa; }
    .token.regex, .token.important, .token.variable { color: #d16969; }
    .token.important, .token.bold { font-weight: bold; }
    .token.italic { font-style: italic; }
  `;
}

/**
 * Get Prism script tags for iframe injection
 * @param {string} [nonce] - A nonce each tag carries, or none.
 * @returns {string}
 */
export function getPrismScriptTags(nonce = "") {
  if (!prismScripts) return "";
  const attribute = /^[0-9a-f]+$/.test(nonce) ? ` nonce="${nonce}"` : "";
  return `
    <script${attribute}>${prismScripts}<\/script>
    <script${attribute}>document.addEventListener('DOMContentLoaded', () => Prism.highlightAll());<\/script>
  `;
}
